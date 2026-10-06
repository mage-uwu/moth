# Golden Tree Snake (GTS) fork, 2026.
"""Chunked SSD scan for GTS's depth-0 trees (the bank of the mixed forest), in Triton.

A depth-0 tree is visited by every token, so a group of them sharing one clock is exactly Mamba-2's SSD with one
key/query group: for head h and channels p of that head,

    Y[t, h, p] = sum_{s < t} exp(a[s+1, h] + ... + a[t, h]) * <C[t], B[s]> * X[s, h, p]          (forward)
    Y[t, h, p] = sum_{s > t} exp(a[t, h] + ... + a[s-1, h]) * <C[t], B[s]> * X[s, h, p]          (reverse)

with per-token log-decays a <= 0 and, unlike Mamba-2, the token's own term s = t excluded. These are GTS's forward
and backward context for a depth-0 tree. GTS's training path forms the (batch, t, s, heads) weights explicitly,
quadratic in length; this runs in chunks with an (N x P) state per head carried between chunks.

In processing order (reversed for ``reverse``) both are one formula: Y[i] = sum_{j < i} exp(c[i] - c[j]) <C, B> X[j]
with the clock c the running sum of a, inclusive. The transposed scans of the backward pass run the other way with
the exclusive running sum (``excl``). No global running sum is ever formed: within a chunk every exponent needs only
the chunk's own sums, and between chunks only each chunk's total. Three kernels, all over (chunk, batch * head):

    _state_kernel   each chunk's own state sum_j exp(T - c[j]) K[j] V[j]^T and its total T
    _pass_kernel    (batch * head programs) turns the own states into the state entering each chunk, in place
    _out_kernel     O1[i] = sum_j w <Q[i], K[j]> V[j]   and/or   O2[i] = sum_j w <U[i], V[j]> K[j]

O1 is the forward output and dX; O2 is dB and dC. The forward's chunk states are reused for dC. The log-decay
gradient is d c[i] = <dY[i], Y[i]> - <X[i], dX[i]>, written by the dX pass and summed from each token onwards
(_suffix_kernel).

Every kernel also has a direction axis (the third grid dimension): ``gts_scan_bi`` runs a bidirectional context,
forward with one query and reverse with another, in the same launches. Tensors both directions share are passed with
a zero direction stride, so nothing is copied.

``gts_scan(C, B, X, a, reverse)`` and ``gts_scan_bi(C_fwd, C_bwd, B, X, a)`` are differentiable;
``gts_scan_reference`` is the quadratic form they are tested against.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the PyTorch path does not need triton
    triton = None

__all__ = ["gts_scan", "gts_scan_bi", "gts_scan_reference", "HAVE_TRITON"]
HAVE_TRITON = triton is not None


def gts_scan_reference(C, B, X, a, reverse=False, excl=False):
    """Quadratic PyTorch form. C, B: (b, l, n); X: (b, l, h, p); a: (b, l, h) per-token log-decays <= 0."""
    if reverse:
        return gts_scan_reference(C.flip(1), B.flip(1), X.flip(1), a.flip(1), False, excl).flip(1)
    length = a.shape[1]
    c = torch.cumsum(a, dim=1)
    if excl:
        c = c - a
    seg = c[:, :, None, :] - c[:, None, :, :]  # [i, j] = c[i] - c[j]
    keep = torch.ones(length, length, dtype=torch.bool, device=a.device).tril(-1)[None, :, :, None]
    w = (C @ B.transpose(1, 2)).unsqueeze(-1) * torch.exp(seg.masked_fill(~keep, -torch.inf))  # (b, i, j, h)
    return torch.einsum("bijh,bjhp->bihp", w, X)


if HAVE_TRITON:

    @triton.jit
    def _chunk_clock(Ap, c, i, L, s_al, rev, CHUNK: tl.constexpr, EXCL: tl.constexpr):
        """Positions of chunk c's rows in processing order (reversed if rev), which are real, the clock relative to
        the chunk start (inclusive or exclusive running sum of a), and the chunk's total."""
        k = c * CHUNK + i
        ok = k < L
        pos = tl.where(rev != 0, L - 1 - k, k)
        a = tl.load(Ap + pos * s_al, mask=ok, other=0.0)
        incl = tl.cumsum(a, axis=0)
        total = tl.sum(a, axis=0)
        clock = incl - a if EXCL else incl
        return ok, pos, clock, total

    @triton.jit
    def _state_kernel(
        Kp, Vp, Ap, STp, TOTp, L, H, BH, n_chunks, rev_base,
        s_kd, s_kb, s_kl, s_vd, s_vb, s_vl, s_vh, s_ab, s_al,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        EXCL: tl.constexpr, PREC: tl.constexpr,
    ):
        c = tl.program_id(0)
        bh = tl.program_id(1)
        d = tl.program_id(2)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        ok, pos, clock, total = _chunk_clock(Ap + b * s_ab + h, c, i, L, s_al, rev_base ^ d, CHUNK, EXCL)
        k = tl.load(Kp + d * s_kd + b * s_kb + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & (n < N)[None, :], other=0.0)
        v = tl.load(Vp + d * s_vd + b * s_vb + h * s_vh + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & (p < P)[None, :], other=0.0)
        wk = k * tl.exp(total - clock)[:, None]  # reference: the inclusive clock at the chunk's last token
        S = tl.dot(tl.trans(wk), v, input_precision=PREC)
        row = (d * BH + bh) * n_chunks + c
        tl.store(STp + row * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :], S)
        tl.store(TOTp + row, total)

    @triton.jit
    def _pass_kernel(STp, TOTp, n_chunks, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr):
        """in[c] = exp(T[c-1]) * in[c-1] + own[c-1], in[0] = 0: the state entering each chunk, in place.
        One program per (direction, batch * head) row of chunks."""
        r = tl.program_id(0)
        tile = STp + r * n_chunks * BLOCK_N * BLOCK_P + tl.arange(0, BLOCK_N)[:, None] * BLOCK_P + tl.arange(0, BLOCK_P)[None, :]
        carry = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)
        for c in range(0, n_chunks):
            own = tl.load(tile + c * BLOCK_N * BLOCK_P)
            tl.store(tile + c * BLOCK_N * BLOCK_P, carry)
            carry = carry * tl.exp(tl.load(TOTp + r * n_chunks + c)) + own

    @triton.jit
    def _out_kernel(
        Qp, Kp, Vp, Up, Ap, STp, O1p, O2p, Yp, DCp, L, H, BH, n_chunks, rev_base,
        s_qd, s_qb, s_ql, s_kd, s_kb, s_kl, s_vd, s_vb, s_vl, s_vh, s_ud, s_ub, s_ul, s_uh, s_ab, s_al,
        s_o1d, s_o1b, s_o1l, s_o1h, s_o2d, s_o2b, s_o2l, s_o2h,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        HAS_O1: tl.constexpr, HAS_O2: tl.constexpr, HAS_DCLOCK: tl.constexpr, EXCL: tl.constexpr, PREC: tl.constexpr,
    ):
        """HAS_DCLOCK (the transposed pass, where V = dY, U = X and O1 = dX): also write <dY, Y> - <X, dX> per row,
        with Y laid out like O1 and the result in DCp, (directions, b, l, h)."""
        c = tl.program_id(0)
        bh = tl.program_id(1)
        d = tl.program_id(2)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        n_ok = n < N
        p_ok = p < P
        ok, pos, clock, total = _chunk_clock(Ap + b * s_ab + h, c, i, L, s_al, rev_base ^ d, CHUNK, EXCL)
        row = (d * BH + bh) * n_chunks + c
        S = tl.load(STp + row * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :])  # zero for c = 0
        dec = tl.exp(clock)  # from the end of the previous chunk (clock 0) to each row
        D = tl.exp(tl.where(i[:, None] > i[None, :], clock[:, None] - clock[None, :], -float("inf")))
        k = tl.load(Kp + d * s_kd + b * s_kb + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
        v = tl.load(Vp + d * s_vd + b * s_vb + h * s_vh + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
        if HAS_O1:
            q = tl.load(Qp + d * s_qd + b * s_qb + pos[:, None] * s_ql + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
            M = tl.dot(q, tl.trans(k), input_precision=PREC) * D
            o1 = tl.dot(M, v, input_precision=PREC) + dec[:, None] * tl.dot(q, S, input_precision=PREC)
            tl.store(O1p + d * s_o1d + b * s_o1b + h * s_o1h + pos[:, None] * s_o1l + p[None, :], o1, mask=ok[:, None] & p_ok[None, :])
        if HAS_O2:
            u = tl.load(Up + d * s_ud + b * s_ub + h * s_uh + pos[:, None] * s_ul + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
            M2 = tl.dot(u, tl.trans(v), input_precision=PREC) * D
            o2 = tl.dot(M2, k, input_precision=PREC) + dec[:, None] * tl.dot(u, tl.trans(S), input_precision=PREC)
            tl.store(O2p + d * s_o2d + b * s_o2b + h * s_o2h + pos[:, None] * s_o2l + n[None, :], o2, mask=ok[:, None] & n_ok[None, :])
        if HAS_DCLOCK:
            y = tl.load(Yp + d * s_o1d + b * s_o1b + h * s_o1h + pos[:, None] * s_o1l + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
            dc = tl.sum(v * y, axis=1) - tl.sum(u * o1, axis=1)
            tl.store(DCp + ((d * (BH // H) + b) * L + pos) * H + h, dc, mask=ok)

    @triton.jit
    def _suffix_kernel(Dp, Op, L, H, BH, s_d, s_b, s_l, rev_base, EXCL: tl.constexpr, BLOCK: tl.constexpr):
        """Op[k] = sum of Dp over the rows at or after k in processing order (strictly after if EXCL); one program
        per (batch * head, direction)."""
        bh = tl.program_id(0)
        d = tl.program_id(1)
        b = bh // H
        h = bh % H
        rev = rev_base ^ d
        Dp += d * s_d + b * s_b + h
        Op += d * s_d + b * s_b + h
        i = tl.arange(0, BLOCK)
        carry = 0.0
        for blk in range(0, tl.cdiv(L, BLOCK)):
            k = L - 1 - (blk * BLOCK + i)  # processing index, walked from the end
            ok = k >= 0
            pos = tl.where(rev != 0, L - 1 - k, k)
            dd = tl.load(Dp + pos * s_l, mask=ok, other=0.0)
            run = carry + tl.cumsum(dd, axis=0)
            tl.store(Op + pos * s_l, run - dd if EXCL else run, mask=ok)
            carry += tl.sum(dd, axis=0)


def _blocks(n, p):
    return max(16, triton.next_power_of_2(n)), max(16, triton.next_power_of_2(p))


def _ds(t, nd):
    """Direction stride: 0 for a tensor both directions share (a plain (b, l, ...) tensor), else its first stride."""
    return t.stride(0) if t.dim() == nd + 1 else 0


def _bl(t, nd):
    """The (b, l, ...) strides of a tensor that may carry a leading direction axis."""
    return t.stride()[1:] if t.dim() == nd + 1 else t.stride()


def _states(K, V, a, rev_base, n_dir, excl, chunk, prec):
    """K: (b, l, n) or (dirs, b, l, n); V: (b, l, h, p) or (dirs, b, l, h, p). Returns the entering states,
    (dirs, b * h, chunks, BN, BP)."""
    b, length, h = a.shape
    p, n = V.shape[-1], K.shape[-1]
    bn, bp = _blocks(n, p)
    nc = triton.cdiv(length, chunk)
    st = torch.empty(n_dir, b * h, nc, bn, bp, device=V.device, dtype=torch.float32)
    tot = torch.empty(n_dir, b * h, nc, device=V.device, dtype=torch.float32)
    kb, vb = _bl(K, 3), _bl(V, 4)
    _state_kernel[(nc, b * h, n_dir)](
        K, V, a, st, tot, length, h, b * h, nc, int(rev_base),
        _ds(K, 3), kb[0], kb[1], _ds(V, 4), vb[0], vb[1], vb[2], a.stride(0), a.stride(1),
        N=n, P=p, BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk, EXCL=excl, PREC=prec,
    )
    _pass_kernel[(n_dir * b * h,)](st, tot, nc, BLOCK_N=bn, BLOCK_P=bp)
    return st


def _outputs(Q, K, V, U, a, st, rev_base, n_dir, excl, want_o1, want_o2, chunk, prec, Y=None):
    """Outputs per direction: O1 (dirs, b, l, h, p), O2 (dirs, b, l, h, n), and with Y the clock gradient
    (dirs, b, l, h)."""
    b, length, h = a.shape
    p, n = V.shape[-1], K.shape[-1]
    dev = V.device
    bn, bp = _blocks(n, p)
    nc = triton.cdiv(length, chunk)
    o1 = torch.empty(n_dir, b, length, h, p, device=dev, dtype=torch.float32) if want_o1 else torch.empty(1, 1, 1, 1, 1, device=dev)
    o2 = torch.empty(n_dir, b, length, h, n, device=dev, dtype=torch.float32) if want_o2 else torch.empty(1, 1, 1, 1, 1, device=dev)
    Q = Q if want_o1 else K
    U = U if want_o2 else V
    dclock = torch.empty(n_dir, b, length, h, device=dev, dtype=torch.float32) if Y is not None else o1
    qb, kb, vb, ub = _bl(Q, 3), _bl(K, 3), _bl(V, 4), _bl(U, 4)
    _out_kernel[(nc, b * h, n_dir)](
        Q, K, V, U, a, st, o1, o2, Y if Y is not None else o1, dclock, length, h, b * h, nc, int(rev_base),
        _ds(Q, 3), qb[0], qb[1], _ds(K, 3), kb[0], kb[1], _ds(V, 4), vb[0], vb[1], vb[2], _ds(U, 4), ub[0], ub[1], ub[2],
        a.stride(0), a.stride(1),
        o1.stride(0), o1.stride(1), o1.stride(2), o1.stride(3), o2.stride(0), o2.stride(1), o2.stride(2), o2.stride(3),
        N=n, P=p, BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk,
        HAS_O1=want_o1, HAS_O2=want_o2, HAS_DCLOCK=Y is not None, EXCL=excl, PREC=prec,
    )
    return (o1 if want_o1 else None), (o2 if want_o2 else None), (dclock if Y is not None else None)


def _suffix(d, rev_base, excl):
    """d: (dirs, b, l, h) -> the same shape, each direction summed onwards in its own processing order."""
    n_dir, b, length, h = d.shape
    out = torch.empty_like(d)
    _suffix_kernel[(b * h, n_dir)](d, out, length, h, b * h, d.stride(0), d.stride(1), d.stride(2), int(rev_base),
                                   EXCL=excl, BLOCK=1024)
    return out


def _unit_last(t):
    """float32 with a unit stride along the last dimension (the kernels take every other stride as given); GTS's B
    and C are column slices of one projection and need no copy."""
    t = t.float()
    return t if t.stride(-1) == 1 else t.contiguous()


class _GTSScan(torch.autograd.Function):
    """One direction (n_dir = 1, C: (b, l, n)) or both (n_dir = 2, C: (2, b, l, n), direction 1 reversed relative to
    direction 0). The output is the sum over directions."""

    @staticmethod
    def forward(ctx, C, B, X, a, reverse, excl, chunk, prec):
        n_dir = 2 if C.dim() == 4 else 1
        C, B, X, a = (_unit_last(t) for t in (C, B, X, a))
        st = _states(B, X, a, reverse, n_dir, excl, chunk, prec)
        Yd, _, _ = _outputs(C, B, X, None, a, st, reverse, n_dir, excl, True, False, chunk, prec)
        ctx.save_for_backward(C, B, X, a, Yd, st)
        ctx.cfg = reverse, excl, chunk, prec, n_dir
        return Yd.sum(0) if n_dir > 1 else Yd[0]

    @staticmethod
    def backward(ctx, dY):
        C, B, X, a, Yd, st = ctx.saved_tensors
        r, e, chunk, prec, n_dir = ctx.cfg
        dY = _unit_last(dY)
        # The transposed scan runs the other way on the other clock: dX[j] = sum_i w <B[j], C[i]> dY[i] and
        # dB[j] = sum_i w <X[j], dY[i]> C[i], one pass. dC[i] = sum_j w <dY[i], X[j]> B[j] reuses the forward's states.
        tst = _states(C, dY, a, not r, n_dir, not e, chunk, prec)
        dX, dB, dclock = _outputs(B, C, dY, X, a, tst, not r, n_dir, not e, True, True, chunk, prec, Yd)
        _, dC, _ = _outputs(None, B, X, dY, a, st, r, n_dir, e, False, True, chunk, prec)
        da = _suffix(dclock, r, e).sum(0)
        dC = dC.sum(3)  # over heads: (dirs, b, l, n)
        return (dC if n_dir > 1 else dC[0]), dB.sum((0, 3)), dX.sum(0), da, None, None, None, None


def _precision(C, precision):
    if precision is None:
        precision = "tf32" if (C.is_cuda and torch.backends.cuda.matmul.allow_tf32) else "ieee"
    return precision


def gts_scan(C, B, X, a, reverse=False, excl=False, chunk=64, precision=None):
    """C, B: (b, l, n); X: (b, l, h, p); a: (b, l, h) per-token log-decays <= 0. Returns (b, l, h, p) float32.

    ``precision`` is "ieee" or "tf32" for the chunk matmuls; by default it follows
    ``torch.backends.cuda.matmul.allow_tf32``. ``excl`` uses the exclusive running sum (the transposed scan)."""
    if not HAVE_TRITON:
        raise RuntimeError("gts_scan needs triton; use gts_scan_reference")
    return _GTSScan.apply(C, B, X, a, reverse, excl, chunk, _precision(C, precision))


def gts_scan_bi(C_fwd, C_bwd, B, X, a, chunk=64, precision=None):
    """gts_scan(C_fwd, B, X, a) + gts_scan(C_bwd, B, X, a, reverse=True), both directions in the same launches."""
    if not HAVE_TRITON:
        raise RuntimeError("gts_scan_bi needs triton; use gts_scan_reference")
    return _GTSScan.apply(torch.stack([C_fwd, C_bwd]), B, X, a, False, False, chunk, _precision(C_fwd, precision))
