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

``gts_scan(C, B, X, a, reverse)`` is differentiable; ``gts_scan_reference`` is the quadratic form it is tested against.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the PyTorch path does not need triton
    triton = None

__all__ = ["gts_scan", "gts_scan_reference", "HAVE_TRITON"]
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
    def _chunk_clock(Ap, c, i, L, s_al, CHUNK: tl.constexpr, REVERSE: tl.constexpr, EXCL: tl.constexpr):
        """Positions of chunk c's rows in processing order, which are real, the clock relative to the chunk start
        (inclusive or exclusive running sum of a), and the chunk's total."""
        k = c * CHUNK + i
        ok = k < L
        pos = (L - 1 - k) if REVERSE else k
        a = tl.load(Ap + pos * s_al, mask=ok, other=0.0)
        incl = tl.cumsum(a, axis=0)
        total = tl.sum(a, axis=0)
        clock = incl - a if EXCL else incl
        return ok, pos, clock, total

    @triton.jit
    def _state_kernel(
        Kp, Vp, Ap, STp, TOTp, L, H, n_chunks,
        s_kb, s_kl, s_vb, s_vl, s_vh, s_ab, s_al,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        REVERSE: tl.constexpr, EXCL: tl.constexpr, PREC: tl.constexpr,
    ):
        c = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        ok, pos, clock, total = _chunk_clock(Ap + b * s_ab + h, c, i, L, s_al, CHUNK, REVERSE, EXCL)
        k = tl.load(Kp + b * s_kb + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & (n < N)[None, :], other=0.0)
        v = tl.load(Vp + b * s_vb + h * s_vh + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & (p < P)[None, :], other=0.0)
        wk = k * tl.exp(total - clock)[:, None]  # reference: the inclusive clock at the chunk's last token
        S = tl.dot(tl.trans(wk), v, input_precision=PREC)
        tl.store(STp + (bh * n_chunks + c) * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :], S)
        tl.store(TOTp + bh * n_chunks + c, total)

    @triton.jit
    def _pass_kernel(STp, TOTp, n_chunks, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr):
        """in[c] = exp(T[c-1]) * in[c-1] + own[c-1], in[0] = 0: the state entering each chunk, in place."""
        bh = tl.program_id(0)
        tile = STp + bh * n_chunks * BLOCK_N * BLOCK_P + tl.arange(0, BLOCK_N)[:, None] * BLOCK_P + tl.arange(0, BLOCK_P)[None, :]
        carry = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)
        for c in range(0, n_chunks):
            own = tl.load(tile + c * BLOCK_N * BLOCK_P)
            tl.store(tile + c * BLOCK_N * BLOCK_P, carry)
            carry = carry * tl.exp(tl.load(TOTp + bh * n_chunks + c)) + own

    @triton.jit
    def _out_kernel(
        Qp, Kp, Vp, Up, Ap, STp, O1p, O2p, Yp, DCp, L, H, n_chunks,
        s_qb, s_ql, s_kb, s_kl, s_vb, s_vl, s_vh, s_ub, s_ul, s_uh, s_ab, s_al,
        s_o1b, s_o1l, s_o1h, s_o2b, s_o2l, s_o2h,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        HAS_O1: tl.constexpr, HAS_O2: tl.constexpr, HAS_DCLOCK: tl.constexpr,
        REVERSE: tl.constexpr, EXCL: tl.constexpr, PREC: tl.constexpr,
    ):
        """HAS_DCLOCK (the transposed pass, where V = dY, U = X and O1 = dX): also write <dY, Y> - <X, dX> per row,
        with Y laid out like O1."""
        c = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        n_ok = n < N
        p_ok = p < P
        ok, pos, clock, total = _chunk_clock(Ap + b * s_ab + h, c, i, L, s_al, CHUNK, REVERSE, EXCL)
        S = tl.load(STp + (bh * n_chunks + c) * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :])  # zero for c = 0
        dec = tl.exp(clock)  # from the end of the previous chunk (clock 0) to each row
        D = tl.exp(tl.where(i[:, None] > i[None, :], clock[:, None] - clock[None, :], -float("inf")))
        k = tl.load(Kp + b * s_kb + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
        v = tl.load(Vp + b * s_vb + h * s_vh + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
        if HAS_O1:
            q = tl.load(Qp + b * s_qb + pos[:, None] * s_ql + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
            M = tl.dot(q, tl.trans(k), input_precision=PREC) * D
            o1 = tl.dot(M, v, input_precision=PREC) + dec[:, None] * tl.dot(q, S, input_precision=PREC)
            tl.store(O1p + b * s_o1b + h * s_o1h + pos[:, None] * s_o1l + p[None, :], o1, mask=ok[:, None] & p_ok[None, :])
        if HAS_O2:
            u = tl.load(Up + b * s_ub + h * s_uh + pos[:, None] * s_ul + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
            M2 = tl.dot(u, tl.trans(v), input_precision=PREC) * D
            o2 = tl.dot(M2, k, input_precision=PREC) + dec[:, None] * tl.dot(u, tl.trans(S), input_precision=PREC)
            tl.store(O2p + b * s_o2b + h * s_o2h + pos[:, None] * s_o2l + n[None, :], o2, mask=ok[:, None] & n_ok[None, :])
        if HAS_DCLOCK:
            y = tl.load(Yp + b * s_o1b + h * s_o1h + pos[:, None] * s_o1l + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
            dc = tl.sum(v * y, axis=1) - tl.sum(u * o1, axis=1)
            tl.store(DCp + (b * L + pos) * H + h, dc, mask=ok)

    @triton.jit
    def _suffix_kernel(Dp, Op, L, H, s_b, s_l, REVERSE: tl.constexpr, EXCL: tl.constexpr, BLOCK: tl.constexpr):
        """Op[k] = sum of Dp over the rows at or after k in processing order (strictly after if EXCL)."""
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H
        Dp += b * s_b + h
        Op += b * s_b + h
        i = tl.arange(0, BLOCK)
        carry = 0.0
        for blk in range(0, tl.cdiv(L, BLOCK)):
            k = L - 1 - (blk * BLOCK + i)  # processing index, walked from the end
            ok = k >= 0
            pos = (L - 1 - k) if REVERSE else k
            d = tl.load(Dp + pos * s_l, mask=ok, other=0.0)
            run = carry + tl.cumsum(d, axis=0)
            tl.store(Op + pos * s_l, run - d if EXCL else run, mask=ok)
            carry += tl.sum(d, axis=0)


def _blocks(n, p):
    return max(16, triton.next_power_of_2(n)), max(16, triton.next_power_of_2(p))


def _states(K, V, a, reverse, excl, chunk, prec):
    b, length, h, p = V.shape
    n = K.shape[-1]
    bn, bp = _blocks(n, p)
    nc = triton.cdiv(length, chunk)
    st = torch.empty(b * h, nc, bn, bp, device=V.device, dtype=torch.float32)
    tot = torch.empty(b * h, nc, device=V.device, dtype=torch.float32)
    _state_kernel[(nc, b * h)](
        K, V, a, st, tot, length, h, nc,
        K.stride(0), K.stride(1), V.stride(0), V.stride(1), V.stride(2), a.stride(0), a.stride(1),
        N=n, P=p, BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk, REVERSE=reverse, EXCL=excl, PREC=prec,
    )
    _pass_kernel[(b * h,)](st, tot, nc, BLOCK_N=bn, BLOCK_P=bp)
    return st


def _outputs(Q, K, V, U, a, st, reverse, excl, want_o1, want_o2, chunk, prec, Y=None):
    b, length, h, p = V.shape
    n = K.shape[-1]
    dev = V.device
    bn, bp = _blocks(n, p)
    nc = triton.cdiv(length, chunk)
    o1 = torch.empty(b, length, h, p, device=dev, dtype=torch.float32) if want_o1 else torch.empty(1, 1, 1, 1, device=dev)
    o2 = torch.empty(b, length, h, n, device=dev, dtype=torch.float32) if want_o2 else torch.empty(1, 1, 1, 1, device=dev)
    Q = Q if want_o1 else K
    U = U if want_o2 else V
    dclock = torch.empty(b, length, h, device=dev, dtype=torch.float32) if Y is not None else o1
    _out_kernel[(nc, b * h)](
        Q, K, V, U, a, st, o1, o2, Y if Y is not None else o1, dclock, length, h, nc,
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1), V.stride(0), V.stride(1), V.stride(2),
        U.stride(0), U.stride(1), U.stride(2), a.stride(0), a.stride(1),
        o1.stride(0), o1.stride(1), o1.stride(2), o2.stride(0), o2.stride(1), o2.stride(2),
        N=n, P=p, BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk,
        HAS_O1=want_o1, HAS_O2=want_o2, HAS_DCLOCK=Y is not None, REVERSE=reverse, EXCL=excl, PREC=prec,
    )
    if Y is not None:
        return o1, o2, dclock
    return (o1 if want_o1 else None), (o2 if want_o2 else None)


def _suffix(d, reverse, excl):
    b, length, h = d.shape
    out = torch.empty_like(d)
    _suffix_kernel[(b * h,)](d, out, length, h, d.stride(0), d.stride(1), REVERSE=reverse, EXCL=excl, BLOCK=1024)
    return out


def _unit_last(t):
    """float32 with a unit stride along the last dimension (the kernels take every other stride as given); GTS's B
    and C are column slices of one projection and need no copy."""
    t = t.float()
    return t if t.stride(-1) == 1 else t.contiguous()


class _GTSScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, B, X, a, reverse, excl, chunk, prec):
        C, B, X, a = (_unit_last(t) for t in (C, B, X, a))
        st = _states(B, X, a, reverse, excl, chunk, prec)
        Y, _ = _outputs(C, B, X, None, a, st, reverse, excl, True, False, chunk, prec)
        ctx.save_for_backward(C, B, X, a, Y, st)
        ctx.cfg = reverse, excl, chunk, prec
        return Y

    @staticmethod
    def backward(ctx, dY):
        C, B, X, a, Y, st = ctx.saved_tensors
        r, e, chunk, prec = ctx.cfg
        dY = _unit_last(dY)
        # The transposed scan runs the other way on the other clock: dX[j] = sum_i w <B[j], C[i]> dY[i] and
        # dB[j] = sum_i w <X[j], dY[i]> C[i], one pass. dC[i] = sum_j w <dY[i], X[j]> B[j] reuses the forward's states.
        dX, dB, dclock = _outputs(B, C, dY, X, a, _states(C, dY, a, not r, not e, chunk, prec), not r, not e, True, True, chunk, prec, Y)
        _, dC = _outputs(None, B, X, dY, a, st, r, e, False, True, chunk, prec)
        return dC.sum(2), dB.sum(2), dX, _suffix(dclock, r, e), None, None, None, None


def gts_scan(C, B, X, a, reverse=False, excl=False, chunk=64, precision=None):
    """C, B: (b, l, n); X: (b, l, h, p); a: (b, l, h) per-token log-decays <= 0. Returns (b, l, h, p) float32.

    ``precision`` is "ieee" or "tf32" for the chunk matmuls; by default it follows
    ``torch.backends.cuda.matmul.allow_tf32``. ``excl`` uses the exclusive running sum (the transposed scan)."""
    if not HAVE_TRITON:
        raise RuntimeError("gts_scan needs triton; use gts_scan_reference")
    if precision is None:
        precision = "tf32" if (C.is_cuda and torch.backends.cuda.matmul.allow_tf32) else "ieee"
    return _GTSScan.apply(C, B, X, a, reverse, excl, chunk, precision)
