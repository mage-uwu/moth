# Golden Tree Snake (GTS) fork, 2026.
"""Chunked SSD scan for GTS's depth-0 trees (the bank of the mixed forest), in Triton.

A depth-0 tree is visited by every token, so a group of them sharing one clock is exactly Mamba-2's SSD with one
key/query group: for head h and channels p of that head,

    Y[t, h, p] = sum_{s < t} exp(cs[t, h] - cs[s, h]) * <C[t], B[s]> * X[s, h, p]          (causal, "forward")
    Y[t, h, p] = sum_{s > t} exp(cs[s, h] - cs[t, h]) * <C[t], B[s]> * X[s, h, p]          (reverse)

where cs is a running sum of non-positive log-decays (so every exponent is <= 0) and, unlike Mamba-2, the token's
own term s = t is excluded. GTS's training path forms the (batch, t, s, heads) weights explicitly, which is quadratic
in length; this computes the same thing in chunks with an (N x P) state per head carried from chunk to chunk.

One kernel serves the forward and both halves of the backward pass. It walks the chunks in order (or in reverse),
and for each chunk can produce
    O1[t] = sum_s w_ts <Q[t], K[s]> V[s]           (P per head)      forward Y, and dX in the backward pass
    O2[t] = sum_s w_ts <U[t], V[s]> K[s]           (N per head)      dB and dC in the backward pass
from the same running state S = sum_s w K[s] V[s]^T. Two schedules: one program per (batch, head) walking its chunks
(``parallel=False``), or chunk-parallel passes with a short sequential pass in between (the default). The log-decay gradient follows in closed form:
    d cs[t] = <dY[t], Y[t]> - <X[t], dX[t]>        (negated for the reverse direction)

``gts_scan(C, B, X, cs, reverse)`` is differentiable; ``gts_scan_reference`` is the quadratic PyTorch form it is
tested against.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the PyTorch path does not need triton
    triton = None

__all__ = ["gts_scan", "gts_scan_reference", "HAVE_TRITON"]
HAVE_TRITON = triton is not None


def gts_scan_reference(C, B, X, cs, reverse=False):
    """Quadratic PyTorch form. C, B: (b, l, n); X: (b, l, h, p); cs: (b, l, h) running sum of log-decays."""
    length = cs.shape[1]
    G = C @ B.transpose(1, 2)  # (b, t, s)
    seg = cs[:, :, None, :] - cs[:, None, :, :]  # [t, s] = cs[t] - cs[s]
    keep = torch.ones(length, length, dtype=torch.bool, device=cs.device)
    keep = keep.triu(1) if reverse else keep.tril(-1)
    if reverse:
        seg = -seg
    w = G.unsqueeze(-1) * torch.exp(seg.masked_fill(~keep[None, :, :, None], -torch.inf))  # (b, t, s, h)
    return torch.einsum("btsh,bshp->bthp", w, X)


if HAVE_TRITON:

    @triton.jit
    def _scan_kernel(
        Qp, Kp, Vp, Up, CSp, O1p, O2p,
        L, H,
        s_qb, s_ql, s_kb, s_kl,
        s_vb, s_vl, s_vh, s_ub, s_ul, s_uh,
        s_cb, s_cl,
        s_o1b, s_o1l, s_o1h, s_o2b, s_o2l, s_o2h,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        HAS_O1: tl.constexpr, HAS_O2: tl.constexpr, REVERSE: tl.constexpr, PREC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // H
        h = pid % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        n_ok = n < N
        p_ok = p < P
        Qp += b * s_qb
        Kp += b * s_kb
        Vp += b * s_vb + h * s_vh
        Up += b * s_ub + h * s_uh
        CSp += b * s_cb + h
        O1p += b * s_o1b + h * s_o1h
        O2p += b * s_o2b + h * s_o2h

        # The first position's clock is the reference for chunk 0, so no exponent ever exceeds 0.
        first = L - 1 if REVERSE else 0
        cs_ref = tl.load(CSp + first * s_cl)
        if REVERSE:
            cs_ref = -cs_ref
        S = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)  # sum_s exp(cs_ref - cs[s]) K[s] V[s]^T
        causal = i[:, None] > i[None, :]  # strictly earlier in processing order
        for c in range(0, tl.cdiv(L, CHUNK)):
            k_idx = c * CHUNK + i
            ok = k_idx < L
            pos = (L - 1 - k_idx) if REVERSE else k_idx
            cs = tl.load(CSp + pos * s_cl, mask=ok, other=0.0)
            if REVERSE:
                cs = -cs
            last = tl.minimum(L - c * CHUNK, CHUNK) - 1
            cs_end = tl.sum(tl.where(i == last, cs, 0.0), axis=0)
            cs = tl.where(ok, cs, cs_end)  # padding rows: any finite value; their K, V, U are zero
            k = tl.load(Kp + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
            v = tl.load(Vp + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
            D = tl.exp(tl.where(causal, cs[:, None] - cs[None, :], -float("inf")))  # (CHUNK, CHUNK) decay, 0 off the strict triangle
            dec = tl.exp(cs - cs_ref)  # decay from the chunk boundary to each row
            if HAS_O1:
                q = tl.load(Qp + pos[:, None] * s_ql + n[None, :], mask=ok[:, None] & n_ok[None, :], other=0.0)
                M = tl.dot(q, tl.trans(k), input_precision=PREC) * D
                o1 = tl.dot(M, v, input_precision=PREC) + dec[:, None] * tl.dot(q, S, input_precision=PREC)
                tl.store(O1p + pos[:, None] * s_o1l + p[None, :], o1, mask=ok[:, None] & p_ok[None, :])
            if HAS_O2:
                u = tl.load(Up + pos[:, None] * s_ul + p[None, :], mask=ok[:, None] & p_ok[None, :], other=0.0)
                M2 = tl.dot(u, tl.trans(v), input_precision=PREC) * D
                o2 = tl.dot(M2, k, input_precision=PREC) + dec[:, None] * tl.dot(u, tl.trans(S), input_precision=PREC)
                tl.store(O2p + pos[:, None] * s_o2l + n[None, :], o2, mask=ok[:, None] & n_ok[None, :])
            # Move the reference to this chunk's last token and add the chunk's own writes.
            wk = k * tl.where(ok, tl.exp(cs_end - cs), 0.0)[:, None]
            S = S * tl.exp(cs_end - cs_ref) + tl.dot(tl.trans(wk), v, input_precision=PREC)
            cs_ref = cs_end

    @triton.jit
    def _chunk_pos(c, i, L, CHUNK: tl.constexpr, REVERSE: tl.constexpr):
        k_idx = c * CHUNK + i
        return k_idx < L, (L - 1 - k_idx) if REVERSE else k_idx

    @triton.jit
    def _cs_end(CSp, c, L, s_cl, CHUNK: tl.constexpr, REVERSE: tl.constexpr):
        """Clock at the last token of chunk c, in processing order (negated when reversed)."""
        last = tl.minimum((c + 1) * CHUNK, L) - 1
        v = tl.load(CSp + ((L - 1 - last) if REVERSE else last) * s_cl)
        return -v if REVERSE else v

    @triton.jit
    def _state_kernel(
        Kp, Vp, CSp, STp, L, H, n_chunks,
        s_kb, s_kl, s_vb, s_vl, s_vh, s_cb, s_cl,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        REVERSE: tl.constexpr, PREC: tl.constexpr,
    ):
        """Pass 1, one program per (chunk, batch * head): the chunk's own writes, sum_s exp(cs_end - cs[s]) K[s] V[s]^T."""
        c = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        ok, pos = _chunk_pos(c, i, L, CHUNK, REVERSE)
        CSp += b * s_cb + h
        cs = tl.load(CSp + pos * s_cl, mask=ok, other=0.0)
        if REVERSE:
            cs = -cs
        cs_end = _cs_end(CSp, c, L, s_cl, CHUNK, REVERSE)
        k = tl.load(Kp + b * s_kb + pos[:, None] * s_kl + n[None, :], mask=ok[:, None] & (n < N)[None, :], other=0.0)
        v = tl.load(Vp + b * s_vb + h * s_vh + pos[:, None] * s_vl + p[None, :], mask=ok[:, None] & (p < P)[None, :], other=0.0)
        wk = k * tl.where(ok, tl.exp(cs_end - cs), 0.0)[:, None]
        S = tl.dot(tl.trans(wk), v, input_precision=PREC)
        tl.store(STp + (bh * n_chunks + c) * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :], S)


    @triton.jit
    def _pass_kernel(CSp, STp, L, H, n_chunks, s_cb, s_cl,
                     BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr, REVERSE: tl.constexpr):
        """Between the two parallel passes, one program per (batch, head): turn each chunk's own state into the state
        entering it, in place: in[c] = exp(end[c-1] - end[c-2]) * in[c-1] + own[c-1], referenced to the end of chunk c-1."""
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H
        CSp += b * s_cb + h
        tile = STp + bh * n_chunks * BLOCK_N * BLOCK_P + tl.arange(0, BLOCK_N)[:, None] * BLOCK_P + tl.arange(0, BLOCK_P)[None, :]
        carry = tl.zeros((BLOCK_N, BLOCK_P), dtype=tl.float32)
        prev = _cs_end(CSp, 0, L, s_cl, CHUNK, REVERSE)
        for c in range(0, n_chunks):
            own = tl.load(tile + c * BLOCK_N * BLOCK_P)
            tl.store(tile + c * BLOCK_N * BLOCK_P, carry)
            end = _cs_end(CSp, c, L, s_cl, CHUNK, REVERSE)
            carry = carry * tl.exp(end - prev) + own
            prev = end

    @triton.jit
    def _out_kernel(
        Qp, Kp, Vp, Up, CSp, STp, O1p, O2p, L, H, n_chunks,
        s_qb, s_ql, s_kb, s_kl, s_vb, s_vl, s_vh, s_ub, s_ul, s_uh, s_cb, s_cl,
        s_o1b, s_o1l, s_o1h, s_o2b, s_o2l, s_o2h,
        N: tl.constexpr, P: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_P: tl.constexpr, CHUNK: tl.constexpr,
        HAS_O1: tl.constexpr, HAS_O2: tl.constexpr, REVERSE: tl.constexpr, PREC: tl.constexpr,
    ):
        """Pass 3, one program per (chunk, batch * head): the chunk's outputs, intra-chunk plus the carried state."""
        c = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        i = tl.arange(0, CHUNK)
        n = tl.arange(0, BLOCK_N)
        p = tl.arange(0, BLOCK_P)
        n_ok = n < N
        p_ok = p < P
        CSp += b * s_cb + h
        ok, pos = _chunk_pos(c, i, L, CHUNK, REVERSE)
        cs = tl.load(CSp + pos * s_cl, mask=ok, other=0.0)
        if REVERSE:
            cs = -cs
        cs = tl.where(ok, cs, _cs_end(CSp, c, L, s_cl, CHUNK, REVERSE))
        # carried state, referenced to the clock at the end of chunk c - 1
        # state entering the chunk (from _pass_kernel; zero for chunk 0), referenced to the end of chunk c - 1
        S = tl.load(STp + (bh * n_chunks + c) * BLOCK_N * BLOCK_P + n[:, None] * BLOCK_P + p[None, :])
        ref = _cs_end(CSp, tl.maximum(c - 1, 0), L, s_cl, CHUNK, REVERSE)
        dec = tl.where(c > 0, tl.exp(cs - ref), 0.0)
        causal = i[:, None] > i[None, :]
        D = tl.exp(tl.where(causal, cs[:, None] - cs[None, :], -float("inf")))
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


def _launch(Q, K, V, U, cs, reverse, want_o1, want_o2, chunk=64, precision="ieee"):
    b, length, h, p = V.shape
    n = K.shape[-1]
    dev = V.device
    o1 = torch.empty(b, length, h, p, device=dev, dtype=torch.float32) if want_o1 else torch.empty(1, 1, 1, 1, device=dev)
    o2 = torch.empty(b, length, h, n, device=dev, dtype=torch.float32) if want_o2 else torch.empty(1, 1, 1, 1, device=dev)
    Q = Q if want_o1 else K
    U = U if want_o2 else V
    _scan_kernel[(b * h,)](
        Q, K, V, U, cs, o1, o2,
        length, h,
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1),
        V.stride(0), V.stride(1), V.stride(2), U.stride(0), U.stride(1), U.stride(2),
        cs.stride(0), cs.stride(1),
        o1.stride(0), o1.stride(1), o1.stride(2), o2.stride(0), o2.stride(1), o2.stride(2),
        N=n, P=p, BLOCK_N=max(16, triton.next_power_of_2(n)), BLOCK_P=max(16, triton.next_power_of_2(p)), CHUNK=chunk,
        HAS_O1=want_o1, HAS_O2=want_o2, REVERSE=reverse, PREC=precision,
    )
    return (o1 if want_o1 else None), (o2 if want_o2 else None)


def _states(K, V, cs, reverse, chunk, precision):
    b, length, h, p = V.shape
    n = K.shape[-1]
    bn, bp = max(16, triton.next_power_of_2(n)), max(16, triton.next_power_of_2(p))
    nc = triton.cdiv(length, chunk)
    st = torch.empty(b * h, nc, bn, bp, device=V.device, dtype=torch.float32)
    _state_kernel[(nc, b * h)](
        K, V, cs, st, length, h, nc,
        K.stride(0), K.stride(1), V.stride(0), V.stride(1), V.stride(2), cs.stride(0), cs.stride(1),
        N=n, P=p, BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk, REVERSE=reverse, PREC=precision,
    )
    _pass_kernel[(b * h,)](cs, st, length, h, nc, cs.stride(0), cs.stride(1), BLOCK_N=bn, BLOCK_P=bp, CHUNK=chunk, REVERSE=reverse)
    return st


def _outputs(Q, K, V, U, cs, st, reverse, want_o1, want_o2, chunk, precision):
    b, length, h, p = V.shape
    n = K.shape[-1]
    dev = V.device
    nc = triton.cdiv(length, chunk)
    o1 = torch.empty(b, length, h, p, device=dev, dtype=torch.float32) if want_o1 else torch.empty(1, 1, 1, 1, device=dev)
    o2 = torch.empty(b, length, h, n, device=dev, dtype=torch.float32) if want_o2 else torch.empty(1, 1, 1, 1, device=dev)
    Q = Q if want_o1 else K
    U = U if want_o2 else V
    _out_kernel[(nc, b * h)](
        Q, K, V, U, cs, st, o1, o2, length, h, nc,
        Q.stride(0), Q.stride(1), K.stride(0), K.stride(1), V.stride(0), V.stride(1), V.stride(2),
        U.stride(0), U.stride(1), U.stride(2), cs.stride(0), cs.stride(1),
        o1.stride(0), o1.stride(1), o1.stride(2), o2.stride(0), o2.stride(1), o2.stride(2),
        N=n, P=p, BLOCK_N=max(16, triton.next_power_of_2(n)), BLOCK_P=max(16, triton.next_power_of_2(p)), CHUNK=chunk,
        HAS_O1=want_o1, HAS_O2=want_o2, REVERSE=reverse, PREC=precision,
    )
    return (o1 if want_o1 else None), (o2 if want_o2 else None)


class _GTSScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, B, X, cs, reverse, chunk, precision, parallel):
        C, B, X, cs = (t.float().contiguous() for t in (C, B, X, cs))
        st = None
        if parallel:
            st = _states(B, X, cs, reverse, chunk, precision)
            Y, _ = _outputs(C, B, X, None, cs, st, reverse, True, False, chunk, precision)
        else:
            Y, _ = _launch(C, B, X, None, cs, reverse, True, False, chunk, precision)
        ctx.save_for_backward(C, B, X, cs, Y, *([st] if parallel else []))
        ctx.reverse, ctx.chunk, ctx.precision, ctx.parallel = reverse, chunk, precision, parallel
        return Y

    @staticmethod
    def backward(ctx, dY):
        C, B, X, cs, Y, *st = ctx.saved_tensors
        r, chunk, prec = ctx.reverse, ctx.chunk, ctx.precision
        dY = dY.float().contiguous()
        # Opposite direction: dX[s] = sum_t w <B[s], C[t]> dY[t] and dB[s] = sum_t w <X[s], dY[t]> C[t], one pass.
        # Same direction: dC[t] = sum_s w <dY[t], X[s]> B[s], from the forward pass's chunk states.
        if ctx.parallel:
            dX, dB = _outputs(B, C, dY, X, cs, _states(C, dY, cs, not r, chunk, prec), not r, True, True, chunk, prec)
            _, dC = _outputs(None, B, X, dY, cs, st[0], r, False, True, chunk, prec)
        else:
            dX, dB = _launch(B, C, dY, X, cs, not r, True, True, chunk, prec)
            _, dC = _launch(None, B, X, dY, cs, r, False, True, chunk, prec)
        dcs = (dY * Y).sum(-1) - (X * dX).sum(-1)
        if r:
            dcs = -dcs
        return dC.sum(2), dB.sum(2), dX, dcs, None, None, None, None


def gts_scan(C, B, X, cs, reverse=False, chunk=64, precision="ieee", parallel=True):
    """C, B: (b, l, n); X: (b, l, h, p); cs: (b, l, h), non-increasing along l (a running sum of log-decays <= 0).
    Returns (b, l, h, p) float32. ``precision="tf32"`` lets the chunk matmuls use TF32 on GPUs that have it.
    ``parallel=True`` runs one program per chunk (chunk states, a light sequential pass that turns them into carried
    states, then outputs); ``False`` walks each (batch, head)'s chunks in one program."""
    if not HAVE_TRITON:
        raise RuntimeError("gts_scan needs triton; use gts_scan_reference")
    return _GTSScan.apply(C, B, X, cs, reverse, chunk, precision, parallel)
