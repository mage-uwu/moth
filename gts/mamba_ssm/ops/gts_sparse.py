# Golden Tree Snake (GTS) fork, 2026.
"""Sparse deep trees: each token's branch logits and output computed only on its own path.

The dense route path (``ops/gts_route.py``) computes every node's logit, ``L = x @ W_in^T`` over all trees x nodes
(4 x 1,023 for the 110M model), then ``out = A @ W_out`` with A one-hot on the path. A token uses DEPTH + 1 nodes
per tree, so about 99% of those two GEMMs is multiplied by zero. Here one Triton program walks a block of tokens
through every tree:

* phase 1, per tree and level: the logit of the token's current node as a dot product of its input row with that
  node's row of W_in (a row gather), then the branch (logit > 0) picks the child. The path's node ids and logits go
  to a small (tokens, trees, depth + 1) buffer.
* phase 2, per chunk of d_model: ``out = sum over path nodes of coef(logit) * W_out[node]`` (row gathers again).

The node weights (4 x 1,023 x 768 x 2 bf16 = 12.6 MB for the 110M model) stay in L2, so the cost is gathers, not
FLOPs. The function is the dense path's: same nodes (the logit is rounded to the input dtype before the branch,
as the dense bf16 GEMM rounds it), same output up to summation order. Forward only for now; see ``SparseRoute``.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

__all__ = ["sparse_path_route", "sparse_route_fwd", "top_rows", "sparse_route_fwd_packed", "pack_rows", "HAVE_TRITON"]
HAVE_TRITON = triton is not None

if HAVE_TRITON:

    @triton.jit
    def _coef(x, ACT: tl.constexpr):
        """ACT 0: gelu (also split with no context), 1: linear."""
        if ACT == 0:
            return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        return x

    @triton.jit
    def _sparse_fwd(X, WI, BIAS, WO, OUT, NODES, LOGITS, LT, AT, n_tok, d, s_x, s_o, s_lt, s_at,
                    N_TREES: tl.constexpr, N_NODES: tl.constexpr, DEPTH: tl.constexpr, ACT: tl.constexpr,
                    HAS_BIAS: tl.constexpr, ROUND_BF16: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                    PHASES: tl.constexpr, X_RES: tl.constexpr, DP: tl.constexpr, TOP: tl.constexpr):
        """TOP > 0: the first TOP levels (2^TOP - 1 nodes a tree, shared by many tokens) come as dense logits LT
        from a small GEMM; their coefficients go to AT for the matching output GEMM, and only the deeper levels
        are gathered."""
        NT: tl.constexpr = 2 ** TOP - 1
        tok = tl.program_id(0) * BM + tl.arange(0, BM)
        ok = tok < n_tok
        cols = tl.arange(0, BD)
        P: tl.constexpr = N_TREES * (DEPTH + 1)
        # phase 1: walk every tree, keeping the path's node ids and logits
        if PHASES & 1:
            if X_RES:  # the block's inputs, loaded once for all trees and levels (DP: d_model rounded up to a power of 2)
                full = tl.arange(0, DP)
                fm = full < d
                xr = tl.load(X + tok[:, None] * s_x + full[None, :], mask=ok[:, None] & fm[None, :], other=0.0).to(tl.float32)
            for t in range(N_TREES):
                cur = tl.zeros((BM,), dtype=tl.int32)
                for k in range(DEPTH + 1):
                    node = t * N_NODES + cur
                    if k < TOP:  # a top level: the logit from the dense GEMM (bias included), the coefficient to AT
                        acc = tl.load(LT + tok * s_lt + t * NT + cur, mask=ok, other=0.0).to(tl.float32)
                        tl.store(AT + tok * s_at + t * NT + cur, _coef(acc, ACT).to(AT.dtype.element_ty), mask=ok)
                    elif X_RES:
                        wv = tl.load(WI + node[:, None] * d + full[None, :], mask=ok[:, None] & fm[None, :], other=0.0)
                        acc = tl.sum(xr * wv.to(tl.float32), axis=1)
                    else:
                        acc = tl.zeros((BM,), dtype=tl.float32)
                        for c in range(0, d, BD):
                            cm = (c + cols) < d
                            xv = tl.load(X + tok[:, None] * s_x + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                            wv = tl.load(WI + node[:, None] * d + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                            acc += tl.sum(xv.to(tl.float32) * wv.to(tl.float32), axis=1)
                    if k >= TOP:
                        if HAS_BIAS:
                            acc += tl.load(BIAS + node, mask=ok, other=0.0).to(tl.float32)
                        if ROUND_BF16:  # the dense path's bf16 GEMM hands back bf16 logits: branch on the same value
                            acc = acc.to(tl.bfloat16).to(tl.float32)
                    tl.store(NODES + tok * P + t * (DEPTH + 1) + k, node, mask=ok)
                    tl.store(LOGITS + tok * P + t * (DEPTH + 1) + k, acc, mask=ok)
                    cur = 2 * cur + 1 + (acc > 0).to(tl.int32)
        if not (PHASES & 2):
            return
        # phase 2: the output, a chunk of d_model at a time
        for c in range(0, d, BD):
            cm = (c + cols) < d
            out = tl.zeros((BM, BD), dtype=tl.float32)
            for t in range(N_TREES):
                for k in range(TOP, DEPTH + 1):
                    j = t * (DEPTH + 1) + k
                    node = tl.load(NODES + tok * P + j, mask=ok, other=0)
                    cf = _coef(tl.load(LOGITS + tok * P + j, mask=ok, other=0.0), ACT)
                    wv = tl.load(WO + node[:, None] * d + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                    out += cf[:, None] * wv.to(tl.float32)
            tl.store(OUT + tok[:, None] * s_o + (c + cols)[None, :], out, mask=ok[:, None] & cm[None, :])

    @triton.jit
    def _wtile(P8, SC, node, c, cols, ok, cm, row_bytes, NG, G: tl.constexpr):
        """Rows ``node`` of a 2-bit packed ternary table, columns c .. c + BD (BD divides the group size G):
        code (0, 1, 2 -> -1, 0, +1) times the row's scale for that group, in float32."""
        col = c + cols
        byt = tl.load(P8 + node[:, None] * row_bytes + (col // 4)[None, :], mask=ok[:, None] & cm[None, :], other=1)
        code = ((byt.to(tl.int32) >> ((col % 4) * 2)[None, :]) & 3) - 1
        sc = tl.load(SC + node * NG + c // G, mask=ok, other=0.0).to(tl.float32)
        return code.to(tl.float32) * sc[:, None]

    @triton.jit
    def _sparse_fwd_packed(X, PI, SI, BIAS, PO, SO, OUT, NODES, LOGITS, n_tok, d, s_x, s_o, row_bytes, NG,
                           N_TREES: tl.constexpr, N_NODES: tl.constexpr, DEPTH: tl.constexpr, ACT: tl.constexpr,
                           HAS_BIAS: tl.constexpr, ROUND_BF16: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr,
                           G: tl.constexpr, PHASES: tl.constexpr):
        """_sparse_fwd with both node tables as 2-bit ternary codes plus group scales: ~7.5x fewer bytes per row."""
        tok = tl.program_id(0) * BM + tl.arange(0, BM)
        ok = tok < n_tok
        cols = tl.arange(0, BD)
        P: tl.constexpr = N_TREES * (DEPTH + 1)
        if PHASES & 1:
            for t in range(N_TREES):
                cur = tl.zeros((BM,), dtype=tl.int32)
                for k in range(DEPTH + 1):
                    node = t * N_NODES + cur
                    acc = tl.zeros((BM,), dtype=tl.float32)
                    for c in range(0, d, BD):
                        cm = (c + cols) < d
                        xv = tl.load(X + tok[:, None] * s_x + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                        acc += tl.sum(xv.to(tl.float32) * _wtile(PI, SI, node, c, cols, ok, cm, row_bytes, NG, G), axis=1)
                    if HAS_BIAS:
                        acc += tl.load(BIAS + node, mask=ok, other=0.0).to(tl.float32)
                    if ROUND_BF16:
                        acc = acc.to(tl.bfloat16).to(tl.float32)
                    tl.store(NODES + tok * P + t * (DEPTH + 1) + k, node, mask=ok)
                    tl.store(LOGITS + tok * P + t * (DEPTH + 1) + k, acc, mask=ok)
                    cur = 2 * cur + 1 + (acc > 0).to(tl.int32)
        if not (PHASES & 2):
            return
        for c in range(0, d, BD):
            cm = (c + cols) < d
            out = tl.zeros((BM, BD), dtype=tl.float32)
            for j in range(P):
                node = tl.load(NODES + tok * P + j, mask=ok, other=0)
                cf = _coef(tl.load(LOGITS + tok * P + j, mask=ok, other=0.0), ACT)
                out += cf[:, None] * _wtile(PO, SO, node, c, cols, ok, cm, row_bytes, NG, G)
            tl.store(OUT + tok[:, None] * s_o + (c + cols)[None, :], out, mask=ok[:, None] & cm[None, :])


def top_rows(n_trees, n_nodes, top, device):
    """Global row ids of every tree's first ``top`` levels, tree-major (the column order of LT and AT)."""
    nt = 2 ** top - 1
    return (torch.arange(n_trees, device=device)[:, None] * n_nodes + torch.arange(nt, device=device)[None, :]).reshape(-1)


def sparse_route_fwd(x, w_in, bias, w_out, n_trees, n_nodes, depth, act="gelu", block_m=8, block_d=128, num_warps=2,
                     x_resident=False, phases=3, buffers=None, top=0, top_weights=None):
    """x: (tokens, d); w_in, w_out: (>= trees * nodes, d) node rows (padding rows after the trees are never read);
    bias: (>= trees * nodes,) or None. Returns (out, nodes, logits): out (tokens, d) float32, nodes (tokens, trees,
    depth + 1) int32 global node ids along each path, logits the same shape in float32. ``act`` is "gelu", "split"
    (the same with no context) or "linear". ``top`` > 0: the first ``top`` levels by two small dense GEMMs (their
    rows, from top_rows, can be passed pre-sliced as ``top_weights`` = (w_in_top, bias_top, w_out_top)); the logits
    returned for those levels are then the GEMM's (rounded to x's dtype, as the dense path's)."""
    if not HAVE_TRITON:
        raise RuntimeError("sparse_route_fwd needs triton")
    code = {"gelu": 0, "split": 0, "linear": 1}[act]
    x = x.contiguous()
    n_tok, d = x.shape
    w_in = w_in.to(x.dtype).contiguous()
    w_out = w_out.to(x.dtype).contiguous()
    out = torch.empty(n_tok, d, device=x.device, dtype=torch.float32)
    if buffers is None:  # (benchmarking) phase 2 alone reuses a walk's nodes and logits
        nodes = torch.empty(n_tok, n_trees, depth + 1, device=x.device, dtype=torch.int32)
        logits = torch.empty(n_tok, n_trees, depth + 1, device=x.device, dtype=torch.float32)
    else:
        nodes, logits = buffers
    b = bias.contiguous() if bias is not None else out
    top = min(top, depth + 1)
    if top:
        if top_weights is None:
            rows = top_rows(n_trees, n_nodes, top, x.device)
            top_weights = (w_in[rows], bias[rows] if bias is not None else None, w_out[rows])
        wi_t, b_t, wo_t = top_weights
        lt = torch.nn.functional.linear(x, wi_t.to(x.dtype), b_t.to(x.dtype) if b_t is not None else None)
        at = torch.zeros_like(lt)
    else:
        lt = at = out
    _sparse_fwd[(triton.cdiv(n_tok, block_m),)](
        x, w_in, b, w_out, out, nodes, logits, lt, at, n_tok, d, x.stride(0), out.stride(0), lt.stride(0), at.stride(0),
        N_TREES=n_trees, N_NODES=n_nodes, DEPTH=depth, ACT=code, HAS_BIAS=bias is not None,
        ROUND_BF16=x.dtype == torch.bfloat16, BM=block_m, BD=block_d, PHASES=phases, X_RES=x_resident,
        DP=triton.next_power_of_2(d), TOP=top, num_warps=num_warps)
    if top and phases & 2:
        out += (at @ wo_t.to(at.dtype)).float()
    return out, nodes, logits


@torch.no_grad()
def pack_rows(codes, scales, scale_dtype=torch.bfloat16):
    """codes: (rows, d) in {-1, 0, +1} (any integer or float dtype), d a multiple of 4; scales: (rows, d / group).
    Returns (packed uint8 (rows, d / 4), scales in ``scale_dtype``): four codes a byte, code + 1 in two bits, low
    bits first. In bf16 the dense path's weights are bf16(scale) * code exactly, so these give the same values."""
    rows, d = codes.shape
    assert d % 4 == 0, "d_model must be a multiple of 4"
    u = (codes.to(torch.int16) + 1).to(torch.uint8).view(rows, d // 4, 4)
    packed = u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)
    return packed.contiguous(), scales.to(scale_dtype).contiguous()


def sparse_route_fwd_packed(x, packed_in, packed_out, bias, n_trees, n_nodes, depth, group, act="gelu", block_m=8,
                            block_d=128, num_warps=2, phases=3, buffers=None):
    """sparse_route_fwd with 2-bit node tables: packed_in, packed_out are (packed, scales) pairs from pack_rows.
    ``group``: the ternary group size (columns per scale); block_d must divide it."""
    if not HAVE_TRITON:
        raise RuntimeError("sparse_route_fwd_packed needs triton")
    assert group % block_d == 0, "block_d must divide the ternary group size"
    code = {"gelu": 0, "split": 0, "linear": 1}[act]
    x = x.contiguous()
    n_tok, d = x.shape
    (pi, si), (po, so) = packed_in, packed_out
    out = torch.empty(n_tok, d, device=x.device, dtype=torch.float32)
    if buffers is None:
        nodes = torch.empty(n_tok, n_trees, depth + 1, device=x.device, dtype=torch.int32)
        logits = torch.empty(n_tok, n_trees, depth + 1, device=x.device, dtype=torch.float32)
    else:
        nodes, logits = buffers
    b = bias.contiguous() if bias is not None else out
    _sparse_fwd_packed[(triton.cdiv(n_tok, block_m),)](
        x, pi, si, b, po, so, out, nodes, logits, n_tok, d, x.stride(0), out.stride(0), d // 4, si.shape[1],
        N_TREES=n_trees, N_NODES=n_nodes, DEPTH=depth, ACT=code, HAS_BIAS=bias is not None,
        ROUND_BF16=x.dtype == torch.bfloat16, BM=block_m, BD=block_d, G=group, PHASES=phases, num_warps=num_warps)
    return out, nodes, logits


# ---------------------------------------------------------------------------------------------- training (backward)
#
# With the plain FFF routing gradient (route_ste=False) every gradient of the deep trees lives on the tokens' paths:
#     g[n, j]  = dout[n] . W_out[node(n, j)]           (j over the trees x levels of token n's path)
#     dL[n, j] = coef'(L[n, j]) * g[n, j]
#     dx[n]    = sum_j dL[n, j] * W_in[node(n, j)]
#     dW_in[m] = sum over path entries at node m of dL * x[n];   dbias[m] = sum of dL
#     dW_out[m] = sum over path entries at node m of coef(L) * dout[n]
# which is exactly what the dense route path computes with route_ste=False, without any (tokens x nodes) tensor.

if HAVE_TRITON:

    @triton.jit
    def _dcoef(x, ACT: tl.constexpr):
        if ACT == 0:
            return 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476)) + x * tl.exp(-0.5 * x * x) * 0.3989422804014327
        return tl.full(x.shape, 1.0, tl.float32)

    @triton.jit
    def _path_bwd(DOUT, WI, WO, NODES, LOGITS, DL, DX, n_tok, d, s_do, s_dx,
                  P: tl.constexpr, ACT: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):
        tok = tl.program_id(0) * BM + tl.arange(0, BM)
        ok = tok < n_tok
        cols = tl.arange(0, BD)
        # the logit gradient of every path node (independent of each other: the path is known)
        for j in range(P):
            node = tl.load(NODES + tok * P + j, mask=ok, other=0)
            g = tl.zeros((BM,), dtype=tl.float32)
            for c in range(0, d, BD):
                cm = (c + cols) < d
                dv = tl.load(DOUT + tok[:, None] * s_do + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                wv = tl.load(WO + node[:, None] * d + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                g += tl.sum(dv.to(tl.float32) * wv.to(tl.float32), axis=1)
            lg = tl.load(LOGITS + tok * P + j, mask=ok, other=0.0)
            tl.store(DL + tok * P + j, _dcoef(lg, ACT) * g, mask=ok)
        # the input gradient
        for c in range(0, d, BD):
            cm = (c + cols) < d
            acc = tl.zeros((BM, BD), dtype=tl.float32)
            for j in range(P):
                node = tl.load(NODES + tok * P + j, mask=ok, other=0)
                dl = tl.load(DL + tok * P + j, mask=ok, other=0.0)
                wv = tl.load(WI + node[:, None] * d + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                acc += dl[:, None] * wv.to(tl.float32)
            tl.store(DX + tok[:, None] * s_dx + (c + cols)[None, :], acc, mask=ok[:, None] & cm[None, :])

    @triton.jit
    def _seg_rows(SRC, s_src, NODE, TOK, VAL, DW, DB, n_ent, d,
                  HAS_DB: tl.constexpr, BE: tl.constexpr, BD: tl.constexpr):
        """DW[node] += VAL * SRC[tok] over entries sorted by node: the block's first and last node runs are summed
        in registers (one atomic row each); any node strictly inside the block gets per-entry atomics."""
        e = tl.program_id(0) * BE + tl.arange(0, BE)
        ok = e < n_ent
        cols = tl.program_id(1) * BD + tl.arange(0, BD)
        cm = cols < d
        node = tl.load(NODE + e, mask=ok, other=0)
        tok = tl.load(TOK + e, mask=ok, other=0)
        val = tl.load(VAL + e, mask=ok, other=0.0)
        rows = tl.load(SRC + tok[:, None] * s_src + cols[None, :], mask=ok[:, None] & cm[None, :], other=0.0).to(tl.float32)
        rows = rows * val[:, None]
        n0 = tl.min(tl.where(ok, node, 2147483647), axis=0)
        n1 = tl.max(tl.where(ok, node, -1), axis=0)
        first = ok & (node == n0)
        last = ok & (node == n1) & (n1 != n0)
        mid = ok & (node != n0) & (node != n1)
        tl.atomic_add(DW + n0 * d + cols, tl.sum(tl.where(first[:, None], rows, 0.0), axis=0), mask=cm)
        tl.atomic_add(DW + n1 * d + cols, tl.sum(tl.where(last[:, None], rows, 0.0), axis=0), mask=cm & (n1 != n0))
        tl.atomic_add(DW + node[:, None] * d + cols[None, :], rows, mask=mid[:, None] & cm[None, :])
        if HAS_DB:
            if tl.program_id(1) == 0:
                tl.atomic_add(DB + n0, tl.sum(tl.where(first, val, 0.0), axis=0))
                tl.atomic_add(DB + n1, tl.sum(tl.where(last, val, 0.0), axis=0), mask=n1 != n0)
                tl.atomic_add(DB + node, val, mask=mid)


def _wgrad(src, nodes_sorted, tok_sorted, val_sorted, rows, d, want_bias, block_e=64, block_d=128):
    dw = torch.zeros(rows, d, device=src.device, dtype=torch.float32)
    db = torch.zeros(rows, device=src.device, dtype=torch.float32) if want_bias else dw
    n_ent = nodes_sorted.numel()
    _seg_rows[(triton.cdiv(n_ent, block_e), triton.cdiv(d, block_d))](
        src, src.stride(0), nodes_sorted, tok_sorted, val_sorted, dw, db, n_ent, d,
        HAS_DB=want_bias, BE=block_e, BD=block_d)
    return dw, (db if want_bias else None)


def _path_bwd_impl(dout, x, w_in, w_out, nodes, logits, act, rows, has_bias, kw, want_x, want_w_in, want_w_out):
    code = {"gelu": 0, "split": 0, "linear": 1}[act] if isinstance(act, str) else act
    dt = x.dtype
    wi, wo = w_in.to(dt).contiguous(), w_out.to(dt).contiguous()
    n_tok, d = x.shape
    P = nodes.shape[1] * nodes.shape[2]
    dout = dout.to(dt).contiguous()
    dl = torch.empty(n_tok, P, device=x.device, dtype=torch.float32)
    dx = torch.empty(n_tok, d, device=x.device, dtype=torch.float32)
    bm = kw.get("block_m", 8)
    _path_bwd[(triton.cdiv(n_tok, bm),)](dout, wi, wo, nodes, logits, dl, dx, n_tok, d, dout.stride(0), dx.stride(0),
                                         P=P, ACT=code, BM=bm, BD=kw.get("block_d", 128), num_warps=kw.get("num_warps", 2))
    dw_in = db = dw_out = None
    if want_w_in or want_w_out:
        sorted_nodes, perm = torch.sort(nodes.reshape(-1))
        tok = (perm // P).to(torch.int32)
        if want_w_out:
            coef = torch.nn.functional.gelu(logits) if code == 0 else logits
            dw_out, _ = _wgrad(dout, sorted_nodes, tok, coef.reshape(-1)[perm].contiguous(), rows, d, False)
        if want_w_in:
            dw_in, db = _wgrad(x, sorted_nodes, tok, dl.reshape(-1)[perm].contiguous(), rows, d, has_bias)
    return dx, dw_in, db, dw_out


class SparsePathRoute(torch.autograd.Function):
    """out = the deep trees' output, sparse forward and backward, with the plain FFF routing gradient (no gradient
    through the hard branches): GTS._forward_route_ste with route_ste=False, as a sparse computation. (Eager use;
    under torch.compile, ``sparse_path_route`` goes through the custom ops below instead.)"""

    @staticmethod
    def forward(ctx, x, w_in, bias, w_out, n_trees, n_nodes, depth, act, kw):
        dt = x.dtype
        out, nodes, logits = sparse_route_fwd(x, w_in.to(dt).contiguous(), bias, w_out.to(dt).contiguous(), n_trees,
                                              n_nodes, depth, act, **kw)
        ctx.save_for_backward(x, w_in, w_out, nodes, logits)
        ctx.cfg = (act, bias, kw)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w_in, w_out, nodes, logits = ctx.saved_tensors
        act, bias, kw = ctx.cfg
        dx, dw_in, db, dw_out = _path_bwd_impl(dout, x, w_in, w_out, nodes, logits, act, w_in.shape[0], bias is not None,
                                               kw, True, ctx.needs_input_grad[1] or ctx.needs_input_grad[2],
                                               ctx.needs_input_grad[3])
        return (dx.to(x.dtype), dw_in.to(w_in.dtype) if dw_in is not None else None,
                db.to(bias.dtype) if db is not None else None, dw_out.to(w_out.dtype) if dw_out is not None else None,
                None, None, None, None, None)


# The same as torch.library custom ops, so that torch.compile treats the kernels as opaque calls (tracing into raw
# Triton launches inside an autograd.Function gave wrong values).
@torch.library.custom_op("gts_sparse::path_fwd", mutates_args=())
def _path_fwd_op(x: torch.Tensor, w_in: torch.Tensor, bias: torch.Tensor | None, w_out: torch.Tensor, n_trees: int,
                 n_nodes: int, depth: int, act: int, block_m: int, block_d: int, num_warps: int
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dt = x.dtype
    out, nodes, logits = sparse_route_fwd(x, w_in.to(dt).contiguous(), bias, w_out.to(dt).contiguous(), n_trees, n_nodes,
                                          depth, "gelu" if act == 0 else "linear", block_m=block_m, block_d=block_d,
                                          num_warps=num_warps)
    return out, nodes, logits


@_path_fwd_op.register_fake
def _(x, w_in, bias, w_out, n_trees, n_nodes, depth, act, block_m, block_d, num_warps):
    n = x.shape[0]
    return (x.new_empty(x.shape, dtype=torch.float32), x.new_empty((n, n_trees, depth + 1), dtype=torch.int32),
            x.new_empty((n, n_trees, depth + 1), dtype=torch.float32))


@torch.library.custom_op("gts_sparse::path_bwd", mutates_args=())
def _path_bwd_op(dout: torch.Tensor, x: torch.Tensor, w_in: torch.Tensor, w_out: torch.Tensor, nodes: torch.Tensor,
                 logits: torch.Tensor, has_bias: bool, act: int, block_m: int, block_d: int, num_warps: int
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kw = dict(block_m=block_m, block_d=block_d, num_warps=num_warps)
    dx, dw_in, db, dw_out = _path_bwd_impl(dout, x, w_in, w_out, nodes, logits, act, w_in.shape[0], has_bias, kw,
                                           True, True, True)
    db = db if db is not None else dx.new_zeros(0)
    return dx.to(x.dtype), dw_in.to(w_in.dtype), db, dw_out.to(w_out.dtype)


@_path_bwd_op.register_fake
def _(dout, x, w_in, w_out, nodes, logits, has_bias, act, block_m, block_d, num_warps):
    return (torch.empty_like(x), torch.empty_like(w_in),
            x.new_empty((w_in.shape[0],) if has_bias else (0,), dtype=torch.float32), torch.empty_like(w_out))


def _path_setup(ctx, inputs, output):
    x, w_in, bias, w_out, n_trees, n_nodes, depth, act, block_m, block_d, num_warps = inputs
    _, nodes, logits = output
    ctx.save_for_backward(x, w_in, w_out, nodes, logits)
    ctx.cfg = (bias is not None, bias.dtype if bias is not None else None, act, block_m, block_d, num_warps)


def _path_backward(ctx, dout, _dnodes, _dlogits):
    x, w_in, w_out, nodes, logits = ctx.saved_tensors
    has_bias, b_dt, act, block_m, block_d, num_warps = ctx.cfg
    dx, dw_in, db, dw_out = _path_bwd_op(dout, x, w_in, w_out, nodes, logits, has_bias, act, block_m, block_d, num_warps)
    return dx, dw_in, (db.to(b_dt) if has_bias else None), dw_out, None, None, None, None, None, None, None


_path_fwd_op.register_autograd(_path_backward, setup_context=_path_setup)


def sparse_path_route(x, w_in, bias, w_out, n_trees, n_nodes, depth, act="gelu", **kw):
    """Differentiable sparse deep trees with the plain FFF routing gradient (route_ste=False). x: (tokens, d); returns
    (tokens, d) float32. ``kw``: tile sizes for the kernels (block_m, block_d, num_warps)."""
    kw = {"block_m": 8, "block_d": 128, "num_warps": 2, **{k: v for k, v in kw.items() if k in ("block_m", "block_d", "num_warps")}}
    code = {"gelu": 0, "split": 0, "linear": 1}[act]
    out, _, _ = _path_fwd_op(x, w_in, bias, w_out, n_trees, n_nodes, depth, code, kw["block_m"], kw["block_d"], kw["num_warps"])
    return out
