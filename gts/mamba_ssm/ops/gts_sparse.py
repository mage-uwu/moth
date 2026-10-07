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

__all__ = ["sparse_route_fwd", "HAVE_TRITON"]
HAVE_TRITON = triton is not None

if HAVE_TRITON:

    @triton.jit
    def _coef(x, ACT: tl.constexpr):
        """ACT 0: gelu (also split with no context), 1: linear."""
        if ACT == 0:
            return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        return x

    @triton.jit
    def _sparse_fwd(X, WI, BIAS, WO, OUT, NODES, LOGITS, n_tok, d, s_x, s_o,
                    N_TREES: tl.constexpr, N_NODES: tl.constexpr, DEPTH: tl.constexpr, ACT: tl.constexpr,
                    HAS_BIAS: tl.constexpr, ROUND_BF16: tl.constexpr, BM: tl.constexpr, BD: tl.constexpr):
        tok = tl.program_id(0) * BM + tl.arange(0, BM)
        ok = tok < n_tok
        cols = tl.arange(0, BD)
        P: tl.constexpr = N_TREES * (DEPTH + 1)
        # phase 1: walk every tree, keeping the path's node ids and logits
        for t in range(N_TREES):
            cur = tl.zeros((BM,), dtype=tl.int32)
            for k in range(DEPTH + 1):
                node = t * N_NODES + cur
                acc = tl.zeros((BM,), dtype=tl.float32)
                for c in range(0, d, BD):
                    cm = (c + cols) < d
                    xv = tl.load(X + tok[:, None] * s_x + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                    wv = tl.load(WI + node[:, None] * d + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                    acc += tl.sum(xv.to(tl.float32) * wv.to(tl.float32), axis=1)
                if HAS_BIAS:
                    acc += tl.load(BIAS + node, mask=ok, other=0.0).to(tl.float32)
                if ROUND_BF16:  # the dense path's bf16 GEMM hands back bf16 logits: branch on the same value
                    acc = acc.to(tl.bfloat16).to(tl.float32)
                tl.store(NODES + tok * P + t * (DEPTH + 1) + k, node, mask=ok)
                tl.store(LOGITS + tok * P + t * (DEPTH + 1) + k, acc, mask=ok)
                cur = 2 * cur + 1 + (acc > 0).to(tl.int32)
        # phase 2: the output, a chunk of d_model at a time
        for c in range(0, d, BD):
            cm = (c + cols) < d
            out = tl.zeros((BM, BD), dtype=tl.float32)
            for j in range(P):
                node = tl.load(NODES + tok * P + j, mask=ok, other=0)
                cf = _coef(tl.load(LOGITS + tok * P + j, mask=ok, other=0.0), ACT)
                wv = tl.load(WO + node[:, None] * d + (c + cols)[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
                out += cf[:, None] * wv.to(tl.float32)
            tl.store(OUT + tok[:, None] * s_o + (c + cols)[None, :], out, mask=ok[:, None] & cm[None, :])


def sparse_route_fwd(x, w_in, bias, w_out, n_trees, n_nodes, depth, act="gelu", block_m=32, block_d=64, num_warps=4):
    """x: (tokens, d); w_in, w_out: (>= trees * nodes, d) node rows (padding rows after the trees are never read);
    bias: (>= trees * nodes,) or None. Returns (out, nodes, logits): out (tokens, d) float32, nodes (tokens, trees,
    depth + 1) int32 global node ids along each path, logits the same shape in float32. ``act`` is "gelu", "split"
    (the same with no context) or "linear"."""
    if not HAVE_TRITON:
        raise RuntimeError("sparse_route_fwd needs triton")
    code = {"gelu": 0, "split": 0, "linear": 1}[act]
    x = x.contiguous()
    n_tok, d = x.shape
    w_in = w_in.to(x.dtype).contiguous()
    w_out = w_out.to(x.dtype).contiguous()
    out = torch.empty(n_tok, d, device=x.device, dtype=torch.float32)
    nodes = torch.empty(n_tok, n_trees, depth + 1, device=x.device, dtype=torch.int32)
    logits = torch.empty(n_tok, n_trees, depth + 1, device=x.device, dtype=torch.float32)
    b = bias.contiguous() if bias is not None else out
    _sparse_fwd[(triton.cdiv(n_tok, block_m),)](
        x, w_in, b, w_out, out, nodes, logits, n_tok, d, x.stride(0), out.stride(0),
        N_TREES=n_trees, N_NODES=n_nodes, DEPTH=depth, ACT=code, HAS_BIAS=bias is not None,
        ROUND_BF16=x.dtype == torch.bfloat16, BM=block_m, BD=block_d, num_warps=num_warps)
    return out, nodes, logits
