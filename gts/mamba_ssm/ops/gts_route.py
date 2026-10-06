# Golden Tree Snake (GTS) fork, 2026.
"""Stateless deep trees with the straight-through routing gradient, without forming the path weights.

``GTS._forward_ste`` computes ``out = (pi * coef(L)) @ W_out`` over every node, where pi[m] is the product of the
branch values above node m: the hard step ``L > 0`` going forward, ``sigmoid(L / temp)`` going backward. That builds
pi level by level with stack and cat and evaluates the activation, the sigmoid and the products over all
(tokens x nodes), forward and backward. Two facts make almost all of it unnecessary:

* Going forward pi is exactly the one-hot of the token's path, so ``out = A @ W_out`` with A holding the path nodes'
  coefficients and zeros elsewhere.
* Going backward, with g = dOut @ W_out^T (one value per node), the only nonzero logit gradients are at the path
  nodes. At path node a on level k:

      dL[a] = coef'(L[a]) g[a] + sign * (on[a] - alt[a]) * sigmoid'(L[a] / temp) / temp

  where on[a] sums coef * g over the path below a, alt[a] sums it over the chain that starts at a's other child and
  follows the token's own decisions to a leaf, and sign is +1 if the token went right at a. A depth-10 tree needs
  11 path nodes and 55 chain nodes per token, not 2,047.

Two Triton kernels walk the trees in registers, a block of tokens per program and one tree per program column:
_route_fwd writes A (and the path, if asked for), _route_bwd writes dL. The matmuls are PyTorch's.
Only stateless trees (no context) are covered; ``GTS`` falls back to the dense form otherwise.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

__all__ = ["route_ste_out", "HAVE_TRITON"]
HAVE_TRITON = triton is not None

if HAVE_TRITON:

    @triton.jit
    def _coef(x, ACT: tl.constexpr):
        """ACT 0: gelu (also split with no context), 1: linear."""
        if ACT == 0:
            return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
        return x

    @triton.jit
    def _dcoef(x, ACT: tl.constexpr):
        if ACT == 0:
            return 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476)) + x * tl.exp(-0.5 * x * x) * 0.3989422804014327
        return tl.full(x.shape, 1.0, tl.float32)

    @triton.jit
    def _route_fwd(Lp, Ap, NODESp, n_tok, s_l, s_a, n_trees,
                   N_NODES: tl.constexpr, DEPTH: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr, STORE_NODES: tl.constexpr):
        tok = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tree = tl.program_id(1)
        ok = tok < n_tok
        base = tree * N_NODES
        cur = tl.zeros((BLOCK,), dtype=tl.int32)
        for k in tl.static_range(DEPTH + 1):
            node = base + cur
            lg = tl.load(Lp + tok * s_l + node, mask=ok, other=0.0).to(tl.float32)
            tl.store(Ap + tok * s_a + node, _coef(lg, ACT), mask=ok)
            if STORE_NODES:
                tl.store(NODESp + (tok * n_trees + tree) * (DEPTH + 1) + k, node, mask=ok)
            cur = 2 * cur + 1 + (lg > 0).to(tl.int32)

    @triton.jit
    def _route_bwd(Lp, Gp, DLp, n_tok, s_l, s_g, s_d, inv_temp,
                   N_NODES: tl.constexpr, DEPTH: tl.constexpr, ACT: tl.constexpr, BLOCK: tl.constexpr):
        tok = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tree = tl.program_id(1)
        ok = tok < n_tok
        base = tree * N_NODES
        # pass 1: sum of coef * g over the whole path
        cur = tl.zeros((BLOCK,), dtype=tl.int32)
        total = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(DEPTH + 1):
            lg = tl.load(Lp + tok * s_l + base + cur, mask=ok, other=0.0).to(tl.float32)
            total += _coef(lg, ACT) * tl.load(Gp + tok * s_g + base + cur, mask=ok, other=0.0).to(tl.float32)
            cur = 2 * cur + 1 + (lg > 0).to(tl.int32)
        # pass 2: each path node's gradient
        cur = tl.zeros((BLOCK,), dtype=tl.int32)
        done = tl.zeros((BLOCK,), dtype=tl.float32)  # coef * g summed over the path down to this node
        for k in tl.static_range(DEPTH + 1):
            lg = tl.load(Lp + tok * s_l + base + cur, mask=ok, other=0.0).to(tl.float32)
            gk = tl.load(Gp + tok * s_g + base + cur, mask=ok, other=0.0).to(tl.float32)
            done += _coef(lg, ACT) * gk
            d = _dcoef(lg, ACT) * gk
            right = lg > 0
            if k < DEPTH:
                on = total - done  # the path below this node
                s = 2 * cur + 2 - right.to(tl.int32)  # the other child
                alt = tl.zeros((BLOCK,), dtype=tl.float32)
                for j in tl.static_range(DEPTH - k):
                    ls = tl.load(Lp + tok * s_l + base + s, mask=ok, other=0.0).to(tl.float32)
                    alt += _coef(ls, ACT) * tl.load(Gp + tok * s_g + base + s, mask=ok, other=0.0).to(tl.float32)
                    s = 2 * s + 1 + (ls > 0).to(tl.int32)
                p = 1.0 / (1.0 + tl.exp(-lg * inv_temp))
                d += tl.where(right, on - alt, alt - on) * p * (1.0 - p) * inv_temp
            tl.store(DLp + tok * s_d + base + cur, d, mask=ok)
            cur = 2 * cur + 1 + right.to(tl.int32)


def _grid(n_tok, n_trees, block):
    return (triton.cdiv(n_tok, block), n_trees)


class _RouteSTE(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, L, W, n_trees, n_nodes, depth, act, temp, want_nodes):
        n_tok = L.shape[0]
        L = L.contiguous()
        A = torch.zeros_like(L)
        nodes = torch.empty(n_tok, n_trees, depth + 1, device=L.device, dtype=torch.int32) if want_nodes else A
        block = 128
        _route_fwd[_grid(n_tok, n_trees, block)](L, A, nodes, n_tok, L.stride(0), A.stride(0), n_trees,
                                                 N_NODES=n_nodes, DEPTH=depth, ACT=act, BLOCK=block, STORE_NODES=want_nodes)
        ctx.save_for_backward(L, W)
        ctx.cfg = n_trees, n_nodes, depth, act, temp
        out = A @ W
        if want_nodes:
            ctx.mark_non_differentiable(nodes)
        return out, (nodes if want_nodes else None)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout, _dnodes):
        L, W = ctx.saved_tensors
        n_trees, n_nodes, depth, act, temp = ctx.cfg
        n_tok = L.shape[0]
        dout = dout.contiguous()
        block = 128
        dW = None
        if ctx.needs_input_grad[1]:
            A = torch.zeros_like(L)
            _route_fwd[_grid(n_tok, n_trees, block)](L, A, A, n_tok, L.stride(0), A.stride(0), n_trees,
                                                     N_NODES=n_nodes, DEPTH=depth, ACT=act, BLOCK=block, STORE_NODES=False)
            dW = A.t() @ dout
        dL = None
        if ctx.needs_input_grad[0]:
            G = dout @ W.t()
            dL = torch.zeros_like(L)
            _route_bwd[_grid(n_tok, n_trees, block)](L, G, dL, n_tok, L.stride(0), G.stride(0), dL.stride(0), 1.0 / temp,
                                                     N_NODES=n_nodes, DEPTH=depth, ACT=act, BLOCK=block)
        return dL, dW, None, None, None, None, None, None


def route_ste_out(L, W, n_trees, depth, act="gelu", temp=1.0, want_nodes=False, n_nodes=None):
    """L: (tokens, trees * nodes) every node's logit, in any float dtype (the kernels work in float32; under bf16
    autocast the (tokens x nodes) buffers stay bf16 and the matmuls run in bf16); W: (trees * nodes, d) output rows. Returns (out, nodes): out is
    (tokens, d), the stateless trees' output with the straight-through routing gradient; nodes is (tokens, trees,
    depth + 1) int32 global node ids along each path if ``want_nodes``, else None. ``act`` is "gelu", "split"
    (the same with no context) or "linear"."""
    if not HAVE_TRITON:
        raise RuntimeError("route_ste_out needs triton")
    code = {"gelu": 0, "split": 0, "linear": 1}[act]
    n_nodes = n_nodes or L.shape[1] // n_trees  # L and W may carry padding columns/rows after the trees
    return _RouteSTE.apply(L, W, n_trees, n_nodes, depth, code, float(temp), want_nodes)
