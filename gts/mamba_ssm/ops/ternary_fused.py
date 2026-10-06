# Golden Tree Snake (GTS) fork, 2026.
"""absmean_ternary (modules/ternary.py) as one Triton kernel: one read of the latent weights, one write of
scale * code, straight-through backward with no copies. Under CUDA autocast it writes the autocast dtype, since every
use of a ternary weight is a matmul that would cast it anyway. Same values as the PyTorch form."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

HAVE_TRITON = triton is not None

if HAVE_TRITON:

    @triton.jit
    def _absmean_kernel(Wp, Op, n_groups, eps, G: tl.constexpr, BLOCK_G: tl.constexpr, ROWS: tl.constexpr):
        r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        j = tl.arange(0, BLOCK_G)
        ok = (r < n_groups)[:, None] & (j < G)[None, :]
        w = tl.load(Wp + r[:, None] * G + j[None, :], mask=ok, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.sum(tl.abs(w), axis=1) / G, eps)[:, None]
        q = w / scale
        q = tl.minimum(tl.maximum(q, -1.0), 1.0)
        code = tl.where(q > 0.5, 1.0, tl.where(q < -0.5, -1.0, 0.0))  # torch.round: halves go to even, so 0.5 -> 0
        tl.store(Op + r[:, None] * G + j[None, :], code * scale, mask=ok)


class _AbsmeanSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, g, eps, out_dtype):
        wc = w.contiguous()
        out = torch.empty(w.shape, device=w.device, dtype=out_dtype)
        n_groups = wc.numel() // g
        rows = 16
        _absmean_kernel[(triton.cdiv(n_groups, rows),)](wc, out, n_groups, eps, G=g, BLOCK_G=triton.next_power_of_2(g), ROWS=rows)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


def absmean_ternary_fused(w, g, eps=1e-8):
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else w.dtype
    return _AbsmeanSTE.apply(w, g, eps, dtype)
