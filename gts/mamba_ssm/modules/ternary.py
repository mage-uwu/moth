# Golden Tree Snake (GTS) fork, 2026.
"""Ternary quantisation-aware training primitives.

The weight quantiser is the TernaryLinear recipe from "Ternary Mamba: Grouped Quantization-Aware
Training of W1.58A16 State Space Models" (arXiv:2606.18114), which is BitNet b1.58's absmean
quantiser applied per group of weights:

    scale_g = mean(|w_i| for i in group g)             recomputed on every forward, never learned
    code_i  = round(clip(w_i / scale_g, -1, 1))        in {-1, 0, +1}
    w_hat_i = scale_g * code_i

Gradients pass straight through to the latent full-precision weights (STE). The scale is
deliberately not an nn.Parameter: that paper reports that a learnable scale drives the share of
zero codes towards 90% ("zero-ratio collapse"), while the recomputed absmean self-regulates.

Two additions from the BitNet line of work:

* ``quantize_activations`` - per-token absmax integer activations with STE (BitNet b1.58 uses 8 bits).
* ``lam`` - a blend between the latent and the quantised value, so quantisation can be warmed in:
  ``w + lam * (w_hat - w)``. ``lam=0`` is full precision, ``lam=1`` is exactly ternary.
"""

import torch

__all__ = ["absmean_ternary", "quantize_activations", "pack_ternary", "unpack_ternary", "zero_ratio"]


def _group_size(n, group_size):
    """Largest group size <= group_size that divides n (n itself if group_size is None or too large)."""
    if group_size is None or group_size >= n:
        return n
    g = int(group_size)
    while n % g:
        g -= 1
    return g


def _codes_and_scales(w, group_size, eps):
    n = w.shape[-1]
    g = _group_size(n, group_size)
    wg = w.reshape(*w.shape[:-1], n // g, g)
    scale = wg.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    codes = (wg / scale).clamp(-1, 1).round()
    return codes, scale


_FUSED = True  # set False to use the PyTorch form on CUDA as well


def absmean_ternary(w, group_size=None, lam=1.0, eps=1e-8):
    """Grouped absmean ternarisation with a straight-through estimator.

    w: (..., n). Groups are consecutive runs of ``group_size`` entries along the last dimension,
    so every row of a weight table gets ``n / group_size`` scales. ``group_size=None`` is one scale per row.
    """
    if lam <= 0:
        return w
    if lam >= 1 and w.is_cuda and _FUSED:
        from mamba_ssm.ops.ternary_fused import HAVE_TRITON, absmean_ternary_fused

        if HAVE_TRITON:  # one Triton kernel: one read, one write, same values (scales to the last bit or one ulp)
            return absmean_ternary_fused(w, _group_size(w.shape[-1], group_size), eps)
    codes, scale = _codes_and_scales(w, group_size, eps)
    wq = (codes * scale).reshape(w.shape)
    if lam >= 1:
        # (w - w.detach()) is exactly zero, so the forward value is exactly wq.
        return wq.detach() + (w - w.detach())
    return w + lam * (wq - w).detach()


def quantize_activations(x, bits=8, lam=1.0, eps=1e-5):
    """Per-token absmax integer quantisation of activations with STE (BitNet b1.58's activation_quant)."""
    if lam <= 0 or bits is None:
        return x
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()  # under bf16 autocast, round in float32 so the codes match the float32 export
    scale = qmax / xf.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    xq = ((xf * scale).round().clamp(-qmax - 1, qmax) / scale).to(x.dtype)
    if lam >= 1:
        return xq.detach() + (x - x.detach())
    return x + lam * (xq - x).detach()


@torch.no_grad()
def pack_ternary(w, group_size=None, eps=1e-8):
    """Export form: int8 codes in {-1, 0, +1} with the shape of w, and float scales (..., n_groups)."""
    codes, scale = _codes_and_scales(w, group_size, eps)
    return codes.reshape(w.shape).to(torch.int8), scale.squeeze(-1)


@torch.no_grad()
def unpack_ternary(codes, scales):
    """Inverse of pack_ternary. Equals absmean_ternary(w, group_size) exactly."""
    n = codes.shape[-1]
    n_groups = scales.shape[-1]
    wg = codes.reshape(*codes.shape[:-1], n_groups, n // n_groups).to(scales.dtype)
    return (wg * scales.unsqueeze(-1)).reshape(codes.shape)


@torch.no_grad()
def zero_ratio(w, group_size=None, eps=1e-8):
    """Share of weights whose ternary code is zero. Healthy absmean training sits around a quarter to a third."""
    codes, _ = _codes_and_scales(w, group_size, eps)
    return (codes == 0).float().mean().item()
