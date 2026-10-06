# Golden Tree Snake (GTS) fork, 2026.
"""The fused Triton absmean quantiser against the PyTorch form (Triton's interpreter without a GPU)."""
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

triton = pytest.importorskip("triton")

from mamba_ssm.modules.ternary import _codes_and_scales  # noqa: E402
from mamba_ssm.ops.ternary_fused import absmean_ternary_fused  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("shape,g", [((300, 1024), 128), ((7, 120), 120), ((33, 96), 32), ((5, 4, 256), 128)])
def test_fused_absmean_matches_pytorch(shape, g):
    torch.manual_seed(0)
    w = torch.randn(*shape, device=DEV, requires_grad=True)
    codes, scale = _codes_and_scales(w.detach(), g, 1e-8)
    ref = (codes * scale).reshape(w.shape)
    out = absmean_ternary_fused(w, g)
    assert torch.equal(out == 0, ref == 0) and torch.equal(out > 0, ref > 0)  # same codes
    assert torch.allclose(out, ref, rtol=1e-6, atol=0)
    gr = torch.randn_like(out)
    assert torch.equal(torch.autograd.grad((out * gr).sum(), w)[0], gr)  # straight through


def test_halves_round_to_even():
    w = torch.tensor([[1.0, 3.0] * 64], device=DEV)  # scale 2: 0.5 -> 0, 1.5 -> clamp 1
    assert absmean_ternary_fused(w, 128)[0, :2].tolist() == [0.0, 2.0]
