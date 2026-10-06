# Golden Tree Snake (GTS) fork, 2026.
import torch

from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM
from mamba_ssm.models.gts_vision import GTSVision, GTSVisionConfig, VisionSidecar, lm_with_image


def _tiny():
    return GTSVisionConfig(image_size=64, d_model=64, n_layer=2, bank_trees=8, bank_heads=8, deep_trees=1, deep_depth=3)


def test_default_size_is_38m():
    n = sum(p.numel() for p in GTSVision(GTSVisionConfig()).parameters())
    assert 37.5e6 < n < 38.5e6


def test_shapes_and_other_resolutions():
    torch.manual_seed(0)
    m = GTSVision(_tiny()).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 16, 64)
    assert m(torch.randn(1, 3, 96, 96)).shape == (1, 36, 64)  # position grid interpolated
    assert m.features(torch.randn(2, 3, 64, 64)).shape == (2, 64)


def test_column_layers_see_the_grid_transposed():
    """An odd layer reads the grid in column order and writes back in raster order: permuting the image's rows and
    columns together (a transpose) commutes with a two-layer model only through that reordering, so check the
    reordering itself."""
    m = GTSVision(_tiny())
    n = m.side
    raster = torch.arange(n * n)
    assert torch.equal(raster[m._col][m._col_inv], raster)
    assert torch.equal(m._col.view(n, n), raster.view(n, n).t())


def test_sidecar_feeds_the_lm_and_gradients_reach_the_backbone():
    torch.manual_seed(0)
    v = GTSVision(_tiny())
    lm_cfg = GTSConfig(d_model=64, n_layer=1, vocab_size=200, mixer="mixed", bank_trees=8, bank_heads=8, deep_trees=1,
                       deep_depth=3, causal=False, ternary=True, act_bits=8)
    lm = GTSForMaskedLM(lm_cfg)
    for p in lm.parameters():
        p.requires_grad_(False)
    side = VisionSidecar(64, 64, n_tokens=4)
    vecs = side(v(torch.randn(2, 3, 64, 64)))
    assert vecs.shape == (2, 4, 64)
    ids = torch.full((2, 10), 7)
    ids[:, 0] = 101
    h = lm_with_image(lm, vecs, ids)
    assert h.shape == (2, 10, 64)
    h.sum().backward()
    assert v.stem[0].weight.grad is not None and v.stem[0].weight.grad.abs().sum() > 0
