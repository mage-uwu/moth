# Golden Tree Snake (GTS) fork, 2026.
"""Binarized checkpoints: 2-bit codes and scales reload into a model computing the same function."""
import torch

from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM
from mamba_ssm.utils.ternary_pack import _pack2, _unpack2, load_binarized, save_binarized


def test_pack2_roundtrip():
    codes = torch.randint(-1, 2, (7, 33), dtype=torch.int8)
    assert torch.equal(_unpack2(_pack2(codes), codes.shape), codes)


def test_binarized_checkpoint_same_function(tmp_path):
    torch.manual_seed(0)
    cfg = dict(d_model=32, n_layer=2, vocab_size=60, mixer="mixed", bank_trees=8, bank_heads=4, bank_state=8,
               deep_trees=2, deep_depth=3, ternary=True, act_bits=8, route_ste=True, causal=False)
    a = GTSForMaskedLM(GTSConfig(**cfg))
    with torch.no_grad():
        for p in a.parameters():
            p.add_(torch.randn_like(p) * 0.05)  # move away from the initialisation
    save_binarized(a, cfg, tmp_path / "b.pt")
    b = load_binarized(str(tmp_path / "b.pt"), GTSForMaskedLM(GTSConfig(**cfg)))
    ids = torch.randint(1, 60, (2, 25))
    assert torch.allclose(a(ids).logits, b(ids).logits, atol=1e-4)
    size = (tmp_path / "b.pt").stat().st_size
    assert size < sum(p.numel() for p in a.parameters()) * 4  # smaller than the float model
