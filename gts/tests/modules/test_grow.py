# Golden Tree Snake (GTS) fork, 2026.
import torch

from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM
from mamba_ssm.utils.grow import grow_deep_trees


def _cfg(**kw):
    c = dict(d_model=64, n_layer=2, vocab_size=300, mixer="mixed", bank_trees=8, bank_heads=8, deep_trees=2,
             deep_depth=3, causal=False, ternary=True, act_bits=8, route_ste=True)
    c.update(kw)
    return c


def test_grown_trees_compute_the_same_function():
    for extra in ({}, {"loops": 3, "latent_tokens": 2}):
        torch.manual_seed(0)
        cfg = _cfg(**extra)
        m = GTSForMaskedLM(GTSConfig(**cfg)).eval()
        if "loops" in extra:
            with torch.no_grad():
                m.backbone.loop_gate.fill_(0.3)
        state, cfg2 = grow_deep_trees(m.state_dict(), cfg, add_levels=2)
        assert cfg2["deep_depth"] == 5
        g = GTSForMaskedLM(GTSConfig(**cfg2)).eval()
        g.load_state_dict(state)
        ids = torch.randint(1, 300, (2, 12))
        assert torch.equal(m(ids).logits, g(ids).logits)
        n_old = sum(p.numel() for p in m.parameters())
        n_new = sum(p.numel() for p in g.parameters())
        assert n_new > n_old


def test_new_leaves_learn():
    torch.manual_seed(0)
    cfg = _cfg()
    m = GTSForMaskedLM(GTSConfig(**cfg))
    state, cfg2 = grow_deep_trees(m.state_dict(), cfg, add_levels=1)
    g = GTSForMaskedLM(GTSConfig(**cfg2))
    g.load_state_dict(state)
    ids = torch.randint(1, 300, (2, 12))
    labels = ids.clone()
    g(ids, labels=labels).loss.backward()
    deep = g.backbone.layers[0].mixer.deep
    n_old = 2 ** (cfg["deep_depth"] + 1) - 1
    grad = deep.node_out.grad.view(cfg["deep_trees"], -1, cfg["d_model"])[:, n_old:]
    assert grad.abs().sum() > 0  # the new (zero) output rows get gradients
