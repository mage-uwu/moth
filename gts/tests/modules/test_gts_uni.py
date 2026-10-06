# Golden Tree Snake (GTS) fork, 2026.
"""GTS-Uni: the encoder stack run several times with the same weights, with latent scratch tokens."""
import torch

from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM


def _cfg(**kw):
    base = dict(d_model=64, n_layer=2, vocab_size=300, mixer="mixed", bank_trees=8, bank_heads=8, deep_trees=1,
                deep_depth=3, causal=False, ternary=True, act_bits=8)
    base.update(kw)
    return GTSConfig(**base)


def test_fresh_uni_equals_its_one_pass_weights():
    """With the gates at zero, extra passes and latent tokens change nothing: a GTS-Uni built from a trained
    one-pass model starts out computing exactly what it did."""
    torch.manual_seed(0)
    plain = GTSForMaskedLM(_cfg()).eval()
    uni = GTSForMaskedLM(_cfg(loops=3, latent_tokens=4)).eval()
    missing, unexpected = uni.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and set(missing) == {"backbone.loop_embed", "backbone.loop_gate", "backbone.latents"}
    ids = torch.randint(1, 300, (2, 20))
    assert torch.equal(plain(ids).logits, uni(ids).logits)


def test_passes_change_the_output_once_gated_and_can_be_lowered():
    torch.manual_seed(0)
    uni = GTSForMaskedLM(_cfg(loops=3, latent_tokens=4)).eval()
    with torch.no_grad():
        uni.backbone.loop_gate.fill_(0.5)
    ids = torch.randint(1, 300, (2, 20))
    out = [uni(ids, loops=n).logits for n in (1, 2, 3)]
    assert all(o.shape == (2, 20, 300) for o in out)
    assert not torch.allclose(out[0], out[1]) and not torch.allclose(out[1], out[2])


def test_gradients_reach_gates_latents_and_shared_weights():
    torch.manual_seed(0)
    uni = GTSForMaskedLM(_cfg(loops=3, latent_tokens=4))
    with torch.no_grad():
        uni.backbone.loop_gate.fill_(0.1)
    ids = torch.randint(1, 300, (2, 20))
    labels = torch.full_like(ids, -100)
    labels[:, 5] = ids[:, 5]
    uni(ids, labels=labels).loss.backward()
    bb = uni.backbone
    for p in (bb.loop_gate, bb.loop_embed, bb.latents, bb.layers[0].norm.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_checkpointed_passes_give_the_same_gradients():
    torch.manual_seed(0)
    uni = GTSForMaskedLM(_cfg(loops=2, latent_tokens=2))
    with torch.no_grad():
        uni.backbone.loop_gate.fill_(0.3)
    ids = torch.randint(1, 300, (2, 16))
    grads = []
    for ck in (False, True):
        uni.zero_grad()
        uni.backbone.checkpoint_loops = ck
        uni(ids).logits.square().mean().backward()
        grads.append(uni.backbone.loop_gate.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
