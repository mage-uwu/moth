# Golden Tree Snake (GTS) fork, 2026.
"""GTS-Uni-AR: the causal mixed-forest LM (scripts/shakespeare_ar.TinyLM) run several times with shared weights."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import shakespeare_ar as S  # noqa: E402


def _build(loops):
    a = argparse.Namespace(d_model=64, ternary_group=128, gts_layers=2, gts_depth=3, gts_trees=1, gts_heads=1, gts_act="gelu",
                           gts_state=16, gts_read=False, gts_write_key=False, gts_act_bits=8, gts_route_ste=True,
                           gts_route_temp=1.0, bank_trees=8, bank_heads=8, bank_state=16, loops=loops)
    return S.build("mixed", 300, a)


def test_fresh_looped_ar_equals_its_one_pass_weights():
    torch.manual_seed(0)
    plain = _build(1).eval()
    uni = _build(3).eval()
    missing, unexpected = uni.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and set(missing) == {"loop_embed", "loop_gate"}
    ids = torch.randint(0, 300, (2, 16))
    assert torch.equal(plain(ids), uni(ids))


def test_looped_ar_stays_causal_and_passes_can_be_lowered():
    """Changing a later token must not change earlier logits, at every pass count."""
    torch.manual_seed(0)
    uni = _build(3).eval()
    with torch.no_grad():
        uni.loop_gate.fill_(0.5)
        uni.loop_embed.normal_(0, 0.1)
    ids = torch.randint(0, 300, (1, 16))
    ids2 = ids.clone()
    ids2[0, 10] = (ids2[0, 10] + 1) % 300
    for n in (1, 2, 3):
        a, b = uni(ids, loops=n), uni(ids2, loops=n)
        assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
        assert not torch.allclose(a[:, 10:], b[:, 10:])
    assert not torch.allclose(uni(ids, loops=1), uni(ids, loops=3))


def test_checkpointed_ar_passes_give_the_same_gradients():
    torch.manual_seed(0)
    uni = _build(2)
    with torch.no_grad():
        uni.loop_gate.fill_(0.3)
    ids = torch.randint(0, 300, (2, 12))
    grads = []
    for ck in (False, True):
        uni.zero_grad()
        uni.hidden(ids, checkpoint_loops=ck, checkpoint_layers=ck).square().mean().backward()
        grads.append(uni.loop_gate.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def _build_pause(loops, every=4, m=2):
    a = argparse.Namespace(d_model=64, ternary_group=128, gts_layers=2, gts_depth=3, gts_trees=1, gts_heads=1, gts_act="gelu",
                           gts_state=16, gts_read=False, gts_write_key=False, gts_act_bits=8, gts_route_ste=True,
                           gts_route_temp=1.0, bank_trees=8, bank_heads=8, bank_state=16, loops=loops, pause_every=every,
                           pause_tokens=m)
    return S.build("mixed", 300, a)


def test_pause_tokens_start_inert_and_keep_causality():
    torch.manual_seed(0)
    plain = _build(1).eval()
    uni = _build_pause(3).eval()
    missing, unexpected = uni.load_state_dict(plain.state_dict(), strict=False)
    assert set(missing) == {"loop_embed", "loop_gate", "pause"}
    ids = torch.randint(0, 300, (2, 14))  # not a multiple of pause_every: padded internally
    assert torch.equal(plain(ids), uni(ids))
    with torch.no_grad():
        uni.loop_gate.fill_(0.5)
    ids2 = ids.clone()
    ids2[:, 9] = (ids2[:, 9] + 1) % 300
    a, b = uni(ids), uni(ids2)
    assert torch.allclose(a[:, :9], b[:, :9], atol=1e-5) and not torch.allclose(a[:, 9:], b[:, 9:])
    with torch.no_grad():  # the pause vectors matter once the gates are open
        before = uni(ids)
        uni.pause.add_(1.0)
        assert not torch.allclose(before[:, 4:], uni(ids)[:, 4:])
