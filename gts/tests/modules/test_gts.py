# Golden Tree Snake (GTS) fork, 2026.
"""CPU tests for the GTS mixer. Run with: pytest tests/modules/test_gts.py"""

import torch
import torch.nn.functional as F

from mamba_ssm.modules.gts import GTS, GTSMixed
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM


def make(seed=0, **kwargs):
    torch.manual_seed(seed)
    cfg = dict(d_model=32, depth=4, d_state=8, dtype=torch.float64)
    cfg.update(kwargs)
    m = GTS(**cfg)
    if m.use_context:
        # Make the context term large enough to matter at initialisation, so the tests exercise it.
        with torch.no_grad():
            m.ctx_proj.weight.mul_(4.0)
            m.dt_bias.add_(3.0)
            m.A_log.sub_(2.0)
    if m.d_conv > 0:
        with torch.no_grad():
            m.conv1d.weight.add_(0.2 * torch.randn_like(m.conv1d.weight))
    return m


def test_dense_forward_matches_lazy_kernel():
    """The differentiable quadratic form and the token-at-a-time lazy-decay kernel are the same function."""
    for kwargs in (dict(), dict(selective=False), dict(n_trees=3, depth=3), dict(ternary=True), dict(ternary=True, ternary_group=8, act_bits=8), dict(d_conv=0),
                   dict(causal=True), dict(causal=True, d_conv=4, n_trees=2, depth=3),
                   dict(read_state=True), dict(causal=True, read_state=True, write_logit=False),
                   dict(read_state=True, n_trees=2, depth=3, ternary=True),
                   dict(causal=True, act="linear", n_trees=2, depth=3, read_state=True, write_logit=False),
                   dict(n_trees=4, depth=2, n_heads=2, act="split"), dict(causal=True, n_trees=4, depth=2, n_heads=4, act="split", selective=False),
                   dict(n_trees=2, n_heads=2, depth=3, read_state=True), dict(causal=True, n_trees=6, n_heads=3, depth=1, ternary=True),
                   dict(dense_walk=False), dict(causal=True, n_trees=4, n_heads=2, depth=2, act="split", dense_walk=False)):
        m = make(**kwargs)
        u = torch.randn(3, 24, 32, dtype=torch.float64)
        mask = torch.ones(3, 24)
        mask[1, 17:] = 0
        mask[2, 5:] = 0
        dense = m(u, attention_mask=mask)
        lazy = m.forward_reference(u, attention_mask=mask)
        assert torch.allclose(dense, lazy, atol=1e-9), (kwargs, (dense - lazy).abs().max())
        # and the context term is actually doing something in this test
        m.use_context = False
        assert (m(u, attention_mask=mask) - dense).abs().max() > 1e-3
        m.use_context = True


def test_without_context_is_fff():
    """use_context=False reproduces UltraFastBERT's FFF layer (dense logits, masked to the path)."""
    m = make(use_context=False, d_conv=0, depth=5)
    x = torch.randn(2, 9, 32, dtype=torch.float64)
    logits = x @ m.node_in.T + m.node_bias  # every node, as fff.py does in training
    on_path = torch.zeros_like(logits)
    cur = torch.zeros(2, 9, dtype=torch.long)
    for _ in range(m.n_levels):
        on_path.scatter_(2, cur.unsqueeze(-1), 1.0)
        cur = 2 * cur + 1 + (logits.gather(2, cur.unsqueeze(-1)).squeeze(-1) > 0).long()
        cur = cur.clamp(max=m.n_nodes - 1)
    expected = (F.gelu(logits) * on_path) @ m.node_out
    assert torch.allclose(m(x), expected, atol=1e-10)


def test_padding_does_not_change_real_tokens():
    m = make()
    u = torch.randn(1, 10, 32, dtype=torch.float64)
    short = m(u)
    padded = torch.cat([u, torch.randn(1, 6, 32, dtype=torch.float64)], dim=1)
    mask = torch.cat([torch.ones(1, 10), torch.zeros(1, 6)], dim=1)
    long = m(padded, attention_mask=mask)
    assert torch.allclose(short, long[:, :10], atol=1e-10)
    assert long[:, 10:].abs().max() == 0


def test_context_is_bidirectional():
    """Changing one token changes the output of tokens on both sides of it."""
    m = make(d_conv=0)
    u = torch.randn(1, 12, 32, dtype=torch.float64)
    base = m(u)
    u2 = u.clone()
    u2[0, 6] = torch.randn(32, dtype=torch.float64)
    delta = (m(u2) - base).abs().amax(-1)[0]
    assert delta[:6].max() > 1e-6 and delta[7:].max() > 1e-6


def test_causal_mode_never_looks_ahead():
    """With causal=True, changing a token changes later outputs and no earlier one."""
    m = make(causal=True)
    u = torch.randn(1, 12, 32, dtype=torch.float64)
    base = m(u)
    u2 = u.clone()
    u2[0, 6] = torch.randn(32, dtype=torch.float64)
    delta = (m(u2) - base).abs().amax(-1)[0]
    assert delta[:6].max() == 0 and delta[7:].max() > 1e-6


def test_route_ste_keeps_the_forward_pass_and_changes_the_gradient():
    for kwargs in (dict(), dict(causal=True, n_trees=2, depth=3, n_heads=2, act="split"), dict(use_context=False), dict(ternary=True, act_bits=8)):
        plain, ste = make(**kwargs), make(route_ste=True, **kwargs)  # same seed, same weights
        u = torch.randn(2, 14, 32, dtype=torch.float64)
        a, b = plain(u), ste(u)
        assert torch.allclose(a, b, atol=1e-12), kwargs
        assert torch.allclose(b, ste.forward_reference(u), atol=1e-9), kwargs
        a.pow(2).mean().backward()
        b.pow(2).mean().backward()
        assert torch.isfinite(ste.node_in.grad).all() and not torch.allclose(plain.node_in.grad, ste.node_in.grad), kwargs


def test_route_ste_gradient_is_the_difference_between_the_two_branches():
    """One tree of depth 1, no context: d out / d root logit = gelu'(l0) r0 + p(1 - p) (c_right r_right - c_left r_left)."""
    m = make(depth=1, use_context=False, d_conv=0, route_ste=True, node_bias=False)
    x = torch.randn(1, 1, 32, dtype=torch.float64, requires_grad=True)
    v = torch.randn(32, dtype=torch.float64)
    (m(x)[0, 0] @ v).backward()
    with torch.no_grad():
        l = m.node_in @ x[0, 0]
        dgelu = lambda t: 0.5 * (1 + torch.erf(t / 2**0.5)) + t * torch.exp(-t * t / 2) / (2 * torch.pi) ** 0.5
        rv = m.node_out @ v  # each node's read vector along v
        c = F.gelu(l)
        p = torch.sigmoid(l[0])
        went_right = float(l[0] > 0)
        dl = torch.stack([
            dgelu(l[0]) * rv[0] + p * (1 - p) * (c[2] * rv[2] - c[1] * rv[1]),
            (1 - went_right) * dgelu(l[1]) * rv[1],
            went_right * dgelu(l[2]) * rv[2],
        ])
        expected = dl @ m.node_in
    assert torch.allclose(x.grad[0, 0], expected, atol=1e-10)


def test_gradients_are_finite_and_reach_every_parameter():
    m = make()
    u = torch.randn(2, 16, 32, dtype=torch.float64)
    m(u).pow(2).mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().max() > 0, name


def test_only_path_nodes_get_gradient():
    m = make()
    u = torch.randn(1, 4, 32, dtype=torch.float64)
    out, nodes = m(u, return_paths=True)
    out.sum().backward()
    touched = torch.zeros(m.n_nodes, dtype=torch.bool)
    touched[nodes.flatten()] = True
    assert (m.node_out.grad[~touched] == 0).all() and (m.node_in.grad[~touched] == 0).all()
    assert len(m.path_stats(nodes)) == m.n_levels and m.path_stats(nodes)[0] == 1.0


def test_mixed_forest_matches_reference_in_both_directions_and_handles_padding():
    for kwargs in (dict(), dict(causal=True), dict(ternary=True, act_bits=8)):
        torch.manual_seed(0)
        m = GTSMixed(32, bank_trees=8, bank_heads=4, bank_state=8, deep_trees=2, deep_depth=3, dtype=torch.float64, **kwargs)
        with torch.no_grad():
            m.bank.ctx_proj.weight.mul_(4.0); m.bank.dt_bias.add_(3.0); m.bank.A_log.sub_(2.0)
        u = torch.randn(2, 18, 32, dtype=torch.float64)
        mask = torch.ones(2, 18)
        mask[1, 11:] = 0
        dense = m(u, attention_mask=mask)
        assert torch.allclose(dense, m.forward_reference(u, attention_mask=mask), atol=1e-9), kwargs
        assert dense[1, 11:].abs().max() == 0
        assert torch.allclose(m(u[1:, :11]), dense[1:, :11], atol=1e-10), kwargs  # padding changes nothing
        u2 = u.clone()
        u2[0, 9] = torch.randn(32, dtype=torch.float64)
        delta = (m(u2, attention_mask=mask) - dense).abs().amax(-1)[0]
        assert delta[10:].max() > 1e-6 and (delta[:9].max() > 1e-6) != bool(kwargs.get("causal")), kwargs
        dense.pow(2).mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_masked_lm_with_a_mixed_forest_overfits_a_toy_batch():
    torch.manual_seed(0)
    config = GTSConfig(d_model=48, n_layer=2, vocab_size=40, mixer="mixed", bank_trees=8, bank_heads=4, bank_state=8,
                       deep_trees=2, deep_depth=3, route_ste=True, ternary=True)
    model = GTSForMaskedLM(config)
    ids = torch.randint(2, 40, (8, 20))
    ids[:, 15:] = config.pad_token_id
    labels = torch.full_like(ids, -100)
    masked = torch.zeros_like(ids, dtype=torch.bool)
    masked[:, 3::4] = True
    masked &= ids != config.pad_token_id
    labels[masked] = ids[masked]
    inputs = ids.masked_fill(masked, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = model(inputs, labels=labels).loss
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss) and loss.item() < 0.5 * first, (first, loss.item())


def test_masked_lm_overfits_a_toy_batch():
    torch.manual_seed(0)
    config = GTSConfig(d_model=48, n_layer=2, vocab_size=40, depth=4, d_state=8)
    model = GTSForMaskedLM(config)
    ids = torch.randint(2, 40, (8, 20))
    ids[:, 15:] = config.pad_token_id
    labels = torch.full_like(ids, -100)
    masked = torch.zeros_like(ids, dtype=torch.bool)
    masked[:, 3::4] = True
    masked &= ids != config.pad_token_id
    labels[masked] = ids[masked]
    inputs = ids.masked_fill(masked, 1)  # token 1 plays [MASK]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        loss = model(inputs, labels=labels).loss
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss) and loss.item() < 0.5 * first, (first, loss.item())
