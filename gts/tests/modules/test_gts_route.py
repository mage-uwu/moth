# Golden Tree Snake (GTS) fork, 2026.
"""The Triton route_ste kernels (mamba_ssm/ops/gts_route.py) against GTS's dense straight-through form.

Without a GPU these run in Triton's interpreter, which this module switches on before the kernels are first imported.
"""
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

triton = pytest.importorskip("triton")

from mamba_ssm.modules.gts import GTS, GTSMixed  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _pair(**kw):
    torch.manual_seed(0)
    a = GTS(32, route_ste=True, use_context=False, dense_walk=True, route_kernel=True, **kw).to(DEV)
    b = GTS(32, route_ste=True, use_context=False, dense_walk=True, route_kernel=False, **kw).to(DEV)
    b.load_state_dict(a.state_dict())
    return a, b


@pytest.mark.parametrize("depth,n_trees,act,temp,ternary", [
    (3, 1, "gelu", 1.0, False), (6, 3, "gelu", 0.5, False), (4, 2, "linear", 1.0, False),
    (5, 2, "split", 2.0, False), (6, 4, "gelu", 1.0, True),
])
def test_route_kernel_matches_dense_ste(depth, n_trees, act, temp, ternary):
    a, b = _pair(depth=depth, n_trees=n_trees, act=act, route_ste_temp=temp, d_conv=3, causal=True,
                 ternary=ternary, act_bits=8 if ternary else None)
    u = torch.randn(2, 37, 32, device=DEV)
    mask = torch.ones(2, 37, device=DEV)
    mask[1, 30:] = 0  # padding
    g = torch.randn(2, 37, 32, device=DEV)
    ua, ub = u.clone().requires_grad_(), u.clone().requires_grad_()
    ya, yb = a(ua, attention_mask=mask), b(ub, attention_mask=mask)
    assert torch.allclose(ya, yb, atol=1e-5)
    (ya * g).sum().backward()
    (yb * g).sum().backward()
    assert torch.allclose(ua.grad, ub.grad, atol=1e-5 * (1 + ub.grad.abs().max().item()))
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        err = (pa.grad - pb.grad).abs().max().item()
        assert err <= 1e-5 * (1 + pb.grad.abs().max().item()), f"{name}: {err:.2e}"


def test_route_kernel_paths_and_reference():
    a, b = _pair(depth=5, n_trees=3, d_conv=3, causal=True)
    u = torch.randn(2, 20, 32, device=DEV)
    ya, na = a(u, return_paths=True)
    yb, nb = b(u, return_paths=True)
    assert torch.equal(na, nb)
    assert torch.allclose(ya, a.forward_reference(u), atol=1e-4)


def test_mixed_deep_trees_use_route_kernel():
    torch.manual_seed(0)
    m = GTSMixed(32, bank_trees=8, bank_heads=4, deep_trees=2, deep_depth=4, causal=True).to(DEV)
    u = torch.randn(2, 25, 32, device=DEV)
    m.deep.route_kernel = True
    y = m(u)
    m.deep.route_kernel = False
    assert torch.allclose(y, m(u), atol=1e-5)


@pytest.mark.parametrize("depth,n_trees,act", [(4, 2, "gelu"), (6, 3, "linear")])
def test_route_kernel_without_ste_matches_plain_walk(depth, n_trees, act):
    """route_ste=False: the kernels against the original walk-and-scatter path (FFF routing, no branch gradient)."""
    torch.manual_seed(0)
    kw = dict(depth=depth, n_trees=n_trees, act=act, use_context=False, dense_walk=True, d_conv=3, causal=False)
    a, b = GTS(32, route_kernel=True, **kw).to(DEV), GTS(32, route_kernel=False, **kw).to(DEV)
    b.load_state_dict(a.state_dict())
    u, g = torch.randn(2, 30, 32, device=DEV), torch.randn(2, 30, 32, device=DEV)
    mask = torch.ones(2, 30, device=DEV)
    mask[0, 25:] = 0
    ua, ub = u.clone().requires_grad_(), u.clone().requires_grad_()
    ya, yb = a(ua, attention_mask=mask), b(ub, attention_mask=mask)
    assert torch.allclose(ya, yb, atol=1e-5)
    (ya * g).sum().backward()
    (yb * g).sum().backward()
    assert torch.allclose(ua.grad, ub.grad, atol=1e-5 * (1 + ub.grad.abs().max().item()))
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5 * (1 + pb.grad.abs().max().item())), name
