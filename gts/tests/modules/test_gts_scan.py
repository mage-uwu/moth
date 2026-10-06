# Golden Tree Snake (GTS) fork, 2026.
"""The chunked Triton scan for depth-0 trees (mamba_ssm/ops/gts_scan.py) against the quadratic PyTorch form.

Without a GPU these run in Triton's interpreter, which this module switches on before the kernel is first imported.
"""
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")

triton = pytest.importorskip("triton")

from mamba_ssm.modules.gts import GTS, GTSMixed  # noqa: E402
from mamba_ssm.ops.gts_scan import gts_scan, gts_scan_reference  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("excl", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("b,l,h,p,n,chunk", [(2, 64, 2, 4, 16, 16), (2, 100, 3, 4, 16, 32), (1, 37, 2, 8, 16, 16), (1, 50, 2, 5, 20, 16)])
def test_scan_matches_reference(b, l, h, p, n, chunk, reverse, excl):
    torch.manual_seed(0)
    C, B = (torch.randn(b, l, n, device=DEV, requires_grad=True) for _ in range(2))
    X = torch.randn(b, l, h, p, device=DEV, requires_grad=True)
    a = (-torch.rand(b, l, h, device=DEV) * 0.3).requires_grad_()
    ref, out = gts_scan_reference(C, B, X, a, reverse, excl), gts_scan(C, B, X, a, reverse, excl, chunk, "ieee")
    g = torch.randn_like(ref)
    for x, y in zip((out,) + torch.autograd.grad((out * g).sum(), (C, B, X, a)), (ref,) + torch.autograd.grad((ref * g).sum(), (C, B, X, a))):
        assert torch.allclose(x, y, rtol=1e-4, atol=1e-4 * y.abs().max().item())


def test_scan_reverse_is_gts_backward_context():
    """reverse=True is the decay a[t] + ... + a[s-1] from s > t, which GTS's bidirectional context uses."""
    torch.manual_seed(0)
    C, B = torch.randn(1, 20, 16, device=DEV), torch.randn(1, 20, 16, device=DEV)
    X, a = torch.randn(1, 20, 2, 4, device=DEV), -torch.rand(1, 20, 2, device=DEV)
    cs = torch.cumsum(a, 1)
    cs_ex = cs - a
    w = torch.exp((cs_ex[:, None, :, :] - cs_ex[:, :, None, :]).masked_fill(~torch.ones(20, 20, dtype=torch.bool, device=DEV).triu(1)[None, :, :, None], -torch.inf))
    want = torch.einsum("btsh,bshp->bthp", (C @ B.transpose(1, 2)).unsqueeze(-1) * w, X)
    assert torch.allclose(gts_scan_reference(C, B, X, a, reverse=True), want, atol=1e-5)
    assert torch.allclose(gts_scan(C, B, X, a, reverse=True, chunk=16, precision="ieee"), want, atol=1e-4)


@pytest.mark.parametrize("causal", [True, False])
def test_gts_depth0_scan_matches_dense_and_reference(causal):
    torch.manual_seed(0)
    kw = dict(depth=0, n_trees=8, n_heads=4, d_state=16, act="split", d_conv=3, causal=causal)
    a, b = GTS(32, scan_kernel=True, **kw).to(DEV), GTS(32, scan_kernel=False, **kw).to(DEV)
    b.load_state_dict(a.state_dict())
    u = torch.randn(2, 70, 32, device=DEV)
    ua, ub = u.clone().requires_grad_(), u.clone().requires_grad_()
    ya, yb = a(ua), b(ub)
    assert torch.allclose(ya, yb, atol=1e-5)
    assert torch.allclose(ya, a.forward_reference(u), atol=1e-4)
    g = torch.randn_like(ya)
    (ya * g).sum().backward()
    (yb * g).sum().backward()
    assert torch.allclose(ua.grad, ub.grad, atol=1e-5)
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        err = (pa.grad - pb.grad).abs().max().item()
        assert err <= 1e-5 * (1 + pb.grad.abs().max().item()), f"{name}: max |diff| {err:.3e}, max |grad| {pb.grad.abs().max().item():.3e}"


def test_mixed_bank_uses_scan():
    torch.manual_seed(0)
    m = GTSMixed(32, bank_trees=8, bank_heads=4, bank_state=16, deep_trees=2, deep_depth=3).to(DEV)
    m.bank.scan_kernel = True
    u = torch.randn(2, 40, 32, device=DEV)
    y = m(u)
    m.bank.scan_kernel = False
    assert torch.allclose(y, m(u), atol=1e-5)
