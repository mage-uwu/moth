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
from mamba_ssm.ops.gts_scan import gts_scan, gts_scan_bi, gts_scan_reference  # noqa: E402

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


@pytest.mark.parametrize("l,chunk", [(64, 16), (77, 32), (20, 64)])
def test_bidirectional_scan_matches_reference(l, chunk):
    torch.manual_seed(0)
    Cf, Cb, B = (torch.randn(2, l, 16, device=DEV, requires_grad=True) for _ in range(3))
    X = torch.randn(2, l, 3, 4, device=DEV, requires_grad=True)
    a = (-torch.rand(2, l, 3, device=DEV) * 0.5).requires_grad_()
    ref = gts_scan_reference(Cf, B, X, a) + gts_scan_reference(Cb, B, X, a, reverse=True)
    out = gts_scan_bi(Cf, Cb, B, X, a, chunk, "ieee")
    g = torch.randn_like(ref)
    for x, y in zip((out,) + torch.autograd.grad((out * g).sum(), (Cf, Cb, B, X, a)), (ref,) + torch.autograd.grad((ref * g).sum(), (Cf, Cb, B, X, a))):
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
    """Values and every gradient of a depth-0 GTS with the scan, against the dense path in float64 (the ground truth).
    The scan's float32 error must stay within 10x the dense path's own float32 error, or 1e-4 of the gradient's scale.
    The second bound is for dt_bias and A_log, whose gradients are sums over tokens that cancel: on an A100 the scan's
    log-decay gradient is as accurate as float32 PyTorch's at 512 tokens and ten times more accurate at 8K
    (scripts/diag_scan.py), yet the summed dt_bias gradient can land 3e-5 from float64 where the dense path lands 5e-7."""
    torch.manual_seed(0)
    kw = dict(depth=0, n_trees=8, n_heads=4, d_state=16, act="split", d_conv=3, causal=causal)
    scan, dense = GTS(32, scan_kernel=True, **kw).to(DEV), GTS(32, scan_kernel=False, **kw).to(DEV)
    dense.load_state_dict(scan.state_dict())
    truth = GTS(32, scan_kernel=False, **kw).to(DEV).double()
    truth.load_state_dict(scan.state_dict())
    u, g = torch.randn(2, 70, 32, device=DEV), torch.randn(2, 70, 32, device=DEV)
    results = []
    for m, dtype in ((scan, torch.float32), (dense, torch.float32), (truth, torch.float64)):
        x = u.to(dtype).detach().clone().requires_grad_()
        y = m(x)
        (y * g.to(dtype)).sum().backward()
        results.append([y.double(), x.grad.double()] + [p.grad.double() for p in m.parameters()])
    assert torch.allclose(results[0][0].float(), scan.forward_reference(u), atol=1e-4)
    names = ["output", "input"] + [n for n, _ in scan.named_parameters()]
    for name, a, b, t in zip(names, *results):
        err_scan, err_dense = (a - t).abs().max().item(), (b - t).abs().max().item()
        assert err_scan <= max(10 * err_dense, 1e-4 * t.abs().max().item()) + 1e-6, f"{name}: scan error {err_scan:.2e}, dense float32 error {err_dense:.2e}"


def test_mixed_bank_uses_scan():
    torch.manual_seed(0)
    m = GTSMixed(32, bank_trees=8, bank_heads=4, bank_state=16, deep_trees=2, deep_depth=3).to(DEV)
    m.bank.scan_kernel = True
    u = torch.randn(2, 40, 32, device=DEV)
    y = m(u)
    m.bank.scan_kernel = False
    assert torch.allclose(y, m(u), atol=1e-5)
