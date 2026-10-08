# Golden Tree Snake (GTS) fork, 2026.
"""GTS-OMEGA reference identities (mamba_ssm/omega/reference.py) on small random GeGLU MLPs, float64."""
import math

import pytest
import torch

from mamba_ssm.omega.reference import (Hinge, even_part, geglu, geglu_hinge, geglu_hinge_bound, gelu, jacobian,
                                       odd_part, reglu_region, shared_z, switched_terms, telescope)


@pytest.fixture
def mlp():
    g = torch.Generator().manual_seed(0)
    d, F = 24, 40
    G = torch.randn(F, d, generator=g, dtype=torch.float64) / math.sqrt(d)
    U = torch.randn(F, d, generator=g, dtype=torch.float64) / math.sqrt(d)
    O = torch.randn(d, F, generator=g, dtype=torch.float64) / math.sqrt(F)
    x = 2 * torch.randn(64, d, generator=g, dtype=torch.float64)
    return x, G, U, O


def test_even_and_odd_parts_are_exact(mlp):
    x, G, U, O = mlp
    f, fm = geglu(x, G, U, O), geglu(-x, G, U, O)
    torch.testing.assert_close((f + fm) / 2, even_part(x, G, U, O), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close((f - fm) / 2, odd_part(x, G, U, O), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(f, even_part(x, G, U, O) + odd_part(x, G, U, O), rtol=1e-12, atol=1e-12)


def test_shared_z_expectation(mlp):
    x, G, U, O = mlp
    est = shared_z(x[:8], G, U, O, n_samples=200000, generator=torch.Generator().manual_seed(1))
    f = geglu(x[:8], G, U, O)
    assert ((est - f).norm() / f.norm()).item() < 1e-2  # Monte Carlo, 2e5 shared draws


def test_hinge_bound_parity_and_tails():
    for T, delta in [(4.0, 0.5), (4.0, 0.125), (3.0, 0.25)]:
        h = Hinge(T, delta)
        t = torch.linspace(-12, 12, 200001, dtype=torch.float64)
        err = (gelu(t) - h(t)).abs().max().item()
        assert err <= h.bound() * (1 + 1e-9), (T, delta, err, h.bound())
        torch.testing.assert_close(h(t) - h(-t), t, rtol=0, atol=1e-12)  # symmetric knots: h_D(t) - h_D(-t) = t
        torch.testing.assert_close(h(h.tau), gelu(h.tau), rtol=0, atol=1e-14)  # interpolates at the knots


def test_hinge_compile_bound_and_even_part(mlp):
    x, G, U, O = mlp
    h = Hinge(4.0, 0.25)
    f, fd = geglu(x, G, U, O), geglu_hinge(x, G, U, O, h)
    assert ((f - fd).norm(dim=-1) <= geglu_hinge_bound(x, G, U, O, h) * (1 + 1e-9)).all()
    torch.testing.assert_close((fd + geglu_hinge(-x, G, U, O, h)) / 2, even_part(x, G, U, O), rtol=1e-12, atol=1e-12)


def test_switched_terms_sum_and_continuity(mlp):
    x, G, U, O = mlp
    h = Hinge(4.0, 0.5)
    fs, active = switched_terms(x, G, U, O, h)
    torch.testing.assert_close(fs, geglu_hinge(x, G, U, O, h), rtol=1e-10, atol=1e-10)
    assert (active > 0).all()
    # continuity: move x across the switching hyperplane b_0 = tau_k; the map is continuous (left and right limits agree)
    j, k = 0, 9
    g0 = G[j]
    base = x[0] - (x[0] @ g0 - h.tau[k]) / (g0 @ g0) * g0  # b_0(base) = tau_k
    eps = 1e-7
    lo, hi = base - eps * g0 / g0.norm(), base + eps * g0 / g0.norm()
    flo, _ = switched_terms(lo[None], G, U, O, h)
    fhi, _ = switched_terms(hi[None], G, U, O, h)
    assert (flo - fhi).norm().item() < 1e-5


def test_reglu_regions_and_boundary(mlp):
    x, G, U, O = mlp
    relu = torch.relu
    f = geglu(x, G, U, O, h=relu)
    s = (x @ G.T > 0).to(x.dtype)
    torch.testing.assert_close(f, torch.stack([reglu_region(x[i:i + 1], G, U, O, s[i])[0] for i in range(len(x))]))
    # neighbouring regions (one gate flipped) agree on that gate's hyperplane
    j = 3
    p = x[0] - (x[0] @ G[j]) / (G[j] @ G[j]) * G[j]
    s1 = (p @ G.T > 0).to(x.dtype); s1[j] = 1
    s0 = s1.clone(); s0[j] = 0
    torch.testing.assert_close(reglu_region(p[None], G, U, O, s1), reglu_region(p[None], G, U, O, s0), rtol=0, atol=1e-12)


def test_telescoping(mlp):
    x, G, U, O = mlp
    F = G.shape[0]
    sets = [torch.zeros(F, dtype=x.dtype)]
    for j in (2, 7, 11, 30):
        s = sets[-1].clone(); s[j] = 1; sets.append(s)
    total, leaf = telescope(x, G, U, O, sets)
    torch.testing.assert_close(total, leaf, rtol=1e-12, atol=1e-12)


def test_jacobian(mlp):
    x, G, U, O = mlp
    J = jacobian(x[0], G, U, O)
    Jref = torch.autograd.functional.jacobian(lambda v: geglu(v[None], G, U, O)[0], x[0])
    torch.testing.assert_close(J, Jref, rtol=1e-10, atol=1e-10)
