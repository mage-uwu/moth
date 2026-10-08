# Golden Tree Snake (GTS) fork, 2026.
"""GTS-OMEGA reference: exact identities and controlled conversions of a bias-free GeGLU MLP

    f(x) = O [ (U x) * h(G x) ] = sum_j o_j a_j(x) h(b_j(x)),   a_j = u_j . x,  b_j = g_j . x,  h(t) = t Phi(t)

(ModernBERT: G = Wi[:F], the GELU'd half; U = Wi[F:]; O = Wo; exact erf GELU; no biases). Plain float64 PyTorch,
written for checking (tests/omega/test_reference.py), not for speed. Every function states what it preserves.

Exact:
  even_part(x)        (f(x) + f(-x)) / 2 = Q(x) / 2,  Q(x) = sum_j o_j a_j b_j        (h(t) - h(-t) = t)
  odd_part(x)         (f(x) - f(-x)) / 2 = 1/2 sum_j o_j a_j b_j erf(b_j / sqrt 2)
  shared_z(x, z)      f(x) = E_Z sum_j o_j a_j b_j 1[b_j >= Z],  Z ~ N(0, 1), one Z for all j (Phi(b) = P(Z <= b))
  reglu_region(x, s)  for h = relu: f(x) = Q_s(x) with s = 1[b > 0]; neighbouring regions agree on the hyperplane
  telescope(...)      Q_root + sum over the path of (Q_v - Q_parent) = Q_leaf
Controlled:
  Hinge               piecewise-linear GELU on [-T, T], spacing Delta, tails of slope 0 (left) and 1 (right):
                      sup |h - h_Delta| <= max(sqrt(2/pi) Delta^2 / 8, T Phi(-T)); symmetric knots keep
                      h_Delta(t) - h_Delta(-t) = t exactly, so the even part Q/2 survives the conversion
  geglu_hinge         f_Delta = sum_j o_j a_j h_Delta(b_j); ||f - f_Delta|| <= eps ||O||_op ||U x||
  switched_terms      f_Delta as alpha-term + sum_{j,k} c_k o_j a_j (b_j - tau_k) 1[b_j > tau_k]: a sum of quadratic
                      pieces, each zero on its own switching hyperplane b_j = tau_k (continuity)
"""
import math

import torch

SQRT2 = math.sqrt(2.0)


def gelu(t):
    return t * 0.5 * (1.0 + torch.erf(t / SQRT2))


def geglu(x, G, U, O, h=gelu):
    """f(x) for rows x (n, d); G, U (F, d); O (d_out, F)."""
    return ((x @ U.T) * h(x @ G.T)) @ O.T


def even_part(x, G, U, O):
    """Q(x) / 2: exactly (f(x) + f(-x)) / 2. A vector of quadratic forms with CP rank <= F."""
    return 0.5 * ((x @ U.T) * (x @ G.T)) @ O.T


def odd_part(x, G, U, O):
    """(f(x) - f(-x)) / 2 = 1/2 sum_j o_j a_j psi(b_j), psi(b) = b erf(b / sqrt 2) (even in b, ~|b|)."""
    b = x @ G.T
    return 0.5 * ((x @ U.T) * b * torch.erf(b / SQRT2)) @ O.T


def shared_z(x, G, U, O, n_samples=20000, generator=None):
    """Monte Carlo of f(x) = E_Z sum_j o_j a_j b_j 1[b_j >= Z] with one shared Z ~ N(0, 1) per sample."""
    a, b = x @ U.T, x @ G.T
    z = torch.randn(n_samples, generator=generator, dtype=x.dtype)
    on = (b[None] >= z[:, None, None]).to(x.dtype).mean(0)  # (n, F): the fraction of Z under each gate value = Phi(b)
    return (a * b * on) @ O.T


class Hinge:
    """Piecewise-linear interpolant of GELU on symmetric knots tau_k = -T + k Delta (k = 0..K), constant h(-T) left of
    -T (slope 0) and slope 1 right of T. As alpha + sum_k c_k relu(b - tau_k):  c_0 = s_0, c_k = s_k - s_{k-1},
    c_K = 1 - s_{K-1}, s_k the slope on [tau_k, tau_{k+1}]."""

    def __init__(self, T=4.0, delta=0.25, dtype=torch.float64):
        K = int(round(2 * T / delta))
        assert abs(K * delta - 2 * T) < 1e-9, "Delta must divide 2T (symmetric knots)"
        self.T, self.delta = T, delta
        self.tau = torch.linspace(-T, T, K + 1, dtype=dtype)
        v = gelu(self.tau)
        s = (v[1:] - v[:-1]) / delta  # K segment slopes
        self.alpha = v[0]
        self.c = torch.cat([s[:1], s[1:] - s[:-1], (1.0 - s[-1:])])  # K + 1 slope changes
        self.knots = self.tau

    def __call__(self, b):
        return self.alpha + (self.c * torch.relu(b[..., None] - self.tau)).sum(-1)

    def bound(self):
        """Global sup error: interpolation inside [-T, T] (|h''| <= sqrt(2/pi) at 0), tails T Phi(-T)."""
        tail = self.T * 0.5 * math.erfc(self.T / SQRT2)
        return max(math.sqrt(2 / math.pi) / 8 * self.delta ** 2, tail)


def geglu_hinge(x, G, U, O, hinge):
    return geglu(x, G, U, O, h=hinge)


def geglu_hinge_bound(x, G, U, O, hinge):
    """Per-row bound on ||f(x) - f_Delta(x)||_2: eps ||O||_op ||U x||_2."""
    return hinge.bound() * torch.linalg.matrix_norm(O, ord=2) * (x @ U.T).norm(dim=-1)


def switched_terms(x, G, U, O, hinge):
    """f_Delta written as its threshold-gated pieces: alpha sum_j o_j a_j  +  sum_{j,k} c_k o_j a_j (b_j - tau_k) 1[b_j > tau_k].
    Returns (f_Delta, number of active predicates per row). Same value as geglu_hinge, summed term by term."""
    a, b = x @ U.T, x @ G.T
    out = hinge.alpha * (a @ O.T)
    active = torch.zeros(x.shape[0], dtype=torch.long)
    for k in range(len(hinge.tau)):
        pred = b > hinge.tau[k]
        out = out + hinge.c[k] * ((a * (b - hinge.tau[k]) * pred) @ O.T)
        active += pred.sum(-1)
    return out, active


def reglu_region(x, G, U, O, s):
    """Q_s(x) = sum_j s_j o_j a_j b_j for a fixed sign pattern s (F,) in {0, 1}."""
    return ((x @ U.T) * (x @ G.T) * s) @ O.T


def telescope(x, G, U, O, path_sets):
    """path_sets: the gate sets switched on at each node of a root-to-leaf path (root first; each a superset of its
    parent's). Returns (sum of the root map and child-minus-parent corrections, the leaf map)."""
    total = reglu_region(x, G, U, O, path_sets[0])
    for parent, child in zip(path_sets[:-1], path_sets[1:]):
        total = total + reglu_region(x, G, U, O, child) - reglu_region(x, G, U, O, parent)
    return total, reglu_region(x, G, U, O, path_sets[-1])


def jacobian(x, G, U, O):
    """J_f(x) = O [diag(h(b)) U + diag(a h'(b)) G] for one row x (d,). (h'(t) = Phi(t) + t phi(t).)"""
    a, b = U @ x, G @ x
    hp = 0.5 * (1 + torch.erf(b / SQRT2)) + b * torch.exp(-0.5 * b * b) / math.sqrt(2 * math.pi)
    return O @ (gelu(b)[:, None] * U + (a * hp)[:, None] * G)
