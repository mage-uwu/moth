# Golden Tree Snake (GTS) fork, 2026.
"""GTS-OMEGA geometry diagnostics on ModernBERT-large's MLPs (bias-free GeGLU, f(x) = O[(Ux) * gelu(Gx)]).

Collects each chosen layer's MLP input x (after mlp_norm) and output y on FineWeb-Edu (fit on --train tokens, measure
on --test held-out tokens), then per layer:

  dist      the input distribution: per-dimension spread, PCA rank, kurtosis of projections (Gaussian = 3)
  parity    shares of the exact even part Q/2 and odd part; f(-x) is the teacher off-distribution, so both are
            measured as approximations of f on real x
  affine    least-squares affine fit on real x vs the affine map fitted on a Gaussian with real x's mean and
            covariance (the Hermite constant + linear projection), both scored on real x
  gates     b = Gx: fraction of tokens with each gate on, per-token count in the switching zone |b| < 1, gates
            always on / off and their output energy, range coverage |b| > T, effective rank of the gate-sign pattern
  surrogate ReGLU (relu for gelu: the sign-region quadratic family's floor) and hinge GELU (Delta = 0.5, 0.25)
  quadratic Q = sum_j o_j a_j b_j: the teacher's own top-r terms (output refit by least squares) and a data-fitted
            CP-rank-r quadratic P3[(P1 x) * (P2 x)], vs r; Q's output rank
  jacobian  J_f at held-out tokens: singular spectra, rank for 90% energy, the best rank-168 local error (the GTS
            route's cap: rank-128 affine + 40 nodes), for J, J minus the fitted affine slope, and J on the input's
            principal subspace
  regions   sign-region quadratics Q_v = sum_{j in S_v} o_j a_j b_j along a tree of gate predicates (S_v: gates on
            for most of the region's tokens): error vs depth, size and energy and output rank of child - parent
  families  equal-compute surrogates fitted on the same tokens: affine rank 128; dense GeGLU m = 110 and 256 (init
            from the teacher's own top neurons); affine + CP quadratic; OMEGA-lite (affine + shared quadratic +
            per-leaf bilinear corrections routed by teacher gate signs); multiply-adds and parameters
  e2e       each family spliced in place of the teacher MLP at that one layer (attention and everything else kept):
            KL(teacher || spliced), top-1 agreement and CE at masked positions of held-out sequences

    python scripts/omega_diagnostics.py --layers 4 10 14 17 24 --out results/omega/diagnostics.json
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mamba_ssm.omega.reference import Hinge, gelu  # noqa: E402

TEACHER = os.environ.get("MOHAWK_TEACHER", "answerdotai/ModernBERT-large")
MASK, CLS, SEP = 50284, 50281, 50282


def rel(a, b):
    return float((a - b).pow(2).sum() / b.pow(2).sum())


def collect(layers, n_tokens, seq=512):
    from datasets import load_dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    m = AutoModelForMaskedLM.from_pretrained(TEACHER, dtype=torch.float32, attn_implementation="sdpa").eval()
    rec = {i: {"x": [], "y": []} for i in layers}

    def hook(i):
        def f(mod, args, out):
            rec[i]["x"].append(args[0].reshape(-1, args[0].shape[-1]))
            rec[i]["y"].append(out.reshape(-1, out.shape[-1]))
        return f

    hs = [m.model.layers[i].mlp.register_forward_hook(hook(i)) for i in layers]
    seqs, buf = [], []
    stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    t0 = time.time()
    with torch.no_grad():
        for doc in stream:
            buf += tok(doc["text"], add_special_tokens=False)["input_ids"] + [SEP]
            while len(buf) >= 8 * (seq - 1):
                ids = torch.tensor([[CLS] + buf[k * (seq - 1):(k + 1) * (seq - 1)] for k in range(8)])
                buf = buf[8 * (seq - 1):]
                m(input_ids=ids)
                seqs.append(ids)
                n = sum(s.numel() for s in seqs)
                print(f"  collected {n:,} tokens ({time.time() - t0:.0f} s)", flush=True)
                if n >= n_tokens:
                    for h in hs:
                        h.remove()
                    return {i: {k: torch.cat(v) for k, v in r.items()} for i, r in rec.items()}, m, torch.cat(seqs)
    raise RuntimeError("ran out of text")


def weights(m, i):
    Wi = m.model.layers[i].mlp.Wi.weight.detach()
    Fd = Wi.shape[0] // 2
    return Wi[:Fd].contiguous(), Wi[Fd:].contiguous(), m.model.layers[i].mlp.Wo.weight.detach().contiguous()


def teacher_f(x, G, U, O, h=gelu):
    return ((x @ U.T) * h(x @ G.T)) @ O.T


def rank_for(sv2, frac=0.9):
    c = torch.cumsum(sv2, 0) / sv2.sum()
    return int((c < frac).sum()) + 1


def affine_fit(x, y, eps=1e-4):
    X = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1).double()
    A = X.T @ X
    W = torch.linalg.solve(A + eps * A.diagonal().mean() * torch.eye(len(A), dtype=A.dtype), X.T @ y.double())
    return W.float()


def affine_apply(W, x):
    return x @ W[:-1] + W[-1]


def rrr(W, x, r):
    """Reduced-rank version of an affine fit along the top output directions of its centred fitted values."""
    fit = affine_apply(W, x)
    V = torch.linalg.svd(fit - fit.mean(0), full_matrices=False).Vh[:r].T  # (d, r)
    Wr = W.clone()
    mu = x.mean(0)
    Wr[:-1] = W[:-1] @ V @ V.T
    Wr[-1] = W[-1] + mu @ W[:-1] - mu @ W[:-1] @ V @ V.T
    return Wr


# ------------------------------------------------------------------------------------------------- families
class DenseGeGLU(torch.nn.Module):
    def __init__(self, G, U, O):
        super().__init__()
        self.G, self.U, self.O = (torch.nn.Parameter(t.clone()) for t in (G, U, O))

    def forward(self, x):
        return ((x @ self.U.T) * gelu(x @ self.G.T)) @ self.O.T

    def macs(self):
        return 3 * self.G.shape[0] * self.G.shape[1]


class AffineQuad(torch.nn.Module):
    """Rank-r affine + CP-rank-q quadratic: B(Ax) + c + P3[(P1 x) * (P2 x)]."""

    def __init__(self, Wr, r, P1, P2, P3):
        super().__init__()
        d = Wr.shape[1]
        U_, S_, V_ = torch.linalg.svd(Wr[:-1], full_matrices=False)
        self.A = torch.nn.Parameter((U_[:, :r] * S_[:r]).clone())  # (d, r)
        self.B = torch.nn.Parameter(V_[:r].clone())  # (r, d)
        self.c = torch.nn.Parameter(Wr[-1].clone())
        self.P1, self.P2, self.P3 = (torch.nn.Parameter(t.clone()) for t in (P1, P2, P3)) if P1 is not None else (None, None, None)

    def forward(self, x):
        y = (x @ self.A) @ self.B + self.c
        if self.P1 is not None:
            y = y + ((x @ self.P1.T) * (x @ self.P2.T)) @ self.P3.T
        return y

    def macs(self):
        d, r = self.B.shape[1], self.B.shape[0]
        return 2 * d * r + (3 * d * self.P1.shape[0] if self.P1 is not None else 0)


class OmegaLite(AffineQuad):
    """AffineQuad + routed bilinear corrections: `depth` teacher-gate sign predicates (gates sel) pick one of 2^depth
    leaves; each leaf adds a rank-k bilinear correction sum_l v_l (p_l . x)(q_l . x)."""

    def __init__(self, Wr, r, P1, P2, P3, gates, k):
        super().__init__(Wr, r, P1, P2, P3)
        self.register_buffer("route", gates.clone())  # (depth, d)
        L, d = 2 ** gates.shape[0], gates.shape[1]
        self.Lp = torch.nn.Parameter(torch.randn(L, k, d) * 0.01)
        self.Lq = torch.nn.Parameter(torch.randn(L, k, d) * 0.01)
        self.Lv = torch.nn.Parameter(torch.zeros(L, k, d))

    def leaf(self, x):
        bits = (x @ self.route.T > 0).long()
        return (bits * (2 ** torch.arange(bits.shape[1]))).sum(1)

    def forward(self, x):
        y = super().forward(x)
        lf = self.leaf(x)
        p = torch.einsum("nd,nkd->nk", x, self.Lp[lf])
        q = torch.einsum("nd,nkd->nk", x, self.Lq[lf])
        return y + torch.einsum("nk,nkd->nd", p * q, self.Lv[lf])

    def macs(self):
        return super().macs() + self.route.shape[0] * self.route.shape[1] + 3 * self.Lp.shape[1] * self.Lp.shape[2]


def train(model, xtr, ytr, xte, yte, steps, lr=1e-3, bs=4096, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    yn = ytr.pow(2).sum(1).mean()
    for s in range(steps):
        i = torch.randint(0, len(xtr), (bs,), generator=g)
        loss = (model(xtr[i]) - ytr[i]).pow(2).sum(1).mean() / yn
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        return rel(model(xte), yte)


# ------------------------------------------------------------------------------------------------- analysis
def analyse(i, x, y, G, U, O, ntr, args):
    torch.manual_seed(0)
    xtr, xte, ytr, yte = x[:ntr], x[ntr:], y[:ntr], y[ntr:]
    R = {"layer": i}
    with torch.no_grad():
        R["teacher_check"] = rel(teacher_f(xte, G, U, O), yte)
        # dist
        mu, xc = xtr.mean(0), xtr - xtr.mean(0)
        cov = xc.T @ xc / len(xc)
        ev = torch.linalg.eigvalsh(cov.double()).flip(0).float()
        proj = xc @ torch.randn(x.shape[1], 64) / math.sqrt(x.shape[1])
        pc = xc @ torch.linalg.eigh(cov.double()).eigenvectors[:, -8:].float()
        kurt = lambda z: float(((z - z.mean(0)).pow(4).mean(0) / (z.var(0) ** 2)).median())  # noqa: E731
        R["dist"] = {"pca_rank90": rank_for(ev, 0.9), "pca_rank99": rank_for(ev, 0.99),
                     "top1_var_share": float(ev[0] / ev.sum()), "kurtosis_random_proj": kurt(proj),
                     "kurtosis_top_pcs": kurt(pc), "max_dim_std_over_median": float(xtr.std(0).max() / xtr.std(0).median())}
        # parity
        Qh = 0.5 * ((xte @ U.T) * (xte @ G.T)) @ O.T
        b = xte @ G.T
        Rodd = 0.5 * ((xte @ U.T) * b * torch.erf(b / math.sqrt(2))) @ O.T
        R["parity"] = {"even_Q_half_alone": rel(Qh, yte), "odd_alone": rel(Rodd, yte),
                       "even_energy_share": float(Qh.pow(2).sum() / yte.pow(2).sum()),
                       "odd_energy_share": float(Rodd.pow(2).sum() / yte.pow(2).sum()),
                       "cos_even_odd": float((Qh * Rodd).sum() / (Qh.norm() * Rodd.norm()))}
        # affine: data vs Gaussian surrogate (Hermite constant + linear)
        W = affine_fit(xtr, ytr)
        L = torch.linalg.cholesky(cov.double() + 1e-6 * torch.eye(len(cov), dtype=torch.float64)).float()
        xg = mu + torch.randn(ntr, x.shape[1]) @ L.T
        Wg = affine_fit(xg, teacher_f(xg, G, U, O))
        R["affine"] = {"data_fit": rel(affine_apply(W, xte), yte), "gaussian_hermite_fit": rel(affine_apply(Wg, xte), yte),
                       "rel_diff_slope": float((Wg[:-1] - W[:-1]).norm() / W[:-1].norm()),
                       "rank128": rel(affine_apply(rrr(W, xtr, 128), xte), yte),
                       "teacher_on_gaussian_vs_affine": rel(affine_apply(Wg, xg[:4096]), teacher_f(xg[:4096], G, U, O))}
        # gates
        btr = xtr @ G.T
        pon = (btr > 0).float().mean(0)
        a_te = xte @ U.T
        terms = a_te * gelu(b)
        def share(mask):
            return float(((terms * mask) @ O.T).pow(2).sum() / yte.pow(2).sum())
        always_on, always_off = pon > 0.98, pon < 0.02
        s = (btr[:8192] > 0).float()
        sc = s - s.mean(0)
        sv = torch.linalg.svdvals(sc / (sc.norm(dim=0, keepdim=True) + 1e-6)) ** 2
        R["gates"] = {"n": int(G.shape[0]), "mean_on_fraction": float(pon.mean()),
                      "always_on": int(always_on.sum()), "always_off": int(always_off.sum()),
                      "energy_share_always_on": share(always_on.float()), "energy_share_always_off": share(always_off.float()),
                      "energy_share_switching": share((~always_on & ~always_off).float()),
                      "per_token_in_zone_lt1": float((b.abs() < 1).float().sum(1).mean()),
                      "per_token_in_zone_lt0.25": float((b.abs() < 0.25).float().sum(1).mean()),
                      "frac_outside_4": float((b.abs() > 4).float().mean()), "frac_outside_6": float((b.abs() > 6).float().mean()),
                      "max_abs_gate": float(b.abs().max()),
                      "sign_pattern_participation_ratio": float(sv.sum() ** 2 / (sv ** 2).sum()),
                      "sign_pattern_rank90": rank_for(sv, 0.9)}
        # surrogates
        R["surrogate"] = {"reglu": rel(teacher_f(xte, G, U, O, torch.relu), yte)}
        for d_ in (0.5, 0.25):
            R["surrogate"][f"hinge_T4_delta{d_}"] = rel(teacher_f(xte, G, U, O, Hinge(4.0, d_, torch.float32)), yte)
        # quadratic compressibility
        qterm_tr = (xtr @ U.T) * (xtr @ G.T)
        Qtr, Qte = qterm_tr @ O.T, ((xte @ U.T) * b) @ O.T
        e_j = qterm_tr.pow(2).mean(0) * O.pow(2).sum(0)
        order = e_j.argsort(descending=True)
        sq = torch.linalg.svdvals(Qte - Qte.mean(0)) ** 2
        R["quadratic"] = {"output_rank90": rank_for(sq, 0.9), "teacher_terms": {}, "cp_fit": {}}
        for r in (32, 64, 128, 256, 512, 1024):
            j = order[:r]
            ftr, fte = qterm_tr[:, j], ((xte @ U[j].T) * (xte @ G[j].T))
            Wq = torch.linalg.lstsq(ftr.double(), Qtr.double()).solution.float()
            R["quadratic"]["teacher_terms"][r] = rel(fte @ Wq, Qte)
    for r in (64, 128, 256):
        j = order[:r]
        P1, P2 = G[j].clone(), U[j].clone()
        with torch.no_grad():
            P3 = torch.linalg.lstsq(qterm_tr[:, j].double(), Qtr.double()).solution.float().T.contiguous()
        mod = torch.nn.Module()
        mod.P1, mod.P2, mod.P3 = (torch.nn.Parameter(t) for t in (P1, P2, P3))
        mod.forward = lambda z, m=mod: ((z @ m.P1.T) * (z @ m.P2.T)) @ m.P3.T
        R["quadratic"]["cp_fit"][r] = train(mod, xtr, Qtr, xte, Qte, args.steps)
    # jacobians
    with torch.no_grad():
        idx = torch.linspace(0, len(xte) - 1, args.jac_tokens).long()
        evx, Vx = torch.linalg.eigh(cov.double())
        k90 = rank_for(evx.flip(0).float(), 0.9)
        Pk = Vx[:, -k90:].float()
        Aslope = W[:-1].T  # y = x W[:-1] + c  ->  J_affine = W[:-1]^T
        out = {"J": [], "J_minus_affine": [], "J_on_input_pcs": []}
        for t in idx:
            xt = xte[t]
            a_, b_ = U @ xt, G @ xt
            hp = 0.5 * (1 + torch.erf(b_ / math.sqrt(2))) + b_ * torch.exp(-0.5 * b_ * b_) / math.sqrt(2 * math.pi)
            J = O @ (gelu(b_)[:, None] * U + (a_ * hp)[:, None] * G)
            for name, M in (("J", J), ("J_minus_affine", J - Aslope), ("J_on_input_pcs", J @ Pk)):
                s2 = torch.linalg.svdvals(M) ** 2
                out[name].append((rank_for(s2, 0.9), float((s2[168:].sum() / s2.sum()).sqrt()), float(s2.sum().sqrt())))
        R["jacobian"] = {"input_pca_rank90": k90}
        for name, v in out.items():
            R["jacobian"][name] = {"rank90_median": float(torch.tensor([u[0] for u in v]).float().median()),
                                   "best_rank168_rel_err_median": float(torch.tensor([u[1] for u in v]).median()),
                                   "fro_norm_median": float(torch.tensor([u[2] for u in v]).median())}
    # regions: sign-region quadratics along a tree of gate predicates
    with torch.no_grad():
        imp = e_j * (pon * (1 - pon))  # switching gates with large quadratic terms
        split = imp.argsort(descending=True)[:args.region_depth]
        bte = b
        R["regions"] = []
        full_on_tr = btr > 0
        for depth in range(args.region_depth + 1):
            gates = split[:depth]
            code_tr = (btr[:, gates] > 0).long() @ (2 ** torch.arange(depth)) if depth else torch.zeros(len(xtr), dtype=torch.long)
            code_te = (bte[:, gates] > 0).long() @ (2 ** torch.arange(depth)) if depth else torch.zeros(len(xte), dtype=torch.long)
            err_num, sizes, diff_e, diff_r = 0.0, [], [], []
            for v in code_te.unique():
                tr_m, te_m = code_tr == v, code_te == v
                if tr_m.sum() < 32 or te_m.sum() < 8:
                    continue
                S = full_on_tr[tr_m].float().mean(0) > 0.5
                xv = xte[te_m]
                Qv = ((xv @ U[S].T) * (xv @ G[S].T)) @ O[:, S].T
                err_num += float((Qv - yte[te_m]).pow(2).sum())
                sizes.append(int(S.sum()))
                if depth:
                    parent = code_tr % (2 ** (depth - 1)) == int(v) % (2 ** (depth - 1))
                    Sp = full_on_tr[parent].float().mean(0) > 0.5
                    dS = S ^ Sp
                    sign = (S.float() - Sp.float())[dS]
                    Dv = ((xv @ U[dS].T) * (xv @ G[dS].T) * sign) @ O[:, dS].T if dS.any() else torch.zeros_like(Qv)
                    diff_e.append(float(Dv.pow(2).sum() / yte[te_m].pow(2).sum()))
                    if dS.sum() > 1 and len(xv) > 4:
                        s2 = torch.linalg.svdvals(Dv - Dv.mean(0)) ** 2
                        diff_r.append(rank_for(s2, 0.9) if s2.sum() > 0 else 0)
                    else:
                        diff_r.append(int(dS.sum()))
            R["regions"].append({"depth": depth, "rel_err": err_num / float(yte.pow(2).sum()),
                                 "mean_on_set": float(torch.tensor(sizes).float().mean()) if sizes else 0,
                                 "diff_energy_median": float(torch.tensor(diff_e).median()) if diff_e else 0,
                                 "diff_output_rank90_median": float(torch.tensor(diff_r).float().median()) if diff_r else 0})
        R["regions_oracle_sign_pattern"] = R["surrogate"]["reglu"]
    # families at equal compute
    fam = {}
    Wr = rrr(W, xtr, 128)
    with torch.no_grad():
        fam["affine_r128"] = {"rel": rel(affine_apply(Wr, xte), yte), "macs": 2 * 1024 * 128, "params": 2 * 1024 * 128 + 1024}
    e_neuron = (((xtr @ U.T) * gelu(btr)).pow(2).mean(0) * O.pow(2).sum(0))
    norder = e_neuron.argsort(descending=True)
    models = {}
    for m_ in (110, 256):
        j = norder[:m_]
        models[f"dense_geglu_{m_}"] = DenseGeGLU(G[j], U[j], O[:, j])
    jq = order[:64]
    with torch.no_grad():
        resid = ytr - affine_apply(Wr, xtr)
        P3 = torch.linalg.lstsq(qterm_tr[:, jq].double(), resid.double()).solution.float().T.contiguous()
    models["affine_r128_quad64"] = AffineQuad(Wr, 128, G[jq], U[jq], P3)
    models["omega_lite_d6_k4"] = OmegaLite(Wr, 128, G[jq], U[jq], P3, G[split[:6]], 4)
    for name, mod in models.items():
        fam[name] = {"rel": train(mod, xtr, ytr, xte, yte, args.steps), "macs": int(mod.macs()),
                     "params": int(sum(p.numel() for p in mod.parameters()))}
    fam["teacher"] = {"rel": 0.0, "macs": 3 * 1024 * 2624, "params": 3 * 1024 * 2624}
    R["families"] = fam
    return R, models, Wr


def e2e(m, seqs, layer, surrogates, n_seq=8, seed=0):
    """KL(teacher || spliced) etc. at 15% masked positions of held-out sequences, one layer's MLP replaced."""
    g = torch.Generator().manual_seed(seed)
    ids = seqs[-n_seq:].clone()
    sel = (torch.rand(ids.shape, generator=g) < 0.15)
    sel[:, 0] = False
    labels = ids[sel]
    ids[sel] = MASK
    mlp = m.model.layers[layer].mlp
    orig = mlp.forward
    out = {}
    with torch.no_grad():
        tl = F.log_softmax(m(input_ids=ids).logits[sel].float(), -1)
        out["teacher_ce"] = float(F.nll_loss(tl, labels))
        for name, sur in surrogates.items():
            mlp.forward = lambda h, s=sur: s(h.reshape(-1, h.shape[-1])).reshape(h.shape)
            sl = F.log_softmax(m(input_ids=ids).logits[sel].float(), -1)
            out[name] = {"kl": float(F.kl_div(sl, tl, log_target=True, reduction="batchmean")),
                         "top1_agree": float((sl.argmax(-1) == tl.argmax(-1)).float().mean()),
                         "ce": float(F.nll_loss(sl, labels))}
        mlp.forward = orig
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, nargs="+", default=[4, 10, 14, 17, 24])
    p.add_argument("--train", type=int, default=32768)
    p.add_argument("--test", type=int, default=8192)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--jac-tokens", type=int, default=24)
    p.add_argument("--region-depth", type=int, default=8)
    p.add_argument("--e2e-layers", type=int, nargs="*", default=None, help="layers for the splice test (default: all)")
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    data, m, seqs = collect(a.layers, a.train + a.test + 8 * 512)
    res = {"args": vars(a), "layers": {}}
    for i in a.layers:
        t0 = time.time()
        x, y = data[i]["x"][: a.train + a.test], data[i]["y"][: a.train + a.test]
        G, U, O = weights(m, i)
        R, models, Wr = analyse(i, x, y, G, U, O, a.train, a)
        if a.e2e_layers is None or i in a.e2e_layers:
            surr = {"affine_r128": lambda h, W=Wr: affine_apply(W, h)}
            surr.update({k: v for k, v in models.items()})
            R["e2e"] = e2e(m, seqs, i, surr)
        R["seconds"] = time.time() - t0
        res["layers"][i] = R
        print(json.dumps(R, indent=None)[:4000], flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
