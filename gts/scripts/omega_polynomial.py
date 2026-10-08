# Golden Tree Snake (GTS) fork, 2026.
"""GTS-OMEGA addendum: is the teacher MLP a low-degree polynomial on its own data?

With b = Gx mostly near 0, h(b) = b Phi(b) = b/2 + phi0 b^2 - phi0 b^4 / 6 + phi0 b^6 / 40 - ... (phi0 = 1/sqrt(2 pi)),
so f = Q/2 + phi0 C3 - ... with Q = sum_j o_j a_j b_j (degree 2), C3 = sum_j o_j a_j b_j^2 (degree 3), etc. For each
layer: the error of these truncations (the global smooth expansion), the fraction of gate values where the series is
accurate, and the cost-free structure question it raises: do the degree-2 and degree-3 terms compress (top-r teacher
terms with a least-squares output, as in omega_diagnostics.py)?

    python scripts/omega_polynomial.py --layers 4 10 14 17 24 --tokens 12288 --out results/omega/polynomial.json
"""
import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from omega_diagnostics import collect, rel, teacher_f, weights  # noqa: E402

PHI0 = 1 / math.sqrt(2 * math.pi)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, nargs="+", default=[4, 10, 14, 17, 24])
    p.add_argument("--tokens", type=int, default=12288)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(os.cpu_count())
    data, m, _ = collect(a.layers, a.tokens)
    res = {}
    for i in a.layers:
        x, y = data[i]["x"], data[i]["y"]
        G, U, O = weights(m, i)
        n = len(x) // 2
        with torch.no_grad():
            A, B = x @ U.T, x @ G.T
            R = {"teacher_check": rel(teacher_f(x, G, U, O), y)}
            series = {"deg2 (b/2)": B / 2, "deg3 (+phi0 b^2)": B / 2 + PHI0 * B ** 2,
                      "deg5 (-phi0 b^4/6)": B / 2 + PHI0 * (B ** 2 - B ** 4 / 6),
                      "deg7 (+phi0 b^6/40)": B / 2 + PHI0 * (B ** 2 - B ** 4 / 6 + B ** 6 / 40)}
            R["truncation_rel_err"] = {k: rel((A * v) @ O.T, y) for k, v in series.items()}
            R["frac_gate_abs_lt"] = {t: float((B.abs() < t).float().mean()) for t in (0.5, 1.0, 1.5, 2.0)}
            # compressibility of the degree-2 and degree-3 parts, teacher terms ranked by energy, output refit
            for name, feat in (("quadratic", A * B), ("cubic", A * B ** 2)):
                tgt = feat @ O.T
                e = feat[:n].pow(2).mean(0) * O.pow(2).sum(0)
                order = e.argsort(descending=True)
                R[f"{name}_top_terms"] = {}
                for r in (64, 256, 1024):
                    j = order[:r]
                    W = torch.linalg.lstsq(feat[:n, j].double(), tgt[:n].double()).solution.float()
                    R[f"{name}_top_terms"][r] = rel(feat[n:, j] @ W, tgt[n:])
            # degree-3 polynomial with the teacher's own terms vs the dense teacher at equal kept-neuron budgets:
            # f_r = sum over the top-r neurons of o_j a_j (b_j/2 + phi0 b_j^2), output refit
            feat3 = A * (B / 2 + PHI0 * B ** 2)
            featf = A * (B * 0.5 * (1 + torch.erf(B / math.sqrt(2))))
            e = featf[:n].pow(2).mean(0) * O.pow(2).sum(0)
            order = e.argsort(descending=True)
            R["neuron_budget"] = {}
            for r in (110, 256, 512):
                j = order[:r]
                W3 = torch.linalg.lstsq(feat3[:n, j].double(), y[:n].double()).solution.float()
                Wf = torch.linalg.lstsq(featf[:n, j].double(), y[:n].double()).solution.float()
                R["neuron_budget"][r] = {"poly3": rel(feat3[n:, j] @ W3, y[n:]), "gelu": rel(featf[n:, j] @ Wf, y[n:])}
        res[i] = R
        print(i, json.dumps(R)[:1500], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
