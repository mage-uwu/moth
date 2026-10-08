# Golden Tree Snake (GTS) fork, 2026.
"""How tree-shaped is the teacher's MLP? Ceilings for distilling ModernBERT-large's GeGLU MLPs into hard-routed trees.

On FineWeb-Edu text, for a few layers, it records each MLP's input x (after mlp_norm), its 2,624 neuron values
h_j = gelu(g_j . x) (u_j . x) and its output y = sum_j h_j o_j, and reports relative squared errors (the Stage 2
metric, ||y - y_hat||^2 / ||y||^2, on held-out tokens) of:

  mean      the mean output (how much of y is a constant)
  affine    the best affine map of x (ridge; what no routing at all can do)
  topk      per-token oracle: each token keeps its own k largest neuron terms h_j o_j (k = 10, 40, 160, 640). The tree
            touches 40 nodes per token (4 trees x depth 9 + 1), so top40 is the ceiling for any 40-neuron routing that
            reuses the teacher's own neurons
  region    the same 40-neuron budget with a fixed subset per region: k-means on x into 512 regions (the trees' leaf
            count), each region keeps the 40 neurons with the most output energy on its training tokens (a hard-routed
            "neuron transplant" without the hierarchy constraint)
  rank90    output principal components holding 90% of the output energy

    python scripts/mlp_geometry.py --tokens 32768 --layers 1 7 14 21 27 --out results/mlp_geometry.json
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

TEACHER = os.environ.get("MOHAWK_TEACHER", "answerdotai/ModernBERT-large")


def collect(n_tokens, layers, seq=512):
    from datasets import load_dataset
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEACHER)
    m = AutoModelForMaskedLM.from_pretrained(TEACHER, dtype=torch.float32, attn_implementation="sdpa").eval()
    rec = {i: {"x": [], "h": [], "y": []} for i in layers}
    def io_hook(i):
        def hook(mod, args, out):  # returns None: a forward hook's return value would replace the output
            rec[i]["x"].append(args[0].reshape(-1, args[0].shape[-1]))
            rec[i]["y"].append(out.reshape(-1, out.shape[-1]))
        return hook

    def h_hook(i):
        def hook(mod, args):
            rec[i]["h"].append(args[0].reshape(-1, args[0].shape[-1]))
        return hook

    for i in layers:
        mlp = m.model.layers[i].mlp
        mlp.register_forward_hook(io_hook(i))
        mlp.Wo.register_forward_pre_hook(h_hook(i))
    buf, n, t0 = [], 0, time.time()
    stream = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    with torch.no_grad():
        for doc in stream:
            buf += tok(doc["text"], add_special_tokens=False)["input_ids"] + [tok.sep_token_id]
            while len(buf) >= 8 * (seq - 1):
                ids = torch.tensor([[tok.cls_token_id] + buf[k * (seq - 1):(k + 1) * (seq - 1)] for k in range(8)])
                buf = buf[8 * (seq - 1):]
                m(input_ids=ids)
                n += ids.numel()
                print(f"  {n:,} tokens ({time.time() - t0:.0f} s)", flush=True)
                if n >= n_tokens:
                    return {i: {k: torch.cat(v) for k, v in r.items()} for i, r in rec.items()}, m
    raise RuntimeError("ran out of text")


def rel(a, b):
    return float((a - b).pow(2).sum() / b.pow(2).sum())


def kmeans(x, k, iters=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(x, c).argmin(1)
        for j in range(k):
            sel = a == j
            if sel.any():
                c[j] = x[sel].mean(0)
    return c


def analyse(x, h, y, Wo, k_list=(10, 40, 160, 640), regions=512, budget=40):
    n = len(x)
    tr, te = slice(0, int(0.75 * n)), slice(int(0.75 * n), n)
    out = {"mean": rel(y[tr].mean(0).expand_as(y[te]), y[te])}
    X = torch.cat([x[tr], torch.ones(len(x[tr]), 1)], 1)
    lam = 1e-3 * X.pow(2).sum() / len(X)
    W = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1]), X.T @ y[tr])
    out["affine"] = rel(torch.cat([x[te], torch.ones(len(x[te]), 1)], 1) @ W, y[te])
    on = Wo.norm(dim=0)  # (inter,): each neuron's output-vector norm
    score = h[te].abs() * on
    for k in k_list:
        idx = score.topk(k, dim=1).indices
        hk = torch.zeros_like(h[te]).scatter_(1, idx, h[te].gather(1, idx))
        out[f"top{k}"] = rel(hk @ Wo.T, y[te])
    c = kmeans(x[tr], regions)
    a_tr, a_te = torch.cdist(x[tr], c).argmin(1), torch.cdist(x[te], c).argmin(1)
    energy = torch.zeros(regions, h.shape[1]).index_add_(0, a_tr, (h[tr] * on).pow(2))
    keep = torch.zeros(regions, h.shape[1]).scatter_(1, energy.topk(budget, dim=1).indices, 1.0)
    out[f"region{budget}"] = rel((h[te] * keep[a_te]) @ Wo.T, y[te])
    s = torch.linalg.svdvals(y[tr] - y[tr].mean(0)).pow(2)
    out["rank90"] = int((s.cumsum(0) / s.sum() < 0.9).sum()) + 1
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=32768)
    p.add_argument("--layers", type=int, nargs="+", default=[1, 7, 14, 21, 27])
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--out")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    rec, m = collect(a.tokens, a.layers)
    res = {}
    for i in a.layers:
        r = rec[i]
        res[i] = analyse(r["x"], r["h"], r["y"], m.model.layers[i].mlp.Wo.weight.detach())
        print(f"layer {i:2d}: " + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in res[i].items()),
              flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"tokens": a.tokens, "layers": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
