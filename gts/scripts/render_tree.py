# Golden Tree Snake (GTS) fork, 2026.
"""Render one of a trained GTS masked LM's deep trees as a radial drawing: every node of the tree at its real position
(root at the centre, one ring per level), each edge drawn as bright as the number of real tokens that took that branch.

    python scripts/render_tree.py --ckpt binarized.pt --layer 7 --tree 0 --out assets/gts_tree.png

The traffic is measured, not made up: text goes through the model, a hook catches the deep mixer's input in the chosen
layer, and the tree walk (GTS._walk, the same hard routing the model uses) records which nodes each token visits.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized  # noqa: E402


def sentences(n):
    from datasets import load_dataset

    d = load_dataset("stanfordnlp/sst2", split="train").shuffle(seed=0).select(range(n))
    return [s.strip() for s in d["sentence"]]


@torch.no_grad()
def traffic(model, layer, texts, seq=128):
    from bert_pretrain import _tokenizer

    tok = _tokenizer()
    deep = model.backbone.layers[layer].mixer.deep
    counts = torch.zeros(deep.n_trees * deep.n_nodes)
    caught = {}
    h = deep.register_forward_pre_hook(lambda m, args, kw: caught.__setitem__("u", (args[0], kw.get("attention_mask"))), with_kwargs=True)
    ids = [101]
    for t in texts:
        ids += tok.encode(t, add_special_tokens=False).ids + [102]
    n_tok = 0
    for s in range(0, len(ids) - seq, seq):
        x = torch.tensor([ids[s : s + seq]])
        model.backbone(x)
        u, mask = caught["u"]
        mask = torch.ones(u.shape[:2]) if mask is None else mask.float()
        nodes, _ = deep._walk(deep._local_mix(u, mask))
        counts += torch.bincount(nodes.flatten(), minlength=counts.numel()).float()
        n_tok += seq
    h.remove()
    return counts.view(deep.n_trees, deep.n_nodes), n_tok, deep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--tree", type=int, default=0)
    p.add_argument("--sentences", type=int, default=3000)
    p.add_argument("--name", default="GTS3")
    p.add_argument("--out", default="assets/gts_tree.png")
    a = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GTSForMaskedLM(GTSConfig(**blob["config"]))
    load_binarized(blob, model) if "ternary" in blob else model.load_state_dict(blob["model"])
    model.eval()
    counts, n_tok, deep = traffic(model, a.layer, sentences(a.sentences))
    c = counts[a.tree].numpy()
    n = len(c)
    levels = int(math.log2(n + 1))

    def pos(i):
        lv = int(math.log2(i + 1))
        j = i + 1 - 2 ** lv
        ang = 2 * math.pi * (j + 0.5) / 2 ** lv + math.pi / 2
        r = lv ** 0.92  # rings slightly closer outwards: the leaves have room, the centre stays open
        return r * math.cos(ang), r * math.sin(ang)

    xy = np.array([pos(i) for i in range(n)])
    segs, widths, alphas = [], [], []
    peak = c[1:].max()
    for i in range(1, n):
        par = (i - 1) // 2
        w = (c[i] / peak) ** 0.5 if peak else 0
        # a gentle curve from parent to child: through the point at the parent's radius and the child's angle
        p0, p2 = xy[par], xy[i]
        r0 = math.hypot(*p0)
        ang = math.atan2(p2[1], p2[0])
        p1 = (r0 * math.cos(ang), r0 * math.sin(ang)) if r0 > 0 else p0
        t = np.linspace(0, 1, 12)[:, None]
        curve = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)
        segs.append(curve)
        widths.append(0.15 + 2.6 * w)
        alphas.append(0.07 + 0.93 * w)
    fig = plt.figure(figsize=(12, 12), facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    lc = LineCollection(segs, linewidths=widths, colors=[(1, 1, 1, al) for al in alphas], capstyle="round")
    ax.add_collection(lc)
    vis = c > 0
    ax.scatter(xy[vis, 0], xy[vis, 1], s=0.5 + 14 * (c[vis] / peak) ** 0.5, c="white", linewidths=0, zorder=3)
    ax.scatter(xy[~vis, 0], xy[~vis, 1], s=0.6, c=(1, 1, 1, 0.25), linewidths=0, zorder=2)
    R = (levels - 1) ** 0.92 + 0.6
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.axis("off")
    used = int(vis.sum())
    fig.text(0.5, 0.025, f"{a.name}  ·  layer {a.layer + 1} of {len(model.backbone.layers)}  ·  deep tree {a.tree + 1} of "
             f"{deep.n_trees}  ·  {n:,} nodes, {used:,} visited  ·  line weight = tokens routed ({n_tok:,} tokens of real text)",
             color=(1, 1, 1, 0.55), ha="center", fontsize=9, family="monospace")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="black")
    print(f"wrote {a.out}: {used} of {n} nodes visited by {n_tok} tokens")


if __name__ == "__main__":
    main()
