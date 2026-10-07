# Golden Tree Snake (GTS) fork, 2026.
"""How evenly a GTS masked LM's deep trees spread real text over their nodes: per level, the share of nodes never
visited and the usage perplexity over the nodes at that level (1.0 = perfectly even), averaged over layers and trees.

    python scripts/tree_usage.py checkpoint.pt [--tokens 32768]
"""
import math, os, sys, torch, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM
from mamba_ssm.utils.ternary_pack import load_binarized
from mamba_ssm.utils.ternary_pack import pack_ternary
from bert_pretrain import _tokenizer
from datasets import load_dataset
blob = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = GTSForMaskedLM(GTSConfig(**blob["config"]))
load_binarized(blob, m) if "ternary" in blob else m.load_state_dict(blob["model"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = m.to(dev).eval()
tok = _tokenizer()
texts = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
ids = [101]
for ex in texts:
    ids += tok.encode(ex["text"], add_special_tokens=False).ids + [102]
    if len(ids) > 64 * 512: break
L = len(m.backbone.layers); caught = {}
hooks = [m.backbone.layers[i].mixer.deep.register_forward_pre_hook(lambda mod, a, kw, i=i: caught.__setitem__(i, (a[0], kw.get("attention_mask"))), with_kwargs=True) for i in range(L)]
deep0 = m.backbone.layers[0].mixer.deep
counts = torch.zeros(L, deep0.n_trees * deep0.n_nodes)
with torch.no_grad():
    for s in range(0, 64 * 512, 512):
        m.backbone(torch.tensor([ids[s:s + 512]], device=dev))
        for i in range(L):
            u, mask = caught[i]; mask = torch.ones(u.shape[:2]) if mask is None else mask.float()
            dp = m.backbone.layers[i].mixer.deep
            nodes, _ = dp._walk(dp._local_mix(u, mask))
            counts[i] += torch.bincount(nodes.flatten().cpu(), minlength=counts.shape[1]).float()
n_tok = 64 * 512; depth = deep0.depth; nn_ = deep0.n_nodes
print(f"{n_tok} Wikipedia tokens, {L} layers x {deep0.n_trees} trees x {nn_} nodes (depth {depth})")
print("per level: share of nodes never visited | usage perplexity / nodes at that level (1.0 = perfectly balanced)")
lv_of = np.array([int(math.log2(j + 1)) for j in range(nn_)])
dead_tot = 0; tab = np.zeros((L, depth + 1, 2))
for i in range(L):
    c = counts[i].view(deep0.n_trees, nn_).numpy()
    for lv in range(depth + 1):
        sel = c[:, lv_of == lv]
        dead = (sel == 0).mean()
        ppl = []
        for t in range(sel.shape[0]):
            p = sel[t] / sel[t].sum(); p = p[p > 0]
            ppl.append(math.exp(-(p * np.log(p)).sum()) / sel.shape[1])
        tab[i, lv] = dead, np.mean(ppl)
    dead_tot += (c == 0).sum()
for lv in range(depth + 1):
    print(f"  level {lv} ({2**lv:4d} nodes): dead {100*tab[:, lv, 0].mean():5.1f}%  balance {tab[:, lv, 1].mean():.2f}  (worst layer {tab[:, lv, 1].min():.2f})")
print("per layer: dead nodes %, leaf balance:", " ".join(f"L{i}:{100*(counts[i]==0).float().mean():.0f}%/{tab[i, depth, 1]:.2f}" for i in range(L)))
print(f"overall dead nodes: {100*dead_tot/counts.numel():.1f}%")
# weights of dead vs live nodes (ternary codes of the node input rows)
dead_nz, live_nz = [], []
for i in range(L):
    dp = m.backbone.layers[i].mixer.deep
    w = dp._w_in().detach() if hasattr(dp, "_w_in") else None
    if w is None: break
    nz = (w != 0).float().mean(-1).cpu().numpy()
    c = counts[i].numpy()
    dead_nz += list(nz[c == 0]); live_nz += list(nz[c > 0])
print(f"nonzero share of node input weights: dead nodes {np.mean(dead_nz):.3f}, live nodes {np.mean(live_nz):.3f}")
