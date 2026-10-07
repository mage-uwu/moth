# Golden Tree Snake (GTS) fork, 2026.
"""GTS-Uni-Sys1: a GTS (or GTS-Uni) encoder turned into a System One decision model with Laya's technique, on the
LocalLLaMA typed-decisions benchmark.

    python scripts/sys1_train.py --gts uni_binarized.pt --loops 3 --general 500000 --out results/sys1_uni.json
    python scripts/sys1_train.py --hf answerdotai/ModernBERT-base --general 500000 --out results/sys1_modernbert.json

Following Laya (convaiinnovations/laya, laya-typed-decisions model cards):
- **Every option is scored at its own [MASK] token**, then softmaxed over that question's options: "[CLS] question
  [SEP] state [SEP] option: description [MASK] option: description [MASK] ... [SEP]", options in a fixed 256-token
  head budget at the end so truncation only shortens the state, 512 tokens in all.
- **A decision head trained from scratch on top of the encoder: 2 layers, an option-marker scorer, and an
  act/escalate head.** Laya's 2 layers are transformer layers; here they are GTS blocks for a GTS encoder (the model
  stays attention-free and CPU-fast) and transformer layers for a Hugging Face baseline, as in Laya. The act/escalate
  head reads [CLS] after the head layers and predicts whether the decision's top answer is right (act) or should go to
  a human (escalate); Laya does not document its target, this is the natural one.
- **RLCD**: the policy reports a distribution; exploration adds zero-mean Gaussian noise to the logits; the reward is a
  strictly proper scoring rule, log + spherical, plus the ranked probability score for ordinal (score) questions, taken
  in expectation over the gold distribution; REINFORCE with a group-mean baseline over the noise samples of each
  decision (GRPO-style); **alongside soft cross-entropy against the teacher's distributions** (as laya-typed-decisions).
- **Temperatures per (question type, option count)**, fitted on held-out decisions (Laya fitted them on training data).
Stages, as Laya: general typed decisions first (tasksource-jev-typed-decisions, benchmark workflows left out; scored
zero-shot), then the benchmark's 1,200 training cases (scored again: the fitted regime). Not done: Laya's multi-turn
TD(lambda) and its script router.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from decide_probe import bench_decisions, general_decisions, references, scores  # noqa: E402
from mamba_ssm.models.gts_encoder import GTSBlock, GTSConfig, GTSForMaskedLM, RMSNorm  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized  # noqa: E402


class Sys1(torch.nn.Module):
    def __init__(self, gts=None, hf=None, loops=None, head_layers=2, head_deep_depth=6):
        super().__init__()
        self.loops = loops
        if gts:
            blob = torch.load(gts, map_location="cpu", weights_only=False)
            cfg = blob["config"]
            lm = GTSForMaskedLM(GTSConfig(**cfg))
            if "model" in blob:
                lm.load_state_dict(blob["model"])
            else:
                load_binarized(blob, lm)
            self.backbone, self.hf, d = lm.backbone, None, cfg["d_model"]
            from bert_pretrain import _tokenizer

            tok = _tokenizer()
            self.enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
            self.cls, self.sep, self.mask_id, self.pad = 101, 102, 103, 0
            hcfg = GTSConfig(**{k: v for k, v in cfg.items() if k not in ("loops", "latent_tokens")})
            hcfg.deep_depth, hcfg.n_layer = head_deep_depth, head_layers
            self.layers = torch.nn.ModuleList([GTSBlock(hcfg, i) for i in range(head_layers)])
            self.norm = RMSNorm(d)
        else:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(hf)
            self.backbone, self.hf = AutoModel.from_pretrained(hf), hf
            d = self.backbone.config.hidden_size
            self.enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
            self.cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
            self.sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
            self.mask_id, self.pad = tok.mask_token_id, tok.pad_token_id or 0
            self.layers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, 0.1, "gelu",
                                                                                batch_first=True, norm_first=True)
                                               for _ in range(head_layers)])
            self.norm = torch.nn.LayerNorm(d)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 1))
        self.escalate = torch.nn.Sequential(torch.nn.Linear(d, d // 4), torch.nn.GELU(), torch.nn.Linear(d // 4, 1))

    def encode(self, ex, max_len, head_budget=256):
        """Ids and the [MASK] position of every option (tracked while building, so a literal "[MASK]" in a state cannot
        be mistaken for one). Options go last, within ``head_budget`` tokens."""
        if len(ex["options"]) < 2 or len(ex["target"]) != len(ex["options"]) or sum(ex["target"]) <= 0:
            return None  # not a decision: skipped
        q = self.enc(ex["question"])[:64]
        tail, per = [], max(4, head_budget // max(1, len(ex["options"])) - 1)
        for i, o in enumerate(ex["options"]):
            desc = ex["descriptions"][i] if ex["descriptions"] else None
            tail.append(self.enc(f"{o}: {desc}" if desc else str(o))[:per])
        n_tail = sum(len(t) + 1 for t in tail) + 1
        room = max(8, max_len - 3 - len(q) - n_tail)
        ids = [self.cls] + q + [self.sep] + self.enc(ex["state"])[:room] + [self.sep]
        marks = []
        for t in tail:
            ids += t
            marks.append(len(ids))
            ids.append(self.mask_id)
        ids.append(self.sep)
        if len(ids) > max_len:
            return None
        return ids, marks

    def forward(self, ids, mask, rows, cols, n_opts):
        if self.hf:
            h = self.backbone(input_ids=ids, attention_mask=mask.long()).last_hidden_state
            for layer in self.layers:
                h = layer(h, src_key_padding_mask=~mask)
        else:
            h = self.backbone(ids, attention_mask=mask, loops=self.loops)
            for layer in self.layers:
                h = layer(h, attention_mask=mask)
        h = self.norm(h)
        logits = self.scorer(h[rows, cols]).squeeze(-1).float()
        act = self.escalate(h[:, 0]).squeeze(-1).float()
        return torch.split(logits, n_opts), act


def batches(model, data, a, device, shuffle=False, gen=None):
    enc = [model.encode(ex, a.max_len) for ex in data]
    keep = [i for i, e in enumerate(enc) if e is not None]
    if shuffle:
        keep = [keep[i] for i in torch.randperm(len(keep), generator=gen).tolist()]
    for s in range(0, len(keep), a.batch_size):
        idx = keep[s : s + a.batch_size]
        L = -(-max(len(enc[i][0]) for i in idx) // 64) * 64
        ids = torch.full((len(idx), L), model.pad, dtype=torch.long)
        rows, cols, n_opts, targets = [], [], [], []
        for r, i in enumerate(idx):
            t, m = enc[i]
            ids[r, : len(t)] = torch.tensor(t)
            rows += [r] * len(m)
            cols += m
            n_opts.append(len(m))
            g = torch.tensor(data[i]["target"], dtype=torch.float32)
            targets.append(g / g.sum().clamp(min=1e-9))
        yield (idx, ids.to(device), (ids != model.pad).to(device), torch.tensor(rows, device=device),
               torch.tensor(cols, device=device), n_opts, targets)


def label_index(ex):
    return ex["options"].index(ex["label"]) if ex.get("label") in ex["options"] else int(np.argmax(ex["target"]))


def proper_reward(q, g, ordinal):
    """Expected strictly proper score of the reported distributions q (k, n) under the gold g (n,): log + spherical,
    plus the (negated) ranked probability score for an ordinal question."""
    r = (g * torch.log(q.clamp(min=1e-9))).sum(-1) + (q @ g) / q.norm(dim=-1).clamp(min=1e-9)
    if ordinal:
        r = r - ((q.cumsum(-1) - g.cumsum(-1)) ** 2).sum(-1)
    return r


def train(model, data, a, device, epochs, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=device == "cuda")
    steps = max(1, epochs * math.ceil(len(data) / a.batch_size))
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    g, t0, step = torch.Generator().manual_seed(a.seed), time.time(), 0
    model.train()
    for ep in range(epochs):
        for idx, ids, mask, rows, cols, n_opts, targets in batches(model, data, a, device, True, g):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                outs, act = model(ids, mask, rows, cols, n_opts)
            ce = rl = 0.0
            correct = []
            for z, t, i in zip(outs, targets, idx):
                t = t.to(device)
                ce = ce - (t * F.log_softmax(z, -1)).sum()
                # RLCD: noisy reports a = z + eps, rewarded by a proper scoring rule; REINFORCE, group-mean baseline
                eps = torch.randn(a.rl_samples, z.numel(), device=device) * a.rl_sigma
                act_logits = z.detach() + eps
                reward = proper_reward(F.softmax(act_logits, -1), t, data[i]["type"] == "score")
                adv = reward - reward.mean()
                logp = -((act_logits - z) ** 2).sum(-1) / (2 * a.rl_sigma ** 2)
                rl = rl - (adv.detach() * logp).mean()
                correct.append(float(int(z.argmax()) == label_index(data[i])))
            n = len(outs)
            esc = F.binary_cross_entropy_with_logits(act, torch.tensor(correct, device=device))
            loss = ce / n + a.rl_weight * rl / n + a.esc_weight * esc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % a.log_every == 0:
                print(f"  epoch {ep + 1} step {step}/{steps}  ce {ce.item() / n:.4f}  rl {rl.item() / n:.4f}  escalate {esc.item():.4f}"
                      f"  {(time.time() - t0) / 60:.1f} min", flush=True)
    model.eval()


@torch.no_grad()
def raw_predict(model, data, a, device):
    logits, acts = [None] * len(data), [None] * len(data)
    for idx, ids, mask, rows, cols, n_opts, _ in batches(model, data, a, device):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            outs, act = model(ids, mask, rows, cols, n_opts)
        for j, (i, z) in enumerate(zip(idx, outs)):
            logits[i], acts[i] = z.float().cpu().numpy(), float(torch.sigmoid(act[j]))
    return logits, acts


def fit_temperatures(logits, data):
    """One temperature per (question type, option count), minimising the log loss on held-out decisions; a key with
    fewer than 20 decisions uses the pooled temperature."""
    def best(sel):
        grid, out = np.exp(np.linspace(-1.5, 1.5, 61)), (1e18, 1.0)
        for T in grid:
            nll = 0.0
            for i in sel:
                z = logits[i] / T
                z = z - z.max()
                lq = z - np.log(np.exp(z).sum())
                g = np.asarray(data[i]["target"], dtype=np.float64)
                nll -= float((g / g.sum() * lq).sum())
            out = min(out, (nll, T))
        return float(out[1])

    ok = [i for i, z in enumerate(logits) if z is not None]
    temps = {"*": best(ok)}
    keys = {}
    for i in ok:
        keys.setdefault(f"{data[i]['type']}/{len(data[i]['options'])}", []).append(i)
    for k, sel in keys.items():
        temps[k] = best(sel) if len(sel) >= 20 else temps["*"]
    return temps


def apply(logits, data, temps):
    out = []
    for z, ex in zip(logits, data):
        if z is None:
            out.append(None)
            continue
        T = temps.get(f"{ex['type']}/{len(ex['options'])}", temps["*"])
        z = z / T
        e = np.exp(z - z.max())
        out.append(e / e.sum())
    return out


def evaluate(model, test, cal, a, device):
    lc, _ = raw_predict(model, cal, a, device)
    temps = fit_temperatures(lc, cal)
    lt, acts = raw_predict(model, test, a, device)
    probs = apply(lt, test, temps)
    keep = [i for i, p in enumerate(probs) if p is not None]
    res = {"temperatures": temps, "scores": scores([probs[i] for i in keep], [test[i] for i in keep]),
           "dropped": len(test) - len(keep)}
    # act / escalate: how well the head ranks right answers above wrong ones, and accuracy when acting on the top X%
    right = np.array([int(np.argmax(probs[i])) == label_index(test[i]) for i in keep], dtype=float)
    act = np.array([acts[i] for i in keep])
    order = np.argsort(-act)
    pos, neg = right.sum(), len(right) - right.sum()
    ranks = np.argsort(np.argsort(act)) + 1
    res["escalate"] = {"auroc": float((ranks[right == 1].sum() - pos * (pos + 1) / 2) / max(1, pos * neg)),
                       **{f"accuracy_acting_on_top_{c}%": float(right[order[: max(1, int(len(order) * c / 100))]].mean())
                          for c in (50, 80, 100)}}
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gts")
    src.add_argument("--hf")
    p.add_argument("--loops", type=int, help="GTS-Uni passes")
    p.add_argument("--general", type=int, default=500000)
    p.add_argument("--commercial-only", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--fit-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-layers", type=int, default=2)
    p.add_argument("--rl-weight", type=float, default=1.0)
    p.add_argument("--rl-samples", type=int, default=8, help="noise samples per decision (the group of the baseline)")
    p.add_argument("--rl-sigma", type=float, default=0.5, help="exploration noise on the logits")
    p.add_argument("--esc-weight", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--max-test", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    model = Sys1(a.gts, a.hf, a.loops, a.head_layers).to(device)
    bench_train, test = bench_decisions("train"), bench_decisions("test")
    if a.max_test:
        test = test[: a.max_test]
    cut = int(0.9 * len(bench_train))  # the last 10% of the benchmark's train is held out to calibrate the fitted model
    res = {"model": a.gts or a.hf, "args": vars(a), "references": references(bench_train, test)}
    t0 = time.time()
    if a.general:
        gen = general_decisions(a.general, a.seed, a.commercial_only)
        print(f"general typed decisions: {len(gen):,}", flush=True)
        train(model, gen[:-2000], a, device, a.epochs, a.lr)
        res["zero_shot"] = evaluate(model, test, gen[-2000:], a, device)
        r = res["zero_shot"]["scores"]["all"]
        print(f"zero-shot: accuracy {r['accuracy']:.3f}  KL {r['kl']:.3f}  Brier {r['brier']:.3f}  ECE {r['ece']:.3f}  "
              f"escalate AUROC {res['zero_shot']['escalate']['auroc']:.3f}", flush=True)
    train(model, bench_train[:cut], a, device, a.fit_epochs, a.lr)
    res["fitted"] = evaluate(model, test, bench_train[cut:], a, device)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(a.out, "w"), indent=1)
    r = res["fitted"]["scores"]
    print(f"fitted: accuracy {r['all']['accuracy']:.3f}  KL {r['all']['kl']:.3f}  Brier {r['all']['brier']:.3f}  "
          f"ECE {r['all']['ece']:.3f}  escalate AUROC {res['fitted']['escalate']['auroc']:.3f}  | by type: "
          + "  ".join(f"{t} {v['accuracy']:.3f}" for t, v in r.items() if t != "all"), flush=True)


if __name__ == "__main__":
    main()
