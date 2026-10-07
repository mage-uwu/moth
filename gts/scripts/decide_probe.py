# Golden Tree Snake (GTS) fork, 2026.
"""Typed probabilistic decisions with a GTS encoder: the Laya / Jev "System One" regime, on the LocalLLaMA
typed-decisions benchmark (400 test cases, 2,000 decisions; leaderboard in its README).

    python scripts/decide_probe.py --gts binarized.pt --regime zero-shot --general 200000 --out results/decide_gts3.json
    python scripts/decide_probe.py --gts binarized.pt --regime fitted --out results/decide_gts3_fitted.json
    python scripts/decide_probe.py --hf answerdotai/ModernBERT-base --regime fitted --out ...      # a baseline, same harness

A decision is (state, question, options) -> a probability distribution over the options. The input is
"[CLS] question [SEP] state [SEP] <opt> option: description <opt> option: description ... [SEP]"; a small head scores
the encoder's state at every option marker, and a softmax over a decision's options gives its distribution. Training
minimises the cross-entropy against the gold distribution, the log score, a strictly proper scoring rule (Laya's
RLCD optimises proper scoring rules with a policy gradient; with full gold distributions the direct loss is the same
target). A temperature fitted on held-out decisions calibrates the output.

Regimes, as the benchmark separates them:
  zero-shot: trained on general typed decisions only (tasksource/tasksource-jev-typed-decisions, ``--general``
             of them, any source whose name matches a benchmark workflow left out), then scored on the benchmark's test;
             comparable with the benchmark's zero-shot table (Jev 1.13.0: 0.727).
  fitted:    (after zero-shot training if ``--general`` > 0) fitted on the benchmark's 1,200 training cases; comparable
             with its fitted table (Laya typed-decisions 0.766, ModernBERT-base 0.646).
Metrics: accuracy (argmax against the gold label), KL(gold || prediction), Brier (summed over options) and ECE (top-1
confidence, 10 bins), overall and per decision type, with the Uniform and Prior reference rows recomputed the same way.
KL and Brier reproduce the README's Uniform row exactly (0.444, 0.238) and accuracy up to tie-breaking (0.269 vs 0.308);
the README's ECE (0.169 for Uniform) could not be reproduced from its description, so ECE compares only within this
harness.
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
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized  # noqa: E402

WORKFLOWS = ("agent_trace_observability", "customer_service", "invoice_processing", "security_incidents")


# ------------------------------------------------------------------------------------------------------- data


def bench_decisions(split):
    """The benchmark as one decision per (case, question): dicts of state, question, options, descriptions, gold
    distribution, gold label, type."""
    from datasets import load_dataset

    out = []
    for case in load_dataset("LocalLLaMA/typed-decisions", "all", split=split):
        qs = json.loads(case["questions"]) if isinstance(case["questions"], str) else case["questions"]
        gold = json.loads(case["gold"]) if isinstance(case["gold"], str) else case["gold"]
        state = case["state"] if isinstance(case["state"], str) else json.dumps(case["state"])
        for name, q in qs.items():
            if name not in gold or q is None:
                continue
            crit = q.get("criteria") or {}
            if isinstance(crit, list):  # score questions: levels 0, 1, ... with a description each
                opts, descs = [str(i) for i in range(len(crit))], list(crit)
            else:
                opts, descs = list(crit.keys()), list(crit.values())
            probs = gold[name]["probabilities"]
            if not opts:  # no criteria given: the options are the gold distribution's keys
                opts, descs = list(probs.keys()), [None] * len(probs)
            out.append({"state": state, "question": q.get("instructions", name), "options": opts,
                        "descriptions": descs, "target": [float(probs.get(o, 0.0)) for o in opts],
                        "label": gold[name]["label"], "type": q.get("type", gold[name].get("type")), "qname": name,
                        "workflow": case["workflow"]})
    return out


def general_decisions(n, seed, commercial_only):
    from datasets import load_dataset

    d = load_dataset("tasksource/tasksource-jev-typed-decisions", split="train")
    d = d.filter(lambda e: not any(w in e["source"].lower() for w in WORKFLOWS))
    if commercial_only:
        d = d.filter(lambda e: e["license_use"] == "commercial")
    d = d.shuffle(seed=seed).select(range(min(n, len(d))))
    return [{"state": e["state"], "question": e["question"], "options": list(e["options"]), "descriptions": None,
             "target": list(e["target"]), "label": None, "type": e["kind"], "qname": e["source"], "workflow": "general"}
            for e in d]


# ------------------------------------------------------------------------------------------------------ model


class Encoder(torch.nn.Module):
    """A GTS masked LM's encoder or a Hugging Face encoder, with an option-marker scoring head."""

    def __init__(self, gts=None, hf=None, loops=None):
        super().__init__()
        self.loops = loops
        if gts:
            blob = torch.load(gts, map_location="cpu", weights_only=False)
            lm = GTSForMaskedLM(GTSConfig(**blob["config"]))
            if "model" in blob:
                lm.load_state_dict(blob["model"])
            else:
                load_binarized(blob, lm)
            self.backbone, self.hf, d = lm.backbone, None, blob["config"]["d_model"]
            from bert_pretrain import _tokenizer

            tok = _tokenizer()
            self.enc = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731
            self.cls, self.sep, self.marker, self.pad = 101, 102, 2, 0  # [unused1] marks an option
        else:
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(hf)
            tok.add_special_tokens({"additional_special_tokens": ["[OPT]"]})
            m = AutoModel.from_pretrained(hf)
            m.resize_token_embeddings(len(tok))
            self.backbone, self.hf, d = m, hf, m.config.hidden_size
            self.enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
            self.cls = tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id
            self.sep = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
            self.marker, self.pad = tok.convert_tokens_to_ids("[OPT]"), tok.pad_token_id or 0
        self.head = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 1))
        self.log_temp = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def encode(self, ex, max_len):
        """Token ids with the options last, so truncation only ever shortens the state; marker positions."""
        q = self.enc(ex["question"])[:64]
        opts = []
        for i, o in enumerate(ex["options"]):
            desc = ex["descriptions"][i] if ex["descriptions"] else None
            opts.append([self.marker] + self.enc(f"{o}: {desc}" if desc else str(o))[:40])
        tail = [t for o in opts for t in o] + [self.sep]
        room = max(8, max_len - 3 - len(q) - len(tail))
        ids = [self.cls] + q + [self.sep] + self.enc(ex["state"])[:room] + [self.sep] + tail
        ids = ids[:max_len]
        marks = [i for i, t in enumerate(ids) if t == self.marker]
        return ids, marks

    def forward(self, ids, mask, mark_rows, mark_cols, n_opts):
        if self.hf:
            h = self.backbone(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        else:
            h = self.backbone(ids, attention_mask=mask, loops=self.loops)
        s = self.head(h[mark_rows, mark_cols]).squeeze(-1).float() / self.log_temp.exp()
        return torch.split(s, n_opts)  # one score vector per decision


def batches(model, data, max_len, bs, device, shuffle=False, gen=None):
    enc = [model.encode(ex, max_len) for ex in data]
    keep = [i for i, (_, m) in enumerate(enc) if len(m) == len(data[i]["options"])]  # all markers survived truncation
    order = torch.randperm(len(keep), generator=gen).tolist() if shuffle else range(len(keep))
    order = [keep[i] for i in order]
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        L = max(len(enc[i][0]) for i in idx)
        L = -(-L // 64) * 64  # few shapes: compiled blocks recompile less
        ids = torch.full((len(idx), L), model.pad, dtype=torch.long)
        rows, cols, n_opts, targets = [], [], [], []
        for r, i in enumerate(idx):
            t, m = enc[i]
            ids[r, : len(t)] = torch.tensor(t)
            rows += [r] * len(m)
            cols += m
            n_opts.append(len(m))
            tg = torch.tensor(data[i]["target"], dtype=torch.float32)
            targets.append(tg / tg.sum().clamp(min=1e-9))
        yield (idx, ids.to(device), (ids != model.pad).to(device), torch.tensor(rows, device=device),
               torch.tensor(cols, device=device), n_opts, targets)


# ---------------------------------------------------------------------------------------------------- metrics


def scores(preds, data):
    """preds: list of probability vectors aligned with data. Accuracy, KL, Brier, ECE; overall and per type."""
    def one(sel):
        acc, kl, brier, conf, hit = [], [], [], [], []
        for p, ex in ((preds[i], data[i]) for i in sel):
            g = np.asarray(ex["target"], dtype=np.float64)
            g = g / max(g.sum(), 1e-12)
            p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
            label = ex["options"].index(ex["label"]) if ex["label"] in ex["options"] else int(g.argmax())
            k = int(p.argmax())
            acc.append(k == label)
            kl.append(float(np.sum(np.where(g > 0, g * np.log(np.clip(g, 1e-12, 1) / p), 0.0))))
            brier.append(float(np.sum((p - g) ** 2)))  # summed over options: reproduces the README's Uniform 0.238
            conf.append(p[k])
            hit.append(k == label)
        conf, hit = np.asarray(conf), np.asarray(hit, dtype=np.float64)
        bins = np.minimum((conf * 10).astype(int), 9)
        ece = sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        return {"accuracy": float(np.mean(acc)), "kl": float(np.mean(kl)), "brier": float(np.mean(brier)), "ece": float(ece),
                "n": len(sel)}

    out = {"all": one(range(len(data)))}
    for t in sorted({ex["type"] for ex in data}):
        out[t] = one([i for i, ex in enumerate(data) if ex["type"] == t])
    return out


def references(train, test):
    """Uniform and Prior (the mean gold distribution of the same question in train), to compare with the README."""
    uni = [np.full(len(ex["options"]), 1.0 / len(ex["options"])) for ex in test]
    prior_by = {}
    for ex in train:
        key = (ex["workflow"], ex["qname"])
        g = np.asarray(ex["target"], dtype=np.float64)
        prior_by.setdefault(key, []).append(g / g.sum())
    prior = []
    for ex, u in zip(test, uni):
        v = prior_by.get((ex["workflow"], ex["qname"]))
        prior.append(np.mean(v, 0) if v and len(v[0]) == len(u) else u)
    return {"uniform": scores(uni, test)["all"], "prior": scores(prior, test)["all"]}


# ------------------------------------------------------------------------------------------------------ train


def fit(model, data, a, device, epochs, lr):
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=device == "cuda")
    steps = epochs * math.ceil(len(data) / a.batch_size)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    g = torch.Generator().manual_seed(a.seed)
    t0, step = time.time(), 0
    model.train()
    for ep in range(epochs):
        for _, ids, mask, rows, cols, n_opts, targets in batches(model, data, a.max_len, a.batch_size, device, True, g):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                outs = model(ids, mask, rows, cols, n_opts)
            loss = sum(-(t.to(device) * F.log_softmax(o, -1)).sum() for o, t in zip(outs, targets)) / len(outs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % a.log_every == 0:
                print(f"  epoch {ep + 1} step {step}/{steps}  loss {loss.item():.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    model.eval()


@torch.no_grad()
def predict(model, data, a, device):
    preds = [None] * len(data)
    for idx, ids, mask, rows, cols, n_opts, _ in batches(model, data, a.max_len, a.batch_size, device):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            outs = model(ids, mask, rows, cols, n_opts)
        for i, o in zip(idx, outs):
            preds[i] = F.softmax(o.float(), -1).cpu().numpy()
    return preds


@torch.no_grad()
def fit_temperature(model, data, a, device):
    """One temperature minimising the log loss on held-out decisions (a grid: robust and cheap)."""
    model.log_temp.zero_()
    raw = predict(model, data, a, device)
    best = (1e9, 0.0)
    for lt in np.linspace(-1.5, 1.5, 61):
        nll = 0.0
        for p, ex in zip(raw, data):
            if p is None:
                continue
            z = np.log(np.clip(p, 1e-12, 1)) / math.exp(lt)
            z = z - z.max()
            q = np.exp(z) / np.exp(z).sum()
            g = np.asarray(ex["target"]) / max(sum(ex["target"]), 1e-12)
            nll -= float(np.sum(g * np.log(np.clip(q, 1e-12, 1))))
        best = min(best, (nll, lt))
    model.log_temp.fill_(best[1])
    return math.exp(best[1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gts")
    src.add_argument("--hf")
    p.add_argument("--loops", type=int, help="GTS-Uni passes")
    p.add_argument("--regime", choices=["zero-shot", "fitted"], required=True)
    p.add_argument("--general", type=int, default=200000, help="general typed decisions to train on first (0: none)")
    p.add_argument("--commercial-only", action="store_true", help="general decisions licensed for commercial use only")
    p.add_argument("--epochs", type=int, default=1, help="passes over the general decisions")
    p.add_argument("--fit-epochs", type=int, default=5, help="fitted regime: passes over the benchmark's train")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--max-test", type=int, help="score only this many test decisions (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    model = Encoder(a.gts, a.hf, a.loops).to(device)
    bench_train, test = bench_decisions("train"), bench_decisions("test")
    if a.max_test:
        test = test[: a.max_test]
    res = {"model": a.gts or a.hf, "regime": a.regime, "args": vars(a), "references": references(bench_train, test)}
    print("references (README: Uniform 0.308 / 0.444 / 0.238 / 0.169, Prior 0.470 / 0.347 / 0.189 / 0.088; ECE defined differently): "
          + json.dumps({k: {m: round(v, 3) for m, v in r.items() if m != "n"} for k, r in res["references"].items()}), flush=True)
    t0 = time.time()
    if a.general:
        gen = general_decisions(a.general, a.seed, a.commercial_only)
        print(f"general typed decisions: {len(gen):,}", flush=True)
        cal = gen[-2000:]
        fit(model, gen[:-2000], a, device, a.epochs, a.lr)
    else:
        cal = []
    if a.regime == "fitted" and a.general:  # also score the zero-shot model on the way (same run, comparable)
        res["temperature_zero_shot"] = fit_temperature(model, cal, a, device)
        zp = predict(model, test, a, device)
        res["zero_shot_test"] = scores([zp[i] for i in range(len(test)) if zp[i] is not None],
                                       [test[i] for i in range(len(test)) if zp[i] is not None])
        r = res["zero_shot_test"]["all"]
        print(f"typed-decisions test (zero-shot, before fitting): accuracy {r['accuracy']:.3f}  KL {r['kl']:.3f}  "
              f"Brier {r['brier']:.3f}  ECE {r['ece']:.3f}", flush=True)
        model.log_temp.zero_()
    if a.regime == "fitted":
        cut = int(0.9 * len(bench_train))  # the last 10% of the benchmark's train calibrates
        fit(model, bench_train[:cut], a, device, a.fit_epochs, a.lr)
        cal = bench_train[cut:]
    res["temperature"] = fit_temperature(model, cal, a, device) if cal else 1.0
    preds = predict(model, test, a, device)
    kept = [i for i, p in enumerate(preds) if p is not None]
    res["test"] = scores([preds[i] for i in kept], [test[i] for i in kept])
    res["dropped_decisions"] = len(test) - len(kept)
    res["minutes"] = (time.time() - t0) / 60
    lens = [len(model.encode(ex, a.max_len)[0]) for ex in test[:500]]
    res["mean_tokens_per_decision"] = float(np.mean(lens))
    json.dump(res, open(a.out, "w"), indent=1)
    r = res["test"]["all"]
    print(f"typed-decisions test ({a.regime}): accuracy {r['accuracy']:.3f}  KL {r['kl']:.3f}  Brier {r['brier']:.3f}  "
          f"ECE {r['ece']:.3f}  (temperature {res['temperature']:.2f}; {res['dropped_decisions']} dropped; "
          f"{res['mean_tokens_per_decision']:.0f} tokens per decision)", flush=True)


if __name__ == "__main__":
    main()
