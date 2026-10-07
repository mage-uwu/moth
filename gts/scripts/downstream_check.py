# Golden Tree Snake (GTS) fork, 2026.
"""Quick downstream check for a GTS-L student (mamba_ssm/models/gts_l.py), and for its teacher as the reference: short
fine-tunes on subsets of three GLUE tasks, scored on (subsets of) their dev sets.

    SST-2   8,000 training sentences, 2 epochs, all 872 dev sentences          accuracy
    MNLI   16,000 training pairs, 2 epochs, 2,000 matched-dev pairs            accuracy
    STS-B  all 5,749 training pairs, 3 epochs, all 1,500 dev pairs             Spearman

Same protocol for student and teacher: the teacher's tokenizer, "[CLS] a [SEP] b [SEP]" truncated to 128 tokens, batch
32, AdamW (weight decay 0.01), 10% linear warmup then linear decay, bf16 on a GPU, the best epoch reported (the usual
GLUE dev convention). The student is a fresh copy of the given weights at their current quantisation (the training
model is not touched) with a head on [CLS]'s and the mean final state; padding is masked out of its scans. The
teacher is AutoModelForSequenceClassification. About 5 minutes each on an A100. The subsets make it a trend signal
(SST-2's dev set alone moves about +-1.5 points between seeds), not a GLUE score.

    python scripts/downstream_check.py --ckpt out/eva/final.pt --out results/eva/downstream.json
    python scripts/downstream_check.py --hf answerdotai/ModernBERT-large --out results/eva/downstream_teacher.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mamba_ssm.models.gts_l import GTSLConfig, GTSLForMaskedLM  # noqa: E402

TOKENIZER = os.environ.get("MOHAWK_TEACHER", "answerdotai/ModernBERT-large")
TASKS = {  # name: (text fields, labels (1 = regression), dev split, training subset, dev subset, epochs)
    "sst2": (("sentence", None), 2, "validation", 8000, None, 2),
    "mnli": (("premise", "hypothesis"), 3, "validation_matched", 16000, 2000, 2),
    "stsb": (("sentence1", "sentence2"), 1, "validation", None, None, 3),
}
_DATA = {}


def load_task(task, max_len=128, limit=None):
    """((train ids, labels), (dev ids, labels), n_labels, pad id), tokenised once per process. ``limit`` caps both
    splits (tests)."""
    key = (task, max_len, limit)
    if key not in _DATA:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        fields, n_labels, dev, n_tr, n_dev, _ = TASKS[task]
        d = load_dataset("nyu-mll/glue", task)
        tr, va = d["train"], d[dev]
        n_tr, n_dev = min(x for x in (n_tr, limit, len(tr)) if x), min(x for x in (n_dev, limit, len(va)) if x)
        tr, va = tr.shuffle(seed=0).select(range(n_tr)), va.shuffle(seed=0).select(range(n_dev))

        def enc(ds):
            a, b = fields
            ids = tok(list(ds[a]), list(ds[b]) if b else None, truncation=True, max_length=max_len)["input_ids"]
            return ids, torch.tensor(list(ds["label"]), dtype=torch.float32 if n_labels == 1 else torch.long)

        _DATA[key] = (enc(tr), enc(va), n_labels, tok.pad_token_id)
    return _DATA[key]


def pad_batch(seqs, pad, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    ids = ids.to(device)
    return ids, ids != pad


def metric(task, p, y):
    if task == "stsb":
        rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)  # noqa: E731
        return {"spearman": float(np.corrcoef(rank(p), rank(y))[0, 1]), "pearson": float(np.corrcoef(p, y)[0, 1])}
    return {"accuracy": float((p == y).mean())}


def headline(m):
    return m.get("spearman", m.get("accuracy"))


class GTSLClassifier(nn.Module):
    def __init__(self, lm, n_labels):
        super().__init__()
        self.lm, d = lm, lm.cfg.d_model
        self.drop, self.head = nn.Dropout(0.1), nn.Linear(2 * d, n_labels)

    def forward(self, ids, mask):
        h = self.lm.hidden(ids, mask=mask.float())
        m = mask.unsqueeze(-1).to(h.dtype)
        return self.head(self.drop(torch.cat([h[:, 0], (h * m).sum(1) / m.sum(1)], -1)))

    def parameters_to_train(self):  # the masked-LM head is unused here
        head = {id(p) for p in (self.lm.head_dense.weight, self.lm.head_norm.weight, self.lm.decoder_bias)}
        return [p for p in self.parameters() if id(p) not in head]


class HFClassifier(nn.Module):
    def __init__(self, repo, n_labels):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        try:  # ModernBERT compiles parts of itself by default; keep it eager here
            self.m = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=n_labels, reference_compile=False)
        except TypeError:
            self.m = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=n_labels)

    def forward(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask.long()).logits

    def parameters_to_train(self):
        return list(self.parameters())


def finetune(make_model, task, device, lr, seed=0, batch_size=32, limit=None, epochs=None):
    (xtr, ytr), (xva, yva), n_labels, pad = load_task(task, limit=limit)
    epochs = epochs or TASKS[task][5]
    torch.manual_seed(seed)
    model = make_model(n_labels).to(device)
    params = [p for p in model.parameters_to_train() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, fused=device == "cuda")
    steps = epochs * math.ceil(len(xtr) / batch_size)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    g = torch.Generator().manual_seed(seed)
    best = None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).tolist()
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            ids, mask = pad_batch([xtr[j] for j in b], pad, device)
            with torch.autocast(**amp):
                out = model(ids, mask).float()
            y = ytr[b].to(device)
            loss = F.mse_loss(out.squeeze(-1), y) if n_labels == 1 else F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xva), 128):
                ids, mask = pad_batch(xva[i : i + 128], pad, device)
                with torch.autocast(**amp):
                    out = model(ids, mask).float()
                preds.append(out.squeeze(-1) if n_labels == 1 else out.argmax(-1))
        m = metric(task, torch.cat(preds).cpu().numpy(), yva.numpy())
        if best is None or headline(m) > headline(best):
            best = m
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return best


def run_check(make_model, device, lr, tasks=tuple(TASKS), limit=None, epochs=None):
    """{task: best dev metrics, ..., "score": mean headline x 100, "minutes": wall time}."""
    t0, res = time.time(), {}
    for task in tasks:
        res[task] = finetune(make_model, task, device, lr, limit=limit, epochs=epochs)
    res["score"] = 100 * float(np.mean([headline(res[t]) for t in tasks]))
    res["minutes"] = (time.time() - t0) / 60
    return res


def student_check(student, device, lr=5e-5, **kw):
    """The check on a fresh copy of ``student`` (a GTSLForMaskedLM, compiled or not) at its current quantisation."""
    lam = student.layers[0].mixer.in_proj.lam
    sd = {k.replace("_orig_mod.", ""): v.detach() for k, v in student.state_dict().items()}

    def make(n_labels):
        lm = GTSLForMaskedLM(student.cfg)
        lm.load_state_dict(sd)
        lm.set_quant(lam)
        return GTSLClassifier(lm, n_labels)

    return {"quant": lam, **run_check(make, device, lr, **kw)}


def teacher_check(repo, device, lr=2e-5, **kw):
    return run_check(lambda n: HFClassifier(repo, n), device, lr, **kw)


def summary(res):
    return (f"SST-2 {100 * res['sst2']['accuracy']:.1f}  MNLI-m {100 * res['mnli']['accuracy']:.1f}  "
            f"STS-B {100 * res['stsb']['spearman']:.1f}  mean {res['score']:.1f}  ({res['minutes']:.1f} min)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="GTS-L checkpoint (final.pt, stage3.pt, state.pt) with 'model' (and 'config')")
    src.add_argument("--hf", help="Hugging Face encoder, e.g. answerdotai/ModernBERT-large")
    p.add_argument("--quant", type=float, default=1.0, help="GTS-L: quantisation to evaluate at (1 = ternary)")
    p.add_argument("--lr", type=float, help="default 5e-5 for GTS-L, 2e-5 for --hf")
    p.add_argument("--limit", type=int, help="cap every split (smoke tests)")
    p.add_argument("--epochs", type=int, help="override every task's epochs (smoke tests)")
    p.add_argument("--out")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = dict(limit=a.limit, epochs=a.epochs)
    if a.hf:
        res = teacher_check(a.hf, device, a.lr or 2e-5, **kw)
    else:
        blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        cfg = GTSLConfig(**blob["config"]) if "config" in blob else GTSLConfig(**blob["log"]["config"])
        lm = GTSLForMaskedLM(cfg)
        lm.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in blob["model"].items()})
        lm.set_quant(a.quant)
        res = student_check(lm, device, a.lr or 5e-5, **kw)
    print(summary(res), flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
