# Golden Tree Snake (GTS) fork, 2026.
"""Fine-tune and evaluate encoders on GLUE tasks with one harness, so GTS models and BERT-family baselines are
compared under the same data, budget and schedule.

    python scripts/glue_finetune.py --gts checkpoints/.../binarized.pt --tasks sst2 mrpc rte --out results/glue_gts3.json
    python scripts/glue_finetune.py --hf google-bert/bert-base-uncased --tasks sst2 mrpc rte --out results/glue_bert.json

GTS: the masked-LM encoder (GTSForMaskedLM's backbone, a GTS-Uni too: ``--loops`` passes) fully fine-tuned with its
ternary weights trained through the straight-through estimator as in pretraining; classification head on [CLS]'s and
the mean final state, dropout 0.1; ``--save-dir`` keeps each task's best epoch (GTSClassifier.load). Pairs are "[CLS] a [SEP] b [SEP]" (GTS has no segment embeddings). Baselines:
AutoModelForSequenceClassification. Both: AdamW, linear warmup over 10% then linear decay, bf16 on a GPU, inputs padded
to --max-len (static shapes), the dev set scored after every epoch and the best epoch reported, as is usual for GLUE
dev numbers. Metrics: accuracy (MNLI matched, QNLI, RTE, SST-2), F1 and accuracy (MRPC, QQP), Matthews correlation
(CoLA), Pearson and Spearman (STS-B).
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

TASKS = {  # name: (glue config, text fields, labels (1 = regression), dev split)
    "sst2": ("sst2", ("sentence", None), 2, "validation"),
    "mrpc": ("mrpc", ("sentence1", "sentence2"), 2, "validation"),
    "rte": ("rte", ("sentence1", "sentence2"), 2, "validation"),
    "qnli": ("qnli", ("question", "sentence"), 2, "validation"),
    "mnli": ("mnli", ("premise", "hypothesis"), 3, "validation_matched"),
    "qqp": ("qqp", ("question1", "question2"), 2, "validation"),
    "cola": ("cola", ("sentence", None), 2, "validation"),
    "stsb": ("stsb", ("sentence1", "sentence2"), 1, "validation"),
}


class GTSClassifier(torch.nn.Module):
    def __init__(self, path, n_labels, loops=None):
        super().__init__()
        blob = torch.load(path, map_location="cpu", weights_only=False) if isinstance(path, str) else path
        lm = GTSForMaskedLM(GTSConfig(**blob["config"]))
        if "model" in blob:
            lm.load_state_dict(blob["model"])
        elif blob.get("kind") != "gts-glue":  # a saved classifier is loaded whole by load()
            load_binarized(blob, lm)
        self.config = blob["config"]
        self.backbone, self.loops = lm.backbone, loops
        d = blob["config"]["d_model"]
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(2 * d, n_labels)

    @classmethod
    def load(cls, path):
        """A fine-tuned classifier saved by --save-dir (binarized: ternary codes plus float head)."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(blob, blob["n_labels"], blob["loops"])
        return load_binarized(blob, model)

    def forward(self, ids, mask):
        h = self.backbone(ids, attention_mask=mask, loops=self.loops)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = torch.cat([h[:, 0], (h * m).sum(1) / m.sum(1)], -1)
        return self.head(self.drop(pooled))


class HFClassifier(torch.nn.Module):
    def __init__(self, repo, n_labels):
        super().__init__()
        from transformers import AutoModelForSequenceClassification

        self.m = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=n_labels)

    def forward(self, ids, mask):
        return self.m(input_ids=ids, attention_mask=mask.long()).logits


def encode(ds, fields, tokenize, max_len):
    a, b = fields
    ids = np.zeros((len(ds), max_len), dtype=np.int64)
    for i, ex in enumerate(ds):
        t = tokenize(ex[a], ex[b] if b else None)[:max_len]
        ids[i, : len(t)] = t
    return torch.from_numpy(ids), torch.tensor(ds["label"], dtype=torch.long)


def metric(task, preds, labels):
    if task == "stsb":
        rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)  # noqa: E731 (ties are rare in regression outputs)
        corr = lambda u, v: float(np.corrcoef(u, v)[0, 1])  # noqa: E731
        return {"pearson": corr(preds, labels), "spearman": corr(rank(preds), rank(labels))}
    acc = float((preds == labels).mean())
    if task in ("mrpc", "qqp"):
        tp = ((preds == 1) & (labels == 1)).sum()
        f1 = 2 * tp / max(1, (preds == 1).sum() + (labels == 1).sum())
        return {"accuracy": acc, "f1": float(f1)}
    if task == "cola":
        tp, tn = ((preds == 1) & (labels == 1)).sum(), ((preds == 0) & (labels == 0)).sum()
        fp, fn = ((preds == 1) & (labels == 0)).sum(), ((preds == 0) & (labels == 1)).sum()
        den = math.sqrt(max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        return {"mcc": float((tp * tn - fp * fn) / den)}
    return {"accuracy": acc}


def headline(task, m):
    return m.get("mcc", m.get("spearman", m.get("f1", m.get("accuracy"))))


def run_task(a, task, device):
    from datasets import load_dataset

    cfg, fields, n_labels, dev = TASKS[task]
    d = load_dataset("nyu-mll/glue", cfg)
    tr, va = d["train"], d[dev]
    if a.max_train and len(tr) > a.max_train:
        tr = tr.shuffle(seed=0).select(range(a.max_train))
    if a.max_eval and len(va) > a.max_eval:
        va = va.select(range(a.max_eval))
    if a.gts:
        from bert_pretrain import _tokenizer

        tok = _tokenizer()

        def tokenize(x, y):
            ids = [101] + tok.encode(x, add_special_tokens=False).ids + [102]
            return ids + (tok.encode(y, add_special_tokens=False).ids + [102] if y is not None else [])
    else:
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(a.hf)

        def tokenize(x, y):
            return hf(x, y, truncation=True, max_length=a.max_len)["input_ids"]
    xtr, ytr = encode(tr, fields, tokenize, a.max_len)
    xva, yva = encode(va, fields, tokenize, a.max_len)
    if n_labels == 1:
        ytr, yva = torch.tensor(tr["label"], dtype=torch.float32), torch.tensor(va["label"], dtype=torch.float32)
    torch.manual_seed(a.seed)
    model = (GTSClassifier(a.gts, n_labels, a.loops) if a.gts else HFClassifier(a.hf, n_labels)).to(device)
    if a.compile and device == "cuda" and a.gts:
        layers = model.backbone.layers
        for i in range(len(layers)):
            layers[i] = torch.compile(layers[i], dynamic=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, fused=device == "cuda")
    steps = a.epochs * math.ceil(len(xtr) / a.batch_size)
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    amp = device == "cuda"
    g = torch.Generator().manual_seed(a.seed)
    best, history, t0 = None, [], time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(perm), a.batch_size):
            b = perm[i : i + a.batch_size]
            ids = xtr[b].to(device)
            mask = ids != 0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(ids, mask)
            y = ytr[b].to(device)
            loss = F.mse_loss(out.squeeze(-1).float(), y) if n_labels == 1 else F.cross_entropy(out.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xva), 256):
                ids = xva[i : i + 256].to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    out = model(ids, ids != 0).float()
                preds.append(out.squeeze(-1) if n_labels == 1 else out.argmax(-1))
        p = torch.cat(preds).cpu().numpy()
        m = metric(task, p, yva.numpy())
        history.append(m)
        if best is None or headline(task, m) > headline(task, best):
            best = m
            if a.save_dir and a.gts:  # keep the best epoch's weights
                os.makedirs(a.save_dir, exist_ok=True)
                from mamba_ssm.utils.ternary_pack import save_binarized

                save_binarized(model, model.config, os.path.join(a.save_dir, f"{task}.pt"),
                               {"kind": "gts-glue", "task": task, "n_labels": n_labels, "loops": a.loops,
                                "epoch": ep + 1, "dev": m, "max_len": a.max_len})
        print(f"  {task} epoch {ep + 1}: " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()) + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return {"best": best, "epochs": history, "train_examples": len(xtr), "dev_examples": len(xva), "minutes": (time.time() - t0) / 60}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--gts", help="GTS masked LM: checkpoint.pt or binarized.pt")
    src.add_argument("--hf", help="Hugging Face encoder, e.g. google-bert/bert-base-uncased")
    p.add_argument("--loops", type=int, help="GTS-Uni: passes to fine-tune and evaluate with (default: all)")
    p.add_argument("--tasks", nargs="+", default=["sst2", "mrpc", "rte", "qnli", "mnli", "cola", "stsb"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--max-train", type=int, help="subsample large training sets (MNLI, QQP, QNLI) to this many")
    p.add_argument("--max-eval", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--out", required=True)
    p.add_argument("--save-dir", help="GTS: save each task's best-epoch classifier here as <task>.pt (binarized, "
                   "about 120 MB; GTSClassifier.load reads it back)")
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    res = {"model": a.gts or a.hf, "args": vars(a), "tasks": {}}
    for task in a.tasks:
        res["tasks"][task] = run_task(a, task, device)
        json.dump(res, open(a.out, "w"), indent=1)
    scores = {t: headline(t, r["best"]) for t, r in res["tasks"].items()}
    res["average"] = float(np.mean(list(scores.values())))
    json.dump(res, open(a.out, "w"), indent=1)
    print("GLUE dev: " + "  ".join(f"{t} {s:.4f}" for t, s in scores.items()) + f"  average {res['average']:.4f}", flush=True)


if __name__ == "__main__":
    main()
