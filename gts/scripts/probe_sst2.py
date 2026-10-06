# Golden Tree Snake (GTS) fork, 2026.
"""Light adaptation of a frozen GTS masked LM on SST-2 (sentiment), on a CPU, with frozen BERT-base as a reference.

    python scripts/probe_sst2.py --gts checkpoints/.../binarized.pt [--train 20000] [--out results/sst2_probe.json]

Three ways to read sentiment out of a frozen encoder, nothing inside it trained:
  zero-shot  the masked-LM head on "<sentence> it was [MASK] ." comparing "great"+"good" against "terrible"+"bad";
  probe      a linear classifier on the frozen sentence features (mean of the final states, and [CLS]'s);
  adapter    a bottleneck MLP (features -> 64 -> 2, GELU, dropout) on the same features.
Hyperparameters (weight decay, epochs) are chosen on 2,000 sentences held out from the training subset; the 872
validation sentences (SST-2's labelled evaluation set) are scored once at the end.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized  # noqa: E402

POS, NEG = ["great", "good"], ["terrible", "bad"]


class GTSEncoder:
    name = "GTS3"

    def __init__(self, path):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.model = GTSForMaskedLM(GTSConfig(**blob["config"]))
        if "model" in blob:
            self.model.load_state_dict(blob["model"])
        else:
            load_binarized(blob, self.model)
        self.model.eval()

    def hidden(self, ids, mask):
        return self.model.backbone(ids, attention_mask=mask)

    def logits(self, h):
        return self.model._head(h)


class BertEncoder:
    name = "BERT-base (reference)"

    def __init__(self):
        from transformers import AutoModelForMaskedLM

        self.model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased").eval()

    def hidden(self, ids, mask):
        return self.model.bert(input_ids=ids, attention_mask=mask.long()).last_hidden_state

    def logits(self, h):
        return self.model.cls(h)


def batches(seqs, size):
    """Indices in length order (little padding), in batches."""
    order = np.argsort([len(s) for s in seqs])
    for i in range(0, len(order), size):
        yield order[i : i + size]


def pad(seqs):
    L = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, ids != 0


@torch.no_grad()
def features(enc, seqs, size=128):
    out = np.zeros((len(seqs), 2 * 768), dtype=np.float32)
    for idx in batches(seqs, size):
        ids, mask = pad([seqs[i] for i in idx])
        h = enc.hidden(ids, mask).float()
        mean = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        out[idx] = torch.cat([mean, h[:, 0]], 1).numpy()
    return out


@torch.no_grad()
def zero_shot(enc, prompts, mask_pos, tok, size=128):
    pos = [tok.token_to_id(w) for w in POS]
    neg = [tok.token_to_id(w) for w in NEG]
    score = np.zeros(len(prompts), dtype=np.float32)
    for idx in batches(prompts, size):
        ids, mask = pad([prompts[i] for i in idx])
        h = enc.hidden(ids, mask)
        at = h[torch.arange(len(idx)), torch.tensor([mask_pos[i] for i in idx])]
        lp = torch.log_softmax(enc.logits(at).float(), -1)
        score[idx] = (torch.logsumexp(lp[:, pos], -1) - torch.logsumexp(lp[:, neg], -1)).numpy()
    return score


def train_head(xtr, ytr, xse, yse, hidden=None, wds=(1e-4, 1e-2, 1e-1), epochs=(10, 30, 60), seed=0):
    """A linear head (hidden=None) or a bottleneck adapter; weight decay and epochs chosen on the selection split."""
    def fit(wd, ep):
        torch.manual_seed(seed)
        d = xtr.shape[1]
        head = torch.nn.Linear(d, 2) if hidden is None else torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(hidden, 2))
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=wd)
        X, Y = torch.from_numpy(xtr), torch.from_numpy(ytr)
        for _ in range(ep):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 256):
                b = perm[i : i + 256]
                loss = F.cross_entropy(head(X[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return head.eval()

    best = None
    for wd in wds:
        for ep in epochs:
            head = fit(wd, ep)
            with torch.no_grad():
                acc = (head(torch.from_numpy(xse)).argmax(1).numpy() == yse).mean()
            if best is None or acc > best[0]:
                best = (acc, wd, ep, head)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gts", required=True, help="GTS masked LM: binarized.pt or checkpoint.pt")
    p.add_argument("--train", type=int, default=20000)
    p.add_argument("--no-bert", action="store_true")
    p.add_argument("--out", default="results/sst2_probe.json")
    a = p.parse_args()
    torch.set_num_threads(os.cpu_count())
    from datasets import load_dataset

    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from bert_pretrain import _tokenizer

    tok = _tokenizer()
    d = load_dataset("stanfordnlp/sst2")
    rng = np.random.default_rng(0)
    tr = d["train"].select(rng.permutation(len(d["train"]))[: a.train + 2000])
    va = d["validation"]
    enc_ids = lambda texts: [[101] + e.ids + [102] for e in tok.encode_batch([t.strip() for t in texts], add_special_tokens=False)]  # noqa: E731
    tr_ids, va_ids = enc_ids(tr["sentence"]), enc_ids(va["sentence"])
    ytr_all, yva = np.array(tr["label"]), np.array(va["label"])
    sel = slice(a.train, None)
    prompts, mask_pos = [], []
    for s in va["sentence"]:
        ids = tok.encode(s.strip() + " it was [MASK] .", add_special_tokens=False).ids
        prompts.append([101] + ids + [102])
        mask_pos.append(1 + ids.index(103))
    results = {"task": "SST-2", "train_sentences": a.train, "selection_sentences": 2000, "validation_sentences": len(va), "models": {}}
    encoders = [GTSEncoder(a.gts)] + ([] if a.no_bert else [BertEncoder()])
    for enc in encoders:
        t0 = time.time()
        ftr, fva = features(enc, tr_ids), features(enc, va_ids)
        t_feat = time.time() - t0
        mu, sd = ftr[: a.train].mean(0), ftr[: a.train].std(0) + 1e-6
        ftr, fva = (ftr - mu) / sd, (fva - mu) / sd
        zs = zero_shot(enc, prompts, mask_pos, tok)
        r = {"zero_shot_acc": float(((zs > 0).astype(int) == yva).mean()), "feature_seconds": round(t_feat)}
        for kind, hidden in (("probe", None), ("adapter", 64)):
            acc_sel, wd, ep, head = train_head(ftr[: a.train], ytr_all[: a.train], ftr[sel], ytr_all[sel], hidden)
            with torch.no_grad():
                acc = (head(torch.from_numpy(fva)).argmax(1).numpy() == yva).mean()
            n = sum(p.numel() for p in head.parameters())
            r[kind] = {"val_acc": float(acc), "selection_acc": float(acc_sel), "weight_decay": wd, "epochs": ep, "parameters": n}
        results["models"][enc.name] = r
        print(f"{enc.name}: zero-shot {r['zero_shot_acc']:.3f}  probe {r['probe']['val_acc']:.3f}  "
              f"adapter {r['adapter']['val_acc']:.3f}  ({r['adapter']['parameters']:,} adapter parameters; features in {t_feat:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
