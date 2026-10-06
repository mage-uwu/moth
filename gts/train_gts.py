# Golden Tree Snake (GTS) fork, 2026.
"""Train a GTS masked-language model, in full precision or ternary.

    # 1. full-precision teacher
    python train_gts.py fp  --data corpus.txt --out fp.pt

    # 2. ternary student: quantisation-aware training from the teacher, with distillation
    python train_gts.py qat --data corpus.txt --teacher fp.pt --out ternary.pt

    # baseline: ternary from scratch, cross-entropy only
    python train_gts.py fp  --data corpus.txt --out scratch.pt --ternary

    python train_gts.py eval --data corpus.txt --ckpt ternary.pt

``--data`` is either a text file, read as bytes (vocabulary of 258: pad, mask, 256 bytes), or a
``.npy`` array of token ids, in which case pass ``--vocab-size``, ``--pad-id`` and ``--mask-id``.
The last 5% of the data is held out for evaluation.
"""

import argparse
import dataclasses
import math
import time

import numpy as np
import torch

from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM
from mamba_ssm.modules.gts import GTS
from mamba_ssm.utils.gts_qat import (
    lambda_schedule, make_ternary_student, qat_loss, set_quant_lambda, ternary_stats,
)


# ------------------------------------------------------------------------------------ data

class Corpus:
    def __init__(self, path, vocab_size=None, pad_id=0, mask_id=1, holdout=0.05):
        if path.endswith(".npy"):
            ids = np.load(path).astype(np.int64)
            assert vocab_size is not None, "--vocab-size is required with a .npy token file"
            self.first_regular = 0  # random replacements may draw any id
        else:
            ids = np.frombuffer(open(path, "rb").read(), dtype=np.uint8).astype(np.int64) + 2
            vocab_size, pad_id, mask_id = 258, 0, 1
            self.first_regular = 2
        self.vocab_size, self.pad_id, self.mask_id = vocab_size, pad_id, mask_id
        split = int(len(ids) * (1 - holdout))
        self.train, self.heldout = torch.from_numpy(ids[:split]), torch.from_numpy(ids[split:])

    def batch(self, split, batch_size, seq_len, mlm_prob, generator):
        data = self.train if split == "train" else self.heldout
        starts = torch.randint(0, len(data) - seq_len, (batch_size,), generator=generator)
        ids = torch.stack([data[s : s + seq_len] for s in starts])
        labels = torch.full_like(ids, -100)
        pick = torch.rand(ids.shape, generator=generator) < mlm_prob
        labels[pick] = ids[pick]
        inputs = ids.clone()
        roll = torch.rand(ids.shape, generator=generator)
        inputs[pick & (roll < 0.8)] = self.mask_id  # 80% mask, 10% random token, 10% unchanged
        rand = pick & (roll >= 0.8) & (roll < 0.9)
        inputs[rand] = torch.randint(self.first_regular, self.vocab_size, (int(rand.sum()),), generator=generator)
        return inputs, labels


# ------------------------------------------------------------------------------- training

def param_groups(model, weight_decay):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(step, steps, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


@torch.no_grad()
def evaluate(model, corpus, args, batches=20):
    was_training = model.training
    model.eval()
    g = torch.Generator().manual_seed(1234)  # the same held-out batches every time
    loss = correct = count = 0.0
    used = []
    for _ in range(batches):
        inputs, labels = corpus.batch("heldout", args.batch_size, args.seq_len, args.mlm_prob, g)
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        out = model(inputs, labels=labels)
        sel = labels != -100
        n = int(sel.sum())
        loss += out.loss.item() * n
        correct += (out.logits[sel].argmax(-1) == labels[sel]).sum().item()
        count += n
    mixer = [m for m in model.modules() if isinstance(m, GTS)][-1]
    _, nodes = mixer(torch.randn(2, args.seq_len, mixer.d_model, device=args.device), return_paths=True)
    used = mixer.path_stats(nodes)
    model.train(was_training)
    return {"loss": loss / count, "accuracy": correct / count, "leaf_usage": used[-1]}


def train(model, corpus, args, teacher=None):
    """Cross-entropy training, or QAT against ``teacher`` when one is given."""
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(args.seed)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        set_quant_lambda(model, lambda_schedule(step, args.lambda_warmup))
        inputs, labels = corpus.batch("train", args.batch_size, args.seq_len, args.mlm_prob, g)
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        if teacher is None:
            loss = model(inputs, labels=labels).loss
            info = {}
        else:
            info = qat_loss(model, teacher, inputs, labels, alpha=args.alpha, temperature=args.temperature, beta=args.beta)
            loss = info["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if args.log_every and (step % args.log_every == 0 or step == args.steps - 1):
            msg = f"step {step:5d}  loss {loss.item():.4f}"
            if "path_agreement" in info:
                msg += f"  path_agree {info['path_agreement']:.3f}"
            stats = ternary_stats(model)
            if stats:
                msg += f"  zeros {stats['zero_ratio_mean']:.3f} [{stats['zero_ratio_min']:.3f}, {stats['zero_ratio_max']:.3f}]"
            print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)
    set_quant_lambda(model, 1.0)
    return model


def save(model, path):
    torch.save({"config": dataclasses.asdict(model.config), "state_dict": model.state_dict()}, path)


def load(path):
    ckpt = torch.load(path, map_location="cpu")
    model = GTSForMaskedLM(GTSConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["fp", "qat", "eval"])
    p.add_argument("--data", required=True)
    p.add_argument("--out")
    p.add_argument("--ckpt")
    p.add_argument("--teacher")
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--pad-id", type=int, default=0)
    p.add_argument("--mask-id", type=int, default=1)
    # model
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-layer", type=int, default=12)
    p.add_argument("--depth", type=int, default=11)
    p.add_argument("--n-trees", type=int, default=1)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--ternary", action="store_true", help="fp mode: train ternary from scratch")
    p.add_argument("--ternary-group", type=int, default=128)
    p.add_argument("--act-bits", type=int, default=None)
    # optimisation (QAT defaults follow Ternary Mamba)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5, help="qat: weight of KL against cross-entropy")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0, help="qat: weight of route distillation")
    p.add_argument("--lambda-warmup", type=int, default=0, help="steps over which quantisation is blended in")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-every", type=int, default=100)
    return p


def main():
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    corpus = Corpus(args.data, args.vocab_size, args.pad_id, args.mask_id)
    if args.mode == "eval":
        print(evaluate(load(args.ckpt).to(args.device), corpus, args))
        return
    if args.mode == "fp":
        config = GTSConfig(
            d_model=args.d_model, n_layer=args.n_layer, vocab_size=corpus.vocab_size, depth=args.depth,
            n_trees=args.n_trees, d_state=args.d_state, pad_token_id=corpus.pad_id,
            ternary=args.ternary, ternary_group=args.ternary_group, act_bits=args.act_bits,
        )
        model = train(GTSForMaskedLM(config).to(args.device), corpus, args)
    else:
        teacher = load(args.teacher).to(args.device)
        student = make_ternary_student(teacher, args.ternary_group, args.act_bits)
        model = train(student, corpus, args, teacher=teacher)
    print(evaluate(model, corpus, args))
    if args.out:
        save(model, args.out)


if __name__ == "__main__":
    main()
