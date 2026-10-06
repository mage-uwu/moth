# Golden Tree Snake (GTS) fork, 2026.
"""Width test: the bidirectional mixed forest against a bidirectional Mamba-2 at equal parameters.

    python scripts/width_test.py speed --widths 128 256 512
    python scripts/width_test.py train --arch mixed  --width 256 --data data/tinyshakespeare.txt --out runs/w256_mixed
    python scripts/width_test.py train --arch mamba2 --width 256 --data data/tinyshakespeare.txt --out runs/w256_mamba2

As width doubles, a dense block's parameters and cost both quadruple. The mixed forest keeps its shape and adds
one level to its deep trees per doubling, which doubles its node count: its parameters quadruple too, but a token
touches only four more nodes per layer. ``speed`` measures what that does to kernel time (causal models with random
weights, since speed does not depend on training). ``train`` measures what it does to quality, on masked-byte
modelling with both models ternary, the same wrapper, optimiser and batches.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import shakespeare_ar as S  # noqa: E402
import train_gts as T  # noqa: E402
from mamba_ssm.models.gts_encoder import RMSNorm  # noqa: E402
from mamba_ssm.modules.gts import GTSMixed  # noqa: E402


def deep_depth(width):
    return 6 + round(math.log2(width / 128))


class BiMamba2Ref(nn.Module):
    """Mamba-2 run left to right and right to left with shared weights, the two outputs summed."""

    def __init__(self, d_model):
        super().__init__()
        self.m = S.Mamba2Ref(d_model, d_state=32, expand=2, headdim=32, ternary=True)

    def forward(self, u, attention_mask=None):
        return self.m(u) + self.m(u.flip(1)).flip(1)


class Block(nn.Module):
    def __init__(self, d_model, mixer):
        super().__init__()
        self.norm, self.mixer = RMSNorm(d_model), mixer

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class TinyEncoder(nn.Module):
    """The shared masked-LM wrapper. Only the mixer differs between the two models."""

    def __init__(self, vocab, d_model, n_layer, make_mixer):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.layers = nn.ModuleList([Block(d_model, make_mixer()) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.head_bias = nn.Parameter(torch.zeros(vocab))

    def forward(self, ids, labels):
        x = self.embedding(ids)
        for layer in self.layers:
            x = layer(x)
        logits = F.linear(self.norm_f(x), self.embedding.weight, self.head_bias)
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


def build_encoder(arch, vocab, width):
    if arch == "mixed":
        make = lambda: GTSMixed(width, bank_trees=32, bank_heads=8, bank_state=16, deep_trees=4, deep_depth=deep_depth(width),
                                ternary=True, act_bits=8, route_ste=True)
        return TinyEncoder(vocab, width, 4, make)
    return TinyEncoder(vocab, width, 5, lambda: BiMamba2Ref(width))


@torch.no_grad()
def evaluate(model, corpus, args, batches=40):
    model.eval()
    g = torch.Generator().manual_seed(999)
    loss = correct = count = 0.0
    for _ in range(batches):
        x, y = corpus.batch("heldout", args.batch_size, args.seq_len, args.mlm_prob, g)
        x, y = x.to(args.device), y.to(args.device)
        logits, batch_loss = model(x, y)
        sel = y != -100
        n = int(sel.sum())
        loss += batch_loss.item() * n
        correct += (logits[sel].argmax(-1) == y[sel]).sum().item()
        count += n
    model.train()
    return loss / count, correct / count


def train(args):
    torch.manual_seed(args.seed)
    corpus = T.Corpus(args.data)
    model = build_encoder(args.arch, corpus.vocab_size, args.width).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch} width {args.width}: {n_params:,} parameters", flush=True)
    decay = [p for p in model.parameters() if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)]
    rest = [p for p in model.parameters() if not (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    curve, step_time, run_loss, run_n = [], 0.0, 0.0, 0
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            val, acc = evaluate(model, corpus, args)
            train_loss = run_loss / run_n if run_n else float("nan")
            curve.append([step, val, acc, train_loss])
            print(f"step {step:5d}  val loss {val:.4f}  masked accuracy {acc:.4f}  train {train_loss:.4f}  {step_time / max(step, 1) * 1000:.0f} ms/step", flush=True)
            run_loss, run_n = 0.0, 0
            json.dump({"arch": args.arch, "width": args.width, "params": n_params, "steps": step, "val_loss": val, "masked_accuracy": acc,
                       "ms_per_step": step_time / max(step, 1) * 1000, "curve": curve, "args": vars(args)},
                      open(os.path.join(args.out, "result.json"), "w"), indent=1)
        if step == args.steps:
            break
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / (args.steps - args.warmup)))
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = corpus.batch("train", args.batch_size, args.seq_len, args.mlm_prob, g)
        x, y = x.to(args.device), y.to(args.device)
        t0 = time.perf_counter()
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step_time += time.perf_counter() - t0
        run_loss += loss.item(); run_n += 1


def speed(args):
    kernel = os.path.join(ROOT, "kernel", "ar_bench")
    assert os.path.exists(kernel), "build the kernel first: gcc -O3 -march=native -ffast-math -funroll-loops kernel/ar_bench.c -o kernel/ar_bench -lm"
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for width in args.widths:
        row = {"width": width}
        for arch, tokens in (("mixed", 60000), ("mamba2", max(2000, 60000 * 128 * 128 // (width * width * 4)))):
            torch.manual_seed(0)
            a = argparse.Namespace(d_model=width, gts_layers=4, gts_depth=deep_depth(width), gts_trees=4, gts_heads=1, gts_act="gelu",
                                   gts_state=16, gts_read=False, gts_write_key=False, gts_act_bits=8, gts_route_ste=True, gts_route_temp=1.0,
                                   bank_trees=32, bank_heads=8, bank_state=16, ternary_group=128, m2_layers=5, m2_state=32, m2_headdim=32, m2_act_bits=None)
            model = S.build(arch, 65, a)
            path = os.path.join(args.out, f"speed_{arch}_{width}.bin")
            S.export(model, arch, path, torch.randint(0, 65, (256,)))
            times = []
            for _ in range(args.repeats):
                out = subprocess.run([kernel, path, str(tokens)], capture_output=True, text=True).stdout
                times.append(float(re.search(r"speed: ([0-9.]+) us", out).group(1)))
            row[arch] = {"params": sum(p.numel() for p in model.parameters()), "us_per_token": sorted(times)[len(times) // 2],
                         "top1_agree": int(re.search(r"same top-1 on (\d+)", out).group(1))}
            os.remove(path)
        rows.append(row)
        m, b = row["mixed"], row["mamba2"]
        print(f"width {width:4d}   mixed forest {m['params'] / 1e6:5.2f}M  {m['us_per_token']:7.1f} us   Mamba-2 {b['params'] / 1e6:5.2f}M  {b['us_per_token']:7.1f} us"
              f"   ratio {b['us_per_token'] / m['us_per_token']:.1f}x   (kernel top-1 agreement {m['top1_agree']}/256, {b['top1_agree']}/256)", flush=True)
    json.dump(rows, open(os.path.join(args.out, "speed.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "speed"])
    p.add_argument("--arch", choices=["mixed", "mamba2"], default="mixed")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--widths", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--data", default=os.path.join(ROOT, "data", "tinyshakespeare.txt"))
    p.add_argument("--out", default="runs/width")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    speed(args) if args.mode == "speed" else train(args)


if __name__ == "__main__":
    main()
