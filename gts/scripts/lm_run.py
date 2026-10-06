# Golden Tree Snake (GTS) fork, 2026.
"""Token-level autoregressive run at scale: the ternary mixed-forest GTS against a ternary Mamba-2.

    python scripts/prepare_fineweb.py --out data/fineweb
    python scripts/lm_run.py --arch mixed  --data data/fineweb --out runs/fw_mixed
    python scripts/lm_run.py --arch mamba2 --data data/fineweb --out runs/fw_mamba2

The defaults build about half a billion parameters for either model at width 1024 (roughly 450M in the mixers plus
51M of tied embeddings) and train for 2,000 steps. Both models share the wrapper, optimiser, schedule and batches,
and both have ternary weights; the mixed forest also uses 8-bit activations, as in the small runs.

Each run writes ``result.json``, ``model.pt`` and ``model.bin``. The last is for the C kernel, which checks its
logits against PyTorch's and times token-at-a-time inference on one CPU core:

    gcc -O3 -march=native -ffast-math -funroll-loops kernel/ar_bench.c -o kernel/ar_bench -lm
    kernel/ar_bench runs/fw_mixed/model.bin 2000      # keep the token count small: the output head is slow

Memory: both training paths are quadratic in sequence length and neither is frugal. ``--checkpoint`` (on by
default) recomputes each layer in the backward pass so only one layer's activations are held at a time. If a run
still does not fit, lower ``--batch-size`` and raise ``--grad-accum``, or shorten ``--seq-len``.

Written and smoke-tested on a CPU at toy size. It has never been run at the default size or on a GPU.
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
from torch.utils.checkpoint import checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import shakespeare_ar as S  # noqa: E402


def build(args, vocab):
    """Both models come from shakespeare_ar.build, so its export and the C kernel work unchanged."""
    a = argparse.Namespace(
        d_model=args.width, ternary_group=128,
        # mixed forest: a bank of depth-0 trees for context plus deep stateless trees
        gts_layers=args.layers, gts_depth=args.deep_depth, gts_trees=args.deep_trees, gts_heads=1, gts_act="gelu", gts_state=16,
        gts_read=False, gts_write_key=False, gts_act_bits=8, gts_route_ste=True, gts_route_temp=1.0,
        bank_trees=args.bank_trees, bank_heads=args.bank_heads, bank_state=args.bank_state,
        # Mamba-2
        m2_layers=args.m2_layers, m2_state=args.m2_state, m2_headdim=args.m2_headdim, m2_act_bits=None,
    )
    return S.build(args.arch, vocab, a)


def forward(model, ids, targets=None, use_checkpoint=False):
    """TinyLM.forward, optionally recomputing each layer in the backward pass to save memory."""
    x = model.embedding(ids)
    for layer in model.layers:
        x = checkpoint(layer, x, use_reentrant=False) if use_checkpoint else layer(x)
    logits = F.linear(model.norm_f(x), model.embedding.weight, model.head_bias)
    if targets is None:
        return logits
    return logits, F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))


def get_batch(data, batch_size, seq_len, generator, device):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator).tolist()
    x = torch.from_numpy(np.stack([data[s : s + seq_len] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[s + 1 : s + seq_len + 1] for s in starts]).astype(np.int64))
    return x.to(device), y.to(device)  # drawn on the CPU from a seeded generator: both models see the same batches


@torch.no_grad()
def evaluate(model, data, args):
    model.eval()
    g = torch.Generator().manual_seed(999)
    total = 0.0
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.seq_len, g, args.device)
        with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.amp):
            total += forward(model, x, y)[1].item()
    model.train()
    return total / args.eval_batches


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["mixed", "mamba2"], required=True)
    p.add_argument("--data", required=True, help="directory with train.bin, val.bin and meta.json from prepare_fineweb.py")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=1024)
    # mixed forest: 27 layers x (32 bank trees + 4 deep trees of depth 10) is about 456M mixer parameters at width 1024
    p.add_argument("--layers", type=int, default=27)
    p.add_argument("--bank-trees", type=int, default=32)
    p.add_argument("--bank-heads", type=int, default=8)
    p.add_argument("--bank-state", type=int, default=16)
    p.add_argument("--deep-trees", type=int, default=4)
    p.add_argument("--deep-depth", type=int, default=10)
    # Mamba-2: 68 layers of expand 2, state 128 is about 449M mixer parameters at width 1024
    p.add_argument("--m2-layers", type=int, default=68)
    p.add_argument("--m2-state", type=int, default=128)
    p.add_argument("--m2-headdim", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3, help="a guess: nothing at this size has been tuned")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-checkpoint", action="store_true", help="hold every layer's activations (faster, much more memory)")
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast. Untested with the ternary straight-through estimator")
    p.add_argument("--no-tf32", action="store_true", help="CUDA: train with full float32 matmuls instead of TF32")
    p.add_argument("--no-scan-kernel", action="store_true", help="depth-0 trees: the quadratic PyTorch context, not the Triton scan")
    p.add_argument("--no-route-kernel", action="store_true", help="route_ste trees: the dense PyTorch form, not the Triton walk kernels")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--count-only", action="store_true")
    args = p.parse_args()

    meta = json.load(open(os.path.join(args.data, "meta.json")))
    vocab = meta["vocab_size"]
    train_data = np.memmap(os.path.join(args.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data, "val.bin"), dtype=np.uint16, mode="r")

    torch.manual_seed(args.seed)
    model = build(args, vocab)
    n_params = sum(q.numel() for q in model.parameters())
    n_mixer = sum(q.numel() for layer in model.layers for q in layer.mixer.parameters())
    print(f"{args.arch}: {n_params / 1e6:.1f}M parameters ({n_mixer / 1e6:.1f}M in mixers), width {args.width}, "
          f"{len(model.layers)} layers, vocabulary {vocab}", flush=True)
    if args.count_only:
        return
    model.to(args.device)
    for m in model.modules():
        if args.no_scan_kernel and hasattr(m, "scan_kernel"):
            m.scan_kernel = False
        if args.no_route_kernel and hasattr(m, "route_kernel"):
            m.route_kernel = False
    if args.device.startswith("cuda"):
        # TF32 for training, the same for both models. The export below switches it off again, so the logits the
        # C kernel checks against are full float32.
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = not args.no_tf32

    decay = [q for q in model.parameters() if q.ndim >= 2 and not getattr(q, "_no_weight_decay", False)]
    rest = [q for q in model.parameters() if not (q.ndim >= 2 and not getattr(q, "_no_weight_decay", False))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=args.device.startswith("cuda"))  # same update, one kernel
    g = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    use_checkpoint = not args.no_checkpoint
    cuda = args.device.startswith("cuda")
    curve, run_loss, run_n, train_time = [], 0.0, 0, 0.0
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum

    def record(step):
        val = evaluate(model, val_data, args)
        train = run_loss / run_n if run_n else float("nan")
        curve.append([step, val, train])
        rate = step * tokens_per_step / train_time if train_time else 0.0
        mem = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
        print(f"step {step:5d}  val loss {val:.4f}  train {train:.4f}  {rate:,.0f} tokens/s  peak GPU memory {mem:.1f} GB", flush=True)
        json.dump({"arch": args.arch, "params": n_params, "mixer_params": n_mixer, "steps": step, "val_loss": val, "train_loss": train,
                   "train_tokens_per_s": rate, "peak_gpu_gb": mem, "curve": curve, "args": vars(args), "data": meta},
                  open(os.path.join(args.out, "result.json"), "w"), indent=1)

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            record(step)
            run_loss, run_n = 0.0, 0
        if step == args.steps:
            break
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, args.steps - args.warmup)))
        for group in opt.param_groups:
            group["lr"] = lr
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, args.seq_len, g, args.device)
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.amp):
                loss = forward(model, x, y, use_checkpoint)[1]
            (loss / args.grad_accum).backward()
            run_loss += loss.item(); run_n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if cuda:
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0
        if args.log_every and step % args.log_every == 0:
            print(f"  step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  {train_time / (step + 1):.2f} s/step", flush=True)

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    if not args.no_export:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        test_ids = torch.from_numpy(np.asarray(val_data[:256]).astype(np.int64))
        S.export(model, args.arch, os.path.join(args.out, "model.bin"), test_ids)
        print(f"exported {os.path.join(args.out, 'model.bin')}; time it with: kernel/ar_bench {os.path.join(args.out, 'model.bin')} 2000", flush=True)


if __name__ == "__main__":
    main()
