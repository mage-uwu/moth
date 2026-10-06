# Golden Tree Snake (GTS) fork, 2026.
"""Autoregressive pretraining of the causal ternary GTS mixed forest, optionally looped (GTS-Uni-AR), with the masked-LM
path's conveniences (scripts/bert_pretrain.py).

    python scripts/prepare_fineweb.py --out /workspace/fineweb --train-tokens 2000000000 --val-tokens 2000000
    python scripts/ar_pretrain.py --data /workspace/fineweb --out /workspace/ar --minutes 170 [--loops 3]

Model: scripts/shakespeare_ar.TinyLM with the mixed forest (a bank of depth-0 trees for context, deep stateless trees
with the routing gradient), ternary weights, 8-bit activations; with ``--loops`` > 1 the whole stack runs that many times
with shared weights, gated passes and per-pass embeddings (TinyLM's docstring); each step trains with a pass count
drawn from ``--loop-probs`` so every pass count works at run time.

Training: bf16, torch.compile per block (static shapes), fused AdamW, the head padded to a multiple of 64 rows,
optional activation recomputation per block or per pass, a step that is not finite is skipped. The learning-rate
schedule (warmup, cosine to 10%) is fitted to ``--minutes`` once the step rate is measured. ``--resume`` continues a
run (weights, optimizer, step, curve, data sampler; the rate re-warms from the checkpoint's last value);
``--init-from`` takes the weights alone (e.g. a one-pass model's for a looped one: the new parameters start so that
it computes the same function). ``--new-param-lr`` gives the looped model's own parameters their own rate.

Writes to --out: result.json (curve with the validation loss at every pass count), checkpoint.pt (float weights,
optimizer, step, config; also every --ckpt-minutes), binarized.pt (2-bit ternary codes and scales plus the float
tensors), model.bin for kernel/ar_bench.c (checked against PyTorch there) and samples.txt (continuations of a few
prompts at every pass count).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import shakespeare_ar as S  # noqa: E402
from bert_pretrain import uncompiled  # noqa: E402
from mamba_ssm.utils.ternary_pack import save_binarized  # noqa: E402

PROMPTS = ["The capital of France is", "In 1969, the first person to walk on the Moon", "The best way to learn a new language is",
           "def fibonacci(n):"]


def model_config(a, vocab):
    return dict(vocab=vocab, width=a.width, layers=a.layers, bank_trees=a.bank_trees, bank_heads=a.bank_heads,
                bank_state=a.bank_state, deep_trees=a.deep_trees, deep_depth=a.deep_depth, loops=a.loops)


def build(cfg):
    ns = argparse.Namespace(
        d_model=cfg["width"], ternary_group=128, gts_layers=cfg["layers"], gts_depth=cfg["deep_depth"], gts_trees=cfg["deep_trees"],
        gts_heads=1, gts_act="gelu", gts_state=16, gts_read=False, gts_write_key=False, gts_act_bits=8, gts_route_ste=True,
        gts_route_temp=1.0, bank_trees=cfg["bank_trees"], bank_heads=cfg["bank_heads"], bank_state=cfg["bank_state"],
        loops=cfg.get("loops", 1))
    return S.build("mixed", cfg["vocab"], ns)


def logits_of(model, ids, loops=None, checkpoint_layers=False, checkpoint_loops=False):
    """The model's logits with the head padded to a multiple of 64 rows on a GPU (50,257 makes the GEMM misaligned)."""
    h = model.hidden(ids, loops, checkpoint_layers, checkpoint_loops)
    w, b = model.embedding.weight, model.head_bias
    vocab, pad = w.shape[0], -w.shape[0] % 64
    if pad and h.is_cuda:
        w, b = F.pad(w, (0, 0, 0, pad)), F.pad(b, (0, pad))
    return F.linear(h, w, b)[..., :vocab]


def get_batch(data, a, gen, device):
    starts = torch.randint(0, len(data) - a.seq_len - 1, (a.batch_size,), generator=gen).tolist()
    x = torch.from_numpy(np.stack([data[s : s + a.seq_len] for s in starts]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[s + 1 : s + a.seq_len + 1] for s in starts]).astype(np.int64))
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, a, device, loops=None):
    model.eval()
    g = torch.Generator().manual_seed(999)
    total = 0.0
    for _ in range(a.eval_batches):
        x, y = get_batch(data, a, g, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            lg = logits_of(model, x, loops)
        total += F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), y.reshape(-1)).item()
    model.train()
    return total / a.eval_batches


@torch.no_grad()
def samples(model, a, device, path, n_new=48):
    """Greedy continuations of a few prompts at every pass count (re-running the whole prefix each token: slow, short)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception as e:  # no tokenizer: say so rather than fail at the end of a run
        open(path, "w").write(f"no samples: {e}\n")
        return
    model.eval()
    lines = []
    for n in range(1, model.loops + 1):
        lines.append(f"=== {n} pass{'es' if n > 1 else ''}")
        for p in PROMPTS:
            ids = torch.tensor([enc.encode(p)], device=device)
            for _ in range(n_new):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
                    nxt = logits_of(model, ids[:, -a.seq_len :], n)[0, -1].argmax()
                ids = torch.cat([ids, nxt.view(1, 1)], 1)
            lines.append(repr(enc.decode(ids[0].tolist())))
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[: len(PROMPTS) + 1]), flush=True)
    model.train()


def train(a):
    t_start = time.time()
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    train_data = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.amp = a.amp and device == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    cfg = model_config(a, meta["vocab_size"])
    model = build(cfg).to(device)
    ck = None
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        assert ck["config"] == cfg, f"the checkpoint's model differs: {ck['config']} vs {cfg}"
        model.load_state_dict(ck["model"])
        print(f"resumed from {a.resume} at step {ck['step']}", flush=True)
    elif a.init_from:
        src = torch.load(a.init_from, map_location="cpu", weights_only=False)
        state = src["model"] if "model" in src else src  # checkpoint.pt, or lm_run.py's model.pt (a bare state dict)
        missing, unexpected = model.load_state_dict({k.replace("._orig_mod", ""): v for k, v in state.items()}, strict=False)
        assert not unexpected and set(missing) <= {"loop_embed", "loop_gate"}, f"init-from mismatch: {missing}, {unexpected}"
        print(f"weights from {a.init_from}; new parameters: {sorted(missing)}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GTS{'-Uni-AR' if a.loops > 1 else ''} causal LM: {n_params / 1e6:.1f}M parameters, width {a.width}, {a.layers} layers"
          f"{f', {a.loops} passes' if a.loops > 1 else ''}; {meta.get('source', '')}, {meta['train_tokens']:,} training tokens", flush=True)
    if a.compile and device == "cuda":
        for i in range(len(model.layers)):
            model.layers[i] = torch.compile(model.layers[i], dynamic=False)
    uni = {"loop_embed", "loop_gate"}
    named = [(n.replace("._orig_mod", ""), p) for n, p in model.named_parameters()]
    new = [p for n, p in named if n in uni] if a.new_param_lr else []
    old = [p for n, p in named if all(p is not q for q in new)]
    groups = [{"params": [p for p in old if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)], "weight_decay": a.weight_decay, "lr_scale": 1.0},
              {"params": [p for p in old if not (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False))], "weight_decay": 0.0, "lr_scale": 1.0}]
    if new:
        groups.append({"params": new, "weight_decay": 0.0, "lr_scale": a.new_param_lr / a.lr})
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=device == "cuda")
    os.makedirs(a.out, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed)
    tokens_per_step = a.batch_size * a.seq_len
    total_steps, curve, run_loss, run_n, last_ckpt, t_rate = None, [], 0.0, 0, time.time(), None
    s0, lr0, warm, phases = 0, 0.0, a.warmup, []
    if ck is not None:
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["generator"])
        s0, curve, phases = ck["step"], ck["curve"], ck.get("phases", [])
        lr0, warm = ck["optimizer"]["param_groups"][0]["lr"], a.rewarm
    phases = phases + [dict(vars(a))]

    def lr_at(step):
        k = step - s0
        if k < warm:
            return lr0 + (a.lr - lr0) * (k + 1) / warm
        if total_steps is None:
            return a.lr
        frac = min(1.0, (k - warm) / max(1, total_steps - s0 - warm))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    def plain_state():
        return {k.replace("._orig_mod", ""): v for k, v in model.state_dict().items()}

    def save_float(step):
        torch.save({"model": plain_state(), "optimizer": opt.state_dict(), "step": step, "config": cfg, "args": vars(a),
                    "curve": curve, "generator": gen.get_state(), "phases": phases}, os.path.join(a.out, "checkpoint.pt.tmp"))
        os.replace(os.path.join(a.out, "checkpoint.pt.tmp"), os.path.join(a.out, "checkpoint.pt"))

    def record(step):
        with uncompiled(model.layers):
            vl = evaluate(model, val_data, a, device)
            by = {n: evaluate(model, val_data, a, device, loops=n) for n in range(1, a.loops)}
        tl = run_loss / run_n if run_n else float("nan")
        el = time.time() - t_start
        curve.append({"step": step, "tokens": step * tokens_per_step, "val_loss": vl, "train_loss": tl, "minutes": el / 60,
                      "phase": len(phases), **({"val_by_loops": by} if by else {})})
        print(f"step {step:6d}  tokens {step * tokens_per_step / 1e6:8.1f}M  val loss {vl:.4f}  train {tl:.4f}  {el / 60:6.1f} min"
              + ("  fewer passes: " + "  ".join(f"{n}: {v:.4f}" for n, v in by.items()) if by else ""), flush=True)
        json.dump({"params": n_params, "config": cfg, "args": vars(a), "phases": phases, "data": meta, "total_steps": total_steps,
                   "curve": curve}, open(os.path.join(a.out, "result.json"), "w"), indent=1)

    loop_probs = [float(v) for v in a.loop_probs.split(",")] if a.loops > 1 else [1.0]
    assert len(loop_probs) == a.loops, "--loop-probs needs one probability per pass count"
    loop_rng = random.Random(a.seed + s0)
    step, skipped = s0, 0
    if a.init_from and ck is None and a.loops > 1:
        record(step)  # where the warm start begins
    model.train()
    while True:
        if step % a.eval_every == 0 and step > s0:
            record(step)
            run_loss, run_n = 0.0, 0
        if total_steps is not None and step >= total_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g.get("lr_scale", 1.0)
        x, y = get_batch(train_data, a, gen, device)
        n = loop_rng.choices(range(1, a.loops + 1), weights=loop_probs)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            lg = logits_of(model, x, n, a.checkpoint_layers, a.checkpoint_loops)
        loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        gnorm = loss
        if torch.isfinite(loss):
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gnorm):
            opt.step()
        else:
            skipped += 1
            if skipped <= 10:
                print(f"  step {step}: non-finite loss or gradient; skipped", flush=True)
        step += 1
        if step % a.log_every == 0:
            run_loss += loss.item()
            run_n += 1
            print(f"  step {step:6d}  loss {loss.item():.4f}  passes {n}  lr {lr_at(step):.2e}  {(time.time() - t_start) / 60:.1f} min", flush=True)
        if step - s0 == a.rate_from:
            if device == "cuda":
                torch.cuda.synchronize()
            t_rate = time.time()
        if step - s0 == a.rate_from + a.rate_steps:
            if device == "cuda":
                torch.cuda.synchronize()
            rate = a.rate_steps / (time.time() - t_rate)
            left = a.minutes * 60 - (time.time() - t_start) - a.reserve_minutes * 60
            ev = a.eval_batches * a.loops / 3  # forward-only evaluation batches (every pass count), a third of a step each
            share = ev / (a.eval_every + ev)
            total_steps = step + max(0, int(left * (1 - share) * rate))
            print(f"  {rate:.2f} steps/s = {rate * tokens_per_step:,.0f} tokens/s; schedule fitted to {total_steps} steps, "
                  f"{total_steps - s0} in this phase ({(total_steps - s0) * tokens_per_step / 1e9:.2f}B tokens)", flush=True)
        if time.time() - last_ckpt > a.ckpt_minutes * 60:
            save_float(step)
            last_ckpt = time.time()
            print(f"  checkpoint at step {step}", flush=True)

    record(step)
    save_float(step)
    plain = build(cfg)
    plain.load_state_dict(plain_state())
    save_binarized(plain, cfg, os.path.join(a.out, "binarized.pt"))
    plain = plain.to(device)
    samples(plain, a, device, os.path.join(a.out, "samples.txt"))
    if not a.no_export:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        S.export(plain.float(), "mixed", os.path.join(a.out, "model.bin"), torch.from_numpy(np.asarray(val_data[:256]).astype(np.int64)))
    print(f"saved checkpoint.pt, binarized.pt{'' if a.no_export else ', model.bin'} and samples.txt to {a.out}; "
          f"{(time.time() - t_start) / 60:.1f} min in all", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="prepare_fineweb.py output: train.bin, val.bin, meta.json")
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=140, help="wall-clock budget for this command, all included")
    p.add_argument("--reserve-minutes", type=float, default=6)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--layers", type=int, default=14)
    p.add_argument("--bank-trees", type=int, default=32)
    p.add_argument("--bank-heads", type=int, default=8)
    p.add_argument("--bank-state", type=int, default=16)
    p.add_argument("--deep-trees", type=int, default=4)
    p.add_argument("--deep-depth", type=int, default=9)
    p.add_argument("--loops", type=int, default=1, help="GTS-Uni-AR: passes of the whole stack with shared weights")
    p.add_argument("--loop-probs", default="0.1,0.2,0.7", help="probability of training a step with 1, 2, ... passes")
    p.add_argument("--new-param-lr", type=float, help="peak learning rate of the pass embeddings and gates (default: --lr)")
    p.add_argument("--checkpoint-layers", action="store_true", help="recompute every block in the backward pass")
    p.add_argument("--checkpoint-loops", action="store_true", help="recompute every pass after the first")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--resume", help="float checkpoint.pt to continue: weights, optimizer, step, curve, sampler")
    p.add_argument("--rewarm", type=int, default=1000, help="on resume: steps from the checkpoint's last rate to --lr")
    p.add_argument("--init-from", help="weights only (checkpoint.pt, or lm_run.py's model.pt)")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-minutes", type=float, default=30)
    p.add_argument("--rate-from", type=int, default=100)
    p.add_argument("--rate-steps", type=int, default=100)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
