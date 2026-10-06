# Golden Tree Snake (GTS) fork, 2026.
"""Character-level autoregressive LM on TinyShakespeare: ternary GTS against a parameter-matched ternary Mamba-2.

    python scripts/shakespeare_ar.py --data input.txt --arch gts    --out runs/gts
    python scripts/shakespeare_ar.py --data input.txt --arch mamba2 --out runs/mamba2

Both models share everything except the mixer: the same embedding, pre-norm residual blocks, tied head,
optimiser, schedule, batches and step count. Both are trained ternary from scratch with the same
quantiser (grouped absmean, straight-through). Each run writes ``model.bin`` for ``kernel/ar_bench.c``.
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mamba_ssm.models.gts_encoder import RMSNorm  # noqa: E402
from mamba_ssm.modules.gts import GTS  # noqa: E402
from mamba_ssm.modules.ternary import absmean_ternary, quantize_activations, zero_ratio  # noqa: E402


class Mamba2Ref(nn.Module):
    """Mamba-2 in plain PyTorch, so it runs on CPU.

    The non-fused path of ``modules/mamba2.py`` (defaults: rmsnorm, gate before norm, one D per head, ngroups=1)
    with the SSD scan written in its quadratic form over the whole sequence, as in ``modules/ssd_minimal.py``.
    ``ternary=True`` quantises in_proj and out_proj and nothing else, which is Ternary Mamba's coverage.
    """

    def __init__(self, d_model, d_state=32, d_conv=4, expand=2, headdim=32, d_inner=None, ternary=False, ternary_group=128, act_bits=None,
                 A_init_range=(1, 16), dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.headdim = d_model, d_state, d_conv, headdim
        self.d_inner = d_inner or expand * d_model
        assert self.d_inner % headdim == 0
        self.nheads = self.d_inner // headdim
        self.ternary, self.ternary_group = ternary, ternary_group
        self.act_bits = act_bits  # if set, the inputs of in_proj and out_proj are quantised per token (BitNet style)

        # Order: [z, x, B, C, dt]
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state + self.nheads, bias=False)
        conv_dim = self.d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, bias=True)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.nheads).uniform_(*A_init_range)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm_weight = nn.Parameter(torch.ones(self.d_inner))
        for p in (self.dt_bias, self.A_log, self.D):
            p._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _q(self, w):
        return absmean_ternary(w, self.ternary_group) if self.ternary else w

    def forward(self, u):
        b, l, _ = u.shape
        di, n, h, p = self.d_inner, self.d_state, self.nheads, self.headdim
        if self.act_bits:
            u = quantize_activations(u, self.act_bits)
        z, xBC, dt = torch.split(F.linear(u, self._q(self.in_proj.weight)), [di, di + 2 * n, h], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C = torch.split(xBC, [di, n, n], dim=-1)
        dt = F.softplus(dt + self.dt_bias)  # (b, l, h)
        a = dt * -torch.exp(self.A_log)  # log-decay per token per head
        cs = torch.cumsum(a, dim=1)
        seg = cs[:, :, None, :] - cs[:, None, :, :]  # [t, s] = sum_{r=s+1..t} a_r
        keep = torch.ones(l, l, dtype=torch.bool, device=u.device).tril(0)[None, :, :, None]
        decay = torch.exp(seg.masked_fill(~keep, -torch.inf))  # (b, t, s, h); the diagonal is 1
        w = ((C @ B.transpose(1, 2)).unsqueeze(-1) * decay).permute(0, 3, 1, 2)  # (b, h, t, s)
        xh = x.view(b, l, h, p)
        y = (w @ (xh * dt.unsqueeze(-1)).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)  # (b, t, h, p)
        y = (y + self.D.view(1, 1, h, 1) * xh).reshape(b, l, di)
        y = y * F.silu(z)
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-5) * self.norm_weight
        if self.act_bits:
            y = quantize_activations(y, self.act_bits)
        return F.linear(y, self._q(self.out_proj.weight))


class Hybrid(nn.Module):
    """A GTS tree with a small dense Mamba-2 block beside it: the tree keeps most of the parameters, the trunk
    gives every token a dense channel to the rest of the sequence. Their outputs are summed."""

    def __init__(self, tree, trunk):
        super().__init__()
        self.tree, self.trunk = tree, trunk

    def forward(self, u):
        return self.tree(u) + self.trunk(u)


class Block(nn.Module):
    def __init__(self, d_model, mixer):
        super().__init__()
        self.norm, self.mixer = RMSNorm(d_model), mixer

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class TinyLM(nn.Module):
    """The shared wrapper. Only ``make_mixer`` differs between the two models.

    ``loops`` > 1 makes it a GTS-Uni-AR: the whole stack runs ``loops`` times with the same weights; passes after the
    first add a learned per-pass embedding and add their update through a per-channel gate initialised to zero, so a
    fresh looped model computes exactly what its one-pass weights do. Fewer passes can be chosen at run time. (No latent
    tokens, unlike the masked-LM GTS-Uni: in a causal model, tokens at the start see only the start.)"""

    def __init__(self, vocab, d_model, n_layer, make_mixer, loops=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)
        self.layers = nn.ModuleList([Block(d_model, make_mixer()) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.head_bias = nn.Parameter(torch.zeros(vocab))
        self.loops = loops
        if loops > 1:
            self.loop_embed = nn.Parameter(torch.zeros(loops - 1, d_model))
            self.loop_gate = nn.Parameter(torch.zeros(loops - 1, d_model))
            for q in (self.loop_embed, self.loop_gate):
                q._no_weight_decay = True

    def _stack(self, x, checkpoint_layers=False):
        for layer in self.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False) if checkpoint_layers else layer(x)
        return x

    def hidden(self, ids, loops=None, checkpoint_layers=False, checkpoint_loops=False):
        """Final normed states. ``checkpoint_layers`` recomputes every block in the backward pass, ``checkpoint_loops``
        every pass after the first."""
        n = self.loops if loops is None else loops
        assert 1 <= n <= self.loops
        x = self._stack(self.embedding(ids), checkpoint_layers)
        for t in range(n - 1):
            xin = x + self.loop_embed[t]
            if checkpoint_loops and torch.is_grad_enabled():
                y = torch.utils.checkpoint.checkpoint(self._stack, xin, checkpoint_layers, use_reentrant=False)
            else:
                y = self._stack(xin, checkpoint_layers)
            x = x + self.loop_gate[t] * (y - xin)
        return self.norm_f(x)

    def forward(self, ids, targets=None, loops=None):
        logits = F.linear(self.hidden(ids, loops), self.embedding.weight, self.head_bias)
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


def build(arch, vocab, args):
    def tree():
        return GTS(args.d_model, depth=args.gts_depth, n_trees=args.gts_trees, n_heads=args.gts_heads, act=args.gts_act, d_state=args.gts_state,
                   route_ste=args.gts_route_ste, route_ste_temp=args.gts_route_temp,
                   d_conv=3, causal=True, read_state=args.gts_read, write_logit=not args.gts_write_key, act_bits=args.gts_act_bits,
                   ternary=True, ternary_group=args.ternary_group)

    if arch == "gts":
        return TinyLM(vocab, args.d_model, args.gts_layers, tree)
    if arch == "mixed":
        # A forest of mixed depths, built as two GTS mixers whose outputs are summed: a bank of depth-0 trees
        # (every token visits every one, so they carry the context) and a few deep trees with no state at all
        # (they carry most of the parameters and touch few of them).
        common = dict(d_conv=3, causal=True, ternary=True, ternary_group=args.ternary_group, act_bits=args.gts_act_bits)
        bank = lambda: GTS(args.d_model, depth=0, n_trees=args.bank_trees, n_heads=args.bank_heads, d_state=args.bank_state, act="split", **common)
        deep = lambda: GTS(args.d_model, depth=args.gts_depth, n_trees=args.gts_trees, use_context=False, dense_walk=True,
                           route_ste=args.gts_route_ste, route_ste_temp=args.gts_route_temp, **common)
        return TinyLM(vocab, args.d_model, args.gts_layers, lambda: Hybrid(bank(), deep()), loops=getattr(args, "loops", 1))
    if arch == "hybrid":
        trunk = lambda: Mamba2Ref(args.d_model, d_state=args.trunk_state, d_inner=args.trunk_inner,
                                  headdim=args.trunk_headdim, ternary=True, ternary_group=args.ternary_group)
        return TinyLM(vocab, args.d_model, args.gts_layers, lambda: Hybrid(tree(), trunk()))
    make = lambda: Mamba2Ref(args.d_model, d_state=args.m2_state, expand=2, headdim=args.m2_headdim,
                             ternary=True, ternary_group=args.ternary_group, act_bits=args.m2_act_bits)
    return TinyLM(vocab, args.d_model, args.m2_layers, make)


def get_batch(data, batch_size, seq_len, generator, device="cpu"):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x.to(device), y.to(device)  # batches are drawn on the CPU so every device sees the same ones


@torch.no_grad()
def evaluate(model, data, args, batches=40):
    model.eval()
    g = torch.Generator().manual_seed(999)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, args.batch_size, args.seq_len, g, getattr(args, "device", "cpu"))
        total += model(x, y)[1].item()
    model.train()
    return total / batches


def _gts_dims(m):
    return [m.depth, m.d_state, m.d_conv, int(m.read_state), int(m.write_logit), m.n_trees, ["gelu", "linear", "split"].index(m.act), m.n_heads]


def _gts_tensors(m):
    t = [m._w_in(), m.node_bias, m._w_out(), m.conv1d.weight.squeeze(1), m.conv1d.bias]
    if m.use_context:
        t += [m._w_ctx(), m.dt_bias, m.A_log]
    return t + ([m.read_norm_weight, m._q(m.read_w)] if m.read_state else [])


def _m2_dims(m):
    return [m.d_inner, m.d_state, m.nheads, m.headdim, m.d_conv]


def _m2_tensors(m):
    return [m._q(m.in_proj.weight), m.conv1d.weight.squeeze(1), m.conv1d.bias, m.dt_bias, m.A_log, m.D, m.norm_weight,
            m._q(m.out_proj.weight)]


def export(model, arch, path, test_ids):
    """Float32 weights (ternary tensors written as their effective scale * code values) plus a test sequence and
    the PyTorch logits for it, so the C kernel can prove it computes the same function.
    Format codes: 1 = Mamba-2, 2 = GTS, 3 = GTS with a Mamba-2 trunk."""
    def w(f, t):
        f.write(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())

    model.eval()
    with torch.no_grad(), open(path, "wb") as f:
        vocab, d = model.embedding.weight.shape
        mix = model.layers[0].mixer
        if arch == "gts":
            dims = [2, vocab, d, len(model.layers)] + _gts_dims(mix) if not mix.act_bits else [4, vocab, d, len(model.layers)] + _gts_dims(mix) + [mix.act_bits]
        elif arch == "mixed":  # format 6: two GTS mixers per layer, ten ints each; 7: the same looped (GTS-Uni-AR)
            dims = [6, vocab, d, len(model.layers)] + sum(([*_gts_dims(m), m.act_bits or 0, int(m.use_context)] for m in (mix.tree, mix.trunk)), [])
            if getattr(model, "loops", 1) > 1:
                dims = [7] + dims[1:] + [model.loops]
        elif arch == "hybrid":
            dims = [3, vocab, d, len(model.layers)] + _gts_dims(mix.tree) + _m2_dims(mix.trunk)
        else:
            dims = [1, vocab, d, len(model.layers)] + _m2_dims(mix) if not mix.act_bits else [5, vocab, d, len(model.layers)] + _m2_dims(mix) + [mix.act_bits]
        f.write(struct.pack(f"{len(dims)}i", *dims))
        w(f, model.embedding.weight); w(f, model.head_bias); w(f, model.norm_f.weight)
        for layer in model.layers:
            m = layer.mixer
            w(f, layer.norm.weight)
            tensors = (_gts_tensors(m) if arch == "gts" else _m2_tensors(m) if arch == "mamba2"
                       else _gts_tensors(m.tree) + (_gts_tensors(m.trunk) if arch == "mixed" else _m2_tensors(m.trunk)))
            for t in tensors:
                w(f, t)
        if getattr(model, "loops", 1) > 1:
            w(f, model.loop_embed); w(f, model.loop_gate)
        f.write(struct.pack("i", len(test_ids)))
        f.write(test_ids.numpy().astype(np.int32).tobytes())
        w(f, model(test_ids.unsqueeze(0).to(model.head_bias.device))[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--arch", choices=["gts", "mamba2", "hybrid", "mixed"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--gts-layers", type=int, default=4)
    p.add_argument("--gts-depth", type=int, default=8)
    p.add_argument("--gts-state", type=int, default=16)
    p.add_argument("--gts-trees", type=int, default=1)
    p.add_argument("--gts-heads", type=int, default=1)
    p.add_argument("--gts-route-ste", action="store_true", help="straight-through gradient for branch decisions")
    p.add_argument("--gts-route-temp", type=float, default=1.0)
    p.add_argument("--bank-trees", type=int, default=32, help="mixed: depth-0 trees that carry context")
    p.add_argument("--bank-heads", type=int, default=8)
    p.add_argument("--bank-state", type=int, default=16)
    p.add_argument("--gts-act-bits", type=int, default=None, help="quantise each mixer's input to this many bits per token")
    p.add_argument("--gts-act", choices=["gelu", "linear", "split"], default="gelu")
    p.add_argument("--gts-read", action="store_true", help="read whole node states into the output")
    p.add_argument("--gts-write-key", action="store_true", help="write dt * B instead of dt * logit * B")
    p.add_argument("--trunk-inner", type=int, default=64, help="hybrid: channels of the dense trunk")
    p.add_argument("--trunk-state", type=int, default=16)
    p.add_argument("--trunk-headdim", type=int, default=16)
    p.add_argument("--m2-layers", type=int, default=5)
    p.add_argument("--m2-act-bits", type=int, default=None)
    p.add_argument("--m2-state", type=int, default=32)
    p.add_argument("--m2-headdim", type=int, default=32)
    p.add_argument("--ternary-group", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--count-only", action="store_true")
    args = p.parse_args()

    text = open(args.data).read()
    chars = sorted(set(text))
    data = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    torch.manual_seed(args.seed)
    model = build(args.arch, len(chars), args).to(args.device)
    n_params = sum(q.numel() for q in model.parameters())
    n_mixer = sum(q.numel() for layer in model.layers for q in layer.mixer.parameters())
    print(f"{args.arch}: {n_params:,} parameters ({n_mixer:,} in mixers), vocab {len(chars)}", flush=True)
    if args.count_only:
        return

    decay = [q for q in model.parameters() if q.ndim >= 2 and not getattr(q, "_no_weight_decay", False)]
    rest = [q for q in model.parameters() if not (q.ndim >= 2 and not getattr(q, "_no_weight_decay", False))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(args.seed)  # both models see the same batches
    os.makedirs(args.out, exist_ok=True)
    curve, step_time, run_loss, run_n = [], 0.0, 0.0, 0
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            val = evaluate(model, val_data, args)
            train = run_loss / run_n if run_n else float("nan")  # mean training loss since the last evaluation
            curve.append([step, val, train])
            print(f"step {step:5d}  val loss {val:.4f}  train {train:.4f}  ({val / math.log(2):.3f} bits/char)  {step_time / max(step, 1) * 1000:.0f} ms/step", flush=True)
            run_loss, run_n = 0.0, 0
        if step == args.steps:
            break
        lr = args.lr * (step + 1) / args.warmup if step < args.warmup else \
            args.lr * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / (args.steps - args.warmup)))
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, args.seq_len, g, args.device)
        t0 = time.perf_counter()
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step_time += time.perf_counter() - t0
        run_loss += loss.item(); run_n += 1

    train_loss = evaluate(model, train_data, args)
    zeros = []
    for m in model.modules():
        if isinstance(m, GTS):
            zeros += [zero_ratio(t, m.ternary_group) for t in [m.node_in, m.node_out] + ([m.ctx_proj.weight] if m.use_context else []) + ([m.read_w] if m.read_state else [])]
        elif isinstance(m, Mamba2Ref):
            zeros += [zero_ratio(t, m.ternary_group) for t in (m.in_proj.weight, m.out_proj.weight)]
    result = {"arch": args.arch, "params": n_params, "mixer_params": n_mixer, "steps": args.steps,
              "val_loss": curve[-1][1], "val_bits_per_char": curve[-1][1] / math.log(2), "train_loss": train_loss,
              "ms_per_step": step_time / args.steps * 1000, "zero_ratio": sum(zeros) / len(zeros), "curve": curve}
    trees = [m for m in model.modules() if isinstance(m, GTS)]
    if trees:
        result["tree_ops_per_token_per_layer"] = trees[0].ops_per_token()["total"]
    result["args"] = vars(args)
    json.dump(result, open(os.path.join(args.out, "result.json"), "w"), indent=1)
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    export(model, args.arch, os.path.join(args.out, "model.bin"), val_data[:256])

    model.eval()
    ids = val_data[:1].clone().to(args.device)
    with torch.no_grad():
        for _ in range(200):
            nxt = torch.multinomial(F.softmax(model(ids[-args.seq_len:].unsqueeze(0))[0, -1] / 0.8, dim=-1), 1)
            ids = torch.cat([ids, nxt])
    sample = "".join(chars[i] for i in ids.tolist())
    open(os.path.join(args.out, "sample.txt"), "w").write(sample)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items() if k != "curve"})
    print("--- sample ---\n" + sample, flush=True)


if __name__ == "__main__":
    main()
