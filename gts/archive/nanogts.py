# Golden Tree Snake (GTS) fork, 2026.
"""nanoGTS: GTS3 in one file. A bidirectional, attention-free, ternary masked-language model, its data and its training.

GTS3 (checkpoints/bert110m/phase3) is the best GTS masked LM: 14 layers of width 768, 112.9M parameters, trained on
7.38B tokens of English Wikipedia to validation loss 2.615 / masked accuracy 53.2%. This file is the method in plain
PyTorch, with no Triton and nothing from the rest of the repository. Its parameter names are the repository's, so the
GTS3 checkpoints load as they are (``load``), and it computes the same function as mamba_ssm/models/gts_encoder.py
with the same gradients. Checked on GTS3's own weights in float64 against the repository model: logits agree to
3e-14, the loss exactly, and all 199 parameter gradients (the straight-through ones included) to 3e-14. In float32 a
few tokens differ: a branch logit within rounding of zero can turn the other way when sums run in another order.
``selftest`` checks the fast paths here against their definitions.

The model
---------
token embedding (tied with the output layer, no positions) -> 14 x [x + Mixer(RMSNorm(x))] -> RMSNorm -> logits.
There is no attention and no MLP. Each block's mixer is a forest of binary trees in the manner of fast feedforward
networks (Belcak & Wattenhofer, 2023): a node computes logit = <x, node_in> + bias and adds coef * node_out to the
output; a token walks one root-to-leaf path (right if logit > 0). The forest has two kinds of tree, summed:

* bank: 32 depth-0 trees (every token visits all of them), so they are dense channels. They carry all the context:
  each tree is a channel of a bidirectional Mamba-2-style SSM. Token s writes dt_s * logit_s * B_s into the tree's
  state; token t reads <C_t, decayed state> from both directions, excluding itself:
      ctx[t] = sum_{s<t} exp(a_{s+1}+..+a_t) <C_fwd[t], B[s]> dt_s logit_s + sum_{s>t} exp(a_t+..+a_{s-1}) <C_bwd[t], B[s]> dt_s logit_s
  with a = -exp(A_log) * softplus(dt) <= 0 on one clock per head (8 heads of 4 trees, state 16). B, C_fwd, C_bwd and
  dt come from one small projection (ctx_proj). coef = gelu(logit) + ctx.
* deep: 4 stateless trees of depth 9 (1,023 nodes each, 10 visited per token). They hold most of the mixer's
  parameters and touch 1% of them per token. coef = gelu(logit). Branches train with a straight-through gradient: the
  hard step going forward, sigmoid(logit) going backward, so a branch logit learns from the difference between the
  outputs of its two subtrees, each followed down by the token's own decisions.

Each kind first mixes neighbours with its own centred depthwise conv (width 3, identity at init) and quantises the
result to 8-bit integers per token. node_in, node_out and ctx_proj are ternary: absmean codes in {-1, 0, 1} times
one scale per group of 128 weights, recomputed every step from latent float weights (straight-through). Embeddings,
norms, convs, biases and the decay parameters stay in float.

The training (GTS3)
-------------------
RoBERTa-style masked LM on English Wikipedia (wikimedia/wikipedia 20231101.en; validation from the last shard) in
BERT's uncased WordPiece: 512-token windows of the article stream starting with [CLS], 15% of tokens chosen, of those
80% [MASK], 10% random, 10% kept; loss at the chosen positions only. AdamW (0.9, 0.98, eps 1e-6, weight decay 0.01 on
matrices), batch 64 x 512, gradient clipping 1.0, bf16 autocast, linear warmup then cosine to 10% of the peak.
Three phases, each resuming the previous one's weights, optimizer and sampler, re-warming from its last rate:

    python archive/nanogts.py prep  --out wiki1 --shards 0-6
    python archive/nanogts.py train --data wiki1 --out run --steps 53753  --lr 1.5e-3 --warmup 1000               # 2.815 / 50.5%
    python archive/nanogts.py prep  --out wiki2 --shards 7-19
    python archive/nanogts.py train --data wiki2 --out run --steps 108829 --lr 7.5e-4 --warmup 1000 --resume run/ckpt.pt  # 2.713 / 51.7%
    python archive/nanogts.py prep  --out wiki3 --shards 0-39
    python archive/nanogts.py train --data wiki3 --out run --steps 225197 --lr 5e-4  --warmup 2000 --resume run/ckpt.pt  # 2.615 / 53.2%

(--steps is the absolute step to end at. GTS3 fitted each phase's length to a time budget on one A100; these are the
step counts it reached: 1.76B + 1.80B + 3.81B tokens.)

    python archive/nanogts.py sample --ckpt run/ckpt.pt          # fill-mask examples
    python archive/nanogts.py selftest                           # checks the fast paths against their definitions

Faithful to GTS3, not to its speed: the repository trains with Triton kernels (a chunked scan for the bank, a tree
walk for the deep trees). Here the bank's context is the quadratic form (memory ~ length^2 per head; use
--micro-batch on smaller GPUs, gradients are accumulated exactly) and the deep trees' straight-through gradient is a
torch.autograd.Function that walks the trees with gathers, as the kernel does.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------------------- model


@dataclass
class GTSConfig:
    d_model: int = 768
    n_layer: int = 14
    vocab_size: int = 30522
    bank_trees: int = 32
    bank_heads: int = 8
    bank_state: int = 16
    deep_trees: int = 4
    deep_depth: int = 9
    d_conv: int = 3
    ternary_group: int = 128
    act_bits: int = 8
    route_ste_temp: float = 1.0
    pad_token_id: int = 0
    norm_eps: float = 1e-5


def group_size(n, g):
    """Largest group size <= g that divides n."""
    if g is None or g >= n:
        return n
    while n % g:
        g -= 1
    return g


def ternary(w, group):
    """BitNet b1.58's absmean quantiser per group of weights along each row, straight-through to the latent weights."""
    g = group_size(w.shape[-1], group)
    wg = w.reshape(*w.shape[:-1], -1, g)
    scale = wg.abs().mean(-1, keepdim=True).clamp(min=1e-8)
    wq = ((wg / scale).clamp(-1, 1).round() * scale).reshape(w.shape)
    return wq.detach() + (w - w.detach())


def quantize_activations(x, bits):
    """Per-token absmax integer activations (rounded in float32), straight-through."""
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    scale = qmax / xf.abs().amax(-1, keepdim=True).clamp(min=1e-5)
    xq = ((xf * scale).round().clamp(-qmax - 1, qmax) / scale).to(x.dtype)
    return xq.detach() + (x - x.detach())


def _f32(t):
    """At least float32 (the context and the walk; float64 stays float64)."""
    return t.to(torch.promote_types(t.dtype, torch.float32))


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Trees(nn.Module):
    """What both kinds of tree share: the centred depthwise conv and the node tables (UltraFastBERT's FFF init)."""

    def __init__(self, cfg, n_trees, depth):
        super().__init__()
        d = cfg.d_model
        self.cfg, self.n_trees, self.depth = cfg, n_trees, depth
        self.n_nodes = 2 ** (depth + 1) - 1  # per tree
        total = n_trees * self.n_nodes
        k_in, k_out = math.sqrt(1.0 / d), math.sqrt(1.0 / (n_trees * (depth + 1)))  # out: 1 / sqrt(nodes on a path)
        self.node_in = nn.Parameter(torch.empty(total, d).uniform_(-k_in, k_in))
        self.node_bias = nn.Parameter(torch.empty(total).uniform_(-k_in, k_in))
        self.node_bias._no_weight_decay = True
        self.node_out = nn.Parameter(torch.empty(total, d).uniform_(-k_out, k_out))
        self.conv1d = nn.Conv1d(d, d, cfg.d_conv, groups=d, padding=cfg.d_conv // 2, bias=True)
        with torch.no_grad():  # the identity: the trees start by routing on the token itself
            self.conv1d.weight.zero_()
            self.conv1d.weight[:, 0, cfg.d_conv // 2] = 1.0
            self.conv1d.bias.zero_()

    def q(self, w):
        return ternary(w, self.cfg.ternary_group)

    def local_mix(self, u, mask):
        """Centred depthwise conv (as shifted sums) over the masked input, then 8-bit activations."""
        u = u * mask.unsqueeze(-1)
        k, pad, length = self.cfg.d_conv, self.cfg.d_conv // 2, u.shape[1]
        up = F.pad(u, (0, 0, pad, k - 1 - pad))
        w = self.conv1d.weight.squeeze(1)  # (d, k)
        x = self.conv1d.bias + sum(up[:, j : j + length] * w[:, j] for j in range(k))
        return quantize_activations(x, self.cfg.act_bits)


class Bank(Trees):
    """Depth-0 trees with bidirectional SSM context, one clock per head."""

    def __init__(self, cfg):
        super().__init__(cfg, cfg.bank_trees, 0)
        n, H = cfg.bank_state, cfg.bank_heads
        self.ctx_proj = nn.Linear(cfg.d_model, 3 * n + H, bias=False)  # [B, C_fwd, C_bwd, dt per head]
        dt = torch.exp(torch.rand(H) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus, as Mamba-2
        self.A_log = nn.Parameter(torch.log(torch.empty(H).uniform_(1, 16)))
        self.dt_bias._no_weight_decay = self.A_log._no_weight_decay = True

    def signals(self, x, mask):
        n = self.cfg.bank_state
        p = F.linear(x, self.q(self.ctx_proj.weight))
        B, C_fwd, C_bwd = p[..., :n], p[..., n : 2 * n], p[..., 2 * n : 3 * n]
        dt = F.softplus(p[..., 3 * n :] + self.dt_bias)  # (b, l, heads)
        a = dt * -torch.exp(self.A_log.float()) * mask.unsqueeze(-1)  # per-token log-decay; padding stops no clock
        return B, C_fwd, C_bwd, dt, a

    def forward(self, u, mask):
        b, length, _ = u.shape
        H, T = self.cfg.bank_heads, self.n_trees
        x = self.local_mix(u, mask)
        logit = F.linear(x, self.q(self.node_in), self.node_bias)  # (b, l, trees)
        B, C_fwd, C_bwd, dt, a = self.signals(x, mask)
        src = dt.repeat_interleave(T // H, -1) * mask.unsqueeze(-1) * logit  # what each token writes to each tree
        # weights[t, s, h] = <C[t], B[s]> * decay between s and t on head h's clock, both directions, zero at s = t
        Bf, a = _f32(B), _f32(a)
        cs = torch.cumsum(a, 1)
        cx = cs - a
        lower = torch.ones(length, length, dtype=torch.bool, device=u.device).tril(-1)[None, :, :, None]
        fwd = torch.exp((cs[:, :, None] - cs[:, None, :]).masked_fill(~lower, -torch.inf))
        bwd = torch.exp((cx[:, None, :] - cx[:, :, None]).masked_fill(~lower.transpose(1, 2), -torch.inf))
        w = (_f32(C_fwd) @ Bf.transpose(1, 2)).unsqueeze(-1) * fwd + (_f32(C_bwd) @ Bf.transpose(1, 2)).unsqueeze(-1) * bwd
        ctx = torch.einsum("btsh,bshp->bthp", w, _f32(src).view(b, length, H, T // H)).reshape(b, length, T)
        coef = F.gelu(logit) + ctx.to(logit.dtype)
        return (coef @ self.q(self.node_out)) * mask.unsqueeze(-1)


def _dgelu(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476)) + x * torch.exp(-0.5 * x * x) * 0.3989422804014327


def _walk(L, depth):
    """L: (tokens, trees, nodes). The node index along each token's path in each tree, (tokens, trees, depth + 1)."""
    cur = torch.zeros(L.shape[:2], dtype=torch.long, device=L.device)
    path = []
    for _ in range(depth + 1):
        path.append(cur)
        cur = 2 * cur + 1 + (L.gather(2, cur.unsqueeze(-1)).squeeze(-1) > 0).long()
    return torch.stack(path, -1)


class RouteSTE(torch.autograd.Function):
    """out = sum over each token's path nodes of gelu(logit) * W[node], with the straight-through branch gradient,
    without forming the path weights (``Deep.reference`` is the definition). Backward, with g = dout @ W^T, the only
    nonzero logit gradients are at path nodes; at path node a on level k:
        dL[a] = gelu'(L[a]) g[a] + sign * (on[a] - alt[a]) * sigmoid'(L[a] / temp) / temp
    on[a]: gelu * g summed over the path below a; alt[a]: the same over the chain from a's other child following the
    token's own decisions; sign +1 if the token went right at a."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, L, W, n_trees, depth, temp):
        N = L.shape[0]
        L3 = L.view(N, n_trees, -1)
        path = _walk(L3, depth)
        A = torch.zeros_like(L3).scatter(2, path, F.gelu(_f32(L3.gather(2, path))).to(L.dtype)).view(N, -1)
        ctx.save_for_backward(L, W)
        ctx.cfg = n_trees, depth, temp
        return A @ W

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dout):
        L, W = ctx.saved_tensors
        n_trees, depth, temp = ctx.cfg
        N = L.shape[0]
        L3 = L.view(N, n_trees, -1)
        path = _walk(L3, depth)
        Lp = _f32(L3.gather(2, path))
        A = torch.zeros_like(L3).scatter(2, path, F.gelu(Lp).to(L.dtype)).view(N, -1)
        dW = A.t() @ dout
        G = (dout @ W.t()).view(N, n_trees, -1)
        Gp = _f32(G.gather(2, path))
        c = F.gelu(Lp) * Gp
        on = c.sum(-1, keepdim=True) - c.cumsum(-1)  # the path strictly below each node
        d = _dgelu(Lp) * Gp
        for k in range(depth):
            right = Lp[..., k] > 0
            s = 2 * path[..., k] + 2 - right.long()  # the other child
            alt = torch.zeros_like(on[..., k])
            for _ in range(depth - k):
                ls = _f32(L3.gather(2, s.unsqueeze(-1)).squeeze(-1))
                alt = alt + F.gelu(ls) * _f32(G.gather(2, s.unsqueeze(-1)).squeeze(-1))
                s = 2 * s + 1 + (ls > 0).long()
            p = torch.sigmoid(Lp[..., k] / temp)
            d[..., k] += torch.where(right, on[..., k] - alt, alt - on[..., k]) * p * (1 - p) / temp
        dL = torch.zeros(L3.shape, dtype=d.dtype, device=L.device).scatter(2, path, d).view(N, -1).to(L.dtype)
        return dL, dW, None, None, None


class Deep(Trees):
    """Stateless deep trees, hard routing with the straight-through branch gradient."""

    def __init__(self, cfg):
        super().__init__(cfg, cfg.deep_trees, cfg.deep_depth)

    def forward(self, u, mask):
        b, length, d = u.shape
        x = self.local_mix(u, mask)
        L = F.linear(x, self.q(self.node_in), self.node_bias).reshape(b * length, -1)  # every node's logit
        out = RouteSTE.apply(L, self.q(self.node_out), self.n_trees, self.depth, self.cfg.route_ste_temp)
        return out.view(b, length, d).to(x.dtype) * mask.unsqueeze(-1)

    def reference(self, u, mask):
        """The definition: every node's coefficient times its path weight, the product of the branch values above
        it (the hard step going forward, sigmoid(logit / temp) going backward)."""
        b, length, _ = u.shape
        x = self.local_mix(u, mask)
        al = F.linear(x, self.q(self.node_in), self.node_bias).view(b, length, self.n_trees, self.n_nodes)
        p = torch.sigmoid(al / self.cfg.route_ste_temp)
        right = (al > 0).to(al.dtype) + (p - p.detach())
        levels = [al.new_ones(b, length, self.n_trees, 1)]
        for k in range(self.depth):
            g = right[..., 2**k - 1 : 2 ** (k + 1) - 1]
            levels.append(torch.stack([levels[-1] * (1 - g), levels[-1] * g], -1).flatten(3))  # children 2i+1, 2i+2
        pi = torch.cat(levels, -1).flatten(2)
        return ((pi * F.gelu(al.flatten(2))) @ self.q(self.node_out)) * mask.unsqueeze(-1)


class Mixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bank, self.deep = Bank(cfg), Deep(cfg)

    def forward(self, u, mask):
        return self.bank(u, mask) + self.deep(u, mask)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = Mixer(cfg)

    def forward(self, x, mask):
        return x + self.mixer(self.norm(x), mask)


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[cfg.pad_token_id].zero_()
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, ids):
        mask = (ids != self.cfg.pad_token_id).float()
        x = self.embedding(ids)
        for layer in self.layers:
            x = layer(x, mask.to(x.dtype))
        return self.norm_f(x)


class GTS(nn.Module):
    """Masked LM: the encoder and a tied output layer with a bias."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = Encoder(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.lm_head.weight = self.backbone.embedding.weight
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, ids, labels=None):
        """Without labels: logits (b, l, vocab). With labels (-100 = unscored): the summed cross-entropy and the logits
        at the labelled positions only, (n, vocab), in row-major order."""
        h = self.backbone(ids)
        if labels is None:
            return self.lm_head(h)
        sel = labels != -100
        logits = self.lm_head(h[sel])
        return F.cross_entropy(logits.float(), labels[sel], reduction="sum"), logits


def load(path, device="cpu"):
    """A nanoGTS ckpt.pt, or the repository's checkpoint.pt (float) or binarized.pt (2-bit ternary codes and scales)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    rc = blob["config"]
    assert rc.get("mixer", "mixed") == "mixed" and not rc.get("causal") and rc.get("loops", 1) == 1, "not a GTS3-style model"
    model = GTS(GTSConfig(**{f.name: rc[f.name] for f in fields(GTSConfig) if f.name in rc}))
    if "model" in blob:
        model.load_state_dict(blob["model"])
        return model.to(device)
    state = dict(blob["float"])  # binarized: rebuild latent weights whose absmean quantisation gives back the codes
    for name, e in blob["ternary"].items():
        u = torch.stack([(e["packed"] >> s) & 3 for s in (0, 2, 4, 6)], 1).flatten()
        codes = (u[: math.prod(e["shape"])].float() - 1).reshape(e["shape"])
        cg = codes.reshape(*codes.shape[:-1], -1, e["group"])
        frac = (cg != 0).float().mean(-1, keepdim=True).clamp(min=1.0 / e["group"])
        state[name] = (cg * e["scales"].unsqueeze(-1) / frac).reshape(codes.shape)
    state["lm_head.weight"] = state["backbone.embedding.weight"]
    model.load_state_dict(state)
    return model.to(device)


# -------------------------------------------------------------------------------------------------------------- data

TOKENIZER = "google-bert/bert-base-uncased"
CLS, SEP, MASK, PAD = 101, 102, 103, 0


def tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(TOKENIZER, "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    return tok


def prep(a):
    """Wikipedia parquet shards -> one uint16 stream of articles, each followed by [SEP]; validation: the first
    --val-tokens of the last shard."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    tok = tokenizer()
    files = sorted(f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset") if f.startswith("20231101.en/"))
    lo, hi = (int(v) for v in a.shards.split("-"))
    assert hi < len(files) - 1, "the last shard is the validation shard"
    os.makedirs(a.out, exist_ok=True)

    def stream(name, f, limit):
        pf, n = pq.ParquetFile(hf_hub_download("wikimedia/wikipedia", name, repo_type="dataset")), 0
        for g in range(pf.num_row_groups):
            texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            for enc in tok.encode_batch(texts, add_special_tokens=False):
                ids = np.asarray(enc.ids + [SEP], dtype=np.uint16)[: limit - n]
                ids.tofile(f)
                n += len(ids)
                if n >= limit:
                    return n
        return n

    with open(os.path.join(a.out, "val.bin"), "wb") as f:
        n_val = stream(files[-1], f, a.val_tokens)
    n_train = 0
    with open(os.path.join(a.out, "train.bin"), "wb") as f:
        for i in range(lo, hi + 1):
            n_train += stream(files[i], f, 1 << 62)
            print(f"  shard {i}: {n_train:,} training tokens", flush=True)
    json.dump({"shards": a.shards, "train_tokens": n_train, "val_tokens": n_val}, open(os.path.join(a.out, "meta.json"), "w"))


def get_batch(data, batch, seq_len, gen, mask_prob=0.15, vocab=30522):
    """Windows of the stream starting with [CLS]; BERT's 80/10/10 masking (random tokens from 999 on: real word pieces)."""
    starts = torch.randint(0, len(data) - seq_len, (batch,), generator=gen).tolist()
    ids = torch.from_numpy(np.stack([data[s : s + seq_len - 1] for s in starts]).astype(np.int64))
    ids = torch.cat([torch.full((batch, 1), CLS), ids], 1)
    chosen = (torch.rand(ids.shape, generator=gen) < mask_prob) & (ids != CLS) & (ids != SEP) & (ids != PAD)
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, generator=gen)
    inputs = ids.clone()
    inputs[chosen & (r < 0.8)] = MASK
    rand = chosen & (r >= 0.8) & (r < 0.9)
    inputs[rand] = torch.randint(999, vocab, (int(rand.sum()),), generator=gen)
    return inputs, labels


# ------------------------------------------------------------------------------------------------------------- train


@torch.no_grad()
def evaluate(model, data, a, device):
    """Masked-LM loss and accuracy on the same 20 validation batches every time."""
    model.eval()
    g = torch.Generator().manual_seed(1234)
    loss, correct, total = 0.0, 0, 0
    for _ in range(a.eval_batches):
        x, y = get_batch(data, a.batch_size, a.seq_len, g)
        for xm, ym in zip(x.split(a.micro_batch), y.split(a.micro_batch)):
            xm, ym = xm.to(device), ym.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                l, logits = model(xm, ym)
            loss += l.item()
            correct += (logits.argmax(-1) == ym[ym != -100]).sum().item()
            total += int((ym != -100).sum())
    model.train()
    return loss / total, correct / total


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    train_data = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    model = GTS(GTSConfig()).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)]
    rest = [p for p in model.parameters() if not (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-6, fused=device == "cuda")
    gen = torch.Generator().manual_seed(a.seed)
    step, lr0, curve = 0, 0.0, []
    if a.resume:  # a new phase: weights, AdamW state, step and sampler carry over; warm from the last rate
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        gen.set_state(ck["generator"])
        step, curve = ck["step"], ck["curve"]
        lr0 = ck["optimizer"]["param_groups"][0]["lr"]
    s0 = step
    print(f"nanoGTS: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, steps {s0} -> {a.steps}", flush=True)

    def lr_at(k):  # linear from lr0 over the warmup, then cosine to 10% of --lr at --steps
        k -= s0
        if k < a.warmup:
            return lr0 + (a.lr - lr0) * (k + 1) / a.warmup
        frac = min(1.0, (k - a.warmup) / max(1, a.steps - s0 - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    def save():
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": asdict(model.cfg),
                    "curve": curve, "generator": gen.get_state()}, os.path.join(a.out, "ckpt.pt"))

    os.makedirs(a.out, exist_ok=True)
    t0, run = time.time(), []
    while step < a.steps:
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = get_batch(train_data, a.batch_size, a.seq_len, gen)
        n = int((y != -100).sum())
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for xm, ym in zip(x.split(a.micro_batch), y.split(a.micro_batch)):  # the mean over the whole batch's labels
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss, _ = model(xm.to(device), ym.to(device))
            (loss / n).backward()
            total += loss.item() / n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run.append(total)
        if step % a.log_every == 0:
            print(f"  step {step:6d}  loss {sum(run[-a.log_every:]) / a.log_every:.4f}  lr {lr_at(step):.2e}  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            vl, acc = evaluate(model, val_data, a, device)
            curve.append({"step": step, "tokens": step * a.batch_size * a.seq_len, "val_loss": vl, "val_masked_acc": acc})
            print(f"step {step:6d}  tokens {step * a.batch_size * a.seq_len / 1e6:8.1f}M  val loss {vl:.4f}  masked acc {acc:.4f}", flush=True)
            save()


EXAMPLES = ["The capital of France is [MASK].", "The [MASK] Ocean is the largest ocean on Earth.",
            "He played the [MASK] in the orchestra for twenty years.", "The film was directed by Steven [MASK]."]


@torch.no_grad()
def sample(a):
    model, tok = load(a.ckpt).eval(), tokenizer()
    for text in EXAMPLES:
        ids = tok.encode(text).ids  # [CLS] ... [SEP]
        top = model(torch.tensor([ids]))[0, ids.index(MASK)].topk(5).indices.tolist()
        print(f"{text}  ->  {', '.join(tok.id_to_token(i) for i in top)}")


# ---------------------------------------------------------------------------------------------------------- selftest


def selftest():
    """The fast paths against their definitions, in float64 on a small model."""
    torch.manual_seed(0)
    cfg = GTSConfig(d_model=64, n_layer=1, vocab_size=100, bank_trees=8, bank_heads=2, bank_state=4, deep_trees=2,
                    deep_depth=4, ternary_group=32)
    m = Mixer(cfg).double()
    for t in (m.bank.node_in, m.deep.node_in):  # spread the logits so both branches get traffic
        t.data *= 8
    u = torch.randn(2, 12, 64, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 12, dtype=torch.float64)
    mask[1, 9:] = 0

    # deep trees: the walk Function against the path-weight definition, values and every gradient
    out, ref = m.deep(u, mask), m.deep.reference(u, mask)
    g = torch.randn_like(out)
    ps = [u, m.deep.node_in, m.deep.node_bias, m.deep.node_out, m.deep.conv1d.weight]
    g1, g2 = torch.autograd.grad(out, ps, g), torch.autograd.grad(ref, ps, g)
    assert torch.allclose(out, ref, atol=1e-10), "deep trees: forward"
    assert all(torch.allclose(x, y, atol=1e-9) for x, y in zip(g1, g2)), "deep trees: straight-through gradient"

    # bank: the quadratic form against token-at-a-time states, read then written, forward and backward
    with torch.no_grad():
        bank = m.bank
        x = bank.local_mix(u, mask)
        logit = F.linear(x, bank.q(bank.node_in), bank.node_bias)
        B, C_fwd, C_bwd, dt, a = bank.signals(x, mask)
        H, T = cfg.bank_heads, cfg.bank_trees
        head = torch.arange(T) * H // T
        ctx = torch.zeros_like(logit)
        for b in range(2):
            for order, C in ((range(12), C_fwd), (range(11, -1, -1), C_bwd)):
                state = torch.zeros(T, cfg.bank_state, dtype=torch.float64)
                for t in order:  # arriving at t decays the state by exp(a[t]) in either direction, then read, then write
                    state = state * torch.exp(a[b, t, head]).unsqueeze(-1)
                    ctx[b, t] += (state @ C[b, t].unsqueeze(-1)).squeeze(-1)
                    state = state + (dt[b, t, head] * mask[b, t] * logit[b, t]).unsqueeze(-1) * B[b, t]
        want = ((F.gelu(logit) + ctx) @ bank.q(bank.node_out)) * mask.unsqueeze(-1)
        assert torch.allclose(bank(u, mask), want, atol=1e-9), "bank: context"
    print("selftest passed: deep-tree straight-through walk == path-weight definition; bank quadratic form == recurrence")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--out", required=True)
    q.add_argument("--shards", required=True, help="training shards, e.g. 0-6")
    q.add_argument("--val-tokens", type=int, default=2_000_000)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, required=True, help="the absolute step to end this phase at")
    t.add_argument("--lr", type=float, required=True)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--resume")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--micro-batch", type=int, default=64, help="sequences per forward pass; gradients accumulate exactly")
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    sub.add_parser("selftest")
    a = p.parse_args()
    {"prep": prep, "train": train, "sample": sample, "selftest": lambda _: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
