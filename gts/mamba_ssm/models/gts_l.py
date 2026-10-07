# Golden Tree Snake (GTS) fork, 2026.
"""GTS-L: a ~395M ternary, attention-free masked LM shaped to be distilled from ModernBERT-large with MOHAWK.

Each of the 28 layers mirrors a ModernBERT layer, sub-block for sub-block, so every student piece has a teacher piece
to imitate (MOHAWK, Bick et al. 2024):

    h = h + BiSSD(norm1(h))      <- imitates the attention sub-block   (Stage 1: its mixing matrix; Stage 2: its output)
    h = h + DeepTrees(norm2(h))  <- imitates the GeGLU MLP sub-block   (Stage 2: its output)

* BiSSD is a bidirectional, multi-head Mamba-2 SSD mixer: per head h, y_i = sum_j M_h[i, j] x_j with
  M_h[i, j] = (C_i . B_j) dt_j exp(sum of dt * A between j and i), a forward scan for j <= i plus a reversed scan for
  j > i (Hydra-style quasiseparable mixing). It is linear in length on GPU (two chunked SSD scans) and CPU (two
  recurrences). Heads, head size and state size equal the teacher's (16 x 64, state 64), so C, B and x start from the
  teacher's Q, K and V and each attention head is distilled into one SSD head, as MOHAWK does for Phi-Mamba.
* DeepTrees are GTS stateless trees (4 trees of depth 9, hard routing, ~8.4M parameters a layer like the teacher's
  MLP, ~40 node rows touched per token): the conditional-compute replacement for the dense MLP.
* Embeddings, the norms, the prediction head and the tied decoder are copied from the teacher (weight transfer).
* Projections and tree tables are ternary (absmean, group 128) with 8-bit activations; ``set_quant(lam)`` ramps the
  quantisation in from full precision (lam=0) to ternary (lam=1).
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.gts import GTS
from mamba_ssm.modules.ternary import absmean_ternary, quantize_activations


@dataclass
class GTSLConfig:
    vocab_size: int = 50368
    d_model: int = 1024
    n_layer: int = 28
    n_heads: int = 16
    d_state: int = 64  # = the teacher's head size, so B and C come from K and Q
    deep_trees: int = 4
    deep_depth: int = 9
    ternary_group: int = 128
    act_bits: int = 8
    norm_eps: float = 1e-5
    chunk_size: int = 256
    pad_token_id: int = 50283


class TernaryLinear(nn.Linear):
    """nn.Linear with absmean-ternary weights and per-token 8-bit activations, both blended in by ``lam``."""

    def __init__(self, d_in, d_out, group=128, act_bits=8):
        super().__init__(d_in, d_out, bias=False)
        self.group, self.act_bits, self.lam = group, act_bits, 0.0

    def forward(self, x):
        if self.lam <= 0:
            return F.linear(x, self.weight)
        x = quantize_activations(x, self.act_bits, self.lam) if self.act_bits else x
        return F.linear(x, absmean_ternary(self.weight, self.group, self.lam))


class BiSSD(nn.Module):
    def __init__(self, cfg: GTSLConfig):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.H, self.P, self.N, self.chunk = H, d // H, cfg.d_state, cfg.chunk_size
        assert self.P * H == d
        self.in_proj = TernaryLinear(d, 2 * H * self.N + d, cfg.ternary_group, cfg.act_bits)  # [C | B | x]
        self.dt_proj = nn.Linear(d, H, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.full((H,), 0.2)))
        self.D = nn.Parameter(torch.zeros(H))
        self.out_proj = TernaryLinear(d, d, cfg.ternary_group, cfg.act_bits)
        with torch.no_grad():  # dt ~ 0.05 at init, so the decay per token exp(-0.2 * 0.05) ~ 0.99
            self.dt_proj.weight.mul_(0.01)
            self.dt_proj.bias.fill_(math.log(math.expm1(0.05)))

    def _parts(self, u):
        b, L, _ = u.shape
        z = self.in_proj(u)
        HN = self.H * self.N
        C = z[..., :HN].view(b, L, self.H, self.N)
        B = z[..., HN:2 * HN].view(b, L, self.H, self.N)
        x = z[..., 2 * HN:].view(b, L, self.H, self.P)
        dt = F.softplus(self.dt_proj(u).float())  # (b, L, H)
        A = -torch.exp(self.A_log.float())  # (H,)
        return x, B, C, dt, A

    @staticmethod
    def mixing_matrix(B, C, dt, A):
        """M (b, H, L, L): the forward scan's lower triangle (diagonal included) plus the reversed scan's strict
        upper triangle, in float32."""
        a = (dt * A).transpose(1, 2)  # (b, H, L)
        cs = torch.cumsum(a, -1)
        i = torch.arange(a.shape[-1], device=a.device)
        low = i[:, None] >= i[None, :]
        fwd = cs[..., :, None] - cs[..., None, :]  # sum_{k=j+1..i} a_k for j <= i
        csx = cs - a  # exclusive cumsum
        bwd = csx[..., None, :] - csx[..., :, None]  # sum_{k=i..j-1} a_k for j > i
        logdec = torch.where(low, fwd, bwd)
        logdec = torch.where(low | (i[:, None] < i[None, :]), logdec, torch.zeros_like(logdec))
        cb = torch.einsum("blhn,bmhn->bhlm", C.float(), B.float())
        return cb * torch.exp(torch.clamp(logdec, max=0.0)) * dt.transpose(1, 2)[:, :, None, :]

    def matrix(self, u):
        """Stage 1: the student's mixing matrices for input u, (b, H, L, L)."""
        x, B, C, dt, A = self._parts(u)
        return self.mixing_matrix(B, C, dt, A)

    def forward(self, u):
        b, L, d = u.shape
        x, B, C, dt, A = self._parts(u)
        if u.is_cuda:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

            xx, BB, CC = x.contiguous(), B.contiguous(), C.contiguous()
            yf = mamba_chunk_scan_combined(xx, dt, A, BB, CC, self.chunk)
            yb = mamba_chunk_scan_combined(xx.flip(1), dt.flip(1), A, BB.flip(1), CC.flip(1), self.chunk).flip(1)
            diag = ((C.float() * B.float()).sum(-1) * dt)[..., None] * x.float()  # counted by both scans
            y = yf.float() + yb.float() - diag
        else:  # reference (and CPU) path: the materialised matrix
            M = self.mixing_matrix(B, C, dt, A)
            y = torch.einsum("bhlm,bmhp->blhp", M, x.float())
        y = y + self.D.float()[None, None, :, None] * x.float()
        return self.out_proj(y.to(u.dtype).reshape(b, L, d))


class GTSLBlock(nn.Module):
    def __init__(self, cfg: GTSLConfig, idx):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.Identity() if idx == 0 else nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)  # as ModernBERT
        self.mixer = BiSSD(cfg)
        self.norm2 = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.deep = GTS(d, depth=cfg.deep_depth, n_trees=cfg.deep_trees, use_context=False, d_conv=0, route_ste=True, dense_walk=True,
                        ternary=True, ternary_group=cfg.ternary_group, act_bits=cfg.act_bits, layer_idx=idx)

    def forward(self, h):
        h = h + self.mixer(self.norm1(h))
        return h + self.deep(self.norm2(h))


class GTSLForMaskedLM(nn.Module):
    def __init__(self, cfg: GTSLConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_token_id)
        self.emb_norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.layers = nn.ModuleList([GTSLBlock(cfg, i) for i in range(cfg.n_layer)])
        self.final_norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.head_dense = nn.Linear(d, d, bias=False)
        self.head_norm = nn.LayerNorm(d, eps=cfg.norm_eps, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.set_quant(0.0)

    def set_quant(self, lam):
        """Blend ternary weights and 8-bit activations in: 0 = full precision, 1 = fully quantised."""
        for m in self.modules():
            if isinstance(m, TernaryLinear):
                m.lam = lam
            elif isinstance(m, GTS):
                m.quant_lambda = lam

    def hidden(self, input_ids):
        h = self.emb_norm(self.tok_embeddings(input_ids))
        for layer in self.layers:
            h = layer(h)
        return self.final_norm(h)

    def logits_at(self, h):
        """The prediction head and the tied decoder, on hidden states h (..., d)."""
        z = self.head_norm(F.gelu(self.head_dense(h)))
        return F.linear(z, self.tok_embeddings.weight, self.decoder_bias)

    def forward(self, input_ids, sel=None):
        """Logits at every position, or only where ``sel`` (a bool mask) is set, as (n_selected, vocab)."""
        h = self.hidden(input_ids)
        return self.logits_at(h[sel] if sel is not None else h)

    @torch.no_grad()
    def init_from_modernbert(self, teacher):
        """Weight transfer from a ModernBertForMaskedLM: embeddings, every norm, the head and decoder bias, and each
        attention's Q, K, V, O into the BiSSD's C, B, x, out (C scaled by 1/sqrt(head size), as attention scales
        its logits). The trees start from their own initialisation; Stage 2 fits them to the MLPs."""
        tm = teacher.model
        self.tok_embeddings.weight.copy_(tm.embeddings.tok_embeddings.weight)
        self.emb_norm.weight.copy_(tm.embeddings.norm.weight)
        for i, (s, t) in enumerate(zip(self.layers, tm.layers)):
            if i > 0:
                s.norm1.weight.copy_(t.attn_norm.weight)
            s.norm2.weight.copy_(t.mlp_norm.weight)
            q, k, v = t.attn.Wqkv.weight.chunk(3, 0)
            s.mixer.in_proj.weight.copy_(torch.cat([q * s.mixer.P ** -0.5, k, v], 0))
            s.mixer.out_proj.weight.copy_(t.attn.Wo.weight)
        self.final_norm.weight.copy_(tm.final_norm.weight)
        self.head_dense.weight.copy_(teacher.head.dense.weight)
        self.head_norm.weight.copy_(teacher.head.norm.weight)
        self.decoder_bias.copy_(teacher.decoder.bias)
        return self

    def transferred_parameters(self):
        """Parameters copied from the teacher and kept frozen in the pilot: embeddings, the head, the decoder bias."""
        return [self.tok_embeddings.weight, self.emb_norm.weight, self.head_dense.weight, self.head_norm.weight,
                self.decoder_bias, self.final_norm.weight]


def config_dict(cfg):
    return asdict(cfg)
