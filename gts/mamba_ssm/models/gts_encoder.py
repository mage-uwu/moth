# Golden Tree Snake (GTS) fork, 2026.
"""A bidirectional encoder built from GTS blocks, with a masked-language-model head.

There is no attention and no positional embedding. Order comes from the per-level
decays, from the separate forward and backward queries, and from the local conv.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.gts import GTS, GTSMixed


@dataclass
class GTSConfig:
    d_model: int = 768
    n_layer: int = 12
    vocab_size: int = 30522
    depth: int = 11
    n_trees: int = 1
    d_state: int = 16
    d_conv: int = 3
    causal: bool = False
    selective: bool = True
    use_context: bool = True
    ternary: bool = False
    ternary_group: int = 128
    act_bits: Optional[int] = None
    pad_token_id: int = 0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    # Later mixer options (see GTS.md). mixer="mixed" builds a GTSMixed from the bank_* and deep_* fields
    # and ignores depth, n_trees and d_state.
    mixer: str = "single"
    n_heads: int = 1
    act: str = "gelu"
    route_ste: bool = False
    bank_trees: int = 32
    bank_heads: int = 8
    bank_state: int = 16
    deep_trees: int = 4
    deep_depth: int = 6


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GTSBlock(nn.Module):
    """Pre-norm residual block: x + GTS(norm(x))."""

    def __init__(self, config: GTSConfig, layer_idx: int):
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        if config.mixer == "mixed":
            self.mixer = GTSMixed(
                config.d_model, bank_trees=config.bank_trees, bank_heads=config.bank_heads, bank_state=config.bank_state,
                deep_trees=config.deep_trees, deep_depth=config.deep_depth, d_conv=config.d_conv, causal=config.causal,
                ternary=config.ternary, ternary_group=config.ternary_group, act_bits=config.act_bits,
                route_ste=config.route_ste, layer_idx=layer_idx,
            )
        else:
            self.mixer = GTS(
                config.d_model, depth=config.depth, n_trees=config.n_trees, n_heads=config.n_heads, d_state=config.d_state,
                d_conv=config.d_conv, causal=config.causal, selective=config.selective, use_context=config.use_context,
                act=config.act, route_ste=config.route_ste, ternary=config.ternary, ternary_group=config.ternary_group,
                act_bits=config.act_bits, layer_idx=layer_idx,
            )

    def forward(self, x, attention_mask=None):
        return x + self.mixer(self.norm(x), attention_mask=attention_mask)


class GTSEncoder(nn.Module):
    def __init__(self, config: GTSConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[config.pad_token_id].zero_()
        self.layers = nn.ModuleList([GTSBlock(config, i) for i in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids != self.config.pad_token_id
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return self.norm_f(x)


@dataclass
class MaskedLMOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class GTSForMaskedLM(nn.Module):
    """Masked-language-model head. Decision models score a handful of candidate tokens at a masked position,
    which uses this same head restricted to those candidates."""

    def __init__(self, config: GTSConfig):
        super().__init__()
        self.config = config
        self.backbone = GTSEncoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)
        if config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, labelled_only=False):
        """``labelled_only`` (with labels): score only the positions that carry a label (about 15% under masked-LM
        training) and return their logits, (n_labelled, vocab), in row-major order of the batch. Same loss as the
        full head, for a fraction of its work."""
        hidden = self.backbone(input_ids, attention_mask=attention_mask)
        if labelled_only and labels is not None:
            sel = labels != -100
            hidden, labels = hidden[sel], labels[sel]
        logits = self._head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
        return MaskedLMOutput(logits=logits, loss=loss)

    def _head(self, hidden):
        """lm_head, with the vocabulary padded to a multiple of 64 rows on a GPU (30,522 rows make the GEMM
        misaligned) and the logits sliced back."""
        w, b = self.lm_head.weight, self.lm_head.bias
        vocab, pad = w.shape[0], -w.shape[0] % 64
        if pad and hidden.is_cuda:
            return F.linear(hidden, F.pad(w, (0, 0, 0, pad)), F.pad(b, (0, pad)))[..., :vocab]
        return F.linear(hidden, w, b)
