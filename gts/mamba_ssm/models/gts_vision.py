# Golden Tree Snake (GTS) fork, 2026.
"""A GTS vision backbone, and the sidecar that feeds it into a GTS masked LM.

GTSVision: a small float conv stem (stride 16) turns an image into a grid of tokens, plus a learned position
embedding; then bidirectional ternary mixed-forest GTS blocks, the same blocks as GTSForMaskedLM. A GTS block mixes
along one sequence (a bidirectional scan and a 3-tap conv), so the grid is read in raster order by even layers and in
column order by odd ones: two layers reach every token of the grid in both directions. ``features`` gives the
mean-pooled final tokens (what a linear probe sees); ``forward`` gives the token grid.

VisionSidecar: the glue to a frozen GTS masked LM. The token grid is average-pooled to ``n_tokens`` and an MLP maps
each to the LM's embedding width; those vectors are placed after [CLS] where word embeddings would go, and the LM
reads them like words. Trained by masked caption modelling: the LM, frozen, predicts masked caption words from the
image tokens and the unmasked words.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.gts_encoder import GTSBlock, GTSConfig, RMSNorm


@dataclass
class GTSVisionConfig:
    image_size: int = 224
    patch: int = 16  # the stem's total stride
    d_model: int = 512
    n_layer: int = 17
    bank_trees: int = 32
    bank_heads: int = 8
    bank_state: int = 16
    deep_trees: int = 4
    deep_depth: int = 8
    d_conv: int = 3
    ternary: bool = True
    ternary_group: int = 128
    act_bits: int = 8
    route_ste: bool = True
    norm_eps: float = 1e-5

    def block_config(self):
        return GTSConfig(d_model=self.d_model, n_layer=self.n_layer, vocab_size=1, mixer="mixed", bank_trees=self.bank_trees,
                         bank_heads=self.bank_heads, bank_state=self.bank_state, deep_trees=self.deep_trees,
                         deep_depth=self.deep_depth, d_conv=self.d_conv, causal=False, ternary=self.ternary,
                         ternary_group=self.ternary_group, act_bits=self.act_bits, route_ste=self.route_ste,
                         norm_eps=self.norm_eps)


class GTSVision(nn.Module):
    def __init__(self, config: GTSVisionConfig):
        super().__init__()
        self.config = c = config
        assert c.patch == 16 and c.image_size % 16 == 0
        mid = c.d_model // 4
        # 3 -> mid/2 (stride 2) -> mid (stride 2) -> d_model (stride 4): float, as BitNet keeps its embeddings
        self.stem = nn.Sequential(
            nn.Conv2d(3, mid // 2, 3, 2, 1), nn.BatchNorm2d(mid // 2), nn.GELU(),
            nn.Conv2d(mid // 2, mid, 3, 2, 1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, c.d_model, 4, 4),
        )
        self.side = c.image_size // c.patch
        self.pos = nn.Parameter(torch.zeros(1, self.side * self.side, c.d_model))
        nn.init.normal_(self.pos, std=0.02)
        bc = c.block_config()
        self.layers = nn.ModuleList([GTSBlock(bc, i) for i in range(c.n_layer)])
        self.norm_f = RMSNorm(c.d_model, eps=c.norm_eps)
        n = self.side
        order = torch.arange(n * n).view(n, n).t().reshape(-1)  # column order, as indices into the raster order
        self.register_buffer("_col", order, persistent=False)
        self.register_buffer("_col_inv", torch.argsort(order), persistent=False)
        # Optional per-block callables (e.g. compiled ones from compiled_blocks) used in place of the blocks at the
        # training resolution; each takes and returns raster order. Not part of the state dict.
        self.block_fns = None

    def compiled_blocks(self, **compile_kw):
        """One compiled callable per block. An odd block's switch to column order and back is a transpose of the
        token grid inside its compiled function, so Inductor fuses it into the block's first and last pointwise
        kernels instead of two gathers of the whole grid per block."""
        n = self.side

        def column(layer):
            def fn(x):
                B, _, D = x.shape
                y = layer(x.view(B, n, n, D).transpose(1, 2).reshape(B, n * n, D))
                return y.view(B, n, n, D).transpose(1, 2).reshape(B, n * n, D)
            return fn

        return [torch.compile(layer if i % 2 == 0 else column(layer), **compile_kw) for i, layer in enumerate(self.layers)]

    def forward(self, images):
        """images (B, 3, H, W), normalised -> tokens (B, side * side, d_model) in raster order."""
        x = self.stem(images).flatten(2).transpose(1, 2)
        if x.shape[1] != self.pos.shape[1]:  # another resolution: interpolate the position grid
            side = int(math.isqrt(x.shape[1]))
            pos = F.interpolate(self.pos.view(1, self.side, self.side, -1).permute(0, 3, 1, 2), size=(side, side),
                                mode="bicubic", align_corners=False).permute(0, 2, 3, 1).reshape(1, side * side, -1)
            col = torch.arange(side * side, device=x.device).view(side, side).t().reshape(-1)
            col_inv = torch.argsort(col)
        else:
            pos, col, col_inv = self.pos, self._col, self._col_inv
        x = x + pos
        if self.block_fns is not None and x.shape[1] == self.side * self.side:
            for fn in self.block_fns:
                x = fn(x)
            return self.norm_f(x)
        for i, layer in enumerate(self.layers):
            if i % 2:
                x = layer(x[:, col])[:, col_inv]
            else:
                x = layer(x)
        return self.norm_f(x)

    def features(self, images):
        return self.forward(images).mean(1)


class VisionSidecar(nn.Module):
    """Image tokens -> ``n_tokens`` vectors in a GTS masked LM's embedding space."""

    def __init__(self, d_vision, d_lm, n_tokens=49):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Sequential(nn.LayerNorm(d_vision), nn.Linear(d_vision, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))

    def forward(self, tokens):
        B, N, D = tokens.shape
        side, out = int(math.isqrt(N)), int(math.isqrt(self.n_tokens))
        grid = tokens.transpose(1, 2).reshape(B, D, side, side)
        pooled = F.adaptive_avg_pool2d(grid, out).flatten(2).transpose(1, 2)
        return self.proj(pooled)


def lm_with_image(lm, image_embeds, input_ids, attention_mask=None):
    """Run a GTSForMaskedLM's encoder on [CLS] + image vectors + the rest of ``input_ids``: positions 1 to n of
    ``input_ids`` are placeholders whose embeddings are replaced by ``image_embeds`` (B, n, d). Returns the hidden
    states (B, L, d)."""
    bb = lm.backbone
    x = bb.embedding(input_ids)
    n = image_embeds.shape[1]
    x = torch.cat([x[:, :1], image_embeds.to(x.dtype), x[:, 1 + n :]], 1)
    if attention_mask is None:
        attention_mask = input_ids != bb.config.pad_token_id
        attention_mask[:, 1 : 1 + n] = True
    for layer in bb.layers:
        x = layer(x, attention_mask=attention_mask)
    return bb.norm_f(x)


def config_dict(c):
    return asdict(c)
