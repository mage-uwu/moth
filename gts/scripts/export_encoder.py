# Golden Tree Snake (GTS) fork, 2026.
"""Export a bidirectional mixed-forest masked LM (GTSForMaskedLM, mixer "mixed"; a GTS-Uni too) for kernel/enc_bench.c.

    python scripts/export_encoder.py checkpoints/.../checkpoint.pt runs/enc.bin   # float checkpoint.pt or binarized.pt

Writes the weights (ternary tensors as their effective scale * code values; the kernel packs them) plus a test: a
sequence with masked positions and PyTorch's float32 logits at them, so the kernel can show it computes the same
function. Layout: 12 ints (format 7, vocab, width, layers, bank trees, bank heads, bank state, deep trees, deep depth,
conv taps, activation bits, unused; format 8, a GTS-Uni, adds passes and latent tokens), then float32 tensors in the
order of ``_tensors`` (a GTS-Uni's pass embeddings, gates and latents after the layers), then the test.
"""
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized  # noqa: E402


def load(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["config"]
    model = GTSForMaskedLM(GTSConfig(**cfg))
    if "model" in blob:
        model.load_state_dict(blob["model"])
    else:
        load_binarized(blob, model)
    return model.eval(), cfg


def _tensors(model):
    b = model.backbone
    out = [b.embedding.weight, model.lm_head.bias, b.norm_f.weight]
    for layer in b.layers:
        bank, deep = layer.mixer.bank, layer.mixer.deep
        out += [layer.norm.weight]
        out += [bank._w_in(), bank.node_bias, bank._w_out(), bank.conv1d.weight.squeeze(1), bank.conv1d.bias,
                bank._w_ctx(), bank.dt_bias, bank.A_log]
        out += [deep._w_in(), deep.node_bias, deep._w_out(), deep.conv1d.weight.squeeze(1), deep.conv1d.bias]
    if model.config.loops > 1:  # GTS-Uni
        out += [b.loop_embed, b.loop_gate] + ([b.latents] if model.config.latent_tokens else [])
    return out


@torch.no_grad()
def main():
    src, dst = sys.argv[1], sys.argv[2]
    length = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    model, cfg = load(src)
    assert cfg["mixer"] == "mixed" and not cfg["causal"] and cfg["act_bits"] == 8
    g = torch.Generator().manual_seed(0)
    # a real-ish test: Wikipedia-like text if a tokenised stream is at hand, else random word pieces
    ids = torch.randint(1000, cfg["vocab_size"], (length,), generator=g)
    ids[0] = 101
    masked = torch.nonzero(torch.rand(length, generator=g) < 0.15).flatten()
    masked = masked[masked > 0]
    ids[masked] = 103  # [MASK]
    logits = model(ids.unsqueeze(0)).logits[0, masked]  # float32 PyTorch, the dense training path
    bank = model.backbone.layers[0].mixer.bank
    deep = model.backbone.layers[0].mixer.deep
    head = [7, cfg["vocab_size"], cfg["d_model"], cfg["n_layer"], bank.n_trees, bank.n_heads, bank.d_state,
            deep.n_trees, deep.depth, bank.d_conv, cfg["act_bits"], 0]
    if cfg.get("loops", 1) > 1:  # format 8: GTS-Uni, two more ints (passes, latent tokens)
        head = [8] + head[1:] + [cfg["loops"], cfg.get("latent_tokens", 0)]
    with open(dst, "wb") as f:
        f.write(struct.pack(f"{len(head)}i", *head))
        for t in _tensors(model):
            f.write(t.detach().float().contiguous().numpy().tobytes())
        f.write(struct.pack("ii", length, len(masked)))
        f.write(ids.numpy().astype(np.int32).tobytes())
        f.write(masked.numpy().astype(np.int32).tobytes())
        f.write(logits.float().numpy().tobytes())
    print(f"wrote {dst}: {os.path.getsize(dst) / 2**20:.0f} MB; test of {length} tokens with {len(masked)} masked")


if __name__ == "__main__":
    main()
