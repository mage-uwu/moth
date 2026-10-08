# Golden Tree Snake (GTS) fork, 2026.
"""Export a GTS-L masked LM (mamba_ssm/models/gts_l.py) for the CPU runtime kernel/gtsl_bench.c, fully quantised
(ternary weights, 8-bit activations), with a test sequence and PyTorch's logits at its masked positions.

    python scripts/export_gtsl.py --ckpt out/eva/run/final.pt --out gtsl.bin
    python scripts/export_gtsl.py --random --config '{"d_model": 256, "n_heads": 4, "n_layer": 2}' --out tiny.bin

File (little-endian int32 / float32): format 11; V, D, layers, heads H, state N, head size P, trees, depth, linear
rank R; LayerNorm eps. Embeddings (V x D), embedding norm, final norm, head dense (D x D), head norm, decoder bias
(V). Per layer: has_norm1 (layer 0 has none), norm1, norm2; BiSSD in_proj ((2HN + D) x D, rows [C | B | x]),
dt_proj weight (H x D) and bias, A_log, D; out_proj (D x D); the trees' node_in, node_bias, node_out (nodes x D,
tree-major, heap order); with R > 0, lin_down (R x D), lin_up (D x R), lin_bias. Ternary tensors are written as their
effective float values (scale * code), which the kernel packs back to 2 bits. Then the test: T, M, token ids, masked
positions, PyTorch's logits there (M x V).
"""
import argparse
import json
import os
import struct
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mamba_ssm.models.gts_l import GTSLConfig, GTSLForMaskedLM  # noqa: E402
from mamba_ssm.modules.ternary import absmean_ternary  # noqa: E402

MASK = 50284


def load(a):
    if a.random:
        cfg = GTSLConfig(**json.loads(a.config or "{}"))
        torch.manual_seed(a.seed)
        m = GTSLForMaskedLM(cfg)
        with torch.no_grad():  # a random model whose tree routing and scans are not degenerate
            for blk in m.layers:
                blk.mixer.dt_proj.bias.fill_(-1.0)
            if cfg.linear_rank:
                for blk in m.layers:
                    blk.lin_up.weight.normal_(0, cfg.linear_rank ** -0.5)
        return m
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = GTSLConfig(**(blob.get("config") or blob["log"]["config"]))
    m = GTSLForMaskedLM(cfg)
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in blob["model"].items()})
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt")
    src.add_argument("--random", action="store_true", help="a random model (kernel tests)")
    p.add_argument("--config", help="GTSLConfig fields as JSON, with --random")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--masked", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = load(a).eval()
    m.set_quant(1.0)
    cfg = m.cfg
    g = torch.Generator().manual_seed(a.seed)
    ids = torch.randint(1000, 40000, (1, a.tokens), generator=g)
    ids[0, 0] = 50281  # [CLS]
    pos = torch.randperm(a.tokens - 1, generator=g)[: a.masked] + 1
    ids[0, pos] = MASK
    with torch.no_grad():
        ref = m(ids)[0, pos].float()

    f = open(a.out, "wb")
    ints = lambda *v: f.write(struct.pack(f"<{len(v)}i", *v))  # noqa: E731
    flt = lambda t: f.write(t.detach().float().contiguous().numpy().tobytes())  # noqa: E731
    tern = lambda w, group: flt(absmean_ternary(w.detach().float(), group, 1.0))  # noqa: E731
    H, N, P, d = cfg.n_heads, cfg.d_state, cfg.d_model // cfg.n_heads, cfg.d_model
    ints(11, cfg.vocab_size, d, cfg.n_layer, H, N, P, cfg.deep_trees, cfg.deep_depth, cfg.linear_rank)
    f.write(struct.pack("<f", cfg.norm_eps))
    flt(m.tok_embeddings.weight); flt(m.emb_norm.weight); flt(m.final_norm.weight)
    flt(m.head_dense.weight); flt(m.head_norm.weight); flt(m.decoder_bias)
    for blk in m.layers:
        has1 = isinstance(blk.norm1, torch.nn.LayerNorm)
        ints(int(has1))
        if has1:
            flt(blk.norm1.weight)
        flt(blk.norm2.weight)
        mx = blk.mixer
        tern(mx.in_proj.weight, mx.in_proj.group)
        flt(mx.dt_proj.weight); flt(mx.dt_proj.bias); flt(mx.A_log); flt(mx.D)
        tern(mx.out_proj.weight, mx.out_proj.group)
        dp = blk.deep
        flt(dp._q(dp.node_in)); flt(dp.node_bias); flt(dp._q(dp.node_out))
        if cfg.linear_rank:
            tern(blk.lin_down.weight, blk.lin_down.group); tern(blk.lin_up.weight, blk.lin_up.group); flt(blk.lin_bias)
    ints(a.tokens, len(pos))
    ints(*ids[0].tolist()); ints(*pos.tolist())
    flt(ref)
    f.close()
    print(f"wrote {a.out}: {cfg}; {a.tokens} tokens, {len(pos)} masked ({os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
