# Golden Tree Snake (GTS) fork, 2026.
"""Dense against sparse deep trees on a GPU: one deep-tree layer, then the whole 110M encoder (inference), on real text.

    python scripts/bench_sparse.py --ckpt gts3.pt
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.modules.gts_sparse import sparsify  # noqa: E402
from mamba_ssm.ops.gts_sparse import sparse_route_fwd  # noqa: E402
from mamba_ssm.ops.gts_route import route_ste_out  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized  # noqa: E402


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


def text_batch(batch, seq):
    from bert_pretrain import _tokenizer
    from datasets import load_dataset

    tok = _tokenizer()
    ids = [101]
    for ex in load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True):
        ids += tok.encode(ex["text"], add_special_tokens=False).ids + [102]
        if len(ids) > batch * seq:
            break
    return torch.tensor(ids[: batch * seq]).view(batch, seq)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--layer", type=int, default=7)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(blob["config"], loops=1, latent_tokens=0) if "loops" in blob["config"] else blob["config"]
    model = GTSForMaskedLM(GTSConfig(**cfg))
    load_binarized(blob, model, allow_missing=[n for n, _ in model.named_parameters() if "loop" in n or "latent" in n])
    model = model.cuda().eval()
    ids = text_batch(a.batch, a.seq).cuda()
    print(f"{torch.cuda.get_device_name(0)}; {a.batch} x {a.seq} tokens of Wikipedia; GTS {cfg['d_model']} x {cfg['n_layer']}, "
          f"deep trees {cfg['deep_trees']} x depth {cfg['deep_depth']}", flush=True)

    # one deep-tree layer, on that layer's real input
    deep = model.backbone.layers[a.layer].mixer.deep
    caught = {}
    h = deep.register_forward_pre_hook(lambda m, args, kw: caught.__setitem__("u", (args[0], kw.get("attention_mask"))), with_kwargs=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.backbone(ids)
    h.remove()
    u, mask = caught["u"]
    mask = torch.ones(u.shape[:2], device=u.device) if mask is None else mask.float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = deep._local_mix(u, mask).reshape(-1, u.shape[-1]).to(torch.bfloat16)
    total, pad = deep.n_trees * deep.n_nodes, -(deep.n_trees * deep.n_nodes) % 64
    w_in, w_out = torch.nn.functional.pad(deep._w_in(), (0, 0, 0, pad)), torch.nn.functional.pad(deep._w_out(), (0, 0, 0, pad))
    bias = torch.nn.functional.pad(deep.node_bias, (0, pad)) if deep.node_bias is not None else None
    wi16, wo16 = w_in.bfloat16(), w_out.bfloat16()

    def dense():
        L = torch.nn.functional.linear(x, wi16, bias.bfloat16() if bias is not None else None)
        return route_ste_out(L, wo16, deep.n_trees, deep.depth, deep.act, want_nodes=True, n_nodes=deep.n_nodes)

    ref, ref_nodes = dense()
    t_dense = timed(dense)
    print(f"\none deep-tree layer, forward ({x.shape[0]} tokens): dense route path {t_dense:.3f} ms", flush=True)
    best = None
    for bm in (16, 32, 64):
        for bd in (64, 128):
            for nw in (2, 4, 8):
                try:
                    f = lambda: sparse_route_fwd(x, wi16, bias, wo16, deep.n_trees, deep.n_nodes, deep.depth, deep.act,  # noqa: E731
                                                 block_m=bm, block_d=bd, num_warps=nw)
                    out, nodes, _ = f()
                    t = timed(f)
                except Exception as e:  # a configuration the compiler rejects
                    print(f"  BM {bm} BD {bd} warps {nw}: failed ({type(e).__name__})")
                    continue
                agree = (nodes.long() == ref_nodes.long()).all(-1).float().mean().item()
                err = (out - ref.float()).abs().max().item()
                print(f"  sparse BM {bm:2d} BD {bd:3d} warps {nw}: {t:.3f} ms ({t_dense / t:.2f}x)  paths equal {agree:.4f}  max |diff| {err:.3g}", flush=True)
                if best is None or t < best[0]:
                    best = (t, bm, bd, nw)
    print(f"best sparse: {best[0]:.3f} ms = {t_dense / best[0]:.2f}x the dense layer (BM {best[1]}, BD {best[2]}, warps {best[3]})", flush=True)

    # the whole encoder, inference
    def enc():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.backbone(ids)

    ref_h = enc().float()
    t_d = timed(enc, 10)
    sparsify(model)
    out_h = enc().float()
    t_s = timed(enc, 10)
    rel = ((out_h - ref_h).norm() / ref_h.norm()).item()
    n = ids.numel()
    print(f"\nencoder inference, {n} tokens: dense {t_d:.1f} ms ({n / t_d * 1e3:,.0f} tokens/s), sparse {t_s:.1f} ms "
          f"({n / t_s * 1e3:,.0f} tokens/s) = {t_d / t_s:.2f}x; relative difference of the final states {rel:.2e}", flush=True)
    sparsify(model, enable=False)


if __name__ == "__main__":
    main()
