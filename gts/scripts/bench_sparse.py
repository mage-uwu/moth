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
    for bm in (8, 16, 32):
        for bd in (128, 256):
            for nw in (2, 4):
                for xres in (False, True):
                    kw = dict(block_m=bm, block_d=bd, num_warps=nw, x_resident=xres)
                    try:
                        f = lambda kw=kw: sparse_route_fwd(x, wi16, bias, wo16, deep.n_trees, deep.n_nodes, deep.depth, deep.act, **kw)  # noqa: E731
                        out, nodes, _ = f()
                        t = timed(f)
                    except Exception as e:  # a configuration the compiler rejects
                        print(f"  {kw}: failed ({type(e).__name__}: {str(e)[:80]})")
                        continue
                    agree = (nodes.long() == ref_nodes.long()).all(-1).float().mean().item()
                    print(f"  sparse BM {bm:2d} BD {bd:3d} warps {nw} x-resident {int(xres)}: {t:.3f} ms ({t_dense / t:.2f}x)  "
                          f"paths equal {agree:.4f}", flush=True)
                    if best is None or t < best[0]:
                        best = (t, kw)
    t, kw = best
    print(f"best sparse: {t:.3f} ms = {t_dense / t:.2f}x the dense layer ({kw})", flush=True)
    _, nodes, logits = sparse_route_fwd(x, wi16, bias, wo16, deep.n_trees, deep.n_nodes, deep.depth, deep.act, **kw)
    t1 = timed(lambda: sparse_route_fwd(x, wi16, bias, wo16, deep.n_trees, deep.n_nodes, deep.depth, deep.act, phases=1, **kw))
    t2 = timed(lambda: sparse_route_fwd(x, wi16, bias, wo16, deep.n_trees, deep.n_nodes, deep.depth, deep.act, phases=2,
                                        buffers=(nodes, logits), **kw))
    print(f"  of which the walk (phase 1) {t1:.3f} ms, the output gather (phase 2) {t2:.3f} ms", flush=True)
    L16 = torch.nn.functional.linear(x, wi16, bias.bfloat16() if bias is not None else None)
    tg = timed(lambda: torch.nn.functional.linear(x, wi16, bias.bfloat16() if bias is not None else None))
    tw = timed(lambda: route_ste_out(L16, wo16, deep.n_trees, deep.depth, deep.act, n_nodes=deep.n_nodes))
    print(f"  dense path for comparison: logits GEMM {tg:.3f} ms, walk + output GEMM {tw:.3f} ms", flush=True)
    import mamba_ssm.modules.gts_sparse as gs
    gs.KERNEL_ARGS = kw

    # where dense inference spends its time
    def enc_d():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.backbone(ids)

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        enc_d()
        torch.cuda.synchronize()
    print("\ndense encoder inference, GPU time by operation (top 15):")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15, max_name_column_width=60))

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
    # same masked-LM loss? (hard routing turns last-bit differences into different paths, so states are compared by loss)
    g = torch.Generator(device="cpu").manual_seed(0)
    sel = (torch.rand(ids.shape, generator=g) < 0.15).cuda()
    inp, lab = ids.masked_fill(sel, 103), ids.masked_fill(~sel, -100)

    def mlm():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(inp, labels=lab, labelled_only=True)
        return o.loss.item(), (o.logits.argmax(-1) == lab[lab != -100]).float().mean().item()

    sparsify(model, enable=False)
    ld = mlm()
    sparsify(model)
    ls = mlm()
    print(f"masked-LM loss / accuracy on this batch: dense {ld[0]:.4f} / {ld[1]:.4f}, sparse {ls[0]:.4f} / {ls[1]:.4f}")
    print(f"\nencoder inference, {n} tokens: dense {t_d:.1f} ms ({n / t_d * 1e3:,.0f} tokens/s), sparse {t_s:.1f} ms "
          f"({n / t_s * 1e3:,.0f} tokens/s) = {t_d / t_s:.2f}x; relative difference of the final states {rel:.2e}", flush=True)
    sparsify(model, enable=False)


if __name__ == "__main__":
    main()
