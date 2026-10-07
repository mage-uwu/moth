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
    import mamba_ssm.modules.gts_sparse as gs

    best = None
    for top in (0, 3, 4, 5, 6, 7):
        for bm, nw in ((8, 2), (16, 4), (8, 4)):
            kw = dict(block_m=bm, block_d=128, num_warps=nw, top=top)
            f = lambda kw=kw: sparse_route_fwd(x, wi16, bias, wo16, deep.n_trees, deep.n_nodes, deep.depth, deep.act, **kw)  # noqa: E731
            try:
                out, nodes, _ = f()
                t = timed(f)
            except Exception as e:
                print(f"  {kw}: failed ({type(e).__name__}: {str(e)[:80]})")
                continue
            agree = (nodes.long() == ref_nodes.long()).all(-1).float().mean().item()
            err = ((out - ref.float()).norm() / ref.float().norm()).item()
            print(f"  sparse top {top} BM {bm:2d} warps {nw}: {t:.3f} ms ({t_dense / t:.2f}x)  paths equal {agree:.4f}  rel. diff {err:.2e}", flush=True)
            if best is None or t < best[0]:
                best = (t, kw)
    t, kw = best
    print(f"best sparse: {t:.3f} ms = {t_dense / t:.2f}x the dense layer ({kw})", flush=True)
    gs.KERNEL_ARGS = kw

    # the whole encoder, inference, five ways
    from mamba_ssm.modules.gts_sparse import prepare_inference

    def build(sparse, freeze, comp):
        m = GTSForMaskedLM(GTSConfig(**cfg))
        load_binarized(blob, m, allow_missing=[n for n, _ in m.named_parameters() if "loop" in n or "latent" in n])
        m = m.cuda().eval()
        if freeze:
            prepare_inference(m, torch.bfloat16, sparse=sparse)
        elif sparse:
            sparsify(m)
        if comp:
            for i in range(len(m.backbone.layers)):
                m.backbone.layers[i] = torch.compile(m.backbone.layers[i], dynamic=False)
        return m

    g = torch.Generator(device="cpu").manual_seed(0)
    sel = (torch.rand(ids.shape, generator=g) < 0.15).cuda()
    inp, lab = ids.masked_fill(sel, 103), ids.masked_fill(~sel, -100)
    n = ids.numel()
    print(f"\nencoder inference, {n} tokens (bf16 autocast; masked-LM loss / accuracy on the same masked batch):")
    base = None
    for name, sp, fr, cp in [("dense, as trained", False, False, False), ("dense, weights quantised once", False, True, False),
                             ("dense, quantised once, compiled", False, True, True), ("sparse, quantised once", True, True, False),
                             ("sparse, quantised once, compiled", True, True, True)]:
        try:
            m = build(sp, fr, cp)

            def enc(m=m):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return m.backbone(ids)

            t = timed(enc, 10)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = m(inp, labels=lab, labelled_only=True)
            acc = (o.logits.argmax(-1) == lab[lab != -100]).float().mean().item()
            base = base or t
            print(f"  {name:40s} {t:7.1f} ms  {n / t * 1e3:>10,.0f} tokens/s  {base / t:5.2f}x   loss {o.loss.item():.4f}  acc {acc:.4f}", flush=True)
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  {name}: failed ({type(e).__name__}: {str(e)[:120]})", flush=True)


if __name__ == "__main__":
    main()
