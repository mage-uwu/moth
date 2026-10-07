# Golden Tree Snake (GTS) fork, 2026.
"""Where a compiled training step of the 110M masked LM spends its time (as bert_pretrain.py trains it: bf16 autocast,
torch.compile per block, fused AdamW, labelled-only head, gradient clipping), plus ablations that price the deep trees.

    python scripts/profile_train_step.py --ckpt gts3.pt
"""
import argparse
import os
import sys
import types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.modules.gts import GTSMixed  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized  # noqa: E402


def build(blob, variant):
    cfg = dict(blob["config"], loops=1, latent_tokens=0) if "loops" in blob["config"] else dict(blob["config"])
    m = GTSForMaskedLM(GTSConfig(**cfg))
    load_binarized(blob, m, allow_missing=[n for n, _ in m.named_parameters() if "loop" in n or "latent" in n])
    for mod in m.modules():
        if isinstance(mod, GTSMixed):
            if variant == "no deep trees":
                mod.forward = types.MethodType(lambda self, u, attention_mask=None: self.bank(u, attention_mask=attention_mask), mod)
            elif variant == "deep trees, no side-branch (route_ste) gradient" or variant.startswith("sparse"):
                mod.deep.route_ste = False
    if variant.startswith("sparse"):
        import mamba_ssm.ops.gts_sparse as ops
        from mamba_ssm.modules.gts_sparse import sparsify

        ops.SPLIT_KERNELS = "fused" not in variant
        sparsify(m)
    m = m.cuda().train()
    for i in range(len(m.backbone.layers)):
        m.backbone.layers[i] = torch.compile(m.backbone.layers[i], dynamic=False)
    return m


def run(m, batch, seq, steps, prof=False):
    torch.manual_seed(0)
    x = torch.randint(1000, 30000, (batch, seq), device="cuda")
    sel = torch.rand(batch, seq, device="cuda") < 0.15
    y = x.masked_fill(~sel, -100)
    x = x.masked_fill(sel, 103)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, fused=True)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, labels=y, labelled_only=True).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        return loss, gn

    loss, gn = step()
    print(f"  first step: loss {loss.item():.5f}  gradient norm {gn.item():.5f}", flush=True)
    for _ in range(4):
        step()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(steps):
        step()
    b.record()
    torch.cuda.synchronize()
    ms = a.elapsed_time(b) / steps
    if prof:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as p:
            step()
            torch.cuda.synchronize()
        tot = sum(e.self_device_time_total for e in p.key_averages()) / 1e3
        print(f"  GPU kernel time {tot:.1f} ms in the profiled step; top kernels:")
        for e in sorted(p.key_averages(), key=lambda e: -e.self_device_time_total)[:22]:
            print(f"    {e.self_device_time_total / 1e3:7.2f} ms {100 * e.self_device_time_total / 1e3 / tot:5.1f}%  x{e.count:4d}  {e.key[:110]}")
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    n = a.batch * a.seq
    print(f"{torch.cuda.get_device_name(0)}: training step of the 110M masked LM, {a.batch} x {a.seq} tokens", flush=True)
    base = None
    for variant in ("as trained", "deep trees, no side-branch (route_ste) gradient", "sparse, path-only gradient, fused kernels",
                    "sparse, path-only gradient, split kernels"):
        m = build(blob, variant)
        ms = run(m, a.batch, a.seq, a.steps, prof=variant.startswith("sparse"))
        base = base or ms
        print(f"{variant:52s} {ms:7.1f} ms/step  {n / ms * 1e3:>9,.0f} tokens/s  ({ms / base:.2f} of the step as trained)", flush=True)
        del m
        torch.cuda.empty_cache()
        torch._dynamo.reset()
    wgrad_sweep(blob)


def wgrad_sweep(blob):
    """The weight-gradient routine alone, on one real layer's paths (GTS3 weights, the batch's input to that layer)."""
    import mamba_ssm.ops.gts_sparse as ops

    m = build(blob, "as trained")
    deep = m.backbone.layers[7]._orig_mod.mixer.deep if hasattr(m.backbone.layers[7], "_orig_mod") else m.backbone.layers[7].mixer.deep
    x = torch.randn(64 * 512, deep.node_in.shape[1], device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        _, nodes, logits = ops.sparse_route_fwd(x, deep._w_in().bfloat16(), deep.node_bias, deep._w_out().bfloat16(),
                                                deep.n_trees, deep.n_nodes, deep.depth)
    vals = torch.nn.functional.gelu(logits)
    rows = deep.n_trees * deep.n_nodes

    def t(fn, reps=10):
        fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps

    print("\nweight-gradient routine alone (one layer, 32,768 tokens): dense levels, run-kernel tiles -> ms", flush=True)
    for top in (7, 8, 9):
        for be, bd, nw in ((32, 128, 4), (64, 128, 4), (16, 256, 4), (32, 256, 8), (64, 64, 2), (128, 128, 8)):
            ms = t(lambda: ops._wgrad_levels(x, nodes, vals, rows, deep.n_nodes, top, True, be, bd, nw))
            print(f"  top {top}  BE {be:3d} BD {bd:3d} warps {nw}: {ms:.3f} ms", flush=True)
    a_ = torch.randn(64 * 512, rows, device="cuda", dtype=torch.bfloat16)
    print(f"  (dense equivalent, one (tokens x nodes)^T GEMM: {t(lambda: a_.t() @ x):.3f} ms)", flush=True)


if __name__ == "__main__":
    main()
