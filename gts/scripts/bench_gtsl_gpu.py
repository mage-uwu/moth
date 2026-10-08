# Golden Tree Snake (GTS) fork, 2026.
"""Serving throughput on a GPU: GTS-L (random weights, fully quantised, frozen, sparse deep trees) and ModernBERT-large
(its teacher), bf16, whole 512-token sequences at several batch sizes, encoder only (no masked-LM head). For the
tokens-per-dollar comparison with the CPU runtime (kernel/gtsl_bench.c).

    python scripts/bench_gtsl_gpu.py --out results/gtsl_gpu_bench.json
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mamba_ssm.models.gts_l import GTSLConfig, GTSLForMaskedLM  # noqa: E402


def timed(fn, reps=10):
    fn(); torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--linear-rank", type=int, default=128)
    p.add_argument("--out")
    a = p.parse_args()
    res = {"gpu": torch.cuda.get_device_name(), "seq": a.seq, "gts_l": {}, "modernbert_large": {}}
    m = GTSLForMaskedLM(GTSLConfig(linear_rank=a.linear_rank)).cuda().eval()
    m.set_quant(1.0)
    from mamba_ssm.models.gts_l import TernaryLinear
    from mamba_ssm.modules.gts_sparse import prepare_inference
    from mamba_ssm.modules.ternary import absmean_ternary

    with torch.no_grad():  # projections: ternary weights quantised once, then plain bf16 GEMMs (generous to the GPU)
        for mod in m.modules():
            if isinstance(mod, TernaryLinear):
                mod.weight.copy_(absmean_ternary(mod.weight, mod.group, 1.0))
                mod.lam = 0.0

    prepare_inference(m, dtype=torch.bfloat16)  # frozen ternary weights, sparse deep-tree kernel
    from transformers import AutoModel

    mb = AutoModel.from_pretrained("answerdotai/ModernBERT-large", dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    for b in a.batches:
        ids = torch.randint(1000, 40000, (b, a.seq), device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            dt = timed(lambda: m.hidden(ids))
        res["gts_l"][b] = b * a.seq / dt
        with torch.inference_mode():
            dt2 = timed(lambda: mb(input_ids=ids))
        res["modernbert_large"][b] = b * a.seq / dt2
        print(f"batch {b:4d} x {a.seq}: GTS-L {res['gts_l'][b]:10,.0f} tokens/s   ModernBERT-large {res['modernbert_large'][b]:10,.0f} tokens/s", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
