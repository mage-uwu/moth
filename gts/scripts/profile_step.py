# Golden Tree Snake (GTS) fork, 2026.
"""Where one training step of an lm_run.py model spends its GPU time, by PyTorch operation (torch.profiler).

    python scripts/profile_step.py --arch mixed [--no-scan-kernel] [--no-route-kernel] [--seq-len 512 --batch-size 8]

Builds the model at lm_run.py's defaults on random tokens, runs three warmup steps, profiles two, and prints the step
time and the operations with the most GPU time (self time, so nothing is counted twice).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["mixed", "mamba2"], default="mixed")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-scan-kernel", action="store_true")
    p.add_argument("--no-route-kernel", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile each block")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    d = dict(width=1024, layers=27, bank_trees=32, bank_heads=8, bank_state=16, deep_trees=4, deep_depth=10, m2_layers=68, m2_state=128, m2_headdim=64)
    args = argparse.Namespace(arch=a.arch, **d)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    model = lm_run.build(args, 50257).cuda()
    for m in model.modules():
        if a.no_scan_kernel and hasattr(m, "scan_kernel"):
            m.scan_kernel = False
        if a.no_route_kernel and hasattr(m, "route_kernel"):
            m.route_kernel = False
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=a.fused_adam)
    if a.compile:
        for i, layer in enumerate(model.layers):
            model.layers[i] = torch.compile(layer)
    x = torch.randint(0, 50257, (a.batch_size, a.seq_len), device="cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            loss = lm_run.forward(model, x, x, not a.no_checkpoint)[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / 3
    print(f"{a.arch}, {a.batch_size} x {a.seq_len} tokens, scan {not a.no_scan_kernel}, route kernel {not a.no_route_kernel}, "
          f"checkpoint {not a.no_checkpoint}, amp {a.amp}, fused adam {a.fused_adam}, compile {a.compile}: {dt * 1e3:.0f} ms per step = {a.batch_size * a.seq_len / dt:,.0f} tokens/s, "
          f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(2):
            step()
        torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total = sum(e.self_device_time_total for e in ev) / 2
    print(f"  GPU busy {total / 1e3:.0f} ms per step ({total / 1e3 / (dt * 1e3) * 100:.0f}% of the step)")
    for e in sorted(ev, key=lambda e: -e.self_device_time_total)[: a.top]:
        print(f"    {e.self_device_time_total / 2 / 1e3:8.1f} ms  {e.self_device_time_total / 2 / total * 100:5.1f}%  x{e.count // 2:5d}  {e.key[:80]}")


if __name__ == "__main__":
    main()
