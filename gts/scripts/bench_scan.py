# Golden Tree Snake (GTS) fork, 2026.
"""GPU timing of the depth-0 context (the mixed forest's bank): the quadratic PyTorch path GTS trains with, the
chunked Triton scan in mamba_ssm/ops/gts_scan.py, and upstream Mamba-2's mamba_chunk_scan_combined on the same shape.

    python scripts/bench_scan.py                      # bank shape at width 1024: 32 trees, 8 heads, state 16
    python scripts/bench_scan.py --lengths 512 2048 8192 --layer

Each line is forward + backward of one call, median of --reps after warmup, and the peak memory it allocated.
Upstream's kernel includes the token's own term (s = t), GTS excludes it; the extra diagonal is O(length) and is
left in, so its timing is a slight underestimate of what using it for GTS would cost.
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mamba_ssm.modules.gts import GTS  # noqa: E402
from mamba_ssm.ops.gts_scan import gts_scan, gts_scan_reference  # noqa: E402


def bench(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):  # back to back, as inside a training step: launch overhead overlaps GPU work
        fn()
    torch.cuda.synchronize()
    return times[len(times) // 2] * 1e3, (time.perf_counter() - t) / reps * 1e3, (torch.cuda.max_memory_allocated() - base) / 2**20


def gpu_profile(fn, title, top=12):
    """GPU kernel time of one call, from torch.profiler: the total and the largest kernels."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in prof.key_averages() if e.device_time_total > 0]
    total = sum(e.device_time_total for e in ev) / 5
    print(f"  profile {title}: GPU kernel time {total / 1e3:.3f} ms per call, {sum(e.count for e in ev) // 5} kernels")
    for e in sorted(ev, key=lambda e: -e.device_time_total)[:top]:
        print(f"      {e.device_time_total / 5 / 1e3:8.3f} ms  x{e.count // 5:3d}  {e.key[:90]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--trees", type=int, default=32)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--state", type=int, default=16)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 2048, 8192, 32768])
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--layer", action="store_true", help="also time a whole bank GTS layer with and without the scan")
    p.add_argument("--profile", action="store_true", help="GPU kernel times from torch.profiler")
    a = p.parse_args()
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # as lm_run.py trains
    print(torch.cuda.get_device_name(0), "torch", torch.__version__)
    try:
        from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    except Exception as e:  # pragma: no cover
        print("upstream mamba_chunk_scan_combined unavailable:", repr(e))
        mamba_chunk_scan_combined = None
    b, H, P, N = a.batch, a.heads, a.trees // a.heads, a.state
    print(f"shape: batch {b}, heads {H}, channels per head {P}, state {N}")
    for L in a.lengths:
        torch.manual_seed(0)
        C = torch.randn(b, L, N, device=dev, requires_grad=True)
        B = torch.randn(b, L, N, device=dev, requires_grad=True)
        X = torch.randn(b, L, H, P, device=dev, requires_grad=True)
        dt = (torch.rand(b, L, H, device=dev) * 0.1).requires_grad_()
        A = -torch.rand(H, device=dev) * 4 - 1
        g = torch.randn(b, L, H, P, device=dev)

        def run(f):
            def go():
                out = f()
                torch.autograd.grad(out, (C, B, X, dt), g)
            return go

        rows = []
        if L <= 2048:
            rows.append(("PyTorch quadratic (what GTS trains with)", run(lambda: gts_scan_reference(C, B, X, torch.cumsum(dt * A, 1)))))
        rows.append(("gts_scan sequential chunk  64 tf32", run(lambda: gts_scan(C, B, X, torch.cumsum(dt * A, 1), False, 64, "tf32", parallel=False))))
        for chunk in (32, 64, 128):
            for prec in ("ieee", "tf32"):
                rows.append((f"gts_scan parallel chunk {chunk:3d} {prec}", run(lambda c=chunk, q=prec: gts_scan(C, B, X, torch.cumsum(dt * A, 1), False, c, q))))
        if mamba_chunk_scan_combined is not None:
            for chunk in (64, 128, 256):
                rows.append((f"upstream mamba_chunk_scan_combined {chunk}", run(lambda c=chunk: mamba_chunk_scan_combined(X, dt, A, B.unsqueeze(2), C.unsqueeze(2), c))))
        if a.profile:
            gpu_profile(run(lambda: gts_scan(C, B, X, torch.cumsum(dt * A, 1), False, 64, "tf32")), f"gts_scan parallel 64 tf32, L {L}", 8)
            gpu_profile(run(lambda: gts_scan(C, B, X, torch.cumsum(dt * A, 1), False, 64, "tf32", parallel=False)), f"gts_scan sequential 64 tf32, L {L}", 8)
            if mamba_chunk_scan_combined is not None:
                gpu_profile(run(lambda: mamba_chunk_scan_combined(X, dt, A, B.unsqueeze(2), C.unsqueeze(2), 128)), f"upstream 128, L {L}", 4)
        ref = gts_scan_reference(C, B, X, torch.cumsum(dt * A, 1)) if L <= 2048 else None
        if ref is not None:
            err = (gts_scan(C, B, X, torch.cumsum(dt * A, 1)) - ref).abs().max().item() / ref.abs().max().item()
            print(f"length {L}: gts_scan vs PyTorch, max relative error {err:.1e}")
        for name, fn in rows:
            try:
                ms, tput, mb = bench(fn, a.reps)
                print(f"  L {L:5d}  {name:44s} {ms:8.3f} ms synced  {tput:8.3f} ms back to back  {mb:8.1f} MB")
            except Exception as e:  # report and go on (e.g. out of memory)
                print(f"  L {L:5d}  {name:44s} failed: {type(e).__name__}: {str(e)[:120]}")
                torch.cuda.empty_cache()

        if a.layer:
            for causal in (True,):
                for scan in (False, True):
                    if not scan and L > 2048:
                        continue
                    m = GTS(a.width, depth=0, n_trees=a.trees, n_heads=H, d_state=N, act="split", d_conv=3, causal=causal,
                            ternary=True, act_bits=8, scan_kernel=scan).to(dev)
                    u = torch.randn(b, L, a.width, device=dev, requires_grad=True)
                    gy = torch.randn(b, L, a.width, device=dev)

                    def layer():
                        torch.autograd.grad(m(u), [u] + list(m.parameters()), gy)
                    ms, tput, mb = bench(layer, a.reps)
                    print(f"  L {L:5d}  bank layer, width {a.width}, {'scan     ' if scan else 'quadratic'}          {ms:8.3f} ms synced  {tput:8.3f} ms back to back  {mb:8.1f} MB")
                    if a.profile and L == a.lengths[0]:
                        gpu_profile(layer, f"bank layer {'scan' if scan else 'quadratic'} L {L}")


if __name__ == "__main__":
    main()
