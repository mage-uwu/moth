# Golden Tree Snake (GTS) fork, 2026.
"""Error of gts_scan against its float64 reference, per output and gradient, for every direction and clock variant.

    python scripts/diag_scan.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mamba_ssm.ops.gts_scan import gts_scan, gts_scan_reference  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
b, l, h, p, n = 2, 70, 4, 2, 16
base = dict(C=torch.randn(b, l, n), B=torch.randn(b, l, n), X=torch.randn(b, l, h, p), a=-torch.rand(b, l, h) * 0.3)
g = torch.randn(b, l, h, p)
for chunk in (16, 64):
    for reverse in (False, True):
        for excl in (False, True):
            outs = []
            for dtype, fn in ((torch.float64, gts_scan_reference), (torch.float32, gts_scan_reference), (torch.float32, gts_scan)):
                t = {k: v.to(dev, dtype).requires_grad_() for k, v in base.items()}
                kw = dict(chunk=chunk, precision="ieee") if fn is gts_scan else {}
                y = fn(t["C"], t["B"], t["X"], t["a"], reverse, excl, **kw)
                grads = torch.autograd.grad((y * g.to(dev, dtype)).sum(), [t[k] for k in ("C", "B", "X", "a")])
                outs.append([y.double()] + [q.double() for q in grads])
            line = []
            for name, ref, f32, sc in zip(("Y", "dC", "dB", "dX", "da"), *outs):
                scale = ref.abs().max().item()
                line.append(f"{name} ref32 {(f32 - ref).abs().max().item() / scale:.1e} scan {(sc - ref).abs().max().item() / scale:.1e}")
            print(f"chunk {chunk:2d} reverse {reverse!s:5} excl {excl!s:5} | " + " | ".join(line))

# The tensors of the bidirectional depth-0 GTS in tests/modules/test_gts_scan.py, through each direction separately,
# with float64 as the reference. GTS's log-decays reach about -1.6 per token, five times the range above.
from mamba_ssm.modules.gts import GTS  # noqa: E402

torch.manual_seed(0)
m = GTS(32, depth=0, n_trees=8, n_heads=4, d_state=16, act="split", d_conv=3, causal=False).to(dev)
u = torch.randn(2, 70, 32, device=dev)
with torch.no_grad():
    x = m._local_mix(u, torch.ones(2, 70, device=dev))
    Bk, Cf, Cb, dt, a = m._ctx_signals(x, torch.ones(2, 70, device=dev))
    nodes, logits = m._walk(x)
    src = (dt[..., m.slot_group] * logits).reshape(2, 70, 4, 2)
print(f"GTS tensors: a in [{a.min().item():.3f}, {a.max().item():.3f}], |X| max {src.abs().max().item():.2f}, |B| max {Bk.abs().max().item():.2f}")
g = torch.randn(2, 70, 4, 2, device=dev)
for reverse, Cq in ((False, Cf), (True, Cb)):
    outs = []
    for dtype, fn in ((torch.float64, gts_scan_reference), (torch.float32, gts_scan_reference), (torch.float32, gts_scan)):
        t = [v.to(dtype).detach().clone().requires_grad_() for v in (Cq, Bk, src, a)]
        kw = dict(precision="ieee") if fn is gts_scan else {}
        y = fn(*t, reverse, False, **kw)
        outs.append([y.double()] + [q.double() for q in torch.autograd.grad((y * g.to(dtype)).sum(), t)])
    print(f"GTS reverse {reverse!s:5} | " + " | ".join(
        f"{nm} ref32 {(f - r).abs().max().item() / r.abs().max().item():.1e} scan {(s - r).abs().max().item() / r.abs().max().item():.1e}"
        for nm, r, f, s in zip(("Y", "dC", "dB", "dX", "da"), *outs)))

# How the log-decay gradient's error grows with length, at GTS-like magnitudes (float64 reference, one sequence).
for L in (512, 2048, 8192):
    if dev == "cpu" and L > 512:
        break
    torch.manual_seed(1)
    base = [torch.randn(1, L, 16) * 0.7, torch.randn(1, L, 16) * 0.7, torch.randn(1, L, 4, 2) * 0.1, -torch.rand(1, L, 4) * 1.4]
    g = torch.randn(1, L, 4, 2)
    for reverse in (False, True):
        res = []
        for dtype, fn in ((torch.float64, gts_scan_reference), (torch.float32, gts_scan_reference), (torch.float32, gts_scan)):
            t = [v.to(dev, dtype).requires_grad_() for v in base]
            kw = dict(precision="ieee") if fn is gts_scan else {}
            y = fn(*t, reverse, False, **kw)
            res.append(torch.autograd.grad((y * g.to(dev, dtype)).sum(), t)[3].double())
            del y, t
        ref = res[0]
        print(f"length {L:5d} reverse {reverse!s:5} | da relative error: PyTorch float32 {(res[1] - ref).abs().max().item() / ref.abs().max().item():.1e}, "
              f"scan {(res[2] - ref).abs().max().item() / ref.abs().max().item():.1e}")
        torch.cuda.empty_cache() if dev == "cuda" else None
