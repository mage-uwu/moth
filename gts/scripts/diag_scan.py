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
