# Golden Tree Snake (GTS) fork, 2026.
"""Find where vision training goes non-finite on a GPU: one synthetic step through each part, in several settings,
with hooks reporting the first module whose output or input gradient is not finite.

    python vision/debug_nan.py --lm /root/lm/binarized.pt
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vision"))
from mamba_ssm.models.gts_vision import GTSVision, GTSVisionConfig  # noqa: E402
from vl_pretrain import VLHeads, caption_batch, load_lm, text_embeddings  # noqa: E402


def watch(model, tag, found):
    """Forward and backward hooks on every leaf-ish module; record the first non-finite output and gradient."""
    hooks = []
    for name, m in model.named_modules():
        if not name:
            continue

        def fwd(mod, inp, out, name=name):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and t.is_floating_point() and not torch.isfinite(t).all() and "fwd" not in found:
                found["fwd"] = f"{tag}.{name} ({type(mod).__name__}) output"

        def bwd(mod, gin, gout, name=name):
            for g in gin:
                if torch.is_tensor(g) and not torch.isfinite(g).all() and "bwd" not in found:
                    found["bwd"] = f"{tag}.{name} ({type(mod).__name__}) input gradient"

        hooks.append(m.register_forward_hook(fwd))
        hooks.append(m.register_full_backward_hook(bwd))
    return hooks


def finite(t):
    return bool(torch.isfinite(t).all())


def run(a, amp, length_pad):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = amp and dev == "cuda"
    torch.manual_seed(0)
    lm, cfg = load_lm(a.lm, dev)
    vis = GTSVision(GTSVisionConfig()).to(dev)
    heads = VLHeads(512, cfg["d_model"]).to(dev)
    found = {}
    hooks = watch(vis, "vision", found) + watch(lm, "lm", found)
    B, C = 16, (128 if not length_pad else 126)
    caps = torch.randint(1000, cfg["vocab_size"], (B, C)).numpy().astype("uint16")
    caps[:, 60:] = 0  # padded captions, as most are
    temb = text_embeddings(lm, caps, dev, batch=16)
    print(f"  text embeddings finite: {finite(temb.float())}, max |x| {temb.float().abs().max().item():.1f}")
    x = torch.randn(B, 3, 224, 224, device=dev)
    inputs, labels = caption_batch(caps, 49, 0.4, torch.Generator().manual_seed(0), cfg["vocab_size"])
    inputs, labels = inputs.to(dev), labels.to(dev)
    from mamba_ssm.models.gts_vision import lm_with_image

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        tokens = vis(x)
        print(f"  vision tokens finite: {finite(tokens.float())}, max |x| {tokens.float().abs().max().item():.1f}")
        vecs = heads.sidecar(tokens)
        hidden = lm_with_image(lm, vecs, inputs)
        print(f"  LM hidden finite: {finite(hidden.float())}")
        sel = labels != -100
        logits = lm._head(hidden[sel])
    cap = F.cross_entropy(logits.float(), labels[sel])
    img = F.normalize(heads.img(tokens.mean(1)).float(), dim=-1)
    txt = F.normalize(heads.txt(temb.float()).float(), dim=-1)
    sim = heads.logit_scale.exp() * img @ txt.t()
    t = torch.arange(B, device=dev)
    clip = 0.5 * (F.cross_entropy(sim, t) + F.cross_entropy(sim.t(), t))
    print(f"  losses: clip {clip.item():.4f}  cap {cap.item():.4f}")
    (clip + cap).backward()
    bad = [n for n, p in list(vis.named_parameters()) + list(heads.named_parameters()) if p.grad is not None and not finite(p.grad)]
    print(f"  parameters with non-finite gradients: {len(bad)} {bad[:5]}")
    for k in ("fwd", "bwd"):
        print(f"  first non-finite {k}: {found.get(k, 'none')}")
    for h in hooks:
        h.remove()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lm", required=True)
    a = p.parse_args()
    for amp in (True, False):
        print(f"== bf16 autocast {amp}", flush=True)
        try:
            run(a, amp, False)
        except Exception as e:  # report and go on to the next setting
            print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
