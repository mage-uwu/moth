# Golden Tree Snake (GTS) fork, 2026.
"""Binarized checkpoints of ternary GTS models: every ternary weight as 2-bit codes plus its group scales, everything
else in float32, and a loader that rebuilds a model computing the same function.

File layout (torch.save of a dict):

    {"format": "gts-ternary-v1", "config": {...GTSConfig fields...},
     "ternary": {name: {"packed": uint8 (4 codes per byte, code + 1 in two bits, little end first),
                        "shape": [...], "group": g, "scales": float32 (rows, n / g)}},
     "float": {name: float32 tensor}}       # every other parameter and buffer of the state dict

A ternary weight's value is code * scale, with code in {-1, 0, 1} and one scale per ``group`` consecutive weights of
a row. That is what the model computes with: the latent float weights are only needed to keep training.

Loading rebuilds latent weights whose absmean quantisation gives back exactly these codes and scales: a group with
code fraction f (nonzero codes / group size) gets latent weights code * scale / f, whose absmean is scale and whose
ratios to it round to the codes.
"""

import torch

from mamba_ssm.modules.gts import GTS
from mamba_ssm.modules.ternary import _group_size, pack_ternary

__all__ = ["ternary_tensors", "save_binarized", "load_binarized"]

_TERNARY = ("node_in", "node_out", "ctx_proj.weight", "read_w")


def ternary_tensors(model):
    """{state-dict name: (tensor, group size)} for every weight a ternary GTS mixer quantises."""
    out = {}
    for prefix, m in model.named_modules():
        if isinstance(m, GTS) and m.ternary:
            for name, p in m.named_parameters():
                if name in _TERNARY:
                    out[f"{prefix}.{name}" if prefix else name] = (p, _group_size(p.shape[-1], m.ternary_group))
    return out


def _pack2(codes):
    u = (codes.flatten().to(torch.int16) + 1).to(torch.uint8)
    u = torch.cat([u, u.new_zeros(-len(u) % 4)])
    return u[0::4] | (u[1::4] << 2) | (u[2::4] << 4) | (u[3::4] << 6)


def _unpack2(packed, shape):
    u = torch.stack([(packed >> s) & 3 for s in (0, 2, 4, 6)], dim=1).flatten()
    n = 1
    for d in shape:
        n *= d
    return (u[:n].to(torch.int8) - 1).reshape(shape)


@torch.no_grad()
def save_binarized(model, config, path):
    tern = ternary_tensors(model)
    blob = {"format": "gts-ternary-v1", "config": dict(config), "ternary": {}, "float": {}}
    for name, (w, g) in tern.items():
        codes, scales = pack_ternary(w.float().cpu(), g)
        blob["ternary"][name] = {"packed": _pack2(codes), "shape": list(w.shape), "group": g, "scales": scales.float()}
    seen = set()
    for name, t in model.state_dict().items():
        if name in tern or t.data_ptr() in seen:  # tied weights (embedding and head) are stored once
            continue
        seen.add(t.data_ptr())
        blob["float"][name] = t.detach().float().cpu() if t.is_floating_point() else t.detach().cpu()
    torch.save(blob, path)
    return blob


@torch.no_grad()
def load_binarized(path_or_blob, model, allow_missing=()):
    """Fill ``model`` (built from the saved config) from a binarized checkpoint. Returns the model. ``allow_missing``:
    parameter names the checkpoint may lack, which keep their initial values (a GTS-Uni built from one-pass weights)."""
    blob = torch.load(path_or_blob, map_location="cpu") if isinstance(path_or_blob, str) else path_or_blob
    missing, _ = model.load_state_dict(blob["float"], strict=False)
    params = dict(model.named_parameters())
    for name, e in blob["ternary"].items():
        codes = _unpack2(e["packed"], e["shape"]).float()
        g = e["group"]
        cg = codes.reshape(*codes.shape[:-1], -1, g)
        frac = (cg != 0).float().mean(-1, keepdim=True).clamp(min=1.0 / g)
        latent = (cg * e["scales"].unsqueeze(-1) / frac).reshape(codes.shape)
        params[name].copy_(latent)
    left = [n for n in missing if n not in blob["ternary"]]
    tied = {n for n in left if n.endswith("lm_head.weight")} | set(allow_missing)
    assert not (set(left) - tied), f"not in the checkpoint: {sorted(set(left) - tied)}"
    return model
