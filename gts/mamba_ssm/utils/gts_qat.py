# Golden Tree Snake (GTS) fork, 2026.
"""Ternary quantisation-aware training for GTS models.

The recipe follows "Ternary Mamba" (arXiv:2606.18114): start from a trained full-precision model,
make a ternary copy of it, and train the copy against the frozen original with

    loss = alpha * KL(teacher || student) * T^2 + (1 - alpha) * CE

Quantised: the node tables and ctx_proj of every GTS mixer. Kept in full precision: the decay
parameters (A_log, dt_bias), node biases, the conv, norms, embeddings and the head.

Two things here are specific to GTS, because a GTS token takes a hard path through a tree:

* ``route_distillation`` - wherever the student sits at the same node as the teacher, pull the
  student's branch logit towards the teacher's. Branch decisions otherwise get no direct gradient,
  and ternarising ``node_in`` flips some of them.
* ``path_agreement`` - the share of path slots where student and teacher are at the same node.
"""

import copy

import torch
import torch.nn.functional as F

from mamba_ssm.modules.gts import GTS
from mamba_ssm.modules.ternary import zero_ratio

__all__ = [
    "gts_mixers", "make_ternary_student", "set_quant_lambda", "lambda_schedule", "set_capture_paths",
    "ternary_stats", "path_agreement", "route_distillation", "qat_loss",
]


def gts_mixers(model):
    return [m for m in model.modules() if isinstance(m, GTS)]


def make_ternary_student(teacher, ternary_group=128, act_bits=None):
    """Copy a full-precision model into a ternary student. The teacher is frozen in place."""
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = copy.deepcopy(teacher)
    for p in student.parameters():
        p.requires_grad_(True)
    for m in gts_mixers(student):
        m.ternary, m.ternary_group, m.act_bits = True, ternary_group, act_bits
    config = getattr(student, "config", None)
    if config is not None:
        config.ternary, config.ternary_group, config.act_bits = True, ternary_group, act_bits
    student.train()
    return student


def set_quant_lambda(model, lam):
    for m in gts_mixers(model):
        m.quant_lambda = float(lam)


def lambda_schedule(step, warmup_steps):
    """Linear ramp of the quantisation blend from 0 to 1 over ``warmup_steps``; 1 afterwards."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def set_capture_paths(model, on=True):
    for m in gts_mixers(model):
        m.capture_paths = on
        if not on:
            m.last_nodes = m.last_logits = None


@torch.no_grad()
def ternary_stats(model):
    """Zero-code share of every quantised tensor. Watch it: a climb towards 1 is zero-ratio collapse."""
    ratios = []
    for m in gts_mixers(model):
        if not m.ternary:
            continue
        tensors = [m.node_in, m.node_out] + ([m.ctx_proj.weight] if m.use_context else [])
        ratios += [zero_ratio(t, m.ternary_group) for t in tensors]
    if not ratios:
        return {}
    return {"zero_ratio_mean": sum(ratios) / len(ratios), "zero_ratio_min": min(ratios), "zero_ratio_max": max(ratios)}


def _pairs(student, teacher):
    s, t = gts_mixers(student), gts_mixers(teacher)
    assert len(s) == len(t), "student and teacher must have the same layers"
    return list(zip(s, t))


@torch.no_grad()
def path_agreement(student, teacher, attention_mask=None):
    """Share of path slots where the student is at the teacher's node, from the last captured forward."""
    same, total = 0.0, 0.0
    for s, t in _pairs(student, teacher):
        eq = (s.last_nodes == t.last_nodes).float()
        if attention_mask is not None:
            eq = eq * attention_mask.unsqueeze(-1)
            total += attention_mask.sum().item() * eq.shape[-1]
        else:
            total += eq.numel()
        same += eq.sum().item()
    return same / max(total, 1.0)


def route_distillation(student, teacher, attention_mask=None):
    """Mean squared gap between student and teacher branch logits over the slots where both are at the same node."""
    losses = []
    for s, t in _pairs(student, teacher):
        same = (s.last_nodes == t.last_nodes).to(s.last_logits.dtype)
        if attention_mask is not None:
            same = same * attention_mask.unsqueeze(-1).to(same.dtype)
        gap = (s.last_logits - t.last_logits.detach()).pow(2) * same
        losses.append(gap.sum() / same.sum().clamp(min=1.0))
    return torch.stack(losses).mean()


def qat_loss(student, teacher, input_ids, labels, attention_mask=None, alpha=0.5, temperature=1.0, beta=0.0):
    """One QAT step's loss.

    alpha: weight of the distillation term against cross-entropy (Ternary Mamba uses 0.5).
    beta:  weight of route distillation (0 disables it).
    Both loss terms are taken over the masked positions, i.e. where ``labels != -100``.
    """
    if attention_mask is None:
        attention_mask = input_ids != student.config.pad_token_id
    set_capture_paths(student, True)
    set_capture_paths(teacher, True)
    with torch.no_grad():
        t_logits = teacher(input_ids, attention_mask=attention_mask).logits
    s_logits = student(input_ids, attention_mask=attention_mask).logits

    sel = labels != -100
    s_sel, t_sel, y = s_logits[sel], t_logits[sel], labels[sel]
    ce = F.cross_entropy(s_sel, y)
    kd = F.kl_div(
        F.log_softmax(s_sel / temperature, dim=-1), F.log_softmax(t_sel / temperature, dim=-1),
        log_target=True, reduction="batchmean",
    ) * temperature**2
    loss = alpha * kd + (1 - alpha) * ce
    out = {"ce": ce.detach(), "kd": kd.detach()}
    if beta > 0:
        route = route_distillation(student, teacher, attention_mask)
        loss = loss + beta * route
        out["route"] = route.detach()
    out["path_agreement"] = path_agreement(student, teacher, attention_mask.float())
    out["accuracy"] = (s_sel.argmax(-1) == y).float().mean().detach()
    out["loss"] = loss
    set_capture_paths(student, False)
    set_capture_paths(teacher, False)
    return out
