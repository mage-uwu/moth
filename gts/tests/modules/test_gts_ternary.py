# Golden Tree Snake (GTS) fork, 2026.
"""CPU tests for the ternary training path. Run with: pytest tests/modules/test_gts_ternary.py"""

import torch

from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM
from mamba_ssm.modules.gts import GTS
from mamba_ssm.modules.ternary import absmean_ternary, pack_ternary, quantize_activations, unpack_ternary, zero_ratio
from mamba_ssm.utils.gts_qat import (
    gts_mixers, lambda_schedule, make_ternary_student, qat_loss, set_quant_lambda, ternary_stats,
)


def tiny(**kwargs):
    torch.manual_seed(0)
    cfg = dict(d_model=32, n_layer=2, vocab_size=50, depth=4, d_state=8, ternary_group=16)
    cfg.update(kwargs)
    return GTSForMaskedLM(GTSConfig(**cfg))


def batch(vocab=50):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(2, vocab, (6, 24), generator=g)
    labels = torch.full_like(ids, -100)
    pick = torch.rand(ids.shape, generator=g) < 0.2
    labels[pick] = ids[pick]
    return ids.masked_fill(pick, 1), labels


def test_each_group_is_exactly_ternary():
    w = torch.randn(5, 64)
    q = absmean_ternary(w, group_size=16)
    for row in q.reshape(5, 4, 16).reshape(-1, 16):
        values = row.unique()
        assert values.numel() <= 3
        scale = values.abs().max()
        assert all(v.item() in (-scale.item(), 0.0, scale.item()) for v in values)
    # the scale is the group's mean absolute latent weight
    assert torch.allclose(q.reshape(5, 4, 16).abs().amax(-1), w.reshape(5, 4, 16).abs().mean(-1))


def test_group_size_falls_back_to_a_divisor():
    w = torch.randn(3, 30)
    assert all(row.unique().numel() <= 3 for row in absmean_ternary(w, group_size=128))  # one group per row
    assert absmean_ternary(w, group_size=8).shape == w.shape  # 8 does not divide 30; uses 6


def test_straight_through_gradient():
    w = torch.randn(4, 32, requires_grad=True)
    (absmean_ternary(w, 16) * torch.arange(32.0)).sum().backward()
    assert torch.equal(w.grad, torch.arange(32.0).expand(4, 32))  # identity Jacobian; the scale gets no gradient


def test_lambda_blends_from_full_precision_to_ternary():
    w = torch.randn(4, 32)
    assert torch.equal(absmean_ternary(w, 16, lam=0.0), w)
    half = absmean_ternary(w, 16, lam=0.5)
    assert torch.allclose(half, 0.5 * (w + absmean_ternary(w, 16)))
    assert lambda_schedule(0, 0) == 1.0 and lambda_schedule(4, 10) == 0.5 and lambda_schedule(99, 10) == 1.0


def test_pack_and_unpack_reproduce_the_training_weights():
    w = torch.randn(7, 48)
    codes, scales = pack_ternary(w, 16)
    assert codes.dtype == torch.int8 and set(codes.unique().tolist()) <= {-1, 0, 1} and scales.shape == (7, 3)
    assert torch.equal(unpack_ternary(codes, scales), absmean_ternary(w, 16))
    assert abs(zero_ratio(w, 16) - (codes == 0).float().mean().item()) < 1e-9


def test_activation_quantisation_levels():
    x = torch.randn(3, 5, 32)
    q = quantize_activations(x, bits=8)
    steps = q / (x.abs().amax(-1, keepdim=True) / 127)
    assert torch.allclose(steps, steps.round(), atol=1e-4) and steps.abs().max() <= 128
    assert torch.equal(quantize_activations(x, bits=None), x)


def test_scales_are_not_parameters():
    """Zero-ratio collapse comes from learnable scales, so there must be none."""
    fp, ternary = tiny(), tiny(ternary=True)
    assert [n for n, _ in fp.named_parameters()] == [n for n, _ in ternary.named_parameters()]


def test_student_starts_as_the_teacher_and_is_then_ternary():
    teacher = tiny()
    student = make_ternary_student(teacher, ternary_group=16)
    assert all(not p.requires_grad for p in teacher.parameters()) and all(p.requires_grad for p in student.parameters())
    assert student.config.ternary and not teacher.config.ternary
    ids, _ = batch()
    teacher_logits = teacher(ids).logits
    set_quant_lambda(student, 0.0)
    student.eval()
    assert torch.allclose(student(ids).logits, teacher_logits, atol=1e-6)  # lambda 0: identical function
    set_quant_lambda(student, 1.0)
    assert not torch.allclose(student(ids).logits, teacher_logits, atol=1e-3)
    stats = ternary_stats(student)
    assert 0.15 < stats["zero_ratio_mean"] < 0.45 and ternary_stats(teacher) == {}


def test_qat_loss_trains_routing_and_reports_agreement():
    teacher = tiny()
    student = make_ternary_student(teacher, ternary_group=16)
    ids, labels = batch()
    out = qat_loss(student, teacher, ids, labels, alpha=0.5, beta=1.0)
    assert {"loss", "ce", "kd", "route", "path_agreement", "accuracy"} <= set(out)
    assert 0.0 < out["path_agreement"] <= 1.0 and out["kd"] >= 0
    out["loss"].backward()
    for name, p in student.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), name
    mixers = gts_mixers(student)
    assert all(m.node_in.grad.abs().max() > 0 and m.ctx_proj.weight.grad.abs().max() > 0 for m in mixers)


def test_qat_recovers_most_of_what_ternarising_lost():
    """A short QAT run on a toy task brings the student's distillation loss well below the untrained ternary copy."""
    torch.manual_seed(0)
    teacher = tiny()
    ids, labels = batch()
    opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    for _ in range(80):
        loss = teacher(ids, labels=labels).loss
        opt.zero_grad(); loss.backward(); opt.step()
    student = make_ternary_student(teacher, ternary_group=16)
    before = qat_loss(student, teacher, ids, labels)["kd"].item()
    opt = torch.optim.AdamW(student.parameters(), lr=2e-3)
    for _ in range(80):
        out = qat_loss(student, teacher, ids, labels, alpha=0.5, beta=1.0)
        opt.zero_grad(); out["loss"].backward(); opt.step()
    after = qat_loss(student, teacher, ids, labels)["kd"].item()
    stats = ternary_stats(student)
    assert after < 0.5 * before, (before, after)
    assert stats["zero_ratio_max"] < 0.6, stats
    # and the trained student is still exactly ternary per group
    m = gts_mixers(student)[0]
    codes, scales = pack_ternary(m.node_in, m.ternary_group)
    assert torch.equal(unpack_ternary(codes, scales), m._w_in().detach())
