# Golden Tree Snake (GTS) fork, 2026.
"""GTS-L (mamba_ssm/models/gts_l.py): the BiSSD mixing matrix against an explicit bidirectional recurrence, the GPU
scan path against the matrix, and the weight transfer from a (tiny) ModernBERT."""
import pytest
import torch

from mamba_ssm.models.gts_l import BiSSD, GTSLConfig, GTSLForMaskedLM


def test_matrix_equals_bidirectional_recurrence():
    torch.manual_seed(0)
    c = GTSLConfig(d_model=32, n_heads=2, d_state=8, n_layer=1)
    mix = BiSSD(c)
    x, B, C, dt, A = mix._parts(torch.randn(2, 11, 32))
    M = BiSSD.mixing_matrix(B, C, dt, A)
    b, L, H, P = x.shape
    y = torch.zeros(b, L, H, P)
    for rng in (range(L), range(L - 1, -1, -1)):
        h = torch.zeros(b, H, c.d_state, P)
        for t in rng:
            h = torch.exp(dt[:, t] * A)[..., None, None] * h + dt[:, t, :, None, None] * B[:, t, :, :, None] * x[:, t, :, None, :]
            y[:, t] += torch.einsum("bhn,bhnp->bhp", C[:, t], h)
    y -= ((C * B).sum(-1) * dt)[..., None] * x  # the diagonal is in both scans
    torch.testing.assert_close(torch.einsum("bhlm,bmhp->blhp", M, x), y, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the chunked SSD scan needs CUDA")
def test_gpu_scan_equals_matrix():
    torch.manual_seed(0)
    c = GTSLConfig(d_model=128, n_heads=4, d_state=32, n_layer=1, chunk_size=64)
    mix = BiSSD(c).cuda()
    u = torch.randn(2, 300, 128, device="cuda")
    with torch.no_grad():
        y = mix(u)
        x, B, C, dt, A = mix._parts(u)
        M = BiSSD.mixing_matrix(B, C, dt, A)
        ref = torch.einsum("bhlm,bmhp->blhp", M, x.float()) + mix.D[None, None, :, None] * x.float()
        ref = mix.out_proj(ref.reshape(2, 300, 128))
    torch.testing.assert_close(y, ref, rtol=2e-3, atol=2e-3)


def _padded(device, L=40):
    ids = torch.randint(4, 64, (2, L), device=device)
    mask = torch.ones(2, L, device=device)
    mask[0, 29:] = 0  # padding at the end of one row and in the middle of the other, which the reversed scan
    mask[1, 10:17] = 0  # would otherwise carry into earlier tokens
    return ids, mask


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mixer_mask_hides_padding(device):
    """BiSSD with ``mask`` on a padded batch: every real token's output equals the unpadded sequence's (CPU: the
    materialised matrix; CUDA: the two chunked scans, whose TF32 products hold to ~1e-3)."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    torch.manual_seed(0)
    c = GTSLConfig(d_model=64, n_heads=2, d_state=8, n_layer=1, chunk_size=16)
    mix = BiSSD(c).to(device)
    tol = 1e-4 if device == "cpu" else 2e-3
    with torch.no_grad():
        mix.dt_proj.bias.fill_(0.5)  # dt larger than at init, so a leak through the padding would show
        _, mask = _padded(device)
        u = torch.randn(2, 40, 64, device=device)
        y = mix(u, mask)
        for r in range(2):
            keep = mask[r].bool()
            torch.testing.assert_close(y[r, keep], mix(u[r : r + 1, keep])[0], rtol=tol, atol=tol)
        assert (mix(u)[0, :29] - y[0, :29]).abs().max() > 10 * tol  # without the mask, padding leaks


def test_padding_mask_hides_padding():
    """The whole model with ``mask`` (CPU, exact float32; on a GPU the trees' hard routing can flip on TF32 noise):
    every real token's hidden state equals the unpadded sequence's."""
    torch.manual_seed(0)
    c = GTSLConfig(vocab_size=64, d_model=64, n_heads=2, d_state=8, n_layer=2, deep_trees=2, deep_depth=3, ternary_group=32,
                   pad_token_id=3, chunk_size=16)
    m = GTSLForMaskedLM(c).eval()
    with torch.no_grad():
        for blk in m.layers:
            blk.mixer.dt_proj.bias.fill_(0.5)
        ids, mask = _padded("cpu")
        padded = torch.where(mask.bool(), ids, torch.full_like(ids, 3))
        h = m.hidden(padded, mask=mask)
        for r in range(2):
            keep = mask[r].bool()
            torch.testing.assert_close(h[r, keep], m.hidden(ids[r : r + 1, keep])[0], rtol=1e-4, atol=1e-4)
        assert not torch.allclose(m.hidden(padded)[0, :29], h[0, :29], atol=1e-3)


def test_init_from_modernbert():
    from transformers import ModernBertConfig, ModernBertForMaskedLM

    t = ModernBertForMaskedLM(ModernBertConfig(vocab_size=50368, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                               num_attention_heads=4, local_attention=16, max_position_embeddings=256)).eval()
    s = GTSLForMaskedLM(GTSLConfig(d_model=64, n_heads=4, d_state=16, n_layer=2, deep_trees=2, deep_depth=3)).init_from_modernbert(t)
    assert torch.equal(s.tok_embeddings.weight, t.model.embeddings.tok_embeddings.weight)
    q, k, v = t.model.layers[1].attn.Wqkv.weight.chunk(3, 0)
    w = s.layers[1].mixer.in_proj.weight
    torch.testing.assert_close(w[:64], q / 4.0)
    assert torch.equal(w[64:128], k) and torch.equal(w[128:], v)
    assert torch.equal(s.layers[1].mixer.out_proj.weight, t.model.layers[1].attn.Wo.weight)
    assert torch.equal(s.decoder_bias, t.decoder.bias) and torch.equal(s.head_dense.weight, t.head.dense.weight)
    ids = torch.randint(0, 50000, (2, 20))
    assert s(ids).shape == (2, 20, 50368)


def test_linear_path_init_recovers_a_low_rank_affine_map():
    """init_linear on the sums of an exactly rank-r affine map recovers it (the linear path alone, full precision)."""
    from mamba_ssm.models.gts_l import GTSLBlock

    torch.manual_seed(0)
    c = GTSLConfig(d_model=64, n_heads=2, d_state=8, n_layer=1, deep_trees=1, deep_depth=1, ternary_group=32, linear_rank=8)
    blk = GTSLBlock(c, 1)
    x = torch.randn(4096, 64, dtype=torch.float64)
    A = torch.randn(64, 8, dtype=torch.float64) @ torch.randn(8, 64, dtype=torch.float64) / 8
    b = torch.randn(64, dtype=torch.float64)
    y = x @ A + b
    x1 = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
    blk.init_linear(x1.T @ x1, x1.T @ y, eps=1e-9)
    with torch.no_grad():
        got = blk.lin_up(blk.lin_down(x.float())) + blk.lin_bias
    torch.testing.assert_close(got.double(), y, rtol=1e-3, atol=1e-3)


def test_proj_group_zero_is_one_scale_per_row():
    """proj_group=0: the BiSSD's and the linear path's ternary weights have one scale per row; the trees keep groups."""
    from mamba_ssm.modules.ternary import absmean_ternary

    torch.manual_seed(0)
    c = GTSLConfig(d_model=256, n_heads=4, d_state=16, n_layer=1, deep_trees=1, deep_depth=1, linear_rank=64, proj_group=0)
    blk = GTSLForMaskedLM(c).layers[0]
    for lin in (blk.mixer.in_proj, blk.mixer.out_proj, blk.lin_down, blk.lin_up):
        assert lin.group is None
        q = absmean_ternary(torch.randn_like(lin.weight), lin.group, 1.0)  # (lin_up starts at zero)
        mags = q.abs().masked_fill(q == 0, float("nan"))
        spread = mags.nan_to_num(0).amax(1) - mags.nan_to_num(float("inf")).amin(1)
        assert spread.abs().max() < 1e-6  # every nonzero entry of a row has the same magnitude
    assert blk.deep.ternary_group == c.ternary_group
