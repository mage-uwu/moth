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
