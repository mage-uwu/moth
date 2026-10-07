# Golden Tree Snake (GTS) fork, 2026.
"""Sparse deep trees (ops/gts_sparse.py, modules/gts_sparse.py) against the dense route path.

On a GPU these run on CUDA. Without one they run in Triton's interpreter: TRITON_INTERPRET=1 python -m pytest ...
"""
import os

import pytest
import torch

from mamba_ssm.modules.gts import GTS, GTSMixed
from mamba_ssm.ops.gts_sparse import HAVE_TRITON, pack_rows, sparse_route_fwd, sparse_route_fwd_packed

DEVICE = "cuda" if torch.cuda.is_available() else ("cpu" if os.environ.get("TRITON_INTERPRET") == "1" else None)
pytestmark = pytest.mark.skipif(not HAVE_TRITON or DEVICE is None, reason="needs CUDA, or TRITON_INTERPRET=1 on CPU")


def _dense(x, w_in, bias, w_out, n_trees, n_nodes, depth):
    """Reference: every logit, the hard walk, gelu coefficients, in float64."""
    L = x.double() @ w_in.double()[: n_trees * n_nodes].t()
    if bias is not None:
        L = L + bias.double()[: n_trees * n_nodes]
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float64, device=x.device)
    nodes = torch.empty(x.shape[0], n_trees, depth + 1, dtype=torch.long, device=x.device)
    for t in range(n_trees):
        cur = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for k in range(depth + 1):
            node = t * n_nodes + cur
            lg = L.gather(1, node[:, None]).squeeze(1)
            out += torch.nn.functional.gelu(lg)[:, None] * w_out.double()[node]
            nodes[:, t, k] = node
            cur = 2 * cur + 1 + (lg > 0).long()
    return out, nodes


@pytest.mark.parametrize("xres,top", [(False, 0), (True, 0), (False, 2), (False, 9)])
@pytest.mark.parametrize("n_tok,d,n_trees,depth,bias,pad", [(70, 96, 2, 3, True, 0), (33, 64, 4, 5, False, 64), (5, 48, 1, 0, True, 0)])
def test_kernel_matches_dense(n_tok, d, n_trees, depth, bias, pad, xres, top):
    torch.manual_seed(0)
    n_nodes = 2 ** (depth + 1) - 1
    rows = n_trees * n_nodes + pad
    x = torch.randn(n_tok, d, device=DEVICE)
    w_in, w_out = torch.randn(rows, d, device=DEVICE) / d ** 0.5, torch.randn(rows, d, device=DEVICE)
    b = torch.randn(rows, device=DEVICE) * 0.1 if bias else None
    out, nodes, logits = sparse_route_fwd(x, w_in, b, w_out, n_trees, n_nodes, depth, block_m=16, block_d=32, x_resident=xres, top=top)
    ref, ref_nodes = _dense(x, w_in, b, w_out, n_trees, n_nodes, depth)
    assert torch.equal(nodes.long(), ref_nodes)
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_tok,d,n_trees,depth,group,bias", [(70, 128, 2, 3, 64, True), (33, 256, 4, 5, 128, False)])
def test_packed_kernel_matches_dense(n_tok, d, n_trees, depth, group, bias):
    """2-bit tables: the same paths and output as the dense reference on the dequantised weights."""
    torch.manual_seed(0)
    n_nodes = 2 ** (depth + 1) - 1
    rows = n_trees * n_nodes
    codes_in, codes_out = torch.randint(-1, 2, (rows, d), device=DEVICE), torch.randint(-1, 2, (rows, d), device=DEVICE)
    s_in, s_out = torch.rand(rows, d // group, device=DEVICE) * 0.1, torch.rand(rows, d // group, device=DEVICE)
    pin, pout = pack_rows(codes_in, s_in, torch.float32), pack_rows(codes_out, s_out, torch.float32)
    w_in = codes_in.float() * s_in.repeat_interleave(group, 1)
    w_out = codes_out.float() * s_out.repeat_interleave(group, 1)
    x = torch.randn(n_tok, d, device=DEVICE)
    b = torch.randn(rows, device=DEVICE) * 0.1 if bias else None
    out, nodes, _ = sparse_route_fwd_packed(x, pin, pout, b, n_trees, n_nodes, depth, group, block_m=16, block_d=32)
    ref, ref_nodes = _dense(x, w_in, b, w_out, n_trees, n_nodes, depth)
    assert torch.equal(nodes.long(), ref_nodes)
    torch.testing.assert_close(out.double(), ref, rtol=1e-4, atol=1e-4)


def test_module_matches_dense_path():
    """A GTSMixed layer: deep trees switched to GTSSparse give the dense module's output under no_grad."""
    from mamba_ssm.modules.gts_sparse import sparsify

    torch.manual_seed(0)
    layer = GTSMixed(64, bank_trees=4, bank_heads=2, bank_state=4, deep_trees=2, deep_depth=4, route_ste=True,
                     ternary=True, act_bits=8).to(DEVICE).eval()
    u = torch.randn(2, 40, 64, device=DEVICE)
    mask = torch.ones(2, 40, device=DEVICE)
    mask[1, 30:] = 0
    with torch.no_grad():
        ref = layer.deep(u, attention_mask=mask)
        assert sparsify(layer) == 1
        if DEVICE == "cpu":  # the module only takes the sparse path on CUDA; call it directly here
            layer.deep._sparse_ok = lambda x, r: True
        out = layer.deep(u, attention_mask=mask)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_prepare_inference_matches_dense():
    """prepare_inference (frozen quantisation, 2-bit tables) gives the dense module's output in float32."""
    from mamba_ssm.modules.gts_sparse import prepare_inference

    torch.manual_seed(0)
    layer = GTSMixed(128, bank_trees=4, bank_heads=2, bank_state=4, deep_trees=2, deep_depth=4, route_ste=True,
                     ternary=True, act_bits=8).to(DEVICE).eval()
    u = torch.randn(2, 40, 128, device=DEVICE)
    with torch.no_grad():
        ref = layer(u)
        prepare_inference(layer, dtype=torch.float32)
        if DEVICE == "cpu":
            layer.deep._sparse_ok = lambda x, r: True
        out = layer(u)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n_tok,d,n_trees,depth,bias", [(70, 96, 2, 3, True), (33, 64, 3, 5, False), (130, 32, 1, 2, True)])
def test_path_gradients_match_dense(n_tok, d, n_trees, depth, bias):
    """sparse_path_route: values and every gradient against an autograd reference with the hard walk and no
    gradient through the branches (route_ste=False), in float64."""
    from mamba_ssm.ops.gts_sparse import sparse_path_route

    torch.manual_seed(0)
    n_nodes = 2 ** (depth + 1) - 1
    rows = n_trees * n_nodes
    x = torch.randn(n_tok, d, device=DEVICE)
    w_in, w_out = torch.randn(rows, d, device=DEVICE) / d ** 0.5, torch.randn(rows, d, device=DEVICE)
    b = torch.randn(rows, device=DEVICE) * 0.1 if bias else None
    dout = torch.randn(n_tok, d, device=DEVICE)
    leaves = [x, w_in, w_out] + ([b] if bias else [])
    ins = [t.clone().requires_grad_() for t in leaves]
    xs, wis, wos = ins[:3]
    bs = ins[3] if bias else None
    out = sparse_path_route(xs, wis, bs, wos, n_trees, n_nodes, depth, block_m=16, block_d=32)
    (out * dout).sum().backward()
    ref_ins = [t.double().clone().requires_grad_() for t in leaves]
    xr, wir, wor = ref_ins[:3]
    br = ref_ins[3] if bias else None
    L = xr @ wir.t() + (br if bias else 0)
    ref = torch.zeros(n_tok, d, dtype=torch.float64, device=DEVICE)
    for t in range(n_trees):
        cur = torch.zeros(n_tok, dtype=torch.long, device=DEVICE)
        for k in range(depth + 1):
            node = t * n_nodes + cur
            lg = L.gather(1, node[:, None]).squeeze(1)
            ref = ref + torch.nn.functional.gelu(lg)[:, None] * wor[node]
            cur = 2 * cur + 1 + (lg.detach() > 0).long()
    (ref * dout.double()).sum().backward()
    torch.testing.assert_close(out.double(), ref.detach(), rtol=1e-4, atol=1e-4)
    for a, r in zip(ins, ref_ins):
        torch.testing.assert_close(a.grad.double(), r.grad, rtol=1e-3, atol=1e-3)


def test_module_training_matches_dense():
    """A GTSMixed layer with route_ste=False: the sparse training path (forward and backward) against the dense one,
    every parameter's gradient included (ternary weights through the straight-through estimator, 8-bit activations)."""
    import copy

    from mamba_ssm.modules.gts_sparse import sparsify

    torch.manual_seed(0)
    ref = GTSMixed(64, bank_trees=4, bank_heads=2, bank_state=4, deep_trees=2, deep_depth=4, route_ste=False,
                   ternary=True, act_bits=8).to(DEVICE).double()
    sp = copy.deepcopy(ref)
    sparsify(sp)
    if DEVICE == "cpu":  # the sparse path only switches on on CUDA; force it here
        sp.deep._sparse_train_ok = lambda x: True
        sp.deep._use_route_kernel = lambda x: True
    u = torch.randn(2, 40, 64, device=DEVICE, dtype=torch.float64)
    mask = torch.ones(2, 40, device=DEVICE, dtype=torch.float64)
    mask[1, 33:] = 0
    g = torch.randn(2, 40, 64, device=DEVICE, dtype=torch.float64)
    ua, ub = u.clone().requires_grad_(), u.clone().requires_grad_()
    (ref.deep(ua, attention_mask=mask) * g).sum().backward()
    (sp.deep(ub, attention_mask=mask) * g).sum().backward()
    torch.testing.assert_close(ub.grad, ua.grad, rtol=1e-6, atol=1e-6)
    for (n, a), (_, b) in zip(ref.deep.named_parameters(), sp.deep.named_parameters()):
        if a.grad is None:
            assert b.grad is None or not b.grad.any(), n
            continue
        err = ((b.grad - a.grad).abs().max() / a.grad.abs().max()).item()
        assert err < 1e-5, n
