# Golden Tree Snake (GTS) fork, 2026.
"""Sparse deep trees (ops/gts_sparse.py, modules/gts_sparse.py) against the dense route path.

On a GPU these run on CUDA. Without one they run in Triton's interpreter: TRITON_INTERPRET=1 python -m pytest ...
"""
import os

import pytest
import torch

from mamba_ssm.modules.gts import GTS, GTSMixed
from mamba_ssm.ops.gts_sparse import HAVE_TRITON, sparse_route_fwd

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


@pytest.mark.parametrize("xres", [False, True])
@pytest.mark.parametrize("n_tok,d,n_trees,depth,bias,pad", [(70, 96, 2, 3, True, 0), (33, 64, 4, 5, False, 64), (5, 48, 1, 0, True, 0)])
def test_kernel_matches_dense(n_tok, d, n_trees, depth, bias, pad, xres):
    torch.manual_seed(0)
    n_nodes = 2 ** (depth + 1) - 1
    rows = n_trees * n_nodes + pad
    x = torch.randn(n_tok, d, device=DEVICE)
    w_in, w_out = torch.randn(rows, d, device=DEVICE) / d ** 0.5, torch.randn(rows, d, device=DEVICE)
    b = torch.randn(rows, device=DEVICE) * 0.1 if bias else None
    out, nodes, logits = sparse_route_fwd(x, w_in, b, w_out, n_trees, n_nodes, depth, block_m=16, block_d=32, x_resident=xres)
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
