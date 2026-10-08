"""Repository-native UPC tests (run from gts/). No external data/checkpoint needed."""
import numpy as np
import pytest
import torch
from mamba_ssm.models.gts_upc import UPCConfig, GTS3UPC, OPS
from mamba_ssm.models.upc_frontend import compile_policy, ProgramError, BudgetExceeded
from mamba_ssm.models.upc_runtime import Runtime, Decoder
from mamba_ssm.models.upc_install import install


def tiny():
    return UPCConfig(d_model=64, n_layer=2, bank_trees=8, bank_heads=2,
                     deep_trees=2, deep_depth=3)


def sample():
    names = torch.tensor([[7, 21, 4, 9, 19, 32, 87, 12]])
    ins = torch.tensor([[OPS.index('add'), 21, 7, 9, 21]])
    return ins, names, torch.zeros(1, 8, 4)


def test_parameter_budget():
    assert GTS3UPC().parameters_count() == 9_967_276


def test_shapes_and_gradient():
    torch.manual_seed(23)
    m = GTS3UPC(tiny())
    ins, names, num = sample()
    y = m(ins, names, num)
    assert y['opcode'].shape == (1, 32)
    assert y['pointers'].shape == (1, 4, 8)
    loss = y['opcode'].square().mean() + y['pointers'].square().mean()
    loss.backward()
    for block in m.layers:
        for w in (block.mixer.bank.node_in, block.mixer.bank.ctx_proj.weight,
                  block.mixer.deep.node_in, block.mixer.deep.node_out):
            assert w.grad is not None and torch.isfinite(w.grad).all()
            assert w.grad.abs().sum() > 0


def test_pointer_mask():
    m = GTS3UPC(tiny())
    ins, names, num = sample()
    mask = torch.ones(1, 8, dtype=torch.bool)
    mask[:, -2:] = False
    y = m(ins, names, num, mask)
    assert torch.isneginf(y['pointers'][:, :, -2:]).all()


def test_save_load(tmp_path):
    m = GTS3UPC(tiny()).freeze()
    path = tmp_path / 'checkpoint.pt'
    m.save(path)
    n = GTS3UPC.load(path).freeze()
    ins, names, num = sample()
    with torch.no_grad():
        a, b = m(ins, names, num), n(ins, names, num)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


@pytest.mark.parametrize('source', [
    'import os\nemit(x)', "emit(__import__('os').system('x'))",
    "emit(state('x').shape)",
])
def test_restricted_syntax(source):
    with pytest.raises(ProgramError):
        compile_policy(source)


def test_exact_execution_and_legality():
    p = compile_policy('x=state("x")\nemit(where(x>1, x*2, 0))')
    r = Runtime(Decoder())
    state = {'x': np.array([1, 2, 3], np.float32)}
    np.testing.assert_array_equal(r.scores(p, state), [0, 4, 6])
    assert r.act(p, state, legal=[True, True, False]) == 1
    assert r.act(p, state, temperature=50, legal=[True, False, False]) == 0
    with pytest.raises(ValueError):
        r.act(p, state, temperature=float('nan'))


def test_loop_and_budget():
    p = compile_policy('i=0\nwhile i<4:\n i=i+1\nemit(state("x")+i)')
    np.testing.assert_array_equal(Runtime(Decoder()).scores(p, {'x': np.ones(3)}), [5]*3)
    with pytest.raises(BudgetExceeded):
        Runtime(Decoder(), max_instructions=2).scores(p, {'x': np.ones(3)})


def test_broadcast_budget():
    p = compile_policy('emit(state("x")+state("y"))')
    with pytest.raises(BudgetExceeded):
        Runtime(Decoder(), max_elements=200).scores(p, {'x': np.ones((100, 1)), 'y': np.ones((1, 100))})


def test_install_preserves_policy_and_cache_identity():
    p = compile_policy('emit(-abs(state("x")-0.5))')
    q, stats = install(p, Decoder())
    state = {'x': np.array([.1, .5, .8], np.float32)}
    np.testing.assert_allclose(Runtime(Decoder()).scores(p, state), Runtime(Decoder()).scores(q, state))
    assert q.source != p.source
    assert stats['instructions'] == len(p.code)
