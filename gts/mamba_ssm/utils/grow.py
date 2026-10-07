# Golden Tree Snake (GTS) fork, 2026.
"""Function-preserving growth of a GTS masked LM's deep trees: more parameters at almost no inference cost.

A deep tree is stored in heap order (node i's children are 2i + 1 and 2i + 2), so its levels 0..D are the first
2^(D+1) - 1 rows of a deeper tree. Growing by k levels keeps every existing node where it is and appends the new
levels with fresh routing weights and **zero output rows**: a token's path walks k nodes further, but those nodes
add exactly nothing (a zero row ternarises to zero codes), so the grown model computes exactly what the original does.
Training then gives the new leaves their own outputs. Each tree gets 2^k times the parameters; a token's path gets k
more nodes, so the per-token cost of a deep tree rises by k / (D + 1).

    from mamba_ssm.utils.grow import grow_deep_trees
    state, cfg = grow_deep_trees(checkpoint["model"], checkpoint["config"], add_levels=2)
    model = GTSForMaskedLM(GTSConfig(**cfg)); model.load_state_dict(state)
"""

import math

import torch

__all__ = ["grow_deep_trees"]


@torch.no_grad()
def grow_deep_trees(state, config, add_levels=2, seed=0):
    """``state``: a GTSForMaskedLM state dict (GTS-Uni included); ``config``: its GTSConfig fields. Returns the grown
    state dict and config (deep_depth + add_levels)."""
    assert config.get("mixer") == "mixed", "only the mixed forest has deep trees"
    depth, d = config["deep_depth"], config["d_model"]
    trees = config["deep_trees"]
    n_old, n_new = 2 ** (depth + 1) - 1, 2 ** (depth + add_levels + 1) - 1
    g = torch.Generator().manual_seed(seed)
    k_in = 1.0 / math.sqrt(d)  # the same range GTS initialises node_in and node_bias with
    out = dict(state)
    for key, t in state.items():
        if ".mixer.deep." not in key or key.rsplit(".", 1)[-1] not in ("node_in", "node_bias", "node_out"):
            continue
        name = key.rsplit(".", 1)[-1]
        old = t.view(trees, n_old, *t.shape[1:])
        shape = (trees, n_new - n_old, *t.shape[1:])
        if name == "node_out":
            fresh = torch.zeros(shape, dtype=t.dtype)
        else:
            fresh = (torch.rand(shape, generator=g, dtype=torch.float32) * 2 - 1).to(t.dtype) * k_in
        out[key] = torch.cat([old, fresh], 1).reshape(trees * n_new, *t.shape[1:]).contiguous()
    cfg = dict(config, deep_depth=depth + add_levels)
    return out, cfg
