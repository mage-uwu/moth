# Golden Tree Snake (GTS) fork, 2026.
"""GTSSparse: the stateless deep trees with sparse kernels (``ops/gts_sparse.py``) instead of the dense route GEMMs.

Same parameters, same function: ``sparsify(model)`` switches every deep-tree mixer of a loaded GTS model (encoder or
masked LM) in place, and any checkpoint loads unchanged. On CUDA with triton, a forward pass that needs no gradient
(inference, evaluation, a teacher) runs the sparse kernel; anything else falls back to the dense route path.
"""
import torch

from mamba_ssm.modules.gts import GTS


class GTSSparse(GTS):
    sparse = True

    def _sparse_ok(self, x, return_paths):
        from mamba_ssm.ops.gts_sparse import HAVE_TRITON

        return (self.sparse and HAVE_TRITON and x.is_cuda and not torch.is_grad_enabled() and not self.use_context
                and not self.read_state and not self.capture_paths and self.dense_walk)

    def _forward_route_ste(self, x, mask, return_paths):
        if not self._sparse_ok(x, return_paths):
            return super()._forward_route_ste(x, mask, return_paths)
        from mamba_ssm.ops.gts_sparse import sparse_route_fwd

        batch, length, d = x.shape
        if self._frozen_q:
            w_in, w_out, bias = self._frozen_q["padded"]
        else:
            w_in, w_out, bias = self._w_in(), self._w_out(), self.node_bias
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())
        out, nodes, _ = sparse_route_fwd(x.reshape(batch * length, d), w_in, bias, w_out, self.n_trees, self.n_nodes,
                                         self.depth, self.act)
        out = out.view(batch, length, d).to(x.dtype) * mask.unsqueeze(-1)
        if return_paths:
            return out, nodes.view(batch, length, -1).long()
        return out


def sparsify(model, enable=True):
    """Switch every stateless deep-tree mixer in ``model`` to GTSSparse (in place; parameters untouched). Returns the
    number of mixers switched. ``enable=False`` switches them back to the dense path."""
    n = 0
    for m in model.modules():
        if isinstance(m, GTS) and not m.use_context and not m.read_state and m.depth > 0:
            m.__class__ = GTSSparse if enable else GTS
            n += 1
    return n
