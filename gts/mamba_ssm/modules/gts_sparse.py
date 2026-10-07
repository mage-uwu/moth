# Golden Tree Snake (GTS) fork, 2026.
"""GTSSparse: the stateless deep trees with sparse kernels (``ops/gts_sparse.py``) instead of the dense route GEMMs.

Same parameters, same function: ``sparsify(model)`` switches every deep-tree mixer of a loaded GTS model (encoder or
masked LM) in place, and any checkpoint loads unchanged. On CUDA with triton, a forward pass that needs no gradient
(inference, evaluation, a teacher) runs the sparse kernel; anything else falls back to the dense route path.

For inference, ``prepare_inference(model)`` also quantises every ternary weight once (``GTS.freeze_quantized``) and
packs the deep trees' node tables to 2 bits, so the kernel gathers about 200 bytes per node row instead of 1,536.
The sparse call is a ``torch.library`` custom op, so ``torch.compile`` of a block keeps one graph around it.
"""
import torch

from mamba_ssm.modules.gts import GTS
from mamba_ssm.modules.ternary import pack_ternary

KERNEL_ARGS = {}  # tile sizes for the unpacked kernel (bench_sparse.py sets the fastest it found)
PACKED_ARGS = {}  # and for the 2-bit one
_ACT = {"gelu": 0, "split": 0, "linear": 1}


@torch.library.custom_op("gts_sparse::route_fwd_packed", mutates_args=())
def _route_fwd_packed_op(x: torch.Tensor, pi: torch.Tensor, si: torch.Tensor, po: torch.Tensor, so: torch.Tensor,
                         bias: torch.Tensor | None, n_trees: int, n_nodes: int, depth: int, group: int, act: int,
                         block_m: int, block_d: int, num_warps: int) -> torch.Tensor:
    from mamba_ssm.ops.gts_sparse import sparse_route_fwd_packed

    act_name = "gelu" if act == 0 else "linear"
    out, _, _ = sparse_route_fwd_packed(x, (pi, si), (po, so), bias, n_trees, n_nodes, depth, group, act_name,
                                        block_m=block_m, block_d=block_d, num_warps=num_warps)
    return out


@_route_fwd_packed_op.register_fake
def _(x, pi, si, po, so, bias, n_trees, n_nodes, depth, group, act, block_m, block_d, num_warps):
    return x.new_empty(x.shape, dtype=torch.float32)


class GTSSparse(GTS):
    sparse = True

    def _sparse_ok(self, x, return_paths):
        from mamba_ssm.ops.gts_sparse import HAVE_TRITON

        return (self.sparse and HAVE_TRITON and x.is_cuda and not torch.is_grad_enabled() and not self.use_context
                and not self.read_state and not self.capture_paths and self.dense_walk)

    def freeze_quantized(self, dtype=None):
        """GTS.freeze_quantized, plus the deep trees' node tables packed to 2 bits (scales in ``dtype``, default
        bfloat16: the dense path's weights under bf16 autocast are bf16(scale) * code, which these reproduce)."""
        super().freeze_quantized(dtype)
        if self.ternary:
            sd = dtype or torch.bfloat16
            from mamba_ssm.ops.gts_sparse import pack_rows

            ci, si = pack_ternary(self.node_in.detach(), self.ternary_group)
            co, so = pack_ternary(self.node_out.detach(), self.ternary_group)
            self._frozen_q["packed"] = (pack_rows(ci, si, sd), pack_rows(co, so, sd), self.node_in.shape[-1] // si.shape[-1])

    def _forward_route_ste(self, x, mask, return_paths):
        if not self._sparse_ok(x, return_paths):
            return super()._forward_route_ste(x, mask, return_paths)
        batch, length, d = x.shape
        if torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_gpu_dtype())
        xf = x.reshape(batch * length, d)
        packed = self._frozen_q.get("packed") if self._frozen_q else None
        if packed is not None and not return_paths:
            (pi, si), (po, so), group = packed
            kw = dict(block_m=8, block_d=min(128, group), num_warps=2, **PACKED_ARGS)
            bias = self.node_bias.detach() if self.node_bias is not None else None
            out = _route_fwd_packed_op(xf, pi, si, po, so, bias, self.n_trees, self.n_nodes, self.depth, group,
                                       _ACT[self.act], kw["block_m"], kw["block_d"], kw["num_warps"])
            return out.view(batch, length, d).to(x.dtype) * mask.unsqueeze(-1)
        from mamba_ssm.ops.gts_sparse import sparse_route_fwd

        if self._frozen_q:
            w_in, w_out, bias = self._frozen_q["padded"]
        else:
            w_in, w_out, bias = self._w_in(), self._w_out(), self.node_bias
        out, nodes, _ = sparse_route_fwd(xf, w_in, bias, w_out, self.n_trees, self.n_nodes, self.depth, self.act,
                                         **KERNEL_ARGS)
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


def prepare_inference(model, dtype=torch.bfloat16, sparse=True):
    """For inference with fixed weights: sparse deep trees (2-bit tables) and every ternary weight quantised once.
    Call again after changing the weights."""
    if sparse:
        sparsify(model)
    for m in model.modules():
        if isinstance(m, GTS):
            m.freeze_quantized(dtype)
    return model
