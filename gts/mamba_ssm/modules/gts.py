# Copyright (c) 2024, Tri Dao, Albert Gu.
# Golden Tree Snake (GTS) fork, 2026.
"""Golden Tree Snake (GTS): a tree-routed, bidirectional state space mixer.

Forked from two files in this repository:

* ``modules/mamba2.py``   - the block layout, the per-head scalar decay and its
  initialisation, and the idea that B, C and dt are read off the residual stream.
* ``modules/ssd_minimal.py`` - the quadratic "decay matrix" form used for training.

The change is that Mamba-2's dense inner channels become the nodes of a binary
tree, in the manner of fast feedforward networks (Belcak & Wattenhofer, 2023,
https://arxiv.org/abs/2311.10770). A token walks one root-to-leaf path and only
the nodes on that path are computed, written to and read from.

How the Mamba-2 pieces map:

    Mamba-2                            GTS
    ---------------------------------  ------------------------------------------
    inner channel                      tree node
    in_proj row for x                  node_in row (also decides the branch)
    out_proj column                    node_out row
    head (channels sharing a decay)    tree level (one clock per level)
    B, C from in_proj                  B, C_fwd, C_bwd from ctx_proj
    state per channel, always updated  state per node, updated only when visited

What a token does at each node on its path (level k, node i):

    logit = <x_t, node_in[i]> + bias[i]            one dot product
    ctx   = <C_t, decayed state of node i>         reads tokens that visited i before
    state of node i += dt_t[k] * logit * B_t       writes its own key
    out_t += gelu(logit + ctx) * node_out[i]       one scaled add
    branch right if logit > 0                      hard routing, as in FFF

With ``use_context=False`` this is exactly an FFF layer. With ``depth=0`` every
token shares the single root node and it is a one-channel linear-attention SSM.

Two implementations of the same function live here:

* ``forward``            - vectorised and differentiable; quadratic in sequence
                           length. This is the training path.
* ``forward_reference``  - token at a time with lazily decayed node states.
                           This is the algorithm a CPU kernel implements.

``tests/modules/test_gts.py`` checks that they agree.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.ternary import absmean_ternary, quantize_activations

__all__ = ["GTS", "GTSMixed"]


class GTS(nn.Module):
    def __init__(
        self,
        d_model,
        depth=11,  # tree depth; a path visits depth + 1 nodes
        n_trees=1,  # independent trees (FFF's "parallel_size"); many shallow trees move GTS towards a dense SSM
        n_heads=1,  # trees are split into this many groups; each group has its own clock per level (its own decay)
        d_state=16,  # size of the key/query vectors and of each node's state
        d_conv=3,  # depthwise conv on the residual stream (centred, or causal if causal=True); 0 disables
        causal=False,  # forward context only, for autoregressive models
        selective=True,  # input-dependent decay (Mamba-2 style) vs fixed decay per level
        use_context=True,  # False gives a plain FFF layer
        read_state=False,  # also read each path node's whole state vector into the output (not only <C, state>)
        write_logit=True,  # a token writes dt * logit * B to a node; False writes dt * B
        act="gelu",  # output coefficient gelu(logit + ctx), or "linear": logit + ctx, so left turns contribute too
        route_ste=False,  # straight-through gradient for the branch decisions (training path only; same forward pass)
        route_ste_temp=1.0,  # temperature of the sigmoid whose slope stands in for the step's
        ternary=False,  # ternary node tables and ctx_proj, trained with STE (see modules/ternary.py)
        ternary_group=128,  # weights per absmean scale, along each row
        act_bits=None,  # if set (and ternary), quantise the mixer input to this many bits per token
        node_bias=True,
        dense_walk=None,  # training path: compute every node's logit with one matmul (default when the trees are small)
        scan_kernel=None,  # depth 0: compute the context with the chunked Triton scan (ops/gts_scan.py); default on CUDA when triton is installed
        route_kernel=None,  # stateless trees with route_ste: Triton walk kernels (ops/gts_route.py); default on CUDA when triton is installed
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert causal or d_conv == 0 or d_conv % 2 == 1, "d_conv must be odd when the conv is centred"
        self.d_model = d_model
        self.depth = depth
        self.n_trees = n_trees
        assert n_trees % n_heads == 0, "n_heads must divide n_trees"
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_conv = d_conv
        self.causal = causal
        self.n_queries = 1 if causal else 2  # C_fwd, and C_bwd unless causal
        self.selective = selective
        self.use_context = use_context
        self.read_state = read_state and use_context
        self.write_logit = write_logit
        assert act in ("gelu", "linear", "split")  # split: gelu(logit) + ctx, context enters linearly
        self.act = act
        self.route_ste = route_ste
        self.route_ste_temp = route_ste_temp
        self.ternary = ternary
        self.ternary_group = ternary_group
        self.act_bits = act_bits
        self.quant_lambda = 1.0  # 0 = full precision, 1 = fully quantised; ramp it to warm quantisation in
        self.capture_paths = False  # if True, forward stores last_nodes / last_logits for distillation
        self.layer_idx = layer_idx

        self.n_levels = depth + 1
        self.n_nodes = 2 ** (depth + 1) - 1  # per tree
        self.n_slots = n_trees * self.n_levels  # nodes touched per token
        # Like fff.py, the training path can compute all nodes densely and keep the path's. That costs
        # (total nodes x d_model) per token instead of gathering (slots x d_model) rows, which is cheaper
        # in PyTorch while the node tables are small. Inference kernels never do this.
        self.dense_walk = (n_trees * self.n_nodes <= 2048) if dense_walk is None else dense_walk
        # A depth-0 tree is visited by every token, so its context is Mamba-2's SSD (with the token's own term
        # excluded) and can run as a chunked scan, linear in length, instead of the quadratic form below.
        assert not (scan_kernel and depth > 0), "the scan kernel covers depth-0 trees only"
        self.scan_kernel = scan_kernel
        self.route_kernel = route_kernel
        # slot s of a path belongs to tree s // n_levels and level s % n_levels
        self.register_buffer("slot_level", torch.arange(self.n_levels).repeat(n_trees), persistent=False)
        self.register_buffer("tree_offset", torch.arange(n_trees) * self.n_nodes, persistent=False)
        # A group is (head, level): the nodes that share one clock. slot_group maps a path slot to its group;
        # node_col lays all nodes out group by group, so that a group's nodes are contiguous columns.
        self.n_groups = n_heads * self.n_levels
        head_of_tree = torch.arange(n_trees) * n_heads // n_trees
        self.register_buffer("slot_group", (head_of_tree[:, None] * self.n_levels + torch.arange(self.n_levels)[None, :]).flatten(), persistent=False)
        members = [[] for _ in range(self.n_groups)]
        for tree in range(n_trees):
            for level in range(self.n_levels):
                members[int(head_of_tree[tree]) * self.n_levels + level] += [tree * self.n_nodes + i for i in range(2**level - 1, 2 ** (level + 1) - 1)]
        self.group_bounds = [0]
        for m in members:
            self.group_bounds.append(self.group_bounds[-1] + len(m))
        node_col = torch.empty(n_trees * self.n_nodes, dtype=torch.long)
        node_col[torch.tensor([i for m in members for i in m])] = torch.arange(n_trees * self.n_nodes)
        self.register_buffer("node_col", node_col, persistent=False)
        self.register_buffer("col_node", torch.tensor([i for m in members for i in m]), persistent=False)  # inverse of node_col
        node_group = torch.empty(n_trees * self.n_nodes, dtype=torch.long)
        for g, m in enumerate(members):
            node_group[torch.tensor(m)] = g
        self.register_buffer("node_group", node_group, persistent=False)  # which clock each node runs on

        # Node tables. Initialisation follows UltraFastBERT's FFF.
        total_nodes = n_trees * self.n_nodes
        k_in = math.sqrt(1.0 / d_model)
        self.node_in = nn.Parameter(torch.empty(total_nodes, d_model, **factory_kwargs).uniform_(-k_in, k_in))
        if node_bias:
            self.node_bias = nn.Parameter(torch.empty(total_nodes, **factory_kwargs).uniform_(-k_in, k_in))
            self.node_bias._no_weight_decay = True
        else:
            self.register_parameter("node_bias", None)
        k_out = math.sqrt(1.0 / self.n_slots)
        self.node_out = nn.Parameter(torch.empty(total_nodes, d_model, **factory_kwargs).uniform_(-k_out, k_out))

        if d_conv > 0:
            self.conv1d = nn.Conv1d(
                d_model, d_model, kernel_size=d_conv, groups=d_model, padding=self._conv_pad(), bias=True, **factory_kwargs
            )
            # Start as the identity so the block begins as "route on the token itself".
            with torch.no_grad():
                self.conv1d.weight.zero_()
                self.conv1d.weight[:, 0, self._conv_pad()] = 1.0  # the tap on the token itself
                self.conv1d.bias.zero_()

        if use_context:
            # Order: [B, C_fwd, (C_bwd,) dt]   (Mamba-2's in_proj order is [z, x, B, C, dt])
            d_ctx = (1 + self.n_queries) * d_state + (self.n_groups if selective else 0)
            self.ctx_proj = nn.Linear(d_model, d_ctx, bias=False, **factory_kwargs)
            if self.read_state:
                # One dense read of the path's node states: (queries * slots * d_state) -> d_model.
                d_read = self.n_queries * self.n_slots * d_state
                k_read = 0.5 / math.sqrt(d_read)
                self.read_w = nn.Parameter(torch.empty(d_model, d_read, **factory_kwargs).uniform_(-k_read, k_read))
                self.read_norm_weight = nn.Parameter(torch.ones(d_read, **factory_kwargs))

            # Initialize log dt bias (as in Mamba-2, with one value per tree level instead of per head)
            dt = torch.exp(
                torch.rand(self.n_groups, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            )
            dt = torch.clamp(dt, min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias = nn.Parameter(inv_dt)
            self.dt_bias._no_weight_decay = True

            assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
            A = torch.empty(self.n_groups, dtype=torch.float32, device=device).uniform_(*A_init_range)
            self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype))
            self.A_log._no_weight_decay = True

    # ------------------------------------------------------------------ weights

    def _conv_pad(self):
        return self.d_conv - 1 if self.causal else self.d_conv // 2

    def _q(self, w):
        return absmean_ternary(w, self.ternary_group, self.quant_lambda) if self.ternary else w

    def _w_in(self):
        return self._q(self.node_in)

    def _w_out(self):
        return self._q(self.node_out)

    def _w_ctx(self):
        return self._q(self.ctx_proj.weight)

    def _read(self, v):
        """RMS-normalise the concatenated node states and project them to the residual width."""
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-8) * self.read_norm_weight
        return F.linear(v, self._q(self.read_w))

    # ------------------------------------------------------------------- pieces

    def _local_mix(self, u, mask):
        """Centred depthwise conv so that routing can depend on the neighbouring tokens."""
        u = u * mask.unsqueeze(-1)
        x = u if self.d_conv == 0 else self.conv1d(u.transpose(1, 2))[..., : u.shape[1]].transpose(1, 2)
        if self.ternary and self.act_bits is not None:
            x = quantize_activations(x, self.act_bits, self.quant_lambda)
        return x

    def _walk(self, x):
        """Walk every tree for every token.

        Returns
            nodes:  (batch, length, n_slots) long, global node ids along each path
            logits: (batch, length, n_slots) pre-activations at those nodes
        """
        batch, length, _ = x.shape
        w_in = self._w_in()
        cur = torch.zeros(batch, length, self.n_trees, dtype=torch.long, device=x.device)
        nodes, logits = [], []
        if self.dense_walk:
            all_logits = F.linear(x, w_in, self.node_bias)  # (b, l, total nodes)
        for level in range(self.n_levels):
            gid = cur + self.tree_offset  # (b, l, trees)
            if self.dense_walk:
                logit = all_logits.gather(2, gid)
            else:
                logit = torch.einsum("bld,bltd->blt", x, w_in[gid])
                if self.node_bias is not None:
                    logit = logit + self.node_bias[gid]
            nodes.append(gid)
            logits.append(logit)
            if level < self.n_levels - 1:
                # Hard branch. No gradient flows through the decision itself (as in FFF).
                cur = 2 * cur + 1 + (logit > 0).long()
        self._all_logits = all_logits if self.dense_walk else None  # every node's logit, for the straight-through path
        nodes = torch.stack(nodes, dim=-1).flatten(2)  # (b, l, trees * levels), tree-major
        logits = torch.stack(logits, dim=-1).flatten(2)
        return nodes, logits

    def _ctx_signals(self, x, mask):
        """Per-token key, queries, step size and log-decay. Depends on the token only, not on any state."""
        p = F.linear(x, self._w_ctx())
        n = self.d_state
        B = p[..., :n]
        C_fwd = p[..., n : 2 * n]
        C_bwd = None if self.causal else p[..., 2 * n : 3 * n]
        if self.selective:
            dt = F.softplus(p[..., (1 + self.n_queries) * n :] + self.dt_bias)  # (b, l, levels)
        else:
            dt = F.softplus(self.dt_bias).expand(*p.shape[:2], self.n_groups)
        A = -torch.exp(self.A_log.float())  # (levels,)
        # Log-decay contributed by each token to each level's clock. Padding does not advance the clocks.
        a = dt * A * mask.unsqueeze(-1)
        return B, C_fwd, C_bwd, dt, a

    def _context_dense(self, x, nodes, logits, mask):
        """Context read by every token at every node on its path, in the quadratic SSD form.

        For level k the forward term is

            ctx[t] = sum_{s < t, same node} exp(sum_{r=s+1..t} a_r[k]) * <C_fwd[t], B[s]> * dt_s[k] * logit_s

        which is ssd_minimal's ``Y_diag`` with the decay matrix L additionally masked by
        "token s and token t sit at the same node of this level". The backward term mirrors it for s > t.
        """
        B, C_fwd, C_bwd, dt, a = self._ctx_signals(x, mask)
        length = x.shape[1]
        lvl = self.slot_group
        if not self.read_state:
            return self._context_fast(B, C_fwd, C_bwd, dt, a, nodes, logits, mask), None

        # What token s writes at each slot of its path (compare Mamba-2's dBx = dt * B * x).
        src = dt[..., lvl] * mask.unsqueeze(-1)  # (b, s, slots)
        if self.write_logit:
            src = src * logits
        same = nodes[:, :, None, :] == nodes[:, None, :, :]  # (b, t, s, slots)
        cs = torch.cumsum(a, dim=1)  # inclusive prefix sums, (b, l, levels)
        lower = torch.ones(length, length, dtype=torch.bool, device=x.device).tril(-1)[None, :, :, None]

        def gather(seg, keep, C):
            # Mask before exp so the excluded triangle is exp(-inf) = 0 rather than inf * 0.
            decay = torch.exp(seg.masked_fill(~keep, -torch.inf))  # (b, t, s, levels)
            if self.n_slots != self.n_groups:
                decay = decay[..., lvl]
            m = (decay * same * src.unsqueeze(1)).permute(0, 3, 1, 2)  # (b, slots, t, s)
            state = (m @ B.unsqueeze(1)).permute(0, 2, 1, 3)  # (b, t, slots, d_state): what token t finds at each node
            return state, (state * C.unsqueeze(2)).sum(-1)

        seg_fwd = cs[:, :, None, :] - cs[:, None, :, :]  # [t, s] = sum_{r=s+1..t} a_r
        state, ctx = gather(seg_fwd, lower, C_fwd)
        states = [state]
        if not self.causal:
            cs_ex = cs - a  # exclusive prefix sums
            seg_bwd = cs_ex[:, None, :, :] - cs_ex[:, :, None, :]  # [t, s] = sum_{r=t..s-1} a_r
            state, ctx_bwd = gather(seg_bwd, lower.transpose(1, 2), C_bwd)
            states.append(state)
            ctx = ctx + ctx_bwd
        return ctx, states

    def _context_fast(self, B, C_fwd, C_bwd, dt, a, nodes, logits, mask):
        """The scalar context <C_t, state> without ever forming a (t, s, slots) tensor.

        Lay what every token wrote out by node (one column per node, a group's nodes contiguous), multiply each
        group's columns by that group's decay-weighted key/query matrix, and read each token's own columns back.
        The cost is (length^2 x total nodes) multiply-adds per layer, whatever the number of trees.
        """
        batch, length, _ = nodes.shape
        src = dt[..., self.slot_group] * mask.unsqueeze(-1)
        if self.write_logit:
            src = src * logits
        if self._use_scan(src):
            return self._context_scan(B, C_fwd, C_bwd, a, src)
        w = self._decay_weights(B, C_fwd, C_bwd, a)
        cols = self.node_col[nodes]  # (b, l, slots)
        z = torch.zeros(batch, length, self.n_trees * self.n_nodes, dtype=src.dtype, device=src.device).scatter(2, cols, src)
        y = torch.cat([w[..., g] @ z[:, :, self.group_bounds[g] : self.group_bounds[g + 1]] for g in range(self.n_groups)], dim=2)
        return y.gather(2, cols)

    def _use_scan(self, x):
        if self.depth > 0 or self.scan_kernel is False:
            return False
        from mamba_ssm.ops.gts_scan import HAVE_TRITON

        return HAVE_TRITON and (self.scan_kernel or (self.scan_kernel is None and x.is_cuda))

    def _context_scan(self, B, C_fwd, C_bwd, a, src):
        """Depth 0: slot = tree, and a head's trees are contiguous, so what the tokens write is (b, l, heads, trees per
        head) and each head is one SSD channel group on its own clock."""
        from mamba_ssm.ops.gts_scan import gts_scan

        batch, length, _ = src.shape
        X = src.reshape(batch, length, self.n_heads, self.n_trees // self.n_heads)
        ctx = gts_scan(C_fwd, B, X, a)  # a: (b, l, heads), one clock per head
        if not self.causal:
            ctx = ctx + gts_scan(C_bwd, B, X, a, reverse=True)  # decay a[t] + ... + a[s-1] from s > t
        return ctx.reshape(batch, length, self.n_trees).to(src.dtype)

    def _decay_weights(self, B, C_fwd, C_bwd, a):
        """(b, t, s, groups): <C[t], B[s]> times the decay between s and t on each group's clock; zero on the diagonal."""
        length = a.shape[1]
        cs = torch.cumsum(a, dim=1)  # (b, l, groups)
        lower = torch.ones(length, length, dtype=torch.bool, device=a.device).tril(-1)[None, :, :, None]
        seg = cs[:, :, None, :] - cs[:, None, :, :]  # [t, s] = sum_{r=s+1..t} a_r
        w = (C_fwd @ B.transpose(1, 2)).unsqueeze(-1) * torch.exp(seg.masked_fill(~lower, -torch.inf))
        if not self.causal:
            cs_ex = cs - a
            seg = cs_ex[:, None, :, :] - cs_ex[:, :, None, :]  # [t, s] = sum_{r=t..s-1} a_r
            w = w + (C_bwd @ B.transpose(1, 2)).unsqueeze(-1) * torch.exp(seg.masked_fill(~lower.transpose(1, 2), -torch.inf))
        return w

    def _path_weights(self, all_logits):
        """For every node, 1 if the token's path goes through it and 0 otherwise, with a straight-through gradient.

        A branch is the hard step ``logit > 0`` in the forward pass and ``sigmoid(logit / temp)`` in the backward
        pass. A node's weight is the product of the branch values on the way down to it, so the gradient that reaches
        a branch logit is the difference between what the model would have output had the token gone right and had
        it gone left, each followed down by the token's own decisions.
        """
        batch, length, _ = all_logits.shape
        al = all_logits.view(batch, length, self.n_trees, self.n_nodes)
        p = torch.sigmoid(al / self.route_ste_temp)
        right = (al > 0).to(al.dtype) + (p - p.detach())  # exactly 0 or 1 going forward
        levels = [al.new_ones(batch, length, self.n_trees, 1)]
        for k in range(self.n_levels - 1):
            g = right[..., 2**k - 1 : 2 ** (k + 1) - 1]  # the branches taken at level k's nodes
            levels.append(torch.stack([levels[-1] * (1 - g), levels[-1] * g], dim=-1).flatten(3))  # children 2i+1, 2i+2
        return torch.cat(levels, dim=-1).flatten(2)  # (b, l, total nodes), in node-id order

    def _forward_ste(self, x, mask, all_logits):
        """The same output as the path-only forward, written over all nodes so that branch decisions get a gradient:
        every node's coefficient (and what it would write) is computed, and the path weights zero the rest."""
        assert self.dense_walk and not self.read_state, "route_ste needs the dense training path and no state read"
        pi = self._path_weights(all_logits)
        ctx = 0.0
        if self.use_context:
            B, C_fwd, C_bwd, dt, a = self._ctx_signals(x, mask)
            src = dt[..., self.node_group] * mask.unsqueeze(-1)  # what the token would write at each node
            if self.write_logit:
                src = src * all_logits
            z = (pi * src)[..., self.col_node]
            w = self._decay_weights(B, C_fwd, C_bwd, a)
            y = torch.cat([w[..., g] @ z[:, :, self.group_bounds[g] : self.group_bounds[g + 1]] for g in range(self.n_groups)], dim=2)
            ctx = y[..., self.node_col]  # what the token would read at each node
        return (pi * self._coef(all_logits, ctx)) @ self._w_out()

    def _use_route_kernel(self, x):
        if self.use_context or self.read_state or self.route_kernel is False:
            return False
        from mamba_ssm.ops.gts_route import HAVE_TRITON

        return HAVE_TRITON and (self.route_kernel or (self.route_kernel is None and x.is_cuda))

    def _forward_route_ste(self, x, mask, return_paths):
        """route_ste: every node's logit from one matmul. The walk runs only if the paths are wanted; stateless trees
        on CUDA use the Triton kernels, which never form the path weights."""
        assert self.dense_walk and not self.read_state, "route_ste needs the dense training path and no state read"
        want = return_paths or self.capture_paths
        batch, length, _ = x.shape
        all_logits = F.linear(x, self._w_in(), self.node_bias)  # (b, l, total nodes)
        nodes = None
        if self._use_route_kernel(x):
            from mamba_ssm.ops.gts_route import route_ste_out

            out, nodes = route_ste_out(all_logits.reshape(batch * length, -1), self._w_out(), self.n_trees, self.depth,
                                       self.act, self.route_ste_temp, want)
            out = out.view(batch, length, -1).to(x.dtype)
            if want:
                nodes = nodes.view(batch, length, -1).long()
        else:
            out = self._forward_ste(x, mask, all_logits)
            if want:
                nodes, _ = self._walk(x)
                self._all_logits = None
        if self.capture_paths:
            self.last_nodes, self.last_logits = nodes, all_logits.gather(2, nodes)
        out = out * mask.unsqueeze(-1)
        return (out, nodes) if return_paths else out

    def _coef(self, logit, ctx):
        """Output coefficient of a path node from its branch logit and the context it read."""
        if self.act == "split":
            return F.gelu(logit) + ctx
        pre = logit + ctx
        return pre if self.act == "linear" else F.gelu(pre)

    # ------------------------------------------------------------------ forward

    def forward(self, u, attention_mask=None, return_paths=False):
        """
        u: (batch, length, d_model)
        attention_mask: (batch, length), 1 for real tokens and 0 for padding
        Returns: same shape as u
        """
        batch, length, _ = u.shape
        if attention_mask is None:
            mask = torch.ones(batch, length, dtype=u.dtype, device=u.device)
        else:
            mask = attention_mask.to(dtype=u.dtype)

        x = self._local_mix(u, mask)
        if self.route_ste:
            return self._forward_route_ste(x, mask, return_paths)
        nodes, logits = self._walk(x)
        if self.capture_paths:
            self.last_nodes, self.last_logits = nodes, logits
        self._all_logits = None  # do not keep a graph tensor on the module
        ctx = 0.0
        if self.use_context:
            ctx, states = self._context_dense(x, nodes, logits, mask)
        coef = self._coef(logits, ctx)  # (b, l, slots)
        if self.dense_walk:
            total = self.n_trees * self.n_nodes
            out = torch.zeros(batch, length, total, dtype=coef.dtype, device=coef.device).scatter(2, nodes, coef) @ self._w_out()
        else:
            out = torch.einsum("blk,blkd->bld", coef, self._w_out()[nodes])
        if self.read_state and self.use_context:
            out = out + self._read(torch.cat([v.flatten(2) for v in states], dim=-1))
        out = out * mask.unsqueeze(-1)
        if return_paths:
            return out, nodes
        return out

    @torch.no_grad()
    def forward_reference(self, u, attention_mask=None):
        """Token-at-a-time kernel with lazily decayed node states.

        Slow (Python loops) and not differentiable. It exists to pin down, and to test, the
        algorithm an inference kernel implements:

            pass 1  per token: local conv, the small dense ctx projection, the tree walk
            pass 2  left to right: read then write the node states on the path (forward context)
            pass 3  right to left: the same with the backward states
            pass 4  per token: sum the read vectors on the path

        A node's state is only touched when a token visits it. The decay it missed in between
        is applied on the next visit as exp(clock_now - clock_at_last_write), where each tree
        level keeps one running clock. No table is cleared between sequences: ``state`` below
        is a dict, which a kernel replaces with a per-node sequence stamp.
        """
        batch, length, _ = u.shape
        if attention_mask is None:
            mask = torch.ones(batch, length, dtype=u.dtype, device=u.device)
        else:
            mask = attention_mask.to(dtype=u.dtype)

        x = self._local_mix(u, mask)
        w_in, w_out = self._w_in(), self._w_out()
        if self.use_context:
            B, C_fwd, C_bwd, dt, a = self._ctx_signals(x, mask)
        out = torch.zeros_like(u)

        for b in range(batch):
            valid = [t for t in range(length) if mask[b, t] > 0]

            # pass 1: tree walk
            path = {}
            logit = {}
            for t in valid:
                ids, vals = [], []
                for tree in range(self.n_trees):
                    cur = 0
                    for level in range(self.n_levels):
                        gid = cur + tree * self.n_nodes
                        v = torch.dot(x[b, t], w_in[gid])
                        if self.node_bias is not None:
                            v = v + self.node_bias[gid]
                        ids.append(gid)
                        vals.append(v)
                        cur = 2 * cur + 1 + int(v > 0)
                path[t], logit[t] = ids, vals

            ctx = {t: [0.0] * self.n_slots for t in valid}
            found = {t: [] for t in valid}  # the state vectors token t found, per direction and slot
            if self.use_context:
                for order, C in ((valid, C_fwd),) if self.causal else ((valid, C_fwd), (valid[::-1], C_bwd)):
                    state = {}  # node id -> (state vector, clock value at last write)
                    clock = torch.zeros(self.n_groups, dtype=a.dtype, device=a.device)
                    for t in order:
                        clock = clock + a[b, t]
                        for slot, gid in enumerate(path[t]):
                            level = int(self.slot_group[slot])  # the (head, level) group, i.e. which clock
                            if gid in state:
                                h, stamp = state[gid]
                                h = h * torch.exp(clock[level] - stamp)  # lazy decay
                                ctx[t][slot] = ctx[t][slot] + torch.dot(C[b, t], h)  # read
                            else:
                                h = torch.zeros(self.d_state, dtype=a.dtype, device=a.device)
                            found[t].append(h)
                            h = h + dt[b, t, level] * (logit[t][slot] if self.write_logit else 1.0) * B[b, t]  # write
                            state[gid] = (h, clock[level].clone())

            # pass 4: output
            for t in valid:
                acc = torch.zeros(self.d_model, dtype=u.dtype, device=u.device)
                for slot, gid in enumerate(path[t]):
                    acc = acc + self._coef(logit[t][slot], ctx[t][slot]) * w_out[gid]
                if self.read_state and self.use_context:
                    acc = acc + self._read(torch.cat(found[t]))
                out[b, t] = acc
        return out

    # ---------------------------------------------------------------- utilities

    @torch.no_grad()
    def path_stats(self, nodes, attention_mask=None):
        """Fraction of each level's nodes visited by at least one (real) token.

        Hard routing gets no direct gradient, so subtrees can die. Watch this during training.
        ``nodes`` is the second return value of ``forward(..., return_paths=True)``.
        """
        if attention_mask is not None:
            nodes = nodes[attention_mask.bool()]
        nodes = nodes.reshape(-1, self.n_slots)
        used = []
        for level in range(self.n_levels):
            cols = nodes[:, self.slot_level == level]
            used.append(cols.unique().numel() / (self.n_trees * 2**level))
        return used

    def ops_per_token(self):
        """Arithmetic per token per layer for the inference kernel, as multiply-adds (or adds, if ternary).

        Counts only. Says nothing about cache misses, which is where a tree walk pays.
        """
        d, n = self.d_model, self.d_state
        ops = {
            "local_conv": self.d_conv * d,
            "walk_dots": self.n_slots * d,
            "read_adds": self.n_slots * d,
            "ctx_proj": 0,
            "state_updates": 0,
            "state_read": 0,
        }
        if self.use_context:
            ops["ctx_proj"] = ((1 + self.n_queries) * n + (self.n_groups if self.selective else 0)) * d
            # per direction, per slot: decay, read, write on a d_state vector
            ops["state_updates"] = self.n_queries * self.n_slots * 3 * n
            if self.read_state:
                ops["state_read"] = self.n_queries * self.n_slots * n * d
        ops["total"] = sum(ops.values())
        # Mamba-2 at the same width with its defaults (expand=2, d_state=128, headdim=64), for scale.
        d_inner, m2_state = 2 * d, 128
        ops["mamba2_block_for_scale"] = (
            d * (2 * d_inner + 2 * m2_state + d_inner // 64) + d_inner * d + 3 * d_inner * m2_state
        )
        return ops


class GTSMixed(nn.Module):
    """A forest of mixed depths, as two GTS mixers whose outputs are summed.

    * ``bank``: depth-0 trees. Every token visits every one, so they are dense channels in the Mamba-2 sense and
      carry all the context (in both directions unless ``causal``).
    * ``deep``: a few deep trees with no state. They hold most of the parameters and touch few of them per token.
      Their branches train with the straight-through routing gradient.

    This is the shape that gave the best speed for its quality in the autoregressive runs (see GTS.md).
    """

    def __init__(self, d_model, bank_trees=32, bank_heads=8, bank_state=16, deep_trees=4, deep_depth=6, d_conv=3,
                 causal=False, ternary=False, ternary_group=128, act_bits=None, route_ste=True, route_ste_temp=1.0,
                 layer_idx=None, device=None, dtype=None):
        super().__init__()
        common = dict(d_conv=d_conv, causal=causal, ternary=ternary, ternary_group=ternary_group, act_bits=act_bits,
                      layer_idx=layer_idx, device=device, dtype=dtype)
        self.bank = GTS(d_model, depth=0, n_trees=bank_trees, n_heads=bank_heads, d_state=bank_state, act="split", **common)
        self.deep = GTS(d_model, depth=deep_depth, n_trees=deep_trees, use_context=False, route_ste=route_ste,
                        route_ste_temp=route_ste_temp, dense_walk=True, **common)

    def forward(self, u, attention_mask=None):
        return self.bank(u, attention_mask=attention_mask) + self.deep(u, attention_mask=attention_mask)

    @torch.no_grad()
    def forward_reference(self, u, attention_mask=None):
        return self.bank.forward_reference(u, attention_mask=attention_mask) + self.deep.forward_reference(u, attention_mask=attention_mask)

    def ops_per_token(self):
        bank, deep = self.bank.ops_per_token(), self.deep.ops_per_token()
        return {"bank": bank["total"], "deep": deep["total"], "total": bank["total"] + deep["total"]}
