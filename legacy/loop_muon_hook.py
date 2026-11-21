"""Apply loop_muon to the tied matrices of a real looped transformer, without extra activations.

What this changes: for every tied matrix inside the looped core, the weight gradient is formed from
the whitened loop inputs instead of the raw ones. The forward pass is untouched, so a run with this
enabled produces exactly the same activations as the baseline and differs only through the
optimizer -- which is the whole point of comparing it against the forward-pass method.

Why it is a custom autograd.Function and not a hook: a module backward hook cannot stop autograd
from having already accumulated `grad_y^T x` into `W.grad`. The gradient has to be formed here.

Memory and precision: the s loop inputs are kept by detached reference, not copied -- autograd
would hold them anyway for the backward. Instead of storing the d-wide loop-mean vector `e`
(~100 MB per group at N=16k, d=1536), each group stores only the compact [*batch, s, s] delta
operator `C` (~1 MB per group, ~64 MB across a 31-layer core). Because x~_t = x_t + delta_x_t,
the backward contracts `d_t^T x_t + d_t^T delta_x_t` as two native BF16 Tensor Core GEMMs summed in
fp32: full BF16 exponent resolution stays on the small loop-varying correction delta_x_t instead of
being swamped by the shared content component, while avoiding slow fp32 GEMMs and 500 MB fp32
copies of `grad_y`.

Grouping: inside one layer, q/k/v all read the same pre-attention input and the MLP's first matrix
reads the same pre-MLP input, so one operator is built per (layer, attn|mlp) rather than per
matrix -- 2 solves per layer instead of 7.
"""

import os
import types

import torch
import torch.nn.functional as F

try:
    from .loop_muon import loop_delta_apply, loop_delta_operator
except ImportError:
    from loop_muon import loop_delta_apply, loop_delta_operator


class _Context:
    """Per-microbatch capture state. One instance, reset at the start of every forward."""

    def __init__(self):
        self.gamma = float(os.environ.get("LOOP_MUON_GAMMA", "0"))
        self.eps = float(os.environ.get("LOOP_MUON_RIDGE", "1e-2"))
        self.members = {}      # group -> how many matrices read that group's input
        self.reset()

    def reset(self):
        self.xs = {}           # group -> {loop index: detached input tensor}
        self.op = {}           # group -> (sorted_keys, C or None)
        self.pending = {}      # group -> outstanding backward calls
        self.loop = None       # current gradient-carrying loop index, None = do not capture
        self.n_loops = 0

    # -- forward side -------------------------------------------------------------------------
    def capture(self, group, x):
        if self.loop is None:
            return
        self.xs.setdefault(group, {})[self.loop] = x.detach()

    # -- backward side ------------------------------------------------------------------------
    def raw_and_delta(self, group, t):
        """Return (x_t, delta_x_t) for loop `t`, building the compact operator on first use."""
        store = self.xs.get(group)
        if store is None or t not in store:
            return None, None
        if group not in self.pending:
            keys = sorted(store)
            self.pending[group] = self.members.get(group, 1) * len(keys)
            if len(keys) < 2:
                self.op[group] = (keys, None)             # one live loop is an untied situation
            else:
                xs = [store[i] for i in keys]
                C = loop_delta_operator(xs, gamma=self.gamma, eps=self.eps)
                self.op[group] = (keys, C)
        keys, C = self.op[group]
        x_raw = store[t]
        if C is None:
            return x_raw, None
        xs = [store[i] for i in keys]
        pos = keys.index(t)
        return x_raw, loop_delta_apply(xs, pos, C)

    def release(self, group):
        """Drop a group's buffers once every matrix in it has had every loop's backward."""
        if group not in self.pending:
            return
        self.pending[group] -= 1
        if self.pending[group] <= 0:
            self.xs.pop(group, None)
            self.op.pop(group, None)
            self.pending.pop(group, None)


CTX = _Context()


def enabled() -> bool:
    return CTX.gamma > 0.0


def begin(n_grad_loops: int):
    """Start a new forward. Called once per microbatch, before the loops run."""
    CTX.reset()
    CTX.n_loops = n_grad_loops


def set_loop(t):
    """Index of the gradient-carrying loop about to run, or None outside the backward window."""
    CTX.loop = t


class _WhitenedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group):
        ctx.group = group
        ctx.loop = CTX.loop
        ctx.save_for_backward(weight)
        CTX.capture(group, x)
        return F.linear(x, weight.to(dtype=x.dtype))

    @staticmethod
    def backward(ctx, grad_y):
        (weight,) = ctx.saved_tensors
        grad_x = (grad_y @ weight.to(dtype=grad_y.dtype)) if ctx.needs_input_grad[0] else None
        if ctx.loop is None or not ctx.needs_input_grad[1]:
            if ctx.loop is not None:
                CTX.release(ctx.group)
            return grad_x, None, None
        x_raw, dx_perp = CTX.raw_and_delta(ctx.group, ctx.loop)
        if x_raw is None:
            return grad_x, None, None
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x_raw.reshape(-1, x_raw.shape[-1])
        if grad_y.dtype in (torch.float32, torch.float64):
            if dx_perp is None:
                gw = gy.t() @ xr.to(gy.dtype)
            else:
                xt = xr.to(dx_perp.dtype) + dx_perp.reshape(-1, dx_perp.shape[-1])
                gw = gy.to(xt.dtype).t() @ xt
        else:
            # Split BF16/FP16 Tensor Core GEMMs: baseline raw gradient + full-exponent perp delta.
            gw = (gy.t() @ xr.to(gy.dtype)).float()
            if dx_perp is not None:
                gw = gw + (gy.t() @ dx_perp.reshape(-1, dx_perp.shape[-1]).to(gy.dtype)).float()
        CTX.release(ctx.group)
        return grad_x, gw.to(weight.dtype), None


def _hooked_forward(self, x):
    if CTX.gamma > 0.0 and CTX.loop is not None and torch.is_grad_enabled() and self.weight.requires_grad:
        return _WhitenedMatmul.apply(x, self.weight, self._loop_muon_group)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def wrap_core(model, first_layer: int, last_layer: int, include_out: bool = False) -> int:
    """Route the tied core's residual-stream readers through the operator.

    `first_layer` and `last_layer` bound the physical layers that are replayed every loop.
    Returns how many matrices were wrapped. By default skips the output matrices (attention's
    projection and the MLP's second matrix): their inputs are wider and heavy-tailed, so pass
    `include_out=True` only as an explicit ablation.
    """
    wrapped = 0
    CTX.members.clear()
    for layer in range(first_layer, last_layer + 1):
        block = model.transformer.h[layer]
        groups = [
            (f"{layer}:attn", [getattr(block.attn, n, None) for n in ("c_q", "c_k", "c_v")]),
            (f"{layer}:mlp", [getattr(block.mlp, "c_fc", None)]),
        ]
        if include_out:
            groups.extend([
                (f"{layer}:attn_out", [getattr(block.attn, "c_proj", None)]),
                (f"{layer}:mlp_out", [getattr(block.mlp, "c_proj", None)]),
            ])
        for name, mods in groups:
            mods = [m for m in mods if m is not None]
            if not mods:
                continue
            CTX.members[name] = len(mods)
            for m in mods:
                m._loop_muon_group = name
                m.forward = types.MethodType(_hooked_forward, m)
                wrapped += 1
    return wrapped
