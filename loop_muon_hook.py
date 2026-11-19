"""Apply loop_muon to the tied matrices of a real looped transformer, without extra activations.

What this changes: for every tied matrix inside the looped core, the weight gradient is formed from
the whitened loop inputs instead of the raw ones. The forward pass is untouched, so a run with this
enabled produces exactly the same activations as the baseline and differs only through the
optimizer -- which is the whole point of comparing it against the forward-pass method.

Why it is a custom autograd.Function and not a hook: a module backward hook cannot stop autograd
from having already accumulated `grad_y^T x` into `W.grad`. The gradient has to be formed here.

Memory: the s loop inputs are kept by reference, not copied -- autograd is holding them anyway for
the backward. The only new state per matrix group is the loop-mean direction, the s components
along it and the s x s operator, and each whitened input is rebuilt on demand and freed once that
loop's weight gradient is formed.

Grouping: inside one layer, q/k/v all read the same pre-attention input and the MLP's first matrix
reads the same pre-MLP input, so one operator is built per (layer, attn|mlp) rather than per
matrix -- 2 solves per layer instead of 7.
"""

import os
import types

import torch
import torch.nn.functional as F

from loop_muon import loop_apply, loop_operator


class _Context:
    """Per-microbatch capture state. One instance, reset at the start of every forward."""

    def __init__(self):
        self.gamma = float(os.environ.get("LOOP_MUON_GAMMA", "0"))
        self.eps = float(os.environ.get("LOOP_MUON_RIDGE", "1e-2"))
        self.members = {}      # group -> how many matrices read that group's input
        self.reset()

    def reset(self):
        self.xs = {}           # group -> {loop index: input tensor}
        self.op = {}           # group -> (e, coef, M)
        self.pending = {}      # group -> outstanding backward calls
        self.loop = None       # current gradient-carrying loop index, None = do not capture
        self.n_loops = 0

    # -- forward side -------------------------------------------------------------------------
    def capture(self, group, x):
        if self.loop is None:
            return
        self.xs.setdefault(group, {})[self.loop] = x

    # -- backward side ------------------------------------------------------------------------
    def whitened(self, group, t):
        """The whitened input of loop `t`, building the group's operator on first use."""
        store = self.xs.get(group)
        if store is None or t not in store:
            return None
        xs = [store[i] for i in sorted(store)]
        if len(xs) < 2:
            return None                                   # one live loop is an untied situation
        pos = sorted(store).index(t)
        if group not in self.op:
            self.op[group] = loop_operator(xs, gamma=self.gamma, eps=self.eps)
            self.pending[group] = self.members.get(group, 1) * len(xs)
        e, coef, M = self.op[group]
        return loop_apply(xs, pos, e, coef, M)

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
        grad_x = grad_y @ weight.to(dtype=grad_y.dtype)
        xt = CTX.whitened(ctx.group, ctx.loop) if ctx.loop is not None else None
        if xt is None:
            # Not a captured loop: fall back to the ordinary weight gradient.
            store = CTX.xs.get(ctx.group, {})
            xt = store.get(ctx.loop)
            if xt is None:
                return grad_x, None, None
        gw = grad_y.reshape(-1, grad_y.shape[-1]).t().float() @ xt.reshape(-1, xt.shape[-1]).float()
        CTX.release(ctx.group)
        return grad_x, gw.to(weight.dtype), None


def _hooked_forward(self, x):
    if CTX.gamma > 0.0 and CTX.loop is not None and torch.is_grad_enabled():
        return _WhitenedMatmul.apply(x, self.weight, self._loop_muon_group)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def wrap_core(model, first_layer: int, last_layer: int) -> int:
    """Route the tied core's residual-stream readers through the operator.

    `first_layer` and `last_layer` bound the physical layers that are replayed every loop.
    Returns how many matrices were wrapped. Deliberately skips the output matrices (attention's
    projection and the MLP's second matrix): their inputs are several times wider and heavy
    tailed, so they cost the most and behave the worst. Add them as an ablation, not by default.
    """
    wrapped = 0
    CTX.members.clear()
    for layer in range(first_layer, last_layer + 1):
        block = model.transformer.h[layer]
        groups = [(f"{layer}:attn", [getattr(block.attn, n, None) for n in ("c_q", "c_k", "c_v")]),
                  (f"{layer}:mlp", [getattr(block.mlp, "c_fc", None)])]
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
