"""Cross-loop redundancy removal for weight-tied looped (recurrent-depth) language models.

A looped LM applies one block of layers r times per token:

    h_1 = h_0 + f(h_0);  h_2 = h_1 + f(h_1);  ...  h_r = h_{r-1} + f(h_{r-1})

Because every loop shares the same weights f, successive updates u_t = f(h_{t-1}) end up
nearly collinear: in our baselines, by loop 5 more than 99% of a loop's update energy already
lies in the span of the previous loops' updates. The recurrence collapses to low-rank drift,
so extra loops buy much less than their compute.

This module removes that redundancy. At every loop, project the block's output onto the span
of the previous updates and subtract a fixed fraction gamma of that projection:

    u_t  <-  u_t - gamma * U (U^T U + eps I)^-1 U^T u_t

computed per token, in fp32, where U holds the previous updates still inside the window.

Two details matter and are not free parameters:

  * The gradient must flow through the projection exactly as the forward pass computes it.
    Detaching the history, or scaling the backward path, both cost about half the benefit.
    See `docs/blog.md`, "backward must equal forward".

  * `history_from` must be the first loop that carries a gradient. Under truncated BPTT the
    early loops run under no_grad; if their updates stay in U, autograd believes the
    parameters can rotate the current update out of span(U) for free, while in reality U
    rotates along with it at the next step. Clearing the history where the backward window
    starts removes that bias.

Usage with truncated BPTT over the last `s` of `r` loops:

    dedup = LoopDedup(gamma=0.5)
    state = None
    for t in range(r):
        u = block(h)                                  # the shared block's residual output
        if t == r - s:                                # first loop that carries a gradient
            state = None                              # <- clear history here
        u, state = dedup(u, state)
        h = h + u

The cost is negligible: with a window of 8 the solve is 8x8 per token, under 0.02% of the
block's FLOPs at 1.3B.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class _History:
    """Previous updates (newest last) and their Gram matrix, both per token."""
    updates: list            # each [*batch, d], already fp32
    gram: torch.Tensor       # [*batch, k, k]


class LoopDedup(torch.nn.Module):
    """Subtract the repeated part of each loop's update. No learnable parameters.

    Args:
        gamma: fraction of the projection to remove. 0.5 throughout our experiments;
            0 disables the module and 1 (removing all of it) is worse than 0.5 -- the
            update's own magnitude carries the convergence signal, so full removal hurts.
        window: how many previous updates to keep. Longer was better in our sweeps
            (1 < 2 < 4 < 8); 8 is where we stopped, not a tuned optimum.
            Note what this interacts with: under truncated BPTT the history is cleared at the
            start of the backward window, so with `backprop_depth` gradient-carrying loops it
            never holds more than `backprop_depth - 1` entries. At r=8, backprop_depth=4 the
            lengths are 0,1,2,3 and every window >= 3 is the same computation. The sweep above
            can therefore only have separated windows in a regime where the history actually
            grows past them -- a longer backward window, more loops, or no truncation.
        ridge: Tikhonov term, scaled by the mean Gram diagonal. The history is often
            near-collinear, which is exactly the case this module exists for, so the
            solve needs regularising. Results are flat between 1e-3 and 1e-1.
    """

    def __init__(self, gamma: float = 0.5, window: int = 8, ridge: float = 1e-2):
        super().__init__()
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.gamma, self.window, self.ridge = gamma, window, ridge

    def forward(self, update: torch.Tensor, history: Optional[_History] = None,
                active: Optional[torch.Tensor] = None):
        """Deduplicate one loop's update.

        Args:
            update: [*batch, d] the shared block's residual output for this loop.
            history: state returned by the previous call, or None to start fresh.
                Pass None at the loop where the backward window begins (see module docstring).
            active: optional [*batch] mask, 1 where this sequence is still looping.
                An inactive position never evicts its own history. Required when different
                sequences in a batch run a different number of loops -- without it, finished
                sequences push empty slots into the window and evict real history.
                Two things to know. The window is one shared list, so its length is common to
                the batch: an inactive position stores a zero row rather than nothing, which
                the solve gives zero weight and the ridge now ignores. And the returned update
                is *not* masked -- a caller that adds it back for finished sequences must mask
                it itself.

        Returns:
            (deduplicated update, new history)
        """
        if self.gamma == 0.0:
            return update, history

        dtype = update.dtype
        # The projection is done in at least fp32: the history is near-collinear, so the solve
        # is ill-conditioned and bf16 loses it. A float64 input is kept at float64.
        u = update if update.dtype == torch.float64 else update.float()
        if active is None:
            active = torch.ones(u.shape[:-1], device=u.device, dtype=u.dtype)
        m = active.to(dtype=u.dtype)

        U = list(history.updates) if history is not None else []
        if U:
            G = history.gram
            k = len(U)
            # b_j = <u, U_j>; solve (G + ridge) c = b for the least-squares coefficients.
            b = torch.stack([(h * u).sum(-1) for h in U], -1)
            eye = torch.eye(k, device=u.device, dtype=u.dtype)
            # Mean over the *occupied* slots only. A finished sequence holds zero rows (see below),
            # and averaging over those would shrink the ridge by live/k and make the solve
            # ill-conditioned exactly where the history is most degenerate.
            diag = G.diagonal(dim1=-2, dim2=-1)
            live = (diag > 0).to(diag.dtype).sum(-1).clamp_min(1.0)
            scale = (diag.sum(-1) / live)[..., None, None].clamp_min(1e-6)
            c = torch.linalg.solve(G + self.ridge * scale * eye, b.unsqueeze(-1)).squeeze(-1)
            proj = U[0] * c[..., 0:1]
            for j in range(1, k):
                proj = proj + U[j] * c[..., j:j + 1]
            u = u - self.gamma * proj

        entry = u * m[..., None]
        on = m[..., None] > 0
        if len(U) < self.window:
            # Inactive positions contribute a zero row, which the solve gives zero weight.
            U = U + [entry]
        else:
            # Shift the window only where the sequence is still looping; a finished sequence
            # keeps the history it had, instead of having it evicted by empty loops.
            U = [torch.where(on, U[j + 1], U[j]) for j in range(len(U) - 1)] + \
                [torch.where(on, entry, U[-1])]

        # Recompute the Gram from the window. k <= window (8), so this is negligible, and it
        # keeps the state unambiguous after the masked shift above.
        G = torch.stack([torch.stack([(a * b_).sum(-1) for b_ in U], -1) for a in U], -2)

        return u.to(dtype), _History(updates=U, gram=G)


def loop_forward(block, h, num_loops: int, backprop_depth: int, dedup: Optional[LoopDedup] = None):
    """Reference loop that clears the history where the backward window starts.

    The first `num_loops - backprop_depth` loops run under no_grad; the history is reset at
    the boundary so only gradient-carrying updates are ever projected against.
    """
    history, no_grad_loops = None, max(num_loops - backprop_depth, 0)
    for t in range(num_loops):
        # nullcontext, not enable_grad: under `with torch.no_grad()` at evaluation time
        # enable_grad would switch autograd back on and build a graph for the last loops.
        with torch.no_grad() if t < no_grad_loops else nullcontext():
            u = block(h)
            if t == no_grad_loops:
                history = None
            if dedup is not None:
                u, history = dedup(u, history)
            h = h + u
    return h
