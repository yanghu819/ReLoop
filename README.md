# loop-dedup

Cross-loop redundancy removal for weight-tied looped (recurrent-depth) language models.

In a looped LM the same block runs `r` times per token. Because the weights are tied, the
updates it writes become nearly collinear — in our baselines **more than 99% of a loop's
update energy already lies in the span of the previous loops' updates by loop 5**. The later
loops pay full compute for a fraction of the value.

This removes it. At each loop, subtract the component of the block's output that lies in the
span of the previous updates:

```
u_t  <-  u_t - 0.5 * U (UᵀU + eps I)⁻¹ Uᵀ u_t
```

No learnable parameters, under 0.02% of the block's FLOPs at 1.3B.

## Results so far

Paired validation loss vs. a baseline with identical seed, data order and GPU count, compared
at the same step. Negative is better.

| model | loops | tokens | Δ (at training depth) |
|---|---|---|---|
| 770M | 8 | 2.5B, converged | **−0.0176** |
| 1.3B | 8 | 5.0B, converged | **−0.0145** |
| 1.3B | 16 | 1.9B, running | **−0.0247** |
| 2.1B | 8 | 1.6B, running | **−0.0109** |

The gain is larger at 16 loops than at 8, which is what the mechanism predicts: more loops,
more redundancy to remove.

**This is work in progress.** Downstream task results for this method at scale are still
running, all runs so far are under Chinchilla-scale token budgets, and we have not yet
established optimiser generality. [`docs/blog.md`](docs/blog.md) has the full picture including
the negative results, the two implementation bugs we hit, and what we have *not* shown.

## Using it

```python
from loop_dedup import LoopDedup

dedup = LoopDedup(gamma=0.5)       # window=8, ridge=1e-2 by default
history, no_grad_loops = None, num_loops - backprop_depth

for t in range(num_loops):
    u = block(h)                   # the shared block's residual output
    if t == no_grad_loops:         # first loop that carries a gradient
        history = None             # clear: only gradient-carrying loops belong in the history
    u, history = dedup(u, history, active=still_looping)
    h = h + u
```

`loop_forward` in [`loop_dedup.py`](loop_dedup.py) is a complete reference loop.

### Two things that are not optional

**Do not detach the history.** The gradient must flow through the projection exactly as the
forward pass computes it. Detaching costs about half the benefit — because the loops share
weights, a detached history makes autograd believe the parameters can rotate the current update
out of the history's span for free, when in reality the history rotates with it.

**Clear the history where the backward window starts.** Under truncated BPTT the early loops
carry no gradient and are in exactly that position. Passing `history=None` at that loop is not
a tuning knob; it is what the above requires.

**Pass `active` when sequences in a batch loop a different number of times.** Otherwise
finished sequences push empty slots into the window and evict real history — this was a real
bug in our training code, see the blog.

## Checking it

```bash
python test_loop_dedup.py
```

Verifies the implementation against a brute-force transcription of the definition (<1e-9 in
float64), that a repeated update loses exactly `gamma` of itself while an orthogonal one is
untouched, that the gradient reaches the history, and that inactive loops do not evict history.

## Files

| | |
|---|---|
| [`loop_dedup.py`](loop_dedup.py) | the method, self-contained, PyTorch only |
| [`test_loop_dedup.py`](test_loop_dedup.py) | correctness checks |
| [`docs/blog.md`](docs/blog.md) | the full write-up: diagnosis, mechanism, results, failures |
