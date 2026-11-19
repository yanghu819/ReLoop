# Looped language models repeat themselves. You can just subtract it.

*Work in progress. The numbers below are what we have measured so far, with the caveats
we actually have — including the ones that went against us.*

## The setup

A **looped** (or recurrent-depth) language model applies one block of layers several times
per token instead of stacking more distinct layers:

```
h_1 = h_0 + f(h_0)
h_2 = h_1 + f(h_1)
...
h_r = h_{r-1} + f(h_{r-1})
```

The same `f` every time — the weights are tied across loops. The appeal is that you can spend
more compute at test time by looping more, without more parameters. Models like Ouro are built
this way, and a run's loop count `r` is sampled during training so the model learns to work at
several depths.

## The problem: by loop 5, there is almost nothing new left

Take a trained baseline and ask a simple question: how much of loop `t`'s update
`u_t = f(h_{t-1})` is actually new, versus already inside the span of what the earlier loops
wrote? Project `u_t` onto `span(u_1 ... u_{t-1})` and measure the energy ratio.

In a 140M baseline, per token:

| loop | fraction of the update already in the previous loops' span |
|---|---|
| 2 | 82% |
| 3 | 94% |
| 4 | 98% |
| 5+ | **>99%** |

By the fifth loop the block is essentially re-writing a direction it has already written. The
cosine between adjacent loop updates climbs from 0.84 to 0.99 over training. The recurrence,
which is supposed to be buying depth, has collapsed onto a line.

This is not surprising in hindsight — it is what weight tying encourages. `f` is a single
function; applied repeatedly to a slowly-moving input, it returns a slowly-moving output. But
it means the later loops are mostly paying full price for a fraction of the value.

## The method: remove the repeated half

One sentence: **at every loop, subtract from the block's output the component that lies in the
span of the previous updates inside the current backward window.**

```
u_t  <-  u_t - gamma * U (U^T U + eps I)^-1 U^T u_t
```

per token, in fp32, with `U` the previous updates in a window of 8, `gamma = 0.5` fixed, and a
ridge term scaled by the mean Gram diagonal (the history is near-collinear by construction, so
the solve needs it). The implementation is in [`loop_dedup.py`](../loop_dedup.py); it is about
60 lines of real code and has no learnable parameters.

Two things about this are easy to get wrong, and both cost roughly half the benefit.

### The gradient must flow through the projection

It is tempting to `detach()` the history — it is "just" a reference set, and detaching is
cheaper and more stable-looking. We swept the strength of the backward path through the
projection independently of the forward one. The loss is minimised exactly where
**backward equals forward**. Detaching costs about `0.016` nats; pushing the backward path
harder than the forward one is also worse.

The reason is weight tying. Suppose two loops are nearly collinear, `u_1 = v`, `u_2 = v`. Since
`P_v v = v` identically for every parameter setting, the true map is `u_2 = (1-gamma) v`, with
derivative `(1-gamma) dv` in every direction. Autograd reconstructs that only if it
differentiates through `U` as well as through `u_t`. If `U` is detached, autograd believes the
parameters can rotate `u_t` out of `span(U)` for free — but `U` is produced by the *same
weights*, so it rotates along with it at the next step. The gradient is biased, and the bias
grows with depth.

### The history must only contain loops that carry a gradient

Training uses truncated BPTT: only the last `s` loops (we use 4) get gradients; the earlier
ones run under `no_grad`. Those early updates are in exactly the position described above —
present in `U`, carrying no gradient. So we clear the history at the loop where the backward
window starts. We call this `hbptt`.

This is not a hyperparameter. It is what "backward equals forward" *means* when some loops are
detached by the training procedure rather than by choice.

It also has a consequence we had not drawn until a review pointed it out. Clearing the history at
the boundary caps how long the history can get: with `s` gradient-carrying loops it never holds
more than `s-1` entries. At `r=8, s=4` the lengths per loop are exactly `0,1,2,3` — so every
window of 3 or more is the *same computation*, and the window sweep we quote (`1 < 2 < 4 < 8`)
cannot have been separating anything in the `r=8` runs. Whatever it measured came from a regime
where the history actually grows: the no-grad prefix, evaluation, or `r=16`. The headline `r=8`
numbers should be read as window-3 results that happen to be configured with `window=8`.

## What it buys

Paired validation loss against a baseline trained with the same seed, same data order, same
GPU count, compared at the same step. Negative is better. `Δr` is measured evaluating at `r`
loops.

| model | loops | tokens | Δ8 | Δ16 | Δ4 |
|---|---|---|---|---|---|
| 770M | 8 | 2.5B (converged) | **−0.0176** | −0.0176 | −0.0144 |
| 1.3B | 8 | 5.0B (converged) | **−0.0145** | −0.0153 | −0.0112 |
| 1.3B | 16 | 1.9B (in progress) | — | **−0.0247** | noisy, see below |
| 2.1B | 8 | 1.6B (in progress) | **−0.0109** | −0.0100 | −0.0090 |

A second training seed at 1.3B tracks the first closely: at 1.9B tokens the two seeds give
`−0.0115` and `−0.0112`; at 2.2B, `−0.0111` and `−0.0100`. Early in training the seeds differ
by more (`0.0036` at the first eval point), so the early part of these curves should not be
over-read.

Two things worth noting in that table. The gain is **larger at 16 loops than at 8** — the
deeper the recurrence, the more redundancy there is to remove, which is what the mechanism
predicts. And the gain holds at `r=4` evaluation too, so for `r=8`-trained models this is not
buying depth-16 performance by sacrificing shallow performance.

### Cost

With a window of 8 the solve is 8x8 per token: under 0.02% of the block's FLOPs at 1.3B, and no
extra parameters. The wall-clock cost is larger than the FLOP cost because the fp32 solve
breaks out of the compiled region; an unrolled Gram-Schmidt for the small `k` that `hbptt`
actually produces would reduce that, which we have not yet done.

## What did not work

Reporting these is the point of writing this down.

**Normalising the update.** Four variants, all neutral or worse: rescaling the deduplicated
update back to its original norm (neutral); unit-normalising the history so early large updates
do not dominate (`+0.0003`, noise); channel whitening before the projection (no gain at 140M or
370M); and full unit-normalisation of the update direction, which is *much* worse and gets
worse with depth (`+0.09` at 4 loops, `+0.16` at 16). The magnitude of a loop's update is
itself a signal — successive loops naturally take smaller steps as the iteration settles, and
normalising that away forces every loop to take a full step. Subtracting a direction is fine;
rescaling is not.

**Removing all of the projection.** `gamma=1` is worse than `gamma=0.5`. Partial removal beats
full removal, consistently.

**Projecting against the wrong base.** The update must be measured from the state *after*
whatever per-loop injection the architecture does, not before. Using the raw previous state
instead loses the entire effect.

**A learned gate instead of a projection.** An input-dependent elementwise gate was worse than
the baseline.

## What we have not shown

- **Downstream.** We have converged downstream (CORE) numbers for an *earlier, more complex*
  variant of this method at 140M/370M/770M (`+2.4%`, `+6.7%`, `+2.6%` relative). We do **not**
  yet have them for the simple method described here at scale — those evaluations are running.
  One 370M measurement at 16 loops came out *negative* (`−3.5%` relative), though decomposing
  it showed 79% of the deficit came from a single task (`bigbench_dyck_languages`), so we treat
  it as unresolved rather than as evidence either way.
- **Tokens per parameter.** Every run above is between 3.2 and 3.9 tokens/parameter, well under
  Chinchilla. The honest version of "the gain does not decay" is: it decays over the first
  ~40% of training and then recovers during learning-rate cooldown. Longer runs are in flight.
- **Optimiser generality.** All of the above uses Muon (with AdamW on the non-matrix
  parameters). Our one AdamW comparison was run at a learning rate where both arms were badly
  under-trained and the two seeds disagreed in sign, so it tells us nothing yet.
- **Shallow evaluation of deep-trained models.** Models trained at 16 loops are much worse when
  evaluated at 1 loop, for the baseline too (a 16-loop-trained baseline is `+1.15` nats worse
  at 1 loop than at 16). Our per-depth validation numbers at `r=4` also swing by `±0.07`
  between adjacent checkpoints, about 30x the jitter of the `r=16` column — we traced this to
  each evaluation depth consuming different batches from a stateful validation loader, so that
  column is not currently trustworthy and is excluded from claims.

## Things that bit us

Two implementation bugs, both found by independent review of the code against its own
docstrings, both worth repeating because they are easy to reproduce:

1. **Scaling the wrong quantity.** A flag meant to "scale the core block's output by the
   decay" was implemented as `x_new = x_new * decay`, where `x_new` was the *state plus* the
   output. So the quantity fed to the projection picked up a `(decay − 1) * state` term that
   grew as the decay became anisotropic over training. The comment said one thing, the code did
   another, and nobody diffed them for weeks. Lesson: when a flag transforms an intermediate
   quantity, write a float64 assertion that the quantity equals its analytic form.

2. **Padding the history with zeros.** Sequences in a batch loop a different number of times.
   When a sequence finished, the code still appended its (masked, therefore zero) update to the
   shared history every remaining step — and once the window was full, those zeros evicted real
   history. At 16 loops nearly half the batch entered the gradient-carrying phase with an empty
   history. The reference implementation here takes an `active` mask for exactly this reason.

Three more, in the reference implementation in this repo, found the same way — two models reading
the code against its own docstrings, independently, and agreeing:

3. **`enable_grad` at evaluation.** `loop_forward` selected `torch.enable_grad()` for the loops
   inside the backward window. That *overrides* a caller's `torch.no_grad()`, so evaluating a
   model built this way silently allocated an autograd graph for the last `s` loops. It is
   `nullcontext()` now, and a test evaluates under `no_grad` and asserts nothing requires grad.
4. **Empty slots shrank the ridge.** The window is one shared list, so a finished sequence has to
   store a zero row rather than nothing. The solve already gave those rows zero weight, but the
   ridge was scaled by the mean Gram diagonal *including* them, so a dead slot quietly shrank the
   regularisation for the live history — in exactly the near-collinear case the ridge is there
   for. The mean is now taken over occupied slots only.
5. **Tests that asserted less than their names claimed.** `test_loop_forward_resets_at_the_backward_boundary`
   only checked that a gradient existed; deleting the reset still passed it. Nothing pinned
   "backward equals forward" either, though it is easy to pin: detaching the history moves the
   weight gradient by 89%.

And one methodology bug: comparing a run on 8 GPUs against a baseline on 4. The data loader
shards the file list by world size, so the two runs read different data and see a different
number of tokens per step; subtracting their losses at the same step produced a 0.43 nat
phantom signal. We now refuse such a submission in the launcher rather than relying on
remembering.

## Status

Open research, not a finished result. The claim we think is currently supported is narrow:
*in weight-tied looped LMs, successive loop updates become almost entirely redundant, and
subtracting half of that redundancy — with the gradient flowing through the subtraction, and
only over loops that carry a gradient — gives a consistent validation-loss improvement that
survives from 770M to 2.1B parameters and grows with loop count.* Whether it moves downstream
task accuracy at scale, and whether it holds past Chinchilla-scale token budgets, is what we
are measuring now.
