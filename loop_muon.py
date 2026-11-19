"""Loop-aware preconditioning for the shared weights of a weight-tied looped model.

`loop_dedup` removes cross-loop redundancy in the forward pass. This is the same idea applied to
the gradient instead, and it is the reason this repository is called loop-muon.

A tied matrix W is used once per loop, so its gradient is a sum of per-loop terms that are already
in factored form:

    grad W = sum_t  d_t h_{t-1}^T          h_{t-1} = the loop's input to W
                                           d_t     = the loop's output-side backprop signal

When successive loops write nearly the same update, the h_{t-1} are nearly identical across t, so
every term of that sum points the same way and the gradient collapses onto the directions the
earliest loop already covered. Muon then orthogonalises that collapsed sum, which flattens its
spectrum but cannot recover a direction the sum never contained.

The fix acts on the factors before they are contracted, per token. For token n, collect its loop
inputs as columns:

    H = [h_0, ..., h_{s-1}]  in R^{d x s}

split it in FEATURE space along the loop-mean direction, and whiten only what is left:

    e      = hbar / ||hbar||            hbar = H 1 / s
    H_par  = e (e^T H)                  H_perp = H - H_par
    M      = (1 - gamma) I + gamma (Ahat + eps I)^{-1/2}      Ahat = H_perp^T H_perp / tau
    H~     = H_par + H_perp M

and form grad W from H~ instead of H. Three properties make this the right split, and
`test_loop_muon.py` checks all three:

  * `H_par^T H_perp = 0` and `H_perp 1 = 0` hold identically, so H_perp is automatically the
    loop-centred part and span(H_perp) is the span of the centred updates -- the object the >99%
    collapse statistic actually measures. Whitening H itself instead would act mostly on the shared
    content direction, which is the "projecting against the wrong base" failure in docs/blog.md.
  * M is a function of A_perp, so the two commute, H^T H~ is symmetric PSD, and
    <grad~ W, grad W> >= 0 for EVERY TOKEN and every backprop signal. Splitting along the loop
    index instead of in feature space loses this: with s=2, H=[1,2], D=[3,-1] it produces
    grad = +1 but grad~ = -3.
    Read that guarantee narrowly. It is per token and it does NOT survive summation: two tokens
    that are each aligned can sum to a descent direction that is not. With grad_1 = [10,1] sent to
    [5,1] and grad_2 = [-9,0] left alone, both tokens align (51 and 81) while the sums give -3.
    So this bounds what the operator does to one token's contribution; it is not a claim that the
    optimizer step can never oppose the batch gradient.
  * At s=1 -- an ordinary untied layer -- H_perp is exactly zero, so H~ = H bit for bit. The method
    is inert unless weights are tied AND activations repeat across loops, which is what makes it a
    looped-model optimiser rather than a generic gradient trick.

Optimizer state stays 1x: the per-loop structure is consumed when the gradient is formed, and Muon
sees a single matrix exactly as before.
"""

from typing import Optional

import torch


def loop_whiten(H: torch.Tensor, gamma: float = 0.5, eps: float = 1e-2,
                active: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Whiten the loop-varying part of per-token loop inputs.

    Args:
        H: [*batch, d, s] -- for each token, the s loop inputs of one tied matrix as columns.
        gamma: how much of the whitening to apply. 0 returns H unchanged, 1 whitens fully.
        eps: ridge on the normalised loop Gram. The Gram is near-singular exactly when the loops
            repeat, which is the case this exists for, so it is not optional.
        active: optional [*batch, s] mask, 0 for loops this token did not run. Masked columns are
            held out of the mean, the Gram and the normalisation, and come back unchanged.

    Returns:
        H~ of the same shape and dtype. The computation runs in at least fp32.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if H.ndim < 2:
        raise ValueError(f"H must be [*batch, d, s], got shape {tuple(H.shape)}")
    dtype, s = H.dtype, H.shape[-1]
    if gamma == 0.0 or s == 1:
        # s == 1 is an untied layer: H_perp is zero, so this would be the identity anyway.
        return H

    work = H if H.dtype == torch.float64 else H.float()
    if active is None:
        m = work.new_ones(work.shape[:-2] + (s,))
    else:
        m = active.to(work.dtype)
    live = m.sum(-1).clamp_min(1.0)                               # [*batch]
    mcol = m.unsqueeze(-2)                                        # [*batch, 1, s]

    # Feature-space split along the loop-mean direction.
    hbar = (work * mcol).sum(-1) / live.unsqueeze(-1)             # [*batch, d]
    # e must have norm exactly 1 or exactly 0, or e e^T is not a projector: then H_par and H_perp
    # stop being orthogonal and the descent guarantee goes with them. Clamping the norm instead
    # silently produces ||e|| < 1 for a token whose loop mean is near zero.
    floor = torch.finfo(work.dtype).tiny
    nrm = hbar.norm(dim=-1, keepdim=True)
    e = torch.where(nrm > floor, hbar / nrm.clamp_min(floor), torch.zeros_like(hbar))
    coef = (e.unsqueeze(-2) @ work).squeeze(-2)                   # [*batch, s] = e^T H
    par = e.unsqueeze(-1) * coef.unsqueeze(-2)                    # [*batch, d, s]
    perp = (work - par) * mcol                                    # masked loops contribute nothing

    # Normalised loop Gram of the centred part, then its damped inverse square root.
    A = perp.transpose(-1, -2) @ perp                             # [*batch, s, s]
    diag = A.diagonal(dim1=-2, dim2=-1)
    tau = diag.sum(-1) / (live - 1.0).clamp_min(1.0)
    # Normalising by tau is what makes the operator scale free, so tau may only be guarded against
    # true zero. An absolute floor breaks exactly that: with activations of size 1e-5 it sent
    # Ahat to zero and M to ((1-gamma) + gamma/sqrt(eps)) I, amplifying the update 4x instead of
    # leaving it alone.
    Ahat = A / tau.clamp_min(floor)[..., None, None]
    eye = torch.eye(s, device=work.device, dtype=work.dtype)
    # Symmetric PSD, so eigh is exact and cheap at s <= 8.
    evals, evecs = torch.linalg.eigh(Ahat + eps * eye)
    inv_sqrt = evecs @ torch.diag_embed(evals.clamp_min(eps).rsqrt()) @ evecs.transpose(-1, -2)
    M = (1.0 - gamma) * eye + gamma * inv_sqrt
    # Nothing loop-varying to whiten (one live loop, or identical loops): fall back to the identity
    # rather than to whatever the degenerate solve produced.
    settled = (tau <= floor) | (live <= 1.0)
    M = torch.where(settled[..., None, None], eye.expand_as(M), M)

    out = par + perp @ M
    if active is not None:
        out = torch.where(mcol > 0, out, work)                    # untouched where the loop is dead
    out = torch.where((live <= 1.0)[..., None, None], work, out)  # one live loop: pass through
    return out.to(dtype)


def loop_gradient(D: torch.Tensor, H: torch.Tensor, gamma: float = 0.5, eps: float = 1e-2,
                  active: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The preconditioned gradient of one tied matrix, summed over loops and tokens.

    Args:
        D: [*batch, out_dim, s] output-side backprop signal per loop.
        H: [*batch, in_dim,  s] input activations per loop.
    Returns:
        [out_dim, in_dim] in fp32, to be handed to Muon as an ordinary gradient would be.

    The contraction is done in fp32 on purpose. H_perp is routinely a percent or two of H, so
    adding H_par to H_perp M and rounding the sum to bf16 can lose most of the part this method
    exists to produce.
    """
    Ht = loop_whiten(H, gamma=gamma, eps=eps, active=active).float()
    Dw = D.float()
    if active is not None:
        Dw = Dw * active.to(Dw.dtype).unsqueeze(-2)               # a dead loop contributes nothing
    per_token = Dw @ Ht.transpose(-1, -2)                         # [*batch, out_dim, in_dim]
    return per_token.flatten(0, -3).sum(0) if per_token.ndim > 2 else per_token
