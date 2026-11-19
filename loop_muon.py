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

import math
from typing import Optional

import torch


def _inv_sqrt_perp_minus_eye(Ahat: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute Q ((Q^T Ahat Q + eps I)^{-1/2} - I) Q^T on 1^perp, where Ahat 1 = 0.

    Because H_perp 1 = 0 identically, any component of M along 1 annihilates H_perp. Setting the
    eigenvalue of M on 1 to 1 (so (M - I) 1 = 0) leaves H_perp M exact while keeping M - I free of
    the eps^{-1/2} spike along 1 and letting the compact delta operator cancel ||hbar|| analytically.

    Uses a branchless, cancellation-free closed form for s in {2, 3, 4} (Cardano eigenvalues +
    Newton/Cayley-Hamilton divided differences with exact cancellation of (u_i - u_j)), avoiding
    batched cuSOLVER calls on GPU. Falls back to eigh projected onto 1^perp for s > 4.
    """
    s = Ahat.shape[-1]
    dtype, device = Ahat.dtype, Ahat.device
    if s == 2:
        I2 = torch.eye(2, device=device, dtype=dtype)
        one = torch.full((2, 2), 0.5, device=device, dtype=dtype)
        return ((1.0 + eps) ** -0.5 - 1.0) * (I2 - one).expand_as(Ahat)
    if s == 3:
        Q = torch.tensor([[1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
                          [-1.0 / math.sqrt(2.0), 1.0 / math.sqrt(6.0)],
                          [0.0, -2.0 / math.sqrt(6.0)]], device=device, dtype=dtype)
        B = Q.T @ Ahat @ Q                                        # [*batch, 2, 2]
        a, b, c = B[..., 0, 0], B[..., 1, 1], B[..., 0, 1]
        m = 0.5 * (a + b)
        delta = torch.sqrt(((a - b) * 0.5).square() + c.square())
        l1 = (m + delta).clamp_min(0.0)
        l2 = (m - delta).clamp_min(0.0)
        u1 = torch.sqrt(l1 + eps)
        u2 = torch.sqrt(l2 + eps)
        d12 = -1.0 / (u1 * u2 * (u1 + u2))
        I2 = torch.eye(2, device=device, dtype=dtype)
        fB = (1.0 / u1)[..., None, None] * I2 + d12[..., None, None] * (B - l1[..., None, None] * I2)
        return Q @ (fB - I2) @ Q.T
    if s == 4:
        Q = 0.5 * torch.tensor([[1.0, 1.0, 1.0],
                                [1.0, -1.0, -1.0],
                                [-1.0, 1.0, -1.0],
                                [-1.0, -1.0, 1.0]], device=device, dtype=dtype)
        B = Q.T @ Ahat @ Q                                        # [*batch, 3, 3]
        b00, b11, b22 = B[..., 0, 0], B[..., 1, 1], B[..., 2, 2]
        b01, b02, b12 = B[..., 0, 1], B[..., 0, 2], B[..., 1, 2]
        q = (b00 + b11 + b22) / 3.0
        c00, c11, c22 = b00 - q, b11 - q, b22 - q
        p2 = (c00 * c00 + c11 * c11 + c22 * c22 + 2.0 * (b01 * b01 + b02 * b02 + b12 * b12)) / 6.0
        p = torch.sqrt(p2.clamp_min(0.0))
        floor = torch.finfo(dtype).tiny
        inv_p = torch.where(p > floor, 1.0 / p.clamp_min(floor), torch.zeros_like(p))
        k00, k11, k22 = c00 * inv_p, c11 * inv_p, c22 * inv_p
        k01, k02, k12 = b01 * inv_p, b02 * inv_p, b12 * inv_p
        det = (k00 * (k11 * k22 - k12 * k12)
               - k01 * (k01 * k22 - k12 * k02)
               + k02 * (k01 * k12 - k11 * k02))
        phi = torch.acos((0.5 * det).clamp(-1.0, 1.0)) / 3.0
        l1 = (q + 2.0 * p * torch.cos(phi)).clamp_min(0.0)
        l3 = (q + 2.0 * p * torch.cos(phi + 2.0 * math.pi / 3.0)).clamp_min(0.0)
        l2 = (3.0 * q - l1 - l3).clamp_min(0.0)
        u1 = torch.sqrt(l1 + eps)
        u2 = torch.sqrt(l2 + eps)
        u3 = torch.sqrt(l3 + eps)
        d12 = -1.0 / (u1 * u2 * (u1 + u2))
        d123 = (u1 + u2 + u3) / (u1 * u2 * u3 * (u1 + u2) * (u2 + u3) * (u1 + u3))
        I3 = torch.eye(3, device=device, dtype=dtype)
        B1 = B - l1[..., None, None] * I3
        B2 = B - l2[..., None, None] * I3
        fB = ((1.0 / u1)[..., None, None] * I3
              + d12[..., None, None] * B1
              + d123[..., None, None] * (B1 @ B2))
        return Q @ (fB - I3) @ Q.T
    eye = torch.eye(s, device=device, dtype=dtype)
    evals, evecs = torch.linalg.eigh(Ahat + eps * eye)
    inv_sqrt = evecs @ torch.diag_embed(evals.clamp_min(eps).rsqrt()) @ evecs.transpose(-1, -2)
    P = eye - torch.full((s, s), 1.0 / s, device=device, dtype=dtype)
    return P @ (inv_sqrt - eye) @ P


def _centered_perp_stats(xs):
    """Return (work, hbar, nrm, e, zcoef, A) without stacking [..., d, s] or cancelling ||hbar||^2."""
    s = len(xs)
    work = [x if x.dtype == torch.float64 else x.float() for x in xs]
    floor = torch.finfo(work[0].dtype).tiny
    hbar = sum(work) / s
    nrm = hbar.norm(dim=-1, keepdim=True)
    e = torch.where(nrm > floor, hbar / nrm.clamp_min(floor), torch.zeros_like(hbar))
    zs = [x - hbar for x in work]
    zcoef = torch.stack([(z * e).sum(-1) for z in zs], -1)        # [*batch, s] = e^T (x_i - hbar)
    # Upper-triangle symmetric Gram of z_i = x_i - hbar, then subtract e-component.
    rows = [[None] * s for _ in range(s)]
    for i in range(s):
        for j in range(i, s):
            dot = (zs[i] * zs[j]).sum(-1)
            rows[i][j] = dot
            rows[j][i] = dot
    gram_z = torch.stack([torch.stack(row, -1) for row in rows], -2)
    A = gram_z - zcoef.unsqueeze(-1) * zcoef.unsqueeze(-2)        # [*batch, s, s] = H_perp^T H_perp
    return work, hbar, nrm, e, zcoef, A


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
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
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


def loop_operator(xs, gamma: float = 0.5, eps: float = 1e-2):
    """Build the operator from per-loop inputs without ever stacking them.

    `loop_whiten` wants H as one [*batch, d, s] tensor. In a real model the s loop inputs arrive
    one at a time and each is [*batch, d], so materialising that stack for every hooked matrix
    costs an extra activation per loop per layer. Nothing needs the stack: the Gram follows from
    pairwise inner products, and each whitened input can be rebuilt on demand by `loop_apply`.

    Args:
        xs: list of s tensors [*batch, d], the loop inputs in loop order.
    Returns:
        (e, coef, M) -- the unit loop-mean direction [*batch, d], the components along it
        [*batch, s], and the whitening operator [*batch, s, s]. All fp32 (or fp64 in, fp64 out).
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    s = len(xs)
    work, hbar, nrm, e, zcoef, A = _centered_perp_stats(xs)
    floor = torch.finfo(work[0].dtype).tiny
    coef = nrm * (nrm > floor).to(work[0].dtype) + zcoef                    # [*batch, s] = e^T x_i
    eye = torch.eye(s, device=A.device, dtype=A.dtype)
    if s == 1 or gamma == 0.0:
        return e, coef, eye.expand(A.shape).clone()
    tau = A.diagonal(dim1=-2, dim2=-1).sum(-1) / (s - 1)
    Ahat = A / tau.clamp_min(floor)[..., None, None]
    DmI = gamma * _inv_sqrt_perp_minus_eye(Ahat, eps)
    M = eye + torch.where((tau <= floor)[..., None, None], torch.zeros_like(DmI), DmI)
    return e, coef, M


def loop_apply(xs, t: int, e: torch.Tensor, coef: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """Rebuild the whitened input of loop `t` from `loop_operator`'s output.

        x~_t = sum_j M_jt x_j + (coef_t - sum_j M_jt coef_j) e

    One weighted sum over the s loop inputs, so the peak cost is one extra [*batch, d] buffer
    that can be freed as soon as that loop's weight gradient is formed.
    """
    col = M[..., t]                                                          # [*batch, s]
    acc = None
    for j, x in enumerate(xs):
        w = col[..., j].unsqueeze(-1)
        term = (x if x.dtype == e.dtype else x.float()) * w
        acc = term if acc is None else acc + term
    return acc + (coef[..., t] - (col * coef).sum(-1)).unsqueeze(-1) * e


def loop_delta_operator(xs, gamma: float = 0.5, eps: float = 1e-2) -> torch.Tensor:
    """Build the compact [*batch, s, s] delta operator C without storing e or calling cuSOLVER.

    Since x~_t = x_t + (H_perp (M - I))_t and e = (1 / (s ||hbar||)) sum_j x_j, the correction
    delta_x_t = x~_t - x_t is a pure s-term linear combination of the loop inputs:

        delta_x_t = sum_j C_{jt} x_j

    where (M - I) 1 = 0 lets the large ||hbar|| term cancel analytically before any floating-point
    subtraction. Storing only C costs s*s floats per token (~1 MB per layer at N=16k, s=4) instead
    of storing e (~100 MB per layer), and contracting d_t^T x_t + d_t^T delta_x_t separately keeps
    full BF16 exponent resolution on delta_x_t at full Tensor Core speed.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    s = len(xs)
    work, _, nrm, _, zcoef, A = _centered_perp_stats(xs)
    floor = torch.finfo(work[0].dtype).tiny
    if s == 1 or gamma == 0.0:
        return torch.zeros(A.shape, device=A.device, dtype=A.dtype)
    tau = A.diagonal(dim1=-2, dim2=-1).sum(-1) / (s - 1)
    Ahat = A / tau.clamp_min(floor)[..., None, None]
    DmI = gamma * _inv_sqrt_perp_minus_eye(Ahat, eps)
    DmI = torch.where((tau <= floor)[..., None, None], torch.zeros_like(DmI), DmI)
    # Because DmI 1 = 0, sum_k coef_k DmI_{kt} = sum_k zcoef_k DmI_{kt} identically.
    z_proj = torch.einsum("...i,...ij->...j", zcoef, DmI)                    # [*batch, s]
    alpha = torch.where(nrm > floor, z_proj / (s * nrm.clamp_min(floor)), torch.zeros_like(z_proj))
    return DmI - alpha.unsqueeze(-2)                                         # [*batch, s, s]


def loop_delta_apply(xs, t: int, C: torch.Tensor) -> torch.Tensor:
    """Rebuild delta_x_t = x~_t - x_t = sum_j C_{jt} x_j in fp32 (or fp64)."""
    col = C[..., t]                                                          # [*batch, s]
    acc = None
    for j, x in enumerate(xs):
        w = col[..., j].unsqueeze(-1)
        term = (x if x.dtype == C.dtype else x.float()) * w
        acc = term if acc is None else acc + term
    return acc


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
