"""ReLoop: 2 Scalars (8 Bytes) Fix Deep Looped Transformers.

Core vertical depth-looping law (Driven Euclidean Loop):
    x_{r+1} = x_r + (alpha / R) * (F_core(x_r) - x_r) + (beta / R) * x_pre

where:
    - x_pre is the post-prelude hidden state (computed once before the loop),
    - F_core is the shared weight-tied core stack,
    - R is the number of loop iterations (can vary at test time for depth extrapolation),
    - alpha, beta are two learned fp32 scalar parameters (8 bytes total; zero added RMSNorms).

Initialization note:
    - For architectures with zero-initialized residual output projections (NextTransformer,
      Modded-NanoGPT), start near alpha_init ~ 0.5-1.0 * R (e.g. alpha_init=4.0, beta_init=1.5
      at R=8; alpha_init=2.0, beta_init=0.4 at R=2).
    - For standard non-zero std=0.02 initialization (Llama / LoRM), alpha_init=1.0 settles
      near 0.25-0.5.

Optional spherical variant (`spherical=True`) projects each post-step state back onto the
RMSNorm sphere, stabilizing truncated BPTT boundaries (e.g. 4-BPTT on R=8 loops).

Also includes `TapFeedbackGate`: the 1-scalar (+4 bytes) causally shifted, amplitude-calibrated
mid-trunk feedback gate for horizontal latent-feedback transformers (LoRM).

Pure PyTorch, zero external dependencies (works on PyTorch >= 2.0; uses F.rms_norm when >= 2.4).
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional
import math

import torch
from torch import nn
from torch.nn import functional as F


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),), eps=eps)
    x_f = x.float()
    return (x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)).to(dtype=x.dtype)


class ReLoop(nn.Module):
    """2-scalar driven loop integrator for weight-tied looped transformers.

    Adds exactly 2 learnable fp32 scalar parameters (`alpha`, `beta` = 8 bytes)
    with zero added normalization layers in the default Euclidean mode.

    Args:
        alpha_init: Initial value of the core-delta scale `alpha` (default `4.0` for R=8).
        beta_init: Initial value of the prelude-drive scale `beta` (default `1.5` for R=8).
        spherical: If True, applies parameter-free RMSNorm after each loop step
            (recommended when using truncated BPTT such as 4-BPTT on R=8).
        vector_beta: If True, uses a per-channel vector `beta` of shape `(dim,)`
            instead of a single scalar (default False = strictly 2 floats).
        dim: Hidden dimension (only required when `vector_beta=True`).
    """

    def __init__(
        self,
        alpha_init: float = 4.0,
        beta_init: float = 1.5,
        *,
        spherical: bool = False,
        vector_beta: bool = False,
        dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.alpha_init = float(alpha_init)
        self.beta_init = float(beta_init)
        self.spherical = bool(spherical)
        self.vector_beta = bool(vector_beta)
        self.alpha = nn.Parameter(torch.full((1,), self.alpha_init, dtype=torch.float32))
        if vector_beta:
            if dim is None or dim <= 0:
                raise ValueError("dim must be a positive integer when vector_beta=True")
            self.beta = nn.Parameter(torch.full((dim,), self.beta_init, dtype=torch.float32))
        else:
            self.beta = nn.Parameter(torch.full((1,), self.beta_init, dtype=torch.float32))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.alpha.fill_(self.alpha_init)
        self.beta.fill_(self.beta_init)

    def step(
        self,
        x: torch.Tensor,
        core_out: torch.Tensor,
        x_pre: torch.Tensor,
        loops: int,
    ) -> torch.Tensor:
        """Apply one ReLoop integration step at depth denominator `loops` (R), preserving `x.dtype`."""
        if loops <= 0:
            raise ValueError(f"loops must be positive, got {loops}")
        inv_r = 1.0 / float(loops)
        delta = core_out - x
        alpha = self.alpha.to(dtype=x.dtype)
        beta = self.beta.to(dtype=x.dtype)
        inc = (inv_r * (alpha * delta + beta * x_pre)).to(dtype=x.dtype)
        out = x + inc
        return rms_norm(out) if self.spherical else out

    def forward(
        self,
        x_pre: torch.Tensor,
        core_fn: Callable[[torch.Tensor, int], torch.Tensor],
        loops: int,
        *,
        tbptt_loops: int = 0,
    ) -> torch.Tensor:
        """Run `loops` weight-tied iterations of `core_fn(x, r)` starting from `x_pre`.

        Args:
            x_pre: Post-prelude state tensor of shape `[..., dim]`.
            core_fn: Callable `(x, loop_idx) -> h` executing the shared core stack.
            loops: Number of loop iterations `R` (can differ at test time).
            tbptt_loops: If `0 < tbptt_loops < loops` and `self.training`, backprop only
                through the last `tbptt_loops` iterations (`0` or `>= loops` runs full BPTT).
        """
        if loops <= 0:
            raise ValueError(f"loops must be positive, got {loops}")
        use_tbptt = self.training and (0 < tbptt_loops < loops)
        x = x_pre
        for r in range(loops):
            no_grad = use_tbptt and (r < loops - tbptt_loops)
            if use_tbptt and r == loops - tbptt_loops:
                x = x.detach()
            with torch.no_grad() if no_grad else nullcontext():
                h = core_fn(x, r)
                x = self.step(x, h, x_pre, loops)
        return x


class TapFeedbackGate(nn.Module):
    """1-scalar (+4 bytes) causally shifted, amplitude-calibrated feedback gate (LoRM).

    Fixes the 27.7x step-0 RMS mismatch between token embeddings (~0.02) and
    linearly projected normalized hidden states (~0.55) while enforcing strict causality:
        x_in[:, 0]  = x_emb[:, 0]
        x_in[:, t]  = x_emb[:, t] + (alpha / sqrt(dim)) * fb_proj[:, t - 1]   (t >= 1)

    where `fb_proj = W_fb(RMSNorm(h_tap))` is the unshifted projection of the previous pass's
    mid-trunk tap state (recommended tap depth: ~2L/3, e.g., L=8 for 12L or L=16 for 24L).
    """

    def __init__(self, dim: int, alpha_init: float = 2.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)
        self.alpha_init = float(alpha_init)
        self.inv_sqrt_dim = 1.0 / math.sqrt(self.dim)
        self.alpha = nn.Parameter(torch.full((1,), self.alpha_init, dtype=torch.float32))

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.alpha.fill_(self.alpha_init)

    def forward(self, x_emb: torch.Tensor, fb_proj: torch.Tensor) -> torch.Tensor:
        """Mix `x_emb` [B, T, D] with causally shifted `fb_proj` [B, T, D], preserving `x_emb.dtype`."""
        if x_emb.shape != fb_proj.shape or x_emb.ndim < 2:
            raise ValueError(f"Expected matching [..., T, D] tensors, got {x_emb.shape} vs {fb_proj.shape}")
        if x_emb.size(-2) <= 1:
            return x_emb
        scale = (self.alpha * self.inv_sqrt_dim).to(dtype=x_emb.dtype)
        shifted = (scale * fb_proj[..., :-1, :]).to(dtype=x_emb.dtype)
        return torch.cat((x_emb[..., :1, :], x_emb[..., 1:, :] + shifted), dim=-2)
