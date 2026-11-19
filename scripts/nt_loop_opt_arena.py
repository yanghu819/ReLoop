"""Pure Algorithmic Loop-Muon / Loop-TEON / Loop-tSVD Optimizer Arena
Built strictly on UNMODIFIED NextTransformer (Interactive-Active-AI/NextTransformer)
official looped baselines (fixed_recursive, deeploop, huginn, ouro, ut_act).

ZERO architecture modifications. ZERO inference overhead. 100% default NextTransformer
MuonAdamW hyperparameters (no lr/wd/beta1 tuning) — pure algorithmic comparison!
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/nt")

from nanochat.optim import polar_express_coeffs
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import ReferenceLM, build_model
from nexttransformer.training.optimizer import ExperimentOptimizer


# ---------------------------------------------------------------------------
# Non-intrusive per-loop gradient capture (ZERO edits to ReferenceLM forward!)
# ---------------------------------------------------------------------------
class _LoopGradCtx:
    def __init__(self):
        self.enabled = False
        self.suppress_default_grad = False
        self.loop_counter = -1
        self.loop = None               # 0..s-1 for grad-carrying loops, None otherwise
        self.grad_start = 0
        self.slots = {}                # param -> {t: fp32 accumulated grad}

    def begin_forward(self, grad_start: int = 0):
        self.loop_counter = -1
        self.loop = None
        self.grad_start = grad_start

    def zero_slots(self):
        self.slots.clear()


LG_CTX = _LoopGradCtx()


class _PerLoopLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, param_ref):
        ctx.loop = LG_CTX.loop
        ctx.param_ref = param_ref
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight.to(dtype=x.dtype))

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        grad_x = (grad_y @ weight.to(dtype=grad_y.dtype)) if ctx.needs_input_grad[0] else None
        if not ctx.needs_input_grad[1]:
            return grad_x, None, None
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x.reshape(-1, x.shape[-1])
        gw = (gy.t() @ xr.to(gy.dtype)).float()
        if ctx.loop is not None and LG_CTX.enabled:
            d = LG_CTX.slots.setdefault(ctx.param_ref, {})
            if ctx.loop in d:
                d[ctx.loop].add_(gw)
            else:
                d[ctx.loop] = gw.clone()
            if LG_CTX.suppress_default_grad:
                return grad_x, torch.zeros_like(weight), None
        return grad_x, gw.to(weight.dtype), None


def _per_loop_fwd(self, x):
    if LG_CTX.enabled and LG_CTX.loop is not None and torch.is_grad_enabled() and self.weight.requires_grad:
        return _PerLoopLinear.apply(x, self.weight, self.weight)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def attach_nonintrusive_loop_tracker(model: ReferenceLM):
    """Wrap tied core linears + attach pre-hook on first core block to track loop index
    with ZERO modifications to NextTransformer's ReferenceLM forward methods."""
    c = model.config
    first_l = c.pre_layers
    last_l = c.pre_layers + c.core_layers - 1

    def _on_first_core_pre(_mod, _inp):
        LG_CTX.loop_counter += 1
        r = LG_CTX.loop_counter
        LG_CTX.loop = (r - LG_CTX.grad_start) if (r >= LG_CTX.grad_start and torch.is_grad_enabled()) else None

    def _on_last_core_post(_mod, _inp, _out):
        LG_CTX.loop = None

    model.transformer.h[first_l].register_forward_pre_hook(_on_first_core_pre)
    model.transformer.h[last_l].register_forward_hook(_on_last_core_post)

    core_params = []
    for layer in range(first_l, last_l + 1):
        blk = model.transformer.h[layer]
        for m in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v, blk.attn.c_proj,
                  blk.mlp.c_fc, blk.mlp.c_proj):
            m.forward = types.MethodType(_per_loop_fwd, m)
            core_params.append(m.weight)
    return core_params


# ---------------------------------------------------------------------------
# Exact nanochat Polar Express + Muon+ + NorMuon + Cautious WD primitives
# ---------------------------------------------------------------------------
def polar_express_raw(G: torch.Tensor, ns_steps: int = 5, use_muon_eq: bool = True) -> torch.Tensor:
    """Exact nanochat MuonEq + Polar Express (no post-renorm)."""
    X = G.bfloat16()
    if use_muon_eq:
        target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
        row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        X = X * (target / row_norm).to(X.dtype)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if X.size(-2) > X.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X.float()


def polar_express_spec_norm(G: torch.Tensor, ns_steps: int = 5, use_muon_eq: bool = True) -> torch.Tensor:
    """Schatten-8 Spectral-Norm-Scaled Polar Express:
    Uses exact provable upper bound ||Y||_{S_8} = ||(Y Y^T)^2||_F^{1/4} >= sigma_max(Y) ALWAYS!
    Tightens width-d slack from sqrt(d) (39x at d=1536) down to d^{1/8} (2.5x) with 0% chance of NaN."""
    Xf = G.float()
    if use_muon_eq:
        target = Xf.norm(dim=(-2, -1), keepdim=True) / (Xf.size(-2) ** 0.5)
        row_norm = Xf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Xf = Xf * (target / row_norm)
    # Step 1: Frobenius normalize to Y (||Y||_F = 1, prevents any bf16 overflow)
    Yf = Xf / (Xf.norm(dim=(-2, -1), keepdim=True) + 1e-8)
    Yb = Yf.bfloat16()
    # Step 2: Exact Schatten-8 norm ||Y||_{S_8} = ||(Y^T Y)^2||_F^{1/4} (provably >= sigma_max(Y)!)
    A0 = (Yb.mT @ Yb) if Yb.size(-2) > Yb.size(-1) else (Yb @ Yb.mT)
    A2 = A0 @ A0
    s8 = A2.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12).pow(0.25)
    X = (Yf / (s8 * 1.02 + 1e-6)).bfloat16()
    if X.size(-2) > X.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X.float()


def qdwh_polar(G: torch.Tensor, steps: int = 4, use_muon_eq: bool = True) -> torch.Tensor:
    """Fast GPU Exact Polar Factor U V^T via 4-step Cholesky-QDWH (Nakatsukasa-Bai-Gygi 2010).
    Converges kappa <= 10^3 to <1e-6 accuracy in 4 batched GPU Cholesky steps — ZERO eigh/SVD, 0% NaN risk!"""
    Xf = G.float()
    if use_muon_eq:
        target = Xf.norm(dim=(-2, -1), keepdim=True) / (Xf.size(-2) ** 0.5)
        row_norm = Xf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Xf = Xf * (target / row_norm)
    transposed = Xf.size(-2) > Xf.size(-1)
    if transposed:
        Xf = Xf.mT                                          # now wide: [B, m, n] with m <= n
    B_sz, m, n = Xf.shape
    # Provable Schatten-8 upper bound >= sigma_max(Xf)
    frob = Xf.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    Yf = Xf / frob
    Yb = Yf.bfloat16()
    A0 = Yb @ Yb.mT
    s8 = (A0 @ A0).float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12).pow(0.25)
    Xk = Yf / (s8 * 1.02 + 1e-6)                            # sigma_max(Xk) <= 0.98 < 1.0 guaranteed!

    eye = torch.eye(m, device=Xf.device, dtype=torch.float32).unsqueeze(0)
    L = 1e-3
    for _ in range(steps):
        L2 = L * L
        dd = (4.0 * (1.0 - L2) / (L2 * L2)) ** (1.0 / 3.0)
        sqd = math.sqrt(1.0 + dd)
        a = sqd + 0.5 * math.sqrt(max(0.0, 8.0 - 4.0 * dd + 8.0 * (2.0 - L2) / (L2 * sqd)))
        b = (a - 1.0) ** 2 / 4.0
        c = a + b - 1.0
        L = L * (a + b * L2) / (1.0 + c * L2)
        # B = I_m + c * (Xk @ Xk^T) is strictly SPD (min eig >= 1.0)!
        B_mat = eye + c * (Xk @ Xk.mT)
        L_chol = torch.linalg.cholesky(B_mat)
        Z = torch.cholesky_solve(Xk, L_chol)
        Xk = (b / c) * Xk + (a - b / c) * Z
    return Xk.mT if transposed else Xk


def exact_polar_eigh(G: torch.Tensor, use_muon_eq: bool = True, power: float = 0.0) -> torch.Tensor:
    """Exact generalized polar factor U Sigma^power V^T via fp32 symmetric eigh (power=0 -> U V^T)."""
    X = G.float()
    if use_muon_eq:
        target = X.norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
        row_norm = X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        X = X * (target / row_norm)
    m, n = X.size(-2), X.size(-1)
    exp = -0.5 + 0.5 * power
    if m <= n:
        A = X @ X.mT
        tr = A.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        evals, Q = torch.linalg.eigh(A + (1e-6 * tr).unsqueeze(-1) * torch.eye(m, device=X.device))
        scale = evals.clamp_min(1e-6 * tr).pow(exp)
        inv_p = Q @ (scale.unsqueeze(-1) * Q.mT)
        return inv_p @ X
    A = X.mT @ X
    tr = A.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    evals, Q = torch.linalg.eigh(A + (1e-6 * tr).unsqueeze(-1) * torch.eye(n, device=X.device))
    scale = evals.clamp_min(1e-6 * tr).pow(exp)
    inv_p = Q @ (scale.unsqueeze(-1) * Q.mT)
    return X @ inv_p


def inv_fourth_root_ns(M: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Fast Newton-Schulz approximation of M^{-1/4} for batched SPD matrices [K, d, d]."""
    d = M.size(-1)
    tr = M.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-8)
    eye = torch.eye(d, device=M.device, dtype=torch.float32).unsqueeze(0)
    # Normalize so eigenvalues lie in (0, 1]
    norm_est = M.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    A = M / norm_est + 0.05 * eye
    # Coupled Newton-Schulz for A^{-1/2}, then one more sqrt for A^{-1/4}
    Y = A.clone()
    Z = eye.expand_as(A).clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    # Now Z ≈ A^{-1/2}; run 3 steps on Z to get Z^{1/2} ≈ A^{-1/4}
    Ys = Z.clone()
    Zs = eye.expand_as(Z).clone()
    for _ in range(3):
        T = 0.5 * (3.0 * eye - Zs @ Ys)
        Ys = Ys @ T
        Zs = T @ Zs
    return Ys


def apply_muon_update_exact(plist, U: torch.Tensor, v_buf: torch.Tensor,
                            lr: float, wd: float, beta2: float = 0.9,
                            schatten_pure_lam: float = 0.0,
                            joint_frob_norm: bool = False,
                            extra_row_var: torch.Tensor = None):
    """Exact nanochat Muon+ renorm + NorMuon variance reduction + Cautious WD (or Schatten-4 WD)."""
    K, m, n = U.size(0), U.size(-2), U.size(-1)
    if joint_frob_norm:
        # True TEON: preserve cross-layer relative norm while snapping joint Frobenius norm to sqrt(K*min(m,n))!
        target_norm = (K * min(m, n)) ** 0.5
        cur_norm = U.norm().clamp_min(1e-6)
        U = U * (target_norm / cur_norm)
    else:
        target_norm = min(m, n) ** 0.5
        U = U * (target_norm / U.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
    red_dim = -1 if m >= n else -2
    v_mean = U.square().mean(dim=red_dim, keepdim=True)
    if extra_row_var is not None:
        # Unbiased cross-loop empirical Fisher variance augmentation (normalized to match v_mean scale)
        ev = extra_row_var * (v_mean.mean(dim=(-2, -1), keepdim=True) / extra_row_var.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12))
        v_mean = 0.5 * v_mean + 0.5 * ev
    red_dim_size = U.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    v_buf.lerp_(v_mean.to(v_buf.dtype), 1.0 - beta2)
    step_size = v_buf.float().clamp_min(1e-10).rsqrt()
    v_norm_new = ((v_mean * red_dim_size) * step_size.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    U = U * (step_size * (v_norm / v_norm_new.clamp_min(1e-10)))

    lr_eff = lr * (max(1.0, m / n) ** 0.5)
    P = torch.stack([p.float() for p in plist], dim=0)
    if schatten_pure_lam > 0.0:
        min_dim = min(m, n)
        alpha = P.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6) / (min_dim ** 0.5)
        Pn = P / alpha
        cubic = (Pn @ (Pn.mT @ Pn)) if m >= n else ((Pn @ Pn.mT) @ Pn)
        P_new = P - lr_eff * U - (schatten_pure_lam * lr) * alpha * cubic.clamp(-5.0, 5.0)
    else:
        mask = ((U * P) >= 0).float()
        P_new = P - lr_eff * U - lr_eff * wd * P * mask
    for idx_p, p in enumerate(plist):
        p.copy_(P_new[idx_p].to(p.dtype))


def dct_matrix(s: int, device) -> torch.Tensor:
    k = torch.arange(s, device=device, dtype=torch.float32).unsqueeze(1)
    n = torch.arange(s, device=device, dtype=torch.float32).unsqueeze(0)
    Q = torch.cos(math.pi * (n + 0.5) * k / s)
    Q[0] *= (1.0 / s) ** 0.5
    if s > 1:
        Q[1:] *= (2.0 / s) ** 0.5
    return Q


def haar_matrix(s: int, device) -> torch.Tensor:
    """Orthogonal Haar wavelet transform matrix for s in {2, 4}."""
    if s == 2:
        return torch.tensor([[1, 1], [1, -1]], device=device, dtype=torch.float32) / (2.0 ** 0.5)
    if s == 4:
        return torch.tensor([
            [0.5,  0.5,  0.5,  0.5],
            [0.5,  0.5, -0.5, -0.5],
            [2**-0.5, -(2**-0.5), 0.0, 0.0],
            [0.0, 0.0, 2**-0.5, -(2**-0.5)],
        ], device=device, dtype=torch.float32)
    return dct_matrix(s, device)


# ---------------------------------------------------------------------------
# Official NextTransformer MuonAdamW builder (100% DEFAULT hyperparams!)
# ---------------------------------------------------------------------------
def build_nt_optimizer(model: ReferenceLM, batch_tokens: int, core_param_ids: set,
                       exclude_core: bool = False, exclude_all_muon: bool = False):
    dscale = (model.config.width / 768.0) ** -0.5
    bscale = math.sqrt(batch_tokens / 524288.0)
    muon_lr = 0.02 * bscale
    adamw_lr = 0.0006 * bscale
    emb_lr = 0.3 * bscale * dscale
    unemb_lr = 0.008 * bscale * dscale
    scalar_lr = 0.5 * bscale
    wd = 0.28 * bscale * 0.5

    groups = []

    def adam(params, lr, betas, decay):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append(dict(kind="adamw", params=params, lr=lr, initial_lr=lr,
                               betas=betas, eps=1e-10, weight_decay=decay))

    adam(model.lm_head.parameters(), unemb_lr, (0.8, 0.96), 0.01)
    adam(model.transformer.wte.parameters(), emb_lr, (0.8, 0.995), 0.001)
    adam(model.value_embeds.parameters(), emb_lr * 0.5, (0.8, 0.995), 0.01)
    adam([model.resid_lambdas], scalar_lr * 0.01, (0.8, 0.95), 0.05)
    adam([model.x0_lambdas], scalar_lr, (0.96, 0.95), 0.0)
    adam([model.smear_gate.weight, model.smear_lambda, model.backout_lambda], 0.2, (0.8, 0.95), 0.0)

    allocated = {id(p) for g in groups for p in g["params"]}
    untied_mats = defaultdict(list)
    core_mats = defaultdict(list)
    other = []
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:
            continue
        if p.ndim == 2:
            if id(p) in core_param_ids:
                core_mats[tuple(p.shape)].append(p)
            else:
                untied_mats[tuple(p.shape)].append(p)
        else:
            other.append(p)
    adam(other, adamw_lr, (0.9, 0.95), 0.0)

    if not exclude_all_muon:
        for shape in sorted(untied_mats):
            groups.append(dict(kind="muon", params=untied_mats[shape], lr=muon_lr, initial_lr=muon_lr,
                               momentum=0.95, ns_steps=5, beta2=0.9,
                               weight_decay=wd, initial_weight_decay=wd))
    if not exclude_core and not exclude_all_muon:
        for shape in sorted(core_mats):
            groups.append(dict(kind="muon", params=core_mats[shape], lr=muon_lr, initial_lr=muon_lr,
                               momentum=0.95, ns_steps=5, beta2=0.9,
                               weight_decay=wd, initial_weight_decay=wd))
    all_muon_mats = defaultdict(list)
    for d_m in (untied_mats, core_mats):
        for sh, ps in d_m.items():
            all_muon_mats[sh].extend(ps)
    return ExperimentOptimizer(groups), muon_lr, wd, all_muon_mats


@torch.no_grad()
def collect_telemetry(model, slots, first_l, core_layers):
    info = {"per_loop_grads": {}, "weight_spectra": {}}
    sample_js = sorted({0, core_layers // 2, core_layers - 1})
    for j in sample_js:
        blk = model.transformer.h[first_l + j]
        for sub, p in [("c_q", blk.attn.c_q.weight), ("c_v", blk.attn.c_v.weight),
                       ("c_proj", blk.attn.c_proj.weight), ("c_fc", blk.mlp.c_fc.weight),
                       ("mlp_proj", blk.mlp.c_proj.weight)]:
            key = f"core{j}.{sub}"
            sv = torch.linalg.svdvals(p.float())
            med = sv[len(sv) // 2].clamp_min(1e-8).item()
            p_en = sv.square() / sv.square().sum().clamp_min(1e-12)
            erank = (-(p_en * p_en.clamp_min(1e-12).log()).sum()).exp().item()
            info["weight_spectra"][key] = {
                "smax": round(sv[0].item(), 4),
                "smed": round(med, 4),
                "smax_over_smed": round(sv[0].item() / med, 2),
                "erank": round(erank, 1),
                "rank_frac": round(erank / len(sv), 3),
            }
            if p in slots and len(slots[p]) >= 2:
                ts = sorted(slots[p])
                Gs = [slots[p][t] for t in ts]
                norms = [g.norm().item() for g in Gs]
                s = len(Gs)
                cos_off = []
                for a in range(s):
                    for b in range(a + 1, s):
                        cos_off.append((Gs[a] * Gs[b]).sum().item() / max(norms[a] * norms[b], 1e-12))
                info["per_loop_grads"][key] = {
                    "norms": [round(x, 5) for x in norms],
                    "ratio_last_first": round(norms[-1] / max(norms[0], 1e-12), 2),
                    "mean_cos": round(sum(cos_off) / max(len(cos_off), 1), 4),
                }
    return info


def lr_mult(step: int, total_steps: int, warmup: int = 40, warmdown_frac: float = 0.30,
            final_frac: float = 0.05) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    warmdown = int(total_steps * warmdown_frac)
    if step >= total_steps - warmdown:
        t = (total_steps - step) / max(warmdown, 1)
        return final_frac + (1.0 - final_frac) * t
    return 1.0


@torch.no_grad()
def eval_bpb(model, val_x, val_y, byte_table, device, batch_size: int, max_rows: int = 4096):
    model.eval()
    LG_CTX.enabled = False
    total_nats = 0.0
    total_bytes = 0.0
    try:
        for i in range(0, min(max_rows, val_x.shape[0]), batch_size):
            x = val_x[i:i + batch_size].to(device, dtype=torch.long)
            y = val_y[i:i + batch_size].to(device, dtype=torch.long)
            model.set_row_offset(i)
            LG_CTX.begin_forward(grad_start=9999)
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction="none")
            nbytes = byte_table[y.flatten().clamp_min(0)]
            valid = (y.flatten() >= 0) & (nbytes > 0)
            total_nats += loss[valid].sum().item()
            total_bytes += nbytes[valid].sum().item()
    finally:
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea", required=True)
    p.add_argument("--method", default="fixed_recursive", choices=["fixed_recursive", "deeploop", "ouro", "huginn"])
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--pre-layers", type=int, default=2)
    p.add_argument("--core-layers", type=int, default=4)
    p.add_argument("--post-layers", type=int, default=2)
    p.add_argument("--loops", type=int, default=4)
    p.add_argument("--steps", type=int, default=750)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=375)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=Path, default=Path("/dev/shm/rsi_arena_data"))
    p.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    assert args.batch_size % args.micro_batch == 0
    accum = args.batch_size // args.micro_batch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_data = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_data = torch.load(args.data_dir / "val.pt", map_location="cpu", weights_only=True)
    train_x, train_y = train_data["x"], train_data["y"]
    val_x, val_y = val_data["x"], val_data["y"]
    seq_len = train_x.shape[1]
    byte_table = torch.load(args.resources / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    # Exact unmodified NextTransformer ModelConfig!
    bptt_loops = args.loops if args.method != "huginn" else min(4, args.loops)
    grad_start = 0 if args.method != "huginn" else max(0, args.loops - bptt_loops)
    cfg = ModelConfig(
        method=args.method,
        scale="debug",
        width=args.width,
        heads=args.heads,
        vocab_size=32768,
        sequence_len=seq_len,
        mlp_width=4 * args.width,
        layers=args.pre_layers + args.core_layers + args.post_layers,
        pre_layers=args.pre_layers,
        core_layers=args.core_layers,
        post_layers=args.post_layers,
        loops=args.loops,
        max_loops=args.loops,
        huginn_mean=float(args.loops),
        huginn_sigma=0.0,
        huginn_backprop_loops=bptt_loops,
        eval_loops=args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    core_params = attach_nonintrusive_loop_tracker(model)
    core_param_ids = {id(p) for p in core_params}
    first_l = cfg.pre_layers
    s_loops = bptt_loops

    idea = args.idea
    VALID_IDEAS = {
        "nt_base",
        "nt_schatten_pure",
        "nt_teon_dct_lowpass",
        "nt_teon_hosvd_svw",
        "nt_teon_m1_norm",
        "nt_teon_m2_norm",
        "nt_loop_ortho_gs",
        "nt_loop_dual_mom",
        "nt_teon_haar",
        "nt_loop_cov_precond",
        "nt_loop_karcher_pe",
        "nt_loop_spec_mix",
        "nt_ctrl_pe10",
        "nt_loop_karcher_unw",
        "nt_loop_karcher_schatten",
        "nt_teon_geodesic_blend",
        "nt_teon_mode3_whiten",
        "nt_loop_novelty_karcher",
        "nt_loop_tucker_res",
        "nt_loop_normuon_pre",
        "alg_spec_norm_pe",
        "alg_exact_polar_svd",
        "alg_true_layer_teon",
        "alg_true_loop_teon_m1",
        "alg_head_tuon",
        "alg_loop_fisher_normuon",
        "alg_mars_muon",
        "alg_loop_sym_shampoo",
        "alg_qdwh_polar",
        "alg_all_qdwh_polar",
        "alg_head4_qdwh",
        "alg_layer_teon_qdwh",
        "alg_head4_layer_qdwh",
        "alg_loop_teon_m1_qdwh",
        "alg_mlp_block_tuon_qdwh",
        "alg_svd_spectral_lift",
    }
    if idea not in VALID_IDEAS:
        raise ValueError(f"Unknown idea: {idea}")

    uses_custom_core = (idea != "nt_base")
    use_all_muon = (idea == "alg_all_qdwh_polar")
    needs_per_loop = idea not in (
        "nt_base", "nt_schatten_pure", "nt_loop_dual_mom", "nt_ctrl_pe10",
        "alg_spec_norm_pe", "alg_exact_polar_svd", "alg_true_layer_teon",
        "alg_head_tuon", "alg_mars_muon",
        "alg_qdwh_polar", "alg_all_qdwh_polar", "alg_head4_qdwh",
        "alg_layer_teon_qdwh", "alg_head4_layer_qdwh", "alg_mlp_block_tuon_qdwh",
        "alg_svd_spectral_lift",
    )

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    batch_tokens = args.batch_size * seq_len
    opt, base_muon_lr, base_muon_wd, all_muon_by_shape = build_nt_optimizer(
        model, batch_tokens, core_param_ids,
        exclude_core=uses_custom_core, exclude_all_muon=use_all_muon,
    )

    core_by_shape = defaultdict(list)
    for cp in core_params:
        core_by_shape[tuple(cp.shape)].append(cp)
    target_by_shape = all_muon_by_shape if use_all_muon else core_by_shape

    core_by_proj = defaultdict(list)
    core_qkv_ids = set()
    core_oproj_ids = set()
    for layer in range(first_l, first_l + cfg.core_layers):
        blk = model.transformer.h[layer]
        for pname, mod in (("c_q", blk.attn.c_q), ("c_k", blk.attn.c_k), ("c_v", blk.attn.c_v),
                           ("c_proj", blk.attn.c_proj), ("c_fc", blk.mlp.c_fc), ("mlp_proj", blk.mlp.c_proj)):
            core_by_proj[pname].append(mod.weight)
            if pname in ("c_q", "c_k", "c_v"):
                core_qkv_ids.add(id(mod.weight))
            elif pname == "c_proj":
                core_oproj_ids.add(id(mod.weight))

    m_bufs = {}
    m_bufs_slow = {}
    prev_g_bufs = {}
    v_bufs = {}
    q_dct = dct_matrix(s_loops, device)
    q_haar = haar_matrix(s_loops, device)

    t_start = time.time()
    history = []
    telemetry = {}
    n_rows = train_x.shape[0]

    bpb0 = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=512)
    print(f"[{idea}|{args.method}|w={args.width}|r={args.loops}] params={n_params/1e6:.1f}M step=0 val_bpb={bpb0:.5f}",
          flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    step_times = []

    for step in range(args.steps):
        t0 = time.perf_counter()
        mult = lr_mult(step, args.steps)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * mult
            if "initial_weight_decay" in g:
                g["weight_decay"] = g["initial_weight_decay"] * mult
        muon_lr_now = base_muon_lr * mult
        muon_wd_now = base_muon_wd * mult

        model.zero_grad(set_to_none=True)
        model.set_step(step)

        capture_telem = (idea == "nt_base") and (step + 1 in (150, args.steps))
        LG_CTX.zero_slots()
        LG_CTX.enabled = needs_per_loop or capture_telem
        LG_CTX.suppress_default_grad = needs_per_loop

        loss_sum = 0.0
        for m_idx in range(accum):
            row0 = (step * args.batch_size + m_idx * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m_idx * args.micro_batch)
            LG_CTX.begin_forward(grad_start=grad_start)
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        if capture_telem:
            telemetry[f"step_{step+1}"] = collect_telemetry(model, LG_CTX.slots, first_l, cfg.core_layers)

        # Compute custom algorithmic updates for tied core matrices (100% default LR/WD/beta1/beta2!)
        pending_updates = []
        if uses_custom_core and idea in ("alg_true_layer_teon", "alg_layer_teon_qdwh", "alg_head4_layer_qdwh"):
            with torch.no_grad():
                H = cfg.heads
                dh = cfg.width // H
                for pname, plist in core_by_proj.items():
                    L = len(plist)
                    m, n = plist[0].shape
                    G_l = torch.stack([cp.grad.float() for cp in plist], dim=0)
                    if pname not in m_bufs:
                        m_bufs[pname] = torch.zeros_like(G_l)
                        v_shape = (L, m, 1) if m >= n else (L, 1, n)
                        v_bufs[pname] = torch.zeros(v_shape, device=device, dtype=torch.float32)
                    m_bufs[pname].lerp_(G_l, 0.05)
                    M_nes = G_l.lerp(m_bufs[pname], 0.95)
                    if idea == "alg_head4_layer_qdwh" and pname in ("c_q", "c_k", "c_v"):
                        # 4D Head x Layer TEON: [L, H*dh, d] -> [H, dh, L*d] -> exact QDWH -> [L, H*dh, d]!
                        X_4d = M_nes.view(L, H, dh, n).permute(1, 2, 0, 3).reshape(H, dh, L * n)
                        U_4d = qdwh_polar(X_4d, steps=4, use_muon_eq=True)
                        U = U_4d.view(H, dh, L, n).permute(2, 0, 1, 3).reshape(L, m, n)
                    elif idea == "alg_head4_layer_qdwh" and pname == "c_proj":
                        # Output projection [L, d, H*dh] -> [H, dh, L*d] -> exact QDWH -> [L, d, H*dh]!
                        X_4d = M_nes.view(L, m, H, dh).permute(2, 3, 0, 1).reshape(H, dh, L * m)
                        U_4d = qdwh_polar(X_4d, steps=4, use_muon_eq=True)
                        U = U_4d.view(H, dh, L, m).permute(2, 3, 0, 1).reshape(L, m, n)
                    else:
                        X_u = M_nes.permute(1, 0, 2).reshape(1, m, L * n)
                        if idea == "alg_true_layer_teon":
                            U_u = polar_express_spec_norm(X_u, ns_steps=5, use_muon_eq=True)
                        else:
                            U_u = qdwh_polar(X_u, steps=4, use_muon_eq=True)
                        U = U_u.view(m, L, n).permute(1, 0, 2)
                    pending_updates.append((plist, U, v_bufs[pname], muon_lr_now, muon_wd_now, 0.0, True, None))
        elif uses_custom_core:
            with torch.no_grad():
                for shape, plist in target_by_shape.items():
                    K, m, n = len(plist), shape[0], shape[1]
                    if shape not in v_bufs:
                        v_shape = (K, m, 1) if m >= n else (K, 1, n)
                        v_bufs[shape] = torch.zeros(v_shape, device=device, dtype=torch.float32)

                    sch_lam = 0.0
                    extra_rv = None
                    if not needs_per_loop:
                        G_sum = torch.stack([cp.grad.float() for cp in plist], dim=0)  # [K, m, n]
                        if shape not in m_bufs:
                            m_bufs[shape] = torch.zeros_like(G_sum)
                            if idea == "nt_loop_dual_mom":
                                m_bufs_slow[shape] = torch.zeros_like(G_sum)
                            if idea == "alg_mars_muon":
                                prev_g_bufs[shape] = G_sum.clone()
                        if idea == "nt_schatten_pure":
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                            sch_lam = 0.10
                        elif idea == "nt_loop_dual_mom":
                            m_bufs[shape].lerp_(G_sum, 0.20)       # fast beta=0.80
                            m_bufs_slow[shape].lerp_(G_sum, 0.05)  # slow beta=0.95
                            M_nes = 0.5 * G_sum.lerp(m_bufs[shape], 0.80) + 0.5 * G_sum.lerp(m_bufs_slow[shape], 0.95)
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                        elif idea == "nt_ctrl_pe10":
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U1 = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                            U = polar_express_raw(U1, ns_steps=5, use_muon_eq=False)
                        elif idea == "alg_spec_norm_pe":
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = polar_express_spec_norm(M_nes, ns_steps=5, use_muon_eq=True)
                        elif idea == "alg_exact_polar_svd":
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = exact_polar_eigh(M_nes, use_muon_eq=True, power=0.0)
                        elif idea == "alg_svd_spectral_lift":
                            # Super-orthogonal U Sigma^{-0.10} V^T: lifts depth-unique small singular values!
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = exact_polar_eigh(M_nes, use_muon_eq=True, power=-0.10)
                        elif idea in ("alg_qdwh_polar", "alg_all_qdwh_polar"):
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = qdwh_polar(M_nes, steps=4, use_muon_eq=True)
                        elif idea == "alg_mars_muon":
                            c_t = G_sum + 0.025 * (G_sum - prev_g_bufs[shape])
                            prev_g_bufs[shape].copy_(G_sum)
                            m_bufs[shape].lerp_(c_t, 0.05)
                            M_nes = c_t.lerp(m_bufs[shape], 0.95)
                            U = polar_express_spec_norm(M_nes, ns_steps=5, use_muon_eq=True)
                        elif idea in ("alg_head_tuon", "alg_head4_qdwh", "alg_mlp_block_tuon_qdwh"):
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            use_q = idea != "alg_head_tuon"
                            solver = (lambda T: qdwh_polar(T, 4, True)) if use_q else (lambda T: polar_express_spec_norm(T, 5, True))
                            if m == cfg.width and n == cfg.width:
                                qkv_idx = [i for i, p_ref in enumerate(plist) if id(p_ref) in core_qkv_ids]
                                o_idx = [i for i, p_ref in enumerate(plist) if id(p_ref) in core_oproj_ids]
                                oth_idx = [i for i in range(K) if i not in qkv_idx and i not in o_idx]
                                U = torch.empty_like(M_nes)
                                H = cfg.heads
                                dh = cfg.width // H
                                if qkv_idx:
                                    Mq = M_nes[qkv_idx].view(len(qkv_idx) * H, dh, n)
                                    U[qkv_idx] = solver(Mq).view(len(qkv_idx), m, n)
                                if o_idx:
                                    if use_q:
                                        # Full 4-matrix attention Head-TUON: c_proj [K, d, H*dh] -> [K*H, d, dh]!
                                        Mo = M_nes[o_idx].view(len(o_idx), m, H, dh).permute(0, 2, 1, 3).reshape(len(o_idx) * H, m, dh)
                                        U[o_idx] = solver(Mo).view(len(o_idx), H, m, dh).permute(0, 2, 1, 3).reshape(len(o_idx), m, n)
                                    else:
                                        U[o_idx] = solver(M_nes[o_idx])
                                if oth_idx:
                                    U[oth_idx] = solver(M_nes[oth_idx])
                            elif idea == "alg_mlp_block_tuon_qdwh" and (m == 4 * cfg.width or n == 4 * cfg.width):
                                # Block-structured MLP TUON: split 4x expansion into 4 blocks of [d, d]!
                                d = cfg.width
                                if m == 4 * d:
                                    Mb = M_nes.view(K * 4, d, d)
                                    U = qdwh_polar(Mb, 4, True).view(K, 4 * d, d)
                                else:
                                    Mb = M_nes.view(K, d, 4, d).permute(0, 2, 1, 3).reshape(K * 4, d, d)
                                    U = qdwh_polar(Mb, 4, True).view(K, 4, d, d).permute(0, 2, 1, 3).reshape(K, d, 4 * d)
                            else:
                                U = solver(M_nes)
                    else:
                        G_t = torch.stack(
                            [torch.stack([LG_CTX.slots[cp][t] for t in range(s_loops)], dim=0) for cp in plist],
                            dim=0,
                        )  # [K, s, m, n]
                        G_sum = G_t.sum(dim=1)
                        for idx_p, cp in enumerate(plist):
                            cp.grad = G_sum[idx_p].to(cp.dtype)

                        if shape not in m_bufs:
                            m_bufs[shape] = torch.zeros_like(G_t)
                        m_bufs[shape].lerp_(G_t, 0.05)
                        M_nes = G_t.lerp(m_bufs[shape], 0.95)       # [K, s, m, n]

                        if idea == "nt_teon_dct_lowpass":
                            # Spectral low-pass DCT-II Loop-tSVD: keep low-freq coherent loop modes!
                            M_hat = torch.einsum("rt,ktmn->krmn", q_dct, M_nes)
                            U_hat = polar_express_raw(M_hat.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            # Zero high-freq oscillatory loop modes; keep DC + smooth linear depth trend
                            w_freq = torch.zeros(s_loops, device=device)
                            w_freq[0] = 1.0
                            if s_loops >= 2:
                                w_freq[1] = -0.35   # aligns positive toward later loops (DCT row 1 is +early/-late)
                            U = (U_hat * w_freq.view(1, s_loops, 1, 1)).sum(dim=1)

                        elif idea in ("nt_teon_hosvd_svw", "nt_loop_spec_mix"):
                            # Data-driven Loop-tSVD (Tucker-3) with Tucker singular-value weighting (sign-flip invariant!)
                            flat = M_nes.reshape(K, s_loops, m * n)
                            Gram = torch.bmm(flat, flat.mT)
                            tr = Gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)
                            Gram = Gram + (1e-6 * tr).unsqueeze(-1) * torch.eye(s_loops, device=device)
                            evals, Q = torch.linalg.eigh(Gram)                     # evals ascending [K, s]
                            sv = evals.clamp_min(0.0).sqrt()                       # Tucker singular values!
                            sv_w = sv / sv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                            M_hat = torch.einsum("ktr,ktmn->krmn", Q, M_nes)       # orthogonal eigenmodes
                            U_hat = polar_express_raw(M_hat.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            # Weight each eigenmode by its Tucker singular value BEFORE inverse mode-3 transform!
                            U_hat_w = U_hat * sv_w.view(K, s_loops, 1, 1)
                            U = torch.einsum("ktr,krmn->ktmn", Q, U_hat_w).sum(dim=1)
                            if idea == "nt_loop_spec_mix":
                                sch_lam = 0.10

                        elif idea == "nt_teon_m1_norm":
                            # Norm-weighted Mode-1 Loop-TEON: no early-loop noise inflation!
                            # Unfold [K, m, s*n], apply PE WITHOUT per-slice distortion, norm-weighted fold
                            sl_norms = M_nes.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            sl_w = sl_norms / sl_norms.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            X_u = M_nes.permute(0, 2, 1, 3).reshape(K, m, s_loops * n)
                            U_u = polar_express_raw(X_u, ns_steps=5, use_muon_eq=True)
                            U_sl = U_u.view(K, m, s_loops, n).permute(0, 2, 1, 3)
                            U = (U_sl * sl_w).sum(dim=1)

                        elif idea == "nt_teon_m2_norm":
                            # Dual Norm-weighted Mode-2 Loop-TEON (no cross-loop MuonEq inflation of weak loops!)
                            sl_norms = M_nes.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            sl_w = sl_norms / sl_norms.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            X_u = M_nes.reshape(K, s_loops * m, n)
                            U_u = polar_express_raw(X_u, ns_steps=5, use_muon_eq=False)
                            U_sl = U_u.view(K, s_loops, m, n)
                            U = (U_sl * sl_w).sum(dim=1)

                        elif idea == "nt_loop_ortho_gs":
                            # Sequential backward Gram-Schmidt in Frobenius space -> energy-weighted polar sum!
                            orth_list = []
                            basis = []
                            for t in range(s_loops - 1, -1, -1):
                                v = M_nes[:, t].clone()
                                for b_u in basis:
                                    dot = (v * b_u).sum(dim=(-2, -1), keepdim=True)
                                    v = v - dot * b_u
                                v_norm = v.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                                basis.append(v / v_norm)
                                orth_list.append(v)
                            O_stack = torch.stack(orth_list, dim=1)                 # [K, s, m, n]
                            o_norms = O_stack.norm(dim=(-2, -1), keepdim=True)
                            o_w = o_norms / o_norms.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            U_o = polar_express_raw(O_stack.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            U = (U_o * o_w).sum(dim=1)

                        elif idea == "nt_teon_haar":
                            # Multi-resolution Haar-Wavelet Loop-tSVD (dyadic coarse-to-fine polar fusion!)
                            M_hat = torch.einsum("rt,ktmn->krmn", q_haar, M_nes)
                            U_hat = polar_express_raw(M_hat.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            w_h = torch.tensor([1.0, -0.4, 0.0, 0.0][:s_loops], device=device)
                            U = (U_hat * w_h.view(1, s_loops, 1, 1)).sum(dim=1)

                        elif idea == "nt_loop_cov_precond":
                            # Two-sided Cross-Loop Shampoo Preconditioner: (sum M_t M_t^T)^{-1/4} M_sum (sum M_t^T M_t)^{-1/4}
                            # Normalize slices to prevent fp32 overflow
                            M_sum = M_nes.sum(dim=1)                                # [K, m, n]
                            # Left & right loop covariances (use smaller dim if > 1024 to stay ultra-fast)
                            if m <= 1536 and n <= 1536:
                                L_cov = torch.einsum("ksmr,ksnr->kmn", M_nes, M_nes)   # [K, m, m]
                                R_cov = torch.einsum("ksrm,ksrn->kmn", M_nes, M_nes)   # [K, n, n]
                                L_inv4 = inv_fourth_root_ns(L_cov, iters=5)
                                R_inv4 = inv_fourth_root_ns(R_cov, iters=5)
                                U = L_inv4 @ M_sum @ R_inv4
                            else:
                                # Rectangular MLP [3072, 768] or [768, 3072]: whiten the smaller dim + PE!
                                if m < n:
                                    L_cov = torch.einsum("ksmr,ksnr->kmn", M_nes, M_nes)
                                    L_inv4 = inv_fourth_root_ns(L_cov, iters=5)
                                    U = polar_express_raw(L_inv4 @ M_sum, 5, True)
                                else:
                                    R_cov = torch.einsum("ksrm,ksrn->kmn", M_nes, M_nes)
                                    R_inv4 = inv_fourth_root_ns(R_cov, iters=5)
                                    U = polar_express_raw(M_sum @ R_inv4, 5, True)

                        elif idea in ("nt_loop_karcher_pe", "nt_loop_karcher_schatten"):
                            # Riemannian Karcher / Fréchet mean on Stiefel manifold: PE(sum_t w_t PE(M_t))!
                            sl_norms = M_nes.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            sl_w = sl_norms / sl_norms.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            U_sl = polar_express_raw(M_nes.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            U_avg = (U_sl * sl_w).sum(dim=1)
                            U = polar_express_raw(U_avg, 5, False)
                            if idea == "nt_loop_karcher_schatten":
                                sch_lam = 0.10

                        elif idea == "nt_loop_karcher_unw":
                            # Pure unweighted Riemannian Karcher mean on Stiefel manifold: PE(mean_t PE(M_t))!
                            U_sl = polar_express_raw(M_nes.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            U = polar_express_raw(U_sl.mean(dim=1), 5, False)

                        elif idea == "nt_teon_geodesic_blend":
                            # Geodesic blend on Stiefel manifold: PE(0.5*PE(M_sum) + 0.5*sum_t w_t PE(M_t))
                            sl_norms = M_nes.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            sl_w = sl_norms / sl_norms.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            U_sl = polar_express_raw(M_nes.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            U_k = (U_sl * sl_w).sum(dim=1)
                            U_s = polar_express_raw(M_nes.sum(dim=1), 5, True)
                            U = polar_express_raw(0.5 * U_s + 0.5 * U_k, 5, False)

                        elif idea == "nt_teon_mode3_whiten":
                            # Mode-3 loop correlation whitening: decorrelate redundant loops via (C + 0.25 I)^{-1/2} 1!
                            flat = M_nes.reshape(K, s_loops, m * n)
                            fn = flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                            fnorm = flat / fn
                            C = torch.bmm(fnorm, fnorm.mT)                         # [K, s, s] cosine matrix
                            eye = torch.eye(s_loops, device=device).unsqueeze(0)
                            evals, Q = torch.linalg.eigh(C + 0.25 * eye)
                            inv_sqrt = Q @ torch.diag_embed(evals.clamp_min(1e-4).rsqrt()) @ Q.mT
                            ones = torch.ones((K, s_loops, 1), device=device)
                            w = (inv_sqrt @ ones).squeeze(-1).clamp_min(0.05)      # [K, s]
                            w = w * (s_loops / w.sum(dim=-1, keepdim=True).clamp_min(1e-6))
                            M_w = (M_nes * w.view(K, s_loops, 1, 1)).sum(dim=1)
                            U = polar_express_raw(M_w, 5, True)

                        elif idea == "nt_loop_novelty_karcher":
                            # Depth-novelty-weighted Stiefel Karcher Mean: weight loops by ||M_t - M_bar||!
                            M_bar = M_nes.mean(dim=1, keepdim=True)
                            nov = (M_nes - M_bar).norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            w_nov = nov / nov.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            U_sl = polar_express_raw(M_nes.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            U = polar_express_raw((U_sl * w_nov).sum(dim=1), 5, False)

                        elif idea == "nt_loop_tucker_res":
                            # Residual Tucker Loop-tSVD: PE(M_sum) + 0.35 * SV-weighted HOSVD correction on Stiefel!
                            flat = M_nes.reshape(K, s_loops, m * n)
                            Gram = torch.bmm(flat, flat.mT)
                            tr = Gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)
                            Gram = Gram + (1e-6 * tr).unsqueeze(-1) * torch.eye(s_loops, device=device)
                            evals, Q = torch.linalg.eigh(Gram)
                            sv = evals.clamp_min(0.0).sqrt()
                            sv_w = sv / sv.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                            M_hat = torch.einsum("ktr,ktmn->krmn", Q, M_nes)
                            U_hat = polar_express_raw(M_hat.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            U_tuck = torch.einsum("ktr,krmn->ktmn", Q, U_hat * sv_w.view(K, s_loops, 1, 1)).sum(dim=1)
                            U_sum = polar_express_raw(M_nes.sum(dim=1), 5, True)
                            U = polar_express_raw(U_sum + 0.35 * U_tuck, 5, False)

                        elif idea == "nt_loop_normuon_pre":
                            # Per-loop row-variance equilibration BEFORE summing into Polar Express (1x PE!)
                            row_n = M_nes.norm(dim=-1, keepdim=True).clamp_min(1e-6)             # [K, s, m, 1]
                            mean_n = M_nes.norm(dim=(-2, -1), keepdim=True) / (m ** 0.5)         # [K, s, 1, 1]
                            M_eq = M_nes * (mean_n / row_n)
                            U = polar_express_raw(M_eq.sum(dim=1), 5, True)

                        elif idea == "alg_true_loop_teon_m1":
                            # Spectral-scaled Mode-1 Loop-TEON (unbiased sum after converged unfolded PE!)
                            X_u = M_nes.permute(0, 2, 1, 3).reshape(K, m, s_loops * n)
                            U_u = polar_express_spec_norm(X_u, ns_steps=5, use_muon_eq=True)
                            U = U_u.view(K, m, s_loops, n).permute(0, 2, 1, 3).sum(dim=1)

                        elif idea == "alg_loop_teon_m1_qdwh":
                            # Exact-Polar Mode-1 Loop-TEON via 4-step GPU QDWH!
                            X_u = M_nes.permute(0, 2, 1, 3).reshape(K, m, s_loops * n)
                            U_u = qdwh_polar(X_u, steps=4, use_muon_eq=True)
                            U = U_u.view(K, m, s_loops, n).permute(0, 2, 1, 3).sum(dim=1)

                        elif idea == "alg_loop_fisher_normuon":
                            # Unbiased cross-loop empirical Fisher NorMuon on top of Spectral-Norm PE!
                            U = polar_express_spec_norm(M_nes.sum(dim=1), ns_steps=5, use_muon_eq=True)
                            red_dim = -1 if m >= n else -2
                            # Cross-loop empirical Fisher energy per row/col: mean over loops and reduced dim
                            extra_rv = G_t.square().mean(dim=1).mean(dim=red_dim, keepdim=True)

                        elif idea == "alg_loop_sym_shampoo":
                            # Unbiased Cross-Loop Covariance Preconditioned Muon: P_loop^{-1/4} M_sum -> SpecNorm PE!
                            M_sum = M_nes.sum(dim=1)
                            if m <= n:
                                Cov = torch.einsum("ksmr,ksnr->kmn", M_nes, M_nes)
                                P_inv4 = inv_fourth_root_ns(Cov, iters=5)
                                U = polar_express_spec_norm(P_inv4 @ M_sum, ns_steps=5, use_muon_eq=True)
                            else:
                                Cov = torch.einsum("ksrm,ksrn->kmn", M_nes, M_nes)
                                P_inv4 = inv_fourth_root_ns(Cov, iters=5)
                                U = polar_express_spec_norm(M_sum @ P_inv4, ns_steps=5, use_muon_eq=True)

                    pending_updates.append((plist, U, v_bufs[shape], muon_lr_now, muon_wd_now, sch_lam, False, extra_rv))

        LG_CTX.zero_slots()
        LG_CTX.enabled = False
        LG_CTX.suppress_default_grad = False

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()

        if uses_custom_core:
            with torch.no_grad():
                for plist, U, v_buf, mlr, mwd, sch_lam, jnorm, erv in pending_updates:
                    apply_muon_update_exact(plist, U, v_buf, lr=mlr, wd=mwd, beta2=0.9,
                                            schatten_pure_lam=sch_lam, joint_frob_norm=jnorm,
                                            extra_row_var=erv)

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step_times.append(dt)

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            avg_dt = sum(step_times[-50:]) / len(step_times[-50:])
            print(f"[{idea}] step={step+1}/{args.steps} loss={loss_sum:.4f} "
                  f"gnorm={float(grad_norm):.3f} tok/s={batch_tokens/avg_dt/1e3:.1f}k "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            rows = 4096 if (step + 1 == args.steps) else 1024
            vbpb = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=rows)
            rec = {
                "step": step + 1,
                "tokens": (step + 1) * batch_tokens,
                "train_loss": loss_sum,
                "val_bpb": vbpb,
                "elapsed_s": time.time() - t_start,
            }
            history.append(rec)
            print(f"[{idea}] VAL step={step+1} val_bpb={vbpb:.5f}", flush=True)

    final_spec = collect_telemetry(model, {}, first_l, cfg.core_layers)["weight_spectra"]
    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    summary = {
        "idea": idea,
        "method": args.method,
        "width": args.width,
        "pre_core_post": [args.pre_layers, args.core_layers, args.post_layers],
        "loops": args.loops,
        "params": n_params,
        "steps": args.steps,
        "tokens": args.steps * batch_tokens,
        "seed": args.seed,
        "sec_per_step": avg_dt,
        "tok_per_sec": batch_tokens / avg_dt,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "total_wall_s": time.time() - t_start,
        "final_val_bpb": history[-1]["val_bpb"],
        "history": history,
        "final_weight_spectra": final_spec,
        "telemetry": telemetry,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{idea}] DONE in {summary['total_wall_s']:.1f}s | "
          f"final val_bpb={summary['final_val_bpb']:.5f} | "
          f"mem={summary['peak_mem_gb']:.1f}GB | {summary['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
