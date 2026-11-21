"""Pure Algorithmic Loop-Muon Optimizer Arena
Built strictly on UNMODIFIED NextTransformer (Interactive-Active-AI/NextTransformer)
official looped baselines (fixed_recursive, deeploop, huginn, ouro).

ZERO architecture modifications. ZERO inference overhead. 100% default NextTransformer
MuonAdamW hyperparameters (except explicit Kaiming-He loop-scaling laws `kh_*`).

Active / canonical arms (20):
  - Baseline & non-loop controls: `nt_base`, `alg_pe10_s8`, `alg_whole_qk_pe10`
  - Whole-matrix CLNI family:     `xloop_q_clni_pe10`, `xloop_fejer_clni_pe10`,
                                  `xloop_whole_qk_q_clni`, `xloop_whole_qk_fejer_clni`
  - Kaiming-He simplicity pivot:  `kh_wd_R`, `kh_wd_2x`, `kh_wd_4x`, `kh_wd_8x`,
                                  `kh_schatten_R`, `kh_beta1_sqrtR`, `kh_loopeq_pe5`,
                                  `kh_bessel_pe5`, `kh_combo_2line`, `kh_coh_beta1`,
                                  `kh_loop_muon`, `kh_wd_bessel`, `kh_loop_muon_bessel`
(Historical Series D dead-end arms remain preserved in git history at commit 0eef94a.)
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

sys.path.insert(0, os.environ.get("NEXTTRANSFORMER_DIR", "./NextTransformer"))

from nanochat.optim import polar_express_coeffs
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import ReferenceLM, build_model
from nexttransformer.training.optimizer import ExperimentOptimizer


# ---------------------------------------------------------------------------
# Non-intrusive per-loop gradient capture (ZERO edits to ReferenceLM forward!)
# Safeguard (A): counts ONLY grad-carrying loops via torch.is_grad_enabled(),
# so Huginn stochastic depths (r_total < loops) NEVER drop early BPTT loops!
# ---------------------------------------------------------------------------
class _LoopGradCtx:
    def __init__(self):
        self.enabled = False
        self.suppress_default_grad = False
        self.grad_loop = -1            # 0..s-1 counting ONLY grad-carrying loops
        self.loop = None               # active grad loop index, or None outside core / no_grad
        self.active_loops = 0          # max grad-carrying loops observed in current forward
        self.slots = {}                # param -> {t: fp32 accumulated grad}

    def begin_forward(self):
        self.grad_loop = -1
        self.loop = None
        self.active_loops = 0

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


@torch._dynamo.disable
def _per_loop_fwd(self, x):
    if LG_CTX.enabled and LG_CTX.loop is not None and torch.is_grad_enabled() and self.weight.requires_grad:
        return _PerLoopLinear.apply(x, self.weight, self.weight)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def attach_nonintrusive_loop_tracker(model: ReferenceLM):
    """Wrap tied core linears + attach hooks on first/last core block to track grad-carrying loop
    index dynamically via torch.is_grad_enabled() (Huginn-safe, ZERO ReferenceLM edits)."""
    c = model.config
    first_l = c.pre_layers
    last_l = c.pre_layers + c.core_layers - 1

    @torch._dynamo.disable
    def _on_first_core_pre(_mod, _inp):
        if LG_CTX.enabled and torch.is_grad_enabled():
            LG_CTX.grad_loop += 1
            LG_CTX.loop = LG_CTX.grad_loop
            LG_CTX.active_loops = max(LG_CTX.active_loops, LG_CTX.grad_loop + 1)
        else:
            LG_CTX.loop = None

    @torch._dynamo.disable
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
# Polar Express + Fejer/Subspace-CLNI + Exact NorMuon / Cautious WD primitives
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
    Uses provable upper bound ||Y||_{S_8} = ||(Y Y^T)^2||_F^{1/4} >= sigma_max(Y) ALWAYS."""
    Xf = G.float()
    if use_muon_eq:
        target = Xf.norm(dim=(-2, -1), keepdim=True) / (Xf.size(-2) ** 0.5)
        row_norm = Xf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        Xf = Xf * (target / row_norm)
    Yf = Xf / (Xf.norm(dim=(-2, -1), keepdim=True) + 1e-8)
    Yb = Yf.bfloat16()
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


def polar_express_pe10_s8(G: torch.Tensor, use_muon_eq: bool = True, repeats: int = 2) -> torch.Tensor:
    """100%-TensorCore high-precision polar solver: Schatten-8 pre-norm + repeats x 5 Polar Express."""
    U = polar_express_spec_norm(G, ns_steps=5, use_muon_eq=use_muon_eq)
    for _ in range(repeats - 1):
        U = polar_express_spec_norm(U, ns_steps=5, use_muon_eq=False)
    return U


def fejer_clni_polar(
    M_nes: torch.Tensor,        # [K, s, m, n] Nesterov per-loop tensor
    M_ema: torch.Tensor,        # [K, s, m, n] pure EMA per-loop tensor
    q_cache: dict,
    key,
    step: int,
    repeats: int = 2,
    use_fejer: bool = True,
) -> torch.Tensor:
    """Fejer Outside-Only CLNI Polar (use_fejer=True) or Subspace-CLNI Polar (use_fejer=False).
    Exact bit-for-bit PE10_S8 at s <= 1; zero-slot invariant for stochastic Huginn depths."""
    K, s, m, n = M_nes.shape
    M_sum = M_nes.sum(dim=1)
    target = M_sum.norm(dim=(-2, -1), keepdim=True) / (m ** 0.5)
    row_norm = M_sum.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    eq = target / row_norm
    X_sum = M_sum * eq
    if s <= 1:
        U0 = polar_express_pe10_s8(X_sum, use_muon_eq=False, repeats=repeats)
        return U0 * ((float(min(m, n)) ** 0.5) / U0.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
    transposed = m > n
    if transposed:
        X_sum = X_sum.transpose(-2, -1)
    mm, nn = X_sum.size(-2), X_sum.size(-1)
    refresh_every = (2 if mm <= 768 else 4) if use_fejer else 5
    if (step % refresh_every == 0) or (key not in q_cache):
        src_t = M_ema if use_fejer else M_nes
        E_t = src_t * eq.unsqueeze(1)
        if transposed:
            E_t = E_t.transpose(-2, -1)
        E_sum = E_t.sum(dim=1)
        A = E_sum @ E_sum.mT
        tr = A.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        eye = torch.eye(mm, device=E_sum.device, dtype=torch.float32).unsqueeze(0)
        evals, U_eig = torch.linalg.eigh(A + (1e-6 * tr).unsqueeze(-1) * eye)
        sig = evals.clamp_min(1e-6 * tr).sqrt()
        V_eig = E_sum.mT @ (U_eig / sig.unsqueeze(-2))
        tmp = torch.einsum("kmi,ktmn->ktin", U_eig, E_t)
        proj_t = torch.einsum("ktin,kni->kti", tmp, V_eig)     # [K, s, mm]
        p_sum_sq = proj_t.sum(dim=1).square()                  # [K, mm]
        p_sq_sum = proj_t.square().sum(dim=1)                  # [K, mm]
        delta_neg = (p_sq_sum - p_sum_sq).clamp_min(0.0)
        w = (p_sum_sq / (p_sum_sq + delta_neg + 1e-12)).sqrt().clamp(0.05, 1.0)
        valid_m = (E_sum.square().sum(dim=(-2, -1)) > 1e-12).unsqueeze(-1)
        w = torch.where(valid_m, w, torch.ones_like(w))
        q_cache[key] = U_eig @ (w.unsqueeze(-1) * U_eig.mT)
    Q = q_cache[key]
    if use_fejer:
        U_raw = Q @ polar_express_pe10_s8(X_sum, use_muon_eq=False, repeats=repeats)
    else:
        U_raw = Q @ polar_express_pe10_s8(Q @ X_sum, use_muon_eq=False, repeats=repeats)
    scale = ((float(mm) ** 0.5) / (U_raw.norm(dim=(-2, -1), keepdim=True) + 1e-6)).clamp_max(1.5)
    out = U_raw * scale
    return out.transpose(-2, -1) if transposed else out


def _coherence_beta1(G_t: torch.Tensor, clni_state: dict, key, s_loops: int) -> torch.Tensor:
    """Parameter-free directional cross-loop coherence momentum beta_{1,i} = 0.95^{min(sqrt(s), 1/rho_i)}."""
    K, _s, m, _n = G_t.shape
    device = G_t.device
    if key not in clni_state:
        clni_state[key] = {
            "num": torch.zeros((K, m, 1), device=device, dtype=torch.float32),
            "den": torch.zeros((K, m, 1), device=device, dtype=torch.float32),
        }
    if s_loops <= 1:
        b1_row = torch.full((K, m, 1), 0.95, device=device, dtype=torch.float32)
    else:
        st = clni_state[key]
        G_hat = G_t / (G_t.norm(dim=-1, keepdim=True) + 1e-8)
        c_num = G_hat.sum(dim=1).square().sum(dim=-1, keepdim=True)
        c_den = G_hat.square().sum(dim=-1, keepdim=True).sum(dim=1)
        st["num"].lerp_(c_num, 0.1)
        st["den"].lerp_(c_den, 0.1)
        rho = (st["num"] / (st["den"] + 1e-8)).clamp_max(1.0)
        b1_row = 0.95 ** (1.0 / rho.clamp_min(1.0 / (float(s_loops) ** 0.5)))
    clni_state[key]["b1_mean"] = float(b1_row.mean().item())
    return b1_row


def apply_muon_update_exact(plist, U: torch.Tensor, v_buf: torch.Tensor,
                            lr: float, wd: float, beta2: float = 0.9,
                            skip_frob_renorm: bool = False,
                            cubic_cautious_wd: bool = False):
    """Exact nanochat Muon+ renorm + NorMuon variance reduction + Cautious WD (or cubic spectral WD)."""
    m, n = U.size(-2), U.size(-1)
    if not skip_frob_renorm:
        target_norm = min(m, n) ** 0.5
        U = U * (target_norm / U.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6))
    red_dim = -1 if m >= n else -2
    v_mean = U.square().mean(dim=red_dim, keepdim=True)
    red_dim_size = U.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size).sqrt()
    v_buf.lerp_(v_mean.to(v_buf.dtype), 1.0 - beta2)
    step_size = v_buf.float().clamp_min(1e-10).rsqrt()
    v_norm_new = ((v_mean * red_dim_size) * step_size.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    U = U * (step_size * (v_norm / v_norm_new.clamp_min(1e-10)))

    lr_eff = lr * (max(1.0, m / n) ** 0.5)
    P = torch.stack([p.float() for p in plist], dim=0)
    if cubic_cautious_wd:
        # Law 1b (pre-run audit #65): Scale-matched Cautious Cubic Spectral Decay (2 lines, 0 magic constants)
        denom = P.square().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8) / float(min(m, n))
        W_cub = ((P @ (P.mT @ P)) if m >= n else ((P @ P.mT) @ P)) / denom
        mask = ((U * W_cub) >= 0).float()
        P_new = P - lr_eff * U - lr_eff * wd * W_cub * mask
    else:
        mask = ((U * P) >= 0).float()
        P_new = P - lr_eff * U - lr_eff * wd * P * mask
    for idx_p, p in enumerate(plist):
        p.copy_(P_new[idx_p].to(p.dtype))


# ---------------------------------------------------------------------------
# Official NextTransformer MuonAdamW builder (100% DEFAULT hyperparams!)
# ---------------------------------------------------------------------------
def build_nt_optimizer(model: ReferenceLM, batch_tokens: int, core_param_ids: set,
                       exclude_core: bool = False):
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

    for shape in sorted(untied_mats):
        groups.append(dict(kind="muon", params=untied_mats[shape], lr=muon_lr, initial_lr=muon_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9,
                           weight_decay=wd, initial_weight_decay=wd))
    if not exclude_core:
        for shape in sorted(core_mats):
            groups.append(dict(kind="muon", params=core_mats[shape], lr=muon_lr, initial_lr=muon_lr,
                               momentum=0.95, ns_steps=5, beta2=0.9,
                               weight_decay=wd, initial_weight_decay=wd))
    return ExperimentOptimizer(groups), muon_lr, wd


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
    total_toks = 0.0
    try:
        for i in range(0, min(max_rows, val_x.shape[0]), batch_size):
            x = val_x[i:i + batch_size].to(device, dtype=torch.long)
            y = val_y[i:i + batch_size].to(device, dtype=torch.long)
            model.set_row_offset(i)
            LG_CTX.begin_forward()
            logits = model(x)
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction="none")
            nbytes = byte_table[y.flatten().clamp_min(0)]
            valid = (y.flatten() >= 0) & (nbytes > 0)
            total_nats += loss[valid].sum().item()
            total_bytes += nbytes[valid].sum().item()
            total_toks += valid.sum().item()
    finally:
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1.0)), total_nats / max(total_toks, 1.0)


VALID_IDEAS = {
    # Canonical baselines & non-loop controls
    "nt_base",
    "alg_pe10_s8",
    "alg_whole_qk_pe10",
    # Canonical whole-matrix CLNI family
    "xloop_q_clni_pe10",
    "xloop_fejer_clni_pe10",
    "xloop_whole_qk_q_clni",
    "xloop_whole_qk_fejer_clni",
    # Kaiming-He Simplicity Pivot (1-to-4-line plain 5-step PE laws)
    "kh_wd_R",
    "kh_wd_2x",
    "kh_wd_4x",
    "kh_wd_8x",
    "kh_schatten_R",
    "kh_beta1_sqrtR",
    "kh_loopeq_pe5",
    "kh_bessel_pe5",
    "kh_combo_2line",
    "kh_coh_beta1",
    "kh_loop_muon",
    "kh_wd_bessel",
    "kh_loop_muon_bessel",
}

NO_PER_LOOP_IDEAS = {
    "nt_base",
    "alg_pe10_s8",
    "alg_whole_qk_pe10",
    "kh_wd_R",
    "kh_wd_2x",
    "kh_wd_4x",
    "kh_wd_8x",
    "kh_schatten_R",
    "kh_beta1_sqrtR",
}


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

    idea = args.idea
    if idea not in VALID_IDEAS:
        raise ValueError(f"Unknown idea: {idea!r}. Valid ideas: {sorted(VALID_IDEAS)}")

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

    # Exact unmodified NextTransformer ModelConfig
    is_hug = (args.method == "huginn")
    bptt_loops = args.loops if not is_hug else min(4, args.loops)
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
        max_loops=max(args.loops, 2) if is_hug else args.loops,
        huginn_mean=float(max(args.loops, 2)),
        huginn_sigma=0.5,
        huginn_backprop_loops=max(bptt_loops, 1),
        eval_loops=max(args.loops, 2) if is_hug else args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    core_params = attach_nonintrusive_loop_tracker(model)
    core_param_ids = {id(cp) for cp in core_params}
    first_l = cfg.pre_layers
    s_loops = bptt_loops

    uses_custom_core = (idea != "nt_base")
    needs_per_loop = idea not in NO_PER_LOOP_IDEAS

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    batch_tokens = args.batch_size * seq_len
    opt, base_muon_lr, base_muon_wd = build_nt_optimizer(
        model, batch_tokens, core_param_ids, exclude_core=uses_custom_core,
    )

    core_by_shape = defaultdict(list)
    for cp in core_params:
        core_by_shape[tuple(cp.shape)].append(cp)

    core_by_proj = defaultdict(list)
    for layer in range(first_l, first_l + cfg.core_layers):
        blk = model.transformer.h[layer]
        for pname, mod in (("c_q", blk.attn.c_q), ("c_k", blk.attn.c_k), ("c_v", blk.attn.c_v),
                           ("c_proj", blk.attn.c_proj), ("c_fc", blk.mlp.c_fc), ("mlp_proj", blk.mlp.c_proj)):
            core_by_proj[pname].append(mod.weight)

    m_bufs = {}
    v_bufs = {}
    clni_state = {}
    q_cache = {}

    t_start = time.time()
    history = []
    telemetry = {}
    n_rows = train_x.shape[0]

    bpb0, vloss0 = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=512)
    print(f"[{idea}|{args.method}|w={args.width}|r={args.loops}] params={n_params/1e6:.1f}M "
          f"step=0 val_loss={vloss0:.4f} val_bpb={bpb0:.5f}", flush=True)

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
        # Keep default grad on step 0 for Safeguard (B) exact equivalence check; suppress on step > 0.
        LG_CTX.suppress_default_grad = needs_per_loop and (step > 0)

        loss_sum = 0.0
        active_loops_seen = 0
        for m_idx in range(accum):
            row0 = (step * args.batch_size + m_idx * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m_idx * args.micro_batch)
            LG_CTX.begin_forward()
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum
            if LG_CTX.enabled:
                active_loops_seen = max(active_loops_seen, LG_CTX.active_loops)

        # Safeguard (A): verify no grad-carrying loop slots were dropped on ANY step!
        if needs_per_loop:
            assert 1 <= active_loops_seen <= s_loops, (
                f"Step {step}: expected 1..{s_loops} grad loops, captured {active_loops_seen}"
            )
            for cp in core_params:
                got_keys = len(LG_CTX.slots.get(cp, {}))
                assert got_keys == active_loops_seen, (
                    f"Step {step}: missing loop grad slots ({got_keys} != {active_loops_seen})"
                )
            # Safeguard (B): Step-0 exact gradient sum equivalence against autograd cp.grad!
            if step == 0:
                max_eq_err = 0.0
                for cp in core_params:
                    g_slots = sum(LG_CTX.slots[cp].values())
                    err = float((g_slots - cp.grad.float()).abs().max().item())
                    max_eq_err = max(max_eq_err, err)
                    assert err < 2e-2, f"Step-0 per-loop grad mismatch: {err:.3e}"
                print(f"[{idea}] [SAFEGUARD A+B PASSED] active_loops={active_loops_seen}/{s_loops} "
                      f"max_grad_equiv_err={max_eq_err:.2e}", flush=True)

        if capture_telem:
            telemetry[f"step_{step+1}"] = collect_telemetry(model, LG_CTX.slots, first_l, cfg.core_layers)

        # Compute custom algorithmic updates for tied core matrices
        pending_updates = []  # tuples: (plist, U, v_buf, lr, wd, skip_frob, cubic_wd)
        if uses_custom_core and idea == "alg_whole_qk_pe10":
            with torch.no_grad():
                d = cfg.width
                q_ps, k_ps = core_by_proj["c_q"], core_by_proj["c_k"]
                L = len(q_ps)
                G_qk = torch.cat([
                    torch.stack([cp.grad.float() for cp in q_ps], dim=0),
                    torch.stack([cp.grad.float() for cp in k_ps], dim=0),
                ], dim=-1)  # [L, d, 2*d]
                if "wqk_qk" not in m_bufs:
                    m_bufs["wqk_qk"] = torch.zeros_like(G_qk)
                    v_bufs["wqk_qk"] = torch.zeros((2 * L, d, 1), device=device, dtype=torch.float32)
                m_bufs["wqk_qk"].lerp_(G_qk, 0.05)
                M_qk = G_qk.lerp(m_bufs["wqk_qk"], 0.95)
                U_joint = polar_express_pe10_s8(M_qk, use_muon_eq=True, repeats=2)
                U_qk = torch.cat([U_joint[:, :, :d], U_joint[:, :, d:]], dim=0).contiguous()
                pending_updates.append((q_ps + k_ps, U_qk, v_bufs["wqk_qk"], muon_lr_now, muon_wd_now, False, False))
                for pname in ("c_v", "c_proj", "c_fc", "mlp_proj"):
                    plist_sub = core_by_proj[pname]
                    mm, nn = plist_sub[0].shape
                    G_p = torch.stack([cp.grad.float() for cp in plist_sub], dim=0)
                    if f"wqk_{pname}" not in m_bufs:
                        m_bufs[f"wqk_{pname}"] = torch.zeros_like(G_p)
                        v_sh = (L, mm, 1) if mm >= nn else (L, 1, nn)
                        v_bufs[f"wqk_{pname}"] = torch.zeros(v_sh, device=device, dtype=torch.float32)
                    m_bufs[f"wqk_{pname}"].lerp_(G_p, 0.05)
                    M_p = G_p.lerp(m_bufs[f"wqk_{pname}"], 0.95)
                    U_p = polar_express_pe10_s8(M_p, use_muon_eq=True, repeats=2)
                    pending_updates.append((plist_sub, U_p, v_bufs[f"wqk_{pname}"], muon_lr_now, muon_wd_now, False, False))
        elif uses_custom_core and idea in ("xloop_whole_qk_q_clni", "xloop_whole_qk_fejer_clni"):
            use_fejer = (idea == "xloop_whole_qk_fejer_clni")
            with torch.no_grad():
                d = cfg.width
                q_ps, k_ps = core_by_proj["c_q"], core_by_proj["c_k"]
                L = len(q_ps)

                def _grab_q(plist_sub):
                    Gt = torch.stack(
                        [torch.stack([LG_CTX.slots.get(cp, {}).get(t, torch.zeros_like(cp, dtype=torch.float32))
                                      for t in range(s_loops)], dim=0) for cp in plist_sub],
                        dim=0,
                    )
                    Gs = Gt.sum(dim=1)
                    for idx_p, cp in enumerate(plist_sub):
                        cp.grad = Gs[idx_p].to(cp.dtype)
                    return Gt

                Gt_qk = torch.cat([_grab_q(q_ps), _grab_q(k_ps)], dim=-1)  # [L, s, d, 2*d]
                if "wqclni_qk" not in m_bufs:
                    m_bufs["wqclni_qk"] = torch.zeros_like(Gt_qk)
                    v_bufs["wqclni_qk"] = torch.zeros((2 * L, d, 1), device=device, dtype=torch.float32)
                m_bufs["wqclni_qk"].lerp_(Gt_qk, 0.05)
                M_qk = Gt_qk.lerp(m_bufs["wqclni_qk"], 0.95)
                U_joint = fejer_clni_polar(M_qk, m_bufs["wqclni_qk"], q_cache, "wqclni_qk",
                                           step, repeats=2, use_fejer=use_fejer)
                U_qk = torch.cat([U_joint[:, :, :d], U_joint[:, :, d:]], dim=0).contiguous() * (2.0 ** 0.5)
                pending_updates.append((q_ps + k_ps, U_qk, v_bufs["wqclni_qk"], muon_lr_now, muon_wd_now, True, False))
                for pname in ("c_v", "c_proj", "c_fc", "mlp_proj"):
                    plist_sub = core_by_proj[pname]
                    mm, nn = plist_sub[0].shape
                    Gt_p = _grab_q(plist_sub)
                    if f"wqclni_{pname}" not in m_bufs:
                        m_bufs[f"wqclni_{pname}"] = torch.zeros_like(Gt_p)
                        v_sh = (L, mm, 1) if mm >= nn else (L, 1, nn)
                        v_bufs[f"wqclni_{pname}"] = torch.zeros(v_sh, device=device, dtype=torch.float32)
                    m_bufs[f"wqclni_{pname}"].lerp_(Gt_p, 0.05)
                    M_p = Gt_p.lerp(m_bufs[f"wqclni_{pname}"], 0.95)
                    U_p = fejer_clni_polar(M_p, m_bufs[f"wqclni_{pname}"], q_cache, f"wqclni_{pname}",
                                           step, repeats=2, use_fejer=use_fejer)
                    pending_updates.append((plist_sub, U_p, v_bufs[f"wqclni_{pname}"], muon_lr_now, muon_wd_now, True, False))
        elif uses_custom_core:
            with torch.no_grad():
                for shape, plist in core_by_shape.items():
                    K, m, n = len(plist), shape[0], shape[1]
                    if shape not in v_bufs:
                        v_shape = (K, m, 1) if m >= n else (K, 1, n)
                        v_bufs[shape] = torch.zeros(v_shape, device=device, dtype=torch.float32)

                    skip_frob = False
                    wd_use = muon_wd_now
                    cubic_wd = False
                    if not needs_per_loop:
                        G_sum = torch.stack([cp.grad.float() for cp in plist], dim=0)  # [K, m, n]
                        if shape not in m_bufs:
                            m_bufs[shape] = torch.zeros_like(G_sum)
                        if idea in ("kh_wd_R", "kh_wd_2x", "kh_wd_4x", "kh_wd_8x"):
                            # Law 1a (1 line, plain 5-step PE, 0% overhead): Induced weight-sharing prior (R * wd & brackets)
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                            wd_mult = float(cfg.loops) if idea == "kh_wd_R" else {
                                "kh_wd_2x": 2.0, "kh_wd_4x": 4.0, "kh_wd_8x": 8.0,
                            }[idea]
                            wd_use = wd_mult * muon_wd_now
                        elif idea == "kh_schatten_R":
                            # Law 1b (2 lines, plain 5-step PE, 0% overhead): R * wd scale-matched cubic spectral decay
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                            wd_use = float(cfg.loops) * muon_wd_now
                            cubic_wd = True
                        elif idea == "kh_beta1_sqrtR":
                            # Law 2 (1 line, plain 5-step PE, 0% overhead): Jacobian-horizon core momentum beta1**sqrt(R)
                            b1 = 0.95 ** (float(cfg.loops) ** 0.5)
                            m_bufs[shape].lerp_(G_sum, 1.0 - b1)
                            M_nes = G_sum.lerp(m_bufs[shape], b1)
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                        elif idea == "alg_pe10_s8":
                            m_bufs[shape].lerp_(G_sum, 0.05)
                            M_nes = G_sum.lerp(m_bufs[shape], 0.95)
                            U = polar_express_pe10_s8(M_nes, use_muon_eq=True)
                        else:
                            raise ValueError(f"Unhandled non-per-loop idea: {idea}")
                    else:
                        G_t = torch.stack(
                            [torch.stack([LG_CTX.slots.get(cp, {}).get(t, torch.zeros_like(cp, dtype=torch.float32))
                                          for t in range(s_loops)], dim=0) for cp in plist],
                            dim=0,
                        )  # [K, s, m, n]
                        G_sum = G_t.sum(dim=1)
                        for idx_p, cp in enumerate(plist):
                            cp.grad = G_sum[idx_p].to(cp.dtype)

                        if idea in ("kh_coh_beta1", "kh_loop_muon"):
                            # pre-run audit #67: 3-Line Parameter-Free Directional Coherence Momentum (1x state, 5-step PE)
                            if shape not in m_bufs:
                                m_bufs[shape] = torch.zeros_like(G_sum)
                            b1_row = _coherence_beta1(G_t, clni_state, shape, s_loops)
                            m_bufs[shape].lerp_(G_sum, 1.0 - b1_row)
                            M_nes = G_sum.lerp(m_bufs[shape], b1_row)
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                            if idea == "kh_loop_muon":
                                wd_use = float(cfg.loops) * muon_wd_now
                        else:
                            if shape not in m_bufs:
                                m_bufs[shape] = torch.zeros_like(G_t)
                            if idea == "kh_loop_muon_bessel":
                                b1_4d = _coherence_beta1(G_t, clni_state, shape, s_loops).unsqueeze(1)
                                m_bufs[shape].lerp_(G_t, 1.0 - b1_4d)
                                M_nes = G_t.lerp(m_bufs[shape], b1_4d)
                            else:
                                m_bufs[shape].lerp_(G_t, 0.05)
                                M_nes = G_t.lerp(m_bufs[shape], 0.95)  # [K, s, m, n]

                            if idea in ("xloop_q_clni_pe10", "xloop_fejer_clni_pe10"):
                                U = fejer_clni_polar(M_nes, m_bufs[shape], q_cache, shape, step,
                                                     repeats=2, use_fejer=(idea == "xloop_fejer_clni_pe10"))
                                skip_frob = True
                            elif idea == "kh_loopeq_pe5":
                                # Law 3 (1 line, plain 5-step PE, ZERO eigh): Cancellation-safe LoopEq
                                M_sum = M_nes.sum(dim=1)
                                row_norm = M_nes.square().sum(dim=-1, keepdim=True).sum(dim=1).sqrt().clamp_min(1e-6)
                                target = row_norm.square().mean(dim=-2, keepdim=True).sqrt()
                                U = polar_express_raw(M_sum * (target / row_norm), ns_steps=5, use_muon_eq=False)
                            elif idea in ("kh_bessel_pe5", "kh_combo_2line", "kh_wd_bessel", "kh_loop_muon_bessel"):
                                # Law 4 (3 lines, plain 5-step PE, ZERO eigh): Bessel Cross-Loop Polar
                                M_sum = M_nes.sum(dim=1)
                                target = M_sum.norm(dim=(-2, -1), keepdim=True) / (m ** 0.5)
                                eq = target / M_sum.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                                X_t = M_nes * eq.unsqueeze(1)
                                X_sum = X_t.sum(dim=1)
                                if s_loops <= 1:
                                    U = polar_express_raw(X_sum, ns_steps=5, use_muon_eq=False)
                                else:
                                    transposed = (m > n)
                                    if transposed:
                                        X_t = X_t.transpose(-2, -1)
                                        X_sum = X_sum.transpose(-2, -1)
                                    mm, nn = X_sum.size(-2), X_sum.size(-1)
                                    X_dev = (X_t - X_t.mean(dim=1, keepdim=True)) / ((float(s_loops) - 1.0) ** 0.5)
                                    X_aug = torch.cat([X_sum, X_dev.permute(0, 2, 1, 3).reshape(K, mm, s_loops * nn)], dim=-1)
                                    U_raw = polar_express_raw(X_aug, ns_steps=5, use_muon_eq=False)[:, :, :nn]
                                    scale = ((float(mm) ** 0.5) / (U_raw.norm(dim=(-2, -1), keepdim=True) + 1e-6)).clamp_max(1.5)
                                    U = U_raw * scale
                                    if transposed:
                                        U = U.transpose(-2, -1)
                                    skip_frob = True
                                if idea == "kh_combo_2line":
                                    wd_use = float(cfg.loops) * muon_wd_now
                                    cubic_wd = True
                                elif idea in ("kh_wd_bessel", "kh_loop_muon_bessel"):
                                    wd_use = float(cfg.loops) * muon_wd_now
                            else:
                                raise ValueError(f"Unhandled per-loop idea: {idea}")

                    pending_updates.append((plist, U, v_bufs[shape], muon_lr_now, wd_use, skip_frob, cubic_wd))

        LG_CTX.zero_slots()
        LG_CTX.enabled = False
        LG_CTX.suppress_default_grad = False

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()

        if uses_custom_core:
            with torch.no_grad():
                for plist, U, v_buf, mlr, mwd, sf, cwd in pending_updates:
                    apply_muon_update_exact(plist, U, v_buf, lr=mlr, wd=mwd, beta2=0.9,
                                            skip_frob_renorm=sf, cubic_cautious_wd=cwd)

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
            vbpb, vloss = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=rows)
            b1_vals = [v["b1_mean"] for v in clni_state.values() if isinstance(v, dict) and "b1_mean" in v]
            b1_avg = (sum(b1_vals) / len(b1_vals)) if b1_vals else None
            rec = {
                "step": step + 1,
                "tokens": (step + 1) * batch_tokens,
                "train_loss": loss_sum,
                "val_loss": vloss,
                "val_bpb": vbpb,
                "elapsed_s": time.time() - t_start,
            }
            if b1_avg is not None:
                rec["b1_mean"] = round(b1_avg, 5)
            history.append(rec)
            b1_str = f" b1_mean={b1_avg:.4f}" if b1_avg is not None else ""
            print(f"[{idea}] VAL step={step+1} val_loss={vloss:.5f} val_bpb={vbpb:.5f}{b1_str}", flush=True)

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
        "final_val_loss": history[-1]["val_loss"],
        "final_val_bpb": history[-1]["val_bpb"],
        "history": history,
        "final_weight_spectra": final_spec,
        "telemetry": telemetry,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{idea}] DONE in {summary['total_wall_s']:.1f}s | "
          f"final val_loss={summary['final_val_loss']:.5f} val_bpb={summary['final_val_bpb']:.5f} | "
          f"mem={summary['peak_mem_gb']:.1f}GB | {summary['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
