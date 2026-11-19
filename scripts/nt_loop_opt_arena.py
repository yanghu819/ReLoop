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
                            schatten_pure_lam: float = 0.0):
    """Exact nanochat Muon+ renorm + NorMuon variance reduction + Cautious WD (or Schatten-4 WD)."""
    m, n = U.size(-2), U.size(-1)
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
    }
    if idea not in VALID_IDEAS:
        raise ValueError(f"Unknown idea: {idea}")

    uses_custom_core = (idea != "nt_base")
    needs_per_loop = idea not in ("nt_base", "nt_schatten_pure", "nt_loop_dual_mom")

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    batch_tokens = args.batch_size * seq_len
    opt, base_muon_lr, base_muon_wd = build_nt_optimizer(
        model, batch_tokens, core_param_ids, exclude_core=uses_custom_core
    )

    core_by_shape = defaultdict(list)
    for cp in core_params:
        core_by_shape[tuple(cp.shape)].append(cp)

    m_bufs = {}
    m_bufs_slow = {}
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

        opt.zero_grad(set_to_none=True)
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
        if uses_custom_core:
            with torch.no_grad():
                for shape, plist in core_by_shape.items():
                    K, m, n = len(plist), shape[0], shape[1]
                    if shape not in v_bufs:
                        v_shape = (K, m, 1) if m >= n else (K, 1, n)
                        v_bufs[shape] = torch.zeros(v_shape, device=device, dtype=torch.float32)

                    sch_lam = 0.0
                    if not needs_per_loop:
                        G_sum = torch.stack([cp.grad.float() for cp in plist], dim=0)  # [K, m, n]
                        if shape not in m_bufs:
                            m_bufs[shape] = torch.zeros_like(G_sum)
                            if idea == "nt_loop_dual_mom":
                                m_bufs_slow[shape] = torch.zeros_like(G_sum)
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

                        elif idea == "nt_loop_karcher_pe":
                            # Riemannian Karcher / Fréchet mean on Stiefel manifold: PE(sum_t w_t PE(M_t))!
                            sl_norms = M_nes.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
                            sl_w = sl_norms / sl_norms.sum(dim=1, keepdim=True).clamp_min(1e-8)
                            U_sl = polar_express_raw(M_nes.reshape(K * s_loops, m, n), 5, True).view(K, s_loops, m, n)
                            U_avg = (U_sl * sl_w).sum(dim=1)
                            U = polar_express_raw(U_avg, 5, False)

                    pending_updates.append((plist, U, v_bufs[shape], muon_lr_now, muon_wd_now, sch_lam))

        LG_CTX.zero_slots()
        LG_CTX.enabled = False
        LG_CTX.suppress_default_grad = False

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()

        if uses_custom_core:
            with torch.no_grad():
                for plist, U, v_buf, mlr, mwd, sch_lam in pending_updates:
                    apply_muon_update_exact(plist, U, v_buf, lr=mlr, wd=mwd, beta2=0.9,
                                            schatten_pure_lam=sch_lam)

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
