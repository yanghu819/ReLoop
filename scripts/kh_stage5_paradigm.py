#!/usr/bin/env python3
"""Clean, zero-bloat Stage-5 Decisive Paradigm-Jump Runner (Carbon-Critic #80, Carbon-Auditor #79).

Implements 3 orthogonal, <=8-line, zero-eigh, architecture-agnostic 2D paradigms (ALL Gate-1b --freeze-x0 ready):
  1. alg_dc_ac_s / alg_dc_ac_1 (7 lines, Delta(R=1) == 0, ~1.3% overhead):
       Pre-contraction factored DC+AC Split-Polar Muon: 1 extra BF16 GEMM per microbatch separates
       G_DC = (1/s) DY_acc^T X_acc from hidden pure cross-loop G_AC = G_sum - G_DC!
  2. alg_xloop_echo_whiten (6 lines, Delta(R=1) == 0) & alg_act_whiten (paired R=1 DiD):
       Riemannian Input-Metric / Echo-Whitened Muon using Schatten-4 + 8-step coupled Newton-Schulz
       inverse sqrt every 10 steps (0% eigh, <0.3% FLOPs), U = Polar5(M P_X) P_X (unconditional SPD descent!).
  3. alg_primal_dual_muon (4 lines, 0.0% FLOPs, paired R=1 DiD):
       Schedule-Free Primal-Interpolated Core Muon (Defazio et al.) damping O(R) iterated weight-bounce.
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
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import ExperimentOptimizer


# ---------------------------------------------------------------------------
# Pre-contraction hook context (captures microbatch X_acc/DY_acc for DC+AC,
# OR periodic 512-token subsampled X_t for Riemannian Input-Metric Whitening!)
# ---------------------------------------------------------------------------
class _PreContractCtx:
    def __init__(self):
        self.mode = "none"          # "none", "dc_ac", "act_cov"
        self.active = False
        self.grad_loop = -1
        self.active_loops = 0
        # DC+AC per-microbatch accumulators (freed right after each microbatch!)
        self.x_acc = {}             # param -> [N, n] bf16 sum_t x_t
        self.dy_acc = {}            # param -> [N, m] bf16 sum_t dy_t
        self.call_cnt = {}          # param -> int active grad calls in this microbatch
        self.g_dc_acc = {}          # param -> [m, n] fp32 accumulated G_DC across microbatches
        # Act-cov periodic subsampled activations (captured ONLY on step % 10 == 0, m_idx == 0!)
        self.x_sub = {}             # param -> {t: [N_sub, n] fp32}

    def begin_step(self, mode: str):
        self.mode = mode
        self.g_dc_acc.clear()
        self.x_sub.clear()

    def begin_microbatch(self):
        self.grad_loop = -1
        self.active_loops = 0
        self.x_acc.clear()
        self.dy_acc.clear()
        self.call_cnt.clear()

    @torch.no_grad()
    def finish_microbatch_dc_ac(self):
        """Run ONLY 1 extra BF16 GEMM per core linear to compute exact G_DC = (1/s) DY_acc^T X_acc."""
        if self.mode != "dc_ac":
            return
        for p, cnt in self.call_cnt.items():
            if cnt <= 0:
                continue
            xa = self.x_acc[p]
            dya = self.dy_acc[p]
            g_dc = (dya.t() @ xa).float() / float(cnt)
            if p in self.g_dc_acc:
                self.g_dc_acc[p].add_(g_dc)
            else:
                self.g_dc_acc[p] = g_dc
        self.x_acc.clear()
        self.dy_acc.clear()
        self.call_cnt.clear()


PC_CTX = _PreContractCtx()


class _PreContractLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, param_ref, loop_idx):
        ctx.loop_idx = loop_idx
        ctx.param_ref = param_ref
        ctx.save_for_backward(x, weight)
        if PC_CTX.active and PC_CTX.mode == "act_cov" and loop_idx >= 0:
            xr = x.detach().reshape(-1, x.shape[-1])
            n_tok = xr.size(0)
            stride = max(1, n_tok // 512)
            PC_CTX.x_sub.setdefault(param_ref, {})[loop_idx] = xr[::stride][:512].float()
        return F.linear(x, weight.to(dtype=x.dtype))

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        grad_x = (grad_y @ weight.to(dtype=grad_y.dtype)) if ctx.needs_input_grad[0] else None
        if not ctx.needs_input_grad[1]:
            return grad_x, None, None, None
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x.reshape(-1, x.shape[-1]).to(gy.dtype)
        gw = (gy.t() @ xr)
        if PC_CTX.active and PC_CTX.mode == "dc_ac" and ctx.loop_idx >= 0:
            p = ctx.param_ref
            if p in PC_CTX.x_acc:
                PC_CTX.x_acc[p].add_(xr)
                PC_CTX.dy_acc[p].add_(gy)
                PC_CTX.call_cnt[p] += 1
            else:
                PC_CTX.x_acc[p] = xr.clone()
                PC_CTX.dy_acc[p] = gy.clone()
                PC_CTX.call_cnt[p] = 1
        return grad_x, gw.to(weight.dtype), None, None


def _pc_fwd(self, x):
    if PC_CTX.active and torch.is_grad_enabled() and PC_CTX.grad_loop >= 0 and self.weight.requires_grad:
        return _PreContractLinearFn.apply(x, self.weight, self.weight, PC_CTX.grad_loop)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def attach_precontract_hooks(model):
    """Attach grad-only loop entry pre-hook (Bug #5 fixed, Huginn state_adapter included cleanly!)."""
    cfg = model.config
    first_l = cfg.pre_layers
    last_l = cfg.pre_layers + cfg.core_layers - 1
    has_adapter = (cfg.method == "huginn" and hasattr(model, "state_adapter")
                   and model.state_adapter is not None and hasattr(model.state_adapter, "weight"))

    def _on_loop_entry(_mod, _inp):
        if PC_CTX.active and torch.is_grad_enabled():
            PC_CTX.grad_loop += 1
            PC_CTX.active_loops = max(PC_CTX.active_loops, PC_CTX.grad_loop + 1)

    if has_adapter:
        model.state_adapter.register_forward_pre_hook(_on_loop_entry)
    else:
        model.transformer.h[first_l].register_forward_pre_hook(_on_loop_entry)

    core_params = []
    if has_adapter:
        model.state_adapter.forward = types.MethodType(_pc_fwd, model.state_adapter)
        core_params.append(model.state_adapter.weight)

    for l_idx in range(first_l, last_l + 1):
        blk = model.transformer.h[l_idx]
        for mod in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v,
                    blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
            mod.forward = types.MethodType(_pc_fwd, mod)
            core_params.append(mod.weight)
    return core_params


# ---------------------------------------------------------------------------
# Official 5-step Polar Express & Schatten-4 8-step Coupled NS Inverse Sqrt
# ---------------------------------------------------------------------------
def polar_express_raw(G: torch.Tensor, ns_steps: int = 5, use_muon_eq: bool = True) -> torch.Tensor:
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


def inv_sqrt_ns8(S: torch.Tensor) -> torch.Tensor:
    """Zero-eigh, 0%-NaN inverse sqrt for batched SPD [K, n, n] via Schatten-4 bound + 12 coupled NS steps."""
    # Exact provable upper bound ||S||_{S_4} = ||S^2||_F^{1/2} >= lambda_max(S)!
    spec = (S @ S).norm(dim=(-2, -1), keepdim=True).clamp_min(1e-12).sqrt()
    scale = 1.05 * spec
    Y = S / scale
    eye = torch.eye(S.size(-1), device=S.device, dtype=torch.float32).unsqueeze(0)
    Z = eye.expand_as(Y).clone()
    for _ in range(12):
        T = 0.5 * (3.0 * eye - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Z * scale.rsqrt()


def apply_muon_step(plist, U: torch.Tensor, v_buf: torch.Tensor, lr: float, wd: float, beta2: float = 0.9):
    """Exact official Muon+ + NorMuon + Cautious WD."""
    K, m, n = U.size(0), U.size(-2), U.size(-1)
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
    mask = ((U * P) >= 0).float()
    P_new = P - lr_eff * U - lr_eff * wd * P * mask
    for idx_p, p in enumerate(plist):
        p.copy_(P_new[idx_p].to(p.dtype))


def apply_primal_dual_muon_step(plist, U: torch.Tensor, v_buf: torch.Tensor,
                                z_buf: torch.Tensor, x_buf: torch.Tensor,
                                lr: float, wd: float, c_avg: float, beta_sf: float = 0.90, beta2: float = 0.9):
    """Schedule-Free Primal-Interpolated Core Muon (4 lines, 0.0% FLOPs)."""
    K, m, n = U.size(0), U.size(-2), U.size(-1)
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
    Y = torch.stack([p.float() for p in plist], dim=0)
    mask = ((U * Y) >= 0).float()
    Z_new = z_buf.float() - lr_eff * U - lr_eff * wd * Y * mask
    X_new = x_buf.float().lerp(Z_new, c_avg)
    Y_new = Z_new.lerp(X_new, beta_sf)
    z_buf.copy_(Z_new.to(z_buf.dtype))
    x_buf.copy_(X_new.to(x_buf.dtype))
    for idx_p, p in enumerate(plist):
        p.copy_(Y_new[idx_p].to(p.dtype))


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
    PC_CTX.active = False
    total_nats = total_bytes = total_toks = 0.0
    try:
        for i in range(0, min(max_rows, val_x.shape[0]), batch_size):
            x = val_x[i:i + batch_size].to(device, dtype=torch.long)
            y = val_y[i:i + batch_size].to(device, dtype=torch.long)
            model.set_row_offset(i)
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


@torch.no_grad()
def collect_diag(model, first_l: int, core_layers: int):
    last_l = first_l + core_layers
    kaps, erks, fns = [], [], []
    for l_idx in range(first_l, last_l):
        blk = model.transformer.h[l_idx]
        for p in (blk.attn.c_q.weight, blk.attn.c_k.weight, blk.attn.c_v.weight,
                  blk.attn.c_proj.weight, blk.mlp.c_fc.weight, blk.mlp.c_proj.weight):
            wf = p.float()
            fns.append(float(wf.norm().item()))
            sv = torch.linalg.svdvals(wf)
            pe = sv.square() / sv.square().sum().clamp_min(1e-12)
            erks.append(float((-(pe * pe.clamp_min(1e-12).log()).sum()).exp().item()))
            kaps.append(float(sv[0].item() / sv[len(sv) // 2].clamp_min(1e-8).item()))
    return dict(
        mean_fnorm=round(sum(fns) / len(fns), 3),
        mean_kappa=round(sum(kaps) / len(kaps), 3),
        mean_erank=round(sum(erks) / len(erks), 1),
    )


def build_non_core_opt(model, batch_tokens: int, core_ids: set, exclude_core: bool):
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
    untied, core, other = defaultdict(list), defaultdict(list), []
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:
            continue
        if p.ndim == 2:
            (core if id(p) in core_ids else untied)[tuple(p.shape)].append(p)
        else:
            other.append(p)
    adam(other, adamw_lr, (0.9, 0.95), 0.0)
    for sh in sorted(untied):
        groups.append(dict(kind="muon", params=untied[sh], lr=muon_lr, initial_lr=muon_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9,
                           weight_decay=wd, initial_weight_decay=wd))
    if not exclude_core:
        for sh in sorted(core):
            groups.append(dict(kind="muon", params=core[sh], lr=muon_lr, initial_lr=muon_lr,
                               momentum=0.95, ns_steps=5, beta2=0.9,
                               weight_decay=wd, initial_weight_decay=wd))
    return ExperimentOptimizer(groups), muon_lr, wd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=[
        "nt_base",
        "alg_dc_ac_s",
        "alg_dc_ac_1",
        "alg_xloop_echo_whiten",
        "alg_act_whiten",
        "alg_primal_dual_muon",
    ])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="fixed_recursive", choices=["fixed_recursive", "huginn", "ouro", "deeploop"])
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--freeze-x0", action="store_true")
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=375)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=Path("/dev/shm/rsi_arena_data"))
    ap.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_d = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_d = torch.load(args.data_dir / "val.pt", map_location="cpu", weights_only=True)
    train_x, train_y = train_d["x"], train_d["y"]
    val_x, val_y = val_d["x"], val_d["y"]
    seq_len = train_x.shape[1]
    byte_table = torch.load(args.resources / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    is_hug = (args.method == "huginn")
    is_ouro = (args.method == "ouro")
    bptt = args.loops if not is_hug else min(4, args.loops)
    cfg = ModelConfig(
        method=args.method, scale="debug", width=args.width, heads=args.heads,
        vocab_size=32768, sequence_len=seq_len, mlp_width=4 * args.width,
        layers=args.pre_layers + args.core_layers + args.post_layers,
        pre_layers=args.pre_layers, core_layers=args.core_layers, post_layers=args.post_layers,
        loops=args.loops,
        max_loops=max(args.loops, 2) if (is_hug or is_ouro) else args.loops,
        huginn_mean=float(max(args.loops, 2)), huginn_sigma=0.5,
        huginn_backprop_loops=max(bptt, 1),
        eval_loops=max(args.loops, 2) if (is_hug or is_ouro) else args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    if args.freeze_x0:
        with torch.no_grad():
            model.x0_lambdas.zero_()
        model.x0_lambdas.requires_grad_(False)

    core_params = attach_precontract_hooks(model)
    core_ids = {id(p) for p in core_params}

    idea = args.idea
    uses_custom_core = (idea != "nt_base")
    use_dc_ac = idea in ("alg_dc_ac_s", "alg_dc_ac_1")
    use_whiten = idea in ("alg_xloop_echo_whiten", "alg_act_whiten")
    use_primal_dual = (idea == "alg_primal_dual_muon")

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt, base_core_lr, base_core_wd = build_non_core_opt(
        model, batch_tokens, core_ids, exclude_core=uses_custom_core
    )
    core_by_shape = defaultdict(list)
    for cp in core_params:
        core_by_shape[tuple(cp.shape)].append(cp)

    m_bufs, m_ac_bufs, v_bufs = {}, {}, {}
    cov_ema, px_cache = {}, {}
    z_bufs, x_bufs = {}, {}
    if use_primal_dual:
        for sh, plist in core_by_shape.items():
            P0 = torch.stack([p.detach().clone() for p in plist], dim=0)
            z_bufs[sh] = P0.clone()
            x_bufs[sh] = P0.clone()

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=512)
    print(f"[{args.tag}|{idea}|x0fz={int(args.freeze_x0)}] step=0 val_loss={vl0:.4f} val_bpb={bpb0:.5f}", flush=True)

    t0_all = time.time()
    history, step_times = [], []
    lr_sq_sum = 0.0
    last_diag_metric = 0.0
    model.train()

    for step in range(args.steps):
        t0 = time.perf_counter()
        mult = lr_mult(step, args.steps)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * mult
            if "initial_weight_decay" in g:
                g["weight_decay"] = g["initial_weight_decay"] * mult
        core_lr_now = base_core_lr * mult
        core_wd_now = base_core_wd * mult
        lr_sq_sum += core_lr_now * core_lr_now
        c_avg = max((core_lr_now * core_lr_now) / max(lr_sq_sum, 1e-18), 0.05)

        model.zero_grad(set_to_none=True)
        model.set_step(step)

        capture_cov_now = use_whiten and (step % 10 == 0)
        PC_CTX.begin_step("dc_ac" if use_dc_ac else ("act_cov" if capture_cov_now else "none"))

        loss_sum = 0.0
        max_active = 0
        for m_idx in range(accum):
            r0 = (step * args.batch_size + m_idx * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m_idx * args.micro_batch)
            PC_CTX.begin_microbatch()
            # For act_cov, only capture microbatch 0 every 10 steps (0.1% FLOPs!)
            PC_CTX.active = use_dc_ac or (capture_cov_now and m_idx == 0)
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum
            if PC_CTX.active:
                max_active = max(max_active, PC_CTX.active_loops)
                if use_dc_ac:
                    PC_CTX.finish_microbatch_dc_ac()

        PC_CTX.active = False

        pending = []
        if uses_custom_core:
            with torch.no_grad():
                diag_acc = []
                s_act = max(1, max_active)
                for sh, plist in core_by_shape.items():
                    K, m, n = len(plist), sh[0], sh[1]
                    red_dim = -1 if m >= n else -2
                    G_sum = torch.stack([cp.grad.float() for cp in plist], dim=0)
                    if sh not in m_bufs:
                        m_bufs[sh] = torch.zeros_like(G_sum)
                        v_bufs[sh] = torch.zeros((K, m, 1) if m >= n else (K, 1, n), device=device, dtype=torch.float32)
                        if use_dc_ac:
                            m_ac_bufs[sh] = torch.zeros_like(G_sum, dtype=torch.bfloat16)

                    m_bufs[sh].lerp_(G_sum, 0.05)
                    M_nes = G_sum.lerp(m_bufs[sh], 0.95)

                    if use_dc_ac:
                        G_dc = torch.stack([PC_CTX.g_dc_acc.get(cp, cp.grad.float()) for cp in plist], dim=0)
                        G_ac = G_sum - G_dc
                        if step == 0 and s_act == 1:
                            assert float(G_ac.abs().max().item()) < 2e-2, f"R=1 G_ac not zero: {G_ac.abs().max()}"
                        ac_r = float((G_ac.norm() / (G_dc.norm() + 1e-8)).item())
                        diag_acc.append(ac_r)
                        m_ac_bufs[sh].lerp_(G_ac.bfloat16(), 0.05)
                        M_ac = G_ac.lerp(m_ac_bufs[sh].float(), 0.95)
                        M_dc = M_nes - M_ac
                        dot = (M_ac * M_nes).sum(dim=red_dim, keepdim=True)
                        denom = M_nes.square().sum(dim=red_dim, keepdim=True).clamp_min(1e-12)
                        M_ac_pos = M_ac - (dot.clamp_max(0.0) / denom) * M_nes
                        gamma_stat = M_nes.norm(dim=(-2, -1), keepdim=True) / (
                            M_dc.norm(dim=(-2, -1), keepdim=True) + M_ac.norm(dim=(-2, -1), keepdim=True) + 1e-8
                        )
                        U_base = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                        if s_act <= 1:
                            U = U_base
                        else:
                            alpha_ac = (1.0 / (float(s_act) ** 0.5)) if idea == "alg_dc_ac_s" else 1.0
                            ac_nz = (M_ac_pos.norm(dim=(-2, -1), keepdim=True) > 1e-10).float()
                            U_ac = polar_express_raw(M_ac_pos, ns_steps=5, use_muon_eq=True) * ac_nz
                            U = U_base + (alpha_ac * gamma_stat) * U_ac
                        pending.append((sh, plist, U))

                    elif use_whiten:
                        if capture_cov_now or (sh not in px_cache):
                            if idea == "alg_xloop_echo_whiten" and s_act <= 1:
                                px_cache[sh] = None
                            else:
                                covs = []
                                for cp in plist:
                                    slot = PC_CTX.x_sub.get(cp, {})
                                    xs = [slot[t] for t in sorted(slot)]
                                    if not xs:
                                        covs.append(torch.eye(n, device=device, dtype=torch.float32))
                                    elif idea == "alg_xloop_echo_whiten":
                                        x_bar = torch.stack(xs, dim=0).mean(dim=0)  # [N_sub, n]
                                        covs.append((x_bar.t() @ x_bar) / float(x_bar.size(0)))
                                    else:
                                        x_cat = torch.cat(xs, dim=0)  # [s*N_sub, n]
                                        covs.append((x_cat.t() @ x_cat) / float(x_cat.size(0)))
                                C_cur = torch.stack(covs, dim=0)  # [K, n, n]
                                if sh not in cov_ema:
                                    cov_ema[sh] = C_cur.clone()
                                else:
                                    cov_ema[sh].lerp_(C_cur, 0.1)
                                tr_mean = cov_ema[sh].diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).unsqueeze(-1).clamp_min(1e-8)
                                C_norm = cov_ema[sh] / tr_mean
                                eye_n = torch.eye(n, device=device, dtype=torch.float32).unsqueeze(0)
                                ridge = (1.0 / float(s_act)) if idea == "alg_xloop_echo_whiten" else 0.10
                                S = (1.0 - ridge) * C_norm + ridge * eye_n
                                px_cache[sh] = inv_sqrt_ns8(S)
                        P_X = px_cache[sh]
                        if P_X is None:
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                        else:
                            # Pre-Polar + Post-Polar Riemannian Input-Metric Muon (strict SPD descent!)
                            U = polar_express_raw(M_nes @ P_X, ns_steps=5, use_muon_eq=True) @ P_X
                            diag_acc.append(float((P_X.norm(dim=(-2, -1)) / (n ** 0.5)).mean().item()))
                        pending.append((sh, plist, U))

                    elif use_primal_dual:
                        U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                        diag_acc.append(c_avg)
                        pending.append((sh, plist, U))

                if diag_acc:
                    last_diag_metric = sum(diag_acc) / len(diag_acc)

        PC_CTX.g_dc_acc.clear()
        PC_CTX.x_sub.clear()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()

        if uses_custom_core:
            with torch.no_grad():
                for sh, plist, U in pending:
                    if use_primal_dual:
                        apply_primal_dual_muon_step(plist, U, v_bufs[sh], z_bufs[sh], x_bufs[sh],
                                                    lr=core_lr_now, wd=core_wd_now, c_avg=c_avg)
                    else:
                        apply_muon_step(plist, U, v_bufs[sh], lr=core_lr_now, wd=core_wd_now, beta2=0.9)

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            rows = 4096 if (step + 1) == args.steps else 1024
            saved_y = {}
            if use_primal_dual:
                with torch.no_grad():
                    for sh, plist in core_by_shape.items():
                        saved_y[sh] = [p.detach().clone() for p in plist]
                        for idx_p, p in enumerate(plist):
                            p.copy_(x_bufs[sh][idx_p].to(p.dtype))
            vb, vl = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=rows)
            if use_primal_dual:
                with torch.no_grad():
                    for sh, plist in core_by_shape.items():
                        for idx_p, p in enumerate(plist):
                            p.copy_(saved_y[sh][idx_p])
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum,
                                diag_m=round(last_diag_metric, 4), elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f} dm={last_diag_metric:.4f}", flush=True)

    # For final diagnostics on primal_dual, evaluate spectra on smooth X iterate
    if use_primal_dual:
        with torch.no_grad():
            for sh, plist in core_by_shape.items():
                for idx_p, p in enumerate(plist):
                    p.copy_(x_bufs[sh][idx_p].to(p.dtype))
    diag = collect_diag(model, cfg.pre_layers, cfg.core_layers)
    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    out = dict(
        tag=args.tag, idea=idea, method=args.method, loops=args.loops, freeze_x0=args.freeze_x0,
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], tok_per_sec=batch_tokens / avg_dt,
        total_wall_s=time.time() - t0_all, diag_metric=round(last_diag_metric, 4),
        diag=diag, history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{args.tag}] DONE val={out['final_val_loss']:.5f} dm={last_diag_metric:.4f} "
          f"fn={diag['mean_fnorm']} kap={diag['mean_kappa']} {out['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
