#!/usr/bin/env python3
"""Clean, zero-bloat Stage-4 Runner for internal review #75's Architecture-Agnostic 2D Laws:
  - nt_base: Official 5-step MuonAdamW (with optional --core-lr-mult, --core-wd-mult, --freeze-x0)
  - alg_jfb_cap_pe5 (2 lines, 1.000x mem, 5-step PE): Readout-Anchored Non-Expansive Jacobian Cap
        w_t = min(1.0, n_{s-1} / (n_t + 1e-8)), G_tilde = sum_t w_t G_t, full Muon+/NorMuon norm!
  - alg_jfb_equal_pe5 (diagnostic): Equal-norm per-loop weighting w_t = n_{s-1} / (n_t + 1e-8) (uncapped)
  - alg_tbptt_s4 (diagnostic): Causal test — zero G_0..G_{s-5}, keep only readout loops s-4..s-1
  - alg_tau_beta1 (2 lines, 1.002x mem, 5-step PE): Geometric-Mean Coherence Momentum
        tau_i = min(1.0, kappa_i / sqrt(s_act)), beta_{1,i} = beta_1^(1 / max(tau_i, 1/sqrt(s_act)))
  - alg_jfb_tau_muon (4 lines, 1.002x mem, 5-step PE): Unified Law A + Law B (R=1 exact, zero eigh!)

Includes:
  - Bug #5 fixed from line 1 (dynamic torch.is_grad_enabled() loop counter)
  - Huginn state_adapter.weight properly included in core_params (with pre-hook placed BEFORE state_adapter!)
  - Step-0 exact gradient equivalence assertion (Safeguard B) on ALL core_params (including state_adapter!)
  - Optional --freeze-x0 for internal audit Gate 1b (universal x0_lambdas-free Transformer control!)
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
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import ExperimentOptimizer


class _GradLoopCtx:
    def __init__(self):
        self.enabled = False
        self.suppress_default_grad = False
        self.grad_loop = -1
        self.active_loops = 0
        self.slots = {}

    def begin_forward(self):
        self.grad_loop = -1
        self.active_loops = 0

    def zero_slots(self):
        self.slots.clear()


LG_CTX = _GradLoopCtx()


class _PerLoopLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, param_ref, loop_idx):
        ctx.loop_idx = loop_idx
        ctx.param_ref = param_ref
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight.to(dtype=x.dtype))

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        grad_x = (grad_y @ weight.to(dtype=grad_y.dtype)) if ctx.needs_input_grad[0] else None
        if not ctx.needs_input_grad[1]:
            return grad_x, None, None, None
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x.reshape(-1, x.shape[-1])
        gw = (gy.t() @ xr.to(gy.dtype)).float()
        if LG_CTX.enabled and ctx.loop_idx >= 0:
            slot = LG_CTX.slots.setdefault(ctx.param_ref, {})
            if ctx.loop_idx in slot:
                slot[ctx.loop_idx].add_(gw)
            else:
                slot[ctx.loop_idx] = gw
            if LG_CTX.suppress_default_grad:
                return grad_x, None, None, None
        return grad_x, gw.to(weight.dtype), None, None


def _per_loop_fwd(self, x):
    if LG_CTX.enabled and torch.is_grad_enabled() and LG_CTX.grad_loop >= 0 and self.weight.requires_grad:
        return _PerLoopLinear.apply(x, self.weight, self.weight, LG_CTX.grad_loop)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def attach_grad_loop_tracker(model, include_state_adapter: bool = True):
    """Bug-free grad-only loop tracker; puts pre-hook on state_adapter for Huginn so loop 0 isn't shifted."""
    cfg = model.config
    first_l = cfg.pre_layers
    last_l = cfg.pre_layers + cfg.core_layers - 1
    has_adapter = (include_state_adapter and cfg.method == "huginn"
                   and hasattr(model, "state_adapter") and model.state_adapter is not None
                   and hasattr(model.state_adapter, "weight"))

    def _on_loop_entry(_mod, _inp):
        if LG_CTX.enabled and torch.is_grad_enabled():
            LG_CTX.grad_loop += 1
            LG_CTX.active_loops = max(LG_CTX.active_loops, LG_CTX.grad_loop + 1)

    if has_adapter:
        model.state_adapter.register_forward_pre_hook(_on_loop_entry)
    else:
        model.transformer.h[first_l].register_forward_pre_hook(_on_loop_entry)

    core_params = []
    if has_adapter:
        model.state_adapter.forward = types.MethodType(_per_loop_fwd, model.state_adapter)
        core_params.append(model.state_adapter.weight)

    for l_idx in range(first_l, last_l + 1):
        blk = model.transformer.h[l_idx]
        for mod in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v,
                    blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
            mod.forward = types.MethodType(_per_loop_fwd, mod)
            core_params.append(mod.weight)
    return core_params


def polar_express_raw(G: torch.Tensor, ns_steps: int = 5, use_muon_eq: bool = True) -> torch.Tensor:
    """Exact official 5-step nanochat Polar Express."""
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


def apply_muon_step(plist, U: torch.Tensor, v_buf: torch.Tensor, lr: float, wd: float, beta2: float = 0.9):
    """Exact official Muon+ (full sqrt(min(m,n)) norm!) + NorMuon + Cautious WD."""
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
    x0 = model.x0_lambdas.detach().float().cpu().tolist()
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
        x0_core_mean=round(sum(x0[first_l:last_l]) / max(core_layers, 1), 4),
        mean_fnorm=round(sum(fns) / len(fns), 3),
        mean_kappa=round(sum(kaps) / len(kaps), 3),
        mean_erank=round(sum(erks) / len(erks), 1),
    )


def build_non_core_opt(model, batch_tokens: int, core_ids: set, exclude_core: bool,
                       core_lr_mult: float, core_wd_mult: float):
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
            groups.append(dict(kind="muon", params=core[sh], lr=muon_lr * core_lr_mult,
                               initial_lr=muon_lr * core_lr_mult, momentum=0.95, ns_steps=5, beta2=0.9,
                               weight_decay=wd * core_wd_mult, initial_weight_decay=wd * core_wd_mult))
    return ExperimentOptimizer(groups), muon_lr * core_lr_mult, wd * core_wd_mult


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=[
        "nt_base",
        "alg_jfb_cap_pe5",
        "alg_jfb_equal_pe5",
        "alg_tbptt_s4",
        "alg_tau_beta1",
        "alg_jfb_tau_muon",
    ])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="fixed_recursive", choices=["fixed_recursive", "huginn", "ouro", "deeploop"])
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--core-lr-mult", type=float, default=1.0)
    ap.add_argument("--core-wd-mult", type=float, default=1.0)
    ap.add_argument("--freeze-x0", action="store_true", help="Gate 1b control: freeze x0_lambdas=0 (universal Transformer)")
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

    core_params = attach_grad_loop_tracker(model, include_state_adapter=True)
    core_ids = {id(p) for p in core_params}
    s_max = max(bptt, 1)

    idea = args.idea
    uses_custom_core = (idea != "nt_base")
    use_jfb_cap = idea in ("alg_jfb_cap_pe5", "alg_jfb_tau_muon")
    use_jfb_equal = (idea == "alg_jfb_equal_pe5")
    use_tbptt = (idea == "alg_tbptt_s4")
    use_tau = idea in ("alg_tau_beta1", "alg_jfb_tau_muon")

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt, base_core_lr, base_core_wd = build_non_core_opt(
        model, batch_tokens, core_ids, exclude_core=uses_custom_core,
        core_lr_mult=args.core_lr_mult, core_wd_mult=args.core_wd_mult,
    )
    core_by_shape = defaultdict(list)
    for cp in core_params:
        core_by_shape[tuple(cp.shape)].append(cp)

    m_bufs, v_bufs = {}, {}
    n_ema, n_init = {}, {}
    tau_num, tau_den = {}, {}

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=512)
    print(f"[{args.tag}|{idea}|x0fz={int(args.freeze_x0)}] step=0 val_loss={vl0:.4f} val_bpb={bpb0:.5f}", flush=True)

    t0_all = time.time()
    history, step_times = [], []
    last_b1, last_kap, last_w0 = 0.95, 1.0, 1.0
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

        model.zero_grad(set_to_none=True)
        model.set_step(step)
        LG_CTX.zero_slots()
        LG_CTX.enabled = uses_custom_core
        LG_CTX.suppress_default_grad = uses_custom_core and (step > 0)

        loss_sum = 0.0
        max_active = 0
        for m_idx in range(accum):
            r0 = (step * args.batch_size + m_idx * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m_idx * args.micro_batch)
            LG_CTX.begin_forward()
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum
            if uses_custom_core:
                max_active = max(max_active, LG_CTX.active_loops)

        LG_CTX.enabled = False
        LG_CTX.suppress_default_grad = False

        pending = []
        if uses_custom_core:
            with torch.no_grad():
                b1_acc, kap_acc, w0_acc = [], [], []
                s_act = max(1, max_active)
                for sh, plist in core_by_shape.items():
                    K, m, n = len(plist), sh[0], sh[1]
                    red_dim = -1 if m >= n else -2
                    G_t = torch.stack(
                        [torch.stack([LG_CTX.slots.get(cp, {}).get(t, torch.zeros_like(cp, dtype=torch.float32))
                                      for t in range(s_act)], dim=0) for cp in plist],
                        dim=0,
                    )  # [K, s_act, m, n]
                    if step == 0:
                        ref = torch.stack([cp.grad.float() for cp in plist], dim=0)
                        err = float((G_t.sum(dim=1) - ref).abs().max().item())
                        assert err < 2e-2, f"Step-0 grad equiv failed on {sh}: {err}"
                    if sh not in m_bufs:
                        m_bufs[sh] = torch.zeros((K, m, n), device=device, dtype=torch.float32)
                        v_bufs[sh] = torch.zeros((K, m, 1) if m >= n else (K, 1, n), device=device, dtype=torch.float32)
                        n_ema[sh] = torch.zeros((K, max(s_max, s_act), 1, 1), device=device, dtype=torch.float32)
                        n_init[sh] = [False] * max(s_max, s_act)
                        tau_num[sh] = torch.zeros((K, m, 1) if m >= n else (K, 1, n), device=device, dtype=torch.float32)
                        tau_den[sh] = torch.zeros((K, m, 1) if m >= n else (K, 1, n), device=device, dtype=torch.float32)

                    # Law A: Per-loop Jacobian weighting (2 lines, 1.000x memory!)
                    if s_act > 1 and (use_jfb_cap or use_jfb_equal):
                        gn = G_t.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)  # [K, s_act, 1, 1]
                        if s_act > n_ema[sh].size(1):
                            pad = s_act - n_ema[sh].size(1)
                            n_ema[sh] = torch.cat([n_ema[sh], torch.zeros((K, pad, 1, 1), device=device)], dim=1)
                            n_init[sh].extend([False] * pad)
                        for t in range(s_act):
                            if not n_init[sh][t]:
                                n_ema[sh][:, t].copy_(gn[:, t])
                                n_init[sh][t] = True
                            else:
                                n_ema[sh][:, t].lerp_(gn[:, t], 0.1)
                        n_cur = n_ema[sh][:, :s_act]
                        anchor = n_cur[:, s_act - 1:s_act]
                        w_t = anchor / (n_cur + 1e-8)
                        if use_jfb_cap:
                            w_t = w_t.clamp_max(1.0)
                        G_w = G_t * w_t
                        w0_acc.append(float(w_t[:, 0].mean().item()))
                    elif s_act > 1 and use_tbptt:
                        cut = max(0, s_act - 4)
                        G_w = G_t[:, cut:]
                        w0_acc.append(0.0 if cut > 0 else 1.0)
                    else:
                        G_w = G_t
                        w0_acc.append(1.0)

                    G_sum = G_w.sum(dim=1)  # [K, m, n]
                    for idx_p, cp in enumerate(plist):
                        cp.grad = G_sum[idx_p].to(cp.dtype)

                    # Law B: Geometric-Mean Coherence Momentum (2 lines, 1.002x memory!)
                    s_eff = G_w.size(1)
                    if use_tau and s_eff > 1:
                        c_num = G_sum.square().sum(dim=red_dim, keepdim=True)
                        c_den = G_w.square().sum(dim=red_dim, keepdim=True).sum(dim=1)
                        tau_num[sh].lerp_(c_num, 0.1)
                        tau_den[sh].lerp_(c_den, 0.1)
                        kappa = (tau_num[sh] / (tau_den[sh] + 1e-12)).clamp_min(1e-4)
                        sr = float(s_eff) ** 0.5
                        tau = (kappa / sr).clamp_max(1.0)
                        b1_t = 0.95 ** (1.0 / tau.clamp_min(1.0 / sr))
                        b1_acc.append(float(b1_t.mean().item()))
                        kap_acc.append(float(kappa.mean().item()))
                    else:
                        b1_t = 0.95
                        b1_acc.append(0.95)
                        kap_acc.append(1.0)

                    m_bufs[sh].lerp_(G_sum, 1.0 - b1_t)
                    M_nes = G_sum.lerp(m_bufs[sh], b1_t)
                    U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                    pending.append((plist, U, v_bufs[sh]))

                if b1_acc:
                    last_b1 = sum(b1_acc) / len(b1_acc)
                    last_kap = sum(kap_acc) / len(kap_acc)
                    last_w0 = sum(w0_acc) / len(w0_acc)

        LG_CTX.zero_slots()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()

        if uses_custom_core:
            with torch.no_grad():
                for plist, U, v_buf in pending:
                    apply_muon_step(plist, U, v_buf, lr=core_lr_now, wd=core_wd_now, beta2=0.9)

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            rows = 4096 if (step + 1) == args.steps else 1024
            vb, vl = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=rows)
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum,
                                b1=round(last_b1, 4), kap=round(last_kap, 3), w0=round(last_w0, 3),
                                elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f} "
                  f"b1={last_b1:.4f} kap={last_kap:.2f} w0={last_w0:.3f}", flush=True)

    diag = collect_diag(model, cfg.pre_layers, cfg.core_layers)
    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    out = dict(
        tag=args.tag, idea=idea, method=args.method, loops=args.loops, freeze_x0=args.freeze_x0,
        core_lr_mult=args.core_lr_mult, core_wd_mult=args.core_wd_mult,
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], tok_per_sec=batch_tokens / avg_dt,
        total_wall_s=time.time() - t0_all, diag=diag, history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{args.tag}] DONE val={out['final_val_loss']:.5f} b1={last_b1:.4f} kap={last_kap:.2f} "
          f"w0={last_w0:.3f} fn={diag['mean_fnorm']} {out['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
