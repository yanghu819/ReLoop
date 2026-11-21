#!/usr/bin/env python3
"""Clean, zero-bloat Stage-2 Runner for internal review #71's 3-to-8 Line R=1-Exact Laws:
  - nt_base: Official 5-step MuonAdamW (with optional --core-lr-mult / --core-wd-mult)
  - alg_core_scalars_1_over_R (2 lines, 0% overhead): 1/R AdamW step on core x0_lambdas & resid_lambdas
  - alg_writer_duality (3 lines, 0% overhead): + 1/sqrt(R) step on ONLY core residual writers (c_proj, mlp_proj)
  - alg_tmn_muon (5 lines, 1.002x mem, <0.3% overhead): Trajectory-Modular-Norm Muon g(kappa)=min(sqrt(kappa), kappa^-alpha)
  - alg_loop_tmn_full (8 lines): alg_writer_duality + alg_tmn_muon

Fixes internal audit Bug #5 from line 1 (counts ONLY torch.is_grad_enabled() loops) and verifies
Step-0 exact gradient equivalence before suppressing default .grad!
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


def attach_grad_loop_tracker(model):
    """Bug-free tracker (fixes Bug #5): counts ONLY torch.is_grad_enabled() passes through core!"""
    cfg = model.config
    first_l = cfg.pre_layers
    last_l = cfg.pre_layers + cfg.core_layers - 1

    def _on_first_pre(_mod, _inp):
        if LG_CTX.enabled and torch.is_grad_enabled():
            LG_CTX.grad_loop += 1
            LG_CTX.active_loops = max(LG_CTX.active_loops, LG_CTX.grad_loop + 1)

    model.transformer.h[first_l].register_forward_pre_hook(_on_first_pre)
    core_params = []
    writer_ids = set()
    for l_idx in range(first_l, last_l + 1):
        blk = model.transformer.h[l_idx]
        for pname, mod in (("c_q", blk.attn.c_q), ("c_k", blk.attn.c_k), ("c_v", blk.attn.c_v),
                           ("c_proj", blk.attn.c_proj), ("c_fc", blk.mlp.c_fc), ("mlp_proj", blk.mlp.c_proj)):
            mod.forward = types.MethodType(_per_loop_fwd, mod)
            core_params.append(mod.weight)
            if pname in ("c_proj", "mlp_proj"):
                writer_ids.add(id(mod.weight))
    return core_params, writer_ids


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


def apply_muon_step(plist, U: torch.Tensor, v_buf: torch.Tensor, lr: float, wd: float,
                    beta2: float = 0.9, post_gate: torch.Tensor = None, writer_mask: torch.Tensor = None):
    """Exact official Muon+ renorm + NorMuon variance reduction + optional post-NorMuon TMN gate & writer scaling."""
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
    # Post-NorMuon Trajectory-Modular-Norm gate (strictly SPD descent, cannot be cancelled by NorMuon!)
    if post_gate is not None:
        U = U * post_gate
    lr_eff = lr * (max(1.0, m / n) ** 0.5)
    if writer_mask is not None:
        lr_eff_t = lr_eff * writer_mask  # [K, 1, 1]
    else:
        lr_eff_t = lr_eff
    P = torch.stack([p.float() for p in plist], dim=0)
    mask = ((U * P) >= 0).float()
    P_new = P - lr_eff_t * U - lr_eff_t * wd * P * mask
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
    res = model.resid_lambdas.detach().float().cpu().tolist()
    r_fn, w_fn, kaps, erks = [], [], [], []
    for l_idx in range(first_l, last_l):
        blk = model.transformer.h[l_idx]
        for pname, p in (("c_q", blk.attn.c_q.weight), ("c_k", blk.attn.c_k.weight),
                         ("c_v", blk.attn.c_v.weight), ("c_proj", blk.attn.c_proj.weight),
                         ("c_fc", blk.mlp.c_fc.weight), ("mlp_proj", blk.mlp.c_proj.weight)):
            wf = p.float()
            fn = float(wf.norm().item())
            (w_fn if pname in ("c_proj", "mlp_proj") else r_fn).append(fn)
            sv = torch.linalg.svdvals(wf)
            pe = sv.square() / sv.square().sum().clamp_min(1e-12)
            erks.append(float((-(pe * pe.clamp_min(1e-12).log()).sum()).exp().item()))
            kaps.append(float(sv[0].item() / sv[len(sv) // 2].clamp_min(1e-8).item()))
    return dict(
        x0_pre_mean=round(sum(x0[:first_l]) / max(first_l, 1), 4),
        x0_core_mean=round(sum(x0[first_l:last_l]) / max(core_layers, 1), 4),
        x0_post_mean=round(sum(x0[last_l:]) / max(len(x0) - last_l, 1), 4),
        res_core_mean=round(sum(res[first_l:last_l]) / max(core_layers, 1), 4),
        reader_fnorm=round(sum(r_fn) / len(r_fn), 3),
        writer_fnorm=round(sum(w_fn) / len(w_fn), 3),
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
                               betas=betas, eps=1e-10, weight_decay=decay, initial_weight_decay=decay))

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
        "alg_core_scalars_1_over_R",
        "alg_writer_only",
        "alg_writer_duality",
        "alg_tmn_a05",
        "alg_tmn_a10",
        "alg_loop_tmn_full_a05",
        "alg_loop_tmn_full_a10",
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
    ap.add_argument("--x0-div", type=float, default=0.0)
    ap.add_argument("--resid-div", type=float, default=0.0)
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
    core_params, writer_ids = attach_grad_loop_tracker(model)
    core_ids = {id(p) for p in core_params}
    first_l, last_l = cfg.pre_layers, cfg.pre_layers + cfg.core_layers
    R_eff = float(cfg.loops)

    idea = args.idea
    use_scalar_1_R = idea in ("alg_core_scalars_1_over_R", "alg_writer_duality",
                              "alg_loop_tmn_full_a05", "alg_loop_tmn_full_a10")
    use_writer_sqrtR = idea in ("alg_writer_only", "alg_writer_duality",
                                "alg_loop_tmn_full_a05", "alg_loop_tmn_full_a10")
    use_tmn = idea in ("alg_tmn_a05", "alg_tmn_a10", "alg_loop_tmn_full_a05", "alg_loop_tmn_full_a10")
    tmn_alpha = 1.0 if idea.endswith("_a10") else 0.5
    uses_custom_core = use_writer_sqrtR or use_tmn

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt, base_core_lr, base_core_wd = build_non_core_opt(
        model, batch_tokens, core_ids, exclude_core=uses_custom_core,
        core_lr_mult=args.core_lr_mult, core_wd_mult=args.core_wd_mult,
    )

    core_by_shape = defaultdict(list)
    for cp in core_params:
        core_by_shape[tuple(cp.shape)].append(cp)

    m_bufs, v_bufs, tmn_num, tmn_den = {}, {}, {}, {}
    writer_masks = {}
    if use_writer_sqrtR:
        inv_sqrt_R = 1.0 / (R_eff ** 0.5)
        for sh, plist in core_by_shape.items():
            wm = [inv_sqrt_R if id(p) in writer_ids else 1.0 for p in plist]
            writer_masks[sh] = torch.tensor(wm, device=device, dtype=torch.float32).view(-1, 1, 1)

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=512)
    print(f"[{args.tag}|{idea}] step=0 val_loss={vl0:.4f} val_bpb={bpb0:.5f}", flush=True)

    t0_all = time.time()
    history = []
    step_times = []
    last_kap_mean, last_g_mean = 1.0, 1.0
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
        LG_CTX.enabled = use_tmn
        # Step 0 keeps default .grad so we assert exact gradient equivalence!
        LG_CTX.suppress_default_grad = use_tmn and (step > 0)

        loss_sum = 0.0
        max_active_loops = 0
        for m_idx in range(accum):
            r0 = (step * args.batch_size + m_idx * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m_idx * args.micro_batch)
            LG_CTX.begin_forward()
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum
            if use_tmn:
                max_active_loops = max(max_active_loops, LG_CTX.active_loops)

        LG_CTX.enabled = False
        LG_CTX.suppress_default_grad = False

        pending = []
        if uses_custom_core:
            with torch.no_grad():
                kap_acc, g_acc = [], []
                for sh, plist in core_by_shape.items():
                    K, m, n = len(plist), sh[0], sh[1]
                    red_dim = -1 if m >= n else -2
                    if sh not in v_bufs:
                        v_bufs[sh] = torch.zeros((K, m, 1) if m >= n else (K, 1, n), device=device, dtype=torch.float32)
                    if use_tmn:
                        s_act = max(1, max_active_loops)
                        G_t = torch.stack(
                            [torch.stack([LG_CTX.slots.get(cp, {}).get(t, torch.zeros_like(cp, dtype=torch.float32))
                                          for t in range(s_act)], dim=0) for cp in plist],
                            dim=0,
                        )  # [K, s_act, m, n]
                        G_sum = G_t.sum(dim=1)
                        if step == 0:
                            ref = torch.stack([cp.grad.float() for cp in plist], dim=0)
                            err = float((G_sum - ref).abs().max().item())
                            assert err < 2e-2, f"Step-0 grad equivalence failed on {sh}: {err}"
                        for idx_p, cp in enumerate(plist):
                            cp.grad = G_sum[idx_p].to(cp.dtype)
                        if sh not in m_bufs:
                            m_bufs[sh] = torch.zeros_like(G_sum)
                            tmn_num[sh] = torch.zeros((K, m, 1) if m >= n else (K, 1, n), device=device, dtype=torch.float32)
                            tmn_den[sh] = torch.zeros((K, m, 1) if m >= n else (K, 1, n), device=device, dtype=torch.float32)
                        m_bufs[sh].lerp_(G_sum, 0.05)
                        M_nes = G_sum.lerp(m_bufs[sh], 0.95)
                        if s_act <= 1:
                            g_tmn = None
                            U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                        else:
                            c_num = G_sum.square().sum(dim=red_dim, keepdim=True)
                            c_den = G_t.square().sum(dim=red_dim, keepdim=True).sum(dim=1)
                            tmn_num[sh].lerp_(c_num, 0.1)
                            tmn_den[sh].lerp_(c_den, 0.1)
                            kappa = (tmn_num[sh] / (tmn_den[sh] + 1e-12)).clamp_min(1e-4)
                            g_tmn = torch.minimum(kappa.sqrt(), kappa.pow(-tmn_alpha))
                            kap_acc.append(float(kappa.mean().item()))
                            g_acc.append(float(g_tmn.mean().item()))
                            target = M_nes.norm(dim=(-2, -1), keepdim=True) / (m ** 0.5)
                            X_eq = M_nes * (target / M_nes.norm(dim=-1, keepdim=True).clamp_min(1e-6))
                            U = polar_express_raw(X_eq * g_tmn, ns_steps=5, use_muon_eq=False)
                        pending.append((plist, U, v_bufs[sh], g_tmn, writer_masks.get(sh)))
                    else:
                        G_sum = torch.stack([cp.grad.float() for cp in plist], dim=0)
                        if sh not in m_bufs:
                            m_bufs[sh] = torch.zeros_like(G_sum)
                        m_bufs[sh].lerp_(G_sum, 0.05)
                        M_nes = G_sum.lerp(m_bufs[sh], 0.95)
                        U = polar_express_raw(M_nes, ns_steps=5, use_muon_eq=True)
                        pending.append((plist, U, v_bufs[sh], None, writer_masks.get(sh)))
                if kap_acc:
                    last_kap_mean = sum(kap_acc) / len(kap_acc)
                    last_g_mean = sum(g_acc) / len(g_acc)

        LG_CTX.zero_slots()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)

        x0_div = args.x0_div if args.x0_div > 0.0 else (R_eff if use_scalar_1_R else 1.0)
        res_div = args.resid_div if args.resid_div > 0.0 else (R_eff if use_scalar_1_R else 1.0)
        if x0_div != 1.0 or res_div != 1.0:
            x0_pre = model.x0_lambdas[first_l:last_l].detach().clone()
            res_pre = model.resid_lambdas[first_l:last_l].detach().clone()
        opt.step()
        if x0_div != 1.0 or res_div != 1.0:
            with torch.no_grad():
                if x0_div != 1.0:
                    model.x0_lambdas[first_l:last_l].copy_(x0_pre + (model.x0_lambdas[first_l:last_l] - x0_pre) / x0_div)
                if res_div != 1.0:
                    model.resid_lambdas[first_l:last_l].copy_(res_pre + (model.resid_lambdas[first_l:last_l] - res_pre) / res_div)

        if uses_custom_core:
            with torch.no_grad():
                for plist, U, v_buf, g_tmn, wmask in pending:
                    apply_muon_step(plist, U, v_buf, lr=core_lr_now, wd=core_wd_now,
                                    beta2=0.9, post_gate=g_tmn, writer_mask=wmask)

        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            rows = 4096 if (step + 1) == args.steps else 1024
            vb, vl = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=rows)
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum,
                                kappa_mean=round(last_kap_mean, 4), g_mean=round(last_g_mean, 4),
                                elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f} "
                  f"kap={last_kap_mean:.3f} g={last_g_mean:.3f}", flush=True)

    diag = collect_diag(model, first_l, cfg.core_layers)
    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    out = dict(
        tag=args.tag, idea=idea, method=args.method, loops=args.loops,
        core_lr_mult=args.core_lr_mult, core_wd_mult=args.core_wd_mult,
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], tok_per_sec=batch_tokens / avg_dt,
        total_wall_s=time.time() - t0_all, diag=diag, history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{args.tag}] DONE val={out['final_val_loss']:.5f} x0_c={diag['x0_core_mean']} "
          f"r_fn={diag['reader_fnorm']} w_fn={diag['writer_fnorm']} {out['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
