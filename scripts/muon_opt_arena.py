"""8-GPU Pure Loop-Muon Optimizer Search Arena (ZERO architecture changes, ZERO inference cost).

Architecture is 100% frozen across all 8 GPUs (identical ReferenceLM backbone, seed=42,
pre-tokenized ClimbMix train/val data, and SOTA nanochat MuonAdamW baseline:
Polar Express + MuonEq + Muon+ + NorMuon + Cautious WD).

Only the optimizer rule applied to the weight-tied recurrent core matrices varies across GPUs.
Also records per-loop gradient norms, cross-loop gradient cosines, and core weight singular spectra.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/nt")
sys.path.insert(0, "/root/pn/loop-muon")

from nanochat.optim import polar_express_coeffs
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import ReferenceLM, build_model, norm
from nexttransformer.training.optimizer import ExperimentOptimizer


# ---------------------------------------------------------------------------
# Fast Polar Express helper (exact same 5-step coeffs as nanochat/optim.py)
# ---------------------------------------------------------------------------
def polar_express(G: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """Orthogonalize 2D (or batched [B, m, n]) matrix via Polar Express in bf16."""
    X = G.bfloat16()
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
    out = X.float()
    target_norm = min(out.size(-2), out.size(-1)) ** 0.5
    cur_norm = out.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return out * (target_norm / cur_norm)


# ---------------------------------------------------------------------------
# Per-loop gradient capture for tied core matrices (supports microbatch accum!)
# ---------------------------------------------------------------------------
class _PerLoopGradCtx:
    def __init__(self):
        self.enabled = False
        self.loop = None               # 0..s-1 for grad-carrying loops, None otherwise
        self.accum_scale = 1.0         # 1 / accum
        self.slots = {}                # param -> {t: accumulated fp32 grad}

    def zero_slots(self):
        self.slots.clear()


LG_CTX = _PerLoopGradCtx()


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
            # Return zero to param.grad when we synthesize param.grad before opt.step()
            # (or return gw normally if telemetry-only mode!)
            if LG_CTX.suppress_default_grad:
                return grad_x, torch.zeros_like(weight), None
        return grad_x, gw.to(weight.dtype), None


def _per_loop_fwd(self, x):
    if LG_CTX.enabled and LG_CTX.loop is not None and torch.is_grad_enabled() and self.weight.requires_grad:
        return _PerLoopLinear.apply(x, self.weight, self.weight)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def wrap_core_linears_for_per_loop(model, first_l: int, last_l: int):
    core_params = []
    for layer in range(first_l, last_l + 1):
        blk = model.transformer.h[layer]
        for m in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v, blk.attn.c_proj,
                  blk.mlp.c_fc, blk.mlp.c_proj):
            m.forward = types.MethodType(_per_loop_fwd, m)
            core_params.append(m.weight)
    return core_params


# ---------------------------------------------------------------------------
# Frozen backbone wrapper (clean Huginn without x0 short-circuit, or Pre-Norm)
# ---------------------------------------------------------------------------
def configure_backbone(model: ReferenceLM, backbone: str):
    c = model.config
    first_l, last_l = c.pre_layers, c.pre_layers + c.core_layers - 1
    use_no_x0 = backbone in ("huginn", "prenorm")
    use_prenorm = backbone == "prenorm"

    if use_prenorm:
        # Clean scaled Pre-Norm residual (scale = 1/sqrt(2*core*R) ~ 0.15) so Pre-Norm is stable at r=8,16
        scale = (2 * c.core_layers * c.max_loops) ** -0.5
        def _prenorm_fwd(self_blk, x, ve, cos_sin, window_size, kv_cache):
            x = x + scale * self_blk.attn(norm(x), ve, cos_sin, window_size, kv_cache)
            return x + scale * self_blk.mlp(norm(x))
        for layer in range(first_l, last_l + 1):
            model.transformer.h[layer].forward = types.MethodType(_prenorm_fwd, model.transformer.h[layer])
        if hasattr(model, "state_adapter"):
            model.state_adapter.requires_grad_(False)

    def _run_core_block(self, x, x0, idx, physical, virtual, total):
        if use_no_x0:
            ve = self.value_embeds[str(physical)](idx).to(x.dtype) if str(physical) in self.value_embeds else None
            window = self._window(virtual, total)
            return self.transformer.h[physical](
                x, ve, (self.cos[:, :idx.shape[1]], self.sin[:, :idx.shape[1]]), window, None
            )
        return self._block(x, x0, idx, physical, virtual, total)

    def _clean_loop_hidden(self, idx, targets=None):
        cfg = self.config
        x = self._embed_reference(idx)
        x0 = x
        for i in range(cfg.pre_layers):
            x = self._block(x, x0, idx, i, i, cfg.maximum_depth)
        prelude = x
        n = cfg.max_loops if self.training else cfg.eval_loops
        if not use_prenorm:
            noise = []
            for row in range(idx.shape[0]):
                g = torch.Generator(device=idx.device).manual_seed(cfg.routing_seed + self.row_offset + row)
                noise.append(torch.randn((1, *x.shape[1:]), device=x.device, dtype=x.dtype, generator=g))
            x = torch.cat(noise, 0) * cfg.width ** -0.5
        total = cfg.pre_layers + n * cfg.core_layers + cfg.post_layers
        grad_start = n - cfg.huginn_backprop_loops
        raw_u_list = []

        for r in range(n):
            no_grad = r < grad_start
            with torch.no_grad() if no_grad else nullcontext():
                if r == grad_start:
                    x = x.detach()
                if not use_prenorm:
                    x = norm(self.state_adapter(torch.cat((x, prelude), -1)))
                u_in = x
                LG_CTX.loop = (r - grad_start) if (not no_grad and torch.is_grad_enabled()) else None
                try:
                    for j in range(cfg.core_layers):
                        x = _run_core_block(self, x, x0, idx, cfg.pre_layers + j,
                                            cfg.pre_layers + r * cfg.core_layers + j, total)
                finally:
                    LG_CTX.loop = None
                if not self.training:
                    raw_u_list.append((x - u_in).detach().float())

        if not self.training and len(raw_u_list) >= 2:
            u_last = raw_u_list[-1].reshape(-1, cfg.width)
            U = torch.stack([u.reshape(-1, cfg.width) for u in raw_u_list[:-1]], -1)
            G = U.transpose(-1, -2) @ U
            k = U.shape[-1]
            scale_g = G.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
            eye = torch.eye(k, device=u_last.device, dtype=torch.float32)
            b = (U.transpose(-1, -2) @ u_last.unsqueeze(-1)).squeeze(-1)
            coeff = torch.linalg.solve(G + (1e-6 * scale_g)[..., None, None] * eye, b.unsqueeze(-1))
            proj = (U @ coeff).squeeze(-1)
            frac = (proj.pow(2).sum(-1) / u_last.pow(2).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
            self.last_stats["last_loop_span_frac"] = float(frac.mean().item())

        for j in range(cfg.post_layers):
            x = self._block(x, x0, idx, cfg.pre_layers + cfg.core_layers + j,
                            cfg.pre_layers + n * cfg.core_layers + j, total)
        return self._readout(x, prelude), x.new_zeros(()), None

    model._loop_hidden = types.MethodType(_clean_loop_hidden, model)


# ---------------------------------------------------------------------------
# Optimizer builder (clean separation of tied core vs untied 2D matrices!)
# ---------------------------------------------------------------------------
def build_loop_muon_optimizer(model: ReferenceLM, batch_tokens: int, core_param_ids: set,
                              core_lr_mult: float = 1.0, core_momentum: float = 0.95,
                              core_wd_mult: float = 1.0, core_beta2: float = 0.9):
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
                               betas=betas, eps=1e-10, weight_decay=decay, is_core=False))

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
                           weight_decay=wd, initial_weight_decay=wd, is_core=False))
    for shape in sorted(core_mats):
        clr = muon_lr * core_lr_mult
        cwd = wd * core_wd_mult
        groups.append(dict(kind="muon", params=core_mats[shape], lr=clr, initial_lr=clr,
                           momentum=core_momentum, ns_steps=5, beta2=core_beta2,
                           weight_decay=cwd, initial_weight_decay=cwd, is_core=True))
    return ExperimentOptimizer(groups)


# ---------------------------------------------------------------------------
# Telemetry: per-loop grad norms, cross-loop cosines, and weight spectra!
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_telemetry(model, core_params, slots, first_l, core_layers):
    info = {"per_loop_grads": {}, "weight_spectra": {}}
    # Sample early, mid, late core layers
    sample_js = sorted({0, core_layers // 2, core_layers - 1})
    for j in sample_js:
        blk = model.transformer.h[first_l + j]
        for sub, p in [("c_q", blk.attn.c_q.weight), ("c_v", blk.attn.c_v.weight),
                       ("c_proj", blk.attn.c_proj.weight), ("c_fc", blk.mlp.c_fc.weight),
                       ("mlp_proj", blk.mlp.c_proj.weight)]:
            key = f"core{j}.{sub}"
            sv = torch.linalg.svdvals(p.float())
            med = sv[len(sv) // 2].clamp_min(1e-8).item()
            p_en = (sv.square() / sv.square().sum().clamp_min(1e-12))
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
                        cval = (Gs[a] * Gs[b]).sum().item() / max(norms[a] * norms[b], 1e-12)
                        cos_off.append(cval)
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
def eval_multidepth(model, val_x, val_y, byte_table, device, batch_size: int,
                    depths=(4, 8, 16), max_rows: int = 1024):
    model.eval()
    old_eval = model.config.eval_loops
    out = {}
    collapse_r8 = None
    try:
        for d in depths:
            object.__setattr__(model.config, "eval_loops", d)
            total_nats = 0.0
            total_bytes = 0.0
            fracs = []
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
                if d == 8 and "last_loop_span_frac" in model.last_stats:
                    fracs.append(model.last_stats["last_loop_span_frac"])
            out[str(d)] = total_nats / (math.log(2.0) * max(total_bytes, 1.0))
            if d == 8 and fracs:
                collapse_r8 = sum(fracs) / len(fracs)
    finally:
        object.__setattr__(model.config, "eval_loops", old_eval)
        model.train()
    return out, collapse_r8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea", required=True)
    p.add_argument("--backbone", default="huginn", choices=["huginn", "prenorm", "raw_huginn"])
    p.add_argument("--data-dir", type=Path, default=Path("/dev/shm/rsi_arena_data"))
    p.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=550)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=275)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--core-layers", type=int, default=6)
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

    cfg = ModelConfig(
        method="huginn",
        scale="debug",
        width=args.width,
        heads=args.heads,
        vocab_size=32768,
        sequence_len=seq_len,
        mlp_width=4 * args.width,
        layers=1 + args.core_layers + 1,
        pre_layers=1,
        core_layers=args.core_layers,
        post_layers=1,
        max_loops=8,
        huginn_mean=40.0,
        huginn_sigma=0.5,
        huginn_backprop_loops=4,
        eval_loops=8,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    configure_backbone(model, args.backbone)

    first_l, last_l = cfg.pre_layers, cfg.pre_layers + cfg.core_layers - 1
    core_params = wrap_core_linears_for_per_loop(model, first_l, last_l)
    core_param_ids = {id(p) for p in core_params}

    # Configure optimizer idea (ZERO architecture changes!)
    idea = args.idea
    core_lr_mult = 1.0
    core_momentum = 0.95
    core_wd_mult = 1.0
    core_beta2 = 0.9
    needs_per_loop = False
    polar_wd_strength = 0.0
    schatten_wd_strength = 0.0
    schatten_pure_strength = 0.0      # Pure cubic Schatten-4 decay W (W^T W / alpha^2) replacing L2 WD
    spectral_clip_strength = 0.0      # 1-step power-iteration soft top-sigma cap
    spectral_clip_ratio = 1.5         # target sigma_max / alpha
    spectral_scope = "core"           # "core", "all", or "attn"
    per_loop_mode = None
    per_loop_weights = None
    per_loop_mom_bufs = {}

    if idea == "opt_base":
        pass
    elif idea == "opt_polar_wd":
        polar_wd_strength = 0.05
    elif idea == "opt_polar_wd_050":
        polar_wd_strength = 0.50
    elif idea == "opt_polar_rep_wd":
        # Exact norm-preserving Polar WD REPLACING standard L2 WD on tied core!
        core_wd_mult = 0.0
        polar_wd_strength = 0.50
    elif idea == "opt_schatten_wd":
        schatten_wd_strength = 0.05
    elif idea == "opt_schatten_020":
        schatten_wd_strength = 0.20
    elif idea == "opt_schatten_060":
        schatten_wd_strength = 0.60
    elif idea == "opt_schatten_pure":
        # Pure cubic Schatten-4 weight decay W -= lam*lr*alpha*(Wn @ G_w) REPLACING L2 WD!
        core_wd_mult = 0.0
        schatten_pure_strength = 0.10
    elif idea == "opt_schatten_all_020":
        # Control: apply Schatten-4 (0.20) to ALL Muon 2D layers (untied + tied)
        schatten_wd_strength = 0.20
        spectral_scope = "all"
    elif idea == "opt_schatten_attn_020":
        # Apply Schatten-4 (0.20) ONLY to tied core attention matrices (where rank_frac=0.58)
        schatten_wd_strength = 0.20
        spectral_scope = "attn"
    elif idea == "opt_spectral_clip":
        # Surgical rank-1 power-iteration soft cap on sigma_max ONLY (zero bulk distortion)
        spectral_clip_strength = 0.50
        spectral_clip_ratio = 1.5
    elif idea == "opt_core_wd_3x":
        # Critical causal control: 3x standard cautious L2 WD on tied core (isotropic, kappa unchanged)
        core_wd_mult = 3.0
    elif idea == "opt_core_lr_05":
        core_lr_mult = 0.5
    elif idea == "opt_core_lr_15":
        core_lr_mult = 1.5
    elif idea == "opt_core_beta1_085":
        core_momentum = 0.85
    elif idea == "opt_core_beta1_098":
        core_momentum = 0.98
    elif idea == "opt_core_beta2_050":
        core_beta2 = 0.50
    elif idea == "opt_loop_bal":
        needs_per_loop = True
        per_loop_mode = "bal"
    elif idea == "opt_sum_of_pe":
        needs_per_loop = True
        per_loop_mode = "sum_pe"
    elif idea == "opt_per_loop_mom":
        needs_per_loop = True
        per_loop_mode = "per_loop_mom"
        core_momentum = 0.0
    elif idea == "opt_loop_pcgrad":
        # Project out negative cross-loop interference (<G_t, G_{>t}> < 0) without norm distortion!
        needs_per_loop = True
        per_loop_mode = "pcgrad"
    elif idea == "opt_schatten_pcgrad":
        # Combo: Schatten-4 (0.20) weight spectral conditioning + Loop-PCGrad gradient de-conflicting
        schatten_wd_strength = 0.20
        needs_per_loop = True
        per_loop_mode = "pcgrad"
    elif idea == "opt_loop_late_wt":
        # Downweight noisy/anti-correlated early loops: w = [0.25, 0.5, 0.75, 1.0]
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.25, 0.50, 0.75, 1.00]
    elif idea == "opt_loop_early_wt":
        # Gentle early-loop emphasis without 10x explosion: w = [1.5, 1.25, 1.0, 0.75]
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [1.50, 1.25, 1.00, 0.75]
    else:
        raise ValueError(f"Unknown optimizer idea: {idea}")

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    batch_tokens = args.batch_size * seq_len
    opt = build_loop_muon_optimizer(model, batch_tokens, core_param_ids,
                                    core_lr_mult=core_lr_mult, core_momentum=core_momentum,
                                    core_wd_mult=core_wd_mult, core_beta2=core_beta2)

    # Group core params by shape (for per-loop ops) and spectral-target params by shape
    core_by_shape = defaultdict(list)
    for cp in core_params:
        core_by_shape[tuple(cp.shape)].append(cp)

    spec_by_shape = defaultdict(list)
    if spectral_scope == "all":
        for g in opt.param_groups:
            if g.get("kind") == "muon":
                for p in g["params"]:
                    spec_by_shape[tuple(p.shape)].append(p)
    elif spectral_scope == "attn":
        for layer in range(first_l, last_l + 1):
            blk = model.transformer.h[layer]
            for m in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v, blk.attn.c_proj):
                spec_by_shape[tuple(m.weight.shape)].append(m.weight)
    else:
        spec_by_shape = core_by_shape

    # Power-iteration top-singular vectors (1 vector per matrix -> negligible memory!)
    power_u = {}
    if spectral_clip_strength > 0.0:
        for shape, plist in spec_by_shape.items():
            u = torch.randn((len(plist), shape[0]), device=device, dtype=torch.float32)
            power_u[shape] = F.normalize(u, dim=-1)

    t_start = time.time()
    history = []
    telemetry = {}
    n_rows = train_x.shape[0]

    LG_CTX.enabled = False
    LG_CTX.suppress_default_grad = False
    bpb0, col0 = eval_multidepth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                 depths=(8,), max_rows=512)
    print(f"[{idea}|{args.backbone}] params={n_params/1e6:.1f}M step=0 val_bpb@8={bpb0['8']:.5f} collapse={col0:.4f}",
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

        opt.zero_grad(set_to_none=True)
        model.set_step(step)

        capture_telem = (idea == "opt_base") and (step + 1 in (100, args.steps))
        LG_CTX.zero_slots()
        LG_CTX.enabled = needs_per_loop or capture_telem
        LG_CTX.suppress_default_grad = needs_per_loop

        loss_sum = 0.0
        for m in range(accum):
            row0 = (step * args.batch_size + m * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m * args.micro_batch)
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        if capture_telem:
            telemetry[f"step_{step+1}"] = collect_telemetry(model, core_params, LG_CTX.slots, first_l, cfg.core_layers)

        # Synthesize per-loop preconditioned gradient for tied core matrices
        if needs_per_loop:
            with torch.no_grad():
                for shape, plist in core_by_shape.items():
                    s_loops = cfg.huginn_backprop_loops
                    if per_loop_mode == "bal":
                        for cp in plist:
                            sl = LG_CTX.slots[cp]
                            Gs = [sl[t] for t in range(s_loops)]
                            raw_sum = sum(Gs)
                            raw_norm = raw_sum.norm().clamp_min(1e-8)
                            bal = sum(g / g.norm().clamp_min(1e-8) for g in Gs)
                            bal_norm = bal.norm().clamp_min(1e-8)
                            cp.grad = (bal * (raw_norm / bal_norm)).to(cp.dtype)
                    elif per_loop_mode == "sum_pe":
                        # Batched Polar Express over (len(plist) * s_loops, m, n) in one fast GPU call!
                        stacked = torch.stack([LG_CTX.slots[cp][t] for cp in plist for t in range(s_loops)], dim=0)
                        orth = polar_express(stacked, ns_steps=5).view(len(plist), s_loops, *shape)
                        comb = orth.mean(dim=1)
                        for idx_p, cp in enumerate(plist):
                            cp.grad = comb[idx_p].to(cp.dtype)
                    elif per_loop_mode == "per_loop_mom":
                        beta = 0.95
                        stacked_nes = []
                        for cp in plist:
                            sl = LG_CTX.slots[cp]
                            bufs = per_loop_mom_bufs.get(cp)
                            if bufs is None:
                                bufs = [torch.zeros_like(sl[0]) for _ in range(s_loops)]
                                per_loop_mom_bufs[cp] = bufs
                            for t in range(s_loops):
                                g_t = sl[t]
                                bufs[t].lerp_(g_t, 1.0 - beta)
                                stacked_nes.append(g_t.lerp(bufs[t], beta))
                        stacked = torch.stack(stacked_nes, dim=0)
                        orth = polar_express(stacked, ns_steps=5).view(len(plist), s_loops, *shape)
                        comb = orth.mean(dim=1)
                        for idx_p, cp in enumerate(plist):
                            cp.grad = comb[idx_p].to(cp.dtype)

                    elif per_loop_mode == "pcgrad":
                        # Backward conflict-projection: ensure no earlier loop G_t cancels accumulated later-loop G_{>t}
                        for cp in plist:
                            sl = LG_CTX.slots[cp]
                            acc = sl[s_loops - 1].clone()
                            for t in range(s_loops - 2, -1, -1):
                                g_t = sl[t]
                                dot = (g_t * acc).sum()
                                if dot < 0.0:
                                    g_t = g_t - (dot / acc.square().sum().clamp_min(1e-12)) * acc
                                acc.add_(g_t)
                            cp.grad = acc.to(cp.dtype)
                    elif per_loop_mode == "weighted":
                        ws = per_loop_weights
                        w_mean = sum(ws) / len(ws)
                        for cp in plist:
                            sl = LG_CTX.slots[cp]
                            comb = sum((ws[t] / w_mean) * sl[t] for t in range(s_loops))
                            cp.grad = comb.to(cp.dtype)

        LG_CTX.zero_slots()
        LG_CTX.enabled = False
        LG_CTX.suppress_default_grad = False

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()

        # Post-step weight-space spectral conditioning (1x state, 0 inf cost!)
        if polar_wd_strength > 0.0 or schatten_wd_strength > 0.0 or schatten_pure_strength > 0.0 or spectral_clip_strength > 0.0:
            with torch.no_grad():
                core_lr_now = 0.02 * math.sqrt(batch_tokens / 524288.0) * mult
                for shape, plist in spec_by_shape.items():
                    W = torch.stack([cp.float() for cp in plist], dim=0)   # [K, m, n]
                    min_dim = min(shape[0], shape[1])
                    alpha = W.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6) / (min_dim ** 0.5)
                    if polar_wd_strength > 0.0:
                        W_iso = alpha * polar_express(W, ns_steps=5)
                        step_c = polar_wd_strength * core_lr_now
                        W_new = W - step_c * (W - W_iso)
                    elif schatten_wd_strength > 0.0 or schatten_pure_strength > 0.0:
                        Wn = W / alpha
                        if shape[0] >= shape[1]:
                            G_w = Wn.mT @ Wn
                            cubic = Wn @ G_w
                        else:
                            G_w = Wn @ Wn.mT
                            cubic = G_w @ Wn
                        if schatten_pure_strength > 0.0:
                            pen = cubic
                            step_c = schatten_pure_strength * core_lr_now
                        else:
                            pen = cubic - Wn
                            step_c = schatten_wd_strength * core_lr_now
                        W_new = W - step_c * alpha * pen.clamp(-5.0, 5.0)
                    else:
                        # 1-step power-iteration soft top-singular-value cap
                        u = power_u[shape]                                  # [K, m]
                        v = F.normalize(torch.bmm(W.mT, u.unsqueeze(-1)).squeeze(-1), dim=-1)
                        u = F.normalize(torch.bmm(W, v.unsqueeze(-1)).squeeze(-1), dim=-1)
                        power_u[shape] = u
                        sigma1 = (u * torch.bmm(W, v.unsqueeze(-1)).squeeze(-1)).sum(dim=-1, keepdim=True).unsqueeze(-1)
                        excess = (sigma1 - spectral_clip_ratio * alpha).clamp_min(0.0)
                        step_c = spectral_clip_strength * core_lr_now
                        W_new = W - step_c * excess * (u.unsqueeze(-1) @ v.unsqueeze(-2))
                    for idx_p, cp in enumerate(plist):
                        cp.copy_(W_new[idx_p].to(cp.dtype))

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step_times.append(dt)

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            avg_dt = sum(step_times[-50:]) / len(step_times[-50:])
            print(f"[{idea}] step={step+1}/{args.steps} loss={loss_sum:.4f} "
                  f"gnorm={float(grad_norm):.3f} tok/s={batch_tokens/avg_dt/1e3:.1f}k "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            depths = (4, 8, 16) if (step + 1 == args.steps) else (8,)
            rows = 4096 if (step + 1 == args.steps) else 1024
            bpbs, col = eval_multidepth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                        depths=depths, max_rows=rows)
            rec = {
                "step": step + 1,
                "tokens": (step + 1) * batch_tokens,
                "train_loss": loss_sum,
                "val_bpb": bpbs,
                "collapse_r8": col,
                "elapsed_s": time.time() - t_start,
            }
            history.append(rec)
            print(f"[{idea}] VAL step={step+1} bpb={ {k: round(v, 5) for k, v in bpbs.items()} } "
                  f"collapse@8={col:.4f}", flush=True)

    # Also record final weight spectrum for EVERY optimizer idea so we verify spectral conditioning!
    final_spec = collect_telemetry(model, core_params, {}, first_l, cfg.core_layers)["weight_spectra"]
    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    summary = {
        "idea": idea,
        "backbone": args.backbone,
        "params": n_params,
        "steps": args.steps,
        "tokens": args.steps * batch_tokens,
        "seed": args.seed,
        "sec_per_step": avg_dt,
        "tok_per_sec": batch_tokens / avg_dt,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "total_wall_s": time.time() - t_start,
        "final_val_bpb": history[-1]["val_bpb"],
        "final_collapse_r8": history[-1]["collapse_r8"],
        "history": history,
        "final_weight_spectra": final_spec,
        "telemetry": telemetry,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{idea}] DONE in {summary['total_wall_s']:.1f}s | "
          f"final val_bpb={ {k: round(v, 5) for k, v in summary['final_val_bpb'].items()} } | "
          f"collapse@8={summary['final_collapse_r8']:.4f} | "
          f"mem={summary['peak_mem_gb']:.1f}GB | {summary['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
