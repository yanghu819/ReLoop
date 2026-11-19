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


def polar_express_raw(G: torch.Tensor, ns_steps: int = 5, use_muon_eq: bool = True) -> torch.Tensor:
    """Exact nanochat MuonEq + Polar Express without post-renorm (for TEON / t-SVD)."""
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


def apply_muon_update_exact(plist, U: torch.Tensor, v_buf: torch.Tensor,
                            lr: float, wd: float, beta2: float = 0.9):
    """Exact nanochat Muon+ renorm + NorMuon variance reduction + Cautious WD step."""
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
    mask = ((U * P) >= 0).float()
    P_new = P - lr_eff * U - lr_eff * wd * P * mask
    for idx_p, p in enumerate(plist):
        p.copy_(P_new[idx_p].to(p.dtype))


def dct_matrix(s: int, device) -> torch.Tensor:
    """Orthogonal DCT-II matrix Q of shape [s, s] (Q @ Q.T = I)."""
    k = torch.arange(s, device=device, dtype=torch.float32).unsqueeze(1)
    n = torch.arange(s, device=device, dtype=torch.float32).unsqueeze(0)
    Q = torch.cos(math.pi * (n + 0.5) * k / s)
    Q[0] *= (1.0 / s) ** 0.5
    if s > 1:
        Q[1:] *= (2.0 / s) ** 0.5
    return Q


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
                              core_wd_mult: float = 1.0, core_beta2: float = 0.9,
                              untied_wd_mult: float = 1.0, core_ns_steps: int = 5,
                              core_attn_ids: set = None, core_attn_wd_mult: float = None,
                              exclude_core_from_muon: bool = False):
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
        uwd = wd * untied_wd_mult
        groups.append(dict(kind="muon", params=untied_mats[shape], lr=muon_lr, initial_lr=muon_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9,
                           weight_decay=uwd, initial_weight_decay=uwd, is_core=False))
    if not exclude_core_from_muon:
        for shape in sorted(core_mats):
            clr = muon_lr * core_lr_mult
            if core_attn_ids is not None and core_attn_wd_mult is not None:
                attn_ps = [p for p in core_mats[shape] if id(p) in core_attn_ids]
                mlp_ps = [p for p in core_mats[shape] if id(p) not in core_attn_ids]
                if attn_ps:
                    awd = wd * core_attn_wd_mult
                    groups.append(dict(kind="muon", params=attn_ps, lr=clr, initial_lr=clr,
                                       momentum=core_momentum, ns_steps=core_ns_steps, beta2=core_beta2,
                                       weight_decay=awd, initial_weight_decay=awd, is_core=True))
                if mlp_ps:
                    mwd = wd * core_wd_mult
                    groups.append(dict(kind="muon", params=mlp_ps, lr=clr, initial_lr=clr,
                                       momentum=core_momentum, ns_steps=core_ns_steps, beta2=core_beta2,
                                       weight_decay=mwd, initial_weight_decay=mwd, is_core=True))
            else:
                cwd = wd * core_wd_mult
                groups.append(dict(kind="muon", params=core_mats[shape], lr=clr, initial_lr=clr,
                                   momentum=core_momentum, ns_steps=core_ns_steps, beta2=core_beta2,
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
    p.add_argument("--max-loops", type=int, default=8)
    p.add_argument("--bptt-loops", type=int, default=4)
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

    bptt_loops = min(args.bptt_loops, args.max_loops)
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
        max_loops=args.max_loops,
        huginn_mean=40.0,
        huginn_sigma=0.5,
        huginn_backprop_loops=bptt_loops,
        eval_loops=args.max_loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    configure_backbone(model, args.backbone)

    first_l, last_l = cfg.pre_layers, cfg.pre_layers + cfg.core_layers - 1
    core_params = wrap_core_linears_for_per_loop(model, first_l, last_l)
    core_param_ids = {id(p) for p in core_params}

    core_attn_ids = set()
    for layer in range(first_l, last_l + 1):
        blk = model.transformer.h[layer]
        for m in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v, blk.attn.c_proj):
            core_attn_ids.add(id(m.weight))

    # Configure optimizer idea (ZERO architecture changes!)
    idea = args.idea
    core_lr_mult = 1.0
    core_momentum = 0.95
    core_wd_mult = 1.0
    core_attn_wd_mult = None
    untied_wd_mult = 1.0
    core_beta2 = 0.9
    core_ns_steps = 5
    needs_per_loop = False
    polar_wd_strength = 0.0
    schatten_wd_strength = 0.0
    schatten_pure_strength = 0.0
    cautious_schatten_strength = 0.0
    cubic_frob_strength = 0.0
    plain_l2_strength = 0.0
    spectral_clip_strength = 0.0
    spectral_clip_ratio = 1.5
    spectral_scope = "core"
    stiefel_tan_strength = 0.0        # Pre-Muon Riemannian Stiefel tangent projection on G
    use_dual_mom = False              # Dual-timescale momentum 0.5*M(0.80) + 0.5*M(0.95) on core
    dual_mom_bufs = {}
    tensor_muon_mode = None           # Pure algorithmic TEON / Loop-TEON / Loop-tSVD replacing core Muon!
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
        core_wd_mult = 0.0
        polar_wd_strength = 0.50
    elif idea == "opt_schatten_wd":
        schatten_wd_strength = 0.05
    elif idea == "opt_schatten_020":
        schatten_wd_strength = 0.20
    elif idea == "opt_schatten_060":
        schatten_wd_strength = 0.60
    elif idea == "opt_schatten_pure":
        core_wd_mult = 0.0
        schatten_pure_strength = 0.10
    elif idea == "opt_schatten_pure_020":
        core_wd_mult = 0.0
        schatten_pure_strength = 0.20
    elif idea == "opt_schatten_pure_040":
        core_wd_mult = 0.0
        schatten_pure_strength = 0.40
    elif idea == "opt_schatten_pure_all":
        core_wd_mult = 0.0
        schatten_pure_strength = 0.15
        spectral_scope = "all"
    elif idea == "opt_core_l2_plain":
        # Exact 1-to-1 causal twin of opt_schatten_pure: plain decoupled L2 W -= 0.10*lr*W (no cubic term)
        core_wd_mult = 0.0
        plain_l2_strength = 0.10
    elif idea == "opt_core_wd_8x":
        core_wd_mult = 8.0
    elif idea == "opt_all_wd_8x":
        core_wd_mult = 8.0
        untied_wd_mult = 8.0
    elif idea == "opt_core_wd_16x":
        core_wd_mult = 16.0
    elif idea == "opt_cautious_schatten":
        # Cautious-masked pure cubic Schatten-4 WD (exact twin of Cautious L2 WD with cubic spectrum!)
        core_wd_mult = 0.0
        cautious_schatten_strength = 0.25
    elif idea == "opt_wd8x_late":
        # Clean orthogonal combo: 8x core cautious WD + late-loop BPTT grad discounting
        core_wd_mult = 8.0
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.25, 0.50, 0.75, 1.00]
    elif idea == "opt_cubic_frob_wd":
        # Self-regulating cubic Frobenius decay W -= lam*lr*alpha^2*W (0 extra GEMMs!)
        core_wd_mult = 0.0
        cubic_frob_strength = 0.12
    elif idea == "opt_core_ns2":
        core_ns_steps = 2
    elif idea == "opt_attn_wd12_mlp4":
        # Matrix-type split WD tailored to Attention rank collapse (0.58) vs healthy MLP (0.86)
        core_attn_wd_mult = 12.0
        core_wd_mult = 4.0
    elif idea == "opt_core_beta1_090":
        core_momentum = 0.90
    elif idea == "opt_dual_mom":
        # Dual-timescale core momentum: 0.5 * Nesterov(0.80) + 0.5 * Nesterov(0.95)
        use_dual_mom = True
        core_momentum = 0.0
    elif idea == "opt_stiefel_tan":
        # Soft Riemannian Stiefel tangent-space gradient projection before Muon
        stiefel_tan_strength = 0.5
    elif idea == "opt_loop_coh_pos":
        # Per-row cross-loop gradient coherence preconditioner: scale rows by c_i^{+0.5}
        needs_per_loop = True
        per_loop_mode = "coh_pos"
    elif idea == "opt_loop_coh_neg":
        # Inverse per-row cross-loop coherence: scale rows by (c_i + 0.1)^{-0.5}
        needs_per_loop = True
        per_loop_mode = "coh_neg"
    elif idea == "opt_loop_mgda":
        # 4x4 cross-loop Gram-whitened Pareto gradient (K^{-1} d) in gradient space
        needs_per_loop = True
        per_loop_mode = "mgda"
    elif idea == "opt_loop_cos_gate":
        # Dynamic hyperparameter-free cross-loop cosine gating: w_t = clamp(1 + 2*cos(G_t, G_last), 0.1)
        needs_per_loop = True
        per_loop_mode = "cos_gate"
    elif idea == "opt_all_wd8x_late":
        # Champion combo: all-layer 8x cautious WD + late-loop BPTT grad discounting
        core_wd_mult = 8.0
        untied_wd_mult = 8.0
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.25, 0.50, 0.75, 1.00]
    elif idea == "opt_schatten_all_020":
        schatten_wd_strength = 0.20
        spectral_scope = "all"
    elif idea == "opt_schatten_attn_020":
        schatten_wd_strength = 0.20
        spectral_scope = "attn"
    elif idea == "opt_spectral_clip":
        spectral_clip_strength = 0.50
        spectral_clip_ratio = 1.5
    elif idea == "opt_core_wd_3x":
        core_wd_mult = 3.0
    elif idea == "opt_core_lr_05":
        core_lr_mult = 0.5
    elif idea == "opt_core_lr_15":
        core_lr_mult = 1.5
    elif idea == "opt_core_beta1_085":
        core_momentum = 0.85
    elif idea == "opt_core_beta1_070":
        core_momentum = 0.70
    elif idea == "opt_core_beta1_050":
        core_momentum = 0.50
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
        needs_per_loop = True
        per_loop_mode = "pcgrad"
    elif idea == "opt_schatten_pcgrad":
        schatten_wd_strength = 0.20
        needs_per_loop = True
        per_loop_mode = "pcgrad"
    elif idea == "opt_loop_late_wt":
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.25, 0.50, 0.75, 1.00]
    elif idea == "opt_loop_exp_05":
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.125, 0.25, 0.50, 1.00]
    elif idea == "opt_loop_last_only":
        # Full 4-loop BPTT through state h, but ONLY accumulate parameter gradient from final loop!
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.0, 0.0, 0.0, 1.0]
    elif idea == "opt_loop_early_wt":
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [1.50, 1.25, 1.00, 0.75]
    elif idea == "opt_combo_sp_b85":
        # Pure optimizer combo (0 backward hook, 0 extra state): Schatten-4 (0.15) + Core beta1=0.80
        core_wd_mult = 0.0
        schatten_pure_strength = 0.15
        core_momentum = 0.80
    elif idea == "opt_combo_sp_late":
        # Schatten-4 (0.15) + Late-loop parameter grad discounting [0.25, 0.5, 0.75, 1.0]
        core_wd_mult = 0.0
        schatten_pure_strength = 0.15
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.25, 0.50, 0.75, 1.00]
    elif idea == "opt_combo_all3":
        # Unified Loop-Muon: Pure Schatten-4 (0.15) + Core beta1=0.80 + Late-loop grad discount
        core_wd_mult = 0.0
        schatten_pure_strength = 0.15
        core_momentum = 0.80
        needs_per_loop = True
        per_loop_mode = "weighted"
        per_loop_weights = [0.25, 0.50, 0.75, 1.00]
    elif idea == "teon_loop_m1":
        # Mode-1 Loop-TEON: unfold [M_0..M_{s-1}] -> [m, s*n] -> single Polar Express -> fold & sum!
        needs_per_loop = True
        tensor_muon_mode = "loop_m1"
    elif idea == "teon_loop_m2":
        # Mode-2 Loop-TEON: unfold [M_0;..;M_{s-1}] -> [s*m, n] -> single Polar Express -> fold & sum!
        needs_per_loop = True
        tensor_muon_mode = "loop_m2"
    elif idea == "teon_loop_tsvd_hosvd":
        # Data-driven Loop-tSVD / Tucker-3 (Loop-TUON): mode-3 eigh of sxs loop Gram -> slice PE -> inv!
        needs_per_loop = True
        tensor_muon_mode = "loop_tsvd_hosvd"
    elif idea == "teon_loop_dct_ramp":
        # Frequency-domain Loop-tSVD (DCT-II transform product) + depth-ramped inverse aggregation!
        needs_per_loop = True
        tensor_muon_mode = "loop_dct_ramp"
    elif idea == "teon_layer_m1":
        # Classic Cross-Layer TEON baseline: unfold [core0..core5] per proj -> [m, L*n] -> PE -> fold!
        tensor_muon_mode = "layer_m1"
    else:
        raise ValueError(f"Unknown optimizer idea: {idea}")

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    batch_tokens = args.batch_size * seq_len
    opt = build_loop_muon_optimizer(model, batch_tokens, core_param_ids,
                                    core_lr_mult=core_lr_mult, core_momentum=core_momentum,
                                    core_wd_mult=core_wd_mult, core_beta2=core_beta2,
                                    untied_wd_mult=untied_wd_mult, core_ns_steps=core_ns_steps,
                                    core_attn_ids=core_attn_ids, core_attn_wd_mult=core_attn_wd_mult,
                                    exclude_core_from_muon=(tensor_muon_mode is not None))

    # Tensor-Muon state buffers (exact NorMuon v_buf + per-loop or per-layer momentum tensor)
    tensor_m_bufs = {}
    tensor_v_bufs = {}
    core_by_proj = defaultdict(list)
    for layer in range(first_l, last_l + 1):
        blk = model.transformer.h[layer]
        for pname, mod in (("c_q", blk.attn.c_q), ("c_k", blk.attn.c_k), ("c_v", blk.attn.c_v),
                           ("c_proj", blk.attn.c_proj), ("c_fc", blk.mlp.c_fc), ("mlp_proj", blk.mlp.c_proj)):
            core_by_proj[pname].append(mod.weight)
    q_dct = dct_matrix(cfg.huginn_backprop_loops, device)

    if spectral_scope == "all" and schatten_pure_strength > 0.0:
        for g in opt.param_groups:
            if g.get("kind") == "muon":
                g["weight_decay"] = 0.0
                g["initial_weight_decay"] = 0.0

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
    eval_d_mid = (cfg.max_loops,)
    eval_d_final = (1,) if cfg.max_loops == 1 else ((4, 8, 16) if cfg.max_loops == 8 else (4, 8, 16))
    bpb0, col0 = eval_multidepth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                 depths=eval_d_mid, max_rows=512)
    col0 = col0 or 0.0
    print(f"[{idea}|{args.backbone}|r={cfg.max_loops}] params={n_params/1e6:.1f}M step=0 val_bpb@{cfg.max_loops}={bpb0[str(cfg.max_loops)]:.5f} collapse={col0:.4f}",
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
                    elif per_loop_mode in ("coh_pos", "coh_neg"):
                        # Per-row cross-loop gradient coherence c_i = ||sum_t G_{t,i}|| / sum_t ||G_{t,i}||
                        for cp in plist:
                            sl = LG_CTX.slots[cp]
                            G_stack = torch.stack([sl[t] for t in range(s_loops)], dim=0)  # [s, m, n]
                            G_sum = G_stack.sum(dim=0)                                      # [m, n]
                            num = G_sum.norm(dim=-1, keepdim=True)                          # [m, 1]
                            den = G_stack.norm(dim=-1).sum(dim=0, keepdim=True).T.clamp_min(1e-8)
                            coh = (num / den).clamp(0.0, 1.0)
                            if per_loop_mode == "coh_pos":
                                w = (coh + 0.05).pow(0.5)
                            else:
                                w = (coh + 0.10).pow(-0.5)
                            w = w / w.mean().clamp_min(1e-6)
                            cp.grad = (G_sum * w).to(cp.dtype)
                    elif per_loop_mode == "mgda":
                        # 4x4 cross-loop Frobenius Gram whitening: solve (K + eps*diag(K)) w = ||G||^2
                        for cp in plist:
                            sl = LG_CTX.slots[cp]
                            G_stack = torch.stack([sl[t] for t in range(s_loops)], dim=0)
                            flat = G_stack.reshape(s_loops, -1)
                            K = flat @ flat.T                                               # [s, s]
                            d = K.diagonal().clamp_min(1e-12)
                            K_reg = K + 0.25 * torch.diag(d)
                            w = torch.linalg.solve(K_reg, d).clamp(0.0, 4.0)
                            w = w * (s_loops / w.sum().clamp_min(1e-6))
                            comb = (G_stack * w.view(s_loops, 1, 1)).sum(dim=0)
                            cp.grad = comb.to(cp.dtype)
                    elif per_loop_mode == "cos_gate":
                        # Dynamic hyperparameter-free cross-loop cosine gating vs last loop
                        for cp in plist:
                            sl = LG_CTX.slots[cp]
                            g_ref = sl[s_loops - 1]
                            ref_n = g_ref.norm().clamp_min(1e-8)
                            ws = []
                            for t in range(s_loops):
                                g_t = sl[t]
                                c_t = (g_t * g_ref).sum() / (g_t.norm().clamp_min(1e-8) * ref_n)
                                ws.append(float(max(0.1, min(2.0, 1.0 + 2.0 * c_t.item()))))
                            w_mean = sum(ws) / len(ws)
                            comb = sum((ws[t] / w_mean) * sl[t] for t in range(s_loops))
                            cp.grad = comb.to(cp.dtype)

        # Compute pure-algorithmic Tensor-Muon (Loop-TEON / Loop-tSVD) updates before clearing slots!
        pending_tensor_updates = []
        if tensor_muon_mode is not None and tensor_muon_mode != "layer_m1":
            with torch.no_grad():
                bscale_now = math.sqrt(batch_tokens / 524288.0)
                muon_lr_now = 0.02 * bscale_now * mult
                muon_wd_now = 0.28 * bscale_now * 0.5 * mult
                s_loops = cfg.huginn_backprop_loops
                for shape, plist in core_by_shape.items():
                    K, m, n = len(plist), shape[0], shape[1]
                    G_t = torch.stack(
                        [torch.stack([LG_CTX.slots[cp][t] for t in range(s_loops)], dim=0) for cp in plist],
                        dim=0,
                    )  # [K, s, m, n]
                    # Also populate cp.grad sum so grad_norm logging stays meaningful
                    G_sum = G_t.sum(dim=1)
                    for idx_p, cp in enumerate(plist):
                        cp.grad = G_sum[idx_p].to(cp.dtype)

                    if shape not in tensor_m_bufs:
                        tensor_m_bufs[shape] = torch.zeros_like(G_t)
                        v_shape = (K, m, 1) if m >= n else (K, 1, n)
                        tensor_v_bufs[shape] = torch.zeros(v_shape, device=device, dtype=torch.float32)
                    M_buf = tensor_m_bufs[shape]
                    M_buf.lerp_(G_t, 1.0 - 0.95)
                    M_nes = G_t.lerp(M_buf, 0.95)

                    if tensor_muon_mode == "loop_m1":
                        X_u = M_nes.permute(0, 2, 1, 3).reshape(K, m, s_loops * n)
                        U_u = polar_express_raw(X_u, ns_steps=5, use_muon_eq=True)
                        U = U_u.view(K, m, s_loops, n).permute(0, 2, 1, 3).sum(dim=1)
                    elif tensor_muon_mode == "loop_m2":
                        X_u = M_nes.reshape(K, s_loops * m, n)
                        U_u = polar_express_raw(X_u, ns_steps=5, use_muon_eq=True)
                        U = U_u.view(K, s_loops, m, n).sum(dim=1)
                    elif tensor_muon_mode == "loop_tsvd_hosvd":
                        flat = M_nes.reshape(K, s_loops, m * n)
                        Gram = torch.bmm(flat, flat.mT)
                        tr = Gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-12)
                        Gram = Gram + (1e-6 * tr).unsqueeze(-1) * torch.eye(s_loops, device=device)
                        _, Q = torch.linalg.eigh(Gram)
                        M_hat = torch.einsum("ktr,ktmn->krmn", Q, M_nes)
                        U_hat = polar_express_raw(M_hat.reshape(K * s_loops, m, n), ns_steps=5, use_muon_eq=True).view(K, s_loops, m, n)
                        U = torch.einsum("ktr,krmn->ktmn", Q, U_hat).sum(dim=1)
                    elif tensor_muon_mode == "loop_dct_ramp":
                        M_hat = torch.einsum("rt,ktmn->krmn", q_dct, M_nes)
                        U_hat = polar_express_raw(M_hat.reshape(K * s_loops, m, n), ns_steps=5, use_muon_eq=True).view(K, s_loops, m, n)
                        U_sl = torch.einsum("rt,krmn->ktmn", q_dct, U_hat)
                        w_ramp = torch.linspace(0.25, 1.0, s_loops, device=device).view(1, s_loops, 1, 1)
                        U = (U_sl * w_ramp).sum(dim=1)
                    pending_tensor_updates.append((plist, U, tensor_v_bufs[shape], muon_lr_now, muon_wd_now))

        LG_CTX.zero_slots()
        LG_CTX.enabled = False
        LG_CTX.suppress_default_grad = False

        # Pre-Muon gradient transforms on tied core matrices (Stiefel tangent / dual-timescale momentum)
        if stiefel_tan_strength > 0.0:
            with torch.no_grad():
                for shape, plist in core_by_shape.items():
                    min_dim = min(shape[0], shape[1])
                    for cp in plist:
                        if cp.grad is None:
                            continue
                        W = cp.float()
                        G = cp.grad.float()
                        alpha = W.norm().clamp_min(1e-6) / (min_dim ** 0.5)
                        Wn = W / alpha
                        if shape[0] >= shape[1]:
                            M_wg = Wn.mT @ G
                            sym = 0.5 * (M_wg + M_wg.mT)
                            G_new = G - stiefel_tan_strength * (Wn @ sym)
                        else:
                            M_wg = G @ Wn.mT
                            sym = 0.5 * (M_wg + M_wg.mT)
                            G_new = G - stiefel_tan_strength * (sym @ Wn)
                        cp.grad.copy_(G_new.to(cp.grad.dtype))

        if use_dual_mom:
            with torch.no_grad():
                for cp in core_params:
                    if cp.grad is None:
                        continue
                    g = cp.grad.float()
                    pair = dual_mom_bufs.get(cp)
                    if pair is None:
                        pair = (torch.zeros_like(g), torch.zeros_like(g))
                        dual_mom_bufs[cp] = pair
                    m_fast, m_slow = pair
                    m_fast.lerp_(g, 0.20)   # beta=0.80
                    m_slow.lerp_(g, 0.05)   # beta=0.95
                    nes = 0.5 * g.lerp(m_fast, 0.80) + 0.5 * g.lerp(m_slow, 0.95)
                    cp.grad.copy_(nes.to(cp.grad.dtype))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)

        W_before = None
        if cautious_schatten_strength > 0.0:
            with torch.no_grad():
                W_before = {shape: torch.stack([cp.float() for cp in plist], dim=0)
                            for shape, plist in spec_by_shape.items()}

        opt.step()

        if tensor_muon_mode is not None:
            with torch.no_grad():
                if tensor_muon_mode == "layer_m1":
                    bscale_now = math.sqrt(batch_tokens / 524288.0)
                    muon_lr_now = 0.02 * bscale_now * mult
                    muon_wd_now = 0.28 * bscale_now * 0.5 * mult
                    for pname, plist in core_by_proj.items():
                        L = len(plist)
                        m, n = plist[0].shape
                        G_l = torch.stack([cp.grad.float() for cp in plist], dim=0)
                        if pname not in tensor_m_bufs:
                            tensor_m_bufs[pname] = torch.zeros_like(G_l)
                            v_shape = (L, m, 1) if m >= n else (L, 1, n)
                            tensor_v_bufs[pname] = torch.zeros(v_shape, device=device, dtype=torch.float32)
                        M_buf = tensor_m_bufs[pname]
                        M_buf.lerp_(G_l, 1.0 - 0.95)
                        M_nes = G_l.lerp(M_buf, 0.95)
                        X_u = M_nes.permute(1, 0, 2).reshape(1, m, L * n)
                        U_u = polar_express_raw(X_u, ns_steps=5, use_muon_eq=True)
                        U = U_u.view(m, L, n).permute(1, 0, 2)
                        apply_muon_update_exact(plist, U, tensor_v_bufs[pname],
                                                lr=muon_lr_now, wd=muon_wd_now, beta2=0.9)
                else:
                    for plist, U, v_buf, mlr, mwd in pending_tensor_updates:
                        apply_muon_update_exact(plist, U, v_buf, lr=mlr, wd=mwd, beta2=0.9)

        # Post-step weight-space spectral conditioning (1x state, 0 inf cost!)
        if (polar_wd_strength > 0.0 or schatten_wd_strength > 0.0 or schatten_pure_strength > 0.0
                or cautious_schatten_strength > 0.0 or cubic_frob_strength > 0.0
                or plain_l2_strength > 0.0 or spectral_clip_strength > 0.0):
            with torch.no_grad():
                core_lr_now = 0.02 * math.sqrt(batch_tokens / 524288.0) * mult
                for shape, plist in spec_by_shape.items():
                    W = torch.stack([cp.float() for cp in plist], dim=0)   # [K, m, n]
                    min_dim = min(shape[0], shape[1])
                    alpha = W.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6) / (min_dim ** 0.5)
                    if plain_l2_strength > 0.0:
                        W_new = W - (plain_l2_strength * core_lr_now) * W
                    elif cubic_frob_strength > 0.0:
                        # Self-regulating cubic Frobenius decay: W -= lam*lr*alpha^2*W
                        W_new = W - (cubic_frob_strength * core_lr_now) * (alpha.square().clamp_max(5.0)) * W
                    elif polar_wd_strength > 0.0:
                        W_iso = alpha * polar_express(W, ns_steps=5)
                        step_c = polar_wd_strength * core_lr_now
                        W_new = W - step_c * (W - W_iso)
                    elif schatten_wd_strength > 0.0 or schatten_pure_strength > 0.0 or cautious_schatten_strength > 0.0:
                        Wn = W / alpha
                        if shape[0] >= shape[1]:
                            G_w = Wn.mT @ Wn
                            cubic = Wn @ G_w
                        else:
                            G_w = Wn @ Wn.mT
                            cubic = G_w @ Wn
                        if cautious_schatten_strength > 0.0:
                            # Cautious mask: decay ONLY where Muon step direction aligns with W!
                            mask = ((W_before[shape] - W) * W >= 0).float()
                            step_c = cautious_schatten_strength * core_lr_now
                            W_new = W - step_c * alpha * cubic.clamp(-5.0, 5.0) * mask
                        elif schatten_pure_strength > 0.0:
                            step_c = schatten_pure_strength * core_lr_now
                            W_new = W - step_c * alpha * cubic.clamp(-5.0, 5.0)
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
            depths = eval_d_final if (step + 1 == args.steps) else eval_d_mid
            rows = 4096 if (step + 1 == args.steps) else 1024
            bpbs, col = eval_multidepth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                        depths=depths, max_rows=rows)
            col = col or 0.0
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
        "max_loops": cfg.max_loops,
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
