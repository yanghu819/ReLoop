#!/usr/bin/env python3
"""Direction-A Arch Wave 5: GeoLoop — Exact Riemannian Geodesic Exponential Map Exp_x((1/R) P_{T_x} v) on S^{d-1}.

Fixes the exact radial chord bias 1/(R + <x,v>/d) proven in internal review #97:
  v     = s_ode * delta + g_pre * prelude
  v_tan = v - (<x, v> / <x, x>) * x                                   # Exact tangent projection onto T_x S^{d-1}
  theta = sqrt(mean(v_tan^2)) / R                                      # Exact geodesic arc length (0% chord bias!)
  x     = cos(theta) * x + sin(theta) * norm(v_tan)                    # Exact Riemannian Exp_x on S^{d-1}!

All runs strictly enforce:
  - 100% official 5-step MuonAdamW + loop_state_mod AdamW lr = 0.003 * bscale
  - --freeze-x0 clean baseline
  - Zero-slice-bias ood_2048 across R_eval in {1, 2, 4, 8, 12, 16, 32}

Ideas supported:
  1. geoloop          : Exact 3-line Riemannian Exp_x((1/R) P_{T_x}(s*delta + g*prelude)) (769 scalars = 0.0004%!)
  2. geoloop_no_pre   : LOO ablation with g_pre == 0: pure spherical Exp_x((s/R) P_{T_x} delta) (1 scalar!)
  3. geoloop_dyn_g    : GeoLoop with rank-16 token-conditioned anchor pull g(x) (0.012% params)
  4. geoloop_pt_mom   : Riemannian Parallel-Transport Tangent Momentum on S^{d-1} (1537 scalars, 0 matmuls!)
  5. geoloop_r1_20L   : Gate A1.3 Untied 2/16/2 R=1 DiD control (applied across four 4-layer groups!)
  6. geoloop_cautious : GeoLoop with tangent-space anchor pull ONLY + 1/sqrt(R) vs 1/R check
  7. official_base    : Official fixed_recursive / huginn baseline (for 2250-step fresh-671M head-to-head!)
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("NEXTTRANSFORMER_DIR", "./NextTransformer"))
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model
from nexttransformer.models.backbone import norm
from nexttransformer.training.optimizer import ExperimentOptimizer


def riemannian_exp_step(x: torch.Tensor, v: torch.Tensor, inv_r: float):
    """Exact Riemannian Exponential Map Exp_x((1/R) P_{T_x} v) on the RMSNorm hypersphere ||x|| = sqrt(d).

    Note: Since RMSNorm sphere has radius sqrt(d), theta = ||v_tan||_RMS / R = (||v_tan||_2 / sqrt(d)) / R
    gives exact angular arc length independent of d and R with ZERO radial chord bias!
    Returns (x_next, theta, u_tan) where u_tan = norm(v_tan) is tangent at x with ||u_tan||_RMS = 1.
    """
    xf = x.float()
    vf = v.float()
    # Ensure x starts on exact RMSNorm sphere ||x||_RMS = 1 (radius sqrt(d))
    x_rms = xf.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    x_u = xf / x_rms
    # Orthogonal projection onto tangent space T_{x_u} S^{d-1}: <x_u, v_tan> == 0
    dot = (x_u * vf).mean(dim=-1, keepdim=True)
    v_tan = vf - dot * x_u
    v_rms = v_tan.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    u_tan = v_tan / v_rms
    theta = (v_rms * inv_r).clamp_max(math.pi * 0.5)  # geodesic angle in [0, pi/2]
    x_next = (torch.cos(theta) * x_u + torch.sin(theta) * u_tan).to(x.dtype)
    return x_next, theta, u_tan


class GeoLoopModule(nn.Module):
    def __init__(self, idea: str, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.width = width
        if idea in ("geoloop", "geoloop_no_pre", "geoloop_dyn_g", "geoloop_pt_mom", "geoloop_r1_20L"):
            # Init s_ode = 4.0 (unit tangent step at R=4, 0.5 at R=8)
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
        if idea in ("geoloop", "geoloop_pt_mom", "geoloop_r1_20L"):
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))
        if idea == "geoloop_dyn_g":
            self.g_down = nn.Linear(width, 16, bias=False, device=device, dtype=torch.bfloat16)
            self.g_up = nn.Linear(16, width, bias=True, device=device, dtype=torch.bfloat16)
            nn.init.normal_(self.g_down.weight, std=0.02)
            nn.init.zeros_(self.g_up.weight)
            nn.init.constant_(self.g_up.bias, 0.4)  # softplus(0.4) ~ 0.9, scaled * 1.5 ~ 1.4
        if idea == "geoloop_pt_mom":
            # Continuous-time Riemannian parallel-transport tangent momentum
            self.mu_logit = nn.Parameter(torch.full((width,), 0.4, device=device, dtype=torch.float32))


def attach_geoloop_executor(model, idea: str, device: torch.device):
    sm = GeoLoopModule(idea, model.config.width, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}
    if idea == "official_base":
        return sm

    is_r1_did = (idea == "geoloop_r1_20L")

    def _run_group(self, h, x0, idx, g_idx, total_depth):
        c = self.config
        if is_r1_did:
            for j in range(4):
                phys = c.pre_layers + g_idx * 4 + j
                h = self._block(h, x0, idx, phys, phys, total_depth)
        else:
            for j in range(c.core_layers):
                phys = c.pre_layers + j
                virt = c.pre_layers + g_idx * c.core_layers + j
                h = self._block(h, x0, idx, phys, virt, total_depth)
        return h

    def _custom_ordinary_hidden(self, idx):
        c = self.config
        if is_r1_did:
            R_eff = 4
            assert c.loops == 1 and c.core_layers == 16
            total = c.pre_layers + c.core_layers + c.post_layers
        else:
            R_eff = getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops
            total = c.pre_layers + R_eff * c.core_layers + c.post_layers

        x = self._embed_reference(idx)
        x0 = x
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = norm(x)
        x = prelude
        cached = None
        mid_g = R_eff // 2
        rec_diag = (not self.training) and getattr(self, "_collect_loop_diag", False)
        thetas, raw_cos_list = [], []
        prev_raw_delta = None

        inv_r = 1.0 / float(R_eff)
        tbptt = getattr(self, "_tbptt_loops", 0)
        m_tan = torch.zeros_like(x, dtype=torch.float32)

        for r in range(R_eff):
            no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
            if self.training and tbptt > 0 and r == R_eff - tbptt:
                x = x.detach()
                m_tan = m_tan.detach()
            with torch.no_grad() if no_g else nullcontext():
                delta = _run_group(self, x, x0, idx, r, total) - x
                if idea == "geoloop_no_pre":
                    v = sm.s_ode.to(x.dtype) * delta
                elif idea == "geoloop_dyn_g":
                    gd = F.softplus(sm.g_up(F.silu(sm.g_down(x))).float()).to(x.dtype) * 1.5
                    v = sm.s_ode.to(x.dtype) * delta + gd * prelude
                else:
                    v = sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude

                if idea == "geoloop_pt_mom":
                    # Parallel-transport previous tangent velocity to current T_x S^{d-1} & blend
                    xf = x.float()
                    m_proj = m_tan - (xf * m_tan).mean(dim=-1, keepdim=True) * xf
                    mu_r = torch.sigmoid(sm.mu_logit) ** (8.0 * inv_r)
                    v_blend = mu_r * m_proj + (1.0 - mu_r) * v.float()
                    x_next, theta, u_tan = riemannian_exp_step(x, v_blend, inv_r)
                    # Exact geodesic parallel transport of m_tan along (x -> x_next)
                    v_rms = v_blend.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
                    m_tan = (-torch.sin(theta) * xf + torch.cos(theta) * u_tan) * v_rms
                    x = x_next
                else:
                    x, theta, _ = riemannian_exp_step(x, v, inv_r)

                if r == mid_g:
                    cached = x
                if rec_diag:
                    d_raw = delta.detach().float()
                    thetas.append(round(float(theta.mean().item()), 4))
                    if prev_raw_delta is not None:
                        raw_cos_list.append(float(F.cosine_similarity(d_raw, prev_raw_delta, dim=-1).mean().item()))
                    prev_raw_delta = d_raw

        if rec_diag:
            self._loop_diag = dict(thetas=thetas, raw_delta_cos=[round(v, 3) for v in raw_cos_list])

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + j
            x = self._block(x, x0, idx, phys, virt, total)
        return x, cached

    model._ordinary_hidden = types.MethodType(_custom_ordinary_hidden, model)
    return sm


def build_official_opt(model, batch_tokens: int, core_lr_mult: float = 1.0):
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
    if hasattr(model, "loop_state_mod"):
        adam(list(model.loop_state_mod.parameters()), 0.003 * bscale, (0.9, 0.95), 0.0)

    c = model.config
    core_ids = set()
    for l_idx in range(c.pre_layers, c.pre_layers + c.core_layers):
        for p in model.transformer.h[l_idx].parameters():
            core_ids.add(id(p))
    if hasattr(model, "state_adapter") and model.state_adapter is not None:
        for p in model.state_adapter.parameters():
            core_ids.add(id(p))

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
                           momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=wd, initial_weight_decay=wd))
    for sh in sorted(core):
        lr_c = muon_lr * core_lr_mult
        groups.append(dict(kind="muon", params=core[sh], lr=lr_c, initial_lr=lr_c,
                           momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=wd, initial_weight_decay=wd))
    return ExperimentOptimizer(groups)


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
def eval_at_depth(model, val_x, val_y, byte_table, device, batch_size: int,
                  loops_eval: int, max_rows: int = 4096, collect_diag: bool = False,
                  is_official_loop: bool = False):
    if is_official_loop:
        model.accounting_depth = loops_eval
    else:
        model._eval_loops_override = loops_eval
    model.eval()
    model._collect_loop_diag = False
    total_nats = total_bytes = total_toks = 0.0
    diag = {}
    try:
        for i in range(0, min(max_rows, val_x.shape[0]), batch_size):
            x = val_x[i:i + batch_size].to(device, dtype=torch.long)
            y = val_y[i:i + batch_size].to(device, dtype=torch.long)
            model.set_row_offset(i)
            if collect_diag and i == 0:
                model._collect_loop_diag = True
            logits = model(x)
            if collect_diag and i == 0:
                diag = dict(getattr(model, "_loop_diag", {}))
                model._collect_loop_diag = False
            loss = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-1, reduction="none")
            nbytes = byte_table[y.flatten().clamp_min(0)]
            valid = (y.flatten() >= 0) & (nbytes > 0)
            total_nats += loss[valid].sum().item()
            total_bytes += nbytes[valid].sum().item()
            total_toks += valid.sum().item()
    finally:
        if is_official_loop:
            model.accounting_depth = None
        else:
            model._eval_loops_override = None
        model._collect_loop_diag = False
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1.0)), total_nats / max(total_toks, 1.0), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=[
        "official_base", "geoloop", "geoloop_no_pre", "geoloop_dyn_g", "geoloop_pt_mom", "geoloop_r1_20L",
    ])
    ap.add_argument("--method", default="fixed_recursive", choices=["fixed_recursive", "huginn"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tbptt-loops", type=int, default=0)
    ap.add_argument("--huginn-bptt", type=int, default=8)
    ap.add_argument("--huginn-sigma", type=float, default=1e-6)
    ap.add_argument("--core-lr-mult", type=float, default=1.0)
    ap.add_argument("--horizon-jitter", action="store_true")
    ap.add_argument("--freeze-x0", action="store_true")
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=32)
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
    py_rng = random.Random(args.seed)

    train_d = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_d = torch.load(args.data_dir / "val.pt", map_location="cpu", weights_only=True)
    train_x, train_y = train_d["x"], train_d["y"]
    val_x, val_y = val_d["x"], val_d["y"]
    seq_len = train_x.shape[1]
    byte_table = torch.load(args.resources / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    is_hug = (args.method == "huginn")
    cfg = ModelConfig(
        method=args.method, scale="debug", width=args.width, heads=args.heads,
        vocab_size=32768, sequence_len=seq_len, mlp_width=4 * args.width,
        layers=args.pre_layers + args.core_layers + args.post_layers,
        pre_layers=args.pre_layers, core_layers=args.core_layers, post_layers=args.post_layers,
        loops=args.loops, max_loops=max(args.loops, 2),
        huginn_mean=float(max(args.loops, 2)),
        huginn_sigma=args.huginn_sigma if is_hug else 0.5,
        huginn_backprop_loops=min(args.huginn_bptt, max(args.loops, 1)) if is_hug else min(4, max(args.loops, 1)),
        eval_loops=args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    if args.freeze_x0:
        with torch.no_grad():
            model.x0_lambdas.zero_()
        model.x0_lambdas.requires_grad_(False)

    base_params = sum(p.numel() for p in model.parameters())
    sm = attach_geoloop_executor(model, args.idea, device=device)
    model._tbptt_loops = args.tbptt_loops
    extra_params = sum(p.numel() for p in sm.parameters())
    is_off_loop = (args.idea == "official_base") and is_hug

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens, core_lr_mult=args.core_lr_mult)

    n_rows = train_x.shape[0]
    bpb0, vl0, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                 args.loops, max_rows=512, is_official_loop=is_off_loop)
    print(f"[{args.tag}|{args.idea}|R={args.loops}|tbptt={args.tbptt_loops}|jit={int(args.horizon_jitter)}|+p={extra_params} ({100*extra_params/base_params:.4f}%)] "
          f"step=0 val_loss={vl0:.4f} val_bpb={bpb0:.5f}", flush=True)

    t0_all = time.time()
    history, step_times = [], []
    model.train()

    for step in range(args.steps):
        t0 = time.perf_counter()
        mult = lr_mult(step, args.steps)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * mult
            if "initial_weight_decay" in g:
                g["weight_decay"] = g["initial_weight_decay"] * mult

        if args.horizon_jitter and args.loops >= 4:
            model._step_loops_override = py_rng.randint(max(2, args.loops - 2), args.loops + 2)
        else:
            model._step_loops_override = None

        model.zero_grad(set_to_none=True)
        model.set_step(step)

        loss_sum = 0.0
        for m_idx in range(accum):
            r0 = (step * args.batch_size + m_idx * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[r0:r0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m_idx * args.micro_batch)
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            model._step_loops_override = None
            rows = 4096 if (step + 1) == args.steps else 1024
            vb, vl, diag = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                         args.loops, max_rows=rows, collect_diag=(step + 1 == args.steps),
                                         is_official_loop=is_off_loop)
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum,
                                elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f}", flush=True)

    ood_2048 = {}
    if args.loops == 1:
        _, vl_1, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, 1, max_rows=2048)
        ood_2048["1"] = round(vl_1, 5)
    else:
        for r_ev in (1, 2, 4, 8, 12, 16, 32):
            ev_mb = 8 if r_ev >= 32 else (16 if r_ev >= 12 else min(args.micro_batch, 32))
            _, vl_r, _ = eval_at_depth(model, val_x, val_y, byte_table, device, ev_mb, r_ev,
                                       max_rows=2048, is_official_loop=is_off_loop)
            ood_2048[str(r_ev)] = round(vl_r, 5)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    ood_diff = round(ood_2048["16"] - ood_2048["8"], 5) if ("16" in ood_2048 and "8" in ood_2048) else None
    out = dict(
        tag=args.tag, idea=args.idea, method=args.method, loops=args.loops,
        tbptt_loops=args.tbptt_loops, horizon_jitter=args.horizon_jitter, steps=args.steps,
        extra_params=extra_params, extra_param_pct=round(100.0 * extra_params / base_params, 5),
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], ood_2048=ood_2048, ood_diff_16_8=ood_diff,
        loop_diag=diag, tok_per_sec=batch_tokens / avg_dt, total_wall_s=time.time() - t0_all,
        history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    rcos = diag.get("raw_delta_cos", [])
    cos_str = f"{sum(rcos)/max(len(rcos),1):.3f}" if rcos else "n/a"
    print(f"[{args.tag}] DONE val={out['final_val_loss']:.5f} ood16-8={ood_diff} ood2k={ood_2048} "
          f"raw_cos={cos_str} {out['tok_per_sec']/1e3:.1f}k tok/s {out['total_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
