#!/usr/bin/env python3
"""Direction-A Arch Wave 3: Mandatory Auditor Gate-A1 Controls + Novel Decoupled Explorer/Symplectic Breakthroughs.

Strictly enforces Carbon-Auditor #91 code hygiene:
  - Identical 100% official 5-step MuonAdamW + loop_state_mod AdamW lr = 0.003 * bscale (matching Wave 1!)
  - Extended OOD evaluation R_eval in {1, 2, 4, 8, 12, 16, 32}
  - True pre-norm core update cosine diag (fixing post-norm chord bias!)

Supports BOTH:
  A. Mandatory Official Baselines (--idea official_base with --method fixed_recursive/huginn/deeploop/ouro,
     optional --huginn-bptt / --huginn-sigma / --steps for exact full-BPTT & iso-wall controls!)
  B. Prior-Art Control: loopnorm_ls ([R, d] per-loop RMSNorm + LayerScale, clamped at R-1 for OOD!)
  C. Novel Bitter-Lesson Architectures (0 lookup tables, <= 5 lines, 0.0004% to 0.04% params):
     - explorer_highway_r : 3-line Decoupled Spherical Explorer z + prefix-causal 1/sqrt(r+1) Highway h (769 scalars!)
     - explorer_highway_R : 3-line Decoupled Spherical Explorer z + midpoint 1/R Highway h (769 scalars!)
     - polyak_geodesic    : 3-line Spherical Explorer z + linear Polyak barycenter norm(sum (r+1)*z_r) (768 scalars!)
                            (supports --horizon-jitter for zero-cost Uniform{6..10} mean-8 anytime OOD training!)
     - prelude_true_ode   : Bug-fixed Spherical ODE scaling BOTH tangent forces: norm(x + (1/R)*(s*delta + g*pre))
     - euler_attnpool     : Clean LOO of roloop_ode_attnpool (Euler-ODE + Loop-AttnPool, DROPPING dead RoLoop!)
     - symplectic_pad     : Dual-register (h_r, s_r) state-driven recurrence (0 prelude / 0 DEQ overlap!)
     - symplectic_ode_pool: Dual-register (h_r, s_r) + exact 1/R write scaling + Loop-AttnPool!
     - symplectic_polyak  : Dual-register (h_r, s_r) + linear Polyak readout (0% extra readout params, prefix-causal!)
"""
import argparse
from collections import defaultdict
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

sys.path.insert(0, "/root/nt")
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model
from nexttransformer.models.backbone import norm
from nexttransformer.training.optimizer import ExperimentOptimizer


class Wave3StateModule(nn.Module):
    def __init__(self, idea: str, width: int, loops_train: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.width = width
        self.loops_train = loops_train

        if idea == "loopnorm_ls":
            # Prior-art discrete [R, d] per-loop RMSNorm scale + LayerScale (clamped at R-1 for OOD)
            self.norm_w = nn.Parameter(torch.ones(loops_train, width, device=device, dtype=torch.float32))
            self.ls_w = nn.Parameter(torch.full((loops_train, width), 0.5, device=device, dtype=torch.float32))

        if idea in ("explorer_highway_r", "explorer_highway_R", "polyak_geodesic"):
            self.g_anchor = nn.Parameter(torch.full((width,), 0.25, device=device, dtype=torch.float32))
        if idea in ("explorer_highway_r", "explorer_highway_R"):
            init_sh = 1.0 if idea == "explorer_highway_r" else 4.0
            self.s_h = nn.Parameter(torch.tensor(init_sh, device=device, dtype=torch.float32))

        if idea == "prelude_true_ode":
            # Scales BOTH delta and prelude by 1/R so OOD R=16 converges to the exact same ODE limit!
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_ode = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))

        if idea == "euler_attnpool":
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if idea in ("euler_attnpool", "symplectic_ode_pool"):
            self.ap_wq = nn.Linear(2 * width, 16, bias=False, device=device, dtype=torch.bfloat16)
            self.ap_wk = nn.Linear(width, 16, bias=False, device=device, dtype=torch.bfloat16)
            nn.init.zeros_(self.ap_wq.weight)
            nn.init.normal_(self.ap_wk.weight, std=0.05)
            self.ap_last_bias = nn.Parameter(torch.tensor(2.0, device=device, dtype=torch.float32))

        if idea in ("symplectic_pad", "symplectic_ode_pool", "symplectic_polyak"):
            self.mu_logit = nn.Parameter(torch.full((width,), 0.85, device=device, dtype=torch.float32))
            self.gr_down = nn.Linear(2 * width, 16, bias=False, device=device, dtype=torch.bfloat16)
            self.gr_up = nn.Linear(16, width, bias=True, device=device, dtype=torch.bfloat16)
            nn.init.normal_(self.gr_down.weight, std=0.02)
            nn.init.zeros_(self.gr_up.weight)
            nn.init.constant_(self.gr_up.bias, -2.0)
            self.gw_down = nn.Linear(2 * width, 16, bias=False, device=device, dtype=torch.bfloat16)
            self.gw_up = nn.Linear(16, width, bias=True, device=device, dtype=torch.bfloat16)
            nn.init.normal_(self.gw_down.weight, std=0.02)
            nn.init.zeros_(self.gw_up.weight)
            nn.init.constant_(self.gw_up.bias, 0.0)
            if idea == "symplectic_ode_pool":
                self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))


def attach_wave3_executor(model, idea: str, loops_train: int, device: torch.device):
    sm = Wave3StateModule(idea, model.config.width, loops_train, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}
    if idea == "official_base":
        return sm

    use_attnpool = idea in ("euler_attnpool", "symplectic_ode_pool")

    def _run_core(self, h, x0, idx, loop_r, total_depth):
        c = self.config
        for j in range(c.core_layers):
            phys = c.pre_layers + j
            virt = c.pre_layers + loop_r * c.core_layers + j
            h = self._block(h, x0, idx, phys, virt, total_depth)
        return h

    def _apply_attnpool(states):
        H = torch.stack(states, dim=-2)
        q = sm.ap_wq(torch.cat((norm(states[0]), norm(states[-1])), dim=-1)).float()
        K = sm.ap_wk(norm(H)).float()
        scores = torch.einsum("btd,btkd->btk", q, K) * 0.25
        bias = torch.zeros(len(states), device=H.device, dtype=torch.float32)
        bias[-1] = sm.ap_last_bias
        w = F.softmax(scores + bias, dim=-1).to(H.dtype)
        return torch.einsum("btk,btkd->btd", w, H)

    def _custom_ordinary_hidden(self, idx):
        c = self.config
        R = getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops
        total = c.pre_layers + R * c.core_layers + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_loop = R // 2
        rec_diag = (not self.training) and getattr(self, "_collect_loop_diag", False)
        norms, raw_cos_list = [], []
        prev_raw_delta = None

        states = [prelude] if use_attnpool else None
        z = prelude
        h_acc = prelude
        bar_z = torch.zeros_like(prelude)
        s_reg = torch.zeros_like(prelude) if idea in ("symplectic_pad", "symplectic_ode_pool", "symplectic_polyak") else None

        for r in range(R):
            if idea == "loopnorm_ls":
                idx_l = min(r, sm.loops_train - 1)
                h_in = norm(x) * sm.norm_w[idx_l].to(x.dtype)
                delta = _run_core(self, h_in, x0, idx, r, total) - h_in
                x = x + sm.ls_w[idx_l].to(x.dtype) * delta
            elif idea in ("explorer_highway_r", "explorer_highway_R"):
                delta = _run_core(self, z, x0, idx, r, total) - z
                z = norm(z + delta + sm.g_anchor.to(z.dtype) * prelude)
                scale = (sm.s_h / math.sqrt(float(r + 1))) if idea == "explorer_highway_r" else (sm.s_h / float(R))
                h_acc = h_acc + scale.to(h_acc.dtype) * delta
                x = h_acc
            elif idea == "polyak_geodesic":
                delta = _run_core(self, z, x0, idx, r, total) - z
                z = norm(z + delta + sm.g_anchor.to(z.dtype) * prelude)
                bar_z = bar_z + float(r + 1) * z
                x = z
            elif idea == "prelude_true_ode":
                delta = _run_core(self, x, x0, idx, r, total) - x
                inv_r = 1.0 / float(R)
                x = norm(x + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_ode.to(x.dtype) * prelude))
            elif idea == "euler_attnpool":
                delta = _run_core(self, x, x0, idx, r, total) - x
                x = x + (sm.s_ode / float(R)).to(x.dtype) * delta
                states.append(x)
            elif idea in ("symplectic_pad", "symplectic_ode_pool", "symplectic_polyak"):
                gr = torch.sigmoid(sm.gr_up(F.silu(sm.gr_down(torch.cat((norm(x), norm(s_reg)), dim=-1)))).float()).to(x.dtype)
                h_in = x + gr * s_reg
                delta = _run_core(self, h_in, x0, idx, r, total) - h_in
                mu = torch.sigmoid(sm.mu_logit).to(x.dtype)
                s_reg = mu * s_reg + (1.0 - mu) * delta
                gw = torch.sigmoid(sm.gw_up(F.silu(sm.gw_down(torch.cat((norm(x), norm(delta)), dim=-1)))).float()).to(x.dtype)
                if idea == "symplectic_ode_pool":
                    step_w = (sm.s_ode / float(R)).to(x.dtype) * gw
                    x = x + step_w * delta
                    states.append(x)
                else:
                    # Prefix-causal: scale by 1/sqrt(r+1) for symplectic_polyak or 4/R for symplectic_pad
                    if idea == "symplectic_polyak":
                        x = x + (2.0 * gw / math.sqrt(float(r + 1))) * delta
                        bar_z = bar_z + float(r + 1) * x
                    else:
                        x = x + (2.0 * gw / max(float(R) / 2.0, 1.0)) * delta

            if r == mid_loop:
                cached = x
            if rec_diag:
                d_raw = delta.detach().float()
                norms.append(float(x.float().norm(dim=-1).mean().item()))
                if prev_raw_delta is not None:
                    raw_cos_list.append(float(F.cosine_similarity(d_raw, prev_raw_delta, dim=-1).mean().item()))
                prev_raw_delta = d_raw

        if idea == "polyak_geodesic":
            x = norm(bar_z)
        elif idea == "symplectic_polyak":
            w_sum = 0.5 * float(R) * float(R + 1)
            x = bar_z / w_sum
        elif use_attnpool:
            x = _apply_attnpool(states)

        if rec_diag:
            self._loop_diag = dict(norms=[round(v, 2) for v in norms], raw_delta_cos=[round(v, 3) for v in raw_cos_list])

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + R * c.core_layers + j
            x = self._block(x, x0, idx, phys, virt, total)
        return x, cached

    model._ordinary_hidden = types.MethodType(_custom_ordinary_hidden, model)
    return sm


def build_official_opt(model, batch_tokens: int):
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
    # Strictly locked at Wave-1 lr = 0.003 * bscale (Carbon-Auditor #91 hygiene rule!)
    if hasattr(model, "loop_state_mod"):
        adam(list(model.loop_state_mod.parameters()), 0.003 * bscale, (0.9, 0.95), 0.0)

    allocated = {id(p) for g in groups for p in g["params"]}
    mat_groups, other = defaultdict(list), []
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:
            continue
        if p.ndim == 2:
            mat_groups[tuple(p.shape)].append(p)
        else:
            other.append(p)
    adam(other, adamw_lr, (0.9, 0.95), 0.0)
    for sh in sorted(mat_groups):
        groups.append(dict(kind="muon", params=mat_groups[sh], lr=muon_lr, initial_lr=muon_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9,
                           weight_decay=wd, initial_weight_decay=wd))
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
        "official_base", "loopnorm_ls",
        "explorer_highway_r", "explorer_highway_R", "polyak_geodesic",
        "prelude_true_ode", "euler_attnpool",
        "symplectic_pad", "symplectic_ode_pool", "symplectic_polyak",
    ])
    ap.add_argument("--method", default="fixed_recursive", choices=["fixed_recursive", "huginn", "deeploop", "ouro"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--huginn-bptt", type=int, default=4)
    ap.add_argument("--huginn-sigma", type=float, default=0.5)
    ap.add_argument("--horizon-jitter", action="store_true", help="Sample R_step ~ Uniform{R-2..R+2} (mean R, 1.0x FLOPs)")
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
    is_ouro = (args.method == "ouro")
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
    sm = attach_wave3_executor(model, args.idea, args.loops, device=device)
    extra_params = sum(p.numel() for p in sm.parameters())
    is_off_loop = (args.idea == "official_base") and (args.method in ("huginn", "ouro"))

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                 args.loops, max_rows=512, is_official_loop=is_off_loop)
    print(f"[{args.tag}|{args.idea}|{args.method}|R={args.loops}|jit={int(args.horizon_jitter)}|+p={extra_params} ({100*extra_params/base_params:.4f}%)] "
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

    # Extended OOD test-time sweep across R in {1, 2, 4, 8, 12, 16, 32} (Gate A3.3!)
    ood_losses = {}
    eval_depths = (1,) if (args.loops == 1) else ((1, 2, 4, 8) if is_ouro else (1, 2, 4, 8, 12, 16, 32))
    for r_ev in eval_depths:
        if r_ev == args.loops:
            ood_losses[str(r_ev)] = round(history[-1]["val_loss"], 5)
        elif args.idea == "official_base" and args.method in ("fixed_recursive", "deeploop"):
            # Official non-overridden fixed_recursive/deeploop cannot override loops cleanly via _eval_loops_override
            continue
        else:
            ev_mb = 8 if r_ev >= 32 else (16 if r_ev >= 12 else min(args.micro_batch, 32))
            _, vl_r, _ = eval_at_depth(model, val_x, val_y, byte_table, device, ev_mb, r_ev,
                                       max_rows=1024, is_official_loop=is_off_loop)
            ood_losses[str(r_ev)] = round(vl_r, 5)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    out = dict(
        tag=args.tag, idea=args.idea, method=args.method, loops=args.loops,
        freeze_x0=args.freeze_x0, horizon_jitter=args.horizon_jitter, steps=args.steps,
        extra_params=extra_params, extra_param_pct=round(100.0 * extra_params / base_params, 5),
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], ood_losses=ood_losses,
        loop_diag=diag, tok_per_sec=batch_tokens / avg_dt, total_wall_s=time.time() - t0_all,
        history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    rcos = diag.get("raw_delta_cos", [])
    cos_str = f"{sum(rcos)/max(len(rcos),1):.3f}" if rcos else "n/a"
    print(f"[{args.tag}] DONE val={out['final_val_loss']:.5f} ood={ood_losses} "
          f"raw_cos={cos_str} {out['tok_per_sec']/1e3:.1f}k tok/s {out['total_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
