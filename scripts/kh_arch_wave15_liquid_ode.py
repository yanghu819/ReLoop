#!/usr/bin/env python3
"""Direction-A Arch Wave 15: Liquid Spherical ODE (Delta-Aware Token-Adaptive Continuous Flow on S^{d-1})!

Wave 13/14 Breakthrough:
  - Token-Adaptive Gated Spherical ODE (gated_traj_ode, +76k = 0.037% params, 119.1k tok/s) crushed every other
    low-overhead candidate across BOTH 750s (3.34194, best deterministic R=32 OOD = 3.3857!, R=4 = 3.34969!)
    AND Fresh-2250 (s750 = 3.50457 — beating prelude_ode_det by -0.0077, 2x-param untied by -0.0264,
    1x-param untied 16L by -0.0877, and sitting within 0.0039 of 4x-param untied 20L_iso_ve!).

Wave 15 Upgrades gated_traj_ode into Liquid Spherical ODE:
  1. Delta-Awareness (Residual Convergence Sensing):
     h = SiLU(W_x norm(x_in) + W_d norm(delta) + W_tau P_{1..4}(2*tau - 1))
     [ds, dg] = W_out(h)   (zero-init -> exact bit-for-bit prelude_true_ode at step 0!)
     x_{r+1} = norm(x_in + (1/R) * (s_tok * delta + g_tok * prelude))
     Because the gate observes BOTH current state x_in AND the proposed loop update delta = h_out - x_in,
     easy tokens sense convergence and smoothly brake while multi-hop reasoning tokens accelerate!
  2. Clean Ablations & Mandatory Controls:
     - liquid_ode_r32       : rank-32 delta-aware Liquid Spherical ODE (+100k = 0.048%)
     - liquid_ode_r64       : rank-64 delta-aware Liquid Spherical ODE (+200k = 0.096%)
     - liquid_ode_adaln     : Liquid Spherical ODE + zero-aliasing Legendre AdaLN (+175k = 0.084%)
     - gated_traj_ode       : Wave-13 x-only gated ODE (for 1100-step 4-BPTT iso-wall & R=1 DiD!)
     - liquid_r1_20L        : Mandatory Gate A1.3 Untied 2/16/2 R=1 DiD Control!
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

sys.path.insert(0, "/root/nt")
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model
from nexttransformer.models.backbone import norm
from nexttransformer.training.optimizer import ExperimentOptimizer


def legendre_p1_p4(tau: float, device: torch.device) -> torch.Tensor:
    t = 2.0 * float(tau) - 1.0
    t2 = t * t
    return torch.tensor([
        t,
        0.5 * (3.0 * t2 - 1.0),
        0.5 * t * (5.0 * t2 - 3.0),
        0.125 * (35.0 * t2 * t2 - 30.0 * t2 + 3.0),
    ], device=device, dtype=torch.float32)


class _W15Ctx:
    def __init__(self):
        self.active = False
        self.adaln = None


W15_CTX = _W15Ctx()


class LiquidODEMod(nn.Module):
    def __init__(self, idea: str, core_layers: int, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.width = width
        self.is_r1_did = (idea == "liquid_r1_20L")
        self.use_delta_in = idea in ("liquid_ode_r32", "liquid_ode_r64", "liquid_ode_adaln", "liquid_r1_20L")
        self.use_adaln = (idea == "liquid_ode_adaln")
        self.rank = 64 if idea == "liquid_ode_r64" else 32

        if not self.is_r1_did:
            self.s_base = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_base = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))

        self.gate_x = nn.Linear(width, self.rank, bias=False, device=device, dtype=torch.float32)
        if self.use_delta_in:
            self.gate_d = nn.Linear(width, self.rank, bias=False, device=device, dtype=torch.float32)
        self.gate_leg = nn.Linear(4, self.rank, bias=True, device=device, dtype=torch.float32)
        out_dim = width if self.is_r1_did else 2 * width
        self.gate_out = nn.Linear(self.rank, out_dim, bias=True, device=device, dtype=torch.float32)
        nn.init.zeros_(self.gate_out.weight)
        nn.init.zeros_(self.gate_out.bias)

        if self.use_adaln:
            self.adaln_leg = nn.Parameter(torch.zeros(4, core_layers, 6, width, device=device, dtype=torch.float32))


def attach_wave15_executor(model, idea: str, device: torch.device):
    c = model.config
    is_r1_did = (idea == "liquid_r1_20L")
    n_core = 4 if is_r1_did else c.core_layers
    sm = LiquidODEMod(idea, n_core, c.width, device=device)
    model.loop_state_mod = sm

    if sm.use_adaln:
        for cj in range(c.core_layers):
            blk = model.transformer.h[c.pre_layers + cj]

            def _make_blk_fwd(j):
                def _blk_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    if W15_CTX.active and W15_CTX.adaln is not None:
                        m = W15_CTX.adaln[j]
                        sa, sca, ga, sm_, scm, gm = [m[i].to(x.dtype) for i in range(6)]
                        x = x + (1.0 + ga) * self.attn(norm(x) * (1.0 + sca) + sa, ve, cos_sin, window_size, kv_cache)
                        x = x + (1.0 + gm) * self.mlp(norm(x) * (1.0 + scm) + sm_)
                        return x
                    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
                    x = x + self.mlp(norm(x))
                    return x
                return _blk_fwd
            blk.forward = types.MethodType(_make_blk_fwd(cj), blk)

    def _run_group(self, h, x0, idx, g_idx, total_depth):
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
        R_eff = 4 if is_r1_did else (getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops)
        total = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        W15_CTX.active = False
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        inv_r = 1.0 / float(R_eff)
        tbptt = getattr(self, "_tbptt_loops", 0)
        dt = x.dtype

        W15_CTX.active = True
        try:
            for r in range(R_eff):
                no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
                if self.training and tbptt > 0 and r == R_eff - tbptt:
                    x = x.detach()
                with torch.no_grad() if no_g else nullcontext():
                    tau = (float(r) + 0.5) * inv_r
                    leg = legendre_p1_p4(tau, x.device)
                    if sm.use_adaln:
                        W15_CTX.adaln = torch.einsum("m,mcsw->csw", leg, sm.adaln_leg)

                    x_in = x
                    h_out = _run_group(self, x_in, x0, idx, r, total)
                    delta = h_out - x_in

                    gh_in = sm.gate_x(norm(x_in).float()) + sm.gate_leg(leg)
                    if sm.use_delta_in:
                        gh_in = gh_in + sm.gate_d(norm(delta).float())
                    gh = F.silu(gh_in)

                    if is_r1_did:
                        ds = sm.gate_out(gh)
                        s_tok = (1.0 + 0.5 * torch.tanh(ds)).to(dt)
                        x = x_in + s_tok * delta
                    else:
                        sg = sm.gate_out(gh)
                        ds, dg = sg.chunk(2, dim=-1)
                        s_tok = (sm.s_base * (1.0 + 0.5 * torch.tanh(ds))).to(dt)
                        g_tok = (sm.g_base.view(1, 1, c.width) * (1.0 + 0.5 * torch.tanh(dg))).to(dt)
                        x = norm(x_in + inv_r * (s_tok * delta + g_tok * prelude))

                    if r == mid_g:
                        cached = x
        finally:
            W15_CTX.active = False
            W15_CTX.adaln = None

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + j
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
                  loops_eval: int, max_rows: int = 4096):
    model._eval_loops_override = loops_eval
    model.eval()
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
        model._eval_loops_override = None
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1.0)), total_nats / max(total_toks, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=[
        "liquid_ode_r32", "liquid_ode_r64", "liquid_ode_adaln",
        "gated_traj_ode", "liquid_r1_20L",
    ])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tbptt-loops", type=int, default=0)
    ap.add_argument("--continuum-u412", action="store_true")
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

    cfg = ModelConfig(
        method="fixed_recursive", scale="debug", width=args.width, heads=args.heads,
        vocab_size=32768, sequence_len=seq_len, mlp_width=4 * args.width,
        layers=args.pre_layers + args.core_layers + args.post_layers,
        pre_layers=args.pre_layers, core_layers=args.core_layers, post_layers=args.post_layers,
        loops=args.loops, max_loops=max(args.loops, 2),
        huginn_mean=float(max(args.loops, 2)), huginn_backprop_loops=min(4, max(args.loops, 1)),
        eval_loops=args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    if args.freeze_x0:
        with torch.no_grad():
            model.x0_lambdas.zero_()
        model.x0_lambdas.requires_grad_(False)

    base_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sm = attach_wave15_executor(model, args.idea, device=device)
    model._tbptt_loops = args.tbptt_loops
    extra_params = sum(p.numel() for p in sm.parameters() if p.requires_grad)

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|R={args.loops}|tbptt={args.tbptt_loops}|u412={int(args.continuum_u412)}|"
          f"+p={extra_params} ({100*extra_params/base_params:.4f}%)] step=0 val_loss={vl0:.4f} val_bpb={bpb0:.5f}",
          flush=True)

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

        if args.continuum_u412 and args.loops == 8:
            model._step_loops_override = py_rng.randint(4, 12)
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
            vb, vl = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=rows)
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum,
                                elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f}", flush=True)

    ood_2048 = {}
    if args.loops == 1:
        _, vl_1 = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, 1, max_rows=2048)
        ood_2048["1"] = round(vl_1, 5)
    else:
        for r_ev in (1, 2, 4, 8, 12, 16, 32):
            ev_mb = 8 if r_ev >= 32 else (16 if r_ev >= 12 else min(args.micro_batch, 32))
            _, vl_r = eval_at_depth(model, val_x, val_y, byte_table, device, ev_mb, r_ev, max_rows=2048)
            ood_2048[str(r_ev)] = round(vl_r, 5)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    ood_diff = round(ood_2048["16"] - ood_2048["8"], 5) if ("16" in ood_2048 and "8" in ood_2048) else None
    out = dict(
        tag=args.tag, idea=args.idea, loops=args.loops, tbptt_loops=args.tbptt_loops,
        continuum_u412=args.continuum_u412, steps=args.steps, data_dir=str(args.data_dir),
        extra_params=extra_params, extra_param_pct=round(100.0 * extra_params / base_params, 5),
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], ood_2048=ood_2048, ood_diff_16_8=ood_diff,
        tok_per_sec=batch_tokens / avg_dt, total_wall_s=time.time() - t0_all,
        history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{args.tag}] DONE val={out['final_val_loss']:.5f} ood16-8={ood_diff} ood2k={ood_2048} "
          f"{out['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
