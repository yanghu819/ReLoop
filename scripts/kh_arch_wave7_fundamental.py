#!/usr/bin/env python3
"""Direction-A Arch Wave 7: TRULY FUNDAMENTAL Looped-LM Breakthroughs (Breaking the Identical-Operator Ceiling!).

Goes far beyond scalar RMSNorm/retraction tweaks to attack the core mathematical ceiling of weight-tied loops:
  Why does untied 2/16/2 (3.3277) still beat weight-tied 2/4/2 x 8 (3.3468)?
  Because applying the EXACT SAME operator F_W at every loop traps the model in a 1-parameter semigroup!

Tests 5 genuinely fundamental, Bitter-Lesson-native (0 lookup tables, continuous arbitrary R!) mechanisms:
  1. depth_adaln_zero   : Continuous-Time Loop-AdaLN-Zero (DiT for recurrent depth!): sinusoidal tau=(r+0.5)/R
                          modulates (scale, shift, gate) of EVERY core block's Attn & MLP -> dynamically selects
                          different sparse ReLU^2 sub-circuits at each depth! (~0.3% params, 0 lookup tables!)
  2. state_adaln_zero   : 100% Prefix-Causal State-Conditioned AdaLN-Zero: (scale, shift, gate) predicted from
                          current token state norm(x) (ZERO dependence on R! Pure anytime dynamic circuit routing!)
  3. func_weight_lora   : Continuous-Depth Functional Weight Trajectory W(tau) = W_0 + B diag(phi(tau)) A
                          on all core MLP/Attn projections (rank-32 shared basis modulated by continuous tau=(r+0.5)/R)!
                          Breaks the identical-operator semigroup with ~1.5% params & smooth R=16/32 interpolation!
  4. depth_adaln_anchor : Combines Continuous Loop-AdaLN-Zero + 2-line Spherical Prelude-ODE (tests if orthogonal!)
  5. func_lora_anchor   : Combines Continuous Functional Weight Trajectory W(tau) + 2-line Spherical Prelude-ODE!
  6. depth_adaln_r1_20L : Gate A1.3 Untied 2/16/2 R=1 DiD control for depth_adaln_zero!

Strictly enforces Carbon-Auditor hygiene:
  - 100% official 5-step MuonAdamW + new arch params AdamW lr = 0.003 * bscale
  - --freeze-x0 clean baseline
  - Zero-slice-bias ood_2048 across R_eval in {1, 2, 4, 8, 12, 16, 32}
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


def fourier_depth_embed(tau: float, dim: int, device: torch.device) -> torch.Tensor:
    """Continuous sinusoidal Fourier encoding of normalized depth tau in (0, 1) — generalizes to ANY R!"""
    half = dim // 2
    freqs = math.pi * (2.0 ** torch.linspace(0.0, 4.0, half, device=device, dtype=torch.float32))
    ang = tau * freqs
    return torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)  # [dim]


class DepthAdaLNMod(nn.Module):
    """Continuous-depth (or state-conditioned) AdaLN-Zero across all core layers (0 lookup tables!)."""
    def __init__(self, core_layers: int, width: int, mode: str, with_anchor: bool, device: torch.device):
        super().__init__()
        self.core_layers = core_layers
        self.width = width
        self.mode = mode  # "depth" or "state"
        self.with_anchor = with_anchor
        # Per core layer: 6 modulations (attn: shift, scale, gate; mlp: shift, scale, gate)
        cond_dim = 64
        if mode == "depth":
            self.emb_mlp = nn.Sequential(
                nn.Linear(cond_dim, cond_dim, device=device, dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim, device=device, dtype=torch.float32),
                nn.SiLU(),
            )
            self.to_mod = nn.Linear(cond_dim, core_layers * 6 * width, bias=True, device=device, dtype=torch.float32)
            nn.init.zeros_(self.to_mod.weight)
            nn.init.zeros_(self.to_mod.bias)
        else:
            # Token-state conditioned low-rank AdaLN-Zero (100% prefix-causal, R-independent!)
            self.state_down = nn.Linear(width, 32, bias=False, device=device, dtype=torch.bfloat16)
            self.state_up = nn.Linear(32, core_layers * 6 * width, bias=True, device=device, dtype=torch.bfloat16)
            nn.init.normal_(self.state_down.weight, std=0.02)
            nn.init.zeros_(self.state_up.weight)
            nn.init.zeros_(self.state_up.bias)
        if with_anchor:
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))


class FuncWeightLinear(nn.Module):
    """Continuous-Depth Functional Weight Linear: W(tau) x = W_0 x + B (phi(tau) * (A x)).

    Zero [max_loops] lookup tables! Rank-r shared basis modulated by continuous Fourier depth tau in (0, 1).
    At initialization B = 0 => exact Step-0 parity with standard ReferenceLM!
    """
    def __init__(self, base_linear: nn.Module, rank: int, device: torch.device):
        super().__init__()
        self.base = base_linear
        out_f, in_f = base_linear.weight.shape
        self.rank = rank
        self.A = nn.Parameter(torch.randn(rank, in_f, device=device, dtype=torch.bfloat16) * (in_f ** -0.5))
        self.B = nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=torch.bfloat16))
        self._cur_phi = None  # [rank] set per loop by executor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self._cur_phi is not None:
            # x: [B, T, in_f] -> [B, T, rank] -> modulate by phi(tau) -> [B, T, out_f]
            h = F.linear(x, self.A.to(x.dtype))
            h = h * self._cur_phi.to(x.dtype)
            out = out + F.linear(h, self.B.to(x.dtype))
        return out


class FuncWeightMod(nn.Module):
    """Produces continuous depth basis coefficients phi(tau) for all wrapped core linear layers."""
    def __init__(self, n_wrapped: int, rank: int, with_anchor: bool, width: int, device: torch.device):
        super().__init__()
        self.n_wrapped = n_wrapped
        self.rank = rank
        self.with_anchor = with_anchor
        cond_dim = 64
        self.phi_net = nn.Sequential(
            nn.Linear(cond_dim, cond_dim, device=device, dtype=torch.float32),
            nn.SiLU(),
            nn.Linear(cond_dim, n_wrapped * rank, device=device, dtype=torch.float32),
        )
        # Initialize last layer gently so phi(tau) starts non-zero and diverse across tau!
        nn.init.normal_(self.phi_net[-1].weight, std=0.25)
        nn.init.ones_(self.phi_net[-1].bias)
        if with_anchor:
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))


def attach_wave7_executor(model, idea: str, device: torch.device):
    c = model.config
    is_r1_did = (idea == "depth_adaln_r1_20L")
    use_adaln = idea in ("depth_adaln_zero", "state_adaln_zero", "depth_adaln_anchor", "depth_adaln_r1_20L")
    use_func_w = idea in ("func_weight_lora", "func_lora_anchor")
    with_anchor = idea in ("depth_adaln_anchor", "func_lora_anchor")

    wrapped_linears = []
    if use_adaln:
        n_core = 4 if is_r1_did else c.core_layers
        sm = DepthAdaLNMod(n_core, c.width, "state" if idea == "state_adaln_zero" else "depth",
                           with_anchor=with_anchor, device=device)
    elif use_func_w:
        rank = 32
        for j in range(c.core_layers):
            blk = model.transformer.h[c.pre_layers + j]
            # Wrap MLP c_fc & c_proj + Attn c_q, c_k, c_v, c_proj with continuous functional weights W(tau)!
            for parent, attr in (
                (blk.attn, "c_q"), (blk.attn, "c_k"), (blk.attn, "c_v"), (blk.attn, "c_proj"),
                (blk.mlp, "c_fc"), (blk.mlp, "c_proj"),
            ):
                fw = FuncWeightLinear(getattr(parent, attr), rank=rank, device=device)
                setattr(parent, attr, fw)
                wrapped_linears.append(fw)
        sm = FuncWeightMod(len(wrapped_linears), rank=rank, with_anchor=with_anchor, width=c.width, device=device)
    else:
        sm = nn.Module()

    model.loop_state_mod = sm
    model._loop_diag = {}

    def _run_adaln_block(self, x, x0, idx, phys, virt, total, mod6):
        """Runs one Transformer block with AdaLN-Zero (shift_a, scale_a, gate_a, shift_m, scale_m, gate_m)."""
        sa, sca, ga, sm_, scm, gm = mod6.chunk(6, dim=-1)
        blk = self.transformer.h[phys]
        ve = self.value_embeds[str(phys)](idx).to(x.dtype) if str(phys) in self.value_embeds else None
        window = self._window(virt, total)
        h_a = norm(x) * (1.0 + sca.to(x.dtype)) + sa.to(x.dtype)
        x = x + (1.0 + ga.to(x.dtype)) * blk.attn(h_a, ve, (self.cos[:, :idx.shape[1]], self.sin[:, :idx.shape[1]]), window, None)
        h_m = norm(x) * (1.0 + scm.to(x.dtype)) + sm_.to(x.dtype)
        x = x + (1.0 + gm.to(x.dtype)) * blk.mlp(h_m)
        return x

    def _custom_ordinary_hidden(self, idx):
        R_eff = 4 if is_r1_did else (getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops)
        total = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        rec_diag = (not self.training) and getattr(self, "_collect_loop_diag", False)
        raw_cos_list = []
        prev_raw_delta = None
        inv_r = 1.0 / float(R_eff)
        tbptt = getattr(self, "_tbptt_loops", 0)

        for r in range(R_eff):
            no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
            if self.training and tbptt > 0 and r == R_eff - tbptt:
                x = x.detach()
            with torch.no_grad() if no_g else nullcontext():
                x_in = x
                tau = (float(r) + 0.5) * inv_r

                if use_adaln:
                    if sm.mode == "depth":
                        emb = fourier_depth_embed(tau, 64, x.device)
                        mods = sm.to_mod(sm.emb_mlp(emb)).view(sm.core_layers, 1, 1, 6 * c.width)
                    else:
                        mods = sm.state_up(F.silu(sm.state_down(norm(x)))).view(*x.shape[:-1], sm.core_layers, 6 * c.width)
                    h = x_in
                    for j in range(sm.core_layers):
                        phys = c.pre_layers + (r * 4 + j if is_r1_did else j)
                        virt = phys if is_r1_did else (c.pre_layers + r * c.core_layers + j)
                        mod6 = mods[j] if sm.mode == "depth" else mods[..., j, :]
                        h = _run_adaln_block(self, h, x0, idx, phys, virt, total, mod6)
                    delta = h - x_in

                elif use_func_w:
                    emb = fourier_depth_embed(tau, 64, x.device)
                    phis = sm.phi_net(emb).view(sm.n_wrapped, sm.rank)
                    for idx_w, fw in enumerate(wrapped_linears):
                        fw._cur_phi = phis[idx_w]
                    try:
                        h = x_in
                        for j in range(c.core_layers):
                            phys = c.pre_layers + j
                            virt = c.pre_layers + r * c.core_layers + j
                            h = self._block(h, x0, idx, phys, virt, total)
                    finally:
                        for fw in wrapped_linears:
                            fw._cur_phi = None
                    delta = h - x_in

                if with_anchor:
                    x = norm(x_in + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))
                else:
                    x = x_in + delta

                if r == mid_g:
                    cached = x
                if rec_diag:
                    d_raw = delta.detach().float()
                    if prev_raw_delta is not None:
                        raw_cos_list.append(float(F.cosine_similarity(d_raw, prev_raw_delta, dim=-1).mean().item()))
                    prev_raw_delta = d_raw

        if rec_diag:
            self._loop_diag = dict(raw_delta_cos=[round(v, 3) for v in raw_cos_list])

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + j
            x = self._block(x, x0, idx, phys, virt, total)
        return x, cached

    model._ordinary_hidden = types.MethodType(_custom_ordinary_hidden, model)
    return sm, wrapped_linears


def build_official_opt(model, batch_tokens: int, extra_params_list: list):
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
    if extra_params_list:
        adam(extra_params_list, 0.003 * bscale, (0.9, 0.95), 0.0)

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
                  loops_eval: int, max_rows: int = 4096, collect_diag: bool = False):
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
        model._eval_loops_override = None
        model._collect_loop_diag = False
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1.0)), total_nats / max(total_toks, 1.0), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=[
        "depth_adaln_zero", "state_adaln_zero", "func_weight_lora",
        "depth_adaln_anchor", "func_lora_anchor", "depth_adaln_r1_20L",
    ])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tbptt-loops", type=int, default=0)
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

    base_params = sum(p.numel() for p in model.parameters())
    sm, wrapped = attach_wave7_executor(model, args.idea, device=device)
    model._tbptt_loops = args.tbptt_loops
    extra_params_list = list(sm.parameters()) + [p for fw in wrapped for p in (fw.A, fw.B)]
    extra_params = sum(p.numel() for p in extra_params_list)

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens, extra_params_list)

    n_rows = train_x.shape[0]
    bpb0, vl0, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|R={args.loops}|jit={int(args.horizon_jitter)}|+p={extra_params} ({100*extra_params/base_params:.3f}%)] "
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
                                         args.loops, max_rows=rows, collect_diag=(step + 1 == args.steps))
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
            _, vl_r, _ = eval_at_depth(model, val_x, val_y, byte_table, device, ev_mb, r_ev, max_rows=2048)
            ood_2048[str(r_ev)] = round(vl_r, 5)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    ood_diff = round(ood_2048["16"] - ood_2048["8"], 5) if ("16" in ood_2048 and "8" in ood_2048) else None
    out = dict(
        tag=args.tag, idea=args.idea, loops=args.loops, horizon_jitter=args.horizon_jitter,
        extra_params=extra_params, extra_param_pct=round(100.0 * extra_params / base_params, 4),
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
          f"raw_cos={cos_str} {out['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
