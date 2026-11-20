#!/usr/bin/env python3
"""Direction-A Arch Wave 1: Kaiming-He Simple, Bitter-Lesson Native Looped-LM State Geometries.

Zero optimizer gimmicks (100% standard official 5-step MuonAdamW, --freeze-x0 clean baseline).
Every mechanism uses <= 8 lines, 0% to 0.3% extra params, ZERO per-loop lookup tables,
and generalizes seamlessly to arbitrary unseen test depths R_eval in {1, 2, 4, 8, 16}!

Mechanisms tested:
  1. arch_base             : Exact clean --freeze-x0 fixed_recursive baseline (+ collapse telemetry)
  2. arch_inv_sqrt_r       : 1-line Neural-ODE residual step scaling: x += (F(x) - x) / sqrt(R)
  3. arch_interloop_norm   : 1-line hypersphere projection between loops: x = norm(x)
  4. arch_prelude_anchor   : 2-line clean prelude re-injection: x = norm(x + gate * prelude)
  5. arch_second_order_hb  : 3-line Heavy-Ball 2nd-order velocity state (v_{t+1} = b*v_t + a*d_t)
  6. arch_loop_mhc_4       : 6-line Dynamic 4-stream Doubly-Stochastic Loop-mHC working-memory tape
  7. arch_loop_mhc_4_shift : Same Loop-mHC-4 initialized near cyclic shift + identity
  8. arch_loop_attnres     : 5-line Token-adaptive Cross-Loop Depth Attention over past loop states
  9. arch_depth_rope       : 0-param Orthogonal Conjugation Depth-RoPE: x += R(-t*th)(F(R(t*th)x)-R(t*th)x)
 10. arch_depth_rope_learn : Same Depth-RoPE with d/2 learned frequencies (0.0002% params)
 11. arch_glu_highway      : 3-line State-conditioned Highway/GRU loop gate: x = (1-g)*x + g*F(x)
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/nt")
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model
from nexttransformer.models.backbone import norm
from nexttransformer.training.optimizer import ExperimentOptimizer


def apply_rotary_pair(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact orthogonal 2x2 Givens rotation along the channel dimension (norm-preserving!)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:2 * d2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat((y1, y2), dim=-1)


def sinkhorn_doubly_stochastic(logits: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Birkhoff polytope projection via log-domain Sinkhorn (provably non-expansive, 0 NaNs)."""
    M = logits.float()
    for _ in range(iters):
        M = M - torch.logsumexp(M, dim=-1, keepdim=True)
        M = M - torch.logsumexp(M, dim=-2, keepdim=True)
    return M.exp()


class LoopStateModule(nn.Module):
    """Minimal, depth-agnostic loop-state dynamics module (0 to 0.3% params, generalizes to any R)."""
    def __init__(self, idea: str, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.width = width
        if idea == "arch_prelude_anchor":
            self.gate = nn.Parameter(torch.full((width,), 0.25, device=device, dtype=torch.float32))
        elif idea == "arch_second_order_hb":
            # atanh(0.3) ~ 0.31, alpha init 1.0
            self.beta_raw = nn.Parameter(torch.full((width,), 0.31, device=device, dtype=torch.float32))
            self.alpha = nn.Parameter(torch.ones(width, device=device, dtype=torch.float32))
        elif idea in ("arch_loop_mhc_4", "arch_loop_mhc_4_shift"):
            self.n_streams = 4
            # Tiny dynamic routing head: norm(mean_stream) -> [H_res (16), w_read (4), w_write (4)]
            self.dyn_proj = nn.Linear(width, 24, bias=False, device=device, dtype=torch.bfloat16)
            nn.init.zeros_(self.dyn_proj.weight)
            base_res = torch.eye(4, device=device, dtype=torch.float32) * 2.5
            if idea == "arch_loop_mhc_4_shift":
                shift = torch.roll(torch.eye(4, device=device, dtype=torch.float32), shifts=1, dims=0)
                base_res = torch.eye(4, device=device, dtype=torch.float32) * 1.5 + shift * 1.0
            self.res_bias = nn.Parameter(base_res)
            self.read_bias = nn.Parameter(torch.zeros(4, device=device, dtype=torch.float32))
            self.write_bias = nn.Parameter(torch.zeros(4, device=device, dtype=torch.float32))
            self.write_scale = nn.Parameter(torch.full((4,), 4.0, device=device, dtype=torch.float32))
        elif idea == "arch_loop_attnres":
            # Low-rank token-adaptive depth query: d -> 64 -> d (0.05% params, zero per-loop params!)
            self.q_down = nn.Linear(width, 64, bias=False, device=device, dtype=torch.bfloat16)
            self.q_up = nn.Linear(64, width, bias=False, device=device, dtype=torch.bfloat16)
            nn.init.normal_(self.q_down.weight, std=0.02)
            nn.init.zeros_(self.q_up.weight)
            self.q_static = nn.Parameter(torch.zeros(width, device=device, dtype=torch.float32))
        elif idea in ("arch_depth_rope", "arch_depth_rope_learn"):
            # Log-spaced frequencies in [0.02, pi/4]: smooth continuous phase across arbitrary R!
            d2 = width // 2
            freqs = (math.pi / 4.0) * (100.0 ** (-torch.arange(d2, device=device, dtype=torch.float32) / max(d2 - 1, 1)))
            if idea == "arch_depth_rope_learn":
                self.theta = nn.Parameter(freqs)
            else:
                self.register_buffer("theta", freqs)
        elif idea == "arch_glu_highway":
            # 1-layer gate initialized around sigmoid(-1.0) = 0.27 -> smooth residual highway
            self.gate_proj = nn.Linear(width, width, bias=True, device=device, dtype=torch.bfloat16)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -1.0)


def attach_arch_executor(model, idea: str, device: torch.device):
    """Cleanly wraps ReferenceLM._ordinary_hidden with our minimal loop-state executor."""
    state_mod = LoopStateModule(idea, model.config.width, device=device)
    model.loop_state_mod = state_mod
    model._loop_diag = {}

    def _run_core(self, h, x0, idx, loop_r, total_depth):
        c = self.config
        for j in range(c.core_layers):
            phys = c.pre_layers + j
            virt = c.pre_layers + loop_r * c.core_layers + j
            h = self._block(h, x0, idx, phys, virt, total_depth)
        return h

    def _custom_ordinary_hidden(self, idx):
        c = self.config
        R = getattr(self, "_eval_loops_override", None) or c.loops
        total = c.pre_layers + R * c.core_layers + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_loop = R // 2

        sm = self.loop_state_mod
        rec_diag = (not self.training) and getattr(self, "_collect_loop_diag", False)
        norms, cos_list = [], []
        prev_delta = None

        if idea in ("arch_loop_mhc_4", "arch_loop_mhc_4_shift"):
            # S: [B, T, 4, D] initialized to identical prelude across all 4 streams
            S = prelude.unsqueeze(-2).expand(-1, -1, 4, -1).clone()
            for r in range(R):
                s_mean = norm(S.mean(dim=-2))  # [B, T, D]
                dyn = sm.dyn_proj(s_mean).float()
                res_logits = dyn[..., :16].view(*s_mean.shape[:-1], 4, 4) + sm.res_bias
                H_res = sinkhorn_doubly_stochastic(res_logits, iters=5).to(S.dtype)  # [B, T, 4, 4]
                w_read = F.softmax(dyn[..., 16:20] + sm.read_bias, dim=-1).to(S.dtype)  # [B, T, 4]
                w_write = (F.softmax(dyn[..., 20:24] + sm.write_bias, dim=-1) * sm.write_scale).to(S.dtype)
                h_in = torch.einsum("btn,btnd->btd", w_read, S)
                h_out = _run_core(self, h_in, x0, idx, r, total)
                delta = h_out - h_in
                S_mixed = torch.einsum("btij,btjd->btid", H_res, S)
                S = S_mixed + w_write.unsqueeze(-1) * delta.unsqueeze(-2)
                x = S.mean(dim=-2)
                if r == mid_loop:
                    cached = x
                if rec_diag:
                    norms.append(float(x.float().norm(dim=-1).mean().item()))
                    if prev_delta is not None:
                        cos_list.append(float(F.cosine_similarity(delta.float(), prev_delta, dim=-1).mean().item()))
                    prev_delta = delta.detach().float()
        else:
            vel = torch.zeros_like(x) if idea == "arch_second_order_hb" else None
            hist = [prelude] if idea == "arch_loop_attnres" else None
            hist_keys = [norm(prelude)] if idea == "arch_loop_attnres" else None

            for r in range(R):
                x_prev = x
                if idea == "arch_base":
                    x = _run_core(self, x, x0, idx, r, total)
                elif idea == "arch_inv_sqrt_r":
                    h_out = _run_core(self, x, x0, idx, r, total)
                    x = x + (h_out - x) * (float(R) ** -0.5)
                elif idea == "arch_interloop_norm":
                    x = norm(_run_core(self, x, x0, idx, r, total))
                elif idea == "arch_prelude_anchor":
                    h_out = _run_core(self, x, x0, idx, r, total)
                    x = norm(h_out + sm.gate.to(x.dtype) * prelude)
                elif idea == "arch_second_order_hb":
                    h_out = _run_core(self, x, x0, idx, r, total)
                    delta = h_out - x
                    beta = torch.tanh(sm.beta_raw).to(x.dtype)
                    vel = beta * vel + sm.alpha.to(x.dtype) * delta
                    x = x + vel
                elif idea == "arch_loop_attnres":
                    q = sm.q_static.to(x.dtype) + sm.q_up(F.silu(sm.q_down(norm(x))))
                    # Attend over all previous loop states [prelude, x_1, ..., x_r]
                    K_stack = torch.stack(hist_keys, dim=-2)  # [B, T, r+1, D]
                    V_stack = torch.stack(hist, dim=-2)       # [B, T, r+1, D]
                    scores = torch.einsum("btd,btkd->btk", q.float(), K_stack.float()) * (c.width ** -0.5)
                    attn_w = F.softmax(scores, dim=-1).to(x.dtype)
                    h_in = torch.einsum("btk,btkd->btd", attn_w, V_stack)
                    h_out = _run_core(self, h_in, x0, idx, r, total)
                    x = x + (h_out - h_in)
                    hist.append(x)
                    hist_keys.append(norm(x))
                elif idea in ("arch_depth_rope", "arch_depth_rope_learn"):
                    if r == 0:
                        x = _run_core(self, x, x0, idx, r, total)
                    else:
                        ang = float(r) * sm.theta
                        cos_r = torch.cos(ang).to(x.dtype)
                        sin_r = torch.sin(ang).to(x.dtype)
                        x_rot = apply_rotary_pair(x, cos_r, sin_r)
                        h_out = _run_core(self, x_rot, x0, idx, r, total)
                        delta_rot = h_out - x_rot
                        x = x + apply_rotary_pair(delta_rot, cos_r, -sin_r)
                elif idea == "arch_glu_highway":
                    h_out = _run_core(self, x, x0, idx, r, total)
                    g = torch.sigmoid(sm.gate_proj(norm(x)).float()).to(x.dtype)
                    x = x + g * (h_out - x)

                if r == mid_loop:
                    cached = x
                if rec_diag:
                    delta_step = (x - x_prev).detach().float()
                    norms.append(float(x.float().norm(dim=-1).mean().item()))
                    if prev_delta is not None:
                        cos_list.append(float(F.cosine_similarity(delta_step, prev_delta, dim=-1).mean().item()))
                    prev_delta = delta_step

        if rec_diag:
            self._loop_diag = dict(
                norms=[round(v, 2) for v in norms],
                adj_cos=[round(v, 3) for v in cos_list],
            )

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + R * c.core_layers + j
            x = self._block(x, x0, idx, phys, virt, total)
        return x, cached

    model._ordinary_hidden = types.MethodType(_custom_ordinary_hidden, model)
    return state_mod


def build_official_opt(model, batch_tokens: int):
    """100% official NextTransformer 5-step MuonAdamW (zero optimizer modifications!)."""
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
    # Put all tiny loop_state_mod params cleanly into AdamW at conservative lr (0.003 * bscale)
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
                diag = dict(model._loop_diag)
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
        "arch_base",
        "arch_inv_sqrt_r",
        "arch_interloop_norm",
        "arch_prelude_anchor",
        "arch_second_order_hb",
        "arch_loop_mhc_4",
        "arch_loop_mhc_4_shift",
        "arch_loop_attnres",
        "arch_depth_rope",
        "arch_depth_rope_learn",
        "arch_glu_highway",
    ])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
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
        loops=args.loops, max_loops=args.loops, eval_loops=args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    if args.freeze_x0:
        with torch.no_grad():
            model.x0_lambdas.zero_()
        model.x0_lambdas.requires_grad_(False)

    base_params = sum(p.numel() for p in model.parameters())
    state_mod = attach_arch_executor(model, args.idea, device=device)
    extra_params = sum(p.numel() for p in state_mod.parameters())

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|R={args.loops}|+p={extra_params} ({100*extra_params/base_params:.4f}%)] "
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
            rows = 4096 if (step + 1) == args.steps else 1024
            vb, vl, diag = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                         args.loops, max_rows=rows, collect_diag=(step + 1 == args.steps))
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum,
                                elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f}", flush=True)

    # Multi-depth OOD evaluation across R in {1, 2, 4, 8, 16} (1024 rows fast check, 4096 already done for train R!)
    ood_losses = {}
    for r_ev in (1, 2, 4, 8, 16):
        if r_ev == args.loops:
            ood_losses[str(r_ev)] = round(history[-1]["val_loss"], 5)
        else:
            ev_mb = min(args.micro_batch, 16 if r_ev >= 16 else 32)
            _, vl_r, _ = eval_at_depth(model, val_x, val_y, byte_table, device, ev_mb, r_ev, max_rows=1024)
            ood_losses[str(r_ev)] = round(vl_r, 5)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    out = dict(
        tag=args.tag, idea=args.idea, loops=args.loops, freeze_x0=args.freeze_x0,
        extra_params=extra_params, extra_param_pct=round(100.0 * extra_params / base_params, 5),
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], ood_losses=ood_losses,
        loop_diag=diag, tok_per_sec=batch_tokens / avg_dt, total_wall_s=time.time() - t0_all,
        history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    cos_str = f"{sum(diag.get('adj_cos', []))/max(len(diag.get('adj_cos', [])),1):.3f}" if diag.get("adj_cos") else "n/a"
    print(f"[{args.tag}] DONE val={out['final_val_loss']:.5f} ood={ood_losses} "
          f"mean_cos={cos_str} {out['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
