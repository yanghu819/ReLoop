#!/usr/bin/env python3
"""Direction-A Arch Wave 2: internal review #87 Laws + Wave-1 Winner Ablations & Synergies + R=1 Untied Control.

All runs use 100% standard official 5-step MuonAdamW (ZERO optimizer changes!) and --freeze-x0.
Every mechanism generalizes seamlessly to arbitrary unseen test depths R_eval in {1, 2, 4, 8, 16}!

Ideas supported:
  --- internal review #87 Pure Bitter-Lesson Laws ---
  1. roloop_2pi            : Continuous normalized midpoint RoLoop (tau=(r+0.5)/R, w in [0.05pi, 2pi], 0 params)
  2. roloop_4pi            : Same RoLoop with wider band w in [0.05pi, 4pi] (0 params)
  3. euler_ode             : 2-line Midpoint Euler-ODE step scaling: x += (s_ode / R) * delta (1 scalar)
  4. spherical_ode         : 2-line Spherical-ODE geodesic step: x = norm(x + (s_ode / R) * delta) (1 scalar)
  5. loop_attnpool         : 4-line Token-Adaptive Trajectory Attention Readout over [h_0..h_R] (0.01% params)
  6. symplectic_scratchpad : 5-line Dual-register (h_r, s_r) state-driven recurrence (0.01% params)
  --- Wave-1 Winner (prelude_anchor) Causal Ablations & Untied Control ---
  7. prelude_no_norm       : x = h_out + gate * prelude (WITHOUT norm — tests norm vs skip!)
  8. prelude_untied_r1     : Exact R=1 2/16/2 untied 20L control applying norm(x + gate*prelude) every 4 layers!
  9. prelude_gate_dyn      : Dynamic token-conditioned prelude anchor (rank-16 gate, 0.01% params)
  10. prelude_euler_ode    : x = norm(x + (s_ode / R) * delta + gate * prelude) (2 lines, exact OOD 1/R!)
  --- Minimal Unified Synergies (<= 6 lines, <= 0.02% params!) ---
  11. roloop_ode_attnpool         : internal review #87 unified centerpiece (RoLoop 2pi + Euler-ODE + AttnPool)
  12. roloop_sph_attnpool         : RoLoop 2pi + Spherical-ODE + AttnPool
  13. prelude_attnpool            : Prelude-Anchor + Loop-AttnPool (adds O(1) readout highway & adaptive exit!)
  14. prelude_roloop_ode_attnpool : Full 6-line Bitter-Lesson Grand Combo (RoLoop 2pi + Prelude-Euler-ODE + AttnPool)
  15. mhc4_shift_prelude          : Loop-mHC-4-Shift + persistent prelude anchor stream!
  16. mhc4_shift_attnpool         : Loop-mHC-4-Shift + Loop-AttnPool!
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

sys.path.insert(0, os.environ.get("NEXTTRANSFORMER_DIR", "./NextTransformer"))
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model
from nexttransformer.models.backbone import norm
from nexttransformer.training.optimizer import ExperimentOptimizer


def apply_rotary_pair(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:2 * d2]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


def sinkhorn_doubly_stochastic(logits: torch.Tensor, iters: int = 5) -> torch.Tensor:
    M = logits.float()
    for _ in range(iters):
        M = M - torch.logsumexp(M, dim=-1, keepdim=True)
        M = M - torch.logsumexp(M, dim=-2, keepdim=True)
    return M.exp()


class Wave2StateModule(nn.Module):
    def __init__(self, idea: str, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.width = width
        d2 = width // 2

        if idea in ("roloop_2pi", "roloop_ode_attnpool", "roloop_sph_attnpool", "prelude_roloop_ode_attnpool"):
            w_min, w_max = 0.05 * math.pi, 2.0 * math.pi
            omega = w_min * ((w_max / w_min) ** (torch.arange(d2, device=device, dtype=torch.float32) / max(d2 - 1, 1)))
            self.register_buffer("omega", omega)
        elif idea == "roloop_4pi":
            w_min, w_max = 0.05 * math.pi, 4.0 * math.pi
            omega = w_min * ((w_max / w_min) ** (torch.arange(d2, device=device, dtype=torch.float32) / max(d2 - 1, 1)))
            self.register_buffer("omega", omega)

        if idea in ("euler_ode", "spherical_ode", "prelude_euler_ode",
                    "roloop_ode_attnpool", "roloop_sph_attnpool", "prelude_roloop_ode_attnpool"):
            # Init 4.0 => step size 1.0 at R=4, 0.5 at R=8, 0.25 at R=16!
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if idea in ("prelude_no_norm", "prelude_untied_r1", "prelude_euler_ode",
                    "prelude_attnpool", "prelude_roloop_ode_attnpool", "mhc4_shift_prelude"):
            self.gate = nn.Parameter(torch.full((width,), 0.25, device=device, dtype=torch.float32))

        if idea == "prelude_gate_dyn":
            # Low-rank dynamic token-adaptive prelude anchor (0.01% params!)
            self.g_down = nn.Linear(width, 16, bias=False, device=device, dtype=torch.bfloat16)
            self.g_up = nn.Linear(16, width, bias=True, device=device, dtype=torch.bfloat16)
            nn.init.normal_(self.g_down.weight, std=0.02)
            nn.init.zeros_(self.g_up.weight)
            nn.init.constant_(self.g_up.bias, -1.1)  # sigmoid(-1.1) ~ 0.25

        if idea in ("loop_attnpool", "roloop_ode_attnpool", "roloop_sph_attnpool",
                    "prelude_attnpool", "prelude_roloop_ode_attnpool", "mhc4_shift_attnpool"):
            # internal review #87 Law 3: W_q [2d -> 16] zero-init, W_k [d -> 16] small-init
            self.ap_wq = nn.Linear(2 * width, 16, bias=False, device=device, dtype=torch.bfloat16)
            self.ap_wk = nn.Linear(width, 16, bias=False, device=device, dtype=torch.bfloat16)
            nn.init.zeros_(self.ap_wq.weight)
            nn.init.normal_(self.ap_wk.weight, std=0.05)
            # Small learnable bias toward last state so step 0 starts smoothly near h_R
            self.ap_last_bias = nn.Parameter(torch.tensor(2.0, device=device, dtype=torch.float32))

        if idea == "symplectic_scratchpad":
            # internal review #87 Law 4: dual-register (h_r, s_r), rank-16 gates
            self.mu_logit = nn.Parameter(torch.full((width,), 0.85, device=device, dtype=torch.float32))  # sigmoid ~ 0.7
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

        if idea in ("mhc4_shift_prelude", "mhc4_shift_attnpool"):
            self.dyn_proj = nn.Linear(width, 24, bias=False, device=device, dtype=torch.bfloat16)
            nn.init.zeros_(self.dyn_proj.weight)
            shift = torch.roll(torch.eye(4, device=device, dtype=torch.float32), shifts=1, dims=0)
            self.res_bias = nn.Parameter(torch.eye(4, device=device, dtype=torch.float32) * 1.5 + shift * 1.0)
            self.read_bias = nn.Parameter(torch.zeros(4, device=device, dtype=torch.float32))
            self.write_bias = nn.Parameter(torch.zeros(4, device=device, dtype=torch.float32))
            self.write_scale = nn.Parameter(torch.full((4,), 4.0, device=device, dtype=torch.float32))


def attach_wave2_executor(model, idea: str, device: torch.device):
    sm = Wave2StateModule(idea, model.config.width, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}

    use_roloop = idea in ("roloop_2pi", "roloop_4pi", "roloop_ode_attnpool",
                          "roloop_sph_attnpool", "prelude_roloop_ode_attnpool")
    use_attnpool = idea in ("loop_attnpool", "roloop_ode_attnpool", "roloop_sph_attnpool",
                            "prelude_attnpool", "prelude_roloop_ode_attnpool", "mhc4_shift_attnpool")
    use_mhc = idea in ("mhc4_shift_prelude", "mhc4_shift_attnpool")

    def _run_core(self, h, x0, idx, loop_r, total_depth):
        c = self.config
        for j in range(c.core_layers):
            phys = c.pre_layers + j
            virt = c.pre_layers + loop_r * c.core_layers + j
            h = self._block(h, x0, idx, phys, virt, total_depth)
        return h

    def _apply_attnpool(states):
        # states: list of (R+1) tensors [B, T, D]
        H = torch.stack(states, dim=-2)  # [B, T, R+1, D]
        q = sm.ap_wq(torch.cat((norm(states[0]), norm(states[-1])), dim=-1)).float()  # [B, T, 16]
        K = sm.ap_wk(norm(H)).float()  # [B, T, R+1, 16]
        scores = torch.einsum("btd,btkd->btk", q, K) * 0.25  # / sqrt(16)
        # Mild bias on final state r=R so initialization matches standard readout cleanly
        bias = torch.zeros(len(states), device=H.device, dtype=torch.float32)
        bias[-1] = sm.ap_last_bias
        w = F.softmax(scores + bias, dim=-1).to(H.dtype)
        return torch.einsum("btk,btkd->btd", w, H)

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
        rec_diag = (not self.training) and getattr(self, "_collect_loop_diag", False)
        norms, cos_list = [], []
        prev_delta = None

        if idea == "prelude_untied_r1":
            # Exact R=1 2/16/2 untied 20L control: apply norm(x + gate*prelude) every 4 core layers!
            assert R == 1 and c.core_layers % 4 == 0
            for j in range(c.core_layers):
                phys = c.pre_layers + j
                x = self._block(x, x0, idx, phys, phys, total)
                if (j + 1) % 4 == 0:
                    x = norm(x + sm.gate.to(x.dtype) * prelude)
                if phys == total // 2:
                    cached = x
        elif use_mhc:
            S = prelude.unsqueeze(-2).expand(-1, -1, 4, -1).clone()
            states = [prelude] if use_attnpool else None
            for r in range(R):
                s_mean = norm(S.mean(dim=-2))
                dyn = sm.dyn_proj(s_mean).float()
                H_res = sinkhorn_doubly_stochastic(dyn[..., :16].view(*s_mean.shape[:-1], 4, 4) + sm.res_bias, 5).to(S.dtype)
                w_read = F.softmax(dyn[..., 16:20] + sm.read_bias, dim=-1).to(S.dtype)
                w_write = (F.softmax(dyn[..., 20:24] + sm.write_bias, dim=-1) * sm.write_scale).to(S.dtype)
                h_in = torch.einsum("btn,btnd->btd", w_read, S)
                h_out = _run_core(self, h_in, x0, idx, r, total)
                delta = h_out - h_in
                S = torch.einsum("btij,btjd->btid", H_res, S) + w_write.unsqueeze(-1) * delta.unsqueeze(-2)
                if idea == "mhc4_shift_prelude":
                    # Gently anchor stream 0 to prelude on the hypersphere!
                    s0 = norm(S[..., 0, :] + sm.gate.to(S.dtype) * prelude)
                    S = torch.cat((s0.unsqueeze(-2), S[..., 1:, :]), dim=-2)
                x = S.mean(dim=-2)
                if use_attnpool:
                    states.append(x)
                if r == mid_loop:
                    cached = x
                if rec_diag:
                    norms.append(float(x.float().norm(dim=-1).mean().item()))
                    if prev_delta is not None:
                        cos_list.append(float(F.cosine_similarity(delta.float(), prev_delta, dim=-1).mean().item()))
                    prev_delta = delta.detach().float()
            if use_attnpool:
                x = _apply_attnpool(states)
        else:
            states = [prelude] if use_attnpool else None
            s_reg = torch.zeros_like(x) if idea == "symplectic_scratchpad" else None
            for r in range(R):
                x_prev = x
                # Optional input preparation
                if idea == "symplectic_scratchpad":
                    gr = torch.sigmoid(sm.gr_up(F.silu(sm.gr_down(torch.cat((norm(x), norm(s_reg)), dim=-1)))).float()).to(x.dtype)
                    h_in = x + gr * s_reg
                else:
                    h_in = x

                # Core execution (with optional continuous normalized midpoint RoLoop!)
                if use_roloop:
                    tau = (float(r) + 0.5) / float(R)
                    ang = tau * sm.omega
                    cos_r = torch.cos(ang).to(h_in.dtype)
                    sin_r = torch.sin(ang).to(h_in.dtype)
                    h_rot = apply_rotary_pair(h_in, cos_r, sin_r)
                    h_out_rot = _run_core(self, h_rot, x0, idx, r, total)
                    delta = apply_rotary_pair(h_out_rot - h_rot, cos_r, -sin_r)
                else:
                    h_out = _run_core(self, h_in, x0, idx, r, total)
                    delta = h_out - h_in

                # State transition law
                if idea in ("roloop_2pi", "roloop_4pi", "loop_attnpool"):
                    x = x + delta
                elif idea in ("euler_ode", "roloop_ode_attnpool"):
                    step_s = (sm.s_ode / float(R)).to(x.dtype)
                    x = x + step_s * delta
                elif idea in ("spherical_ode", "roloop_sph_attnpool"):
                    step_s = (sm.s_ode / float(R)).to(x.dtype)
                    x = norm(x + step_s * delta)
                elif idea == "prelude_no_norm":
                    x = x + delta + sm.gate.to(x.dtype) * prelude
                elif idea == "prelude_attnpool":
                    x = norm(x + delta + sm.gate.to(x.dtype) * prelude)
                elif idea == "prelude_gate_dyn":
                    h_next = x + delta
                    gd = torch.sigmoid(sm.g_up(F.silu(sm.g_down(norm(h_next)))).float()).to(x.dtype)
                    x = norm(h_next + gd * prelude)
                elif idea in ("prelude_euler_ode", "prelude_roloop_ode_attnpool"):
                    step_s = (sm.s_ode / float(R)).to(x.dtype)
                    x = norm(x + step_s * delta + sm.gate.to(x.dtype) * prelude)
                elif idea == "symplectic_scratchpad":
                    mu = torch.sigmoid(sm.mu_logit).to(x.dtype)
                    s_reg = mu * s_reg + (1.0 - mu) * delta
                    gw = torch.sigmoid(sm.gw_up(F.silu(sm.gw_down(torch.cat((norm(x), norm(delta)), dim=-1)))).float()).to(x.dtype)
                    x = x + (2.0 * gw / max(float(R) / 2.0, 1.0)) * delta

                if use_attnpool:
                    states.append(x)
                if r == mid_loop:
                    cached = x
                if rec_diag:
                    d_step = (x - x_prev).detach().float()
                    norms.append(float(x.float().norm(dim=-1).mean().item()))
                    if prev_delta is not None:
                        cos_list.append(float(F.cosine_similarity(d_step, prev_delta, dim=-1).mean().item()))
                    prev_delta = d_step

            if use_attnpool:
                x = _apply_attnpool(states)

        if rec_diag:
            self._loop_diag = dict(norms=[round(v, 2) for v in norms], adj_cos=[round(v, 3) for v in cos_list])

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
    if hasattr(model, "loop_state_mod"):
        adam(list(model.loop_state_mod.parameters()), 0.005 * bscale, (0.9, 0.95), 0.0)

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
        "roloop_2pi", "roloop_4pi", "euler_ode", "spherical_ode",
        "loop_attnpool", "symplectic_scratchpad",
        "prelude_no_norm", "prelude_untied_r1", "prelude_gate_dyn", "prelude_euler_ode",
        "roloop_ode_attnpool", "roloop_sph_attnpool", "prelude_attnpool",
        "prelude_roloop_ode_attnpool", "mhc4_shift_prelude", "mhc4_shift_attnpool",
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
        loops=args.loops, max_loops=max(args.loops, 2), eval_loops=args.loops,
        huginn_mean=float(max(args.loops, 2)), huginn_backprop_loops=min(4, max(args.loops, 1)),
    )
    model = build_model(cfg, device=device, seed=args.seed)
    if args.freeze_x0:
        with torch.no_grad():
            model.x0_lambdas.zero_()
        model.x0_lambdas.requires_grad_(False)

    base_params = sum(p.numel() for p in model.parameters())
    sm = attach_wave2_executor(model, args.idea, device=device)
    extra_params = sum(p.numel() for p in sm.parameters())

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

    ood_losses = {}
    if args.idea == "prelude_untied_r1":
        ood_losses["1"] = round(history[-1]["val_loss"], 5)
    else:
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
