#!/usr/bin/env python3
"""Direction-A Arch Wave 8: Truly Fundamental, Zero-Prior-Art-Collision Looped-LM Paradigms.

All mechanisms preserve ReferenceLM._block / resid_lambdas[phys] 100% intact, use ZERO [max_loops]
lookup tables, run with 100% official 5-step MuonAdamW + lr=0.003*bscale for new params, --freeze-x0,
and report exact zero-slice-bias ood_2048 across R_eval in {1, 2, 4, 8, 12, 16, 32}!

Core Paradigms (Carbon-Critic #104 + Carbon-Auditor #103 Bar 1-4):
  1. diff_kv_ode        : 2D Sequence x Depth Multi-Timescale + Scale-Invariant Velocity KV ODE
                          (120 scalars = 0.0004% core params, 0.0% extra FlashAttn FLOPs!)
                          K_bar = (1-lam/R)*K_bar + (lam/R)*K; K_eff = (1-a)*K + a*K_bar + d*R*(K - K_prev)
  2. kv_2d_cross_depth  : 2D Causal Cross-Depth Dual-Branch FlashAttention over (current + depth-EMA KV)
                          (48 scalars = 0.0002% core params, 0 extra weight matrices, 0 future-token leak!)
  3. cayley_subhead     : Isometric Lie-Group SO(12) Continuous Sub-Head Morphing via Cayley U(tau)
                          (~20k params = 0.07% core params, <0.3% FLOPs, exact singular-value isometry!)
  4. relu2_phase_route  : Direct ReLU^2 3072-Neuron Continuous Sparse Phase-Routing (0 [B,T] GEMMs!)
                          a = ReLU(W_fc x * (1 + gamma(tau)) + beta(tau))^2 (~0.7% core params!)
  5. state_relu2_route  : 100% Prefix-Causal Token-State Sparse ReLU^2 Routing (ZERO dependence on R!)
  6. kv_relu2_combo     : Orthogonal pairing: diff_kv_ode + relu2_phase_route (+ sanitary spherical ODE)
  7. full_paradigm_combo: diff_kv_ode + cayley_subhead + relu2_phase_route (+ sanitary spherical ODE)
  8. diff_kv_standalone : Pure diff_kv_ode WITHOUT spherical prelude anchor (measures standalone gain!)
  9. r1_20L_relu2_route : Gate A1.3 Untied 2/16/2 R=1 DiD control for relu2_phase_route!
 10. r1_20L_diff_kv     : Gate A1.3 Untied 2/16/2 R=1 DiD control for diff_kv_ode!

Supports --continuum-u412 (compute-neutral R_step ~ Uniform{4..12}, E[R]=8, 0% extra FLOPs!)
and --tbptt-loops 4 for matched iso-wall-clock comparisons!
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
from nexttransformer.models.backbone import norm, apply_rotary_emb
from nexttransformer.training.optimizer import ExperimentOptimizer
from nanochat.flash_attention import flash_attn


def fourier_depth_embed(tau: float, dim: int, device: torch.device) -> torch.Tensor:
    half = dim // 2
    freqs = math.pi * (2.0 ** torch.linspace(0.0, 4.0, half, device=device, dtype=torch.float32))
    ang = tau * freqs
    return torch.cat((torch.sin(ang), torch.cos(ang)), dim=-1)


class _LoopExecCtx:
    def __init__(self):
        self.r = 0
        self.R = 8
        self.tau = 0.5
        self.active = False
        self.kv_state = {}      # layer_idx -> dict(k_bar, v_bar, k_prev, v_prev)
        self.mem_kv = {}        # layer_idx -> (k_mem, v_mem) post-RoPE/QK-norm for 2D FlashAttn
        self.cayley_U = None    # [core_layers, 3, 12, 12]
        self.relu2_gb = None    # [core_layers, 2, 4*width]


LOOP_CTX = _LoopExecCtx()


class Wave8Mod(nn.Module):
    def __init__(self, idea: str, core_layers: int, n_head: int, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.core_layers = core_layers
        self.n_head = n_head
        self.width = width
        self.use_anchor = idea not in ("diff_kv_standalone", "r1_20L_relu2_route", "r1_20L_diff_kv")
        self.use_diff_kv = idea in ("diff_kv_ode", "kv_relu2_combo", "full_paradigm_combo",
                                    "diff_kv_standalone", "r1_20L_diff_kv")
        self.use_2d_attn = (idea == "kv_2d_cross_depth")
        self.use_cayley = idea in ("cayley_subhead", "full_paradigm_combo")
        self.use_relu2 = idea in ("relu2_phase_route", "kv_relu2_combo", "full_paradigm_combo", "r1_20L_relu2_route")
        self.use_state_relu2 = (idea == "state_relu2_route")

        if self.use_anchor:
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))

        if self.use_diff_kv:
            # 120 scalars total (4 layers x 6 heads x 5 params): lambda_ema, alpha_k, alpha_v, delta_k, delta_v
            self.kv_lam = nn.Parameter(torch.full((core_layers, n_head), 2.0, device=device, dtype=torch.float32))
            init_a = torch.linspace(-2.5, 0.0, n_head, device=device, dtype=torch.float32).unsqueeze(0).repeat(core_layers, 1)
            self.kv_alpha_k = nn.Parameter(init_a.clone())
            self.kv_alpha_v = nn.Parameter(init_a.clone())
            self.kv_delta_k = nn.Parameter(torch.zeros(core_layers, n_head, device=device, dtype=torch.float32))
            self.kv_delta_v = nn.Parameter(torch.zeros(core_layers, n_head, device=device, dtype=torch.float32))

        if self.use_2d_attn:
            # 48 scalars: per-layer, per-head memory blend gate + EMA decay
            self.mem_gate = nn.Parameter(torch.full((core_layers, n_head), -2.0, device=device, dtype=torch.float32))
            self.mem_lam = nn.Parameter(torch.full((core_layers, n_head), 2.0, device=device, dtype=torch.float32))

        if self.use_cayley:
            # 12 sub-heads of dim 64 (12*64 = 768); skew-symmetric 12x12 has 66 free upper-tri entries per (layer, Q/K/V)
            self.n_sub = width // 64
            self.n_tri = self.n_sub * (self.n_sub - 1) // 2
            self.cayley_net = nn.Sequential(
                nn.Linear(32, 32, device=device, dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(32, core_layers * 3 * self.n_tri, device=device, dtype=torch.float32),
            )
            nn.init.zeros_(self.cayley_net[-1].weight)
            nn.init.zeros_(self.cayley_net[-1].bias)
            tri_i, tri_j = torch.triu_indices(self.n_sub, self.n_sub, offset=1, device=device)
            self.register_buffer("tri_i", tri_i)
            self.register_buffer("tri_j", tri_j)

        if self.use_relu2:
            # Direct 3072-neuron ReLU^2 phase routing via bottleneck rank-8 (~200k params = 0.7% core, 0 [B,T] GEMMs!)
            mlp_w = 4 * width
            self.r2_net = nn.Sequential(
                nn.Linear(32, 32, device=device, dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(32, 8, device=device, dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(8, core_layers * 2 * mlp_w, device=device, dtype=torch.float32),
            )
            nn.init.zeros_(self.r2_net[-1].weight)
            nn.init.zeros_(self.r2_net[-1].bias)

        if self.use_state_relu2:
            # Prefix-causal token-state ReLU^2 neuron routing (rank-8 per core layer, ZERO dependence on R!)
            mlp_w = 4 * width
            self.sr2_down = nn.ModuleList([nn.Linear(width, 8, bias=False, device=device, dtype=torch.bfloat16) for _ in range(core_layers)])
            self.sr2_up = nn.ModuleList([nn.Linear(8, 2 * mlp_w, bias=True, device=device, dtype=torch.bfloat16) for _ in range(core_layers)])
            for d_l, u_l in zip(self.sr2_down, self.sr2_up):
                nn.init.normal_(d_l.weight, std=0.02)
                nn.init.zeros_(u_l.weight)
                nn.init.zeros_(u_l.bias)


def build_cayley_matrices(sm: Wave8Mod, tau: float, device: torch.device) -> torch.Tensor:
    """Computes exact SO(n_sub) Cayley orthogonal matrices U(tau) = (I-S)^{-1}(I+S) once per loop!"""
    emb = fourier_depth_embed(tau, 32, device)
    tri = sm.cayley_net(emb).view(sm.core_layers, 3, sm.n_tri)
    S = torch.zeros((sm.core_layers, 3, sm.n_sub, sm.n_sub), device=device, dtype=torch.float32)
    S[:, :, sm.tri_i, sm.tri_j] = tri
    S[:, :, sm.tri_j, sm.tri_i] = -tri
    eye = torch.eye(sm.n_sub, device=device, dtype=torch.float32).view(1, 1, sm.n_sub, sm.n_sub)
    # Solve (I - S) U = (I + S) -> exact U in SO(n_sub)!
    U = torch.linalg.solve(eye - S, eye + S)
    return U


def attach_wave8_executor(model, idea: str, device: torch.device):
    c = model.config
    is_r1_did = idea.startswith("r1_20L_")
    n_core = 4 if is_r1_did else c.core_layers
    sm = Wave8Mod(idea, n_core, c.heads, c.width, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}

    # Hook core Attention and MLP cleanly WITHOUT ever bypassing self._block / resid_lambdas[phys]!
    layer_range = range(c.pre_layers, c.pre_layers + c.core_layers)
    for l_idx in layer_range:
        blk = model.transformer.h[l_idx]
        core_j = (l_idx - c.pre_layers) % n_core

        if sm.use_diff_kv or sm.use_2d_attn or sm.use_cayley:
            orig_attn = blk.attn

            def _make_attn_fwd(attn_mod, cj):
                def _attn_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    B, T, C = x.size()
                    q_lin = self.c_q(x)
                    k_lin = self.c_k(x)
                    v_lin = self.c_v(x)

                    if LOOP_CTX.active and sm.use_cayley and LOOP_CTX.cayley_U is not None:
                        Uq = LOOP_CTX.cayley_U[cj, 0].to(x.dtype)
                        Uk = LOOP_CTX.cayley_U[cj, 1].to(x.dtype)
                        Uv = LOOP_CTX.cayley_U[cj, 2].to(x.dtype)
                        q_lin = torch.einsum("ij,btjd->btid", Uq, q_lin.view(B, T, sm.n_sub, 64)).reshape(B, T, C)
                        k_lin = torch.einsum("ij,btjd->btid", Uk, k_lin.view(B, T, sm.n_sub, 64)).reshape(B, T, C)
                        v_lin = torch.einsum("ij,btjd->btid", Uv, v_lin.view(B, T, sm.n_sub, 64)).reshape(B, T, C)

                    q = q_lin.view(B, T, self.n_head, self.head_dim)
                    k = k_lin.view(B, T, self.n_kv_head, self.head_dim)
                    v = v_lin.view(B, T, self.n_kv_head, self.head_dim)

                    if LOOP_CTX.active and sm.use_diff_kv:
                        st = LOOP_CTX.kv_state.get(cj)
                        inv_r = 1.0 / float(LOOP_CTX.R)
                        if st is None:
                            LOOP_CTX.kv_state[cj] = dict(k_bar=k, v_bar=v, k_prev=k, v_prev=v)
                        else:
                            lam = (F.softplus(sm.kv_lam[cj]) * inv_r).clamp_max(1.0).view(1, 1, self.n_kv_head, 1).to(k.dtype)
                            ak = torch.sigmoid(sm.kv_alpha_k[cj]).view(1, 1, self.n_kv_head, 1).to(k.dtype)
                            av = torch.sigmoid(sm.kv_alpha_v[cj]).view(1, 1, self.n_kv_head, 1).to(v.dtype)
                            dk = sm.kv_delta_k[cj].view(1, 1, self.n_kv_head, 1).to(k.dtype)
                            dv = sm.kv_delta_v[cj].view(1, 1, self.n_kv_head, 1).to(v.dtype)
                            k_bar_prev, v_bar_prev = st["k_bar"], st["v_bar"]
                            k_prev, v_prev = st["k_prev"], st["v_prev"]
                            k_bar_new = (1.0 - lam) * k_bar_prev + lam * k
                            v_bar_new = (1.0 - lam) * v_bar_prev + lam * v
                            # Exact R-invariant continuous KV velocity: R * (K^(r) - K^(r-1)) = dK/dtau!
                            r_f = float(LOOP_CTX.R)
                            k_eff = (1.0 - ak) * k + ak * k_bar_prev + dk * (r_f * (k - k_prev))
                            v_eff = (1.0 - av) * v + av * v_bar_prev + dv * (r_f * (v - v_prev))
                            LOOP_CTX.kv_state[cj] = dict(k_bar=k_bar_new, v_bar=v_bar_new, k_prev=k, v_prev=v)
                            k, v = k_eff, v_eff

                    if ve is not None:
                        ve_v = ve.view(B, T, self.n_kv_head, self.head_dim)
                        gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                        v = v + gate.unsqueeze(-1) * ve_v

                    cos, sin = cos_sin
                    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                    q, k = norm(q), norm(k)
                    q = q * 1.2
                    k = k * 1.2

                    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

                    if LOOP_CTX.active and sm.use_2d_attn:
                        mem = LOOP_CTX.mem_kv.get(cj)
                        inv_r = 1.0 / float(LOOP_CTX.R)
                        if mem is None:
                            LOOP_CTX.mem_kv[cj] = (k, v)
                        else:
                            k_mem, v_mem = mem
                            y_mem = flash_attn.flash_attn_func(q, k_mem, v_mem, causal=True, window_size=window_size)
                            w = torch.sigmoid(sm.mem_gate[cj]).view(1, 1, self.n_head, 1).to(y.dtype)
                            y = (1.0 - w) * y + w * y_mem
                            lam = (F.softplus(sm.mem_lam[cj]) * inv_r).clamp_max(1.0).view(1, 1, self.n_kv_head, 1).to(k.dtype)
                            LOOP_CTX.mem_kv[cj] = ((1.0 - lam) * k_mem + lam * k, (1.0 - lam) * v_mem + lam * v)

                    y = y.contiguous().view(B, T, -1)
                    return self.c_proj(y)
                return _attn_fwd

            orig_attn.forward = types.MethodType(_make_attn_fwd(orig_attn, core_j), orig_attn)

        if sm.use_relu2 or sm.use_state_relu2:
            orig_mlp = blk.mlp

            def _make_mlp_fwd(cj):
                def _mlp_fwd(self, x):
                    h = self.c_fc(x)
                    if LOOP_CTX.active and sm.use_relu2 and LOOP_CTX.relu2_gb is not None:
                        gam = LOOP_CTX.relu2_gb[cj, 0].to(h.dtype)
                        bet = LOOP_CTX.relu2_gb[cj, 1].to(h.dtype)
                        h = h * (1.0 + gam) + bet
                    elif LOOP_CTX.active and sm.use_state_relu2:
                        gb = sm.sr2_up[cj](F.silu(sm.sr2_down[cj](x)))
                        gam, bet = gb.chunk(2, dim=-1)
                        h = h * (1.0 + gam.to(h.dtype)) + bet.to(h.dtype)
                    h = F.relu(h).square()
                    return self.c_proj(h)
                return _mlp_fwd

            orig_mlp.forward = types.MethodType(_make_mlp_fwd(core_j), orig_mlp)

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
        LOOP_CTX.active = False
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

        LOOP_CTX.kv_state.clear()
        LOOP_CTX.mem_kv.clear()
        LOOP_CTX.R = R_eff
        LOOP_CTX.active = True
        try:
            for r in range(R_eff):
                no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
                if self.training and tbptt > 0 and r == R_eff - tbptt:
                    x = x.detach()
                    for k_st in list(LOOP_CTX.kv_state.keys()):
                        LOOP_CTX.kv_state[k_st] = {kk: vv.detach() for kk, vv in LOOP_CTX.kv_state[k_st].items()}
                    for k_m in list(LOOP_CTX.mem_kv.keys()):
                        km, vm = LOOP_CTX.mem_kv[k_m]
                        LOOP_CTX.mem_kv[k_m] = (km.detach(), vm.detach())

                with torch.no_grad() if no_g else nullcontext():
                    LOOP_CTX.r = r
                    LOOP_CTX.tau = (float(r) + 0.5) * inv_r
                    if sm.use_cayley:
                        LOOP_CTX.cayley_U = build_cayley_matrices(sm, LOOP_CTX.tau, x.device)
                    if sm.use_relu2:
                        emb = fourier_depth_embed(LOOP_CTX.tau, 32, x.device)
                        LOOP_CTX.relu2_gb = sm.r2_net(emb).view(sm.core_layers, 2, 4 * c.width)

                    x_in = x
                    h_out = _run_group(self, x_in, x0, idx, r, total)
                    delta = h_out - x_in

                    if sm.use_anchor:
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
        finally:
            LOOP_CTX.active = False
            LOOP_CTX.kv_state.clear()
            LOOP_CTX.mem_kv.clear()
            LOOP_CTX.cayley_U = None
            LOOP_CTX.relu2_gb = None

        if rec_diag:
            self._loop_diag = dict(raw_delta_cos=[round(v, 3) for v in raw_cos_list])

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
        "diff_kv_ode", "kv_2d_cross_depth", "cayley_subhead",
        "relu2_phase_route", "state_relu2_route",
        "kv_relu2_combo", "full_paradigm_combo",
        "diff_kv_standalone", "r1_20L_relu2_route", "r1_20L_diff_kv",
    ])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tbptt-loops", type=int, default=0)
    ap.add_argument("--continuum-u412", action="store_true", help="Compute-neutral R_step ~ Uniform{4..12} (E[R]=8)")
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
    sm = attach_wave8_executor(model, args.idea, device=device)
    model._tbptt_loops = args.tbptt_loops
    extra_params = sum(p.numel() for p in sm.parameters())

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|R={args.loops}|u412={int(args.continuum_u412)}|+p={extra_params} ({100*extra_params/base_params:.4f}%)] "
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
        tag=args.tag, idea=args.idea, loops=args.loops, continuum_u412=args.continuum_u412,
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
          f"raw_cos={cos_str} {out['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
