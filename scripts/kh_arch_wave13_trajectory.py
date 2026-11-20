#!/usr/bin/env python3
"""Direction-A Arch Wave 13: Breaking the 1-Step 768-Dim Markov Bottleneck on Top of Zero-Overlap Cycle!

Empirically Proven Law from Waves 1-12 (~190 B300 runs):
  - Adjacent Transformer layers l -> l+1 MUST have 0% overlapping weights (Cycle beats 2-rev by -0.021,
    1-rev DepthSlide/RibbonLoop by -0.047, and 0-rev Layer-Repeat by -0.074!).
  - Single-matrix tiny weight modulations (LoRA, AdaLN, SO(k) LieLoop) plateau at -0.002..-0.006 over prelude_ode.
  - Meanwhile, in Wave 1, Multi-Stream mHC-4 ALONE (WITHOUT prelude_ode!) was our #2 biggest architectural jump EVER
    (-0.03051 nats: 3.3647 vs 3.3952!), only dropped for running Sinkhorn 32x per forward (-17% speed).

Wave 13 Attacks the TRUE Remaining Architectural Bottleneck — the 1-Step 768-Dim Markov Residual Stream:
  Keep 100% disjoint Cycle [0,1,2,3] x R core blocks (exact 28.3M core weights, 1.00x core FLOPs),
  and expand CROSS-LOOP STATE BANDWIDTH & TRAJECTORY MEMORY at the LOOP BOUNDARY ONLY (< 2% wall-clock!):

Ideas tested in Wave 13 (ALL preserving resid_lambdas, --freeze-x0, exact ood_2048, ZERO [max_loops] lookup!):
  1. depth_attnres    : Continuous Token-Adaptive Depth-Attention over ALL past loop states [H_0=pre, H_1..H_r]
                        at loop input AND coda readout (5 lines, < 0.05% params, arbitrary continuous R=1..32!).
  2. loop_mhc4_ode    : Loop-Boundary M=4 Parallel Residual Streams (3,072-dim recurrent working memory!) with
                        Sinkhorn Doubly-Stochastic mixing H_res in B_4 + 1/R spherical ODE scaling (< 2% overhead!).
  3. loop_mhc4_plain  : Pure Euclidean Loop-Boundary M=4 mHC-4 (NO inter-loop norm() wipe! Tests #109 on Fresh-2250!).
  4. loop_mhc8_ode    : Loop-Boundary M=8 Parallel Residual Streams (6,144-dim recurrent working memory!) + ODE!
  5. mhc4_depth_attn  : Loop-Boundary M=4 Parallel Streams + Continuous Depth-Attention Trajectory Readout!
  6. gated_traj_ode   : Token-Adaptive Gated Spherical ODE (each token dynamically gates its update step & prelude pull!).
  7. kv_ema_bounded   : Bounded Cross-Loop KV EMA memory (ZERO exploding R*(k-k_prev) term!) + prelude_ode!
  8. mhc4_r1_20L      : Mandatory Gate A1.3 Untied 2/16/2 R=1 DiD control for loop_mhc4_ode!
  9. depth_attn_r1_20L: Mandatory Gate A1.3 Untied 2/16/2 R=1 DiD control for depth_attnres!
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


def sinkhorn_doubly_stochastic(logits: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Projects [..., M, M] onto the Birkhoff polytope of doubly-stochastic matrices (exact norm non-expansion!)."""
    P = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    for _ in range(n_iters):
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)
        P = P / (P.sum(dim=-2, keepdim=True) + 1e-6)
    return P


def legendre_p1_p4(tau: float, device: torch.device) -> torch.Tensor:
    t = 2.0 * float(tau) - 1.0
    t2 = t * t
    return torch.tensor([
        t,
        0.5 * (3.0 * t2 - 1.0),
        0.5 * t * (5.0 * t2 - 3.0),
        0.125 * (35.0 * t2 * t2 - 30.0 * t2 + 3.0),
    ], device=device, dtype=torch.float32)


class _KVBCtx:
    def __init__(self):
        self.active = False
        self.R = 8
        self.state = {}


KVB_CTX = _KVBCtx()


class TrajMod(nn.Module):
    def __init__(self, idea: str, core_layers: int, n_head: int, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.width = width
        self.use_mhc = idea in ("loop_mhc4_ode", "loop_mhc4_plain", "loop_mhc8_ode", "mhc4_depth_attn", "mhc4_r1_20L")
        self.mhc_plain = (idea == "loop_mhc4_plain")
        self.M = 8 if idea == "loop_mhc8_ode" else (4 if self.use_mhc else 1)
        self.use_dattn = idea in ("depth_attnres", "mhc4_depth_attn", "depth_attn_r1_20L")
        self.use_gated_ode = (idea == "gated_traj_ode")
        self.use_kv_ema = (idea == "kv_ema_bounded")
        self.is_r1_did = idea in ("mhc4_r1_20L", "depth_attn_r1_20L")

        # Base spherical ODE scalars for anchor-using variants
        if idea in ("depth_attnres", "kv_ema_bounded", "depth_attn_r1_20L"):
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))

        if self.use_mhc:
            M = self.M
            # Per-stream dynamic mixing at loop boundary ONLY (8x per forward -> < 2% wall-clock!)
            # Base identity-biased doubly stochastic logits + continuous Legendre P1..P4 depth modulation!
            eye_logits = torch.eye(M, device=device, dtype=torch.float32) * 3.0
            self.H_base = nn.Parameter(eye_logits)                              # [M, M]
            self.H_leg = nn.Parameter(torch.zeros(4, M, M, device=device))      # [4, M, M] continuous depth!
            # Token-adaptive dynamic read/write weights across M streams (zero-init dynamic part!)
            self.h_in_base = nn.Parameter(torch.zeros(M, device=device))        # softmax -> uniform 1/M at step 0
            self.h_in_leg = nn.Parameter(torch.zeros(4, M, device=device))
            self.h_in_tok = nn.Linear(width, M, bias=False, device=device, dtype=torch.float32)
            nn.init.zeros_(self.h_in_tok.weight)

            self.h_out_base = nn.Parameter(torch.ones(M, device=device))
            self.h_out_leg = nn.Parameter(torch.zeros(4, M, device=device))
            self.h_out_tok = nn.Linear(width, M, bias=False, device=device, dtype=torch.float32)
            nn.init.zeros_(self.h_out_tok.weight)

            # Final readout weights over M streams
            self.readout_w = nn.Parameter(torch.zeros(M, device=device))
            # Continuous ODE step & prelude anchor per stream (unit-RMS matched to prelude_true_ode!)
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            if not self.mhc_plain:
                self.g_pre = nn.Parameter(torch.full((M, width), 1.5, device=device, dtype=torch.float32))
                with torch.no_grad():
                    self.h_out_leg[0] = torch.linspace(-0.4, 0.4, M, device=device)
                    self.h_out_leg[1] = torch.cos(torch.linspace(0, math.pi, M, device=device)) * 0.3
            else:
                with torch.no_grad():
                    self.h_out_leg[0] = torch.linspace(-0.3, 0.3, M, device=device)
                    self.h_out_leg[1] = torch.cos(torch.linspace(0, math.pi, M, device=device)) * 0.25

        if self.use_dattn:
            # Continuous Token-Adaptive Depth-Attention over [H_0=prelude, H_1..H_r]
            # Tiny 64-dim Q/K projections + continuous Legendre depth query -> ZERO [max_loops] lookup table!
            self.da_dim = 64
            self.da_q = nn.Linear(width, self.da_dim, bias=False, device=device, dtype=torch.float32)
            self.da_k = nn.Linear(width, self.da_dim, bias=False, device=device, dtype=torch.float32)
            self.da_q_leg = nn.Linear(4, self.da_dim, bias=False, device=device, dtype=torch.float32)
            self.da_k_leg = nn.Linear(4, self.da_dim, bias=False, device=device, dtype=torch.float32)
            nn.init.normal_(self.da_q.weight, std=0.02)
            nn.init.normal_(self.da_k.weight, std=0.02)
            nn.init.zeros_(self.da_q_leg.weight)
            nn.init.zeros_(self.da_k_leg.weight)
            # Zero-init residual mix gates (loop input & coda readout) so step=0 is bit-for-bit exact prelude_ode!
            self.da_in_gate = nn.Parameter(torch.zeros(width, device=device, dtype=torch.float32))
            self.da_out_q = nn.Parameter(torch.zeros(self.da_dim, device=device, dtype=torch.float32))
            self.da_out_gate = nn.Parameter(torch.zeros(width, device=device, dtype=torch.float32))

        if self.use_gated_ode:
            # Token-Adaptive Gated Spherical ODE: per-token step size s(x, tau) & prelude pull g(x, tau)!
            # Bottleneck width -> 32 -> 2*width (~74k params = 0.036%!)
            self.s_base = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_base = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))
            self.gate_in = nn.Linear(width, 32, bias=False, device=device, dtype=torch.float32)
            self.gate_leg = nn.Linear(4, 32, bias=True, device=device, dtype=torch.float32)
            self.gate_out = nn.Linear(32, 2 * width, bias=True, device=device, dtype=torch.float32)
            nn.init.zeros_(self.gate_out.weight)
            nn.init.zeros_(self.gate_out.bias)

        if self.use_kv_ema:
            # Bounded Cross-Loop KV EMA (NO unbounded R*(k-k_prev) finite diff -> 100% bounded OOD R=16,32!)
            self.kv_lam = nn.Parameter(torch.full((core_layers, n_head), 2.0, device=device, dtype=torch.float32))
            init_a = torch.linspace(-2.5, -0.5, n_head, device=device, dtype=torch.float32).unsqueeze(0).repeat(core_layers, 1)
            self.kv_alpha_k = nn.Parameter(init_a.clone())
            self.kv_alpha_v = nn.Parameter(init_a.clone())


def attach_wave13_executor(model, idea: str, device: torch.device):
    c = model.config
    is_r1_did = idea in ("mhc4_r1_20L", "depth_attn_r1_20L")
    n_core = 4 if is_r1_did else c.core_layers
    sm = TrajMod(idea, n_core, c.heads, c.width, device=device)
    model.loop_state_mod = sm

    if sm.use_kv_ema:
        for l_idx in range(c.pre_layers, c.pre_layers + c.core_layers):
            blk = model.transformer.h[l_idx]
            core_j = (l_idx - c.pre_layers) % n_core

            def _make_attn_fwd(cj):
                def _attn_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    B, T, C = x.size()
                    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
                    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
                    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
                    if KVB_CTX.active:
                        st = KVB_CTX.state.get(cj)
                        inv_r = 1.0 / float(KVB_CTX.R)
                        if st is None:
                            KVB_CTX.state[cj] = (k, v)
                        else:
                            lam = (F.softplus(sm.kv_lam[cj]) * inv_r).clamp_max(1.0).view(1, 1, self.n_kv_head, 1).to(k.dtype)
                            ak = torch.sigmoid(sm.kv_alpha_k[cj]).view(1, 1, self.n_kv_head, 1).to(k.dtype)
                            av = torch.sigmoid(sm.kv_alpha_v[cj]).view(1, 1, self.n_kv_head, 1).to(v.dtype)
                            kb, vb = st
                            kb_new = (1.0 - lam) * kb + lam * k
                            vb_new = (1.0 - lam) * vb + lam * v
                            k = (1.0 - ak) * k + ak * kb
                            v = (1.0 - av) * v + av * vb
                            KVB_CTX.state[cj] = (kb_new, vb_new)
                    if ve is not None:
                        ve_v = ve.view(B, T, self.n_kv_head, self.head_dim)
                        gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                        v = v + gate.unsqueeze(-1) * ve_v
                    cos, sin = cos_sin
                    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                    q, k = norm(q) * 1.2, norm(k) * 1.2
                    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
                    return self.c_proj(y.contiguous().view(B, T, -1))
                return _attn_fwd
            blk.attn.forward = types.MethodType(_make_attn_fwd(core_j), blk.attn)

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

    def _depth_attn_query(q_vec: torch.Tensor, H_list: list[torch.Tensor], K_list: list[torch.Tensor]) -> torch.Tensor:
        """Token-level causal depth-attention over trajectory states H_0..H_r.
        q_vec: [B, T, da_dim]; K_list: list of [B, T, da_dim]; H_list: list of [B, T, C].
        """
        K_stack = torch.stack(K_list, dim=2)  # [B, T, L, da_dim]
        scores = torch.einsum("btd,btld->btl", q_vec, K_stack) * (sm.da_dim ** -0.5)
        w = torch.softmax(scores.float(), dim=-1).to(H_list[0].dtype)  # [B, T, L]
        H_stack = torch.stack(H_list, dim=2)  # [B, T, L, C]
        return torch.einsum("btl,btlc->btc", w, H_stack)

    def _custom_ordinary_hidden(self, idx):
        R_eff = 4 if is_r1_did else (getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops)
        total = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        KVB_CTX.active = False
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        inv_r = 1.0 / float(R_eff)
        dt = x.dtype

        # Multi-stream state X: [B, T, M, C] for mHC variants
        if sm.use_mhc:
            X = x.unsqueeze(2).expand(-1, -1, sm.M, -1).clone()

        # Trajectory memory for Continuous Depth-Attention
        H_traj = []
        K_traj = []
        if sm.use_dattn:
            leg0 = legendre_p1_p4(0.0, x.device)
            H_traj.append(prelude)
            K_traj.append((sm.da_k(norm(prelude).float()) + sm.da_k_leg(leg0)).to(dt))

        KVB_CTX.state.clear()
        KVB_CTX.R = R_eff
        KVB_CTX.active = True
        try:
            for r in range(R_eff):
                tau = (float(r) + 0.5) * inv_r
                leg = legendre_p1_p4(tau, x.device)  # [4]

                if sm.use_mhc:
                    # 1. Read from M parallel streams into 768-dim core input (token + depth adaptive!)
                    x_mean = X.mean(dim=2)  # [B, T, C]
                    in_logits = (sm.h_in_base + (leg @ sm.h_in_leg)).view(1, 1, sm.M) + sm.h_in_tok(norm(x_mean).float())
                    w_in = torch.softmax(in_logits, dim=-1).to(dt)  # [B, T, M]
                    x_in = torch.einsum("btm,btmc->btc", w_in, X)

                    # Optional trajectory attention readout into core input
                    if sm.use_dattn and len(H_traj) > 1:
                        q_v = (sm.da_q(norm(x_in).float()) + sm.da_q_leg(leg)).to(dt)
                        x_mem = _depth_attn_query(q_v, H_traj, K_traj)
                        x_in = x_in + sm.da_in_gate.to(dt) * (x_mem - x_in)

                    h_out = _run_group(self, x_in, x0, idx, r, total)
                    delta = h_out - x_in

                    # 2. Doubly-stochastic Sinkhorn mixing across M streams + dynamic stream write!
                    H_logits = sm.H_base + torch.einsum("m,mij->ij", leg, sm.H_leg)  # [M, M]
                    H_mix = sinkhorn_doubly_stochastic(H_logits, n_iters=5).to(dt)  # [M, M]
                    X_mix = torch.einsum("ij,btjc->btic", H_mix, X)                 # [B, T, M, C]

                    w_out = ((sm.h_out_base + (leg @ sm.h_out_leg)).view(1, 1, sm.M)
                             + sm.h_out_tok(norm(delta).float())).to(dt)            # [B, T, M]
                    write_term = w_out.unsqueeze(-1) * delta.unsqueeze(2)           # [B, T, M, C]

                    if is_r1_did or sm.mhc_plain:
                        X = X_mix + write_term
                    else:
                        # Per-stream spherical ODE with prelude anchor!
                        pre_pull = sm.g_pre.to(dt).view(1, 1, sm.M, c.width) * prelude.unsqueeze(2)
                        X = norm(X_mix + inv_r * (sm.s_ode.to(dt) * write_term + pre_pull))

                    x_cur = X.mean(dim=2)
                    if sm.use_dattn:
                        H_traj.append(x_cur)
                        K_traj.append((sm.da_k(norm(x_cur).float()) + sm.da_k_leg(leg)).to(dt))
                    if r == mid_g:
                        cached = x_cur
                else:
                    x_in = x
                    if sm.use_dattn and len(H_traj) > 1:
                        q_v = (sm.da_q(norm(x_in).float()) + sm.da_q_leg(leg)).to(dt)
                        x_mem = _depth_attn_query(q_v, H_traj, K_traj)
                        x_in = x_in + sm.da_in_gate.to(dt) * (x_mem - x_in)

                    h_out = _run_group(self, x_in, x0, idx, r, total)
                    delta = h_out - x_in

                    if is_r1_did:
                        x = x_in + delta
                    elif sm.use_gated_ode:
                        gh = F.silu(sm.gate_in(norm(x_in).float()) + sm.gate_leg(leg))  # [B, T, 32]
                        sg = sm.gate_out(gh)  # [B, T, 2C]
                        ds, dg = sg.chunk(2, dim=-1)
                        # Bounded smooth per-token modulation around s_base=4.0 and g_base=1.5
                        s_tok = (sm.s_base * (1.0 + 0.5 * torch.tanh(ds))).to(dt)
                        g_tok = (sm.g_base.view(1, 1, c.width) * (1.0 + 0.5 * torch.tanh(dg))).to(dt)
                        x = norm(x_in + inv_r * (s_tok * delta + g_tok * prelude))
                    else:
                        x = norm(x_in + inv_r * (sm.s_ode.to(dt) * delta + sm.g_pre.to(dt) * prelude))

                    if sm.use_dattn:
                        H_traj.append(x)
                        K_traj.append((sm.da_k(norm(x).float()) + sm.da_k_leg(leg)).to(dt))
                    if r == mid_g:
                        cached = x
        finally:
            KVB_CTX.active = False
            KVB_CTX.state.clear()

        if sm.use_mhc:
            rw = torch.softmax(sm.readout_w, dim=0).to(dt)  # [M]
            x = torch.einsum("m,btmc->btc", rw, X)

        if sm.use_dattn and len(H_traj) > 1:
            q_out = sm.da_out_q.view(1, 1, sm.da_dim).expand(x.size(0), x.size(1), -1).to(dt)
            x_ro = _depth_attn_query(q_out, H_traj, K_traj)
            x = x + sm.da_out_gate.to(dt) * (x_ro - x)

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
        "depth_attnres", "loop_mhc4_ode", "loop_mhc4_plain", "loop_mhc8_ode",
        "mhc4_depth_attn", "gated_traj_ode", "kv_ema_bounded",
        "mhc4_r1_20L", "depth_attn_r1_20L",
    ])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
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

    base_params = sum(p.numel() for p in model.parameters())
    sm = attach_wave13_executor(model, args.idea, device=device)
    extra_params = sum(p.numel() for p in sm.parameters())

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|R={args.loops}|u412={int(args.continuum_u412)}|"
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
        tag=args.tag, idea=args.idea, loops=args.loops, continuum_u412=args.continuum_u412,
        steps=args.steps, data_dir=str(args.data_dir),
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
