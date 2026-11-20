#!/usr/bin/env python3
"""Direction-A Arch Wave 9: LieLoop — Full-Rank Continuous Isometric Lie-Group Flow on SO(k) for Looped LMs!

Wave-8 Discovery:
  - Additive rank-32 LoRA (func_weight_lora, +1.82M params) FAILED (3.385-3.397).
  - Isometric SO(12) Cayley sub-head rotation (cayley_subhead, ONLY +28k params = 0.013%!)
    was the #1 single-factor breakthrough (s375 = 3.69865, -0.0076 on top of prelude_true_ode!),
    and combined with diff_kv_ode + relu2_phase_route smashed to s375 = 3.69362!

Why LieLoop is a Fundamental Paradigm (Zero Prior-Art Collision, 100% Bitter-Lesson Native):
  Instead of additive low-rank patches (Bae et al. 2025) or diagonal scaling (DiT AdaLN),
  LieLoop continuously evolves EVERY shared weight matrix along a geodesic in the Special
  Orthogonal Lie Group SO(k) via exact Cayley transforms U(tau) = (I - S(tau))^{-1}(I + S(tau)):
      W_eff(tau) = U_L(tau) W U_R(tau)^T  <=>  full-rank operator morphing with EXACT spectral isometry!
  - Preserves ALL singular values sigma_i(W_eff(tau)) == sigma_i(W) for all tau in [0, 1] and all R!
  - ZERO [max_loops] lookup tables; smooth continuum interpolation to arbitrary unseen R=16, 32!
  - < 0.1% params, < 1.5% FLOPs (one k x k linear solve ONCE per loop, ZERO [B, T] GEMMs!).

Ideas tested in Wave 9 (ALL preserving resid_lambdas[phys], --freeze-x0, exact ood_2048):
  1. lieloop_attn_so12  : Isometric SO(12) Cayley on Attn Q, K, V, O (left + right!)
  2. lieloop_full_so12  : Full-Block Isometric SO(12) LieLoop on BOTH Attn (Q,K,V,O) AND MLP (up, down)!
  3. lieloop_full_so24  : Finer-grained SO(24) (32-dim blocks) Full-Block Isometric LieLoop on Attn + MLP!
  4. lieloop_givens_state: 100% Token-Causal State-Driven Isometric Givens SO(2)^{d/2} Flow (ZERO tau/R, 0 batch leak!)
  5. lieloop_full_kv    : Full-Block SO(12) LieLoop + 120-scalar 2D Sequence x Depth diff_kv_ode!
  6. lieloop_full_r2_32 : Full-Block SO(12) LieLoop + Rank-32 Direct ReLU^2 Sparse Phase-Routing (fixes #106 rank-8!)
  7. lieloop_grand      : Full-Block SO(12) + diff_kv_ode + Rank-32 ReLU^2 (pass --continuum-u412 for U{4..12}!)
  8. relu2_rank32_only  : Isolated Rank-32 Direct ReLU^2 Phase-Routing (fixes #106 rank-8 bottleneck LOO!)
  9. adaln_anchor_fixed : Wave-7 depth_adaln_anchor WITH #103 resid_lambdas[phys] bug FIXED (clean prior-art control!)
 10. lieloop_r1_20L     : Gate A1.3 Untied 2/16/2 R=1 DiD control for Full-Block SO(12) LieLoop!
 11. lieloop_no_anchor  : Pure Full-Block SO(12) LieLoop WITH 1/R Euler scaling ONLY (NO prelude anchor! Tests standalone!)
 12. lieloop_so12_conj  : Full-Block SO(12) LieLoop with EXTRA Attn-input right-rotation (7 slots: Attn_in, Q, K, V, O, MLP_in, MLP_out!)
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


class _LieCtx:
    def __init__(self):
        self.r = 0
        self.R = 8
        self.tau = 0.5
        self.active = False
        self.kv_state = {}
        self.U_mats = None     # [core_layers, n_slots, k, k]
        self.relu2_gb = None   # [core_layers, 2, 4*width]
        self.adaln_mod = None  # [core_layers, 6, width]


LIE_CTX = _LieCtx()


class LieLoopMod(nn.Module):
    def __init__(self, idea: str, core_layers: int, n_head: int, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.core_layers = core_layers
        self.n_head = n_head
        self.width = width

        self.use_anchor = idea not in ("lieloop_no_anchor", "lieloop_r1_20L")
        self.use_euler_only = (idea == "lieloop_no_anchor")
        self.use_givens_state = (idea == "lieloop_givens_state")
        self.use_lie = idea in (
            "lieloop_attn_so12", "lieloop_full_so12", "lieloop_full_so24",
            "lieloop_so12_conj", "lieloop_full_kv", "lieloop_full_r2_32",
            "lieloop_grand", "lieloop_r1_20L", "lieloop_no_anchor",
        )
        self.use_attn_in_lie = (idea == "lieloop_so12_conj")
        self.use_mlp_lie = idea in (
            "lieloop_full_so12", "lieloop_full_so24", "lieloop_so12_conj",
            "lieloop_full_kv", "lieloop_full_r2_32",
            "lieloop_grand", "lieloop_r1_20L", "lieloop_no_anchor",
        )
        self.use_diff_kv = idea in ("lieloop_full_kv", "lieloop_grand")
        self.use_relu2_32 = idea in ("relu2_rank32_only", "lieloop_full_r2_32", "lieloop_grand")
        self.use_adaln_fixed = (idea == "adaln_anchor_fixed")

        if self.use_anchor:
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))
        elif self.use_euler_only:
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if self.use_givens_state:
            # 100% token-causal autonomous isometric Givens SO(2)^{width/2} rotation: zero tau, zero R, zero batch leak!
            self.givens_attn_in = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(width, 16, bias=False, device=device, dtype=torch.float32),
                    nn.SiLU(),
                    nn.Linear(16, width // 2, bias=True, device=device, dtype=torch.float32),
                ) for _ in range(core_layers)
            ])
            self.givens_mlp_in = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(width, 16, bias=False, device=device, dtype=torch.float32),
                    nn.SiLU(),
                    nn.Linear(16, width // 2, bias=True, device=device, dtype=torch.float32),
                ) for _ in range(core_layers)
            ])
            for mlist in (self.givens_attn_in, self.givens_mlp_in):
                for m in mlist:
                    nn.init.zeros_(m[-1].weight)
                    nn.init.zeros_(m[-1].bias)

        if self.use_lie:
            # SO(12) => block dim 64 (12*64=768); SO(24) => block dim 32 (24*32=768)
            self.k_sub = 24 if idea == "lieloop_full_so24" else 12
            self.d_sub = width // self.k_sub
            assert width % self.k_sub == 0
            # Slots per layer: 0:Q, 1:K, 2:V, 3:Attn_O, [4:MLP_in, 5:MLP_out], [6:Attn_in]
            self.n_slots = 7 if self.use_attn_in_lie else (6 if self.use_mlp_lie else 4)
            self.n_tri = self.k_sub * (self.k_sub - 1) // 2
            hid_dim = 48 if self.k_sub == 24 else 32
            self.lie_net = nn.Sequential(
                nn.Linear(32, hid_dim, device=device, dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(hid_dim, core_layers * self.n_slots * self.n_tri, device=device, dtype=torch.float32),
            )
            nn.init.zeros_(self.lie_net[-1].weight)
            nn.init.zeros_(self.lie_net[-1].bias)
            tri_i, tri_j = torch.triu_indices(self.k_sub, self.k_sub, offset=1, device=device)
            self.register_buffer("tri_i", tri_i)
            self.register_buffer("tri_j", tri_j)

        if self.use_diff_kv:
            self.kv_lam = nn.Parameter(torch.full((core_layers, n_head), 2.0, device=device, dtype=torch.float32))
            init_a = torch.linspace(-2.5, 0.0, n_head, device=device, dtype=torch.float32).unsqueeze(0).repeat(core_layers, 1)
            self.kv_alpha_k = nn.Parameter(init_a.clone())
            self.kv_alpha_v = nn.Parameter(init_a.clone())
            self.kv_delta_k = nn.Parameter(torch.zeros(core_layers, n_head, device=device, dtype=torch.float32))
            self.kv_delta_v = nn.Parameter(torch.zeros(core_layers, n_head, device=device, dtype=torch.float32))

        if self.use_relu2_32:
            # Direct Fourier(32) -> SiLU(32) -> 2*3072 (4x higher rank than Wave 8's rank-8 bottleneck!)
            mlp_w = 4 * width
            self.r2_net = nn.Sequential(
                nn.Linear(32, 32, device=device, dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(32, core_layers * 2 * mlp_w, device=device, dtype=torch.float32),
            )
            nn.init.zeros_(self.r2_net[-1].weight)
            nn.init.zeros_(self.r2_net[-1].bias)

        if self.use_adaln_fixed:
            self.adaln_net = nn.Sequential(
                nn.Linear(64, 64, device=device, dtype=torch.float32),
                nn.SiLU(),
                nn.Linear(64, core_layers * 6 * width, device=device, dtype=torch.float32),
            )
            nn.init.zeros_(self.adaln_net[-1].weight)
            nn.init.zeros_(self.adaln_net[-1].bias)


def build_cayley_U(sm: LieLoopMod, cond: torch.Tensor) -> torch.Tensor:
    """Computes exact SO(k) Cayley orthogonal matrices U = (I-S)^{-1}(I+S) ONCE per loop!"""
    tri = sm.lie_net(cond).view(sm.core_layers, sm.n_slots, sm.n_tri)
    S = torch.zeros((sm.core_layers, sm.n_slots, sm.k_sub, sm.k_sub), device=cond.device, dtype=torch.float32)
    S[:, :, sm.tri_i, sm.tri_j] = tri
    S[:, :, sm.tri_j, sm.tri_i] = -tri
    eye = torch.eye(sm.k_sub, device=cond.device, dtype=torch.float32).view(1, 1, sm.k_sub, sm.k_sub)
    return torch.linalg.solve(eye - S, eye + S)


def apply_so_rot(x: torch.Tensor, U: torch.Tensor, k_sub: int, d_sub: int) -> torch.Tensor:
    B, T, C = x.shape
    return torch.einsum("ij,btjd->btid", U.to(x.dtype), x.view(B, T, k_sub, d_sub)).reshape(B, T, C)


def apply_givens_rot(x: torch.Tensor, net: nn.Module) -> torch.Tensor:
    """Exact token-causal isometric 2x2 Givens rotation SO(2)^{C/2}: ||G(x_t) x_t||_2 == ||x_t||_2."""
    theta = net(x.float())
    c = torch.cos(theta).to(x.dtype)
    s = torch.sin(theta).to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    y1 = c * x1 - s * x2
    y2 = s * x1 + c * x2
    return torch.stack((y1, y2), dim=-1).flatten(-2)


def attach_wave9_executor(model, idea: str, device: torch.device):
    c = model.config
    is_r1_did = (idea == "lieloop_r1_20L")
    n_core = 4 if is_r1_did else c.core_layers
    sm = LieLoopMod(idea, n_core, c.heads, c.width, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}

    for l_idx in range(c.pre_layers, c.pre_layers + c.core_layers):
        blk = model.transformer.h[l_idx]
        core_j = (l_idx - c.pre_layers) % n_core

        # Hook Block.forward ONLY for adaln_anchor_fixed (still called INSIDE self._block so resid_lambdas[phys] is 100% preserved!)
        if sm.use_adaln_fixed:
            def _make_blk_fwd(cj):
                def _blk_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    if LIE_CTX.active and LIE_CTX.adaln_mod is not None:
                        m = LIE_CTX.adaln_mod[cj]  # [6, width]
                        sa, sca, ga, sm_, scm, gm = [m[i].to(x.dtype) for i in range(6)]
                        x = x + (1.0 + ga) * self.attn(norm(x) * (1.0 + sca) + sa, ve, cos_sin, window_size, kv_cache)
                        x = x + (1.0 + gm) * self.mlp(norm(x) * (1.0 + scm) + sm_)
                        return x
                    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
                    x = x + self.mlp(norm(x))
                    return x
                return _blk_fwd
            blk.forward = types.MethodType(_make_blk_fwd(core_j), blk)

        if sm.use_lie or sm.use_diff_kv or sm.use_givens_state:
            def _make_attn_fwd(cj):
                def _attn_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    B, T, C = x.size()
                    x_attn = x
                    if LIE_CTX.active and sm.use_givens_state:
                        x_attn = apply_givens_rot(x_attn, sm.givens_attn_in[cj])
                    elif LIE_CTX.active and sm.use_attn_in_lie and LIE_CTX.U_mats is not None:
                        x_attn = apply_so_rot(x_attn, LIE_CTX.U_mats[cj, 6], sm.k_sub, sm.d_sub)
                    q_lin = self.c_q(x_attn)
                    k_lin = self.c_k(x_attn)
                    v_lin = self.c_v(x_attn)

                    if LIE_CTX.active and sm.use_lie and LIE_CTX.U_mats is not None:
                        q_lin = apply_so_rot(q_lin, LIE_CTX.U_mats[cj, 0], sm.k_sub, sm.d_sub)
                        k_lin = apply_so_rot(k_lin, LIE_CTX.U_mats[cj, 1], sm.k_sub, sm.d_sub)
                        v_lin = apply_so_rot(v_lin, LIE_CTX.U_mats[cj, 2], sm.k_sub, sm.d_sub)

                    q = q_lin.view(B, T, self.n_head, self.head_dim)
                    k = k_lin.view(B, T, self.n_kv_head, self.head_dim)
                    v = v_lin.view(B, T, self.n_kv_head, self.head_dim)

                    if LIE_CTX.active and sm.use_diff_kv:
                        st = LIE_CTX.kv_state.get(cj)
                        inv_r = 1.0 / float(LIE_CTX.R)
                        if st is None:
                            LIE_CTX.kv_state[cj] = dict(k_bar=k, v_bar=v, k_prev=k, v_prev=v)
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
                            r_f = float(LIE_CTX.R)
                            k_eff = (1.0 - ak) * k + ak * k_bar_prev + dk * (r_f * (k - k_prev))
                            v_eff = (1.0 - av) * v + av * v_bar_prev + dv * (r_f * (v - v_prev))
                            LIE_CTX.kv_state[cj] = dict(k_bar=k_bar_new, v_bar=v_bar_new, k_prev=k, v_prev=v)
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
                    y = y.contiguous().view(B, T, -1)
                    y = self.c_proj(y)
                    if LIE_CTX.active and sm.use_lie and LIE_CTX.U_mats is not None:
                        y = apply_so_rot(y, LIE_CTX.U_mats[cj, 3], sm.k_sub, sm.d_sub)
                    return y
                return _attn_fwd
            blk.attn.forward = types.MethodType(_make_attn_fwd(core_j), blk.attn)

        if sm.use_mlp_lie or sm.use_relu2_32 or sm.use_givens_state:
            def _make_mlp_fwd(cj):
                def _mlp_fwd(self, x):
                    if LIE_CTX.active and sm.use_givens_state:
                        x = apply_givens_rot(x, sm.givens_mlp_in[cj])
                    elif LIE_CTX.active and sm.use_mlp_lie and LIE_CTX.U_mats is not None:
                        x = apply_so_rot(x, LIE_CTX.U_mats[cj, 4], sm.k_sub, sm.d_sub)
                    h = self.c_fc(x)
                    if LIE_CTX.active and sm.use_relu2_32 and LIE_CTX.relu2_gb is not None:
                        gam = LIE_CTX.relu2_gb[cj, 0].to(h.dtype)
                        bet = LIE_CTX.relu2_gb[cj, 1].to(h.dtype)
                        h = h * (1.0 + gam) + bet
                    h = F.relu(h).square()
                    y = self.c_proj(h)
                    # Isometric SO(k) left-rotation on MLP output (rotates write-back subspace in residual stream!)
                    if LIE_CTX.active and sm.use_mlp_lie and LIE_CTX.U_mats is not None:
                        y = apply_so_rot(y, LIE_CTX.U_mats[cj, 5], sm.k_sub, sm.d_sub)
                    return y
                return _mlp_fwd
            blk.mlp.forward = types.MethodType(_make_mlp_fwd(core_j), blk.mlp)

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
        LIE_CTX.active = False
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

        LIE_CTX.kv_state.clear()
        LIE_CTX.R = R_eff
        LIE_CTX.active = True
        try:
            for r in range(R_eff):
                no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
                if self.training and tbptt > 0 and r == R_eff - tbptt:
                    x = x.detach()
                    for k_st in list(LIE_CTX.kv_state.keys()):
                        LIE_CTX.kv_state[k_st] = {kk: vv.detach() for kk, vv in LIE_CTX.kv_state[k_st].items()}

                with torch.no_grad() if no_g else nullcontext():
                    LIE_CTX.r = r
                    LIE_CTX.tau = (float(r) + 0.5) * inv_r
                    if sm.use_lie:
                        cond = fourier_depth_embed(LIE_CTX.tau, 32, x.device)
                        LIE_CTX.U_mats = build_cayley_U(sm, cond)
                    if sm.use_relu2_32:
                        emb32 = fourier_depth_embed(LIE_CTX.tau, 32, x.device)
                        LIE_CTX.relu2_gb = sm.r2_net(emb32).view(sm.core_layers, 2, 4 * c.width)
                    if sm.use_adaln_fixed:
                        emb64 = fourier_depth_embed(LIE_CTX.tau, 64, x.device)
                        LIE_CTX.adaln_mod = sm.adaln_net(emb64).view(sm.core_layers, 6, c.width)

                    x_in = x
                    h_out = _run_group(self, x_in, x0, idx, r, total)
                    delta = h_out - x_in

                    if sm.use_anchor:
                        x = norm(x_in + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))
                    elif sm.use_euler_only:
                        x = x_in + (sm.s_ode * inv_r).to(x.dtype) * delta
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
            LIE_CTX.active = False
            LIE_CTX.kv_state.clear()
            LIE_CTX.U_mats = None
            LIE_CTX.relu2_gb = None
            LIE_CTX.adaln_mod = None

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
        "lieloop_attn_so12", "lieloop_full_so12", "lieloop_full_so24",
        "lieloop_so12_conj", "lieloop_givens_state", "lieloop_full_kv",
        "lieloop_full_r2_32", "lieloop_grand", "relu2_rank32_only",
        "adaln_anchor_fixed", "lieloop_r1_20L", "lieloop_no_anchor",
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

    base_params = sum(p.numel() for p in model.parameters())
    sm = attach_wave9_executor(model, args.idea, device=device)
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
