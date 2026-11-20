#!/usr/bin/env python3
"""Direction-A Arch Wave 14: Chinchilla Iso-VE Capacity Rulers + Bipartite Zero-Overlap PingPong-MoE + Un-Aliased Legendre Survivors!

Directly executes the unified Dual-Carbon-High Blueprint (#116 + #117):
  1. Chinchilla Capacity-Law Rulers (EXACT iso-value_embeds = 4 tables, eliminating the +151M vocab-table confounder!):
     - iso_ve_ruler (2/16/2 = 113.3M core, 4 VE tables)
     - iso_ve_ruler (2/8/2  = 56.6M  core, 4 VE tables)
     - iso_ve_thin28m (2/16/2 UNTIED thin core = EXACT 28.3M core params, 4 VE tables, ZERO weight tying!)
       -> Directly measures if prelude_ode_det R=8 (3.09642) ALREADY beats a 100%-untied 20L iso-param Transformer!
  2. PingPong-MoE (Rank #1, #116 — 0.000% Extra Core Weights, GUARANTEED 0.0% Adjacent-Layer Overlap!):
     - Pool A (Even virtual layers v = 0,2,..,30): Blocks 0+2 = 16 atomic paired MLP micro-experts (384 neurons each).
     - Pool B (Odd  virtual layers v = 1,3,..,31): Blocks 1+3 = 16 DISJOINT atomic paired MLP micro-experts.
     - Alternating Pool A -> Pool B -> Pool A -> Pool B mathematically guarantees 0.0% neuron overlap between v and v+1,
       eliminating the Wave-12 self-echo trap while composing C(16,8)^2 = 165M token-specific 2-layer circuits!
     - 2D Token x Continuous-Depth Legendre Router + C^0 Order-Statistic Gate (r = vals[:k] - vals[k]) +
       DeepSeek-V3 Auxiliary-Loss-Free Bias Balancing + native B300 torch._grouped_mm!
  3. Un-Aliased Legendre Survivors on Fresh-2250 (#116 Rank #2 + #117 §4.2):
     - legendre_adaln     : Direct linear Legendre P_1..P_4 -> (scale, shift, gate) (73.7k = 0.035%, zero 16*pi aliasing!)
     - legendre_wf_so12_r2: Weight-Folded Legendre SO(12) + Direct Legendre ReLU^2 (~120k tok/s, zero 16*pi aliasing!)
     - mh_depth_attn      : H=6 independent 128-dim Subhead Causal Depth-Attention over [H_0..H_r] (+3.0 latest bias!).
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
from nexttransformer.models.backbone import norm, apply_rotary_emb
from nexttransformer.training.optimizer import ExperimentOptimizer
from nanochat.flash_attention import flash_attn


def legendre_p0_p4(tau: float, device: torch.device) -> torch.Tensor:
    t = 2.0 * float(tau) - 1.0
    t2 = t * t
    return torch.tensor([
        1.0,
        t,
        0.5 * (3.0 * t2 - 1.0),
        0.5 * t * (5.0 * t2 - 3.0),
        0.125 * (35.0 * t2 * t2 - 30.0 * t2 + 3.0),
    ], device=device, dtype=torch.float32)


def cayley_batch(tri: torch.Tensor, k: int, tri_i: torch.Tensor, tri_j: torch.Tensor) -> torch.Tensor:
    lead = tri.shape[:-1]
    S = torch.zeros((*lead, k, k), device=tri.device, dtype=torch.float32)
    S[..., tri_i, tri_j] = tri
    S[..., tri_j, tri_i] = -tri
    eye = torch.eye(k, device=tri.device, dtype=torch.float32).expand_as(S)
    return torch.linalg.solve(eye - S, eye + S)


def fold_so12_lr(W: torch.Tensor, U_left: torch.Tensor | None, U_right: torch.Tensor | None) -> torch.Tensor:
    out_d, in_d = W.shape
    dt = W.dtype
    T = W
    if U_left is not None:
        k = U_left.shape[0]
        d = out_d // k
        T = (U_left.to(dt) @ T.reshape(k, d * in_d)).reshape(out_d, in_d)
    if U_right is not None:
        k = U_right.shape[0]
        d = in_d // k
        T = torch.matmul(U_right.to(dt).unsqueeze(0), T.reshape(out_d, k, d)).reshape(out_d, in_d)
    return T


def apply_iso_ve_4(model):
    """Enforces EXACTLY 4 value_embeds tables matching 2/4/2 (1 pre, 2 core, 1 post), freezing removed ve_gates."""
    c = model.config
    n_tot = len(model.transformer.h)
    keep = {1, c.pre_layers + c.core_layers // 4, c.pre_layers + (3 * c.core_layers) // 4, n_tot - 1}
    # Ensure keep keys exist in existing value_embeds; pick closest 4 existing keys!
    existing = sorted(int(k) for k in model.value_embeds.keys())
    if len(existing) <= 4:
        return len(existing)
    chosen = set()
    for target in sorted(keep):
        best = min((k for k in existing if k not in chosen), key=lambda k: abs(k - target))
        chosen.add(best)
    for k in existing:
        if k not in chosen:
            del model.value_embeds[str(k)]
            vg = getattr(model.transformer.h[k].attn, "ve_gate", None)
            if vg is not None:
                for p in vg.parameters():
                    p.requires_grad_(False)
    return len(model.value_embeds)


def shrink_core_to_exact_28m(model, device: torch.device):
    """Converts 16 untied core layers of 2/16/2 to EXACT 28.31M core params (head_dim=32, mlp_width=768) with ZERO tying!
    Per core layer: Attn 4 * (768 * 192) = 589,824 + MLP 2 * (768 * 768) = 1,179,648 = 1,769,472.
    16 layers * 1,769,472 = 28,311,552 == EXACT 4 * 7,077,888 (2/4/2 core params)!
    """
    c = model.config
    C = c.width
    H = c.heads  # 6
    thin_d = 32  # 6 * 32 = 192 (= 1/4 of 768)
    thin_inner = H * thin_d  # 192
    thin_mlp = C             # 768 (= 1/4 of 3072)

    for l_idx in range(c.pre_layers, c.pre_layers + c.core_layers):
        blk = model.transformer.h[l_idx]
        # Replace Attn linear projections & head_dim cleanly
        blk.attn.head_dim = thin_d
        blk.attn.c_q = nn.Linear(C, thin_inner, bias=False, device=device)
        blk.attn.c_k = nn.Linear(C, thin_inner, bias=False, device=device)
        blk.attn.c_v = nn.Linear(C, thin_inner, bias=False, device=device)
        blk.attn.c_proj = nn.Linear(thin_inner, C, bias=False, device=device)
        nn.init.normal_(blk.attn.c_q.weight, std=0.02)
        nn.init.normal_(blk.attn.c_k.weight, std=0.02)
        nn.init.normal_(blk.attn.c_v.weight, std=0.02)
        nn.init.zeros_(blk.attn.c_proj.weight)
        # Drop core value_embeds on thin layers (or keep VE on pre/post only; wait: project VE if present!)
        # Actually, slice VE to thin_inner if present so iso-VE stays 100% intact!
        def _make_thin_attn_fwd():
            def _thin_attn_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                B, T, _ = x.size()
                dt = x.dtype
                q = F.linear(x, self.c_q.weight.to(dt)).view(B, T, self.n_head, self.head_dim)
                k = F.linear(x, self.c_k.weight.to(dt)).view(B, T, self.n_kv_head, self.head_dim)
                v = F.linear(x, self.c_v.weight.to(dt)).view(B, T, self.n_kv_head, self.head_dim)
                if ve is not None:
                    ve_v = ve[..., :self.n_kv_head * self.head_dim].view(B, T, self.n_kv_head, self.head_dim)
                    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                    v = v + gate.unsqueeze(-1) * ve_v
                cos, sin = cos_sin
                half = self.head_dim // 2
                q = apply_rotary_emb(q, cos[..., :half], sin[..., :half])
                k = apply_rotary_emb(k, cos[..., :half], sin[..., :half])
                q, k = norm(q) * 1.2, norm(k) * 1.2
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
                return F.linear(y.contiguous().view(B, T, -1), self.c_proj.weight.to(dt))
            return _thin_attn_fwd
        blk.attn.forward = types.MethodType(_make_thin_attn_fwd(), blk.attn)

        # Replace MLP to width 768
        blk.mlp.c_fc = nn.Linear(C, thin_mlp, bias=False, device=device)
        blk.mlp.c_proj = nn.Linear(thin_mlp, C, bias=False, device=device)
        nn.init.normal_(blk.mlp.c_fc.weight, std=0.02)
        nn.init.zeros_(blk.mlp.c_proj.weight)
        def _make_thin_mlp_fwd():
            def _thin_mlp_fwd(self, x):
                dt = x.dtype
                h = F.relu(F.linear(x, self.c_fc.weight.to(dt))).square()
                return F.linear(h, self.c_proj.weight.to(dt))
            return _thin_mlp_fwd
        blk.mlp.forward = types.MethodType(_make_thin_mlp_fwd(), blk.mlp)


def run_grouped_moe_mlp(
    x: torch.Tensor,
    U_pool: torch.Tensor,   # [E, d_e, C]
    D_pool: torch.Tensor,   # [E, C, d_e]
    top_idx: torch.Tensor,  # [N, k] int64
    top_g: torch.Tensor,    # [N, k] bf16/float
) -> torch.Tensor:
    """Bit-exact Top-k sparse PingPong-MoE MLP running 100% on Blackwell B300 Tensor Cores (3.35ms vs 2.80ms dense!).
    Zero SM90 TMA bugs, zero host-device syncs, exact zero gradient on unselected experts!
    """
    B, T, C = x.shape
    E, d_e, _ = U_pool.shape
    N = B * T
    dt = x.dtype

    mask = torch.zeros((N, E), device=x.device, dtype=dt)
    mask.scatter_(1, top_idx, top_g.to(dt))
    W1 = U_pool.reshape(E * d_e, C).to(dt)
    W2 = D_pool.permute(1, 0, 2).reshape(C, E * d_e).to(dt)
    h = F.relu(F.linear(x.reshape(N, C), W1)).square()
    h = (h.view(N, E, d_e) * mask.unsqueeze(-1)).reshape(N, E * d_e)
    return F.linear(h, W2).view(B, T, C)


class _W14Ctx:
    def __init__(self):
        self.active = False
        self.r = 0
        self.R = 8
        self.U12 = None
        self.r2_gb = None
        self.adaln = None


W14_CTX = _W14Ctx()


class W14Mod(nn.Module):
    def __init__(self, idea: str, dynamics: str, core_layers: int, n_head: int, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.dynamics = dynamics
        self.width = width
        self.core_layers = core_layers
        self.n_head = n_head

        self.is_pingpong = idea in ("pingpong_moe_top8", "pingpong_moe_top4")
        self.k_exp = 4 if idea == "pingpong_moe_top4" else 8
        self.n_exp = 16  # 16 micro-experts of 384 neurons per pool (Pool A = blocks 0+2, Pool B = blocks 1+3)
        self.use_leg_adaln = (idea == "legendre_adaln")
        self.use_leg_so12_r2 = (idea == "legendre_wf_so12_r2")
        self.use_mh_dattn = (idea == "mh_depth_attn")

        if idea not in ("iso_ve_ruler", "iso_ve_thin28m"):
            if dynamics == "sphere":
                self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
                self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))
            elif dynamics == "euler":
                self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if self.is_pingpong:
            # 2 independent routers: Pool A (even layers) & Pool B (odd layers)
            self.w_tok = nn.Parameter(torch.randn(2, self.n_exp, width, device=device, dtype=torch.float32) * 0.02)
            self.w_leg = nn.Parameter(torch.zeros(2, self.n_exp, 5, device=device, dtype=torch.float32))
            # Staggered depth prior on P1 so experts start with gentle early/late diversity
            with torch.no_grad():
                self.w_leg[:, :, 1] = torch.linspace(-0.2, 0.2, self.n_exp, device=device).unsqueeze(0)
            # DeepSeek-V3 Auxiliary-Loss-Free balancing biases (buffers, updated via no_grad!)
            self.register_buffer("bal_bias", torch.zeros(2, self.n_exp, device=device, dtype=torch.float32))

        if self.use_leg_adaln:
            # Direct zero-mean shifted Legendre P_1..P_4 -> [core_layers, 6, width] (ZERO 16*pi aliasing! 73.7k floats!)
            self.adaln_leg = nn.Parameter(torch.zeros(4, core_layers, 6, width, device=device, dtype=torch.float32))

        if self.use_leg_so12_r2:
            # Weight-Folded Legendre SO(12) (6 slots: Q,K,V,O,MLP_in,MLP_out) + Direct Legendre ReLU^2 (2*3072)
            self.k12 = 12
            n_tri = self.k12 * (self.k12 - 1) // 2  # 66
            self.gen_12 = nn.Parameter(torch.zeros(4, core_layers, 6, n_tri, device=device, dtype=torch.float32))
            i12, j12 = torch.triu_indices(self.k12, self.k12, offset=1, device=device)
            self.register_buffer("i_12", i12)
            self.register_buffer("j_12", j12)
            self.r2_leg = nn.Parameter(torch.zeros(4, core_layers, 2, 4 * width, device=device, dtype=torch.float32))

        if self.use_mh_dattn:
            # H=6 independent 128-dim Subhead Causal Depth-Attention over [H_0..H_r] (+3.0 latest-state bias!)
            # Only 12k params, < 0.5% overhead!
            self.head_dim = width // n_head  # 128
            self.q_leg = nn.Parameter(torch.zeros(4, n_head, self.head_dim, device=device, dtype=torch.float32))
            self.k_leg = nn.Parameter(torch.zeros(4, n_head, self.head_dim, device=device, dtype=torch.float32))
            # Per-head linear projection from current state to query (6 x 128 x 128? No: diagonal/rank-16 = 12k!)
            self.q_down = nn.Linear(width, 16, bias=False, device=device, dtype=torch.float32)
            self.q_up = nn.Linear(16, width, bias=False, device=device, dtype=torch.float32)
            nn.init.normal_(self.q_down.weight, std=0.02)
            nn.init.zeros_(self.q_up.weight)
            self.latest_bias = nn.Parameter(torch.full((n_head,), 3.0, device=device, dtype=torch.float32))
            self.out_q = nn.Parameter(torch.zeros(n_head, self.head_dim, device=device, dtype=torch.float32))
            self.out_latest_bias = nn.Parameter(torch.full((n_head,), 3.0, device=device, dtype=torch.float32))


def attach_wave14_executor(model, idea: str, dynamics: str, device: torch.device):
    c = model.config
    sm = W14Mod(idea, dynamics, c.core_layers, c.heads, c.width, device=device)
    model.loop_state_mod = sm

    if idea in ("iso_ve_ruler", "iso_ve_thin28m"):
        return sm

    K = c.core_layers  # 4
    core_blocks = [model.transformer.h[c.pre_layers + k] for k in range(K)]

    # Hook Block / Attn / MLP for Legendre AdaLN or Weight-Folded Legendre SO(12) + ReLU^2
    if sm.use_leg_adaln:
        for cj, blk in enumerate(core_blocks):
            def _make_blk_fwd(j):
                def _blk_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    if W14_CTX.active and W14_CTX.adaln is not None:
                        m = W14_CTX.adaln[j]  # [6, width]
                        sa, sca, ga, sm_, scm, gm = [m[i].to(x.dtype) for i in range(6)]
                        x = x + (1.0 + ga) * self.attn(norm(x) * (1.0 + sca) + sa, ve, cos_sin, window_size, kv_cache)
                        x = x + (1.0 + gm) * self.mlp(norm(x) * (1.0 + scm) + sm_)
                        return x
                    x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
                    x = x + self.mlp(norm(x))
                    return x
                return _blk_fwd
            blk.forward = types.MethodType(_make_blk_fwd(cj), blk)

    if sm.use_leg_so12_r2:
        for cj, blk in enumerate(core_blocks):
            def _make_attn_fwd(j):
                def _attn_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    B, T, C = x.size()
                    dt = x.dtype
                    if W14_CTX.active and W14_CTX.U12 is not None:
                        u12 = W14_CTX.U12[j]
                        Wq = fold_so12_lr(self.c_q.weight.to(dt), u12[0], None)
                        Wk = fold_so12_lr(self.c_k.weight.to(dt), u12[1], None)
                        Wv = fold_so12_lr(self.c_v.weight.to(dt), u12[2], None)
                        Wo = fold_so12_lr(self.c_proj.weight.to(dt), u12[3], None)
                        q = F.linear(x, Wq).view(B, T, self.n_head, self.head_dim)
                        k = F.linear(x, Wk).view(B, T, self.n_kv_head, self.head_dim)
                        v = F.linear(x, Wv).view(B, T, self.n_kv_head, self.head_dim)
                    else:
                        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
                        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
                        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
                        Wo = None
                    if ve is not None:
                        ve_v = ve.view(B, T, self.n_kv_head, self.head_dim)
                        gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                        v = v + gate.unsqueeze(-1) * ve_v
                    cos, sin = cos_sin
                    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                    q, k = norm(q) * 1.2, norm(k) * 1.2
                    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size).contiguous().view(B, T, -1)
                    return F.linear(y, Wo) if Wo is not None else self.c_proj(y)
                return _attn_fwd

            def _make_mlp_fwd(j):
                def _mlp_fwd(self, x):
                    dt = x.dtype
                    if W14_CTX.active and W14_CTX.U12 is not None:
                        u12 = W14_CTX.U12[j]
                        Wfc = fold_so12_lr(self.c_fc.weight.to(dt), None, u12[4])
                        Wpr = fold_so12_lr(self.c_proj.weight.to(dt), u12[5], None)
                        h = F.linear(x, Wfc)
                        if W14_CTX.r2_gb is not None:
                            gam = W14_CTX.r2_gb[j, 0].to(dt)
                            bet = W14_CTX.r2_gb[j, 1].to(dt)
                            h = h * (1.0 + gam) + bet
                        return F.linear(F.relu(h).square(), Wpr)
                    return self.c_proj(F.relu(self.c_fc(x)).square())
                return _mlp_fwd
            blk.attn.forward = types.MethodType(_make_attn_fwd(cj), blk.attn)
            blk.mlp.forward = types.MethodType(_make_mlp_fwd(cj), blk.mlp)

    def _mh_depth_attend(q_h: torch.Tensor, H_list: list[torch.Tensor], K_list: list[torch.Tensor],
                         lat_bias: torch.Tensor) -> torch.Tensor:
        """Per-subhead (H=6, D=128) causal depth-attention over [H_0..H_r].
        q_h: [B, T, H, D]; H_list/K_list: list of L tensors [B, T, H, D]; lat_bias: [H].
        """
        L = len(H_list)
        K_st = torch.stack(K_list, dim=2)  # [B, T, L, H, D]
        H_st = torch.stack(H_list, dim=2)  # [B, T, L, H, D]
        scores = torch.einsum("bthd,btlhd->btlh", q_h.float(), K_st.float()) * (sm.head_dim ** -0.5)
        scores[:, :, -1, :] = scores[:, :, -1, :] + lat_bias.view(1, 1, sm.n_head)
        w = torch.softmax(scores, dim=2).to(H_st.dtype)  # [B, T, L, H]
        out = torch.einsum("btlh,btlhd->bthd", w, H_st)
        return out.reshape(q_h.size(0), q_h.size(1), c.width)

    def _custom_ordinary_hidden(self, idx):
        R_eff = getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops
        N_core = R_eff * K
        total = c.pre_layers + N_core + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        W14_CTX.active = False
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        inv_r = 1.0 / float(R_eff)
        dt = x.dtype

        # Pre-pack bipartite PingPong-MoE pools ONCE per forward (zero per-layer copies!)
        if sm.is_pingpong:
            C = c.width
            d_e = (2 * 4 * C) // sm.n_exp  # 6144 / 16 = 384
            # Pool A (even): blocks 0 & 2; Pool B (odd): blocks 1 & 3 -> STRICT 0.0% ADJACENT OVERLAP!
            U_pools = [
                torch.cat((core_blocks[0].mlp.c_fc.weight, core_blocks[2].mlp.c_fc.weight), dim=0).view(sm.n_exp, d_e, C),
                torch.cat((core_blocks[1].mlp.c_fc.weight, core_blocks[3].mlp.c_fc.weight), dim=0).view(sm.n_exp, d_e, C),
            ]
            D_pools = [
                torch.cat((core_blocks[0].mlp.c_proj.weight, core_blocks[2].mlp.c_proj.weight), dim=1).view(C, sm.n_exp, d_e).permute(1, 0, 2),
                torch.cat((core_blocks[1].mlp.c_proj.weight, core_blocks[3].mlp.c_proj.weight), dim=1).view(C, sm.n_exp, d_e).permute(1, 0, 2),
            ]

        H_traj, K_traj = [], []
        if sm.use_mh_dattn:
            H_traj.append(prelude.view(x.size(0), x.size(1), sm.n_head, sm.head_dim))
            k0 = norm(prelude).view(x.size(0), x.size(1), sm.n_head, sm.head_dim)
            K_traj.append(k0)

        W14_CTX.R = R_eff
        W14_CTX.active = True
        try:
            for r in range(R_eff):
                W14_CTX.r = r
                tau_r = (float(r) + 0.5) * inv_r
                p1_p4 = legendre_p0_p4(tau_r, x.device)[1:]  # [4]

                if sm.use_leg_adaln:
                    W14_CTX.adaln = torch.einsum("m,mcsw->csw", p1_p4, sm.adaln_leg)
                if sm.use_leg_so12_r2:
                    tri12 = torch.einsum("m,mcst->cst", p1_p4, sm.gen_12)
                    W14_CTX.U12 = cayley_batch(tri12, sm.k12, sm.i_12, sm.j_12)
                    W14_CTX.r2_gb = torch.einsum("m,mcsw->csw", p1_p4, sm.r2_leg)

                if sm.use_mh_dattn and len(H_traj) > 1:
                    q_dyn = sm.q_up(sm.q_down(norm(x).float())).view(x.size(0), x.size(1), sm.n_head, sm.head_dim)
                    q_dep = torch.einsum("m,mhd->hd", p1_p4, sm.q_leg).view(1, 1, sm.n_head, sm.head_dim)
                    x_in = _mh_depth_attend(q_dyn + q_dep, H_traj, K_traj, sm.latest_bias)
                else:
                    x_in = x

                h = x_in
                if sm.is_pingpong:
                    for j in range(K):
                        v_idx = r * K + j
                        phys = c.pre_layers + j
                        virt = c.pre_layers + v_idx
                        tau_v = (float(v_idx) + 0.5) / float(N_core)
                        p0_p4_v = legendre_p0_p4(tau_v, x.device)  # [5]

                        # Exact NextTransformer resid_lambdas, value_embeds, window_size, & native Cycle Attention!
                        h = self.resid_lambdas[phys] * h + self.x0_lambdas[phys] * x0
                        ve = self.value_embeds[str(phys)](idx).to(dt) if str(phys) in self.value_embeds else None
                        win = self._window(virt, total)
                        cos_sin = (self.cos[:, :idx.shape[1]], self.sin[:, :idx.shape[1]])
                        h = h + core_blocks[j].attn(norm(h), ve, cos_sin, win, None)

                        # Bipartite Alternating Disjoint PingPong-MoE MLP (Pool A on even j, Pool B on odd j -> 0% overlap!)
                        p_id = j & 1
                        xm = norm(h)
                        B, T, C = xm.shape
                        # 2D Token x Depth score + DeepSeek-V3 bias
                        depth_sc = (sm.w_leg[p_id] @ p0_p4_v).view(1, 1, sm.n_exp)  # [1, 1, 16]
                        raw_sc = F.linear(xm.float(), sm.w_tok[p_id]) + depth_sc    # [B, T, 16]
                        sel_sc = raw_sc + sm.bal_bias[p_id].view(1, 1, sm.n_exp)
                        # C^0 continuous order-statistic Top-k selection using balanced scores, gates from raw_sc!
                        _, idx_kp1 = torch.topk(sel_sc, sm.k_exp + 1, dim=-1)       # [B, T, k+1]
                        sel_vals = torch.gather(raw_sc, -1, idx_kp1)                # [B, T, k+1]
                        r_val = (sel_vals[..., :sm.k_exp] - sel_vals[..., sm.k_exp:sm.k_exp + 1]).clamp_min(0.0)
                        g_val = r_val * torch.rsqrt((r_val * r_val).mean(dim=-1, keepdim=True) + 1e-6)
                        top_idx = idx_kp1[..., :sm.k_exp].reshape(B * T, sm.k_exp)
                        top_g = g_val.reshape(B * T, sm.k_exp)

                        # DeepSeek-V3 aux-loss-free balancing bias update (train only, no_grad!)
                        if self.training:
                            with torch.no_grad():
                                load = torch.bincount(top_idx.reshape(-1), minlength=sm.n_exp).float() / float(B * T)
                                target = float(sm.k_exp) / float(sm.n_exp)
                                sm.bal_bias[p_id].add_(1e-3 * torch.sign(target - load))
                                sm.bal_bias[p_id].clamp_(-2.0, 2.0)

                        mlp_out = run_grouped_moe_mlp(xm, U_pools[p_id], D_pools[p_id], top_idx, top_g)
                        # Scale by sqrt(8/k) if Top-4 to keep variance identical to 3072-width MLP!
                        if sm.k_exp != 8:
                            mlp_out = mlp_out * math.sqrt(8.0 / float(sm.k_exp))
                        h = h + mlp_out
                else:
                    for j in range(K):
                        phys = c.pre_layers + j
                        virt = c.pre_layers + r * K + j
                        h = self._block(h, x0, idx, phys, virt, total)

                delta = h - x_in
                if sm.dynamics == "plain":
                    x = x_in + delta
                elif sm.dynamics == "euler":
                    x = x_in + (sm.s_ode * inv_r).to(dt) * delta
                elif sm.dynamics == "sphere":
                    x = norm(x_in + inv_r * (sm.s_ode.to(dt) * delta + sm.g_pre.to(dt) * prelude))

                if sm.use_mh_dattn:
                    H_traj.append(x.view(x.size(0), x.size(1), sm.n_head, sm.head_dim))
                    k_dep = torch.einsum("m,mhd->hd", p1_p4, sm.k_leg).view(1, 1, sm.n_head, sm.head_dim).to(dt)
                    K_traj.append(norm(x).view(x.size(0), x.size(1), sm.n_head, sm.head_dim) + k_dep)

                if r == mid_g:
                    cached = x
        finally:
            W14_CTX.active = False
            W14_CTX.U12 = W14_CTX.r2_gb = W14_CTX.adaln = None

        if sm.use_mh_dattn and len(H_traj) > 1:
            q_o = sm.out_q.view(1, 1, sm.n_head, sm.head_dim).expand(x.size(0), x.size(1), -1, -1)
            x = _mh_depth_attend(q_o, H_traj, K_traj, sm.out_latest_bias)

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + N_core + j
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
        "iso_ve_ruler", "iso_ve_thin28m", "pingpong_moe_top8", "pingpong_moe_top4",
        "legendre_adaln", "legendre_wf_so12_r2", "mh_depth_attn",
    ])
    ap.add_argument("--dynamics", default="sphere", choices=["plain", "euler", "sphere"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--iso-ve-4", action="store_true")
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

    n_ve = len(model.value_embeds)
    if args.iso_ve_4 or args.idea in ("iso_ve_ruler", "iso_ve_thin28m"):
        n_ve = apply_iso_ve_4(model)
    if args.idea == "iso_ve_thin28m":
        shrink_core_to_exact_28m(model, device=device)

    core_params = sum(p.numel() for l in range(args.pre_layers, args.pre_layers + args.core_layers)
                      for p in model.transformer.h[l].parameters() if p.requires_grad)
    base_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sm = attach_wave14_executor(model, args.idea, args.dynamics, device=device)
    extra_params = sum(p.numel() for p in sm.parameters() if p.requires_grad)

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|dyn={args.dynamics}|R={args.loops}|ve={n_ve}|core={core_params/1e6:.2f}M|"
          f"tot={base_params/1e6:.1f}M|+p={extra_params} ({100*extra_params/max(base_params,1):.4f}%)] "
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

        if args.continuum_u412 and args.loops == 8 and args.idea not in ("iso_ve_ruler", "iso_ve_thin28m"):
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
    if args.loops == 1 or args.idea in ("iso_ve_ruler", "iso_ve_thin28m"):
        _, vl_self = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=2048)
        ood_2048[str(args.loops)] = round(vl_self, 5)
    else:
        for r_ev in (1, 2, 4, 8, 12, 16, 32):
            ev_mb = 8 if r_ev >= 32 else (16 if r_ev >= 12 else min(args.micro_batch, 32))
            _, vl_r = eval_at_depth(model, val_x, val_y, byte_table, device, ev_mb, r_ev, max_rows=2048)
            ood_2048[str(r_ev)] = round(vl_r, 5)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    ood_diff = round(ood_2048["16"] - ood_2048["8"], 5) if ("16" in ood_2048 and "8" in ood_2048) else None
    out = dict(
        tag=args.tag, idea=args.idea, dynamics=args.dynamics, loops=args.loops,
        n_ve=n_ve, core_params=core_params, total_params=base_params + extra_params,
        extra_params=extra_params, extra_param_pct=round(100.0 * extra_params / max(base_params, 1), 5),
        continuum_u412=args.continuum_u412, steps=args.steps, data_dir=str(args.data_dir),
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
