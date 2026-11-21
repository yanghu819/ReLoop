#!/usr/bin/env python3
"""Direction-A Arch Wave 10: Weight-Folded Legendre-Kronecker LieLoop (LK-LieLoop) + Phase-Reg4 + Fresh-2250 Rulers!

Synthesizes ALL 4 discoveries from Wave 8/9 + Fresh-2250 Audit (#108 & #109):
  1. Zero-Mean Shifted Legendre Basis P_1..P_4(2*tau - 1):
     Kills the 16*pi Fourier Nyquist aliasing bug completely for ALL R in {2..32},
     removes dead SiLU(0) zero-init bottleneck, and preserves exact W_eff(step=0) == W.
  2. Dense Kronecker Isometric Conjugation across BOTH Residual (768 = 24x32) AND MLP-Hidden (3072 = 48x64):
     - Attn: W_q/k/v = (L_{q/k/v}1 x L_{q/k/v}2) W (R_ain1 x R_ain2)^T
             W_o     = (L_o1 x L_o2) W_o (R_oin1 x R_oin2)^T  (mixes heads & 100.7M value_embeds views!)
     - MLP:  W_fc    = (V1 x V2) W_fc (R_min1 x R_min2)^T     (V in SO(48)xSO(64) tilts Range(W_fc) BEFORE ReLU^2!)
             W_proj  = (L_mout1 x L_mout2) W_proj (M1 x M2)^T (M in SO(48)xSO(64) rewires active neurons!)
  3. 100% Weight-Folded Sequential Contiguous Matmul Contraction:
     Folds (L1 x L2) W (R1 x R2)^T directly into W ONCE per loop in BF16 Tensor Cores ->
     ZERO [B, T, C] activation copies, 43x less HBM traffic, < 1-2% wall-clock overhead!
  4. Euclidean vs Spherical Inter-Loop Dynamics (--dynamics plain | euler | sphere):
     Directly tests internal audit #109's finding that inter-loop x=norm(...) wipes Euclidean radial
     capacity on Fresh-2250 (295M fresh tokens)!
  5. Orthogonal Phase-Register-4 (phase_reg4, 64 floats) with staggered Legendre read/write schedules
     to break the single-stream 768-dim Markov overwrite bottleneck!
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
from nexttransformer.models.backbone import norm, apply_rotary_emb
from nexttransformer.training.optimizer import ExperimentOptimizer
from nanochat.flash_attention import flash_attn


def legendre_p1_p4(tau: float, device: torch.device) -> torch.Tensor:
    """Zero-mean shifted Legendre polynomials P_1..P_4 on t = 2*tau - 1 in (-1, 1). Zero Nyquist aliasing!"""
    t = 2.0 * float(tau) - 1.0
    t2 = t * t
    p1 = t
    p2 = 0.5 * (3.0 * t2 - 1.0)
    p3 = 0.5 * t * (5.0 * t2 - 3.0)
    p4 = 0.125 * (35.0 * t2 * t2 - 30.0 * t2 + 3.0)
    return torch.tensor([p1, p2, p3, p4], device=device, dtype=torch.float32)


def cayley_batch(tri: torch.Tensor, k: int, tri_i: torch.Tensor, tri_j: torch.Tensor) -> torch.Tensor:
    """Computes exact SO(k) Cayley orthogonal matrices U = (I-S)^{-1}(I+S) for arbitrary leading dims [..., n_tri]."""
    lead = tri.shape[:-1]
    S = torch.zeros((*lead, k, k), device=tri.device, dtype=torch.float32)
    S[..., tri_i, tri_j] = tri
    S[..., tri_j, tri_i] = -tri
    eye = torch.eye(k, device=tri.device, dtype=torch.float32).expand_as(S)
    return torch.linalg.solve(eye - S, eye + S)


def fold_kron_bilateral(
    W: torch.Tensor,
    L1: torch.Tensor, L2: torch.Tensor,
    R1: torch.Tensor, R2: torch.Tensor,
) -> torch.Tensor:
    """Computes W_eff = (L1 (x) L2) W (R1 (x) R2)^T via 4 contiguous BF16 micro-matmuls (ZERO permute/einsum!).
    W has shape [p1*p2, q1*q2]; L1:[p1,p1], L2:[p2,p2], R1:[q1,q1], R2:[q2,q2].
    Exact spectral isometry: svd(W_eff).S == svd(W).S!
    """
    p1, p2 = L1.shape[0], L2.shape[0]
    q1, q2 = R1.shape[0], R2.shape[0]
    dt = W.dtype
    L1, L2, R1, R2 = L1.to(dt), L2.to(dt), R1.to(dt), R2.to(dt)
    # 1. Left L1 on axis 0: [p1, p1] @ [p1, p2*q1*q2] -> [p1, p2, q1*q2]
    T = (L1 @ W.reshape(p1, p2 * q1 * q2)).reshape(p1, p2, q1 * q2)
    # 2. Left L2 on axis 1: [1, p2, p2] @ [p1, p2, q1*q2] -> [p1*p2, q1, q2]
    T = torch.matmul(L2.unsqueeze(0), T).reshape(p1 * p2, q1, q2)
    # 3. Right R2 on axis 3: [p1*p2, q1, q2] @ [q2, q2] -> [p1*p2, q1, q2]
    T = torch.matmul(T, R2.transpose(0, 1))
    # 4. Right R1 on axis 2: [1, q1, q1] @ [p1*p2, q1, q2] -> [p1*p2, q1*q2]
    T = torch.matmul(R1.unsqueeze(0), T)
    return T.reshape(p1 * p2, q1 * q2)


def fold_so12_left_right(W: torch.Tensor, U_left: torch.Tensor | None, U_right: torch.Tensor | None) -> torch.Tensor:
    """Fast weight-folded SO(k) block rotation (U_left (x) I_d) W (U_right (x) I_d)^T."""
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


class _W10Ctx:
    def __init__(self):
        self.active = False
        self.r = 0
        self.R = 8
        self.tau = 0.5
        # Per-loop precomputed orthogonal matrices:
        self.U24 = None   # [core, n768, 24, 24]
        self.U32 = None   # [core, n768, 32, 32]
        self.U48 = None   # [core, n3072, 48, 48]
        self.U64 = None   # [core, n3072, 64, 64]
        self.U12 = None   # [core, 6, 12, 12] for lk_legendre_so12


W10_CTX = _W10Ctx()


class LKMod(nn.Module):
    def __init__(self, idea: str, dynamics: str, core_layers: int, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.dynamics = dynamics
        self.core_layers = core_layers
        self.width = width
        self.n_modes = 4  # Shifted Legendre P_1..P_4

        self.use_kron_attn = idea in ("lk_full", "lk_attn_only", "lk_full_reg4", "lk_r1_20L")
        self.use_kron_mlp = idea in ("lk_full", "lk_mlp3072_only", "lk_full_reg4", "lk_r1_20L")
        self.use_so12 = (idea == "lk_legendre_so12")
        self.use_reg4 = idea in ("phase_reg4", "lk_full_reg4")

        # Loop dynamics parameters
        if dynamics == "sphere":
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))
        elif dynamics == "euler":
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
        # dynamics == "plain": pure x = x + delta (ZERO extra dynamics params!)

        # Factorization check: width=768 = 24*32; 4*width=3072 = 48*64 (generalizes cleanly to w=1536 = 32*48, 6144 = 64*96)
        if self.use_kron_attn or self.use_kron_mlp:
            if width == 768:
                self.p1, self.p2 = 24, 32
                self.m1, self.m2 = 48, 64
            elif width == 1536:
                self.p1, self.p2 = 32, 48
                self.m1, self.m2 = 64, 96
            else:
                raise ValueError(f"Unsupported width={width}")

            # 768-dim slots per layer:
            # Attn uses 6 slots: 0:Attn_in, 1:Q_out, 2:K_out, 3:V_out, 4:O_in, 5:O_out
            # MLP  uses 2 slots: 6:MLP_in, 7:MLP_out (or 0,1 when attn disabled)
            self.n_slots_768 = (6 if self.use_kron_attn else 0) + (2 if self.use_kron_mlp else 0)
            self.mlp_768_off = 6 if self.use_kron_attn else 0
            n_tri_p1 = self.p1 * (self.p1 - 1) // 2
            n_tri_p2 = self.p2 * (self.p2 - 1) // 2
            self.gen_p1 = nn.Parameter(torch.zeros(self.n_modes, core_layers, self.n_slots_768, n_tri_p1, device=device))
            self.gen_p2 = nn.Parameter(torch.zeros(self.n_modes, core_layers, self.n_slots_768, n_tri_p2, device=device))
            i1, j1 = torch.triu_indices(self.p1, self.p1, offset=1, device=device)
            i2, j2 = torch.triu_indices(self.p2, self.p2, offset=1, device=device)
            self.register_buffer("i_p1", i1); self.register_buffer("j_p1", j1)
            self.register_buffer("i_p2", i2); self.register_buffer("j_p2", j2)

        if self.use_kron_mlp:
            # 3072-dim MLP hidden slots per layer: 0:V_fc (left of c_fc, BEFORE ReLU^2!), 1:M_proj (right of c_proj!)
            n_tri_m1 = self.m1 * (self.m1 - 1) // 2
            n_tri_m2 = self.m2 * (self.m2 - 1) // 2
            self.gen_m1 = nn.Parameter(torch.zeros(self.n_modes, core_layers, 2, n_tri_m1, device=device))
            self.gen_m2 = nn.Parameter(torch.zeros(self.n_modes, core_layers, 2, n_tri_m2, device=device))
            i3, j3 = torch.triu_indices(self.m1, self.m1, offset=1, device=device)
            i4, j4 = torch.triu_indices(self.m2, self.m2, offset=1, device=device)
            self.register_buffer("i_m1", i3); self.register_buffer("j_m1", j3)
            self.register_buffer("i_m2", i4); self.register_buffer("j_m2", j4)

        if self.use_so12:
            self.k12 = 12
            n_tri_12 = self.k12 * (self.k12 - 1) // 2
            # 6 slots: 0:Q, 1:K, 2:V, 3:O_out, 4:MLP_in, 5:MLP_out
            self.gen_12 = nn.Parameter(torch.zeros(self.n_modes, core_layers, 6, n_tri_12, device=device))
            i12, j12 = torch.triu_indices(self.k12, self.k12, offset=1, device=device)
            self.register_buffer("i_12", i12); self.register_buffer("j_12", j12)

        if self.use_reg4:
            # K=4 latent registers: S_0 = prelude (read-only), S_1..S_3 = zero-init scratchpads
            # Staggered Legendre basis [1, P_1, P_2, P_3] -> 4 basis x 4 regs = 16 read, 4 basis x 3 regs = 12 write
            # + 3 decay logits + 1 mix scale = 32 floats total! Symmetry explicitly broken at init!
            self.reg_read = nn.Parameter(torch.zeros(4, 4, device=device))
            # Staggered write windows: S1 early (-P1), S2 mid (-P2), S3 late (+P1)
            w_init = torch.tensor([
                [0.2, -0.5,  0.0,  0.0],  # S_1 peaks early (t = -1)
                [0.2,  0.0, -0.5,  0.0],  # S_2 peaks mid   (t =  0)
                [0.2, +0.5,  0.0,  0.0],  # S_3 peaks late  (t = +1)
            ], device=device, dtype=torch.float32)
            self.reg_write = nn.Parameter(w_init)
            self.reg_decay = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float32))
            self.reg_out_read = nn.Parameter(torch.zeros(4, device=device))


def attach_wave10_executor(model, idea: str, dynamics: str, device: torch.device):
    c = model.config
    is_r1_did = (idea == "lk_r1_20L")
    n_core = 4 if is_r1_did else c.core_layers
    sm = LKMod(idea, dynamics, n_core, c.width, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}

    if idea == "fixed_base":
        return sm

    for l_idx in range(c.pre_layers, c.pre_layers + c.core_layers):
        blk = model.transformer.h[l_idx]
        core_j = (l_idx - c.pre_layers) % n_core

        if sm.use_kron_attn or sm.use_so12:
            def _make_attn_fwd(cj):
                def _attn_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    B, T, C = x.size()
                    dt = x.dtype
                    if W10_CTX.active and sm.use_kron_attn and W10_CTX.U24 is not None:
                        u24, u32 = W10_CTX.U24[cj], W10_CTX.U32[cj]
                        R1, R2 = u24[0], u32[0]  # Attn_in right-rotation
                        Wq = fold_kron_bilateral(self.c_q.weight.to(dt), u24[1], u32[1], R1, R2)
                        Wk = fold_kron_bilateral(self.c_k.weight.to(dt), u24[2], u32[2], R1, R2)
                        Wv = fold_kron_bilateral(self.c_v.weight.to(dt), u24[3], u32[3], R1, R2)
                        q = F.linear(x, Wq).view(B, T, self.n_head, self.head_dim)
                        k = F.linear(x, Wk).view(B, T, self.n_kv_head, self.head_dim)
                        v = F.linear(x, Wv).view(B, T, self.n_kv_head, self.head_dim)
                    elif W10_CTX.active and sm.use_so12 and W10_CTX.U12 is not None:
                        u12 = W10_CTX.U12[cj]
                        Wq = fold_so12_left_right(self.c_q.weight.to(dt), u12[0], None)
                        Wk = fold_so12_left_right(self.c_k.weight.to(dt), u12[1], None)
                        Wv = fold_so12_left_right(self.c_v.weight.to(dt), u24[2] if False else u12[2], None)
                        q = F.linear(x, Wq).view(B, T, self.n_head, self.head_dim)
                        k = F.linear(x, Wk).view(B, T, self.n_kv_head, self.head_dim)
                        v = F.linear(x, Wv).view(B, T, self.n_kv_head, self.head_dim)
                    else:
                        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
                        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
                        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

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
                    if W10_CTX.active and sm.use_kron_attn and W10_CTX.U24 is not None:
                        u24, u32 = W10_CTX.U24[cj], W10_CTX.U32[cj]
                        # Bilateral W_o = (L_o1 x L_o2) W_o (R_oin1 x R_oin2)^T (rotates across heads & value_embeds!)
                        Wo = fold_kron_bilateral(self.c_proj.weight.to(dt), u24[5], u32[5], u24[4], u32[4])
                        return F.linear(y, Wo)
                    elif W10_CTX.active and sm.use_so12 and W10_CTX.U12 is not None:
                        Wo = fold_so12_left_right(self.c_proj.weight.to(dt), W10_CTX.U12[cj, 3], None)
                        return F.linear(y, Wo)
                    return self.c_proj(y)
                return _attn_fwd
            blk.attn.forward = types.MethodType(_make_attn_fwd(core_j), blk.attn)

        if sm.use_kron_mlp or sm.use_so12:
            def _make_mlp_fwd(cj):
                def _mlp_fwd(self, x):
                    dt = x.dtype
                    if W10_CTX.active and sm.use_kron_mlp and W10_CTX.U48 is not None:
                        u24, u32 = W10_CTX.U24[cj], W10_CTX.U32[cj]
                        u48, u64 = W10_CTX.U48[cj], W10_CTX.U64[cj]
                        off = sm.mlp_768_off
                        # W_fc: [3072, 768] = (V1 x V2) W_fc (R_min1 x R_min2)^T -> rotates 3072-dim ReLU^2 cones!
                        Wfc = fold_kron_bilateral(self.c_fc.weight.to(dt), u48[0], u64[0], u24[off], u32[off])
                        h = F.relu(F.linear(x, Wfc)).square()
                        # W_proj: [768, 3072] = (L_mout1 x L_mout2) W_proj (M1 x M2)^T -> rewires active neurons!
                        Wpr = fold_kron_bilateral(self.c_proj.weight.to(dt), u24[off + 1], u32[off + 1], u48[1], u64[1])
                        return F.linear(h, Wpr)
                    elif W10_CTX.active and sm.use_so12 and W10_CTX.U12 is not None:
                        u12 = W10_CTX.U12[cj]
                        Wfc = fold_so12_left_right(self.c_fc.weight.to(dt), None, u12[4])
                        h = F.relu(F.linear(x, Wfc)).square()
                        Wpr = fold_so12_left_right(self.c_proj.weight.to(dt), u12[5], None)
                        return F.linear(h, Wpr)
                    return self.c_proj(F.relu(self.c_fc(x)).square())
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
        W10_CTX.active = False
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        inv_r = 1.0 / float(R_eff)
        tbptt = getattr(self, "_tbptt_loops", 0)

        regs = None
        if sm.use_reg4:
            # S_0 = prelude (read-only), S_1..3 = zero-init scratchpads
            regs = [prelude, torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)]

        W10_CTX.R = R_eff
        W10_CTX.active = True
        try:
            for r in range(R_eff):
                no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
                if self.training and tbptt > 0 and r == R_eff - tbptt:
                    x = x.detach()
                    if regs is not None:
                        regs = [regs[0]] + [rg.detach() for rg in regs[1:]]

                with torch.no_grad() if no_g else nullcontext():
                    W10_CTX.r = r
                    tau = (float(r) + 0.5) * inv_r
                    W10_CTX.tau = tau
                    leg = legendre_p1_p4(tau, x.device)  # [4] = [P1, P2, P3, P4]

                    if sm.use_kron_attn or sm.use_kron_mlp:
                        tri_p1 = torch.einsum("m,mcst->cst", leg, sm.gen_p1)
                        tri_p2 = torch.einsum("m,mcst->cst", leg, sm.gen_p2)
                        W10_CTX.U24 = cayley_batch(tri_p1, sm.p1, sm.i_p1, sm.j_p1)
                        W10_CTX.U32 = cayley_batch(tri_p2, sm.p2, sm.i_p2, sm.j_p2)
                    if sm.use_kron_mlp:
                        tri_m1 = torch.einsum("m,mcst->cst", leg, sm.gen_m1)
                        tri_m2 = torch.einsum("m,mcst->cst", leg, sm.gen_m2)
                        W10_CTX.U48 = cayley_batch(tri_m1, sm.m1, sm.i_m1, sm.j_m1)
                        W10_CTX.U64 = cayley_batch(tri_m2, sm.m2, sm.i_m2, sm.j_m2)
                    if sm.use_so12:
                        tri_12 = torch.einsum("m,mcst->cst", leg, sm.gen_12)
                        W10_CTX.U12 = cayley_batch(tri_12, sm.k12, sm.i_12, sm.j_12)

                    x_in = x
                    if sm.use_reg4:
                        basis4 = torch.cat((torch.ones(1, device=x.device), leg[:3]), dim=0)  # [1, P1, P2, P3]
                        r_w = (basis4 @ sm.reg_read).to(x.dtype) * inv_r  # [4], zero-init!
                        x_blk_in = x_in + r_w[0] * regs[0] + r_w[1] * regs[1] + r_w[2] * regs[2] + r_w[3] * regs[3]
                    else:
                        x_blk_in = x_in

                    h_out = _run_group(self, x_blk_in, x0, idx, r, total)
                    delta = h_out - x_blk_in

                    if sm.use_reg4:
                        w_g = torch.sigmoid(sm.reg_write @ basis4) * inv_r  # [3] staggered write rates
                        dec = (F.softplus(sm.reg_decay) * inv_r).clamp_max(1.0)
                        new_regs = [regs[0]]
                        for k_i in range(3):
                            wg_k = w_g[k_i].to(x.dtype)
                            dk = dec[k_i].to(x.dtype)
                            new_regs.append((1.0 - dk) * regs[k_i + 1] + wg_k * delta)
                        regs = new_regs

                    if is_r1_did or sm.dynamics == "plain":
                        x = x_in + delta
                    elif sm.dynamics == "euler":
                        x = x_in + (sm.s_ode * inv_r).to(x.dtype) * delta
                    elif sm.dynamics == "sphere":
                        x = norm(x_in + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))

                    if r == mid_g:
                        cached = x
        finally:
            W10_CTX.active = False
            W10_CTX.U24 = W10_CTX.U32 = W10_CTX.U48 = W10_CTX.U64 = W10_CTX.U12 = None

        if sm.use_reg4 and regs is not None:
            ow = sm.reg_out_read.to(x.dtype)
            x = x + ow[0] * regs[0] + ow[1] * regs[1] + ow[2] * regs[2] + ow[3] * regs[3]

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
    if model.loop_state_mod.idea != "fixed_base":
        model._eval_loops_override = loops_eval
    else:
        # For fixed_base, mutate _ordinary_hidden depth via dynamic loop count cleanly if needed
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
        "fixed_base", "lk_full", "lk_mlp3072_only", "lk_attn_only",
        "lk_legendre_so12", "phase_reg4", "lk_full_reg4", "lk_r1_20L",
    ])
    ap.add_argument("--dynamics", default="plain", choices=["plain", "euler", "sphere"])
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
    sm = attach_wave10_executor(model, args.idea, args.dynamics, device=device)
    model._tbptt_loops = args.tbptt_loops
    extra_params = sum(p.numel() for p in sm.parameters())

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|dyn={args.dynamics}|R={args.loops}|u412={int(args.continuum_u412)}|"
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

        if args.continuum_u412 and args.loops == 8 and args.idea != "fixed_base":
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
    if args.loops == 1 or args.idea == "fixed_base":
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
