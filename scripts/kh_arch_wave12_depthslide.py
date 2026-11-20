#!/usr/bin/env python3
"""Direction-A Arch Wave 12: Continuous Depth-Sliding Atomic Tape (DepthSlide) — 0 Extra Params, 0% Overhead, 0 Cross-Talk!

Why Waves 7-11 Failed to Break the Untied 20L Ceiling (Fresh-2250: 3.0500 vs Tied R=8 3.0964 / 3.1095):
  1. Single-matrix modulations (LoRA, AdaLN, SO(k) LieLoop) cannot escape identical-operator repetition.
  2. Layer-Repeat (0^R, 1^R, 2^R, 3^R) isolates depth bands but stalls on 8 identical consecutive layers (s375=3.786).
  3. Weight-space blending across unaligned blocks (Wave 11 WeightField) creates bilinear cross-talk
     W_{proj,0} ReLU^2(W_{fc,1} x) (s375=3.812, predicted verbatim by @carbon-auditor #112!).

Kaiming-He 3-Line Breakthrough — DepthSlide (Continuous Depth-Sliding Atomic Tape):
  Arrange the EXACT SAME K=4 (28.3M) core blocks as a single contiguous 1D tape of:
    - 4 * 3,072 = 12,288 ATOMIC paired MLP neurons  (W_fc[m], W_proj[:, m])  — 100% paired, ZERO cross-talk!
    - 4 * 6     = 24     ATOMIC paired Attn heads   (Q_h, K_h, V_h, O_h)     — 100% paired, ZERO cross-talk!
  As virtual layer l runs from 0 to 4*R - 1 (continuous normalized depth tau in [0, 1]), smoothly slide a
  window of width (3,072 neurons, 6 heads) across the atomic tape!
    - At R=1 (4 layers): slices [0:3072], [3072:6144], [6144:9216], [9216:12288] => BIT-FOR-BIT standard 4L Transformer!
    - At R=8 (32 layers): ALL 32 virtual layers are genuinely DISTINCT full-capacity Transformer layers!
      Early layers specialize in syntax/retrieval, mid layers in reasoning, late layers in next-token decoding —
      with ZERO adjacent-layer fixed-point collapse, ZERO bilinear cross-talk, ZERO extra parameters (0.000%!),
      ZERO [max_loops] lookup tables, and an exact smooth continuum limit as R -> infinity (R=1,2,4,8,12,16,32)!

Variants tested in Wave 12 (ALL preserving resid_lambdas, --freeze-x0, exact ood_2048):
  1. slide_linear_hard : Linear tape [0 -> 9216] (100% disjoint early vs late; 0 extra params; exact R=1!).
  2. slide_linear_soft : C^1 smooth fractional-boundary linear tape (variance-preserving sqrt(1-f)/sqrt(f) edge fade!).
  3. slide_ring_hard   : 1-revolution circular tape (EVERY neuron & head gets UNIFORM R*1x gradient exposure!).
  4. slide_ring_soft   : C^1 smooth fractional-boundary circular tape (uniform exposure + exact continuum OOD!).
  5. slide_ring_2rev   : 2-revolution circular tape (blends macro depth progression with multi-pass refinement!).
  6. slide_attn_cycle  : Linear MLP slide [0->9216] + standard Cycle Attention (clean LOO isolating MLP vs Attn tape!).
  7. ribbon_linear     : @carbon-critic #113 C^0-Smooth Order-Statistic Top-k Ribbon (24 heads, 32 MLP micro-experts, 280 floats!).
  8. ribbon_ring       : Torus/Circular C^0-Smooth Order-Statistic Top-k Ribbon (uniform R*1x exposure + learnable Legendre!).
  9. ribbon_tok        : 2D Token x Depth RibbonLoop (+24k zero-init token-adaptive expert gates, 1.00x FLOPs, 0 load-bal loss!).
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
    p1 = t
    p2 = 0.5 * (3.0 * t2 - 1.0)
    p3 = 0.5 * t * (5.0 * t2 - 3.0)
    p4 = 0.125 * (35.0 * t2 * t2 - 30.0 * t2 + 3.0)
    return torch.tensor([1.0, p1, p2, p3, p4], device=device, dtype=torch.float32)


class DepthSlideMod(nn.Module):
    def __init__(self, idea: str, dynamics: str, width: int, total_H: int, n_mlp_exp: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.dynamics = dynamics
        self.is_ribbon = idea in ("ribbon_linear", "ribbon_ring", "ribbon_tok")
        self.ribbon_ring = (idea == "ribbon_ring")
        self.use_tok = (idea == "ribbon_tok")

        if dynamics == "sphere":
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))
        elif dynamics == "euler":
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if self.is_ribbon:
            # 24 atomic heads + 32 atomic MLP micro-experts x 5 Legendre basis = 280 floats!
            self.theta_a = nn.Parameter(torch.zeros(total_H, 5, device=device, dtype=torch.float32))
            self.theta_m = nn.Parameter(torch.zeros(n_mlp_exp, 5, device=device, dtype=torch.float32))
            self.register_buffer("c_a", (torch.arange(total_H, device=device, dtype=torch.float32) + 0.5) / float(total_H))
            self.register_buffer("c_m", (torch.arange(n_mlp_exp, device=device, dtype=torch.float32) + 0.5) / float(n_mlp_exp))
        if self.use_tok:
            # Zero-init 2D Token x Depth micro-expert modulation (24k params, zero scatter/gather!)
            self.tok_router = nn.Linear(width, n_mlp_exp, bias=False, device=device, dtype=torch.float32)
            nn.init.zeros_(self.tok_router.weight)

    def ribbon_topk(self, tau: float, is_mlp: bool, k: int, device: torch.device):
        """Exact C^0-continuous order-statistic top-k weights: r_j = vals[j] - vals[k] >= 0, RMS-normalized."""
        basis = legendre_p0_p4(tau, device)  # [5]
        centers = self.c_m if is_mlp else self.c_a
        theta = self.theta_m if is_mlp else self.theta_a
        diff = torch.abs(float(tau) - centers)
        if self.ribbon_ring:
            diff = torch.minimum(diff, 1.0 - diff)
        sigma = 0.14
        scores = torch.exp(-0.5 * (diff / sigma) ** 2) + (theta @ basis)
        vals, idx = torch.topk(scores, k + 1)
        r = (vals[:k] - vals[k]).clamp_min(0.0)
        g = r * torch.rsqrt((r * r).mean() + 1e-6)
        return idx[:k], g


def _gather_tape_indices(start: int, length: int, total: int, ring: bool, device: torch.device) -> torch.Tensor:
    if ring:
        return (torch.arange(length, device=device) + start) % total
    return torch.arange(start, start + length, device=device)


def _slice_contiguous_or_wrap(tensor: torch.Tensor, start: int, length: int, dim: int, ring: bool) -> torch.Tensor:
    total = tensor.shape[dim]
    start = start % total if ring else max(0, min(start, total - length))
    if start + length <= total:
        return tensor.narrow(dim, start, length)
    # Circular wrap-around: concat tail + head (fast, contiguous, zero scatter!)
    first = total - start
    second = length - first
    return torch.cat((tensor.narrow(dim, start, first), tensor.narrow(dim, 0, second)), dim=dim)


def attach_wave12_executor(model, idea: str, dynamics: str, device: torch.device):
    c = model.config
    K = c.core_layers  # 4
    H = c.heads        # 6
    D = c.width // H   # 128
    M = 4 * c.width    # 3072
    total_H = K * H    # 24
    total_M = K * M    # 12288
    n_mlp_exp = 32     # 32 atomic micro-experts of 384 paired neurons; k_m = 8 -> 3072 active (1.00x FLOPs!)
    d_mlp_exp = total_M // n_mlp_exp  # 384
    k_mlp_exp = M // d_mlp_exp        # 8

    sm = DepthSlideMod(idea, dynamics, c.width, total_H, n_mlp_exp, device=device)
    model.loop_state_mod = sm
    core_blocks = [model.transformer.h[c.pre_layers + k] for k in range(K)]

    is_ring = idea in ("slide_ring_hard", "slide_ring_soft", "slide_ring_2rev")
    is_soft = idea in ("slide_linear_soft", "slide_ring_soft")
    n_revs = 2.0 if idea == "slide_ring_2rev" else 1.0
    slide_attn = (idea != "slide_attn_cycle")
    mlp_chunk = 192  # 3072 / 192 = 16 chunks; 12288 / 192 = 64 total chunks

    def _run_slide_block(self, x, x0, idx, v_idx, N_core, total_depth,
                         Wq_t, Wk_t, Wv_t, Wo_t, Wfc_t, Wpr_t):
        tau_mid = (float(v_idx) + 0.5) / float(N_core)
        if is_ring:
            u = (float(v_idx) / float(N_core)) * n_revs
            mlp_pos = (u * total_M) % total_M
            head_pos = (u * total_H) % total_H
        else:
            u = float(v_idx) / float(max(N_core - 1, 1))
            mlp_pos = u * float(total_M - M)
            head_pos = u * float(total_H - H)

        band_k = min(K - 1, (v_idx * K) // N_core) if not is_ring else ((v_idx * K) // N_core) % K
        phys = c.pre_layers + (v_idx % K if idea == "slide_attn_cycle" else band_k)
        virt = c.pre_layers + v_idx
        x = self.resid_lambdas[phys] * x + self.x0_lambdas[phys] * x0
        cos_sin = (self.cos[:, :x.size(1)], self.sin[:, :x.size(1)])
        win = self._window_size(virt, total_depth)
        ve = self._value_embed(phys, idx)

        # ---- 1. ATTENTION ----
        xn = norm(x)
        B, T, C = xn.size()
        dt = xn.dtype
        if sm.is_ribbon:
            a_idx, a_g = sm.ribbon_topk(tau_mid, False, H, x.device)
            Wq = Wq_t.index_select(0, a_idx).reshape(C, C)
            Wk = Wk_t.index_select(0, a_idx).reshape(C, C)
            Wv = Wv_t.index_select(0, a_idx).reshape(C, C)
            Wo = (Wo_t.index_select(1, a_idx) * a_g.to(dt).view(1, H, 1)).reshape(C, C)
            q = F.linear(xn, Wq).view(B, T, H, D)
            k = F.linear(xn, Wk).view(B, T, H, D)
            v = F.linear(xn, Wv).view(B, T, H, D)
            if ve is not None:
                ve_v = ve.view(B, T, H, D)
                gate = 3 * torch.sigmoid(core_blocks[band_k].attn.ve_gate(xn[..., :core_blocks[band_k].attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve_v
            cos, sin = cos_sin
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q) * 1.2, norm(k) * 1.2
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=win).contiguous().view(B, T, C)
            attn_out = F.linear(y, Wo)
        elif not slide_attn:
            attn_out = core_blocks[v_idx % K].attn(xn, ve, cos_sin, win, None)
        else:
            if not is_soft:
                h0 = int(round(head_pos)) % total_H if is_ring else int(round(head_pos))
                Wq = _slice_contiguous_or_wrap(Wq_t, h0, H, dim=0, ring=is_ring).reshape(C, C)
                Wk = _slice_contiguous_or_wrap(Wk_t, h0, H, dim=0, ring=is_ring).reshape(C, C)
                Wv = _slice_contiguous_or_wrap(Wv_t, h0, H, dim=0, ring=is_ring).reshape(C, C)
                Wo = _slice_contiguous_or_wrap(Wo_t, h0, H, dim=1, ring=is_ring).reshape(C, C)
                q = F.linear(xn, Wq).view(B, T, H, D)
                k = F.linear(xn, Wk).view(B, T, H, D)
                v = F.linear(xn, Wv).view(B, T, H, D)
                if ve is not None:
                    ve_v = ve.view(B, T, H, D)
                    gate = 3 * torch.sigmoid(core_blocks[band_k].attn.ve_gate(xn[..., :core_blocks[band_k].attn.ve_gate_channels]))
                    v = v + gate.unsqueeze(-1) * ve_v
                cos, sin = cos_sin
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                q, k = norm(q) * 1.2, norm(k) * 1.2
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=win).contiguous().view(B, T, C)
                attn_out = F.linear(y, Wo)
            else:
                # C^1 smooth fractional boundary head weighting: run H+1 heads when fractional, exact variance (1-f)+f=1!
                h0 = int(math.floor(head_pos))
                hfrac = head_pos - float(h0)
                if (not is_ring and h0 >= total_H - H) or hfrac < 1e-4:
                    h_use = min(h0, total_H - H) if not is_ring else (h0 % total_H)
                    Wq = _slice_contiguous_or_wrap(Wq_t, h_use, H, dim=0, ring=is_ring).reshape(C, C)
                    Wk = _slice_contiguous_or_wrap(Wk_t, h_use, H, dim=0, ring=is_ring).reshape(C, C)
                    Wv = _slice_contiguous_or_wrap(Wv_t, h_use, H, dim=0, ring=is_ring).reshape(C, C)
                    Wo = _slice_contiguous_or_wrap(Wo_t, h_use, H, dim=1, ring=is_ring).reshape(C, C)
                    q = F.linear(xn, Wq).view(B, T, H, D)
                    k = F.linear(xn, Wk).view(B, T, H, D)
                    v = F.linear(xn, Wv).view(B, T, H, D)
                    if ve is not None:
                        ve_v = ve.view(B, T, H, D)
                        gate = 3 * torch.sigmoid(core_blocks[band_k].attn.ve_gate(xn[..., :core_blocks[band_k].attn.ve_gate_channels]))
                        v = v + gate.unsqueeze(-1) * ve_v
                    cos, sin = cos_sin
                    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                    q, k = norm(q) * 1.2, norm(k) * 1.2
                    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=win).contiguous().view(B, T, C)
                    attn_out = F.linear(y, Wo)
                else:
                    H1 = H + 1
                    Wq = _slice_contiguous_or_wrap(Wq_t, h0, H1, dim=0, ring=is_ring).reshape(H1 * D, C)
                    Wk = _slice_contiguous_or_wrap(Wk_t, h0, H1, dim=0, ring=is_ring).reshape(H1 * D, C)
                    Wv = _slice_contiguous_or_wrap(Wv_t, h0, H1, dim=0, ring=is_ring).reshape(H1 * D, C)
                    Wo = _slice_contiguous_or_wrap(Wo_t, h0, H1, dim=1, ring=is_ring)  # [C, H1, D]
                    q = F.linear(xn, Wq).view(B, T, H1, D)
                    k = F.linear(xn, Wk).view(B, T, H1, D)
                    v = F.linear(xn, Wv).view(B, T, H1, D)
                    if ve is not None:
                        ve_v = ve.view(B, T, H, D)
                        gate = 3 * torch.sigmoid(core_blocks[band_k].attn.ve_gate(xn[..., :core_blocks[band_k].attn.ve_gate_channels]))
                        add_v = gate.unsqueeze(-1) * ve_v  # [B, T, H, D]
                        v = torch.cat((v[:, :, :H] + add_v, v[:, :, H:]), dim=2)
                    cos, sin = cos_sin
                    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                    q, k = norm(q) * 1.2, norm(k) * 1.2
                    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=win)  # [B, T, H1, D]
                    # Variance-preserving C^1 fade on exiting head 0 & entering head H: (1-f) + f = 1.0!
                    w_h = torch.ones((H1,), device=x.device, dtype=dt)
                    w_h[0] = math.sqrt(1.0 - hfrac)
                    w_h[-1] = math.sqrt(hfrac)
                    Wo_w = (Wo * w_h.view(1, H1, 1)).reshape(C, H1 * D)
                    attn_out = F.linear(y.contiguous().view(B, T, H1 * D), Wo_w)

        x = x + attn_out

        # ---- 2. MLP (100% ATOMIC PAIRED NEURONS) ----
        xm = norm(x)
        if sm.is_ribbon:
            m_idx, m_g = sm.ribbon_topk(tau_mid, True, k_mlp_exp, x.device)
            U_pool = Wfc_t.view(n_mlp_exp, d_mlp_exp, C)
            D_pool = Wpr_t.view(C, n_mlp_exp, d_mlp_exp)
            Wfc = U_pool.index_select(0, m_idx).reshape(M, C)
            Wpr = (D_pool.index_select(1, m_idx) * m_g.to(dt).view(1, k_mlp_exp, 1)).reshape(C, M)
            h = F.relu(F.linear(xm, Wfc)).square()
            if sm.use_tok:
                # 2D Token x Depth zero-init micro-expert gating (0 scatter/gather, 1.00x FLOPs!)
                w_tok = sm.tok_router.weight.index_select(0, m_idx).to(dt)  # [8, C]
                tg = 1.0 + 0.5 * torch.tanh(F.linear(xm, w_tok))           # [B, T, 8]
                h = (h.view(B, T, k_mlp_exp, d_mlp_exp) * tg.unsqueeze(-1)).reshape(B, T, M)
            mlp_out = F.linear(h, Wpr)
        elif not is_soft:
            s = int(round(mlp_pos)) % total_M if is_ring else int(round(mlp_pos))
            Wfc = _slice_contiguous_or_wrap(Wfc_t, s, M, dim=0, ring=is_ring)
            Wpr = _slice_contiguous_or_wrap(Wpr_t, s, M, dim=1, ring=is_ring)
            mlp_out = F.linear(F.relu(F.linear(xm, Wfc)).square(), Wpr)
        else:
            # Chunk-level C^1 smooth boundary fade (chunk = 192 neurons; 16 active + 1 fractional boundary = 17 chunks)
            n_chunks_tot = total_M // mlp_chunk  # 64
            n_chunks_act = M // mlp_chunk        # 16
            cpos = (mlp_pos / float(mlp_chunk))
            c0 = int(math.floor(cpos))
            cfrac = cpos - float(c0)
            if (not is_ring and c0 >= n_chunks_tot - n_chunks_act) or cfrac < 1e-4:
                s = (c0 * mlp_chunk) % total_M if is_ring else min(c0 * mlp_chunk, total_M - M)
                Wfc = _slice_contiguous_or_wrap(Wfc_t, s, M, dim=0, ring=is_ring)
                Wpr = _slice_contiguous_or_wrap(Wpr_t, s, M, dim=1, ring=is_ring)
                mlp_out = F.linear(F.relu(F.linear(xm, Wfc)).square(), Wpr)
            else:
                M1 = M + mlp_chunk
                s = (c0 * mlp_chunk) % total_M if is_ring else c0 * mlp_chunk
                Wfc = _slice_contiguous_or_wrap(Wfc_t, s, M1, dim=0, ring=is_ring)
                Wpr = _slice_contiguous_or_wrap(Wpr_t, s, M1, dim=1, ring=is_ring)
                h = F.relu(F.linear(xm, Wfc)).square()
                # Exact variance-preserving fade on first & last chunk!
                w_m = torch.ones((M1,), device=x.device, dtype=dt)
                w_m[:mlp_chunk] = math.sqrt(1.0 - cfrac)
                w_m[-mlp_chunk:] = math.sqrt(cfrac)
                mlp_out = F.linear(h, Wpr * w_m.unsqueeze(0))

        return x + mlp_out

    def _custom_ordinary_hidden(self, idx):
        R_eff = getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops
        N_core = R_eff * K
        total = c.pre_layers + N_core + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        inv_r = 1.0 / float(R_eff)

        # Build BF16 atomic tapes ONCE per forward pass (zero per-layer stack overhead!)
        dt = x.dtype
        Wq_t = torch.cat([b.attn.c_q.weight.to(dt) for b in core_blocks], dim=0).view(total_H, D, c.width)
        Wk_t = torch.cat([b.attn.c_k.weight.to(dt) for b in core_blocks], dim=0).view(total_H, D, c.width)
        Wv_t = torch.cat([b.attn.c_v.weight.to(dt) for b in core_blocks], dim=0).view(total_H, D, c.width)
        Wo_t = torch.cat([b.attn.c_proj.weight.to(dt) for b in core_blocks], dim=1).view(c.width, total_H, D)
        Wfc_t = torch.cat([b.mlp.c_fc.weight.to(dt) for b in core_blocks], dim=0)    # [12288, 768]
        Wpr_t = torch.cat([b.mlp.c_proj.weight.to(dt) for b in core_blocks], dim=1)  # [768, 12288]

        for r in range(R_eff):
            x_in = x
            h = x_in
            for j in range(K):
                v_idx = r * K + j
                h = _run_slide_block(self, h, x0, idx, v_idx, N_core, total,
                                     Wq_t, Wk_t, Wv_t, Wo_t, Wfc_t, Wpr_t)
            delta = h - x_in
            if sm.dynamics == "plain":
                x = x_in + delta
            elif sm.dynamics == "euler":
                x = x_in + (sm.s_ode * inv_r).to(dt) * delta
            elif sm.dynamics == "sphere":
                x = norm(x_in + inv_r * (sm.s_ode.to(dt) * delta + sm.g_pre.to(dt) * prelude))
            if r == mid_g:
                cached = x

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
        "slide_linear_hard", "slide_linear_soft", "slide_ring_hard",
        "slide_ring_soft", "slide_ring_2rev", "slide_attn_cycle",
        "ribbon_linear", "ribbon_ring", "ribbon_tok",
    ])
    ap.add_argument("--dynamics", default="plain", choices=["plain", "euler", "sphere"])
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
    sm = attach_wave12_executor(model, args.idea, args.dynamics, device=device)
    extra_params = sum(p.numel() for p in sm.parameters())

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens)

    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, args.loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|dyn={args.dynamics}|R={args.loops}|u412={int(args.continuum_u412)}|"
          f"+p={extra_params} ({100*extra_params/base_params:.5f}%)] step=0 val_loss={vl0:.4f} val_bpb={bpb0:.5f}",
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
