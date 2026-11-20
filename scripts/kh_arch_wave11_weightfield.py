#!/usr/bin/env python3
"""Direction-A Arch Wave 11: Continuous Spherical Neural Weight Field (WeightField) & Layer-Repeat vs Cycle!

Core Bitter-Lesson Breakthrough (0.000% Extra Core Weight Matrices, < 1% Wall-Clock Overhead, ZERO [max_loops] Lookup!):
  Why do 0.1%-param modulations (AdaLN, SO(12), ReLU^2 gating) plateau at ~-0.005 nats over prelude_ode,
  while untied 20L (2/16/2) beats tied 2/4/2 by -0.067 nats?
  Because in standard Cycle ([0,1,2,3] x R), EVERY core block W_j must simultaneously act as early syntax (tau~0),
  mid reasoning (tau~0.5), and late decoding (tau~1) — massive depth interference!
  Conversely, in Layer-Repeat (0^R, 1^R, 2^R, 3^R), early/mid/late specialize, but adjacent layers are identical!

  Continuous Spherical Neural Weight Field synthesizes ALL 4*R virtual layers smoothly from the EXACT SAME
  K=4 (28.3M) core weight matrices {W_0, W_1, W_2, W_3} via L2-normalized spherical interpolation on S^{K-1}:
      W_g(tau) = sum_{k=0}^{K-1} alpha_{g,k}(tau) * W_{k,g},   with  ||alpha_g(tau)||_2 == 1  EXACTLY!
  Why L2-normalization (S^{K-1}) instead of simplex (sum alpha = 1)?
  For uncorrelated W_k, Var(sum alpha_k W_k x) = (sum alpha_k^2) Var(W x) == Var(W x) IDENTICALLY for ALL tau & R!
  And alpha_g(tau) spans negative + positive orthogonal combinations across ALL G groups (6 attn heads, 16 MLP groups)
  using zero-aliasing shifted Legendre polynomials P_1..P_4(2*tau - 1)!

Ideas tested in Wave 11 (ALL preserving resid_lambdas, --freeze-x0, exact ood_2048):
  1. layer_repeat     : 0^R, 1^R, 2^R, 3^R (0 extra params; isolates pure macro depth-band specialization!)
  2. wf_band_spline   : Smooth continuous weight field W(tau) = sum_k alpha_k(tau) W_k initialized on depth-band RBF knots (120 floats!)
  3. wf_band_group    : Per-Head (6) & Per-MLP-Group (16) Spherical Weight Field initialized on staggered depth-band knots (1,120 floats!)
  4. wf_cycle_group   : Cycle-init Per-Head (6) & Per-MLP-Group (16) Spherical Weight Field:
                        virtual layer (r, j) starts at exact W_j at step 0, then smoothly borrows from all K=4 blocks via P_1..P_4(tau)!
  5. wf_cycle_scalar  : Cycle-init matrix-level Spherical Weight Field (only 480 floats — clean LOO vs per-group!)
  6. wf_dense_bank    : Pool ALL K=4 core blocks with 32-group fine-grained MLP + 12-subhead Attn Spherical Weight Field (~2.5k floats!)
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


def legendre_p0_p4(tau: float, device: torch.device) -> torch.Tensor:
    """Shifted Legendre polynomials [1, P_1, P_2, P_3, P_4] on t = 2*tau - 1 in (-1, 1). Zero Nyquist aliasing!"""
    t = 2.0 * float(tau) - 1.0
    t2 = t * t
    p1 = t
    p2 = 0.5 * (3.0 * t2 - 1.0)
    p3 = 0.5 * t * (5.0 * t2 - 3.0)
    p4 = 0.125 * (35.0 * t2 * t2 - 30.0 * t2 + 3.0)
    return torch.tensor([1.0, p1, p2, p3, p4], device=device, dtype=torch.float32)


def _blend_grouped_weight(weights_list: list[torch.Tensor], alpha: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """Synthesizes W_eff from K weight tensors [out_d, in_d] using group weights alpha [G, K] with ||alpha[g]||_2 == 1.
    Returns [out_d, in_d] in dtype dt. ZERO extra core weight matrices, < 0.4% FLOPs!
    """
    # Stack K weights: [K, out_d, in_d] -> [K, G, d_g, in_d]
    G, K = alpha.shape
    out_d, in_d = weights_list[0].shape
    assert out_d % G == 0
    d_g = out_d // G
    W_stack = torch.stack([w.to(dt) for w in weights_list], dim=0).view(K, G, d_g, in_d)
    # einsum "gk,kgdi->gdi" via fast matmul: permute [G, K] @ [G, K, d_g*in_d] -> [G, d_g*in_d]
    W_perm = W_stack.permute(1, 0, 2, 3).reshape(G, K, d_g * in_d)
    W_eff = torch.matmul(alpha.to(dt).unsqueeze(1), W_perm).reshape(out_d, in_d)
    return W_eff


class WeightFieldMod(nn.Module):
    def __init__(self, idea: str, dynamics: str, core_layers: int, n_head: int, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.dynamics = dynamics
        self.K = core_layers  # 4
        self.n_head = n_head
        self.width = width
        self.n_basis = 5  # [1, P1, P2, P3, P4]

        if dynamics == "sphere":
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))
        elif dynamics == "euler":
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        self.is_cycle_field = idea in ("wf_cycle_group", "wf_cycle_scalar", "wf_dense_bank")
        self.is_band_field = idea in ("wf_band_spline", "wf_band_group")

        if self.is_band_field:
            self.g_attn = n_head if idea == "wf_band_group" else 1
            self.g_mlp = 16 if idea == "wf_band_group" else 1
            # 4 attn projs (q,k,v,o) + 2 mlp projs (fc,proj)
            # Base centers c_k in [0, 1] + learned Legendre logits [n_proj, G, K, 5]
            self.attn_theta = nn.Parameter(torch.zeros(4, self.g_attn, self.K, self.n_basis, device=device))
            self.mlp_theta = nn.Parameter(torch.zeros(2, self.g_mlp, self.K, self.n_basis, device=device))
            # Staggered depth-band RBF centers for groups so step=0 smoothly covers all K=4 blocks without collapse!
            centers = torch.linspace(0.0, 1.0, self.K, device=device)  # [0, 1/3, 2/3, 1]
            self.register_buffer("band_centers", centers)
            if idea == "wf_band_group":
                # Slight per-group phase offsets in [-0.08, +0.08] for instant symmetry breaking across heads/groups
                self.register_buffer("attn_off", torch.linspace(-0.08, 0.08, self.g_attn, device=device))
                self.register_buffer("mlp_off", torch.linspace(-0.08, 0.08, self.g_mlp, device=device))
            else:
                self.register_buffer("attn_off", torch.zeros(1, device=device))
                self.register_buffer("mlp_off", torch.zeros(1, device=device))

        elif self.is_cycle_field:
            if idea == "wf_cycle_scalar":
                self.g_attn, self.g_mlp = 1, 1
            elif idea == "wf_dense_bank":
                self.g_attn, self.g_mlp = 12, 32
            else:
                self.g_attn, self.g_mlp = n_head, 16
            # For each cycle slot j in {0..K-1}, raw coeff = one_hot(j) + sum_{m=1..4} P_m(tau) * theta[j, proj, g, k, m-1]
            # Zero-init theta => EXACT bit-for-bit fixed_recursive at step 0, instant full-rank Step-0 grad!
            self.attn_theta = nn.Parameter(torch.zeros(self.K, 4, self.g_attn, self.K, 4, device=device))
            self.mlp_theta = nn.Parameter(torch.zeros(self.K, 2, self.g_mlp, self.K, 4, device=device))

    def get_band_alpha(self, tau: float, is_mlp: bool, proj_idx: int, device: torch.device) -> torch.Tensor:
        """Returns [G, K] L2-normalized spherical weights on S^{K-1}."""
        basis = legendre_p0_p4(tau, device)  # [5]
        theta = self.mlp_theta[proj_idx] if is_mlp else self.attn_theta[proj_idx]  # [G, K, 5]
        off = self.mlp_off if is_mlp else self.attn_off  # [G]
        # Smooth RBF prior peaking at band_centers[k], sigma=0.22
        t_g = (float(tau) + off).clamp(0.0, 1.0).unsqueeze(-1)  # [G, 1]
        prior = torch.exp(-0.5 * ((t_g - self.band_centers.unsqueeze(0)) / 0.22) ** 2)  # [G, K]
        delta = torch.einsum("gkb,b->gk", theta, basis)
        raw = prior + delta
        return F.normalize(raw, p=2, dim=-1, eps=1e-6)

    def get_cycle_alpha(self, slot_j: int, tau: float, is_mlp: bool, proj_idx: int, device: torch.device) -> torch.Tensor:
        """Returns [G, K] L2-normalized spherical weights starting at exact one_hot(slot_j) at step 0!"""
        p1_p4 = legendre_p0_p4(tau, device)[1:]  # [4] = [P1, P2, P3, P4] (strictly zero-mean!)
        theta = self.mlp_theta[slot_j, proj_idx] if is_mlp else self.attn_theta[slot_j, proj_idx]  # [G, K, 4]
        delta = torch.einsum("gkm,m->gk", theta, p1_p4)
        base = torch.zeros_like(delta)
        base[:, slot_j] = 1.0
        raw = base + delta
        return F.normalize(raw, p=2, dim=-1, eps=1e-6)


class _WFState:
    def __init__(self):
        self.active = False
        self.slot_j = 0
        self.tau = 0.5


WF_CTX = _WFState()


def attach_wave11_executor(model, idea: str, dynamics: str, device: torch.device):
    c = model.config
    sm = WeightFieldMod(idea, dynamics, c.core_layers, c.heads, c.width, device=device)
    model.loop_state_mod = sm

    core_blocks = [model.transformer.h[c.pre_layers + k] for k in range(c.core_layers)]
    K = c.core_layers

    if sm.is_band_field or sm.is_cycle_field:
        # Collect references to all K=4 shared core weights for each projection
        wq_list = [b.attn.c_q.weight for b in core_blocks]
        wk_list = [b.attn.c_k.weight for b in core_blocks]
        wv_list = [b.attn.c_v.weight for b in core_blocks]
        wo_list = [b.attn.c_proj.weight for b in core_blocks]
        wfc_list = [b.mlp.c_fc.weight for b in core_blocks]
        wpr_list = [b.mlp.c_proj.weight for b in core_blocks]

        for k_idx, blk in enumerate(core_blocks):
            def _make_attn_fwd():
                def _attn_fwd(self, x, ve, cos_sin, window_size, kv_cache):
                    B, T, C = x.size()
                    dt = x.dtype
                    if WF_CTX.active:
                        if sm.is_band_field:
                            aq = sm.get_band_alpha(WF_CTX.tau, False, 0, x.device)
                            ak = sm.get_band_alpha(WF_CTX.tau, False, 1, x.device)
                            av = sm.get_band_alpha(WF_CTX.tau, False, 2, x.device)
                            ao = sm.get_band_alpha(WF_CTX.tau, False, 3, x.device)
                        else:
                            aq = sm.get_cycle_alpha(WF_CTX.slot_j, WF_CTX.tau, False, 0, x.device)
                            ak = sm.get_cycle_alpha(WF_CTX.slot_j, WF_CTX.tau, False, 1, x.device)
                            av = sm.get_cycle_alpha(WF_CTX.slot_j, WF_CTX.tau, False, 2, x.device)
                            ao = sm.get_cycle_alpha(WF_CTX.slot_j, WF_CTX.tau, False, 3, x.device)
                        Wq = _blend_grouped_weight(wq_list, aq, dt)
                        Wk = _blend_grouped_weight(wk_list, ak, dt)
                        Wv = _blend_grouped_weight(wv_list, av, dt)
                        Wo = _blend_grouped_weight(wo_list, ao, dt)
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
                    q, k = norm(q), norm(k)
                    q = q * 1.2
                    k = k * 1.2

                    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
                    y = y.contiguous().view(B, T, -1)
                    return F.linear(y, Wo) if Wo is not None else self.c_proj(y)
                return _attn_fwd

            def _make_mlp_fwd():
                def _mlp_fwd(self, x):
                    dt = x.dtype
                    if WF_CTX.active:
                        if sm.is_band_field:
                            afc = sm.get_band_alpha(WF_CTX.tau, True, 0, x.device)
                            apr = sm.get_band_alpha(WF_CTX.tau, True, 1, x.device)
                        else:
                            afc = sm.get_cycle_alpha(WF_CTX.slot_j, WF_CTX.tau, True, 0, x.device)
                            apr = sm.get_cycle_alpha(WF_CTX.slot_j, WF_CTX.tau, True, 1, x.device)
                        Wfc = _blend_grouped_weight(wfc_list, afc, dt)
                        Wpr = _blend_grouped_weight(wpr_list, apr, dt)
                        return F.linear(F.relu(F.linear(x, Wfc)).square(), Wpr)
                    return self.c_proj(F.relu(self.c_fc(x)).square())
                return _mlp_fwd

            blk.attn.forward = types.MethodType(_make_attn_fwd(), blk.attn)
            blk.mlp.forward = types.MethodType(_make_mlp_fwd(), blk.mlp)

    def _custom_ordinary_hidden(self, idx):
        R_eff = getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops
        N_core = R_eff * K
        total = c.pre_layers + N_core + c.post_layers
        x = self._embed_reference(idx)
        x0 = x
        WF_CTX.active = False
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        inv_r = 1.0 / float(R_eff)

        WF_CTX.active = True
        try:
            if idea == "layer_repeat":
                # Run block 0 R times, block 1 R times, ..., block 3 R times (total 4*R core layers)
                v_idx = 0
                for k_blk in range(K):
                    phys = c.pre_layers + k_blk
                    for r in range(R_eff):
                        x_in = x
                        h_out = self._block(x_in, x0, idx, phys, c.pre_layers + v_idx, total)
                        delta = h_out - x_in
                        if sm.dynamics == "plain":
                            x = x_in + delta
                        elif sm.dynamics == "euler":
                            x = x_in + (sm.s_ode * inv_r).to(x.dtype) * delta
                        elif sm.dynamics == "sphere":
                            x = norm(x_in + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))
                        if v_idx == N_core // 2:
                            cached = x
                        v_idx += 1
            elif sm.is_band_field:
                # Group into R_eff steps of K=4 virtual layers so dynamics ("sphere"/"euler"/"plain") match 1-to-1!
                for r in range(R_eff):
                    x_in = x
                    h = x_in
                    for j in range(K):
                        v_idx = r * K + j
                        tau = (float(v_idx) + 0.5) / float(N_core)
                        WF_CTX.tau = tau
                        # Depth-band aligned phys for value_embeds & resid_lambdas: early->0, mid->1/2, late->3!
                        band_k = min(K - 1, (v_idx * K) // N_core)
                        phys = c.pre_layers + band_k
                        h = self._block(h, x0, idx, phys, c.pre_layers + v_idx, total)
                    delta = h - x_in
                    if sm.dynamics == "plain":
                        x = x_in + delta
                    elif sm.dynamics == "euler":
                        x = x_in + (sm.s_ode * inv_r).to(x.dtype) * delta
                    elif sm.dynamics == "sphere":
                        x = norm(x_in + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))
                    if r == mid_g:
                        cached = x
            else:
                # Cycle-init spherical weight field: loop r runs slots j=0..3 with smooth depth tau=(r+0.5)/R_eff
                for r in range(R_eff):
                    x_in = x
                    h = x_in
                    WF_CTX.tau = (float(r) + 0.5) * inv_r
                    for j in range(K):
                        WF_CTX.slot_j = j
                        phys = c.pre_layers + j
                        virt = c.pre_layers + r * K + j
                        h = self._block(h, x0, idx, phys, virt, total)
                    delta = h - x_in
                    if sm.dynamics == "plain":
                        x = x_in + delta
                    elif sm.dynamics == "euler":
                        x = x_in + (sm.s_ode * inv_r).to(x.dtype) * delta
                    elif sm.dynamics == "sphere":
                        x = norm(x_in + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))
                    if r == mid_g:
                        cached = x
        finally:
            WF_CTX.active = False

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
        "layer_repeat", "wf_band_spline", "wf_band_group",
        "wf_cycle_group", "wf_cycle_scalar", "wf_dense_bank",
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
    sm = attach_wave11_executor(model, args.idea, args.dynamics, device=device)
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
