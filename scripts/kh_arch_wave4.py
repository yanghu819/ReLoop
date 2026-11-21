#!/usr/bin/env python3
"""Direction-A Arch Wave 4: Continuous-Time Latent Nesterov-ODE LOO Ladder, R=1 Untied DiD, & Same-Slice OOD.

Strictly fixes the 1024-vs-4096 OOD slice offset caught in Wave 3:
  - final_val_loss is evaluated on 4096 rows (100% comparable to Waves 1-3).
  - ood_2048 evaluates EVERY R_eval in {1, 2, 4, 8, 12, 16, 32} on the EXACT SAME 2048 rows!
    => ood_delta_16_8 = ood_2048['16'] - ood_2048['8'] has ZERO eval-slice bias!

Implements internal review #94 + internal audit #93 exact LOO & DiD suite (lr = 0.003 * bscale locked!):
  1. nesterov_ode_vec    : 3-line, 1537-scalar (0.0007% param, ~0.4% cost!) continuous Latent Nesterov-ODE
                           mu_r = sigmoid(mu)**(4/R); h_in = x + g_look*s; d = F(h_in)-h_in;
                           s = mu_r*s + (1-mu_r)*d; x = x + (s_ode/R)*d
  2. nesterov_no_gw      : Dynamic additive gr(x, s) lookahead (NO cat!) + scalar (s_ode/R)*d (<1.8% cost)
  3. gated_euler_no_sreg : MANDATORY Auditor #93 LOO: s_reg == 0, dynamic additive gw(x, d) ONLY!
  4. nesterov_ode_fast   : Full dynamic additive gr + gw (NO cat, <2.5% cost) + continuous mu_r = mu**(8/R)!
  5. nesterov_ode_attnpool : nesterov_ode_fast + Loop-AttnPool readout!
  6. nesterov_prelude_vec  : 4-line, 2305-scalar (0.001% param, <1% cost) Nesterov lookahead + True Spherical Prelude ODE!
  7. r1_20L_nesterov_fast  : Gate A1.3 Untied 2/16/2 R=1 DiD control (applied across four 4-layer groups!)
  8. r1_20L_gated_euler    : Gate A1.3 Untied 2/16/2 R=1 DiD control for gated_euler_no_sreg!
  9. r1_20L_nesterov_vec   : Gate A1.3 Untied 2/16/2 R=1 DiD control for 3-line nesterov_ode_vec!
 10. huginn_fullbptt_lr2   : Gate A1.4 Hidden-LR control (--core-lr-mult 2.0 on full-8-BPTT Huginn!)
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
from nexttransformer.models.backbone import norm
from nexttransformer.training.optimizer import ExperimentOptimizer


class FastAddGate(nn.Module):
    """No-cat additive rank-16 gate: sigmoid(W_up(silu(W_a(norm(a)) + W_b(norm(b)))) + bias). <1% overhead!"""
    def __init__(self, width: int, rank: int, bias_init: float, device: torch.device):
        super().__init__()
        self.wa = nn.Linear(width, rank, bias=False, device=device, dtype=torch.bfloat16)
        self.wb = nn.Linear(width, rank, bias=False, device=device, dtype=torch.bfloat16)
        self.wup = nn.Linear(rank, width, bias=True, device=device, dtype=torch.bfloat16)
        nn.init.normal_(self.wa.weight, std=0.02)
        nn.init.normal_(self.wb.weight, std=0.02)
        nn.init.zeros_(self.wup.weight)
        nn.init.constant_(self.wup.bias, bias_init)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.wa(norm(a)) + self.wb(norm(b)))
        return torch.sigmoid(self.wup(h).float()).to(a.dtype)


class Wave4StateModule(nn.Module):
    def __init__(self, idea: str, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        self.width = width

        if idea in ("nesterov_ode_vec", "nesterov_prelude_vec", "r1_20L_nesterov_vec"):
            self.mu_logit = nn.Parameter(torch.full((width,), 0.85, device=device, dtype=torch.float32))
            self.g_look = nn.Parameter(torch.full((width,), 0.12, device=device, dtype=torch.float32))
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            if idea == "nesterov_prelude_vec":
                self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))

        if idea in ("prelude_true_ode", "polyak_geodesic"):
            self.g_pre = nn.Parameter(torch.full((width,), 1.5 if idea == "prelude_true_ode" else 0.25, device=device, dtype=torch.float32))
            if idea == "prelude_true_ode":
                self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if idea == "nesterov_no_gw":
            self.mu_logit = nn.Parameter(torch.full((width,), 0.85, device=device, dtype=torch.float32))
            self.gr = FastAddGate(width, 16, bias_init=-2.0, device=device)
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if idea in ("gated_euler_no_sreg", "r1_20L_gated_euler"):
            self.gw = FastAddGate(width, 16, bias_init=0.0, device=device)
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if idea in ("nesterov_ode_fast", "nesterov_ode_attnpool", "r1_20L_nesterov_fast"):
            self.mu_logit = nn.Parameter(torch.full((width,), 0.85, device=device, dtype=torch.float32))
            self.gr = FastAddGate(width, 16, bias_init=-2.0, device=device)
            self.gw = FastAddGate(width, 16, bias_init=0.0, device=device)
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))

        if idea == "nesterov_ode_attnpool":
            self.ap_wq = nn.Linear(2 * width, 16, bias=False, device=device, dtype=torch.bfloat16)
            self.ap_wk = nn.Linear(width, 16, bias=False, device=device, dtype=torch.bfloat16)
            nn.init.zeros_(self.ap_wq.weight)
            nn.init.normal_(self.ap_wk.weight, std=0.05)
            self.ap_last_bias = nn.Parameter(torch.tensor(2.0, device=device, dtype=torch.float32))


def attach_wave4_executor(model, idea: str, device: torch.device):
    sm = Wave4StateModule(idea, model.config.width, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}
    if idea == "official_base":
        return sm

    is_r1_did = idea.startswith("r1_20L_")

    def _run_group(self, h, x0, idx, g_idx, total_depth):
        c = self.config
        if is_r1_did:
            # 4 untied groups of 4 layers each (core_layers == 16)
            for j in range(4):
                phys = c.pre_layers + g_idx * 4 + j
                h = self._block(h, x0, idx, phys, phys, total_depth)
        else:
            for j in range(c.core_layers):
                phys = c.pre_layers + j
                virt = c.pre_layers + g_idx * c.core_layers + j
                h = self._block(h, x0, idx, phys, virt, total_depth)
        return h

    def _apply_attnpool(states):
        H = torch.stack(states, dim=-2)
        q = sm.ap_wq(torch.cat((norm(states[0]), norm(states[-1])), dim=-1)).float()
        K = sm.ap_wk(norm(H)).float()
        scores = torch.einsum("btd,btkd->btk", q, K) * 0.25
        bias = torch.zeros(len(states), device=H.device, dtype=torch.float32)
        bias[-1] = sm.ap_last_bias
        w = F.softmax(scores + bias, dim=-1).to(H.dtype)
        return torch.einsum("btk,btkd->btd", w, H)

    def _custom_ordinary_hidden(self, idx):
        c = self.config
        if is_r1_did:
            R_eff = 4
            assert c.loops == 1 and c.core_layers == 16
            total = c.pre_layers + c.core_layers + c.post_layers
        else:
            R_eff = getattr(self, "_eval_loops_override", None) or getattr(self, "_step_loops_override", None) or c.loops
            total = c.pre_layers + R_eff * c.core_layers + c.post_layers

        x = self._embed_reference(idx)
        x0 = x
        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
        prelude = x
        cached = None
        mid_g = R_eff // 2
        rec_diag = (not self.training) and getattr(self, "_collect_loop_diag", False)
        norms, raw_cos_list = [], []
        prev_raw_delta = None

        s_reg = torch.zeros_like(prelude)
        bar_z = torch.zeros_like(prelude)
        states = [prelude] if idea == "nesterov_ode_attnpool" else None
        inv_r = 1.0 / float(R_eff)
        tbptt = getattr(self, "_tbptt_loops", 0)

        for r in range(R_eff):
            no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
            if self.training and tbptt > 0 and r == R_eff - tbptt:
                x = x.detach()
                s_reg = s_reg.detach()
                bar_z = bar_z.detach()
            with torch.no_grad() if no_g else nullcontext():
                if idea in ("nesterov_ode_vec", "nesterov_prelude_vec", "r1_20L_nesterov_vec"):
                    mu_r = (torch.sigmoid(sm.mu_logit) ** (4.0 * inv_r)).to(x.dtype)
                    h_in = x + sm.g_look.to(x.dtype) * s_reg
                    delta = _run_group(self, h_in, x0, idx, r, total) - h_in
                    s_reg = mu_r * s_reg + (1.0 - mu_r) * delta
                    if idea == "nesterov_prelude_vec":
                        x = norm(x + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))
                    else:
                        x = x + (sm.s_ode * inv_r).to(x.dtype) * delta

                elif idea == "prelude_true_ode":
                    delta = _run_group(self, x, x0, idx, r, total) - x
                    x = norm(x + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))

                elif idea == "polyak_geodesic":
                    delta = _run_group(self, x, x0, idx, r, total) - x
                    x = norm(x + delta + sm.g_pre.to(x.dtype) * prelude)
                    bar_z = bar_z + float(r + 1) * x

                elif idea == "nesterov_no_gw":
                    mu_r = (torch.sigmoid(sm.mu_logit) ** (8.0 * inv_r)).to(x.dtype)
                    gr = sm.gr(x, s_reg)
                    h_in = x + gr * s_reg
                    delta = _run_group(self, h_in, x0, idx, r, total) - h_in
                    s_reg = mu_r * s_reg + (1.0 - mu_r) * delta
                    x = x + (sm.s_ode * inv_r).to(x.dtype) * delta

                elif idea in ("gated_euler_no_sreg", "r1_20L_gated_euler"):
                    delta = _run_group(self, x, x0, idx, r, total) - x
                    gw = sm.gw(x, delta)
                    x = x + (2.0 * sm.s_ode * inv_r).to(x.dtype) * gw * delta

                elif idea in ("nesterov_ode_fast", "nesterov_ode_attnpool", "r1_20L_nesterov_fast"):
                    mu_r = (torch.sigmoid(sm.mu_logit) ** (8.0 * inv_r)).to(x.dtype)
                    gr = sm.gr(x, s_reg)
                    h_in = x + gr * s_reg
                    delta = _run_group(self, h_in, x0, idx, r, total) - h_in
                    s_reg = mu_r * s_reg + (1.0 - mu_r) * delta
                    gw = sm.gw(x, delta)
                    x = x + (2.0 * sm.s_ode * inv_r).to(x.dtype) * gw * delta
                    if idea == "nesterov_ode_attnpool":
                        states.append(x)

                if r == mid_g:
                    cached = x
                if rec_diag:
                    d_raw = delta.detach().float()
                    norms.append(float(x.float().norm(dim=-1).mean().item()))
                    if prev_raw_delta is not None:
                        raw_cos_list.append(float(F.cosine_similarity(d_raw, prev_raw_delta, dim=-1).mean().item()))
                    prev_raw_delta = d_raw

        if idea == "polyak_geodesic":
            x = norm(bar_z)
        elif idea == "nesterov_ode_attnpool":
            x = _apply_attnpool(states)
        if rec_diag:
            self._loop_diag = dict(norms=[round(v, 2) for v in norms], raw_delta_cos=[round(v, 3) for v in raw_cos_list])

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + j
            x = self._block(x, x0, idx, phys, virt, total)
        return x, cached

    model._ordinary_hidden = types.MethodType(_custom_ordinary_hidden, model)
    return sm


def build_official_opt(model, batch_tokens: int, core_lr_mult: float = 1.0):
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

    c = model.config
    core_ids = set()
    for l_idx in range(c.pre_layers, c.pre_layers + c.core_layers):
        for p in model.transformer.h[l_idx].parameters():
            core_ids.add(id(p))
    if hasattr(model, "state_adapter") and model.state_adapter is not None:
        for p in model.state_adapter.parameters():
            core_ids.add(id(p))

    allocated = {id(p) for g in groups for p in g["params"]}
    untied, core, other = defaultdict(list), defaultdict(list), []
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:
            continue
        if p.ndim == 2:
            (core if id(p) in core_ids else untied)[tuple(p.shape)].append(p)
        else:
            other.append(p)
    adam(other, adamw_lr, (0.9, 0.95), 0.0)
    for sh in sorted(untied):
        groups.append(dict(kind="muon", params=untied[sh], lr=muon_lr, initial_lr=muon_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=wd, initial_weight_decay=wd))
    for sh in sorted(core):
        lr_c = muon_lr * core_lr_mult
        groups.append(dict(kind="muon", params=core[sh], lr=lr_c, initial_lr=lr_c,
                           momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=wd, initial_weight_decay=wd))
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
                  loops_eval: int, max_rows: int = 4096, collect_diag: bool = False,
                  is_official_loop: bool = False):
    if is_official_loop:
        model.accounting_depth = loops_eval
    else:
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
        if is_official_loop:
            model.accounting_depth = None
        else:
            model._eval_loops_override = None
        model._collect_loop_diag = False
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1.0)), total_nats / max(total_toks, 1.0), diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=[
        "official_base",
        "nesterov_ode_vec", "nesterov_no_gw", "gated_euler_no_sreg",
        "nesterov_ode_fast", "nesterov_ode_attnpool", "nesterov_prelude_vec",
        "prelude_true_ode", "polyak_geodesic",
        "r1_20L_nesterov_fast", "r1_20L_gated_euler", "r1_20L_nesterov_vec",
    ])
    ap.add_argument("--method", default="fixed_recursive", choices=["fixed_recursive", "huginn"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tbptt-loops", type=int, default=0, help="If >0, only backprop through last K loops (matching Huginn 4-BPTT speed!)")
    ap.add_argument("--huginn-bptt", type=int, default=8)
    ap.add_argument("--huginn-sigma", type=float, default=1e-6)
    ap.add_argument("--core-lr-mult", type=float, default=1.0)
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

    is_hug = (args.method == "huginn")
    cfg = ModelConfig(
        method=args.method, scale="debug", width=args.width, heads=args.heads,
        vocab_size=32768, sequence_len=seq_len, mlp_width=4 * args.width,
        layers=args.pre_layers + args.core_layers + args.post_layers,
        pre_layers=args.pre_layers, core_layers=args.core_layers, post_layers=args.post_layers,
        loops=args.loops, max_loops=max(args.loops, 2),
        huginn_mean=float(max(args.loops, 2)),
        huginn_sigma=args.huginn_sigma if is_hug else 0.5,
        huginn_backprop_loops=min(args.huginn_bptt, max(args.loops, 1)) if is_hug else min(4, max(args.loops, 1)),
        eval_loops=args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    if args.freeze_x0:
        with torch.no_grad():
            model.x0_lambdas.zero_()
        model.x0_lambdas.requires_grad_(False)

    base_params = sum(p.numel() for p in model.parameters())
    sm = attach_wave4_executor(model, args.idea, device=device)
    model._tbptt_loops = args.tbptt_loops
    extra_params = sum(p.numel() for p in sm.parameters())
    is_off_loop = (args.idea == "official_base") and is_hug

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_official_opt(model, batch_tokens, core_lr_mult=args.core_lr_mult)

    n_rows = train_x.shape[0]
    bpb0, vl0, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                 args.loops, max_rows=512, is_official_loop=is_off_loop)
    print(f"[{args.tag}|{args.idea}|R={args.loops}|lr={args.core_lr_mult}|jit={int(args.horizon_jitter)}|+p={extra_params} ({100*extra_params/base_params:.4f}%)] "
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
                                         args.loops, max_rows=rows, collect_diag=(step + 1 == args.steps),
                                         is_official_loop=is_off_loop)
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum,
                                elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f}", flush=True)

    # ZERO-SLICE-BIAS OOD SWEEP: evaluate EVERY R_eval on the EXACT SAME 2048 val rows!
    ood_2048 = {}
    if args.loops == 1:
        _, vl_1, _ = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, 1, max_rows=2048)
        ood_2048["1"] = round(vl_1, 5)
    else:
        for r_ev in (1, 2, 4, 8, 12, 16, 32):
            ev_mb = 8 if r_ev >= 32 else (16 if r_ev >= 12 else min(args.micro_batch, 32))
            _, vl_r, _ = eval_at_depth(model, val_x, val_y, byte_table, device, ev_mb, r_ev,
                                       max_rows=2048, is_official_loop=is_off_loop)
            ood_2048[str(r_ev)] = round(vl_r, 5)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    ood_diff = round(ood_2048["16"] - ood_2048["8"], 5) if ("16" in ood_2048 and "8" in ood_2048) else None
    out = dict(
        tag=args.tag, idea=args.idea, method=args.method, loops=args.loops,
        core_lr_mult=args.core_lr_mult, horizon_jitter=args.horizon_jitter,
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
