#!/usr/bin/env python3
"""Direction-A Arch Wave 16: internal audit #127 Fresh-2250 Sub-Symbol LOO Shield + Cross-Backbone Huginn Transfer.

Seals every Reviewer-#2 / #102 ("just RMSNorm?", "just nGPT?", "just DEQ?", "why 1/R?", "2 scalars vs 769?",
"Fresh-2250 ve=4 R=1 DiD", and "cross-backbone SandwichBlock Huginn transfer") on exact Fresh-2250 (295M 1-epoch
fresh ClimbMix tokens, --freeze-x0, ve=4, exact same-slice ood_2048 across R in {1,2,4,8,12,16,32}):

  1. interloop_norm           : x = norm(x + delta)                                   [0 params]
  2. euler_only               : x = x + (s_ode / R) * delta                           [1 scalar]
  3. sphere_only              : x = norm(x + (s_ode / R) * delta)                     [1 scalar, g_pre == 0]
  4. prelude_nonorm           : x = x + (1/R) * (s_ode * delta + g_pre * prelude)     [769 scalars, NO norm]
  5. prelude_no_invR          : x = norm(x + delta + g_pre * prelude)                 [768 scalars, NO 1/R]
  6. scalar_g_ode             : x = norm(x + (1/R) * (s_ode * delta + g_s * prelude)) [2 scalars ONLY!]
  7. prelude_true_ode         : x = norm(x + (1/R) * (s_ode * delta + g_pre * prelude))[769 scalars, Champion]
  8. r1_20L_iso_ve_prelude_did: Untied 2/16/2 (113.25M core, ve=4!) + prelude_true_ode across 4 untied 4L groups
  9. huginn_spherical_ode     : Official Huginn SandwichBlock (2/4/2, NO state_adapter) + prelude_true_ode (769 scalars)
 10. fixed_base               : Clean fixed_recursive baseline (x = x + delta)        [0 params]
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
from nexttransformer.models.looped import SandwichBlock
from nexttransformer.training.optimizer import ExperimentOptimizer


def apply_iso_ve_4(model):
    """Exact same 4-VE-table matcher as Wave 14 (1 pre, 2 core, 1 post), freezing removed ve_gates."""
    c = model.config
    n_tot = len(model.transformer.h)
    keep = {1, c.pre_layers + c.core_layers // 4, c.pre_layers + (3 * c.core_layers) // 4, n_tot - 1}
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


class Wave16StateModule(nn.Module):
    def __init__(self, idea: str, width: int, device: torch.device):
        super().__init__()
        self.idea = idea
        if idea in ("euler_only", "sphere_only", "r1_20L_iso_ve_euler_did"):
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
        elif idea == "prelude_no_invR":
            # Matches Step-0 restoring ratio (1.5 / 8 = 0.1875 -> ~0.25) with fixed unit delta (NO 1/R!)
            self.g_pre = nn.Parameter(torch.full((width,), 0.25, device=device, dtype=torch.float32))
        elif idea in ("scalar_g_ode", "scalar_g_nonorm"):
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.tensor(1.5, device=device, dtype=torch.float32))
        elif idea in ("prelude_nonorm", "prelude_true_ode", "r1_20L_iso_ve_prelude_did",
                      "r1_20L_iso_ve_nonorm_did", "huginn_spherical_ode", "huginn_nonorm_ode"):
            self.s_ode = nn.Parameter(torch.tensor(4.0, device=device, dtype=torch.float32))
            self.g_pre = nn.Parameter(torch.full((width,), 1.5, device=device, dtype=torch.float32))


def attach_wave16_executor(model, idea: str, device: torch.device, tbptt_loops: int = 0):
    if idea in ("huginn_spherical_ode", "huginn_nonorm_ode"):
        # Wrap every block with official Huginn SandwichBlock (exact NextTransformer ReferenceLM parity, NO state_adapter!)
        for i in range(len(model.transformer.h)):
            model.transformer.h[i] = SandwichBlock(model.transformer.h[i])

    sm = Wave16StateModule(idea, model.config.width, device=device)
    model.loop_state_mod = sm
    model._loop_diag = {}
    model._tbptt_loops = int(tbptt_loops)
    is_r1_did = idea in ("r1_20L_iso_ve_prelude_did", "r1_20L_iso_ve_nonorm_did",
                         "r1_20L_iso_ve_euler_did", "r1_20L_iso_ve_base")

    def _run_group(self, h, x0, idx, g_idx, total_depth):
        c = self.config
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
        inv_r = 1.0 / float(R_eff)

        rec_diag = (not self.training) and getattr(self, "_collect_loop_diag", False)
        norms, raw_cos_list = [], []
        prev_raw_delta = None
        tbptt = int(getattr(self, "_tbptt_loops", 0))

        for r in range(R_eff):
            no_g = (self.training and tbptt > 0 and r < R_eff - tbptt)
            if self.training and tbptt > 0 and r == R_eff - tbptt:
                x = x.detach()
            with torch.no_grad() if no_g else nullcontext():
                h = _run_group(self, x, x0, idx, r, total)
                delta = h - x
                if idea in ("fixed_base", "r1_20L_iso_ve_base"):
                    x = h
                elif idea == "interloop_norm":
                    x = norm(h)
                elif idea in ("euler_only", "r1_20L_iso_ve_euler_did"):
                    x = x + (sm.s_ode * inv_r).to(x.dtype) * delta
                elif idea == "sphere_only":
                    x = norm(x + (sm.s_ode * inv_r).to(x.dtype) * delta)
                elif idea in ("prelude_nonorm", "scalar_g_nonorm", "r1_20L_iso_ve_nonorm_did", "huginn_nonorm_ode"):
                    x = x + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude)
                elif idea == "prelude_no_invR":
                    x = norm(h + sm.g_pre.to(x.dtype) * prelude)
                elif idea in ("scalar_g_ode", "prelude_true_ode", "r1_20L_iso_ve_prelude_did", "huginn_spherical_ode"):
                    x = norm(x + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude))
                else:
                    raise ValueError(f"Unknown idea: {idea}")

            if r == mid_g:
                cached = x
            if rec_diag:
                d_raw = delta.detach().float()
                norms.append(float(x.float().norm(dim=-1).mean().item()))
                if prev_raw_delta is not None:
                    raw_cos_list.append(float(F.cosine_similarity(d_raw, prev_raw_delta, dim=-1).mean().item()))
                prev_raw_delta = d_raw

        if rec_diag:
            self._loop_diag = dict(norms=[round(v, 2) for v in norms],
                                   raw_delta_cos=[round(v, 3) for v in raw_cos_list])

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + (c.core_layers if is_r1_did else R_eff * c.core_layers) + j
            x = self._block(x, x0, idx, phys, virt, total)
        return x, cached

    model._ordinary_hidden = types.MethodType(_custom_ordinary_hidden, model)
    return sm


def build_official_opt(model, batch_tokens: int, clean_x0_lr: bool = False):
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
    x0_lr = (0.003 * bscale) if clean_x0_lr else scalar_lr
    x0_betas = (0.9, 0.95) if clean_x0_lr else (0.96, 0.95)
    adam([model.x0_lambdas], x0_lr, x0_betas, 0.0)
    adam([model.smear_gate.weight, model.smear_lambda, model.backout_lambda], 0.2, (0.8, 0.95), 0.0)
    if hasattr(model, "loop_state_mod"):
        adam(list(model.loop_state_mod.parameters()), 0.003 * bscale, (0.9, 0.95), 0.0)

    allocated = {id(p) for g in groups for p in g["params"]}
    matrices, other = defaultdict(list), []
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:
            continue
        if p.ndim == 2:
            matrices[tuple(p.shape)].append(p)
        else:
            other.append(p)
    adam(other, adamw_lr, (0.9, 0.95), 0.0)
    for sh in sorted(matrices):
        groups.append(dict(kind="muon", params=matrices[sh], lr=muon_lr, initial_lr=muon_lr,
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
                  loops_eval: int, max_rows: int = 4096, collect_diag: bool = False):
    model._eval_loops_override = loops_eval
    model._collect_loop_diag = collect_diag
    model.eval()
    total_nats = total_bytes = total_toks = 0.0
    try:
        for i in range(0, min(max_rows, val_x.shape[0]), batch_size):
            xb = val_x[i:i + batch_size].to(device, dtype=torch.long)
            yb = val_y[i:i + batch_size].to(device, dtype=torch.long)
            model.set_row_offset(i)
            logits = model(xb)
            loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten(), ignore_index=-1, reduction="none")
            nbytes = byte_table[yb.flatten().clamp_min(0)]
            valid = (yb.flatten() >= 0) & (nbytes > 0)
            total_nats += loss[valid].sum().item()
            total_bytes += nbytes[valid].sum().item()
            total_toks += valid.sum().item()
            if collect_diag:
                model._collect_loop_diag = False
    finally:
        model._eval_loops_override = None
        model._collect_loop_diag = False
        model.train()
    return total_nats / max(total_toks, 1.0), total_nats / (math.log(2.0) * max(total_bytes, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=(
        "interloop_norm", "euler_only", "sphere_only", "prelude_nonorm",
        "prelude_no_invR", "scalar_g_ode", "scalar_g_nonorm", "prelude_true_ode",
        "r1_20L_iso_ve_prelude_did", "r1_20L_iso_ve_nonorm_did",
        "r1_20L_iso_ve_euler_did", "r1_20L_iso_ve_base",
        "huginn_spherical_ode", "huginn_nonorm_ode", "fixed_base",
    ))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tbptt-loops", type=int, default=0)
    ap.add_argument("--continuum-u412", action="store_true")
    ap.add_argument("--steps", type=int, default=2250)
    ap.add_argument("--eval-every", type=int, default=750)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-x0", action="store_true")
    ap.add_argument("--clean-x0-lr", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=Path("/dev/shm/rsi_arena_data_1b"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    py_rng = random.Random(args.seed)

    is_r1_did = args.idea in ("r1_20L_iso_ve_prelude_did", "r1_20L_iso_ve_nonorm_did",
                              "r1_20L_iso_ve_euler_did", "r1_20L_iso_ve_base")
    core_layers = 16 if is_r1_did else 4
    loops = 1 if is_r1_did else args.loops
    cfg = ModelConfig(
        method="fixed_recursive",
        width=768, heads=6, vocab_size=32768, sequence_len=1024, mlp_width=3072,
        layers=2 + core_layers + 2,
        pre_layers=2, core_layers=core_layers, post_layers=2,
        loops=loops, max_loops=max(2, loops), eval_loops=loops,
        huginn_mean=float(max(loops, 2)), huginn_backprop_loops=min(4, max(loops, 1)),
        window_pattern="SSSL", backend="reference", scale="debug",
    )
    model = build_model(cfg, device, seed=args.seed)
    ve_count = apply_iso_ve_4(model) if is_r1_did else len(model.value_embeds)
    if args.freeze_x0:
        model.x0_lambdas.data.zero_()
        model.x0_lambdas.requires_grad_(False)

    base_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sm = attach_wave16_executor(model, args.idea, device=device, tbptt_loops=args.tbptt_loops)
    extra_params = sum(p.numel() for p in sm.parameters() if p.requires_grad)

    train_d = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_d = torch.load(args.data_dir / "val.pt", map_location="cpu", weights_only=True)
    train_x, train_y = train_d["x"], train_d["y"]
    val_x, val_y = val_d["x"], val_d["y"]
    byte_table = torch.load("/dev/shm/nt-runtime/resources/tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    batch_rows = args.micro_batch * args.accum
    batch_tokens = batch_rows * cfg.sequence_len
    opt = build_official_opt(model, batch_tokens, clean_x0_lr=args.clean_x0_lr)

    val0, bpb0 = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, loops, max_rows=512)
    print(f"[{args.tag}|{args.idea}|R={loops}|ve={ve_count}|+p={extra_params} ({100*extra_params/base_params:.4f}%)] "
          f"step=0 val_loss={val0:.4f} val_bpb={bpb0:.5f}", flush=True)

    history = [dict(step=0, val_loss=val0, val_bpb=bpb0)]
    n_train = train_x.size(0)
    t0 = time.perf_counter()
    step_toks = 0

    for step in range(args.steps):
        m = lr_mult(step, args.steps)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * m
            if "initial_weight_decay" in g:
                g["weight_decay"] = g["initial_weight_decay"] * m
        if args.continuum_u412 and (not is_r1_did) and loops == 8:
            model._step_loops_override = py_rng.randint(4, 12)
        else:
            model._step_loops_override = None
        model.zero_grad(set_to_none=True)
        model.set_step(step)
        for a in range(args.accum):
            idx0 = (step * batch_rows + a * args.micro_batch) % (n_train - args.micro_batch)
            xb = train_x[idx0:idx0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            yb = train_y[idx0:idx0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * batch_rows + a * args.micro_batch)
            loss = model(xb, yb)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()
        step_toks += batch_tokens

        step1 = step + 1
        if step1 % args.eval_every == 0 or step1 == args.steps:
            model._step_loops_override = None
            rows = 4096 if step1 == args.steps else 1024
            vl, vb = eval_at_depth(model, val_x, val_y, byte_table, device, args.micro_batch, loops, max_rows=rows)
            history.append(dict(step=step1, val_loss=vl, val_bpb=vb))
            print(f"[{args.tag}] VAL step={step1} val_loss={vl:.5f} val_bpb={vb:.5f}", flush=True)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    tok_per_s = step_toks / max(1e-6, elapsed)

    ood_depths = [1] if is_r1_did else [1, 2, 4, 8, 12, 16, 32]
    ood_2048 = {}
    for ev_r in ood_depths:
        vl, _ = eval_at_depth(model, val_x, val_y, byte_table, device, min(args.micro_batch, 16),
                              ev_r, max_rows=2048, collect_diag=(ev_r == loops))
        ood_2048[str(ev_r)] = round(vl, 5)
    ood16_8 = None if is_r1_did else round(ood_2048["16"] - ood_2048["8"], 5)

    learned = {}
    if hasattr(sm, "s_ode"):
        learned["s_ode"] = round(float(sm.s_ode.detach().item()), 4)
    if hasattr(sm, "g_pre"):
        gp = sm.g_pre.detach().float()
        learned["g_pre_mean"] = round(float(gp.mean().item()), 4)
        learned["g_pre_std"] = round(float(gp.std().item()), 4) if gp.numel() > 1 else 0.0

    res = dict(
        tag=args.tag, idea=args.idea, loops=loops, steps=args.steps, seed=args.seed,
        ve_count=ve_count, base_params=base_params, extra_params=extra_params,
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        tok_per_s=tok_per_s, elapsed_s=elapsed,
        ood_2048=ood_2048, ood_delta_16_8=ood16_8,
        loop_diag=model._loop_diag, learned=learned,
        data_dir=str(args.data_dir), history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"[{args.tag}] DONE val={res['final_val_loss']:.5f} ood16-8={ood16_8} "
          f"ood2k={ood_2048} learned={learned} {tok_per_s/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
