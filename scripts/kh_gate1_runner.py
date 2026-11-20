#!/usr/bin/env python3
"""Minimal, zero-bloat Gate-1 Tuned-Baseline (core_lr_mult, core_wd_mult) Runner.

Directly tests Carbon-Auditor (#70) Gate 1:
  - Plain official 5-step NextTransformer MuonAdamW (ZERO custom tricks, ZERO eigh!)
  - Scales ONLY tied-core Muon (lr, wd) by (--core-lr-mult, --core-wd-mult)
  - Exact eval_loops=loops for non-Huginn (bug-free R=1 iso-param & iso-depth 2/16/2!)
  - Logs core weight Frobenius norm ||W||_F, kappa, erank, and val_loss/val_bpb.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/nt")
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model
from nexttransformer.training.optimizer import ExperimentOptimizer


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
def eval_bpb(model, val_x, val_y, byte_table, device, batch_size: int, max_rows: int = 4096):
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
        model.train()
    return total_nats / (math.log(2.0) * max(total_bytes, 1.0)), total_nats / max(total_toks, 1.0)


@torch.no_grad()
def core_diagnostics(model, first_l: int, core_layers: int):
    kappas, eranks, fnorms = [], [], []
    sample_js = sorted({0, core_layers // 2, core_layers - 1})
    for j in sample_js:
        blk = model.transformer.h[first_l + j]
        for p in (blk.attn.c_q.weight, blk.attn.c_k.weight, blk.attn.c_v.weight,
                  blk.attn.c_proj.weight, blk.mlp.c_fc.weight, blk.mlp.c_proj.weight):
            wf = p.float()
            fnorms.append(float(wf.norm().item()))
            sv = torch.linalg.svdvals(wf)
            med = sv[len(sv) // 2].clamp_min(1e-8).item()
            pe = sv.square() / sv.square().sum().clamp_min(1e-12)
            eranks.append(float((-(pe * pe.clamp_min(1e-12).log()).sum()).exp().item()))
            kappas.append(float(sv[0].item() / med))
    return dict(
        mean_fnorm=round(sum(fnorms) / len(fnorms), 3),
        mean_kappa=round(sum(kappas) / len(kappas), 3),
        mean_erank=round(sum(eranks) / len(eranks), 1),
    )


def build_opt(model, batch_tokens: int, core_ids: set, core_lr_mult: float, core_wd_mult: float, core_beta1: float = 0.95):
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
                               betas=betas, eps=1e-10, weight_decay=decay, initial_weight_decay=decay))

    adam(model.lm_head.parameters(), unemb_lr, (0.8, 0.96), 0.01)
    adam(model.transformer.wte.parameters(), emb_lr, (0.8, 0.995), 0.001)
    adam(model.value_embeds.parameters(), emb_lr * 0.5, (0.8, 0.995), 0.01)
    adam([model.resid_lambdas], scalar_lr * 0.01, (0.8, 0.95), 0.05)
    adam([model.x0_lambdas], scalar_lr, (0.96, 0.95), 0.0)
    adam([model.smear_gate.weight, model.smear_lambda, model.backout_lambda], 0.2, (0.8, 0.95), 0.0)

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
                           momentum=0.95, ns_steps=5, beta2=0.9,
                           weight_decay=wd, initial_weight_decay=wd))
    c_lr = muon_lr * core_lr_mult
    c_wd = wd * core_wd_mult
    for sh in sorted(core):
        groups.append(dict(kind="muon", params=core[sh], lr=c_lr, initial_lr=c_lr,
                           momentum=core_beta1, ns_steps=5, beta2=0.9,
                           weight_decay=c_wd, initial_weight_decay=c_wd))
    return ExperimentOptimizer(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--method", default="fixed_recursive", choices=["fixed_recursive", "huginn"])
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--pre-layers", type=int, default=2)
    ap.add_argument("--core-layers", type=int, default=4)
    ap.add_argument("--post-layers", type=int, default=2)
    ap.add_argument("--loops", type=int, default=4)
    ap.add_argument("--core-lr-mult", type=float, default=1.0)
    ap.add_argument("--core-wd-mult", type=float, default=1.0)
    ap.add_argument("--core-beta1", type=float, default=0.95)
    ap.add_argument("--freeze-x0", action="store_true")
    ap.add_argument("--steps", type=int, default=750)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--micro-batch", type=int, default=64)
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

    is_hug = (args.method == "huginn")
    bptt = args.loops if not is_hug else min(4, args.loops)
    cfg = ModelConfig(
        method=args.method, scale="debug", width=args.width, heads=args.heads,
        vocab_size=32768, sequence_len=seq_len, mlp_width=4 * args.width,
        layers=args.pre_layers + args.core_layers + args.post_layers,
        pre_layers=args.pre_layers, core_layers=args.core_layers, post_layers=args.post_layers,
        loops=args.loops,
        max_loops=max(args.loops, 2) if is_hug else args.loops,
        huginn_mean=float(max(args.loops, 2)), huginn_sigma=0.5,
        huginn_backprop_loops=max(bptt, 1),
        eval_loops=max(args.loops, 2) if is_hug else args.loops,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    if args.freeze_x0:
        with torch.no_grad():
            model.x0_lambdas.zero_()
        model.x0_lambdas.requires_grad_(False)
    core_ids = set()
    if is_hug and hasattr(model, "state_adapter") and model.state_adapter is not None and hasattr(model.state_adapter, "weight"):
        core_ids.add(id(model.state_adapter.weight))
    for l_idx in range(cfg.pre_layers, cfg.pre_layers + cfg.core_layers):
        blk = model.transformer.h[l_idx]
        for mod in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
            core_ids.add(id(mod.weight))

    batch_tokens = args.batch_size * seq_len
    accum = args.batch_size // args.micro_batch
    opt = build_opt(model, batch_tokens, core_ids, args.core_lr_mult, args.core_wd_mult, args.core_beta1)
    n_rows = train_x.shape[0]
    bpb0, vl0 = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=512)
    print(f"[{args.tag}] step=0 val_loss={vl0:.4f} val_bpb={bpb0:.5f}", flush=True)

    t0_all = time.time()
    history = []
    model.train()
    for step in range(args.steps):
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

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.steps:
            rows = 4096 if (step + 1) == args.steps else 1024
            vb, vl = eval_bpb(model, val_x, val_y, byte_table, device, args.micro_batch, max_rows=rows)
            history.append(dict(step=step + 1, val_loss=vl, val_bpb=vb, train_loss=loss_sum, elapsed_s=time.time() - t0_all))
            print(f"[{args.tag}] VAL step={step+1} val_loss={vl:.5f} val_bpb={vb:.5f}", flush=True)

    diag = core_diagnostics(model, cfg.pre_layers, cfg.core_layers)
    out = dict(
        tag=args.tag, method=args.method, core_layers=args.core_layers, loops=args.loops,
        core_lr_mult=args.core_lr_mult, core_wd_mult=args.core_wd_mult,
        final_val_loss=history[-1]["val_loss"], final_val_bpb=history[-1]["val_bpb"],
        mid_val_loss=history[0]["val_loss"], total_wall_s=time.time() - t0_all,
        core_diag=diag, history=history,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[{args.tag}] DONE val_loss={out['final_val_loss']:.5f} fnorm={diag['mean_fnorm']} kap={diag['mean_kappa']} {out['total_wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
