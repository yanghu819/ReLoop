#!/usr/bin/env python3
"""Fast 8xB300 Test-Time Depth Extrapolation Sweep for Saved NextTransformer Looped Checkpoints.

Evaluates a saved `ckpt_*.pt` checkpoint across arbitrary `eval_loops` (e.g. 1,2,3,4,6,8,12,16)
on the official 20.97M-token ClimbMix validation shard + computes core weight spectra (erank, kappa).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, "/root/nt")
from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.formal import evaluate_bpb
from nexttransformer.models.lm import build_model
from nexttransformer.protocol.admission import review_config
from nexttransformer.runtime.data import verify_resources
from nexttransformer.runtime.process import initialize


@torch.no_grad()
def compute_core_spectra(model):
    cfg = model.config
    eranks, kappas = [], []
    by_proj = {}
    for pname in ("c_q", "c_k", "c_v", "c_proj", "c_fc", "mlp_proj"):
        by_proj[pname] = dict(erank=[], kappa=[], fnorm=[])
    for l_idx in range(cfg.pre_layers, cfg.pre_layers + cfg.core_layers):
        blk = model.transformer.h[l_idx]
        for pname, mod in (
            ("c_q", blk.attn.c_q),
            ("c_k", blk.attn.c_k),
            ("c_v", blk.attn.c_v),
            ("c_proj", blk.attn.c_proj),
            ("c_fc", blk.mlp.c_fc),
            ("mlp_proj", blk.mlp.c_proj),
        ):
            w = mod.weight.float()
            s = torch.linalg.svdvals(w)
            p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
            er = float(torch.exp(-(p * p.log()).sum()).item())
            kp = float((s[0] / s[len(s) // 2].clamp_min(1e-8)).item())
            fn = float(w.norm().item())
            eranks.append(er)
            kappas.append(kp)
            by_proj[pname]["erank"].append(er)
            by_proj[pname]["kappa"].append(kp)
            by_proj[pname]["fnorm"].append(fn)
    proj_summary = {
        k: dict(
            mean_erank=sum(v["erank"]) / len(v["erank"]),
            mean_kappa=sum(v["kappa"]) / len(v["kappa"]),
            mean_fnorm=sum(v["fnorm"]) / len(v["fnorm"]),
        )
        for k, v in by_proj.items()
    }
    return dict(
        mean_erank=sum(eranks) / len(eranks),
        mean_kappa=sum(kappas) / len(kappas),
        by_proj=proj_summary,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--depths", type=str, default="1,2,3,4,6,8,12,16")
    ap.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--val-tokens", type=int, default=20971520)
    args = ap.parse_args()

    os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
    rank, world, device = initialize(42)
    is_master = (rank == 0)

    config, _, _ = review_config(args.config)
    data_dir = args.data_dir or (args.resources / "base_data_climbmix")
    verify_resources(args.resources, data_dir)
    tokenizer = RustBPETokenizer.from_directory(args.resources / "tokenizer")

    model = build_model(config, device, seed=42)
    payload = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    results = {}
    t0 = time.monotonic()
    for d_r in depths:
        object.__setattr__(model.config, "loops", d_r)
        if getattr(model.config, "eval_loops", None) is not None:
            object.__setattr__(model.config, "eval_loops", d_r)
        vres = evaluate_bpb(
            model, tokenizer, data_dir, args.resources,
            batch=args.eval_batch, rank=rank, world=world,
            device=device, tokens=args.val_tokens, split="val", smoke=False,
        )
        results[f"r{d_r}"] = dict(
            val_bpb=vres["bpb"],
            val_loss_est=vres["bpb"] * 3.285715,
            seconds=vres["seconds"],
        )
        if is_master:
            print(f"[DEPTH r={d_r:2d}] val_bpb={vres['bpb']:.5f} val_loss={vres['bpb']*3.285715:.5f} ({vres['seconds']:.1f}s)", flush=True)

    spectra = compute_core_spectra(model) if is_master else {}
    out = dict(
        ckpt=str(args.ckpt),
        idea=payload.get("idea", "unknown"),
        step=payload.get("step", -1),
        method=config.method,
        depths=results,
        core_spectra=spectra,
        total_seconds=time.monotonic() - t0,
    )
    if is_master:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[SAVED] {args.output} ({out['total_seconds']:.1f}s)", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
