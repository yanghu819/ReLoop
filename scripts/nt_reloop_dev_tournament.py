"""Arena-v1 Compliant Dev-Split (shard_06541.parquet ONLY) Tournament for NextTransformer ReLoop.

Strictly obeys NextTransformer Arena Protocol v1 Rules #2 and #3:
  - Uses ONLY training shards (shard_00000+) and the official dev split `shard_06541.parquet`.
  - NEVER touches the reserved validation shard `shard_06542.parquet` or `CORE-22`.
  - Uses the UNMODIFIED official NextTransformer `build_model`, `build_optimizer`, `set_schedule`,
    `BestFitStream`, and `Protocol` hyperparameters (global_batch_tokens=1,048,576, seed=42, bf16).
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

NT_DIR = Path(os.environ.get("NEXTTRANSFORMER_DIR", "/dev/shm/nt-official"))
if str(NT_DIR) not in sys.path:
    sys.path.insert(0, str(NT_DIR))

from nanochat.common import COMPUTE_DTYPE
from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.models.config import ModelConfig, candidate_config
from nexttransformer.models.lm import build_model
from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.native.optimizer import set_schedule
from nexttransformer.protocol.spec import Protocol
from nexttransformer.runtime.data import BestFitStream
from nexttransformer.training.optimizer import build_optimizer

DEV_SHARD = "shard_06541.parquet"


def build_arm_config(arm: str) -> ModelConfig:
    if arm == "s0_fixed_rec_r2_sssl":
        return candidate_config("fixed_recursive")
    specs = {
        "s1_reloop_4_8_4_r2_sssl": dict(
            layers=16, mlp_width=6144, window_pattern="SSSL",
            options=dict(pre_layers=4, post_layers=4, loops=2, alpha_init=2.0, beta_init=0.4, use_clean_x0=True),
        ),
        "s2_reloop_6_6_6_r2_sssl": dict(
            layers=18, mlp_width=6144, window_pattern="SSSL",
            options=dict(pre_layers=6, post_layers=6, loops=2, alpha_init=2.0, beta_init=0.4, use_clean_x0=True),
        ),
        "s3_reloop_6_4_6_r3_sssl": dict(
            layers=16, mlp_width=6144, window_pattern="SSSL",
            options=dict(pre_layers=6, post_layers=6, loops=3, alpha_init=2.6, beta_init=0.4, use_clean_x0=True),
        ),
        "s4_reloop_7_4_7_r3_sssl": dict(
            layers=18, mlp_width=5376, window_pattern="SSSL",
            options=dict(pre_layers=7, post_layers=7, loops=3, alpha_init=2.6, beta_init=0.4, use_clean_x0=True),
        ),
        "s5_reloop_6_6_6_r2_pure2": dict(
            layers=18, mlp_width=6144, window_pattern="SSSL",
            options=dict(pre_layers=6, post_layers=6, loops=2, alpha_init=2.0, beta_init=0.4, use_clean_x0=False),
        ),
        "s6_reloop_6_4_6_r3_pure2": dict(
            layers=16, mlp_width=6144, window_pattern="SSSL",
            options=dict(pre_layers=6, post_layers=6, loops=3, alpha_init=2.6, beta_init=0.4, use_clean_x0=False),
        ),
        "s7_reloop_6_6_6_r2_L": dict(
            layers=18, mlp_width=5760, window_pattern="L",
            options=dict(pre_layers=6, post_layers=6, loops=2, alpha_init=2.0, beta_init=0.4, use_clean_x0=True),
        ),
        "s8_reloop_6_4_6_r3_L": dict(
            layers=16, mlp_width=5760, window_pattern="L",
            options=dict(pre_layers=6, post_layers=6, loops=3, alpha_init=2.6, beta_init=0.4, use_clean_x0=True),
        ),
    }
    if arm not in specs:
        raise ValueError(f"Unknown arm: {arm}")
    sp = specs[arm]
    return ModelConfig(
        method="reloop",
        layers=sp["layers"],
        mlp_width=sp["mlp_width"],
        window_pattern=sp["window_pattern"],
        options=sp["options"],
    )


def load_dev_batches(dev_dir: Path, tokenizer, seq_len: int, batch_size: int, tokens: int, rank: int, world: int):
    """Build deterministic dev-split batches EXCLUSIVELY from shard_06541.parquet (NEVER shard_06542)."""
    shard = (dev_dir / DEV_SHARD).resolve()
    assert shard.exists(), f"Missing official dev shard: {shard}"
    stream = BestFitStream(dev_dir, tokenizer, "val", rank, world, seq_len)
    # Replace paths BEFORE any document is read so shard_06542.parquet is never opened.
    stream.paths = [shard]
    assert [p.name for p in stream.paths] == [DEV_SHARD]
    rows_needed = max(1, math.ceil(tokens / (world * batch_size * seq_len)))
    cpu = torch.device("cpu")
    return [stream.batch(batch_size, cpu) for _ in range(rows_needed)]


@torch.no_grad()
def evaluate_dev_shard(model, dev_batches, byte_table, device, world: int):
    model.eval()
    t0 = time.perf_counter()
    nats_sum = torch.zeros((), device=device, dtype=torch.float64)
    bytes_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_sum = torch.zeros((), device=device, dtype=torch.float64)
    for x_cpu, y_cpu in dev_batches:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        loss = model(x, y, loss_reduction="none").float()
        valid = (y.reshape(-1) >= 0)
        nats_sum += loss[valid].sum().to(torch.float64)
        bytes_sum += byte_table[y.reshape(-1)[valid]].sum().to(torch.float64)
        tok_sum += valid.sum().to(torch.float64)
    if world > 1:
        dist.all_reduce(nats_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(bytes_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_sum, op=dist.ReduceOp.SUM)
    val_loss = float((nats_sum / tok_sum.clamp_min(1.0)).item())
    val_bpb = float((nats_sum / (bytes_sum.clamp_min(1.0) * math.log(2.0))).item())
    model.train()
    return dict(dev_loss=val_loss, dev_bpb=val_bpb, tokens=int(tok_sum.item()), seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--dev-tokens", type=int, default=20971520)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--dev-dir", type=Path, required=True)
    ap.add_argument("--resources", type=Path, default=Path(os.environ.get("NEXTTRANSFORMER_DIR", "./NextTransformer")) / "resources")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mlp-width", type=int, default=0, help="Optional override from official calibrate")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_master = (rank == 0)

    p = Protocol()
    cfg = build_arm_config(args.arm)
    if args.mlp_width > 0:
        cfg = replace(cfg, mlp_width=args.mlp_width)

    # Exact NextTransformer official train config scaled to args.steps
    tc = NativeTrainConfig(total_tokens=args.steps * p.global_batch_tokens)
    accum = p.global_batch_tokens // (world * args.micro_batch * cfg.sequence_len)
    assert world * accum * args.micro_batch * cfg.sequence_len == p.global_batch_tokens

    tokenizer = RustBPETokenizer.from_directory(args.resources / "tokenizer")
    byte_table = torch.load(args.resources / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    # Training stream (shard_00000+, explicitly excluding both DEV_SHARD and VALIDATION_SHARD)
    stream = BestFitStream(args.data_dir, tokenizer, "train", rank, world, cfg.sequence_len)
    stream.paths = [pt for pt in stream.paths if pt.name != DEV_SHARD]
    assert all(pt.name not in (DEV_SHARD, "shard_06542.parquet") for pt in stream.paths)
    dev_batches = load_dev_batches(args.dev_dir, tokenizer, cfg.sequence_len, args.micro_batch, args.dev_tokens, rank, world)

    model = build_model(cfg, device=device, seed=p.seed)
    model.set_training_stage(0)
    opt = build_optimizer(model, NativeTrainConfig())
    wrapped = DDP(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    if is_master:
        args.output.mkdir(parents=True, exist_ok=True)
        print(
            f"[DEV-TOURNAMENT] arm={args.arm} method={cfg.method} layers={cfg.n_layer} "
            f"mlp={cfg.mlp_width} win={cfg.window_pattern} opts={dict(cfg.options)} "
            f"params={n_params/1e6:.2f}M steps={args.steps} tok={args.steps*p.global_batch_tokens/1e9:.2f}B",
            flush=True,
        )

    events_path = args.output / "metrics.jsonl"

    def log_ev(d):
        if is_master:
            with events_path.open("a") as f:
                f.write(json.dumps(d) + "\n")
            print(json.dumps(d), flush=True)

    def get_scalars():
        out = {}
        if hasattr(model, "reloop_alpha"):
            out["alpha"] = round(float(model.reloop_alpha.item()), 4)
        if hasattr(model, "reloop_beta"):
            out["beta"] = round(float(model.reloop_beta.item()), 4)
        if getattr(model, "reloop_x0", None) is not None:
            out["x0_mean"] = round(float(model.reloop_x0.mean().item()), 4)
        return out

    d0 = evaluate_dev_shard(model, dev_batches, byte_table, device, world)
    log_ev(dict(event="dev", step=0, tokens=0, shard=DEV_SHARD, **d0, scalars=get_scalars()))

    rows_per_step = p.global_batch_tokens // cfg.sequence_len
    t_start = time.monotonic()
    tok_s_hist = []

    for step in range(args.steps):
        t0 = time.perf_counter()
        model.set_step(step)
        mult = set_schedule(opt, step, tc)
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for m_idx in range(accum):
            x, y = stream.batch(args.micro_batch, device)
            model.set_row_offset(step * rows_per_step + rank * accum * args.micro_batch + m_idx * args.micro_batch)
            ctx = wrapped.no_sync() if (world > 1 and m_idx < accum - 1) else torch.enable_grad()
            with ctx:
                loss = wrapped(x, y)
                (loss / accum).backward()
            loss_acc += float(loss.detach().item()) / accum
        opt.step()
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        tps = p.global_batch_tokens / dt
        if step >= 5:
            tok_s_hist.append(tps)

        if is_master and (step % 50 == 0 or step == args.steps - 1):
            vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            print(
                f"[{args.arm}] step={step+1}/{args.steps} tok={(step+1)*p.global_batch_tokens/1e9:.3f}B "
                f"loss={loss_acc:.4f} mult={mult:.3f} scalars={get_scalars()} "
                f"tok/s={tps/1e3:.1f}k vram={vram:.1f}GB el={(time.monotonic()-t_start)/60:.1f}m",
                flush=True,
            )

        if (step + 1) % args.val_every == 0 or (step + 1) == args.steps:
            dv = evaluate_dev_shard(model, dev_batches, byte_table, device, world)
            log_ev(dict(event="dev", step=step + 1, tokens=(step + 1) * p.global_batch_tokens,
                        shard=DEV_SHARD, **dv, scalars=get_scalars()))

    if is_master:
        summary = dict(
            arm=args.arm,
            method=cfg.method,
            layers=cfg.n_layer,
            mlp_width=cfg.mlp_width,
            window_pattern=cfg.window_pattern,
            options=dict(cfg.options),
            params=n_params,
            steps=args.steps,
            total_tokens=args.steps * p.global_batch_tokens,
            dev_shard=DEV_SHARD,
            final_dev_loss=dv["dev_loss"],
            final_dev_bpb=dv["dev_bpb"],
            scalars=get_scalars(),
            avg_tok_s=sum(tok_s_hist) / max(1, len(tok_s_hist)),
            wall_min=(time.monotonic() - t_start) / 60.0,
            peak_vram_gb=torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        )
        (args.output / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("[DEV-SUMMARY]", json.dumps(summary), flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
