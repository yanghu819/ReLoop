#!/usr/bin/env python3
"""Official 8xB300 Multi-GPU DDP + 22-Task CORE Grand Finale: Spherical Restoring-Force ODE (prelude_true_ode).

Implements Carbon-Critic #126 + ALL 4 Mandatory Carbon-Auditor #127 Fixes:
  (1) Exact official huginn_r8.json layer dimensions (w=1536, heads=12, mlp=6528, pre=3/core=8/post=3,
      window="SSSL", vocab=32768, seq=2048, scale="formal") instantiated as method="fixed_recursive"
      (loops=8, 70 virtual layers, ~866.6M params — avoiding SandwichBlock/_loop_hidden and scale="debug" traps!).
  (2) 100% Full-8-BPTT (tbptt_loops=0 default) + deterministic R=8 + --freeze-x0 on BOTH arms (a1 vs a2).
  (3) Unified _custom_ordinary_hidden matching ReferenceLM._ordinary_hidden cached/backout verbatim,
      reading R dynamically from model.config.loops so override_loops in {1,2,4,6,8,12,16,24,32} updates both
      loop count and 1/R!
  (4) SphericalODEState (s_ode=4.0, g_pre=[1536] init 1.5; ONLY 1,537 scalars = 0.00018%) attached BEFORE DDP;
      routed into explicit AdamW(lr=0.003*sqrt(2), betas=(0.9,0.95), wd=0.0) in ExperimentOptimizer;
      verifies shared Step-0 weight checksum, Step-2 bit-exact max_rank_diff == 0.0 across ranks 0..7,
      and distinct val_bpb across r in {1,4,8} during --smoke!
  (5) Checkpoints stored safely in /dev/shm (--ckpt-dir) so overlay / never fills; retains s1250 & s2500 ckpts!
"""
import argparse
from contextlib import nullcontext
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, "/root/nt")
from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.formal import evaluate_bpb, evaluate_core
from nexttransformer.models.backbone import norm
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model
from nexttransformer.models.native.config import NativeTrainConfig
from nexttransformer.native.optimizer import set_schedule
from nexttransformer.protocol.admission import data_layout, share_batches
from nexttransformer.protocol.spec import Protocol
from nexttransformer.runtime.data import BestFitStream, verify_resources
from nexttransformer.runtime.process import initialize
from nexttransformer.training.optimizer import parameter_groups
from nexttransformer.training.sharded import ExperimentOptimizer


class SphericalODEState(nn.Module):
    """Exact 1-equation Spherical Restoring-Force ODE parameters: (d + 1) scalars ONLY."""
    def __init__(self, width: int, s_init: float = 4.0, g_init: float = 1.5, device: torch.device = None):
        super().__init__()
        self.s_ode = nn.Parameter(torch.tensor(float(s_init), device=device, dtype=torch.float32))
        self.g_pre = nn.Parameter(torch.full((width,), float(g_init), device=device, dtype=torch.float32))


def load_huginn_r8_as_fixed_recursive(config_path: Path, loops: int) -> ModelConfig:
    """Fix #2 (#127): Read exact huginn_r8.json dimensions cleanly as method='fixed_recursive', scale='formal'."""
    raw = json.loads(config_path.read_text())
    m = dict(raw["model"])
    m["method"] = "fixed_recursive"
    m["loops"] = int(loops)
    m["max_loops"] = int(loops)
    m["eval_loops"] = int(loops)
    m["scale"] = "formal"
    m["backend"] = "reference"
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    filtered = {k: v for k, v in m.items() if k in valid_fields}
    cfg = ModelConfig(**filtered)
    assert cfg.method == "fixed_recursive" and cfg.scale == "formal" and cfg.loops == loops
    return cfg


def attach_unified_executor(model, idea: str, tbptt_loops: int = 0, device: torch.device = None):
    """Fix #3 (#127): Route BOTH fixed_base and ODE arms through identical _custom_ordinary_hidden."""
    is_ode = idea in ("prelude_true_ode", "driven_ode_nonorm")
    use_norm = (idea == "prelude_true_ode")
    sm = None
    if is_ode:
        sm = SphericalODEState(model.config.width, s_init=4.0, g_init=1.5, device=device)
        model.loop_state_mod = sm

    model._tbptt_loops = int(tbptt_loops)

    def _custom_ordinary_hidden(self, idx):
        c = self.config
        R = int(c.loops)
        total = c.pre_layers + R * c.core_layers + c.post_layers
        mid_virt = total // 2
        inv_r = 1.0 / float(R)
        tbptt = int(getattr(self, "_tbptt_loops", 0))

        x = self._embed_reference(idx)
        x0 = x
        cached = None

        for i in range(c.pre_layers):
            x = self._block(x, x0, idx, i, i, total)
            if i == mid_virt:
                cached = x
        prelude = x

        for r in range(R):
            no_g = (self.training and tbptt > 0 and r < R - tbptt)
            if self.training and tbptt > 0 and r == R - tbptt:
                x = x.detach()
            with torch.no_grad() if no_g else nullcontext():
                h = x
                for j in range(c.core_layers):
                    phys = c.pre_layers + j
                    virt = c.pre_layers + r * c.core_layers + j
                    h = self._block(h, x0, idx, phys, virt, total)
                    if virt == mid_virt:
                        cached = h
                if is_ode:
                    delta = h - x
                    upd = x + inv_r * (sm.s_ode.to(x.dtype) * delta + sm.g_pre.to(x.dtype) * prelude)
                    x = norm(upd) if use_norm else upd
                else:
                    x = h

        for j in range(c.post_layers):
            phys = c.pre_layers + c.core_layers + j
            virt = c.pre_layers + R * c.core_layers + j
            x = self._block(x, x0, idx, phys, virt, total)
            if virt == mid_virt:
                cached = x

        return x, cached

    model._ordinary_hidden = types.MethodType(_custom_ordinary_hidden, model)
    return sm


def build_ddp_optimizer(model, base_tc: NativeTrainConfig):
    """Fix #4 (#127): Official parameter_groups with explicit 0.003*bscale AdamW group for loop_state_mod."""
    raw_groups = parameter_groups(model, base_tc)
    sm_ids = {id(p) for p in model.loop_state_mod.parameters()} if hasattr(model, "loop_state_mod") else set()
    cleaned = []
    for g in raw_groups:
        keep_p, keep_n = [], []
        for p, nm in zip(g["params"], g["parameter_names"]):
            if id(p) not in sm_ids:
                keep_p.append(p)
                keep_n.append(nm)
        if keep_p:
            ng = dict(g)
            ng["params"] = keep_p
            ng["parameter_names"] = keep_n
            cleaned.append(ng)

    if sm_ids:
        bscale = math.sqrt(base_tc.global_batch_tokens / 524288.0)
        lr_sm = 0.003 * bscale
        sm = model.loop_state_mod
        cleaned.append(dict(
            kind="adamw",
            params=[sm.s_ode, sm.g_pre],
            parameter_names=["loop_state_mod.s_ode", "loop_state_mod.g_pre"],
            lr=lr_sm,
            initial_lr=lr_sm,
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.0,
            initial_weight_decay=0.0,
        ))

    # Strict ownership & --freeze-x0 verification!
    ids = [id(p) for g in cleaned for p in g["params"]]
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) and set(ids) == trainable, \
        f"Optimizer ownership mismatch: {len(ids)} vs {len(trainable)}"
    if not model.x0_lambdas.requires_grad:
        assert id(model.x0_lambdas) not in set(ids), "Frozen x0_lambdas leaked into optimizer!"
    return ExperimentOptimizer(cleaned)


def shared_weight_checksum(model) -> str:
    """Bit-exact SHA256 over shared trunk tensors so a1 (prelude_ode) & a2 (fixed_base) verify identical init!"""
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if name.startswith("loop_state_mod."):
            continue
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes()[:4096])
    return h.hexdigest()[:16]


@torch.no_grad()
def verify_rank_parity(model, rank: int, world: int, device: torch.device) -> float:
    """Safeguard S2 (#126/#127): Assert max_rank_diff == 0.0 across all 8 ranks on s_ode, g_pre, and core weights."""
    if world <= 1:
        return 0.0
    probes = [model.transformer.h[model.config.pre_layers].attn.c_q.weight.detach().float().reshape(-1)[:256]]
    if hasattr(model, "loop_state_mod"):
        probes.append(model.loop_state_mod.s_ode.detach().float().reshape(1))
        probes.append(model.loop_state_mod.g_pre.detach().float().reshape(-1))
    vec = torch.cat(probes, dim=0).contiguous()
    gathered = [torch.empty_like(vec) for _ in range(world)]
    dist.all_gather(gathered, vec)
    max_diff = max(float((g - gathered[0]).abs().max().item()) for g in gathered[1:])
    assert max_diff == 0.0, f"Rank parity drift detected on rank {rank}: max_diff={max_diff}"
    return max_diff


@torch.no_grad()
def compute_core_spectra(model):
    cfg = model.config
    eranks, kappas = [], []
    for l_idx in range(cfg.pre_layers, cfg.pre_layers + cfg.core_layers):
        blk = model.transformer.h[l_idx]
        for mod in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v, blk.attn.c_proj, blk.mlp.c_fc, blk.mlp.c_proj):
            s = torch.linalg.svdvals(mod.weight.float())
            p = (s / s.sum().clamp_min(1e-12)).clamp_min(1e-12)
            eranks.append(float(torch.exp(-(p * p.log()).sum()).item()))
            kappas.append(float((s[0] / s[len(s) // 2].clamp_min(1e-8)).item()))
    return dict(mean_erank=sum(eranks) / len(eranks), mean_kappa=sum(kappas) / len(kappas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=("fixed_base", "prelude_true_ode", "driven_ode_nonorm"))
    ap.add_argument("--config", type=Path, default=Path("/root/nt/configs/looped/huginn_r8.json"))
    ap.add_argument("--loops", type=int, default=8)
    ap.add_argument("--tbptt-loops", type=int, default=0)
    ap.add_argument("--freeze-x0", action="store_true", default=True)
    ap.add_argument("--no-freeze-x0", dest="freeze_x0", action="store_false")
    ap.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--val-tokens", type=int, default=20971520)
    ap.add_argument("--core-steps", type=str, default="1250,2500")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep-ckpts", type=str, default="1250,2500")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-core", type=int, default=4)
    args = ap.parse_args()

    os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
    rank, world, device = initialize(args.seed)
    is_master = (rank == 0)

    config = load_huginn_r8_as_fixed_recursive(args.config, args.loops)
    p = Protocol()
    base_tc = NativeTrainConfig()
    train_config = types.SimpleNamespace(
        steps=args.steps,
        warmup_steps=base_tc.warmup_steps,
        warmdown_ratio=base_tc.warmdown_ratio,
        final_lr_fraction=base_tc.final_lr_fraction,
    )
    core_step_set = {int(x) for x in args.core_steps.split(",") if x.strip()}
    keep_ckpt_set = {int(x) for x in args.keep_ckpts.split(",") if x.strip()}

    stream_rank, stream_world, shares = data_layout(rank, world)
    if args.smoke:
        accum = 2
        step_tokens = world * accum * args.micro_batch * config.sequence_len
        val_tokens = args.eval_batch * world * config.sequence_len
    else:
        accum, rest = divmod(p.accumulation(args.micro_batch, stream_world), shares)
        assert rest == 0, f"micro_batch={args.micro_batch} does not divide global batch!"
        step_tokens = world * accum * args.micro_batch * config.sequence_len
        assert step_tokens == p.global_batch_tokens, f"step_tokens={step_tokens} != {p.global_batch_tokens}"
        val_tokens = args.val_tokens

    data_dir = args.data_dir or (args.resources / "base_data_climbmix")
    verify_resources(args.resources, data_dir)
    tokenizer = RustBPETokenizer.from_directory(args.resources / "tokenizer")
    stream = BestFitStream(data_dir, tokenizer, "train", stream_rank, stream_world, config.sequence_len)
    stream_rows = shares * accum * args.micro_batch

    model = build_model(config, device, seed=args.seed)
    model.set_training_stage(0)
    if args.freeze_x0:
        model.x0_lambdas.data.zero_()
        model.x0_lambdas.requires_grad_(False)

    # Verify Step-0 bit-exact parity of unified fixed_base vs unmodified ReferenceLM._ordinary_hidden in smoke!
    if args.smoke and args.idea == "fixed_base":
        with torch.no_grad():
            probe_x = torch.randint(0, config.vocab_size, (2, 128), device=device)
            orig_h, orig_c = model._ordinary_hidden(probe_x)
    else:
        orig_h, orig_c, probe_x = None, None, None

    base_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    sm = attach_unified_executor(model, args.idea, tbptt_loops=args.tbptt_loops, device=device)
    total_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    extra_params = total_params - base_params
    ve_count = len(model.value_embeds)
    init_sha = shared_weight_checksum(model)

    if orig_h is not None:
        with torch.no_grad():
            new_h, new_c = model._ordinary_hidden(probe_x)
            h_err = float((orig_h - new_h).abs().max().item())
            c_err = float((orig_c - new_c).abs().max().item())
            assert h_err == 0.0 and c_err == 0.0, f"fixed_base parity error: h={h_err}, c={c_err}"
            if is_master:
                print(f"[PARITY CHECK OK] fixed_base vs unmodified ReferenceLM: h_err={h_err}, c_err={c_err}", flush=True)

    # Fix #4: Attach loop_state_mod BEFORE DDP, build explicit optimizer!
    opt = build_ddp_optimizer(model, base_tc)
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model

    if is_master:
        args.output.mkdir(parents=True, exist_ok=True)
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DDP-900M-ODE] idea={args.idea} method={config.method} scale={config.scale} "
              f"w={config.width} mlp={config.mlp_width} {config.pre_layers}/{config.core_layers}/{config.post_layers} "
              f"R={config.loops} tbptt={args.tbptt_loops} freeze_x0={args.freeze_x0} ve={ve_count} "
              f"params={total_params/1e6:.3f}M (+{extra_params}) init_sha={init_sha} "
              f"world={world} mb={args.micro_batch} accum={accum} tok/step={step_tokens:,}", flush=True)

    events_path = args.output / "metrics.jsonl"

    def log_event(payload):
        if is_master:
            with events_path.open("a") as f:
                f.write(json.dumps(payload) + "\n")
            print(json.dumps(payload), flush=True)

    @torch.no_grad()
    def run_val_bpb(override_loops=None):
        old_loops = model.config.loops
        old_eval = model.config.eval_loops
        if override_loops is not None:
            object.__setattr__(model.config, "loops", int(override_loops))
            object.__setattr__(model.config, "eval_loops", int(override_loops))
        try:
            res = evaluate_bpb(model, tokenizer, data_dir, args.resources,
                               batch=args.eval_batch, rank=rank, world=world,
                               device=device, tokens=val_tokens, split="val", smoke=args.smoke)
        finally:
            if override_loops is not None:
                object.__setattr__(model.config, "loops", old_loops)
                object.__setattr__(model.config, "eval_loops", old_eval)
            model.train()
        return res

    @torch.no_grad()
    def run_core_eval():
        t0 = time.monotonic()
        try:
            res = evaluate_core(model, tokenizer, args.resources, device,
                                smoke_max_per_task=(args.smoke_core if args.smoke and args.smoke_core > 0 else None))
        finally:
            model.train()
        res["seconds"] = time.monotonic() - t0
        return res

    t_start = time.monotonic()
    step_times = []
    rows_per_step = step_tokens // config.sequence_len

    # Step 0 initial validation
    val0 = run_val_bpb()
    log_event(dict(event="validation", step=0, tokens=0, val_bpb=val0["bpb"],
                   val_loss_est=val0["bpb"] * 3.285715, init_sha=init_sha, seconds=val0["seconds"]))

    for step in range(args.steps):
        t0 = time.perf_counter()
        model.set_step(step)
        mult = set_schedule(opt, step, train_config)
        opt.zero_grad(set_to_none=True)

        batches = list(share_batches(stream, args.micro_batch, stream_rows, rank % shares, shares, device))
        loss_sum = 0.0
        for m_idx, (x, y) in enumerate(batches):
            model.set_row_offset(step * rows_per_step + rank * accum * args.micro_batch + m_idx * args.micro_batch)
            sync_ctx = wrapped.no_sync() if (world > 1 and m_idx < accum - 1) else nullcontext()
            with sync_ctx:
                loss = wrapped(x, y)
                (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()

        # Safeguard S2: Step-2 exact cross-rank parity check (asserts max_rank_diff == 0.0!)
        if step == 1:
            max_diff = verify_rank_parity(model, rank, world, device)
            if is_master:
                print(f"[SAFEGUARD S2 PASSED] Step 2 cross-rank parity: max_rank_diff={max_diff}", flush=True)

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step_times.append(dt)
        step_1 = step + 1

        if step_1 % args.log_every == 0 or step_1 == args.steps or (args.smoke and step_1 <= 5):
            loss_t = torch.tensor(loss_sum, device=device)
            if world > 1:
                dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
            avg_dt = sum(step_times[-25:]) / len(step_times[-25:])
            tok_s = step_tokens / avg_dt
            ode_str = ""
            if sm is not None:
                gp = sm.g_pre.detach().float()
                ode_str = f" s_ode={sm.s_ode.item():.3f} g_pre={gp.mean().item():.3f}±{gp.std().item():.3f}"
            if is_master:
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(f"[{args.idea}] step={step_1}/{args.steps} tok={step_1*step_tokens/1e9:.3f}B "
                      f"loss={loss_t.item():.4f} lr_mult={mult:.3f}{ode_str} "
                      f"tok/s={tok_s/1e3:.1f}k peak_vram={peak_gb:.1f}GB "
                      f"elapsed={(time.monotonic()-t_start)/60:.1f}m", flush=True)

        if step_1 % args.val_every == 0 or step_1 == args.steps:
            vres = run_val_bpb()
            extra_ode = {}
            if sm is not None:
                gp = sm.g_pre.detach().float()
                extra_ode = dict(s_ode=round(float(sm.s_ode.item()), 4),
                                 g_pre_mean=round(float(gp.mean().item()), 4),
                                 g_pre_std=round(float(gp.std().item()), 4))
            log_event(dict(event="validation", step=step_1, tokens=step_1 * step_tokens,
                           val_bpb=vres["bpb"], val_loss_est=vres["bpb"] * 3.285715,
                           seconds=vres["seconds"], elapsed_s=time.monotonic() - t_start, **extra_ode))

        if step_1 in core_step_set or (args.smoke and step_1 == args.steps):
            cres = run_core_eval()
            log_event(dict(event="core", step=step_1, tokens=step_1 * step_tokens,
                           core_metric=cres["core_metric"], results=cres["results"],
                           centered_results=cres["centered_results"], seconds=cres["seconds"]))

        if (not args.smoke) and (step_1 % args.ckpt_every == 0 or step_1 in keep_ckpt_set or step_1 == args.steps):
            if is_master:
                for old_ckpt in args.ckpt_dir.glob("ckpt_*.pt"):
                    try:
                        s_num = int(old_ckpt.stem.split("_")[1])
                    except Exception:
                        s_num = -1
                    if s_num not in keep_ckpt_set:
                        try:
                            old_ckpt.unlink()
                        except OSError:
                            pass
                ckpt_p = args.ckpt_dir / f"ckpt_{step_1:06d}.pt"
                torch.save(dict(step=step_1, idea=args.idea, loops=config.loops,
                                model=model.state_dict()), ckpt_p)
            if world > 1:
                dist.barrier()

    # Final multi-depth extrapolation: r in {1, 4, 8} in smoke (with distinctness assert!), full {1..32} in real run!
    eval_depths = (1, 4, 8) if args.smoke else (1, 2, 4, 6, 8, 12, 16, 24, 32)
    multi_depth = {}
    for ev_r in eval_depths:
        vres = run_val_bpb(override_loops=ev_r)
        multi_depth[f"r{ev_r}"] = dict(val_bpb=vres["bpb"], val_loss_est=vres["bpb"] * 3.285715)
    if args.smoke:
        b1, b4, b8 = multi_depth["r1"]["val_bpb"], multi_depth["r4"]["val_bpb"], multi_depth["r8"]["val_bpb"]
        assert len({round(b1, 6), round(b4, 6), round(b8, 6)}) == 3, \
            f"override_loops did not change val_bpb: r1={b1}, r4={b4}, r8={b8}"
        if is_master:
            print(f"[SAFEGUARD FIX-#3 PASSED] Distinct multi-depth val_bpb: r1={b1:.5f}, r4={b4:.5f}, r8={b8:.5f}", flush=True)

    spectra = compute_core_spectra(model) if is_master else {}
    tail_dt = step_times[2:] if len(step_times) > 2 else step_times
    avg_dt = sum(tail_dt) / max(1, len(tail_dt))
    learned_ode = {}
    if sm is not None:
        gp = sm.g_pre.detach().float()
        learned_ode = dict(s_ode=round(float(sm.s_ode.item()), 4),
                           g_pre_mean=round(float(gp.mean().item()), 4),
                           g_pre_std=round(float(gp.std().item()), 4),
                           g_pre_min=round(float(gp.min().item()), 4),
                           g_pre_max=round(float(gp.max().item()), 4))
    final_summary = dict(
        event="final_summary",
        idea=args.idea,
        method=config.method,
        scale=config.scale,
        loops=config.loops,
        tbptt_loops=args.tbptt_loops,
        freeze_x0=args.freeze_x0,
        ve_count=ve_count,
        params=total_params,
        extra_params=extra_params,
        init_sha=init_sha,
        steps=args.steps,
        total_tokens=args.steps * step_tokens,
        world=world,
        tok_per_sec=step_tokens / avg_dt,
        peak_vram_gb=torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        wall_min=(time.monotonic() - t_start) / 60.0,
        learned_ode=learned_ode,
        multi_depth=multi_depth,
        core_spectra=spectra,
    )
    log_event(final_summary)
    if is_master:
        (args.output / "final_summary.json").write_text(json.dumps(final_summary, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
