#!/usr/bin/env python3
"""Official 8xB300 Multi-GPU DDP + 22-Task CORE Grand Finale for ifm-ai/LoRM 778M (24L, w=1536, h=12, mlp=4096).

Implements ALL pre-run audit (#167 + #168) Safeguards:
  - Verbatim vendor/lorm core (`Transformer`, `FeedbackTransformer`, `NorMuon`, `SampledPasses`,
    `training_flops_per_token`, `_CrossEntropy`, `initialize`).
  - Safeguard D1 (#168 lethal NorMuon fix): NO DDP wrapper & NO fake 0.0*fusion.sum() phantom updates!
    Explicit `dist.all_reduce(p.grad, AVG)` over active grads only, identical `SampledPasses(..., seed=args.seed)`
    across all 8 ranks, and Step-3 `verify_rank_parity` asserting `max_rank_diff == 0.0` after BOTH passes=1 & passes=2!
  - Safeguard D2: Thin `LoRMEvalAdapter` plugging directly into official `evaluate_bpb` (20.97M val shard across
    full P in {1,2,4,8} or R in {1,2,3,4}) and official 22-task `evaluate_core` (at P=4 for LoRM/Sandwich, R=2 for 2D ODE).
  - Safeguard D3: Exact F=4 fixed-pool support (`--unique-steps K`), caching exact U unique tokens on CPU RAM
    so extra Iso-FLOP steps strictly repeat U without leaking into fresh tokens.
  - Exact LoRM 778M hyperparameters (runs.csv): dim=1536, n_layers=24, n_heads=12, hidden_dim=4096, seq_len=2048,
    lr=0.0173, wd=0.0049, embedding_lr=lr*0.90509668, head_lr=lr*0.08, beta2=0.95, mom_warmup=300,
    global batch = 256 seq x 2048 = 524,288 tok/step, programmatic exact C_total Iso-FLOP!
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types
from typing import ClassVar

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
for cand in (REPO_ROOT / "vendor", Path(__file__).resolve().parent.parent / "vendor"):
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
sys.path.insert(0, os.environ.get("NEXTTRANSFORMER_DIR", "./NextTransformer"))

from lorm.model.config import InstanceConfig, ModelConfig
from lorm.model.feedback import FeedbackConfig, FeedbackTransformer
from lorm.model.transformer import Transformer, initialize as lorm_initialize
from lorm.train.algo import PassMix, SampledPasses, training_flops_per_token
from lorm.train.loss import _CrossEntropy
from lorm.train.optim import build_optimizers

from nanochat.tokenizer import RustBPETokenizer
from nexttransformer.evaluation.formal import evaluate_bpb, evaluate_core
from nexttransformer.protocol.admission import data_layout, share_batches
from nexttransformer.runtime.data import BestFitStream, verify_resources
from nexttransformer.runtime.process import initialize as dist_initialize


# ---------------------------------------------------------------------------
# 778M Model Variants (Verbatim vendor/lorm core!)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sandwich778Config(FeedbackConfig):
    name: ClassVar[str] = "sandwich_778m"
    k_in: int = 0
    k_tap: int = 12
    use_c1: bool = True
    alpha_init: float = 2.0


class Sandwich778Transformer(FeedbackTransformer):
    config_class = Sandwich778Config

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        inst = config.instance
        assert 0 <= inst.k_in < inst.k_tap <= config.n_layers
        self.fb_alpha = nn.Parameter(torch.tensor(float(inst.alpha_init))) if inst.use_c1 else None

    def preserved_parameters(self) -> list[nn.Parameter]:
        return [self.fb_alpha] if self.fb_alpha is not None else []

    def mix(self, base: Tensor, tapped: Tensor) -> Tensor:
        dim = self.config.dim
        f_raw = self.fusion(F.rms_norm(tapped, (dim,), eps=self.config.norm_eps))
        if self.fb_alpha is not None:
            scale = (self.fb_alpha * (dim ** -0.5)).to(dtype=base.dtype)
            fed = f_raw * scale
        else:
            fed = f_raw
        return torch.cat([base[:, :1], base[:, 1:] + fed[:, :-1]], dim=1)

    def forward(self, input_ids: Tensor, *, passes: int | None = None) -> Tensor:
        passes = self.passes_of(passes)
        inst = self.config.instance
        k_in, k_tap = int(inst.k_in), int(inst.k_tap)
        x, cos, sin = self.embed(input_ids)
        h_in = self.blocks(x, cos, sin, 0, k_in) if k_in > 0 else x
        tapped = None
        for index in range(passes):
            is_last = (index == passes - 1)
            with torch.set_grad_enabled(torch.is_grad_enabled() and is_last):
                base = h_in if is_last else h_in.detach()
                state = base if index == 0 else self.mix(base, tapped.detach())
                if is_last:
                    top = self.blocks(state, cos, sin, k_in, None)
                else:
                    tapped = self.blocks(state, cos, sin, k_in, k_tap)
        return self.readout(top)

    def _window_blocks_macs(self, shape: tuple[int, int]) -> int:
        batch, length = shape
        dim = self.config.dim
        inst = self.config.instance
        return sum(b.forward_macs((batch, length, dim)) for b in self.backbone.blocks[inst.k_in:inst.k_tap])

    def feedback_macs(self, shape: tuple[int, int], passes: int) -> int:
        batch, length = shape
        return (passes - 1) * (self._window_blocks_macs(shape) + batch * length * self.fusion.weight.numel())

    def no_grad_macs(self, shape: tuple[int, int], *, passes: int | None = None, **options) -> int:
        passes = self.passes_of(passes)
        if passes == 1:
            return 0
        batch, length = shape
        fusion = batch * length * self.fusion.weight.numel()
        return (passes - 1) * self._window_blocks_macs(shape) + (passes - 2) * fusion + fusion // 2


@dataclass(frozen=True)
class ODE2DH778Config(InstanceConfig):
    name: ClassVar[str] = "isoparam_2d_ode_h_778m"
    pre_layers: int = 8
    core_layers: int = 8
    post_layers: int = 8
    loops: int = 2
    s_ode_init: float = 1.0
    g_pre_init: float = 0.5
    g_fb_init: float = 12.0


class ODE2DH778Transformer(Transformer):
    """778M (8/8/8 R=2) Unified 2D Spatio-Temporal Driven ODE (x->h fixed, 100% bit-exact Train == AR-Decode)."""
    config_class = ODE2DH778Config

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        inst = config.instance
        assert inst.pre_layers + inst.core_layers + inst.post_layers == config.n_layers
        self.fusion = nn.Linear(config.dim, config.dim, bias=False)
        self.s_ode = nn.Parameter(torch.tensor(float(inst.s_ode_init)))
        self.g_pre = nn.Parameter(torch.tensor(float(inst.g_pre_init)))
        self.g_fb = nn.Parameter(torch.tensor(float(inst.g_fb_init)))

    def hidden_matrices(self) -> list[nn.Parameter]:
        return super().hidden_matrices() + [self.fusion.weight]

    def preserved_parameters(self) -> list[nn.Parameter]:
        return [self.s_ode, self.g_pre, self.g_fb]

    def forward(self, input_ids: Tensor, *, loops: int | None = None) -> Tensor:
        inst = self.config.instance
        r_eff = inst.loops if loops is None else int(loops)
        dim = self.config.dim
        x, cos, sin = self.embed(input_ids)
        x = self.blocks(x, cos, sin, 0, inst.pre_layers)
        x_pre = x
        inv_r = 1.0 / float(r_eff)
        s_step = (self.s_ode * inv_r).to(dtype=x.dtype)
        g_step = (self.g_pre * inv_r).to(dtype=x.dtype)
        fb_step = (self.g_fb * (dim ** -0.5) * inv_r).to(dtype=x.dtype)
        c_start, c_end = inst.pre_layers, inst.pre_layers + inst.core_layers
        for _ in range(r_eff):
            h = self.blocks(x, cos, sin, c_start, c_end)
            fb_raw = self.fusion(F.rms_norm(h, (dim,), eps=self.config.norm_eps))
            fb_shift = F.pad(fb_raw[:, :-1], (0, 0, 1, 0))
            x = x + s_step * (h - x) + g_step * x_pre + fb_step * fb_shift
        x = self.blocks(x, cos, sin, c_end, None)
        return self.readout(x)

    def forward_macs(self, shape: tuple[int, int], *, loops: int | None = None, **options) -> int:
        inst = self.config.instance
        r_eff = inst.loops if loops is None else int(loops)
        batch, length = shape
        dim = self.config.dim
        c_start, c_end = inst.pre_layers, inst.pre_layers + inst.core_layers
        pre_m = sum(b.forward_macs((batch, length, dim)) for b in self.backbone.blocks[:c_start])
        core_m = sum(b.forward_macs((batch, length, dim)) for b in self.backbone.blocks[c_start:c_end])
        post_m = sum(b.forward_macs((batch, length, dim)) for b in self.backbone.blocks[c_end:])
        return pre_m + r_eff * core_m + post_m + r_eff * batch * length * self.fusion.weight.numel() + batch * length * self.head.weight.numel()


class LoRMEvalAdapter(nn.Module):
    """Thin adapter connecting vendor/lorm models to official NextTransformer evaluate_bpb & evaluate_core."""
    def __init__(self, model: Transformer, device: torch.device, seq_len: int, vocab_size: int):
        super().__init__()
        self.model = model
        self._device = device
        self.config = types.SimpleNamespace(sequence_len=seq_len, vocab_size=vocab_size, scale="formal")
        self.row_offset = 0
        self.fwd_kwargs: dict = {}

    def get_device(self) -> torch.device:
        return self._device

    def set_row_offset(self, offset: int) -> None:
        self.row_offset = int(offset)

    def forward(self, idx: Tensor, targets: Tensor | None = None, loss_reduction: str = "mean"):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(idx, **self.fwd_kwargs)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                               ignore_index=-1, reduction=loss_reduction)
        if loss_reduction == "none":
            return loss.view(targets.shape)
        return loss


ARMS = (
    "vanilla_778m",
    "lorm_verbatim_778m",
    "c1_tap_l12_778m",
    "c1_tap_l16_778m",
    "c1_sandwich_4_16_778m",
    "c1_sandwich_4_16_curr_778m",
    "isoparam_2d_ode_h_778m",
)


def make_778m_model(arm: str, seq_len: int = 2048, vocab_size: int = 32768):
    base_kw = dict(
        vocab_size=vocab_size, dim=1536, n_heads=12, n_kv_heads=12, hidden_dim=4096,
        max_seq_len=seq_len, n_layers=24, tie_embeddings=False, rope_theta=10000.0,
        norm_eps=1e-5, attention_backend="flash", qk_norm=True,
    )
    if arm == "vanilla_778m":
        return Transformer(ModelConfig(instance=None, **base_kw)), "vanilla", None
    if arm == "lorm_verbatim_778m":
        return FeedbackTransformer(ModelConfig(instance=FeedbackConfig(passes=2), **base_kw)), "feedback", (0.15, 0.85)
    if arm == "c1_tap_l12_778m":
        cfg = ModelConfig(instance=Sandwich778Config(passes=2, k_in=0, k_tap=12, use_c1=True, alpha_init=2.0), **base_kw)
        return Sandwich778Transformer(cfg), "feedback", (0.15, 0.85)
    if arm == "c1_tap_l16_778m":
        cfg = ModelConfig(instance=Sandwich778Config(passes=2, k_in=0, k_tap=16, use_c1=True, alpha_init=2.0), **base_kw)
        return Sandwich778Transformer(cfg), "feedback", (0.15, 0.85)
    if arm == "c1_sandwich_4_16_778m":
        # #167/#168 width-scaling fix: alpha_init = 0.85 * sqrt(1536) = 33.31 (zero climb lag!)
        cfg = ModelConfig(instance=Sandwich778Config(passes=2, k_in=4, k_tap=16, use_c1=True, alpha_init=33.3), **base_kw)
        return Sandwich778Transformer(cfg), "feedback", (0.15, 0.85)
    if arm == "c1_sandwich_4_16_curr_778m":
        cfg = ModelConfig(instance=Sandwich778Config(passes=2, k_in=4, k_tap=16, use_c1=True, alpha_init=33.3), **base_kw)
        return Sandwich778Transformer(cfg), "feedback", "curr_warm10_late40"
    if arm == "isoparam_2d_ode_h_778m":
        cfg = ModelConfig(instance=ODE2DH778Config(), **base_kw)
        return ODE2DH778Transformer(cfg), "ode2d", None
    raise ValueError(f"Unknown arm: {arm}")


def expected_flops(model: Transformer, seq_len: int, kind: str, sched) -> float:
    if kind != "feedback":
        return float(training_flops_per_token(model, seq_len))
    f1 = float(training_flops_per_token(model, seq_len, passes=1))
    f2 = float(training_flops_per_token(model, seq_len, passes=2))
    if sched == "curr_warm10_late40":
        p2 = 0.60 * 0.10 + 0.40 * 0.80
        return (1.0 - p2) * f1 + p2 * f2
    p1, p2 = sched
    return p1 * f1 + p2 * f2


def wsd_mult(step: int, max_steps: int, warmup: int, decay: int) -> float:
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    if decay > 0 and step >= max_steps - decay:
        rem = max(0, max_steps - step)
        return float(rem) / float(decay)
    return 1.0


def build_lorm_optimizers(model: Transformer, base_lr: float, wd: float, gate_lr: float,
                          momentum_warmup: int) -> dict[str, torch.optim.Optimizer]:
    opts = build_optimizers(
        model,
        lr=base_lr,
        weight_decay=wd,
        embedding_lr=base_lr * 0.9050966799189999,
        head_lr=base_lr * 0.08,
        embedding_weight_decay=0.0,
        adamw={"betas": (0.9, 0.95), "eps": 1e-8},
        muon={"momentum": 0.95, "momentum_start": 0.85, "momentum_warmup_steps": momentum_warmup, "beta2": 0.95},
        matrix_optimizer="lorm.train.optim.normuon:NorMuon",
    )
    preserved = set(model.preserved_parameters())
    if preserved and "adamw" in opts:
        adamw = opts["adamw"]
        gate_params = []
        for g in adamw.param_groups:
            keep = [p for p in g["params"] if p not in preserved]
            found = [p for p in g["params"] if p in preserved]
            g["params"] = keep
            gate_params.extend(found)
        adamw.param_groups = [g for g in adamw.param_groups if g["params"]]
        if gate_params:
            adamw.add_param_group({"params": gate_params, "lr": gate_lr, "weight_decay": 0.0})
    for opt in opts.values():
        for g in opt.param_groups:
            g["initial_lr"] = g["lr"]
    return opts


def shared_trunk_sha(model: Transformer) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if name in ("fb_alpha", "s_ode", "g_pre", "g_fb") or name.startswith("fusion."):
            continue
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes()[:2048])
    return h.hexdigest()[:16]


@torch.no_grad()
def verify_rank_parity(model: Transformer, rank: int, world: int, device: torch.device) -> float:
    if world <= 1:
        return 0.0
    probes = [model.backbone.blocks[0].attention.q_proj.weight.detach().float().reshape(-1)[:256]]
    if hasattr(model, "fusion"):
        probes.append(model.fusion.weight.detach().float().reshape(-1)[:256])
    for nm in ("fb_alpha", "s_ode", "g_pre", "g_fb"):
        p = getattr(model, nm, None)
        if p is not None:
            probes.append(p.detach().float().reshape(-1))
    vec = torch.cat(probes, dim=0).contiguous()
    gathered = [torch.empty_like(vec) for _ in range(world)]
    dist.all_gather(gathered, vec)
    max_diff = max(float((g - gathered[0]).abs().max().item()) for g in gathered[1:])
    assert max_diff == 0.0, f"Rank parity drift on rank {rank}: max_diff={max_diff}"
    return max_diff


def gate_summary(model: nn.Module) -> dict[str, float]:
    out = {}
    for name in ("fb_alpha", "s_ode", "g_pre", "g_fb"):
        p = getattr(model, name, None)
        if p is not None:
            out[name] = round(float(p.detach().item()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--ref-steps", type=int, default=4000, help="Reference steps for lorm_verbatim_778m (4000 = 2.097B tok)")
    ap.add_argument("--unique-steps", type=int, default=0, help="0=F=1 fresh stream; >0=cache exact U=K*524288 unique tokens for F>1")
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2, help="Global batch = world(8)*16*2 = 256 seq x 2048 = 524,288 tok/step")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-tokens", type=int, default=20971520)
    ap.add_argument("--lr", type=float, default=0.0173)
    ap.add_argument("--wd", type=float, default=0.0049)
    ap.add_argument("--gate-lr", type=float, default=0.0173)
    ap.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-core", type=int, default=16)
    args = ap.parse_args()

    os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
    rank, world, device = dist_initialize(args.seed)
    is_master = (rank == 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    seq_len = 2048
    vocab_size = 32768
    data_dir = args.data_dir or (args.resources / "base_data_climbmix")
    verify_resources(args.resources, data_dir)
    tokenizer = RustBPETokenizer.from_directory(args.resources / "tokenizer")

    # Exact programmatic Iso-FLOP against verbatim LoRM 778M
    ref_model, ref_kind, ref_sched = make_778m_model("lorm_verbatim_778m", seq_len, vocab_size)
    e_flops_ref = expected_flops(ref_model, seq_len, ref_kind, ref_sched)
    del ref_model

    model, kind, sched = make_778m_model(args.arm, seq_len, vocab_size)
    lorm_initialize(model, std=0.02)
    model = model.to(device=device)
    e_flops_arm = expected_flops(model, seq_len, kind, sched)
    f1_tok = float(training_flops_per_token(model, seq_len, passes=1)) if kind == "feedback" else e_flops_arm
    f2_tok = float(training_flops_per_token(model, seq_len, passes=2)) if kind == "feedback" else e_flops_arm

    accum = 2 if args.smoke else args.accum
    step_tokens = world * args.micro_batch * accum * seq_len
    c_total = args.ref_steps * step_tokens * e_flops_ref
    max_steps = 4 if args.smoke else max(1, round(c_total / (step_tokens * e_flops_arm)))
    warmup_steps = 1 if args.smoke else max(1, round(0.05 * max_steps))
    decay_steps = 1 if args.smoke else max(1, round(0.20 * max_steps))
    mom_warmup = 2 if args.smoke else min(300, max(1, round(0.075 * max_steps)))
    val_tokens = (args.eval_batch * world * seq_len) if args.smoke else args.val_tokens

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    init_sha = shared_trunk_sha(model)
    opts = build_lorm_optimizers(model, args.lr, args.wd, args.gate_lr, mom_warmup)

    if not args.no_compile:
        for i, blk in enumerate(model.backbone.blocks):
            model.backbone.blocks[i] = torch.compile(blk)

    adapter = LoRMEvalAdapter(model, device, seq_len, vocab_size)
    stream_rank, stream_world, shares = data_layout(rank, world)
    stream = BestFitStream(data_dir, tokenizer, "train", stream_rank, stream_world, seq_len)
    stream_rows = shares * accum * args.micro_batch

    # Safeguard D3: For F>1 (--unique-steps K), pre-cache exact K steps on CPU RAM (uint16, ~130MB/rank for K=1000!)
    cached_pool: list[list[tuple[Tensor, Tensor]]] | None = None
    if args.unique_steps > 0:
        cached_pool = []
        for _ in range(args.unique_steps):
            step_mbs = []
            for x, y in share_batches(stream, args.micro_batch, stream_rows, rank % shares, shares, torch.device("cpu")):
                step_mbs.append((x.to(torch.int32).contiguous(), y.to(torch.int32).contiguous()))
            cached_pool.append(step_mbs)
        if is_master:
            print(f"[SAFEGUARD D3] Cached exact U={args.unique_steps * step_tokens / 1e6:.2f}M unique tokens ({args.unique_steps} steps) on CPU RAM.", flush=True)

    # Safeguard D1: Identical seed across ALL 8 ranks (NEVER seed+rank!)
    algo_static = SampledPasses(PassMix(list(sched)), seed=args.seed) if (kind == "feedback" and sched != "curr_warm10_late40") else None
    algo_early = SampledPasses(PassMix([0.90, 0.10]), seed=args.seed) if sched == "curr_warm10_late40" else None
    algo_late = SampledPasses(PassMix([0.20, 0.80]), seed=args.seed + 1000) if sched == "curr_warm10_late40" else None
    switch_step = round(0.60 * max_steps)

    if is_master:
        args.output.mkdir(parents=True, exist_ok=True)
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[DDP-778M-LoRM] arm={args.arm} kind={kind} params={n_params/1e6:.3f}M init_sha={init_sha} "
            f"e_flops/tok={e_flops_arm/1e9:.4f}G (ratio={e_flops_arm/e_flops_ref:.4f}x) "
            f"max_steps={max_steps} tok={max_steps*step_tokens/1e9:.3f}B unique_steps={args.unique_steps} "
            f"world={world} mb={args.micro_batch} accum={accum} tok/step={step_tokens:,}",
            flush=True,
        )

    events_path = args.output / "metrics.jsonl"

    def log_event(payload: dict):
        if is_master:
            with events_path.open("a") as f:
                f.write(json.dumps(payload) + "\n")
            print(json.dumps(payload), flush=True)

    @torch.no_grad()
    def eval_bpb_cfg(**fwd_kwargs) -> dict:
        adapter.fwd_kwargs = fwd_kwargs
        try:
            res = evaluate_bpb(adapter, tokenizer, data_dir, args.resources,
                               batch=args.eval_batch, rank=rank, world=world,
                               device=device, tokens=val_tokens, split="val", smoke=args.smoke)
        finally:
            adapter.fwd_kwargs = {}
            model.train()
        return res

    @torch.no_grad()
    def eval_core_cfg(**fwd_kwargs) -> dict:
        adapter.fwd_kwargs = fwd_kwargs
        t0 = time.monotonic()
        try:
            res = evaluate_core(adapter, tokenizer, args.resources, device,
                                smoke_max_per_task=(args.smoke_core if args.smoke else None))
        finally:
            adapter.fwd_kwargs = {}
            model.train()
        res["seconds"] = time.monotonic() - t0
        return res

    def eval_primary_bpb(step_num: int):
        if kind == "feedback":
            r4 = eval_bpb_cfg(passes=4)
            log_event(dict(event="validation", step=step_num, tokens=step_num * step_tokens,
                           val_bpb=r4["bpb"], val_loss_est=r4["bpb"] * 3.285715, passes=4,
                           gates=gate_summary(model), seconds=round(r4["seconds"], 2)))
        elif kind == "ode2d":
            r2 = eval_bpb_cfg(loops=2)
            log_event(dict(event="validation", step=step_num, tokens=step_num * step_tokens,
                           val_bpb=r2["bpb"], val_loss_est=r2["bpb"] * 3.285715, loops=2,
                           gates=gate_summary(model), seconds=round(r2["seconds"], 2)))
        else:
            r1 = eval_bpb_cfg()
            log_event(dict(event="validation", step=step_num, tokens=step_num * step_tokens,
                           val_bpb=r1["bpb"], val_loss_est=r1["bpb"] * 3.285715,
                           gates={}, seconds=round(r1["seconds"], 2)))

    eval_primary_bpb(0)

    t_start = time.monotonic()
    realized_flops = 0.0
    p2_count = 0
    step_times = []

    for step in range(max_steps):
        t0 = time.perf_counter()
        mult = wsd_mult(step, max_steps, warmup_steps, decay_steps)
        for opt in opts.values():
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * mult
            opt.zero_grad(set_to_none=True)

        if kind == "feedback":
            if args.smoke:
                # Force BOTH passes=1 (step 0, 2) and passes=2 (step 1, 3) to verify Safeguard D1!
                cur_passes = 1 if (step % 2 == 0) else 2
            elif sched == "curr_warm10_late40":
                alg = algo_early if step < switch_step else algo_late
                alg.begin_update(step)
                cur_passes = alg.passes
            else:
                algo_static.begin_update(step)
                cur_passes = algo_static.passes
            realized_flops += step_tokens * (f2_tok if cur_passes == 2 else f1_tok)
            if cur_passes == 2:
                p2_count += 1
        else:
            cur_passes = None
            realized_flops += step_tokens * e_flops_arm

        if cached_pool is not None:
            raw_mbs = cached_pool[step % len(cached_pool)]
            batches = [(x.to(device, dtype=torch.long, non_blocking=True),
                        y.to(device, dtype=torch.long, non_blocking=True)) for x, y in raw_mbs]
        else:
            batches = list(share_batches(stream, args.micro_batch, stream_rows, rank % shares, shares, device))

        loss_sum = 0.0
        for x, y in batches:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, passes=cur_passes) if kind == "feedback" else model(x)
            targets = y.reshape(-1)
            valid_mask = targets >= 0
            n_valid = valid_mask.sum().clamp_min(1)
            safe_targets = torch.where(valid_mask, targets, torch.full_like(targets, -100))
            ce_sum = _CrossEntropy.apply(logits.float().reshape(-1, logits.shape[-1]), safe_targets, "sum")
            (ce_sum / (n_valid * accum)).backward()
            loss_sum += float((ce_sum / n_valid).detach().item()) / accum

        # Safeguard D1 (#168): Explicit all_reduce ONLY on active grads (p.grad is not None!)
        if world > 1:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        for opt in opts.values():
            opt.step()

        # Verify bit-exact cross-rank parity at Step 3 (after both passes=1 and passes=2 in smoke!)
        if step == min(2, max_steps - 1):
            max_diff = verify_rank_parity(model, rank, world, device)
            if is_master:
                print(f"[SAFEGUARD D1 PASSED] Step {step+1} cross-rank parity: max_rank_diff={max_diff}", flush=True)

        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step_times.append(dt)
        step_1 = step + 1

        if step_1 % 50 == 0 or step_1 == max_steps or args.smoke:
            lt = torch.tensor(loss_sum, device=device)
            if world > 1:
                dist.all_reduce(lt, op=dist.ReduceOp.AVG)
            avg_dt = sum(step_times[-25:]) / len(step_times[-25:])
            tps = step_tokens / avg_dt
            if is_master:
                peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                print(
                    f"[{args.arm}] step={step_1}/{max_steps} tok={step_1*step_tokens/1e9:.3f}B "
                    f"loss={lt.item():.4f} mult={mult:.3f} gates={gate_summary(model)} "
                    f"tok/s={tps/1e3:.1f}k vram={peak_gb:.1f}GB el={(time.monotonic()-t_start)/60:.1f}m",
                    flush=True,
                )

        if (not args.smoke) and (step_1 % args.val_every == 0) and (step_1 < max_steps):
            eval_primary_bpb(step_1)

    # Final multi-pass / multi-depth official 20.97M Val + 22-task CORE
    final_eval = {}
    if kind == "feedback":
        for p in (1, 2, 4, 8):
            r = eval_bpb_cfg(passes=p)
            final_eval[f"P{p}"] = dict(val_bpb=r["bpb"], val_loss_est=r["bpb"] * 3.285715)
        core_res = eval_core_cfg(passes=4)
    elif kind == "ode2d":
        for l_eval in (1, 2, 3, 4):
            r = eval_bpb_cfg(loops=l_eval)
            final_eval[f"R{l_eval}"] = dict(val_bpb=r["bpb"], val_loss_est=r["bpb"] * 3.285715)
        core_res = eval_core_cfg(loops=2)
    else:
        r = eval_bpb_cfg()
        final_eval["P1"] = dict(val_bpb=r["bpb"], val_loss_est=r["bpb"] * 3.285715)
        core_res = eval_core_cfg()

    log_event(dict(event="core", step=max_steps, tokens=max_steps * step_tokens,
                   core_metric=core_res["core_metric"], results=core_res["results"],
                   seconds=round(core_res["seconds"], 2)))

    wall_min = (time.monotonic() - t_start) / 60.0
    if is_master:
        ckpt_path = args.ckpt_dir / f"{args.arm}_final.pt"
        torch.save({"model": model.state_dict(), "arm": args.arm, "step": max_steps, "gates": gate_summary(model)}, ckpt_path)
        summary = {
            "event": "final_summary",
            "arm": args.arm,
            "kind": kind,
            "params": n_params,
            "init_sha": init_sha,
            "ref_steps": args.ref_steps,
            "max_steps": max_steps,
            "unique_steps": args.unique_steps,
            "total_tokens": max_steps * step_tokens,
            "unique_tokens": (args.unique_steps * step_tokens) if args.unique_steps > 0 else (max_steps * step_tokens),
            "e_flops_per_tok": e_flops_arm,
            "e_flops_ref": e_flops_ref,
            "realized_flops_ratio_vs_ctotal": round(realized_flops / c_total, 5) if not args.smoke else 1.0,
            "p2_fraction": round(p2_count / max_steps, 4) if kind == "feedback" else 0.0,
            "wall_min": round(wall_min, 2),
            "tok_per_sec": round((max_steps * step_tokens) / max(sum(step_times), 1e-6), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2),
            "gates": gate_summary(model),
            "final_eval": final_eval,
            "core_metric": core_res["core_metric"],
            "core_results": core_res["results"],
        }
        (args.output / "final_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[DONE {args.arm}] core={core_res['core_metric']:.5f} final_eval={final_eval} wall={wall_min:.1f}m", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
