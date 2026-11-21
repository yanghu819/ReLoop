#!/usr/bin/env python3
"""LoRM Wave-2 Arena (W2-G0..G7): Breaking Vanilla Iso-FLOP Even at F=1 & Crushing F=4!

Implements pre-run audit (#157 + #158) unified 8-GPU Wave-2 matrix on verbatim vendor/lorm:
  - W2-G0 (c1_unthrottled)        : C1 with alpha_init=2.0, gate_lr=base_lr (4000 steps)
  - W2-G1 (c1_tap_l8)             : Tap-L8 Theorem (h^{(8)}->u_t, -33% Pass-0 MACs, unblinded future attn grads, ~4256 steps)
  - W2-G2 (c1_passmix_50_50)      : C1 + static PassMix([0.50, 0.50]) (~4330 steps)
  - W2-G3 (c1_curr_warm10_late40) : C1 + Warm-10 (first 60%) -> Late-40 [0.20, 0.80] curriculum (~4447 steps, zero W_fb shock)
  - W2-G4 (c1_mid_top_2scalar)    : C1 + 2-Scalar Mid(L6)+Top(L12) Tap (0 extra matmuls, 1.00x AR, 4000 steps)
  - W2-G5 (c1_boundary_gate)      : C1 + uP .mean(-1) Token-Boundary Reset Gate (inert at P=1, 4000 steps)
  - W2-G6 (isoparam_driven_ode_12L): True 135.3M Iso-Param Vertical 2-Scalar ODE (4/4/4 R=2, s_ode=1.0, g_pre=0.5)
  - W2-G7 (isoparam_2d_ode_12L)   : True 135.9M Iso-Param 2D Spatio-Temporal ODE (4/4/4 R=2, bug-fixed d^-0.5 on g_fb=4.0)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import ClassVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
for cand in (REPO_ROOT / "vendor", Path(__file__).resolve().parent.parent / "vendor"):
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))

from lorm.model.config import InstanceConfig, ModelConfig
from lorm.model.feedback import FeedbackConfig, FeedbackTransformer
from lorm.model.transformer import Transformer, initialize
from lorm.registry import MODELS
from lorm.train.algo import PassMix, SampledPasses, training_flops_per_token
from lorm.train.loss import _CrossEntropy
from lorm.train.optim import build_optimizers


# ---------------------------------------------------------------------------
# Wave-2 Registered LoRM Subclasses (Zero edits to vendor/lorm core!)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C1Config(FeedbackConfig):
    name: ClassVar[str] = "c1_unthrottled"
    alpha_init: float = 2.0


class LoRMC1Unthrottled(FeedbackTransformer):
    config_class = C1Config

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.fb_alpha = nn.Parameter(torch.tensor(float(config.instance.alpha_init)))

    def preserved_parameters(self) -> list[nn.Parameter]:
        return [self.fb_alpha]

    def mix(self, embedding: Tensor, top: Tensor) -> Tensor:
        dim = self.config.dim
        f_raw = self.fusion(F.rms_norm(top, (dim,), eps=self.config.norm_eps))
        scale = (self.fb_alpha * (dim ** -0.5)).to(dtype=embedding.dtype)
        fed = f_raw * scale
        return torch.cat([embedding[:, :1], embedding[:, 1:] + fed[:, :-1]], dim=1)


@dataclass(frozen=True)
class TapL8Config(C1Config):
    name: ClassVar[str] = "c1_tap_l8"
    tap_layer: int = 8


class LoRMTapL8(LoRMC1Unthrottled):
    """Tap-L8 Theorem (#158): feedback reads h_{t-1}^{(tap_layer)} in BOTH training and 1.00x AR decoding.
    Skipping blocks[tap_layer:] on passes < passes-1 is BIT-EXACT (0.0 Gate-L2 mismatch) and cuts Pass-0 MACs 33%
    while unblinding future attention gradients dL_{t+1..T}/dh_t^{(8)} through blocks[8..11] for FREE!"""
    config_class = TapL8Config

    def forward(self, input_ids: Tensor, *, passes: int | None = None) -> Tensor:
        passes = self.passes_of(passes)
        tap = int(self.config.instance.tap_layer)
        x, cos, sin = self.embed(input_ids)
        state = x
        tapped = None
        for index in range(passes):
            is_last = (index == passes - 1)
            with torch.set_grad_enabled(torch.is_grad_enabled() and is_last):
                if index:
                    state = self.mix(x, tapped.detach())
                if is_last:
                    top = self.blocks(state, cos, sin)
                else:
                    tapped = self.blocks(state, cos, sin, 0, tap)
        return self.readout(top)

    def _tap_blocks_macs(self, shape: tuple[int, int]) -> int:
        batch, length = shape
        dim = self.config.dim
        tap = int(self.config.instance.tap_layer)
        return sum(b.forward_macs((batch, length, dim)) for b in self.backbone.blocks[:tap])

    def feedback_macs(self, shape: tuple[int, int], passes: int) -> int:
        batch, length = shape
        return (passes - 1) * (self._tap_blocks_macs(shape) + batch * length * self.fusion.weight.numel())

    def no_grad_macs(self, shape: tuple[int, int], *, passes: int | None = None, **options) -> int:
        passes = self.passes_of(passes)
        if passes == 1:
            return 0
        batch, length = shape
        fusion = batch * length * self.fusion.weight.numel()
        return (passes - 1) * self._tap_blocks_macs(shape) + (passes - 2) * fusion + fusion // 2


@dataclass(frozen=True)
class MidTopConfig(C1Config):
    name: ClassVar[str] = "c1_mid_top_2scalar"
    mid_layer: int = 6
    beta_mid_init: float = 0.5


class LoRMMidTop2Scalar(LoRMC1Unthrottled):
    """2-Scalar Mid(L6)+Top(L12) Tap: zero extra matmuls, exact 1.00x AR decode."""
    config_class = MidTopConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.beta_mid = nn.Parameter(torch.tensor(float(config.instance.beta_mid_init)))

    def preserved_parameters(self) -> list[nn.Parameter]:
        return [self.fb_alpha, self.beta_mid]

    def forward(self, input_ids: Tensor, *, passes: int | None = None) -> Tensor:
        passes = self.passes_of(passes)
        mid = int(self.config.instance.mid_layer)
        dim = self.config.dim
        eps = self.config.norm_eps
        x, cos, sin = self.embed(input_ids)
        state = x
        h_mix = None
        for index in range(passes):
            is_last = (index == passes - 1)
            with torch.set_grad_enabled(torch.is_grad_enabled() and is_last):
                if index:
                    state = self.mix(x, h_mix.detach())
                h_mid = self.blocks(state, cos, sin, 0, mid)
                top = self.blocks(h_mid, cos, sin, mid, None)
                if not is_last:
                    h_mix = F.rms_norm(top, (dim,), eps=eps) + self.beta_mid.to(dtype=top.dtype) * F.rms_norm(h_mid, (dim,), eps=eps)
        return self.readout(top)


@dataclass(frozen=True)
class BoundaryGateConfig(C1Config):
    name: ClassVar[str] = "c1_boundary_gate"


class LoRMBoundaryGate(LoRMC1Unthrottled):
    """uP .mean(-1) Token-Boundary Reset Gate (#158 Catch W2-2): 100% inert at P=1, 1.00x AR decode."""
    config_class = BoundaryGateConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.w_e = nn.Parameter(torch.zeros(config.dim))
        self.w_h = nn.Parameter(torch.zeros(config.dim))
        self.b_g = nn.Parameter(torch.tensor(0.0))

    def preserved_parameters(self) -> list[nn.Parameter]:
        return [self.fb_alpha, self.w_e, self.w_h, self.b_g]

    def mix(self, embedding: Tensor, top: Tensor) -> Tensor:
        dim = self.config.dim
        eps = self.config.norm_eps
        e_norm = F.rms_norm(embedding[:, 1:], (dim,), eps=eps)
        h_norm = F.rms_norm(top[:, :-1], (dim,), eps=eps)
        we = self.w_e.to(dtype=e_norm.dtype)
        wh = self.w_h.to(dtype=h_norm.dtype)
        logit = self.b_g.to(dtype=e_norm.dtype) + (we * e_norm).mean(dim=-1, keepdim=True) + (wh * h_norm).mean(dim=-1, keepdim=True)
        gate = 2.0 * torch.sigmoid(logit)
        f_raw = self.fusion(h_norm)
        scale = (self.fb_alpha * (dim ** -0.5)).to(dtype=embedding.dtype)
        fed = f_raw * (scale * gate.to(dtype=embedding.dtype))
        return torch.cat([embedding[:, :1], embedding[:, 1:] + fed], dim=1)


@dataclass(frozen=True)
class IsoParamODEConfig(InstanceConfig):
    name: ClassVar[str] = "isoparam_driven_ode_12L"
    pre_layers: int = 4
    core_layers: int = 4
    post_layers: int = 4
    loops: int = 2
    s_ode_init: float = 1.0
    g_pre_init: float = 0.5


class IsoParamDrivenODE12L(Transformer):
    """True 135.3M Iso-Param Vertical 2-Scalar Driven ODE (4 pre + 4 core x R=2 + 4 post = 16 virtual layers)."""
    config_class = IsoParamODEConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        inst = config.instance
        assert inst.pre_layers + inst.core_layers + inst.post_layers == config.n_layers
        self.s_ode = nn.Parameter(torch.tensor(float(inst.s_ode_init)))
        self.g_pre = nn.Parameter(torch.tensor(float(inst.g_pre_init)))

    def preserved_parameters(self) -> list[nn.Parameter]:
        return [self.s_ode, self.g_pre]

    def forward(self, input_ids: Tensor, *, loops: int | None = None) -> Tensor:
        inst = self.config.instance
        r_eff = inst.loops if loops is None else int(loops)
        x, cos, sin = self.embed(input_ids)
        x = self.blocks(x, cos, sin, 0, inst.pre_layers)
        x_pre = x
        inv_r = 1.0 / float(r_eff)
        s_step = (self.s_ode * inv_r).to(dtype=x.dtype)
        g_step = (self.g_pre * inv_r).to(dtype=x.dtype)
        c_start, c_end = inst.pre_layers, inst.pre_layers + inst.core_layers
        for _ in range(r_eff):
            h = self.blocks(x, cos, sin, c_start, c_end)
            x = x + s_step * (h - x) + g_step * x_pre
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
        return pre_m + r_eff * core_m + post_m + batch * length * self.head.weight.numel()


@dataclass(frozen=True)
class IsoParam2DODEConfig(IsoParamODEConfig):
    name: ClassVar[str] = "isoparam_2d_ode_12L"
    g_fb_init: float = 4.0


class IsoParam2DODE12L(IsoParamDrivenODE12L):
    """True 135.9M Iso-Param 2D Spatio-Temporal Driven ODE (bug-fixed dim^-0.5 scaling on g_fb!)."""
    config_class = IsoParam2DODEConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.fusion = nn.Linear(config.dim, config.dim, bias=False)
        self.g_fb = nn.Parameter(torch.tensor(float(config.instance.g_fb_init)))

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
            fb_raw = self.fusion(F.rms_norm(x, (dim,), eps=self.config.norm_eps))
            fb_shift = F.pad(fb_raw[:, :-1], (0, 0, 1, 0))
            x = x + s_step * (h - x) + g_step * x_pre + fb_step * fb_shift
        x = self.blocks(x, cos, sin, c_end, None)
        return self.readout(x)

    def forward_macs(self, shape: tuple[int, int], *, loops: int | None = None, **options) -> int:
        inst = self.config.instance
        r_eff = inst.loops if loops is None else int(loops)
        batch, length = shape
        return super().forward_macs(shape, loops=r_eff) + r_eff * batch * length * self.fusion.weight.numel()


MODELS["c1_unthrottled"] = LoRMC1Unthrottled
MODELS["c1_tap_l8"] = LoRMTapL8
MODELS["c1_mid_top_2scalar"] = LoRMMidTop2Scalar
MODELS["c1_boundary_gate"] = LoRMBoundaryGate
MODELS["isoparam_driven_ode_12L"] = IsoParamDrivenODE12L
MODELS["isoparam_2d_ode_12L"] = IsoParam2DODE12L


ARMS = (
    "c1_unthrottled",
    "c1_tap_l8",
    "c1_passmix_50_50",
    "c1_curr_warm10_late40",
    "c1_mid_top_2scalar",
    "c1_boundary_gate",
    "isoparam_driven_ode_12L",
    "isoparam_2d_ode_12L",
)


def make_model(arm: str, seq_len: int, vocab_size: int = 32768) -> tuple[Transformer, bool, tuple[float, float] | str]:
    base_kw = dict(
        vocab_size=vocab_size, dim=768, n_heads=6, n_kv_heads=6, hidden_dim=2048,
        max_seq_len=seq_len, tie_embeddings=False, rope_theta=10000.0,
        norm_eps=1e-5, attention_backend="flash", qk_norm=True,
    )
    if arm == "c1_unthrottled":
        return LoRMC1Unthrottled(ModelConfig(n_layers=12, instance=C1Config(passes=2), **base_kw)), True, (0.15, 0.85)
    if arm == "c1_tap_l8":
        return LoRMTapL8(ModelConfig(n_layers=12, instance=TapL8Config(passes=2, tap_layer=8), **base_kw)), True, (0.15, 0.85)
    if arm == "c1_passmix_50_50":
        return LoRMC1Unthrottled(ModelConfig(n_layers=12, instance=C1Config(passes=2), **base_kw)), True, (0.50, 0.50)
    if arm == "c1_curr_warm10_late40":
        return LoRMC1Unthrottled(ModelConfig(n_layers=12, instance=C1Config(passes=2), **base_kw)), True, "curr_warm10_late40"
    if arm == "c1_mid_top_2scalar":
        return LoRMMidTop2Scalar(ModelConfig(n_layers=12, instance=MidTopConfig(passes=2), **base_kw)), True, (0.15, 0.85)
    if arm == "c1_boundary_gate":
        return LoRMBoundaryGate(ModelConfig(n_layers=12, instance=BoundaryGateConfig(passes=2), **base_kw)), True, (0.15, 0.85)
    if arm == "isoparam_driven_ode_12L":
        return IsoParamDrivenODE12L(ModelConfig(n_layers=12, instance=IsoParamODEConfig(), **base_kw)), False, (1.0, 0.0)
    if arm == "isoparam_2d_ode_12L":
        return IsoParam2DODE12L(ModelConfig(n_layers=12, instance=IsoParam2DODEConfig(), **base_kw)), False, (1.0, 0.0)
    raise ValueError(f"Unknown arm: {arm}")


def expected_flops(model: Transformer, seq_len: int, is_feedback: bool, sched) -> float:
    if not is_feedback:
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


def param_fingerprint(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().float().cpu().numpy().tobytes()[:1024])
    return h.hexdigest()[:16]


@torch.no_grad()
def eval_model(model: Transformer, val_x: Tensor, val_y: Tensor, byte_table: Tensor,
               device: torch.device, batch_size: int, max_rows: int, **fwd_kwargs) -> tuple[float, float]:
    model.eval()
    total_nats = total_bytes = total_toks = 0.0
    n_rows = min(max_rows, val_x.shape[0])
    try:
        for i in range(0, n_rows, batch_size):
            xb = val_x[i:i + batch_size].to(device, dtype=torch.long)
            yb = val_y[i:i + batch_size].to(device, dtype=torch.long)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb, **fwd_kwargs)
            loss = F.cross_entropy(logits.float().flatten(0, 1), yb.flatten(), ignore_index=-1, reduction="none")
            nbytes = byte_table[yb.flatten().clamp_min(0)]
            valid = (yb.flatten() >= 0) & (nbytes > 0)
            total_nats += loss[valid].sum().item()
            total_bytes += nbytes[valid].sum().item()
            total_toks += valid.sum().item()
    finally:
        model.train()
    return total_nats / max(total_toks, 1.0), total_nats / (math.log(2.0) * max(total_bytes, 1.0))


def gate_summary(model: nn.Module) -> dict[str, float]:
    out = {}
    for name in ("fb_alpha", "beta_mid", "b_g", "s_ode", "g_pre", "g_fb"):
        if hasattr(model, name):
            out[name] = round(float(getattr(model, name).detach().item()), 4)
    for name in ("w_e", "w_h"):
        if hasattr(model, name):
            v = getattr(model, name).detach().float()
            out[f"{name}_rms"] = round(float(v.square().mean().sqrt().item()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref-steps", type=int, default=4000)
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--unique-rows", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.016333374512377017)
    ap.add_argument("--wd", type=float, default=0.0188)
    ap.add_argument("--gate-lr", type=float, default=0.016333374512377017)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-rows", type=int, default=4096)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=Path("/dev/shm/rsi_arena_data_1b"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Exact C_total reference = Wave-1 G2 (verbatim LoRM FeedbackTransformer, p=[0.15, 0.85])
    ref_cfg = ModelConfig(
        vocab_size=32768, dim=768, n_heads=6, n_kv_heads=6, hidden_dim=2048,
        max_seq_len=args.seq_len, n_layers=12, tie_embeddings=False, rope_theta=10000.0,
        norm_eps=1e-5, attention_backend="flash", qk_norm=True, instance=FeedbackConfig(passes=2),
    )
    ref_model = FeedbackTransformer(ref_cfg)
    e_flops_ref = 0.15 * training_flops_per_token(ref_model, args.seq_len, passes=1) + \
                  0.85 * training_flops_per_token(ref_model, args.seq_len, passes=2)
    del ref_model

    model, is_feedback, sched = make_model(args.arm, args.seq_len)
    initialize(model, std=0.02)
    model = model.to(device=device)
    e_flops_arm = expected_flops(model, args.seq_len, is_feedback, sched)
    f1_tok = float(training_flops_per_token(model, args.seq_len, passes=1)) if is_feedback else e_flops_arm
    f2_tok = float(training_flops_per_token(model, args.seq_len, passes=2)) if is_feedback else e_flops_arm

    batch_rows = args.micro_batch * args.accum
    batch_tokens = batch_rows * args.seq_len
    c_total = args.ref_steps * batch_tokens * e_flops_ref
    max_steps = max(1, round(c_total / (batch_tokens * e_flops_arm)))
    warmup_steps = max(1, round(0.05 * max_steps))
    decay_steps = max(1, round(0.20 * max_steps))
    mom_warmup = min(300, max(1, round(0.075 * max_steps)))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    init_sha = param_fingerprint(model)
    opts = build_lorm_optimizers(model, args.lr, args.wd, args.gate_lr, mom_warmup)

    if not args.no_compile:
        for i, blk in enumerate(model.backbone.blocks):
            model.backbone.blocks[i] = torch.compile(blk)

    train_d = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_d = torch.load(args.data_dir / "val.pt", map_location="cpu", weights_only=True)
    train_x, train_y = train_d["x"], train_d["y"]
    val_x, val_y = val_d["x"], val_d["y"]
    byte_table = torch.load("/dev/shm/nt-runtime/resources/tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    n_pool = args.unique_rows if args.unique_rows > 0 else train_x.size(0)
    assert n_pool <= train_x.size(0)

    algo_static = SampledPasses(PassMix(list(sched)), seed=args.seed) if (is_feedback and sched != "curr_warm10_late40") else None
    algo_early = SampledPasses(PassMix([0.90, 0.10]), seed=args.seed) if sched == "curr_warm10_late40" else None
    algo_late = SampledPasses(PassMix([0.20, 0.80]), seed=args.seed + 1000) if sched == "curr_warm10_late40" else None
    switch_step = round(0.60 * max_steps)

    print(
        f"[{args.tag}|{args.arm}] params={n_params/1e6:.2f}M init_sha={init_sha} "
        f"e_flops/tok={e_flops_arm/1e9:.4f}G (ratio={e_flops_arm/e_flops_ref:.4f}x) "
        f"max_steps={max_steps} tok={max_steps*batch_tokens/1e6:.1f}M pool={n_pool*args.seq_len/1e6:.1f}M",
        flush=True,
    )

    def run_eval_suite(step_num: int, full_suite: bool) -> dict:
        rows = args.eval_rows if full_suite else min(1024, args.eval_rows)
        rec = {"step": step_num, "tokens": step_num * batch_tokens}
        if is_feedback:
            p_list = (1, 2, 4, 8) if full_suite else (1, 2, 4)
            for p in p_list:
                vl, vb = eval_model(model, val_x, val_y, byte_table, device, args.micro_batch, rows, passes=p)
                rec[f"P{p}_loss"] = round(vl, 6)
                rec[f"P{p}_bpb"] = round(vb, 6)
            rec["primary_loss"] = rec.get("P8_loss", rec["P4_loss"])
            rec["primary_bpb"] = rec.get("P8_bpb", rec["P4_bpb"])
            if "P8_loss" in rec:
                rec["drift_P8_P2"] = round(rec["P8_loss"] - rec["P2_loss"], 6)
        else:
            r_list = (1, 2, 3, 4, 6) if full_suite else (2,)
            for r_eval in r_list:
                vl, vb = eval_model(model, val_x, val_y, byte_table, device, args.micro_batch, rows, loops=r_eval)
                rec[f"R{r_eval}_loss"] = round(vl, 6)
                rec[f"R{r_eval}_bpb"] = round(vb, 6)
            rec["primary_loss"] = rec["R2_loss"]
            rec["primary_bpb"] = rec["R2_bpb"]
        rec["gates"] = gate_summary(model)
        return rec

    history = [run_eval_suite(0, full_suite=False)]
    print(f"[{args.tag}] step=0 eval={history[-1]}", flush=True)

    eval_points = sorted({max(1, max_steps // 2), max_steps})
    row_cursor = 0
    realized_flops = 0.0
    p2_count = 0
    t0 = time.perf_counter()
    train_loss_ema = 0.0

    for step in range(1, max_steps + 1):
        mult = wsd_mult(step - 1, max_steps, warmup_steps, decay_steps)
        for opt in opts.values():
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * mult
            opt.zero_grad(set_to_none=True)

        if is_feedback:
            if sched == "curr_warm10_late40":
                alg = algo_early if (step - 1) < switch_step else algo_late
                alg.begin_update(step - 1)
                cur_passes = alg.passes
            else:
                algo_static.begin_update(step - 1)
                cur_passes = algo_static.passes
            realized_flops += batch_tokens * (f2_tok if cur_passes == 2 else f1_tok)
            if cur_passes == 2:
                p2_count += 1
        else:
            cur_passes = None
            realized_flops += batch_tokens * e_flops_arm

        step_loss = 0.0
        for _ in range(args.accum):
            if row_cursor + args.micro_batch <= n_pool:
                xb = train_x[row_cursor:row_cursor + args.micro_batch]
                yb = train_y[row_cursor:row_cursor + args.micro_batch]
                row_cursor += args.micro_batch
                if row_cursor >= n_pool:
                    row_cursor = 0
            else:
                rem = n_pool - row_cursor
                need = args.micro_batch - rem
                xb = torch.cat([train_x[row_cursor:n_pool], train_x[:need]], dim=0)
                yb = torch.cat([train_y[row_cursor:n_pool], train_y[:need]], dim=0)
                row_cursor = need

            xb = xb.to(device, dtype=torch.long, non_blocking=True)
            yb = yb.to(device, dtype=torch.long, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb, passes=cur_passes) if is_feedback else model(xb)
            targets = yb.reshape(-1)
            valid_mask = targets >= 0
            n_valid = valid_mask.sum().clamp_min(1)
            safe_targets = torch.where(valid_mask, targets, torch.full_like(targets, -100))
            loss_sum = _CrossEntropy.apply(logits.float().reshape(-1, logits.shape[-1]), safe_targets, "sum")
            loss = loss_sum / (n_valid * args.accum)
            loss.backward()
            step_loss += float((loss_sum / n_valid).detach().item()) / args.accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts.values():
            opt.step()

        train_loss_ema = step_loss if step == 1 else (0.95 * train_loss_ema + 0.05 * step_loss)
        if step % 250 == 0 or step == max_steps:
            el = time.perf_counter() - t0
            tps = (step * batch_tokens) / max(el, 1e-6)
            print(
                f"[{args.tag}] step={step}/{max_steps} loss={train_loss_ema:.4f} mult={mult:.3f} "
                f"tok/s={tps/1e3:.1f}k gates={gate_summary(model)} el={el/60:.1f}m",
                flush=True,
            )

        if step in eval_points:
            rec = run_eval_suite(step, full_suite=(step == max_steps))
            history.append(rec)
            print(f"[{args.tag}] EVAL step={step}: {json.dumps(rec)}", flush=True)

    wall_min = (time.perf_counter() - t0) / 60.0
    final_rec = history[-1]
    summary = {
        "tag": args.tag,
        "arm": args.arm,
        "params": n_params,
        "init_sha": init_sha,
        "e_flops_per_tok": e_flops_arm,
        "e_flops_ref": e_flops_ref,
        "flops_ratio_vs_g2": round(e_flops_arm / e_flops_ref, 5),
        "realized_flops_ratio_vs_ctotal": round(realized_flops / c_total, 5),
        "p2_fraction": round(p2_count / max_steps, 4) if is_feedback else 0.0,
        "max_steps": max_steps,
        "total_tokens": max_steps * batch_tokens,
        "unique_tokens": n_pool * args.seq_len,
        "epochs": round((max_steps * batch_tokens) / float(n_pool * args.seq_len), 3),
        "wall_min": round(wall_min, 2),
        "tok_per_sec": round((max_steps * batch_tokens) / max(wall_min * 60.0, 1e-6), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "final_eval": final_rec,
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[{args.tag}] DONE -> {args.out} primary_loss={final_rec['primary_loss']} primary_bpb={final_rec['primary_bpb']} rel_flops={summary['realized_flops_ratio_vs_ctotal']}", flush=True)


if __name__ == "__main__":
    main()
