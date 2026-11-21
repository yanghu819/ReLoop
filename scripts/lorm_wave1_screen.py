#!/usr/bin/env python3
"""LoRM Wave-1 Rapid-Iteration Arena (G0..G7) on Verbatim ifm-ai/LoRM Core.

Follows Dual Carbon High (#153/#154) Gates L1-L4 & Catches #1-#5 strictly:
  - Imports unmodified `lorm` core (`Transformer`, `FeedbackTransformer`, `NorMuon`,
    `SampledPasses`, `training_flops_per_token`, `_CrossEntropy`, `initialize`).
  - Gate L1: Programmatic exact Iso-FLOP (`C_total = ref_steps * batch_tok * E_flops(G2)`)
    for G0, G2..G7; G1 is exact Iso-Token (`ref_steps`) empirical slope ruler.
  - Gate L2: Evaluates feedback arms at P in {1, 2, 4, 8} (scoring 1.0x-AR at P=8 and
    logging P8 - P2 Jacobi contraction drift); ODE arms at R in {1, 2, 3, 4, 6}.
  - Gate L3: Supports both Wave-1A (F=1 fresh tokens, --unique-rows 0) and Wave-1B
    (F=4 fixed unique pool, --unique-rows N, reshuffle=False).
  - Gate L4: Strict single-factor LOO across C1 (G3), C1+C2 (G4), C1+C3 (G5),
    Track-A Vertical 2-Scalar ODE (G6), and Track-B 2D Spatio-Temporal ODE (G7).
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
for cand in (REPO_ROOT / "vendor", Path("/root/pn/vendor")):
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
# Registered LoRM Subclasses (Zero edits to vendor/lorm core!)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class C1GateConfig(FeedbackConfig):
    name: ClassVar[str] = "lorm_c1_snr_gate"
    alpha_init: float = 0.25


class LoRMC1SNRGate(FeedbackTransformer):
    """C1 isolated: cures the 27.7x Step-0 SNR cliff via d^(-0.5) * alpha scaling."""
    config_class = C1GateConfig

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
class C2DistDriveConfig(FeedbackConfig):
    name: ClassVar[str] = "lorm_c2_dist_drive"
    alpha_init: float = 0.25
    beta_init: float = 0.25


class LoRMC2DistDrive(FeedbackTransformer):
    """C1 + C2: compute f_{t-1} ONCE, inject at layer 0 (d^-0.5 * alpha) and layers 1..L-1 (beta_l / L).
    Zero extra matmuls; exact 1.00x autoregressive decoding FLOPs!"""
    config_class = C2DistDriveConfig

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.fb_alpha = nn.Parameter(torch.tensor(float(config.instance.alpha_init)))
        self.fb_beta = nn.Parameter(torch.full((config.n_layers - 1,), float(config.instance.beta_init)))

    def preserved_parameters(self) -> list[nn.Parameter]:
        return [self.fb_alpha, self.fb_beta]

    def _run_blocks_driven(self, x: Tensor, cos: Tensor, sin: Tensor, shifted_fb: Tensor | None) -> Tensor:
        inv_l = 1.0 / float(self.config.n_layers)
        n_blocks = len(self.backbone.blocks)
        for i, block in enumerate(self.backbone.blocks):
            x = block(x, cos, sin)
            if shifted_fb is not None and i < n_blocks - 1:
                x = x + (self.fb_beta[i] * inv_l).to(dtype=x.dtype) * shifted_fb
        return x

    def forward(self, input_ids: Tensor, *, passes: int | None = None) -> Tensor:
        passes = self.passes_of(passes)
        x, cos, sin = self.embed(input_ids)
        dim = self.config.dim
        top = None
        for index in range(passes):
            with torch.set_grad_enabled(torch.is_grad_enabled() and index == passes - 1):
                if index == 0:
                    state, shifted = x, None
                else:
                    f_raw = self.fusion(F.rms_norm(top.detach(), (dim,), eps=self.config.norm_eps))
                    shifted = F.pad(f_raw[:, :-1], (0, 0, 1, 0))
                    scale0 = (self.fb_alpha * (dim ** -0.5)).to(dtype=x.dtype)
                    state = x + scale0 * shifted
                top = self._run_blocks_driven(state, cos, sin, shifted)
        return self.readout(top)


@dataclass(frozen=True)
class C3TopKBpttConfig(FeedbackConfig):
    name: ClassVar[str] = "lorm_c3_top3_bptt"
    alpha_init: float = 0.25
    bptt_top_layers: int = 3


class LoRMC3TopKBPTT(LoRMC1SNRGate):
    """C1 + C3: unblinds dL_{t+1}/dh_t by running top K blocks of Pass (passes-2) WITH gradients
    and passing un-detached top into self.mix() (Catch #4)."""
    config_class = C3TopKBpttConfig

    def forward(self, input_ids: Tensor, *, passes: int | None = None) -> Tensor:
        passes = self.passes_of(passes)
        x, cos, sin = self.embed(input_ids)
        k_top = int(self.config.instance.bptt_top_layers)
        split = max(0, self.config.n_layers - k_top)
        grad_on = torch.is_grad_enabled()
        state = x
        top = None
        for index in range(passes):
            is_last = (index == passes - 1)
            is_penult = (index == passes - 2)
            if is_last:
                with torch.set_grad_enabled(grad_on):
                    if index:
                        # Un-detached top when coming from penultimate pass with grad!
                        state = self.mix(x, top)
                    top = self.blocks(state, cos, sin)
            elif is_penult and grad_on:
                with torch.no_grad():
                    if index:
                        state = self.mix(x, top.detach())
                    h = self.blocks(state, cos, sin, 0, split)
                h = h.detach()
                with torch.set_grad_enabled(True):
                    top = self.blocks(h, cos, sin, split, None)
            else:
                with torch.no_grad():
                    if index:
                        state = self.mix(x, top.detach())
                    top = self.blocks(state, cos, sin)
        return self.readout(top)

    def no_grad_macs(self, shape: tuple[int, int], *, passes: int | None = None, **options) -> int:
        passes = self.passes_of(passes)
        if passes == 1:
            return 0
        batch, length = shape
        dim = self.config.dim
        k_top = int(self.config.instance.bptt_top_layers)
        split = max(0, self.config.n_layers - k_top)
        early_blocks = sum(b.forward_macs((batch, length, dim)) for b in self.backbone.blocks[:split])
        all_blocks = self.backbone.forward_macs((batch, length, dim))
        fusion = batch * length * self.fusion.weight.numel()
        # Passes 0..passes-3: full no_grad (blocks + fusion); Pass passes-2: only blocks[:split] no_grad;
        # blocks[split:] and final fusion run WITH full grad (0 no_grad_macs).
        return (passes - 2) * (all_blocks + fusion) + early_blocks


@dataclass(frozen=True)
class DrivenODEConfig(InstanceConfig):
    name: ClassVar[str] = "track_a_driven_ode"
    pre_layers: int = 2
    core_layers: int = 4
    post_layers: int = 2
    loops: int = 3
    s_ode_init: float = 2.0
    g_pre_init: float = 1.0


class TrackADrivenODE(Transformer):
    """Vertical 2-Scalar Driven Euclidean ODE inside verbatim LoRM (Catch #5)."""
    config_class = DrivenODEConfig

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
class SpatioTemporalODEConfig(DrivenODEConfig):
    name: ClassVar[str] = "track_b_2d_spatiotemporal"
    g_fb_init: float = 0.25


class TrackB2DSpatioTemporalODE(TrackADrivenODE):
    """2D Spatio-Temporal Driven Euclidean ODE: vertical R-loop ODE + horizontal t-1->t feedback
    fused in ONE unified differentiable pass (100% bit-exact Train == AR-Decode DAG!)."""
    config_class = SpatioTemporalODEConfig

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
        fb_step = (self.g_fb * inv_r).to(dtype=x.dtype)
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


MODELS["lorm_c1_snr_gate"] = LoRMC1SNRGate
MODELS["lorm_c2_dist_drive"] = LoRMC2DistDrive
MODELS["lorm_c3_top3_bptt"] = LoRMC3TopKBPTT
MODELS["track_a_driven_ode"] = TrackADrivenODE
MODELS["track_b_2d_spatiotemporal"] = TrackB2DSpatioTemporalODE


# ---------------------------------------------------------------------------
# Builder, Exact Iso-FLOP, WSD Scheduler, Evaluation
# ---------------------------------------------------------------------------

ARMS = (
    "vanilla_isoflop",
    "vanilla_isotok",
    "lorm_verbatim",
    "lorm_c1_snr_gate",
    "lorm_c2_dist_drive",
    "lorm_c3_top3_bptt",
    "track_a_driven_ode",
    "track_b_2d_spatiotemporal",
)


def make_model_and_cfg(arm: str, seq_len: int, vocab_size: int = 32768) -> tuple[Transformer, ModelConfig, bool]:
    base_kw = dict(
        vocab_size=vocab_size, dim=768, n_heads=6, n_kv_heads=6, hidden_dim=2048,
        max_seq_len=seq_len, tie_embeddings=False, rope_theta=10000.0,
        norm_eps=1e-5, attention_backend="flash", qk_norm=True,
    )
    is_feedback = arm in ("lorm_verbatim", "lorm_c1_snr_gate", "lorm_c2_dist_drive", "lorm_c3_top3_bptt")
    if arm in ("vanilla_isoflop", "vanilla_isotok"):
        cfg = ModelConfig(n_layers=12, instance=None, **base_kw)
        model = Transformer(cfg)
    elif arm == "lorm_verbatim":
        cfg = ModelConfig(n_layers=12, instance=FeedbackConfig(passes=2), **base_kw)
        model = FeedbackTransformer(cfg)
    elif arm == "lorm_c1_snr_gate":
        cfg = ModelConfig(n_layers=12, instance=C1GateConfig(passes=2), **base_kw)
        model = LoRMC1SNRGate(cfg)
    elif arm == "lorm_c2_dist_drive":
        cfg = ModelConfig(n_layers=12, instance=C2DistDriveConfig(passes=2), **base_kw)
        model = LoRMC2DistDrive(cfg)
    elif arm == "lorm_c3_top3_bptt":
        cfg = ModelConfig(n_layers=12, instance=C3TopKBpttConfig(passes=2, bptt_top_layers=3), **base_kw)
        model = LoRMC3TopKBPTT(cfg)
    elif arm == "track_a_driven_ode":
        cfg = ModelConfig(n_layers=8, instance=DrivenODEConfig(pre_layers=2, core_layers=4, post_layers=2, loops=3), **base_kw)
        model = TrackADrivenODE(cfg)
    elif arm == "track_b_2d_spatiotemporal":
        cfg = ModelConfig(n_layers=8, instance=SpatioTemporalODEConfig(pre_layers=2, core_layers=4, post_layers=2, loops=3), **base_kw)
        model = TrackB2DSpatioTemporalODE(cfg)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return model, cfg, is_feedback


def expected_flops_per_token(model: Transformer, seq_len: int, is_feedback: bool,
                             pass_mix: tuple[float, float] = (0.15, 0.85)) -> float:
    if is_feedback:
        return sum(p * training_flops_per_token(model, seq_len, passes=k) for k, p in enumerate(pass_mix, 1))
    return float(training_flops_per_token(model, seq_len))


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
    # Split preserved_parameters (scalar/vector ODE/feedback gates) to clean gate_lr with wd=0.0 (#153/#154)
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
    for name in ("fb_alpha", "s_ode", "g_pre", "g_fb"):
        if hasattr(model, name):
            out[name] = round(float(getattr(model, name).detach().item()), 4)
    if hasattr(model, "fb_beta"):
        b = model.fb_beta.detach().float()
        out["fb_beta_mean"] = round(float(b.mean().item()), 4)
        out["fb_beta_min"] = round(float(b.min().item()), 4)
        out["fb_beta_max"] = round(float(b.max().item()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref-steps", type=int, default=4000, help="Reference steps for G2 (lorm_verbatim)")
    ap.add_argument("--micro-batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--unique-rows", type=int, default=0, help="0=all fresh rows (F=1); >0=fixed prefix repeated (F>1)")
    ap.add_argument("--lr", type=float, default=0.016333374512377017)
    ap.add_argument("--wd", type=float, default=0.0188)
    ap.add_argument("--gate-lr", type=float, default=0.003)
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

    # Exact Gate L1 programmatic Iso-FLOP computation against G2 (lorm_verbatim)
    ref_model, _, _ = make_model_and_cfg("lorm_verbatim", args.seq_len)
    e_flops_ref = expected_flops_per_token(ref_model, args.seq_len, True)
    del ref_model

    model, cfg, is_feedback = make_model_and_cfg(args.arm, args.seq_len)
    initialize(model, std=0.02)
    model = model.to(device=device)
    e_flops_arm = expected_flops_per_token(model, args.seq_len, is_feedback)

    batch_rows = args.micro_batch * args.accum
    batch_tokens = batch_rows * args.seq_len
    c_total = args.ref_steps * batch_tokens * e_flops_ref
    if args.arm == "vanilla_isotok":
        max_steps = args.ref_steps
    else:
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
    algo = SampledPasses(PassMix([0.15, 0.85]), seed=args.seed) if is_feedback else None

    print(
        f"[{args.tag}|{args.arm}] params={n_params/1e6:.2f}M init_sha={init_sha} "
        f"e_flops/tok={e_flops_arm/1e9:.4f}G (ratio={e_flops_arm/e_flops_ref:.4f}x) "
        f"max_steps={max_steps} tok={max_steps*batch_tokens/1e6:.1f}M pool={n_pool*args.seq_len/1e6:.1f}M "
        f"warmup={warmup_steps} decay={decay_steps}",
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
        elif args.arm in ("track_a_driven_ode", "track_b_2d_spatiotemporal"):
            r_list = (1, 2, 3, 4, 6) if full_suite else (3,)
            for r_eval in r_list:
                vl, vb = eval_model(model, val_x, val_y, byte_table, device, args.micro_batch, rows, loops=r_eval)
                rec[f"R{r_eval}_loss"] = round(vl, 6)
                rec[f"R{r_eval}_bpb"] = round(vb, 6)
            rec["primary_loss"] = rec["R3_loss"]
            rec["primary_bpb"] = rec["R3_bpb"]
        else:
            vl, vb = eval_model(model, val_x, val_y, byte_table, device, args.micro_batch, rows)
            rec["primary_loss"] = round(vl, 6)
            rec["primary_bpb"] = round(vb, 6)
        rec["gates"] = gate_summary(model)
        return rec

    history = [run_eval_suite(0, full_suite=False)]
    print(f"[{args.tag}] step=0 eval={history[-1]}", flush=True)

    eval_points = sorted({max(1, max_steps // 2), max_steps})
    row_cursor = 0
    t0 = time.perf_counter()
    train_loss_ema = 0.0

    for step in range(1, max_steps + 1):
        mult = wsd_mult(step - 1, max_steps, warmup_steps, decay_steps)
        for opt in opts.values():
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * mult
            opt.zero_grad(set_to_none=True)

        if is_feedback:
            algo.begin_update(step - 1)
            cur_passes = algo.passes
        else:
            cur_passes = None

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
        "max_steps": max_steps,
        "total_tokens": max_steps * batch_tokens,
        "unique_tokens": n_pool * args.seq_len,
        "epochs": round((max_steps * batch_tokens) / float(n_pool * args.seq_len), 3),
        "total_flops": max_steps * batch_tokens * e_flops_arm,
        "wall_min": round(wall_min, 2),
        "tok_per_sec": round((max_steps * batch_tokens) / max(wall_min * 60.0, 1e-6), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "final_eval": final_rec,
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"[{args.tag}] DONE -> {args.out} primary_loss={final_rec['primary_loss']} primary_bpb={final_rec['primary_bpb']}", flush=True)


if __name__ == "__main__":
    main()
