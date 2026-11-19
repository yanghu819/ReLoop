"""8-GPU Parallel RSI Arena on a single 8xB300 box (one independent idea per GPU).

Every GPU runs world_size=1 on the EXACT SAME pre-tokenized ClimbMix train/val tensors,
identical seed (42), identical ReferenceLM (Huginn r=8, s=4) backbone, identical Muon+AdamW
optimizer and WSD schedule. Produces apples-to-apples multi-depth val_bpb (r=1,2,4,8,16),
cross-loop collapse rate, and wall-clock tok/s for 8 ideas every ~12-15 minutes.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/nt")
sys.path.insert(0, "/root/pn/loop-muon")

from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import ReferenceLM, build_model, norm
from nexttransformer.training.optimizer import ExperimentOptimizer
from loop_dedup import LoopDedup
import loop_muon_hook as lm_hook


# ---------------------------------------------------------------------------
# Idea R1-7: 4-line Checkpoint-Safe Streaming Row-Orthogonal Loop-Muon v2
# ---------------------------------------------------------------------------
class _RowOrthCtx:
    def __init__(self, gamma: float = 0.5, eps: float = 1e-8):
        self.gamma = gamma
        self.eps = eps
        self.active = False
        self.acc = {}      # param_id -> running accumulated grad [out, in] (fp32)
        self.left = {}     # param_id -> remaining loops in this microbatch
        self.n_loops = 0

    def begin(self, n_grad_loops: int):
        self.acc.clear()
        self.left.clear()
        self.n_loops = n_grad_loops
        self.active = False

    def set_active(self, flag: bool):
        self.active = flag


ROW_CTX = _RowOrthCtx()


class _RowOrthLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, pid):
        ctx.pid = pid
        ctx.save_for_backward(x, weight)
        return F.linear(x, weight.to(dtype=x.dtype))

    @staticmethod
    def backward(ctx, grad_y):
        x, weight = ctx.saved_tensors
        grad_x = (grad_y @ weight.to(dtype=grad_y.dtype)) if ctx.needs_input_grad[0] else None
        if not ctx.needs_input_grad[1]:
            return grad_x, None, None
        gy = grad_y.reshape(-1, grad_y.shape[-1])
        xr = x.reshape(-1, x.shape[-1])
        gw = (gy.t() @ xr.to(gy.dtype)).float()                    # native BF16 Tensor Core GEMM -> fp32
        pid = ctx.pid
        if pid not in ROW_CTX.acc:
            ROW_CTX.acc[pid] = gw.clone()
            ROW_CTX.left[pid] = ROW_CTX.n_loops - 1
            out = gw
        else:
            acc = ROW_CTX.acc[pid]
            # Per-row positive-cosine orthogonalization: guaranteed <g_tilde, g> >= (1-gamma)||g||^2 > 0
            dot = (gw * acc).sum(dim=-1, keepdim=True).clamp_min(0.0)
            den = acc.pow(2).sum(dim=-1, keepdim=True).clamp_min(ROW_CTX.eps)
            out = gw - (ROW_CTX.gamma * dot / den) * acc
            ROW_CTX.acc[pid] = acc + out
            ROW_CTX.left[pid] -= 1
        if ROW_CTX.left.get(pid, 0) <= 0:
            ROW_CTX.acc.pop(pid, None)
            ROW_CTX.left.pop(pid, None)
        return grad_x, out.to(weight.dtype), None


def _row_orth_forward(self, x):
    if ROW_CTX.active and torch.is_grad_enabled() and self.weight.requires_grad:
        return _RowOrthLinear.apply(x, self.weight, self._row_orth_pid)
    return F.linear(x, self.weight.to(dtype=x.dtype))


def wrap_row_orth(model, first_layer: int, last_layer: int, gamma: float = 0.5) -> int:
    ROW_CTX.gamma = gamma
    wrapped = 0
    for layer in range(first_layer, last_layer + 1):
        blk = model.transformer.h[layer]
        for m in (blk.attn.c_q, blk.attn.c_k, blk.attn.c_v, blk.mlp.c_fc):
            m._row_orth_pid = id(m.weight)
            m.forward = types.MethodType(_row_orth_forward, m)
            wrapped += 1
    return wrapped


# ---------------------------------------------------------------------------
# Custom Loop Forward Dispatcher (exact Huginn backbone + clean idea hooks)
# ---------------------------------------------------------------------------
def attach_idea(model: ReferenceLM, idea: str):
    """Patch model._loop_hidden (and optional 1D params / linear hooks) cleanly for `idea`."""
    c = model.config
    first_l, last_l = c.pre_layers, c.pre_layers + c.core_layers - 1

    # Reset global hooks
    lm_hook.CTX.gamma = 0.0
    lm_hook.CTX.reset()
    ROW_CTX.begin(0)

    dedup_macro = None
    dedup_mid = None
    use_adj_dedup = False
    use_loop_muon = False
    use_row_orth = False
    use_signed_krylov = False
    use_loop_attnres = False
    use_neg_attnres = False
    use_depth_scale = False
    use_ortho_rot = False
    gamma = 0.5

    if idea == "base":
        pass
    elif idea == "dedup":
        dedup_macro = LoopDedup(gamma=0.5, window=8, ridge=1e-2)
    elif idea == "adj_dedup":
        use_adj_dedup = True
        gamma = 0.5
    elif idea == "loop_muon":
        use_loop_muon = True
        lm_hook.CTX.gamma = 0.5
        lm_hook.CTX.eps = 1e-2
        n_w = lm_hook.wrap_core(model, first_l, last_l)
        assert n_w == c.core_layers * 4, n_w
    elif idea == "dedup_muon":
        dedup_macro = LoopDedup(gamma=0.5, window=8, ridge=1e-2)
        use_loop_muon = True
        lm_hook.CTX.gamma = 0.5
        lm_hook.CTX.eps = 1e-2
        n_w = lm_hook.wrap_core(model, first_l, last_l)
        assert n_w == c.core_layers * 4, n_w
    elif idea == "subblock_dedup":
        dedup_mid = LoopDedup(gamma=0.5, window=8, ridge=1e-2)
        dedup_macro = LoopDedup(gamma=0.5, window=8, ridge=1e-2)
    elif idea == "signed_krylov":
        # 28 unconstrained learnable scalars for r=0..7 mixing past updates u_0..u_{r-1}
        use_signed_krylov = True
        n_pairs = c.max_loops * (c.max_loops - 1) // 2
        model.krylov_alpha = nn.Parameter(torch.zeros(n_pairs, device=next(model.parameters()).device))
    elif idea == "row_orth_muon":
        use_row_orth = True
        n_w = wrap_row_orth(model, first_l, last_l, gamma=0.5)
        assert n_w == c.core_layers * 4, n_w
    elif idea == "loop_attnres":
        # Positive softmax depth attention over past loop states (Kaiming-clean contrast to negative dedup)
        use_loop_attnres = True
        dev = next(model.parameters()).device
        model.loop_queries = nn.Parameter(torch.zeros(c.max_loops, c.width, device=dev))
    elif idea == "dedup_krylov":
        dedup_macro = LoopDedup(gamma=0.5, window=8, ridge=1e-2)
        use_signed_krylov = True
        n_pairs = c.max_loops * (c.max_loops - 1) // 2
        model.krylov_alpha = nn.Parameter(torch.zeros(n_pairs, device=next(model.parameters()).device))
    elif idea == "adj_dedup_row_orth":
        use_adj_dedup = True
        gamma = 0.5
        use_row_orth = True
        n_w = wrap_row_orth(model, first_l, last_l, gamma=0.5)
        assert n_w == c.core_layers * 4, n_w
    else:
        raise ValueError(f"Unknown idea: {idea}")

    mid_j = c.core_layers // 2 - 1

    def _arena_loop_hidden(self, idx, targets=None):
        cfg = self.config
        x = self._embed_reference(idx)
        x0 = x
        for i in range(cfg.pre_layers):
            x = self._block(x, x0, idx, i, i, cfg.maximum_depth)
        prelude = x
        n = cfg.max_loops if self.training else cfg.eval_loops
        # Deterministic per-row noise matching Huginn
        noise = []
        for row in range(idx.shape[0]):
            g = torch.Generator(device=idx.device).manual_seed(cfg.routing_seed + self.row_offset + row)
            noise.append(torch.randn((1, *x.shape[1:]), device=x.device, dtype=x.dtype, generator=g))
        x = torch.cat(noise, 0) * cfg.width ** -0.5
        total = cfg.pre_layers + n * cfg.core_layers + cfg.post_layers

        grad_start = n - cfg.huginn_backprop_loops
        n_grad = max(n - grad_start, 0)
        lm_on = use_loop_muon and torch.is_grad_enabled()
        ro_on = use_row_orth and torch.is_grad_enabled()
        if lm_on:
            lm_hook.begin(n_grad)
        if ro_on:
            ROW_CTX.begin(n_grad)

        hist_macro = None
        hist_mid = None
        u_prev = None
        past_updates = []
        past_states = [x]
        raw_u_list = []

        for r in range(n):
            no_grad = r < grad_start
            with torch.no_grad() if no_grad else nullcontext():
                if r == grad_start:
                    x = x.detach()
                x = norm(self.state_adapter(torch.cat((x, prelude), -1)))
                u_in = x

                if lm_on:
                    lm_hook.set_loop((r - grad_start) if not no_grad else None)
                if ro_on:
                    ROW_CTX.set_active(not no_grad)
                try:
                    for j in range(cfg.core_layers):
                        if dedup_mid is not None and j == 0:
                            mid_in = x
                        x = self._block(x, x0, idx, cfg.pre_layers + j,
                                        cfg.pre_layers + r * cfg.core_layers + j, total)
                        if dedup_mid is not None and j == mid_j:
                            if r == grad_start:
                                hist_mid = None
                            u_m = x - mid_in
                            u_m, hist_mid = dedup_mid(u_m, hist_mid)
                            x = mid_in + u_m
                finally:
                    if lm_on:
                        lm_hook.set_loop(None)
                    if ro_on:
                        ROW_CTX.set_active(False)

                u_raw = x - u_in
                if not self.training:
                    raw_u_list.append(u_raw.detach().float())

                u = u_raw
                if dedup_macro is not None:
                    if r == grad_start:
                        hist_macro = None
                    u, hist_macro = dedup_macro(u, hist_macro)

                if use_adj_dedup:
                    if r == grad_start:
                        u_prev = None
                    if u_prev is not None:
                        uf, pf = u.float(), u_prev.float()
                        dot = (uf * pf).sum(-1, keepdim=True)
                        den = pf.pow(2).sum(-1, keepdim=True).clamp_min(1e-6)
                        u = u - (gamma * (dot / den) * pf).to(u.dtype)
                    u_prev = u

                if use_signed_krylov:
                    # Mix past updates with unconstrained signed weights
                    if r > 0:
                        row_idx = min(r, cfg.max_loops - 1)
                        offset = row_idx * (row_idx - 1) // 2
                        k_prev = len(past_updates)
                        # If eval depth > max_loops, apply last row's weights to the most recent max_loops-1 updates
                        src = past_updates[-row_idx:]
                        ws = self.krylov_alpha[offset:offset + row_idx]
                        mix = sum(ws[k].to(u.dtype) * src[k] for k in range(len(src)))
                        u = u + mix
                    past_updates.append(u)

                x = u_in + u

                if use_loop_attnres:
                    past_states.append(x)
                    q = self.loop_queries[min(r, cfg.max_loops - 1)]
                    keys = torch.stack([norm(s) for s in past_states], dim=-2)       # [B, T, k, d]
                    vals = torch.stack(past_states, dim=-2)
                    scores = (keys.float() * q.float()).sum(-1) / math.sqrt(cfg.width)
                    w = scores.softmax(dim=-1).to(x.dtype)
                    x = (vals * w.unsqueeze(-1)).sum(-2)

        if not self.training and len(raw_u_list) >= 2:
            # Record loop-end collapse statistic
            u_last = raw_u_list[-1].reshape(-1, cfg.width)
            U = torch.stack([u.reshape(-1, cfg.width) for u in raw_u_list[:-1]], -1)
            G = U.transpose(-1, -2) @ U
            k = U.shape[-1]
            scale = G.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
            eye = torch.eye(k, device=u_last.device, dtype=torch.float32)
            b = (U.transpose(-1, -2) @ u_last.unsqueeze(-1)).squeeze(-1)
            coeff = torch.linalg.solve(G + (1e-6 * scale)[..., None, None] * eye, b.unsqueeze(-1))
            proj = (U @ coeff).squeeze(-1)
            frac = (proj.pow(2).sum(-1) / u_last.pow(2).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
            self.last_stats["last_loop_span_frac"] = float(frac.mean().item())

        for j in range(cfg.post_layers):
            x = self._block(x, x0, idx, cfg.pre_layers + cfg.core_layers + j,
                            cfg.pre_layers + n * cfg.core_layers + j, total)
        return self._readout(x, prelude), x.new_zeros(()), None

    model._loop_hidden = types.MethodType(_arena_loop_hidden, model)


def build_arena_optimizer(model: ReferenceLM, batch_tokens: int):
    """Exact nexttransformer Muon+AdamW parameter grouping, scaled cleanly for batch_tokens."""
    from collections import defaultdict
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

    # Custom idea auxiliary parameters (never put tiny vectors/queries into Muon)
    idea_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in ("krylov_alpha", "loop_queries"))]
    adam(idea_params, 0.02 * bscale, (0.9, 0.95), 0.0)

    allocated = {id(p) for g in groups for p in g["params"]}
    matrices = defaultdict(list)
    other = []
    for p in model.parameters():
        if not p.requires_grad or id(p) in allocated:
            continue
        if p.ndim == 2:
            matrices[tuple(p.shape)].append(p)
        else:
            other.append(p)
    adam(other, adamw_lr, (0.9, 0.95), 0.0)
    for shape in sorted(matrices):
        groups.append(dict(kind="muon", params=matrices[shape], lr=muon_lr, initial_lr=muon_lr,
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
def eval_multidepth(model, val_x, val_y, byte_table, device, batch_size: int,
                    depths=(4, 8, 16), max_rows: int = 1024):
    model.eval()
    old_eval = model.config.eval_loops
    out = {}
    collapse_r8 = None
    try:
        for d in depths:
            object.__setattr__(model.config, "eval_loops", d)
            total_nats = 0.0
            total_bytes = 0.0
            fracs = []
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
                if d == 8 and "last_loop_span_frac" in model.last_stats:
                    fracs.append(model.last_stats["last_loop_span_frac"])
            out[str(d)] = total_nats / (math.log(2.0) * max(total_bytes, 1.0))
            if d == 8 and fracs:
                collapse_r8 = sum(fracs) / len(fracs)
    finally:
        object.__setattr__(model.config, "eval_loops", old_eval)
        model.train()
    return out, collapse_r8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("/dev/shm/rsi_arena_data"))
    p.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--micro-batch", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--core-layers", type=int, default=6)
    args = p.parse_args()

    assert args.batch_size % args.micro_batch == 0
    accum = args.batch_size // args.micro_batch

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_data = torch.load(args.data_dir / "train.pt", map_location="cpu", weights_only=True)
    val_data = torch.load(args.data_dir / "val.pt", map_location="cpu", weights_only=True)
    train_x, train_y = train_data["x"], train_data["y"]
    val_x, val_y = val_data["x"], val_data["y"]
    seq_len = train_x.shape[1]
    byte_table = torch.load(args.resources / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    cfg = ModelConfig(
        method="huginn",
        scale="debug",
        width=args.width,
        heads=args.heads,
        vocab_size=32768,
        sequence_len=seq_len,
        mlp_width=4 * args.width,
        layers=1 + args.core_layers + 1,
        pre_layers=1,
        core_layers=args.core_layers,
        post_layers=1,
        max_loops=8,
        huginn_mean=40.0,
        huginn_sigma=0.5,
        huginn_backprop_loops=4,
        eval_loops=8,
    )
    model = build_model(cfg, device=device, seed=args.seed)
    attach_idea(model, args.idea)

    n_params = sum(t.numel() for t in model.parameters() if t.requires_grad)
    batch_tokens = args.batch_size * seq_len
    opt = build_arena_optimizer(model, batch_tokens)

    t_start = time.time()
    history = []
    n_rows = train_x.shape[0]

    # Step 0 quick check (only r=8, 256 rows -> ~0.3s)
    bpb0, col0 = eval_multidepth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                 depths=(8,), max_rows=512)
    print(f"[{args.idea}] params={n_params/1e6:.1f}M step=0 val_bpb@8={bpb0['8']:.5f} collapse={col0:.4f}",
          flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    step_times = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        mult = lr_mult(step, args.steps)
        for g in opt.param_groups:
            g["lr"] = g["initial_lr"] * mult
            if "initial_weight_decay" in g:
                g["weight_decay"] = g["initial_weight_decay"] * mult

        opt.zero_grad(set_to_none=True)
        model.set_step(step)
        loss_sum = 0.0
        for m in range(accum):
            row0 = (step * args.batch_size + m * args.micro_batch) % (n_rows - args.micro_batch)
            x = train_x[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            y = train_y[row0:row0 + args.micro_batch].to(device, dtype=torch.long, non_blocking=True)
            model.set_row_offset(step * args.batch_size + m * args.micro_batch)
            loss = model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach().item() / accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        step_times.append(dt)

        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            avg_dt = sum(step_times[-50:]) / len(step_times[-50:])
            print(f"[{args.idea}] step={step+1}/{args.steps} loss={loss_sum:.4f} "
                  f"gnorm={float(grad_norm):.3f} tok/s={batch_tokens/avg_dt/1e3:.1f}k "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            depths = (4, 8, 16) if (step + 1 == args.steps) else (8,)
            rows = 4096 if (step + 1 == args.steps) else 1024
            bpbs, col = eval_multidepth(model, val_x, val_y, byte_table, device, args.micro_batch,
                                        depths=depths, max_rows=rows)
            rec = {
                "step": step + 1,
                "tokens": (step + 1) * batch_tokens,
                "train_loss": loss_sum,
                "val_bpb": bpbs,
                "collapse_r8": col,
                "elapsed_s": time.time() - t_start,
            }
            history.append(rec)
            print(f"[{args.idea}] VAL step={step+1} bpb={ {k: round(v, 5) for k, v in bpbs.items()} } "
                  f"collapse@8={col:.4f}", flush=True)

    avg_dt = sum(step_times[10:]) / max(len(step_times[10:]), 1)
    extra = {}
    if hasattr(model, "krylov_alpha"):
        extra["krylov_alpha"] = [round(v, 4) for v in model.krylov_alpha.detach().cpu().tolist()]
    summary = {
        "idea": args.idea,
        "params": n_params,
        "steps": args.steps,
        "tokens": args.steps * batch_tokens,
        "seed": args.seed,
        "sec_per_step": avg_dt,
        "tok_per_sec": batch_tokens / avg_dt,
        "peak_mem_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "total_wall_s": time.time() - t_start,
        "final_val_bpb": history[-1]["val_bpb"],
        "final_collapse_r8": history[-1]["collapse_r8"],
        "history": history,
        "extra": extra,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{args.idea}] DONE in {summary['total_wall_s']:.1f}s | "
          f"final val_bpb={ {k: round(v, 5) for k, v in summary['final_val_bpb'].items()} } | "
          f"collapse@8={summary['final_collapse_r8']:.4f} | "
          f"mem={summary['peak_mem_gb']:.1f}GB | {summary['tok_per_sec']/1e3:.1f}k tok/s", flush=True)


if __name__ == "__main__":
    main()
