"""Measure cross-loop collapse, per-layer Ahat_perp spectra, per-loop gradient geometry,
and stateless multi-depth BPB on a trained nexttransformer Huginn checkpoint.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("NEXTTRANSFORMER_DIR", "./NextTransformer"))
from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import build_model, norm
from nexttransformer.methods.loop_muon import loop_delta_apply, loop_delta_operator
from nexttransformer.runtime.data import BestFitStream
from nexttransformer.runtime.checkpoint import load_checkpoint


def effective_rank_from_evals(evals: torch.Tensor) -> torch.Tensor:
    p = evals.clamp_min(0.0)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1)
    return ent.exp()


def measure_collapse_and_spectra(model, batches, device, dedup_gamma: float = 0.0):
    """Capture macro-block updates u_0..u_{r-1} and per-layer norm(x) over the 4 grad loops."""
    c = model.config
    n = c.eval_loops
    grad_start = n - c.huginn_backprop_loops
    n_core = c.core_layers

    # Accumulators
    macro_span_num = [0.0] * n
    macro_span_cnt = [0] * n
    macro_cos_sum = [0.0] * n
    macro_unorm_sum = [0.0] * n

    # Per-layer [n_core] for attn and mlp
    layer_stats = {
        kind: {
            "perp_ratio": [0.0] * n_core,
            "evals": [torch.zeros(3) for _ in range(n_core)],
            "eff_rank": [0.0] * n_core,
            "count": 0,
        }
        for kind in ("attn", "mlp")
    }

    # Register light forward pre-hooks on core c_q (attn input) and c_fc (mlp input)
    captured = {"attn": {}, "mlp": {}}
    cur_loop = {"r": -1}
    handles = []

    def make_hook(kind, j):
        def _fn(_mod, inp):
            r = cur_loop["r"]
            if r >= grad_start:
                captured[kind].setdefault(j, {})[r - grad_start] = inp[0].detach()
        return _fn

    for j in range(n_core):
        blk = model.transformer.h[c.pre_layers + j]
        handles.append(blk.attn.c_q.register_forward_pre_hook(make_hook("attn", j)))
        handles.append(blk.mlp.c_fc.register_forward_pre_hook(make_hook("mlp", j)))

    from nexttransformer.methods.loop_dedup import LoopDedup
    dedup = LoopDedup(gamma=dedup_gamma, window=8, ridge=1e-2) if dedup_gamma > 0 else None

    try:
        model.eval()
        with torch.no_grad():
            for b_idx, (idx, _tgt) in enumerate(batches):
                idx = idx.to(device)
                captured["attn"].clear()
                captured["mlp"].clear()

                x = model._embed_reference(idx)
                x0 = x
                for i in range(c.pre_layers):
                    x = model._block(x, x0, idx, i, i, c.maximum_depth)
                prelude = x
                noise = []
                for row in range(idx.shape[0]):
                    g = torch.Generator(device=idx.device).manual_seed(c.routing_seed + b_idx * idx.shape[0] + row)
                    noise.append(torch.randn((1, *x.shape[1:]), device=x.device, dtype=x.dtype, generator=g))
                x = torch.cat(noise, 0) * c.width ** -0.5
                total = c.pre_layers + n * c.core_layers + c.post_layers

                raw_updates = []
                dedup_hist = None
                for r in range(n):
                    cur_loop["r"] = r
                    x = norm(model.state_adapter(torch.cat((x, prelude), -1)))
                    u_in = x
                    for j in range(n_core):
                        x = model._block(x, x0, idx, c.pre_layers + j, c.pre_layers + r * n_core + j, total)
                    u_raw = x - u_in
                    raw_updates.append(u_raw.float())

                    if dedup is not None:
                        if r == grad_start:
                            dedup_hist = None
                        u_mod, dedup_hist = dedup(u_raw, dedup_hist)
                        x = u_in + u_mod

                cur_loop["r"] = -1

                # 1. Macro-block span energy fraction & adjacent cosine
                # Use raw_updates: [B, T, d]
                prev_list = []
                for r in range(n):
                    u = raw_updates[r].reshape(-1, c.width)       # [N, d]
                    unorm = u.norm(dim=-1)
                    macro_unorm_sum[r] += unorm.mean().item()
                    if r > 0:
                        prev_u = raw_updates[r - 1].reshape(-1, c.width)
                        cos = (u * prev_u).sum(-1) / (unorm * prev_u.norm(dim=-1)).clamp_min(1e-12)
                        macro_cos_sum[r] += cos.mean().item()
                        # Project u onto span(prev_list) via ridge=1e-6 least squares per token
                        U = torch.stack(prev_list, -1)            # [N, d, r]
                        G = U.transpose(-1, -2) @ U               # [N, r, r]
                        scale = G.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
                        eye = torch.eye(r, device=device, dtype=torch.float32)
                        b = (U.transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1)
                        coeff = torch.linalg.solve(G + (1e-6 * scale)[..., None, None] * eye, b.unsqueeze(-1))
                        proj = (U @ coeff).squeeze(-1)
                        frac = (proj.pow(2).sum(-1) / u.pow(2).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
                        macro_span_num[r] += frac.mean().item()
                        macro_span_cnt[r] += 1
                    prev_list.append(u)

                # 2. Per-layer Ahat_perp spectrum across the 4 grad loops
                s = c.huginn_backprop_loops
                for kind in ("attn", "mlp"):
                    layer_stats[kind]["count"] += 1
                    for j in range(n_core):
                        xs = [captured[kind][j][t].float().reshape(-1, c.width) for t in range(s)]
                        H = torch.stack(xs, -1)                   # [N, d, s]
                        hbar = H.mean(-1, keepdim=True)
                        e = hbar / hbar.norm(dim=-2, keepdim=True).clamp_min(1e-12)
                        perp = H - e * (e.transpose(-1, -2) @ H)
                        ratio = (perp.norm(dim=(-2, -1)) / H.norm(dim=(-2, -1)).clamp_min(1e-12)).mean().item()
                        A = perp.transpose(-1, -2) @ perp         # [N, s, s]
                        tau = A.diagonal(dim1=-2, dim2=-1).sum(-1) / (s - 1)
                        Ahat = A / tau.clamp_min(1e-12)[..., None, None]
                        evals = torch.linalg.eigvalsh(Ahat).clamp_min(0.0)  # ascending [N, s]
                        top3 = evals[:, -3:].flip(-1)             # descending [N, 3]
                        erank = effective_rank_from_evals(top3).mean().item()
                        layer_stats[kind]["perp_ratio"][j] += ratio
                        layer_stats[kind]["evals"][j] += top3.mean(0).cpu()
                        layer_stats[kind]["eff_rank"][j] += erank
    finally:
        for h in handles:
            h.remove()

    nb = len(batches)
    macro = []
    for r in range(n):
        macro.append({
            "loop": r,
            "mean_update_norm": macro_unorm_sum[r] / nb,
            "adj_cosine": (macro_cos_sum[r] / nb) if r > 0 else None,
            "span_energy_frac": (macro_span_num[r] / macro_span_cnt[r]) if r > 0 else None,
        })

    spectra = {}
    for kind in ("attn", "mlp"):
        cnt = layer_stats[kind]["count"]
        spectra[kind] = [
            {
                "core_layer": j,
                "perp_ratio": layer_stats[kind]["perp_ratio"][j] / cnt,
                "evals": (layer_stats[kind]["evals"][j] / cnt).tolist(),
                "eff_rank": layer_stats[kind]["eff_rank"][j] / cnt,
            }
            for j in range(n_core)
        ]
    return macro, spectra


def measure_per_loop_gradients(model, batch, device, dedup_gamma: float = 0.0,
                               sample_layers=None):
    """Run one forward+backward on sampled core layers and inspect G_0..G_3 vs Loop-Muon G~."""
    c = model.config
    s = c.huginn_backprop_loops
    if sample_layers is None:
        sample_layers = tuple(sorted({0, c.core_layers // 2, c.core_layers - 1}))

    os.environ["LOOP_DEDUP_GAMMA"] = str(dedup_gamma)
    idx, tgt = batch[0].to(device), batch[1].to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    # Capture per-loop (x_t, dy_t) for sampled layers' c_q, c_v, c_fc, c_proj
    targets = []
    for j in sample_layers:
        blk = model.transformer.h[c.pre_layers + j]
        for sub_name, mod in [
            ("attn.c_q", blk.attn.c_q),
            ("attn.c_v", blk.attn.c_v),
            ("mlp.c_fc", blk.mlp.c_fc),
            ("mlp.c_proj", blk.mlp.c_proj),
        ]:
            targets.append((f"core{j}.{sub_name}", mod))

    store_x = {name: [] for name, _ in targets}
    store_dy = {name: [] for name, _ in targets}
    handles = []

    def make_fwd(name):
        def _fn(_mod, inp, _out):
            if torch.is_grad_enabled():
                store_x[name].append(inp[0].detach())
        return _fn

    def make_bwd(name):
        def _fn(_mod, _gin, gout):
            store_dy[name].append(gout[0].detach())
        return _fn

    for name, mod in targets:
        handles.append(mod.register_forward_hook(make_fwd(name)))
        handles.append(mod.register_full_backward_hook(make_bwd(name)))

    try:
        model.set_step(0)
        model.set_row_offset(0)
        loss = model(idx, tgt)
        loss.backward()
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)

    results = {}
    for name, _ in targets:
        xs = store_x[name]                                        # forward order: t = 0..s-1
        dys = list(reversed(store_dy[name]))                      # backward is reverse -> flip to 0..s-1
        assert len(xs) == s and len(dys) == s, (name, len(xs), len(dys))
        Gs = []
        for t in range(s):
            dy = dys[t].reshape(-1, dys[t].shape[-1]).float()
            x = xs[t].reshape(-1, xs[t].shape[-1]).float()
            Gs.append(dy.t() @ x)
        norms = [g.norm().item() for g in Gs]
        cos_mat = [[0.0] * s for _ in range(s)]
        for i in range(s):
            for j in range(s):
                cos_mat[i][j] = (Gs[i] * Gs[j]).sum().item() / max(norms[i] * norms[j], 1e-12)

        G_raw = sum(Gs)
        C = loop_delta_operator(xs, gamma=0.5, eps=1e-2)
        G_delta = torch.zeros_like(G_raw)
        for t in range(s):
            dx = loop_delta_apply(xs, t, C).reshape(-1, xs[t].shape[-1]).float()
            dy = dys[t].reshape(-1, dys[t].shape[-1]).float()
            G_delta = G_delta + dy.t() @ dx
        G_muon = G_raw + G_delta

        s_raw = torch.linalg.svdvals(G_raw)
        s_muon = torch.linalg.svdvals(G_muon)
        erank_raw = effective_rank_from_evals(s_raw.square()).item()
        erank_muon = effective_rank_from_evals(s_muon.square()).item()
        align_cos = (G_muon * G_raw).sum().item() / max(G_muon.norm().item() * G_raw.norm().item(), 1e-12)
        delta_rel = G_delta.norm().item() / max(G_raw.norm().item(), 1e-12)

        results[name] = {
            "per_loop_norms": norms,
            "norm_ratio_last_to_first": norms[-1] / max(norms[0], 1e-12),
            "cos_matrix": cos_mat,
            "mean_offdiag_cos": sum(cos_mat[i][j] for i in range(s) for j in range(s) if i != j) / (s * (s - 1)),
            "raw_grad_eff_rank": erank_raw,
            "muon_grad_eff_rank": erank_muon,
            "batch_alignment_cos": align_cos,
            "delta_to_raw_norm": delta_rel,
        }
    return results


@torch.no_grad()
def measure_multidepth_bpb(model, batches, byte_table, device, depths=(1, 2, 4, 8, 12, 16),
                           dedup_gamma: float = 0.0):
    """Stateless multi-depth BPB on identical fixed batches."""
    os.environ["LOOP_DEDUP_GAMMA"] = str(dedup_gamma)
    model.eval()
    old_eval = model.config.eval_loops
    out = {}
    try:
        for d in depths:
            object.__setattr__(model.config, "eval_loops", d)
            total_nats = 0.0
            total_bytes = 0.0
            for b_idx, (idx, tgt) in enumerate(batches):
                idx, tgt = idx.to(device), tgt.to(device)
                model.set_row_offset(b_idx * idx.shape[0])
                logits = model(idx)
                loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=-1, reduction="none")
                nbytes = byte_table[tgt.flatten().clamp_min(0)]
                valid = (tgt.flatten() >= 0) & (nbytes > 0)
                total_nats += loss[valid].sum().item()
                total_bytes += nbytes[valid].sum().item()
            out[str(d)] = total_nats / (math.log(2.0) * max(total_bytes, 1.0))
    finally:
        object.__setattr__(model.config, "eval_loops", old_eval)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--resources", type=Path, default=Path("/dev/shm/nt-runtime/resources"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dedup-gamma", type=float, default=0.0)
    p.add_argument("--n-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    os.environ["LOOP_DEDUP_GAMMA"] = str(args.dedup_gamma)
    os.environ["LOOP_MUON_GAMMA"] = "0"

    saved = torch.load(args.ckpt / "model.pt", map_location="cpu", weights_only=False)
    cfg = ModelConfig(**saved["model_config"])
    model = build_model(cfg, device=device)
    load_checkpoint(args.ckpt, model)

    from nanochat.tokenizer import get_tokenizer
    os.environ["NANOCHAT_BASE_DIR"] = str(args.resources)
    tokenizer = get_tokenizer()
    byte_table = torch.load(args.resources / "tokenizer/token_bytes.pt", map_location=device, weights_only=True)

    stream = BestFitStream(args.data_dir, tokenizer, "val", 0, 1, cfg.sequence_len)
    batches = []
    for _ in range(args.n_batches):
        x, y = stream.batch(args.batch_size, torch.device("cpu"))
        batches.append((x, y))

    macro, spectra = measure_collapse_and_spectra(model, batches, device, dedup_gamma=args.dedup_gamma)
    grads = measure_per_loop_gradients(model, batches[0], device, dedup_gamma=args.dedup_gamma)
    bpb = measure_multidepth_bpb(model, batches, byte_table, device, dedup_gamma=args.dedup_gamma)

    report = {
        "ckpt": str(args.ckpt),
        "step": saved["state"]["step"],
        "dedup_gamma": args.dedup_gamma,
        "macro_collapse": macro,
        "layer_spectra": spectra,
        "per_loop_gradients": grads,
        "multidepth_bpb": bpb,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"=== Diagnostics ({args.ckpt.name}, dedup_gamma={args.dedup_gamma}) ===")
    print("Macro update collapse (span_energy_frac / adj_cosine):")
    for m in macro:
        sf = f"{m['span_energy_frac']:.4f}" if m["span_energy_frac"] is not None else "  --  "
        ac = f"{m['adj_cosine']:+.4f}" if m["adj_cosine"] is not None else "   --  "
        print(f"  loop {m['loop']}: span_frac={sf}  adj_cos={ac}  ||u||={m['mean_update_norm']:.3f}")
    for kind in ("attn", "mlp"):
        mean_r = sum(x["perp_ratio"] for x in spectra[kind]) / len(spectra[kind])
        mean_er = sum(x["eff_rank"] for x in spectra[kind]) / len(spectra[kind])
        mean_e = [sum(x["evals"][k] for x in spectra[kind]) / len(spectra[kind]) for k in range(3)]
        print(f"Layer {kind:4s} inputs: mean ||H_perp||/||H||={mean_r:.4f}  "
              f"mean evals={[round(v, 3) for v in mean_e]}  mean eff_rank={mean_er:.3f}/3.0")
    print("Per-loop gradients (sample layers):")
    for k, v in grads.items():
        print(f"  {k:18s}: last/first={v['norm_ratio_last_to_first']:.2f}  "
              f"offdiag_cos={v['mean_offdiag_cos']:+.3f}  "
              f"erank {v['raw_grad_eff_rank']:.1f}->{v['muon_grad_eff_rank']:.1f}  "
              f"batch_align={v['batch_alignment_cos']:+.4f}  delta/raw={v['delta_to_raw_norm']:.3f}")
    print("Stateless multi-depth BPB:", {k: round(v, 5) for k, v in bpb.items()})


if __name__ == "__main__":
    main()
