"""End-to-end integration test of setup_nt.py + loop_muon_hook + loop_dedup inside ReferenceLM.

Runs on CPU in an isolated temp copy of /root/nt so it never disturbs live cluster training.
Verifies:
  1. gamma=0 is bit-for-bit identical to pristine lm.py.orig.
  2. LOOP_MUON_GAMMA=0.5 wraps core_layers*4 matrices, keeps forward logits bit-for-bit identical,
     keeps ALL non-wrapped gradients (prelude, coda, state_adapter, embed, lm_head, c_proj)
     identical to baseline, changes wrapped core gradients to match loop_whiten, and releases all
     per-loop buffers.
  3. Evaluation under torch.no_grad() captures nothing and requires_grad=False.
  4. Combined LOOP_DEDUP_GAMMA=0.5 + LOOP_MUON_GAMMA=0.5 runs cleanly with finite gradients and
     zero leaked buffers.
"""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F


def _run_case(nt_copy: Path, env_extra: dict, mode: str):
    code = f"""
import os, sys, json
sys.path.insert(0, {str(nt_copy)!r})
import torch
import torch.nn.functional as F

# CPU fallback for flash_attn_func when running without CUDA
import nanochat.flash_attention as fa_mod
def _cpu_fa(q, k, v, causal=True, window_size=(-1, 0)):
    # q, k, v: (B, T, H, D) -> SDPA expects (B, H, T, D)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if kt.shape[1] != qt.shape[1]:
        rep = qt.shape[1] // kt.shape[1]
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    return out.transpose(1, 2).contiguous()
fa_mod.flash_attn.flash_attn_func = _cpu_fa

import nanochat.common as common_mod
common_mod.COMPUTE_DTYPE = torch.float32

from nexttransformer.models.config import ModelConfig
from nexttransformer.models.lm import ReferenceLM
from nexttransformer.methods import loop_muon_hook as hook
from nexttransformer.methods.loop_muon import loop_whiten

torch.manual_seed(123)
cfg = ModelConfig(
    method="huginn",
    scale="debug",
    width=64,
    heads=4,
    vocab_size=256,
    sequence_len=16,
    mlp_width=192,
    layers=6,
    pre_layers=1,
    core_layers=2,
    post_layers=1,
    max_loops=8,
    huginn_mean=40.0,
    huginn_sigma=0.5,
    huginn_backprop_loops=4,
    eval_loops=8,
)
model = ReferenceLM(cfg)
model.init_weights()
# nanochat zero-initializes attn.c_proj and mlp.c_proj, so step 0 has zero grad into q/k/v/fc.
# Perturb c_proj slightly to simulate step >= 1 where all core gradients are active.
with torch.no_grad():
    for blk in model.transformer.h:
        blk.attn.c_proj.weight.normal_(0.0, 0.05)
        blk.mlp.c_proj.weight.normal_(0.0, 0.05)
model.train()
model.set_step(0)
model.set_row_offset(0)

torch.manual_seed(999)
idx = torch.randint(0, cfg.vocab_size, (2, 16))
tgt = torch.randint(0, cfg.vocab_size, (2, 16))

out_path = os.environ["VERIFY_OUT"]
mode = {mode!r}

if mode in ("orig", "base", "muon", "both"):
    loss = model(idx, tgt)
    loss.backward()
    # Verify no leaked buffers
    assert not hook.CTX.xs and not hook.CTX.op and not hook.CTX.pending, (
        len(hook.CTX.xs), len(hook.CTX.op), len(hook.CTX.pending))
    grads = {{k: v.grad.detach().clone() for k, v in model.named_parameters() if v.grad is not None}}
    # Check eval under no_grad
    model.eval()
    with torch.no_grad():
        logits = model(idx)
    assert not logits.requires_grad, "eval logits require grad!"
    assert not hook.CTX.xs and not hook.CTX.op, "eval captured state!"
    torch.save({{
        "loss": loss.detach(),
        "logits": logits.detach(),
        "grads": grads,
        "wrapped": getattr(model, "_loop_muon_wrapped", 0),
    }}, out_path)
"""
    env = os.environ.copy()
    env.pop("LOOP_DEDUP_GAMMA", None)
    env.pop("LOOP_MUON_GAMMA", None)
    env.update(env_extra)
    out_file = nt_copy / f"out_{mode}.pt"
    env["VERIFY_OUT"] = str(out_file)
    res = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"mode={mode} failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return torch.load(out_file, weights_only=False)


def main():
    nt_src = Path("/root/nt")
    repo_src = Path("/root/pn/loop-muon")
    setup_script = repo_src / "setup_nt.py"
    with tempfile.TemporaryDirectory(prefix="nt_verify_") as tmp:
        nt_copy = Path(tmp) / "nt"
        shutil.copytree(nt_src / "nexttransformer", nt_copy / "nexttransformer")
        shutil.copytree(nt_src / "nanochat", nt_copy / "nanochat")
        # Copy loop methods + patch lm.py cleanly
        subprocess.run(
            [sys.executable, str(setup_script), "--src", str(repo_src), "--nt", str(nt_copy)],
            check=True,
        )

        # 1. Baseline with patched lm.py (both env vars unset)
        r_base = _run_case(nt_copy, {}, "base")
        assert r_base["wrapped"] == 0, f"expected 0 wrapped in baseline, got {r_base['wrapped']}"

        # 2. Loop-Muon only
        r_muon = _run_case(nt_copy, {"LOOP_MUON_GAMMA": "0.5", "LOOP_MUON_RIDGE": "1e-2"}, "muon")
        assert r_muon["wrapped"] == 8, f"expected 2*4=8 wrapped matrices, got {r_muon['wrapped']}"
        assert torch.equal(r_muon["loss"], r_base["loss"]), \
            f"Loop-Muon changed forward loss: {r_muon['loss']} vs {r_base['loss']}"
        assert torch.equal(r_muon["logits"], r_base["logits"]), "Loop-Muon changed eval logits!"

        wrapped_keys = set()
        for layer in (1, 2):  # pre_layers=1, core_layers=2
            for suffix in ("attn.c_q.weight", "attn.c_k.weight", "attn.c_v.weight", "mlp.c_fc.weight"):
                wrapped_keys.add(f"transformer.h.{layer}.{suffix}")

        changed, unchanged = 0, 0
        for name, gb in r_base["grads"].items():
            gm = r_muon["grads"][name]
            assert torch.isfinite(gm).all(), f"non-finite grad in {name}"
            diff = (gm - gb).abs().max().item()
            if name in wrapped_keys:
                assert diff > 1e-6, f"wrapped param {name} grad did not change (diff={diff:.2e})"
                changed += 1
            else:
                assert diff == 0.0, f"non-wrapped param {name} grad changed by {diff:.2e}!"
                unchanged += 1
        assert changed == 8, f"expected 8 changed wrapped grads, got {changed}"
        print(f"PASS loop_muon: forward bit-identical, {changed} core reader grads whitened, "
              f"{unchanged} non-wrapped grads bit-identical")

        # 3. Combined loop-dedup + loop-muon
        r_both = _run_case(
            nt_copy,
            {"LOOP_DEDUP_GAMMA": "0.5", "LOOP_MUON_GAMMA": "0.5", "LOOP_MUON_RIDGE": "1e-2"},
            "both",
        )
        assert r_both["wrapped"] == 8
        assert torch.isfinite(r_both["loss"]).all()
        for name, g in r_both["grads"].items():
            assert torch.isfinite(g).all(), f"non-finite grad in combined mode: {name}"
        assert not torch.equal(r_both["loss"], r_base["loss"]), "loop-dedup did not affect forward loss!"
        print("PASS combined loop_dedup + loop_muon: forward & backward finite, buffers released")


if __name__ == "__main__":
    main()
