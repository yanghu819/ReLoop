# Research Log — Scalable Looped Language Models

Single source of truth for all experiments, diagnostics, hypotheses, and negative results on the 16×B300 cluster (`a1`, `a2`).

## Design Principles (Kaiming He Standard)
1. **Simplicity**: Every method must reduce to a clean, inevitable core (~5–20 lines of PyTorch), no fragile hyperparameter jungles.
2. **Scalability**: Must hold (or grow stronger) as model size increases (`480M → 1.32B → 2.1B+`), loop depth increases (`r = 8 → 16`), and training extends **past Chinchilla (`>20 tok/param`)** with real **22-task CORE** downstream transfer.
3. **Honest Pairing**: Identical seed (`42`), identical 301-shard ClimbMix inventory (`3fbc9325…`), identical world size (`8` GPUs per arm). Report all negative results verbatim.

---

## Experiment Matrix

### Series A: 1.32B Huginn (`huginn_r8_1b3.json`, 3+31+3 layers, `r=8, s=4`, 1M tok/step, Muon+AdamW)

| Arm | Box | Config / Env | Step 0 val_bpb | Step 250 val_bpb | Step 500 val_bpb | Δ vs Base (500) | s/step | Peak GB | Status |
|---|---|---|---|---|---|---|---|---|---|
| `big-base` | `a1` | baseline | `3.16875` | `1.02648` | `0.96596` | — | 13.88 | 201.9 | Paused at `checkpoint_00000500` (resumable) |
| `big-dedup` | `a2` | `LOOP_DEDUP_GAMMA=0.5` | `3.16890` | **`1.01893`** (−0.00755) | **`0.96058`** | **−0.00538** | 14.07 | 202.5 | **Running uninterrupted** (past step 510 → 1000+) |

### Major Diagnostic Discovery on 1.32B `big-base` (`checkpoint_00000500`)
1. **Exact 100.00% macro-update collapse by loop 5**:
   - `span_energy_frac` across loops 1..7: `0.9156, 0.9904, 0.9968, 0.9990, 1.0000, 1.0000, 1.0000` (`adj_cos = +0.9960`).
   - **Stateless multi-depth BPB (`r = 1, 2, 4, 8, 12, 16`)**: `0.99767, 0.99543, 0.99119, 0.99118, 0.99118, 0.99120` — **loops 5..16 do literally ZERO work!**
2. **Root-cause architectural pathology discovered in `nexttransformer` Huginn (`lm.py:219`)**:
   - `x0_lambdas` (residual skip from raw embedding `x0`, lr=0.5) blew up to **`6.3–7.0`** in `big-base` and **`6.8–9.7`** in `big-dedup` across core layers 0..24 (vs `resid_lambdas ≈ 0.94`), overwhelming the Post-Norm `SandwichBlock` recurrent state by `(0.94 / 6.5)^25 ≈ 10^{-21}`!
   - Result: **core layers 0..24 had literal `0.0` gradient** on all 4 BPTT loops; only the last ~5 core layers (`core26..30`) + loop 7 carried gradient!
   - **Actionable RSI insight**: In Round 2 (after Round 1 finishes in ~12 min), test **disabling `x0_lambdas` short-circuit in the recurrent core** + **true Pre-Norm / clean residual scaling** vs Huginn `SandwichBlock`!

### Series B: 8-GPU Parallel RSI Arena on `a1` (`208.8M` Huginn `r=8, s=4`, 1 GPU/idea, 167.8M pre-tokenized ClimbMix)
- **Round 1** (launched 17:50, 400 steps = 52.4M tok, 8×B300 at 98–100% util):
  `base`, `dedup`, `adj_dedup`, `loop_muon`, `dedup_muon`, `subblock_dedup`, `signed_krylov`, `row_orth_muon`.
  (25-step smoke already showed `loop_muon` −0.00189 bpb & `row_orth_muon` −0.00200 bpb at **0.4% overhead**!)

**Pre-registered bar for `Loop-Muon` v1**: standalone recovers `>= 30%` of same-run `loop-dedup` Δ, OR combined adds `>= +25%` on top of `loop-dedup`.

---

## RSI Hypothesis Queue (Beyond Current Idea)

Driven by 1.32B step-500 real diagnostics (`diagnose_loop_collapse.py`), then rapidly screened on `huginn_r8` (~480M, 18 min/val, 1.1h/1B tok, 100% GPU util) before scaling winners to 1.32B & past-Chinchilla:

1. **H1 (Simplification test — 1-step vs subspace dedup)**: Does a 2-line adjacent-loop orthogonalization (`u_t - γ proj(u_t, u_{t-1})`, O(d), zero matrix solve, no window) match full subspace `loop-dedup`, or does multi-step `span(u_{<t})` genuinely win?
2. **H2 (Negative vs Positive cross-loop mixing)**: Contrast `loop-dedup` (negative projection pushing *out* of collapsed span) vs `Loop-AttnRes` (positive softmax mixing over past loop states) vs combined — testing whether escaping fixed-point collapse requires negative mixing.
3. **H3 (Multi-scale / interior dedup vs macro-boundary dedup)**: Guided by 31-layer `Ahat_perp` diagnostics — test whether deduplicating at sub-block boundaries (e.g., mid-core / per-block) beats macro-only dedup, and whether it fixes the `hbptt` train/eval boundary mismatch.
4. **H4 (Checkpoint-safe / Row-orthogonal / Loop-equalized Optimizer v2)**: Guided by per-loop gradient norm & cosine diagnostics — if `||G_7|| / ||G_4||` is skewed or `Loop-Muon` v1 needs a cleaner checkpoint-safe form, test 4-line streaming row-orthogonalized / norm-equalized loop gradient accumulation.
5. **H5 (Long-training Chinchilla+ & CORE & AdamW generality)**: Run the RSI champion vs baseline past **20+ tok/param** on `huginn_r8` (~9h overnight) with full 22-task CORE evals + stateless multi-depth `r ∈ {1,2,4,8,12,16}` + AdamW generality check, then confirm at **1.32B** (`checkpoint_00000500` resume) and **`r=16`**.
