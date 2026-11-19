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

#### Round 1 Leaderboard (400 steps = 52.4M tok, WSD cooldown, identical seed=42 & data, ~7.7–11.1 min)
| Rank | Idea | val_bpb@8 | Δ@8 vs base | val_bpb@16 | Δ@16 vs base | val_bpb@4 | collapse@8 | tok/s | mem GB |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `row_orth_muon` | **1.10021** | −0.00019 | 1.10024 | −0.00018 | 1.10333 | 0.9996 | 133.1k | 88.4 |
| 2 | `loop_muon` | **1.10034** | −0.00006 | 1.10036 | −0.00006 | 1.10316 | 0.9996 | 93.3k | 88.4 |
| 3 | `base` | **1.10040** | +0.00000 | 1.10042 | +0.00000 | 1.10344 | 0.9996 | 133.4k | 88.6 |
| 4 | `signed_krylov` | **1.10248** | +0.00208 | 1.11155 | +0.01113 | 1.11920 | 0.9964 | 132.0k | 89.3 |
| 5 | `adj_dedup` | **1.11744** | +0.01703 | 1.11805 | +0.01763 | 1.14838 | 0.9713 | 131.5k | 89.7 |
| 6 | `subblock_dedup` | **1.11892** | +0.01852 | 1.12251 | +0.02209 | 1.15862 | 0.8892 | 119.7k | 90.9 |
| 7 | `dedup` | **1.12121** | +0.02080 | 1.12361 | +0.02319 | 1.12895 | 0.9849 | 126.4k | 89.7 |
| 8 | `dedup_muon` | **1.12163** | +0.02123 | 1.12464 | +0.02422 | 1.12669 | 0.9895 | 89.4k | 89.5 |

**Key Round 1 Diagnosis:**
- `x0_lambdas` short-circuit + Post-Norm `SandwichBlock` trapped `base`, `loop_muon`, and `row_orth_muon` at `collapse@8 = 0.9996` and `val_bpb@8 ≈ val_bpb@16` (zero test-time depth scaling!), starving optimizer methods of multi-loop gradients.
- On Post-Norm `SandwichBlock` with `post_layers=1`, subtracting `0.5 * Proj(x_out - x_in)` pulls the last-loop state halfway back toward `x_in`, hurting by `+0.0208 bpb` (and note 2-line `adj_dedup` beat full `dedup` by `−0.0038 bpb`!).

#### Round 2 (launched 18:07, 550 steps = 72.1M tok, ~10.5 min, clean un-short-circuited recurrence!)
- `GPU 0`: `no_x0_huginn` (kill `x0_lambdas` short-circuit in core, keep Huginn `SandwichBlock` + `state_adapter`)
- `GPU 1`: `prenorm_loop` (true Pre-Norm residual looped LM `h_t = h_{t-1} + core(h_{t-1})`, no `x0` short-circuit, no `state_adapter` overwrite)
- `GPU 2`: `prenorm_dedup` (true Pre-Norm + `loop-dedup` γ=0.5)
- `GPU 3`: `prenorm_adj_dedup` (true Pre-Norm + 2-line $O(d)$ `adj_dedup` γ=0.5)
- `GPU 4`: `prenorm_row_orth` (true Pre-Norm + 4-line checkpoint-safe `row_orth_muon` γ=0.5)
- `GPU 5`: `prenorm_step_mod` (true Pre-Norm + continuous sinusoidal depth modulation)
- `GPU 6`: `prenorm_momentum` (true Pre-Norm + 2-line 1-scalar Heavy-Ball depth momentum)
- `GPU 7`: `prenorm_signed_vel` (true Pre-Norm + 2-line per-channel signed 2nd-order depth recurrence, unifying momentum & dedup!)

**Pre-registered bar for `Loop-Muon` v1**: standalone recovers `>= 30%` of same-run `loop-dedup` Δ, OR combined adds `>= +25%` on top of `loop-dedup`.

---

## RSI Hypothesis Queue (Beyond Current Idea)

Driven by 1.32B step-500 real diagnostics (`diagnose_loop_collapse.py`), then rapidly screened on `huginn_r8` (~480M, 18 min/val, 1.1h/1B tok, 100% GPU util) before scaling winners to 1.32B & past-Chinchilla:

1. **H1 (Simplification test — 1-step vs subspace dedup)**: Does a 2-line adjacent-loop orthogonalization (`u_t - γ proj(u_t, u_{t-1})`, O(d), zero matrix solve, no window) match full subspace `loop-dedup`, or does multi-step `span(u_{<t})` genuinely win?
2. **H2 (Negative vs Positive cross-loop mixing)**: Contrast `loop-dedup` (negative projection pushing *out* of collapsed span) vs `Loop-AttnRes` (positive softmax mixing over past loop states) vs combined — testing whether escaping fixed-point collapse requires negative mixing.
3. **H3 (Multi-scale / interior dedup vs macro-boundary dedup)**: Guided by 31-layer `Ahat_perp` diagnostics — test whether deduplicating at sub-block boundaries (e.g., mid-core / per-block) beats macro-only dedup, and whether it fixes the `hbptt` train/eval boundary mismatch.
4. **H4 (Checkpoint-safe / Row-orthogonal / Loop-equalized Optimizer v2)**: Guided by per-loop gradient norm & cosine diagnostics — if `||G_7|| / ||G_4||` is skewed or `Loop-Muon` v1 needs a cleaner checkpoint-safe form, test 4-line streaming row-orthogonalized / norm-equalized loop gradient accumulation.
5. **H5 (Long-training Chinchilla+ & CORE & AdamW generality)**: Run the RSI champion vs baseline past **20+ tok/param** on `huginn_r8` (~9h overnight) with full 22-task CORE evals + stateless multi-depth `r ∈ {1,2,4,8,12,16}` + AdamW generality check, then confirm at **1.32B** (`checkpoint_00000500` resume) and **`r=16`**.
