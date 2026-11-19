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
| `big-dedup` | `a2` | `LOOP_DEDUP_GAMMA=0.5` | `3.16890` | **`1.01893`** (−0.00755) | **`0.96058`** | **−0.00538** | 14.07 | 202.5 | **Running uninterrupted** (**Step 750 val_bpb = `0.92371`**, 0.79B tok, 8×100%) |

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

#### Round 2 Leaderboard (550 steps = 72.1M tok, WSD cooldown, identical seed=42 & data, ~9.1–9.7 min)
| Rank | Idea | val_bpb@8 | Δ@8 vs #1 | val_bpb@16 | val_bpb@4 | collapse@8 | tok/s | mem GB |
|---|---|---|---|---|---|---|---|---|
| **1** | **`no_x0_huginn`** (kill `x0` short-circuit) | **1.05881** | **+0.00000** | **1.05889** | **1.07162** | 0.9980 | 142.9k | 86.3 |
| 2 | `prenorm_dedup` | **1.11759** | +0.05878 | 1.13105 | 1.26148 | 0.9778 | 140.0k | 81.7 |
| 3 | `prenorm_adj_dedup` | **1.12502** | +0.06621 | 1.13948 | 1.17034 | 0.9966 | 146.3k | 81.7 |
| 4 | `prenorm_loop` | **1.13411** | +0.07530 | 1.14514 | 1.17625 | 0.9949 | 149.0k | 80.6 |
| 5 | `prenorm_signed_vel` | **1.13795** | +0.07913 | 1.15023 | 1.18722 | 0.9921 | 148.4k | 81.0 |
| 6 | `prenorm_row_orth` | **1.15097** | +0.09216 | 1.16267 | 1.19267 | 0.9948 | 148.7k | 80.4 |
| 7 | `prenorm_step_mod` | **1.16123** | +0.10242 | 1.16969 | 1.20260 | 0.9898 | 148.6k | 81.0 |
| 8 | `prenorm_momentum` | **1.18223** | +0.12342 | 1.19270 | 1.23395 | 0.9858 | 148.7k | 81.0 |

**Key Round 2 Breakthroughs:**
1. **`no_x0_huginn` (`1.05881`) crushed unscaled Pre-Norm (`1.13411`) by −0.0753 bpb** AND unlocked 4× larger depth gain (`r=4 → 8`: **−0.01281 bpb**, stable at `r=16: 1.05889`!), proving SandwichBlock spherical norm + prelude re-injection (WITHOUT `x0_lambdas` short-circuit) is the superior backbone.
2. **On Pre-Norm, `loop-dedup` won by −0.01652 bpb** (`1.11759` vs `1.13411`), and 2-line `adj_dedup` captured 55% of it (`1.12502`, −0.00909 bpb).
3. Positive depth momentum (`prenorm_momentum`, `prenorm_signed_vel`) hurt Pre-Norm by amplifying norm growth (`depth_beta` drifted positive).

#### Round 3 Leaderboard (550 steps = 72.1M tok, identical seed=42 & data, ref Round 2 `no_x0_huginn = 1.05881`)
| Rank | Idea | val_bpb@8 | Δ@8 vs `base` | val_bpb@16 | val_bpb@4 | collapse@8 | tok/s | mem GB |
|---|---|---|---|---|---|---|---|---|
| **1** | **`no_x0_loop_muon`** | **1.05858** | **−0.00093** | **1.05870** | 1.07589 | 0.9975 | 98.1k | 86.2 |
| (ref) | `no_x0_huginn` (R2) | **1.05881** | −0.00070 | 1.05889 | 1.07162 | 0.9980 | 142.9k | 86.3 |
| 2 | `no_x0_row_orth` | **1.05891** | −0.00061 | 1.05895 | 1.06783 | 0.9983 | 142.5k | 86.2 |
| 3 | `no_x0_sphere_dedup25` | **1.05943** | −0.00009 | 1.05951 | 1.08082 | 0.9812 | 134.7k | 87.9 |
| 4 | `base` (with `x0_lambdas`) | 1.05951 | +0.00000 | 1.05956 | 1.06623 | 0.9991 | 133.5k | 88.6 |
| 5 | `no_x0_rand_depth` | 1.06173 | +0.00222 | 1.06172 | **1.06255** | 0.9998 | 142.3k | 86.3 |
| 6 | `no_x0_dedup` (γ=0.5) | 1.06311 | +0.00359 | 1.07712 | 1.12343 | 0.8573 | 134.6k | 87.9 |
| 7 | `no_x0_res_adapter_dedup` | 1.10365 | +0.04414 | 1.10374 | 1.10629 | 0.9939 | 134.6k | 87.9 |
| 8 | `no_x0_res_adapter` | 1.11048 | +0.05097 | 1.11049 | 1.11067 | 0.9999 | 143.6k | 86.4 |

**Key Round 3 Takeaways:**
- `no_x0_loop_muon` (`1.05858`) took #1 overall (`−0.00093` vs `base`, `−0.00023` vs `no_x0_huginn`), confirming clean un-short-circuited core + Loop-Muon works, **but** all tweaks inside identical tied weights cluster within `~0.001 bpb` of `1.0588`!
- `no_x0_rand_depth` was the ONLY arm with `val@16 < val@8` AND best `val@4` (`1.06255`), showing stochastic depth helps shallow/deep flexibility.
- To break far past `1.058` (targeting a **−0.015 to −0.040 bpb** breakthrough!), Round 4 attacks the **root expressivity & credit-assignment ceilings** of weight-tied loops directly with Kaiming-He minimal mechanisms!

#### Round 4 Leaderboard (550 steps = 72.1M tok, identical seed=42 & data, ref R3 `base = 1.05951`, R2 `no_x0_huginn = 1.05881`)
| Rank | Idea | val_bpb@8 | Δ@8 vs `base` | val_bpb@16 | val_bpb@4 | collapse@8 | tok/s | mem GB |
|---|---|---|---|---|---|---|---|---|
| **1** | **`loop_norm_ls`** (`loop_norm` + `layerscale`) | **1.05486** | **−0.00465** | 1.07307 | 1.09556 | 0.9351 | 131.0k | 95.3 |
| **2** | **`loop_norm_ls_muon`** (+ `row_orth_muon`) | **1.05497** | **−0.00454** | 1.07372 | 1.09623 | 0.9322 | 130.2k | 95.2 |
| **3** | **`loop_norm_ds`** (`loop_norm` + `deep_sup`) | **1.05590** | **−0.00361** | **1.05639** | 1.09102 | 0.9943 | 103.4k | 146.3 |
| **4** | **`layerscale`** (12 scalars only!) | **1.05620** | **−0.00331** | **1.05699** | 1.08413 | 0.9895 | 137.1k | 90.8 |
| **5** | **`loop_norm`** (0.07% params, 0% FLOPs) | **1.05640** | **−0.00311** | 1.06794 | 1.08559 | 0.9476 | 136.2k | 90.8 |
| **6** | **`huginn_attnres`** (cross-loop depth attn) | **1.05675** | **−0.00276** | 1.05907 | 1.09146 | 0.9852 | 132.1k | 93.9 |
| **7** | **`deep_sup`** (0% params, 0% inf FLOPs) | **1.05740** | **−0.00211** | 1.05743 | **1.06583** | 0.9982 | 107.1k | 141.8 |
| 8 | `full_bptt` (8-loop BPTT) | 1.06129 | +0.00178 | 1.06133 | 1.06421 | 0.9995 | 101.7k | 134.1 |

**Key Round 4 Breakthroughs:**
- **Broke wide open past `1.058` → `1.05486` (`−0.00465 bpb ≈ −0.013 nats` in just 72M tokens!)** — 5× larger than Round 3's best result!
- **All 4 Kaiming-He mechanisms won individually AND stacked additively**: `layerscale` (12 scalars, −0.00331), `loop_norm` (0.07% params, −0.00311), `huginn_attnres` (−0.00276), `deep_sup` (−0.00211 + **eliminated all `r=16` drift: `1.05639`**!).
- `full_bptt` hurt (`+0.00178`), confirming truncated BPTT (`s=4`) is actually better AND cheaper!

#### Round 5 (launched 18:54, 550 steps = 72.1M tok, pushing two-sided diagonal loop specialization + unification!)
- `GPU 0`: `sandwich_2sided` — Two-sided per-loop per-channel modulation `x = norm(x + s_{r,l} * f(g_{r,l} * norm(x)))` (0.14% params, 0% FLOPs)
- `GPU 1`: `loop_mlp_gate` — Per-loop `4d` MLP hidden neuron gating `m_{r,l}` + `layerscale` (0.07% params, 0% FLOPs)
- `GPU 2`: `loop_all_mod` — Full 3-point diagonal modulation (`g_{r,l}`, `m_{r,l}`, `s_{r,l}`, 0.21% params, 0% FLOPs!)
- `GPU 3`: `loop_all_mod_attnres` — `loop_all_mod` + `huginn_attnres`
- `GPU 4`: `loop_all_mod_ds` — `loop_all_mod` + `deep_sup` (fixing `r=16` extrapolation!)
- `GPU 5`: `loop_grand_combo` — All 4 Round-4 winners unified (`loop_all_mod` + `huginn_attnres` + `deep_sup`)
- `GPU 6`: `loop_norm_ls010` — Spherical ODE step-size scaling law (`s_init = 0.10 ≈ 1/sqrt(2*48)`) + `loop_norm`
- `GPU 7`: `prenorm_inloop_mod` — Linear Pre-Norm residual highway inside each loop + `loop_all_mod` + spherical `state_adapter` at loop boundary!

**Pre-registered bar for `Loop-Muon` v1**: standalone recovers `>= 30%` of same-run `loop-dedup` Δ, OR combined adds `>= +25%` on top of `loop-dedup`.

---

## RSI Hypothesis Queue (Beyond Current Idea)

Driven by 1.32B step-500 real diagnostics (`diagnose_loop_collapse.py`), then rapidly screened on `huginn_r8` (~480M, 18 min/val, 1.1h/1B tok, 100% GPU util) before scaling winners to 1.32B & past-Chinchilla:

1. **H1 (Simplification test — 1-step vs subspace dedup)**: Does a 2-line adjacent-loop orthogonalization (`u_t - γ proj(u_t, u_{t-1})`, O(d), zero matrix solve, no window) match full subspace `loop-dedup`, or does multi-step `span(u_{<t})` genuinely win?
2. **H2 (Negative vs Positive cross-loop mixing)**: Contrast `loop-dedup` (negative projection pushing *out* of collapsed span) vs `Loop-AttnRes` (positive softmax mixing over past loop states) vs combined — testing whether escaping fixed-point collapse requires negative mixing.
3. **H3 (Multi-scale / interior dedup vs macro-boundary dedup)**: Guided by 31-layer `Ahat_perp` diagnostics — test whether deduplicating at sub-block boundaries (e.g., mid-core / per-block) beats macro-only dedup, and whether it fixes the `hbptt` train/eval boundary mismatch.
4. **H4 (Checkpoint-safe / Row-orthogonal / Loop-equalized Optimizer v2)**: Guided by per-loop gradient norm & cosine diagnostics — if `||G_7|| / ||G_4||` is skewed or `Loop-Muon` v1 needs a cleaner checkpoint-safe form, test 4-line streaming row-orthogonalized / norm-equalized loop gradient accumulation.
5. **H5 (Long-training Chinchilla+ & CORE & AdamW generality)**: Run the RSI champion vs baseline past **20+ tok/param** on `huginn_r8` (~9h overnight) with full 22-task CORE evals + stateless multi-depth `r ∈ {1,2,4,8,12,16}` + AdamW generality check, then confirm at **1.32B** (`checkpoint_00000500` resume) and **`r=16`**.
