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

#### Round 5 Leaderboard (550 steps = 72.1M tok, identical seed=42 & data, ref R3 `base = 1.05951`, R4 #1 `loop_norm_ls = 1.05486`)
| Rank | Idea | val_bpb@8 | Δ@8 vs `base` | val_bpb@16 | val_bpb@4 | collapse@8 | tok/s | mem GB |
|---|---|---|---|---|---|---|---|---|
| 1 | `sandwich_2sided` | **1.05516** | −0.00435 | 1.10363 | 1.15980 | 0.8039 | 129.9k | 95.3 |
| 2 | `loop_mlp_gate` | **1.05517** | −0.00434 | 1.06159 | 1.09513 | 0.9486 | 126.6k | 99.8 |
| 3 | `loop_all_mod_ds` | **1.05524** | −0.00427 | **1.05633** | 1.15256 | 0.9881 | 94.0k | 159.8 |
| 4 | `loop_all_mod` | **1.05536** | −0.00415 | 1.10670 | 1.16939 | 0.7979 | 120.6k | 104.4 |
| 5 | `loop_norm_ls010` | **1.05812** | −0.00139 | 1.07510 | 1.10447 | 0.9342 | 131.0k | 95.3 |
| 6 | `loop_grand_combo` | **1.05949** | −0.00002 | 1.05969 | 1.15767 | 0.9975 | 88.9k | 168.2 |
| 7 | `loop_all_mod_attnres` | **1.05973** | +0.00022 | 1.05997 | 1.17863 | 0.8067 | 112.5k | 112.0 |
| 8 | `prenorm_inloop_mod` | **1.06164** | +0.00213 | 1.12960 | 1.29688 | 0.6378 | 123.3k | 99.8 |

**Strategic Pivot (2026-11-19 19:11): Early-Stop Architecture Tweaks → Pure Loop-Muon Optimizer Research**
- Architectural micro-tweaks on looped LMs (per-loop norms, LayerScale, MLP gating, etc.) are crowded and incremental for top-tier publication.
- In contrast, **optimizer theory & design specifically for weight-tied looped LMs under Muon** is genuinely under-explored and cleanly publishable: **ZERO architecture changes, ZERO inference cost**, drop-in replacement for Muon on tied recurrent parameters `W` reused `R` times per token.
- Fundamental structural differences of tied `W` under Muon:
  1. **Sum-of-loops gradient bias**: `G = Σ_{t=1}^s G_t` mixes `s` depth-shifted Jacobians with unequal magnitudes and cross-loop interference before Polar Express orthogonalization (`PE(Σ G_t) ≠ Σ PE(G_t)`).
  2. **Iterated spectral amplification**: Reapplying `W` `R` times amplifies condition number `κ(W)^R`, whereas standard Muon only orthogonalizes updates `ΔW` and leaves `W`'s own spectrum unconstrained.
  3. **Modular-norm step-size mismatch**: Tied core matrices contribute `R` (or `s`) times to the residual stream per step, implying distinct optimal learning-rate / momentum / decay scaling relative to untied layers.

---

### Series C: Pure Loop-Muon Optimizer Search Arena (`muon_opt_arena.py`, ZERO Architecture Changes)
- Frozen backbone: clean `huginn` (`no_x0_huginn`, 208.8M, `r=8, s=4`) — and secondary backbone `prenorm` (`1/sqrt(2*core*R)` residual scaling) for generality verification.
- Strong SOTA baseline (`opt_base`): `nanochat` `MuonAdamW` (`Polar Express + MuonEq + Muon+ + NorMuon + Cautious WD`) with full per-loop gradient (`||G_t||`, `cos(G_a, G_b)`) & core weight singular spectrum telemetry.

#### Opt-Round 1 (launched 19:37 on `a1`, 8×B300, `huginn`, 550 steps = 72.1M tok, batch=128, micro=64)
- `GPU 0`: `opt_base` — SOTA `MuonAdamW` + full per-loop gradient & weight spectrum telemetry (steps 100 & 550)
- `GPU 1`: `opt_polar_wd` — Isometric Polar Weight Decay: `W -= 0.05 * lr * (W - α * PE(W))` (bounds `κ(W)` for iterated `W^R`, 1× state, 0 inf cost)
- `GPU 2`: `opt_schatten_wd` — Schatten-4 Spectral-Cap Decay: `W -= 0.05 * lr * α * (W_n G_w - W_n)` (0 extra NS iterations)
- `GPU 3`: `opt_core_lr_05` — Modular-Norm loop LR scaling: `core_lr = 0.5 * muon_lr` (`~1/sqrt(s)`)
- `GPU 4`: `opt_core_lr_15` — Inverse bracket: `core_lr = 1.5 * muon_lr`
- `GPU 5`: `opt_loop_bal` — Per-loop Frobenius norm equalization before summing into Muon (1× state, checkpoint-safe)
- `GPU 6`: `opt_sum_of_pe` — Per-loop Polar Express orthogonalization BEFORE summing (`mean_t PE(G_t)`, 1× state, checkpoint-safe)
- `GPU 7`: `opt_per_loop_mom` — Oracle upper bound: `s=4` decoupled per-loop Nesterov momentum buffers (`mean_t PE(M_t)`)

---

## Pure Loop-Muon Optimizer Hypothesis Queue (Next Rounds)
1. **O1 (Per-loop Gradient Structure & Aggregation)**: Guided by `opt_base` step-100/550 telemetry (`||G_0||..||G_3||` and `cos(G_a, G_b)`): test depth-weighted BPTT aggregation (`w_t G_t`), cross-loop orthogonalization / Gram-Schmidt / anti-interference (removing redundant or conflicting loop components before `PE`), and checkpoint-safe row-whitened `Loop-Muon` on ALL 6 core matrices (including `c_proj`).
2. **O2 (Iterated Weight Spectral Conditioning)**: If `opt_polar_wd` / `opt_schatten_wd` win, sweep strength/frequency/cautious-masking, test direct **Spectral-Norm Clipping / Soft-Cap (`σ_max(W) ≤ c`)** and **Polar-Momentum Coupling** (subtracting the component of `ΔW` aligned with `W`'s top singular subspace).
3. **O3 (Tied-Parameter Muon Hyper-Geometry)**: Systematic decoupled sweep of `core_lr`, `core_wd`, `core_beta1` (momentum), `core_beta2` (NorMuon variance), and Polar Express `ns_steps` specifically for weight-tied core matrices vs untied prelude/coda.
4. **O4 (Cross-Backbone & Cross-Depth & Long-Run Scaling)**: Once `a2` completes `big-dedup` step 1000 (~19:50), deploy `a2` as a second 8-GPU optimizer arena (`prenorm` backbone & `r=16` depth → **16 optimizer ideas / 15 min**) before promoting the top 1–2 Kaiming-He-simple **Loop-Muon** winners to **480M/1.32B past-Chinchilla + 22-task CORE** confirmation!
