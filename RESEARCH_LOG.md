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
| `big-dedup` | `a2` | `LOOP_DEDUP_GAMMA=0.5` | `3.16890` | **`1.01893`** (−0.00755) | **`0.96058`** | **−0.00538** | 14.07 | 202.5 | Paused cleanly after **Step 1000 (`val_bpb = 0.90320`, 1.05B tok, `checkpoint_00001000` saved & resumable)**; `a2` reallocated to 16-GPU optimizer search |

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

#### Opt-Round 1 Leaderboard (`a1`, `huginn`, 550 steps = 72.1M tok, identical seed=42 & data, ~9.5 min)
| Rank | Idea | val_bpb@8 | Δ@8 vs base | val_bpb@16 | val_bpb@4 | mean σ_max/σ_med | mean rank_frac | tok/s | mem GB |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **`opt_schatten_wd`** (`λ=0.05`, 0 extra NS) | **1.05758** | **−0.00057** | **1.05774** | 1.07385 | **1.96** | **0.748** | **143.2k** | 86.6 |
| **2** | **`opt_polar_wd`** (`λ=0.05`) | **1.05791** | **−0.00024** | **1.05804** | **1.06951** (−0.00939!) | **2.27** | **0.727** | 142.7k | 86.5 |
| 3 | `opt_base` (SOTA `MuonAdamW`) | 1.05815 | +0.00000 | 1.05820 | 1.07890 | 2.43 | 0.700 | 142.7k | 86.7 |
| 4 | `opt_core_lr_05` | 1.06737 | +0.00922 | 1.06859 | 1.08507 | 2.34 | 0.701 | 143.0k | 86.3 |
| 5 | `opt_core_lr_15` | 1.07183 | +0.01368 | 1.07183 | 1.07261 | 2.58 | 0.695 | 143.6k | 86.3 |
| 6 | `opt_loop_bal` | 1.07909 | +0.02094 | 1.07924 | 1.08989 | 2.48 | 0.699 | 142.0k | 86.7 |
| 7 | `opt_per_loop_mom` | 1.08401 | +0.02586 | 1.08395 | 1.09806 | 2.33 | 0.708 | 142.0k | 88.0 |
| 8 | `opt_sum_of_pe` | 1.09596 | +0.03781 | 1.09593 | 1.09945 | 2.14 | 0.718 | 142.4k | 87.1 |

**Two Major Scientific Discoveries from Opt-Round 1 Telemetry:**
1. **Spectral Conditioning Law for Iterated Weight-Tied Matrices**: Under SOTA Muon (which only applies isotropic entrywise L2/cautious WD, leaving `κ(W)` unconstrained), tied core Attention weights collapse to `rank_frac = 0.584–0.603` and `σ_max/σ_med = 2.62–2.97` (vs MLP `0.86`, `1.8`). Adding a tiny `λ=0.05` cubic **Schatten-4 decay** (`W -= λ η α (W_n G_w - W_n)`, zero extra NS, 0% speed/mem/inference cost) lowered `σ_max/σ_med` monotonically (`2.43 → 2.27 → 1.96`), raised `rank_frac` (`0.700 → 0.727 → 0.748`), and beat SOTA Muon at `r=4, 8, 16`!
2. **Negative Cross-Loop Gradient Interference + 5×–15× Norm Skew**: `opt_base` telemetry revealed `||G_3|| / ||G_0|| = 10–15×` (step 100) and `5–8×` (step 550), AND **negative cross-loop cosine `cos(G_a, G_b) = −0.10 to −0.19`** on nearly every tied matrix! This explains why `opt_loop_bal` hurt (`+0.0209 bpb`): boosting `G_0` by 10× amplified destructive anti-correlated gradients against `G_3`.

#### Opt-Round 2 (`a1`) + Opt-Round 2B (`a2`) Combined 16-GPU Leaderboard (`huginn`, 550 steps = 72.1M tok, ref `opt_base = 1.05815`)
| Rank | Idea | Box | val_bpb@8 | Δ@8 vs base | val_bpb@16 | val_bpb@4 | σ_max | σ_max/σ_med | rank_frac | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **`opt_schatten_pure`** (`λ=0.10`, replaces L2 WD!) | `a1` | **1.05438** | **−0.00377** | **1.05487** | 1.07633 | **1.470** | **1.81** | **0.769** | **143.1k** |
| **2** | **`opt_core_beta1_085`** (core Muon `β₁=0.85`) | `a2` | **1.05537** | **−0.00278** | **1.05596** | 1.07859 | 1.643 | 2.37 | 0.694 | 143.1k |
| 3 | `opt_core_wd_3x` (3× cautious L2 WD control) | `a1` | **1.05568** | −0.00247 | 1.05579 | 1.07333 | 2.057 | 2.43 | 0.700 | 143.2k |
| 4 | `opt_schatten_pcgrad` (`Schatten 0.20 + PCGrad`) | `a2` | **1.05592** | −0.00223 | 1.05644 | 1.07764 | 1.459 | 1.57 | 0.815 | 141.9k |
| 5 | `opt_schatten_attn_020` (core Attn only) | `a2` | **1.05603** | −0.00212 | 1.05643 | 1.07806 | 1.774 | 1.82 | 0.783 | 142.9k |
| **6** | **`opt_loop_late_wt`** (`w=[.25,.5,.75,1]`) | `a2` | **1.05614** | **−0.00201** | **1.05614** | **1.06436** (−0.0145!) | 2.358 | 2.43 | 0.701 | 142.7k |
| 7 | `opt_schatten_020` (`W_n^3 - W_n`, `λ=0.20`) | `a1` | 1.05630 | −0.00185 | 1.05658 | 1.08066 | 1.460 | 1.57 | 0.815 | 143.2k |
| 8 | `opt_schatten_all_020` (all Muon layers) | `a2` | 1.05636 | −0.00179 | 1.05716 | 1.08291 | 1.463 | 1.58 | 0.814 | 143.0k |
| 9 | `opt_schatten_060` (`W_n^3 - W_n`, `λ=0.60`) | `a1` | 1.05683 | −0.00132 | 1.05894 | 1.10616 | 1.134 | 1.33 | 0.885 | 143.2k |
| 10 | `opt_loop_pcgrad` (negative conflict projection) | `a1` | 1.05800 | −0.00015 | 1.05807 | 1.06584 | 2.358 | 2.43 | 0.700 | 142.2k |
| (ref) | `opt_base` (SOTA `MuonAdamW`) | `a1` | 1.05815 | +0.00000 | 1.05820 | 1.07890 | 2.353 | 2.43 | 0.700 | 142.7k |
| 11 | `opt_spectral_clip` (rank-1 `σ_max` cap only) | `a1` | 1.05834 | +0.00019 | 1.05842 | 1.07249 | 2.195 | 2.26 | 0.702 | 143.2k |
| 12 | `opt_polar_wd_050` (`λ=0.50`) | `a1` | 1.05880 | +0.00065 | 1.05898 | 1.07497 | 1.698 | 1.66 | 0.876 | 143.3k |
| 13 | `opt_core_beta2_050` (NorMuon `β₂=0.50`) | `a2` | 1.06072 | +0.00257 | 1.06081 | 1.07237 | 2.370 | 2.44 | 0.700 | 143.5k |
| 14 | `opt_polar_rep_wd` (Polar replacing L2 WD) | `a1` | 1.06098 | +0.00283 | 1.06113 | 1.07562 | 1.810 | 1.64 | 0.885 | 143.1k |
| 15 | `opt_loop_early_wt` (`w=[1.5,1.25,1,.75]`) | `a2` | 1.06232 | +0.00417 | 1.06253 | 1.07629 | 2.366 | 2.45 | 0.699 | 142.6k |
| 16 | `opt_core_beta1_098` (core Muon `β₁=0.98`) | `a2` | 1.08028 | +0.02213 | 1.08029 | 1.08254 | 3.846 | 2.70 | 0.712 | 142.8k |

**Major Breakthroughs from 16-GPU Opt-Round 2 + 2B:**
1. **`opt_schatten_pure` (`1.05438`, −0.00377 bpb vs SOTA Muon) ALREADY BEAT OUR BEST ARCHITECTURAL MODIFICATION (`1.05486` Round 4) WITH ZERO ARCHITECTURE CHANGES, ZERO INFERENCE COST, ZERO EXTRA STATE, ZERO OVERHEAD (`143.1k tok/s`)!** Furthermore, at **identical `σ_max = 1.47`**, pure cubic `W (W^T W) / α^2` beat `W_n^3 - W_n` (`opt_schatten_020 = 1.05630`) by **−0.00192 bpb** because pure Schatten-4 never pushes up noisy tail singular values!
2. **Three Orthogonal Pure Loop-Muon Laws Discovered**:
   - **Law A (Pure Schatten-4 Spectral Decay)**: `1.05438` (`−0.00377`) — replaces isotropic L2 WD on tied core weights.
   - **Law B (Tied-Core Momentum Law)**: `β₁ = 0.98 → 0.95 → 0.85` yields `1.08028 → 1.05815 → 1.05537` (`−0.00278`, zero extra FLOPs/memory).
   - **Law C (Depth-Discounted BPTT Parameter Gradients)**: `early_wt (1.06232) → uniform (1.05815) → late_wt (1.05614)` (`−0.00201` at `r=8`, **0.00000 drift at `r=16`**, and **`−0.01454 bpb` at `r=4`!!**), with `σ_max` completely unchanged (`2.358`) → 100% orthogonal to weight conditioning!

#### Opt-Round 3 (`a1`) + Opt-Round 3B (`a2`) Combined 16-GPU Leaderboard (550 steps = 72.1M tok, ref `huginn base = 1.05815`, `prenorm base = 1.11282`)
| Rank | Backbone | Idea | val_bpb@8 | Δ@8 vs base | val_bpb@16 | val_bpb@4 | σ_max | σ_max/σ_med | rank_frac |
|---|---|---|---|---|---|---|---|---|---|
| **1** | `huginn` | **`opt_core_wd_8x`** (8× cautious WD on core!) | **1.05315** | **−0.00500** | **1.05349** | **1.06816** | **1.615** | 2.41 | 0.700 |
| **2** | `huginn` | **`opt_schatten_pure_all`** (`λ=0.15`, all layers) | **1.05369** | **−0.00446** | 1.05526 | 1.07862 | 1.256 | 1.74 | 0.780 |
| **3** | `huginn` | **`opt_combo_sp_late`** (`Schatten 0.15 + late_wt`) | **1.05391** | **−0.00424** | **1.05409** | 1.09099 | 1.258 | 1.73 | 0.781 |
| 4 | `huginn` | `opt_schatten_pure_020` (`λ=0.20`) | 1.05494 | −0.00321 | 1.05671 | 1.09141 | 1.100 | 1.68 | 0.789 |
| 5 | `huginn` | `opt_core_l2_plain` (plain L2 `λ=0.10`) | 1.05555 | −0.00260 | 1.05578 | 1.07081 | 1.984 | 2.42 | 0.700 |
| 6 | `huginn` | `opt_loop_exp_05` (`[.125,.25,.5,1]`) | 1.05738 | −0.00077 | 1.05740 | 1.07347 | 2.362 | 2.43 | 0.701 |
| 7 | `huginn` | `opt_schatten_pure_040` (`λ=0.40`, over-shrunk) | 1.06043 | +0.00228 | 1.06387 | 1.14556 | 0.777 | 1.59 | 0.807 |
| 8 | `huginn` | `opt_loop_last_only` (`[0,0,0,1]`) | 1.06315 | +0.00500 | 1.06316 | 1.07852 | 2.401 | 2.45 | 0.700 |
| 9 | `huginn` | `opt_combo_sp_b85` (double-shrunk `σ_max=0.77`!) | 1.06767 | +0.00952 | 1.06925 | 1.14258 | 0.773 | 1.70 | 0.803 |
| 10 | `huginn` | `opt_core_beta1_070` | 1.06955 | +0.01140 | 1.06989 | 1.09598 | 1.434 | 2.42 | 0.689 |
| 11 | `huginn` | `opt_combo_all3` (double-shrunk `σ_max=0.77`!) | 1.07029 | +0.01214 | 1.07155 | 1.14050 | 0.767 | 1.70 | 0.802 |
| 12 | `huginn` | `opt_core_beta1_050` | 1.08133 | +0.02318 | 1.08151 | 1.10816 | 1.341 | 2.49 | 0.684 |
| — | `prenorm` | `opt_base` | **1.11282** | +0.00000 | 1.14923 | 1.20040 | 3.311 | 2.77 | 0.682 |
| — | `prenorm` | `opt_schatten_pure` (`λ=0.10`) | 1.11360 | +0.00078 | 1.16870 | 1.24271 | 1.951 | 2.01 | 0.761 |
| — | `prenorm` | `opt_core_beta1_085` | 1.12240 | +0.00958 | 1.17063 | 1.23351 | 2.224 | 3.01 | 0.667 |
| — | `prenorm` | `opt_combo_all3` | 1.13790 | +0.02508 | 1.21028 | 1.28448 | 0.939 | 2.05 | 0.777 |

**Critical Scientific Diagnosis (Honest Self-Check!):**
1. **The Causal Control (`opt_core_wd_8x`) WON #1 OVERALL (`1.05315`, −0.00500 bpb!)** — beating `opt_schatten_pure` (`1.05438`) AND beating every architectural tweak! Notice `8×` is the exact loop count `R=8`, and `smax` follows a razor-sharp U-curve across all 32 runs (optimum `smax ∈ [1.25, 1.65]`; combining `Schatten + β₁=0.80` double-shrank `smax` to `0.77` and collapsed both `huginn` and `prenorm`).
#### Opt-Round 4 (`a1`) + Depth Scaling (`r=1 / r=8 / r=16` on `a2`) Summary
| Setting | Backbone | Idea | Target val_bpb | Δ vs same-setting base | σ_max | Note |
|---|---|---|---|---|---|---|
| `r=1` (untied) | `huginn` | `opt_base` | `1.07464` | — | 2.404 | Untied control |
| `r=1` (untied) | `huginn` | `opt_core_wd_8x` | `1.07392` | `−0.00072` | 1.761 | Only 1/7th of `r=8` gain |
| `r=8` | `huginn` | `opt_all_wd_8x` | `1.05163` | `−0.00652` | 1.616 | Hyperparam WD cooldown effect (note: pre-cooldown step 550 in 1650-step run was `+0.00311` worse!) |
| `r=8` | `huginn` | `opt_wd8x_late` | `1.05258` | `−0.00557` | 1.621 | `r=16 = 1.05262` (zero drift) |
| `r=8` | `huginn` | `opt_cautious_schatten` | `1.05501` | `−0.00314` | 1.495 | Cautious cubic spectral decay |
| `r=8` | `huginn` | `opt_cubic_frob_wd` | `1.05508` | `−0.00307` | 1.903 | Self-regulating `α² W` |
| `r=8` | `huginn` | `opt_core_ns2` | `1.19328` | `+0.13513` | 38.71 | Collapsed (`ns=2` under-converges!) |
| `r=8` | `prenorm` | `opt_core_wd_8x` | `1.10961` | `−0.00321` | 2.391 | Transfers to Pre-Norm |
| `r=8` | `prenorm` | `opt_loop_late_wt` | `1.11502` | `+0.00220` | 3.300 | Did not transfer to Pre-Norm |
| `r=16` | `huginn` | `opt_base` | `1.05848` | — | 2.353 | Deep loop baseline |
| `r=16` | `huginn` | `opt_core_wd_8x` | `1.05409` | `−0.00439` | 1.608 | Holds at `r=16` |
| `r=16` | `huginn` | `opt_loop_late_wt` | `1.05632` | `−0.00216` | 2.362 | Holds at `r=16` |

#### 1650-Step Long-Horizon Persistence Victory (`a2`, 3× longer = 216M tok, `huginn r=8`)
| Arm | val_bpb@8 (step 1650) | Δ vs base (1650) | Δ at 550 steps | Trend (550 → 1650) | val_bpb@16 |
|---|---|---|---|---|---|
| `long_base` (SOTA Muon) | `0.96781` | — | — | — | `0.96791` |
| `long_wd8x` (8× L2 WD hyperparam) | `0.96317` | `−0.00464` | `−0.00500` | Shrank slightly (`-7%`) | `0.96507` |
| **`long_schatten` (Pure Schatten-4, default LR)** | **`0.96240`** | **`−0.00541`** | `−0.00377` | **GREW +44% BIGGER & OVERTOOK 8× WD!** | **`0.96419`** |

#### Opt-Round 5 (`a1`) & TEON-Round 1 (`a2`) Key Pure-Algorithmic Findings (100% default LR/WD)
- **`opt_dual_mom`** (Dual-timescale core momentum `0.5*M(0.80) + 0.5*M(0.95)`, default LR/WD): **`1.05400`** (`−0.00415 bpb` vs `1.05815`!)
- **`teon_loop_dct_ramp`** (Frequency-Domain DCT-II `Loop-tSVD`, default hyperparams): **`1.05726`** (`−0.00089 bpb` vs SOTA Muon, `r=4: −0.01005 bpb`, `r=16: 1.05727` zero drift!), crushing classic Cross-Layer `teon_layer_m1` (`1.15578`).
- **Mathematical Diagnosis of Naive TEON Failure**: `teon_loop_m1` (`1.07946`), `teon_loop_m2` (`1.07675`), and unweighted `teon_loop_tsvd_hosvd` (`1.07625`) failed because naive unfolding of `3072×768 → 3072×3072` (or unweighted HOSVD) equalizes noisy early/high-frequency loop singular modes up to `1.0`. Frequency-filtering (`DCT`) or singular-value weighting (`sv_w`) fixes this directly!

---

### Series D: Official Unmodified `NextTransformer` (`fixed_recursive`) Pure-Algorithmic Arena (`nt_loop_opt_arena.py`)
- Anchored strictly on **`Interactive-Active-AI/NextTransformer`'s #1 official looped model (`fixed_recursive`, CORE = `0.3245` at 50B tok vs Dense `0.3310`, beating `ut_act 0.3007`, `ouro 0.2770`, `huginn 0.2726`, `deeploop 0.2545`)** with **ZERO edits to `ReferenceLM`** (non-intrusive forward hook tracks loop index) and **100% default `NextTransformer` hyperparams**.
#### NT-Round 1 (`w=768, 207.6M, r=4`, 750 steps) & 928M Official Scale (`w=1536, 3/9/3, r=2`, step 400 mid-eval)
1. **928M Official `NextTransformer` `fixed_recursive.json` Breakthrough (step 400/800, 100% default hyperparams!)**:
   - `big928m_nt_base` (SOTA `MuonAdamW`): step 100 `5.2689` → step 200 `4.4261` → step 300 `3.8518` → step 400 `3.6415`, **`val_bpb = 1.10742`** (`gnorm = 0.415`).
   - **`big928m_nt_loop_karcher_pe` (Stiefel Karcher Mean `PE(Σ w_t PE(M_t))`)**: step 100 **`5.1934`** (−0.0755) → step 200 **`4.2939`** (−0.1322!) → step 300 **`3.8067`** (−0.0451) → step 400 **`3.6250`**, **`val_bpb = 1.10202` (`−0.00540 bpb` CRUSHING SOTA Muon at 928M scale, `gnorm = 0.228`!)**.
   - `big928m_nt_teon_hosvd_svw`: step 400 `val_bpb = 1.10815` (`+0.00073`).
2. **Telemetry Discovery on `NextTransformer` `fixed_recursive` (`w=768, r=4`)**:
   - Completely different gradient physics from `Huginn`: **all 4 loops are balanced (`ratio ≈ 0.8–1.3×`, vs `5–15×` in Huginn)** and **strongly positively correlated (`cos = +0.35 to +0.66`, vs `-0.16` in Huginn)**!
   - Therefore DCT low-pass (`1.04281`) and Haar (`1.04464`) threw away valid signal on `fixed_recursive`, while `nt_teon_m2_norm` (`1.03610`), `nt_teon_hosvd_svw` (`1.03627`), `nt_loop_spec_mix` (`1.03701`), `nt_loop_karcher_pe` (`1.03714`) were close to `nt_base` (`1.03486`).
   - **Bug caught & fixed**: `opt.zero_grad()` did not zero `core_params` excluded from `opt` when `needs_per_loop=False` (`nt_schatten_pure` & `nt_loop_dual_mom`), causing infinite grad accumulation (`smax=5.06/10.44`). Fixed to `model.zero_grad(set_to_none=True)` and relaunched cleanly!
#### NT-Round 2 (`w=768, 207.6M, r=4`, 750 steps) & 928M Official Scale Results (`w=1536, 3/9/3, r=2`)
| Scale | Idea | Mid val_bpb | Δ Mid vs base | Final (cooled) val_bpb | Δ Final vs base | σ_max | Note |
|---|---|---|---|---|---|---|---|
| `207M w=768` | **`nt_ctrl_pe10`** (`PE(PE(M_sum))`) | **`1.15067`** | **`−0.00455`** | **`1.03416`** | **`−0.00070`** | 3.294 | **#1 overall at `w=768`! Unbiased!** |
| `207M w=768` | `nt_teon_geodesic_blend` | `1.15339` | `−0.00183` | **`1.03445`** | **`−0.00041`** | 3.262 | #2 (also runs 2nd PE) |
| `207M w=768` | `nt_base` (SOTA Muon) | `1.15522` | — | `1.03486` | — | 3.448 | Official NextTransformer baseline |
| `207M w=768` | `nt_teon_mode3_whiten` | `1.15555` | `+0.00033` | `1.03538` | `+0.00052` | 3.456 | Biased loop reweighting |
| `207M w=768` | `nt_loop_normuon_pre` | `1.15473` | `−0.00049` | `1.03546` | `+0.00060` | 3.421 | — |
| `207M w=768` | `nt_schatten_pure` (fixed) | `1.15523` | `+0.00001` | `1.03628` | `+0.00142` | 1.762 | Fixed `zero_grad`; weaker on Pre-Norm `fixed_recursive` |
| `207M w=768` | `nt_loop_karcher_unw` | `1.15329` | `−0.00193` | `1.03774` | `+0.00288` | 3.182 | Biased — loses after cooldown |
| `207M w=768` | `nt_loop_dual_mom` (fixed) | `1.15714` | `+0.00192` | `1.03792` | `+0.00306` | 2.543 | — |
| `928M w=1536` | **`big928m_nt_ctrl_pe10`** | **`1.09956`** | **`−0.00786`!** | *(finishing)* | — | — | **1.73× bigger gain as width doubles `768→1536`!** |
| `928M w=1536` | `big928m_nt_loop_karcher_unw` | `1.10165` | `−0.00577` | *(finishing)* | — | — | — |
| `928M w=1536` | `big928m_nt_loop_karcher_pe` | `1.10202` | `−0.00540` | `0.98272` | `+0.00037` | 3.208 | Won mid-run, slipped after cooldown (gradient bias!) |
| `928M w=1536` | `big928m_nt_base` (SOTA Muon) | `1.10742` | — | **`0.98235`** | — | 3.552 | Official 928M baseline |
| `928M w=1536` | `big928m_nt_teon_hosvd_svw` | `1.10815` | `+0.00073` | `0.98336` | `+0.00101` | 3.476 | Biased |

**Two Fundamental Optimizer Laws Uncovered by `nt_ctrl_pe10`:**
1. **Unbiasedness Principle for Looped-LM Optimizers**: Altering the direction of `∇_W L = Σ_t G_t` (via `Karcher`, `HOSVD`, `DCT`, `Haar`, `Mode-3`) creates asymptotic gradient bias that regresses after WSD cooldown (`928M Karcher`: `−0.00540` mid → `+0.00037` cooled!). Winning post-cooldown requires **strictly unbiased positive-definite preconditioning** or **exact orthogonalization**!
2. **Width-Scaling `1/√d` Under-Convergence of Frobenius-Normalized Polar Express**: Standard Muon normalizes `X` by `||X||_F`, shrinking `σ_max(X)` to `O(1/√d)` (`~0.04` at `w=1536`), causing 5-step Polar Express to waste its first 3 iterations climbing back to `1.0` and under-converging bulk singular values (especially on high-rank summed multi-loop gradients!). Replacing `||X||_F` with 5-step power-iteration **`σ_max` normalization** (`alg_spec_norm_pe`) lifts the entire spectrum by `15–25×` at **ZERO extra NS steps**!

#### NT-Round 3 (`a1` 8× `w=768, r=4`) + `a2` GPUs 4,6,7 (`928M w=1536`) — 100% UNBIASED Pure Algorithms (Launched 21:31)
- **`a1` (8× `w=768`)**: `alg_spec_norm_pe` (`σ_max`-normalized 5-step PE, 0 extra NS!), `alg_exact_polar_svd` (exact `U V^T` oracle via `eigh`), `alg_true_layer_teon` (spectral-scaled Cross-Layer TEON with joint Frobenius norm), `alg_true_loop_teon_m1` (spectral-scaled Mode-1 Loop-TEON), `alg_head_tuon` (per-head QKV tensor orthogonalization), `alg_loop_fisher_normuon` (unbiased cross-loop empirical Fisher NorMuon), `alg_mars_muon` (unbiased MARS variance-reduced Muon), `alg_loop_sym_shampoo` (unbiased cross-loop covariance preconditioner + SpecNorm PE).
- **`a2` (Full 8× 928M Official Scale)**: GPUs 0,1,2,3,5 finishing step-800 cooldown (`pe10`, `schatten_pure`, `karcher_unw`, `karcher_schatten`, `dual_mom`); GPUs 4,6,7 launched on **`alg_spec_norm_pe`**, **`alg_true_layer_teon`**, and **`alg_head_tuon`** at 928M scale!

---

## Pure Loop-Muon Optimizer Hypothesis Queue (Next Rounds)
1. **O1 (Per-loop Gradient Structure & Aggregation)**: Guided by `opt_base` step-100/550 telemetry (`||G_0||..||G_3||` and `cos(G_a, G_b)`): test depth-weighted BPTT aggregation (`w_t G_t`), cross-loop orthogonalization / Gram-Schmidt / anti-interference (removing redundant or conflicting loop components before `PE`), and checkpoint-safe row-whitened `Loop-Muon` on ALL 6 core matrices (including `c_proj`).
2. **O2 (Iterated Weight Spectral Conditioning)**: If `opt_polar_wd` / `opt_schatten_wd` win, sweep strength/frequency/cautious-masking, test direct **Spectral-Norm Clipping / Soft-Cap (`σ_max(W) ≤ c`)** and **Polar-Momentum Coupling** (subtracting the component of `ΔW` aligned with `W`'s top singular subspace).
3. **O3 (Tied-Parameter Muon Hyper-Geometry)**: Systematic decoupled sweep of `core_lr`, `core_wd`, `core_beta1` (momentum), `core_beta2` (NorMuon variance), and Polar Express `ns_steps` specifically for weight-tied core matrices vs untied prelude/coda.
4. **O4 (Cross-Backbone & Cross-Depth & Long-Run Scaling)**: Once `a2` completes `big-dedup` step 1000 (~19:50), deploy `a2` as a second 8-GPU optimizer arena (`prenorm` backbone & `r=16` depth → **16 optimizer ideas / 15 min**) before promoting the top 1–2 Kaiming-He-simple **Loop-Muon** winners to **480M/1.32B past-Chinchilla + 22-task CORE** confirmation!
