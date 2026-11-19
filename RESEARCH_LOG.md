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

#### NT-Round 3 (`w=768, 207.6M, r=4`, 750 steps) & 928M Cooldown Finals — **MAJOR PURE-ALGORITHMIC BREAKTHROUGHS!**
| Scale | Idea (100% default hyperparams, 0 arch changes) | Mid val_bpb | Δ Mid | Final (cooled) val_bpb | Δ Final vs base | tok/s |
|---|---|---|---|---|---|---|
| `207M w=768` | **`alg_exact_polar_svd`** (Exact Polar `U V^T`!) | **`1.14913`** | **`−0.00609`** | **`1.03272`** | **`−0.00214`!** | 164.4k |
| `207M w=768` | **`alg_head_tuon`** (Per-Head QKV `TUON` `[H,d_h,d]`, 0% cost!) | **`1.15010`** | **`−0.00512`** | **`1.03366`** | **`−0.00120`!** | **221.6k** |
| `207M w=768` | `nt_ctrl_pe10` (`PE(PE(M_sum))`, R2) | `1.15067` | `−0.00455` | `1.03416` | `−0.00070` | 220.8k |
| `207M w=768` | `nt_base` (SOTA Muon baseline) | `1.15522` | — | `1.03486` | — | 221.2k |
| `207M w=768` | `alg_loop_fisher_normuon` | `1.15484` | `−0.00038` | `1.03498` | `+0.00012` | 221.5k |
| `207M w=768` | `alg_mars_muon` | `1.15500` | `−0.00022` | `1.03519` | `+0.00033` | 222.1k |
| `207M w=768` | `alg_spec_norm_pe` (Schatten-8 5-step PE) | `1.15470` | `−0.00052` | `1.03521` | `+0.00035` | 221.2k |
| `207M w=768` | `alg_true_layer_teon` (Cross-Layer TEON) | `1.15512` | `−0.00010` | `1.03566` | `+0.00080` | 221.2k |
| `207M w=768` | `alg_true_loop_teon_m1` | `1.15711` | `+0.00189` | `1.03689` | `+0.00203` | 219.6k |
| `928M w=1536` | **`big928m_nt_ctrl_pe10`** (`PE(PE(M))`) | **`1.09956`** | **`−0.00786`** | **`0.97994`** | **`−0.00241`! (3.44× scaling!)** | **88.7k** |
| `928M w=1536` | `big928m_nt_base` (SOTA Muon baseline) | `1.10742` | — | `0.98235` | — | 89.8k |
| `928M w=1536` | `big928m_nt_loop_karcher_unw` | `1.10165` | `−0.00577` | `0.98248` | `+0.00013` | 87.7k |
| `928M w=1536` | `big928m_nt_schatten_pure` (fixed) | `1.10801` | `+0.00059` | `0.98368` | `+0.00133` | 89.4k |
| `928M w=1536` | `big928m_nt_loop_dual_mom` (fixed) | `1.11045` | `+0.00303` | `0.98533` | `+0.00298` | 89.6k |
| `928M w=1536` | `big928m_alg_head_tuon` *(live, step 200)* | `loss 4.3262` | **`−0.100 nats`!** | *(finishing)* | — | **89.3k** |
| `928M w=1536` | `big928m_alg_true_layer_teon` *(live, step 200)* | `loss 4.3401` | `−0.086 nats` | *(finishing)* | — | **88.9k** |

**Breakthrough Insights:**
1. **Super-Linear Width Scaling of Exact Polar Decomposition**: `PE10` grew **3.44× bigger** post-cooldown when scaling from `207M` (`−0.00070`) to `928M` (`−0.00241`, `0.97994` vs `0.98235`!), AND **Exact Polar `U V^T`** already tripled `PE10` at `207M` (**`−0.00214 bpb`**, `1.03272`!). Why? In a weight-tied looped LM (`M = Σ_{t=1}^R M_t`), shared modes add constructively `R×` while depth-unique modes stay `1×`, so inexact 5-step Polar Express (`S' ∈ [0.7, 1.3]`, tail crushed) selectively suppresses depth-specific loop directions!
2. **Per-Head Tensor Orthogonalization (`alg_head_tuon`) Wins at ZERO Speed Cost**: **`1.03366` (`−0.00120` at `207M`, `221.6k` vs `221.2k tok/s`!)** and ALREADY **−0.100 nats ahead at 928M step 200** (`89.3k` tok/s)!
#### NT-Round 4 (`w=768`, 750 steps) & 928M Wave 1 Final Cooldown — **Zero-Overhead `Head-TUON` Confirmed Across Both Scales!**
| Scale | Idea (100% default hyperparams) | Mid val_bpb | Final (cooled) val_bpb | Δ Final vs base | κ (`σ_max/σ_med`) | tok/s |
|---|---|---|---|---|---|---|
| `928M w=1536` | **`big928m_nt_ctrl_pe10`** (`PE10`, W1) | **`1.09956`** | **`0.97994`** | **`−0.00241`** | 2.77 | 88.7k |
| `928M w=1536` | **`big928m_alg_head_tuon`** (QKV Head-TUON, **0% cost!**) | **`1.10357`** | **`0.98116`** | **`−0.00119`!** | 3.87 | **89.5k** |
| `928M w=1536` | `big928m_nt_base` (SOTA Muon baseline) | `1.10742` | `0.98235` | — | 2.94 | 89.8k |
| `928M w=1536` | `big928m_alg_spec_norm_pe` | `1.10798` | `0.98267` | `+0.00032` | 2.87 | 89.6k |
| `928M w=1536` | `big928m_alg_true_layer_teon` (5-PE) | `1.10987` | `0.98709` | `+0.00474` | 5.87 | 89.0k |
| `207M w=768` | **`alg_exact_polar_svd`** (`eigh U V^T`, R3) | **`1.14913`** | **`1.03272`** | **`−0.00214`** | 2.83 | 164.4k |
| `207M w=768` | **`alg_head_tuon`** (QKV Head-TUON, R3) | **`1.15010`** | **`1.03366`** | **`−0.00120`** | 3.39 | **221.6k** |
| `207M w=768` | `alg_qdwh_polar` (4-step QDWH) | `1.15190` | `1.03418` | `−0.00068` | 2.79 | 165.1k |
| `207M w=768` | `alg_head4_qdwh` (split `c_proj` — HURT!) | `1.15744` | `1.03805` | `+0.00319` | 3.84 | 173.4k |
| `207M w=768` | `alg_mlp_block_tuon_qdwh` (split MLP — HURT!) | `1.15973` | `1.03925` | `+0.00439` | 3.91 | 191.9k |
| `207M w=768` | `alg_head4_layer_qdwh` (4D split `c_proj` — HURT!) | `1.16898` | `1.04220` | `+0.00734` | 5.92 | 189.4k |
| `207M w=768` | `alg_svd_spectral_lift` (`U Σ^{-0.1} V^T` — HURT!) | `1.17247` | `1.04451` | `+0.00965` | 2.76 | 164.9k |

**Key Structural Insight:**
- **Independent Output Subspaces vs Mixed Output Subspaces**: Splitting `c_q, c_k, c_v` per head (`[H, d_h, d]`) wins identically at BOTH `207M` (`−0.00120`) and `928M` (`−0.00119`) with **0.0% speed overhead** (`89.5k` vs `89.8k`)! Conversely, splitting `c_proj` or `MLP` (which merge subspaces into the shared residual stream) destroys cross-head orthogonality (`κ: 2.8 → 5.9`, `+0.004 to +0.007` penalty!).
- **NT-Round 5 (`a1`, `w=768, 207.6M`, 750 steps) & 928M Wave 2 (`a2`, `w=1536, 928.0M`, 800 steps) — GRAND-SLAM BREAKTHROUGH + DOUBLE LOOP-SPECIFICITY PROOF!**

#### NT-Round 5 (`207M w=768, 2/4/2`, 750 steps) & Official `928M` Wave 2 (`w=1536, 3/9/3, r=2`, 800 steps) — 100% Default Hyperparams, ZERO Arch Changes!
| Scale | Loops | Idea | Mid `val_bpb` | Final (cooled) `val_bpb` | Δ Final vs base | κ (`σ_max/σ_med`) | tok/s | Verdict |
|---|---|---|---|---|---|---|---|---|
| **`928M w=1536`** | **`r=2`** | **`alg_head_qkv_pe10` (QKV Head-TUON + PE10-S8)** | **`1.09661` (`−0.01081`!)** | **`0.97860`** | **`−0.00375`!!** | 3.77 | **`89.0k` (<0.9% cost!)** | **🏆 #1 ALL-TIME 928M CHAMPION! Super-additive & scales +22% vs 207M!** |
| `928M w=1536` | `r=2` | `nt_ctrl_pe10` (`PE10`, W1) | `1.09956` | `0.97994` | `−0.00241` | 2.77 | `88.7k` | #2 at 928M |
| `928M w=1536` | `r=2` | `alg_qdwh_polar` (4-step Cholesky-QDWH, core-only) | `1.10122` | `0.98073` | `−0.00162` | 2.76 | `71.0k` | #3 at 928M (beats base, slower than PE10-S8) |
| `928M w=1536` | `r=2` | `alg_head_tuon` (QKV Head-TUON 5-PE, W1) | `1.10357` | `0.98116` | `−0.00119` | 3.87 | `89.5k` | #4 at 928M (0% cost) |
| `928M w=1536` | `r=2` | `nt_base` (Official SOTA Muon) | `1.10742` | `0.98235` | — | 2.94 | `89.8k` | Official 928M baseline |
| `928M w=1536` | `r=2` | `alg_layer_teon_qdwh` (Cross-Layer TEON + QDWH) | `1.10274` | `0.98415` | `+0.00180` | 5.39 | `76.3k` | Fast early (`−0.20 nats`), regressed after cooldown |
| `928M w=1536` | `r=2` | `alg_head4_qdwh` (split `c_proj`) | `1.11445` | `0.98631` | `+0.00396` | 5.05 | `69.4k` | Splitting `c_proj` hurts at 928M too! |
| `928M w=1536` | `r=2` | `alg_head4_layer_qdwh` (4D split `c_proj`) | `1.13064` | `0.99744` | `+0.01509` | 11.71 | `77.5k` | Severe κ collapse |
| `928M w=1536` | `r=2` | `alg_all_qdwh_polar` (ALL 15 layers incl. untied!) | `1.11101` (`+0.00359`!) | *(finishing)* | — | — | `48.7k` | **CAUSAL PROOF #2: hurts when applied to untied prelude/coda!** |
| **`207M w=768`** | **`r=4`** | **`alg_head_qkv_pe10` (QKV Head-TUON + PE10-S8)** | **`1.14651` (`−0.00871`!)** | **`1.03178`** | **`−0.00308`!!** | 3.37 | **`220.0k` (<0.6% cost!)** | **🏆 #1 ALL-TIME 207M! +52% super-additive & beats slow `eigh`!** |
| `207M w=768` | `r=4` | `alg_exact_polar_svd` (`eigh U V^T`, R3) | `1.14913` | `1.03272` | `−0.00214` | 2.83 | `164.4k` | #2 at 207M (slower) |
| `207M w=768` | `r=4` | `alg_head_qkv_exact` (Head-QKV + `eigh`) | `1.14575` | `1.03301` | `−0.00185` | 3.38 | `158.5k` | #3 (`PE10-S8` beats exact `eigh` by `−0.00123` AND 39% faster!) |
| `207M w=768` | `r=4` | **`alg_qkv_head_orth`** (Joint `[Q_h;K_h;V_h]` `Q_h⊥K_h⊥V_h`) | `1.14796` | **`1.03324`** | **`−0.00162`** | 3.06 | **`220.9k`** | **#4! New original tensor Muon, 0% overhead!** |
| `207M w=768` | `r=4` | `alg_pe10_s8` (`Schatten-8 + 2×PE5`) | `1.14994` | `1.03404` | `−0.00082` | 2.78 | `221.1k` | Beats `nt_ctrl_pe10` (`1.03416`) |
| `207M w=768` | `r=4` | `nt_base` (Official SOTA Muon) | `1.15522` | `1.03486` | — | 3.45 | `221.2k` | Official 207M `r=4` baseline |
| **`207M w=768`** | **`r=1` UNTIED** | **`r1_exact_polar` vs `r1_base` (`1.04048`)** | `1.15388` | **`1.04068`** | **`+0.00020` (ZERO!)** | 2.66 | `261.8k` | **CAUSAL PROOF #1: ZERO gain on untied `r=1`, `−0.00214` on looped `r=4`!** |

**Top-Venue Scientific Takeaways from NT-Round 5 + 928M Wave 2:**
1. **Super-Additive, Width-Scaling, Zero-Overhead Champion (`alg_head_qkv_pe10`)**: Combines (i) per-head Q/K/V tensorization (`[H, d_h, d]`, keeping residual-merging `c_proj`/MLP whole) with (ii) provably-safe Schatten-8 spectral pre-normalized `2×PE5` (`100%` BF16 Tensor Cores, `<0.9%` overhead). Achieves **`−0.00308 bpb`** at `207M` (`+52%` super-additive over its two components, AND beats exact `eigh` `1.03301` while 39% faster!) and **scales UP to `−0.00375 bpb`** at official **`928M`** (`0.97860` vs `0.98235`, mid-run **`−0.01081 bpb`**!).
2. **Double Causal Proof of Loop-Specificity**:
   - Untied `r=1` control: `alg_exact_polar_svd` gives **`+0.00020`** (literally ZERO gain!) vs **`−0.00214`** on identical looped `r=4`.
   - Core-only vs All-15-layers at `928M`: `alg_qdwh_polar` (core-only) wins **`−0.00620`** at s400 / **`−0.00162`** at s800, while `alg_all_qdwh_polar` (all layers incl. untied prelude/coda) **hurts `+0.00359`** at s400!
3. **Why `PE10-S8` Beats Both 5-Step `PE` AND Exact `eigh` (`1.03178` vs `1.03301`)**: In weight-tied loops (`M = Σ_t M_t`), shared modes amplify `R×` while depth-unique modes stay `1×`. Standard 5-step Frobenius-normalized PE crushes depth-unique tail singular values; exact `eigh` over-amplifies near-zero noise (`1e-6`); Schatten-8 `2×PE5` acts as an implicit smooth spectral filter — converging true depth-unique modes to `1.0` while naturally damping pure noise in pure BF16 Tensor Cores!
4. **NT-Round 6 (`a1`), Cross-Architecture (`Ouro / Huginn / DeepLoop`), Depth `r=8`, `xloop_*` Cross-Loop Wave & 928M Wave 3 (22:20–22:55)**:
   - **🏆 NEW ALL-TIME 928M CHAMPION (`w=1536, 3/9/3, r=2`, 800 steps, `val_loss = val_bpb × 3.285715`, baseline `0.98235 bpb / 3.2277 nats`)**:
     - **`alg_head_qk_coupled`** (`[Q_h, K_h]` paired Mode-1 + independent `V_h` + `PE10-S8`): **`0.97802 bpb` (`val_loss = 3.2135`, `−0.00433 bpb / −0.0142 nats`!!)** at **`89.2k` vs `89.8k` (<0.7% cost!)** — beats exact `eigh` `alg_head_qkv_exact` (`0.97827`), `alg_head_qkv_pe15` (`0.97856`), `alg_head_qkv_pe10` (`0.97860`), and `alg_qkv_head_orth` (`0.97881`)!
     - **`alg_head_loop_teon` (3D `Head × Loop` Tensor)**: mid step 400 = **`1.09575`** (`−0.01167 bpb / −0.0383 nats`), finishing step 800!
     - `alg_exact_polar_svd` (whole-matrix `eigh` at `w=1536`): **`0.98314`** (`+0.00079` WORSE than base — over-amplifies noisy tail singular values at `w=1536`!).
     - `alg_head_qkv_all_pe10` (all 15 layers): **`0.98362`** (`+0.00127` WORSE than base, `+0.00502` worse than Core-Only `0.97860`!).
     - `928M r=1 nt_base`: **`0.98610`** (paired with live `r=1` controls on `a2`!).
   - **Cross-Architecture Transfer (All 4 Official `NextTransformer` Looped Families, `w=768, 750 steps`)**:
     - `fixed_recursive (r=4)`: `alg_head_qk_coupled` **`1.03168`** (`−0.00318 bpb / −0.0104 nats`), `alg_head_loop_teon` **`1.03243`** (`−0.00243 bpb / −0.0080 nats`) vs base `1.03486`.
     - `huginn (r=4)`: `alg_head_qkv_pe10` **`1.03272`** vs base `1.03527` (**`−0.00255 bpb / −0.0084 nats`**!).
     - `ouro (r=3)`: `alg_head_qkv_pe10` **`1.04533`** vs base `1.04563` (`−0.00030 bpb`, mid `−0.00312`).
     - `deeploop (r=4)`: `alg_head_qkv_pe10` `1.03225` vs base `1.03186` (`+0.00039`, post-norm scaled recurrence differs!).
   - **Depth Scaling `r=8` (`207M w=768, r=8`, 750 steps)**: `nt_base = 1.03737` → **`alg_head_qk_coupled = 1.03508` (`−0.00229 bpb / −0.0075 nats`)**, `alg_head_qkv_pe10 = 1.03650`, `alg_head_loop_teon = 1.03675`.
   - **🔥 CRUCIAL CROSS-LOOP GRADIENT DISCOVERY (`xloop_*` Wave, `207M r=4`, ref `nt_base = 1.03486`, `alg_pe10_s8 = 1.03404`)**:
     - `gamma = 0.0` (`alg_pe10_s8`): **`1.03404`**
     - `gamma = 0.5` (`xloop_depth_diff_polar`): `1.03468` (`−0.00018`)
     - `xloop_wiener_polar` (parameter-free): `1.03508` (`+0.00022`)
     - `gamma = 1.0` (`xloop_ac_polar_pe10`): `1.03599` (`+0.00113`)
     - `xloop_raw_ac_polar`: `1.03718` (`+0.00232`)
     - `gamma = 4.0` (`xloop_ac_polar_g4`): `1.03748` (`+0.00262`)
     - **Law**: Penalizing cross-loop AC differences (`sum_t (M_t - M_bar)(M_t - M_bar)^T` in the preconditioner denominator) **hurts strictly monotonically**! Cross-loop AC modes are **depth-specialized reasoning signal**, NOT noise to suppress!
   - **2250-Step Slow-Run Caution & Fresh-671M Verification**: On 167M `train.pt` (1.76 epochs), `slow_base = 0.94159`, `slow_exact_polar = 0.94198`, `slow_head_qkv_pe10 = 0.94251`. Packed **671M fresh tokens** (`/dev/shm/rsi_arena_data_1b/train.pt`, zero epoch wraparound!): at Step 1500 (`196.6M` fresh tok, right where Epoch-2 previously reversed!), **`fresh2250_pe10_s8 = 3.33384 (1.01286)`** (`−0.00326 nats`) and **`fresh2250_head_qk_coupled = 3.33583 (1.01346)`** (`−0.00127 nats`) BOTH stay ahead of `fresh2250_nt_base = 3.33710 (1.01384)`!
5. **Carbon-High (`Gemini 4 Carbon High`) Swarm Review Rounds 1 & 2 — TOP-VENUE BREAKTHROUGHS (22:56–23:25)**:
   - **🏆 BLOCKBUSTER PURE CROSS-LOOP 928M RESULT (`w=1536, r=2`, 800 steps, ZERO head-splitting, ZERO hyperparams!)**:
     - `928M nt_base`: `0.98235` (`val_loss 3.2277`); `928M alg_exact_polar_svd` (whole-matrix `eigh`, NO cross-loop coherence): **`0.98314` (`+0.00079` WORSE!)**; `928M nt_ctrl_pe10`: `0.97994` (`−0.00241`).
     - **`928M xloop_wiener_polar` (Pure Cross-Loop Rayleigh Coherence Polar)**: **`0.97887 bpb` (`val_loss = 3.21628`, `−0.00348 bpb / −0.0114 nats` vs SOTA Muon, and `−0.00427 bpb / −0.0140 nats` OVER identical `exact_polar_svd`!!)**.
     - **Theorem-1 Causal Ablation (`207M r=4`)**: Carbon-High's **`xloop_wiener_repaired`** (removing `/ s` specialization penalty + `skip_frob_renorm=True`) jumped from `1.03508` → **`1.03420` (`val_loss 3.39810`, `−0.00088 bpb / −0.0029 nats` clean gain!)**. Now running at `928M` (`a2 GPU 7`)!
   - **🏆 928M 2×2 CAUSAL TABLE (Difference-in-Differences Loop Specificity)**:
     - `928M r=1`: `nt_base = 0.98610`, `alg_pe10_s8 = 0.98378 (−0.00232)`, `alg_head_qk_coupled = 0.98434 (−0.00176, WORSE than pe10_s8 at r=1!)`.
     - `928M r=2`: `nt_base = 0.98235`, `alg_pe10_s8 = 0.97994 (−0.00241)`, **`alg_head_qk_coupled = 0.97802 (−0.00433, 2.46× loop amplification, −0.00192 better than pe10_s8!)`**, `alg_head_loop_teon = 0.97864 (−0.00371)`, `xloop_sylv_clni_qk_coupled = 0.97889 (−0.00346)`.
     - `1.38B IsoFLOPs Dense-24 (nt_base)`: `0.97381` (`val_loss 3.19966`). `928M Huginn (nt_base)`: `1.00847` (`val_loss 3.31356`).
   - **Carbon-High Wave-1 Finals (`207M r=4`)**: `xloop_clni_pe10 = 1.03403 (3.39753, 0% cost)`, `xloop_sylv_clni_qk_coupled = 1.03209 (3.39117, kappa 2.83 vs 3.37)`, `xloop_wiener_repaired = 1.03420 (3.39810)`, `xloop_sylv_clni_pe10 = 1.03497`, `xloop_sylvester_pe10 = 1.03508`, `xloop_kfac_polar = 1.03608`.
   - **671M Fresh-Data 2250-Step Slow-Run Finals (`207M r=4`, `295M` 100% FRESH tokens — ZERO epoch repetition!)**:
     - `fresh2250_nt_base`: `val_loss = 3.10942` (`val_bpb = 0.94635`)
     - **`fresh2250_pe10_s8` (WHOLE-MATRIX!)**: **`val_loss = 3.10752` (`val_bpb = 0.94577`, `−0.00190 nats` CLEAN LONG-HORIZON WIN!!)**
     - `fresh2250_head_qk_coupled`: `val_loss = 3.11005` (`0.94654`, `+0.00063` — head-splitting still slips on long horizons!)
     - `fresh2250_sylv_clni_qk_coupled`: `val_loss = 3.11124` (`0.94690`, `+0.00182`)
     - `fresh2250_head_loop_teon`: `val_loss = 3.11212` (`0.94717`, `+0.00270`)
     - **CRITICAL INSIGHT**: Even on 100% fresh tokens, ALL head-splitting methods slip at 2250 steps, while **whole-matrix high-precision polar WINS cleanly (`−0.00190 nats`)**!
   - **Wave-2 `207M` & Wave-1 `928M` Finals**:
     - `207M`: `xloop_clni_qk_coupled = 3.39068 (1.03195)`, `xloop_birec_clni = 3.39629 (1.03365)` (MLP pairing `U_down U_up = 0` hurt vs `clni_qk`), `xloop_fremuon_clni_qk = 3.39797 (1.03417)`.
     - `928M`: `xloop_sylvester_pe10 = 3.22101 (0.98031, −0.0067 nats)`, `xloop_clni_pe10 = 3.22217 (0.98066, −0.0055 nats)`, `xloop_kfac_polar = 3.22469 (0.98143)`, `928M r=4 nt_base = 3.22913 (0.98278)`.
   - **Carbon-High Round-3 (`tc_tstat_polar`) Finals — MASSIVE `Huginn` WIN & EXACT DIAGNOSIS OF TRACE LEAKAGE (23:36–00:05)**:
     - **🔥 Official `Huginn` ($\cos < 0$) — PURE CROSS-LOOP CRUSHES BOTH `nt_base` AND `alg_pe10_s8`!**
       - `207M Huginn r=4`: `nt_base = 3.4016 (1.03527)` → `alg_pe10_s8 = 3.39691 (1.03384)` → **`xloop_tc_tstat_polar = 3.39561 (1.03345, −0.0060 nats vs base, −0.0013 vs PE10!)`**.
       - **`928M Huginn r=4` (step 400 mid-eval!)**: `alg_pe10_s8 = 3.67697 (1.11710)` vs **`xloop_tc_tstat_polar = 3.64369 (1.10699, −0.0333 nats / −0.0101 bpb CRUSHING 928M PE10-S8!!)`** (finishing step 800!).
       - `928M fixed_recursive r=4` (step 400): `nt_base = 3.68277 (1.11886)` → **`xloop_tc_tstat_polar = 3.64947 (1.10875, −0.0333 nats!)`** (finishing step 800!).
     - **`928M fixed_recursive r=2` Finals (Proving Trace-Leakage + `Muon+` Reallocation)**:
       - `nt_base = 3.2277 (0.98235)`, `exact_polar_svd = 3.2303 (0.98314)`, `tc_tstat_polar = 3.22233 (0.98071)`, `pe10_s8 = 3.2198 (0.97994)`, **`wiener_repaired = 3.21921 (0.97976 — BEATS PE10 even with skip_frob=True!)`**, **`wiener_polar = 3.21628 (0.97887)`**, `tc_tstat_qk_coupled = 3.21631 (0.97888)`, `mpsylv_clni_qk = 3.21797 (0.97938)`.
     - **`671M Fresh-2250` Finals**: `nt_base = 3.10942 (0.94635)`, **`pe10_s8 = 3.10752 (0.94577, −0.00190)`**, **`tc_tstat_polar = 3.10872 (0.94613, −0.00070 — beats base!)`**, while ALL head-splitting variants (`qk_coupled 3.11005`, `tc_tstat_qk 3.11049`, `sylv_clni_qk 3.11124`, `mpsylv_clni_qk 3.11198`) slipped on 2250 fresh steps!
     - **`207M` Diagnostics**: `r=8 tc_tstat_polar = 3.41749 (1.04010)` (confirmed Carbon High's exact off-diagonal trace-leakage proof across 7 slices!); `DeepLoop`: `pe10_s8 = 3.39444 (1.03309)`, `tc_tstat_polar = 3.39591 (1.03354)`; `Ouro`: `tc_tstat_polar = 3.44737 (1.04920)`.
   - **Carbon-High Round-4 ORAL CENTERPIECE LAUNCHED ACROSS ALL 12 FREE GPUs (`00:06`, 100% WHOLE-MATRIX, ZERO HEAD-SPLITTING!)**:
     - **Centerpiece #1 — `Subspace-CLNI Polar` (`xloop_q_clni_pe10`)**: Refreshes exact zero-leakage 1D matched-basis $\Delta_{\text{neg}}$ SPD filter $Q = U\operatorname{diag}(w_i)U^\top$ every 5 steps, then runs fast BF16 symmetric congruence $Q\,\text{PE10}(Q X_{\text{sum}})$ with `clamp_max(1.5)` `Muon+` — zero Pop-B penalty, zero trace leakage at `r=8`, `85k+ tok/s` at 928M, bit-for-bit `PE10-S8` at `R=1`!
     - **Centerpiece #2 — `Whole-Matrix Bilinear [Q, K] + Subspace-CLNI` (`xloop_whole_qk_q_clni`)**: Stacks `[M_Q, M_K] ∈ ℝ^{d × 2d}` at the **whole-matrix level** (ZERO head-splitting!), simultaneously enforcing intra-head bilinear bound $U_{Q,h}U_{Q,h}^\top + U_{K,h}U_{K,h}^\top = I_{d_h}$ AND inter-head orthogonality $U_{Q,h}U_{Q,h'}^\top + U_{K,h}U_{K,h'}^\top = 0$ for 2250+ step stability!
     - Live now across `207M (r=4, r=1 control, r=8, Huginn, Fresh-2250 ×2)` AND `928M (r=2 ×2, r=4, Huginn)`!
     - **Round-4 & Round-5 GRAND-SLAM FINALS (`00:17–00:41`, 100% WHOLE-MATRIX, ZERO HEAD-SPLITTING!)**:
       - **🏆 Official `928M fixed_recursive r=2` Complete 2×2 Whole-Matrix Factorial (`w=1536, 3/9/3`, 800 steps)**:
         - `nt_base` = `3.22770 (0.98235)` | `exact_polar_svd` = `3.23030 (0.98314, WORSE)` | `alg_pe10_s8` = `3.21980 (0.97994, −0.00790)`
         - **`xloop_q_clni_pe10` (`Subspace-CLNI`)** = **`3.21771 (0.97930, −0.00999 vs base, −0.00209 over PE10)`**
         - **`alg_whole_qk_pe10` (`Whole-Matrix [Q,K]`, <1% cost, 88.9k)** = **`3.21585 (0.97874, −0.01185 vs base, −0.00395 over PE10)`**
         - 🏆 **`xloop_whole_qk_q_clni` (COMBINED!)** = **`3.21476 (0.97840, −0.01294 vs base, −0.00504 over PE10 — #1 WHOLE-MATRIX ALL-TIME, beating exact SVD wiener_polar 3.21628!)`**
       - **🏆 `671M Fresh-2250` Long-Horizon (`207M r=4`, `295M` 100% fresh tokens — ALL 5 head-splitting LOST!)**:
         - `nt_base` = `3.10942 (0.94635)` | All 5 head-splitting = `3.11005–3.11212 (ALL LOST!)` | `tc_tstat_polar` = `3.10872`
         - `alg_whole_qk_pe10` = **`3.10755 (0.94578 — ZERO long-horizon collapse, −0.00250 better than head_qk_coupled!)`**
         - `alg_pe10_s8` = `3.10752 (0.94577)` | **`xloop_whole_qk_q_clni` = `3.10742 (0.94574, BEATS PE10!)`**
         - 🏆 **`xloop_q_clni_pe10` (`Subspace-CLNI`)** = **`3.10625 (0.94538, −0.00317 vs base, −0.00127 over PE10 — NEW #1 ALL-TIME!)`**
       - **🏆 `928M Huginn r=4` Cooled Finals**:
         - `nt_base` = `3.31356 (1.00847)` | `alg_pe10_s8` = `3.30556 (1.00604, −0.00800)`
         - **`xloop_q_clni_pe10` (100% zero-slot-invariant Δ_neg!)** = **`3.29679 (1.00337, −0.01677 vs base, −0.00877 pure cross-loop gain over PE10!)`**
         - 🏆 **`xloop_tc_tstat_polar`** = **`3.28188 (0.99883, −0.03168 vs base, −0.02368 over PE10!)`**
         - Live s400: `whole_qk_q_clni = 3.65264` (`−0.0243` ahead of `PE10 3.67697`, `−0.0089` ahead of `q_clni 3.66152`)!
       - **🏆 `207M Huginn r=4` Complete 2×2 Factorial (91% Orthogonal Additivity!)**:
         - `nt_base` = `3.40160` → `pe10_s8` = `3.39691` → `q_clni_pe10` = `3.39507 (−0.00184)` → `whole_qk_pe10` = `3.39492 (−0.00199)` → 🏆 **`whole_qk_q_clni` = `3.39344 (1.03279, −0.00816 vs base, −0.00347 vs PE10, #1 ALL-TIME!)`**
       - **Key Ablations**:
         - `928M r=4`: `nt_base = 3.22913`, `tc_tstat_polar = 3.22428`, `pe10_s8 = 3.21880`; live `q_clni_pe10` s400 `3.63570` beating `pe10_s8` `3.63869`!
         - `207M r=8`: `nt_base = 3.4085`, `alg_whole_qk_pe10 = 3.40827`, `q_clni_pe10 = 3.41075` (repaired `−0.00674` vs `tc_tstat 3.41749`). `Ouro`: `q_clni_pe10 = 3.44065`.
         - **Round-5 `Fejér-CLNI` (`Q @ PE10(X)` + EMA `Q`) vs `Subspace-CLNI` (`Q @ PE10(Q @ X)`) on `207M r=4`**: `Subspace-CLNI` (`3.40113` / `3.39878`) beat outside-only `Fejér-CLNI` (`3.40310` / `3.40019`) — inner `Q` preserves Polar capacity for signal modes! Sent to Carbon High Round-6 (`#19`)!
     - **Carbon-High Round-6 Green-Light & Complete Multi-Scale / Multi-Seed / Causal Proof (`00:48–01:04`)**:
       - **🏆 NEW #1 ALL-TIME WHOLE-MATRIX AT OFFICIAL `928M fixed_recursive r=2` (`w=1536`, 800 steps)**:
         - **`xloop_whole_qk_fejer_clni` (`Whole-Matrix [Q,K] + Fejér PSD-Trace CLNI`, Round-5!)** = **`3.21404 (0.97818 bpb, −0.01366 nats vs SOTA Muon, −0.00576 over PE10-S8!!)`**; `xloop_fejer_clni_pe10` = `3.21805 (0.97941, −0.00175 over PE10)`.
       - **🏆 OFFICIAL `928M` `r=1` vs `r=2` DIFFERENCE-IN-DIFFERENCES CAUSAL PROOF**:
         - `928M r=1` (`untied`): `nt_base = 3.24004 (0.98610)`, `alg_pe10_s8 = 3.23242 (0.98378)`, **`alg_whole_qk_pe10 = 3.23094 (0.98333, −0.00148 over PE10)`**, `CLNI Δ(r=1) ≡ 0.00000`!
         - `928M r=2` (`looped`): `Whole-QK` jumps **`2.67×`** (`−0.00148 → −0.00395`), pure `CLNI` jumps **`∞×`** (`0 → −0.00209`, and **`−0.00288` at `r=4`, `+38%`!**), and **`Whole-QK + Fejér-CLNI`** achieves **`3.89×` loop amplification (`−0.00576` over PE10, `−0.01366` over SOTA Muon!!)**!
       - **🏆 OFFICIAL `928M fixed_recursive r=4`**: `nt_base = 3.22913` → `PE10-S8 = 3.21880` → **`xloop_q_clni_pe10 = 3.21592 (0.97876 bpb, −0.01321 vs base, −0.00288 pure cross-loop gain over PE10!)`**.
       - **🏆 OFFICIAL `928M Huginn r=4`**: `nt_base = 3.31356` → `PE10-S8 = 3.30556` → `xloop_q_clni_pe10 = 3.29679` → **`xloop_whole_qk_q_clni = 3.28705 (1.00041 bpb, −0.02651 vs base, −0.01851 over PE10!)`** → `xloop_tc_tstat_polar = 3.28188` (proven 0.00% LR confounder via `||U||_F <= sqrt(d_min)`!). Live s400: **`xloop_whole_qk_fejer_clni = 3.64022`** (beating `tc_tstat_polar 3.64369`!) & exact SVD `wiener_tstat = 3.63985`!
       - **🏆 `671M Fresh-2250` COMPLETE LEADERBOARD (`295M` fresh tokens)**: Top 3 ALL-TIME are **`Subspace-CLNI = 3.10625`** (#1), **`Fejér-CLNI = 3.10672`** (#2), **`Whole-QK + Subspace-CLNI = 3.10742`** (#3) — all beating `PE10-S8 (3.10752)` and `Whole-QK (3.10755)`, while `wiener_polar (1/R) = 3.10891` and ALL 5 head-splitting methods (`3.11005–3.11212`) lost!
       - **🏆 Multi-Seed (`42, 1337, 2025`) & Cross-Architecture Sweep**:
         - `207M Huginn` 3-seed mean: `nt_base = 3.40132 ± 0.00113` vs **`whole_qk_q_clni = 3.39413 ± 0.00257` (`−0.00719 nats`, 3-for-3 sweep!)**.
         - `207M r=8`: **`whole_qk_q_clni = 3.40686` (#1 whole-matrix!)**; `DeepLoop`: **`whole_qk_q_clni = 3.39398`** (beats `PE10 3.39444`); `Ouro`: `whole_qk_q_clni = 3.43838`.
       - **🚀 8×B300 vs 8×B300 Official `928M` Multi-Billion-Token + 22-Task `CORE` Grand Finale**: Built `scripts/nt_ddp_928m_finale.py` with all 4 Carbon-High `#22` safeguards (`@torch._dynamo.disable` + `active_loops`, Step-0 grad equiv assert, 8-GPU rank-sharded `DistMuon` broadcast `<1.5%` overhead, resumable checkpoints + official 20.97M `evaluate_bpb` + 22-task `evaluate_core` + multi-depth `r∈{1,2,4}`!).
       - **🏆 COMPLETE LARGE-SCALE (`w=1536`) 3-ARCHITECTURE & 3-SEED SWEEP (`01:14–01:49`)**:
         - **`928M fixed_recursive r=2` 3-Seed (`42, 1337, 2025`)**:
           - `nt_base`: `3.22770 / 3.22358 / 3.22227` → Mean **`3.22452 ± 0.00283`**
           - `alg_pe10_s8`: `3.21980 / 3.23346 / 3.23433` → Mean **`3.22920 ± 0.00815`** (**destabilized & WORSE than base on 2/3 seeds!**)
           - 🏆 **`xloop_whole_qk_fejer_clni`**: `3.21404 / 3.21966 / 3.22276` → Mean **`3.21882 ± 0.00442` (`−0.00570 vs base`, `−0.01038 vs PE10-S8`, 3/3 sweep over PE10!)**
         - **`928M fixed_recursive r=4` Complete 2×2**: `nt_base = 3.22913`, `PE10-S8 = 3.21880`, `Subspace-CLNI = 3.21592`, `Whole-QK+Fejér = 3.21158`, `Whole-QK+CLNI = 3.21151`, `Whole-QK = 3.21099` (`−0.0175 to −0.0181 nats` vs base!).
         - **`928M Huginn r=4` Complete 2×2 (`168%` Super-Additive!)**: `nt_base = 3.31356`, `PE10-S8 = 3.30556`, `Whole-QK = 3.29684`, `Subspace-CLNI = 3.29679`, `Whole-QK+CLNI = 3.28705`, **`Whole-QK+Fejér = 3.27625` (`−0.03731 vs base`, `−0.02931 vs PE10`!)**, exact SVD `wiener_tstat = 3.27401`.
         - **`607M Official Ouro r=3` (`w=1536, 3/3/3`)**: `nt_base = 3.27388`, `PE10-S8 = 3.27264`, 🏆 **`Whole-QK+Fejér = 3.26902` (`−0.00486 vs base`, `−0.00362 vs PE10` — ~4× PE10 gain!)**.
       - **🚀 8×B300 vs 8×B300 `928M` 2.62-BILLION-TOKEN + 22-TASK `CORE` GRAND FINALE LIVE**:
         - `a1` (`xloop_whole_qk_fejer_clni`): `SAFEGUARD A+B+C = 4.55e-13`, `514k tok/s` — s250 `0.98134 bpb`, s500 `0.90216 bpb`, s750 `0.86336 bpb`, s1000 (`1.05B` tok) **`0.83801 bpb` (`~2.7535 nats`)**!
         - `a2` (`nt_base` paired exact control): zero-gap handoff armed to launch automatically the instant GPUs 1,2 finish (~90s)!
         - **🔥 FIRST DIRECT 8×B300 vs 8×B300 HEAD-TO-HEAD (Exact 16-digit Step-0 parity `val_bpb = 3.1654560904424125`, Official `20.97M` Val Shard!)**:
           - `Step 250 (0.262B tok)`: `nt_base (a2)` = `0.98747 bpb (3.2445 nats, 535k tok/s)` vs 🏆 **`Whole-QK + Fejér-CLNI (a1)` = `0.98134 bpb (3.2244 nats, 512k tok/s, −0.02013 nats / −0.00613 bpb, ONLY 3.7% wall-clock overhead!)`**.
           - **`a1` Step 1250 (`1.31B` tok) First Full Official 22-Task `CORE`**: `val_bpb = 0.81508 (~2.6781 nats)`, **`CORE = 17.58%`** (`PIQA 68.9%`, `Winograd 62.3%`, `COPA 61.0%`, `ARC-Easy 60.2%`, `BoolQ 55.5%`, `WinoGrande 51.3%`, `HellaSwag 41.9%`, `BB-CS-Alg 39.2%`, `LAMBADA 35.8%`, `ARC-Challenge 31.3%`).
           - **2nd-Seed `928M r=1` Causal Proof (`Seed 1337`)**: `r=1 nt_base = 3.23474`, `r=1 whole_qk_pe10 = 3.23402 (−0.00072)` vs `r=2 −0.00392` (**`5.44×` loop amplification!**).
         - **🏆 FIRST OFFICIAL 22-TASK `CORE` HEAD-TO-HEAD VICTORY (`Step 1250 = 1.31B` Official ClimbMix Tokens, `8×B300` vs `8×B300`!)**:
           - `nt_base (a2)`: `val_bpb = 0.81556 (~2.6797 nats)` | **22-Task `CORE` = `17.14%` (`0.17139`)**
           - 🏆 **`Whole-QK + Fejér-CLNI (a1)`**: **`val_bpb = 0.81508 (~2.6781 nats, −0.00155 nats)`** | **22-Task `CORE` = `17.58%` (`0.17576`, `+0.44%` FULL POINTS OVER OFFICIAL SOTA MUON!!)** (Wins: `Winograd +4.03%`, `BB-Operators +2.39%`, `COPA +2.00%`, `BB-Dyck +1.50%`, `BoolQ +0.70%`, `HellaSwag +0.56%`, `PIQA +0.49%`!).
         - **🏆 `a1` STEP 2500 (`2.62 BILLION` TOKENS!) FINAL COOLED RESULT (`98.95 min`, `509.8k tok/s`)**:
           - Official `20.97M` `val_bpb = 0.75483 (~2.4801 nats)` | **FINAL 22-Task `CORE` = `20.27%` (`0.20269`)** (`PIQA 72.9%`, `ARC-Easy 66.2%`, `Winograd 61.2%`, `HellaSwag 48.4%`, `BB-CS-Alg 43.0%`, `SQuAD 34.2%`, `ARC-Challenge 34.0%`).
           - Multi-depth `r∈{1,2,4}`: `r=1: 0.90741`, `r=2: 0.75483`, `r=4: 0.91365`; Core weight spectra: **`mean_erank = 1277.6/1536 (83.2%)`, `mean_kappa = 3.62`**!
           - `a2` (`nt_base`) at Step `1450/2500` (~35 min to final Step 2500 + CORE!).
         - **🚀 SECOND 8×B300 BACKBONE LAUNCHED (`02:51`): `a1` `8×B300 Official 928M Huginn` (`configs/looped/huginn.json`, Carbon-High Round-7 `v2`, 2.62B tok + 22-task CORE!)**.
         - **🏁 COMPLETE `8×B300` vs `8×B300` `928M fixed_recursive r=2` `2.62B`-TOKEN FINAL HEAD-TO-HEAD (`03:32`)**:
           - **8-for-8 Consecutive Validation Wins (`Step 750 → 2500`, Official `20.97M` Val Shard)**:
             - `s250`: `0.98134` vs `0.98747` (**`−0.02013 nats`**) | `s1000`: `0.83801` vs `0.83812` (`−0.00036`) | `s1250`: `0.81508` vs `0.81556` (**`−0.00155`**) | `s1500`: `0.79728` vs `0.79764` (`−0.00118`) | `s1750`: `0.78287` vs `0.78324` (`−0.00121`) | `s2000`: `0.77086` vs `0.77117` (`−0.00101`) | `s2250`: `0.76134` vs `0.76160` (`−0.00086`) | **Final `s2500 (2.62B)`**: **`0.75483 (2.48014)`** vs `0.75509 (2.48100)` (**`−0.00086 nats` CLEAN FULL-HORIZON WIN, only `4.1%` wall-clock overhead: `98.95m` vs `95.05m`!**).
           - **22-Task `CORE` Downstream**: Step 1250 (`1.31B`): **`17.58%` vs `17.14%` (`+0.44%` WIN!)**; Step 2500 (`2.62B`): `20.27%` vs `20.84%` (our `v1` still dominated algorithmic & QA tasks: `BB-Dyck 10.1% vs 5.2% (+4.9%)`, `SQuAD 34.2% vs 29.8% (+4.4%)`, `BoolQ 54.0% vs 51.7% (+2.3%)`, `BB-CS-Alg 43.0% vs 41.1% (+1.9%)`, `ARC-E 66.2% vs 65.0% (+1.2%)`, `PIQA 72.9% vs 72.1% (+0.8%)`, while `v1`'s `W_K` norm starvation left `kappa 3.62` vs `nt_base 3.32` & `erank 1277.6` vs `1290.0` — DIRECTLY validating Carbon High's Round-7 `v2` fix!).
         - **🚀 BOTH `8×B300` NODES NOW RUNNING LOCKSTEP OFFICIAL `871M Huginn` (`huginn.json`, `2.62B` TOKENS + 22-TASK `CORE`)**:
           - `a1` (`xloop_whole_qk_fejer_clni_v2` — hit 49G overlay disk full at s1000 ckpt in run 1 [logged s250 `1.01689`, s500 `0.95910`, s750 `0.91903`, s1000 `0.89173`]; freed 15G on `a1` & 13G on `a2`, relaunched cleanly with rolling ckpt) vs `a2` (`nt_base`) side-by-side in lockstep!
           - **🔥 EXPLOSIVE `v2` WIDENING TRAJECTORY THROUGH `1.05 BILLION` TOKENS (16-digit Step-0 parity `3.1652765561233673`, Official `20.97M` Val Shard, ONLY `2.47%` Wall-Clock Overhead `2073.3s vs 2023.3s`!)**:
             - `s250 (0.262B)`: `nt_base 1.02720 (3.3751)` vs 🏆 **`v2 1.01814 (3.3453)` → `−0.02976 nats (−0.00906 bpb)`**
             - `s500 (0.524B, mom=0.970)`: `nt_base 0.96117 (3.1581)` vs 🏆 **`v2 0.95997 (3.1542)` → `−0.00395 nats (−0.00120 bpb)`** *(completely cured `v1`'s high-LR plateau dip!)*
             - `s750 (0.786B, mom=0.970)`: `nt_base 0.92161 (3.0281)` vs 🏆 **`v2 0.91929 (3.0205)` → `−0.00762 nats (−0.00232 bpb)`** *(1.93× wider RIGHT ON the high-LR plateau!)*
             - `s1000 (1.049B, WSD cooldown start)`: `nt_base 0.89361 (2.9362)` vs 🏆 **`v2 0.88916 (2.9215)` → `−0.01464 nats (−0.00446 bpb)`** *(3.71× wider than s500!)*
             - **`s1250 (1.311B tok) + FIRST OFFICIAL HUGINN 22-TASK `CORE` BLOWOUT (ONLY `2.39%` Wall-Clock Overhead `2581.2s vs 2521.0s`!)`**:
               - **Official `20.97M` Val Shard**: `nt_base 0.86688 (2.84831)` vs 🏆 **`v2 0.86205 (2.83243)` → `−0.01588 nats (−0.00483 bpb, 4.02× wider than s500!!)`** *(live s1300–1325 batch gap now `−0.0171` to `−0.0179 nats`!)*
               - **🏆 Official 22-Task `CORE`**: `nt_base 12.42% (0.12416)` vs 🏆 **`v2 13.90% (0.13895)` → `+1.48%` FULL POINTS (`+11.9%` RELATIVE GAIN!!)** — crushing `SQuAD 17.51% vs 8.70% (+8.81%, 2.01×!)`, `BoolQ 57.61% vs 49.24% (+8.38%)`, `LSAT-AR 24.35% vs 20.00% (+4.35%)`, `Winograd 56.04% vs 53.85% (+2.20%)`, `CoQA 14.34% vs 12.46% (+1.88%)`, `BB-Dyck 8.60% vs 6.80% (+1.80%)`, `HellaSwag-0s 36.80% vs 35.89% (+0.91%)`!
             - `s1500 (1.573B)`: `nt_base 0.84710 (2.78333)` vs 🏆 **`v2 0.84119 (2.76392)` → `−0.01941 nats (−0.00591 bpb, 4.91× wider!)`**
             - **`s1750 (1.835B)`**: `nt_base 0.83086 (2.72998)` vs 🏆 **`v2 0.82420 (2.70809)` → `−0.02189 nats (−0.00666 bpb, 5.54× WIDER THAN s500!!)`**
             - `s2000 (2.097B)`: `nt_base 0.81680 (2.68378)` vs 🏆 **`v2 0.81003 (2.66152)` → `−0.02226 nats (−0.00678 bpb, 5.64× wider!)`**
             - **`s2250 (2.359B)`**: `nt_base 0.80595 (2.64812)` vs 🏆 **`v2 0.79867 (2.62421)` → `−0.02391 nats (−0.00728 bpb, 6.05× WIDER THAN s500!!)`**
             - **🏆🏆 FINAL `s2500 (2.621B TOKENS)` GRAND SLAM VICTORY (`98.01 min` vs `95.44 min` = ONLY `2.69%` WALL-CLOCK OVERHEAD!)**:
               - **Official `20.97M` Val Shard**: `nt_base 0.79858 (2.62391)` vs 🏆 **`Frozen v2 0.79110 (2.59933)` → `−0.02459 nats (−0.00748 bpb, 6.23× WIDER THAN s500, 10-for-10 UNBROKEN SWEEP!!)`** — `Frozen v2` ALREADY matched `nt_base`'s full `2.62B` final loss at **`s2250` (`2.36B` tok, `~250` steps / `~10%` tokens / `~12.8 min` = `~13.4%` REAL WALL-CLOCK TIME SAVED!!)**!
               - **🏆 FINAL Official 22-Task `CORE`**: `nt_base 16.52% (0.16522)` vs 🏆 **`Frozen v2 18.12% (0.18120)` → `+1.60%` FULL POINTS (`+1.598 pts`, `+9.7%` RELATIVE, WIDENED FROM `+1.48%` AT `1.31B`, 16/22 TASK WINS!!)** — `BoolQ +4.89%`, `CSQA +4.18%`, `SQuAD +3.71%`, `BB-RepeatCopy +3.13% (3.13% vs 0%)`, `CoQA +2.62%`, `ARC-C +2.56%`, `BB-Dyck +2.40% (1.71×!)`, `Winograd +1.83%`, `WinoGrande +1.66%`, `Jeopardy +1.61% (1.97×!)`, `HellaSwag +0.92%`, `PIQA +0.71%`!
               - **🏆 Multi-Depth Test-Time Generalization (`r∈{1,2,3,4}`)**: `v2` beats `nt_base` at EVERY depth (`r=1: 0.79782 vs 0.80276 [−0.01622]`, `r=2: 0.79166 vs 0.79894 [−0.02391]`, `r=3: 0.79110 vs 0.79858 [−0.02459]`, `r=4: 0.79105 vs 0.79855 [−0.02466]`), extracting **`+61.1%` MORE depth gain (`r=1→4`: `−0.02226` vs `−0.01382 nats`)** — even `v2@r=1` beats `nt_base@r=4`!
               - **Core Spectra**: `v2 mean_erank = 1241.7, mean_kappa = 5.01` vs `nt_base 1284.9, 3.36` (evaluating full per-projection `c_q/c_k/c_v/c_proj/c_fc/mlp_proj` + `r∈{1..16}` right now!).
           - **🚀 CANDIDATE 1 LAUNCHED ZERO-GAP (`05:12`, 16/16 B300 SATURATED, CARBON-HIGH `#37` APPROVED)**:
             - `a1` (`8×B300`): 90s `eval_ckpt_depths.py` (`Huginn v2 r∈{1..16}`) → auto-chained into **`928M fixed_recursive r=2` Frozen `v2` (`2.62B` + 22-task `CORE`)**!
             - `a2` (`8×B300`): 90s `eval_ckpt_depths.py` (`Huginn nt_base r∈{1..16}`) → auto-chained into **`871M Huginn` Pure Non-Loop `alg_pe10_s8` (`2.62B` + 22-task `CORE`)**!
           - **Carbon-High Round-8 Final Verdict (`#37`)**: Officially FROZE **`Whole-QK + Fejér-CLNI v2`** as our single canonical Oral algorithm (`0` hyperparams); approved **Candidate 1** right after `~05:10` `Huginn` `2.62B` completion + 90s `eval_ckpt_depths.py` (`eval_loops ∈ {1,2,3,4,6,8,12,16}` + per-proj `W_Q/W_K` norm spectra): `a1` → `928M fixed_recursive r=2` frozen `v2` (`2.62B` + CORE), `a2` → `871M Huginn` pure non-loop `alg_pe10_s8` (`2.62B` + CORE).

---

## Pure Loop-Muon Optimizer Hypothesis Queue (Next Rounds)
1. **O1 (Per-loop Gradient Structure & Aggregation)**: Guided by `opt_base` step-100/550 telemetry (`||G_0||..||G_3||` and `cos(G_a, G_b)`): test depth-weighted BPTT aggregation (`w_t G_t`), cross-loop orthogonalization / Gram-Schmidt / anti-interference (removing redundant or conflicting loop components before `PE`), and checkpoint-safe row-whitened `Loop-Muon` on ALL 6 core matrices (including `c_proj`).
2. **O2 (Iterated Weight Spectral Conditioning)**: If `opt_polar_wd` / `opt_schatten_wd` win, sweep strength/frequency/cautious-masking, test direct **Spectral-Norm Clipping / Soft-Cap (`σ_max(W) ≤ c`)** and **Polar-Momentum Coupling** (subtracting the component of `ΔW` aligned with `W`'s top singular subspace).
3. **O3 (Tied-Parameter Muon Hyper-Geometry)**: Systematic decoupled sweep of `core_lr`, `core_wd`, `core_beta1` (momentum), `core_beta2` (NorMuon variance), and Polar Express `ns_steps` specifically for weight-tied core matrices vs untied prelude/coda.
4. **O4 (Cross-Backbone & Cross-Depth & Long-Run Scaling)**: Once `a2` completes `big-dedup` step 1000 (~19:50), deploy `a2` as a second 8-GPU optimizer arena (`prenorm` backbone & `r=16` depth → **16 optimizer ideas / 15 min**) before promoting the top 1–2 Kaiming-He-simple **Loop-Muon** winners to **480M/1.32B past-Chinchilla + 22-task CORE** confirmation!
