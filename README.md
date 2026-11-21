# ReLoop

**2 Scalars (8 Bytes) Fix Deep Looped Transformers.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

In a weight-tied looped transformer, running the same core stack $R$ times per token lets a small model simulate deep computation — yet standard residual looping stalls, drifts, and explodes when changing test-time depth.

**ReLoop** fixes deep looping with **one equation, two learned scalar floats (8 bytes total), and zero added normalization layers**:

$$x_{r+1} = x_r + \frac{1}{R}\Big(\alpha\big(F_{\text{core}}(x_r) - x_r\big) + \beta\,x_{\text{pre}}\Big), \qquad \alpha,\beta \in \mathbb{R}$$

where $x_{\text{pre}}$ is the post-prelude hidden state, $F_{\text{core}}$ is the shared weight-tied core stack, and $R$ is the loop count (which can vary freely at test time).

---

## Headline Results

All numbers below come directly from paired `8×B300` / `16×B300` DDP runs with identical initialization checksums, identical data order, and archived raw JSON receipts in [`results/`](results/).

### 1. Vertical Depth-Looping (`866M`, `3/8/3`, `R = 8` → `64` Virtual Core Layers, `2.62B` Tokens)

Paired on `8×B300` DDP under the `NextTransformer` ClimbMix protocol (`width=1536`, `heads=12`, `mlp=6528`, `seq=2048`, `global_batch=1M` tokens). Both Full-8-BPTT arms share identical trunk weights (`init_sha = 0301da0f934d4ca2`) and matched clean `x0_lambdas` LR (`--clean-x0-lr`), isolating the entire gain strictly to the `+2` scalar parameters.

| Model (`866M`, `R = 8`) | Extra Params | BPTT | Val Loss ↓ | 22-Task CORE ↑ | `r = 6` Val Loss (`−25%` Core FLOPs) | `r = 32` (`256L`) Val Loss | Throughput | Peak VRAM | Receipt |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `Fixed Recursive` (Steelman) | `+0` | Full 8 | `2.55535` | `19.11%` (`14.35%` @ mid) | `26.903` (collapsed) | `2.7097` | `196.0k tok/s` | `162.3 GB` | [JSON](results/kh_finale_866m/a2_fixed_base_cleanx0_r8/final_summary.json) |
| **`ReLoop` (Euclidean)** | **`+2` (`8 B`)** | **Full 8** | **`2.45869`** (**`−0.09665`**) | **`21.97%`** (**`+2.86%`**, `16W/0T/6L`) | **`2.4918`** (beats baseline `r = 8`!) | **`2.4894`** (**`−0.220`**, `5×` flatter) | `195.1k tok/s` (`0.46%` parity) | `163.0 GB` | [JSON](results/kh_finale_866m/a1_scalar2_ode_r8/final_summary.json) |
| `ReLoop-Spherical` (`4-BPTT`) | `+1,537` (`6 KB`) | Trunc. 4 | **`2.52069`** | `18.84%` | `2.5525` | **`2.5259`** (`+0.0052` drift @ `4×` depth) | **`277.1k tok/s`** (**`+41.3%`**) | **`101.0 GB`** (**`−38%`**) | [JSON](results/kh_finale_866m/a1_sph_ode_tbptt4_r8/final_summary.json) |

Key takeaways:
- **30.0% data savings & 27.9% wall-clock savings**: `ReLoop` at `1.84B` tokens already surpasses the Steelman baseline at `2.62B` tokens.
- **Zero-shot depth elasticity**: Dropping test-time depth from `r = 8` to `r = 6` (`−25%` core compute) still beats the baseline at `r = 8` (`2.4918` vs `2.5553`, while the baseline collapses to `26.90`). Extrapolating `4×` deeper to `r = 32` (`256` virtual core layers) stays stable (`2.4894`).
- **Memory-saving truncated BPTT**: Under `4-BPTT`, `ReLoop-Spherical` cuts VRAM by `38%`, runs `41.3%` faster, beats Full-8-BPTT `Fixed Recursive` (`2.5207` vs `2.5553`), and outperforms the official `871M` `Huginn r=8` reference (`2.5973` Val Loss, `18.06%` CORE → **`−0.0766` Val Loss, `+0.78%` CORE**).

---

### 2. Horizontal Latent Feedback (`ifm-ai/LoRM`, `778M`, `24L`, `16×B300`, Complete 6-Cell Iso-FLOP Matrix)

Applying the same scale-matching principle to horizontal latent-feedback transformers ([`ifm-ai/LoRM`](https://github.com/ifm-ai/LoRM)), we identified two root bottlenecks in verbatim `LoRM`:
1. **Step-0 SNR cliff (`27.7×` RMS / `768×` variance mismatch)**: token embeddings initialize at `RMS ≈ 0.020`, whereas `W_fb(RMSNorm(h_top))` arrives at `RMS ≈ 0.554`, shocking early training.
2. **Myopic top-layer tap (`L = 24`)**: tapping the final layer provides zero causal attention layers above the tap during Pass 1 and burns `100%` of the trunk on the `no_grad` prefill pass.

`TapFeedbackGate` adds **`+1` scalar float (`4 bytes`)** via $x_{\text{in}, t} = x_{\text{emb}, t} + (\alpha/\sqrt{d})\,W_{\text{fb}}(\text{RMSNorm}(h_{\text{tap}, t-1}))$ at mid-trunk depth (`2L/3`, i.e., `L = 16` for `24L`), preserving exact `1.00×` standard autoregressive decoding.

Complete 6-cell Iso-FLOP (`C_total`) matrix at `778M` (`24L, w=1536, h=12, mlp=4096, seq=2048`, exact `NorMuon` optimizer, `22-task CORE`, AR fixed-point `P=8`):

| Data Regime | Model (`778M`, Iso-FLOP `C_total`) | Steps / Tokens | Val Loss ↓ | 22-Task CORE ↑ | Wall-Clock | Receipt |
|---|---|---|---:|---:|---:|---|
| **`F = 1` (Fresh)** | `Vanilla 778M` | `5,069 / 2.658B` | **`2.54961`** | **`19.36%`** | `55.6m` | [JSON](results/lorm_778m/a2_f1_vanilla/final_summary.json) |
| **`F = 1` (Fresh)** | `Verbatim LoRM 778M` (`0→24`) | `4,000 / 2.097B` | `2.58249` | `17.19%` | `63.0m` | [JSON](results/lorm_778m/a2_f1_lorm_verbatim/final_summary.json) |
| **`F = 1` (Fresh)** | **`Tap-L16 + 1 Scalar` (`0→16`, `+4 B`)** | `4,300 / 2.254B` | **`2.55460`** (`−0.02789` vs LoRM) | **`17.63%`** (`+0.44%` vs LoRM, `14W/1T/7L`) | `60.7m` (`−2.3m`) | [JSON](results/lorm_778m/a1_f1_c1_tap_l16/final_summary.json) |
| **`F = 4` (`524M × 4`)** | `Vanilla 778M` | `5,069 / 2.658B` | `2.80099` | `14.07%` | `51.9m` | [JSON](results/lorm_778m/a1_f4_vanilla/final_summary.json) |
| **`F = 4` (`524M × 4`)** | `Verbatim LoRM 778M` (`0→24`) | `4,000 / 2.097B` | **`2.72026`** (`−0.08073` vs Vanilla) | `15.49%` (`+1.42%` vs Vanilla) | `59.4m` | [JSON](results/lorm_778m/a2_f4_lorm_verbatim/final_summary.json) |
| **`F = 4` (`524M × 4`)** | **`Tap-L16 + 1 Scalar` (`0→16`, `+4 B`)** | `4,300 / 2.254B` | `2.73269` (`−0.06830` vs Vanilla) | **`16.04%`** (**`+0.55%` vs LoRM, `+1.97%` vs Vanilla**) | `57.1m` (`−2.3m`) | [JSON](results/lorm_778m/a1_f4_c1_tap_l16/final_summary.json) |

At `134M` (`12L, w=768`, `~4.8 tok/param`, [`results/lorm_w1a..w3b/`](results/)), `Tap-L6 + 1 Scalar` (`3.0597`) beats both `LoRM` (`3.0900`) and `Vanilla` (`3.0646`) on `F = 1`, while `F = 4` exhibits an inverted-U hop-depth curve peaking at `Tap-L10` (`3.1667` vs `LoRM 3.1716` vs `Vanilla 3.2157`) — explaining why `L = 16` (`67%` depth) dominates `F = 1` and wins `#1` on `F = 4` `CORE` (`16.04%`) while `L = 24` (`100%` depth) holds a `0.0124` Val Loss edge on `F = 4` regularization.

---

## Quickstart

[`reloop.py`](reloop.py) is a standalone, zero-dependency PyTorch implementation of both `ReLoop` and `TapFeedbackGate`:

```python
import torch
from reloop import ReLoop, TapFeedbackGate

# 1. Vertical depth-looping (+2 floats = 8 bytes, zero added norms)
reloop = ReLoop(alpha_init=4.0, beta_init=1.5)  # use spherical=True for truncated BPTT
x = reloop(x_pre, lambda h, r: core_stack(h), loops=8)

# 2. Horizontal latent feedback (+1 float = 4 bytes, strictly causal shift)
gate = TapFeedbackGate(dim=1536, alpha_init=2.0)
x_in = gate(x_emb, fb_proj)  # fb_proj = W_fb(rms_norm(h_tap))
```

Run the unit test suite (verifies parameter counts, brute-force numerical equivalence, causality, BF16 dtype preservation, `tbptt_loops` boundary behavior, and depth extrapolation `R = 2..32`):

```bash
python test_reloop.py
```

---

## Why It Works

1. **Angular step-size annealing without normalization**: In a Pre-Norm residual network, the hidden state norm $\|x_r\|$ grows monotonically across loops, naturally annealing the effective angular update size $\propto 1/\|x_r\|$. Inserting an `RMSNorm` between loops destroys this annealing under Full-BPTT; omitting the `1/R` step scaling and `x_pre` anchor causes trajectory drift and test-time depth collapse (`r = 6` explodes to `26.90`).
2. **Continuous-depth stability**: Scaling the tied core residual by $\alpha/R$ and injecting a constant prelude anchor $(\beta/R)\,x_{\text{pre}}$ turns discrete loop stacking into a well-posed driven Euler integration whose trajectory stays coherent from `r = 6` through `r = 32` (`256` virtual core layers).
3. **Why Spherical ReLoop wins under truncated BPTT**: When early loops are detached (`tbptt_loops < R`), autograd can no longer coordinate the late-loop residual scale with early-loop norm growth; projecting onto the unit RMSNorm sphere restores a stationary scale across the `detach()` boundary.

---

## Honest Scope & What Didn't Work (~380+ B300 Runs)

Every negative result and confounder we encountered is preserved in [`RESEARCH_LOG.md`](RESEARCH_LOG.md):

- **Clean baseline attribution**: Both `866M` Full-8-BPTT runs include `--clean-x0-lr` (`AdamW lr = 0.003` instead of the native `0.707` scalar LR on `x0_lambdas`), which was a major non-loop confounder in early sweeps.
- **Token horizon**: The `866M` vertical runs (`2.62B` tokens) and `778M` `LoRM` runs (`2.10B–2.66B` tokens = `2.7–3.4 tok/param`) are sub-Chinchilla; at `778M F = 1`, `Vanilla` (`2.5496 / 19.36%`) still leads `Tap-L16` (`2.5546 / 17.63%`) on `F = 1`, while both feedback models strongly outperform `Vanilla` on `F = 4`. A full `49.152B`-token official `NextTransformer` (`nextt-reference-50b-v1`) leaderboard run is in progress.
- **Falsified approaches (archived in [`RESEARCH_LOG.md`](RESEARCH_LOG.md) and [`legacy/`](legacy/))**:
  - **Pure optimizer tweaks on weight-tied loops** (`loop_muon`, per-loop gradient whitening, Fejer/CLNI, polar-express step counts): once `x0_lambdas` and LR schedule confounders were controlled, pure weight-space Muon modifications yielded $\le 0.004$ Val Loss net loop-specific gain.
  - **Heavy architectural loop hooks**: inter-loop `RMSNorm` under Full-BPTT, per-loop `LoRA`, Lie-group rotations, `AdaLN`, `mHC`, `DepthAttnRes`, `WeightField`, `DepthSlide`, and `Ribbon` all either regressed or lost to 2-scalar `ReLoop` at `0%` overhead.
  - **Curriculum scheduling hacks on `LoRM`**: `Warm10/Late40` pass-mix scheduling won on `F = 1` (`134M`) but collapsed by `+0.021` Val Loss on `F = 4` and was rejected.

---

## Repository Structure

| Path | Description |
|---|---|
| [`reloop.py`](reloop.py) | Standalone PyTorch reference implementation (`ReLoop` & `TapFeedbackGate`) |
| [`test_reloop.py`](test_reloop.py) | Unit tests for `reloop.py` |
| [`RESEARCH_LOG.md`](RESEARCH_LOG.md) | Complete chronological experimental ledger across all waves & negative results |
| [`results/`](results/) | Raw JSON / JSONL receipts (`kh_finale_866m/`, `lorm_778m/`, `lorm_w1a..w3b/`, `ddp_928m/`) |
| [`scripts/`](scripts/) | Reproducible `8×B300` DDP training, depth extrapolation, and arena screening scripts |
| [`vendor/lorm/`](vendor/lorm/) | Vendored verbatim `ifm-ai/LoRM` core modules (`Transformer`, `FeedbackTransformer`, `NorMuon`) |
| [`legacy/`](legacy/) | Archived Phase-1 `loop_dedup` & `loop_muon` implementations and tests ([`docs/blog.md`](docs/blog.md)) |

---

## License

MIT — see [`LICENSE`](LICENSE).
