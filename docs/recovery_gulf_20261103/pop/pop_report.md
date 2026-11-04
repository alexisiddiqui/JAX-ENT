# pop: MoPrP population landscape under median-scaled BV

This pass is read-only: no repository edits and no optimisation fits through the framework. Scripts are in this folder and run from the repo root with `.venv/bin/python`. Every number is tagged with the script that produced it. Figures are given as MSAss / Filtered unless a row says otherwise.

**Inputs**
- Features: `_featurise_a101_k109`, 97 residues, kints in min⁻¹.
- Data: all 14 peptides × 15 timepoints (210 cells), with the row-normalised peptide map (trim 2).
- Uncertainties: `moprp.weights` read as 1/σ per cell. "Pooled σ" is the per-peptide RMS of those σ.
- Target populations: 97.1 / 2.4 / 0.5 (Folded / PUF1 / PUF2). Recovery = 100·(1−√JSD₂).

**Recovery anchors**
- Uniform prior: 73.3 (MSAss) / 80.8 (Filtered).
- Pure Folded (100/0/0): 87.9. A recovery of 88 or more therefore needs PUF1 > 0.

**Decoy states**
- **MSAss:** PUF3 (11 frames) and unfolded (15 frames).
- **Filtered:** has no PUF3 or unfolded frames. Its only non-target state is one "PUF2-like" frame, which I treat as Filtered's decoy. So "PUF3 + U ≤ c" means "PUF2-like ≤ c" for Filtered.

**Scripts**

| tag | script | purpose |
|---|---|---|
| `[p0]` | `p0_common.py` | shared code; reuses `gci2/c0_common.py` |
| `[p1]` | `p1_scaled_models.py` | T1 |
| `[p1b]` | `p1b_state_lnpf.py` | per-state lnPF |
| `[p1c]` | `p1c_shift_clamp.py` | the lnPF ≥ 0 shift variant |
| `[p2]` | `p2_landscape.py` | T2 grid, best points, reference points, local-optimum test |
| `[p2c]` | `p2_check.py` | checks |
| `[p2f]` | `p2_fig.py` | figures and profiles |
| `[p3]` | `p3_frames.py` | T3 |
| `[p4]` | `p4_summary.py` | T4 summary and sharpness |
| `[p5]` | `p5_tables.py` | the tables below |
| `[p6]` | `p6_residuals.py` | per-peptide residuals |

**Checks**
- **Rate mode:** my NumPy implementation is mean_f(k_i·e^(−lnPF_if)), then 1−exp(−k̄t), then the peptide map. It matches `BV_uptake_ForwardPass(frame_averaging_mode="rate").average_frames` (`jaxent/src/models/HDX/forward.py:100`) to ≤ 3e-17. frame_uptake matches to 3e-15 `[p1]`.
- **State-level tables:** they reproduce frame-level predictions to ≤ 1e-13. The vectorised recovery equals `calculate_recovery_percentage` to 2e-14 `[p2c]`.

**About median.pfact**
- It has 101 rows: 24 unmeasured (−1), **28 zeros** (censored lower bound) and **49 positive values**. So there are 77 non-negative entries, not 76.
- The median of the positive values is **4.120 ln**. All 49 positive residues are featurised.
- By region, 1–30 / 31–58 / 59–101 contain 18 / 11 / 20 of these residues.

## T1 — median-matched forward models (uniform weights) `[p1]`, `[p1c]`, `[p1b]` → `p1_t1.csv`, `p1_models.json`, `p1b.csv`

**Matching rule.** The ensemble median is the median, over the 49 residues with pfact > 0, of the uniform-mean per-frame lnPF. Before scaling it is 7.218 / 7.294 at BV(0.35, 2.0).

**(a) and (c) are the same model.** lnPF = bc·N_c + bh·N_h has no intercept, and a median scales exactly with s > 0. So scaling (bc, bh) by s and scaling lnPF by s give identical results, and the "root find" is closed-form: s = 4.120 / 7.218.

**Two models I added, labelled as additions:**
- **(b′) shift with lnPF ≥ 0.** Plain (b) gives 20% of Folded residue-frames and 59% of MSAss unfolded residue-frames PF < 1, i.e. k > k_int `[p1b]`. (b′) clamps that. Its δ comes from a real 1-D root find (brentq) so the median still matches.
- **(c′) affine-IQR.** This matches both the median and the IQR of lnPF to pfact, so the spread is matched as well as the median. It is not a least-squares fit.

**Model parameters:**
- **(a)=(c) scale:** s = 0.571 / 0.565, giving (bc, bh) = (0.200, 1.141) / (0.198, 1.130).
- **(b) shift:** δ = **−3.098 / −3.174**.
  - The earlier −2.46 / −2.79 was the median of the per-residue differences. This value is the difference of the medians, which is what was asked.
- **(b′) shift, lnPF ≥ 0:** δ = −3.108 / −3.174; about 22% of residue-frames are clamped.
- **(c′) affine-IQR:** slope 0.846 / 0.873 about the medians.
- **Medians over all 97 residues instead of the 49 matched ones:** s = 0.599 / 0.580 and δ = −2.755 / −2.989. Not carried forward.

The region columns are median(pfact − model lnPF), so a positive value means the model under-protects. The uptake columns are the uniform-prior uptake residual (mean of exp − pred, and the fraction of cells with exp > pred) under each averaging mode.

| model | 1–30 | 31–58 | 59–101 | all | frame_uptake: mean res, frac exp>pred | rate: mean res, frac exp>pred |
|---|---|---|---|---|---|---|
| unscaled 0.35/2.0 | −2.30 / −2.47 | −0.58 / −0.65 | −3.16 / −3.34 | −2.46 / −2.79 | +0.217 / +0.230, 0.90 / 0.95 | +0.086 / +0.209, 0.71 / 0.90 |
| (a)=(c) scale | +0.48 / +0.37 | +1.99 / +2.04 | +0.14 / +0.12 | +0.44 / +0.39 | +0.018 / +0.023, 0.53 / 0.54 | −0.035 / +0.011, 0.41 / 0.52 |
| (b) shift | +0.80 / +0.70 | +2.52 / +2.52 | −0.07 / −0.16 | +0.64 / +0.38 | −0.024 / −0.018, 0.40 / 0.42 | **−0.223** / −0.058, 0.10 / 0.34 |
| (b′) shift, lnPF ≥ 0 `[p1c]` | +0.71 / +0.69 | +2.51 / +2.52 | −0.09 / −0.16 | +0.46 / +0.36 | −0.005 / +0.001, 0.51 / 0.53 | −0.092 / −0.025, 0.31 / 0.45 |
| (c′) affine-IQR | +0.65 / +0.60 | +2.38 / +2.41 | +0.01 / −0.07 | +0.43 / +0.54 | −0.010 / −0.007, 0.43 / 0.45 | −0.151 / −0.038, 0.18 / 0.38 |

**Reading.**
- **Median matching removes the global offset.** The uptake bias falls from about +0.22 to about 0 under frame_uptake, and about half the cells are now under-predicted.
- **Region 31–58 stays under-protected by about 2 ln under every scaled model.** The other regions are within ±0.8. This is a regional, not global, model error.
- **Rate mode is very sensitive to the fastest frames.** Under (b) the MSAss unfolded state has median ln(k̄/k_int) = +1.7: it exchanges about 5.6 × faster than intrinsic `[p1b]`. So the uniform prior over-predicts (−0.22) under (b) + rate.

## T2 — state-level population landscape `[p2]`, `[p2f]`, `[p4]`

**Grid.** About 119k points per configuration (MSAss), made up of:
- the full simplex at 5% steps;
- a 1% grid with decoy mass ≤ 5% over the whole F range;
- a 1% grid with F ≥ 80%;
- a 0.25% grid near the target (F ≥ 90%, decoys 0);
- the exact target and uniform points.

Smooth scores are then polished with SLSQP on the simplex under the constraint. Outputs: `p2_grid_*.npz` (every point and score), `p2_best.csv`, `p2_ref.csv`, `p2_localopt.csv`.

**Figures:**
- `F1_face_{ens}.png`: the (F, PUF1, PUF2) face with decoys = 0. Note that "PUF2 = 0.5%, PUF3 = U = 0" makes the (F, PUF1) face one-dimensional (F = 99.5 − PUF1). So I plot the full PUF1 × PUF2 face, with that slice drawn dotted.
- `F2_profile_F.png`: F from 50 to 100% at the target PUF1:PUF2 ratio.
- `F3_slices.png`: PUF1 at PUF2 = 0.5%, and decoy mass mixed into the target.

### Fit at the exact target versus the uniform prior `[p2]` → `p2_ref.csv`

The last column is the percentage of the decoy-free grid that scores better than the target. That grid is dense near the target, so read this column as a rank, not a probability.

| ens | model | mode | χ²/cell tgt / uni | MSE×100 tgt / uni | ρ tgt / uni | max abs z tgt / uni | % decoy-free grid better than tgt (χ² / MSE / ρ) |
|---|---|---|---|---|---|---|---|
| MSAss | (a)=(c) scale | frame | 35.38 / 34.15 | 3.60 / 3.52 | 0.850 / 0.852 | 22.1 / 22.3 | 81.8 / 63.2 / 95.2 |
| MSAss | (a)=(c) scale | rate | 34.52 / 40.37 | 3.57 / 4.26 | 0.854 / 0.854 | 22.1 / 25.2 | 6.3 / 2.4 / 24.5 |
| MSAss | (b) shift | frame | 27.20 / 27.27 | 2.71 / 2.77 | 0.871 / 0.872 | 18.8 / 19.1 | 27.5 / 8.5 / 44.2 |
| MSAss | (b) shift | rate | 30.27 / 96.49 | 3.10 / 9.91 | 0.869 / 0.850 | 19.0 / 32.8 | 1.2 / 1.3 / 1.4 |
| MSAss | (b') shift,lnPF≥0 | frame | 27.01 / 26.79 | 2.71 / 2.73 | 0.874 / 0.875 | 18.8 / 19.1 | 38.7 / 13.2 / 69.2 |
| MSAss | (b') shift,lnPF≥0 | rate | 26.91 / 43.35 | 2.78 / 4.66 | 0.877 / 0.866 | 19.0 / 26.1 | 1.1 / 1.0 / 1.5 |
| MSAss | (c') affine-IQR | frame | 28.48 / 27.92 | 2.82 / 2.81 | 0.870 / 0.870 | 19.9 / 20.2 | 46.3 / 28.9 / 50.7 |
| MSAss | (c') affine-IQR | rate | 28.52 / 64.96 | 2.91 / 6.66 | 0.874 / 0.862 | 20.0 / 30.0 | 1.3 / 1.3 / 1.6 |
| MSAss | unscaled | frame | 85.00 / 76.31 | 9.35 / 8.52 | 0.857 / 0.858 | 25.3 / 24.5 | 99.2 / 99.2 / 37.4 |
| MSAss | unscaled | rate | 75.24 / 38.26 | 8.52 / 4.39 | 0.860 / 0.859 | 24.4 / 23.1 | 99.1 / 99.1 / 1.7 |
| Filt | (a)=(c) scale | frame | 35.57 / 35.06 | 3.62 / 3.59 | 0.849 / 0.851 | 22.4 / 22.3 | 79.5 / 55.2 / 77.9 |
| Filt | (a)=(c) scale | rate | 34.93 / 34.60 | 3.60 / 3.62 | 0.851 / 0.855 | 22.4 / 22.4 | 10.4 / 3.4 / 34.6 |
| Filt | (b) shift | frame | 27.45 / 27.22 | 2.75 / 2.75 | 0.871 / 0.872 | 19.2 / 19.1 | 17.1 / 4.8 / 24.2 |
| Filt | (b) shift | rate | 29.55 / 35.79 | 3.04 / 3.63 | 0.871 / 0.858 | 19.4 / 19.4 | 1.4 / 1.5 / 1.7 |
| Filt | (b') shift,lnPF≥0 | frame | 27.17 / 26.93 | 2.74 / 2.73 | 0.873 / 0.874 | 19.2 / 19.1 | 28.3 / 5.6 / 41.7 |
| Filt | (b') shift,lnPF≥0 | rate | 27.14 / 29.30 | 2.81 / 3.06 | 0.876 / 0.873 | 19.4 / 19.4 | 1.4 / 1.3 / 1.9 |
| Filt | (c') affine-IQR | frame | 28.28 / 27.92 | 2.81 / 2.79 | 0.869 / 0.870 | 20.1 / 20.0 | 37.2 / 12.6 / 52.3 |
| Filt | (c') affine-IQR | rate | 28.38 / 31.92 | 2.90 / 3.25 | 0.873 / 0.868 | 20.2 / 20.3 | 1.5 / 1.6 / 1.6 |
| Filt | unscaled | frame | 85.08 / 83.11 | 9.34 / 9.19 | 0.856 / 0.859 | 25.3 / 25.0 | 99.1 / 99.1 / 30.9 |
| Filt | unscaled | rate | 77.13 / 71.23 | 8.66 / 8.21 | 0.858 / 0.857 | 24.4 / 24.1 | 99.0 / 99.0 / 5.8 |


**Reading.**
- **Rate mode with (b), (b′) or (c′) separates the target from the prior.**
  - MSAss χ²/cell is 30.3 at the target versus 96.5 at the uniform prior under (b), and 26.9 versus 43.4 under (b′).
  - Only about 1–2% of the decoy-free grid beats the target on χ², MSE or Spearman.
- **frame_uptake mode barely separates them.** Target and uniform are within 1–4% on every score, and 17–82% of the grid beats the target on χ² (5–63% on MSE).
- **The unscaled model ranks the target in the worst 1% on χ² and MSE under both modes.**

**The target is never an exact local optimum** (`p2_localopt.csv`: ±0.25% mass moves between every pair of states). The best improving move:
- **rate with (b), (b′) or (c′):** always **PUF2 → Folded**. These models want PUF2 = 0.
- **rate with (a) scale:** Folded → PUF1 or PUF2. It wants more minor states.
- **frame_uptake:** mostly Folded → PUF2, PUF1, or the decoy.

### Best population per score `[p2]` → `p2_best.csv` (SLSQP-refined rows, `how=refined`)

Cells are F / PUF1 / PUF2 / decoys (%) → recovery.

**Constraint PUF3 + U = 0** (Filtered: PUF2-like = 0):

| ens | model | mode | chi2_pool | mse | wr2 | spearman | maxz |
|---|---|---|---|---|---|---|---|
| MSAss | (a)=(c) scale | frame | 0/99.3/0.7/0.0 → **4** | 0/100.0/0.0/0.0 → **4** | 0/95.1/4.9/0.0 → **5** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (a)=(c) scale | rate | 77/22.8/0.0/0.0 → **72** | 87/12.8/0.0/0.0 → **82** | 0/100.0/0.0/0.0 → **4** | 39/61.0/0.0/0.0 → **42** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (b) shift | frame | 39/60.6/0.0/0.0 → **42** | 61/38.9/0.0/0.0 → **58** | 7/93.4/0.0/0.0 → **14** | 8/92.0/0.0/0.0 → **15** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (b) shift | rate | 99/0.8/0.0/0.0 → **92** | 99/0.6/0.0/0.0 → **92** | 99/0.7/0.0/0.0 → **92** | 98/1.5/0.0/0.0 → **94** | 100/0.0/0.0/0.0 → **88** |
| MSAss | (b') shift,lnPF≥0 | frame | 27/72.9/0.0/0.0 → **33** | 57/43.4/0.0/0.0 → **55** | 0/100.0/0.0/0.0 → **4** | 4/96.0/0.0/0.0 → **11** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (b') shift,lnPF≥0 | rate | 98/2.2/0.0/0.0 → **95** | 98/1.5/0.0/0.0 → **94** | 97/3.1/0.0/0.0 → **95** | 96/4.0/0.0/0.0 → **94** | 100/0.0/0.0/0.0 → **88** |
| MSAss | (c') affine-IQR | frame | 12/88.4/0.0/0.0 → **19** | 36/63.6/0.0/0.0 → **40** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (c') affine-IQR | rate | 98/2.0/0.0/0.0 → **95** | 98/1.6/0.0/0.0 → **94** | 98/2.1/0.0/0.0 → **95** | 94/5.5/0.0/0.0 → **91** | 100/0.0/0.0/0.0 → **88** |
| MSAss | unscaled | frame | 0/28.2/71.8/0.0 → **4** | 0/30.0/70.0/0.0 → **4** | 0/19.2/80.8/0.0 → **4** | 0/100.0/0.0/0.0 → **4** | 0/0.0/100.0/0.0 → **1** |
| MSAss | unscaled | rate | 0/76.0/24.0/0.0 → **5** | 0/86.1/13.9/0.0 → **5** | 0/0.0/100.0/0.0 → **1** | 93/7.2/0.0/0.0 → **89** | 0/0.0/100.0/0.0 → **1** |
| Filt | (a)=(c) scale | frame | 0/98.5/1.5/0.0 → **5** | 0/100.0/0.0/0.0 → **4** | 0/96.6/3.4/0.0 → **5** | 0/100.0/0.0/0.0 → **4** | 0/19.0/81.0/0.0 → **4** |
| Filt | (a)=(c) scale | rate | 63/37.5/0.0/0.0 → **59** | 79/20.5/0.0/0.0 → **74** | 0/100.0/0.0/0.0 → **4** | 9/91.0/0.0/0.0 → **16** | 0/100.0/0.0/0.0 → **4** |
| Filt | (b) shift | frame | 38/61.6/0.0/0.0 → **41** | 67/32.8/0.0/0.0 → **63** | 0/100.0/0.0/0.0 → **4** | 2/98.0/0.0/0.0 → **8** | 0/28.0/72.0/0.0 → **4** |
| Filt | (b) shift | rate | 99/1.3/0.0/0.0 → **94** | 99/1.2/0.0/0.0 → **94** | 99/1.5/0.0/0.0 → **94** | 97/3.2/0.0/0.0 → **94** | 100/0.0/0.0/0.0 → **88** |
| Filt | (b') shift,lnPF≥0 | frame | 32/67.8/0.0/0.0 → **37** | 67/33.1/0.0/0.0 → **63** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** | 0/0.0/100.0/0.0 → **1** |
| Filt | (b') shift,lnPF≥0 | rate | 96/3.5/0.0/0.0 → **94** | 97/2.6/0.0/0.0 → **95** | 95/5.0/0.0/0.0 → **92** | 94/6.0/0.0/0.0 → **91** | 100/0.0/0.0/0.0 → **88** |
| Filt | (c') affine-IQR | frame | 15/84.7/0.0/0.0 → **23** | 47/52.8/0.0/0.0 → **48** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** | 0/4.0/96.0/0.0 → **3** |
| Filt | (c') affine-IQR | rate | 97/3.1/0.0/0.0 → **95** | 97/2.6/0.0/0.0 → **95** | 96/3.7/0.0/0.0 → **94** | 94/6.5/0.0/0.0 → **90** | 100/0.0/0.0/0.0 → **88** |
| Filt | unscaled | frame | 0/25.9/74.1/0.0 → **4** | 0/30.1/69.9/0.0 → **4** | 0/18.3/81.7/0.0 → **4** | 0/100.0/0.0/0.0 → **4** | 0/0.0/100.0/0.0 → **1** |
| Filt | unscaled | rate | 0/86.6/13.4/0.0 → **5** | 0/88.8/11.2/0.0 → **5** | 0/68.2/31.8/0.0 → **5** | 72/28.0/0.0/0.0 → **67** | 0/0.0/100.0/0.0 → **1** |


**Unconstrained:**

| ens | model | mode | chi2_pool | mse | wr2 | spearman | maxz |
|---|---|---|---|---|---|---|---|
| MSAss | (a)=(c) scale | frame | 0/96.8/0.0/3.2 → **4** | 0/96.5/0.0/3.5 → **4** | 0/95.1/4.9/0.0 → **5** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (a)=(c) scale | rate | 77/22.8/0.0/0.0 → **72** | 87/12.8/0.0/0.0 → **82** | 0/100.0/0.0/0.0 → **4** | 1/98.0/0.0/1.0 → **6** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (b) shift | frame | 39/60.6/0.0/0.0 → **42** | 61/38.9/0.0/0.0 → **58** | 7/93.4/0.0/0.0 → **14** | 8/92.0/0.0/0.0 → **15** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (b) shift | rate | 100/0.4/0.0/0.0 → **91** | 100/0.3/0.0/0.0 → **91** | 99/0.5/0.0/0.0 → **91** | 98/1.5/0.0/0.0 → **94** | 100/0.0/0.0/0.0 → **88** |
| MSAss | (b') shift,lnPF≥0 | frame | 27/72.9/0.0/0.0 → **33** | 57/43.4/0.0/0.0 → **55** | 0/100.0/0.0/0.0 → **4** | 4/96.0/0.0/0.0 → **11** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (b') shift,lnPF≥0 | rate | 98/2.2/0.0/0.0 → **95** | 98/1.5/0.0/0.0 → **94** | 97/3.1/0.0/0.0 → **95** | 91/8.0/0.0/1.0 → **86** | 100/0.0/0.0/0.0 → **88** |
| MSAss | (c') affine-IQR | frame | 12/88.4/0.0/0.0 → **19** | 36/63.6/0.0/0.0 → **40** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** | 0/100.0/0.0/0.0 → **4** |
| MSAss | (c') affine-IQR | rate | 98/1.9/0.0/0.0 → **95** | 99/1.4/0.0/0.0 → **94** | 98/2.1/0.0/0.0 → **95** | 79/20.0/0.0/1.0 → **73** | 100/0.0/0.0/0.0 → **88** |
| MSAss | unscaled | frame | 0/37.8/0.0/62.2 → **3** | 26/5.6/0.0/68.2 → **29** | 0/39.9/0.0/60.1 → **3** | 0/100.0/0.0/0.0 → **4** | 0/20.0/0.0/80.0 → **3** |
| MSAss | unscaled | rate | 0/95.9/0.0/4.1 → **4** | 0/94.1/0.0/5.9 → **4** | 0/95.2/0.0/4.8 → **4** | 81/18.0/0.0/1.0 → **75** | 0/0.0/95.0/5.0 → **1** |
| Filt | (a)=(c) scale | frame | 0/0.0/0.0/100.0 → **0** | 0/0.0/0.0/100.0 → **0** | 0/0.0/0.0/100.0 → **0** | 0/0.0/0.0/100.0 → **0** | 0/0.0/0.0/100.0 → **0** |
| Filt | (a)=(c) scale | rate | 0/5.8/0.0/94.2 → **2** | 0/3.3/0.0/96.7 → **1** | 0/0.0/1.0/99.0 → **0** | 0/10.0/0.0/90.0 → **2** | 0/0.0/0.0/100.0 → **0** |
| Filt | (b) shift | frame | 2/0.0/0.0/98.5 → **3** | 26/0.0/0.0/73.7 → **27** | 0/0.0/0.0/100.0 → **0** | 5/15.0/0.0/80.0 → **10** | 0/0.0/65.0/35.0 → **1** |
| Filt | (b) shift | rate | 0/1.0/0.0/99.0 → **1** | 0/0.8/0.0/99.2 → **1** | 0/1.1/0.0/98.9 → **1** | 15/5.0/0.0/80.0 → **19** | 0/0.0/0.0/100.0 → **0** |
| Filt | (b') shift,lnPF≥0 | frame | 0/0.0/0.0/100.0 → **0** | 28/0.0/0.0/71.9 → **28** | 0/0.0/0.0/100.0 → **0** | 0/15.0/0.0/85.0 → **3** | 0/0.0/65.0/35.0 → **1** |
| Filt | (b') shift,lnPF≥0 | rate | 0/1.6/0.0/98.4 → **1** | 0/0.9/0.0/99.2 → **1** | 0/2.1/0.0/97.9 → **1** | 0/5.0/0.0/95.0 → **2** | 0/0.0/0.0/100.0 → **0** |
| Filt | (c') affine-IQR | frame | 0/0.0/0.0/100.0 → **0** | 10/0.0/0.0/89.5 → **13** | 0/0.0/0.0/100.0 → **0** | 0/0.0/0.0/100.0 → **0** | 0/0.0/0.0/100.0 → **0** |
| Filt | (c') affine-IQR | rate | 0/1.9/0.0/98.1 → **1** | 0/1.4/0.0/98.6 → **1** | 0/2.0/0.0/98.0 → **1** | 0/5.0/0.0/95.0 → **2** | 0/0.0/0.0/100.0 → **0** |
| Filt | unscaled | frame | 0/0.0/45.3/54.7 → **1** | 0/0.0/37.5/62.5 → **1** | 0/0.0/34.0/66.0 → **1** | 0/0.0/0.0/100.0 → **0** | 0/0.0/30.0/70.0 → **1** |
| Filt | unscaled | rate | 0/0.0/13.2/86.8 → **1** | 0/0.0/12.0/88.0 → **1** | 0/0.0/12.3/87.7 → **1** | 0/5.0/0.0/95.0 → **2** | 0/0.0/30.0/70.0 → **1** |


The decoy ≤ 0.01 and ≤ 0.05 versions are in `p5_tables.log` and `p2_best.csv`.

**Effect of the constraint.**
- **MSAss:** it rarely binds for the scaled models in rate mode. Their unconstrained optima already have about 0% decoys.
- **Filtered:** it binds hard. Unconstrained, every score in both modes and every model puts 35–100% (mostly ≥ 90%) of the mass on the single PUF2-like frame, giving recovery 0–28.
- `F3_slices.png` bottom row shows that adding the Filtered decoy improves every score monotonically. In MSAss, adding decoy mass worsens every rate-mode MSE, w·r² and χ² for the scaled models. Only Spearman has a shallow optimum, at 0.5–1% decoy.

### Sharpness of the optimum (decoy-free grid) `[p4]` → `p4_sharp.csv`

The near-optimal set is defined per score:
- **MSE, mean(w r²) and χ²:** treat each score as a χ² with σ rescaled so that χ²/ν = 1 at the minimum, and keep points with Δχ² ≤ 2.
- **Spearman:** Δρ ≤ 0.002.
- **max|z|:** within 1%. The set is the whole simplex, so max|z| is uninformative. Its worst cell is always 83–90 at 12.5–24 h; see `[p6]`.

| ens | model | mode | chi2_pool | mse | spearman |
|---|---|---|---|---|---|
| MSAss | (c') affine-IQR | rate | F 96–99, P1 0.8–4.5, rec 92–95 | F 96–100, P1 0.5–4.0, rec 91–95 | F 89–98, P1 1.5–11.0, rec 84–98 |
| MSAss | unscaled | rate | F 0–42, P1 0.0–97.0, rec 1–43 | F 0–56, P1 0.0–100.0, rec 4–53 | F 70–99, P1 1.2–30.0, rec 65–100 |
| MSAss | (a)=(c) scale | rate | F 47–92, P1 7.5–53.0, rec 48–89 | F 65–98, P1 2.0–35.0, rec 61–100 | F 0–61, P1 38.0–100.0, rec 4–59 |
| MSAss | (b') shift,lnPF≥0 | rate | F 95–100, P1 0.2–5.2, rec 91–98 | F 96–100, P1 0.0–4.2, rec 88–97 | F 85–99, P1 0.5–15.0, rec 79–100 |
| MSAss | (b) shift | rate | F 98–100, P1 0.2–1.5, rec 90–94 | F 98–100, P1 0.2–1.5, rec 90–94 | F 96–100, P1 0.5–4.0, rec 91–95 |
| Filt | (c') affine-IQR | rate | F 93–99, P1 1.0–7.0, rec 89–95 | F 93–100, P1 0.5–6.8, rec 90–95 | F 74–98, P1 1.2–26.0, rec 69–98 |
| Filt | unscaled | rate | F 0–19, P1 62.0–97.0, rec 5–27 | F 0–29, P1 54.0–98.0, rec 5–35 | F 0–87, P1 13.0–100.0, rec 4–82 |
| Filt | (a)=(c) scale | rate | F 10–87, P1 13.0–90.0, rec 17–82 | F 44–96, P1 3.2–56.0, rec 46–97 | F 0–26, P1 74.0–100.0, rec 4–32 |
| Filt | (b') shift,lnPF≥0 | rate | F 92–99, P1 1.0–8.5, rec 87–98 | F 93–100, P1 0.2–7.2, rec 89–98 | F 73–98, P1 2.0–27.0, rec 68–100 |
| Filt | (b) shift | rate | F 97–100, P1 0.5–3.0, rec 91–95 | F 97–100, P1 0.2–3.2, rec 90–95 | F 90–99, P1 1.0–10.0, rec 85–95 |

(frame_uptake rows omitted here; all of them span F ≈ 0–92% and recovery ≈ 4–89, see `p4_sharp.csv`.)

**1-D profile along F at the target PUF1:PUF2 ratio** (`p2_profile_optima.csv`, `F2_profile_F.png`) `[p2f]`:
- **rate with (a):** MSE is minimised at F = 96.5% (recovery 98.5) / 96.0% (97.4); χ² at F = 93% / 91%.
- **rate with (b), (b′) or (c′):** minimised at F = 99–99.5% (recovery 92–94).
- **frame_uptake with (a):** minimised at the F = 50% boundary for every score.
- **frame_uptake with (b):** MSE minimised at F = 85.5% / 96%; χ² at 70% / 75.5%.

**PUF1 slice at PUF2 = 0.5%** `[p2f]`:
- **rate with (b) or (c′):** every score is minimised at PUF1 = 0 (recovery 89).
- **rate with (a):** MSE is minimised at PUF1 = 10.75% (MSAss) / 18% (Filtered).
- **frame_uptake:** every score is minimised at the 30% edge.

## T3 — frame-level fits with decoy mass fixed `[p3]` → `p3_t3.csv`, `p3_weights.npz`

**Setup.**
- Decoy frames hold the fixed mass c uniformly. The remaining frames are free with total 1 − c.
- **NNLS:** L-BFGS-B on x ≥ 0, with w = (1 − c)·x/Σx.
  - frame_uptake rows are cross-checked against `scipy.optimize.nnls`. L-BFGS is equal or better in all 54 cases, by at most 1.9e-3 in objective.
  - rate mode is non-convex, so NNLS is started from both uniform and the S = 100 MaxEnt solution, and the better result is kept.
- **MaxEnt:** objective χ²/cell (pooled σ) + (1/S)·KL(w‖uniform), using a softmax parametrisation and continuation in S from 0.03 to 100.
- Convergence: 3 of 360 runs ended with ABNORMAL line-search termination at a flat optimum; none is a run quoted below.
- ESS is Kish ESS / N. An ESS of 0.2–0.6% means 1–3 frames.

**Constraint PUF3 + U = 0:**

| ens | model | mode | NNLS chi2_pool | NNLS mse | NNLS wr2 | ME χ² S0.03 | ME χ² S0.3 | ME χ² S3 | ME χ² S100 |
|---|---|---|---|---|---|---|---|---|---|
| MSAss | (a)=(c) scale | frame | 70/30.3/0.0/0.0 → **65** (ESS 0.3%, χ² 24.3) | 73/26.7/0.0/0.0 → **68** (ESS 0.3%, χ² 24.4) | 46/53.8/0.0/0.0 → **47** (ESS 0.6%, χ² 24.9) | 80/16.9/2.8/0.0 → **76** (ESS 93.5%, χ² 34.2) | 61/32.0/6.9/0.0 → **60** (ESS 22.2%, χ² 30.2) | 64/34.1/2.1/0.0 → **62** (ESS 0.5%, χ² 24.5) | 70/30.4/0.0/0.0 → **65** (ESS 0.3%, χ² 24.3) |
| MSAss | (a)=(c) scale | rate | 94/6.4/0.0/0.0 → **90** (ESS 0.2%, χ² 24.0) | 94/5.7/0.0/0.0 → **91** (ESS 0.3%, χ² 24.3) | 89/11.1/0.0/0.0 → **84** (ESS 0.4%, χ² 24.4) | 83/15.8/1.2/0.0 → **79** (ESS 92.9%, χ² 33.6) | 79/21.1/0.1/0.0 → **74** (ESS 14.0%, χ² 29.7) | 88/11.8/0.0/0.0 → **83** (ESS 0.3%, χ² 24.1) | 94/6.5/0.0/0.0 → **90** (ESS 0.2%, χ² 24.0) |
| MSAss | (b) shift | frame | 64/35.8/0.0/0.0 → **61** (ESS 0.7%, χ² 19.3) | 61/39.4/0.0/0.0 → **58** (ESS 1.0%, χ² 19.5) | 21/78.5/0.0/0.0 → **28** (ESS 0.6%, χ² 19.8) | 82/16.1/2.1/0.0 → **78** (ESS 93.9%, χ² 26.5) | 70/28.4/1.3/0.0 → **67** (ESS 29.5%, χ² 23.5) | 59/40.6/0.0/0.0 → **57** (ESS 1.5%, χ² 19.5) | 64/35.8/0.0/0.0 → **61** (ESS 0.7%, χ² 19.3) |
| MSAss | (b) shift | rate | 98/2.3/0.0/0.0 → **95** (ESS 0.4%, χ² 17.9) | 100/0.1/0.0/0.0 → **89** (ESS 0.6%, χ² 18.5) | 81/19.1/0.0/0.0 → **75** (ESS 0.5%, χ² 17.9) | 88/11.8/0.0/0.0 → **83** (ESS 87.0%, χ² 25.6) | 89/10.8/0.0/0.0 → **84** (ESS 13.5%, χ² 21.7) | 90/10.1/0.0/0.0 → **85** (ESS 0.7%, χ² 18.1) | 96/3.6/0.0/0.0 → **94** (ESS 0.4%, χ² 17.9) |
| MSAss | (c') affine-IQR | frame | 64/35.9/0.0/0.0 → **61** (ESS 0.4%, χ² 19.0) | 59/40.6/0.0/0.0 → **57** (ESS 0.7%, χ² 19.4) | 27/72.6/0.0/0.0 → **33** (ESS 0.8%, χ² 19.4) | 81/16.5/2.4/0.0 → **77** (ESS 93.8%, χ² 27.5) | 67/30.8/2.5/0.0 → **64** (ESS 24.0%, χ² 24.0) | 60/39.7/0.0/0.0 → **58** (ESS 0.8%, χ² 19.1) | 64/36.4/0.0/0.0 → **60** (ESS 0.4%, χ² 19.0) |
| MSAss | (c') affine-IQR | rate | 99/0.7/0.0/0.0 → **92** (ESS 0.3%, χ² 17.9) | 100/0.4/0.0/0.0 → **91** (ESS 0.4%, χ² 18.6) | 99/1.4/0.0/0.0 → **94** (ESS 0.4%, χ² 18.5) | 87/13.3/0.1/0.0 → **82** (ESS 88.9%, χ² 26.5) | 87/12.7/0.0/0.0 → **82** (ESS 11.2%, χ² 22.5) | 95/4.6/0.0/0.0 → **93** (ESS 0.4%, χ² 18.1) | 99/0.7/0.0/0.0 → **92** (ESS 0.3%, χ² 17.9) |
| Filt | (a)=(c) scale | frame | 44/56.1/0.0/0.0 → **45** (ESS 0.6%, χ² 25.5) | 58/41.9/0.0/0.0 → **56** (ESS 0.4%, χ² 26.0) | 24/76.1/0.0/0.0 → **30** (ESS 0.6%, χ² 25.7) | 83/14.4/2.2/0.0 → **79** (ESS 98.9%, χ² 34.8) | 67/27.4/5.1/0.0 → **65** (ESS 23.9%, χ² 31.4) | 45/54.0/0.5/0.0 → **47** (ESS 0.8%, χ² 25.6) | 44/56.2/0.0/0.0 → **45** (ESS 0.6%, χ² 25.5) |
| Filt | (a)=(c) scale | rate | 92/7.9/0.0/0.0 → **88** (ESS 0.2%, χ² 25.3) | 92/7.6/0.0/0.0 → **88** (ESS 0.2%, χ² 25.8) | 76/24.4/0.0/0.0 → **70** (ESS 0.6%, χ² 25.8) | 85/14.0/1.1/0.0 → **81** (ESS 98.5%, χ² 34.2) | 79/20.6/0.1/0.0 → **74** (ESS 20.5%, χ² 30.7) | 85/15.2/0.0/0.0 → **79** (ESS 0.6%, χ² 25.5) | 92/8.0/0.0/0.0 → **88** (ESS 0.2%, χ² 25.3) |
| Filt | (b) shift | frame | 54/46.5/0.0/0.0 → **53** (ESS 0.4%, χ² 18.7) | 66/34.2/0.0/0.0 → **62** (ESS 0.5%, χ² 18.8) | 51/48.9/0.0/0.0 → **51** (ESS 0.6%, χ² 18.9) | 84/13.9/1.7/0.0 → **80** (ESS 99.1%, χ² 27.0) | 75/24.6/0.6/0.0 → **70** (ESS 22.0%, χ² 24.0) | 55/44.5/0.0/0.0 → **54** (ESS 0.7%, χ² 18.8) | 54/46.4/0.0/0.0 → **53** (ESS 0.4%, χ² 18.7) |
| Filt | (b) shift | rate | 100/0.4/0.0/0.0 → **91** (ESS 0.3%, χ² 18.2) | 100/0.3/0.0/0.0 → **90** (ESS 0.3%, χ² 18.6) | 100/0.4/0.0/0.0 → **91** (ESS 0.3%, χ² 18.3) | 89/10.8/0.2/0.0 → **85** (ESS 94.4%, χ² 26.6) | 91/8.9/0.0/0.0 → **87** (ESS 18.4%, χ² 22.7) | 99/0.7/0.0/0.0 → **92** (ESS 0.5%, χ² 18.4) | 100/0.3/0.0/0.0 → **91** (ESS 0.3%, χ² 18.2) |
| Filt | (c') affine-IQR | frame | 50/49.9/0.0/0.0 → **50** (ESS 0.4%, χ² 18.6) | 61/39.1/0.0/0.0 → **58** (ESS 0.4%, χ² 18.7) | 47/53.3/0.0/0.0 → **48** (ESS 0.5%, χ² 18.8) | 84/14.1/1.9/0.0 → **80** (ESS 99.0%, χ² 27.7) | 73/26.2/1.2/0.0 → **69** (ESS 20.6%, χ² 24.4) | 53/47.5/0.0/0.0 → **52** (ESS 0.5%, χ² 18.7) | 50/49.9/0.0/0.0 → **50** (ESS 0.4%, χ² 18.6) |
| Filt | (c') affine-IQR | rate | 99/0.9/0.0/0.0 → **93** (ESS 0.3%, χ² 18.3) | 99/0.8/0.0/0.0 → **92** (ESS 0.3%, χ² 18.4) | 99/1.1/0.0/0.0 → **93** (ESS 0.3%, χ² 18.5) | 88/12.0/0.2/0.0 → **83** (ESS 95.6%, χ² 26.9) | 89/11.0/0.0/0.0 → **84** (ESS 18.7%, χ² 23.1) | 98/1.7/0.0/0.0 → **95** (ESS 0.4%, χ² 18.6) | 99/0.9/0.0/0.0 → **93** (ESS 0.3%, χ² 18.3) |


**Constraint PUF3 + U = 1%:**

| ens | model | mode | NNLS chi2_pool | NNLS mse | NNLS wr2 | ME χ² S0.03 | ME χ² S0.3 | ME χ² S3 | ME χ² S100 |
|---|---|---|---|---|---|---|---|---|---|
| MSAss | (a)=(c) scale | frame | 70/28.7/0.0/1.0 → **66** (ESS 0.3%, χ² 24.4) | 73/26.0/0.0/1.0 → **68** (ESS 0.3%, χ² 24.5) | 45/54.1/0.0/1.0 → **46** (ESS 0.6%, χ² 25.0) | 80/16.7/2.7/1.0 → **75** (ESS 95.3%, χ² 34.1) | 61/31.4/6.6/1.0 → **60** (ESS 22.9%, χ² 30.2) | 64/33.4/1.6/1.0 → **62** (ESS 0.5%, χ² 24.5) | 70/28.8/0.0/1.0 → **66** (ESS 0.3%, χ² 24.4) |
| MSAss | (a)=(c) scale | rate | 97/1.8/0.0/1.0 → **91** (ESS 0.2%, χ² 25.1) | 93/5.6/0.0/1.0 → **89** (ESS 0.2%, χ² 25.2) | 78/20.8/0.0/1.0 → **72** (ESS 0.3%, χ² 25.5) | 83/15.3/1.1/1.0 → **78** (ESS 94.8%, χ² 34.6) | 80/19.0/0.1/1.0 → **74** (ESS 14.9%, χ² 31.0) | 92/6.9/0.0/1.0 → **87** (ESS 0.2%, χ² 25.2) | 97/1.8/0.0/1.0 → **91** (ESS 0.2%, χ² 25.1) |
| MSAss | (b) shift | frame | 64/35.4/0.0/1.0 → **60** (ESS 0.7%, χ² 19.4) | 60/39.4/0.0/1.0 → **57** (ESS 1.0%, χ² 19.6) | 19/79.8/0.0/1.0 → **26** (ESS 0.6%, χ² 19.9) | 81/15.9/2.0/1.0 → **77** (ESS 95.7%, χ² 26.6) | 70/27.7/1.1/1.0 → **67** (ESS 30.3%, χ² 23.6) | 59/39.9/0.0/1.0 → **57** (ESS 1.5%, χ² 19.6) | 63/35.6/0.0/1.0 → **60** (ESS 0.7%, χ² 19.4) |
| MSAss | (b) shift | rate | 99/0.0/0.0/1.0 → **86** (ESS 0.4%, χ² 41.8) | 99/0.0/0.0/1.0 → **86** (ESS 0.4%, χ² 41.9) | 28/70.9/0.0/1.0 → **33** (ESS 0.3%, χ² 42.3) | 87/11.4/0.1/1.0 → **82** (ESS 90.6%, χ² 48.7) | 90/9.1/0.0/1.0 → **85** (ESS 20.9%, χ² 45.5) | 93/5.7/0.0/1.0 → **89** (ESS 0.9%, χ² 42.1) | 99/0.0/0.0/1.0 → **86** (ESS 0.4%, χ² 41.8) |
| MSAss | (c') affine-IQR | frame | 63/35.8/0.0/1.0 → **60** (ESS 0.4%, χ² 19.1) | 59/39.8/0.0/1.0 → **57** (ESS 0.7%, χ² 19.5) | 26/72.9/0.0/1.0 → **32** (ESS 0.7%, χ² 19.5) | 80/16.3/2.3/1.0 → **76** (ESS 95.5%, χ² 27.5) | 67/30.1/2.2/1.0 → **64** (ESS 24.7%, χ² 24.0) | 60/39.0/0.0/1.0 → **58** (ESS 0.8%, χ² 19.2) | 63/36.2/0.0/1.0 → **60** (ESS 0.4%, χ² 19.1) |
| MSAss | (c') affine-IQR | rate | 99/0.0/0.0/1.0 → **86** (ESS 0.2%, χ² 26.6) | 99/0.0/0.0/1.0 → **86** (ESS 0.3%, χ² 26.8) | 99/0.0/0.0/1.0 → **86** (ESS 0.2%, χ² 26.6) | 86/12.8/0.2/1.0 → **81** (ESS 91.6%, χ² 35.2) | 90/8.9/0.0/1.0 → **85** (ESS 13.2%, χ² 31.5) | 98/1.4/0.0/1.0 → **91** (ESS 0.3%, χ² 26.8) | 99/0.0/0.0/1.0 → **86** (ESS 0.2%, χ² 26.6) |
| Filt | (a)=(c) scale | frame | 43/55.9/0.0/1.0 → **45** (ESS 0.6%, χ² 25.5) | 57/41.5/0.0/1.0 → **56** (ESS 0.4%, χ² 26.0) | 23/75.9/0.0/1.0 → **29** (ESS 0.6%, χ² 25.7) | 83/14.2/2.2/1.0 → **78** (ESS 96.1%, χ² 34.7) | 67/27.1/5.1/1.0 → **64** (ESS 24.3%, χ² 31.3) | 45/53.8/0.5/1.0 → **46** (ESS 0.9%, χ² 25.6) | 43/55.9/0.0/1.0 → **45** (ESS 0.6%, χ² 25.5) |
| Filt | (a)=(c) scale | rate | 91/7.8/0.0/1.0 → **86** (ESS 0.2%, χ² 25.3) | 92/7.4/0.0/1.0 → **87** (ESS 0.2%, χ² 25.8) | 75/23.9/0.0/1.0 → **70** (ESS 0.6%, χ² 25.8) | 84/13.8/1.1/1.0 → **80** (ESS 95.7%, χ² 34.0) | 79/20.3/0.1/1.0 → **73** (ESS 21.4%, χ² 30.6) | 84/14.9/0.0/1.0 → **78** (ESS 0.6%, χ² 25.6) | 91/8.0/0.0/1.0 → **86** (ESS 0.2%, χ² 25.3) |
| Filt | (b) shift | frame | 53/46.1/0.0/1.0 → **52** (ESS 0.5%, χ² 18.7) | 65/33.9/0.0/1.0 → **61** (ESS 0.5%, χ² 18.8) | 50/48.7/0.0/1.0 → **50** (ESS 0.6%, χ² 18.9) | 84/13.8/1.7/1.0 → **79** (ESS 96.3%, χ² 26.9) | 74/24.3/0.6/1.0 → **70** (ESS 22.6%, χ² 24.0) | 55/44.2/0.0/1.0 → **53** (ESS 0.7%, χ² 18.8) | 53/46.1/0.0/1.0 → **52** (ESS 0.5%, χ² 18.7) |
| Filt | (b) shift | rate | 99/0.4/0.0/1.0 → **88** (ESS 0.3%, χ² 18.2) | 99/0.3/0.0/1.0 → **88** (ESS 0.3%, χ² 18.2) | 99/0.4/0.0/1.0 → **88** (ESS 0.3%, χ² 18.3) | 88/10.6/0.2/1.0 → **83** (ESS 92.2%, χ² 26.3) | 90/8.8/0.0/1.0 → **85** (ESS 19.9%, χ² 22.7) | 98/0.7/0.0/1.0 → **90** (ESS 0.5%, χ² 18.4) | 99/0.3/0.0/1.0 → **88** (ESS 0.3%, χ² 18.2) |
| Filt | (c') affine-IQR | frame | 49/49.6/0.0/1.0 → **49** (ESS 0.4%, χ² 18.6) | 60/38.9/0.0/1.0 → **58** (ESS 0.4%, χ² 18.7) | 46/53.1/0.0/1.0 → **47** (ESS 0.5%, χ² 18.8) | 83/14.0/1.8/1.0 → **79** (ESS 96.2%, χ² 27.6) | 72/25.9/1.2/1.0 → **68** (ESS 21.1%, χ² 24.4) | 52/47.2/0.0/1.0 → **51** (ESS 0.5%, χ² 18.7) | 49/49.6/0.0/1.0 → **49** (ESS 0.4%, χ² 18.6) |
| Filt | (c') affine-IQR | rate | 98/0.9/0.0/1.0 → **90** (ESS 0.3%, χ² 18.3) | 98/0.8/0.0/1.0 → **90** (ESS 0.3%, χ² 18.4) | 98/1.1/0.0/1.0 → **90** (ESS 0.3%, χ² 18.5) | 87/11.8/0.2/1.0 → **82** (ESS 93.3%, χ² 26.7) | 88/10.8/0.0/1.0 → **83** (ESS 20.0%, χ² 23.1) | 97/1.6/0.0/1.0 → **91** (ESS 0.4%, χ² 18.6) | 98/0.8/0.0/1.0 → **90** (ESS 0.3%, χ² 18.3) |


**Unconstrained** (compare the previous pass: NNLS on the unscaled frame_uptake model ran to decoys, with an NNLS recovery ceiling of 24 / 1):

| ens | model | mode | NNLS chi2_pool | NNLS mse | NNLS wr2 | ME χ² S0.03 | ME χ² S0.3 | ME χ² S3 | ME χ² S100 |
|---|---|---|---|---|---|---|---|---|---|
| MSAss | (a)=(c) scale | frame | 74/19.9/0.0/6.5 → **68** (ESS 0.3%, χ² 24.0) | 73/6.3/0.0/20.9 → **64** (ESS 0.4%, χ² 24.3) | 45/51.5/0.0/3.1 → **46** (ESS 0.6%, χ² 24.9) | 76/15.6/2.4/6.1 → **71** (ESS 98.3%, χ² 33.6) | 60/26.2/3.9/9.6 → **58** (ESS 25.4%, χ² 29.7) | 65/24.6/0.1/10.8 → **61** (ESS 0.5%, χ² 24.1) | 73/20.2/0.0/6.5 → **68** (ESS 0.3%, χ² 24.0) |
| MSAss | (a)=(c) scale | rate | 96/3.4/0.0/0.2 → **93** (ESS 0.2%, χ² 23.9) | 96/4.1/0.0/0.3 → **92** (ESS 0.2%, χ² 24.0) | 89/10.9/0.0/0.0 → **84** (ESS 0.4%, χ² 24.3) | 83/15.4/1.1/0.8 → **78** (ESS 94.1%, χ² 33.7) | 79/20.6/0.1/0.2 → **74** (ESS 14.5%, χ² 29.7) | 90/9.6/0.0/0.2 → **85** (ESS 0.3%, χ² 24.0) | 96/3.5/0.0/0.2 → **93** (ESS 0.2%, χ² 23.9) |
| MSAss | (b) shift | frame | 64/35.3/0.0/0.7 → **61** (ESS 0.7%, χ² 19.3) | 59/26.8/0.0/13.9 → **56** (ESS 1.1%, χ² 19.8) | 21/78.5/0.0/0.0 → **28** (ESS 0.6%, χ² 19.8) | 79/15.2/1.8/4.0 → **74** (ESS 98.5%, χ² 26.7) | 70/25.9/0.9/3.6 → **66** (ESS 31.8%, χ² 23.5) | 59/38.3/0.0/2.9 → **56** (ESS 1.6%, χ² 19.5) | 64/35.6/0.0/0.9 → **60** (ESS 0.8%, χ² 19.3) |
| MSAss | (b) shift | rate | 89/11.4/0.0/0.0 → **83** (ESS 0.4%, χ² 17.0) | 100/0.1/0.0/0.0 → **89** (ESS 0.4%, χ² 17.2) | 42/58.5/0.0/0.0 → **44** (ESS 0.4%, χ² 17.2) | 88/11.5/0.0/0.0 → **83** (ESS 87.3%, χ² 24.5) | 89/10.7/0.0/0.0 → **84** (ESS 14.8%, χ² 20.7) | 86/13.5/0.0/0.0 → **81** (ESS 0.7%, χ² 17.2) | 88/11.5/0.0/0.0 → **83** (ESS 0.4%, χ² 17.0) |
| MSAss | (c') affine-IQR | frame | 62/33.8/0.0/4.4 → **59** (ESS 0.5%, χ² 18.9) | 59/26.1/0.0/14.9 → **56** (ESS 0.8%, χ² 19.3) | 27/72.6/0.0/0.0 → **33** (ESS 0.8%, χ² 19.4) | 78/15.4/2.0/4.8 → **73** (ESS 98.6%, χ² 27.5) | 66/26.9/1.5/5.7 → **63** (ESS 26.6%, χ² 23.7) | 59/35.2/0.0/5.5 → **57** (ESS 0.9%, χ² 19.1) | 61/34.2/0.0/4.6 → **58** (ESS 0.5%, χ² 18.9) |
| MSAss | (c') affine-IQR | rate | 100/0.4/0.0/0.1 → **91** (ESS 0.2%, χ² 17.2) | 100/0.3/0.0/0.1 → **90** (ESS 0.3%, χ² 17.7) | 86/14.2/0.0/0.1 → **80** (ESS 0.3%, χ² 17.3) | 87/13.1/0.1/0.1 → **82** (ESS 89.3%, χ² 25.9) | 88/12.2/0.0/0.1 → **82** (ESS 11.9%, χ² 21.9) | 92/8.0/0.0/0.1 → **88** (ESS 0.3%, χ² 17.5) | 100/0.4/0.0/0.1 → **91** (ESS 0.2%, χ² 17.2) |
| Filt | (a)=(c) scale | frame | 44/56.1/0.0/0.0 → **45** (ESS 0.6%, χ² 25.5) | 58/41.9/0.0/0.0 → **56** (ESS 0.4%, χ² 26.0) | 13/72.6/0.0/14.4 → **20** (ESS 0.7%, χ² 25.8) | 83/14.3/2.2/0.3 → **79** (ESS 99.1%, χ² 34.8) | 66/26.7/5.0/2.4 → **63** (ESS 23.7%, χ² 31.3) | 37/51.2/0.3/11.2 → **41** (ESS 1.0%, χ² 25.6) | 44/56.2/0.0/0.0 → **45** (ESS 0.6%, χ² 25.5) |
| Filt | (a)=(c) scale | rate | 92/7.9/0.0/0.0 → **88** (ESS 0.2%, χ² 25.3) | 92/7.6/0.0/0.0 → **88** (ESS 0.2%, χ² 25.8) | 51/18.3/0.0/31.0 → **49** (ESS 0.5%, χ² 26.0) | 85/14.0/1.1/0.4 → **80** (ESS 98.6%, χ² 34.1) | 77/19.5/0.1/3.6 → **71** (ESS 21.2%, χ² 30.5) | 79/13.3/0.0/7.3 → **72** (ESS 0.6%, χ² 25.6) | 92/8.0/0.0/0.0 → **88** (ESS 0.2%, χ² 25.3) |
| Filt | (b) shift | frame | 54/46.5/0.0/0.0 → **53** (ESS 0.4%, χ² 18.7) | 66/34.2/0.0/0.0 → **62** (ESS 0.5%, χ² 18.8) | 51/48.9/0.0/0.0 → **51** (ESS 0.6%, χ² 18.9) | 84/13.9/1.7/0.3 → **80** (ESS 99.3%, χ² 27.0) | 73/23.7/0.6/2.7 → **69** (ESS 22.3%, χ² 23.9) | 51/42.7/0.0/6.0 → **51** (ESS 0.7%, χ² 18.9) | 54/46.4/0.0/0.0 → **53** (ESS 0.4%, χ² 18.7) |
| Filt | (b) shift | rate | 100/0.4/0.0/0.0 → **91** (ESS 0.3%, χ² 18.2) | 100/0.3/0.0/0.0 → **90** (ESS 0.3%, χ² 18.2) | 100/0.4/0.0/0.0 → **91** (ESS 0.3%, χ² 18.3) | 89/10.7/0.2/0.6 → **84** (ESS 94.3%, χ² 26.4) | 87/8.5/0.0/4.5 → **80** (ESS 21.5%, χ² 22.6) | 95/0.7/0.0/3.9 → **84** (ESS 0.6%, χ² 18.4) | 100/0.3/0.0/0.0 → **91** (ESS 0.3%, χ² 18.2) |
| Filt | (c') affine-IQR | frame | 50/49.9/0.0/0.0 → **50** (ESS 0.4%, χ² 18.6) | 61/39.1/0.0/0.0 → **58** (ESS 0.4%, χ² 18.7) | 47/53.3/0.0/0.0 → **48** (ESS 0.5%, χ² 18.8) | 84/14.1/1.8/0.3 → **80** (ESS 99.2%, χ² 27.7) | 71/25.4/1.2/2.7 → **67** (ESS 20.8%, χ² 24.3) | 48/45.5/0.0/6.8 → **48** (ESS 0.6%, χ² 18.7) | 50/49.9/0.0/0.0 → **50** (ESS 0.4%, χ² 18.6) |
| Filt | (c') affine-IQR | rate | 99/0.9/0.0/0.0 → **93** (ESS 0.3%, χ² 18.3) | 99/0.8/0.0/0.0 → **92** (ESS 0.3%, χ² 18.4) | 99/1.1/0.0/0.0 → **93** (ESS 0.3%, χ² 18.5) | 87/11.9/0.2/0.5 → **83** (ESS 95.6%, χ² 26.8) | 85/10.2/0.0/4.5 → **78** (ESS 20.7%, χ² 23.0) | 95/1.5/0.0/3.9 → **85** (ESS 0.4%, χ² 18.6) | 99/0.9/0.0/0.0 → **93** (ESS 0.3%, χ² 18.3) |


**Reading of T3.**
- **Rate mode with the decoy constraint lands on Folded-majority solutions** with small PUF1:
  - **NNLS-χ² and NNLS-MSE:** F 92–100%, PUF1 0.1–7.9%, recovery **88–95**. NNLS-wr2 is less stable (F 76–100%, recovery 70–94).
  - **MaxEnt S ≥ 3:** recovery 79–95.
  - **Best:** (b) MSAss NNLS-χ² at 98 / 2.3 / 0 / 0 → **95**, and (c′) Filtered MaxEnt S = 3 at 98 / 1.7 / 0 / 0 → **95**.
  - **But these are 1–2-frame solutions (ESS 0.2–0.4%).** The state populations come from a handful of folded frames that happen to fit the per-residue errors.
- **The high-ESS end is more honest.** Rate-mode MaxEnt at S = 0.03–0.1 keeps ESS at 55–99% and gives F 82–90%, PUF1 9–18%, PUF2 0–1.2%, recovery 77–86. The best runs with ESS > 50% are:
  - rate: 84.0–85.7 under (b); 81.6–83.6 under (c′); 78.8–80.8 under (a);
  - frame_uptake: 76–80.
  
  These move the prior towards Folded: 73 → 79–85 for MSAss, 81 → 81–86 for Filtered (rate mode).
- **frame_uptake mode does the opposite at the frame level.** NNLS and MaxEnt with S ≥ 0.3 drift to PUF1-heavy solutions: Folded 44–75%, PUF1 24–56%, recovery 45–70 (wr2: F 21–51%). Only its high-entropy end (S = 0.03) carries PUF2 > 0 (1.7–2.8%) together with Folded 80–84%.
- **Decoys, unconstrained:**
  - **Rate mode:** NNLS no longer runs to decoys. They are ≤ 0.3% in MSAss and 0% in Filtered, except (a) wr2 in Filtered, which puts 31% on PUF2-like.
  - **frame_uptake:** NNLS-MSE still pulls 14–21% decoys into MSAss.
  - **Fixing 1% decoy mass in MSAss under (b) rate** raises χ²/cell from 17.9 to 41.8, because the PF < 1 unfolded frames are 5.6 × k_int. The fit then sets PUF1 to 0 (recovery 86). (b′) avoids this artefact in T2.
- **χ²/cell stays at 17–26 at best** (≫ 1, with about 0.1 D model error per cell; compare gci2 `[h4b]`).
  - At the target, every model leaves 60–74 under-predicted by +0.26 and 83–90 over-predicted by −0.20 (mean over timepoints) `[p6]`.
  - The max|z| cell is always 83–90 at 12.5–24 h.
  - No population vector removes either error.

## T4 — summary `[p4]` → `p4_summary.csv`

`p4_summary.csv` has one row per (ensemble × model × averaging mode × selection score/objective × constraint) with the selected F, PUF1, PUF2, decoys, recovery, ESS, all scores and the script tag:
- state level (`level=state(T2)`, script `p2_landscape`): 5 models × 5 scores × 5 constraints;
- frame level (`level=frame(T3)`, script `p3_frames`): 3 models × 10 objectives × 3 constraints.

**How often the selection reaches recovery ≥ 80:**

| | MSAss rate | Filt rate | either ensemble, frame_uptake |
|---|---|---|---|
| state level `[p4]` (of 25 per cell) | (b) 25, (b′) 25, (c′) 21, (a) 5, unscaled 1 | (b) 20, (b′) 19, (c′) 20, (a) 0, unscaled 0 | **0** for every model |
| frame level `[p4]` (of 60 per model, both ensembles) | (c′) 58, (b) 56, (a) 31 | (same rows) | (b) 2, (c′) 1, (a) 0 |

**Closest to target** (top of `p5_tables.log` "top-25"). All are recovery 94–95, all rate mode, and all have PUF2 = 0:
- **state level, (b′) rate:**
  - MSAss χ²: 98 / 2.2 / 0 / 0 → **95**;
  - MSAss wr2: 97 / 3.1 → **95**;
  - Filtered MSE (decoy-free): 97 / 2.6 → **95**.
- **state level, (c′) rate:**
  - MSAss χ²: 98 / 1.9 → **95**;
  - Filtered χ² (decoy-free): 97 / 3.1 → **95**.
- **frame level:** (b) MSAss NNLS-χ² 98 / 2.3 → **95**, at ESS 0.4%.
- **On the target-ratio F profile:** (a) rate MSE reaches F = 96.5% (98.5, MSAss), the only optimum with PUF2 > 0 near the target, but only along that 1-D line `[p2f]`.

## Conclusion

1. **Rate averaging gives a Folded-majority optimum; frame_uptake does not.** Under median-matched BV, with rate averaging and either an additive shift (b, or b′ with PF ≥ 1) or the affine median+IQR match (c′), the data prefer a Folded-majority ensemble:
   - **state level:** F 96–99%, PUF1 1–3.5%, decoys ≈ 0, recovery 94–95. This holds for χ², MSE and w·r², with or without the PUF3/U constraint in MSAss. Filtered needs its single PUF2-like decoy frame excluded.
   - **PUF2 is always driven to 0.** The target's 0.5% PUF2 is never preferred. The target is close to the optimum, in the top 1–2% of the grid, but never a local optimum: the improving move is always PUF2 → Folded.
   - **So the physical picture is reproduced for PUF1 only.**
   - **Under frame_uptake averaging** no scaled model gives a Folded-majority optimum with small minor states. The landscape is almost flat, with χ² changing by about 1–4% across F = 50–100%, and it drifts to PUF1-heavy (recovery 4–63), or to decoys when unconstrained.
2. **Sharpest optimum: rate mode with (b) or (c′) and the pooled-σ χ² or MSE.**
   - Under (b) the near-optimal set is F 98–100%, PUF1 0.2–1.5% (MSAss) / F 97–100%, PUF1 0.2–3% (Filtered), recovery 90–95.
   - (b′) is slightly broader (PUF1 up to 5–8.5%, recovery 87–98) and is the physically defensible variant: the plain shift makes 20–59% of residue-frames PF < 1.
   - Spearman is broader still, and max|z| is uninformative.
   - The multiplicative scale (a)=(c) in rate mode has a much broader optimum (χ²: F 47–92% MSAss, 10–87% Filtered). Its best points lie at F 63–87% with PUF1 13–38% (recovery 59–82).
   - **Why rate mode works:** frame_uptake makes uptake linear in the population, so partially exchanged peptides need tens of percent of open frames. Rate mode averages the exchange rates (the EX2 limit, fast interconversion), so a few percent of PUF1 is enough and any extra fast-exchanging mass is penalised.
3. **Recovery ≥ 80 is reached:**
   - **reliably at state level** by (b), (b′) or (c′) in rate mode under every score except max|z|: 19–25 of 25 selections per ensemble, best 95;
   - **at frame level** by rate-mode NNLS/MaxEnt under the decoy constraint (88–95), but only at ESS ≤ 0.6%;
   - **with ESS > 50%** by rate-mode MaxEnt at S = 0.03–0.1 under (b), (c′) or (a): 77–86. Here the gain over the prior is up to about 11 points for MSAss and about 5 for Filtered.
4. **Caveats.**
   - The best fits still have χ²/cell of about 17–27 under pooled σ. The misfit is per-residue model error (60–74 under-predicted, 83–90 over-predicted, and 31–58 under-protected by about 2 ln in lnPF) that no population vector fixes.
   - The optimum's location is therefore set by how each averaging mode trades those errors against the minor-state uptake. It is not a clean measurement of PUF1 or PUF2.
   - Filtered's single PUF2-like frame is preferred by every score whenever it is allowed.
