# stk: one self-tuning lnPF kernel per ensemble, used two ways (two-sided Σ in selection; OMC Laplacian)

This was a read-only analysis: no repo edits and no framework fits. Everything is NumPy/SciPy, run from the repo root with `.venv/bin/python` and `OMP_NUM_THREADS=4`.

I reused existing code rather than rewriting it:
- from `selgci/q0_common.py`: data, the b′ rate model, p0, the EF and dual solvers, and the splits;
- from `lap/l0_common.fit_omc`: the OMC solver;
- from gci2 `h2`: the walk-covariance eigen-form.

Every number is tagged with the script that produced it. The full tables are in `s8_tables.md`.

| tag | script | outputs |
|---|---|---|
| `[s0]` | `s0_common.py` | shared: kernel, walk covariance, Σ scorer, rules, pure-OMC solver |
| `[s1]` | `s1_kernel.py` | `s1_kernel.csv`, `s1_primary.npz` (L_st, C_walk, λ_geo), `s1.log` |
| `[s2]` | `s2_t1_sweep.py` | `s2_sweep.csv`, `s2_weights.npz`, `s2_flags.npz`, `s2.log` |
| `[s3]` | `s3_t1_select.py` | `s3_selected.csv`, `s3_summary.csv`, `s3_flags_p0.csv`, `s3.log` |
| `[s4]` | `s4_t2_omc.py` | `s4_sweep.csv`, `s4_gamma_scale.csv`, `s4_weights.npz`, `s4.log` |
| `[s5]` | `s5_t3_iso.py` | `s5_sweep.csv`, `s5_selected.csv`, `s5_summary.csv`, `s5.log` |
| `[s6]` | `s6_t2_select.py` | `s6_selected.csv`, `s6_summary.csv`, `s6.log` |
| `[s7]` | `s7_fig.py` | `F_a_neff_vs_k.png`, `F_b_ruleD_vs_lam.png`, `F_c_ruleD_vs_gamma.png` |
| `[s8]` | `s8_tables.py` (formatting only) | `s8_tables.md` |

## Bottom line

1. **The two-sided variance changes which S is picked, but not the recovery; it is inert on Filtered** `[s3]`.
   - **MSAss.** At λ_geo (diag), rule D picks a different S in 6 of 6 MSAss folds. Mean recovery stays within noise: 93.14 ± 1.22 → 92.76 ± 1.14, with ESS 53 → 60%. D|ESS≥50 gives 92.99 → 92.83.
   - **Filtered.** D picks p0 in 4/6 folds at every λ in diag form (94.95 ± 3.12 throughout).
   - **λ = 0 reproduces selgci exactly:** 48/48 picks identical, flags identical, |Δrec| ≤ 0.008.
2. **The flags that disappear are the ensemble-uncertain cells, as expected** `[s3]`.
   - At λ_geo, MSAss p0 flags go from 128 to 79 (49 disappear); Filtered goes from 99 to 91 (8 disappear).
   - In MSAss, 34 of the 49 are 32–45 (16/16) and 33–40 (18/18), where λ_geo·C_ii/σ² ≈ 3.9–6.1.
   - The flags that persist are the BV-error peptides, where the ratio is ≲ 1: 60–74 (21 of 27 remain), 83–90 (18/20), 11–26 (15/15), 27–31 (16/18).
   - So the two-sided Σ separates "the ensemble is uncertain here" from "the forward model is wrong here". That split is its most useful output.
3. **ISO: the diag form roughly holds; the full Mahalanobis form fails; rule A becomes "always p0"** `[s5]`.
   - **Union kernel at λ_geo, diag:** rule D on ISO_BI gives 96.4 ± 1.1 (17/18 ≥ 95, minimum 93.4) at σ = 0.01 and 96.6 ± 0.8 (18/18) at σ = 0.02. The λ = 0 baseline is 98.4 / 98.2, both 18/18.
   - **Full form:** D collapses to 74.9 ± 14.7 (3/18) and 87.3 ± 12.8 (8/18).
   - **Why the full form fails.** On ISO_BI, 99.99% of the truth − p0 prediction shift lies along C_walk's top eigenvector, the open/closed contrast. Under Σ_full its Mahalanobis length drops from 40,468 to 1.8. C_walk at uniform weights encodes "the population is uncertain", which is exactly the quantity the reweighting estimates. The full form therefore forgives the signal.
   - **Rule A.** All flags vanish at λ ≥ λ_geo/3, so A returns p0 (ISO_BI 0/18).
   - **Dense kernel (λ_geo = 3.1/N, connected):** passes 18/18 in both diag and full, at both σ. On MoPrP it is neutral: D 93.10 ± 1.22 (MSAss), 94.95 ± 3.12 (Filtered) `[s3, uptD]`.
4. **The self-tuning OMC term does not help** `[s6]`.
   - It raises ESS: MSAss D goes 60 → 80% at γ = 3.
   - It lowers recovery: MSAss D goes 92.76 → 91.69, with the minimum 91.7 → 88.9. On Filtered it adds variance (minimum down to 84.6 at γ = 0.3) without a mean gain.
   - **Pure OMC (no KL) drifts off the ΔG populations** toward the L_st null vector w ∝ √d. That attractor has recovery 73.2 / 80.8 and PUF1 14.5 / 13.3% `[s1]`. Rule D: MSAss 85.6 ± 12.0 (minimum 61.6), Filtered 91.4 ± 7.1 (minimum 79.8).
   - Lap found the same thing with the unweighted graph. Self-tuning weights do not change the conclusion.
5. **Recommended carry-forward:**
   - **λ = 0 for selection** (selgci rule D unchanged).
   - **γ = 0.**
   - **ESS ≥ 50% as an acceptance check, not a candidate filter.** D|ESS≥50 used as a filter fails ISO_BI: it picks 86% at ESS 54%.
   - **Report the λ_geo two-sided z per cell as a diagnostic** that labels each flag as ensemble-uncertain or model-error.
   - If simulation variance must enter selection, use the **dense-kernel, diag-only** form (λ ≈ 3/N for ISO, about 0.11 / 0.036 for MoPrP). It passes ISO and changes nothing on MoPrP.
   - Never use the full Mahalanobis form with C_walk computed at uniform weights.

## K1. Kernel construction and λ_geo `[s1]` (full table: `s8_tables.md` T0; figure `F_a_neff_vs_k.png`)

### Metric space and kernel
- **Metric space.** MoPrP uses per-frame scaled-BV lnPF, max(0.35·N_c + 2.0·N_h + δ, 0), with δ = −3.108 / −3.174. That is 97 residues × 500 frames, L2. ISO uses 0.35·N_c + 2.0·N_h over 294 residues: 874 frames (BI) and 2225 frames (TRI).
- **Kernel.**
  - σ_i = mean (or RMS) distance to the k nearest neighbours.
  - K_ij = exp(−d²/(σ_iσ_j)), on the symmetric-union kNN support plus the diagonal, or dense.
  - α = ½ normalisation: K̃ = K/√(q_iq_j).
  - From that: S = D^-½K̃D^-½, L_st = I − S, and the lazy walk P = (I + P₀)/2.
- **Walk covariance.** C_walk = (1/N²) F̃ᵀ G F̃ with G = I + Σ_{k≥2}(g(μ_k) − 1)φ_kφ_kᵀ and g = (1 + μ′)/(1 − μ′). This is gci2 h2's symmetric eigen-form of 2(I − P)⁻¹ − I with the stationary mode removed.
  - λ_geo = tr C_walk / tr C_ens, which is the inflation divided by N (gci2's convention, so the numbers are comparable to 0.366 / 0.074). N_eff = N / inflation.
  - **New: g is capped at 2N − 1.** This is a finite chain: no mode can be more correlated than N identical frames. When the support is disconnected, the stationary mode is pinned to √d.
  - **Why the cap was needed:** the ISO_BI kNN-union graph has exactly 2 components, open (44 frames) and closed (830). Without the cap, λ_geo diverges.
- **Check.** Re-running gci2's own construction (uptake space, default BV, dense, c = 1) gives 0.5361 / 0.2495 (MSAss k = 10 / 20) and 0.0898 / 0.0613 (Filtered). gci2 reported 0.54 / 0.25 and 0.090 / 0.061, so it reproduces `[s1]`.

### Which per-frame quantity F to average (rate mode)
- **Delta method** ('delta'): linearise the uptake around the uniform-weight mean rate, J·C_walk(rate)·Jᵀ. This gives unphysical uptake SDs, up to 0.59 D in MSAss. The heavy-tailed open-frame rates make the linearisation fail. Per-cell C_walk/σ² has median 2.9 and maximum 102 `[s1]`.
- **Per-frame uptake curves** ('upt'): the SD stays at most 0.16 D (MSAss) and 0.06 D (Filtered) `[s1]`.
- **I use 'upt' as primary.** This is the uptake-averaging approximation to the rate-mode forward model. Delta is carried as a sensitivity in `s8_tables.md` T1 (delta); it degrades Filtered rule A to minimum 72.7.
- **C_walk is evaluated at uniform weights,** not at p0 or at the candidate w. That is an approximation throughout.

### Primary kernel
**k = 20, mean σ, union, lazy** is used for both roles (Σ and L_st):

| ensemble | λ_geo (N_eff), union k = 10 / **20** | dense k = 10 / 20 | non-lazy union k = 20 | gci2 adopted (uptake space, dense) |
|---|---|---|---|---|
| MSAss (upt) | 0.893 (1.1) / **0.614 (1.6)** | 0.157 (6.4) / 0.110 (9.1) | 0.306 (3.3) | 0.366 (2.7); bracket 0.044–0.54 |
| MSAss (delta) | 0.454 (2.2) / 0.237 (4.2) | 0.131 / 0.071 | 0.118 | |
| Filtered (upt) | 0.314 (3.2) / **0.183 (5.5)** | 0.045 (22) / 0.036 (28) | 0.090 (11.1) | 0.074 (13.5); bracket 0.013–0.09 |
| ISO_BI (exact) | 0.121 (8.3) / **0.116 (8.6)** (2 components, capped) | 0.0035 (282) / 0.0035 (283) | 0.110 (9.1) | 5/N = 0.0057 |
| ISO_TRI (exact) | 0.150 (6.7) / **0.088 (11.3)** | 0.0014 (702) / 0.0014 (704) | 0.044 (22.7) | 5/N = 0.0022 |

- **Bandwidth.** Mean vs RMS changes λ_geo by ≤ 2% everywhere.
- **k and support dominate.**
  - The union (kNN-restricted) support is slowly mixing: μ₂ = 0.995 / 0.991 for MoPrP. It gives λ_geo 1.7–2.5× gci2's MSAss/Filtered values. N_eff is 1.6 / 5.5 versus gci2's 2.7 / 13.5.
  - The dense kernel in lnPF space is 4–8× smaller than gci2's: N_eff 9 / 28.
  - **In high-dimensional lnPF space** (ISO: 294 dims) pairwise distances concentrate. The self-tuning σ_i is then close to a typical distance, so the dense kernel is nearly complete: μ₂ = 0.14–0.19, and N_eff ≈ N/3, which is the lazy-walk floor.
- **ISO vs gci2 T6's ≈ 5/N.**
  - The union kernel gives 101/N (BI, about 20× 5/N) and 197/N (TRI, about 39× 5/N).
  - The dense kernel gives 3.09/N and 3.16/N, about 0.62× 5/N.
  - **So the ≈ 5/N that controlled the test's size corresponds to the dense (near-independent) kernel, not the kNN-union walk.**
- **C_walk vs σ_exp² (primary).**
  - MSAss: median ratio 0.29, maximum 7.3; 43 of 210 cells have λ_geo·C_ii > σ². Per peptide the ratio is largest for 33–40 (5.2), 32–45 (3.0) and 5–10 (2.6); 60–74 is 0.72 and 83–90 is 0.27.
  - Filtered: median 0.013, maximum 1.03; 1 cell above σ² `[s1, s3]`.
- **L_st null vector.** w* ∝ √d has ESS 99.7%. Recovery is 73.2 (MSAss; populations 78.3 / 14.5 / 2.0 / 2.2 / 3.0) and 80.8 (Filtered; 84.8 / 13.3 / 1.8) `[s1]`.

## T1. Two-sided variance in selection `[s2, s3]` (full tables: `s8_tables.md` T1 upt / uptD / delta; figure `F_b_ruleD_vs_lam.png`)

### Setup
- The sweep is selgci q1 exactly: scaled BV b′, rate averaging, KL to binary-openness p0, floored pooled σ, 3 contiguous 2-fold block splits, S ∈ {0} ∪ logspace(−3.5, 0.5, 33), with the EF solver.
- Held-out scoring uses Σ = D_exp + λ·C_shape, where C_shape = C_walk/λ_geo. With this convention:
  - λ = λ_geo gives D_exp + C_walk exactly;
  - λ = 1/N gives the independent-frame magnitude tr C_ens/N, in C_walk's shape.
- **Diag version.** z_i = r_i/√Σ_ii is used for the Šidák flags and for F = ½Σz² + KL.
- **Full version.** Flags still come from the marginal diag z (a per-cell test). F = ½rᵀΣ⁻¹r + KL. **No ridge was needed:** cond(Σ) ≤ 502 ('upt'), because D_exp ≥ 0.0586² keeps Σ positive definite `[s2]`.
- **Rules and candidates.** A, B, D, plus D|ESS≥50. Candidates are ESS ≥ 10% ∪ {S = 0}.
- **Reproduction:** λ = 0 matches `selgci/q3_selected.csv` in 48/48 picks (S and flags identical; |Δrec| ≤ 0.008) `[s3]`.

### Rule D (and A), primary 'upt', mean ± SD over 6 folds `[s3]`

| ens | λ | D diag: rec · ESS | D full: rec · ESS | D\|ESS≥50 diag | A diag | flags at D pick: Σ(λ) / λ = 0 | D: folds with changed S |
|---|---|---|---|---|---|---|---|
| MSAss | 0 | 93.14 ± 1.22 · 53% | (same) | 92.99 ± 1.04 · 74% | 92.90 ± 1.21 · 64% | 18.8 / 18.8 | – |
| MSAss | 1/N = 0.002 | 93.13 ± 1.24 · 53% | 93.20 ± 1.20 · 53% | 92.98 · 74% | 92.97 · 60% | 18.7 / 19.0 | 1 |
| MSAss | λ_geo/3 = 0.205 | 93.10 ± 1.26 · 50% | 93.42 ± 1.44 · 45% | 92.99 · 74% | 92.56 · 70% (p0 4/6) | 14.7 / 19.7 | 5 |
| MSAss | **λ_geo = 0.614** | **92.76 ± 1.14 · 60%** | 92.76 ± 1.14 · 68% | 92.83 · 73% | 92.33 · 72% (p0 4/6) | **12.7 / 20.0** | 6 |
| MSAss | 3λ_geo = 1.84 | 93.00 ± 1.08 · 60% | 92.97 ± 1.05 · 69% | 93.06 · 72% | 92.08 · 83% (p0 5/6) | 7.2 / 19.8 | 6 |
| Filtered | 0 | 94.95 ± 3.12 · 65% (p0 4/6) | (same) | 96.07 · 91% | 96.07 · 91% (p0 6/6) | 16.5 / 16.5 | – |
| Filtered | 1/N | 94.95 ± 3.12 | 94.95 ± 3.12 | 96.07 | 96.07 | 16.5 / 16.5 | 0 |
| Filtered | λ_geo/3 = 0.061 | 94.94 ± 3.11 | 94.77 ± 3.07 | 96.07 | 96.07 | 15.5 / 16.5 | 1 |
| Filtered | **λ_geo = 0.183** | **94.95 ± 3.12 · 65%** | 94.34 ± 3.18 · 64% | 95.30 ± 1.91 · 85% | 96.07 | **15.2 / 16.5** | 0 |
| Filtered | 3λ_geo | 94.95 ± 3.12 | 93.82 ± 3.66 | 95.30 | 94.93 ± 2.81 (min 89.2) | 15.0 / 16.5 | 0 |

**Observations.**
- **Populations at the picks do not move.** MSAss D at λ_geo gives Folded 96.6 / PUF1 2.95 / PUF2 0.00, against 96.8 / 2.83 / 0.00 at λ = 0. Filtered gives 96.2 / 3.14 / 0.24 at every diag λ `[s3]`.
- **Per-fold detail** `[s3]`:
  - MSAss picks move toward p0 in the two folds where the uncertain 32–45/33–40 cells are held out:
    - cut31|32 C 32-101: S 2.37 → 0.237, recovery 94.7 → 93.7, ESS 13 → 75%;
    - cut45|60 N 5-45: D stays at p0-like S.
  - Rule A, with fewer flags to separate candidates, returns p0 in 4/6 folds at λ_geo (2/6 at λ = 0).
  - Filtered picks are unchanged except D|ESS≥50: one fold leaves p0 for S = 0.21 (91.4).
- **Full vs diag.**
  - Full raises MSAss D slightly at λ_geo/3 (93.42) but lowers Filtered (94.34 at λ_geo; 93.82 at 3λ_geo).
  - Full B/D move 2/6 Filtered folds off p0 (cut45|60 N 5-45: 96.1 → 92.5).
  - Neither version helps consistently.
- **Delta-method C** (sensitivity): MSAss D 93.24 ± 1.15 at λ_geo. Filtered rule A drops to 90.6 ± 9.3 (minimum 72.7), because the unphysically large delta variances let over-fit points reach zero flags `[s3]`.
- **Dense-kernel C ('uptD', λ_geo 0.110 / 0.036):** MSAss D 93.10 ± 1.22 (5/6 folds change S); Filtered 94.95 ± 3.12 (0 changes). It is neutral `[s3]`.

### Which flags disappear `[s3]` (`s3_flags_p0.csv`; p0, upt)

| ensemble | λ_geo/3 | λ_geo | 3λ_geo |
|---|---|---|---|
| MSAss: p0 flags 128 → | 90 (−38) | 79 (−49) | 45 (−83) |
| Filtered: p0 flags 99 → | 94 (−5) | 91 (−8) | 91 (−8) |

At λ_geo, per peptide (flags λ = 0 → λ_geo; median λ_geo·C_ii/σ² of the disappeared | persisting cells):

| MSAss | 32–45 | 33–40 | 60–74 | 68–75 | 27–31 | 83–90 | 11–26 |
|---|---|---|---|---|---|---|---|
| flags | 16 → 0 | 18 → 0 | 27 → 21 | 12 → 9 | 18 → 16 | 20 → 18 | 15 → 15 |
| ratio gone \| kept | 3.87 \| – | 6.08 \| – | 0.90 \| 1.07 | 2.01 \| 0.44 | 0.94 \| 0.75 | 0.31 \| 0.35 | – \| 0.29 |

- Filtered loses only its 33–40 flags (7 → 0; ratio 0.33) plus one 82–101 cell. 60–74, 83–90, 11–26, 27–31 and 68–75 are all untouched (ratio 0.01–0.09).
- **The disappearing flags are where the ensemble's own spread in predicted uptake is comparable to or larger than σ_exp:** the β1–α1 segment 32–45/33–40, mostly in MSAss.
- **The persistent ones are the BV model-error peptides identified in dg / gci2 / selgci.**

## T2. OMC with the self-tuning L_st `[s4, s6]` (full tables: `s8_tables.md` T2; figure `F_c_ruleD_vs_gamma.png`)

### Setup
- Objective: χ² + KL(w‖p0)/S + γ_abs·wᵀL_st w, using lap's 500-logit solver warm-started along S.
- Scaling: γ_abs = γ·c, with c = median over folds of (KL/S)/(wᵀL_st w) at the γ = 0 rule-D pick under the T1 diag λ_geo z (lap/l3's convention).
  - **c = 724.4 for MSAss** (5/6 folds have S > 0) and **78.4 for Filtered** (2/6 folds) `[s4]` (`s4_gamma_scale.csv`).
- **Pure OMC:** χ² + (c/S)·wᵀL_st w, with no KL. Each S is cold-started from log p0 ('pomc') or from uniform ('pomcu'). In both, the S = 0 row is p0, the reference.
- **Stationarity:** candidates reach 7.5e-6 at the 99th percentile. One non-picked pomc candidate reaches 0.11. Every pick is ≤ 3.1e-4 `[s6]`.

### Results at λ_geo, diag (rule D | D|ESS≥50; recovery ± SD (minimum) · ESS) `[s6]`

| | γ = 0 | 0.1 | 0.3 | 1 | 3 | pure OMC (p0 start) | pure OMC (uniform start) |
|---|---|---|---|---|---|---|---|
| MSAss D | 92.76 ± 1.14 (91.7) · 60% | 91.87 ± 1.20 (90.0) · 61% | 91.87 ± 1.05 (90.4) · 66% | 91.71 ± 1.29 (89.6) · 73% | 91.69 ± 1.55 (88.9) · 80% | 85.58 ± 12.07 (61.6) · 66% | 85.55 ± 12.04 (61.6) · 65% |
| MSAss D\|ESS≥50 | 92.83 · 73% | 92.79 · 73% | 92.63 · 72% | 91.71 · 73% | 91.69 · 80% | 88.69 (72.2) · 79% | 88.55 (72.3) · 79% |
| Filtered D | 94.95 ± 3.12 (88.6) · 65% | 94.51 ± 3.84 (86.7) · 65% | 94.13 ± 4.66 (84.6) · 65% | 94.59 ± 4.12 (86.2) · 69% | 95.16 ± 3.61 (88.1) · 74% | 91.37 ± 7.06 (79.8) · 64% | 91.46 ± 7.14 (79.7) · 64% |
| Filtered D\|ESS≥50 | 95.30 · 85% | 96.44 · 84% | 96.44 · 84% | 95.17 · 85% | 95.38 · 78% | 90.05 (76.3) · 82% | 90.05 (76.3) · 82% |

- **Structural smoothness does not help recovery.**
  - MSAss loses 0.9–1.1 points at every γ > 0. The populations shift toward PUF1: at γ ≥ 0.1, Folded 95.7–95.8 and PUF1 3.7–3.8, against 96.6 / 2.95 at γ = 0.
  - It buys ESS: 60 → 80% at γ = 3; 6/6 folds above 50% at γ ≥ 1.
  - Filtered: no mean gain at any γ, and the worst fold gets worse (minimum 88.6 → 84.6–86.7 at γ = 0.1–1). There is a marginal +1.1 for D|ESS≥50 at γ = 0.1–0.3, coming from 2 folds: cut31|32 N 5-31 returns to p0 (91.4 → 96.1) and cut31|32 C 32-101 goes 96.1 → 98.3.
  - The same pattern holds under λ = 0 scoring and under full Σ (`s8_tables.md` T2). It is the lap result again: the OMC pull is toward w ∝ √d, which is PUF1-heavy (14.5% / 13.3%) and lower in recovery than p0.
- **Pure OMC drifts off the ΔG populations** `[s6]` (mean over folds along the path):
  - **Filtered.** It drifts immediately, and the p0 start does not matter (pomc ≡ pomcu, because p0 has full support). At S = 7e-4: recovery 86.9, PUF1 9.0%. At S = 0.024: 69.0, PUF1 26.9%, against p0's 96.1 / 3.3%.
  - **MSAss with the p0 start.** Frames that p0 sets to about 0 (log p0 down to −3e5, including all PUF2) have zero softmax gradient and never return. The small-S path therefore stays near p0 (recovery 92.7–91.7 for S ≤ 0.0075), then drifts into PUF1: 5.8% at S = 0.024, 16% at 0.24, 30% at 0.75.
  - **MSAss with the uniform start.** PUF2 comes back to 0.5–4%, but KL(w‖p0) becomes 1700–15000 and recovery falls to 89 → 59.
  - The rule picks inherit this: D 85.6 (MSAss), 91.4 (Filtered), with minima 61.6 / 79.8.
  - **With the S grid scaled by c, the only path point at or near p0's recovery is the trivial S = 0.**

## T3. ISO control `[s5]` (full tables: `s8_tables.md` T3)

### Setup
- The selgci q2 sweeps are repeated exactly: self-consistent BI/TRI features, residue target plus noise σ ∈ {0.01, 0.02} × 3 seeds, 6 block folds, uniform p0, and the exact dual solver.
- Σ = σ²I + λ·C_shape uses the same kernel construction in ISO lnPF space.
- **Reproduction:** λ = 0 matches selgci q3 in 216/216 picks (|Δrec| 1e-14).

### ISO_BI, rule D (recovery mean ± SD (minimum) · #≥ 95 of 18)

| λ | σ = 0.01 diag | σ = 0.01 full | σ = 0.02 diag | σ = 0.02 full |
|---|---|---|---|---|
| 0 (selgci) | 98.4 ± 0.4 (97.6) · 18/18 | – | 98.2 ± 0.6 (96.7) · 18/18 | – |
| 1/N | 97.7 ± 1.2 (93.4) · 17 | 95.7 ± 1.4 (93.4) · 13 | 97.5 ± 1.7 (90.9) · 17 | 96.0 ± 1.8 (90.9) · 14 |
| 5/N | 95.3 ± 4.0 (86.2) · 14 | 93.3 ± 4.0 (86.2) · 8 | 97.2 ± 1.3 (93.7) · 16 | 95.0 ± 1.8 (91.0) · 11 |
| λ_geo/3 | 96.6 ± 1.0 (93.4) · 17 | 81.0 ± 15.1 (62.6) · 5 | 97.1 ± 0.5 (96.3) · 18 | 94.2 ± 4.2 (79.8) · 10 |
| **λ_geo = 0.116 (union)** | **96.4 ± 1.1 (93.4) · 17** | **74.9 ± 14.7 (62.6) · 3** | **96.6 ± 0.8 (95.5) · 18** | **87.3 ± 12.8 (62.6) · 8** |
| 3λ_geo | 96.1 ± 1.3 (93.4) · 15 | 70.7 ± 11.3 (62.6) · 0 | 96.6 ± 0.9 (95.5) · 18 | 79.9 ± 15.7 (62.6) · 5 |
| dense λ_geo = 0.0035 | **98.6 ± 0.4 (97.8) · 18** | **97.6 ± 0.7 (96.6) · 18** | **98.3 ± 0.5 (97.0) · 18** | **97.5 ± 0.6 (96.3) · 18** |

**The rules no longer all pass with the union kernel.**
- **Rule A, diag.** At λ ≥ λ_geo/3 it returns p0 (62.6%) in 12–18/18 replicates, because every candidate has 0 flags.
- **D, diag.** D falls back to B (argmin F) and is 1 replicate short at σ = 0.01: seed 0, cut196, held-out C, recovery 93.4 at S = 0.068. There every one of the 35 candidates is flag-free, and the C_ii-weighted F minimum moves to a smaller S.
- **Intermediate λ (1/N, 5/N) also cost D replicates:** 14–17/18. A few flags remain, so the minimum-flag set truncates before B's optimum.
- **D|ESS≥50 fails at every λ** (86% at ESS 54%). This confirms selgci: ESS ≥ 50% must be an acceptance check, never a candidate filter.
- **The full form fails, and this is structural.** The C_walk top mode carries 92% of tr C_walk and 99.99% of the truth − p0 shift, and Σ_full whitens it away (Mahalanobis 40,468 → 1.8) `[s5 check, inline]`.
  - On MoPrP the shift is less aligned: 48–61% in the top-5 modes. That is why full Σ only halves its length there (for example 153 → 74) and does little harm.

**ISO_TRI** (ceiling ≈ 62).
- Diag D at λ_geo gives 58.1 ± 2.6 (σ = 0.01) and 55.4 ± 3.1 (σ = 0.02), against 60.1 and 56.5 at λ = 0.
- Full lowers these by about 2.5 more.
- The dense kernel gives 59.9 and 56.3.

## T4. Figures `[s7]`

- `F_a_neff_vs_k.png`: N_eff and λ_geo vs k, per bandwidth × support, for MSAss, Filtered, ISO_BI and ISO_TRI. MoPrP 'upt' is solid and 'delta' dashed; gci2's adopted values, 5/N and 1/N are marked.
- `F_b_ruleD_vs_lam.png`: rule-D (and D|ESS≥50) recovery and ESS vs λ, diag and full, for both MoPrP ensembles ('upt' solid, 'delta' dotted). The λ = 0 baseline is the grey band. The right column shows ISO_BI and ISO_TRI rule-D recovery vs λ, including the dense kernel.
- `F_c_ruleD_vs_gamma.png`: rule A/B/D/D|ESS≥50 recovery and ESS vs γ, including both pure-OMC variants. Σ at λ_geo is solid; λ = 0 is dotted.

## Conclusion

- **Does the two-sided variance change which model is selected?**
  - **S changes, the model does not.** On MSAss, D's S changes in 5–6/6 folds once λ ≥ λ_geo/3. The selected populations and recovery stay within 0.4 points of selgci, and ESS rises slightly (53 → 60%).
  - On Filtered nothing moves in diag form; D stays at p0 in 4/6 folds.
  - **What changes most is the flags:** 38% of the MSAss p0 flags (all of 32–45/33–40) are explained by ensemble uncertainty. The 60–74 / 83–90 / 11–26 / 27–31 flags are not.
- **Does it still pass ISO?**
  - Not strictly with the kNN-union kernel. Diag D gives 17/18 at σ = 0.01 (minimum 93.4) and 18/18 at σ = 0.02.
  - Full Σ fails (3–8/18), and rule A degenerates to p0.
  - With the dense self-tuning kernel (λ ≈ 3/N) everything passes 18/18, and it is neutral on MoPrP.
  - **So the union-kernel λ_geo is too large to use inside the selection rule.** It is a correct statement about how badly the ensemble mixes between basins (ISO_BI's open and closed states are not connected at all). But that uncertainty is precisely what reweighting resolves, and it should not be credited back to the residuals.
- **Does the self-tuning OMC help?** No. It trades about 1 recovery point (MSAss) or worst-fold stability (Filtered) for ESS. Pure OMC abandons the ΔG populations for the √d attractor (PUF1-rich, 73–81% recovery).
- **Recommended for production:**
  - **λ = 0 in the selection rule**, or at most the dense-kernel λ_geo in diag form only;
  - **γ = 0**;
  - **rule D** (flags, then F), with ESS ≥ 50% as an acceptance check;
  - **the λ_geo (union, diag) two-sided z per cell as a reported diagnostic** that labels each flag as ensemble-uncertain or model-error.

## Caveats

- **C_walk at uniform weights.** It is computed once per ensemble, not at p0 or at each candidate w. For MSAss, p0 sets PUF2 and some PUF1/U frames to about 0, so the uniform-weight variance includes frames the reweighted ensemble excludes.
- **Rate mode.** The primary C uses per-frame uptake curves, i.e. the uptake-averaging approximation. The delta-method rate covariance is unphysical (SD up to 0.59 D) and is reported only as a sensitivity.
- **The finite-chain cap g ≤ 2N − 1** is my addition. It binds only on the disconnected ISO_BI union graph and on the slowest MSAss k = 10 mode. Without it, λ_geo for ISO_BI is infinite or arbitrary: 0.015 vs 0.110 depending on the eigenvector rotation, before the √d pinning.
- **Folds.** MoPrP has 6 folds per ensemble, so ±1 recovery point is within fold noise. The ISO counts are over 18 replicates (3 seeds × 6 folds).
- **The OMC γ scale for Filtered** (c = 78.4) rests on the 2 folds whose γ = 0 D pick has S > 0.
