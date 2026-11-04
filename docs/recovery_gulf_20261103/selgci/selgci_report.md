# selgci: two simple S-selection rules, tested on sweeps with a known answer

This was a read-only analysis: no repository edits and no framework fits. Every optimisation used a NumPy/SciPy solver. All scripts are in this folder and run from the repo root with `.venv/bin/python`. Every number below is tagged with the script that produced it.

| tag | script | output |
|---|---|---|
| `[q0]` | `q0_common.py` | shared code: forward models, p0, splits, solvers, held-out scores |
| `[q1]` | `q1_moprp_sweep.py` | `q1_sweep.csv`, `q1_units.npz`, `q1_weights.npz`, `q1.log` |
| `[q2]` | `q2_iso_sweep.py bi\|tri` | `q2_sweep_{bi,tri}.csv`, `q2_units_*.npz`, `q2_{bi,tri}.log` |
| `[q3]` | `q3_select.py` | `q3_selected.csv` (every pick), `q3_summary.csv`, `q3.log` |
| `[q4]` | `q4_fig.py` | `F_*.png` |
| `[q5]` | `q5_tables.py` (formatting only) | `q5_tables.md` |
| `[q6]` | `q6_checks.py` | `q6.log` (solver checks, B/C agreement, irreducible flags, σ-floor sensitivity) |

## Setup

### MoPrP (AF2_MSAss and AF2_filtered)

- **Forward model.** Scaled BV (b′): lnPF = max(0.35·N_c + 2.0·N_h + δ, 0), with δ = −3.108 (MSAss) and −3.174 (Filtered), read from `pop/p1_models.json`. Rate averaging. `_featurise_a101_k109` features. All 14 peptides × 15 timepoints, row-normalised map (trim 2) `[q0]`.
- **Reference distribution p0.** This is dg T2's rule-feature (family A) MaxEnt on segment set S11, with targets p_open = K/(1+K). I recomputed it in log space; it matches `dg/d3_weights.npz` to 1e-18 `[q1]`.
  - Recovery 91.7 (MSAss) and 96.1 (Filtered); ESS 84% and 91%.
  - **MSAss caveat.** The MSAss A-constraints are jointly infeasible, so p0 puts weight ≈ 0 (log p0 down to −3e5) on all 10 PUF2 frames and on some PUF1/U frames. Under KL(w‖p0) these frames cannot come back at any S, so **MSAss PUF2 is 0 along the whole sweep**.
  - **Filtered.** Only the β1 and β2 constraints are feasible; the α1/core group is dropped `[q1]`.
- **Objective.** χ²_train(w) + KL(w‖p0)/S, where χ²_train is the mean over training cells of (r/σ_pooled)².
  - Grid: S ∈ {0} ∪ logspace(−3.5, 0.5, 33).
  - Solver: L-BFGS on μ in the exponential family that contains the stationary point, w ∝ p0·exp(μ·Φ), with Φ = standardised per-residue exchange rates of the training residues. Warm-started along S.
  - Validation: this reproduces the 500-logit L-BFGS optimum (χ² 7.509 / 6.760 / 6.011 at S = 0.1 / 1 / 3.16, identical in both parametrisations).
  - Stationarity residual on every candidate is ≤ 7.3e-6 `[q6]`.
  - Sweep range: on full data, ESS falls from 84% at S = 0 to 82% at S = 0.1, 29% at S = 1 and 3% at S = 3.16 (MSAss, σ floored). **The candidate set is the grid points with ESS ≥ 10%**, which corresponds to S ≈ 0.003–2 on the χ² scale.
- **σ.** The pooled σ per peptide is the RMS over timepoints of 1/weight, floored at 0.0586. With the floor, 13 of 14 peptides sit at the floor; only 75-79 is above it, at 0.070. The unfloored variant uses 0.026–0.070.
- **Splits.** Three contiguous 2-fold block splits on peptide end residue:
  - `cut45|60`: {5–45} vs {60–101}
  - `cut31|32`: {5–31} vs {32–101}
  - `cut75|75`: {5–75} vs {75–101}

  Each fold trains on one block and is scored on the other. That gives 6 folds per ensemble, each with 60–150 held-out cells.

### Rules (applied per replicate to the candidates with ESS ≥ 10%) `[q3]`

- **A, flags.** z = r/σ_pooled on the held-out cells. n_flags = #|z| > t_n, where t_n is the Šidák threshold at α = 0.05 for the number of held-out cells (t_n = 3.33–3.58 for 60–150 cells; 3.9–4.0 for ISO). Pick the smallest S (most regularised) among the points with the minimum flag count.
- **B, free energy.** F = ½Σ_held-out z² + KL(w‖p0). Pick argmin F. F_rel = F − F(p0).
- **C.** Argmin held-out χ²/cell.
- **C1SE.** The smallest S whose χ² is within one SE of the minimum. SE is the SD of held-out χ²/cell at the argmin over 1000 bootstrap resamples of held-out peptides (residues for ISO).
- **D.** Argmin F within the minimum-flag set.
- **References:** ref_p0 (S = 0) and ref_oracle (maximum recovery in the candidate set).

### ISO control `[q2]`

- **Data.** Self-consistent ISO_BI (874 frames; truth 40/60 open/closed) and ISO_TRI (2225 frames; 1351 unassigned) features. BV(0.35, 2.0) frame_uptake. The residue-level self-consistent target (294 residues × 5 times) plus Gaussian noise at σ = 0.01 and 0.02, with 3 noise seeds each. z is computed with the true noise σ.
- **Prior and sweep.** p0 = uniform. Recovery under p0: BI 62.6, TRI 31.1. S ∈ {0} ∪ logspace(−3, 2.5, 34).
- **Splits.** Contiguous residue blocks cut at 1/2, 1/3 and 2/3 of the 294 residues (2 folds each). That gives 18 replicates per system × σ.
- **Solver.** The exact convex dual of the linear frame_uptake problem: one multiplier per training cell, w ∝ exp(−S·Fᵀλ), solved with L-BFGS. It reproduces the primal logits solution to 6 decimals in J. Median stationarity residual is 5e-9; the maximum is 1e-2, reached only at the largest S `[q6]`.
- **Truth reference.** The true 40/60 weights have ESS 28%. Their held-out χ²/cell ≈ 1.5 at σ = 0.01, because the self-consistent target carries a representation error of about 0.7 σ².
- **Pass criterion.** Recovery ≥ 95% on every replicate. TRI has no pass criterion: its ceiling from unassigned mass is about 62%.

## Verdict

1. **ISO_BI: B, C, D and C1SE pass; A fails at σ = 0.02** `[q3]` (T4).
   - At σ = 0.01, all rules pick 97.4–98.5% mean recovery and 18/18 replicates are ≥ 95%. Minimum per replicate: A 97.5, B/C 97.8, D 97.6, C1SE 96.5.
   - At σ = 0.02, B, C and D give 98.2–98.3% (minimum 96.7–97.0) and C1SE gives 96.5% (minimum 95.5); all are 18/18.
   - **A gives 95.5 ± 1.1 with minimum 93.3, and 5/18 replicates are below 95.** With larger noise the flag count reaches its minimum (often 0) early, at S ≈ 0.3–0.6, ESS 33%. "Most regularised among the minimum" then stops on the rising flank of the recovery curve (`F_ISO_BI_s0.02.png`).
   - B, C and D pick ESS 25–27%, close to the truth's 28%.
2. **ISO_TRI: B, C and D land closest to the ceiling** `[q3]`.
   - σ = 0.01: 59.8–60.1 (oracle within the candidate set 61.5).
   - σ = 0.02: 56.5–56.7 (oracle 57.9).
   - A: 59.8 / 53.7. C1SE: 56.8 / 53.1.
   - The TRI oracle sits at the ESS-10% edge in 17–18 of 18 replicates (T4, `at ESS-10% edge` column). The candidate-set floor therefore binds there; a bigger sweep would get closer to 62.
3. **B is the same rule as C in practice.** At the B pick, KL(w‖p0) is 0–1.7, while ½Σz² is 168–2413. B and C pick the same S in 23/24 MoPrP and 69/72 ISO replicates `[q6]`. For F to differ from held-out χ², the KL term would need a cell-count-comparable weight (for example ½Σz² + n_cells·KL), which is not the rule as specified.
4. **MoPrP: no rule moves the ensemble far from p0. The data term adds about +1.4 recovery points over p0 on average (at most +3.0 in a single fold), and its in-sweep oracle is only 93.6 (MSAss) / 96.5 (Filtered)** `[q3]` (T1).

   | ensemble (σ floor) | p0 | A | B = C | C1SE | D | oracle in sweep | state-level optimum (pop, decoy-free, b′ rate χ²) |
   |---|---|---|---|---|---|---|---|
   | MSAss rec | 91.7 | 92.9 ± 1.2 | 93.1 ± 1.2 | 91.7 (= p0, 6/6) | 93.1 ± 1.2 | 93.6 ± 1.3 | 94.9 |
   | MSAss ESS | 84% | 64 ± 29% (4/6 > 50%) | 50 ± 37% (3/6) | 84% | 53 ± 35% (3/6) | 68% | – |
   | Filtered rec | 96.1 | 96.1 ± 0 (= p0, 6/6) | 94.9 ± 3.1 (min 88.6) | 96.1 (= p0) | 94.9 ± 3.1 (min 88.6) | 96.5 ± 1.0 | 94.2 |
   | Filtered ESS | 91% | 91% (6/6) | 65 ± 40% (4/6) | 91% | 65 ± 40% (4/6) | 82% | – |

   - **Every MoPrP pick is Folded-majority** (Folded ≥ 93.8% in every replicate, all rules, both σ) `[q6]`.
   - Populations at the picks are about 96.7 / 2.8 / 0 / 0.1 / 0.3 (MSAss) and 96.2 / 3.1–3.3 / 0.2–0.3 (Filtered).
   - **Highest and most consistent recovery with ESS > 50%: rule A.** Pooled over both ensembles it gives 94.5 ± 1.8 with ESS 77%, and 10/12 picks have ESS > 50% `[q6]`. But A is close to "pick p0": it picks p0 in 8/12 floored replicates. Its gain over p0 is +1.2 (MSAss) and 0 (Filtered).
   - B, C and D give 94.0 ± 2.5 pooled with ESS 57% (7/12 > 50%). On Filtered they can go *below* p0: cut31|32, held-out N 5–31 gives 88.6 at ESS 11%, because training on 32–101 over-fits PUF1 away.
   - C1SE always returns p0 on MoPrP (24/24) `[q6]`. The peptide-bootstrap SE (1.5–3.8 χ²/cell floored; 3.9–18 unfloored) exceeds the whole χ² dynamic range of the sweep (≤ 1.14 χ²/cell between p0 and the minimum, floored).
   - **Distance from the 94–95% state-level optimum** (`pop/p2_best.csv`; MSAss 94.9, Filtered 94.2, b′ rate, decoys = 0):
     - MSAss lands 1.8–2.0 below with A/B/C/D, and 3.2 below with C1SE/p0. Even the in-sweep oracle is 1.3 below. This is because p0 has removed PUF2 and the reweighting cannot restore it.
     - Filtered lands **above** the state-level optimum: 96.1 for A/C1SE/p0, and 94.9 mean for B/C/D. The ΔG-derived p0 contributes PUF1 3.3 / PUF2 0.3, which the b′ χ² surface alone does not prefer.
5. **Why the flags carry little information on MoPrP** `[q6]`, T3.
   - Held-out χ²/cell is 6–10 even at the best S (floored σ), so the model error is far above noise.
   - 80% of all p0 flags fall in five persistent BV-error peptides, the same ones as in dg/gci2:
     - 60-74: 9 of 15 timepoints in every fold that holds it out
     - 83-90: 6–7
     - 27-31: 6
     - 11-26: 5
     - 68-75: 4
   - The minimum flag count over the sweep is 88% (MSAss) and 100% (Filtered) of the p0 count, i.e. irreducible by reweighting. Unfloored, it is 97% and 100%.
   - So rule A's minimum-flag set is wide, and "most regularised" returns p0 or near-p0.
6. **σ-floor sensitivity** (floored → unfloored, same fold) `[q6]`.

   | rule | mean Δrec | mean \|Δrec\| | max \|Δrec\| | MoPrP pooled rec, floor → unfloored |
   |---|---|---|---|---|
   | C1SE | 0 | 0 | 0 | always p0 |
   | A | −0.22 | 0.22 | 1.16 | 94.5 → 94.3; ESS 77 → 87% |
   | D | −0.66 | 0.68 | 4.35 | 94.0 → 93.4 |
   | B = C | −0.61 | 0.78 | 4.35 | 94.0 → 93.4 |

   Unflooring:
   - raises χ²/cell 3×, to 24–26;
   - doubles the flag counts (MSAss 21 → 37 per fold);
   - makes almost all flags irreducible, which pushes A even closer to p0;
   - lets B/C chase a few low-σ peptides. The 4.35-point drop is Filtered cut75|75, held-out C 75–101: 96.1 → 91.7.

**Bottom line.**
- **D (flags, then F)** is the only rule that passes ISO_BI at both noise levels, stays near the TRI ceiling and is no worse than B/C on MoPrP. That makes it the defensible production rule. B/C are equivalent alternatives.
- **A** has the most consistent MoPrP numbers, but it fails the ISO σ = 0.02 control (5/18 below 95). Its MoPrP consistency mostly comes from returning p0.
- **C1SE** passes ISO but is degenerate on MoPrP: it always picks p0.
- On MoPrP the choice of rule matters by only about 1 recovery point on average. **The p0 (ΔG MaxEnt) is what sets recovery** (91.7 / 96.1). The HDX data, through any of these rules, add about +1.4 on average for MSAss (at most +3.0 in a single fold). They can subtract up to 7.5 (Filtered, B/C/D at ESS ≈ 11%, floored σ), or 8.2 with unfloored σ.
- For a production arm, combine D with an ESS ≥ 50% acceptance check (an acceptance criterion, not a candidate filter: the ISO truth itself is at ESS 28%). Otherwise D's low-ESS Filtered picks (2/6 at 11–15%) are the failure mode.

## Figures `[q4]`

- `F_MoPrP_AF2_MSAss_floor.png`, `F_MoPrP_AF2_filtered_floor.png`: primary.
- `F_MoPrP_AF2_MSAss_unfloored.png`, `F_MoPrP_AF2_filtered_unfloored.png`: σ sensitivity.
- `F_ISO_BI_s0.01.png`, `F_ISO_BI_s0.02.png`, `F_ISO_TRI_s0.01.png`, `F_ISO_TRI_s0.02.png`: noise seed 0 only.

Each figure has rows for recovery, ESS, held-out χ²/cell, flags and F − F(p0) against S. Columns are the three block splits, with both folds overlaid. Rule picks are marked: A ▲, B ■, C ●, C1SE ◆, D ★.

## Caveats

- **Candidate set.** Restricting to ESS ≥ 10% binds for ISO_TRI (the oracle sits at the edge) and for a few MoPrP folds whose B/C/D pick has ESS 11–16%.
- **Block folds are unbalanced.** A held-out block has 4–10 peptides. Training on a C-terminal-only block (`cut45|60` held-out N, `cut75|75` held-out N) cannot inform N-subdomain states, and recovery collapses at large S (to 48–57% at ESS < 10%). The rules correctly stay near p0 there.
- **Single noise realisation per seed for ISO.** With 3 seeds × 6 folds, the ISO pass/fail counts are over 18 replicates.
- **Solver flags.** The L-BFGS "success" flag is False for about 40% of fits: an ABNORMAL line-search stop at ftol = 1e-16 after convergence. Stationarity of w ∝ p0·exp(−S·∂χ²/∂w) is the actual check (MoPrP ≤ 7e-6; ISO median 5e-9, maximum 1e-2 at the largest S) `[q6]`.

---

# Tables (generated by `q5_tables.py` from `q3_selected.csv`)

## T1. MoPrP: rule summary over 6 folds (3 block splits × 2 folds), mean ± SD `[q3]`

| ensemble | σ | rule | S median [min–max] | ESS % | Folded / PUF1 / PUF2 / PUF3 / U (PUF2-like) % | recovery % (min) | held-out χ²/cell | flags | F − F(p0) | rec ≥ 95 | ESS > 50% | picks p0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AF2_MSAss | floor | A_flags | 0.0956 [0 (p0)–1.33] | 64 ± 29 | 96.7 / 2.8 / 0.0 / 0.1 / 0.3 | 92.9 ± 1.2 (91.7) | 7.94 ± 1.74 | 18.8 ± 9.1 | -13.9 ± 18.5 | 0/6 | 4/6 | 2/6 |
| AF2_MSAss | floor | B_F | 0.378 [0.00133–2.37] | 50 ± 37 | 96.7 / 2.9 / 0.0 / 0.1 / 0.3 | 93.1 ± 1.2 (91.7) | 7.89 ± 1.75 | 19.3 ± 9.0 | -16.7 ± 20.8 | 0/6 | 3/6 | 0/6 |
| AF2_MSAss | floor | C_minchi2 | 0.378 [0.00133–2.37] | 49 ± 38 | 96.7 / 2.9 / 0.0 / 0.1 / 0.3 | 93.1 ± 1.2 (91.7) | 7.89 ± 1.76 | 19.2 ± 9.2 | -16.7 ± 20.8 | 0/6 | 3/6 | 0/6 |
| AF2_MSAss | floor | C_1SE | 0 (p0) [0 (p0)–0 (p0)] | 84 ± 0 | 96.4 / 2.8 / 0.0 / 0.2 / 0.6 | 91.7 ± 0.0 (91.7) | 8.25 ± 1.51 | 21.3 ± 10.3 | 0.0 ± 0.0 | 0/6 | 6/6 | 6/6 |
| AF2_MSAss | floor | D_flags_then_F | 0.218 [0.00133–2.37] | 53 ± 35 | 96.8 / 2.8 / 0.0 / 0.1 / 0.3 | 93.1 ± 1.2 (91.7) | 7.90 ± 1.75 | 18.8 ± 9.1 | -16.4 ± 20.9 | 0/6 | 3/6 | 0/6 |
| AF2_MSAss | floor | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 84 ± 0 | 96.4 / 2.8 / 0.0 / 0.2 / 0.6 | 91.7 ± 0.0 (91.7) | 8.25 ± 1.51 | 21.3 ± 10.3 | 0.0 ± 0.0 | 0/6 | 6/6 | 6/6 |
| AF2_MSAss | floor | ref_oracle | 0.117 [0.00316–2.37] | 68 ± 28 | 97.0 / 2.7 / 0.0 / 0.1 / 0.2 | 93.6 ± 1.3 (91.7) | 7.97 ± 1.76 | 19.3 ± 8.7 | -12.7 ± 24.2 | 0/6 | 5/6 | 0/6 |
| AF2_MSAss | unfloored | A_flags | 0.00119 [0 (p0)–0.0422] | 83 ± 1 | 96.6 / 2.8 / 0.0 / 0.2 / 0.4 | 92.5 ± 0.9 (91.7) | 26.24 ± 8.90 | 35.5 ± 17.3 | -16.2 ± 59.0 | 0/6 | 6/6 | 3/6 |
| AF2_MSAss | unfloored | B_F | 0.0125 [0 (p0)–0.422] | 65 ± 30 | 96.6 / 3.0 / 0.0 / 0.1 / 0.3 | 93.1 ± 1.1 (91.7) | 25.61 ± 9.06 | 37.5 ± 16.8 | -46.0 ± 59.1 | 0/6 | 4/6 | 1/6 |
| AF2_MSAss | unfloored | C_minchi2 | 0.0125 [0 (p0)–0.422] | 65 ± 30 | 96.6 / 3.0 / 0.0 / 0.1 / 0.3 | 93.1 ± 1.1 (91.7) | 25.61 ± 9.06 | 37.5 ± 16.8 | -46.0 ± 59.1 | 0/6 | 4/6 | 1/6 |
| AF2_MSAss | unfloored | C_1SE | 0 (p0) [0 (p0)–0 (p0)] | 84 ± 0 | 96.4 / 2.8 / 0.0 / 0.2 / 0.6 | 91.7 ± 0.0 (91.7) | 26.47 ± 8.92 | 36.7 ± 17.0 | 0.0 ± 0.0 | 0/6 | 6/6 | 6/6 |
| AF2_MSAss | unfloored | D_flags_then_F | 0.00442 [0 (p0)–0.0422] | 83 ± 1 | 96.7 / 2.8 / 0.0 / 0.1 / 0.4 | 92.7 ± 1.0 (91.7) | 26.08 ± 9.08 | 35.5 ± 17.3 | -22.2 ± 59.1 | 0/6 | 6/6 | 1/6 |
| AF2_MSAss | unfloored | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 84 ± 0 | 96.4 / 2.8 / 0.0 / 0.2 / 0.6 | 91.7 ± 0.0 (91.7) | 26.47 ± 8.92 | 36.7 ± 17.0 | 0.0 ± 0.0 | 0/6 | 6/6 | 6/6 |
| AF2_MSAss | unfloored | ref_oracle | 0.0185 [0.00133–1.33] | 70 ± 29 | 97.0 / 2.7 / 0.0 / 0.1 / 0.2 | 93.6 ± 1.3 (91.8) | 26.01 ± 9.14 | 38.2 ± 17.9 | -25.8 ± 69.7 | 0/6 | 5/6 | 0/6 |
| AF2_filtered | floor | A_flags | 0 (p0) [0 (p0)–0 (p0)] | 91 ± 0 | 96.2 / 3.3 / 0.3 / 0.0 / 0.0 (0.17) | 96.1 ± 0.0 (96.1) | 7.67 ± 1.72 | 16.5 ± 7.1 | 0.0 ± 0.0 | 6/6 | 6/6 | 6/6 |
| AF2_filtered | floor | B_F | 0 (p0) [0 (p0)–1.78] | 65 ± 40 | 96.2 / 3.1 / 0.2 / 0.0 / 0.0 (0.42) | 94.9 ± 3.1 (88.6) | 7.60 ± 1.75 | 16.5 ± 7.1 | -3.0 ± 4.9 | 5/6 | 4/6 | 4/6 |
| AF2_filtered | floor | C_minchi2 | 0 (p0) [0 (p0)–1.78] | 65 ± 40 | 96.2 / 3.1 / 0.2 / 0.0 / 0.0 (0.42) | 94.9 ± 3.1 (88.6) | 7.60 ± 1.75 | 16.5 ± 7.1 | -3.0 ± 4.9 | 5/6 | 4/6 | 4/6 |
| AF2_filtered | floor | C_1SE | 0 (p0) [0 (p0)–0 (p0)] | 91 ± 0 | 96.2 / 3.3 / 0.3 / 0.0 / 0.0 (0.17) | 96.1 ± 0.0 (96.1) | 7.67 ± 1.72 | 16.5 ± 7.1 | 0.0 ± 0.0 | 6/6 | 6/6 | 6/6 |
| AF2_filtered | floor | D_flags_then_F | 0 (p0) [0 (p0)–1.78] | 65 ± 40 | 96.2 / 3.1 / 0.2 / 0.0 / 0.0 (0.42) | 94.9 ± 3.1 (88.6) | 7.60 ± 1.75 | 16.5 ± 7.1 | -3.0 ± 4.9 | 5/6 | 4/6 | 4/6 |
| AF2_filtered | floor | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 91 ± 0 | 96.2 / 3.3 / 0.3 / 0.0 / 0.0 (0.17) | 96.1 ± 0.0 (96.1) | 7.67 ± 1.72 | 16.5 ± 7.1 | 0.0 ± 0.0 | 6/6 | 6/6 | 6/6 |
| AF2_filtered | floor | ref_oracle | 0 (p0) [0 (p0)–1] | 82 ± 22 | 96.4 / 3.1 / 0.3 / 0.0 / 0.0 (0.14) | 96.5 ± 1.0 (96.1) | 7.66 ± 1.71 | 16.5 ± 7.1 | -0.5 ± 1.1 | 6/6 | 5/6 | 5/6 |
| AF2_filtered | unfloored | A_flags | 0 (p0) [0 (p0)–0 (p0)] | 91 ± 0 | 96.2 / 3.3 / 0.3 / 0.0 / 0.0 (0.17) | 96.1 ± 0.0 (96.1) | 24.84 ± 9.71 | 36.3 ± 14.6 | 0.0 ± 0.0 | 6/6 | 6/6 | 6/6 |
| AF2_filtered | unfloored | B_F | 0.0375 [0 (p0)–1.33] | 59 ± 39 | 96.0 / 3.2 / 0.2 / 0.0 / 0.0 (0.56) | 93.7 ± 3.3 (87.9) | 24.72 ± 9.67 | 36.7 ± 15.1 | -7.2 ± 14.8 | 3/6 | 4/6 | 3/6 |
| AF2_filtered | unfloored | C_minchi2 | 0.0375 [0 (p0)–1.33] | 59 ± 39 | 96.0 / 3.2 / 0.2 / 0.0 / 0.0 (0.56) | 93.7 ± 3.3 (87.9) | 24.72 ± 9.67 | 36.7 ± 15.1 | -7.2 ± 14.8 | 3/6 | 4/6 | 3/6 |
| AF2_filtered | unfloored | C_1SE | 0 (p0) [0 (p0)–0 (p0)] | 91 ± 0 | 96.2 / 3.3 / 0.3 / 0.0 / 0.0 (0.17) | 96.1 ± 0.0 (96.1) | 24.84 ± 9.71 | 36.3 ± 14.6 | 0.0 ± 0.0 | 6/6 | 6/6 | 6/6 |
| AF2_filtered | unfloored | D_flags_then_F | 0.0375 [0 (p0)–0.562] | 63 ± 34 | 95.9 / 3.3 / 0.2 / 0.0 / 0.0 (0.56) | 94.0 ± 3.5 (87.9) | 24.74 ± 9.70 | 36.3 ± 14.6 | -5.3 ± 10.2 | 4/6 | 4/6 | 3/6 |
| AF2_filtered | unfloored | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 91 ± 0 | 96.2 / 3.3 / 0.3 / 0.0 / 0.0 (0.17) | 96.1 ± 0.0 (96.1) | 24.84 ± 9.71 | 36.3 ± 14.6 | 0.0 ± 0.0 | 6/6 | 6/6 | 6/6 |
| AF2_filtered | unfloored | ref_oracle | 0 (p0) [0 (p0)–0.316] | 85 ± 14 | 96.3 / 3.2 / 0.3 / 0.0 / 0.0 (0.16) | 96.3 ± 0.5 (96.1) | 24.83 ± 9.70 | 36.3 ± 14.6 | -0.4 ± 0.9 | 6/6 | 6/6 | 5/6 |

## T2a. MoPrP per fold, σ floor `[q3]`

Cells: S · ESS% · rec% · χ²/cell · flags. Held-out block is named; training = the other block.

| ensemble | split | held-out | n cells (t_n) | min flags | A_flags | B_F | C_minchi2 | C_1SE | D_flags_then_F | ref_p0 | ref_oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AF2_MSAss | cut31|32 | C 32-101 | 150 (3.58) | 21 | 0.75 · 47 · 94.0 · 8.74 · 21 | 2.37 · 13 · 94.7 · 8.60 · 21 | 2.37 · 13 · 94.7 · 8.60 · 21 | 0 (p0) · 84 · 91.7 · 9.29 · 30 | 2.37 · 13 · 94.7 · 8.60 · 21 | 0 (p0) · 84 · 91.7 · 9.29 · 30 | 2.37 · 13 · 94.7 · 8.60 · 21 |
| AF2_MSAss | cut31|32 | N 5-31 | 60 (3.33) | 10 | 1.33 · 12 · 94.1 · 5.59 · 10 | 1 · 20 · 94.2 · 5.59 · 11 | 1.33 · 12 · 94.1 · 5.59 · 10 | 0 (p0) · 84 · 91.7 · 6.73 · 12 | 1.33 · 12 · 94.1 · 5.59 · 10 | 0 (p0) · 84 · 91.7 · 6.73 · 12 | 0.316 · 67 · 94.2 · 5.63 · 11 |
| AF2_MSAss | cut45|60 | C 60-101 | 120 (3.52) | 20 | 0 (p0) · 84 · 91.7 · 9.30 · 20 | 0.00133 · 83 · 92.4 · 9.29 · 20 | 0.00133 · 83 · 92.4 · 9.29 · 20 | 0 (p0) · 84 · 91.7 · 9.30 · 20 | 0.00133 · 83 · 92.4 · 9.29 · 20 | 0 (p0) · 84 · 91.7 · 9.30 · 20 | 0.1 · 81 · 94.9 · 9.56 · 21 |
| AF2_MSAss | cut45|60 | N 5-45 | 90 (3.45) | 23 | 0 (p0) · 84 · 91.7 · 7.58 · 23 | 0.00237 · 84 · 91.7 · 7.56 · 23 | 0.00237 · 84 · 91.7 · 7.56 · 23 | 0 (p0) · 84 · 91.7 · 7.58 · 23 | 0.00237 · 84 · 91.7 · 7.56 · 23 | 0 (p0) · 84 · 91.7 · 7.58 · 23 | 0.00316 · 84 · 91.7 · 7.56 · 23 |
| AF2_MSAss | cut75|75 | C 75-101 | 90 (3.45) | 7 | 0.178 · 74 · 93.9 · 6.35 · 7 | 0.75 · 16 · 93.5 · 6.22 · 8 | 0.75 · 16 · 93.5 · 6.22 · 8 | 0 (p0) · 84 · 91.7 · 6.51 · 8 | 0.422 · 43 · 93.8 · 6.26 · 7 | 0 (p0) · 84 · 91.7 · 6.51 · 8 | 0.133 · 78 · 93.9 · 6.38 · 8 |
| AF2_MSAss | cut75|75 | N 5-75 | 120 (3.52) | 32 | 0.0133 · 84 · 92.1 · 10.07 · 32 | 0.00562 · 84 · 92.0 · 10.07 · 33 | 0.00562 · 84 · 92.0 · 10.07 · 33 | 0 (p0) · 84 · 91.7 · 10.10 · 35 | 0.0133 · 84 · 92.1 · 10.07 · 32 | 0 (p0) · 84 · 91.7 · 10.10 · 35 | 0.0178 · 84 · 92.1 · 10.07 · 32 |
| AF2_filtered | cut31|32 | C 32-101 | 150 (3.58) | 21 | 0 (p0) · 91 · 96.1 · 8.83 · 21 | 1.78 · 15 · 96.8 · 8.66 · 21 | 1.78 · 15 · 96.8 · 8.66 · 21 | 0 (p0) · 91 · 96.1 · 8.83 · 21 | 1.78 · 15 · 96.8 · 8.66 · 21 | 0 (p0) · 91 · 96.1 · 8.83 · 21 | 1 · 36 · 98.5 · 8.78 · 21 |
| AF2_filtered | cut31|32 | N 5-31 | 60 (3.33) | 11 | 0 (p0) · 91 · 96.1 · 6.01 · 11 | 1.33 · 11 · 88.6 · 5.76 · 11 | 1.33 · 11 · 88.6 · 5.76 · 11 | 0 (p0) · 91 · 96.1 · 6.01 · 11 | 1.33 · 11 · 88.6 · 5.76 · 11 | 0 (p0) · 91 · 96.1 · 6.01 · 11 | 0 (p0) · 91 · 96.1 · 6.01 · 11 |
| AF2_filtered | cut45|60 | C 60-101 | 120 (3.52) | 20 | 0 (p0) · 91 · 96.1 · 9.50 · 20 | 0 (p0) · 91 · 96.1 · 9.50 · 20 | 0 (p0) · 91 · 96.1 · 9.50 · 20 | 0 (p0) · 91 · 96.1 · 9.50 · 20 | 0 (p0) · 91 · 96.1 · 9.50 · 20 | 0 (p0) · 91 · 96.1 · 9.50 · 20 | 0 (p0) · 91 · 96.1 · 9.50 · 20 |
| AF2_filtered | cut45|60 | N 5-45 | 90 (3.45) | 14 | 0 (p0) · 91 · 96.1 · 6.05 · 14 | 0 (p0) · 91 · 96.1 · 6.05 · 14 | 0 (p0) · 91 · 96.1 · 6.05 · 14 | 0 (p0) · 91 · 96.1 · 6.05 · 14 | 0 (p0) · 91 · 96.1 · 6.05 · 14 | 0 (p0) · 91 · 96.1 · 6.05 · 14 | 0 (p0) · 91 · 96.1 · 6.05 · 14 |
| AF2_filtered | cut75|75 | C 75-101 | 90 (3.45) | 7 | 0 (p0) · 91 · 96.1 · 6.29 · 7 | 0 (p0) · 91 · 96.1 · 6.29 · 7 | 0 (p0) · 91 · 96.1 · 6.29 · 7 | 0 (p0) · 91 · 96.1 · 6.29 · 7 | 0 (p0) · 91 · 96.1 · 6.29 · 7 | 0 (p0) · 91 · 96.1 · 6.29 · 7 | 0 (p0) · 91 · 96.1 · 6.29 · 7 |
| AF2_filtered | cut75|75 | N 5-75 | 120 (3.52) | 26 | 0 (p0) · 91 · 96.1 · 9.32 · 26 | 0 (p0) · 91 · 96.1 · 9.32 · 26 | 0 (p0) · 91 · 96.1 · 9.32 · 26 | 0 (p0) · 91 · 96.1 · 9.32 · 26 | 0 (p0) · 91 · 96.1 · 9.32 · 26 | 0 (p0) · 91 · 96.1 · 9.32 · 26 | 0 (p0) · 91 · 96.1 · 9.32 · 26 |

## T2b. MoPrP per fold, σ unfloored `[q3]`

Cells: S · ESS% · rec% · χ²/cell · flags. Held-out block is named; training = the other block.

| ensemble | split | held-out | n cells (t_n) | min flags | A_flags | B_F | C_minchi2 | C_1SE | D_flags_then_F | ref_p0 | ref_oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AF2_MSAss | cut31|32 | C 32-101 | 150 (3.58) | 51 | 0.0422 · 83 · 93.5 · 32.67 · 51 | 0.422 · 42 · 94.3 · 32.17 · 55 | 0.422 · 42 · 94.3 · 32.17 · 55 | 0 (p0) · 84 · 91.7 · 34.32 · 54 | 0.0422 · 83 · 93.5 · 32.67 · 51 | 0 (p0) · 84 · 91.7 · 34.32 · 54 | 1.33 · 12 · 94.9 · 32.24 · 56 |
| AF2_MSAss | cut31|32 | N 5-31 | 60 (3.33) | 18 | 0.00237 · 83 · 93.0 · 13.00 · 18 | 0.0237 · 81 · 94.3 · 12.25 · 19 | 0.0237 · 81 · 94.3 · 12.25 · 19 | 0 (p0) · 84 · 91.7 · 13.92 · 21 | 0.0075 · 82 · 93.9 · 12.42 · 18 | 0 (p0) · 84 · 91.7 · 13.92 · 21 | 0.0562 · 74 · 94.3 · 12.29 · 19 |
| AF2_MSAss | cut45|60 | C 60-101 | 120 (3.52) | 33 | 0 (p0) · 84 · 91.7 · 35.98 · 33 | 0.00133 · 83 · 93.2 · 35.69 · 36 | 0.00133 · 83 · 93.2 · 35.69 · 36 | 0 (p0) · 84 · 91.7 · 35.98 · 33 | 0.000316 · 84 · 92.1 · 35.84 · 33 | 0 (p0) · 84 · 91.7 · 35.98 · 33 | 0.0237 · 82 · 94.8 · 36.16 · 42 |
| AF2_MSAss | cut45|60 | N 5-45 | 90 (3.45) | 40 | 0 (p0) · 84 · 91.7 · 18.51 · 40 | 0.00133 · 84 · 91.8 · 18.28 · 40 | 0.00133 · 84 · 91.8 · 18.28 · 40 | 0 (p0) · 84 · 91.7 · 18.51 · 40 | 0.00133 · 84 · 91.8 · 18.28 · 40 | 0 (p0) · 84 · 91.7 · 18.51 · 40 | 0.00133 · 84 · 91.8 · 18.28 · 40 |
| AF2_MSAss | cut75|75 | C 75-101 | 90 (3.45) | 14 | 0.0237 · 81 · 93.3 · 26.09 · 14 | 0.237 · 14 · 93.1 · 24.05 · 18 | 0.237 · 14 · 93.1 · 24.05 · 18 | 0 (p0) · 84 · 91.7 · 24.89 · 15 | 0.0237 · 81 · 93.3 · 26.09 · 14 | 0 (p0) · 84 · 91.7 · 24.89 · 15 | 0.01 · 83 · 93.4 · 25.84 · 15 |
| AF2_MSAss | cut75|75 | N 5-75 | 120 (3.52) | 57 | 0 (p0) · 84 · 91.7 · 31.19 · 57 | 0 (p0) · 84 · 91.7 · 31.19 · 57 | 0 (p0) · 84 · 91.7 · 31.19 · 57 | 0 (p0) · 84 · 91.7 · 31.19 · 57 | 0 (p0) · 84 · 91.7 · 31.19 · 57 | 0 (p0) · 84 · 91.7 · 31.19 · 57 | 0.0133 · 84 · 92.3 · 31.28 · 57 |
| AF2_filtered | cut31|32 | C 32-101 | 150 (3.58) | 54 | 0 (p0) · 91 · 96.1 · 32.68 · 54 | 1.33 · 10 · 94.7 · 32.16 · 56 | 1.33 · 10 · 94.7 · 32.16 · 56 | 0 (p0) · 91 · 96.1 · 32.68 · 54 | 0.562 · 32 · 96.5 · 32.32 · 54 | 0 (p0) · 91 · 96.1 · 32.68 · 54 | 0.316 · 56 · 97.3 · 32.64 · 54 |
| AF2_filtered | cut31|32 | N 5-31 | 60 (3.33) | 18 | 0 (p0) · 91 · 96.1 · 12.61 · 18 | 0.316 · 13 · 87.9 · 12.43 · 18 | 0.316 · 13 · 87.9 · 12.43 · 18 | 0 (p0) · 91 · 96.1 · 12.61 · 18 | 0.316 · 13 · 87.9 · 12.43 · 18 | 0 (p0) · 91 · 96.1 · 12.61 · 18 | 0 (p0) · 91 · 96.1 · 12.61 · 18 |
| AF2_filtered | cut45|60 | C 60-101 | 120 (3.52) | 40 | 0 (p0) · 91 · 96.1 · 36.19 · 40 | 0 (p0) · 91 · 96.1 · 36.19 · 40 | 0 (p0) · 91 · 96.1 · 36.19 · 40 | 0 (p0) · 91 · 96.1 · 36.19 · 40 | 0 (p0) · 91 · 96.1 · 36.19 · 40 | 0 (p0) · 91 · 96.1 · 36.19 · 40 | 0 (p0) · 91 · 96.1 · 36.19 · 40 |
| AF2_filtered | cut45|60 | N 5-45 | 90 (3.45) | 34 | 0 (p0) · 91 · 96.1 · 14.61 · 34 | 0 (p0) · 91 · 96.1 · 14.61 · 34 | 0 (p0) · 91 · 96.1 · 14.61 · 34 | 0 (p0) · 91 · 96.1 · 14.61 · 34 | 0 (p0) · 91 · 96.1 · 14.61 · 34 | 0 (p0) · 91 · 96.1 · 14.61 · 34 | 0 (p0) · 91 · 96.1 · 14.61 · 34 |
| AF2_filtered | cut75|75 | C 75-101 | 90 (3.45) | 22 | 0 (p0) · 91 · 96.1 · 23.21 · 22 | 0.075 · 62 · 91.7 · 23.17 · 22 | 0.075 · 62 · 91.7 · 23.17 · 22 | 0 (p0) · 91 · 96.1 · 23.21 · 22 | 0.075 · 62 · 91.7 · 23.17 · 22 | 0 (p0) · 91 · 96.1 · 23.21 · 22 | 0 (p0) · 91 · 96.1 · 23.21 · 22 |
| AF2_filtered | cut75|75 | N 5-75 | 120 (3.52) | 50 | 0 (p0) · 91 · 96.1 · 29.74 · 50 | 0 (p0) · 91 · 96.1 · 29.74 · 50 | 0 (p0) · 91 · 96.1 · 29.74 · 50 | 0 (p0) · 91 · 96.1 · 29.74 · 50 | 0 (p0) · 91 · 96.1 · 29.74 · 50 | 0 (p0) · 91 · 96.1 · 29.74 · 50 | 0 (p0) · 91 · 96.1 · 29.74 · 50 |

## T3. MoPrP flagged peptides (σ floor) at p0 and at each rule's pick `[q3]`

Format: peptide(number of flagged timepoints).

| ensemble | held-out | p0 | A | B | C | C1SE | D |
|---|---|---|---|---|---|---|---|
| AF2_MSAss | cut31|32 C 32-101 | 32-45(5);33-40(6);60-74(9);68-75(4);83-90(6) | 33-40(2);60-74(9);68-75(4);83-90(6) | 33-40(1);60-74(9);68-75(4);75-82(1);83-90(6) | 33-40(1);60-74(9);68-75(4);75-82(1);83-90(6) | 32-45(5);33-40(6);60-74(9);68-75(4);83-90(6) | 33-40(1);60-74(9);68-75(4);75-82(1);83-90(6) |
| AF2_MSAss | cut31|32 N 5-31 | 11-24(1);11-26(5);27-31(6) | 11-26(5);27-31(5) | 11-26(5);27-31(6) | 11-26(5);27-31(5) | 11-24(1);11-26(5);27-31(6) | 11-26(5);27-31(5) |
| AF2_MSAss | cut45|60 C 60-101 | 60-74(9);68-75(4);83-90(7) | 60-74(9);68-75(4);83-90(7) | 60-74(9);68-75(4);82-101(1);83-90(6) | 60-74(9);68-75(4);82-101(1);83-90(6) | 60-74(9);68-75(4);83-90(7) | 60-74(9);68-75(4);82-101(1);83-90(6) |
| AF2_MSAss | cut45|60 N 5-45 | 11-26(5);27-31(6);32-45(6);33-40(6) | 11-26(5);27-31(6);32-45(6);33-40(6) | 11-26(5);27-31(6);32-45(6);33-40(6) | 11-26(5);27-31(6);32-45(6);33-40(6) | 11-26(5);27-31(6);32-45(6);33-40(6) | 11-26(5);27-31(6);32-45(6);33-40(6) |
| AF2_MSAss | cut75|75 C 75-101 | 82-101(1);83-90(7) | 82-101(1);83-90(6) | 82-101(2);83-90(6) | 82-101(2);83-90(6) | 82-101(1);83-90(7) | 82-101(1);83-90(6) |
| AF2_MSAss | cut75|75 N 5-75 | 11-26(5);27-31(6);32-45(5);33-40(6);60-74(9);68-75(4) | 11-26(5);27-31(6);32-45(4);33-40(4);60-74(9);68-75(4) | 11-26(5);27-31(6);32-45(5);33-40(4);60-74(9);68-75(4) | 11-26(5);27-31(6);32-45(5);33-40(4);60-74(9);68-75(4) | 11-26(5);27-31(6);32-45(5);33-40(6);60-74(9);68-75(4) | 11-26(5);27-31(6);32-45(4);33-40(4);60-74(9);68-75(4) |
| AF2_filtered | cut31|32 C 32-101 | 33-40(2);60-74(9);68-75(4);83-90(6) | 33-40(2);60-74(9);68-75(4);83-90(6) | 33-40(2);60-74(9);68-75(4);83-90(6) | 33-40(2);60-74(9);68-75(4);83-90(6) | 33-40(2);60-74(9);68-75(4);83-90(6) | 33-40(2);60-74(9);68-75(4);83-90(6) |
| AF2_filtered | cut31|32 N 5-31 | 11-26(5);27-31(6) | 11-26(5);27-31(6) | 11-26(5);27-31(6) | 11-26(5);27-31(6) | 11-26(5);27-31(6) | 11-26(5);27-31(6) |
| AF2_filtered | cut45|60 C 60-101 | 60-74(9);68-75(4);82-101(1);83-90(6) | 60-74(9);68-75(4);82-101(1);83-90(6) | 60-74(9);68-75(4);82-101(1);83-90(6) | 60-74(9);68-75(4);82-101(1);83-90(6) | 60-74(9);68-75(4);82-101(1);83-90(6) | 60-74(9);68-75(4);82-101(1);83-90(6) |
| AF2_filtered | cut45|60 N 5-45 | 11-26(5);27-31(6);33-40(3) | 11-26(5);27-31(6);33-40(3) | 11-26(5);27-31(6);33-40(3) | 11-26(5);27-31(6);33-40(3) | 11-26(5);27-31(6);33-40(3) | 11-26(5);27-31(6);33-40(3) |
| AF2_filtered | cut75|75 C 75-101 | 82-101(1);83-90(6) | 82-101(1);83-90(6) | 82-101(1);83-90(6) | 82-101(1);83-90(6) | 82-101(1);83-90(6) | 82-101(1);83-90(6) |
| AF2_filtered | cut75|75 N 5-75 | 11-26(5);27-31(6);33-40(2);60-74(9);68-75(4) | 11-26(5);27-31(6);33-40(2);60-74(9);68-75(4) | 11-26(5);27-31(6);33-40(2);60-74(9);68-75(4) | 11-26(5);27-31(6);33-40(2);60-74(9);68-75(4) | 11-26(5);27-31(6);33-40(2);60-74(9);68-75(4) | 11-26(5);27-31(6);33-40(2);60-74(9);68-75(4) |

## T4. ISO control: rule summary over 3 noise seeds × 3 block splits × 2 folds = 18 replicates, mean ± SD `[q3]`

| system | noise σ | rule | S median [min–max] | ESS % | open / closed / unassigned % | recovery % (min) | rec ≥ 95 | held-out χ²/cell | flags | at ESS-10% edge |
|---|---|---|---|---|---|---|---|---|---|---|
| ISO_BI | 0.01 | A_flags | 0.464 [0.215–3.16] | 28 ± 2 | 37.6 / 62.4 / 0.0 | 97.9 ± 0.3 (97.5) | 18/18 | 1.60 ± 0.07 | 2.6 ± 1.7 | 0.00 |
| ISO_BI | 0.01 | B_F | 0.681 [0.316–1.47] | 27 ± 1 | 38.3 / 61.7 / 0.0 | 98.5 ± 0.3 (97.8) | 18/18 | 1.57 ± 0.08 | 2.9 ± 1.7 | 0.00 |
| ISO_BI | 0.01 | C_minchi2 | 0.681 [0.316–1.47] | 27 ± 1 | 38.3 / 61.7 / 0.0 | 98.5 ± 0.3 (97.8) | 18/18 | 1.57 ± 0.08 | 2.9 ± 1.7 | 0.00 |
| ISO_BI | 0.01 | C_1SE | 0.316 [0.147–0.681] | 30 ± 1 | 37.0 / 63.0 / 0.0 | 97.4 ± 0.4 (96.5) | 18/18 | 1.64 ± 0.09 | 4.1 ± 2.6 | 0.00 |
| ISO_BI | 0.01 | D_flags_then_F | 0.681 [0.215–3.16] | 27 ± 2 | 38.1 / 61.9 / 0.0 | 98.4 ± 0.4 (97.6) | 18/18 | 1.58 ± 0.08 | 2.6 ± 1.7 | 0.00 |
| ISO_BI | 0.01 | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 100 ± 0 | 5.0 / 95.0 / 0.0 | 62.6 ± 0.0 (62.6) | 0/18 | 28.12 ± 2.59 | 122.8 ± 39.2 | 0.00 |
| ISO_BI | 0.01 | ref_oracle | 68.1 [4.64–316] | 15 ± 4 | 39.6 / 60.4 / 0.0 | 99.7 ± 0.4 (98.5) | 18/18 | 2.19 ± 0.33 | 15.8 ± 9.7 | 0.44 |
| ISO_BI | 0.02 | A_flags | 0.573 [0.316–1.47] | 33 ± 3 | 34.8 / 65.2 / 0.0 | 95.5 ± 1.1 (93.3) | 13/18 | 1.24 ± 0.05 | 0.4 ± 0.7 | 0.00 |
| ISO_BI | 0.02 | B_F | 2.15 [1.47–3.16] | 25 ± 2 | 38.0 / 62.0 / 0.0 | 98.2 ± 0.5 (97.0) | 18/18 | 1.15 ± 0.05 | 0.4 ± 0.7 | 0.00 |
| ISO_BI | 0.02 | C_minchi2 | 2.15 [1.47–3.16] | 25 ± 2 | 38.0 / 62.0 / 0.0 | 98.3 ± 0.5 (97.0) | 18/18 | 1.15 ± 0.05 | 0.4 ± 0.7 | 0.00 |
| ISO_BI | 0.02 | C_1SE | 0.841 [0.464–1.47] | 31 ± 2 | 36.0 / 64.0 / 0.0 | 96.5 ± 0.5 (95.5) | 18/18 | 1.19 ± 0.05 | 0.5 ± 0.9 | 0.00 |
| ISO_BI | 0.02 | D_flags_then_F | 2.15 [1–3.16] | 25 ± 2 | 37.9 / 62.1 / 0.0 | 98.2 ± 0.6 (96.7) | 18/18 | 1.15 ± 0.05 | 0.4 ± 0.7 | 0.00 |
| ISO_BI | 0.02 | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 100 ± 0 | 5.0 / 95.0 / 0.0 | 62.6 ± 0.0 (62.6) | 0/18 | 7.74 ± 0.52 | 68.1 ± 23.4 | 0.00 |
| ISO_BI | 0.02 | ref_oracle | 49.9 [10–316] | 12 ± 2 | 39.5 / 60.5 / 0.0 | 99.5 ± 0.7 (97.2) | 18/18 | 1.34 ± 0.10 | 1.4 ± 1.3 | 0.28 |
| ISO_TRI | 0.01 | A_flags | 3.16 [1–10] | 14 ± 2 | 26.2 / 45.1 / 28.7 | 59.8 ± 3.2 (54.2) | 0/18 | 2.20 ± 0.55 | 14.3 ± 13.2 | 0.11 |
| ISO_TRI | 0.01 | B_F | 4.64 [1–6.81] | 14 ± 1 | 26.2 / 45.2 / 28.7 | 59.8 ± 3.1 (54.9) | 0/18 | 2.18 ± 0.55 | 15.2 ± 13.7 | 0.00 |
| ISO_TRI | 0.01 | C_minchi2 | 4.64 [1–6.81] | 14 ± 1 | 26.2 / 45.2 / 28.7 | 59.8 ± 3.1 (54.9) | 0/18 | 2.18 ± 0.55 | 15.2 ± 13.7 | 0.00 |
| ISO_TRI | 0.01 | C_1SE | 1.47 [0.681–2.15] | 18 ± 2 | 24.0 / 43.4 / 32.6 | 56.8 ± 2.8 (52.4) | 0/18 | 2.33 ± 0.64 | 18.9 ± 15.6 | 0.00 |
| ISO_TRI | 0.01 | D_flags_then_F | 4.64 [1–10] | 13 ± 2 | 26.3 / 45.3 / 28.4 | 60.1 ± 3.3 (54.6) | 0/18 | 2.20 ± 0.55 | 14.3 ± 13.2 | 0.11 |
| ISO_TRI | 0.01 | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 100 ± 0 | 2.0 / 37.3 / 60.7 | 31.1 ± 0.0 (31.1) | 0/18 | 14.18 ± 0.95 | 114.2 ± 33.3 | 0.00 |
| ISO_TRI | 0.01 | ref_oracle | 10 [4.64–10] | 11 ± 0 | 27.4 / 46.0 / 26.6 | 61.5 ± 3.7 (55.4) | 0/18 | 2.27 ± 0.54 | 16.3 ± 13.8 | 1.00 |
| ISO_TRI | 0.02 | A_flags | 2.15 [1–14.7] | 19 ± 4 | 21.8 / 41.8 / 36.4 | 53.7 ± 2.9 (48.7) | 0/18 | 1.42 ± 0.18 | 2.9 ± 3.9 | 0.06 |
| ISO_TRI | 0.02 | B_F | 4.64 [3.16–10] | 14 ± 2 | 24.0 / 43.2 / 32.9 | 56.5 ± 3.6 (50.0) | 0/18 | 1.38 ± 0.20 | 3.0 ± 4.0 | 0.17 |
| ISO_TRI | 0.02 | C_minchi2 | 6.81 [3.16–10] | 14 ± 2 | 24.1 / 43.3 / 32.7 | 56.7 ± 3.8 (50.0) | 0/18 | 1.38 ± 0.20 | 3.0 ± 4.0 | 0.17 |
| ISO_TRI | 0.02 | C_1SE | 2.15 [1.47–4.64] | 21 ± 3 | 21.2 / 41.5 / 37.2 | 53.1 ± 2.9 (47.3) | 0/18 | 1.43 ± 0.21 | 3.8 ± 4.9 | 0.00 |
| ISO_TRI | 0.02 | D_flags_then_F | 4.64 [2.15–14.7] | 14 ± 3 | 23.9 / 43.2 / 32.9 | 56.5 ± 3.7 (48.7) | 0/18 | 1.38 ± 0.20 | 2.9 ± 3.9 | 0.22 |
| ISO_TRI | 0.02 | ref_p0 | 0 (p0) [0 (p0)–0 (p0)] | 100 ± 0 | 2.0 / 37.3 / 60.7 | 31.1 ± 0.0 (31.1) | 0/18 | 4.28 ± 0.31 | 45.5 ± 10.4 | 0.00 |
| ISO_TRI | 0.02 | ref_oracle | 10 [6.81–14.7] | 11 ± 1 | 25.1 / 43.8 / 31.1 | 57.9 ± 4.1 (50.0) | 0/18 | 1.39 ± 0.20 | 2.9 ± 4.1 | 0.94 |
