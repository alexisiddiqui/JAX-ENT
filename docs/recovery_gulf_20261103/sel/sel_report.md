# sel: Selection functions on the existing MoPrP and ISO candidate pools

This pass is read-only: no repo edits and no fits. Every script lives in this folder and runs from the repo root with `.venv/bin/python`. Numbers carry the tag of the script that produced them:

| tag | script | what it does |
|---|---|---|
| `[s0]` | `s0_common.py` | Shared paths and the P2 filter. Also the new-feature NumPy forward pass, which reuses `gci2/c0_common.py`. |
| `[s1]` | `s1_extract_moprp.py` | MoPrP P2 weights. Computes predictions through the native jaxent `predict` on the fit's own features (`_featurise`, 12-peptide native sparse map, the arm's averaging mode), and through NumPy on `_featurise_a101_k109` with the row-normalised map (14 peptides). Replays the existing scores (see Faithfulness). |
| `[s2]` | `s2_extract_iso.py` | ISO P2 weights and predictions in the exact `run_iso_policy_sidecars.analyze()` context. Replays `val_mse`. |
| `[s3]` | `s3_score.py` | All score functions on every evaluation scope. Output goes to `work/s3_scores_*.parquet`. |
| `[s4]` | `s4_select.py` | Selection per split replicate, plus the within-pool Spearman correlations. Writes `sel_selected.csv`, `sel_summary.csv` (n = 3 per split type), `sel_summary6.csv` (n = 6), `sel_spearman*.csv`. |
| `[s5]` | `s5_report.py` | Verdict and ISO tables (`sel_verdict.csv`, `sel_iso_check.csv`, `sel_pops_maxent.csv`, `tables_*.md`) and figures. |
| `[s6]` | `s6_highrec_check.py` | Characterises the candidates that already sit above 80% in the pools (`s6_highrec.csv`). |

## Setup

**Candidate pools (policy P2).** I dropped `candidate_kind == running_best` and `step < 100`.

| arm | P2 candidates |
|---|---|
| Ex2 full-uptake | 31,491 |
| Ex2 rate | 31,501 |
| ISO | 46,231 |

**Selection unit.** One model in one split replicate. The minimum is taken over every S, bandwidth and saved step of that model; ties are broken by S, h, step and candidate index.

- **MoPrP models:** MaxEnt (MSAss, Filtered); OMC work-scale and dihedral-RMSD, each with k = 0 and k = 20 (MSAss only).
- **ISO models:** BI MaxEnt; TRI MaxEnt; TRI OMC with pyrosetta, rmsd and work_scale graphs.
- **Pool size per replicate:** about 730 candidates for MaxEnt and about 1,100 for each OMC model `[s4]`.

**Evaluation scopes for MoPrP.** Selection is done separately on each scope:

| scope | features and map | peptides |
|---|---|---|
| `old_val` (**primary, fit-faithful**) | `_featurise`, native 12-peptide map | the split's validation peptides |
| `old_all` | same | all 12 |
| `new_val` | `_featurise_a101_k109` + row-normalised map | the same validation peptides |
| `new_all` | same | all 14 |

- New-feature predictions use the arm's own averaging mode: frame_uptake for the full-uptake arm, rate for the rate arm.
- ISO has two scopes: `val` and `all` (train + val residues).

**Definitions.**
- Residual r = pred − obs.
- Weights w = moprp.weights; σ_cell = 1/w.
- σ_pooled,p = RMS over timepoints of 1/w. **There is no floor**, as specified.
- ε = 0.01 in the log-uptake score.
- Work scale = RT·|Σ_f (w_f − 1/N)·mean_res lnPF_f| at BV 0.35/2.0. This is identical to `scoring.py:100`'s work_scale_kj for a per-residue weighted mean lnPF.
- ESS% = 100/(N·Σw²).
- **ISO has no experimental σ and no weights.** So for ISO, f1 = f2 = plain MSE, and σ = 0.0586 is a nominal constant. As a result f3 ≡ f2, f6 = max|r| and f7a = f7b.

**Selector names.**
- `f1_wmse` through `f10_offset_mse` are functions 1 to 10 of the brief:
  - f5a = Spearman, f5b = Pearson (both maximised)
  - f7a = Q_n with σ_pooled, f7b = Q_n with homoscedastic σ
- `f11_{wmse|mse}_ess_b{β}`: score + β(1 − ESS/N).
- `f12_ws_{wmse|mse}_{5|10}pct`: the minimum work scale among candidates within 5% or 10% of the minimum validation score.
- `iso_closed_sigma`: the existing closed-Σ selector, read from `val_closed_sigma_mse` in candidates.csv.
- Reference rows, which are **not** selection functions:
  - `ref_min_work_scale`: pure prior-proximity.
  - `ref_oracle_max_recovery`: the best candidate in the pool.

**Faithfulness checks.**
- `[s1]`: the native predictions replay candidates.csv `val_mse` and `val_weighted_mse` to a maximum relative error of 9.3e-6 across all 24 shards.
  - Recovery replays to < 1e-5.
  - Saved BV parameters equal the native ones in every state.
  - The full-dataset observed values equal `moprp.dexp` rows 0–11 to within 3e-8. The weight surface equals `moprp.weights` to within 1.2e-5.
- `[s3]`: f1/f2 on `old_val` reproduce `val_weighted_mse`/`val_mse` within rtol 2e-5 (asserted).
- `[s2]`: ISO `val_mse` replays to a maximum relative error of 4.7e-5 (JAX float32). Recovery replays to < 1e-4.

---

## Ranked verdict

### (a) ISO check

**Criterion.** On ISO_BI the selected recovery must be ≥ 95% for both split-type means, with no replicate below 90%. The uniform prior gives 62.6% and the pool best is 100% `[s5]`.

**Pass:** f1/f2, f3, f4, f5a, f5b, f6, f8, f9, f10, all four f12 variants, and closed-Σ. All of these select BI models at 98.2–99.8% `[s5]`.

**Fail:**
- **f7a/f7b (|Q_n − 1|)** select 62.4%, i.e. the prior. On noise-free synthetic ISO data, "χ² ≈ 1" with a nominal σ just picks a fit of size σ. ISO has no real noise model, so this check is partly unfair to f7. But it shows f7 has no fallback to fit quality.
- **All four f11 (ESS-penalised) variants** select 62.9–67.3%. ISO MSE is about 1e-3, so even β = 0.01 dominates and the selector returns near-prior states. The penalty is not scale-free.
- **`ref_min_work_scale`** gives 62.6%.

On ISO_TRI every passing selector lands at about 61–62% (MaxEnt) and 57–60% (OMC). The TRI pool best is 63.8% / 62.7% (MaxEnt). So TRI does not separate the selectors `[s5]`.

### (b) MoPrP

**Primary scope `old_val`, ISO-passers only.** Each figure is the mean recovery over 72 replicate selections (2 arms × 6 models × 2 split types × 3 replicates) `[s5]`.

| rank | selector | mean recovery (%) | Folded-majority reps | cells ≥ 80 |
|---|---|---|---|---|
| 1 | f12_ws_wmse_10pct | 49.4 | 54% | 1/24 |
| 2 | f10_offset_mse | 49.1 | 56% | 1/24 |
| 3 | f5b Pearson | 48.4 | 54% | 1/24 |
| 4 | f5a Spearman | 48.3 | 53% | 2/24 |
| 5 | f12_ws_mse_10pct | 47.1 | 53% | 1/24 |
| 6 | f9 mean-centred | 46.8 | 50% | 0/24 |
| 7 | f12_ws_wmse_5pct | 45.5 | — | — |
| 8 | f12_ws_mse_5pct | 44.1 | — | — |
| 9 | f1 (current) | 39.4 | 42% | 0/24 |
| 10 | f6 | 39.2 | — | — |
| 11 | f3 | 38.8 | — | — |
| 12 | f2 | 38.8 | — | — |
| 13 | f8 | 38.2 | — | — |
| 14 | f4 log-uptake | 34.5 | — | — |

- **Mean replicate SD** is 16.6–21.0 points for the top six, so no selector is consistent `[s5]`.
- **By arm, the rate arm is always higher:**
  - top-ranked selector: rate 64.7, full-uptake 34.1
  - current f1: rate 52.3, full-uptake 26.6 `[s5]`
- **On the `new_all` evaluation** (new features, row-normalised map, all 14 peptides) the order barely changes:
  - f12_ws_mse_10pct 52.9 (62% Folded-majority)
  - f5b 48.7
  - f12_ws_wmse_10pct 47.8
  - f1 39.8
  - Still no cell reaches ≥ 80 for any ISO-passer except f5a (2/24) and f5b (1/24) `[s5]`.

**Does any selector reach ≥ 80%? No.**
- No ISO-passing selector reaches ≥ 80% on MoPrP in any scope.
- The few cells ≥ 80 are artefacts. They are the full-uptake OMC dihedral k0 spatial cells, where f5a, f5b and f10 select a **single Folded frame**: ESS 0.2%, recovery 87.9 = the all-Folded cap, in all three replicates `[s4]`.
- **The only selectors that reach the prior range (75–77%) with Folded majorities** are `ref_min_work_scale` and f11_mse_ess_b0.1:
  - `ref_min_work_scale`: 77.2 / 75.3 on old_val / new_all, Folded-majority 99% / 96%
  - f11_mse_ess_b0.1: 68.3 / 74.1, Folded-majority 86% / 94% `[s5]`
  - Both get there by **not fitting**: ESS 67–82%, they return states next to the prior, and they fail the ISO check.

**Bottom line.** No selection function both passes ISO and beats the uniform prior (73.3 / 80.8) on MoPrP, let alone reaches 80%. Among the ISO-passers the shape-only and offset-insensitive scores, and the work-scale tie-break at 10%, do best: about 48–53% versus 39% for the current f1. The physically expected populations (Folded majority, PUF3/U ≈ 0) are not obtained; PUF3 + U averages 26–31% for the top ISO-passers `[s5]`.

### Why selection cannot fix this

**1. Every fit-quality score is anti-monotone with recovery on MoPrP but strongly monotone on ISO** `[s4]`. Values are mean within-replicate Spearman ρ(−score, recovery); positive means a better score goes with higher recovery.

| scores | ISO | MoPrP full-uptake old_val | MoPrP rate old_val |
|---|---|---|---|
| f1, f2, f3, f4, f6, f8 | +0.84 | −0.64 to −0.66 | −0.33 to −0.48 |

- The shape-only scores (f5a, f5b, f10) are about uncorrelated on MoPrP (−0.37 to +0.14). That is why they rank top: they are less wrong, not right.
- Only prior-proximity correlates positively with recovery: work scale gives +0.53 (full-uptake) and +0.39 (rate). On ISO it gives −0.03, and −0.87 for BI MaxEnt.

**2. The pools do contain good answers, and every score ranks them in the bottom half** `[s6]`.
- **MaxEnt pools never exceed the prior:** max 73.4 (MSAss full-uptake), 75.2 (MSAss rate), about 81.0 (Filtered). So MaxEnt is capped at the prior under any selector.
- **OMC pools hold candidates at 92–98%,** including ones with ESS ≥ 10%.
- In the rate arm's k0 pools these are not collapses: >80 candidates have median ESS 61–63%.
- Yet the median percentile rank of the >80 candidates is 0.59–0.90 under f1 and 0.54–0.80 under f5a/f10. In other words, the scores rank them below an average candidate. The exceptions are Filtered MaxEnt (f5a 0.36 full-uptake and 0.29 rate; f10 0.29 rate), where the >80 candidates are just the prior (ESS 99.9%).

**3. This matches the known BV under-prediction** (see memory `recovery-gulf-diagnosis`). Lower error is bought with decoy (PUF3 / unfolded / PUF2) mass. The new features and the 14-peptide map do not remove that bias: on `new_all`, f1's ρ is still −0.60 for full-uptake `[s4]`. The fix has to come from the forward model or the bias correction, not from the selector.

---

## Figures `[s5]`

**Primary (old_val, fit-faithful).** Recovery of the selected model by selection function. Bars are split types with replicate SD; dashed line = uniform prior, dotted = all-Folded cap 87.9, thin red = 80.
- `F_full_uptake_old_val.png`
- `F_rate_old_val.png`

**All-14-peptide evaluation (new features).**
- `F_full_uptake_new_all.png`
- `F_rate_new_all.png`

**Other scopes.**
- `F_full_uptake_old_all.png`
- `F_full_uptake_new_val.png`
- `F_rate_old_all.png`
- `F_rate_new_val.png`

**ISO check (validation scope).**
- `F_iso_val.png`

**Spearman heatmap.** Score versus recovery, per arm and scope.
- `F_spearman.png`

All paths are under `/home/alexi/Documents/JAX-ENT/artifacts/recovery_gulf_20261103/sel/`.

---

## Tables

### T1. ISO ground-truth check: validation-scope selection, recovery % `[s5]` (`sel_iso_check.csv`)

| selector | BI seq | BI spatial | BI worst rep | TRI MaxEnt seq/spa | TRI OMC (mean of 3 graphs × 2 splits) | ISO pass |
|---|---|---|---|---|---|---|
| f1_wmse | 99.5 ± 0.2 | 99.8 ± 0.2 | 99.4 | 61.5 / 61.9 | 59.1 | PASS |
| f2_mse | 99.5 ± 0.2 | 99.8 ± 0.2 | 99.4 | 61.5 / 61.9 | 59.1 | PASS |
| f3_chi2_pooled | 99.5 ± 0.2 | 99.8 ± 0.2 | 99.4 | 61.5 / 61.9 | 59.1 | PASS |
| f4_logerr | 99.4 ± 0.2 | 99.8 ± 0.2 | 99.1 | 61.5 / 62.1 | 59.0 | PASS |
| f5a_neg_spearman | 99.6 ± 0.3 | 99.2 ± 0.6 | 98.6 | 62.1 / 61.9 | 59.0 | PASS |
| f5b_neg_pearson | 99.5 ± 0.2 | 99.7 ± 0.1 | 99.4 | 61.5 / 61.9 | 59.1 | PASS |
| f6_maxabsz | 98.9 ± 0.9 | 98.2 ± 1.1 | 97.2 | 61.3 / 61.7 | 58.7 | PASS |
| f7a_Qn_pooled | 62.4 ± 0.1 | 62.5 ± 0.6 | 61.9 | 31.3 / 31.7 | 40.0 | fail |
| f7b_Qn_homo | 62.4 ± 0.1 | 62.5 ± 0.6 | 61.9 | 31.3 / 31.7 | 40.0 | fail |
| f8_pep_norm_mse | 99.5 ± 0.4 | 99.5 ± 0.6 | 98.8 | 62.0 / 61.7 | 57.7 | PASS |
| f9_meancentred_mse | 99.3 ± 0.4 | 99.6 ± 0.2 | 99.0 | 61.5 / 61.9 | 58.7 | PASS |
| f10_offset_mse | 99.5 ± 0.2 | 99.8 ± 0.2 | 99.4 | 61.5 / 61.9 | 59.1 | PASS |
| f11_wmse_ess_b0.01 | 65.7 ± 3.5 | 67.3 ± 3.1 | 63.7 | 31.7 / 31.8 | 30.7 | fail |
| f11_wmse_ess_b0.1 | 63.1 ± 0.3 | 62.9 ± 0.2 | 62.7 | 31.1 / 31.1 | 29.9 | fail |
| f11_mse_ess_b0.01 | 65.7 ± 3.5 | 67.3 ± 3.1 | 63.7 | 31.7 / 31.8 | 30.7 | fail |
| f11_mse_ess_b0.1 | 63.1 ± 0.3 | 62.9 ± 0.2 | 62.7 | 31.1 / 31.1 | 29.9 | fail |
| f12_ws_wmse_5pct | 98.7 ± 0.2 | 98.9 ± 0.3 | 98.5 | 61.6 / 62.1 | 58.3 | PASS |
| f12_ws_wmse_10pct | 98.8 ± 0.8 | 98.8 ± 0.6 | 97.8 | 62.0 / 62.0 | 57.9 | PASS |
| f12_ws_mse_5pct | 98.7 ± 0.2 | 98.9 ± 0.3 | 98.5 | 61.6 / 62.1 | 58.3 | PASS |
| f12_ws_mse_10pct | 98.8 ± 0.8 | 98.8 ± 0.6 | 97.8 | 62.0 / 62.0 | 57.9 | PASS |
| iso_closed_sigma | 99.2 ± 0.2 | 99.0 ± 0.4 | 98.6 | 61.5 / 62.1 | 58.5 | PASS |
| ref_min_work_scale | 62.6 ± 0.0 | 62.6 ± 0.0 | 62.6 | 31.1 / 31.1 | 45.3 | fail |
| ref_oracle_max_recovery | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 | 63.8 / 62.7 | 67.6 | PASS |

### T2. MoPrP verdict per selector `[s5]` (`sel_verdict.csv`)

**Verdict, scope old_val (24 MoPrP cells = 2 arms × 6 ensemble/model × 2 split types; 72 replicate selections) [s5]**

| selector | ISO | MoPrP mean rec | full-uptake | rate | MaxEnt | OMC | mean rep SD | cells ≥80 (/24) | reps ≥80 (/72) | Folded-majority reps | Folded % | PUF3+U % | ESS % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_oracle_max_recovery | PASS | 87.1 | 87.7 | 86.6 | 77.3 | 92.1 | 1.7 | 20 | 60 | 100% | 91.8 | 1.1 | 53.2 |
| f12_ws_wmse_10pct | PASS | 49.4 | 34.1 | 64.7 | 57.3 | 45.5 | 16.7 | 1 | 7 | 54% | 51.8 | 29.9 | 28.5 |
| f10_offset_mse | PASS | 49.1 | 36.2 | 62.0 | 56.4 | 45.4 | 16.6 | 1 | 11 | 56% | 51.2 | 26.3 | 30.3 |
| f5b_neg_pearson | PASS | 48.4 | 36.7 | 60.1 | 54.3 | 45.5 | 18.0 | 1 | 11 | 54% | 50.7 | 28.3 | 28.5 |
| f5a_neg_spearman | PASS | 48.3 | 37.1 | 59.4 | 57.2 | 43.8 | 21.0 | 2 | 14 | 53% | 50.6 | 30.2 | 29.1 |
| f12_ws_mse_10pct | PASS | 47.1 | 31.7 | 62.5 | 58.3 | 41.6 | 17.3 | 1 | 5 | 53% | 49.3 | 31.4 | 29.2 |
| f9_meancentred_mse | PASS | 46.8 | 33.7 | 59.9 | 46.2 | 47.1 | 18.2 | 0 | 6 | 50% | 48.4 | 27.6 | 26.3 |
| f12_ws_wmse_5pct | PASS | 45.5 | 31.7 | 59.4 | 54.1 | 41.2 | 17.6 | 1 | 5 | 50% | 47.2 | 31.1 | 23.1 |
| f12_ws_mse_5pct | PASS | 44.1 | 30.2 | 58.0 | 54.1 | 39.1 | 16.5 | 1 | 6 | 47% | 45.6 | 32.2 | 23.3 |
| f1_wmse | PASS | 39.4 | 26.6 | 52.3 | 42.8 | 37.8 | 17.4 | 0 | 2 | 42% | 40.4 | 33.6 | 17.0 |
| f6_maxabsz | PASS | 39.2 | 27.3 | 51.0 | 45.8 | 35.8 | 20.3 | 0 | 6 | 40% | 40.3 | 34.7 | 16.3 |
| f3_chi2_pooled | PASS | 38.8 | 27.6 | 50.1 | 42.0 | 37.3 | 17.5 | 0 | 2 | 40% | 39.7 | 33.6 | 18.8 |
| f2_mse | PASS | 38.8 | 26.1 | 51.5 | 42.0 | 37.2 | 18.2 | 0 | 2 | 40% | 39.6 | 34.6 | 17.0 |
| f8_pep_norm_mse | PASS | 38.2 | 27.0 | 49.4 | 40.2 | 37.1 | 17.3 | 0 | 1 | 40% | 39.2 | 38.5 | 15.6 |
| f4_logerr | PASS | 34.5 | 25.6 | 43.5 | 44.3 | 29.7 | 17.7 | 0 | 1 | 38% | 35.0 | 42.1 | 12.8 |
| ref_min_work_scale | fail | 77.2 | 76.6 | 77.8 | 77.1 | 77.3 | 4.3 | 9 | 28 | 99% | 82.9 | 3.5 | 68.6 |
| f11_mse_ess_b0.1 | fail | 68.3 | 59.4 | 77.2 | 74.3 | 65.3 | 10.2 | 5 | 19 | 86% | 73.0 | 14.0 | 71.9 |
| f11_mse_ess_b0.01 | fail | 44.1 | 27.0 | 61.1 | 49.4 | 41.4 | 19.8 | 0 | 2 | 47% | 45.6 | 33.4 | 32.2 |
| f11_wmse_ess_b0.1 | fail | 41.2 | 26.6 | 55.9 | 44.1 | 39.8 | 17.8 | 0 | 2 | 43% | 42.4 | 33.6 | 21.8 |
| f11_wmse_ess_b0.01 | fail | 39.4 | 26.6 | 52.3 | 42.8 | 37.8 | 17.4 | 0 | 2 | 42% | 40.4 | 33.6 | 17.0 |
| f7a_Qn_pooled | fail | 38.8 | 27.6 | 50.1 | 42.0 | 37.3 | 17.5 | 0 | 2 | 40% | 39.7 | 33.6 | 18.8 |
| f7b_Qn_homo | fail | 38.8 | 26.1 | 51.5 | 42.0 | 37.2 | 18.2 | 0 | 2 | 40% | 39.6 | 34.6 | 17.0 |

**Verdict, scope old_all (24 MoPrP cells = 2 arms × 6 ensemble/model × 2 split types; 72 replicate selections) [s5]**

| selector | ISO | MoPrP mean rec | full-uptake | rate | MaxEnt | OMC | mean rep SD | cells ≥80 (/24) | reps ≥80 (/72) | Folded-majority reps | Folded % | PUF3+U % | ESS % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_oracle_max_recovery | PASS | 87.1 | 87.7 | 86.6 | 77.3 | 92.1 | 1.7 | 20 | 60 | 100% | 91.8 | 1.1 | 53.2 |
| f10_offset_mse | PASS | 50.0 | 41.1 | 58.9 | 52.4 | 48.8 | 21.4 | 0 | 10 | 58% | 52.9 | 25.1 | 27.4 |
| f12_ws_mse_10pct | PASS | 49.8 | 36.2 | 63.5 | 49.3 | 50.1 | 15.2 | 0 | 4 | 56% | 52.4 | 24.7 | 24.2 |
| f12_ws_wmse_10pct | PASS | 48.9 | 34.7 | 63.1 | 48.6 | 49.1 | 16.7 | 0 | 6 | 54% | 51.6 | 24.6 | 23.7 |
| f5b_neg_pearson | PASS | 48.0 | 41.2 | 54.7 | 52.1 | 45.9 | 20.2 | 0 | 7 | 58% | 50.7 | 27.1 | 24.8 |
| f12_ws_mse_5pct | PASS | 46.4 | 34.6 | 58.1 | 44.1 | 47.5 | 14.8 | 0 | 1 | 50% | 48.3 | 26.4 | 18.7 |
| f12_ws_wmse_5pct | PASS | 46.2 | 34.3 | 58.0 | 44.2 | 47.2 | 15.2 | 0 | 2 | 50% | 48.1 | 26.8 | 17.2 |
| f5a_neg_spearman | PASS | 46.2 | 40.4 | 51.9 | 53.0 | 42.7 | 27.3 | 0 | 12 | 51% | 48.5 | 27.8 | 25.4 |
| f9_meancentred_mse | PASS | 41.9 | 32.9 | 51.0 | 37.7 | 44.0 | 16.7 | 0 | 2 | 42% | 42.7 | 32.0 | 20.0 |
| f8_pep_norm_mse | PASS | 40.6 | 31.3 | 49.9 | 35.9 | 42.9 | 14.4 | 0 | 1 | 42% | 41.7 | 32.0 | 13.4 |
| f3_chi2_pooled | PASS | 40.0 | 30.6 | 49.3 | 35.6 | 42.2 | 13.3 | 0 | 1 | 40% | 40.8 | 31.4 | 13.8 |
| f2_mse | PASS | 40.0 | 30.4 | 49.6 | 35.6 | 42.2 | 13.5 | 0 | 1 | 40% | 40.7 | 30.6 | 13.6 |
| f1_wmse | PASS | 39.9 | 30.7 | 49.2 | 35.8 | 42.0 | 13.5 | 0 | 1 | 40% | 40.6 | 30.5 | 13.4 |
| f4_logerr | PASS | 32.9 | 25.8 | 39.9 | 36.5 | 31.1 | 17.4 | 0 | 1 | 32% | 32.8 | 41.8 | 12.4 |
| f6_maxabsz | PASS | 32.0 | 24.1 | 39.9 | 36.7 | 29.7 | 17.8 | 0 | 2 | 31% | 31.8 | 38.7 | 9.7 |
| ref_min_work_scale | fail | 77.2 | 76.6 | 77.8 | 77.1 | 77.3 | 4.3 | 9 | 28 | 99% | 82.9 | 3.5 | 68.6 |
| f11_mse_ess_b0.1 | fail | 69.7 | 63.1 | 76.4 | 75.6 | 66.8 | 5.4 | 3 | 13 | 88% | 74.3 | 11.5 | 76.1 |
| f11_mse_ess_b0.01 | fail | 45.3 | 31.3 | 59.4 | 40.1 | 47.9 | 14.8 | 0 | 1 | 49% | 46.8 | 28.7 | 29.9 |
| f11_wmse_ess_b0.1 | fail | 40.1 | 30.7 | 49.5 | 36.2 | 42.0 | 13.6 | 0 | 1 | 40% | 40.8 | 30.4 | 14.6 |
| f7a_Qn_pooled | fail | 40.0 | 30.6 | 49.3 | 35.6 | 42.2 | 13.3 | 0 | 1 | 40% | 40.8 | 31.4 | 13.8 |
| f7b_Qn_homo | fail | 40.0 | 30.4 | 49.6 | 35.6 | 42.2 | 13.5 | 0 | 1 | 40% | 40.7 | 30.6 | 13.6 |
| f11_wmse_ess_b0.01 | fail | 39.9 | 30.7 | 49.2 | 35.8 | 42.0 | 13.5 | 0 | 1 | 40% | 40.6 | 30.5 | 13.4 |

**Verdict, scope new_val (24 MoPrP cells = 2 arms × 6 ensemble/model × 2 split types; 72 replicate selections) [s5]**

| selector | ISO | MoPrP mean rec | full-uptake | rate | MaxEnt | OMC | mean rep SD | cells ≥80 (/24) | reps ≥80 (/72) | Folded-majority reps | Folded % | PUF3+U % | ESS % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_oracle_max_recovery | PASS | 87.1 | 87.7 | 86.6 | 77.3 | 92.1 | 1.7 | 20 | 60 | 100% | 91.8 | 1.1 | 53.2 |
| f10_offset_mse | PASS | 49.8 | 38.9 | 60.8 | 54.8 | 47.3 | 19.4 | 2 | 11 | 56% | 51.9 | 25.8 | 30.6 |
| f12_ws_wmse_10pct | PASS | 49.7 | 34.2 | 65.3 | 60.0 | 44.6 | 16.5 | 1 | 8 | 56% | 52.3 | 29.2 | 32.0 |
| f12_ws_mse_10pct | PASS | 48.7 | 36.0 | 61.3 | 60.6 | 42.7 | 15.7 | 1 | 8 | 54% | 51.0 | 28.8 | 32.5 |
| f5b_neg_pearson | PASS | 47.0 | 35.8 | 58.2 | 53.9 | 43.6 | 19.8 | 0 | 10 | 53% | 49.0 | 28.4 | 28.7 |
| f12_ws_wmse_5pct | PASS | 46.1 | 33.9 | 58.2 | 55.0 | 41.6 | 16.2 | 1 | 6 | 50% | 48.1 | 32.1 | 25.5 |
| f9_meancentred_mse | PASS | 45.8 | 35.0 | 56.7 | 45.1 | 46.2 | 16.6 | 0 | 5 | 46% | 47.2 | 26.2 | 24.9 |
| f12_ws_mse_5pct | PASS | 45.4 | 31.9 | 58.9 | 55.3 | 40.4 | 16.8 | 1 | 5 | 49% | 47.4 | 31.9 | 25.0 |
| f5a_neg_spearman | PASS | 44.5 | 32.3 | 56.8 | 54.3 | 39.6 | 21.4 | 1 | 12 | 49% | 46.4 | 31.7 | 27.7 |
| f8_pep_norm_mse | PASS | 41.4 | 30.3 | 52.5 | 45.2 | 39.5 | 17.2 | 0 | 3 | 43% | 42.6 | 34.1 | 19.7 |
| f6_maxabsz | PASS | 41.3 | 29.3 | 53.3 | 47.1 | 38.5 | 20.9 | 0 | 6 | 44% | 42.7 | 32.2 | 18.2 |
| f3_chi2_pooled | PASS | 40.2 | 27.7 | 52.6 | 42.9 | 38.8 | 17.7 | 0 | 3 | 40% | 41.4 | 33.9 | 19.2 |
| f2_mse | PASS | 40.2 | 29.1 | 51.2 | 40.7 | 39.9 | 17.3 | 0 | 3 | 39% | 41.2 | 33.7 | 18.9 |
| f1_wmse | PASS | 39.5 | 28.5 | 50.4 | 39.6 | 39.4 | 18.5 | 1 | 4 | 42% | 40.5 | 34.9 | 18.1 |
| f4_logerr | PASS | 35.2 | 27.2 | 43.2 | 44.1 | 30.8 | 18.9 | 0 | 1 | 38% | 35.7 | 41.1 | 12.4 |
| ref_min_work_scale | fail | 75.3 | 76.2 | 74.3 | 77.1 | 74.4 | 6.6 | 7 | 27 | 96% | 80.5 | 3.5 | 66.8 |
| f11_mse_ess_b0.1 | fail | 69.4 | 61.1 | 77.7 | 74.4 | 66.9 | 9.8 | 6 | 22 | 86% | 74.2 | 13.4 | 73.5 |
| f11_mse_ess_b0.01 | fail | 46.1 | 31.5 | 60.7 | 53.5 | 42.4 | 18.8 | 0 | 5 | 49% | 47.9 | 31.0 | 34.2 |
| f11_wmse_ess_b0.1 | fail | 42.1 | 28.5 | 55.6 | 44.2 | 41.0 | 18.3 | 0 | 4 | 46% | 43.4 | 32.6 | 23.2 |
| f11_wmse_ess_b0.01 | fail | 40.2 | 28.5 | 51.8 | 41.7 | 39.4 | 17.6 | 1 | 4 | 43% | 41.3 | 33.9 | 19.5 |
| f7a_Qn_pooled | fail | 40.2 | 27.7 | 52.6 | 42.9 | 38.8 | 17.7 | 0 | 3 | 40% | 41.4 | 33.9 | 19.2 |
| f7b_Qn_homo | fail | 40.2 | 29.1 | 51.2 | 40.7 | 39.9 | 17.3 | 0 | 3 | 39% | 41.2 | 33.7 | 18.9 |

**Verdict, scope new_all (24 MoPrP cells = 2 arms × 6 ensemble/model × 2 split types; 72 replicate selections) [s5]**

| selector | ISO | MoPrP mean rec | full-uptake | rate | MaxEnt | OMC | mean rep SD | cells ≥80 (/24) | reps ≥80 (/72) | Folded-majority reps | Folded % | PUF3+U % | ESS % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_oracle_max_recovery | PASS | 87.1 | 87.7 | 86.6 | 77.3 | 92.1 | 1.7 | 20 | 60 | 100% | 91.8 | 1.1 | 53.2 |
| f12_ws_mse_10pct | PASS | 52.9 | 40.3 | 65.6 | 56.1 | 51.3 | 13.6 | 0 | 6 | 62% | 56.2 | 21.9 | 24.6 |
| f5b_neg_pearson | PASS | 48.7 | 39.8 | 57.6 | 53.3 | 46.4 | 21.7 | 1 | 12 | 57% | 51.3 | 25.7 | 29.3 |
| f12_ws_wmse_10pct | PASS | 47.8 | 33.9 | 61.7 | 55.8 | 43.8 | 17.6 | 0 | 8 | 56% | 49.7 | 22.6 | 26.0 |
| f12_ws_mse_5pct | PASS | 47.1 | 32.8 | 61.3 | 46.8 | 47.2 | 14.3 | 0 | 4 | 50% | 49.4 | 27.2 | 20.8 |
| f10_offset_mse | PASS | 46.6 | 38.9 | 54.3 | 52.7 | 43.5 | 23.8 | 0 | 12 | 54% | 48.5 | 24.7 | 28.8 |
| f5a_neg_spearman | PASS | 45.2 | 36.2 | 54.1 | 53.4 | 41.0 | 22.7 | 2 | 21 | 50% | 47.1 | 20.3 | 27.7 |
| f12_ws_wmse_5pct | PASS | 44.6 | 31.9 | 57.3 | 48.5 | 42.7 | 14.9 | 0 | 4 | 46% | 45.3 | 26.5 | 22.0 |
| f9_meancentred_mse | PASS | 42.0 | 33.7 | 50.3 | 44.2 | 41.0 | 13.3 | 0 | 3 | 46% | 42.1 | 27.2 | 25.5 |
| f2_mse | PASS | 41.7 | 30.8 | 52.6 | 36.5 | 44.3 | 15.1 | 0 | 3 | 42% | 42.9 | 31.5 | 15.3 |
| f3_chi2_pooled | PASS | 40.2 | 29.0 | 51.5 | 36.7 | 42.0 | 14.8 | 0 | 4 | 40% | 41.0 | 31.3 | 17.8 |
| f1_wmse | PASS | 39.8 | 26.7 | 53.0 | 39.3 | 40.1 | 14.9 | 0 | 4 | 38% | 39.8 | 29.2 | 18.5 |
| f8_pep_norm_mse | PASS | 39.1 | 29.7 | 48.5 | 47.5 | 34.9 | 11.1 | 0 | 2 | 38% | 38.4 | 27.3 | 28.3 |
| f6_maxabsz | PASS | 36.3 | 28.0 | 44.6 | 41.6 | 33.7 | 19.2 | 0 | 2 | 36% | 37.1 | 31.7 | 11.3 |
| f4_logerr | PASS | 34.5 | 26.8 | 42.2 | 38.5 | 32.5 | 18.8 | 0 | 1 | 33% | 34.2 | 32.8 | 11.9 |
| ref_min_work_scale | fail | 75.3 | 76.2 | 74.3 | 77.1 | 74.4 | 6.6 | 7 | 27 | 96% | 80.5 | 3.5 | 66.8 |
| f11_mse_ess_b0.1 | fail | 74.1 | 70.9 | 77.3 | 75.9 | 73.2 | 8.5 | 6 | 25 | 94% | 79.0 | 7.4 | 81.8 |
| f11_mse_ess_b0.01 | fail | 46.9 | 30.9 | 62.8 | 45.1 | 47.8 | 17.1 | 0 | 3 | 51% | 48.9 | 29.6 | 32.9 |
| f7b_Qn_homo | fail | 41.7 | 30.8 | 52.6 | 36.5 | 44.3 | 15.1 | 0 | 3 | 42% | 42.9 | 31.5 | 15.3 |
| f11_wmse_ess_b0.1 | fail | 40.8 | 26.9 | 54.8 | 39.9 | 41.3 | 15.7 | 0 | 4 | 40% | 41.1 | 28.9 | 20.5 |
| f7a_Qn_pooled | fail | 40.2 | 29.0 | 51.5 | 36.7 | 42.0 | 14.8 | 0 | 4 | 40% | 41.0 | 31.3 | 17.8 |
| f11_wmse_ess_b0.01 | fail | 39.8 | 26.7 | 53.0 | 39.3 | 40.1 | 14.9 | 0 | 4 | 38% | 39.8 | 29.2 | 18.6 |


### T3. MoPrP recovery per arm × ensemble/model × split type, scope old_val (primary) `[s4/s5]`

Full columns are in `sel_summary.csv`: ESS, all populations, the selected S per replicate, h, and the median step.

**full_uptake arm, scope old_val: recovery % (mean ± SD over 3 replicates) [s4/s5]**

| selector | MSAss maxent seq | MSAss maxent spa | MSAss work_k0 seq | MSAss work_k0 spa | MSAss dihed_k0 seq | MSAss dihed_k0 spa | MSAss work_k20 seq | MSAss work_k20 spa | MSAss dihed_k20 seq | MSAss dihed_k20 spa | filtered maxent seq | filtered maxent spa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1_wmse | 40.9 ± 10.1 | 29.2 ± 5.4 | 14.5 ± 25.0 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.9 ± 21.6 | 38.6 ± 7.2 | 9.0 ± 12.6 | 23.0 ± 13.5 | 19.1 ± 8.2 |
| f2_mse | 30.4 ± 15.7 | 29.2 ± 5.3 | 14.5 ± 25.0 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.3 ± 21.1 | 29.4 ± 22.4 | 8.3 ± 11.4 | 37.4 ± 12.0 | 20.1 ± 8.3 |
| f3_chi2_pooled | 40.9 ± 10.1 | 29.2 ± 5.4 | 34.8 ± 31.4 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.9 ± 21.6 | 29.4 ± 22.4 | 9.0 ± 12.6 | 23.0 ± 13.5 | 20.1 ± 8.3 |
| f4_logerr | 32.0 ± 18.7 | 28.0 ± 5.8 | 0.0 ± 0.0 | 30.4 ± 26.3 | 25.5 ± 5.4 | 28.3 ± 34.5 | 35.1 ± 28.4 | 23.8 ± 20.7 | 28.2 ± 21.7 | 8.2 ± 11.4 | 37.4 ± 12.0 | 30.3 ± 13.4 |
| f5a_neg_spearman | 34.2 ± 6.4 | 30.0 ± 7.4 | 44.6 ± 44.0 | 31.0 ± 26.9 | 46.8 ± 36.1 | 87.9 ± 0.0 | 21.7 ± 17.2 | 14.2 ± 21.2 | 14.2 ± 21.3 | 10.3 ± 11.4 | 44.3 ± 30.3 | 65.5 ± 25.7 |
| f5b_neg_pearson | 41.6 ± 10.1 | 31.5 ± 10.1 | 30.7 ± 30.5 | 31.8 ± 27.5 | 25.5 ± 5.4 | 87.9 ± 0.0 | 41.9 ± 18.3 | 46.9 ± 9.8 | 14.1 ± 17.8 | 10.3 ± 11.4 | 20.3 ± 11.5 | 58.3 ± 27.5 |
| f6_maxabsz | 34.5 ± 6.8 | 25.6 ± 3.1 | 35.7 ± 31.8 | 14.9 ± 25.8 | 25.5 ± 5.4 | 36.7 ± 44.9 | 40.9 ± 19.7 | 27.1 ± 14.0 | 27.3 ± 21.4 | 8.2 ± 11.4 | 20.2 ± 10.9 | 31.3 ± 15.1 |
| f7a_Qn_pooled | 40.9 ± 10.1 | 29.2 ± 5.4 | 34.8 ± 31.4 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.9 ± 21.6 | 29.4 ± 22.4 | 9.0 ± 12.6 | 23.0 ± 13.5 | 20.1 ± 8.3 |
| f7b_Qn_homo | 30.4 ± 15.7 | 29.2 ± 5.3 | 14.5 ± 25.0 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.3 ± 21.1 | 29.4 ± 22.4 | 8.3 ± 11.4 | 37.4 ± 12.0 | 20.1 ± 8.3 |
| f8_pep_norm_mse | 32.2 ± 18.5 | 26.7 ± 9.7 | 12.7 ± 21.9 | 30.4 ± 26.3 | 25.5 ± 5.4 | 28.4 ± 34.4 | 35.0 ± 28.4 | 28.6 ± 24.8 | 38.5 ± 7.2 | 9.0 ± 12.6 | 37.4 ± 12.0 | 19.3 ± 8.2 |
| f9_meancentred_mse | 39.7 ± 7.7 | 30.0 ± 13.7 | 49.2 ± 16.0 | 31.8 ± 27.5 | 24.2 ± 3.4 | 28.3 ± 34.5 | 39.0 ± 27.2 | 46.8 ± 10.0 | 40.2 ± 6.5 | 15.3 ± 19.9 | 35.5 ± 13.4 | 24.0 ± 11.7 |
| f10_offset_mse | 41.6 ± 10.1 | 31.5 ± 10.1 | 30.7 ± 30.5 | 31.8 ± 27.5 | 25.5 ± 5.4 | 87.9 ± 0.0 | 41.9 ± 18.3 | 46.9 ± 9.8 | 4.9 ± 1.7 | 10.3 ± 11.4 | 20.3 ± 11.5 | 60.5 ± 23.3 |
| f11_wmse_ess_b0.01 | 40.9 ± 10.1 | 29.2 ± 5.4 | 14.5 ± 25.0 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.9 ± 21.6 | 38.6 ± 7.2 | 9.0 ± 12.6 | 23.0 ± 13.5 | 19.1 ± 8.2 |
| f11_wmse_ess_b0.1 | 40.9 ± 10.1 | 29.2 ± 5.4 | 14.5 ± 25.0 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.9 ± 21.6 | 38.6 ± 7.2 | 9.0 ± 12.6 | 23.0 ± 13.5 | 19.1 ± 8.2 |
| f11_mse_ess_b0.01 | 32.2 ± 18.5 | 29.2 ± 5.3 | 14.5 ± 25.0 | 30.4 ± 26.3 | 25.5 ± 5.4 | 24.7 ± 28.4 | 35.1 ± 28.4 | 28.3 ± 21.1 | 29.4 ± 22.4 | 8.3 ± 11.4 | 46.5 ± 27.7 | 20.1 ± 8.3 |
| f11_mse_ess_b0.1 | 70.0 ± 0.8 | 58.9 ± 23.3 | 50.9 ± 44.2 | 39.4 ± 36.4 | 76.6 ± 3.0 | 73.0 ± 1.2 | 35.1 ± 28.4 | 55.9 ± 27.1 | 29.4 ± 22.4 | 62.4 ± 35.7 | 80.5 ± 0.3 | 80.4 ± 0.4 |
| f12_ws_wmse_5pct | 43.1 ± 11.4 | 28.5 ± 5.3 | 35.0 ± 31.7 | 31.7 ± 27.5 | 25.3 ± 3.3 | 28.8 ± 34.5 | 40.9 ± 19.7 | 27.2 ± 20.1 | 19.6 ± 23.4 | 9.1 ± 12.8 | 51.0 ± 11.0 | 39.7 ± 7.7 |
| f12_ws_wmse_10pct | 43.3 ± 11.0 | 30.4 ± 7.6 | 35.6 ± 32.4 | 32.4 ± 28.0 | 25.5 ± 7.2 | 28.8 ± 34.5 | 40.9 ± 19.7 | 38.8 ± 3.0 | 27.3 ± 19.3 | 9.1 ± 12.8 | 54.2 ± 7.0 | 43.4 ± 9.6 |
| f12_ws_mse_5pct | 42.7 ± 11.4 | 28.5 ± 5.3 | 14.5 ± 25.0 | 31.7 ± 27.4 | 25.3 ± 3.2 | 28.8 ± 34.5 | 35.1 ± 28.4 | 38.2 ± 2.3 | 19.6 ± 23.4 | 8.3 ± 11.4 | 50.5 ± 10.5 | 39.7 ± 7.7 |
| f12_ws_mse_10pct | 43.3 ± 11.0 | 30.7 ± 8.1 | 15.3 ± 26.5 | 32.5 ± 28.1 | 25.5 ± 7.2 | 28.8 ± 34.5 | 35.1 ± 28.4 | 39.2 ± 3.3 | 27.3 ± 19.3 | 8.6 ± 11.9 | 54.9 ± 6.0 | 39.7 ± 7.7 |
| ref_min_work_scale | 73.1 ± 0.1 | 73.2 ± 0.2 | 73.2 ± 15.1 | 77.7 ± 9.1 | 79.1 ± 7.8 | 72.6 ± 1.6 | 68.6 ± 20.1 | 81.3 ± 4.2 | 81.0 ± 5.9 | 77.9 ± 12.8 | 80.7 ± 0.0 | 80.9 ± 0.0 |
| ref_oracle_max_recovery | 73.2 ± 0.1 | 73.2 ± 0.1 | 93.3 ± 1.3 | 90.4 ± 3.0 | 90.9 ± 1.9 | 89.0 ± 1.8 | 96.0 ± 1.4 | 94.9 ± 3.4 | 94.3 ± 3.2 | 96.1 ± 1.0 | 80.8 ± 0.0 | 80.9 ± 0.1 |

**rate arm, scope old_val: recovery % (mean ± SD over 3 replicates) [s4/s5]**

| selector | MSAss maxent seq | MSAss maxent spa | MSAss work_k0 seq | MSAss work_k0 spa | MSAss dihed_k0 seq | MSAss dihed_k0 spa | MSAss work_k20 seq | MSAss work_k20 spa | MSAss dihed_k20 seq | MSAss dihed_k20 spa | filtered maxent seq | filtered maxent spa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1_wmse | 71.3 ± 1.4 | 67.7 ± 4.4 | 74.9 ± 5.1 | 44.5 ± 38.7 | 77.8 ± 3.9 | 73.7 ± 5.8 | 22.3 ± 26.9 | 28.6 ± 41.9 | 51.0 ± 38.2 | 24.6 ± 39.1 | 63.6 ± 9.6 | 27.6 ± 10.6 |
| f2_mse | 71.3 ± 1.4 | 67.7 ± 4.4 | 74.9 ± 5.1 | 44.5 ± 38.7 | 77.8 ± 3.9 | 76.0 ± 2.4 | 30.4 ± 38.8 | 28.6 ± 41.9 | 51.0 ± 38.2 | 16.0 ± 24.3 | 52.0 ± 16.3 | 27.6 ± 10.6 |
| f3_chi2_pooled | 71.7 ± 1.3 | 68.7 ± 5.2 | 75.7 ± 3.9 | 45.9 ± 40.1 | 78.3 ± 3.4 | 73.7 ± 5.8 | 26.4 ± 42.2 | 3.0 ± 2.9 | 51.0 ± 38.2 | 24.6 ± 39.1 | 52.0 ± 16.3 | 30.4 ± 7.5 |
| f4_logerr | 67.6 ± 7.0 | 64.7 ± 6.9 | 51.6 ± 44.8 | 0.0 ± 0.0 | 71.6 ± 13.7 | 70.3 ± 7.3 | 30.6 ± 21.8 | 24.9 ± 36.4 | 23.6 ± 33.9 | 22.8 ± 36.0 | 66.1 ± 7.4 | 28.0 ± 10.5 |
| f5a_neg_spearman | 72.2 ± 1.0 | 57.0 ± 20.0 | 75.9 ± 4.3 | 25.4 ± 43.8 | 80.6 ± 6.7 | 77.1 ± 1.7 | 30.8 ± 39.2 | 32.0 ± 48.7 | 58.1 ± 39.2 | 49.5 ± 44.0 | 76.2 ± 5.3 | 78.6 ± 3.5 |
| f5b_neg_pearson | 71.7 ± 1.3 | 56.0 ± 19.2 | 76.1 ± 4.6 | 46.8 ± 41.1 | 78.1 ± 3.7 | 76.9 ± 1.7 | 29.6 ± 39.5 | 32.4 ± 48.4 | 56.6 ± 41.8 | 42.7 ± 42.1 | 76.3 ± 5.4 | 78.4 ± 3.6 |
| f6_maxabsz | 71.7 ± 1.3 | 66.4 ± 7.9 | 53.0 ± 46.7 | 24.4 ± 42.3 | 79.2 ± 7.5 | 70.3 ± 7.3 | 1.7 ± 2.1 | 29.7 ± 44.7 | 50.3 ± 37.5 | 48.1 ± 42.4 | 65.0 ± 10.8 | 51.9 ± 27.1 |
| f7a_Qn_pooled | 71.7 ± 1.3 | 68.7 ± 5.2 | 75.7 ± 3.9 | 45.9 ± 40.1 | 78.3 ± 3.4 | 73.7 ± 5.8 | 26.4 ± 42.2 | 3.0 ± 2.9 | 51.0 ± 38.2 | 24.6 ± 39.1 | 52.0 ± 16.3 | 30.4 ± 7.5 |
| f7b_Qn_homo | 71.3 ± 1.4 | 67.7 ± 4.4 | 74.9 ± 5.1 | 44.5 ± 38.7 | 77.8 ± 3.9 | 76.0 ± 2.4 | 30.4 ± 38.8 | 28.6 ± 41.9 | 51.0 ± 38.2 | 16.0 ± 24.3 | 52.0 ± 16.3 | 27.6 ± 10.6 |
| f8_pep_norm_mse | 70.8 ± 1.9 | 56.0 ± 19.2 | 74.9 ± 5.1 | 45.6 ± 39.7 | 73.4 ± 10.6 | 75.3 ± 3.0 | 50.0 ± 35.3 | 3.0 ± 2.8 | 49.2 ± 36.2 | 14.6 ± 25.4 | 52.0 ± 16.3 | 27.6 ± 10.6 |
| f9_meancentred_mse | 73.0 ± 0.2 | 73.0 ± 0.1 | 79.1 ± 2.5 | 51.7 ± 44.8 | 79.0 ± 3.1 | 78.5 ± 2.0 | 30.1 ± 39.1 | 71.0 ± 24.0 | 55.9 ± 41.2 | 33.4 ± 44.6 | 51.5 ± 15.3 | 42.7 ± 28.0 |
| f10_offset_mse | 72.9 ± 0.2 | 69.7 ± 4.0 | 75.5 ± 4.0 | 51.5 ± 44.6 | 78.8 ± 3.5 | 77.1 ± 1.4 | 29.6 ± 39.5 | 32.4 ± 48.4 | 55.9 ± 41.2 | 45.4 ± 42.5 | 76.3 ± 5.4 | 78.6 ± 3.7 |
| f11_wmse_ess_b0.01 | 71.3 ± 1.4 | 67.7 ± 4.4 | 74.9 ± 5.1 | 44.5 ± 38.7 | 77.8 ± 3.9 | 73.7 ± 5.8 | 22.3 ± 26.9 | 28.6 ± 41.9 | 51.0 ± 38.2 | 24.6 ± 39.1 | 63.6 ± 9.6 | 27.6 ± 10.6 |
| f11_wmse_ess_b0.1 | 71.6 ± 1.3 | 68.3 ± 4.8 | 74.9 ± 5.1 | 45.9 ± 40.1 | 77.6 ± 4.1 | 73.7 ± 5.8 | 29.8 ± 39.7 | 27.5 ± 39.9 | 51.0 ± 38.2 | 49.8 ± 43.4 | 73.0 ± 3.2 | 27.6 ± 10.6 |
| f11_mse_ess_b0.01 | 71.6 ± 1.3 | 70.8 ± 1.0 | 76.7 ± 2.7 | 49.6 ± 43.0 | 77.4 ± 4.2 | 76.0 ± 2.4 | 51.4 ± 36.8 | 29.6 ± 43.5 | 55.6 ± 41.0 | 49.8 ± 43.4 | 77.1 ± 0.6 | 47.5 ± 27.0 |
| f11_mse_ess_b0.1 | 72.4 ± 0.5 | 72.0 ± 0.8 | 77.2 ± 2.7 | 74.6 ± 0.9 | 78.1 ± 3.7 | 74.1 ± 1.8 | 78.4 ± 4.4 | 79.4 ± 2.2 | 81.2 ± 1.4 | 78.1 ± 2.5 | 80.4 ± 0.3 | 80.2 ± 0.3 |
| f12_ws_wmse_5pct | 72.5 ± 0.7 | 69.6 ± 3.7 | 75.9 ± 3.9 | 44.3 ± 38.5 | 78.0 ± 3.9 | 74.9 ± 4.2 | 36.7 ± 41.5 | 28.6 ± 41.9 | 56.0 ± 30.3 | 47.6 ± 37.2 | 80.0 ± 1.7 | 48.0 ± 7.9 |
| f12_ws_wmse_10pct | 73.0 ± 0.9 | 71.0 ± 3.6 | 75.6 ± 3.9 | 48.8 ± 42.3 | 78.7 ± 3.1 | 76.2 ± 2.2 | 58.9 ± 23.1 | 42.0 ± 35.7 | 61.3 ± 34.6 | 47.6 ± 37.2 | 80.9 ± 0.1 | 61.9 ± 21.3 |
| f12_ws_mse_5pct | 72.6 ± 0.8 | 69.1 ± 3.6 | 76.1 ± 3.9 | 47.0 ± 40.7 | 78.1 ± 3.8 | 76.4 ± 2.0 | 30.8 ± 29.1 | 28.1 ± 42.4 | 62.1 ± 33.1 | 26.1 ± 37.7 | 80.9 ± 0.1 | 48.4 ± 8.2 |
| f12_ws_mse_10pct | 73.3 ± 0.8 | 71.0 ± 3.3 | 75.6 ± 3.9 | 50.5 ± 43.8 | 78.7 ± 3.1 | 76.4 ± 2.0 | 39.7 ± 37.7 | 28.1 ± 42.4 | 56.3 ± 42.5 | 47.6 ± 37.2 | 80.9 ± 0.1 | 72.2 ± 6.2 |
| ref_min_work_scale | 73.7 ± 1.0 | 73.1 ± 0.2 | 76.6 ± 0.4 | 75.3 ± 1.1 | 76.3 ± 1.1 | 75.0 ± 0.5 | 80.8 ± 6.9 | 79.5 ± 7.6 | 80.0 ± 1.9 | 81.3 ± 6.4 | 80.9 ± 0.1 | 80.8 ± 0.1 |
| ref_oracle_max_recovery | 74.5 ± 0.7 | 73.4 ± 0.8 | 89.2 ± 1.5 | 90.7 ± 3.6 | 92.4 ± 1.9 | 89.0 ± 1.8 | 92.8 ± 1.8 | 92.0 ± 3.6 | 91.7 ± 4.0 | 90.6 ± 3.0 | 81.0 ± 0.0 | 81.0 ± 0.1 |


### T4. Same, scope new_all (new features, row-normalised map, all 14 peptides) `[s4/s5]`

**full_uptake arm, scope new_all: recovery % (mean ± SD over 3 replicates) [s4/s5]**

| selector | MSAss maxent seq | MSAss maxent spa | MSAss work_k0 seq | MSAss work_k0 spa | MSAss dihed_k0 seq | MSAss dihed_k0 spa | MSAss work_k20 seq | MSAss work_k20 spa | MSAss dihed_k20 seq | MSAss dihed_k20 spa | filtered maxent seq | filtered maxent spa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1_wmse | 40.6 ± 6.2 | 29.3 ± 13.1 | 34.4 ± 25.1 | 44.3 ± 17.7 | 24.2 ± 3.4 | 38.5 ± 25.3 | 24.3 ± 14.8 | 24.0 ± 20.8 | 12.0 ± 14.4 | 8.8 ± 12.4 | 20.5 ± 11.3 | 19.6 ± 8.5 |
| f2_mse | 30.4 ± 15.7 | 28.0 ± 11.6 | 53.4 ± 9.1 | 48.1 ± 2.6 | 24.2 ± 3.4 | 38.5 ± 25.3 | 41.6 ± 18.7 | 24.0 ± 20.8 | 30.3 ± 23.5 | 8.5 ± 11.8 | 20.5 ± 11.3 | 21.8 ± 9.6 |
| f3_chi2_pooled | 30.5 ± 15.7 | 28.4 ± 11.9 | 53.4 ± 9.1 | 49.3 ± 1.6 | 24.2 ± 3.4 | 34.8 ± 19.1 | 40.8 ± 18.7 | 24.0 ± 20.8 | 11.3 ± 13.2 | 8.7 ± 12.2 | 20.5 ± 11.3 | 21.8 ± 9.6 |
| f4_logerr | 41.9 ± 9.5 | 29.2 ± 5.4 | 34.8 ± 31.4 | 42.0 ± 15.4 | 25.5 ± 5.4 | 28.3 ± 34.5 | 34.3 ± 7.2 | 24.0 ± 20.8 | 13.6 ± 12.9 | 8.7 ± 12.3 | 20.5 ± 11.3 | 19.1 ± 8.2 |
| f5a_neg_spearman | 39.8 ± 30.0 | 26.3 ± 9.4 | 87.9 ± 0.0 | 20.0 ± 27.5 | 60.0 ± 48.4 | 60.0 ± 48.4 | 11.4 ± 12.6 | 2.6 ± 2.2 | 2.8 ± 2.4 | 10.2 ± 11.1 | 56.2 ± 42.3 | 57.5 ± 40.3 |
| f5b_neg_pearson | 38.6 ± 27.0 | 29.4 ± 13.3 | 60.4 ± 0.9 | 47.2 ± 6.2 | 59.9 ± 30.2 | 60.0 ± 48.4 | 33.7 ± 34.2 | 15.5 ± 19.6 | 10.7 ± 15.3 | 10.2 ± 11.1 | 54.8 ± 41.1 | 57.1 ± 39.9 |
| f6_maxabsz | 34.5 ± 6.8 | 37.2 ± 6.5 | 49.2 ± 16.0 | 49.5 ± 1.5 | 25.5 ± 5.4 | 7.4 ± 9.5 | 40.8 ± 18.7 | 26.6 ± 23.2 | 17.9 ± 24.7 | 8.2 ± 11.4 | 20.2 ± 10.9 | 19.1 ± 8.2 |
| f7a_Qn_pooled | 30.5 ± 15.7 | 28.4 ± 11.9 | 53.4 ± 9.1 | 49.3 ± 1.6 | 24.2 ± 3.4 | 34.8 ± 19.1 | 40.8 ± 18.7 | 24.0 ± 20.8 | 11.3 ± 13.2 | 8.7 ± 12.2 | 20.5 ± 11.3 | 21.8 ± 9.6 |
| f7b_Qn_homo | 30.4 ± 15.7 | 28.0 ± 11.6 | 53.4 ± 9.1 | 48.1 ± 2.6 | 24.2 ± 3.4 | 38.5 ± 25.3 | 41.6 ± 18.7 | 24.0 ± 20.8 | 30.3 ± 23.5 | 8.5 ± 11.8 | 20.5 ± 11.3 | 21.8 ± 9.6 |
| f8_pep_norm_mse | 57.3 ± 10.2 | 64.9 ± 3.8 | 28.0 ± 8.0 | 44.7 ± 11.2 | 26.3 ± 5.7 | 40.6 ± 13.3 | 19.1 ± 17.5 | 14.1 ± 24.3 | 7.3 ± 7.0 | 10.4 ± 15.8 | 20.5 ± 11.3 | 23.0 ± 10.8 |
| f9_meancentred_mse | 42.4 ± 8.8 | 31.0 ± 15.6 | 47.9 ± 14.6 | 50.2 ± 1.2 | 24.2 ± 3.4 | 39.6 ± 24.5 | 32.1 ± 26.9 | 32.5 ± 28.2 | 21.6 ± 11.3 | 29.7 ± 30.2 | 22.0 ± 12.7 | 31.2 ± 19.6 |
| f10_offset_mse | 38.6 ± 27.0 | 29.4 ± 13.3 | 50.1 ± 16.6 | 47.2 ± 6.2 | 59.9 ± 30.2 | 60.0 ± 48.4 | 33.7 ± 34.2 | 21.0 ± 24.1 | 10.7 ± 15.3 | 10.2 ± 11.1 | 54.8 ± 41.1 | 51.7 ± 35.7 |
| f11_wmse_ess_b0.01 | 40.6 ± 6.2 | 29.3 ± 13.1 | 34.4 ± 25.1 | 44.3 ± 17.7 | 24.2 ± 3.4 | 38.5 ± 25.3 | 24.3 ± 14.8 | 24.0 ± 20.8 | 12.0 ± 14.4 | 8.8 ± 12.4 | 20.5 ± 11.3 | 19.6 ± 8.5 |
| f11_wmse_ess_b0.1 | 42.4 ± 8.8 | 29.3 ± 13.1 | 34.4 ± 25.1 | 44.3 ± 17.7 | 24.2 ± 3.4 | 38.5 ± 25.3 | 24.3 ± 14.8 | 24.0 ± 20.8 | 12.0 ± 14.4 | 8.8 ± 12.4 | 20.5 ± 11.3 | 19.6 ± 8.5 |
| f11_mse_ess_b0.01 | 32.3 ± 18.4 | 28.0 ± 11.6 | 53.4 ± 9.1 | 48.1 ± 2.6 | 24.2 ± 3.4 | 38.5 ± 25.3 | 41.6 ± 18.7 | 24.0 ± 20.8 | 30.3 ± 23.5 | 8.5 ± 11.8 | 20.5 ± 11.3 | 21.8 ± 9.6 |
| f11_mse_ess_b0.1 | 70.6 ± 0.5 | 71.0 ± 0.9 | 78.1 ± 4.0 | 73.1 ± 1.3 | 76.6 ± 3.1 | 73.3 ± 1.2 | 71.8 ± 29.4 | 57.0 ± 49.4 | 55.9 ± 45.5 | 62.9 ± 34.9 | 80.1 ± 0.1 | 80.4 ± 0.4 |
| f12_ws_wmse_5pct | 43.6 ± 10.6 | 41.7 ± 3.2 | 36.4 ± 18.4 | 47.4 ± 20.6 | 25.1 ± 7.0 | 39.7 ± 24.7 | 29.6 ± 6.4 | 27.2 ± 24.4 | 19.8 ± 13.7 | 8.9 ± 12.4 | 34.2 ± 20.6 | 29.6 ± 10.9 |
| f12_ws_wmse_10pct | 51.8 ± 11.8 | 48.4 ± 11.1 | 36.4 ± 18.5 | 48.3 ± 21.3 | 25.8 ± 7.6 | 30.1 ± 32.9 | 36.5 ± 10.3 | 32.5 ± 28.2 | 13.5 ± 13.2 | 8.9 ± 12.4 | 40.4 ± 25.5 | 34.5 ± 18.0 |
| f12_ws_mse_5pct | 34.3 ± 20.2 | 32.0 ± 9.5 | 54.6 ± 9.9 | 50.7 ± 1.9 | 24.1 ± 5.7 | 39.7 ± 24.7 | 42.1 ± 18.1 | 25.5 ± 22.4 | 19.4 ± 13.9 | 8.8 ± 12.2 | 33.1 ± 19.3 | 29.6 ± 10.9 |
| f12_ws_mse_10pct | 45.3 ± 8.4 | 35.5 ± 11.3 | 55.5 ± 10.6 | 51.2 ± 2.7 | 31.6 ± 4.2 | 51.3 ± 27.9 | 41.0 ± 18.4 | 40.0 ± 3.8 | 18.6 ± 13.4 | 29.7 ± 30.2 | 48.8 ± 11.4 | 34.8 ± 18.4 |
| ref_min_work_scale | 73.1 ± 0.1 | 73.2 ± 0.2 | 73.2 ± 15.1 | 73.7 ± 2.7 | 79.1 ± 7.8 | 72.6 ± 1.7 | 67.4 ± 22.1 | 81.9 ± 5.3 | 81.0 ± 5.9 | 77.9 ± 12.8 | 80.7 ± 0.0 | 80.9 ± 0.0 |
| ref_oracle_max_recovery | 73.2 ± 0.1 | 73.2 ± 0.1 | 93.3 ± 1.3 | 90.4 ± 3.0 | 90.9 ± 1.9 | 89.0 ± 1.8 | 96.0 ± 1.4 | 94.9 ± 3.4 | 94.3 ± 3.2 | 96.1 ± 1.0 | 80.8 ± 0.0 | 80.9 ± 0.1 |

**rate arm, scope new_all: recovery % (mean ± SD over 3 replicates) [s4/s5]**

| selector | MSAss maxent seq | MSAss maxent spa | MSAss work_k0 seq | MSAss work_k0 spa | MSAss dihed_k0 seq | MSAss dihed_k0 spa | MSAss work_k20 seq | MSAss work_k20 spa | MSAss dihed_k20 seq | MSAss dihed_k20 spa | filtered maxent seq | filtered maxent spa |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f1_wmse | 73.2 ± 0.5 | 66.7 ± 6.3 | 76.9 ± 4.5 | 72.2 ± 2.7 | 77.9 ± 3.9 | 77.3 ± 3.5 | 29.6 ± 39.5 | 24.2 ± 22.1 | 51.0 ± 38.2 | 22.1 ± 23.9 | 37.7 ± 28.4 | 27.0 ± 11.5 |
| f2_mse | 72.5 ± 1.5 | 53.3 ± 26.8 | 76.5 ± 4.5 | 72.4 ± 2.4 | 78.2 ± 3.5 | 74.1 ± 7.1 | 51.8 ± 36.6 | 24.2 ± 22.1 | 42.3 ± 29.6 | 20.2 ± 24.5 | 38.0 ± 28.3 | 27.5 ± 11.4 |
| f3_chi2_pooled | 73.2 ± 0.5 | 53.8 ± 27.2 | 76.5 ± 4.5 | 72.4 ± 2.4 | 77.9 ± 3.9 | 77.3 ± 3.5 | 26.4 ± 42.2 | 24.2 ± 22.1 | 51.0 ± 38.2 | 20.2 ± 24.5 | 37.7 ± 28.4 | 27.5 ± 11.4 |
| f4_logerr | 69.5 ± 3.8 | 64.0 ± 8.0 | 25.0 ± 43.3 | 45.4 ± 39.4 | 71.6 ± 13.7 | 70.4 ± 7.2 | 30.1 ± 39.1 | 25.0 ± 36.3 | 31.9 ± 40.9 | 9.0 ± 8.6 | 36.4 ± 25.2 | 27.6 ± 10.6 |
| f5a_neg_spearman | 73.8 ± 0.7 | 53.8 ± 27.2 | 81.7 ± 6.0 | 74.2 ± 3.9 | 57.8 ± 46.6 | 77.2 ± 1.6 | 30.7 ± 49.6 | 24.2 ± 22.1 | 6.0 ± 2.6 | 50.1 ± 36.6 | 57.8 ± 40.3 | 61.8 ± 33.1 |
| f5b_neg_pearson | 73.4 ± 0.0 | 53.8 ± 27.2 | 76.5 ± 4.5 | 73.5 ± 5.4 | 80.0 ± 3.0 | 78.2 ± 3.3 | 48.6 ± 42.1 | 24.2 ± 22.1 | 43.6 ± 28.2 | 20.2 ± 24.5 | 57.8 ± 40.3 | 61.8 ± 33.2 |
| f6_maxabsz | 72.8 ± 1.0 | 65.4 ± 7.5 | 52.5 ± 46.4 | 71.1 ± 2.9 | 54.3 ± 47.5 | 53.1 ± 42.4 | 23.9 ± 40.4 | 26.2 ± 36.9 | 24.3 ± 33.2 | 8.3 ± 11.1 | 53.1 ± 31.3 | 30.7 ± 17.6 |
| f7a_Qn_pooled | 73.2 ± 0.5 | 53.8 ± 27.2 | 76.5 ± 4.5 | 72.4 ± 2.4 | 77.9 ± 3.9 | 77.3 ± 3.5 | 26.4 ± 42.2 | 24.2 ± 22.1 | 51.0 ± 38.2 | 20.2 ± 24.5 | 37.7 ± 28.4 | 27.5 ± 11.4 |
| f7b_Qn_homo | 72.5 ± 1.5 | 53.3 ± 26.8 | 76.5 ± 4.5 | 72.4 ± 2.4 | 78.2 ± 3.5 | 74.1 ± 7.1 | 51.8 ± 36.6 | 24.2 ± 22.1 | 42.3 ± 29.6 | 20.2 ± 24.5 | 38.0 ± 28.3 | 27.5 ± 11.4 |
| f8_pep_norm_mse | 73.5 ± 0.7 | 68.0 ± 8.7 | 79.4 ± 3.1 | 76.8 ± 1.1 | 78.0 ± 1.4 | 76.6 ± 1.0 | 30.7 ± 43.3 | 1.4 ± 2.4 | 12.9 ± 13.6 | 12.3 ± 7.0 | 37.6 ± 28.6 | 35.1 ± 17.3 |
| f9_meancentred_mse | 73.6 ± 0.7 | 73.1 ± 0.2 | 79.6 ± 2.7 | 76.9 ± 1.1 | 78.2 ± 1.4 | 78.8 ± 2.3 | 6.2 ± 5.6 | 40.9 ± 39.4 | 5.9 ± 4.8 | 10.9 ± 9.4 | 37.1 ± 27.5 | 42.7 ± 28.0 |
| f10_offset_mse | 73.9 ± 0.6 | 53.8 ± 27.2 | 80.0 ± 2.9 | 78.1 ± 1.4 | 55.9 ± 44.8 | 79.2 ± 1.9 | 24.9 ± 39.6 | 25.9 ± 24.3 | 39.8 ± 26.4 | 20.0 ± 24.6 | 57.8 ± 40.3 | 61.8 ± 33.2 |
| f11_wmse_ess_b0.01 | 73.2 ± 0.5 | 66.7 ± 6.3 | 76.5 ± 4.5 | 72.2 ± 2.7 | 77.9 ± 3.9 | 77.3 ± 3.5 | 29.6 ± 39.5 | 24.2 ± 22.1 | 51.0 ± 38.2 | 22.1 ± 23.9 | 37.7 ± 28.4 | 27.0 ± 11.5 |
| f11_wmse_ess_b0.1 | 73.2 ± 0.5 | 68.4 ± 3.9 | 76.5 ± 4.5 | 71.6 ± 2.8 | 77.9 ± 3.9 | 77.3 ± 3.5 | 29.6 ± 39.5 | 24.2 ± 22.1 | 51.0 ± 38.2 | 42.2 ± 39.9 | 39.1 ± 30.6 | 27.0 ± 11.5 |
| f11_mse_ess_b0.01 | 72.2 ± 0.6 | 72.1 ± 0.5 | 77.6 ± 3.2 | 74.1 ± 3.1 | 77.9 ± 3.9 | 74.8 ± 1.4 | 53.4 ± 38.2 | 40.0 ± 38.5 | 55.8 ± 41.2 | 42.2 ± 39.9 | 55.5 ± 37.9 | 58.1 ± 36.3 |
| f11_mse_ess_b0.1 | 72.8 ± 0.1 | 72.7 ± 0.2 | 77.7 ± 3.2 | 74.3 ± 1.7 | 78.2 ± 3.6 | 74.4 ± 1.7 | 72.9 ± 13.8 | 83.1 ± 4.7 | 81.2 ± 0.9 | 80.1 ± 1.6 | 79.7 ± 0.5 | 80.2 ± 0.5 |
| f12_ws_wmse_5pct | 73.5 ± 0.3 | 68.6 ± 4.3 | 77.3 ± 4.6 | 73.3 ± 2.2 | 78.1 ± 3.8 | 77.1 ± 3.6 | 36.2 ± 37.8 | 26.1 ± 24.6 | 56.4 ± 29.7 | 24.7 ± 24.0 | 58.4 ± 35.8 | 38.2 ± 14.3 |
| f12_ws_wmse_10pct | 73.7 ± 1.0 | 70.0 ± 3.2 | 77.1 ± 4.5 | 75.1 ± 1.3 | 78.7 ± 3.1 | 77.2 ± 3.5 | 39.5 ± 37.5 | 21.1 ± 25.2 | 55.8 ± 42.1 | 44.5 ± 38.8 | 68.8 ± 21.0 | 59.0 ± 30.9 |
| f12_ws_mse_5pct | 73.9 ± 0.8 | 69.2 ± 3.7 | 77.3 ± 4.6 | 73.3 ± 2.2 | 78.7 ± 3.1 | 77.2 ± 3.5 | 60.1 ± 25.1 | 17.7 ± 27.1 | 62.2 ± 31.0 | 43.8 ± 37.8 | 64.2 ± 21.5 | 38.2 ± 14.3 |
| f12_ws_mse_10pct | 73.9 ± 0.8 | 70.0 ± 3.2 | 76.7 ± 4.0 | 76.2 ± 1.1 | 78.7 ± 3.1 | 77.2 ± 3.5 | 64.3 ± 18.4 | 29.1 ± 22.8 | 55.8 ± 42.1 | 44.3 ± 38.5 | 72.0 ± 15.0 | 68.8 ± 13.5 |
| ref_min_work_scale | 73.7 ± 1.0 | 73.2 ± 0.3 | 77.0 ± 0.9 | 75.5 ± 0.9 | 76.4 ± 1.1 | 75.0 ± 0.5 | 78.3 ± 10.8 | 83.0 ± 1.7 | 52.4 ± 41.7 | 65.8 ± 25.4 | 80.9 ± 0.1 | 80.8 ± 0.1 |
| ref_oracle_max_recovery | 74.5 ± 0.7 | 73.4 ± 0.8 | 89.2 ± 1.5 | 90.7 ± 3.6 | 92.4 ± 1.9 | 89.0 ± 1.8 | 92.8 ± 1.8 | 92.0 ± 3.6 | 91.7 ± 4.0 | 90.6 ± 3.0 | 81.0 ± 0.0 | 81.0 ± 0.1 |


The `old_all` and `new_val` versions of T3/T4 are in `tables_old_all.md` and `tables_new_val.md`.


### T5. MaxEnt selections: ESS, state populations (F/PUF1/PUF2/PUF3/U, in %; PUF2 includes PUF2-like) and the selected S `[s4/s5]`

OMC populations are in `sel_summary.csv` (columns `*_mean`), and per replicate in `sel_selected.csv`.

**MaxEnt selections, scope old_val [s4/s5]**

| arm | ens | split | selector | recovery | ESS % | F/PUF1/PUF2/PUF3/U % | S selected (rep 0/1/2) | median step |
|---|---|---|---|---|---|---|---|---|
| full_uptake | MSAss | seq | f1_wmse | 40.9 ± 10.1 | 11.5 | 40/11/2/9/38 | 50/50/1000 | 1500 |
| full_uptake | MSAss | seq | f2_mse | 30.4 ± 15.7 | 5.3 | 28/9/2/21/40 | 100/1000/1000 | 2600 |
| full_uptake | MSAss | seq | f3_chi2_pooled | 40.9 ± 10.1 | 11.5 | 40/11/2/9/38 | 50/50/1000 | 1500 |
| full_uptake | MSAss | seq | f4_logerr | 32.0 ± 18.7 | 11.3 | 30/9/2/18/41 | 50/1000/1000 | 1200 |
| full_uptake | MSAss | seq | f5a_neg_spearman | 34.2 ± 6.4 | 1.3 | 32/11/1/24/33 | 1000/50/1000 | 800 |
| full_uptake | MSAss | seq | f6_maxabsz | 34.5 ± 6.8 | 1.3 | 32/11/1/24/33 | 1000/50/1000 | 2600 |
| full_uptake | MSAss | seq | f8_pep_norm_mse | 32.2 ± 18.5 | 11.3 | 30/9/2/19/40 | 50/1000/1000 | 2300 |
| full_uptake | MSAss | seq | f9_meancentred_mse | 39.7 ± 7.7 | 5.6 | 39/11/2/11/37 | 100/50/1000 | 3800 |
| full_uptake | MSAss | seq | f10_offset_mse | 41.6 ± 10.1 | 11.6 | 41/11/2/9/37 | 50/50/1000 | 3800 |
| full_uptake | MSAss | seq | f12_ws_wmse_5pct | 43.1 ± 11.4 | 14.0 | 43/11/2/8/36 | 50/50/1000 | 3800 |
| full_uptake | MSAss | seq | f12_ws_mse_10pct | 43.3 ± 11.0 | 14.0 | 43/11/2/9/35 | 50/50/1000 | 1600 |
| full_uptake | MSAss | seq | f11_mse_ess_b0.1 | 70.0 ± 0.8 | 97.5 | 75/15/2/3/5 | 10/5/5 | 4700 |
| full_uptake | MSAss | seq | ref_min_work_scale | 73.1 ± 0.1 | 99.9 | 78/14/2/2/3 | 1/1/1 | 4700 |
| full_uptake | MSAss | seq | ref_oracle_max_recovery | 73.2 ± 0.1 | 99.8 | 78/14/2/2/3 | 1/1/1 | 4200 |
| full_uptake | MSAss | spa | f1_wmse | 29.2 ± 5.4 | 0.9 | 26/5/1/14/54 | 500/500/500 | 600 |
| full_uptake | MSAss | spa | f2_mse | 29.2 ± 5.3 | 0.9 | 26/5/1/13/55 | 500/500/500 | 700 |
| full_uptake | MSAss | spa | f3_chi2_pooled | 29.2 ± 5.4 | 0.9 | 26/5/1/14/54 | 500/500/500 | 600 |
| full_uptake | MSAss | spa | f4_logerr | 28.0 ± 5.8 | 0.9 | 25/5/1/12/57 | 500/500/500 | 800 |
| full_uptake | MSAss | spa | f5a_neg_spearman | 30.0 ± 7.4 | 1.0 | 27/5/1/16/51 | 500/100/1000 | 1000 |
| full_uptake | MSAss | spa | f6_maxabsz | 25.6 ± 3.1 | 1.0 | 22/4/0/16/57 | 500/1000/1000 | 2900 |
| full_uptake | MSAss | spa | f8_pep_norm_mse | 26.7 ± 9.7 | 0.9 | 24/5/1/17/54 | 1000/500/500 | 800 |
| full_uptake | MSAss | spa | f9_meancentred_mse | 30.0 ± 13.7 | 1.3 | 28/6/1/17/49 | 1000/50/500 | 1000 |
| full_uptake | MSAss | spa | f10_offset_mse | 31.5 ± 10.1 | 1.3 | 29/6/1/17/48 | 500/50/1000 | 1000 |
| full_uptake | MSAss | spa | f12_ws_wmse_5pct | 28.5 ± 5.3 | 0.9 | 25/5/1/17/53 | 500/500/1000 | 1900 |
| full_uptake | MSAss | spa | f12_ws_mse_10pct | 30.7 ± 8.1 | 1.1 | 28/5/1/17/49 | 500/100/1000 | 1400 |
| full_uptake | MSAss | spa | f11_mse_ess_b0.1 | 58.9 ± 23.3 | 66.7 | 61/12/2/5/20 | 1/5/500 | 3200 |
| full_uptake | MSAss | spa | ref_min_work_scale | 73.2 ± 0.2 | 99.9 | 78/14/2/2/3 | 1/1/1 | 4000 |
| full_uptake | MSAss | spa | ref_oracle_max_recovery | 73.2 ± 0.1 | 99.9 | 78/14/2/2/3 | 1/1/1 | 4000 |
| full_uptake | filtered | seq | f1_wmse | 23.0 ± 13.5 | 0.6 | 17/19/63/0/0 | 1000/1000/500 | 1400 |
| full_uptake | filtered | seq | f2_mse | 37.4 ± 12.0 | 1.6 | 33/23/44/0/0 | 1000/50/500 | 1900 |
| full_uptake | filtered | seq | f3_chi2_pooled | 23.0 ± 13.5 | 0.6 | 17/19/63/0/0 | 1000/1000/500 | 1400 |
| full_uptake | filtered | seq | f4_logerr | 37.4 ± 12.0 | 1.6 | 33/23/44/0/0 | 1000/50/500 | 1900 |
| full_uptake | filtered | seq | f5a_neg_spearman | 44.3 ± 30.3 | 33.5 | 41/32/28/0/0 | 1000/10/1000 | 3700 |
| full_uptake | filtered | seq | f6_maxabsz | 20.2 ± 10.9 | 0.5 | 14/25/61/0/0 | 1000/1000/1000 | 1900 |
| full_uptake | filtered | seq | f8_pep_norm_mse | 37.4 ± 12.0 | 1.6 | 33/23/44/0/0 | 1000/50/500 | 1900 |
| full_uptake | filtered | seq | f9_meancentred_mse | 35.5 ± 13.4 | 1.6 | 31/32/37/0/0 | 1000/50/1000 | 3700 |
| full_uptake | filtered | seq | f10_offset_mse | 20.3 ± 11.5 | 0.6 | 14/27/59/0/0 | 1000/1000/1000 | 3700 |
| full_uptake | filtered | seq | f12_ws_wmse_5pct | 51.0 ± 11.0 | 3.5 | 52/11/37/0/0 | 500/50/50 | 4200 |
| full_uptake | filtered | seq | f12_ws_mse_10pct | 54.9 ± 6.0 | 4.5 | 57/12/32/0/0 | 100/50/50 | 2300 |
| full_uptake | filtered | seq | f11_mse_ess_b0.1 | 80.5 ± 0.3 | 99.9 | 84/13/2/0/0 | 1/5/1 | 113 |
| full_uptake | filtered | seq | ref_min_work_scale | 80.7 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 3100 |
| full_uptake | filtered | seq | ref_oracle_max_recovery | 80.8 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 4300 |
| full_uptake | filtered | spa | f1_wmse | 19.1 ± 8.2 | 0.4 | 13/8/79/0/0 | 1000/1000/1000 | 3300 |
| full_uptake | filtered | spa | f2_mse | 20.1 ± 8.3 | 0.4 | 14/8/78/0/0 | 1000/1000/1000 | 3400 |
| full_uptake | filtered | spa | f3_chi2_pooled | 20.1 ± 8.3 | 0.4 | 14/8/78/0/0 | 1000/1000/1000 | 3400 |
| full_uptake | filtered | spa | f4_logerr | 30.3 ± 13.4 | 0.9 | 26/11/63/0/0 | 100/1000/1000 | 3300 |
| full_uptake | filtered | spa | f5a_neg_spearman | 65.5 ± 25.7 | 66.8 | 67/12/21/0/0 | 10/1/500 | 2100 |
| full_uptake | filtered | spa | f6_maxabsz | 31.3 ± 15.1 | 1.0 | 27/11/62/0/0 | 100/1000/1000 | 3300 |
| full_uptake | filtered | spa | f8_pep_norm_mse | 19.3 ± 8.2 | 0.4 | 14/8/79/0/0 | 1000/1000/1000 | 2200 |
| full_uptake | filtered | spa | f9_meancentred_mse | 24.0 ± 11.7 | 0.5 | 19/9/72/0/0 | 1000/500/1000 | 4900 |
| full_uptake | filtered | spa | f10_offset_mse | 60.5 ± 23.3 | 40.6 | 62/12/26/0/0 | 50/1/500 | 1700 |
| full_uptake | filtered | spa | f12_ws_wmse_5pct | 39.7 ± 7.7 | 1.1 | 38/9/54/0/0 | 100/500/500 | 4200 |
| full_uptake | filtered | spa | f12_ws_mse_10pct | 39.7 ± 7.7 | 1.2 | 38/9/54/0/0 | 100/500/500 | 3200 |
| full_uptake | filtered | spa | f11_mse_ess_b0.1 | 80.4 ± 0.4 | 99.9 | 84/13/2/0/0 | 10/5/5 | 100 |
| full_uptake | filtered | spa | ref_min_work_scale | 80.9 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 4100 |
| full_uptake | filtered | spa | ref_oracle_max_recovery | 80.9 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/5/1 | 1700 |
| rate | MSAss | seq | f1_wmse | 71.3 ± 1.4 | 90.6 | 76/15/2/2/5 | 1/5/50 | 2600 |
| rate | MSAss | seq | f2_mse | 71.3 ± 1.4 | 90.6 | 76/15/2/2/5 | 1/5/50 | 2600 |
| rate | MSAss | seq | f3_chi2_pooled | 71.7 ± 1.3 | 93.5 | 77/15/2/2/4 | 1/1/50 | 4700 |
| rate | MSAss | seq | f4_logerr | 67.6 ± 7.0 | 62.5 | 72/14/2/2/10 | 1/100/50 | 3500 |
| rate | MSAss | seq | f5a_neg_spearman | 72.2 ± 1.0 | 92.6 | 77/15/2/3/3 | 100/1/1 | 2400 |
| rate | MSAss | seq | f6_maxabsz | 71.7 ± 1.3 | 93.5 | 77/15/2/2/4 | 1/1/50 | 4700 |
| rate | MSAss | seq | f8_pep_norm_mse | 70.8 ± 1.9 | 89.6 | 76/15/2/2/5 | 1/5/50 | 3600 |
| rate | MSAss | seq | f9_meancentred_mse | 73.0 ± 0.2 | 98.3 | 78/14/2/3/3 | 50/1/1 | 4200 |
| rate | MSAss | seq | f10_offset_mse | 72.9 ± 0.2 | 98.3 | 78/15/2/3/3 | 50/1/1 | 4000 |
| rate | MSAss | seq | f12_ws_wmse_5pct | 72.5 ± 0.7 | 90.4 | 78/14/2/2/4 | 1/5/100 | 4600 |
| rate | MSAss | seq | f12_ws_mse_10pct | 73.3 ± 0.8 | 94.9 | 78/14/2/2/3 | 1/1/100 | 1100 |
| rate | MSAss | seq | f11_mse_ess_b0.1 | 72.4 ± 0.5 | 99.4 | 78/15/2/2/4 | 1/1/5 | 3800 |
| rate | MSAss | seq | ref_min_work_scale | 73.7 ± 1.0 | 95.3 | 79/14/2/2/3 | 1/1/100 | 4200 |
| rate | MSAss | seq | ref_oracle_max_recovery | 74.5 ± 0.7 | 62.5 | 81/11/1/1/6 | 10/1000/100 | 2800 |
| rate | MSAss | spa | f1_wmse | 67.7 ± 4.4 | 51.2 | 73/14/2/2/9 | 50/100/50 | 3200 |
| rate | MSAss | spa | f2_mse | 67.7 ± 4.4 | 51.2 | 73/14/2/2/9 | 50/100/50 | 3200 |
| rate | MSAss | spa | f3_chi2_pooled | 68.7 ± 5.2 | 67.5 | 74/14/2/3/8 | 5/100/50 | 3900 |
| rate | MSAss | spa | f4_logerr | 64.7 ± 6.9 | 39.1 | 69/13/2/3/12 | 100/500/50 | 1900 |
| rate | MSAss | spa | f5a_neg_spearman | 57.0 ± 20.0 | 37.4 | 60/11/1/21/7 | 500/100/50 | 2600 |
| rate | MSAss | spa | f6_maxabsz | 66.4 ± 7.9 | 64.1 | 71/13/2/3/10 | 5/500/50 | 3200 |
| rate | MSAss | spa | f8_pep_norm_mse | 56.0 ± 19.2 | 35.5 | 59/11/1/21/7 | 500/100/50 | 3200 |
| rate | MSAss | spa | f9_meancentred_mse | 73.0 ± 0.1 | 99.8 | 78/14/2/2/3 | 1/1/1 | 3900 |
| rate | MSAss | spa | f10_offset_mse | 69.7 ± 4.0 | 70.8 | 75/14/2/2/7 | 1/100/50 | 3300 |
| rate | MSAss | spa | f12_ws_wmse_5pct | 69.6 ± 3.7 | 70.6 | 75/14/2/2/7 | 5/100/50 | 2000 |
| rate | MSAss | spa | f12_ws_mse_10pct | 71.0 ± 3.3 | 81.5 | 76/14/2/2/6 | 5/50/10 | 4100 |
| rate | MSAss | spa | f11_mse_ess_b0.1 | 72.0 ± 0.8 | 98.2 | 77/14/2/2/4 | 1/5/10 | 3000 |
| rate | MSAss | spa | ref_min_work_scale | 73.1 ± 0.2 | 99.8 | 78/14/2/2/3 | 1/1/1 | 3900 |
| rate | MSAss | spa | ref_oracle_max_recovery | 73.4 ± 0.8 | 96.6 | 79/14/2/2/3 | 1/1/100 | 3900 |
| rate | filtered | seq | f1_wmse | 63.6 ± 9.6 | 10.8 | 67/12/20/0/0 | 500/50/100 | 2900 |
| rate | filtered | seq | f2_mse | 52.0 ± 16.3 | 4.7 | 52/29/19/0/0 | 500/50/1000 | 2900 |
| rate | filtered | seq | f3_chi2_pooled | 52.0 ± 16.3 | 4.7 | 52/29/19/0/0 | 500/50/1000 | 2900 |
| rate | filtered | seq | f4_logerr | 66.1 ± 7.4 | 19.9 | 70/13/17/0/0 | 100/50/100 | 2600 |
| rate | filtered | seq | f5a_neg_spearman | 76.2 ± 5.3 | 76.7 | 80/13/6/0/0 | 100/1/10 | 3100 |
| rate | filtered | seq | f6_maxabsz | 65.0 ± 10.8 | 25.7 | 69/13/19/0/0 | 50/50/100 | 3100 |
| rate | filtered | seq | f8_pep_norm_mse | 52.0 ± 16.3 | 4.7 | 52/29/19/0/0 | 500/50/1000 | 2900 |
| rate | filtered | seq | f9_meancentred_mse | 51.5 ± 15.3 | 2.8 | 51/34/15/0/0 | 1000/50/1000 | 2400 |
| rate | filtered | seq | f10_offset_mse | 76.3 ± 5.4 | 76.7 | 80/13/6/0/0 | 100/1/10 | 3200 |
| rate | filtered | seq | f12_ws_wmse_5pct | 80.0 ± 1.7 | 99.1 | 84/13/3/0/0 | 1/5/1 | 3700 |
| rate | filtered | seq | f12_ws_mse_10pct | 80.9 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/1/1 | 4000 |
| rate | filtered | seq | f11_mse_ess_b0.1 | 80.4 ± 0.3 | 99.9 | 84/13/2/0/0 | 1/1/1 | 163 |
| rate | filtered | seq | ref_min_work_scale | 80.9 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/1/1 | 4000 |
| rate | filtered | seq | ref_oracle_max_recovery | 81.0 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 3700 |
| rate | filtered | spa | f1_wmse | 27.6 ± 10.6 | 0.7 | 22/30/49/0/0 | 1000/1000/1000 | 4400 |
| rate | filtered | spa | f2_mse | 27.6 ± 10.6 | 0.7 | 22/30/49/0/0 | 1000/1000/1000 | 4400 |
| rate | filtered | spa | f3_chi2_pooled | 30.4 ± 7.5 | 0.7 | 25/30/45/0/0 | 500/1000/1000 | 3400 |
| rate | filtered | spa | f4_logerr | 28.0 ± 10.5 | 0.7 | 22/29/48/0/0 | 1000/1000/1000 | 4400 |
| rate | filtered | spa | f5a_neg_spearman | 78.6 ± 3.5 | 90.8 | 83/14/4/0/0 | 1/1/50 | 2600 |
| rate | filtered | spa | f6_maxabsz | 51.9 ± 27.1 | 33.8 | 50/29/21/0/0 | 10/1000/500 | 3900 |
| rate | filtered | spa | f8_pep_norm_mse | 27.6 ± 10.6 | 0.7 | 22/30/49/0/0 | 1000/1000/1000 | 4400 |
| rate | filtered | spa | f9_meancentred_mse | 42.7 ± 28.0 | 9.1 | 41/24/35/0/0 | 1000/100/1000 | 3300 |
| rate | filtered | spa | f10_offset_mse | 78.6 ± 3.7 | 90.5 | 83/14/4/0/0 | 5/5/50 | 3900 |
| rate | filtered | spa | f12_ws_wmse_5pct | 48.0 ± 7.9 | 2.2 | 47/27/26/0/0 | 100/500/500 | 4200 |
| rate | filtered | spa | f12_ws_mse_10pct | 72.2 ± 6.2 | 46.7 | 76/16/8/0/0 | 50/100/10 | 3000 |
| rate | filtered | spa | f11_mse_ess_b0.1 | 80.2 ± 0.3 | 99.8 | 84/13/2/0/0 | 5/5/1 | 3400 |
| rate | filtered | spa | ref_min_work_scale | 80.8 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/1/1 | 3600 |
| rate | filtered | spa | ref_oracle_max_recovery | 81.0 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/5/1 | 3500 |

**MaxEnt selections, scope new_all [s4/s5]**

| arm | ens | split | selector | recovery | ESS % | F/PUF1/PUF2/PUF3/U % | S selected (rep 0/1/2) | median step |
|---|---|---|---|---|---|---|---|---|
| full_uptake | MSAss | seq | f1_wmse | 40.6 ± 6.2 | 5.6 | 40/11/2/12/35 | 100/50/1000 | 3800 |
| full_uptake | MSAss | seq | f2_mse | 30.4 ± 15.7 | 5.3 | 28/9/2/21/40 | 100/1000/1000 | 3800 |
| full_uptake | MSAss | seq | f3_chi2_pooled | 30.5 ± 15.7 | 5.3 | 28/9/2/21/40 | 100/1000/1000 | 3800 |
| full_uptake | MSAss | seq | f4_logerr | 41.9 ± 9.5 | 11.6 | 41/11/2/9/37 | 50/50/1000 | 1200 |
| full_uptake | MSAss | seq | f5a_neg_spearman | 39.8 ± 30.0 | 33.9 | 39/9/1/16/35 | 1/1000/1000 | 3800 |
| full_uptake | MSAss | seq | f6_maxabsz | 34.5 ± 6.8 | 1.3 | 32/11/1/24/33 | 1000/50/1000 | 2600 |
| full_uptake | MSAss | seq | f8_pep_norm_mse | 57.3 ± 10.2 | 43.5 | 60/14/3/8/14 | 100/10/10 | 3800 |
| full_uptake | MSAss | seq | f9_meancentred_mse | 42.4 ± 8.8 | 11.7 | 42/11/2/9/35 | 50/50/1000 | 2900 |
| full_uptake | MSAss | seq | f10_offset_mse | 38.6 ± 27.0 | 32.2 | 38/10/1/17/35 | 10/1000/1000 | 2900 |
| full_uptake | MSAss | seq | f12_ws_wmse_5pct | 43.6 ± 10.6 | 14.0 | 44/11/2/9/35 | 50/50/1000 | 2900 |
| full_uptake | MSAss | seq | f12_ws_mse_10pct | 45.3 ± 8.4 | 14.1 | 46/11/2/7/35 | 50/50/500 | 1600 |
| full_uptake | MSAss | seq | f11_mse_ess_b0.1 | 70.6 ± 0.5 | 98.2 | 76/15/2/3/5 | 5/5/5 | 113 |
| full_uptake | MSAss | seq | ref_min_work_scale | 73.1 ± 0.1 | 99.9 | 78/14/2/2/3 | 1/1/1 | 4700 |
| full_uptake | MSAss | seq | ref_oracle_max_recovery | 73.2 ± 0.1 | 99.8 | 78/14/2/2/3 | 1/1/1 | 4200 |
| full_uptake | MSAss | spa | f1_wmse | 29.3 ± 13.1 | 1.1 | 27/5/1/21/47 | 1000/100/1000 | 3400 |
| full_uptake | MSAss | spa | f2_mse | 28.0 ± 11.6 | 1.0 | 25/5/1/20/49 | 1000/100/1000 | 2400 |
| full_uptake | MSAss | spa | f3_chi2_pooled | 28.4 ± 11.9 | 1.0 | 26/5/1/21/48 | 1000/100/1000 | 3200 |
| full_uptake | MSAss | spa | f4_logerr | 29.2 ± 5.4 | 1.0 | 26/5/1/17/52 | 500/500/1000 | 1900 |
| full_uptake | MSAss | spa | f5a_neg_spearman | 26.3 ± 9.4 | 0.9 | 23/4/0/20/51 | 1000/500/1000 | 2400 |
| full_uptake | MSAss | spa | f6_maxabsz | 37.2 ± 6.5 | 1.3 | 36/7/1/8/48 | 50/100/1000 | 3900 |
| full_uptake | MSAss | spa | f8_pep_norm_mse | 64.9 ± 3.8 | 60.1 | 70/13/2/3/12 | 10/10/10 | 3300 |
| full_uptake | MSAss | spa | f9_meancentred_mse | 31.0 ± 15.6 | 1.5 | 29/6/1/21/44 | 1000/50/1000 | 3200 |
| full_uptake | MSAss | spa | f10_offset_mse | 29.4 ± 13.3 | 1.1 | 27/5/1/21/46 | 1000/100/1000 | 2400 |
| full_uptake | MSAss | spa | f12_ws_wmse_5pct | 41.7 ± 3.2 | 1.7 | 42/8/1/4/45 | 50/100/500 | 1200 |
| full_uptake | MSAss | spa | f12_ws_mse_10pct | 35.5 ± 11.3 | 1.7 | 34/7/1/13/45 | 500/50/500 | 1200 |
| full_uptake | MSAss | spa | f11_mse_ess_b0.1 | 71.0 ± 0.9 | 97.4 | 76/14/2/3/5 | 5/5/5 | 112 |
| full_uptake | MSAss | spa | ref_min_work_scale | 73.2 ± 0.2 | 99.9 | 78/14/2/2/3 | 1/1/1 | 4000 |
| full_uptake | MSAss | spa | ref_oracle_max_recovery | 73.2 ± 0.1 | 99.9 | 78/14/2/2/3 | 1/1/1 | 4000 |
| full_uptake | filtered | seq | f1_wmse | 20.5 ± 11.3 | 0.6 | 14/27/58/0/0 | 1000/1000/1000 | 3700 |
| full_uptake | filtered | seq | f2_mse | 20.5 ± 11.3 | 0.6 | 14/27/58/0/0 | 1000/1000/1000 | 3700 |
| full_uptake | filtered | seq | f3_chi2_pooled | 20.5 ± 11.3 | 0.6 | 14/27/58/0/0 | 1000/1000/1000 | 3700 |
| full_uptake | filtered | seq | f4_logerr | 20.5 ± 11.3 | 0.6 | 14/27/58/0/0 | 1000/1000/1000 | 3700 |
| full_uptake | filtered | seq | f5a_neg_spearman | 56.2 ± 42.3 | 66.7 | 58/9/33/0/0 | 1/1000/1 | 2600 |
| full_uptake | filtered | seq | f6_maxabsz | 20.2 ± 10.9 | 0.5 | 14/25/61/0/0 | 1000/1000/1000 | 1900 |
| full_uptake | filtered | seq | f8_pep_norm_mse | 20.5 ± 11.3 | 0.6 | 14/27/58/0/0 | 1000/1000/1000 | 3700 |
| full_uptake | filtered | seq | f9_meancentred_mse | 22.0 ± 12.7 | 0.6 | 16/27/57/0/0 | 1000/1000/1000 | 3000 |
| full_uptake | filtered | seq | f10_offset_mse | 54.8 ± 41.1 | 65.4 | 56/10/34/0/0 | 10/1000/10 | 4500 |
| full_uptake | filtered | seq | f12_ws_wmse_5pct | 34.2 ± 20.6 | 1.0 | 32/7/61/0/0 | 500/500/100 | 2400 |
| full_uptake | filtered | seq | f12_ws_mse_10pct | 48.8 ± 11.4 | 3.2 | 49/10/40/0/0 | 100/100/50 | 2400 |
| full_uptake | filtered | seq | f11_mse_ess_b0.1 | 80.1 ± 0.1 | 99.8 | 84/14/2/0/0 | 5/5/5 | 100 |
| full_uptake | filtered | seq | ref_min_work_scale | 80.7 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 3100 |
| full_uptake | filtered | seq | ref_oracle_max_recovery | 80.8 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 4300 |
| full_uptake | filtered | spa | f1_wmse | 19.6 ± 8.5 | 0.4 | 14/8/78/0/0 | 1000/1000/1000 | 3400 |
| full_uptake | filtered | spa | f2_mse | 21.8 ± 9.6 | 0.5 | 16/8/76/0/0 | 1000/500/1000 | 3400 |
| full_uptake | filtered | spa | f3_chi2_pooled | 21.8 ± 9.6 | 0.5 | 16/8/76/0/0 | 1000/500/1000 | 3400 |
| full_uptake | filtered | spa | f4_logerr | 19.1 ± 8.2 | 0.4 | 13/8/79/0/0 | 1000/1000/1000 | 3300 |
| full_uptake | filtered | spa | f5a_neg_spearman | 57.5 ± 40.3 | 66.8 | 59/9/32/0/0 | 1000/1/1 | 1700 |
| full_uptake | filtered | spa | f6_maxabsz | 19.1 ± 8.2 | 0.4 | 13/8/79/0/0 | 1000/1000/1000 | 3300 |
| full_uptake | filtered | spa | f8_pep_norm_mse | 23.0 ± 10.8 | 0.5 | 18/8/74/0/0 | 1000/500/1000 | 3400 |
| full_uptake | filtered | spa | f9_meancentred_mse | 31.2 ± 19.6 | 0.9 | 28/7/65/0/0 | 1000/500/100 | 4100 |
| full_uptake | filtered | spa | f10_offset_mse | 51.7 ± 35.7 | 32.5 | 53/10/37/0/0 | 1000/100/50 | 1500 |
| full_uptake | filtered | spa | f12_ws_wmse_5pct | 29.6 ± 10.9 | 0.6 | 26/6/68/0/0 | 500/500/500 | 3200 |
| full_uptake | filtered | spa | f12_ws_mse_10pct | 34.8 ± 18.4 | 1.1 | 32/7/61/0/0 | 500/500/100 | 2600 |
| full_uptake | filtered | spa | f11_mse_ess_b0.1 | 80.4 ± 0.4 | 99.9 | 84/13/2/0/0 | 10/5/5 | 100 |
| full_uptake | filtered | spa | ref_min_work_scale | 80.9 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 4100 |
| full_uptake | filtered | spa | ref_oracle_max_recovery | 80.9 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/5/1 | 1700 |
| rate | MSAss | seq | f1_wmse | 73.2 ± 0.5 | 66.8 | 80/11/1/2/6 | 1/1000/1 | 2800 |
| rate | MSAss | seq | f2_mse | 72.5 ± 1.5 | 44.6 | 79/11/3/1/7 | 1/1000/500 | 2600 |
| rate | MSAss | seq | f3_chi2_pooled | 73.2 ± 0.5 | 66.8 | 80/11/1/2/6 | 1/1000/1 | 2800 |
| rate | MSAss | seq | f4_logerr | 69.5 ± 3.8 | 60.8 | 75/13/2/2/9 | 1/500/50 | 4400 |
| rate | MSAss | seq | f5a_neg_spearman | 73.8 ± 0.7 | 66.5 | 80/11/1/1/6 | 10/1000/1 | 3200 |
| rate | MSAss | seq | f6_maxabsz | 72.8 ± 1.0 | 61.4 | 79/11/1/1/7 | 1/1000/50 | 2600 |
| rate | MSAss | seq | f8_pep_norm_mse | 73.5 ± 0.7 | 95.3 | 79/14/2/3/3 | 100/1/1 | 4200 |
| rate | MSAss | seq | f9_meancentred_mse | 73.6 ± 0.7 | 95.2 | 79/14/2/3/3 | 100/1/1 | 3400 |
| rate | MSAss | seq | f10_offset_mse | 73.9 ± 0.6 | 66.5 | 80/11/1/1/6 | 10/1000/1 | 3200 |
| rate | MSAss | seq | f12_ws_wmse_5pct | 73.5 ± 0.3 | 66.8 | 80/11/1/1/6 | 1/1000/1 | 3400 |
| rate | MSAss | seq | f12_ws_mse_10pct | 73.9 ± 0.8 | 62.3 | 80/11/1/1/6 | 1/1000/100 | 4200 |
| rate | MSAss | seq | f11_mse_ess_b0.1 | 72.8 ± 0.1 | 99.6 | 78/14/2/2/3 | 1/1/5 | 122 |
| rate | MSAss | seq | ref_min_work_scale | 73.7 ± 1.0 | 95.3 | 79/14/2/2/3 | 1/1/100 | 4200 |
| rate | MSAss | seq | ref_oracle_max_recovery | 74.5 ± 0.7 | 62.5 | 81/11/1/1/6 | 10/1000/100 | 2800 |
| rate | MSAss | spa | f1_wmse | 66.7 ± 6.3 | 45.9 | 72/14/2/3/10 | 100/500/1 | 3000 |
| rate | MSAss | spa | f2_mse | 53.3 ± 26.8 | 41.1 | 56/10/1/26/6 | 1000/100/10 | 2200 |
| rate | MSAss | spa | f3_chi2_pooled | 53.8 ± 27.2 | 41.8 | 56/10/1/26/6 | 1000/100/1 | 2000 |
| rate | MSAss | spa | f4_logerr | 64.0 ± 8.0 | 38.7 | 69/13/2/3/14 | 100/1000/50 | 1800 |
| rate | MSAss | spa | f5a_neg_spearman | 53.8 ± 27.2 | 41.7 | 56/10/1/26/6 | 1000/100/1 | 2200 |
| rate | MSAss | spa | f6_maxabsz | 65.4 ± 7.5 | 43.9 | 70/13/2/3/12 | 100/500/10 | 2900 |
| rate | MSAss | spa | f8_pep_norm_mse | 68.0 ± 8.7 | 69.9 | 73/13/2/3/9 | 1/1000/1 | 800 |
| rate | MSAss | spa | f9_meancentred_mse | 73.1 ± 0.2 | 99.8 | 78/14/2/2/3 | 1/1/1 | 4100 |
| rate | MSAss | spa | f10_offset_mse | 53.8 ± 27.2 | 41.8 | 56/10/1/26/6 | 1000/100/1 | 2000 |
| rate | MSAss | spa | f12_ws_wmse_5pct | 68.6 ± 4.3 | 50.5 | 74/14/2/2/9 | 100/100/1 | 2000 |
| rate | MSAss | spa | f12_ws_mse_10pct | 70.0 ± 3.2 | 65.7 | 75/14/2/2/7 | 50/50/1 | 2600 |
| rate | MSAss | spa | f11_mse_ess_b0.1 | 72.7 ± 0.2 | 99.5 | 78/14/2/2/4 | 1/1/5 | 4000 |
| rate | MSAss | spa | ref_min_work_scale | 73.2 ± 0.3 | 99.8 | 78/14/2/2/3 | 1/1/1 | 3900 |
| rate | MSAss | spa | ref_oracle_max_recovery | 73.4 ± 0.8 | 96.6 | 79/14/2/2/3 | 1/1/100 | 3900 |
| rate | filtered | seq | f1_wmse | 37.7 ± 28.4 | 3.8 | 36/27/38/0/0 | 500/1000/1000 | 2900 |
| rate | filtered | seq | f2_mse | 38.0 ± 28.3 | 3.8 | 36/27/37/0/0 | 500/1000/1000 | 2400 |
| rate | filtered | seq | f3_chi2_pooled | 37.7 ± 28.4 | 3.8 | 36/27/38/0/0 | 500/1000/1000 | 2900 |
| rate | filtered | seq | f4_logerr | 36.4 ± 25.2 | 1.4 | 34/30/36/0/0 | 1000/1000/1000 | 4300 |
| rate | filtered | seq | f5a_neg_spearman | 57.8 ± 40.3 | 66.7 | 59/9/32/0/0 | 1/1000/1 | 3700 |
| rate | filtered | seq | f6_maxabsz | 53.1 ± 31.3 | 24.8 | 54/10/36/0/0 | 50/500/100 | 2600 |
| rate | filtered | seq | f8_pep_norm_mse | 37.6 ± 28.6 | 3.8 | 35/27/38/0/0 | 500/1000/1000 | 2900 |
| rate | filtered | seq | f9_meancentred_mse | 37.1 ± 27.5 | 1.4 | 35/30/36/0/0 | 1000/1000/1000 | 2400 |
| rate | filtered | seq | f10_offset_mse | 57.8 ± 40.3 | 66.7 | 59/9/32/0/0 | 1/1000/1 | 3700 |
| rate | filtered | seq | f12_ws_wmse_5pct | 58.4 ± 35.8 | 65.0 | 59/10/31/0/0 | 5/500/10 | 4100 |
| rate | filtered | seq | f12_ws_mse_10pct | 72.0 ± 15.0 | 67.7 | 76/13/12/0/0 | 1/50/1 | 4600 |
| rate | filtered | seq | f11_mse_ess_b0.1 | 79.7 ± 0.5 | 99.2 | 84/13/3/0/0 | 1/5/5 | 104 |
| rate | filtered | seq | ref_min_work_scale | 80.9 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/1/1 | 4000 |
| rate | filtered | seq | ref_oracle_max_recovery | 81.0 ± 0.0 | 99.9 | 85/13/2/0/0 | 1/1/1 | 3700 |
| rate | filtered | spa | f1_wmse | 27.0 ± 11.5 | 0.7 | 21/30/49/0/0 | 1000/1000/1000 | 3300 |
| rate | filtered | spa | f2_mse | 27.5 ± 11.4 | 0.7 | 22/30/48/0/0 | 1000/1000/1000 | 3300 |
| rate | filtered | spa | f3_chi2_pooled | 27.5 ± 11.4 | 0.7 | 22/30/48/0/0 | 1000/1000/1000 | 3300 |
| rate | filtered | spa | f4_logerr | 27.6 ± 10.6 | 0.7 | 22/30/49/0/0 | 1000/1000/1000 | 4400 |
| rate | filtered | spa | f5a_neg_spearman | 61.8 ± 33.1 | 66.8 | 63/10/27/0/0 | 500/5/1 | 3600 |
| rate | filtered | spa | f6_maxabsz | 30.7 ± 17.6 | 1.1 | 26/25/49/0/0 | 1000/1000/500 | 3300 |
| rate | filtered | spa | f8_pep_norm_mse | 35.1 ± 17.3 | 1.3 | 31/24/45/0/0 | 1000/500/500 | 3300 |
| rate | filtered | spa | f9_meancentred_mse | 42.7 ± 28.0 | 9.1 | 41/24/35/0/0 | 1000/100/1000 | 3300 |
| rate | filtered | spa | f10_offset_mse | 61.8 ± 33.2 | 66.8 | 63/10/27/0/0 | 500/5/1 | 3400 |
| rate | filtered | spa | f12_ws_wmse_5pct | 38.2 ± 14.3 | 1.3 | 35/25/41/0/0 | 500/500/500 | 3200 |
| rate | filtered | spa | f12_ws_mse_10pct | 68.8 ± 13.5 | 44.4 | 72/14/13/0/0 | 100/100/10 | 2400 |
| rate | filtered | spa | f11_mse_ess_b0.1 | 80.2 ± 0.5 | 99.7 | 84/13/2/0/0 | 5/10/5 | 130 |
| rate | filtered | spa | ref_min_work_scale | 80.8 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/1/1 | 3600 |
| rate | filtered | spa | ref_oracle_max_recovery | 81.0 ± 0.1 | 99.9 | 85/13/2/0/0 | 1/5/1 | 3500 |


### T6. Within-replicate Spearman ρ(−score, recovery) across the pool, mean over replicates and models `[s4]`

A positive value means a better score goes with higher recovery. Per-model values are in `sel_spearman.csv` and `sel_spearman_summary.csv`.

| score | full_uptake/new_all | full_uptake/new_val | full_uptake/old_all | full_uptake/old_val | iso/all | iso/val | rate/new_all | rate/new_val | rate/old_all | rate/old_val |
|---|---|---|---|---|---|---|---|---|---|---|
| f1_wmse | -0.6 | -0.63 | -0.68 | -0.66 | 0.86 | 0.84 | -0.36 | -0.33 | -0.48 | -0.4 |
| f2_mse | -0.61 | -0.59 | -0.68 | -0.64 | 0.86 | 0.84 | -0.42 | -0.31 | -0.49 | -0.4 |
| f3_chi2_pooled | -0.63 | -0.62 | -0.69 | -0.65 | 0.86 | 0.84 | -0.38 | -0.31 | -0.49 | -0.35 |
| f4_logerr | -0.63 | -0.64 | -0.66 | -0.64 | 0.86 | 0.84 | -0.55 | -0.51 | -0.59 | -0.48 |
| f5a_neg_spearman | 0.1 | -0.24 | -0.0 | -0.22 | 0.86 | 0.84 | 0.12 | -0.05 | -0.05 | -0.08 |
| f5b_neg_pearson | -0.07 | -0.34 | -0.15 | -0.37 | 0.86 | 0.84 | -0.05 | -0.01 | -0.11 | -0.06 |
| f6_maxabsz | -0.63 | -0.64 | -0.67 | -0.65 | 0.84 | 0.84 | -0.34 | -0.35 | -0.52 | -0.33 |
| f7a_Qn_pooled | -0.63 | -0.62 | -0.69 | -0.65 | -0.48 | -0.46 | -0.38 | -0.31 | -0.49 | -0.35 |
| f7b_Qn_homo | -0.61 | -0.59 | -0.68 | -0.64 | -0.48 | -0.46 | -0.42 | -0.31 | -0.49 | -0.4 |
| f8_pep_norm_mse | -0.27 | -0.57 | -0.67 | -0.64 | 0.86 | 0.84 | -0.08 | -0.28 | -0.49 | -0.38 |
| f9_meancentred_mse | -0.42 | -0.44 | -0.53 | -0.52 | 0.86 | 0.84 | -0.03 | 0.12 | -0.33 | -0.07 |
| f10_offset_mse | -0.06 | -0.23 | -0.11 | -0.32 | 0.86 | 0.84 | 0.11 | 0.14 | 0.02 | 0.04 |
| f11_wmse_ess_b0.01 | -0.6 | -0.62 | -0.69 | -0.66 | -0.25 | -0.24 | -0.36 | -0.32 | -0.47 | -0.38 |
| f11_wmse_ess_b0.1 | -0.6 | -0.62 | -0.69 | -0.65 | -0.34 | -0.34 | -0.33 | -0.25 | -0.45 | -0.29 |
| f11_mse_ess_b0.01 | -0.57 | -0.51 | -0.67 | -0.56 | -0.25 | -0.24 | -0.1 | -0.01 | -0.26 | -0.11 |
| f11_mse_ess_b0.1 | 0.06 | 0.05 | -0.02 | -0.0 | -0.34 | -0.34 | 0.1 | 0.13 | 0.05 | 0.09 |
| work_scale | 0.53 | 0.53 | 0.53 | 0.53 | -0.03 | -0.03 | 0.39 | 0.39 | 0.39 | 0.39 |
| val_closed_sigma_mse | nan | nan | nan | nan | nan | 0.84 | nan | nan | nan | nan |

**MaxEnt pools only [s4]**

| arm | scope | ensemble | f1_wmse | f2_mse | f4_logerr | f9_meancentred_mse | f10_offset_mse | f5a_neg_spearman | work_scale |
|---|---|---|---|---|---|---|---|---|---|
| full_uptake | new_all | AF2_MSAss | -0.75 | -0.87 | -0.86 | -0.82 | -0.58 | -0.29 | 0.99 |
| full_uptake | new_all | AF2_filtered | -0.99 | -0.98 | -0.99 | -0.88 | 0.14 | 0.41 | 0.99 |
| full_uptake | new_val | AF2_MSAss | -0.84 | -0.83 | -0.88 | -0.77 | -0.66 | -0.76 | 0.99 |
| full_uptake | new_val | AF2_filtered | -0.97 | -0.94 | -0.91 | -0.87 | -0.36 | -0.11 | 0.99 |
| full_uptake | old_all | AF2_MSAss | -0.96 | -0.96 | -0.95 | -0.86 | -0.63 | -0.63 | 0.99 |
| full_uptake | old_all | AF2_filtered | -0.99 | -0.99 | -0.99 | -0.98 | 0.14 | 0.35 | 0.99 |
| full_uptake | old_val | AF2_MSAss | -0.88 | -0.88 | -0.9 | -0.84 | -0.8 | -0.81 | 0.99 |
| full_uptake | old_val | AF2_filtered | -0.97 | -0.93 | -0.9 | -0.92 | -0.62 | -0.02 | 0.99 |
| iso | all | ISO_BI | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | -0.87 |
| iso | all | ISO_TRI | 0.94 | 0.94 | 0.95 | 0.94 | 0.94 | 0.95 | -0.73 |
| iso | val | ISO_BI | 0.97 | 0.97 | 0.96 | 0.96 | 0.97 | 0.95 | -0.87 |
| iso | val | ISO_TRI | 0.92 | 0.92 | 0.93 | 0.92 | 0.92 | 0.92 | -0.73 |
| rate | new_all | AF2_MSAss | 0.35 | 0.03 | -0.29 | 0.17 | 0.32 | 0.12 | 0.81 |
| rate | new_all | AF2_filtered | -0.97 | -0.99 | -0.99 | -0.68 | 0.39 | 0.38 | 0.99 |
| rate | new_val | AF2_MSAss | 0.21 | 0.31 | -0.34 | 0.63 | 0.56 | 0.23 | 0.81 |
| rate | new_val | AF2_filtered | -0.88 | -0.85 | -0.64 | -0.28 | 0.64 | 0.59 | 0.99 |
| rate | old_all | AF2_MSAss | -0.21 | -0.26 | -0.47 | 0.33 | 0.32 | 0.06 | 0.81 |
| rate | old_all | AF2_filtered | -0.98 | -0.99 | -0.99 | -0.96 | 0.37 | 0.38 | 0.99 |
| rate | old_val | AF2_MSAss | 0.03 | 0.02 | -0.34 | 0.56 | 0.59 | 0.08 | 0.81 |
| rate | old_val | AF2_filtered | -0.88 | -0.85 | -0.49 | -0.5 | 0.62 | 0.61 | 0.99 |

### T7. The >80% candidates already present in the MoPrP pools (old_val scores) `[s6]`

| arm | ens | model | pool n | % cands >80 | median ESS% of >80 | % of >80 that are 1-frame | max rec (ESS≥10%) | f1 pct-rank of >80 (median) | f10 pct-rank | f5a pct-rank |
|---|---|---|---|---|---|---|---|---|---|---|
| full_uptake | MSAss | maxent | 2186 | 0.0 | nan | nan | 73.4 | nan | nan | nan |
| full_uptake | MSAss | omc_dihed_k0 | 6624 | 52.7 | 0.2 | 73.0 | 92.4 | 0.73 | 0.62 | 0.56 |
| full_uptake | MSAss | omc_dihed_k20 | 6916 | 18.0 | 0.2 | 82.0 | 97.8 | 0.86 | 0.62 | 0.57 |
| full_uptake | MSAss | omc_work_k0 | 6687 | 43.2 | 0.2 | 68.0 | 94.2 | 0.76 | 0.69 | 0.67 |
| full_uptake | MSAss | omc_work_k20 | 6881 | 13.2 | 0.2 | 81.0 | 98.3 | 0.87 | 0.65 | 0.69 |
| full_uptake | filtered | maxent | 2197 | 32.6 | 99.9 | 0.0 | 81.0 | 0.84 | 0.8 | 0.36 |
| rate | MSAss | maxent | 2165 | 0.0 | nan | nan | 75.2 | nan | nan | nan |
| rate | MSAss | omc_dihed_k0 | 6686 | 29.9 | 62.9 | 30.0 | 94.2 | 0.8 | 0.75 | 0.76 |
| rate | MSAss | omc_dihed_k20 | 6874 | 22.7 | 0.4 | 71.0 | 93.1 | 0.59 | 0.68 | 0.66 |
| rate | MSAss | omc_work_k0 | 6708 | 27.8 | 60.8 | 36.0 | 94.8 | 0.81 | 0.77 | 0.77 |
| rate | MSAss | omc_work_k20 | 6872 | 21.2 | 0.4 | 76.0 | 94.9 | 0.64 | 0.54 | 0.58 |
| rate | filtered | maxent | 2196 | 20.0 | 99.9 | 0.0 | 81.1 | 0.9 | 0.29 | 0.29 |

---

## Caveats

- **The ISO check is weak for σ-based criteria.** ISO data have no experimental σ, so f3 ≡ f2 and f6 = max|r|. f7's failure partly reflects the missing noise model.
- **The f11 failure is a scale artefact.** β is absolute. A penalty normalised to the score's scale (for example relative to the prior's score) was not tested.
- **f1 on ISO is plain MSE**, as the brief specifies.
- **Train and validation overlap in 4 of 6 MoPrP splits** (for example spatial split 0 has val {0–5, 11} and train {4–11}). So `old_val` is not a clean hold-out. The `old_all` and `new_all` scopes are reported for that reason.
- **No floor is applied to σ_pooled.** The earlier gci2 pass floored it at 0.0586. Unfloored, f3/f6/f7a are slightly more weighted toward the low-σ peptides.
- **PUF2 includes the single PUF2-like frame (label 5)** in AF2_filtered.
- **Recovery comes from candidates.csv,** which uses `calculate_recovery_percentage` with the `state_ratios.json` target (97.1 / 2.4 / 0.5). The populations come from the P2 weights and the cluster labels in `_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters/`.
- **The `work/` subfolder holds about 0.66 GB of intermediates** (weights and predictions per candidate). It can be deleted. Rebuild it with s1 → s2 → s3 → s4 → s5 → s6, each taking under a minute or two with 8 processes.
