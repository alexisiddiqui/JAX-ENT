# Averaging-mode soft-regression baseline

Recorded on 2026-09-24 for the Example 1--3 test campaigns. This document is a
soft-regression reference for frame-averaging semantics, validation-based model
selection, and conformational recovery. It is not an exact numerical test:
optimizer, JAX/XLA, hardware, and dependency changes can cause small differences.

## Adopted fitting and selection policy (2026-10-04)

Going forward, fit the experimental uptake with ordinary MSE, then select each
replicate using its validation data: raw peptide × timepoint `MoPrP.weights`
weighted MSE for MoPrP, and closed-coordinate Sigma-MSE for ISO validation.
Use `MoPrP.weights` directly, without squaring, inversion, or division by their
sum. Keep optimization and selection on the identical fitted forward model and
saved parameters. Recovery, ESS, and ground-truth Sigma are evaluation-only.
The historical campaigns below retain their original stated conventions.

The ISO policy SI campaign uses full frame-wise uptake, sequence-cluster and
spatial splits, and three replicates. All trajectories are fitted with MSE;
solid and dashed curves compare ordinary MSE and closed-Sigma selection on the
existing candidate pool: saved trajectory, convergence, and running-best
states, deduplicated by step and weights. Closed-Sigma retains
alpha zero, numerical stabilization, full inversion before validation subsetting,
and trace normalization.

The first SI sweeps MaxEnt scaling for ISO BI and TRI. The second compares
all-pairs Work Scale, optimal C-alpha RMSD, and unrelaxed PyRosetta `ref2015`
kernels for ISO TRI, with independent strength and bandwidth sweeps. Each
distance matrix is divided by its positive pairwise-distance median; this
structural bandwidth scaling does not normalize experimental loss weights.
MaxEnt references are black. Recovery and ESS each have curve and heatmap
exports, with means and sample SD over three replicates.

Run `jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/run_iso_policy_sidecars.py`
with `.venv/bin/python`, `--phase all --jobs 8`. `--phase prepare`, `fit`, and
`analyze` support preparation, resumable fitting, and replay independently;
`--smoke` uses a separate output directory and short native fits. PyRosetta
scoring runs in the installed Python 3.10 environment. The default outputs are
under `artifacts/iso_policy_sidecars/`, including `report.html`, PNG/SVG/PDF
figures, candidate and selected tables, and a replay audit. Both selectors use
the identical existing saved-state pool. Individual replicate traces and the
available mean follow the existing sidecar plotting convention; replicate
counts are recorded in the tables.

Invalidated 2026-10-05: the accelerated campaign passed loss replay but failed
full-length fitting equivalence. For ISO TRI spatial split 0, MaxEnt S=100000,
the original scalar path records a convergence checkpoint at step 879; the
cached single-lane path records step 1456; the cached strength batch records
none. The uncached single-lane control matches the original final weights and
MSE exactly. A native pairwise OMC RMSD h=0.01, S=100 control still records no
checkpoint, so acceleration does not explain every missing candidate.

The accelerated figures and tables are withdrawn and archived. The campaign
now uses the original optimizer, native frame-uptake forward pass, and direct
pairwise OMC evaluation; accelerated histories cannot be reused. The new
convergence-only candidate filter and three-replicate plotting gate have also
been removed to restore the existing selection and plotting conventions. The
replacement run must finish and be audited before new figures are accepted.
See the [campaign report](../artifacts/iso_policy_sidecars/report.html) for status.

Replacement completed 2026-10-07: all 882 fits were audited using the original
fitting execution and existing saved-state candidate pool. The two validation
selectors produce 1,764 selections from 48,480 saved candidates, with no missing
selections or incomplete replicate groups. Maximum absolute native-MSE replay
discrepancy is `1.86e-8`. All six figures have PNG, SVG, and PDF exports; the
recovery sweep and construction heatmaps were visually checked. The focused
test suite passed 35 tests after the conventions were restored.

## Conventions

- Model selection is performed independently for each split replicate by minimum
  validation MSE.
- Test MSE is reported only and is never used for selection.
- Recovery is reported as the mean and sample standard deviation across the three
  selected split replicates.
- MaxEnt uses `KL weight = 1 / maxent_scaling`.
- `uptake` means the cluster/group uptake-mixture implementation, not
  per-frame `frame_uptake`.
- Each comparison cell below is `recovery % +/- SD | mean validation MSE | mean
  test MSE`.

As a soft check, investigate changes of roughly 5 recovery percentage points or
10% relative MSE unless the change is expected. Large changes in which averaging
mode wins validation selection should also be reviewed.

## Post-hoc averaging comparison without refitting

For this comparison, the saved frame weights and BV parameters were reused.
Uptake was re-predicted under each averaging semantic, after which candidates
were reselected by validation MSE. The alternative averaging columns are
counterfactual rescoring results, not refitted models.

### IsoValidation

The source optimization used `uptake` averaging.

| Ensemble / split | logPF | Rate | Uptake |
|---|---:|---:|---:|
| ISO_BI sequence | 90.07 +/- 6.07 \| .08560 \| .08785 | 82.54 +/- 7.57 \| .06621 \| .06584 | 78.03 +/- 3.77 \| .06685 \| .06612 |
| ISO_BI spatial | 84.76 +/- 1.39 \| .08726 \| .08760 | 79.81 +/- 5.05 \| .06674 \| .06663 | 75.88 +/- 4.77 \| .06711 \| .06700 |
| ISO_TRI sequence | 54.59 +/- 17.18 \| .08650 \| .08868 | 39.14 +/- 14.91 \| .06530 \| .06364 | 29.93 +/- 1.61 \| .06525 \| .06295 |
| ISO_TRI spatial | 46.33 +/- 13.92 \| .08799 \| .08884 | 31.43 +/- 0.66 \| .06395 \| .06351 | 31.43 +/- 0.66 \| .06369 \| .06317 |

Rate and uptake predictions fit held-out uptake substantially better than logPF
predictions in this run, although logPF-selected candidates have higher recovery.

### CrossValidation

The source optimization used `log_pf` averaging.

| Ensemble / split | logPF | Rate | Uptake |
|---|---:|---:|---:|
| AF2_MSAss sequence | 37.36 +/- 1.80 \| .04997 \| .03494 | 48.47 +/- 16.99 \| .03010 \| .03832 | 37.63 +/- 1.40 \| .03274 \| .02622 |
| AF2_MSAss spatial | 35.05 +/- 4.26 \| .03067 \| .02307 | 58.74 +/- 18.25 \| .02850 \| .04088 | 32.85 +/- 6.70 \| .02167 \| .02152 |
| AF2_filtered sequence | 26.88 +/- 10.29 \| .10048 \| .07745 | 33.29 +/- 15.69 \| .09581 \| .07579 | 30.28 +/- 13.16 \| .09679 \| .07603 |
| AF2_filtered spatial | 17.29 +/- 7.45 \| .08556 \| .07261 | 25.67 +/- 22.58 \| .08414 \| .07360 | 25.07 +/- 14.39 \| .08373 \| .07354 |

### CrossValidationBV

The source optimization used `log_pf` averaging. Saved fitted BV parameters were
reused without modification.

| Ensemble / split | logPF | Rate | Uptake |
|---|---:|---:|---:|
| AF2_MSAss sequence | 43.08 +/- 33.89 \| .03897 \| .03854 | 53.96 +/- 17.09 \| .03011 \| .03816 | 47.56 +/- 21.75 \| .02942 \| .02903 |
| AF2_MSAss spatial | 54.65 +/- 24.85 \| .02480 \| .03405 | 54.65 +/- 24.85 \| .02923 \| .04619 | 54.65 +/- 24.85 \| .02207 \| .03342 |
| AF2_filtered sequence | 80.97 +/- 0.68 \| .04336 \| .03750 | 80.97 +/- 0.68 \| .04152 \| .03516 | 80.97 +/- 0.68 \| .04183 \| .03598 |
| AF2_filtered spatial | 68.75 +/- 12.82 \| .04780 \| .03210 | 80.32 +/- 0.43 \| .04643 \| .03813 | 50.73 +/- 32.47 \| .04779 \| .03014 |

### Joint selection over averaging mode

Allowing `log_pf`, `rate`, and `uptake` to compete independently in each split
replicate produced the following validation-selected modes.

| Experiment | Ensemble / split | Selected modes by replicate | Recovery |
|---|---|---|---:|
| IsoValidation | ISO_BI sequence | rate, rate, rate | 82.54 +/- 7.57% |
| IsoValidation | ISO_BI spatial | rate, rate, rate | 79.81 +/- 5.05% |
| IsoValidation | ISO_TRI sequence | uptake, rate, uptake | 39.14 +/- 14.91% |
| IsoValidation | ISO_TRI spatial | uptake, uptake, uptake | 31.43 +/- 0.66% |
| CrossValidation | AF2_MSAss sequence | rate, logPF, uptake | 37.36 +/- 1.80% |
| CrossValidation | AF2_MSAss spatial | uptake, uptake, rate | 32.85 +/- 6.70% |
| CrossValidation | AF2_filtered sequence | rate, uptake, rate | 30.28 +/- 13.16% |
| CrossValidation | AF2_filtered spatial | rate, logPF, rate | 29.17 +/- 20.80% |
| CrossValidationBV | AF2_MSAss sequence | rate, uptake, uptake | 47.56 +/- 21.75% |
| CrossValidationBV | AF2_MSAss spatial | uptake, uptake, uptake | 54.65 +/- 24.85% |
| CrossValidationBV | AF2_filtered sequence | rate, rate, rate | 80.97 +/- 0.68% |
| CrossValidationBV | AF2_filtered spatial | rate, rate, rate | 80.32 +/- 0.43% |

## Full rate-averaged refit

All three campaigns were subsequently rerun with `rate` averaging for both
optimization and post-processing/model selection. Selection remained minimum
validation MSE.

| Experiment | Ensemble | Split | Recovery | Val MSE | Test MSE |
|---|---|---|---:|---:|---:|
| IsoValidation | ISO_TRI | sequence | 26.45 +/- 19.87% | 0.03095 | 0.02593 |
| IsoValidation | ISO_TRI | spatial | 44.33 +/- 0.32% | 0.03343 | 0.02582 |
| IsoValidation | ISO_BI | sequence | 54.57 +/- 2.27% | 0.03603 | 0.02942 |
| IsoValidation | ISO_BI | spatial | 54.29 +/- 2.50% | 0.03275 | 0.02624 |
| CrossValidation | AF2_filtered | sequence | 47.67 +/- 4.69% | 0.10050 | 0.07722 |
| CrossValidation | AF2_filtered | spatial | 27.22 +/- 12.06% | 0.08980 | 0.07247 |
| CrossValidation | AF2_MSAss | sequence | 71.70 +/- 2.60% | 0.04223 | 0.03541 |
| CrossValidation | AF2_MSAss | spatial | 65.56 +/- 2.71% | 0.03970 | 0.02985 |
| CrossValidationBV | AF2_filtered | sequence | 75.69 +/- 5.75% | 0.04189 | 0.03569 |
| CrossValidationBV | AF2_filtered | spatial | 42.06 +/- 22.21% | 0.05114 | 0.03657 |
| CrossValidationBV | AF2_MSAss | sequence | 70.16 +/- 1.14% | 0.03217 | 0.03023 |
| CrossValidationBV | AF2_MSAss | spatial | 69.70 +/- 2.12% | 0.03702 | 0.02878 |

### Rate-refit run artifacts

- IsoValidation:
  `jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_optimise_test_FIGURE_SIGMA_5000_rate_20260924_222417`
- CrossValidation:
  `jaxent/examples/2_CrossValidation/fitting/jaxENT/_optimise_quick_test_SIGMA_5000__20260924_223118`
- CrossValidationBV:
  `jaxent/examples/3_CrossValidationBV/fitting/jaxENT/_optimise_quick_test_SIGMA_5000_lr1.0_BV_objectve_scale1.0__20260924_223526`

The corresponding processed directories use the same basename with the
`_processed_` prefix. Selected archives are under `val_mse_min`.

The IsoValidation rows above supersede the originally reported values. The raw
fit was not rerun: its uptake was re-predicted and reselected after correcting
post-processing to use the fitted exposure times
`[0.167, 1, 10, 60, 120]` minutes rather than CSV column positions
`[0, 1, 2, 3, 4]`. The corrected processed directory is
`_processed_rate_timefix_20260924_222417`.

### Runtime and integrity

| Campaign | Wall time | Result files | Selected archives |
|---|---:|---:|---:|
| IsoValidation | 353.97 s | 84 | 4 |
| CrossValidation | 222.29 s | 84 | 4 |
| CrossValidationBV | 412.37 s | 168 | 4 |
| Total | 988.63 s | 336 | 12 |

- No error signatures were found in the successful run logs.
- Ten focused tests passed after the averaging-mode changes.
- An initial CrossValidation launch at timestamp `20260924_223032` failed before
  producing any result files because of a missing `m_key` import. The import was
  fixed before the successful `20260924_223118` run; the failed directory is not
  part of this baseline.

## Linear-BV uptake refit versus standard rate-averaged BV

Recorded 2026-09-25. These are scalar CPU refits. Both columns were selected
independently per split replicate by minimum validation MSE, and each cell is
`recovery % +/- sample SD | mean validation MSE | mean test MSE`.

**Invalidated control:** the linear-BV columns below must not be used as a
scientific comparison. Although IsoValidation and CrossValidation set
`optimize_bv_params=False`, the optimizer's model transform still applies
`keep_params_nonnegative()` after the gradient mask. Consequently the negative
initial `raw_bv_bc` was projected from about `-0.8697` to `0` on the first step,
changing physical `bv_bc` from `0.35` to about `0.6931` even with a zero model
gradient. Post-processing then reconstructed the original `0.35` default rather
than the parameter used for most fitting steps. CrossValidationBV additionally
requested model-parameter optimization and suffered the same inappropriate
projection for its unconstrained raw slopes and signed interval offsets. The
table is retained only as a record of the invalidated run.

`Standard rate BV` is the preceding rate-averaged fit. `Linear BV` is the
additive interval-hazard implementation: contacts are averaged once, positive
hazards are accumulated over exposure intervals, and uptake is bounded with the
survival transform. In IsoValidation and CrossValidation its default BV slopes
and zero interval offsets were intended to be held fixed; CrossValidationBV was
configured to fit the slopes and interval offsets with the existing L1 BV
regularization. The optimizer projection bug described above invalidates both
cases.

| Experiment | Ensemble / split | Standard rate BV | Linear BV | Recovery delta |
|---|---|---:|---:|---:|
| IsoValidation | ISO_BI sequence | 54.57 +/- 2.27 \| .03603 \| .02942 | 76.10 +/- 19.10 \| .25877 \| .26957 | +21.53 pp |
| IsoValidation | ISO_BI spatial | 54.29 +/- 2.50 \| .03275 \| .02624 | 80.01 +/- 11.30 \| .25962 \| .27019 | +25.72 pp |
| IsoValidation | ISO_TRI sequence | 26.45 +/- 19.87 \| .03095 \| .02593 | 33.41 +/- 3.61 \| .26003 \| .27123 | +6.96 pp |
| IsoValidation | ISO_TRI spatial | 44.33 +/- 0.32 \| .03343 \| .02582 | 33.06 +/- 0.89 \| .26565 \| .26989 | -11.27 pp |
| CrossValidation | AF2_MSAss sequence | 71.70 +/- 2.60 \| .04223 \| .03541 | 80.84 +/- 9.60 \| .02142 \| .02473 | +9.14 pp |
| CrossValidation | AF2_MSAss spatial | 65.56 +/- 2.71 \| .03970 \| .02985 | 79.94 +/- 15.52 \| .03213 \| .02334 | +14.38 pp |
| CrossValidation | AF2_filtered sequence | 47.65 +/- 4.70 \| .10050 \| .07722 | 68.14 +/- 32.90 \| .02010 \| .03103 | +20.49 pp |
| CrossValidation | AF2_filtered spatial | 27.12 +/- 12.21 \| .08980 \| .07247 | 81.73 +/- 9.90 \| .03195 \| .02429 | +54.61 pp |
| CrossValidationBV | AF2_MSAss sequence | 70.16 +/- 1.14 \| .03217 \| .03023 | 38.29 +/- 7.04 \| .04247 \| .03154 | -31.87 pp |
| CrossValidationBV | AF2_MSAss spatial | 69.70 +/- 2.12 \| .03702 \| .02878 | 43.32 +/- 0.85 \| .02540 \| .01791 | -26.38 pp |
| CrossValidationBV | AF2_filtered sequence | 75.63 +/- 5.71 \| .04189 \| .03569 | 37.88 +/- 21.09 \| .09575 \| .07496 | -37.75 pp |
| CrossValidationBV | AF2_filtered spatial | 41.85 +/- 22.06 \| .05114 \| .03657 | 44.42 +/- 36.04 \| .09955 \| .07747 | +2.57 pp |

No performance conclusion should be drawn from this table. In addition to the
optimizer/post-processing mismatch, this control is not a pure
algebraic-linearity comparison because the linear model applies the declared
`s^-1` intrinsic-rate to minute-time conversion, while the legacy standard BV
forward pass consumes the configured timepoints directly.

### Linear-BV run artifacts

- IsoValidation:
  `jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_optimise_test_FIGURE_SIGMA_5000_linear_bv_rate_20260925_023804`
- CrossValidation:
  `jaxent/examples/2_CrossValidation/fitting/jaxENT/_optimise_quick_test_SIGMA_5000_linear_bv__20260925_024232`
- CrossValidationBV:
  `jaxent/examples/3_CrossValidationBV/fitting/jaxENT/_optimise_quick_test_SIGMA_5000_linear_bv_BV__20260925_024539`

The campaigns produced 84, 84, and 168 result archives respectively, plus four
validation-selected archives per campaign. No error signatures were found in
their successful logs. The linear-BV and standard-BV unit tests passed (32
tests), as did fit/post-processing smoke tests for fixed and fitted linear-BV
parameters.

## Corrected linear-BV rerun

Recorded 2026-09-26 after replacing the blanket nonnegative optimizer transform
with parameter-class-aware projection and adding a dynamic-slot gradient mask.
The scalar CPU reruns used this parameter policy:

- IsoValidation and CrossValidation: `bc=0.35`, `bh=2.0`, and every interval
  offset fixed.
- CrossValidationBV: fit only the unconstrained `raw_bv_bc` and `raw_bv_bh`;
  every interval offset fixed at zero. The existing L1 BV-parameter penalty was
  retained.
- Frame weights were fitted in every campaign. Selection was minimum validation
  MSE independently per split replicate. Test MSE was report-only.

Each model cell is `recovery % +/- sample SD | mean validation MSE | mean test
MSE`. The standard-rate baseline is unchanged from the corrected table above.

| Experiment | Ensemble / split | Standard rate BV | Corrected linear BV | Recovery delta |
|---|---|---:|---:|---:|
| IsoValidation | ISO_BI sequence | 54.57 +/- 2.27 \| .03603 \| .02942 | 52.10 +/- 0.39 \| .25426 \| .24892 | -2.47 pp |
| IsoValidation | ISO_BI spatial | 54.29 +/- 2.50 \| .03275 \| .02624 | 52.28 +/- 0.15 \| .25932 \| .24683 | -2.01 pp |
| IsoValidation | ISO_TRI sequence | 26.45 +/- 19.87 \| .03095 \| .02593 | 15.35 +/- 20.77 \| .25175 \| .24692 | -11.10 pp |
| IsoValidation | ISO_TRI spatial | 44.33 +/- 0.32 \| .03343 \| .02582 | 3.12 +/- 0.04 \| .24255 \| .23555 | -41.21 pp |
| CrossValidation | AF2_MSAss sequence | 71.70 +/- 2.60 \| .04223 \| .03541 | 60.21 +/- 5.47 \| .02019 \| .01959 | -11.49 pp |
| CrossValidation | AF2_MSAss spatial | 65.56 +/- 2.71 \| .03970 \| .02985 | 71.75 +/- 6.24 \| .03030 \| .02080 | +6.20 pp |
| CrossValidation | AF2_filtered sequence | 47.65 +/- 4.70 \| .10050 \| .07722 | 73.28 +/- 11.17 \| .02138 \| .02207 | +25.63 pp |
| CrossValidation | AF2_filtered spatial | 27.12 +/- 12.21 \| .08980 \| .07247 | 85.74 +/- 15.58 \| .03045 \| .02069 | +58.62 pp |
| CrossValidationBV | AF2_MSAss sequence | 70.16 +/- 1.14 \| .03217 \| .03023 | 76.97 +/- 0.47 \| .02017 \| .02190 | +6.81 pp |
| CrossValidationBV | AF2_MSAss spatial | 69.70 +/- 2.12 \| .03702 \| .02878 | 75.99 +/- 3.30 \| .02701 \| .02538 | +6.28 pp |
| CrossValidationBV | AF2_filtered sequence | 75.63 +/- 5.71 \| .04189 \| .03569 | 85.21 +/- 6.25 \| .02085 \| .02186 | +9.58 pp |
| CrossValidationBV | AF2_filtered spatial | 41.85 +/- 22.06 \| .05114 \| .03657 | 81.73 +/- 11.24 \| .02691 \| .02458 | +39.88 pp |

The corrected result reverses the invalidated CrossValidationBV conclusion:
fitting only the two contact scalings improves mean recovery in all four cells
and improves validation MSE. Fixed linear BV remains badly model-mismatched for
IsoValidation, despite fitting real CrossValidation uptake substantially better
by MSE.

### Parameter and artifact audit

- Every parameter snapshot in all 84 IsoValidation and 84 CrossValidation
  archives retained `raw_bv_bc=-0.8697232` (`bc=0.35`),
  `raw_bv_bh=1.8545866` (`bh=2.0`), and zero interval offsets.
- Across all 168 CrossValidationBV archives, interval offsets remained exactly
  zero. The raw contact parameters changed; among the 12 validation-selected
  replicates, physical `bc` ranged from `0.3231` to `0.4097` and physical `bh`
  from `1.6417` to `2.2412`.
- Result counts were 84, 84, and 168, with four selected archives per campaign
  and no error signatures in the successful logs.
- End-to-end wall times were approximately 240 s, 189 s, and 352 s
  respectively (about 13 minutes total).
- Thirty-six focused model/projection tests passed, plus three 10-step history
  smoke campaigns that directly checked the saved parameter values.

Corrected raw artifacts:

- IsoValidation:
  `jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_optimise_test_FIGURE_SIGMA_5000_linear_bv_fixed_rate_20260926_021621`
- CrossValidation:
  `jaxent/examples/2_CrossValidation/fitting/jaxENT/_optimise_quick_test_SIGMA_5000_linear_bv_fixed__20260926_022111`
- CrossValidationBV:
  `jaxent/examples/3_CrossValidationBV/fitting/jaxENT/_optimise_quick_test_SIGMA_5000_linear_bv_bc_bh__20260926_022424`

An initial CrossValidation launch at timestamp `20260926_022032` used bare
system Python, failed imports, and produced zero result archives. The launchers
now use the project `uv` environment and propagate failures; that empty
diagnostic directory is not part of this baseline.

## Full-uptake reference refit

Recorded 2026-09-26. `Full uptake` means the standard BV model with
`frame_uptake` averaging: the complete nonlinear uptake curve is evaluated for
each frame first, then those uptake curves are averaged using the fitted frame
weights. These are scalar CPU refits, not post-hoc rescoring. Parameter fitting
matches the other standard-BV campaigns: Examples 1--2 fit frame weights only;
CrossValidationBV fits frame weights plus `bc/bh` under its L1 sweep. Selection
is minimum validation MSE independently per split replicate.

Each cell is `recovery % +/- sample SD | mean validation MSE | mean test MSE`.

| Experiment | Ensemble / split | Rate | Corrected linear BV | Full uptake |
|---|---|---:|---:|---:|
| IsoValidation | ISO_BI sequence | 54.57 +/- 2.27 \| .03603 \| .02942 | 52.10 +/- 0.39 \| .25426 \| .24892 | 83.38 +/- 2.06 \| .01360 \| .01352 |
| IsoValidation | ISO_BI spatial | 54.29 +/- 2.50 \| .03275 \| .02624 | 52.28 +/- 0.15 \| .25932 \| .24683 | 77.69 +/- 6.61 \| .01474 \| .01315 |
| IsoValidation | ISO_TRI sequence | 26.45 +/- 19.87 \| .03095 \| .02593 | 15.35 +/- 20.77 \| .25175 \| .24692 | 37.31 +/- 1.52 \| .01367 \| .01306 |
| IsoValidation | ISO_TRI spatial | 44.33 +/- 0.32 \| .03343 \| .02582 | 3.12 +/- 0.04 \| .24255 \| .23555 | 36.07 +/- 1.40 \| .01474 \| .01331 |
| CrossValidation | AF2_MSAss sequence | 71.70 +/- 2.60 \| .04223 \| .03541 | 60.21 +/- 5.47 \| .02019 \| .01959 | 18.59 +/- 5.89 \| .04488 \| .03484 |
| CrossValidation | AF2_MSAss spatial | 65.56 +/- 2.71 \| .03970 \| .02985 | 71.75 +/- 6.24 \| .03030 \| .02080 | 27.23 +/- 4.92 \| .02332 \| .02110 |
| CrossValidation | AF2_filtered sequence | 47.65 +/- 4.70 \| .10050 \| .07722 | 73.28 +/- 11.17 \| .02138 \| .02207 | 33.00 +/- 17.68 \| .10032 \| .07991 |
| CrossValidation | AF2_filtered spatial | 27.12 +/- 12.21 \| .08980 \| .07247 | 85.74 +/- 15.58 \| .03045 \| .02069 | 22.45 +/- 10.66 \| .08700 \| .07245 |
| CrossValidationBV | AF2_MSAss sequence | 70.16 +/- 1.14 \| .03217 \| .03023 | 76.97 +/- 0.47 \| .02017 \| .02190 | 52.00 +/- 18.96 \| .04058 \| .03670 |
| CrossValidationBV | AF2_MSAss spatial | 69.70 +/- 2.12 \| .03702 \| .02878 | 75.99 +/- 3.30 \| .02701 \| .02538 | 31.49 +/- 5.84 \| .02137 \| .02136 |
| CrossValidationBV | AF2_filtered sequence | 75.63 +/- 5.71 \| .04189 \| .03569 | 85.21 +/- 6.25 \| .02085 \| .02186 | 47.62 +/- 33.07 \| .03663 \| .03557 |
| CrossValidationBV | AF2_filtered spatial | 41.85 +/- 22.06 \| .05114 \| .03657 | 81.73 +/- 11.24 \| .02691 \| .02458 | 63.68 +/- 24.75 \| .04212 \| .03298 |

For the same validation-selected models, the table below adds effective sample
size. ESS is `1 / sum(w_i^2)` after normalizing each selected frame-weight
vector. Each cell is `recovery % +/- sample SD | ESS +/- sample SD (relative
ESS % +/- sample SD)`. Relative ESS divides by the ensemble size (874 frames
for ISO_BI, 2,225 for ISO_TRI, and 500 for both AF2 ensembles).

| Experiment | Ensemble / split | Rate | Corrected linear BV | Full uptake |
|---|---|---:|---:|---:|
| IsoValidation | ISO_BI sequence | 54.57 +/- 2.27 \| 2.59 +/- 0.36 (0.30% +/- 0.04%) | 52.10 +/- 0.39 \| 50.71 +/- 85.89 (5.80% +/- 9.83%) | 83.38 +/- 2.06 \| 336.84 +/- 29.81 (38.54% +/- 3.41%) |
| IsoValidation | ISO_BI spatial | 54.29 +/- 2.50 \| 2.42 +/- 0.52 (0.28% +/- 0.06%) | 52.28 +/- 0.15 \| 1.14 +/- 0.03 (0.13% +/- 0.00%) | 77.69 +/- 6.61 \| 149.28 +/- 195.70 (17.08% +/- 22.39%) |
| IsoValidation | ISO_TRI sequence | 26.45 +/- 19.87 \| 1.91 +/- 0.09 (0.09% +/- 0.00%) | 15.35 +/- 20.77 \| 129.23 +/- 221.28 (5.81% +/- 9.95%) | 37.31 +/- 1.52 \| 454.94 +/- 461.26 (20.45% +/- 20.73%) |
| IsoValidation | ISO_TRI spatial | 44.33 +/- 0.32 \| 3.40 +/- 1.11 (0.15% +/- 0.05%) | 3.12 +/- 0.04 \| 1.10 +/- 0.00 (0.05% +/- 0.00%) | 36.07 +/- 1.40 \| 385.98 +/- 42.48 (17.35% +/- 1.91%) |
| CrossValidation | AF2_MSAss sequence | 71.70 +/- 2.60 \| 391.81 +/- 132.83 (78.36% +/- 26.57%) | 60.21 +/- 5.47 \| 14.21 +/- 11.83 (2.84% +/- 2.37%) | 18.59 +/- 5.89 \| 41.42 +/- 45.03 (8.28% +/- 9.01%) |
| CrossValidation | AF2_MSAss spatial | 65.56 +/- 2.71 \| 107.97 +/- 65.27 (21.59% +/- 13.05%) | 71.75 +/- 6.24 \| 74.27 +/- 57.87 (14.85% +/- 11.57%) | 27.23 +/- 4.92 \| 5.88 +/- 3.82 (1.18% +/- 0.76%) |
| CrossValidation | AF2_filtered sequence | 47.65 +/- 4.70 \| 70.66 +/- 24.10 (14.13% +/- 4.82%) | 73.28 +/- 11.17 \| 243.19 +/- 238.12 (48.64% +/- 47.62%) | 33.00 +/- 17.68 \| 17.71 +/- 14.12 (3.54% +/- 2.82%) |
| CrossValidation | AF2_filtered spatial | 27.12 +/- 12.21 \| 3.32 +/- 1.40 (0.66% +/- 0.28%) | 85.74 +/- 15.58 \| 52.07 +/- 53.84 (10.41% +/- 10.77%) | 22.45 +/- 10.66 \| 3.97 +/- 3.32 (0.79% +/- 0.66%) |
| CrossValidationBV | AF2_MSAss sequence | 70.16 +/- 1.14 \| 479.68 +/- 13.54 (95.94% +/- 2.71%) | 76.97 +/- 0.47 \| 489.26 +/- 1.70 (97.85% +/- 0.34%) | 52.00 +/- 18.96 \| 188.09 +/- 270.56 (37.62% +/- 54.11%) |
| CrossValidationBV | AF2_MSAss spatial | 69.70 +/- 2.12 \| 416.92 +/- 104.07 (83.38% +/- 20.81%) | 75.99 +/- 3.30 \| 468.60 +/- 30.78 (93.72% +/- 6.16%) | 31.49 +/- 5.84 \| 7.21 +/- 2.99 (1.44% +/- 0.60%) |
| CrossValidationBV | AF2_filtered sequence | 75.63 +/- 5.71 \| 364.70 +/- 216.15 (72.94% +/- 43.23%) | 85.21 +/- 6.25 \| 331.87 +/- 278.89 (66.37% +/- 55.78%) | 47.62 +/- 33.07 \| 175.78 +/- 279.49 (35.16% +/- 55.90%) |
| CrossValidationBV | AF2_filtered spatial | 41.85 +/- 22.06 \| 8.23 +/- 6.59 (1.65% +/- 1.32%) | 81.73 +/- 11.24 \| 173.79 +/- 282.55 (34.76% +/- 56.51%) | 63.68 +/- 24.75 \| 298.08 +/- 255.86 (59.62% +/- 51.17%) |

Full uptake is the strongest IsoValidation reference except for ISO_TRI spatial,
where rate has higher recovery despite worse held-out MSE. Corrected linear BV
is strongest in all four CrossValidationBV recovery cells. The CrossValidation
and CrossValidationBV MSAss-spatial rows are especially clear examples that the
lowest validation MSE need not imply the best conformational recovery: full
uptake wins validation MSE there but has much lower recovery.

### Full-uptake artifacts and runtime

- IsoValidation (about 1,315 s):
  `jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_optimise_test_FIGURE_SIGMA_5000_full_uptake_frame_uptake_20260926_133450`
- CrossValidation (about 407 s):
  `jaxent/examples/2_CrossValidation/fitting/jaxENT/_optimise_quick_test_SIGMA_5000_full_uptake__20260926_135652`
- CrossValidationBV (about 803 s):
  `jaxent/examples/3_CrossValidationBV/fitting/jaxENT/_optimise_quick_test_SIGMA_5000_full_uptake_BV__20260926_140412`

Total end-to-end time was about 2,525 s (42 minutes). The campaigns produced
84, 84, and 168 result archives respectively, plus four validation-selected
archives per campaign. No error signatures were found in the successful logs.

## CUDA scalar-versus-batch benchmark

Recorded 2026-09-25 on one NVIDIA RTX 3090 (24 GB), driver 580.173.02, using a
temporary Python 3.11 environment with JAX 0.4.35, jaxlib 0.4.34, and the CUDA
12 plugin. Each row is one `sequence_cluster` split-000 sweep for the listed
ensemble, with rate averaging, 5,000 steps, learning rate 1.0, forward-model
scaling 1000, and MaxEnt scales `1,5,10,50,100,500,1000`. Scalar candidates ran
serially in one process. Batch width equals the candidate count; the BV sweep
also crosses BV penalties `0.5,1.0` with L1 regularization.

| Experiment | Representative ensemble | Candidates | Scalar | Batch | Speedup |
|---|---|---:|---:|---:|---:|
| IsoValidation | ISO_BI | 7 | 204.49 s | 124.12 s | 1.65x |
| CrossValidation | AF2_MSAss | 7 | 138.27 s | 79.47 s | 1.74x |
| CrossValidationBV | AF2_MSAss | 14 | 288.79 s | 110.59 s | 2.61x |
| Total | - | 28 | 631.55 s | 314.18 s | 2.01x |

The identical representative sweeps were then rerun with the CPU backend in
the same temporary environment. `CPU/CUDA` greater than 1 means CUDA was
faster; less than 1 means CPU was faster.

| Experiment | Version | CPU | CUDA | CPU/CUDA |
|---|---|---:|---:|---:|
| IsoValidation | scalar | 200.87 s | 204.49 s | 0.982x |
| IsoValidation | batch-7 | 124.73 s | 124.12 s | 1.005x |
| CrossValidation | scalar | 141.55 s | 138.27 s | 1.024x |
| CrossValidation | batch-7 | 81.62 s | 79.47 s | 1.027x |
| CrossValidationBV | scalar | 285.31 s | 288.79 s | 0.988x |
| CrossValidationBV | batch-14 | 95.43 s | 110.59 s | 0.863x |
| Scalar total | - | 627.73 s | 631.55 s | 0.994x |
| Batch total | - | 301.78 s | 314.18 s | 0.961x |
| Overall | - | 929.51 s | 945.73 s | 0.983x |

For these cold, relatively small graphs, CUDA did not improve total wall time:
the scalar total was 0.6% slower on CUDA, the batch total was 4.1% slower, and
the combined total was 1.7% slower. The largest hardware difference was the BV
batch, where CPU was 15.9% faster. Batching itself remained beneficial on both
backends: 2.08x overall on CPU and 2.01x overall on CUDA. CPU and CUDA produced
bit-identical saved best-state frame weights for every matched candidate and
selected the same minimum-total-validation-loss candidate in all six sweeps.

These are cold-start wall times and include XLA compilation. All 112 expected
HDF5 files were produced across the two implementations and two backends, and
all inspected weights and losses were finite. Matched scalar/batch CUDA
best-state comparisons were:

| Experiment | Mean frame-weight RMSE | Maximum frame-weight absolute difference | Same minimum total-validation-loss candidate |
|---|---:|---:|---|
| IsoValidation | 1.073e-08 | 6.384e-07 | yes (`maxent=1000`) |
| CrossValidation | 1.877e-09 | 1.425e-07 | yes (`maxent=50`) |
| CrossValidationBV | 3.680e-06 | 2.371e-04 | yes (`maxent=1`, `bvreg=0.5`) |

For CrossValidationBV, maximum absolute scalar/batch differences in the fitted
BV parameters were 1.126e-04 for `bv_bc` and 8.893e-04 for `bv_bh`.
