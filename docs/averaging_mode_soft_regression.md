# Averaging-mode soft-regression baseline

Recorded on 2026-09-24 for the Example 1--3 test campaigns. This document is a
soft-regression reference for frame-averaging semantics, validation-based model
selection, and conformational recovery. It is not an exact numerical test:
optimizer, JAX/XLA, hardware, and dependency changes can cause small differences.

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
| IsoValidation | ISO_TRI | sequence | 30.85 +/- 0.11% | 0.06557 | 0.06326 |
| IsoValidation | ISO_TRI | spatial | 30.75 +/- 0.16% | 0.06384 | 0.06319 |
| IsoValidation | ISO_BI | sequence | 62.67 +/- 0.01% | 0.06755 | 0.06663 |
| IsoValidation | ISO_BI | spatial | 63.23 +/- 0.39% | 0.06781 | 0.06660 |
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

