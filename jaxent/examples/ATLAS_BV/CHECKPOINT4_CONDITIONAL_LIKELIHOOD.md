# Checkpoint 4 — per-residue conditional likelihood

Completed on 2026-08-28 for all 111 systems and 333 replica holdouts. The model uses raw or
training-only z-scored absolute per-residue PF-change vectors. Ridge predicts the conditional mean;
a second ridge predicts log residual variance from residuals generated across training replicas.
A replica-cross-fitted conformal factor calibrates scale without using the held-out replica.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-vector-likelihood --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-likelihood-compare
```

Every system is atomically checkpointed under `checkpoint4_likelihood/parts/`; interrupted runs
resume automatically. The population run was assembled only after all system checkpoints passed
the three-fold validation contract.

## Results

Distribution fit is measured as dimensionless probability mass recovery,
`100 * (1 - sqrt(JSD))`:

| Endpoint | Features | Recovery | Absolute L1 | L2 | KLD | Nominal-90% coverage |
|---|---|---:|---:|---:|---:|---:|
| Global RMSD | raw | 75.20% | 0.535 | 0.180 | 0.223 | 81.85% |
| Global RMSD | z-scored | 76.11% | 0.510 | 0.173 | 0.219 | 82.71% |
| W1 q5 | raw | 83.55% | 0.303 | 0.104 | 0.093 | 79.63% |
| W1 q5 | z-scored | 86.17% | 0.257 | 0.100 | 0.065 | 74.40% |

Z-scoring improves W1-q5 recovery over raw features by a median paired `1.93` percentage points
(95% system-bootstrap interval `1.14–2.32`; 78.4% of systems; Holm-adjusted `p=2.55e-8`). The
global-RMSD median improvement is `0.59` points; its bootstrap interval touches zero
(`-0.01–1.23`) despite a positive paired-rank test. Mean-centring/scaling is therefore useful for
W1 distribution shape, but not a calibration repair.

Coverage is close to or above nominal in central regimes and drops sharply with structural scale:
global RMSD reaches only 81.9–82.7%, while W1 q5 reaches 79.6% raw and 74.4% z-scored. The improved
z-scored W1 fit and worsened coverage are not contradictory: pooled probability mass can have the
right shape while individual conditional intervals remain too narrow.

The variance model is not stably residue-specific. `98.27%` of fits choose the largest variance
ridge penalty (`10000`), and the median cross-fitted scale is `0.959`. Thus the current linear
variance arm collapses toward a nearly homoscedastic model. The strong result comes primarily from
the per-residue conditional mean plus probabilistic smoothing, not from learned heteroscedastic
residue weights.

This likelihood recovery must remain separate from point-histogram recovery: it predicts a full
conditional density for each pair rather than collapsing each pair to one RMSD/W1 estimate.

## Conclusion

The conditional-likelihood direction is supported, especially for W1. Training-only z-scoring is
the preferred distribution-fit representation, but raw features are better calibrated in the W1
tail. The next calibration model should make scale a nonlinear function of the cross-validated
predicted structural mean (for example, monotone or binned scale calibration), rather than fitting
another high-dimensional linear residue coefficient vector.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint4_likelihood/likelihood_recovery_and_coverage_bands.png`
- `outputs/analysis/pairwise_geometry/checkpoint4_likelihood/likelihood_results.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint4_likelihood/likelihood_hyperparameters.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint4_likelihood/likelihood_band_summary.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint4_likelihood/likelihood_paired_preprocessing.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint4_likelihood/checkpoint4a_report.yaml`
- `outputs/analysis/pairwise_geometry/checkpoint4_likelihood/checkpoint4b_report.yaml`
