# Checkpoint 7 — exact directional PF support

Completed on 2026-08-28 for 111 systems and 333 replica holdouts. This final support diagnostic
uses exact nearest-training-vector distance in the complete, training-only z-scored per-residue PF
change space. Distances are normalized by the square root of residue count. The experiment uses the
same deterministic 5,000 training and 10,000 held-out pair caps as Checkpoint 3C.

Every fold contains two likelihood arms with an identical replica-tuned ridge mean and identical
pair samples:

- a constant 90th-percentile residual scale;
- a scale constrained to stay flat or increase with exact nearest-training PF distance.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-nearest-calibration --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-nearest-compare
```

## Result

Directional support distance transfers a real but insufficient signal:

| Endpoint | Arm | Features | Recovery | 90% coverage | Median NLL |
|---|---|---|---:|---:|---:|
| Global RMSD | constant | raw | 75.43% | 84.01% | 1.743 |
| Global RMSD | nearest | raw | 76.88% | 82.98% | 1.702 |
| W1 q5 | constant | raw | 81.02% | 41.61% | 0.687 |
| W1 q5 | nearest | raw | 83.03% | 47.62% | 0.583 |
| W1 q5 | constant | z-scored | 81.31% | 38.49% | 0.663 |
| W1 q5 | nearest | z-scored | 83.36% | 45.75% | 0.637 |

After folds are averaged within systems, raw nearest-distance calibration improves W1-q5 recovery
over constant scale by a paired median `1.36` percentage points (95% system-bootstrap interval
`0.60–1.78`), coverage by `3.48` points (`1.96–4.49`), and NLL by `-0.086`
(`-0.160–-0.041`). Z-scored features improve recovery by `1.12` points and coverage by `3.51`
points. Global-RMSD changes are small and uncertain.

The capped experiment has lower absolute W1-tail coverage than the full-pair Checkpoint 4 model
because its ridge mean is trained on only 5,000 pairs. It must therefore be interpreted through the
matched nearest-minus-constant contrast, not by comparing its absolute coverage directly with the
full-pair likelihood.

Exact support distance is much better at pooled probability-mass recovery than the identically
capped kNN conditional mixture: median paired gains are about 12.9 points for global RMSD and
32–33 points for W1 q5. Supervised ridge mean structure remains essential; local neighbours alone
do not reconstruct the target distribution.

## Conclusion

Directional nearest-PF distance detects some transferable support failure, unlike radial norm.
However, it repairs only a few coverage points and leaves catastrophic W1-tail undercoverage.
Support calibration is therefore a secondary correction, not the explanation for the tail.

Across Checkpoints 3–7, residue-aware ridge improves structural-distribution fit, probabilistic
likelihoods improve pooled recovery, and exact local support modestly improves uncertainty. None
produces calibrated long-distance W1 intervals. Under the tested replica-cross-validated models,
the fixed BV protection-factor representation does not contain enough transferable information to
identify the magnitude of large structural departures. This is an empirical limit of the tested
representation—not a proof that no conceivable nonlinear model could extract more signal.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint7_nearest/nearest_recovery_coverage_bands.png`
- `outputs/analysis/pairwise_geometry/checkpoint7_nearest/nearest_results.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint7_nearest/nearest_hyperparameters.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint7_nearest/nearest_paired_vs_constant.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint7_nearest/nearest_vs_knn_conditional.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint7_nearest/checkpoint7a_report.yaml`
- `outputs/analysis/pairwise_geometry/checkpoint7_nearest/checkpoint7b_report.yaml`
