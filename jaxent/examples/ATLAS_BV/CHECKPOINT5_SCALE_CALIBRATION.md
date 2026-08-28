# Checkpoint 5 — nonlinear predicted-mean scale calibration

Completed on 2026-08-28 for 111 systems and 333 replica holdouts. This checkpoint replaces the
collapsed per-residue linear variance model with a piecewise-linear 90th-percentile residual scale
conditioned on the cross-validated predicted structural mean. Candidate bin counts
`{1,3,5,10,20}` are selected across training replicas using the proper central interval score; one
bin is the constant-scale baseline.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-scale-calibration --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-scale-compare
```

## Result

The model improves pooled distribution shape but worsens tail calibration:

| Endpoint | Features | Recovery | 90% coverage | Median NLL |
|---|---|---:|---:|---:|
| Global RMSD | raw | 76.07% | 82.91% | 1.768 |
| Global RMSD | z-scored | 77.17% | 82.34% | 1.755 |
| W1 q5 | raw | 87.50% | 68.14% | 0.182 |
| W1 q5 | z-scored | 87.60% | 66.67% | 0.184 |

Against Checkpoint 4's linear-variance likelihood, raw W1-q5 recovery improves by a paired median
`3.35` percentage points (95% system-bootstrap interval `2.26–4.90`), but coverage falls by `9.48`
points (`-10.47–-7.75`). Z-scored recovery improves by `1.87` points (`1.22–2.66`), while coverage
falls by `5.29` points (`-6.44–-4.14`). Global RMSD changes are small and uncertain.

The failure is informative. A genuinely distant held-out structure is often assigned too small a
predicted mean. A scale function that sees only that mean consequently treats the structure as
ordinary and assigns an interval appropriate to the wrong regime. Better pooled probability mass
does not fix this query-level failure.

Bin selection does not collapse completely: constant scale is selected in 37.5% of fits, while 3,
5, 10 and 20 bins are selected in 15.5%, 12.5%, 14.9% and 19.5%. Nevertheless, the extra scale
flexibility follows the predicted coordinate rather than structural novelty and therefore sharpens
the wrong tail intervals.

## Conclusion

Predicted-mean-only calibration does not repair the long-distance tail and should not replace the
Checkpoint 4 model. The next scale model must receive an out-of-support signal from the full PF
change vector—such as training-standardized feature norm, nearest-training-vector distance or ridge
leverage—in addition to the predicted mean. This directly tests whether tail undercoverage is
detectable as PF-feature novelty; if it is not, the remaining failure is representation
degeneracy rather than calibration.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint5_scale/scale_recovery_coverage_comparison.png`
- `outputs/analysis/pairwise_geometry/checkpoint5_scale/scale_results.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint5_scale/scale_hyperparameters.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint5_scale/scale_paired_vs_linear_variance.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint5_scale/checkpoint5a_report.yaml`
- `outputs/analysis/pairwise_geometry/checkpoint5_scale/checkpoint5b_report.yaml`
