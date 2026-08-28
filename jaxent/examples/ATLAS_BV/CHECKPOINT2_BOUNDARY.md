# Checkpoint 2 — common PF support and boundary extrapolation

Completed on 2026-08-27 for all 111 systems and all 333 replica holdouts. The command is:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-boundary-audit --workers 4
```

This checkpoint separates held-out pairs whose PF distance lies inside the training PF-distance
range from extrapolation pairs. It compares the original isotonic prediction
(`out_of_bounds="clip"`) with a continuous, non-negative endpoint-linear extension. The extension
is anchored at each isotonic boundary and estimates its slope by non-negative least squares over
the outer 10% of training PF distances. This changes only out-of-support predictions; it does not
silently treat them as common-support observations.

For frame-centred PF L2 in the global RMSD band (`>2.5 Å`), only 110,688 of 5,686,369 pairs
(`1.95%`) are outside PF support. Eighty-six of 247 contributing system-folds contain any such
pairs, and the median non-zero fold fraction is only `0.178%`. Restricting evaluation to common PF
support moves median 90% coverage from `76.81%` to `77.04%`. Endpoint extrapolation leaves median
coverage unchanged at `76.81%`; its paired mean fold change is `+0.23` percentage points. Median
MAE changes from `0.7609 Å` to `0.7576 Å`, while the paired mean fold improvement is only
`0.0032 Å`. Therefore isotonic clipping is not the principal cause of global-RMSD undercoverage.

The coordinate-distribution W1 upper tail has more PF boundary mismatch: 93,098 of 1,108,167 q5
pairs (`8.40%`) lie outside PF support. Nevertheless, endpoint extrapolation changes median q5
coverage by zero and increases median MAE from `0.2860 Å` to `0.2878 Å` (paired mean change
`+0.00057 Å`). Common-support q5 coverage remains only `20.39%`, essentially the clipped value of
`20.43%`. The severe W1 tail failure therefore persists even after boundary cases are removed.

These intervals still use the historical same-fit residual-decile construction. They are retained
only to isolate boundary behaviour; three-way train/calibration/test conformal intervals belong to
Checkpoint 3. The small extrapolation-only subsets are also reported explicitly, but their
per-fold medians can be unstable and should not be confused with the all-pair estimates.

Checkpoint conclusion: boundary clipping contributes very little to the RMSD tail and does not
repair the W1 tail. Proceeding to three-way conformal calibration is justified, subject to review.
Per-residue vector models remain paused until after that calibration checkpoint.

Plots:

- `outputs/analysis/pairwise_geometry/checkpoint2_boundary/boundary_coverage_comparison.png`
- `outputs/analysis/pairwise_geometry/checkpoint2_boundary/tail_distribution_fit_methods.png`
- `outputs/analysis/pairwise_geometry/checkpoint2_boundary/w1_fit_error_across_bands.png`
- `outputs/analysis/pairwise_geometry/checkpoint2_boundary/distribution_fit_metrics_across_w1_bands.png`
- `outputs/analysis/pairwise_geometry/checkpoint2_boundary/pf_distance_function_comparison.png`
- `outputs/analysis/pairwise_geometry/checkpoint2_boundary/pf_distance_function_recovery.png`

The last plot reports dimensionless probability-distribution fit error. Within each held-out
structural-W1 band, predicted and target values are converted to probability masses on 20 equal
bins spanning the training-defined band. Values outside that support are retained in the boundary
bins. The primary plotted error is `sqrt(JSD)` and the result table also records probability-mass
L1, L2, JSD, target-to-prediction KLD, and recovery `1 - sqrt(JSD)`. These are distinct from paired
absolute prediction error in angstroms.

For frame-centred PF L2 with the clipped mapping, median `sqrt(JSD)` across q0–q5 is `0.719`,
`0.765`, `0.601`, `0.451`, `0.679`, and `0.594`. Corresponding recovery
`1 - sqrt(JSD)` is `28.2%`, `23.6%`, `39.9%`, `54.9%`, `32.1%`, and `40.6%`. Common-support
filtering and endpoint extrapolation change these values negligibly. The predicted probability
distribution therefore fits poorly throughout the range, with q3 the least-bad band; the earlier
angstrom-valued paired-error curve must not be interpreted as distribution fit.

The distance-function extension compares raw Absolute-L1, raw L2, frame-centred L2, raw cosine,
and raw correlation distance on identical held-out frame pairs. Absolute-L1 has the lowest median
`sqrt(JSD)` in every W1 band except q0, where raw L2 is lower by only `0.0003`. Its q0–q5 values
are `0.717`, `0.738`, `0.481`, `0.438`, `0.594`, and `0.550`. Across the 1,965 complete
system-fold-band comparisons, Absolute-L1 is the lowest-error arm in `58.0%`, followed by raw L2
in `24.4%`, cosine in `7.4%`, correlation in `5.5%`, and frame-centred L2 in `4.7%`. Cosine and
correlation therefore do not repair distribution fit; retaining absolute magnitude with L1 or raw
L2 is materially better, especially in q2 and the upper W1 bands.
