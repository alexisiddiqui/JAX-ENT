# Checkpoint 9 — strict conditional-likelihood baseline

Completed on 2026-08-28 for all 111 systems and all six ordered A-fit/B-calibrate/C-test replica
assignments. This checkpoint establishes the distribution-likelihood baseline for the planned
opening-probability screen.

The test pairs, W1 bands, structural-support labels and effective-frame accounting are exactly those
from Checkpoint 8. Replica A alone fits ridge and its preprocessing statistics, replica B supplies
finite-sample conformal residual quantiles, and untouched replica C supplies the probability-mass
fit and coverage results. Gaussian conditional mass is evaluated on 20 structural-target bins.

## Primary W1-q5 result

Common-support population medians are:

| Features | Calibration | Recovery | L1 | L2 | sqrt(JSD) | KLD | Coverage | Interval score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Raw log-PF | marginal | 83.14% | 0.325 | 0.129 | 0.169 | 0.134 | 59.98% | 0.899 |
| Raw log-PF | Mondrian | 81.29% | 0.381 | 0.137 | 0.187 | 0.170 | 70.12% | 0.875 |
| A-only z-score | marginal | 83.18% | 0.325 | 0.128 | 0.168 | 0.134 | 59.90% | 0.903 |
| A-only z-score | Mondrian | 81.29% | 0.381 | 0.137 | 0.187 | 0.170 | 70.38% | 0.874 |

Cosine and correlation probability-mass distances agree with the other metrics: raw marginal values
are `0.0396` and `0.0471`, respectively. These quantities are dimensionless distribution-fit errors;
they are not structural prediction errors in Angstroms.

Z-scoring is not promoted. Its paired W1-q5 Mondrian recovery effect is only `+0.012` percentage
points (95% system-bootstrap CI `+0.002` to `+0.021`), while its coverage and interval-score effects
include zero. The numerical effect is statistically detectable only because the arms are almost
identical; it is not scientifically material.

## Decision

The baseline does not pass the declared joint fit/calibration gate. Marginal recovery approaches but
does not reach 85%, and its 60% coverage is unacceptable. Mondrian calibration improves the proper
interval score and coverage, but reaches only 70% and reduces distribution recovery to 81%.

Proceed to the opening-probability and naive-distance screen. Keep raw log-PF as the baseline and
retain both marginal and Mondrian variants; do not treat z-scoring as a selected model.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-baseline --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-baseline-report
```

Outputs are under `outputs/analysis/pairwise_geometry/checkpoint9_strict_likelihood/`. Fit and
calibration are deliberately plotted separately in `strict_likelihood_distribution_fit.png` and
`strict_likelihood_calibration.png`.
