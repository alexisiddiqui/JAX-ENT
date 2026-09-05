# Checkpoint 11 — strict per-residue conditional likelihood

Completed on 2026-08-28 for all 111 systems and all six ordered A-fit/B-calibrate/C-test replica
assignments. Four matched models were evaluated: raw absolute per-residue log-PF changes and raw
absolute per-residue opening-probability changes, each with either a ridge-Gaussian conditional
density or an exact local kNN mixture.

Replica A alone selects ridge regularization or `(k, feature bandwidth)` using endpoint-frame-
disjoint validation. The kNN training support is capped deterministically at 5,000 A pairs. Replica
B conformalizes Gaussian residuals or central kNN-mixture quantiles; untouched C supplies all
distribution-fit and coverage measurements. Every one of the 5,328 model/target assignments used a
valid frame-disjoint split.

## Primary common-support W1-q5 result

| Model | Marginal recovery | Mondrian coverage | Interval score |
|---|---:|---:|---:|
| Raw log-PF ridge Gaussian | **83.09%** | **70.12%** | **0.885** |
| Opening ridge Gaussian | 79.14% | 21.83% | 2.236 |
| Raw log-PF kNN mixture | 50.72% | 54.20% | 1.279 |
| Opening kNN mixture | 46.91% | 38.46% | 1.559 |

Against the matched raw log-PF Gaussian baseline, opening Gaussian loses `3.84` recovery percentage
points (95% system-bootstrap CI `-4.67` to `-2.73`). Raw log-PF kNN loses `33.04` points (CI
`-35.04` to `-31.10`), and opening kNN loses `36.42` points (CI `-37.70` to `-34.56`). All three
also worsen the proper interval score.

The strict raw log-PF Gaussian result reproduces Checkpoint 9 within expected differences from
frame-disjoint rather than pair-random A tuning: `83.09%` versus `83.14%` recovery and exactly
`70.12%` Mondrian coverage.

## Interpretation and decision

The local-mixture improvement seen in the earlier two-replica training experiment does not survive
strict role separation. Here the mixture may learn neighbours only from replica A; replica B is
reserved for calibration. Its neighbour targets do not transfer into replica C's far structural
tail, so a flexible empirical mixture is worse than Gaussian smoothing around the residue-aware
ridge mean.

Opening probability again loses structural magnitude information and is now rejected as both a
point geometry and a conditional-likelihood representation. Raw log-PF per-residue ridge Gaussian
remains the selected baseline, but it still fails the declared `>=85%` recovery and `85–95%`
coverage gate.

Proceed to the low-rank joint-dependence stage using raw log-PF profiles. Per the declared sequence,
contact-community features are evaluated only if low-rank modes improve W1-q5 recovery by at least
two points with a bootstrap interval above zero and no interval-score deterioration.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-likelihood --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-likelihood-report
```

Outputs are under `outputs/analysis/pairwise_geometry/checkpoint11_conditional_likelihood/`.
