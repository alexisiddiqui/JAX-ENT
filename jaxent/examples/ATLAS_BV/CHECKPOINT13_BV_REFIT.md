# Checkpoint 13 — strict per-system BV coefficient refit

Completed on 2026-09-01 for all 111 systems and all six ordered A-fit/B-calibrate/C-test replica
assignments. This is the final contingent analysis in the fixed-BV representation ladder.

For each system and assignment, replica A alone selects one shared `(bc,bh)` pair for RMSD and W1
from a preregistered 5×5 multiplier grid around `(0.35,2.0)`. Selection minimizes the mean
normalized validation MAE across both targets using endpoint-frame-disjoint A subsets. The final
ridge predictor is fitted on all A pairs, replica B supplies finite-sample marginal and Mondrian
conformal residuals, and untouched replica C supplies all reported probability-distribution and
coverage results. Thus neither coefficients nor ridge penalties are selected on B or C.

## Primary common-support W1-q5 result

| Readout | Per-system refit result |
|---|---:|
| marginal distribution recovery | **83.24%** |
| Mondrian empirical 90% coverage | **72.38%** |
| paired recovery change vs fixed BV | **+0.13 percentage points** |
| 95% system-bootstrap CI for change | **+0.03 to +0.32 points** |
| median Mondrian interval-score change | **-0.081** |

The small recovery improvement is statistically positive but scientifically negligible. It misses
the declared 85% recovery threshold, and 72.38% tail coverage remains far below the acceptable
85–95% range. The final gate therefore fails.

## Coefficient stability

Across 666 ordered assignments, 97.60% select at least one boundary multiplier and none select the
default `(1,1)` multiplier pair. The dominant choice is `bc×2, bh×0.5` (528/666 assignments).
Although 64.86% of systems choose the same pair in every assignment, this apparent stability is
mostly convergence to a grid corner, not identification of an interior optimum. The result must not
be promoted as a newly calibrated universal or system-specific BV parameter set.

## Final conclusion

Strict calibration, support stratification, scalar distances, raw residue vectors, kNN conditional
mixtures, opening-probability transformations, low-rank joint modes, and now per-system coefficient
refitting have all been tested under replica isolation. Fixed BV contact/acceptor features retain
useful local structural-distribution signal, but do not encode enough transferable information to
recover or calibrate the large-displacement W1 tail. More tuning within this representation is not
the justified next step. A new forward representation—most naturally the preregistered M8
SASA/nearest-polar-distance features, or a separately designed residence-time opening model—is
required for a further scientific test.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-bv-refit --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-bv-refit-report
```

Outputs are under `outputs/analysis/pairwise_geometry/checkpoint13_bv_refit/`.
