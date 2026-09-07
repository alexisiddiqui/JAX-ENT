# Checkpoint 34: basin recovery conditional on MaxEnt difficulty

## Purpose and protocol

Checkpoint 32 found positive aggregate basin recovery for the Work Scale all-pairs RBF
Laplacian, but recovery was unresolved for the largest protein-length quartile. Checkpoint
33 found no evidence that scaling sparse `k` with residue count solved this problem.

This checkpoint is a post-hoc reanalysis of saved replica-C results from the 24-system
development and 87-system confirmation cohorts. It retains the locked Work Scale all-pairs
RBF graph with bandwidth at the 16% positive-distance quantile. No frame weights,
regularization strengths, graph parameters, or MaxEnt fits were recomputed.

Each of the three basin-transfer challenges per system is binned into a quartile of baseline
MaxEnt weight-TV error. Recovery is reported as absolute TV gain and as gain divided by the
MaxEnt error. Confidence intervals use 10,000 system-cluster bootstrap samples, keeping the
three challenges from each protein together.

## Results

| Difficulty | Mean MaxEnt TV | Mean RBF TV gain | 95% CI | Fraction recovered | Positive challenges |
|---|---:|---:|---:|---:|---:|
| D1, easiest | 0.0140 | 0.00037 | 0.00009–0.00073 | 2.5% | 58.3% |
| D2 | 0.0337 | 0.00254 | 0.00148–0.00373 | 7.5% | 75.9% |
| D3 | 0.0613 | 0.00633 | 0.00329–0.00964 | 9.8% | 66.3% |
| D4, hardest | 0.1141 | 0.01367 | 0.00555–0.02266 | 11.5% | 57.8% |

RBF benefit rises with the error available to recover. Protein length is also strongly
associated with difficulty (`rho=-0.429`, `p=2.44e-16`): Q4 contributes 39 observations to
the easiest bin but only 7 to the hardest, whereas Q1 contributes 7 to the easiest and 39 to
the hardest. Much of the raw length trend is therefore a difficulty-composition effect.

Conditioning does not remove length completely. A descriptive standardized regression of
absolute gain on MaxEnt error and residue count gives coefficients `+0.00542` for baseline
error and `-0.00137` for length. The marginal length/gain association is `rho=-0.196`
(`p=0.00033`). The interaction bins are unevenly populated, so this remaining association is
descriptive rather than an independently confirmed mechanism.

## Decision

Retain Work Scale all-pairs RBF at the locked 16% bandwidth. Do not replace it with
residue-scaled sparse `k`. Report basin recovery conditional on baseline MaxEnt difficulty
as well as protein length. The largest-protein result reflects both reduced recovery
headroom and a smaller residual length effect, not categorical failure of the RBF graph.

Artifacts are in `outputs/analysis/pairwise_geometry/checkpoint34_basin_difficulty/`:

- `basin_recovery_by_maxent_error_bin.png`
- `basin_recovery_conditioned_on_difficulty.png`
- `basin_difficulty_bins.csv`
- `basin_length_difficulty_interaction.csv`
- `basin_difficulty_rows.parquet`
- `checkpoint34_report.yaml`
