# Checkpoint 8 — strict replica-separated conformal audit

## Question and decision

This checkpoint tests the last unresolved alternative: was long-distance undercoverage mainly
caused by reusing predictor-fitting replicas to calibrate uncertainty, or does the fixed BV
representation lack enough information about large structural departures?

The result is **representation-limited under the predeclared gate**. Strict calibration repairs a
large fraction of the previous error and is approximately calibrated marginally, but it does not
repair conditional coverage in the farthest coordinate-W1 band.

## Isolation and models

For every one of 111 systems, all six ordered assignments of the three replicas are evaluated:

- replica A fits the structural predictor;
- replica B supplies the finite-sample conformal residual quantile;
- replica C is untouched until coverage and distribution-fit evaluation.

Each role uses 25,000 deterministic within-replica frame pairs. Hyperparameters are selected only
within A. The prespecified models are scalar raw Absolute-L1 with monotone isotonic prediction and
endpoint-linear extrapolation, and ridge regression on the raw absolute per-residue PF-change
vector. Both marginal and ten-region Mondrian conformal intervals use the corrected order statistic
`ceil((n + 1) * 0.90)`. Results are reported for all pairs and for mutually exclusive common-support,
PF-extrapolation, PF-vector-out-of-support, and structurally-novel strata. Support thresholds are
fixed using A/B only. Pair audits report unique and effective frame counts so repeated pairs from a
few frames are not presented as independent conformations.

## Main results

| Evaluation | Absolute-L1 | Per-residue ridge |
|---|---:|---:|
| RMSD, common-support marginal coverage | 91.0% | 91.4% |
| W1, common-support marginal coverage | 92.3% | 91.6% |
| W1 q5, common-support coverage | 39.8% | **60.0%** |
| W1 q5, common-support Mondrian coverage | 53.4% | **70.1%** |
| W1 q5, common-support distribution recovery | 44.0% | **52.3%** |
| W1 q5, historical same-fit coverage | 23.9% | 34.5% |

The system-bootstrap 95% interval for ridge common-support W1-q5 coverage is **54.5–65.3%**. Its
median effective-frame count is **134.3**, so the conclusion is not based merely on a large nominal
pair count. Relative to Absolute-L1, ridge improves W1-q5 coverage by a paired system median of
**19.1 percentage points** (95% CI 15.9–23.2; positive in 90.1% of systems) and recovery by **7.8
points** (95% CI 5.9–9.9; positive in 96.4% of systems).

## Interpretation

The approximately nominal common-support marginal results show that strict replica-separated
conformal calibration is functioning. The rise from 34.5% historical to 60.0% strict W1-q5 coverage
also shows that calibration leakage and heteroscedasticity were material problems. They were not the
whole problem: the farthest W1 band remains 30 percentage points below nominal after removing PF
extrapolation, using an independent calibration replica, retaining residue identity, and applying
conditional Mondrian calibration, which reaches only 70.1%.

The vector improvement confirms scalar PF-distance degeneracy: which residues change contains
information that Absolute-L1 discards. The remaining failure supports the narrower conclusion that
the **fixed BV PF representation does not transfer enough information to calibrate the magnitude of
large structural coordinate-W1 departures across replicas**. This does not mean BV features contain
no structural information, nor does it invalidate their useful central/common-scale recovery.

## Reproduction and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-conformal-strict
./jaxent/examples/ATLAS_BV/commands.sh geometry-conformal-compare
```

Outputs are under `outputs/analysis/pairwise_geometry/checkpoint8_strict_conformal/`. The primary
artifacts are `checkpoint8_final_report.yaml`, `strict_conformal_assignment_summary.parquet`,
`strict_conformal_population.parquet`, `strict_conformal_marginal_population.parquet`,
`strict_conformal_paired_model_effects.parquet`, `strict_conformal_coverage_recovery.png`, and
`strict_conformal_w1q5_support.png`. Per-system pair audits and resumable checkpoints are retained in
the same directory.
