# Final vector-model comparison

Completed on 2026-08-27. This comparison uses the two declared primary endpoints—global RMSD and
W1 q5—and evaluates probability-distribution fit as
`100 * (1 - sqrt(JSD))`. The metric is dimensionless; no distribution error is reported in
Angstroms.

The command is:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-vector-compare
```

## Primary point-model result

Raw per-residue ridge is the selected point model. Against raw Absolute-L1 on the same full
held-out pair sets, its median system recovery improvement is:

| Endpoint | Recovery | Paired improvement | 95% system-bootstrap CI | Systems improved | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Global RMSD | 64.43% | +5.43 pp | +3.87 to +7.55 pp | 77.6% | 8.34e-10 |
| W1 q5 | 53.64% | +7.67 pp | +6.17 to +10.14 pp | 83.8% | 2.11e-14 |

The corresponding raw-ridge median errors are:

| Endpoint | Absolute L1 | L2 | sqrt(JSD) | KLD target-to-prediction |
|---|---:|---:|---:|---:|
| Global RMSD | 0.860 | 0.371 | 0.356 | 1.075 |
| W1 q5 | 1.166 | 0.638 | 0.464 | 2.558 |

All four ridge/PCA arms survive Holm correction, but neither PCA nor z-scoring improves aggregate
recovery over raw ridge. The simplest supported choice is therefore raw ridge. This is consistent
with PF-distance degeneracy: residue identity contains predictive information that a scalar norm
discards.

Raw L2, frame-centred L2, cosine, and correlation do not improve over Absolute-L1 in paired system
tests. Exact kNN point prediction also fails on its identically capped comparison: raw kNN changes
recovery by -0.18 points globally and -1.87 points in W1 q5. The figure separates full-pair and
capped evaluations so their absolute bar heights are not treated as paired comparisons.

## Conditional-distribution result

Neighbour conditional mixtures are a separate prediction family because they estimate local
target probability mass rather than first collapsing each query to a point. Against the
identically capped Absolute-L1 baseline, their paired recovery gains are:

| Features | Global RMSD | W1 q5 |
|---|---:|---:|
| Raw | +3.78 pp | +4.65 pp |
| Z-scored | +3.64 pp | +6.42 pp |

All four conditional tests survive Holm correction within that family. This supports developing a
proper per-residue conditional likelihood model, but does not displace raw ridge as the selected
point model.

## Statistical contract

Replica folds are averaged within each system before inference. Confidence intervals bootstrap
independent systems. Improvement tests are one-sided paired Wilcoxon tests, with Holm correction
across both endpoints and every tested arm within the point family; conditional-mixture tests are
corrected separately. Full-pair ridge/scalar comparisons and capped kNN/scalar comparisons use
their own matched Absolute-L1 baseline—mismatched pair samples are never used as paired evidence.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint3_vector/final_point_model_recovery.png`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/final_point_familywise_tests.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/final_conditional_familywise_tests.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/final_distribution_metrics.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/final_familywise_report.yaml`

Conclusion: scalar distance-function exploration is complete. Retaining per-residue identity with
raw ridge repairs a meaningful part of the tail; local conditional mixtures provide the bridge to
the next joint/per-residue likelihood analysis.
