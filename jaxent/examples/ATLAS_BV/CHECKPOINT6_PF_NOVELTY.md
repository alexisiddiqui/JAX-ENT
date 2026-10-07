# Checkpoint 6 — PF-vector novelty calibration

Completed on 2026-08-28 for 111 systems and 333 replica holdouts. PF novelty is the Euclidean norm
of the training-standardized complete per-residue absolute PF-change vector, divided by the square
root of residue count. Scale is constrained to remain flat or increase with novelty. Bin count is
selected across training replicas by central interval score, with constant scale retained as a
candidate.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-novelty-calibration --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-novelty-compare
```

## Result

Radial PF novelty does not repair tail calibration:

| Endpoint | Features | Recovery | 90% coverage | Median NLL |
|---|---|---:|---:|---:|
| Global RMSD | raw | 77.39% | 81.63% | 1.702 |
| Global RMSD | z-scored | 77.32% | 80.00% | 1.718 |
| W1 q5 | raw | 86.81% | 52.79% | 0.424 |
| W1 q5 | z-scored | 86.80% | 51.97% | 0.433 |

Against Checkpoint 4's linear-variance likelihood, raw novelty calibration changes W1-q5 recovery
by a paired median `+3.11` percentage points, but reduces coverage by `19.47` points (95%
system-bootstrap interval `-21.76–-17.97`) and worsens NLL by `0.160`. Z-scored novelty reduces
coverage by `15.86` points (`-18.69–-14.40`) and worsens NLL by `0.213`.

It is also worse calibrated than Checkpoint 5's rejected predicted-mean scale model: W1-q5 coverage
falls by another `9.48` raw and `9.93` z-scored percentage points. The monotone constraint rules out
the trivial failure in which larger novelty receives a narrower interval.

Only 16.7% of folds select constant scale; 38.1% select 20 novelty bins. Thus the calibrator finds
replica-specific structure in radial novelty, but that structure does not transfer to the held-out
replica's long-distance conformations.

## Interpretation

Large structural W1 departures are not reliably large radial departures in the complete PF-change
vector. Consequently, the remaining tail error is not repaired by a simple out-of-support
inflation rule. This strengthens the directional-degeneracy interpretation: distinct residue
patterns can occupy similar PF radius while representing very different structural motions.

This result rejects **radial norm novelty**, not every possible support statistic. Exact
nearest-training-vector distance is the remaining stronger support diagnostic because it preserves
direction and local density. If that also fails to identify the tail, the fixed BV representation
has reached its measurable limit for large structural departures.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint6_novelty/novelty_recovery_coverage_comparison.png`
- `outputs/analysis/pairwise_geometry/checkpoint6_novelty/scale_results.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint6_novelty/scale_hyperparameters.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint6_novelty/novelty_paired_comparisons.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint6_novelty/checkpoint6a_report.yaml`
- `outputs/analysis/pairwise_geometry/checkpoint6_novelty/checkpoint6b_report.yaml`
