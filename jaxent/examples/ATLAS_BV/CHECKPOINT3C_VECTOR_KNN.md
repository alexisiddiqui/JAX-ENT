# Checkpoint 3C — exact complete-vector kNN

Completed on 2026-08-27 for 111 systems and all 333 replica holdouts. This checkpoint uses the
complete per-residue absolute PF-change vector, exact brute-force neighbours, deterministic caps
of 5,000 training and 10,000 held-out pairs, and replica-isolated inner selection of
`k in {5, 15, 50, 150}`. The command is:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-vector-knn --workers 4
```

Two distribution predictions are deliberately reported separately. The **point-prediction
distribution** is the histogram of inverse-distance-weighted kNN target estimates. The
**neighbour conditional mixture** instead pools the normalized target histograms of the selected
training neighbours for every held-out query. Both are scored as dimensionless probability-mass
fit using `100 * (1 - sqrt(JSD))`; neither y-axis is an error in Angstroms.

The kNN point model does not repair the scalar-distance failure. Against the identically capped
raw Absolute-L1 scalar baseline, raw kNN changes median system recovery by `-0.18` percentage
points for global RMSD (95% system-bootstrap interval `-1.78–0.00`; positive in 33.8% of 80
contributing systems) and `-1.87` points for W1 q5 (`-2.36–-1.00`; positive in 22.5% of all 111
systems). Z-scoring also fails (`-0.15` and `-1.16` points). Thus, simply giving a local nonlinear
model the complete vector is insufficient; the Checkpoint 3B ridge gain comes from supervised,
global residue weighting rather than vector availability alone.

The conditional mixture is more promising. Raw features improve median system recovery over the
capped scalar baseline by `3.78` points globally (`2.03–5.83`; positive in 76.2%) and `4.65`
points in W1 q5 (`3.81–5.43`; positive in 88.3%). Z-scored conditional mixtures give `3.64` and
`6.42` points, respectively. Aggregate median conditional recovery is `61.04%` for global RMSD
and `46.34%` for raw W1 q5, or `61.00%` and `47.77%` after z-scoring.

This conditional result is not ranked as though it were another point regressor: it answers a
different and more relevant question—what target distribution is supported locally by this PF
change profile? It therefore supports the proposed joint/per-residue likelihood direction. The
final comparison should keep point-distribution and conditional-mixture families separate.

Selected `k` values span the grid rather than collapsing to one boundary. For raw RMSD the
selection frequencies for `k=5/15/50/150` are 12.9/33.3/28.5/25.2%; for raw W1 they are
11.4/50.2/27.0/11.4%.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint3_vector/knn_tail_recovery.png`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/knn_results.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/knn_hyperparameters.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/checkpoint3c_report.yaml`

Checkpoint conclusion: exact local kNN point prediction fails to repair the tail, while a local
conditional target distribution materially improves fit. The predeclared final familywise
comparison remains paused pending review.
