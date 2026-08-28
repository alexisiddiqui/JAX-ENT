# Checkpoint 3A — per-residue vector support and scalar reproduction

Completed on 2026-08-27 for 111 systems and all 333 outer replica holdouts. The command is:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-vector-audit --workers 4
```

The audit constructs pair features as `abs(z[:, i] - z[:, j])`, preserving one column per
residue. Systems contain 55–237 residues (median 116). Every outer fold contains exactly 75,000
training and 50,000 test pairs; the predeclared exact-kNN comparison uses deterministic caps of
5,000 and 10,000. Both inner tuning directions contain 5,000 within-replica pairs. All 333 outer
folds and all 666 A-to-B/B-to-A directions pass replica-isolation assertions. Cross-replica pairs
are absent from inner tuning and retained only for the later outer refit.

Training-pair z-scoring is feasible without dropping features: no residue falls below the variance
floor in any fold. The feature dimension remains system-specific, and neither held-out frames nor
held-out pair statistics enter centring or scaling.

The capped scalar baselines reproduce the important full-pair conclusion for structural W1:
Absolute-L1 has the highest median q5 recovery (`40.60%`), followed by raw L2 (`40.11%`), cosine
(`36.91%`), frame-centred L2 (`36.59%`), and correlation (`36.45%`). For global RMSD the capped
ranking is closer: Absolute-L1 gives `52.95%`, cosine `52.79%`, raw L2 `52.61%`, frame-centred L2
`51.85%`, and correlation `51.60%`. The full-pair RMSD ranking is not identical, showing the
expected sampling sensitivity of the capped comparison; all later capped models must therefore use
these exact cap memberships and be compared pairwise rather than against the full-pair medians.

Checkpoint decision: the feature and validation support is valid. Ridge/PCA-ridge remains paused
until review.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint3_vector/feature_support_audit.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/capped_scalar_baselines.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/capped_scalar_tail_recovery.png`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/vector_feature_support.png`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/checkpoint3a_report.yaml`

