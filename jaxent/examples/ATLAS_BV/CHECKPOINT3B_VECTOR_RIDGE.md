# Checkpoint 3B — per-residue ridge and PCA-ridge

Completed on 2026-08-27 for 111 systems and all 333 replica holdouts. Each target and preprocessing
arm selects hyperparameters from A-to-B and B-to-A within-replica validation, then refits on all
75,000 outer-training pairs and evaluates all 50,000 held-out pairs. The command is:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-vector-ridge --workers 4
```

Raw ridge gives the strongest aggregate tail recovery: `64.43%` for global RMSD and `53.64%` for
W1 q5. The corresponding full-pair Absolute-L1 scalar baselines are `54.00%` and `45.01%`.
After averaging folds within systems, raw ridge improves recovery by a median `5.43` percentage
points for global RMSD (95% system-bootstrap interval `3.87–7.55`; 85 systems with a global band)
and `7.67` points for W1 q5 (`6.17–10.14`; all 111 systems). It improves 77.6% and 83.8% of
contributing systems, respectively. These are checkpoint diagnostics; the predeclared final
familywise decision remains after kNN.

Z-scoring does not improve ridge recovery (`63.12%` RMSD, `53.53%` W1 q5). PCA also does not help:
raw PCA-ridge reaches `64.13%`/`52.86%`, while z-scored PCA-ridge reaches `63.66%`/`53.47%`.
For W1, 99% retained variance is selected in 67.9% of raw and 58.0% of z-scored folds, indicating
that aggressive compression removes useful information. Raw ridge is therefore the leading vector
arm for the next comparison.

Hyperparameter surfaces are not uniformly sharp: depending on target/model, 9–50% of selections
hit the minimum alpha and 1–19% hit the maximum. The outer improvements nevertheless remain
positive under paired system aggregation. Coefficient identity is moderately reproducible across
replica folds: raw-ridge fold-pair Spearman correlation has median `0.580` for RMSD and `0.603` for
W1; median sign-stable residue fractions are `57.5%` and `54.8%`.

Checkpoint conclusion: retaining residue identity materially repairs both structural tails, which
supports scalar PF-distance degeneracy. Exact capped kNN remains paused until review.

Outputs:

- `outputs/analysis/pairwise_geometry/checkpoint3_vector/ridge_pca_tail_recovery.png`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/ridge_pca_results.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/ridge_pca_hyperparameters.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/ridge_pca_coefficients.parquet`
- `outputs/analysis/pairwise_geometry/checkpoint3_vector/checkpoint3b_report.yaml`

