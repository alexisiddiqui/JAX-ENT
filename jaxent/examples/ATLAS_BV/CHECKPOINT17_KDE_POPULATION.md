# Checkpoint 17: MD KDE target populations

This experiment changes the target, not the BV representation. Within each MD replica, each
frame receives a local probability-density estimate from a Gaussian kernel over the frame-to-frame
Wasserstein-1 geometry matrix. The bandwidth is learned only from replica A as its median kth-neighbour
distance (primary `k=10`; sensitivity `5, 20, 50`) and is then held fixed for replicas B and C.

The observed pairwise target is `log rho_i - log rho_j`. Fixed BV predicts its signed value from the
sum of residue log-protection-factor differences. Magnitude-only L1, L2, cosine, correlation, and
per-residue ridge models are reported separately; they must not be interpreted as signed population
predictions. Replica A fits model scale, replica B conformally calibrates 90% intervals, and untouched
replica C evaluates them, rotating through all six ordered assignments. A leave-one-system-out scale is
also evaluated without allowing the target system to tune its own coefficient.

Distribution fit is `100 * (1 - sqrt(JSD))`, not an error in angstroms. Angstrom values occur only on
the structural x-axis: globally shared W1 or RMSD bands.

Run:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-kde-population --workers 2
./jaxent/examples/ATLAS_BV/commands.sh geometry-kde-population-report
```

The principal plot is `outputs/analysis/pairwise_geometry/checkpoint17_kde_population/
kde_population_recovery_global_w1.png`. The fixed-versus-fitted plot explicitly tests whether apparent
failure at alpha=1 is predominantly a scale mismatch.

## Completed result (111 systems)

The rank-10 signed fixed-BV sum with a per-system scale recovers a median 72.5%, 60.1%, 51.3%,
44.1%, 37.2%, and 31.0% across global W1 bands q0--q5. The leave-one-system-out, residue-normalized
signed model is effectively identical (72.7%, 60.7%, 51.2%, 44.5%, 36.5%, 30.8%). In contrast,
forcing alpha=1 gives only 17.3--24.9%; the paired diagnostic confirms that this is predominantly a
scale mismatch rather than an absence of signed BV signal.

For magnitude-only comparisons, cosine and correlation are strongest overall (roughly 50% mean
recovery across bands), followed by per-residue ridge (48%), then L2 and absolute L1 (37% and 36%).
These metrics do not encode the direction of the population change and therefore remain a separate
question from the signed BV result.

The bandwidth sensitivity is small: signed local-alpha recovery averaged over the six W1 bands is
49.4%, 49.4%, 49.6%, and 50.9% at neighbour ranks 5, 10, 20, and 50. Strict 90% conformal coverage
for the signed local-alpha model averages 88.0% across W1 bands, showing mild residual undercoverage
but not the severe tail collapse seen in the earlier RMSD-prediction formulation.
