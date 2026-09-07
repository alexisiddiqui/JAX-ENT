# Work Scale RBF Laplacian: ISO validation litmus test

## Purpose

The litmus test asks one end-to-end question:

> Does the locked Work Scale all-pairs RBF regularizer recover the known ISO ensemble
> populations better than MaxEnt without degrading held-out HDX prediction?

This is an integration and transferability test, not a new graph-selection experiment.
Passing it would justify a larger ISO evaluation; it would not by itself establish general
validity or authorize replacement of MaxEnt everywhere.

## Locked configuration

Use the configuration selected by the ATLAS-BV experiments without retuning its geometry:

- Work Scale frame distance.
- Weighted all-pairs RBF graph.
- RBF bandwidth equal to the 16th percentile of positive pairwise Work Scale distances.
- Prior-relative Laplacian penalty on `log(w / p0)`.
- Graph constructed once before optimization and held fixed.
- Existing ISO BV forward model, optimizer, initialization, and data splits.
- Regularization strength selected using validation HDX data only.

Do not transfer sparse `k=20` into the primary ISO test. Checkpoint 33 found no benefit from
making sparse `k` depend on protein length, while the all-pairs RBF graph remained the
stronger basin-recovery benchmark.

## Minimal experimental matrix

Run both synthetic ensembles:

- ISO-BI.
- ISO-TRI.

For each ensemble, run:

- random peptide splits;
- sequence-held-out peptide splits;
- all three existing split replicates;
- the covariance-aware Sigma-MSE HDX objective.

Evaluate four arms on identical train, validation, and held-out splits:

1. Existing MaxEnt.
2. Work Scale all-pairs RBF Laplacian.
3. Rewired Work Scale RBF Laplacian.
4. Uniform all-pairs regularization.

The rewired control tests whether the physical correspondence between Work Scale distances
and frames matters. The uniform complete-graph control tests whether any benefit is merely
generic smoothing of `log(w / p0)`.

## Measurements

The ISO synthetic truth has known cluster populations and weights that are uniform within
each structural cluster. Measure:

- absolute error in each cluster population;
- BI/TRI mode-ratio recovery;
- frame-weight total-variation distance from the synthetic truth;
- held-out peptide HDX error;
- validation-selected regularization strength;
- final ESS, as a diagnostic only;
- optimization convergence and frame-weight simplex validity.

Report both absolute RBF gain over MaxEnt and fractional recovery:

```text
absolute TV gain = TV(MaxEnt, truth) - TV(RBF, truth)
fractional recovery = absolute TV gain / TV(MaxEnt, truth)
```

Bin results by the corresponding MaxEnt frame-weight TV error. The ATLAS-BV difficulty
analysis showed that the Laplacian has little measurable headroom when MaxEnt is already
close to the truth, whereas its absolute and fractional benefit increase with residual
MaxEnt error.

## Required plots and tables

Produce:

- A paired scatter plot with MaxEnt TV error on the x-axis and RBF TV gain on the y-axis,
  coloured by ISO-BI/ISO-TRI and shaped by random/sequence split.
- A binned plot of absolute and fractional recovery against MaxEnt-error quartile.
- A paired cluster-population-error plot for MaxEnt, RBF, rewired RBF, and uniform
  all-pairs.
- A held-out HDX-error comparison for the same four arms.
- A table with one row per ensemble, split type, replicate, strength, and arm.
- A summary table containing paired effects, confidence intervals, selected strengths,
  ESS, and pass/fail status for every gate.

## Litmus-test gate

The ISO integration passes only if all of the following hold:

- RBF reduces median cluster-population error relative to MaxEnt.
- RBF reduces median frame-weight TV relative to MaxEnt.
- Median held-out HDX error is no more than 1% worse than MaxEnt.
- The physical RBF graph outperforms both rewired RBF and uniform all-pairs controls.
- Improvement is present in both ISO-BI and ISO-TRI rather than being driven by one
  ensemble.
- Splits with meaningful residual MaxEnt error show positive fractional recovery.
- Every fitted weight vector is finite, non-negative, normalized, and numerically stable.

Strength selection must use validation data only. Held-out HDX observations and synthetic
truth weights must not be used to choose the strength or graph configuration.

## Interpretation limits

The ISO truth is uniform within each structural cluster. Consequently, this example can
test:

- recovery of BI/TRI basin populations;
- preservation of within-cluster uniformity;
- held-out HDX generalization;
- whether physical Work Scale geometry adds value beyond generic smoothing.

It cannot validate arbitrary within-basin frame-weight structure or demonstrate that the
regularizer recovers a continuous Boltzmann distribution inside a basin.

If the litmus test passes, proceed to a larger preregistered ISO comparison. If it fails,
the Work Scale RBF mechanism should not replace MaxEnt in ISO, even if it remains useful in
the ATLAS-BV synthetic recovery experiments.

## Supporting ATLAS-BV evidence

- [`../ATLAS_BV/CHECKPOINT32_LAPLACIAN_TOPOLOGY.md`](../ATLAS_BV/CHECKPOINT32_LAPLACIAN_TOPOLOGY.md)
- [`../ATLAS_BV/CHECKPOINT33_RESIDUE_K_SCALING.md`](../ATLAS_BV/CHECKPOINT33_RESIDUE_K_SCALING.md)
- [`../ATLAS_BV/CHECKPOINT34_BASIN_DIFFICULTY.md`](../ATLAS_BV/CHECKPOINT34_BASIN_DIFFICULTY.md)
