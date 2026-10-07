# Narrower ISO_TRI hybrid bandwidths

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hybrid-narrow --phase all --workers 3
```

Phases are prepare/run/report/all; `--output` defaults to `analysis/omc_hybrid_narrow` within the ISO example. Three new fits use 0.25, 0.5 and 0.75 times the original q=0.02 scalar bandwidth (0.004081632653054612 mean-logPF units). The original 1× hybrid fit is copied as a reference; original results are preserved.

The symmetric 20-neighbour full-profile mask stays fixed. Each narrower scalar Gaussian uses the existing exponent cap of 80. Retained off-diagonal edges are rescaled to the total coupling of the original 1× hybrid graph; diagonal entries stay one and absent edges remain zero. This tests redistribution of a fixed amount of coupling, not a reduction in regularisation strength. Multipliers are not distance quantiles. Preparation verifies that the 1× construction reproduces the original kernel.

The original grouped-uptake fitter, frame-wise target, candidate frames, OMC strength 0.1, normalised all-residue MSE, two initialisations, optimiser and convergence checkpoints remain fixed. Recovery is evaluation-only. Fits that fail numerical convergence are saved and excluded from selection; no checkpoint extension or extra bandwidths are automatic. The selected fit has minimum all-residue MSE among converged results, with smaller multiplier breaking exact ties.

Per-fit locks, checksums and input/code hashes support resumption. Changed scientific inputs or code require a fresh output directory. The HTML report includes all four results, valid differences from the reference, both-start population diagnostics, conditional ESS, RMSD coverage, residue/timepoint residuals and fixed-coupling checks. Predictions and explicit pairwise OMC objectives are independently reconstructed from archives.
