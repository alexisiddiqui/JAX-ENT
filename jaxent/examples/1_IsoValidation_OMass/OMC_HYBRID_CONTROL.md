# Hybrid ISO OMC

Run from the repository root:

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hybrid-control --phase all --workers 10
```

The phases are `prepare`, `run`, `report`, and `all`. `--output` selects an isolated directory; the default is `analysis/omc_hybrid_control` within this example. `--smoke` prepares or validates inputs, executes 20 steps on ISO_BI, and exits without scientific fits. Normal execution also checks the smoke fit before launching the twelve configurations.

This experiment reuses the verified ISO frame-wise-uptake target, unchanged candidates, and the grouped-uptake fitter. It copies all 38 grouped control archives, including the unresolved scalar ISO_TRI quantile 0.08 fit, for a total of 50 configurations after twelve new hybrid fits. Mean-rate fitting is not part of this experiment.

Full-profile RMS log-PF distance defines each frame's 20 nearest other frames. Ties use original frame index. The symmetric mask is the union of directed neighbour selections; self-neighbours are excluded. Neither state labels nor target populations construct this mask. Disconnected components are retained without added bridges.

For each of the six original scalar bandwidths (quantiles 0.02, 0.04, 0.08, 0.16, 0.32, 0.64), retain the original scalar Gaussian edge weights inside the mask and set other off-diagonal entries to exactly zero. Rescale retained edges so mean off-diagonal coupling matches the original scalar kernel. Diagonal entries are one. Bandwidths are copied unchanged, not estimated from retained edges. Scaled kernel entries can exceed one.

The existing OMC fitter is called explicitly through its scalar-OMC penalty branch with the supplied hybrid kernel; external results are labelled `hybrid_logpf`. The original scientific code is unchanged. Strength 0.1, variance-normalised all-residue MSE, two seeded initialisations, float64 Adam at 0.05, and checkpoints 1000/3000/10000 retain their existing definitions. Both starts must plateau within 1% over 250 steps and agree in objective within 1%. Unresolved fits are saved and excluded from selection. Select converged configurations by all-residue MSE, with bandwidth order breaking exact ties.

Population truth is evaluation-only. Recovery uses the existing base-2 JSD score, including intermediate mass. Reports show recovery, TV error and MSE against ESS; conditional ESS and RMSD coverage within states; graph components and state composition; degree statistics; edge density and cross-state coupling; residue/timepoint residuals; and initialisation population differences. Same-bandwidth comparisons require both fits converged. Nearest-ESS comparisons use a maximum two-percentage-point gap and no interpolation.

Preparation hashes source and copied artifacts; per-arm locks and checksummed completion records protect resumption. Changed scientific code or inputs require a new output directory. Reports independently reconstruct predictions and evaluate the explicit pairwise OMC penalty, including all copied controls. Numerical completion alone does not establish an advantage in recovery or coverage.
