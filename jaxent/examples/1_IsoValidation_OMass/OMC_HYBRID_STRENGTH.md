# Lower-strength OMC on wider ISO_TRI hybrid graphs

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hybrid-strength --phase all --workers 6
```

Phases are prepare/run/report/all. `--output` defaults to `analysis/omc_hybrid_strength` within the ISO example. `--smoke` runs 20 steps at q=0.16 and strength 0.01, then exits without scientific fits. Normal execution performs this smoke check before launching the six configurations.

The experiment adds strengths 0.01 and 0.03 at bandwidth quantiles 0.16, 0.32 and 0.64. It copies all six original ISO_TRI hybrid fits, six MaxEnt fits and the unregularised fit as controls: 19 configurations in the final report. Existing unresolved flags are preserved. Original experiment code and outputs remain unchanged.

Kernels are byte-identical copies of the original hybrid graphs. No recomputation or scaling compensates for lower strength. The 20-neighbour full-profile mask, scalar edge weights at each bandwidth, candidate frames, frame-wise target, intrinsic rates and grouped-uptake fitting semantics stay fixed. The general fitter receives the requested strength directly; the hybrid wrapper that hardcodes 0.1 is not used.

The objective is all-residue/all-timepoint MSE divided by the frozen target variance plus 1e-8, plus the requested strength times the existing OMC penalty. Float64 Adam at 0.05, uniform and seeded perturbed logits, checkpoints 1000/3000/10000, and the original plateau and between-start agreement requirements are preserved. Both starts are saved. No automatic checkpoint extension, mean-rate fitting or hierarchical graph experiment is included.

Select the lowest-MSE converged hybrid across the available combinations, and separately within each strength. Exact ties use quantile then strength order. Target populations remain evaluation-only. Same-bandwidth differences compare lower strengths to 0.1 only when both fits converge. Nearest-ESS comparisons use the original hybrid and MaxEnt controls with a two-percentage-point tolerance and no interpolation.

The report includes all configurations, populations, recovery, intermediate mass, ESS, raw MSE, conditional ESS, within-state RMSD distributions, residue/timepoint residuals, and between-start population differences. It reconstructs predictions and explicit pairwise OMC objectives from saved weights. Input/code hashes, per-fit locks and checksummed completion records protect resumption; changed scientific inputs or code require a fresh output directory.
