# ISO OMC population recovery

Run from the repository root:

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-control --phase all --workers 10
```

Phases: `prepare`, `run`, `report`, `all`. `--output` selects an isolated directory. `--smoke` runs a short numerical check after preparation and exits without scientific fits. The default report is `analysis/omc_control/index.html` within this example. Existing ISO and ATLAS outputs are preserved.

The target is full-precision frame-wise uptake from the complete reference trajectories, weighted 40% open and 60% closed (uniform within each source state). Target and candidates use hard BV contacts, coefficients 0.35/2.0, JAX-ENT intrinsic rates in inverse minutes, 300 K, pD 7, and times 0.167/1/10/60/120 minutes. Candidate frames are unchanged; frame ordering and assignments are checked against trajectory RMSDs. No further clustering is performed. Preparation regenerates candidate features with the existing ISO featuriser when an isolated provenance manifest is absent.

Grouped fitting uses the existing ISO `uptake` semantics: weighted mean exchange rate within each fixed conformational group, conversion to uptake, then mixing using fitted group masses. Group assignments provide prior structural information. This differs from averaging log-PF and from averaging each frame's uptake. Mean-rate fitting uses one weighted rate average across the whole candidate. The target never changes between stages.

Each candidate receives 19 configurations: unregularised; reverse-KL MaxEnt strengths 1e-5 through 1 in decades; scalar and full-profile log-PF OMC at strength 0.1 and distance quantiles 0.02/0.04/0.08/0.16/0.32/0.64. Profile distances are RMS over residues. Gaussian off-diagonal mean coupling is matched to scalar at each quantile, and graphs are frozen across stages. Kernel entries can exceed one after matching; they are coupling weights, not probabilities.

For N normalised weights, the OMC penalty is

`N²/2 * sum_ij w_i*w_j*K_ij*(w_i-w_j)²`.

MaxEnt uses `KL(uniform || w)`. The data loss is all-residue/all-timepoint MSE divided by target variance plus 1e-8. Raw MSE is reported. Neither population truth nor validation splits select parameters.

Fits use softmax weights, float64 Adam at 0.05, uniform and seeded N(0,0.01) logits, and checkpoints 1000/3000/10000. Both starts must plateau within 1% over 250 steps, have finite outputs, and agree in objective within 1%. Select the lower-objective start; select converged configurations by raw MSE, resolving exact ties by fixed arm order. Population differences between starts are reported separately: objective agreement alone does not establish unique weights.

Automatic mean-rate progression requires a converged configuration for each of the four methods on both candidates, and strictly greater than 50% recovery for either OMC family's MSE-selected ISO_TRI fit. Recovery uses the existing base-2 JSD over open/closed/intermediate populations: `100*(1-sqrt(JSD))`. Only roundoff outside [0,1] within 1e-12 is clipped. This is an absolute score, not improvement over initial weights. Unconverged arms cannot trigger progression. The gate is recorded in `gate.json`.

Outputs include both starts and predictions, fit/selected tables, residue/timepoint residuals, initialisation population tables, graph coupling diagnostics, nearest-ESS comparisons (maximum 0.02 fraction gap; no interpolation), within-state conditional ESS and RMSD cumulative distributions. Exact-population predictions under each averaging mode diagnose forward approximation and candidate coverage; these weights are never fitting initialisations. Input/code hashes and per-arm checksums protect resumption. Changed scientific inputs or code require a fresh output directory.
