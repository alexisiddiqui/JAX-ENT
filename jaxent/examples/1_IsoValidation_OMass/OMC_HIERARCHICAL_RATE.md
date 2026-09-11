# Mean-rate hierarchical ISO_TRI comparison

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hierarchical-rate --phase all --workers 10
```

Phases: `prepare`, `run`, `report`, `all`; `--output` chooses an isolated directory.
The default report is `analysis/omc_hierarchical_rate/index.html` in this example.

Eighteen new fits use the existing RMSD, W1 and full-profile log-PF hierarchical
graphs, six original scalar bandwidth quantiles (0.02 through 0.64), and strength
0.1. All kernels are copied byte-for-byte from the grouped-uptake experiment.
The original 2,225-frame candidate, hard-contact rates, frame-wise uptake target,
initialisations, optimiser and convergence rules are unchanged.

The forward model is the existing global mean-rate mode:
`uptake(t,r) = 1 - exp(-t * sum_i w_i * rate(r,i))`.
It averages exchange rates before uptake, across all frames without state groups.
It does not average log-PF or uptake. State labels only enter reported population
metrics. Bandwidth selection minimises all-residue MSE among converged fits.

The runner reuses the original fit executor and its scalar-log-PF OMC branch,
passing the supplied hierarchy kernel and `rate` mode explicitly. Saved aggregate
rows retain the correct hierarchical family name. There are 19 copied mean-rate
controls (unregularised, MaxEnt, scalar and profile OMC) and 18 copied grouped
hierarchy controls. All source and copied artifacts are hashed. Existing runs
resume only verified archives; changed scientific code requires a fresh output.

The report independently reconstructs every saved prediction, MSE, population
metric and pairwise objective in its actual forward-model mode. It includes full
bandwidth tables, MSE selections, same-graph mean-rate versus grouped comparisons,
convergence flags and PNG/SVG plots. Paired differences require both fits to
converge. An unresolved configuration is reported but cannot be selected.

```bash
.venv/bin/python -m pytest -q tests/test_iso_omc_hierarchical_rate.py
```

Tests distinguish global mean rate from grouped uptake, verify independence from
state labels, and check all 18 configurations route the correct mode, strength
and OMC branch to the existing fitter. No lower-strength experiments are run.
