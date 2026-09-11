# Mean-rate target fitted with grouped uptake

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hierarchical-reverse --phase all --workers 10
```

Phases: `prepare`, `run`, `report`, `all`; `--output` isolates another run.
Default report: `analysis/omc_hierarchical_reverse/index.html` in this example.

The target is global mean-rate uptake from the original full reference ensemble,
using its original 40% open / 60% closed weights and hard-contact exchange rates:
`1-exp(-time * weighted_mean_reference_rate)` per residue. The original frame-wise
target is reconstructed first to verify alignment and conventions. The new
target must be finite, nondegenerate and at least as large as frame-wise uptake
at the same weights (concavity check). Data-loss scaling follows the existing
rule, using the new target variance plus 1e-8.

The candidate predictor is grouped uptake: rates are averaged within fixed
candidate state groups, uptake computed per group and mixed by group mass.
Candidate frames, hard-contact rates, hierarchy graphs, scalar edge weights,
bandwidths, optimiser, initialisations and convergence criteria stay unchanged.

There are 18 hierarchy fits at strength 0.1 (three graphs, six bandwidths), plus
19 newly fitted baseline configurations (unregularised, six MaxEnt strengths,
six scalar OMC and six full-profile OMC). Prior fits to the old target are not
reused as controls. There are no lower-strength hierarchy fits in this experiment.

The report independently verifies target construction, prediction, MSE, population
metrics and pairwise objective from saved artifacts. It includes complete tables,
MSE-based selections among converged fits, preflight errors at the generating
reference/candidate masses, and PNG/SVG plots. Unresolved fits remain visible.
All source and copied artifacts are hashed and completed fits are resumable.

Recovery evaluates the generating 40:60:0 masses, which need not minimise loss
under a different forward approximation. MSE against this new target should not
be treated as directly paired with MSE against the old frame-wise target.

```bash
.venv/bin/python -m pytest -q tests/test_iso_omc_hierarchical_reverse.py
```

Tests cover reference mean-rate construction, the uptake-mixture inequality,
actual executor routing of grouped prediction with the new target, baseline
and hierarchical arm saving/resumption, and the 37-configuration grid.
