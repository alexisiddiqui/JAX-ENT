# Consistent mean-rate target and prediction

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hierarchical-matched-rate --phase all --workers 10
```

Phases: `prepare`, `run`, `report`, `all`; `--output` selects a separate directory.
Default report: `analysis/omc_hierarchical_matched_rate/index.html` in this example.

This copies the previous reverse experiment's mean-rate reference target, target
variance normalisation, candidate rates, frame order and graph kernels byte for
byte. Both target and prediction now use
`uptake(t,r) = 1-exp(-t * sum_i weight_i * rate(r,i))`.
State groups are used only to evaluate populations, not in the prediction.

All 37 configurations are newly fitted: 18 hierarchy arms at strength 0.1 across
three distances and six bandwidths, plus 19 baseline arms (unregularised, six
MaxEnt strengths, six scalar OMC and six profile OMC settings). Optimiser,
initialisations, convergence criteria and all-residue-MSE selection are unchanged.
The previous 37 grouped-prediction fits are copied as same-target controls.

The shared report validator reconstructs mean-rate target and prediction, MSE,
pairwise regularisation objective and population metrics from saved artifacts.
Tables and PNG/SVG plots cover the full sweep, selected populations and recovery
tradeoffs. Paired CSV comparisons show mean-rate minus grouped-prediction results
only where both fits converged. Unresolved fits remain visible and cannot be
selected. Source/copy hashes and atomic completion markers support verified
resumption; changed scientific code requires a fresh output directory.

Matching forward-model conventions removes that approximation mismatch but does
not guarantee identifiable state populations or sufficient candidate coverage.
Preflight errors at the reference masses are retained for interpreting these
remaining limitations. No lower-strength hierarchy fits are included.

```bash
.venv/bin/python -m pytest -q tests/test_iso_omc_hierarchical_matched_rate.py
```

Tests verify exact target/prediction agreement at identical rates and weights,
distinguish grouped uptake, and exercise actual baseline/hierarchy executor
saving and resumption with mean-rate mode and the complete 37-arm grid.
