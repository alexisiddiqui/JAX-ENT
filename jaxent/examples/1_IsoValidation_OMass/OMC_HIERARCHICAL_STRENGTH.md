# Lower strengths on wider hierarchical graphs

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hierarchical-strength --phase all --workers 10
```

Phases: `prepare`, `run`, `report`, `all`; optional `--output` selects an isolated
directory. The default HTML report is
`analysis/omc_hierarchical_strength/index.html` under this example.

The experiment adds 18 configurations: RMSD, W1 and full-profile log-PF merge
graphs at original scalar bandwidth quantiles 0.16, 0.32 and 0.64, each at
strengths 0.01 and 0.03. The corresponding strength-0.1 kernels are copied
byte-for-byte; there is no compensating coupling normalisation. All frozen
hard-contact ISO_TRI inputs, frame order, target, grouped-uptake forward model,
two initialisations and convergence rules are unchanged.

All 43 prior archives are copied as controls with source and destination hashes.
Each distance has its own immutable input/kernel folder. The existing general
lower-strength fitter receives the requested strength, writes resumable per-arm
archives, and publishes completed artifacts to the aggregate fits folder.
Preparation refuses changed scientific code or inputs; fitting verifies saved
archive hashes on resume. New results use arms 43 through 60.

The report independently reconstructs predictions, MSE, pairwise objective and
population metrics from every saved fit. It shows all three strengths at each
wider bandwidth, valid same-bandwidth differences, MSE-selected configurations,
convergence status and recovery/ESS curves. It separately selects among the
three common wider bandwidths, because strength 0.1 also has three narrower
bandwidths that were not tested at lower strengths. Unresolved fits are shown
but excluded from selection and from valid paired differences.

```bash
.venv/bin/python -m pytest -q tests/test_iso_omc_hierarchical_strength.py
```

Tests check the 18-arm grid, actual requested-strength forwarding to the existing
fitter, strength-independent kernel paths, MSE-based selection and exclusion of
unresolved paired controls. No improvement in recovery or ESS is assumed.
