# ISO_TRI hierarchical graph comparison

Run from the repository root:

```bash
bash jaxent/examples/1_IsoValidation_OMass/commands.sh omc-hierarchical-control --phase all --workers 10
```

Phases are `prepare`, `run`, `report` and `all`. Results and the HTML report are
under `analysis/omc_hierarchical_control/`. A different `--output` can isolate a
new run. Preparation and fitting resume only verified artifacts; scientific code
or input changes require a fresh output directory.

The experiment uses the frozen 2,225-frame ISO_TRI candidate and hard-contact
features from `omc_control`, its frame-wise uptake target, and the grouped-uptake
fitter. The target is 40% open, 60% closed and 0% intermediate. No contact
regeneration, candidate filtering, target change or lower-strength fits occur.

## Graphs

Three average-linkage trees use exact pairwise optimally aligned protein Cα RMSD,
the ATLAS-style 256-quantile W1 approximation of internal Cα-distance
distributions, and RMS distance between full log-PF profiles. Candidate coordinate
frame order is checked against both saved reference-RMSD columns. W1 is audited
against exact empirical distances on 1,024 pairs and 16 complete anchor searches.

For every hierarchy merge, select the closest original-distance frame pair across
the two child clusters. Resolve equal closest pairs lexicographically by original
frame indices. SciPy average linkage receives the fixed original frame order;
its linkage tie handling is deterministic in this environment. Keep all merge
links, producing a connected undirected tree with N−1 edges. No state labels enter
this construction. This is average linkage plus representative merge links, not
a minimum-spanning-tree or a cut into population groups.

Retained edges use the original mean-log-PF Gaussian weights at the original
scalar bandwidth quantiles 0.02, 0.04, 0.08, 0.16, 0.32 and 0.64. Rescale each
masked kernel to the original scalar off-diagonal total coupling at the same
bandwidth. Absent edges stay zero and the conventional diagonal is one. Save
edge fractions, effective weighted edge counts, degree distributions, state-pair
coupling fractions and graph overlaps. State-pair fractions are directed and
sum to one; symmetric off-diagonal state pairs should be summed when interpreted
as an undirected pair.

## Fits and interpretation

There are 18 new configurations, two starts each, strength 0.1, using the existing
float64 Adam fitter and unchanged checkpoint/plateau/start-agreement rules. All
25 previous grouped ISO_TRI controls are copied with hashes. Unresolved outcomes
remain visible; each family's selected configuration minimises all-residue MSE
among converged fits. No converged outcome means no selection.

The report independently reconstructs prediction, MSE, explicit pairwise OMC
objective and population metrics from saved weights. It includes complete CSV
tables, bandwidth curves, recovery/MSE/ESS comparisons, selected populations and
graph diagnostics, in PNG and SVG. Recovery is the existing JSD-based score.

Between-new-graph comparisons isolate distance choice within this construction.
Comparisons to old dense/kNN graphs also change topology. Equal total coupling
does not imply equal spectral connectivity or state coupling, especially on a
tree. W1 approximation quality is reported separately from physical validity.

Validation command:

```bash
.venv/bin/python -m pytest -q tests/test_iso_omc_hierarchical.py tests/test_atlas_graph_representation_audit.py
```

Tests cover independent RMSD/W1 identities, closest-pair merge links, ties, tree
connectivity, masked scalar weights, total coupling, explicit pairwise penalty
and selection excluding unresolved fits. Completion requires all 18 outcomes,
validated saved results and an HTML report; improvement is not assumed.
