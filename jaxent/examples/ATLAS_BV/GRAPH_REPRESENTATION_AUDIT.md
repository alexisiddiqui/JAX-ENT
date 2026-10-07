# ATLAS graph representation audit

Run from the repository root:

```bash
bash jaxent/examples/ATLAS_BV/commands.sh geometry-graph-representation --phase all --workers 10
```

The report is `outputs/analysis/pairwise_geometry/graph_representation_audit_bradshaw_0p1/index.html`
under this example. `--phase report` regenerates tables and plots after verifying
source, scientific code and artifact hashes. Completed systems resume only when
those hashes match. Use a fresh `--output` directory after scientific changes.

This compares representations on 111 ATLAS systems, using 256 evenly spaced
post-10-ns frames from each of three replicas. It regenerates Bradshaw switched
contacts through the standard BV featuriser, leaving historical feature banks and
the example's global configuration unchanged. Radii are 6.5 Å for heavy contacts
and 2.4 Å for acceptors; BV coefficients are 0.35 and 2. Both switch scales are
explicitly 0.1 Å in the MDAnalysis coordinate convention, as requested. The switch
is `1/(1+((distance-centre)/scale)^6)`. Frame indices, fresh features and topology
are saved. This supersedes the mistakenly generated hard-contact audit, which
remains in its separate historical output directory.

Metrics are pairwise optimally aligned Cα RMSD, the existing ATLAS 256-quantile W1
approximation over within-frame Cα distance distributions, full residue-profile
log-PF RMS difference, and absolute mean-log-PF difference. All use identical
frames. Graphs use 20 nearest neighbours and symmetric-union adjacency, with a
second audit restricting neighbours to other replicas. Average-linkage partitions
are compared at 2, 3, 5, 10 and 20 clusters without selecting a preferred count.

Exact empirical W1 is checked on 1,024 pairs per system, half random and half
approximate neighbours, plus exact neighbour searches for 16 sampled anchors.
The latter measures approximation effects where graph construction needs accuracy.

Report medians and interquartile ranges treat systems as the units of comparison.
Structural metrics are comparators, not known physical weight labels. Geometry
agreement alone cannot establish population recovery, an appropriate coupling
strength, or whether distinct structures should receive different weights. The
audit includes no fitting. RMSF groups are descriptive rather than causal.

Validation:

```bash
.venv/bin/python -m pytest -q tests/test_atlas_graph_representation_audit.py
```

Tests cover exact RMSD against MDAnalysis, rigid transformations and reflections,
replica restrictions, scalar/profile distinctions, exact W1 and cluster counts.
