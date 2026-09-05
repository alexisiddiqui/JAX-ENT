# Checkpoint 26: structural-W1 graph PyRosetta energy geodesics

## Method

Each post-equilibration frame is a node in a replica-specific structural-W1 graph. Edges are the
symmetrized union of the 5, 10, or 20 nearest structural-W1 neighbours. Standard `ref2015` total
energies are robustly scaled from replica A and used in three positive shortest-path costs:
accumulated energy variation, directed uphill action, and energy-weighted W1 length. Endpoint
energy ordering supplies the sign. This avoids the invalid alternative of summing signed energy
increments, which telescopes exactly to the direct endpoint energy difference.

Replica A fits the nonnegative population scale, replica B independently selects graph parameters
by unstratified MAE, and replica C is tested once. Signed and magnitude targets are tuned
separately. Geometry-only and deterministically shuffled-energy graphs distinguish an energy gain
from reuse of the W1 geometry that also defines the MD KDE target.

The fixed 24-system pilot contains 12 q0-eligible and 12 q5-eligible systems selected across
protein-size quartiles without using recovery. No system is eligible in both global extreme bands.

## Result

The predefined pilot gate failed. A subsequent requested full-cohort run was nevertheless completed
to measure the result without relying on the small extreme-band subsets.

For magnitude recovery, direct fitted `ref2015` gives 72.5% in q0 and 58.1% in q5. Uphill and
energy-weighted paths show small apparent q0 gains in paired systems (+5.6 and +6.2 percentage
points), but lose 13.9 and 12.9 points in q5. Their aggregate q5 recoveries fall to 34.3% and 34.6%.
Accumulated energy variation gives 72.0% in q0 and 42.4% in q5, a paired q5 loss of 9.6 points.
The graph models are also nearly indistinguishable from geometry-only or shuffled-energy controls.

Signed recovery is not repaired. Seven of 24 direct and energy-graph fits choose alpha=0, showing
that PyRosetta endpoint energy ordering is inconsistent with the signed MD KDE target in a
substantial fraction of systems. None of the graph families improves both extremes.

Work Scale remains the strongest balanced magnitude predictor in this pilot: 89.5% q0 and 56.7%
q5. Direct `ref2015` remains preferable to its graph transformations when a PyRosetta control is
needed.

The direct magnitude `ref2015` alpha retains its strong inverse relationship with feature variance
(Spearman rho=-0.97, 95% bootstrap interval [-0.99, -0.87]). Graph transformation weakens this to
rho values between -0.27 and -0.39, with every interval crossing zero. Thus the graph does not
preserve the particularly clean scale-compensation property that motivated this experiment.

## Full-cohort confirmation

All 111 systems completed. The median per-system runtime was 16.4 seconds; 40 individual low-k
candidate graphs were disconnected and excluded, but every system retained complete connected
candidates and produced results.

The full cohort confirms the pilot conclusion. Magnitude recovery across q0/q5 is 71.5%/61.9% for
direct `ref2015`, 72.2%/43.4% for total variation, 72.4%/33.6% for uphill action, and 73.5%/33.2%
for energy-weighted W1. Work Scale gives 89.2%/59.2%. Thus the graph variants provide no material
q0 improvement and lose between 11.4 and 15.4 paired percentage points at q5. Bootstrap intervals
for the mean q5 losses exclude zero for every graph family.

Signed recovery is likewise not improved. Thirty of 111 direct signed fits select alpha=0, and the
graph families select zero in 30--32 systems. The direct magnitude alpha retains rho=-0.93 with
feature variance, compared with -0.38 for total variation, -0.54 for uphill action, and -0.63 for
energy-weighted W1.

## Command and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-pyrosetta-graph --pilot --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-pyrosetta-graph --full --workers 4
```

The `pilot` and `full` subdirectories under
`outputs/analysis/pairwise_geometry/checkpoint26_pyrosetta_graph` contain recovery and alpha-variance
figures, per-system results and graph audits, bootstrap tail comparisons, selected parameters, and
runtimes. The pilot additionally contains the machine-readable `pilot_gate.yaml`. Per-system parts
make both runs resumable.
