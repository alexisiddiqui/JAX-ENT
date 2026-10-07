# Checkpoint 27: Work-metric graph geodesics

## Method

All 111 systems are evaluated with the same replica-specific structural-W1 graphs and strict
replica A-fit/B-tune/C-test assignment as checkpoint 26. The graph edge signal is separately given
by Work Scale, Work Shape, or legacy-Zq Work Density. Three path constructions are selected by
unstratified replica-B MAE: accumulated Work change, additive W1 plus Work change, and W1 length
weighted by Work change. Direct Work metrics and a geometry-only shortest path are controls.

Magnitude is the common headline because Shape and Density are residue-vector distances without a
well-defined thermodynamic sign. Signed Work Scale is retained as a secondary diagnostic.

## Results

| Metric | Construction | q0 recovery | q5 recovery |
|---|---|---:|---:|
| Work Scale | direct | 89.2% | 59.2% |
| Work Scale | accumulated | 79.6% | 45.9% |
| Work Scale | weighted W1 | 75.1% | 34.9% |
| Work Shape | direct | 23.4% | 41.2% |
| Work Shape | accumulated | 69.0% | 40.9% |
| Work Shape | weighted W1 | 70.7% | 35.1% |
| Legacy Work Density | direct | 61.4% | 54.2% |
| Legacy Work Density | accumulated | 71.1% | 45.2% |
| Legacy Work Density | weighted W1 | 73.7% | 38.2% |
| Geometry-only control | shortest W1 path | 69.8% | 35.6% |

Graph transformation consistently pulls the predictors toward the geometry-only curve. For Work
Scale this destroys useful information: even the least harmful accumulated path loses 5.3 paired
points at q0 and 7.6 at q5. Signed Scale is also not improved.

Work Shape appears to gain about 46--48 paired points at q0, but its graph curves nearly coincide
with the geometry-only control. The graph is supplying the apparent information, not Shape.
Legacy Density retains some independent local signal and gains 9--11 paired q0 points, but its
accumulated path loses 1.8 median paired points at q5 and the more W1-dominated paths lose about
7.6--7.9 points. Bootstrap intervals for the mean q5 loss exclude zero for every Density graph
family.

The shortest-path construction therefore does not improve the desired q0/q5 balance. Direct Work
Scale remains the strongest overall Work predictor. Direct legacy Density remains complementary
but should not be graph-transformed.

All systems completed. The median runtime was 50.4 seconds per system. A total of 120 individual
low-k candidate graphs were disconnected and excluded automatically; every system retained valid
connected candidates.

## Command

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-work-graph --workers 6
```

Outputs are stored in
`outputs/analysis/pairwise_geometry/checkpoint27_work_graph`, including the recovery plot,
per-system results, selected parameters, graph audits, paired tail comparisons, and resumable
per-system parts.
