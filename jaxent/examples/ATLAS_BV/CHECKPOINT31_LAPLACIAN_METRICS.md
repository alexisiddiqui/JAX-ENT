# Checkpoint 31: legacy-Zq and PyRosetta Laplacian metrics

## Question and protocol

This 24-system pilot tests legacy Work Density (`Pi=Zq`) and cached PyRosetta
`ref2015` total score as standalone frame-graph coordinates before any hybrid graph is
considered. PF-L2 and Work Scale are matched controls. Legacy-Zq uses mean residue-wise
L1 distance; each other candidate uses its natural Euclidean/absolute distance. Graphs
are symmetric self-tuning kNN graphs with `k in {5,10,20,50}`.

Replica A establishes eligibility, replica B selects k independently for each metric,
and replica C is held out. Reweighting compares each graph with MaxEnt, no regularizer,
the structural-W1 oracle, and a topology-preserving node-relabelled control. Continuous
smooth and shuffled challenges use ESS only to calibrate their severity. Basin challenges
instead overpopulate the smallest structural basin by exactly 0.10, 0.25, or 0.40 total
probability mass. ESS is recorded but never enters a loss or gate.

PyRosetta scores are read from the already validated 333-trajectory cache. REU are used
only to order neighbours; no conversion to physical energy or fitted scale is asserted.
Legacy `Zq` remains an empirical non-normalized transformation and is not called entropy.

## Pilot result

All 24 systems completed. The selected neighbourhoods and held-out graph audits were:

| metric | k | density gain | structural-edge gain | basin-edge purity |
|---|---:|---:|---:|---:|
| PF-L2 | 5 | 0.645 | 0.603 | 0.878 |
| Work Scale | 5 | 0.517 | 0.446 | 0.822 |
| legacy-Zq | 5 | 0.243 | 0.191 | 0.721 |
| PyRosetta ref2015 | 20 | 0.174 | 0.060 | 0.618 |

All graph-audit gates passed, although legacy-Zq was significant against rewiring in
17/24 systems (70.8%, just above the 70% threshold) and PyRosetta in 19/24.

Held-out weight-TV gains over MaxEnt were:

| metric | smooth gain (95% CI) | basin gain (95% CI) | physical / rewired median gain |
|---|---:|---:|---:|
| PF-L2 | 0.0415 (0.0359, 0.0473) | 0.00825 (0.00485, 0.0121) | 0.0195 / 0.00247 |
| Work Scale | 0.0297 (0.0246, 0.0353) | 0.00564 (0.00212, 0.00972) | 0.00819 / 0.00072 |
| legacy-Zq | 0.0120 (0.00911, 0.0150) | 0.00205 (0.00040, 0.00391) | 0.00219 / 0.00009 |
| PyRosetta ref2015 | 0.0203 (0.0168, 0.0242) | 0.00502 (0.00194, 0.00887) | 0.00703 / 0.00577 |

All candidates met HDX-MSE non-inferiority and the literal rewired-control criterion.
The corrected exact-mass basin benchmark also makes PF-L2 and Work Scale pass; this does
not overturn checkpoint 30, whose different ESS-calibrated binary challenge had unequal
achieved severities.

The candidates are not equally compelling. PF-L2 remains the strongest and most
topology-specific graph. Legacy-Zq adds a small but clearly graph-specific effect.
PyRosetta has a larger raw gain than legacy-Zq, but rewiring retains most of it, consistent
with its weak structural-edge gain and purity. Therefore checkpoint 31 authorizes a full
comparison but does not authorize ISO use or a hybrid graph.

## Commands and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-metrics --phase all --workers 6
```

Results and per-system resumable parts are under
`outputs/analysis/pairwise_geometry/checkpoint31_laplacian_metrics/pilot/`. The formal
decision is in `checkpoint31_report.yaml`.
