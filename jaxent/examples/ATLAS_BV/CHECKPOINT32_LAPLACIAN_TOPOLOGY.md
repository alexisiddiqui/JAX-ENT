# Checkpoint 32: Laplacian topology selection

## Protocol

This development checkpoint separates metric choice, neighbour selection, edge weighting, and
generic shrinkage. PF-L2, Work Scale, legacy-Zq and cached PyRosetta `ref2015` each receive:

- self-tuned symmetric-union kNN at `k = 3, 5, 10, 20, 40`;
- uniform-edge kNN on the identical neighbour sets;
- weighted complete RBF graphs with bandwidths at the 2%, 4%, 8% and 16% positive-distance
  quantiles.

A uniform complete graph is the geometry-free control. Its regularizer is evaluated through the
exact linear-cost variance identity rather than materializing all pairs. Weighted complete graphs
retain all pairs and remain quadratic. Mutual kNN exists in the core API but is not part of this
preregistered comparison.

Replica A supplies graph eligibility: connectedness, positive structural-edge gain, and basin-edge
purity above rewiring. Replica B selects one candidate using the smaller of its median relative
smooth- and basin-recovery gains over MaxEnt. Among candidates within one standard error of the
best, ties prefer rewiring sensitivity, basin purity, then fewer edges. Only that locked candidate
is evaluated on replica C. Basin challenges use exact probability-mass transfer; ESS is diagnostic
only.

## Development result

The 24-system development run completed. To make the exhaustive comparison tractable, all 56
candidates were first screened at 64 frames and 100 optimizer steps. The best sparse and dense
candidate for each metric was then rerun at 256 frames and 500 steps. This retained eight finalists
before the preregistered replica-B selection.

The screen favored broad smoothing: sparse finalists reached `k=40` for all metrics (legacy-Zq
selected uniform rather than self-tuned edges), and dense finalists used the 16% bandwidth except
PyRosetta at 8%. At full resolution, the one-standard-error rule selected the Work Scale weighted
all-pairs RBF graph at the 16% positive-distance quantile. Its conservative replica-B score was
close to PF-L2, but it had the largest advantage over rewiring among the near-optimal candidates.

The locked graph passed replica C:

| challenge | mean TV gain over MaxEnt | 95% CI | median HDX-MSE change |
|---|---:|---:|---:|
| smooth | 0.0456 | 0.0390 to 0.0526 | -48.8% |
| basin | 0.00852 | 0.00414 to 0.0137 | -0.04% |

Its median combined physical gain was 0.0184, versus 0.00423 after edge-weight rewiring and 0.00731
for the geometry-free uniform complete graph. Thus distance-dependent Work Scale weighting adds
value beyond generic complete-graph shrinkage. Replica B selected all-pairs over its kNN finalist;
replica C was deliberately used only for the locked all-pairs candidate. The development gate passes.

This is not confirmatory evidence: the same 24 systems informed screening and topology selection.
The metric, topology, bandwidth rule, strength grid and gates must now be locked and evaluated on
the remaining 87 ATLAS systems before ISO use.

## Confirmation result

The locked Work Scale all-pairs configuration was evaluated on the remaining 87 systems, excluding
all 24 development systems. Its aggregate confirmation gate passed. Smooth-bias TV gain over
MaxEnt was 0.0517 (95% CI 0.0478–0.0557) with a 54.9% median HDX-MSE reduction. Basin-bias gain was
0.00493 (95% CI 0.00306–0.00702) with a 2.49% median HDX-MSE reduction. The combined selected gain
was 0.0124, versus 0.00320 after rewiring and 0.00622 for uniform all-pairs.

The preregistered protein-size stability requirement did not pass. System-level basin gains by
length quartile were:

| size quartile | mean gain | 95% CI | positive systems |
|---|---:|---:|---:|
| Q1, smallest | 0.0125 | 0.00337–0.0238 | 79.2% |
| Q2 | 0.00569 | 0.00052–0.0121 | 50.0% |
| Q3 | 0.00387 | 0.00055–0.00832 | 76.2% |
| Q4, largest | 0.00054 | -0.00005–0.00150 | 54.5% |

Smooth gains remained positive in every size quartile, but the largest-protein basin interval
crossed zero. Consequently the aggregate method is confirmed, but robust basin benefit is not;
ISO integration remains unauthorized pending resolution of this size dependence.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-topology \
  --phase all --smoke --limit 1 --permutations 2 --steps 2 \
  --frame-cap 32 --strengths 0,0.01
```

The completed development comparison used:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-topology \
  --phase all --workers 6
```

The locked confirmation used:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-laplacian-topology --confirm --workers 6
```
