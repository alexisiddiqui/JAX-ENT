# OMC calibration by cluster filtering

## Frozen experimental setup

This single-system development experiment uses 1yoz_B replica 1 and up to 512 evenly
spaced source frames. Fixed-BV pseudo-uptake is computed once on the source ensemble;
its uniform average is the target. Filtering never recomputes centering or the target.
Original cluster proportions define the known population targets.

KMeans (20 initialisations, configuration seed) partitions structural W1 distance
profiles at k=2 through 8. Precomputed-distance silhouette ranks partitions. Positive
silhouette and at least 20 frames per cluster are required. The best eligible k from
each band 2–3, 4–5 and 6–8 is retained, with ties favouring smaller k.

Within each retained partition, nested seeded filtering retains 50%, 25% or 12.5%
of each cluster individually and of the two largest clusters together. Every cluster
retains at least two frames. For k=2, simultaneous thinning acts as a sample-size
control: both clusters shrink at approximately the same rate, leaving proportions
nearly unchanged apart from integer rounding. An unfiltered control is retained.
Candidate sizes vary and are recorded.

## Fitting and measurement

All residues and uptake times enter the MSE for fitting and selection; there is no
validation split. Data loss is MSE/(target variance + 1e-8). The grid comprises zero
and strengths 1e-7 through 1e2, with sigma at positive Work Scale distance quantiles
0.02, 0.04, 0.08, 0.16, 0.32 and 0.64 for each candidate. Arms are original OMC,
separately labelled normalised OMC, rewired OMC, MaxEnt and the locked prior-relative
Laplacian. This experiment preserves the loss definitions; it changes the challenge.

Adam (learning rate 0.05) starts at uniform candidate weights. Each trajectory runs
1000 steps, extended to 3000 if relative objective change over the final 250 steps
exceeds 1%. Relative change uses an absolute denominator floor of 1e-12. Unresolved
convergence remains visible in every summary. Padding frames receive exactly zero
weight, and N in the OMC energy counts only actual candidate frames.

Every fit records all-residue MSE, cluster-population TV, overall ESS and fraction,
support and plateau statistics, structural dispersion, gradient norm and convergence.
Population tables record target, initial and recovered masses, actual retained counts,
required enrichment and conditional ESS. Frame weights, source identities and frozen
labels are retained. No framewise TV against a made-up candidate truth is used.

Positive-strength heatmaps and tradeoff plots precede MSE-selected summaries. Operating
ranges report positive-strength fits within 1%, 5% and 10% of the best observed MSE,
with absolute tolerance floor 1e-8. These are descriptive; zero strength winning is
not a failure gate. Source subsets overlap the original reference: this is controlled
population-bias calibration, not independent predictive validation.

## Commands

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-cluster-filtering-omc
./jaxent/examples/ATLAS_BV/commands.sh geometry-cluster-filtering-omc --report-only
```

Outputs live in `outputs/analysis/pairwise_geometry/cluster_filtering_omc/1yoz_B/`.
Only this system is authorised for calibration; expansion follows review of these
results and a frozen hyperparameter range. No ISO or cross-system verdict is made.

## Results

The complete grid contains **5,720 fits across 26 candidates**: 21 population-biased
candidates, two unfiltered controls and three two-cluster sample-size controls. The
source has 512 frames. Silhouette retained the following partitions:

| k | silhouette | source cluster sizes |
|---|---:|---|
| 2 | 0.3486 | 190, 322 |
| 4 | 0.2277 | 50, 77, 175, 210 |

The k=6–8 partitions had minimum sizes 16, 9 and 9; none met the 20-frame criterion.
Targets, frame identities, zero weights on removed frames, normalisation, all-residue
MSE and cluster-population TV were independently checked from all saved weight arrays.
All 5,720 fits have finite metrics and valid simplexes.

### Population recovery

Across population-biased candidates, median initial population TV changed from 0.2741
to 0.1669 for k=2 and from 0.1501 to 0.1358 for k=4 under minimum-MSE OMC selection.
These describe total reweighting recovery, not an isolated OMC advantage: the
minimum-MSE comparator fits generally have very similar populations. Several
four-cluster cases achieve low MSE while retaining substantial population error.

### Distribution control at positive strength

At strength 0.1, varying bandwidth gives visibly different tradeoffs:

| candidate | q change | ESS fraction | population TV | all-residue MSE |
|---|---|---|---|---|
| k=2, retain 12.5% of cluster 1 | 0.02 → 0.64 | 0.447 → 0.597 | 0.307 → 0.359 | 1.09e-4 → 2.72e-4 |
| k=4, retain 12.5% of clusters 3 and 2 | 0.02 → 0.32 | 0.586 → 0.721 | 0.464 → 0.447 | 1.26e-4 → 1.96e-4 |

All fits in these examples pass the final-window convergence criterion. The second
example improves population error while spreading weights more broadly, at an MSE
cost; the first spreads weights while worsening both recovery and MSE. Thus bandwidth
does control distributions, but the useful tradeoff depends on the cluster challenge.
Neither example establishes recovery of the true populations.

Under the tighter 5%-of-best-MSE envelope (absolute floor 1e-8), the positive-OMC ESS
max/min ratio across bandwidths and strengths has median 1.045 and maximum 1.077 over
the 21 biased candidates. Large diversity changes generally require a larger MSE
allowance here. This envelope is descriptive and does not select using population truth.

### Hyperparameter ranges for the next development pass

Retain all six bandwidth quantiles. Refine primary/re-wired strengths within
**1e-5 to 1e-1**, and the separately labelled normalised arm within **1e-6 to 1e-2**,
using values 1 and 3 per decade plus a zero-strength reference. The exact proposed
grids are saved in `proposed_hyperparameter_ranges.yaml`. These post-calibration
ranges cover the observed onset of distribution changes and the increasing MSE cost;
they are not confirmed optimal settings or an equivalence between the two objectives.

Convergence remains a development issue: 1,975 trajectories stopped at 1,000 steps;
2,026 more passed after extension; 1,719 (30.1%) still exceeded 1% relative objective
change at 3,000. Of the 21 MSE-selected OMC fits for biased candidates, 10 passed.
Near-zero-error controls also inflate relative-change flags; absolute errors and
gradients are retained for interpretation. No cohort expansion was run.

### Viewing the evidence

Open `outputs/analysis/pairwise_geometry/cluster_filtering_omc/1yoz_B/index.html` for
the complete plots and tables. `population_recovery.png` compares target, candidate
and fitted populations; `bandwidth_control.png` displays fixed-positive-strength
curves; each candidate has a six-panel heatmap. Downloadable tables include
`selected_populations.csv` and `recovery_summary.csv`; all fitted weights and frozen
frame identities are retained in `parts/`.
