# Checkpoint 39: sparse within-system population extrapolation

## Question

If the relative populations of only a few conformational regions are known, can
direct PF or Work geometry predict the populations of other regions in the same
protein? How many population labels are required, and does the answer change for
literal extrapolation beyond the range of the known labels?

Checkpoint 38 showed that a pooled alpha loses relatively little recovery for
direct PF and most direct Work predictors, while local-variance alpha is much more
system-dependent. That was a cross-system scale test, not population
extrapolation. The current checkpoint turns the pairwise distance model into an
anchored population estimator and measures a sparse-label learning curve.

## Primary local-neighbourhood experiment

The analysis reuses the corrected Bradshaw-0.1 graph-audit artifacts for the exact
24-system checkpoint-38 cohort: 256 post-equilibration frames from each of three
replicas. It does not featurise trajectories again.

For each system, 64 structural landmarks are chosen by deterministic maximin
sampling in pooled structural-W1 space. Selection uses neither PF nor population.
At each landmark and in each replica, a structural-W1 Gaussian kernel defines:

- local occupancy, reported as log kernel mass;
- a kernel-weighted PF/Work neighbourhood descriptor;
- effective neighbour count as a support diagnostic.

The bandwidth is the median replica-A tenth-neighbour distance, matching the
existing KDE population target. Direct Work Scale/Shape/Density/Opt/Fitting/
Magnitude and PF L1/L2 are tested. Their checkpoint-36 variance-magnitude forms
reuse the A-fit/B-selected neighbourhood size and shrinkage. Rg is a compactness
control, structural W1 is an oracle, and label-mean, nearest-label and shuffled
geometries are negative controls.

For known regions `S`, alpha is fitted only from known-known pairs:

```text
alpha = sum[d(i,j) * abs(log p(i) - log p(j))] / sum[d(i,j)^2]
```

For an unseen region `u`, the predicted log occupancy is the exact scalar
least-squares solution whose absolute differences from the known log occupancies
best match `alpha * d(u,i)`. This supplies the direction that the original
absolute pairwise model lacks. Equal-loss solutions use the one nearest the median
known log occupancy; numerical degeneracies are retained and flagged. A signed
Work-Scale regression is reported separately to show whether discarding direction
is the limiting factor.

Two validation modes are kept separate:

1. `within_C`: reveal 2, 4, 8, 16 or 32 replica-C labels and predict the other C
   landmarks;
2. `AB_to_C`: pool the same selected labels from replicas A and B, then predict C
   without using a C population label.

The primary subsets are 100 deterministic, nested random orders shared by every
metric. Metric-space maximin selection is a secondary active-selection result.
Held-out regions are split into interpolation and value extrapolation according to
whether their true C log occupancy lies inside or outside the known-label range.

### What is scored

Alpha is fitted by least squares, but **MAE is not the scientific endpoint**. The
primary score is the same MD distribution-recovery quantity used in the preceding
population analyses. Among the held-out regions, the analysis forms every
absolute pairwise log-population change,

```text
abs(log p(i) - log p(j)).
```

Predicted and observed changes are histogrammed using bin edges fixed from the
source MD distribution. If `JSD` is the Jensen-Shannon divergence between those
histograms, then

```text
distribution recovery = 1 - sqrt(JSD).
```

A value of 1 (100%) means identical distributions of population changes. Lower
values mean that the predicted landscape has the wrong spread or shape. NMAE is
retained only as a diagnostic and is not used to decide how much data is needed.

Recovery alone is insufficient for local extrapolation: permuting predictions
among neighbourhoods can leave their distribution nearly unchanged. Therefore
the report always pairs recovery with held-out Spearman correlation. Recovery
asks, “Did we reproduce the range of population changes?” Spearman asks, “Did we
assign high and low populations to the correct neighbourhoods?” A useful local
model needs both.

## Structural-cluster sensitivity

The existing pooled structural-W1 average-linkage partitions at `K=5` and `K=10`
provide an intuitive population-mass readout. These are deliberately called
**structural subdivisions**, not metastable basins: the earlier strict census found
no systems with three robust basins, and forced clusters are often sparsely sampled
per replica.

Replica populations use a Jeffreys pseudocount of 0.5 so zero-count clusters remain
visible. The analysis reports MD distribution recovery, population total variation,
rank correlation, top-unseen-cluster recovery, minimum raw C count and the number
of zero-count C clusters.

## “How much data?” rule

Repetitions are averaged within a system before the 24 systems are bootstrapped.
Two thresholds are reported for each predictor and validation mode.

`required_labels` is the first **detectable-signal** count that:

- has higher recovery than the label-mean and shuffled-geometry controls, with
  paired 95% bootstrap intervals above zero;
- has higher Spearman correlation than shuffled geometry by the same criterion;
- retains at least 90% of the improvement achieved with 32 labels at that and all
  larger label counts.

`moderate_localization_labels` additionally requires median held-out Spearman
correlation of at least 0.5. This second threshold is the more useful answer to
“how much data is needed to predict which neighbourhood is populated?” The 0.5
cutoff is an explicit descriptive benchmark, not a physical phase transition.

The two-label result is always reported explicitly, including interpolation and
literal extrapolation. Failure to meet the rule is reported as “not reached,” not
silently replaced by the largest tested count.

## Run and outputs

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-sparse-population-extrapolation --workers 4
```

For a smoke run:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-sparse-population-extrapolation \
  --limit 1 --repeats 2 --landmarks 8 --label-counts 2,4 --workers 1 \
  --output /tmp/atlas-cp39-smoke
```

Outputs are written beneath
`outputs/analysis/pairwise_geometry/checkpoint39_sparse_population_extrapolation/`:

- `local_extrapolation_results.parquet`
- `cluster_extrapolation_results.parquet`
- `local_two_label_predictions.parquet`
- `data_requirements.csv`
- `sparse_label_learning_curves.png`
- `error_vs_label_distance.png`
- `cluster_population_recovery.png`
- `two_label_examples.png`
- `checkpoint39_report.yaml`

The 24-system pilot does not automatically authorize a 111-system expansion.

## Pilot result

The full 24-system run completed with 64 landmarks and 100 nested random label
orders per system. Repetitions were averaged within each protein before the
cohort bootstrap.

The main result is that **two labels reproduce much of the distribution but do
not localize the populations**. For the primary within-replica-C test:

| predictor | known labels | median recovery | median Spearman |
|---|---:|---:|---:|
| PF L1 direct | 2 | 70.9% | 0.007 |
| PF L1 direct | 8 | 70.2% | 0.443 |
| PF L1 direct | 16 | 71.5% | 0.630 |
| PF L1 direct | 32 | 72.2% | 0.698 |
| Work Opt direct | 2 | 71.2% | -0.089 |
| Work Opt direct | 8 | 69.3% | 0.426 |
| Work Opt direct | 16 | 70.8% | 0.594 |
| Work Opt direct | 32 | 71.3% | 0.659 |

The label-mean control has only 37.9% median recovery, so the two-label predictors
clearly learn the approximate *amount* of population variation. But their
near-zero rank correlations show that they do not know *where* that variation
belongs. This is exactly why the recovery and assignment panels must be read
together.

### How many labels are needed?

Within replica C, direct PF L1/L2 and most direct Work metrics first show a
statistically detectable recovery-and-ranking signal at **4 labels**. That should
not be read as accurate prediction: median rank correlation at four labels is only
0.213 for PF L1 and 0.150 for Work Opt. Both first cross the explicit moderate
localization benchmark at **16 of 64 labels**, reaching median Spearman 0.630 and
0.594. Thirty-two labels improve these to 0.698 and 0.659. Thus roughly one quarter
of the landmark populations is needed for moderately useful within-replica
localization, and half gives the strongest tested ordering.

Work Scale is weaker: it has a detectable positive signal with two labels, but
never reaches median Spearman 0.5 (0.377 at 32 labels). “Detectable” is therefore
not synonymous with “enough.”

Literal value extrapolation remains the harder problem. At 32 labels, PF L1 and
Work Opt recover about 46% of the out-of-range change distribution with median
Spearman only 0.34 and 0.29. For held-out regions inside the known population
range, recovery is about 74% and Spearman about 0.69. The evidence therefore
supports interpolation across structurally sampled neighbourhoods much more than
prediction beyond the observed population range.

The A/B-to-C result is negative for local assignment. At 32 labels, PF L1 and Work
Opt still show 63.1% and 61.7% recovery, but their median Spearman correlations
are -0.395 and -0.293. Work Scale is near zero (0.063). The metrics carry a
replica-transferable estimate of the *distribution width*, but not a transferable
map from structure to the population of each local region.

The forced-cluster sensitivity is consistent with the local result. With two
known clusters, direct PF/Work population TV is worse than the label-mean control
at both `K=5` and `K=10`, and rank correlations are negative. Sparse or zero-count
replica-C clusters make this an illustrative secondary result rather than evidence
about metastable states.

### Conclusion

Checkpoint 38's cross-system result and checkpoint 39's within-system result are
not contradictory. Direct PF/Work geometry contains enough information to recover
the broad distribution of population changes from only two anchors. It does not
follow that two anchors identify which local region has which population.

The practical answer is:

- **2 labels:** broad distribution recovery, essentially no localization;
- **4 labels:** first statistically detectable within-replica local signal;
- **16 labels:** moderate within-replica localization for PF L1 and Work Opt;
- **32 labels:** stronger interpolation, but out-of-range extrapolation remains
  weak;
- **A/B-to-C:** distribution shape partly transfers, local assignments do not.

MAE is not used for these conclusions. Variance-magnitude results remain reported,
but they do not change the central distinction between recovering a distribution
and predicting the correct local populations.
