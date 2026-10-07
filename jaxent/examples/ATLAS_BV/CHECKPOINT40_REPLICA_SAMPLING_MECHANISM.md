# Checkpoint 40: replica sampling mechanism

## Question

Checkpoint 39 found that direct PF L1 and Work Opt recover much of the
distribution of population changes but assign those changes to the wrong local
neighbourhoods in A/B-to-C transfer. This checkpoint tests whether that failure
comes from finite sampling, temporal nonstationarity, or genuinely different
replica-specific sub-basins.

The analysis is restricted to the same 24 systems, 64 structural-W1 landmarks,
100 nested population-label orders, and Bradshaw-0.1 features used by checkpoints
38 and 39. Earlier checkpoint outputs are read-only inputs.

## Prespecified sequence

### 1. Coordinate-decomposed overlap

For every system, pooled A+B is compared with C using histogram
Jensen-Shannon divergence in bits. Histogram edges are fixed from all A+B+C
frames. Separate axes are reported for reference-aligned C-alpha RMSD, Rg,
global PF, the mean per-residue PF-profile divergence, and the mean per-residue
Work-Opt-profile divergence. A-B, A-C and B-C comparisons are retained as
diagnostics.

The primary response is the 32-label assignment loss

```text
rho(within C) - rho(A/B to C)
```

for direct PF L1 and Work Opt. Sign-flip magnitude and the assignment-loss area
under the complete label curve are sensitivities. System bootstrap intervals,
system-label permutations and within-family FDR correction are reported.

### 2. Temporal half transfer

Replica C is split into its first and second 128 ordered post-equilibration
frames. C1-to-C2 is primary and C2-to-C1 is a prespecified reverse-direction
sensitivity analysis. Each direction chooses landmarks from its source half
only. Alpha and anchor populations are fitted only in the source half; target
half populations are used only for scoring.

Matched within-half controls distinguish failure caused simply by having 128
frames from failure to transfer across time. Recovery measures whether the
population-change distribution is reproduced, while held-out Spearman rho tests
whether the changes are assigned to the correct neighbourhoods.

### 3. Effective sample size

The existing initial-positive-autocorrelation estimator is applied separately
to RMSD, Rg, global PF, and mean Work-Opt magnitude in each replica:

```text
N_eff = number of frames / integrated autocorrelation time.
```

Non-overlapping batch-means estimates at 8, 16 and 32 frames are sensitivities.
The checkpoint tests both minimum effective support and the absolute log
imbalance between C and the geometric mean of A/B. Effects are compared with
overlap diagnostics using bootstrap Spearman coefficients and leave-one-system-
out error.

Checkpoint 38 is then refitted with predictor-family-matched sampling
covariates. PF/Work predictors use PF or Work-Opt support, Rg uses Rg support,
and RMSD/W1 use RMSD support. Models contain predictor fixed effects and test
heterogeneity alone, effective support alone, both together, and overlap plus
effective-support imbalance.

### 4. Pooled-feature intervention

The original A/B-to-C result is paired with two common-feature estimators:

1. A+B-pooled descriptors used for both fit and query;
2. A+B+C-pooled descriptors used for both fit and query.

Direct predictors pool their kernel-weighted PF/Work descriptors. Variance-
magnitude predictors recompute local variance on the pooled frames using the
checkpoint-36 neighbourhood size and shrinkage. Alpha still uses only A/B
population labels. C population values remain completely held out until scoring.

The A+B+C intervention is therefore transductive: it uses unlabeled C features,
but never C population labels. Rescue requires a positive paired 95% interval
for the Spearman improvement and a nonnegative cohort median. Median rho at least
0.5 is reported as strong rescue.

Direct-feature rescue demonstrates descriptor-estimation mismatch. Only rescue
of variance-magnitude predictors directly supports the finite-variance-estimation
hypothesis; the two claims are kept separate.

## Run

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-replica-sampling-mechanism \
  --phase all --workers 4
```

Each component is independently resumable with `--phase overlap`, `temporal`,
`neff`, `pool`, or `report`. A smoke run is:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-replica-sampling-mechanism \
  --phase all --limit 1 --repeats 2 --landmarks 8 --label-counts 2,4 \
  --bootstrap-samples 100 --permutations 100 --workers 1 \
  --output /tmp/atlas-cp40-smoke
```

Primary outputs beneath
`outputs/analysis/pairwise_geometry/checkpoint40_replica_sampling_mechanism/`
include combined overlap, effective-sample-size, temporal and pooled-feature
parquets; mechanism and checkpoint-38 regression tables; temporal and pooled
summaries; four figures; and `checkpoint40_report.yaml`.

The report must not describe a pooled direct-feature rescue as confirmation of
the harmonic variance equation, and failure to rescue is evidence consistent
with—not proof of—a multimodal or state-specific theoretical breakdown.

## 24-system result

The full run completed with 64 landmarks and 100 nested label selections per
system. Its strongest result comes from the temporal control: **local assignment
does not transfer reliably even between the two halves of replica C.** At 32
labels:

| predictor | within C1 | C1 to C2 | within C2 | C2 to C1 |
|---|---:|---:|---:|---:|
| PF L1 direct | 0.752 | -0.104 | 0.741 | -0.220 |
| Work Opt direct | 0.718 | 0.144 | 0.757 | -0.203 |

The paired temporal-minus-within-half confidence intervals are entirely below
zero in both directions and for both predictors. Recovery also falls to about
50--55%. Thus the A/B-to-C failure cannot be attributed only to independent
replicas occupying different states. The structure-to-population map changes
substantially across time within one nominal replica, while fitting and testing
against the same half produces the high correlations seen in checkpoint 39.

### Overlap and effective sample size

RMSD and Rg divergence do not predict the 32-label assignment loss. Work Opt has
no FDR-significant association with any overlap or effective-sample-size axis.
For PF L1, profile divergences are significant but have the opposite sign from
the proposed mechanism: larger PF/Work-Opt profile divergence is associated with
*smaller*, not larger, assignment loss or sign-flip magnitude. This specificity
therefore refutes the simple claim that histogram separation causes the flip.

Minimum PF effective support also trends in the opposite direction: systems with
more effective samples tend to have larger PF L1 sign flips. The checkpoint-38
refit similarly gives a positive coefficient for fit-replica log effective size
against alpha mismatch (`estimate=0.390`, nominal `p=0.020`, FDR `q=0.072`). If
finite-sample imprecision were the cause, the expected coefficient would be
negative. Neither effective-sample-size imbalance nor matched overlap explains
the checkpoint-38 recovery gap after correction.

These results do not say the trajectories are converged. They say the proposed
scalar measures of convergence do not explain which systems fail transfer, and
the observed directions are inconsistent with “too few independent frames” as
the primary mechanism.

### Pooled-feature intervention

Using one common descriptor geometry for both fit and query changes the sign of
the direct-metric assignment:

| predictor | original A/B to C | A+B common | A+B+C common |
|---|---:|---:|---:|
| PF L1 direct | -0.395 | 0.346 | 0.381 |
| Work Opt direct | -0.293 | 0.342 | 0.335 |

Relative to the original estimator, A+B+C pooling meets the prespecified rescue
criterion for both predictors. It is a partial, not strong, rescue because neither
median reaches 0.5. Distribution recovery decreases slightly, from 63.1% to
59.8% for PF L1 and from 61.7% to 58.8% for Work Opt: reference-locking improves
where populations are assigned while modestly worsening the predicted spread.

Crucially, adding unlabeled C frames supplies no significant increment beyond the
A+B-only common coordinate. The paired median A+B+C-minus-A+B changes are 0.029
for PF L1 (`95% CI -0.006 to 0.071`) and 0.046 for Work Opt (`-0.021 to 0.114`).
Adding C features does not explain the rescue. However, this intervention uses
the same descriptor geometry for both sides; the A+B arm is specifically a
source-copy control and does not transport C's independently estimated feature
geometry. It therefore demonstrates that cross-descriptor comparison is
involved, but does not by itself identify a replica-specific coordinate origin.
Checkpoint 41 separates fixed reference, target-self distance transport, and
nonlinear aggregation order.

Variance-magnitude predictors do not show the same rescue: their median rhos are
already weakly positive in the original transfer and remain only about 0.24--0.28
after A+B+C pooling. The classic finite-variance-estimation mechanism is therefore
not confirmed.

### Conclusion

The four tests reject a simple “A/B and C merely need more independent samples”
explanation. Population assignment is time-window dependent within C; RMSD/Rg,
profile overlap, and effective sample size do not explain failures in the expected
direction; and extra C frames do not improve on the A+B source-copy geometry.
The precise corrective mechanism is resolved in checkpoint 41: Work Opt becomes
transferable when raw PF is neighbourhood-averaged before the nonlinear Work
transformation, whereas a fixed external origin supplies no significant
same-order improvement.

The direct PF/Work sign flip is primarily an estimator-coordinate problem:
independently estimated replica descriptors are not on a stable common local
coordinate. Reference-locking removes the negative ordering and recovers a modest
shared population ranking, but it does not restore strong transfer and does not
validate the harmonic variance-to-population mechanism.
