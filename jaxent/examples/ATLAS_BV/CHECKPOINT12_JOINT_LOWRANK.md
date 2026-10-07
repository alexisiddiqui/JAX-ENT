# Checkpoint 12 — A-only low-rank joint log-PF modes

Completed on 2026-09-01 for all 111 systems and all six ordered A-fit/B-calibrate/C-test replica
assignments. PCA is fitted to frame-level raw log-PF profiles from replica A only. Candidate retained
variance levels of 80%, 90%, 95% and 99% are selected by endpoint-frame-disjoint validation within
A, separately for ridge-Gaussian and exact kNN-mixture likelihoods. B conformalizes uncertainty and
untouched C supplies every reported result.

## Primary common-support W1-q5 result

| Model | Marginal recovery | Mondrian coverage | Interval-score change vs raw |
|---|---:|---:|---:|
| Raw per-residue log-PF Gaussian | **83.09%** | **70.12%** | reference |
| Low-rank log-PF Gaussian | 81.71% | 51.53% | +0.397 (worse) |
| Low-rank log-PF kNN mixture | 49.68% | 52.44% | +0.432 (worse) |

The low-rank Gaussian has a paired median recovery effect of `-2.32` percentage points (95%
system-bootstrap CI `-3.04` to `-1.14`). The low-rank kNN mixture loses `33.49` points (CI `-34.90`
to `-31.82`). Neither approaches the declared contact-community trigger of at least `+2` points
with a confidence interval above zero and no interval-score deterioration.

All 2,664 target/model assignments used valid frame-disjoint validation. For W1, A most often
selects the 80% representation: 282/666 Gaussian assignments and 254/666 kNN assignments. Median
component counts range from 16–17 at 80% variance to 56–57 at 99%; the failure is therefore not
caused solely by an extremely small latent dimension.

## Interpretation and gate decision

Which residues change is more transferable than which A-specific covariance mode changes. PCA
rotates the residue profile into modes estimated from a single replica; those modes and their
structural meaning do not transfer well enough to B/C. The raw per-residue Gaussian retains both
better probability-mass recovery and better interval calibration.

The contact-community trigger is **closed**. Per the declared hierarchy, contact communities are
not run: adding a topology-specific aggregation after the more general low-rank dependence model
loses information would be an unplanned search rather than a justified follow-up.

The representation ladder is now exhausted. Proceed to the contingent per-system BV coefficient
refit, retaining raw per-residue log-PF ridge Gaussian as the downstream model.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-joint --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-opening-joint-report
```

Outputs are under `outputs/analysis/pairwise_geometry/checkpoint12_joint_lowrank/`.
