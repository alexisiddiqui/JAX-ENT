# Checkpoint 21: PF-W1 and variance-scaled information-distance pilot

This checkpoint asks whether two alternative reductions of the fixed-BV residue profile improve the
MD population-change target. The target is unchanged from checkpoint 17: a rank-10 KDE is built only
from the MD frame-geometry W1 matrix, and the response for a frame pair is the magnitude of its log
KDE-density change. The predictors do not receive a KDE and the BV coefficients remain fixed at
`bc=0.35`, `bh=2.0`.

The experiment uses checkpoint 19's 12 size-stratified systems and deterministic pair sample. Replica
1 estimates every model coefficient and every residue variance, replica 2 selects the variance
shrinkage, and replica 3 is evaluated once. Recovery is `100*(1-sqrt(JSD))` between the predicted and
target value distributions. The x-axis uses the global **structural** W1 bands in angstroms; PF-W1 is
a dimensionless predictor. Pairwise structural W1 itself is included as a geometry-only positive
control. It is not an independent scientific predictor because the target KDE uses the same
geometry, but it tests whether density differences are trivially determined by pair separation.
Periodic backbone dRMSD is a second geometry-only control. It is the RMS over all topology-defined
phi/psi angle differences after wrapping each difference into `[-pi, pi]`; it changes only the
pairwise predictor and does not define a new KDE.

The circular z-score variants estimate each angle's circular mean and mean squared wrapped
deviation using replica 1 only. The regularized standardized differences are evaluated both as an
RMS distance and as their mean square; replica 2 selects the same shrinkage grid used for the PF
information metrics.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-pf-information-pilot
```

## Metrics

For a fixed-BV log-PF profile `z[:, frame]`, raw PF-W1 is the mean absolute difference between the
two sorted residue profiles. Mean-centred PF-W1 first subtracts each frame's residue mean. Both are
ordinary equal-weight 1D Wasserstein distances and deliberately ignore residue identity.

The information arm retains residue identity. Replica 1 supplies the per-residue variance `v_r`,
regularized as

```
v_tilde_r = v_r + lambda * median(v_r > 0),  lambda in {0.001, 0.01, 0.1}.
```

The two predictors are the mean squared standardized change and its square root. These are a
diagonal Gaussian quadratic cost and Mahalanobis distance, not empirical PF KLD/JSD. Each scalar
predictor uses a nonnegative, zero-intercept alpha fitted on replica 1.

## Result

No new metric passed the predeclared full-run gate. Median recovery by structural-W1 band was:

| model | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Work Scale | 91.22 | 86.87 | 80.05 | 67.10 | 66.26 | 51.62 |
| structural W1 control | 72.18 | 52.53 | 47.60 | 44.51 | 40.79 | 34.32 |
| periodic backbone dRMSD control | 23.37 | 33.95 | 35.11 | 36.87 | 32.52 | 31.30 |
| circular z-dRMSD control | 20.69 | 28.39 | 33.48 | 33.05 | 31.29 | 30.93 |
| circular z-quadratic control | 21.85 | 33.12 | 40.35 | 39.06 | 36.58 | 36.51 |
| raw PF-W1 | 31.84 | 47.71 | 50.26 | 54.95 | 50.52 | 52.56 |
| centred PF-W1 | 25.20 | 36.60 | 45.28 | 41.43 | 37.01 | 40.87 |
| variance-scaled quadratic | 34.14 | 46.70 | 53.14 | 53.59 | 51.52 | 54.29 |
| variance-scaled Mahalanobis | 24.18 | 34.48 | 42.02 | 38.99 | 38.89 | 39.60 |

Only six systems contribute to q5. Although the unpaired q5 median of the quadratic information
score is 54.29% versus 51.62% for Work Scale, the within-system comparison is negative: median
change `-3.88` percentage points, only `2/6` systems improve by at least three points, and median MAE
worsens by `0.672`. Raw PF-W1 has a paired median q5 change of `-1.84` points and improves `0/6`
systems by three points. At q4, Work Scale exceeds every new metric by at least 11.2 paired median
points.

The direct structural-W1 control declines from 72.18% in q0 to 34.32% in q5 and loses a paired
19.24 points to Work Scale in q5. Therefore the MD KDE-density target is not a trivial monotone
function of the W1 separation of the two queried frames: it depends on each frame's complete local
neighbourhood in the trajectory.

Periodic phi/psi dRMSD is weaker still, reaching 32.52% in q4 and 31.30% in q5. Its paired q5
change from Work Scale is `-26.68` percentage points, and none of the six contributing systems
improves by at least three points. This is not a missing-angle artefact: the 12 systems retain
97.1--99.2% of their protein residues, with a median of 98.4%, after excluding termini where both
phi and psi are not defined.

Circular RMS z-scoring does not improve raw dRMSD: its paired q5 change is `-0.60` points. Retaining
the squared standardized cost is more useful, improving raw dRMSD by a paired `+5.11` points in q5
with `4/6` systems improved and a modest MAE improvement relative to raw dRMSD. Nevertheless, its
36.51% q5 recovery remains a paired `-19.58` points below Work Scale, only `1/6` systems improves by
three points over Work Scale, and median MAE against the target worsens by `0.676`. Variance scaling
therefore reveals some rigid-torsion signal but does not make backbone dRMSD competitive with the BV
scale predictor.

Raw PF-W1 is strongly correlated with Work Scale on replica 2 (median Spearman `0.900`) and has a
negative correlation with the remaining Work Scale residual (`-0.215`). Mean-centred PF-W1 removes
that dominant scale signal but supplies essentially no residual signal (`-0.005`). The information
metrics likewise have weak negative residual correlation (`-0.066`). These diagnostics do not
support a combination run.

The checkpoint therefore stops at the pilot. This result does not show that fixed BV lacks useful
population signal: Work Scale remains strong through the local and intermediate W1 bands. It shows
only that PF-W1 and diagonal variance scaling do not improve its transferable tail prediction under
this replica-isolated test.
