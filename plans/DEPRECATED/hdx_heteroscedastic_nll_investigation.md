# Investigation: matching target and predicted conformational PF variance

Status: **corrected residue and expected-moment peptide experiments complete**
Owner: (unassigned)
Created: 2026-07-13
Corrected: 2026-07-14

## Conclusion currently justified

The earlier heteroscedastic-NLL investigation did not support its stated NO-GO conclusion.
It mixed curve covariance, conformational spread, and observation variance, and its
misfit-correlation decision gate was circular for targets whose residual was injected noise.

The corrected question is:

> Does explicitly matching the target and predicted **conformational log-PF conditional
> variance** improve recovery of the known ensemble and rejection of the TRI decoy state,
> without degrading held-out uptake-mean fits or collapsing the ensemble?

No replicate uncertainty is involved. No production loss has been added.

## Locked interpretation

There are two distinct covariance objects:

1. `W_curve`: the existing trace-normalised precision used by covariance-MSE. It is retained
   as fixed **curve geometry** for fitting mean uptake and is not called observation noise.
2. `C_PF(w)`: weighted conformational covariance of residue log-PF over structural frames.
   It supplies a second-moment constraint and is never placed in the residual denominator.

The production BV forward model remains **average-first**:

    z_bar(w) = sum_f w_f z_f
    uptake_mean(w, t) = 1 - exp(-k_int * t / exp(z_bar(w)))

Transform-first/average-after uptake is used only to construct a deliberately mismatched
synthetic target sensitivity. It is never used for prediction or reweighting.

## PF covariance and conditional variance

For frame log-PFs `z_f` and normalised ensemble probabilities `w`:

    C_PF(w) = sum_f w_f (z_f - z_bar(w)) (z_f - z_bar(w))^T

This is a weighted **population** covariance: the frames define a discrete ensemble, so
there is no Bessel correction. It produces one residue-by-residue matrix. There is no need
to create or stack covariance matrices over uptake timepoints.

Target and prediction are constructed symmetrically:

- `C_PF,target`: ISO_BI frame covariance at the known 40:60 open/closed weights.
- `C_PF,pred(w)`: BI or TRI frame covariance at the fitted weights.

Both receive the same differentiable isotropic shrinkage:

    C_tilde = (1-alpha) C + alpha * trace(C)/d * I + epsilon * I
    epsilon = 1e-8 * trace(C)/d + 1e-12

The matched profile is the conditional variance

    v_cond,i = 1 / diag(C_tilde^-1)_i

computed with a Cholesky solve rather than an explicit inverse. `diag(C^-1)` itself is a
conditional **precision**, not a variance. Under a multivariate-Gaussian description,
`v_cond,i` is the residual variance of residue `i` after linear conditioning on the others.

Only this profile is optimised. It does not uniquely identify the full off-diagonal
covariance, so full-matrix Frobenius error and off-diagonal correlation are diagnostics.

For future peptide-level matching, the same residue-to-peptide mapping must be applied to
both target and prediction before shrinkage:

    C_peptide = M C_PF M^T

The current IsoVal experiment is residue-level because the 293 BV feature rows align with
full-target fragment indices 0..292. Target fragment 293 is a terminal residue absent from
the BV features and is excluded consistently from train/validation indices.

## Corrected objective

The experimental objective is

    L = L_mean / L_mean(uniform) + gamma * L_PF-var + eta * KL(w || uniform)

where the mean term is the current covariance-MSE geometry:

    L_mean = mean_t [ 0.5/n * r_t^T W_curve r_t ]

and the symmetric, dimensionless conditional-variance match is

    L_PF-var = mean_i [ log(v_pred,i / v_target,i)^2 ]

This is variance matching, not a Gaussian NLL. Swapping target and prediction leaves the
variance loss unchanged.

Raw frame logits are optimised and softmaxed exactly once. Saved normalised HDF5 weights
must be converted to logits before reuse; feeding probabilities through `Simulation` would
softmax them a second time and make them artificially uniform.

## Controlled experiment

Canonical truth:

- Target ensemble: plausible ISO_BI frames.
- Population: 40% open, 60% closed, distributed uniformly within each cluster.
- Fixed BV coefficients: `Bc=0.35`, `Bh=2.0`.
- No injected noise or replicate uncertainty.

Target-mean constructions:

1. `average_first`: well-specified production-semantics control.
2. `average_after`: target-only forward-model-mismatch sensitivity.

Candidate ensembles:

- ISO_BI: plausible ensemble recovery.
- ISO_TRI: contains the same open/closed frames plus 546 `-1` decoy frames.

Primary grid:

- conditional-variance scaling `gamma in {0, 0.01, 0.1, 1, 10}`;
- MaxEnt scaling `eta in {1, 10, 100, 1000}`;
- shrinkage `alpha in {0.01, 0.05, 0.10}` for variance-matching runs;
- sequence-cluster and spatial splits, stored splits 000/001/002;
- uniform initial logits plus two seeded `N(0, 0.01)` perturbations;
- prediction always average-first.

`gamma=0` is the covariance-MSE baseline and is run only with `alpha=0.05` because the PF
profile is not part of its objective. For each hyperparameter setting, the start with lowest
training objective is retained. Hyperparameters are then selected using held-out
covariance-MSE, never using cluster labels or the candidate's own variance score.

## Outputs and recovery interpretation

The runner writes:

- `manifest.json`: complete configuration, input paths and SHA-256 hashes, JAX backend,
  and the average-first/average-after Jensen gap;
- `raw_results.csv`: every fit and common evaluation metric;
- `selected_results.csv`: best-start and held-out-selected fits plus full covariance
  diagnostics;
- `selected_weights.npz`: selected frame probabilities keyed by run ID;
- `cluster_recovery_diagnostics.csv`: cluster JSD recovery, gradient alignment, and
  open-cluster conditional ESS;
- `decision.json`: cluster-recovery interpretation or `not_evaluated` for incomplete/smoke
  grids;
- `selected_method_summary.png`: common-metric comparison.

The relevant recovery distribution is over known structural populations, not individual
frames. For BI it is `[open, closed]`; for TRI it is `[open, closed, decoy]`, with target
`[0.4, 0.6, 0]`. Recovery is reported as:

    recovery (%) = 100 * (1 - sqrt(JSD_base2(predicted, target)))

Whole-ensemble ESS is not used as a gate: rejecting 546 known TRI decoys necessarily removes
support, and MaxEnt strength can be tuned separately. The diversity diagnostic relevant to
the rare open state is ESS conditional on its 13 frame weights, but it is reported only
descriptively. The post-run interpretation checks cluster recovery, held-out covariance-MSE,
and whether the variance-loss gradient at the mean-fit baseline decreases cluster JSD. No
ESS quantity or threshold contributes to the decision.

Any positive result is **synthetic-only**. A real MoPrP fit cannot use this term unless an
independent, defensible target PF conformational covariance is available. Covariance of the
experimental uptake curve over time is not a symmetric substitute for covariance of log-PF
over structural frames.

## Results (2026-07-14)

The corrected grid completed 1,248 finite fits: both target constructions, BI and TRI,
both split types, three stored splits, all `gamma`, MaxEnt and shrinkage values, and 2,000
optimisation steps. The final grid used one uniform start. A preceding three-start pilot
showed agreement in objectives and metrics at approximately `1e-6`, so the two perturbed
starts were omitted from the full grid. This is a documented deviation from the original
three-start design, not evidence that arbitrary initialisations can never matter.

The variance-matching candidate selected `gamma=10` and MaxEnt scaling 1 in all 24 paired
comparisons. It improved population L1 error, TRI decoy mass, held-out covariance-MSE, and
the conditional-variance loss in every comparison. All three shrinkage values preserved
those improvements. Median held-out covariance-MSE reductions were:

- BI average-first: 38.0% (range 34.1--43.6%);
- BI average-after: 11.5% (10.2--12.9%);
- TRI average-first: 63.5% (58.9--66.5%);
- TRI average-after: 16.4% (12.2--22.0%).

TRI decoy mass fell from 0.515--0.522 to 0.337--0.354 for average-first targets, and from
0.558--0.576 to 0.391--0.428 for average-after targets. The open fraction conditional on
non-decoy mass also moved toward the true 0.4 in every TRI comparison: 0.159--0.222 became
0.280--0.334 for average-first, and 0.079--0.124 became 0.150--0.262 for average-after.
Thus the lower population error is not solely an artefact of its algebraic dependence on
decoy mass.

The full covariance diagnostics, which were not optimised, also moved in the expected
direction. Across all four ensemble/target groups, relative Frobenius error decreased and
off-diagonal correlation increased. This does not prove that a conditional-variance
profile identifies the full covariance, but it argues against improvement being confined
to an unrelated diagonal statistic in this synthetic system.

Under the cluster-population definition, recovery improved in all 24 comparisons:

- BI average-first: median 87.14% to 90.07%, a gain of 2.98 percentage points;
- BI average-after: 80.68% to 85.15%, a gain of 4.42 points;
- TRI average-first: 41.01% to 54.85%, a gain of 13.75 points;
- TRI average-after: 35.30% to 48.70%, a gain of 13.48 points.

The variance-loss gradient at each selected mean-fit baseline had positive cosine with the
cluster-JSD gradient and a small normalized variance-gradient descent step reduced JSD in
all 24 comparisons. Thus the added term locally points toward the target population rather
than merely correlating with a better selected endpoint. From uniform weights, this was
already true for all 12 TRI comparisons but only 2 of 12 BI comparisons. For BI, the mean
fit first establishes a useful region, after which the variance term points correctly. This
distinction should be retained when reasoning about optimization schedules.

Open-cluster conditional ESS remained high relative to the 13 available open frames:

- BI average-first: 12.88--12.97 after matching;
- BI average-after: 12.70--12.82;
- TRI average-first: 12.74--12.84, an improvement over baseline in every split;
- TRI average-after: 10.81--12.37, or 83.1--95.2% of the available open frames.

With whole-ensemble ESS removed from the interpretation, every comparison improves cluster
recovery, preserves the mean fit, has a correctly aligned marginal gradient at baseline,
and retains at least 80% open-frame ESS. The corrected algorithmic result is therefore
**GO for further synthetic development and MaxEnt tuning**, not yet GO for real-data use.

## Conditional versus simple marginal PF variance

A direct comparator repeated the complete 1,248-fit grid using the shrunk marginal profile

    v_marginal = diag(C_tilde)

in place of the conditional profile `1 / diag(C_tilde^-1)`. Everything else was identical:
targets, weighted covariance, symmetric shrinkage, log-ratio loss, splits, optimizer,
hyperparameter grid, held-out selection, cluster-recovery metric, and open-cluster ESS.

Both profiles improved cluster recovery in all 24 comparisons and had correctly aligned
variance/JSD gradients at every mean-fit baseline. The distinction appears when the
candidate ensemble contains the known TRI intermediate/decoy population:

- BI average-first: conditional 90.07%, marginal 90.58% median recovery; essentially tied,
  with marginal winning 5/6 paired splits;
- BI average-after: conditional 85.15%, marginal 87.43%; marginal won 6/6;
- TRI average-first: conditional 54.85%, marginal 49.90%; conditional won 6/6, by a median
  5.16 percentage points;
- TRI average-after: conditional 48.70%, marginal 46.89%; conditional won 6/6, by 2.46
  points.

For the most relevant well-specified TRI average-first case, conditional matching also had
lower median held-out covariance-MSE (0.000167 versus 0.000253) and lower decoy mass (0.345
versus 0.412). Marginal matching retained slightly more open-frame diversity in TRI
average-after (median open ESS 12.48 versus 11.60), while both remained high relative to 13.

The simple marginal profile is therefore a strong baseline and is sufficient for improving
BI recovery. The correlation-aware conditional profile earns its extra complexity in this
synthetic test specifically through stronger TRI intermediate/decoy rejection, not through
a universal advantage on every ensemble.

## Fake-peptide TRI expected-moment investigation (2026-07-15)

The peptide topology correction remains unchanged: intended exchangeable intervals are
defined first, the physical start is placed one residue earlier to account for
`peptide_trim=1`, sparse-map rows are normalized over represented exchangeable residues,
and residue uptake is calculated before peptide mapping.

The scale-dependent target is now defined by analytic moments, with no sampled target mean:

    z_bar = sum_f w_f z_f
    C_z,res = sum_f w_f (z_f - z_bar) (z_f - z_bar)^T
    E[Z_s] = z_bar
    Cov(Z_s) = s^2 C_z,res
    uptake_target(t) = 1 - exp(-k_int * t / exp(z_bar))
    C_target,peptide(s) = M (s^2 C_z,res) M^T

This is central covariance; the raw second moment is
`z_bar z_bar^T + s^2 C_z,res`. Under the locked average-first forward model, averaging the
latent distribution means transforming `E[Z_s]`. Using `E[uptake(Z_s)]` would instead be
an average-after model. There is no replicate or observation uncertainty.

The TRI prediction is never scaled by `s`: its ordinary frame log-PFs enter both the
average-first uptake prediction and fitted covariance. Therefore `s=1` is the coherent
population-recovery condition. At `s=0.1`, the target is `0.01 C`, which is not the
covariance of the unscaled 40:60 TRI truth; it is retained only as a mean/variance-tension
diagnostic. `s=0` has zero covariance and no population information and is preflighted but
not fitted.

The mapped raw target covariance is subset to train/validation peptides before identical
shrinkage and profile extraction. The mean target and curve precision are identical at both
fitted scales, so covariance-MSE baselines are computed once per panel/split and reused.
The nine registered fragment-index splits are now locked explicitly after a regression
check exposed a sequence-clustering tie instability in `equal/split_002`; the previously
documented overlapping train/validation IDs remain.

### Validation and corrected results

Map congruence passed with maximum float32 relative error `1.023e-6`; covariance scaling
was exact and `s=0` had exactly zero trace. A three-start pilot failed the `1e-6`
stability tolerance (maximum objective range `0.0149`, recovery range `0.572` percentage
points), so all three starts were retained. The complete merged run contains 5,400 finite
fits, 54 held-out-selected fits, and 36 paired comparisons.

Corrected median cluster-population recovery percentages are:

| Target scale | Panel | Covariance-MSE | Marginal | Conditional |
|---:|---|---:|---:|---:|
| 0.1 | Equal | 32.79 | 33.80 | 33.50 |
| 0.1 | Random fixed | 33.39 | 34.41 | 33.59 |
| 0.1 | Random variable | 34.10 | 34.15 | 35.78 |
| 1 | Equal | 32.79 | 34.37 | 33.89 |
| 1 | Random fixed | 33.39 | 35.45 | 34.98 |
| 1 | Random variable | 34.10 | 35.92 | 35.70 |

At the primary `s=1` condition, both profiles improved recovery and had correctly aligned
variance/JSD gradients in all 18 panel/split/method comparisons. Median
marginal/conditional gains were `+2.19/+1.23` points for equal peptides,
`+2.91/+1.59` for random-fixed peptides, and `+1.22/+1.90` for random-variable
peptides. Median held-out covariance-MSE ratios were `0.86--0.98`, below the `1.05`
preservation gate. Marginal matching remains stronger for equal and random-fixed panels,
while conditional matching remains stronger for the random-variable panel.

Corrected `s=1` reproduced the old clean `s=0` condition: selected hyperparameters were
identical and the maximum recovery difference was `0.00012` percentage points. This is
expected because both conditions use `uptake(z_bar)` and `C_z,res`. The former
single-draw `s=1` collapse was therefore a target-mean mismatch, not a failure of variance
matching.

At diagnostic `s=0.1`, median marginal/conditional gains were `+1.48/+0.72`,
`+0.77/+1.09`, and `+0.05/+1.29` points for the three panels. However, random-fixed
conditional gradient alignment fell to one of three splits and some paired gains were
negative. These values describe the tension created by an artificially contracted target
covariance; they are not evidence for or against recovery of the 40:60 structural
population. Open-cluster ESS remains descriptive only at both scales.

The residue-level investigation does not require rerunning. It already uses the analytic
40:60 target mean and discrete conformational covariance without sampled target means.

## Covariance-construction litmus test (2026-07-15)

The litmus was simplified to remove single-realization scale/seed branches. It retains the
weighted discrete BI log-PF covariance, timepoint-specific discrete uptake covariance,
Jacobian propagation, and 100,000-draw nonlinear Monte Carlo controls. Covariance across
timepoints is now evaluated once for the fixed analytic mean curve.

All numerical gates passed: sparse-map covariance error was `1.50e-15`, Monte Carlo log-PF
covariance error `0.842%`, split-half covariance distance `0.846%`, minimum split-half
correlation `0.99976`, and curve-precision reproduction error `3.48e-6`. At `s=0.1`,
the Jacobian approximation passed against nonlinear Monte Carlo with maximum raw covariance
error `4.25%`, normalized distance `1.60%`, marginal log-RMSE `0.024`, and conditional
log-RMSE `0.016`.

Discrete uptake covariance remained materially time-dependent: maximum trace-normalized
distance was `0.334`, marginal log-RMSE `0.954`, and conditional log-RMSE `0.821`.
Nonlinear Monte Carlo reached maximum distance `0.392`. An uptake-space variance objective
must therefore use a timepoint-specific covariance.

The fixed-mean curve surrogate failed every registered comparison. Against log-PF
covariance, worst marginal correlation/log-RMSE were `0.466/0.447`, while conditional
values were `-0.194/0.523`. Against timepoint-specific uptake covariance, worst
correlations were negative and the curve trace was up to `47.37` times larger. Covariance
across five uptake timepoints has rank at most four and describes curve geometry, not
conformational spread.

The physical PF target remains the weighted discrete BI **log-PF covariance over structural
frames**, mapped with `M C M^T` before profile extraction. Curve precision may remain fixed
mean-residual geometry but must not be interpreted as PF variance.

## Peptide overlap and redundancy study (2026-07-16)

The follow-up tested whether conditional variance improves relative to marginal variance as
peptide number and overlap increase. The target remained the coherent `s=1` analytic mean
and BI conformational covariance; TRI prediction was unscaled. A geometry litmus generated
123 layouts: equal-tile controls plus 20 random-fixed and 20 random-variable layouts at each
of 15, 30, and 60 peptides. Mean random-layout redundancy degree increased from about
`0.9` to `1.9` to `3.8`. Four layouts per count were selected using only equal/low/median/high
redundancy roles.

Overlap weights were fixed from the row-normalized sparse map:

    G_ij = cosine(M_i, M_j)
    q_i = 1 / sum_j G_ij

with `q` normalized to mean one and recomputed within each train/validation subset. They
weighted only the symmetric marginal or conditional log-ratio loss; covariance-MSE remained
the common baseline. The existing sequence-cluster splitter was run unchanged and its exact
outputs, including any duplicate IDs, were recorded rather than gated.

The historical gamma/MaxEnt/shrinkage grid was calibrated on one median-redundancy layout
per peptide count. Frozen settings used `gamma=10` and MaxEnt `1` for every PF method.
Marginal used `alpha=0.01`; conditional used `alpha=0.10` at 15 peptides and `0.01` at
30/60 peptides. The representative run contained 540 finite fits and 180 selected endpoints.

Median recovery and gain over covariance-MSE were:

| Peptides | Method | Recovery | Gain (pp) | Mean-loss ratio |
|---:|---|---:|---:|---:|
| 15 | Marginal | 35.02 | +2.28 | 0.873 |
| 15 | Marginal overlap | 35.01 | +2.25 | 0.874 |
| 15 | Conditional | 34.38 | +1.60 | 0.926 |
| 15 | Conditional overlap | 34.26 | +1.51 | 0.925 |
| 30 | Marginal | 37.31 | +7.04 | 0.671 |
| 30 | Marginal overlap | 37.27 | +7.10 | 0.677 |
| 30 | Conditional | 36.48 | +5.92 | 0.673 |
| 30 | Conditional overlap | 36.54 | +5.91 | 0.675 |
| 60 | Marginal | 40.87 | +4.19 | 0.791 |
| 60 | Marginal overlap | 40.80 | +4.12 | 0.794 |
| 60 | Conditional | 39.44 | +3.18 | 0.859 |
| 60 | Conditional overlap | 39.38 | +3.23 | 0.853 |

Every method had positive median recovery gain and preserved the mean fit at every peptide
count. Profile gradients reduced cluster JSD in 11/12 comparisons per method at 15 peptides
and 12/12 at 30 and 60 peptides.

The registered conditional-scaling hypothesis failed for random overlapping layouts.
Weighted conditional-minus-marginal recovery was `-0.340` points at 15 peptides and
`-0.394` at 60; its count Spearman correlation was `-0.053` and redundancy correlation
`-0.267`. Equal-tile controls differed: conditional exceeded marginal by about `2.0` and
`2.34` points at 30 and 60 peptides. Thus peptide count alone can help conditional matching,
but added overlap does not expose an advantage over the marginal profile in this setup.

Inverse-overlap weighting was nearly neutral in median recovery. It reduced conditional
layout-gain IQR slightly (`3.65` to `3.62` points) and satisfied the registered non-degradation
plus IQR gate for conditional, but not for marginal (`3.91` to `3.95`). It should remain an
optional geometry correction rather than replace the unweighted loss.

The overlap-projected full-covariance litmus was more promising. After removing cosine-map
modes below `1e-6` relative eigenvalue and applying identical shrinkage, the symmetric
log-Euclidean covariance gradient reduced cluster JSD in all `123/123` layouts. Gradient
cosines were `0.306--0.752`, and covariance-loss rankings were stable across projection
thresholds (`Spearman=0.9998` between `1e-4` and `1e-8`). It passes the registered gate for
a separate optimization experiment; it was not fitted in this study.

## Projected full-covariance optimization (2026-07-16)

The gated follow-up optimized the overlap-projected log-Euclidean peptide PF covariance
directly. Train and validation projections were constructed independently from their
row-normalized sparse-map subsets at the fixed `1e-6` relative eigenvalue threshold. The
objective was the normalized uptake-curve covariance-MSE plus `gamma` times projected PF
covariance loss plus MaxEnt KL. Target covariance remained the analytic BI conformational
covariance `M C_res M^T`; recovery was not used for calibration.

One global setting was calibrated on the selected 30-peptide median-overlap layout using
the historical gamma `{0.01, 0.1, 1, 10}`, MaxEnt `{1, 10, 100, 1000}`, and shrinkage
`{0.01, 0.05, 0.10}` grid. Of 48 settings, 24 passed the preregistered median validation
curve-MSE ratio gate of `<=1.05`. Minimum validation projected-covariance loss then selected
`gamma=10`, MaxEnt `1`, and `alpha=0.10`. This setting was frozen across all 12 layouts,
three splits, and three starts. All 108 new fits and 36 selected endpoints were finite; the
previous covariance-MSE, marginal, and conditional endpoints were reused unchanged.

| Peptides | Projected recovery | Covariance-MSE | Marginal | Conditional | Projected - marginal (pp) | Projected/marginal covariance loss |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 35.72 | 32.84 | 35.02 | 34.38 | +0.64 | 0.977 |
| 30 | 36.13 | 30.29 | 37.31 | 36.48 | -0.60 | 1.009 |
| 60 | 37.72 | 36.49 | 40.87 | 39.44 | -2.01 | 1.084 |

Projected covariance improved recovery over covariance-MSE in all `36/36` paired
comparisons, with median gains of `+2.83`, `+5.85`, and `+1.15` points at 15, 30, and 60
peptides. Its median validation curve-MSE ratios to covariance-MSE were `0.868`, `0.716`,
and `0.939`, so failure was not caused by sacrificing the mean fit. Its loss gradient at
the covariance-MSE endpoints reduced cluster JSD in `35/36` comparisons.

It nevertheless failed the promotion gate. Overall median recovery was `0.493` points below
marginal, the projected-minus-marginal difference was positive only at 15 peptides, and it
did not lower projected validation covariance loss relative to marginal at 30 or 60
peptides. Thus optimizing the full overlap-projected covariance is a valid improvement over
the curve-only baseline but does not justify replacing marginal PF variance for these
synthetic peptide layouts. The recommendation remains **marginal PF variance**. Conditional
variance remains a useful reported comparator, while overlap weighting and projected full
covariance should not be default fitting objectives from this evidence. ESS was not used in
calibration, gating, or interpretation.

## Implementation and reproduction

Reusable, differentiable operations:

    jaxent/src/analysis/pf_variance.py

Controlled runner:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        investigate_pf_conditional_variance.py

Fake-peptide TRI runner and shard merger:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        investigate_pf_peptides.py
        merge_pf_peptide_shards.py

Covariance-construction diagnostic and registered outputs:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        analyze_pf_covariance_constructions.py
        _pf_covariance_litmus/

Corrected merged peptide outputs:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        _pf_peptide_moment_final/

Overlap/redundancy study:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        investigate_pf_peptide_overlap.py
        _pf_peptide_overlap_litmus/
        _pf_peptide_overlap_calibration/
        _pf_peptide_overlap_final/

Projected full-covariance optimization:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        investigate_pf_peptide_overlap.py
        _pf_peptide_projected_calibration/
        _pf_peptide_projected_final/

Focused tests:

    jaxent/tests/unit/analysis/test_pf_variance.py
    jaxent/tests/unit/analysis/test_pf_peptide_latent_runner.py
    jaxent/tests/unit/analysis/test_pf_covariance_constructions.py

Smoke integration check:

    python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/\
        investigate_pf_conditional_variance.py \
        --smoke --output-dir /tmp/jaxent_pf_variance_smoke

Full decision run:

    python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/\
        investigate_pf_conditional_variance.py \
        --output-dir jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/\
            _pf_conditional_variance

The smoke run is an integration check only and deliberately produces
`decision.status="not_evaluated"`. It must not be quoted as scientific evidence.

## Verification implemented

Tests cover:

- weighted population covariance without Bessel correction;
- conditional variance of a diagonal covariance;
- Cholesky results against an explicit small-matrix inverse;
- zero-at-truth and target/prediction swap symmetry;
- finite profiles and gradients for identical or effectively single-frame ensembles;
- exactly-one-softmax probability normalisation;
- equality of the helper's average-first uptake and the real BV forward pass;
- distinction between the average-first prediction and average-after target sensitivity.
- exact analytic `s^2 C` target scaling, zero covariance at `s=0`, and exclusion of that
  non-identifiable endpoint from fitting;
- target scale never entering the TRI prediction forward pass;
- locked fake-peptide fragment-index splits and map-before-profile extraction.

## Superseded findings

The prior `RESULTS`, `RESULTS 2`, and `RESULTS 2b` sections are superseded as decision
evidence. In particular:

- per-peptide curve variance across timepoints is signal, not conformational target
  variance;
- conformational variance and curve variance must not be added as independent Gaussian
  observation variances;
- at synthetic truth, residual magnitude equals injected noise magnitude, so demanding
  correlation with noise but not residual was circular;
- static variance scaling at fixed residuals did not establish an accessible optimisation
  direction;
- residual/spread correlation alone cannot predict recovery or decoy rejection.
- time-averaged uptake covariance with diagonal or trace scaling is not the target
  conformational covariance; the temporary validator and its generated results were removed.
- single sampled latent means are not used as expected target means; their scale/seed fitting
  results and generated artifacts were removed.

Those probes motivated useful numerical safeguards, but they do not determine the outcome
of the corrected conditional-variance matching experiment.

## Effective-rate covariance extraction checkpoint (2026-07-16)

Status: **correspondence litmus complete; Stage 2 pending user review**.

This section supersedes the preceding mean/rate exploratory interpretation as the active
decision record. The fitting question is whether an ordinary target uptake curve contains
enough information to construct a covariance target corresponding to the underlying BV
trajectory physics.

The primary amide coordinate is the linear effective exchange rate

    k_f,i = k_int,i * exp(-z_f,i)

and the ground-truth panel matrix is

    C_k = Cov_w(k_f)
    C_k,peptide = M C_k M^T.

Intrinsic rates therefore enter covariance entries pairwise. `Cov(z)` and
`D_kint Cov(z) D_kint` are registered coordinate-mismatch controls, while
`D_kint Cov(exp(-z)) D_kint` is algebraically identical to the exact rate covariance.
The known BI 40:60 weights are primary; uniform weights are a sensitivity condition.

### What is and is not observable

The ordinary target contains one mean peptide-uptake vector at each of five timepoints.
Framewise uptake covariance is available only in the synthetic ISO system and is retained
as an oracle upper bound, not described as experimentally extractable. Candidate
observable constructions were covariance across timepoints of raw uptake, `uptake/t`,
apparent rate `-log(1-uptake)/t`, adjacent-time survival slopes, and timepoint-RMS-normalized
uptake. Complete matrices were compared before marginal or conditional profile extraction.

### Exact non-identifiability control

Frame identities were permuted independently for each amide within equal-weight groups.
This preserves every amide marginal rate distribution and hence every mean uptake value at
every timepoint, while changing inter-amide coupling.

| Panel | Maximum mean-uptake change | Full covariance change | Marginal change | Conditional change |
|---|---:|---:|---:|---:|
| Equal | 3.33e-16 | 0.009 | 0.001 | 0.003 |
| Random fixed | 4.44e-16 | 0.197 | 0.195 | 0.130 |
| Random variable | 4.44e-16 | 0.165 | 0.108 | 0.164 |

Thus a complete noiseless mean uptake curve does not uniquely determine full or
conditional effective-rate covariance. It also does not determine mapped peptide marginal
variance, because `diag(M C_k M^T)` contains unobserved cross-amide covariance terms.

### Observable construction results

The primary full-matrix gates required trace ratio `0.5--2.0`, normalized distance at most
`0.25`, and off-diagonal correlation at least `0.80`. Marginal and conditional gates
required scale ratio `0.5--2.0`, Pearson and Spearman correlations at least `0.90`, and
log-RMSE at most `0.25`, with shrinkage sensitivity at `alpha=0.01/0.10` around primary
`alpha=0.05`.

| Uptake-only construction | Full normalized distance | Off-diagonal correlation | Trace ratio | Full gate |
|---|---:|---:|---:|---:|
| Adjacent survival slope | 1.158 | 0.193 | 0.000171 | FAIL |
| Apparent rate | 1.127 | 0.194 | 0.00404 | FAIL |
| Raw uptake | 1.201 | 0.018 | 0.000835 | FAIL |
| Timepoint-RMS normalized | 1.129 | 0.119 | 0.000290 | FAIL |
| Uptake divided by time | 1.139 | 0.180 | 0.00312 | FAIL |

None passed the marginal, conditional, or full-covariance promotion gates on all three
panels. Covariance across five timepoints has rank at most four (adjacent slopes at most
three) and describes kinetic curve geometry rather than the joint distribution of rates
over structural frames.

### Marginal-information ceiling

For an individual amide, survival is the Laplace transform
`S_i(t)=E[exp(-k_i t)]`; its near-zero curvature contains the first two rate cumulants.
At automatically scaled asymptotic times, quadratic log-survival recovery gave mean-rate
relative error `6.65e-10`, marginal-variance error `9.70e-5`, and marginal correlation
`1.000`. At the five production times the corresponding errors were `0.997` and `1.000`,
because many rates were already outside the local cumulant regime. After mapping to only
15 peptide curves, production-grid marginal-variance errors remained approximately one.

This separates two facts: amide marginal rate variance is theoretically encoded in a
sufficiently resolved early-time survival curve, but the present five production times and
peptide aggregation do not recover it. Cross-amide covariance is not encoded in separate
mean uptake curves even with ideal time resolution.

### Consequence for fitting

No uptake-only covariance construction is promoted to decoy/gradient testing. The exact
rate covariance remains a valid synthetic oracle for measuring the best possible benefit
of covariance alignment, but using it as though it were extracted from ordinary target
uptake would leak trajectory truth into the fit. Stage 2 must not fit marginal,
conditional, or full effective-rate covariance from these five-point target curves unless
new joint/replicate information or a defensible structural prior is supplied.

Implementation and complete outputs:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        investigate_uptake_rate_covariance.py
        _uptake_rate_covariance_litmus/

No TRI decoy test, recovery gradient, new panel, or optimization was run.

### Superseded mean-construction notes

The removed mean-construction comparison remains available in its generated diagnostic
directory for provenance, but it is not part of the active covariance decision. Stage 2
panel construction and all optimization remain **pending explicit user approval**.

## Registered next step: peptide-level Monte Carlo target covariance

Status: **complete (2026-07-16); see Stage 1B results below**.

The next experiment must not sample latent residue log-PFs or effective-rate frames and
then pass them through the synthetic BV forward model. That construction is available only
because ISO exposes residue/frame truth; it does not represent covariance extractable from
target uptake data and is therefore excluded as the target generator.

Instead, Monte Carlo sampling will occur directly in the observed peptide-uptake residual
coordinate. For panel target mean uptake `Y_target` and a registered peptide-level target
covariance `Sigma_target`, generate

    Y_draw,m = Y_target + epsilon_m
    epsilon_m ~ Normal(0, s^2 Sigma_target).

These are additive target-data draws, not new structural frames and not BV forward-model
predictions. Uptake bounds will not be enforced by clipping because clipping changes the
registered covariance; scales that generate material out-of-range mass will be reported
as invalid for an additive-Gaussian interpretation. The primary condition is `s=1`, with
`s=0.1` as a numerical small-noise control and `s=0` as the exact-mean control.

For a peptide-by-peptide `Sigma_target`, draw one independent correlated peptide residual
vector per timepoint and replicate, using the same registered matrix at every timepoint;
estimate recovery separately at each timepoint before pooling summaries. For a stacked
peptide-by-time candidate, vectorize the complete target curve and draw once from its
`(T*P) x (T*P)` covariance, retaining cross-time blocks. These two sampling models must
not be mixed in one estimator or comparison.

### Stage 1B: sampling and covariance-estimator validation

Use both uniform and known 40:60 BI target means on the existing equal, random-fixed, and
random-variable panels. No panel construction or fitting enters this stage.

1. Construct peptide-level `Sigma_target` candidates using only the target uptake matrix:
   raw uptake-curve covariance, uptake divided by time, apparent-rate covariance,
   adjacent-time survival-slope covariance, and timepoint-RMS-normalized covariance.
   Preserve the distinction between peptide-by-peptide covariance across time and a fully
   stacked peptide-by-time covariance. The latter is reported separately and is not
   silently collapsed into a peptide matrix.
2. Draw additive multivariate Gaussian target replicas at sample sizes
   `N={25, 50, 100, 250, 500, 1000, 5000}` with 20 deterministic seeds per condition.
   Estimate covariance with weighted-population normalization for controlled comparisons
   and ordinary sample normalization as a sensitivity check.
3. Verify Monte Carlo recovery of the covariance that was actually supplied: raw
   full-matrix relative error, trace ratio, normalized Frobenius distance, off-diagonal
   correlation, numerical/effective rank, marginal profile, and conditional profile.
   Apply shrinkage only after the complete matrix is estimated, with primary `alpha=0.05`
   and `0.01/0.10` sensitivity.
4. Quantify sampling uncertainty separately from construction mismatch. Report bias,
   across-seed standard deviation, confidence intervals, and the probability that each
   registered full/marginal/conditional fidelity gate passes at every `N`.
5. Report the fraction of additive draws outside physical uptake bounds `[0,1]`. A
   condition with more than 1% out-of-range entries is marked incompatible with an
   untruncated additive-Gaussian target model and cannot be promoted merely because its
   sample covariance converges.

This stage answers whether a covariance chosen in uptake space can be estimated stably
from finite Monte Carlo target draws. Convergence to the imposed `Sigma_target` is a
sampling result; it does not by itself establish correspondence with conformational
effective-rate covariance. The exact ISO rate covariance remains an external oracle used
only to report that separate physical correspondence.

### Stage 1B gates

A target covariance/profile can proceed to alignment testing only if, at a finite sample
size no larger than `N=1000`:

- median trace ratio is within `0.9--1.1`;
- median normalized full-matrix distance is at most `0.10`;
- median off-diagonal correlation is at least `0.90`;
- marginal and conditional median Pearson/Spearman correlations are at least `0.95` with
  log-RMSE at most `0.10`;
- at least 90% of seeds pass the relevant representation gate;
- the conclusion is unchanged at `alpha=0.01` and `0.10`;
- no more than 1% of additive draws lie outside `[0,1]` at the promoted scale.

Marginal, conditional, and full covariance qualify independently. If no construction
passes, the result is recorded and covariance alignment does not advance by default.

Required outputs are a manifest with input hashes and seeds, construction registry,
sample-level metrics, across-seed summaries, bound-violation table, full matrices before
profiles, shrinkage sensitivity, convergence plots, gate results, and a human-readable
report. Unit tests must cover seeded sampling, analytic covariance recovery, population
versus sample normalization, matrix-before-profile ordering, shrinkage stability, bound
violation accounting, and integration across both weight sets and all three panels.

## Stage 1B results (2026-07-16)

Status: **complete; no representation is recommended for promotion**.

The run drew additive Gaussian residuals in the observed peptide-uptake coordinate around the
target mean, exactly as registered: no latent frames, no BV forward passes, no clipping, no
fitting, no decoy test, and no cluster recovery. It produced 30,240 sample-level rows and
1,512 across-seed summaries at 20 seeds per condition, over both weight sets, all three
panels, both sampling models, `N in {25...5000}`, `s in {0, 0.1, 1}`, and both estimator
normalizations.

One registered construction could not be expressed in one of the sampling models. The
adjacent-survival-slope transform lives on `T-1` intervals, so it cannot form a covariance of
the `T`-length drawn uptake curve; it is retained peptide-by-peptide and excluded from the
stacked model rather than embedded. The stacked candidates are the separable Kronecker form
`Cov_time(transform) kron Sigma_peptide`. A single mean curve is one observation of the
vectorized curve and cannot furnish a general `(T*P) x (T*P)` covariance, so the separable
form is the constructible candidate carrying cross-time blocks, and is registered as such.

### The sampling question is answered, and it is easy

Every construction converged, in both models and both weight sets. At `N=5000` the median
normalized full-matrix distance was `0.0004--0.006` and off-diagonal correlation `>= 0.9998`.
The metrics-only full gate is met by peptide-by-peptide constructions at `N=250--500`.

This is easy for a reason that undercuts its value: **every candidate is nearly rank-one**.
Effective rank was `1.01--1.27` for the peptide-by-peptide matrices and `1.03--1.64` for the
stacked ones, against dimensions of 15 and 75. These matrices are essentially a single outer
product, so recovering them from finite draws is a trivial estimation target. Fast
convergence here measures the degeneracy of the object, not the quality of the target.

### Qualification is a scale convention, not a result

Twelve qualifications were recorded, all in the stacked model (`curve_raw_uptake` and
`curve_timepoint_rms_normalized`, full/marginal/conditional, both weight sets). No
peptide-by-peptide construction qualified.

That split is an artefact of the noise-scale convention and must not be read as the stacked
model being better. Peptide-by-peptide constructions pass every estimator metric and are
blocked **solely** by the bound-violation gate: at `s=1` their median out-of-range entry
fraction is `0.029--0.313`, far above the registered `0.01`. The stacked constructions pass
that gate only because the Kronecker product multiplies two small traces, making the stacked
covariance `5--33%` of the peptide-by-peptide trace for the same construction. `s=1`
therefore denotes a materially different noise magnitude in each model, and each construction
carries its own natural magnitude. There is no common noise scale across constructions or
models, so `s=1` is not a comparable condition, and the registered instruction not to mix the
two sampling models in one comparison is load-bearing.

The honest reading of the bound accounting is the opposite of promotion: at the primary
scale, an additive untruncated Gaussian target model in uptake space is **invalid** for every
peptide-by-peptide construction, and survives in the stacked model only where the covariance
happens to be small.

### Marginal and conditional variance are the most estimable and the least meaningful

Marginal and conditional profiles are the easiest objects in the study. Against the covariance
actually supplied, both reach Pearson `>= 0.996` and log-RMSE `<= 0.09` at `N=25` for every
construction, in both sampling models and both weight sets, and they qualify at smaller `N`
than the full matrix (conditional from `N=25--50`, marginal from `N=250`, full only from
`N=500--1000`). This ordering is expected: they are low-dimensional summaries of a
near-rank-one object. Like the full matrix, the peptide-by-peptide profiles are blocked only
by the bound-violation gate, not by any estimator metric.

Estimability is not correspondence. Against the exact mapped effective-rate oracle, at the
primary `alpha=0.05`, **0 of 15 construction/panel combinations pass the profile gate for
marginal, and 0 of 15 for conditional**. The failure is not marginal: profile log-RMSE is
`0.38--2.96` against a `0.10` bar, and the correlations are frequently negative (marginal
Pearson `-0.357--0.914`, conditional Pearson `-0.305--0.712`). The single closest case,
`curve_apparent_rate` marginal on the random-fixed panel, reaches Pearson `0.914` but still
carries log-RMSE `0.93`, so it agrees on ordering while being wrong by roughly an order of
magnitude in scale.

So the sampling gates and the physical question point in opposite directions for exactly these
two representations: they are the quantities Stage 1B can estimate best from finite draws, and
also the quantities furthest from what the target is supposed to represent. A gate that only
asks "can this be estimated" will always favour them. That is the trap this checkpoint existed
to detect.

### Physical correspondence remains absent

The oracle table reproduces the 2026-07-16 correspondence finding on the same inputs. Against
the exact mapped effective-rate covariance `M Cov_w(k) M^T`, every uptake-only construction
gave normalized distance `0.90--1.40`, off-diagonal correlation `-0.004--0.472`, and trace
ratio `1e-6--6e-3`. Sampling convergence and physical correspondence are separate questions,
and only the first one passed.

### Consequence

Stage 1B's narrow question — can a covariance chosen in uptake space be estimated from finite
target draws — is answered **yes**, but the answer is uninformative: the estimable objects are
near-rank-one curve geometry that does not correspond to conformational spread, and the
qualifications that cleared the gates did so on a scale convention rather than on physics.

**No representation is promoted to Stage 2 alignment testing.** Per the registered rule, if no
construction passes, covariance alignment does not advance by default; the qualifications here
are not a defensible exception. Stage 2 remains **pending explicit user approval** and should
not be authorized on this evidence. Reconsidering it needs new joint or replicate information,
or a defensible structural prior, exactly as the correspondence litmus concluded.

### Scope: this does not touch the marginal PF variance recommendation

"Marginal variance" names two different objects in this document and they must not be
conflated. Stage 1B concerns **uptake-space** covariance constructed from a target curve, and
finds its marginal and conditional profiles physically empty. The standing recommendation of
**marginal PF variance** from the overlap/redundancy study is a different coordinate: the
weighted conformational covariance of log-PF over structural frames, `M C_res M^T`, with a
target available only because ISO exposes frame truth.

Stage 1B neither supports nor weakens that recommendation, and nothing here changes the
synthetic-only status of the PF variance work. The two are connected only by a negative:
Stage 1B was the registered attempt to find an experimentally constructible stand-in for that
PF target, and it did not find one. Marginal PF variance therefore remains the recommended
objective for synthetic work and remains unavailable for real MoPrP data, for the same reason
as before -- no defensible independent target PF conformational covariance exists.

Implementation and complete outputs:

    jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/
        investigate_peptide_mc_covariance.py
        _peptide_mc_covariance_litmus/

    jaxent/tests/unit/analysis/test_peptide_mc_covariance.py

## MoPrP real-data correspondence canary (2026-07-17)

Status: **complete; the synthetic conclusion reproduces on real data**.

Every conclusion up to this point rested on the ISO synthetic system: 874 TeaA frames, five
timepoints, known 40:60 truth. The open risk was that "curve covariance is not conformational
covariance" was an artefact of ISO rather than a fact about HDX. The most concrete version of
that risk: curve covariance has rank at most `T-1`, and ISO's candidates came out at effective
rank `1.01--1.64`. Real MoPrP has **15 timepoints**, so its curve covariance could have been
materially richer.

The correspondence litmus was therefore run on real MoPrP experimental data, with the two AF2
ensembles as references at uniform weights: Filtered registered as the mildly-positive
reference, MSAss as the negative one. This is deliberately weak evidence and was scoped as a
canary — there is no oracle, neither reference is truth, and n=2 references has no statistical
power. It could fail loudly but not succeed quietly.

Inputs are the real 14x15 `MoPrP_dfrac.dat` curve, the 14 experimental segments at
`peptide_trim=2`, and the existing 96-residue x 500-frame AF2 feature files. `k_ints` are
byte-identical across the two ensembles, so the references differ **only** by conformation.
The pipeline fits 12 peptides while `Sigma.npz` is 14x14; the 12 are primary and the
discrepancy is recorded rather than reconciled.

### Preflight: the canary has resolving power

The two references are genuinely distinguishable: normalized distance `0.428`, off-diagonal
correlation `0.584`, trace ratio `5.57` (MSAss is the more heterogeneous ensemble, consistent
with its 4 clusters against Filtered's 2). This clears the registered `0.05` threshold, so a
null result is informative rather than an artefact of two identical references.

### The 15-timepoint hypothesis is dead

Effective rank of the real-data constructions is `1.04--1.52`, against ISO's `1.01--1.64`.
Numerical rank reaches the full 12, but the spectrum is still concentrated in essentially one
direction. **Tripling the timepoint count did not enrich the curve covariance.** The
near-rank-one degeneracy is not an artefact of five timepoints; it is what covariance across
uptake timepoints is.

### No correspondence, and the negative control wins

No construction passes the full gate against either reference: **0 of 8**. Marginal and
conditional profile gates also pass **0 of 8**. Trace ratios run `31--105`, so the curve
covariance is one to two orders of magnitude off in scale from conformational rate covariance,
and off-diagonal correlations are `-0.13--0.08`, i.e. unrelated.

The sharpest result is that the registered **coordinate-mismatch negative control beats every
observable construction**. `control_logpf` (`M Cov_w(z) M^T`) sits at distance `0.33--0.61`
from the references, while every uptake-only construction sits at `0.77--1.09`. Log-PF
covariance is the wrong coordinate but is at least conformational; curve covariance is not
conformational at all, and it shows.

### The registered direction is not supported

Five of six observable constructions are closer to the **negative** MSAss reference than to
the mildly-positive Filtered one. Only `curve_timepoint_rms_normalized`, the shape-only
control, favours Filtered (`delta = +0.086`).

This should not be over-read as "curve covariance prefers the worse ensemble". Every distance
is `0.77--1.09`, i.e. essentially unrelated to both references, and the deltas are only
`0.15--0.38` of the reference separation. The defensible statement is the null one: no
construction is close to either reference, and the registered direction is not supported.

### Identifiability reproduces on real data, without any oracle

The strongest result here needs no truth ensemble. Permuting frames within equal-weight groups
changed predicted mean uptake by at most `6.7e-16` (MSAss) and `7.8e-16` (Filtered) — exactly
zero — while changing mapped full/marginal/conditional rate covariance by `0.766/0.559/0.487`
and `0.546/0.493/0.363` respectively.

On the real system, the mean uptake curve does not determine conformational coupling. This does
not depend on either AF2 ensemble being correct, so it survives the canary's weakness entirely.

### What `Sigma.npz` actually is

`compute_sigma_real.py:162` computes `Sigma = np.cov(dfrac_values) + 1e-6 I` over the
(14 peptides x 15 timepoints) curve. Verified numerically: `Sigma` equals the registered
`curve_raw_uptake` construction scaled by the Bessel factor plus that ridge, to `1e-6`, and
`production_sigma` and `curve_raw_uptake` land on identical distances (`0.7679` vs `0.7679`)
and deltas (`-0.16101` vs `-0.16101`). It is a curve construction, not a measurement.

It is named "observation noise covariance" and consumed by the production `Sigma_MSE` loss
(`optimise_ISO_TRI_BI_splits_maxENT.py:225-228`). MoPrP has **no replicates** — `MoPrP_dfrac.dat`
is a bare matrix of means and `HDX_peptide` carries no uncertainty field — so there is no
observation noise in these data to estimate. Its spread is across timepoints, which is signal.

This is a naming and interpretation issue, not necessarily a defect in the fit: as
trace-normalized precision weighting for the mean residual, it occupies exactly the legitimate
`W_curve` role this document already reserves for fixed curve geometry. The rule it must not
cross is being read as a PF conformational variance, and the effective ranks above show why.
The 14x14/12-peptide dimension discrepancy is recorded for separate follow-up.

### Consequence

The canary confirms the synthetic conclusion on real data and removes its most plausible
escape route. Curve covariance remains fixed mean-residual geometry, not conformational
variance, and no uptake-only construction is a defensible PF target on real MoPrP. Stage 2
remains **pending explicit user approval** with nothing promoted.

Implementation and complete outputs:

    jaxent/examples/2_CrossValidation/fitting/jaxENT/
        investigate_moprp_uptake_covariance.py
        _moprp_uptake_covariance_litmus/

    jaxent/tests/unit/analysis/test_moprp_uptake_covariance.py

## Stage 2 after the Monte Carlo checkpoint

Status: **pending explicit user approval; Stage 1B promoted nothing and the MoPrP canary
confirmed the synthetic result**.

Stage 2 will first test whether the Stage 1B-qualified uptake covariance representation
provides useful alignment information before any large fitting grid is launched.

1. Freeze each qualified target construction, sample size, covariance estimator,
   shrinkage value, and representation (`marginal`, `conditional`, or `full`) without using
   cluster recovery for selection.
2. Express the predicted covariance in the same peptide-uptake coordinate as the sampled
   target covariance. Predicted frame log-PF covariance or effective-rate covariance must
   not be matched directly to an uptake-space target. Prediction must calculate
   framewise residue uptake with the candidate ensemble's intrinsic rates, map residue
   uptake to peptides, and form the complete weighted peptide covariance before extracting
   a profile or full-matrix loss.
3. At uniform weights and at the existing mean-fit baseline, compare target and predicted
   covariance losses for BI and TRI. Test gradient cosine with cluster-JSD gradient,
   one-step JSD change, open/closed sensitivity, TRI decoy sensitivity, mean-uptake
   preservation, and covariance effective rank. ESS remains descriptive only.
4. Treat the exact ISO framewise uptake covariance as an oracle positive control and raw
   log-PF covariance as a coordinate-mismatch negative control. The MC-sampled target
   covariance must be judged by how closely its gradient and discrimination behavior
   approaches the oracle, not merely by endpoint covariance error.
5. Promote a representation to fitting only if it reduces cluster JSD in every registered
   split at the mean-fit baseline, points away from the TRI decoy population, preserves
   validation mean-curve MSE within a ratio of `1.05`, and outperforms the coordinate-
   mismatch negative control. Failure leaves the covariance term diagnostic only.

Only after that gradient/decoy gate may the previously registered Stage 2 panel and fitting
work begin: deterministic 15-peptide panel review if still requested, followed by the
frozen covariance-MSE, marginal, and conditional objectives; MaxEnt sweep
`{1,10,100,1000}`; three sequence-cluster splits; three starts; and 2,000 steps. Selection
must use validation mean-curve behavior and registered covariance criteria without using
cluster recovery. No Stage 2 optimization is authorized by this handoff update alone.

## Shared-rate peptide kinetic diagnostic

Status: **capacity audit complete; three-rate model rejected (2026-07-17)**.

The real-data follow-up uses a shared exponential-rate mixture rather than treating
covariance across uptake timepoints as conformational covariance. For peptide `p`,

    y_p(t) = sum_c q_pc * (1 - exp(-k_c * t)),

with one to three ordered positive rates shared across peptides and a peptide-specific
simplex containing the exchanging fractions plus a zero-rate fraction. Component count and
shrinkage are selected by five chronological timepoint folds and the one-standard-error
rule. The model is linear in peptide weights once the shared rates are fixed, but it is not
the existing `linear_BV_ForwardPass`, which remains untouched.

Only three uncertainty/heterogeneity objects are registered:

1. joint penalized-Hessian uncertainty, including shared-rate uncertainty;
2. conditional peptide-weight uncertainty with shared rates fixed;
3. empirical and Ledoit-Wolf covariance of fitted peptide scores across locations.

The first two are model/residual fit uncertainty and the third is spatial/kinetic
heterogeneity. None is called experimental conformational covariance, cross-amide coupling,
or observation-noise covariance. Trajectory log-PF is explanatory only.

The method is calibrated first on the known ISO panels under average-first and frame-mixture
semantics. Noisy calibration uses unclipped Gaussian draws at fractional-uptake scales
`0.005`, `0.01`, and `0.02`, 100 deterministic seeds, and registered reconstruction,
coverage, Hessian-condition, physical-bound, and BI-versus-TRI discrimination gates. A
failed gate leaves the real-data result descriptive.

MoPrP then uses all 14 mapped peptides, with the production-compatible first 12 as a
sensitivity. Uniform AF2-Filtered and AF2-MSAss trajectories are compared in the common
experimental rate basis using raw curve RMSE and peptide-score Mahalanobis distance under
both average-first production semantics and frame-mixture sensitivity. No trajectory
weights, BV coefficients, production loss, or residue PFs are fitted.

Implementation:

    jaxent/src/analysis/hdx_rate_mixture.py
    jaxent/examples/2_CrossValidation/fitting/jaxENT/
        investigate_hdx_rate_mixture.py
    jaxent/tests/unit/analysis/test_hdx_rate_mixture.py
    jaxent/tests/unit/analysis/test_hdx_rate_mixture_stages.py

### Noiseless capacity result

The gate-aware runner first tested the best unregularized approximation available with
`K={1,2,3}`, 20 deterministic starts, all three ISO panels, and both average-first and
frame-mixture semantics. All 18 optimizations converged. Every unit selected `K=3`, but the
best RMSE remained `0.01696--0.01891`, against the registered `0.005` ceiling:

| Panel | Average-first RMSE | Frame-mixture RMSE |
|---|---:|---:|
| equal | 0.01862 | 0.01696 |
| random fixed | 0.01891 | 0.01740 |
| random variable | 0.01792 | 0.01756 |

The monotonic improvement from one to three components, finite objectives, and successful
optimizers identify a shared-basis capacity limit rather than a failed search. The proposed
three-rate representation cannot reconstruct even noiseless target curves accurately enough
to support Hessian coverage claims. The registered pipeline therefore stops before noisy
calibration and before MoPrP interpretation. The earlier reduced smoke comparison between
AF2 trajectories remains integration-only and carries no scientific direction.

Complete capacity outputs:

    jaxent/examples/2_CrossValidation/fitting/jaxENT/
        _hdx_rate_mixture_capacity/

Reconsideration requires changing the model class or relaxing the scientific target; merely
running more noise seeds cannot repair the noiseless approximation error.

### Exploratory component-capacity extension

Status: **K=6 is the smallest common capacity pass; optimization-stability gate passed
(2026-07-17)**.

An explicitly exploratory `K={4,5,6,7,8}` extension tested whether the rejection above was
specific to the registered three-rate ceiling. It was. `K=6` is the smallest component count
that reconstructs every ISO panel/semantics unit below the same `0.005` RMSE gate:

| Panel | K=6 average-first RMSE | K=6 frame-mixture RMSE |
|---|---:|---:|
| equal | 0.003019 | 0.001344 |
| random fixed | 0.002758 | 0.001545 |
| random variable | 0.002908 | 0.001352 |

`K=8` further lowers the six RMSEs to `0.000427--0.001338`, but adds parameters without being
needed for the registered capacity threshold. At `K=6` the prediction Jacobian has numerical
rank `96/96` under relative tolerance `1e-8`, yet effective rank is only about `38--43` and its
condition number is `4.26e3--8.19e4`. Thus formal numerical rank is not evidence that local
Hessian uncertainty is calibrated; the model remains strongly anisotropic.

Because several capacity fits reached their iteration limit, a separate audit ran 20 truly
independent, single-start `K=6` fits per unit at 3,000 maximum iterations. Near-optimal meant
RMSE within `max(1e-4, 10% of best)`; convergence meant optimizer success or final gradient
norm `<=1e-5`. All six units had `20/20` near-optimal and converged fits. Across those fits,
maximum log-rate SD was `0.010--0.040`, mean peptide-score SD was
`0.00023--0.00070`, mean prediction SD was `3.0e-5--5.6e-5`, and median flattened-score
correlation exceeded `0.99996`. The apparent capacity-fit optimizer issue was therefore an
iteration-limit artifact, not observed multi-start instability.

This result authorizes only the next **K=6 noisy ISO pilot**. It does not validate Hessian
coverage, authorize the full 100-seed calibration, or authorize MoPrP interpretation. The
pilot must now test whether joint and conditional Hessian intervals achieve registered
coverage under known injected noise despite the ill-conditioned Jacobian. Failure stops the
pipeline again; success permits the full registered calibration.

Complete exploratory outputs:

    jaxent/examples/2_CrossValidation/fitting/jaxENT/
        _hdx_rate_mixture_capacity_k4_8/
        _hdx_rate_mixture_stability_k6/

### K=6 noisy ISO pilot result

Status: **pilot failed the registered calibration gate; full calibration and MoPrP remain
stopped (2026-07-17)**.

The authorized pilot used `K=6`, chronological shrinkage selection, two starts per fit, ten
deterministic noise draws at `sigma=0.01`, and both semantics for every ISO panel. All six
noiseless capacity checks passed again. The selected shrinkage was `1e-3` in five units and
`1e-4` for equal/frame-mixture.

The three average-first noise conditions were excluded from coverage because the mean fraction
of deliberately unclipped Gaussian observations outside `[0,1]` was `0.0138`, `0.0191`, and
`0.0147`, exceeding the registered `0.01` physical-bound ceiling. Their reconstruction,
coverage, Hessian-validity, and BI-versus-TRI summaries were otherwise within the registered
ranges, but excluded conditions cannot count as passes.

All three frame-mixture conditions were physically valid. Random-fixed and random-variable
passed every pilot gate. Equal/frame-mixture failed only BI-versus-TRI discrimination: the
positive model beat TRI in `8/10` draws against the registered `9/10` requirement. Its mean
weight coverage was `0.874`, curve coverage `0.932`, RMSE to the clean curve `0.00688`, and all
ten Hessians were numerically valid. Across noisy conditions the Hessian condition numbers
were nevertheless large (condition means approximately `2.5e7--2.3e8`), reinforcing that
coverage must remain empirical rather than inferred from optimizer stability.

This is not evidence that the K=6 fit is unusable as a descriptive kinetic basis. It is evidence
that the current pilot did not establish the complete inferential claim under its frozen gates.
Increasing from ten to 100 seeds after observing `8/10`, or silently clipping/renormalizing the
noise to rescue the physical gate, would change the registered decision rule after seeing the
answer. The defensible next step is therefore a protocol decision: either retain this failed
pilot and keep Hessian/MoPrP interpretation stopped, or preregister a revised physical noise
model and an independent calibration run. No such revision is authorized by this result.

Complete pilot outputs:

    jaxent/examples/2_CrossValidation/fitting/jaxENT/
        _hdx_rate_mixture_pilot_k6/

## Physics-first MoPrP reconstruction (2026-07-17)

Status: **implemented; mean-only residue EX2 and descriptive kinetic tracks are now cleanly
separated**.

The earlier real-data canary used the production-derived `peptide_trim=2` map and rounded
times. A direct audit against the official exPfact validation bundle found three construction
differences that materially affect the physics:

1. exPfact predicts residue IDs `start+1 ... end`, dropping exactly one peptide N-terminal
   residue and excluding prolines; trim two changes the active-amide denominator for all 14
   peptides;
2. `moprp.list`, used for fitting, ends peptide 8 at residue 74, while the downstream
   clustering file `moprp.ass` ends it at 73;
3. the exact source clock begins at `0.0834` min and contains values such as `1.0002` min,
   rather than the rounded production header.

The new physical track implements

    D_p(t) = (1/N_p) sum_i [1 - exp(-k_int,i exp(-lnP_i) t)]

on explicit residue IDs. The canonical MoPrP intrinsic rates are the numeric output of the
official exPfact 3Ala calculation at 298 K and pH 4.4. Experimental pD 4 is stored separately.
The bundled HDXer feature rates remain a named sensitivity: their median ratio to the
canonical rates is `0.638`, with nonuniform special-residue differences, so PF coordinates
cannot be compared without rate provenance.

The source harmonic construction is also reproduced exactly. At lambda `1e-8`, the first
dropped residue is fitted as a smoothing-only boundary but retains zero uptake weight. The
optimizer keeps every finite multistart solution and never forms a residue-wise median across
modes. One neutral `lnP=5` reference start, the published median as a basin probe, and 18
source-style best-of-10,000 random initializations make up each 20-solution condition.

### Residue EX2 result

The best unregularized and source-harmonic fits have curve RMSE `0.039877` and objectives
`0.023853` and `0.023869`, consistent with the published exPfact cross-validation error scale.
The harmonic term reduces the median randomized objective from `0.1003` to `0.02558`, but the
median solution still places 16 represented residues at the lower log-PF bound. It stabilizes
the search without making residue PFs unique.

On the 27-residue NMR overlap, the best curve-fitting solution has Pearson/Spearman
`0.58/0.55` after excluding the two source-declared outliers at residues 91 and 94, and
`0.29/0.32` when all residues are retained under the published NMR rate convention. Both
versions are reported. NMR is a holdout diagnostic and is not used to choose a solution.

The published `median.pfact` and regional cluster centers are not coherent global forward
solutions. Seeding from the median converges to the same low-error basin as the neutral start,
while the fitted regional coordinates remain `1.13--2.17` log-PF units from the nearest
published centers. The cluster files are therefore treated as regional descriptive summaries,
not parity vectors or combinable whole-protein solutions.

### Peptide kinetic and trajectory results

Blocked-timepoint selection chooses a five-rate mixture with shrinkage `1e-4`; its training
curve RMSE is `0.01201`. This is a more flexible peptide curve embedding and its lower training
error does not make its component weights residue PFs. Outputs are limited to predicted curves,
exchanging amplitude, unresolved-slow fraction, and rate-distribution moments/quantiles. No
Hessian covariance is interpreted.

Residue 101 is an active amide in peptide 12 but is absent from both trajectory feature sets.
Trajectory scoring therefore uses the 13 fully represented peptides and records peptide 12 as
excluded; it does not renormalize away the missing amide. With canonical rates, average-first
curve RMSE is `0.2713` for AF2-Filtered and `0.2622` for AF2-MSAss. Stored HDXer rates worsen
these to `0.3121` and `0.3035`. Frame-mixture sensitivity improves each by only
`0.003--0.006`, so neither ensemble currently explains the experimental curves under the
fixed BV coefficients.

For peptide 1, the exact pre-quench Poisson-binomial distribution of exchanged amide counts is
emitted at 1 min, 1 h, and 24 h. Its mean is regression-tested to equal the centroid uptake. At
1 h the best experimental EX2 fit concentrates on 3--4 exchanged amides, whereas both BV
average-first predictions concentrate on 1--2.

The official fully protonated spectrum is used as the natural-isotope/instrument baseline. A
single effective deuterium-survival probability is fitted only to the fully deuterated control,
then frozen for every exchange-time spectrum and every candidate model. The calibrated survival
is `0.4981`; the held-out fully deuterated control is reconstructed with RMSE `0.01295` and
`R^2=0.9868`. After binomial quench thinning and convolution with the protonated baseline, the
20 residue-EX2 solutions span envelope `R^2` ranges of `0.991--0.996`, `0.978--0.997`, and
`0.485--0.973` at 1 min, 1 h, and 24 h. The near-optimal centroid solutions remain at about
`0.995`, `0.997`, and `0.973`; the low envelope scores belong to poorer randomized curve fits.

The best BV envelope results are `0.906`, `0.399`, and `0.774` at the same three times, all from
MSAss under canonical rates (frame-mixture at 1 min/1 h and average-first at 24 h). Thus the
independent envelope shape supports the residue-EX2 construction and rejects the present fixed-
coefficient BV predictions most strongly at 1 h. This is still an effective control-calibrated
quench model: it does not claim residue-specific back-exchange kinetics or fit an instrument
line shape to the exchange spectra.

Implementation and canonical outputs:

    jaxent/src/analysis/hdx_ex2.py
    jaxent/examples/2_CrossValidation/fitting/jaxENT/
        investigate_moprp_ex2_physics.py
        _moprp_ex2_physics/
    jaxent/tests/unit/analysis/test_hdx_ex2.py

The absence of raw triplicate and control-level measurements remains binding. Multistart ranges
are sensitivity ranges, not confidence intervals, and no calibrated experimental PF covariance
is claimed.

## BV contact-parity correction and all-peptide rerun (2026-07-21)

Status: **implemented; the corrected fixed-coefficient BV comparison fails more strongly under
the literature-parity contact construction**.

The earlier trajectory audit exposed a second, independent construction problem. The generic
topology adapter excluded both protein termini, although the experimental HDX convention drops
the peptide/protein N-terminal amide but retains an exchangeable C-terminal backbone amide. It
therefore omitted residue 101 and made peptide 12 impossible to score. Terminal handling is now
an explicit chain-position policy (`none`, `n`, `c`, or `both`); the MoPrP BV path uses N-only
exclusion and represents 97 residues over all 500 frames, including residue 101. Prolines remain
excluded separately.

Contact construction was also made explicit. The canonical BV comparison now uses binary
protein-heavy contacts within 6.5 Angstrom of the amide N and binary protein-oxygen contacts
within 2.4 Angstrom of the amide H, excluding ordinal sequence neighbours -2 through +2 within
the same chain. This avoids two easy errors: treating numeric residue-ID differences as sequence
distance and masking identically numbered residues in other chains. The previous rational
switched contacts are retained only as a named sensitivity. On the 96 shared rows, regenerated
switched features reproduce the untouched legacy features exactly; this isolates the new
C-terminal row from the contact-definition change.

Intrinsic rates are now joined to feature rows by exact `(chain, residue)` identity. Missing,
duplicate, nonfinite, or nonpositive active rates fail immediately. The canonical run uses the
official exPfact 3Ala rates; the shorter historical HDXer rate vector remains a labelled
13-peptide sensitivity and cannot silently displace the residue map.

With all 14 peptides and canonical rates, the curve RMSE results are:

| ensemble | contact definition | average-first | frame-mixture |
| --- | --- | ---: | ---: |
| AF2-Filtered | binary hard count | 0.3680 | 0.3624 |
| AF2-MSAss | binary hard count | 0.3596 | 0.3483 |
| AF2-Filtered | rational switched sensitivity | 0.2771 | 0.2746 |
| AF2-MSAss | rational switched sensitivity | 0.2683 | 0.2640 |

Residue 101 restores peptide 12 as a valid comparison, but does not rescue it: its hard-count
RMSE is `0.4350` for AF2-Filtered and `0.4330` for AF2-MSAss under average-first semantics. The
peptide-1 isotope-envelope control also worsens under hard counts. The best hard-count envelope
R-squared values at 1 min, 1 h, and 24 h are `0.878`, `-0.585`, and `0.045`, compared with about
`0.996`, `0.997`, and `0.973` for the best retained experimental EX2 solution.

The physics interpretation is narrower than “the trajectories are wrong.” With the published
fixed coefficients, binary counts create substantially greater protection than the switched
construction and the trajectories exchange too slowly. This rejects the present combination of
contact convention, coefficients, and ensembles; it does not by itself identify whether the BV
coefficients are non-transferable to this implementation or whether both ensembles are too
protected. The switched result was an optimistic construction sensitivity, not literature
parity. No trajectory reweighting, coefficient fitting, Hessian covariance, or PF covariance is
authorized by this audit.

Versioned implementation and outputs:

    jaxent/examples/2_CrossValidation/fitting/jaxENT/
        featurise_CrossVal_MSAss_Filtered.py
        _featurise_physics_v2/
        investigate_moprp_ex2_physics.py
        _moprp_ex2_physics_bv_v2/

The output audit includes peptide/time residuals and a construction-attribution table, so future
work can separate terminal mapping, intrinsic-rate provenance, contact definition, and averaging
semantics rather than interpreting their aggregate difference as conformational evidence.
