# Checkpoint 35: original OMC Laplacian — cheap pre-ISO evidence

## Purpose and protocol

This preregistered ATLAS screen asks whether the original, weight-weighted OMC Laplacian can
recover a rare structure carrying high target population and whether its bandwidth controls
ensemble diversity without sacrificing recovery. It uses 6 systems, replica 1, 128 evenly spaced
frames, one uniform MD prior, and target basin masses 0.30, 0.50 and 0.70. No ISO data are read.

The existing ISO RBF litmus protocol locks the prior-relative penalty at bandwidth quantile 0.16.
This checkpoint tests a different objective. A pass does not validate that protocol as written;
it motivates a revised downstream protocol whose regulariser, bandwidth rule and gates are
derived from these results. It does not authorise ISO deployment.

## Objective and known hazards

For fixed structural distances, the primary arm uses

```text
S_ij     = exp(-d_ij^2 / (2 sigma^2))
S_w,ij   = S_ij w_i w_j
R_omc(w) = N^2 * 0.5 * sum_ij S_w,ij (w_i - w_j)^2.
```

The `N^2` factor makes the energy invariant when every frame is duplicated at half weight. It is
a fixed-ensemble strength rescaling and does not alter the optimum. The secondary `omc_norm[q]`
arm divides this result by `sum_ij S_ij w_i w_j`. That denominator depends on fitted weights, so
this arm is a different method rather than a rescaling.

The strict-positive indicator in the paper cannot fire under softmax. It is omitted: `w_i w_j`
already removes near-zero frames continuously. A stop-gradient mask at `w > tau/N` remains a
possible follow-up and is not run here.

The energy vanishes for uniform weights, a point mass, and weights uniform on any subset. Softmax
can approach all these boundary solutions. `ESS < 2` catches little of this behavior. Every fit
therefore records support size at `w >= 1/(10N)`, a plateau flag when the coefficient of variation
on that support is below 0.05, recovered basin mass, and structural dispersion
`sum_ij w_i w_j d_ij`. ESS describes concentration only and must be interpreted with recovery and
dispersion. The direction of the sigma-to-ESS relationship is not preregistered; G4 uses `|rho|`.

## Challenge construction and eligibility

The explicit `screen` phase examines all 24 systems in the established pilot cohort. On replica 1,
after the 128-frame `linspace` subsample, KMeans is fitted to rows of the pairwise structural W1
matrix for 8, 10, 12, 14 and 16 clusters (`n_init=20`, fixed configuration seed). A cluster is
eligible when its natural mass is 0.06–0.15, it has at least 8 frames, and its mean within-cluster
distance divided by the ensemble mean pairwise distance is below 1.0. The eligible cluster with
smallest compactness is selected, with deterministic ties on cluster count and index.

Systems without an eligible cluster are written to `exclusions.parquet`. Qualifying systems are
sorted by sequence length and every `len // 6`-th system is taken until six are selected. If fewer
than six qualify, the mass band is widened once to 0.05–0.20. Fewer than six after that makes the
checkpoint inconclusive on cohort grounds. The selected cohort is stored in `screen/cohort.yaml`
and must be copied into Results before the pilot.

For each selected basin the prior is uniform. Truth assigns absolute mass `m` uniformly across
basin members and `1-m` uniformly outside it, for `m = 0.30, 0.50, 0.70`. Construction asserts at
least twofold enrichment. Basin selection is BV-blind. Observables are pseudo-uptake from the fixed
protocol values `bv_bc=0.35`, `bv_bh=2.0`; targets are noiseless `flat @ truth`. A deterministic
60/20/20 split is made over flattened observables.

## Arms and fitting

There are 20 arms: MaxEnt; `omc[q]`, `omc_rewired[q]`, and `omc_norm[q]` for
`q = 0.02, 0.04, 0.08, 0.16, 0.32, 0.64`; and the prior-relative incumbent
`prior_rbf_locked` at 0.16. Work Scale distances define every graph. Sigma is the stated quantile
of positive pairwise distances. Rewiring applies one seeded row-and-column permutation, preserving
the distance multiset while destroying frame correspondence.

Strengths are `0, 1e-3, 1e-2, 1e-1, 1, 10`. Every arm-strength result is persisted; selection
occurs only during analysis. Parameters are 128 frame logits initialized to `log(1/128)` and
mapped through softmax. Fixed BV observables and distances mean no BV parameter is fitted. The data
term is training MSE divided by training-target variance plus `1e-8`. Adam uses learning rate 0.05
for 300 steps. Dense arms are vmapped under one summed batch objective; independent trajectories
share no fitted parameters. MaxEnt and the incumbent use the unchanged checkpoint optimizer.

## Measurements and diagnostics

One row is written for each `(system, target mass, arm, sigma quantile, strength)`: 2160 rows for
the preregistered pilot. Rows contain identifiers and challenge audit fields; sigma and strength;
train, validation, and test MSE; recovered basin mass and absolute target error; weight TV and the
ISO `(1-sqrt(JSD))*100` recovery score; overall and target-conditional ESS; support size and plateau
flag; structural dispersion; final gradient norm and objectives at steps 150 and 300; and an
asserted simplex-valid flag.

Convergence is assessed using relative objective change from step 150 to 300 and final gradient
norm. One system is rerun for 1000 steps; if its verdicts change, optimizer length is the finding.
Recovery gain is binned by MaxEnt target error to expose lack of headroom. More than 25% selected
strengths at 0 or 10 marks a mis-centred grid. Negative gate results are interpreted only after
these diagnostics are clean.

The analysis unit is `(system, challenge)`, 18 units. Intervals resample six systems with
replacement, keeping their challenges together, for 10,000 percentile-bootstrap replicates.
Exact sign tests use the six per-system median effects. The minimum nonzero two-sided p-value is
`2/64 = 0.031`; this pilot is a screen and is not powered for a confident interval.

## Gates and stop rule

- **G1 Recovery:** jointly select primary OMC bandwidth and strength by validation MSE. For each
  unit compute MaxEnt target-mass error minus selected-OMC error. Pass when the clustered median
  interval excludes zero and the exact sign test on system medians has `p <= 0.05`.
- **G2 Specificity:** at the selected physical bandwidth, select the rewired strength by validation
  MSE. Pass when rewired error minus physical error has positive median and interval excluding zero.
- **G3 Non-inferiority:** pass when median selected-OMC held-out test MSE is no more than 1% worse
  than selected MaxEnt.
- **G4 Useful diversity control:** within each unit and strength, restrict bandwidths to target
  error no worse than MaxEnt and test MSE within 1%. Across acceptable bandwidths compute Spearman
  `|rho|` and max/min ratio for ESS fraction. Pass when median `|rho| >= 0.7`, its clustered
  interval excludes 0.5, and median ratio is at least 2. Report the same summaries for target ESS
  and structural dispersion. Every strength remains visible.
- **G5 Comparators:** descriptively compare primary OMC with `prior_rbf_locked` and `omc_norm` on
  target error, recovery score, and test MSE.
- **G6 Numerical validity:** require finite simplex weights. Report plateau, support-size, and
  uniform-subset-degeneracy distributions.

G1–G4 all passing yields a screen pass and permits writing a revised downstream ISO protocol.
Any poor convergence, absent recovery headroom, boundary-saturated strength grid, or cohort below
six yields `inconclusive` and a specific cheap follow-up. Missing gates with clean diagnostics yield
`fail`: close the mechanism without extending to 24 systems or running ISO. A six-system pass is
never confirmatory.

## Commands

```bash
uv run --no-sync pytest jaxent/tests/unit/opt/test_original_omc_laplacian.py -q
uv run --no-sync pytest tests/test_atlas_original_omc_checkpoint35.py -q
./jaxent/examples/ATLAS_BV/commands.sh geometry-original-omc \
  --smoke --limit 1 --frame-cap 32 --steps 20 --target-masses 0.5 --strengths 0,0.1
./jaxent/examples/ATLAS_BV/commands.sh geometry-original-omc --phase screen
./jaxent/examples/ATLAS_BV/commands.sh geometry-original-omc --phase pilot
./jaxent/examples/ATLAS_BV/commands.sh geometry-original-omc \
  --phase pilot --limit 1 --steps 1000 --output-suffix steps1000
```

## Results

The eligibility screen completed before pilot fitting. All 24 pilot-cohort systems qualified under
the preregistered 0.06–0.15 natural-mass band, so the widening contingency was not used. The
deterministic length-stratified cohort is pinned as:

```text
4xo1_A, 3fbl_A, 4ndt_A, 1yoz_B, 2w86_A, 4o6g_A
```

The preregistered pilot completed all 2,160 rows (6 systems × 3 challenges × 20 arms ×
6 strengths). Every fitted vector was finite and simplex-valid. The selected primary OMC fits had
median target-mass error 0.0240, median recovery 95.86%, median test MSE `1.12e-5`, and median ESS
fraction 0.315. Their median support size was 128 (range 124–128).

The measured gates were:

| gate | raw result | measured statistic |
|---|---|---|
| G1 recovery | fail | median error gain 0.000393; clustered 95% interval -0.000000134 to 0.001122; sign-test p=0.6875 |
| G2 specificity | fail | median rewired-minus-physical error 0; interval 0 to 0 |
| G3 non-inferiority | fail | median relative test-MSE change +11.32% |
| G4 diversity control | fail | median acceptable-bandwidth `|rho|` 0 (interval 0–0); median ESS ratio 1 (interval 1–1) |
| G6 numerical validity | pass | all rows finite and simplex-valid; plateau fraction 2.22%; median support size 125 across all fits |

These are raw gate results because the diagnostic stop rule supersedes them. Seventeen of the 18
selected primary fits chose strength 0; the boundary-selection fraction was 94.4%, versus the 25%
limit. Median relative objective change from step 150 to 300 was 18.8%, while median final gradient
norm was `2.44e-4`. The required 1,000-step sensitivity run on `1yoz_B` retained strength 0 and
bandwidth quantile 0.02 for all three challenges. Its median target-mass error fell from 0.0179 to
0.00243, showing continued improvement of the unregularised fit, while the checkpoint verdict and
selected regularisation configuration remained unchanged.

For context, validation-selected median `(target error, recovery %, test MSE)` was
`(0.0256, 96.57, 9.73e-6)` for MaxEnt, `(0.0240, 95.86, 1.12e-5)` for primary OMC,
`(0.0243, 95.86, 1.04e-5)` for rewired OMC, `(0.0243, 95.86, 1.04e-5)` for normalised OMC, and
`(0.0214, 96.69, 7.12e-6)` for the locked prior-relative comparator. The six systems' median
MaxEnt target errors ranged from 0.00983 to 0.105, so recovery headroom was not uniformly absent.

## Decision

**Inconclusive.** The 94.4% boundary-strength selection triggers the preregistered stop rule. The
raw gate misses therefore do not establish that the OMC mechanism is ineffective. The cheap
follow-up is a strength refinement between 0 and `1e-3` (for example `0, 1e-7, 1e-6, 1e-5,
1e-4, 1e-3`) with 1,000 optimizer steps, retaining the same cohort, challenges, validation
selection, and diagnostics. No 24-system extension or ISO test is authorised.
