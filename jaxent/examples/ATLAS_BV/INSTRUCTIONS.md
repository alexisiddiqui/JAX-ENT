# ATLAS BV runnable sequence

## Pairwise geometry measurement (current redesign)

The current Stage 1 redesign measures fixed-BV protection-factor geometry directly against
pairwise C-alpha RMSD, without structural basins, PC1 occupancy bins, or coefficient fitting:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-stage1 --limit 1
./jaxent/examples/ATLAS_BV/commands.sh geometry-stage1 --workers 2
```

The first command is the required one-system smoke run. The full command uses all 111 systems and
three leave-one-replica-out folds. It compares normalized L1/L2 distances for raw, frame-centred,
residue-standardised, and combined representations. Results are written under
`outputs/analysis/pairwise_geometry/`; `stage1_geometry_measurement.yaml` is explicitly
measurement-only and never authorizes Stage 2.

The full 111-system measurement completed on 2026-08-27. Median held-out skill over a
training-median RMSD predictor was `0.2105` for raw L1 and `0.2381` for raw L2. Frame centring gave
`0.2339`/`0.2403`; residue standardisation reduced skill. The compactness controls gave `0.0518`
for Rg and `0.0273` for native contacts. These are observations, not a model-selection decision.
Conditional 90% interval coverage was about 84%, so the conditional calibration is informative but
not fully calibrated.

Checkpoints 1 and 2 of the support/W1 follow-up are complete. See
`CHECKPOINT1_SUPPORT_W1.md`, `CHECKPOINT2_BOUNDARY.md`, and their corresponding directories under
`outputs/analysis/pairwise_geometry/`. Checkpoint 2 finds that common-PF-support filtering and
continuous endpoint extrapolation do not repair the long-distance undercoverage. The earlier
Checkpoint 3 placeholder (three-way conformal calibration) has now been completed as Checkpoint 8
after the vector and support diagnostics described below.

Checkpoint 3A of the vector-degeneracy experiment is complete. See
`CHECKPOINT3A_VECTOR_SUPPORT.md` and
`outputs/analysis/pairwise_geometry/checkpoint3_vector/`. Fold isolation, per-residue feature
support, deterministic kNN caps, and capped scalar baselines are validated. Ridge/PCA-ridge is
complete; vector modelling now precedes three-way conformal calibration.

Checkpoint 3B is complete. See `CHECKPOINT3B_VECTOR_RIDGE.md`. Full-pair raw ridge improves both
global-RMSD and W1-q5 distribution recovery over Absolute-L1; PCA and z-scoring do not add further
recovery.

Checkpoint 3C is complete. See `CHECKPOINT3C_VECTOR_KNN.md`. Exact complete-vector kNN point
predictions do not beat the capped Absolute-L1 baseline. Neighbour conditional mixtures do improve
global-RMSD and W1-q5 probability-distribution recovery, supporting a per-residue conditional
likelihood model.

The final paired familywise comparison is complete. See `FINAL_VECTOR_COMPARISON.md`. Raw
per-residue ridge is the selected point model, improving recovery by 5.43 percentage points for
global RMSD and 7.67 points for W1 q5 over matched Absolute-L1 baselines; both improvements survive
Holm correction. PCA and z-scoring add no benefit, scalar alternatives and kNN point prediction do
not improve the baseline, and conditional kNN mixtures remain a separately supported bridge to a
per-residue likelihood model.

Checkpoint 4 is complete. See `CHECKPOINT4_CONDITIONAL_LIKELIHOOD.md`. A replica-isolated
heteroscedastic ridge likelihood raises conditional-distribution recovery to 75–76% for global RMSD
and 84–86% for W1 q5. Z-scoring improves W1-q5 fit but worsens its nominal-90% coverage from 79.6%
to 74.4%. Almost every variance fit selects maximum regularization, so the next calibration step
should model scale nonlinearly from the cross-validated predicted structural mean rather than from
a second full per-residue linear vector.

Checkpoint 5 is complete. See `CHECKPOINT5_SCALE_CALIBRATION.md`. Predicted-mean scale binning
raises W1-q5 recovery to about 87.5%, but worsens empirical 90% coverage to 67–68%. The predicted
mean hides novel tail structures when it underpredicts them, so this calibration arm is rejected.
The next diagnostic must condition scale on PF-vector support/novelty in addition to predicted mean.

Checkpoint 6 is complete. See `CHECKPOINT6_PF_NOVELTY.md`. Monotone uncertainty calibration from
training-standardized radial PF-vector novelty worsens W1-q5 coverage to about 52%, despite retaining
high pooled recovery. Radial support inflation is therefore rejected. Exact directional
nearest-training-vector distance is the remaining stronger support diagnostic before declaring the
fixed BV tail failure representational.

Checkpoint 7 is complete. See `CHECKPOINT7_NEAREST_SUPPORT.md`. Exact directional nearest-PF
distance provides a modest matched W1-q5 improvement over constant scale (about +3.5 coverage
points and improved NLL), but leaves severe tail undercoverage. Support calibration is therefore a
secondary correction. The completed Checkpoints 3–7 indicate that fixed BV PF features retain
useful residue-aware distribution signal but do not transfer enough information to calibrate the
magnitude of large structural departures across replicas.

Checkpoint 8 is complete. See `CHECKPOINT8_STRICT_CONFORMAL.md`. In every system, each of the six
ordered replica assignments uses A only to fit, B only to calibrate finite-sample conformal
residuals, and untouched C only to test. Common-support marginal coverage is approximately nominal
(`91.4%` for RMSD and `91.6%` for W1 with per-residue ridge), proving that calibration works
globally. It does not repair the large-displacement tail: common-support W1-q5 coverage is `60.0%`
(95% system-bootstrap CI `54.5–65.3%`), or `70.1%` with conditional Mondrian calibration. The
historical same-fit value was `34.5%`. Per-residue ridge improves W1-q5 coverage by a paired median
`19.1` percentage points and recovery by `7.8` points over scalar Absolute-L1, yet remains far below
the nominal 90% target. Under the predeclared decision rule this resolves the remaining ambiguity as
a fixed-BV representation limit for large structural departures, not a purely methodological
calibration failure.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-conformal-strict
./jaxent/examples/ATLAS_BV/commands.sh geometry-conformal-compare
```

Checkpoint 9 is complete. See `CHECKPOINT9_STRICT_LIKELIHOOD.md`. The strict Gaussian conditional
baseline reaches `83.1%` common-support W1-q5 recovery with marginal calibration but only `60.0%`
coverage. Mondrian calibration raises coverage to `70.1%` while lowering recovery to `81.3%`.
A-only z-scoring is numerically indistinguishable and is not promoted. The declared joint
fit/calibration gate remains closed, so the next reviewed checkpoint is the opening-probability and
naive-distance screen.

Checkpoint 10 is complete. See `CHECKPOINT10_OPENING_SCREEN.md`. Converting fixed-BV log-PF to
`p_open = expit(-logPF)` loses rather than restores transferable W1-tail information. The honest
A-selected opening pipeline reaches only `40.5%` common-support W1-q5 recovery and `25.2%` Mondrian
coverage, losing `11.2` recovery points against raw log-PF vector ridge. L1/L2, cosine,
correlation, Bernoulli sqrt(JSD), symmetric KLD, mean centring and A-only z-scoring all fail to beat
their matched log-PF baselines. Stage 3 should therefore retain raw log-PF ridge as primary while
testing raw opening changes only as the prespecified physical comparison.

Checkpoint 11 is complete. See `CHECKPOINT11_CONDITIONAL_LIKELIHOOD.md`. Strict raw log-PF ridge
Gaussian remains best at `83.1%` common-support W1-q5 recovery and `70.1%` Mondrian coverage.
Opening Gaussian falls to `79.1%`/`21.8%`; raw log-PF and opening kNN mixtures reach only
`50.7%`/`54.2%` and `46.9%`/`38.5%`, respectively. The earlier local-mixture gain does not survive
when A alone supplies neighbours and B is reserved for calibration. Stage 3 therefore fails its
joint fit/calibration gate. Proceed to A-only low-rank joint log-PF modes; run contact-community
features only if the low-rank trigger passes.

Checkpoint 12A is complete. See `CHECKPOINT12_JOINT_LOWRANK.md`. A-only PCA joint modes worsen both
fit and calibration: low-rank Gaussian reaches `81.7%` common-support W1-q5 recovery and `51.5%`
Mondrian coverage, a paired `-2.32` recovery-point change versus raw per-residue Gaussian (95% CI
`-3.04` to `-1.14`). Low-rank kNN reaches only `49.7%`/`52.4%`. The declared `+2`-point
contact-community trigger is therefore closed, so contact blocks are skipped. The remaining
contingent stage is system-specific BV coefficient refitting with strict replica isolation.

Checkpoint 13 is complete. See `CHECKPOINT13_BV_REFIT.md`. Strict per-system coefficient refitting
raises common-support W1-q5 recovery by only `0.13` percentage points (95% CI `0.03–0.32`) to
`83.24%`, while Mondrian coverage remains `72.38%`. In 97.60% of ordered assignments at least one
coefficient hits the search boundary, most commonly `bc×2, bh×0.5`; this is not an identified new
BV parameter set. The final fixed-BV gate fails. The staged geometry investigation is complete;
any continuation should introduce new forward information such as M8 SASA/polar-distance features,
not further recalibration of the same contact/acceptor representation.

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-bv-refit --workers 4
./jaxent/examples/ATLAS_BV/commands.sh geometry-bv-refit-report
```

The later population track uses an MD structural-W1 KDE target rather than structural-distance
prediction. Checkpoints 17 and 18 contain the full fixed-BV and thermodynamic metric runs. The
checkpoint-21 pilot adds raw/mean-centred PF-W1 and replica-A variance-scaled information distances:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-pf-information-pilot
```

No checkpoint-21 candidate passes the q4/q5 expansion gate. Work Scale remains the strongest
standalone predictor through q4; neither PF-W1 nor variance scaling improves its paired q5 result.
Direct structural W1 and periodic backbone phi/psi dRMSD controls also underperform Work Scale.
Circular z-scoring improves the squared dRMSD control modestly relative to raw dRMSD but remains far
below Work Scale in the tail.
See `CHECKPOINT21_PF_INFORMATION_PILOT.md` for the definitions and complete tail comparison.

Checkpoint 22 stratifies the same MD structural-W1 population target by A-only structural clusters:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-cluster-stratified
```

The 12-system pilot finds higher provisional Work Scale q5 recovery within clusters (69.8%) than
between clusters (50.4%), but only two and five systems respectively support those cells. The
tail-aware 8/12 expansion gate therefore fails; the 111-system run is intentionally not launched.
See `CHECKPOINT22_CLUSTER_STRATIFIED.md` for all definitions and the density-clustering audit.

Checkpoint 23 adds an independent OpenMM CHARMM36 protein-vacuum energy control. The disposable
OpenMM environment remains separate from JAX-ENT:

```bash
./jaxent/examples/ATLAS_BV/commands.sh openmm-env
./jaxent/examples/ATLAS_BV/commands.sh openmm-audit
./jaxent/examples/ATLAS_BV/commands.sh openmm-score --pilot --platform OpenCL
./jaxent/examples/ATLAS_BV/commands.sh geometry-openmm-energy
```

All 111 topologies pass. In the 12-system pilot, fitted total/torsional/nonbonded vacuum-energy
differences reach roughly 58–62% q5 recovery, while direct 300 K Boltzmann predictions reach only
19–24%. See `CHECKPOINT23_OPENMM_VACUUM.md`.

Checkpoint 24 tests PyRosetta `ref2015` and `ref2015_cart` scores on the same raw coordinates and
population target. It reuses the existing `neuralplexer_dev` Python 3.10 environment and runs
independent systems in parallel:

```bash
./jaxent/examples/ATLAS_BV/commands.sh pyrosetta-audit
PYROSETTA_WORKERS=6 ./jaxent/examples/ATLAS_BV/commands.sh pyrosetta-score-parallel
./jaxent/examples/ATLAS_BV/commands.sh geometry-pyrosetta-energy
```

All 111 systems pass the mapping audit. In the five-system q5 pilot subset, fitted `ref2015_cart`
reaches 69.0% median recovery and its Cartesian-bonded component reaches 70.0%, compared with 52.8%
for Work Scale and 59.2% for OpenMM total energy. Raw unit-scale REU differences remain poor and
are not physical Boltzmann predictions. See `CHECKPOINT24_PYROSETTA_ENERGY.md`.

Checkpoint 25 compares each fitted system alpha with the variance of its exact unscaled replica-A
pair predictor:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-alpha-variance --workers 6
```

Alpha is strongly inversely rank-correlated with predictor variance for Work Scale (rho=-0.86),
legacy Work Density (-0.85), `ref2015` total (-0.98), and `ref2015_cart` total (-0.93).
The completed 111-system OpenMM total-energy cohort gives rho=-0.56. Correlations with MD target
variance are weak for the thermodynamic and Rosetta metrics, whereas OpenMM is moderately positive
(rho=0.56). See `CHECKPOINT25_ALPHA_VARIANCE.md`.

Checkpoint 26 tests whether structural-W1 graph geodesics can improve standard `ref2015` energy
differences without rescoring trajectories:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-pyrosetta-graph --pilot --workers 4
```

The predefined 24-system pilot gate fails. Energy-weighted and uphill paths gain about 6 paired
recovery points at q0 but lose 13--14 points at q5 and do not separate cleanly from geometry-only or
energy-shuffled controls. Signed fits collapse to alpha=0 in 7/24 systems. Work Scale remains the
best balanced inexpensive predictor, while direct `ref2015` is preferable to the graph variants.
A subsequently requested full 111-system run confirms this: direct `ref2015` magnitude recovery is
71.5%/61.9% at q0/q5, versus 73.5%/33.2% for energy-weighted W1 and 89.2%/59.2% for Work Scale.
See `CHECKPOINT26_PYROSETTA_GRAPH.md`.

Checkpoint 27 repeats the graph construction using Work Scale, Work Shape, and legacy Work Density:

```bash
./jaxent/examples/ATLAS_BV/commands.sh geometry-work-graph --workers 6
```

Across all 111 systems, graph transformation degrades Work Scale from 89.2%/59.2% q0/q5 recovery
to at best 79.6%/45.9%. Shape and legacy Density show large local q0 gains, but their paths converge
toward the 69.8%/35.6% geometry-only control and lose or fail to improve q5. Direct Work Scale
therefore remains the best balanced Work predictor. See `CHECKPOINT27_WORK_GRAPH.md`.

## Historical occupancy sequence

Run from the repository root:

```bash
./jaxent/examples/ATLAS_BV/commands.sh benchmark
./jaxent/examples/ATLAS_BV/commands.sh featurise --workers 2
./jaxent/examples/ATLAS_BV/commands.sh basin-census --workers 2
./jaxent/examples/ATLAS_BV/commands.sh stage1
```

The historical Stage 1 command runs the `within_basin_pc1_v1` redesign on each shared dominant basin and writes
`outputs/analysis/stage1_decision.yaml`. It uses structural PC1 bins defined on two replicas and
evaluated on the third. Do not run Stage 2 after a failed gate; `commands.sh stage2` enforces this.
Absolute-L1 is the primary coordinate. The decision report contains separate Signed-L1,
Absolute-L1, and L2 population block-permutation results; only the Absolute-L1 gate controls Stage 2.
Replica disagreement is diagnostic uncertainty and is not an exclusion.

Current result: Absolute-L1 median held-out rho `0.279`, population-null `p=0.000999`, expected
negative slope in `96.9%`, and `52.3%` beating the strongest compactness control. The final criterion
misses its declared `60%` threshold, so Stage 2 correctly refuses to run.

The predeclared gate remains closed. A user-authorized exploratory continuation must be explicit;
it does not alter `stage1_pass` or `stage2_authorized`. Benchmark before the full Stage 2 fit:

```bash
./jaxent/examples/ATLAS_BV/commands.sh stage2 --exploratory-override --benchmark
./jaxent/examples/ATLAS_BV/commands.sh stage2 --exploratory-override
```

Stop after the benchmark and obtain explicit approval for its projected full-run cost. The primary
Stage 2 decision is whether the one-sided 95% paired system-bootstrap lower bound for
`rho_fitted-rho_default` is above zero. PMF MAE and the strongest compactness coordinate remain
secondary diagnostics.

The approved exploratory run completed on 2026-08-27: 65/65 systems and 65,000/65,000 bootstrap
draws. Median paired held-out delta rho is `-0.0162` (one-sided 95% lower bound `-0.0342`), so the
predictive criterion fails. The fitted PMF MAE falls from median `23.04` to `0.246 kcal/mol`, but
69.2% of pooled fits lie on a coefficient boundary; this is scale collapse, not improved ordering.

The independent convergence diagnostics can be generated with:

```bash
./jaxent/examples/ATLAS_BV/commands.sh convergence --workers 2
```

They are optional diagnostics for the within-basin analysis and are not an exclusion gate.
