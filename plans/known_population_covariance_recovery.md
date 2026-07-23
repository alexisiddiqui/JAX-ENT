# Investigation: known-population MoPrP covariance recovery and the population-free covariance-shape prior

Status: **Stages A–I complete; former Stage J failed its ISO gate and is closed; replacement
HDX target-variance inference is implemented and awaits full TeaA/ISO qualification**
Created: 2026-07-21
Parent: `plans/hdx_heteroscedastic_nll_investigation.md` (this is the Stage-2 follow-on that was
"pending explicit user approval")

## Context and question

The parent investigation established a hard negative: **covariance across uptake timepoints is not
conformational covariance**, and nothing extractable from an experimental uptake curve is a
defensible PF-variance target (frame-permutation identifiability, rank-≈1 curve covariance,
negative-control-beats-observables). The standing recommendation, *marginal PF variance*
(`M C_logPF Mᵀ`), stayed synthetic-only because real MoPrP has no independent experimental
covariance target.

This investigation changes the target instead of trying to extract it:

- the **candidate ensemble frames** are the prior that already encodes covariance geometry;
- a **known NMR-derived state population** (`state_ratios.json`: Folded 0.9712, PUF1 0.0236,
  PUF2 0.0052, all other states 0) is controlled ground truth;
- reference weights `w_NMR` spread each state's target mass uniformly among its frames; zero-target
  frames stay as decoys;
- target `C(w_NMR)` and prediction `C(w)` come from the **same** frames at different weights →
  symmetric matching, zero loss at `w = w_NMR`.

Primary question: does ensemble covariance carry enough information to recover the known population
while fitting the mean HDX, and reject decoys? Follow-on question (Stages C–H): can the covariance
target be made **population-free** — a regularising *prior* we project the data through — so the
method works without knowing the population?

## Verified data reality (both ensembles well-posed)

Per-frame state labels: `analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters/global_frame_to_cluster_ensemble.csv`
(the `_test` clustering is current; the non-`_test` one is stale/coarser). `state_mapping`
(config.yaml): 0=Folded 1=PUF1 2=PUF2 3=PUF3 4=unfolded 5=PUF2-like.

| State | Target | AF2-Filtered frames | AF2-MSAss frames |
|---|---|---|---|
| Folded | 0.9712 | 424 | 392 |
| PUF1 | 0.0236 | 66 | 72 |
| PUF2 | 0.0052 | 9 | 10 |
| decoys (target 0) | 0 | 1 (PUF2-like) | 26 (PUF3 + unfolded) |

Features: corrected physics-v2 hard-count contacts, canonical exPfact-3Ala rates,
`fitting/jaxENT/_featurise_physics_v2/features_AF2_{MSAss,filtered}_hard.npz` (97 residues × 500
frames, incl. residue 101). Per-frame residue log-PF = `Bc·heavy + Bh·acceptor`. All 14 peptides
via the trim-one exPfact map; peptide 1 is held out of calibration, CV, and selection throughout.

## Coefficient settings (frozen)

Shared non-negative `(Bc, Bh)` fit across both ensembles at `w_NMR`, peptide 1 excluded, average-first:
- `published` = (0.35, 2.0), MSE 0.127 — **dropped** (standard-condition calibration, not pD 4);
- `constrained_optimum` = **(0.229, 0)**, MSE 0.041 — free optimum, `Bh→0` boundary reported as
  *model inadequacy*, not a physical estimate;
- `scaled_published` = **(0.186, 1.064)** (published direction × s=0.532), MSE 0.045 — keeps `Bh` and
  the published ratio, matches the target mean-uptake scale. **Primary physical setting.**

Runner: `fitting/jaxENT/moprp_coefficient_lock.py`.

## Stage A — population-space identifiability oracle (diagnostic)

`fitting/jaxENT/moprp_population_oracle.py`. Optimising only state populations (covariance-only, no
mean/KL). **Finding:** the known population is the unique lowest-loss minimiser for the *projected*
coordinates (recovery ~98% from uniform/Dirichlet starts), but the covariance-only objective is
**multimodal** — higher-loss local minima at wrong non-decoy populations, and decoy-collapse basins
for the *marginal* coordinates. Jacobian at truth is rank-2 of 4–5 population dims. Under the strict
"every neutral start" gate all four coordinates FAIL, but this is a property of covariance-only
matching in isolation. **Decision: advance all four to Stage B; the oracle is a diagnostic.** (This
multimodality recurs as optimisation instability in Stage H.)

## Stage B — frame reweighting with the true target (the core result)

`fitting/jaxENT/moprp_covariance_recovery.py`, audit `moprp_recovery_audit.py`. 500 frame weights,
four loss regimes (baseline / symmetric / dynamic / fixed-reference), five blocked 3-timepoint CV
folds on peptides 2–14, selection gated on the **mean-only baseline** (not uniform — a bug fixed in
`_select`), recovery never used for selection.

**Result — symmetric *projected* covariance matching is promotable in 4/4 ensemble × coefficient
cells** (both defensible coefficients, after dropping published):

| Cell | logpf_projected | uptake_projected |
|---|---|---|
| MSAss / constrained | 80→90% (+9.6) | 80→90% (+9.2) |
| MSAss / scaled | 58→76% (+17.7) | 58→74% (+16.0) |
| Filtered / constrained | 70→88% (+17.9) | 70→85% (+15.2) |
| Filtered / scaled | 88→93% (+4.7) | 88→92% (+4.1) |

Marginal coordinates weaker (uptake 3/4, logpf 1/4). **dynamic / fixed-reference fail the mean gate
everywhere** (they buy recovery by degrading the mean fit). Held-out peptide-1 generalises
reasonably; within-state ESS is low (sparse solutions). Coefficient setting is decisive throughout —
reported separately, never pooled.

## Stage C — is the covariance target population-free? (structure yes, magnitude no)

Modules `src/analysis/elastic_network.py` (ANM/GNM), `src/analysis/covariance_comparison.py` (Mantel +
metrics). Runners `moprp_elastic_network_prior.py`, `moprp_covariance_linear_model.py`.

- **ANM sweep + GNM.** GNM did **not** help (weaker than ANM on structure and variance profile). ANM
  improves with a *large* cutoff at the peptide scale: **ANM rc=24 Å** → peptide Mantel 0.60/0.65 vs
  the target; the variance profile (diagonal) stays weak (0.33/0.34). Structure ≫ magnitude.
- **Linear model, leave-one-ensemble-out transfer.** *Marginal variance* transfers poorly (best
  ~0.13 R²; ANM/GNM MSF ≈ 0). *Off-diagonal structure* transfers partially (MSAss→Filtered predicted
  correlation Mantel 0.76, beating raw ANM 0.65; reverse only 0.35).

**Conclusion:** the covariance **correlation structure** is largely population-free (unweighted
trajectory peptide Mantel 0.57 MSAss / 0.97 Filtered; ANM ~0.6), but the **variance magnitude/scale**
is population-dependent and does not transfer.

## Stage D — population-free structure + a (small) scale recovers the population

`covariance_comparison.rebuild_covariance`; runner `moprp_noncircular_recovery.py`. Build
`C_approx = D^½ R D^½` from population-free structure `R` (unweighted or ANM) and a scale `D`, and
reweight against it (Stage B symmetric loss, `target=C_approx`).

**The non-circular target works.** A population-free structure at an appropriate scale recovers the
population **as well as or better than the true `C(w_NMR)`**: MSAss 57→**90.8%** (unweighted, scale
0.3, passes the gate) up to 95%; Filtered 90→**96%**. Recovery is high for scale 0.03–0.3× the
structure's own magnitude (native-consistent, small) and degrades for scale ≥1. The mean-derived
scale is a *partial* automatic success (75–95% for 3/4 structure×ensemble combos).

## Stage E — can the mean supply the scale automatically? (no)

`covariance_comparison.trace_match_scale`; runner `moprp_scale_calibration.py`. Four rules deriving
the scale from the mean-only fit's covariance (diagonal, trace-match, MaxEnt-regularised, iterative).

**Honest negative.** The mean-only fit is underdetermined — fitting the mean does not force native
concentration — so its covariance is *not small* (MSAss `trace(C_baseline) ≈ trace(C_unw)`, ratio
1.09; ≈15× the ANM trace). Trace-match → 62–63% (barely above baseline); iterative *diverges*; only
mean-fit-diagonal reaches 75% within gate. **Fundamentally**, average protection (the mean) and
population *concentration* (the covariance scale) are independent — the same mean is consistent with
concentrated or spread populations (the frame-permutation wall). The one population-free handle that
works is the mean **gate** as a soft selector (Stage D), not the mean-fit covariance.

## Reframe: the covariance is a regularising *prior*, not a target to reproduce

The intent is a population-free covariance-**shape** prior we *project the data through*; the scale
is just **regularisation strength** (a hyperparameter). This dissolves the Stage E scale worry — we
never needed to reproduce the scale.

## Stage F — which shape-prior mechanism works? (M2, and it's insensitive to source)

`moprp_shape_prior.py`. Three scale-free mechanisms: M1 mode-subspace projection, **M2 soft shape
regulariser** (projected log-Euclidean of `corr(C(w))` to the prior correlation), M3 GLS residual
whitening — sourced from unweighted vs target shape. **M2 is the effective one** (MSAss 57→89%,
Filtered ~88%, within gate; M1/M3 do nothing). All three are **insensitive to prior source** (≤1.1pp)
because unweighted ≈ target shape — which is the non-circular win: the unweighted shape regularises as
well as the true-target shape.

## Stage G — the discriminative "difference modes" (negative)

`moprp_difference_modes.py`. The population shift `corr(C_target) − corr(C_unweighted)`, and whether
it is predictable from population-free features via transfer. **Negative:** the difference is
**diffuse** (top mode 16–17%, no clean discriminative mode) and **not transferable** (transfer R²
−3.5/−0.9; only nonzero coefficient is `unw_xcorr` = regression to the mean; ANM/sequence ≈ 0). On
n=2 ensembles the difference-mode projection is not reconstructable non-circularly — n=2 cannot
separate "not transferable" from "too few points". **Decision:** use the common shape (M2); defer
difference-modes to ISO.

## Stage H — wire up the common-shape regulariser on MoPrP

`src/analysis/state_population.py::correlation_of` / `correlation_shape_loss`; runner
`moprp_shape_prior_reweighting.py`. Reweight against the **unweighted** (population-free) shape:
`L = mean_MSE/mean_uniform + γ·correlation_shape_loss(C(w), prior_corr) + η·KL`; 5 CV folds; γ selected
population-free as the **largest γ whose validation mean MSE ≤ 1.05× the mean-only baseline**; recovery
reported, never selected on. The true-target shape is a reference ceiling.

**Clean win for constrained_optimum; unstable for scaled_published.**
- constrained_optimum: unweighted shape prior lifts recovery **MSAss 52→88% (+36pp)**, **Filtered
  61→88% (+28pp)**, decoy→0, within gate, and unweighted ≈ true_shape — the non-circular win.
- scaled_published: **unstable** — recovery vs γ is non-monotonic for MSAss/unweighted (γ0.1=63%,
  γ1=38%, γ10=89%, γ30=82%), the Stage-A multimodality biting with only 2 starts; the
  largest-γ-within-gate rule landed on γ=1 (a bad basin) because the good γ=10 (89%) is just outside
  the gate. Filtered/scaled is stable but its baseline (90%) has no headroom (prior ~neutral).

**Two robustness fixes needed:** (1) more optimisation starts (2 is too few for the multimodal
landscape; Stage F got 88.8% at γ=1 with 3 starts + full data); (2) a less brittle γ-selector (the
non-monotonic recovery/γ curve breaks "largest-γ-within-gate").

## Overall conclusion (A→H)

1. Ensemble covariance **does** recover the known population (Stage B): symmetric *projected* matching,
   +4–18pp over mean-only, promotable in all four defensible ensemble×coefficient cells.
2. The covariance **structure/shape is population-free** (Stages C/D/F) — the unweighted trajectory (or
   ANM) shape regularises as well as the true-target shape (Stage D matches the ceiling; Stage F/H are
   insensitive to prior source).
3. The covariance **scale is population-dependent and cannot be read off the mean** (Stage E) — but
   scale is just regularisation strength, so as a *prior* this is a hyperparameter, not a blocker.
4. The **discriminative difference-mode** route fails on n=2 real ensembles (Stage G).
5. The wired-up shape regulariser gives a clean non-circular win (+28–36pp) for constrained_optimum,
   with two robustness fixes needed before it is production-solid (Stage H).

Everything is **coefficient-dependent** and reported per cell, never pooled. n=2 ensembles is the
binding limitation on every transfer/generalisation claim.

## Implementation and reproduction

Reusable, tested numerics (`jaxent/src/analysis/`):
- `state_population.py` — targets, frame states, `w_NMR`, strict full-support JSD/recovery, covariance
  coordinates, `correlation_of`, `correlation_shape_loss`, `shrunk_trace_normalized_precision`.
- `elastic_network.py` — `anm_covariance`, `gnm_covariance`.
- `covariance_comparison.py` — `to_correlation`, `mantel_test`, `covariance_metrics`,
  `rebuild_covariance`, `trace_match_scale`.

Recovery-helper fix: `examples/2_CrossValidation/analysis/compute_recovery%_PUF.py`
(`legacy_target_support` flag — retains decoy mass over the full support).

Runners (`examples/2_CrossValidation/fitting/jaxENT/`): shared loader `_moprp_recovery_common.py`;
`moprp_coefficient_lock.py`, `moprp_population_oracle.py`, `moprp_covariance_recovery.py`,
`moprp_recovery_audit.py`, `moprp_elastic_network_prior.py`, `moprp_covariance_linear_model.py`,
`moprp_noncircular_recovery.py`, `moprp_scale_calibration.py`, `moprp_shape_prior.py`,
`moprp_difference_modes.py`, `moprp_shape_prior_reweighting.py`.

Tests (`jaxent/tests/unit/analysis/`; current suite 163 pass): `test_state_population.py`,
`test_moprp_recovery_inputs.py`, `test_moprp_reweighting.py`, `test_elastic_network.py`,
`test_covariance_comparison.py`, `test_moprp_covariance_linear_model.py`,
`test_moprp_shape_prior_reweighting.py`, `test_joint_covariance_geometry.py`,
`test_joint_geometry_reweighting.py`, `test_iso_joint_geometry_prior.py`, and
`test_moprp_joint_geometry_validation.py`.

Outputs land in `examples/2_CrossValidation/fitting/jaxENT/_moprp_*` directories.

## Stage I — joint reweighting + BV-coefficient fitting (2026-07-21, done)

Motivation: the clean recovery rode on `constrained_optimum` (Bh=0, a degenerate pre-fit), and the
γ regulariser strength was an unwanted swept knob. Runner
`fitting/jaxENT/moprp_joint_reweight_fit.py` (+ `test_moprp_joint_reweight_fit.py`): jointly optimise
per-ensemble frame weights + **shared non-negative (Bc, Bh)** (softplus), γ **fixed = 1**, 5 starts,
shape prior frozen at the published direction. 145 analysis tests pass.

| Setting | fitted (Bc, Bh) | MSAss rec / decoy | Filtered rec / decoy |
|---|---|---|---|
| γ=1 (shape prior) | (0.234, **0.057**) | 69.9% / 0.078 | 73.8% / 0.014 |
| γ=0 (mean-only ablation) | (0.239, **0.150**) | 62.2% / 0.226 | 51.1% / 0.075 |

**Two findings:**
1. **The covariance shape prior helps, even with coefficients fit jointly** — γ=1 vs γ=0 lifts
   recovery +7.7pp (MSAss) / +22.7pp (Filtered) and cuts decoy mass (0.078 vs 0.226; 0.014 vs 0.075).
   So the regulariser is real, and the parameter-free version (γ=1 fixed, one joint fit, 5 starts —
   no sweep, no selector) works and is stable.
2. **The H-bond channel is genuinely unsupported — the covariance does NOT break the Bh=0
   degeneracy.** Both fits collapse `Bh` to small values (0.057 with the shape prior, 0.150 mean-only;
   the shape prior actually pushed it *lower*), nowhere near the published 2.0. So the `Bh≈0`
   result is **not a two-stage artifact** — under the corrected hard-count contact features neither
   the mean nor the covariance supports a standard BV H-bond term; protection is essentially
   heavy-contact-driven. (Consistent with the prior audit that hard-count BV over-protects.)

**Consequence:** the `constrained_optimum` (Bh=0) recovery win should be read as "protection ≈
heavy-contact-only under these features", not "a degenerate artifact to be fixed by joint fitting".
Joint recovery (70–74%, near-heavy-only, shape prior on) is a more honest number than the two-stage
88% (which also rode on Bh=0 plus extra mean-gate headroom). The γ=1 fixed joint fit is the clean,
parsimonious procedure going forward.

## Former Stage J — correction and disposition (2026-07-22)

The ISO-trained joint-geometry experiment used frozen covariance pseudo-targets and then optimised
frame weights and BV coefficients against them. It therefore did **not** test whether HDX curves can
identify target effective-rate variances. Its results must not be interpreted as target-variance
inference.

The completed ISO qualification was negative: no transferable candidate passed the preregistered
held-out mean gate (`median mean-MSE ratio <= 1.05`). The best recorded ratios were 1.662 for the
unlearned prior, 1.665 for the family prior, and 2.146 for the point prior. The automatically written
`primary_method: unlearned` value was only a fallback ranking after gate failure, not a qualified
selection.

**Disposition:** do not run former Stage J on MoPrP. Keep its outputs unchanged as provenance and
keep its code archived except for a fail-closed launch guard; do not overwrite, extend, or
reinterpret the artifacts. In particular,
`validate_moprp_joint_geometry_prior.py` and artifacts under `_iso_joint_geometry_prior/` are not an
authorised path to MoPrP validation.

## Replacement experiment — geometry-regularised HDX target-variance inference

The new scientific question is whether uptake curves, with the BV mean held fixed, can identify the
diagonal amplitudes of a target effective-rate covariance when population-free structural geometry
is used only as regularisation:

```text
HDX curves + fixed BV mean
            ↓
infer effective-rate variances D with geometry regularisation
            ↓
C_HDX = D½ R_geometry D½
            ↓
later compare C_traj(w, Bc, Bh) with C_HDX
```

This experiment stops before ensemble reweighting. Neither frame weights nor BV coefficients are
optimised.

### Locked construction

- Work in effective-rate coordinates, `k_fi = k_int,i exp(-z_fi)`, because uptake directly
  constrains the rate distribution.
- Infer positive residue variances `D = diag(d_i)` and construct
  `C_HDX = D½ R_geometry D½`; map it to peptides only by the congruence `M C_HDX Mᵀ`.
- Obtain `R_geometry` from the uniform-trajectory effective-rate correlation, without target
  populations. Apply a PSD compact-support distance taper at 8 Å and retain sequential neighbours
  with a PSD nearest-neighbour kernel. Never hard-mask covariance entries in a way that can destroy
  positive semidefiniteness.
- Compare frozen covariance-only, distance-only, sequence-only, covariance×distance/sequence,
  identity, and shuffled-geometry constructions.

### Estimators

Run two no-reweighting estimators:

1. **Curve moment:** fit `D` over the complete uptake-time curve with a positive two-moment rate
   distribution whose mean is fixed by BV.
2. **Structured residual:** fit `D` with
   `Σ_t = M J_t D½ R_geometry D½ J_t Mᵀ + εI` and a Gaussian quasi-likelihood containing
   both the quadratic residual and log determinant. Until ISO establishes correspondence with true
   conformational variance, report this only as model-discrepancy/structured-residual inference.

Choose the estimator, geometry, and regularisation using held-out TeaA HDX reconstruction only,
never population recovery.

### No-fitting numerical litmus (implemented; passed 2026-07-22)

Before any variance optimisation, build all six geometries for both ISO_BI and ISO_TRI and map them
through all three peptide panels. Check geometry/covariance PSD, exact diagonal `D`, 8 Å compact
support with the sequential-neighbour exception, peptide congruence, the zero-variance fixed-mean
limit, and finite curve/structured objectives evaluated without an optimiser.

The real-input litmus passed 36/36 ensemble × panel × geometry cells. Its manifest records
`optimization_performed: false`, `variance_fitting_performed: false`, and no weight/BV fitting.

### HDX-only development (implemented; full run pending)

Development fits only `D` and compares both estimators, all six frozen geometries, and the registered
regularisation grid. Selection uses within-cell held-out HDX reconstruction ranks only. Population
recovery and variance-truth columns are excluded from the selector.

Development uses registered split 0. It writes a non-qualified selection artifact with
`can_launch_moprp_validation: false`; pilot artifacts also cannot launch formal TeaA qualification.
Registered splits 1 and 2 remain untouched for subsequent frozen-settings qualification. A short
pilot has established execution of the complete estimator × geometry matrix, but its ranking is not
scientific evidence.

### TeaA/ISO qualification

- Generate a coherent exact frame-mixture target from the known 40:60 TeaA populations and
  published BV coefficients. Keep the published artificial TeaA curve as a separate model-mismatch
  stress test.
- Do not optimise frame weights or BV coefficients.
- Test ISO_BI and ISO_TRI across the existing equal, random-fixed, and random-variable panels and
  registered peptide/time splits.
- Evaluate held-out uptake reconstruction, residue-rate variance recovery, mapped marginal and
  conditional peptide variance, fold stability, PSD/finite-objective checks, and improvement over
  constant-variance, identity, and shuffled controls.

Qualification requires all of the following: held-out mean-MSE ratio `<= 1.05`; median log-variance
Spearman correlation `>= 0.5`; mapped variance log-RMSE at least 20% lower than constant variance;
and better performance than shuffled geometry in every panel. Freeze all choices only after this
gate passes.

### MoPrP validation boundary

MoPrP remains blocked unless TeaA/ISO qualification passes. If it passes, apply the frozen estimator
to the real 15-timepoint curves with the canonical physical peptide map. Exclude the peptide that
contains unmapped residue 101 and keep peptide 1 fully held out. Run AF2-MSAss and AF2-Filtered
independently under identical frozen rules.

Reveal the NMR populations only after inference, then compute the effective-rate variance
pseudo-truth in the same coordinate and at the same frozen BV coefficients. NMR populations may not
tune the estimator, cutoff, geometry, or regularisation. Peptide 1 is an independent envelope and
shape diagnostic, never a fitted target.

Do not begin weight/BV optimisation unless the HDX-inferred variance beats constant and shuffled
controls and correlates positively with the NMR pseudo-truth in both ensembles.

### Exploratory MoPrP sweep test bed (2026-07-22; non-promotable)

At the user's explicit request, `investigate_moprp_target_variance_sweep.py` provides a separate
diagnostic matrix over both estimators, all six geometries, and the registered regularisation grid.
It excludes peptide 1 and the residue-101 peptide, chooses one shared candidate across AF2-MSAss and
AF2-Filtered from held-out HDX only, and writes the blinded variances/selection manifest before any
NMR read. Its default execution stops there; revealing every candidate requires the explicit
`--reveal-all-diagnostic` flag. Every artifact is fail-closed (`qualified: false`,
`can_launch_moprp_validation: false`, `weight_bv_optimization_authorized: false`). Scalar constant
`D` is independently fit from the same training HDX mask while retaining the candidate `R`; exact
peptide/time folds, residue coverage, numerical knobs, input-array hashes, structure hash, and code
revision are recorded.

The explicit reveal-all pilot exhausted MoPrP as a future confirmatory blind set: all 96 swept
candidates and their NMR pseudo-truth scores are now available. MoPrP must therefore be described as
an **exploratory benchmark**; any future confirmatory claim requires a new external blinded system.
The reviewed interleaved-fold pilot was negative. HDX-only selection chose structured residual +
covariance×distance/sequence + `lambda=1`, but shuffled geometry ranked better (0.111 versus 0.130).
Post-reveal residue variance ordering was positive in both ensembles (Spearman 0.739 MSAss, 0.945
Filtered), yet the selected mapped covariance lost to shuffled in both and to independently fitted
constant `D` in Filtered. The diagnostic gate failed and no weight/BV optimisation is authorised.

## TeaA ↔ MoPrP target-variance results and decision

These tables are the human-readable record of the completed **pilot** sweeps. They are deliberately
separate from formal TeaA qualification: TeaA used only the equal-peptide panel, development split
0, and one held-out time fold; MoPrP used one interleaved peptide/time fold. Each pilot evaluated 96
fits (two ensembles × two estimators × six geometries × four regularisation strengths). Lower
predictive-NLL rank and lower log-RMSE are better; higher Spearman correlation is better.

### Scope and authority

| Property | TeaA/ISO pilot | MoPrP exploratory pilot |
|---|---|---|
| HDX source | Coherent exact 40:60 frame mixture | Real 15-timepoint uptake curves |
| Ensembles | ISO_BI, ISO_TRI | AF2-MSAss, AF2-Filtered |
| Fitted quantity | Effective-rate diagonal `D` only | Effective-rate diagonal `D` only |
| BV coefficients | Published `(Bc, Bh) = (0.35, 2.0)`, fixed | Same published coefficients, fixed |
| Peptide/time scope | Equal panel; split 0; one time fold | Peptide 1 held out; peptide 12/residue 101 excluded; one interleaved peptide/time fold |
| Truth use | Exact mixture truth revealed only for pilot diagnostics | NMR pseudo-truth revealed after blinded files and HDX-only selection were written |
| Constant-`D` comparator | **Oracle-scaled** from the synthetic true variance profile | Independently fitted scalar `D` from the same training HDX mask, retaining candidate `R` |
| Formal authority | Non-promotable pilot; full registered panels/folds not run | Non-promotable exploratory benchmark; NMR blind exhausted |

### Primary HDX-only selections

| System | Selected estimator | Selected physical geometry | `lambda` | Selected rank | Best identity rank | Best shuffled rank | HDX-only decision |
|---|---|---|---:|---:|---:|---:|---|
| TeaA/ISO | Curve moment | Covariance×distance/sequence | 1.0 | **0.184** | 0.053 | 0.053 | Identity and shuffled both beat the selected physical geometry |
| MoPrP | Structured residual | Covariance×distance/sequence | 1.0 | **0.130** | 0.426 | 0.111 | Shuffled beats selected; covariance-only ties selected at 0.130 |

The estimator flip is substantive: exact-mixture TeaA selects the positive curve-moment model,
whereas real MoPrP selects structured residual inference whose predicted mean curve is fixed and
whose `D` should therefore be read as model-discrepancy structure, not yet conformational variance.

### Complete TeaA pilot ranking by estimator and geometry

For each estimator/geometry pair, this table reports its best eligible regularisation. Ranks are
computed only after removing candidate settings that failed convergence/finite/PSD safeguards.

| Estimator | Geometry | Best `lambda` | Predictive-NLL rank | Held-out mean-MSE ratio |
|---|---|---:|---:|---:|
| Curve moment | Identity | 0.01 | **0.053** | 0.079 |
| Curve moment | Shuffled | 0.10 | **0.053** | 0.079 |
| Curve moment | Covariance×distance/sequence | 1.00 | 0.184 | 0.066 |
| Curve moment | Sequence-only | 1.00 | 0.237 | 0.067 |
| Curve moment | Distance-only | 1.00 | 0.316 | 0.065 |
| Curve moment | Covariance-only | 1.00 | 0.342 | 0.066 |
| Structured residual | Sequence-only | 0.10 | 0.500 | 1.000 |
| Structured residual | Identity | 0.10 | 0.539 | 1.000 |
| Structured residual | Covariance-only | 0.10 | 0.605 | 1.000 |
| Structured residual | Distance-only | 0.01 | 0.618 | 1.000 |
| Structured residual | Shuffled | 1.00 | 0.645 | 1.000 |
| Structured residual | Covariance×distance/sequence | 0.10 | 0.671 | 1.000 |

### TeaA truth diagnostics for the selected physical construction

The matched identity and shuffled values below use the selected estimator and `lambda=1`, so this is
an apples-to-apples geometry comparison. The constant comparator has the synthetic truth-derived
scalar scale noted above.

| ISO source | Mean-MSE ratio | Residue log-variance Spearman | Selected mapped log-RMSE | Constant-`D` log-RMSE | Identity log-RMSE | Shuffled log-RMSE | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| ISO_BI | 0.0660 | 0.9937 | 1.1033 | 9.4911 | 1.0699 | **1.0571** | Beats constant; loses identity and shuffled |
| ISO_TRI | 0.0652 | 0.9937 | 1.1038 | 9.4839 | 1.0699 | **1.0657** | Beats constant; loses identity and shuffled |

The selected construction reduces mapped log-RMSE by approximately 88.4% versus the oracle-scaled
constant profile and almost perfectly recovers residue variance ordering. Nevertheless, identity
and shuffled geometries are slightly better. The high variance recovery therefore comes primarily
from the complete uptake-curve information, not from the proposed physical geometry.

### Complete MoPrP pilot ranking by estimator and geometry

| Estimator | Geometry | Best `lambda` | Predictive-NLL rank | Held-out mean-MSE ratio |
|---|---|---:|---:|---:|
| Structured residual | Shuffled | 1.00 | **0.111** | 1.000 |
| Structured residual | Covariance×distance/sequence | 1.00 | 0.130 | 1.000 |
| Structured residual | Covariance-only | 0.10 | 0.130 | 1.000 |
| Structured residual | Distance-only | 1.00 | 0.167 | 1.000 |
| Structured residual | Sequence-only | 1.00 | 0.167 | 1.000 |
| Structured residual | Identity | 1.00 | 0.426 | 1.000 |
| Curve moment | Identity | 0.01 | 0.630 | 0.945 |
| Curve moment | Sequence-only | 0.00 | 0.685 | 0.944 |
| Curve moment | Covariance-only | 0.10 | 0.722 | 1.010 |
| Curve moment | Distance-only | 0.01 | 0.722 | 0.942 |
| Curve moment | Covariance×distance/sequence | 0.00 | 0.759 | 0.944 |
| Curve moment | Shuffled | 0.00 | 0.796 | 0.944 |

The structured-residual mean-MSE ratio is exactly 1 because this estimator leaves the fixed-BV mean
curve unchanged; it is selected by the propagated-covariance predictive score, not by improving the
mean curve.

### MoPrP post-reveal NMR pseudo-truth diagnostics

| Ensemble | Residue log-variance Spearman | Selected mapped log-RMSE | Fitted constant-`D` log-RMSE | Matched shuffled log-RMSE | Beats constant? | Beats shuffled? | Peptide-1 envelope coverage |
|---|---:|---:|---:|---:|---|---|---:|
| AF2-MSAss | 0.7393 | 8.2029 | 10.2090 | **7.9198** | Yes | **No** | 1.00 |
| AF2-Filtered | 0.9453 | 6.5447 | 6.1058 | **6.0357** | **No** | **No** | 1.00 |

Residue variance **ordering** agrees positively with NMR in both ensembles, but the mapped
covariance **magnitude and peptide structure** do not pass the controls. This is the central
calibration failure: a high Spearman correlation is not sufficient evidence that the inferred
covariance is quantitatively correct.

### Numerical safeguards and convergence

| Pilot | Full-`D` fits converged | Scalar controls converged | Finite objectives | PSD covariances |
|---|---:|---:|---|---|
| TeaA/ISO | 78 / 96 | Not independently fitted in this pilot | 96 / 96 | 96 / 96 |
| MoPrP | 69 / 96 | 96 / 96 | 96 / 96 | 96 / 96 |

Non-converged candidates were excluded from HDX selection. The short pilot iteration limit means
the ranking is suitable for test-bed triage, not final method selection.

### Gate and decision

| Requirement | TeaA pilot | MoPrP pilot | Decision |
|---|---|---|---|
| Fixed-mean MSE within 1.05 | Pass for selected construction | Pass by construction for structured residual | Necessary but non-discriminating |
| Positive/strong variance ordering | Pass: ~0.994 in both ISO sources | Pass: 0.739 and 0.945 | Rank recovery alone is insufficient |
| Beat constant variance | Pass in TeaA, with an oracle-scaled comparator | Mixed: pass MSAss, fail Filtered | MoPrP gate fails |
| Beat shuffled geometry | **Fail in TeaA** | **Fail in both MoPrP ensembles** | Physical geometry unsupported |
| All registered folds/panels | Not run | Pilot fold only | No formal qualification |
| Begin weight/BV optimisation | No | No | **Not authorised** |

### High-entropy findings

1. **The proposed physical geometry never wins its decisive negative control.** Shuffled geometry
   beats covariance×distance/sequence on held-out HDX in TeaA and MoPrP, and on mapped NMR
   pseudo-truth in both MoPrP ensembles.
2. **TeaA identifies `D`, not `R`.** Near-perfect residue variance Spearman and an 88% mapped-error
   reduction coexist with identity/shuffled outperforming the physical geometry.
3. **The preferred estimator flips on real data.** Curve moment dominates coherent exact-mixture
   TeaA, but structured residual dominates MoPrP. That is direct evidence of model mismatch and
   supports retaining the “model discrepancy” label.
4. **Variance rank and covariance calibration separate sharply.** MoPrP Spearman is strongly
   positive while mapped log-RMSE loses to controls, especially in AF2-Filtered.
5. **MoPrP is no longer confirmatory.** Revealing all swept candidates against NMR exhausted the
   blind; it can remain an exploratory benchmark only. A future confirmatory claim requires a new
   external blinded system.

**Decision:** do not freeze covariance×distance/sequence, do not launch formal MoPrP validation,
and do not begin frame-weight or BV-coefficient optimisation. The next scientifically valid step is
full TeaA registered-fold development with a geometry that must beat shuffled/identity controls;
if it cannot, variance-amplitude inference should be retained without claiming geometry recovery.

Source artifacts used to construct the tables:

| Result | Artifact |
|---|---|
| TeaA full 96-fit pilot matrix | `/tmp/jaxent_teaa_target_variance_pilot_reg_sweep_20260722/pilot_raw.csv` |
| TeaA HDX-only selection/ranking | `/tmp/jaxent_teaa_target_variance_pilot_reg_sweep_20260722/pilot_selection.json` |
| MoPrP blinded HDX matrix | `/tmp/jaxent_moprp_target_variance_diagnostic_pilot_final_20260722/blinded_hdx_sweep.csv` |
| MoPrP blinded selection and provenance | `/tmp/jaxent_moprp_target_variance_diagnostic_pilot_final_20260722/blinded_selection_manifest.json` |
| MoPrP post-reveal metrics | `/tmp/jaxent_moprp_target_variance_diagnostic_pilot_final_20260722/nmr_pseudotruth_diagnostic_metrics.csv` |
| MoPrP final decision | `/tmp/jaxent_moprp_target_variance_diagnostic_pilot_final_20260722/diagnostic_decision.json` |

### Required safeguards

- Verify every `D½ R D½` is PSD, has diagonal `D`, and maps correctly through overlapping peptides.
- Verify zero variance recovers the fixed-BV mean and recover known variance from small synthetic
  positive-rate mixtures.
- Verify the distance and sequence kernels are PSD and the 8 Å taper removes unsupported spatial
  coupling.
- Keep target weights and NMR populations out of all estimator inputs.
- Add a hard launch guard so failed former-Stage-J artifacts cannot start MoPrP.
- Preserve all former Stage J artifacts as provenance.

Implementation is isolated from the production HDX/reweighting loss:

- `jaxent/src/analysis/hdx_target_variance.py` contains the PSD geometry constructions, curve-moment
  and structured-residual estimators, qualification gate, and frozen-artifact boundary.
- `investigate_iso_target_variance.py` separates the no-fitting litmus, HDX-only development, and
  frozen-settings qualification; it keeps the published artificial curve as a separate
  model-mismatch stress test and freezes a promotable artifact only after a qualification pass.
- `validate_moprp_target_variance.py` performs NMR-blinded inference, writes the blinded result, and
  only then reveals the pseudo-ground truth for separate AF2-MSAss/AF2-Filtered evaluation.
- `investigate_moprp_target_variance_sweep.py` is the non-promotable exploratory sweep test bed; its
  default is inference-only and its explicit reveal-all output permanently exhausts MoPrP blinding.
- The archived `validate_moprp_joint_geometry_prior.py` now fails closed before any former-Stage-J
  MoPrP execution.

The full TeaA/ISO qualification has not been run, so no MoPrP target-variance validation is yet
authorised. Stage H robustness work remains separate.

## Coefficient-consistency re-run and the D-only verdict (2026-07-23)

Review of the pilots found a coefficient inconsistency: the MoPrP target-variance runners fixed the
BV mean at the **dropped** published coefficients `(Bc, Bh) = (0.35, 2.0)` (`_moprp_recovery_common.py`
`PUBLISHED_BC/BH`, read by `validate_moprp_target_variance.py` and hardcoded in
`investigate_moprp_target_variance_sweep.py`). The coefficient-lock stage explicitly rejected
published for MoPrP ("standard-condition calibration, not pD 4"), and Stage I showed the H-bond
channel is unsupported under the corrected hard-count features. Because the structured-residual
estimator fits `D` on the residual about the fixed mean, any mean misfit is absorbed into `D` as
spurious variance — contaminating the amplitude and its NMR-magnitude comparison.

The sweep was re-run with the mean fixed at the two frozen settings, reported separately (never
pooled). Artifacts: `_moprp_target_variance_scaled_published_20260723/` and
`_moprp_target_variance_constrained_optimum_20260723/`.

| Metric | Pilot (published 0.35/2.0) | scaled_published (0.186/1.064) | constrained_optimum (0.229/0) |
|---|---|---|---|
| MSAss beats constant | yes | yes | yes |
| **Filtered beats constant** | **NO** | **yes** | **yes** |
| MSAss residue log-var Spearman | 0.739 | 0.845 | 0.860 |
| Filtered residue log-var Spearman | 0.945 | 0.885 | 0.880 |
| MSAss mapped log-RMSE / shuffled | 8.20 / 7.92 | 6.91 / 5.88 | 7.17 / 6.78 |
| Filtered mapped log-RMSE / shuffled | 6.55 / 6.04 | 6.17 / 5.37 | 7.46 / 6.55 |
| beats_shuffled (NMR truth), both ensembles | no | no | no |
| `diagnostic_variance_gate_passes` | false | false | false |

**The coefficient fix was material.** Correcting the mean moved every magnitude metric in the
predicted direction, most decisively flipping **AF2-Filtered from failing to beating the constant
control** in both settings. The pilot's "Filtered fails constant" verdict was partly an artifact of
the rejected published mean and must not be carried forward.

**Two core verdicts survive the correction:**

1. **The estimator does not flip.** Structured residual still wins the HDX-only selection under both
   corrected means; curve moment now fits the mean well (held-out mean-MSE ratio ~0.03) but still
   loses the covariance-predictive selection. So the pilot's curve→structured flip was **not** purely
   mean misfit — structured residual robustly dominates the predictive score on real MoPrP. The
   "model discrepancy" label on MoPrP's `D` stays on.
2. **Geometry still fails on truth.** `beats_shuffled_in_every_fold = False` and
   `diagnostic_variance_gate_passes = False` in both ensembles and both settings. No authorisation to
   proceed; the launch remains fail-closed.

**New wrinkle (does not rescue geometry).** On held-out HDX *reconstruction* (the blinded selection
metric, distinct from NMR-truth), the ranking reversed vs the pilot: physical locality now beats
shuffled (scaled_published: `distance_only` 0.042 < shuffled 0.128). But (a) the winner is a **bare
distance/sequence locality kernel with zero trajectory content** — the trajectory-derived
`covariance_only` / `covariance_distance_sequence` rank *below* it, so this is generic local
smoothness regularising the propagated peptide covariance, **not** recovery of the population-free
trajectory geometry `R`; and (b) it does not carry to truth — the same construction still loses to
shuffled on NMR-mapped magnitude. Predicting held-out HDX better ≠ producing a covariance that
matches the independent structure.

### Verdict: the D-only retreat

The corrected MoPrP result reproduces the TeaA lesson on a properly calibrated mean:

> **HDX curves identify the variance amplitude `D` (ordering strong, beats constant in all four
> cells, magnitude improved) but do not recover the covariance geometry `R`.** Locality helps HDX
> self-prediction only as smoothing; trajectory-derived geometry adds nothing beyond that and fails
> the NMR-truth control.

State of the four falsifiable questions after the re-run:

- **Q1 (recover `D`):** Yes — cleaner now, beats constant in all four cells.
- **Q2 (geometry beats controls):** No on truth; the only "win" is generic locality on HDX
  self-prediction, and trajectory-derived `R` never wins.
- **Q3 (agree with NMR):** Ordering yes, magnitude/structure no — gate fails both ensembles.
- **Q4 (reweighting target):** Not authorised; the supportable target is `diag(D)`, not
  `C = D½ R D½`.

**Scientifically clean path forward — reduce scope to `D` only.** Reweight to match residue
effective-rate variance amplitudes (`diag(D)`); drop the `R` geometry claim unless a future
construction beats shuffled **on truth**, not merely on held-out HDX. Two method notes carry into
that work:

- The qualification gate's fixed-mean MSE criterion is **vacuous for structured residual** (ratio
  1.0 by construction). "Beat shuffled on truth" must be a **hard** gate criterion, not a diagnostic.
- Fold-count is still n=1 in these sweeps; any freeze needs the registered multi-fold path.

**Next major step (to design together): step back to the physics of the reduced (`D`-only) scope
before iterating.** Discussion pending.

## Corrected-mean MoPrP apples-to-apples rerun (2026-07-23)

The original exploratory pilot above used the unscaled published BV coefficients `(Bc, Bh) =
(0.35, 2.0)`. That was not the pD 4 mean setting endorsed by the earlier MoPrP coefficient
investigation. The same 96-cell pilot was therefore rerun independently at the two frozen settings
from `_moprp_recovery_coefficient_lock/coefficient_lock.json`:

- primary `scaled_published`: exact `(0.1862401670, 1.0642295258)`, displayed as
  `(0.186, 1.064)`;
- boundary diagnostic `constrained_optimum`: exact `(0.2288930418, 0)`, displayed as `(0.229, 0)`.

The two settings were never pooled. Each run retained the original two ensembles, two estimators,
six geometries, four regularisation values, interleaved peptide/time fold 0, 300-iteration pilot
limit, peptide-1 holdout, and residue-101 peptide exclusion. The original artifacts were not
overwritten. The coefficients were frozen before these fits, but their upstream coefficient-lock
calibration used fixed NMR populations; these reruns are consequently exploratory robustness checks,
not a restored blind validation.

### Execution record

```text
uv run python jaxent/examples/2_CrossValidation/fitting/jaxENT/investigate_moprp_target_variance_sweep.py \
  --bv-setting scaled_published --pilot --fold-scheme interleaved \
  --regularizations 0,0.01,0.1,1 --pilot-maxiter 300 --reveal-all-diagnostic \
  --output-dir jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_target_variance_scaled_published_20260723

uv run python jaxent/examples/2_CrossValidation/fitting/jaxENT/investigate_moprp_target_variance_sweep.py \
  --bv-setting constrained_optimum --pilot --fold-scheme interleaved \
  --regularizations 0,0.01,0.1,1 --pilot-maxiter 300 --reveal-all-diagnostic \
  --output-dir jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_target_variance_constrained_optimum_20260723
```

### Cross-setting HDX-only selection

Lower predictive-NLL rank is better. “Best curve moment” is the best eligible physical-geometry
curve-moment candidate by that same predictive-NLL ranking, not the candidate with the lowest mean
MSE in isolation.

| Fixed mean | Selected estimator | Selected physical geometry | `lambda` | Selected rank | Identity rank | Shuffled rank | Best curve-moment physical candidate | Curve rank | Curve mean-MSE ratio | Diagnostic gate |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---|
| Published `(0.35, 2.0)` | Structured residual | Covariance×distance/sequence | 1.0 | 0.1296 | 0.4259 | **0.1111** | Sequence-only, `lambda=0` | 0.6852 | 0.9443 | Fail |
| Scaled published `(0.186, 1.064)` | Structured residual | Distance-only | 0.1 | **0.0806** | 0.4839 | **0.0806** | Distance-only, `lambda=1` | 0.4032 | 0.8138 | Fail |
| Constrained optimum `(0.229, 0)` | Structured residual | Sequence-only | 1.0 | **0.0641** | 0.3590 | 0.0769 | Distance-only, `lambda=1` | 0.3718 | 0.8835 | Fail |

The estimator flip against TeaA persists under both corrected means: structured residual remains the
MoPrP HDX-only selection. Curve moment reconstructs held-out means substantially better after the
mean correction, but its predictive covariance NLL still ranks well behind structured residual.

The structured-residual mean-MSE ratio is identically `1.000` for every setting by construction:
that estimator leaves the fixed-BV mean prediction unchanged, and the denominator is the same fixed
mean. It cannot diagnose mean correction through the ratio. The absolute fixed-mean held-out MSE
does respond:

| Fixed mean | AF2-MSAss fixed-mean MSE | AF2-Filtered fixed-mean MSE |
|---|---:|---:|
| Published `(0.35, 2.0)` | 0.052735 | 0.092828 |
| Scaled published `(0.186, 1.064)` | 0.058210 | 0.044618 |
| Constrained optimum `(0.229, 0)` | **0.050370** | **0.035544** |

The corrected means strongly improve AF2-Filtered but not uniformly AF2-MSAss. This is why the
coefficient change helps curve-moment reconstruction without changing the selected estimator.

### Complete `scaled_published` ranking by estimator and geometry

For each estimator/geometry pair, the best eligible regularisation is shown.

| Estimator | Geometry | Best `lambda` | Predictive-NLL rank | Held-out mean-MSE ratio |
|---|---|---:|---:|---:|
| Structured residual | Distance-only | 0.10 | **0.0806** | 1.0000 |
| Structured residual | Shuffled | 1.00 | **0.0806** | 1.0000 |
| Structured residual | Covariance-only | 0.10 | 0.0968 | 1.0000 |
| Structured residual | Sequence-only | 1.00 | 0.1452 | 1.0000 |
| Structured residual | Covariance×distance/sequence | 1.00 | 0.1935 | 1.0000 |
| Curve moment | Distance-only | 1.00 | 0.4032 | 0.8138 |
| Curve moment | Covariance×distance/sequence | 1.00 | 0.4677 | 0.7968 |
| Curve moment | Sequence-only | 1.00 | 0.4839 | 0.7529 |
| Structured residual | Identity | 1.00 | 0.4839 | 1.0000 |
| Curve moment | Shuffled | 1.00 | 0.5000 | 0.8435 |
| Curve moment | Identity | 1.00 | 0.6290 | 0.7482 |
| Curve moment | Covariance-only | 0.01 | 0.6452 | 0.7786 |

Distance-only does not beat the shuffled control on held-out HDX; it ties it. The combined
covariance×distance/sequence construction falls from the original selected physical geometry to
fifth place within the structured-residual block.

### Complete `constrained_optimum` ranking by estimator and geometry

| Estimator | Geometry | Best `lambda` | Predictive-NLL rank | Held-out mean-MSE ratio |
|---|---|---:|---:|---:|
| Structured residual | Sequence-only | 1.00 | **0.0641** | 1.0000 |
| Structured residual | Distance-only | 1.00 | 0.0769 | 1.0000 |
| Structured residual | Shuffled | 1.00 | 0.0769 | 1.0000 |
| Structured residual | Covariance×distance/sequence | 1.00 | 0.1026 | 1.0000 |
| Structured residual | Covariance-only | 0.10 | 0.1154 | 1.0000 |
| Structured residual | Identity | 1.00 | 0.3590 | 1.0000 |
| Curve moment | Distance-only | 1.00 | 0.3718 | 0.8835 |
| Curve moment | Sequence-only | 1.00 | 0.4231 | 0.8239 |
| Curve moment | Covariance×distance/sequence | 1.00 | 0.4231 | 0.8658 |
| Curve moment | Shuffled | 1.00 | 0.4231 | 0.9205 |
| Curve moment | Identity | 1.00 | 0.5256 | 0.8088 |
| Curve moment | Covariance-only | 0.01 | 0.5513 | 0.8388 |

Sequence-only narrowly beats shuffled on held-out HDX at the Bh=0 boundary. This is not sufficient
geometry validation: the post-reveal covariance comparison below reverses that ordering in both
ensembles.

### Post-reveal NMR pseudo-truth comparison

| Fixed mean | Ensemble | Residue log-variance Spearman | Selected mapped log-RMSE | Fitted constant-`D` log-RMSE | Matched shuffled log-RMSE | Beats constant? | Beats shuffled? | Peptide-1 MSE | Envelope coverage |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| Published | AF2-MSAss | 0.7393 | 8.2029 | 10.2090 | **7.9198** | Yes | **No** | 0.016258 | 1.00 |
| Published | AF2-Filtered | 0.9453 | 6.5447 | **6.1058** | **6.0357** | **No** | **No** | 0.015143 | 1.00 |
| Scaled published | AF2-MSAss | 0.8453 | 6.9131 | 8.8323 | **5.8838** | Yes | **No** | 0.024394 | 1.00 |
| Scaled published | AF2-Filtered | 0.8848 | 6.1695 | 7.3928 | **5.3692** | Yes | **No** | 0.006041 | 1.00 |
| Constrained optimum | AF2-MSAss | 0.8598 | 7.1713 | 10.2752 | **6.7826** | Yes | **No** | 0.021111 | 1.00 |
| Constrained optimum | AF2-Filtered | 0.8797 | 7.4624 | 7.9503 | **6.5509** | Yes | **No** | 0.006112 | 1.00 |

The old constant-variance magnitude failure is **not robust** to correcting the mean: both corrected
settings beat their independently fitted constant-`D` control in both ensembles. The geometry
failure **is robust**: the matched shuffled geometry has lower mapped log-RMSE in all four corrected
setting × ensemble cells. Positive residue rank agreement therefore remains insufficient evidence
for the inferred off-diagonal structure.

Peptide 1 stays fully outside fitting and selection. Its envelope coverage remains 1.00 everywhere;
its mean-shape MSE improves sharply for AF2-Filtered under both corrected settings but worsens for
AF2-MSAss under `scaled_published`, reinforcing the ensemble-specific mean mismatch.

### Numerical safeguards and provenance

| Fixed mean | Full-`D` fits converged | Scalar controls converged | Finite objectives | PSD covariances | Candidate rows |
|---|---:|---:|---:|---:|---:|
| Scaled published | 74 / 96 | 96 / 96 | 96 / 96 | 96 / 96 | 96 |
| Constrained optimum | 83 / 96 | 96 / 96 | 96 / 96 | 96 / 96 | 96 |

Non-converged full-`D` candidates were excluded before ranking. Both corrected manifests reproduce
the original fold definitions, peptide/time cells, canonical mapping, observed uptake, structures,
and other non-rate input hashes. They additionally record hashes of each ensemble's
`rates_by_frame` and `mean_rates`, the exact coefficient values, and the coefficient-lock hash.
Both decisions remain fail-closed: `qualified: false`, `can_launch_moprp_validation: false`, and
`weight_bv_optimization_authorized: false`.

| Result | Artifact directory |
|---|---|
| Original published-mean pilot | `/tmp/jaxent_moprp_target_variance_diagnostic_pilot_final_20260722` |
| Corrected primary `scaled_published` pilot | `jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_target_variance_scaled_published_20260723` |
| Boundary `constrained_optimum` pilot | `jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_target_variance_constrained_optimum_20260723` |

### Corrected decision

1. **The estimator flip persists.** Correcting the fixed mean materially improves curve-moment mean
   reconstruction, especially for AF2-Filtered, but structured residual remains the HDX-only
   predictive-NLL winner under both settings. The flip is not explained solely by the old mean.
2. **Magnitude and geometry must be separated.** Corrected inference now beats fitted constant `D`
   in both ensembles, so the original magnitude failure was mean-sensitive. Shuffled geometry still
   beats the selected physical covariance after reveal in every corrected cell, so the
   off-diagonal-geometry claim remains unsupported.
3. **The selected geometry is unstable across means.** It changes from combined geometry under
   `(0.35, 2.0)`, to distance-only under `scaled_published`, to sequence-only at the Bh=0 boundary.
   This is further evidence against freezing a physical `R_geometry` from this pilot.
4. **Retain D-only inference as the viable branch.** The corrected results support heterogeneous
   variance amplitudes more than constant variance, but not the proposed physical correlation
   structure. Any continuation should treat `R` as identity/nuisance or explicitly model-average
   it rather than claim geometry recovery.
5. **Do not begin ensemble fitting.** TeaA did not pass the formal geometry gate, MoPrP is
   exploratory after the all-candidate reveal, and neither corrected MoPrP run beats shuffled
   geometry after reveal. Weight/BV optimisation remains unauthorised.
