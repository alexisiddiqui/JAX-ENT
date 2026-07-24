# Handoff: physics of HDX effective-rate *variance* inference (D-only scope)

Status: **open investigation.** The covariance-*geometry* program (`C_HDX = D½ R D½`) has been reduced
to a **D-only** scope after the geometry `R` repeatedly failed its decisive negative control (see
verdict below). This document is the self-contained brief for iterating on the *physics* of what the
variance amplitude `D` actually is, whether it is identifiable from centroid uptake, and what — if
anything — could ever license a geometry `R` claim.
Created: 2026-07-23
Parent: `plans/known_population_covariance_recovery.md` (full stage history A–I, the target-variance
replacement experiment, and the coefficient-fix re-run live there).

## Who this is for

- A **deep-research agent** working the *physics/statistics* questions (HDX mechanism, EX1/EX2,
  isotope-envelope information content, rate-heterogeneity identifiability, elastic-network priors).
  Everything it needs mathematically is in §§2–5; its questions are in §7.
- A **codebase-exploration agent** grounding the maths against the implementation. Exact file/symbol
  pointers are in §6; the equations in §§2–4 are transcribed from that code so the two can be
  cross-checked.

Both should treat the guardrails in §8 as hard constraints.

---

## 1. Where we are (the D-only verdict)

The falsifiable programme asked four questions. After the coefficient-consistency re-run
(2026-07-23), the answers are:

| Q | Answer | Evidence |
|---|---|---|
| 1. Recover residue effective-rate variance amplitude **D**? | **Yes** | TeaA residue log-var Spearman ≈ 0.99; MoPrP 0.85 / 0.88; beats constant control in all four cells |
| 2. Does structural geometry **R** beat identity/shuffled controls? | **No (on truth)** | shuffled beats/ties `R` on NMR-mapped covariance in both MoPrP ensembles and on TeaA held-out HDX; when a physical geometry *does* win held-out HDX it is a bare distance/sequence locality kernel, **not** the trajectory-derived correlation |
| 3. Mapped covariance agrees with NMR/known truth? | **Ordering yes, magnitude/structure no** | gate fails both ensembles |
| 4. Serve as a reweighting target? | **Only `diag(D)` is supportable** | `C = D½ R D½` unsupported over `diag(D)` |

**Verdict carried into this investigation:** HDX centroid curves identify the *variance amplitude*
`D` (a diagonal, marginal, second-moment quantity) but do **not** recover the covariance *geometry*
`R`. The clean path is to reweight (later) to match `diag(D)` — residue effective-rate variance
amplitudes — and drop the `R` claim unless a future construction beats shuffled **on truth**, not
merely on held-out HDX self-prediction. This document is the step-back to understand the *physics*
of `D` before any iteration.

Two method facts that constrain interpretation:
- The **structured-residual** estimator holds the fixed-BV mean curve unchanged, so its held-out
  mean-MSE ratio is **1.0 by construction** — the qualification gate's mean criterion is *vacuous*
  for the estimator that actually gets selected on real data. "Beat shuffled on truth" must become a
  **hard** gate criterion.
- On synthetic TeaA the **curve-moment** estimator wins; on real MoPrP the **structured-residual**
  estimator wins even after the mean is corrected. That estimator flip is direct evidence of BV
  mean-model discrepancy being absorbed into `D` — hence `D` on real data is currently a
  *model-discrepancy* quantity, not yet certified conformational variance.

---

## 2. The HDX forward model (the fixed mean)

All symbols: residue index `i`, frame index `f`, timepoint `t`, peptide index `p`.

### 2.1 Contacts (BV inputs, "physics-v2 hard count")

Per residue `i` and frame `f`, from the MD trajectory
(`jaxent/src/models/func/contacts.py::calc_BV_contacts_universe`, hard cutoff `switch=False`):

- **Heavy contacts** `h_{i,f}` = number of protein heavy (non-H) atoms within `r_heavy = 6.5 Å` of
  the amide **N** of residue `i`, excluding residues in the window `(−2, +2)` around `i`.
- **Acceptor (oxygen) contacts** `o_{i,f}` = number of oxygen H-bond-acceptor atoms within
  `r_O = 2.4 Å` of the amide **H** of residue `i`, same residue-ignore window.

Counting is a **hard cutoff**: `sum(dists <= radius)`. (A legacy rational switch
`1/(1+(r/r0)^6)` exists but is off.) Defaults: `heavy_radius=6.5`, `o_radius=2.4`,
`residue_ignore=(-2,2)` in `jaxent/src/models/config.py`.

### 2.2 Log protection factor and effective rate

BV linear model (`jaxent/src/models/HDX/forward.py::BV_ForwardPass`):

```
z_{i,f}  ≡  log PF_{i,f}  =  Bc · h_{i,f}  +  Bh · o_{i,f}
PF_{i,f} =  exp(z_{i,f})
```

Under EX2, the observable exchange (effective) rate is the intrinsic rate divided by protection:

```
k_{i,f}  =  k_int,i · exp(−z_{i,f})  =  k_int,i / PF_{i,f}
```

`k_int,i` is a **frame-independent** chemistry term (see §2.4). This is the coordinate the
target-variance module works in (`hdx_target_variance.effective_rates`:
`k = k_int · exp(−log_pf)`).

### 2.3 Residue and peptide uptake (EX2, single exponential per frame)

Per-frame residue deuterium uptake
(`jaxent/src/models/HDX/forward.py::BV_uptake_ForwardPass`,
`jaxent/src/analysis/pf_variance.uptake_from_log_pf`):

```
u_{i,f}(t)  =  1 − exp(−k_{i,f} · t)  =  1 − exp(−k_int,i · t / PF_{i,f})
```

Peptide uptake is the **sum over the peptide's residues** via the sparse map `M` (§3):

```
U_{p}(t)  =  Σ_i  M_{p,i}  u_{i}(t)
```

### 2.4 Intrinsic rate `k_int,i`

From `hdxrate.k_int_from_sequence` (Linderstrøm-Lang / Englander chemistry;
`jaxent/src/models/func/uptake.py::calculate_intrinsic_rates`), canonical exPfact 3-Ala reference
rates at MoPrP conditions (pD 4). Per residue:

**Correction (2026-07-23):** this pointer is stale for the MoPrP pipeline — see §9.4.

```
k_int  =  10^lgkA + 10^lgkB + 10^lgkW
lgkA   =  lgkAref − (EaA/ln10/R)(1/Texp − 1/Tref)  +  adj_L_acid  + adj_R_acid  − pD
lgkB   =  lgkBref − (EaB/ln10/R)(1/Texp − 1/Tref)  +  adj_L_base  + adj_R_base  − pKD + pD
lgkW   =  lgkWref − (EaW/ln10/R)(1/Texp − 1/Tref)  +  adj_L_base  + adj_R_base
```

`adj_{L,R}` are left/right sequence-neighbour acid/base factors. N-terminal residue and prolines →
`k_int = ∞` (no measurable amide, excluded). Temperature/pD are experiment-fixed; `k_int` carries
**no conformational** information.

### 2.5 The fixed "mean" curve

The estimator is handed a fixed residue **mean effective-rate** vector `k̄_i` and predicts the
mean peptide curve by the zero-variance limit `U_p^{mean}(t) = Σ_i M_{p,i} (1 − exp(−k̄_i t))`
(`predict_fixed_mean_uptake`). Production BV semantics elsewhere are **average-first in log-PF**
(`average_first_uptake`: `z̄_i = Σ_f w_f z_{i,f}` then transform), which is *not* the same as the
mean of `k_{i,f}` (Jensen gap). **Which coordinate `k̄_i` is computed in is a live question — see
§7.1.** The three frozen BV coefficient settings are:

| Setting | `(Bc, Bh)` | Note |
|---|---|---|
| published (**dropped**) | (0.35, 2.0) | standard-condition calibration, not pD 4; contaminated the pilots |
| scaled_published (**primary**) | (0.186, 1.064) | published direction × 0.532, matches target mean-uptake scale at pD 4 |
| constrained_optimum | (0.229, 0) | free optimum; `Bh→0` reported as model inadequacy (H-bond channel unsupported under hard-count features, confirmed by joint fit in parent Stage I) |

---

## 3. The peptide sparse map `M`

`M ∈ R^{P×N}` (peptides × residues), the **trim-one exPfact** map: each peptide row is the incidence
(summation) over its exchange-competent residues, dropping the N-terminal residue(s) and prolines
that do not report.

**Correction (2026-07-23):** confirmed row-normalized (`1/N_active` per member residue), not a raw
incidence sum — see §9.3.

Overlapping peptides share residues → `M` has overlapping support, which is the
only lever that lets residue-level `D` be partially resolved from peptide-level curves. Built via
the topology subsystem (`jaxent/src/interfaces/topology/`, `create_sparse_map` in
`jaxent/src/data/splitting/sparse_map.py`). MoPrP: 14 peptides, 97 residues (incl. residue 101);
**peptide 1 held out**; the peptide containing unmapped residue 101 excluded from fits.

Peptide-level covariance is obtained by **congruence**: `Cov_peptide = M C M^T`
(`map_hdx_covariance`).

---

## 4. Target-variance inference (no reweighting, no BV fit)

Everything here fits **only** the residue variances `D = diag(d_i)`. `d_i` is parameterised
multiplicatively about the fixed mean rate:

```
d_i  =  k̄_i² · exp(β_i),      β_i ∈ [−18, 8]
```

so `exp(β_i)` is the squared coefficient of variation of the residue effective rate. (`d_i = k̄_i²
exp(β_i)` in `fit_curve_moment_variance` / `fit_structured_residual_variance`.)

### 4.1 Estimator A — curve moment (positive Gamma two-moment closure)

Model each residue's effective rate as Gamma-distributed with the fixed mean `k̄_i` and fitted
variance `d_i`; shape `a_i = k̄_i²/d_i`, scale `θ_i = d_i/k̄_i`. The Gamma Laplace transform gives
the expected survival, hence expected residue uptake
(`positive_two_moment_uptake` / `_gamma_two_moment_uptake_jax`):

```
E_f[exp(−k_i t)]  =  (1 + θ_i t)^{−a_i}  =  (1 + (d_i/k̄_i) t)^{−k̄_i²/d_i}
ū_i(t)            =  1 − (1 + (d_i/k̄_i) t)^{−k̄_i²/d_i}          (→ 1 − exp(−k̄_i t) as d_i→0)
```

Peptide prediction `Û_p(t) = Σ_i M_{p,i} ū_i(t)`. Objective (`fit_curve_moment_variance`):

```
L(β)  =  mean_{(p,t)∈mask} ( Û_p(t) − U_p^obs(t) )²   +   λ · Penalty(β, R)
```

Here `R` (the geometry) enters **only through the regulariser** (§4.3); the likelihood sees the
diagonal `D` alone. This is the estimator that wins on **synthetic** TeaA.

### 4.2 Estimator B — structured residual (Gaussian quasi-likelihood)

Hold the fixed-mean curve; treat residuals `r_{p,t} = U_p^obs(t) − U_p^{mean}(t)` as zero-mean
Gaussian with residue-rate covariance `C = D½ R D½` propagated through the uptake Jacobian
(`fit_structured_residual_variance`, `propagated_uptake_covariance`, `structured_residual_nll`):

```
J_t   =  diag( ∂u_i/∂k_i |_{k̄_i} )  =  diag( t · exp(−k̄_i t) )      (effective-rate coordinate)
Σ_t   =  M J_t C J_t M^T  +  ε I ,        ε = noise_variance = 1e-4
NLL   =  Σ_t  ½ [ r_t^T Σ_t^{-1} r_t  +  log det Σ_t  +  n_t log 2π ] / N   +   λ · Penalty(β, R)
```

The mean curve is untouched → **held-out mean-MSE ratio ≡ 1.0**. `R` enters **both** the likelihood
(via `C`) and the regulariser. This is the estimator that wins on **real** MoPrP — i.e. it explains
the departure of the observed curve from the fixed-BV mean as propagated variance, which is exactly
why any BV mean-model error leaks into `D`.

### 4.3 Geometry regulariser (graph-Laplacian smoothing of log-variance)

`_regularization_penalty(β, R)`:

```
Penalty(β, R)  =  [ Σ_{i≠j} |R_{ij}| (β_i − β_j)² ] / max( Σ_{i≠j} |R_{ij}|, 1 )   +   0.01 · mean( (β − mean β)² )
```

i.e. neighbours in `R` are pushed toward equal **log-variance**, plus a weak centering term.
`λ ∈ {0, 0.01, 0.1, 1.0}`. **Note the reduced-scope reading:** even without any covariance claim,
this is a legitimate *smoothing prior on the amplitude* `D` — "spatially/sequentially adjacent
residues have similar variance magnitude" — which is a strictly weaker and more defensible use of
geometry than "R is the covariance structure."

### 4.4 Geometry constructions `R` (all PSD, unit diagonal)

`build_rate_geometries`:

- `covariance_only`: `R = corr(Σ_uniform)`, `Σ_uniform` = uniform-weight population covariance of
  `k_{i,f}` over frames (`uniform_rate_correlation`). **Population-free** (uniform weights, no
  targets).
- `distance_only`: Wendland C2 compact-support kernel on residue coordinates,
  `φ(r) = (1 − r/r_c)^4 (1 + 4 r/r_c)` for `r < r_c`, else 0; `r_c = 8 Å`; PSD in 3-D.
- `sequence_only`: `I + ρ A`, `A_{ij} = 1` iff `|resid_i − resid_j| = 1`, `ρ = 0.25` (`≤ 0.5` ⇒ PSD).
- `covariance_distance_sequence`: `covariance_only  ∘  ½(distance + sequence)` (Schur/Hadamard
  product ⇒ PSD). The "physical" candidate.
- `identity`: `I` (control).
- `shuffled_geometry`: simultaneous row/column permutation of `covariance_distance_sequence`
  — **spectrum-preserving, locality-destroying decisive negative control.**

Full construction: `C_HDX = D½ R D½`, verified diagonal-`D` and PSD (`build_hdx_covariance`).

### 4.5 Selection and the qualification gate

Selection is **held-out HDX reconstruction only** (blinded; NMR/known truth excluded), lowest median
predictive-NLL rank among *physical* geometries; identity/shuffled are compared post hoc. The frozen
gate (`qualification_gate`) requires **all** of: held-out mean-MSE ratio ≤ 1.05; residue log-variance
Spearman ≥ 0.5; mapped-variance log-RMSE ≥ 20 % below a constant-`D` control; and **beat shuffled in
every panel**. Only the last currently fails — but it is the one that matters.

---

## 5. Reweighting-side losses (context for the eventual Q4 `diag(D)` target — NOT run yet)

For when/if `D` is certified and used to constrain an ensemble. These are the production losses the
D-only target would plug into (`jaxent/src/analysis/state_population.py`,
`jaxent/src/analysis/pf_variance.py`):

- **Mean fit** (average-first BV): `Û = uptake_from_log_pf(Σ_f w_f z_{i,f}, k_int, t)`;
  `L_mean = MSE(M Û, U_obs)` normalised by the uniform-weight baseline.
- **Weight prior**: `η · KL(w ‖ uniform)`.
- **Covariance shape loss** (full-`R` version, to be *replaced* by a `diag(D)` match in the reduced
  scope) — projected symmetric log-Euclidean between correlations:
  ```
  P              = overlap_projection(M)                # non-redundant peptide cosine-overlap modes
  shrink(C)      = (1−α) C + α (tr C / n) I + ridge·I,  α = 0.05
  L_shape        = mean( ( logm(shrink(Pᵀ Ĉ_pred P)) − logm(shrink(Pᵀ Ĉ_prior P)) )² )
  Ĉ_pred = corr(C(w)),  Ĉ_prior = prior correlation      # scale-free
  ```
- **`diag(D)` match (the reduced-scope target)**: compare the weighted marginal effective-rate
  variance `diag(Cov_f[k_{i,f}; w])` to the inferred `d_i`, e.g. the symmetric log-ratio profile loss
  `log_ratio_profile_loss(pred, target) = mean( (log(pred/target))² )`. The pivot-consistent
  target artifacts are `_moprp_target_variance_scaled_published_20260724/` and
  `_moprp_target_variance_constrained_optimum_20260724/` (see §10.6).
- **Recovery diagnostic (never a training signal)**: `100 · (1 − √JSD₂(population(w), target))`,
  `population(w) = membership @ w`; and `ESS = 1 / Σ_f w_f²`.

---

## 6. Code & artifact pointers (for the codebase agent)

Core numerics (all tested; `jaxent/tests/unit/analysis/`):
- `jaxent/src/analysis/hdx_target_variance.py` — geometries, both estimators, mapping, gate, PSD
  safeguards. **The whole target-variance experiment lives here.**
- `jaxent/src/analysis/pf_variance.py` — `uptake_from_log_pf`, `uptake_log_pf_jacobian`,
  `weighted_population_covariance`, `shrink_covariance`, `projected_log_euclidean_covariance_loss`,
  `overlap_projection`, `jensen_shannon_recovery_percent`.
- `jaxent/src/analysis/state_population.py` — targets, `w_NMR`, `correlation_of`,
  `correlation_shape_loss`, recovery.
- `jaxent/src/analysis/elastic_network.py` — `anm_covariance`, `gnm_covariance` (Stage C priors).
- `jaxent/src/models/HDX/forward.py`, `jaxent/src/models/func/{contacts,uptake}.py`,
  `jaxent/src/models/config.py` — the BV forward model, contacts, intrinsic rates, radii.
- `jaxent/src/data/splitting/sparse_map.py`, `jaxent/src/interfaces/topology/` — the peptide map `M`.
- **(added 2026-07-23) `jaxent/src/analysis/hdx_ex2.py`** — the isotope-envelope forward model:
  `PeptideExchangeMap` (row-normalized `M`, enforced), `load_intrinsic_rate_file` (the actual MoPrP
  k_int source), `peptide_deuteron_count_distribution` / `thin_deuteron_count_distribution` /
  `convolve_isotope_and_deuteron_distributions` (Poisson-binomial envelope, quench thinning, isotope
  convolution). See §9.5.
- **(added 2026-07-23) `jaxent/src/analysis/hdx_rate_mixture.py`** — a third, separate shared-rate-
  mixture peptide kinetic embedding, independent of both the BV/D-only and EX2/envelope tracks.

Runners (`jaxent/examples/2_CrossValidation/fitting/jaxENT/`):
- `validate_moprp_target_variance.py` (reads `published_bc/bh` from settings — the corrected re-run
  passed scaled_published / constrained_optimum), `investigate_moprp_target_variance_sweep.py`
  (hardcodes `common.PUBLISHED_BC/BH` at the call site — override there for a corrected sweep),
  `_moprp_recovery_common.py` (`PUBLISHED_BC=0.35`, `PUBLISHED_BH=2.0`, feature/rate loaders).
- **(added 2026-07-23) `investigate_moprp_ex2_physics.py`** — the EX2/envelope/rate-mixture runner;
  `_peptide1_envelope_scores` already scores real peptide-1 raw spectra. Output:
  `_moprp_ex2_physics_bv_v2/` (`peptide1_envelope_calibration.json`, `peptide1_envelope_scores.csv`).
  See §9.5.

Latest corrected artifacts (2026-07-23):
- `_moprp_target_variance_scaled_published_20260723/` and `_moprp_target_variance_constrained_optimum_20260723/`
  (`diagnostic_decision.json`, `blinded_hdx_sweep.csv`, `nmr_pseudotruth_diagnostic_metrics.csv`).
  Both: `diagnostic_variance_gate_passes=false`, `beats_shuffled=false`, `beats_constant=true` both
  ensembles.

Features: `fitting/jaxENT/_featurise_physics_v2/features_AF2_{MSAss,filtered}_hard.npz` (97 residues
× 500 frames). Full stage history + the coefficient re-run section: `plans/known_population_covariance_recovery.md`.

---

## 7. Open physics/statistics questions (the actual task)

### 7.1 Coordinate of `D` — the central ambiguity
Uptake constrains the **rate** distribution, but BV's mean is average-first in **log-PF** `z`.
Because `k = k_int e^{−z}` (with `k_int` frame-independent), to first order `Var_f(k_i) ≈ k_i²
Var_f(z_i)` and the uptake Jacobians differ by exactly a factor `−k`: `∂u/∂z = −k · ∂u/∂k`
(`uptake_log_pf_jacobian` vs the effective-rate `J_t` in §4.2). **Question:** in which coordinate
(`k`, `z = log PF`, or `log k`) is `D` most identifiable and physically meaningful, and is the fixed
mean `k̄_i` currently consistent with that choice (average-first vs average-after; Jensen gap)? This
is the most likely place for a silent inconsistency.

**Update (2026-07-24):** the canonical pivot is closed as
`k(z̄) = k_int·exp(−E_f[ln PF])`, matching production `average_first_uptake`. Both MoPrP
runners use this uniform-frame pivot; regenerated D-only artifacts are the `_20260724`
directories listed in §10.6. The archived `_20260723` directories remain unchanged as
provenance.

### 7.2 What physically *is* `d_i`?
Since `k_int,i` is conformation-independent, `Var_f(k_i) = k_int,i² · Var_f(e^{−z_{i,f}})` — `D` is
**entirely** the spread of protection across conformers, scaled by chemistry. So `D` *is*
conformational heterogeneity in contacts. **Question:** what magnitude/pattern of conformational
contact spread produces a *detectable* curve signature (departure from single-exponential), given
EX2 and the peptide summation? Where is the detectability floor?

### 7.3 Identifiability under peptide summation + EX2 (why `D` survives where full covariance didn't)
The parent investigation hit a **frame-permutation wall**: full covariance across timepoints is
rank-≈1 and permutation-degenerate. Yet the *marginal* `D` (a lower-order, permutation-invariant
second moment = curve stretching) *is* recoverable. **Question:** formalise why marginal variance is
identifiable while the correlation structure is not; characterise the resolving power of overlapping
peptides for per-residue `D`; state the conditions under which peptide-level centroid curves under-
vs over-determine `D`.

### 7.4 Separating conformational variance from BV mean-model discrepancy
The estimator flip (curve-moment on clean synthetic → structured-residual on real data, even after
the coefficient fix) means real-data `D` absorbs BV mean error. **Question:** is there any observable
or internal consistency check that separates genuine conformational rate variance from mean-model
discrepancy? (Candidates: the held-out peptide-1 envelope; cross-estimator agreement; residual
autocorrelation across timepoints.)

### 7.5 Is centroid uptake even the right observable? (isotope-envelope width)
Centroid/uptake is the **first moment** of the peptide mass distribution. Rate heterogeneity within a
peptide broadens the **isotope envelope** — a *second-moment* observable that centroids discard.
**Question:** is envelope-width (or bimodality / EX1 signature) data available or derivable for
TeaA/MoPrP, and would it identify `D` (or even `R`) where centroids provably cannot? This may decide
whether the whole `D`-from-centroids programme is well-posed or fundamentally underdetermined.

**Update (2026-07-23):** yes — envelope inference already exists (`hdx_ex2.py`) and has already been
run once on real peptide-1 raw spectra, with a mixed, construction-sensitive result. See §9.5 before
building anything new here.

### 7.6 Is the geometry failure fundamental or construction-limited?
`R = corr(Σ_uniform)` is population-free but Stage C found the *magnitude/structure* population-
dependent; the true `R` shifts with population. Distance/sequence locality helps HDX self-prediction
(smoothing) but never truth. Stage C's ANM at large cutoff (`rc = 24 Å`) reached peptide Mantel ≈ 0.6
against the target. **Question:** is there *any* population-free structural quantity whose correlation
matches the true conformational rate covariance, or is `R` intrinsically population-dependent (hence
unrecoverable without the very populations we refuse to use)? If the latter, `R` should be retired
and geometry kept only as the §4.3 smoothing prior on `D`.

### 7.7 Deep-research targets (literature)
EX1/EX2 regimes and how heterogeneity manifests in uptake vs envelope; existing methods that infer
per-residue rate distributions from peptide HDX-MS (deconvolution, HDXsite/ExPfact-style, Bayesian);
isotope-envelope information content; elastic-network (ANM/GNM) covariance as an HDX prior; whether
"marginal variance amplitude" is a recognised, defensible HDX observable in the literature.

---

## 8. Guardrails (hard constraints)

- **No reweighting, no BV-coefficient optimisation** in this phase. Fit only `D`.
- **Never** feed target frame weights, state populations, or NMR pseudo-truth into any estimator,
  cutoff, geometry, or regulariser input. Truth is read **only** after blinded inference is written,
  for evaluation.
- Report **per cell** (ensemble × coefficient setting × estimator × geometry); **never pool**.
- Keep the three controls live: **constant-`D`**, **identity**, **shuffled**. "Beat shuffled on
  truth" is the criterion that has repeatedly failed and must be a **hard** gate, not a diagnostic.
- Verify every `D½ R D½` is PSD with diagonal `D` and maps correctly through overlapping peptides;
  verify the `d_i → 0` limit reproduces the fixed-BV mean.
- MoPrP is **exhausted as a confirmatory blind** (all candidates already scored against NMR); it is
  an *exploratory* benchmark only. Any confirmatory geometry/`D` claim needs a **new external blinded
  system**. TeaA/ISO registered multi-fold qualification (not just the single-fold pilot) is the
  remaining internal gate.
- Preserve all former Stage J and pilot artifacts as provenance; the MoPrP launch guard stays
  fail-closed.

---

## 9. Codebase-grounding findings (2026-07-23)

A codebase-exploration pass cross-checked every equation in §§2–4 against the exact §6 pointers, then
traced the actual MoPrP call sites (not just the module definitions) for the places most likely to
hide drift. Two findings are corrections to this document (§9.3, §9.4), one resolves the code-side
half of §7.1 (§9.2), and one materially changes the framing of §7.5 (§9.5).

### 9.1 Confirmed accurate (no action needed)

Contacts (§2.1: `contacts.py::calc_BV_contacts_universe`, hard-cutoff `sum(dists<=radius)` at line
251, the `1/(1+(r/r0)^6)` switch at lines 228–231, chain-aware `residue_ignore` window at lines
209–220, defaults in `config.py:20-24`), log-PF/effective-rate (§2.2: `forward.py::BV_ForwardPass`
line 31, `hdx_target_variance.py::effective_rates` lines 69–78), the uptake forward map (§2.3:
`forward.py::BV_uptake_ForwardPass` line 74, `pf_variance.py::uptake_from_log_pf` lines 228–238), the
`d_i = k̄_i² exp(β_i)` parametrisation and `[-18,8]` bounds (§4), the regularisation penalty (§4.3:
`_regularization_penalty`, lines 366–376, exact match including the `0.01` centering weight), all six
geometry constructions (§4.4: `build_rate_geometries`, lines 198–236 — `r_c=8Å`, `ρ=0.25`,
Schur-product combination, permutation shuffle), the qualification-gate thresholds (§4.5), and the
`published` coefficient row `(0.35, 2.0)` (§2.5, confirmed as `PUBLISHED_BC`/`PUBLISHED_BH` in
`_moprp_recovery_common.py:54-55`) all match the transcribed equations exactly.

### 9.2 §7.1 partially resolved: `mean_rates` is a rate-space mean, not average-first-log-PF

`validate_moprp_target_variance.py:110-111`:

```python
rates = effective_rates(log_pf, inputs.k_ints)   # k_int * exp(-log_pf), per frame
mean_rates = inputs.k_ints * np.exp(-np.mean(log_pf, axis=1))  # k(z̄), uniform frame weights
```

This is now the canonical average-first-in-log-PF pivot (`k(z̄)`) used by production
`average_first_uptake`. Both runners document this choice at their `mean_rates` call sites.
**Action before Stage 5 wiring: done.** This closes Stage 4 of
`plans/research/secondorder_HDX_physics.md`.

### 9.3 Correction to §3: `M` is a row-normalized average, not a summation

§3 (pre-correction) described `M` as "the incidence (summation) over exchange-competent residues."
The code enforces the opposite: `sparse_map.py::create_sparse_map` weights each entry
`overlap_count / exp_residue_count` (line 173); `_moprp_recovery_common.py:9` documents "the
trim-one exPfact peptide map... row-normalized over active amides"; and
`hdx_ex2.py::PeptideExchangeMap.__post_init__` (lines 113–114) *hard-fails* unless
`matrix.sum(axis=1) == 1.0`, with its own module docstring stating the forward model as
`D_p(t) = mean_i[1 - exp(-k_int_i * exp(-lnP_i) * t)]` (line 6) — a mean, not a sum.
**Consequence:** the Thread-3 identifiability argument in `secondorder_HDX_physics.md` is unaffected
(rescaling every row by a known constant carries no information), but absolute-scale reasoning is
not — in particular `noise_variance=1e-4` in `propagated_uptake_covariance` (§4.2) only makes sense
as a noise floor on a *fractional* (≤1) observable, which is further evidence the row-normalized
convention is the one actually in force. Read §2.3/§3's `M_{p,i}` as `1/N_active` per member residue,
not `1`.

### 9.4 Correction to §2.4: intrinsic-rate provenance is external, not `uptake.py`

§2.4 (pre-correction) cited `uptake.py::calculate_intrinsic_rates` and described it as wrapping
`hdxrate.k_int_from_sequence`. That wrapper is actually a different function in the same file,
`calculate_HDXrate` (lines 12–89); `calculate_intrinsic_rates` (lines 93–248) is a separate
hand-rolled reimplementation with **hardcoded `pD=7.4`** (line 123), not a parameter. Neither is on
the MoPrP critical path: `_moprp_recovery_common.py` imports `hdx_ex2.load_intrinsic_rate_file`
(`hdx_ex2.py:222-239`), which reads a flat two-column (`residue_id`, `rate/min`) file — k_int
provenance for the reported MoPrP results is an **external exPfact-generated file**, not computed
anywhere in this codebase. `load_expfact_dataset`'s defaults (`experimental_pd=4.0,
intrinsic_rate_ph=4.4`, lines 292–293) are consistent with this document's "pD 4" claim, but this
does not confirm the rate file's own generation — flagged, not confirmed; would need the rate file
itself (`expfact_kint_pH4p4_298K_min.dat`, referenced in `investigate_moprp_ex2_physics.py`'s
manifest) traced back to its generating command.

### 9.5 Major finding: an isotope-envelope track already exists and has already been run (updates §7.4/§7.5)

§7.5 (pre-update) asked whether envelope-width/bimodality data is "available or derivable" for
MoPrP — it is not an open question, it is built and has already produced a real (mixed) result,
entirely outside this document's §6 pointer list:

- `jaxent/src/analysis/hdx_ex2.py` implements the full Stage-1 forward model from
  `secondorder_HDX_physics.md`: `peptide_deuteron_count_distribution` (Poisson-binomial pre-quench
  deuteron counts per peptide/timepoint, lines 374–409), `thin_deuteron_count_distribution`
  (binomial back-exchange/quench thinning, lines 412–434), `convolve_isotope_and_deuteron_distributions`
  (natural-isotope convolution to a spectrum-comparable envelope, lines 437–455).
- `jaxent/examples/2_CrossValidation/fitting/jaxENT/investigate_moprp_ex2_physics.py` (989 lines,
  committed — `ada0a75`) exercises this against **real raw peptide-1 mass spectra** (protonated
  control, 3 exchange timepoints, fully-deuterated control) via `_peptide1_envelope_scores`
  (lines 643–759).
- A third, separate track, `jaxent/src/analysis/hdx_rate_mixture.py` (shared-rate-mixture peptide
  kinetic embedding), is used alongside EX2 in the same script.

Latest local run artifacts (`_moprp_ex2_physics_bv_v2/`, 2026-07-21): back-exchange calibration is
solid (effective survival ≈0.498, control R²≈0.987), but predicted-vs-observed envelope R² is
timepoint- and BV-construction-dependent — reported per condition, not pooled:

| BV construction | t=1 min | t=60 min | t=1440 min |
|---|---|---|---|
| `BV_hard` | R²=0.77 | R²=**−1.30** | R²=0.04 |
| `BV_legacy_switched_missing_c` | R²=0.80 | R²=0.37 | R²=0.77 |

The sign flip at t≥60min is construction-dependent, independently corroborating §1's own finding
that BV mean-model discrepancy is currently confounded with what gets attributed to `D`. Only
peptide 1 has raw-spectra data recovered locally at
`jaxent/examples/2_CrossValidation/data/_MoPrP/spectra/`, downloaded from
`pacilab/exPfact`'s `validation/pep1.1.txt` through `validation/pep1.5.txt` on 2026-07-23;
the directory's `README.json` records source URLs, sizes, and SHA-256 hashes. It remains n=1
peptide and is thin evidence for any covariance claim; coverage for more MoPrP peptides or TeaA
is unconfirmed.

**Implication:** Stage 1 of `secondorder_HDX_physics.md` ("add the isotopic envelope as an
observable") is not a blank-slate build — the forward model exists and has already run once. The
next step there is diagnostic (why does R² go negative at t≥60min, and is it construction or
back-exchange calibration) before it's treated as a usable second-moment signal, per
`hdx-investigation-follow-data-not-hypothesis` (build/verify/compare/fit before concluding).

**Second-moment fork update (2026-07-23):** `diagnose_moprp_ex2_second_moment.py` performs the
centroid/shape diagnostic without fitting an estimator or reweighting frames. It consumes the
committed pre-quench distributions and recovered spectra, and writes `_moprp_ex2_second_moment/`.
The raw-spectrum anchors are reproduced: at t=60, `experimental_EX2_fit` rank 0 gives
R²=`0.996846`, while `BV_hard`/`average_first` gives R²=`-1.296763` (survival `0.498054`).
The moment table (centroid / variance) is:

| time (min) | observed | BV_hard average_first | BV_hard frame_mixture |
|---:|---:|---:|---:|
| 1 | 0.734 / 0.780 | 0.438 / 0.664 | 0.555 / 0.851 |
| 60 | 1.996 / 1.361 | 0.648 / 0.845 | 0.943 / 1.477 |
| 1440 | 2.665 / 1.314 | 1.606 / 1.360 | 1.516 / 1.600 |

At t=60, the primary 10-bin (windowed) treatment assigns 64.4% of the BV_hard/average_first SSE to
the centroid shift; after alignment its width ratio is `0.621`, versus `1.084` for frame_mixture.
Shifting on padded pre-truncation support instead assigns 94.7% of the same SSE to the centroid and
gives width ratios `0.804` (average_first) / `1.259` (frame_mixture). Either way the centroid is the
robustly dominant mismatch, so the verdict is `mean_confounded` at t=60
(`flagged_latent_width_signal=true`, `proceed_to_envelope_estimator=false`).

**Precision framing (corrected 2026-07-23):** the **expected precision of the envelope width channel
is ~20–25%** — so the `experimental_EX2_fit` ceiling's centroid-aligned width ratio of `1.213`
(21% high while fitting the shape at R²=0.997) is the *physical precision floor of the observable, not
a broken statistic*. Read against a ±25% band (≈0.75–1.25): under the faithful windowed observation
model, `average_first` (0.621) is **beyond precision — genuinely too narrow**, while `frame_mixture`
(1.084) matches observed *within* precision. That mean-only-too-narrow / conformer-spread-matches
separation is exactly the conformational second-moment signature, and it **exceeds** the precision
floor. It is **not** in-principle unresolvable; the only thing that breaks it is boundary handling —
under the padded mode `average_first` (0.804) re-enters the band and the separation collapses. So the
limiter is the width-metric's **boundary convention**, not measurement precision. The latent
conformational second moment is therefore *plausibly present and beyond precision*, gated on a
resolvable metric-convention choice plus coverage — not on a physical wall. The result remains
exploratory because this is one peptide (§9.6 item 5).

**Stage (b) boundary resolution (2026-07-23): pass; proceed to item 5.** The ad hoc windowed/padded
pair has been replaced by one physical observation order,
`true_mass_shift_then_fixed_window`: conservatively translate the full convolution by a sub-bin
mass shift, then select and normalize the fixed observed window. The diagnostic now writes
`edge_mass_check.csv` as well as a single-convention `centroid_shape_decomposition.csv`.

- The committed 10-bin window is adequate. The last-bin and above-window intensity fractions are
  zero for both controls and every timepoint. Bin 0 carries 71.1% (protonated control), 50.3%
  (1 min), 8.85% (60 min), 2.99% (1440 min), and 3.52% (fully-deuterated control), as expected
  because it is the physical zero-isotope/deuteron boundary rather than a missing lower window.
  Raw intensity below that boundary is 0–0.918% and is recorded separately as baseline/noise;
  no widening is required at the 1% containment tolerance.
- At t=60, centroid alignment requires shifts of `+1.348` bins for
  `BV_hard`/`average_first`, `+1.054` for `BV_hard`/`frame_mixture`, and only `−0.147` for the
  `experimental_EX2_fit` ceiling; the post-alignment centroid gaps are <`5e-10` bins. The ceiling's
  aligned width ratio is `1.235`, which defines the empirical ±25% band `[0.927, 1.544]`.
  `average_first` remains beyond precision on the narrow side (`0.788`), while `frame_mixture`
  remains within precision (`1.120`). The separation survives the faithful convention **but is
  reinterpreted after the item-5 synthetic control (2026-07-24): it is attributed to BV mean-model
  under-dispersion, not conformational heterogeneity — see the reinterpretation note below.**
- The mean confound is not removed: centroid alignment explains 92.6% of the t=60
  `BV_hard`/`average_first` SSE (76.1% for `frame_mixture`). The Stage-(b) result is therefore
  `mean_confounded_with_independent_width_signal`, not permission to build the estimator.
  `second_moment_verdict.json` records `proceed_to_item_5=true`,
  `proceed_to_envelope_estimator=false`.

### 9.6 Updated priority order given 9.1–9.5

1. Fix the pivot convention (§9.2) — **done**; `k(z̄)` is implemented and the `_20260724`
   artifacts supersede `_20260723` while retaining the latter as provenance.
2. Diagnose the existing envelope run (§9.5) before writing any new envelope-inference code — **done**.
   `diagnose_moprp_ex2_envelope.py` consumes the committed `_moprp_ex2_physics_bv_v2/` artifacts and
   writes `_moprp_ex2_envelope_diagnosis/`. The `experimental_EX2_fit` control remains the live
   positive control: with the same effective survival calibration (`0.498054`) and the same active
   peptide-1 residue set, it gives R²≈0.995–0.997 at t=1/60 min and ≈0.973 at 1440 min. The anchor
   is reproduced at t=60 (`BV_hard` R²=−1.2968; EX2-fit R²=0.9968). Thus back-exchange calibration
   (a) and residue activation (b) are ruled out. BV's predicted centroid already misses the EX2-fit
   reference, and the residue-level t=60 probability comparison remains construction-sensitive
   (`BV_switched` is closer than `BV_hard` but still mismatched): the cause is `bv_mean_model`.
   This is the same BV mean-model discrepancy already confounding `D` (§1, §7.4), re-exposed by a
   more sensitive observable, not a new failure mode.
3. **Gated — deferred (one condition passed, two remain):** the second-moment fork is
`mean_confounded_with_independent_width_signal` at the max-information point (t=60). The
conformer-spread width signal is beyond the ~20–25% expected precision and survives the resolved
physical observation convention (§ above), so it is plausibly real rather than a precision or
boundary artifact. Do **not** add the envelope as a third estimator in `hdx_target_variance.py`
until **all three** conditions hold:
   - **(a) BV mean-model correction** — the centroid explains 64–95% of t=60 SSE, so an envelope
     estimator would re-fit mean error, not width, until the mean is fixed (this is the same
     `bv_mean_model` confound as §1/§7.4; note it lies outside the D-only guardrails, which forbid BV
     coefficient optimisation).
   - **(b) width-metric boundary convention resolved — passed 2026-07-23.** Edge-mass containment
     validates the 10-bin window, and full-support sub-bin translation followed by that fixed window
     retains the t=60 separation (`average_first=0.788`, `frame_mixture=1.120`; empirical ceiling
     band `[0.927, 1.544]`).
   - **item 5 — coverage beyond peptide 1** — n=1 cannot distinguish a real signal from a
     single-peptide coincidence.
   **Item 5 synthetic control (2026-07-24): failed closed.**
   `validate_second_moment_synthetic.py` writes
   `_moprp_ex2_second_moment_synthetic/` and exercises the shared Stage-(b) decomposition on
   independent observed/model frame draws for N={4,6,9}, ten injected log-PF variance levels
   (`D_true=0...3.2`), and three seeds. The zero-heterogeneity negative control passes for every
   geometry/seed: no separation is called, and the maximum average-first/frame-mixture width-ratio
   gap is `1.78e-15`. The physical resolved convention is at least as conservative as the retired
   padded sensitivity (both call zero positives).

   The forward chain does carry the injected truth: the frame-mixture aligned width ratio stays
   near one, the average-first ratio decreases monotonically, and their recovered width excess has
   Spearman `rho=1.0` in every geometry. However, the required per-envelope EX2-analog ceiling
   fitted with `fit_ex2_solution_set` narrows in parallel with average-first. Its ratio falls from
   about one at `D_true=0` to `0.718/0.596/0.466` at the most extreme draws (N=4/6/9), instead of
   remaining in the expected `1.0--1.25` precision regime. Consequently its ±25% band follows the
   missing width, `separation_survives` is false in all 90 cases, and no finite `D_min` exists under
   the specified decision rule. This remains true across the range implied by peptide-1's selected
   D-only artifacts: converting `d_i` to the comparable log-rate coordinate as
   `log(1+d_i/kbar_i^2)` gives medians `2.964` (AF2_MSAss) and `2.045` (AF2_filtered).

   `synthetic_second_moment_validation.json` therefore records `method_validated=false`,
   `finite_detectability_floor=false`, and `ex2_floor_sanity_passes=false`. This does not erase the
   observed monotonic width channel; it shows that the same-envelope EX2 ceiling is not an
   independent precision floor and makes the current beyond-precision decision self-masking on
   known truth. Per the mandatory negative/positive control gate, item 3 remains false and is
   **not** gated solely on stage (a). Revisit the decomposition/precision calibration before any BV
   mean-fix or envelope-estimator work.

   The sequenced next action is therefore to repair and independently validate the precision-floor
   definition, not stage (a).

   **Reinterpretation of the real peptide-1 Stage-(b) signal (2026-07-24).** The synthetic control
   shows *why* the real signal is not what it appeared. EX2 fits a single per-residue PF vector — it
   is a single-conformer / mean-structure model and physically cannot represent conformational
   (across-frame) heterogeneity. Under genuine injected heterogeneity the EX2-analog width ratio
   therefore **narrows** in lockstep with `average_first` (falls to `0.60–0.72`). But on real
   peptide-1 the EX2 ceiling stayed **wide** (`1.235`), and EX2 reproduced the observed envelope at
   R²≈0.99 — i.e. the real observed envelope is well described by a *single conformer* and carries
   **no conformational-mixture broadening**. Consequently `average_first` being narrow on real data
   reflects **BV under-dispersing its per-residue rates within the mean structure** (a contact /
   mean-model deficiency), **not** conformational `D`. The apparent "independent width signal" is the
   same `bv_mean_model` confound as §1/§7.4, re-exposed once more — the envelope did **not** certify
   conformational second-moment signal for MoPrP peptide-1.

   **Re-gate for item 3.** Item 3 is no longer gated solely on stage (a). It now requires, in order:
   (i) **replace the EX2-analog precision floor with a measurement-noise-based floor** (ion-counting /
   replicate spread — note the synthetic ran with `poisson_counting_noise=false`, so EX2 was the only
   precision source, which is why the degeneracy was total); (ii) re-evaluate real peptide-1 under
   that floor; (iii) only if a signal survives, the stage-(a) BV mean fix (out of D-only scope) and
   item-5 coverage. Given real peptide-1's single-conformer-reproducible envelope, the expected
   finding under a corrected floor is *little conformational signal to detect* — so item 3 moves from
   "parked" toward **"retire unless re-specified."**
   Stage (a) remains explicitly outside the D-only guardrails and must not start without a separate
   scope decision. If all gates eventually pass, reuse the frozen `identity`/`shuffled_geometry`
   controls and `qualification_gate` so a real "R beats shuffled on truth" result is comparable to
   the existing verdict table, not a new incommensurate metric.
4. Peptide 1 is already both the D-only held-out peptide (`peptide_partitions` in
   `validate_moprp_target_variance.py:56-67`) and the one with real envelope spectra — use this
   doubly-independent channel deliberately rather than as an audit-script side effect.
5. Check envelope-data coverage beyond peptide 1 (more MoPrP peptides, or TeaA) before leaning on
   n=1 for any R claim.

## 10. Stage 4: pivot-convention experiment

### 10.1 Definitions and canonical pivot choice

- `k̄_after_i = E_f[k_i]` (rate-space mean over frames).
- `k̄_first_i = k_int,i · exp(-E_f[ln PF_i])` (average-first-in-log-PF).
- Canonical Stage-5 target for a future rerun is `k̄_first` to match production `average_first`
  uptake semantics; this is the boundary documented at the `infer_blinded()` pivot line in
  `validate_moprp_target_variance.py`.
- Stage 4 is closed: both runners now use `k̄_first = k(z̄)` with uniform frame weights.
  The deferred diagnostic sweep is code-fixed; its artifacts remain pending as an optional
  parallelized rerun.

### 10.2 Tier-1 protocol (fast census)

- Load fixed per-ensemble blinded MoPrP inputs via `_moprp_recovery_common.load_blinded_ensemble_inputs`
  and coefficient settings from `_moprp_recovery_coefficient_lock/coefficient_lock.json`.
- For each coefficient setting and ensemble:
  - Build `rates = effective_rates(log_pf, k_ints)` with `log_pf = bc*heavy + bh*acceptor`.
  - Compute `k̄_after` and `k̄_first`.
  - Write `pivot_gap.csv` with per-cell median / p90 / max of:
    - `rel_gap = abs(k̄_after-k̄_first)/k̄_first`,
    - `jensen_guard = k̄_after >= k̄_first` (tolerance check),
    - `abs(rel_gap - 0.5·Var(log_pf))` sanity residual.
- Tier-1 thresholds: `median_rel_gap<=0.02`, `p90_rel_gap<=0.05`.
- Replay a single archived `E_f[k]` cell with the same primary estimator/geometry/regularisation to
  lock script wiring; abort on replay mismatch (`RMSE > 1e-6` or objective diff > 1e-6).

### 10.3 Tier-2 protocol (conditional)

- Run only when Tier-1 fails or `--force-tier2`.
- For each cell, estimator, and geometry (`primary_geometry_from_artifact`, `shuffled_geometry`):
  - fit with `k̄_first` (`fit_curve_moment_variance`, `fit_structured_residual_variance`),
  - no BV tuning, no reweighting, no NMR inputs,
  - holdout score via peptide-1 only (`heldout_mean_mse_ratio`),
  - constant-D control always included.
- Compare each row to archived `E_f[k]` counterpart by:
  - `log_variance_spearman`, `mapped_variance_log_rmse`, `constant_mapped_variance_log_rmse`,
  - mapped `d_i` ordering and `Δβ = log(d_i / k̄^2)` summaries.

### 10.4 Decision logic

- Build `pivot_refit_gate.csv` with rows in `qualification_gate` schema plus:
  - `coefficient`, `panel="pivot"`, comparison deltas (`objective_vs_archived`, `mapped_rmse_vs_archived`,
    `d_rmse_vs_archived`, `d_ordering_spearman_vs_archived`, `beta_delta_*`).
- Call:
  - `qualification_gate(..., required_panels=("pivot",), required_ensembles=("AF2_MSAss","AF2_filtered"))`
- `pivot_decision.json` should include `tiers_executed`, full Tier-1 counters, replay rows, and Tier-2
  qualification/cell decisions.
- Set `decision`:
  - `"future_correctness"` if all thresholds/gates pass and comparison deltas are non-degrading.
  - `"investigation_wide"` if any gate check fails or any archived comparison degrades beyond tolerances.

### 10.5 Expected output shape

- Expected per-cell gate rows: 16 (`2 coefficient settings × 2 ensembles × 2 estimators × 2 geometries`).
- Output files:
  - `_pivot_convention/pivot_gap.csv`,
  - `_pivot_convention/pivot_refit_gate.csv` (Tier-2 only),
  - `_pivot_convention/pivot_decision.json`.
- Placeholder keys to persist in `pivot_decision.json`: `decision`, `tiers_executed`, `tier1`,
  `replay`, `tier2`, `pivot_gap_path`.

### 10.6 Closure and regenerated artifacts (2026-07-24)

- **Pivot closed:** `k(z̄)` is the canonical D-only coordinate. Both runners are fixed, and
  manifest provenance records the actual frozen coefficient setting.
- The selected D-only artifacts in
  `_moprp_target_variance_scaled_published_20260724/` and
  `_moprp_target_variance_constrained_optimum_20260724/` supersede the matching `_20260723`
  directories, which are retained unchanged as provenance. Section 5's future `diag(D)` target
  points to these `_20260724` artifacts.
- Verdicts are pivot-invariant: both settings retain
  `diagnostic_variance_gate_passes=false`, `beats_constant=true`, and overall
  `beats_shuffled=false`. New truth-recovery medians are scaled-published
  (MSAss 0.837, filtered 0.863) and constrained-optimum (MSAss 0.811, filtered 0.845).
- D ordering shifts localize to AF2_MSAss (structured-residual replay ≈0.76–0.79 versus archived;
  AF2_filtered ≈0.97–0.98). The AF2_MSAss objective is near-flat (d-RMSE 0.0029 at
  Δobjective 3e−14), so the reshuffle is not treated as fully pivot-driven signal.
- Deferred 2-D sweep: **code-fixed, artifacts pending**. Run its four independent branches in
  parallel when needed; it is not required for closing Stage 4 or selecting `diag(D)`.

## 11. Information-weighted D-only timepoints

Uniform objective mass is materially misaligned with the single-exponential sensitivity. On the
committed canonical-pivot MoPrP inputs, the five lowest-information timepoints receive about 33%
of uniform objective mass but only about 15% of summed Fisher information in
`scaled_published` (about 27% / 8% for `constrained_optimum`). The dominant mismatch is the early
end: four timepoints with `t <= 1 min` and `u <= 0.1` receive about 27% of objective mass but only
about 9% of information; saturation is a smaller issue for MoPrP. Information peaks near
160--240 min (`u` about 0.66--0.72).

The D-only estimators now expose opt-in `timepoint_weighting="fisher"` and an explicit
`timepoint_weights` override. `fisher_timepoint_weights` uses only the fixed mean rates and
computes the residue-summed `(k̄ t exp(-k̄ t))²` profile, normalized to the number of timepoints.
The default uniform branch is retained for exact archived-artifact replay, and fit provenance
records the selected weighting. This reallocates objective emphasis; it does not change geometry
`R`, address the frame-permutation/R-identifiability wall, or implement parked envelope item 3.
The related epsilon-noise-floor / heteroscedastic-noise question remains distinct and is tracked in
`plans/hdx_heteroscedastic_nll_investigation.md`.

The per-cell experiment is in
`examples/2_CrossValidation/fitting/jaxENT/investigate_moprp_timepoint_weighting.py` and writes
`_moprp_timepoint_weighting/timepoint_weighting_decision.json`. The `_20260724` uniform anchors
all replayed exactly (`d_rmse=0`, objective differences at machine precision). Across both
coefficient settings, both ensembles, and primary/shuffled geometries, Fisher weighting increased
curve-vs-structured D Spearman agreement by `0.0013--0.0308`, did not reduce NMR truth-recovery
Spearman, and improved the mean mapped-profile RMSE delta in every cell. Per-cell decisions are
therefore `reduces_leakage`; `beats_shuffled` remains false throughout, including AF2_MSAss, whose
near-flat marginal-D caveat is not treated as resolved. The headline agreement deltas and all
controls remain per-cell in the emitted CSV/JSON rather than pooled.

**Bounded reading (do not over-read `reduces_leakage`).** On the *physical* primary geometry
(`covariance_only`) the cross-estimator agreement gain is small — about +0.001–0.009 Spearman per cell;
the larger deltas up to +0.031 are on the `shuffled_geometry` control and are not physically meaningful.
The substantive support for the leakage mechanism is not magnitude but an **asymmetry**: the
structured-residual estimator — the mean-contaminated one that is selected on real MoPrP — improves its
NMR truth-recovery Spearman (e.g. scaled/MSAss 0.837→0.852) and its mapped RMSE (~3–6% lower) in every
cell under Fisher weighting, while curve-moment is essentially unchanged (~0.889→0.889). That is exactly
what de-contaminating the leaky estimator predicts, and 8/8 same-sign is unlikely under a true null.
Net: Fisher weighting is a directionally-correct, downside-free **refinement** that slightly
de-contaminates the structured estimator; it does **not** dent the dominant BV mean-model confound, and
`beats_shuffled` stays false. The `reduces_leakage` label should be read as "small, consistent, no
downside," resting on cross-cell consistency plus the estimator asymmetry — not on any single cell's
delta (a +0.0013 cell is within noise).

---

## 12. Investigation checkpoint (2026-07-24)

Three independent tracks pursued after the D-only verdict have all terminated on the **same
finding: BV mean-model error is the dominant confound**, and none rescued a certified conformational
geometry or a second moment beyond it.

| Track | Status | Terminal finding |
|---|---|---|
| **Pivot convention** (§10) | Closed | `k(z̄)` canonical; verdict pivot-invariant; D shift localizes to AF2_MSAss (near-flat/weakly identified). No new signal. |
| **Envelope second moment** (§9.5–9.6, §above) | Effectively retired | Negative t≥60 R² = BV mean-model error (item 2). Stage-(b) width "signal" downgraded to BV per-residue **under-dispersion**, not conformational `D` (item-5 synthetic proved the EX2 precision floor is self-masking; real peptide-1 envelope is single-conformer-reproducible). |
| **Fisher timepoint weighting** (§11) | Landed as opt-in | Small, consistent, downside-free de-contamination of the structured estimator; does not dent the confound; verdict unchanged. |

**Consolidated picture.** HDX centroid curves identify the marginal variance **amplitude** `D`, but
on real MoPrP that `D` is currently a **model-discrepancy** quantity (BV mean error absorbed), not
certified conformational variance (§1). The geometry `R` remains unrecovered (never beats shuffled on
truth). The isotope envelope — the one observable that could in principle carry a second moment
centroids discard — shows, for MoPrP peptide-1, **no conformational-mixture broadening** a single
conformer can't reproduce; its apparent width signal is the same BV mean-model deficiency again.

**What this means for next steps.** Every remaining lever now points at the **BV mean model itself**
(contact→log-PF map: hard-count features, `Bh→0`, per-residue rate under-dispersion), which is
**outside the D-only guardrails** (§8 forbids BV coefficient/feature optimisation). So the D-only
scope is at a genuine checkpoint: within it, the productive work (pivot, weighting) is done and the
verdict is stable; beyond it, progress requires an explicit decision to reopen BV mean-model work as
its own investigation. **Stage 6 below is that scoped reopening: the joint-BV phase relaxes only the
BV `(Bc,Bh)` freeze.** The envelope thread should not be resumed without first replacing the
EX2-analog precision floor with a measurement-noise floor (§9.6 item 3 re-gate).

**Data-provenance note.** MoPrP raw envelope spectra exist for **peptide 1 only**
(`data/_MoPrP/spectra/`, from `pacilab/exPfact` `validation/`); other peptides have centroids but no
raw spectra. Any confirmatory second-moment or `R` claim needs a new external blinded system (§8);
MoPrP is exhausted as a confirmatory blind and TeaA/ISO registered multi-fold qualification remains
the open internal gate.

## 13. Stage 5 — `diag(D)` reweighting (2026-07-24)

The terminal in-scope D-only deliverable is implemented in
`examples/2_CrossValidation/fitting/jaxENT/moprp_diag_d_reweighting.py`. It loads the selected
HDX-only `structured_residual / covariance_only / λ=0.1` candidate from each `_20260724` artifact,
asserts the candidate's feature-residue hash and 97-residue shape, and reweights each
ensemble × coefficient cell independently. The mean term is production average-first `k(z̄)`;
the predicted target is the weighted marginal effective-rate variance
`diag(Cov_f[k_int exp(-log-PF)])`. `baseline`, `full_R_shape`, `diag_d_absolute`, and
`diag_d_scalefree` are all retained as separate arms. Recovery and ESS below are post-fit
validation diagnostics only; they never entered loss construction or gamma/eta selection.

The completed run used the full five-fold × gamma/eta grid with four parallel cell workers and a
bounded execution budget of 400 Adam steps and one start per optimization (`reweighting_manifest.json`;
the runner defaults remain 2000 steps and two starts). Selected per-cell recovery percent / ESS:

| coefficient / ensemble | baseline | full_R_shape | diag_d_absolute | diag_d_scalefree |
|---|---:|---:|---:|---:|
| scaled_published / AF2_MSAss | 56.2 / 2.3 | 40.6 / 1.0 | 30.4 / 7.9 | **90.1 / 2.0** |
| scaled_published / AF2_filtered | 92.1 / 1.1 | 89.4 / 1.0 | 52.4 / 13.5 | 88.5 / 4.6 |
| constrained_optimum / AF2_MSAss | 72.7 / 1.6 | 90.1 / 1.0 | 65.8 / 9.0 | **88.4 / 1.1** |
| constrained_optimum / AF2_filtered | 66.4 / 1.9 | 90.1 / 1.0 | 54.3 / 11.9 | 88.0 / 8.4 |

Absolute amplitude matching is not supported by this reweighting evidence: its selected gamma
failed the 1.05 held-out mean-MSE gate in every cell (fallback selection), and its recovery fell
below baseline in every cell despite its larger ESS. The scale-free arm is the better-supported
target form: it improved recovery over baseline in three of four cells and consistently avoided
the approximately unit ESS produced by the retired full-`R` shape arm. Its ESS is still low in
absolute terms, and it also fails the mean gate in both AF2_filtered cells, so this is evidence for
the scale-free form rather than a claim of a generally safe production prior.

The new `scale_free_log_ratio_profile_loss` is covered in
`tests/unit/analysis/test_pf_variance.py`; the focused variance/reweighting tests pass, and the
analysis suite passes (`198 passed`, with only existing Beartype deprecation warnings). This closes the D-only program: on real
MoPrP, `D` remains partly model-discrepancy (§1/§12), so the shipped prior is honest-but-imperfect.
Its confirmatory test must use the new external blinded system, not MoPrP; reopening BV mean-model
work is outside the D-only verdict. Stage 6 below is the explicitly scoped reopening of that work.

## 14. Stage 6 — joint-BV phase (2026-07-24)

Stage 6 is the named follow-on to the exhausted D-only scope. It relaxes exactly one guardrail: the
shared BV coefficients `(Bc,Bh)` are fitted jointly with per-ensemble frame logits. Blind inference
and selection, per-cell analysis (never pooled), the live controls, peptide-1 holdout, and the frozen
non-circular target discipline remain in force. NMR/state information is used only for post-fit
recovery, ESS, and decoy diagnostics. The phase tests three falsifiable questions:

1. whether the mean can adapt enough for a frozen `diag(D)` target to pass the mean gate;
2. whether the covariance target identifies `Bh>0` where mean-only fitting approached the `Bh=0`
   degeneracy; and
3. whether absolute `diag(D)` matching becomes viable once the mean is free to move.

### 14.1 Experiment 1 — frozen scaled-published target, coarse v1 first pass

`moprp_joint_diag_d_fit.py` writes
`examples/2_CrossValidation/fitting/jaxENT/_moprp_joint_diag_d_fit/` (the preserved v1 directory). It loads the selected
structured-residual / covariance-only / λ=0.1 residue target from
`_moprp_target_variance_scaled_published_20260724` once per ensemble, drops peptide 1 from the mean
map, and optimizes the two arms (`diag_d_absolute`, `diag_d_scalefree`) over
`γ={0,.01,.03,.1,.3,1,3,10,30}` and `η={0,.01,.1}`. Every row records fitted `(Bc,Bh)`, mean
gate, recovery, ESS, decoy, and `val_diag_d_loss`; `cliff_comparison.csv` compares pass boundaries
with the fixed-coefficient Stage-5 raw sweep.

The recorded artifact used the bounded execution budget in its manifest (`400` Adam steps, one
start; runner defaults remain 2000 steps and five starts). The γ=0/η=.01 full-budget ablation was
also replayed against `moprp_joint_reweight_fit.py`: both give `(Bc,Bh)=(0.23925,0.15026)` and
per-ensemble `val_mse` 0.029034 (AF2_MSAss) / 0.034382 (AF2_filtered), establishing the mean-only
anchor. The bounded grid's largest passing cells were:

| arm / ensemble | pass-boundary γ | fitted `(Bc,Bh)` at boundary | `val_diag_d_loss` | recovery / ESS | decoy |
|---|---:|---:|---:|---:|---:|
| absolute / AF2_MSAss | 0 | (0.20849, 0.85335) | 16.093 | 57.1% / 4.0 | 0.167 |
| absolute / AF2_filtered | 0 | (0.20101, 0.87200) | 71.062 | 91.0% / 1.0 | 0.0004 |
| scale-free / AF2_MSAss | 0.03 | (0.20888, 0.51946) | 3.411 | 89.7% / 1.1 | 4.0e-6 |
| scale-free / AF2_filtered | 0.01 | (0.19909, 0.80294) | 2.398 | 88.4% / 1.8 | 0.0032 |

The gate cliff therefore moved only for the scale-free arm: the joint pass boundaries were 0.03
(AF2_MSAss) and 0.01 (AF2_filtered), versus Stage-5's 0.1 in AF2_MSAss and no tested positive
gamma in AF2_filtered. Absolute matching did not recover: no positive-gamma absolute row passed
the joint mean gate, and no positive Stage-5 absolute gamma passed either. The free mean remained
positive in every bounded cell; the scale-free covariance-constrained cells moved to
`Bh=0.519--0.803`, above the full-budget mean-only `Bh=0.150`, so the result supports movement away
from the mean-only near-zero-H-bond solution but does not by itself establish unique H-bond
identification. The large absolute losses (16.1 and 71.1 at the γ=0 rows shown) likewise do not
support absolute amplitude matching.

These are joint-BV results, not a revision of the D-only verdict: the extra mean degrees of freedom
make the test less falsifiable, and all coefficient movement is auditable in the row-level CSV.

The v1 result was deliberately a coarse first pass: one split, η only at `{0,.01,.1}`, and a bounded
`400`-step / one-start artifact. Its ESS≈1 gate-valid cells therefore did not resolve whether the
frontier was a split artifact, an η-resolution artifact, or an optimizer basin.

### 14.2 Experiment 1 refined re-run — split replicates and finer η

The refined artifact is preserved at
`examples/2_CrossValidation/fitting/jaxENT/_moprp_joint_diag_d_fit_replicated/`; the attempted
diagnostics-first regeneration is
`examples/2_CrossValidation/fitting/jaxENT/_moprp_joint_diag_d_fit_replicated_diag/`. It preserves the
same frozen target, arms, blind/per-cell controls, peptide-1 holdout, and 1.05 mean gate, but uses
`γ={0,.01,.03,.1,.3,1,3}`, `η={0,.01,.022,.046,.1}`, three disjoint diagonal interleaved
peptide×timepoint split pairs, and the agreed `2000` Adam steps / `5` starts. The final directory
contains split rows (`joint_diag_d_fit_replicates.csv`), mean±std aggregates
(`joint_diag_d_fit.csv`), and `restart_diagnostics.csv`; the production run used four parallel
workers and was merged only after all 420 expected split/arm/grid/ensemble rows were present.

At the conservative cliff boundary (all three held-out replicates must pass), the aggregate rows are:

| arm / ensemble | pass-boundary γ | fitted `Bc` | fitted `Bh` | held-out `val_mse` | ESS | recovery | `val_diag_d_loss` |
|---|---:|---:|---:|---:|---:|---:|---:|
| absolute / AF2_MSAss | 0 | 0.224±0.032 | 0.387±0.482 | 0.03588±0.01468 | 1.08±0.06 | 53.5±37.5% | 33.56 |
| absolute / AF2_filtered | 0 | 0.224±0.032 | 0.387±0.482 | 0.03624±0.01042 | 1.002±0.0001 | 60.3±39.5% | 134.02 |
| scale-free / AF2_MSAss | 0.03 | 0.214±0.011 | 0.429±0.229 | 0.03264±0.01359 | 1.003±0.0005 | 88.0±0.01% | 3.21 |
| scale-free / AF2_filtered | 0.01 | 0.199±0.004 | 0.770±0.381 | 0.03308±0.01269 | 1.002±0.0003 | 87.9±0.001% | 2.19 |

The ESS/gate frontier at the scale-free boundary γ is the decisive readout (entries are
`mean_gate_passed` fraction, then ESS mean±std across the three splits):

| ensemble / γ | η=0 | η=.01 | η=.022 | η=.046 | η=.1 |
|---|---:|---:|---:|---:|---:|
| AF2_MSAss / .03 | 1 / 1.00±0.00 | 1/3 / 9.81±6.20 | 1/3 / 67.65±32.83 | 1/3 / 249.94±7.95 | 1/3 / 343.57±1.02 |
| AF2_filtered / .01 | 1 / 1.00±0.00 | 2/3 / 9.27±10.29 | 1/3 / 137.19±81.91 | 1/3 / 338.05±22.85 | 1/3 / 436.14±2.16 |

ESS is monotone in η for every arm/ensemble/γ cell in the refined aggregate. There is no cell with
both all-three-replicate gate validity and ESS≥5: the all-pass boundary remains at ESS≈1, while the
first healthy-ESS points lose one or more held-out replicates. Thus the gate↔ESS anti-alignment
survives finer η resolution and split replication; it is not explained by v1's single split or coarse
η grid. Absolute matching still has no positive-γ all-replicate pass (although isolated one-third
passes occur), so absolute recovery is not supported. `Bh` is positive but not stable across splits
at the gate boundary (`0.429±0.229` and `0.770±0.381` for the scale-free ensembles), and therefore
the covariance target still does not identify a unique H-bond channel.

At γ=0 the two arms reuse exactly the same fit in every split/grid cell, so fitted coefficients,
held-out mean metrics, gate flags, recovery, ESS, and restart diagnostics are identical; only the
reported arm-specific `val_diag_d_loss` differs by definition. Their refined free-mean anchor is
`Bc=0.224±0.032`, `Bh=0.387±0.482` at η=0; the full-data v1/mean-only anchor
`(0.23925,0.15026)` lies within the split scatter, while held-out MSE is reported separately in the
aggregate CSV. The five-start diagnostic shows substantial restart ESS spread in some cells
(maximum aggregate spread 8.68) and the best-objective restart is the lowest-ESS restart only for a
fraction of cells, so ESS≈1 is also a recurring optimization basin rather than a universally unique
minimum. This does not rescue a healthy-ESS gate cell: the pass boundary remains at ESS≈1.

The refined result changes the Experiment-2 framing. Target staleness is no longer the first blocker:
the frozen-target joint-BV phase already loses the mean gate when η raises ESS into the useful range.
Experiment 2 remains queued, but if run it should explicitly test whether block-coordinate target
updates alter that KL/ESS floor, not assume that fitted-target consistency alone will produce a
healthy-ESS gate-valid solution.

### 14.3 Joint-BV diagnostics and figures are first-class outputs

The post-fit diagnostics are implemented once in
`examples/2_CrossValidation/fitting/jaxENT/joint_diag_d_diagnostics.py` and are shared by the
frozen-target runner and the queued block-coordinate fitted-target runner. Each split row persists
`ess_<state>` and `mass_<state>` for every state in `FULL_STATE_SUPPORT`, together with the dominant
frame's raw cluster label and weight. These fields are post-hoc only: state and cluster labels never
enter the loss, mean gate, or cell selection. The aggregate CSV carries the corresponding mean/std
numeric fields and the dominant-cluster mode.

`plot_joint_diag_d_fit.py` consumes only the persisted aggregate and split tables and writes the ESS,
recovery, held-out-MSE, decoy-mass, gate-ratio-versus-η, and per-cluster ESS figures into the artifact
directory. Gate-pass fractions are drawn on heatmap cells, and split standard deviations are used for
gate-ratio error bars. The same schema and plotter are intended for Experiment 2, so its
block-coordinate rounds can be compared without a second diagnostic implementation.

The regenerated diagnostics record the qualitative cluster result that was previously only an
ad-hoc scratchpad observation: gate-valid ESS≈1 collapse lands on target-state frames, while decoy
leakage appears only as η is raised into the high-ESS regime. This supports the refined frontier read
above and does not turn the post-fit cluster assignment into evidence used by the fit. The preserved
v2 tables remain the numeric reproduction anchor. The generated `_replicated_diag` directory has the
complete first-class diagnostics and figures, but its first run under the single-threaded CPU
contention workaround differed from v2 at small floating-point levels (maximum aggregate deltas:
`Bc 1.23e-10`, `Bh 1.30e-11`, `val_mse 1.08e-8`, `recovery 2.17e-4`, `ESS 1.65e-4`). A second run in
the default environment was started for the exact anchor but exceeded the available execution window
inside the JAX scan and was stopped before writing workers. Therefore `_replicated_diag` is not yet
accepted as a numeric supersession; the mismatch is recorded as reproducibility/nondeterminism to
resolve rather than silently overwriting the v2 cited values.

### 14.4 Experiment 2 — fitted consistent target (queued)

Reserved slot. If Experiment 1 demonstrates a useful gate movement, run the block-coordinate
variant from both frozen starting priors (`scaled_published` and `constrained_optimum`): fit
`(Bc,Bh)` and weights at a frozen target, re-infer the target once with the D-only estimator, and
repeat outer rounds until coefficient and target deltas reach a fixed point. Re-inference must stay
between rounds, never inside gradient steps; report per-round deltas and an oscillation check before
interpreting fitted coefficients or recovery.
