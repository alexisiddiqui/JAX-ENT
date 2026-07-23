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
  `log_ratio_profile_loss(pred, target) = mean( (log(pred/target))² )`.
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

**Update (2026-07-23):** the code-side half is answered — see §9.2. `mean_rates` is computed as
`E_f[k]` (average-after, rate-space), which is internally consistent with both estimators but
mismatched with production `average_first_uptake` at the eventual Stage-5 reweighting boundary. The
physics question — which coordinate is most identifiable/meaningful — is still open for the
deep-research agent.

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
mean_rates = np.mean(rates, axis=1)               # E_f[k_i], uniform frame weights
```

This is a true rate-space mean (`E_f[k_i]`), **not** the "average-first-in-log-PF" convention
(`average_first_uptake`, `z̄_i = Σ w_f z_{i,f}` then transform) that production BV semantics use
elsewhere (`BV_ForwardPass.average_first=True`). That said, it is *not* a bug in isolation: both
estimators' own constructions are delta-method/two-moment expansions around the true first moment of
the rate, so `E_f[k]` is the internally-correct pivot for `curve_moment`'s Gamma closure and
`structured_residual`'s Jacobian linearisation. **The exposure is at the module boundary, not
inside it.** `E_f[k] - k(z̄) ≈ ½k̄·Var_f(z)` — a correction the *same order* as `D` itself — so when
§5's `diag(D)` target is eventually matched against the production `average_first_uptake` mean-fit
loss, the two "mean curves" will not coincide, and part of `D`'s own signal will already be absorbed
into that gap. **Action before Stage 5 wiring:** pick one canonical pivot (`E_f[k]` or `k(z̄)`) for
both the D-only fit and whatever mean-fit loss consumes `diag(D)`, and document it at the
`mean_rates` call site. This is Stage 4 of `plans/research/secondorder_HDX_physics.md` ("handle the
Jensen/coordinate bias explicitly... essentially free, do it regardless"), now scoped to one line.

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
peptide 1 has raw-spectra data referenced locally (via an external `--expfact-validation-dir` CLI
arg, not committed in-repo) — n=1 peptide is thin evidence for any covariance claim; coverage for
more MoPrP peptides or TeaA is unconfirmed.

**Implication:** Stage 1 of `secondorder_HDX_physics.md` ("add the isotopic envelope as an
observable") is not a blank-slate build — the forward model exists and has already run once. The
next step there is diagnostic (why does R² go negative at t≥60min, and is it construction or
back-exchange calibration) before it's treated as a usable second-moment signal, per
`hdx-investigation-follow-data-not-hypothesis` (build/verify/compare/fit before concluding).

### 9.6 Updated priority order given 9.1–9.5

1. Fix the pivot convention (§9.2) — cheap, do first, blocks nothing else.
2. Diagnose the existing envelope run (§9.5) before writing any new envelope-inference code — don't
   re-derive what's already been tried.
3. If the envelope signal survives (2): add it as a third estimator in `hdx_target_variance.py`,
   reusing the frozen `identity`/`shuffled_geometry` controls and `qualification_gate` so a real "R
   beats shuffled on truth" result is comparable to the existing verdict table, not a new
   incommensurate metric.
4. Peptide 1 is already both the D-only held-out peptide (`peptide_partitions` in
   `validate_moprp_target_variance.py:56-67`) and the one with real envelope spectra — use this
   doubly-independent channel deliberately rather than as an audit-script side effect.
5. Check envelope-data coverage beyond peptide 1 (more MoPrP peptides, or TeaA) before leaning on
   n=1 for any R claim.
