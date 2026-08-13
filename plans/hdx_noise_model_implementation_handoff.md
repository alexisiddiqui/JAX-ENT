# HDX noise model — implementation & investigation handoff

**Status:** handoff written 2026-08-12; amended 2026-08-13 with the finite-gating (three-state)
kinetic backend (§6, Phase 2.5, Phase 3.5), the resolved uptake-normalisation provenance
(§7, closing Phase 0.2), the HDXrate/PDLA validation plus deferred exPfact diagnostic
(§2.2, Phase 0.1, Phase 5), the peptide-1 EX1 spot check (§8, closing Phase 0.3), and the mask /
PF-conditioned residue set (§9, closing Phase 0.4). The only code written so far is the Phase 0.3
spot-check runner; no noise-model machinery has been implemented.

**Bridges** `plans/hdx_noise_model_design.md` (the frozen mathematical design, written abstracted
from the codebase) to the actual JAX-ENT code surfaces, and fixes the staged test/investigation
order. **Relates to** `plans/hdx_moprp_pivot_calibration.md` (§11–§12: the Σ pause point this
work resolves) and `plans/hdx_moprp_kint_provenance_handoff.md` (§8: rate-source obligations).

The design document is normative for the mathematics. Where this handoff records a conflict
between a design assumption and codebase reality, the conflict is listed in §2 and must be
resolved as stated there, not silently.

---

## 1. Code surfaces to reuse (verified, with locations)

### 1.1 Inputs and conventions

| need | surface |
|---|---|
| canonical inputs (features, `k_ints`, `mapping` (P,R) row-normalised, `observed_uptake` (P,T), timepoints, `w_NMR`, rate provenance) | `fitting/jaxENT/_moprp_recovery_common.py` — `load_ensemble_inputs` (:233), `EnsembleInputs`/`BlindedEnsembleInputs` (:92/:117), `RATE_SOURCES` (:46), `rate_source_provenance` (:151) |
| held-out peptide | `PEPTIDE1_INDEX = 0` in the same module; carried by every existing runner |
| published coefficients | `PUBLISHED_BC = 0.35`, `PUBLISHED_BH = 2.0` (:66) |
| BV log-PF reference arm | `inputs.log_pf_by_frame(bc, bh)` method on the inputs object |

### 1.2 PF inverse solutions (design §8)

`jaxent/src/analysis/hdx_ex2.py`: `fit_ex2_solution_set(observed_uptake, intrinsic_rates_min,
timepoints_min, peptide_map, *, starts=20, seed, log_pf_bounds, harmonic_strength=0.0,
initial_log_pf_vectors, maxiter)` (:536) returning `EX2SolutionSet` (:186) with per-start
`EX2Fit` objects (`log_pf (R,)`, `predicted (P,T)`, `objective`, `rmse`, `initialization`).
`moprp_pivot_litmus.py:229` shows the existing call pattern. Design §8.1 requires `starts ≥ 50`;
the litmus used 20 — raise it, and cluster full multivariate solutions per overlap region rather
than reusing `solution_range` (which is a per-residue min/max, not a mode structure).

`predict_ex2_uptake` in the same module is the machine-precision anchor for the strict EX2 mean
(design §14.2).

### 1.3 Sensitivities, covariance primitives (design §6, §7)

- `jaxent/src/analysis/pf_variance.py`: `framewise_uptake` (:280) and the `pr,trf->tpf` mapping
  einsum (:306) are the shape patterns for building `A_m ∈ R^{PT×R}`;
  `weighted_population_covariance` (:23); `shrink_covariance` (:39, **compatibility exports
  only** — design §19 forbids heavy post-hoc shrinkage in the primary Σ).
- `jaxent/src/analysis/state_population.py`: `peptide_uptake_covariances` (:215) produces the
  `(T,P,P)` block-diagonal comparator; `shrunk_trace_normalized_precision` (:235) is used
  **only** for the trace-normalised legacy export (design §12.3), never in the primary fit.
- ANM: `jaxent/src/analysis/elastic_network.py` — `anm_hessian` (:61), `anm_covariance` (:85),
  `gnm_covariance` (:54); tests in `jaxent/tests/unit/analysis/test_elastic_network.py`.
  Convert the ANM covariance to a **unit-diagonal correlation** before use (design §7.1 uses it
  purely as `R_ANM`); `correlation_of` (`state_population.py:265`) does this.
- CA-coordinate subsetting to the 97 feature residues: `_structure` in
  `fitting/jaxENT/moprp_covariance_linear_model.py:47` — but see the stale path in §2.3.

### 1.4 Cross-fitting and downstream substitution

- Blocked timepoint folds: `_time_folds(n_timepoints, block=3)` in
  `moprp_covariance_recovery.py:57`. Design §11.2 additionally requires that **PF modes are
  refit inside each fold** — no existing runner does this; it is new work.
- Peptide/overlap-region folds: new; peptide overlap structure comes from `inputs.mapping`
  row supports.
- Downstream substitution target (Phase 4): the real-uptake population fit in
  `moprp_population_pivot.py` (population-logit optimizer, injection point
  `pivot_observable` in `moprp_pivot_litmus.py:46`, pure JAX, three pivots).

### 1.5 Loss integration (design §12)

`jaxent/src/opt/losses.py:1517` `hdx_uptake_sigma_MSE_loss` consumes a single time-independent
`(P,P)` matrix that is **already a precision** (`Dataset.covariance_matrix` stores Σ⁻¹;
slicing via `create_covariance_mat`, `sparse_map.py:18`; `_trace_normalise` at `loader.py:202`
rescales to `trace(W)=n` at dataset creation). Consequences:

- the new `hdx_uptake_joint_gaussian_loss` / `hdx_uptake_joint_mixture_loss` must **bypass** the
  loader's covariance pathway entirely (it imposes trace normalisation, which design §10.3
  forbids in the primary likelihood) — carry the frozen Cholesky factor(s) as loss-owned
  constants, registered additively in the loss registry (`get_loss_function`, :1917);
- the trace-normalised compat export plugs into the existing pathway unchanged;
- the loss alias fix in `examples/common/losses.py` (MSE → eye) must be re-verified before any
  MSE-vs-Σ comparison (pivot calibration §7 pre-flight).

---

## 2. Design-vs-codebase conflicts found during grounding

1. **Envelope screen is data-limited to peptide 1, but per-cell empirical weights exist.**
   The only isotopic-envelope data in the repo is `data/_MoPrP/spectra/pep1.{1..5}.txt`
   (exPfact `validation/`, provenance in `spectra/README.json`) — five timepoints of the
   **held-out** peptide only. Design §5.6's per-peptide EX1 gate cannot be executed as written:
   run it on peptide 1 as a spot check of the EX2 premise, record the remaining 13 peptides as
   **unscreened** in the manifest. This is a stated limitation, not a pass.

   **However** (found 2026-08-12): `data/_MoPrP/moprp.weights` (SHA-256
   `edb0c4f1462c2397749a887a39519925bcdd1ee7a93d66aa08fdf20a18706f03`) is a 15×15 file in the
   `moprp.dexp` layout — time column plus **14 per-peptide weights per timepoint**, values
   ~12–271. Read as inverse standard deviations these imply per-cell uptake SDs of ~0.004–0.08,
   a physically plausible dfrac uncertainty scale. This is the closest thing the dataset has to
   point-specific measurement uncertainty, and design §9.4 already mandates that such data
   "replace or anchor" the generic heteroscedastic shape. Roles (see Phase 0.5 and Phase 3):
   - **anchor/validation for the acquisition-noise diagonal** `σ_exp²·D_m` — compare the fitted
     `d(μ;κ)` profile against the empirical `1/w` (or `1/√w`) surface, and optionally fit with
     `D` fixed to the empirical surface as its own hierarchy arm;
   - **not** a replacement for the EX1 gate — weights carry variance magnitude, not envelope
     shape, so the EX2-premise limitation above stands;
   - two provenance items to pin in Phase 0.5 before use: the convention (`w = 1/σ` vs
     `w = 1/σ²` — decide from upstream exPfact's cost function, not by assumption), and
     circularity (if these weights entered the shipped `median.pfact` χ² fit, they are part of
     the PF reference's likelihood and an anchoring role must be reported as such, while use as
     an *independent validation target* for our fitted diagonal remains legitimate).
2. **PDLA implementation and rate-source ordering.** The historical PDLA intrinsic-rate model
   used by exPfact is the same model implemented by HDXrate's `reference="poly"` arm, subject to
   implementation conventions (units, residue indexing, terminal handling, and pH semantics).
   This equivalence must be validated in JAX-ENT before downstream scientific interpretation.
   The validated HDXrate/PDLA vector becomes the corrected-rate sensitivity arm; existing defaults
   and artifacts remain unchanged for compatibility. The final exPfact refit using the supplied
   MoPrP kints is deferred until all primary and sensitivity analyses are complete. It is a
   diagnostic comparison only, not an input to earlier model selection.
3. **Stale ANM structure path.** `moprp_covariance_linear_model.py:38` points at
   `data/_MoPrP/MoPrP_max_plddt_4334.pdb`, which does not exist; the file is at
   `data/MoPrP_max_plddt_4334.pdb` (also in `notebooks_DEPRECATED/...`). The new runner must
   define its own `STRUCTURE` constant with the correct path + SHA-256 and not import the stale
   one. (Fixing the old script is optional and out of scope.)
4. **Multistart machinery returns solutions, not modes.** `EX2SolutionSet` is sorted by objective
   with a `solution_range`; the clustering/mode bookkeeping of design §8.1 (regional clustering,
   per-mode centres, variance profiles, uptake-space separation) is entirely new code.
5. **`np.cov`-era consumers.** `compute_sigma_real.py` and the shipped `Sigma.npz` remain in the
   tree as the broken baseline (pivot calibration §2.3). They are comparison artefacts; nothing
   new may import from them.
6. **Uptake normalisation: RESOLVED 2026-08-13 (literature provenance, user-supplied — see
   §7).** `moprp.dexp` is peptide-wise **maxD-normalised fractional uptake**, not raw Da and not
   an absolute pre-quench occupancy reconstruction. Design §4.4's open question is closed; the
   consequences are stated in §7 and must be reproduced verbatim in every manifest.
7. **The "49 of 97 residues" and "one peptide outside the overlap" claims are artifacts of the
   wrong PF reference (found 2026-08-13, see §9).** Design §4.3 bullets 2–3 state that only 49 of
   the 97 feature residues are resolved and that one peptide lies outside that overlap. Both trace
   to `_output/MoPrP_pfactors.dat`, a 49-row export that
   `plans/hdx_moprp_kint_provenance_handoff.md:176` already flags as "a training by-product". It
   drops 28 residues that are both peptide-covered and fitted, and includes residue 4, which no
   peptide covers. Against `median.pfact` — the self-consistent reference — **76 residues are
   covered and all 76 are resolved, and all 14 peptides are fully resolved**. The design document
   is frozen, so the correction is recorded here: read design §4.3 bullets 2–3 as superseded by §9.

---

## 3. Staged investigation (order fixed; each stage names what it can kill)

### Phase 0 — blocking preconditions (no model code)

| item | action | kills if failed |
|---|---|---|
| 0.1 rates | **RESOLVED 2026-08-13.** JAX-ENT explicitly calls HDXrate 0.2.2 with `reference="poly"`, `exchange_type="HD"`, `d_percentage=100.0`, and `ph_correction=False`; the native s⁻¹ vector is converted to min⁻¹ only during MoPrP materialisation. The immutable output SHA-256 is `43f9178630136a886bb9eb6e7a3ce922e398b96ff34fa15f23938271e280c83c`; manifest and residue-by-residue validation are under `_moprp_kint_provenance/validated_hdxrate_pdla/`. The existing 3Ala default remains unchanged. | resolved; validated PDLA is the corrected-rate sensitivity arm |
| 0.2 uptake provenance | **RESOLVED — see §7.** Remaining action is clerical: paste the §7.1 manifest paragraph into `_moprp_recovery_common.py` and every manifest; assert no second back-exchange factor exists anywhere in the pipeline | (was: interpretation of the peptide-persistent term — now *fixes* that interpretation, see §7.3) |
| 0.3 EX1 spot check | **RESOLVED — see §8.** Peptide 1 favours EX2 over correlated all-or-none exchange; 13/14 peptides remain unscreened | centroid-EX2 premise survives for peptide 1 only |
| 0.4 masks | **RESOLVED — see §9.** PF reference is `median.pfact`; `R` = the 76 peptide-covered residues; no peptide is dropped. Remaining action is the Phase 1 construction obligation in §9.4 | (was: silent sentinel corruption of C_z — the mechanism is real but narrower than assumed, see §9.2) |
| 0.5 weights provenance | **RESOLVED — see §10.** `moprp.weights` is `w = 1/σ`. The weights did **not** enter `median.pfact` (upstream `--weights` defaults off and the documented generating command omits it), so the Phase 3 empirical-variance comparison is non-circular. Two carry-forwards: an upstream squaring bug (§10.1) that any Phase 5 exPfact refit must avoid, and the PF-reference bias constraint (§10.3). | resolved |

### Phase 1 — machinery correctness (unit tests first, design §17.1)

New runner `fitting/jaxENT/moprp_sigma_noise_model.py` + unit tests under
`jaxent/tests/unit/analysis/` (pattern: `test_elastic_network.py`, `test_moprp_recovery_inputs.py`).
Priority order by silent-failure risk:

1. time-major vectorisation round trip (`index = j*P + p`) and block extraction against the
   `(T,P,P)` comparator — a transposed stacking stays PSD and still "fits";
2. strict EX2 mean == `predict_ex2_uptake` at machine precision; interval recursion at `a = 0`
   identical at the real irregular timepoints; **plus the finite-gating backend's EX2-limit,
   conservation, AD and degenerate-branch tests — §6.5**;
3. analytic `∂ν/∂z` vs central finite differences, including `x ≫ 1` saturation and `x ≈ 1`;
4. component invariants: every component symmetric PSD; Schur-square unit diagonal; domain-flip
   eigenspectrum preservation; likelihood → homoscedastic MSE when only `σ_exp²` survives;
5. Cholesky NLL vs dense reference; mixture `logsumexp` stability.

### Phase 2 — simulation identifiability study (the decisive investigation, design §17.2)

Generate synthetic 14×15 surfaces **from the model itself** at MoPrP's real geometry (real `M`,
real `k_ints`, real timepoints) with known θ. Four questions, in order of expense-avoided:

1. **ANM detectability (run first).** Generate at `λ = 0` and `λ > 0`; find the smallest λ whose
   fitted value separates from zero *and* from the shuffled-geometry null under held-out
   scoring. If that threshold exceeds any plausible λ, the structural-correlation arm is dead
   before real data — the design's most speculative component eliminated at simulation cost.
2. **Sign-arm power.** Generate under signed `R`, fit signed/flip/unsigned: can held-out density
   distinguish them at N = 210? This is the resolvability pre-check the pivot litmus (§13)
   taught us to run *before* the comparison, not after.
3. **Scale recovery / confounding maps.** Recovery-error heatmaps over true component ratios for
   `(σ_exp, τ_z, τ_P, τ_T, κ)`, empirically mapping the design §10.5 confounding pairs.
4. **Delta-method validity.** MC propagation vs `A C_z Aᵀ` across realistic within-mode PF
   spreads; set the design §6.3 tolerance from these results.

Calibrate every metric's **null behaviour** on these simulations (the §15.1 lesson from the
pivot record: never interpret a raw score whose null moves).

### Phase 2.5 — finite-gating detectability (simulation only, see §6)

Runs inside the Phase 2 synthetic study, after question 1 (ANM detectability), using the same
real-geometry generator. One question: **at MoPrP's timepoint grid, N = 210, and the fitted noise
level, what gating speed γ is slow enough to be distinguishable from strict EX2?**

1. generate surfaces from the LL backend at a ladder of γ (fast → comparable to `k_int`), fit both
   K0 (strict EX2) and K2 (global γ), score held-out;
2. report the smallest γ⁻¹ whose held-out advantage separates from zero — the **detectability
   floor**;
3. run the reverse direction: generate at strict EX2, fit K2, and confirm γ is driven to the fast
   limit rather than absorbing noise (the null must not move — §15.1 pivot lesson);
4. **confounding check (mandatory):** fit K2 with the covariance components free and record whether
   γ trades against `κ`, `τ_z`, or `τ_T`. Finite gating and heteroscedastic noise both bend the
   time profile of the residual; if they are not separable in simulation they cannot be separated
   on real data, and Phase 3.5 runs with Σ frozen (see below).

If the detectability floor is slower than any physically plausible MoPrP gating rate, the kinetic
ablation is dead before real data, exactly as for the ANM arm — record that and skip Phase 3.5.

### Phase 3 — real-data model hierarchy (design §11.4)

**Mean model is strict EX2 throughout this phase — no exception.** The covariance hierarchy is
walked with the mean held fixed so that Σ architecture and exchange kinetics cannot compensate for
each other. The design §5.5 flexible interval-hazard arm (`a(log t)`) is **withdrawn as a real-data
arm** and superseded by the LL backend of Phase 3.5: an arbitrary time-dependent hazard relaxation
and finite gating are competing explanations of the same mean misspecification, and fitting both
makes neither interpretable. The hazard recursion is retained only as a Phase 1 unit-test anchor
(`a = 0` must reproduce strict EX2).

Walk the nested sequence — diagonal homoscedastic → +PF term (`R = I`) → +peptide-persistent →
+timepoint-common → heteroscedastic `D(κ)` → ANM `λ > 0` → mixture — scored by blocked-time and
peptide-region cross-fitted log predictive density with PF modes refit per fold. Discipline:

- the diagonal homoscedastic baseline is the null every component must beat;
- apply the §14.6 kill rules mechanically; removed components stay removed;
- report §14.7 spectral diagnostics for every accepted Σ against the shipped matrix's
  90.6%-in-one-direction / effective-rank-3 pathology — the "did we fix the original problem"
  readout;
- short-circuit: check uptake-space mode separation (§8.3) before any mixture calibration;
- **empirical-variance validation (new, from §2.1):** compare every accepted model's marginal
  per-cell SDs against the `moprp.weights`-implied SDs (both are on the maxD-normalised scale —
  §7.2 — so no unit conversion is involved) — correlation and calibration slope over
  the 14×15 surface. Add one hierarchy arm with `D` **fixed** to the empirical weight surface
  (zero fitted shape parameters) as a strong baseline the fitted `d(μ;κ)` must justify itself
  against. If the fitted diagonal disagrees with the empirical surface in *shape* (not just
  scale), that is a structured-residual failure under §14.5, not a tuning opportunity.
- **pre-registered outcome:** if the winner is "diagonal + peptide term" with all structure
  rejected, that is the finding — HDX peptide noise at this resolution is effectively
  unstructured — and it is reported as such, not treated as a failed stage.

### Phase 3.5 — kinetic ablation (only if Phase 2.5 passed; Σ architecture frozen first)

Entry condition: Phase 3 has selected and **frozen** an accepted Σ architecture, and Phase 2.5
showed finite gating is detectable at this geometry. The frozen Σ is what prevents a flexible
noise model from silently absorbing kinetic misspecification (or the reverse).

| arm | mean backend | free kinetic parameters | purpose |
|---|---|---|---|
| K0 | strict EX2 | none | the null; the Phase 3 accepted model unchanged |
| K1 | LL with γ pinned at the fast limit | none | **numerical control** — must reproduce K0 to tolerance; any difference is an implementation bug, not physics |
| K2 | LL, one global γ shared by all residues | 1 | the actual test: is there evidence of finite gating anywhere? |
| K3 | LL, per-domain γ | #domains | run **only** if K2 is decisively supported; domains pre-registered before fitting, as with the ANM sign-flip arm |

Rules:

- scored by the same blocked-time and peptide-region cross-fitted log predictive density as Phase 3,
  with PF modes refit per fold;
- Σ components stay at their Phase 3 values; K2 does **not** simultaneously re-fit `κ`, `τ_P`, `τ_T`;
- one free kinetic parameter added at a time, K3 gated on K2;
- pre-registered outcome: γ pinned at the fast limit with no held-out gain is the expected and
  reportable result — it is positive evidence that MoPrP centroid uptake is EX2 at this resolution,
  and it retires the EX2 caveat rather than failing a stage;
- if K2 *is* supported, the claim is bounded: it is evidence of finite gating in the centroid first
  moment, not an EX1 diagnosis. The envelope gate (Phase 0.3, limited to peptide 1 per §2.1) is
  still the only envelope evidence available, and a supported K2 makes acquiring envelopes for the
  remaining 13 peptides the top follow-up.

### Phase 4 — downstream consequence (pivot calibration §12 step 4)

Freeze the accepted target (means, weights, Cholesky, scale — design §12.2) and substitute into
the `moprp_population_pivot.py` real-uptake fit at locked coefficients, versus eye-MSE and the
shipped-Σ arm. Readouts: recovery %, decoy mass, and whether any Σ advantage now survives away
from the jitter-inversion regime (the §2.3 decisive test, resolved by construction).
Regression checks: default paths byte-identical; joint loss equals block-diagonal loss when
cross-time blocks are zeroed; frozen Σ invariant under frame-weight optimisation.

### Phase 5 — final exPfact diagnostic (after all primary and sensitivity analyses)

Only after the validated HDXrate/PDLA arm and all pre-registered sensitivity analyses are complete,
rerun exPfact using the supplied MoPrP kints as the intrinsic-rate input, where the exPfact
workflow permits that input path. Materialise the resulting protection factors as a separate,
fully hashed diagnostic artifact; do not replace the historical, HDXrate/PDLA, or compatibility
rate sources and do not use the refit to choose the earlier model specification.

Compare the refit against the held-out experimental results and the pre-registered metrics. An
improvement confined to fitted/training data, without improvement in held-out experimental
prediction, is evidence of overfitting and must be reported as such. A held-out improvement is
evidence that the supplied MoPrP kints contain a rate convention relevant to the experiment, but
does not by itself establish that the exPfact refit is the historical generator of `median.pfact`.

---

## 4. Outputs, manifests, and record-keeping

- Artifacts under `fitting/jaxENT/_moprp_sigma_noise_model/` exactly as design §16 (target
  modes NPZ, compat exports, diagnostics CSVs, `covariance_report.md`).
- Every manifest: git commit, rate-source name/path/SHA-256, structure path/SHA-256, PF start
  count and seed, vector-order string, masks, uptake-normalisation status (**the §7.1 paragraph verbatim**), ANM variant,
  numerical floor, seeds, **uptake backend (`ex2` | `ll`) and, for `ll`, the γ parameterisation
  and fitted/pinned log γ**. Any result JSON lacking a rate-source field is retroactively the
  3Ala default (kint handoff convention).
- Results are recorded as a new section of `plans/hdx_moprp_pivot_calibration.md` (it is §12
  step 3 of that record); this handoff and the design doc stay frozen.
- Rate-source regression anchor: a default-source run must leave existing artifacts
  byte-identical, verified against the §16.3 CSV SHA-256
  `3545d50d548b7dc6a42a820699ec19b05bad19c2d1efd5995b93e9c4ec336f32`.

## 5. Out of scope (inherited)

Σ-weighted Jacobian arms before Phase 3 acceptance; the pivot decision itself (`fast` remains
the default forward path, `legacy` the verification arm); free 210×210 covariances; an
envelope-level EX1 likelihood (its diagnostic gate is §3 Phase 0.3, limited per §2.1); the
third HDXer rate file. Newly scoped **in**, architecturally only: the finite-gating uptake backend
(§6) — it is built and unit-tested in Phase 1 but is not fitted to real data before Phase 3.5.
Newly scoped **out**: the design §5.5 flexible interval-hazard mean as a real-data arm (superseded,
see Phase 3); residue-specific `k_open`/`k_close` as free parameters at any phase.

---

## 6. Uptake backend interface (build in Phase 1, activate in Phase 2.5/3.5)

**Principle: architecturally integrate now; scientifically activate later.** The reason this is in
the handoff rather than left as a post-hoc analysis is twofold — it keeps the kinetic ablation a
true drop-in instead of a refactor, and it pre-registers the test so that a finite-gating model
cannot later look like a rescue invented after seeing the EX2 residuals.

### 6.1 Status of strict EX2

`d(t) = 1 − exp(−k_int t / PF)` is **not** an ad hoc approximation. It is the EX2 limit of the
three-state scheme below. The hierarchy is: exact finite-gating kinetics ⊃ strict EX2 (fast-gating
limit). Replacing EX2 with the matrix-exponential model as the *primary* mean would enlarge the
inverse problem — adding a conformational timescale the experiment may not identify — before the
simpler covariance components have been shown identifiable at all. Hence: EX2 stays primary through
Phase 3; the LL backend is an interchangeable implementation of the same interface.

### 6.2 The three-state model

Per residue *i*, with C and O the unexchanged closed/open states and D exchanged:

    C_i  ⇌ (k_open,i / k_close,i)  O_i  →(k_int,i)  D_i

Generator over the unexchanged subspace `p = (p_C, p_O)`:

    A_i = [[ −k_open,i ,          k_close,i          ],
           [  k_open,i , −(k_close,i + k_int,i)      ]]

initialised at conformational equilibrium

    p_i(0) = (1 − p_open,i , p_open,i),

propagated as `p_i(t) = exp(A_i t) p_i(0)`, giving fractional uptake

    d_i(t) = 1 − p_C,i(t) − p_O,i(t).

`A_i` is 2×2 with real distinct eigenvalues in the physical regime, so **use the closed-form
eigendecomposition, not a generic `expm`** — it is cheaper, `jax.grad`-clean, and avoids
`expm` accuracy loss at stiff `k_int/γ` ratios. Guard the near-degenerate eigenvalue case with a
series expansion (this is the numerically dangerous branch; unit-test it directly).

### 6.3 Parameterisation — one knob, PF preserved

The ablation must not re-parameterise protection factors, or K0/K2 stop being comparable. Fix

    p_open,i = 1 / PF_i = exp(−z_i)      (equivalently 1/(1+e^{z}) if PF is defined as 1 + k_c/k_o —
                                          pick one in Phase 1, assert it in the EX2-limit test)
    k_open,i = γ · p_open,i ,   k_close,i = γ · (1 − p_open,i)

so γ = k_open + k_close is the **total gating speed** and the single new parameter. PF is unchanged;
γ → ∞ recovers strict EX2 exactly. K2 fits one global log γ; K3 fits one per pre-registered domain.
Residue-specific k_open/k_close are out of scope at every phase (§5).

### 6.4 Interface to expose

Extend design §15.4 so that the mean and its Jacobian come from a swappable backend:

```python
class UptakeBackend(Protocol):
    def residue_uptake(self, log_pf, k_int, times, kinetics) -> Array:      # (R, T)
    def logpf_sensitivity(self, log_pf, k_int, times, kinetics) -> Array:   # (R, T) = ∂ν/∂z
```

- `EX2Backend` ignores `kinetics`; it is the design §5.1/§6.1 pair, analytic.
- `LLBackend` consumes `kinetics = (log_gamma,)` (scalar or per-domain).
- Everything downstream — `stack_propagation_matrix`, `build_joint_covariance`, `Z_T`'s
  `∂μ/∂log t`, the losses — consumes only these two arrays and must not import an EX2-specific
  closed form. **This is the whole point of doing it now**: keep the sensitivity path from
  hard-coding `−x e^{−x}`.
- Both `Z_T` and the PF Jacobian must be derived from the *active* backend. `∂μ/∂log t` for the LL
  backend has no tidy closed form — take it by `jax.jvp` and check against finite differences.
- Get `LLBackend`'s Jacobian by AD over the closed-form propagator rather than hand-deriving it;
  the finite-difference test is then a genuine independent check.

### 6.5 Phase 1 tests for the backend (added to §3 Phase 1 item 2)

1. **EX2-limit test:** `LLBackend` at large γ matches `predict_ex2_uptake` to tolerance, and the
   tolerance tightens monotonically as γ grows. This is the K1 control, run as a unit test.
2. **Conservation:** `p_C + p_O + d = 1` to machine precision; `d` monotone non-decreasing and in
   `[0,1]` across the real irregular timepoints and across the γ ladder.
3. **AD consistency:** `LLBackend.logpf_sensitivity` vs central finite differences, at `x ≪ 1`,
   `x ≈ 1`, `x ≫ 1`, and across γ from fast-limit to `γ ≈ k_int`.
4. **Degenerate branch:** eigenvalue near-collision handled without NaN in both value and gradient
   (check `jax.grad`, not just the forward pass — the classic failure is a forward-clean,
   backward-NaN `sqrt` in the eigenvalue split).

---

## 7. Uptake normalisation and back-exchange — resolved (Phase 0.2)

**Provenance of this section:** supplied 2026-08-13 from Moulick et al. (the MoPrP HDX-MS study)
and the Stofella thesis (exPfact input definition). Sourced from the literature, **not** verified
against the repo's raw files or re-derived here; the manifest should cite it as literature
provenance, and anyone with the papers to hand should confirm the two equations below before the
first absolute-number run.

### 7.1 What `moprp.dexp` is (manifest paragraph — copy verbatim)

> MoPrP uptake is peptide-wise maxD normalised using a completely deuterated control subjected to
> the same quench, digestion, LC and MS processing. This normalisation implicitly compensates
> peptide-dependent mean back-exchange in centroid uptake but does not reconstruct absolute
> pre-quench deuterium occupancy or explicitly model residue-specific / time-dependent
> back-exchange. Labelling experiments were conducted at approximately 95% D₂O.

Definitions. Moulick et al. report retention, `%H_retention(t) = 100·(M_D − M(t))/(M_D − M_P)`;
the fractional uptake used downstream is the complement,

    D(t) = (M(t) − M_P) / (M_D − M_P),

with `M_P` the protonated (0%) control and `M_D` the **experimental** maximally deuterated control
measured through the identical workflow. Stofella states the exPfact input identically as
`D = (m − m_0%)/(m_100% − m_0%)` with `m_100%` the experimental, not theoretical, fully-labelled
peptide.

Status per property: peptide-specific 0% control **yes**; peptide-specific maxD control **yes**;
maxD through identical LC/MS **yes**; normalised to measured maxD **yes**; major peptide-specific
back-exchange compensated **yes, implicitly**; explicit physical back-exchange model per timepoint
**no**; corrected to theoretical `N_exchangeable` **no**; raw Da **no** (fractional 0–1).

### 7.2 Hard rule: no second correction

**Do not apply any further generic back-exchange factor, D₂O-fraction factor, or
`N_exchangeable` rescaling to `moprp.dexp`.** Doing so double-corrects what maxD normalisation
already removed. Design §4.4's "must not correct the same effect twice" is therefore a concrete
assertion to write into Phase 0.2: grep the loading path and confirm no such factor exists.
An explicit back-exchange model is appropriate only for generating raw mass shifts or isotopic
envelopes — i.e. it belongs to the Phase 0.3 envelope work, never to the centroid target.

Consequently the model observable is

    D_obs[p,t] = Δm_measured[p,t] / Δm_maxD_measured[p],

and `moprp.weights` are uncertainties **on that same maxD-normalised scale** — which is the scale
the fitted `σ_exp²·D_m` diagonal lives on, so the Phase 3 empirical-variance comparison is
apples-to-apples with no unit conversion. This closes the last open question about that comparison.

### 7.3 What this fixes in the covariance model

The ceiling `D = 1` is the **experimentally processed maxD**, not "every exchangeable amide
occupied by D". The experimental maxD has itself back-exchanged, and Stofella shows the
discrepancy from the theoretical fully-deuterated envelope is large and **strongly peptide
dependent** (peptides 1, 5, 13 conspicuously so).

That is a direct physical warrant for the design §9.2 primary loading:

    Z_P[(p,j),q] = 1[p = q] · μ_pjm      (multiplicative, persistent across the peptide's curve)

τ_P is now interpretable as **residual uncertainty in each peptide's maxD reference** after the
measured control has been applied — an error in `Δm_maxD[p]` scales that peptide's entire curve,
which is exactly a multiplicative peptide-persistent term. The additive-offset arm drops to
sensitivity status, and the design §9.2 kill rule ("if maxD correction explains peptide
persistence, remove `τ_P`") should be read the other way round: maxD normalisation *creates* a
named, peptide-dependent residual, so `τ_P` enters the hierarchy with a prior expectation of being
supported. If Phase 3 rejects it, that is the surprising outcome and deserves comment.

### 7.4 The 95% D wrinkle — record, do not model

Labelling was into ~95% D₂O (1:20 dilution of protonated protein), while the maxD control started
from protein completely deuterated in D₂O. So the normalisation ceiling is an experimental maxD
reference, not the thermodynamic asymptote of a peptide exchanging indefinitely in 95% D — a naive
model of the latter would asymptote near 0.95 occupancy, not 1.

Decision: **the strict EX2 mean keeps its `ν → 1` asymptote and no 0.95 factor is introduced.**
Both numerator and denominator of `D(t)` are measured through the same 95%-D labelling and the
same processing, so the ratio is normalised to the achievable, not theoretical, ceiling. Introducing
a 0.95 factor on the mean would be exactly the double correction §7.2 forbids.

What must not happen is the data being described as "absolute fractional occupancy" — that 5% is a
real distinction and belongs in the manifest (§7.1) and in §18-style interpretation boundaries. Two
places to watch it: (i) any future comparison of these PFs against absolute PFs from a differently
normalised dataset, where the ceilings differ; (ii) the Phase 3.5 LL backend, whose `p_open` is
tied to PF — if a future arm ever predicts *raw* occupancy rather than maxD-normalised uptake, the
0.95 and the peptide-dependent maxD deficit both re-enter, and that arm needs its own observation
map rather than a reused one.

---

## 8. Phase 0.3 result — peptide-1 EX1 spot check

Resolved 2026-08-13 by `fitting/jaxENT/moprp_ex1_spot_check.py`. The committed file roles are
`pep1.1` = protonated/natural-abundance control, `pep1.5` = fully deuterated (maxD) control, and
`pep1.2`, `.3`, `.4` = 1, 60, and 1440 min. The two hypotheses were fitted only to the observed
centroid and passed through the identical operator: deuteron-count distribution, control-calibrated
binomial thinning (survival `0.498054`), convolution with the measured protonated control, and a
normalised 10-bin window.

| t (min) | observed mean | observed variance | EX2 variance | EX1 variance | EX2 SSE | EX1 SSE |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.734 | 0.780 | 0.941 | 1.466 | 0.00122 | 0.03570 |
| 60 | 1.996 | 1.361 | 1.734 | 2.893 | 0.00216 | 0.07724 |
| 1440 | 2.665 | 1.314 | 1.895 | 2.358 | 0.00589 | 0.01234 |

All three observed envelopes are unimodal. EX2 wins by SSE at every primary timepoint (about
29×, 36×, and 2.1×), and the width ordering is `observed < homogeneous EX2 < EX1`. Because a
heterogeneous independent-residue EX2 model is narrower than the homogeneous-binomial bound at a
fixed mean, this is the expected EX2 ordering. The observed moments reproduce
`hdx_effective_rate_variance_physics.md` §9.5 to three decimals. Sweeping the survival probability
by ±0.1 and `n_bins ∈ {8, 10, 12}` produces no EX1 win; at survival −0.1 the 1440-min fit saturates
both models at their fully exchanged parameter boundary and is therefore an uninformative tie.

**Verdict:** the centroid-EX2 premise survives this spot check for peptide 1. This is weak but real
evidence, not a dataset-wide validation: the other 13/14 peptides have no envelope data and remain
unscreened, and `N = 5` plus roughly 50% back-exchange gives low power (only about 1.8× EX1/EX2
variance separation near exchanged fraction 0.5, rather than the naive factor of `N`).

---

## 9. Phase 0.4 result — masks, sentinels, and the PF-conditioned residue set

Resolved 2026-08-13 by read-only inspection of `median.pfact`, the intrinsic-rate file, and the
peptide map. No new analysis code was required.

### 9.1 The 24 `-1` sentinels are two categorically different things

| group | n | residue ids | disposition |
|---|---:|---|---|
| structural non-exchangers | 4 | 1, 14, 35, 42 — `G` (protein N-terminus) and all three prolines | no backbone amide exists; excluded by construction and independently marked `-1` in the rate file |
| exchangeable, zero peptide coverage | 20 | 2, 3, 10, 26, 31, 45–59 | **excluded entirely** — they enter no peptide mean, so there is nothing for a prior to be latent about |
| covered by ≥1 peptide | 76 | — | all 76 carry a resolved PF |

The sequence has exactly three prolines (P14, P35, P42). Together with G1 they are precisely the
four residues marked `-1` in **both** `median.pfact` and `expfact_kint_pH4p4_298K_min.dat` — an
independent cross-confirmation of the residue numbering.

Note the block structure of the 20: **45–59 is one contiguous 15-residue run** (`QYSNQNNFVHDCVNI`)
with no peptide coverage, plus scattered singles at 2, 3, 10, 26, 31. Contiguous missingness is the
worst case for a structural correlation prior, which would be interpolating across it with no data
to check against. Relevant to the Phase 2 ANM arm.

### 9.2 The sentinel guard is real but narrower than assumed

`predict_ex2_uptake` (`jaxent/src/analysis/hdx_ex2.py:364`) guards with
`represented = (rates > 0) & np.isfinite(log_pf)`. Since `np.isfinite(-1.0)` is `True`, this catches
only the **4 rate-file sentinels**, not the 20 PF sentinels. Those 20 are protected solely by
`M[p,r] == 0`, verified to hold across all 14 peptide rows — but unasserted anywhere.

Why `-1` is a dangerous sentinel: it is an in-range value for the quantity it masquerades as
(ln PF = −1 means PF = 0.37, exchanging *faster* than intrinsic). It raises nothing and produces no
NaN. A `value <= 0` test is also wrong, because residue 12 legitimately carries ln PF = `0.0`
(PF = 1, unprotected). Detection must be exact equality, or better, a boolean mask carried
alongside the values. Leak behaviour at t = 1 min: PF-sentinel residues give uptake ≈ 1.0
("fully exchanged"), and residues with both sentinels give uptake = −14.15. Neither raises.

Zero-weighting is also not a general defence: at t = 1440 min a doubly-sentinel residue overflows to
`-inf`, and `0.0 × -inf = NaN`, which would propagate through an entire covariance factorisation.
Today the `rates > 0` guard catches those four first. That ordering is load-bearing and untested.

### 9.3 Decision

**PF reference is `median.pfact`. `R` = the 76 peptide-covered residues. No peptide is dropped.**

`median.pfact` resolves 77 residues, which is the 76 covered plus residue 4 — peptide 1's trimmed
N-terminal residue, carried by the `start+1..end` convention
(`build_expfact_peptide_map:249-263`). Residue 4 is excluded from `R` for exactly the same reason as
the 20: zero coverage, so it enters no peptide mean. The property that matters holds exactly:
**every covered residue has a resolved PF** (`covered ⊆ resolved`, with `resolved − covered = {4}`).

Do **not** use `_output/MoPrP_pfactors.dat` as the PF reference — see §2.7.

### 9.4 Phase 1 construction obligations left by this

1. Build `C_z`, `D_z` and `R_ANM` over the 76 **by construction**. Never rely on downstream
   multiplication by zero: `A_m = M·diag(s)` inherits the mapping's zeros, but `R_ANM` is built from
   **structure** and has no zeros at excluded residues, and `D_z` is simply meaningless there.
2. Assert that no sentinel reaches a log or rate equation, and assert the `M[p,r] == 0` invariant
   for every excluded residue, as regression guards rather than incidental properties.

### 9.5 Numbering caveat

`moprp.list`'s fourth column is offset by one from `moprp.seq`: row 1 reads `4 9 YMLGSA`, but 1-based
sequence positions 4–9 are `GYMLGS`. The code convention — residue_id = 1-based position in
`moprp.seq` (`build_expfact_peptide_map:259`) — is the correct one, independently confirmed by the
proline/N-terminus agreement in §9.1, and `load_expfact_dataset` already ignores that column
deliberately. Anyone eyeballing `moprp.list` to check residue identities will get an off-by-one.

---

## 10. Phase 0.5 result — `moprp.weights` semantics and the PF-reference bias

Resolved 2026-08-13 by read-only computation (`predict_ex2_uptake` against `median.pfact`,
`moprp.kint`, `moprp.dexp`) plus a direct read of upstream exPfact source
(`github.com/pacilab/exPfact` @ `34d1329`, 2024-02-07). No new analysis code was committed.

### 10.1 Convention: the weights file is `w = 1/σ`, but exPfact consumes it as `1/σ²`

**Upstream documentation and upstream code disagree by exactly one square.** This is the failure
mode §2.1 flagged, and it exists in exPfact itself, not in our reading of it.

- `testing/README.md:36` — "If available, use the **inverse of the standard deviation** on measured
  deuterium uptake." → the file-authoring convention is `w = 1/σ`. This matches the paper.
- `python/calculate.py:22-34` — `calculate_rms` computes
  `rms = [weights[i] * (dpred[i] - dexp[i])**2 …]; return 1/nj * sum(rms)`. A weight multiplying a
  **squared** residual is a precision, `1/σ²`. Whatever the docs say, this is what the optimiser does.

**Disposition: `moprp.weights` is `1/σ`.** The file was authored under the documented convention,
and three independent internal checks confirm the magnitude is only coherent that way:

1. **Magnitude.** On the maxD-normalised 0–1 scale, `σ = 1/w` gives min/median/max
   `0.0018 / 0.0296 / 0.1202`; `σ = 1/√w` gives `0.0424 / 0.1722 / 0.3467`. A median per-cell SD of
   17% of full dynamic range is not a credible HDX-MS centroid uncertainty; 3% is.
2. **Signal scaling.** Peptide 13 — the near-zero-uptake peptide, max dfrac `0.205` — carries the
   smallest median `1/w` of all 14 (`0.0092`, vs `0.021–0.047` elsewhere). Under `1/σ` that is
   ~4.5% of its range, in line with the rest; under `1/σ²` it is `σ = 0.096`, ~47% of the peptide's
   entire range, which is incoherent.
3. **Noise-floor bound.** The `median.pfact` residual sd is `0.096` raw and `0.0756` after removing
   per-peptide bias — *below* the `1/σ²`-implied median σ of `0.172`. A non-minimising median PF
   vector cannot fit closer than the measurement noise floor.

The upstream squaring bug is harmless for MoPrP because the weights never entered the fit (§10.2),
but it must be recorded: **any future exPfact run passing `--weights moprp.weights` would apply
`1/σ` as if it were `1/σ²`**, silently mis-weighting by a square. Do not pass that flag, or correct
the file to `w²` first. The Phase 5 exPfact refit is directly exposed to this.

Consequence for Phase 3: the empirical SD surface is `1/w`, on the maxD-normalised scale (§7.2), so
the comparison against the fitted `σ_exp²·D_m` diagonal remains unit-free as stated.

Also recorded: `corr(log w, D_obs) = −0.441` — weights fall as uptake rises, the expected
heteroscedastic direction, and the empirical shape the fitted `d(μ;κ)` must be compared against.

### 10.2 Circularity: RESOLVED — the weights did **not** enter `median.pfact`

Three lines of upstream evidence, all consistent:

1. **`--weights` is optional and defaults off.** `python/exPfact.py:160,238-241`:
   `if opts.weights: config['weights'] = read_dexp(...)  else: config['weights'] = None`.
2. **The documented command that produced the shipped artifacts does not pass it.**
   `validation/README.md` ("Multiple minimizations"):
   `exPfact.py --temp 298 --pH 4.4 --dexp moprp.dexp --ass moprp.list --harm 1e-8 --rand 10000
   --seq moprp.seq --out out --rep 5000`, followed by
   `descriptive_statistics.py --res out --top 50` → `average.pfact`, `median.pfact`, `minmax.pfact`,
   `all.sp`. No `--weights` at any stage; the same is true of the single-minimisation and
   cross-validation commands.
3. **Our residual test predicted this.** `w`-scaling failed to flatten the residual (§10.2 prior
   version), which is the expected signature of an unweighted fit.

**`moprp.weights` is therefore an independent object with respect to `median.pfact`.** The Phase 3
empirical-variance comparison (§3, Phase 3) is a genuine external validation target, not a
circular one, and the anchoring caveat in §2.1 does not apply. This is the strongest available
outcome for that item.

**Corollary — `median.pfact` is a median of the top 50 of 5,000 harmonic-penalised
(`--harm 1e-8`) random-restart fits.** That fully explains the non-minimiser property found below:
it optimises no objective, and even the constituent fits minimise unweighted least squares *plus*
a smoothness penalty. `write.py:63` confirms the per-peptide `.diff` costs are always written
unweighted.

Still unknown: **how `moprp.weights` itself was generated.** No generator exists in the upstream
repo (`write_combined_replicates`, `write.py:69-92`, pools replicate SDs of `.Dpred` predictions,
not of experimental uptake, and does not emit a `.weights` file). Presumed replicate SDs from
Moulick et al.; record as unverified provenance.

### 10.3 New finding — the PF reference carries peptide-level systematic bias

The ±0.1 per-peptide bias above is **larger than the ~0.03 noise scale it is meant to sit inside**.
This was not anticipated by §9, which established `median.pfact` as the self-consistent PF reference
on coverage grounds alone and never checked its predictive residual.

Risk: that bias propagates into `C_z` and lands directly in the peptide-persistent term `τ_P` —
which §7.3 already primes to expect support. `τ_P` could then be "supported" for a reason having
nothing to do with maxD residual, and §7.3's pre-registered expectation would be confirmed
spuriously.

Two decisions required before Phase 1 construction:

1. whether the PF reference should be a **refit** (`fit_ex2_solution_set`, ≥50 starts — already
   required by §1.2/§2.4 for mode structure) with the shipped median retained only as a comparator;
2. whether the per-peptide bias vector must be reported alongside any fitted `τ_P`, as a mandatory
   confound readout, so the two cannot be conflated.

Until resolved, this supersedes nothing in §9 (the residue set and sentinel findings stand) but
adds a constraint on how the reference is *used*.
