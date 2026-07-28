# Handoff: shared-residue discrepancy and HDX error construction (TeaA/ISO scope)

**Status:** open investigation. Opened 2026-07-24; substantially revised after scientific review.
**Relationship to prior work:** sibling of `plans/hdx_effective_rate_variance_physics.md` (the
D-only handoff), whose verdicts stand unmodified. Takes over one unresolved contradiction (§1).

> **To execute this, start at §13 (work breakdown), not §1.** §§1–12 are the scientific
> specification; §13 is the ordered chunk list with dependencies, gates, and the compute budget.

> **Out of scope here — handled separately.** The rate-space averaging / pivot convention question
> (average-first in log-PF vs average-after in rate space) is being run as its own full fitting test
> across averaging types. §2.3 states the convention this investigation assumes; do not re-open it
> here. See `[[hdx-rate-space-pivot-handoff-exists]]`.

---

## 1. The scientific question

### 1.1 The contradiction that motivates it

- **(A)** `Sigma_MSE` — inverse-covariance-weighted L2 — improved recovery substantially in the
  real-data workflows.
- **(B)** A MoPrP study (`_moprp_val_score_correlation_20260724/`) found **no** observed-only score
  discriminates recovery: six candidates all failed (best fixed-ESS Spearman ρ ≈ −0.11,
  sign-flipping); the weaker decoy-avoidance bar failed in the *wrong direction* (−0.09 to −0.32).

(B) is not evidence against (A) — they tested different objects (§6.1: the redundancy candidate was
a literal no-op) and different roles (training loss vs post-hoc score).

### 1.2 Primary question (narrow, falsifiable)

> **When peptide observations are averages of shared residue-level signals, does accounting for the
> resulting shared residue-level discrepancy improve recovery of a known, reweighting-reachable
> conformational population, relative to independent-peptide MSE, at matched information content
> and matched regularization?**

"Matched information content" is load-bearing and is enforced in §7.3 and §8.1 — it is what most of
the review feedback turned on.

### 1.3 Secondary questions (kept deliberately separate)

- **S1.** Does timepoint *sensitivity* weighting improve recovery? (A heuristic, **not** an error
  model — §5.)
- **S2.** Can any historical Σ construction help as a **fixed heuristic geometry**, even though none
  of them is an observation-error covariance? (§9.)

These must not be merged into the primary question: they are different mechanisms with different
justifications, and conflating them is what made the original framing unanswerable.

---

## 2. Forward model

Symbols: residue `i`, frame `f`, timepoint `t`, peptide `p`.

### 2.1 Contacts (BV, hard count)
`h_{i,f}` = protein heavy atoms within `6.5 Å` of amide **N**; `o_{i,f}` = oxygen acceptors within
`2.4 Å` of amide **H**; both excluding the `(−2,+2)` residue window.
(`jaxent/src/models/func/contacts.py::calc_BV_contacts_universe`, `switch=False`.)

### 2.2 Log protection factor and rate
```
z_{i,f} = Bc·h_{i,f} + Bh·o_{i,f}        k_{i,f} = k_int,i · exp(−z_{i,f})
```
`k_int,i` is frame-independent chemistry and carries no conformational information.

### 2.3 EX2 uptake and the assumed pivot
```
z̄_i = Σ_f w_f z_{i,f}     k̄_i = k_int,i·exp(−z̄_i)     ū_i(t) = 1 − exp(−k̄_i t)
```
Average-first in log-PF. **Assumed, not tested here** (see the out-of-scope note above).

### 2.4 Peptide mapping
```
U_p(t) = Σ_i M_{p,i} u_i(t)
```
`M ∈ R^{P×N}` row-normalized (`1/N_active`); `peptide_trim=1`; prolines dropped.
`create_sparse_map` (`jaxent/src/data/splitting/sparse_map.py`).

---

## 3. The error model (the missing statement)

Everything in this investigation depends on an explicit residual model, which the first draft
lacked. There are **two different shared-discrepancy mechanisms**, and they must not be collapsed
into one:

```
y_t = M u_t(w*, b) + M a_t + ε_t
```

- `ε_t` — **independent peptide measurement noise**, potentially heteroscedastic by peptide and
  timepoint.
- `a_t` — **time-local residue discrepancy** on the uptake scale. Overlapping peptides share it
  within a timepoint, but `a_t` and `a_s` are independent for `t != s`. This is a spatial-only
  random-effect control, not the primary kinetic model.
- `b` — **persistent kinetic discrepancy** at residue resolution: BV/contact-model
  misspecification, intrinsic-rate error, or another residue-level kinetic deviation. The same `b`
  affects every timepoint through the nonlinear uptake curve.

For small persistent discrepancy in `η = log k = log k_int - z`, linearize about the fitted mean:

```
J_t = diag(∂u_i(t)/∂η_i) = diag(k_i t exp(−k_i t))
```

and stack the residual vector over all peptide/timepoint pairs. Its block covariance is

```
Σ_ts =
    1[t=s] · [τ_a² M D_a,t Mᵀ + diag(σ_pt²)]
    + τ_b² M J_t C_b J_s Mᵀ
```

The `M J_t C_b J_s Mᵀ` term is load-bearing: persistent kinetic misspecification produces both
peptide covariance and **cross-time covariance**. The earlier `M D Mᵀ` model is only the
time-diagonal `a_t` special case.

The generator therefore has four nested conditions:

| Condition | Injects | Purpose |
|---|---|---|
| `measurement_only` | `ε` | negative control; overlap is information only |
| `spatial_local` | `ε + a` | shared peptide support within each timepoint, no temporal persistence |
| `kinetic_persistent` | `ε + b` | primary BV/contact/rate-misspecification model |
| `full_shared` | `ε + a + b` | tests whether time-local and persistent components can be separated |

Generate `b` **nonlinearly** in the latent coordinate, e.g.

```
z_i* = z_i + b_i
u_i(t) = 1 − exp[−k_int,i exp(−z_i*) t]
```

and compare inference using the exact generated curves against the Jacobian covariance
approximation. This distinguishes failure of the covariance hypothesis from failure of the
linearization.

**Five consequences that reshape the whole plan:**

1. **`M Mᵀ` is justified only in a narrow special case** — time-local, approximately independent,
   homoscedastic residue discrepancy on the uptake scale (`D_a,t ≈ I`), or persistent log-rate
   discrepancy with nearly constant `J_t`. Otherwise bare `M Mᵀ` is a design-balancing heuristic,
   not an observation likelihood.
2. **If `τ_a = τ_b = 0`, the correct reference is the measurement-noise model.** Under independent
   homoscedastic Gaussian noise this reduces to plain MSE. Peptide overlap is then **information,
   not correlated error**: two overlapping peptides are two genuine measurements, and their
   overlap legitimately improves residue-level localization.
3. **The physical covariance is the complete sum**, including measurement noise:
   `Σ = Σ_measurement + Σ_spatial-local + Σ_persistent`. A numerical ridge is not a substitute for
   the `σ_pt²` nugget or for the ratio of variance components.
4. **A synthetic oracle answers a conditional question only.** Supplying `τ_a`, `τ_b`, `C_b`, and
   `σ_pt` demonstrates what correctly specified GLS can do; it does not show that real data support
   a nonzero shared component. A train-only variance-component estimation arm is required to answer
   that empirical question.
5. **This is distinct from molecular co-exchange covariance.** Centroid uptake remains blind to the
   within-molecule residue-residue exchange covariance discussed in
   `plans/research/secondorder_HDX_physics.md`. Here `Σ` is a repeated-dataset/model-discrepancy
   covariance imposed or estimated for the observation residual.

### 3.1 The physically defensible baseline

With conformational residue/peptide variance excluded (this scope), the defensible data term is
**plain centroid MSE, interpreted as an equal-variance Gaussian likelihood**:

```
−log p(y_ptr | μ_pt, σ_pt) = (y_ptr − μ_pt)²/(2σ_pt²) + log σ_pt + boundary
```

With no replicate uncertainties available, adopt `σ_pt = σ` as the **equal-variance reference
assumption**; it is not implied by the absence of uncertainty estimates. Under that assumption the
data term reduces exactly to MSE.
(Replicate-level truncated-Gaussian with per-peptide/timepoint σ is the established richer form —
Saltzberg et al., PMC5693600 — and is the target once replicate data exist.)

**Neither Fisher-weighted MSE nor `Sigma_MSE` is a physical error likelihood.** Both are heuristic
geometries and must be labelled as such throughout.

### 3.2 Likelihood normalization — a real defect for this experiment

A likelihood is a **sum** over observations. `hdx_uptake_sigma_MSE_loss` (and the `eye` variant)
**averages**: `total_loss / (T · n_fragments)` (`jaxent/src/opt/losses.py:1586`, `:1645`).

Because Block 1 deliberately varies the peptide count across constructions, averaging means adding
independent measurements does **not** increase the strength of the evidence relative to the KL
regularizer — so `tile10` (P≈29) and `overlap10_s3` (P≈97) would be regularized at effectively
different strengths purely from normalization. **Fix required before any cross-construction
comparison:** use summed NLL with a fixed prior strength, or state and verify the equivalent
count-rescaling convention.

This statement applies to genuinely independent added observations. Under correlated shared
discrepancy, observation count is not information count and duplicating an overlapping peptide need
not double the evidence. The complete joint likelihood and its log determinant determine the
incremental information. The normalization regression test in §11 therefore uses an independent
block-diagonal duplication, while §7.3 matches constructions using Fisher information rather than
raw `P`.

### 3.3 Bounded centroid likelihoods

The available observable is centroid fractional uptake `y_pt ∈ [0,1]`; no raw isotope envelopes or
ion-count data are available. Count-level and envelope-level likelihoods are therefore **out of
scope**. In particular, multiplying `dfrac` by the number of active amides does not create observed
binomial counts: the centroid is a continuous ensemble summary, and residue exchange probabilities
are heterogeneous.

Two bounded-geometry alternatives belong in Block 1:

1. **Mean-precision beta likelihood (`bounded_beta`)**
   ```
   y_pt ~ Beta(μ_pt φ, (1−μ_pt) φ)       μ_pt = [M u_t(w)]_p
   ```
   This preserves the existing forward-model mean and imposes the bounded variance geometry
   `Var(y_pt | μ_pt) = μ_pt(1−μ_pt)/(φ+1)`. Use a global precision `φ` estimated from training data
   only; replace it with replicate-calibrated `φ_pt` if suitable uncertainties later become
   available. Include the beta normalizing terms: this is a likelihood, not a weighted squared
   error.
2. **Complementary-log-log normal likelihood (`cloglog_normal`)**
   ```
   g(y_pt) ~ Normal(g(μ_pt), σ²)         g(x) = log(−log(1−x))
   ```
   This is a link-geometry sensitivity analysis. It is mechanistically aligned with EX2 at residue
   level because `g(u_i(t)) = log k_i + log t`, but after peptide averaging `g(Mu) ≠ M g(u)`;
   consequently it is not promoted over the beta mean model. Include the transformation Jacobian
   when reporting predictive density or comparing held-out NLL across likelihood families.

Neither continuous likelihood has finite density at exact reported `0` or `1`. Treat endpoints as
censored observations using acquisition/reporting limits fixed before fitting; do not silently clip
or add pseudocounts. This is not a minor edge case: direct inspection of the bundled TeaA target
finds **300/1470 exact endpoints** (296 ones, 4 zeros). Its artificial reporting/rounding precision
must be recorded and used to define the censoring limits. Record the limits and number of censored
observations in every cell. Both likelihoods remain independent-peptide observation models: they
test bounded support and mean-dependent variance/link geometry, **not** shared discrepancy or
redundancy.

### 3.4 Inverse-variance (`1/SE²`) weighting — scope, correctness, and what it cannot reach

Raised in PI review: *"typically you would just weight the MSE loss by `1/SE_j²`."* The general form
is right and is exactly §3.1's Gaussian NLL with per-observation `σ_pt` restored:

```
L = Σ_j (y_j − ŷ_j)² / SE_j²
```

i.e. diagonal GLS. Three findings on whether it applies under this synthetic setup.

**(a) It is exactly correct when — and only when — the noise is injected.** If the generator draws
`y = M u_t(w*) + ε`, `ε ~ N(0, σ_j²)`, then `1/σ_j²` is the *true* weight rather than an estimate.
Synthetic data is the one regime where the correct weighting is available for free, and this is the
strongest form of the argument for adopting it. Two riders:

- If the injected noise is **homoscedastic**, `1/σ²` factors out and the loss reduces to plain MSE up
  to a constant — no behavioural change, only a rescaling against the KL/MaxEnt term. This is §3.1
  restated: absent per-point uncertainties, MSE *is* the inverse-variance loss. The weighting only
  does work if `σ` is deliberately made to vary across peptides and/or timepoints.
- The rescaling interacts with the §3.2 normalization defect. Adopting summed `Σ_j r_j²/σ_j²` while
  the existing losses average by `(T·n_fragments)` changes the data/regularizer balance, so C0's
  summed-likelihood fix is a **prerequisite** for comparing weighted against unweighted runs.

**(b) Two tempting sources of `SE_j` are not error.** Under synthetic data the injected `σ` may not
be the most *available* quantity, and substituting an available one silently changes the object:

1. **Spread of predicted peptide values across frames.** This is conformational variance — the
   signal being fitted. Down-weighting high-spread peptides down-weights exactly the observations
   that discriminate Open from Closed. It is the `Sigma_unweighted` object (§9.2), a model-geometry
   prior; using it as `1/SE²` mislabels it as an observation likelihood.
2. **Timepoint sensitivity `s_t`.** Ruled out in §5: OLS already yields the Fisher information, and
   re-weighting by `s_t²` asserts an unsupported `Var(y_t) ∝ 1/s_t²`. It is the S1 heuristic, not an
   SE, and must not be used as a surrogate for a noise model.

**(c) The diagonal cannot reach the primary question.** `1/SE_pt²` is the measurement term of §3
done properly. The spatial-local term `M D_a,t Mᵀ` and persistent term
`M J_t C_b J_s Mᵀ` are off-diagonal in peptide space, and the latter is also off-diagonal in time.
No per-observation SE, however well calibrated, captures either. The PI's correction and the
shared-discrepancy question are therefore orthogonal, not competing.

**Consequence for the generator (load-bearing).** The four nested conditions in §3 are required,
with both homoscedastic and heteroscedastic measurement-noise subcases. Under
`measurement_only`, the correctly specified diagonal likelihood is optimal in expected predictive
NLL; misspecified Σ arms need not be numerically identical to it, but they should show no systematic
advantage over repeated noise realizations. Under `spatial_local`, only within-time peptide
covariance is present. Under `kinetic_persistent`, the cross-time blocks are the identifying
feature. Under `full_shared`, the fitted method must not force both components into one variance
term.

For `b`, stage the physical source structures rather than immediately running all seven non-empty
combinations of contact, BV-coefficient, and intrinsic-rate error:

1. iid log-rate/log-PF offsets;
2. residue-heteroscedastic offsets;
3. regional/contact-correlated offsets.

Only split contact-count, BV-coefficient, and intrinsic-rate sources into a further factorial if
these three structures behave differently in the litmus. Also include non-shared nuisance controls
that can masquerade as overlap: timepoint/run-level common shifts, peptide-specific offsets
persistent across time, and peptide-length/intensity-dependent measurement variance.

`σ`, `τ_a`, `τ_b`, and generator covariance specs are recorded in the run manifest. Oracle arms may
consume them; estimated arms may not. These quantities are not population labels, so oracle use does
not breach Guardrail 1, but oracle results must be labelled as conditional method validation rather
than evidence that real data support the component.

---

## 4. `Sigma_MSE` as implemented

`jaxent/src/opt/losses.py::hdx_uptake_sigma_MSE_loss` (1541–1598):

```
L = (1/(T·P)) · Σ_t  ½ · r_tᵀ W r_t
```

`W` is loaded from the **`"Sigma_inv"`** key, Frobenius-normalized, then trace-normalized per split
(`jaxent/src/data/loader.py:215-217`). It is **frozen** — never recomputed during fitting, so it
carries no ESS/diversity confound. `W = I` recovers plain MSE (`hdx_uptake_eye_MSE_loss`).

### 4.1 The existing loss is a generic quadratic form — and is insufficient for the physical model

Verified by inspection: `hdx_uptake_sigma_MSE_loss` and `hdx_uptake_eye_MSE_loss` have **identical
bodies**; the latter merely passes `jnp.eye(...)`. The existing peptide-only heuristic arms can
therefore be expressed as constructions feeding the `"Sigma_inv"` key. That does **not** extend to
the persistent-kinetic likelihood in §3, which requires a new stacked peptide×time covariance path.

Three structural limits constrain what the shared body can express:

1. **One `W` per split, reused across all timepoints.** The compute loop iterates timepoints but
   holds `cov_matrix` fixed (`losses.py:1547`). Per-timepoint SE (`σ_pt`) and cross-time covariance
   **cannot be represented**. C0 must add at least a `(P,T)` diagonal path and a stacked
   `(P·T,P·T)` covariance path.
2. **Trace normalization erases absolute scale.** `_trace_normalise` forces `trace(W) = n`
   (`loader.py:215-217`), so relative heteroscedasticity survives but the overall precision and the
   ratios `τ_a/σ`, `τ_b/σ` do not. A trace-normalized diagonal arm is not the PI's likelihood.
3. **The log determinant is absent.** For one fixed `W`, `log|Σ|` is constant with respect to frame
   weights and does not change that arm's optimizer. It is nevertheless mandatory when estimating
   variance components, comparing covariance models, or reporting predictive likelihood.

### 4.2 Two explicit covariance pathways

Do not ask one loss body to serve two incompatible interpretations:

1. **`gaussian_joint_nll` — physical path.** Consume unnormalized diagonal `(P,T)` or full stacked
   `(P·T,P·T)` covariance and compute the quadratic, `log|Σ|`, and all normalization terms. Use this
   for measurement-only, spatial-local, persistent-kinetic, full-shared, oracle, and
   estimated-component arms. The MaxEnt prior strength is fixed or selected without population
   labels; covariance scale is not trace-normalized away.
2. **`quadratic_geometry` — heuristic path.** Retain trace-normalized `W` for
   `Sigma_unweighted`, `Sigma_observed`, and other deliberately non-likelihood geometries. Report
   recovery and fixed quadratic scores, never cross-family predictive likelihood.

The diagonal implementation remains low-risk, but per-timepoint heteroscedasticity is not postponed:
it is part of C0 because the registered heteroscedastic generator otherwise cannot be represented.

---

## 5. Timepoint sensitivity weighting — a heuristic, not an error model

For `η = log k`, sensitivity is `s_t = ∂u/∂η = k t e^{−k t}`, maximal at `u ≈ 0.632`.

**Why this is not the statistically correct weighting.** Under Gaussian noise, ordinary least
squares *already* yields the Fisher information:

```
I(η) = Σ_t s_t² / σ_t²
```

Multiplying each squared residual again by `s_t²` changes the curvature to `s_t⁴/σ_t²`. That
implicitly asserts a noise model `Var(y_t) ∝ 1/s_t²`, which nothing in the data supports.

**Consequences for the plan:**

- Relabel throughout as a **sensitivity-weighted heuristic**, never "the correct error weighting".
- **The construction-independence prediction is withdrawn.** Actual information depends on `M`, the
  parameter being inferred, and the noise covariance — so there is no justification for predicting
  a uniform benefit across peptide constructions.
- `fisher_timepoint_weights` (`jaxent/src/analysis/hdx_target_variance.py:340`) requires **residue
  mean rates**, not merely peptide observations — so it is not a purely observation-side weight.
- Its proper role is **experimental design**: diagnosing identifiability, choosing labelling times,
  allocating replicates — not reweighting already-observed residuals.
- The small D-only gain previously observed is evidence for a useful heuristic *under model
  discrepancy*, not for a correct likelihood.

**Distinguish from measurement-noise weighting.** A GLS/WLS weight `1/Var(y_obs)` is driven by
ion-counting statistics, replicate spread, and back-exchange correction. It correlates with `s_t`
but is a different physical quantity. No committed per-point noise model exists for TeaA/ISO — that
is a real, data-gated gap, and the sensitivity weight must not be used as a surrogate for it.

---

## 6. Verified codebase state (all confirmed by direct inspection)

### 6.1 `redund_mse` was a no-op
`max |mse − redund_mse| = 0.0` exactly across all 630 rows of
`_moprp_val_score_correlation_20260724/val_score_cells.csv`. Conclusion (B) never tested redundancy.

### 6.2 Two different Σ constructions

| Path | Source | Covariance axis | Weight-dependent? |
|---|---|---|---|
| Real (`compute_sigma_real.py:160-164`) | observed `dfrac` `(P,T)` | across **timepoints** | no |
| Synthetic (`compute_sigma_synthetic.py:313-323`) | predicted uptake → mean over `t` → `(P,F)` | across **frames** | yes |

### 6.3 **Defect — the ISO Σ leaks ground truth**
`optimise_ISO_TRI_BI_splits_Sigma.py:223-225` loads `ISO_BI_Sigma_weighted.npz`; `Sigma_weighted` is
built from frame weights at `target_ratios = {"open":0.4,"closed":0.6}`
(`compute_sigma_synthetic.py:229-234`) — the known answer. Leak-free `Sigma_unweighted.npz` exists
but is unused.

### 6.4 **Defect — hardcoded ensemble**
The same line uses **ISO_BI**'s matrix even when fitting ISO_TRI.

### 6.5 **Blocker — shipped ISO peptides have zero redundancy**
All 294 segments are length 1 (`1 2`, `2 3`, …). `M` is effectively diagonal; **peptide overlap
cannot be tested on ISO as shipped.** Generating real peptides is a hard prerequisite (§8.1).

### 6.6 **Defect — precision matrices are subset incorrectly**
`create_covariance_mat` (`sparse_map.py:33-40`) returns `covariance_matrix[ix_(indices,indices)]` —
a plain submatrix — and the fitting scripts pass `Sigma_inv`. So the code computes

```
(Σ⁻¹)_SS      instead of      (Σ_SS)⁻¹
```

These are different objects: the former is a block of the joint precision, while the latter is the
**marginal** precision appropriate to a standalone subset likelihood. A genuine conditional
likelihood additionally requires the train/validation cross-block and the conditional residual
mean; using `(Σ⁻¹)_SS` alone is not conditional scoring.

**Required fix:** subset Σ first, then regularize → invert, and trace-normalize only on the heuristic
path. Retain the joint covariance cross-block for §7.4 conditional validation. Additionally, for
`Sigma_observed`, building the training geometry from all observations lets validation targets
influence training — direct leakage. A validation geometry estimated from those same validation
outcomes is also not a proper predictive score, so `Sigma_observed` remains descriptive only.

### 6.7 **Blocker — the verification anchor is impossible as first written**
The target has **294** segment rows spanning residue IDs 1–310; the model has **293** residue
features. At least one target residue has no feature. `create_sparse_map` can therefore emit an
empty row, and `normalize_sparse_map_rows` **raises** `ValueError("Cannot normalize a sparse mapping
with an empty fragment row")` (`sparse_map.py:250`).

So these two requirements **cannot both hold**:
- every `M` row sums to one;
- all 294 shipped targets are exactly reproduced.

An explicit **boundary policy** must be chosen and recorded before anything else (§7.1).

### 6.8 Conditioning and rank
`np.cov` of `(P,T)` has rank ≤ `min(P, T−1)`. ISO: `T=5` → **rank ≤ 4**. MoPrP: `P=14,T=15`,
nominally full rank but cond ≈ 9.1e5, `|Σ⁻¹|max ≈ 6.5e5`.

Independently corroborated: `plans/hdx_heteroscedastic_nll_investigation.md` reports the real-data
curve constructions have **effective rank 1.04–1.52** (ISO 1.01–1.64), and that tripling the
timepoint count did **not** enrich the curve covariance. `Sigma_observed` is essentially a rank-1
object.

### 6.9 Data substrate
Residue-level ground truth, 294 rows × 5 timepoints {0.167, 1, 10, 60, 120} min.
`features_iso_bi` 293×**344** frames; `features_iso_tri` 293×**890**. Clusters by RMSD ≤ 1.0 Å
(`0=Open, 1=Closed, −1=unassigned`).

### 6.10 **Corrected provenance** (supersedes the first draft's §5.8)
The first draft stated the target was "Persson–Halle, switch-function". **That is not supported by
the bundled generation path.** Verified:

- `create_mixed_target_data.py` computes `avelnpi = Σ_f (Bc·contacts + Bh·hbonds)` then
  `1 − exp(−k_int/exp(avelnpi)·t)` — i.e. **average-first in log-PF**, the same convention JAX-ENT
  uses. It is *not* average-after.
- It uses the **same BV/Radou log-PF equation with `Bc=0.35, Bh=2.0`**.
- The bundled instructions specify the **Radou** method (`calc_hdx/README.md:13`), and Radou's
  documented default contact mode is a **hard cutoff, not switching** (`Methods.py:27`).
- Peptide values are `np.nanmean` over residues — consistent with row-normalized `M`.
- The original contact files are absent, so whether the authors overrode the contact default
  **remains unverified**.

**Consequence:** forward-model mismatch is far smaller than first assumed. The BV-self-consistent
arm is therefore **not** a mandatory experimental block; it is replaced by a cheap preflight (§7.2).

### 6.11 What "reweighting-reachable" does and does not guarantee
The 40:60 frame population is reachable. But reachable *weights* do not imply reachable
*observations*: if any residual misspecification exists, the model prediction at the correct weights
need not be the closest achievable prediction to the target. This is why §7.2 is mandatory.

---

## 7. Phase 0 — mandatory preflight (no fitting until all four pass)

The review's central procedural finding: these issues affect the interpretation of *every* fit and
**cannot be repaired post hoc**. Phase 0 runs first, in full.

### 7.1 Resolve the 293/294 boundary policy
Decide and record one of: (a) drop targets lacking features (P = 293-consistent subset);
(b) extend features to cover the missing residue(s); (c) allow sub-unit row sums with documented
semantics. State which rows are dropped and why. **The verification anchor must then be restated to
match the chosen policy** (§11).

### 7.2 Population-path preflight
For each candidate loss, profile it along a controlled open-fraction path `α ∈ [0,1]` (weights
interpolating Open:Closed) and check that its minimum lies near **α = 0.4**.

This is the cheap replacement for the mandatory BV-self-consistent arm. If a loss's optimum is not
near 0.4, then any later "recovery improvement" from that loss may merely be compensating for
residual misspecification rather than doing inference. **A loss that fails this preflight is not
interpretable in Block 1** and must be reported as such.

### 7.3 Diagnose every `M` before use
For each peptide construction and seed, compute and persist:
peptide count and total measurement budget; active-residue coverage; number of **uniquely** covered
residues; **rank and effective rank** of `M`; **singular spectrum**; and a baseline recoverability
estimate.

Reporting these quantities is necessary but does not constitute matched information. For the
known-population TeaA question, also calculate the independent-noise Fisher information in the
open-fraction direction:

```
I_α = (∂μ/∂α)ᵀ Σ_measurement⁻¹ (∂μ/∂α)
```

and the corresponding local curvature/condition number for the frame-weight parameterization.
Match layouts by `I_α` where possible through peptide subsampling or replicate allocation. Where it
is impossible, use the paired difference-in-differences in §8.1 and do not attribute raw recovery
differences to redundancy.

### 7.4 Validate without shared-residual leakage

Under the model in §3, random overlapping peptide splits have nonzero train/validation covariance:

```
Σ = [[Σ_TT, Σ_TV],
     [Σ_VT, Σ_VV]]
```

The statement that overlap inflation applies equally to every loss is false: it depends on the
covariance model and layout. Register three valid evaluation modes:

1. **Conditional same-realization score:** evaluate `p(y_V | y_T)` using
   `μ_V|T = μ_V + Σ_VT Σ_TT⁻¹(y_T−μ_T)` and
   `Σ_V|T = Σ_VV − Σ_VT Σ_TT⁻¹Σ_TV`.
2. **Independent-realization score:** generate validation with a new draw of measurement and
   discrepancy components; score the marginal predictive density.
3. **Residue/digest-disjoint score:** use sequence-cluster or synthetic digest-level splits as a
   transfer negative control.

The first two are primary for the synthetic study. A random overlapping split scored as though
`Σ_TV=0` is invalid. `Sigma_observed` built from validation outcomes is likewise not a proper
held-out likelihood; it remains a descriptive historical geometry only.

---

## 8. Block 1 — error-function ladder, leakage-free (no off-diagonal Σ)

### 8.1 Peptide constructions — and why matching matters

| Construction | Length | Position | Redundancy | Role |
|---|---|---|---|---|
| `res1` | 1 | stride 1 | none | shipped baseline / regression anchor |
| `tile10` | ~10 | stride 10 | none (tiling) | length-matched zero-overlap control |
| `overlap10_s3` | ~10 | stride 3 | high | redundancy arm |
| `overlap15_s3` | ~15 | stride 3 | very high | redundancy stress |
| `random_L5-20` | random 5–20 | random start | irregular | realistic proteolytic case |

**Critical correction from review.** These do **not** vary length and redundancy independently as
first claimed. `tile10` and `overlap10_s3` match *length* but differ in peptide count, coverage,
row-space rank, and measurement budget. Random layouts add gaps and different identifiability.
**Recovery differences therefore cannot be attributed to redundancy alone** unless the §7.3
Fisher information is matched. Reporting the variables without matching them is not sufficient.

Use two remedies:

1. construct matched-information subsets/replicate allocations using `I_α`; and
2. report the paired interaction
   ```
   Δ_redundancy =
       (full/estimated covariance − correct diagonal)_overlap
       − (full/estimated covariance − correct diagonal)_tile
   ```
   within the same noise realization and regularization-selection rule.

Where neither remedy is possible, report the layout descriptively and make no redundancy-causal
claim.

**Published-generator anchor.** Before the novel layouts, reproduce the original TeaA/HDXer
fragment-size and error controls: contiguous fragment sizes 5/10/15/20/50, reduced coverage from the
10-residue panel, independent Gaussian noise at the published scales, and regional forward-model
coefficient error. These are regression anchors, not additional discovery axes in the final sweep.

**Random arm requirements (revised):**
- **≈20+ layout seeds, not 3.** Three is far too few to support the proposed correlation between
  Σ⁻¹ margin and coverage variance.
- Match mean coverage to `overlap10_s3`; report realized coverage per seed.
- Persist per-seed coverage/length/gap distributions; report per seed, never averaged into one number.
- Allow uncovered residues — gaps are realistic and change identifiability.
- Use paired layout/noise seeds across covariance arms so method differences are not inflated by
  different random realizations.

### 8.2 The ladder

Primary and secondary are now clearly separated by justification:

1. **`mse` — equal-variance reference.** Gaussian centroid likelihood (§3.1). This remains the
   reference rather than merely an engineering control.
2. **`sigma_diag_noise` — inverse-variance reference (§3.4).** `W_pt = 1/σ_pt²` with `σ_pt` taken
   from the injected-noise spec recorded in the manifest, including `(P,T)` variation. Under a
   correctly specified heteroscedastic generator it should improve **expected held-out NLL** over
   repeated noise realizations; under homoscedastic measurement-only noise it equals `mse` up to a
   common scale. It need not improve population recovery in every finite realization.
3. **`bounded_beta` — primary bounded-support alternative.** Mean-precision beta centroid NLL
   (§3.3), with the existing prediction `μ=M u`, training-only precision estimation, and censored
   handling of exact endpoints. This is the primary test of whether respecting fractional support
   and its mean-dependent variance improves inference.
4. **`cloglog_normal` — link-geometry sensitivity.** Proper transformed-normal centroid NLL (§3.3),
   including its Jacobian for held-out scoring and the same endpoint-censoring policy. It tests the
   EX2-aligned complementary-log-log geometry without asserting that peptide averages are residue
   log-rates.
5. **Robust observation sensitivities.** Common-scale Student-*t* centroid NLL guards against
   integration failures/outliers. Add a small ReX-inspired length-dependent Laplace pilot because
   the layouts deliberately vary peptide length; promote it only if a preflight shows that length
   dependence is calibrated in fractional-uptake coordinates. Gaussian remains primary.
6. **`mcMSE` — shape arm, tested early (not deferred).** Already registered
   (`losses.py:1918`, `hdx_uptake_mean_centred_eye_MSE_loss`). Rationale for promoting it: BV
   mean-model error is the established dominant confound elsewhere, and a loss insensitive to
   overall offset/amplitude miscalibration is a direct hedge against exactly that failure mode.
   *Test it, don't assume it.*
7. **`sensitivity_mse` — labelled heuristic ablation.** Retained for S1 only, with §5's caveats
   attached and **no** construction-independence prediction.
8. **Replicate-calibrated pointwise dispersion** — deferred, data-gated (requires replicate
   uncertainties); this may supply Gaussian `σ_pt` or beta `φ_pt`.

Excluded from Block 1: count/envelope likelihoods (no raw isotope data), target residue/peptide
variance, and all **off-diagonal** Σ constructions (that is Block 2).

### 8.3 Evaluation
- **Primary — as a training loss:** fit per (ensemble × construction × layout seed × noise seed ×
  loss × split), select on **held-out predictive log-likelihood and calibration**, with
  regularization chosen **without population labels**. Use §7.4 conditional or
  independent-realization validation; never treat a correlated random peptide split as marginally
  independent.
- For `bounded_beta` and `cloglog_normal`, report censoring-aware NLL, calibration, fitted
  training-only dispersion, and endpoint counts. Retain all normalization and Jacobian terms needed
  for proper density comparison.
- **Secondary — as a held-out score:** reuse fits to measure score↔recovery discrimination,
  mirroring study (B).
- **Recovery (40:60) is revealed only as a synthetic oracle diagnostic**, never as a selection input.
- Correct-noise arms are expected to win in mean/median paired predictive NLL with uncertainty
  intervals, not necessarily in every realization or for every recovery metric.
- Report paired effects over **both** noise and layout replicates, with confidence intervals and a
  preregistered minimum scientifically relevant recovery gain. Do not erase the cell structure by
  naive pooling; a hierarchical or paired summary is allowed and necessary.

---

## 9. Block 2 — Σ arms, each with its own mechanism and prediction

The review's key structural point: **the arms are not variants of one hypothesis.** A generic
"Σ⁻¹ should win under overlap" prediction is invalid; each arm gets its own.

### 9.0 Physical joint-covariance arms — primary

These use `gaussian_joint_nll`, the complete stacked covariance, and no trace normalization:

1. **`Sigma_measurement_oracle`** — `diag(σ_pt²)`.
2. **`Sigma_spatial_oracle`** —
   `1[t=s]·[τ_a² M D_a,t Mᵀ + diag(σ_pt²)]`.
3. **`Sigma_kinetic_oracle`** —
   `M J_t(τ_b² C_b)J_s Mᵀ + 1[t=s]·diag(σ_pt²)`.
4. **`Sigma_full_oracle`** — the complete sum in §3.
5. **`Sigma_components_estimated`** — estimate nonnegative `τ_a`, `τ_b`, and measurement scale
   from training data only, using fixed registered structures for `D_a,t` and `C_b`. Compare the
   nested `τ_a=0` and `τ_b=0` boundaries using conditional/independent validation or a parametric
   bootstrap; do not use population recovery for component selection.
6. **`Sigma_jacobian_misspecified`** — apply the linearized covariance to data generated by the
   exact nonlinear latent perturbation. This measures approximation error.

Sweep the variance ratios from zero through weak, intermediate, and dominant shared discrepancy.
The oracle arms answer whether correctly specified covariance can improve recovery. Only the
estimated arm answers whether the data reveal the component without generator knowledge.

### 9.1 `Sigma_structural` (`M Mᵀ`-derived) — clean support-geometry approximation
Pure shared-support geometry, no data. It is a useful approximation/control for two reasons:
- **Rank comes from residue count, not timepoint count** (`rank ≤ min(P,N)`, `N=294`), so unlike
  `Sigma_observed` it does not collapse when `P > T−1` — and most real HDX maps have `P ≫ T`.
- It is the only arm with no data-noise confound to hide behind.

It is **not** the complete physical covariance: it omits the measurement nugget, time dependence,
cross-time blocks, and nonidentity residue variance. Use `MMᵀ + λI` only as a registered
misspecification, with `λ` interpreted as a physical variance ratio when used in the physical path
and as a ridge only when used in the heuristic path.

**Prediction:** after matched-information control, any systematic advantage over the correct
diagonal should be near zero under `measurement_only`, confined to within-time effects under
`spatial_local`, and generally weaker than the correctly specified joint covariance under
`kinetic_persistent`/`full_shared`. On `random_L5-20`, test whether the paired margin scales with
coverage variance across ≈20+ seeds. A flat paired interaction falsifies the support-redundancy
story; individual finite-realization wins do not establish it.
**Required decomposition:** compare `I` vs a **diagonal** structural control vs **full `M Mᵀ`**, so
that off-diagonal decorrelation is isolated from mere diagonal reweighting.

### 9.2 `Sigma_unweighted` (predicted peptide covariance across frames, uniform prior)
**Conformational/model covariance, not error covariance.** Frozen, population-free, no leakage.
**Prediction is different in kind:** it may help as a geometry prior, and a gain on `res1` would
**not** contradict anything about redundancy — it can be non-diagonal even for single-residue
observations. Relates by analogy to the validated population-free covariance-shape prior
(`[[hdx-covariance-prior-intent]]`), but the mechanism differs (that regularized frame weights; this
reweights residuals), so it is suggestive, not transferred evidence.

### 9.3 `Sigma_observed` — historical curve-geometry control **only**
`np.cov` of observed dfrac across timepoints: covariance of a **deterministic kinetic signal**, not
observation noise. With no replicates it cannot be measurement covariance, and its **effective rank
is 1.04–1.52** (§6.8). Ledoit–Wolf shrinkage can stabilize inversion but **cannot turn five
non-identically-distributed timepoints into replicate noise samples** — it fills missing directions
with the shrinkage target, a no-op for redundancy in exactly those directions.

Additionally, MoPrP's `Σ⁻¹` up-weights directions where peptides covary *least* — which is a known
GLS pathology in ill-conditioned small-sample covariances, **not** the redundancy discount it is
sometimes assumed to be. **Demoted:** run as a labelled historical control, never presented as a
proposed error covariance, and always with its eigenvalue spectrum and condition number attached.

### 9.4 `Sigma_weighted` — leakage positive control
Not a candidate mechanism. Exists solely to quantify how much of the historical ISO gain was
ground-truth leakage (§6.3): subtract the leakage-free arms' margins from its margin.

### 9.5 Construction discipline (applies to every arm)
Build **Σ**, subset it per split, then regularize → invert, and normalize only if the registered
pathway calls for heuristic trace normalization (§6.6). Never subset an
already-inverted matrix. Never build a training Σ from observations that include the validation
split. For physical arms, do **not** trace-normalize, and retain the train/validation cross-blocks
needed for §7.4 conditional scoring. For independent-realization scoring, use the marginal
validation covariance. For heuristic arms, trace normalization remains allowed but must be recorded.

---

### 9.6 `sigma_diag_noise` — actual measurement-error precision
`W_pt = 1/σ_pt²`, `σ_pt` from the injected-noise spec (§3.4). It appears in the Block 1 ladder and
is the measurement term inside every physical joint-covariance arm in §9.0.

**Predictions, by generator condition (§3):**
- Heteroscedastic `measurement_only`: improves expected held-out NLL over unweighted MSE. Failure
  after adequate paired noise replicates is stop-the-line.
- Homoscedastic `measurement_only`: equals MSE up to the registered common scale.
- `spatial_local`: the correctly specified within-time covariance should outperform the diagonal
  only when `τ_a/σ` is large enough to matter.
- `kinetic_persistent`: the joint cross-time covariance should outperform both the diagonal and the
  time-diagonal structural approximation when `τ_b/σ` is detectable.
- `full_shared`: the estimated two-component model must be compared with both one-component nested
  models; a single shared term winning does not show that the components were separated.

These are expectations over paired replicates, not per-realization absolutes, and the primary metric
is proper predictive NLL. Recovery is the blinded synthetic oracle diagnostic.

**Two properties that make it structurally useful beyond its own result:**

1. **Immune to the §6.6 subsetting defect.** For diagonal `W`, `(Σ⁻¹)_SS = (Σ_SS)⁻¹` exactly —
   subsetting and inversion commute. With all other loss scaling held fixed, this arm must give
   bit-identical results before and after the isolated subsetting fix, while at least one
   non-diagonal arm must change. This does not imply bit identity across the other C0 likelihood
   changes.
2. **Distinct from §9.1's diagonal control.** That control is `diag(M Mᵀ)` (coverage-derived); this
   is noise-derived. Keep both: comparing them separates "any sensible diagonal reweighting helps"
   from "the *correct* diagonal helps", which the three-way `I` / diagonal / full decomposition in
   §9.1 cannot do on its own.

## 10. Guardrails

1. **No ground-truth leakage** into any loss, weighting matrix, or selection step. 40:60 populations
   and cluster labels are post-hoc oracle diagnostics only — sole exception is the labelled
   `Sigma_weighted` control.
2. **Regularization selected without population labels**; compare methods at matched ESS or
   validation-selected endpoints.
3. **Oracle versus estimated covariance is explicit.** Oracle components may use generator specs;
   `Sigma_components_estimated` and every selection step may use training observations only.
4. **Physical and heuristic paths never share likelihood language.** Physical Σ is unnormalized and
   includes `log|Σ|`; heuristic `W` may be trace-normalized and is scored only as geometry.
5. **Validation respects `Σ_TV`.** Use conditional, independent-realization, or residue/digest-disjoint
   scoring (§7.4); never score a correlated random peptide split as marginally independent.
6. **Summed likelihood with fixed prior convention** whenever peptide count varies (§3.2); raw
   observation count is not substituted for Fisher information under correlation.
7. Per-cell storage (ensemble × construction × layout seed × noise seed × generator condition ×
   split), plus paired/hierarchical summaries with uncertainty intervals; never naive pooling.
8. Preserve-and-supersede: new dated dirs; defects §6.3/§6.4/§6.6 documented here and fixed **only in
   new code paths**, leaving historical artifacts reproducible.

---

## 11. Verification

- **Boundary policy (§7.1) is stated first**, and the reproduction anchor is expressed in its terms —
  the original "all 294 rows reproduced *and* all rows sum to 1" is impossible (§6.7).
- **Peptide generator:** `M` row-sums = 1 for all retained rows; coverage matches segs; `res1`
  reproduces the shipped dfrac for retained rows to ≤ 1e-10.
- **Random arm:** seeds recorded and reproducible; ≈20+ seeds; realized coverage matched to
  `overlap10_s3` within a stated tolerance; per-seed distributions persisted.
- **Published anchor:** original contiguous 5/10/15/20/50-residue panels, reduced-coverage panel,
  published independent-noise scales, and regional forward-model-error control reproduce the
  expected qualitative HDXer behavior before novel layouts are interpreted.
- **Information matching:** persist `I_α`, frame-weight curvature, and the exact matched-information
  subset/replicate-allocation decision per layout. Raw `P`, rank, or coverage alone cannot label a
  comparison as matched.
- **Normalization:** verify that duplicating an independent block doubles the summed data term
  relative to a fixed prior. Do not require this for a correlated duplicate; verify its information
  through the joint likelihood instead.
- **Precision subsetting:** assert `(Σ_SS)⁻¹` is used, with a regression test that it differs from
  `(Σ⁻¹)_SS` on a non-diagonal example.
- **Every Σ:** symmetric, PSD after physical nugget/ridge, reported eigenvalue spectrum, condition
  number, effective rank; `Σ = I` recovers `mse` exactly.
- **Joint covariance:** exact block construction is tested against Monte Carlo residual covariance
  for `spatial_local`, `kinetic_persistent`, and `full_shared`; cross-time blocks are zero only in
  the registered time-local model. Record the Jacobian-approximation error against nonlinear latent
  perturbations.
- **Physical NLL:** quadratic + `log|Σ|` + normalization constants reproduce a trusted multivariate
  normal implementation; trace normalization is rejected on this path.
- **Conditional validation:** Schur-complement mean/covariance agree with direct conditional-normal
  calculations. Independent-realization scoring has zero train/validation residual covariance.
- **Diagonal arm as regression anchor (§9.6):** `sigma_diag_noise` results are bit-identical before
   and after an **isolated** C0 subsetting fix with loss scaling held constant (diagonal `W` commutes
   with subsetting), while at least one non-diagonal arm changes. Do not demand bit identity across
   the separate summed-loss or physical-NLL changes.
- **Inverse-variance sanity:** under homoscedastic injected `σ`, `sigma_diag_noise` equals `mse` to
  numerical tolerance; under heteroscedastic measurement-only noise it improves paired expected
  predictive NLL with a preregistered uncertainty criterion.
- **Variance-component sanity:** at `τ_a=τ_b=0`, the estimated arm selects/collapses to the
  measurement model at the registered false-positive rate; power curves are reported as each
  component increases. Oracle and estimated results are never conflated.
- **Generator manifest:** every run records `σ_pt`, `τ_a`, `τ_b`, `D_a,t`, `C_b`, physical source
  structure, nuisance controls, nonlinear-versus-linearized generation, and one of
  `measurement_only` / `spatial_local` / `kinetic_persistent` / `full_shared`.
- **Endpoint policy:** reporting precision, censoring thresholds, and the bundled 300/1470 endpoint
  count are reproduced exactly.
- **Trace-normalization decision (§4.2) recorded per arm** — mandatory `false` for physical NLL,
  explicitly registered for heuristic geometry.
- **Preflight:** each loss's population-path profile persisted with its argmin.
- `pytest jaxent/tests/unit/` green; ruff clean.

---

## 12. Open questions

1. **Are `τ_a` or `τ_b` nonzero and distinguishable?** If both shared components are negligible,
   the measurement model is correct and peptide overlap should be treated purely as information.
   Oracle injection does not answer this; the train-only estimated arm and its power/false-positive
   curves do.
2. **Contact implementation of the target** remains unverified (§6.10) — the original contact files
   are absent. Bounded claim only: same log-PF equation, same average-first convention, contact mode
   unproven.
3. **Redundancy vs conditioning** for `Sigma_observed`: up-weighting low-covariance directions is not
   the same claim as discounting shared support. §9.1's diagonal-vs-full decomposition is designed to
   separate these.
4. **`P ≫ T` regime.** The observed-covariance construction is rank-deficient for most real HDX
   datasets. If Block 2 shows Σ matters, this is the practical blocker for generalizing it — and the
   reason `Sigma_structural` is the arm that scales.
5. **Can shared kinetic discrepancy be separated from global experimental nuisances?** Run/timepoint
   common shifts, peptide offsets persistent across time, back-exchange/normalization error, and
   peptide-length/intensity-dependent noise can imitate parts of the proposed covariance. The
   synthetic nuisance controls in §3 establish specificity; real separation remains replicate- and
   metadata-gated.
6. **Ultimate observable.** The centroid is a summary of the isotope envelope; equal centroids can
   correspond to distinguishable envelope shapes (Kan et al. 2013). Envelope-level likelihood is the
   principled endpoint but is data-gated and that track is separately retired.
7. **Component coordinate.** Persistent discrepancy is primary in `log k`/`log PF`, while the
   time-local control is written on uptake scale. If exact nonlinear generation and the Jacobian
   approximation diverge materially, covariance inference must remain nonlinear rather than moving
   the component to uptake scale for convenience.

---

## 13. Work breakdown and execution order

§§1–12 specify *what* is being tested. This section specifies *how it is executed*: eight chunks
with explicit dependencies, exit gates, and a compute budget. **Do not start a sweep before C3.**

### 13.1 Chunk list

| # | Chunk | Depends on | Exit gate |
|---|---|---|---|
| **C0** | **Likelihood/covariance plumbing + prerequisite fixes** | — | diagonal `(P,T)` and stacked `(P·T,P·T)` NLL tests green; `Σ=I` reproduces summed Gaussian loss; subsetting regression passes |
| **C1** | **Peptide generator, published anchors, `M`/Fisher diagnostics, and four nested error generators** | C0 (boundary policy) | `res1` reproduces shipped dfrac; published controls pass; Monte Carlo covariance and nuisance-specificity gates pass |
| **C2** | **Population-path preflight** (§7.2) | C1 | per-loss argmin recorded against α=0.4 |
| **C3** | **Sizing/power decision** — fix noise seeds, layout seeds, variance-ratio levels, split and regularization axes against measured cost | C1 | written budget, timing probe, and simulation-based power target |
| **C4** | **Block 1 ladder** (§8.2) | C0–C3 | held-out predictive LL + calibration per cell |
| **C5** | **Physical joint-covariance arms** (§9.0, §9.6), staged litmus then fits | C4 | oracle/estimated distinction, component power, nonlinear/Jacobian comparison, and four-condition contrasts reported |
| **C6** | **Heuristic/history Σ arms and leakage quantification** (§9.1–9.4) | C5 | mechanism-specific predictions reported; leak margin separated from every leak-free arm |
| **C7** | **Validation/split transfer contrast** (§7.4) | C5 | conditional, independent-realization, and residue/digest-disjoint conclusions compared |

C0 contains both engineering and registered statistical choices; C1 contains generator science and
engineering. C2 and C3 both depend only on C1 and can run in parallel.

### 13.2 C0 — prerequisite fixes, collected

Scattered across §§3–7 in the specification; collected here because they must all land first:

1. **Summed likelihood** (§3.2) — replace the `/(T·n_fragments)` average, or rescale the KL term by
   the explicitly fixed prior convention. Test independent, not correlated, duplication.
2. **Precision subsetting** (§6.6) — subset Σ, *then* regularize → invert; normalize only on the
   heuristic path. Build train and validation marginal covariances from their subsets and retain
   cross-blocks for conditional scoring.
3. **Per-ensemble Σ** (§6.4) — stop hardcoding `ISO_BI`.
4. **Boundary policy** (§6.7, §7.1) — resolve 293 vs 294 and restate the reproduction anchor in its
   terms.
5. **Per-timepoint diagonal covariance** — thread `(P,T)` `σ_pt` through `Dataset` and the loss.
6. **Stacked joint covariance** — support `(P·T,P·T)` Σ with stable factorization, quadratic,
   `log|Σ|`, and normalization constants.
7. **Separate physical and heuristic paths** (§4.2) — physical covariance is never trace-normalized;
   historical geometry remains reproducible without inheriting likelihood language.
8. **Conditional validation** (§7.4) — retain and score the train/validation covariance cross-block.

### 13.3 C2 gate — what happens when a loss fails the preflight

Profile the **expected/noiseless** loss and the distribution of noisy argmins separately, with
`near` defined before inspection. A loss whose expected population-path optimum is not near α=0.4
is excluded from Block 1 and reported as excluded, with its profile persisted. It is not silently
carried forward and is not grounds for halting other arms. If `mse` itself fails, that is
stop-the-line.

The one-dimensional α path is necessary but not sufficient. Also run an oracle unconstrained/simplex
fit from multiple starts and record target-point gradient, Hessian/Fisher curvature, off-path minima,
and population degeneracy. A loss can pass the α path while preferring a decoy or within-state
reweighting direction elsewhere.

### 13.4 C3 — sizing decision (first pass)

The earlier 7,200-fit estimate is withdrawn. It omitted generator condition, noise realization,
variance-ratio level, nuisance/misspecification case, and the estimated-component arm, and therefore
understated the experiment by at least an order determined only after the power pilot.

Use staged promotion:

| Stage | Scope | Promotion gate |
|---|---|---|
| C1a algebra/MC | one ensemble, `res1` + one overlapping layout, no optimization | generated and analytic joint covariances agree; nuisance controls do not masquerade as the registered component |
| C1b published anchor | original fragment/coverage/noise/model-error controls | expected qualitative behavior reproduced |
| C3 power/timing pilot | `tile10` + `overlap10_s3`, four generator conditions, small variance-ratio grid, paired noise seeds | false-positive rate controlled; minimum relevant NLL/recovery effect detectable; wall time measured |
| C4 loss pilot | MSE, correct diagonal, beta, cloglog, robust arms on two layouts | only preflight-passing likelihoods promoted |
| C5 physical pilot | measurement, spatial, kinetic, full oracle + estimated components | correct nesting and cross-time discrimination demonstrated |
| C5 expansion | two ensembles, seven primary layouts, promoted ratios/arms only | main paired effects |
| C5 coverage correlation | ≈20 random layouts, only `I`/diagonal/full support geometries and promoted physical comparator | seed count preserved; arm count minimized |
| C6/C7 controls | historical/leakage and transfer validation | run only after physical conclusions are stable |

The C3 budget must explicitly multiply:

```
ensemble × layout × layout_seed × noise_seed × generator_condition
× variance_ratio × nuisance_case × arm × split/evaluation_mode × regularization
```

Do not use three random peptide splits as a substitute for noise replicates. Prefer one registered
conditional split plus independent-realization replicates for power; retain residue/digest-disjoint
splits as the C7 transfer contrast. Determine the number of noise seeds by simulation-based power
for paired predictive-NLL and recovery effects, not by convenience.

If cost must be cut, reduce nuisance/source structures and heuristic arms before cutting the four
nested generator conditions, variance-ratio zero point, paired noise seeds, or the ≈20 layouts
needed for the coverage-variance correlation.

### 13.5 Reporting discipline across chunks
Every chunk writes to its own dated artifact directory with a manifest recording inputs, the C3
budget actually used, and which cells were excluded by the C2 gate. Persist per-cell results and
paired identifiers throughout; report paired/hierarchical summaries with uncertainty intervals,
never naive pooled means.
