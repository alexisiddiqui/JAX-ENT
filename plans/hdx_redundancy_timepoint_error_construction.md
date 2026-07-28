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
lacked. Write the observation as

```
y_t  =  M u_t(w*)  +  M δ_t  +  ε_t
```

- `δ_t` — **shared residue-level discrepancy** (model error at residue resolution: BV
  misspecification, contact-model error, residue-level kinetic deviation). It is *shared* because
  overlapping peptides average the same `δ_i`.
- `ε_t` — **independent peptide measurement noise** (per peptide/timepoint, from the measurement
  process itself).

Then the residual covariance is

```
Cov(r_t)  =  τ_r² · M D Mᵀ  +  τ_m² · I
```

**Three consequences that reshape the whole plan:**

1. **`M Mᵀ` is justified only in a special case** — approximately independent, homoscedastic
   residue-level discrepancy (`D ≈ I`). Without stating this model, "redundancy is an error source"
   is ambiguous, and `M Mᵀ` is a *design-balancing heuristic*, not an observation likelihood.
2. **If `τ_r = 0`, the correct loss is plain MSE.** Peptide overlap is then **information, not
   correlated error**: two overlapping peptides are two genuine measurements, and their overlap
   legitimately improves residue-level localization (the basis of global HDX fitting —
   Fajer et al. 2012; Skinner et al. 2019). Whitening it away would *discard* information.
3. **The primary question is therefore a question about `τ_r`** — does the data support a nonzero
   shared-discrepancy component, and does modelling it help recovery?

### 3.1 The physically defensible baseline

With conformational residue/peptide variance excluded (this scope), the defensible data term is
**plain centroid MSE, interpreted as an equal-variance Gaussian likelihood**:

```
−log p(y_ptr | μ_pt, σ_pt) = (y_ptr − μ_pt)²/(2σ_pt²) + log σ_pt + boundary
```

With no replicate uncertainties available, `σ_pt = σ` and the data term reduces exactly to MSE.
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
comparison:** use summed NLL, or rescale the KL/MaxEnt term by the observation count. This is a
prerequisite, not an optimization.

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
or add pseudocounts. Record the limits and number of censored observations in every cell. Both
likelihoods remain independent-peptide observation models: they test bounded support and
mean-dependent variance/link geometry, **not** shared discrepancy or redundancy.

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

**(c) The diagonal cannot reach the primary question.** `1/SE_j²` is the `τ_m²·I` term of §3 done
properly. The shared residue-level discrepancy `τ_r²·M D Mᵀ` is **off-diagonal by construction** —
overlapping peptides average the same `δ_i`. No per-observation SE, however well calibrated,
captures it. The PI's correction and the redundancy question are therefore orthogonal, not competing.

**Consequence for the generator (load-bearing).** If the generator injects only independent
peptide-level `ε`, then `τ_r = 0` *by construction*, and §3's consequence #2 applies: MSE (or
SE-weighted MSE) is the correct loss, peptide overlap is pure information, and every Σ⁻¹ arm should
show **no** benefit. That makes such a run a genuine **negative control** for Block 2 rather than a
competitor to it — but it also means Block 2 is unanswerable unless a second generator variant
injects a residue-level `δ_i` as well. Both variants are therefore required:

| Generator variant | Injects | Expected result |
|---|---|---|
| `noise_indep` | peptide-level `ε` only (`τ_r = 0`) | diagonal weighting optimal; all Σ arms collapse to the diagonal result |
| `noise_shared` | residue-level `δ_i` **and** peptide `ε` (`τ_r > 0`) | the only condition in which §9.1's `M Mᵀ` arm can win |
| `noise_hetero` | `ε` with `σ` varying across peptides/timepoints | diagonal weighting must beat unweighted MSE, or something upstream is broken |

`σ` and `τ_r` specs are generator-side knowledge and are recorded in the run manifest; they are **not**
fitted, and they are not population labels, so this does not breach Guardrail 1.

---

## 4. `Sigma_MSE` as implemented

`jaxent/src/opt/losses.py::hdx_uptake_sigma_MSE_loss` (1541–1598):

```
L = (1/(T·P)) · Σ_t  ½ · r_tᵀ W r_t
```

`W` is loaded from the **`"Sigma_inv"`** key, Frobenius-normalized, then trace-normalized per split
(`jaxent/src/data/loader.py:215-217`). It is **frozen** — never recomputed during fitting, so it
carries no ESS/diversity confound. `W = I` recovers plain MSE (`hdx_uptake_eye_MSE_loss`).

### 4.1 The loss is a generic quadratic form — arms differ only in `W`

Verified by inspection: `hdx_uptake_sigma_MSE_loss` and `hdx_uptake_eye_MSE_loss` have **identical
bodies**; the latter merely passes `jnp.eye(...)`. So every Σ arm in §9, including the diagonal
inverse-variance arm of §3.4, is a *construction* feeding the `"Sigma_inv"` key — **no new loss
function is required**, only new Σ builders. Implementation risk for the diagonal arm is therefore
near zero.

Two structural limits of that shared body constrain what any arm can express:

1. **One `W` per split, reused across all timepoints.** The compute loop iterates timepoints but
   holds `cov_matrix` fixed (`losses.py:1547`). Per-timepoint SE (`σ_pt`) — which is what real HDX
   replicate spread actually gives, since early and late timepoints differ substantially — **cannot
   be represented**. Supporting it requires threading a `(P,T)` weight array through `Dataset`, or
   accepting peptide-only SE and stating that restriction. This is the same gap §8.2's deferred
   rung 7 will hit; see Open Question 6.
2. **Trace normalization erases absolute scale.** `_trace_normalise` forces `trace(W) = n`
   (`loader.py:215-217`), so relative heteroscedasticity across peptides survives but the overall
   precision level `1/σ²` does not. A trace-normalized diagonal arm is thus *not* the PI's likelihood
   — it is that likelihood with its scale absorbed into the effective regularization strength. Given
   §3.2 this is the safer default for cross-cell comparability; either document it or exempt the
   diagonal arm from normalization deliberately, but do not leave it implicit.

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

These are different objects: the former is a **conditional** precision (conditioning on the excluded
peptides), the latter the **marginal** precision appropriate to a subset likelihood.

**Required fix:** subset Σ first, *then* regularize → invert → normalize, independently for train
and validation. Additionally, for `Sigma_observed`, building the full Σ from **all** observations
lets validation targets influence training geometry — a direct leakage-guardrail violation. Train
and validation precisions must be constructed from their own covariance subsets.

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

## 7. Phase 0 — mandatory preflight (no fitting until all three pass)

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
estimate. These are the matching variables for §8.1 — without them, recovery differences cannot be
attributed to redundancy.

---

## 8. Block 1 — error-function ladder, leakage-free (no Σ⁻¹)

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
variables are matched or explicitly reported and controlled for. Report them alongside every result;
where matching is impossible, say so rather than attributing the difference.

**Random arm requirements (revised):**
- **≈20+ layout seeds, not 3.** Three is far too few to support the proposed correlation between
  Σ⁻¹ margin and coverage variance.
- Match mean coverage to `overlap10_s3`; report realized coverage per seed.
- Persist per-seed coverage/length/gap distributions; report per seed, never averaged into one number.
- Allow uncovered residues — gaps are realistic and change identifiability.

### 8.2 The ladder

Primary and secondary are now clearly separated by justification:

1. **`mse` — equal-variance reference.** Gaussian centroid likelihood (§3.1). This remains the
   reference rather than merely an engineering control.
2. **`sigma_diag_noise` — inverse-variance reference (§3.4).** `W = diag(1/σ_j²)` with `σ_j` taken
   from the injected-noise spec recorded in the manifest. Under `noise_hetero` it **must** beat
   unweighted `mse`; under `noise_indep` with homoscedastic `σ` it must equal it to numerical
   tolerance. Listed here rather than only in Block 2 because it is a diagonal observation-error
   weight, not a redundancy geometry — but it is built and consumed through the Block 2 Σ pathway
   (§4.1, §9.6).
3. **`bounded_beta` — primary bounded-support alternative.** Mean-precision beta centroid NLL
   (§3.3), with the existing prediction `μ=M u`, training-only precision estimation, and censored
   handling of exact endpoints. This is the primary test of whether respecting fractional support
   and its mean-dependent variance improves inference.
4. **`cloglog_normal` — link-geometry sensitivity.** Proper transformed-normal centroid NLL (§3.3),
   including its Jacobian for held-out scoring and the same endpoint-censoring policy. It tests the
   EX2-aligned complementary-log-log geometry without asserting that peptide averages are residue
   log-rates.
5. **`student_t` — robustness sensitivity.** Common-scale Student-*t* centroid NLL, guarding against
   integration failures/outliers. Gaussian remains primary.
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
variance, and all Σ⁻¹ (that is Block 2).

### 8.3 Evaluation
- **Primary — as a training loss:** fit per (ensemble × construction × seed × loss × split), select
  on **held-out predictive log-likelihood and calibration**, with regularization chosen **without
  population labels** (match at equal ESS or use validation-selected endpoints).
- For `bounded_beta` and `cloglog_normal`, report censoring-aware NLL, calibration, fitted
  training-only dispersion, and endpoint counts. Retain all normalization and Jacobian terms needed
  for proper density comparison.
- **Secondary — as a held-out score:** reuse fits to measure score↔recovery discrimination,
  mirroring study (B).
- **Recovery (40:60) is revealed only as a synthetic oracle diagnostic**, never as a selection input.
- Report over **both** noise and layout replicates. Per cell; never pooled.

---

## 9. Block 2 — Σ arms, each with its own mechanism and prediction

The review's key structural point: **the arms are not variants of one hypothesis.** A generic
"Σ⁻¹ should win under overlap" prediction is invalid; each arm gets its own.

### 9.1 `Sigma_structural` (`M Mᵀ`-derived) — the only clean redundancy arm
Pure shared-support geometry, no data. Physically privileged for two reasons:
- **Rank comes from residue count, not timepoint count** (`rank ≤ min(P,N)`, `N=294`), so unlike
  `Sigma_observed` it does not collapse when `P > T−1` — and most real HDX maps have `P ≫ T`.
- It is the only arm with no data-noise confound to hide behind.

**Prediction:** advantage over `mse` ≈ 0 on `res1`/`tile10`, largest on `overlap*`, and on
`random_L5-20` scaling with the variance of per-residue coverage across seeds (needs ≈20+ seeds).
Flat margin across constructions **falsifies** the redundancy story cleanly.
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
Build **Σ**, subset it per split, then regularize → invert → normalize (§6.6). Never subset an
already-inverted matrix. Never build Σ from observations that include the validation split.

---

### 9.6 `sigma_diag_noise` — the only arm that is an actual error precision
`W = diag(1/σ_j²)`, `σ_j` from the injected-noise spec (§3.4). Every other arm in §9 is a covariance
of a deterministic kinetic or conformational signal; this is the one construction that is a genuine
observation-error precision, which is precisely why it belongs alongside them rather than instead of
them. It appears in the Block 1 ladder (§8.2 rung 2) and is reused unchanged as a Block 2 arm.

**Predictions, by generator variant (§3.4):**
- `noise_hetero`: beats `mse`. Failure here invalidates every downstream arm.
- `noise_indep`, homoscedastic: equals `mse` to numerical tolerance.
- `noise_shared` (`τ_r > 0`): beaten by `Sigma_structural`, since the diagonal cannot represent
  `τ_r²·M D Mᵀ`. This is the sharpest available discriminator for the primary question — it isolates
  off-diagonal shared discrepancy against a *correctly specified* diagonal, not against an
  unweighted straw man.

**Two properties that make it structurally useful beyond its own result:**

1. **Immune to the §6.6 subsetting defect.** For diagonal `W`, `(Σ⁻¹)_SS = (Σ_SS)⁻¹` exactly —
   subsetting and inversion commute. So this arm must give bit-identical results before and after
   the C0 fix, while every non-diagonal arm must change. That is a strictly stronger regression test
   than §11's `Σ = I` check, which is invariant for trivial reasons.
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
3. **Every Σ frozen** before fitting; train/validation precisions built from their own subsets.
4. **Summed likelihood (or count-rescaled KL)** whenever peptide count varies across cells (§3.2).
5. Per-cell reporting (ensemble × construction × seed × split); never pooled.
6. Preserve-and-supersede: new dated dirs; defects §6.3/§6.4/§6.6 documented here and fixed **only in
   new code paths**, leaving historical artifacts reproducible.

---

## 11. Verification

- **Boundary policy (§7.1) is stated first**, and the reproduction anchor is expressed in its terms —
  the original "all 294 rows reproduced *and* all rows sum to 1" is impossible (§6.7).
- **Peptide generator:** `M` row-sums = 1 for all retained rows; coverage matches segs; `res1`
  reproduces the shipped dfrac for retained rows to ≤ 1e-10.
- **Random arm:** seeds recorded and reproducible; ≈20+ seeds; realized coverage matched to
  `overlap10_s3` within a stated tolerance; per-seed distributions persisted.
- **Normalization:** verify that doubling the peptide count doubles the data term relative to the
  regularizer (the §3.2 fix actually took effect).
- **Precision subsetting:** assert `(Σ_SS)⁻¹` is used, with a regression test that it differs from
  `(Σ⁻¹)_SS` on a non-diagonal example.
- **Every Σ:** symmetric, PSD after ridge, reported eigenvalue spectrum, condition number, effective
  rank; `Σ = I` recovers `mse` exactly.
- **Diagonal arm as regression anchor (§9.6):** `sigma_diag_noise` results are bit-identical before
  and after the C0 subsetting fix (diagonal `W` commutes with subsetting), while at least one
  non-diagonal arm changes. Stronger than the `Σ = I` check, which is trivially invariant.
- **Inverse-variance sanity:** under homoscedastic injected `σ`, `sigma_diag_noise` equals `mse` to
  numerical tolerance; under `noise_hetero` it beats `mse`. Failure of the latter is stop-the-line.
- **Generator manifest:** every run records the injected `σ` spec and `τ_r`, and which of
  `noise_indep` / `noise_hetero` / `noise_shared` produced the data.
- **Trace-normalization decision (§4.1) recorded per arm** — normalized or exempt — since it
  determines whether the diagonal arm is the likelihood or the likelihood-up-to-scale.
- **Preflight:** each loss's population-path profile persisted with its argmin.
- `pytest jaxent/tests/unit/` green; ruff clean.

---

## 12. Open questions

1. **Is `τ_r` nonzero at all?** If shared residue-level discrepancy is negligible, plain MSE is
   correct and peptide overlap should be treated purely as information (§3). This is the question,
   not an assumption.
2. **Contact implementation of the target** remains unverified (§6.10) — the original contact files
   are absent. Bounded claim only: same log-PF equation, same average-first convention, contact mode
   unproven.
3. **Redundancy vs conditioning** for `Sigma_observed`: up-weighting low-covariance directions is not
   the same claim as discounting shared support. §9.1's diagonal-vs-full decomposition is designed to
   separate these.
4. **`P ≫ T` regime.** The observed-covariance construction is rank-deficient for most real HDX
   datasets. If Block 2 shows Σ matters, this is the practical blocker for generalizing it — and the
   reason `Sigma_structural` is the arm that scales.
5. **Per-timepoint SE is not representable (§4.1).** The loss body holds one `W` per split across all
   timepoints, so `σ_pt` cannot be expressed — only peptide-level `σ_p`. Real replicate spread varies
   substantially between early and late timepoints, so this restriction will bind as soon as
   replicate data arrive (§8.2 rung 8). Resolving it means threading a `(P,T)` weight array through
   `Dataset`. Out of scope here; must be stated as a limitation on any inverse-variance result.
6. **Ultimate observable.** The centroid is a summary of the isotope envelope; equal centroids can
   correspond to distinguishable envelope shapes (Kan et al. 2013). Envelope-level likelihood is the
   principled endpoint but is data-gated and that track is separately retired.

---

## 13. Work breakdown and execution order

§§1–12 specify *what* is being tested. This section specifies *how it is executed*: seven chunks
with explicit dependencies, exit gates, and a compute budget. **Do not start a sweep before C3.**

### 13.1 Chunk list

| # | Chunk | Depends on | Exit gate |
|---|---|---|---|
| **C0** | **Prerequisite code fixes** (all four, one chunk) | — | regression tests green; `Σ=I` reproduces `mse` exactly; diagonal arm invariant across the subsetting fix (§11) |
| **C1** | **Peptide generator + `M` diagnostics** (§7.3) + **noise generator variants** (§3.4) | C0 (boundary policy) | `res1` reproduces shipped dfrac on retained rows ≤1e-10; all §7.3 variables persisted per layout |
| **C2** | **Population-path preflight** (§7.2) | C1 | per-loss argmin recorded against α=0.4 |
| **C3** | **Sizing decision** — fix split/seed/reg axes against a measured per-fit cost | C1 | written budget; timing probe on one real fit |
| **C4** | **Block 1 ladder** (§8.2) | C0–C3 | held-out predictive LL + calibration per cell |
| **C5** | **Block 2 Σ arms** (§9), incl. coverage-variance correlation | C4 | per-arm predictions from §9.1–9.3 and §9.6 evaluated separately; `noise_shared` vs `noise_indep` contrast reported |
| **C6** | **Leakage quantification** (`Sigma_weighted`, §9.4) | C5 | leak margin = control margin − leak-free margins |

C0 and C1 are pure engineering and can start immediately. C2 and C3 both depend only on C1 and can
run in parallel.

### 13.2 C0 — the four prerequisite fixes, collected

Scattered across §§3–7 in the specification; collected here because they must all land first:

1. **Summed likelihood** (§3.2) — replace the `/(T·n_fragments)` average, or rescale the KL term by
   observation count. Without this, varying peptide count silently changes regularization strength.
2. **Precision subsetting** (§6.6) — subset Σ, *then* regularize → invert → normalize. Build train
   and validation precisions from their own subsets.
3. **Per-ensemble Σ** (§6.4) — stop hardcoding `ISO_BI`.
4. **Boundary policy** (§6.7, §7.1) — resolve 293 vs 294 and restate the reproduction anchor in its
   terms.

### 13.3 C2 gate — what happens when a loss fails the preflight

A loss whose population-path optimum is not near α = 0.4 is **excluded from Block 1 and reported as
excluded**, with its profile persisted. It is *not* silently carried forward, and it is *not* grounds
for halting the other arms. If **`mse` itself fails**, that is a stop-the-line result: the baseline
is uninterpretable and the whole comparison is void until the cause is understood.

### 13.4 C3 — sizing decision (first pass)

The grid as specified in §§8–9 is not affordable. Naive full factorial:

| | layouts | ens | arms | splits | reg | fits |
|---|---|---|---|---|---|---|
| Block 1 | 24 | 2 | 6 | 15 | 6 | 25,920 |
| Block 2 | 24 | 2 | 7 | 15 | 6 | 30,240 |
| | | | | | **total** | **56,160** |

**Two cuts, both justified rather than arbitrary:**

- **Split axis 15 → 3.** Use one strategy (`random`, 3 replicates) for the layout sweep. `random`
  preserves the train/validation peptide overlap that *is* the phenomenon under study;
  `sequence_cluster` is explicitly non-redundant and would remove it. Overlap does make held-out
  scores partly predictable from training data through shared residues — but that inflation applies
  equally to every loss, so between-loss comparisons remain valid. `sequence_cluster` is retained as
  a reduced-scope contrast (C7 row below) and is the natural negative control: redundancy-aware
  losses should help *less* there.
- **Seed axis staged.** The ~20 random seeds are only needed for the §9.1 coverage-variance
  correlation. The ladder itself runs on 4 regular layouts + 3 random seeds.

| Stage | layouts | ens | arms | splits | reg | fits |
|---|---|---|---|---|---|---|
| C4 Block-1 ladder | 7 | 2 | 7 | 3 | 6 | 1,764 |
| C5a Block-2 arms | 7 | 2 | 8 | 3 | 6 | 2,016 |
| C5b coverage-variance correlation | 20 | 2 | 3 | 3 | 6 | 2,160 |
| C6 leakage control | 7 | 2 | 1 | 3 | 6 | 252 |
| C7 split-strategy contrast (`sequence_cluster`) | 4 | 2 | 7 | 3 | 6 | 1,008 |
| | | | | | **total** | **7,200** |

**≈7.8× reduction (56,160 → 7,200)**, with the ~20-seed requirement preserved exactly where it is
load-bearing (C5b) and dropped where it is not.

**Still unmeasured:** per-fit wall-clock. C3 does not exit until a timing probe on one real fit is
run and the budget is written down — 7,200 fits is only affordable if a fit is seconds, not minutes.
If the probe says otherwise, cut C5b's arm count before cutting seeds, since the seed count is what
makes that correlation meaningful at all.

### 13.5 Reporting discipline across chunks
Every chunk writes to its own dated artifact directory with a manifest recording inputs, the C3
budget actually used, and which cells were excluded by the C2 gate. Per-cell results throughout;
never pooled.
