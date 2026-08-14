# MoPrP pivot calibration — population/Jacobian structure under pivot × loss × coefficients

**Status:** step 1 complete; steps 2--4 remain staged. Written 2026-08-05.

**Inherits from** `plans/hdx_rate_space_pivot_reweighting.md` — §13 (Stage 1 semantics matrix) and
§6 Stage 2 (τ/EX1 arm). That document is a **closed ISO record**; results from this stage belong
here, not there.

---

## 1. Context

The ISO investigation is closed. It established two things:

- **The frame-averaging pivot is a first-order, sign-definite error source** (§13). On ISO_BI
  residue the self-consistent diagonal is 0.006–0.008 absolute open-population error while the worst
  pivot mismatch is 0.175 — roughly 20×. The sign follows the Jensen ordering (AM ≥ GM) and is
  predictable a priori.
- **EX1 contamination breaks `fast`/`slow2` far worse than `legacy`** (§6 Stage 2). At
  `τ·k_int,median ≈ 10`, 83% of `fast` fits collapse below ESS 2 and 73% expel open mass entirely,
  while `legacy` never collapses at either τ.

What ISO could **not** establish is **which pivot is correct**. Every ISO target was self-consistent
by construction, so the diagonal measures identifiability, not correctness. Worse, the shipped ISO
target was itself manufactured with `k̄_first` (`create_mixed_target_data.py`), so that benchmark
cannot certify the pivot it was built with.

Answering "which is correct" requires real data with an **independent** population truth. MoPrP has
one: `analysis/state_ratios.json` gives an NMR-derived thermodynamic population of
**97.1% Folded / 2.4% PUF1 / 0.52% PUF2**.

This document covers **only the calibration stage** — lifting the synthetic/Jacobian structure onto
MoPrP so a later fitting experiment has units. The fitting experiment is deliberately not designed
here.

## 2. Three facts that shape the design

### 2.1 The truth sits below the ISO floor

Minority mass totals **2.9%**, at or under the ISO-established recovery floor of 0.02–0.06 absolute
population error. Pivot effects at the true population may be unresolvable in principle.

Consequence: the stage must **locate** the floor rather than assume it, and the population grid must
extend well above `w_NMR` so there is a resolvable regime to measure from.

### 2.2 Pivot and (bc, bh) are near-degenerate

The legacy→fast gap is a rate multiplier `≈ exp(½Var_f(z))` with
`z = bc·N_heavy + bh·N_acceptor`, so `Var_f(z)` scales **quadratically** with coefficient magnitude:
bc/bh *set* the pivot effect size. And per §7 of the ISO document, both a pivot change and a bc/bh
rescale are pure **position shifts on log-t**.

On ISO this was neutralised by freezing bc/bh at 0.35/2.0 (the D-only guardrail), which is precisely
why the pivot showed up as a first-order effect there. **That guardrail does not transfer.** MoPrP's
published coefficients are not representative, so the final experiment must tune bc/bh *and*
reweight jointly — and jointly-tuned coefficients may absorb the pivot entirely.

Measuring that absorption is therefore the **primary deliverable** of this stage.

### 2.3 The shipped Σ is not an observation-noise covariance

`data/_MoPrP_covariance_matrices/Sigma.npz` is 14×14 with eigenvalues spanning **1.04e-6 to 0.944,
condition number 9.1e5**. The conditioning is a symptom; the cause is what the matrix contains.

From `1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py:162`:

```python
Sigma = np.cov(_dfrac_values) + np.diag(np.full(_dfrac_values.shape[0], 1e-6))
```

`_dfrac_values` is `(14 peptides, 15 timepoints)`, so `np.cov` builds a peptide–peptide covariance
by treating **the 15 timepoints as 15 independent samples**. It measures how similarly peptide
uptake *curves rise over time* — signal correlation between peptides, not measurement error. No
replicate data enters it, and MoPrP ships none (`_MoPrP/_output/` contains only dfrac, pfactors,
segments, rates). The script's "observation noise covariance" label is inaccurate.

Three verified consequences:

1. **Rank is capped by construction.** 15 samples ⇒ maximum rank 14 for a 14×14 matrix: borderline
   singular before anything else happens. Because every uptake curve is a monotone sigmoid, the
   peptides are additionally near-collinear — **90.6% of variance in one eigendirection, 98.8% in
   two**, effective rank ~3–4. The dominant direction is essentially the mean uptake curve shape.
2. **The bottom of the spectrum is literally the jitter.** Shipped Σ minus raw `np.cov` equals
   exactly `1e-6` on every one of the bottom eigenvalues. The raw covariance's smallest eigenvalue
   is `4.1e-8`; the shipped value is `1.04e-6` — **96% of it is the regularisation constant**.
3. **Σ⁻¹ therefore inverts the jitter.** That direction receives weight `1/1.04e-6 ≈ 10⁶`. Σ-MSE is
   not weighting peptides by uncertainty; it is concentrating weight on directions defined by an
   arbitrary numerical constant. Changing `1e-6` to `1e-5` changes the loss's character entirely.

**Resolution.** The shipped matrix is not repairable by rescaling; Σ must be rebuilt from a
different estimand — see **§11**. Until that is settled, the weighting is an explicit axis rather
than an inherited matrix. Replace the single Σ-MSE arm of §5 with:

- **eye/MSE** — unweighted baseline.
- **diagonal Σ** — per-peptide variance only. Well-conditioned, reduces to weighted MSE, defensible
  as "peptides with more dynamic range carry more signal". Inspect peptide 13 first: its variance is
  0.003 against 0.05–0.11 elsewhere, so it would receive ~30× weight.
- **shrunk Σ, shrinkage α swept** — via the existing `shrink_covariance` /
  `shrunk_trace_normalized_precision` (`jaxent/src/analysis/state_population.py:235`, default
  `alpha=0.05`), which `moprp_population_oracle.py` already uses.

**The decisive test:** if Σ-MSE beats MSE only as `α → 0`, its advantage is jitter inversion rather
than information, and the result must be reported that way. The shipped `Sigma.npz` is a broken
baseline to report against, not the definition the comparison rests on.

## 3. Prior art — reuse, do not rebuild

| artefact | what it already provides |
|---|---|
| `fitting/jaxENT/moprp_population_oracle.py` | "Stage A4: population-space identifiability oracle". Population recovery from deterministic starts, Jacobian w.r.t. state-population logits, singular spectra, null-direction detection, zero-target decoy rejection — across 4 covariance coordinates × 2 coefficient settings. **This is the scaffolding.** |
| `fitting/jaxENT/_moprp_recovery_common.py` | Canonical input assembly: physics-v2 hard-count BV features (97 residues × 500 frames), canonical exPfact intrinsic rates, trim-one 14-peptide map, real 14×15 uptake, and `w_NMR`. **All inputs come from here** so conventions stay identical across runners. |
| `fitting/jaxENT/moprp_coefficient_lock.py` | Stage A3 shared `(Bc, Bh)` lock. **Not yet run** — no output JSON. See §4. |
| `jaxent/examples/common/analysis/frame_averaging.py` | `residue_uptake_legacy` / `_fast` / `_slow2` and `effective_rates`, property-tested during the ISO work. The pivot oracle. |
| `fitting/jaxENT/investigate_pivot_convention.py` | Existing MoPrP `k̄_after` vs `k̄_first` comparison, for target-variance inference. Check its conventions match before adding a third pivot. |

Extend the oracle with the pivot and loss axes. Do not write a second oracle.

## 4. Required correction before anything else

`moprp_coefficient_lock.py` fits `(Bc, Bh)` **"using average-first semantics"** — i.e. under
`legacy`. Its `constrained_optimum` is therefore **pivot-conditioned**. Feeding that single value to
`fast`/`slow2` would score those models using coefficients fitted by their competitor: the ISO
circularity reappearing in a new form.

**Lock coefficients per pivot.** Run the Stage A3 lock three times, once under each pivot, producing
`constrained_optimum[legacy | fast | slow2]`. Retain `published` = (0.35, 2.0) as the shared frozen
control. A boundary optimum (e.g. `Bh → 0`) is reported as **model inadequacy**, not a physical
estimate — as that script already specifies.

The lock has not been run, so this is free to get right now rather than something to unwind later.

## 5. Design

**Ensemble: AF2-MSAss only.** It has 5 populated clusters (labels 0–4 →
Folded / PUF1 / PUF2 / PUF3 / unfolded via `DEFAULT_STATE_MAPPING`), matching "one structure per
cluster" exactly. AF2-Filtered has 4 populated clusters and one of them holds a single frame —
excluded.

**Structures: one medoid per cluster**, selected in **BV feature space** (the `log_pf` vector the
forward model actually sees, via `inputs.log_pf_by_frame`) rather than by RMSD. Frame weights then
*are* state populations, and the Jacobian is exactly the population Jacobian.

> **Caveat, to carry into every result from this stage.** With one structure per state there is no
> within-state variance, so the legacy↔fast spread is driven by **between-state variance alone**.
> The measured pivot effect is a **lower bound** on what the full 500-frame ensemble would show.
> Verify against the full ensemble before any conclusion transfers.

**Population grid.** PUF1:PUF2 held at the NMR ratio (≈82:18), Folded taking the remainder. Decoy
states (PUF3, unfolded) carry zero true mass but remain free in the fit, so decoy rejection is still
tested:

- **broad** — minority total ∈ {0.40, 0.20, 0.10, 0.05}
- **dense around `w_NMR`** — minority total ∈ {0.045, 0.035, 0.029, 0.025, 0.020, 0.015}

**Axes.** 3 pivots × 4 weightings (eye/MSE, diagonal Σ, shrunk Σ at α ∈ {0.05, →0}; see §2.3) ×
2 coefficient settings (`published`, per-pivot `constrained_optimum`) × 10 population points.
Forward evaluation plus small population-space optimisations over 5 frames — cheap, as intended for
a stage that precedes a fitting experiment.

## 6. Measurements

1. **Pivot/coefficient absorption — primary.** Build the Jacobian of the peptide-uptake observable
   w.r.t. `(population logits, bc, bh)` jointly. Project the pivot-induced change in the observable
   onto the span of the `∂/∂(bc,bh)` columns and report the fraction captured.
   - ≈100% ⇒ the pivot is **unidentifiable** once coefficients are tuned. On MoPrP that is the
     answer to the original question, and it means the ISO effect size does not transfer to any
     workflow that fits coefficients.
   - <100% ⇒ the residual component is what carries population information, and its size bounds what
     the fitting experiment can hope to resolve.
2. **Resolution floor** per (pivot, loss, coefficient setting) — the minority population at which
   recovery error stops shrinking. Compare against the ISO 0.02–0.06 floor and against `w_NMR`'s
   2.9%.
3. **Jacobian conditioning and null directions** in population space, using the oracle's existing
   singular-spectrum machinery.
4. **Decoy rejection** — mass placed on PUF3 / unfolded, which carry zero true mass.
5. **Weighting comparison across all of the above.** This is a **loss axis, not a metric axis**:
   report which weighting lowers the floor, and whether any Σ-MSE gain survives shrinkage. Per §2.3,
   a gain that appears only as `α → 0` is jitter inversion, not information, and must be reported as
   such.

## 7. Pre-flight checks

- **Σ handling.** `optimise_ISO_TRI_BI_splits_maxENT.py:241` still divides the covariance by its
  Frobenius norm before `loader.py`'s `_trace_normalise` rescales it to `trace(W) == n`. The second
  rescale should override the first, making the double normalisation harmless — **confirm
  numerically rather than assume**, and confirm which of `Sigma` / `Sigma_inv` the loss consumes.
  Note that trace-normalisation cannot repair §2.3: it fixes the overall scale, not the spectrum,
  so the 10⁶ weight ratio between eigendirections survives it untouched.
- **Loss aliasing.** Verify `hdx_uptake_MSE_loss` still aliases `hdx_uptake_eye_MSE_loss`
  (`jaxent/examples/common/losses.py:31`) so the two losses are genuinely different. This alias was
  previously wrong, silently making MSE and Σ-MSE the same function.
- **Medoid adequacy.** Confirm each of the 5 medoids reproduces its cluster's mean `log_pf` to a
  stated tolerance. If a cluster is too heterogeneous for one structure to represent — plausible for
  Folded, which holds ~392 of 500 frames — say so rather than proceeding.

## 8. Verification

- At `w = w_NMR`, `published` coefficients, `legacy` pivot: predicted peptide uptake must match the
  existing MoPrP baseline assembled by `_moprp_recovery_common.py`. This is the check that the
  5-structure reduction has not silently changed the forward path.
- Jacobian columns validated against finite differences.
- Results are written into this document, not into `plans/hdx_rate_space_pivot_reweighting.md`.

## 9. Explicitly out of scope

The fitting experiment itself. This stage produces the units — resolution floors, Jacobian
conditioning, and the pivot/coefficient absorption fraction — that a fitting design needs before it
can be specified.

## 10. Pre-registered risks

- **The absorption result may be null-but-decisive.** If coefficients absorb the pivot entirely,
  there is no MoPrP pivot experiment to run. That is a legitimate and valuable outcome, and must not
  be treated as a failed stage or worked around by re-freezing bc/bh — freezing them would recreate
  the ISO condition rather than test the real one.
- **The 2.9% truth may be unreachable by every combination.** Also a legitimate outcome; it would
  say the MoPrP population is not identifiable from HDX uptake at this peptide resolution,
  independent of pivot choice.
- **Single-medoid reduction understates the pivot spread** (§5). Any positive effect measured here
  is a floor on the real one; any null result needs the full-ensemble check before it is believed.

---

## 11. Rebuilding Σ

§2.3 establishes that the shipped Σ is a peptide covariance over uptake *curves*, estimated with
timepoints as samples. Rebuilding it is a prerequisite for any Σ-weighted result.

### 11.1 The constraint that rules the design

**Residuals alone do not fix the rank problem.** Any 14×14 peptide covariance estimated from 15
timepoint-samples is rank-capped at 14 with `n ≈ p`, whatever quantity is being covaried. Using
residuals fixes the *collinearity* — removing the shared sigmoid shape that currently occupies 90.6%
of the variance — but not the *sample count*. A usable construction must **increase effective
samples or change the estimand**. Shrinkage remains mandatory in every option below.

Checked and unavailable: `moprp.dexp` is `(15, 15)` = one time column plus 14 peptides, so there are
no per-observation replicates. `HDXExperimentProtocol(replicate_count=3)` is protocol metadata, not
data. **True observation noise cannot be recovered from what ships**, so every option below estimates
something else and must be named accordingly.

### 11.2 Candidate constructions

**(A) Model-residual covariance.** `residual = observed_uptake − predict_ex2_uptake(exPfact best PF)`,
covaried over peptides. Samples remain 15, so still rank-marginal, but the spectrum should flatten
substantially once the shared curve shape is removed. Estimand: *what the EX2 forward model cannot
explain* — defensible as a misfit weighting. Cheap; use as a cross-check.

**(B) exPfact multistart-solution covariance — recommended primary.**
`fit_ex2_solution_set(..., starts=20)` retains every finite start, and `EX2SolutionSet` is explicitly
documented as performing no residue-wise averaging across modes. Covary the *predicted uptake* across
those solutions. Estimand: **the data's own degeneracy** — the directions in peptide space the
experiment cannot constrain. That is precisely the right thing to down-weight, and sample count is
set by `starts`, which is cheap to raise well above 14.

Circularity note: (B) is derived from a fit to the same uptake data, so it is not independent noise.
It is, however, **population-independent**, which is the circularity that matters here — unlike an
ensemble-derived covariance, whose population dependence is exactly what
`moprp_noncircular_recovery.py` (Stage D) exists to address. Keep that distinction explicit.

**(C) Ensemble uptake covariance.** `peptide_uptake_covariances` /
`peptide_logpf_covariance` (`jaxent/src/analysis/state_population.py`), 500 frames. Richest sample
count, but it is a *conformational-heterogeneity* covariance and is population-dependent, so using it
as a loss weight reintroduces the Stage D circularity. Already explored by the oracle as the
"dynamic geometry" coordinate — treat as a known-circular comparator, not as the fix.

### 11.3 Acceptance criteria

Any replacement Σ must be reported with: sample count and resulting rank; the eigenvalue spectrum
and the fraction of variance in the top direction; the shrinkage level required for a stable
inverse; and an explicit statement of estimand ("misfit", "data degeneracy", "conformational
spread") rather than the unqualified label "observation noise".

## 12. Staging — pause point before Σ-weighted work

Σ construction is unresolved, so the stage is split and **must not run straight through**:

1. **Litmus test (fit-free, first).** exPfact protection factors are fitted per-residue directly from
   the uptake data, independent of any ensemble or pivot. So comparing the ensemble's
   pivot-averaged `log_pf` against the exPfact `log_pf` is a **direct pivot comparison with no
   reweighting at all** — the cheapest discriminator available, and the first real-data evidence on
   which pivot is correct.
   - Source: `_MoPrP/_output/MoPrP_pfactors.dat`, 49 residues (ids 4–101, ln PF 0.221–8.142).
   - **Alignment constraint:** exPfact resolves 49 residues against the 97 feature residues, so the
     comparison is restricted to that overlap. State the overlap explicitly.
   - `TrajectoryHDXComparison` already carries `average_first_curves` / `frame_mixture_curves`, and
     `compare_trajectory_hdx` may cover part of this — check before writing anything new.
2. **Synthetic Jacobian on eye/MSE only** (§5, §6), with the Σ arms omitted.
3. **PAUSE. Take stock of Σ construction** against §11 — decide the estimand, rebuild, and report the
   §11.3 acceptance criteria before proceeding.
4. **Jacobian + Σ-weighted arms**, only once step 3 has settled.

Steps 1 and 2 are independent of the Σ question and can proceed now. Step 4 is blocked on step 3;
no Σ-weighted number from the shipped matrix should enter a conclusion.

---

## 13. Step 1 result — full-ensemble exPfact pivot litmus

**Intrinsic-rate provenance amendment (2026-08-10; see the kint handoff §8):** the shipped
`median.pfact` was fitted by exPfact with rates calculated in memory at 298 K/pH 4.4 using the 2021
PDLA constants. It was not fitted with the adjacent `moprp.kint`, and the historical vector also
predates and differs from the current 3Ala `expfact_kint_pH4p4_298K_min.dat`. Therefore this
section's BV-versus-exPfact PF scoring is not an exact same-rate comparison. Retain it as a
rate-source-conditional diagnostic; do not use its absolute PF distances to rank pivots or describe
the current canonical file as the source of the shipped PF reference.

Implemented in `fitting/jaxENT/moprp_pivot_litmus.py` and run on the 500-frame AF2-MSAss
ensemble at fixed `w_NMR`. No frame weight was fitted or changed. The only search was a 41×41
coefficient grid, `bc ∈ [0,1]` and `bh ∈ [0,4]`. Machine-readable output is in
`fitting/jaxENT/_moprp_pivot_litmus/`.

### 13.1 Resolvability verdict (must be read first)

**Inconclusive: MoPrP is underpowered for pivot discrimination by this test.** Twenty finite
multistart EX2 solutions were retained. The published-coefficient `legacy-fast` log-PF gap exceeded
the full multistart solution range on **0/48 testable overlap residues (0%)**, far short of the
pre-registered “most residues” gate. The nominal pivot differences below are effect sizes, not a
resolved choice of pivot.

There are 49 exPfact/feature overlap residues exactly as expected. Residue 4 is in that scalar
overlap but is inactive after the trim-one peptide construction, so its unregularized multistart
range is undefined and the gate tests 48 residues. The overlap touches 13 of 14 peptides:
**1--7 and 9--14** (not peptide 8).

**The gate failed conservatively, not marginally.** The `legacy-fast` gap scales roughly as
`λ²·½Var_f(z)` under a coefficient rescale `(bc,bh) → λ(bc,bh)`, and the published `bc` is
1.6--2.3× the scan optimum. The gap tested against the multistart range is therefore *inflated*
relative to correctly-scaled coefficients, and it still lost on every one of the 48 residues. The
inconclusive verdict errs in the safe direction.

**Skill against no-skill baselines.** Read the RMSEs in §13.2--13.3 against what a constant
predictor achieves on the same data:

| space | baseline | RMSE | best model | verdict |
|---|---|---:|---:|---|
| peptide uptake | per-timepoint mean | 0.2200 | fast 0.2012 (scanned) | modest real skill (~8%) |
| peptide uptake | per-timepoint mean | 0.2200 | fast 0.3362 (published) | **negative skill** |
| residue log-PF | exPfact mean (sd 1.926) | 1.9258 | fast 2.0964 (scanned) | **negative skill** |

The correct uptake baseline is the *per-timepoint* cross-peptide mean (0.2200), not the pooled sd
(0.3265): 9 of the 15 timepoints lie below 1 min and the pooled figure is inflated by the kinetic
trend, which any forward model reproduces for free. On that fair baseline the coefficient-scanned
models do carry real, if modest, skill.

The negative log-PF skill is the weaker of the two claims and should not be over-read. The shipped
exPfact PF vector is produced under exPfact's second-difference (harmonic) smoothing, so it is a
*smoothed* reference; a per-residue BV prediction with genuine residue-to-residue roughness is
penalised in both RMSE and Spearman against it, independently of whether the BV prediction is
right. The comparison is indicative, not a clean skill test.

**On the `bh = 0` boundary optima.** These are expected here rather than diagnostic of BV itself:
the ensemble is AF2-derived, and predicted structures do not carry reliable hydrogen-bond geometry,
so the acceptor-count channel has little trustworthy signal to contribute. Record them as an
ensemble-source limitation, not as evidence that the acceptor term is unnecessary in general. The
published `(0.35, 2.0)` coefficients remain clearly unsuited to this system on the separate evidence
of the +3.86 to +4.85 ln-unit systematic over-protection in §13.2.

### 13.2 Panel A — scalar effective log-PF

At the frozen published coefficients `(bc,bh)=(0.35,2.0)`:

| pivot | mean signed difference vs exPfact | RMSE | Spearman ρ |
|---|---:|---:|---:|
| legacy | +4.8525 | 6.0793 | 0.2492 |
| fast | +3.8554 | 5.0566 | 0.4253 |

The 41×41 coefficient scan reduced the best attainable PF RMSE to **2.1769** for legacy at
`(0.150,0.700)` and **2.0964** for fast at `(0.175,0.300)`. Thus fast has the lower nominal floor
by 0.0805 log-PF units (3.7%), at a substantially different coefficient pair. Because the
resolvability gate failed completely, this small floor difference cannot identify fast as the
correct pivot; coefficient absorption and exPfact inverse degeneracy are both larger than the
discriminating signal.

**Do not carry the fast-over-legacy ordering forward as a preference.** Both floors sit above the
constant-predictor baseline of 1.926 (§13.1), so the 3.7% separation is a gap between two models
that the reference cannot rank, measured in a regime where neither demonstrates skill against that
reference. It is an effect size only.

Jensen ordering held on every residue. The observed `legacy-fast` gap tracked the second-order
`½ Var_w(log_pf)` term strongly (Spearman ρ = **0.9560**), but the approximation overestimated the
gap by 0.6548 log-PF units on average and had RMSE 1.7357 at the published coefficients.

### 13.3 Panel B — 14×15 peptide uptake

Raw RMSEs at published coefficients were:

| pivot | versus observed uptake | versus exPfact reconstruction |
|---|---:|---:|
| legacy | 0.37012 | 0.48755 |
| fast | 0.33623 | 0.45211 |
| slow-N | 0.36678 | 0.48394 |

After scanning coefficients, the best RMSEs versus exPfact reconstruction were tightly grouped:
legacy **0.22917** at `(0.150,0.200)`, fast **0.22597** at `(0.175,0.000)`, and slow-N **0.22890**
at `(0.150,0.200)`. Versus the observations they were 0.20706, 0.20118, and 0.20639,
respectively, with all three optima at or near `(0.225,0.000)`. These nearly common uptake floors
are consistent with strong coefficient absorption. The boundary `bh=0` optima are not physical
coefficient estimates, but they are also not general model-inadequacy flags — see §13.1 on the
AF2-derived hydrogen-bond geometry, which is the more likely explanation here.

`slow-N` is intentionally absent from Panel A. A weighted mixture of frame-level exponentials has
no exact scalar effective PF. exPfact itself assumes EX2 with one PF per residue, so mixture pivots
are structurally disadvantaged in this comparison; no curve-matched pseudo-PF was manufactured.

### 13.4 Verification and caveats

- The two independent effective-rate implementations agreed bit-for-bit (maximum absolute
  difference 0.0); all Jensen violations were zero.
- Concentrating all weight on one frame made legacy, fast, and slow-N curves identical (maximum
  absolute difference 0.0).
- The new legacy path reproduced `predict_trajectory_ex2` at `w_NMR` and published coefficients
  exactly (maximum absolute difference 0.0).
- The 49-row exPfact export supplies the scalar PF reference. Its regression-tested uptake
  reconstruction uses the complete `median.pfact` vector, including exPfact's `-1` sentinel values
  for unresolved residues.
- exPfact PFs were fitted from these same 14 peptide curves, so they are not an independent
  measurement. They are independent of the ensemble and pivot, which is the independence this
  litmus needs.
- The 49 residues are a peptide-map-constrained, non-random subset. `w_NMR` remains a pseudo-truth
  whose state populations are spread uniformly within each state; within-state weighting is an
  assumption, not a measurement.

**Consequence for staging:** step 1 supplies no data-resolved pivot choice. Its multistart solution
set directly supports the option-B Σ construction in §11, while the eye/MSE synthetic Jacobian in
step 2 remains independent and may proceed.

One ordering question is now open. Every coefficient optimum in §13.2--13.3 lies far from published
`(0.35, 2.0)` — `bc` at 0.15--0.225 and `bh` on the zero boundary — and at published coefficients
the forward model has negative skill against the per-timepoint baseline. Step 2's synthetic Jacobian
would therefore measure its resolution floors in a coefficient regime the data actively rejects.
**Consider promoting the per-pivot coefficient re-lock (§4) ahead of step 2** so the floors are
measured where the model has demonstrated skill. This is a scheduling decision, not a blocker:
step 2 is internally valid at any fixed coefficient setting, and its conclusions are conditional on
that setting either way.

---

## 14. Per-pivot BV coefficient re-lock

The promoted §4 lock was implemented and run before step 2. It uses fixed `w_NMR`, excludes held-out
peptide 1, and fits one shared non-negative `(bc,bh)` pair across both 500-frame ensembles separately
for `legacy`, `fast`, and `slow-N`. No frame weights were fitted or changed. The machine-readable
lock and the complete 41×41 three-pivot profile are in
`fitting/jaxENT/_moprp_recovery_coefficient_lock/coefficient_lock.json` and
`coefficient_profile.csv`.

For backward compatibility, `frozen_settings` still contains the legacy `published`,
`constrained_optimum`, and `scaled_published` settings expected by current consumers. The new
`frozen_settings_by_pivot` block carries the corresponding settings for all three pivots.

### 14.1 Shared locks and calibration floors

| pivot | constrained `(bc,bh)` | own-optimum MSE (RMSE) | scaled-published `λ` | scaled-published MSE | published MSE |
|---|---:|---:|---:|---:|---:|
| legacy | (0.228893, 0) | 0.041430 (0.203545) | 0.532115 | 0.045075 | 0.127242 |
| fast | (0.232237, 0) | 0.041861 (0.204600) | 0.540363 | 0.045632 | 0.118200 |
| slow-N | (0.228957, 0) | 0.041163 (0.202887) | 0.531926 | 0.044880 | 0.126370 |

All scaled-published optima are safely interior to the pre-specified `λ ∈ (0.001,3.0)` interval.
The published coefficients are rejected again after removing peptide 1 and adding AF2-Filtered:
their MSE is roughly three times each pivot's attainable floor.

Every constrained optimum has `bh=0`. As in §13.1, this is recorded as an **AF2 hydrogen-bond
geometry limitation**, not general BV model inadequacy: predicted structures provide little
trustworthy acceptor-channel signal. The constrained fit therefore cannot assign a physical
interpretation to `bh=0`.

### 14.2 Absorption readout

The three optimum vectors are exactly collinear (all pairwise angles **0°**), but this carries no
information: every optimum has `bh = 0`, so all three vectors lie on the `bc` axis by construction
and the angle would be 0° whatever the pivot did. It is reported only to forestall reading it as
evidence. Relative optimum magnitudes are `fast/legacy = 1.01461`, `slow-N/legacy = 1.00028`, and
`slow-N/fast = 0.98588`: the fitted pivot correction is a 0--1.5% rescale of `bc`.

**The rescale does not cancel the pivot.** Measured on AF2-MSAss at `w_NMR`:

| quantity | mean ln PF |
|---|---:|
| legacy at its own optimum (`bc = 0.228893`) | 4.7224 |
| fast at its own optimum (`bc = 0.232237`) | 4.6293 |
| legacy→fast gap at fixed `bc = 0.228893` | 0.1561 |
| **residual after each pivot moves to its own optimum** | **0.0931** |

The 1.46% `bc` increase absorbs only about **40%** of the mean legacy→fast log-PF gap; a ~0.093
ln-unit offset (≈10% in protection) survives re-locking.

The second-order rate-multiplier diagnostic `exp(½Var_f(z))`, pooled across the two ensembles at
each optimum, has median **1.0448--1.0461** and mean **1.0783--1.0808**. It correctly predicts a
small upward coefficient/rate compensation for `fast`, but overstates the fitted norm ratio
(1.0146). This approximation is residue-level and rate-space local, whereas the lock minimizes a
peptide/time uptake loss, so numerical equality is not expected.

The independently optimized floors span only 0.000698 MSE (RMSE 0.2029--0.2046). Combined with the
failed §13 resolvability gate, this is the §4 result: **MoPrP does not identify the pivot once BV
coefficients are free.**

**But the mechanism is weak identifiability, not absorption**, and the distinction is consequential.
True absorption would mean the coefficient reparameterises the pivot away, in which case *no*
observable could separate the pivots. What actually happens is that a real 0.093 ln-unit mismatch
survives re-locking and simply costs almost nothing in this objective — the uptake loss is flat
along that direction. A different observable, in particular the population-space Jacobian of step 2,
may therefore still separate the pivots. This keeps step 2 worth running.

One weak signal is worth recording rather than dismissing as noise: the floors are **monotone in how
much rate-averaging each pivot performs** — `slow-N` (0.041163) < `legacy` (0.041430) < `fast`
(0.041861) — which is the direction Jensen ordering predicts. At a 1.7% spread this cannot rank the
pivots, but it is ordered as theory expects rather than arbitrarily.

The scaled-published direction remains worse than the free direction by 0.00364--0.00377 MSE, or
**8.8--9.0%**, for every pivot. Hence the published `Bc:Bh` direction is itself wrong for these AF2
ensembles; the discrepancy cannot be repaired solely by rescaling its magnitude.

### 14.3 Per-ensemble diagnostic

| ensemble | legacy `(bc,bh)` | fast `(bc,bh)` | slow-N `(bc,bh)` | MSE range |
|---|---:|---:|---:|---:|
| AF2-MSAss | (0.228816, 0) | (0.232475, 0) | (0.228884, 0) | 0.041091--0.041845 |
| AF2-Filtered | (0.228970, 0) | (0.232000, 0) | (0.229031, 0) | 0.041235--0.041877 |

The ensemble-specific `bc` values differ by at most 0.00048 for a given pivot (below 0.21%), and
both independently put `bh` on zero. The shared lock is therefore not a material compromise between
incompatible ensemble optima.

### 14.4 Verification

- The refactored legacy forward path reproduces the former hardcoded calculation bit-for-bit on
  both ensembles at `(0.35,2.0)` (maximum absolute difference 0.0).
- Re-including peptide 1 for AF2-MSAss gives legacy MSE **0.13698897**, whose square root is
  **0.3701202**, reproducing the §13 all-14-peptide value.
- Jensen ordering holds with zero violations at the published coefficients on both ensembles.
- Putting all mass on one frame makes all three pivot curves identical (maximum absolute difference
  0.0).
- Every scaled-published optimum is strictly inside its scalar-search bounds.

**Consequence for staging:** step 2 should use `frozen_settings_by_pivot[*].constrained_optimum`.
The legacy-compatible `frozen_settings` block is retained only so existing consumers do not change
semantics silently.

Step 2 proceeds as staged, with its framing sharpened by §14.2: uptake-space non-identifiability is
now established, so step 2's distinctive contribution is specifically whether **population-space**
recovery separates the pivots where the uptake loss does not. If it does not either, the pivot
question is closed for MoPrP and the answer is that the data cannot decide it — which is itself a
publishable result about the method, not a failed experiment.

---

## 15. Step 2 — population recovery and Jacobian by pivot

Implemented in `fitting/jaxENT/moprp_population_pivot.py`, reusing the population-logit optimizer
and start sets from `moprp_population_oracle.py` and the shared differentiable pivot injection point
in `moprp_pivot_litmus.py`. Machine-readable results are in
`fitting/jaxENT/_moprp_population_oracle_pivot/population_pivot_results.json`; the complete 180-cell
synthetic sweep is in `synthetic_resolution_sweep.csv`.

The primary target is the real measured uptake and therefore has only a **fitter-pivot** axis. The
target-pivot axis below belongs exclusively to the labelled synthetic instrument calibration. All
fits retain the full five-state support; PUF3 and unfolded are never removed or renormalised away.
Peptide 1 is held out throughout uptake fitting. The real-data objective is the regime-1 uptake MSE
normalised by uniform-population MSE plus `eta=0.01` KL to the uniform state population.

### 15.1 Resolution gate — read this before the primary result

**Corrected 2026-08-07.** This section originally reported the sweep as underpowered with a
resolution floor "above 40% minority mass", on the grounds that no cell reached the pre-registered
99% strict full-support recovery criterion. **That criterion was never calibrated against what the
metric can deliver, and the conclusion drawn from it was wrong.**

Doing nothing at all — uniform frame weights, no fitting — already scores 70.6--83.3% by
`strict_recovery_percent`, because full-support JSD recovery depends on the *entropy of the truth*
and every grid truth is dominated by Folded. Worse, that null baseline moves **non-monotonically**
across the grid:

| minority mass | 0.015 | 0.029 | 0.065 | 0.134 | 0.193 | 0.278 | 0.400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| null recovery | 70.6 | 73.3 | 78.2 | 82.9 | **83.3** | 80.4 | 73.0 |

So raw `recovery_percent` is **not comparable across grid points**, and the apparent degradation of
matched-legacy from 94.3% at 1.5% minority to 82.2% at 40% is largely this metric artefact, not a
loss of identifiability.

**Measured as gain over that null, the diagonal is positive at every one of the ten grid points, for
every pivot and both coefficient settings** — the per-cell range is +3.8 to +26.2 pp. There is
population information available across the entire 1.5--40% range. **The resolution floor is not
above 40%, and the sweep is not underpowered.**

#### The mismatch matrix

Mean gain over null across the grid, which is the quantity the sweep was built to produce:

*published coefficients*

| target ↓ / fitter → | fast | legacy | slow-N |
|---|---:|---:|---:|
| **fast** | **+18.9** | −27.8 | −36.3 |
| **legacy** | +1.5 | **+13.1** | +13.7 |
| **slow-N** | +1.6 | +11.7 | **+14.7** |

*per-pivot locked coefficients*

| target ↓ / fitter → | fast | legacy | slow-N |
|---|---:|---:|---:|
| **fast** | **+18.0** | −11.5 | −34.7 |
| **legacy** | +5.1 | **+14.0** | +12.7 |
| **slow-N** | +5.0 | +15.0 | **+14.4** |

This is the direct MoPrP analogue of ISO §13, and it reproduces that document's central structure on
real-ensemble features: the self-consistent diagonal is positive throughout, while pivot mismatch is
large and **sign-asymmetric**. Fitting a `fast` target with `legacy` or `slow-N` is *worse than not
fitting at all* (−11.5 to −36.3 pp), whereas the reverse direction merely surrenders most of the
gain (+1.5 to +5.1 pp). The asymmetry follows the Jensen ordering: a `fast` target presents as less
protected, so a `legacy` fitter over-protects and must push population toward open states,
overshooting into the decoys.

`fast` is also the best-conditioned fitter on the diagonal (mean +18.4 pp versus +13.5 for legacy and
+14.6 for slow-N) and the most stable across minority mass (+11.3 to +26.2, versus legacy's +3.8 to
+24.1).

**Consequence:** the primary results below are interpretable, contrary to the original reading. The
sanity gate that failed was a badly-chosen threshold, not evidence of an unresolvable instrument.
What remains genuinely limiting is stated in §15.5.

### 15.2 Primary real-uptake fit — full 500-frame ensemble

At each pivot's own locked coefficients:

| fitter pivot | strict recovery | decoy mass | uptake MSE | PF-space RMSE |
|---|---:|---:|---:|---:|
| legacy | 63.80% | 0.02644 | 0.041339 | 2.3844 |
| fast | 93.25% | 0.000114 | 0.041470 | 2.4532 |
| slow-N | 66.43% | 0.10350 | **0.040495** | n/a |

The uptake loss repeats §14's warning: slow-N has the best uptake MSE while recovering the truth
poorly and assigning 10.35% to decoys; fast has slightly worse MSE but the highest recovery. The
three MSEs span only 0.000975 — the uptake loss cannot tell these three apart, exactly as §14.2
found, yet they disagree by 29.5 recovery points. **Population space separates what uptake space
cannot**, which is the question step 2 was posed to answer.

Per the corrected §15.1, this separation is interpretable. It also agrees with the synthetic sweep
run independently of it: `fast` has both the highest diagonal gain over null (+18.4 pp mean) and the
highest real-data recovery (93.25%), while `legacy` is lowest on both (+13.5 pp, 63.80%). Two
independent routes producing the same ordering is the first positive pivot discrimination in this
investigation. It is a consistent signal, not yet a proof — see §15.5 for what would settle it.

The recovered locked-coefficient populations were:

| pivot | Folded | PUF1 | PUF2 | PUF3 | unfolded |
|---|---:|---:|---:|---:|---:|
| legacy | 0.67560 | 0.29472 | 0.00325 | 0.00014 | 0.02630 |
| fast | 0.99011 | 0.00966 | 0.00012 | 0.00002 | 0.00009 |
| slow-N | 0.72338 | 0.17272 | 0.00040 | 0.00007 | 0.10342 |
| NMR truth | 0.97119 | 0.02364 | 0.00517 | 0 | 0 |

Published coefficients make every result worse and confirm that this control is outside the
data-supported regime: recovery is 12.58% / 33.54% / 15.17% for legacy / fast / slow-N, with uptake
MSE 0.06249 / 0.05404 / 0.05376. The coefficient regime changes the apparent population answer far
more than the tiny uptake-floor differences, so published-coefficient population results have no
interpretive weight.

At the locked real-data optima, the uptake-Jacobian effective ranks were 3, 4, and 2 for legacy,
fast, and slow-N respectively (threshold `s/max(s) >= 1e-3`). Their singular spectra were:

- legacy: `(0.16356, 0.06220, 0.001089, 0.0000712, 3.1e-16)`;
- fast: `(0.04513, 0.007829, 0.001080, 0.000244, 7.0e-17)`;
- slow-N: `(0.42727, 0.05304, 0.000208, 0.0000393, 3.9e-16)`.

Thus fast has the most locally resolved population directions at its optimum — the local
conditioning agrees with both the synthetic diagonal and the real-data recovery ordering.

### 15.3 Protection-factor mirror and 2×2 cross-score

**Intrinsic-rate provenance amendment (2026-08-10; see the kint handoff §8):** this mirror compares
BV predictions made with the current 3Ala 298 K/pH 4.4 vector against `median.pfact`, which was
fitted with exPfact's historical 2021 PDLA vector at the same nominal conditions. It is therefore a
cross-rate as well as a cross-model score. The existing conclusion that the PF mirror cannot rank
pivots is unchanged, but its absolute PF RMSEs are not same-rate validation scores.

The PF mirror is restricted to legacy and fast. `slow-N` remains uptake-only because a mixture of
exponentials has no exact scalar effective PF. At locked coefficients:

| fit space / score space | legacy | fast |
|---|---:|---:|
| uptake fit → uptake MSE | 0.041339 | 0.041470 |
| uptake fit → PF RMSE | 2.3844 | 2.4532 |
| PF fit → uptake MSE | 0.056217 | 0.054337 |
| PF fit → PF RMSE | 1.9491 | 1.8788 |

The PF-fit recoveries were 39.84% for legacy and 74.58% for fast, with decoy masses 0.5510 and
0.03481 respectively. This is a relative nominal advantage for fast, but it is not resolved:

- the target is the same-data, smoothed exPfact inverse solution and the BV model has negative
  absolute skill in this space (§13.1);
- the 49 residues are a peptide-map-constrained non-random subset of the 97 features;
- the exPfact multistart solution-range band has median 28.14 and mean 27.46 ln units over the 48
  finite overlap residues, vastly larger than the 0.0703 locked PF-RMSE separation.

The PF-space comparison therefore corroborates neither pivot. It does show strong model-error
dependence: fitting PF space degrades uptake MSE, and fitting uptake space does not improve PF RMSE.

### 15.4 Five-medoid cross-check

One medoid per state was selected in published-coefficient BV `log_pf` space. Relative distances
from the cluster mean were 4.0% (Folded), 6.7% (PUF1), 8.2% (PUF2), 13.9% (PUF3), and 26.5%
(unfolded). The last two medoids are poor summaries, and PUF2/PUF3/unfolded contain only 10/11/15
frames. This cross-check is therefore a between-state lower bound, not an adequate replacement for
the full ensemble.

At locked coefficients the medoid recoveries were 42.59% / 46.08% / 43.89% for legacy / fast /
slow-N, with uptake MSE 0.04246 / 0.04226 / 0.04234. Removing within-state variance collapses the
nominal full-ensemble fast advantage and leaves the three fits nearly indistinguishable. This is
consistent with the apparent separation being driven by within-state details, but the failed
resolution gate means it is not evidence for a correct pivot.

### 15.5 Verification and conclusion

- The refactored legacy forward path reproduces `moprp_covariance_recovery.py` regime 1 bit-for-bit
  at `w_NMR` and published coefficients (maximum absolute difference 0.0).
- Every analytic population-Jacobian column agrees with a central finite difference; the maximum
  absolute discrepancy over all recorded fits is `2.23e-9`.
- Concentrating all mass on one frame makes all three pivot curves equal to `2.22e-16`; Jensen
  ordering has zero violations across every checked fitted population and coefficient setting.
- The PF mirror reproduces the §13 published-coefficient effective PFs with maximum absolute errors
  0.0 (legacy) and `1.78e-15` (fast).
- Decoy states remain free in every fit, and all primary scores use strict full-state support.

#### Known defect in the shipped output

**`uptake_mse` is meaningless for every synthetic-sweep row** in
`synthetic_resolution_sweep.csv`. `moprp_population_pivot.py:212` computes it against
`inputs.observed_uptake` regardless of which target was actually fitted, so synthetic rows report
real-data MSE rather than misfit to their own synthetic target — which is why the column sits at
~0.127 (legacy) and ~0.117 (fast), reproducing §14.1's *published-coefficient* calibration MSEs
almost exactly. The fits themselves are correct: targets are built properly and the objective uses
them. Only the reported column is wrong. Fix it to score against the fitted target, or drop it from
synthetic rows. No conclusion in this section rests on it.

**Step-2 answer:** **population space separates the pivots where uptake space does not.** The three
pivots' real-data uptake MSEs span 0.000975 — indistinguishable, as §14.2 predicted — yet their
recovered populations differ by 29.5 recovery points, and the synthetic mismatch matrix (§15.1)
shows large, sign-asymmetric mismatch costs on a diagonal that is positive at every grid point.
`fast` leads on all three independent readouts: synthetic diagonal gain (+18.4 pp mean), real-data
recovery (93.25%), and local Jacobian conditioning.

This is the first positive pivot discrimination in the investigation, and it inverts the working
assumption inherited from ISO, where `legacy` was the robust choice. Two cautions keep it from being
final:

1. **The real-data readout has no error bar.** One real target yields one recovered population per
   pivot; nothing bounds how much of the 29.5-point separation is noise. Resampling peptides or
   timepoints would supply one.
2. **`fast`'s advantage may be partly prior-driven.** `fast` recovers 99.01% Folded against a truth
   of 97.12%, while the null baseline is already 78.4% Folded — a fitter biased toward concentrating
   mass on Folded scores well here regardless of physics. The synthetic diagonal partly controls for
   this, but a truth with substantial minority mass and a *non-Folded-dominated* structure would
   test it directly.

Both are cheap and would settle whether `fast` is genuinely the correct pivot for MoPrP or merely
the best-scoring one under this particular truth.

---

## 16. Robustness to Folded-concentration bias

**Decision (2026-08-07): retract the unconditional §15 `fast` headline.** The pre-registered TVD
advantage survives for Folded-dominant truths, is weaker and coefficient-dependent for
PUF1-dominant truths, and does not survive against `slow-N` for balanced truths. The result is not
explained by a simple truth-independent pull toward Folded, but it is family-dependent and therefore
cannot identify `fast` as the generally robust MoPrP pivot.

The sweep now covers three truth families. `folded_dominant` retains the original ten-point grid;
because the available JAX installation was CPU-only, the pre-registered runtime fallback was used
for `puf1_dominant` and `balanced`: six points spanning 0.015--0.400. Every cell uses all six
available starts (uniform plus three deterministic Dirichlet and two present-state decoy-saturated starts), and
selection remains lowest objective without access to truth. The refreshed CSV has 396 rows.

### 16.1 TVD gain over the frame-uniform null

The null population is exactly `(0.784, 0.144, 0.020, 0.022, 0.030, 0)` on full support. TVD gain is
`TVD(null, truth) - TVD(fit, truth)`, so positive is better than doing nothing. Mean diagonal gains
were:

| truth family | coefficients | legacy | fast | slow-N | diagonal winner |
|---|---|---:|---:|---:|---|
| Folded-dominant | published | 0.1145 | **0.1275** | 0.1141 | fast |
| Folded-dominant | locked | 0.1043 | **0.1285** | 0.1124 | fast |
| PUF1-dominant | published | 0.6451 | **0.6723** | 0.6538 | fast |
| PUF1-dominant | locked | 0.6545 | 0.6579 | **0.6701** | slow-N |
| balanced | published | 0.2895 | 0.2822 | **0.3302** | slow-N |
| balanced | locked | 0.2723 | 0.2954 | **0.3225** | slow-N |

Thus `fast` beats `legacy` by 0.0130/0.0242 TVD in the two Folded controls. In PUF1-dominant it
beats `legacy` by 0.0272 at published coefficients but only 0.0034 at the lock and loses to `slow-N`
there by 0.0122. In balanced it loses to `slow-N` by 0.0480/0.0270; even the `fast`--`legacy` sign
changes with coefficients (-0.0073 published, +0.0231 locked). This fails the requirement that the
`fast` diagonal advantage persist in both non-Folded families.

### 16.2 Concentration diagnostic

There is no simple, truth-independent Folded-attractor signature. On the matched diagonal, mean
`fast` Folded mass is about 0.857 for Folded-dominant truths, 0.182/0.195 for PUF1-dominant truths,
and 0.440/0.451 for balanced truths (published/locked where two values are shown). It is not
systematically more Folded than both alternatives. Likewise, `fast` has slightly *higher* entropy
than the alternatives in the Folded family (0.405/0.412), but lower entropy in the two non-Folded
families (PUF1 0.427/0.458; balanced 1.039/1.042). The confound is therefore not the proposed crude
mechanism of always concentrating onto Folded. It is a broader truth-family dependence: the pivot
ranking changes when the geometry of the target population changes.

### 16.3 Start-coverage isolation

The required single-start Folded control reproduces the original 180-row CSV byte-for-byte
(SHA-256 `3545d50d548b7dc6a42a820699ec19b05bad19c2d1efd5995b93e9c4ec336f32`). Enabling all starts
changes mean diagonal recovery by +1.48/-1.20/-1.16 pp at published coefficients and
-0.15/-0.34/-0.21 pp at locked coefficients for legacy/fast/slow-N. The corresponding changes in
mean TVD gain are -0.0076/-0.0060/-0.0104 and -0.0105/+0.0026/-0.0069. Objective selection can
therefore move truth recovery in either direction, as expected when truth is not part of the
selection rule, but start coverage alone does not create the Folded-family `fast` lead.

### 16.4 Verification and amendment to §15

- Equality gives TVD 0 and strict recovery 100%.
- The recomputed Folded null recoveries reproduce 70.6/73.3/78.2/82.9/83.3/80.4/73.0% at the
  previously tabulated grid points.
- Every family sums to one, has zero PUF3/unfolded/PUF2-like truth mass, and PUF1-dominant tends to
  all-PUF1 as `minority` tends to zero.
- Batched evaluation of the six independent starts matches the sequential implementation in the
  smoke control to `6.1e-16` TVD, with no selected-start changes.
- Jensen ordering has zero violations and one-frame pivot equality remains exact in the refreshed
  output. The synthetic `uptake_mse` field now scores against its fitted synthetic target, fixing
  the §15.5 reporting defect.

**Explicit §15 amendment:** retain §15's real-data fits, Jacobians, and Folded-dominant synthetic
result as measurements, but withdraw the statements that their shared ordering establishes general
positive pivot discrimination or that `fast` is the robust MoPrP pivot. The supported claim is only
conditional: **`fast` is the best diagonal fitter for these Folded-dominated synthetic ensembles.**
Balanced truths prefer `slow-N`, and PUF1-dominant truths do not give a coefficient-robust `fast`
winner. The deferred §15.5 resampling item remains out of scope and cannot restore the withdrawn
population-family generalisation without a separate test.

### 16.5 Truth-weighted log-PF spread regression

**Falsification outcome (2026-08-07): the spread-only mechanism is rejected.** Before evaluating
the regression, family was pre-registered as meaningfully explanatory if adding it to M1 produced
either partial R² at least 0.02 or ΔAIC = AIC(M1) - AIC(M2) at least 2; confirmation additionally
required overlap of the family spread ranges. M1 was
`TVD gain ~ Var_w + pivot + Var_w:pivot`, and M2 added population family. M1 explains only 0.053
of the pooled diagonal variance, while M2 explains 0.783. Family's partial R² is **0.771** and
ΔAIC is **190.6**, exceeding both thresholds by a wide margin. Population family is therefore not
merely a proxy for truth-weighted spread, and the proposed scalar-spread account is withdrawn.
The §16 family dependence remains unexplained.

The comparison is not defeated by separation: the observed Folded/balanced, PUF1/balanced, and
Folded/PUF1 range-overlap widths are 1.097, 1.027, and 1.105 Var_w units. The quadratic diagonal
curves do cross, with `fast = slow-N` at Var_w = 0.429, but a 2,000-replicate bootstrap resampling
the 44 generated truths as clusters gives a broad 95% interval of 0.217--0.802 (1,915 valid
crossings). Since the spread-only model fails its falsification test, this is a descriptive
crossover, not a pre-fit pivot rule.

The exact locked-coefficient recomputation gives grid means 0.249 (Folded-dominant), 0.451
(PUF1-dominant), and 0.699 (balanced reduced grid). Equal-thirds balanced is 0.764/0.787/0.765 for
legacy/fast/slow-N. Thus the quoted 0.265/0.478 low- and middle-spread figures are not reproduced by
the specified uniform-within-state calculation on the refreshed grid, although their ordering is;
the quoted 0.764 equal-thirds value is reproduced. Pooling published and locked settings gives grid
means 0.547, 0.967, and 1.471 because coefficient magnitude changes Var_w quadratically.

The independent mismatch prediction is present but much weaker than §13's ordering result.
For all 264 off-diagonal rows, regressing `TVD(off diagonal) - TVD(matched diagonal)` on
`exp(Var_w/2) - 1` gives Spearman ρ = 0.420 and Pearson r = 0.395. Restricting to the 88 directed
legacy/fast mismatches gives ρ = 0.557 and r = 0.522. Both slopes are positive, but neither result
supports a magnitude calibration or rescues the rejected family-free account.

MoPrP's real NMR population is itself Folded-dominant. At locked coefficients its Var_w is
0.1493/0.1537/0.1494 for legacy/fast/slow-N, below the point estimate and below the lower bootstrap
limit of the descriptive crossover. Together with §16's direct Folded-family sweep, this means
`fast` remains the best-supported pivot **for this system**, with margin on the measured spread
axis; it does not make `fast` generally robust. The near-degenerate low-spread location also gives
a common physical setting for §14's weak real-data identifiability and §16's family dependence,
but the failed M1/M2 comparison prevents claiming that spread alone explains both.

Four qualifications amend the preceding §16 account. First, the Folded-concentration hypothesis is
positively refuted: `fast` does not systematically return more Folded or lower-entropy solutions
irrespective of truth. Second, the real MoPrP population is Folded-dominant, so the conditional
`fast` result applies directly to this system. Third, `fast` is the most consistent diagonal fitter:
combining coefficient settings within each family, its TVD standard deviation is 0.010--0.013,
versus 0.007--0.053 for legacy (legacy is slightly steadier only in PUF1-dominant). Fourth, the
diagonal TVDs have median 0.037 and IQR 0.021--0.054, inside or near the ISO 0.02--0.06 floor band;
the regression is resolving effects of instrument-floor scale.

Verification preserved all 396 joins: 132 diagonal and 264 off-diagonal rows, with no drops.
One-frame Var_w is exactly zero and doubling both coefficients gives exactly four times Var_w.
A named-state point mass is not a one-frame point mass under the required uniform-within-state map:
all Folded truth still has within-Folded Var_w = 0.408 at published coefficients. This distinction
corrects the proposed single-state zero check without changing the analysis. The effective sample
size is about 44 generated truths rather than 132 pivot rows; coefficient setting moves observations
along the x-axis by construction and is not a clean nuisance factor. Results are generated by
`moprp_population_spread_regression.py`; no new population fits were run.

## 17. Phase 3 HDX noise-model decision (2026-08-14)

Phase 3 of `hdx_noise_model_implementation_handoff.md` is complete. The PF reference was refitted
unweighted from `moprp.dexp` using the shipped `moprp.kint`, 50 deterministic starts (seed 1729),
zero harmonic strength, and six disconnected peptide-overlap regions. Every outer fold repeated a
50-start PF refit. Five blocked-time folds and seven overlap-safe peptide folds were scored by
conditional Gaussian predictive density; the other seven peptide holdouts, including isolated
peptide 1, cannot be predicted without an unregistered latent-PF prior and are recorded as a
limitation rather than silently scored.

The accepted covariance architecture is **homoscedastic diagonal acquisition noise plus a
peptide-persistent term**, with strict EX2 frozen as the mean. The peptide term improved paired
overlap-safe peptide-fold NLL by 0.307 per held-out cell (SE 0.102) and made no material change to
blocked-time prediction. PF-propagated covariance, the timepoint-common term, fitted
heteroscedastic shape, and the fixed `moprp.weights` diagonal failed their component-deletion
comparisons and are rejected. ANM remains rejected by Phase 2; no mixture is promoted because mode
weights were not calibrated and multistart hit counts are not probabilities.

The frozen fit has `sigma_exp = 0.0586156` and `tau_peptide = 0.0143907`; all other scientific
covariance scales are zero. Its leading variance fraction is 0.00774, effective rank 209.36, and
condition number 1.66, resolving the shipped covariance's 0.906 leading-direction pathology.
Primary and compatibility artifacts, fold scores, PF-reference comparison, spectral diagnostics,
and the full manifest are under `fitting/jaxENT/_moprp_sigma_noise_model/`. Phase 4 may now consume
the frozen mean and Cholesky; Phase 3.5 remains gated on a separate Phase 2.5 finite-gating study.

## 18. Synthetic gradient study — Sigma-MSE vs eye-MSE on the oracle geometry (2026-08-14)

Run before any Phase 4 downstream substitution, as a cheaper gradient-level test of what the
frozen Phase 3 covariance does to optimisation. Runner:
`moprp_sigma_gradient_synthetic.py`; results in
`_moprp_sigma_gradient_synthetic/gradient_synthetic_results.json`.

Geometry is the §15 population oracle: five present states (Folded, PUF1, PUF2, PUF3, unfolded)
with uniform-within-state frame weights on AF2_MSAss, `fast` pivot at published coefficients
(bc 0.35, bh 2.0). Synthetic uptake was generated at the NMR target populations; noise was drawn
from the frozen Phase 3 Cholesky (time-major vector order verified against the stored
covariance). Four losses on the same residual: eye-MSE (control); Sigma-MSE with the Phase 3
trace-normalised collapsed precision (the exact per-timepoint math of
`hdx_uptake_sigma_MSE_loss`); Sigma-MSE with the *previous shipped* `Sigma_inv`
(`data/_MoPrP_covariance_matrices/Sigma.npz`, trace-normalised as the loader pathway would); and
the joint Gaussian with the full frozen 210×210 Σ through its Cholesky.

Findings:

1. **All four gradients are correct.** Analytic gradients match central finite differences to
   relative error ≤ 8e-9 at the truth, at uniform logits, and at a decoy-saturated corner —
   including the shipped arm. The shipped matrix's defect was never a broken gradient; it is a
   broken metric.
2. **The new Σ barely changes the gradient direction.** Cosine against the eye-MSE gradient at
   the uniform start is 1 − 9e-9 (collapsed) and 1 − 2e-5 (joint); norms differ only by scale
   (~0.5× and ~120×), which a learning rate absorbs. This is the expected consequence of the
   accepted Σ's near-isotropy (condition number 1.66, σ_exp-dominated): whitening barely
   rotates. Any Phase 4 Σ-vs-MSE difference is attributable to scale/conditioning, not
   direction.
3. **The shipped precision weakens and rotates the gradient.** Its uniform-start gradient is
   ~6× smaller than MSE and rotated (cosine 0.985 vs MSE; descent alignment toward the truth
   0.32 vs ~0.40 for every other arm). Its Gauss-Newton spectrum suppresses the third
   population direction by ~3× (3.9e-4 vs ~1.1e-3), consistent with the §2.3
   90.6%-one-direction pathology down-weighting exactly the directions the residual needs.
4. **No loss is biased at the optimum.** Clean gradients at the truth are numerically zero
   (~1e-18 to 1e-20) for all arms; over 64 noise draws the mean gradient stays inside its SD.
5. **The downstream consequence appears under noise.** Worst neutral-start recovery in the
   noisy regime: MSE 95.0%, new Sigma-MSE 95.1%, joint 95.1%, shipped 87.4%. Noiseless, all
   four plateau together at ~94% — that shared shortfall against the §15 99% gate tracks the
   observable's near-null fourth singular direction (~2e-5 relative to the leading ~0.6), an
   optimisation plateau, not a loss property.
6. **Identifiability is metric-invariant for the new Σ.** The whitened Gauss-Newton spectra of
   MSE, new Sigma-MSE, and joint share the same shape — two strong directions, a weak third
   (~1e-3), a near-dead fourth, plus the exact softmax-shift null. Σ-whitening opens no
   population direction that MSE cannot see; because τ_peptide contributes only ~6% of the
   variance, this is a property of the accepted Σ, not of Σ-weighting in general.

Pre-Phase-4 headline established synthetically: new-Σ ≈ MSE (no harm, negligible rotation, sane
conditioning), while the shipped precision actively degrades recovery under realistic noise.

## 19. Phase 2.5 finite-gating detectability — Phase 3.5 skipped (2026-08-14)

Phase 2.5 of `hdx_noise_model_implementation_handoff.md` (§3 Phase 2.5, §6) is complete. It is a
simulation-only study at MoPrP's real geometry (14 peptides, 15 timepoints, 76 peptide-covered
residues) asking the single registered question: at this timepoint grid, N = 210, and the fitted
noise level, what gating speed is slow enough to be distinguished from strict EX2? Surfaces were
generated from the finite-gating LL backend across the registered ladder
`c ∈ {1e4, 1e2, 10, 3, 1, 0.3}` with `gamma = c · median(k_int)`, plus a strict-EX2 reverse-null
row, 25 replicates per rung, scored by blocked three-timepoint conditional Gaussian held-out NLL.
Truth noise scales were the frozen Phase 3 values (`sigma_exp = 0.0586156`,
`tau_peptide = 0.0143907`); the primary arms' covariance is gamma-independent under the accepted
Phase 3 architecture, so only the mean depends on gamma.

**The detectability floor is `c = 3`** — finite gating is distinguishable only once
`gamma <= 3 · median(k_int)`, and the floor is strictly bracketed as `[3, 10)`. The `c = 10` rung's
mean advantage (0.0090) clears the null while its 5th percentile (−0.0023) does not, and the
`c = 3` detection is marginal (q05 0.00666 against null q95 0.003795, about 1.75× separation).
The floor is quoted dimensionlessly because its absolute value is rate-source dependent: under the
validated HDXrate/PDLA rates used by all of Phase 2 and 2.5 it is `gamma = 2.4599 min^-1`
(`gamma^-1 = 0.41 min`), while under the `moprp_shipped` rates that Phase 3 froze the same rung is
`gamma = 1.0696 min^-1`.

**Decision: Phase 3.5 is skipped.** Because `p_open = 1/PF << 1`, `gamma ≈ k_close`, so detection
requires `k_close <~ 3 k_int` — whereas EX2 is defined by `k_close >> k_int`. The detectable window
therefore lies entirely in the EX1/EXX crossover rather than the EX2 regime. On the stated premise
that native-state closing rates for a folded protein exceed 0.04 s^-1 by orders of magnitude (a
domain premise recorded as such, not derived here), no physically plausible MoPrP gating rate falls
inside the detectable window, and the §3 Phase 2.5 kill rule applies exactly as it did for the ANM
arm. This is the pre-registered expected outcome: it is positive evidence that MoPrP centroid
uptake is EX2 at this resolution and it retires the EX2 caveat rather than failing a stage. The
peptide-1 envelope spot check (handoff §8: EX2 favoured by 29×, 36× and 2.1× SSE, with
`observed < EX2 < EX1` widths) points the same way independently.

Controls all pass. The K1 fast-limit control was decoupled from the ladder and pinned at `c = 1e6`,
giving max `|LL − EX2| = 3.2256e-07` against a 1e-6 tolerance; it is deliberately not
`GAMMA_LADDER[0]`, because `c = 1e4` carries real `O(k_int/gamma)` physics (3.2248e-05) and would
fail a fast-limit assertion for physical rather than numerical reasons. The propagator's functional
form is confirmed by clean first-order scaling — consecutive decade ratios 9.998393 and 9.999417
against a target of 10. On the reverse null the minimum fitted `c` over 25 EX2-generated replicates
is 9.33, so no replicate entered the detectable zone and the null cannot manufacture a false
positive; fitted gamma above the floor is a lower bound rather than an estimate, because the
log-gamma direction is flat there. A standing positive control was added to the summary: where
gamma is identifiable it is recovered essentially unbiased (median fitted/true `c` = 0.953, 1.007
and 1.013 at `c` = 3, 1 and 0.3).

The mandatory confounding check found the strongest partner to be
`corr(log gamma, sigma_exp) = −0.5844` at `c = 1`, with `kappa`, `tau_time` and `tau_peptide` all
at `|corr| <= 0.23`. This confirms the pre-registered constraint that a Phase 3.5 K2 would have had
to run with Sigma frozen and that a fitted-Sigma K2 is inadmissible; it also deviates from §2.5
step 4's expectation that gamma would trade against `kappa`, `tau_z` or `tau_time`.

Limitations recorded rather than smoothed over: Phase 2 and 2.5 ran on `hdxrate_pdla_validated`
rates while Phase 3 froze its covariance under `moprp_shipped`, so the truth noise scales are
imported across rate arms — the dimensionless floor and the kill argument are invariant to this,
but the absolute gamma is not. A guard rejecting overflowing optimiser trial points was added
mid-study, after registration; it is a numerical guard rather than an objective change, but it can
only affect the K2 arm and is asymmetric by construction. Convergence was K0 175/175, K2 172/175,
confounding K2 75/75, with all three K2 failures in the flat undetected region and none in a
detected rung. Artifacts, controls and full provenance are under
`fitting/jaxENT/_moprp_sigma_identifiability/gamma_full/`; the decision is also recorded in that
directory's `phase2_decisions.md`.

## 20. Phase 4 frozen-Sigma substitution — results available, freeze withheld (2026-08-14)

Phase 4 was implemented before the gated Phase 5 exPfact diagnostic, using the AF2-MSAss full
500-frame ensemble and the `moprp_shipped` intrinsic-rate arm under which Phase 3 fitted the
covariance. Frame weights remain uniform within each of the five present structural states, so the
optimised reweighting variables are state populations rather than 500 independent frame weights.
Peptide 1 is removed from the uptake objective by the registered time-major strided marginal:
indices `i % 14 != 0`, reducing the frozen covariance from 210×210 to 195×195 before a fresh
Cholesky factorisation. This is not the superficially valid but incorrect contiguous `[15:,15:]`
slice.

Three uptake-residual metrics were compared. `eye_mse` is the untouched mean squared error.
`shipped_sigma` applies the shipped 14×14 `Sigma_inv` independently at each timepoint after removing
peptide 1 and trace-normalising the retained 13×13 precision. `frozen_joint` applies the full
195×195 frozen Cholesky in time-major order. The joint log determinant is constant with respect to
population/BV optimisation and is removed from the optimisation step. Every arm is divided by its
own uniform-population baseline, so the KL coefficient retains the same relative scale. The frozen
Cholesky is carried through `stop_gradient` and cannot become candidate-dependent.

### 20.1 Registered reweighting-only sweep at locked BV coefficients

The registered sweep crossed three pivots (`legacy`, `fast`, `slow-N`), two coefficient settings
(`published`, `constrained_optimum`), and three metrics. Averaging the six pivot/coefficient cells
gives:

| Arm | Mean recovery | Mean decoy mass |
|---|---:|---:|
| eye-MSE | 36.229% | 0.39454 |
| shipped Sigma | 47.390% | 0.06916 |
| frozen joint Sigma | 39.696% | 0.36753 |

Those averages conceal a dominant pivot/coefficient interaction:

| Pivot / coefficients | eye-MSE recovery, decoy | shipped Sigma recovery, decoy | frozen joint Sigma recovery, decoy |
|---|---:|---:|---:|
| legacy / published | 0.004%, 0.99999 | 50.855%, 0.39768 | 0.005%, 0.99999 |
| legacy / constrained | 50.885%, 0.00526 | 31.657%, 0.00016 | 59.151%, 0.00055 |
| fast / published | 16.383%, 0.21620 | 66.094%, 0.01643 | 16.184%, 0.17024 |
| fast / constrained | 92.353%, 0.00005 | 38.976%, 0.00044 | 95.054%, 0.00005 |
| slow-N / published | 0.762%, 0.99472 | 89.788%, 0.00009 | 5.689%, 0.92702 |
| slow-N / constrained | 56.985%, 0.15104 | 6.971%, 0.00014 | 62.094%, 0.10733 |

The frozen joint metric remains close to eye-MSE, as predicted by §18, but the aggregate claim that
the shipped precision simply underperforms is false at locked coefficients. Its apparent advantage
under published BV is not stable: it reverses under the constrained coefficients. The most relevant
previously supported specification, `fast` with constrained BV, gives 92.35% recovery for eye-MSE,
95.05% for frozen joint Sigma, and 38.98% for shipped Sigma. The shipped matrix therefore changes
the interaction between mean-model error and population reweighting rather than contributing a
generally superior population metric. Recovery is measured against the independent NMR population,
not against a population that generated the real uptake, so a lower weighted uptake objective is
not itself evidence of better recovery.

### 20.2 Follow-up diagnostic with BV fitting enabled

After reviewing the locked-BV result, a matched diagnostic jointly optimised the state populations
and non-negative `(Bc, Bh)` in every pivot/metric cell. It used both published and constrained BV
initialisations, the neutral/random/adversarial population starts, 12 starts per cell, and 1,500
Adam steps. Each metric's normaliser was fixed at its published-BV uniform-population value; it was
not recomputed at candidate BV coefficients, which would create a circular escape route through the
denominator. Since coefficients are free, this comparison has three pivot cells per arm rather than
the registered six locked-setting cells.

| Arm | Mean recovery | Mean decoy mass |
|---|---:|---:|
| eye-MSE | 60.371% | 0.17277 |
| shipped Sigma | 23.980% | 0.52702 |
| frozen joint Sigma | 59.799% | 0.17670 |

| Pivot | eye-MSE recovery, decoy | shipped Sigma recovery, decoy | frozen joint Sigma recovery, decoy |
|---|---:|---:|---:|
| legacy | 48.763%, 0.11721 | 56.986%, 0.31217 | 49.059%, 0.12653 |
| fast | 94.702%, 0.00004 | 14.911%, 0.26903 | 90.812%, 0.00004 |
| slow-N | 37.648%, 0.40107 | 0.043%, 0.99986 | 39.528%, 0.40353 |

This diagnostic restores the §18 ordering: frozen joint Sigma is effectively comparable to
eye-MSE, while the shipped precision is substantially worse and selects an almost pure PUF3 decoy
under `slow-N`. It also reproduces the known BV degeneracy: nearly every selected solution drives
`Bh` to approximately zero. The result is therefore useful for diagnosing the covariance/BV
interaction but is not a physically credible new BV calibration and does not supersede the locked
coefficient registration.

### 20.3 What the accepted frozen Sigma contains

The accepted Phase 3 covariance requires a clarification. Phase 3 refitted a 76-residue PF centre,
tested nonlinear EX2 propagation of PF uncertainty with 5,000 draws, and used the PF-derived mean
uptake to construct candidate geometry. Cross-validation then rejected the propagated-PF component
(`tau_z = 0`). The frozen `peptide_only` covariance is therefore

`Sigma = sigma_exp^2 I + tau_peptide^2 Z_p Z_p^T + 1e-10 I`,

with `sigma_exp = 0.0586156`, `tau_peptide = 0.0143907`, and
`Z_p[(t,p),p] = mean_uptake[p,t]`. PFs affect the accepted Sigma indirectly through this mean-based
peptide-persistent loading, but the final covariance does not contain the rejected nonlinear PF
uncertainty covariance. Phase 4 itself consumes only predicted uptake, observed uptake, and the
frozen Cholesky; it does not add PF observations to the downstream likelihood.

### 20.4 Verification status and decision

Two registered covariance checks pass. Zeroing every cross-time block and refactorising gives a
joint NLL exactly equal to the sum of the per-timepoint block NLLs (absolute difference 0.0 against
a 1e-10 tolerance). The frozen marginal Cholesky is bitwise identical before and after optimisation,
with SHA-256 `07e72032624c77644077929a03f85f0189f9c1c493ab038f79a66944dfbd374b`, and is explicitly
carried under `stop_gradient`. The focused sigma/loss-registry suite passed 58 tests; Ruff and
`git diff --check` pass.

The registered default-path byte-identity gate does **not** pass. The pre-existing default artifact
had SHA-256 `17971aa7c77a7f6ff5abc3ed9833d8ab39e3879137d49fca23a32ceb351e5323`, while two literal
no-flag reruns of the otherwise unchanged default runner produced different hashes,
`da2f7e45629869d33a0db634869977ad8c95c6e74b658e1b9ff8f31716f7ea9a` and
`663531fa8763e91e56aa465bdd80940d63d2243be24f315e9328c529f4f10733`. The `metric=None` uptake-loss
body is retained as a separate textually unchanged branch, so this exposes pre-existing
cross-process byte nondeterminism in the full JAX optimisation/output path rather than a demonstrated
metric-hook drift. Nevertheless, the registered gate is byte-level and cannot be declared passed.

**Status: Phase 4 results exist but the Phase 4 manifest is not frozen.** The scientific readout is
new-Sigma approximately equal to eye-MSE, with no stable new-Sigma advantage, while shipped-Sigma
behaviour is strongly coupled to BV/pivot misspecification and degrades sharply once BV is fitted.
The failed byte-reproducibility gate must be replaced by an agreed deterministic or numerical
equivalence criterion, or made reproducible in the execution environment, before Phase 4 can be
frozen. Phase 5 remains gated and must not be run with `--primary-results-frozen` yet. Registered
artifacts are under `fitting/jaxENT/_moprp_sigma_phase4/`; the BV-enabled diagnostic is under
`fitting/jaxENT/_moprp_sigma_phase4_bv/`.
