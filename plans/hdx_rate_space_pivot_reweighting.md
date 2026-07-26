# Handoff: test arithmetic rate-space pivot (`k̄_after`) as a *live* reweighting forward-model

**Status:** deferred, opened 2026-07-26. Not started; no guardrails or code written yet.
**Relationship:** spins out of `hdx_effective_rate_variance_physics.md` §10 (Stage 4,
closed) and `hdx_redundancy_timepoint_error_construction.md`. Standalone, not a section
of either.

## Who this is for

Someone deciding whether to reopen BV mean-model scope (currently forbidden by the
D-only guardrails) to test an alternative frame-averaging pivot inside actual
reweighting fits.

## The question

Does the ensemble-averaging pivot choice change conformational recovery when used as
the live pivot during fitting, vs. only ever having been compared as a static target
with frame weights frozen?

## High-entropy findings (the point of this doc)

1. **Geometric mean of per-frame rates ≡ current production pivot, exactly.**
   `k_f = k_int·exp(-z_f)`, `k_int` frame-independent ⇒
   `GM_w(k) = exp(Σw_f ln k_f) = k_int·exp(-z̄) = k̄_first`. Not a new proposal — it's
   what `average_first=True` already computes, and what every closed MoPrP result
   already used. No code change needed for *that* idea.
2. **The real alternative, `k̄_after = Σw_f k_f` (true arithmetic rate mean), has never
   been tested in a live fit.** Stage 4 (`hdx_effective_rate_variance_physics.md` §10)
   compared `k̄_after` vs `k̄_first` but explicitly excluded reweighting (§10.3: "no BV
   tuning, no reweighting, no NMR inputs"). Every existing pivot comparison used frozen
   frame weights. This is the actual gap Stage 4's closure does not cover.
3. **`k̄_after ≥ k̄_first` always — forced by AM≥GM, not just empirical.** Gap ≈
   `½·k̄·Var_f(z)` (matches the existing `gaussian_mean_rate` second-order term already
   in `investigate_uptake_rate_covariance.py`).
4. **This needs a real architecture change, not a loss-side opt-in** (unlike the Fisher
   timepoint-weighting precedent). `average_first: bool` (declared identically in
   `HDX/forward.py`, `SAXS/forward.py`, `XLMS/forward.py`) is a binary hook in
   `Simulation.forward_pure` (`models/core.py:298-305`) distinguishing "average inputs"
   (features, pre-nonlinearity → `k̄_first`) from "average outputs" (uptake,
   post-nonlinearity → frame_mixture). Arithmetic rate-averaging needs to average the
   *intermediate* quantity `k = k_int·exp(-z)`, which has no averaging hook today —
   extending this touches the `ForwardPass` contract across every model, not just HDX.
5. **Landmine for a careless implementation:** `if getattr(fp, "average_first", True):`
   (`models/core.py:299`) is a truthy check. Setting `average_first` to any non-bool
   truthy value without rewriting that branch silently falls into the existing
   "average inputs" path instead of erroring. A real implementation needs a typed mode
   replacing the bool, not an overload.
6. **Scope collision, not just an engineering-effort call.** The pivot is part of "the
   fixed mean curve" (sibling handoff §2.5) — i.e. this is BV mean-model work, which
   the D-only checkpoint (memory: `hdx-d-only-checkpoint-bv-mean-model`) already
   flagged as out of scope, requiring "an explicit decision to reopen BV mean-model
   work as its own investigation," since all three closed D-only tracks hit a wall
   attributed to BV mean-model error.

## What "doing this later" concretely means

- Extend `ForwardPass`'s `average_first: bool` into a typed frame-averaging mode
  (e.g. `log_pf` / `uptake` / `rate`), threaded through `Simulation.forward_pure` and
  every model declaring the attribute.
- Add the `rate` branch: per-frame `k_f = k_int·exp(-z_f)`, arithmetic-average across
  frames with `frame_weights`, then `u(t) = 1 - exp(-k̄_after·t)`.
- Wire it into an actual MaxEnt/frame-weight reweighting fit (not a frozen-weight
  diagnostic) on a system with a known answer — TeaA/ISO 40:60 is the natural choice
  (per `hdx_redundancy_timepoint_error_construction.md`), since it's the one system
  where recovery is directly checkable.
- Compare recovery under `k̄_after`-as-pivot vs. the `k̄_first` (current) baseline —
  no existing artifact substitutes for this comparison.

## Non-goals for now

- Not touching BV coefficients (`bc`, `bh`) or contact features — pivot only.
- Not resurrecting D/R residue-covariance recovery — orthogonal, stays parked.
