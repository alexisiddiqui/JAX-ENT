# Investigation: frame-averaging regime as a live reweighting forward-model

**Status:** scoped, design agreed 2026-08-04. Supersedes the deferred handoff of 2026-07-26
(retained as §9). No code written yet.
**Relationship:** spins out of `hdx_effective_rate_variance_physics.md` §10 (Stage 4, closed)
and `hdx_redundancy_timepoint_error_construction.md`. Standalone.

## 1. What changed since the original handoff

The original handoff proposed testing the arithmetic rate pivot `k̄_after` as a live pivot in a
reweighting fit on ISO 40:60. **That experiment cannot work as scoped.** The ISO target data was
itself manufactured with the `k̄_first` pivot, so the comparison is circular.

Evidence — `_Bradshaw/Reproducibility_pack_v2/data/artificial_HDX_data/create_mixed_target_data.py`,
invoked per its `README` as `-n 16552 731 -w 0.6 0.4`:

```python
contacts[:,endframes[i]:endframes[j]] *= weight/framelist[j]   # per-state frame weights
...
avelnpi = np.sum(Bc*contacts + Bh*hbonds, axis=1)              # weighted mean log-PF, POOLED
byres_deutfrac = 1.0 - np.exp(num / np.exp(denom))             # one rate from that mean
```

The 60:40 "mixture" is applied as frame weights *inside the log-PF average*, pooled over both
states, and a single exponential taken from the result. That is exactly `k̄_first`. There is no
uptake mixture anywhere in the generation path. Fitting this target with `k̄_after` would show
`k̄_first` winning by construction.

Consequence: the investigation is restructured around **regenerating targets** under known,
differing semantics from the same 60:40 populations.

## 2. Settled by physics — not swept

**Peptide aggregation is linear in uptake space.** Deuterium is additive over amides; the peptide
observable's centroid is `Σ_i u_i(t)`. This is mass conservation, not a modelling convention. Any
peptide-level single effective rate is wrong unless all amides share a rate — never true, since
`k_int` alone varies ~2 orders of magnitude with sequence.

Production already complies (`map_residue_uptake_to_peptides`, `pf_variance.py:300`), and so does
the Bradshaw generator (`np.nanmean(byres_deutfrac, axis=1)`, skipping each segment's first
residue). `map_frame_log_pf_to_peptides` (`pf_variance.py:325`) remains valid as a covariance
*coordinate* but is not a candidate forward model.

**The peptide axis is therefore excluded from the sweep.** Only the frame axis is open.

## 3. The physical argument for rate space

`P = k_cl/k_op` is a **ratio of rates** — dimensionless, an equilibrium constant, carrying no
frequency. It is a reciprocal occupancy:

```
f_open = 1/(1+P) ≈ 1/P     (P ≫ 1)
k_obs  = k_int · f_open      (EX2)
```

The quantity linear in ensemble population is `f_open = 1/P`. Ensemble-averaging a population means
arithmetic-averaging it:

```
f_open^ens = E_w[e^{-z_f}]   ⇒   k̄ = k_int·E_w[e^{-z_f}] = k̄_after
```

i.e. arithmetic mean of *inverse* PF = harmonic mean of PF = the `harmonic_mean_rate` already in
`investigate_iso_mean_rate_uptake.py:65`. **The linear coordinate is `1/PF`, not `PF` and not
`log PF`.** Averaging `log P` averages a free energy — sensible for a potential, not for an
occupancy. `exp(E[z])` is not the limit of any kinetic model.

This derivation needs no timescale argument; it follows from `f_open` being a population, and is
weighted above the Jensen-bracket reasoning below.

**Assumption to keep explicit:** it presumes BV's per-frame `z_f` means "log reciprocal open
fraction *given* frame `f`'s local environment" — the reading consistent with `z = bc·N_c + bh·N_h`
as a continuous surrogate free energy rather than an open/closed indicator. The rate-space case
rests on this.

**Supporting bracket argument.** `1-e^{-kt}` is concave in `k`, so
`E_f[1-e^{-k_f t}] ≤ 1-e^{-E_f[k_f]t}` for all `t`, equality as `t→0`. The two kinetic limits
bracket the truth, and as `t→0`, `u ≈ E[k]·t` in both — the short-time asymptote pins the pivot to
the arithmetic mean rate unconditionally. `k̄_first` does not reproduce it, and its signed error
against exact frame-averaging changes sign with residue and time (per the Stage 4 litmus), i.e. it
wanders outside the admissible band.

## 4. What the frame axis actually is

Not a coordinate choice — a claim about interconversion timescale relative to labelling
(seconds–minutes):

- **fast** — the amide samples the whole frame ensemble within one labelling interval; aggregate in
  **rate** space.
- **slow** — frames are kinetically distinct populations labelling independently, mixed only at the
  detector; aggregate in **uptake** space.

Populations are the recovery *target*, not a physical claim, so ISO admits both and both are
modelled.

**Slow means two-state, not per-frame.** Frames within one basin interconvert fast relative to
labelling; the two clustered states (16552 closed / 731 open) are the only plausibly-slow degree of
freedom. A full 17283-frame uptake mixture would assert every MD frame is a kinetically isolated
species — not a physical model of anything, and 3 orders of magnitude more expensive.

## 5. Target regeneration

New script alongside `create_mixed_target_data.py` (do not modify it). Reuses the existing
`Contacts_*.tmp` / `Hbonds_*.tmp` files and `TeaA_mixed_Intrinsic_rates.dat`. `bc/bh` stay at
0.35/2.0 by construction. Forward-only — no fitting — so cheap.

Three target semantics, identical 60:40 populations:

| target | construction |
|---|---|
| `legacy` | `k̄_first` — current Bradshaw path. Baseline / continuity with all closed results. |
| `fast` | `k̄ = k_int·E_w[e^{-z_f}]` pooled over all frames, then `u = 1-e^{-k̄t}` |
| `slow2` | pool within each state → two rates → per-state uptake → mix 60:40 **in uptake space** |

### EX-regime parameter

`k_op` is not derivable from an equilibrium ensemble; `P` fixes only the ratio. So assert one global
timescale `τ`, per frame:

```
P_f = exp(log_Pf) = 1/f_open
k_obs,f = k_int/(P_f·(1 + τ·k_int))               [full Linderstrøm-Lang]
```

`τ→0` recovers EX2 exactly (`k_obs → k_int/P`); large `τ` drives EX1. Build `τ` in as a parameter
from day one; the marginal cost of using it later is compute, not code.

## 5b. Stage 0 — synthetic population sweep (run first)

Calibration gate. Without it, off-diagonal recovery errors in §6 have no units: a recovered 0.55 vs
true 0.60 is uninterpretable until you know how sharply population is identifiable in the *matched*
case.

**Clusters.** From `_clustering_results/` (RMSD to two references, `extract_OpenClosed_clusters.py`,
threshold 1.0 Å). `-1` (unassigned) **is** the intermediate class — no third reference needed.

| ensemble | total | open | closed | intermediate (`-1`) |
|---|---|---|---|---|
| ISO_TRI (`TeaA_initial_sliced`) | 2225 | 44 (2.0%) | 830 (37.3%) | 1351 (60.7%) |
| ISO_BI (`TeaA_filtered_sliced`) | 874 | 44 (5.0%) | 830 (95.0%) | 0 |

**Design.** Two axes — open population and intermediate population (the confounder); closed takes
the remainder; renormalise. Coarse grid (3×3 or 4×4), not crossed 1-D scans, since the point of
calling intermediate a confounder is that it *interacts* with open recovery and crossed scans
cannot see interaction. Matched fitter to matched target semantics throughout — diagonal only.

Run on **both** frame sets (full and sliced/trimmed), since recovery is computed by clustering
whatever candidate ensemble is used, so cluster populations are well defined on either.

**Report:**
- Recovered vs true as a 2×2 Jacobian. Condition number, and the eigenvector of the sloppy
  direction — tells you *which combination* of open/intermediate is unidentifiable, which two
  separate slopes cannot.
- ESS per cell (`plot_ablation_ess.py` machinery). 44 open frames means putting 40% of the mass
  there is an ~8× tilt (BI) / ~20× (TRI), with ESS on the open state floored near 44 regardless of
  convergence. Hard identifiability limit independent of everything else here.
- Plot against KL-from-prior, not raw population — the prior sits at 5.0% open (BI) / 2.0% (TRI),
  so difficulty tracks tilt magnitude. Include a cell **at** the prior point: zero work, recovery
  must be near-exact, and if it is not something upstream is broken.

**Gate:** if the matched-diagonal Jacobian is badly conditioned, §6's off-diagonals are noise and
Stage 1 should not run. Threshold to be set by user before running.

**This also measures §8 directly.** Under `fast` the observable depends on `w` only through the
scalar `E_w[e^{-z}]` per residue, so open and intermediate trade off whenever some intermediate mass
reproduces that scalar; under `slow2` the constraint is the full uptake vector. Prediction: the
open↔intermediate degeneracy is markedly worse under `fast`. If it holds, this is a headline result,
not just calibration — and cheaper than the matrix that was meant to produce it.

**Caveat:** every cell is self-consistent by construction, so good recovery here is not evidence any
pivot is correct. Same trap as the `legacy` diagonal.

## 6. Experiment matrix — staged

**Stage 1 (v1, `τ=0`): 3 targets × 3 fitters = 9 cells.**
Fitters: `legacy` (`average_first=True`), `fast` (arithmetic rate mean), `slow2` (two-state uptake
mixture). Diagonal = self-consistency, must recover 60:40 near-exactly or something is broken.
**Off-diagonals are the payload** — recovery error attributable purely to a wrong averaging regime.
No existing artifact substitutes for this.

**Stage 2 (EX1-contamination arm): 2–3 `τ` values × 3 fitters, on `fast` and `slow2` targets only.**
Not a competition — all three fitters are misspecified here by construction, since no EX1 forward
model exists in jaxent and none can. Measures robustness bias when data carries EX1 character the
fitter structurally cannot represent.

Stage 2 must be reported separately from Stage 1, or it will be read as a pivot result when it is a
robustness bound on an assumption we cannot check.

## 7. Metrics

MSE alone is insufficient, for a specific geometric reason: **a pivot change is a pure horizontal
translation on log-time** (`k̄_after/k̄_first ≈ exp(½Var_f(z))` is a scalar rate multiplier, and
rescaling `k` in `1-e^{-kt}` shifts the curve along `log t` without touching shape), whereas **a
regime change is a width change** (`slow2` is a sum of two sigmoids, broadening the transition over
`log t`). MSE projects both onto one scalar where they trade off.

Report three things per fit:

1. **Recovery** (primary) — recovered vs true 60:40. Everything else is diagnostic for *why*.
2. **Signed bias** — mean residual per residue and per timepoint, sign retained. AM≥GM guarantees a
   *directional* offset between pivots; an unsigned statistic discards the only theoretically
   guaranteed signal.
3. **Position / width** — effective log-rate (position) and transition width per curve. Position
   error ⇒ pivot mis-scaling; width error ⇒ regime misspecification. Reuse
   `select_shared_rate_model` / `fit_shared_rate_mixture` (`jaxent/src/analysis/hdx_rate_mixture.py:314`)
   — CV over component count and shrinkage already; component count ≥2 winning *is* a width signal.
   Do not build a parallel sigmoid-fitter.

**Pre-committed power caveat.** ISO has 5 timepoints (`0.167, 1.0, 10.0, 60.0, 120.0` min, per the
generator invocation). Five points over ~3 decades supports roughly two shape DOF, so the width
metric is low-power — suggestive at best. Position and bias will be well-determined. Same resolution
wall as `hdx_redundancy_timepoint_error_construction.md`. State this up front rather than
discovering it post hoc.

**Loss:** plain eye/MSE, not the Sigma variants. A covariance-weighted loss redefines per-peptide
bias and fights metric 2; and the covariance path is a moving surface (`loader.py:203` now
trace-normalises to `trace(W)==n`; `hdx_uptake_MSE_loss` correctly aliased to
`hdx_uptake_eye_MSE_loss` at `examples/common/losses.py:31`) not worth entangling with a pivot
experiment.

## 7b. Weight-smoothness hypothesis (exploratory, secondary)

**Hypothesis (user, 2026-08-04, not established):** the fast/slow regime can be read off
empirically from how smooth recovered frame weights are with respect to local structural
similarity — slow smooth, fast jagged.

### Instrument: the Boltzmann coordinate, not a similarity graph

`plans/hdx_boltzmann_frame_weight_consistency_loss.md` derives a scalar per-frame free-energy
proxy `G_i = Σ_r log_Pf_i(r) = bc·total_heavy_i + bh·total_acceptor_i` with Boltzmann expectation
`ln w_i - ln w_j ≈ G_i - G_j`. So the statistic is a 1-D residual:

```
x_i = log w_i - α·G_i        statistic = Var(x)
```

— exactly that doc's penalty variable, with its exact closed form `(2n/(n-1))·Var(x)` for the
uniform-edge pairwise sum. Preferred over graph Moran's I: scalar coordinate not a graph, `O(n)`
not `O(n²)`, and `G` is an **externally defined reference** rather than derived from the fit, so
markedly less circular.

Fit `α` freely rather than fixing `α=1` — the source doc absorbs scale into its regularisation
hyperparameter, and a free scale separates "wrong scale" from "wrong shape". Only the latter is the
misspecification signal.

**Hard constraint: do NOT enable `boltzmann_frame_consistency` as a loss during the sweep.** It
would directly optimise the quantity being measured. Compute `Var(log w - αG)` post hoc from
recovered weights.

Two reasons diagnostic use is safer than loss use here: (a) the source doc flags `G_i → E_i` as a
modelling assumption whose wrong sign actively misdirects weight — as a readout we never assert the
prior, so the sign risk does not transfer; (b) its accepted `bc`/`bh` gradient-leak tradeoff is
void because we freeze `bc/bh`, making `G` a static per-frame vector for the whole sweep.

### Mechanism and directional prediction

Under MaxEnt, `log w` is linear in the constraint features: `log w_f ∝ -Σ_c λ_c g_c(f)`. `G` is
linear in `z`. The fitters tilt on:

- **fast** — `g = e^{-z}`, strongly convex in `z` ⇒ poorly approximated by any linear-in-`G`
  relation ⇒ large residual ("jagged")
- **slow2** — `g = u = 1-e^{-k_int·t·e^{-z}}`, bounded and saturating, closer to linear in `z` over
  the sampled range ⇒ small residual ("smooth")

This matches the hypothesis and gives it a mechanism. **But the driver is the fitter's tilt
geometry, not the generating physics** — fit a fast pivot to slow-generated data and weights are
still jagged. Misspecification enters only second-order, via larger `λ` needed to match data the
model cannot naturally produce.

### Therefore: row-relative only

```
Var(x | fitter F, target T)   vs   Var(x | fitter F, target F)
```

Never compared across fitters. Requirements:

1. **Matched regularisation and matched fit quality**, or the statistic measures `λ`.
2. **Within-cluster only.** Use the existing RMSD-to-reference clusters —
   `_clustering_results/cluster_assignments_ISO_TRI.csv` (open/intermediate/closed) from
   `extract_OpenClosed_clusters.py`, not k-means. The between-cluster weight jump is the 60:40
   recovery itself and is expected in *both* regimes; on a full-ensemble statistic it swamps
   everything. Note `cluster_by_rmsd` assigns `-1` beyond `rmsd_threshold=1.0` — the unassigned
   class needs an explicit rule, not a silent drop.
3. **Report both `w` and `log w`.** `log w` is linear in the feature map so it tests whether the
   feature is smooth wrt structure; `w` additionally carries the exp amplification.

### Confidence and confound

Secondary/exploratory, not a third primary metric. Expected dominant signal is fitter identity;
the misspecification component is second-order and may not clear noise on a 5-timepoint system.
Worth computing because it is nearly free from artifacts the matrix already produces.

**Named in advance:** if the smoothness statistic correlates with recovery error across cells, that
is expected mechanically (both worsen with misspecification) and is **not** independent evidence
that smoothness diagnoses regime.

## 8. Identifiability note

The regimes change what `w` is identifiable *through*:

- **fast** — observable depends on `w` only via the scalar `E_w[e^{-z_f}]` per residue. Linear in
  `w`, but a single scalar per residue is a hard bottleneck, and it is dominated by the most-open
  frames.
- **slow2** — linear in the two state populations directly, in uptake space.

Both linear in `w` (unlike the geometric pivot, which is log-linear — a further reason the
admissible choices are also the better-conditioned ones), but `slow2` couples to the target
populations far more directly. Expect `slow2` to recover 60:40 more sharply regardless of which
semantics generated the data. **If that holds it is a result about the reweighting problem, not
about HDX physics** — report it separately from the pivot conclusion.

Opposing consideration, worth measuring rather than assuming: `E[e^{-z}]` is heavy-tailed and
dominated by the few most-open frames, so `k̄_after` is the physically right quantity and the
statistically worst-conditioned one. Under MaxEnt reweighting, weights will chase exactly those
frames. Physics says rate space; finite sampling says rate space is high-variance. That tension is
the sweep's real question.

## 9. Retained findings from the original handoff

1. **Geometric mean of per-frame rates ≡ current production pivot, exactly.** `k_f = k_int·exp(-z_f)`
   with `k_int` frame-independent ⇒ `GM_w(k) = k_int·exp(-z̄) = k̄_first`. Not a new proposal.
2. `k̄_after ≥ k̄_first` always — forced by AM≥GM. Gap ≈ `½·k̄·Var_f(z)`, matching the existing
   `gaussian_mean_rate` second-order term in `investigate_uptake_rate_covariance.py`.
3. **Architecture, not a loss-side opt-in.** `average_first: bool` (declared identically in
   `HDX/forward.py`, `SAXS/forward.py`, `XLMS/forward.py`) is a binary hook in
   `Simulation.forward_pure` (`models/core.py:298-311`) distinguishing "average inputs" from
   "average outputs". Rate-averaging needs the *intermediate* `k = k_int·exp(-z)`, which has no
   averaging hook — extending it touches the `ForwardPass` contract across every model.
4. **Landmine:** `if getattr(fp, "average_first", True):` (`models/core.py:311`) is a truthy check.
   Setting `average_first` to a non-bool truthy value silently falls into the "average inputs" path
   instead of erroring. Needs a **typed mode** replacing the bool, not an overload.
5. Also present but unused as a candidate: `average_after_uptake` (`pf_variance.py:258`),
   `predict_trajectory_ex2`'s `frame_mixture` (`hdx_ex2.py`), and `gaussian_rate_closures`.

## 10. Cheap-evaluation symmetry

Frames enter only through the scalar `k_f = k_int·e^{-z_f}`, so the sufficient statistic is the
per-residue **distribution of `z` across frames**, not the frames. Bin `z_f` per residue once; bins
are frame subsets, so bin mass is linear in `w` and stays exact under reweighting. The mixture
becomes a sum over ~30 bins instead of `F` frames. The Gaussian closure already in the codebase is
the 2-moment version of the same idea. Neither helps if `bc/bh` move — another reason to freeze them.

## 11. Guardrails

- `bc/bh` frozen at 0.35/2.0 throughout. This keeps the work inside the D-only checkpoint (memory:
  `hdx-d-only-checkpoint-bv-mean-model`) — the pivot is varied, the BV mean model is not refit.
- If `log_pf` were to be kept as a live *candidate* rather than a baseline (defensible, since BV
  coefficients were calibrated against `E[contacts]`), the sweep would have to refit `bc/bh` per
  pivot, reopening BV mean-model scope and roughly tripling the work. **Not doing this.**
- Not touching contact features.
- Not resurrecting D/R residue-covariance recovery — orthogonal, stays parked.

## 12. Non-obvious failure modes to watch

- `legacy` being read as "average-first validated" when its diagonal cell is only self-consistency.
- ISO isotope distributions do not exist for any of these experiments, so the bimodality
  discriminator between fast and slow mixing is unavailable — centroids largely cannot separate the
  two limits directly. This is why the recovery matrix is the instrument rather than a direct
  measurement.
