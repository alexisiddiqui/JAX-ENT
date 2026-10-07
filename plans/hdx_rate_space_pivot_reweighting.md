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

### Stage 2 outcome (2026-08-05): EX1 contamination produces a signed robustness failure

The intrinsic rates provide ample leverage for this arm: 293 residues span 135.8–163,228 min⁻¹
(median 1,029.9; 5th–95th percentile spread 23.2×). Targets used `τ=0.001` and `0.01` min, giving
`τ·k_int,median=1.03` and `10.30`. The first keeps the median target at 0.608 uptake at 10 min; the
second moves the informative region later, with median 0.639 at 60 min. Across all written targets
the corresponding 10/60-min median ranges were 0.499–0.695 / 0.742–0.999 and
0.153–0.348 / 0.529–0.710. Thus neither arm lost the whole curve outside the five-timepoint window.

All 28 fast/slow2 `τ=0` targets (dfrac and segment files) were regenerated byte-identically to the
committed Phase-2 targets before fitting. The nonzero arm then ran 504 fits: two τ values × two
target semantics × three fitters over the same 14 configuration/population points and three fixed
splits. Results are separate from the pivot matrix in `_phase2_tau_matrix/phase2_tau_matrix.csv`
and `phase2_tau_summary.csv`; the committed Phase-2 diagonal is joined only as the `τ=0` recovery
baseline, not refitted.

The direct observable has an unambiguous sign. Every one of the 36 grouped cells has positive mean
fitted-minus-target uptake, and every one-component shared-rate position shift is positive:

| τ (min) | mean uptake bias range | median log-rate position-shift range |
|---:|---:|---:|
| 0.001 | +0.0002 to +0.0527 | +0.053 to +0.268 |
| 0.01 | +0.1190 to +0.1416 | +1.449 to +1.831 |

This is the predicted direction: the τ=0 fitters retain too much `k_int`-driven rate and therefore
exchange too quickly relative to targets whose rates have been flattened toward `1/(Pτ)`. Bias is
largest around the informative middle of the time window (group-average 10-min bias +0.017–0.035
at `τ=0.001`, +0.142–0.155 at `τ=0.01`). A component-count/width claim is not made: five timepoints
leave that readout low-power, while the signed position shift is already decisive.

Population compensation is fitter-specific. Values below are mean signed open-population bias,
shown as the range over fast and slow2 targets:

| configuration | τ | fast fitter | legacy fitter | slow2 fitter |
|---|---:|---:|---:|---:|
| BI residue | 0.001 | −0.188 to −0.138 | −0.032 to +0.088 | +0.034 to +0.045 |
| BI residue | 0.01 | −0.200 to −0.198† | −0.109 to −0.069 | +0.157 to +0.167 |
| BI width-10 | 0.001 | −0.187 to −0.117 | +0.003 to +0.079 | +0.083 to +0.096 |
| BI width-10 | 0.01 | −0.200 to −0.200† | −0.122 to −0.089 | −0.053 to −0.027 |
| TRI residue | 0.001 | −0.101 to −0.086 | −0.068 to −0.066 | +0.024 to +0.055 |
| TRI residue | 0.01 | −0.116 to −0.116† | −0.093 to −0.090 | +0.130 to +0.132 |

† Degenerate boundary solution, not a population-sensitivity estimate. Across all `τ=0.01`
fast-fitter runs, 83% have `ESS < 2` and 73% have recovered open mass numerically equal to zero
(`recovered_open ≤ 1e-6`). The identical TRI bias of −0.116 for both
target semantics is exactly minus the mean true-open population: the optimiser has expelled the
open state. The same boundary failure has already begun at `τ=0.001`, where 23% of fast-fitter
runs have numerically zero open mass. Consequently, the apparent loss of target-semantics
dependence at `τ=0.01` is partly a simplex clamp, not solely the physical removal of `k_int` from
the target rates.

Most shifts clear the pre-established 0.02/0.04/0.06 recovery floors, already at `τ=0.001`.
At `τ=0.01`, target-semantics dependence nearly disappears within each fitter, as expected when
`k_int` drops out; the fitter's compensation geometry dominates. Fast fitters consistently remove
open population, whereas slow2 fitters generally add it. BI width-10 slow2 is the exception and is
not interpreted directionally: its signed mean is small relative to its 0.118–0.120 MAE and the
layout was already at its recovery floor.

The collapse is strongly fitter-dependent:

| fitter | median ESS, τ=0.001 | median ESS, τ=0.01 | ESS < 2 at τ=0.01 | zero-open at τ=0.01 |
|---|---:|---:|---:|---:|
| fast | 20.1 | 1.06 | 83% | 73% |
| legacy | 402 | 52.6 | 0% | 0% |
| slow2 | 12.0 | 2.68 | 33% | 0% |

An ESS near one is a single-frame solution, not an ensemble. Legacy never collapses at either τ
and at `τ=0.01` carries the smallest-magnitude population bias (−0.07 to −0.12, versus roughly
−0.12 to −0.20 for fast). Its geometric pivot is compressive, so misspecification cannot drive it
to a simplex boundary as readily: it fails soft rather than hard. This exposes the
production-relevant tradeoff: **fast/slow2 are better-conditioned when correctly specified (§8,
§13), but more brittle under contamination they cannot represent.** Production currently uses
legacy, whereas the pivots Phase 2 argues for switching to are the ones that fail most severely
when the EX regime is wrong.

The large-τ arm is therefore a severe extrapolation failure, not a pivot ranking. Median MSE remains
0.023–0.070 while the τ=0 models spend their available weight freedom without representing the
slower kinetics. **Robustness bound:** EX1 contamination at `τ·k_int,median≈1` is already resolvable,
and at `≈10` it overwhelms the 0.02–0.06 population-resolution floor. No conclusion here identifies
one τ=0 pivot as physically correct, because all fitters are structurally misspecified in this arm.

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

### Outcome (2026-08-05): readout does not pass validation

Computed post hoc on all 378 Phase-2 histories, with no refitting and without enabling
`boltzmann_frame_consistency` as a loss. For each recovered simplex, `G_i = Σ_r log_Pf_i(r)` was
computed directly from the frozen BV features. Both `log(w)` and `w` were regressed on `G` with a
free slope and intercept separately inside every shipped cluster (`{0,1}` for BI and `{-1,0,1}`
for TRI); cluster residual variances were pooled using recovered cluster mass. The analyzer writes
the per-fit values to `phase2_semantics_matrix.csv` and paired row-relative summaries to
`phase2_smoothness_summary.csv`.

The arithmetic and frame selection pass a direct HDF5 check: for one BI open-cluster cell the
independent 44-frame regression reproduced both the stored slope and residual variance to floating
precision. Existing recovery fields are byte-identical prefixes of the augmented CSV rows.

The originally proposed diagonal sign check was ill-posed for these synthetic targets. Their true
weights are exactly uniform inside each cluster (`weights[mask] = cluster_mass / count`), so the
true within-cluster slope is `α=0` by construction. The observed diagonal result—only 42.9% positive
slopes, with median effectively zero (5th–95th percentile −0.0113–0.0139)—is therefore the expected
null, not evidence against the assumed Boltzmann relation. The direct recomputation and label-set
checks still establish that the statistic was implemented on the intended frames, but ISO cannot
validate its central coordinate assumption.

Off-diagonal fits are usually rougher than their same-fitter, same-population, same-split diagonal:

| configuration | `log(w)` pairs with ratio > 1 | `w` pairs with ratio > 1 | median off-diagonal ratio range (`log(w)` / `w`) |
|---|---:|---:|---:|
| BI residue | 98.1% | 88.9% | 7.65–74.3 / 1.44–1036 |
| BI width-10 | 81.5% | 77.8% | 1.16–3.76 / 0.55–232 |
| TRI residue | 86.8% | 76.4% | 1.26–25.5 / 0.75–165 |

Those large ratios do **not** rescue the hypothesis. The diagonal truth weights are uniform inside
each cluster, so a successful diagonal fit drives the reference residual toward zero and makes the
ratio ill-conditioned. More importantly, within-fitter `log(w)` variance tracks absolute recovery
error (Spearman ρ 0.02–0.78 across configurations) and especially total tilt/KL (ρ 0.05–0.95).
That is the pre-registered null: a misspecified fit spends more tilt and becomes less smooth while
also recovering population less accurately. It is not independent evidence for fast versus slow
interconversion, and the direction is not regime-specific.

**Verdict:** the misspecification-associated roughness clears replicate noise, but the proposed
Boltzmann-coordinate readout cannot be validated by cluster-uniform ISO truth weights and does not
clear the recovery/tilt confound. Drop it from the τ arm rather than carrying a confounded
secondary statistic into new fits.

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

---

## 13. Phase 2 result — the semantics matrix (378 fits, τ=0, maxent ≤ 0.001, tilt < 1)

Source: `jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_phase2_semantics_matrix/phase2_semantics_summary.csv`.
MAE of recovered open population; rows = target semantics, columns = fitter semantics.
Stage 0 resolution floors: BI residue 0.02, BI width-10 0.04, TRI residue 0.06.

**ISO_BI residue** (the instrument that works)

| target \ fitter | fast | legacy | slow2 |
|---|---|---|---|
| fast | **0.008** | 0.102 | 0.057 |
| legacy | 0.175 | **0.006** | 0.039 |
| slow2 | 0.143 | 0.016 | **0.007** |

**ISO_BI width-10**

| target \ fitter | fast | legacy | slow2 |
|---|---|---|---|
| fast | **0.049** | 0.122 | 0.063 |
| legacy | 0.164 | **0.056** | 0.077 |
| slow2 | 0.065 | 0.071 | **0.053** |

**ISO_TRI residue**

| target \ fitter | fast | legacy | slow2 |
|---|---|---|---|
| fast | **0.069** | 0.069 | 0.086 |
| legacy | 0.104 | **0.058** | 0.053 |
| slow2 | 0.093 | 0.073 | **0.030** |

### Findings

1. **Frame-averaging semantics is a first-order error source, not a rounding detail.** On BI
   residue the diagonal is 0.006–0.008 — an order of magnitude below the 0.02 floor — while the
   worst mismatch is 0.175. The pivot choice costs ~20× the self-consistent recovery error.
2. **The error is a pure sign-definite bias, not noise.** For every `fast` fitter off-diagonal,
   `|bias_open| == mae_open` to three decimals: legacy target → −0.175, slow2 target → −0.143.
   The arithmetic rate mean over-weights open frames, so the fitter compensates by removing open
   population. Symmetrically, a `legacy` fitter on a `fast` target biases **+0.102**. Direction is
   exactly the Jensen ordering (AM ≥ GM), so the sign is predictable a priori.
3. **`legacy` and `slow2` are near-neighbours; `fast` is the outlier.** legacy↔slow2 costs
   0.016–0.039 on BI residue, versus 0.102–0.175 against `fast`. Averaging log-PF is a better proxy
   for slow interconversion than for fast.
4. **Misspecification is visible in ESS, not only in MSE.** A `legacy` fitter on non-legacy targets
   lands at median ESS 16–95 against 600–730 on the diagonal: it burns tilt budget fighting a
   forward model it cannot satisfy. ESS collapse is therefore a usable in-practice misspecification
   alarm where the truth is unknown.
5. **Peptide width degrades but does not invert the picture.** BI width-10 diagonals (0.049–0.056)
   sit at the 0.04 floor, so only legacy/fast (0.164) and fast/legacy (0.122) clear it. Residue
   heterogeneity within a peptide broadens curves the same way a frame mixture does (§7), which is
   the expected confound.
6. **TRI residue is mostly at its floor and must not be over-read.** With floor 0.06 only
   legacy/fast (0.104) and slow2/fast (0.093) are resolvable; the entire `fast`-target row
   (0.069/0.069/0.086) is not. The one clean signal there is again that `fast` is the outlier.

### Caveats

- All targets are self-consistent by construction; the diagonal measures identifiability, not the
  correctness of any pivot. Nothing here validates `legacy` as physics.
- τ = 0 throughout — the EX1 arm (§6 Stage 2) is untested. The zero-fit §7b smoothness readout was
  subsequently evaluated and rejected as tilt/recovery-confounded (see its outcome above).
- TRI width-10 excluded (fails the Stage 0 gate; see the Stage 0 record).
