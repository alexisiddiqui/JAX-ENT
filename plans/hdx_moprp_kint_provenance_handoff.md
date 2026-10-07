# MoPrP intrinsic-rate provenance — handoff for an A/B rerun against `moprp.kint`

**Status:** completed sensitivity rerun. Written and run 2026-08-10; results in §7.

**Relates to** `plans/hdx_moprp_pivot_calibration.md` (§13--§16, the MoPrP pivot record) and
`plans/hdx_rate_space_pivot_reweighting.md` (§13, the closed ISO record).

---

## 1. What this handoff is for

Every MoPrP pivot result recorded to date takes its intrinsic rates from a single file,
`data/_MoPrP/expfact_kint_pH4p4_298K_min.dat`, wired in as `CANONICAL_RATE_FILE` in
`fitting/jaxENT/_moprp_recovery_common.py:46` and delivered to every runner as
`inputs.k_ints`. The dataset directory also ships a second, materially different rate vector,
`data/_MoPrP/moprp.kint`, and a third, `_output/out__train_MoPrP_af_clean_1Intrinsic_rates.dat`.

The question this handoff exists to answer is **not** "which rate file is prettier" but:

> Do the §13--§16 conclusions — the failed resolvability gate, the per-pivot coefficient lock, the
> synthetic mismatch matrix, the real-population recovery ordering, and the family dependence —
> survive a change of intrinsic-rate source?

Any conclusion that flips under a rate-file swap was never a statement about pivots. It was a
statement about `expfact_kint_pH4p4_298K_min.dat`.

## 2. Measured difference between the two files (already done)

Both files carry 101 rows; all 97 physics-v2 feature residues (ids 2--101) are present and
positive in both. Restricted to those 97 residues, with `r = k_canonical / k_moprp.kint`:

| quantity | value |
|---|---:|
| ratio range | 1.4149 -- 14.3337 |
| ratio median / mean | 3.1360 / 3.7785 |
| mean ln ratio | **+1.2024** |
| sd of ln ratio | **0.4118** |
| Spearman-free Pearson corr of `ln k` between files | 0.9227 |

SHA-256 prefixes: canonical `f069ca46ecd7fd6a`, `moprp.kint` `7be0f0d23ecec367`.

**Two consequences set the whole design.**

1. **The difference is not a uniform offset.** The residue-wise spread is 0.41 ln units about a
   +1.20 ln mean. Since `k_obs = k_int / PF`, switching files is equivalent to shifting every
   residue's effective `ln PF` by a residue-specific amount with that mean and spread.
2. **BV coefficients cannot absorb it.** The lock has `bh = 0` throughout (§14.1), so `log_pf` is
   `bc · N_heavy`, a ray through the origin, while the required correction is an additive
   residue-wise field. The best-absorbing rescale of the locked `bc = 0.228893` is
   `bc' = 0.279039` (**×1.219**), and it leaves a residual of **0.6291 ln PF RMS**.

Put that residual beside the effects §14--§16 are built on: the legacy→fast log-PF gap that
survives re-locking is **0.093 ln units** (§14.2), and the whole three-pivot uptake MSE floor
spans **0.000698** (§14.1). The rate-file choice perturbs the system roughly **7×** harder than
the pivot signal it is being used to measure. This is the reason the rerun is worth its cost, and
also the reason to expect it may move headline numbers substantially.

## 3. Blast radius

`inputs.k_ints` is the single delivery point, so a swap is confined to one constant but reaches
every runner that consumes it. Directly affected recorded results:

| plan section | experiment | what changes |
|---|---|---|
| §13 | exPfact pivot litmus | PF reconstruction curves, multistart PF solutions, solution ranges, observed/reconstruction RMSEs, coefficient-grid optima |
| §14 | per-pivot coefficient lock | every fitted `bc`, `bh`, calibration MSE, locked-coefficient forward curve |
| §15.1 | synthetic resolution/mismatch sweep | all synthetic targets and recovered populations; diagonal gains; the mismatch matrix |
| §15.2 | real population recovery | recovered populations, decoy masses, uptake MSEs, the fast/legacy/slow-N ordering |
| §15.2 | population Jacobians | Jacobian values, singular spectra, effective ranks |
| §15.3 | PF mirror | PF-to-uptake cross-scores, fitted populations, PF RMSE and uptake MSE |
| §15.4 | five-medoid check | medoid identities may hold in log-PF space, but every uptake fit and recovery changes |
| §16 | robustness sweep | fitted recoveries, TVD gains, pivot winners, all coefficient-dependent conclusions |
| §16.5 | spread regression | the fitted outcomes are the regression response, so the regression and crossover conclusions need regeneration |

Headline values that cannot be assumed to carry over: the §14 locked coefficients; the §15 decoy
masses 0.02644 / 0.000114 / 0.10350; the claim that `fast` leads real-population recovery; the
synthetic mismatch matrices; the Jacobian effective ranks; the §16 family-dependent pivot
ordering; the §16.5 spread-regression result.

Other consumers of `inputs.k_ints`, all of which have recorded numbers that must be checked
against their manifest before reuse: `moprp_coefficient_lock.py`, `moprp_pivot_litmus.py`,
`moprp_population_pivot.py`, `moprp_population_oracle.py`, `moprp_covariance_recovery.py`,
`moprp_noncircular_recovery.py`, `moprp_recovery_audit.py`, `moprp_scale_calibration.py`,
`moprp_joint_reweight_fit.py`, `moprp_diag_d_reweighting.py`, `moprp_shape_prior*.py`, the
target-variance and timepoint-weighting investigations, and the joint-geometry and
covariance-linear-model investigations.

## 4. Provenance is unresolved and should not gate the rerun

A cheap probe was run and **did not discriminate**. Reconstructing the 14×15 uptake from
`median.pfact` under EX2 gives RMSE 0.36626 (canonical), 0.37889 (`moprp.kint`), and 0.37421
(`out__train_...Intrinsic_rates.dat`). The separation is small and the test is confounded: only
49 of 101 residues carry a resolved PF and the `-1` sentinels contribute zero uptake, so the
reconstruction is dominated by the sentinel convention rather than by the rates. **Do not cite
these three numbers as evidence for a preferred file.**

The honest position is that the canonical file is a *recomputation* (its header states 298 K,
pH 4.4, exPfact `validation/README.md` convention) while `moprp.kint` ships with the dataset and
is the more likely input to the shipped `median.pfact` / `MoPrP_pfactors.dat`. If that is true it
is a real internal inconsistency in §13 and §15.3, which score BV predictions against an exPfact
PF reference that may itself have been fitted under the *other* rate vector. Establishing it
requires provenance work (exPfact run logs, the `moprp.seq`/`moprp.times` conditions, the
temperature/pH actually used for the shipped fit), not more curve-fitting.

**Therefore the rerun is designed as a sensitivity analysis, not as a correction.** Neither file
is declared correct. Both are run; the deliverable is the difference.

## 5. Protocol

### 5.1 Mechanism

Add a rate-source selector to `_moprp_recovery_common.py` rather than editing
`CANONICAL_RATE_FILE` in place:

- keep `CANONICAL_RATE_FILE` as the default so existing runners are byte-identical when the
  selector is unset — this is the regression anchor;
- add `RATE_SOURCES = {"expfact_recomputed": ..., "moprp_shipped": MOPRP / "moprp.kint"}` with the
  provider/temperature/pH metadata each file actually claims (`moprp.kint` carries no header, so
  record its conditions as **unknown**, not as 298 K / pH 4.4);
- thread an optional `rate_source` through `load_blinded_ensemble_inputs` /
  `load_ensemble_inputs`, defaulting to `expfact_recomputed`;
- record the resolved file path and its SHA-256 in every output manifest. Any result JSON without
  a rate-source field is retroactively `expfact_recomputed`.

### 5.2 What to rerun, in order

1. **§14 coefficient lock** under `moprp_shipped`, all three pivots, both ensembles. Nothing
   downstream is interpretable until the coefficients are re-locked, because §15.2 already showed
   that the coefficient regime moves the population answer far more than the pivot does.
2. **§13 litmus** under `moprp_shipped` — the resolvability gate, the PF panel, the uptake panel,
   and the 41×41 coefficient scan.
3. **§15.2 real-uptake fit** at the new per-pivot locks: recoveries, decoy masses, uptake MSEs,
   PF RMSE, and the population-Jacobian spectra.
4. **§15.1 synthetic mismatch matrix**, Folded-dominant grid, both coefficient settings.
5. **§16 three-family TVD sweep** and §16.5 regression — only if steps 1--4 show any conclusion
   moving. If steps 1--4 are stable, §16 is expensive and can be deferred with that stated.

### 5.3 Pre-registered comparison table

For every rerun quantity, report the pair `(expfact_recomputed, moprp_shipped)` and the delta.
The primary readouts, with the pre-registered question each answers:

| readout | question | stability criterion |
|---|---|---|
| locked `(bc, bh)` per pivot | does `bh = 0` survive? | `bh = 0` on the boundary under both files |
| locked `bc` ratio between files | is the swap a pure `bc` rescale? | ratio within 5% of the predicted ×1.219 |
| calibration MSE floor spread across pivots | is the pivot still unidentifiable in uptake space? | spread stays below 1% of the floor |
| §13 resolvability gate pass count | does the gate still fail on ~all residues? | still 0/48 |
| §15.2 recovery ordering | does `fast` still lead? | same ordering, same sign of the gaps |
| §15.1 mismatch matrix signs | is mismatch still sign-asymmetric with a positive diagonal? | all diagonal cells positive; `fast`-target off-diagonals still negative |
| §16 family winners | does the family dependence persist? | same winner per family per coefficient setting |

### 5.4 Decision rules, fixed in advance

- **All criteria hold.** The §13--§16 conclusions are rate-source-robust. Record that as a
  strengthening amendment and close the question. This is the outcome the current record needs.
- **The coefficient lock moves but the orderings hold.** The pivot conclusions stand; the locked
  coefficients become rate-source-conditional and must be quoted with their file.
- **Any ordering flips.** The affected conclusion is withdrawn to "rate-source-dependent" in the
  pivot document, exactly as §16 withdrew the unconditional `fast` headline. Provenance work then
  becomes blocking, because the answer now depends on which file is correct.

### 5.5 Verification, carried over from the existing record

Each rerun must reproduce the checks its section already specifies, under the new rates:
Jensen ordering with zero violations; one-frame pivot equality exact; analytic Jacobian columns
agreeing with central finite differences; the peptide-1 hold-out and full-support decoy freedom
unchanged. Additionally: a `rate_source="expfact_recomputed"` run must reproduce the existing
shipped artifacts **bit-for-bit** (compare the §16.3 CSV against SHA-256
`3545d50d548b7dc6a42a820699ec19b05bad19c2d1efd5995b93e9c4ec336f32`). If it does not, the
threading changed semantics and nothing else in the rerun is trustworthy.

## 6. Scope and known limitations

- **The third file is out of scope.** `_output/out__train_MoPrP_af_clean_1Intrinsic_rates.dat`
  covers 48 rather than 49 overlap residues and appears to be a training by-product. Note its
  existence; do not add a third arm.
- **This does not resolve which rate vector is physically right**, and is not designed to. A
  sensitivity result is a bound on how much that unresolved question matters.
- **The §15.5 caveats survive unchanged**: the real-data readout still has no error bar, and one
  real target still yields one recovered population per pivot. A rate-file swap gives a second
  point on a nuisance axis, not a resampling distribution.
- **§16.5's response variable is regenerated, not reweighted.** The regression cannot be updated
  by rescaling; if step 5 runs, it runs from new fits.

---

## 7. Completed sensitivity result (2026-08-10)

**Status:** steps 1--5 completed. The real-population ordering and family winners survive, but the
Folded synthetic mismatch-sign claim is **rate-source-dependent**. Locked coefficients must be
quoted with their intrinsic-rate source.

Artifacts are under
`jaxent/examples/2_CrossValidation/fitting/jaxENT/_moprp_kint_sensitivity/moprp_shipped/`:
`coefficient_lock/`, `pivot_litmus/`, `population_pivot/` (full Folded grid),
`population_pivot_all_families/` (the §16 reduced three-family grid), and
`spread_regression/`. Every JSON manifest records the resolved rate file, full SHA-256, provider,
and the unknown shipped temperature/pH as null.

### 7.1 Pre-registered comparison

| readout | `expfact_recomputed` | `moprp_shipped` | result |
|---|---:|---:|---|
| locked legacy `(bc, bh)` | `(0.228893, 0)` | `(0.169895, 0)` | boundary survives; shipped/canonical `bc = 0.7422` |
| locked fast `(bc, bh)` | `(0.232237, 0)` | `(0.171798, 0)` | boundary survives; ratio `0.7398` |
| locked slow-N `(bc, bh)` | `(0.228957, 0)` | `(0.169870, 0)` | boundary survives; ratio `0.7419` |
| calibration MSE range | `0.041163--0.041861` | `0.045294--0.045604` | shipped spread is 0.69% of its floor |
| §13 gate | 0/48, fail | 0/48, fail | stable |
| real recovery ordering | fast > slow-N > legacy | fast > slow-N > legacy | stable |
| Jacobian ranks, legacy/fast/slow-N | 3 / 4 / 2 | 3 / 3 / 2 | fast rank moves |
| §16 family winners, published | fast / fast / slow-N | fast / fast / slow-N | stable (Folded / PUF1 / balanced) |
| §16 family winners, locked | fast / slow-N / slow-N | fast / slow-N / slow-N | stable |

The anticipated `bc` multiplier of 1.219 does not describe the refit in the stated
shipped/canonical direction. The fitted multiplier is about 0.741 for every pivot, while the
absolute calibration floor rises by 0.00374--0.00413 MSE.

At the locked coefficients, the canonical Folded sweep had negative recovery gain for every
off-diagonal fitter applied to a fast-generated target. Under `moprp_shipped`, five low-minority
cells become positive: legacy at minority 0.015, 0.0216, and 0.02881, and slow-N at 0.015 and
0.0216. The corresponding largest positive recovery/TVD gains are +8.684 percentage points and
+0.08461. This fails the pre-registered mismatch-sign criterion, so that sign-asymmetry statement
is withdrawn to **rate-source-dependent**. All matched diagonal recovery gains remain positive.

### 7.2 Real recovery and verification

At the per-source locked coefficients, real recovery percentages change from
63.80 / 93.25 / 66.43 to 50.88 / 92.35 / 56.98 for legacy / fast / slow-N. Decoy masses change
from 0.02644 / 0.000114 / 0.10350 to 0.005263 / 0.0000469 / 0.15104. Thus the fast lead survives,
but the absolute population answers are rate-source-conditional.

Both shipped-rate population runs have zero Jensen violations, exact one-frame equality to
floating-point precision (maximum `5.55e-17`), and analytic/central-finite-difference Jacobian
agreement better than `9.5e-11` maximum absolute error. The §16.5 result also survives: the
spread-only mechanism remains unconfirmed; family partial R-squared changes 0.7711 to 0.7783,
delta AIC changes 190.63 to 194.86, and the fitted fast/slow-N crossover changes 0.42894 to
0.38263 (shipped cluster-bootstrap 95% interval 0.1832--0.6523).

The explicit default-source regression guard also passes: regenerating the canonical §16.3
single-start control produces SHA-256
`3545d50d548b7dc6a42a820699ec19b05bad19c2d1efd5995b93e9c4ec336f32` byte-for-byte.

Per the pre-registered decision rules, provenance work is now blocking for any use of the
Folded fast-target mismatch sign and for quoting absolute fitted coefficients or populations.
The robust statements are the failed §13 resolvability gate, `bh = 0`, the fast-led real recovery
ordering, the §16 family winners, and the negative spread-only mechanism result.

---

## 8. Provenance verdict (2026-08-10)

**Verdict: `moprp.kint` did not produce the shipped `median.pfact`. The fit used exPfact's
internally recomputed rates at 298 K and pH 4.4. However, because the shipped median was committed
in 2021, those rates used exPfact's then-current poly-DL-alanine (PDLA) reference constants. They
are close to, but are not identical to, the post-2023 3Ala vector in
`expfact_kint_pH4p4_298K_min.dat`. Thus neither existing selector arm is the exact historical fit
vector.**

### 8.1 Archival evidence chain

The local seven-file bundle (`moprp.seq/.kint/.dexp/.times/.ass/.list` and `median.pfact`) is the
`pacilab/exPfact` `validation/` bundle byte-for-byte after normalising CRLF to LF. In particular,
the normalised upstream/local hashes are `66b81b74...` for `moprp.kint` and `4a42fc08...` for
`median.pfact`; the differing raw local hashes are solely line-ending changes. The upstream
validation README documents the complete production path:

- the experiment is described as 25 °C and pH 4 (with the executable analysis convention stated
  more precisely as `--temp 298 --pH 4.4`);
- the 5,000-solution command runs `exPfact.py` at 298 K/pH 4.4 with harmonic penalty `1e-8`;
- `descriptive_statistics.py --res out --top 50` then writes `median.pfact` from the best 50
  solutions.

Most decisively, `exPfact.py` calls `calculate_kint_for_sequence(..., temperature, pH)` inside the
fit. It has no `moprp.kint` argument and does not read that file. Git history agrees with this
execution path: `moprp.kint` entered at commit `93f4e40a` on 2021-09-13, while `median.pfact`
entered later at `1f99a0cc` on 2021-11-17. The 3Ala reference was not introduced until commit
`965391ba` on 2023-02-24; before then the constants now named `*_pdla` were the only/default
constants. Therefore the shipped median's exact intrinsic-rate provenance is **exPfact 2021 PDLA,
298 K, pH 4.4, calculated in memory**, not the adjacent `moprp.kint` file and not the later 3Ala
recomputation.

The local `_MoPrP.tar` contains the same bundle. `extract_data_ValDX.py` only reformats
`moprp.dexp`, `moprp.list`, and `median.pfact`; it does not generate rates or refit PFs. The
deprecated MoPrP notebooks consume the separate HDXer `out__train_...Intrinsic_rates.dat`, and
`spectra/README.json` independently identifies `pacilab/exPfact/validation` as the source of the
isotopic-envelope files. None supplies an alternative generation path for the shipped median.

Primary sources: [upstream validation README](https://github.com/pacilab/exPfact/blob/master/validation/README.md),
[upstream fit implementation](https://github.com/pacilab/exPfact/blob/master/python/exPfact.py), and
[the validation directory](https://github.com/pacilab/exPfact/tree/master/validation). The paper
also identifies the GitHub repository as the scripts/data source and describes the same MoPrP
validation experiment ([Stofella et al., 2022](https://pubs.acs.org/doi/10.1021/jasms.2c00005)).

### 8.2 Condition-identification computation

The grid reused upstream exPfact's own `calculate_kint_for_sequence` and explicitly converted its
hr⁻¹ return to min⁻¹. With current exPfact revision `34d13293`, the 298 K/pH 4.4 control reproduces
the canonical file across 97 positive residues with maximum absolute rate error
`2.13e-14 min^-1` and maximum absolute ln-ratio `4.22e-15`. This validates the machinery and the
canonical file's stated conditions.

No tested point reproduces `moprp.kint`. On the requested grid augmented with the documented pH
4.0, the best current-code point is 293 K/pH 4.0, but its residue-wise RMS ln-ratio is **0.4320**
and maximum absolute ln-ratio is **1.4467**, far from a numerical match. Repeating the grid with
the 2021 PDLA implementation also gives no match (best: 298 K/pH 4.0; RMS ln-ratio **0.4332**,
maximum **1.6110**). Consequently the physical/computational conditions encoded by the orphaned
`moprp.kint` file remain **unresolved**; temperature or pH alone under either relevant exPfact
implementation cannot explain it.

The historical 2021 rate vector at the documented fit conditions is much closer to the canonical
3Ala vector than to `moprp.kint`, but is still distinct: versus canonical its mean/sd ln-ratio is
`+0.3722/0.0305` (RMS `0.3735`), whereas versus `moprp.kint` it is approximately
`+0.8302/0.4194`. The third HDXer intrinsic-rate file remains out of scope as pre-registered.

Artifacts are confined to
`fitting/jaxENT/_moprp_kint_provenance/`: the reusable grid script, current-3Ala and historical-PDLA
CSVs, and SHA-256-stamped manifests. No PF-space refit was run: step 3 was conditional on archival
and condition evidence remaining ambiguous, while the source code and dated production recipe
directly establish which rates the fit used. Avoiding a modern surrogate refit also avoids
confusing optimizer reproduction with provenance.

### 8.3 Decision-rule bookkeeping

The pre-registered binary premise was incomplete: the shipped fit used neither candidate exactly.
Accordingly:

- §13 and §15.3 did **not** score BV against a reference fitted with `moprp.kint`; they scored
  against a reference fitted with the historical 2021 PDLA vector. The present canonical 3Ala arm
  is a near-source sensitivity arm, not an exact same-rate comparison.
- The Folded fast-target off-diagonal mismatch sign remains **rate-source-dependent** and is not
  quotable as an unconditional pivot claim. Under the current 3Ala source all tested locked cells
  are negative; under `moprp.kint` five low-minority cells are positive. No sign is assigned to the
  unrun historical-PDLA arm.
- Absolute locked `bc` values and recovered populations remain quotable only together with the
  actual rate file and SHA-256. The §7 canonical and shipped values may be quoted as 3Ala and
  `moprp.kint` sensitivity results respectively; neither may be labelled the exact shipped-fit-rate
  result.
- The verified default for reproducing the shipped exPfact PF reference is a new
  **`expfact_historical_pdla_2021` (298 K, pH 4.4)** source, to be materialised before any future
  absolute-number run. Until that selector exists and is run, there are no absolute BV
  coefficients/populations under the exact verified source to quote. The existing
  `expfact_recomputed` 3Ala default remains unchanged for regression compatibility, but must not be
  described as the historical source of `median.pfact`.

The rate-source-robust conclusions from §7 remain quotable: the §13 resolvability gate fails,
`bh = 0`, real recovery is ordered fast > slow-N > legacy in both completed arms, the §16 family
winners are unchanged, and the spread-only mechanism remains rejected. The deferred §12 step-3 Σ
rebuild should consume the historical-PDLA vector once it is materialised; no Σ work is performed
here.
