# omc_scalar: the sidecars' scalar-distance OMC, and the self-tuning OMC, run in the new (selgci/stk) setup

Read-only analysis: no repo edits and no framework fits. Everything is NumPy/SciPy, run from the repo root with `.venv/bin/python`
and `OMP_NUM_THREADS=4` (5 worker processes). The brief changed mid-task: T2 (scalar two-sided variance) was dropped and not run.
The main task became the sidecars' original scalar-distance OMC construction, with T1 (self-tuning OMC on ISO) kept as a section.

**What I reused rather than rewrote:**
- `stk/s0_common.py`: kernel, walk covariance, `Scorer`, rule `select()`, S grids.
- `selgci/q0_common.py`: ISO/MoPrP models, p0, splits, ISO exact dual.
- `lap/l0_common.fit_omc`: the objective form.
- The sidecar kernel recipe (`run_maxent_omc_sidecar.regularizer` → `original_omc_laplacian.build_omc_kernel`) and the sidecar
  geometry files.
- **γ = 0 baselines are not refitted:** ISO = stk s5 / selgci q2 (reproduced 432/432 picks `[o2]`); MoPrP = stk s2/s3 at λ = 0.

Every number is tagged with its script. Full tables: `o7_tables.md`.

| tag | script | outputs |
|---|---|---|
| `[i0]` | `i0_common.py` | shared: sidecar kernel, penalty forms, whitened augmented dual, active-set pure-OMC QP, primal solver |
| `[o1]` | `o1_geometry.py` | `o1_kernels.csv`, `o1_graphs.npz`, `o1_moprp_*.npz`, `o1_bi_in_tri.npy`, `o1.log` |
| `[o2]` | `o2_base.py` | γ = 0 ISO sweep: `o2_sweep.csv`, `o2_weights.npz`, `o2.log` |
| `[o3]` | `o3_omc_iso.py` | ISO OMC sweeps (T1 graphs + scalar kernels): `o3_scale.csv`, `o3_sweep.csv`, `o3.log` |
| `[o3b]` | `o3b_followup.py` | 3-seed follow-up of RMSD h = 0.1 L_sym (+ RMSD h = 1 L_sym): `o3b_sweep.csv`, `o3b.log` |
| `[o4]` | `o4_omc_moprp.py` | MoPrP OMC sweeps: `o4_scale.csv`, `o4_sweep.csv`; follow-up (`followup` arg) `o4b_sweep.csv`, `o4c.log` |
| `[o5]` | `o5_select.py` | rules, summaries, paired Δ vs γ = 0: `o5_summary_{iso,moprp}.csv`, `o5_selected_*.csv`, `o5.log` |
| `[o6]` | `o6_fig.py` | `F1_T1_gamma.png`, `F2_iso_heat.png`, `F3_moprp_heat.png` |
| `[o7]` | `o7_tables.py` (formatting only) | `o7_tables.md` |
| `[o8]` | `o8_attractor.py` | null-vector "attractor" of each penalty, label-coordinate check: `o8_attractor.csv`, `o8.log` |

## Bottom line

1. **The sidecar construction does not beat MaxEnt with Work Scale, PyRosetta, the complete graph (h ≥ 10) or the MoPrP metrics** `[o5]`.
   - **On ISO_TRI** these kernels are at best tied with γ = 0 (|Δ| ≤ 0.35 at γ = 0.1) and lose monotonically with γ: Δ −1.6 to −4.0 at γ = 1 and −4.4 to −8.2 at γ = 3.
   - **On ISO_BI** they pass the ≥ 95% control at γ ≤ 1. At γ = 3 with σ = 0.02 they fail: complete 0/6, Work Scale 0–3/6, PyRosetta 0–4/6.
   - **On MoPrP** no configuration is ahead on AF2_MSAss. The best is −0.16 ± 0.18 (original quartic energy, Work Scale h = 0.01); typical values are −1 to −4.5 at γ = 1–3, and pure OMC gives −1.5 to −9.5.
   - **On AF2_filtered** the only positive cells (+1.1 to +1.25, 1 fold of 6) come from rule D falling back to p0 in one more fold, not from OMC-shaped populations.
   - **The old-sidecar ordering therefore survives in the new setup for these metrics.** There OMC trailed MaxEnt by 3.4–4.9 points on TRI (61.9 vs 57.1–58.5) and roughly tied on MoPrP. Here, rule D with KL to uniform or to p0 removes the gap at small γ but never reverses it.
2. **One scalar-distance configuration reverses the ordering, and it passes BI: Cα-RMSD at h = 0.1 in the L_sym form** `[o3, o3b, o5]`. Over 3 seeds × 6 folds:
   - ISO_TRI rule D goes 60.07 → 64.92 ± 2.04 at γ = 1, 65.80 at γ = 3 and 67.12 ± 2.16 (minimum 63.1) for pure OMC at σ = 0.01. At σ = 0.02 it goes 56.46 → 61.83 / 63.69 / 65.50.
   - The paired Δ is +4.9 to +9.0, with SE ≈ 0.6.
   - Unassigned mass falls from 28.4 to 20.0%, and ESS is unchanged at 12–15%.
   - These picks exceed the MaxEnt path's own oracle (61.5 / 57.9).
   - ISO_BI passes 18/18 at both σ for every γ (98.5–99.6).

   **It is fragile, and part of it is a structural prior aligned with how the labels are defined:**
   - The same kernel in the L = D − K form gives only 0.0 to +1.6 with KL, and +2.2 / +2.8 as pure OMC (seed 0).
   - RMSD h = 1 in L_sym loses: −1.9 / −2.8 over 3 seeds.
   - The ISO labels are themselves defined by Cα RMSD ≤ 1 Å to the open/closed references.
   - The L_sym null vector (√degree) of this kernel already moves TRI from 31.1 to 35.9 and shifts mass to closed (37 → 50%) with no data at all `[o8]`.
   - It was picked post hoc from about 60 seed-0 configurations; seeds 1–2 confirm it, but the bandwidth choice was not pre-registered.
   - **The MoPrP analogue (dihedral h = 0.1, L_sym) does not help:** Δ −0.2 to −0.5 (MSAss), −0.5 to +0.2 (Filtered); pure OMC = p0 `[o4 followup, o5]`.
3. **T1: the self-tuning OMC term helps a little on ISO and passes the control; the self-tuning weights themselves add nothing** `[o3, o5]` (3 seeds, λ = 0 scoring).
   - **ISO_TRI:** 60.07 → 61.79 ± 2.18 at γ = 0.3 (Δ +1.71 ± 0.43) and 61.53 at γ = 1 (σ = 0.01); 56.46 → 58.92 / 58.73 at σ = 0.02 (Δ +2.45 / +2.26).
   - **ISO_BI:** 18/18 at every γ, with recovery rising 98.4 → 99.2–99.7. Pure OMC reaches 99.7.
   - **The sidecars' own kNN20 graph, unweighted with L_sym, gives the same numbers within ±0.4** (TRI 61.76 at γ = 0.3). It has exactly the same edge support, 32,978 edges `[o1]`. The improvement comes from the lnPF-kNN support, not from the self-tuning weights.
   - **On TeaA this reverses the old-sidecar MaxEnt-vs-OMC ordering** (61.9 vs 57–58 there; 60.1 vs 61.5–61.8 here). The size is about +1.5 to +2.8 points.
   - **It contradicts the MoPrP result.** stk T2 found the same self-tuning term cost about 1 point on MSAss.
4. **Recommendation.**
   - Keep γ = 0 (MaxEnt/KL + rule D) as the production default. No OMC variant improves MoPrP, which is the system the recipe is for, and every scalar-metric term that helps ISO is either small (kNN/L_st, +1.5–2.8) or tied to the label coordinate (RMSD h = 0.1 L_sym).
   - If an OMC term is carried as a sensitivity, use the lnPF-kNN20 L_sym graph at γ = 0.3. It is the only one that is positive on both ISO systems, passes BI and is insensitive to weighting. Report it alongside γ = 0, never instead of it.

## M0. Construction, as run `[i0, o1]`

### Kernel
- **Reproduced exactly from the sidecars:**
  - each distance matrix is divided by its median positive upper-triangle distance;
  - K_ij = exp(−min(d̃²/(2h²), 80)), zero diagonal;
  - h ∈ {0.01, 0.1, 1, 10, 100, 1000}.
- **Check:** recomputed medians and normalised matrices equal the stored sidecar files exactly (max |Δ| 0). The MoPrP Work Scale recipe matches to 7.7e-8 and the dihedral recipe to 0 `[o1]`.
- **One deviation:** entries at the exponent cap (e^−80 = 1.8e-35) are set to exactly 0. In sidecar units they are inert, but after γ rescaling a uniform e^−80 floor would turn into a complete graph.

### Penalty forms
- **Primary, as instructed:** 'lap' = wᵀ(D − K)w = ½Σ K_ij (w_i − w_j)².
- **Sensitivity:** 'sym' = wᵀ(I − D^−½KD^−½)w.
- **Sensitivity (MoPrP only):** 'orig' = the sidecars' actual `original_omc_laplacian` energy, ½Σ K_ij w_i w_j (w_i − w_j)², which is quartic and weight-weighted. Its N² factor is absorbed by the scaling.
- **The complete graph is an inverse-ESS penalty:** in the 'lap' form, wᵀ(NI − 11ᵀ)w = NΣw² − 1 = 1/ESS − 1. That is what h ≥ 10 becomes.

### Metrics
- **ISO:**
  - all-pairs Work Scale (|Δ mean lnPF|), Cα RMSD and PyRosetta |Δ ref2015| from `artifacts/iso_policy_sidecars/geometry/`, which are identical to `…_knn20/geometry/`;
  - TRI has 2225 frames;
  - BI = the exact, order-preserving 874-frame subset of TRI (identical Nc/Nh rows, labels agree), re-median-scaled on BI `[o1]`.
- **MoPrP:**
  - Work Scale and periodic φ/ψ RMS from the fresh sidecar geometry for AF2_MSAss (stored files used);
  - AF2_filtered recomputed with the same recipe from its archived trajectory (same 96 feature residues, 192 angles).
  - Work Scale from the sidecar features vs the A101 features: corr 0.9999 / 0.9990 `[o1]`.

### Kernel diagnostics `[o1]`, `o1_kernels.csv`
- **Rel. deviation from the complete graph after rescaling:**
  - h = 100: ≤ 8e-5 (ISO), 1e-4 (MoPrP dihedral), 8e-3 / 7e-4 (MoPrP Work Scale);
  - h = 10: 3e-3 – 8e-3 (ISO).
- **Collapsed to the complete graph:** ISO h ∈ {10, 100, 1000} and MoPrP h = 1000 (plus dihedral h = 100). On MoPrP the dihedral h = 10 kernel (rel. dev. 1.2e-2) gives rule-D results identical to 'complete' within 0.01 `[o5]`.
- **Inert or dropped:** ISO RMSD h = 0.01 has no edges, so it is inert (= γ = 0). MoPrP dihedral h = 0.01 connects ≤ 14 of 500 frames (c ≈ 6e9), so I dropped it.
- **RMSD h = 0.1 is effectively a nearest-structure graph.** 75–89% of pairs are nonzero, but the weights span e^−80 up to about 1e-5 (median degree about 1e-5 on TRI, 1e-9 on BI `[o8]`), so each frame's few nearest structures dominate.

### γ scaling, as in stk/lap
- γ_abs = γ·c, with c = median over fold-runs of (KL/S)/pen(w) at the γ = 0 rule-D pick (λ = 0 scoring, S > 0 picks). γ = 1 therefore makes the term equal KL/S at the MaxEnt pick.
- **ISO:** c is pooled over 36 fold-runs per system; for example c = 516.5 (TRI L_st), 177.6 (TRI complete), 520.6 (TRI RMSD h = 0.1 sym).
- **MoPrP:** for example 239 (MSAss complete) and 97.4 (Filtered complete). Filtered rests on its 2/6 folds with S > 0.
- Penalties are divided by their max absolute row sum; c absorbs this. All values are in `o7_tables.md` T-scale `[o3, o4]`.

### Pure OMC
- **Objective:** χ² + (c/S)·pen(w), no KL. This equals the brief's (1/S′)·pen with S′ = S/c, so the S grid lines up with the KL variant.
- **ISO:** exact simplex QP by a primal-dual active-set method. It agrees with FISTA to the QP gap and converges in 5–18 iterations (gap ≤ 6e-12). An entropic-dual approximation was rejected because it differs (rec 33.6 vs 39.9 at S = 1e-3) `[i0; scratch test]`.
- **MoPrP:** softmax L-BFGS cold-started from uniform at each S (stk 'pomcu').

### Solvers
- **ISO KL + OMC:** exact augmented dual in whitened coordinates. It agrees with the primal to 1e-8 and is 5× faster than the unwhitened dual.
- **MoPrP:** primal 500-logit L-BFGS (lap form).
- **Stationarity:** ISO max 3.0e-4 overall and 1.7e-4 at picks (χ² units). MoPrP: 6.1e-4 at picks; two non-candidate pure-OMC rows at ESS < 0.5% reach 0.05 and 0.22 `[o5]`.

### Compute-budget reductions (stated, not hidden)
- **ISO scalar grid:** seed 0 only (2 σ × 6 folds), h ∈ {0.01, 0.1, 1} + complete, and L_sym only for h ≤ 0.1.
- **3 seeds** for T1 and for the RMSD follow-up.
- **MoPrP L_sym / orig** run only for the MSAss Work Scale kernels and 'complete', plus the dihedral h = 0.1 L_sym follow-up.

## M1. ISO, main scalar-kernel grid `[o3, o5]` (seed 0; full cells in `o7_tables.md`; figure `F2_iso_heat.png`)

**Baseline, γ = 0, seed 0:**
- BI 98.28 (6/6) at both σ;
- TRI 59.69 (σ = 0.01) and 55.67 (σ = 0.02), with unassigned mass 28.8 / 34.0%.

The 3-seed baseline is BI 98.36 / 98.19 (18/18) and TRI 60.07 / 56.46 `[o2, o5]`.

**ISO_TRI rule D, paired Δ vs γ = 0 (σ = 0.01 | σ = 0.02):**

| kernel (lap form) | γ 0.1 | γ 0.3 | γ 1 | γ 3 | pure OMC | L_sym γ 1 |
|---|---|---|---|---|---|---|
| complete (h ≥ 10) | +0.1 \| +0.3 | −0.3 \| −0.3 | −1.6 \| −3.3 | −4.4 \| −7.0 | −1.0 \| +0.1 | (= lap) |
| Work Scale h 0.01 / 0.1 / 1 | −0.2…0.0 \| 0.0…−0.3 | −0.4…−1.1 \| −0.7…−0.9 | −2.4…−2.7 \| −3.4…−4.0 | −5.4…−5.9 \| −7.7…−8.2 | −0.5…−0.7 \| −0.5…−1.3 | −1.7…−2.0 \| −2.8…−2.9 |
| PyRosetta h 0.01 / 0.1 / 1 | −0.1…−0.3 \| 0.0 | −0.2…−0.4 \| −0.7…−1.1 | −1.9…−2.2 \| −3.5…−4.0 | −4.9…−5.1 \| −7.5…−8.1 | h 0.01: −24.5 (picks uniform); h 0.1/1: −1.0…−1.4 \| −0.9…−2.3 | −1.6…−2.0 \| −3.1…−3.5 |
| RMSD h 0.1 | +0.3 \| +0.9 | 0.0 \| +1.0 | +0.6 \| +1.6 | +1.4 \| +0.8 | +2.2 \| +2.8 | **+6.0 \| +3.1** |
| RMSD h 1 | +0.9 \| +1.6 | +1.2 \| +2.0 | +1.0 \| −0.5 | −1.1 \| −3.8 | +0.8 \| +0.4 | (3 seeds: −1.9 \| −2.8) |

**ISO_BI control (rule D, #≥ 95 of 6; σ = 0.01 | σ = 0.02):**
- **Pass at γ ≤ 1 everywhere:** 6/6, with minimum ≥ 95.0 at σ = 0.02.
- **Fail at γ = 3, σ = 0.02** for complete (93.2, 0/6), Work Scale (94.7 3/6, 93.7 0/6, 93.3 0/6) and PyRosetta (95.1 4/6, 94.2 1/6, 93.5 0/6).
- **Pure-OMC failures:**
  - PyRosetta h = 0.01 → 62.6 (0/6, rule D returns uniform p0) at both σ;
  - Work Scale h = 0.01 at σ = 0.02 → 80.3 (3/6).
- **RMSD kernels raise BI slightly** (+0.3 to +1.3) and pass everywhere `[o5]`.

**Reading.**
- Work Scale and PyRosetta are scalar (1-D) frame descriptors. Their Gaussian graphs smooth weights between frames that merely share mean protection or energy. For h ≥ 0.1 their results track the complete-graph (inverse-ESS) penalty within about 0.5 points.
- That penalty's attractor is uniform, and uniform is 31.1% recovery on TRI and 62.6% on BI `[o8]`. So γ ≥ 1 drags TRI toward unassigned frames: unassigned goes 28.8 → 31–36% (σ = 0.01) and 34 → 38–44% (σ = 0.02).
- This is the old sidecars' "OMC trails MaxEnt" result, and in this setup it is a pull toward uniform.

## M2. ISO follow-up: RMSD h = 0.1, L_sym, 3 seeds `[o3b, o5]` (3 seeds × 6 folds; `o7_tables.md` T-ISO-T1 rows `rmsd|h0.1`)

Reproduction: the seed-0 γ = 1 rows recomputed in o3b equal o3 exactly (840 rows, |Δrec| = 0) `[o5]`.

| | γ 0.1 | γ 0.3 | γ 1 | γ 3 | pure OMC |
|---|---|---|---|---|---|
| TRI σ 0.01 (γ=0: 60.07 ± 3.32, min 54.6) | 61.43 (+1.35) | 63.23 (+3.16) | 64.92 ± 2.04 (60.4) (+4.85 ± 0.62) | 65.80 ± 1.69 (61.6) (+5.73) | 67.12 ± 2.16 (63.1) (+7.05 ± 0.53) |
| TRI σ 0.02 (γ=0: 56.46 ± 3.69, min 48.7) | 59.26 (+2.79) | 61.10 (+4.63) | 61.83 (+5.36) | 63.69 (+7.22) | 65.50 ± 2.34 (60.2) (+9.04 ± 0.54) |
| BI σ 0.01, #≥95 /18 | 98.61 · 18 | 98.68 · 18 | 98.63 · 18 | 98.86 · 18 | 99.55 · 18 |
| BI σ 0.02, #≥95 /18 | 98.46 · 18 | 98.59 · 18 | 98.53 · 18 | 98.45 (min 95.8) · 18 | 99.56 · 18 |

- **TRI populations at pure OMC:** open 31.4%, unassigned 20.0% (vs 26.3 / 28.4 at γ = 0); ESS 12–13%.
- **Rules A and B agree:** pure OMC A 66.7, B 67.1 at σ = 0.01.
- **The MaxEnt-path oracle on TRI is 61.51 / 57.88. The L_sym path's oracle is 67.5 / 66.0** `[o5]`.

**Why I do not promote it:**
1. **It is specific to one bandwidth and one form.**
   - Same kernel in the lap form: 0.0 to +2.8 (seed 0).
   - RMSD h = 1 in L_sym: −1.9 / −2.8 over 3 seeds.
   - Neither Work Scale nor PyRosetta in L_sym helps (−1.6 to −3.5).
2. **Its coordinate is the label coordinate** `[o8]`.
   - TRI clusters are defined by Cα RMSD to the open/closed references: assigned frames have min-RMSD ≤ 1.00 Å, intermediates 1.00–2.02 Å.
   - In this kernel, closed frames have higher degree than open or intermediate frames (median log10 degree −4.4 vs −4.9).
   - The L_sym null vector √deg alone gives TRI 35.9 (closed 49.9%, unassigned 48.7%), against uniform 31.1 (37.3 / 60.7).
   - Part of the gain is therefore a density prior in the coordinate that scores recovery.
   - On BI the same attractor is worse (59.2, open 3.1%), yet the fit still improves BI (open 38.4–40.3%). So it is not only the attractor.
3. **It was found post hoc** as 1 of about 60 seed-0 cells. Seeds 1–2 are independent noise replicates and confirm it, but the selection of h = 0.1 and L_sym was not blind.
4. **It does not transfer to MoPrP.** Dihedral h = 0.1 in L_sym gives:
   - MSAss: 92.68–92.93 (Δ −0.2 to −0.5), pure OMC 91.66 = p0;
   - Filtered: 94.41–95.12 (Δ −0.5 to +0.2), pure OMC 96.07 = p0 `[o4 followup, o5]`.

## M3. MoPrP `[o4, o5]` (6 folds, rule D, λ = 0; figure `F3_moprp_heat.png`; `o7_tables.md` T-MoPrP)

**Baseline, γ = 0 (= stk s3 / selgci):**
- MSAss 93.14 ± 1.22 (minimum 91.7), ESS ≥ 50% in 3/6 folds, Folded/PUF1 96.8/2.83;
- Filtered 94.95 ± 3.12 (88.6), ESS ≥ 50% in 4/6, p0 picked 4/6.

**AF2_MSAss — every configuration is below γ = 0:**
- Best: 'orig' Work Scale h = 0.01 γ = 1, 92.97 (Δ −0.16 ± 0.18); dihedral h = 0.1 lap γ = 0.1, 92.81 (−0.33).
- γ = 1: 90.1–92.1 (Δ −1.0 to −3.0). γ = 3: 88.7–91.5 (−1.6 to −4.5).
- Pure OMC: 83.6–91.7 (−1.5 to −9.5).
- **The OMC term buys ESS:** ESS ≥ 50% in 5–6/6 folds at γ ≥ 1, against 3/6.
- **It moves weight into PUF1:** 2.83 → 4–6.3% at γ ≥ 1 (complete γ = 3: Folded 93.3, PUF1 6.27).
- **'lap', 'sym' and 'orig' agree in sign.** 'orig' is the least harmful (−0.2 to −1.9), and L_sym ≈ lap for h ≥ 1.

**AF2_filtered:**
- γ ≤ 3: 93.8–94.8 (Δ −0.2 to −1.1).
- **Positive cells**, each +1/−0 or +1/−1 folds:
  - Work Scale h = 0.01 γ = 3: 96.20 (+1.25 ± 1.24, p0 5/6);
  - Work Scale h = 0.1 γ = 3: 96.18 (+1.23);
  - pure OMC on Work Scale h = 0.01 and dihedral h = 0.1 (both 96.07 = p0 in 6/6, +1.13).
  - All of them come from rule D returning p0 (the ΔG prior, recovery 96.07) in the one fold where γ = 0 had left it (88.6 → 96.1). The OMC candidates there are flagged; their populations are not better.
- **Pure OMC elsewhere:** 89.4–93.8 (−1.1 to −5.5).

**Verdict.** On MoPrP the scalar-distance OMC never beats γ = 0 on its own merit. "Roughly tied" in the old sidecars becomes "tied at γ = 0.1, worse beyond" in the scaled-BV + KL-to-p0 setup.

## T1. Self-tuning OMC term on ISO `[o1, o3, o5]` (3 seeds × 6 folds per σ; figure `F1_T1_gamma.png`; `o7_tables.md` T-ISO-T1)

### Graphs `[o1]`
- **Self-tuning L_st:** the stk s1 primary kernel, rebuilt. λ_geo reproduces stk s1 exactly (BI 0.1158, TRI 0.088426). BI has 2 components, open | closed.
- **Sidecar graph:** `ISO_TRI_logpf_l2_knn20.npz` (stored distances = lnPF L2 to 4.7e-7), unweighted, L_sym. BI uses the induced subgraph.
- **Both graphs have the same 32,978 edges.** The comparison therefore isolates the weighting: self-tuning K̃ with α = ½ vs 0/1.
- **Null-vector attractors:** TRI 31.2 / 31.6 (≈ uniform); BI 62.7 / 61.8 `[o8]`.

### Rule D, λ = 0 (paired Δ vs γ = 0, mean ± SE over 18)

| | γ=0 | L_st γ 0.1 | 0.3 | 1 | 3 | pure | kNN L_sym γ 0.3 | 1 | pure |
|---|---|---|---|---|---|---|---|---|---|
| TRI σ 0.01 | 60.07 (54.6) | 61.06 (+0.99) | **61.79 (+1.71 ± 0.43)** | 61.53 (+1.46) | 60.48 (+0.41) | 61.04 (+0.97) | 61.76 (+1.69) | 61.43 (+1.35) | 61.16 (+1.09) |
| TRI σ 0.02 | 56.46 (48.7) | 58.24 (+1.78) | **58.92 (+2.45 ± 0.45)** | 58.73 (+2.26) | 56.83 (+0.37) | 58.15 (+1.69) | 59.29 (+2.82) | 58.04 (+1.57) | 58.33 (+1.87) |
| BI σ 0.01 (#≥95/18) | 98.36 · 18 | 98.84 · 18 | 98.94 · 18 | 99.16 · 18 | 99.17 · 18 | 99.72 · 18 | 98.84 · 18 | 99.07 · 18 | 99.67 · 18 |
| BI σ 0.02 | 98.19 · 18 | 98.76 · 18 | 99.13 · 18 | 99.38 · 18 | 99.46 · 18 | 99.67 · 18 | 98.96 · 18 | 99.15 · 18 | 99.70 · 18 |

- **ESS:** TRI 13 → 14–18% and BI 25–27 → 27–29%, rising with γ.
- **TRI unassigned (σ = 0.01):** 28.4 → 26.2–27.7%.
- **TRI minimum per fold** rises: 54.6 → 58.8–58.9 at γ = 0.3–1 (σ = 0.01).
- **Under λ_geo scoring (stk T3 diag)** the baseline reproduces stk: BI 96.40 (17/18) and 96.57 (18/18).
  - L_st γ = 1 gives BI 97.02 (17/18) / 97.43 (18/18) and TRI 58.08 → 61.14 (σ = 0.01).
  - Pure OMC lifts BI to 99.59 / 18/18.
  - Rule A degenerates to p0 under λ_geo as in stk (BI 62.6) `[o7]`.

### Answers to the T1 questions
- **Does it pass BI ≥ 95% at both σ?** Yes: 18/18 at every γ and in pure OMC (minimum ≥ 95.2) under λ = 0.
- **Do the self-tuning weights change the MaxEnt-vs-OMC ordering?** The ordering changes relative to the old sidecars: OMC is now ahead of MaxEnt on TRI by 1.5–2.8 points, where the sidecars had it behind by 3.4–4.9. The cause is the new setup (exact KL path, rule D, γ scaled to the MaxEnt pick), not the self-tuning weights. Unweighted kNN20 L_sym gives the same numbers within ±0.4 at every γ and σ.
- **Caveat:** on MoPrP the same self-tuning term cost MSAss 0.9–1.1 points (stk T2). ISO's benefit therefore does not carry over.

## Figures `[o6]`

- **`F1_T1_gamma.png`:** T1 rule-D recovery (± SD) and ESS vs γ, for self-tuning L_st and sidecar kNN20 L_sym, BI and TRI, both σ. Pure OMC is shown as open markers, with the BI 95% and ESS 50% lines.
- **`F2_iso_heat.png`:** ISO scalar-kernel grid (seed 0). Rows are metric × h, columns are γ / pure / L_sym. Colour is the paired Δ vs γ = 0, clipped at ±8. Text gives recovery plus BI #≥ 95 or TRI unassigned %.
- **`F3_moprp_heat.png`:** the same for MoPrP. Text gives recovery and the number of folds with ESS ≥ 50%.

## Caveats

- **Scalar grid on 1 seed.** The ISO scalar-kernel grid is seed 0 only (6 folds per σ). Only the T1 graphs and RMSD h = 0.1/1 L_sym have 3 seeds. Single-seed Δ within about ±1 is noise; the SE column in `o7_tables.md` is per-fold.
- **Penalty form.** L = D − K (as instructed) is not the sidecars' quartic, weight-weighted energy. 'orig' was run on MoPrP (MSAss Work Scale + complete) and agrees in sign and is milder. It was not run on ISO, because the quartic term has no exact dual.
- **Not the old run.** The old sidecars used MSE (unweighted) χ², transient selection and an unscaled 1/S strength. Here the χ² is σ-weighted, the γ is rescaled, and selection is by rule D on the full S path. The comparison is construction-in-new-setup, not a replication.
- **Rule D / λ = 0 throughout** (the stk carry-forward). T1 also reports λ_geo diag scoring.
- **Recovery is a label metric.** The RMSD-kernel result is partly label-aligned; see M2.

## Conclusion

- **On TeaA,** the sidecars' scalar-distance OMC is either harmful or neutral for Work Scale, PyRosetta and the complete graph (h ≥ 10 = inverse-ESS penalty). It pulls toward uniform and costs 2–8 points at γ ≥ 1, failing BI at γ = 3, σ = 0.02.
- **The exception is Cα-RMSD at h = 0.1 in the L_sym form.** It beats MaxEnt on TRI by 5–9 points over 3 seeds and passes BI. But it is bandwidth- and form-specific, it shares the coordinate that defines the recovery labels, and its MoPrP analogue does nothing.
- **The self-tuning OMC term (T1)** gives a small, robust gain on ISO (+1.5–2.8 TRI, +0.5–1.3 BI, BI 18/18). That gain is equally available from the sidecars' unweighted kNN20 graph.
- **On MoPrP** no OMC configuration beats γ = 0. The production recommendation is unchanged: γ = 0, rule D, ESS ≥ 50% as an acceptance check.
