# gci2: externally fixed Σ for the MoPrP GCI (λ_geo from ensemble geometry, ISO representation term, work-scale selector)

Read-only. Scripts live in this folder and run from the repo root with `.venv/bin/python`. Script tags:
- `[h0]` frame-weight loader
- `[h1]` T1
- `[h2]` T2
- `[h3]` T3
- `[h4]`/`[h4b]` T4
- `[h5]` T5

**Note.** The previous scratch folders (`gci/`, `calib/`, `resel/`) had been wiped from /tmp by the time this pass started. I recreated `c0_common.py`, `g0_common.py` and `g1_corrections.py` here from the copies read at the start of the session. They reproduce the earlier constants exactly `[g1]`:
- prior mean residual +0.217 / +0.230
- global δ −2.457 / −2.792
- bc/bh refit (wr2) 0.261/0 and 0.256/0

Throughout, figures are MSAss / Filtered.

**Setup.** Same inputs as before: `_featurise_a101_k109`, row-normalised map with trim 2, 14 × 15 = 210 cells, frame_uptake, target 97.1/2.4/0.5. The GCI triple uses a_n = 1ᵀR1/n and b_n = tr(R²)/n with Šidák t₂₁₀ = 3.94.
- **D_exp:** per-peptide pooled RMS σ, floored at 0.0586. In practice this is 0.0586 everywhere except 75–79 (0.0703).
- **κ = 1** throughout T4. Nothing is tuned on residuals unless a variant is explicitly labelled that way.

---

## T2 — λ_geo = 1/N_eff from the self-tuning kernel `[h2]` → `h2_t2.csv`, `h2_struct.csv`

**Method.**
- F is 500 frames × 210 cells. d is the Euclidean distance between frames.
- σ_i = mean distance to the k nearest neighbours, multiplied by c.
- Kernel and walk: α = ½ density normalisation, then row normalisation, then the lazy walk P = (I + P₀)/2.
- C_mean = F̃ᵀ G F̃ / N², where G = I + Σ_{slowest m}(g(μ_k) − 1)φ_kφ_kᵀ and g = (1 + μ′)/(1 − μ′).
- Inflation = tr(C_mean) / tr(C_ens/N); λ_geo = inflation/N; N_eff = N/inflation.
- **Truncation does not matter:** m = 10, 50 and all agree to within 1–5%.

**Main table** (lazy, m = all, default BV 0.35/2.0):

| ensemble | k | c = 0.5 | **c = 1** | c = 2 | top-eigvec N_eff (c = 1) |
|---|---|---|---|---|---|
| MSAss | 10 | N_eff 0.0005 (λ ≫ 1, graph disconnected) | **N_eff 1.9, λ 0.54** | 14.9, λ 0.067 | 1.4 |
| MSAss | 20 | 0.04 | **4.0, λ 0.25** | 22.5, λ 0.044 | 2.9 |
| Filtered | 10 | 0.49 | **11.1, λ 0.090** | 55, λ 0.018 | 6.8 |
| Filtered | 20 | 1.2 | **16.3, λ 0.061** | 80, λ 0.013 | 10.0 |

- **c = 0.5 is degenerate.** Up to 86 eigenvalues sit above 0.9, so the walk is near-reducible. I do not use it.
- **Non-lazy walk:** N_eff is about 2× larger (MSAss 3.7–8.1; Filtered 23–34 at c = 1). Note that the lazy walk carries an intrinsic ×3 floor: g = 3 when μ = 0.
- **With the T1 high-confidence parameters (hc90, wr2):** MSAss N_eff 3.9–7.7, Filtered 25–35 (c = 1, lazy).

**Structural bracket** (cluster-robust "sandwich" variance of the mean, with clusters treated as units):

| labels | MSAss N_eff (λ) | Filtered N_eff (λ) |
|---|---|---|
| rule-based, 5 / 4 clusters | 17.3 (0.058) | 37.3 (0.027) |
| k-medoids, k = 3 / 5 / 8 / 12 / 20 | 13.6 / 17.5 / 24 / 31 / 44 | 15.3 / 20.4 / 24 / 30 / 39 |
| cluster-count floor 1/K | λ ≤ 0.20 | λ ≤ 0.25 |

**Adopted values** (used in T4 and T5). λ_geo is the geometric mean of k = 10 and 20 at c = 1, lazy:
- **MSAss: λ_geo = 0.366, N_eff ≈ 2.7.** Bracket 0.044–0.54 over k × c ∈ {1, 2}; structural 0.058.
- **Filtered: λ_geo = 0.074, N_eff ≈ 13.5.** Bracket 0.013–0.09; structural 0.027.

**Interpretation.**
- In predicted-uptake space the 500 MSAss frames behave like about 3–20 independent draws, and Filtered like about 10–40. That is 1/N_eff ≈ 25–180 × 1/N.
- This is the same order as the residual-fitted λ̂ from the previous pass (MSAss global 0.17, smooth5 0.04).
- So a large λ is justified by geometry, not only by residual absorption. But the MSAss Σ then becomes nearly rank-1 (spectral ratio 0.87 at ×1), and the S_n and Q_n thresholds are not trustworthy there. T_n (Šidák) remains valid.

## T4 — externally fixed Σ = D_exp + λ·C_ens `[h4]` → `h4_gci.csv`, `h4_path.csv`; `[h4b]` → `h4b_flags.csv`, `h4b_kappa.csv`

**Setup.**
- **s\*** = argmin |S_n(prior)| under the λ_geo × 1 Σ: global 1.15 / 1.05, smooth5 1.10 / 1.10.
- **bcbh_all** is the previous all-cell refit. **bcbh_hc90/hc85** come from T1 (wr2).
- **Codes:** letters = failed statistics (t, s, q); number = FWER-flagged cells.

Results at λ_geo × 1:

| ens | corr | prior | Folded | NNLS_in | NNLS_cv |
|---|---|---|---|---|---|
| MSAss | none | tsq65 | tsq67 | tq9 | tsq72 |
| MSAss | global 1.15 | tq16 | tq15 | tq7 | tq56 |
| MSAss | smooth5 1.10 | tq13 | tq15 | tq10 | tq16 |
| MSAss | bcbh_all | tq25 | tsq27 | tq16 | tq37 |
| MSAss | bcbh_hc90 | tsq52 | tsq53 | tq7 | tq74 |
| MSAss | bcbh_hc85 | tsq24 | tsq25 | tq14 | tq36 |
| Filtered | none | tsq86 | tsq86 | tsq67 | tsq86 |
| Filtered | global 1.05 | tq26 | tq26 | tq21 | tq29 |
| Filtered | smooth5 1.10 | tq20 | tq21 | tsq14 | tsq25 |
| Filtered | bcbh_all | tsq34 | tsq36 | tsq27 | tsq37 |
| Filtered | bcbh_hc90 | tsq61 | tsq61 | tsq48 | tsq73 |
| Filtered | bcbh_hc85 | tsq35 | tsq37 | tsq30 | tsq37 |

**Sensitivity.**
- **Zero of the 2,400 model × setting rows pass** `[h4]`. That covers λ_geo × 0.5, × 1, × 2, λ_struct, + rep_mult (+5/N·C), + rep_floor (+0.0067² on the diagonal), every correction, both s values, and all 25 models (prior, Folded, NNLS_in, NNLS_cv and the 21-point path).
- The bracket moves only the flag counts. For example, MSAss global prior has 9–24 flagged cells across settings; Filtered smooth5 prior has 19–20.
- **Acceptance interval along w(t): empty in every configuration** `[h4]`.
- The representation-term variants change thresholds by under 1%.

**Persistent flags** (prior at s\*, × 1) `[h4b]`:

| | 60–74 | 83–90 | 95–101 | other |
|---|---|---|---|---|
| MSAss global | 8 cells | 5 | – | 11–26 ×2 |
| MSAss smooth5 | 3 | 4 | 6 | |
| Filtered global | 8 | 5 | – | 11–26, 68–75, 27–31 |
| Filtered smooth5 | 6 | 5 | 8 | |

- **60–74 is flagged in every configuration**, including all bcbh refits: 9–14 of its 15 cells, the most under hc90.
- In-sample NNLS reduces 60–74 to 1–4 cells but keeps 83–90 (3–5 cells).

**How much is missing.** This is the smallest κ on D_exp at which each model passes, holding λ_geo fixed `[h4b]`:
- prior passes at κ = 5.2 / 6.7 (global) and 3.9 / 4.7 (smooth5). That corresponds to an extra homoscedastic model-error SD of **0.10–0.14 D**.
- Folded needs about the same κ (5.5 / 6.7 global).
- NNLS_in needs κ = 3.3–4.7.
- Uncorrected ("none"), the prior never passes on the grid up to κ = 400, because S_n is +40 / +55.
- bcbh_hc90 needs κ ≈ 190–280.

## T5 — work-scale selector `[h5]` → `h5_candidates.csv`, `h5_path.csv`, `h5_select.csv`, `F4_work_vs_recovery.png`

**Definition.** Work scale = R·T·|mean_res lnPF(w) − mean_res lnPF(u)| (`scoring.py:100`). Settings:
- T = 298 K. The function's default is 300 K; this does not change any ranking.
- lnPF is per-residue, frame-weighted, over the 97 featurised residues, with BV 0.35/2.0 (the fitted model). Additive corrections cancel.

**Pool** `[h0]`:
- ex2_fresh MaxEnt runs: 84 runs = 2 ensembles × 2 split types × 3 splits × 7 strengths.
- Policy P2 (no running_best, step ≥ 100) leaves 4,383 states.
- Weights were reloaded from the HDF5 histories with the repo's candidate ordering. Recovery matches `candidates.csv` to 2e-8.

**Σ used:**
- "fixed" = T4 at λ_geo × 1.
- "prior-floor" = κ_p·D_exp + λ_geo·C, with κ_p the smallest value at which the uniform prior passes (5.2 / 6.7). This variant is residual-tuned and is included only for contrast.

Recovery of the work-scale-minimising state, per replicate (mean ± SD over 3 splits):

| ens | split | WS-min recovery | ESS | WS-min passes GCI (fixed / prior-floor, global s\*) | val-wMSE selector (P2) recovery | Spearman(WS, rec) |
|---|---|---|---|---|---|---|
| MSAss | seq-cluster | **73.13 ± 0.13** | 99.9% | 0/3 / 3/3 | 40.9 ± 10.1 | −0.98 |
| MSAss | spatial | **73.21 ± 0.16** | 99.9% | 0/3 / 3/3 | 29.2 ± 5.4 | −1.00 |
| Filtered | seq-cluster | **80.75 ± 0.04** | 99.9% | 0/3 / 3/3 | 23.0 ± 13.5 | −0.99 |
| Filtered | spatial | **80.87 ± 0.04** | 99.9% | 0/3 / 3/3 | 19.1 ± 8.2 | −0.99 |

**What the work-scale minimiser picks.** It is always the strength = 1 state (the weakest data term), at late steps. Its recovery equals the prior's (73.3 / 80.8).

**Along the path.** Work scale is linear in w, so its minimum is t = 0, the prior. Recovery falls monotonically with work scale: path WS runs 0–0.69 / 0–0.50 kJ/mol while recovery runs 73 → 52 / 81 → 52.

**GCI acceptance:**
- **Fixed Σ:** 0 of 4,383 candidates pass, under both "none" and "global".
- **Prior-floor Σ, global s\*:** 836 / 1,172 candidates pass. Their recovery spans **63–73 (MSAss) and 51–81 (Filtered)**, with work scale ≤ 1.27 / 0.60 kJ/mol. These are strengths 1–10 (MSAss) or 1–100 (Filtered). The whole path passes (recovery 52–73 / 52–81).
- **The existing validation selector never lands in an acceptance set:** 0 of 12 replicates under global.
- **Prior-floor Σ, uncorrected forward model ("none"):** the prior and the work-scale minimisers *fail*, because the bias is still there. The accepted MSAss candidates have recovery 15–40, and the accepted path is t ≥ 0.75 (recovery 2.6–25). No Filtered candidate passes.

## T3 — representation-error term from ISO_BI `[h3]` → `h3_t3.csv`, `h3_size.csv`

| residual | rms | max abs | λ: ‖r‖²/tr C (×1/N) | λ: off-diagonal moment (×1/N) | share of ‖r‖² in top-5 C modes | smallest λ with size ≤ 5% (×1/N), σ = .005 / .01 / .02 / .05 | diagonal floor σ_rep: rejection at σ = .005 / .01 / .02 / .05 |
|---|---|---|---|---|---|---|---|
| target − F·w_true (40/60) | 0.0067 | 0.040 | 1.95 | 0.49 | 1.4% | **5 / 3 / 2 / 2** | 0.1 / 1.9 / 1.5 / 4.4% |
| target − F·w_NNLS (best achievable; 40.1% open, ESS 159) | 0.0022 | 0.017 | 0.21 | ≈0 | 0.02% | 1 / 0.5 / 0.5 / 0.5 | 0.3 / 1.1 / 1.3 / 4.1% |

**Which form transfers.**
- The representation error is **not aligned with C_ens**: under 2% of it lies in the top-5 modes. A C-multiplier controls size only by inflating the diagonal indirectly, so its value (2–5/N) depends on how C is scaled in the ISO system.
- **The diagonal floor σ_rep ≈ 0.007 (true weights) or 0.002 (best weights) is in uptake units and controls size at every σ.** This is the transferable form.
- **For MoPrP neither form matters.** σ_rep² is about 1.3% of 0.0586², and +5/N·C adds 3% / 13% to λ_geo. T4 confirms neither changes any pass/fail.
- ISO measures frame-subset error under a forward model that is correct by construction. It cannot represent the BV model error that dominates MoPrP: about 0.10–0.14 D `[h4b]`, roughly 15–20 × σ_rep.

## T1 (secondary) — bc/bh refit on high-pLDDT cells `[h1]` → `h1_t1.csv`, `h1_params.json`

**Which peptides qualify.** pLDDT is taken from the reference PDB CA B-factors. A peptide qualifies if all of its trimmed residues meet the threshold:
- **≥ 85:** 11 peptides. Excluded: 68–75, 82–101, 95–101.
- **≥ 90:** 5 peptides (5–10, 27–31, 32–45, 33–40, 83–90). 60–74 drops out because residues 67–74 have pLDDT 85–88.
- **≥ 95:** none. Here I used the weighted variant.

**Fits.** Multi-start L-BFGS with bounds bc ∈ [0, 2], bh ∈ [0, 10]. The table shows MSAss | Filtered.

| thr | objective | bc | bh | objective at fit / at default | prior mean residual (all 210 cells) | 60–74 mean residual | flagged cells, all (nominal Σ) | S_n on high-confidence cells |
|---|---|---|---|---|---|---|---|---|
| 85 | wr2 | 0.262 \| 0.256 | 0.32 \| 0.31 | 1.49 / 2.81 | .071 \| .072 | .317 \| .316 | 67 \| 66 | 18.4 \| 19.6 |
| 85 | MSE | 0.227 \| 0.222 | 1.16 \| 1.14 | .031 / .066 | .060 \| .060 | .321 \| .321 | 70 \| 66 | 13.0 \| 13.9 |
| 90 | wr2 | 0.197 \| 0.181 | **5.08 \| 5.43** | .53 / 1.13 | .157 \| .162 | .500 \| .518 | 89 \| 88 | 5.1 \| 6.2 |
| 90 | MSE | 0.192 \| 0.177 | 4.53 \| 4.91 | .012 / .032 | .139 \| .145 | .472 \| .492 | 85 \| 85 | 1.2 \| 2.5 |
| 95 (weighted) | wr2 / MSE | 0.356 / 0.311 \| 0.348 / 0.304 | **0 \| 0** | | .17 / .12 | .43 / .37 | 95 / 80 | – |

**Reading.**
- **bh no longer vanishes once the low-pLDDT peptides are excluded.** It is 0.3–1.2 at ≥ 85 and 4.5–5.4 at ≥ 90. It returns to 0 only in the pLDDT-weighted ≥ 95 fit, which down-weights everything and is dominated by 60–75/75–84.
- At ≥ 90 the fit rests on 5 peptides and is driven by 83–90, which needs more protection. These parameters then over-protect the rest: 60–74 residual rises to +0.50 and the mean residual to +0.16.
- **So (bc, bh) is not identified. bh = 0 is a consequence of which peptides are included, not a property of the data.**
- Single-structure variant: dropped, as instructed.

---

## Conclusion

1. **With Σ fixed externally, the prior does not pass.** Σ = D_exp (floored) + λ_geo·C_ens, with λ_geo from the kernel random walk (N_eff ≈ 3 MSAss, ≈ 13 Filtered). Nothing passes: not the prior, not Folded, not in-sample or held-out NNLS, and no point on the path. This holds for every correction and across λ_geo × 0.5–2, λ_struct, and both ISO representation terms `[h4]`. The data need about 0.10–0.14 D of extra, model-error SD per cell `[h4b]`. That is 2–2.5 × the experimental σ, and about 15–20 × anything ISO attributes to frame-subset error `[h3]`.
2. **bh does not vanish once low-pLDDT residues are excluded** `[h1]`. It is 0.3–1.2 (≥ 85) and about 5 (≥ 90). It is not identified: its value is set by which of 60–74/68–75 (pull bh down) and 83–90 (pull it up) are included.
3. **The 60–74 flag survives everything.** It appears for every correction, every bcbh variant (worst under hc90), every λ, and for prior and Folded alike. 83–90 (over-predicted) and, under smooth5, 95–101 survive as well `[h4b]`.
4. **The work-scale selector lands on the prior.**
   - Recovery: **MSAss 73.13 ± 0.13 (sequence_cluster) and 73.21 ± 0.16 (spatial); Filtered 80.75 ± 0.04 and 80.87 ± 0.04.** ESS is 99.9%, at strength = 1 `[h5]`.
   - Within each replicate, work scale and recovery are almost perfectly anti-ranked (ρ ≈ −0.99).
   - Its picks are inside the acceptance set only under the residual-tuned prior-floor Σ with the global bias correction. That set spans recovery 63–73 / 51–81, so the GCI does not discriminate there.
   - Under the uncorrected model even the prior-floor Σ rejects them.
   - The current validation-MSE selector (29–41 / 19–23) falls outside every acceptance set.
