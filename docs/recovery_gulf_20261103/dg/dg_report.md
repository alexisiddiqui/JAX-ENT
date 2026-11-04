# dg: per-segment ΔG_op restraints for the MoPrP AF2 ensembles

This was a read-only analysis. All scripts are in this folder and run from the repo root with `.venv/bin/python`. Every number below is tagged with the script that produced it:

| tag | script |
|---|---|
| `[d0]` | `d0_common.py` (constants, Table 1, scaled BV) |
| `[d1]` | `d1_kint_check.py` |
| `[d2a]` | `d2a_native_contacts.py` |
| `[d2]` | `d2_t1_local_open.py` |
| `[d3]` | `d3_t2_reweight.py` |
| `[d4]` | `d4_t3_hdx.py` |
| `[d5]` | `d5_t4_joint.py` |
| `[d6]` | `d6_tables.py` (formatting only) |

- **Forward model:** the median-scaled BV model, `bc,bh = s·(0.35, 2.0)` with s = 0.571 (MSAss) and 0.565 (Filtered). It is recomputed in `[d0]` and is identical to `pop/p1_models.json` `scale|matched`.
- **Secondary model:** the shift model, with δ = −2.457 (MSAss) and −2.792 (Filtered) `[d0]`.
- Figures are given as MSAss / Filtered.

## 0. Two corrections to the brief (they change which segments are usable)

**1. Numbering is seq = mouse − 122, not mouse − 104.**
- The archived 101-residue construct is mouse 123–223. `key_residues.MD` states "PDB residue number + 122 = original residue number".
- Residue identity confirms it. In `moprp.seq`: M128 → 6, I138 → 16, Y154 → 32, Q159 → 37, V160 → 38, Y161 → 39, R163 → 41, H186 → 64, C178 → 56, C213 → 91. Under −104 the same positions would be D, Y, N, D, C, V, I.
- Every Table 1 segment is one of our 14 HDX-MS peptides:

| mouse | seq | peptide |
|---|---|---|
| 127–132 | 5–10 | `YMLGSA` |
| 133–148 | 11–26 | |
| 149–153 | 27–31 | |
| 154–167 | 32–45 | |
| 155–162 | 33–40 | |
| 182–196 | 60–74 | |
| 190–197 | 68–75 | |
| 197–201 | 75–79 | |
| 197–204 | 75–82 | |
| 197–206 | 75–84 | |
| 204–224 | 82–101 | |
| 205–212 | 83–90 | |
| 217–223 | 95–101 | |

  The one exception is 133–143, which is a Table 1 segment (seq 11–21) without a matching peptide; the nearest peptide is 11–24.
- **Consequence:** 204–224, 205–212 and 217–223 lie inside the construct. I added 205–212 and 217–223 to the restraint set. 204–224 has two slow phases and is reported but not restrained.

**2. Full Table 1 and the slow-residue list.**
- Table 1 and Table 2 (the HDX-NMR slow-exchanging residues used for each segment's k_int) were read from the paper: PMC4646174 (Moulick, Das & Udgaonkar, JBC 2015, 290:25227).
- ΔG_op = RT ln(k_int/k_slow), with RT = 0.5925 kcal/mol `[d0]`. This reproduces every ΔG in the brief to ±0.01.
- Extra slow phases:

| segment | k_int (s⁻¹) | k_slow (s⁻¹) | ΔG (kcal/mol) | K |
|---|---|---|---|---|
| 204–224, slow phase 1 (2 NH) | 8.1e-3 | 3.3e-4 | 1.90 | 0.041 |
| 204–224, slow phase 2 (5 NH) | 8.1e-3 | 1.4e-5 | 3.77 | 0.0017 |
| 205–212 | 6.8e-3 | 2.1e-6 | **4.79** | 3.1e-4 |
| 217–223 | 5.3e-3 | 1.1e-4 | 2.30 | 0.020 |

- 205–212 is the segment the paper equates with global unfolding.
- 190–197 and 197–201 have no slow phase: they exchange at k_int.

## 1. k_int cross-check: 133–143 is **9e-4 s⁻¹ (as printed)** `[d1]` → `d1_kint.csv`

| segment | slow residues (seq) | paper k_int (s⁻¹) | ours: HDXrate pD 4.0, 298 K | ratio ours/paper | exPfact pH 4.4 ratio |
|---|---|---|---|---|---|
| 127–132 | M6 | 4.0e-3 | 5.03e-3 | 1.26 | 4.02 |
| **133–143** | **I16** | **9e-4** | **1.11e-3** | **1.23** (vs 9e-3: 0.12) | 3.91 |
| 133–148 | I16, R25, Y26 | 5.8e-3 | 7.40e-3 | 1.28 | 3.36 |
| 149–153 | 27–31 | 1.0e-2 | 1.25e-2 | 1.25 | 3.72 |
| 154–167 | 32, 34, 37, 38, 39, 41 | 3.9e-3 | 4.63e-3 | 1.19 | 3.78 |
| 155–162 | 37, 38, 39 | 3.8e-3 | 4.76e-3 | 1.25 | 3.99 |
| 182–196 | Q63 | 5.5e-3 | 6.95e-3 | 1.26 | 4.03 |
| 197–204 / 206 | 80, 81, 82 | 3.1e-3 | 3.93e-3 | 1.27 | 3.82 |
| 204–224 | 85, 87, 89, 91, 94, 96, 97 | 8.1e-3 | 1.01e-2 | 1.24 | 3.83 |
| 205–212 | 85, 87, 89 | 6.8e-3 | 8.31e-3 | 1.22 | 3.51 |
| 217–223 | 97, 100, 101 | 5.3e-3 | 8.15e-3 | 1.54 | 3.50 |

**The paper's k_int column is on our HDXrate pD 4.0 scale, up to a uniform factor of 1.26.**
- The median over the other 11 segments is 1.26, with a range of 1.19–1.28. The only outlier is 217–223 at 1.54, because Y224 lies outside the construct.
- This uniform factor shifts every ΔG by only 0.14 kcal/mol.
- exPfact pH 4.4 is about 3.8× too fast, which agrees with the earlier finding.
- Ile138 alone gives 1.11e-3 / 1.26 = **8.8e-4 s⁻¹**. So 9e-4 is correct and 9e-3 is ruled out (that would need a ratio of 0.12).

**Resulting values for 133–143:** ΔG_op = 0.92 kcal/mol, K = 0.21, p_open = 0.17.
- One caveat: the paper's 133–143 range does not match an MS peptide. The ΔG is a single-amide value (Ile138).
- It lies well below the PUF1 band (2.2). I therefore keep it out of the PUF1-type average used in the alternative target (§3).

## 2. T1: segment → region → feature → threshold, and uniform open fraction vs K_op `[d2]` → `d2_t1.csv`, `d2_indicators.npz`

**Data provenance.**
- The per-frame features are the 11 features in `MoPrP_unfolding_spec.json`, median-centred per ensemble. They were read from `_..._AF2_test/clusters/*/..._all_frames_features.csv`.
- Frame order was checked two ways:
  - Cluster labels equal those in `frame_to_cluster.csv`.
  - Recomputing `compactness_core` and `compactness_a1` from the archived xtc matches the CSV to 1e-15 `[d2a]`.
  - The xtc sha256 equals the featurisation manifest `[d2a]`.
- **What the features cover:**
  - Dihedral features exist only for β1 (seq 5–10), β2 (38–41) and α1 (22–31).
  - The only α2/α3 feature is the internal RMSD of the whole α2–α3 core.
  - No rule-spec feature resolves the β1–α1 loop (11–21), the α2 C-terminus or the α3 termini.

I therefore used three indicator families.

**A — rule-spec features** (thresholds taken from `MoPrP_rules_spec.json` transforms):

| region | "open" if | segments |
|---|---|---|
| β1 | NOT `b1_strand_intact` (φ ∉ [−17, 15] or ψ ∉ [−21, 30]) | 127–132 |
| α1 | NOT `a1_helix_intact` (\|φ\| or \|ψ\| > 10) OR `displacement_a1_vs_a3` > 0.75 (`a1_displacement`) | 133–148 (featured part 22–26; 2 of its 3 slow residues), 149–153 |
| β2 | NOT `b2_strand_intact` (\|φ\| or \|ψ\| > 21) OR `displacement_b2_vs_core` > 0.25 (`b2_displacement`) | 154–167, 155–162 (the 32–37 loop is not featured) |
| core | `compactness_core` > 0.5 Å OR `extreme_displacement`. **The rules spec has no compactness_core threshold.** I chose 0.5 Å because it is above the Filtered maximum (0.23) and the MSAss 95th percentile (0.15). | 182–196, 197–204, 197–206, 204–224, 205–212, 217–223 |
| — | not featured | 133–143 (excluded from A) |

**B — per-segment native-contact fraction Q_seg.**
- Native contacts are CA pairs with |i − j| ≥ 3 and d < 8 Å in `MoPrP_max_plddt_4334.pdb`. A contact counts as formed if d < d_ref + 1.5 Å. Every residue in the segment is included.
- Open if Q < 0.7 (B70); a Q < 0.5 variant is in the CSV.
- **This is the only family that resolves each segment separately** `[d2a]`.

**C — scaled-BV-implied K.**
- Per frame, K_BV,f = Σ_r k_int,r·exp(−lnPF_rf) / Σ_r k_int,r over the Table 2 slow residues. This is the EX2 identity K_op = 1/PF.
- It is reported as ΔG_BV = −RT ln⟨K_BV⟩ under uniform weights.
- **I report C as a diagnostic only.** As a restraint it collapses ESS to about 0.2% (`[d3]`, rows C).

**Uniform-weight open fraction vs p_exp = K/(1 + K)** (ratio = open / p_exp; "within a factor of 2" means 0.5–2):

| seg (mouse) | seq | region (A) | ΔG_exp | p_exp | MSAss A | MSAss B70 | Filtered A | Filtered B70 | ΔG_BV scaled, MSAss / Filtered |
|---|---|---|---|---|---|---|---|---|---|
| 127–132 | 5–10 | β1 | 1.95 | 0.036 | 0.216 (6×) | 0.134 (3.7×) | 0.150 (4.2×) | 0.070 (1.9×) | 1.19 / 1.57 |
| 133–143 | 11–21 | – | 0.92 | 0.174 | – | 0.062 (0.36×) | – | 0.004 (0.02×) | 0.80 / 0.84 |
| 133–148 | 11–26 | α1 | 2.35 | 0.019 | 0.050 (2.7×) | 0.052 (2.8×) | 0 | 0 | 2.17 / 2.25 |
| 149–153 | 27–31 | α1 | 2.92 | 0.0071 | 0.050 (7×) | 0 | 0 | 0 | 2.31 / 2.34 |
| 154–167 | 32–45 | β2 | 3.10 | 0.0054 | 0.074 (14×) | 0.094 (18×) | 0.022 (4.1×) | 0.020 (3.7×) | 1.47 / 1.62 |
| 155–162 | 33–40 | β2 | 3.08 | 0.0055 | 0.074 (13×) | 0.088 (16×) | 0.022 (4.0×) | 0.016 (2.9×) | 1.48 / 1.66 |
| 182–196 | 60–74 | core | 3.25 | 0.0042 | 0.036 (8.6×) | 0.016 (3.8×) | 0 | 0 | 2.29 / 2.59 |
| 197–204 | 75–82 | core | 1.98 | 0.034 | 0.036 (1.1×) | 0.012 (0.35×) | 0 | 0 | 2.40 / 2.79 |
| 197–206 | 75–84 | core | 2.23 | 0.023 | 0.036 (1.6×) | 0.012 (0.53×) | 0 | 0 | 2.40 / 2.79 |
| 204–224 s1 / s2 | 82–101 | core | 1.90 / 3.77 | 0.039 / 0.0017 | 0.036 (0.9× / 21×) | 0.016 | 0 | 0 | 2.48 / 2.71 |
| 205–212 | 83–90 | core | **4.79** | 0.0003 | 0.036 (117×) | 0.016 (52×) | 0 | 0 | 2.22 / 2.35 |
| 217–223 | 95–101 | core | 2.30 | 0.020 | 0.036 (1.8×) | 0.012 (0.59×) | 0 | 0 | 1.46 / 1.51 |

**Is the uniform ensemble already consistent with the per-segment ΔG_op? No.** `[d2]`

- **N-subdomain is too open** (β1, α1, β2):
  - MSAss is 3–18× too open. β2 is the worst at 13–18×, because the 10 PUF2, 11 PUF3 and 15 unfolded frames all open β2.
  - Filtered is 2–4× too open in β1 and β2.
  - **Rule family A coincides with the state labels by construction.** β1 is open in every non-Folded frame and in no Folded frame (MSAss: PUF1 72/72, PUF2 10/10, PUF3 11/11, unfolded 15/15). So "too open in β1" is the same statement as "PUF1 + PUF2 + PUF3 + U = 21.6% vs 3.6%".
- **C-terminal α2/α3 has no local opening anywhere.**
  - Filtered has zero open frames in every α2/α3 segment, under both A and B. It cannot reproduce K_op = 0.004–0.04 for 182–196, 197–206 or 217–223 under any reweighting.
  - In MSAss the α3 segments are "open" only in the 15 unfolded frames (plus 3 PUF3 under A). The apparent agreement within a factor of 2 for 197–204, 197–206 and 217–223 comes from global unfolding, not local fraying. The same frames make 205–212 52–117× too open. The paper puts 205–212 at the global ΔG of 4.8.
- **133–143 (Ile138, ΔG 0.92) is too closed:** 0.36× in MSAss and 0.02× in Filtered.
- **Wrong order.** Experimentally, α3 N- and C-termini (2.0–2.3) open more easily than β2 (3.1). In the AF2 ensembles every α3-open frame is also β2-open. The required ordering, p(α3 termini) > p(β2), is therefore **infeasible by reweighting** for both ensembles.
- **The scaled-BV-implied ΔG is compressed** to 0.8–2.8 kcal/mol, against an experimental range of 0.9–4.8:
  - β2: 1.5 vs 3.1
  - 205–212: 2.2 vs 4.8
  - 217–223: 1.5 vs 2.3
  - The order of stability is not reproduced.

## 3. T2: restrained reweighting (no BV, no uptake) `[d3]` → `d3_t2.csv`, `d3_weights.npz`; achieved fractions `[d6]` → `d6_tables.md`

**Methods.**
- **Exact MaxEnt** (dual, w ∝ u·exp(λ·O)): indicator rows that are identical are merged with a geometric-mean target. Rows with no open frame are dropped as infeasible.
- **Soft:** minimise Σ_seg (ln p_seg − ln p_exp)² + KL(w‖u)/S, over one row per segment.
- **Segment sets:**
  - S9 = the brief's nine segments, renumbered.
  - S11 = S9 + 205–212 + 217–223.
  - SN = N-subdomain segments only.

**Targets:**
- ORIG = 97.1 / 2.4 / 0.5.
- **ALT1** (brief's rule): PUF1 = mean K over PUF1-type {127–132, 133–148, 197–204, 197–206, 217–223} = 2.72%; PUF2 = mean K over PUF2-type {149–153, 154–167, 155–162} = 0.60%; Folded 96.7%.
- **ALT2** (hierarchical: a PUF1 segment is open in PUF1 + PUF2 + U; U = K(205–212)): Folded 97.3 / PUF1 2.11 / PUF2 0.57 / U 0.03%.
- ALT1 and ALT2 are 98.9 and 98.5 recovery against ORIG `[d0]`. **The per-segment data independently reproduce the whole-construct target.**

| ensemble | restraint | ESS | Folded | PUF1 | PUF2 | PUF3 | U | PUF2-like | rec ORIG | rec ALT1 | rec ALT2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MSAss | uniform | 100% | 78.4 | 14.4 | 2.0 | 2.2 | 3.0 | – | 73.3 | 74.1 | 73.1 |
| MSAss | A S11 exact† | 83.9% | 96.4 | 2.77 | 0 | 0.25 | 0.60 | – | **91.7** | 91.5 | 91.8 |
| MSAss | A S11 soft, S = 1–1000 | 83.9–84.1% | 96.3–96.4 | 2.8–2.9 | 0 | 0.19 | 0.60 | – | 91.8 | 91.6 | 91.9–92.0 |
| MSAss | A SN soft, S = 100 | 83.9% | 96.4 | 2.83 | 0 | 0.35 | 0.44 | – | 91.8 | 91.6 | 91.9 |
| MSAss | B70 S11 soft, S = 100 | 44.4% | 82.9 | 16.1 | 0.29 | 0 | 0.67 | – | 77.8 | 78.6 | 77.4 |
| Filtered | uniform | 100% | 84.8 | 13.2 | 1.8 | – | – | 0.2 | 80.8 | 81.8 | 80.3 |
| Filtered | A (any set) exact = soft | 90.8% | 96.2 | 3.28 | 0.34 | – | – | 0.17 | **96.1** | 96.4 | 95.4 |
| Filtered | B70 S11 soft, S = 100 | 92.5% | 87.9 | 10.0 | 1.88 | – | – | 0.21 | 84.5 | 85.5 | 84.0 |

All populations are in %.

**Reading the table.**
- † **MSAss A is jointly infeasible.**
  - α1-open frames (PUF3 + U) are a subset of β2-open frames, but p_exp(α1) = 0.012 > p_exp(β2) = 0.0054.
  - Core-open frames (U + 3 PUF3) are also β2-open.
  - The exact solver therefore returns a least-violating compromise (max error 3e-3). Achieved values are β1 0.036, α1 0.008, β2 0.008, core 0.007.
  - **PUF2 goes to 0 because the β2 budget is consumed by the PUF3/U frames that the α1 and core restraints need.**
- **Filtered A:** only β1 and β2 are feasible. Achieved values are β1 0.036 and β2 0.0054, both exact. α1 and core stay at 0.
- **The soft results are insensitive to S** over 1–1000. The restraint term dominates.
- **Restraint family B (Q-contacts) gives a different state picture.**
  - Only 31 of 72 MSAss PUF1 frames (26/66 Filtered) are β1-open by Q.
  - Matching β1 by Q therefore down-weights only those frames, and leaves PUF1 at 10–16%.
  - The exact B70 MaxEnt is infeasible (max error 0.06–0.12; ESS 24% / 58%) because of the 197–204 / 197–206 / 217–223 ordering. Use the soft rows.
- **Family C** (continuous K_BV): ESS 0.2–27%. Degenerate, so not used.

## 4. T3: HDX consistency of the ΔG-restrained ensembles (14 peptides × 15 times) `[d4]` → `d4_t3.csv`, `d4_t3_pep.csv`

Scores: χ² is the pooled-σ χ², mean per cell. wr2 = mean(w·r²), with w from `moprp.weights`.

| ensemble | model | mode | uniform χ² / wr2 / MSE | A S11 soft100 χ² / wr2 / MSE | B70 S11 soft100 χ² | Δχ² (A − uni) |
|---|---|---|---|---|---|---|
| MSAss | scale | frame_uptake | 34.15 / 1.633 / 0.0352 | 35.29 / 1.663 / 0.0359 | 34.86 | +1.14 |
| MSAss | scale | rate | 40.37 / 1.915 / 0.0426 | **35.00** / 1.668 / 0.0363 | 35.36 | **−5.37** |
| MSAss | shift | frame_uptake | 28.90 | 30.65 | 29.86 | +1.75 |
| MSAss | shift | rate | 71.77 | **34.98** | 44.80 | **−36.8** |
| Filtered | scale | frame_uptake | 35.06 / 1.658 / 0.0360 | 35.54 / 1.682 / 0.0362 | 35.24 | +0.48 |
| Filtered | scale | rate | 34.60 / 1.649 / 0.0362 | 34.86 / 1.667 / 0.0360 | 34.61 | +0.26 |
| Filtered | shift | frame_uptake | 27.97 | 28.41 | 28.15 | +0.44 |
| Filtered | shift | rate | 32.74 | **27.66** | 31.91 | **−5.08** |

**Per peptide** (scale model, Δχ² of A S11 soft100 vs uniform):

| peptide | MSAss frame | MSAss rate | Filtered frame | Filtered rate |
|---|---|---|---|---|
| 5–10 (β1) | +12.5 | +4.3 | +4.9 | +10.0 |
| 32–45 / 33–40 (β2) | −5.1 / −4.0 | −33.0 / −32.7 | −1.5 / −1.0 | −6.4 / −4.9 |
| 11–26, 27–31 (α1) | −1.3, −0.7 | −4.9, −3.1 | ≈ 0 | ≈ 0 |
| 60–74, 68–75 | +8.0, +7.7 | +25.4, +13.0 | +1.5, +2.2 | +2.1, +2.8 |
| 83–90 | −3.0 | −35.7 | +1.0 | +0.8 |
| 75–84 (rate) | ≈ 0 | −9.6 | ≈ 0 | ≈ 0 |

All other peptides change by less than 1.

**Does matching the per-segment ΔG_op improve the HDX fit relative to uniform?**
- **Rate averaging: yes, for MSAss** (χ² 40.4 → 35.0 scaled; 71.8 → 35.0 shift). Removing the PUF2/PUF3/unfolded mass removes the over-exchange in β2 (32–45, 33–40) and 83–90.
- **Frame_uptake: no.** It is +0.5 to +1.1 worse (scaled).
- **Filtered: unchanged within 0.5 under the scaled model.**
- **The data and the restraints agree on:**
  - β2 and α1 (both rate and frame_uptake improve);
  - 83–90 under MSAss rate.
- **They disagree on:**
  - **5–10 (β1).** Uptake wants more β1 exchange than K_op = 0.036 allows: the mean residual rises from +0.10 to +0.13 under MSAss frame. The paper's own Table 1 shows a fast β1 amide exchanging near k_int (43% amplitude), which the BV model protects.
  - **60–74 and 68–75.** These are the persistent BV-error peptides from gci2 and are under-predicted by about 0.2–0.28 D. Folded-only frames make them worse.
- These three peptides (5–10, 60–74, 68–75) plus 83–90 make up 68–82% of the summed per-peptide χ² (uniform, scaled model; 81 / 68% MSAss frame / rate, 82% Filtered frame) `[d4]`. **Reweighting moves χ² by only ±1–5 units out of 35 (frame_uptake and scaled rate), and by up to −37 under the shift-rate model.**
- Cross-check against `pop/p2_ref.csv` (that agent's output): placing the exact 97.1/2.4/0.5 state populations on the scaled model gives χ² 35.38 (frame) and 34.52 (rate) for MSAss. These are the same magnitudes as my restrained ensembles.

## 5. T4: joint objective `[d5]` → `d5_t4.csv`, `d5_weights.npz`, **`F_t4_joint.png`**

**Setup.**
- J(w) = χ²_pool(w) [scaled BV, mean per cell] + β·Σ_S11 (ln p_seg − ln p_exp)² + KL(w‖u)/S, with w = softmax(θ).
- Optimiser: L-BFGS, cold start from uniform. (Warm starts along β got stuck in one-hot softmax saturation, so I discarded them.)
- **Grid:** β ∈ {0, 0.1, 1, 10} plus 100; S ∈ {10, 100, 1000} as briefed, plus S ∈ {0.1, 1} because of the ESS collapse described next. The figure shows rule family A (solid) and Q family B70 (dashed).

**Scale caveat.** With a mean-per-cell χ² of about 25–35 and KL ≤ ln 500 = 6.2, S ≥ 10 makes the entropy term negligible. Every S ≥ 10 solution has **ESS 0.2–0.6%**, meaning a handful of frames. The β = 0 rows at S ≥ 10 are also non-convex local optima: Filtered frame_uptake lands at 100% PUF1 with χ² 26.36, while β = 0.1 reaches χ² 25.61. So the briefed S range amounts to an almost unregularised fit. S = 0.1 is the regime where ESS stays at 67–89%.

Key rows (family A):

| ens · mode | S | β | χ² | ESS | Folded / PUF1 / PUF2 / PUF3 / U (%) | rec ORIG / ALT1 / ALT2 |
|---|---|---|---|---|---|---|
| MSAss · frame | 0.1 | 0 | 32.47 | 81% | 70.8 / 18.3 / 3.1 / 2.2 / 5.6 | 66.6 / 67.3 / 66.4 |
| MSAss · frame | 0.1 | 1 | 34.30 | 79% | 93.8 / 5.4 / 0 / 0.15 / 0.65 | 89.5 / 89.8 / 89.2 |
| MSAss · frame | 0.1 | 10 | 34.46 | 77% | 96.2 / 3.0 / 0 / 0.14 / 0.65 | **91.7** / 91.6 / 91.9 |
| MSAss · frame | 100 | 0 | 24.22 | 0.36% | 71.5 / 7.3 / 0 / 21.2 / 0 | 63.7 / 63.9 / 63.4 |
| MSAss · frame | 100 | 10 | 24.80 | 0.22% | 96.4 / 2.9 / 0 / 0 / 0.76 | 91.9 / 91.8 / 92.1 |
| MSAss · rate | 0.1 | 0 | 32.33 | 77% | 81.8 / 17.2 / 0.5 / 0.2 / 0.2 | 77.0 / 77.9 / 76.6 |
| MSAss · rate | 0.1 | 10 | 32.73 | 67% | 96.2 / 3.0 / 0 / 0.5 / 0.3 | 91.8 / 91.6 / 91.8 |
| MSAss · rate | 100 | 1 | 23.97 | 0.22% | 96.4 / 2.9 / 0 / 0.5 / 0.2 | 91.9 / 91.8 / 91.9 |
| Filtered · frame | 0.1 | 0 | 33.98 | 87% | 78.8 / 17.4 / 3.3 / – / – (PUF2-like 0.6) | 74.6 / 75.6 / 74.2 |
| Filtered · frame | 0.1 | 10 | 34.88 | 84% | 96.0 / 3.5 / 0.27 / – / – (0.25) | **95.2** / 95.5 / 94.5 |
| Filtered · frame | 100 | 0 | 26.36 | 0.29% | 0 / 100 / 0 | 4.1 / 4.6 / 3.8 |
| Filtered · frame | 100 | 10 | 26.72 | 0.42% | 95.8 / 3.6 / 0 / – / – (0.54) | 92.1 / 92.1 / 91.5 |
| Filtered · rate | 0.1 | 10 | 33.58 | 78% | 95.8 / 3.6 / 0.13 / – / – (0.40) | 93.7 / 93.9 / 93.1 |
| Filtered · rate | 100 | 10 | 25.47 | 0.22% | 95.8 / 3.7 / 0 / – / – (0.54) | 92.0 / 92.1 / 91.4 |

**Findings.**
- **Restraint cost.** For every S and mode, going from β = 0 to β ≥ 1 changes χ² by **−0.6 to +2.0** (out of 24–35; the negative values occur where the β = 0 run was a local optimum) and lifts recovery from 4–78 to **86–95** (**91.7–95.2** at β ≥ 10).
  - The data term prefers PUF1/PUF3-rich or β1-open mixtures, which is the known under-prediction bias.
  - Its preference is weak compared with the per-segment ΔG restraints.
- **Ceiling.** Recovery saturates at about 92 (MSAss) and about 92–95 (Filtered) for β ≥ 1. The restraints cannot create PUF2: MSAss PUF2 is 0 (§3), and Filtered's β2 budget is split between PUF2 and the single PUF2-like frame.
- **Family B70 (dashed in the figure)** keeps PUF1 at 10–24% and recovery at 38–88. The Q-defined openings do not map onto the rule-defined states.
- Achieved open fractions for the joint optimum (S = 0.1, β = 10, frame_uptake) are in `d6_tables.md` `[d6]`. Examples: MSAss β1 0.038, α1 0.008, β2 0.008, core 0.007; Filtered β1 0.037, β2 0.005, α1/core 0.

![T4](F_t4_joint.png)

## 6. Conclusion

1. **The k_int for 133–143 is 9e-4 s⁻¹, as printed** `[d1]`. Our HDXrate pD 4.0 rates are 1.26× the paper's, uniformly across 11 segments. Ile138 alone implies 8.8e-4. So 133–143 has ΔG 0.92 and K 0.21. It is a weakly protected single amide, not part of the 2.2 kcal/mol PUF1 band.

2. **The brief's numbering offset was wrong (−104).** The correct offset is −122. The Table 1 segments are our own HDX-MS peptides, and 205–212 and 217–223 are inside the construct and usable `[d0]`.

3. **The uniform AF2 ensembles do not satisfy the per-segment ΔG_op** `[d2]`.
   - The N-subdomain (β1, α1, β2) is 2–18× too open. This is the excess PUF1/PUF2/PUF3/U mass.
   - The α2/α3 C-terminal half has no local opening at all. Filtered has zero frames. MSAss has only globally unfolded frames, which makes the global-unfolding segment 205–212 52–117× too open.
   - The experimental ordering (α3 termini 2.0–2.3 open more easily than β2 3.1, and 205–212 at 4.8) cannot be reached by any reweighting of these frames.

4. **Do the restraints move the ensemble toward Folded-majority with non-zero PUF1/PUF2? Yes for Folded and PUF1, partly for PUF2** `[d3]`.
   - With rule-feature restraints (MaxEnt or soft; ESS 84% / 91%), MSAss reaches Folded 96.4 / PUF1 2.8 / PUF2 0 / PUF3 0.2 / U 0.6, recovery **91.7–91.8** (from 73.3). Filtered reaches Folded 96.2 / PUF1 3.3 / PUF2 0.34, recovery **96.1** (from 80.8).
   - Using the segment K values as the target instead (ALT1 96.7/2.7/0.6, ALT2 97.3/2.1/0.6/U 0.03) gives the same picture: 91.5–92.0 and 95.4–96.4.
   - The segment data independently reproduce the whole-construct 97/2.4/0.5 target (98.5–98.9 recovery between targets).
   - **PUF2 stays at 0 in MSAss** because its α1-open frames (PUF3/U) are also β2-open, so the α1 and β2 restraints are jointly infeasible.
   - With native-contact (Q) indicators, PUF1 stays at 10–16% (recovery 78 / 84). **The result depends on how "local opening" is defined**, and the rule definitions are the ones that coincide with the state labels.

5. **Agreement with HDX uptake under scaled BV: weakly compatible, not confirmatory** `[d4]`, `[d5]`.
   - **Small cost or gain.** Imposing the restraints changes χ² by only +0.3 to +1.1 (frame_uptake), or improves it by 5.4 (MSAss rate scaled; up to 37 under shift-rate). The joint objective reaches about 92–95 recovery at a χ² change of −0.6 to +2 units out of 24–35.
   - **Where they agree:** β2/α1 (32–45, 33–40, 11–26, 27–31) and, in rate mode, 83–90 and 75–84 improve.
   - **Where they disagree:** 5–10 (β1) and 60–74/68–75 get worse. These are where the bias-removed BV still under-predicts exchange: β1 has a fast amide in the paper, and 60–74 is the persistent flag from gci2.
   - **Conclusion:** once the global BV offset is scaled out, the uptake data have little power over state populations. They neither contradict a Folded-majority ensemble with about 3% PUF1 nor support it beyond the β2 signal. Without restraints the fit drifts to PUF1/PUF3-rich subsets at near-identical χ². **The per-segment ΔG_op restraints are what pin the populations.**
   - **Remaining limits:**
     - The AF2 ensembles lack a C-terminal α3-frayed / β2-intact state, which the paper's PUF1 has.
     - They lack a PUF2 state whose α1 opening is separate from global unfolding.
     - The rule spec has no α3 or loop features.
     - Scaled BV compresses the per-segment ΔG range (0.8–2.8 vs 0.9–4.8 kcal/mol) `[d2]`.

### Files (all in this folder)

| file | contents |
|---|---|
| `d0_common.py` | constants, Table 1 / Table 2, scaled and shift BV, scores |
| `d0b_solvers.py` | MaxEnt dual, soft / joint L-BFGS |
| `d1_kint_check.py` → `d1_kint.csv`, `d1.log` | k_int cross-check |
| `d2a_native_contacts.py` → `d2a_Q.npz`, `d2a.log` | Q_seg + frame-order verification |
| `d2_t1_local_open.py` → `d2_t1.csv`, `d2_indicators.npz`, `d2.log` | T1 |
| `d3_t2_reweight.py` → `d3_t2.csv`, `d3_weights.npz`, `d3.log` | T2 |
| `d4_t3_hdx.py` → `d4_t3.csv`, `d4_t3_pep.csv`, `d4.log` | T3 |
| `d5_t4_joint.py` → `d5_t4.csv`, `d5_weights.npz`, `F_t4_joint.png`, `d5.log` | T4 |
| `d6_tables.py` → `d6_tables.md` | achieved per-segment open fractions |
