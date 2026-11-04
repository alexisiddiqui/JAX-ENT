# lap: Laplacian-smoothed openness for the ΔG MaxEnt reference p0

Read-only analysis: no repo edits and no framework fits. Everything is NumPy/SciPy, run from the repo root with `.venv/bin/python` and `OMP_NUM_THREADS=4`. The data, b′ model, splits, solvers and rules are reused from `selgci/q0_common.py`; the indicators, MaxEnt dual and log-LS soft fit come from `dg/`. Every number below is tagged with the script that produced it.

| tag | script | outputs |
|---|---|---|
| `[l0]` | `l0_common.py` | shared: graphs, Laplacians, smoothing, p0 re-solve + LP feasibility, χ²+KL/S+γwᵀLw solver, `select` (= q3 rules + D\|ESS≥50) |
| `[l1]` | `l1_t1_smooth.py` | `l1_t1_summary.csv`, `l1_t1_states.csv`, `l1_spectrum.csv`, `l1_O.npz`, `l1.log` |
| `[l2]` | `l2_t2_p0.py` | `l2_t2.csv`, `l2_constraints.csv`, `l2_p0.npz`, `l2.log` |
| `[l3]` | `l3_t3_sweep.py base\|omc` | `l3_sweep_{base,omc}.csv`, `l3_units_*.npz`, `l3_gamma_scale.csv`, `l3_{base,omc}.log` |
| `[l4]` | `l4_select.py` | `l4_selected.csv` (every pick), `l4_summary.csv`, `l4.log` |
| `[l5]` | `l5_fig.py` | `F_p0_vs_tau.png`, `F_ruleD_vs_tau.png` |
| `[l6]` | `l6_tables.py` (formatting only) | `l6_tables.md` (full T1/T2/T3 tables) |

## Bottom line

1. **Smoothing cannot make PUF2 reachable in MSAss, for any τ, either Laplacian, either graph, or either openness family.** This is structural, not a tuning problem `[l2]`.
   - In MSAss every α1-open frame is also β2-open: 25 α1-open frames, 37 β2-open, α1 ⊂ β2.
   - (I + τL)⁻¹ is entrywise ≥ 0 for both L = D − A and L_sym, because I + τL is a nonsingular M-matrix. So the nesting survives smoothing: o_β2,τ ≥ o_α1,τ on every frame. The smallest gap is +3.5e-5 to +2.1e-2 (L) and +6.6e-12 to +2.8e-3 (L_sym) `[l2]`.
   - Any weights therefore give E_w[o_α1] ≤ E_w[o_β2], but the targets require α1 = 1.15% > β2 = 0.54%.
   - LP check: the system is never feasible for τ > 0. The minimum relative L1 constraint violation **gets worse** with smoothing: 0.71 binary → 1.02 / 1.21 (L, τ = 0.1 / 0.3) and 0.73 → 1.27 (L_sym, τ = 0.1 → 10) `[l2]`.
   - PUF2 weight under the production p0 is ≤ 1.4e-9 % for every τ > 0 (target 0.52%). After the data term, the highest PUF2 at any rule pick is 5e-5 %, and 0.08 % even for the in-sweep oracle `[l2, l4]`.
2. **Smoothing costs Folded/PUF1 shape and ESS** `[l2]`.
   - Smoothing pushes the β2/α1 budget conflict onto PUF1 frames, whose graph neighbours are PUF2/PUF3/U. It also spreads β1 openness onto Folded "border" frames.
   - With L, PUF1 is removed (2.77 → 0 at τ = 1) and recovery falls 91.7 → 86–88. The reference collapses to a few frames (ESS 84 → 46 / 7 / 5 %, τ = 0.1 / 0.3 / 1).
   - At τ ≥ 3 every frame's smoothed openness exceeds every target, so all constraints become unattainable.
   - L_sym is gentler and roughly neutral: recovery 91.8 / 91.8 / 91.6 / 91.1 and ESS 83 / 81 / 75 / 39 % at τ = 0.1 / 0.3 / 1 / 3.
3. **Filtered** has no α1/core constraints, and only β1 and β2 are feasible.
   - L_sym at τ = 3–10 moves p0 toward the target on PUF2 (0.34 → 0.44 / 0.57 %) and PUF1 (3.28 → 2.83 / 2.20 %). Some weight leaks onto the single PUF2-like frame (0.17 → 0.23 / 0.31 %). Recovery is 96.3 / 96.0 vs 96.1 binary, and ESS is 90 / 89 % `[l2]`.
   - L at τ = 1 is harmful: recovery 93.3, PUF2 0.05 %.
4. **The rule picks get no better at any τ** `[l4]`.
   - MSAss rule D: 93.1 ± 1.2 (binary) vs 91.7–93.0 (L_sym, τ = 0.1–3) and ≤ 92.8 (L, τ = 0.1). For L at τ ≥ 0.3 the result is a degenerate 85.9–87.9 with ESS ≤ 7 %.
   - Filtered rule D: 94.9 ± 3.1 (binary) vs 94.5–95.1 (L_sym) and 92.1 or 87.9 (L, τ ≥ 1).
   - **The rule ranking from selgci is unchanged** (§T3).
5. **The OMC term γ·wᵀLw adds nothing useful** `[l4]`.
   - At τ = 0 it raises rule-D ESS (MSAss 53 → 61 / 79 %) but lowers recovery: MSAss 93.1 → 92.2 / 91.7, Filtered 94.9 → 94.4 / 93.9 (minimum 83.9).
   - Its one benefit is stabilising the harmful Filtered L τ = 1 reference: D 92.1 ± 2.9 → 93.4 ± 1.1. That is still below binary p0 (96.1).
6. **Recommendation.**
   - **Do not carry Laplacian smoothing of O into production, and keep τ = 0 (binary).**
   - If anything is carried, use L_sym with τ ≤ 1, which is harmless but useless for MSAss. Filtered L_sym τ = 3 is a marginal, optional tweak (+0.2 recovery, PUF2 closer to target).
   - The MSAss PUF2 problem has to be fixed in the restraint definition, not by smoothing. Options:
     - a PUF2-specific openness feature that is not nested under β2;
     - dropping or re-targeting the α1 (133–148 / 149–153) constraint, whose target 1.15% exceeds β2's 0.54% while α1-open ⊂ β2-open;
     - a hierarchical target such as dg's ALT2.

## Setup

- **Graph** `[l0, l1]`.
  - **MSAss:** the stored sidecar graph `_moprp_policy_sidecars_fresh_20261023/geometry/logpf_knn20.npz`. It is k = 20, Euclidean L2 over per-frame BV log-PF profiles (`_featurise` features, 96 residues), symmetric union, unweighted, 7215 edges, connected, degree 20–82 (mean 28.9).
    - The sidecar OMC runs used weights exp(−(d/h)²/2) with h = 10–1000 median units, which is effectively unweighted, so I used it as stored.
    - 94% of its edges join same-state frames. The 10 PUF2 frames have only 26% same-state neighbours.
  - **Filtered:** no graph is stored (the sidecars built one for MSAss only), so **I rebuilt it**.
    - Recipe: L2 over per-frame BV lnPF (`_featurise_a101_k109`, 97 residues; unshifted and unfloored, since distances are shift-invariant), k = 20 excluding self, symmetric union.
    - Gaussian edge weights exp(−d²/2h²) at the median-scaled bandwidth: h = median positive pairwise distance = 6.44. This gives 7257 edges and a median weight of 0.82.
  - **Check:** the same rebuild for MSAss (h = 7.24) gives 7231 edges, 6653 of them shared with the stored graph (92%; Jaccard 0.85). This confirms that frame order and features agree. It is carried through T1/T2 as the "rebuilt" row.
- **Laplacians.** L = D − A and L_sym = I − D^-½AD^-½. Smoothing is o_τ = (I + τL)⁻¹o for each row.
- **O families.**
  - **A** (rule features): four distinct region rows, β1 / α1 / β2 / core, which map to the S11 segments. This is the family behind selgci's p0.
  - **B70:** the native-contact variant (`dg/d2a`, Q_seg < 0.7), one row per segment.
- **p0 re-solve** `[l2]`. Same as q0/d3:
  - Rows that are identical in the raw O are merged with a geometric-mean target. A row is dropped when its target is outside (min, max) of the smoothed row (unattainable).
  - The exact MaxEnt dual is solved, plus an LP feasibility test (max ε s.t. Ow = t, Σw = 1, w ≥ ε; also the minimum relative L1 violation).
  - **Production p0** = the exact solution if it is feasible and no extra row was dropped; otherwise d3's log-least-squares soft fit (S = 100), as the brief specified.
  - τ = 0 reproduces selgci's p0 exactly (max |Δp| = 0) `[l2]`.
  - Because the soft fit at S = 100 collapses ESS under infeasibility, the soft fit at S = 1 ("soft1") is reported alongside as a sensitivity.
- **Sweeps** `[l3]`. Identical to selgci q1, σ floored only:
  - b′ rate-averaged scaled BV, 14 peptides × 15 timepoints, three contiguous 2-fold block splits, S ∈ {0} ∪ logspace(−3.5, 0.5, 33).
  - γ = 0 uses q0's exponential-family solver. γ > 0 uses a 500-logit L-BFGS, because the Lw term leaves that family.
  - A γ = 0 full-logit control reproduces the EF solver on every candidate point (|ΔJ| ≤ 4e-8, |Δrec| ≤ 5e-6) `[l3]`.
  - Stationarity residual on candidate points: 99th percentile 6e-6. The maximum is 0.037, on 3 soft1 τ = 10 points; none of these is a pick in a headline row.
- **Rules** `[l4]`. selgci's A / B / C / C1SE / D, plus refs p0 and oracle, with the same bootstrap seed. Two changes:
  - The candidate set is {ESS ≥ 10%} ∪ {S = 0}, because some smoothed p0 have ESS < 10%.
  - **D|ESS≥50** (rule D restricted to candidates with ESS ≥ 50%) is added as the ESS acceptance check. "#ESS≥50" counts accepted folds.
  - τ = 0 reproduces selgci T1 exactly (MSAss A 92.9 ± 1.2 / ESS 64%; B 93.1 ± 1.2; Filtered B 94.9 ± 3.1).

## T1. What smoothing does to O `[l1]` (full tables: `l6_tables.md` T1a–c)

- **Scale.** Unnormalised L on this graph is strong.
  - MSAss: λ_2 = 0.22, λ_med = 27, λ_max = 83. The smoother keeps 139 / 59 / 21 / 8 / 3 / 2 effective modes (trace, out of 500) at τ = 0.1 / 0.3 / 1 / 3 / 10 / 30.
  - At the median eigenvalue the filter 1/(1+τλ) is 0.27 at τ = 0.1 and 0.036 at τ = 1. **L with τ ≥ 1 is close to global averaging.**
  - Filtered behaves the same way (λ_med 20, traces 172 / 79 / 29 / 11 / 4 / 2).
  - L_sym (λ ∈ [0, 1.27]) is about 30× gentler: traces 455 / 385 / 253 / 130 / 50 / 19.
- **Column sums.**
  - L preserves Σ_i o_τ,i to ≤ 1e-14 (1ᵀL = 0).
  - L_sym does not, because its null vector is D^½1, not 1. MSAss loses 0.06–5.7% of the segment open fraction (τ = 0.1–30). Filtered β2 loses 1.7–35%.
  - L_sym smoothing is also not a convex average: for example, PUF2 β1 openness reaches 1.04.
- **Fractional frames.**
  - With L at τ = 0.1, 498/500 frames are already fractional for β1 and 110–126 for α1/β2/core. At τ ≥ 1 all 500 frames are fractional for every row.
  - With L_sym it is 49–96 frames at τ = 0.1, 79–316 at τ = 1 and 267–500 at τ = 10.
  - Mean |o_τ − o| is 0.012–0.026 (L, τ = 0.1), 0.05–0.07 (τ = 1) and 0.06–0.23 (τ = 10). For L_sym it is 0.002–0.003, 0.012–0.016 and 0.036–0.048.
- **Per-state openness, MSAss, A**:

  | L, τ | β1 Folded / PUF1 / PUF2 / PUF3 / U | α1 | β2 | core |
  |---|---|---|---|---|
  | binary | 0 / 1 / 1 / 1 / 1 | 0 / 0 / 0 / 1 / 0.93 | 0 / 0.01 / 1 / 1 / 1 | 0 / 0 / 0 / 0.27 / 1 |
  | L 0.1 | 0.007 / 0.96 / 0.99 / 1.00 / 1.00 | 0.000 / 0.05 / 0.27 / 0.67 / 0.78 | 0.000 / 0.10 / 0.70 / 0.82 / 0.92 | 0.000 / 0.03 / 0.19 / 0.36 / 0.67 |
  | L 1 | 0.043 / 0.82 / 0.88 / 0.90 / 0.91 | 0.006 / 0.12 / 0.30 / 0.39 / 0.44 | 0.010 / 0.19 / 0.47 / 0.53 / 0.59 | 0.004 / 0.09 / 0.21 / 0.26 / 0.34 |
  | L 10 | 0.150 / 0.44 / 0.47 / 0.48 / 0.49 | 0.033 / 0.10 / 0.13 / 0.15 / 0.16 | 0.049 / 0.14 / 0.20 / 0.21 / 0.22 | 0.024 / 0.07 / 0.10 / 0.10 / 0.11 |
  | L_sym 1 | 0.004 / 0.98 / 1.03 / 0.99 / 0.98 | 0.000 / 0.02 / 0.19 / 0.78 / 0.83 | 0.000 / 0.06 / 0.83 / 0.90 / 0.95 | 0.000 / 0.01 / 0.13 / 0.34 / 0.78 |
  | L_sym 10 | 0.019 / 0.90 / 1.02 / 0.95 / 0.93 | 0.002 / 0.09 / 0.34 / 0.49 / 0.55 | 0.003 / 0.16 / 0.62 / 0.66 / 0.72 | 0.001 / 0.06 / 0.24 / 0.31 / 0.43 |

  Targets: β1 3.6%, α1 1.15%, β2 0.54%, core 0.73%.
  - Smoothing gives PUF2 partial α1 openness, but always less than its β2 openness.
  - It gives PUF1 0.1–0.2 of α1/β2 openness, i.e. 10–35× those targets.
  - It gives Folded frames a floor: L τ = 10 Folded β1 = 0.15, which exceeds every target. That floor is why all A rows become unattainable at τ ≥ 3 under L.

## T2. Reference p0_τ: the key table `[l2]` (all graphs/families: `l6_tables.md` T2; `F_p0_vs_tau.png`)

Family A, primary graph. Pops in %. "LP" is the minimum relative L1 violation (0 = feasible). prod is the production p0; soft1 is the S = 1 sensitivity.

| ens | L | τ | LP | p0 | ESS | Folded / PUF1 / PUF2 / PUF3 / U | rec | PUF2 % |
|---|---|---|---|---|---|---|---|---|
| MSAss | – | 0 | 0.71 (infeas.) | exact (q0) | 84 | 96.4 / 2.77 / 0 / 0.25 / 0.60 | 91.7 | 0 |
| MSAss | L | 0.1 | 1.02 | soft | 46 | 96.8 / 2.30 / 1e-11 / 3e-8 / 0.94 | 91.5 | 1e-11 |
| MSAss | L | 0.1 | | soft1 | 82 | 96.9 / 2.12 / 3e-9 / 3e-6 / 0.93 | 91.5 | 3e-9 |
| MSAss | L | 0.3 | 1.21 | soft | 7 | 98.9 / 0 / 0 / 0 / 1.06 | 85.9 | 6e-20 |
| MSAss | L | 1 | 0.36 (β2 dropped) | soft | 5 | 99.3 / 0 / 0 / 0 / 0.71 | 86.6 | 1e-9 |
| MSAss | L | 1 | | soft1 | 78 | 99.5 / 0.004 / 0 / 7e-5 / 0.53 | 87.0 | 2e-31 |
| MSAss | L | 3–30 | all 4 rows unattainable | soft | 0.3–1.8 | 100 / 0 / 0 / 0 / 0 | 87.9 | ≤ 6e-9 |
| MSAss | L_sym | 0.1 | 0.73 | soft | 83 | 96.4 / 2.78 / 0 / 0.05 / 0.75 | 91.8 | 1e-82 |
| MSAss | L_sym | 0.3 | 0.77 | soft | 81 | 96.5 / 2.73 / 0 / 0.03 / 0.79 | 91.8 | 3e-49 |
| MSAss | L_sym | 1 | 0.88 | soft | 75 | 96.4 / 2.66 / 0 / 2e-8 / 0.90 | 91.6 | 1e-19 |
| MSAss | L_sym | 3 | 1.05 | soft | 39 | 96.4 / 2.49 / 0 / 0 / 1.07 | 91.1 | 2e-15 |
| MSAss | L_sym | 10 | 1.27 | soft | 12 | 98.7 / 0 / 0 / 0 / 1.28 | 85.6 | 2e-18 |
| MSAss | L_sym | 10 | | soft1 | 80 | 98.0 / 0.77 / 0 / 2e-6 / 1.24 | 89.1 | 2e-8 |
| Filtered | – | 0 | feasible (β1, β2) | exact (q0) | 91 | 96.2 / 3.28 / 0.34 (PUF2-like 0.17) | 96.1 | 0.34 |
| Filtered | L | 0.1 | feasible | exact | 90 | 96.7 / 2.79 / 0.28 (0.20) | 96.3 | 0.28 |
| Filtered | L | 0.3 | feasible | exact | 89 | 97.2 / 2.39 / 0.22 (0.22) | 96.1 | 0.22 |
| Filtered | L | 1 | feasible | exact | 86 | 98.6 / 1.10 / 0.05 (0.22) | 93.3 | 0.05 |
| Filtered | L | 3–30 | β1, β2 unattainable | soft | 1–18 | ≈100 / 0 / 0 | 87.9 | ≈ 0 |
| Filtered | L_sym | 1 | feasible | exact | 90 | 96.4 / 3.02 / 0.37 (0.20) | 96.3 | 0.37 |
| Filtered | L_sym | 3 | feasible | exact | 90 | 96.5 / 2.83 / 0.44 (0.23) | 96.3 | 0.44 |
| Filtered | L_sym | 10 | feasible | exact | 89 | 96.9 / 2.20 / 0.57 (0.31) | 96.0 | 0.57 |
| Filtered | L_sym | 30 | feasible | exact | 62 | 96.5 / 0.34 / 0.03 (3.1) | 84.6 | 0.03 |

- **Per-constraint mismatch, MSAss** `[l2]` `l2_constraints.csv`.
  - The infeasibility always lands on the α1/β2 pair: achieved/target is α1 0.63–0.74 and β2 1.56–2.07 for all τ and both L. β1 is at 0.86–1.16 and core at 0.73–1.01.
  - At τ = 0 the exact dual already shows this: α1 0.735, β2 1.563.
- **Other variants** `[l2]` (`l6_tables.md` T2).
  - **MSAss rebuilt Gaussian graph.** Same conclusion: LP is infeasible at τ ≤ 1 and rows become unattainable from τ = 3. Production p0 recovery is 91.9 / 92.0 / 86.6 at τ = 0.1 / 0.3 / 1 (ESS 74 / 51 / 15%), and PUF2 ≤ 1e-8 %.
  - **MSAss B70.** Binary is already infeasible (LP 1.85), and it stays infeasible at 1.7–3.6 under smoothing. Production PUF2 is ≤ 1.4e-9 % at every τ > 0.
  - **Filtered B70.** Here smoothing helps the weaker family: binary exact recovery 81.8 → L_sym τ = 0.3 soft 92.9 (ESS 67%). It still stays below family A (96.1).
- **Answer to the key question.** Smoothing does **not** make PUF2 reachable in MSAss. What it costs depends on the Laplacian:
  - With **L**, PUF1 (all of it by τ = 0.3–1), 5–6 recovery points, and ESS (84 → ≤ 7% at τ ≥ 0.3). soft1 keeps ESS at about 78% but not the recovery.
  - With **L_sym**, essentially nothing up to τ = 1: at most −0.1 recovery and ESS 84 → 75%.
  - In both cases there is no gain.

## T3. Sweeps and rule picks `[l3, l4]` (full: `l6_tables.md` T3; `F_ruleD_vs_tau.png`)

Cell: recovery mean ± SD (min) over 6 folds · ESS % · #folds with ESS ≥ 50. All picks have PUF2 ≤ 5e-5 % (MSAss).

| ens | variant | τ | p0 | A | B = C | C1SE | D | D\|ESS≥50 | oracle |
|---|---|---|---|---|---|---|---|---|---|
| MSAss | binary (selgci) | 0 | 91.7 · 84 | 92.9 ± 1.2 (91.7) · 64 · 4 | 93.1 ± 1.2 · 50 · 3 | = p0 | 93.1 ± 1.2 (91.7) · 53 · 3 | 93.0 ± 1.0 (91.7) · 74 · 6 | 93.6 |
| MSAss | L | 0.1 | 91.5 · 46 | 92.6 ± 1.3 · 45 · 0 | 93.1 ± 1.1 · 35 · 0 | = p0 | 92.8 ± 1.2 (91.5) · 35 · 0 | 91.5 (= p0) · 46 · 0 | 94.1 |
| MSAss | L | 0.3 / 1 / 3 / 10 | 85.9 / 86.6 / 87.9 / 87.9, ESS 0–7 | all rules = p0 (no candidate with ESS ≥ 10%) | | | | | |
| MSAss | L, soft1 p0 | 1 / 10 | 87.0 / 87.9 · 78 / 71 | 87.1 / 88.5 | 87.5 / 88.5 | = p0 | 87.5 ± 0.3 / 88.5 ± 1.8 | 87.4 / 87.8 | 88.0 / 89.8 |
| MSAss | L_sym | 0.1 | 91.8 · 83 | 92.6 ± 0.9 · 61 · 4 | 91.8 ± 1.7 (89.7) · 40 · 2 | = p0 | 91.7 ± 1.2 (90.5) · 42 · 2 | 92.6 ± 0.7 (92.0) · 67 · 6 | 93.6 |
| MSAss | L_sym | 0.3 | 91.8 · 81 | 92.6 ± 0.9 · 69 · 5 | 92.8 ± 1.3 · 44 · 3 | = p0 | 92.8 ± 0.9 (91.7) · 49 · 3 | 92.9 ± 0.9 (91.7) · 70 · 6 | 93.9 |
| MSAss | L_sym | 1 | 91.6 · 75 | 92.7 ± 1.2 · 72 · 6 | 92.8 ± 1.8 (89.8) · 43 · 2 | = p0 | 92.6 ± 1.7 (89.8) · 45 · 2 | 92.8 ± 1.5 (90.1) · 63 · 6 | 93.9 |
| MSAss | L_sym | 3 | 91.1 · 39 | 92.4 ± 1.2 · 36 · 0 | 93.2 ± 1.2 · 30 · 0 | = p0 | 93.0 ± 1.1 (91.3) · 34 · 0 | 91.1 (= p0) · 39 · 0 | 94.0 |
| MSAss | L_sym | 10 | 85.6 · 12 | 86.6 ± 0.8 · 12 | 87.4 ± 0.2 · 12 | = p0 | 88.0 ± 1.4 · 12 · 0 | = p0 | 88.0 |
| Filtered | binary (selgci) | 0 | 96.1 · 91 | = p0 (6/6) | 94.9 ± 3.1 (88.6) · 65 · 4 | = p0 | 94.9 ± 3.1 (88.6) · 65 · 4 | = p0 · 91 · 6 | 96.5 |
| Filtered | L | 0.1 / 0.3 | 96.3 / 96.1 | = p0 | 95.0 ± 3.3 / 94.5 ± 3.4 | = p0 | 95.0 ± 3.3 (88.2) / 94.5 ± 3.4 (87.7) | 96.3 / 95.8 | 96.7 / 96.5 |
| Filtered | L | 1 | 93.3 · 86 | = p0 | 92.1 ± 2.9 (86.2) · 61 | = p0 | 92.1 ± 2.9 (86.2) · 71 · 5 | 93.2 ± 0.4 · 84 · 6 | 94.5 |
| Filtered | L | 3 / 10 | 87.9, ESS 1–3 | all rules = p0 | | | | | |
| Filtered | L_sym | 0.1–3 | 96.1–96.3 · 90–91 | = p0 | 95.0–95.1 ± 3.2–3.7 (87.6–88.6) · 64–65 | = p0 | same as B | = p0 | 96.5–96.7 |
| Filtered | L_sym | 10 | 96.0 · 89 | 96.1 ± 0.2 · 77 · 5 | 94.5 ± 4.0 (86.5) · 64 · 4 | = p0 | 94.5 ± 4.0 (86.5) · 64 · 4 | 96.0 ± 0.1 · 89 · 6 | 96.2 |

- **Rule ranking: unchanged from selgci.**
  - C1SE returns p0 in essentially every fold (5–6 of 6), so it is degenerate on MoPrP.
  - B = C = D chase the χ² minimum, with ESS 30–53% on MSAss and low-ESS failures on Filtered (minimum 86–88).
  - A and D|ESS≥50 are the most consistent. On MSAss they are within about 0.5 of D or better, with 6/6 folds accepted at ESS ≥ 50% for D|ESS≥50 whenever p0's own ESS is ≥ 50%. On Filtered they return p0.
  - No τ lifts any rule above its binary value by more than noise (≤ +0.2). The best smoothed MSAss rule-D mean is 93.0 (L_sym τ = 3), vs 93.1 binary, at lower ESS (34 vs 53%).
  - The in-sweep oracle rises slightly with L_sym (93.6 → 93.9–94.0, MSAss) and L τ = 0.1 (94.1). None of the rules captures this.
- **OMC term (γ·wᵀLw, L = primary unnormalised Laplacian)** `[l3, l4]`.
  - **Scaling** `[l3]` `l3_gamma_scale.csv`: γ_abs = γ·c, where c is the median over folds of (KL/S)/(wᵀLw) at the γ = 0 rule-D pick. So γ = 1 makes the OMC term equal to the KL term at the typical selected point.
  - c = 45.2 (MSAss τ = 0; 6/6 folds), 3.67 (Filtered τ = 0; 2/6), 157 (Filtered τ = 1; 5/6).
  - For the degenerate references (MSAss τ = 1, 10; Filtered τ = 10) I used the fallback at the S ≈ 0.1 grid point: c = 0.13, 0.012, 0.031.

  | ens, τ | γ = 0: D · ESS | γ = 0.1: D · ESS | γ = 1: D · ESS | γ = 0.1 / 1: A |
  |---|---|---|---|---|
  | MSAss, 0 | 93.1 ± 1.2 · 53 (3/6) | 92.2 ± 0.6 · 61 (4/6) | 91.7 ± 1.7 (88.5) · 79 (6/6) | 92.7 / 92.5 |
  | MSAss, 1 / 10 | 86.6 / 87.9 (= p0) | = p0 | = p0 | = p0 |
  | Filtered, 0 | 94.9 ± 3.1 (88.6) · 65 | 94.4 ± 3.8 (86.6) · 64 | 93.9 ± 4.9 (83.9) · 60 | 95.9 / 95.9 |
  | Filtered, 1 | 92.1 ± 2.9 (86.2) · 71 | 92.6 ± 1.9 (88.8) · 79 | 93.4 ± 1.1 (92.0) · 85 (6/6) | 93.3 / 93.3 |
  | Filtered, 10 | 87.9 (= p0) | = p0 | = p0 | = p0 |

  - Because wᵀLw is minimised by uniform weights, the OMC term pulls w toward uniform. That raises ESS and adds PUF/U mass, which lowers recovery on the good (τ = 0) references. It shrinks fold spread only where the reference is already bad (Filtered L τ = 1).
  - **It does not change which rule wins, and it does not help PUF2** (still 0 in MSAss).

## Figures `[l5]`

- **`F_p0_vs_tau.png`.** Rows: p0 recovery, ESS and PUF2 weight vs τ. Columns: the two ensembles. Lines: L and L_sym; solid is the production p0, dashed is soft S = 1.
- **`F_ruleD_vs_tau.png`.** Rule-D recovery and ESS (mean ± SD over 6 folds) vs τ, for γ = 0 (L, L_sym) and γ = 0.1, 1 (OMC, L).

## Caveats

- **Production p0 for infeasible τ.** It uses d3's soft fit at S = 100, as the brief said. Under smoothing this concentrates weight (ESS 5–46% with L), so the soft1 rows show the S-sensitivity.
  - soft1 restores ESS (about 78–83%) but not recovery or PUF2.
  - Neither solver can beat the nesting argument.
- **Native-contact family.** B70 was only taken through T1/T2. Smoothing helps it in Filtered, but it stays below family A.
- **Filtered graph.** It is rebuilt with Gaussian weights; MSAss uses the stored unweighted graph. The MSAss rebuilt-graph check gives the same conclusions.
- **Stored graph features.** The stored graph uses the older `_featurise` features (96 residues). Its kNN topology matches the a101_k109 rebuild on 92% of edges.
- **σ.** Floored σ only. selgci's unfloored variant was not repeated.
- **Solver flags.** The L-BFGS "success" flag is unreliable, as in selgci. Stationarity is the check: 99th percentile 6e-6 on candidates.
  - On 26 of the 408 points in the γ = 0 control, the full-logit and EF solvers reach different stationary points (|ΔJ| > 1e-6; rate-mode χ² is non-convex). All 26 have ESS < 5%, so none is a candidate.
