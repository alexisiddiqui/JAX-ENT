# Control-MD trajectory provenance and the 2026-11-05 repair

Source of truth: Pegasus, `/opig-shared/projects/hussian-simulation_hdx/projects/aSyn_simulations/system_output_{coil,hairpin,rod}/simulation_output/{control,tris}/{8..13}_prod_rep{1,2,3}/`
(symlinked from `~/hussain-simulation_hdx` on the gate; ssh alias `Pegasus-Gate`).
Each production stage is a 20 ns chunk written every 10 ps (2001 frames incl. t=0);
`md_mol_center.xtc` is the centred, whole-molecule protein(+Tris) subset produced by
`scripts/PBC.sh` in the `aSyn_simulations` repo. Six chunks x three replicates x three
starting conformers = 108 chunks per condition, 1080 ns.

Local assembly (`data/_aSyn/{control_MD,tris_MD}/`): per-replicate files are the six chunks
concatenated in stage order with each chunk's own time stamps (0-20000 ps, repeated);
family files are rep1,rep2,rep3; `*_all_combined.xtc` is rod1-3, coil1-3, hairpin1-3.
`truncate_trajectories.sh` takes a proportional prefix of every replicate for the 0.25/0.5 us sets.

## Defect found 2026-11-05

`control_coil_rep1_combined.xtc` had 10,328 frames instead of 12,006: its last chunk
(`13_prod_rep1`) was a 323-frame partial. On Pegasus that stage had been launched twice
into the same directory on 2026-05-15 (05:42 and 07:53; the second produced the partial and
is what reached the local copy), then cleanly re-run from the same starting state with a
new seed on 2026-05-17/18 via `_submission_scripts/aSyn/run/prod_simulation_control_topup.sh`.
The completed run (2001 frames) never reached the local assembly. Consequently the local
control ensemble, its 0.25/0.5 us truncations, the control feature sets and every control
fit in the manuscript used 1063.8 ns (106,376 frames) rather than 1080 ns.

A second, Pegasus-side defect: the centred file for `control/11_prod_rep1` was regenerated
on 2026-05-18 and lost the frame at t = 4180 ps (2000 frames; the raw `md.xtc` and the
2026-05-17 backup `#md_mol_center.xtc.1#` have all 2001). The local copy was built from
the complete version, so no action was needed locally, but anyone re-deriving from Pegasus
must use `md.xtc` or the backup for that chunk.

All other 106 chunks were verified frame-for-frame identical between the local assembly
and Pegasus for coil rep1 chunks 8-12 and by frame count (12,006 per replicate) for the rest.
Tris-MD is complete (108,045 frames) and untouched.

## Repair

Fetched the completed `13_prod_rep1/md_mol_center.xtc` (and chunks 8-12 for verification)
into `control_MD/_pegasus_topup/`. Rebuilt with MDAnalysis (positions round-trip exactly):

| file | before | after |
|---|---|---|
| control_coil_rep1_combined.xtc | 10,328 | 12,006 |
| control_coil_combined.xtc | 34,340 | 36,018 |
| control_all_combined.xtc | 106,376 | 108,054 |
| control_all_combined_0.25us.xtc | 24,623 | 25,011 |
| control_all_combined_0.5us.xtc | 49,245 | 50,022 |

The 323 partial frames were dropped (they are a different stochastic realisation of the
same start and would double-count). Every frame outside the replaced chunk is bit-identical
to the previous file. Previous files and feature sets are kept in `control_MD/_pre_topup/`
until the control refit is done. Control features (`features`, `features_0.25us`,
`features_0.5us`) were regenerated with `featurise_benchmark.sh` (BV, switch, peptide_trim 0).
Tris truncations were rewritten by the same script run and are unchanged in content.

## Feature-set changes from re-featurising with the current code (2026-11-05)

`featurise_benchmark.sh` re-ran all five truncation/control sets (its skip test looked for a
`<name>.npz` file that the CLI never writes; fixed to test `features.npz` + `topology.json`),
and the Tris 1.0 us set was then regenerated too so that all six sets come from the same
featuriser (old Tris set kept in `tris_MD/_pre_topup/features/`). Differences versus the
May 2026 feature files:

- 134 residues instead of 133: His50 (`HIE`) is now named and the C-terminal residue 140 is
  included (the old topology coded His50 as `X` and omitted 140). Residue ids 2-139 are
  otherwise identical. Residue 140 has no experimental PF, so it does not enter the fits.
- Contacts on shared residues and frames are identical except 17 of 4.8 million heavy-contact
  entries that differ by exactly 0.5 (an atom exactly at the switching midpoint).
- The `k_ints` column is 10^0.4 smaller than before (featurise CLI pH default), and His50/51
  now have non-zero rates. The residue-PF fits (`optimise_aSyn_conditions_BV.py`) do not read
  this column; the experimental targets' intrinsic rates come from `data/extract_PF.py`.

Frame/residue counts after the rebuild:

| set | shape |
|---|---|
| tris_MD/features | 134 x 108,045 |
| tris_MD/features_0.5us | 134 x 50,022 |
| tris_MD/features_0.25us | 134 x 25,011 |
| control_MD/features | 134 x 108,054 |
| control_MD/features_0.5us | 134 x 50,022 |
| control_MD/features_0.25us | 134 x 25,011 |

## Experimental log-PF targets regenerated (2026-11-05)

`data/extract_PF.py` divided the observed rates (s^-1 in `ac2c03183_si_002.xlsx`) by 60 with
the comment "convert to min^-1" while the intrinsic rates stayed in s^-1, inflating every
log-PF target by ln 60 = 4.09 since 2026-04-07. Fixed (no conversion; `unit="s^-1"` passed
explicitly; plot axis labels corrected) and the targets regenerated with the same inputs
(131 residues x 4 conditions; identical residue sets). Previous outputs are in
`data/_aSyn/_pre_units_fix/`.

The regenerated values are lower than the May 2026 targets by 5.015 +- 0.003 ln units at every
residue except His50 (4.58) and Gly51 (4.68): ln 60 from the units fix plus 0.4 ln 10 = 0.92
because the intrinsic-rate convention changed in the library on 2026-08-13. Until then
`calculate_HDXrate` let HDXrate apply its automatic pH -> pD (+0.4) correction, so the May
targets effectively used pD 7.8 / 7.8 / 7.6 / 5.3; JAX-ENT now passes the value through
unchanged (`HDXRATE_PH_CORRECTION = False`), i.e. pD = measured pH (7.4 / 7.4 / 7.2 / 4.9),
the same convention chosen for MoPrP. Both effects are uniform per condition. New medians:
Tris 7.40, Extracellular 6.73, Intracellular 6.30, Lysosomal 1.21 (19 Lysosomal residues
are now < 0, i.e. observed faster than intrinsic at pD 4.9; with the +0.4 convention they
would be 0.92 higher). `aSyn_PF_Tris_only.dat` (PyMOL reference) regenerated with
`analysis/create_tris_reference_dat.py`, whose hard-coded `/Users` path was replaced.

`analysis/plot_feature_distributions_aSyn_conditions.py` resolved the example directory with
`metric_dir.parents[3]`, which is `fitting/` for the benchmark layout, so the experimental
target line (SI 13) and the Tris-bound/Tris-contact plots (SI 12) were silently skipped; it now
uses the script location.

## Manuscript fit recovered from the MacBook (2026-11-05)

The four-condition fit behind Fig 5 / SI 12-15, `fitting/_optimise_aSyn_BV_FIGURE_5000_20260522_020324`
(480 runs: 4 conditions x 3 sequence_cluster splits x 20 MaxEnt values 1-1000 x L1 bvreg {0.5, 1.0};
5000 Adam steps, lr 1.0, `hdx_pf_l2_loss`, BV parameters optimised, forward_model_scaling 1000,
EMA 0.5), its `_processed_` directory (selection metrics incl. `val_mse_min`, extracted models,
scores) and the two `analysis/_analysis*_20260522_020324` directories were copied from Teal
(MacBook) via Pegasus (`~/_transfer/asyn_fit_20260522_020324/` in the Pegasus home, staged copy
left in place) to this machine: 6,345 files, 4,540,054,796 bytes, byte-exact. These used the
106,376-frame control ensemble (fits are per condition on Tris-MD features only, so the control
defect affects only the benchmark runs) and the ln60-inflated / pD+0.4 targets.

## Experimental data re-read from the SI spreadsheet (2026-11-05)

`ac2c03183_si_002.xlsx` sheet "Figure S11" holds the raw HDX data per condition: 30 peptides
(1-17 ... 135-140) each with the whole-peptide series (fragment 1) and up to 13 ETD fragments
(c/z ions, 'RS'), a 9-point uptake series (10 for Lysosomal) in Da with replicate SDs, a fitted
rate `k_exp`, `q` = number of exchanging amides (= length - 1 - prolines, i.e. first residue
excluded), `rmse`, `X_plateau` (time to plateau). Extracted to `data/_aSyn/uptake/S11_fragment_uptake_raw.csv`.
Sheet "Figure S10" (the residue table `aSyn_k_obs.csv` was copied from) equals the S11 rates
mapped onto residues.

Findings (all from the data themselves; SI PDF not yet obtained):
- The time axis of the Tris/Extracellular/Intracellular series is milliseconds:
  t = 0, 50, 100, 250, 500, 1000 ms, 10 s, 30 s, then 60 s or 300 s (last point ambiguous).
  A stretched-exponential fit on this grid reproduces `k_exp` exactly (ratio 1.000) for Tris.
  Hence the "Observed rate constant (s-1)" values are really ms^-1: Tris N-terminal peptides
  exchange at ~3.5 s^-1, not 0.0035 s^-1. Every previous aSyn log-PF target was therefore
  inflated by ln 1000 = 6.9 on top of the ln 60 units bug.
- Extracellular and Intracellular `k_exp` are the fitted rates multiplied by 1.180 and 1.721,
  exactly the bradykinin ratios in sheet "Figure S5-middle panel" (6.23/5.28, 6.23/3.62): the
  reported rates are Tris-normalised (the paper's "empirical correction"); the uptake curves are raw.
- Lysosomal: same relative time spacing, absolute scale not identifiable from the curves
  (physically ~100x longer, plateau by ~10^3 s); needs the SI PDF.
- With rates in s^-1, Tris peptide ln PF (ln mean k_int / k_obs, pD 7.4, 293 K) is 0.4-0.7 over
  residues 4-94, 1.2-1.5 for 90-113 and 2.0-3.4 in the acidic tail 114-140 (electrostatic
  suppression of base-catalysed exchange, absent from sequence-based k_int).
- Gate check (unweighted priors vs raw uptake, BV bc 0.35 bh 2.0, rate averaging, k_int at pD,
  SD floor 2 % of q, global ln PF shift fitted): Tris-MD on Tris data: best shift -1.3, chi2/point
  195; Control-MD 221. Before any shift the Tris-MD prior over-protects the N-terminus by ~3.6 ln
  (prior ln PF ~4.0 vs 0.4 observed) and the NAC by ~1.8 ln, and under-protects the tail by
  1.5-2 ln. Rate averaging beats log averaging (chi2 286) everywhere. The two ensembles differ
  far less than either differs from the data.

Implication: the manuscript's aSyn fits targeted residue log-PFs ~10 ln units too high; the
L1-regularised bc/bh fits and ESS 0.01-0.1 % selections absorbed that. All aSyn results must be
regenerated against the raw uptake curves (peptide-level, 8 timepoints, SDs), which the validated
MoPrP pipeline handles directly.
