# Recovery-gulf investigation (2026-11-02 … 2026-11-04)

Read-only diagnosis of why conformational recovery across Examples 1–3 was
inconsistent, the resulting pipeline fixes, and the validated MoPrP recipe.
Reports, scripts and key figures are copied here from the gitignored
`artifacts/recovery_gulf_20261103/` (large CSV/NPZ intermediates are not
tracked; every script regenerates them). Numbers in each report are tagged
with the script that produced them.

| dir | question | report |
|---|---|---|
| `hgeom/` | amide-H geometry in all structures/trajectories | `check_h.py` (N–H 1.02 Å, outward, in-plane, all frames) |
| `gci2/` | externally fixed Σ, geometric N_eff, pLDDT bc/bh refit, work-scale selector | `gci2_report.md` |
| `pop/` | state-level population landscape under median-scaled BV | `pop_report.md` |
| `dg/` | per-segment ΔG_op restraints (Moulick 2015 Table 1) → p0 | `dg_report.md` |
| `sel/` | twelve selection functions on the existing candidate pools | `sel_report.md` |
| `selgci/` | rules A/B/C/1-SE/D on sweeps with a known answer (ISO control) | `selgci_report.md` |
| `lap/` | Laplacian-smoothed openness for p0; OMC term | `lap_report.md` |
| `stk/` | self-tuning kNN kernel: two-sided variance in selection, OMC | `stk_report.md` |
| `iso_stk/` | sidecars' scalar-distance OMC (Work Scale / RMSD / PyRosetta / dihedral) in the new setup; self-tuning OMC on ISO | `omc_scalar_report.md` |

The earlier reports of the same investigation (ISO/Ex2/Ex3 deep dives,
diagnostics digest, pipeline-mechanics audit, calibration, re-selection,
first GCI pass) were lost with a session scratch wipe; their conclusions are
summarised in `docs/averaging_mode_soft_regression.md`
("Recovery-gulf diagnosis and validated MoPrP recipe").
