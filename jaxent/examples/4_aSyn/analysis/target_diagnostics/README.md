# aSyn residue-PF target diagnostics (2026-11-08)

Closed-form checks of the corrected residue ln PF targets (`data/_aSyn/aSyn_PF.csv`, k_obs read as ms⁻¹)
against the Tris-MD and Shaw (a99SB-disp, `_aSyn/a99sb_features`) contact features. Under `log_pf`
frame averaging the BV prediction is linear in the parameters and in the frame weights, so every
model here is a least-squares problem and needs no optimiser.

Run from the repo root with the `.venv` interpreter:

```bash
python jaxent/examples/4_aSyn/analysis/target_diagnostics/fit_models.py
python jaxent/examples/4_aSyn/analysis/target_diagnostics/reachability.py
python jaxent/examples/4_aSyn/analysis/target_diagnostics/xlsx_to_csv.py \
    jaxent/examples/4_aSyn/data/_aSyn/ac2c03183_si_002.xlsx /tmp/si_sheets
python jaxent/examples/4_aSyn/analysis/target_diagnostics/peptide_kinetics.py /tmp/si_sheets/Figure_S11.csv
```

- `fit_models.py`: per condition × ensemble: prior, median/mean shift, scale, offset+scale, OLS
  (b0, bc, bh), NNLS (bc, bh ≥ 0, free offset), WLS with the reported k_obs SD; region means;
  leave-one-segment-out influence; per-residue noise.
- `reachability.py`: non-negative least squares over frames (free offset) = the best any
  reweighting could do; fraction of single frames positively correlated with the targets.
- `peptide_kinetics.py`: whole-peptide uptake series from SI Figure S11 per condition, fraction of
  plateau per time point, single-exponential refit against the reported `k_exp`.
- `xlsx_to_csv.py`: stdlib dump of every SI worksheet (no openpyxl in the venv).
