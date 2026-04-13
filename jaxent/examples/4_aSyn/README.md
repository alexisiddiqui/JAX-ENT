# Example 4 — aSyn condition-wise BV analysis

## Overview

This example runs cross-validation for **alpha-synuclein (aSyn)** using residue-level protection factors (PF) across four conditions:

- `Tris_only`
- `Extracellular`
- `Intracellular`
- `Lysosomal`

Conditions are treated as the top-level comparison axis (ensemble-like grouping).

Recovery is PF-based:

`recovery (%) = (1 - sqrt(JSD(logPf_pred, logPf_exp))) * 100`

## Data and paths

- Shared features: `data/_cluster_aSyn/features/{features.npz, topology.json}`
- Split data: `fitting/_datasplits/<Condition>/<split_type>/split_XXX/`
- Optimization outputs: `fitting/_optimise_aSyn_BV_<timestamp>/`
- Config: `config.yaml` (includes placeholder condition colors under `style.ensemble_colors`)

## Analysis scripts

Located in `analysis/`:

- `recovery_analysis_aSyn_conditions_2d_bv.py`
- `weights_validation_aSyn_conditions_2d_bv.py`
- `analyse_loss_aSyn_conditions_2d_bv.py`
- `process_optimisation_results_aSyn_conditions.py`
- `score_models_aSyn_conditions.py`
- `analyse_scores_mixed_linear_model_aSyn_conditions.py`
- `plot_selected_models_aSyn_conditions.py`
- `extract_selected_models_aSyn_conditions.py`

## Runners

Run from `jaxent/examples/4_aSyn`:

```bash
# Fit + then automatically run the comprehensive analysis pipeline
bash fitting/run_maxent_parallel_BV_aSyn_conditions.sh

# Quick validation (recovery + weights + loss)
bash validate_run.sh [results_dir]

# Full pipeline (runs all analysis scripts)
bash run_comprehensive_analysis.sh [results_dir]
```

To run fitting only (skip automatic analysis), add:

```bash
bash fitting/run_maxent_parallel_BV_aSyn_conditions.sh --skip-analysis
```

If `results_dir` is omitted, both scripts default to:

`fitting/_optimise_aSyn_BV_20260402_021807`

Both runners are **continue-on-error** and write per-step logs to:

`<results_dir>/logs/`

## Main outputs

Validation:

- `analysis/_analysis_<results_basename>/` (PF recovery plots)
- `analysis/_analysis_weights_2d_<results_basename>/`
- `analysis/_analysis_loss_output/`

Comprehensive:

- `analysis/_analysis_<results_basename>/` (recovery outputs)
- `analysis/_analysis_weights_2d_<results_basename>/`
- `analysis/_analysis_loss_output/`
- `fitting/_processed_<results_basename>/`
- `fitting/_processed_<results_basename>/_scores__processed_<results_basename>/model_scores.csv`
- `fitting/_processed_<results_basename>/_analysis__scores__processed_<results_basename>/`
- `fitting/_processed_<results_basename>/_analysis__scores__processed_<results_basename>_filtered/`
- `fitting/_processed_<results_basename>/_extracted__processed_<results_basename>/`
