# Phase 2 Implementation Plan: Analysis Script Families

## Context

References:
- plans/REFACTOR_EXAMPLES_PLAN.md (Overall refactor strategy)
- plans/implementation_plan.md (Summary of phases)

Phase 1 established the common-modules architecture and validated it with a 90%+ code reduction in `analyse_loss_ISO_TRI_BI.py`. Phase 2 applies the same pattern to the remaining **Analysis Script** families. All target scripts are currently long (600–1665 lines) and share large blocks of identical code across 3 experiments. The common module infrastructure (`config`, `loading`, `analysis`, `plotting`, `paths`, `optimization`) already exists — this phase fills the remaining gaps and refactors the scripts.

**Key learnings from pilot (applied throughout):**
0. Apply plan in stages to avoid overwhelming changes and allow incremental validation
1. Archive originals to `_archive/` *before* overwriting
2. Verify scripts actually run after refactoring — manual check required
3. Use `results_prefix` + `find_most_recent_dir()` for timestamped result directories
4. Config must include ALL keys referenced in data (check colour palettes against data)
5. 2D sweeps have deeper nesting (7 levels) — handle the `bv_reg_fn` dimension explicitly
6. Some plotting functions don't exist yet — add them incrementally rather than assuming

---

## Scripts in Scope

| Script | Exp 1 | Exp 2 | Exp 3 | Lines (each) | Priority |
|---|:---:|:---:|:---:|:---:|:---:|
| `analyse_split_*.py` | `analyse_split_ISO_TRI_BI.py` | `analyse_split_CrossVal.py` | — | ~550 | 1 (easiest) |
| `process_optimisation_results.py` | ✓ | ✓ | ✓ | ~700 | 2 |
| `score_models_ISO_TRI_BI.py` | ✓ | ✓ | ✓ | ~700 | 3 |
| `weights_validation_ISO_TRI_BI_precluster.py` | ✓ | ✓ | `*_2D_BV.py` | 1000–1665 | 4 |
| `recovery_analysis_ISO_TRI_BI_precluster.py` | ✓ | ✓ | `*_2D_BV.py` | ~600 | 5 |

> **Note**: `plot_compare_jaxENT_HDXer.py`, `plot_selected_models_ISO_TRI_BI.py`, and `analyse_scores_mixed_linear_model.py` were not explored in this research pass and will be planned separately.

---
<!-- 
## Step 0: Add Missing Common Module Functions

Do this in **one pass** across all modules before touching any script. Reference implementations come from the existing scripts (noted below).

### `jaxent/examples/common/analysis.py`

| Function to add | Source / Notes |
|---|---|
| `_is_valid_array(arr)` | Helper — from Exp3 `score_models` |
| `_align_uptake_shape(arr, target_ndim)` | Helper — from Exp3 `score_models` |
| `calculate_mse(pred, exp)` | From all 3 `score_models`; use Exp3's robust version with shape guards |
| `get_experimental_uptake(data)` | Identical across all 3 `score_models`; extracts `dfrac` from `HDX_peptide` list |
| `find_best_convergence_threshold(history)` | Identical in Exp3 `weights_validation` and Exp3 `recovery_analysis`; returns step index of min val_loss |
| `extract_frame_weights_kl(results, n_frames_by_ensemble=None)` | Unify Exp1 and Exp2 `weights_validation` — adds optional frame-count guard (Exp2 behaviour) and `convergence_fraction` field |
| `extract_final_weights(results)` | From Exp2 `weights_validation` (standalone); Exp1 has inline equivalent |
| `extract_final_weights_2d(results)` | From Exp3 `weights_validation`; adds `weight_min/max/mean/std` stats |
| `extract_weights_over_convergence_steps(results)` | Exp1 `weights_validation` only; samples at 10/30/50/70/90/100% of training |
| `compute_pairwise_kld_between_splits(results, per_step=False)` | Unify Exp1 (final weights only) and Exp2 (per convergence step) |
| `compute_pairwise_kld_between_splits_2d(results)` | Exp3 `weights_validation` only |
| `compute_sequential_maxent_kld(weights_data)` | Exp1 `weights_validation` only; KLD between consecutive maxent values |
| `analyze_conformational_recovery(clustering_data, results, metric='jsd', best_step_only=False)` | Unify all 3 `recovery_analysis`: `metric='ratio'` for Exp1 binary recovery, `metric='jsd'` for Exp2/3; `best_step_only=True` for Exp3 (selects step via `find_best_convergence_threshold`) |

**Already in `common/analysis.py` — scripts just need to import from there:**
`kl_divergence`, `effective_sample_size`, `calculate_cluster_ratios`, `calculate_dMSE`, `calculate_work_metrics`, `calculate_recovery_percentage`, `calculate_recovery_JSD`

### `jaxent/examples/common/loading.py`

| Function to add / extend | Source / Notes |
|---|---|
| `load_processed_run_info(processed_data_dir, ensemble_pattern)` | Scan `_processed_*` subdirs, apply regex to run_id dir names, return `(all_run_info, unique_configs)` — refactors the identical 60-line loop in all 3 `score_models` scripts |
| `load_split_data(split_dir)` | From `analyse_split_*.py` (both identical except `np.vstack` vs `np.array`); standardise to `np.array` |
| Extend `load_clustering_results()` | Add support for Exp1 flat-CSV format: `cluster_assignments_<ensemble>.csv` with `cluster_assignment` column (currently only handles Exp2/3 nested-subdir format) |

**Already in `common/loading.py`:** `load_all_optimization_results_with_maxent`, `load_all_optimization_results_2d`, `load_clustering_for_ensemble`, `load_features_and_topology`, `load_experimental_data`

### `jaxent/examples/common/plotting.py`

| Function to add | Source / Notes |
|---|---|
| `create_violin_plots(scores_df, output_dir, metric_columns=None)` | Identical across all 3 `score_models` |
| `plot_metric_heatmap(df, metric, convergence_rates, output_dir, cmap, vmin, vmax, title_prefix)` | Generalises `plot_recovery_heatmap` (Exp1), `plot_recovery_jsd_heatmap` (Exp2), `plot_ess_heatmaps`, `plot_kld_uniform_heatmaps`, `plot_kld_between_splits_heatmap` (Exp2) — all share the same maxent × convergence_step grid structure |
| `plot_weight_distribution_lines(weights_data, output_dir, split_name_mapping=None)` | Unify Exp1 (2×2 subplot grid) and Exp2 (single fig per split_type) `weights_validation` plots |
| `plot_2d_heatmaps_grid(df, output_dir, metric, metric_label, cmap=None, vmin=None, vmax=None)` | Nearly identical between Exp3 `weights_validation` and Exp3 `recovery_analysis` |
| `plot_1d_slices_2d_sweep(df, output_dir, metric, metric_label)` | `plot_1d_slices_maxent_vs_bvreg` (Exp3 weights) and `plot_1d_slices` (Exp3 recovery) are structurally identical |
| `plot_split_distributions(splits_df, output_dir)` | 100% identical between both `analyse_split_*.py` |
| `plot_enhanced_split_heatmap(splits_df, output_dir)` | 100% identical |
| `plot_split_heatmap(splits_df, output_dir)` | 100% identical |
| `plot_uptake_heatmap(peptides, dfracs, title, filename, split_type)` | 100% identical |
| `plot_combined_uptake_comparison(train_top, train_dfrac, val_top, val_dfrac, split_name, output_dir)` | 100% identical |
| `plot_gap_analysis(splits_df, output_dir)` | 100% identical |

### `jaxent/examples/common/paths.py`

| Function to add | Notes |
|---|---|
| `derive_processed_output_dir(results_dir)` | Returns `Path(results_dir).parent / ("_processed_" + Path(results_dir).name)` — used by all 3 `process_optimisation_results.py` |

### `config.yaml` additions

Add `ground_truth_ratios` to `ScoringConfig` in `1_IsoValidation_OMass/config.yaml`:
```yaml
scoring:
  ground_truth_ratios:
    open: 0.4
    closed: 0.6
``` -->

● Bash(.venv/bin/python -c "from jaxent.examples.common import analysis, loading, plotting, paths; print('Import OK')" 2>&1)                                                     
  ⎿  Import OK                                                                                                                                                                 
                                                                                                                                                                                 
● Bash(.venv/bin/python -c "                                                                                                                                                     
      from jaxent.examples.common import analysis, loading, plotting, paths…)
  ⎿  All functions present                                                                                                                                                       
                                                                                                                                                                               
● Bash(.venv/bin/python -c "                                                                                                                                                     
      from jaxent.examples.common.config import ExperimentConfig…)
  ⎿  Exp1 config OK — recovery_metric=ratio, ground_truth_ratios={'open': 0.4, 'closed': 0.6}                                                                                    
     Exp2 config OK — recovery_metric=jsd                                                                                                                                      
     Exp3 config OK — recovery_metric=jsd                                                                                                                                        
     Config OK
                                                                                                                                                                                 
● All three verification checks pass. Here's a summary of what was completed in Step 0:                                                                                          
  
  common/config.py — Added ground_truth_ratios and recovery_metric fields to ScoringConfig                                                                                       
                                                                                                                                                                               
  common/paths.py — Added derive_processed_output_dir

  common/analysis.py — Added 13 public + 3 private functions: _is_valid_array, _align_uptake_shape, get_experimental_uptake, calculate_mse, find_best_convergence_threshold,
  extract_frame_weights_kl, extract_final_weights, extract_final_weights_2d, extract_weights_over_convergence_steps, compute_pairwise_kld_between_splits,
  compute_pairwise_kld_between_splits_2d, compute_sequential_maxent_kld, analyze_conformational_recovery

  common/loading.py — Extended load_clustering_results with Exp1 flat-CSV detection, added load_processed_run_info and load_split_data

  common/plotting.py — Added 11 functions: create_violin_plots, plot_metric_heatmap, plot_weight_distribution_lines, plot_2d_heatmaps_grid, plot_1d_slices_2d_sweep, and all 6
  split-analysis plots from analyse_split_ISO_TRI_BI.py

  3 YAML config files — Added recovery_metric (and ground_truth_ratios for Exp1)

  All verifications pass. I won't proceed to Steps 1–5 as instructed.

---
<!-- 
## Step 1: Refactor `analyse_split_*.py` — both scripts → ~20 lines each

**Files:**
- `jaxent/examples/1_IsoValidation_OMass/analysis/analyse_split_ISO_TRI_BI.py`
- `jaxent/examples/2_CrossValidation/analysis/analyse_split_CrossVal.py`

**Why first:** All 6 plot functions and the data loader are 100% identical — zero experiment-specific logic remains in the scripts after extraction. This is the cleanest possible case.

**Approach:**
1. Archive originals to `_archive/`
2. Extract all functions to `common/plotting.py` and `common/loading.py` (Step 0 above)
3. Each script becomes `argparse` + path setup + calls to common functions:
   ```python
   from jaxent.examples.common import loading, plotting
   all_splits_df = loading.load_split_data(args.split_dir)
   plotting.plot_split_distributions(all_splits_df, output_dir)
   plotting.plot_enhanced_split_heatmap(all_splits_df, output_dir)
   # ... remaining 4 plot calls
   ```

**Differences remaining in script (none — only config):**
- Experiment-specific default paths (handled by `--split-dir` / `--output-dir` args) -->

● Both scripts ran cleanly with no errors. Summary:                                                                                                                              
                                                                                                                                                                                 
  analyse_split_ISO_TRI_BI.py — processed 5 split types (stratified, sequence, spatial, sequence_cluster, random), 3 splits each, plots saved to _analysis_split_iso_tri_bi/     
                                                                                                                                                                                 
  analyse_split_CrossVal.py — same 5 split types, 3 splits each, plots saved to _analysis_split_CrossVal/                                                                        
                                                                                                                                                                                 
  Step 1 of Phase 2 is fully verified.                               

---

## Step 2: Refactor `process_optimisation_results.py` — all 3 experiments → ~200–250 lines each

**Files:**
- `jaxent/examples/1_IsoValidation_OMass/analysis/process_optimisation_results.py`
- `jaxent/examples/2_CrossValidation/analysis/process_optimisation_results.py`
- `jaxent/examples/3_CrossValidationBV/analysis/process_optimisation_results.py`

**Approach:**
1. Archive originals
2. Replace inline definitions with imports from common:
   - `BV_uptake_ForwardPass_frames` ← `common.optimization` (already there, just not used)
   - `kl_divergence`, `calculate_cluster_ratios` ← `common.analysis`
   - `load_all_optimization_results_with_maxent` or `load_all_optimization_results_2d` ← `common.loading`
   - `load_clustering_results` ← `common.loading` (after Exp1 flat-CSV support added)
   - `derive_processed_output_dir` ← `common.paths`

**Differences that stay in each script:**
| Aspect | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| Ensemble names | `["ISO_TRI", "ISO_BI"]` | `["AF2_MSAss", "AF2_filtered"]` | `["AF2_MSAss", "AF2_filtered"]` |
| Feature file naming | `ensemble.lower()` | `ENSEMBLE_FEATURE_MAP` dict | `ENSEMBLE_FEATURE_MAP` dict |
| Results loading | `with_maxent` (1D) | `with_maxent` (1D) | `2d` (2D sweep) |
| Prior computation | framewise → average | framewise → average | average features → forward |
| Extra output files | — | — | `bv_bc.npy`, `bv_bh.npy` |
| Clustering CSV format | flat CSVs | nested subdirs | nested subdirs (cross-experiment) |

**Validation:** Check that `_processed_*` output directories contain the same `.npy` and `.csv` files as pre-refactor runs.

---

## Step 3: Refactor `score_models_ISO_TRI_BI.py` — all 3 experiments → ~150–200 lines each

**Files:**
- `jaxent/examples/1_IsoValidation_OMass/analysis/score_models_ISO_TRI_BI.py`
- `jaxent/examples/2_CrossValidation/analysis/score_models_ISO_TRI_BI.py`
- `jaxent/examples/3_CrossValidationBV/analysis/score_models_ISO_TRI_BI.py`

**Approach:**
1. Archive originals
2. Replace inline definitions with imports:
   - `calculate_mse`, `get_experimental_uptake`, `calculate_dMSE`, `calculate_work_metrics`, `calculate_recovery_percentage` ← `common.analysis`
   - `load_features_and_topology`, `load_experimental_data`, `load_processed_run_info` ← `common.loading`
   - `create_violin_plots` ← `common.plotting`
   - `ExperimentConfig` ← `common.config`

**Differences that stay in each script:**
| Aspect | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| Ensemble regex group | `(ISO_BI\|ISO_TRI)` | `(AF2_MSAss\|AF2_filtered)` | `(AF2_MSAss\|AF2_filtered)` + `bv_reg` groups |
| Recovery metric | Binary ratio (gt from config `ground_truth_ratios`) | JSD with state map | JSD with state map |
| State mapping/ratios | `config.scoring.ground_truth_ratios` | `config.scoring.state_mapping` + `state_ratios.json` | Same as Exp 2 |
| Extra score columns | — | — | `bv_reg_function`, `bv_reg_value` |

The `ScoringConfig` in `config.yaml` already provides `ensemble_feature_map`, `ensemble_clustering_map`, `state_mapping`, `scoring_weights`.

**Validation:** Compare `model_scores.csv` row counts and column values against pre-refactor baselines.

---

## Step 4: Refactor `weights_validation_ISO_TRI_BI_precluster.py` — 2×1D + 1×2D → ~100–200 lines each

**Files:**
- `jaxent/examples/1_IsoValidation_OMass/analysis/weights_validation_ISO_TRI_BI_precluster.py`
- `jaxent/examples/2_CrossValidation/analysis/weights_validation_ISO_TRI_BI_precluster.py`
- `jaxent/examples/3_CrossValidationBV/analysis/weights_validation_ISO_TRI_BI_2D_BV.py`

**Approach:**
1. Archive originals
2. All analysis functions move to `common/analysis.py` (Step 0)
3. All plotting functions move to `common/plotting.py` (Step 0)
4. Replace inline copies: `kl_divergence`, `effective_sample_size`, `load_all_optimization_results_with_maxent/2d`, `load_clustering_for_ensemble`

**Differences that stay in each script:**
| Aspect | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| Results loading | `with_maxent`, auto-discover split_types | `with_maxent`, explicit split_types | `load_all_optimization_results_2d` |
| Weight extraction | `extract_frame_weights_kl` + `extract_weights_over_convergence_steps` | `extract_frame_weights_kl` (with frame guard) + `extract_final_weights` | `extract_final_weights_2d` + `find_best_convergence_threshold` |
| Pairwise KLD | `compute_pairwise_kld_between_splits(results, per_step=False)` | `compute_pairwise_kld_between_splits(results, per_step=True)` | `compute_pairwise_kld_between_splits_2d(results)` |
| Sequential KLD | `compute_sequential_maxent_kld` (Exp1 only) | — | — |
| Plot type | 1D line + heatmaps | 1D heatmaps | 2D heatmaps + 1D slices |

---

## Step 5: Refactor `recovery_analysis_ISO_TRI_BI_precluster.py` — 2×1D + 1×2D → ~80–150 lines each

**Files:**
- `jaxent/examples/1_IsoValidation_OMass/analysis/recovery_analysis_ISO_TRI_BI_precluster.py`
- `jaxent/examples/2_CrossValidation/analysis/recovery_analysis_ISO_TRI_BI_precluster.py`
- `jaxent/examples/3_CrossValidationBV/analysis/recovery_analysis_ISO_TRI_BI_2D_BV.py`

**Approach:** After Step 0 adds `analyze_conformational_recovery`, these scripts become thin wrappers:

```python
recovery_df = analysis.analyze_conformational_recovery(
    clustering_data, results,
    metric=config.scoring.recovery_metric,   # 'ratio' or 'jsd'
    best_step_only=is_2d_sweep,              # True for Exp3
)
```

**Differences that stay in each script:**
| Aspect | Exp 1 | Exp 2 | Exp 3 |
|---|---|---|---|
| Results loading | `with_maxent` | `with_maxent` | `2d` |
| `metric` | `'ratio'` | `'jsd'` | `'jsd'` |
| `best_step_only` | `False` | `False` | `True` |
| Plot function | `plot_metric_heatmap(cmap='PiYG', metric='open_recovery')` | `plot_metric_heatmap(cmap='RdYlGn', metric='recovery_percent')` | `plot_2d_heatmaps_grid` + `plot_1d_slices_2d_sweep` |

Add `recovery_metric: ratio` to Exp1's `ScoringConfig` in `config.yaml`; Exp2/3 already use JSD.

---

## Verification Plan (Each Step)

After each step:

1. **Module import check:**
   ```bash
   ./venv/bin/python -c "from jaxent.examples.common import analysis, loading, plotting, paths; print('OK')"
   ```

2. **Run refactored script** from its original directory, compare key output files against `_archive/` baseline:
   - `analyse_split_*.py`: plot file count and directory structure
   - `process_optimisation_results.py`: `.npy` files per run, `cluster_ratios.csv` values
   - `score_models_ISO_TRI_BI.py`: `model_scores.csv` row count and score column distributions
   - `weights_validation` / `recovery_analysis`: plot file count, CSV row counts

3. **Cross-check config completeness:** Run each script, confirm no `KeyError` from missing config keys (color palettes especially).

---

## Critical Files

| File | Role |
|---|---|
| `jaxent/examples/common/analysis.py` | Primary target for new functions (Step 0) |
| `jaxent/examples/common/loading.py` | Add `load_processed_run_info`, `load_split_data`; extend `load_clustering_results` |
| `jaxent/examples/common/plotting.py` | Add all new plot functions (6 split + 5 scoring/weights) |
| `jaxent/examples/common/paths.py` | Add `derive_processed_output_dir` |
| `jaxent/examples/1_IsoValidation_OMass/config.yaml` | Add `ground_truth_ratios`, `recovery_metric` to `scoring` block |
| `jaxent/examples/2_CrossValidation/config.yaml` | Add `recovery_metric: jsd` to `scoring` block |
| `jaxent/examples/3_CrossValidationBV/config.yaml` | Add `recovery_metric: jsd` to `scoring` block |
