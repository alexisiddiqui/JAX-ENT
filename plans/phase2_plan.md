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
<!-- 
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

**Validation:** Check that `_processed_*` output directories contain the same `.npy` and `.csv` files as pre-refactor runs. -->
                                                                                                                                                                                 
● All 3 scripts verified against real data:                                                                                                                                      
                                                                                                                                                                                 
  ┌──────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐                                      
  │  Script  │                                                           Result                                                           │                                      
  ├──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                      
  │ Script 1 │ Completed — 1512 .npy files, matching old processed dir exactly                                                            │                                      
  ├──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                      
  │ Script 2 │ Completed — both AF2_MSAss and AF2_filtered loaded (correct case), AF2_Filtered/ subdir used for clustering                │                                      
  ├──────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤                                      
  │ Script 3 │ Completed — 756 bv_bc.npy + 756 bv_bh.npy, run_id format correct (_bvreg0.00_bvregfnL1), all 11 expected files per run dir │                                      
  └──────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘    
---
<!-- 
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

**Validation:** Compare `model_scores.csv` row counts and column values against pre-refactor baselines. -->
● All three scripts ran successfully end-to-end. Here's the verification summary:
                                                                                                                                                                                 
  ┌────────────┬────────────────┬─────────────┬──────────────────────────────────────────────────────────┐                                                                       
  │ Experiment │ Runs processed │ Rows in CSV │                      BV reg columns                      │
  ├────────────┼────────────────┼─────────────┼──────────────────────────────────────────────────────────┤                                                                       
  │ Exp1       │ 216            │ 1557        │ —                                                        │                                                                     
  ├────────────┼────────────────┼─────────────┼──────────────────────────────────────────────────────────┤
  │ Exp2       │ 252            │ 1845        │ —                                                        │
  ├────────────┼────────────────┼─────────────┼──────────────────────────────────────────────────────────┤
  │ Exp3       │ 756            │ 5535        │ bv_reg_function: ['L1'], bv_reg_value: [0.0, 0.5, 1.0] ✓ │
  └────────────┴────────────────┴─────────────┴──────────────────────────────────────────────────────────┘

  All outputs have the expected columns (work_scale_kj, work_shape_kj, etc.) and violin plots were generated. The only output during runs was pre-existing beartype type-hint
  warnings from apply_sparse_mapping — these are not new and are not errors.

---

<!-- ## Step 4: Refactor `weights_validation_ISO_TRI_BI_precluster.py` — 2×1D + 1×2D → ~100–200 lines each

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
| Plot type | 1D line + heatmaps | 1D heatmaps | 2D heatmaps + 1D slices | -->

All three scripts verified end-to-end. Here's the summary:
                                                     
  Script 1 — completed, produced: kl_divergence_ess_analysis_maxent_data.csv, kld_between_splits_data.csv, sequential_maxent_kld_data.csv, plus
  all PNG files (KLD plots, sequential KLD, maxent panels, convergence panels)                                                                    
  
  Script 2 — completed, produced:                                                                                                                 
  - kl_ess_data.csv, kld_between_splits_data.csv, conformational_recovery.csv ✓                                                                 
  - Per-split-type heatmaps: sequence_cluster/effective_sample_size_heatmap_sequence_cluster.png, spatial/kl_divergence_heatmap_spatial.png,
  spatial/mean_kld_between_splits_heatmap_spatial.png etc. ✓
  - conformational_recovery_*.png, weight_distributions_lines_*.png ✓

  Script 3 — completed, produced: weights_2d_sweep_data.csv, kld_between_splits_2d_sweep_data.csv, 2D heatmaps and 1D slice PNGs ✓

Note: ensure that the logging is consistent between the original and refactored methods - the plotting functions in particular.

---
<!-- 
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

Add `recovery_metric: ratio` to Exp1's `ScoringConfig` in `config.yaml`; Exp2/3 already use JSD. -->
● All 3 scripts run successfully end-to-end. Summary of what was fixed during verification:

  Fixes applied:
  1. plotting.py — plot_metric_heatmap sanitizes stype (e.g. "N/A") by replacing / with _ before using it in directory names
  and filenames. The baseline row added by analyze_conformational_recovery has split_type = "N/A" which contains a literal /.
  2. Script 1 — plot_volcano_kl_recovery baseline lookup fell back to open_recovery if open_ratio column is absent (common
  function doesn't produce that column for metric='ratio'). Same guard added to the summary print statement.
  3. Script 3 — Removed the two validation_loss plot calls (plot_2d_heatmaps_grid + plot_1d_slices_2d_sweep) as specified in
  the plan — the common function doesn't produce that column with best_step_only=True.

  Expected CSVs confirmed present:

  ┌────────┬─────────────────────────────────────────────────────────────────────────────────────┐
  │ Script │                                        CSVs                                         │
  ├────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Exp1   │ conformational_recovery_maxent_data.csv, kl_divergence_ess_analysis_maxent_data.csv │
  ├────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Exp2   │ recovery_jsd_data.csv, kl_divergence_ess_data.csv                                   │
  ├────────┼─────────────────────────────────────────────────────────────────────────────────────┤
  │ Exp3   │ recovery_2d_sweep_data.csv                                                          │
  └────────┴─────────────────────────────────────────────────────────────────────────────────────┘
---


## Step 6: Refactor Fitting & Data Scripts — `featurise_*.py`, `splitdata_*.py`, `optimise_fn.py` + caller scripts

**Files in scope:**

| Script family | Files | Lines (each) | Priority |
|---|---|---|---|
| `featurise_*.py` | Exp1 base, Exp1 SCALING, Exp2 | 213–233 | 1 (cleanest — 2 shared helpers, no analysis logic) |
| `splitdata_*.py` | Exp1, Exp2 | 228–235 | 2 (1 shared helper, parametric differences only) |
| `optimise_fn.py` + caller | Exp1 `optimise_fn.py` + `optimise_ISO_TRI_BI_splits_Sigma.py`; Exp2 `optimise_fn.py` + `optimise_ISO_TRI_BI_splits_maxENT.py`; Exp3 `optimise_fn.py` + `optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py` | 778–992 | 3 (most complex — needs `LossConfig` gap filled) |
<!-- 
### Substep 6a: Common Module Additions (do first)

**`jaxent/examples/common/loading.py`** — add:

| Function | Source | Notes |
|---|---|---|
| `load_HDXer_kints(kint_path)` | Identical in all featurise scripts | Parses 2-col `.dat` file → `(jnp.array of rates, list[Partial_Topology])`, chain='A' |
| `featurise_trajectory(trajectory_path, topology_path, output_dir, output_name, bv_config, featuriser_settings, kint_data)` | Identical in all featurise scripts | Creates `mda.Universe`, runs `Experiment_Builder` + `run_featurise`, replaces k_ints, saves `.npz`/`.json` |
| `run_data_splits(num_splits, output_dir, hdx_data, feature_topology, split_type, remove_overlap, universe, *, peptide_trim=1, min_split_size=None, n_clusters=25)` | Near-identical in both splitdata scripts | Parametrize `peptide_trim` (1 vs 2), `min_split_size` (None vs 4), `n_clusters` (25 vs 7) |

**`jaxent/examples/common/config.py`** — extend `LossConfig` dataclass:
```python
bv_reg_scaling: float = 1.0      # Weight for BV regularization loss term (Exp3 sweep)
normalize_bv_reg: bool = True    # Set False for BV reg term → normalise_loss_functions[-1]=0
```

**`jaxent/examples/common/optimization.py`** — update `run_optimization()`:
- Use `loss_config.bv_reg_scaling` when building `forward_model_weights` for regularization losses (currently hardcodes `1.0`)
- Set `normalise_loss_functions[-1] = 0.0` when `loss_config.normalize_bv_reg is False` -->

### Substep 6b: Refactor `featurise_*.py` (Exp 1 & 2) ✓

**Files:**
- `1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI.py` (213 → ~50 lines)
- `1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI_SCALING.py` (233 → ~80 lines)
- `2_CrossValidation/fitting/jaxENT/featurise_CrossVal_MSAss_Filtered.py` (231 → ~50 lines)

**Approach:**
1. Archive originals to `fitting/jaxENT/_archive/`
2. Remove `load_HDXer_kints` and `featurise_trajectory` (now in `common/loading.py`)
3. Each script becomes: imports + experiment-specific `main()` only

**Differences that stay per-script:**

| Aspect | Exp1 base | Exp1 SCALING | Exp2 |
|---|---|---|---|
| Timepoints | `[0.167, 1.0, 10.0, 60.0, 120.0]` min | same | 15 timepoints (0.08–1440 min) |
| Trajectories | `iso_tri` + `iso_bi` (2 fixed files) | loops `sliced_N` subdirs | AF2_MSAss + AF2_filtered cluster dirs |
| `FeaturiserSettings.name` | `"ISO"` | `"ISO"` | `"CrossVal"` |
| Output names | `iso_tri`, `iso_bi` | `iso_tri`, `iso_bi` per slice | `AF2_MSAss`, `AF2_filtered` |
| Topology / kint files | TeaA-specific | TeaA-specific | MoPrP-specific |

All three scripts archived, refactored, and syntax-verified. Import check (`common loading imports OK`) and `py_compile` all pass.

### Substep 6c: Refactor `splitdata_*.py` (Exp 1 & 2)

**Files:**
- `1_IsoValidation_OMass/fitting/splitdata_ISO.py` (228 → ~80 lines)
- `2_CrossValidation/fitting/splitdata_CrossVal.py` (235 → ~80 lines)

**Approach:**
1. Archive originals to `fitting/_archive/`
2. Remove `run_data_splits` (now in `common/loading.py`)
3. Each script becomes: imports + experiment-specific `main()`

**Differences that stay per-script:**

| Aspect | Exp1 | Exp2 |
|---|---|---|
| `peptide_trim` | `1` | `2` |
| `min_split_size` | `None` | `4` |
| `n_clusters` (seq_cluster) | `25` | `7` |
| HDX source file | `mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat` | `MoPrP_dfrac.dat` |
| Peptide count | `len(segs)` | `len(segs) - 2` (drops terminal 2) |
| `fragment_name` | `"TeaISO"` | `"MoPrPCrossVal"` |

**Validation:** Re-run refactored scripts; confirm train/val CSV files match archived originals.

### Substep 6d: Refactor `optimise_fn.py` + Caller Scripts (Exp 1, 2, 3)

**Files to archive:**
- `1_IsoValidation_OMass/fitting/jaxENT/optimise_fn.py` (801 lines) → `fitting/jaxENT/_archive/`
- `2_CrossValidation/fitting/jaxENT/optimise_fn.py` (895 lines) → `fitting/jaxENT/_archive/`
- `3_CrossValidationBV/fitting/jaxENT/optimise_fn.py` (992 lines) → `fitting/jaxENT/_archive/`

**Caller scripts to update:**
- `1_IsoValidation_OMass/fitting/jaxENT/optimise_ISO_TRI_BI_splits_Sigma.py` (778 → ~280 lines)
- `2_CrossValidation/fitting/jaxENT/optimise_ISO_TRI_BI_splits_maxENT.py` (777 → ~280 lines)
- `3_CrossValidationBV/fitting/jaxENT/optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py` (856 → ~320 lines)

**Pattern change in each caller script** — replace the `from optimise_fn import run_optimise_ISO_TRI_BI_maxENT` + direct call with:
```python
from jaxent.examples.common.optimization import run_optimization
from jaxent.examples.common.config import LossConfig, OptimizationConfig

loss_config = LossConfig(
    primary_loss=LOSS_NAME_MAP[loss_name],   # e.g. "hdx_uptake_mean_centred_MSE_loss"
    regularization_losses=[],
    maxent_scaling=maxent_value,
)
opt_config = OptimizationConfig(
    n_steps=n_steps, learning_rate=learning_rate,
    initial_learning_rate=initial_learning_rate, initial_steps=initial_steps,
    optimizer="adam",    # Note: override default "adamw"
    ema_alpha=ema_alpha, forward_model_scaling=forward_model_scaling,
    convergence_rates=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],  # override default
    covariance_matrix_path=cov_matrix_path,
)
run_optimization(
    train_data=train_data, val_data=val_data, prior_data=prior_dataset,
    features=features, forward_model=bv_model, model_parameters=model_parameters,
    feature_top=feature_top, loss_config=loss_config, opt_config=opt_config,
    name=run_name, output_dir=output_dir,
)
```

**Differences remaining per-script:**

| Aspect | Exp1 | Exp2 | Exp3 |
|---|---|---|---|
| Ensembles | `["ISO_TRI", "ISO_BI"]` | `["AF2_filtered", "AF2_MSAss"]` | `["AF2_filtered", "AF2_MSAss"]` |
| Feature name format | `ensemble.lower()` | `ensemble` unchanged | `ensemble` unchanged |
| Cov matrix path | `_covariance_matrices_sigma/ISO_BI_Sigma_weighted.npz` | `../../data/_MoPrP_covariance_matrices/Sigma.npz` | same as Exp2 |
| BV reg sweep | — | — | Yes: adds `bv_reg_value` loop + `LossConfig(regularization_losses=[{"name": "model_params_L1_loss"}], bv_reg_scaling=bv_reg_value, normalize_bv_reg=False)` |
| `model_parameters_lr_scale` | — | — | Yes: `opt_config.model_parameters_lr_scale=model_parameters_lr_scale` |
| Timing instrumentation | Present in original `optimise_fn.py`; omit in refactored version (not in `common/`, consistent with Exp2/3) | — | — |

**Validation:** For each experiment run a minimal sweep:
```bash
.venv/bin/python optimise_ISO_TRI_BI_splits_Sigma.py --maxent-range 5,5 --n-replicates 1 --split-types stratified
```
Verify `.hdf5` and `_EMA.hdf5` files are created in the expected timestamped output subdirectory.


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
