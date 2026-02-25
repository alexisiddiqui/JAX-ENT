# Verification Findings: Analysis Scripts Plots

This document summarizes the plot verification for the refactored analysis scripts against their `_archive` equivalents. The goal was to ensure the refactored versions produce the exact same visual outputs (apart from explicitly excluded scripts).

## Summary of Findings

1. **`analyse_loss_*.py`**: 🔴 **DIFFERENT**
   - **Archived**: Generated `error_vs_convergence.png`, `final_performance_comparison.png`, `split_variability.png`, and a series of maxent/convergence heatmaps.
   - **Refactored**: ONLY generates the heatmaps (`train_loss_convergence_maxent_heatmap`, `val_loss_convergence_maxent_heatmap`, `model_score_heatmap`). 
   - **Explanation**: The 1D plotting functions (`plot_loss_convergence` and `plot_split_variability`) were intentionally removed during Phase 1 refactoring because they were missing in the new `common.plotting` module (as documented in `REFACTOR_EXAMPLES_PLAN.md`).

2. **`analyse_split_*.py`**: 🟢 **IDENTICAL**
   - **Archived**: Generated `peptide_split_distribution.png`, `enhanced_peptide_split_heatmap.png`, `gap_analysis.png`, and various `uptake_*.png` heatmaps.
   - **Refactored**: Delegates exactly to the same drawing logic now housed in `common.plotting`. The resulting plots are identical.

3. **`process_optimisation_results.py`**: 🟢 **IDENTICAL (No Plots)**
   - **Archived**: Did not generate any plots (processed data to `.npy` arrays).
   - **Refactored**: Does not generate any plots. Behavior is identical.

4. **`score_models_*.py`**: 🟢 **IDENTICAL**
   - **Archived**: Generated violin plots using `plt.savefig`.
   - **Refactored**: Calls `plotting.create_violin_plots()` which perfectly mirrors the original logic. 

5. **`weights_validation_*.py`**: 🔴 **DIFFERENT / INCOMPLETE**
   - **Archived**: Generated numerous KLD plots and weight distribution panels.
   - **Refactored**: Generates `weight_distributions_convergence_panels_*.png`, `weight_distributions_maxent_panels_*.png`, and KLD plots. However, the exact plot output shapes and quantity differed in testing. The original file also contained commented-out stubs (`[Additional plotting functions would go here - copying from original script]`), suggesting it was an incomplete refactoring attempt copied into `_archive`.

6. **`plot_compare_jaxENT_HDXer.py`**: ⚪ **N/A**
   - **Status**: Not yet refactored.
   - **Explanation**: There is no `.original` file present in the `_archive/` directories of any experiment because Phase 2 of the refactoring plan has not yet reached the visualization scripts.

## Next Steps Recommended
* **Restore Missing 1D Plots:** Implement the missing `plot_loss_convergence` and `plot_split_variability` functions into `common/plotting.py` so that `analyse_loss_*.py` can generate identical outputs as the archive.
* **Refactor Visualization Scripts:** Proceed to implement the remainder of Phase 2, which includes fully refactoring `plot_compare_jaxENT_HDXer.py` and `plot_selected_models_ISO_TRI_BI.py`.
* **Fix Missing Dependencies in Runtime:** Ensured `pyyaml` is listed in project requirements, as it broke on the first run of the refactored script.
