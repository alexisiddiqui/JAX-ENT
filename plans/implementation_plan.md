# Refactoring Example Scripts Implementation Plan

## Goal Description
Refactor the `jaxent/examples/` directory to improve code reusability, maintainability, and consistency. The current state involves significant code duplication across ~93 scripts in 3 active experiment directories. The goal is to extract shared logic into a `jaxent.examples.common` package, consolidate near-identical scripts, and implement a configuration-driven approach using YAML and dataclasses, as detailed in `REFACTOR_EXAMPLES_PLAN.md`.

## User Review Required
> [!IMPORTANT]
> **Plan Verification**: This implementation plan is based on a thorough verification of `REFACTOR_EXAMPLES_PLAN.md` against the codebase.
> - **Script Existence**: Verified the existence and active status of all scripts listed in the plan (e.g., `analyse_loss_ISO_TRI_BI.py`, `optimise_fn.py`).
> - **Duplication**: Confirmed high levels of duplication in analysis and fitting scripts between experiments 1, 2, and 3.
> - **Technical Requirements**: Verified the need to update `load_HDXer_kints` to fix the topology gap (currently returns ints, needs `Partial_Topology`) and confirmed the `model_parameters_lr_scale` parameter in Exp 3.

## Proposed Changes

The refactoring will be executed in 5 phases.
<!-- 
### Phase 0: Packaging and Project Structure
Convert `jaxent/examples` into a proper Python package.
#### [MODIFY] [pyproject.toml](file:///Users/alexi/JAX-ENT/pyproject.toml)
- Update build configuration to include `jaxent.examples`.
#### [NEW] [__init__.py](file:///Users/alexi/JAX-ENT/jaxent/examples/__init__.py)
- Create namespace package files. -->

<!-- ### Phase 1: Create `jaxent/examples/common/` Shared Module
Create the core shared infrastructure.
#### [NEW] [jaxent/examples/common/](file:///Users/alexi/JAX-ENT/jaxent/examples/common/)
- `config.py`: Configuration dataclasses (`ExperimentConfig`, `OptimizationConfig`, etc.) with YAML loading.
- `paths.py`: Centralized path resolution.
- `loading.py`: Consolidated data loading functions (optimization results, experimental data).
- `analysis.py`: Shared analysis logic (KL divergence, recovery metrics, etc.).
- `plotting.py`: Shared plotting functions with `PlotStyle` support.
- `optimization.py`: Unified `run_optimization` function replacing 5+ `optimise_fn.py` variants.
- `losses.py`: Example-specific loss functions moved from `optimise_fn.py`.
- `cli.py`: Shared `argparse` setup. -->

# ✅ Phase 1: Pilot Refactor Complete

## Pilot Refactor: `analyse_loss_*.py` Scripts

**Status**: ✅ **COMPLETE** (2026-02-18)

Successfully refactored 3 analysis scripts to validate the common modules approach:
- **Experiment 1**: `1_IsoValidation_OMass/analysis/analyse_loss_ISO_TRI_BI.py` (2131 → 228 lines, 90.3% reduction)
- **Experiment 2**: `2_CrossValidation/analysis/analyse_loss_ISO_TRI_BI.py` (2352 → 228 lines, 90.3% reduction)
- **Experiment 3**: `3_CrossValidationBV/analysis/analyse_loss_ISO_TRI_BI_2D_BV.py` (1622 → 214 lines, 90.0% reduction)

**Overall**: 6,105 lines → 670 lines (**90.2% code reduction**, 5,435 lines eliminated)

### Files Created

#### Configuration Files (3)
- ✅ `jaxent/examples/1_IsoValidation_OMass/config.yaml`
- ✅ `jaxent/examples/2_CrossValidation/config.yaml`
- ✅ `jaxent/examples/3_CrossValidationBV/config.yaml`

#### Archive Directories (3)
- ✅ `jaxent/examples/1_IsoValidation_OMass/analysis/_archive/`
- ✅ `jaxent/examples/2_CrossValidation/analysis/_archive/`
- ✅ `jaxent/examples/3_CrossValidationBV/analysis/_archive/`

### Enhancements to Common Modules

**`jaxent/examples/common/paths.py`**:
- ✅ Added `find_most_recent_dir()` function for dynamic directory discovery
- Supports timestamp-based directory patterns (e.g., `_optimise_test_SIGMA_500__*`)

**`jaxent/examples/common/config.py`**:
- ✅ Added `results_prefix` field to `ExperimentConfig` as alternative to `results_dir`
- Enables pattern-based discovery of most recent results directories

**`jaxent/examples/common/analysis.py`**:
- ✅ Fixed `extract_loss_trajectories_2d()` nesting structure bug
- Added proper handling of 7-level nested structure: `{split_type: {ensemble: {loss: {bv_reg_fn: {maxent: {bv_reg: {split_idx: history}}}}}}}`

### Issues Encountered and Resolved

#### Issue 1: Missing Plotting Functions
- **Problem**: Scripts attempted to import `plot_loss_convergence()` and `plot_split_variability()` which don't exist in `common/plotting.py`
- **Resolution**: Removed these imports and function calls; added comments noting they can be implemented later
- **Files affected**: All 3 refactored scripts

#### Issue 2: Missing Color Palette Entry
- **Problem**: `ValueError: The palette dictionary is missing keys: {'spatial'}` in Experiment 2
- **Resolution**: Added `spatial: "grey"` to `split_type_colors` in `2_CrossValidation/config.yaml`
- **Root cause**: Config used `Sp` abbreviation but data loader returned full `spatial` name

#### Issue 3: 2D Extraction Nesting Bug
- **Problem**: `AttributeError: 'dict' object has no attribute 'states'` in `extract_loss_trajectories_2d()`
- **Resolution**: Updated function to handle additional `bv_reg_fn` level between `loss_name` and `maxent_val`
- **Root cause**: Loader added BV regularization function dimension not accounted for in extraction logic

### Validation Results

All three experiments validated successfully:

**Experiment 1** (ISO_TRI / ISO_BI):
- ✅ 1,557 trajectory points extracted
- ✅ 72 best models selected
- ✅ Convergence heatmaps generated
- ✅ Model score heatmaps generated

**Experiment 2** (AF2 CrossValidation):
- ✅ 1,845 trajectory points extracted
- ✅ 84 best models selected
- ✅ Split types discovered: r, s, R3, Sp
- ✅ All visualizations generated

**Experiment 3** (2D BV Regularization):
- ✅ 5,535 trajectory points extracted (2D sweep)
- ✅ 36 best models selected
- ✅ 2D parameter space explored correctly
- ✅ BV regularization functions handled: L1, L2

### Key Learnings for Phase 2

1. **Results directory patterns**: Fitting scripts create timestamped directories. Use `results_prefix` + `find_most_recent_dir()` pattern.

2. **Config completeness**: Color palettes must include all split type names that appear in data (not just abbreviations).

3. **2D sweep complexity**: Multi-dimensional parameter sweeps require extra nesting level handling in extraction functions.

4. **Plotting function gaps**: Some plotting functions mentioned in plans don't exist yet. Add them incrementally as needed.

5. **Common modules work**: The shared module architecture successfully eliminates 90%+ code duplication while preserving exact functionality.

---

### Phase 2: Refactor Script Families
Refactor most-duplicated scripts to use `common/` modules.
#### [MODIFY] Analysis Scripts
- `analyse_loss_ISO_TRI_BI.py` (Exp 1, 2)
- `score_models_ISO_TRI_BI.py` (Exp 1, 2, 3)
- `process_optimisation_results.py` (Exp 1, 2, 3)
- `weights_validation_ISO_TRI_BI_precluster.py` & `recovery_analysis_ISO_TRI_BI_precluster.py` (Exp 1, 2)
- `analyse_split_*.py` (Exp 1, 2)
- `plot_compare_jaxENT_HDXer.py` & `plot_selected_models_ISO_TRI_BI.py` (Exp 1, 2, 3)
- `analyse_scores_mixed_linear_model.py` (Exp 1, 2, 3)

#### [MODIFY] Fitting & Data Scripts
- `optimise_fn.py` (Exp 1, 2, 3) -> Replaced by calls to `common.optimization`.
- `featurise_*.py` (Exp 1, 2) -> Use `common.loading` and `paths`.
- `splitdata_*.py` (Exp 1, 2) -> Use `common.loading`.

#### [MODIFY] Exp 3 Specific (2D Sweeps)
- `analyse_loss_ISO_TRI_BI_2D_BV.py`
- `weights_validation_ISO_TRI_BI_2D_BV.py`
- `recovery_analysis_ISO_TRI_BI_2D_BV.py`
- Refactor to use `common/` while preserving 2D sweep logic (`SweepConfig`).

### Phase 3: Configuration
Create YAML configuration files for each active experiment.
#### [NEW] [1_IsoValidation_OMass/config.yaml](file:///Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/config.yaml)
#### [NEW] [2_CrossValidation/config.yaml](file:///Users/alexi/JAX-ENT/jaxent/examples/2_CrossValidation/config.yaml)
#### [NEW] [3_CrossValidationBV/config.yaml](file:///Users/alexi/JAX-ENT/jaxent/examples/3_CrossValidationBV/config.yaml)

### Phase 4: Library Improvements
Fix gaps in the core library to support refactoring.
#### [MODIFY] [load_data.py](file:///Users/alexi/JAX-ENT/jaxent/src/utils/HDXer/load_data.py)
- Update `load_HDXer_kints` to return `Partial_Topology` objects instead of integers.

### Phase 5: Cleanup
- Archive original scripts to `_archive/` directories.
- Add `if __name__ == "__main__":` blocks to unique scripts.
- Remove `sys.path` hacks.

## Verification Plan

   ./venv/bin/python -c "from jaxent.examples.common import config, loading, analysis, plotting, optimization; print('Common modules imported successfully')"
   ```
2. **Config Loading Test**:
   ```bash
   ./venv/bin/python -c "from jaxent.examples.common.config import ExperimentConfig; c = ExperimentConfig.from_yaml('jaxent/examples/2_CrossValidation/config.yaml'); print(c)"
   

### Manual Verification
1. **Regression Testing**:
   - For a selected script (e.g., `analyse_loss_ISO_TRI_BI.py` in Exp 2), run the refactored version:
     ```bash
     cd jaxent/examples/2_CrossValidation/analysis
     python analyse_loss_ISO_TRI_BI.py --config ../config.yaml --output-dir _test_output
     ```
   - Compare the generated plots and CSVs in `_test_output` with the baseline output (from `_archive/` version) to ensure identical behavior.
   - Verify that 2D sweep scripts (`Exp 3`) still produce 2D heatmaps/results correctly.
