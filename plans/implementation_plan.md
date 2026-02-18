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

### Phase 1: Create `jaxent/examples/common/` Shared Module
Create the core shared infrastructure.
#### [NEW] [jaxent/examples/common/](file:///Users/alexi/JAX-ENT/jaxent/examples/common/)
- `config.py`: Configuration dataclasses (`ExperimentConfig`, `OptimizationConfig`, etc.) with YAML loading.
- `paths.py`: Centralized path resolution.
- `loading.py`: Consolidated data loading functions (optimization results, experimental data).
- `analysis.py`: Shared analysis logic (KL divergence, recovery metrics, etc.).
- `plotting.py`: Shared plotting functions with `PlotStyle` support.
- `optimization.py`: Unified `run_optimization` function replacing 5+ `optimise_fn.py` variants.
- `losses.py`: Example-specific loss functions moved from `optimise_fn.py`.
- `cli.py`: Shared `argparse` setup.

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
