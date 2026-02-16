# Plan: Refactor Example Scripts for Clarity and Reusability

## Context

The `jaxent/examples/` directory contains ~93 Python scripts (excluding vendored `_Bradshaw/` code) across 7 experiment directories. Analysis scripts are near-identical copies across experiments, differing only in configuration (ensemble names, color schemes, scoring formulas, regex patterns). This makes maintenance painful: bug fixes applied to one copy don't propagate, and adding a new experiment means duplicating dozens of files. The scripts also underutilize existing library utilities (e.g. `load_HDXer_kints()`, `frame_average_features()`, analysis plot functions).

> **Scope note**: `4_CrossValidation_FunctionalMaxENT/` is excluded from this refactor plan for now. `combined_fixed_effects/` and `predict_traj/` contain standalone scripts without cross-experiment duplication and require documentation only.

**Goal**: Extract shared logic into a reusable `common/` module so each experiment script becomes a thin config-driven wrapper, while preserving the existing directory structure.

**Key decisions**:
- Old scripts will be moved to `_archive/` subdirectories during transition (not deleted)
- Divergent scoring formulas will be preserved via config-driven `ScoringConfig` weights (no behavior change)
- Hardcoded default paths will live in per-experiment `config.py` files, overridable by CLI args

---

## Phase 1: Create `jaxent/examples/common/` shared module

Create a new package at `jaxent/examples/common/` with the following modules:

### 1a. `common/__init__.py`
Re-export public API.

### 1b. `common/config.py` — Experiment configuration dataclasses (YAML-driven)

Extract the repeated configuration into declarative dataclasses with YAML serialization:

```python
import yaml
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs (used by fitting scripts)."""
    n_steps: int = 10000
    tolerance: float = 1e-10
    convergence_rates: list[float] = field(default_factory=lambda: [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
    learning_rate: float = 1e-3
    initial_learning_rate: float = 1e0
    initial_steps: int = 2
    optimizer: str = "adamw"
    ema_alpha: float = 0.5
    forward_model_scaling: float = 100.0
    clip_value: float | None = None

@dataclass
class LossConfig:
    """Configuration for loss function composition."""
    primary_loss: str  # e.g. "hdx_uptake_mean_centred_MSE_loss"
    regularization_losses: list[dict] = field(default_factory=list)  # [{"name": "...", "weight": 1.0}]
    optimize_bv_params: bool = False
    maxent_scaling: float = 1.0

@dataclass
class ExperimentConfig:
    """Top-level config for an experiment (analysis + fitting)."""
    ensembles: list[str]
    loss_functions: list[str]
    num_splits: int
    convergence_rates: list[float]
    # Default paths relative to the experiment directory
    results_dir: str          # e.g. "../fitting/jaxENT/_optimise"
    features_dir: str         # e.g. "../fitting/jaxENT/_featurise"
    datasplit_dir: str        # e.g. "../fitting/jaxENT/_datasplits"
    clustering_dir: str | None = None
    output_dir: str | None = None
    # Optimization settings (for fitting scripts)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss_configs: dict[str, LossConfig] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        # Recursively construct nested dataclasses
        if "optimization" in data:
            data["optimization"] = OptimizationConfig(**data["optimization"])
        if "loss_configs" in data:
            data["loss_configs"] = {k: LossConfig(**v) for k, v in data["loss_configs"].items()}
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import dataclasses
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj
        with open(path, "w") as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)

@dataclass
class PlotStyle:
    """Visual config for a specific experiment."""
    ensemble_colors: dict[str, str]
    loss_markers: dict[str, str]
    split_type_colors: dict[str, str] | None = None
    dpi: int = 300
    figsize_wide: tuple = (16, 6)
    figsize_square: tuple = (10, 10)

@dataclass
class ScoringConfig:
    """Config for model scoring / recovery analysis."""
    ensemble_feature_map: dict[str, str]
    ensemble_clustering_map: dict[str, str] | None = None
    state_mapping: dict[int, str] | None = None
    scoring_weights: dict[str, float]  # e.g. {"train_penalty": 1.0, "kl_penalty": 0.1}
```

Each experiment directory will have a `config.yaml` file (see Phase 3).

### 1c. `common/loading.py` — Shared data loading functions

Consolidate the duplicated loading patterns:

- `load_all_optimization_results(results_dir, config, grid_params=None) -> dict` — the HDF5 result loader currently copy-pasted in every `analyse_loss_*.py` and `process_optimisation_results.py`. When `grid_params` is provided (e.g. `["maxent_scaling", "bv_reg"]`), loads 2D sweep results keyed by both parameters (used by §2l 2D BV sweep scripts)
- `load_all_optimization_results_with_maxent(results_dir, config, ema=False) -> dict` — the extended loader with maxent regex parsing
- `load_clustering_results(clustering_dir, ensemble_clustering_map) -> dict` — cluster CSV loader
- `load_features_and_topology(features_dir, ensemble, feature_map) -> tuple[BV_input_features, list]` — BV feature + topology loader
- `load_experimental_data(datasplit_dir, split_type, split_idx) -> tuple` — train/val/test data loader
- `resolve_paths(args, script_dir) -> dict` — the repeated path-resolution logic

These all exist as near-identical copies across scripts today. The key change: they accept `ExperimentConfig` instead of inline parameters.

### 1d. `common/analysis.py` — Shared analysis functions

Consolidate the core analysis computations:

- `extract_loss_trajectories(results, split_type) -> pd.DataFrame` — currently duplicated in every `analyse_loss_*.py`
- `compute_model_scores(df, scoring_config) -> pd.DataFrame` — the `-log10(val_loss)` scoring with configurable penalty weights, replacing the divergent copies in `score_models_*.py`
- `kl_divergence(p, q, eps=1e-10) -> float` — currently duplicated in `process_optimisation_results.py`
- `calculate_cluster_ratios(assignments, weights) -> dict` — currently duplicated
- `calculate_recovery_jsd(pred, prior, exp, sparse_map) -> float` — currently duplicated across `recovery_analysis_*.py`
- `calculate_mse(pred, exp) -> float` — currently duplicated across `score_models_*.py`

### 1e. `common/plotting.py` — Shared plotting functions

Consolidate the repeated visualization code:

- `plot_loss_convergence(df, convergence_rates, output_dir, style, split_type=None)` — currently ~100 lines duplicated in every `analyse_loss_*.py`
- `plot_split_variability(df, convergence_rates, output_dir, style, split_type=None)` — same pattern
- `plot_final_performance_comparison(df, output_dir, style, split_type=None)` — boxplot comparisons
- `plot_cross_split_type_comparison(df, convergence_rates, output_dir, style)` — cross-split analysis
- `plot_split_distribution_heatmap(train_indices, val_indices, ..., output_dir, style)` — from `analyse_split_*.py`
- `plot_intrinsic_rates_comparison(hdxer_rates, jaxent_rates, ..., output_dir, style)` — from `plot_intrinsic_rates_*.py`
- `setup_publication_style()` — the repeated `sns.set_style("ticks"); sns.set_context("paper", ...)` block

All plotting functions accept a `PlotStyle` dataclass instead of hardcoded colors/markers.

### 1f. `common/cli.py` — Shared argparse patterns

- `add_common_args(parser)` — adds `--results-dir`, `--output-dir`, `--features-dir`, `--datasplit-dir`, `--ema`, `--absolute-paths`, `--config` (YAML config path)
- `resolve_script_paths(args, script_dir) -> dict` — the repeated relative/absolute path resolution

### 1g. `common/optimization.py` — Shared optimization/fitting functions

Consolidate the duplicated fitting script logic (currently 876+ lines in `optimise_fn.py`):

```python
from typing import Sequence, Tuple, cast, List
from jaxent.src.opt.base import InitialisedSimulation, JaxEnt_Loss
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationState
from jaxent.src.data.loader import ExpD_Dataloader

def create_data_loaders(
    hdx_data: List[HDX_peptide],
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    features: BV_input_features,
    feature_top: list[Partial_Topology],
    cov_matrix: Array | None = None,
) -> ExpD_Dataloader:
    """Create data loaders for training and validation datasets."""
    ...

def optimise_sweep(
    simulation: InitialisedSimulation,
    data_to_fit: Sequence[ExpD_Dataloader | Model_Parameters | Output_Features],
    config: OptimizationConfig,
    loss_functions: Sequence[JaxEnt_Loss],
    optimizer: OptaxOptimizer,
    opt_state: OptimizationState,
) -> Tuple[InitialisedSimulation, OptaxOptimizer]:
    """EMA-based optimization with relative convergence thresholds."""
    ...

def run_optimization(
    config: ExperimentConfig,
    loss_config: LossConfig,
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    features: BV_input_features,
    feature_top: List[Partial_Topology],
    output_dir: str,
    name: str,
) -> None:
    """Single entry point replacing all run_optimise_* variants."""
    ...

def get_loss_function_by_name(name: str) -> JaxEnt_Loss:
    """Lookup loss function from string name (for YAML config)."""
    LOSS_REGISTRY = {
        "hdx_uptake_mean_centred_MSE_loss": hdx_uptake_mean_centred_MSE_loss,
        "hdx_uptake_MSE_loss": hdx_uptake_MSE_loss,
        "hdx_uptake_MAE_loss_vectorized": hdx_uptake_MAE_loss_vectorized,
        "maxent_convexKL_loss": maxent_convexKL_loss,
        ...
    }
    return LOSS_REGISTRY[name]
```

This consolidates the 5+ near-identical `run_optimise_ISO_TRI_BI_*` variants in `optimise_fn.py` into a single configurable function.

### 1h. `common/paths.py` — Centralized path utilities

```python
from pathlib import Path

def resolve_example_path(example_name: str, subdir: str = "") -> Path:
    """Resolve path relative to an example directory."""
    ...

def get_data_dir(example_name: str) -> Path:
    """Get the data directory for an example."""
    ...

def ensure_output_dir(path: Path) -> Path:
    """Create output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
```

Replaces the fragile `os.path.join(os.path.dirname(__file__), "../../data/...")` patterns.

## Phase 2: Refactor the most-duplicated script families

Work through each script family, replacing inline logic with calls to `common/`. Priority order based on duplication count:

### 2a. `analyse_loss_ISO_TRI_BI.py` (exists in 5 experiments)

**Before** (~770 lines each, nearly identical):
- Inline `load_all_optimization_results()`, `extract_loss_trajectories()`, all plot functions, hardcoded config

**After** (~40 lines each):
```python
from jaxent.examples.common import ExperimentConfig, PlotStyle, loading, analysis, plotting

CONFIG = ExperimentConfig(ensembles=["ISO_TRI", "ISO_BI"], ...)
STYLE = PlotStyle(ensemble_colors={"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}, ...)

def main():
    results = loading.load_all_optimization_results(results_dir, CONFIG)
    df = analysis.extract_loss_trajectories(results)
    plotting.plot_loss_convergence(df, CONFIG.convergence_rates, output_dir, STYLE)
    ...
```

### 2b. `analyse_loss_MoPrP.py` (exists in 3 experiments)
Same pattern as 2a but with MoPrP-specific config (different ensembles, colors).

### 2c. `score_models_ISO_TRI_BI.py` (exists in 3 experiments)
Replace duplicated scoring logic with `common.analysis.compute_model_scores()`. The divergent scoring formulas between experiment 2 and 3 become different `ScoringConfig` instances.

### 2d. `process_optimisation_results.py` (exists in 3 experiments)
Replace duplicated loading, KL divergence, cluster ratio code with `common/` calls. Keep the custom `BV_uptake_ForwardPass_frames` class in `common/analysis.py` since it's used across experiments.

### 2e. `weights_validation_ISO_TRI_BI_precluster.py` (exists in 4 experiments)
The diff shows these are nearly identical (1-line difference). Thin wrapper over `common/`.

### 2f. `recovery_analysis_ISO_TRI_BI_precluster.py` (exists in 4 experiments)
Same treatment.

### 2g. `analyse_split_*.py` (exists in 5 experiments)
Extract heatmap generation into `common/plotting.py`.

### 2h. `plot_intrinsic_rates_simple.py` (exists in 2 experiments)
Replace manual HDXer loading with library's `load_HDXer_kints()` from `jaxent/src/utils/HDXer/load_data.py`.

### 2i. Fitting scripts: `optimise_fn.py` and variants (exists in 3 experiments)

**Before** (`optimise_fn.py` in `2_CrossValidation/fitting/jaxENT/`, 876 lines):
- 5+ near-identical functions: `run_optimise_ISO_TRI_BI_maxENT`, `run_optimise_ISO_TRI_BI_maxENT_BV`, `run_optimise_ISO_TRI_BI_maxENT_MAE`, etc.
- Duplicated `create_data_loaders()`, `optimise_sweep()` functions
- Only differences: loss function combinations, parameter masks, learning rate schedules

**After** (~50 lines per experiment, config-driven):
```python
from jaxent.examples.common import ExperimentConfig, optimization

config = ExperimentConfig.from_yaml("config.yaml")
loss_config = config.loss_configs["maxENT"]

optimization.run_optimization(
    config=config,
    loss_config=loss_config,
    train_data=train_data,
    val_data=val_data,
    features=features,
    feature_top=feature_top,
    output_dir="_optimise",
    name="CrossVal_maxENT_split001",
)
```

### 2j. Fitting scripts: `featurise_*.py` (exists in 3 experiments)
- Import `load_HDXer_kints()` from `common.loading`
- Use `paths.resolve_example_path()` instead of hardcoded relative paths
- Reduce from ~229 lines to ~100 lines

### 2k. Data prep scripts: `splitdata_*.py` (exists in 3 experiments)
- Import `save_split_data()` from `common.loading`
- Use centralized path management

### 2l. 2D BV sweep scripts (exists in 2 experiments: 3, ignoring 4)

**Scripts**: `analyse_loss_ISO_TRI_BI_2D_BV.py`, `recovery_analysis_ISO_TRI_BI_2D_BV.py`, `weights_validation_ISO_TRI_BI_2D_BV.py`

These are 2D-parameter-sweep variants of the 1D families in §2a/2e/2f. They iterate over a grid of `(maxent_scaling, bv_reg)` values and load results keyed by both parameters.

**Refactoring approach**: Merge into the 1D scripts with a `--2d-sweep` / `--grid-params` CLI flag. Extend `common/loading.load_all_optimization_results()` to accept an optional `grid_params` argument for 2D result structures. This keeps a single codebase for both 1D convergence-rate sweeps and 2D parameter sweeps.

### 2m. PUF analysis scripts (exists in 3 experiments: 2, 3, ignoring 4)

**Scripts**:
- `analyse_LocalFeatures_PUFs.py` (~2552 lines) — local structural feature analysis
- `analyse_cluster_GlobalFeatures_PUFs.py` (~1232 lines) — global feature PCA + clustering
- `cluster_LocalFeatures_PUF.py` (~2018 lines) — PCA + k-means/rules-based clustering
- `compute_recovery%_PUF.py` — PUF recovery percentage calculation

These are large, CLI-driven scripts with proper `if __name__ == "__main__"` guards and argparse. Main duplication is in hardcoded colour maps, structural-region JSON paths, and path defaults.

**Refactoring approach**: These scripts are too large/specialized to collapse into `common/`. Instead:
1. Diff cross-experiment copies to verify they are identical (or identify divergences)
2. If identical: keep one canonical copy in `common/puf_analysis/` and symlink or thin-wrapper from each experiment
3. If divergent: extract differing config (colours, region JSON paths) into per-experiment `config.yaml`
4. Add `--config` flag to each script's argparse to load defaults from YAML

### 2n. Comparison and visualization scripts (exists in 2–3 experiments)

**Scripts**:
- `plot_compare_jaxENT_HDXer.py` (×3, ~704 lines) — bar plots comparing HDXer vs jaxENT results
- `plot_selected_models_ISO_TRI_BI.py` (×2) — model selection visualization

**Refactoring approach**:
1. Diff cross-experiment copies; if identical keep one canonical copy
2. Extract ~60 lines of hardcoded dicts (`EXPERIMENT_COLOURS`, `SPLIT_NAME_MAPPING`, `SPLIT_TYPE_NORMALIZATION`) into `config.yaml`
3. Move shared plot helpers (hatching, annotation) into `common/plotting.py`

### 2o. Statistical analysis scripts (exists in 1–3 experiments)

**Scripts**:
- `analyse_scores_mixed_linear_model.py` (×3 copies)
- `calculate_state_ratios.py` (×3, ~214 lines) — thermodynamic equilibrium calculator
- `fixed_effects_linear_model_ISO_TRI_BI.py` (×1, **empty file — delete**)
- `cross_experiment_mixed_effects_model.py` (×1, experiment 3 only — unique, leave as-is)

**Refactoring approach**:
1. Delete the empty `fixed_effects_linear_model_ISO_TRI_BI.py`
2. Diff `calculate_state_ratios.py` copies — if identical, keep one; if not, parameterize (ΔG values, temperature) via CLI args
3. Diff `analyse_scores_mixed_linear_model.py` copies — extract experiment-specific formula strings and colour maps into config

### 2p. Remaining unique/utility scripts (no cross-experiment duplication)

The following scripts exist only in one experiment and don't need cross-experiment dedup. They only need `__main__` guards and `sys.path` cleanup (covered by §5c):

- `1_IsoValidation_OMass/analysis/combined_cluster_analysis/Failure_analysis_ISO_TRI_BI_maxent_CV.py`
- `1_IsoValidation_OMass/analysis/combined_cluster_analysis/Failure_analysis_ISO_TRI_BI_maxent_weights.py`
- `1_IsoValidation_OMass/analysis/combined_cluster_analysis/ratio_recovery_ISO_TRI_BI_maxent.py`
- `1_IsoValidation_OMass/data/jaxENT_prepare_TeaA_data.py`
- `1_IsoValidation_OMass/fitting/jaxENT/compute_covariance_matrices.py`
- `1_IsoValidation_OMass/fitting/jaxENT/compute_cprior.py`
- `1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI_SCALING.py` (consider merging with `featurise_ISO_TRI_BI.py` via a `--scaling` flag)
- `combined_fixed_effects/plot_kendalls_combined.py`
- `predict_traj/predict_traj.py`

---

## Phase 3: Add per-experiment YAML config files

Create a `config.yaml` at each experiment's root directory. Scripts load config via `ExperimentConfig.from_yaml()`.

**Files to create:**
- `1_IsoValidation_OMass/config.yaml`
- `2_CrossValidation/config.yaml`
- `3_CrossValidationBV/config.yaml`

Example `2_CrossValidation/config.yaml`:

```yaml
# Experiment configuration
ensembles:
  - AF2_MSAss
  - AF2_filtered
loss_functions:
  - mcMSE
  - MSE
  - Sigma_MSE
num_splits: 3
convergence_rates: [1.0, 0.1, 0.01, 0.001, 0.0001, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8]

# Paths (relative to experiment directory)
results_dir: "./fitting/jaxENT/_optimise"
features_dir: "./fitting/jaxENT/_featurise"
dasplit_dir: "./fitting/jaxENT/_datasplits"
output_dir: "./analysis/_output"

# Optimization settings (for fitting scripts)
optimization:
  n_steps: 10000
  tolerance: 1.0e-10
  learning_rate: 1.0e-3
  initial_learning_rate: 1.0
  initial_steps: 2
  optimizer: adamw
  ema_alpha: 0.5
  forward_model_scaling: 100.0

# Loss function configurations
loss_configs:
  maxENT:
    primary_loss: hdx_uptake_mean_centred_MSE_loss
    regularization_losses:
      - name: maxent_convexKL_loss
        weight: 1.0
    maxent_scaling: 1.0
  maxENT_BV:
    primary_loss: hdx_uptake_mean_centred_MSE_loss
    regularization_losses:
      - name: maxent_convexKL_loss
        weight: 1.0
    optimize_bv_params: true
    maxent_scaling: 1.0

# Plot styling
style:
  ensemble_colors:
    AF2_MSAss: "RoyalBlue"
    AF2_filtered: "Cyan"
  loss_markers:
    mcMSE: "o"
    MSE: "s"
    Sigma_MSE: "^"

# Scoring configuration
scoring:
  ensemble_feature_map:
    AF2_MSAss: AF2_MSAss
    AF2_filtered: AF2_filtered
  ensemble_clustering_map:
    AF2_MSAss: AF2_MSAss
    AF2_filtered: AF2_Filtered
  state_mapping:
    0: Folded
    1: PUF1
    2: PUF2
  scoring_weights:
    train_penalty: 0.9
    kl_penalty: 0.0
```

---

## Phase 4: Replace library function duplication

In the refactored `common/` modules, use existing library functions instead of reimplementing:

| Duplicated Code | Library Replacement |
|---|---|
| Manual HDXer `.dat` file parsing in `plot_intrinsic_rates_*.py` | `jaxent.src.utils.HDXer.load_data.load_HDXer_kints()` |
| Manual HDF5 loading (already partially used) | `jaxent.src.utils.hdf.load_optimization_history_from_file()` (keep using) |
| Manual frame averaging in `process_optimisation_results.py` | `jaxent.src.utils.jax_fn.frame_average_features()` (already used in some copies) |
| `sys.path.insert(0, base_dir)` hack at top of every script | Proper package import via `jaxent.examples.common` (since jaxent is installed) |

---

## Phase 5: Archive old scripts and clean up

### 5a. Archive originals
For each experiment directory, create an `analysis/_archive/` subdirectory and move the pre-refactor versions of all modified scripts there. This preserves easy side-by-side comparison during the transition period. Example:
```
2_CrossValidation/analysis/_archive/analyse_loss_ISO_TRI_BI.py  (original)
2_CrossValidation/analysis/analyse_loss_ISO_TRI_BI.py           (refactored)
```

### 5b. Delete stale copies
Remove obviously stale files (files with "copy" or "OLD" suffixes) that are not the canonical versions:
- `1_IsoValidation_OMass/analysis/score_models_ISO_TRI_BI copy.py`
- `2_CrossValidation/analysis/cluster_LocalFeatures_PUF_OLD.py`
- `3_CrossValidationBV/analysis/cluster_LocalFeatures_PUF_OLD.py`
- `4_CrossValidation_FunctionalMaxENT/analysis/cluster_LocalFeatures_PUF_OLD.py`

### 5c. Add `__main__` guards
For any remaining scripts not covered in Phase 2 (data prep scripts, one-off analyses), ensure they:
1. Wrap top-level code in `def main()` + `if __name__ == "__main__":` guard
2. Remove the `sys.path.insert` hack (rely on package installation)

---

## Files to create
- `jaxent/examples/common/__init__.py`
- `jaxent/examples/common/config.py`
- `jaxent/examples/common/loading.py`
- `jaxent/examples/common/analysis.py`
- `jaxent/examples/common/plotting.py`
- `jaxent/examples/common/cli.py`
- `jaxent/examples/common/optimization.py`
- `jaxent/examples/common/paths.py`
- `jaxent/examples/1_IsoValidation/config.yaml`
- `jaxent/examples/1_IsoValidation_OMass/config.yaml`
- `jaxent/examples/2_CrossValidation/config.yaml`
- `jaxent/examples/3_CrossValidationBV/config.yaml`
- `jaxent/examples/4_CrossValidation_FunctionalMaxENT/config.yaml`
- `jaxent/examples/README.md`

## Files to modify (major refactors)

### Analysis scripts (§2a–2h)
- All `analyse_loss_ISO_TRI_BI.py` (5 files)
- All `analyse_loss_MoPrP.py` (3 files)
- All `score_models_ISO_TRI_BI.py` (3 files)
- All `process_optimisation_results.py` (3 files)
- All `weights_validation_*.py` (6 files)
- All `recovery_analysis_*.py` (5 files)
- All `analyse_split_*.py` (5 files)
- All `plot_intrinsic_rates_*.py` (3 files)

### 2D BV sweep scripts (§2l)
- All `analyse_loss_ISO_TRI_BI_2D_BV.py` (2 files — experiments 3, 4)
- All `recovery_analysis_ISO_TRI_BI_2D_BV.py` (2 files)
- All `weights_validation_ISO_TRI_BI_2D_BV.py` (2 files)

### PUF analysis scripts (§2m)
- All `analyse_LocalFeatures_PUFs.py` (3 files)
- All `analyse_cluster_GlobalFeatures_PUFs.py` (3 files)
- All `cluster_LocalFeatures_PUF.py` (3 files)
- All `compute_recovery%_PUF.py` (3 files)

### Comparison/visualization scripts (§2n)
- All `plot_compare_jaxENT_HDXer.py` (3 files)
- All `plot_selected_models_ISO_TRI_BI.py` (2 files)

### Statistical analysis scripts (§2o)
- All `analyse_scores_mixed_linear_model.py` (3 files)
- All `calculate_state_ratios.py` (3 files)

### Fitting scripts (§2i–2k)
- All `optimise_fn.py` variants (3 files, ~876 lines → ~50 lines each)
- All `optimise_ISO_TRI_BI*.py` variants (7 files)
- All `featurise_*.py` (3 files, ~229 lines → ~100 lines each)
- All `splitdata_*.py` (3 files)

## Files to archive (move to `_archive/`)
- All original versions of the scripts listed in "Files to modify" above
- Include both `analysis/_archive/` and `fitting/_archive/` subdirectories

## Files to delete (stale/empty files)
- `1_IsoValidation_OMass/analysis/score_models_ISO_TRI_BI copy.py`
- `1_IsoValidation_OMass/analysis/fixed_effects_linear_model_ISO_TRI_BI.py` (empty file, 0 bytes)
- `2_CrossValidation/analysis/cluster_LocalFeatures_PUF_OLD.py`
- `3_CrossValidationBV/analysis/cluster_LocalFeatures_PUF_OLD.py`
- `4_CrossValidation_FunctionalMaxENT/analysis/cluster_LocalFeatures_PUF_OLD.py`

---

## Verification

After each phase:
1. Run `python -c "from jaxent.examples.common import config, loading, analysis, plotting, optimization"` to verify module imports
2. For each refactored script, run it from its original directory and verify output files match (diff PNGs and CSVs against pre-refactor baselines)
3. Run `python -m pytest` on any existing tests
4. Spot-check that the `sys.path` hack removal doesn't break imports by running scripts from different working directories
5. Test YAML config loading: `python -c "from jaxent.examples.common.config import ExperimentConfig; c = ExperimentConfig.from_yaml('config.yaml'); print(c)"`

### Unit tests for new modules
Create test files at `jaxent/tests/examples/`:
- `test_common_config.py` — test YAML loading/saving, dataclass validation
- `test_common_loading.py` — test data loading functions
- `test_common_optimization.py` — test optimization configuration


# Plan verification report

## Resolved findings

- ~~Scope mismatch~~: **Resolved** — plan now covers PUF scripts (§2m), 2D BV sweeps (§2l), comparison/viz (§2n),
  statistical analyses (§2o), and unique utilities (§2p). OMass fitting variants have been removed from codebase.
  `4_CrossValidation_FunctionalMaxENT/` is explicitly excluded from scope.
- ~~Special-case 2D sweeps~~: **Resolved** — `common/loading.load_all_optimization_results()` now specifies a
  `grid_params` argument for 2D result structures (§2l).

## Open findings

- **Packaging**: `jaxent/examples` isn't a package — no `__init__.py`, and `pyproject.toml` only packages `jaxent/src`.
  `import jaxent.examples.common` will fail unless packaging/sys.path is changed.
- **Loss function divergence**: Wrappers define covariance-weighted and shape-adjusted losses (e.g.,
  `hdx_uptake_mean_centred_MSE_loss` in `optimise_ISO_TRI_BI_splits_maxENT.py`) that differ from
  `jaxent/src/opt/losses.py`. A simple `LOSS_REGISTRY` mapping to library functions would change behaviour.
- **Featurisation topology gap**: Scripts build `TopologyFactory` entries from HDXer rates;
  `load_HDXer_kints()` returns only intrinsic rates. Reusing the library helper as-is drops topology data.
- **Config/CLI gaps**: YAML sketch uses `dasplit_dir` (typo for `datasplit_dir`) and omits fields routinely
  parsed in scripts (clustering dirs, state ratios JSON, covariance toggles, grid axes).
  `ExperimentConfig.from_yaml` should validate required fields.

## Next steps

1. Decide how to package examples (add `__init__.py` and adjust `pyproject.toml`, or keep scripts adding `sys.path`)
   before building `common/`.
2. Map all loss/optimizer variants across `optimise_fn.py` files and wrappers to a registry/config that preserves
   current covariance/partition behaviour.
3. Design `ExperimentConfig`/YAML to validate required paths/aux data (clustering, state ratios, grid axes) so
   failures surface early.