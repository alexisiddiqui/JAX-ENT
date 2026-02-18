# Plan: Refactor Example Scripts for Clarity and Reusability

## Context

The `jaxent/examples/` directory contains ~93 ./venv/bin/python  scripts (excluding vendored `_Bradshaw/` code) across 7 experiment directories. Analysis scripts are near-identical copies across experiments, differing only in configuration (ensemble names, color schemes, scoring formulas, regex patterns). This makes maintenance painful: bug fixes applied to one copy don't propagate, and adding a new experiment means duplicating dozens of files. The scripts also underutilize existing library utilities (e.g. `load_HDXer_kints()`, `frame_average_features()`, analysis plot functions).

> **Scope note**: `4_CrossValidation_FunctionalMaxENT/` has been **fully moved to `deprecated/`** and is excluded from this refactor plan. Some scripts from experiment 3 also have deprecated copies in `deprecated/3_CrossValidation_BV/`. `combined_fixed_effects/` and `predict_traj/` contain standalone scripts without cross-experiment duplication and require documentation only.
>
> **Active experiments**: `1_IsoValidation_OMass`, `2_CrossValidation`, `3_CrossValidationBV`.
> Duplication counts below refer to **active copies only** unless otherwise noted.

**Goal**: Extract shared logic into a reusable `common/` module so each experiment script becomes a thin config-driven wrapper, while preserving the existing directory structure.

**Key decisions**:
- Old scripts will be moved to `_archive/` subdirectories during transition (not deleted)
- Divergent scoring formulas will be preserved via config-driven `ScoringConfig` weights (no behavior change)
- Hardcoded default paths will live in per-experiment `config.py` files, overridable by CLI args

---

## Phase 0: Packaging and Project Structure

> **Decision (2026-02-18)**: Option A approved — make `jaxent/examples` a proper package.

To allow `import jaxent.examples.common`, the examples directory must be treated as a package.

### 0a. Update `pyproject.toml`

The current build config uses `hatchling` as the backend, with `setuptools`-style `package-dir` and `packages.find` pointing at `jaxent/src`. To include `jaxent.examples`, update to explicit package discovery:

```toml
# Replace the current [tool.setuptools.*] sections with hatch-native config:
[tool.hatch.build.targets.wheel]
packages = ["jaxent"]
```

Alternatively, if keeping `setuptools`-style config:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["jaxent*"]
```

> [!NOTE]
> The current `pyproject.toml` has conflicting config: `[tool.setuptools.packages.find]` with `where = ["jaxent/src"]` and `[tool.setuptools.package-dir]` `"" = "jaxent/src"`, but uses `hatchling` as the build backend. These `setuptools` keys are ignored by `hatchling`. The fix should use `[tool.hatch.build]` config instead. Verify with `pip install -e .` after the change.

### 0b. Create `__init__.py` files
- `jaxent/examples/__init__.py` (required)
- `jaxent/examples/common/__init__.py` (required)
- `jaxent/examples/1_IsoValidation_OMass/__init__.py` (optional, for consistency)
- `jaxent/examples/2_CrossValidation/__init__.py` (optional)
- `jaxent/examples/3_CrossValidationBV/__init__.py` (optional)

---

## Phase 1: Create `jaxent/examples/common/` shared module

Create a new package at `jaxent/examples/common/` with the following modules:

### 1a. `common/__init__.py`
Re-export public API.

### 1b. `common/config.py` — Experiment configuration dataclasses (YAML-driven)

Extract the repeated configuration into declarative dataclasses with YAML serialization:

```./venv/bin/python 
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
    covariance_matrix_path: str | None = None  # e.g. "../../data/_MoPrP_covariance_matrices/Sigma.npz"

@dataclass
class SweepConfig:
    """Configuration for 2D parameter sweeps (used by BV objective scripts)."""
    maxent_values: list[float] = field(default_factory=list)   # e.g. [1, 2, 5, 10, 20, 50]
    bv_reg_values: list[float] = field(default_factory=list)   # e.g. [0.01, 0.1, 1.0, 10.0]

@dataclass
class LossConfig:
    """Configuration for loss function composition."""
    primary_loss: str  # e.g. "hdx_uptake_mean_centred_MSE_loss"
    regularization_losses: list[dict] = field(default_factory=list)  # [{"name": "...", "weight": 1.0}]
    optimize_bv_params: bool = False
    maxent_scaling: float = 1.0

_REQUIRED_FIELDS = {"ensembles", "results_dir", "features_dir", "datasplit_dir", "num_splits", "convergence_rates"}

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
    state_ratios_json: str | None = None  # e.g. "./analysis/state_ratios.json"
    # Optimization settings (for fitting scripts)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss_configs: dict[str, LossConfig] = field(default_factory=dict)
    sweep: SweepConfig = field(default_factory=SweepConfig)  # 2D parameter sweep axes
    # Plot and scoring config (optional, used by analysis scripts)
    style: PlotStyle | None = None
    scoring: ScoringConfig | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file with validation."""
        with open(path) as f:
            data = yaml.safe_load(f)
        # Validate required fields
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")
        # Warn on unknown keys (catches typos like 'dasplit_dir')
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = data.keys() - known
        if unknown:
            import warnings
            warnings.warn(f"Unknown config keys (typo?): {unknown}", stacklevel=2)
        # Filter out unknown keys (after warning) to prevent TypeError
        data = {k: v for k, v in data.items() if k in known}
        # Recursively construct nested dataclasses
        if "optimization" in data:
            data["optimization"] = OptimizationConfig(**data["optimization"])
        if "loss_configs" in data:
            data["loss_configs"] = {k: LossConfig(**v) for k, v in data["loss_configs"].items()}
        if "sweep" in data:
            data["sweep"] = SweepConfig(**data["sweep"])
        if "style" in data and data["style"] is not None:
            data["style"] = PlotStyle(**data["style"])
        if "scoring" in data and data["scoring"] is not None:
            data["scoring"] = ScoringConfig(**data["scoring"])
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

| Function | Active Copies | Source Scripts |
|----------|:---:|---|
| `load_all_optimization_results(results_dir, config)` | 5 | `analyse_loss_*.py` (×2), `process_optimisation_results.py` (×3) |
| `load_all_optimization_results_2d(results_dir, config)` | 3 | `analyse_loss_*_2D_BV.py`, `recovery_analysis_*_2D_BV.py`, `weights_validation_*_2D_BV.py` |
| `load_all_optimization_results_with_maxent(results_dir, config)` | 6 | `weights_validation_*.py` (×3), `recovery_analysis_*.py` (×3) |
| `load_clustering_results(clustering_dir, map)` | 6 | 2 signature variants: dict-return (Exp 1) vs DataFrame-return (Exp 2/3) |
| `load_features_and_topology(features_dir, ensemble)` | 3 | `score_models_*.py` |
| `load_experimental_data(datasplit_dir, split_type, idx)` | 3 | `score_models_*.py` |
| `load_HDXer_kints(kint_path)` | 2 | `featurise_*.py` — identical |
| `extract_maxent_value_from_filename(filename)` | 5+ | regex helper used across many scripts |

These all exist as near-identical copies across scripts today. The key change: they accept `ExperimentConfig` instead of inline parameters.

When `grid_params` is provided (e.g. `["maxent_scaling", "bv_reg"]`), `load_all_optimization_results` loads 2D sweep results keyed by both parameters (used by §2l 2D BV sweep scripts).

### 1d. `common/analysis.py` — Shared analysis functions

Consolidate the core analysis computations:

| Function | Active Copies | Notes |
|----------|:---:|---|
| `kl_divergence(p, q, eps)` | **11** | Identical everywhere — most-duplicated function (11 active + 13 deprecated = 24 total) |
| `effective_sample_size(weights)` | **6** | `1/sum(w²)` — identical everywhere |
| `extract_loss_trajectories(results, ...)` | 3 (1D) + 1 (2D) | 2D variant adds `bv_reg_function/value` columns |
| `compute_model_scores(df, scoring_config)` | 3 | Scoring weights differ per experiment → accept `ScoringConfig` |
| `calculate_cluster_ratios(assignments, weights)` | 6 | Identical |
| `calculate_recovery_percentage(observed, ground_truth)` | 3+ | Simple ratio comparison |
| `calculate_recovery_JSD(assignments, weights, target, state_map)` | 3 | JSD-based; Exp 1 uses simpler signature |
| `calculate_dMSE(pred, exp)` | 3 | In `score_models_*.py` — identical |
| `calculate_work_metrics(pred, exp)` | 3 | In `score_models_*.py` — identical |
| `calculate_mse(pred, exp)` | 3 | In `score_models_*.py` — identical |
| `BV_uptake_ForwardPass_frames` class | 3 | In `process_optimisation_results.py` — identical |

### 1e. `common/plotting.py` — Shared plotting functions

Consolidate the repeated visualization code:

| Function | Active Copies | Notes |
|----------|:---:|---|
| `setup_publication_style()` | **~12** (inline block) | `sns.set_style("ticks"); sns.set_context(...)` |
| `plot_loss_convergence(...)` | 3 | ~100 lines each |
| `plot_score_heatmap(...)` | 3 | Heatmap of scores across maxent × convergence |
| `plot_recovery_heatmap(...)` | 2 (1D) + 1 (2D grid) | Recovery % heatmaps |
| `plot_ess_heatmaps(...)` | 2 | ESS across convergence × maxent |
| `plot_weight_distribution_lines(...)` | 2 | Weight distribution line plots |
| `plot_volcano_kl_recovery(...)` | 2 | Volcano plots |
| `create_violin_plots(...)` | 3 | In `score_models_*.py` |
| `plot_split_distribution_heatmap(...)` | 2 | From `analyse_split_*.py` |
| `plot_split_variability(...)` | 2 | Same pattern as convergence |
| `plot_final_performance_comparison(...)` | 2 | Boxplot comparisons |

All plotting functions accept a `PlotStyle` dataclass instead of hardcoded colors/markers.

### 1f. `common/cli.py` — Shared argparse patterns

- `add_common_args(parser)` — adds `--results-dir`, `--output-dir`, `--features-dir`, `--datasplit-dir`, `--clustering-dir`, `--state-ratios-json`, `--covariance-matrix-path`, `--ema`, `--absolute-paths`, `--config` (YAML config path)
- `resolve_script_paths(args, script_dir) -> dict` — the repeated relative/absolute path resolution

### 1g. `common/optimization.py` — Shared optimization/fitting functions

Consolidate the duplicated fitting script logic (currently 876+ lines in `optimise_fn.py`):

| Function | Active Copies | Notes |
|----------|:---:|---|
| `create_data_loaders(...)` | 3 identical | Sparse map creation + dataloader setup |
| `optimise_sweep(...)` | 3 identical (~250 lines each) | EMA-based convergence with thresholds |
| `run_optimise_ISO_TRI_BI_maxENT(...)` | 3 | Differs only in loss combos + param masks |
| `run_optimise_ISO_TRI_BI_maxENT_BV(...)` | 2 (Exp 2, 3) | Same but optimizes BV params |
| `run_optimise_ISO_TRI_BI_maxENT_MAE(...)` | 3 | MAE loss variant |
| `run_optimise_ISO_TRI_BI_maxENT_MAE_Sigma(...)` | 3 | Sigma-weighted MAE variant |
| `run_optimise_ISO_TRI_BI_maxENT_MAE_Sigma_Sigma(...)` | 2 (Exp 1, 2) | Double-sigma variant |
| `run_optimise_ISO_TRI_BI_maxENT_BV_objective(...)` | 1 (Exp 3 only) | BV objective variant |

> [!NOTE]
> Exp 3 adds a `model_parameters_lr_scale: float = 1.0` parameter to all `run_optimise_*` variants that is absent in Exp 1. The unified `run_optimization` must support this.

All `run_optimise_*` variants collapse into a single config-driven function:

```./venv/bin/python 
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

```./venv/bin/python 
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


### 1i. `common/losses.py` — Shared example-specific loss functions

Move the example-specific loss functions here to keep the core library clean:
- `hdx_uptake_mean_centred_MSE_loss`
- `hdx_uptake_MSE_loss`
- `hdx_uptake_MAE_loss_vectorized`

These will be registered in `LOSS_REGISTRY` in `common/optimization.py` but defined here.

### Build order

Modules should be built in dependency order:

1. `paths.py` — no internal deps
2. `config.py` — no internal deps
3. `analysis.py` — pure computation, no internal deps
4. `loading.py` — depends on `config`, `paths`
5. `losses.py` — depends on JAX-ENT core only
6. `plotting.py` — depends on `config` (for `PlotStyle`), `analysis`
7. `optimization.py` — depends on `config`, `losses`, `loading`
8. `cli.py` — depends on `config`, `paths`
9. `__init__.py` — re-exports all above

### Pilot refactoring recommendation

Refactor `analyse_loss_*.py` (2 × 1D + 1 × 2D) first as the pilot because it exercises `loading.py`, `analysis.py`, and `plotting.py` — validating the approach before tackling fitting scripts.

---

## Phase 2: Refactor the most-duplicated script families

Work through each script family, replacing inline logic with calls to `common/`. Priority order based on duplication count:

### 2a. `analyse_loss_ISO_TRI_BI.py` (2 active 1D copies + 1 active 2D variant)

> **Inventory correction**: Plan originally said "5 experiments" — 2 of those are deprecated. Exp 3 has only the 2D BV variant (`analyse_loss_ISO_TRI_BI_2D_BV.py`), not a 1D copy.

**Before** (~770 lines each, nearly identical):
- Inline `load_all_optimization_results()`, `extract_loss_trajectories()`, all plot functions, hardcoded config

**After** (~40 lines each):
```./venv/bin/python 
from jaxent.examples.common import ExperimentConfig, PlotStyle, loading, analysis, plotting

CONFIG = ExperimentConfig(ensembles=["ISO_TRI", "ISO_BI"], ...)
STYLE = PlotStyle(ensemble_colors={"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}, ...)

def main():
    results = loading.load_all_optimization_results(results_dir, CONFIG)
    df = analysis.extract_loss_trajectories(results)
    plotting.plot_loss_convergence(df, CONFIG.convergence_rates, output_dir, STYLE)
    ...
```

### 2b. ~~`analyse_loss_MoPrP.py`~~ (REMOVED — fully deprecated)

> **Inventory correction**: No active copies exist. Both copies are in `deprecated/`. This item is removed from the refactor scope.

### 2c. `score_models_ISO_TRI_BI.py` (3 active copies — count confirmed ✓)
Replace duplicated scoring logic with `common.analysis.compute_model_scores()`. The divergent scoring formulas between experiment 2 and 3 become different `ScoringConfig` instances.

### 2d. `process_optimisation_results.py` (3 active copies — count confirmed ✓)
Replace duplicated loading, KL divergence, cluster ratio code with `common/` calls. Keep the custom `BV_uptake_ForwardPass_frames` class in `common/analysis.py` since it's used across experiments.

### 2e. `weights_validation_ISO_TRI_BI_precluster.py` (2 active 1D + 1 active 2D variant)

> **Inventory correction**: Plan originally said "4 experiments" — 2 are deprecated. Exp 3 has only the 2D BV variant.

The 1D copies are nearly identical (1-line difference). Thin wrapper over `common/`.

### 2f. `recovery_analysis_ISO_TRI_BI_precluster.py` (2 active 1D + 1 active 2D variant)

> **Inventory correction**: Plan originally said "4 experiments" — 2 are deprecated. Exp 3 has only the 2D BV variant.

Same treatment as §2e.

### 2g. `analyse_split_*.py` (2 active copies)

> **Inventory correction**: Plan originally said "5 experiments" — only Exp 1 (`analyse_split_ISO_TRI_BI.py`) and Exp 2 (`analyse_split_CrossVal.py`) have active copies. Exp 3 has none. 1 deprecated copy exists.

Extract heatmap generation into `common/plotting.py`.

### 2h. `plot_intrinsic_rates_simple.py` (1 active copy — Exp 1 only)

> **Inventory correction**: Plan originally said "2 experiments" — only Exp 1 has this script. No cross-experiment dedup needed; can still benefit from using library's `load_HDXer_kints()`.

Replace manual HDXer loading with library's `load_HDXer_kints()` from `jaxent/src/utils/HDXer/load_data.py`.

### 2i. Fitting scripts: `optimise_fn.py` and variants (3 active copies — count confirmed ✓)

**Before** (`optimise_fn.py` in `2_CrossValidation/fitting/jaxENT/`, 876 lines):
- 5+ near-identical functions: `run_optimise_ISO_TRI_BI_maxENT`, `run_optimise_ISO_TRI_BI_maxENT_BV`, `run_optimise_ISO_TRI_BI_maxENT_MAE`, etc.
- Duplicated `create_data_loaders()`, `optimise_sweep()` functions
- Only differences: loss function combinations, parameter masks, learning rate schedules

**After** (~50 lines per experiment, config-driven):
```./venv/bin/python 
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

> **Review addition (2026-02-18)**: The following per-split runner scripts import from `optimise_fn.py` and are downstream dependents that **must be updated in the same phase**:
> - `1_IsoValidation_OMass/fitting/jaxENT/optimise_ISO_TRI_BI_splits_Sigma.py`
> - `2_CrossValidation/fitting/jaxENT/optimise_ISO_TRI_BI_splits_maxENT.py`
> - `3_CrossValidationBV/fitting/jaxENT/optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py`
>
> These should be refactored to call `common.optimization.run_optimization()` with the appropriate `LossConfig` from `config.yaml`.

### 2j. Fitting scripts: `featurise_*.py` (2 active + 1 variant)

> **Inventory correction**: Plan originally said "3 experiments" — Exp 3 has no featurise script. Active: `featurise_ISO_TRI_BI.py` (Exp 1), `featurise_CrossVal_MSAss_Filtered.py` (Exp 2), plus `featurise_ISO_TRI_BI_SCALING.py` variant (Exp 1).

- Import `load_HDXer_kints()` from `common.loading`
- Use `paths.resolve_example_path()` instead of hardcoded relative paths
- Reduce from ~229 lines to ~100 lines

### 2k. Data prep scripts: `splitdata_*.py` (2 active copies)

> **Inventory correction**: Plan originally said "3 experiments" — Exp 3 has no splitdata script. Active: `splitdata_ISO.py` (Exp 1), `splitdata_CrossVal.py` (Exp 2).

- Import `save_split_data()` from `common.loading`
- Use centralized path management

### 2l. 2D BV sweep scripts (Exp 3 only — 3 scripts, no cross-experiment duplication)

**Scripts**: `analyse_loss_ISO_TRI_BI_2D_BV.py`, `recovery_analysis_ISO_TRI_BI_2D_BV.py`, `weights_validation_ISO_TRI_BI_2D_BV.py`

> **Inventory correction**: These are **Exp 3's only analysis scripts** — it has no 1D variants at all. Do NOT merge these into the 1D scripts; instead, refactor them to use `common/` directly while preserving 2D sweep behavior.

These are 2D-parameter-sweep variants of the 1D families in §2a/2e/2f. They iterate over a grid of `(maxent_scaling, bv_reg)` values and load results keyed by both parameters.

**Refactoring approach**: Refactor these scripts to use `common/` helpers (loading, analysis, plotting) with the 2D sweep configuration from `SweepConfig`. Extend `common/loading.load_all_optimization_results()` to accept an optional `grid_params` argument for 2D result structures.

### 2m. PUF analysis scripts (Exp 2 only — no cross-experiment duplication)

> **Inventory correction**: Plan originally said "3 experiments (2, 3, ignoring 4)" — Exp 3 has **no PUF scripts**. `analyse_cluster_GlobalFeatures_PUFs.py` has **no active copy** (only in `deprecated/`). Active copies are all in Exp 2 only.

**Active scripts (Exp 2 only):**
- `analyse_LocalFeatures_PUFs.py` (~2552 lines) — local structural feature analysis
- `cluster_LocalFeatures_PUF.py` (~2018 lines) — PCA + k-means/rules-based clustering
- `compute_recovery%_PUF.py` — PUF recovery percentage calculation

**Removed from scope:**
- `analyse_cluster_GlobalFeatures_PUFs.py` — fully deprecated, no active copy

These are large, CLI-driven scripts with proper `if __name__ == "__main__"` guards and argparse. Since they only exist in one experiment, no cross-experiment dedup is needed.

**Refactoring approach**: Internal cleanup only:
1. Extract hardcoded colour maps and structural-region JSON paths into `config.yaml`
2. Add `--config` flag to each script's argparse to load defaults from YAML
3. Clean up `sys.path` hacks

### 2n. Comparison and visualization scripts (3 active copies each)

**Scripts**:
- `plot_compare_jaxENT_HDXer.py` (×3 — count confirmed ✓) — bar plots comparing HDXer vs jaxENT results
- `plot_selected_models_ISO_TRI_BI.py` (×3, not ×2 as originally claimed) — model selection visualization

> **Inventory correction**: `plot_selected_models_ISO_TRI_BI.py` exists in all 3 active experiments, not 2.

**Refactoring approach**:
1. Diff cross-experiment copies; if identical keep one canonical copy
2. Extract ~60 lines of hardcoded dicts (`EXPERIMENT_COLOURS`, `SPLIT_NAME_MAPPING`, `SPLIT_TYPE_NORMALIZATION`) into `config.yaml`
3. Move shared plot helpers (hatching, annotation) into `common/plotting.py`

### 2o. Statistical analysis scripts (exists in 1–3 experiments)

**Scripts**:
- `analyse_scores_mixed_linear_model.py` (×3 copies — count confirmed ✓)
- `calculate_state_ratios.py` (×1 active, Exp 2 only — not ×3 as claimed; 1 deprecated copy exists)
- `fixed_effects_linear_model_ISO_TRI_BI.py` — **not found** (may already be deleted or never existed)
- `cross_experiment_mixed_effects_model.py` (×1, `combined_fixed_effects/` — unique, leave as-is)

> **Inventory correction**: `calculate_state_ratios.py` has only 1 active copy (Exp 2). `fixed_effects_linear_model_ISO_TRI_BI.py` was not found in the filesystem.

**Refactoring approach**:
1. ~~Delete the empty `fixed_effects_linear_model_ISO_TRI_BI.py`~~ — already absent
2. `calculate_state_ratios.py` — only 1 copy, no dedup needed; internal cleanup only
3. Diff `analyse_scores_mixed_linear_model.py` copies — extract experiment-specific formula strings and colour maps into config

### 2p. Remaining unique/utility scripts (no cross-experiment duplication)

The following scripts exist only in one experiment and don't need cross-experiment dedup. They only need `__main__` guards and `sys.path` cleanup (covered by §5c):

- `1_IsoValidation_OMass/analysis/combined_cluster_analysis/Failure_analysis_ISO_TRI_BI_maxent_CV.py`
- `1_IsoValidation_OMass/analysis/combined_cluster_analysis/Failure_analysis_ISO_TRI_BI_maxent_weights.py`
- `1_IsoValidation_OMass/analysis/combined_cluster_analysis/ratio_recovery_ISO_TRI_BI_maxent.py`
- `1_IsoValidation_OMass/data/jaxENT_prepare_TeaA_data.py`
- `1_IsoValidation_OMass/fitting/jaxENT/compute_covariance_matrices.py`
- ~~`1_IsoValidation_OMass/fitting/jaxENT/compute_cprior.py`~~ — does not exist (removed from plan)
- `1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI_SCALING.py` (consider merging with `featurise_ISO_TRI_BI.py` via a `--scaling` flag)
- `combined_fixed_effects/plot_kendalls_combined.py`
- `predict_traj/predict_traj.py`

> **Review addition (2026-02-18)**: The following active scripts were missing from this list and also need `__main__` guards and `sys.path` cleanup:
>
> - `1_IsoValidation_OMass/analysis/CV_validation_ISO_TRI_BI_precluster.py` (also has duplicated `kl_divergence` and `effective_sample_size` — replace with `common/analysis.py` imports)
> - `1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py`
> - `1_IsoValidation_OMass/data/extract_synthetic_data.py`
> - `1_IsoValidation_OMass/data/get_HDXer_AutoValidation_data.py`
> - `1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py`
> - `1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_synthetic.py`
> - `1_IsoValidation_OMass/fitting/jaxENT/optimise_ISO_TRI_BI_splits_Sigma.py` (downstream dependent of `optimise_fn.py` — see §2i)
> - `2_CrossValidation/data/extract_data_ValDX.py`
> - `2_CrossValidation/data/renumber_pdb.py`
> - `2_CrossValidation/fitting/jaxENT/optimise_ISO_TRI_BI_splits_maxENT.py` (downstream dependent of `optimise_fn.py` — see §2i)
> - `3_CrossValidationBV/fitting/jaxENT/optimise_ISO_TRI_BI_splits_maxENT_BV_Objective.py` (downstream dependent of `optimise_fn.py` — see §2i)

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
datasplit_dir: "./fitting/jaxENT/_datasplits"
clustering_dir: "./fitting/jaxENT/_clustering"      # optional
output_dir: "./analysis/_output"
state_ratios_json: "./analysis/state_ratios.json"    # optional

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
  covariance_matrix_path: "./data/_MoPrP_covariance_matrices/Sigma.npz"  # optional

# 2D parameter sweep (for BV objective scripts, §2l)
sweep:
  maxent_values: [1, 2, 5, 10, 20, 50]
  bv_reg_values: [0.01, 0.1, 1.0, 10.0]

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
| Manual HDXer `.dat` file parsing in `plot_intrinsic_rates_*.py` | `jaxent.src.utils.HDXer.load_data.load_HDXer_kints()` (requires update to return topologies) |
| Manual HDF5 loading (already partially used) | `jaxent.src.utils.hdf.load_optimization_history_from_file()` (keep using) |
| Manual frame averaging in `process_optimisation_results.py` | `jaxent.src.utils.jax_fn.frame_average_features()` (already used in some copies) |
| `sys.path.insert(0, base_dir)` hack at top of every script | Proper package import via `jaxent.examples.common` (since jaxent is installed) |

#### Detailed Plan: Fix Featurisation Topology Gap in `load_HDXer_kints`

> **Review correction (2026-02-18)**: `load_HDXer_kints` currently returns `tuple[Array, list[int]]` (intrinsic rates + bare residue IDs as integers), not "only intrinsic rates" as previously stated. The docstring incorrectly claims it returns `Partial_Topology` objects — this is a stale docstring that should be fixed alongside the upgrade below.

To enable the replacement above, `load_HDXer_kints` must be updated to return actual `Partial_Topology` objects instead of bare residue IDs.

1. **Modify `jaxent/src/utils/HDXer/load_data.py`**:
   - Add `chain: str = "A"` parameter to `load_HDXer_kints`.
   - Update return type from `tuple[Array, list[int]]` to `tuple[Array, list[Partial_Topology]]`.
   - Replace `topology_list.append(resid)` with `TopologyFactory.from_single(chain=chain, residue=resid)` for each rate entry.
   - Fix the stale docstring to match the updated return type.

2. **Update `jaxent/tests/modules/utils/test_HDXer_load_save.py`**:
   - Assert `topology_list` contains `Partial_Topology` objects (not plain integers).
   - Verify residue IDs and chain assignments.

---

## Phase 5: Archive old scripts and clean up

### 5a. Archive originals
For each experiment directory, create an `analysis/_archive/` subdirectory and move the pre-refactor versions of all modified scripts there. This preserves easy side-by-side comparison during the transition period. Example:
```
2_CrossValidation/analysis/_archive/analyse_loss_ISO_TRI_BI.py  (original)
2_CrossValidation/analysis/analyse_loss_ISO_TRI_BI.py           (refactored)
```

### 5b. Delete stale copies
~~Remove obviously stale files (files with "copy" or "OLD" suffixes) that are not the canonical versions:~~

> **Review update (2026-02-18)**: Both files previously listed here (`score_models_ISO_TRI_BI copy.py` and `cluster_LocalFeatures_PUF_OLD.py`) have already been deleted. No action needed. `3_CrossValidationBV/analysis/cluster_LocalFeatures_PUF_OLD.py` and `4_CrossValidation_FunctionalMaxENT/analysis/cluster_LocalFeatures_PUF_OLD.py` are already in `deprecated/` — no action needed.

### 5c. Add `__main__` guards
For any remaining scripts not covered in Phase 2 (data prep scripts, one-off analyses), ensure they:
1. Wrap top-level code in `def main()` + `if __name__ == "__main__":` guard
2. Remove the `sys.path.insert` hack (rely on package installation)

---

## Files to create
- `jaxent/examples/__init__.py`
- `jaxent/examples/common/__init__.py`
- `jaxent/examples/common/config.py`
- `jaxent/examples/common/loading.py`
- `jaxent/examples/common/analysis.py`
- `jaxent/examples/common/plotting.py`
- `jaxent/examples/common/cli.py`
- `jaxent/examples/common/optimization.py`
- `jaxent/examples/common/losses.py`
- `jaxent/examples/common/paths.py`
- `jaxent/examples/1_IsoValidation_OMass/config.yaml`
- `jaxent/examples/2_CrossValidation/config.yaml`
- `jaxent/examples/3_CrossValidationBV/config.yaml`

> **Note**: `1_IsoValidation/` doesn't exist as a separate directory. `4_CrossValidation_FunctionalMaxENT/` is deprecated — no config.yaml needed.
- `jaxent/examples/README.md`

## Files to modify (major refactors)

### Analysis scripts (§2a–2h) — corrected active counts
- `analyse_loss_ISO_TRI_BI.py` — 2 active 1D (Exp 1, 2)
- ~~`analyse_loss_MoPrP.py`~~ — removed (fully deprecated)
- `score_models_ISO_TRI_BI.py` — 3 active (Exp 1, 2, 3)
- `process_optimisation_results.py` — 3 active (Exp 1, 2, 3)
- `weights_validation_ISO_TRI_BI_precluster.py` — 2 active 1D (Exp 1, 2)
- `recovery_analysis_ISO_TRI_BI_precluster.py` — 2 active 1D (Exp 1, 2)
- `analyse_split_*.py` — 2 active (Exp 1, 2)
- `plot_intrinsic_rates_simple.py` — 1 active (Exp 1 only)

### 2D BV sweep scripts (§2l) — Exp 3 only
- `analyse_loss_ISO_TRI_BI_2D_BV.py` — 1 active (Exp 3)
- `recovery_analysis_ISO_TRI_BI_2D_BV.py` — 1 active (Exp 3)
- `weights_validation_ISO_TRI_BI_2D_BV.py` — 1 active (Exp 3)

### PUF analysis scripts (§2m) — Exp 2 only
- `analyse_LocalFeatures_PUFs.py` — 1 active (Exp 2)
- ~~`analyse_cluster_GlobalFeatures_PUFs.py`~~ — removed (fully deprecated)
- `cluster_LocalFeatures_PUF.py` — 1 active (Exp 2)
- `compute_recovery%_PUF.py` — 1 active (Exp 2)

### Comparison/visualization scripts (§2n)
- `plot_compare_jaxENT_HDXer.py` — 3 active (Exp 1, 2, 3)
- `plot_selected_models_ISO_TRI_BI.py` — 3 active (Exp 1, 2, 3)

### Statistical analysis scripts (§2o)
- `analyse_scores_mixed_linear_model.py` — 3 active (Exp 1, 2, 3)
- `calculate_state_ratios.py` — 1 active (Exp 2 only)

### Fitting scripts (§2i–2k)
- `optimise_fn.py` — 3 active (Exp 1, 2, 3)
- `optimise_ISO_TRI_BI*.py` variants — deprecated (in `deprecated/` dirs)
- `featurise_*.py` — 2 active + 1 variant (Exp 1, 2)
- `splitdata_*.py` — 2 active (Exp 1, 2)

## Files to archive (move to `_archive/`)
- All original versions of the scripts listed in "Files to modify" above
- Include both `analysis/_archive/` and `fitting/_archive/` subdirectories

## Files to delete (stale/empty files)
~~`1_IsoValidation_OMass/analysis/score_models_ISO_TRI_BI copy.py`~~ — already deleted
~~`2_CrossValidation/analysis/cluster_LocalFeatures_PUF_OLD.py`~~ — already deleted

> **Note**: `fixed_effects_linear_model_ISO_TRI_BI.py` was not found in the filesystem. Other `_OLD.py` files are in `deprecated/` already. No stale files remain to delete.

---

## Verification

After each phase:
1. Run `./venv/bin/python  -c "from jaxent.examples.common import config, loading, analysis, plotting, optimization"` to verify module imports
2. For each refactored script, run it from its original directory and verify output files match (diff PNGs and CSVs against pre-refactor baselines)
3. Run `./venv/bin/python  -m pytest` on any existing tests
4. Spot-check that the `sys.path` hack removal doesn't break imports by running scripts from different working directories
5. Test YAML config loading: `./venv/bin/python  -c "from jaxent.examples.common.config import ExperimentConfig; c = ExperimentConfig.from_yaml('config.yaml'); print(c)"`

### Unit tests for new modules
Create test files at `jaxent/tests/examples/`:
- `test_common_config.py` — test YAML loading/saving, dataclass validation
- `test_common_loading.py` — test data loading functions
- `test_common_optimization.py` — test optimization configuration


# Plan verification report

## Resolved findings

- [x] **Featurisation topology gap**: Scripts build `TopologyFactory` entries from HDXer rates;
  `load_HDXer_kints()` returns `tuple[Array, list[int]]` (intrinsic rates + bare residue IDs), not `Partial_Topology` objects. Reusing the library helper as-is drops proper topology data. **(Resolved in Phase 4 revision)** The docstring is also stale and will be fixed.
- [x] **Loss function divergence**: Wrappers define covariance-weighted and shape-adjusted losses (e.g.,
  `hdx_uptake_mean_centred_MSE_loss` in `optimise_ISO_TRI_BI_splits_maxENT.py`) that differ from
  `jaxent/src/opt/losses.py`. **Resolution**: Pin the 3 example losses in `examples/common/losses.py` as a stability layer (initially re-exporting from core library), decoupling examples from future core library refactoring.

- [x] **Config/CLI gaps**: YAML sketch used `dasplit_dir` (typo for `datasplit_dir`) and omitted fields routinely
  parsed in scripts. **(Resolved)**: fixed typo, added `state_ratios_json`, `covariance_matrix_path`,
  `SweepConfig` (2D grid axes), `clustering_dir` to `ExperimentConfig`/YAML. Added `from_yaml` validation
  (required-field check + unknown-key warning). Expanded `add_common_args` in `common/cli.py`.

- [x] **Inventory verification (script counts)**: Filesystem audit revealed the plan over-counted active scripts
  by including deprecated copies (Exp 4 fully deprecated, partial Exp 3 deprecated). **(Resolved)**: Updated all
  §2a–2p headings with corrected active counts. Removed §2b (`analyse_loss_MoPrP.py`) and
  `analyse_cluster_GlobalFeatures_PUFs.py` from scope. Corrected §2l to not merge 2D→1D (Exp 3 only has 2D).
  Fixed `plot_selected_models` count from ×2→×3. Corrected `calculate_state_ratios` from ×3→×1.

## Open findings

(None — all findings resolved)

## Next steps

1. ~~Decide how to package examples~~ → **Resolved**: Option A approved (add `__init__.py` and `pyproject.toml`).
2. ~~Map all loss/optimizer variants~~ → **Resolved**: See §1g inventory table. 6 `run_optimise_*` variants across 3 experiments differ only in loss combos and param masks. Exp 3 adds `model_parameters_lr_scale`.
3. Implement Phase 0 (packaging) + Phase 1 modules in build order: `paths` → `config` → `analysis` → `loading` → `losses` → `plotting` → `optimization` → `cli` → `__init__`.
4. Pilot refactor `analyse_loss_*.py` to validate the approach.
5. Proceed with remaining Phase 2 families.