# `jaxent.examples.common` — Shared Infrastructure for Example Workflows

Shared infrastructure module used by all three active JAX-ENT example suites (`1_IsoValidation_OMass`, `2_CrossValidation`, `3_CrossValidationBV`). Consolidates configuration, data loading, analysis, plotting, and optimization code that was previously duplicated across 48+ example scripts.

This module serves as a **decoupling layer** between example scripts and the core `jaxent.src` library. Example scripts import from `common` rather than directly from `jaxent.src`, which insulates them from core API changes.

---

## Module map

```
common/
├── __init__.py          # Public API: config dataclasses + path utilities
├── config.py            # ExperimentConfig hierarchy with YAML I/O
├── paths.py             # Path resolution and directory helpers
├── cli.py               # Shared argparse patterns + config loading
├── loading.py           # Result loading, feature loading, data splitting
├── losses.py            # Loss function registry (string name → callable)
├── optimization.py      # Unified optimization orchestration
├── analysis/            # Analysis computations (numpy/pandas, no JAX)
│   ├── stats.py         #   KL divergence, effective sample size
│   ├── convergence.py   #   Best convergence threshold selection
│   ├── loss_trajectories.py  # Flatten nested results → DataFrame
│   ├── validation.py    #   Array validation and MSE calculation
│   ├── weights.py       #   Frame weight extraction and KL/ESS tracking
│   ├── scoring.py       #   Model scoring and selection
│   ├── clustering.py    #   Cluster ratios, JSD recovery, dMSE
│   ├── pairwise_kld.py  #   Pairwise symmetric KLD between splits
│   ├── mlm.py           #   Regression, stability, mixed-effects analysis
│   └── selected_models.py  # Ground-truth scoring, fixed-effects, concordance
└── plotting/            # Matplotlib/seaborn visualisations
    ├── style.py         #   Publication-quality defaults (PlotStyle injection)
    ├── heatmaps.py      #   Convergence x maxent heatmaps
    ├── scores.py        #   Score heatmaps, violin plots
    ├── comparisons.py   #   Best model comparison plots
    ├── distributions.py #   Frame weight distribution plots
    ├── convergence.py   #   Train/val loss convergence plots
    ├── kld.py           #   KLD between splits, sequential maxent KLD
    ├── sweeps.py        #   2D heatmap grids, 1D slices, best hyperparams
    ├── volcano.py       #   KL vs recovery volcano plots
    ├── recovery.py      #   Metric vs regularisation strength
    ├── splits.py        #   Split distribution and composition heatmaps
    ├── uptake.py        #   HDX uptake heatmaps and comparisons
    ├── gaps.py          #   Gap/coverage analysis plots
    ├── mlm.py           #   Regression coefficient and variance plots
    └── selected_models.py  # Score, minimax, rank, and fixed-effects panels
```

---

## Core modules

### `config.py` — Configuration dataclasses

Defines the configuration hierarchy used by all example scripts. All configs support YAML serialisation.

| Dataclass | Purpose | Key fields |
|-----------|---------|------------|
| `ExperimentConfig` | Top-level experiment config | `ensembles`, `loss_functions`, `num_splits`, paths, sub-configs |
| `OptimizationConfig` | Optimization hyperparameters | `n_steps`, `learning_rate`, `convergence_rates`, `ema_alpha` |
| `LossConfig` | Loss function composition | `primary_loss`, `regularization_losses`, `optimize_bv_params` |
| `SweepConfig` | 2D hyperparameter sweeps | `maxent_values`, `bv_reg_values` |
| `PlotStyle` | Visual styling | `ensemble_colors`, `loss_markers`, `dpi`, `figsize_*` |
| `ScoringConfig` | Model evaluation | `state_mapping`, `ground_truth_ratios`, `recovery_metric` |

Key methods:
- `ExperimentConfig.from_yaml(path)` — load and validate (checks required fields, warns on typos)
- `ExperimentConfig.to_yaml(path)` — serialise to YAML

### `paths.py` — Path utilities

Replaces fragile `os.path.join(os.path.dirname(__file__), "../../...")` patterns.

- `get_examples_root()` — returns `jaxent/examples/` directory
- `resolve_example_path(example_name, subdir)` — resolve within an example directory
- `ensure_output_dir(path)` — create directory tree, return Path
- `find_most_recent_dir(base_path, prefix)` — find timestamped result directories
- `derive_processed_output_dir(results_dir)` — sibling `_processed_<name>` convention

### `cli.py` — Shared CLI patterns

- `add_common_args(parser)` — register standard flags (`--config`, `--results-dir`, `--output-dir`, `--ema`, etc.)
- `load_config_with_overrides(args, script_dir)` — load YAML config, apply CLI overrides

### `loading.py` — Data loading and result aggregation

The largest module (~900 lines). Consolidates result loading across all nesting levels.

**Result loading** (returns nested dicts keyed by ensemble/loss/split/maxent):
- `load_all_optimization_results(...)` — basic loading
- `load_all_optimization_results_with_maxent(...)` — auto-discovers split types
- `load_all_optimization_results_2d(...)` — 2D sweeps (maxent + BV regularisation)

**Feature and data loading:**
- `load_features_and_topology(features_dir, ensemble_name, feature_map)` — returns `(BV_input_features, list[Partial_Topology])`
- `load_experimental_data(results_dir, datasplit_dir, split_type, split_idx)` — returns `(train, val, test, timepoints)`
- `load_clustering_results(clustering_dir, clustering_map)` — cluster assignments + frame data

**Data splitting:**
- `run_data_splits(num_splits, output_dir, hdx_data, ...)` — execute and save N splits
- `load_split_data(split_dir)` — reload saved splits

**Featurisation:**
- `load_HDXer_kints(kint_path)` — parse intrinsic rates from .dat file
- `featurise_trajectory(...)` — single trajectory featurisation

### `losses.py` — Loss function registry

Re-exports core library losses and maps YAML string names to callables.

```python
from jaxent.examples.common.losses import get_loss_function_by_name, LOSS_REGISTRY

loss_fn = get_loss_function_by_name("hdx_uptake_mean_centred_MSE_loss")
```

Available losses:
- **Data-fit:** `hdx_uptake_mean_centred_MSE_loss`, `hdx_uptake_MSE_loss`, `hdx_uptake_sigma_MSE_loss`, `hdx_uptake_MAE_loss_vectorized`
- **Regularisation:** `maxent_convexKL_loss`, `maxent_L2_loss`, `model_params_L1_loss`, `model_params_L2_loss`

### `optimization.py` — Unified optimization

Replaces 5+ near-identical `run_optimise_*` variants.

- `run_optimization(train_data, val_data, prior_data, features, forward_model, ...)` — single entry point for all fitting scripts. Builds loss list from `LossConfig`, constructs `Simulation`, initialises optimizer with partition masks, runs `optimise_sweep()`, saves results to HDF5.
- `create_data_loaders(hdx_data, train_data, val_data, features, feature_top, cov_matrix)` — create `ExpD_Dataloader` with optional covariance matrix.
- `optimise_sweep(simulation, data, n_steps, tolerance, convergence, ...)` — EMA-based optimization with multi-threshold convergence sweep.
- `BV_uptake_ForwardPass_frames` — per-frame uptake predictions (for reweighting in post-processing).

---

## `analysis/` subpackage

Pure numpy/pandas computations — no JAX imports required.

| Module | Key functions | Purpose |
|--------|---------------|---------|
| `stats.py` | `kl_divergence(p, q)`, `effective_sample_size(weights)` | Core statistics (previously duplicated 11x and 6x respectively) |
| `convergence.py` | `find_best_convergence_threshold(history)` | Select step with lowest validation loss |
| `loss_trajectories.py` | `extract_loss_trajectories(results)`, `extract_loss_trajectories_2d(results)` | Flatten nested result dicts into DataFrames |
| `validation.py` | `get_experimental_uptake(data)`, `calculate_mse(pred, exp)` | Shape alignment and MSE with robustness checks |
| `weights.py` | `extract_frame_weights_kl(results)`, `extract_final_weights(results)` | Frame weight extraction with KL/ESS at every step |
| `scoring.py` | `compute_model_scores(df)`, `select_best_models(df)`, `calculate_work_metrics(...)` | Composite scoring and thermodynamic work metrics |
| `clustering.py` | `calculate_recovery_JSD(...)`, `analyze_conformational_recovery(...)` | Cluster population recovery via Jensen-Shannon divergence |
| `pairwise_kld.py` | `compute_pairwise_kld_between_splits(data)` | Pairwise symmetric KLD between replicate splits |
| `mlm.py` | `multiple_regression_analysis(...)`, `mixed_effects_analysis(...)` | Statistical modeling (statsmodels-based) |
| `selected_models.py` | `run_fixed_effects_analysis(...)`, `calculate_concordance_maps(...)` | Ground-truth scoring and fixed-effects regression |

---

## `plotting/` subpackage

All plotting functions accept an optional `PlotStyle` dataclass. If not provided, defaults from `style.py` are used. Call `setup_publication_style()` once at the start of any plotting script.

| Module | Key functions | Purpose |
|--------|---------------|---------|
| `style.py` | `setup_publication_style()` | Set matplotlib/seaborn defaults (replaces blocks duplicated in ~12 scripts) |
| `heatmaps.py` | `plot_convergence_maxent_heatmaps(...)` | Convergence threshold x maxent value heatmaps |
| `scores.py` | `plot_model_score_heatmaps(...)`, `create_violin_plots(...)` | Score distributions across ensembles/loss functions |
| `convergence.py` | `plot_loss_convergence(...)`, `plot_split_variability(...)` | Train/val error vs convergence threshold |
| `distributions.py` | `plot_weight_distribution_lines(...)` | Frame weight distributions with maxent/convergence panels |
| `kld.py` | `plot_kld_between_splits(...)` | KLD between replicate splits |
| `sweeps.py` | `plot_2d_heatmaps_grid(...)`, `plot_1d_slices_2d_sweep(...)` | 2D hyperparameter sweep visualisation |
| `volcano.py` | `plot_volcano_kl_recovery(...)` | KL divergence vs recovery percentage |
| `recovery.py` | `plot_metric_vs_regularization_strength(...)` | Metric vs regularisation parameter plots |
| `uptake.py` | `plot_uptake_heatmap(...)`, `plot_combined_uptake_comparison(...)` | HDX uptake comparison heatmaps |
| `selected_models.py` | `plot_score_panel(...)`, `plot_fixed_effects(...)` | Multi-panel model selection visualisations |

---

## Configuration: YAML + CLI

Example scripts are driven by a `config.yaml` file that maps to `ExperimentConfig`. CLI flags can override any path.

### Example `config.yaml`

```yaml
# Required fields
ensembles:
  - ISO_TRI
  - ISO_BI

loss_functions:
  - mcMSE
  - MSE

num_splits: 3

convergence_rates:
  - 1.0e-3
  - 1.0e-4
  - 1.0e-5
  - 1.0e-6
  - 1.0e-7

# Paths (relative to experiment directory by default)
results_prefix: "fitting/jaxENT/_optimise_test_SIGMA_500__"
features_dir: "fitting/jaxENT/_featurise"
datasplit_dir: "fitting/jaxENT/_datasplits"
clustering_dir: "data/_clustering_results"
output_dir: "analysis/_analysis_loss_output"

# Plot styling
style:
  ensemble_colors:
    ISO_BI: "saddlebrown"
    ISO_TRI: "indigo"
  loss_markers:
    mcMSE: "o"
    MSE: "s"
  dpi: 300

# Model evaluation
scoring:
  state_mapping:
    0: "open"
    1: "closed"
  ground_truth_ratios:
    open: 0.4
    closed: 0.6
  recovery_metric: jsd
```

### CLI override pattern

```bash
python analyse_loss.py \
    --config config.yaml \
    --results-dir /absolute/path/to/results \
    --ema
```

Required fields in config: `ensembles`, `features_dir`, `datasplit_dir`, `num_splits`, `convergence_rates`. Either `results_dir` or `results_prefix` must also be present.

---

## Data flow

The typical analysis pipeline follows this sequence:

```
config.yaml
    │
    ▼
cli.load_config_with_overrides()  →  ExperimentConfig
    │
    ▼
loading.load_all_optimization_results_with_maxent()
    │  returns: {split_type: {ensemble: {loss: {maxent: {split_idx: history}}}}}
    ▼
analysis.extract_loss_trajectories()
    │  returns: pandas DataFrame (one row per model variant)
    ▼
analysis.compute_model_scores()
    │  returns: DataFrame with composite scores
    ▼
plotting.plot_loss_convergence()
plotting.plot_model_score_heatmaps()
    │  saves: figures to output_dir
    ▼
analysis.analyze_conformational_recovery()  (if clustering data available)
    │  returns: DataFrame with JSD recovery metrics
    ▼
plotting.plot_volcano_kl_recovery()
```

---

## Import patterns

Example scripts use two import styles. Both work; direct submodule imports are more common.

**Namespace imports:**
```python
from jaxent.examples.common import analysis, loading, paths
from jaxent.examples.common.config import ExperimentConfig

config = ExperimentConfig.from_yaml("config.yaml")
results = loading.load_all_optimization_results_with_maxent(...)
df = analysis.extract_loss_trajectories(results)
```

**Direct imports:**
```python
from jaxent.examples.common.config import LossConfig, OptimizationConfig
from jaxent.examples.common.optimization import run_optimization
from jaxent.examples.common.loading import load_HDXer_kints, featurise_trajectory
from jaxent.examples.common.plotting import setup_publication_style, plot_loss_convergence
from jaxent.examples.common.analysis import (
    extract_loss_trajectories,
    compute_model_scores,
    select_best_models,
)
```

---

## Extension guide

### Adding a new analysis function

1. Add the function to the appropriate module in `analysis/` (or create a new module if it represents a distinct concern).
2. Export it from `analysis/__init__.py` and add to `__all__`.
3. Keep the function pure numpy/pandas — no JAX imports in analysis modules.

### Adding a new plotting function

1. Add to the appropriate module in `plotting/` (or create a new module).
2. Accept an optional `style: PlotStyle | None = None` parameter. Use `_get_style(style)` from `plotting/style.py` for defaults.
3. Export from `plotting/__init__.py` and add to `__all__`.

### Adding a new loss function

1. Implement in `jaxent.src.opt.losses` (core library) or define locally in `losses.py`.
2. Add an entry to `LOSS_REGISTRY` in `losses.py` mapping a string name to the callable.
3. The string name can then be used in `config.yaml` under `loss_configs.primary_loss` or `regularization_losses`.

### Adding a new config field

1. Add the field to the relevant dataclass in `config.py` (with a default value for backward compatibility).
2. If it's a nested dataclass, add construction logic in `ExperimentConfig.from_yaml()`.
3. Update `_REQUIRED_FIELDS` if the new field is mandatory.

---

## Conventions and gotchas

- **Path resolution:** Paths in `config.yaml` are relative to the experiment directory by default. Use `--absolute-paths` flag to treat them as absolute. The `resolve_script_paths()` function handles this.
- **`results_prefix` vs `results_dir`:** Use `results_prefix` to auto-discover the most recent timestamped results directory matching a prefix. Use `results_dir` for an explicit path.
- **Nested result dicts:** Loading functions return deeply nested dicts (up to 6 levels for 2D sweeps). The `extract_loss_trajectories*` functions flatten these into DataFrames for analysis.
- **PlotStyle injection:** All plotting functions accept an optional `PlotStyle`. Pass `config.style` from your `ExperimentConfig`, or omit for sensible defaults.
- **EMA results:** Pass `--ema` to load `results_EMA.hdf5` instead of `results.hdf5`. This uses exponential moving average parameters.
- **`state_mapping` keys:** YAML parses these as integers. The `from_yaml()` method handles `int(k)` conversion automatically.
- **Loss aliases:** `hdx_uptake_mean_centred_MSE_loss` is an alias for `hdx_uptake_eye_MSE_loss`; `hdx_uptake_MSE_loss` is an alias for `hdx_uptake_sigma_MSE_loss`. Both names work in config files.
