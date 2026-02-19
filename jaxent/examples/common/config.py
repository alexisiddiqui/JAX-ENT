"""
Experiment configuration dataclasses with YAML serialization.

Extracts the repeated configuration dictionaries (ensemble names, color schemes,
scoring weights, convergence rates, loss function names, etc.) scattered across
every example script into declarative dataclasses.
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path



# ---------------------------------------------------------------------------
# Optimisation
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs (used by fitting scripts)."""

    n_steps: int = 10_000
    tolerance: float = 1e-10
    convergence_rates: list[float] = field(
        default_factory=lambda: [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    )
    learning_rate: float = 1e-3
    initial_learning_rate: float = 1e0
    initial_steps: int = 2
    optimizer: str = "adamw"
    ema_alpha: float = 0.5
    forward_model_scaling: float = 100.0
    clip_value: float | None = None
    covariance_matrix_path: str | None = None
    # Exp 3 adds this; Exp 1/2 leave at default 1.0 (backward compatible)
    model_parameters_lr_scale: float = 1.0


@dataclass
class SweepConfig:
    """Configuration for 2D parameter sweeps (Exp 3 BV objective scripts)."""

    maxent_values: list[float] = field(default_factory=list)
    bv_reg_values: list[float] = field(default_factory=list)


@dataclass
class LossConfig:
    """Configuration for loss function composition."""

    primary_loss: str  # e.g. "hdx_uptake_mean_centred_MSE_loss"
    regularization_losses: list[dict] = field(default_factory=list)
    optimize_bv_params: bool = False
    maxent_scaling: float = 1.0


# ---------------------------------------------------------------------------
# Plotting & scoring
# ---------------------------------------------------------------------------

@dataclass
class PlotStyle:
    """Visual configuration for a specific experiment."""

    ensemble_colors: dict[str, str] = field(default_factory=dict)
    loss_markers: dict[str, str] = field(default_factory=dict)
    split_type_colors: dict[str, str] = field(default_factory=dict)
    split_name_mapping: dict[str, str] = field(default_factory=dict)
    dpi: int = 300
    figsize_wide: tuple[int, int] = (16, 6)
    figsize_square: tuple[int, int] = (10, 10)


@dataclass
class ScoringConfig:
    """Config for model scoring / recovery analysis."""

    ensemble_feature_map: dict[str, str] = field(default_factory=dict)
    ensemble_clustering_map: dict[str, str] = field(default_factory=dict)
    state_mapping: dict[int, str] = field(default_factory=dict)
    scoring_weights: dict[str, float] = field(default_factory=dict)
    ground_truth_ratios: dict[str, float] = field(default_factory=dict)
    recovery_metric: str = "jsd"  # "jsd" or "ratio"


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = frozenset(
    {"ensembles", "features_dir", "datasplit_dir", "num_splits", "convergence_rates"}
)
# Either results_dir or results_prefix must be present (checked separately)


@dataclass
class ExperimentConfig:
    """Top-level config tying an experiment's analysis + fitting settings together."""

    # Core identifiers
    ensembles: list[str] = field(default_factory=list)
    loss_functions: list[str] = field(default_factory=list)
    num_splits: int = 3
    convergence_rates: list[float] = field(default_factory=list)

    # Paths (relative to experiment directory by default)
    results_dir: str = ""
    results_prefix: str | None = None  # Alternative to results_dir: finds most recent matching directory
    features_dir: str = ""
    datasplit_dir: str = ""
    clustering_dir: str | None = None
    output_dir: str | None = None
    state_ratios_json: str | None = None

    # Sub-configs
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss_configs: dict[str, LossConfig] = field(default_factory=dict)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    style: PlotStyle | None = None
    scoring: ScoringConfig | None = None

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file with validation."""
        import yaml

        with open(path) as f:
            data: dict = yaml.safe_load(f)

        # Required-field check
        missing = _REQUIRED_FIELDS - data.keys()
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

        # Either results_dir or results_prefix must be present
        if not data.get("results_dir") and not data.get("results_prefix"):
            raise ValueError("Config must specify either 'results_dir' or 'results_prefix'")

        # Unknown-key warning (catches typos like 'dasplit_dir')
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = data.keys() - known
        if unknown:
            warnings.warn(f"Unknown config keys (typo?): {unknown}", stacklevel=2)

        # Filter unknown keys after warning
        data = {k: v for k, v in data.items() if k in known}

        # Recursively construct nested dataclasses
        if "optimization" in data and isinstance(data["optimization"], dict):
            data["optimization"] = OptimizationConfig(**data["optimization"])
        if "loss_configs" in data and isinstance(data["loss_configs"], dict):
            data["loss_configs"] = {k: LossConfig(**v) for k, v in data["loss_configs"].items()}
        if "sweep" in data and isinstance(data["sweep"], dict):
            data["sweep"] = SweepConfig(**data["sweep"])
        if "style" in data and isinstance(data["style"], dict):
            data["style"] = PlotStyle(**data["style"])
        if "scoring" in data and isinstance(data["scoring"], dict):
            # state_mapping keys come as ints from YAML but dict annotation is dict[int, str]
            scoring_data = data["scoring"]
            if "state_mapping" in scoring_data:
                scoring_data["state_mapping"] = {
                    int(k): v for k, v in scoring_data["state_mapping"].items()
                }
            data["scoring"] = ScoringConfig(**scoring_data)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        import yaml

        def _to_dict(obj):  # noqa: ANN001
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)
