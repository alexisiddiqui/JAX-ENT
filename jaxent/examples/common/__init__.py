"""
``jaxent.examples.common`` — shared infrastructure for example scripts.

Re-exports the public API so callers can write
``from jaxent.examples.common import ExperimentConfig, loading, analysis``.
"""

from .config import (
    ExperimentConfig,
    LossConfig,
    OptimizationConfig,
    PlotStyle,
    ScoringConfig,
    SweepConfig,
)
from .paths import ensure_output_dir, get_examples_root, resolve_example_path

__all__ = [
    # Config dataclasses
    "ExperimentConfig",
    "OptimizationConfig",
    "SweepConfig",
    "LossConfig",
    "PlotStyle",
    "ScoringConfig",
    # Path utilities
    "resolve_example_path",
    "ensure_output_dir",
    "get_examples_root",
]
