"""
Shared argparse patterns for JAX-ENT example scripts.

Provides ``add_common_args`` to register standard CLI flags, and
``load_config_with_overrides`` to load a YAML config with CLI overrides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard set of analysis/fitting arguments to *parser*.

    Flags added:
    ``--config``, ``--results-dir``, ``--output-dir``, ``--features-dir``,
    ``--datasplit-dir``, ``--clustering-dir``, ``--state-ratios-json``,
    ``--covariance-matrix-path``, ``--ema``, ``--absolute-paths``.
    """
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    parser.add_argument("--results-dir", type=str, default=None, help="Override: results directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Override: output directory")
    parser.add_argument("--features-dir", type=str, default=None, help="Override: features directory")
    parser.add_argument("--datasplit-dir", type=str, default=None, help="Override: datasplit directory")
    parser.add_argument("--clustering-dir", type=str, default=None, help="Override: clustering directory")
    parser.add_argument("--state-ratios-json", type=str, default=None, help="Override: state ratios JSON path")
    parser.add_argument("--covariance-matrix-path", type=str, default=None, help="Override: covariance matrix path")
    parser.add_argument("--ema", action="store_true", default=False, help="Use EMA results (results_EMA.hdf5)")
    parser.add_argument("--absolute-paths", action="store_true", default=False, help="Treat all paths as absolute")


def load_config_with_overrides(
    args: argparse.Namespace,
    script_dir: Path | str | None = None,
) -> ExperimentConfig:
    """Load config from ``--config`` YAML, then apply any CLI overrides.

    Parameters
    ----------
    args:
        Parsed namespace from ``argparse``.
    script_dir:
        Directory of the calling script. Used to resolve relative paths in
        the YAML config.  If ``None``, uses the current working directory.

    Returns
    -------
    Fully-resolved ``ExperimentConfig``.
    """
    if script_dir is None:
        script_dir = Path.cwd()
    script_dir = Path(script_dir)

    if args.config is not None:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        # Build a minimal config from CLI args only
        config = ExperimentConfig()

    # Apply CLI overrides to paths
    override_map = {
        "results_dir": args.results_dir,
        "output_dir": args.output_dir,
        "features_dir": args.features_dir,
        "datasplit_dir": args.datasplit_dir,
        "clustering_dir": args.clustering_dir,
        "state_ratios_json": args.state_ratios_json,
    }

    for field_name, cli_value in override_map.items():
        if cli_value is not None:
            setattr(config, field_name, cli_value)

    # Override covariance matrix path in optimisation config
    if args.covariance_matrix_path is not None:
        config.optimization.covariance_matrix_path = args.covariance_matrix_path

    return config
