#!/usr/bin/env python3
"""Weights validation wrapper for 4_aSyn 2D sweeps."""

import argparse
import os
from pathlib import Path

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir

BV_REG_FUNCTIONS = ["L1", "L2"]


def _resolve_path(script_dir: Path, value: str | None, absolute_paths: bool) -> Path | None:
    if value is None:
        return None
    return Path(value) if absolute_paths else (script_dir / value).resolve()


def _resolve_results_dir(script_dir: Path, exp_dir: Path, cfg: ExperimentConfig, args) -> Path:
    if args.results_dir:
        return _resolve_path(script_dir, args.results_dir, args.absolute_paths)
    if cfg.results_prefix:
        prefix_path = Path(cfg.results_prefix)
        base_path = exp_dir / prefix_path.parent
        found = find_most_recent_dir(base_path, prefix_path.name)
        if found is None:
            raise FileNotFoundError(
                f"No directory matching prefix '{prefix_path.name}' found in {base_path}"
            )
        return found
    if cfg.results_dir:
        return (exp_dir / cfg.results_dir).resolve()
    raise ValueError("Provide --results-dir or set results_prefix/results_dir in config")


def main() -> None:
    parser = argparse.ArgumentParser(description="4_aSyn weight validation")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--config", default=None)
    parser.add_argument("--absolute-paths", action="store_true", default=False)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir.parent
    cfg = ExperimentConfig.from_yaml(
        Path(args.config).resolve() if args.config else exp_dir / "config.yaml"
    )

    results_dir = _resolve_results_dir(script_dir, exp_dir, cfg, args)

    if args.output_dir:
        output_dir = _resolve_path(script_dir, args.output_dir, args.absolute_paths)
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = script_dir / f"_analysis_weights_2d_{base_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results directory : {results_dir}")
    print(f"Output directory  : {output_dir}")
    print(f"EMA               : {args.ema}")

    results = loading.load_all_optimization_results_2d(
        results_dir=str(results_dir),
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        bv_reg_functions=BV_REG_FUNCTIONS,
        num_splits=cfg.num_splits,
        EMA=args.ema,
        verbose=True,
    )

    weights_df = analysis.extract_final_weights_2d(results)
    if not weights_df.empty:
        weights_df.to_csv(output_dir / "weights_2d_sweep_data.csv", index=False)
        plotting.plot_2d_heatmaps_grid(
            weights_df,
            str(output_dir),
            metric="effective_sample_size",
            metric_label="Effective Sample Size",
        )
        plotting.plot_2d_heatmaps_grid(
            weights_df,
            str(output_dir),
            metric="kl_divergence",
            metric_label="KL Divergence",
        )
        plotting.plot_1d_slices_2d_sweep(
            weights_df,
            str(output_dir),
            metric="effective_sample_size",
            metric_label="Effective Sample Size",
        )
        plotting.plot_1d_slices_2d_sweep(
            weights_df,
            str(output_dir),
            metric="kl_divergence",
            metric_label="KL Divergence",
        )

    kld_df = analysis.compute_pairwise_kld_between_splits_2d(results)
    if not kld_df.empty:
        kld_df.to_csv(output_dir / "kld_between_splits_2d_sweep_data.csv", index=False)
        plotting.plot_2d_heatmaps_grid(
            kld_df,
            str(output_dir),
            metric="mean_kld_between_splits",
            metric_label="Mean KLD Between Splits",
            cmap="Blues",
        )

    print("Weights validation completed.")


if __name__ == "__main__":
    main()
