#!/usr/bin/env python3
"""Analyze 2D loss landscapes for 4_aSyn condition sweeps."""

import argparse
from pathlib import Path

from jaxent.examples.common.analysis import (
    compute_model_scores,
    extract_loss_trajectories_2d,
    select_best_models,
)
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.loading import load_all_optimization_results_2d
from jaxent.examples.common.paths import find_most_recent_dir
from jaxent.examples.common.plotting import (
    plot_best_model_comparisons,
    plot_loss_convergence_2d,
    plot_split_variability_2d,
    setup_publication_style,
)

BV_REG_FUNCTIONS = ["L1", "L2"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 4_aSyn 2D optimization landscapes")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    exp_dir = script_dir.parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else exp_dir / args.config
    cfg = ExperimentConfig.from_yaml(config_path)

    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif cfg.results_prefix:
        prefix_parts = cfg.results_prefix.rsplit("/", 1)
        if len(prefix_parts) == 2:
            search_dir = exp_dir / prefix_parts[0]
            results_dir = find_most_recent_dir(search_dir, prefix_parts[1])
        else:
            results_dir = find_most_recent_dir(exp_dir, cfg.results_prefix)
        if not results_dir:
            raise FileNotFoundError(f"No directory found matching prefix: {cfg.results_prefix}")
    elif cfg.results_dir:
        results_dir = exp_dir / cfg.results_dir
    else:
        raise ValueError("Config must specify either results_dir or results_prefix")

    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / (cfg.output_dir or "analysis/_analysis_loss_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_publication_style()

    print(f"Results directory: {results_dir}")
    print(f"Output directory : {output_dir}")

    results = load_all_optimization_results_2d(
        results_dir=str(results_dir),
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        bv_reg_functions=BV_REG_FUNCTIONS,
        num_splits=cfg.num_splits,
        EMA=args.ema,
    )
    if not results:
        print("No results loaded.")
        return

    loss_df = extract_loss_trajectories_2d(results)
    if loss_df.empty:
        print("No loss data extracted.")
        return

    loss_df.to_csv(output_dir / "all_loss_trajectories_2d.csv", index=False)

    if cfg.scoring:
        loss_df = compute_model_scores(loss_df, cfg.scoring)

    best_df = select_best_models(loss_df)
    if not best_df.empty:
        best_df.to_csv(output_dir / "best_models_2d.csv", index=False)
        plot_best_model_comparisons(df=best_df, output_dir=str(output_dir), style=cfg.style)

    plot_loss_convergence_2d(df=loss_df, output_dir=str(output_dir), style=cfg.style, split_type="all")
    plot_split_variability_2d(df=loss_df, output_dir=str(output_dir), style=cfg.style, split_type="all")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
