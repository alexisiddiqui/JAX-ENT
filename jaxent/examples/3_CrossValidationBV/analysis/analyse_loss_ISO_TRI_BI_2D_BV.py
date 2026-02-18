#!/usr/bin/env python3
"""
Analyze 2D loss landscapes (MaxEnt × BV regularization) for AF2 models.

REFACTORED: Reduced from 1622 lines to ~220 by using jaxent.examples.common modules.
Original archived at: _archive/analyse_loss_ISO_TRI_BI_2D_BV.py.original
"""

import argparse
from pathlib import Path

import pandas as pd

from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.loading import (
    load_all_optimization_results_2d,
    load_clustering_results,
)
from jaxent.examples.common.analysis import (
    extract_loss_trajectories_2d,
    compute_model_scores,
    select_best_models,
)
from jaxent.examples.common.plotting import (
    setup_publication_style,
    plot_best_model_comparisons,
)
from jaxent.examples.common.paths import find_most_recent_dir


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze 2D optimization loss landscapes (MaxEnt × BV regularization)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to experiment config file (default: config.yaml)",
    )
    parser.add_argument(
        "--results-dir",
        help="Override results directory from config",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Use EMA results (default: False)",
    )
    args = parser.parse_args()

    # Load configuration
    script_dir = Path(__file__).parent
    exp_dir = script_dir.parent
    config_path = exp_dir / args.config

    print("=" * 80)
    print("EXPERIMENT 3: 2D BV Regularization Loss Landscape Analysis (REFACTORED)")
    print("=" * 80)
    print(f"Loading config from: {config_path}")

    config = ExperimentConfig.from_yaml(config_path)

    # Apply CLI overrides
    if args.results_dir:
        config.results_dir = args.results_dir
        config.results_prefix = None  # Clear prefix if dir is explicitly provided
    if args.output_dir:
        config.output_dir = args.output_dir

    # Resolve paths - handle results_prefix pattern
    if config.results_dir:
        results_dir = exp_dir / config.results_dir
    elif config.results_prefix:
        # Find most recent directory matching prefix
        prefix_parts = config.results_prefix.rsplit("/", 1)
        if len(prefix_parts) == 2:
            search_dir = exp_dir / prefix_parts[0]
            results_dir = find_most_recent_dir(search_dir, prefix_parts[1])
            if not results_dir:
                raise FileNotFoundError(
                    f"No directory found matching prefix: {config.results_prefix}\n"
                    f"Searched in: {search_dir}"
                )
        else:
            results_dir = find_most_recent_dir(exp_dir, config.results_prefix)
            if not results_dir:
                raise FileNotFoundError(
                    f"No directory found matching prefix: {config.results_prefix}"
                )
    else:
        raise ValueError("Config must specify either results_dir or results_prefix")

    output_dir = exp_dir / (config.output_dir or "analysis/_analysis_loss_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup plotting style
    setup_publication_style()

    # Extract BV reg functions from config or use default
    bv_reg_functions = ["L1", "L2"]  # Default BV regularization functions

    print(f"\nConfiguration:")
    print(f"  Ensembles: {config.ensembles}")
    print(f"  Loss functions: {config.loss_functions}")
    print(f"  BV reg functions: {bv_reg_functions}")
    print(f"  Number of splits: {config.num_splits}")
    print(f"  MaxEnt values: {config.sweep.maxent_values if config.sweep else 'default'}")
    print(f"  BV reg values: {config.sweep.bv_reg_values if config.sweep else 'default'}")
    print(f"  Results: {results_dir}")
    print(f"  Output: {output_dir}")

    # Load 2D optimization results
    print("\nLoading 2D optimization results...")
    results = load_all_optimization_results_2d(
        results_dir=str(results_dir),
        ensembles=config.ensembles,
        loss_functions=config.loss_functions,
        bv_reg_functions=bv_reg_functions,
        num_splits=config.num_splits,
        EMA=args.ema,
    )

    if not results:
        print("ERROR: No results loaded!")
        return

    print(f"Loaded 2D results for split types: {list(results.keys())}")

    # Load clustering data if available
    clustering_data = {}
    if config.clustering_dir:
        clustering_dir = Path(config.clustering_dir)
        if not clustering_dir.is_absolute():
            clustering_dir = exp_dir / clustering_dir
        if clustering_dir.exists():
            print(f"\nLoading clustering data from {clustering_dir}...")
            clustering_data = load_clustering_results(
                str(clustering_dir),
                config.scoring.ensemble_clustering_map if config.scoring else None,
            )
        else:
            print(f"\nWarning: Clustering directory not found: {clustering_dir}")

    # Extract 2D loss trajectories
    print("\n" + "=" * 80)
    print("Extracting 2D loss trajectories...")
    print("=" * 80)

    loss_df = extract_loss_trajectories_2d(results)

    if loss_df.empty:
        print("ERROR: No loss data extracted!")
        return

    print(f"Extracted {len(loss_df)} trajectory points")
    print(f"Columns: {list(loss_df.columns)}")

    # Save combined trajectories
    combined_csv_path = output_dir / "all_loss_trajectories_2d.csv"
    loss_df.to_csv(combined_csv_path, index=False)
    print(f"Saved combined loss trajectories: {len(loss_df)} points -> {combined_csv_path}")

    # Compute model scores
    print("\nComputing model scores...")
    if config.scoring:
        loss_df = compute_model_scores(loss_df, config.scoring)
        print(f"Model scores computed using:")
        print(f"  Train penalty: {config.scoring.scoring_weights.get('train_penalty', 1.0)}")
        print(f"  KL penalty: {config.scoring.scoring_weights.get('kl_penalty', 0.0)}")

    # Select best models
    print("\nSelecting best models...")
    best_models_df = select_best_models(loss_df)

    if not best_models_df.empty:
        best_models_csv_path = output_dir / "best_models_2d.csv"
        best_models_df.to_csv(best_models_csv_path, index=False)
        print(f"Selected {len(best_models_df)} best models -> {best_models_csv_path}")

        # Display sample of best models
        print("\nSample of best models:")
        display_cols = ["ensemble", "loss_function", "bv_reg_function", "split_type",
                        "maxent_value", "bv_reg_value", "convergence_step",
                        "train_loss", "val_loss"]
        available_cols = [c for c in display_cols if c in best_models_df.columns]
        print(best_models_df[available_cols].head(10).to_string(index=False))

        # Plot best model comparisons
        print("\nGenerating best model comparison plots...")
        plot_best_model_comparisons(
            df=best_models_df,
            output_dir=str(output_dir),
            style=config.style,
        )
    else:
        print("WARNING: No best models selected")

    # Generate 2D heatmaps (MaxEnt vs BV reg)
    print("\nNote: 2D sweep-specific heatmaps (MaxEnt × BV reg) require custom plotting.")
    print("      Use the saved CSV files to create custom visualizations as needed.")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
