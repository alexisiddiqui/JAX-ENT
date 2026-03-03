#!/usr/bin/env python3
"""
Analyze optimization loss landscapes for AF2_filtered and AF2_MSAss models.

REFACTORED: Reduced from 2352 lines to ~244 by using jaxent.examples.common modules.
Original archived at: _archive/analyse_loss_ISO_TRI_BI.py.original
"""

import argparse
from pathlib import Path

import pandas as pd

from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.loading import (
    load_all_optimization_results_with_maxent,
    load_clustering_results,
)
from jaxent.examples.common.analysis import (
    extract_loss_trajectories,
    compute_model_scores,
    select_best_models,
)
from jaxent.examples.common.plotting import (
    setup_publication_style,
    plot_convergence_maxent_heatmaps,
    plot_model_score_heatmaps,
    plot_best_model_comparisons,
    plot_loss_convergence,
    plot_split_variability,
)
from jaxent.examples.common.paths import find_most_recent_dir


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Analyze optimization loss landscapes for AF2_filtered and AF2_MSAss models"
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
    print("EXPERIMENT 2: AF2 CrossValidation Loss Landscape Analysis (REFACTORED)")
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

    print(f"\nConfiguration:")
    print(f"  Ensembles: {config.ensembles}")
    print(f"  Loss functions: {config.loss_functions}")
    print(f"  Number of splits: {config.num_splits}")
    print(f"  Results: {results_dir}")
    print(f"  Output: {output_dir}")

    # Load optimization results (auto-discovers split types)
    print("\nLoading optimization results...")
    results = load_all_optimization_results_with_maxent(
        results_dir=str(results_dir),
        ensembles=config.ensembles,
        loss_functions=config.loss_functions,
        num_splits=config.num_splits,
        EMA=args.ema,
    )

    if not results:
        print("ERROR: No results loaded!")
        return

    print(f"Loaded results for split types: {list(results.keys())}")

    # Load clustering data if available
    clustering_data = {}
    if config.clustering_dir:
        clustering_dir = exp_dir / config.clustering_dir
        if clustering_dir.exists():
            print(f"\nLoading clustering data from {clustering_dir}...")
            clustering_data = load_clustering_results(
                str(clustering_dir),
                config.scoring.ensemble_clustering_map if config.scoring else None,
            )
        else:
            print(f"\nWarning: Clustering directory not found: {clustering_dir}")

    # Process each split type
    all_loss_data = []

    for split_type, split_results in results.items():
        print(f"\n{'=' * 80}")
        print(f"Processing split type: {split_type}")
        print(f"{'=' * 80}")

        # Extract loss trajectories
        print("  Extracting loss trajectories...")
        loss_df = extract_loss_trajectories(
            results=split_results,
            split_type=split_type,
            cluster_assignments=clustering_data,
        )

        if loss_df.empty:
            print(f"  WARNING: No loss data for {split_type}")
            continue

        print(f"  Extracted {len(loss_df)} trajectory points")
        all_loss_data.append(loss_df)

        # Generate convergence-maxent heatmaps
        print("  Generating convergence-maxent heatmaps...")
        plot_convergence_maxent_heatmaps(
            df=loss_df,
            convergence_rates=config.convergence_rates,
            output_dir=str(output_dir),
            style=config.style,
            split_type=split_type,
        )

        # Generate model score heatmaps
        print("  Generating model score heatmaps...")
        plot_model_score_heatmaps(
            df=loss_df,
            convergence_rates=config.convergence_rates,
            output_dir=str(output_dir),
            style=config.style,
            split_type=split_type,
        )

        print("  Generating loss convergence 1D plots...")
        plot_loss_convergence(
            df=loss_df,
            convergence_rates=config.convergence_rates,
            output_dir=str(output_dir),
            style=config.style,
            split_type=split_type,
        )

        print("  Generating split variability plots...")
        plot_split_variability(
            df=loss_df,
            convergence_rates=config.convergence_rates,
            output_dir=str(output_dir),
            style=config.style,
            split_type=split_type,
        )

    # Combine all data and select best models
    if all_loss_data:
        print("\n" + "=" * 80)
        print("Combining data and selecting best models...")
        print("=" * 80)

        combined_df = pd.concat(all_loss_data, ignore_index=True)
        combined_csv_path = output_dir / "all_loss_trajectories.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"Saved combined loss trajectories: {len(combined_df)} points -> {combined_csv_path}")

        # Select best models
        print("\nSelecting best models across all splits...")
        best_models_df = select_best_models(combined_df)

        if not best_models_df.empty:
            best_models_csv_path = output_dir / "best_models.csv"
            best_models_df.to_csv(best_models_csv_path, index=False)
            print(f"Selected {len(best_models_df)} best models -> {best_models_csv_path}")

            # Plot best model comparisons
            print("\nGenerating best model comparison plots...")
            plot_best_model_comparisons(
                df=best_models_df,
                output_dir=str(output_dir),
                style=config.style,
            )
        else:
            print("WARNING: No best models selected")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
