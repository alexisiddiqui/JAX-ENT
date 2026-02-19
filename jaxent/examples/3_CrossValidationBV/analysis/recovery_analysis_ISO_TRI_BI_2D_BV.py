"""
2D Hyperparameter Sweep Recovery Analysis for MoPrP System

Function:
Analyzes conformational state recovery across a 2D grid of MaxEnt and BV regularization scalings:
- Selects best model per grid point based on validation loss.
- Computes Jensen-Shannon Divergence (JSD) between learned and target state populations.
- Generates 2D heatmaps and 1D slices of recovery percentages.

Requirements:
- `--results-dir`: Directory containing `results.hdf5` files from 2D sweep.
- `--clustering-dir`: Directory with `frame_to_cluster.csv` files.
- `--state-ratios-json`: JSON file with target state ratios.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Add base directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir

BV_REG_FUNCTIONS = ["L1", "L2"]


# ---------------------------------------------------------------------------
# Script-specific inline plotting functions (no common equivalent)
# ---------------------------------------------------------------------------


def plot_best_hyperparameters(
    recovery_df: pd.DataFrame,
    output_dir: str,
    metric: str = "recovery_percent",
):
    """
    Plot a summary showing the best (maxent, bv_reg) combination for each loss-reg pairing.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    df = recovery_df[recovery_df["loss_function"] != "Original"].copy()

    if len(df) == 0:
        print("No data available for best hyperparameters plot")
        return

    split_types = sorted(df["split_type"].unique())

    for split_type in split_types:
        print(f"Creating best hyperparameters summary for {split_type}...")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df[df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())

        for ensemble in ensembles:
            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            loss_functions = sorted(ensemble_df["loss_function"].unique())
            bv_reg_functions = sorted(ensemble_df["bv_reg_function"].unique())

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"Best Hyperparameters by {metric} - {ensemble} - {split_type}",
                fontsize=16,
                fontweight="bold",
            )

            for i, bv_reg_fn in enumerate(bv_reg_functions):
                for j, loss_fn in enumerate(loss_functions):
                    ax = axes[i, j]

                    combo_df = ensemble_df[
                        (ensemble_df["loss_function"] == loss_fn)
                        & (ensemble_df["bv_reg_function"] == bv_reg_fn)
                    ]

                    if len(combo_df) > 0:
                        # Find best combination
                        best_idx = combo_df[metric].idxmax()
                        best_row = combo_df.loc[best_idx]

                        # Create scatter plot with best highlighted
                        maxent_vals = combo_df["maxent_value"].unique()
                        bvreg_vals = combo_df["bv_reg_value"].unique()

                        for maxent_val in maxent_vals:
                            for bvreg_val in bvreg_vals:
                                subset = combo_df[
                                    (combo_df["maxent_value"] == maxent_val)
                                    & (combo_df["bv_reg_value"] == bvreg_val)
                                ]
                                if len(subset) > 0:
                                    avg_metric = subset[metric].mean()
                                    color = "red" if (
                                        maxent_val == best_row["maxent_value"]
                                        and bvreg_val == best_row["bv_reg_value"]
                                    ) else "blue"
                                    size = 200 if color == "red" else 100
                                    marker = "*" if color == "red" else "o"

                                    ax.scatter(
                                        maxent_val,
                                        bvreg_val,
                                        s=size,
                                        c=color,
                                        marker=marker,
                                        alpha=0.7,
                                        edgecolors="black",
                                        linewidth=1,
                                    )

                                    # Annotate with value
                                    ax.text(
                                        maxent_val,
                                        bvreg_val,
                                        f"{avg_metric:.1f}%",
                                        ha="center",
                                        va="center",
                                        fontsize=8,
                                    )

                        ax.set_xscale("log")
                        ax.set_xlabel("MaxEnt Value (log scale)")
                        ax.set_ylabel("BV Reg Value")
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}\nBest: MaxEnt={best_row['maxent_value']:.1f}, "
                                    f"BVReg={best_row['bv_reg_value']:.2f}")
                        ax.grid(True, alpha=0.3)

                        # Add legend
                        red_patch = mpatches.Patch(color="red", label="Best")
                        blue_patch = mpatches.Patch(color="blue", label="Other")
                        ax.legend(handles=[red_patch, blue_patch], loc="best")
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}")

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, f"best_hyperparameters_{ensemble}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main function for 2D hyperparameter sweep recovery analysis."""
    parser = argparse.ArgumentParser(
        description="2D hyperparameter sweep recovery analysis for MoPrP."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory",
    )
    parser.add_argument(
        "--clustering-dir",
        default=None,
        help="Clustering directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, derived from results-dir.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Use EMA results",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to ../config.yaml relative to this script.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret --results-dir / --output-dir as absolute paths.",
    )
    args = parser.parse_args()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = script_dir.parent

    config_path = Path(args.config) if args.config else script_dir / ".." / "config.yaml"
    cfg = ExperimentConfig.from_yaml(config_path)

    state_mapping = cfg.scoring.state_mapping

    # Resolve results directory
    if args.results_dir:
        results_dir = (
            Path(args.results_dir)
            if args.absolute_paths
            else (script_dir / args.results_dir).resolve()
        )
    elif cfg.results_prefix:
        prefix_path = Path(cfg.results_prefix)
        base_path = exp_dir / prefix_path.parent
        prefix = prefix_path.name
        found = find_most_recent_dir(base_path, prefix)
        if found is None:
            raise FileNotFoundError(
                f"No directory matching prefix '{prefix}' found in {base_path}"
            )
        results_dir = found
    elif cfg.results_dir:
        results_dir = exp_dir / cfg.results_dir
    else:
        raise ValueError("Provide --results-dir or set results_prefix/results_dir in config.")

    results_dir = str(results_dir)

    # Resolve clustering directory
    if args.clustering_dir:
        clustering_dir = (
            args.clustering_dir
            if args.absolute_paths
            else str((script_dir / args.clustering_dir).resolve())
        )
    elif cfg.clustering_dir:
        clustering_dir = str(exp_dir / cfg.clustering_dir)
    else:
        raise ValueError("Provide --clustering-dir or set clustering_dir in config.")

    if args.output_dir:
        output_dir = (
            args.output_dir
            if args.absolute_paths
            else str((script_dir / args.output_dir).resolve())
        )
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = str(script_dir / ("_analysis_" + base_name))

    print(f"Results directory: {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory: {output_dir}")
    print(f"EMA flag: {args.ema}")

    # Load target ratios
    state_ratios_path = exp_dir / cfg.state_ratios_json
    try:
        with open(state_ratios_path) as f:
            ratios_data = json.load(f)
        target_ratios = {
            "Folded": ratios_data["fractional_populations"]["folded"]["fraction"],
            "PUF1":   ratios_data["fractional_populations"]["PUF1"]["fraction"],
            "PUF2":   ratios_data["fractional_populations"]["PUF2"]["fraction"],
            "unfolded": 0,
        }
        print("\nTarget state ratios:")
        for state, ratio in target_ratios.items():
            print(f"  {state}: {ratio:.3f}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading state ratios: {e}")
        return

    # Load clustering data
    print("\nLoading clustering data...")
    clustering_data = loading.load_clustering_results(
        clustering_dir, cfg.scoring.ensemble_clustering_map
    )

    if not clustering_data:
        print("Error: Could not load clustering data.")
        return

    # Load optimization results
    print("\nLoading optimization results...")
    results = loading.load_all_optimization_results_2d(
        results_dir=results_dir,
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        bv_reg_functions=BV_REG_FUNCTIONS,
        num_splits=cfg.num_splits,
        EMA=args.ema,
        verbose=True,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Analyze recovery
    print("\n" + "=" * 60)
    print("ANALYZING CONFORMATIONAL RECOVERY - 2D SWEEP")
    print("=" * 60)

    recovery_df = analysis.analyze_conformational_recovery(
        clustering_data, results, target_ratios, state_mapping, metric="jsd", best_step_only=True
    )

    if len(recovery_df) > 0:
        print(f"\nExtracted {len(recovery_df)} recovery data points")

        # Save recovery data
        recovery_path = os.path.join(output_dir, "recovery_2d_sweep_data.csv")
        recovery_df.to_csv(recovery_path, index=False)
        print(f"Recovery data saved to: {recovery_path}")

        # Generate 2D heatmaps
        print("\nGenerating 2D heatmaps...")
        plotting.plot_2d_heatmaps_grid(
            recovery_df,
            output_dir,
            metric="recovery_percent",
            metric_label="Recovery (%)",
        )

        # Generate 1D slices
        print("\nGenerating 1D slices...")
        plotting.plot_1d_slices_2d_sweep(
            recovery_df,
            output_dir,
            metric="recovery_percent",
            metric_label="Recovery (%)",
        )

        # Generate best hyperparameters summary
        print("\nGenerating best hyperparameters summary...")
        plot_best_hyperparameters(recovery_df, output_dir, metric="recovery_percent")

        # Print summary
        print("\n" + "=" * 60)
        print("RECOVERY SUMMARY - 2D SWEEP")
        print("=" * 60)

        for ensemble in cfg.ensembles:
            print(f"\n{ensemble}:")
            ensemble_data = recovery_df[recovery_df["ensemble"] == ensemble]

            # Original
            orig_data = ensemble_data[ensemble_data["loss_function"] == "Original"]
            if len(orig_data) > 0:
                orig = orig_data.iloc[0]
                print(f"  Original (unweighted): Recovery = {orig['recovery_percent']:.1f}%")

            # Best optimized
            optimized = ensemble_data[ensemble_data["loss_function"] != "Original"]
            if len(optimized) > 0:
                best = optimized.loc[optimized["recovery_percent"].idxmax()]
                print(
                    f"  Best optimized: {best['loss_function']} + {best['bv_reg_function']}, "
                    f"MaxEnt={best['maxent_value']:.1f}, BVReg={best['bv_reg_value']:.2f}"
                )
                print(f"    Recovery = {best['recovery_percent']:.1f}%")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
