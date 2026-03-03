"""
recovery_analysis_ISO_TRI_BI_precluster.py

Analyzes the recovery of open/closed states using pre-computed clusters.
Updated Analysis Script - Uses pre-computed clustering results from CSV files.

This script analyzes the %recovery of the ratio of the two conformations used in the IsoValidation process,
including analysis across different maxent regularization values.

Requirements:
    - Optimization results (results_EMA.hdf5)
    - Clustering results (_clustering_results/)
    - Data splits mapping (metadata)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/analysis/recovery_analysis_ISO_TRI_BI_precluster.py --results-dir ...
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir

EMA = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """
    Main function to run the complete analysis using pre-computed clustering results.
    """
    parser = argparse.ArgumentParser(
        description="IsoValidation analysis: set results/output dirs and EMA flag. Paths are interpreted relative to the script unless --absolute-paths is given."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory. If omitted, auto-discovered from config results_prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (relative to script dir by default). If omitted, derived from results-dir basename prefixed with '_analysis'.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Use EMA results (results_EMA.hdf5). Default: False",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret provided results/output directories as absolute paths",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to ../config.yaml relative to this script.",
    )
    args = parser.parse_args()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = script_dir.parent

    config_path = Path(args.config) if args.config else script_dir / ".." / "config.yaml"
    cfg = ExperimentConfig.from_yaml(config_path)

    convergence_rates = cfg.convergence_rates

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

    # clustering_dir stays as before (relative to script)
    clustering_dir = str(script_dir / ".." / "data" / "_clustering_results")
    ema_flag = args.ema

    # Determine output_dir:
    if args.output_dir:
        if args.absolute_paths:
            output_dir = args.output_dir
        else:
            output_dir = str((script_dir / args.output_dir).resolve())
    else:
        # Derive from results_dir basename, prefix with "_analysis"
        base_name = os.path.basename(os.path.normpath(results_dir))
        out_name = "_analysis" + base_name
        if args.absolute_paths:
            # Place output next to results_dir (same parent)
            parent = os.path.dirname(os.path.normpath(results_dir))
            output_dir = os.path.join(parent, out_name)
        else:
            # Place output inside script directory for relative mode
            output_dir = str(script_dir / out_name)

    # Show resolved paths and EMA flag
    print(f"Resolved results_dir: {results_dir}")
    print(f"Resolved clustering_dir: {clustering_dir}")
    print(f"Resolved output_dir: {output_dir}")
    print(f"EMA flag: {ema_flag}")

    # Check if required directories exist
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if not os.path.exists(clustering_dir):
        raise FileNotFoundError(
            f"Clustering directory not found: {clustering_dir}. "
            "Please run clustering_analysis.py first."
        )

    print("Starting Complete IsoValidation Analysis with MaxEnt Values...")
    print(f"Results directory: {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ensembles: {cfg.ensembles}")
    print(f"Loss functions: {cfg.loss_functions}")
    print(f"Number of splits: {cfg.num_splits}")
    print(f"Convergence rates: {convergence_rates}")
    print("-" * 60)

    # Load clustering results
    print("Loading clustering results...")
    clustering_results = loading.load_clustering_results(clustering_dir)

    # Load all optimization results with maxent
    print("Loading optimization results with maxent values...")
    results = loading.load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        num_splits=cfg.num_splits,
        EMA=ema_flag,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Part 1: KL Divergence Analysis with MaxEnt
    print("\n" + "=" * 60)
    print("PART 1: KL DIVERGENCE AND ESS ANALYSIS WITH MAXENT")
    print("=" * 60)

    # Extract KL divergences and ESS
    print("Extracting frame weights and calculating KL divergences and ESS...")
    kl_ess_df = analysis.extract_frame_weights_kl(results)

    if len(kl_ess_df) > 0:
        print(
            f"Extracted {len(kl_ess_df)} KL divergence and ESS data points from optimization histories"
        )

        # Save the KL divergence and ESS dataset
        kl_ess_df_path = os.path.join(output_dir, "kl_divergence_ess_analysis_maxent_data.csv")
        kl_ess_df.to_csv(kl_ess_df_path, index=False)
        print(f"KL divergence and ESS dataset saved to: {kl_ess_df_path}")

        # Rename step column to convergence_threshold_step for backward compat with inline plots
        if "step" in kl_ess_df.columns and "convergence_threshold_step" not in kl_ess_df.columns:
            kl_ess_df = kl_ess_df.rename(columns={"step": "convergence_threshold_step"})

        # Generate KL divergence plots
        print("Generating KL divergence heatmaps...")
        plotting.plot_metric_heatmap(
            kl_ess_df, "kl_divergence", convergence_rates, output_dir,
            cmap="Blues", columns_col="convergence_threshold_step"
        )

        print("Generating KL regularization strength plots...")
        plotting.plot_metric_vs_regularization_strength(
            kl_ess_df,
            "kl_divergence",
            "KL Divergence",
            convergence_rates,
            output_dir,
            "kl_divergence_vs_regularization_strength.png",
        )

        print("Generating KL maxent comparison plots...")
        plotting.plot_metric_maxent_comparison(
            kl_ess_df,
            "kl_divergence",
            "KL Divergence",
            output_dir,
            "kl_divergence_maxent_comparison.png",
        )

        # Generate ESS plots
        print("Generating ESS heatmaps...")
        plotting.plot_metric_heatmap(
            kl_ess_df, "effective_sample_size", convergence_rates, output_dir,
            cmap="viridis", columns_col="convergence_threshold_step"
        )

        print("Generating ESS regularization strength plots...")
        plotting.plot_metric_vs_regularization_strength(
            kl_ess_df,
            "effective_sample_size",
            "Effective Sample Size",
            convergence_rates,
            output_dir,
            "ess_vs_regularization_strength.png",
        )

        print("Generating ESS maxent comparison plots...")
        plotting.plot_metric_maxent_comparison(
            kl_ess_df,
            "effective_sample_size",
            "Effective Sample Size",
            output_dir,
            "ess_maxent_comparison.png",
        )

    else:
        print("No frame weights data found! Skipping KL divergence and ESS analysis.")
        kl_ess_df = pd.DataFrame()

    # Part 2: Conformational Recovery Analysis with MaxEnt
    print("\n" + "=" * 60)
    print("PART 2: CONFORMATIONAL RECOVERY ANALYSIS WITH MAXENT")
    print("=" * 60)

    if clustering_results:
        print("Analyzing conformational recovery with maxent values...")
        recovery_df = analysis.analyze_conformational_recovery(
            clustering_results,
            results,
            target_ratios=cfg.scoring.ground_truth_ratios,
            state_mapping=cfg.scoring.state_mapping,
            metric="ratio",
            best_step_only=False,
        )

        if len(recovery_df) > 0:
            print(f"Extracted {len(recovery_df)} conformational recovery data points")

            # Generate recovery plots
            print("Generating conformational recovery heatmaps...")
            plotting.plot_metric_heatmap(
                recovery_df, "open_recovery", convergence_rates, output_dir,
                cmap="PiYG", columns_col="convergence_step"
            )

            print("Generating regularization strength plots...")
            plotting.plot_recovery_vs_regularization_strength(recovery_df, convergence_rates, output_dir)

            print("Generating maxent comparison plots...")
            plotting.plot_maxent_comparison(recovery_df, output_dir)

            # Generate volcano plots if KL data is available
            if len(kl_ess_df) > 0:
                print("Generating volcano plot for KL divergence vs Open Recovery Fold Change...")
                plotting.plot_volcano_kl_recovery(
                    kl_ess_df,
                    recovery_df,
                    convergence_rates,
                    output_dir,
                    recovery_col="open_recovery",
                    baseline_col="open_ratio",
                    target_value=0.4,
                    target_label="Target\n(40:60)",
                    fold_change_col="open_recovery_fold_change",
                    current_col="current_open_ratio",
                    baseline_output_col="baseline_open_ratio",
                    xlabel_suffix="Open Recovery vs Unweighted Baseline",
                    title_keyword="Open Recovery",
                    xlim=(-6, 6),
                    save_target_csv=True,
                )

                print(
                    "Generating averaged volcano plot for KL divergence vs Open Recovery Fold Change..."
                )
                plotting.plot_volcano_kl_recovery_averaged(
                    kl_ess_df,
                    recovery_df,
                    convergence_rates,
                    output_dir,
                    recovery_col="open_recovery",
                    baseline_col="open_ratio",
                    target_value=0.4,
                    target_label="Target\n(40:60)",
                    fold_change_col="open_recovery_fold_change",
                    current_col="current_open_ratio",
                    baseline_output_col="baseline_open_ratio",
                    xlabel_suffix="Open Recovery vs Unweighted Baseline",
                    title_keyword="Open Recovery",
                )
            else:
                print("No KL divergence data available for volcano plots")

            # Save the recovery dataset
            recovery_df_path = os.path.join(output_dir, "conformational_recovery_maxent_data.csv")
            recovery_df.to_csv(recovery_df_path, index=False)
            print(f"Conformational recovery dataset saved to: {recovery_df_path}")

            # Print summary statistics
            print("\nConformational Recovery Summary with MaxEnt:")
            print("-" * 40)

            # Summary by maxent value
            maxent_summary = (
                recovery_df[recovery_df["loss_function"] != "Original"]
                .groupby(["split_type", "ensemble", "loss_function", "maxent_value"])
                .last()
                .reset_index()
            )

            for split_type in maxent_summary["split_type"].unique():
                print(f"\nSplit Type: {split_type}")
                split_summary = maxent_summary[maxent_summary["split_type"] == split_type]

                for _, row in split_summary.iterrows():
                    open_ratio_str = (
                        f", Open Ratio = {row['open_ratio']:.3f}"
                        if "open_ratio" in row.index and pd.notna(row.get("open_ratio"))
                        else ""
                    )
                    print(
                        f"  {row['ensemble']} - {row['loss_function']} - MaxEnt {row['maxent_value']:.0f}: "
                        f"Open Recovery = {row['open_recovery']:.1f}%"
                        f"{open_ratio_str}"
                    )
        else:
            print("No conformational recovery data generated!")
    else:
        print("No clustering results available. Skipping conformational recovery analysis.")

    print("\n" + "=" * 60)
    print("ANALYSIS WITH MAXENT VALUES COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
