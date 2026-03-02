"""
[Script Name] recovery_analysis_ISO_TRI_BI_precluster.py

[Brief Description of Functionality]
Analyzes the recovery of conformational states (Folded, PUF1, PUF2, Unfolded) for the MoPrP system
using Jensen-Shannon Divergence (JSD). It uses the optimized frame weights to calculate reweighted
state populations and compares them to target ratios.
Generates heatmaps of recovery percentage and volcano plots showing the trade-off between
recovery and fitting error.

Requirements:
    - Optimization results (HDF5 files).
    - Cluster assignments CSV.
    - `state_ratios.json`.

Usage:
    python jaxent/examples/2_CrossValidation/analysis/recovery_analysis_ISO_TRI_BI_precluster.py \
        --results_dir ... --cluster_assignments ...

Output:
    - Recovery heatmaps, volcano plots, and summary CSVs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add the base directory to the path to import the HDF5 utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Main function to run the MoPrP recovery analysis."""
    parser = argparse.ArgumentParser(description="MoPrP recovery analysis with JSD-based metrics.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory",
    )
    parser.add_argument(
        "--clustering-dir",
        default=None,
        help="Clustering directory containing AF2_MSAss and AF2_Filtered subdirectories",
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

    convergence_rates = cfg.convergence_rates

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

    # Resolve output directory
    if args.output_dir:
        output_dir = (
            args.output_dir
            if args.absolute_paths
            else str((script_dir / args.output_dir).resolve())
        )
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = str(script_dir / f"_analysis{base_name}")

    print(f"Results directory: {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory: {output_dir}")
    print(f"EMA flag: {args.ema}")

    # Load target ratios from state_ratios.json
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

    # Load clustering data for each ensemble
    print("\nLoading clustering data...")
    clustering_data = loading.load_clustering_results(
        clustering_dir, cfg.scoring.ensemble_clustering_map
    )

    if not clustering_data:
        print("Error: Could not load clustering data.")
        return

    # Load optimization results
    print("\nLoading optimization results...")
    results = loading.load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        num_splits=cfg.num_splits,
        EMA=args.ema,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Analyze recovery
    print("\n" + "=" * 60)
    print("ANALYZING CONFORMATIONAL RECOVERY WITH JSD")
    print("=" * 60)

    recovery_df = analysis.analyze_conformational_recovery(
        clustering_data, results, target_ratios, state_mapping, metric="jsd", best_step_only=False
    )

    if len(recovery_df) > 0:
        print(f"\nExtracted {len(recovery_df)} recovery data points")

        # Save recovery data
        recovery_path = os.path.join(output_dir, "recovery_jsd_data.csv")
        recovery_df.to_csv(recovery_path, index=False)
        print(f"Recovery data saved to: {recovery_path}")

        # Generate recovery plots
        print("\nGenerating recovery plots...")
        plotting.plot_metric_heatmap(
            recovery_df, "recovery_percent", convergence_rates, output_dir,
            cmap="RdYlGn", columns_col="convergence_step"
        )

        # Extract KL divergence and ESS
        print("\n" + "=" * 60)
        print("EXTRACTING KL DIVERGENCE AND ESS")
        print("=" * 60)

        kl_ess_df = analysis.extract_frame_weights_kl(results)

        if len(kl_ess_df) > 0:
            print(f"\nExtracted {len(kl_ess_df)} KL divergence and ESS data points")

            # Save KL/ESS data
            kl_ess_path = os.path.join(output_dir, "kl_divergence_ess_data.csv")
            kl_ess_df.to_csv(kl_ess_path, index=False)
            print(f"KL divergence and ESS data saved to: {kl_ess_path}")

            # Rename step column for backward compat with inline volcano plots
            if "step" in kl_ess_df.columns and "convergence_threshold_step" not in kl_ess_df.columns:
                kl_ess_df = kl_ess_df.rename(columns={"step": "convergence_threshold_step"})

            # Generate volcano plots
            print("\nGenerating volcano plots...")
            plotting.plot_volcano_kl_recovery(kl_ess_df, recovery_df, convergence_rates, output_dir)

            print("\nGenerating averaged volcano plots...")
            plotting.plot_volcano_kl_recovery_averaged(kl_ess_df, recovery_df, convergence_rates, output_dir)
        else:
            print("No KL divergence data available, skipping volcano plots")

        # Print summary
        print("\n" + "=" * 60)
        print("RECOVERY SUMMARY")
        print("=" * 60)

        for ensemble in cfg.ensembles:
            print(f"\n{ensemble}:")
            ensemble_data = recovery_df[recovery_df["ensemble"] == ensemble]

            # Original (unweighted)
            orig_data = ensemble_data[ensemble_data["loss_function"] == "Original"]
            if len(orig_data) > 0:
                orig = orig_data.iloc[0]
                print("  Original (unweighted):")
                print(f"    Recovery: {orig['recovery_percent']:.1f}%")
                print(f"    JS Distance: {orig['js_distance']:.4f}")
                for state in target_ratios:
                    print(
                        f"    {state}: {orig[f'{state}_current']:.3f} (target: {orig[f'{state}_target']:.3f})"
                    )

            # Final optimized
            optimized = ensemble_data[
                (ensemble_data["loss_function"] != "Original")
                & (ensemble_data["convergence_step"] == ensemble_data["convergence_step"].max())
            ]
            if len(optimized) > 0:
                print("\n  Optimized (final convergence):")
                for _, row in optimized.iterrows():
                    print(f"    {row['loss_function']} (maxent={row['maxent_value']:.1f}):")
                    print(f"      Recovery: {row['recovery_percent']:.1f}%")
                    print(f"      JS Distance: {row['js_distance']:.4f}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
