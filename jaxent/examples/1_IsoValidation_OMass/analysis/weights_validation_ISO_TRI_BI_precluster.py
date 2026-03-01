"""
weights_validation_ISO_TRI_BI_precluster.py  (Exp1 — IsoValidation_OMass)

Validates optimized frame weights: KL divergence, ESS, split consistency,
and sequential MaxEnt KLD.  Generates weight distribution and KLD plots.

Requirements:
    - Optimization results (results.hdf5 / results_EMA.hdf5)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/analysis/weights_validation_ISO_TRI_BI_precluster.py \
        [--results-dir <dir>] [--output-dir <dir>] [--ema]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.examples.common import analysis, loading, plotting
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.paths import find_most_recent_dir

# Publication-ready style
sns.set_style("ticks")
sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    },
)

SPLIT_NAME_MAPPING = {
    "R3": "Non-Redundant",
    "Sp": "Spatial",
    "r": "Random",
    "s": "Stratified",
}


def _create_ensemble_loss_key(ensemble, loss_function):
    return f"{ensemble}_{loss_function}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Exp1 weights validation: KL/ESS analysis with MaxEnt sweep."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory. If omitted, auto-discovered from config results_prefix.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, derived from results-dir basename.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        default=False,
        help="Use EMA results (results_EMA.hdf5).",
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

    os.makedirs(output_dir, exist_ok=True)

    print(f"Results directory : {results_dir}")
    print(f"Output directory  : {output_dir}")
    print(f"EMA               : {args.ema}")
    print(f"Ensembles         : {cfg.ensembles}")
    print(f"Loss functions    : {cfg.loss_functions}")
    print(f"Num splits        : {cfg.num_splits}")
    print("-" * 60)

    # Load all optimization results
    print("\nLoading optimization results with MaxEnt values...")
    results = loading.load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        num_splits=cfg.num_splits,
        EMA=args.ema,
    )

    # Part 1: KL divergence and ESS at every step
    print("\n" + "=" * 60)
    print("PART 1: KL DIVERGENCE AND ESS ANALYSIS WITH MAXENT")
    print("=" * 60)

    kl_ess_df = analysis.extract_frame_weights_kl(results)

    if not kl_ess_df.empty:
        print(f"Extracted {len(kl_ess_df)} KL divergence / ESS data points")
        kl_ess_path = os.path.join(output_dir, "kl_divergence_ess_analysis_maxent_data.csv")
        kl_ess_df.to_csv(kl_ess_path, index=False)
        print(f"KL/ESS data saved to: {kl_ess_path}")

    # Extract final weights (DataFrame with 'weights' column)
    final_df = analysis.extract_final_weights(results)
    print(f"\nExtracted {len(final_df)} final weight distributions")

    # Extract weights over convergence steps for panel plots
    conv_df = analysis.extract_weights_over_convergence_steps(results)
    print(f"Extracted {len(conv_df)} convergence-step weight distributions")

    # Pairwise KLD between splits (final weights, no per-step grouping)
    kld_df = pd.DataFrame()
    if not final_df.empty:
        kld_df = analysis.compute_pairwise_kld_between_splits(final_df, per_step=False)
        if not kld_df.empty:
            kld_path = os.path.join(output_dir, "kld_between_splits_data.csv")
            kld_df.to_csv(kld_path, index=False)
            print(f"KLD between splits data saved to: {kld_path}")

    # Sequential MaxEnt KLD
    seq_kld_df = pd.DataFrame()
    if not final_df.empty:
        seq_kld_df = analysis.compute_sequential_maxent_kld(final_df)
        if not seq_kld_df.empty:
            seq_kld_path = os.path.join(output_dir, "sequential_maxent_kld_data.csv")
            seq_kld_df.to_csv(seq_kld_path, index=False)
            print(f"Sequential MaxEnt KLD data saved to: {seq_kld_path}")

    # Plots
    print("\n" + "=" * 60)
    print("CREATING PLOTS")
    print("=" * 60)

    if not final_df.empty:
        plotting.plot_weight_distribution_lines(final_df, output_dir, SPLIT_NAME_MAPPING)
    else:
        print("No final weights data for distribution plots")

    if not kld_df.empty:
        plotting.plot_kld_between_splits(kld_df, output_dir, SPLIT_NAME_MAPPING)
    else:
        print("No KLD data for between-splits plot")

    if not seq_kld_df.empty:
        plotting.plot_sequential_maxent_kld(seq_kld_df, output_dir, SPLIT_NAME_MAPPING)
    else:
        print("No sequential KLD data for sequential plot")

    if not conv_df.empty:
        plotting.plot_weight_distribution_maxent_panels(conv_df, output_dir, SPLIT_NAME_MAPPING)
        plotting.plot_weight_distribution_convergence_panels(conv_df, output_dir, SPLIT_NAME_MAPPING)
    else:
        print("No convergence weights data for panel plots")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
