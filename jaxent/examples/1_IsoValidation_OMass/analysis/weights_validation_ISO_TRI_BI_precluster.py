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
# Script-specific inline plotting functions (no common equivalent)
# ---------------------------------------------------------------------------


def plot_kld_between_splits(kld_df, output_dir):
    """Plot mean KLD between splits across maxent values as line plots."""
    print("Creating KLD between splits plot...")

    if kld_df.empty:
        print("  No KLD data available for plotting.")
        return

    kld_df = kld_df.copy()
    kld_df["ensemble_loss"] = kld_df.apply(
        lambda row: _create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    available_ensemble_loss = kld_df["ensemble_loss"].unique()
    n_combinations = len(available_ensemble_loss)
    n_cols = min(n_combinations, 2)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, ensemble_loss in enumerate(available_ensemble_loss):
        if idx >= len(axes):
            break
        ax = axes[idx]
        el_data = kld_df[kld_df["ensemble_loss"] == ensemble_loss]

        if el_data.empty:
            ax.set_visible(False)
            continue

        available_split_types = el_data["split_type"].unique()
        n_st = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_st))
        color_map = dict(zip(available_split_types, colors))

        for split_type in available_split_types:
            st_data = el_data[el_data["split_type"] == split_type].sort_values("maxent_value")
            color = color_map[split_type]
            label = SPLIT_NAME_MAPPING.get(split_type, split_type)
            ax.errorbar(
                st_data["maxent_value"],
                st_data["mean_kld_between_splits"],
                yerr=st_data["sem_kld_between_splits"],
                color=color,
                alpha=0.8,
                label=label,
                linewidth=2,
                marker="o",
                markersize=4,
                capsize=3,
            )

        ax.set_xscale("log")
        ax.set_xlabel("MaxEnt Value")
        ax.set_ylabel("Mean KLD Between Splits")
        ax.set_title(ensemble_loss)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(len(available_ensemble_loss), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("KL Divergence Between Splits Across MaxEnt Values", fontsize=16)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "kld_between_splits_vs_maxent.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: kld_between_splits_vs_maxent.png")
    plt.close()


def plot_sequential_maxent_kld(sequential_kld_df, output_dir):
    """Plot KLD between sequential maxent values."""
    print("Creating sequential maxent KLD plot...")

    if sequential_kld_df.empty:
        print("  No sequential KLD data available for plotting.")
        return

    seq_df = sequential_kld_df.copy()
    seq_df["ensemble_loss"] = seq_df.apply(
        lambda row: _create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    available_ensemble_loss = seq_df["ensemble_loss"].unique()
    n_combinations = len(available_ensemble_loss)
    n_cols = min(n_combinations, 2)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_combinations == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, ensemble_loss in enumerate(available_ensemble_loss):
        if idx >= len(axes):
            break
        ax = axes[idx]
        el_data = seq_df[seq_df["ensemble_loss"] == ensemble_loss]

        if el_data.empty:
            ax.set_visible(False)
            continue

        available_split_types = el_data["split_type"].unique()
        n_st = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_st))
        color_map = dict(zip(available_split_types, colors))

        for split_type in available_split_types:
            st_data = el_data[el_data["split_type"] == split_type]
            color = color_map[split_type]
            label = SPLIT_NAME_MAPPING.get(split_type, split_type)

            for split_idx in st_data["split_idx"].unique():
                si_data = st_data[st_data["split_idx"] == split_idx].sort_values("current_maxent")
                if len(si_data) > 0:
                    ax.plot(
                        si_data["current_maxent"],
                        si_data["kld_to_previous"],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                        marker=".",
                        markersize=2,
                    )

            maxent_stats = (
                st_data.groupby("current_maxent")["kld_to_previous"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            if len(maxent_stats) > 0:
                ax.errorbar(
                    maxent_stats["current_maxent"],
                    maxent_stats["mean"],
                    yerr=maxent_stats["std"] / np.sqrt(maxent_stats["count"]),
                    color=color,
                    alpha=0.8,
                    label=label,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    capsize=3,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Current MaxEnt")
        ax.set_ylabel("KLD to Previous MaxEnt (or Uniform)")
        ax.set_title(ensemble_loss)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(len(available_ensemble_loss), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("KL Divergence Between Sequential MaxEnt Values", fontsize=16)
    plt.tight_layout()
    filepath = os.path.join(output_dir, "sequential_maxent_kld_vs_maxent.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: sequential_maxent_kld_vs_maxent.png")
    plt.close()


def plot_weight_distribution_maxent_panels(conv_weights_df, output_dir):
    """Weight distributions with maxent values as panels and convergence as lines."""
    print("Creating weight distribution plots with maxent panels...")

    if conv_weights_df is None or len(conv_weights_df) == 0:
        print("  No convergence weights data available for plotting")
        return

    weights_df = (
        pd.DataFrame(conv_weights_df)
        if not isinstance(conv_weights_df, pd.DataFrame)
        else conv_weights_df.copy()
    )
    weights_df["ensemble_loss"] = weights_df.apply(
        lambda row: _create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    weight_bins = np.logspace(-50, 0, 50)
    bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

    for ensemble_loss in weights_df["ensemble_loss"].unique():
        el_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if el_data.empty:
            continue

        for split_type in sorted(el_data["split_type"].unique()):
            split_data = el_data[el_data["split_type"] == split_type]
            if split_data.empty:
                continue

            maxent_values = sorted(split_data["maxent_value"].unique())
            if len(maxent_values) < 2:
                continue

            n_maxent = len(maxent_values)
            n_cols = min(4, n_maxent)
            n_rows = (n_maxent + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes_flat = np.array(axes).flatten() if n_maxent > 1 else [axes]

            for idx, maxent_val in enumerate(maxent_values):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]
                maxent_data = split_data[split_data["maxent_value"] == maxent_val]
                if maxent_data.empty:
                    ax.set_visible(False)
                    continue

                conv_groups = {}
                for _, row in maxent_data.iterrows():
                    cf = row["convergence_fraction"]
                    if cf not in conv_groups:
                        conv_groups[cf] = []
                    conv_groups[cf].append(row["weights"])

                conv_fractions = sorted(conv_groups.keys())
                colors = plt.cm.plasma(np.linspace(0, 1, len(conv_fractions)))

                for conv_frac, color in zip(conv_fractions, colors):
                    hist_counts = []
                    for w in conv_groups[conv_frac]:
                        if len(w) > 0 and np.sum(w) > 0:
                            counts, _ = np.histogram(w, bins=weight_bins, density=True)
                            hist_counts.append(counts)
                    if hist_counts:
                        mean_counts = np.mean(hist_counts, axis=0)
                        std_counts = (
                            np.std(hist_counts, axis=0)
                            if len(hist_counts) > 1
                            else np.zeros_like(mean_counts)
                        )
                        ax.plot(
                            bin_centers, mean_counts,
                            color=color, alpha=0.8,
                            label=f"{conv_frac:.1%}", linewidth=2,
                        )
                        if len(hist_counts) > 1:
                            ax.fill_between(
                                bin_centers,
                                mean_counts - std_counts,
                                mean_counts + std_counts,
                                color=color, alpha=0.2,
                            )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Density")
                ax.set_title(f"MaxEnt = {maxent_val:.0e}")
                ax.legend(title="Convergence", fontsize=8, title_fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(len(maxent_values), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            split_name = SPLIT_NAME_MAPPING.get(split_type, split_type)
            plt.suptitle(
                f"Weight Evolution Over Convergence — {ensemble_loss} — {split_name}",
                fontsize=16, y=0.98,
            )
            plt.tight_layout()
            fname = (
                f"weight_distributions_maxent_panels_{ensemble_loss}_{split_type}"
                .replace("/", "_").replace(" ", "_") + ".png"
            )
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
            print(f"  Saved: {fname}")
            plt.close()


def plot_weight_distribution_convergence_panels(conv_weights_df, output_dir):
    """Weight distributions with convergence fractions as panels and maxent as lines."""
    print("Creating weight distribution plots with convergence panels...")

    if conv_weights_df is None or len(conv_weights_df) == 0:
        print("  No convergence weights data available for plotting")
        return

    weights_df = (
        pd.DataFrame(conv_weights_df)
        if not isinstance(conv_weights_df, pd.DataFrame)
        else conv_weights_df.copy()
    )
    weights_df["ensemble_loss"] = weights_df.apply(
        lambda row: _create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    weight_bins = np.logspace(-50, 0, 50)
    bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

    for ensemble_loss in weights_df["ensemble_loss"].unique():
        el_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if el_data.empty:
            continue

        for split_type in sorted(el_data["split_type"].unique()):
            split_data = el_data[el_data["split_type"] == split_type]
            if split_data.empty:
                continue

            conv_fractions = sorted(split_data["convergence_fraction"].unique())
            if len(conv_fractions) < 2:
                continue

            n_conv = len(conv_fractions)
            n_cols = min(3, n_conv)
            n_rows = (n_conv + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes_flat = np.array(axes).flatten() if n_conv > 1 else [axes]

            for idx, conv_frac in enumerate(conv_fractions):
                if idx >= len(axes_flat):
                    break
                ax = axes_flat[idx]
                conv_data = split_data[split_data["convergence_fraction"] == conv_frac]
                if conv_data.empty:
                    ax.set_visible(False)
                    continue

                maxent_groups = {}
                for _, row in conv_data.iterrows():
                    maxent = row["maxent_value"]
                    if maxent not in maxent_groups:
                        maxent_groups[maxent] = []
                    maxent_groups[maxent].append(row["weights"])

                maxent_values = sorted(maxent_groups.keys())
                colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))

                for maxent_val, color in zip(maxent_values, colors):
                    hist_counts = []
                    for w in maxent_groups[maxent_val]:
                        if len(w) > 0 and np.sum(w) > 0:
                            counts, _ = np.histogram(w, bins=weight_bins, density=True)
                            hist_counts.append(counts)
                    if hist_counts:
                        mean_counts = np.mean(hist_counts, axis=0)
                        ax.plot(
                            bin_centers, mean_counts,
                            color=color, alpha=0.8,
                            label=f"{maxent_val:.0e}", linewidth=2,
                        )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Density")
                ax.set_title(f"Convergence: {conv_frac:.1%}")
                ax.legend(title="MaxEnt", fontsize=8, title_fontsize=8)
                ax.grid(True, alpha=0.3)

            for idx in range(len(conv_fractions), len(axes_flat)):
                axes_flat[idx].set_visible(False)

            split_name = SPLIT_NAME_MAPPING.get(split_type, split_type)
            plt.suptitle(
                f"MaxEnt Comparison Across Convergence — {ensemble_loss} — {split_name}",
                fontsize=16, y=0.98,
            )
            plt.tight_layout()
            fname = (
                f"weight_distributions_convergence_panels_{ensemble_loss}_{split_type}"
                .replace("/", "_").replace(" ", "_") + ".png"
            )
            plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
            print(f"  Saved: {fname}")
            plt.close()


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
        plot_kld_between_splits(kld_df, output_dir)
    else:
        print("No KLD data for between-splits plot")

    if not seq_kld_df.empty:
        plot_sequential_maxent_kld(seq_kld_df, output_dir)
    else:
        print("No sequential KLD data for sequential plot")

    if not conv_df.empty:
        plot_weight_distribution_maxent_panels(conv_df, output_dir)
        plot_weight_distribution_convergence_panels(conv_df, output_dir)
    else:
        print("No convergence weights data for panel plots")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
