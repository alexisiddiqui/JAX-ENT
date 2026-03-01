"""
weights_validation_ISO_TRI_BI_precluster.py  (Exp2 — CrossValidation)

Validates optimized frame weights via KL divergence, ESS, split consistency,
and conformational state recovery.  Generates heatmaps and distribution plots.

Requirements:
    - Optimization results (results.hdf5 / results_EMA.hdf5)
    - Clustering results (clustering_dir)

Usage:
    python jaxent/examples/2_CrossValidation/analysis/weights_validation_ISO_TRI_BI_precluster.py \
        [--results-dir <dir>] [--clustering-dir <dir>] [--output-dir <dir>] [--ema]
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

# Ensemble → clustering file mapping
ENSEMBLE_CLUSTERING_MAP = {
    "AF2_MSAss": "AF2_MSAss_frame_to_cluster.csv",
    "AF2_filtered": "AF2_Filtered_frame_to_cluster.csv",
}

# Cluster id → conformational state
STATE_MAPPING = {
    0: "Folded",
    1: "PUF1",
    2: "PUF2",
    4: "unfolded",
}

split_type_colours = {
    "random": "#9467bd",
}

split_name_mapping = {
    "random": "Random",
}

SPLIT_NAME_MAPPING = {
    "r": "Random",
    "s": "Sequence",
    "R3": "Non-Redundant",
    "sequence_cluster": "Non-Redundant",
    "Sp": "Spatial",
    "spatial": "Spatial",
    "random": "Random",
}


def _create_ensemble_loss_key(ensemble, loss_function):
    return f"{ensemble}_{loss_function}"


# ---------------------------------------------------------------------------
# Script-specific inline functions (no common equivalent)
# ---------------------------------------------------------------------------


def calculate_conformational_recovery(weights_df, clustering_dir):
    """Calculate weighted conformational state proportions for each optimization result.

    Parameters
    ----------
    weights_df:
        DataFrame from ``analysis.extract_final_weights`` (must have 'weights' column).
    clustering_dir:
        Directory containing per-ensemble clustering subdirectories.

    Returns
    -------
    DataFrame with state proportion columns per run.
    """
    print("Calculating conformational recovery...")
    recovery_data = []

    for _, row in weights_df.iterrows():
        ensemble = row["ensemble"]
        weights = row["weights"]

        if ensemble not in ENSEMBLE_CLUSTERING_MAP:
            print(f"  Skipping {ensemble} — no clustering file mapped")
            continue

        clustering_file = ENSEMBLE_CLUSTERING_MAP[ensemble]
        ensemble_subdir = clustering_file.replace("_frame_to_cluster.csv", "")
        clustering_path = os.path.join(clustering_dir, ensemble_subdir, clustering_file)

        if not os.path.exists(clustering_path):
            print(f"  Warning: Clustering file not found: {clustering_path}")
            continue

        cluster_df = pd.read_csv(clustering_path)

        if len(weights) != len(cluster_df):
            print(
                f"  Warning: Length mismatch for {ensemble} — "
                f"weights:{len(weights)}, clusters:{len(cluster_df)}"
            )
            continue

        cluster_assignments = cluster_df["cluster_label"].values

        cluster_proportions = {}
        for cluster_id in np.unique(cluster_assignments):
            mask = cluster_assignments == cluster_id
            cluster_proportions[cluster_id] = float(np.sum(weights[mask]))

        state_proportions = {}
        for cluster_id, state_name in STATE_MAPPING.items():
            if cluster_id in cluster_proportions:
                state_proportions[state_name] = (
                    state_proportions.get(state_name, 0.0) + cluster_proportions[cluster_id]
                )

        recovery_dict = {
            "ensemble": ensemble,
            "split_type": row["split_type"],
            "split": row["split"],
            "loss_function": row["loss_function"],
            "maxent_value": row["maxent_value"],
        }
        for state_name in STATE_MAPPING.values():
            recovery_dict[f"{state_name}_proportion"] = state_proportions.get(state_name, 0.0)

        recovery_data.append(recovery_dict)

    return pd.DataFrame(recovery_data)


def plot_conformational_recovery_scatter(recovery_df, output_dir):
    """Scatter plots of conformational state recovery vs maxent values."""
    print("Creating conformational recovery scatter plots...")

    if recovery_df.empty:
        print("  No recovery data available for plotting")
        return

    recovery_df = recovery_df.copy()
    recovery_df["ensemble_loss"] = recovery_df.apply(
        lambda row: _create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    state_names = [
        col.replace("_proportion", "")
        for col in recovery_df.columns
        if col.endswith("_proportion")
    ]

    for ensemble_loss in recovery_df["ensemble_loss"].unique():
        el_data = recovery_df[recovery_df["ensemble_loss"] == ensemble_loss]
        if el_data.empty:
            continue

        print(f"  Creating recovery plots for: {ensemble_loss}")

        split_types = sorted(el_data["split_type"].unique())
        n_states = len(state_names)
        fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5))
        if n_states == 1:
            axes = [axes]

        for idx, state_name in enumerate(state_names):
            ax = axes[idx]
            col_name = f"{state_name}_proportion"

            for split_type in split_types:
                st_data = el_data[el_data["split_type"] == split_type]
                color = split_type_colours.get(split_type, "#9467bd")
                label = split_name_mapping.get(split_type, split_type)

                ax.scatter(
                    st_data["maxent_value"],
                    st_data[col_name] * 100,
                    c=[color],
                    alpha=0.7,
                    label=label,
                    s=60,
                    edgecolors="w",
                    linewidths=1.5,
                )

                for split_idx in st_data["split"].unique():
                    si_data = st_data[st_data["split"] == split_idx]
                    if len(si_data) > 1:
                        si_data = si_data.sort_values("maxent_value")
                        ax.plot(
                            si_data["maxent_value"],
                            si_data[col_name] * 100,
                            color=color,
                            alpha=0.3,
                            linewidth=1,
                        )

            ax.set_xscale("log")
            ax.set_xlabel("MaxEnt Value", fontsize=12)
            ax.set_ylabel("Proportion (%)", fontsize=12)
            ax.set_title(state_name, fontsize=14, weight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            if idx == 0:
                ax.legend(fontsize=10)

        plt.suptitle(
            f"Conformational Recovery — {ensemble_loss}", fontsize=16, weight="bold", y=1.02
        )
        sns.despine()
        plt.tight_layout()

        fname = f"conformational_recovery_{ensemble_loss}.png".replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches="tight")
        print(f"  Saved: {fname}")
        plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Exp2 weights validation: KL/ESS analysis and conformational recovery."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory. If omitted, auto-discovered from config results_prefix.",
    )
    parser.add_argument(
        "--clustering-dir",
        default=None,
        help="Clustering directory. If omitted, taken from config.clustering_dir.",
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
        help="Interpret provided directories as absolute paths.",
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

    # Resolve clustering directory
    if args.clustering_dir:
        clustering_dir = (
            args.clustering_dir
            if args.absolute_paths
            else str((script_dir / args.clustering_dir).resolve())
        )
    elif cfg.clustering_dir:
        clustering_dir = str((exp_dir / cfg.clustering_dir).resolve())
    else:
        clustering_dir = str(script_dir / "_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters")

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

    print(f"Results directory  : {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory   : {output_dir}")
    print(f"EMA                : {args.ema}")
    print(f"Ensembles          : {cfg.ensembles}")
    print(f"Loss functions     : {cfg.loss_functions}")
    print(f"Num splits         : {cfg.num_splits}")
    print("-" * 60)

    # Load optimization results
    print("\nLoading optimization results...")
    results = loading.load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=cfg.ensembles,
        loss_functions=cfg.loss_functions,
        num_splits=cfg.num_splits,
        EMA=args.ema,
    )

    # Extract KL divergence and ESS at every convergence step
    print("\n" + "=" * 60)
    print("EXTRACTING KL DIVERGENCE AND ESS DATA")
    print("=" * 60)

    kl_ess_df = analysis.extract_frame_weights_kl(results)

    if not kl_ess_df.empty:
        kl_ess_path = os.path.join(output_dir, "kl_ess_data.csv")
        kl_ess_df.to_csv(kl_ess_path, index=False)
        print(f"KL/ESS data saved to: {kl_ess_path}")

        convergence_rates = sorted(kl_ess_df["step"].unique().tolist())
        print(
            f"Convergence steps: {convergence_rates[:10]}..."
            if len(convergence_rates) > 10
            else f"Convergence steps: {convergence_rates}"
        )
    else:
        convergence_rates = []

    # Extract final weights
    print("\n" + "=" * 60)
    print("EXTRACTING FINAL WEIGHTS")
    print("=" * 60)

    final_df = analysis.extract_final_weights(results)
    print(f"Extracted {len(final_df)} final weight distributions")

    # Extract weights over convergence steps for panel plots
    print("\n" + "=" * 60)
    print("EXTRACTING CONVERGENCE WEIGHTS")
    print("=" * 60)
    
    conv_df = analysis.extract_weights_over_convergence_steps(results)
    print(f"Extracted {len(conv_df)} convergence weights distributions")

    # Compute pairwise KLD between splits (per convergence step)
    print("\n" + "=" * 60)
    print("COMPUTING KLD BETWEEN SPLITS")
    print("=" * 60)

    kld_df = pd.DataFrame()
    if not kl_ess_df.empty:
        kld_df = analysis.compute_pairwise_kld_between_splits(kl_ess_df, per_step=True)
        if not kld_df.empty:
            kld_path = os.path.join(output_dir, "kld_between_splits_data.csv")
            kld_df.to_csv(kld_path, index=False)
            print(f"KLD between splits data saved to: {kld_path}")

    # Sequential MaxEnt KLD
    print("\n" + "=" * 60)
    print("COMPUTING SEQUENTIAL MAXENT KLD")
    print("=" * 60)

    seq_kld_df = pd.DataFrame()
    if not final_df.empty:
        seq_kld_df = analysis.compute_sequential_maxent_kld(final_df)
        if not seq_kld_df.empty:
            seq_kld_path = os.path.join(output_dir, "sequential_maxent_kld_data.csv")
            seq_kld_df.to_csv(seq_kld_path, index=False)
            print(f"Sequential MaxEnt KLD data saved to: {seq_kld_path}")

    # Conformational recovery
    print("\n" + "=" * 60)
    print("CALCULATING CONFORMATIONAL RECOVERY")
    print("=" * 60)

    recovery_df = pd.DataFrame()
    if not final_df.empty:
        recovery_df = calculate_conformational_recovery(final_df, clustering_dir)
        if not recovery_df.empty:
            recovery_path = os.path.join(output_dir, "conformational_recovery.csv")
            recovery_df.to_csv(recovery_path, index=False)
            print(f"Conformational recovery data saved to: {recovery_path}")

    # Heatmaps
    print("\n" + "=" * 60)
    print("CREATING HEATMAPS")
    print("=" * 60)

    if not kl_ess_df.empty and convergence_rates:
        plotting.plot_metric_heatmap(
            kl_ess_df, "effective_sample_size", convergence_rates, output_dir,
            cmap="viridis", columns_col="step",
        )
        plotting.plot_metric_heatmap(
            kl_ess_df, "kl_divergence", convergence_rates, output_dir,
            cmap="Blues", columns_col="step",
        )
    else:
        print("  Skipping KL/ESS heatmaps — no data")

    if not kld_df.empty and convergence_rates:
        plotting.plot_metric_heatmap(
            kld_df, "mean_kld_between_splits", convergence_rates, output_dir,
            cmap="Blues", columns_col="step",
        )
    else:
        print("  Skipping KLD between splits heatmap — no data")

    # Additional plots
    print("\n" + "=" * 60)
    print("CREATING ADDITIONAL PLOTS")
    print("=" * 60)

    if not final_df.empty:
        plotting.plot_weight_distribution_lines(final_df, output_dir, SPLIT_NAME_MAPPING)

    if not conv_df.empty:
        plotting.plot_weight_distribution_maxent_panels(conv_df, output_dir, SPLIT_NAME_MAPPING)
        plotting.plot_weight_distribution_convergence_panels(conv_df, output_dir, SPLIT_NAME_MAPPING)
        
    if not kld_df.empty:
        plotting.plot_kld_between_splits(kld_df, output_dir, SPLIT_NAME_MAPPING)

    if not seq_kld_df.empty:
        plotting.plot_sequential_maxent_kld(seq_kld_df, output_dir, SPLIT_NAME_MAPPING)

    if not recovery_df.empty:
        plot_conformational_recovery_scatter(recovery_df, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
