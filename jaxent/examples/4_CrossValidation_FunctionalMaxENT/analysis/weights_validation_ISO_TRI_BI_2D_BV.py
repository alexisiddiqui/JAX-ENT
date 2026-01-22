"""
2D Hyperparameter Sweep Weights Validation Analysis for MoPrP System

Analyzes frame weights and statistical properties across a 2D grid of:
- maxent scaling values (x-axis)
- bv_reg scaling values (y-axis)
- Multiple error loss functions (columns: MSE, Sigma_MSE, mcMSE)
- BV regularization functions (rows: L1, L2)

For each (maxent, bv_reg) combination, analyzes KL divergence, effective sample size,
and between-split consistency metrics using final converged weights.

Key metrics:
- KL divergence vs uniform prior (how non-uniform are the learned weights?)
- Effective Sample Size (ESS) - how many frames are effectively used?
- KLD between splits (consistency across CV folds)
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add base directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.src.utils.hdf import load_optimization_history_from_file

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# Ensemble to clustering file mapping
ENSEMBLE_CLUSTERING_MAP = {
    "AF2_MSAss": "AF2_MSAss_frame_to_cluster.csv",
    "AF2_filtered": "AF2_Filtered_frame_to_cluster.csv",
}

# State mapping for conformational analysis
STATE_MAPPING = {
    0: "Folded",
    1: "PUF1",
    2: "PUF2",
}

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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Calculate KL divergence between two probability distributions.

    Args:
        p: First probability distribution (frame_weights)
        q: Second probability distribution (uniform prior)
        eps: Small value to avoid log(0)

    Returns:
        KL divergence KL(p||q)
    """
    p = p / np.sum(p)
    q = q / np.sum(q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Calculate Effective Sample Size (ESS) as 1/sum(weights^2).

    Args:
        weights: Frame weights (should be normalized to sum to 1)

    Returns:
        Effective sample size
    """
    normalized_weights = weights / np.sum(weights)
    ess = 1.0 / np.sum(normalized_weights**2)
    return float(ess)


def load_clustering_for_ensemble(ensemble_name: str, clustering_base_dir: str) -> pd.DataFrame:
    """Load clustering results for a specific ensemble."""
    if ensemble_name not in ENSEMBLE_CLUSTERING_MAP:
        raise ValueError(
            f"Unknown ensemble: {ensemble_name}. Expected one of {list(ENSEMBLE_CLUSTERING_MAP.keys())}"
        )

    clustering_subdir = ENSEMBLE_CLUSTERING_MAP[ensemble_name].replace("_frame_to_cluster.csv", "")
    clustering_path = os.path.join(
        clustering_base_dir, clustering_subdir, ENSEMBLE_CLUSTERING_MAP[ensemble_name]
    )

    if not os.path.exists(clustering_path):
        raise FileNotFoundError(f"Clustering file not found: {clustering_path}")

    cluster_df = pd.read_csv(clustering_path)
    print(f"Loaded clustering for {ensemble_name}: {len(cluster_df)} frames")

    return cluster_df


def load_all_optimization_results_2d_sweep(
    results_dir: str,
    ensembles: List[str],
    loss_functions: List[str],
    bv_reg_functions: List[str],
    num_splits: int = 3,
    EMA: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Load all optimization results from HDF5 files for 2D hyperparameter sweep.
    
    Returns nested dict: results[split_type][ensemble][loss_fn][bv_reg_fn][maxent][bv_reg][split_idx] = history
    """
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]

    if verbose:
        print(f"\nDiscovered split types: {split_types}")

    hdf_pattern = "results_EMA.hdf5" if EMA else "results.hdf5"
    print(f"Looking for HDF5 files ending with: {hdf_pattern}\n")

    total_files_found = 0

    for split_type in split_types:
        results[split_type] = {}
        split_type_dir = os.path.join(results_dir, split_type)

        if verbose:
            print(f"Processing split type: {split_type}")

        for ensemble in ensembles:
            results[split_type][ensemble] = {}

            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}

                for bv_reg_fn in bv_reg_functions:
                    results[split_type][ensemble][loss_name][bv_reg_fn] = {}

                    # Filter files matching this combination
                    all_files = os.listdir(split_type_dir)
                    files = [
                        f
                        for f in all_files
                        if f.startswith(f"{ensemble}_{loss_name}_{split_type}_split")
                        and f"func{bv_reg_fn}" in f
                        and f.endswith(hdf_pattern)
                    ]

                    if verbose and len(files) > 0:
                        print(f"  {ensemble} + {loss_name} + {bv_reg_fn}: Found {len(files)} files")

                    for filename in files:
                        # Extract maxent, bvreg, split_idx from filename
                        match = re.search(
                            r"split(\d{3})_maxent([\d.]+)_bvreg([\d.]+)_func(.+?)(?:_results)?(?:_EMA)?\.hdf5",
                            filename,
                        )
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = float(match.group(2))
                            bvreg_val = float(match.group(3))
                            bvreg_fn_found = match.group(4)

                            if bvreg_fn_found != bv_reg_fn:
                                continue

                            if maxent_val not in results[split_type][ensemble][loss_name][bv_reg_fn]:
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val] = {}

                            if (
                                bvreg_val
                                not in results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val]
                            ):
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][
                                    bvreg_val
                                ] = {}

                            filepath = os.path.join(split_type_dir, filename)

                            try:
                                history = load_optimization_history_from_file(filepath)
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][
                                    bvreg_val
                                ][split_idx] = history
                                total_files_found += 1
                                if verbose:
                                    print(f"    ✓ Loaded: {filename}")
                            except Exception as e:
                                if verbose:
                                    print(f"    ✗ Failed to load {filename}: {str(e)[:100]}")
                                results[split_type][ensemble][loss_name][bv_reg_fn][maxent_val][
                                    bvreg_val
                                ][split_idx] = None

    print(f"\n{'='*60}")
    print(f"Total HDF5 files loaded: {total_files_found}")
    print(f"{'='*60}\n")

    return results


# ============================================================================
# WEIGHTS EXTRACTION AND ANALYSIS
# ============================================================================


def extract_final_weights_2d_sweep(results: Dict) -> pd.DataFrame:
    """
    Extract final (converged) frame weights and compute KL divergence and ESS.
    
    For each (maxent, bv_reg) combination and split, get the final weights
    and compute statistical metrics against uniform prior.
    """
    data_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for bv_reg_fn in results[split_type][ensemble][loss_name]:
                    for maxent_val in results[split_type][ensemble][loss_name][bv_reg_fn]:
                        for bvreg_val in results[split_type][ensemble][loss_name][bv_reg_fn][
                            maxent_val
                        ]:
                            for split_idx, history in results[split_type][ensemble][loss_name][
                                bv_reg_fn
                            ][maxent_val][bvreg_val].items():
                                if history is None or not history.states:
                                    continue

                                # Get final state
                                final_state = history.states[-1]

                                if (
                                    hasattr(final_state.params, "frame_weights")
                                    and final_state.params.frame_weights is not None
                                ):
                                    frame_weights = np.array(final_state.params.frame_weights)

                                    # Handle NaN/inf values
                                    if np.any(np.isnan(frame_weights)) or np.any(
                                        np.isinf(frame_weights)
                                    ):
                                        frame_weights = np.nan_to_num(
                                            frame_weights, nan=0.0, posinf=0.0, neginf=0.0
                                        )

                                    # Normalize weights
                                    if np.sum(frame_weights) > 0:
                                        frame_weights = frame_weights / np.sum(frame_weights)

                                        # Compute metrics
                                        uniform_prior = np.ones(len(frame_weights)) / len(
                                            frame_weights
                                        )
                                        try:
                                            kl_div = kl_divergence(frame_weights, uniform_prior)
                                            ess = effective_sample_size(frame_weights)

                                            data_rows.append(
                                                {
                                                    "split_type": split_type,
                                                    "ensemble": ensemble,
                                                    "loss_function": loss_name,
                                                    "bv_reg_function": bv_reg_fn,
                                                    "maxent_value": maxent_val,
                                                    "bv_reg_value": bvreg_val,
                                                    "split": split_idx,
                                                    "kl_divergence": kl_div,
                                                    "effective_sample_size": ess,
                                                    "num_frames": len(frame_weights),
                                                    "weight_min": float(np.min(frame_weights)),
                                                    "weight_max": float(np.max(frame_weights)),
                                                    "weight_mean": float(np.mean(frame_weights)),
                                                    "weight_std": float(np.std(frame_weights)),
                                                }
                                            )
                                        except Exception as e:
                                            print(
                                                f"Failed to compute metrics for {split_type}/{ensemble}_{loss_name}_maxent{maxent_val}_bvreg{bvreg_val}_split{split_idx}: {e}"
                                            )

    return pd.DataFrame(data_rows)


def compute_pairwise_kld_between_splits_2d(results: Dict) -> pd.DataFrame:
    """
    Compute pairwise KLD between splits for each (maxent, bv_reg) combination.
    
    For each ensemble, split_type, loss, bv_reg, maxent pair, compute the symmetric
    KL divergence between final weights across splits.
    """
    print("Computing pairwise KLD between splits (2D sweep)...")
    kld_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_func in results[split_type][ensemble]:
                for bv_reg_fn in results[split_type][ensemble][loss_func]:
                    for maxent_val in results[split_type][ensemble][loss_func][bv_reg_fn]:
                        for bvreg_val in results[split_type][ensemble][loss_func][bv_reg_fn][
                            maxent_val
                        ]:
                            # Collect final weights for this (maxent, bvreg) combination across splits
                            weights_list = []
                            split_indices = []

                            for split_idx, history in results[split_type][ensemble][loss_func][
                                bv_reg_fn
                            ][maxent_val][bvreg_val].items():
                                if history is None or not history.states:
                                    continue

                                final_state = history.states[-1]
                                if (
                                    hasattr(final_state.params, "frame_weights")
                                    and final_state.params.frame_weights is not None
                                ):
                                    w = np.array(final_state.params.frame_weights)
                                    if np.sum(w) <= 0 or len(w) == 0:
                                        continue

                                    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                                    if np.sum(w) > 0:
                                        w = w / np.sum(w)
                                        weights_list.append(w)
                                        split_indices.append(split_idx)

                            if len(weights_list) < 2:
                                continue

                            # Compute pairwise symmetric KLD
                            pair_klds = []
                            for i in range(len(weights_list)):
                                for j in range(i + 1, len(weights_list)):
                                    wi = weights_list[i]
                                    wj = weights_list[j]
                                    min_len = min(len(wi), len(wj))

                                    if min_len == 0:
                                        continue

                                    wi_trim = wi[:min_len]
                                    wj_trim = wj[:min_len]

                                    try:
                                        kld_ij = kl_divergence(wi_trim, wj_trim)
                                        kld_ji = kl_divergence(wj_trim, wi_trim)

                                        if not (np.isnan(kld_ij) or np.isnan(kld_ji)):
                                            pair_klds.append((kld_ij + kld_ji) / 2.0)
                                    except Exception:
                                        continue

                            if len(pair_klds) > 0:
                                mean_kld = float(np.mean(pair_klds))
                                std_kld = float(np.std(pair_klds))
                                sem_kld = (
                                    float(std_kld / np.sqrt(len(pair_klds)))
                                    if len(pair_klds) > 0
                                    else np.nan
                                )

                                kld_rows.append(
                                    {
                                        "ensemble": ensemble,
                                        "split_type": split_type,
                                        "loss_function": loss_func,
                                        "bv_reg_function": bv_reg_fn,
                                        "maxent_value": maxent_val,
                                        "bv_reg_value": bvreg_val,
                                        "mean_kld_between_splits": mean_kld,
                                        "std_kld_between_splits": std_kld,
                                        "sem_kld_between_splits": sem_kld,
                                        "n_pairs": len(pair_klds),
                                        "n_splits": len(weights_list),
                                    }
                                )

    return pd.DataFrame(kld_rows)


# ============================================================================
# 2D HEATMAP PLOTTING FUNCTIONS
# ============================================================================


def plot_2d_heatmaps_grid(
    weights_df: pd.DataFrame,
    output_dir: str,
    metric: str = "effective_sample_size",
    metric_label: str = "Effective Sample Size",
):
    """
    Plot 2D heatmaps of weight metrics across (maxent, bv_reg) sweep.
    
    Grid layout: columns = loss functions, rows = BV reg functions
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    if len(weights_df) == 0:
        print("No data available for 2D heatmaps")
        return

    split_types = sorted(weights_df["split_type"].unique())

    for split_type in split_types:
        print(f"Creating 2D heatmaps for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = weights_df[weights_df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for ensemble in ensembles:
            print(f"  {ensemble}:")

            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"2D Weights Analysis - {ensemble} - {split_type} splits\n{metric_label}",
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
                        # Average across replicates (splits)
                        pivot_data = combo_df.pivot_table(
                            values=metric,
                            index="bv_reg_value",
                            columns="maxent_value",
                            aggfunc="mean",
                        )

                        if not pivot_data.empty:
                            # Sort indices for better visualization
                            pivot_data = pivot_data.sort_index(ascending=False)
                            pivot_data = pivot_data.sort_index(axis=1)

                            # Determine colormap and limits
                            if metric == "effective_sample_size":
                                vmin = 0
                                vmax = combo_df[metric].max()
                                cmap = "viridis"
                            elif metric == "kl_divergence":
                                vmin = 0
                                vmax = combo_df[metric].max()
                                cmap = "YlOrRd"
                            else:
                                vmin = pivot_data.min().min()
                                vmax = pivot_data.max().max()
                                cmap = "viridis"

                            im = sns.heatmap(
                                pivot_data,
                                annot=True,
                                fmt=".2f",
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax,
                                cbar_kws={"label": metric_label},
                                ax=ax,
                                cbar=(j == len(loss_functions) - 1),
                            )

                            ax.set_title(f"{loss_fn} + {bv_reg_fn}")
                            ax.set_xlabel("MaxEnt Value")
                            ax.set_ylabel("BV Reg Value")

                            # Format labels
                            xticklabels = [f"{float(t.get_text()):.0f}" for t in ax.get_xticklabels()]
                            ax.set_xticklabels(xticklabels, rotation=45, ha="right")

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
                os.path.join(split_output_dir, f"2d_heatmap_{ensemble}_{metric}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            print(f"    Saved heatmap for {ensemble}")


def plot_2d_kld_between_splits_heatmaps(
    kld_df: pd.DataFrame,
    output_dir: str,
):
    """
    Plot 2D heatmaps of KLD between splits across (maxent, bv_reg) sweep.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    if len(kld_df) == 0:
        print("No KLD data available for 2D heatmaps")
        return

    split_types = sorted(kld_df["split_type"].unique())

    for split_type in split_types:
        print(f"Creating KLD between splits heatmaps for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = kld_df[kld_df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for ensemble in ensembles:
            print(f"  {ensemble}:")

            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"KLD Between Splits - {ensemble} - {split_type} splits",
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
                        pivot_data = combo_df.pivot_table(
                            values="mean_kld_between_splits",
                            index="bv_reg_value",
                            columns="maxent_value",
                            aggfunc="mean",
                        )

                        if not pivot_data.empty:
                            pivot_data = pivot_data.sort_index(ascending=False)
                            pivot_data = pivot_data.sort_index(axis=1)

                            vmin = 0
                            vmax = combo_df["mean_kld_between_splits"].max()

                            im = sns.heatmap(
                                pivot_data,
                                annot=True,
                                fmt=".3f",
                                cmap="Blues",
                                vmin=vmin,
                                vmax=vmax,
                                cbar_kws={"label": "Mean KLD Between Splits"},
                                ax=ax,
                                cbar=(j == len(loss_functions) - 1),
                            )

                            ax.set_title(f"{loss_fn} + {bv_reg_fn}")
                            ax.set_xlabel("MaxEnt Value")
                            ax.set_ylabel("BV Reg Value")

                            xticklabels = [f"{float(t.get_text()):.0f}" for t in ax.get_xticklabels()]
                            ax.set_xticklabels(xticklabels, rotation=45, ha="right")

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
                os.path.join(split_output_dir, f"2d_heatmap_{ensemble}_kld_between_splits.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            print(f"    Saved KLD heatmap for {ensemble}")


# ============================================================================
# 1D SLICE PLOTTING FUNCTIONS
# ============================================================================


def plot_1d_slices_maxent_vs_bvreg(
    weights_df: pd.DataFrame,
    output_dir: str,
    metric: str = "effective_sample_size",
    metric_label: str = "Effective Sample Size",
):
    """
    Plot 1D slices: metric vs maxent (for fixed bv_reg values) and vice versa.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    if len(weights_df) == 0:
        print("No data available for 1D slices")
        return

    split_types = sorted(weights_df["split_type"].unique())

    for split_type in split_types:
        print(f"Creating 1D slices for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = weights_df[weights_df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for ensemble in ensembles:
            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            # Plot 1: Metric vs MaxEnt (for each BV reg value and loss-reg combo)
            print(f"  Plotting {metric} vs MaxEnt for {ensemble}...")

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"1D Slices: {metric_label} vs MaxEnt - {ensemble} - {split_type}",
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
                        bv_reg_values = sorted(combo_df["bv_reg_value"].unique())
                        colors = plt.cm.viridis(np.linspace(0, 1, len(bv_reg_values)))

                        for k, bv_reg_val in enumerate(bv_reg_values):
                            bv_subset = combo_df[combo_df["bv_reg_value"] == bv_reg_val]
                            avg_data = bv_subset.groupby("maxent_value")[metric].agg(["mean", "std"])
                            avg_data = avg_data.reset_index()

                            ax.errorbar(
                                avg_data["maxent_value"],
                                avg_data["mean"],
                                yerr=avg_data["std"],
                                marker="o",
                                label=f"BV Reg={bv_reg_val:.2f}",
                                color=colors[k],
                                capsize=5,
                                capthick=2,
                                linewidth=2,
                                markersize=6,
                            )

                        ax.set_xscale("log")
                        ax.set_xlabel("MaxEnt Value (log scale)")
                        ax.set_ylabel(metric_label)
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}")
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=8, loc="best")
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
                os.path.join(split_output_dir, f"1d_slice_maxent_{ensemble}_{metric}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Plot 2: Metric vs BV Reg (for each MaxEnt value and loss-reg combo)
            print(f"  Plotting {metric} vs BV Reg for {ensemble}...")

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(5 * len(loss_functions), 4 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"1D Slices: {metric_label} vs BV Reg - {ensemble} - {split_type}",
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
                        maxent_values = sorted(combo_df["maxent_value"].unique())
                        colors = plt.cm.plasma(np.linspace(0, 1, len(maxent_values)))

                        for k, maxent_val in enumerate(maxent_values):
                            maxent_subset = combo_df[combo_df["maxent_value"] == maxent_val]
                            avg_data = maxent_subset.groupby("bv_reg_value")[metric].agg(
                                ["mean", "std"]
                            )
                            avg_data = avg_data.reset_index()

                            ax.errorbar(
                                avg_data["bv_reg_value"],
                                avg_data["mean"],
                                yerr=avg_data["std"],
                                marker="s",
                                label=f"MaxEnt={maxent_val:.1f}",
                                color=colors[k],
                                capsize=5,
                                capthick=2,
                                linewidth=2,
                                markersize=6,
                            )

                        ax.set_xlabel("BV Reg Value")
                        ax.set_ylabel(metric_label)
                        ax.set_title(f"{loss_fn} + {bv_reg_fn}")
                        ax.grid(True, alpha=0.3)
                        ax.legend(fontsize=8, loc="best", ncol=2)
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
                os.path.join(split_output_dir, f"1d_slice_bvreg_{ensemble}_{metric}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main function for 2D hyperparameter sweep weights validation analysis."""
    parser = argparse.ArgumentParser(
        description="2D hyperparameter sweep weights validation analysis for MoPrP."
    )
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_quick_test_SIGMA_50_lr0.1_BV_objectve_20250918_171508",
        help="Results directory",
    )
    parser.add_argument(
        "--clustering-dir",
        default="_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters",
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
    args = parser.parse_args()

    # Parameters
    ensembles = ["AF2_MSAss", "AF2_filtered"]
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
    bv_reg_functions = ["ElasticNet", "ElasticMAE", "KLD_PF", "Work_Fitting", "Work_Magnitude"]

    # Resolve directories
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, args.results_dir)
    clustering_dir = os.path.join(script_dir, args.clustering_dir)

    if args.output_dir:
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = os.path.join(script_dir, "_analysis_weights_2d_sweep_" + base_name)

    print(f"Results directory: {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory: {output_dir}")
    print(f"EMA flag: {args.ema}")

    os.makedirs(output_dir, exist_ok=True)

    # Load optimization results
    print("\nLoading optimization results...")
    results = load_all_optimization_results_2d_sweep(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        bv_reg_functions=bv_reg_functions,
        EMA=args.ema,
        verbose=True,
    )

    # Extract weights and compute metrics
    print("\n" + "=" * 60)
    print("EXTRACTING WEIGHTS AND COMPUTING METRICS")
    print("=" * 60)

    weights_df = extract_final_weights_2d_sweep(results)

    if len(weights_df) > 0:
        print(f"\nExtracted {len(weights_df)} weight distributions")

        # Save weights data
        weights_path = os.path.join(output_dir, "weights_2d_sweep_data.csv")
        weights_df.to_csv(weights_path, index=False)
        print(f"Weights data saved to: {weights_path}")

        # Generate 2D heatmaps for ESS and KL divergence
        print("\nGenerating 2D heatmaps...")
        plot_2d_heatmaps_grid(
            weights_df,
            output_dir,
            metric="effective_sample_size",
            metric_label="Effective Sample Size",
        )
        plot_2d_heatmaps_grid(
            weights_df,
            output_dir,
            metric="kl_divergence",
            metric_label="KL Divergence (vs Uniform Prior)",
        )

        # Generate 1D slices
        print("\nGenerating 1D slices...")
        plot_1d_slices_maxent_vs_bvreg(
            weights_df,
            output_dir,
            metric="effective_sample_size",
            metric_label="Effective Sample Size",
        )
        plot_1d_slices_maxent_vs_bvreg(
            weights_df,
            output_dir,
            metric="kl_divergence",
            metric_label="KL Divergence (vs Uniform Prior)",
        )

        # Print summary statistics
        print("\n" + "=" * 60)
        print("WEIGHTS SUMMARY - 2D SWEEP")
        print("=" * 60)

        for ensemble in ensembles:
            print(f"\n{ensemble}:")
            ensemble_data = weights_df[weights_df["ensemble"] == ensemble]

            if len(ensemble_data) > 0:
                print(f"  Mean KL divergence: {ensemble_data['kl_divergence'].mean():.3f}")
                print(f"  Mean ESS: {ensemble_data['effective_sample_size'].mean():.1f}")
                print(
                    f"  Mean weight range: [{ensemble_data['weight_min'].mean():.4f}, {ensemble_data['weight_max'].mean():.4f}]"
                )

    # Compute KLD between splits
    print("\n" + "=" * 60)
    print("COMPUTING KLD BETWEEN SPLITS")
    print("=" * 60)

    kld_df = compute_pairwise_kld_between_splits_2d(results)

    if len(kld_df) > 0:
        print(f"\nComputed KLD between splits for {len(kld_df)} parameter combinations")

        # Save KLD data
        kld_path = os.path.join(output_dir, "kld_between_splits_2d_sweep_data.csv")
        kld_df.to_csv(kld_path, index=False)
        print(f"KLD data saved to: {kld_path}")

        # Generate KLD heatmaps
        print("\nGenerating KLD between splits heatmaps...")
        plot_2d_kld_between_splits_heatmaps(kld_df, output_dir)

        # Print KLD summary
        print("\n" + "=" * 60)
        print("KLD BETWEEN SPLITS SUMMARY")
        print("=" * 60)

        for ensemble in ensembles:
            ensemble_kld = kld_df[kld_df["ensemble"] == ensemble]
            if len(ensemble_kld) > 0:
                print(f"\n{ensemble}:")
                print(f"  Mean KLD between splits: {ensemble_kld['mean_kld_between_splits'].mean():.3f}")
                print(
                    f"  Std KLD between splits: {ensemble_kld['mean_kld_between_splits'].std():.3f}"
                )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()