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
import re
import sys
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

from jaxent.src.utils.hdf import load_optimization_history_from_file

# Ensemble to clustering subdirectory mapping
ENSEMBLE_CLUSTERING_MAP = {
    "AF2_MSAss": "AF2_MSAss",
    "AF2_filtered": "AF2_Filtered",
}


def calculate_recovery_JSD(cluster_assignments, weights, target_ratios, state_mapping):
    """
    Compute Jensen-Shannon divergence between observed and target state proportions.
    Returns JS divergence (float). If invalid, returns np.nan.

    Recovery% = (1 - sqrt(JSD)) * 100
    """
    state_to_clusters = {}
    for cluster_id, state_name in state_mapping.items():
        state_to_clusters.setdefault(state_name, []).append(cluster_id)

    current_proportions = {state: 0.0 for state in target_ratios}
    for state_name, cluster_ids in state_to_clusters.items():
        state_mask = cluster_assignments.isin(cluster_ids)
        current_proportions[state_name] = float(np.sum(weights[state_mask.to_numpy()]))

    states = list(target_ratios.keys())
    P = np.array([current_proportions.get(s, 0.0) for s in states], dtype=float)
    Q = np.array([target_ratios.get(s, 0.0) for s in states], dtype=float)

    sumP = P.sum()
    sumQ = Q.sum()
    if sumP > 0:
        P = P / sumP
    else:
        return np.nan, current_proportions

    if sumQ > 0:
        Q = Q / sumQ
    else:
        return np.nan, current_proportions

    M = 0.5 * (P + Q)

    def kld(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    js = 0.5 * (kld(P, M) + kld(Q, M))
    return float(js), current_proportions


def load_clustering_for_ensemble(ensemble_name: str, clustering_base_dir: str) -> pd.DataFrame:
    """Load clustering results for a specific ensemble."""
    if ensemble_name not in ENSEMBLE_CLUSTERING_MAP:
        raise ValueError(
            f"Unknown ensemble: {ensemble_name}. Expected one of {list(ENSEMBLE_CLUSTERING_MAP.keys())}"
        )

    clustering_subdir = ENSEMBLE_CLUSTERING_MAP[ensemble_name]
    clustering_path = os.path.join(
        clustering_base_dir, clustering_subdir, f"{clustering_subdir}_frame_to_cluster.csv"
    )

    if not os.path.exists(clustering_path):
        raise FileNotFoundError(f"Clustering file not found: {clustering_path}")

    print(f"Loading clustering for {ensemble_name} from: {clustering_path}")
    cluster_df = pd.read_csv(clustering_path)

    if "cluster_label" not in cluster_df.columns:
        raise ValueError(f"Expected 'cluster_label' column in {clustering_path}")

    print(
        f"  Loaded {len(cluster_df)} frames with {cluster_df['cluster_label'].nunique()} unique clusters"
    )

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
            all_files_in_dir = os.listdir(split_type_dir)
            hdf5_files_in_dir = [f for f in all_files_in_dir if f.endswith('.hdf5')]
            print(f"  Total files in directory: {len(all_files_in_dir)}")
            print(f"  HDF5 files in directory: {len(hdf5_files_in_dir)}")

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
                        and f"bvregfn{bv_reg_fn}" in f
                        and f.endswith(hdf_pattern)
                    ]
                    
                    if verbose and len(files) > 0:
                        print(f"  {ensemble} + {loss_name} + {bv_reg_fn}: Found {len(files)} files")

                    for filename in files:
                        # Extract maxent, bvreg, split_idx from filename
                        # Use [A-Za-z0-9] instead of \w to avoid matching underscore
                        match = re.search(
                            r"split(\d{3})_maxent([\d.]+)_bvreg([\d.]+)_bvregfn([A-Za-z0-9]+)",
                            filename,
                        )
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = float(match.group(2))
                            bvreg_val = float(match.group(3))
                            bvreg_fn_found = match.group(4)

                            if bvreg_fn_found != bv_reg_fn:
                                if verbose:
                                    print(f"    WARNING: BV reg mismatch: expected {bv_reg_fn}, got {bvreg_fn_found}")
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
                        else:
                            if verbose:
                                print(f"    WARNING: Could not parse filename: {filename}")

    print(f"\n{'='*60}")
    print(f"Total HDF5 files loaded: {total_files_found}")
    print(f"{'='*60}\n")

    return results


def find_best_convergence_threshold(history):
    """
    Find the convergence threshold (step) with the lowest validation loss.
    
    Returns: (best_step_idx, best_val_loss, state)
    """
    if history is None or not history.states:
        return None, np.inf, None

    best_val_loss = np.inf
    best_step_idx = 0
    best_state = None

    for step_idx, state in enumerate(history.states):
        if hasattr(state, "losses") and hasattr(state.losses, "val_losses"):
            val_losses = state.losses.val_losses
            if val_losses is not None and len(val_losses) > 0:
                val_loss = val_losses[0]  # First validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step_idx = step_idx
                    best_state = state

    return best_step_idx, best_val_loss, best_state


def analyze_recovery_2d_sweep(
    clustering_data: Dict[str, pd.DataFrame],
    results_dict: Dict,
    target_ratios: Dict[str, float],
    state_mapping: Dict[int, str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Analyze conformational recovery across 2D sweep of (maxent, bv_reg).
    
    For each combination, selects the convergence threshold with lowest validation loss.
    """
    recovery_data = []

    for ensemble_name, cluster_df in clustering_data.items():
        print(f"\nAnalyzing recovery for {ensemble_name}...")

        cluster_assignments = cluster_df["cluster_label"]
        n_frames = len(cluster_assignments)

        # Original (unweighted) baseline
        uniform_weights = np.ones(n_frames) / n_frames
        js_div_orig, current_props_orig = calculate_recovery_JSD(
            cluster_assignments, uniform_weights, target_ratios, state_mapping
        )

        if not np.isnan(js_div_orig):
            recovery_orig = (1.0 - np.sqrt(js_div_orig)) * 100.0
        else:
            recovery_orig = 0.0

        recovery_data.append(
            {
                "ensemble": ensemble_name,
                "loss_function": "Original",
                "bv_reg_function": "N/A",
                "split_type": "N/A",
                "split": "N/A",
                "maxent_value": 0.0,
                "bv_reg_value": 0.0,
                "best_convergence_step": "N/A",
                "validation_loss": np.nan,
                "js_divergence": js_div_orig if not np.isnan(js_div_orig) else 0.0,
                "js_distance": np.sqrt(js_div_orig) if not np.isnan(js_div_orig) else 0.0,
                "recovery_percent": recovery_orig,
                **{f"{state}_current": current_props_orig[state] for state in target_ratios},
                **{f"{state}_target": target_ratios[state] for state in target_ratios},
                "total_frames": n_frames,
            }
        )

        # Optimized weights from sweep
        optimized_count = 0
        
        for split_type in results_dict:
            if ensemble_name not in results_dict[split_type]:
                continue

            for loss_name in results_dict[split_type][ensemble_name]:
                for bv_reg_fn in results_dict[split_type][ensemble_name][loss_name]:
                    for maxent_val in results_dict[split_type][ensemble_name][loss_name][
                        bv_reg_fn
                    ]:
                        for bvreg_val in results_dict[split_type][ensemble_name][loss_name][
                            bv_reg_fn
                        ][maxent_val]:
                            for split_idx, history in results_dict[split_type][ensemble_name][
                                loss_name
                            ][bv_reg_fn][maxent_val][bvreg_val].items():
                                if history is None:
                                    continue

                                # Find best convergence threshold
                                best_step_idx, best_val_loss, best_state = (
                                    find_best_convergence_threshold(history)
                                )

                                if best_state is None:
                                    continue

                                if (
                                    hasattr(best_state.params, "frame_weights")
                                    and best_state.params.frame_weights is not None
                                ):
                                    frame_weights = np.array(best_state.params.frame_weights)

                                    if (
                                        len(frame_weights) == n_frames
                                        and np.sum(frame_weights) > 0
                                    ):
                                        normalized_weights = frame_weights / np.sum(frame_weights)

                                        js_div, current_props = calculate_recovery_JSD(
                                            cluster_assignments,
                                            normalized_weights,
                                            target_ratios,
                                            state_mapping,
                                        )

                                        if not np.isnan(js_div):
                                            recovery_pct = (1.0 - np.sqrt(js_div)) * 100.0
                                        else:
                                            recovery_pct = 0.0

                                        recovery_data.append(
                                            {
                                                "ensemble": ensemble_name,
                                                "loss_function": loss_name,
                                                "bv_reg_function": bv_reg_fn,
                                                "split_type": split_type,
                                                "split": split_idx,
                                                "maxent_value": maxent_val,
                                                "bv_reg_value": bvreg_val,
                                                "best_convergence_step": best_step_idx,
                                                "validation_loss": float(best_val_loss),
                                                "js_divergence": js_div
                                                if not np.isnan(js_div)
                                                else 0.0,
                                                "js_distance": np.sqrt(js_div)
                                                if not np.isnan(js_div)
                                                else 0.0,
                                                "recovery_percent": recovery_pct,
                                                **{
                                                    f"{state}_current": current_props[state]
                                                    for state in target_ratios
                                                },
                                                **{
                                                    f"{state}_target": target_ratios[state]
                                                    for state in target_ratios
                                                },
                                                "total_frames": n_frames,
                                            }
                                        )
                                        optimized_count += 1

        if verbose:
            print(f"  Added {optimized_count} optimized recovery records for {ensemble_name}")

    return pd.DataFrame(recovery_data)


def plot_2d_heatmaps_grid(
    recovery_df: pd.DataFrame,
    output_dir: str,
    metric: str = "recovery_percent",
    metric_label: str = "Recovery (%)",
):
    """
    Plot 2D heatmaps of recovery metrics across (maxent, bv_reg) sweep.
    
    Grid layout: columns = loss functions, rows = BV reg functions
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Filter out original data
    df = recovery_df[recovery_df["loss_function"] != "Original"].copy()

    if len(df) == 0:
        print("No data available for 2D heatmaps")
        return

    split_types = sorted(df["split_type"].unique())

    for split_type in split_types:
        print(f"Creating 2D heatmaps for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df[df["split_type"] == split_type]

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
                f"2D Hyperparameter Sweep - {ensemble} - {split_type} splits\n{metric_label} at Best Validation Loss",
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
                        # Average across replicates
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

                            # Determine colormap limits
                            if metric == "recovery_percent":
                                vmin, vmax = 0, 100
                                cmap = "RdYlGn"
                            else:
                                vmin, vmax = pivot_data.min().min(), pivot_data.max().max()
                                cmap = "viridis"

                            im = sns.heatmap(
                                pivot_data,
                                annot=True,
                                fmt=".1f",
                                cmap=cmap,
                                vmin=vmin,
                                vmax=vmax,
                                cbar_kws={"label": metric_label},
                                ax=ax,
                                cbar=(j == len(loss_functions) - 1),  # Only rightmost
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


def plot_1d_slices(
    recovery_df: pd.DataFrame,
    output_dir: str,
    metric: str = "recovery_percent",
    metric_label: str = "Recovery (%)",
):
    """
    Plot 1D slices: recovery vs maxent (for fixed bv_reg values) and vice versa.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    # Filter out original data
    df = recovery_df[recovery_df["loss_function"] != "Original"].copy()

    if len(df) == 0:
        print("No data available for 1D slices")
        return

    split_types = sorted(df["split_type"].unique())

    for split_type in split_types:
        print(f"Creating 1D slices for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df[df["split_type"] == split_type]

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for ensemble in ensembles:
            ensemble_df = split_df[split_df["ensemble"] == ensemble]

            # Plot 1: Recovery vs MaxEnt (for each BV reg value and loss-reg combo)
            print(f"  Plotting recovery vs MaxEnt for {ensemble}...")

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
                        # Group by bv_reg_value
                        bv_reg_values = sorted(combo_df["bv_reg_value"].unique())

                        colors = plt.cm.viridis(np.linspace(0, 1, len(bv_reg_values)))

                        for k, bv_reg_val in enumerate(bv_reg_values):
                            bv_subset = combo_df[combo_df["bv_reg_value"] == bv_reg_val]

                            # Average across replicates and convergence steps
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

                        if metric == "recovery_percent":
                            ax.set_ylim([0, 100])
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

            # Plot 2: Recovery vs BV Reg (for each MaxEnt value and loss-reg combo)
            print(f"  Plotting recovery vs BV Reg for {ensemble}...")

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
                        # Group by maxent_value
                        maxent_values = sorted(combo_df["maxent_value"].unique())

                        colors = plt.cm.plasma(np.linspace(0, 1, len(maxent_values)))

                        for k, maxent_val in enumerate(maxent_values):
                            maxent_subset = combo_df[combo_df["maxent_value"] == maxent_val]

                            # Average across replicates and convergence steps
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

                        if metric == "recovery_percent":
                            ax.set_ylim([0, 100])
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


def main():
    """Main function for 2D hyperparameter sweep recovery analysis."""
    parser = argparse.ArgumentParser(
        description="2D hyperparameter sweep recovery analysis for MoPrP."
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
        "--state-ratios-json",
        default="../analysis/state_ratios.json",
        help="Path to JSON with target state ratios",
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
    bv_reg_functions = ["L1", "L2"]

    state_mapping = {
        0: "Folded",
        1: "PUF1",
        2: "PUF2",
        4: "unfolded",
    }

    # Resolve directories
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, args.results_dir)
    clustering_dir = os.path.join(script_dir, args.clustering_dir)

    if args.output_dir:
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = os.path.join(script_dir, "_analysis_" + base_name)

    print(f"Results directory: {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory: {output_dir}")
    print(f"EMA flag: {args.ema}")

    # Load target ratios
    state_ratios_path = os.path.join(script_dir, args.state_ratios_json)
    try:
        with open(state_ratios_path, "r") as f:
            ratios_data = json.load(f)
        target_ratios = {
            "Folded": ratios_data["fractional_populations"]["folded"]["fraction"],
            "PUF1": ratios_data["fractional_populations"]["PUF1"]["fraction"],
            "PUF2": ratios_data["fractional_populations"]["PUF2"]["fraction"],
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
    clustering_data = {}
    for ensemble in ensembles:
        try:
            clustering_data[ensemble] = load_clustering_for_ensemble(ensemble, clustering_dir)
        except Exception as e:
            print(f"Error loading clustering for {ensemble}: {e}")
            return

    # Load optimization results
    print("\nLoading optimization results...")
    results = load_all_optimization_results_2d_sweep(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        bv_reg_functions=bv_reg_functions,
        EMA=args.ema,
        verbose=True,  # Enable verbose output
    )

    os.makedirs(output_dir, exist_ok=True)

    # Analyze recovery
    print("\n" + "=" * 60)
    print("ANALYZING CONFORMATIONAL RECOVERY - 2D SWEEP")
    print("=" * 60)

    recovery_df = analyze_recovery_2d_sweep(
        clustering_data, results, target_ratios, state_mapping, verbose=True
    )

    if len(recovery_df) > 0:
        print(f"\nExtracted {len(recovery_df)} recovery data points")

        # Save recovery data
        recovery_path = os.path.join(output_dir, "recovery_2d_sweep_data.csv")
        recovery_df.to_csv(recovery_path, index=False)
        print(f"Recovery data saved to: {recovery_path}")

        # Generate 2D heatmaps
        print("\nGenerating 2D heatmaps...")
        plot_2d_heatmaps_grid(
            recovery_df,
            output_dir,
            metric="recovery_percent",
            metric_label="Recovery (%)",
        )
        plot_2d_heatmaps_grid(
            recovery_df,
            output_dir,
            metric="validation_loss",
            metric_label="Validation Loss",
        )

        # Generate 1D slices
        print("\nGenerating 1D slices...")
        plot_1d_slices(
            recovery_df,
            output_dir,
            metric="recovery_percent",
            metric_label="Recovery (%)",
        )
        plot_1d_slices(
            recovery_df,
            output_dir,
            metric="validation_loss",
            metric_label="Validation Loss",
        )

        # Generate best hyperparameters summary
        print("\nGenerating best hyperparameters summary...")
        plot_best_hyperparameters(recovery_df, output_dir, metric="recovery_percent")

        # Print summary
        print("\n" + "=" * 60)
        print("RECOVERY SUMMARY - 2D SWEEP")
        print("=" * 60)

        for ensemble in ensembles:
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