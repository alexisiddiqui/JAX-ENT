"""
2D Hyperparameter Sweep Loss Analysis Script for MoPrP System

This script loads optimization histories across a 2D grid of:
- maxent scaling values (x-axis)
- bv_reg scaling values (y-axis)

Performs comprehensive analyses including:
- Convergence-maxent-bvreg heatmaps
- Model scoring and best model selection
- Best model comparisons with bar charts
- JSD-based recovery metrics
- Cluster proportion analysis

Updated to use 2D hyperparameter sweep structure with publication-ready plotting.
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List

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

# Publication-ready style configuration
sns.set_style("ticks")
sns.set_context(
    "paper",
    rc={
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 14,
        "ytick.labelsize": 10,
    },
)

# Ensemble to clustering subdirectory mapping
ENSEMBLE_CLUSTERING_MAP = {
    "AF2_MSAss": "AF2_MSAss",
    "AF2_filtered": "AF2_Filtered",
}

# MoPrP ensemble coloring scheme
ENSEMBLE_COLOURS = {
    "AF2_MSAss": "RoyalBlue",
    "AF2_filtered": "Cyan",
}

# State mapping for MoPrP
STATE_MAPPING = {
    0: "Folded",
    1: "PUF1",
    2: "PUF2",
    4: "unfolded",
}

# Loss function markers
LOSS_MARKERS = {"mcMSE": "o", "MSE": "s", "Sigma_MSE": "^"}

# Split type coloring scheme
SPLIT_TYPE_COLOURS = {
    "sequence_cluster": "green",
    "spatial": "grey",
    "random": "fuchsia",
    "sequence": "black",
}

SPLIT_NAME_MAPPING = {
    "sequence_cluster": "Non-Redundant",
    "spatial": "Spatial",
    "random": "Random",
    "sequence": "Sequence",
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_all_optimization_results_2d(
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


def extract_loss_trajectories_2d(results: Dict) -> pd.DataFrame:
    """
    Extract loss trajectories from 2D hyperparameter sweep results.
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

                                for step_idx, state in enumerate(history.states):
                                    if state.losses is not None:
                                        data_rows.append(
                                            {
                                                "split_type": split_type,
                                                "ensemble": ensemble,
                                                "loss_function": loss_name,
                                                "bv_reg_function": bv_reg_fn,
                                                "maxent_value": maxent_val,
                                                "bv_reg_value": bvreg_val,
                                                "split": split_idx,
                                                "convergence_step": step_idx + 1,
                                                "train_loss": float(state.losses.train_losses[0]),
                                                "val_loss": float(state.losses.val_losses[0]),
                                                "step_number": state.step if hasattr(state, "step") else step_idx,
                                            }
                                        )

    return pd.DataFrame(data_rows)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================


def compute_model_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model scores as -log10(validation error).
    """
    df = df.copy()
    eps = 1e-300
    
    val_loss = df.get("val_loss", pd.Series(dtype=float))
    val_loss = val_loss.fillna(np.nan).clip(lower=eps)
    
    train_loss = df.get("train_loss", pd.Series(dtype=float))
    train_loss = train_loss.fillna(np.nan).clip(lower=eps)
    
    # Base score: negative log10 of validation loss with training contribution
    base_score = -np.log10(val_loss) - (0.9 * np.log10(train_loss))
    df["model_score"] = base_score
    
    return df


def select_best_models_2d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select best scoring models for each parameter combination.
    For each (split_type, ensemble, loss_function, bv_reg_function, split), 
    find the (maxent, bv_reg, convergence_step) with highest score.
    
    NOW: Selects best models SEPARATELY for each bv_reg_function.
    """
    df_scored = compute_model_scores(df)
    
    # Group by the fixed parameters INCLUDING bv_reg_function and select best
    best_models = df_scored.loc[
        df_scored.groupby(
            ["split_type", "ensemble", "loss_function", "bv_reg_function", "split"]
        )["model_score"].idxmax()
    ]
    
    return best_models


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

    print(f"  Loaded {len(cluster_df)} frames with {cluster_df['cluster_label'].nunique()} unique clusters")
    return cluster_df


def calculate_recovery_JSD(cluster_assignments, weights, target_ratios, state_mapping):
    """Compute Jensen-Shannon divergence based recovery metric."""
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


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Calculate KL divergence between two probability distributions."""
    p = p / np.sum(p)
    q = q / np.sum(q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def augment_best_models_with_metrics(
    best_models_df: pd.DataFrame,
    results: Dict,
    clustering_data: Dict[str, pd.DataFrame],
    target_ratios: Dict[str, float],
) -> pd.DataFrame:
    """
    Augment best models with KL divergence and JSD-based recovery metrics.
    """
    augmented_rows = []

    for _, row in best_models_df.iterrows():
        split_type = row["split_type"]
        ensemble = row["ensemble"]
        loss_func = row["loss_function"]
        bv_reg_fn = row["bv_reg_function"]
        split_idx = int(row["split"])
        maxent_val = row["maxent_value"]
        bvreg_val = row["bv_reg_value"]
        conv_step = int(row["convergence_step"])

        # Navigate nested results structure
        try:
            history = results[split_type][ensemble][loss_func][bv_reg_fn][maxent_val][bvreg_val][
                split_idx
            ]
        except (KeyError, TypeError):
            print(
                f"Warning: No results for {split_type}/{ensemble}/{loss_func}/{bv_reg_fn}/maxent{maxent_val}/bvreg{bvreg_val}/split{split_idx}"
            )
            continue

        if history is None or not hasattr(history, "states") or not history.states:
            print(
                f"Warning: No states in history for {split_type}/{ensemble}/{loss_func}/{bv_reg_fn}/maxent{maxent_val}/bvreg{bvreg_val}/split{split_idx}"
            )
            continue

        if conv_step <= 0 or conv_step > len(history.states):
            print(
                f"Warning: convergence_step {conv_step} out of range for {split_type}/{ensemble}/{loss_func}/{bv_reg_fn}/maxent{maxent_val}/bvreg{bvreg_val}/split{split_idx}"
            )
            continue

        state = history.states[conv_step - 1]

        if (
            hasattr(state, "params")
            and hasattr(state.params, "frame_weights")
            and state.params.frame_weights is not None
        ):
            weights = np.array(state.params.frame_weights)
            uniform_prior = np.ones(len(weights)) / len(weights)
            kl_div = kl_divergence(weights, uniform_prior)

            js_div = np.nan
            js_distance = np.nan
            recovery_percent = np.nan

            if ensemble in clustering_data:
                cluster_df = clustering_data[ensemble]
                cluster_assignments = cluster_df["cluster_label"]

                if len(weights) == len(cluster_assignments):
                    normalized_weights = weights / np.sum(weights)
                    js_div, current_props = calculate_recovery_JSD(
                        cluster_assignments, normalized_weights, target_ratios, STATE_MAPPING
                    )

                    if not np.isnan(js_div):
                        js_distance = np.sqrt(js_div)
                        recovery_percent = (1.0 - js_distance) * 100.0

            row_dict = row.to_dict()
            row_dict["kl_divergence"] = kl_div
            row_dict["js_divergence"] = js_div if not np.isnan(js_div) else 0.0
            row_dict["js_distance"] = js_distance if not np.isnan(js_distance) else 0.0
            row_dict["recovery_percent"] = recovery_percent if not np.isnan(recovery_percent) else 0.0
            augmented_rows.append(row_dict)

    return pd.DataFrame(augmented_rows)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================


def plot_convergence_heatmaps(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """
    Plot heatmaps of training and validation error across (maxent, bv_reg) for each loss-bvreg combo.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    if len(df) == 0:
        print("No data available for convergence heatmaps")
        return

    split_types = sorted(df["split_type"].unique()) if split_type is None else [split_type]

    for stype in split_types:
        print(f"  Creating convergence heatmaps for split type: {stype}")
        split_output_dir = os.path.join(output_dir, stype) if stype else output_dir
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df[df["split_type"] == stype] if stype else df

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for error_type in ["train_loss", "val_loss"]:
            error_label = "Training Error" if error_type == "train_loss" else "Validation Error"

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(6 * len(loss_functions), 5 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"{error_label} Heatmap: MaxEnt vs BV Reg{' - ' + stype if stype else ''}",
                fontsize=22,
                fontweight="bold",
            )

            for i, bv_reg_fn in enumerate(bv_reg_functions):
                for j, loss_func in enumerate(loss_functions):
                    ax = axes[i, j]

                    # Average across all ensembles and splits for each (maxent, bvreg) combo
                    combo_df = split_df[
                        (split_df["loss_function"] == loss_func)
                        & (split_df["bv_reg_function"] == bv_reg_fn)
                    ]

                    if len(combo_df) > 0:
                        # Find the final convergence step for each (maxent, bvreg, split)
                        final_df = combo_df.loc[
                            combo_df.groupby(["maxent_value", "bv_reg_value", "split"])[
                                "convergence_step"
                            ].idxmax()
                        ]

                        # Average across splits
                        pivot_data = final_df.pivot_table(
                            values=error_type,
                            index="bv_reg_value",
                            columns="maxent_value",
                            aggfunc="mean",
                        )

                        if not pivot_data.empty:
                            pivot_data = pivot_data.sort_index(ascending=False)
                            pivot_data = pivot_data.sort_index(axis=1)

                            sns.heatmap(
                                np.log10(pivot_data),
                                annot=False,
                                cmap="viridis",
                                cbar_kws={"label": f"log10({error_label})"},
                                ax=ax,
                            )

                            ax.set_title(f"{loss_func} + {bv_reg_fn}", fontweight="bold")
                            ax.set_xlabel("MaxEnt Value", fontweight="bold")
                            ax.set_ylabel("BV Reg Value", fontweight="bold")
                            ax.set_xticklabels([f"{float(t.get_text()):.0f}" for t in ax.get_xticklabels()], rotation=45, ha="right")
                            sns.despine(ax=ax)
                        else:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            ax.set_title(f"{loss_func} + {bv_reg_fn}")
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{loss_func} + {bv_reg_fn}")

            plt.tight_layout()
            filename = (
                f"{error_type}_convergence_heatmap_{stype}.png"
                if stype
                else f"{error_type}_convergence_heatmap.png"
            )
            plt.savefig(os.path.join(split_output_dir, filename), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"    Saved: {filename}")


def plot_model_score_heatmaps(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """
    Plot heatmaps of model scores across (maxent, bv_reg).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    df_scored = compute_model_scores(df)

    if len(df_scored) == 0:
        print("No data available for model score heatmaps")
        return

    split_types = sorted(df_scored["split_type"].unique()) if split_type is None else [split_type]

    for stype in split_types:
        print(f"  Creating model score heatmaps for split type: {stype}")
        split_output_dir = os.path.join(output_dir, stype) if stype else output_dir
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df_scored[df_scored["split_type"] == stype] if stype else df_scored

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())
        bv_reg_functions = sorted(split_df["bv_reg_function"].unique())

        for ensemble in ensembles:
            ensemble_df = split_df[split_df["ensemble"] == ensemble]
            if ensemble_df.empty:
                continue

            print(f"    Heatmaps for ensemble: {ensemble}")

            fig, axes = plt.subplots(
                len(bv_reg_functions),
                len(loss_functions),
                figsize=(6 * len(loss_functions), 5 * len(bv_reg_functions)),
                squeeze=False,
            )

            fig.suptitle(
                f"Model Scores (-log10 Val Error){' - ' + stype if stype else ''} - {ensemble}",
                fontsize=22,
                fontweight="bold",
            )

            for i, bv_reg_fn in enumerate(bv_reg_functions):
                for j, loss_func in enumerate(loss_functions):
                    ax = axes[i, j]

                    combo_df = ensemble_df[
                        (ensemble_df["loss_function"] == loss_func)
                        & (ensemble_df["bv_reg_function"] == bv_reg_fn)
                    ]

                    if len(combo_df) > 0:
                        # Get final step scores for each (maxent, bvreg, split)
                        final_df = combo_df.loc[
                            combo_df.groupby(["maxent_value", "bv_reg_value", "split"])[
                                "convergence_step"
                            ].idxmax()
                        ]

                        # Average scores across splits
                        pivot_data = final_df.pivot_table(
                            values="model_score",
                            index="bv_reg_value",
                            columns="maxent_value",
                            aggfunc="mean",
                        )

                        if not pivot_data.empty:
                            pivot_data = pivot_data.sort_index(ascending=False)
                            pivot_data = pivot_data.sort_index(axis=1)

                            sns.heatmap(
                                pivot_data,
                                annot=False,
                                cmap="RdYlGn",
                                cbar_kws={"label": "-log10(Val Error)"},
                                ax=ax,
                            )

                            ax.set_title(f"{loss_func} + {bv_reg_fn}", fontweight="bold")
                            ax.set_xlabel("MaxEnt Value", fontweight="bold")
                            ax.set_ylabel("BV Reg Value", fontweight="bold")
                            ax.set_xticklabels([f"{float(t.get_text()):.0f}" for t in ax.get_xticklabels()], rotation=45, ha="right")
                            sns.despine(ax=ax)
                        else:
                            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                            ax.set_title(f"{loss_func} + {bv_reg_fn}")
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{loss_func} + {bv_reg_fn}")

            plt.tight_layout()
            filename = (
                f"model_score_heatmap_{stype}_{ensemble}.png"
                if stype
                else f"model_score_heatmap_{ensemble}.png"
            )
            plt.savefig(os.path.join(split_output_dir, filename), dpi=300, bbox_inches="tight")
            plt.close()
            print(f"      Saved: {filename}")


def plot_loss_convergence(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """Plot error vs hyperparameters with training and validation error separate."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Error vs Hyperparameters{title_suffix}", fontsize=22, fontweight="bold")

    split_types = sorted(df["split_type"].unique()) if split_type is None else [split_type]

    for stype in split_types:
        split_df = df[df["split_type"] == stype] if stype else df

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        # Get final convergence step for each (ensemble, loss, maxent, bvreg, split)
        final_df = split_df.loc[
            split_df.groupby(["ensemble", "loss_function", "maxent_value", "bv_reg_value", "split"])[
                "convergence_step"
            ].idxmax()
        ]

        # Plot training errors (ax1)
        ax = ax1
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    stats = subset.groupby("maxent_value").agg({"train_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["maxent_value", "train_mean", "train_std"]

                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.errorbar(
                        stats["maxent_value"],
                        stats["train_mean"],
                        yerr=stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        capsize=4,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Training Error", fontweight="bold")
        ax.set_title("Training Error vs MaxEnt", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best", frameon=True, fancybox=False, edgecolor="black")
        sns.despine(ax=ax)

        # Plot validation errors (ax2)
        ax = ax2
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    stats = subset.groupby("maxent_value").agg({"val_loss": ["mean", "std"]}).reset_index()
                    stats.columns = ["maxent_value", "val_mean", "val_std"]

                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.errorbar(
                        stats["maxent_value"],
                        stats["val_mean"],
                        yerr=stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        capsize=4,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Validation Error", fontweight="bold")
        ax.set_title("Validation Error vs MaxEnt", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best", frameon=True, fancybox=False, edgecolor="black")
        sns.despine(ax=ax)

    plt.tight_layout()
    filename = f"error_vs_hyperparameters_{stype}.png" if stype else "error_vs_hyperparameters.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filename}")


def plot_split_variability(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """Plot standard deviation across splits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Standard Deviation Across Splits{title_suffix}", fontsize=22, fontweight="bold")

    split_types = sorted(df["split_type"].unique()) if split_type is None else [split_type]

    for stype in split_types:
        split_df = df[df["split_type"] == stype] if stype else df

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        # Get final convergence step
        final_df = split_df.loc[
            split_df.groupby(["ensemble", "loss_function", "maxent_value", "bv_reg_value", "split"])[
                "convergence_step"
            ].idxmax()
        ]

        # Plot training loss std (ax1)
        ax = ax1
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    std_stats = subset.groupby("maxent_value").agg({"train_loss": "std"}).reset_index()
                    std_stats.columns = ["maxent_value", "train_std"]

                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.plot(
                        std_stats["maxent_value"],
                        std_stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Training Error Std Dev", fontweight="bold")
        ax.set_title("Training Error Standard Deviation", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
        sns.despine(ax=ax)

        # Plot validation loss std (ax2)
        ax = ax2
        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = final_df[
                    (final_df["ensemble"] == ensemble) & (final_df["loss_function"] == loss_func)
                ]

                if len(subset) > 0:
                    std_stats = subset.groupby("maxent_value").agg({"val_loss": "std"}).reset_index()
                    std_stats.columns = ["maxent_value", "val_std"]

                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    ax.plot(
                        std_stats["maxent_value"],
                        std_stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("MaxEnt Value", fontweight="bold")
        ax.set_ylabel("Validation Error Std Dev", fontweight="bold")
        ax.set_title("Validation Error Standard Deviation", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="best")
        sns.despine(ax=ax)

    plt.tight_layout()
    filename = f"split_variability_{stype}.png" if stype else "split_variability.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {filename}")


def plot_best_model_comparisons(df: pd.DataFrame, output_dir: str):
    """Plot bar charts comparing best models with loss-reg pairs as panels."""
    if df.empty:
        print("No data for best model comparisons")
        return

    df = df.copy()
    df["split_name"] = df["split_type"].map(SPLIT_NAME_MAPPING).fillna(df["split_type"])

    metrics = [
        ("kl_divergence", "KL Divergence"),
        ("js_divergence", "JS Divergence"),
        ("recovery_percent", "Recovery %"),
        ("train_loss", "Training Loss"),
        ("val_loss", "Validation Loss"),
    ]

    for metric, label in metrics:
        if metric not in df.columns:
            continue

        # Create subplots for each loss_function + bv_reg_function combination
        loss_functions = sorted(df["loss_function"].unique())
        bv_reg_functions = sorted(df["bv_reg_function"].unique())
        
        n_panels = len(loss_functions) * len(bv_reg_functions)
        ncols = len(bv_reg_functions)
        nrows = len(loss_functions)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows), squeeze=False)

        split_names_in_data = df["split_name"].unique()
        palette = {
            name: SPLIT_TYPE_COLOURS.get(
                df[df["split_name"] == name]["split_type"].iloc[0], "grey"
            )
            for name in split_names_in_data
        }

        for row_idx, loss_func in enumerate(loss_functions):
            for col_idx, bv_reg_fn in enumerate(bv_reg_functions):
                ax = axes[row_idx, col_idx]
                combo_df = df[
                    (df["loss_function"] == loss_func) & (df["bv_reg_function"] == bv_reg_fn)
                ]

                if not combo_df.empty:
                    sns.barplot(
                        data=combo_df,
                        x="ensemble",
                        y=metric,
                        hue="split_name",
                        ax=ax,
                        ci="sd",
                        palette=palette,
                        edgecolor="black",
                        linewidth=1.2,
                    )

                    ax.set_xlabel("Ensemble", fontweight="bold")
                    ax.set_ylabel(label, fontweight="bold")
                    ax.set_title(f"{loss_func} + {bv_reg_fn}", fontweight="bold")

                    for tick_label in ax.get_xticklabels():
                        ensemble_name = tick_label.get_text()
                        color = ENSEMBLE_COLOURS.get(ensemble_name, "black")
                        tick_label.set_color(color)
                        tick_label.set_fontweight("bold")

                    handles, labels_leg = ax.get_legend_handles_labels()
                    ax.legend(
                        handles=handles,
                        labels=labels_leg,
                        title="Split Type",
                        title_fontsize=12,
                        fontsize=10,
                        frameon=True,
                        fancybox=False,
                        edgecolor="black",
                    )

                    if "loss" in metric:
                        ax.set_yscale("log")

                    if metric == "recovery_percent":
                        ax.axhline(y=100, color="red", linestyle="--", linewidth=2, alpha=0.7, zorder=0)

                    sns.despine(ax=ax)
                else:
                    ax.set_visible(False)

        plt.suptitle(f"Best Model Comparison: {label}", fontsize=22, fontweight="bold")
        plt.tight_layout()

        filename = f"best_model_comparison_{metric}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


def plot_cluster_proportions_for_best_models(
    best_models_df: pd.DataFrame,
    results: Dict,
    clustering_data: Dict[str, pd.DataFrame],
    output_dir: str,
):
    """Plot cluster proportions for best models with loss-reg pairs as panels."""
    if best_models_df.empty:
        print("No best models data available")
        return

    print("Extracting cluster proportions for best models...")
    cluster_data_rows = []

    for _, row in best_models_df.iterrows():
        split_type = row["split_type"]
        ensemble = row["ensemble"]
        loss_func = row["loss_function"]
        bv_reg_fn = row["bv_reg_function"]
        split_idx = int(row["split"])
        maxent_val = row["maxent_value"]
        bvreg_val = row["bv_reg_value"]
        conv_step = int(row["convergence_step"])

        try:
            history = results[split_type][ensemble][loss_func][bv_reg_fn][maxent_val][bvreg_val][
                split_idx
            ]
        except (KeyError, TypeError):
            continue

        if history is None or not hasattr(history, "states") or not history.states:
            continue

        if conv_step <= 0 or conv_step > len(history.states):
            continue

        state = history.states[conv_step - 1]

        if not hasattr(state, "params") or not hasattr(state.params, "frame_weights"):
            continue

        weights = np.array(state.params.frame_weights)

        if ensemble not in clustering_data:
            continue

        cluster_df = clustering_data[ensemble]
        cluster_assignments = cluster_df["cluster_label"].values

        if len(weights) != len(cluster_assignments):
            continue

        normalized_weights = weights / np.sum(weights)

        unique_clusters = np.unique(cluster_assignments)
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            proportion = np.sum(normalized_weights[cluster_mask])

            cluster_data_rows.append(
                {
                    "ensemble": ensemble,
                    "loss_function": loss_func,
                    "bv_reg_function": bv_reg_fn,
                    "split_type": split_type,
                    "split": split_idx,
                    "cluster_label": int(cluster_id),
                    "proportion": float(proportion),
                }
            )

    if not cluster_data_rows:
        print("No cluster proportion data could be extracted")
        return

    cluster_prop_df = pd.DataFrame(cluster_data_rows)

    csv_path = os.path.join(output_dir, "best_models_cluster_proportions.csv")
    cluster_prop_df.to_csv(csv_path, index=False)
    print(f"Cluster proportions data saved to: {csv_path}")

    cluster_prop_df["proportion_pct"] = cluster_prop_df["proportion"] * 100.0

    ensembles = sorted(cluster_prop_df["ensemble"].unique())

    for ensemble in ensembles:
        df_ensemble = cluster_prop_df[cluster_prop_df["ensemble"] == ensemble]

        split_types = sorted(df_ensemble["split_type"].unique())
        loss_functions = sorted(df_ensemble["loss_function"].unique())
        bv_reg_functions = sorted(df_ensemble["bv_reg_function"].unique())

        # Create grid: rows = split_types, cols = loss_function * bv_reg_function combinations
        n_loss_reg_combos = len(loss_functions) * len(bv_reg_functions)
        nrows = len(split_types)
        ncols = n_loss_reg_combos

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)

        all_clusters = sorted(df_ensemble["cluster_label"].unique())
        n_clusters = len(all_clusters)

        colors = sns.color_palette("tab10", n_colors=max(n_clusters, 10))
        cluster_colors = {cluster: colors[i % len(colors)] for i, cluster in enumerate(all_clusters)}

        for row_idx, split_type in enumerate(split_types):
            col_idx = 0
            for loss_func in loss_functions:
                for bv_reg_fn in bv_reg_functions:
                    if col_idx >= ncols:
                        break

                    ax = axes[row_idx, col_idx]

                    df_combo = df_ensemble[
                        (df_ensemble["split_type"] == split_type)
                        & (df_ensemble["loss_function"] == loss_func)
                        & (df_ensemble["bv_reg_function"] == bv_reg_fn)
                    ]

                    if df_combo.empty:
                        ax.set_visible(False)
                        col_idx += 1
                        continue

                    stats = (
                        df_combo.groupby("cluster_label")["proportion_pct"]
                        .agg(["mean", "sem"])
                        .reset_index()
                    )

                    stats = (
                        stats.set_index("cluster_label")
                        .reindex(all_clusters, fill_value=0)
                        .reset_index()
                    )
                    stats["sem"] = stats["sem"].fillna(0)

                    x_pos = np.arange(len(all_clusters))
                    ax.bar(
                        x_pos,
                        stats["mean"],
                        yerr=stats["sem"],
                        color=[cluster_colors[c] for c in all_clusters],
                        capsize=4,
                        edgecolor="black",
                        linewidth=1.5,
                        alpha=0.85,
                    )

                    ax.set_xlabel("Cluster Label", fontsize=11, fontweight="bold")
                    ax.set_ylabel("Proportion (%)", fontsize=11, fontweight="bold")
                    ax.set_title(
                        f"{split_type} - {loss_func} + {bv_reg_fn}",
                        fontsize=12,
                        fontweight="bold",
                        color=ENSEMBLE_COLOURS.get(ensemble, "black"),
                    )
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels([str(c) for c in all_clusters])
                    ax.set_ylim(0, min(max(stats["mean"].max() * 1.15, 5), 100))
                    ax.grid(axis="y", alpha=0.3, linestyle="--")
                    sns.despine(ax=ax)

                    col_idx += 1

        plt.suptitle(
            f"Cluster Proportions for Best Models - {ensemble}",
            fontsize=22,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        filename = f"cluster_proportions_best_models_{ensemble}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main function for 2D hyperparameter sweep loss analysis."""
    parser = argparse.ArgumentParser(
        description="2D hyperparameter sweep loss analysis with best model selection"
    )
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_quick_test_SIGMA_50_lr0.1_BV_objectve_20250918_171508",
        help="Results directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, derived from results-dir.",
    )
    parser.add_argument(
        "--clustering-dir",
        default="_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters",
        help="Clustering directory",
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
    bv_reg_functions = ["ElasticNet", "ElasticMAE", "KLD_PF", "Work_Fitting", "Work_Magnitude"]

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

    # Load target state ratios
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

    os.makedirs(output_dir, exist_ok=True)

    # Load optimization results
    print("\nLoading optimization results...")
    results = load_all_optimization_results_2d(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        bv_reg_functions=bv_reg_functions,
        EMA=args.ema,
        verbose=True,
    )

    # Extract loss trajectories
    print("\n" + "=" * 60)
    print("EXTRACTING LOSS TRAJECTORIES")
    print("=" * 60)

    df = extract_loss_trajectories_2d(results)

    if len(df) == 0:
        print("No data found!")
        return

    print(f"Extracted {len(df)} data points\n")

    # Save full dataset
    df_path = os.path.join(output_dir, "full_analysis_data.csv")
    df.to_csv(df_path, index=False)
    print(f"Dataset saved to: {df_path}\n")

    # Generate plots
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    print("\nGenerating convergence heatmaps...")
    plot_convergence_heatmaps(df, output_dir)

    print("\nGenerating model score heatmaps...")
    plot_model_score_heatmaps(df, output_dir)

    print("\nGenerating error vs hyperparameters plots...")
    plot_loss_convergence(df, output_dir)

    print("\nGenerating split variability plots...")
    plot_split_variability(df, output_dir)

    # Best model selection
    print("\n" + "=" * 60)
    print("BEST MODEL SELECTION (SEPARATE BY BV REG FUNCTION)")
    print("=" * 60)

    print("\nSelecting best models (separately for each BV reg function)...")
    best_models = select_best_models_2d(df)
    print(f"Selected {len(best_models)} best models\n")

    if len(best_models) > 0:
        # Augment with metrics
        print("Computing KL divergence and JSD-based recovery metrics...")
        best_models_augmented = augment_best_models_with_metrics(
            best_models, results, clustering_data, target_ratios
        )

        if len(best_models_augmented) > 0:
            # Save best models
            best_models_path = os.path.join(output_dir, "best_models.csv")
            best_models_augmented.to_csv(best_models_path, index=False)
            print(f"Best models saved to: {best_models_path}\n")

            # Generate comparison plots
            print("Generating best model comparison plots (loss-reg pairs as panels)...")
            plot_best_model_comparisons(best_models_augmented, output_dir)

            print("\nGenerating cluster proportion plots (loss-reg pairs as panels)...")
            plot_cluster_proportions_for_best_models(
                best_models_augmented, results, clustering_data, output_dir
            )

            # Print summary
            print("\n" + "=" * 60)
            print("BEST MODELS SUMMARY (BY BV REG FUNCTION)")
            print("=" * 60)

            for split_type in sorted(best_models_augmented["split_type"].unique()):
                print(f"\nSplit Type: {split_type}")
                split_best = best_models_augmented[best_models_augmented["split_type"] == split_type]
                
                for bv_reg_fn in sorted(split_best["bv_reg_function"].unique()):
                    print(f"\n  BV Reg Function: {bv_reg_fn}")
                    bv_split_best = split_best[split_best["bv_reg_function"] == bv_reg_fn]
                    
                    for _, row in bv_split_best.iterrows():
                        print(
                            f"    {row['ensemble']} - {row['loss_function']} - Split {row['split']}: "
                            f"MaxEnt={row['maxent_value']:.1f}, BV Reg={row['bv_reg_value']:.2f}, "
                            f"Val Loss={row['val_loss']:.6f}, Recovery={row['recovery_percent']:.1f}%"
                        )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()