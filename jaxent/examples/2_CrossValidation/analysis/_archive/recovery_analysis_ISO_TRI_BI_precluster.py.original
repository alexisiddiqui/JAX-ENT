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
import re
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the base directory to the path to import the HDF5 utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

from jaxent.src.utils.hdf import load_optimization_history_from_file

# Ensemble to clustering subdirectory mapping
ENSEMBLE_CLUSTERING_MAP = {
    "AF2_MSAss": "AF2_MSAss",
    "AF2_filtered": "AF2_Filtered",  # Note: capital F in the directory name
}


def calculate_recovery_JSD(cluster_assignments, weights, target_ratios, state_mapping):
    """
    Compute Jensen-Shannon divergence between observed and target state proportions.
    Returns JS divergence (float). If invalid, returns np.nan.

    Recovery% = (1 - sqrt(JSD)) * 100
    """
    # Invert mapping: state -> cluster ids
    state_to_clusters = {}
    for cluster_id, state_name in state_mapping.items():
        state_to_clusters.setdefault(state_name, []).append(cluster_id)

    # Compute current proportions (weighted)
    current_proportions = {state: 0.0 for state in target_ratios}
    for state_name, cluster_ids in state_to_clusters.items():
        state_mask = cluster_assignments.isin(cluster_ids)
        current_proportions[state_name] = float(np.sum(weights[state_mask.to_numpy()]))

    # Order states consistently with target_ratios
    states = list(target_ratios.keys())
    P = np.array([current_proportions.get(s, 0.0) for s in states], dtype=float)
    Q = np.array([target_ratios.get(s, 0.0) for s in states], dtype=float)

    # Normalize distributions
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

    # Jensen-Shannon divergence (base 2)
    M = 0.5 * (P + Q)

    def kld(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    js = 0.5 * (kld(P, M) + kld(Q, M))
    return float(js), current_proportions


def load_clustering_for_ensemble(ensemble_name: str, clustering_base_dir: str) -> pd.DataFrame:
    """
    Load clustering results for a specific ensemble.

    Args:
        ensemble_name: Name of the ensemble (e.g., 'AF2_MSAss', 'AF2_filtered')
        clustering_base_dir: Base clustering directory (e.g., 'analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters')

    Returns:
        DataFrame with cluster assignments
    """
    if ensemble_name not in ENSEMBLE_CLUSTERING_MAP:
        raise ValueError(
            f"Unknown ensemble: {ensemble_name}. Expected one of {list(ENSEMBLE_CLUSTERING_MAP.keys())}"
        )

    clustering_subdir = ENSEMBLE_CLUSTERING_MAP[ensemble_name]
    # Path is: clustering_base_dir/AF2_MSAss/AF2_MSAss_frame_to_cluster.csv
    clustering_path = os.path.join(
        clustering_base_dir, clustering_subdir, f"{clustering_subdir}_frame_to_cluster.csv"
    )

    if not os.path.exists(clustering_path):
        raise FileNotFoundError(f"Clustering file not found: {clustering_path}")

    print(f"Loading clustering for {ensemble_name} from: {clustering_path}")
    cluster_df = pd.read_csv(clustering_path)

    # Ensure the DataFrame has the expected column (cluster_label)
    if "cluster_label" not in cluster_df.columns:
        raise ValueError(f"Expected 'cluster_label' column in {clustering_path}")

    print(
        f"  Loaded {len(cluster_df)} frames with {cluster_df['cluster_label'].nunique()} unique clusters"
    )

    return cluster_df


def load_all_optimization_results_with_maxent(
    results_dir: str,
    ensembles: List[str],
    loss_functions: List[str],
    num_splits: int = 3,
    maxent_values: List[float] = None,
    EMA: bool = False,
) -> Dict:
    """Load all optimization results from HDF5 files, including maxent values."""
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]

    hdf_pattern = "results_EMA.hdf5" if EMA else "results.hdf5"

    for split_type in split_types:
        results[split_type] = {}
        split_type_dir = os.path.join(results_dir, split_type)

        for ensemble in ensembles:
            results[split_type][ensemble] = {}

            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}

                pattern = f"{ensemble}_{loss_name}_{split_type}_split"
                files = [
                    f
                    for f in os.listdir(split_type_dir)
                    if f.startswith(pattern) and f.endswith(hdf_pattern)
                ]

                for filename in files:
                    match = re.search(r"split(\d{3})_maxent(\d+(?:\.\d+)?)", filename)
                    if match:
                        split_idx = int(match.group(1))
                        maxent_val = float(match.group(2))

                        if maxent_val not in results[split_type][ensemble][loss_name]:
                            results[split_type][ensemble][loss_name][maxent_val] = {}

                        filepath = os.path.join(split_type_dir, filename)

                        try:
                            history = load_optimization_history_from_file(filepath)
                            results[split_type][ensemble][loss_name][maxent_val][split_idx] = (
                                history
                            )
                            print(f"Loaded: {filepath}")
                        except Exception as e:
                            print(f"Failed to load {filepath}: {e}")
                            results[split_type][ensemble][loss_name][maxent_val][split_idx] = None

    return results


def analyze_conformational_recovery_jsd(
    clustering_data: Dict[str, pd.DataFrame],
    results_dict: Dict,
    target_ratios: Dict[str, float],
    state_mapping: Dict[int, str],
) -> pd.DataFrame:
    """
    Analyze conformational recovery using JSD-based method.

    Args:
        clustering_data: Dict mapping ensemble names to their cluster DataFrames
        results_dict: Optimization results containing frame weights
        target_ratios: Target state proportions
        state_mapping: Mapping from cluster ID to state name

    Returns:
        DataFrame with recovery analysis results
    """
    recovery_data = []

    for ensemble_name, cluster_df in clustering_data.items():
        print(f"Analyzing recovery for {ensemble_name}...")

        cluster_assignments = cluster_df["cluster_label"]
        n_frames = len(cluster_assignments)

        # Calculate unweighted (original) recovery
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
                "split_type": "N/A",
                "split": "N/A",
                "maxent_value": 0.0,
                "convergence_step": "N/A",
                "js_divergence": js_div_orig if not np.isnan(js_div_orig) else 0.0,
                "js_distance": np.sqrt(js_div_orig) if not np.isnan(js_div_orig) else 0.0,
                "recovery_percent": recovery_orig,
                **{f"{state}_current": current_props_orig[state] for state in target_ratios},
                **{f"{state}_target": target_ratios[state] for state in target_ratios},
                "total_frames": n_frames,
            }
        )

        # Analyze with optimized frame weights
        for split_type in results_dict:
            if ensemble_name in results_dict[split_type]:
                for loss_name in results_dict[split_type][ensemble_name]:
                    for maxent_val in results_dict[split_type][ensemble_name][loss_name]:
                        for split_idx, history in results_dict[split_type][ensemble_name][
                            loss_name
                        ][maxent_val].items():
                            if history is not None and history.states:
                                for step_idx, state in enumerate(history.states):
                                    if (
                                        hasattr(state.params, "frame_weights")
                                        and state.params.frame_weights is not None
                                    ):
                                        frame_weights = np.array(state.params.frame_weights)

                                        if (
                                            len(frame_weights) == n_frames
                                            and np.sum(frame_weights) > 0
                                        ):
                                            # Normalize weights
                                            normalized_weights = frame_weights / np.sum(
                                                frame_weights
                                            )

                                            # Calculate JSD-based recovery
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
                                                    "split_type": split_type,
                                                    "split": split_idx,
                                                    "maxent_value": maxent_val,
                                                    "convergence_step": step_idx,
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
                                        else:
                                            if len(frame_weights) != n_frames:
                                                print(
                                                    f"    Frame count mismatch: {len(frame_weights)} vs {n_frames}"
                                                )

    return pd.DataFrame(recovery_data)


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
    # Normalize to ensure they sum to 1
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Add small epsilon to avoid log(0)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # Calculate KL divergence: KL(p||q) = Σ p(i) * log(p(i)/q(i))
    return np.sum(p * np.log(p / q))


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Calculate Effective Sample Size (ESS) as 1/sum(weights^2).

    Args:
        weights: Frame weights (should be normalized to sum to 1)

    Returns:
        Effective sample size
    """
    # Normalize weights to sum to 1
    normalized_weights = weights / np.sum(weights)

    # Calculate ESS = 1 / sum(w_i^2)
    ess = 1.0 / np.sum(normalized_weights**2)

    return ess


def extract_frame_weights_kl_with_maxent(
    results: Dict, clustering_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Extract frame weights and calculate KL divergence and ESS including maxent values.

    Args:
        results: Dictionary containing optimization histories.
        clustering_data: Dictionary mapping ensemble names to clustering DataFrames

    Returns:
        DataFrame containing KL divergence and ESS values for analysis.
    """
    data_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
            # Get the number of frames for this ensemble
            if ensemble not in clustering_data:
                continue
            n_frames = len(clustering_data[ensemble])

            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is not None and history.states:
                            for step_idx, state in enumerate(history.states):
                                if (
                                    hasattr(state.params, "frame_weights")
                                    and state.params.frame_weights is not None
                                ):
                                    frame_weights = np.array(state.params.frame_weights)
                                    if len(frame_weights) == 0 or np.sum(frame_weights) == 0:
                                        continue

                                    if len(frame_weights) != n_frames:
                                        continue

                                    uniform_prior = np.ones(len(frame_weights)) / len(frame_weights)
                                    try:
                                        kl_div = kl_divergence(frame_weights, uniform_prior)
                                        ess = effective_sample_size(frame_weights)

                                        data_rows.append(
                                            {
                                                "split_type": split_type,
                                                "ensemble": ensemble,
                                                "loss_function": loss_name,
                                                "maxent_value": maxent_val,
                                                "split": split_idx,
                                                "step": step_idx,
                                                "convergence_threshold_step": step_idx,
                                                "kl_divergence": float(kl_div),
                                                "effective_sample_size": float(ess),
                                                "num_frames": len(frame_weights),
                                                "step_number": state.step
                                                if hasattr(state, "step")
                                                else step_idx,
                                            }
                                        )
                                    except Exception as e:
                                        print(
                                            f"Failed to calculate KL/ESS for {split_type}/{ensemble}_{loss_name}_maxent{maxent_val}_split{split_idx}, step {step_idx}: {e}"
                                        )
                                        continue
    return pd.DataFrame(data_rows)


def plot_recovery_jsd_heatmap(
    recovery_df: pd.DataFrame, convergence_rates: List[float], output_dir: str
):
    """Plot heatmaps of JSD-based recovery across convergence thresholds and maxent values."""
    plt.style.use("seaborn-v0_8-whitegrid")

    recovery_df["convergence_step"] = pd.to_numeric(
        recovery_df["convergence_step"], errors="coerce"
    )

    split_types = recovery_df[recovery_df["split_type"] != "N/A"]["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating recovery heatmaps for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = recovery_df[recovery_df["split_type"] == split_type]

        ensembles = sorted(split_df[split_df["loss_function"] != "Original"]["ensemble"].unique())
        loss_functions = sorted(
            split_df[split_df["loss_function"] != "Original"]["loss_function"].unique()
        )

        if not ensembles or not loss_functions:
            print(f"    No data to plot for split type {split_type}")
            continue

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"JSD-Based Recovery Heatmaps - {split_type} splits",
            fontsize=16,
            fontweight="bold",
        )

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_df = split_df[
                    (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                ]

                if len(combo_df) > 0:
                    maxent_vals = sorted(combo_df["maxent_value"].unique())
                    conv_steps = sorted(combo_df["convergence_step"].unique())
                    conv_steps = [s for s in conv_steps if s > 0]

                    pivot_data = combo_df.pivot_table(
                        values="recovery_percent",
                        index="maxent_value",
                        columns="convergence_step",
                        aggfunc="mean",
                    )

                    if not pivot_data.empty:
                        pivot_data = pivot_data[[c for c in conv_steps if c in pivot_data.columns]]
                        pivot_data = pivot_data.sort_index(ascending=False)

                        col_labels = []
                        for step in pivot_data.columns:
                            if step < len(convergence_rates):
                                col_labels.append(f"{convergence_rates[int(step) - 1]:.0e}")
                            else:
                                col_labels.append(f"Step {step}")

                        sns.heatmap(
                            pivot_data,
                            annot=True,
                            fmt=".1f",
                            cmap="RdYlGn",
                            vmin=0,
                            vmax=100,
                            cbar_kws={"label": "Recovery (%)"},
                            ax=ax,
                        )

                        ax.set_title(f"{ensemble} - {loss_func}")
                        ax.set_xlabel("Convergence Threshold")
                        ax.set_ylabel("MaxEnt Value")
                        ax.set_xticklabels(col_labels, rotation=45, ha="right")
                        ax.set_yticklabels([f"{v:.0f}" for v in pivot_data.index], rotation=0)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No data available",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"{ensemble} - {loss_func}")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{ensemble} - {loss_func}")

        plt.tight_layout()
        plt.savefig(
            os.path.join(split_output_dir, "recovery_jsd_heatmap.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_volcano_kl_recovery(kl_ess_df, recovery_df, convergence_rates, output_dir):
    """
    Plot volcano plot with KL divergence vs recovery fold change.

    Args:
        kl_ess_df (pd.DataFrame): KL divergence and ESS analysis results
        recovery_df (pd.DataFrame): Recovery analysis results
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Creating volcano plot...")
    print(f"KL/ESS DataFrame shape: {kl_ess_df.shape}")
    print(f"Recovery DataFrame shape: {recovery_df.shape}")

    # Filter out Original data from recovery_df
    recovery_filtered = recovery_df[recovery_df["loss_function"] != "Original"].copy()

    if len(recovery_filtered) == 0:
        print("No recovery data available for volcano plot")
        return

    if len(kl_ess_df) == 0:
        print("No KL divergence data available for volcano plot")
        return

    # Convert convergence steps to numeric
    kl_ess_df = kl_ess_df.copy()
    recovery_filtered = recovery_filtered.copy()

    kl_conv_col = "convergence_threshold_step"
    recovery_conv_col = "convergence_step"

    kl_ess_df[kl_conv_col] = pd.to_numeric(kl_ess_df[kl_conv_col], errors="coerce")
    recovery_filtered[recovery_conv_col] = pd.to_numeric(
        recovery_filtered[recovery_conv_col], errors="coerce"
    )

    # Merge columns
    merge_columns = ["split_type", "ensemble", "loss_function", "maxent_value", "split"]

    left_cols = merge_columns + [kl_conv_col]
    right_cols = merge_columns + [recovery_conv_col]

    merged_df = pd.merge(
        kl_ess_df, recovery_filtered, left_on=left_cols, right_on=right_cols, how="inner"
    )

    if len(merged_df) == 0:
        print("No exact matches found, trying to merge by taking final convergence steps...")

        kl_final = kl_ess_df.groupby(merge_columns).last().reset_index()
        recovery_final = recovery_filtered.groupby(merge_columns).last().reset_index()

        merged_df = pd.merge(
            kl_final, recovery_final, on=merge_columns, how="inner", suffixes=("_kl", "_recovery")
        )

        if recovery_conv_col in merged_df.columns:
            merged_df["plot_convergence_step"] = merged_df[recovery_conv_col]
        elif f"{recovery_conv_col}_recovery" in merged_df.columns:
            merged_df["plot_convergence_step"] = merged_df[f"{recovery_conv_col}_recovery"]
        else:
            merged_df["plot_convergence_step"] = 1
    else:
        merged_df["plot_convergence_step"] = merged_df[recovery_conv_col]

    if len(merged_df) == 0:
        print("No matching data found for volcano plot after all merge attempts")
        return

    print(f"Merged {len(merged_df)} data points for volcano plot")

    # Get unweighted data for baseline
    unweighted_recovery = recovery_df[recovery_df["loss_function"] == "Original"].copy()
    print(f"Found {len(unweighted_recovery)} unweighted baseline records")

    # Create baseline recovery dictionary per ensemble
    baseline_recoveries = {}
    for ensemble in unweighted_recovery["ensemble"].unique():
        baseline_data = unweighted_recovery[unweighted_recovery["ensemble"] == ensemble]
        if not baseline_data.empty:
            baseline_recoveries[ensemble] = baseline_data["recovery_percent"].iloc[0]
            print(f"Baseline recovery for {ensemble}: {baseline_recoveries[ensemble]:.2f}%")
        else:
            baseline_recoveries[ensemble] = 0.0

    # Calculate fold change relative to unweighted recovery
    fold_change_data = []

    for _, row in merged_df.iterrows():
        ensemble = row["ensemble"]
        baseline_recovery = baseline_recoveries.get(ensemble, 0.0)

        if baseline_recovery > 0:
            current_recovery = row["recovery_percent"]
            fold_change = current_recovery / baseline_recovery
            log2_fold_change = np.log2(fold_change) if fold_change > 0 else 0
        else:
            fold_change = 1.0
            log2_fold_change = 0.0

        fold_change_data.append(
            {
                **row.to_dict(),
                "recovery_fold_change": fold_change,
                "log2_fold_change": log2_fold_change,
                "baseline_recovery": baseline_recovery,
                "current_recovery": current_recovery
                if baseline_recovery > 0
                else row.get("recovery_percent", 0),
            }
        )

    if not fold_change_data:
        print("No fold change data could be calculated")
        return

    volcano_df = pd.DataFrame(fold_change_data)
    print(f"Calculated fold changes for {len(volcano_df)} data points")

    # Target recovery is 100% (perfect match to target ratios)
    target_recovery = 100.0
    target_fold_changes = {}

    print("Calculating target fold changes based on 100% recovery...")

    for ensemble in volcano_df["ensemble"].unique():
        unweighted_recovery = baseline_recoveries.get(ensemble, 0.0)

        if unweighted_recovery > 0:
            target_fold_change = target_recovery / unweighted_recovery
            target_log2_fold_change = np.log2(target_fold_change)
            target_fold_changes[ensemble] = target_log2_fold_change
            print(
                f"  {ensemble}: Target fold change = {target_fold_change:.3f} (log2 = {target_log2_fold_change:.3f})"
            )
        else:
            target_fold_changes[ensemble] = 0

    # Create figure with subplots
    split_types = volcano_df["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating volcano plot for split type: {split_type}")
        split_data = volcano_df[volcano_df["split_type"] == split_type]

        if len(split_data) == 0:
            continue

        ensembles = sorted(split_data["ensemble"].unique())
        loss_functions = sorted(split_data["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"Volcano Plot: KL Divergence vs Recovery Fold Change - {split_type}",
            fontsize=16,
            fontweight="bold",
        )

        # Create colormap for MaxEnt values
        maxent_values = sorted(split_data["maxent_value"].unique())
        if len(maxent_values) > 1:
            log_maxent = np.log10([max(1, val) for val in maxent_values])
            norm = plt.Normalize(vmin=min(log_maxent), vmax=max(log_maxent))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        # Size mapping for convergence rates
        conv_steps = sorted(split_data["plot_convergence_step"].dropna().unique())
        max_size = 150
        min_size = 30

        size_map = {}
        for i, step in enumerate(conv_steps):
            step_int = int(step)
            if step_int > 0 and step_int <= len(convergence_rates):
                size = min_size + (i / max(1, len(conv_steps) - 1)) * (max_size - min_size)
                size_map[step] = size
            else:
                size_map[step] = min_size

        # Marker styles for replicates
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "P", "X"]

        # Calculate global axis limits
        all_x_data = split_data["log2_fold_change"]
        all_y_data = split_data["kl_divergence"]

        x_margin = (all_x_data.max() - all_x_data.min()) * 0.05
        y_margin = (all_y_data.max() - all_y_data.min()) * 0.05

        global_xlim = [all_x_data.min() - x_margin, all_x_data.max() + x_margin]
        global_ylim = [all_y_data.min() - y_margin, all_y_data.max() + y_margin]

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_data = split_data[
                    (split_data["ensemble"] == ensemble)
                    & (split_data["loss_function"] == loss_func)
                ]

                if len(combo_data) > 0:
                    splits = sorted(combo_data["split"].unique())

                    # Plot each split with different marker
                    for k, split_idx in enumerate(splits):
                        split_data_subset = combo_data[combo_data["split"] == split_idx]

                        if len(split_data_subset) > 0:
                            x_sub = split_data_subset["log2_fold_change"]
                            y_sub = split_data_subset["kl_divergence"]

                            # Colors based on log MaxEnt value
                            colors_sub = []
                            for maxent_val in split_data_subset["maxent_value"]:
                                if len(maxent_values) > 1:
                                    colors_sub.append(cmap(norm(np.log10(max(1, maxent_val)))))
                                else:
                                    colors_sub.append("blue")

                            # Sizes based on convergence step
                            sizes_sub = [
                                size_map.get(step, min_size)
                                for step in split_data_subset["plot_convergence_step"]
                            ]

                            scatter = ax.scatter(
                                x_sub,
                                y_sub,
                                c=colors_sub,
                                s=sizes_sub,
                                marker=markers[k % len(markers)],
                                alpha=0.7,
                                edgecolors="black",
                                linewidth=0.5,
                                label=f"Replicate {split_idx}",
                            )

                    # Add reference lines
                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

                    # Add target fold change line
                    if ensemble in target_fold_changes:
                        target_x = target_fold_changes[ensemble]
                        ax.axvline(
                            x=target_x, color="orange", linestyle=":", alpha=0.8, linewidth=2
                        )

                        ax.text(
                            target_x,
                            ax.get_ylim()[1] * 0.95,
                            "Target\n(100%)",
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    # Set axis limits
                    ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

                    # Add quadrant labels
                    ax.text(
                        0.98,
                        0.98,
                        "High KL\nHigh Recovery",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                        fontsize=8,
                    )

                    ax.text(
                        0.02,
                        0.98,
                        "High KL\nLow Recovery",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                        fontsize=8,
                    )

                    ax.set_xlabel("Log2 Fold Change (Recovery vs Unweighted Baseline)")
                    ax.set_ylabel("KL Divergence")
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.grid(True, alpha=0.3)

                    # Add legend for first subplot
                    if i == 0 and j == 0:
                        legend_elements = []

                        for k, split_idx in enumerate(splits[:6]):
                            legend_elements.append(
                                plt.Line2D(
                                    [0],
                                    [0],
                                    marker=markers[k % len(markers)],
                                    color="w",
                                    markerfacecolor="gray",
                                    markersize=8,
                                    label=f"Replicate {split_idx}",
                                    markeredgecolor="black",
                                    markeredgewidth=0.5,
                                )
                            )

                        legend_elements.append(
                            plt.Line2D(
                                [0], [0], color="red", linestyle="--", label="No Fold Change"
                            )
                        )
                        legend_elements.append(
                            plt.Line2D(
                                [0], [0], color="orange", linestyle=":", label="Target (100%)"
                            )
                        )

                        ax.legend(
                            handles=legend_elements,
                            bbox_to_anchor=(1.05, 1),
                            loc="upper left",
                            fontsize=8,
                        )
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{ensemble} - {loss_func}")

        plt.tight_layout()

        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(split_output_dir, "volcano_plot_kl_recovery.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Save volcano plot dataset
    volcano_df["target_log2_fold_change"] = volcano_df["ensemble"].map(target_fold_changes)
    volcano_df_path = os.path.join(output_dir, "volcano_plot_data.csv")
    volcano_df.to_csv(volcano_df_path, index=False)
    print(f"Volcano plot dataset saved to: {volcano_df_path}")

    return volcano_df


def plot_volcano_kl_recovery_averaged(kl_ess_df, recovery_df, convergence_rates, output_dir):
    """
    Plot averaged volcano plot with KL divergence vs recovery fold change, with error bars.

    Args:
        kl_ess_df (pd.DataFrame): KL divergence and ESS analysis results
        recovery_df (pd.DataFrame): Recovery analysis results
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    print("Creating averaged volcano plot with error bars...")

    # First get the processed volcano data
    volcano_df = plot_volcano_kl_recovery(kl_ess_df, recovery_df, convergence_rates, output_dir)

    if volcano_df is None or len(volcano_df) == 0:
        print("No volcano data available for averaged plot")
        return

    # Calculate averages across replicates
    grouping_cols = [
        "split_type",
        "ensemble",
        "loss_function",
        "maxent_value",
        "plot_convergence_step",
    ]

    averaged_df = (
        volcano_df.groupby(grouping_cols)
        .agg(
            {
                "log2_fold_change": ["mean", "std", "count"],
                "kl_divergence": ["mean", "std", "count"],
                "current_recovery": ["mean", "std"],
                "baseline_recovery": "first",
                "target_log2_fold_change": "first",
            }
        )
        .reset_index()
    )

    # Flatten column names
    averaged_df.columns = [
        "split_type",
        "ensemble",
        "loss_function",
        "maxent_value",
        "plot_convergence_step",
        "log2_fold_change_mean",
        "log2_fold_change_std",
        "log2_fold_change_count",
        "kl_divergence_mean",
        "kl_divergence_std",
        "kl_divergence_count",
        "current_recovery_mean",
        "current_recovery_std",
        "baseline_recovery",
        "target_log2_fold_change",
    ]

    # Fill NaN std with 0
    averaged_df["log2_fold_change_std"] = averaged_df["log2_fold_change_std"].fillna(0)
    averaged_df["kl_divergence_std"] = averaged_df["kl_divergence_std"].fillna(0)
    averaged_df["current_recovery_std"] = averaged_df["current_recovery_std"].fillna(0)

    print(f"Averaged across replicates: {len(averaged_df)} unique parameter combinations")

    # Create plots
    split_types = averaged_df["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating averaged volcano plot for split type: {split_type}")
        split_data = averaged_df[averaged_df["split_type"] == split_type]

        if len(split_data) == 0:
            continue

        ensembles = sorted(split_data["ensemble"].unique())
        loss_functions = sorted(split_data["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"Averaged Volcano Plot: KL Divergence vs Recovery Fold Change - {split_type}",
            fontsize=16,
            fontweight="bold",
        )

        # Colormap and sizes
        maxent_values = sorted(split_data["maxent_value"].unique())
        if len(maxent_values) > 1:
            log_maxent = np.log10([max(1, val) for val in maxent_values])
            norm = plt.Normalize(vmin=min(log_maxent), vmax=max(log_maxent))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.viridis

        # Calculate global limits
        all_x_data = split_data["log2_fold_change_mean"]
        all_y_data = split_data["kl_divergence_mean"]

        x_with_error = np.concatenate(
            [
                all_x_data + split_data["log2_fold_change_std"],
                all_x_data - split_data["log2_fold_change_std"],
            ]
        )
        y_with_error = np.concatenate(
            [
                all_y_data + split_data["kl_divergence_std"],
                all_y_data - split_data["kl_divergence_std"],
            ]
        )

        x_margin = (x_with_error.max() - x_with_error.min()) * 0.05
        y_margin = (y_with_error.max() - y_with_error.min()) * 0.05

        global_xlim = [x_with_error.min() - x_margin, x_with_error.max() + x_margin]
        global_ylim = [y_with_error.min() - y_margin, y_with_error.max() + y_margin]

        # Size mapping
        conv_steps = sorted(split_data["plot_convergence_step"].dropna().unique())
        max_size = 150
        min_size = 30

        size_map = {}
        for i, step in enumerate(conv_steps):
            step_int = int(step)
            if step_int > 0 and step_int <= len(convergence_rates):
                size = min_size + (i / max(1, len(conv_steps) - 1)) * (max_size - min_size)
                size_map[step] = size
            else:
                size_map[step] = min_size

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_data = split_data[
                    (split_data["ensemble"] == ensemble)
                    & (split_data["loss_function"] == loss_func)
                ]

                if len(combo_data) > 0:
                    x_vals = combo_data["log2_fold_change_mean"]
                    y_vals = combo_data["kl_divergence_mean"]
                    x_errs = combo_data["log2_fold_change_std"]
                    y_errs = combo_data["kl_divergence_std"]

                    # Colors based on MaxEnt
                    colors_vals = []
                    for maxent_val in combo_data["maxent_value"]:
                        if len(maxent_values) > 1:
                            colors_vals.append(cmap(norm(np.log10(max(1, maxent_val)))))
                        else:
                            colors_vals.append("blue")

                    # Sizes
                    sizes_vals = [
                        size_map.get(step, min_size) for step in combo_data["plot_convergence_step"]
                    ]

                    # Plot points with error bars
                    scatter = ax.scatter(
                        x_vals,
                        y_vals,
                        c=colors_vals,
                        s=sizes_vals,
                        marker="o",
                        alpha=0.8,
                        edgecolors="black",
                        linewidth=0.5,
                        zorder=3,
                    )

                    ax.errorbar(
                        x_vals,
                        y_vals,
                        xerr=x_errs,
                        yerr=y_errs,
                        fmt="none",
                        ecolor="black",
                        alpha=0.5,
                        capsize=2,
                        capthick=1,
                        zorder=2,
                    )

                    # Add reference lines
                    ax.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=2)

                    # Target line
                    target_log2_fold_change = combo_data["target_log2_fold_change"].iloc[0]
                    if pd.notna(target_log2_fold_change):
                        ax.axvline(
                            x=target_log2_fold_change,
                            color="orange",
                            linestyle=":",
                            alpha=0.8,
                            linewidth=2,
                        )
                        ax.text(
                            target_log2_fold_change,
                            ax.get_ylim()[1] * 0.95,
                            "Target\n(100%)",
                            ha="center",
                            va="top",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        )

                    # Set limits
                    ax.set_xlim(global_xlim)
                    ax.set_ylim(global_ylim)

                    # Quadrant labels
                    ax.text(
                        0.98,
                        0.98,
                        "High KL\nHigh Recovery",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                        fontsize=8,
                    )

                    ax.text(
                        0.02,
                        0.98,
                        "High KL\nLow Recovery",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
                        fontsize=8,
                    )

                    ax.set_xlabel("Log2 Fold Change (Recovery vs Unweighted Baseline)")
                    ax.set_ylabel("KL Divergence")
                    ax.set_title(f"{ensemble} - {loss_func}")
                    ax.grid(True, alpha=0.3)

        plt.tight_layout()

        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(split_output_dir, "volcano_plot_kl_recovery_averaged.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Save averaged dataset
    averaged_df_path = os.path.join(output_dir, "volcano_plot_averaged_data.csv")
    averaged_df.to_csv(averaged_df_path, index=False)
    print(f"Averaged volcano plot dataset saved to: {averaged_df_path}")


def main():
    """Main function to run the MoPrP recovery analysis."""
    parser = argparse.ArgumentParser(description="MoPrP recovery analysis with JSD-based metrics.")
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_quick_test_splits__20250918_171508",
        help="Results directory",
    )
    parser.add_argument(
        "--clustering-dir",
        default="_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters",
        help="Clustering directory containing AF2_MSAss and AF2_Filtered subdirectories",
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

    # Define parameters
    ensembles = ["AF2_MSAss", "AF2_filtered"]
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
    maxent_values = [
        1,
        2,
        5,
        10,
        50,
        100,
        500,
        1000,
        10000,
        100000,
        1000000,
        10000000,
        100000000,
        1000000000,
    ]
    convergence_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # State mapping for MoPrP
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
        output_dir = os.path.join(script_dir, "_analysis" + base_name)

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

    # Load clustering data for each ensemble
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
    results = load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        maxent_values=maxent_values,
        EMA=args.ema,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Analyze recovery
    print("\n" + "=" * 60)
    print("ANALYZING CONFORMATIONAL RECOVERY WITH JSD")
    print("=" * 60)

    recovery_df = analyze_conformational_recovery_jsd(
        clustering_data, results, target_ratios, state_mapping
    )

    if len(recovery_df) > 0:
        print(f"\nExtracted {len(recovery_df)} recovery data points")

        # Save recovery data
        recovery_path = os.path.join(output_dir, "recovery_jsd_data.csv")
        recovery_df.to_csv(recovery_path, index=False)
        print(f"Recovery data saved to: {recovery_path}")

        # Generate recovery plots
        print("\nGenerating recovery plots...")
        plot_recovery_jsd_heatmap(recovery_df, convergence_rates, output_dir)

        # Extract KL divergence and ESS
        print("\n" + "=" * 60)
        print("EXTRACTING KL DIVERGENCE AND ESS")
        print("=" * 60)

        kl_ess_df = extract_frame_weights_kl_with_maxent(results, clustering_data)

        if len(kl_ess_df) > 0:
            print(f"\nExtracted {len(kl_ess_df)} KL divergence and ESS data points")

            # Save KL/ESS data
            kl_ess_path = os.path.join(output_dir, "kl_divergence_ess_data.csv")
            kl_ess_df.to_csv(kl_ess_path, index=False)
            print(f"KL divergence and ESS data saved to: {kl_ess_path}")

            # Generate volcano plots
            print("\nGenerating volcano plots...")
            plot_volcano_kl_recovery(kl_ess_df, recovery_df, convergence_rates, output_dir)

            print("\nGenerating averaged volcano plots...")
            plot_volcano_kl_recovery_averaged(kl_ess_df, recovery_df, convergence_rates, output_dir)
        else:
            print("No KL divergence data available, skipping volcano plots")

        # Print summary
        print("\n" + "=" * 60)
        print("RECOVERY SUMMARY")
        print("=" * 60)

        for ensemble in ensembles:
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
