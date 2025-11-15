"""
This script loads the optimization histories of models over both loss functions
and performs comprehensive analyses including convergence-maxent heatmaps, model scoring,
and best model selection with bar chart comparisons.

Updated to use JSD-based recovery calculation for MoPrP system.
Integrated with publication-ready plotting style from TeaA analysis.
"""

import argparse
import glob
import json
import os
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

# Import the HDF5 loading functions from the provided script
from jaxent.src.utils.hdf import load_optimization_history_from_file

# Publication-ready style configuration (from TeaA)
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
    "AF2_filtered": "AF2_Filtered",  # Note: capital F in the directory name
}

# MoPrP ensemble coloring scheme (retained from original)
ENSEMBLE_COLOURS = {
    "AF2_MSAss": "RoyalBlue",  # Blue
    "AF2_filtered": "Cyan",   # Cyan
}

# State mapping for MoPrP
STATE_MAPPING = {
    0: "Folded",
    1: "PUF1",
    2: "PUF2",
}

# Loss function markers (from TeaA)
LOSS_MARKERS = {"mcMSE": "o", "MSE": "s", "Sigma_MSE": "^"}

# Split type coloring scheme (from TeaA)
SPLIT_TYPE_COLOURS = {
    "r": "fuchsia",
    "s": "black",
    "R3": "green",
    "sequence_cluster": "green",  # Non-Redundant
    "Sp": "grey",
}

SPLIT_NAME_MAPPING = {
    "r": "Random",
    "s": "Sequence",
    "R3": "Non-Redundant",
    "sequence_cluster": "Non-Redundant",
    "Sp": "Spatial",
}


def load_all_optimization_results(
    results_dir: str,
    split_type: str = None,
    ensembles: List[str] = ["AF2_MSAss", "AF2_filtered"],
    loss_functions: List[str] = ["mcMSE", "MSE", "Sigma_MSE"],
    num_splits: int = 3,
    EMA: bool = False,
    maxent_values: List[float] = None,
) -> Dict:
    """
    Load all optimization results from HDF5 files for a specific split type.

    Args:
        results_dir: Directory containing the HDF5 result files
        split_type: Name of the split type subdirectory (if None, loads from results_dir directly)
        ensembles: List of ensemble names
        loss_functions: List of loss function names
        num_splits: Number of data splits
        EMA: Use EMA results (results_EMA.hdf5) if True
        maxent_values: List of maxent values to look for (if None, auto-detect)

    Returns:
        Dictionary containing loaded optimization histories organized by ensemble, loss, split, and maxent
    """
    results = {}

    # Determine the actual directory to load from
    if split_type:
        load_dir = os.path.join(results_dir, split_type)
    else:
        load_dir = results_dir

    if not os.path.exists(load_dir):
        print(f"Directory not found: {load_dir}")
        return results

    # Determine file suffix based on EMA flag
    if EMA:
        hdf_suffix = "results_EMA.hdf5"
    else:
        hdf_suffix = "results.hdf5"

    for ensemble in ensembles:
        results[ensemble] = {}

        for loss_name in loss_functions:
            results[ensemble][loss_name] = {}

            for split_idx in range(num_splits):
                # Pattern to match files with maxent values
                if split_type:
                    pattern = f"{ensemble}_{loss_name}_{split_type}_split{split_idx:03d}_maxent*_{hdf_suffix}"
                else:
                    pattern = f"{ensemble}_{loss_name}_split{split_idx:03d}_maxent*_{hdf_suffix}"

                search_path = os.path.join(load_dir, pattern)
                matching_files = glob.glob(search_path)

                if matching_files:
                    # Store all maxent results for this split
                    results[ensemble][loss_name][split_idx] = {}

                    for filepath in matching_files:
                        # Extract maxent value from filename
                        filename = os.path.basename(filepath)
                        try:
                            # Parse maxent value from filename
                            maxent_str = filename.split("_maxent")[1].split("_")[0]
                            maxent_val = float(maxent_str)

                            history = load_optimization_history_from_file(filepath)
                            results[ensemble][loss_name][split_idx][maxent_val] = history
                            print(f"Loaded: {filename}")
                        except Exception as e:
                            print(f"Failed to load {filename}: {e}")
                else:
                    print(f"No files found matching pattern: {pattern}")
                    results[ensemble][loss_name][split_idx] = None

    return results


def extract_loss_trajectories(results: Dict, split_type: str = None) -> pd.DataFrame:
    """
    Extract loss trajectories from optimization results.

    Args:
        results: Dictionary containing optimization histories
        split_type: Name of the split type (for labeling)

    Returns:
        DataFrame containing loss trajectories for analysis
    """
    data_rows = []

    for ensemble in results:
        for loss_name in results[ensemble]:
            for split_idx in results[ensemble][loss_name]:
                split_results = results[ensemble][loss_name][split_idx]

                if split_results is None:
                    continue

                # Handle both old format (direct history) and new format (dict of maxent histories)
                if isinstance(split_results, dict):
                    # New format with maxent values
                    for maxent_val, history in split_results.items():
                        if history is not None and history.states:
                            # Extract data from ALL convergence steps, not just final
                            for step_idx, state in enumerate(history.states):
                                if state.losses is not None:
                                    data_rows.append(
                                        {
                                            "ensemble": ensemble,
                                            "loss_function": loss_name,
                                            "split": split_idx,
                                            "split_type": split_type,
                                            "maxent_value": maxent_val,
                                            "convergence_step": step_idx + 1,  # 1-indexed
                                            "train_loss": float(state.losses.train_losses[0]),
                                            "val_loss": float(state.losses.val_losses[0]),
                                            "step_number": state.step,
                                        }
                                    )
                else:
                    # Old format - single history with multiple states
                    history = split_results
                    if history is not None and history.states:
                        for step_idx, state in enumerate(history.states):
                            if state.losses is not None:
                                data_rows.append(
                                    {
                                        "ensemble": ensemble,
                                        "loss_function": loss_name,
                                        "split": split_idx,
                                        "split_type": split_type,
                                        "step": step_idx,
                                        "convergence_step": step_idx + 1,
                                        "train_loss": float(state.losses.train_losses[0]),
                                        "val_loss": float(state.losses.val_losses[0]),
                                        "step_number": state.step,
                                    }
                                )

    return pd.DataFrame(data_rows)


def plot_convergence_maxent_heatmaps(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None
):
    """
    Plot heatmaps of training and validation error across convergence thresholds and maxent values.
    Creates separate figures for training and validation errors.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
        split_type: Name of the split type (for titles)
    """
    # Filter to only data with maxent values
    if "maxent_value" not in df.columns:
        print("No maxent_value column found in data")
        return

    df_maxent = df[df["maxent_value"] > 0].copy()

    if df_maxent.empty:
        print("No data with maxent values found")
        return

    split_types = df_maxent["split_type"].unique() if split_type is None else [split_type]

    for stype in split_types:
        print(f"  Creating convergence-maxent heatmaps for split type: {stype}")
        split_output_dir = os.path.join(output_dir, stype) if stype else output_dir
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df_maxent[df_maxent["split_type"] == stype] if stype else df_maxent

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        # Create separate figures for training and validation errors
        for error_type in ["train_loss", "val_loss"]:
            error_label = "Training Error" if error_type == "train_loss" else "Validation Error"

            fig, axes = plt.subplots(
                len(ensembles),
                len(loss_functions),
                figsize=(8 * len(loss_functions), 6 * len(ensembles)),
                squeeze=False,
            )

            fig.suptitle(
                f"{error_label} Heatmap: Convergence vs MaxEnt{' - ' + stype if stype else ''}",
                fontsize=22,
                fontweight="bold",
            )

            for i, ensemble in enumerate(ensembles):
                for j, loss_func in enumerate(loss_functions):
                    ax = axes[i, j]

                    combo_df = split_df[
                        (split_df["ensemble"] == ensemble)
                        & (split_df["loss_function"] == loss_func)
                    ]

                    if len(combo_df) > 0:
                        # Average across splits for each maxent/convergence combination
                        pivot_data = combo_df.pivot_table(
                            values=error_type,
                            index="maxent_value",
                            columns="convergence_step",
                            aggfunc="mean",
                        )

                        # Filter to valid convergence steps
                        valid_steps = [s for s in pivot_data.columns if s <= len(convergence_rates)]
                        pivot_data = pivot_data[valid_steps]

                        if not pivot_data.empty:
                            # Sort by maxent (descending)
                            pivot_data = pivot_data.sort_index(ascending=False)

                            # Create column labels with convergence rates
                            col_labels = []
                            for step in pivot_data.columns:
                                if step - 1 < len(convergence_rates):
                                    col_labels.append(f"{convergence_rates[int(step) - 1]:.0e}")
                                else:
                                    col_labels.append(f"Step {step}")

                            # Plot heatmap with log scale for values
                            sns.heatmap(
                                np.log10(pivot_data),
                                annot=False,
                                cmap="viridis",
                                cbar_kws={"label": f"log10({error_label})"},
                                ax=ax,
                            )

                            ax.set_title(f"{ensemble} - {loss_func}", fontweight="bold")
                            ax.set_xlabel("Convergence Threshold", fontweight="bold")
                            ax.set_ylabel("MaxEnt Value", fontweight="bold")
                            ax.set_xticklabels(col_labels, rotation=45, ha="right")
                            ax.set_yticklabels([f"{v:.0e}" for v in pivot_data.index], rotation=0)
                            sns.despine(ax=ax)
                        else:
                            ax.text(
                                0.5,
                                0.5,
                                "No data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"{ensemble} - {loss_func}")
                    else:
                        ax.text(
                            0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
                        )
                        ax.set_title(f"{ensemble} - {loss_func}")

            plt.tight_layout()
            filename = (
                f"{error_type}_convergence_maxent_heatmap_{stype}.png"
                if stype
                else f"{error_type}_convergence_maxent_heatmap.png"
            )
            plt.savefig(os.path.join(split_output_dir, filename), dpi=300, bbox_inches="tight")
            plt.close()


def compute_model_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model scores as -log10(validation error) and optionally include a
    maxent/kl_divergence bonus when kl_divergence is available and valid.

    Args:
        df: DataFrame containing loss data

    Returns:
        DataFrame with added score column
    """
    df = df.copy()

    # Safeguard against zero or negative validation losses
    eps = 1e-300
    val_loss = df.get("val_loss", pd.Series(dtype=float))
    val_loss = val_loss.fillna(np.nan).clip(lower=eps)

    train_loss = df.get("train_loss", pd.Series(dtype=float))
    train_loss = train_loss.fillna(np.nan).clip(lower=eps)

    # Base score: negative log10 of validation loss
    base_score = -np.log10(val_loss) - (0.9 * np.log10(train_loss))

    # Initialize model_score with base_score
    df["model_score"] = base_score

    # If kl_divergence and maxent_value are present, add the extra term where valid
    if "kl_divergence" in df.columns and "maxent_value" in df.columns:
        kl = df["kl_divergence"].replace(0, np.nan)  # avoid division by zero
        maxent = df["maxent_value"]

        # Compute bonus only where kl is finite and maxent is finite
        bonus = pd.Series(0.0, index=df.index)
        valid_mask = kl.notna() & np.isfinite(kl) & maxent.notna() & np.isfinite(maxent)

        if valid_mask.any():
            bonus.loc[valid_mask] = kl.loc[valid_mask] / maxent.loc[valid_mask]

        # Add bonus to model_score
        df["model_score"] = df["model_score"]

    return df


def plot_model_score_heatmaps(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None
):
    """
    Plot heatmaps of model scores (-log(val_loss)) averaged over split replicates.

    Args:
        df: DataFrame containing loss data with scores
        convergence_rates: List of convergence rates
        output_dir: Directory to save plots
        split_type: Name of the split type
    """
    df_scored = compute_model_scores(df)
    df_maxent = df_scored[df_scored["maxent_value"] > 0].copy()

    if df_maxent.empty:
        print("No data with maxent values for scoring")
        return

    split_types = df_maxent["split_type"].unique() if split_type is None else [split_type]

    for stype in split_types:
        print(f"  Creating model score heatmaps for split type: {stype}")
        split_output_dir = os.path.join(output_dir, stype) if stype else output_dir
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = df_maxent[df_maxent["split_type"] == stype] if stype else df_maxent

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
            squeeze=False,
        )

        fig.suptitle(
            f"Model Scores: -log10(Val Error){' - ' + stype if stype else ''}",
            fontsize=22,
            fontweight="bold",
        )

        for i, ensemble in enumerate(ensembles):
            for j, loss_func in enumerate(loss_functions):
                ax = axes[i, j]

                combo_df = split_df[
                    (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                ]

                if len(combo_df) > 0:
                    # Average scores across splits
                    pivot_data = combo_df.pivot_table(
                        values="model_score",
                        index="maxent_value",
                        columns="convergence_step",
                        aggfunc="mean",
                    )

                    valid_steps = [s for s in pivot_data.columns if s <= len(convergence_rates)]
                    pivot_data = pivot_data[valid_steps]

                    if not pivot_data.empty:
                        pivot_data = pivot_data.sort_index(ascending=False)

                        col_labels = []
                        for step in pivot_data.columns:
                            if step - 1 < len(convergence_rates):
                                col_labels.append(f"{convergence_rates[int(step) - 1]:.0e}")
                            else:
                                col_labels.append(f"Step {step}")

                        sns.heatmap(
                            pivot_data,
                            annot=False,
                            cmap="RdYlGn",
                            cbar_kws={"label": "-log10(Val Error)"},
                            ax=ax,
                        )

                        ax.set_title(f"{ensemble} - {loss_func}", fontweight="bold")
                        ax.set_xlabel("Convergence Threshold", fontweight="bold")
                        ax.set_ylabel("MaxEnt Value", fontweight="bold")
                        ax.set_xticklabels(col_labels, rotation=45, ha="right")
                        ax.set_yticklabels([f"{v:.0e}" for v in pivot_data.index], rotation=0)
                        sns.despine(ax=ax)
                    else:
                        ax.text(
                            0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
                        )
                        ax.set_title(f"{ensemble} - {loss_func}")
                else:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{ensemble} - {loss_func}")

        plt.tight_layout()
        filename = f"model_score_heatmap_{stype}.png" if stype else "model_score_heatmap.png"
        plt.savefig(os.path.join(split_output_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()


def select_best_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select best scoring models for each split type, ensemble, loss function, and split replicate.

    Args:
        df: DataFrame containing loss data with scores

    Returns:
        DataFrame containing only the best models for each condition
    """
    df_scored = compute_model_scores(df)
    df_maxent = df_scored[df_scored["maxent_value"] > 0].copy()

    if df_maxent.empty:
        print("No data with maxent values for model selection")
        return pd.DataFrame()

    # Group by split_type, ensemble, loss_function, and split
    # Select the row with the highest model_score for each group
    best_models = df_maxent.loc[
        df_maxent.groupby(["split_type", "ensemble", "loss_function", "split"])[
            "model_score"
        ].idxmax()
    ]

    return best_models


def load_clustering_for_ensemble(ensemble_name: str, clustering_base_dir: str) -> pd.DataFrame:
    """
    Load clustering results for a specific ensemble.

    Args:
        ensemble_name: Name of the ensemble (e.g., 'AF2_MSAss', 'AF2_filtered')
        clustering_base_dir: Base clustering directory

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


def augment_best_models_with_metrics(
    best_models_df: pd.DataFrame,
    results: Dict,
    clustering_data: Dict[str, pd.DataFrame],
    target_ratios: Dict[str, float],
) -> pd.DataFrame:
    """
    Augment best models DataFrame with KL divergence and JSD-based recovery metrics.

    Args:
        best_models_df: DataFrame containing best models
        results: Dictionary of optimization results
        clustering_data: Dictionary mapping ensemble names to cluster DataFrames
        target_ratios: Dictionary of target state ratios

    Returns:
        Augmented DataFrame with additional metrics
    """
    augmented_rows = []

    for _, row in best_models_df.iterrows():
        ensemble = row["ensemble"]
        loss_func = row["loss_function"]
        split_idx = int(row["split"])
        maxent_val = row["maxent_value"]
        conv_step = int(row["convergence_step"])
        split_type = row["split_type"]

        # Safe nested lookup
        ensemble_dict = results.get(ensemble, {})
        loss_dict = ensemble_dict.get(loss_func, {})
        split_entry = loss_dict.get(split_idx, None)

        if split_entry is None:
            print(
                f"Warning: No results for {ensemble}/{loss_func}/split{split_idx} (split_type={split_type})"
            )
            continue

        # Determine history based on format
        history = None
        if isinstance(split_entry, dict):
            if maxent_val in split_entry:
                history = split_entry[maxent_val]
            else:
                alt_key = str(maxent_val)
                if alt_key in split_entry:
                    history = split_entry[alt_key]
                else:
                    print(
                        f"Warning: maxent value {maxent_val} not found for {ensemble}/{loss_func}/split{split_idx}"
                    )
                    continue
        else:
            history = split_entry

        # Validate history
        if history is None:
            print(
                f"Warning: History is None for {ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}"
            )
            continue

        if not hasattr(history, "states") or not history.states:
            print(
                f"Warning: No states in history for {ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}"
            )
            continue

        # Ensure convergence step exists
        if conv_step <= 0 or conv_step > len(history.states):
            print(
                f"Warning: convergence_step {conv_step} out of range (1..{len(history.states)}) for {ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}"
            )
            continue

        state = history.states[conv_step - 1]  # Convert to 0-indexed

        if (
            hasattr(state, "params")
            and hasattr(state.params, "frame_weights")
            and state.params.frame_weights is not None
        ):
            weights = np.array(state.params.frame_weights)

            # Compute KL divergence vs uniform
            uniform_prior = np.ones(len(weights)) / len(weights)
            kl_div = kl_divergence(weights, uniform_prior)

            # Compute JSD-based recovery if cluster assignments available
            js_div = np.nan
            js_distance = np.nan
            recovery_percent = np.nan

            if ensemble in clustering_data:
                cluster_df = clustering_data[ensemble]
                cluster_assignments = cluster_df["cluster_label"]

                if len(weights) == len(cluster_assignments):
                    # Normalize weights
                    normalized_weights = weights / np.sum(weights)

                    # Calculate JSD-based recovery
                    js_div, current_props = calculate_recovery_JSD(
                        cluster_assignments, normalized_weights, target_ratios, STATE_MAPPING
                    )

                    if not np.isnan(js_div):
                        js_distance = np.sqrt(js_div)
                        recovery_percent = (1.0 - js_distance) * 100.0
                else:
                    print(
                        f"Warning: length mismatch between weights ({len(weights)}) and clusters ({len(cluster_assignments)}) for {ensemble} split {split_idx}"
                    )

            # Add metrics to row
            row_dict = row.to_dict()
            row_dict["kl_divergence"] = kl_div
            row_dict["js_divergence"] = js_div if not np.isnan(js_div) else 0.0
            row_dict["js_distance"] = js_distance if not np.isnan(js_distance) else 0.0
            row_dict["recovery_percent"] = (
                recovery_percent if not np.isnan(recovery_percent) else 0.0
            )
            augmented_rows.append(row_dict)
        else:
            print(
                f"Warning: No frame_weights found in state for {ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}"
            )

    return pd.DataFrame(augmented_rows)


def plot_best_model_comparisons(df: pd.DataFrame, output_dir: str):
    """
    Plot bar charts comparing best models across different metrics.
    Uses publication-ready styling from TeaA with split type coloring.

    Args:
        df: DataFrame containing best models with metrics
        output_dir: Directory to save plots
    """
    if df.empty:
        print("No data for best model comparisons")
        return

    # Map split_type to readable names
    df = df.copy()
    df["split_name"] = df["split_type"].map(SPLIT_NAME_MAPPING).fillna(df["split_type"])

    metrics = [
        ("kl_divergence", "KL Divergence"),
        ("js_divergence", "JS Divergence"),
        ("recovery_percent", "Recovery %"),
        ("train_loss", "Training Loss"),
        ("val_loss", "Validation Loss"),
    ]

    # Create separate figures for each metric
    for metric, label in metrics:
        if metric not in df.columns:
            print(f"Metric {metric} not found in data")
            continue

        # Get unique loss functions
        loss_functions = sorted(df["loss_function"].unique())

        fig, axes = plt.subplots(
            1, len(loss_functions), figsize=(7 * len(loss_functions), 6), squeeze=False
        )
        axes = axes.flatten()

        # Create color palette for split types
        split_names_in_data = df["split_name"].unique()
        palette = {
            name: SPLIT_TYPE_COLOURS.get(
                df[df["split_name"] == name]["split_type"].iloc[0], "grey"
            )
            for name in split_names_in_data
        }

        for idx, loss_func in enumerate(loss_functions):
            ax = axes[idx]
            loss_df = df[df["loss_function"] == loss_func]

            if not loss_df.empty:
                sns.barplot(
                    data=loss_df,
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
                ax.set_title(f"{loss_func}", fontweight="bold")

                # Color x-axis tick labels by ensemble
                for tick_label in ax.get_xticklabels():
                    ensemble_name = tick_label.get_text()
                    color = ENSEMBLE_COLOURS.get(ensemble_name, "black")
                    tick_label.set_color(color)
                    tick_label.set_fontweight("bold")

                # Customize legend
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles=handles,
                    labels=labels,
                    title="Split Type",
                    title_fontsize=12,
                    fontsize=10,
                    frameon=True,
                    fancybox=False,
                    edgecolor="black",
                )

                # Use log scale for losses
                if "loss" in metric:
                    ax.set_yscale("log")

                # Add reference line at 100% for recovery metric
                if metric == "recovery_percent":
                    ax.axhline(
                        y=100,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label="Target (100%)",
                        zorder=0,
                    )
                    # Update legend to include the reference line
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(
                        handles=handles,
                        labels=labels,
                        title="Split Type",
                        title_fontsize=12,
                        fontsize=10,
                        frameon=True,
                        fancybox=False,
                        edgecolor="black",
                    )

                # Remove top and right spines
                sns.despine(ax=ax)
            else:
                ax.set_visible(False)

        # Hide unused subplots
        for idx in range(len(loss_functions), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Best Model Comparison: {label}", fontsize=22, fontweight="bold")
        plt.tight_layout()

        filename = f"best_model_comparison_{metric}.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
        plt.close()


def plot_cluster_proportions_for_best_models(
    best_models_df: pd.DataFrame,
    results: Dict,
    clustering_data: Dict[str, pd.DataFrame],
    output_dir: str,
):
    """
    Plot cluster proportions for the best selected models.

    Creates separate figures for each ensemble, with panels for each split type × loss function.
    Shows mean ± SEM across data splits.

    Args:
        best_models_df: DataFrame containing best models
        results: Dictionary of optimization results
        clustering_data: Dictionary mapping ensemble names to cluster DataFrames
        output_dir: Directory to save plots
    """
    if best_models_df.empty:
        print("No best models data available to plot cluster proportions.")
        return

    print("\nExtracting cluster proportions for best models...")

    # Collect cluster proportion data
    cluster_data_rows = []

    for _, row in best_models_df.iterrows():
        ensemble = row["ensemble"]
        loss_func = row["loss_function"]
        split_idx = int(row["split"])
        maxent_val = row["maxent_value"]
        conv_step = int(row["convergence_step"])
        split_type = row["split_type"]

        # Get the optimization history
        ensemble_dict = results.get(ensemble, {})
        loss_dict = ensemble_dict.get(loss_func, {})
        split_entry = loss_dict.get(split_idx, None)

        if split_entry is None:
            continue

        # Get history based on format
        history = None
        if isinstance(split_entry, dict):
            history = split_entry.get(maxent_val, None)
            if history is None:
                # Try string key
                history = split_entry.get(str(maxent_val), None)
        else:
            history = split_entry

        if history is None or not hasattr(history, "states") or not history.states:
            continue

        if conv_step <= 0 or conv_step > len(history.states):
            continue

        state = history.states[conv_step - 1]

        if not hasattr(state, "params") or not hasattr(state.params, "frame_weights"):
            continue

        weights = np.array(state.params.frame_weights)

        # Get cluster assignments
        if ensemble not in clustering_data:
            continue

        cluster_df = clustering_data[ensemble]
        cluster_assignments = cluster_df["cluster_label"].values

        if len(weights) != len(cluster_assignments):
            print(
                f"Warning: length mismatch for {ensemble} split {split_idx}: weights={len(weights)}, clusters={len(cluster_assignments)}"
            )
            continue

        # Normalize weights
        normalized_weights = weights / np.sum(weights)

        # Calculate proportion for each cluster
        unique_clusters = np.unique(cluster_assignments)
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            proportion = np.sum(normalized_weights[cluster_mask])

            cluster_data_rows.append(
                {
                    "ensemble": ensemble,
                    "loss_function": loss_func,
                    "split_type": split_type,
                    "split": split_idx,
                    "cluster_label": int(cluster_id),
                    "proportion": float(proportion),
                }
            )

    if not cluster_data_rows:
        print("No cluster proportion data could be extracted.")
        return

    cluster_prop_df = pd.DataFrame(cluster_data_rows)

    # Save the data
    csv_path = os.path.join(output_dir, "best_models_cluster_proportions.csv")
    cluster_prop_df.to_csv(csv_path, index=False)
    print(f"Cluster proportions data saved to: {csv_path}")

    # Convert to percentage
    cluster_prop_df["proportion_pct"] = cluster_prop_df["proportion"] * 100.0

    # Create plots for each ensemble
    ensembles = sorted(cluster_prop_df["ensemble"].unique())

    for ensemble in ensembles:
        df_ensemble = cluster_prop_df[cluster_prop_df["ensemble"] == ensemble]

        # Get unique split types and loss functions for this ensemble
        split_types = sorted(df_ensemble["split_type"].unique())
        loss_functions = sorted(df_ensemble["loss_function"].unique())

        # Determine number of panels needed
        n_combinations = len(split_types) * len(loss_functions)

        # Create subplot grid
        if n_combinations <= 2:
            nrows, ncols = 1, n_combinations
        elif n_combinations <= 4:
            nrows, ncols = 2, 2
        else:
            ncols = 3
            nrows = int(np.ceil(n_combinations / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False)
        axes = axes.flatten()

        # Get all unique clusters for this ensemble
        all_clusters = sorted(df_ensemble["cluster_label"].unique())
        n_clusters = len(all_clusters)

        # Create color palette for clusters
        colors = sns.color_palette("tab10", n_colors=max(n_clusters, 10))
        cluster_colors = {
            cluster: colors[i % len(colors)] for i, cluster in enumerate(all_clusters)
        }

        plot_idx = 0
        for split_type in split_types:
            for loss_func in loss_functions:
                if plot_idx >= len(axes):
                    break

                ax = axes[plot_idx]

                # Filter data for this combination
                df_combo = df_ensemble[
                    (df_ensemble["split_type"] == split_type)
                    & (df_ensemble["loss_function"] == loss_func)
                ]

                if df_combo.empty:
                    ax.set_visible(False)
                    plot_idx += 1
                    continue

                # Calculate mean and SEM across splits for each cluster
                stats = (
                    df_combo.groupby("cluster_label")["proportion_pct"]
                    .agg(["mean", "sem"])
                    .reset_index()
                )

                # Ensure all clusters are represented
                stats = (
                    stats.set_index("cluster_label")
                    .reindex(all_clusters, fill_value=0)
                    .reset_index()
                )
                stats["sem"] = stats["sem"].fillna(0)

                # Create bar plot
                x_pos = np.arange(len(all_clusters))
                bars = ax.bar(
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
                    f"{split_type} - {loss_func}",
                    fontsize=12,
                    fontweight="bold",
                    color=ENSEMBLE_COLOURS.get(ensemble, "black"),
                )
                ax.set_xticks(x_pos)
                ax.set_xticklabels([str(c) for c in all_clusters])
                ax.set_ylim(0, min(max(stats["mean"].max() * 1.15, 5), 100))
                ax.grid(axis="y", alpha=0.3, linestyle="--")
                sns.despine(ax=ax)

                plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f"Cluster Proportions for Best Models - {ensemble}",
            fontsize=22,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()

        # Save figure
        filename = f"cluster_proportions_best_models_{ensemble}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved cluster proportions plot: {filename}")
        plt.close()


def plot_loss_convergence(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None
):
    """
    Plot error vs convergence with training and validation error separate.
    Ensembles are shown as different colors, loss functions as different markers.
    Uses publication-ready styling.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization (can be maxent values)
        output_dir: Directory to save plots
        split_type: Name of the split type (for titles)
    """
    # Create separate plots for training and validation errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Error vs Convergence Threshold{title_suffix}", fontsize=22, fontweight="bold")

    ensembles = sorted(df["ensemble"].unique())
    loss_functions = sorted(df["loss_function"].unique())

    # Plot training errors
    ax = ax1
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            # Filter data for this combination
            if "maxent_value" in df.columns:
                # New format with maxent values
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]

                if len(subset) > 0:
                    # Calculate mean and std across splits for each maxent value
                    stats = (
                        subset.groupby("maxent_value")
                        .agg({"train_loss": ["mean", "std"]})
                        .reset_index()
                    )

                    stats.columns = ["convergence_rate", "train_mean", "train_std"]

                    # Plot training loss
                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    line = ax.errorbar(
                        stats["convergence_rate"],
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
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                # Old format - skip step 0
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df["convergence_threshold_step"] > 0)
                ]

                if len(subset) > 0:
                    stats = (
                        subset.groupby("convergence_threshold_step")
                        .agg({"train_loss": ["mean", "std"]})
                        .reset_index()
                    )

                    stats.columns = ["step", "train_mean", "train_std"]
                    stats["convergence_rate"] = stats["step"].apply(
                        lambda x: convergence_rates[x - 1]
                        if x - 1 < len(convergence_rates)
                        else None
                    )
                    stats = stats.dropna(subset=["convergence_rate"])

                    if len(stats) > 0:
                        color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                        marker = LOSS_MARKERS.get(loss_func, "o")
                        label = f"{ensemble} - {loss_func}"

                        ax.errorbar(
                            stats["convergence_rate"],
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
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Training Error", fontweight="bold")
    ax.set_title("Training Error vs Convergence", fontweight="bold")

    # Create custom legend with colored text
    legend = ax.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    # Color legend text by ensemble
    for text, handle in zip(legend.get_texts(), legend_handles):
        label_text = text.get_text()
        ensemble_name = label_text.split(" - ")[0]
        color = ENSEMBLE_COLOURS.get(ensemble_name, "black")
        text.set_color(color)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    # Plot validation errors
    ax = ax2
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            if "maxent_value" in df.columns:
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]

                if len(subset) > 0:
                    stats = (
                        subset.groupby("maxent_value")
                        .agg({"val_loss": ["mean", "std"]})
                        .reset_index()
                    )

                    stats.columns = ["convergence_rate", "val_mean", "val_std"]

                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    line = ax.errorbar(
                        stats["convergence_rate"],
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
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df["convergence_threshold_step"] > 0)
                ]

                if len(subset) > 0:
                    stats = (
                        subset.groupby("convergence_threshold_step")
                        .agg({"val_loss": ["mean", "std"]})
                        .reset_index()
                    )

                    stats.columns = ["step", "val_mean", "val_std"]
                    stats["convergence_rate"] = stats["step"].apply(
                        lambda x: convergence_rates[x - 1]
                        if x - 1 < len(convergence_rates)
                        else None
                    )
                    stats = stats.dropna(subset=["convergence_rate"])

                    if len(stats) > 0:
                        color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                        marker = LOSS_MARKERS.get(loss_func, "o")
                        label = f"{ensemble} - {loss_func}"

                        ax.errorbar(
                            stats["convergence_rate"],
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
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Validation Error", fontweight="bold")
    ax.set_title("Validation Error vs Convergence", fontweight="bold")

    # Create custom legend with colored text
    legend = ax.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    # Color legend text by ensemble
    for text, handle in zip(legend.get_texts(), legend_handles):
        label_text = text.get_text()
        ensemble_name = label_text.split(" - ")[0]
        color = ENSEMBLE_COLOURS.get(ensemble_name, "black")
        text.set_color(color)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    filename = (
        f"error_vs_convergence_{split_type}.png" if split_type else "error_vs_convergence.png"
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_split_variability(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None
):
    """
    Plot standard deviation across splits for each convergence threshold.
    Uses publication-ready styling.
    """
    # Create separate plots for training and validation standard deviations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Standard Deviation Across Splits{title_suffix}", fontsize=22, fontweight="bold")

    ensembles = sorted(df["ensemble"].unique())
    loss_functions = sorted(df["loss_function"].unique())

    # Plot training loss standard deviations
    ax = ax1
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            if "maxent_value" in df.columns:
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]

                if len(subset) > 0:
                    std_stats = (
                        subset.groupby("maxent_value").agg({"train_loss": "std"}).reset_index()
                    )

                    std_stats.columns = ["convergence_rate", "train_std"]

                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    (line,) = ax.plot(
                        std_stats["convergence_rate"],
                        std_stats["train_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df["convergence_threshold_step"] > 0)
                ]

                if len(subset) > 0:
                    std_stats = (
                        subset.groupby("convergence_threshold_step")
                        .agg({"train_loss": "std"})
                        .reset_index()
                    )

                    std_stats.columns = ["step", "train_std"]
                    std_stats["convergence_rate"] = std_stats["step"].apply(
                        lambda x: convergence_rates[x - 1]
                        if x - 1 < len(convergence_rates)
                        else None
                    )
                    std_stats = std_stats.dropna(subset=["convergence_rate"])

                    if len(std_stats) > 0:
                        color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                        marker = LOSS_MARKERS.get(loss_func, "o")
                        label = f"{ensemble} - {loss_func}"

                        ax.plot(
                            std_stats["convergence_rate"],
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
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Training Error Std Dev", fontweight="bold")
    ax.set_title("Training Error Standard Deviation", fontweight="bold")

    # Create custom legend with colored text
    legend = ax.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    for text in legend.get_texts():
        label_text = text.get_text()
        ensemble_name = label_text.split(" - ")[0]
        color = ENSEMBLE_COLOURS.get(ensemble_name, "black")
        text.set_color(color)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    # Plot validation loss standard deviations
    ax = ax2
    legend_handles = []
    legend_labels = []

    for ensemble in ensembles:
        for loss_func in loss_functions:
            if "maxent_value" in df.columns:
                subset = df[(df["ensemble"] == ensemble) & (df["loss_function"] == loss_func)]

                if len(subset) > 0:
                    std_stats = (
                        subset.groupby("maxent_value").agg({"val_loss": "std"}).reset_index()
                    )

                    std_stats.columns = ["convergence_rate", "val_std"]

                    color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                    marker = LOSS_MARKERS.get(loss_func, "o")
                    label = f"{ensemble} - {loss_func}"

                    (line,) = ax.plot(
                        std_stats["convergence_rate"],
                        std_stats["val_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=8,
                        markeredgewidth=1.5,
                        markeredgecolor="black",
                    )
                    legend_handles.append(line)
                    legend_labels.append(label)
            else:
                subset = df[
                    (df["ensemble"] == ensemble)
                    & (df["loss_function"] == loss_func)
                    & (df["convergence_threshold_step"] > 0)
                ]

                if len(subset) > 0:
                    std_stats = (
                        subset.groupby("convergence_threshold_step")
                        .agg({"val_loss": "std"})
                        .reset_index()
                    )

                    std_stats.columns = ["step", "val_std"]
                    std_stats["convergence_rate"] = std_stats["step"].apply(
                        lambda x: convergence_rates[x - 1]
                        if x - 1 < len(convergence_rates)
                        else None
                    )
                    std_stats = std_stats.dropna(subset=["convergence_rate"])

                    if len(std_stats) > 0:
                        color = ENSEMBLE_COLOURS.get(ensemble, "grey")
                        marker = LOSS_MARKERS.get(loss_func, "o")
                        label = f"{ensemble} - {loss_func}"

                        ax.plot(
                            std_stats["convergence_rate"],
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
    ax.set_xlabel("Convergence Threshold", fontweight="bold")
    ax.set_ylabel("Validation Error Std Dev", fontweight="bold")
    ax.set_title("Validation Error Standard Deviation", fontweight="bold")

    # Create custom legend with colored text
    legend = ax.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        fancybox=False,
        edgecolor="black",
    )
    for text in legend.get_texts():
        label_text = text.get_text()
        ensemble_name = label_text.split(" - ")[0]
        color = ENSEMBLE_COLOURS.get(ensemble_name, "black")
        text.set_color(color)

    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

    plt.tight_layout()
    filename = f"split_variability_{split_type}.png" if split_type else "split_variability.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_final_performance_comparison(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """
    Plot comparison of final performance across ensembles and loss functions.
    """
    # Get final convergence step data (last step for each combination)
    final_data = df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    title_suffix = f" - {split_type}" if split_type else ""
    fig.suptitle(f"Final Performance Comparison{title_suffix}", fontsize=22, fontweight="bold")

    # Training loss comparison
    sns.boxplot(data=final_data, x="ensemble", y="train_loss", hue="loss_function", ax=ax1)
    ax1.set_yscale("log")
    ax1.set_title("Final Training Loss", fontweight="bold")
    ax1.set_ylabel("Training Loss (log scale)", fontweight="bold")
    ax1.set_xlabel("Ensemble", fontweight="bold")
    sns.despine(ax=ax1)

    # Validation loss comparison
    sns.boxplot(data=final_data, x="ensemble", y="val_loss", hue="loss_function", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_title("Final Validation Loss", fontweight="bold")
    ax2.set_ylabel("Validation Loss (log scale)", fontweight="bold")
    ax2.set_xlabel("Ensemble", fontweight="bold")
    sns.despine(ax=ax2)

    plt.tight_layout()
    filename = (
        f"final_performance_comparison_{split_type}.png"
        if split_type
        else "final_performance_comparison.png"
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def generate_summary_statistics(df: pd.DataFrame, output_dir: str, split_type: str = None):
    """
    Generate summary statistics and save to CSV.
    """
    # Summary statistics for final performance
    final_data = df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()

    summary = (
        final_data.groupby(["ensemble", "loss_function"])
        .agg(
            {"train_loss": ["mean", "std", "min", "max"], "val_loss": ["mean", "std", "min", "max"]}
        )
        .round(6)
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()

    # Save to CSV
    filename = f"summary_statistics_{split_type}.csv" if split_type else "summary_statistics.csv"
    summary_path = os.path.join(output_dir, filename)
    summary.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")

    # Print summary to console
    print(f"\nSummary Statistics for {split_type if split_type else 'All Data'}:")
    print("=" * 80)
    print(summary)

    return summary


def main():
    """
    Main function to run the complete analysis with multiple split types.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Model loss analysis with best model selection and JSD-based recovery"
    )
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_quick_test__20251016_160214",
        help="Results directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (if omitted, derived from results-dir)",
    )
    parser.add_argument(
        "--clustering-dir",
        default="_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters",
        help="Directory containing cluster assignment subdirectories",
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
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        default=False,
        help="Interpret paths as absolute",
    )
    args = parser.parse_args()

    # Define parameters
    ensembles = ["AF2_MSAss", "AF2_filtered"]
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
    num_splits = 3
    convergence_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # Resolve paths
    script_dir = os.path.dirname(__file__)
    if args.absolute_paths:
        base_results_dir = args.results_dir
        clustering_dir = args.clustering_dir
        state_ratios_path = args.state_ratios_json
    else:
        base_results_dir = os.path.join(script_dir, args.results_dir)
        clustering_dir = os.path.join(script_dir, args.clustering_dir)
        state_ratios_path = os.path.join(script_dir, args.state_ratios_json)

    # Determine output directory
    if args.output_dir:
        if args.absolute_paths:
            output_base_dir = args.output_dir
        else:
            output_base_dir = os.path.join(script_dir, args.output_dir)
    else:
        base_name = os.path.basename(os.path.normpath(base_results_dir))
        out_name = "_analysis" + base_name
        if args.absolute_paths:
            parent = os.path.dirname(os.path.normpath(base_results_dir))
            output_base_dir = os.path.join(parent, out_name)
        else:
            output_base_dir = os.path.join(script_dir, out_name)

    ema_flag = args.ema

    # Show resolved paths
    print(f"Resolved results_dir: {base_results_dir}")
    print(f"Resolved clustering_dir: {clustering_dir}")
    print(f"Resolved state_ratios_json: {state_ratios_path}")
    print(f"Resolved output_dir: {output_base_dir}")
    print(f"EMA flag: {ema_flag}")

    # Check directories exist
    if not os.path.exists(base_results_dir):
        raise FileNotFoundError(f"Results directory not found: {base_results_dir}")

    # Load target state ratios
    print("\nLoading target state ratios...")
    try:
        with open(state_ratios_path, "r") as f:
            ratios_data = json.load(f)
        target_ratios = {
            "Folded": ratios_data["fractional_populations"]["folded"]["fraction"],
            "PUF1": ratios_data["fractional_populations"]["PUF1"]["fraction"],
            "PUF2": ratios_data["fractional_populations"]["PUF2"]["fraction"],
        }
        print("Target state ratios:")
        for state, ratio in target_ratios.items():
            print(f"  {state}: {ratio:.4f}")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading state ratios: {e}")
        return

    print("Starting Model Analysis with Best Model Selection...")
    print(f"Base results directory: {base_results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Base output directory: {output_base_dir}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"Number of splits: {num_splits}")
    print("-" * 60)

    # Detect split types
    split_types = [
        d for d in os.listdir(base_results_dir) if os.path.isdir(os.path.join(base_results_dir, d))
    ]

    if not split_types:
        print("No split type subdirectories found. Using flat structure.")
        split_types = [None]

    print(f"Found split types: {split_types}")

    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)

    # Load cluster assignments
    print("\nLoading cluster assignments...")
    clustering_data = {}
    for ensemble in ensembles:
        try:
            clustering_data[ensemble] = load_clustering_for_ensemble(ensemble, clustering_dir)
        except Exception as e:
            print(f"Error loading clustering for {ensemble}: {e}")
            return

    # Store all data for analysis
    all_split_data = []
    all_results = {}

    for split_type in split_types:
        if split_type:
            print(f"\n--- Analyzing split type: {split_type} ---")
            output_dir = os.path.join(output_base_dir, split_type)
        else:
            print("\n--- Analyzing results (flat structure) ---")
            output_dir = output_base_dir

        os.makedirs(output_dir, exist_ok=True)

        # Load optimization results
        print("Loading optimization results...")
        results = load_all_optimization_results(
            results_dir=base_results_dir,
            split_type=split_type,
            ensembles=ensembles,
            loss_functions=loss_functions,
            num_splits=num_splits,
            EMA=ema_flag,
        )

        # Store for later use
        all_results[split_type] = results

        # Extract loss trajectories
        print("Extracting loss trajectories...")
        df = extract_loss_trajectories(results, split_type)

        if len(df) == 0:
            print(f"No data found for split type {split_type}! Skipping.")
            continue

        print(f"Extracted {len(df)} data points from optimization histories")

        # Store data
        all_split_data.append(df)

        # Generate standard plots
        print("Generating error vs convergence plots...")
        plot_loss_convergence(df, convergence_rates, output_dir, split_type)

        print("Generating split variability plots...")
        plot_split_variability(df, convergence_rates, output_dir, split_type)

        print("Generating final performance comparison...")
        plot_final_performance_comparison(df, output_dir, split_type)

        # Generate summary statistics
        print("Generating summary statistics...")
        summary = generate_summary_statistics(df, output_dir, split_type)

        # Generate convergence-maxent heatmaps
        print("Generating convergence-maxent heatmaps...")
        plot_convergence_maxent_heatmaps(df, convergence_rates, output_dir, split_type)

        # Generate model score heatmaps
        print("Generating model score heatmaps...")
        plot_model_score_heatmaps(df, convergence_rates, output_dir, split_type)

        # Save the full dataset
        filename = (
            f"full_analysis_data_{split_type}.csv" if split_type else "full_analysis_data.csv"
        )
        df_path = os.path.join(output_dir, filename)
        df.to_csv(df_path, index=False)
        print(f"Dataset saved to: {df_path}")

        print(
            f"Analysis for {split_type if split_type else 'flat structure'} complete. Outputs saved to {output_dir}"
        )

    # Best model selection and comparison
    if len(all_split_data) > 0:
        print("\n--- Performing Best Model Selection ---")

        # Combine all data
        combined_df = pd.concat(all_split_data, ignore_index=True)

        # Select best models
        print("Selecting best models...")
        best_models = select_best_models(combined_df)

        if not best_models.empty:
            print(f"Selected {len(best_models)} best models")

            # Augment with KL divergence and JSD-based recovery metrics
            print("Computing KL divergence and JSD-based recovery metrics for best models...")

            # Combine all results dictionaries
            combined_results = {}
            for split_type, results in all_results.items():
                if not results:
                    continue
                for ensemble, ensemble_dict in results.items():
                    combined_results.setdefault(ensemble, {})
                    for loss_func, loss_dict in (ensemble_dict or {}).items():
                        combined_results[ensemble].setdefault(loss_func, {})
                        if not isinstance(loss_dict, dict):
                            continue
                        for split_k, split_v in loss_dict.items():
                            if split_v is None:
                                continue
                            existing = combined_results[ensemble][loss_func].get(split_k, None)
                            if existing is not None:
                                continue
                            combined_results[ensemble][loss_func][split_k] = split_v

            best_models_augmented = augment_best_models_with_metrics(
                best_models, combined_results, clustering_data, target_ratios
            )

            if not best_models_augmented.empty:
                # Save best models data
                best_models_path = os.path.join(output_base_dir, "best_models.csv")
                best_models_augmented.to_csv(best_models_path, index=False)
                print(f"Best models data saved to: {best_models_path}")

                # Generate comparison plots
                print("Generating best model comparison plots...")
                plot_best_model_comparisons(best_models_augmented, output_base_dir)

                # Generate cluster proportion plots
                print("Generating cluster proportion plots for best models...")
                plot_cluster_proportions_for_best_models(
                    best_models_augmented, combined_results, clustering_data, output_base_dir
                )

                # Print summary of best models
                print("\nBest Models Summary:")
                print("=" * 80)
                for split_type in best_models_augmented["split_type"].unique():
                    print(f"\nSplit Type: {split_type}")
                    split_best = best_models_augmented[
                        best_models_augmented["split_type"] == split_type
                    ]
                    for _, row in split_best.iterrows():
                        print(
                            f"  {row['ensemble']} - {row['loss_function']} - Split {row['split']}: "
                            f"MaxEnt={row['maxent_value']:.1f}, Conv Step={row['convergence_step']}, "
                            f"Val Loss={row['val_loss']:.6f}, Score={row['model_score']:.3f}, "
                            f"KL Div={row['kl_divergence']:.4f}, JS Div={row['js_divergence']:.4f}, "
                            f"Recovery={row['recovery_percent']:.1f}%"
                        )
            else:
                print("Failed to augment best models with metrics")
        else:
            print("No best models selected")

    print("\nAnalysis completed successfully!")
    print(f"All outputs saved to: {output_base_dir}")


if __name__ == "__main__":
    main()