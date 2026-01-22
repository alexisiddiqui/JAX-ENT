"""
This script loads the optimization histories of the ISO TRI and BI models over both loss functions
and performs comprehensive analyses including convergence-maxent heatmaps, model scoring,
and best model selection with bar chart comparisons.

Updated to handle multiple split types and maxent values with convergence rate analysis.
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
import numpy as np

# Add the base directory to the path to import the HDF5 utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

# Import the HDF5 loading functions from the provided script
from jaxent.src.utils.hdf import load_optimization_history_from_file

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

# Define color schemes
ENSEMBLE_COLOURS = {
    "ISO_BI": "indigo",
    "ISO_TRI": "saddlebrown",
    "ISO-BiModal": "indigo",
    "ISO-TriModal": "saddlebrown",
}

SPLIT_TYPE_COLOURS = {
    "r": "fuchsia",
    "s": "black",
    "R3": "green",
    "sequence_cluster": "green",  # Non-Redundant
    "Sp": "grey",
    "spatial": "grey",  # Add lowercase variant
}

SPLIT_NAME_MAPPING = {
    "r": "Random",
    "s": "Sequence",
    "R3": "Non-Redundant",
    "sequence_cluster": "Non-Redundant",
    "Sp": "Spatial",
    "spatial": "Spatial",  # Add lowercase variant
}

LOSS_MARKERS = {"mcMSE": "o", "MSE": "s", "Sigma_MSE": "^"}


def load_all_optimization_results(
    results_dir: str,
    split_type: str = None,
    ensembles: List[str] = ["ISO_TRI", "ISO_BI"],
    loss_functions: List[str] = ["mcMSE", "MSE"],
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


def extract_loss_trajectories(results: Dict, split_type: str = None, cluster_assignments: Dict = None) -> pd.DataFrame:
    """
    Extract loss trajectories from optimization results.

    Args:
        results: Dictionary containing optimization histories
        split_type: Name of the split type (for labeling)
        cluster_assignments: Dictionary of cluster assignments by ensemble (optional)

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
                                    row_data = {
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
                                    
                                    # Extract KL divergence if frame weights are available
                                    if hasattr(state, "params") and hasattr(state.params, "frame_weights") and state.params.frame_weights is not None:
                                        weights = np.array(state.params.frame_weights)
                                        kl_div = compute_kl_divergence(weights)
                                        row_data["kl_divergence"] = kl_div
                                        
                                        # Also compute recovery if cluster assignments available
                                        if cluster_assignments and ensemble in cluster_assignments:
                                            clusters = cluster_assignments[ensemble]
                                            if len(weights) == len(clusters):
                                                recovery = compute_recovery_percentage(clusters, weights)
                                                row_data["recovery_percent"] = recovery
                                    
                                    data_rows.append(row_data)
                else:
                    # Old format - single history with multiple states
                    history = split_results
                    if history is not None and history.states:
                        for step_idx, state in enumerate(history.states):
                            if state.losses is not None:
                                row_data = {
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
                                
                                # Extract KL divergence if frame weights are available
                                if hasattr(state, "params") and hasattr(state.params, "frame_weights") and state.params.frame_weights is not None:
                                    weights = np.array(state.params.frame_weights)
                                    kl_div = compute_kl_divergence(weights)
                                    row_data["kl_divergence"] = kl_div
                                    
                                    # Also compute recovery if cluster assignments available
                                    if cluster_assignments and ensemble in cluster_assignments:
                                        clusters = cluster_assignments[ensemble]
                                        if len(weights) == len(clusters):
                                            recovery = compute_recovery_percentage(clusters, weights)
                                            row_data["recovery_percent"] = recovery
                                
                                data_rows.append(row_data)

    return pd.DataFrame(data_rows)


def plot_convergence_maxent_heatmaps(df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None):
    """
    Plot heatmaps of training and validation error across convergence thresholds and maxent values.
    Creates separate figures for training and validation errors.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
        split_type: Name of the split type (for titles)
    """
    plt.style.use("seaborn-v0_8-whitegrid")

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
                fontsize=16,
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

                            ax.set_title(f"{ensemble} - {loss_func}")
                            ax.set_xlabel("Convergence Threshold")
                            ax.set_ylabel("MaxEnt Value")
                            ax.set_xticklabels(col_labels, rotation=45, ha="right")
                            ax.set_yticklabels([f"{v:.0e}" for v in pivot_data.index], rotation=0)
                        else:
                            ax.text(
                                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
                            )
                            ax.set_title(f"{ensemble} - {loss_func}")
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.set_title(f"{ensemble} - {loss_func}")

            plt.tight_layout()
            filename = f"{error_type}_convergence_maxent_heatmap_{stype}.png" if stype else f"{error_type}_convergence_maxent_heatmap.png"
            plt.savefig(os.path.join(split_output_dir, filename), dpi=300, bbox_inches="tight")
            plt.close()


def compute_model_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute model scores as -log10(validation error) with penalties for training error,
    validation variance across splits, KL divergence, and bonus for consistent performance across replicates.

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

    # Base score: negative log10 of validation loss with training loss penalty
    base_score = -np.log10(val_loss) - (1.0 * np.log10(train_loss))
    
    # Add KL divergence penalty if available
    kl_penalty = 0
    if "kl_divergence" in df.columns:
        kl_div = df["kl_divergence"].fillna(0).clip(lower=0)
        # Penalize higher KL divergence (less uniform = lower score)
        # Scale by factor to make penalty significant but not overwhelming
        kl_penalty = 0.1 * kl_div
        maxent_val = df.get("maxent_value", pd.Series(dtype=float))
    
    # Compute validation error variance penalty and mean bonus across splits
    # Group by split_type, ensemble, loss_function, maxent_value, convergence_step
    grouping_cols = ["split_type", "ensemble", "loss_function", "maxent_value", "convergence_step"]
    available_cols = [col for col in grouping_cols if col in df.columns]
    
    if len(available_cols) > 0:
        # Calculate coefficient of variation (std/mean) for validation loss
        val_variance = df.groupby(available_cols)["val_loss"].transform("std") / df.groupby(available_cols)["val_loss"].transform("mean")
        val_variance = val_variance.fillna(0)  # Handle cases with single split
        
        # Penalize score by variance (higher variance = lower score)
        # Use log scale to keep penalty proportional
        variance_penalty = - np.log10(1 + val_variance)
        
        # Add bonus for low mean validation error across replicates
        # Lower mean validation error = higher bonus
        val_mean = df.groupby(available_cols)["val_loss"].transform("mean")
        val_mean = val_mean.clip(lower=eps)
        train_mean = df.groupby(available_cols)["train_loss"].transform("mean")
        train_mean = train_mean.clip(lower=eps)
        mean_score =  -np.log10(val_mean)- (1 * np.log10(train_mean))
        

    # Initialize model_score with base_score plus KL penalty
    df["model_score"] = base_score 

    return df


def plot_model_score_heatmaps(df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None):
    """
    Plot heatmaps of model scores (-log(val_loss)) averaged over split replicates.

    Args:
        df: DataFrame containing loss data with scores
        convergence_rates: List of convergence rates
        output_dir: Directory to save plots
        split_type: Name of the split type
    """
    plt.style.use("seaborn-v0_8-whitegrid")

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

                        ax.set_title(f"{ensemble} - {loss_func}")
                        ax.set_xlabel("Convergence Threshold")
                        ax.set_ylabel("MaxEnt Value")
                        ax.set_xticklabels(col_labels, rotation=45, ha="right")
                        ax.set_yticklabels([f"{v:.0e}" for v in pivot_data.index], rotation=0)
                    else:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
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
        df_maxent.groupby(["split_type", "ensemble", "loss_function", "split"])["model_score"].idxmax()
    ]

    return best_models


def load_cluster_assignments(clustering_dir: str) -> Dict:
    """
    Load cluster assignments from CSV files.

    Args:
        clustering_dir: Directory containing cluster assignment CSV files

    Returns:
        Dictionary mapping ensemble names to cluster assignments
    """
    cluster_assignments = {}

    for ensemble in ["ISO_TRI", "ISO_BI"]:
        csv_path = os.path.join(clustering_dir, f"cluster_assignments_{ensemble}.csv")

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            cluster_assignments[ensemble] = df["cluster_assignment"].values
            print(f"Loaded cluster assignments for {ensemble}: {len(df)} frames")
        else:
            print(f"Warning: Cluster assignment file not found: {csv_path}")

    return cluster_assignments


def compute_kl_divergence(weights: np.ndarray) -> float:
    """
    Compute KL divergence between frame weights and uniform distribution.

    Args:
        weights: Frame weights (will be normalized)

    Returns:
        KL divergence value
    """
    if len(weights) == 0 or np.sum(weights) == 0:
        return np.nan

    # Normalize weights
    p = weights / np.sum(weights)

    # Uniform distribution
    q = np.ones(len(weights)) / len(weights)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    # Compute KL divergence
    kl_div = np.sum(p * np.log(p / q))

    return kl_div


def compute_recovery_percentage(cluster_assignments: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Compute open state recovery percentage using 1 - sqrt(JSD) against the target distribution.
    """
    if cluster_assignments is None or len(cluster_assignments) == 0:
        return np.nan

    if weights is None:
        weights = np.ones(len(cluster_assignments), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != len(cluster_assignments) or np.sum(weights) <= 0:
        return np.nan

    weights = weights / np.sum(weights)

    # Predicted distribution (open=0, closed=1, remaining treated as intermediate)
    open_ratio = float(np.sum(weights[cluster_assignments == 0]))
    closed_ratio = float(np.sum(weights[cluster_assignments == 1]))
    intermediate_ratio = max(0.0, 1.0 - open_ratio - closed_ratio)

    pred_dist = np.array([open_ratio, closed_ratio, intermediate_ratio], dtype=float)
    pred_dist = np.clip(pred_dist, 0.0, 1.0)
    if pred_dist.sum() == 0:
        return np.nan
    pred_dist = pred_dist / pred_dist.sum()

    # Ground-truth distribution
    gt_dist = np.array([0.4, 0.6, 0.0], dtype=float)
    gt_dist = gt_dist / gt_dist.sum()

    eps = 1e-12
    pred_dist = np.clip(pred_dist, eps, 1.0)
    gt_dist = np.clip(gt_dist, eps, 1.0)
    midpoint = np.clip(0.5 * (pred_dist + gt_dist), eps, 1.0)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log2(p / q)))

    jsd = 0.5 * _kl(pred_dist, midpoint) + 0.5 * _kl(gt_dist, midpoint)
    jsd = max(0.0, jsd)

    return 100.0 * max(0.0, 1.0 - np.sqrt(jsd))


def compute_cluster_percentages(weights: np.ndarray, cluster_assignments: np.ndarray) -> Dict[str, float]:
    """
    Compute cluster population percentages from frame weights and cluster assignments.
    
    Args:
        weights: Frame weights (will be normalized)
        cluster_assignments: Cluster assignments (0=open, 1=closed, 2+=intermediate)
    
    Returns:
        Dictionary with raw_open_percentage, raw_closed_percentage, raw_intermediate_percentage
    """
    if len(weights) == 0 or len(cluster_assignments) == 0 or len(weights) != len(cluster_assignments):
        return {
            "raw_open_percentage": np.nan,
            "raw_closed_percentage": np.nan,
            "raw_intermediate_percentage": np.nan,
        }
    
    # Normalize weights
    weights = np.asarray(weights, dtype=float)
    if np.sum(weights) <= 0:
        return {
            "raw_open_percentage": np.nan,
            "raw_closed_percentage": np.nan,
            "raw_intermediate_percentage": np.nan,
        }
    
    weights = weights / np.sum(weights)
    
    # Calculate percentages
    open_percentage = float(np.sum(weights[cluster_assignments == 0]) * 100)
    closed_percentage = float(np.sum(weights[cluster_assignments == 1]) * 100)
    intermediate_percentage = 100.0 - open_percentage - closed_percentage
    intermediate_percentage = max(0.0, intermediate_percentage)  # Ensure non-negative
    
    return {
        "raw_open_percentage": open_percentage,
        "raw_closed_percentage": closed_percentage,
        "raw_intermediate_percentage": intermediate_percentage,
    }


def augment_best_models_with_metrics(
    best_models_df: pd.DataFrame,
    results: Dict,
    cluster_assignments: Dict,
) -> pd.DataFrame:
    """
    Augment best models DataFrame with KL divergence and recovery metrics.

    Args:
        best_models_df: DataFrame containing best models
        results: Dictionary of optimization results
        cluster_assignments: Dictionary of cluster assignments by ensemble

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

        # Safe nested lookup: results structure may contain None for a split or different formats
        ensemble_dict = results.get(ensemble, {})
        loss_dict = ensemble_dict.get(loss_func, {})

        # Get split-level entry; could be None, a dict mapping maxent->history, or a direct history
        split_entry = loss_dict.get(split_idx, None)

        if split_entry is None:
            # No data for this split index
            print(f"Warning: No results for {ensemble}/{loss_func}/split{split_idx} (split_type={split_type})")
            continue

        # Determine history based on format
        history = None
        if isinstance(split_entry, dict):
            # New format: dict of maxent_val -> history
            # Use safe lookup for float keys (maxent_val may be float)
            # Accept exact match; if not present, skip
            if maxent_val in split_entry:
                history = split_entry[maxent_val]
            else:
                # try string-key fallback (in case keys were stored as strings)
                # e.g., "0.1" vs 0.1
                alt_key = str(maxent_val)
                if alt_key in split_entry:
                    history = split_entry[alt_key]
                else:
                    print(
                        f"Warning: maxent value {maxent_val} not found for {ensemble}/{loss_func}/split{split_idx}"
                    )
                    continue
        else:
            # Old format: split_entry itself is a history object
            history = split_entry

        # Validate history and states
        if history is None:
            print(f"Warning: History is None for {ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}")
            continue

        if not hasattr(history, "states") or not history.states:
            print(f"Warning: No states in history for {ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}")
            continue

        # Ensure convergence step exists in history
        if conv_step <= 0 or conv_step > len(history.states):
            print(
                f"Warning: convergence_step {conv_step} out of range (1..{len(history.states)}) for "
                f"{ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}"
            )
            continue

        state = history.states[conv_step - 1]  # Convert to 0-indexed

        if hasattr(state, "params") and hasattr(state.params, "frame_weights") and state.params.frame_weights is not None:
            weights = np.array(state.params.frame_weights)

            # Compute KL divergence
            kl_div = compute_kl_divergence(weights)

            # Compute recovery if cluster assignments available
            recovery = np.nan
            cluster_percentages = {}
            if ensemble in cluster_assignments:
                clusters = cluster_assignments[ensemble]
                if len(weights) == len(clusters):
                    recovery = compute_recovery_percentage(clusters, weights)
                    # Compute cluster percentages
                    cluster_percentages = compute_cluster_percentages(weights, clusters)
                else:
                    print(
                        f"Warning: length mismatch between weights ({len(weights)}) and clusters ({len(clusters)}) "
                        f"for {ensemble} split {split_idx}"
                    )

            # Add metrics to row
            row_dict = row.to_dict()
            row_dict["kl_divergence"] = kl_div
            row_dict["recovery_percent"] = recovery
            
            # Add cluster percentages
            row_dict.update(cluster_percentages)
            
            augmented_rows.append(row_dict)
        else:
            print(f"Warning: No frame_weights found in state for {ensemble}/{loss_func}/split{split_idx} maxent={maxent_val}")

    return pd.DataFrame(augmented_rows)


def plot_best_model_comparisons(df: pd.DataFrame, output_dir: str):
    """
    Plot bar charts comparing best models across different metrics.
    Hue by split type, x-axis is ensemble, separate panels for different metrics.

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
            1, len(loss_functions), 
            figsize=(7 * len(loss_functions), 6), 
            squeeze=False, 
            sharey=True
        )
        axes = axes.flatten()

        # Create color palette for split types
        split_names_in_data = df["split_name"].unique()
        palette = {name: SPLIT_TYPE_COLOURS.get(df[df["split_name"] == name]["split_type"].iloc[0], "grey") 
                   for name in split_names_in_data}

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
                ensembles = loss_df["ensemble"].unique()
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
                        color='red', 
                        linestyle='--', 
                        linewidth=2, 
                        alpha=0.7, 
                        label='Target (100%)',
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


def plot_loss_convergence(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str, split_type: str = None
):
    """
    Plot error vs convergence with training and validation error separate.
    Ensembles are shown as different colors, loss functions as different markers.

    Args:
        df: DataFrame containing loss data
        convergence_rates: List of convergence rates used in optimization (can be maxent values)
        output_dir: Directory to save plots
        split_type: Name of the split type (for titles)
    """
    # Create separate plots for training and validation errors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {SPLIT_NAME_MAPPING.get(split_type, split_type)}" if split_type else ""
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
        edgecolor="black"
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
        edgecolor="black"
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
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors and markers (same as main plots)
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}
    loss_markers = {"mcMSE": "o", "MSE": "s", "Sigma_MSE": "^"}
    default_color = "#777777"
    default_marker = "o"

    # Create separate plots for training and validation standard deviations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    title_suffix = f" - {SPLIT_NAME_MAPPING.get(split_type, split_type)}" if split_type else ""
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
                    marker = LOSS_MARKERS.get(loss_func, default_marker)
                    label = f"{ensemble} - {loss_func}"

                    line, = ax.plot(
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
                        marker = LOSS_MARKERS.get(loss_func, default_marker)
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
        edgecolor="black"
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
                    marker = LOSS_MARKERS.get(loss_func, default_marker)
                    label = f"{ensemble} - {loss_func}"

                    line, = ax.plot(
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
                        marker = LOSS_MARKERS.get(loss_func, default_marker)
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
        edgecolor="black"
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
    fig.suptitle(f"Final Performance Comparison{title_suffix}", fontsize=16, fontweight="bold")

    # Training loss comparison
    sns.boxplot(data=final_data, x="ensemble", y="train_loss", hue="loss_function", ax=ax1)
    ax1.set_yscale("log")
    ax1.set_title("Final Training Loss")
    ax1.set_ylabel("Training Loss (log scale)")

    # Validation loss comparison
    sns.boxplot(data=final_data, x="ensemble", y="val_loss", hue="loss_function", ax=ax2)
    ax2.set_yscale("log")
    ax2.set_title("Final Validation Loss")
    ax2.set_ylabel("Validation Loss (log scale)")

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


def plot_cluster_population_composition(recovery_df: pd.DataFrame, best_gammas: pd.DataFrame, output_dir: str, ensemble_order: List[str] = None, analysis_split_types: List[str] = None):
    """
    Plot cluster population composition (open, closed, intermediate percentages) for best gamma values.
    Creates separate subplots for each loss function.
    
    Args:
        recovery_df: DataFrame containing cluster population data with columns:
                     ensemble, split_type, split_name, replicate, gamma, loss_function,
                     raw_open_percentage, raw_closed_percentage, raw_intermediate_percentage
        best_gammas: DataFrame containing best gamma values per ensemble, split_type, and loss_function
        output_dir: Directory to save the plot
        ensemble_order: Order of ensembles for plotting (default: ["ISO_BI", "ISO_TRI"])
        analysis_split_types: Split types to include in analysis (default: detected from data)
    """
    if ensemble_order is None:
        ensemble_order = ["ISO_BI", "ISO_TRI"]
    
    if analysis_split_types is None:
        analysis_split_types = sorted(recovery_df["split_type"].unique())
    
    # Merge to get only data for best models
    merge_cols = ["ensemble", "split_type", "loss_function"]
    plot_df = pd.merge(
        recovery_df,
        best_gammas[["ensemble", "split_type", "loss_function", "gamma"]],
        on=merge_cols + ["gamma"],
        how="inner",
    )

    if plot_df.empty:
        print("No data available for cluster population composition plot")
        return None

    # Get unique loss functions
    loss_functions = sorted(plot_df["loss_function"].unique())
    
    # Aggregate across replicates for each ensemble, split_type, and loss_function
    agg = (
        plot_df.groupby(["ensemble", "split_type", "loss_function"])
        .agg(
            raw_open_percentage=("raw_open_percentage", "mean"),
            raw_closed_percentage=("raw_closed_percentage", "mean"),
            raw_intermediate_percentage=("raw_intermediate_percentage", "mean"),
        )
        .reset_index()
    )

    state_map = {
        "raw_open_percentage": "Open",
        "raw_closed_percentage": "Closed",
        "raw_intermediate_percentage": "Intermediate",
    }
    melted = agg.melt(
        id_vars=["ensemble", "split_type", "loss_function"],
        value_vars=list(state_map.keys()),
        var_name="cluster",
        value_name="percentage",
    )
    melted["cluster_label"] = melted["cluster"].map(state_map)

    cluster_order = ["Open", "Closed", "Intermediate"]
    
    # Create palette with all split types present in data
    split_types_in_data = melted["split_type"].unique()
    palette = {st: SPLIT_TYPE_COLOURS.get(st, "grey") for st in split_types_in_data}

    # Create subplots: rows for ensembles, columns for loss functions
    fig, axes = plt.subplots(
        len(ensemble_order), len(loss_functions), 
        figsize=(6 * len(loss_functions), 5 * len(ensemble_order)), 
        sharey=True, sharex=True
    )
    
    # Ensure axes is 2D
    if len(ensemble_order) == 1 and len(loss_functions) == 1:
        axes = np.array([[axes]])
    elif len(ensemble_order) == 1:
        axes = axes.reshape(1, -1)
    elif len(loss_functions) == 1:
        axes = axes.reshape(-1, 1)

    for row_idx, ensemble in enumerate(ensemble_order):
        for col_idx, loss_func in enumerate(loss_functions):
            ax = axes[row_idx, col_idx]
            
            # Filter data for this ensemble and loss function
            subset_data = melted[
                (melted["ensemble"] == ensemble) & 
                (melted["loss_function"] == loss_func)
            ]

            if not subset_data.empty:
                sns.barplot(
                    data=subset_data,
                    x="cluster_label",
                    y="percentage",
                    hue="split_type",
                    order=cluster_order,
                    hue_order=analysis_split_types,
                    palette=palette,
                    ax=ax,
                    capsize=0.06,
                    errwidth=1,
                    edgecolor="black",
                    linewidth=0.8,
                )
                
                # Title and labels
                if row_idx == 0:
                    ax.set_title(f"{loss_func}", fontsize=18, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{ensemble}\nCluster Population (%)", 
                                fontsize=16, fontweight="bold",
                                color=ENSEMBLE_COLOURS.get(ensemble, "black"))
                else:
                    ax.set_ylabel("")
                    
                ax.set_xlabel("Cluster State" if row_idx == len(ensemble_order) - 1 else "", 
                            fontsize=14, fontweight="bold")
                ax.set_ylim(0, 100)
                ax.grid(False, alpha=0)
                
                # Color x-axis elements
                col = ENSEMBLE_COLOURS.get(ensemble, "black")
                ax.tick_params(axis="x", colors=col, labelsize=12)
                ax.tick_params(axis="y", labelsize=12)
                
                # Remove individual legends
                if ax.get_legend() is not None:
                    ax.legend_.remove()
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=14)
                ax.set_xlabel("")
                ax.set_ylabel("")

    # Create unified legend
    handles = [
        Patch(
            facecolor=SPLIT_TYPE_COLOURS.get(st, "grey"), 
            edgecolor="black", 
            label=SPLIT_NAME_MAPPING.get(st, st)
        )
        for st in analysis_split_types
    ]
    fig.legend(
        handles,
        [SPLIT_NAME_MAPPING.get(st, st) for st in analysis_split_types],
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        title="Split Type",
        title_fontsize=14,
        fontsize=12,
        frameon=True,
        framealpha=0.95,
        edgecolor="black",
    )
    
    fig.suptitle("Cluster Population Composition - Best Models", fontsize=20, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 0.88, 0.99])
    
    output_path = os.path.join(output_dir, "cluster_population_composition.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved cluster population composition plot to {output_path}")
    plt.close(fig)

    return fig

def save_weights_for_best_models(
    best_models_df: pd.DataFrame,
    results: Dict,
    output_dir: str,
) -> None:
    """
    Extract frame weights from best models and save as NPZ files.
    Saves individual replicate weights and their averages per ensemble-loss-split_type combination.

    Args:
        best_models_df: DataFrame containing best models
        results: Dictionary of optimization results
        output_dir: Directory to save weight NPZ files
    """
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Group by ensemble, loss_function, and split_type
    grouping = best_models_df.groupby(["ensemble", "loss_function", "split_type"])

    for (ensemble, loss_func, split_type), group_df in grouping:
        print(f"\n  Processing weights for {ensemble} - {loss_func} - {split_type}")

        # Storage for all replicate weights
        replicate_weights = {}
        valid_replicates = []

        # Extract weights for each split replicate
        for _, row in group_df.iterrows():
            split_idx = int(row["split"])
            maxent_val = row["maxent_value"]
            conv_step = int(row["convergence_step"])

            # Safe nested lookup in results
            ensemble_dict = results.get(ensemble, {})
            loss_dict = ensemble_dict.get(loss_func, {})
            split_entry = loss_dict.get(split_idx, None)

            if split_entry is None:
                print(f"    Warning: No results for split {split_idx}")
                continue

            # Determine history based on format
            history = None
            if isinstance(split_entry, dict):
                # New format: dict of maxent_val -> history
                if maxent_val in split_entry:
                    history = split_entry[maxent_val]
                else:
                    alt_key = str(maxent_val)
                    if alt_key in split_entry:
                        history = split_entry[alt_key]
                    else:
                        print(f"    Warning: maxent value {maxent_val} not found for split {split_idx}")
                        continue
            else:
                # Old format: split_entry itself is a history object
                history = split_entry

            # Validate history and states
            if history is None or not hasattr(history, "states") or not history.states:
                print(f"    Warning: Invalid history for split {split_idx}")
                continue

            # Ensure convergence step exists
            if conv_step <= 0 or conv_step > len(history.states):
                print(f"    Warning: convergence_step {conv_step} out of range for split {split_idx}")
                continue

            state = history.states[conv_step - 1]  # Convert to 0-indexed

            # Extract weights
            if hasattr(state, "params") and hasattr(state.params, "frame_weights") and state.params.frame_weights is not None:
                weights = np.array(state.params.frame_weights, dtype=np.float32)
                # Handle NaN arrays by converting to uniform weights
                if np.any(np.isnan(weights)):
                    weights = np.ones_like(weights) / len(weights)
                replicate_weights[split_idx] = weights
                valid_replicates.append(split_idx)
                print(f"    Extracted weights for split {split_idx} (shape: {weights.shape})")
            else:
                print(f"    Warning: No frame_weights found for split {split_idx}")

        # Save individual replicate weights
        if valid_replicates:
            for split_idx in valid_replicates:
                weights = replicate_weights[split_idx]
                filename = f"weights_{ensemble}_{loss_func}_{split_type}_split{split_idx:03d}.npz"
                filepath = os.path.join(weights_dir, filename)
                np.savez_compressed(filepath, weights=weights)
                print(f"    Saved: {filename}")

            # Compute and save average weights across replicates
            avg_weights = np.mean(
                [replicate_weights[idx] for idx in valid_replicates],
                axis=0,
                dtype=np.float32
            )
            # Renormalize average weights to sum to 1
            avg_weights = avg_weights / np.sum(avg_weights)
            
            avg_filename = f"weights_{ensemble}_{loss_func}_{split_type}_average.npz"
            avg_filepath = os.path.join(weights_dir, avg_filename)
            np.savez_compressed(
                avg_filepath,
                weights=avg_weights,
                n_replicates=len(valid_replicates),
                replicate_indices=np.array(valid_replicates, dtype=np.int32)
            )
            print(f"    Saved average: {avg_filename} (from {len(valid_replicates)} replicates)")
        else:
            print(f"    No valid weights found for {ensemble} - {loss_func} - {split_type}")

    print(f"\nAll weights saved to: {weights_dir}")

def save_selected_recovery_and_kl(recovery_df: pd.DataFrame, best_gammas: pd.DataFrame, output_dir: str, atol: float = 1e-8):
    """
    For each best gamma (ensemble, split_type, loss_function, gamma) find matching rows in recovery_df
    (per replicate) and save per-replicate CSVs plus a combined CSV and NPZ.
    """
    sel_dir = os.path.join(output_dir, "selected_metrics")
    os.makedirs(sel_dir, exist_ok=True)

    selected_rows = []
    # Ensure gamma column present in recovery_df
    if "gamma" not in recovery_df.columns:
        # try common alternatives
        if "maxent_value" in recovery_df.columns:
            recovery_df = recovery_df.copy()
            recovery_df["gamma"] = recovery_df["maxent_value"]
        else:
            print("  Warning: recovery_df has no 'gamma' or 'maxent_value' column. Skipping selected metrics save.")
            return

    for _, bg in best_gammas.iterrows():
        ensemble = bg.get("ensemble")
        split_type = bg.get("split_type")
        loss_func = bg.get("loss_function") if "loss_function" in bg.index else None
        best_gamma = float(bg["gamma"])

        mask = (recovery_df["ensemble"] == ensemble) & (recovery_df["split_type"] == split_type)
        if loss_func is not None:
            mask = mask & (recovery_df["loss_function"] == loss_func)

        candidates = recovery_df[mask].copy()
        if candidates.empty:
            print(f"  Warning: no recovery rows for {ensemble} {split_type} {loss_func if loss_func else ''}")
            continue

        gamma_mask = np.isclose(candidates["gamma"].values.astype(float), best_gamma, atol=atol, rtol=1e-6)
        matches = candidates[gamma_mask]
        if matches.empty:
            print(f"  Warning: no matching gamma rows for {ensemble} {split_type} {loss_func if loss_func else ''} gamma={best_gamma}")
            continue

        for _, row in matches.iterrows():
            rep = row.get("replicate", row.get("split", "NA"))
            fname = f"selected_metrics_{ensemble}_{split_type}_{loss_func if loss_func else 'na'}_{rep}.csv"
            path = os.path.join(sel_dir, fname)
            row.to_frame().T.to_csv(path, index=False)
            selected_rows.append(row)

    if len(selected_rows) == 0:
        print("  No selected recovery rows saved.")
        return

    combined = pd.DataFrame(selected_rows)
    combined_path = os.path.join(sel_dir, "selected_metrics_all.csv")
    combined.to_csv(combined_path, index=False)
    npz_path = os.path.join(sel_dir, "selected_metrics_all.npz")
    np.savez_compressed(npz_path, **{c: combined[c].to_numpy(dtype=object) for c in combined.columns})
    print(f"  Saved selected metrics: {combined_path} and {npz_path}")

def main():
    """
    Main function to run the complete analysis with multiple split types.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="ISO model loss analysis with best model selection and comprehensive metrics"
    )
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_quick_SIGMA_test__20251015_014923",
        help="Results directory",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (if omitted, derived from results-dir)",
    )
    parser.add_argument(
        "--clustering-dir",
        default="../data/_clustering_results",
        help="Directory containing cluster assignment CSV files",
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
    ensembles = ["ISO_TRI", "ISO_BI"]
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
    num_splits = 3
    convergence_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    # Resolve paths
    script_dir = os.path.dirname(__file__)
    if args.absolute_paths:
        base_results_dir = args.results_dir
        clustering_dir = args.clustering_dir
    else:
        base_results_dir = os.path.join(script_dir, args.results_dir)
        clustering_dir = os.path.join(script_dir, args.clustering_dir)

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
    print(f"Resolved output_dir: {output_base_dir}")
    print(f"EMA flag: {ema_flag}")

    # Check directories exist
    if not os.path.exists(base_results_dir):
        raise FileNotFoundError(f"Results directory not found: {base_results_dir}")

    print("Starting ISO Model Analysis with Best Model Selection...")
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
    cluster_assignments = load_cluster_assignments(clustering_dir)

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
        df = extract_loss_trajectories(results, split_type, cluster_assignments)

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

        # NEW: Generate convergence-maxent heatmaps
        print("Generating convergence-maxent heatmaps...")
        plot_convergence_maxent_heatmaps(df, convergence_rates, output_dir, split_type)

        # NEW: Generate model score heatmaps
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

    # NEW: Best model selection and comparison
    if len(all_split_data) > 0:
        print("\n--- Performing Best Model Selection ---")

        # Combine all data
        combined_df = pd.concat(all_split_data, ignore_index=True)

        # Select best models
        print("Selecting best models...")
        best_models = select_best_models(combined_df)

        if not best_models.empty:
            print(f"Selected {len(best_models)} best models")

            # Augment with KL divergence and recovery metrics
            print("Computing KL divergence and recovery metrics for best models...")

            # Combine all results dictionaries
            combined_results = {}
            # Merge per-split results across split types but avoid overwriting valid histories
            for split_type, results in all_results.items():
                if not results:
                    continue
                for ensemble, ensemble_dict in results.items():
                    combined_results.setdefault(ensemble, {})
                    for loss_func, loss_dict in (ensemble_dict or {}).items():
                        combined_results[ensemble].setdefault(loss_func, {})
                        # loss_dict maps split_idx -> history or None
                        if not isinstance(loss_dict, dict):
                            continue
                        for split_k, split_v in loss_dict.items():
                            # Skip missing entries
                            if split_v is None:
                                continue
                            # If we already have a non-None entry for this split, keep it (do not overwrite)
                            existing = combined_results[ensemble][loss_func].get(split_k, None)
                            if existing is not None:
                                continue
                            combined_results[ensemble][loss_func][split_k] = split_v

            best_models_augmented = augment_best_models_with_metrics(
                best_models, combined_results, cluster_assignments
            )

            if not best_models_augmented.empty:
                # Save best models data
                best_models_path = os.path.join(output_base_dir, "best_models.csv")
                best_models_augmented.to_csv(best_models_path, index=False)
                print(f"Best models data saved to: {best_models_path}")

                # --- New: save per-replicate selected recovery & KL for chosen best gammas ---
                # Prepare recovery_df and best_gammas used elsewhere in the script
                recovery_df = best_models_augmented.copy()
                # ensure columns align with helper expectations
                if "maxent_value" in recovery_df.columns and "gamma" not in recovery_df.columns:
                    recovery_df["gamma"] = recovery_df["maxent_value"]
                # build best_gammas table matching ensemble, split_type, loss_function -> gamma
                best_gammas = (
                    best_models_augmented.groupby(["ensemble", "split_type", "loss_function"])
                    .agg({"maxent_value": "first"})
                    .reset_index()
                    .rename(columns={"maxent_value": "gamma"})
                )
                save_selected_recovery_and_kl(recovery_df, best_gammas, output_base_dir)

                # Generate best model comparison plots
                print("Generating best model comparison plots...")
                plot_best_model_comparisons(best_models_augmented, output_dir)

                # NEW: Save frame weights
                print("Extracting and saving frame weights...")
                save_weights_for_best_models(best_models_augmented, combined_results, output_base_dir)

                # Generate cluster population composition plot
                if "raw_open_percentage" in best_models_augmented.columns and "raw_closed_percentage" in best_models_augmented.columns:
                    print("Generating cluster population composition plot...")
                    
                    # Prepare data for cluster population plot
                    recovery_df = best_models_augmented.copy()
                    recovery_df["gamma"] = recovery_df["maxent_value"]
                    recovery_df["replicate"] = recovery_df["split"]
                    recovery_df["split_name"] = recovery_df["split_type"].map(SPLIT_NAME_MAPPING)
                    
                    # Select best gammas (maxent values) per ensemble, split type, AND loss function
                    best_gammas = (
                        best_models_augmented.groupby(["ensemble", "split_type", "loss_function"])
                        .agg({"maxent_value": "first"})
                        .reset_index()
                        .rename(columns={"maxent_value": "gamma"})
                    )
                    
                    plot_cluster_population_composition(
                        recovery_df, 
                        best_gammas, 
                        output_base_dir,
                        ensemble_order=["ISO_BI", "ISO_TRI"]
                    )
                else:
                    print("Warning: Cluster percentage data not available. Skipping cluster population plot.")

                # Print summary of best models
                print("\nBest Models Summary:")
                print("=" * 80)
                for split_type in best_models_augmented["split_type"].unique():
                    print(f"\nSplit Type: {split_type}")
                    split_best = best_models_augmented[
                        best_models_augmented["split_type"] == split_type
                    ]
                    for _, row in split_best.iterrows():
                        kl_str = f"KL Div={row['kl_divergence']:.4f}" if not pd.isna(row.get('kl_divergence')) else "KL Div=N/A"
                        rec_str = f"Recovery={row['recovery_percent']:.1f}%" if not pd.isna(row.get('recovery_percent')) else "Recovery=N/A"
                        print(
                            f"  {row['ensemble']} - {row['loss_function']} - Split {row['split']}: "
                            f"MaxEnt={row['maxent_value']:.1f}, Conv Step={row['convergence_step']}, "
                            f"Val Loss={row['val_loss']:.6f}, Score={row['model_score']:.3f}, "
                            f"{kl_str}, {rec_str}"
                        )
            else:
                print("Failed to augment best models with metrics")
        else:
            print("No best models selected")

    print("\nAnalysis completed successfully!")
    print(f"All outputs saved to: {output_base_dir}")


if __name__ == "__main__":
    main()