"""
MoPrP Weights Validation Analysis Script
Analyzes frame weights from optimization results with MaxEnt regularization.
Includes clustering-based conformational analysis and heatmap visualizations.
"""

import argparse
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

# Import the HDF5 loading functions
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

# Color schemes
ensemble_loss_colours = {
    "AF2_MSAss_MSE": "#1f77b4",  # blue
    "AF2_filtered_MSE": "#ff7f0e",  # orange
}

split_type_colours = {
    "random": "#9467bd",  # purple
}

split_name_mapping = {
    "random": "Random",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_ensemble_loss_key(ensemble, loss_function):
    """Create a unique key for ensemble-loss combinations."""
    return f"{ensemble}_{loss_function}"


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


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_clustering_results(clustering_dir, ensemble_name):
    """
    Load clustering results for a specific ensemble.

    Args:
        clustering_dir: Directory containing clustering subdirectories
        ensemble_name: Name of the ensemble (e.g., 'AF2_MSAss')

    Returns:
        DataFrame with frame-to-cluster assignments
    """
    if ensemble_name not in ENSEMBLE_CLUSTERING_MAP:
        print(f"Warning: No clustering file mapped for ensemble {ensemble_name}")
        return None

    clustering_file = ENSEMBLE_CLUSTERING_MAP[ensemble_name]
    ensemble_subdir = clustering_file.replace("_frame_to_cluster.csv", "")
    clustering_path = os.path.join(clustering_dir, ensemble_subdir, clustering_file)

    if not os.path.exists(clustering_path):
        print(f"Warning: Clustering file not found: {clustering_path}")
        return None

    cluster_df = pd.read_csv(clustering_path)
    print(f"Loaded clustering for {ensemble_name}: {len(cluster_df)} frames")

    return cluster_df


def load_all_optimization_results_with_maxent(
    results_dir: str,
    ensembles: List[str],
    loss_functions: List[str],
    split_types: List[str],
    num_splits: int = 3,
    maxent_values: List[float] = None,
    EMA: bool = False,
) -> Dict:
    """
    Load all optimization results from HDF5 files, including maxent values.

    Args:
        results_dir: Directory containing subdirectories for each split type
        ensembles: List of ensemble names
        loss_functions: List of loss function names
        split_types: List of split type names
        num_splits: Number of data splits per type
        maxent_values: List of expected maxent values
        EMA: Whether to load EMA results

    Returns:
        Dictionary with results organized by split_type, ensemble, loss, maxent, and split
    """

    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    hdf_pattern = "results_EMA.hdf5" if EMA else "results.hdf5"

    for split_type in split_types:
        results[split_type] = {}
        split_type_dir = os.path.join(results_dir, split_type)

        if not os.path.exists(split_type_dir):
            print(f"Split type directory not found: {split_type_dir}")
            continue

        for ensemble in ensembles:
            results[split_type][ensemble] = {}

            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}

                # Find all files for this ensemble/loss combination
                pattern = f"{ensemble}_{loss_name}_{split_type}_split"
                files = [
                    f
                    for f in os.listdir(split_type_dir)
                    if f.startswith(pattern) and f.endswith(hdf_pattern)
                ]

                for filename in files:
                    # Extract split index and maxent value
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
                            print(f"Loaded: {filename}")
                        except Exception as e:
                            print(f"Failed to load {filepath}: {e}")
                            results[split_type][ensemble][loss_name][maxent_val][split_idx] = None

    return results


# ============================================================================
# WEIGHT EXTRACTION FUNCTIONS
# ============================================================================


def extract_frame_weights_kl_with_maxent(results: Dict) -> pd.DataFrame:
    """
    Extract frame weights and calculate KL divergence and ESS including maxent values.

    Args:
        results: Dictionary containing optimization histories

    Returns:
        DataFrame containing KL divergence and ESS values for analysis
    """
    data_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
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

                                    uniform_prior = np.ones(len(frame_weights)) / len(frame_weights)
                                    try:
                                        kl_div = kl_divergence(frame_weights, uniform_prior)
                                        ess = effective_sample_size(frame_weights)

                                        # Calculate convergence fraction
                                        n_states = len(history.states)
                                        convergence_fraction = (
                                            step_idx / (n_states - 1) if n_states > 1 else 1.0
                                        )

                                        # Use step_idx + 1 as convergence_threshold_step (1-indexed)
                                        convergence_threshold_step = step_idx + 1

                                        data_rows.append(
                                            {
                                                "split_type": split_type,
                                                "ensemble": ensemble,
                                                "loss_function": loss_name,
                                                "maxent_value": maxent_val,
                                                "split": split_idx,
                                                "step": step_idx,
                                                "convergence_threshold_step": convergence_threshold_step,
                                                "convergence_fraction": convergence_fraction,
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


def extract_final_weights(results: Dict) -> List[Dict]:
    """
    Extract final (converged) frame weights for plotting and analysis.

    Args:
        results: Dictionary containing optimization histories

    Returns:
        List of dictionaries containing final weights data
    """
    weights_data = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is not None and history.states:
                            final_state = history.states[-1]

                            if (
                                hasattr(final_state, "params")
                                and hasattr(final_state.params, "frame_weights")
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

                                    weights_data.append(
                                        {
                                            "ensemble": ensemble,
                                            "split_type": split_type,
                                            "split": split_idx,
                                            "loss_function": loss_name,
                                            "maxent_value": maxent_val,
                                            "weights": frame_weights,
                                            "convergence_step": len(history.states) - 1,
                                        }
                                    )

    print(f"Extracted {len(weights_data)} final weight distributions")
    return weights_data


# ============================================================================
# KLD ANALYSIS FUNCTIONS
# ============================================================================


def compute_kl_divergence_between_distributions(p, q, epsilon=1e-10):
    """Compute KL divergence between two probability distributions."""
    if np.sum(p) > 0:
        p = p / np.sum(p)
    else:
        return np.nan

    if np.sum(q) > 0:
        q = q / np.sum(q)
    else:
        return np.nan

    p_safe = p + epsilon
    q_safe = q + epsilon
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)

    kl_div = np.sum(p_safe * np.log(p_safe / q_safe))
    return kl_div


def compute_pairwise_kld_between_splits(results):
    """
    Compute pairwise KLD between splits for each ensemble, split_type, loss, maxent and convergence step.
    Returns a DataFrame with mean/std/sem KLD for each (ensemble, split_type, loss_function, maxent_value, convergence_step).
    """
    print("Computing pairwise KLD between splits (per convergence step)...")
    kld_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_func in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_func]:
                    # Collect per-split per-step weight distributions
                    per_split_step_weights = {}  # split_idx -> { step_idx (1-based) : weights }
                    for split_idx, history in results[split_type][ensemble][loss_func][
                        maxent_val
                    ].items():
                        if history is None or not getattr(history, "states", None):
                            continue
                        step_map = {}
                        for step_idx, state in enumerate(history.states):
                            if (
                                hasattr(state.params, "frame_weights")
                                and state.params.frame_weights is not None
                            ):
                                w = np.array(state.params.frame_weights)
                                if np.sum(w) <= 0 or len(w) == 0:
                                    continue
                                # normalize
                                w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                                if np.sum(w) > 0:
                                    w = w / np.sum(w)
                                    step_map[step_idx + 1] = (
                                        w  # use 1-based step index for consistency
                                    )
                        if step_map:
                            per_split_step_weights[split_idx] = step_map

                    if len(per_split_step_weights) < 2:
                        # Need at least two splits with data
                        continue

                    # Determine union of steps present across splits and iterate over them
                    # We'll consider steps that appear in at least two splits
                    all_steps = {}
                    for split_idx, step_map in per_split_step_weights.items():
                        for s in step_map:
                            all_steps.setdefault(s, 0)
                            all_steps[s] += 1

                    valid_steps = [s for s, cnt in all_steps.items() if cnt >= 2]
                    valid_steps = sorted(valid_steps)

                    for conv_step in valid_steps:
                        # gather weights for this conv_step across splits (only splits that have this step)
                        weights_list = []
                        splits_with_step = []
                        for split_idx, step_map in per_split_step_weights.items():
                            if conv_step in step_map:
                                weights_list.append(step_map[conv_step])
                                splits_with_step.append(split_idx)

                        n_splits_here = len(weights_list)
                        if n_splits_here < 2:
                            continue

                        pair_klds = []
                        # compute symmetric pairwise KLDs
                        for i in range(n_splits_here):
                            for j in range(i + 1, n_splits_here):
                                wi = weights_list[i]
                                wj = weights_list[j]
                                min_len = min(len(wi), len(wj))
                                if min_len == 0:
                                    continue
                                wi_trim = wi[:min_len]
                                wj_trim = wj[:min_len]

                                # compute symmetric average KLD
                                try:
                                    kld_ij = compute_kl_divergence_between_distributions(
                                        wi_trim, wj_trim
                                    )
                                    kld_ji = compute_kl_divergence_between_distributions(
                                        wj_trim, wi_trim
                                    )
                                    if not (np.isnan(kld_ij) or np.isnan(kld_ji)):
                                        pair_klds.append((kld_ij + kld_ji) / 2.0)
                                except Exception:
                                    continue

                        if len(pair_klds) == 0:
                            continue

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
                                "split_name": split_name_mapping.get(split_type, split_type),
                                "loss_function": loss_func,
                                "maxent_value": maxent_val,
                                "convergence_threshold_step": conv_step,
                                "mean_kld_between_splits": mean_kld,
                                "std_kld_between_splits": std_kld,
                                "sem_kld_between_splits": sem_kld,
                                "n_pairs": len(pair_klds),
                                "n_splits": n_splits_here,
                            }
                        )

    return pd.DataFrame(kld_rows)


# ============================================================================
# HEATMAP PLOTTING FUNCTIONS
# ============================================================================


def plot_ess_heatmaps(kl_ess_df, convergence_rates, output_dir):
    """
    Plot heatmaps of Effective Sample Size across convergence thresholds and maxent values.

    Args:
        kl_ess_df (pd.DataFrame): ESS analysis results
        convergence_rates (List[float]): List of convergence threshold values
        output_dir (str): Output directory for plots
    """
    print("Creating ESS heatmaps...")

    if kl_ess_df.empty:
        print("  No ESS data available for heatmaps")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    kl_ess_df["convergence_threshold_step"] = pd.to_numeric(
        kl_ess_df["convergence_threshold_step"], errors="coerce"
    )

    ensembles = kl_ess_df["ensemble"].unique()
    loss_functions = kl_ess_df["loss_function"].unique()
    split_types = kl_ess_df["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating ESS heatmaps for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = kl_ess_df[kl_ess_df["split_type"] == split_type]

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
        )

        fig.suptitle(
            f"Effective Sample Size Heatmaps - {split_type} splits", fontsize=16, fontweight="bold"
        )

        for i, ensemble in enumerate(sorted(ensembles)):
            for j, loss_func in enumerate(sorted(loss_functions)):
                # Determine if axes is a single Axes object, a 1D array, or a 2D array
                if len(ensembles) == 1 and len(loss_functions) == 1:
                    ax = axes
                elif len(ensembles) == 1:
                    ax = axes[j]
                elif len(loss_functions) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                combo_df = split_df[
                    (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                ]

                if len(combo_df) > 0:
                    maxent_vals = sorted(combo_df["maxent_value"].unique())
                    conv_steps = sorted(combo_df["convergence_threshold_step"].unique())
                    conv_steps = [s for s in conv_steps if s > 0]

                    pivot_data = combo_df.pivot_table(
                        values="effective_sample_size",
                        index="maxent_value",
                        columns="convergence_threshold_step",
                        aggfunc="mean",
                    )

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
                        cmap="viridis",
                        cbar_kws={"label": "Effective Sample Size"},
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

        plt.tight_layout()
        plt.savefig(
            os.path.join(split_output_dir, "ess_heatmap_maxent_convergence.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  Saved: {split_type}/ess_heatmap_maxent_convergence.png")
        plt.close(fig)


def plot_kld_uniform_heatmaps(kl_ess_df, convergence_rates, output_dir):
    """
    Plot heatmaps of KL divergence against uniform distribution across convergence thresholds and maxent values.

    Args:
        kl_ess_df (pd.DataFrame): KL divergence analysis results
        convergence_rates (List[float]): List of convergence threshold values
        output_dir (str): Output directory for plots
    """
    print("Creating KLD vs Uniform heatmaps...")

    if kl_ess_df.empty:
        print("  No KLD data available for heatmaps")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    kl_ess_df["convergence_threshold_step"] = pd.to_numeric(
        kl_ess_df["convergence_threshold_step"], errors="coerce"
    )

    ensembles = kl_ess_df["ensemble"].unique()
    loss_functions = kl_ess_df["loss_function"].unique()
    split_types = kl_ess_df["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating KL heatmaps for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = kl_ess_df[kl_ess_df["split_type"] == split_type]

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
        )

        fig.suptitle(
            f"KL Divergence Heatmaps - {split_type} splits", fontsize=16, fontweight="bold"
        )

        for i, ensemble in enumerate(sorted(ensembles)):
            for j, loss_func in enumerate(sorted(loss_functions)):
                # Determine if axes is a single Axes object, a 1D array, or a 2D array
                if len(ensembles) == 1 and len(loss_functions) == 1:
                    ax = axes
                elif len(ensembles) == 1:
                    ax = axes[j]
                elif len(loss_functions) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                combo_df = split_df[
                    (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                ]

                if len(combo_df) > 0:
                    maxent_vals = sorted(combo_df["maxent_value"].unique())
                    conv_steps = sorted(combo_df["convergence_threshold_step"].unique())
                    conv_steps = [s for s in conv_steps if s > 0]

                    pivot_data = combo_df.pivot_table(
                        values="kl_divergence",
                        index="maxent_value",
                        columns="convergence_threshold_step",
                        aggfunc="mean",
                    )

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
                        fmt=".2f",
                        cmap="Blues",
                        vmin=0,
                        cbar_kws={"label": "KL Divergence"},
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

        plt.tight_layout()
        plt.savefig(
            os.path.join(split_output_dir, "kl_divergence_heatmap_maxent_convergence.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  Saved: {split_type}/kl_divergence_heatmap_maxent_convergence.png")
        plt.close(fig)


def plot_kld_between_splits_heatmap(kld_df, output_dir):
    """
    Plot heatmap of KL divergence between splits across maxent values and convergence steps
    using the same maxent-vs-convergence style as the other heatmaps.
    """
    print("Creating KLD between splits heatmap (maxent vs convergence)...")

    if kld_df.empty:
        print("  No KLD between splits data available for heatmap")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    kld_df["convergence_threshold_step"] = pd.to_numeric(
        kld_df["convergence_threshold_step"], errors="coerce"
    )

    ensembles = kld_df["ensemble"].unique()
    loss_functions = kld_df["loss_function"].unique()
    split_types = kld_df["split_type"].unique()

    for split_type in split_types:
        print(f"  Creating KLD between splits heatmap for split type: {split_type}")
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        split_df = kld_df[kld_df["split_type"] == split_type]

        fig, axes = plt.subplots(
            len(ensembles),
            len(loss_functions),
            figsize=(8 * len(loss_functions), 6 * len(ensembles)),
        )

        fig.suptitle(
            f"KL Divergence Between Splits - {split_type} splits", fontsize=16, fontweight="bold"
        )

        for i, ensemble in enumerate(sorted(ensembles)):
            for j, loss_func in enumerate(sorted(loss_functions)):
                # Determine axes object
                if len(ensembles) == 1 and len(loss_functions) == 1:
                    ax = axes
                elif len(ensembles) == 1:
                    ax = axes[j]
                elif len(loss_functions) == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                combo_df = split_df[
                    (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                ]

                if len(combo_df) > 0:
                    pivot_data = combo_df.pivot_table(
                        values="mean_kld_between_splits",
                        index="maxent_value",
                        columns="convergence_threshold_step",
                        aggfunc="mean",
                    )

                    # keep only positive step columns and sort
                    cols = sorted([c for c in pivot_data.columns if pd.notnull(c)])
                    pivot_data = pivot_data[cols]
                    pivot_data = pivot_data.sort_index(ascending=False)

                    col_labels = [f"Step {int(c)}" for c in pivot_data.columns]

                    sns.heatmap(
                        pivot_data,
                        annot=True,
                        fmt=".3f",
                        cmap="Blues",
                        cbar_kws={"label": "Mean KLD Between Splits"},
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

        plt.tight_layout()
        plt.savefig(
            os.path.join(split_output_dir, "kld_between_splits_heatmap_maxent_convergence.png"),
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  Saved: {split_type}/kld_between_splits_heatmap_maxent_convergence.png")
        plt.close(fig)


# ============================================================================
# ADDITIONAL PLOTTING FUNCTIONS
# ============================================================================


def plot_weight_distributions(weights_data, output_dir):
    """
    Plot weight distributions as line plots with maxent values as hue.
    Matches the style from the original script.
    """
    print("Creating weight distribution line plots...")

    if not weights_data:
        print("  No weights data available for plotting")
        return

    weights_df = pd.DataFrame(weights_data)

    # Create ensemble-loss combinations
    weights_df["ensemble_loss"] = weights_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    print(f"  Available data: {len(weights_df)} weight distributions")
    print(f"  Unique ensemble-loss combinations: {weights_df['ensemble_loss'].unique()}")
    print(f"  Unique maxent values: {sorted(weights_df['maxent_value'].unique())}")

    available_ensemble_loss = weights_df["ensemble_loss"].unique()

    for ensemble_loss in available_ensemble_loss:
        ensemble_loss_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if ensemble_loss_data.empty:
            continue

        print(f"  Creating weight distribution plot for: {ensemble_loss}")

        split_types = sorted(ensemble_loss_data["split_type"].unique())

        for split_type in split_types:
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]

            if split_data.empty:
                continue

            print(f"    Processing split type: {split_type}")

            # Group by maxent and compute average histogram across splits
            maxent_groups = {}
            for _, row in split_data.iterrows():
                maxent = row["maxent_value"]
                if maxent not in maxent_groups:
                    maxent_groups[maxent] = []
                maxent_groups[maxent].append(row["weights"])

            # Create colormap for maxent values
            maxent_values = sorted(maxent_groups.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))

            print(f"      MaxEnt values: {maxent_values}")

            # Define weight bins (log scale)
            weight_bins = np.logspace(-50, 0, 50)
            bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            for maxent_val, color in zip(maxent_values, colors):
                weights_list = maxent_groups[maxent_val]

                # Compute histogram for each split and average
                hist_counts = []
                for weights in weights_list:
                    if len(weights) > 0 and np.sum(weights) > 0:
                        counts, _ = np.histogram(weights, bins=weight_bins, density=True)
                        hist_counts.append(counts)

                if len(hist_counts) > 0:
                    # Average across splits
                    mean_counts = np.mean(hist_counts, axis=0)
                    std_counts = (
                        np.std(hist_counts, axis=0)
                        if len(hist_counts) > 1
                        else np.zeros_like(mean_counts)
                    )

                    # Plot line with error band
                    ax.plot(
                        bin_centers,
                        mean_counts,
                        color=color,
                        alpha=0.8,
                        label=f"MaxEnt={maxent_val:.0e}",
                        linewidth=2,
                    )
                    if len(hist_counts) > 1:
                        ax.fill_between(
                            bin_centers,
                            mean_counts - std_counts,
                            mean_counts + std_counts,
                            color=color,
                            alpha=0.2,
                        )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Weight Value", fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            ax.set_title(
                f"Weight Distributions - {ensemble_loss} - {split_name_mapping.get(split_type, split_type)}",
                fontsize=16,
                weight="bold",
            )
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3, linestyle="--")
            sns.despine()

            plt.tight_layout()

            # Save
            filename = f"weight_distributions_{ensemble_loss}_{split_type}.png".replace(
                "/", "_"
            ).replace(" ", "_")
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {filename}")
            plt.close()


def plot_conformational_recovery_scatter(recovery_df, output_dir):
    """
    Plot scatter plots of conformational state recovery vs maxent values.
    """
    print("Creating conformational recovery scatter plots...")

    if recovery_df.empty:
        print("  No recovery data available for plotting")
        return

    # Create ensemble-loss combinations
    recovery_df["ensemble_loss"] = recovery_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    available_ensemble_loss = recovery_df["ensemble_loss"].unique()
    state_names = [
        col.replace("_proportion", "") for col in recovery_df.columns if col.endswith("_proportion")
    ]

    for ensemble_loss in available_ensemble_loss:
        ensemble_loss_data = recovery_df[recovery_df["ensemble_loss"] == ensemble_loss]

        if ensemble_loss_data.empty:
            continue

        print(f"  Creating recovery plots for: {ensemble_loss}")

        split_types = sorted(ensemble_loss_data["split_type"].unique())

        # Create subplots for each state
        n_states = len(state_names)
        fig, axes = plt.subplots(1, n_states, figsize=(6 * n_states, 5))
        if n_states == 1:
            axes = [axes]

        for idx, state_name in enumerate(state_names):
            ax = axes[idx]
            col_name = f"{state_name}_proportion"

            for split_type in split_types:
                split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
                color = split_type_colours.get(split_type, "#9467bd")
                label = split_name_mapping.get(split_type, split_type)

                # Plot scatter
                ax.scatter(
                    split_data["maxent_value"],
                    split_data[col_name] * 100,  # Convert to percentage
                    c=[color],
                    alpha=0.7,
                    label=label,
                    s=60,
                    edgecolors="w",
                    linewidths=1.5,
                )

                # Connect points for each split
                for split_idx in split_data["split"].unique():
                    split_idx_data = split_data[split_data["split"] == split_idx]
                    if len(split_idx_data) > 1:
                        split_idx_data = split_idx_data.sort_values("maxent_value")
                        ax.plot(
                            split_idx_data["maxent_value"],
                            split_idx_data[col_name] * 100,
                            color=color,
                            alpha=0.3,
                            linewidth=1,
                        )

            ax.set_xscale("log")
            ax.set_xlabel("MaxEnt Value", fontsize=12)
            ax.set_ylabel("Proportion (%)", fontsize=12)
            ax.set_title(f"{state_name}", fontsize=14, weight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            if idx == 0:
                ax.legend(fontsize=10)

        plt.suptitle(
            f"Conformational Recovery - {ensemble_loss}", fontsize=16, weight="bold", y=1.02
        )
        sns.despine()
        plt.tight_layout()

        # Save
        filename = f"conformational_recovery_{ensemble_loss}.png".replace("/", "_").replace(
            " ", "_"
        )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


# ============================================================================


def calculate_conformational_recovery(weights_data, clustering_dir):
    """
    Calculate conformational state recovery for each optimization result.

    Args:
        weights_data: List of dictionaries containing weights
        clustering_dir: Directory containing clustering results

    Returns:
        DataFrame with recovery metrics
    """
    print("Calculating conformational recovery...")
    recovery_data = []

    for weight_dict in weights_data:
        ensemble = weight_dict["ensemble"]
        split_type = weight_dict["split_type"]
        split_idx = weight_dict["split"]
        loss_func = weight_dict["loss_function"]
        maxent_val = weight_dict["maxent_value"]
        weights = weight_dict["weights"]

        # Load clustering for this ensemble
        cluster_df = load_clustering_results(clustering_dir, ensemble)

        if cluster_df is None:
            print(f"  Skipping {ensemble} - no clustering data")
            continue

        # Ensure weights and clusters have same length
        if len(weights) != len(cluster_df):
            print(
                f"  Warning: Length mismatch for {ensemble} - weights:{len(weights)}, clusters:{len(cluster_df)}"
            )
            continue

        cluster_assignments = cluster_df["cluster_label"].values

        # Calculate weighted cluster proportions
        cluster_proportions = {}
        for cluster_id in np.unique(cluster_assignments):
            mask = cluster_assignments == cluster_id
            cluster_proportions[cluster_id] = float(np.sum(weights[mask]))

        # Map to states
        state_proportions = {}
        for cluster_id, state_name in STATE_MAPPING.items():
            if cluster_id in cluster_proportions:
                if state_name not in state_proportions:
                    state_proportions[state_name] = 0.0
                state_proportions[state_name] += cluster_proportions[cluster_id]

        # Store results
        recovery_dict = {
            "ensemble": ensemble,
            "split_type": split_type,
            "split": split_idx,
            "loss_function": loss_func,
            "maxent_value": maxent_val,
        }

        for state_name in STATE_MAPPING.values():
            recovery_dict[f"{state_name}_proportion"] = state_proportions.get(state_name, 0.0)

        recovery_data.append(recovery_dict)

    return pd.DataFrame(recovery_data)


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main function to run the complete analysis."""
    parser = argparse.ArgumentParser(description="MoPrP Weights validation analysis with heatmaps")
    parser.add_argument(
        "--results-dir",
        default="../fitting/jaxENT/_optimise_quick_test__20251016_160214",
        help="Results directory",
    )
    parser.add_argument(
        "--clustering-dir",
        default="_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters",
        help="Clustering directory",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument("--ema", action="store_true", default=False, help="Use EMA results")
    args = parser.parse_args()

    # Define parameters for MoPrP
    ensembles = ["AF2_MSAss", "AF2_filtered"]
    loss_functions = ["mcMSE", "MSE", "Sigma_MSE"]
    num_splits = 3
    maxent_values = [1.0, 10.0]  # Based on your directory structure

    # Resolve paths
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, args.results_dir)

    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]
    clustering_dir = os.path.join(script_dir, args.clustering_dir)

    # Determine output directory
    if args.output_dir:
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        base_name = os.path.basename(os.path.normpath(results_dir))
        output_dir = os.path.join(script_dir, f"_analysis{base_name}")

    print(f"Results directory: {results_dir}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Output directory: {output_dir}")
    print(f"EMA flag: {args.ema}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"Split types: {split_types}")
    print(f"MaxEnt values: {maxent_values}")
    print("-" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load optimization results
    print("\nLoading optimization results...")
    results = load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        split_types=split_types,
        num_splits=num_splits,
        maxent_values=maxent_values,
        EMA=args.ema,
    )

    # Extract KL divergence and ESS data
    print("\n" + "=" * 60)
    print("EXTRACTING KL DIVERGENCE AND ESS DATA")
    print("=" * 60)
    kl_ess_df = extract_frame_weights_kl_with_maxent(results)

    if len(kl_ess_df) > 0:
        kl_ess_path = os.path.join(output_dir, "kl_ess_data.csv")
        kl_ess_df.to_csv(kl_ess_path, index=False)
        print(f"Saved KL/ESS data to: {kl_ess_path}")

        # Extract convergence rates from the actual step_number values
        # Group by ensemble/loss/split to get convergence rates
        convergence_rates = []
        if "step_number" in kl_ess_df.columns:
            for _, group in kl_ess_df.groupby(
                ["ensemble", "loss_function", "maxent_value", "split"]
            ):
                if len(group) > 0:
                    # Get unique step numbers sorted
                    step_numbers = sorted(group["step_number"].unique())
                    if len(step_numbers) > len(convergence_rates):
                        convergence_rates = step_numbers
                    break  # Just need one example to get the pattern

        # If we couldn't extract convergence rates, create default ones
        if not convergence_rates:
            max_steps = kl_ess_df["convergence_threshold_step"].max()
            convergence_rates = list(range(1, int(max_steps) + 1))

        print(
            f"Detected convergence rates: {convergence_rates[:10]}..."
            if len(convergence_rates) > 10
            else f"Detected convergence rates: {convergence_rates}"
        )
    else:
        convergence_rates = []

    # Extract final weights
    print("\n" + "=" * 60)
    print("EXTRACTING FINAL WEIGHTS")
    print("=" * 60)
    weights_data = extract_final_weights(results)

    # Compute pairwise KLD between splits
    print("\n" + "=" * 60)
    print("COMPUTING KLD BETWEEN SPLITS")
    print("=" * 60)
    kld_df = compute_pairwise_kld_between_splits(results)

    if not kld_df.empty:
        kld_path = os.path.join(output_dir, "kld_between_splits_data.csv")
        kld_df.to_csv(kld_path, index=False)
        print(f"Saved KLD between splits data to: {kld_path}")

    # Calculate conformational recovery
    print("\n" + "=" * 60)
    print("CALCULATING CONFORMATIONAL RECOVERY")
    print("=" * 60)
    recovery_df = calculate_conformational_recovery(weights_data, clustering_dir)

    if not recovery_df.empty:
        recovery_path = os.path.join(output_dir, "conformational_recovery.csv")
        recovery_df.to_csv(recovery_path, index=False)
        print(f"Saved conformational recovery data to: {recovery_path}")

    # Create heatmaps
    print("\n" + "=" * 60)
    print("CREATING HEATMAPS")
    print("=" * 60)

    if not kl_ess_df.empty and convergence_rates:
        plot_ess_heatmaps(kl_ess_df, convergence_rates, output_dir)
        plot_kld_uniform_heatmaps(kl_ess_df, convergence_rates, output_dir)
    else:
        print("  Skipping heatmaps - no data or convergence rates available")

    if not kld_df.empty:
        plot_kld_between_splits_heatmap(kld_df, output_dir)

    # Create additional plots
    print("\n" + "=" * 60)
    print("CREATING ADDITIONAL PLOTS")
    print("=" * 60)

    if weights_data:
        plot_weight_distributions(weights_data, output_dir)

    if not recovery_df.empty:
        plot_conformational_recovery_scatter(recovery_df, output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
