""" """

import os
import re
import sys
from typing import Dict, List, Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms

# Add the base directory to the path to import the HDF5 utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

# Import the HDF5 loading functions from the provided script
from jaxent.src.utils.hdf import load_optimization_history_from_file


def extract_maxent_value_from_filename(filename: str) -> Optional[float]:
    """
    Extract maxent value from filename.

    Args:
        filename: HDF5 filename containing maxent value

    Returns:
        Maxent value or None if not found
    """
    match = re.search(r"maxent(\d+(?:\.\d+)?)", filename)
    if match:
        return float(match.group(1))
    return None


def compute_rmsd_to_references(trajectory_path, topology_path, reference_paths):
    """
    Compute RMSD of trajectory frames to reference structures.

    Args:
        trajectory_path (str): Path to trajectory file
        topology_path (str): Path to topology file
        reference_paths (list): List of paths to reference structures

    Returns:
        np.ndarray: RMSD values (n_frames, n_refs)
    """
    # Load trajectory
    traj = mda.Universe(topology_path, trajectory_path)

    # Initialize RMSD arrays
    n_frames = len(traj.trajectory)
    n_refs = len(reference_paths)
    rmsd_values = np.zeros((n_frames, n_refs))

    # Compute RMSD for each reference structure
    for j, ref_path in enumerate(reference_paths):
        # Create a new Universe with the trajectory and reference selection
        mobile = mda.Universe(topology_path, trajectory_path)
        reference = mda.Universe(ref_path)

        # Select CA atoms
        mobile_ca = mobile.select_atoms("name CA")
        ref_ca = reference.select_atoms("name CA")

        # Ensure selecting same atoms from both
        if len(ref_ca) != len(mobile_ca):
            print(
                f"Warning: CA atom count mismatch - Trajectory: {len(mobile_ca)}, Reference {j}: {len(ref_ca)}"
            )

        # Calculate RMSD
        R = rms.RMSD(mobile, reference, select="name CA", ref_frame=0)
        R.run()

        # Store RMSD values (column 2 has the RMSD after rotation)
        rmsd_values[:, j] = R.rmsd[:, 2]

    return rmsd_values


def cluster_by_rmsd(rmsd_values, rmsd_threshold=1.0):
    """
    Cluster frames based on RMSD to reference structures.

    Args:
        rmsd_values (np.ndarray): RMSD values to reference structures (n_frames, n_refs)
        rmsd_threshold (float): RMSD threshold for clustering

    Returns:
        np.ndarray: Cluster assignments (0 = open-like, 1 = closed-like)
    """
    # Simple clustering: assign to closest reference if within threshold
    cluster_assignments = np.argmin(rmsd_values, axis=1)

    # Check if frames are within threshold of any reference
    min_rmsd = np.min(rmsd_values, axis=1)
    valid_clusters = min_rmsd <= rmsd_threshold

    # Set invalid clusters to -1
    cluster_assignments[~valid_clusters] = -1

    return cluster_assignments


def calculate_cluster_ratios(cluster_assignments, frame_weights=None):
    """
    Calculate ratios of clusters based on assignments and optional frame weights.

    Args:
        cluster_assignments (np.ndarray): Cluster assignments
        frame_weights (np.ndarray, optional): Frame weights from optimization

    Returns:
        dict: Cluster ratios
    """
    if frame_weights is None:
        frame_weights = np.ones(len(cluster_assignments))

    # Normalize frame weights
    frame_weights = frame_weights / np.sum(frame_weights)

    # Calculate weighted ratios
    ratios = {}
    unique_clusters = np.unique(cluster_assignments)

    for cluster in unique_clusters:
        if cluster >= 0:  # Skip invalid clusters (-1)
            mask = cluster_assignments == cluster
            ratios[f"cluster_{cluster}"] = np.sum(frame_weights[mask])

    return ratios


def calculate_recovery_percentage(observed_ratios, ground_truth_ratios):
    """
    Calculate recovery percentage of conformational ratios.

    Args:
        observed_ratios (dict): Observed cluster ratios
        ground_truth_ratios (dict): Ground truth ratios (60:40 Open:Closed)

    Returns:
        dict: Recovery percentages
    """
    recovery = {}

    # Assuming cluster_0 is open-like and cluster_1 is closed-like
    open_observed = observed_ratios.get("cluster_0", 0.0)
    closed_observed = observed_ratios.get("cluster_1", 0.0)

    open_truth = ground_truth_ratios.get("open", 0.4)
    closed_truth = ground_truth_ratios.get("closed", 0.6)

    # Calculate recovery as percentage of truth recovered
    if open_truth > 0:
        # recovery["open_recovery"] = min(100.0, (open_observed / open_truth) * 100.0)
        recovery["open_recovery"] = min(200.0, (open_observed / open_truth) * 100.0)

    else:
        recovery["open_recovery"] = 0.0

    if closed_truth > 0:
        # recovery["closed_recovery"] = min(100.0, (closed_observed / closed_truth) * 100.0)
        recovery["closed_recovery"] = min(200.0, (closed_observed / closed_truth) * 100.0)

    else:
        recovery["closed_recovery"] = 0.0

    return recovery


def load_all_optimization_results_with_maxent(
    results_dir: str,
    ensembles: List[str] = ["ISO_TRI", "ISO_BI"],
    loss_functions: List[str] = ["mcMSE", "MSE"],
    num_splits: int = 3,
    maxent_values: List[float] = None,
) -> Dict:
    """
    Load all optimization results from HDF5 files, including maxent values.

    Args:
        results_dir: Directory containing subdirectories for each split type.
        ensembles: List of ensemble names.
        loss_functions: List of loss function names.
        num_splits: Number of data splits per type.
        maxent_values: List of expected maxent values (if None, will discover from files).

    Returns:
        Dictionary with results organized by split_type, ensemble, loss, maxent, and split.
    """
    results = {}
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return results

    split_types = [
        d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))
    ]

    for split_type in split_types:
        results[split_type] = {}
        split_type_dir = os.path.join(results_dir, split_type)

        for ensemble in ensembles:
            results[split_type][ensemble] = {}

            for loss_name in loss_functions:
                results[split_type][ensemble][loss_name] = {}

                # Discover all files for this ensemble/loss combination
                pattern = f"{ensemble}_{loss_name}_{split_type}_split"
                files = [
                    f
                    for f in os.listdir(split_type_dir)
                    if f.startswith(pattern) and f.endswith(".hdf5")
                ]

                for filename in files:
                    # Extract split index and maxent value
                    match = re.search(r"split(\d{3})_maxent(\d+(?:\.\d+)?)", filename)
                    if match:
                        split_idx = int(match.group(1))
                        maxent_val = float(match.group(2))

                        # Initialize nested dict if needed
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
                    else:
                        # Handle files without maxent value (original optimization)
                        match = re.search(r"split(\d{3})", filename)
                        if match:
                            split_idx = int(match.group(1))
                            maxent_val = 0.0  # Use 0 for no maxent regularization

                            if maxent_val not in results[split_type][ensemble][loss_name]:
                                results[split_type][ensemble][loss_name][maxent_val] = {}

                            filepath = os.path.join(split_type_dir, filename)

                            try:
                                history = load_optimization_history_from_file(filepath)
                                results[split_type][ensemble][loss_name][maxent_val][split_idx] = (
                                    history
                                )
                                print(f"Loaded (no maxent): {filepath}")
                            except Exception as e:
                                print(f"Failed to load {filepath}: {e}")
                                results[split_type][ensemble][loss_name][maxent_val][split_idx] = (
                                    None
                                )

    return results


def analyze_conformational_recovery_with_maxent(
    trajectory_paths, topology_path, reference_paths, results_dict
):
    """
    Analyze conformational ratio recovery for trajectories with maxent values.

    Args:
        trajectory_paths (dict): Dictionary of trajectory paths by ensemble name
        topology_path (str): Path to topology file
        reference_paths (list): Paths to reference structures [open, closed]
        results_dict (dict): Optimization results containing frame weights

    Returns:
        pd.DataFrame: Recovery analysis results
    """
    ground_truth_ratios = {"open": 0.4, "closed": 0.6}
    recovery_data = []

    for ensemble_name, traj_path in trajectory_paths.items():
        print(f"Analyzing conformational recovery for {ensemble_name}...")

        # Compute RMSD to references
        rmsd_values = compute_rmsd_to_references(traj_path, topology_path, reference_paths)

        # Cluster by RMSD
        cluster_assignments = cluster_by_rmsd(rmsd_values, rmsd_threshold=1.0)

        # Calculate unweighted (original) ratios
        original_ratios = calculate_cluster_ratios(cluster_assignments)
        original_recovery = calculate_recovery_percentage(original_ratios, ground_truth_ratios)

        recovery_data.append(
            {
                "ensemble": ensemble_name,
                "loss_function": "Original",
                "split_type": "N/A",
                "split": "N/A",
                "maxent_value": 0.0,
                "convergence_step": "N/A",
                "open_ratio": original_ratios.get("cluster_0", 0.0),
                "closed_ratio": original_ratios.get("cluster_1", 0.0),
                "open_recovery": original_recovery["open_recovery"],
                "closed_recovery": original_recovery["closed_recovery"],
                "total_frames": len(cluster_assignments),
                "clustered_frames": np.sum(cluster_assignments >= 0),
            }
        )

        # Analyze with optimized frame weights including maxent
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
                                            len(frame_weights) == len(cluster_assignments)
                                            and np.sum(frame_weights) > 0
                                        ):
                                            # Calculate weighted ratios
                                            weighted_ratios = calculate_cluster_ratios(
                                                cluster_assignments, frame_weights
                                            )
                                            weighted_recovery = calculate_recovery_percentage(
                                                weighted_ratios, ground_truth_ratios
                                            )

                                            recovery_data.append(
                                                {
                                                    "ensemble": ensemble_name,
                                                    "loss_function": loss_name,
                                                    "split_type": split_type,
                                                    "split": split_idx,
                                                    "maxent_value": maxent_val,
                                                    "convergence_step": step_idx,
                                                    "open_ratio": weighted_ratios.get(
                                                        "cluster_0", 0.0
                                                    ),
                                                    "closed_ratio": weighted_ratios.get(
                                                        "cluster_1", 0.0
                                                    ),
                                                    "open_recovery": weighted_recovery[
                                                        "open_recovery"
                                                    ],
                                                    "closed_recovery": weighted_recovery[
                                                        "closed_recovery"
                                                    ],
                                                    "total_frames": len(cluster_assignments),
                                                    "clustered_frames": np.sum(
                                                        cluster_assignments >= 0
                                                    ),
                                                }
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


def extract_frame_weights_kl_with_maxent(results: Dict) -> pd.DataFrame:
    """
    Extract frame weights and calculate KL divergence and ESS including maxent values.

    Args:
        results: Dictionary containing optimization histories.

    Returns:
        DataFrame containing KL divergence and ESS values for analysis.
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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-ready style
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

# Define color schemes
ensemble_loss_colours = {
    "ISO_TRI_MSE": "#1f77b4",  # blue
    "ISO_TRI_mcMSE": "#ff7f0e",  # orange
    "ISO_BI_MSE": "#2ca02c",  # green
    "ISO_BI_mcMSE": "#d62728",  # red
}

split_type_colours = {
    "R3": "#9467bd",  # purple
    "Sp": "#8c564b",  # brown
    "r": "#e377c2",  # pink
    "s": "#7f7f7f",  # gray
}

split_name_mapping = {"R3": "Non-Redundant", "Sp": "Spatial", "r": "Random", "s": "Sequence"}


def create_ensemble_loss_key(ensemble, loss_function):
    """Create a unique key for ensemble-loss combinations."""
    return f"{ensemble}_{loss_function}"


def plot_weight_distribution_lines(weights_data, output_dir):
    """
    Plot weight distributions as 2D line plots with maxent values as hue.
    Modified to handle ensemble-loss combinations as separate entities.
    """
    print("Creating weight distribution line plots...")

    # Convert to DataFrame for easier manipulation
    weights_df = pd.DataFrame(weights_data)

    if weights_df.empty:
        print("  No weights data available for plotting")
        return

    # Create ensemble-loss combinations
    weights_df["ensemble_loss"] = weights_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    print(f"  Available data: {len(weights_df)} weight distributions")
    print(f"  Unique ensemble-loss combinations: {weights_df['ensemble_loss'].unique()}")
    print(f"  Unique split types: {weights_df['split_type'].unique()}")
    print(f"  Unique maxent values: {sorted(weights_df['maxent_value'].unique())}")

    # Create plots for each ensemble-loss combination
    available_ensemble_loss = weights_df["ensemble_loss"].unique()

    for ensemble_loss in available_ensemble_loss:
        ensemble_loss_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if ensemble_loss_data.empty:
            continue

        print(f"  Creating plots for ensemble-loss: {ensemble_loss}")

        # Get unique split types for this ensemble-loss combination
        split_types = sorted(ensemble_loss_data["split_type"].unique())
        print(f"    Split types: {split_types}")

        if not split_types:
            continue

        # Create figure with subplots
        n_plots = min(len(split_types), 4)  # Max 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, split_type in enumerate(split_types[:4]):  # Max 4 subplots
            ax = axes[idx]
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]

            if split_data.empty:
                ax.set_visible(False)
                continue

            print(f"      Split {split_type}: {len(split_data)} data points")

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

            print(f"        MaxEnt values: {maxent_values}")

            # Define weight bins
            weight_bins = np.logspace(-50, 0, 50)
            bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

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
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Density")
            ax.set_title(f"{split_name_mapping.get(split_type, split_type)}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(split_types), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Weight Distributions - {ensemble_loss}", fontsize=16, y=0.98)
        plt.tight_layout()

        # Save figure
        filename = (
            f"weight_distributions_lines_{ensemble_loss.replace('/', '_').replace(' ', '_')}.png"
        )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


def plot_weight_recovery_scatter(recovery_df, output_dir):
    """
    Plot scatter plots of open state recovery vs maxent parameters.
    Modified to handle ensemble-loss combinations.
    """
    print("Creating open state recovery scatter plots...")

    if recovery_df.empty:
        print("  No recovery data available for plotting.")
        return

    # Create ensemble-loss combinations
    recovery_df["ensemble_loss"] = recovery_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    print(f"  Available recovery data: {len(recovery_df)} points")
    print(f"  Unique ensemble-loss combinations: {recovery_df['ensemble_loss'].unique()}")
    print(f"  Unique split types: {recovery_df['split_type'].unique()}")

    # Get unique ensemble-loss combinations from the actual data
    available_ensemble_loss = recovery_df["ensemble_loss"].unique()

    for ensemble_loss in available_ensemble_loss:
        ensemble_loss_data = recovery_df[recovery_df["ensemble_loss"] == ensemble_loss]

        if ensemble_loss_data.empty:
            continue

        print(f"  Creating recovery plot for ensemble-loss: {ensemble_loss}")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot for each split type
        available_split_types = ensemble_loss_data["split_type"].unique()

        # Create a color map for split types
        n_split_types = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_split_types))
        split_color_map = dict(zip(available_split_types, colors))

        for i, split_type in enumerate(available_split_types):
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
            color = split_color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)

            print(f"    Split {split_type}: {len(split_data)} points")

            # Plot scatter
            ax.scatter(
                split_data["maxent_value"],
                split_data["open_recovery"],
                c=[color],
                alpha=0.7,
                label=label,
                s=60,
                edgecolors="w",
            )

            # Connect points for each split
            for split_idx in split_data["split"].unique():
                split_idx_data = split_data[split_data["split"] == split_idx]
                if len(split_idx_data) > 1:
                    split_idx_data = split_idx_data.sort_values("maxent_value")
                    ax.plot(
                        split_idx_data["maxent_value"],
                        split_idx_data["open_recovery"],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                    )

        ax.set_xscale("log")
        ax.set_xlabel("MaxEnt Value")
        ax.set_ylabel("Open State Recovery (%)")
        ax.set_title(f"Open State Recovery vs MaxEnt - {ensemble_loss}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        filename = (
            f"open_state_recovery_scatter_{ensemble_loss.replace('/', '_').replace(' ', '_')}.png"
        )
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {filename}")
        plt.close()


def plot_kld_between_splits(kld_df, output_dir):
    """
    Plot mean KLD between splits across maxent values.
    Modified to handle ensemble-loss combinations.
    """
    print("Creating KLD between splits plot...")

    if kld_df.empty:
        print("  No KLD data available for plotting.")
        return

    # Create ensemble-loss combinations
    kld_df["ensemble_loss"] = kld_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    print(f"  Available KLD data: {len(kld_df)} points")
    print(f"  Unique ensemble-loss combinations: {kld_df['ensemble_loss'].unique()}")
    print(f"  Unique split types: {kld_df['split_type'].unique()}")

    # Get unique ensemble-loss combinations from actual data
    available_ensemble_loss = kld_df["ensemble_loss"].unique()

    # Create figure with subplots for each ensemble-loss combination
    n_combinations = len(available_ensemble_loss)
    n_cols = min(n_combinations, 2)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    if n_combinations == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, ensemble_loss in enumerate(available_ensemble_loss):
        if idx >= len(axes):
            break

        ax = axes[idx]
        ensemble_loss_data = kld_df[kld_df["ensemble_loss"] == ensemble_loss]

        if ensemble_loss_data.empty:
            ax.set_visible(False)
            continue

        print(f"  Creating KLD plot for ensemble-loss: {ensemble_loss}")

        # Get available split types
        available_split_types = ensemble_loss_data["split_type"].unique()
        n_split_types = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_split_types))
        split_color_map = dict(zip(available_split_types, colors))

        # Plot each split type
        for split_type in available_split_types:
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
            color = split_color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)

            print(f"    Split {split_type}: {len(split_data)} points")

            # Sort by maxent for proper line plotting
            split_data = split_data.sort_values("maxent_value")

            x_vals = split_data["maxent_value"].values
            y_vals = split_data["mean_kld_between_splits"].values
            y_err = split_data["sem_kld_between_splits"].values

            # Plot line with error bars
            ax.errorbar(
                x_vals,
                y_vals,
                yerr=y_err,
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

    # Hide unused subplots
    for idx in range(len(available_ensemble_loss), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("KL Divergence Between Splits Across MaxEnt Values", fontsize=16)
    plt.tight_layout()

    # Save figure
    filename = "kld_between_splits_vs_maxent.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


def plot_sequential_maxent_kld(sequential_kld_df, output_dir):
    """
    Plot KLD between sequential maxent values.
    Modified to handle ensemble-loss combinations.
    """
    print("Creating sequential maxent KLD plot...")

    if sequential_kld_df.empty:
        print("  No sequential KLD data available for plotting.")
        return

    # Create ensemble-loss combinations
    sequential_kld_df["ensemble_loss"] = sequential_kld_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    print(f"  Available sequential KLD data: {len(sequential_kld_df)} points")
    print(f"  Unique ensemble-loss combinations: {sequential_kld_df['ensemble_loss'].unique()}")
    print(f"  Unique split types: {sequential_kld_df['split_type'].unique()}")

    # Get unique ensemble-loss combinations from actual data
    available_ensemble_loss = sequential_kld_df["ensemble_loss"].unique()

    # Create figure with subplots for each ensemble-loss combination
    n_combinations = len(available_ensemble_loss)
    n_cols = min(n_combinations, 2)
    n_rows = (n_combinations + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    if n_combinations == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, ensemble_loss in enumerate(available_ensemble_loss):
        if idx >= len(axes):
            break

        ax = axes[idx]
        ensemble_loss_data = sequential_kld_df[sequential_kld_df["ensemble_loss"] == ensemble_loss]

        if ensemble_loss_data.empty:
            ax.set_visible(False)
            continue

        print(f"  Creating sequential KLD plot for ensemble-loss: {ensemble_loss}")

        # Get available split types
        available_split_types = ensemble_loss_data["split_type"].unique()
        n_split_types = len(available_split_types)
        colors = plt.cm.Set1(np.linspace(0, 1, n_split_types))
        split_color_map = dict(zip(available_split_types, colors))

        # Plot each split type
        for split_type in available_split_types:
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
            color = split_color_map[split_type]
            label = split_name_mapping.get(split_type, split_type)

            print(f"    Split {split_type}: {len(split_data)} points")

            # Plot individual splits as light lines
            for split_idx in split_data["split_idx"].unique():
                split_idx_data = split_data[split_data["split_idx"] == split_idx].sort_values(
                    "current_maxent"
                )

                if len(split_idx_data) > 0:
                    x_vals = split_idx_data["current_maxent"].values
                    y_vals = split_idx_data["kld_to_previous"].values

                    ax.plot(
                        x_vals,
                        y_vals,
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                        marker=".",
                        markersize=2,
                    )

            # Compute and plot mean with error bars
            maxent_stats = (
                split_data.groupby("current_maxent")["kld_to_previous"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )

            if len(maxent_stats) > 0:
                x_vals = maxent_stats["current_maxent"].values
                y_vals = maxent_stats["mean"].values
                y_err = maxent_stats["std"].values / np.sqrt(maxent_stats["count"].values)

                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=y_err,
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

    # Hide unused subplots
    for idx in range(len(available_ensemble_loss), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("KL Divergence Between Sequential MaxEnt Values", fontsize=16)
    plt.tight_layout()

    # Save figure
    filename = "sequential_maxent_kld_vs_maxent.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


def compute_pairwise_kld_between_splits(weights_data):
    """
    Compute pairwise KLD between splits for each ensemble, split_type, loss, and maxent combination.
    """
    print("Computing pairwise KLD between splits...")
    kld_data = []

    # Convert to DataFrame for easier grouping
    weights_df = pd.DataFrame(weights_data)

    # Group by ensemble, split_type, loss_function, and maxent_value
    for (ensemble, split_type, loss_func, maxent_val), group in weights_df.groupby(
        ["ensemble", "split_type", "loss_function", "maxent_value"]
    ):
        splits = group["split"].values
        weights_list = group["weights"].tolist()

        if len(splits) < 2:
            continue

        # Compute pairwise KLD between all pairs of splits
        pairwise_klds = []

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                weights_i = weights_list[i]
                weights_j = weights_list[j]

                # Ensure both weight arrays have the same length
                min_len = min(len(weights_i), len(weights_j))
                weights_i = weights_i[:min_len]
                weights_j = weights_j[:min_len]

                # Compute KLD in both directions and take the average (symmetric)
                kld_ij = compute_kl_divergence_between_distributions(weights_i, weights_j)
                kld_ji = compute_kl_divergence_between_distributions(weights_j, weights_i)

                if not (np.isnan(kld_ij) or np.isnan(kld_ji)):
                    avg_kld = (kld_ij + kld_ji) / 2.0
                    pairwise_klds.append(avg_kld)

        if len(pairwise_klds) > 0:
            mean_kld = np.mean(pairwise_klds)
            std_kld = np.std(pairwise_klds)
            sem_kld = std_kld / np.sqrt(len(pairwise_klds))

            kld_data.append(
                {
                    "ensemble": ensemble,
                    "split_type": split_type,
                    "split_name": split_name_mapping.get(split_type, split_type),
                    "loss_function": loss_func,
                    "maxent_value": maxent_val,
                    "mean_kld_between_splits": mean_kld,
                    "std_kld_between_splits": std_kld,
                    "sem_kld_between_splits": sem_kld,
                    "n_pairs": len(pairwise_klds),
                    "n_splits": len(splits),
                }
            )

    return pd.DataFrame(kld_data)


def compute_sequential_maxent_kld(weights_data):
    """
    Compute KLD between sequential maxent values for each ensemble, split_type, loss, and split combination.
    """
    print("Computing KLD between sequential maxent values...")
    sequential_kld_data = []

    # Convert to DataFrame for easier grouping
    weights_df = pd.DataFrame(weights_data)

    # Group by ensemble, split_type, loss_function, and split
    for (ensemble, split_type, loss_func, split_idx), group in weights_df.groupby(
        ["ensemble", "split_type", "loss_function", "split"]
    ):
        # Sort by maxent_value for proper sequential comparison
        group_sorted = group.sort_values("maxent_value")
        maxent_values = group_sorted["maxent_value"].values
        weights_list = group_sorted["weights"].tolist()

        if len(maxent_values) < 2:
            continue

        # For each maxent (except the first), compute KLD with previous maxent
        for i in range(len(maxent_values)):
            current_maxent = maxent_values[i]
            current_weights = weights_list[i]

            if i == 0:
                # Compare first (lowest) maxent to uniform distribution
                n_frames = len(current_weights)
                uniform_weights = np.ones(n_frames) / n_frames

                kld_to_previous = compute_kl_divergence_between_distributions(
                    current_weights, uniform_weights
                )
                previous_maxent = None
                comparison_type = "vs_uniform"
            else:
                # Compare to previous maxent
                previous_maxent = maxent_values[i - 1]
                previous_weights = weights_list[i - 1]

                # Ensure both weight arrays have the same length
                min_len = min(len(current_weights), len(previous_weights))
                current_weights_trimmed = current_weights[:min_len]
                previous_weights_trimmed = previous_weights[:min_len]

                kld_to_previous = compute_kl_divergence_between_distributions(
                    current_weights_trimmed, previous_weights_trimmed
                )
                comparison_type = "vs_previous_maxent"

            if not np.isnan(kld_to_previous):
                sequential_kld_data.append(
                    {
                        "ensemble": ensemble,
                        "split_type": split_type,
                        "split_name": split_name_mapping.get(split_type, split_type),
                        "loss_function": loss_func,
                        "split_idx": split_idx,
                        "current_maxent": current_maxent,
                        "previous_maxent": previous_maxent,
                        "kld_to_previous": kld_to_previous,
                        "comparison_type": comparison_type,
                    }
                )

    return pd.DataFrame(sequential_kld_data)


def compute_kl_divergence_between_distributions(p, q, epsilon=1e-10):
    """
    Compute KL divergence between two probability distributions.
    """
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


def extract_weights_over_convergence_steps(results):
    """
    Extract frame weights at different convergence steps for plotting.
    This creates a dataset with weights at multiple steps in the optimization process.
    """
    print("Extracting weights over convergence steps...")
    convergence_weights_data = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is not None and history.states:
                            # Extract weights from multiple steps (not just final)
                            n_states = len(history.states)

                            # Sample steps across the optimization trajectory
                            if n_states >= 10:
                                # Take steps at different fractions of completion
                                step_fractions = [
                                    0.1,
                                    0.3,
                                    0.5,
                                    0.7,
                                    0.9,
                                    1.0,
                                ]  # 10%, 30%, 50%, 70%, 90%, 100%
                                step_indices = [int(f * (n_states - 1)) for f in step_fractions]
                            else:
                                # If few states, take all available
                                step_indices = list(range(n_states))

                            for step_idx in step_indices:
                                state = history.states[step_idx]
                                if (
                                    hasattr(state, "params")
                                    and hasattr(state.params, "frame_weights")
                                    and state.params.frame_weights is not None
                                ):
                                    frame_weights = np.array(state.params.frame_weights)

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

                                        # Calculate convergence fraction
                                        convergence_fraction = (
                                            step_idx / (n_states - 1) if n_states > 1 else 1.0
                                        )

                                        convergence_weights_data.append(
                                            {
                                                "ensemble": ensemble,
                                                "split_type": split_type,
                                                "split": split_idx,
                                                "loss_function": loss_name,
                                                "maxent_value": maxent_val,
                                                "weights": frame_weights,
                                                "convergence_step": step_idx,
                                                "convergence_fraction": convergence_fraction,
                                                "total_steps": n_states,
                                            }
                                        )

    print(
        f"Extracted {len(convergence_weights_data)} weight distributions across convergence steps"
    )
    return convergence_weights_data


def plot_weight_distribution_maxent_panels(convergence_weights_data, output_dir):
    """
    Plot weight distributions with maxent values as panels and convergence steps as lines within panels.
    Each panel represents a different maxent value, and lines show evolution over convergence.
    """
    print("Creating weight distribution plots with maxent panels...")

    if not convergence_weights_data:
        print("  No convergence weights data available for plotting")
        return

    # Convert to DataFrame for easier manipulation
    weights_df = pd.DataFrame(convergence_weights_data)

    # Create ensemble-loss combinations
    weights_df["ensemble_loss"] = weights_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    print(f"  Available data: {len(weights_df)} weight distributions across steps")
    print(f"  Unique ensemble-loss combinations: {weights_df['ensemble_loss'].unique()}")
    print(f"  Unique split types: {weights_df['split_type'].unique()}")
    print(f"  Unique maxent values: {sorted(weights_df['maxent_value'].unique())}")

    # Get available ensemble-loss combinations
    available_ensemble_loss = weights_df["ensemble_loss"].unique()

    for ensemble_loss in available_ensemble_loss:
        ensemble_loss_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if ensemble_loss_data.empty:
            continue

        print(f"  Creating maxent panel plots for ensemble-loss: {ensemble_loss}")

        # Get unique split types for this ensemble-loss combination
        split_types = sorted(ensemble_loss_data["split_type"].unique())

        for split_type in split_types:
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
            if split_data.empty:
                continue

            print(f"    Processing split type: {split_type}")

            # Get unique maxent values for this combination
            maxent_values = sorted(split_data["maxent_value"].unique())

            if len(maxent_values) < 2:
                print(f"      Skipping {split_type} - insufficient maxent values")
                continue

            # Create subplot grid for maxent values
            n_maxent = len(maxent_values)
            n_cols = min(4, n_maxent)  # Max 4 columns
            n_rows = (n_maxent + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_maxent == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()

            for idx, maxent_val in enumerate(maxent_values):
                if idx >= len(axes):
                    break

                ax = axes[idx]
                maxent_data = split_data[split_data["maxent_value"] == maxent_val]

                if maxent_data.empty:
                    ax.set_visible(False)
                    continue

                print(f"      MaxEnt {maxent_val}: {len(maxent_data)} data points")

                # Group by convergence fraction and compute average histogram across splits
                convergence_groups = {}
                for _, row in maxent_data.iterrows():
                    conv_frac = row["convergence_fraction"]
                    if conv_frac not in convergence_groups:
                        convergence_groups[conv_frac] = []
                    convergence_groups[conv_frac].append(row["weights"])

                # Create colormap for convergence fractions
                conv_fractions = sorted(convergence_groups.keys())
                colors = plt.cm.plasma(np.linspace(0, 1, len(conv_fractions)))

                print(f"        Convergence fractions: {conv_fractions}")

                # Define weight bins (log scale)
                weight_bins = np.logspace(-50, 0, 50)
                bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

                for conv_frac, color in zip(conv_fractions, colors):
                    weights_list = convergence_groups[conv_frac]

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
                            label=f"{conv_frac:.1%}",
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
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Density")
                ax.set_title(f"MaxEnt = {maxent_val:.0e}")
                ax.legend(title="Convergence", fontsize=8, title_fontsize=8)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(len(maxent_values), len(axes)):
                axes[idx].set_visible(False)

            # Set overall title
            split_name = split_name_mapping.get(split_type, split_type)
            plt.suptitle(
                f"Weight Evolution Over Convergence - {ensemble_loss} - {split_name}",
                fontsize=16,
                y=0.98,
            )
            plt.tight_layout()

            # Save figure
            filename = (
                f"weight_distributions_maxent_panels_{ensemble_loss}_{split_type}".replace(
                    "/", "_"
                ).replace(" ", "_")
                + ".png"
            )
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {filename}")
            plt.close()


def plot_weight_distribution_convergence_panels(convergence_weights_data, output_dir):
    """
    Alternative version: Plot weight distributions with convergence fractions as panels
    and maxent values as lines within panels.
    """
    print("Creating weight distribution plots with convergence panels...")

    if not convergence_weights_data:
        print("  No convergence weights data available for plotting")
        return

    # Convert to DataFrame for easier manipulation
    weights_df = pd.DataFrame(convergence_weights_data)

    # Create ensemble-loss combinations
    weights_df["ensemble_loss"] = weights_df.apply(
        lambda row: create_ensemble_loss_key(row["ensemble"], row["loss_function"]), axis=1
    )

    # Get available ensemble-loss combinations
    available_ensemble_loss = weights_df["ensemble_loss"].unique()

    for ensemble_loss in available_ensemble_loss:
        ensemble_loss_data = weights_df[weights_df["ensemble_loss"] == ensemble_loss]
        if ensemble_loss_data.empty:
            continue

        print(f"  Creating convergence panel plots for ensemble-loss: {ensemble_loss}")

        # Get unique split types for this ensemble-loss combination
        split_types = sorted(ensemble_loss_data["split_type"].unique())

        for split_type in split_types:
            split_data = ensemble_loss_data[ensemble_loss_data["split_type"] == split_type]
            if split_data.empty:
                continue

            print(f"    Processing split type: {split_type}")

            # Get unique convergence fractions
            conv_fractions = sorted(split_data["convergence_fraction"].unique())

            if len(conv_fractions) < 2:
                print(f"      Skipping {split_type} - insufficient convergence fractions")
                continue

            # Create subplot grid for convergence fractions
            n_conv = len(conv_fractions)
            n_cols = min(3, n_conv)  # Max 3 columns
            n_rows = (n_conv + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_conv == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()

            for idx, conv_frac in enumerate(conv_fractions):
                if idx >= len(axes):
                    break

                ax = axes[idx]
                conv_data = split_data[split_data["convergence_fraction"] == conv_frac]

                if conv_data.empty:
                    ax.set_visible(False)
                    continue

                print(f"      Convergence {conv_frac:.1%}: {len(conv_data)} data points")

                # Group by maxent value
                maxent_groups = {}
                for _, row in conv_data.iterrows():
                    maxent = row["maxent_value"]
                    if maxent not in maxent_groups:
                        maxent_groups[maxent] = []
                    maxent_groups[maxent].append(row["weights"])

                # Create colormap for maxent values
                maxent_values = sorted(maxent_groups.keys())
                colors = plt.cm.viridis(np.linspace(0, 1, len(maxent_values)))

                # Define weight bins (log scale)
                weight_bins = np.logspace(-50, 0, 50)
                bin_centers = (weight_bins[:-1] + weight_bins[1:]) / 2

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

                        # Plot line
                        ax.plot(
                            bin_centers,
                            mean_counts,
                            color=color,
                            alpha=0.8,
                            label=f"{maxent_val:.0e}",
                            linewidth=2,
                        )

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Density")
                ax.set_title(f"Convergence: {conv_frac:.1%}")
                ax.legend(title="MaxEnt", fontsize=8, title_fontsize=8)
                ax.grid(True, alpha=0.3)

            # Hide unused subplots
            for idx in range(len(conv_fractions), len(axes)):
                axes[idx].set_visible(False)

            # Set overall title
            split_name = split_name_mapping.get(split_type, split_type)
            plt.suptitle(
                f"MaxEnt Comparison Across Convergence - {ensemble_loss} - {split_name}",
                fontsize=16,
                y=0.98,
            )
            plt.tight_layout()

            # Save figure
            filename = (
                f"weight_distributions_convergence_panels_{ensemble_loss}_{split_type}".replace(
                    "/", "_"
                ).replace(" ", "_")
                + ".png"
            )
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"  Saved: {filename}")
            plt.close()


# Add this to the main() function after the existing weight extraction:
def update_main_function_with_convergence_plots():
    """
    Code snippet to add to main() function after the existing weight extraction section.
    Insert this after the line: print(f"Extracted {len(weights_data)} weight distributions for plotting")
    """


def main():
    """
    Main function to run the complete analysis including maxent values.
    """
    # Define parameters
    ensembles = ["ISO_TRI", "ISO_BI"]
    loss_functions = ["mcMSE", "MSE"]
    num_splits = 3
    # MAXENT_VALUES=(0.1 0.01 0.001 0.0001 1.0)
    maxent_values = [0.1, 0.01, 0.001, 0.0001, 1.0]
    # 1 2 5 10 50 100 500 1000
    maxent_values = [1, 2, 5, 10, 50, 100, 500, 1000]
    # 1 2 5 10 50 100 500 1000 10000 100000 1000000 10000000 100000000 1000000000
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
    # maxent_values = [
    #     1,
    #     2,
    #     5,
    #     10,
    #     50,
    #     100,
    #     500,
    #     1000,
    #     10000,
    # ]
    convergence_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # Define directories
    results_dir = "../fitting/jaxENT/_optimise_maxent_MAEneps_adam"

    results_dir = os.path.join(os.path.dirname(__file__), results_dir)

    output_dir = "_failure_analysis_maxent_weights_MAEneps_adam"

    output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    # Define trajectory and reference paths
    traj_dir = "../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    bi_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_filtered.xtc"
    tri_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_autovalidation/_TeaA/trajectories/TeaA_initial_sliced.xtc"

    trajectory_paths = {
        "ISO_TRI": tri_path,
        "ISO_BI": bi_path,
    }

    topology_path = os.path.join(traj_dir, "TeaA_ref_closed_state.pdb")
    reference_paths = [
        os.path.join(traj_dir, "TeaA_ref_open_state.pdb"),  # Index 0: Open
        os.path.join(traj_dir, "TeaA_ref_closed_state.pdb"),  # Index 1: Closed
    ]

    # Check if required directories and files exist
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    if not os.path.exists(traj_dir):
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    for path in [topology_path] + list(trajectory_paths.values()) + reference_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")

    print("Starting Complete IsoValidation Analysis with MaxEnt Values...")
    print(f"Results directory: {results_dir}")
    print(f"Trajectory directory: {traj_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"Number of splits: {num_splits}")
    print(f"MaxEnt values: {maxent_values}")
    print("-" * 60)

    # Load all optimization results with maxent
    print("Loading optimization results with maxent values...")
    results = load_all_optimization_results_with_maxent(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        num_splits=num_splits,
        maxent_values=maxent_values,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Part 1: KL Divergence Analysis with MaxEnt
    print("\n" + "=" * 60)
    print("PART 1: KL DIVERGENCE AND ESS ANALYSIS WITH MAXENT")
    print("=" * 60)

    # Extract frame weights and calculate KL divergences
    print("Extracting frame weights and calculating KL divergences and ESS...")
    kl_ess_df = extract_frame_weights_kl_with_maxent(results)

    if len(kl_ess_df) > 0:
        print(
            f"Extracted {len(kl_ess_df)} KL divergence and ESS data points from optimization histories"
        )

        # Save the KL divergence and ESS dataset
        kl_ess_df_path = os.path.join(output_dir, "kl_divergence_ess_analysis_maxent_data.csv")
        kl_ess_df.to_csv(kl_ess_df_path, index=False)
        print(f"KL divergence and ESS dataset saved to: {kl_ess_df_path}")

    # Extract weights for failure analysis plotting
    print("\nExtracting weights for failure analysis...")

    # Convert kl_ess_df to weights_data format for plotting
    weights_data = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is not None and history.states:
                            # Get final (converged) state
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

    print(f"Extracted {len(weights_data)} weight distributions for plotting")
    # Extract weights over convergence steps for additional plots
    print("\nExtracting weights over convergence steps for additional analysis...")
    convergence_weights_data = extract_weights_over_convergence_steps(results)

    # Additional plotting section (add this in the plotting section after existing plots)
    print("\nCreating additional weight distribution plots...")

    # Weight distribution plots with maxent as panels
    if len(convergence_weights_data) > 0:
        plot_weight_distribution_maxent_panels(convergence_weights_data, output_dir)
        plot_weight_distribution_convergence_panels(convergence_weights_data, output_dir)
    else:
        print("No convergence weights data available for additional distribution plots")
    # Conformational Recovery Analysis
    print("\n" + "=" * 60)
    print("CONFORMATIONAL RECOVERY ANALYSIS WITH MAXENT")
    print("=" * 60)

    # Check if trajectory files exist
    missing_files = []
    for name, path in trajectory_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("Warning: The following trajectory files are missing:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("Skipping conformational recovery analysis.")
        recovery_df = pd.DataFrame()
    else:
        print("Analyzing conformational recovery with maxent values...")
        recovery_df = analyze_conformational_recovery_with_maxent(
            trajectory_paths, topology_path, reference_paths, results
        )

        if len(recovery_df) > 0:
            # Save the recovery dataset
            recovery_df_path = os.path.join(output_dir, "conformational_recovery_maxent_data.csv")
            recovery_df.to_csv(recovery_df_path, index=False)
            print(f"Conformational recovery dataset saved to: {recovery_df_path}")

    # Compute additional analysis for plotting
    print("\nComputing additional analysis for failure plots...")

    # Compute pairwise KLD between splits
    kld_df = pd.DataFrame()
    if len(weights_data) > 0:
        kld_df = compute_pairwise_kld_between_splits(weights_data)
        if not kld_df.empty:
            kld_path = os.path.join(output_dir, "kld_between_splits_data.csv")
            kld_df.to_csv(kld_path, index=False)
            print(f"KLD between splits data saved to: {kld_path}")

    # Compute sequential maxent KLD
    sequential_kld_df = pd.DataFrame()
    if len(weights_data) > 0:
        sequential_kld_df = compute_sequential_maxent_kld(weights_data)
        if not sequential_kld_df.empty:
            seq_kld_path = os.path.join(output_dir, "sequential_maxent_kld_data.csv")
            sequential_kld_df.to_csv(seq_kld_path, index=False)
            print(f"Sequential maxent KLD data saved to: {seq_kld_path}")

    # Create failure analysis plots
    print("\n" + "=" * 60)
    print("CREATING FAILURE ANALYSIS PLOTS")
    print("=" * 60)

    # Weight distribution plots
    if len(weights_data) > 0:
        plot_weight_distribution_lines(weights_data, output_dir)
    else:
        print("No weights data available for distribution plots")

    # Recovery scatter plots
    if not recovery_df.empty:
        plot_weight_recovery_scatter(recovery_df, output_dir)
    else:
        print("No recovery data available for scatter plots")

    # KLD plots
    if not kld_df.empty:
        plot_kld_between_splits(kld_df, output_dir)
    else:
        print("No KLD data available for between-splits plots")

    # Sequential KLD plots
    if not sequential_kld_df.empty:
        plot_sequential_maxent_kld(sequential_kld_df, output_dir)
    else:
        print("No sequential KLD data available for sequential plots")

    print("\n" + "=" * 60)
    print("ANALYSIS WITH MAXENT VALUES AND FAILURE PLOTS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
