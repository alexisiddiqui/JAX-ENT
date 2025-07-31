"""

This script is used to analyse the %recovery of the ratio of the two conformations used in the IsoValidation process.

This proceeds by loading the ISO TRI and ISO BI trajectories and the reference open and closed states and clustering by RMSD (1.0 A).
The cluster assignments are then used to calculate the ratios of the two clusters in the ensembles, via their frame weights.
The results are then plotted as a bar chart, showing the open state %recovery of the ratio of the two conformations.
open state %recovery is calculated from the ratio of the open cluster against the ground truth ratio of the two clusters:
    (60:40, Open:Closed).


This script also  analyzes the KL divergence of frame_weights against a uniform prior
from the optimization histories of the ISO TRI and BI models.
It plots how the frame weight distributions deviate from uniform weighting
across different convergence thresholds.
"""

import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis import rms

# Add the base directory to the path to import the HDF5 utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.insert(0, base_dir)

# Import the HDF5 loading functions from the provided script
from jaxent.src.utils.hdf import load_optimization_history_from_file


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

    open_truth = ground_truth_ratios.get("open", 0.6)
    closed_truth = ground_truth_ratios.get("closed", 0.4)

    # Calculate recovery as percentage of truth recovered
    if open_truth > 0:
        recovery["open_recovery"] = min(100.0, (open_observed / open_truth) * 100.0)
    else:
        recovery["open_recovery"] = 0.0

    if closed_truth > 0:
        recovery["closed_recovery"] = min(100.0, (closed_observed / closed_truth) * 100.0)
    else:
        recovery["closed_recovery"] = 0.0

    return recovery


def analyze_conformational_recovery(trajectory_paths, topology_path, reference_paths, results_dict):
    """
    Analyze conformational ratio recovery for trajectories.

    Args:
        trajectory_paths (dict): Dictionary of trajectory paths by ensemble name
        topology_path (str): Path to topology file
        reference_paths (list): Paths to reference structures [open, closed]
        results_dict (dict): Optimization results containing frame weights

    Returns:
        pd.DataFrame: Recovery analysis results
    """
    ground_truth_ratios = {"open": 0.6, "closed": 0.4}
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
                "split": "N/A",
                "convergence_step": "N/A",
                "open_ratio": original_ratios.get("cluster_0", 0.0),
                "closed_ratio": original_ratios.get("cluster_1", 0.0),
                "open_recovery": original_recovery["open_recovery"],
                "closed_recovery": original_recovery["closed_recovery"],
                "total_frames": len(cluster_assignments),
                "clustered_frames": np.sum(cluster_assignments >= 0),
            }
        )

        # Analyze with optimized frame weights
        if ensemble_name in results_dict:
            for loss_name in results_dict[ensemble_name]:
                for split_idx in results_dict[ensemble_name][loss_name]:
                    history = results_dict[ensemble_name][loss_name][split_idx]

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
                                            "split": split_idx,
                                            "convergence_step": step_idx,
                                            "open_ratio": weighted_ratios.get("cluster_0", 0.0),
                                            "closed_ratio": weighted_ratios.get("cluster_1", 0.0),
                                            "open_recovery": weighted_recovery["open_recovery"],
                                            "closed_recovery": weighted_recovery["closed_recovery"],
                                            "total_frames": len(cluster_assignments),
                                            "clustered_frames": np.sum(cluster_assignments >= 0),
                                        }
                                    )

    return pd.DataFrame(recovery_data)


def plot_conformational_recovery(recovery_df, convergence_rates, output_dir):
    """
    Plot conformational ratio recovery analysis.

    Args:
        recovery_df (pd.DataFrame): Recovery analysis results
        output_dir (str): Output directory for plots
    """
    # Convert convergence_step to numeric, coercing errors to NaN
    recovery_df["convergence_step"] = pd.to_numeric(
        recovery_df["convergence_step"], errors="coerce"
    )

    # Set up plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}

    # Plot 1: Open state recovery comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Conformational Ratio Recovery Analysis", fontsize=16, fontweight="bold")

    # Get final convergence data (last step for each combination)
    final_data = (
        recovery_df[recovery_df["loss_function"] != "Original"]
        .groupby(["ensemble", "loss_function", "split"])
        .last()
        .reset_index()
    )

    # Add original data
    original_data = recovery_df[recovery_df["loss_function"] == "Original"].copy()
    combined_data = pd.concat([final_data, original_data], ignore_index=True)

    # Plot open state recovery
    sns.barplot(data=combined_data, x="ensemble", y="open_recovery", hue="loss_function", ax=ax1)
    ax1.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")
    ax1.set_title("Open State Recovery (%)")
    ax1.set_ylabel("Recovery Percentage")
    ax1.set_ylim(0, 110)
    ax1.legend()

    # Plot ratio comparison
    ratio_data = []
    for _, row in combined_data.iterrows():
        ratio_data.append(
            {
                "ensemble": row["ensemble"],
                "loss_function": row["loss_function"],
                "split": row["split"],
                "state": "Open",
                "ratio": row["open_ratio"],
                "ground_truth": 0.6,
            }
        )
        ratio_data.append(
            {
                "ensemble": row["ensemble"],
                "loss_function": row["loss_function"],
                "split": row["split"],
                "state": "Closed",
                "ratio": row["closed_ratio"],
                "ground_truth": 0.4,
            }
        )

    ratio_df = pd.DataFrame(ratio_data)

    # Plot ratios
    x_positions = np.arange(len(combined_data["ensemble"].unique()))
    width = 0.15

    ensembles = sorted(combined_data["ensemble"].unique())
    loss_functions = sorted(combined_data["loss_function"].unique())

    for i, ensemble in enumerate(ensembles):
        for j, loss_func in enumerate(loss_functions):
            subset = combined_data[
                (combined_data["ensemble"] == ensemble)
                & (combined_data["loss_function"] == loss_func)
            ]
            if len(subset) > 0:
                mean_open = subset["open_ratio"].mean()
                mean_closed = subset["closed_ratio"].mean()

                x_pos = x_positions[i] + (j - len(loss_functions) / 2 + 0.5) * width

                ax2.bar(
                    x_pos,
                    mean_open,
                    width,
                    label=f"{loss_func} (Open)" if i == 0 else "",
                    color=ensemble_colors[ensemble],
                    alpha=0.7,
                )
                ax2.bar(
                    x_pos,
                    mean_closed,
                    width,
                    bottom=mean_open,
                    label=f"{loss_func} (Closed)" if i == 0 else "",
                    color=ensemble_colors[ensemble],
                    alpha=0.4,
                )

    # Add ground truth lines
    ax2.axhline(y=0.6, color="red", linestyle="--", alpha=0.7, label="Ground Truth Open (60%)")
    ax2.axhline(y=1.0, color="red", linestyle=":", alpha=0.7, label="Ground Truth Total")

    ax2.set_title("Conformational Ratios")
    ax2.set_ylabel("Ratio")
    ax2.set_xlabel("Ensemble")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(ensembles)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "conformational_recovery_analysis.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Plot 2: Recovery vs convergence threshold
    convergence_data = recovery_df[
        (recovery_df["loss_function"] != "Original")
        & (recovery_df["convergence_step"] > 0)  # Skip pre-optimization step
    ].copy()
    if len(convergence_data) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = convergence_data[
                    (convergence_data["ensemble"] == ensemble)
                    & (convergence_data["loss_function"] == loss_func)
                ]
                if len(subset) > 0:
                    # Group by convergence step and calculate mean
                    step_means = (
                        subset.groupby("convergence_step")
                        .agg({"open_recovery": ["mean", "std"]})
                        .reset_index()
                    )

                    step_means.columns = ["step", "recovery_mean", "recovery_std"]

                    # Map steps to convergence rates (consistent with KL divergence plots)
                    step_means["convergence_rate"] = step_means["step"].apply(
                        lambda x: convergence_rates[int(x) - 1]
                        if int(x) - 1 < len(convergence_rates)
                        else None
                    )

                    # Remove rows where convergence rate mapping failed
                    step_means = step_means.dropna(subset=["convergence_rate"])

                    if len(step_means) > 0:
                        color = ensemble_colors[ensemble]
                        marker = "o" if loss_func == "mcMSE" else "s"
                        label = f"{ensemble} - {loss_func}"

                        ax.errorbar(
                            step_means["convergence_rate"],
                            step_means["recovery_mean"],
                            yerr=step_means["recovery_std"],
                            label=label,
                            marker=marker,
                            color=color,
                            linewidth=2,
                            capsize=3,
                            markersize=6,
                        )

        ax.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")
        ax.set_xscale("log")  # Use log scale like KL divergence plots
        ax.set_xlabel("Convergence Threshold")
        ax.set_ylabel("Open State Recovery (%)")
        ax.set_title("Open State Recovery vs Convergence Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "recovery_vs_convergence.png"), dpi=300, bbox_inches="tight"
        )
        plt.show()


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


def load_all_optimization_results(
    results_dir: str,
    ensembles: List[str] = ["ISO_TRI", "ISO_BI"],
    loss_functions: List[str] = ["mcMSE", "MSE"],
    num_splits: int = 3,
) -> Dict:
    """
    Load all optimization results from HDF5 files.

    Args:
        results_dir: Directory containing the HDF5 result files
        ensembles: List of ensemble names
        loss_functions: List of loss function names
        num_splits: Number of data splits

    Returns:
        Dictionary containing loaded optimization histories organized by ensemble, loss, and split
    """
    results = {}

    for ensemble in ensembles:
        results[ensemble] = {}

        for loss_name in loss_functions:
            results[ensemble][loss_name] = {}

            for split_idx in range(num_splits):
                filename = f"{ensemble}_{loss_name}_split{split_idx:03d}_results.hdf5"
                filepath = os.path.join(results_dir, filename)

                if os.path.exists(filepath):
                    try:
                        history = load_optimization_history_from_file(filepath)
                        results[ensemble][loss_name][split_idx] = history
                        print(f"Loaded: {filename}")
                    except Exception as e:
                        print(f"Failed to load {filename}: {e}")
                        results[ensemble][loss_name][split_idx] = None
                else:
                    print(f"File not found: {filename}")
                    results[ensemble][loss_name][split_idx] = None

    return results


def extract_frame_weights_kl_divergences(results: Dict) -> pd.DataFrame:
    """
    Extract frame weights and calculate KL divergence against uniform prior.

    Args:
        results: Dictionary containing optimization histories

    Returns:
        DataFrame containing KL divergence values for analysis
    """
    data_rows = []

    for ensemble in results:
        for loss_name in results[ensemble]:
            for split_idx in results[ensemble][loss_name]:
                history = results[ensemble][loss_name][split_idx]

                if history is not None and history.states:
                    for step_idx, state in enumerate(history.states):
                        # Check if frame_weights exist in the state's parameters
                        if (
                            hasattr(state.params, "frame_weights")
                            and state.params.frame_weights is not None
                        ):
                            frame_weights = np.array(state.params.frame_weights)

                            # Skip if frame_weights is empty or all zeros
                            if len(frame_weights) == 0 or np.sum(frame_weights) == 0:
                                continue

                            # Create uniform prior with same length
                            uniform_prior = np.ones(len(frame_weights)) / len(frame_weights)

                            # Calculate KL divergence
                            try:
                                kl_div = kl_divergence(frame_weights, uniform_prior)

                                data_rows.append(
                                    {
                                        "ensemble": ensemble,
                                        "loss_function": loss_name,
                                        "split": split_idx,
                                        "step": step_idx,
                                        "convergence_threshold_step": step_idx,
                                        "kl_divergence": float(kl_div),
                                        "num_frames": len(frame_weights),
                                        "step_number": state.step
                                        if hasattr(state, "step")
                                        else step_idx,
                                    }
                                )
                            except Exception as e:
                                print(
                                    f"Failed to calculate KL divergence for {ensemble}_{loss_name}_split{split_idx}, step {step_idx}: {e}"
                                )
                                continue

    return pd.DataFrame(data_rows)


def plot_kl_divergence_convergence(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str
):
    """
    Plot KL divergence vs convergence threshold.
    Ensembles are shown as different colors, loss functions as different markers.

    Args:
        df: DataFrame containing KL divergence data
        convergence_rates: List of convergence rates used in optimization
        output_dir: Directory to save plots
    """
    # Set up the plotting style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define colors and markers (same as main plots)
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}  # Blue and Orange
    loss_markers = {"mcMSE": "o", "MSE": "s"}  # Circle and Square

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create plot for KL divergence vs convergence
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(
        "Frame Weights KL Divergence vs Convergence Threshold", fontsize=16, fontweight="bold"
    )

    ensembles = sorted(df["ensemble"].unique())
    loss_functions = sorted(df["loss_function"].unique())

    for ensemble in ensembles:
        for loss_func in loss_functions:
            # Filter data for this combination and exclude step 0 (pre-optimization)
            subset = df[
                (df["ensemble"] == ensemble)
                & (df["loss_function"] == loss_func)
                & (df["convergence_threshold_step"] > 0)  # Skip pre-optimization step
            ]

            if len(subset) > 0:
                # Calculate mean and std across splits for each convergence step
                stats = (
                    subset.groupby("convergence_threshold_step")
                    .agg({"kl_divergence": ["mean", "std"]})
                    .reset_index()
                )

                # Flatten column names
                stats.columns = ["step", "kl_mean", "kl_std"]

                # Map steps to convergence rates (step 1 -> convergence_rates[0], step 2 -> convergence_rates[1], etc.)
                stats["convergence_rate"] = stats["step"].apply(
                    lambda x: convergence_rates[x - 1] if x - 1 < len(convergence_rates) else None
                )

                # Remove rows where convergence rate mapping failed
                stats = stats.dropna(subset=["convergence_rate"])

                if len(stats) > 0:
                    # Plot KL divergence
                    color = ensemble_colors[ensemble]
                    marker = loss_markers[loss_func]
                    label = f"{ensemble} - {loss_func}"

                    ax.errorbar(
                        stats["convergence_rate"],
                        stats["kl_mean"],
                        yerr=stats["kl_std"],
                        label=label,
                        marker=marker,
                        color=color,
                        linewidth=2,
                        capsize=3,
                        markersize=6,
                    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Convergence Threshold")
    ax.set_ylabel("KL Divergence from Uniform Prior")
    ax.set_title("Frame Weights KL Divergence")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "frame_weights_kl_divergence.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_kl_divergence_distribution(df: pd.DataFrame, output_dir: str):
    """
    Plot distribution of KL divergence values across ensembles and loss functions.

    Args:
        df: DataFrame containing KL divergence data
        output_dir: Directory to save plots
    """
    # Get final convergence step data (last step for each combination)
    final_data = df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle("Final KL Divergence Distribution", fontsize=16, fontweight="bold")

    # KL divergence comparison
    sns.boxplot(data=final_data, x="ensemble", y="kl_divergence", hue="loss_function", ax=ax)
    ax.set_yscale("log")
    ax.set_title("Final Frame Weights KL Divergence")
    ax.set_ylabel("KL Divergence from Uniform Prior (log scale)")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "final_kl_divergence_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main():
    """
    Main function to run the complete analysis including both KL divergence and conformational recovery.
    """
    # Define parameters (should match those used in the optimization script)
    ensembles = ["ISO_TRI", "ISO_BI"]
    loss_functions = ["mcMSE", "MSE"]
    num_splits = 3
    # Remove the '0' convergence rate as it represents pre-optimization values
    convergence_rates = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # Define directories
    results_dir = "../fitting/jaxENT/_optimise"
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)

    output_dir = "_analysis_complete"
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    # Define trajectory and reference paths
    traj_dir = "../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    trajectory_paths = {
        "ISO_TRI": os.path.join(traj_dir, "sliced_trajectories/TeaA_filtered_sliced.xtc"),
        "ISO_BI": os.path.join(traj_dir, "sliced_trajectories/TeaA_initial_sliced.xtc"),
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

    print("Starting Complete IsoValidation Analysis...")
    print(f"Results directory: {results_dir}")
    print(f"Trajectory directory: {traj_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ensembles: {ensembles}")
    print(f"Loss functions: {loss_functions}")
    print(f"Number of splits: {num_splits}")
    print("-" * 60)

    # Load all optimization results
    print("Loading optimization results...")
    results = load_all_optimization_results(
        results_dir=results_dir,
        ensembles=ensembles,
        loss_functions=loss_functions,
        num_splits=num_splits,
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Part 1: KL Divergence Analysis
    print("\n" + "=" * 60)
    print("PART 1: KL DIVERGENCE ANALYSIS")
    print("=" * 60)

    # Extract KL divergences
    print("Extracting frame weights and calculating KL divergences...")
    kl_df = extract_frame_weights_kl_divergences(results)

    if len(kl_df) > 0:
        print(f"Extracted {len(kl_df)} KL divergence data points from optimization histories")

        # Generate KL divergence plots
        print("Generating KL divergence vs convergence plots...")
        plot_kl_divergence_convergence(kl_df, convergence_rates, output_dir)

        print("Generating KL divergence distribution plots...")
        plot_kl_divergence_distribution(kl_df, output_dir)

        # Save the KL divergence dataset
        kl_df_path = os.path.join(output_dir, "kl_divergence_analysis_data.csv")
        kl_df.to_csv(kl_df_path, index=False)
        print(f"KL divergence dataset saved to: {kl_df_path}")
    else:
        print("No frame weights data found! Skipping KL divergence analysis.")

    # Part 2: Conformational Recovery Analysis
    print("\n" + "=" * 60)
    print("PART 2: CONFORMATIONAL RECOVERY ANALYSIS")
    print("=" * 60)

    # Check if trajectory files exist before proceeding
    missing_files = []
    for name, path in trajectory_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("Warning: The following trajectory files are missing:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("Skipping conformational recovery analysis.")
    else:
        print("Analyzing conformational recovery...")
        recovery_df = analyze_conformational_recovery(
            trajectory_paths, topology_path, reference_paths, results
        )

        if len(recovery_df) > 0:
            print(f"Extracted {len(recovery_df)} conformational recovery data points")

            # Generate recovery plots
            print("Generating conformational recovery plots...")
            plot_conformational_recovery(recovery_df, convergence_rates, output_dir)

            # Save the recovery dataset
            recovery_df_path = os.path.join(output_dir, "conformational_recovery_data.csv")
            recovery_df.to_csv(recovery_df_path, index=False)
            print(f"Conformational recovery dataset saved to: {recovery_df_path}")

            # Print summary statistics
            print("\nConformational Recovery Summary:")
            print("-" * 40)
            final_recovery = (
                recovery_df[recovery_df["loss_function"] != "Original"]
                .groupby(["ensemble", "loss_function"])
                .last()
                .reset_index()
            )

            for _, row in final_recovery.iterrows():
                print(
                    f"{row['ensemble']} - {row['loss_function']}: "
                    f"Open Recovery = {row['open_recovery']:.1f}%, "
                    f"Open Ratio = {row['open_ratio']:.3f}"
                )
        else:
            print("No conformational recovery data generated!")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
