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
                "split_type": "N/A",
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
        for split_type in results_dict:
            if ensemble_name in results_dict[split_type]:
                for loss_name in results_dict[split_type][ensemble_name]:
                    for split_idx, history in results_dict[split_type][ensemble_name][
                        loss_name
                    ].items():
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
                                        print(
                                            f"  Processing optimized frame weights for {ensemble_name}/{loss_name}/{split_idx} at step {step_idx}"
                                        )
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
                                                "convergence_step": step_idx,
                                                "open_ratio": weighted_ratios.get("cluster_0", 0.0),
                                                "closed_ratio": weighted_ratios.get(
                                                    "cluster_1", 0.0
                                                ),
                                                "open_recovery": weighted_recovery["open_recovery"],
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


def plot_conformational_recovery(recovery_df, convergence_rates, output_dir):
    """
    Plot conformational ratio recovery analysis for each split type.

    Args:
        recovery_df (pd.DataFrame): Recovery analysis results
        output_dir (str): Output directory for plots
    """
    # Set up plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}

    # Convert convergence_step to numeric, coercing errors to NaN
    recovery_df["convergence_step"] = pd.to_numeric(
        recovery_df["convergence_step"], errors="coerce"
    )

    split_types = recovery_df[recovery_df["split_type"] != "N/A"]["split_type"].unique()

    for split_type in split_types:
        print(f"  Plotting recovery for split type: {split_type}")
        split_df = recovery_df[
            (recovery_df["split_type"] == split_type) | (recovery_df["loss_function"] == "Original")
        ]
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        # Plot 1: Open state recovery comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle(
            f"Conformational Ratio Recovery Analysis ({split_type} splits)",
            fontsize=16,
            fontweight="bold",
        )

        # Get final convergence data for this split type
        final_data = (
            split_df[split_df["loss_function"] != "Original"]
            .groupby(["ensemble", "loss_function", "split"])
            .last()
            .reset_index()
        )

        # Add original data
        original_data = recovery_df[recovery_df["loss_function"] == "Original"].copy()
        combined_data = pd.concat([final_data, original_data], ignore_index=True)

        # Plot open state recovery
        sns.barplot(
            data=combined_data, x="ensemble", y="open_recovery", hue="loss_function", ax=ax1
        )
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
                    "state": "Open",
                    "ratio": row["open_ratio"],
                }
            )
            ratio_data.append(
                {
                    "ensemble": row["ensemble"],
                    "loss_function": row["loss_function"],
                    "state": "Closed",
                    "ratio": row["closed_ratio"],
                }
            )
        ratio_df = pd.DataFrame(ratio_data)

        # Manually create stacked and dodged bar plot for conformational ratios
        agg_df = (
            ratio_df.groupby(["ensemble", "loss_function", "state"])["ratio"]
            .mean()
            .unstack()
            .reset_index()
        )

        ensembles = agg_df["ensemble"].unique()
        loss_functions = sorted(agg_df["loss_function"].unique())
        n_ensembles = len(ensembles)
        n_loss = len(loss_functions)

        x = np.arange(n_ensembles)  # the label locations
        total_width = 0.8
        bar_width = total_width / n_loss  # the width of the bars

        colors = sns.color_palette("deep", n_loss)

        for i, loss_func in enumerate(loss_functions):
            loss_df = agg_df[agg_df["loss_function"] == loss_func]
            if loss_df.empty:
                continue

            # Calculate bar positions for dodging
            bar_pos = x - (total_width / 2) + (i * bar_width) + (bar_width / 2)

            # Plot the 'Open' and 'Closed' bars
            ax2.bar(bar_pos, loss_df["Open"], bar_width, label=loss_func, color=colors[i])
            ax2.bar(
                bar_pos,
                loss_df["Closed"],
                bar_width,
                bottom=loss_df["Open"],
                color=colors[i],
                alpha=0.7,
            )

        ax2.set_xticks(x)
        ax2.set_xticklabels(ensembles)

        ax2.axhline(y=0.6, color="red", linestyle="--", alpha=0.7, label="Ground Truth Open (60%)")
        ax2.set_title("Conformational Ratios")
        ax2.set_ylabel("Ratio")
        ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.savefig(
            os.path.join(split_output_dir, "conformational_recovery_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Plot 2: Recovery vs convergence threshold
        convergence_data = split_df[
            (split_df["loss_function"] != "Original")
            & (split_df["convergence_step"] > 0)  # Skip pre-optimization step
        ].copy()
        if len(convergence_data) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ensembles = sorted(convergence_data["ensemble"].unique())
            loss_functions = sorted(convergence_data["loss_function"].unique())

            for ensemble in ensembles:
                for loss_func in loss_functions:
                    subset = convergence_data[
                        (convergence_data["ensemble"] == ensemble)
                        & (convergence_data["loss_function"] == loss_func)
                    ]
                    if len(subset) > 0:
                        step_means = (
                            subset.groupby("convergence_step")
                            .agg({"open_recovery": ["mean", "std"]})
                            .reset_index()
                        )
                        step_means.columns = ["step", "recovery_mean", "recovery_std"]
                        step_means["convergence_rate"] = step_means["step"].apply(
                            lambda x: convergence_rates[int(x) - 1]
                            if int(x) - 1 < len(convergence_rates)
                            else None
                        )
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
            ax.set_xscale("log")
            ax.set_xlabel("Convergence Threshold")
            ax.set_ylabel("Open State Recovery (%)")
            ax.set_title(f"Open State Recovery vs Convergence ({split_type} splits)")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                os.path.join(split_output_dir, "recovery_vs_convergence.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)


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
    Load all optimization results from HDF5 files, accounting for split types.

    Args:
        results_dir: Directory containing subdirectories for each split type.
        ensembles: List of ensemble names.
        loss_functions: List of loss function names.
        num_splits: Number of data splits per type.

    Returns:
        Dictionary with results organized by split_type, ensemble, loss, and split.
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

                for split_idx in range(num_splits):
                    filename = (
                        f"{ensemble}_{loss_name}_{split_type}_split{split_idx:03d}_results.hdf5"
                    )
                    filepath = os.path.join(split_type_dir, filename)

                    if os.path.exists(filepath):
                        try:
                            history = load_optimization_history_from_file(filepath)
                            results[split_type][ensemble][loss_name][split_idx] = history
                            print(f"Loaded: {filepath}")
                        except Exception as e:
                            print(f"Failed to load {filepath}: {e}")
                            results[split_type][ensemble][loss_name][split_idx] = None
                    else:
                        print(f"File not found: {filepath}")
                        results[split_type][ensemble][loss_name][split_idx] = None
    return results


def extract_frame_weights_kl_divergences(results: Dict) -> pd.DataFrame:
    """
    Extract frame weights and calculate KL divergence against uniform prior.

    Args:
        results: Dictionary containing optimization histories.

    Returns:
        DataFrame containing KL divergence values for analysis.
    """
    data_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for split_idx, history in results[split_type][ensemble][loss_name].items():
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
                                    data_rows.append(
                                        {
                                            "split_type": split_type,
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
                                        f"Failed to calculate KL for {split_type}/{ensemble}_{loss_name}_split{split_idx}, step {step_idx}: {e}"
                                    )
                                    continue
    return pd.DataFrame(data_rows)


def plot_kl_divergence_convergence(
    df: pd.DataFrame, convergence_rates: List[float], output_dir: str
):
    """
    Plot KL divergence vs convergence threshold for each split type.

    Args:
        df: DataFrame containing KL divergence data.
        convergence_rates: List of convergence rates used in optimization.
        output_dir: Directory to save plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}
    loss_markers = {"mcMSE": "o", "MSE": "s"}

    split_types = df["split_type"].unique()

    for split_type in split_types:
        print(f"  Plotting KL convergence for split type: {split_type}")
        split_df = df[df["split_type"] == split_type]
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(
            f"Frame Weights KL Divergence vs Convergence ({split_type} splits)",
            fontsize=16,
            fontweight="bold",
        )

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = split_df[
                    (split_df["ensemble"] == ensemble)
                    & (split_df["loss_function"] == loss_func)
                    & (split_df["convergence_threshold_step"] > 0)
                ]
                if len(subset) > 0:
                    stats = (
                        subset.groupby("convergence_threshold_step")
                        .agg({"kl_divergence": ["mean", "std"]})
                        .reset_index()
                    )
                    stats.columns = ["step", "kl_mean", "kl_std"]
                    stats["convergence_rate"] = stats["step"].apply(
                        lambda x: convergence_rates[x - 1]
                        if x - 1 < len(convergence_rates)
                        else None
                    )
                    stats = stats.dropna(subset=["convergence_rate"])

                    if len(stats) > 0:
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
            os.path.join(split_output_dir, "frame_weights_kl_divergence.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_kl_divergence_distribution(df: pd.DataFrame, output_dir: str):
    """
    Plot distribution of final KL divergence values for each split type.

    Args:
        df: DataFrame containing KL divergence data.
        output_dir: Directory to save plots.
    """
    split_types = df["split_type"].unique()

    for split_type in split_types:
        print(f"  Plotting KL distribution for split type: {split_type}")
        split_df = df[df["split_type"] == split_type]
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        final_data = split_df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle(
            f"Final KL Divergence Distribution ({split_type} splits)",
            fontsize=16,
            fontweight="bold",
        )

        sns.boxplot(data=final_data, x="ensemble", y="kl_divergence", hue="loss_function", ax=ax)
        ax.set_yscale("log")
        ax.set_title("Final Frame Weights KL Divergence")
        ax.set_ylabel("KL Divergence from Uniform Prior (log scale)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(split_output_dir, "final_kl_divergence_distribution.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_recovery_vs_kl(df: pd.DataFrame, output_dir: str):
    """
    Plot open state recovery vs KL divergence.

    Args:
        df (pd.DataFrame): Merged dataframe with recovery and KL divergence data.
        output_dir (str): Output directory for plots.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}
    loss_markers = {"mcMSE": "o", "MSE": "s"}

    split_types = df["split_type"].unique()

    for split_type in split_types:
        print(f"  Plotting recovery vs KL for split type: {split_type}")
        split_df = df[df["split_type"] == split_type]
        split_output_dir = os.path.join(output_dir, split_type)
        os.makedirs(split_output_dir, exist_ok=True)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(
            f"Open State Recovery vs. KL Divergence ({split_type} splits)",
            fontsize=16,
            fontweight="bold",
        )

        ensembles = sorted(split_df["ensemble"].unique())
        loss_functions = sorted(split_df["loss_function"].unique())

        ax.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")

        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = split_df[
                    (split_df["ensemble"] == ensemble) & (split_df["loss_function"] == loss_func)
                ]
                if not subset.empty:
                    color = ensemble_colors.get(ensemble, "#333333")
                    marker = loss_markers.get(loss_func, "x")
                    label = f"{ensemble} - {loss_func}"

                    ax.scatter(
                        subset["kl_divergence"],
                        subset["open_recovery"],
                        label=label,
                        marker=marker,
                        color=color,
                        alpha=0.7,
                    )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

        ax.set_xscale("log")
        ax.set_xlabel("KL Divergence from Uniform Prior (log scale)")
        ax.set_ylabel("Open State Recovery (%)")
        ax.set_title("Recovery vs. KL Divergence")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(split_output_dir, "recovery_vs_kl_divergence.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_combined_analysis_faceted(recovery_df, kl_df, merged_df, convergence_rates, output_dir):
    """
    Create a combined plot with all analysis faceted across split_types.
    Uses only the first panel (open state recovery) from conformational analysis.

    Args:
        recovery_df (pd.DataFrame): Recovery analysis results
        kl_df (pd.DataFrame): KL divergence analysis results
        merged_df (pd.DataFrame): Merged recovery and KL data
        convergence_rates (List[float]): List of convergence rates
        output_dir (str): Output directory for plots
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    ensemble_colors = {"ISO_TRI": "#1f77b4", "ISO_BI": "#ff7f0e"}
    loss_markers = {"mcMSE": "o", "MSE": "s"}

    # Get all split types (excluding 'N/A' for original data)
    split_types = recovery_df[recovery_df["split_type"] != "N/A"]["split_type"].unique()
    n_splits = len(split_types)

    if n_splits == 0:
        print("No split types found for combined plot")
        return

    # Create figure with subplots: 4 plots x n_split_types
    fig, axes = plt.subplots(4, n_splits, figsize=(6 * n_splits, 20))

    # Ensure axes is 2D even for single split type
    if n_splits == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(
        "Combined IsoValidation Analysis Across Split Types", fontsize=20, fontweight="bold", y=0.98
    )

    for col_idx, split_type in enumerate(split_types):
        print(f"  Creating combined plots for split type: {split_type}")

        # Row 1: Open State Recovery Bar Chart (first panel from conformational analysis)
        ax1 = axes[0, col_idx]

        # Get data for this split type
        split_recovery_df = recovery_df[
            (recovery_df["split_type"] == split_type) | (recovery_df["loss_function"] == "Original")
        ]

        # Get final convergence data
        final_data = (
            split_recovery_df[split_recovery_df["loss_function"] != "Original"]
            .groupby(["ensemble", "loss_function", "split"])
            .last()
            .reset_index()
        )

        # Add original data
        original_data = recovery_df[recovery_df["loss_function"] == "Original"].copy()
        combined_data = pd.concat([final_data, original_data], ignore_index=True)

        # Plot open state recovery
        sns.barplot(
            data=combined_data, x="ensemble", y="open_recovery", hue="loss_function", ax=ax1
        )
        ax1.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")
        ax1.set_title(f"Open State Recovery - {split_type}")
        ax1.set_ylabel("Recovery Percentage")
        ax1.set_ylim(0, 110)
        if col_idx == 0:  # Only show legend on first column
            ax1.legend()
        else:
            ax1.get_legend().remove()

        # Row 2: KL Divergence vs Convergence
        ax2 = axes[1, col_idx]

        split_kl_df = kl_df[kl_df["split_type"] == split_type]

        ensembles = sorted(split_kl_df["ensemble"].unique())
        loss_functions = sorted(split_kl_df["loss_function"].unique())

        for ensemble in ensembles:
            for loss_func in loss_functions:
                subset = split_kl_df[
                    (split_kl_df["ensemble"] == ensemble)
                    & (split_kl_df["loss_function"] == loss_func)
                    & (split_kl_df["convergence_threshold_step"] > 0)
                ]
                if len(subset) > 0:
                    stats = (
                        subset.groupby("convergence_threshold_step")
                        .agg({"kl_divergence": ["mean", "std"]})
                        .reset_index()
                    )
                    stats.columns = ["step", "kl_mean", "kl_std"]
                    stats["convergence_rate"] = stats["step"].apply(
                        lambda x: convergence_rates[x - 1]
                        if x - 1 < len(convergence_rates)
                        else None
                    )
                    stats = stats.dropna(subset=["convergence_rate"])

                    if len(stats) > 0:
                        color = ensemble_colors[ensemble]
                        marker = loss_markers[loss_func]
                        label = f"{ensemble} - {loss_func}"
                        ax2.errorbar(
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

        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Convergence Threshold")
        ax2.set_ylabel("KL Divergence from Uniform Prior")
        ax2.set_title(f"KL Divergence vs Convergence - {split_type}")
        if col_idx == 0:  # Only show legend on first column
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Row 3: KL Divergence Distribution
        ax3 = axes[2, col_idx]

        final_kl_data = (
            split_kl_df.groupby(["ensemble", "loss_function", "split"]).last().reset_index()
        )

        if not final_kl_data.empty:
            sns.boxplot(
                data=final_kl_data, x="ensemble", y="kl_divergence", hue="loss_function", ax=ax3
            )
            ax3.set_yscale("log")
            ax3.set_title(f"Final KL Divergence Distribution - {split_type}")
            ax3.set_ylabel("KL Divergence (log scale)")
            if col_idx == 0:  # Only show legend on first column
                pass  # Keep legend for boxplot
            else:
                ax3.get_legend().remove()
        else:
            ax3.text(
                0.5, 0.5, "No data available", ha="center", va="center", transform=ax3.transAxes
            )
            ax3.set_title(f"Final KL Divergence Distribution - {split_type}")

        # Row 4: Recovery vs KL Divergence
        ax4 = axes[3, col_idx]

        split_merged_df = (
            merged_df[merged_df["split_type"] == split_type]
            if not merged_df.empty
            else pd.DataFrame()
        )

        if not split_merged_df.empty:
            ax4.axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Perfect Recovery")

            ensembles = sorted(split_merged_df["ensemble"].unique())
            loss_functions = sorted(split_merged_df["loss_function"].unique())

            for ensemble in ensembles:
                for loss_func in loss_functions:
                    subset = split_merged_df[
                        (split_merged_df["ensemble"] == ensemble)
                        & (split_merged_df["loss_function"] == loss_func)
                    ]
                    if not subset.empty:
                        color = ensemble_colors.get(ensemble, "#333333")
                        marker = loss_markers.get(loss_func, "x")
                        label = f"{ensemble} - {loss_func}"

                        ax4.scatter(
                            subset["kl_divergence"],
                            subset["open_recovery"],
                            label=label,
                            marker=marker,
                            color=color,
                            alpha=0.7,
                        )

            ax4.set_xscale("log")
            ax4.set_xlabel("KL Divergence from Uniform Prior (log scale)")
            ax4.set_ylabel("Open State Recovery (%)")
            ax4.set_title(f"Recovery vs KL Divergence - {split_type}")
            if col_idx == 0:  # Only show legend on first column
                ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "No merged data available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title(f"Recovery vs KL Divergence - {split_type}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle

    # Save combined plot
    combined_plot_path = os.path.join(output_dir, "combined_analysis_faceted.png")
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Combined faceted plot saved to: {combined_plot_path}")


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
    results_dir = "../fitting/jaxENT/_optimise_cKL"
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)

    output_dir = "_analysis_complete_cKL"
    output_dir = os.path.join(os.path.dirname(__file__), output_dir)

    # Define trajectory and reference paths
    traj_dir = "../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    trajectory_paths = {
        "ISO_TRI": os.path.join(traj_dir, "sliced_trajectories/TeaA_filtered_sliced.xtc"),
        # "ISO_BI": os.path.join(traj_dir, "sliced_trajectories/TeaA_initial_sliced.xtc"),
        "ISO_BI": "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc",
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
            raise FileNotFoundError(f"Trajectory file not found: {path}")

    if missing_files:
        print("Warning: The following trajectory files are missing:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("Skipping conformational recovery analysis.")
        recovery_df = pd.DataFrame()
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
                .groupby(["split_type", "ensemble", "loss_function"])
                .last()
                .reset_index()
            )

            for _, row in final_recovery.iterrows():
                print(
                    f"{row['split_type']}/{row['ensemble']} - {row['loss_function']}: "
                    f"Open Recovery = {row['open_recovery']:.1f}%, "
                    f"Open Ratio = {row['open_ratio']:.3f}"
                )
        else:
            print("No conformational recovery data generated!")

    # Part 3: KL Divergence vs Recovery Analysis
    print("\n" + "=" * 60)
    print("PART 3: KL DIVERGENCE VS RECOVERY ANALYSIS")
    print("=" * 60)

    if kl_df.empty or recovery_df.empty:
        print("KL divergence or recovery data is missing. Skipping combined analysis.")
        merged_df = pd.DataFrame()
    else:
        print("Merging KL divergence and recovery data for combined analysis...")
        # Rename column for merging
        kl_df_renamed = kl_df.rename(columns={"convergence_threshold_step": "convergence_step"})

        # Ensure correct types for merging
        recovery_df["convergence_step"] = pd.to_numeric(
            recovery_df["convergence_step"], errors="coerce"
        )
        recovery_df["split"] = pd.to_numeric(recovery_df["split"], errors="coerce")
        kl_df_renamed["convergence_step"] = pd.to_numeric(
            kl_df_renamed["convergence_step"], errors="coerce"
        )
        kl_df_renamed["split"] = pd.to_numeric(kl_df_renamed["split"], errors="coerce")

        # Define merge keys
        merge_cols = ["split_type", "ensemble", "loss_function", "split", "convergence_step"]

        # Drop rows with NaN in merge keys to ensure a clean merge
        recovery_df.dropna(subset=merge_cols, inplace=True)
        kl_df_renamed.dropna(subset=merge_cols, inplace=True)

        # Perform the merge
        merged_df = pd.merge(recovery_df, kl_df_renamed, on=merge_cols, how="inner")

        if not merged_df.empty:
            print(f"Successfully merged data, resulting in {len(merged_df)} data points.")
            print("Generating KL divergence vs recovery plots...")
            plot_recovery_vs_kl(merged_df, output_dir)

            # Save merged data
            merged_df_path = os.path.join(output_dir, "kl_vs_recovery_analysis_data.csv")
            merged_df.to_csv(merged_df_path, index=False)
            print(f"KL vs recovery dataset saved to: {merged_df_path}")
        else:
            print(
                "Merged dataframe is empty. No common data points found between KL and recovery analysis. Skipping plot."
            )

    # Part 4: Combined Faceted Analysis
    print("\n" + "=" * 60)
    print("PART 4: COMBINED FACETED ANALYSIS")
    print("=" * 60)

    if not recovery_df.empty and not kl_df.empty:
        print("Generating combined faceted plot across all split types...")
        plot_combined_analysis_faceted(recovery_df, kl_df, merged_df, convergence_rates, output_dir)
    else:
        print("Missing required data for combined plot. Skipping combined analysis.")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
