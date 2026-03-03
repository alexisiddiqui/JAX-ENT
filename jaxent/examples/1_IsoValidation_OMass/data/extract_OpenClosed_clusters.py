"""
extract_OpenClosed_clusters.py

Extracts clusters representing Open and Closed states for validation comparisons.
Used to define the ground truth states for the validation of the reweighting.
Performs RMSD calculation and clustering, saves results as CSV files.

Requirements:
    - Valid MDAnalysis universe (generated in data/get_HDXer_AutoValidation_data.py)

Usage:
    python jaxent/examples/1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py

Output:
    - Cluster assignment CSVs in data/_clustering_results/
"""

import os

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms


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
        recovery["open_recovery"] = min(200.0, (open_observed / open_truth) * 100.0)
    else:
        recovery["open_recovery"] = 0.0

    if closed_truth > 0:
        recovery["closed_recovery"] = min(200.0, (closed_observed / closed_truth) * 100.0)
    else:
        recovery["closed_recovery"] = 0.0

    return recovery


def perform_clustering_analysis(
    trajectory_paths, topology_path, reference_paths, output_dir, rmsd_threshold=1.0
):
    """
    Perform clustering analysis for all trajectories and save results to CSV files.

    Args:
        trajectory_paths (dict): Dictionary of trajectory paths by ensemble name
        topology_path (str): Path to topology file
        reference_paths (list): Paths to reference structures [open, closed]
        output_dir (str): Directory to save clustering results
        rmsd_threshold (float): RMSD threshold for clustering
    """
    ground_truth_ratios = {"open": 0.4, "closed": 0.6}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    clustering_results = {}

    for ensemble_name, traj_path in trajectory_paths.items():
        print(f"Performing clustering analysis for {ensemble_name}...")

        if not os.path.exists(traj_path):
            print(f"Warning: Trajectory file not found: {traj_path}")
            continue

        # Compute RMSD to references
        print("  Computing RMSD to reference structures...")
        rmsd_values = compute_rmsd_to_references(traj_path, topology_path, reference_paths)

        # Cluster by RMSD
        print(f"  Clustering frames by RMSD (threshold: {rmsd_threshold} Å)...")
        cluster_assignments = cluster_by_rmsd(rmsd_values, rmsd_threshold=rmsd_threshold)

        # Calculate unweighted (original) ratios
        print("  Calculating cluster ratios...")
        original_ratios = calculate_cluster_ratios(cluster_assignments)
        original_recovery = calculate_recovery_percentage(original_ratios, ground_truth_ratios)

        # Store results
        clustering_results[ensemble_name] = {
            "rmsd_values": rmsd_values,
            "cluster_assignments": cluster_assignments,
            "original_ratios": original_ratios,
            "original_recovery": original_recovery,
        }

        # Save RMSD values
        rmsd_df = pd.DataFrame(rmsd_values, columns=["rmsd_open", "rmsd_closed"])
        rmsd_df["frame"] = range(len(rmsd_values))
        rmsd_df["ensemble"] = ensemble_name
        rmsd_path = os.path.join(output_dir, f"rmsd_values_{ensemble_name}.csv")
        rmsd_df.to_csv(rmsd_path, index=False)
        print(f"  Saved RMSD values to: {rmsd_path}")

        # Save cluster assignments
        cluster_df = pd.DataFrame(
            {
                "frame": range(len(cluster_assignments)),
                "ensemble": ensemble_name,
                "cluster_assignment": cluster_assignments,
                "rmsd_open": rmsd_values[:, 0],
                "rmsd_closed": rmsd_values[:, 1],
                "min_rmsd": np.min(rmsd_values, axis=1),
            }
        )
        cluster_path = os.path.join(output_dir, f"cluster_assignments_{ensemble_name}.csv")
        cluster_df.to_csv(cluster_path, index=False)
        print(f"  Saved cluster assignments to: {cluster_path}")

        # Print summary statistics
        n_frames = len(cluster_assignments)
        n_clustered = np.sum(cluster_assignments >= 0)
        n_open = np.sum(cluster_assignments == 0)
        n_closed = np.sum(cluster_assignments == 1)
        n_unclustered = np.sum(cluster_assignments == -1)

        print(f"  Clustering Summary for {ensemble_name}:")
        print(f"    Total frames: {n_frames}")
        print(f"    Clustered frames: {n_clustered} ({n_clustered / n_frames * 100:.1f}%)")
        print(f"    Open-like (cluster 0): {n_open} ({n_open / n_frames * 100:.1f}%)")
        print(f"    Closed-like (cluster 1): {n_closed} ({n_closed / n_frames * 100:.1f}%)")
        print(f"    Unclustered: {n_unclustered} ({n_unclustered / n_frames * 100:.1f}%)")
        print(f"    Original open recovery: {original_recovery['open_recovery']:.1f}%")
        print(f"    Original closed recovery: {original_recovery['closed_recovery']:.1f}%")

    # Save summary statistics
    summary_data = []
    for ensemble_name, results in clustering_results.items():
        cluster_assignments = results["cluster_assignments"]
        original_ratios = results["original_ratios"]
        original_recovery = results["original_recovery"]

        n_frames = len(cluster_assignments)
        n_clustered = np.sum(cluster_assignments >= 0)

        summary_data.append(
            {
                "ensemble": ensemble_name,
                "total_frames": n_frames,
                "clustered_frames": n_clustered,
                "clustering_efficiency": n_clustered / n_frames,
                "open_frames": np.sum(cluster_assignments == 0),
                "closed_frames": np.sum(cluster_assignments == 1),
                "unclustered_frames": np.sum(cluster_assignments == -1),
                "original_open_ratio": original_ratios.get("cluster_0", 0.0),
                "original_closed_ratio": original_ratios.get("cluster_1", 0.0),
                "original_open_recovery": original_recovery["open_recovery"],
                "original_closed_recovery": original_recovery["closed_recovery"],
                "rmsd_threshold": rmsd_threshold,
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "clustering_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved clustering summary to: {summary_path}")

    # Save metadata
    metadata = {
        "rmsd_threshold": rmsd_threshold,
        "reference_structures": reference_paths,
        "topology_path": topology_path,
        "trajectory_paths": trajectory_paths,
        "ground_truth_ratios": ground_truth_ratios,
    }

    metadata_df = pd.DataFrame([metadata])
    metadata_path = os.path.join(output_dir, "clustering_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"Saved clustering metadata to: {metadata_path}")

    return clustering_results


def main():
    """
    Main function to run clustering analysis.
    """
    # Define directories and paths
    traj_dir = "../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    bi_path = "sliced_trajectories/TeaA_filtered_sliced.xtc"
    tri_path = "sliced_trajectories/TeaA_initial_sliced.xtc"

    bi_path = os.path.join(traj_dir, bi_path)
    tri_path = os.path.join(traj_dir, tri_path)

    trajectory_paths = {
        "ISO_TRI": tri_path,
        "ISO_BI": bi_path,
    }

    topology_path = os.path.join(traj_dir, "TeaA_ref_closed_state.pdb")
    reference_paths = [
        os.path.join(traj_dir, "TeaA_ref_open_state.pdb"),  # Index 0: Open
        os.path.join(traj_dir, "TeaA_ref_closed_state.pdb"),  # Index 1: Closed
    ]

    output_dir = os.path.join(os.path.dirname(__file__), "_clustering_results")

    # Check if required directories and files exist
    if not os.path.exists(traj_dir):
        raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

    missing_files = []
    for path in [topology_path] + list(trajectory_paths.values()) + reference_paths:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        print("Warning: The following files are missing:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("Cannot proceed with clustering analysis.")
        return

    print("Starting Clustering Analysis...")
    print(f"Trajectory directory: {traj_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Topology: {topology_path}")
    print(f"References: {reference_paths}")
    print(f"Trajectories: {trajectory_paths}")
    print("-" * 60)

    # Perform clustering analysis
    clustering_results = perform_clustering_analysis(
        trajectory_paths=trajectory_paths,
        topology_path=topology_path,
        reference_paths=reference_paths,
        output_dir=output_dir,
        rmsd_threshold=1.0,
    )

    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
