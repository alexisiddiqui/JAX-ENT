"""

Ensemble comparison comparing graph/network metrics - compare KL divergence of ensemble-average residue-wise network metrics. For global (whole protein) metrics compare the KL divergence of the global metrics distribution across whole structures. The reference for the KL comparison is the original ensemble.
Observe how this varies as the ensemble is "de-noised".  De-noising here means clustering using an RMSD cutoff to the reference structures, decreasing the cutoff reduces the noise in the ensemble. I want to compare this across various shell intervals used in netHDX contact definitions.

The script takes the following arguments:


- reference_structure_1
- reference_structure_2
- trajectory
- topology
- structure_names (compact vs extended)
- network config names (definitions are described inside the script, just use this to select the config)
- cutoffs (list of floats for RMSD Ca cutoff for clustering) if None - no cutoff
- seed (for reproducibility, 42)
- output_dir (default to current script directory)
- reference_ratio (structure_1:structure_2 ratio to set when sampling the ensemble - if None, sample randomly)
- sample_size (number of structures to sample from the ensemble - if None, sample all structures, default to 100)


This script builds the ensemble and plots the distributions of the metrics for each ensemble described by the list of cutoffs. Taking care to remove nan containing residues (for residue metrics) and structures (for global metrics).

The ensemble building starts with clustering structures based on minimum RMSD to one of reference structures. The structures in each cluster are aligned by distance to their respective reference. For each cutoff structures are sampled from each cluster, in accordance with the reference_ratio if present. Plot a PCA on the pairwise CA atoms of the ensemble to show the clustering.

For each ensemble-config-cutoff combination the ensemble-average, residue-wise network metrics are calculated. For the global metrics, the distribution of the metrics is calculated across the whole ensemble.


reference_structure_1 = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
reference_structure_2 = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"

trajectory = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/HOIP_sampled_500.xtc"
topology = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/HOIP_overall_combined_stripped.pdb

Script usage

python network_metrics_denoising.py \
    --reference_structure_1 /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb \
    --reference_structure_2 /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb \
    --trajectory /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/HOIP_sampled_500.xtc \
    --topology /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/HOIP_overall_combined_stripped.pdb \
    --cutoffs 2.0 5.0 None 

python /home/alexi/Documents/JAX-ENT/notebooks/netHDXSensitivity/HOIP/jaxENT_run_netHDX_EnsembleMetric_DeNoising.py \
    --reference_structure_1 /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb \
    --reference_structure_2 /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb \
    --trajectory /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/HOIP_sampled_500.xtc \
    --topology /home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/HOIP_overall_combined_stripped.pdb \
    --cutoffs 10 15 20 25

"""

import argparse
import os
import random
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis import align, rms
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from tqdm import tqdm

from jaxent.src.models.config import BV_model_Config, NetHDXConfig
from jaxent.src.models.func.netHDX import build_hbond_network
from jaxent.src.models.HDX.netHDX.features import NetworkMetrics


# ---- Functions for metric data preparation ----
def parse_cutoff(value):
    if value.lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Cannot convert '{value}' to float or None")


def prepare_metric_data(
    network_metrics: List[NetworkMetrics], residue_ids: List[int]
) -> Dict[str, np.ndarray]:
    """
    Convert list of NetworkMetrics into arrays for plotting.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame
        residue_ids: List of residue IDs

    Returns:
        Dictionary mapping metric names to arrays of shape (n_frames, n_residues)
    """
    n_frames = len(network_metrics)
    n_residues = len(residue_ids)

    # Initialize arrays for each metric
    metric_arrays = {
        "Degree": np.zeros((n_frames, n_residues)),
        "Clustering Coefficient": np.zeros((n_frames, n_residues)),
        "Betweenness": np.zeros((n_frames, n_residues)),
        "K-core Number": np.zeros((n_frames, n_residues)),
        "Min Path Length": np.zeros((n_frames, n_residues)),
        "Mean Path Length": np.zeros((n_frames, n_residues)),
        "Max Path Length": np.zeros((n_frames, n_residues)),
        # New metrics
        "Closeness": np.zeros((n_frames, n_residues)),
        "PageRank": np.zeros((n_frames, n_residues)),
        "Harmonic Centrality": np.zeros((n_frames, n_residues)),
        "HITS Hub": np.zeros((n_frames, n_residues)),
        "HITS Authority": np.zeros((n_frames, n_residues)),
        "Avg Communicability": np.zeros((n_frames, n_residues)),
    }

    # Fill arrays
    for frame_idx, frame_metrics in enumerate(network_metrics):
        for res_idx, res_id in enumerate(residue_ids):
            # Original metrics
            metric_arrays["Degree"][frame_idx, res_idx] = frame_metrics.degrees.get(res_id, np.nan)
            metric_arrays["Clustering Coefficient"][frame_idx, res_idx] = (
                frame_metrics.clustering_coeffs.get(res_id, np.nan)
            )
            metric_arrays["Betweenness"][frame_idx, res_idx] = frame_metrics.betweenness.get(
                res_id, np.nan
            )
            metric_arrays["K-core Number"][frame_idx, res_idx] = frame_metrics.kcore_numbers.get(
                res_id, np.nan
            )

            # Path length metrics - special handling for infinity values
            min_path = frame_metrics.min_path_lengths.get(res_id, float("inf"))
            mean_path = frame_metrics.mean_path_lengths.get(res_id, float("inf"))
            max_path = frame_metrics.max_path_lengths.get(res_id, float("inf"))

            metric_arrays["Min Path Length"][frame_idx, res_idx] = (
                np.nan if min_path == float("inf") else min_path
            )
            metric_arrays["Mean Path Length"][frame_idx, res_idx] = (
                np.nan if mean_path == float("inf") else mean_path
            )
            metric_arrays["Max Path Length"][frame_idx, res_idx] = (
                np.nan if max_path == float("inf") else max_path
            )

            # New metrics
            metric_arrays["Closeness"][frame_idx, res_idx] = frame_metrics.closeness.get(
                res_id, np.nan
            )
            metric_arrays["PageRank"][frame_idx, res_idx] = frame_metrics.pagerank.get(
                res_id, np.nan
            )
            metric_arrays["Harmonic Centrality"][frame_idx, res_idx] = (
                frame_metrics.harmonic_centrality.get(res_id, np.nan)
            )
            metric_arrays["HITS Hub"][frame_idx, res_idx] = frame_metrics.hits_hub.get(
                res_id, np.nan
            )
            metric_arrays["HITS Authority"][frame_idx, res_idx] = frame_metrics.hits_authority.get(
                res_id, np.nan
            )
            metric_arrays["Avg Communicability"][frame_idx, res_idx] = (
                frame_metrics.avg_communicability.get(res_id, np.nan)
            )

    return metric_arrays


def prepare_global_metric_data(network_metrics: List[NetworkMetrics]) -> Dict[str, np.ndarray]:
    """
    Extract global metrics from NetworkMetrics list.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame

    Returns:
        Dictionary mapping global metric names to arrays of shape (n_frames,)
    """
    n_frames = len(network_metrics)

    # Initialize arrays for each global metric
    global_metric_arrays = {
        "Degree Assortativity": np.zeros(n_frames),
        "Local Efficiency": np.zeros(n_frames),
    }

    # Fill arrays
    for frame_idx, frame_metrics in enumerate(network_metrics):
        # Check for valid values
        degree_assortativity = frame_metrics.degree_assortativity
        local_efficiency = frame_metrics.local_efficiency

        # Use NaN for invalid values
        global_metric_arrays["Degree Assortativity"][frame_idx] = (
            np.nan
            if degree_assortativity is None or np.isinf(degree_assortativity)
            else degree_assortativity
        )
        global_metric_arrays["Local Efficiency"][frame_idx] = (
            np.nan if local_efficiency is None or np.isinf(local_efficiency) else local_efficiency
        )

    return global_metric_arrays


# ---- KL Divergence functions ----


def calculate_kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions.

    Args:
        p: First distribution
        q: Second distribution
        epsilon: Small value to avoid division by zero

    Returns:
        KL divergence value
    """
    # Filter out NaN values
    valid_mask = ~np.isnan(p) & ~np.isnan(q)
    p = p[valid_mask]
    q = q[valid_mask]

    # If too few valid points, return NaN
    if len(p) < 2 or len(q) < 2:
        return np.nan

    # Replace zeros with small epsilon
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)

    # Normalize to probability distributions
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    # Check if sums are valid
    if p_sum <= 0 or q_sum <= 0:
        return np.nan

    p = p / p_sum
    q = q / q_sum

    # Calculate KL divergence
    kl_div = np.sum(p * np.log(p / q))

    # Return NaN if result is invalid
    return np.nan if np.isnan(kl_div) or np.isinf(kl_div) else kl_div


def calculate_js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.

    JS divergence is a symmetric measure and is bounded between 0 and 1.

    Args:
        p: First distribution
        q: Second distribution
        epsilon: Small value to avoid division by zero

    Returns:
        JS divergence value
    """
    # Filter out NaN values
    valid_mask = ~np.isnan(p) & ~np.isnan(q)
    p = p[valid_mask]
    q = q[valid_mask]

    # If too few valid points, return NaN
    if len(p) < 2 or len(q) < 2:
        return np.nan

    # Replace zeros with small epsilon
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)

    # Normalize
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    if p_sum <= 0 or q_sum <= 0:
        return np.nan

    p = p / p_sum
    q = q / q_sum

    # Compute the mixture distribution
    m = (p + q) / 2.0

    # Calculate JS divergence as average of KL divergences
    js_div = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

    return np.nan if np.isnan(js_div) or np.isinf(js_div) else js_div


def calculate_residue_wise_kl_divergence(
    reference_metrics: Dict[str, np.ndarray],
    ensemble_metrics: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Calculate KL divergence between reference and ensemble metrics across all residues.

    Args:
        reference_metrics: Dictionary of reference metrics (shape: n_metrics x n_residues)
        ensemble_metrics: Dictionary of ensemble metrics (shape: n_ensembles x n_metrics x n_residues)

    Returns:
        Dictionary mapping metric names to KL divergence values
    """
    kl_divergences = {}

    for metric_name in reference_metrics:
        if metric_name not in ensemble_metrics:
            continue

        ref_metric = reference_metrics[metric_name]
        ens_metric = ensemble_metrics[metric_name]

        # Filter out residues with NaN values in either reference or ensemble
        valid_mask = ~np.isnan(ref_metric) & ~np.isnan(ens_metric)

        if np.sum(valid_mask) < 2:  # Need at least 2 valid values for KL divergence
            kl_divergences[metric_name] = np.nan
            continue

        # Get values for valid residues
        ref_values = ref_metric[valid_mask]
        ens_values = ens_metric[valid_mask]

        # Calculate KL divergence across all valid residues
        kl_divergences[metric_name] = calculate_kl_divergence(ref_values, ens_values)

    return kl_divergences


def calculate_global_metric_kl_divergence(
    reference_metrics: Dict[str, np.ndarray],
    ensemble_metrics: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Calculate KL divergence between reference and ensemble distributions of global metrics.
    Uses KDE to handle variable sized ensembles.

    Args:
        reference_metrics: Dictionary of reference global metrics
        ensemble_metrics: Dictionary of ensemble global metrics

    Returns:
        Dictionary mapping metric names to KL divergence values
    """
    from scipy.stats import gaussian_kde

    kl_divergences = {}

    for metric_name in reference_metrics:
        if metric_name not in ensemble_metrics:
            continue

        ref_distribution = reference_metrics[metric_name]
        ens_distribution = ensemble_metrics[metric_name]

        # Check if we have scalar values (ensemble averages) or distributions (frame-wise)
        if np.isscalar(ref_distribution) or np.isscalar(ens_distribution):
            # For scalar values, we can't compute KL divergence
            kl_divergences[metric_name] = np.nan
            continue

        # Filter out NaN values
        ref_valid = ref_distribution[~np.isnan(ref_distribution)]
        ens_valid = ens_distribution[~np.isnan(ens_distribution)]

        # Check if we have enough valid data points for KDE
        if len(ref_valid) < 3 or len(ens_valid) < 3:
            kl_divergences[metric_name] = np.nan
            continue

        try:
            # Compute KDEs for both distributions
            ref_kde = gaussian_kde(ref_valid)
            ens_kde = gaussian_kde(ens_valid)

            # Create a common grid spanning both distributions with more points in areas of higher density
            grid_min = min(ref_valid.min(), ens_valid.min())
            grid_max = max(ref_valid.max(), ens_valid.max())
            grid_range = grid_max - grid_min

            # Ensure we have a reasonable range to avoid numerical issues
            if grid_range < 1e-6:
                grid_min -= 0.5
                grid_max += 0.5
                grid_range = grid_max - grid_min

            # Create evaluation grid
            grid = np.linspace(grid_min - 0.1 * grid_range, grid_max + 0.1 * grid_range, 1000)

            # Evaluate KDEs on the grid
            ref_pdf = ref_kde(grid)
            ens_pdf = ens_kde(grid)

            # Normalize to ensure they're proper PDFs
            ref_pdf = ref_pdf / np.sum(ref_pdf)
            ens_pdf = ens_pdf / np.sum(ens_pdf)

            # Calculate KL divergence between KDEs
            kl_divergences[metric_name] = calculate_kl_divergence(ref_pdf, ens_pdf)

        except Exception as e:
            print(f"Error calculating KDE KL divergence for {metric_name}: {str(e)}")
            kl_divergences[metric_name] = np.nan

    return kl_divergences


def plot_metric_kl_divergence(
    kl_divergences: Dict[float, Dict[str, float]],
    title: str = "KL Divergence by Metric",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot KL divergence for each metric across different cutoffs.

    Args:
        kl_divergences: Dictionary mapping cutoffs to dictionaries of KL divergences by metric
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get cutoffs in sorted order
    cutoffs = sorted([c for c in kl_divergences.keys() if c is not None])

    # Get all metrics
    all_metrics = set()
    for cutoff in cutoffs:
        all_metrics.update(kl_divergences[cutoff].keys())
    all_metrics = sorted(all_metrics)

    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))

    # Plot KL divergence for each metric
    for i, metric_name in enumerate(all_metrics):
        kl_values = []
        valid_cutoffs = []

        for cutoff in cutoffs:
            if metric_name in kl_divergences[cutoff] and not np.isnan(
                kl_divergences[cutoff][metric_name]
            ):
                kl_values.append(kl_divergences[cutoff][metric_name])
                valid_cutoffs.append(cutoff)

        if valid_cutoffs:
            ax.plot(
                valid_cutoffs, kl_values, "o-", label=metric_name, color=colors[i % len(colors)]
            )

    # Set labels and title
    ax.set_xlabel("RMSD Cutoff")
    ax.set_ylabel("KL Divergence")
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig


# ---- RMSD and clustering functions ----


def calculate_pairwise_rmsd_to_references(
    universe: Universe,
    reference_structures: List[Universe],
    align_selection: str = "name CA",
    rmsd_selection: str = "name CA",
) -> np.ndarray:
    """
    Calculate RMSD of each frame to each reference structure.

    Args:
        universe: MDAnalysis Universe containing trajectory
        reference_structures: List of reference structure Universes
        align_selection: Selection string for alignment
        rmsd_selection: Selection string for RMSD calculation

    Returns:
        Array of shape (n_frames, n_references) containing RMSD values
    """
    n_frames = universe.trajectory.n_frames
    n_refs = len(reference_structures)
    rmsd_values = np.zeros((n_frames, n_refs))

    # Get selections for trajectory
    mobile_align = universe.select_atoms(align_selection)
    mobile_rmsd = universe.select_atoms(rmsd_selection)

    # Get selections for references
    ref_align_sels = [ref.select_atoms(align_selection) for ref in reference_structures]
    ref_rmsd_sels = [ref.select_atoms(rmsd_selection) for ref in reference_structures]

    # Reference coordinates
    ref_coords = [ref_sel.positions for ref_sel in ref_rmsd_sels]

    # Calculate RMSD for each frame against each reference
    for i, ts in enumerate(tqdm(universe.trajectory, desc="Calculating RMSDs", total=n_frames)):
        for j, (ref_align, ref_coord) in enumerate(zip(ref_align_sels, ref_coords)):
            # Align to reference
            align.alignto(mobile_align, ref_align, weights="mass")

            # Calculate RMSD
            rmsd_values[i, j] = rms.rmsd(mobile_rmsd.positions, ref_coord, superposition=False)

    return rmsd_values


def cluster_structures_by_rmsd(rmsd_to_refs: np.ndarray) -> np.ndarray:
    """
    Cluster structures based on which reference they're closest to.

    Args:
        rmsd_to_refs: Array of RMSD values to each reference (n_frames, n_refs)

    Returns:
        Array of cluster assignments (0 = first reference, 1 = second reference, etc.)
    """
    # Check if we have valid data
    if rmsd_to_refs.shape[1] == 0:
        # No references provided
        print("Warning: No reference structures provided for clustering")
        return np.zeros(rmsd_to_refs.shape[0], dtype=int)

    # Check for NaN values
    if np.any(np.isnan(rmsd_to_refs)):
        print("Warning: NaN values found in RMSD data. Replacing with large values.")
        rmsd_to_refs = np.nan_to_num(rmsd_to_refs, nan=9999.0)

    # Assign each frame to the reference with minimum RMSD
    assignments = np.argmin(rmsd_to_refs, axis=1)

    # Log some information about the clustering
    unique_assignments = np.unique(assignments)
    print(f"Found {len(unique_assignments)} unique clusters: {unique_assignments}")
    counts = np.bincount(assignments)
    print(f"Cluster sizes: {counts}")

    return assignments


def filter_structures_by_cutoff(
    rmsd_to_refs: np.ndarray,
    cluster_assignments: np.ndarray,
    cutoff: float,
) -> List[int]:
    """
    Filter structures by applying an RMSD cutoff to each cluster.

    Args:
        rmsd_to_refs: Array of RMSD values to each reference
        cluster_assignments: Array of cluster assignments
        cutoff: RMSD cutoff value

    Returns:
        List of frame indices that pass the cutoff
    """
    n_frames = rmsd_to_refs.shape[0]
    selected_frames = []

    for i in range(n_frames):
        ref_idx = cluster_assignments[i]
        rmsd = rmsd_to_refs[i, ref_idx]

        if rmsd <= cutoff:
            selected_frames.append(i)

    return selected_frames


def plot_rmsd_distributions(
    rmsd_to_refs: np.ndarray,
    structure_names: List[str],
    output_dir: str,
    title: str = "RMSD Distributions to Reference Structures",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot and save the RMSD distributions to each reference structure.

    Args:
        rmsd_to_refs: Array of RMSD values to each reference (n_frames, n_refs)
        structure_names: Names of reference structures
        output_dir: Output directory to save the plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    n_refs = rmsd_to_refs.shape[1]

    # 1. Plot histogram for each reference
    ax = axes[0]
    for i in range(n_refs):
        rmsd_values = rmsd_to_refs[:, i]
        name = structure_names[i] if i < len(structure_names) else f"Reference {i + 1}"

        # Create histogram
        ax.hist(rmsd_values, bins=30, alpha=0.6, label=name)

    # Set labels
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Frequency")
    ax.set_title("RMSD Distribution Histograms")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # 2. Plot density comparison
    ax = axes[1]
    for i in range(n_refs):
        rmsd_values = rmsd_to_refs[:, i]
        name = structure_names[i] if i < len(structure_names) else f"Reference {i + 1}"

        # Calculate kernel density estimation
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(rmsd_values)
        x = np.linspace(np.min(rmsd_values), np.max(rmsd_values), 100)
        ax.plot(x, kde(x), label=name)

    # Set labels
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Density")
    ax.set_title("RMSD Density Distribution")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add stats as text
    stats_text = "RMSD Statistics:\n"
    for i in range(n_refs):
        rmsd_values = rmsd_to_refs[:, i]
        name = structure_names[i] if i < len(structure_names) else f"Reference {i + 1}"
        stats_text += f"{name}:\n"
        stats_text += f"  Min: {np.min(rmsd_values):.2f}Å\n"
        stats_text += f"  Max: {np.max(rmsd_values):.2f}Å\n"
        stats_text += f"  Mean: {np.mean(rmsd_values):.2f}Å\n"
        stats_text += f"  Median: {np.median(rmsd_values):.2f}Å\n\n"

    fig.text(
        0.98,
        0.5,
        stats_text,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
        verticalalignment="center",
        horizontalalignment="right",
    )

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "rmsd_distributions.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")

    print(f"RMSD distribution plot saved to: {plot_path}")
    return fig


def sample_from_clusters(
    cluster_assignments: np.ndarray,
    selected_frames: List[int],
    reference_ratio: Optional[List[float]] = None,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
) -> List[int]:
    """
    Sample frames from clusters according to specified ratio.

    Args:
        cluster_assignments: Array of cluster assignments
        selected_frames: List of frame indices that pass the cutoff
        reference_ratio: Ratio of samples from each cluster [cluster1_ratio, cluster2_ratio, ...]
        sample_size: Number of structures to sample (if None, use all selected frames)
        random_seed: Random seed for reproducibility

    Returns:
        List of sampled frame indices
    """
    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Get cluster assignments for selected frames
    selected_clusters = cluster_assignments[selected_frames]
    unique_clusters = np.unique(selected_clusters)
    n_clusters = len(unique_clusters)

    # If no unique clusters found, return all frames
    if n_clusters == 0:
        print("Warning: No clusters found! Using all selected frames.")
        return selected_frames

    # If no sample size specified, return all selected frames
    if sample_size is None:
        return selected_frames

    # If no ratio specified, sample equally from each cluster
    if reference_ratio is None:
        reference_ratio = [1.0 / n_clusters] * n_clusters

    # Normalize ratio
    reference_ratio = np.array(reference_ratio) / sum(reference_ratio)

    # Calculate number of samples from each cluster
    samples_per_cluster = {}
    remaining = sample_size

    for i, cluster in enumerate(unique_clusters):
        if i < len(reference_ratio):
            n_samples = int(sample_size * reference_ratio[i])
            samples_per_cluster[cluster] = n_samples
            remaining -= n_samples

    # Distribute remaining samples
    for cluster in unique_clusters:
        if remaining > 0:
            samples_per_cluster[cluster] += 1
            remaining -= 1

    # Sample from each cluster
    sampled_frames = []

    for cluster in unique_clusters:
        cluster_frames = [
            selected_frames[i] for i, c in enumerate(selected_clusters) if c == cluster
        ]
        n_samples = min(samples_per_cluster[cluster], len(cluster_frames))

        if n_samples > 0:
            sampled = random.sample(cluster_frames, n_samples)
            sampled_frames.extend(sampled)

    return sampled_frames


def create_ensemble_from_indices(
    universe: Universe,
    frame_indices: List[int],
    output_pdb: Optional[str] = None,
) -> Universe:
    """
    Create a new ensemble from selected frame indices by saving to temporary trajectory
    and reloading with the original topology.

    Args:
        universe: MDAnalysis Universe containing trajectory
        frame_indices: List of frame indices to include in ensemble
        output_pdb: Path to output PDB file (if None, don't write file)

    Returns:
        New Universe containing only the selected frames
    """
    # Sort frame indices to maintain trajectory order
    frame_indices = sorted(frame_indices)

    # If no frames selected, return None
    if not frame_indices:
        print("Warning: No frames selected for ensemble creation")
        return None

    # Get output directory from output_pdb or create a temporary one
    if output_pdb:
        output_dir = os.path.dirname(output_pdb)
        base_name = os.path.splitext(os.path.basename(output_pdb))[0]
    else:
        output_dir = tempfile.mkdtemp(prefix="md_ensemble_")
        base_name = "ensemble"

    os.makedirs(output_dir, exist_ok=True)

    # Save trajectory with selected frames
    trajectory_path = os.path.join(output_dir, f"{base_name}.xtc")

    with mda.Writer(trajectory_path, universe.atoms.n_atoms) as writer:
        for i in frame_indices:
            try:
                universe.trajectory[i]
                writer.write(universe.atoms)
            except Exception as e:
                print(f"Error writing frame {i}: {str(e)}")

    print(f"Saved {len(frame_indices)} frames to: {trajectory_path}")

    # Write first frame to PDB if requested
    if output_pdb:
        universe.trajectory[frame_indices[0]]
        universe.atoms.write(output_pdb)
        print(f"Saved first frame to: {output_pdb}")

    # Create new universe by loading the saved trajectory with original topology
    try:
        # Get the original topology file
        topology = universe.filename
        if topology is None:
            # If topology is not available, use the PDB we just wrote
            if output_pdb:
                topology = output_pdb
            else:
                # Create a temporary PDB file for topology
                temp_pdb = os.path.join(output_dir, f"{base_name}_topology.pdb")
                universe.trajectory[frame_indices[0]]
                universe.atoms.write(temp_pdb)
                topology = temp_pdb

        # Create new universe with the saved trajectory
        new_universe = Universe(topology, trajectory_path)
        print(
            f"Created new universe with {new_universe.trajectory.n_frames} frames from {topology}"
        )

        return new_universe

    except Exception as e:
        print(f"Error creating ensemble universe: {str(e)}")
        import traceback

        traceback.print_exc()

        # Return None if we can't create a new universe
        return None


# ---- PCA and visualization functions ----


def perform_pca_on_ca_distances(
    universe: Universe, n_components: int = 2
) -> Tuple[np.ndarray, float]:
    """
    Perform PCA on pairwise CA atom distances.

    Args:
        universe: MDAnalysis Universe containing trajectory
        n_components: Number of PCA components

    Returns:
        Tuple of (PCA coordinates, explained variance ratio)
    """
    # Select CA atoms
    ca_atoms = universe.select_atoms("name CA")
    n_frames = universe.trajectory.n_frames

    # Calculate pairwise distances for each frame
    n_ca = len(ca_atoms)
    n_pairs = (n_ca * (n_ca - 1)) // 2
    pairwise_distances = np.zeros((n_frames, n_pairs))

    for i, ts in enumerate(
        tqdm(universe.trajectory, desc="Calculating distances for PCA", total=n_frames)
    ):
        distances = pdist(ca_atoms.positions)
        pairwise_distances[i] = distances

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(pairwise_distances)

    return pca_result, pca.explained_variance_ratio_


def plot_pca_with_clusters(
    pca_coords: np.ndarray,
    cluster_assignments: np.ndarray,
    selected_frames: List[int],
    explained_variance: np.ndarray,
    structure_names: List[str],
    title: str = "PCA of CA Distances with Cluster Assignments",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Create a PCA plot colored by cluster assignments.

    Args:
        pca_coords: PCA coordinates
        cluster_assignments: Array of cluster assignments
        selected_frames: List of frame indices that pass the cutoff
        explained_variance: PCA explained variance ratios
        structure_names: Names of reference structures for legend
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all points in gray
    ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1], c="lightgray", alpha=0.3, s=20, label="Filtered out"
    )

    # Plot selected points colored by cluster
    unique_clusters = np.unique(cluster_assignments[selected_frames])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    for i, cluster in enumerate(unique_clusters):
        cluster_indices = [idx for idx in selected_frames if cluster_assignments[idx] == cluster]
        cluster_name = (
            structure_names[cluster] if i < len(structure_names) else f"Cluster {cluster}"
        )

        ax.scatter(
            pca_coords[cluster_indices, 0],
            pca_coords[cluster_indices, 1],
            c=[colors[i]],
            s=50,
            alpha=0.7,
            label=cluster_name,
        )

    # Set labels and title
    ax.set_xlabel(f"PC1 ({explained_variance[0]:.2%})")
    ax.set_ylabel(f"PC2 ({explained_variance[1]:.2%})")
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig


def plot_metric_distributions(
    reference_metrics: Dict[str, np.ndarray],
    ensemble_metrics: Dict[str, np.ndarray],
    metric_name: str,
    cutoff: float,
    title: str = "Metric Distribution Comparison",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot distributions of a metric from reference and ensemble.

    Args:
        reference_metrics: Dictionary of reference metrics
        ensemble_metrics: Dictionary of ensemble metrics
        metric_name: Name of the metric to plot
        cutoff: RMSD cutoff value used for the ensemble
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if metric_name not in reference_metrics or metric_name not in ensemble_metrics:
        ax.text(
            0.5,
            0.5,
            f"Metric '{metric_name}' not found in data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Get data
    ref_data = reference_metrics[metric_name]
    ens_data = ensemble_metrics[metric_name]

    # Remove NaN values
    ref_valid = ~np.isnan(ref_data)
    ens_valid = ~np.isnan(ens_data)

    ref_data = ref_data[ref_valid]
    ens_data = ens_data[ens_valid]

    # Check if we have enough valid data
    if len(ref_data) < 2 or len(ens_data) < 2:
        ax.text(
            0.5,
            0.5,
            f"Not enough valid data for metric '{metric_name}'",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    # Print stats for diagnostics
    print(f"Metric: {metric_name}, Cutoff: {cutoff}")
    print(f"  Reference: {len(ref_data)}/{len(ref_valid)} valid values")
    print(f"  Ensemble: {len(ens_data)}/{len(ens_valid)} valid values")

    # Create histograms
    bins = np.linspace(
        min(np.min(ref_data), np.min(ens_data)), max(np.max(ref_data), np.max(ens_data)), 30
    )

    ax.hist(ref_data, bins=bins, alpha=0.5, label=f"Reference (n={len(ref_data)})")
    ax.hist(
        ens_data, bins=bins, alpha=0.5, label=f"Ensemble (Cutoff = {cutoff}, n={len(ens_data)})"
    )

    # Calculate KL divergence
    kl_div = calculate_kl_divergence(ref_data, ens_data)
    js_div = calculate_js_divergence(ref_data, ens_data)

    # Set labels and title
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title}\nKL Divergence = {kl_div:.4f}, JS Divergence = {js_div:.4f}")

    # Add stats as text
    stats_text = "Stats:\n"
    stats_text += f"Reference: mean={np.mean(ref_data):.4f}, std={np.std(ref_data):.4f}\n"
    stats_text += f"Ensemble: mean={np.mean(ens_data):.4f}, std={np.std(ens_data):.4f}"

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
    )

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig


def plot_residue_wise_kl_divergence(
    kl_divergences: Dict[str, Dict[str, np.ndarray]],
    residue_ids: List[int],
    metric_name: str,
    title: str = "Residue-wise KL Divergence",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot residue-wise KL divergence for different cutoffs.

    Args:
        kl_divergences: Dictionary mapping cutoffs to dictionaries of KL divergences
        residue_ids: List of residue IDs
        metric_name: Name of the metric to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get cutoffs in sorted order
    cutoffs = sorted([c for c in kl_divergences.keys() if c is not None])

    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(cutoffs)))

    # Plot KL divergence for each cutoff
    for i, cutoff in enumerate(cutoffs):
        if metric_name not in kl_divergences[cutoff]:
            continue

        kl_data = kl_divergences[cutoff][metric_name]

        # Plot line
        ax.plot(residue_ids, kl_data, label=f"Cutoff = {cutoff}", color=colors[i], linewidth=2)

    # Set labels and title
    ax.set_xlabel("Residue ID")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"{title}\nMetric: {metric_name}")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig


def plot_kl_divergence_vs_cutoff(
    kl_divergences: Dict[float, Dict[str, float]],
    title: str = "KL Divergence vs. RMSD Cutoff",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot KL divergence vs RMSD cutoff for global metrics.

    Args:
        kl_divergences: Dictionary mapping cutoffs to dictionaries of KL divergences
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get cutoffs in sorted order
    cutoffs = sorted([c for c in kl_divergences.keys() if c is not None])

    # Get all metrics
    all_metrics = set()
    for cutoff in cutoffs:
        all_metrics.update(kl_divergences[cutoff].keys())

    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_metrics)))

    # Plot KL divergence for each metric
    for i, metric_name in enumerate(sorted(all_metrics)):
        kl_values = []
        valid_cutoffs = []

        for cutoff in cutoffs:
            if metric_name in kl_divergences[cutoff] and not np.isnan(
                kl_divergences[cutoff][metric_name]
            ):
                kl_values.append(kl_divergences[cutoff][metric_name])
                valid_cutoffs.append(cutoff)

        if valid_cutoffs:
            ax.plot(
                valid_cutoffs, kl_values, "o-", label=metric_name, color=colors[i % len(colors)]
            )

    # Set labels and title
    ax.set_xlabel("RMSD Cutoff")
    ax.set_ylabel("KL Divergence")
    ax.set_title(title)

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig


def plot_ensemble_average_metrics(
    ensemble_metrics_by_cutoff: Dict[float, Dict[str, np.ndarray]],
    residue_ids: List[int],
    metric_name: str,
    title: str = "Ensemble Average Metrics by Cutoff",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot ensemble average metrics for different cutoffs.

    Args:
        ensemble_metrics_by_cutoff: Dictionary mapping cutoffs to dictionaries of metrics
        residue_ids: List of residue IDs
        metric_name: Name of the metric to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get cutoffs in sorted order
    cutoffs = sorted([c for c in ensemble_metrics_by_cutoff.keys() if c is not None])

    # Create color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(cutoffs)))

    # Plot metric for each cutoff
    for i, cutoff in enumerate(cutoffs):
        if metric_name not in ensemble_metrics_by_cutoff[cutoff]:
            continue

        metric_data = ensemble_metrics_by_cutoff[cutoff][metric_name]

        # Plot line
        ax.plot(residue_ids, metric_data, label=f"Cutoff = {cutoff}", color=colors[i], linewidth=2)

    # Set labels and title
    ax.set_xlabel("Residue ID")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{title}\nMetric: {metric_name}")

    # Add legend
    ax.legend()

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ensemble comparison with graph/network metrics")

    parser.add_argument(
        "--reference_structure_1", required=True, help="Path to first reference structure"
    )
    parser.add_argument(
        "--reference_structure_2", required=True, help="Path to second reference structure"
    )
    parser.add_argument("--trajectory", required=True, help="Path to trajectory file")
    parser.add_argument("--topology", required=True, help="Path to topology file")
    parser.add_argument(
        "--structure_names",
        nargs=2,
        default=["Compact", "Extended"],
        help="Names of reference structures",
    )
    parser.add_argument(
        "--network_configs",
        nargs="+",
        default=["netHDX_excl2", "netHDX_excl1", "netHDX_excl0", "BV_standard"],
        help="Network configuration names",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=parse_cutoff,
        default=[2.0, 3.0, 4.0, 5.0, None],
        help="RMSD cutoffs for clustering",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output_dir", default="./ensemble_analysis_results", help="Output directory"
    )
    parser.add_argument(
        "--reference_ratio",
        nargs="+",
        type=float,
        help="Ratio of structures from each reference cluster",
    )
    parser.add_argument(
        "--sample_size", type=int, default=25, help="Number of structures to sample"
    )
    parser.add_argument(
        "--reference_cutoff",
        type=parse_cutoff,
        default=None,
        help="RMSD cutoff to use as reference for KL divergence calculations (default: None)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    netHDX_2 = NetHDXConfig()

    netHDX_2.residue_ignore = (-2, 2)
    netHDX_1 = NetHDXConfig()
    netHDX_1.residue_ignore = (-1, 1)
    netHDX_0 = NetHDXConfig()
    netHDX_0.residue_ignore = (-0, 0)
    # Set up network configurations
    network_configs = {
        # "netHDX_standard": NetHDXConfig(),
        "BV_standard": BV_model_Config(),
        # Add custom configurations here, for example:
        "netHDX_excl2": netHDX_2,
        "netHDX_excl1": netHDX_1,
        "netHDX_excl0": netHDX_0,
    }

    # Load reference structures
    print("Loading reference structures...")
    reference_structures = [
        Universe(args.reference_structure_1),
        Universe(args.reference_structure_2),
    ]

    # Load trajectory
    print("Loading trajectory...")
    universe = Universe(args.topology, args.trajectory)

    # Calculate RMSD to reference structures
    print("Calculating RMSD to reference structures...")
    rmsd_to_refs = calculate_pairwise_rmsd_to_references(universe, reference_structures)

    # Plot RMSD distributions before clustering
    print("Plotting RMSD distributions to reference structures...")
    rmsd_plot = plot_rmsd_distributions(
        rmsd_to_refs,
        args.structure_names,
        args.output_dir,
        title="RMSD Distributions to Reference Structures",
    )
    plt.close(rmsd_plot)

    # Cluster structures based on minimum RMSD to reference
    print("Clustering structures by minimum RMSD...")
    cluster_assignments = cluster_structures_by_rmsd(rmsd_to_refs)

    # Count structures in each cluster
    unique_clusters, cluster_counts = np.unique(cluster_assignments, return_counts=True)
    print(f"Cluster distribution: {dict(zip(unique_clusters, cluster_counts))}")

    # Check if clustering was successful
    if len(unique_clusters) == 0:
        print(
            "ERROR: Clustering failed to identify any clusters. Check your reference structures and RMSD calculations."
        )
        return

    # Create ensembles for different cutoffs
    ensembles = {}

    # For storing PCA results
    pca_results = None
    explained_variance = None

    print("Creating ensembles for different RMSD cutoffs...")
    for cutoff in args.cutoffs:
        if cutoff is None:
            print("Creating ensemble with no cutoff (all structures)")
            selected_frames = list(range(universe.trajectory.n_frames))
        else:
            print(f"Creating ensemble with RMSD cutoff = {cutoff}")
            selected_frames = filter_structures_by_cutoff(rmsd_to_refs, cluster_assignments, cutoff)

        print(f"Selected {len(selected_frames)} frames")

        # Sample from clusters
        if args.reference_ratio:
            sampled_frames = sample_from_clusters(
                cluster_assignments,
                selected_frames,
                reference_ratio=args.reference_ratio,
                sample_size=args.sample_size,
                random_seed=args.seed,
            )
        else:
            sampled_frames = sample_from_clusters(
                cluster_assignments,
                selected_frames,
                sample_size=args.sample_size,
                random_seed=args.seed,
            )

        print(f"Sampled {len(sampled_frames)} frames from clusters")

        # Create ensemble universe
        ensemble_dir = os.path.join(args.output_dir, f"cutoff_{cutoff}")
        os.makedirs(ensemble_dir, exist_ok=True)

        ensemble_pdb = os.path.join(ensemble_dir, "ensemble.pdb")
        ensemble_universe = create_ensemble_from_indices(
            universe, sampled_frames, output_pdb=ensemble_pdb
        )

        if ensemble_universe is not None:
            ensembles[cutoff] = {
                "universe": ensemble_universe,
                "frames": sampled_frames,
                "output_dir": ensemble_dir,
            }

    # Perform PCA on CA distances (do once for all frames)
    if pca_results is None:
        print("Performing PCA on CA distances...")
        pca_results, explained_variance = perform_pca_on_ca_distances(universe)

    # Plot PCA with clusters for each cutoff
    print("Creating PCA visualizations...")
    for cutoff, ensemble_data in ensembles.items():
        pca_dir = os.path.join(ensemble_data["output_dir"], "pca")
        os.makedirs(pca_dir, exist_ok=True)

        pca_plot = plot_pca_with_clusters(
            pca_results,
            cluster_assignments,
            ensemble_data["frames"],
            explained_variance,
            args.structure_names,
            title=f"PCA of CA Distances with Cluster Assignments (Cutoff = {cutoff})",
        )

        pca_file = os.path.join(pca_dir, "pca_clusters.png")
        pca_plot.savefig(pca_file, dpi=300, bbox_inches="tight")
        plt.close(pca_plot)

    # Process each network configuration
    for config_name in args.network_configs:
        if config_name not in network_configs:
            print(f"Warning: Configuration '{config_name}' not found, skipping...")
            continue

        config = network_configs[config_name]
        print(f"\nProcessing network configuration: {config_name}")

        config_dir = os.path.join(args.output_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Dictionary to store metrics for each cutoff
        ensemble_metrics_by_cutoff = {}
        residue_metrics_by_cutoff = {}
        global_metrics_by_cutoff = {}

        # Calculate network metrics for each ensemble
        for cutoff, ensemble_data in ensembles.items():
            print(f"Calculating network metrics for cutoff = {cutoff}...")

            ensemble_universe = ensemble_data["universe"]
            ensemble_output_dir = os.path.join(config_dir, f"cutoff_{cutoff}")
            os.makedirs(ensemble_output_dir, exist_ok=True)

            # Build network and compute metrics
            features = build_hbond_network([ensemble_universe], config)
            residue_ids = features.residue_ids
            network_metrics = features.network_metrics

            # Calculate residue-wise metrics
            metric_arrays = prepare_metric_data(network_metrics, residue_ids)

            # Calculate global metrics
            global_metric_arrays = prepare_global_metric_data(network_metrics)

            # Calculate ensemble average for residue-wise metrics
            ensemble_avg_metrics = {}
            for metric_name, metric_data in metric_arrays.items():
                # Average across frames (axis 0), handle empty arrays
                if metric_data.size > 0:
                    ensemble_avg_metrics[metric_name] = np.nanmean(metric_data, axis=0)
                else:
                    ensemble_avg_metrics[metric_name] = np.array([])

            # Store metrics
            ensemble_metrics_by_cutoff[cutoff] = ensemble_avg_metrics
            residue_metrics_by_cutoff[cutoff] = metric_arrays
            global_metrics_by_cutoff[cutoff] = global_metric_arrays

        # Now determine reference cutoff after we've processed all regular ensembles
        reference_cutoff = args.reference_cutoff

        # If no reference cutoff provided, use entire ensemble (None)
        if reference_cutoff is None:
            reference_cutoff = None
            print(
                "No reference cutoff specified. Using entire ensemble (cutoff = None) as reference"
            )

        print(f"Using cutoff = {reference_cutoff} as reference for KL divergence calculations")

        # Process the reference ensemble if it wasn't in the original cutoffs
        if reference_cutoff not in ensemble_metrics_by_cutoff:
            print(f"Processing reference ensemble with cutoff = {reference_cutoff}...")

            # Create the reference ensemble if it doesn't exist
            if reference_cutoff not in ensembles:
                print(f"Creating reference ensemble with cutoff = {reference_cutoff}")

                if reference_cutoff is None:
                    selected_frames = list(range(universe.trajectory.n_frames))
                else:
                    selected_frames = filter_structures_by_cutoff(
                        rmsd_to_refs, cluster_assignments, reference_cutoff
                    )

                print(f"Selected {len(selected_frames)} frames for reference ensemble")

                # Sample from clusters using the same logic as other ensembles
                sampled_frames = sample_from_clusters(
                    cluster_assignments,
                    selected_frames,
                    reference_ratio=args.reference_ratio,
                    sample_size=args.sample_size,
                    random_seed=args.seed,
                )

                print(f"Sampled {len(sampled_frames)} frames for reference ensemble")

                # Create ensemble universe
                reference_dir = os.path.join(config_dir, f"cutoff_{reference_cutoff}")
                os.makedirs(reference_dir, exist_ok=True)

                reference_pdb = os.path.join(reference_dir, "ensemble.pdb")
                reference_universe = create_ensemble_from_indices(
                    universe, sampled_frames, output_pdb=reference_pdb
                )

                if reference_universe is None:
                    print(
                        f"Error: Failed to create reference ensemble with cutoff = {reference_cutoff}"
                    )
                    continue

                ensembles[reference_cutoff] = {
                    "universe": reference_universe,
                    "frames": sampled_frames,
                    "output_dir": reference_dir,
                }

            # Calculate metrics for reference ensemble
            reference_ensemble = ensembles[reference_cutoff]
            reference_universe = reference_ensemble["universe"]

            # Build network and compute metrics for reference
            features = build_hbond_network([reference_universe], config)
            residue_ids = features.residue_ids
            network_metrics = features.network_metrics

            # Calculate metrics for reference ensemble
            metric_arrays = prepare_metric_data(network_metrics, residue_ids)
            global_metric_arrays = prepare_global_metric_data(network_metrics)

            # Calculate ensemble averages for reference
            ensemble_avg_metrics = {}
            for metric_name, metric_data in metric_arrays.items():
                if metric_data.size > 0:
                    ensemble_avg_metrics[metric_name] = np.nanmean(metric_data, axis=0)
                else:
                    ensemble_avg_metrics[metric_name] = np.array([])

            # Store reference metrics
            ensemble_metrics_by_cutoff[reference_cutoff] = ensemble_avg_metrics
            residue_metrics_by_cutoff[reference_cutoff] = metric_arrays
            global_metrics_by_cutoff[reference_cutoff] = global_metric_arrays

        print(f"Using cutoff = {reference_cutoff} as reference for KL divergence calculations")

        # Check if we have valid reference metrics
        if reference_cutoff not in ensemble_metrics_by_cutoff:
            print(f"Error: Reference cutoff {reference_cutoff} metrics not available")
            continue

        reference_metrics = ensemble_metrics_by_cutoff[reference_cutoff]

        # Calculate KL divergence between reference and each ensemble
        residue_kl_divergences = {}
        global_kl_divergences = {}

        for cutoff in ensemble_metrics_by_cutoff:
            if cutoff == reference_cutoff:
                continue

            # Residue-wise KL divergence
            residue_kl_divergences[cutoff] = calculate_residue_wise_kl_divergence(
                reference_metrics, ensemble_metrics_by_cutoff[cutoff]
            )

            # Global metrics KL divergence
            global_kl_divergences[cutoff] = calculate_global_metric_kl_divergence(
                global_metrics_by_cutoff[reference_cutoff], global_metrics_by_cutoff[cutoff]
            )

        # Plot results
        plot_dir = os.path.join(config_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # Get all metrics from reference
        all_metrics = list(reference_metrics.keys())

        # 1. Plot KL divergence vs cutoff for global metrics
        kl_vs_cutoff_plot = plot_kl_divergence_vs_cutoff(
            global_kl_divergences,
            title=f"Global Metrics KL Divergence vs. RMSD Cutoff ({config_name})",
        )
        kl_vs_cutoff_file = os.path.join(plot_dir, "global_kl_vs_cutoff.png")
        kl_vs_cutoff_plot.savefig(kl_vs_cutoff_file, dpi=300, bbox_inches="tight")
        plt.close(kl_vs_cutoff_plot)

        # 2. Plot metric KL divergence across all residues
        metric_kl_plot = plot_metric_kl_divergence(
            residue_kl_divergences,
            title=f"KL Divergence by Metric Across All Residues ({config_name})",
        )
        metric_kl_file = os.path.join(plot_dir, "metric_kl_divergence.png")
        metric_kl_plot.savefig(metric_kl_file, dpi=300, bbox_inches="tight")
        plt.close(metric_kl_plot)

        # 3. Plot ensemble average metrics for each cutoff
        for metric_name in all_metrics:
            ensemble_avg_plot = plot_ensemble_average_metrics(
                ensemble_metrics_by_cutoff,
                residue_ids,
                metric_name,
                title=f"Ensemble Average Metrics ({config_name})",
            )
            metric_file = os.path.join(
                plot_dir, f"ensemble_avg_{metric_name.replace(' ', '_')}.png"
            )
            ensemble_avg_plot.savefig(metric_file, dpi=300, bbox_inches="tight")
            plt.close(ensemble_avg_plot)

        # 4. Plot metric distributions for global metrics
        for metric_name in global_metrics_by_cutoff[reference_cutoff]:
            for cutoff in global_metrics_by_cutoff:
                if cutoff == reference_cutoff:
                    continue

                distribution_plot = plot_metric_distributions(
                    global_metrics_by_cutoff[reference_cutoff],
                    global_metrics_by_cutoff[cutoff],
                    metric_name,
                    cutoff,
                    title=f"Global Metric Distribution ({config_name})",
                )
                metric_file = os.path.join(
                    plot_dir, f"global_distribution_{metric_name.replace(' ', '_')}_{cutoff}.png"
                )
                distribution_plot.savefig(metric_file, dpi=300, bbox_inches="tight")
                plt.close(distribution_plot)

        # Save metrics data for later analysis
        metrics_data = {
            "config_name": config_name,
            "residue_ids": residue_ids,
            "ensemble_metrics_by_cutoff": ensemble_metrics_by_cutoff,
            "global_metrics_by_cutoff": global_metrics_by_cutoff,
            "residue_kl_divergences": residue_kl_divergences,
            "global_kl_divergences": global_kl_divergences,
        }

        np.save(os.path.join(config_dir, "metrics_data.npy"), metrics_data)

    print("\nAnalysis completed successfully!")


def create_summary_report(
    output_dir: str,
    network_configs: List[str],
    cutoffs: List[float],
) -> None:
    """
    Create a summary report comparing results across configurations.
    Generates publication-quality visualizations comparing metrics across different cutoffs and configurations.

    Args:
        output_dir: Output directory
        network_configs: List of network configuration names
        cutoffs: List of RMSD cutoffs
    """
    # Set up figure styles for publication quality
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.dpi": 150,
        }
    )

    # Create summary directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Load metrics data for each configuration
    config_data = {}

    for config_name in network_configs:
        config_dir = os.path.join(output_dir, config_name)
        metrics_file = os.path.join(config_dir, "metrics_data.npy")

        if os.path.exists(metrics_file):
            try:
                metrics_data = np.load(metrics_file, allow_pickle=True).item()
                config_data[config_name] = metrics_data
            except Exception as e:
                print(f"Error loading metrics data for {config_name}: {str(e)}")

    if not config_data:
        print("No metrics data found, cannot create summary report")
        return

    # Get valid cutoffs and sort them
    valid_cutoffs = sorted([c for c in cutoffs if c is not None])

    # Set up a distinct color palette for configurations and cutoffs
    config_colors = plt.cm.tab10(np.linspace(0, 1, len(network_configs)))
    cutoff_colors = plt.cm.viridis(np.linspace(0, 1, len(valid_cutoffs)))
    config_color_dict = {config: config_colors[i] for i, config in enumerate(network_configs)}
    cutoff_color_dict = {cutoff: cutoff_colors[i] for i, cutoff in enumerate(valid_cutoffs)}

    # 1. Create line plots for each metric (one metric per subplot)
    # Find all global metrics across all configurations
    all_global_metrics = set()
    for config_name, metrics_data in config_data.items():
        for cutoff in metrics_data["global_kl_divergences"]:
            all_global_metrics.update(metrics_data["global_kl_divergences"][cutoff].keys())

    # Create a multi-panel figure for global metrics
    n_metrics = len(all_global_metrics)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), dpi=150, constrained_layout=True
    )
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, metric_name in enumerate(sorted(all_global_metrics)):
        if i < len(axes):
            ax = axes[i]

            for config_name, metrics_data in config_data.items():
                kl_values = []
                available_cutoffs = []

                for cutoff in valid_cutoffs:
                    if (
                        cutoff in metrics_data["global_kl_divergences"]
                        and metric_name in metrics_data["global_kl_divergences"][cutoff]
                        and not np.isnan(metrics_data["global_kl_divergences"][cutoff][metric_name])
                    ):
                        kl_values.append(metrics_data["global_kl_divergences"][cutoff][metric_name])
                        available_cutoffs.append(cutoff)

                if available_cutoffs:
                    ax.plot(
                        available_cutoffs,
                        kl_values,
                        "o-",
                        label=config_name,
                        color=config_color_dict[config_name],
                        linewidth=2,
                        markersize=6,
                    )

            ax.set_xlabel("RMSD Cutoff (Å)")
            ax.set_ylabel("KL Divergence")
            ax.set_title(metric_name)
            ax.grid(True, linestyle="--", alpha=0.7)

            # Add a light background for better visibility
            ax.set_facecolor("#f9f9f9")

            # Customize ticks
            ax.tick_params(direction="out", length=6, width=1.5)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(5, len(network_configs)),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.suptitle(
        "Global Metrics KL Divergence by Configuration and Cutoff", fontweight="bold", y=1.02
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    plt.savefig(
        os.path.join(summary_dir, "global_metrics_line_comparison.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.savefig(
        os.path.join(summary_dir, "global_metrics_line_comparison.pdf"),
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)

    # 2. Create bar charts with config on x-axis and cutoffs as hues
    for metric_name in sorted(all_global_metrics):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Collect data for plotting
        x_positions = np.arange(len(network_configs))
        width = 0.8 / len(valid_cutoffs)  # Bar width

        for i, cutoff in enumerate(valid_cutoffs):
            kl_values = []

            for config_name in network_configs:
                if (
                    config_name in config_data
                    and cutoff in config_data[config_name]["global_kl_divergences"]
                    and metric_name in config_data[config_name]["global_kl_divergences"][cutoff]
                ):
                    kl_value = config_data[config_name]["global_kl_divergences"][cutoff][
                        metric_name
                    ]
                    if np.isnan(kl_value):
                        kl_values.append(0)  # Replace NaN with 0 for plotting
                    else:
                        kl_values.append(kl_value)
                else:
                    kl_values.append(0)

            # Position bars side by side
            pos = x_positions + (i - len(valid_cutoffs) / 2 + 0.5) * width
            ax.bar(
                pos,
                kl_values,
                width=width,
                label=f"Cutoff {cutoff}Å",
                color=cutoff_color_dict[cutoff],
                alpha=0.8,
                edgecolor="black",
                linewidth=1,
            )

        # Set x-axis labels and ticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels(network_configs, rotation=45, ha="right")
        ax.set_xlabel("Network Configuration")
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"KL Divergence for {metric_name} by Configuration and Cutoff")

        # Add grid lines for y-axis only
        ax.yaxis.grid(True, linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        # Add legend with a transparent background
        ax.legend(
            title="RMSD Cutoff",
            loc="upper left",
            bbox_to_anchor=(1, 1),
            framealpha=0.9,
            fancybox=True,
            shadow=True,
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(summary_dir, f"bar_chart_{metric_name.replace(' ', '_')}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(summary_dir, f"bar_chart_{metric_name.replace(' ', '_')}.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

    # 3. Create a heatmap summary of global metrics across configurations and cutoffs
    for metric_name in sorted(all_global_metrics):
        # Create a matrix for the heatmap
        heatmap_data = np.zeros((len(network_configs), len(valid_cutoffs)))
        heatmap_data[:] = np.nan  # Fill with NaN initially

        for i, config_name in enumerate(network_configs):
            for j, cutoff in enumerate(valid_cutoffs):
                if (
                    config_name in config_data
                    and cutoff in config_data[config_name]["global_kl_divergences"]
                    and metric_name in config_data[config_name]["global_kl_divergences"][cutoff]
                ):
                    heatmap_data[i, j] = config_data[config_name]["global_kl_divergences"][cutoff][
                        metric_name
                    ]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        import seaborn as sns

        mask = np.isnan(heatmap_data)
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            ax=ax,
            mask=mask,
            cbar_kws={"label": "KL Divergence"},
        )

        # Set labels
        ax.set_xlabel("RMSD Cutoff (Å)")
        ax.set_ylabel("Network Configuration")
        ax.set_title(f"KL Divergence Heatmap for {metric_name}")

        # Set tick labels
        ax.set_xticks(np.arange(len(valid_cutoffs)) + 0.5)
        ax.set_yticks(np.arange(len(network_configs)) + 0.5)
        ax.set_xticklabels([f"{c}Å" for c in valid_cutoffs])
        ax.set_yticklabels(network_configs)

        plt.tight_layout()
        plt.savefig(
            os.path.join(summary_dir, f"heatmap_{metric_name.replace(' ', '_')}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(summary_dir, f"heatmap_{metric_name.replace(' ', '_')}.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)

    # 4. Create a summary of best cutoff by metric for each configuration
    summary_df = {"Metric": [], "Configuration": [], "Best Cutoff": [], "KL Divergence": []}

    for metric_name in sorted(all_global_metrics):
        for config_name in network_configs:
            if config_name in config_data:
                metric_values = {}
                for cutoff in valid_cutoffs:
                    if (
                        cutoff in config_data[config_name]["global_kl_divergences"]
                        and metric_name in config_data[config_name]["global_kl_divergences"][cutoff]
                    ):
                        kl_value = config_data[config_name]["global_kl_divergences"][cutoff][
                            metric_name
                        ]
                        if not np.isnan(kl_value):
                            metric_values[cutoff] = kl_value

                if metric_values:
                    # Find cutoff with minimum KL divergence
                    best_cutoff = min(metric_values, key=metric_values.get)

                    summary_df["Metric"].append(metric_name)
                    summary_df["Configuration"].append(config_name)
                    summary_df["Best Cutoff"].append(best_cutoff)
                    summary_df["KL Divergence"].append(metric_values[best_cutoff])

    # Create a table visualization of the summary
    if summary_df["Metric"]:
        fig, ax = plt.subplots(figsize=(12, len(summary_df["Metric"]) * 0.4 + 2), dpi=150)
        ax.axis("tight")
        ax.axis("off")

        table_data = list(
            zip(
                summary_df["Metric"],
                summary_df["Configuration"],
                summary_df["Best Cutoff"],
                [f"{v:.4f}" for v in summary_df["KL Divergence"]],
            )
        )

        colLabels = ["Metric", "Configuration", "Best Cutoff", "KL Divergence"]

        table = ax.table(
            cellText=table_data,
            colLabels=colLabels,
            loc="center",
            cellLoc="center",
            colColours=["#f0f0f0"] * len(colLabels),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#4472C4")
            elif key[0] % 2 == 0:  # Alternating row colors
                cell.set_facecolor("#E6F1FF")

        plt.title(
            "Summary of Best RMSD Cutoffs by Metric and Configuration", pad=20, fontweight="bold"
        )
        plt.savefig(
            os.path.join(summary_dir, "best_cutoff_summary_table.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.5,
        )
        plt.savefig(
            os.path.join(summary_dir, "best_cutoff_summary_table.pdf"),
            bbox_inches="tight",
            pad_inches=0.5,
        )
        plt.close(fig)

    print(f"Enhanced summary report created in: {summary_dir}")
    # =================== RESIDUE METRICS SECTION ===================
    # Create a new directory for residue metrics
    residue_dir = os.path.join(summary_dir, "residue_metrics")
    os.makedirs(residue_dir, exist_ok=True)

    # Find all residue metrics across all configurations
    all_residue_metrics = set()
    for config_name, metrics_data in config_data.items():
        for cutoff in metrics_data["residue_kl_divergences"]:
            all_residue_metrics.update(metrics_data["residue_kl_divergences"][cutoff].keys())

    # Get residue IDs from data
    residue_ids = None
    for config_name, metrics_data in config_data.items():
        if "residue_ids" in metrics_data:
            residue_ids = metrics_data["residue_ids"]
            break

    if residue_ids is None:
        print("Warning: No residue IDs found in the data, using placeholder IDs")
        residue_ids = np.arange(1, 100)  # Placeholder IDs

    # 1. Create multi-panel line plots for residue metrics: KL divergence vs cutoff
    n_metrics = len(all_residue_metrics)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), dpi=150, constrained_layout=True
    )
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, metric_name in enumerate(sorted(all_residue_metrics)):
        if i < len(axes):
            ax = axes[i]

            for config_name, metrics_data in config_data.items():
                # Average across all residues for a single value per cutoff
                kl_values = []
                available_cutoffs = []

                for cutoff in valid_cutoffs:
                    if (
                        cutoff in metrics_data["residue_kl_divergences"]
                        and metric_name in metrics_data["residue_kl_divergences"][cutoff]
                    ):
                        kl_value = metrics_data["residue_kl_divergences"][cutoff][metric_name]
                        if not np.isnan(kl_value):
                            kl_values.append(kl_value)
                            available_cutoffs.append(cutoff)

                if available_cutoffs:
                    ax.plot(
                        available_cutoffs,
                        kl_values,
                        "o-",
                        label=config_name,
                        color=config_color_dict[config_name],
                        linewidth=2,
                        markersize=6,
                    )

            ax.set_xlabel("RMSD Cutoff (Å)")
            ax.set_ylabel("KL Divergence")
            ax.set_title(metric_name)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_facecolor("#f9f9f9")
            ax.tick_params(direction="out", length=6, width=1.5)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=min(5, len(network_configs)),
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.suptitle(
        "Residue Metrics KL Divergence by Configuration and Cutoff", fontweight="bold", y=1.02
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])

    plt.savefig(
        os.path.join(residue_dir, "residue_metrics_line_comparison.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.savefig(
        os.path.join(residue_dir, "residue_metrics_line_comparison.pdf"),
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)

    # 2. Create per-residue plots for important metrics
    important_metrics = sorted(list(all_residue_metrics))[: min(4, len(all_residue_metrics))]

    for metric_name in important_metrics:
        # For each important metric, create per-residue plots comparing configurations
        for cutoff in valid_cutoffs:
            fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

            for config_name, metrics_data in config_data.items():
                if (
                    "ensemble_metrics_by_cutoff" in metrics_data
                    and cutoff in metrics_data["ensemble_metrics_by_cutoff"]
                    and metric_name in metrics_data["ensemble_metrics_by_cutoff"][cutoff]
                ):
                    residue_data = metrics_data["ensemble_metrics_by_cutoff"][cutoff][metric_name]

                    # Plot per-residue data
                    ax.plot(
                        residue_ids[: len(residue_data)],
                        residue_data,
                        label=config_name,
                        color=config_color_dict[config_name],
                        linewidth=2,
                    )

            ax.set_xlabel("Residue ID")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Per-Residue {metric_name} (Cutoff = {cutoff}Å)")
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.set_facecolor("#f9f9f9")

            # Add legend
            ax.legend(
                title="Configuration",
                loc="upper right",
                framealpha=0.9,
                fancybox=True,
                shadow=True,
            )

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    residue_dir, f"per_residue_{metric_name.replace(' ', '_')}_{cutoff}.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(
                    residue_dir, f"per_residue_{metric_name.replace(' ', '_')}_{cutoff}.pdf"
                ),
                bbox_inches="tight",
            )
            plt.close(fig)

        # 3. Create residue heatmaps showing metric values across cutoffs
        for config_name in network_configs:
            if config_name not in config_data:
                continue

            # Create subdirectory for each configuration
            config_residue_dir = os.path.join(residue_dir, config_name)
            os.makedirs(config_residue_dir, exist_ok=True)

            for metric_name in important_metrics:
                if "ensemble_metrics_by_cutoff" not in config_data[config_name]:
                    continue

                # Determine number of residues
                n_residues = 0
                for cutoff in valid_cutoffs:
                    if (
                        cutoff in config_data[config_name]["ensemble_metrics_by_cutoff"]
                        and metric_name
                        in config_data[config_name]["ensemble_metrics_by_cutoff"][cutoff]
                    ):
                        n_residues = max(
                            n_residues,
                            len(
                                config_data[config_name]["ensemble_metrics_by_cutoff"][cutoff][
                                    metric_name
                                ]
                            ),
                        )

                if n_residues == 0:
                    continue

                # Create heatmap matrix
                heatmap_data = np.zeros((n_residues, len(valid_cutoffs)))
                heatmap_data[:] = np.nan  # Fill with NaN initially

                for j, cutoff in enumerate(valid_cutoffs):
                    if (
                        cutoff in config_data[config_name]["ensemble_metrics_by_cutoff"]
                        and metric_name
                        in config_data[config_name]["ensemble_metrics_by_cutoff"][cutoff]
                    ):
                        metric_data = config_data[config_name]["ensemble_metrics_by_cutoff"][
                            cutoff
                        ][metric_name]
                        heatmap_data[: len(metric_data), j] = metric_data

                # Create heatmap
                plt.figure(figsize=(10, max(8, n_residues * 0.2)), dpi=150)

                import seaborn as sns

                mask = np.isnan(heatmap_data)
                ax = sns.heatmap(
                    heatmap_data,
                    cmap="viridis",
                    mask=mask,
                    cbar_kws={"label": metric_name},
                    yticklabels=residue_ids[:n_residues],
                    xticklabels=[f"{c}Å" for c in valid_cutoffs],
                )

                plt.xlabel("RMSD Cutoff (Å)")
                plt.ylabel("Residue ID")
                plt.title(f"{metric_name} Heatmap by Residue and Cutoff ({config_name})")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        config_residue_dir, f"residue_heatmap_{metric_name.replace(' ', '_')}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    os.path.join(
                        config_residue_dir, f"residue_heatmap_{metric_name.replace(' ', '_')}.pdf"
                    ),
                    bbox_inches="tight",
                )
                plt.close()

        # Add residue metrics to the summary table
        for metric_name in sorted(all_residue_metrics):
            for config_name in network_configs:
                if config_name in config_data:
                    metric_values = {}
                    for cutoff in valid_cutoffs:
                        if (
                            cutoff in config_data[config_name]["residue_kl_divergences"]
                            and metric_name
                            in config_data[config_name]["residue_kl_divergences"][cutoff]
                        ):
                            kl_value = config_data[config_name]["residue_kl_divergences"][cutoff][
                                metric_name
                            ]
                            if not np.isnan(kl_value):
                                metric_values[cutoff] = kl_value

                    if metric_values:
                        # Find cutoff with minimum KL divergence
                        best_cutoff = min(metric_values, key=metric_values.get)
                        summary_df["Metric"].append(f"Residue {metric_name}")
                        summary_df["Configuration"].append(config_name)
                        summary_df["Best Cutoff"].append(best_cutoff)
                        summary_df["KL Divergence"].append(metric_values[best_cutoff])


if __name__ == "__main__":
    try:
        main()

        # Parse args again to create summary
        parser = argparse.ArgumentParser(description="Create summary report")
        parser.add_argument(
            "--output_dir", default="./ensemble_analysis_results", help="Output directory"
        )
        parser.add_argument(
            "--network_configs",
            nargs="+",
            default=["netHDX_excl2", "netHDX_excl1", "netHDX_excl0", "BV_standard"],
            help="Network configuration names",
        )
        parser.add_argument(
            "--cutoffs",
            nargs="+",
            type=parse_cutoff,
            default=[2.0, 3.0, 4.0, 5.0, None],
            help="RMSD cutoffs for clustering",
        )

        args, _ = parser.parse_known_args()

        create_summary_report(args.output_dir, args.network_configs, args.cutoffs)

    except Exception as e:
        import traceback

        print(f"Error during execution: {str(e)}")
        traceback.print_exc()
