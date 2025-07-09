import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis import Universe
from MDAnalysis.analysis import rms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
from tqdm import tqdm

from jaxent.src.models.config import BV_model_Config, NetHDXConfig
from jaxent.src.models.func.netHDX import build_hbond_network
from jaxent.src.models.HDX.netHDX.features import NetworkMetrics

# ---- Functions from the original script ----


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
            metric_arrays["Degree"][frame_idx, res_idx] = frame_metrics.degrees.get(res_id, 0)
            metric_arrays["Clustering Coefficient"][frame_idx, res_idx] = (
                frame_metrics.clustering_coeffs.get(res_id, 0)
            )
            metric_arrays["Betweenness"][frame_idx, res_idx] = frame_metrics.betweenness.get(
                res_id, 0
            )
            metric_arrays["K-core Number"][frame_idx, res_idx] = frame_metrics.kcore_numbers.get(
                res_id, 0
            )
            metric_arrays["Min Path Length"][frame_idx, res_idx] = (
                frame_metrics.min_path_lengths.get(res_id, float("inf"))
            )
            metric_arrays["Mean Path Length"][frame_idx, res_idx] = (
                frame_metrics.mean_path_lengths.get(res_id, float("inf"))
            )
            metric_arrays["Max Path Length"][frame_idx, res_idx] = (
                frame_metrics.max_path_lengths.get(res_id, float("inf"))
            )

            # New metrics
            metric_arrays["Closeness"][frame_idx, res_idx] = frame_metrics.closeness.get(res_id, 0)
            metric_arrays["PageRank"][frame_idx, res_idx] = frame_metrics.pagerank.get(res_id, 0)
            metric_arrays["Harmonic Centrality"][frame_idx, res_idx] = (
                frame_metrics.harmonic_centrality.get(res_id, 0)
            )
            metric_arrays["HITS Hub"][frame_idx, res_idx] = frame_metrics.hits_hub.get(res_id, 0)
            metric_arrays["HITS Authority"][frame_idx, res_idx] = frame_metrics.hits_authority.get(
                res_id, 0
            )
            metric_arrays["Avg Communicability"][frame_idx, res_idx] = (
                frame_metrics.avg_communicability.get(res_id, 0)
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
        global_metric_arrays["Degree Assortativity"][frame_idx] = frame_metrics.degree_assortativity
        global_metric_arrays["Local Efficiency"][frame_idx] = frame_metrics.local_efficiency

    return global_metric_arrays


# ---- Distance calculation functions ----


def calculate_rmsd(u: Universe, ref_frame: int = 0) -> np.ndarray:
    """
    Calculate RMSD for all frames in a trajectory against a reference frame.

    Args:
        u: MDAnalysis Universe object
        ref_frame: Reference frame index

    Returns:
        Array of RMSD values for each frame compared to the reference
    """
    # Select CA atoms
    ca_atoms = u.select_atoms("name CA")
    n_frames = u.trajectory.n_frames

    # Set reference frame
    u.trajectory[ref_frame]
    ref_coordinates = ca_atoms.positions.copy()

    # Calculate RMSD for each frame
    rmsd_values = np.zeros(n_frames)
    for i, _ in tqdm(enumerate(u.trajectory), total=n_frames, desc="Calculating RMSD"):
        rmsd_values[i] = rms.rmsd(ca_atoms.positions, ref_coordinates, superposition=True)

    return rmsd_values


def calculate_pairwise_rmsd(u: Universe) -> np.ndarray:
    """
    Calculate pairwise RMSD between all frames in a trajectory.

    Args:
        u: MDAnalysis Universe object

    Returns:
        Square matrix of pairwise RMSD values
    """
    # Select CA atoms
    ca_atoms = u.select_atoms("name CA")
    n_frames = u.trajectory.n_frames

    # Initialize matrix for pairwise RMSD
    pairwise_rmsd = np.zeros((n_frames, n_frames))

    # Store coordinates for each frame
    coordinates = []
    for ts in tqdm(u.trajectory, total=n_frames, desc="Loading coordinates"):
        coordinates.append(ca_atoms.positions.copy())

    # Calculate pairwise RMSD
    total_pairs = n_frames * (n_frames - 1) // 2
    with tqdm(total=total_pairs, desc="Calculating pairwise RMSD") as pbar:
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                rmsd_val = rms.rmsd(coordinates[i], coordinates[j], superposition=True)
                pairwise_rmsd[i, j] = rmsd_val
                pairwise_rmsd[j, i] = rmsd_val
                pbar.update(1)

    return pairwise_rmsd


def calculate_w1_distance(u: Universe, ref_frame: int = 0) -> np.ndarray:
    """
    Calculate Wasserstein-1 distance for all frames against a reference frame.

    Args:
        u: MDAnalysis Universe object
        ref_frame: Reference frame index

    Returns:
        Array of W1 distances for each frame compared to the reference
    """
    # Select CA atoms
    ca_atoms = u.select_atoms("name CA")
    n_frames = u.trajectory.n_frames

    # Get reference coordinates
    u.trajectory[ref_frame]
    ref_coordinates = ca_atoms.positions.copy()
    ref_distances = squareform(pdist(ref_coordinates))

    # Calculate W1 distances
    w1_distances = np.zeros(n_frames)
    for i, _ in tqdm(enumerate(u.trajectory), total=n_frames, desc="Calculating W1 distances"):
        # Calculate pairwise distances for current frame
        current_distances = squareform(pdist(ca_atoms.positions))

        # Calculate Wasserstein distance between distance distributions
        ref_dist_flat = ref_distances.flatten()
        curr_dist_flat = current_distances.flatten()
        w1_distances[i] = wasserstein_distance(ref_dist_flat, curr_dist_flat)

    return w1_distances


def calculate_pairwise_w1_distance(u: Universe) -> np.ndarray:
    """
    Calculate pairwise Wasserstein-1 distances between all frames.

    Args:
        u: MDAnalysis Universe object

    Returns:
        Square matrix of pairwise W1 distances
    """
    # Select CA atoms
    ca_atoms = u.select_atoms("name CA")
    n_frames = u.trajectory.n_frames

    # Initialize matrix for pairwise W1 distances
    pairwise_w1 = np.zeros((n_frames, n_frames))

    # Store distance matrices for each frame
    distance_matrices = []
    for ts in tqdm(u.trajectory, total=n_frames, desc="Computing distance matrices"):
        dist_matrix = squareform(pdist(ca_atoms.positions))
        distance_matrices.append(dist_matrix.flatten())

    # Calculate pairwise W1 distances
    total_pairs = n_frames * (n_frames - 1) // 2
    with tqdm(total=total_pairs, desc="Calculating pairwise W1 distances") as pbar:
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                w1_dist = wasserstein_distance(distance_matrices[i], distance_matrices[j])
                pairwise_w1[i, j] = w1_dist
                pairwise_w1[j, i] = w1_dist
                pbar.update(1)

    return pairwise_w1


# ---- Network metric comparison functions ----


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
    # Replace NaN values with zeros before computing KL divergence
    p = np.nan_to_num(p, nan=0.0)
    q = np.nan_to_num(q, nan=0.0)

    # Add small epsilon to avoid zeros
    p = p + epsilon
    q = q + epsilon

    # Normalize to probability distributions
    p_sum = np.sum(p)
    q_sum = np.sum(q)

    # Check if sums are valid
    if p_sum <= 0 or q_sum <= 0:
        return 0.0

    p = p / p_sum
    q = q / q_sum

    # Calculate KL divergence
    kl_div = np.sum(p * np.log(p / q))

    # Return 0 if result is invalid
    return 0.0 if np.isnan(kl_div) or np.isinf(kl_div) else kl_div


def calculate_feature_kl_divergence(
    network_metrics: List[NetworkMetrics], residue_ids: List[int], ref_frame: int = 0
) -> Dict[str, np.ndarray]:
    """
    Calculate KL divergence for each network feature compared to a reference frame.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame
        residue_ids: List of residue IDs
        ref_frame: Reference frame index

    Returns:
        Dictionary mapping feature names to KL divergence values for each frame
    """
    # Prepare metric data
    metric_arrays = prepare_metric_data(network_metrics, residue_ids)
    n_frames = len(network_metrics)

    # Initialize dictionaries for KL divergence values
    kl_divergences = {}

    # Calculate KL divergence for each metric
    for metric_name, metric_data in metric_arrays.items():
        kl_divergences[metric_name] = np.zeros(n_frames)

        # Get reference distribution
        ref_distribution = metric_data[ref_frame]

        # Calculate KL divergence for each frame
        for i in range(n_frames):
            if i == ref_frame:
                kl_divergences[metric_name][i] = 0
            else:
                kl_divergences[metric_name][i] = calculate_kl_divergence(
                    ref_distribution, metric_data[i]
                )

    return kl_divergences


def calculate_pairwise_feature_kl_divergence(
    network_metrics: List[NetworkMetrics], residue_ids: List[int]
) -> Dict[str, np.ndarray]:
    """
    Calculate pairwise KL divergence between all frames for each network feature.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame
        residue_ids: List of residue IDs

    Returns:
        Dictionary mapping feature names to matrices of pairwise KL divergence values
    """
    # Prepare metric data
    metric_arrays = prepare_metric_data(network_metrics, residue_ids)
    n_frames = len(network_metrics)

    # Initialize dictionary for KL divergence matrices
    pairwise_kl_divergences = {}

    # Calculate pairwise KL divergence for each metric
    for metric_name, metric_data in metric_arrays.items():
        pairwise_kl_divergences[metric_name] = np.zeros((n_frames, n_frames))

        # Calculate pairwise KL divergence
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                kl_div = calculate_kl_divergence(metric_data[i], metric_data[j])
                pairwise_kl_divergences[metric_name][i, j] = kl_div
                pairwise_kl_divergences[metric_name][j, i] = kl_div

    return pairwise_kl_divergences


def calculate_global_metric_percent_difference(
    network_metrics: List[NetworkMetrics], ref_frame: int = 0
) -> Dict[str, np.ndarray]:
    """
    Calculate percentage difference for global metrics compared to a reference frame.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame
        ref_frame: Reference frame index

    Returns:
        Dictionary mapping global metric names to percentage difference values for each frame
    """
    # Prepare global metric data
    global_metric_arrays = prepare_global_metric_data(network_metrics)
    n_frames = len(network_metrics)

    # Initialize dictionary for percentage differences
    percent_differences = {}

    # Calculate percentage difference for each global metric
    for metric_name, metric_data in global_metric_arrays.items():
        percent_differences[metric_name] = np.zeros(n_frames)
        ref_value = metric_data[ref_frame]

        # Avoid division by zero
        if abs(ref_value) < 1e-10:
            ref_value = 1e-10

        # Calculate percentage difference
        for i in range(n_frames):
            if i == ref_frame:
                percent_differences[metric_name][i] = 0
            else:
                percent_differences[metric_name][i] = (
                    100 * abs(metric_data[i] - ref_value) / abs(ref_value)
                )

    return percent_differences


def calculate_pairwise_global_metric_percent_difference(
    network_metrics: List[NetworkMetrics],
) -> Dict[str, np.ndarray]:
    """
    Calculate pairwise percentage difference for global metrics between all frames.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame

    Returns:
        Dictionary mapping global metric names to matrices of pairwise percentage differences
    """
    # Prepare global metric data
    global_metric_arrays = prepare_global_metric_data(network_metrics)
    n_frames = len(network_metrics)

    # Initialize dictionary for pairwise percentage differences
    pairwise_percent_differences = {}

    # Calculate pairwise percentage difference for each global metric
    for metric_name, metric_data in global_metric_arrays.items():
        pairwise_percent_differences[metric_name] = np.zeros((n_frames, n_frames))

        # Calculate pairwise percentage difference
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                # Avoid division by zero
                denominator = abs(metric_data[i])
                if denominator < 1e-10:
                    denominator = 1e-10

                # Calculate percentage difference
                pct_diff = 100 * abs(metric_data[j] - metric_data[i]) / denominator
                pairwise_percent_differences[metric_name][i, j] = pct_diff
                pairwise_percent_differences[metric_name][j, i] = pct_diff

    return pairwise_percent_differences


# ---- Plotting and visualization functions ----


def plot_kl_divergence_vs_distance(
    kl_divergences: Dict[str, np.ndarray],
    distance_measure: np.ndarray,
    metric_names: List[str] = None,  # This parameter is now optional
    distance_name: str = "Distance",
    title: str = "KL Divergence vs. Distance",
    figsize: tuple = (18, 12),
) -> plt.Figure:
    """
    Create scatter plots comparing KL divergence of network features against a distance measure.

    Now plots all metrics in the same order for consistent comparison.

    Args:
        kl_divergences: Dictionary mapping feature names to KL divergence values
        distance_measure: Array of distance values
        metric_names: Optional list of metric names to plot (if None, all metrics are plotted)
        distance_name: Name of the distance measure
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Use all metrics if metric_names is not provided, sort them for consistent ordering
    if metric_names is None:
        metric_names = sorted(kl_divergences.keys())

    # Calculate grid dimensions
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.95)

    # Flatten axes for easier indexing
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_metrics == 1 else axes

    # Create scatter plots for each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        kl_div = kl_divergences[metric_name]

        # Skip reference frame (distance=0, KL=0) and invalid values
        mask = (distance_measure > 0) & ~np.isnan(kl_div) & ~np.isinf(kl_div) & (kl_div > 0)

        # Check if we have valid data points
        if np.sum(mask) <= 1:
            ax.text(
                0.5,
                0.5,
                "Insufficient valid data points",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlabel(distance_name)
            ax.set_ylabel("KL Divergence")
            ax.set_title(f"{metric_name} (No valid data)")
            continue

        # Create scatter plot
        scatter = ax.scatter(
            distance_measure[mask],
            kl_div[mask],
            alpha=0.7,
            c=np.arange(len(distance_measure))[mask],
            cmap="viridis",
            s=50,
            edgecolors="k",
            linewidths=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Frame Index")

        # Add best fit line
        try:
            x = distance_measure[mask]
            y = kl_div[mask]
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            ax.plot(x, poly(x), "r--", linewidth=2)

            # Calculate R-squared
            y_pred = poly(x)
            r2 = r2_score(y, y_pred)
            ax.text(
                0.05,
                0.95,
                f"R² = {r2:.3f}\ny = {coeffs[0]:.3f}x + {coeffs[1]:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
            )
        except Exception as e:
            # Handle any errors in fitting
            ax.text(
                0.05,
                0.95,
                f"Fit error: {str(e)}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
            )

        # Set labels
        ax.set_xlabel(distance_name)
        ax.set_ylabel("KL Divergence")
        ax.set_title(metric_name)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig


def plot_rmsd_vs_w1_distance(
    rmsd: np.ndarray,
    w1_distance: np.ndarray,
    title: str = "RMSD vs. W1 Distance",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Create scatter plot comparing RMSD and W1 distance.

    Args:
        rmsd: Array of RMSD values
        w1_distance: Array of W1 distance values
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Skip reference frame (distance=0)
    mask = (rmsd > 0) & (w1_distance > 0)

    # Create scatter plot
    scatter = ax.scatter(
        rmsd[mask],
        w1_distance[mask],
        alpha=0.7,
        c=np.arange(len(rmsd))[mask],
        cmap="viridis",
        s=50,
        edgecolors="k",
        linewidths=0.5,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Frame Index")

    # Add best fit line
    if np.sum(mask) > 1:
        x = rmsd[mask]
        y = w1_distance[mask]
        coeffs = np.polyfit(x, y, 1)
        poly = np.poly1d(coeffs)
        ax.plot(x, poly(x), "r--", linewidth=2)

        # Calculate R-squared and Pearson correlation
        y_pred = poly(x)
        r2 = r2_score(y, y_pred)
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}\nCorrelation = {corr:.3f}\ny = {coeffs[0]:.3f}x + {coeffs[1]:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
        )

    # Set labels
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("W1 Distance")
    ax.set_title("Comparison of Distance Metrics")

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_global_metrics_vs_distance(
    percent_differences: Dict[str, np.ndarray],
    distance_measure: np.ndarray,
    distance_name: str,
    title: str = "Global Metrics % Difference vs. Distance",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Create scatter plots comparing percentage difference of global metrics against a distance measure.

    Args:
        percent_differences: Dictionary mapping global metric names to percentage difference values
        distance_measure: Array of distance values
        distance_name: Name of the distance measure
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Calculate grid dimensions
    n_metrics = len(percent_differences)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.95)

    # Flatten axes for easier indexing
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Create scatter plots for each global metric
    for i, (metric_name, percent_diff) in enumerate(percent_differences.items()):
        ax = axes[i]

        # Skip reference frame (distance=0, percent_diff=0)
        mask = distance_measure > 0

        # Create scatter plot
        scatter = ax.scatter(
            distance_measure[mask],
            percent_diff[mask],
            alpha=0.7,
            c=np.arange(len(distance_measure))[mask],
            cmap="viridis",
            s=50,
            edgecolors="k",
            linewidths=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Frame Index")

        # Add best fit line
        if np.sum(mask) > 1:
            x = distance_measure[mask]
            y = percent_diff[mask]
            coeffs = np.polyfit(x, y, 1)
            poly = np.poly1d(coeffs)
            ax.plot(x, poly(x), "r--", linewidth=2)

            # Calculate R-squared
            y_pred = poly(x)
            r2 = r2_score(y, y_pred)
            ax.text(
                0.05,
                0.95,
                f"R² = {r2:.3f}\ny = {coeffs[0]:.3f}x + {coeffs[1]:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"),
            )

        # Set labels
        ax.set_xlabel(distance_name)
        ax.set_ylabel("% Difference")
        ax.set_title(metric_name)

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig


def plot_feature_kl_heatmap(
    pairwise_kl_divergences: Dict[str, np.ndarray],
    metric_names: List[str],
    title: str = "Pairwise KL Divergence Heatmap",
    figsize: tuple = (18, 15),
) -> plt.Figure:
    """
    Create heatmaps for pairwise KL divergence of network features.

    Args:
        pairwise_kl_divergences: Dictionary mapping feature names to matrices of pairwise KL divergence values
        metric_names: List of metric names to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Calculate grid dimensions
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.95)

    # Flatten axes for easier indexing
    if n_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes] if n_metrics == 1 else axes

    # Create heatmaps for each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]
        kl_matrix = pairwise_kl_divergences[metric_name]

        # Create heatmap
        im = ax.imshow(kl_matrix, cmap="viridis", origin="lower")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("KL Divergence")

        # Set labels
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Frame Index")
        ax.set_title(metric_name)

        # Add frame ticks
        n_frames = kl_matrix.shape[0]
        tick_step = max(1, n_frames // 10)
        ax.set_xticks(np.arange(0, n_frames, tick_step))
        ax.set_yticks(np.arange(0, n_frames, tick_step))

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig


def plot_metric_cross_correlation(
    metric_arrays: Dict[str, np.ndarray],
    title: str = "Network Metrics Cross-Correlation",
    figsize: tuple = (14, 12),
) -> plt.Figure:
    """
    Create a heatmap showing the cross-correlation between different network metrics.

    Args:
        metric_arrays: Dictionary mapping metric names to arrays of shape (n_frames, n_residues)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Get list of metric names
    metric_names = list(metric_arrays.keys())
    n_metrics = len(metric_names)

    # Calculate average values per frame for each metric
    metric_avgs = {}
    for metric_name, metric_data in metric_arrays.items():
        # Average across residues for each frame
        metric_avgs[metric_name] = np.nanmean(metric_data, axis=1)

    # Calculate correlation matrix
    corr_matrix = np.zeros((n_metrics, n_metrics))
    for i, metric1 in enumerate(metric_names):
        for j, metric2 in enumerate(metric_names):
            # Compute correlation coefficient
            valid_mask = ~np.isnan(metric_avgs[metric1]) & ~np.isnan(metric_avgs[metric2])
            if np.sum(valid_mask) > 1:
                corr = np.corrcoef(
                    metric_avgs[metric1][valid_mask], metric_avgs[metric2][valid_mask]
                )[0, 1]
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0
            else:
                corr_matrix[i, j] = 0

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation Coefficient")

    # Add metric labels
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_metrics))
    ax.set_xticklabels(metric_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(metric_names)

    # Loop over data dimensions and create text annotations
    for i in range(n_metrics):
        for j in range(n_metrics):
            text = ax.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr_matrix[i, j]) > 0.5 else "black",
            )

    # Set title and layout
    ax.set_title(title)
    fig.tight_layout()

    return fig


def plot_summary_statistics_by_distance(
    kl_divergences: Dict[str, np.ndarray],
    rmsd: np.ndarray,
    w1_distance: np.ndarray,
    distance_type: str,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Create summary statistics plots for the analysis, separated by distance calculation type.

    Args:
        kl_divergences: Dictionary mapping feature names to KL divergence values
        rmsd: Array of RMSD values
        w1_distance: Array of W1 distance values
        distance_type: Type of distance measure ('RMSD' or 'W1')
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f"Summary Statistics - {distance_type} Distance", fontsize=16, y=0.95)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Select distance measure based on type
    if distance_type == "RMSD":
        distance_measure = rmsd
        distance_label = "RMSD (Å)"
    else:  # W1
        distance_measure = w1_distance
        distance_label = "W1 Distance"

    # 1. KL divergence distribution by feature
    ax = axes[0]

    # Prepare data for boxplot
    kl_data = []
    labels = []
    for metric_name, kl_div in kl_divergences.items():
        # Skip reference frame (KL=0) and invalid values
        mask = (kl_div > 0) & ~np.isnan(kl_div) & ~np.isinf(kl_div)
        if np.sum(mask) > 0:
            kl_data.append(kl_div[mask])
            labels.append(metric_name)

    # Create boxplot if we have data
    if kl_data:
        ax.boxplot(kl_data, vert=True, patch_artist=True, tick_labels=labels)

        # Set labels and rotate x-tick labels
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence Distribution by Feature")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid KL divergence data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # 2. Distance measure distribution
    ax = axes[1]

    # Skip reference frame (distance=0)
    dist_mask = distance_measure > 0

    # Create histogram
    if np.sum(dist_mask) > 0:
        ax.hist(distance_measure[dist_mask], bins=20, alpha=0.7, color="steelblue")

        # Set labels
        ax.set_xlabel(distance_label)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{distance_type} Distance Distribution")
    else:
        ax.text(
            0.5, 0.5, "No valid distance data", ha="center", va="center", transform=ax.transAxes
        )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # 3. Correlation matrix between KL divergence and distance
    ax = axes[2]

    # Prepare correlation data
    corr_data = {}
    for metric_name, kl_div in kl_divergences.items():
        # Skip reference frame and invalid values
        dist_mask = (distance_measure > 0) & (kl_div > 0) & ~np.isnan(kl_div) & ~np.isinf(kl_div)

        if np.sum(dist_mask) > 1:
            corr = np.corrcoef(distance_measure[dist_mask], kl_div[dist_mask])[0, 1]
            if not np.isnan(corr):
                corr_data[metric_name] = corr

    # Create correlation bar chart if we have data
    if corr_data:
        # Sort by absolute correlation value
        sorted_items = sorted(corr_data.items(), key=lambda x: abs(x[1]), reverse=True)
        metric_names = [item[0] for item in sorted_items]
        correlations = [item[1] for item in sorted_items]

        # Plot as horizontal bar chart
        colors = [plt.cm.RdBu(0.5 * (corr + 1)) for corr in correlations]  # Map -1:1 to colormap
        ax.barh(metric_names, correlations, color=colors, alpha=0.7)

        # Set labels
        ax.set_xlabel("Correlation Coefficient")
        ax.set_title(f"Feature Correlations with {distance_type} Distance")

        # Add reference line at zero
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    else:
        ax.text(
            0.5, 0.5, "No valid correlation data", ha="center", va="center", transform=ax.transAxes
        )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # 4. Feature importance based on correlation with distance
    ax = axes[3]

    # Use absolute correlation as importance
    if corr_data:
        # Sort by absolute correlation value
        importance_items = sorted(corr_data.items(), key=lambda x: abs(x[1]), reverse=True)
        imp_metrics = [item[0] for item in importance_items]
        importances = [abs(item[1]) for item in importance_items]

        # Plot as horizontal bar chart
        ax.barh(imp_metrics, importances, color="#2ca02c", alpha=0.7)

        # Set labels
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Feature Importance Based on {distance_type} Distance")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid feature importance data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig


def perform_significance_test(
    config_results: Dict[str, Dict], metric_name: str, distance_type: str, alpha: float = 0.05
) -> Dict[str, Dict]:
    """
    Perform significance testing between configurations for a specific metric.

    Args:
        config_results: Dictionary mapping config names to result dictionaries
        metric_name: Name of the metric to test
        distance_type: Type of distance measure ('RMSD' or 'W1')
        alpha: Significance level

    Returns:
        Dictionary with significance test results
    """
    from scipy import stats

    configs = list(config_results.keys())
    n_configs = len(configs)

    # Initialize results dictionary
    test_results = {"p_values": {}, "significant": {}, "test_used": "Mann-Whitney U test"}

    # For each pair of configurations
    for i in range(n_configs):
        for j in range(i + 1, n_configs):
            config1 = configs[i]
            config2 = configs[j]
            pair_key = f"{config1} vs {config2}"

            # Get KL divergence values for each config
            kl_div1 = config_results[config1]["kl_divergences"][metric_name]
            kl_div2 = config_results[config2]["kl_divergences"][metric_name]

            # Get distance measure
            if distance_type == "RMSD":
                dist1 = config_results[config1]["rmsd"]
                dist2 = config_results[config2]["rmsd"]
            else:  # W1
                dist1 = config_results[config1]["w1"]
                dist2 = config_results[config2]["w1"]

            # Skip reference frame (distance=0) and invalid values
            mask1 = (dist1 > 0) & (kl_div1 > 0) & ~np.isnan(kl_div1) & ~np.isinf(kl_div1)
            mask2 = (dist2 > 0) & (kl_div2 > 0) & ~np.isnan(kl_div2) & ~np.isinf(kl_div2)

            # Only perform test if we have enough data
            if np.sum(mask1) > 5 and np.sum(mask2) > 5:
                # Mann-Whitney U test (non-parametric, doesn't assume normal distribution)
                stat, p_value = stats.mannwhitneyu(
                    kl_div1[mask1], kl_div2[mask2], alternative="two-sided"
                )

                # Store results
                test_results["p_values"][pair_key] = p_value
                test_results["significant"][pair_key] = p_value < alpha
            else:
                # Not enough data
                test_results["p_values"][pair_key] = None
                test_results["significant"][pair_key] = False

    return test_results


def compare_configs_by_distance(
    results: Dict[str, Dict],
    output_dir: str,
    distance_type: str,
    alpha: float = 0.05,
    figsize: tuple = (14, 10),
) -> None:
    """
    Compare results across configurations for a specific distance type.

    Args:
        results: Dictionary mapping config names to result dictionaries
        output_dir: Output directory for results
        distance_type: Type of distance measure ('RMSD' or 'W1')
        alpha: Significance level for tests
        figsize: Figure size
    """
    # Get configs and find all unique metrics
    config_names = list(results.keys())

    # Collect all unique metrics across configurations
    all_metrics = set()
    for config_name, result in results.items():
        if "metric_correlations" in result and result["metric_correlations"]:
            for metric in result["metric_correlations"].keys():
                all_metrics.add(metric)

    # Skip comparison if there are no metrics to compare
    if not all_metrics:
        print(f"No valid metrics found for {distance_type} comparison across configurations")
        return

    # Create figure for correlation comparison
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for bar chart grouped by metric
    metrics_to_plot = list(all_metrics)
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(config_names)

    # Plot bars for each configuration
    for i, config_name in enumerate(config_names):
        correlations = []
        for metric in metrics_to_plot:
            if "metric_correlations" in results[config_name]:
                correlations.append(results[config_name]["metric_correlations"].get(metric, 0))
            else:
                correlations.append(0)

        # Plot bars
        ax.bar(
            x + i * width - 0.4 + width / 2,
            correlations,
            width,
            label=config_name,
            alpha=0.7,
        )

    # Set up plot
    ax.set_ylabel(f"Correlation with {distance_type} Distance")
    ax.set_title(f"Comparison of Metric Correlations Across Configurations - {distance_type}")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save figure
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"metric_correlation_comparison_{distance_type}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Perform significance testing for all metrics
    significance_results = {}

    # Only test metrics that appear in all configs
    metrics_in_all_configs = []
    for metric in all_metrics:
        in_all = True
        for config in config_names:
            if metric not in results[config]["kl_divergences"] or np.all(
                np.isnan(results[config]["kl_divergences"][metric])
            ):
                in_all = False
                break
        if in_all:
            metrics_in_all_configs.append(metric)

    # Test all metrics that appear in all configurations
    metrics_to_test = metrics_in_all_configs

    # Create table for significance test results
    for metric in metrics_to_test:
        significance_results[metric] = perform_significance_test(
            results, metric, distance_type, alpha
        )

    # Create a summary table as a plot
    if significance_results:
        # Group results by config pair for better organization
        pair_results = {}
        all_pairs = set()

        for metric, result in significance_results.items():
            for pair in result["p_values"].keys():
                all_pairs.add(pair)

        for pair in all_pairs:
            pair_results[pair] = {"metrics": [], "p_values": [], "significant": []}

            for metric, result in significance_results.items():
                if pair in result["p_values"] and result["p_values"][pair] is not None:
                    pair_results[pair]["metrics"].append(metric)
                    pair_results[pair]["p_values"].append(result["p_values"][pair])
                    pair_results[pair]["significant"].append(result["significant"][pair])

        # Create a figure for each config pair to avoid overly large tables
        for pair, data in pair_results.items():
            if not data["metrics"]:
                continue

            # Sort metrics by p-value for each pair
            sorted_indices = np.argsort(data["p_values"])
            sorted_metrics = [data["metrics"][i] for i in sorted_indices]
            sorted_p_values = [data["p_values"][i] for i in sorted_indices]
            sorted_significant = [data["significant"][i] for i in sorted_indices]

            # Limit to top 30 metrics for readability if there are many
            max_display = 30
            if len(sorted_metrics) > max_display:
                print(
                    f"Limiting significance table for {pair} to top {max_display} metrics with lowest p-values"
                )
                sorted_metrics = sorted_metrics[:max_display]
                sorted_p_values = sorted_p_values[:max_display]
                sorted_significant = sorted_significant[:max_display]

            # Create figure for this pair's results
            fig_height = min(
                20, 2 + len(sorted_metrics) * 0.3
            )  # Scale height based on number of metrics
            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.axis("tight")
            ax.axis("off")

            # Prepare table data
            table_data = [["Metric", "p-value", "Significant"]]
            for i in range(len(sorted_metrics)):
                table_data.append(
                    [
                        sorted_metrics[i],
                        f"{sorted_p_values[i]:.5f}",
                        "Yes" if sorted_significant[i] else "No",
                    ]
                )

            # Create table
            table = ax.table(
                cellText=table_data, loc="center", cellLoc="center", colWidths=[0.6, 0.2, 0.2]
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(10)

            # Color the header row
            for j, cell in enumerate(table_data[0]):
                table[(0, j)].set_facecolor("#4472C4")
                table[(0, j)].set_text_props(color="white")

            # Color the significant results
            for i in range(1, len(table_data)):
                if table_data[i][2] == "Yes":
                    table[(i, 2)].set_facecolor("#C6EFCE")
                else:
                    table[(i, 2)].set_facecolor("#FFC7CE")

            # Set title
            plt.title(
                f"Significance Tests: {pair} ({distance_type} Distance, α={alpha})", fontsize=14
            )
            plt.tight_layout()

            # Save figure
            safe_pair_name = pair.replace(" ", "_").replace("/", "_")
            plt.savefig(
                os.path.join(output_dir, f"significance_test_{safe_pair_name}_{distance_type}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        # Create a summary figure showing counts of significant metrics for each pair
        fig, ax = plt.subplots(figsize=(10, 6))

        # Count significant metrics for each pair
        pairs = list(pair_results.keys())
        sig_counts = []
        total_counts = []

        for pair in pairs:
            sig_count = sum(pair_results[pair]["significant"])
            total_count = len(pair_results[pair]["significant"])
            sig_counts.append(sig_count)
            total_counts.append(total_count)

        # Create stacked bar chart
        non_sig_counts = [total - sig for total, sig in zip(total_counts, sig_counts)]

        ax.bar(pairs, sig_counts, label="Significant", color="#C6EFCE")
        ax.bar(pairs, non_sig_counts, bottom=sig_counts, label="Not Significant", color="#FFC7CE")

        # Add percentage labels
        for i, (sig, total) in enumerate(zip(sig_counts, total_counts)):
            if total > 0:
                percent = (sig / total) * 100
                ax.text(
                    i, total / 2, f"{percent:.1f}%", ha="center", va="center", fontweight="bold"
                )

        # Set labels and title
        ax.set_ylabel("Number of Metrics")
        ax.set_title(f"Significance Test Summary ({distance_type} Distance, α={alpha})")
        ax.legend()

        # Rotate x-tick labels if there are many pairs
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save figure
        plt.savefig(
            os.path.join(output_dir, f"significance_test_summary_{distance_type}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    else:
        print(f"No valid significance test results for {distance_type}")

    print(f"{distance_type} configuration comparison completed.")


def analyze_md_ensemble(
    topology_path: str,
    trajectory_path: str,
    config,
    config_name: str,
    output_dir: str = "md_ensemble_analysis",
    ref_frame: int = 0,
    top_k_metrics: int = 5,
) -> None:
    """
    Perform comprehensive analysis of an MD ensemble.

    Args:
        topology_path: Path to topology file
        trajectory_path: Path to trajectory file
        config: Configuration object for network analysis
        config_name: Name of the configuration
        output_dir: Output directory for results
        ref_frame: Reference frame index
        top_k_metrics: Number of top metrics to analyze in detail
    """
    # Create config-specific output directory
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    print(f"Starting analysis of MD ensemble with config: {config_name}")
    print(f"Topology: {topology_path}")
    print(f"Trajectory: {trajectory_path}")
    print(f"Reference frame: {ref_frame}")
    print("-" * 80)

    try:
        # Load universe and create output directories
        universe = Universe(topology_path, trajectory_path)
        n_frames = universe.trajectory.n_frames

        # Create config and build network
        print("Building hydrogen bond network...")
        features = build_hbond_network([universe], config)
        residue_ids = features.residue_ids
        network_metrics = features.network_metrics

        # Calculate distance measures from reference
        print(f"Calculating distance measures using reference frame {ref_frame}...")
        rmsd_to_ref = calculate_rmsd(universe, ref_frame)
        w1_to_ref = calculate_w1_distance(universe, ref_frame)

        # Calculate pairwise distances
        print("Calculating pairwise distance measures between all frames...")
        pairwise_rmsd = calculate_pairwise_rmsd(universe)
        pairwise_w1 = calculate_pairwise_w1_distance(universe)

        # Calculate KL divergence from reference
        print(f"Calculating KL divergence using reference frame {ref_frame}...")
        kl_divergences = calculate_feature_kl_divergence(network_metrics, residue_ids, ref_frame)

        # Calculate pairwise KL divergence
        print("Calculating pairwise KL divergence between all frames...")
        pairwise_kl_divergences = calculate_pairwise_feature_kl_divergence(
            network_metrics, residue_ids
        )

        # Calculate global metric percent differences
        print("Calculating global metric percentage differences...")
        global_percent_diffs = calculate_global_metric_percent_difference(
            network_metrics, ref_frame
        )
        pairwise_global_percent_diffs = calculate_pairwise_global_metric_percent_difference(
            network_metrics
        )

        # Get all metric names
        all_metrics = list(kl_divergences.keys())

        # Find top k metrics with highest correlation to distances
        print(f"Identifying top {top_k_metrics} metrics by correlation...")
        metric_correlations = {}
        for metric_name, kl_div in kl_divergences.items():
            # Skip reference frame and invalid values
            mask = (
                (rmsd_to_ref > 0)
                & (w1_to_ref > 0)
                & (kl_div > 0)
                & ~np.isnan(kl_div)
                & ~np.isinf(kl_div)
            )
            if np.sum(mask) > 1:
                rmsd_corr = abs(np.corrcoef(rmsd_to_ref[mask], kl_div[mask])[0, 1])
                w1_corr = abs(np.corrcoef(w1_to_ref[mask], kl_div[mask])[0, 1])
                # Check for valid correlation values
                if not (np.isnan(rmsd_corr) or np.isnan(w1_corr)):
                    metric_correlations[metric_name] = (rmsd_corr + w1_corr) / 2

        # Sort metrics by correlation
        sorted_metrics = sorted(metric_correlations.items(), key=lambda x: x[1], reverse=True)
        top_metrics = [metric for metric, _ in sorted_metrics[:top_k_metrics]]

        print(f"Top {min(top_k_metrics, len(sorted_metrics))} metrics for config {config_name}:")
        for i, (metric, corr) in enumerate(sorted_metrics[:top_k_metrics]):
            print(f"{i + 1}. {metric} (correlation: {corr:.3f})")

        # Create plots directory
        plots_dir = os.path.join(config_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Plot KL divergence vs. RMSD for ALL metrics (not just top ones)
        print("Creating KL divergence vs. RMSD plots...")
        try:
            fig_kl_rmsd = plot_kl_divergence_vs_distance(
                kl_divergences,
                rmsd_to_ref,
                distance_name="RMSD (Å)",
                title=f"KL Divergence vs. RMSD - {config_name} (Reference: Frame {ref_frame})",
            )
            fig_kl_rmsd.savefig(
                os.path.join(plots_dir, "kl_divergence_vs_rmsd.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig_kl_rmsd)
        except Exception as e:
            print(f"Error creating KL vs RMSD plot: {str(e)}")

        # 2. Plot KL divergence vs. W1 distance for ALL metrics (not just top ones)
        print("Creating KL divergence vs. W1 distance plots...")
        try:
            fig_kl_w1 = plot_kl_divergence_vs_distance(
                kl_divergences,
                w1_to_ref,
                distance_name="W1 Distance",
                title=f"KL Divergence vs. W1 Distance - {config_name} (Reference: Frame {ref_frame})",
            )
            fig_kl_w1.savefig(
                os.path.join(plots_dir, "kl_divergence_vs_w1.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig_kl_w1)
        except Exception as e:
            print(f"Error creating KL vs W1 plot: {str(e)}")

        # 3. Plot RMSD vs. W1 distance
        print("Creating RMSD vs. W1 distance plot...")
        try:
            fig_rmsd_w1 = plot_rmsd_vs_w1_distance(
                rmsd_to_ref,
                w1_to_ref,
                title=f"RMSD vs. W1 Distance - {config_name} (Reference: Frame {ref_frame})",
            )
            fig_rmsd_w1.savefig(
                os.path.join(plots_dir, "rmsd_vs_w1.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig_rmsd_w1)
        except Exception as e:
            print(f"Error creating RMSD vs W1 plot: {str(e)}")

        # 4. Plot global metrics vs. RMSD
        print("Creating global metrics vs. RMSD plots...")
        try:
            fig_global_rmsd = plot_global_metrics_vs_distance(
                global_percent_diffs,
                rmsd_to_ref,
                "RMSD (Å)",
                title=f"Global Metrics % Difference vs. RMSD - {config_name} (Reference: Frame {ref_frame})",
            )
            fig_global_rmsd.savefig(
                os.path.join(plots_dir, "global_metrics_vs_rmsd.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig_global_rmsd)
        except Exception as e:
            print(f"Error creating global metrics vs RMSD plot: {str(e)}")

        # 5. Plot global metrics vs. W1 distance
        print("Creating global metrics vs. W1 distance plots...")
        try:
            fig_global_w1 = plot_global_metrics_vs_distance(
                global_percent_diffs,
                w1_to_ref,
                "W1 Distance",
                title=f"Global Metrics % Difference vs. W1 Distance - {config_name} (Reference: Frame {ref_frame})",
            )
            fig_global_w1.savefig(
                os.path.join(plots_dir, "global_metrics_vs_w1.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig_global_w1)
        except Exception as e:
            print(f"Error creating global metrics vs W1 plot: {str(e)}")

        # 6. Plot heatmaps for ALL metrics, not just top ones
        print("Creating KL divergence heatmaps for all metrics...")
        try:
            fig_heatmap = plot_feature_kl_heatmap(
                pairwise_kl_divergences,
                all_metrics,
                title=f"Pairwise KL Divergence Heatmap - {config_name} (All Metrics)",
            )
            fig_heatmap.savefig(
                os.path.join(plots_dir, "kl_divergence_heatmap_all_metrics.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_heatmap)
        except Exception as e:
            print(f"Error creating KL heatmap for all metrics: {str(e)}")

        # 7a. Plot summary statistics for RMSD
        print("Creating RMSD summary statistics plots...")
        try:
            fig_summary_rmsd = plot_summary_statistics_by_distance(
                kl_divergences, rmsd_to_ref, w1_to_ref, "RMSD"
            )
            fig_summary_rmsd.savefig(
                os.path.join(plots_dir, "summary_statistics_rmsd.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig_summary_rmsd)
        except Exception as e:
            print(f"Error creating RMSD summary statistics plot: {str(e)}")

        # 7b. Plot summary statistics for W1
        print("Creating W1 summary statistics plots...")
        try:
            fig_summary_w1 = plot_summary_statistics_by_distance(
                kl_divergences, rmsd_to_ref, w1_to_ref, "W1"
            )
            fig_summary_w1.savefig(
                os.path.join(plots_dir, "summary_statistics_w1.png"), dpi=300, bbox_inches="tight"
            )
            plt.close(fig_summary_w1)
        except Exception as e:
            print(f"Error creating W1 summary statistics plot: {str(e)}")

        # 8. Plot cross-correlation heatmap of all metrics
        print("Creating cross-correlation heatmap of all metrics...")
        try:
            # Prepare metric data for cross-correlation
            metric_arrays = prepare_metric_data(network_metrics, residue_ids)

            fig_cross_corr = plot_metric_cross_correlation(
                metric_arrays,
                title=f"Network Metrics Cross-Correlation - {config_name}",
            )
            fig_cross_corr.savefig(
                os.path.join(plots_dir, "metrics_cross_correlation.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig_cross_corr)
        except Exception as e:
            print(f"Error creating metrics cross-correlation plot: {str(e)}")

        # Save analysis data
        print("Saving analysis data...")
        np.savez(
            os.path.join(config_dir, "analysis_data.npz"),
            rmsd_to_ref=rmsd_to_ref,
            w1_to_ref=w1_to_ref,
            pairwise_rmsd=pairwise_rmsd,
            pairwise_w1=pairwise_w1,
            residue_ids=residue_ids,
        )

        print(f"Analysis for config '{config_name}' completed successfully!")
        print(f"Results saved in: {config_dir}")
        return {
            "config_name": config_name,
            "kl_divergences": kl_divergences,
            "rmsd": rmsd_to_ref,
            "w1": w1_to_ref,
            "top_metrics": top_metrics,
            "metric_correlations": metric_correlations,
        }

    except Exception as e:
        print(f"Error during analysis with config '{config_name}': {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def test_compare_md_ensemble():
    """Test the H-bond network based ensemble comparison with multiple configurations"""
    # Test parameters
    topology_path = "./tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "./tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    # Alternative test files
    # topology_path = "./tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
    # trajectory_path = topology_path.replace(".pdb", "_small.xtc")

    # Create main output directory
    output_dir = "./tests/_plots/md_ensemble_analysis_results"
    os.makedirs(output_dir, exist_ok=True)

    print("\nTesting H-bond network based MD ensemble comparison...")
    print("-" * 80)

    # H-bond parameter configurations
    configs = {
        "netHDX_standard": NetHDXConfig(
            distance_cutoff=[2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2, 6.5],
            angle_cutoff=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        "BV_standard": BV_model_Config(),
    }

    # Run analysis for each configuration
    results = {}
    for config_name, config in configs.items():
        print(f"\nConfiguration: {config_name}")

        # Run analysis for current configuration
        result = analyze_md_ensemble(
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            config=config,
            config_name=config_name,
            output_dir=output_dir,
            ref_frame=0,
            top_k_metrics=5,
        )

        if result:
            results[config_name] = result

    # Compare results across configurations if more than one configuration was successful
    if len(results) > 1:
        print("\nComparing results across configurations...")
        compare_dir = os.path.join(output_dir, "config_comparison")
        os.makedirs(compare_dir, exist_ok=True)

        # Compare configs by RMSD distance
        print("Comparing configurations using RMSD distance...")
        compare_configs_by_distance(results, compare_dir, "RMSD")

        # Compare configs by W1 distance
        print("Comparing configurations using W1 distance...")
        compare_configs_by_distance(results, compare_dir, "W1")

        print(f"Configuration comparison saved in: {compare_dir}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_compare_md_ensemble()
