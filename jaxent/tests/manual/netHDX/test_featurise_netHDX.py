import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis import Universe
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from jaxent.src.models.config import NetHDXConfig
from jaxent.src.models.func.netHDX import build_hbond_network
from jaxent.src.models.HDX.netHDX.features import NetworkMetrics


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


def plot_global_network_metrics(
    network_metrics: List[NetworkMetrics],
    title: str = "Global Network Metrics",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Create line plots showing the global network metrics across frames.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    global_metric_arrays = prepare_global_metric_data(network_metrics)
    n_frames = len(network_metrics)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Plot each metric
    x = np.arange(n_frames)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (metric_name, metric_data) in enumerate(global_metric_arrays.items()):
        # Handle NaN values
        valid_mask = ~np.isnan(metric_data)

        # Only plot if we have valid data
        if np.any(valid_mask):
            ax.plot(
                x[valid_mask],
                metric_data[valid_mask],
                label=metric_name,
                marker="o",
                markersize=4,
                color=colors[i % len(colors)],
                alpha=0.8,
                linewidth=2,
            )

    # Add labels and legend
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.tick_params(axis="both", labelsize=10)

    # Add frame information
    ax.text(
        0.02,
        0.02,
        f"Total frames: {n_frames}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()

    return fig


def plot_network_metrics_distribution(
    network_metrics: List[NetworkMetrics],
    residue_ids: List[int],
    title: str = "Network Metrics Distribution",
    figsize: tuple = (24, 18),  # Increased size for more subplots
) -> plt.Figure:
    """
    Create box plots showing the distribution of network metrics across frames for each residue.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame
        residue_ids: List of residue IDs
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    metric_arrays = prepare_metric_data(network_metrics, residue_ids)

    # Calculate grid dimensions
    num_metrics = len(metric_arrays)
    cols = 3  # Increase columns to 3
    rows = (num_metrics + cols - 1) // cols  # Ceiling division

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.95)
    axes = axes.flatten()

    # Plot each metric
    for i, ((metric_name, metric_data), ax) in enumerate(zip(metric_arrays.items(), axes)):
        # Create box plot
        bp = ax.boxplot(
            [metric_data[:, i] for i in range(len(residue_ids))],
            positions=residue_ids,
            widths=0.7,
            showfliers=False,
        )

        # Customize box plot colors
        plt.setp(bp["boxes"], color="blue", alpha=0.6)
        plt.setp(bp["medians"], color="red")
        plt.setp(bp["whiskers"], color="black", alpha=0.6)
        plt.setp(bp["caps"], color="black", alpha=0.6)

        # Customize axis
        ax.set_title(metric_name)
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("Value")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Rotate x-axis labels for better readability
        if len(residue_ids) > 20:
            ax.tick_params(axis="x", rotation=45)

        # Set reasonable y-axis limits
        if "Path Length" in metric_name:
            ax.set_ylim(bottom=0)
            # Replace inf values with NaN for better visualization
            metric_data[metric_data == float("inf")] = np.nan

    # Remove empty subplots
    for ax in axes[num_metrics:]:
        fig.delaxes(ax)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_framewise_network_metrics_heatmap(
    network_metrics: List[NetworkMetrics],
    residue_ids: List[int],
    title: str = "Framewise Network Metrics Heatmap",
    figsize: tuple = (24, 18),  # Increased size for more subplots
) -> plt.Figure:
    """
    Create heatmaps showing the framewise network metrics for each residue.

    Args:
        network_metrics: List of NetworkMetrics objects for each frame
        residue_ids: List of residue IDs
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    metric_arrays = prepare_metric_data(network_metrics, residue_ids)

    # Calculate grid dimensions
    num_metrics = len(metric_arrays)
    cols = 3  # Increase columns to 3
    rows = (num_metrics + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.95)
    axes = axes.flatten()

    for ax, (metric_name, metric_data) in zip(axes, metric_arrays.items()):
        im = ax.imshow(
            metric_data.T,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            extent=[0, len(network_metrics), residue_ids[0], residue_ids[-1]],
        )
        ax.set_title(metric_name)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Residue ID")
        fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)

    # Remove empty subplots
    for ax in axes[num_metrics:]:
        fig.delaxes(ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_pca_global_metrics(
    universe: Universe,
    network_metrics: List[NetworkMetrics],
    title: str = "PCA of CA Distances Colored by Global Metrics",
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """
    Create PCA plots of CA atom pairwise distances colored by global network metrics.

    Args:
        universe: MDAnalysis Universe object containing trajectory
        network_metrics: List of NetworkMetrics objects for each frame
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """

    # Extract global metrics
    global_metric_arrays = prepare_global_metric_data(network_metrics)
    n_metrics = len(global_metric_arrays)

    # Select CA atoms
    ca_atoms = universe.select_atoms("name CA")
    n_ca = len(ca_atoms)
    n_frames = universe.trajectory.n_frames

    # Initialize array to store distances
    num_pairs = (n_ca * (n_ca - 1)) // 2
    pairwise_distances = np.zeros((n_frames, num_pairs))

    # Loop through trajectory and calculate pairwise distances
    for i, ts in enumerate(universe.trajectory):
        # Calculate all pairwise distances using pdist
        positions = ca_atoms.positions
        distances = pdist(positions)
        pairwise_distances[i] = distances

    # Perform PCA on pairwise distances
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(pairwise_distances)

    # Create figure with subplots (one per metric)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # If only one metric, make axes iterable
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for i, (metric_name, metric_data) in enumerate(global_metric_arrays.items()):
        ax = axes[i]

        # Create scatter plot with color based on metric
        scatter = ax.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=metric_data,
            cmap="viridis",
            alpha=0.8,
            s=50,
            edgecolors="k",
            linewidths=0.5,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(metric_name)

        # Set labels
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        ax.set_title(f"PCA colored by {metric_name}")

        # Add grid
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def test_network_metrics_plotting():
    """Test function for network metrics visualization"""

    # Load test data and compute metrics
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )

    # Create test output directory
    test_dir = "tests/_plots/network_metrics_viz"
    os.system(f"rm -rf {test_dir}")
    os.makedirs(test_dir, exist_ok=True)

    print("\nTesting network metrics visualization functionality...")
    print("-" * 80)

    try:
        # Create universe and compute network metrics
        universe = Universe(topology_path, trajectory_path)
        config = NetHDXConfig()

        # Create configuration-specific directory
        config_name = "netHDX_standard"
        config_dir = os.path.join(test_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Build network and get metrics
        features = build_hbond_network([universe], config)

        # Create visualization for per-residue metrics
        fig = plot_network_metrics_distribution(
            features.network_metrics,
            features.residue_ids,
            title=f"Network Metrics Distribution\n{universe.trajectory.n_frames} frames",
        )

        # Save plot
        output_path = os.path.join(config_dir, "network_metrics_distribution.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization: {output_path}")

        # Create heatmap visualization
        fig_heatmap = plot_framewise_network_metrics_heatmap(
            features.network_metrics,
            features.residue_ids,
            title=f"Framewise Network Metrics Heatmap\n{universe.trajectory.n_frames} frames",
        )

        # Save heatmap plot
        output_path_heatmap = os.path.join(config_dir, "network_metrics_heatmap.png")
        fig_heatmap.savefig(output_path_heatmap, dpi=300, bbox_inches="tight")
        plt.close(fig_heatmap)
        print(f"Saved visualization: {output_path_heatmap}")

        # Create and save global metrics plot
        fig_global = plot_global_network_metrics(
            features.network_metrics,
            title=f"Global Network Metrics\n{universe.trajectory.n_frames} frames",
        )
        output_path_global = os.path.join(config_dir, "global_network_metrics.png")
        fig_global.savefig(output_path_global, dpi=300, bbox_inches="tight")
        plt.close(fig_global)
        print(f"Saved visualization: {output_path_global}")

        # Create and save PCA with global metrics plot
        fig_pca = plot_pca_global_metrics(
            universe,
            features.network_metrics,
            title=f"PCA of CA Distances Colored by Global Metrics\n{universe.trajectory.n_frames} frames",
        )
        output_path_pca = os.path.join(config_dir, "pca_global_metrics.png")
        fig_pca.savefig(output_path_pca, dpi=300, bbox_inches="tight")
        plt.close(fig_pca)
        print(f"Saved visualization: {output_path_pca}")

        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    test_network_metrics_plotting()
