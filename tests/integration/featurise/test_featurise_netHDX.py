from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from jaxent.forwardmodels.netHDX_functions import NetworkMetrics


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
    }

    # Fill arrays
    for frame_idx, frame_metrics in enumerate(network_metrics):
        for res_idx, res_id in enumerate(residue_ids):
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

    return metric_arrays


def plot_network_metrics_distribution(
    network_metrics: List[NetworkMetrics],
    residue_ids: List[int],
    title: str = "Network Metrics Distribution",
    figsize: tuple = (20, 14),
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

    # Create subplots
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, y=0.95)
    axes = axes.flatten()

    # Plot each metric
    for (metric_name, metric_data), ax in zip(metric_arrays.items(), axes):
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

    # Remove empty subplot if number of metrics is odd
    if len(metric_arrays) < len(axes):
        fig.delaxes(axes[-1])

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_framewise_network_metrics_heatmap(
    network_metrics: List[NetworkMetrics],
    residue_ids: List[int],
    title: str = "Framewise Network Metrics Heatmap",
    figsize: tuple = (20, 14),
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

    metrics = list(metric_arrays.keys())
    num_metrics = len(metrics)
    cols = 2
    rows = (num_metrics + 1) // cols

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


def test_network_metrics_plotting():
    """Test function for network metrics visualization"""
    from MDAnalysis import Universe

    from jaxent.forwardmodels.netHDX_functions import NetHDXConfig, build_hbond_network

    # Load test data and compute metrics
    topology_path = (
        # "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    )
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
    trajectory_path = "/home/alexi/Documents/interpretable-hdxer/notebooks/Figure_5_Poisoned_Ensemble/combined_ensembles/HOIP/AF2-Cleaned/HOIP_add-noise_poison0.xtc"
    # trajectory_path = "/home/alexi/Documents/interpretable-hdxer/notebooks/Figure_5_Poisoned_Ensemble/combined_ensembles/HOIP/AF2-MSAss/HOIP_add-noise_poison0.xtc"
    # topology_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated_max_plddt_1050.pdb"
    # trajectory_path = "/home/alexi/Documents/ValDX/raw_data/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10000_protonated.xtc"
    try:
        # Create universe and compute network metrics
        universe = Universe(topology_path, trajectory_path)
        config = NetHDXConfig()

        # Build network and get metrics
        features = build_hbond_network([universe], config)

        # Create visualization
        fig = plot_network_metrics_distribution(
            features.network_metrics,
            features.residue_ids,
            title=f"Network Metrics Distribution\n{universe.trajectory.n_frames} frames",
        )

        # Save plot
        output_path = "network_metrics_distribution.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Create heatmap visualization
        fig_heatmap = plot_framewise_network_metrics_heatmap(
            features.network_metrics,
            features.residue_ids,
            title=f"Framewise Network Metrics Heatmap\n{universe.trajectory.n_frames} frames",
        )

        # Save heatmap plot
        output_path_heatmap = "network_metrics_heatmap.png"
        fig_heatmap.savefig(output_path_heatmap, dpi=300, bbox_inches="tight")
        plt.close(fig_heatmap)

        print(f"Successfully created visualizations:\n{output_path}\n{output_path_heatmap}")
    except Exception as e:
        print(f"Error during testing: {str(e)}")


if __name__ == "__main__":
    test_network_metrics_plotting()
