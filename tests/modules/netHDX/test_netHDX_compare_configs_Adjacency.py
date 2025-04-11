import os
from typing import Dict, Union

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.rms import rmsd
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

from jaxent.models.config import BV_model_Config, NetHDXConfig
from jaxent.models.func.netHDX import calculate_trajectory_hbonds, calculate_trajectory_hbonds_BV


def calculate_matrix_rmsd(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Calculate RMSD between two adjacency matrices.

    Args:
        matrix1: First adjacency matrix
        matrix2: Second adjacency matrix

    Returns:
        RMSD value
    """
    return np.sqrt(np.mean((matrix1 - matrix2) ** 2))


def calculate_frame_ca_rmsd(universe: mda.Universe, frame1: int, frame2: int) -> float:
    """
    Calculate RMSD between CA atom coordinates for two frames of the same universe.

    Args:
        universe: MDAnalysis Universe
        frame1: First frame number
        frame2: Second frame number

    Returns:
        RMSD value after alignment
    """
    # Select CA atoms
    ca_atoms = universe.select_atoms("name CA")

    # Get coordinates for frame 1
    universe.trajectory[frame1]
    coords1 = ca_atoms.positions.copy()

    # Get coordinates for frame 2
    universe.trajectory[frame2]
    coords2 = ca_atoms.positions.copy()

    # Calculate RMSD after optimal alignment
    return rmsd(coords1, coords2, superposition=True)


def compare_adjacency_matrices(
    universe: mda.Universe,
    configs: Dict[str, Union[NetHDXConfig, BV_model_Config]],
    max_frames: int = 100,
    output_dir: str = "config_comparison_results",
) -> Dict[str, Dict[str, float]]:
    """
    Compare adjacency matrices created by different configurations.

    Args:
        universe: MDAnalysis Universe containing trajectory
        configs: Dictionary mapping config names to configuration objects
        max_frames: Maximum number of frames to use for comparison (limits computational cost)
        output_dir: Directory to save output files

    Returns:
        Dictionary with statistical metrics for each config
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    n_frames = min(len(universe.trajectory), max_frames)
    frames = np.linspace(0, len(universe.trajectory) - 1, n_frames, dtype=int)

    print(f"Using {n_frames} frames for pairwise comparison")

    # Create all pairs of frames
    frame_pairs = []
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            frame_pairs.append((frames[i], frames[j]))

    print(f"Created {len(frame_pairs)} frame pairs for comparison")

    # Calculate CA RMSDs between all frame pairs
    ca_rmsds = {}
    for pair in tqdm(frame_pairs, desc="Calculating CA RMSDs"):
        frame1, frame2 = pair
        ca_rmsds[pair] = calculate_frame_ca_rmsd(universe, frame1, frame2)

    print("Calculated CA RMSDs between all frame pairs")

    # Calculate adjacency matrices for each config
    all_matrices = {}
    rmsd_data = {}

    for config_name, config in configs.items():
        print(f"Processing config: {config_name}")

        # Calculate adjacency matrices for the entire trajectory
        if isinstance(config, NetHDXConfig):
            all_frame_matrices = calculate_trajectory_hbonds(universe, config)
        elif isinstance(config, BV_model_Config):
            all_frame_matrices = calculate_trajectory_hbonds_BV(universe, config)
        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        # Calculate matrix RMSDs between all frame pairs
        pair_data = {}
        for pair in tqdm(frame_pairs, desc=f"Computing matrix RMSDs for {config_name}"):
            frame1, frame2 = pair
            matrix1 = all_frame_matrices[frame1]
            matrix2 = all_frame_matrices[frame2]

            matrix_rmsd = calculate_matrix_rmsd(matrix1, matrix2)

            pair_data[pair] = {"matrix_rmsd": matrix_rmsd, "ca_rmsd": ca_rmsds[pair]}

        rmsd_data[config_name] = pair_data

    # Evaluate statistical power
    stat_power = {}

    for config_name, pair_data in rmsd_data.items():
        # Extract matrix and CA RMSDs
        matrix_rmsds = [data["matrix_rmsd"] for data in pair_data.values()]
        ca_rmsds_values = [data["ca_rmsd"] for data in pair_data.values()]

        # Calculate correlation
        pearson_corr, pearson_p = pearsonr(matrix_rmsds, ca_rmsds_values)
        spearman_corr, spearman_p = spearmanr(matrix_rmsds, ca_rmsds_values)

        # Calculate range and variance ratios
        ca_range = max(ca_rmsds_values) - min(ca_rmsds_values)
        matrix_range = max(matrix_rmsds) - min(matrix_rmsds)
        range_ratio = matrix_range / ca_range if ca_range > 0 else 0

        ca_std = np.std(ca_rmsds_values)
        matrix_std = np.std(matrix_rmsds)
        normalized_std_ratio = matrix_std / ca_std if ca_std > 0 else 0

        # Sensitivity score: combination of correlation and range ratio
        sensitivity_score = (abs(pearson_corr) + abs(spearman_corr)) / 2 * range_ratio

        stat_power[config_name] = {
            "pearson_corr": pearson_corr,
            "pearson_p": pearson_p,
            "spearman_corr": spearman_corr,
            "spearman_p": spearman_p,
            "range_ratio": range_ratio,
            "normalized_std_ratio": normalized_std_ratio,
            "sensitivity_score": sensitivity_score,
            "matrix_rmsds": matrix_rmsds,
            "ca_rmsds": ca_rmsds_values,
        }

    # Visualize results
    visualize_results(stat_power, output_dir)

    return stat_power


def visualize_results(stat_power: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """
    Create visualizations for config comparison results.

    Args:
        stat_power: Dictionary with statistical metrics for each config
        output_dir: Directory to save visualization files
    """
    # 1. Scatter plots for each config (Matrix RMSD vs CA RMSD)
    n_configs = len(stat_power)
    n_cols = min(2, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 7, n_rows * 5))

    for i, (config_name, metrics) in enumerate(stat_power.items()):
        plt.subplot(n_rows, n_cols, i + 1)

        x = metrics["ca_rmsds"]
        y = metrics["matrix_rmsds"]

        # Use hexbin for dense scatter plots
        hb = plt.hexbin(x, y, gridsize=30, cmap="Blues", mincnt=1)
        plt.colorbar(hb, label="Count")

        # Add regression line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        # Get min and max x values for line
        x_min, x_max = min(x), max(x)
        x_line = np.linspace(x_min, x_max, 100)
        plt.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

        plt.title(
            f"{config_name}\nPearson r: {metrics['pearson_corr']:.3f}, p: {metrics['pearson_p']:.3e}"
        )
        plt.xlabel("CA RMSD (Å)")
        plt.ylabel("Adjacency Matrix RMSD")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "config_scatter_plots.png"), dpi=300)
    plt.close()

    # 2. Combined plot showing all configs with regression lines
    plt.figure(figsize=(12, 10))

    # First plot a 2D density plot for each config
    all_x = []
    all_y = []
    for config_name, metrics in stat_power.items():
        x = metrics["ca_rmsds"]
        y = metrics["matrix_rmsds"]
        all_x.extend(x)
        all_y.extend(y)

        # Plot regression line for each config
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        # Get min and max x values for line
        x_min, x_max = min(x), max(x)
        x_line = np.linspace(x_min, x_max, 100)
        plt.plot(
            x_line,
            p(x_line),
            linewidth=3,
            alpha=0.7,
            label=f"{config_name} (r={metrics['pearson_corr']:.3f})",
        )

    # Create overall density plot
    try:
        # Create KDE density plot
        from scipy.stats import gaussian_kde

        # Calculate point density
        xy = np.vstack([all_x, all_y])
        z = gaussian_kde(xy)(xy)

        # Sort points by density
        idx = z.argsort()
        x_sorted, y_sorted, z_sorted = np.array(all_x)[idx], np.array(all_y)[idx], z[idx]

        plt.scatter(x_sorted, y_sorted, c=z_sorted, s=30, alpha=0.5, cmap="Greys", edgecolor="none")
    except:
        # Fallback to hexbin if KDE fails
        hb = plt.hexbin(all_x, all_y, gridsize=40, cmap="Greys", alpha=0.5, mincnt=1)
        plt.colorbar(hb, label="Count")

    plt.xlabel("CA RMSD (Å)")
    plt.ylabel("Adjacency Matrix RMSD")
    plt.title("Comparison of H-Bond Network Configurations")
    plt.grid(alpha=0.3)
    plt.legend(loc="best", frameon=True, facecolor="white", edgecolor="gray", framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_scatter_plot.png"), dpi=300)
    plt.close()

    # 3. Bar chart comparing statistical metrics
    metrics_to_plot = [
        "pearson_corr",
        "spearman_corr",
        "range_ratio",
        "normalized_std_ratio",
        "sensitivity_score",
    ]
    metric_labels = [
        "Pearson Correlation",
        "Spearman Correlation",
        "Range Ratio",
        "Std Ratio",
        "Sensitivity Score",
    ]

    plt.figure(figsize=(12, 8))

    configs = list(stat_power.keys())
    x = np.arange(len(configs))
    width = 0.15

    for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [metrics[metric] for metrics in stat_power.values()]
        plt.bar(x + (i - 2) * width, values, width, label=label)

    plt.xlabel("Configuration")
    plt.ylabel("Value")
    plt.title("Statistical Metrics by Configuration")
    plt.xticks(x, configs, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_metrics.png"), dpi=300)
    plt.close()

    # 4. Ranking table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    # Compute overall score and rank configs
    scores = {}
    for config, metrics in stat_power.items():
        scores[config] = metrics["sensitivity_score"]

    ranked_configs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Create table data
    table_data = []
    table_data.append(
        [
            "Rank",
            "Configuration",
            "Sens. Score",
            "Pearson r",
            "Spearman r",
            "Range Ratio",
            "Std Ratio",
        ]
    )

    for rank, (config, score) in enumerate(ranked_configs, 1):
        metrics = stat_power[config]
        table_data.append(
            [
                str(rank),
                config,
                f"{metrics['sensitivity_score']:.3f}",
                f"{metrics['pearson_corr']:.3f}",
                f"{metrics['spearman_corr']:.3f}",
                f"{metrics['range_ratio']:.3f}",
                f"{metrics['normalized_std_ratio']:.3f}",
            ]
        )

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.savefig(os.path.join(output_dir, "config_ranking.png"), dpi=300)
    plt.close()


def test_config_comparison():
    """
    Test function that compares different H-bond network configurations.
    """
    # Define test data paths
    topology_path = "./tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "./tests/inst/clean/BPTI/BPTI_sampled_500.xtc"

    # Create test output directory
    test_dir = "tests/_plots/config_comparison"
    os.makedirs(test_dir, exist_ok=True)

    print("\nTesting H-bond network configuration comparison...")
    print("-" * 80)

    # Define configurations to compare
    configs = {
        "netHDX_standard": NetHDXConfig(),
        "BV_standard": BV_model_Config(),
    }

    try:
        # Load universe
        universe = mda.Universe(topology_path, trajectory_path)

        # Run comparison
        results = compare_adjacency_matrices(
            universe=universe,
            configs=configs,
            max_frames=100,  # Use 50 frames to limit computational cost
            output_dir=test_dir,
        )

        # Print summary of results
        print("\nResults Summary:")
        print("-" * 50)
        print(f"{'Configuration':<15} {'Sensitivity':<12} {'Pearson r':<12} {'Spearman r':<12}")
        print("-" * 50)

        # Sort configs by sensitivity score
        sorted_configs = sorted(
            [(config, metrics["sensitivity_score"]) for config, metrics in results.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        for config, score in sorted_configs:
            metrics = results[config]
            print(
                f"{config:<15} {metrics['sensitivity_score']:.4f}      {metrics['pearson_corr']:.4f}      {metrics['spearman_corr']:.4f}"
            )

        print("\nTest completed successfully!")
        print(f"Results saved to: {test_dir}")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_config_comparison()
