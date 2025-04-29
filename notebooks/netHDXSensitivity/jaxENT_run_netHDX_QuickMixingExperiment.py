"""
This script is used to run a mixing experiment using netHDX and BV models


The script takes the following arguments:

- reference_structure_1
- reference_structure_2
- structure_names (compact vs extended)
- num intervals (min 2)
- network config names (definitions are described inside the script, just use this to select the config)
- output directory (defaults to creating a folder inside current directory of the script)
- dropout_rate (default 0.5)
- num_replicates (default 10)
- seed (default 42)

reference_structure_1 = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
reference_structure_2 = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"


For each structure and config description, the script will compute the adjancency matrices.
The mixing experiment takes the two adjacency matrices and computes the mixed matrix.
The mixed matrix then has its MSD compared to the first reference structure. This is repeated over the number intervals between 0 and 100%, inclusive.
Multiple replicates are run with random but deterministic dropout to compute confidence intervals.


"""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis import Universe
from tqdm import tqdm

mpl.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 12,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "font.size": 24,  # default for all text (fallback)
    }
)

from jaxent.models.config import BV_model_Config, NetHDXConfig
from jaxent.models.func.netHDX import calculate_trajectory_hbonds, calculate_trajectory_hbonds_BV

reference_structure_1 = "tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
reference_structure_2 = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
current_dir = os.path.dirname(os.path.abspath(__file__))


def parse_arguments():
    """Parse command line arguments for the mixing experiment."""
    parser = argparse.ArgumentParser(description="Run mixing experiment with netHDX and BV models.")
    parser.add_argument(
        "--reference_structure_1",
        # required=True,
        help="Path to first reference structure",
        default=reference_structure_1,
    )
    parser.add_argument(
        "--reference_structure_2",
        # required=True,
        help="Path to second reference structure",
        default=reference_structure_2,
    )
    parser.add_argument(
        "--structure_names",
        nargs=2,
        default=["compact", "extended"],
        help="Names of structures (default: compact extended)",
    )
    parser.add_argument(
        "--num_intervals", type=int, default=10, help="Number of intervals for mixing (min 2)"
    )
    parser.add_argument(
        "--network_config_names",
        nargs="+",
        default=[
            "netHDX_standard",
            "BV_standard",
            "BV_Oxy",
            "BV_Heavy",
            "netHDX_1",
            "netHDX_3",
            "netHDX_5",
            "netHDX_7",
            "netHDX_9",
        ],
        help="Network configuration names",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join(current_dir, "mixing_experiment_results"),
        help="Output directory",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate for matrix comparison (default: 0.5)",
    )
    parser.add_argument(
        "--num_replicates", type=int, default=10, help="Number of dropout replicates (default: 10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic dropout (default: 42)"
    )
    parser.add_argument(
        "--atom_selection",
        choices=["heavy", "CA"],
        default="heavy",
        help="Atom selection for pairwise distance RMSD calculation (heavy or CA atoms)",
    )
    parser.add_argument(
        "--plot_heatmaps",
        default=True,
        help="Plot heatmaps of adjacency matrices",
    )
    parser.add_argument(
        "--plot_all_mixing_ratios",
        default=True,
        help="Plot heatmaps for all mixing ratios (default: only key ratios)",
    )
    parser.add_argument(
        "--heatmap_colormap",
        default="magma",
        help="Colormap to use for heatmaps (default: viridis)",
    )
    return parser.parse_args()


def calculate_matrix_rmsd(matrix1, matrix2):
    """Calculate RMSD between two adjacency matrices."""
    return np.sqrt(np.mean((matrix1 - matrix2) ** 2))


def calculate_matrix_rmsd_with_dropout(matrix1, matrix2, dropout_rate=0.5, seed=None):
    """Calculate RMSD between two adjacency matrices with dropout."""
    if seed is not None:
        np.random.seed(seed)

    # Create a mask for dropout (True means keep, False means dropout)
    mask = np.random.rand(*matrix1.shape) > dropout_rate

    # round the mask to ensure it is binary
    mask = np.round(mask).astype(int)

    # Calculate squared differences
    squared_diff = (matrix1 - matrix2) ** 2

    # Apply mask and calculate mean of non-dropped elements
    if np.sum(mask) > 0:  # Make sure we have some elements left
        return np.sqrt(np.sum(squared_diff * mask) / np.sum(mask))
    else:
        return 0.0  # Default if all elements were dropped


def mix_matrices(matrix1, matrix2, mixing_ratio):
    """Mix two matrices according to the mixing ratio."""
    return (1 - mixing_ratio) * matrix1 + mixing_ratio * matrix2


def get_config(config_name):
    """Get configuration object based on name."""
    if config_name == "netHDX_standard":
        return NetHDXConfig()
    elif config_name == "BV_standard":
        return BV_model_Config()
    elif config_name == "BV_Oxy":
        config = BV_model_Config()
        config.bv_bc = np.array([0.0])
        return config

    elif config_name == "BV_Heavy":
        config = BV_model_Config()
        config.bv_bh = np.array([0.0])
        return config

    elif config_name == "netHDX_1":
        config = NetHDXConfig(distance_cutoff=[2.6], angle_cutoff=[0.0])
        config.residue_ignore = (-2, 2)
        return config
    # elif config_name == "netHDX_2":
    #     config = NetHDXConfig(distance_cutoff=[2.6, 2.7], angle_cutoff=[0.0, 0.0])
    #     return config
    elif config_name == "netHDX_3":
        config = NetHDXConfig(distance_cutoff=[2.6, 2.7, 2.8], angle_cutoff=[0.0, 0.0, 0.0])
        config.residue_ignore = (-2, 2)

        return config
    # elif config_name == "netHDX_4":
    #     config = NetHDXConfig(
    #         distance_cutoff=[2.6, 2.7, 2.8, 2.9], angle_cutoff=[0.0, 0.0, 0.0, 0.0]
    #     )
    #     return config
    elif config_name == "netHDX_5":
        config = NetHDXConfig(
            distance_cutoff=[2.6, 2.7, 2.8, 2.9, 3.1], angle_cutoff=[0.0, 0.0, 0.0, 0.0, 0.0]
        )
        config.residue_ignore = (-2, 2)

        return config

    # elif config_name == "netHDX_6":
    #     config = NetHDXConfig(
    #         distance_cutoff=[2.6, 2.7, 2.8, 2.9, 3.1, 3.3], angle_cutoff=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     )
    #     return config

    elif config_name == "netHDX_7":
        config = NetHDXConfig(
            distance_cutoff=[2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6],
            angle_cutoff=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        config.residue_ignore = (-2, 2)

        return config

    elif config_name == "netHDX_9":
        config = NetHDXConfig(
            distance_cutoff=[2.6, 2.7, 2.8, 2.9, 3.1, 3.3, 3.6, 4.2, 5.2],
            angle_cutoff=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        config.residue_ignore = (-2, 2)

        return config

    else:
        raise ValueError(f"Unknown configuration: {config_name}")


def compute_adjacency_matrix(structure_path, config):
    """Compute adjacency matrix for a structure using the given configuration."""
    universe = Universe(structure_path)

    if isinstance(config, NetHDXConfig):
        adjacency_matrices = calculate_trajectory_hbonds(universe, config)
    elif isinstance(config, BV_model_Config):
        adjacency_matrices = calculate_trajectory_hbonds_BV(universe, config)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")

    # Return the adjacency matrix for the first frame
    return adjacency_matrices[0]


def calculate_pairwise_distance_rmsd(universe1, universe2, selection="heavy"):
    """
    Calculate RMSD of pairwise distances between two structures.

    Parameters:
    -----------
    universe1, universe2 : MDAnalysis.Universe
        The two structure universes to compare
    selection : str
        'heavy' for heavy atoms, 'CA' for alpha carbons

    Returns:
    --------
    float
        RMSD of pairwise distances
    """
    # Select atoms based on the selection parameter
    if selection == "heavy":
        atoms1 = universe1.select_atoms("not name H*")
        atoms2 = universe2.select_atoms("not name H*")
    elif selection == "CA":
        atoms1 = universe1.select_atoms("name CA")
        atoms2 = universe2.select_atoms("name CA")
    else:
        raise ValueError(f"Unsupported selection: {selection}. Use 'heavy' or 'CA'.")

    # Check if number of atoms match
    if len(atoms1) != len(atoms2):
        raise ValueError(f"Number of selected atoms don't match: {len(atoms1)} vs {len(atoms2)}")

    # Calculate pairwise distances for both structures
    positions1 = atoms1.positions
    positions2 = atoms2.positions

    # Calculate pairwise distance matrices
    n_atoms = len(positions1)
    dist_matrix1 = np.zeros((n_atoms, n_atoms))
    dist_matrix2 = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist1 = np.linalg.norm(positions1[i] - positions1[j])
            dist2 = np.linalg.norm(positions2[i] - positions2[j])
            dist_matrix1[i, j] = dist_matrix1[j, i] = dist1
            dist_matrix2[i, j] = dist_matrix2[j, i] = dist2

    # Calculate RMSD between the distance matrices
    return np.sqrt(np.mean((dist_matrix1 - dist_matrix2) ** 2))


def calculate_pairwise_distance_matrix(universe, selection="heavy"):
    """
    Calculate matrix of pairwise distances between atoms in a structure.

    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The structure universe
    selection : str
        'heavy' for heavy atoms, 'CA' for alpha carbons

    Returns:
    --------
    numpy.ndarray
        Matrix of pairwise distances
    atoms : MDAnalysis.AtomGroup
        The selected atoms
    """
    # Select atoms based on the selection parameter
    if selection == "heavy":
        atoms = universe.select_atoms("not name H*")
    elif selection == "CA":
        atoms = universe.select_atoms("name CA")
    else:
        raise ValueError(f"Unsupported selection: {selection}. Use 'heavy' or 'CA'.")

    # Calculate pairwise distances
    positions = atoms.positions
    n_atoms = len(positions)
    dist_matrix = np.zeros((n_atoms, n_atoms))

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    return dist_matrix, atoms


def mix_distance_matrices(dist_matrix1, dist_matrix2, mixing_ratio):
    """
    Mix two distance matrices according to the mixing ratio.

    Parameters:
    -----------
    dist_matrix1, dist_matrix2 : numpy.ndarray
        The two distance matrices to mix
    mixing_ratio : float
        Ratio for mixing (0.0 = all matrix1, 1.0 = all matrix2)

    Returns:
    --------
    numpy.ndarray
        The mixed distance matrix
    """
    return (1 - mixing_ratio) * dist_matrix1 + mixing_ratio * dist_matrix2


def calculate_distance_matrix_rmsd(matrix1, matrix2):
    """
    Calculate RMSD between two distance matrices.

    Parameters:
    -----------
    matrix1, matrix2 : numpy.ndarray
        The two distance matrices to compare

    Returns:
    --------
    float
        RMSD between the matrices
    """
    return np.sqrt(np.mean((matrix1 - matrix2) ** 2))


# Then modify run_mixing_experiment() function:
def run_mixing_experiment(args):
    """Run the mixing experiment."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get configurations
    configs = {name: get_config(name) for name in args.network_config_names}

    # Load the reference structures for structural RMSD calculation
    universe1 = Universe(args.reference_structure_1)
    universe2 = Universe(args.reference_structure_2)

    # Calculate pairwise distance matrices for both structures
    print(f"Calculating pairwise distance matrices ({args.atom_selection} atoms)...")
    dist_matrix1, atoms1 = calculate_pairwise_distance_matrix(universe1, args.atom_selection)
    dist_matrix2, atoms2 = calculate_pairwise_distance_matrix(universe2, args.atom_selection)

    # Verify atoms match between structures
    if len(atoms1) != len(atoms2):
        raise ValueError(f"Number of selected atoms don't match: {len(atoms1)} vs {len(atoms2)}")

    # Calculate the baseline RMSD between reference structures
    structure_distance_rmsd = calculate_distance_matrix_rmsd(dist_matrix1, dist_matrix2)
    print(
        f"Pairwise distance RMSD between reference structures ({args.atom_selection} atoms): {structure_distance_rmsd:.4f}"
    )

    # Generate mixing ratios
    num_points = max(2, args.num_intervals + 1)  # Ensure at least 2 points (0% and 100%)
    mixing_ratios = np.linspace(0, 1, num_points)

    # Calculate the structural RMSDs for each mixing ratio
    structure_rmsds = []
    for ratio in mixing_ratios:
        # Mix the distance matrices
        mixed_dist_matrix = mix_distance_matrices(dist_matrix1, dist_matrix2, ratio)
        # Calculate RMSD to reference structure 1
        rmsd = calculate_distance_matrix_rmsd(mixed_dist_matrix, dist_matrix1)
        structure_rmsds.append(rmsd)

    # Results and adjacency matrices storage
    all_results = {}
    all_adjacency_matrices = {}

    for config_name, config in configs.items():
        print(f"Processing configuration: {config_name}")

        # Compute adjacency matrices for both structures
        print(f"  Computing adjacency matrix for {args.structure_names[0]}...")
        adjacency_1 = compute_adjacency_matrix(args.reference_structure_1, config)

        print(f"  Computing adjacency matrix for {args.structure_names[1]}...")
        adjacency_2 = compute_adjacency_matrix(args.reference_structure_2, config)

        # Calculate RMSD for each mixing ratio with multiple replicates
        rmsd_values = []
        rmsd_stdevs = []
        rmsd_ci_lower = []
        rmsd_ci_upper = []

        # Store mixed matrices for visualization if needed
        mixed_matrices = []

        print(
            f"  Calculating RMSD for {len(mixing_ratios)} mixing ratios with {args.num_replicates} replicates..."
        )
        for ratio in tqdm(mixing_ratios):
            mixed_matrix = mix_matrices(adjacency_1, adjacency_2, ratio)

            # Store the matrix if we'll be plotting heatmaps
            if args.plot_heatmaps:
                mixed_matrices.append(mixed_matrix.copy())

            # Run multiple replicates with different dropout patterns
            replicate_rmsds = []
            for rep in range(args.num_replicates):
                # Use a deterministic but different seed for each replicate
                rep_seed = args.seed + rep if args.seed is not None else None
                rmsd = calculate_matrix_rmsd_with_dropout(
                    mixed_matrix, adjacency_1, dropout_rate=args.dropout_rate, seed=rep_seed
                )
                replicate_rmsds.append(rmsd)

            # Calculate statistics
            mean_rmsd = np.mean(replicate_rmsds)
            std_rmsd = np.std(replicate_rmsds)

            # Calculate 95% confidence interval (1.96 * std for approximately 95% CI)
            ci_lower = mean_rmsd - 1.96 * std_rmsd / np.sqrt(args.num_replicates)
            ci_upper = mean_rmsd + 1.96 * std_rmsd / np.sqrt(args.num_replicates)

            rmsd_values.append(mean_rmsd)
            rmsd_stdevs.append(std_rmsd)
            rmsd_ci_lower.append(ci_lower)
            rmsd_ci_upper.append(ci_upper)

        # Store results and matrices
        all_results[config_name] = {
            "mixing_ratios": mixing_ratios,
            "rmsd_values": rmsd_values,
            "rmsd_stdevs": rmsd_stdevs,
            "rmsd_ci_lower": rmsd_ci_lower,
            "rmsd_ci_upper": rmsd_ci_upper,
            "structure_rmsds": structure_rmsds,  # Add the structure RMSDs for each mixing ratio
        }

        if args.plot_heatmaps:
            all_adjacency_matrices[config_name] = mixed_matrices

    # Visualize results
    visualize_results(all_results, args)

    # Create the new plot comparing matrix RMSD to structural RMSD
    plot_matrix_vs_structure_rmsd(all_results, args)

    # Plot heatmaps of adjacency matrices if requested
    if args.plot_heatmaps:
        print("Plotting adjacency matrix heatmaps...")
        plot_adjacency_heatmaps(
            all_results,
            all_adjacency_matrices,
            args,
            plot_all=args.plot_all_mixing_ratios,
            colormap=args.heatmap_colormap,
        )

    return all_results


def plot_matrix_vs_structure_rmsd(results, args):
    """
    Plot matrix RMSD against pairwise distance RMSD for mixed structures.

    Parameters:
    -----------
    results : dict
        Results from the mixing experiment
    args : argparse.Namespace
        Command line arguments
    """
    plt.figure(figsize=(10, 6))

    for config_name, data in results.items():
        # Plot matrix RMSD vs structural RMSD
        plt.plot(data["structure_rmsds"], data["rmsd_values"], marker="o", label=config_name)

        # Add error bars/confidence intervals
        plt.fill_between(
            data["structure_rmsds"], data["rmsd_ci_lower"], data["rmsd_ci_upper"], alpha=0.2
        )

    plt.xlabel(f"Mixed Distance Matrix RMSD vs Reference 1 ({args.atom_selection} atoms)")
    plt.ylabel("Adjacency Matrix RMSD")
    plt.title(f"Matrix RMSD vs. Structural RMSD ({args.atom_selection} atoms)")
    plt.grid(alpha=0.3)
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, f"matrix_vs_structure_rmsd_{args.atom_selection}.png"),
        dpi=300,
    )
    plt.close()


def visualize_results(results, args):
    """Create visualizations for mixing experiment results with confidence intervals."""
    # Plot RMSD vs Mixing Ratio for each configuration with confidence intervals
    plt.figure(figsize=(10, 6))

    for config_name, data in results.items():
        # Convert to percentage for x-axis
        x = data["mixing_ratios"] * 100
        y = data["rmsd_values"]
        ci_lower = data["rmsd_ci_lower"]
        ci_upper = data["rmsd_ci_upper"]

        # Plot mean line
        line = plt.plot(x, y, marker="o", label=config_name)
        color = line[0].get_color()

        # Plot confidence interval as a filled area
        plt.fill_between(x, ci_lower, ci_upper, color=color, alpha=0.2)

    plt.xlabel(f"Mixing Ratio (% of {args.structure_names[1]})")
    plt.ylabel("Matrix RMSD")
    plt.title(
        f"Mixing Experiment: {args.structure_names[0]} to {args.structure_names[1]} (with {args.dropout_rate:.1f} dropout)"
    )
    plt.grid(alpha=0.3)
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "mixing_experiment_rmsd_with_ci.png"), dpi=300)
    plt.close()

    # Create a bar chart for fully mixed (100%) condition showing confidence intervals
    plt.figure(figsize=(12, 6))

    # Find index of fully mixed condition (mixing_ratio = 1.0)
    fully_mixed_idx = list(results[list(results.keys())[0]]["mixing_ratios"]).index(1.0)

    # Prepare data for bar chart
    config_names = list(results.keys())
    # Calculate confidence interval widths
    ci_widths = [
        results[cfg]["rmsd_ci_upper"][fully_mixed_idx]
        - results[cfg]["rmsd_ci_lower"][fully_mixed_idx]
        for cfg in config_names
    ]

    # Create bar chart with confidence interval widths
    x_pos = np.arange(len(config_names))
    plt.bar(x_pos, ci_widths, align="center", alpha=0.7)

    # Add labels and title
    plt.xticks(x_pos, config_names, rotation=45, ha="right")
    plt.ylabel("Confidence Interval Width")
    plt.title(
        f"95% Confidence Interval Width at Fully Mixed State ({args.structure_names[0]} to {args.structure_names[1]})"
    )
    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(args.output_dir, "fully_mixed_confidence_intervals.png"), dpi=300)
    plt.close()

    # Create a table of results with confidence intervals
    fig, ax = plt.subplots(figsize=(14, len(results) + 2))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = [
        ["Configuration"]
        + [f"{ratio:.0f}%" for ratio in results[list(results.keys())[0]]["mixing_ratios"] * 100]
    ]

    for config_name, data in results.items():
        row = [config_name]
        for i in range(len(data["rmsd_values"])):
            rmsd = data["rmsd_values"][i]
            ci_lower = data["rmsd_ci_lower"][i]
            ci_upper = data["rmsd_ci_upper"][i]
            cell_text = f"{rmsd:.4f} ({ci_lower:.4f}-{ci_upper:.4f})"
            row.append(cell_text)
        table_data.append(row)

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.savefig(os.path.join(args.output_dir, "mixing_experiment_table_with_ci.png"), dpi=300)
    plt.close()

    # Save numerical results to CSV
    with open(os.path.join(args.output_dir, "mixing_experiment_results_with_ci.csv"), "w") as f:
        # Write header
        f.write("Mixing Ratio")
        for config_name in results:
            f.write(f",{config_name}_mean,{config_name}_ci_lower,{config_name}_ci_upper")
        f.write("\n")

        # Write data
        for i, ratio in enumerate(results[list(results.keys())[0]]["mixing_ratios"]):
            f.write(f"{ratio:.4f}")
            for config_name in results:
                rmsd = results[config_name]["rmsd_values"][i]
                ci_lower = results[config_name]["rmsd_ci_lower"][i]
                ci_upper = results[config_name]["rmsd_ci_upper"][i]
                f.write(f",{rmsd:.6f},{ci_lower:.6f},{ci_upper:.6f}")
            f.write("\n")


def plot_adjacency_heatmaps(results, adjacency_matrices, args, plot_all=False, colormap="viridis"):
    """
    Plot heatmaps of mixed adjacency matrices at each mixing ratio.

    Parameters:
    -----------
    results : dict
        Results from the mixing experiment
    adjacency_matrices : dict
        Dictionary containing adjacency matrices for different configurations and mixing ratios
    args : argparse.Namespace
        Command line arguments
    plot_all : bool, optional
        Whether to plot all mixing ratios or just key ones (0%, 25%, 50%, 75%, 100%)
    colormap : str, optional
        Matplotlib colormap to use for the heatmaps
    """
    # Create a directory for heatmaps
    heatmap_dir = os.path.join(args.output_dir, "adjacency_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # For each configuration
    for config_name, matrices in adjacency_matrices.items():
        print(f"  Generating heatmaps for {config_name}...")

        # Create a subdirectory for this configuration
        config_dir = os.path.join(heatmap_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)

        # Get mixing ratios
        mixing_ratios = results[config_name]["mixing_ratios"]

        # Determine which mixing ratios to plot
        if plot_all:
            # Plot all mixing ratios
            indices_to_plot = range(len(mixing_ratios))
        else:
            # Plot only key mixing ratios (0%, ~25%, ~50%, ~75%, 100%)
            key_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
            indices_to_plot = [np.abs(mixing_ratios - ratio).argmin() for ratio in key_ratios]

        # Plot individual heatmaps for selected mixing ratios
        for i in indices_to_plot:
            ratio = mixing_ratios[i]
            matrix = matrices[i]

            # Create heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(matrix, cmap=colormap, interpolation="nearest")
            plt.colorbar(label="Connection strength")

            # Add labels and title
            mixing_percent = int(ratio * 100)
            plt.title(
                f"{config_name}: {mixing_percent}% {args.structure_names[1]} / {100 - mixing_percent}% {args.structure_names[0]}"
            )
            plt.xlabel("Residue index")
            plt.ylabel("Residue index")

            # Save figure
            filename = f"{config_name}_mix_{mixing_percent:03d}.png"
            plt.savefig(os.path.join(config_dir, filename), dpi=300)
            plt.close()

        # Create a figure showing key mixing ratios side by side
        fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(4 * len(indices_to_plot), 4))

        if len(indices_to_plot) == 1:
            axes = [axes]  # Handle case with only one subplot

        for j, idx in enumerate(indices_to_plot):
            ratio = mixing_ratios[idx]
            matrix = matrices[idx]
            mixing_percent = int(ratio * 100)

            # Plot on the corresponding axis
            im = axes[j].imshow(matrix, cmap=colormap, interpolation="nearest")
            axes[j].set_title(f"{mixing_percent}% {args.structure_names[1]}")
            axes[j].set_xlabel("Residue index")

            # Only add y-label for the first plot
            if j == 0:
                axes[j].set_ylabel("Residue index")

        # Add a colorbar that applies to all subplots
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Connection strength")

        plt.suptitle(f"{config_name} Adjacency Matrices at Different Mixing Ratios")
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])

        # Save the side-by-side comparison
        plt.savefig(os.path.join(config_dir, f"{config_name}_key_comparisons.png"), dpi=300)
        plt.close()

        # Create difference plots (difference from 0% mixing)
        base_matrix = matrices[0]  # 0% mixing (all structure 1)

        for i in indices_to_plot[1:]:  # Skip the first one (0% mixing)
            ratio = mixing_ratios[i]
            matrix = matrices[i]
            mixing_percent = int(ratio * 100)

            # Calculate difference matrix
            diff_matrix = matrix - base_matrix

            # Create heatmap of differences
            plt.figure(figsize=(10, 8))
            # Use a diverging colormap for differences
            plt.imshow(diff_matrix, cmap="RdBu_r", interpolation="nearest")
            plt.colorbar(label="Connection strength difference")

            # Add labels and title
            plt.title(f"{config_name}: Difference {mixing_percent}% - 0% mixing")
            plt.xlabel("Residue index")
            plt.ylabel("Residue index")

            # Save figure
            filename = f"{config_name}_diff_{mixing_percent:03d}.png"
            plt.savefig(os.path.join(config_dir, filename), dpi=300)
            plt.close()


def main():
    """Main function to run the mixing experiment."""
    args = parse_arguments()

    # Validate inputs
    if args.num_intervals < 2:
        print("Warning: num_intervals should be at least 2. Setting to 2.")
        args.num_intervals = 2

    # Validate dropout-related inputs
    if args.dropout_rate < 0 or args.dropout_rate >= 1:
        print("Warning: dropout_rate should be between 0 and 1. Setting to 0.5.")
        args.dropout_rate = 0.5

    if args.num_replicates < 1:
        print("Warning: num_replicates should be at least 1. Setting to 10.")
        args.num_replicates = 10

    print("Running mixing experiment with:")
    print(f"  Reference 1: {args.reference_structure_1} ({args.structure_names[0]})")
    print(f"  Reference 2: {args.reference_structure_2} ({args.structure_names[1]})")
    print(f"  Intervals: {args.num_intervals}")
    print(f"  Configurations: {', '.join(args.network_config_names)}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Number of replicates: {args.num_replicates}")
    print(f"  Random seed: {args.seed}")
    print(f"  Atom selection for structural RMSD: {args.atom_selection}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Plot heatmaps: {args.plot_heatmaps}")
    if args.plot_heatmaps:
        print(f"  Plot all mixing ratios: {args.plot_all_mixing_ratios}")
        print(f"  Heatmap colormap: {args.heatmap_colormap}")

    # Run the mixing experiment
    results = run_mixing_experiment(args)

    print(f"Mixing experiment completed. Results saved to {args.output_dir}")

    print(f"Mixing experiment completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
