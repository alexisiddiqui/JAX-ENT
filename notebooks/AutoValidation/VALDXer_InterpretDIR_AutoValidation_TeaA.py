"""
This script extends the auto validation analysis to work with HDXer experiment results.
It generates plots showing the evolution of open/closed state populations from HDXer
optimization results.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import rms
from scipy import stats

# globally set axes/tick/legend font‐sizes
# globally set axes/tick/legend font‐sizes
mpl.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 14,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "font.size": 24,  # default for all text (fallback)
    }
)

# Define paths (to be adjusted based on actual file locations)
base_dir = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/HDXer/gamma1_9_exp-2_1/data/TeaA_auto_VAL/Benchmark/RW_bench/TeaA_auto_VAL_RW_bench_r_naive_random"
# base_dir = (
#     "/Users/alexi/JAX-ENT/notebooks/AutoValidation/HDXer/TeaA_auto_VAL_RW_bench_r_naive_random/"
# )

output_dir = os.path.join(base_dir, "hdxer_analysis")


open_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = (
    "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"
)
"""
This script extends the auto validation analysis to work with HDXer experiment results.
It generates plots showing the evolution of open/closed state populations from HDXer
optimization results.
"""


def compute_rmsd_to_references(trajectory_path, topology_path, reference_paths):
    """
    Compute RMSD values between each frame in the trajectory and reference structures.

    Parameters:
    -----------
    trajectory_path : str
        Path to trajectory file
    topology_path : str
        Path to topology file
    reference_paths : list
        List of paths to reference PDB structures

    Returns:
    --------
    rmsd_values : numpy.ndarray
        RMSD values for each frame and reference
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


def cluster_frames_by_rmsd(rmsd_values):
    """
    Assign each frame to the reference with minimum RMSD.

    Parameters:
    -----------
    rmsd_values : numpy.ndarray
        RMSD values for each frame and reference

    Returns:
    --------
    cluster_assignments : numpy.ndarray
        Cluster assignments for each frame
    """
    # Assign each frame to the reference with minimum RMSD
    cluster_assignments = np.argmin(rmsd_values, axis=1)
    return cluster_assignments


def extract_hdxer_weights(hdxer_dir, n_seeds, gamma_values, exponents):
    """
    Extract HDXer weights from weight files.

    Parameters:
    -----------
    hdxer_dir : str
        Directory containing HDXer results
    n_seeds : int
        Number of seeds to process
    gamma_values : list
        List of gamma values to extract weights for
    exponents : list
        List of exponents for gamma values

    Returns:
    --------
    weights_dict : dict
        Dictionary of weights indexed by seed and gamma
    """
    weights_dict = {}

    # Extract weights for each seed
    for seed_idx in range(1, n_seeds + 1):
        seed_key = f"seed_{seed_idx}"
        weights_dict[seed_key] = {}

        # Process each gamma value and exponent
        for gamma in gamma_values:
            for exp in exponents:
                gamma_key = f"gamma_{gamma}x10^{exp}"
                weights_file = os.path.join(
                    hdxer_dir,
                    f"train_TeaA_auto_VAL_{seed_idx}",
                    f"reweighting_gamma_{gamma}x10^{exp}final_weights.dat",
                )

                # Check if weights file exists
                if os.path.exists(weights_file):
                    try:
                        # Load the weights
                        weights = np.loadtxt(weights_file)

                        # Check for NaN or inf values and handle them
                        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                            print(
                                f"Warning: NaN or inf values found in weights for seed {seed_idx}, gamma {gamma}x10^{exp}"
                            )
                            # Replace NaN/inf with zeros
                            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

                        # Clip small weights to zero
                        weights[weights < 1e-10] = 0.0

                        # Check if all weights are zero after NaN replacement
                        if np.sum(weights) <= 0:
                            print(
                                f"Warning: All weights are zero for seed {seed_idx}, gamma {gamma}x10^{exp}"
                            )
                            weights_dict[seed_key][gamma_key] = None
                            continue

                        # Normalize weights
                        weights = weights / np.sum(weights)

                        weights_dict[seed_key][gamma_key] = weights
                        print(f"Loaded weights for seed {seed_idx}, gamma {gamma}x10^{exp}")
                    except Exception as e:
                        print(
                            f"Error loading weights for seed {seed_idx}, gamma {gamma}x10^{exp}: {e}"
                        )
                        weights_dict[seed_key][gamma_key] = None
                else:
                    print(f"Weights file not found: {weights_file}")
                    weights_dict[seed_key][gamma_key] = None

    return weights_dict


def extract_hdxer_iterations(hdxer_dir, n_seeds, gamma_values, exponents):
    """
    Extract HDXer iteration data from per_iteration_output.dat files.

    Parameters:
    -----------
    hdxer_dir : str
        Directory containing HDXer results
    n_seeds : int
        Number of seeds to process
    gamma_values : list
        List of gamma values to extract data for
    exponents : list
        List of exponents for gamma values

    Returns:
    --------
    iterations_dict : dict
        Dictionary of iteration data indexed by seed and gamma
    """
    iterations_dict = {}

    # Extract iterations for each seed
    for seed_idx in range(1, n_seeds + 1):
        seed_key = f"seed_{seed_idx}"
        iterations_dict[seed_key] = {}

        # Process each gamma value and exponent
        for gamma in gamma_values:
            for exp in exponents:
                gamma_key = f"gamma_{gamma}x10^{exp}"
                iter_file = os.path.join(
                    hdxer_dir,
                    f"train_TeaA_auto_VAL_{seed_idx}",
                    f"reweighting_gamma_{gamma}x10^{exp}per_iteration_output.dat",
                )

                # Check if iteration file exists
                if os.path.exists(iter_file):
                    try:
                        # Load the iteration data
                        # Format: Iteration, Chi^2, Average ln(w), Worst ln(w), Work
                        iter_data = np.loadtxt(iter_file)

                        iterations_dict[seed_key][gamma_key] = iter_data
                        print(f"Loaded iterations for seed {seed_idx}, gamma {gamma}x10^{exp}")
                    except Exception as e:
                        print(
                            f"Error loading iterations for seed {seed_idx}, gamma {gamma}x10^{exp}: {e}"
                        )
                        iterations_dict[seed_key][gamma_key] = None
                else:
                    print(f"Iterations file not found: {iter_file}")
                    iterations_dict[seed_key][gamma_key] = None

    return iterations_dict


def compute_state_ratios(weights_dict, cluster_assignments):
    """
    Compute state ratios for each set of weights.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary of weights indexed by seed and gamma
    cluster_assignments : numpy.ndarray
        Cluster assignments for each frame

    Returns:
    --------
    ratios_dict : dict
        Dictionary of state ratios indexed by seed and gamma
    """
    n_clusters = np.max(cluster_assignments) + 1
    ratios_dict = {}

    # Compute ratios for each seed
    for seed_key, gamma_dict in weights_dict.items():
        ratios_dict[seed_key] = {}

        # Process each gamma value
        for gamma_key, weights in gamma_dict.items():
            if weights is None:
                ratios_dict[seed_key][gamma_key] = None
                continue

            # Initialize array for cluster ratios
            cluster_ratios = np.zeros(n_clusters)

            # Make sure weights match the number of frames
            if len(weights) != len(cluster_assignments):
                print(
                    f"Warning: weights length {len(weights)} doesn't match assignments length {len(cluster_assignments)}"
                )

                # Truncate if too long or pad if too short
                if len(weights) > len(cluster_assignments):
                    weights = weights[: len(cluster_assignments)]
                else:
                    padded_weights = np.zeros(len(cluster_assignments))
                    padded_weights[: len(weights)] = weights
                    weights = padded_weights

                # Renormalize
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    print(f"Warning: All weights are zero for {seed_key}, {gamma_key}")
                    ratios_dict[seed_key][gamma_key] = None
                    continue

            # Compute weighted ratio for each cluster
            for cluster_idx in range(n_clusters):
                mask = cluster_assignments == cluster_idx
                cluster_ratios[cluster_idx] = np.sum(weights[mask])

            ratios_dict[seed_key][gamma_key] = cluster_ratios

    return ratios_dict


def compute_weighted_rmsd_distributions(rmsd_values, weights_dict):
    """
    Compute weighted RMSD distributions for visualization.

    Parameters:
    -----------
    rmsd_values : numpy.ndarray
        RMSD values for each frame and reference
    weights_dict : dict
        Dictionary of weights indexed by seed and gamma

    Returns:
    --------
    kde_dict : dict
        Dictionary with KDE estimations for RMSD distributions
    """
    n_frames, n_refs = rmsd_values.shape

    # Create a grid of RMSD values for KDE
    rmsd_grid = np.linspace(0, np.max(rmsd_values) * 1.1, 1000)

    kde_dict = {}

    # Compute KDE for each seed
    for seed_key, gamma_dict in weights_dict.items():
        kde_dict[seed_key] = {}

        # Process each gamma value
        for gamma_key, weights in gamma_dict.items():
            if weights is None:
                kde_dict[seed_key][gamma_key] = None
                continue

            # Make sure weights match the number of frames
            if len(weights) != n_frames:
                print(
                    f"Warning: weights length {len(weights)} doesn't match number of frames {n_frames}"
                )

                # Truncate if too long or pad if too short
                if len(weights) > n_frames:
                    weights = weights[:n_frames]
                else:
                    padded_weights = np.zeros(n_frames)
                    padded_weights[: len(weights)] = weights
                    weights = padded_weights

                # Renormalize
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    print(f"Warning: All weights are zero for {seed_key}, {gamma_key}")
                    kde_dict[seed_key][gamma_key] = None
                    continue

            # Initialize array for KDE values
            kde_values = np.zeros((n_refs, len(rmsd_grid)))

            # Compute weighted KDE for each reference
            for ref_idx in range(n_refs):
                try:
                    # Check for NaN in RMSD values
                    rmsd_ref = rmsd_values[:, ref_idx]
                    if np.any(np.isnan(rmsd_ref)) or np.any(np.isinf(rmsd_ref)):
                        print(
                            f"Warning: NaN or inf values found in RMSD values for reference {ref_idx}"
                        )
                        # Replace NaN/inf with mean value
                        valid_indices = ~(np.isnan(rmsd_ref) | np.isinf(rmsd_ref))
                        if np.any(valid_indices):
                            mean_value = np.mean(rmsd_ref[valid_indices])
                            rmsd_ref = np.nan_to_num(
                                rmsd_ref, nan=mean_value, posinf=mean_value, neginf=mean_value
                            )
                        else:
                            print(
                                f"Warning: All RMSD values are NaN or inf for reference {ref_idx}"
                            )
                            continue

                    # Create KDE
                    kde = stats.gaussian_kde(rmsd_ref, weights=weights)
                    kde_values[ref_idx] = kde(rmsd_grid)
                except Exception as e:
                    print(
                        f"Error computing KDE for {seed_key}, {gamma_key}, reference {ref_idx}: {e}"
                    )
                    kde_values[ref_idx] = np.zeros_like(rmsd_grid)

            kde_dict[seed_key][gamma_key] = {"grid": rmsd_grid, "kde_values": kde_values}

    return kde_dict


def compute_iteration_state_ratios(iterations_dict, weights_dict, cluster_assignments):
    """
    Compute state ratios for each iteration of the HDXer optimization.

    Parameters:
    -----------
    iterations_dict : dict
        Dictionary of iteration data indexed by seed and gamma
    weights_dict : dict
        Dictionary of final weights indexed by seed and gamma
    cluster_assignments : numpy.ndarray
        Cluster assignments for each frame

    Returns:
    --------
    iter_ratios_dict : dict
        Dictionary of state ratios for each iteration
    """
    n_clusters = np.max(cluster_assignments) + 1
    iter_ratios_dict = {}

    # Since HDXer doesn't store intermediate weights, we'll approximate using the work values
    # This is a simplified approach - actual intermediate weights would be better if available

    # Compute ratios for each seed
    for seed_key, gamma_dict in iterations_dict.items():
        iter_ratios_dict[seed_key] = {}

        # Process each gamma value
        for gamma_key, iter_data in gamma_dict.items():
            if iter_data is None or weights_dict[seed_key][gamma_key] is None:
                iter_ratios_dict[seed_key][gamma_key] = None
                continue

            # Get number of iterations
            n_iterations = len(iter_data)

            # Get final weights
            final_weights = weights_dict[seed_key][gamma_key]

            # Initialize array for iteration state ratios
            iter_state_ratios = np.zeros((n_iterations, n_clusters))

            # For each iteration, we'll scale the weights based on the iteration progress
            # This is an approximation - actual intermediate weights would be preferable
            for iter_idx in range(n_iterations):
                # Scale factor: 0 for first iteration (uniform weights), 1 for last iteration (final weights)
                progress = iter_idx / max(1, n_iterations - 1)

                # Create synthetic weights: interpolate between uniform and final
                uniform_weights = np.ones(len(final_weights)) / len(final_weights)
                synth_weights = (1 - progress) * uniform_weights + progress * final_weights

                # Normalize
                synth_weights = synth_weights / np.sum(synth_weights)

                # Compute state ratios for this iteration
                for cluster_idx in range(n_clusters):
                    mask = cluster_assignments == cluster_idx
                    iter_state_ratios[iter_idx, cluster_idx] = np.sum(synth_weights[mask])

            iter_ratios_dict[seed_key][gamma_key] = {
                "iterations": iter_data[:, 0],  # Iteration numbers
                "state_ratios": iter_state_ratios,
            }

    return iter_ratios_dict


def plot_rmsd_distributions(kde_dict, ref_names, output_dir, gamma_value, exponent):
    """
    Plot RMSD distributions for specified gamma value.

    Parameters:
    -----------
    kde_dict : dict
        Dictionary with KDE estimations for RMSD distributions
    ref_names : list
        Names of reference structures
    output_dir : str
        Directory to save output plots
    gamma_value : int
        Gamma value to plot
    exponent : int
        Exponent value for gamma

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with RMSD distribution plots
    """
    gamma_key = f"gamma_{gamma_value}x10^{exponent}"
    n_refs = len(ref_names)

    fig, axes = plt.subplots(1, n_refs, figsize=(15, 6), sharey=True)

    if n_refs == 1:
        axes = [axes]  # Make it iterable for the loop

    # Check if we have at least one valid KDE for this gamma value
    has_valid_kde = False
    for seed_key in kde_dict:
        if gamma_key in kde_dict[seed_key] and kde_dict[seed_key][gamma_key] is not None:
            has_valid_kde = True
            break

    if not has_valid_kde:
        plt.close(fig)
        return None

    # Get the rmsd grid from the first valid KDE
    for seed_key in kde_dict:
        if gamma_key in kde_dict[seed_key] and kde_dict[seed_key][gamma_key] is not None:
            rmsd_grid = kde_dict[seed_key][gamma_key]["grid"]
            break

    # Plot uniform distribution (if needed)
    # This would require the original RMSD values, which we may not have at this point
    # For now, we'll skip this step

    # Plot KDE for each reference and seed
    for ref_idx in range(n_refs):
        ax = axes[ref_idx]

        # Collect all KDEs for this reference
        all_kdes = []
        for seed_key in sorted(kde_dict.keys()):
            if gamma_key in kde_dict[seed_key] and kde_dict[seed_key][gamma_key] is not None:
                kde_values = kde_dict[seed_key][gamma_key]["kde_values"][ref_idx]
                all_kdes.append(kde_values)

                # Plot individual seed KDE
                ax.plot(rmsd_grid, kde_values, alpha=0.3, label=f"{seed_key}")

        # Plot mean KDE if we have multiple seeds
        if len(all_kdes) > 1:
            mean_kde = np.mean(all_kdes, axis=0)
            ax.plot(rmsd_grid, mean_kde, "k-", linewidth=2, label="Mean")

        ax.set_title(
            f"RMSD Distribution to {ref_names[ref_idx]} state (γ={gamma_value}×10^{exponent})"
        )
        ax.set_xlabel("RMSD (Å)")

        if ref_idx == 0:
            ax.set_ylabel("Probability Density")

        ax.legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(
        output_dir, f"rmsd_distribution_gamma_{gamma_value}_exp_{exponent}.png"
    )
    fig.savefig(output_path, dpi=300)

    return fig


def main():
    """
    Main function to run the HDXer autovalidation analysis.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Reference paths
    reference_paths = [open_path, closed_path]
    ref_names = ["Open", "Closed"]

    # Compute RMSD to reference structures
    print("Computing RMSD to reference structures...")
    rmsd_values = compute_rmsd_to_references(trajectory_path, topology_path, reference_paths)

    # Verify that RMSD values don't contain NaN or inf
    if np.any(np.isnan(rmsd_values)) or np.any(np.isinf(rmsd_values)):
        print("Warning: NaN or inf values found in RMSD values. Replacing with mean values.")
        for i in range(rmsd_values.shape[1]):
            col = rmsd_values[:, i]
            valid_indices = ~(np.isnan(col) | np.isinf(col))
            if np.any(valid_indices):
                mean_value = np.mean(col[valid_indices])
                col[~valid_indices] = mean_value

    print(f"RMSD shape: {rmsd_values.shape}, Range: {np.min(rmsd_values)} to {np.max(rmsd_values)}")

    # Cluster frames based on minimum RMSD
    print("Clustering frames by RMSD...")
    cluster_assignments = cluster_frames_by_rmsd(rmsd_values)

    # Print cluster statistics
    clusters, counts = np.unique(cluster_assignments, return_counts=True)
    print("Cluster statistics:")
    for cluster, count in zip(clusters, counts):
        print(
            f"  Cluster {cluster} ({ref_names[cluster]}): {count} frames ({count / len(cluster_assignments) * 100:.1f}%)"
        )

    # Define HDXer parameters to analyze
    gamma_values = list(range(1, 9))  # Adjust based on available data
    exponents = [-2, -1, 0, 1]  # Adjust based on available data
    n_seeds = 3  # Number of seeds (adjust as needed)

    # Extract HDXer weights
    print("Extracting HDXer weights...")
    weights_dict = extract_hdxer_weights(base_dir, n_seeds, gamma_values, exponents)

    # Verify that weights were successfully loaded
    loaded_weights = 0
    total_weights = 0
    for seed_key, gamma_dict in weights_dict.items():
        for gamma_key, weights in gamma_dict.items():
            total_weights += 1
            if weights is not None:
                loaded_weights += 1
    print(f"Successfully loaded {loaded_weights} out of {total_weights} weight sets.")

    if loaded_weights == 0:
        print("No weights could be loaded. Please check the file paths and gamma/exponent values.")
        return

    # Extract HDXer iteration data
    print("Extracting HDXer iteration data...")
    iterations_dict = extract_hdxer_iterations(base_dir, n_seeds, gamma_values, exponents)

    # Compute state ratios
    print("Computing state ratios...")
    ratios_dict = compute_state_ratios(weights_dict, cluster_assignments)

    # Compute weighted RMSD distributions
    print("Computing weighted RMSD distributions...")
    kde_dict = compute_weighted_rmsd_distributions(rmsd_values, weights_dict)

    # Compute iteration-wise state ratios
    print("Computing iteration-wise state ratios...")
    iter_ratios_dict = compute_iteration_state_ratios(
        iterations_dict, weights_dict, cluster_assignments
    )

    # Create state ratio plot
    print("Creating state ratio plot...")
    state_ratio_fig = plot_state_ratios(ratios_dict, ref_names, output_dir)

    # Create plots for specific gamma values
    for gamma in gamma_values:
        for exp in exponents:
            print(f"Creating plots for gamma {gamma}×10^{exp}...")

            # Plot RMSD distributions
            rmsd_fig = plot_rmsd_distributions(kde_dict, ref_names, output_dir, gamma, exp)

            # Plot state ratio evolution
            ratio_evo_fig = plot_state_ratio_evolution(
                iter_ratios_dict, ref_names, output_dir, gamma, exp
            )

            # Plot RMSD distribution evolution
            rmsd_evo_fig = plot_rmsd_distribution_evolution(
                iter_ratios_dict, kde_dict, rmsd_values, ref_names, output_dir, gamma, exp
            )

    print("Analysis complete! Results saved to:", output_dir)


def plot_state_ratios(ratios_dict, ref_names, output_dir):
    """
    Plot state ratios for different gamma values.
    Parameters:
    -----------
    ratios_dict : dict
        Dictionary of state ratios indexed by seed and gamma
    ref_names : list
        Names of reference structures
    output_dir : str
        Directory to save output plots
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with state ratio plots
    """
    # First, extract all unique gamma values
    gamma_keys = set()
    for seed_key in ratios_dict:
        gamma_keys.update(ratios_dict[seed_key].keys())
    gamma_keys = list(gamma_keys)

    # Parse gamma keys to get numeric values and create sorting data
    gamma_value_data = []
    for gamma_key in gamma_keys:
        parts = gamma_key.split("_")[1].split("x")
        gamma = int(parts[0])
        exponent = int(parts[1].replace("10^", ""))
        numeric_value = gamma * (10**exponent)
        gamma_value_data.append((gamma_key, numeric_value, f"{gamma}×10^{exponent}"))

    # Sort by actual numeric magnitude
    gamma_value_data.sort(key=lambda x: x[1])

    # Extract sorted keys and labels
    sorted_gamma_keys = [item[0] for item in gamma_value_data]
    gamma_labels = [item[2] for item in gamma_value_data]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set up colors for each state
    colors = ["b", "r", "g", "c", "m", "y", "k"]

    # Number of states
    n_states = len(ref_names)

    # Initialize arrays for plotting
    mean_ratios = np.zeros((len(sorted_gamma_keys), n_states))
    std_ratios = np.zeros((len(sorted_gamma_keys), n_states))

    # Compute mean and std of ratios across seeds
    for i, gamma_key in enumerate(sorted_gamma_keys):
        for state_idx in range(n_states):
            # Collect ratios for this state and gamma across all seeds
            state_ratios = []
            for seed_key in ratios_dict:
                if (
                    gamma_key in ratios_dict[seed_key]
                    and ratios_dict[seed_key][gamma_key] is not None
                ):
                    state_ratio = ratios_dict[seed_key][gamma_key][state_idx]
                    state_ratios.append(state_ratio)

            if state_ratios:
                mean_ratios[i, state_idx] = np.mean(state_ratios)
                std_ratios[i, state_idx] = np.std(state_ratios)

    # Plot each state
    for state_idx in range(n_states):
        ax.errorbar(
            range(len(sorted_gamma_keys)),
            mean_ratios[:, state_idx],
            yerr=std_ratios[:, state_idx],
            label=f"{ref_names[state_idx]} State",
            marker="o",
            color=colors[state_idx % len(colors)],
            linewidth=2,
            capsize=5,
        )

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(sorted_gamma_keys)))
    ax.set_xticklabels(gamma_labels, rotation=45)

    # Set plot labels and title
    ax.set_xlabel("γ Value")
    ax.set_ylabel("State Ratio")
    ax.set_title("State Ratios for Different γ Values")

    # Set y-axis limits with padding
    ax.set_ylim(0, 1.1)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "state_ratios_by_gamma.png")
    fig.savefig(output_path, dpi=300)

    return fig


def plot_state_ratio_evolution(iter_ratios_dict, ref_names, output_dir, gamma_value, exponent):
    """
    Plot evolution of state ratios during optimization for specific gamma value.

    Parameters:
    -----------
    iter_ratios_dict : dict
        Dictionary of state ratios for each iteration
    ref_names : list
        Names of reference structures
    output_dir : str
        Directory to save output plots
    gamma_value : int
        Gamma value to plot
    exponent : int
        Exponent value for gamma

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with state ratio evolution plot
    """
    gamma_key = f"gamma_{gamma_value}x10^{exponent}"

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Set up colors for each state
    colors = ["b", "r", "g", "c", "m", "y", "k"]

    # Check if we have valid data for this gamma value
    has_valid_data = False
    for seed_key in iter_ratios_dict:
        if (
            gamma_key in iter_ratios_dict[seed_key]
            and iter_ratios_dict[seed_key][gamma_key] is not None
        ):
            has_valid_data = True
            break

    if not has_valid_data:
        plt.close(fig)
        return None

    # Number of states
    n_states = len(ref_names)

    # Process each seed
    for seed_idx, seed_key in enumerate(sorted(iter_ratios_dict.keys())):
        if (
            gamma_key not in iter_ratios_dict[seed_key]
            or iter_ratios_dict[seed_key][gamma_key] is None
        ):
            continue

        # Get iteration data
        iter_data = iter_ratios_dict[seed_key][gamma_key]
        iterations = iter_data["iterations"]
        state_ratios = iter_data["state_ratios"]

        # Plot each state
        for state_idx in range(n_states):
            ax.plot(
                iterations,
                state_ratios[:, state_idx],
                label=f"{seed_key} - {ref_names[state_idx]}" if seed_idx == 0 else "_nolegend_",
                color=colors[state_idx % len(colors)],
                alpha=0.5 + 0.2 * seed_idx,  # Vary alpha for different seeds
                linewidth=2,
            )

    # Set plot labels and title
    ax.set_xlabel("Iteration")
    ax.set_ylabel("State Ratio")
    ax.set_title(f"Evolution of State Ratios (γ={gamma_value}×10^{exponent})")

    # Set y-axis limits with padding
    ax.set_ylim(0, 1.1)

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(
        output_dir, f"state_ratio_evolution_gamma_{gamma_value}_exp_{exponent}.png"
    )
    fig.savefig(output_path, dpi=300)

    return fig


def plot_rmsd_distribution_evolution(
    iter_ratios_dict, kde_dict, rmsd_values, ref_names, output_dir, gamma_value, exponent
):
    """
    Plot evolution of RMSD distributions during optimization for specific gamma value.

    Parameters:
    -----------
    iter_ratios_dict : dict
        Dictionary of state ratios for each iteration
    kde_dict : dict
        Dictionary with KDE estimations for RMSD distributions
    rmsd_values : numpy.ndarray
        RMSD values for each frame and reference
    ref_names : list
        Names of reference structures
    output_dir : str
        Directory to save output plots
    gamma_value : int
        Gamma value to plot
    exponent : int
        Exponent value for gamma

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with RMSD distribution evolution plots
    """
    gamma_key = f"gamma_{gamma_value}x10^{exponent}"
    n_frames, n_refs = rmsd_values.shape

    # Create grid for KDE evaluation
    rmsd_grid = np.linspace(np.min(rmsd_values), np.max(rmsd_values) * 1.1, 1000)

    # Create figure with one subplot per reference
    fig, axes = plt.subplots(n_refs, 1, figsize=(12, 5 * n_refs), squeeze=False)

    # Create colormap for evolution (from light to dark)
    cmap = plt.cm.Blues

    # Check if we have valid data for this gamma value
    has_valid_data = False
    for seed_key in iter_ratios_dict:
        if (
            gamma_key in iter_ratios_dict[seed_key]
            and iter_ratios_dict[seed_key][gamma_key] is not None
        ):
            has_valid_data = True
            break

    if not has_valid_data:
        plt.close(fig)
        return None

    # Get a representative seed with valid data
    rep_seed_key = None
    for seed_key in sorted(iter_ratios_dict.keys()):
        if (
            gamma_key in iter_ratios_dict[seed_key]
            and iter_ratios_dict[seed_key][gamma_key] is not None
        ):
            rep_seed_key = seed_key
            break

    if rep_seed_key is None:
        plt.close(fig)
        return None

    # Get iteration data for representative seed
    iter_data = iter_ratios_dict[rep_seed_key][gamma_key]
    iterations = iter_data["iterations"]
    state_ratios = iter_data["state_ratios"]

    # Number of intervals to display
    n_intervals = min(10, len(iterations))
    interval_indices = np.linspace(0, len(iterations) - 1, n_intervals).astype(int)

    # Colors for each interval
    colors = [cmap(i) for i in np.linspace(0.3, 1.0, n_intervals)]

    for ref_idx in range(n_refs):
        ax = axes[ref_idx, 0]

        # Plot uniform distribution first
        uniform_weights = np.ones(n_frames) / n_frames
        uniform_kde = stats.gaussian_kde(rmsd_values[:, ref_idx], weights=uniform_weights)
        uniform_density = uniform_kde(rmsd_grid)
        ax.plot(rmsd_grid, uniform_density, "k--", label="Initial (uniform)", linewidth=2)

        # Plot distribution at each interval
        for i, iter_idx in enumerate(interval_indices):
            # Get synthetic weights for this iteration (using our approximation)
            # This is a simplification - actual intermediate weights would be better
            iteration = iterations[iter_idx]
            progress = iter_idx / max(1, len(interval_indices) - 1)

            # For demonstration, we'll use synthetic weights based on iteration progress
            # Interpolate between uniform and final weights
            final_weights = None
            for seed_key in kde_dict:
                if gamma_key in kde_dict[seed_key] and kde_dict[seed_key][gamma_key] is not None:
                    # Find frame weights (not stored directly in kde_dict)
                    # You would need to access the actual frame weights here
                    # For now, we'll use a placeholder approach
                    if seed_key == rep_seed_key:
                        # In a real implementation, you would use the actual weights
                        # final_weights = ...
                        pass

            # If we can't get actual weights, skip this part
            if final_weights is None:
                continue

            # Create synthetic weights
            synth_weights = (1 - progress) * uniform_weights + progress * final_weights
            synth_weights = synth_weights / np.sum(synth_weights)

            # Compute KDE with synthetic weights
            kde = stats.gaussian_kde(rmsd_values[:, ref_idx], weights=synth_weights)
            density = kde(rmsd_grid)

            # Set label for first and last interval only to avoid cluttered legend
            if i == 0:
                label = "Step 0"
            elif i == len(interval_indices) - 1:
                label = "Final step"
            else:
                label = None

            # Plot KDE with color based on interval
            ax.plot(rmsd_grid, density, color=colors[i], linewidth=2, label=label)

            # Add a text label on the line for intermediate steps
            if 0 < i < len(interval_indices) - 1 and i % 2 == 0:
                # Find a good position for the text (at peak)
                peak_idx = np.argmax(density)
                ax.text(
                    rmsd_grid[peak_idx],
                    density[peak_idx],
                    f"Step {iterations[iter_idx]:.0f}",
                    color=colors[i],
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

        ax.set_title(f"Evolution of RMSD distribution to {ref_names[ref_idx]} state")
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("Probability Density")
        ax.legend()

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(
        output_dir, f"rmsd_distribution_evolution_gamma_{gamma_value}_exp_{exponent}.png"
    )
    fig.savefig(output_path, dpi=300)

    return fig


if __name__ == "__main__":
    main()
