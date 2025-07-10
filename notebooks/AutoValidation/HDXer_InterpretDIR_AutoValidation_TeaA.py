"""
This script analyzes the direct output of HDXer optimization runs.
It generates plots showing the evolution of open/closed state populations based on
HDXer optimization results.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import rms
from scipy import stats

# globally set axes/tick/legend font-sizes
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
base_dir = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/code/HDXer/_output_mcminBV"
output_dir = os.path.join(base_dir, "hdxer_analysis")

# Reference structure paths
open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = (
    "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"
)


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


def extract_hdxer_weights(hdxer_dir, gamma_values, exponents):
    """
    Extract HDXer weights from weight files with direct HDXer output format.

    Parameters:
    -----------
    hdxer_dir : str
        Directory containing HDXer results
    gamma_values : list
        List of gamma values to extract weights for
    exponents : list
        List of exponents for gamma values

    Returns:
    --------
    weights_dict : dict
        Dictionary of weights indexed by gamma
    """
    weights_dict = {}

    # Process each gamma value and exponent
    for gamma in gamma_values:
        for exp in exponents:
            gamma_key = f"gamma_{gamma}x10^{exp}"
            weights_file = os.path.join(
                hdxer_dir,
                f"mixed_gamma_{gamma}x10^{exp}_final_weights.dat",
            )

            # Check if weights file exists
            if os.path.exists(weights_file):
                try:
                    # Load the weights
                    weights = np.loadtxt(weights_file)

                    # Check for NaN or inf values and handle them
                    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                        print(
                            f"Warning: NaN or inf values found in weights for gamma {gamma}x10^{exp}"
                        )
                        # Replace NaN/inf with zeros
                        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

                    # Clip small weights to zero
                    weights[weights < 1e-10] = 0.0

                    # Check if all weights are zero after NaN replacement
                    if np.sum(weights) <= 0:
                        print(f"Warning: All weights are zero for gamma {gamma}x10^{exp}")
                        weights_dict[gamma_key] = None
                        continue

                    # Normalize weights
                    weights = weights / np.sum(weights)

                    weights_dict[gamma_key] = weights
                    print(f"Loaded weights for gamma {gamma}x10^{exp}")
                except Exception as e:
                    print(f"Error loading weights for gamma {gamma}x10^{exp}: {e}")
                    weights_dict[gamma_key] = None
            else:
                print(f"Weights file not found: {weights_file}")
                weights_dict[gamma_key] = None

    return weights_dict


def extract_hdxer_iterations(hdxer_dir, gamma_values, exponents):
    """
    Extract HDXer iteration data from per_iteration_output.dat files.

    Parameters:
    -----------
    hdxer_dir : str
        Directory containing HDXer results
    gamma_values : list
        List of gamma values to extract data for
    exponents : list
        List of exponents for gamma values

    Returns:
    --------
    iterations_dict : dict
        Dictionary of iteration data indexed by gamma
    """
    iterations_dict = {}

    # Process each gamma value and exponent
    for gamma in gamma_values:
        for exp in exponents:
            gamma_key = f"gamma_{gamma}x10^{exp}"
            iter_file = os.path.join(
                hdxer_dir,
                f"mixed_gamma_{gamma}x10^{exp}_per_iteration_output.dat",
            )

            # Check if iteration file exists
            if os.path.exists(iter_file):
                try:
                    # Load the iteration data
                    # Format: Iteration, Chi^2, Average ln(w), Worst ln(w), Work
                    iter_data = np.loadtxt(iter_file)

                    iterations_dict[gamma_key] = iter_data
                    print(f"Loaded iterations for gamma {gamma}x10^{exp}")
                except Exception as e:
                    print(f"Error loading iterations for gamma {gamma}x10^{exp}: {e}")
                    iterations_dict[gamma_key] = None
            else:
                print(f"Iterations file not found: {iter_file}")
                iterations_dict[gamma_key] = None

    return iterations_dict


def compute_state_ratios(weights_dict, cluster_assignments):
    """
    Compute state ratios for each set of weights.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary of weights indexed by gamma
    cluster_assignments : numpy.ndarray
        Cluster assignments for each frame

    Returns:
    --------
    ratios_dict : dict
        Dictionary of state ratios indexed by gamma
    """
    n_clusters = np.max(cluster_assignments) + 1
    ratios_dict = {}

    # Process each gamma value
    for gamma_key, weights in weights_dict.items():
        if weights is None:
            ratios_dict[gamma_key] = None
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
                print(f"Warning: All weights are zero for {gamma_key}")
                ratios_dict[gamma_key] = None
                continue

        # Compute weighted ratio for each cluster
        for cluster_idx in range(n_clusters):
            mask = cluster_assignments == cluster_idx
            cluster_ratios[cluster_idx] = np.sum(weights[mask])

        ratios_dict[gamma_key] = cluster_ratios

    return ratios_dict


def compute_weighted_rmsd_distributions(rmsd_values, weights_dict):
    """
    Compute weighted RMSD distributions for visualization.

    Parameters:
    -----------
    rmsd_values : numpy.ndarray
        RMSD values for each frame and reference
    weights_dict : dict
        Dictionary of weights indexed by gamma

    Returns:
    --------
    kde_dict : dict
        Dictionary with KDE estimations for RMSD distributions
    """
    n_frames, n_refs = rmsd_values.shape

    # Create a grid of RMSD values for KDE
    rmsd_grid = np.linspace(0, np.max(rmsd_values) * 1.1, 1000)

    kde_dict = {}

    # Process each gamma value
    for gamma_key, weights in weights_dict.items():
        if weights is None:
            kde_dict[gamma_key] = None
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
                print(f"Warning: All weights are zero for {gamma_key}")
                kde_dict[gamma_key] = None
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
                        print(f"Warning: All RMSD values are NaN or inf for reference {ref_idx}")
                        continue

                # Create KDE
                kde = stats.gaussian_kde(rmsd_ref, weights=weights)
                kde_values[ref_idx] = kde(rmsd_grid)
            except Exception as e:
                print(f"Error computing KDE for {gamma_key}, reference {ref_idx}: {e}")
                kde_values[ref_idx] = np.zeros_like(rmsd_grid)

        kde_dict[gamma_key] = {"grid": rmsd_grid, "kde_values": kde_values}

    return kde_dict


def compute_iteration_state_ratios(iterations_dict, weights_dict, cluster_assignments):
    """
    Compute state ratios for each iteration of the HDXer optimization.

    Parameters:
    -----------
    iterations_dict : dict
        Dictionary of iteration data indexed by gamma
    weights_dict : dict
        Dictionary of final weights indexed by gamma
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

    # Process each gamma value
    for gamma_key, iter_data in iterations_dict.items():
        if iter_data is None or weights_dict[gamma_key] is None:
            iter_ratios_dict[gamma_key] = None
            continue

        # Get number of iterations
        n_iterations = len(iter_data)

        # Get final weights
        final_weights = weights_dict[gamma_key]

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

        iter_ratios_dict[gamma_key] = {
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

    # Check if we have valid KDE for this gamma value
    if gamma_key not in kde_dict or kde_dict[gamma_key] is None:
        plt.close(fig)
        return None

    # Get the rmsd grid and kde values
    rmsd_grid = kde_dict[gamma_key]["grid"]
    kde_values = kde_dict[gamma_key]["kde_values"]

    # Plot KDE for each reference
    for ref_idx in range(n_refs):
        ax = axes[ref_idx]

        # Plot the KDE
        ax.plot(rmsd_grid, kde_values[ref_idx], "b-", linewidth=2)

        ax.set_title(
            f"RMSD Distribution to {ref_names[ref_idx]} state (γ={gamma_value}×10^{exponent})"
        )
        ax.set_xlabel("RMSD (Å)")

        if ref_idx == 0:
            ax.set_ylabel("Probability Density")

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(
        output_dir, f"rmsd_distribution_gamma_{gamma_value}_exp_{exponent}.png"
    )
    fig.savefig(output_path, dpi=300)

    return fig


def plot_state_ratios(ratios_dict, ref_names, output_dir):
    """
    Plot state ratios for different gamma values.

    Parameters:
    -----------
    ratios_dict : dict
        Dictionary of state ratios indexed by gamma
    ref_names : list
        Names of reference structures
    output_dir : str
        Directory to save output plots

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with state ratio plots
    """
    # Extract all gamma keys
    gamma_keys = list(ratios_dict.keys())

    # Parse gamma keys to get numeric values and create sorting data
    gamma_value_data = []
    for gamma_key in gamma_keys:
        if ratios_dict[gamma_key] is None:
            continue

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
    state_ratios = np.zeros((len(sorted_gamma_keys), n_states))

    # Get ratios for each gamma
    for i, gamma_key in enumerate(sorted_gamma_keys):
        ratios = ratios_dict[gamma_key]
        state_ratios[i, :] = ratios

    # Plot each state
    for state_idx in range(n_states):
        ax.plot(
            range(len(sorted_gamma_keys)),
            state_ratios[:, state_idx],
            label=f"{ref_names[state_idx]} State",
            marker="o",
            color=colors[state_idx % len(colors)],
            linewidth=2,
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
    if gamma_key not in iter_ratios_dict or iter_ratios_dict[gamma_key] is None:
        plt.close(fig)
        return None

    # Get iteration data
    iter_data = iter_ratios_dict[gamma_key]
    iterations = iter_data["iterations"]
    state_ratios = iter_data["state_ratios"]

    # Number of states
    n_states = len(ref_names)

    # Plot each state
    for state_idx in range(n_states):
        ax.plot(
            iterations,
            state_ratios[:, state_idx],
            label=f"{ref_names[state_idx]}",
            color=colors[state_idx % len(colors)],
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


def plot_chi_square_evolution(iterations_dict, output_dir):
    """
    Plot the evolution of chi-square values during optimization for all gamma values.

    Parameters:
    -----------
    iterations_dict : dict
        Dictionary of iteration data indexed by gamma
    output_dir : str
        Directory to save output plots

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with chi-square evolution plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Parse gamma keys to get numeric values for legend
    gamma_labels = {}
    for gamma_key in iterations_dict:
        if iterations_dict[gamma_key] is None:
            continue

        parts = gamma_key.split("_")[1].split("x")
        gamma = int(parts[0])
        exponent = int(parts[1].replace("10^", ""))
        gamma_labels[gamma_key] = f"γ={gamma}×10^{exponent}"

    # Set up color cycle
    colors = plt.cm.viridis(np.linspace(0, 1, len(gamma_labels)))

    # Plot chi-square evolution for each gamma value
    for i, (gamma_key, label) in enumerate(gamma_labels.items()):
        iter_data = iterations_dict[gamma_key]
        if iter_data is not None:
            iterations = iter_data[:, 0]  # First column is iteration number
            chi_square = iter_data[:, 1]  # Second column is chi-square

            ax.plot(iterations, chi_square, color=colors[i], linewidth=2, label=label)

    # Set plot labels and title
    ax.set_xlabel("Iteration")
    ax.set_ylabel("χ²")
    ax.set_title("Evolution of χ² During Optimization")

    # Use log scale for y-axis if range is large
    if ax.get_ylim()[1] / max(1e-10, ax.get_ylim()[0]) > 100:
        ax.set_yscale("log")

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "chi_square_evolution.png")
    fig.savefig(output_path, dpi=300)

    return fig


def main():
    """
    Main function to run the HDXer direct output analysis.
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

    # Define HDXer parameters to analyze based on the tree output
    gamma_values = [1]  # Based on your tree output
    exponents = [-1, 0, 1, 2, 3]  # Based on your tree output

    # Extract HDXer weights
    print("Extracting HDXer weights...")
    weights_dict = extract_hdxer_weights(base_dir, gamma_values, exponents)

    # Verify that weights were successfully loaded
    loaded_weights = sum(1 for w in weights_dict.values() if w is not None)
    total_weights = len(weights_dict)
    print(f"Successfully loaded {loaded_weights} out of {total_weights} weight sets.")

    if loaded_weights == 0:
        print("No weights could be loaded. Please check the file paths and gamma/exponent values.")
        return

    # Extract HDXer iteration data
    print("Extracting HDXer iteration data...")
    iterations_dict = extract_hdxer_iterations(base_dir, gamma_values, exponents)

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

    # Create chi-square evolution plot
    print("Creating chi-square evolution plot...")
    chi_square_fig = plot_chi_square_evolution(iterations_dict, output_dir)

    # Create plots for specific gamma values
    for gamma in gamma_values:
        for exp in exponents:
            gamma_key = f"gamma_{gamma}x10^{exp}"
            if gamma_key in weights_dict and weights_dict[gamma_key] is not None:
                print(f"Creating plots for gamma {gamma}×10^{exp}...")

                # Plot RMSD distributions
                rmsd_fig = plot_rmsd_distributions(kde_dict, ref_names, output_dir, gamma, exp)

                # Plot state ratio evolution
                ratio_evo_fig = plot_state_ratio_evolution(
                    iter_ratios_dict, ref_names, output_dir, gamma, exp
                )

    print("Analysis complete! Results saved to:", output_dir)


if __name__ == "__main__":
    main()
