"""
This script is used to interpret the results of the JAXENT model for AutoValidation.
For now we are just focussed on the simple case of L2 loss.

This script aims to interpret the results of a simple experiment using a synethic experimental dataset,
the dataset is created by selecting two reference structures and then predicting the uptake using a known distribution.
By fitting to this dataset we can check that the model is able to predict the uptake correctly. We do this over multiple datasets (seeds)

We want to plot the distributions of RMSD the generated ensembles to each of the reference structures. (Seperate Axes for each).
- Use confidence intervals to show the spread across the seeds.
Cluster the dataset based on the minimum RMSD to each of the reference structures.
Using the weights from the model we can then plot the distribution of weights for each cluster. Plot the weights as a histogram, hue by dataset (seed).

output_dir = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/TeaA/quick_auto_validation_results"
base_output_dir = os.path.join(
    output_dir, "_analysis"
)  # This should match the directory used in the autovalidation script
open_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
# trajectory_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_initial_sliced.xtc"
trajectory_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"

"""

import glob
import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import rms
from scipy import stats

from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters

# Import utility functions from the provided files
from jaxent.src.utils.hdf import load_optimization_history_from_file

# Set up paths - these should be adjusted based on the actual file locations

output_dir = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/TeaA_simple_0.01_adam/quick_auto_validation_results"
base_output_dir = os.path.join(
    output_dir, "_analysis"
)  # This should match the directory used in the autovalidation script
open_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = (
    "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"
)


# Function to extract states at regular intervals from optimization histories
def extract_interval_states(histories, num_intervals=10):
    """
    Extract optimization states at regular intervals from the histories.

    Parameters:
    -----------
    histories : list
        List of optimization histories
    num_intervals : int
        Number of intervals to extract (including initial and final states)

    Returns:
    --------
    interval_states : list
        List of lists, where each inner list contains states from all histories at a specific interval
    """
    interval_states = []

    for interval_idx in range(num_intervals):
        states_at_interval = []

        for history in histories:
            total_steps = len(history.states)
            if total_steps <= 1:
                # If history only has one state, use it for all intervals
                states_at_interval.append(history.states[0])
            else:
                # Calculate position for this interval
                if interval_idx == 0:
                    # First interval is always the initial state
                    pos = 0
                elif interval_idx == num_intervals - 1:
                    # Last interval is always the final state
                    pos = total_steps - 1
                else:
                    # Intermediate intervals are evenly spaced
                    pos = int((interval_idx / (num_intervals - 1)) * (total_steps - 1))

                states_at_interval.append(history.states[pos])

        interval_states.append(states_at_interval)

    return interval_states


# Function to visualize RMSD distribution evolution using KDE
def plot_rmsd_distribution_evolution_kde(rmsd_values, interval_states, ref_names, num_intervals=10):
    """
    Plot the evolution of RMSD distributions during optimization using KDE.

    Parameters:
    -----------
    rmsd_values : numpy.ndarray
        RMSD values for each frame and reference
    interval_states : list
        List of lists of optimization states at each interval
    ref_names : list
        Names of reference structures
    num_intervals : int
        Number of intervals to display

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with RMSD distribution evolution plots
    """
    n_frames, n_refs = rmsd_values.shape

    # Create grid for KDE evaluation
    rmsd_grid = np.linspace(np.min(rmsd_values), np.max(rmsd_values) * 1.1, 1000)

    # Create figure with one subplot per reference
    fig, axes = plt.subplots(n_refs, 1, figsize=(12, 5 * n_refs), squeeze=False)

    # Create colormap for evolution (from light to dark)
    cmap = plt.cm.Blues
    colors = [cmap(i) for i in np.linspace(0.3, 1.0, num_intervals)]

    for ref_idx in range(n_refs):
        ax = axes[ref_idx, 0]

        # Plot uniform distribution first
        uniform_weights = np.ones(n_frames) / n_frames
        uniform_kde = stats.gaussian_kde(rmsd_values[:, ref_idx], weights=uniform_weights)
        uniform_density = uniform_kde(rmsd_grid)
        ax.plot(rmsd_grid, uniform_density, "k--", label="Initial (uniform)", linewidth=2)

        # Plot distribution at each interval
        for interval_idx, states in enumerate(interval_states):
            # Extract weights from all states at this interval
            all_weights = []
            for state in states:
                weights = np.array(state.params.frame_weights).flatten()
                if len(weights) > n_frames:
                    weights = weights[:n_frames]
                weights = weights / np.sum(weights)
                all_weights.append(weights)

            # Average weights across all seeds
            avg_weights = np.mean(all_weights, axis=0)
            avg_weights = avg_weights / np.sum(avg_weights)  # Renormalize

            # Compute KDE with average weights
            kde = stats.gaussian_kde(rmsd_values[:, ref_idx], weights=avg_weights)
            density = kde(rmsd_grid)

            # Set label for first and last interval only to avoid cluttered legend
            if interval_idx == 0:
                label = "Step 0"
            elif interval_idx == len(interval_states) - 1:
                label = "Final step"
            else:
                label = None

            # Plot KDE with color based on interval
            line = ax.plot(rmsd_grid, density, color=colors[interval_idx], linewidth=2, label=label)

            # Add a text label on the line for intermediate steps
            if 0 < interval_idx < len(interval_states) - 1 and interval_idx % 2 == 0:
                # Find a good position for the text (at peak)
                peak_idx = np.argmax(density)
                ax.text(
                    rmsd_grid[peak_idx],
                    density[peak_idx],
                    f"Step {interval_idx}",
                    color=colors[interval_idx],
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                )

        ax.set_title(f"Evolution of RMSD distribution to {ref_names[ref_idx]} state")
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("Probability Density")
        ax.legend()

    plt.tight_layout()
    return fig


# Function to visualize RMSD distribution evolution using histograms
def plot_rmsd_histogram_evolution(
    rmsd_values, interval_states, ref_names, num_intervals=10, n_bins=30
):
    """
    Plot the evolution of RMSD distributions during optimization using histograms.

    Parameters:
    -----------
    rmsd_values : numpy.ndarray
        RMSD values for each frame and reference
    interval_states : list
        List of lists of optimization states at each interval
    ref_names : list
        Names of reference structures
    num_intervals : int
        Number of intervals to display
    n_bins : int
        Number of bins for histograms

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with RMSD histogram evolution plots
    """
    n_frames, n_refs = rmsd_values.shape

    # Create figure with one subplot per reference
    fig, axes = plt.subplots(n_refs, 1, figsize=(12, 5 * n_refs), squeeze=False)

    # Create colormap for evolution (from light to dark)
    cmap = plt.cm.Blues
    colors = [cmap(i) for i in np.linspace(0.3, 1.0, num_intervals)]

    # Select subset of intervals to display (to avoid overcrowding)
    if num_intervals > 5:
        display_indices = [
            0,
            num_intervals // 4,
            num_intervals // 2,
            3 * num_intervals // 4,
            num_intervals - 1,
        ]
    else:
        display_indices = range(num_intervals)

    for ref_idx in range(n_refs):
        ax = axes[ref_idx, 0]

        # Calculate histogram bins once to ensure consistency
        hist_range = (np.min(rmsd_values[:, ref_idx]), np.max(rmsd_values[:, ref_idx]))
        bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Plot uniform distribution
        uniform_weights = np.ones(n_frames) / n_frames
        hist, _ = np.histogram(rmsd_values[:, ref_idx], bins=bin_edges, weights=uniform_weights)
        hist = hist / np.sum(hist)  # Normalize
        ax.step(
            bin_centers,
            hist,
            where="mid",
            color="k",
            linestyle="--",
            linewidth=2,
            label="Initial (uniform)",
        )

        # Plot distribution at selected intervals
        for i, interval_idx in enumerate(display_indices):
            if interval_idx >= len(interval_states):
                continue

            states = interval_states[interval_idx]

            # Extract weights from all states at this interval
            all_weights = []
            for state in states:
                weights = np.array(state.params.frame_weights).flatten()
                if len(weights) > n_frames:
                    weights = weights[:n_frames]
                weights = weights / np.sum(weights)
                all_weights.append(weights)

            # Average weights across all seeds
            avg_weights = np.mean(all_weights, axis=0)
            avg_weights = avg_weights / np.sum(avg_weights)  # Renormalize

            # Compute histogram with average weights
            hist, _ = np.histogram(rmsd_values[:, ref_idx], bins=bin_edges, weights=avg_weights)
            hist = hist / np.sum(hist)  # Normalize

            # Set label
            if interval_idx == 0:
                label = "Step 0"
            elif interval_idx == len(interval_states) - 1:
                label = "Final step"
            else:
                label = f"Step {interval_idx}"

            # Plot histogram
            ax.step(
                bin_centers, hist, where="mid", color=colors[interval_idx], linewidth=2, label=label
            )

        ax.set_title(f"Evolution of RMSD histogram to {ref_names[ref_idx]} state")
        ax.set_xlabel("RMSD (Å)")
        ax.set_ylabel("Normalized Frequency")
        ax.legend()

    plt.tight_layout()
    return fig


# Function to visualize state ratio evolution
def plot_state_ratio_evolution(cluster_assignments, interval_states, ref_names, true_ratios=None):
    """
    Plot the evolution of state ratios during optimization.

    Parameters:
    -----------
    cluster_assignments : numpy.ndarray
        Cluster assignments for each frame
    interval_states : list
        List of lists of optimization states at each interval
    ref_names : list
        Names of reference structures
    true_ratios : list, optional
        True ratios for each state

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with state ratio evolution plot
    """
    n_frames = len(cluster_assignments)
    n_clusters = np.max(cluster_assignments) + 1
    n_intervals = len(interval_states)

    # Calculate state ratios at each interval
    interval_ratios = np.zeros((n_intervals, n_clusters))

    for interval_idx, states in enumerate(interval_states):
        # Extract weights from all states at this interval
        all_ratios = np.zeros((len(states), n_clusters))

        for i, state in enumerate(states):
            weights = np.array(state.params.frame_weights).flatten()
            if len(weights) > n_frames:
                weights = weights[:n_frames]
            weights = weights / np.sum(weights)

            # Calculate ratio for each cluster
            for cluster_idx in range(n_clusters):
                mask = cluster_assignments == cluster_idx
                all_ratios[i, cluster_idx] = np.sum(weights[mask])

        # Average ratios across all seeds
        interval_ratios[interval_idx] = np.mean(all_ratios, axis=0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create x-axis for optimization steps
    # Use actual time steps from first history (average across seeds)
    steps = [states[0].step for states in interval_states]

    # Plot ratio evolution for each state
    for cluster_idx in range(n_clusters):
        ax.plot(
            steps,
            interval_ratios[:, cluster_idx],
            "o-",
            linewidth=2,
            label=f"{ref_names[cluster_idx]} state",
            color=["blue", "orange"][cluster_idx],
        )

    # Plot true ratios if provided
    if true_ratios is not None:
        for cluster_idx, ratio in enumerate(true_ratios):
            ax.axhline(
                ratio,
                color=["blue", "orange"][cluster_idx],
                linestyle="--",
                label=f"True {ref_names[cluster_idx]} ratio",
            )

    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("State Ratio")
    ax.set_title("Evolution of State Ratios During Optimization")
    ax.legend()

    # Set y-axis limits with some padding
    ax.set_ylim(0, 1)

    # Only show a subset of x-ticks to avoid crowding
    if len(steps) > 5:
        step_ticks = [
            steps[0],
            steps[len(steps) // 4],
            steps[len(steps) // 2],
            steps[3 * len(steps) // 4],
            steps[-1],
        ]
        ax.set_xticks(step_ticks)

    plt.tight_layout()
    return fig


# Function to load optimization histories for all seeds
def load_all_optimization_histories(output_dir):
    histories = []
    # Find all optimization history files
    history_files = glob.glob(os.path.join(output_dir, "seed_*_optimization_history.h5"))

    for file in sorted(history_files):
        try:
            history = load_optimization_history_from_file(
                file, default_model_params_cls=BV_Model_Parameters
            )
            histories.append(history)
            print(f"Loaded history from {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return histories


# Function to compute RMSD between each frame in trajectory and reference structures
def compute_rmsd_to_references(trajectory_path, topology_path, reference_paths):
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


# Function to cluster frames based on minimum RMSD
def cluster_frames_by_rmsd(rmsd_values):
    # Assign each frame to the reference with minimum RMSD
    cluster_assignments = np.argmin(rmsd_values, axis=1)
    return cluster_assignments


# Function to compute weighted RMSD distributions
def compute_weighted_rmsd_distributions(rmsd_values, histories):
    n_seeds = len(histories)
    n_frames, n_refs = rmsd_values.shape

    # Create a grid of RMSD values for KDE
    rmsd_grid = np.linspace(0, np.max(rmsd_values) * 1.1, 1000)

    # Initialize arrays for KDE estimates
    kde_values = np.zeros((n_seeds, n_refs, len(rmsd_grid)))

    for i, history in enumerate(histories):
        # Get weights from final optimization state
        final_state = history.states[-1]
        weights = np.array(final_state.params.frame_weights).flatten()

        # Make sure weights length matches number of frames
        if len(weights) != n_frames:
            print(
                f"Warning: weights length {len(weights)} doesn't match number of frames {n_frames}"
            )
            weights = weights[:n_frames]  # Truncate if too long
            weights = weights / np.sum(weights)  # Renormalize

        # Compute weighted KDE for each reference
        for j in range(n_refs):
            kde = stats.gaussian_kde(rmsd_values[:, j], weights=weights)
            kde_values[i, j] = kde(rmsd_grid)

    # Compute original (uniform weights) distribution
    uniform_weights = np.ones(n_frames) / n_frames
    uniform_kde_values = np.zeros((n_refs, len(rmsd_grid)))
    for j in range(n_refs):
        kde = stats.gaussian_kde(rmsd_values[:, j], weights=uniform_weights)
        uniform_kde_values[j] = kde(rmsd_grid)

    return rmsd_grid, kde_values, uniform_kde_values


# Function to compute weighted cluster ratios
def compute_weighted_cluster_ratios(cluster_assignments, histories):
    n_seeds = len(histories)
    n_clusters = np.max(cluster_assignments) + 1

    # Initialize array for cluster ratios
    cluster_ratios = np.zeros((n_seeds, n_clusters))

    for i, history in enumerate(histories):
        # Get weights from final optimization state
        final_state = history.states[-1]
        weights = np.array(final_state.params.frame_weights).flatten()

        # Make sure weights length matches cluster assignments
        if len(weights) != len(cluster_assignments):
            print(
                f"Warning: weights length {len(weights)} doesn't match assignments length {len(cluster_assignments)}"
            )
            weights = weights[: len(cluster_assignments)]  # Truncate if too long
            weights = weights / np.sum(weights)  # Renormalize

        # Compute weighted ratio for each cluster
        for j in range(n_clusters):
            mask = cluster_assignments == j
            cluster_ratios[i, j] = np.sum(weights[mask])

    # Compute original (uniform weights) cluster ratio
    uniform_weights = np.ones(len(cluster_assignments)) / len(cluster_assignments)
    uniform_cluster_ratios = np.zeros(n_clusters)
    for j in range(n_clusters):
        mask = cluster_assignments == j
        uniform_cluster_ratios[j] = np.sum(uniform_weights[mask])

    return cluster_ratios, uniform_cluster_ratios


# Function to plot RMSD distributions with confidence intervals (KDE method)
def plot_rmsd_distributions(rmsd_grid, kde_values, uniform_kde_values, ref_names):
    n_refs = kde_values.shape[1]
    fig, axes = plt.subplots(1, n_refs, figsize=(15, 6), sharey=True)

    if n_refs == 1:
        axes = [axes]  # Make it iterable for the loop

    for j in range(n_refs):
        ax = axes[j]

        # Compute mean and confidence intervals across seeds
        mean_kde = np.mean(kde_values[:, j, :], axis=0)
        std_kde = np.std(kde_values[:, j, :], axis=0)
        lower_ci = mean_kde - 1.96 * std_kde / np.sqrt(kde_values.shape[0])
        upper_ci = mean_kde + 1.96 * std_kde / np.sqrt(kde_values.shape[0])

        # Plot individual seed distributions
        for i in range(kde_values.shape[0]):
            ax.plot(rmsd_grid, kde_values[i, j, :], alpha=0.2, color="blue")

        # Plot mean curve and confidence interval
        ax.plot(rmsd_grid, mean_kde, "b-", linewidth=2, label="Optimized (Mean)")
        ax.fill_between(rmsd_grid, lower_ci, upper_ci, alpha=0.3, color="blue", label="95% CI")

        # Plot original (uniform weights) distribution
        ax.plot(rmsd_grid, uniform_kde_values[j], "k--", linewidth=2, label="Original")

        ax.set_title(f"RMSD to {ref_names[j]} state")
        ax.set_xlabel("RMSD (Å)")
        if j == 0:
            ax.set_ylabel("Probability Density")

        ax.legend()

    plt.tight_layout()
    return fig


# Function to plot RMSD distributions as histograms (no KDE)
def plot_rmsd_histograms(rmsd_values, histories, ref_names, n_bins=30):
    n_refs = rmsd_values.shape[1]
    fig, axes = plt.subplots(1, n_refs, figsize=(15, 6), sharey=True)

    if n_refs == 1:
        axes = [axes]  # Make it iterable for the loop

    # Create color maps for each state
    state_colors = ["blue", "orange"]

    for j in range(n_refs):
        ax = axes[j]

        # Calculate histogram bins once to ensure consistency
        hist_range = (np.min(rmsd_values[:, j]), np.max(rmsd_values[:, j]))
        bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Plot histogram for uniform weights first (original)
        uniform_weights = np.ones(len(rmsd_values)) / len(rmsd_values)
        hist, _ = np.histogram(rmsd_values[:, j], bins=bin_edges, weights=uniform_weights)
        # Normalize histogram to sum to 1
        hist = hist / np.sum(hist)
        ax.bar(
            bin_centers,
            hist,
            width=bin_width * 0.8,
            alpha=0.5,
            color="gray",
            label="Original (uniform)",
        )

        # Collect weighted histograms for each seed
        all_hists = []
        for i, history in enumerate(histories):
            # Get weights from final optimization state
            final_state = history.states[-1]
            weights = np.array(final_state.params.frame_weights).flatten()

            # Make sure weights length matches number of frames
            if len(weights) != len(rmsd_values):
                weights = weights[: len(rmsd_values)]  # Truncate if too long
                weights = weights / np.sum(weights)  # Renormalize

            # Compute weighted histogram
            hist, _ = np.histogram(rmsd_values[:, j], bins=bin_edges, weights=weights)
            # Normalize histogram to sum to 1
            hist = hist / np.sum(hist)
            all_hists.append(hist)

            # Plot individual seed histograms with low alpha
            ax.plot(
                bin_centers,
                hist,
                alpha=0.3,
                color=state_colors[j],
                label=f"Seed {i + 1}" if i == 0 and j == 0 else None,
            )

        # Calculate and plot the mean histogram
        mean_hist = np.mean(all_hists, axis=0)
        # Need to renormalize since mean of normalized histograms may not sum to 1
        mean_hist = mean_hist / np.sum(mean_hist)
        ax.step(
            bin_centers,
            mean_hist,
            where="mid",
            linewidth=2,
            color=state_colors[j],
            label=f"Optimized Mean ({ref_names[j]})",
        )

        ax.set_title(f"RMSD to {ref_names[j]} state")
        ax.set_xlabel("RMSD (Å)")
        if j == 0:
            ax.set_ylabel("Normalized Frequency")

        # Only show legend on first plot to avoid redundancy
        if j == 0:
            ax.legend()

    plt.tight_layout()
    return fig


# Function to plot cluster ratios with state-matching colors
def plot_cluster_ratios(cluster_ratios, uniform_cluster_ratios, ref_names, true_ratios=[0.6, 0.4]):
    n_seeds = cluster_ratios.shape[0]
    n_clusters = cluster_ratios.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for each state
    state_colors = ["blue", "orange"]  # Colors for Open and Closed states

    # Set up positions for grouped bars
    bar_width = 0.8 / (n_seeds + 1)  # +1 for uniform ratio
    positions = np.arange(n_clusters)

    # Plot uniform ratio first
    for j in range(n_clusters):
        ax.bar(
            j - 0.4,
            uniform_cluster_ratios[j],
            width=bar_width,
            color=state_colors[j],
            alpha=0.3,
            label=f"Original {ref_names[j]}" if j == 0 else f"Original {ref_names[j]}",
        )

    # Plot optimized ratios for each seed, colored by state
    for i in range(n_seeds):
        for j in range(n_clusters):
            offset = -0.4 + (i + 1) * bar_width
            ax.bar(
                j + offset,
                cluster_ratios[i, j],
                width=bar_width,
                color=state_colors[j],
                alpha=0.5 + 0.5 * (i / n_seeds),
                label=f"Seed {i + 1} {ref_names[j]}" if j == 0 and i == 0 else None,
            )

    # Add true ratio reference if provided
    if true_ratios is not None:
        for j, ratio in enumerate(true_ratios):
            ax.axhline(
                ratio,
                color=state_colors[j],
                linestyle="--",
                linewidth=2,
                label=f"True {ref_names[j]} Ratio ({ratio * 100:.0f}%)",
            )
    ax.set_ylim(0, 1)

    ax.set_xlabel("State")
    ax.set_ylabel("Weighted Ratio")
    ax.set_title("Weighted State Ratios Across Seeds")
    ax.set_xticks(positions)
    ax.set_xticklabels(ref_names)

    # Create legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig


# Function to plot a box plot of cluster ratios
def plot_cluster_ratio_boxplot(cluster_ratios, uniform_cluster_ratios, ref_names, true_ratios=None):
    n_clusters = cluster_ratios.shape[1]

    # Reshape for box plot
    data = []
    for j in range(n_clusters):
        data.append(cluster_ratios[:, j])

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(data, labels=ref_names)

    # Add uniform ratio as triangles
    for j in range(n_clusters):
        ax.plot(
            j + 1,
            uniform_cluster_ratios[j],
            "^",
            color="gray",
            markersize=10,
            label="Original" if j == 0 else "",
        )

    # Add true ratio reference if provided
    if true_ratios is not None:
        for j, ratio in enumerate(true_ratios):
            ax.plot(
                j + 1,
                ratio,
                "x",
                color=["r", "b"][j],
                markersize=10,
                label=f"True {ref_names[j]} Ratio ({ratio * 100:.0f}%)"
                if j == 0
                else f"True {ref_names[j]} Ratio",
            )

    ax.set_ylabel("Weighted Ratio")
    ax.set_title("Distribution of Weighted State Ratios Across Seeds")

    # Create a legend without duplicate entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig


# Main function to run the analysis
def main():
    # Create output directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    # Load all optimization histories
    print("Loading optimization histories...")
    histories = load_all_optimization_histories(output_dir)

    if not histories:
        print("No optimization histories found. Exiting.")
        return

    # Compute RMSD to reference structures
    reference_paths = [open_path, closed_path]
    ref_names = ["Open", "Closed"]
    true_ratios = [0.6, 0.4]  # The true ratios used to generate synthetic data

    print("Computing RMSD to reference structures...")
    rmsd_values = compute_rmsd_to_references(trajectory_path, topology_path, reference_paths)

    # Cluster frames based on minimum RMSD
    print("Clustering frames by RMSD...")
    cluster_assignments = cluster_frames_by_rmsd(rmsd_values)

    # Compute weighted RMSD distributions
    print("Computing weighted RMSD distributions...")
    rmsd_grid, kde_values, uniform_kde_values = compute_weighted_rmsd_distributions(
        rmsd_values, histories
    )

    # Compute weighted cluster ratios
    print("Computing weighted cluster ratios...")
    cluster_ratios, uniform_cluster_ratios = compute_weighted_cluster_ratios(
        cluster_assignments, histories
    )

    # Print summary statistics
    print("\nSummary Statistics:")
    print("-----------------")
    print(f"Number of seeds: {len(histories)}")
    print(f"Number of frames: {len(cluster_assignments)}")
    print(f"Original cluster ratios: {uniform_cluster_ratios}")
    print(f"Mean optimized cluster ratios: {np.mean(cluster_ratios, axis=0)}")
    print(f"Std dev optimized cluster ratios: {np.std(cluster_ratios, axis=0)}")

    # Plot RMSD distributions (KDE method)
    print("\nPlotting RMSD distributions with KDE...")
    rmsd_fig = plot_rmsd_distributions(rmsd_grid, kde_values, uniform_kde_values, ref_names)
    rmsd_path = os.path.join(base_output_dir, "rmsd_distributions_kde.png")
    rmsd_fig.savefig(rmsd_path, dpi=300, bbox_inches="tight")
    print(f"Saved KDE RMSD distributions to {rmsd_path}")

    # Plot RMSD distributions as histograms (no KDE)
    print("Plotting RMSD histograms (no KDE)...")
    hist_fig = plot_rmsd_histograms(rmsd_values, histories, ref_names, n_bins=30)
    hist_path = os.path.join(base_output_dir, "rmsd_histograms.png")
    hist_fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    print(f"Saved RMSD histograms to {hist_path}")

    # Plot cluster ratios
    print("Plotting cluster ratios...")
    ratio_fig = plot_cluster_ratios(cluster_ratios, uniform_cluster_ratios, ref_names, true_ratios)
    ratio_path = os.path.join(base_output_dir, "cluster_ratios.png")
    ratio_fig.savefig(ratio_path, dpi=300, bbox_inches="tight")
    print(f"Saved cluster ratios to {ratio_path}")

    # Plot box plot of cluster ratios
    print("Plotting cluster ratio box plot...")
    box_fig = plot_cluster_ratio_boxplot(
        cluster_ratios, uniform_cluster_ratios, ref_names, true_ratios
    )
    box_path = os.path.join(base_output_dir, "cluster_ratio_boxplot.png")
    box_fig.savefig(box_path, dpi=300, bbox_inches="tight")
    print(f"Saved cluster ratio box plot to {box_path}")

    # Extract states at regular intervals for evolution plots
    print("\nGenerating optimization evolution plots...")
    num_intervals = 10
    evolution_dir = os.path.join(base_output_dir, "evolution")
    os.makedirs(evolution_dir, exist_ok=True)

    print("Extracting states at regular intervals...")
    interval_states = extract_interval_states(histories, num_intervals)

    # Plot RMSD distribution evolution (KDE)
    print("Plotting RMSD distribution evolution with KDE...")
    kde_evo_fig = plot_rmsd_distribution_evolution_kde(
        rmsd_values, interval_states, ref_names, num_intervals
    )
    kde_evo_path = os.path.join(evolution_dir, "rmsd_kde_evolution.png")
    kde_evo_fig.savefig(kde_evo_path, dpi=300, bbox_inches="tight")
    print(f"Saved RMSD KDE evolution to {kde_evo_path}")

    # Plot RMSD distribution evolution (histogram)
    print("Plotting RMSD histogram evolution...")
    hist_evo_fig = plot_rmsd_histogram_evolution(
        rmsd_values, interval_states, ref_names, num_intervals
    )
    hist_evo_path = os.path.join(evolution_dir, "rmsd_histogram_evolution.png")
    hist_evo_fig.savefig(hist_evo_path, dpi=300, bbox_inches="tight")
    print(f"Saved RMSD histogram evolution to {hist_evo_path}")

    # Plot state ratio evolution
    print("Plotting state ratio evolution...")
    ratio_evo_fig = plot_state_ratio_evolution(
        cluster_assignments, interval_states, ref_names, true_ratios
    )
    ratio_evo_path = os.path.join(evolution_dir, "state_ratio_evolution.png")
    ratio_evo_fig.savefig(ratio_evo_path, dpi=300, bbox_inches="tight")
    print(f"Saved state ratio evolution to {ratio_evo_path}")

    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()
