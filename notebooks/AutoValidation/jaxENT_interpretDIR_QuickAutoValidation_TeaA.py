"""
This script extends the auto validation analysis to work across multiple experiments
in a directory structure. It generates plots for each experiment and also creates
plots showing average performance across different parameters.
"""

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from MDAnalysis.analysis import rms
from scipy import stats

from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.utils.hdf import load_optimization_history_from_file

mpl.rcParams.update(
    {
        "axes.titlesize": 20,
        "axes.labelsize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 20,
        "legend.fontsize": 16,
        "font.size": 24,  # default for all text (fallback)
    }
)


def plot_boxplots(results, metric_name, method_names=None, title=None, ylabel=None):
    """
    Create box plots of metric values between seeds with each method in its own panel.
    Parameters are ordered in ascending order.

    Parameters:
    -----------
    results : dict
        Dictionary with metric values grouped by method and parameter
    metric_name : str
        Name of the metric (e.g., "KL Divergence" or "W1 Distance")
    method_names : list, optional
        List of method names for the legend
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The box plot figure
    """
    # Extract methods with valid data
    valid_methods = [method for method, param_dict in results.items() if any(param_dict.values())]

    if not valid_methods:
        # Create a simple figure if no valid data
        fig = plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            f"No valid {metric_name} data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    # Create figure with subplots (one for each method)
    fig, axes = plt.subplots(
        1,
        len(valid_methods),
        figsize=(7 * len(valid_methods), 7),  # Adjusted figsize
        sharey=True,  # Share y-axis across all panels
        gridspec_kw={"wspace": 0.05},  # Reduce space between subplots
    )

    # Handle single subplot case
    if len(valid_methods) == 1:
        axes = [axes]

    # Process each method in its own subplot
    for method_idx, method in enumerate(valid_methods):
        ax = axes[method_idx]
        param_dict = results[method]

        # Prepare data for boxplot
        method_data_filtered = []
        method_labels = []
        param_colors_filtered = []
        original_indices = []  # Keep track of original parameter index for color mapping

        # Convert parameters to sortable values for ordering
        sorted_params = []

        for param_idx, (param, values) in enumerate(param_dict.items()):
            # Filter out non-numeric or invalid values
            valid_values = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]

            if not valid_values:
                print(f"Skipping parameter {param} for method {method} due to no valid data.")
                continue  # Skip if no valid data after filtering

            # Calculate sort value and label based on parameter type
            if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                gamma, exponent = param
                # Use actual numerical value γ × 10^exponent for sorting
                sort_value = gamma * (10**exponent)
                label = f"γ={gamma}×10^{exponent}"
            elif method == "HDXer":
                try:
                    sort_value = float(param)
                    label = f"γ={param}"
                except (ValueError, TypeError):
                    sort_value = param_idx  # Fallback sort value
                    label = str(param)  # Fallback label
            else:
                try:
                    sort_value = float(param)
                    label = f"α={param:.2e}"
                except (ValueError, TypeError):
                    sort_value = param_idx  # Fallback sort value
                    label = str(param)  # Fallback label

            sorted_params.append((sort_value, param, valid_values, label, param_idx))

        # Sort parameters by their numerical value in ascending order
        sorted_params.sort(key=lambda x: x[0])

        # Define colors for parameters based on the original unsorted list size
        param_colors = plt.cm.viridis(np.linspace(0, 1, max(len(param_dict), 1)))

        # Extract sorted data and map colors correctly
        for i, (_, _, valid_values, label, original_idx) in enumerate(sorted_params):
            method_data_filtered.append(valid_values)
            method_labels.append(label)
            param_colors_filtered.append(param_colors[original_idx])
            original_indices.append(i)  # Store the new index after sorting

        # Create box plot with seaborn for this method
        if method_data_filtered:
            # Boxplot uses positions 0, 1, 2...
            sns.boxplot(data=method_data_filtered, ax=ax, palette=param_colors_filtered, width=0.6)

            # Add individual data points using the correct positions and colors
            for i, data in enumerate(method_data_filtered):
                # 'i' is the position on the x-axis (0, 1, 2...)
                # Use jitter around the integer position 'i'
                jitter = np.random.normal(loc=i, scale=0.04, size=len(data))
                # Use the correctly mapped color
                ax.scatter(jitter, data, color=param_colors_filtered[i], s=10, alpha=0.5, zorder=1)

            # Explicitly set x-tick positions and labels AFTER plotting
            ax.set_xticks(range(len(method_labels)))
            ax.set_xticklabels(method_labels, rotation=45, ha="right")

            # Add method title
            ax.set_title(method)

            # Only show y-label on the first subplot
            if method_idx == 0:
                ax.set_ylabel(ylabel if ylabel else metric_name)
            else:
                ax.tick_params(labelleft=False)  # Hide y-tick labels for other subplots

            # Add grid
            ax.grid(True, axis="y", alpha=0.3)  # Grid only on y-axis
        else:
            ax.text(
                0.5,
                0.5,
                "No valid data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_xticks([])  # No ticks if no data
            ax.set_yticks([])  # No ticks if no data

    # Add overall title
    fig.suptitle(title if title else f"{metric_name} Comparison", fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust rect slightly for labels

    return fig


# Reuse functions from the original script
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


def plot_state_ratio_evolution(
    cluster_assignments, interval_states, ref_names, true_ratios=[0.6, 0.4]
):
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
    state_colors = ["blue", "orange"]  # Colors for Open and Closed states

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
    for j, ratio in enumerate(true_ratios):
        ax.axhline(
            ratio,
            color=state_colors[j],
            linestyle="--",
            linewidth=2,
            label=f"True {ref_names[j]} Ratio ({ratio * 100:.0f}%)",
        )

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


def cluster_frames_by_rmsd(rmsd_values):
    # Assign each frame to the reference with minimum RMSD
    cluster_assignments = np.argmin(rmsd_values, axis=1)
    return cluster_assignments


def compute_weighted_rmsd_distributions(rmsd_values, histories):
    n_seeds = len(histories)
    n_frames, n_refs = rmsd_values.shape

    # Create a grid of RMSD values for KDE
    rmsd_grid = np.linspace(0, np.max(rmsd_values) * 1.1, 100)

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
    # if true_ratios is not None:
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


def plot_cluster_ratio_boxplot(
    cluster_ratios, uniform_cluster_ratios, ref_names, true_ratios=[0.6, 0.4]
):
    n_clusters = cluster_ratios.shape[1]
    state_colors = ["blue", "orange"]  # Colors for Open and Closed states

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
    for j, ratio in enumerate(true_ratios):
        ax.axhline(
            ratio,
            color=state_colors[j],
            linestyle="--",
            linewidth=2,
            label=f"True {ref_names[j]} Ratio ({ratio * 100:.0f}%)",
        )
    ax.set_ylabel("Weighted Ratio")
    ax.set_title("Distribution of Weighted State Ratios Across Seeds")

    # Create a legend without duplicate entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig


# New functions for directory traversal and parameter extraction
def find_experiment_directories(base_dir):
    """
    Find all experiment directories in the given base directory.

    Parameters:
    -----------
    base_dir : str
        Base directory to search in

    Returns:
    --------
    experiments : list
        List of tuples (directory, method, param) with experiment information
    """
    experiments = []

    # Find all directories that match the pattern
    exp_dirs = glob.glob(os.path.join(base_dir, "*"))
    exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]

    for exp_dir in exp_dirs:
        # Extract method and parameter from directory name
        dirname = os.path.basename(exp_dir)

        # Expected format: <method>_<param>_adam or similar
        parts = dirname.split("_")

        if len(parts) >= 3:
            # Assuming format like "TeaA_simple_0.01_adam"
            method = "_".join(parts[:-2])  # Everything before the parameter and optimizer
            param_str = parts[-2]  # Parameter value before the optimizer

            try:
                # Try to convert parameter to appropriate type
                if param_str == "0":
                    param = 0
                else:
                    param = float(param_str)

                # Check if this directory has results
                results_dir = os.path.join(exp_dir, "quick_auto_validation_results")
                if os.path.isdir(results_dir):
                    experiments.append((results_dir, method, param))
            except ValueError:
                # Skip if parameter cannot be converted
                continue

    return experiments


def extract_and_sort_parameters(experiments):
    """
    Extract parameters from experiment directories and create sorted parameter lists
    with appropriate labels.

    Parameters:
    -----------
    experiments : list
        List of tuples (directory, method, param) with experiment information

    Returns:
    --------
    sorted_params : list
        List of tuples (method, sorted_params) where sorted_params is a list of
        tuples (sort_value, param, directories, label)
    methods : list
        List of unique methods found
    """
    # Group experiments by method
    method_groups = {}
    for directory, method, param in experiments:
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append((directory, param))

    methods = list(method_groups.keys())
    all_sorted_params = []

    for method, values in method_groups.items():
        # Group directories by parameter value
        param_dict = {}
        for directory, param in values:
            if param not in param_dict:
                param_dict[param] = []
            param_dict[param].append(directory)

        # Sort parameters and create labels
        sorted_params = []
        for param, directories in param_dict.items():
            if not directories:  # Skip params with no values
                continue

            # Calculate sort value and label based on parameter type
            if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                gamma, exponent = param
                # Convert to actual numerical value for sorting
                sort_value = gamma * (10**exponent)
                label = f"γ={gamma}×10^{exponent}"
            elif method == "HDXer":
                sort_value = param
                label = f"γ={param}"
            else:
                # For non-HDXer methods, treat param as alpha value
                sort_value = param
                # Format alpha in standard form
                if isinstance(param, float):
                    # Handle the case where param is exactly 0
                    if param == 0:
                        label = "α=0"
                    else:
                        # Extract mantissa and exponent
                        exponent = int(np.floor(np.log10(abs(param))))
                        mantissa = param / (10**exponent)
                        # Format using standard form
                        if exponent == 0:
                            label = f"α={mantissa:.1f}"
                        else:
                            label = f"α={mantissa:.1f}×10^{exponent}"
                else:
                    label = f"α={param}"

            sorted_params.append((sort_value, param, directories, label))

        # Sort by the numerical value
        sorted_params.sort(key=lambda x: x[0])
        all_sorted_params.append((method, sorted_params))

    return all_sorted_params, methods


def analyze_experiments(base_dir, reference_paths, topology_path, trajectory_path):
    """
    Analyze all experiments in the given directory structure.

    Parameters:
    -----------
    base_dir : str
        Base directory containing experiment directories
    reference_paths : list
        List of paths to reference PDB structures
    topology_path : str
        Path to topology file
    trajectory_path : str
        Path to trajectory file

    Returns:
    --------
    results : dict
        Dictionary containing analysis results for all experiments
    """
    # Find all experiment directories
    print("Finding experiment directories...")
    experiments = find_experiment_directories(base_dir)

    if not experiments:
        print("No experiments found. Exiting.")
        return None

    print(f"Found {len(experiments)} experiments")

    # Extract and sort parameters
    print("Extracting parameters...")
    sorted_params_by_method, methods = extract_and_sort_parameters(experiments)

    # Pre-compute RMSD values (only needs to be done once)
    print("Computing RMSD to reference structures...")
    ref_names = ["Open", "Closed"]
    rmsd_values = compute_rmsd_to_references(trajectory_path, topology_path, reference_paths)

    # Cluster frames based on minimum RMSD
    print("Clustering frames by RMSD...")
    cluster_assignments = cluster_frames_by_rmsd(rmsd_values)

    # Store results for each experiment
    results = {
        "rmsd_values": rmsd_values,
        "cluster_assignments": cluster_assignments,
        "ref_names": ref_names,
        "experiments": {},
        "sorted_params_by_method": sorted_params_by_method,
        "methods": methods,
    }

    # Process each experiment
    for directory, method, param in experiments:
        print(f"\nProcessing experiment: {os.path.basename(os.path.dirname(directory))}")

        # Create output directory for analysis results
        base_output_dir = os.path.join(directory, "_analysis")
        os.makedirs(base_output_dir, exist_ok=True)

        # Load optimization histories
        histories = load_all_optimization_histories(directory)

        if not histories:
            print(f"No optimization histories found for {directory}. Skipping.")
            continue

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

        # Extract states at regular intervals for evolution plots
        print("Extracting states at regular intervals...")
        num_intervals = 10
        interval_states = extract_interval_states(histories, num_intervals)

        # Store results for this experiment
        exp_results = {
            "method": method,
            "param": param,
            "histories": histories,
            "rmsd_grid": rmsd_grid,
            "kde_values": kde_values,
            "uniform_kde_values": uniform_kde_values,
            "cluster_ratios": cluster_ratios,
            "uniform_cluster_ratios": uniform_cluster_ratios,
            "interval_states": interval_states,
        }

        results["experiments"][(method, param)] = exp_results

        # Create individual experiment plots
        print("Creating plots for this experiment...")

        # Plot RMSD distributions (KDE method)
        rmsd_fig = plot_rmsd_distributions(rmsd_grid, kde_values, uniform_kde_values, ref_names)
        rmsd_path = os.path.join(base_output_dir, "rmsd_distributions_kde.png")
        rmsd_fig.savefig(rmsd_path, dpi=300, bbox_inches="tight")
        plt.close(rmsd_fig)

        # Plot RMSD distributions as histograms (no KDE)
        hist_fig = plot_rmsd_histograms(rmsd_values, histories, ref_names, n_bins=30)
        hist_path = os.path.join(base_output_dir, "rmsd_histograms.png")
        hist_fig.savefig(hist_path, dpi=300, bbox_inches="tight")
        plt.close(hist_fig)

        # Plot cluster ratios
        ratio_fig = plot_cluster_ratios(cluster_ratios, uniform_cluster_ratios, ref_names)
        ratio_path = os.path.join(base_output_dir, "cluster_ratios.png")
        ratio_fig.savefig(ratio_path, dpi=300, bbox_inches="tight")
        plt.close(ratio_fig)

        # Plot box plot of cluster ratios
        box_fig = plot_cluster_ratio_boxplot(cluster_ratios, uniform_cluster_ratios, ref_names)
        box_path = os.path.join(base_output_dir, "cluster_ratio_boxplot.png")
        box_fig.savefig(box_path, dpi=300, bbox_inches="tight")
        plt.close(box_fig)

        # Create evolution plots
        evolution_dir = os.path.join(base_output_dir, "evolution")
        os.makedirs(evolution_dir, exist_ok=True)

        # Plot RMSD distribution evolution (KDE)
        kde_evo_fig = plot_rmsd_distribution_evolution_kde(
            rmsd_values, interval_states, ref_names, num_intervals
        )
        kde_evo_path = os.path.join(evolution_dir, "rmsd_kde_evolution.png")
        kde_evo_fig.savefig(kde_evo_path, dpi=300, bbox_inches="tight")
        plt.close(kde_evo_fig)

        # Plot RMSD histogram evolution
        hist_evo_fig = plot_rmsd_histogram_evolution(
            rmsd_values, interval_states, ref_names, num_intervals
        )
        hist_evo_path = os.path.join(evolution_dir, "rmsd_histogram_evolution.png")
        hist_evo_fig.savefig(hist_evo_path, dpi=300, bbox_inches="tight")
        plt.close(hist_evo_fig)

        # Plot state ratio evolution
        ratio_evo_fig = plot_state_ratio_evolution(cluster_assignments, interval_states, ref_names)
        ratio_evo_path = os.path.join(evolution_dir, "state_ratio_evolution.png")
        ratio_evo_fig.savefig(ratio_evo_path, dpi=300, bbox_inches="tight")
        plt.close(ratio_evo_fig)

        print(f"Saved all plots for {os.path.basename(os.path.dirname(directory))}")

    return results


# Functions to create plots comparing results across parameters
def plot_parameter_comparison_rmsd(results, output_dir):
    """
    Create plots comparing RMSD distributions across different parameter values.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_experiments
    output_dir : str
        Directory to save output plots
    """
    rmsd_values = results["rmsd_values"]
    ref_names = results["ref_names"]

    # Create a directory for parameter comparison plots
    param_dir = os.path.join(output_dir, "parameter_comparison")
    os.makedirs(param_dir, exist_ok=True)

    # Process each method separately
    for method, sorted_params in results["sorted_params_by_method"]:
        if not sorted_params:
            continue

        method_dir = os.path.join(param_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Create a figure for RMSD distributions for each reference
        for ref_idx, ref_name in enumerate(ref_names):
            # Plot KDE RMSD distributions for all parameters
            fig, ax = plt.subplots(figsize=(12, 8))

            # Use a colormap for different parameter values
            n_params = len(sorted_params)
            cmap = plt.cm.viridis
            colors = [cmap(i) for i in np.linspace(0, 1, n_params)]

            # Get grid range
            min_rmsd = np.min(rmsd_values[:, ref_idx])
            max_rmsd = np.max(rmsd_values[:, ref_idx]) * 1.1
            grid = np.linspace(min_rmsd, max_rmsd, 1000)

            # Plot uniform distribution first
            uniform_weights = np.ones(len(rmsd_values)) / len(rmsd_values)
            uniform_kde = stats.gaussian_kde(rmsd_values[:, ref_idx], weights=uniform_weights)
            uniform_density = uniform_kde(grid)
            ax.plot(grid, uniform_density, "k--", label="Initial (uniform)", linewidth=2)

            # Plot each parameter
            for i, (sort_value, param, directories, label) in enumerate(sorted_params):
                exp_results = results["experiments"].get((method, param))
                if exp_results is None:
                    continue

                # Get mean KDE values for this parameter
                kde_values = exp_results["kde_values"]
                mean_kde = np.mean(kde_values[:, ref_idx, :], axis=0)

                # Create grid if needed
                if len(mean_kde) != len(grid):
                    # Interpolate to match grid
                    old_grid = exp_results["rmsd_grid"]
                    mean_kde = np.interp(grid, old_grid, mean_kde)

                # Plot mean KDE curve
                ax.plot(grid, mean_kde, color=colors[i], linewidth=2, label=label)

            ax.set_xlabel("RMSD (Å)")
            ax.set_ylabel("Probability Density")
            ax.set_title(f"RMSD Distribution to {ref_name} State by Parameter ({method})")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Save figure
            plt.tight_layout()
            fig.savefig(
                os.path.join(method_dir, f"rmsd_kde_{ref_name.lower()}_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    print(f"Saved parameter comparison RMSD plots to {param_dir}")


def plot_parameter_comparison_ratio(results, output_dir, reverse_xscale=False):
    """
    Create plots comparing cluster ratios across different parameter values.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_experiments
    output_dir : str
        Directory to save output plots
    """
    ref_names = results["ref_names"]
    cluster_assignments = results["cluster_assignments"]

    # Create a directory for parameter comparison plots
    param_dir = os.path.join(output_dir, "parameter_comparison")
    os.makedirs(param_dir, exist_ok=True)
    true_ratios = [0.6, 0.4]  # Example true ratios for Open and Closed states
    # Process each method separately
    for method, sorted_params in results["sorted_params_by_method"]:
        if not sorted_params:
            continue

        method_dir = os.path.join(param_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Create dataframe to store ratio data for plotting
        ratio_data = []

        # Collect data for each parameter and cluster
        for sort_value, param, directories, label in sorted_params:
            exp_results = results["experiments"].get((method, param))
            if exp_results is None:
                continue

            cluster_ratios = exp_results["cluster_ratios"]

            # Calculate mean and std for each cluster
            for cluster_idx, cluster_name in enumerate(ref_names):
                mean_ratio = np.mean(cluster_ratios[:, cluster_idx])
                std_ratio = np.std(cluster_ratios[:, cluster_idx])

                ratio_data.append(
                    {
                        "Parameter": label,
                        "Sort Value": sort_value,
                        "Cluster": cluster_name,
                        "Mean Ratio": mean_ratio,
                        "Std Ratio": std_ratio,
                    }
                )

        if not ratio_data:
            continue

        # Convert to DataFrame
        df = pd.DataFrame(ratio_data)

        # Create line plot of ratio vs parameter
        fig, ax = plt.subplots(figsize=(12, 8))

        for cluster_name in ref_names:
            cluster_df = df[df["Cluster"] == cluster_name]
            # Sort by parameter value
            cluster_df = cluster_df.sort_values("Sort Value")

            ax.errorbar(
                cluster_df["Parameter"],
                cluster_df["Mean Ratio"],
                yerr=cluster_df["Std Ratio"],
                marker="o",
                linestyle="-",
                label=f"{cluster_name} State",
            )

        for cluster_idx, ratio in enumerate(true_ratios):
            ax.axhline(
                ratio,
                color=["blue", "orange"][cluster_idx],
                linestyle="--",
                label=f"True {ref_names[cluster_idx]} ratio",
            )

        ax.set_xlabel("Convergence Threshold")
        ax.set_ylabel("State Ratio")
        ax.set_title(f"Average State Ratio by Parameter ({method})")
        ax.legend()
        if reverse_xscale:
            plt.gca().invert_xaxis()  # Reverse the x-axis (Noise SD)

        ax.grid(True)
        ax.set_ylim(0, 1)

        # Adjust x-tick labels for readability
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        fig.savefig(
            os.path.join(method_dir, "state_ratio_by_parameter.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

        # Create a stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Pivot data for stacked bars
        pivot_df = df.pivot(index="Parameter", columns="Cluster", values="Mean Ratio")

        # Sort rows by Sort Value
        sort_order = df.drop_duplicates("Parameter").sort_values("Sort Value")["Parameter"].tolist()
        pivot_df = pivot_df.reindex(sort_order)

        # Create stacked bar chart
        pivot_df.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        if reverse_xscale:
            plt.gca().invert_xaxis()  # Reverse the x-axis (Noise SD)

        ax.set_xlabel("Convergence Threshold")
        ax.set_ylabel("State Ratio")
        ax.set_title(f"Average State Composition by Parameter ({method})")
        ax.legend(title="State")
        ax.grid(True, axis="y")

        # Adjust x-tick labels for readability
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        fig.savefig(
            os.path.join(method_dir, "state_composition_by_parameter.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Saved parameter comparison ratio plots to {param_dir}")


def plot_parameter_comparison_evolution(results, output_dir):
    """
    Create plots showing the evolution of state ratios for different parameter values.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_experiments
    output_dir : str
        Directory to save output plots
    """
    ref_names = results["ref_names"]
    cluster_assignments = results["cluster_assignments"]

    # Create a directory for parameter comparison plots
    param_dir = os.path.join(output_dir, "parameter_comparison")
    os.makedirs(param_dir, exist_ok=True)

    # Process each method separately
    for method, sorted_params in results["sorted_params_by_method"]:
        if not sorted_params:
            continue

        method_dir = os.path.join(param_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Create a figure for each reference/cluster
        for cluster_idx, cluster_name in enumerate(ref_names):
            fig, ax = plt.subplots(figsize=(12, 8))

            # Use a colormap for different parameter values
            n_params = len(sorted_params)
            cmap = plt.cm.viridis
            colors = [cmap(i) for i in np.linspace(0, 1, n_params)]

            # Process each parameter
            for i, (sort_value, param, directories, label) in enumerate(sorted_params):
                exp_results = results["experiments"].get((method, param))
                if exp_results is None:
                    continue

                interval_states = exp_results["interval_states"]
                if not interval_states:
                    continue

                n_intervals = len(interval_states)
                n_frames = len(cluster_assignments)

                # Calculate state ratios at each interval
                interval_ratios = np.zeros(n_intervals)

                for interval_idx, states in enumerate(interval_states):
                    # Extract weights from all states at this interval
                    all_ratios = []

                    for state in states:
                        weights = np.array(state.params.frame_weights).flatten()
                        if len(weights) > n_frames:
                            weights = weights[:n_frames]
                        weights = weights / np.sum(weights)

                        # Calculate ratio for this cluster
                        mask = cluster_assignments == cluster_idx
                        all_ratios.append(np.sum(weights[mask]))

                    # Average ratios across all seeds
                    interval_ratios[interval_idx] = np.mean(all_ratios)

                # Use first history to get step numbers
                steps = [states[0].step for states in interval_states]

                # Plot ratio evolution for this parameter
                ax.plot(steps, interval_ratios, "o-", linewidth=2, label=label, color=colors[i])

            ax.set_xlabel("Optimization Step")
            ax.set_ylabel("State Ratio")
            ax.set_title(f"Evolution of {cluster_name} State Ratio by Parameter ({method})")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True)

            # Set y-axis limits with some padding
            ax.set_ylim(0, 1)

            plt.tight_layout()
            fig.savefig(
                os.path.join(
                    method_dir, f"{cluster_name.lower()}_ratio_evolution_by_parameter.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    print(f"Saved parameter comparison evolution plots to {param_dir}")


def plot_summary_heatmap(results, output_dir):
    """
    Create a heatmap showing the state ratios across different parameter values.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_experiments
    output_dir : str
        Directory to save output plots
    """
    ref_names = results["ref_names"]

    # Create a directory for summary plots
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Process each method separately
    for method, sorted_params in results["sorted_params_by_method"]:
        if not sorted_params:
            continue

        method_dir = os.path.join(summary_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Collect data for heatmap
        param_labels = []
        ratios = []

        for sort_value, param, directories, label in sorted_params:
            exp_results = results["experiments"].get((method, param))
            if exp_results is None:
                continue

            cluster_ratios = exp_results["cluster_ratios"]

            # Calculate mean for each cluster
            mean_ratios = np.mean(cluster_ratios, axis=0)
            ratios.append(mean_ratios)
            param_labels.append(label)

        if not ratios:
            continue

        # Convert to arrays
        ratios = np.array(ratios)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, len(param_labels) * 0.5 + 2))

        im = ax.imshow(ratios, aspect="auto", cmap="viridis", vmin=0, vmax=1)

        # Add text annotations
        for i in range(len(param_labels)):
            for j in range(len(ref_names)):
                text = ax.text(
                    j,
                    i,
                    f"{ratios[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="w" if ratios[i, j] > 0.5 else "k",
                )

        # Set tick labels
        ax.set_xticks(np.arange(len(ref_names)))
        ax.set_yticks(np.arange(len(param_labels)))
        ax.set_xticklabels(ref_names)
        ax.set_yticklabels(param_labels)

        # Rotate x tick labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("State Ratio", rotation=-90, va="bottom")

        ax.set_title(f"State Ratio Heatmap by Parameter ({method})")

        plt.tight_layout()
        fig.savefig(
            os.path.join(method_dir, "state_ratio_heatmap.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"Saved summary heatmaps to {summary_dir}")


def cluster_and_select_representative_structures(
    weights_dict,
    topology_path,
    trajectory_path,
    output_dir,
    n_components=10,
    n_clusters=50,
    n_top=20,
):
    """
    Cluster the MD trajectory in PCA space and select representative structures
    based on summed weights across seeds.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary with weights for each method, parameter, and seed
    topology_path : str
        Path to topology file
    trajectory_path : str
        Path to trajectory file
    output_dir : str
        Directory to save output trajectories
    n_components : int
        Number of PCA components to use
    n_clusters : int
        Number of clusters for K-means
    n_top : int
        Number of top clusters to include

    Returns:
    --------
    cluster_results : dict
        Dictionary with cluster information and visualizations
    """
    import os

    import MDAnalysis as mda
    import numpy as np
    from MDAnalysis.analysis import align
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    cluster_results = {}

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load MD universe
        md_universe = mda.Universe(topology_path, trajectory_path)
        n_frames = md_universe.trajectory.n_frames

        # Select CA atoms for alignment
        ca_atoms = md_universe.select_atoms("name CA")

        # Align trajectory to the first frame using CA atoms
        print("Aligning trajectory...")
        align.AlignTraj(
            md_universe, md_universe, select="name CA", in_memory=True, weights="mass", ref_frame=0
        ).run()

        # Extract pairwise distances for each frame
        print(f"Extracting pairwise distances for {n_frames} frames...")
        pairwise_distances = []
        for ts in md_universe.trajectory:
            coords = ca_atoms.positions
            pairwise_dists = pdist(coords)
            pairwise_distances.append(pairwise_dists)

        pairwise_distances = np.array(pairwise_distances)

        # Perform PCA with n_components
        print(f"Performing PCA with {n_components} components...")
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(pairwise_distances)

        # Perform K-means clustering
        print(f"Clustering with K-means ({n_clusters} clusters)...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_data)

        # Create a mapping from cluster to frames
        cluster_to_frames = {}
        for i in range(n_clusters):
            cluster_to_frames[i] = np.where(cluster_labels == i)[0]

        # Process each method and parameter
        for method, param_dict in weights_dict.items():
            cluster_results[method] = {}

            for param, seed_dict in param_dict.items():
                # Skip if no valid seeds
                valid_seeds = {k: v for k, v in seed_dict.items() if v is not None}
                if not valid_seeds:
                    print(f"Skipping {method}, param={param}: no valid seeds")
                    continue

                # Initialize cluster weights
                cluster_weights = np.zeros(n_clusters)

                # Sum weights for each cluster across seeds
                for seed_key, weights in valid_seeds.items():
                    # Ensure weights array length matches frames
                    if len(weights) > n_frames:
                        weights = weights[:n_frames]
                    elif len(weights) < n_frames:
                        # Pad weights
                        padding = np.zeros(n_frames - len(weights))
                        weights = np.concatenate([weights, padding])
                        weights = weights / np.sum(weights)

                    # Accumulate weights for each cluster
                    for cluster_idx in range(n_clusters):
                        cluster_frames = cluster_to_frames[cluster_idx]
                        cluster_weights[cluster_idx] += np.sum(weights[cluster_frames])

                # Normalize cluster weights
                if np.sum(cluster_weights) > 0:
                    cluster_weights = cluster_weights / np.sum(cluster_weights)

                # Store cluster weights
                cluster_results[method][param] = {
                    "cluster_weights": cluster_weights,
                    "cluster_to_frames": cluster_to_frames,
                }

                # Select top n_top clusters
                top_clusters = np.argsort(cluster_weights)[-n_top:]

                # Define output trajectory name
                if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                    gamma, exponent = param
                    output_name = f"{method}_gamma{gamma}_exp{exponent}_cluster{n_top}.pdb"
                elif method == "HDXer":
                    output_name = f"{method}_gamma{param}_cluster{n_top}.pdb"
                else:
                    output_name = f"{method}_{param}_cluster{n_top}.pdb"

                # Full path to output trajectory
                output_traj_path = os.path.join(output_dir, output_name)

                # Create trajectory with representative structures
                try:
                    with mda.coordinates.PDB.PDBWriter(output_traj_path, multiframe=True) as writer:
                        # Write each frame in order of descending cluster weight
                        for cluster_idx in sorted(
                            top_clusters, key=lambda x: cluster_weights[x], reverse=True
                        ):
                            # Find the representative frame for this cluster
                            cluster_frames = cluster_to_frames[cluster_idx]

                            # If the cluster has frames, use the one closest to centroid
                            if len(cluster_frames) > 0:
                                # Calculate distances to centroid in PCA space
                                cluster_center = kmeans.cluster_centers_[cluster_idx]
                                frames_pca = pca_data[cluster_frames]

                                # Find index of frame closest to centroid
                                distances = np.linalg.norm(frames_pca - cluster_center, axis=1)
                                closest_idx = cluster_frames[np.argmin(distances)]

                                # Write this frame to trajectory
                                md_universe.trajectory[closest_idx]
                                writer.write(md_universe.atoms)

                    print(
                        f"Wrote {n_top} representative structures for {method}, param={param} to {output_traj_path}"
                    )

                    # Store information about selected frames
                    cluster_results[method][param]["top_clusters"] = top_clusters
                    cluster_results[method][param]["representative_frames"] = [
                        cluster_to_frames[cluster_idx][
                            np.argmin(
                                np.linalg.norm(
                                    pca_data[cluster_to_frames[cluster_idx]]
                                    - kmeans.cluster_centers_[cluster_idx],
                                    axis=1,
                                )
                            )
                        ]
                        if len(cluster_to_frames[cluster_idx]) > 0
                        else None
                        for cluster_idx in top_clusters
                    ]

                except Exception as e:
                    print(f"Error writing trajectory for {method}, param={param}: {e}")

        # Add general clustering information to results
        cluster_results["clustering_info"] = {
            "pca": pca,
            "kmeans": kmeans,
            "pca_data": pca_data,
            "cluster_labels": cluster_labels,
        }

    except Exception as e:
        print(f"Error in cluster_and_select_representative_structures: {e}")

    return cluster_results


def plot_clustered_dataset(cluster_results, output_dir=None):
    """
    Visualize the clustered dataset in 2D PCA space, coloring points by cluster.

    Parameters:
    -----------
    cluster_results : dict
        Results from the clustering function
    output_dir : str, optional
        Directory to save the plot

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract clustering information
    if "clustering_info" not in cluster_results:
        print("No clustering information found")
        return None

    clustering_info = cluster_results["clustering_info"]
    pca_data = clustering_info["pca_data"]
    cluster_labels = clustering_info["cluster_labels"]
    kmeans = clustering_info["kmeans"]

    # Create a 2D visualization using the first two PCA components
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a colormap for the clusters
    cmap = plt.cm.get_cmap("tab20", np.max(cluster_labels) + 1)

    # Plot each point, colored by cluster
    scatter = ax.scatter(
        pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap=cmap, alpha=0.5, s=10
    )

    # Plot cluster centers
    centers = kmeans.cluster_centers_
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=200,
        marker="*",
        c=range(len(centers)),
        cmap=cmap,
        edgecolors="k",
        linewidths=1,
    )

    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Clustered Trajectory in PCA Space")

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster")

    # Save figure if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "clustered_trajectory.png")
        fig.savefig(output_path, dpi=300)
        print(f"Clustered trajectory plot saved to {output_path}")

    return fig


def compute_cluster_w1_distances(
    cluster_results, weights_dict, reference_path, topology_path, trajectory_path
):
    """
    Compute W1 distances between reference structure and representative structures
    from the clustering approach.

    Parameters:
    -----------
    cluster_results : dict
        Results from the clustering function
    weights_dict : dict
        Original weights dictionary
    reference_path : str
        Path to reference PDB file
    topology_path : str
        Path to topology file
    trajectory_path : str
        Path to trajectory file

    Returns:
    --------
    w1_results : dict
        Dictionary with W1 distances for clustered structures
    """
    import MDAnalysis as mda
    import numpy as np
    from scipy.spatial.distance import pdist
    from scipy.stats import wasserstein_distance

    w1_results = {}

    try:
        # Load reference structure
        ref_universe = mda.Universe(reference_path)
        ref_ca = ref_universe.select_atoms("name CA")
        n_ref_frames = ref_universe.trajectory.n_frames

        # Extract pairwise distances from reference
        ref_pairwise_distances = []
        for ts in ref_universe.trajectory:
            coords = ref_ca.positions
            pairwise_dists = pdist(coords)
            ref_pairwise_distances.append(pairwise_dists)

        ref_pairwise_distances = np.array(ref_pairwise_distances)

        # Reference weights (uniform)
        ref_weights = np.ones(n_ref_frames) / n_ref_frames

        # Compute average pairwise distances for reference
        ref_avg_pairwise = np.average(ref_pairwise_distances, axis=0, weights=ref_weights)

        # Load MD universe
        md_universe = mda.Universe(topology_path, trajectory_path)
        md_ca = md_universe.select_atoms("name CA")

        # Extract pairwise distances from MD trajectory
        md_pairwise_distances = []
        for ts in md_universe.trajectory:
            coords = md_ca.positions
            pairwise_dists = pdist(coords)
            md_pairwise_distances.append(pairwise_dists)

        md_pairwise_distances = np.array(md_pairwise_distances)

        # Process each method and parameter
        for method, method_results in cluster_results.items():
            if method == "clustering_info":
                continue

            w1_results[method] = {}

            for param, param_results in method_results.items():
                if (
                    "top_clusters" not in param_results
                    or "representative_frames" not in param_results
                ):
                    continue

                # Get representative frames for this method/parameter
                rep_frames = param_results["representative_frames"]
                rep_frames = [f for f in rep_frames if f is not None]

                if not rep_frames:
                    continue

                # Create binary weights (1 for representative frames, 0 for others)
                n_frames = len(md_pairwise_distances)
                weights = np.zeros(n_frames)
                weights[rep_frames] = 1.0 / len(rep_frames)

                # Compute W1 distance
                try:
                    # Calculate weighted average for MD
                    md_avg_pairwise = np.average(md_pairwise_distances, axis=0, weights=weights)

                    # Calculate Wasserstein distance
                    w1_dist = wasserstein_distance(ref_avg_pairwise, md_avg_pairwise)

                    # Store result
                    w1_results[method][param] = [
                        w1_dist
                    ]  # As a list for compatibility with boxplot function
                except Exception as e:
                    print(f"Error computing W1 distance for {method}, param={param}: {e}")
                    w1_results[method][param] = []

    except Exception as e:
        print(f"Error in compute_cluster_w1_distances: {e}")

    return w1_results


def plot_cluster_distribution(cluster_results, output_dir=None):
    """
    Create bar charts showing the distribution of clusters for each method and parameter.

    Parameters:
    -----------
    cluster_results : dict
        Results from the clustering function
    output_dir : str, optional
        Directory to save the plots

    Returns:
    --------
    figs : dict
        Dictionary of generated figures
    """
    import os

    import matplotlib.pyplot as plt

    figs = {}

    # Skip clustering_info
    methods = [m for m in cluster_results.keys() if m != "clustering_info"]

    if not methods:
        print("No methods found in cluster results")
        return figs

    # Process each method
    for method in methods:
        method_results = cluster_results[method]
        params = list(method_results.keys())

        # Skip if no parameters
        if not params:
            continue

        # Create a figure for this method
        n_params = len(params)
        fig, axes = plt.subplots(
            n_params, 1, figsize=(12, 4 * n_params), squeeze=False, sharex=True
        )

        axes = axes.flatten()

        # Process each parameter
        for i, param in enumerate(params):
            param_results = method_results[param]

            if "cluster_weights" not in param_results:
                continue

            # Get cluster weights
            cluster_weights = param_results["cluster_weights"]

            # Get top clusters if available
            top_clusters = param_results.get("top_clusters", [])

            # Create bar chart
            ax = axes[i]
            bars = ax.bar(range(len(cluster_weights)), cluster_weights, alpha=0.7)

            # Highlight top clusters if available
            if len(top_clusters) > 0:
                for cluster_idx in top_clusters:
                    bars[cluster_idx].set_color("red")
                    bars[cluster_idx].set_alpha(1.0)

            # Add grid and labels
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Cluster Index")
            ax.set_ylabel("Normalized Weight")

            # Set title based on parameter type
            if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                gamma, exponent = param
                title = f"{method}: γ={gamma}×10^{exponent}"
            elif method == "HDXer":
                title = f"{method}: γ={param}"
            else:
                title = f"{method}: {param}"

            ax.set_title(title)

        # Set overall title
        fig.suptitle(f"Cluster Weight Distribution - {method}", fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"cluster_distribution_{method}.png")
            fig.savefig(output_path, dpi=300)
            print(f"Cluster distribution plot for {method} saved to {output_path}")

        # Store figure
        figs[method] = fig

    return figs


def run_clustering_analysis(
    weights_dict,
    reference_path,
    topology_path,
    trajectory_path,
    output_dir,
    n_components=10,
    n_clusters=50,
    n_top=20,
):
    """
    Run the complete clustering analysis workflow.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary with weights for each method, parameter, and seed
    reference_path : str
        Path to reference PDB file
    topology_path : str
        Path to topology file
    trajectory_path : str
        Path to trajectory file
    output_dir : str
        Directory to save output files and plots
    n_components : int
        Number of PCA components to use
    n_clusters : int
        Number of clusters for K-means
    n_top : int
        Number of top clusters to select

    Returns:
    --------
    results : dict
        Dictionary with all analysis results
    """
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 1. Perform clustering and select representative structures
    print("Performing clustering and selecting representative structures...")
    cluster_results = cluster_and_select_representative_structures(
        weights_dict,
        topology_path,
        trajectory_path,
        os.path.join(output_dir, "cluster_trajectories"),
        n_components=n_components,
        n_clusters=n_clusters,
        n_top=n_top,
    )

    # 2. Visualize the clustered dataset
    print("Visualizing clustered dataset...")
    cluster_plot = plot_clustered_dataset(cluster_results, output_dir=output_dir)

    # 3. Compute W1 distances for clustered structures
    print("Computing W1 distances for clustered structures...")
    w1_results = compute_cluster_w1_distances(
        cluster_results, weights_dict, reference_path, topology_path, trajectory_path
    )

    # 4. Visualize W1 distances
    print("Visualizing W1 distances...")
    w1_plot = plot_boxplots(
        w1_results,
        metric_name="Cluster W1 Distance",
        title="Wasserstein-1 Distance to Reference using Clustered Structures",
        ylabel="Wasserstein-1 Distance (Clustered Structures, Å)",
    )

    # Save W1 plot
    w1_plot_path = os.path.join(output_dir, "cluster_w1_distance.png")
    w1_plot.savefig(w1_plot_path, dpi=300)
    print(f"Cluster W1 distance plot saved to {w1_plot_path}")

    # 5. Create cluster distribution plots
    print("Creating cluster distribution plots...")
    distribution_plots = plot_cluster_distribution(cluster_results, output_dir=output_dir)

    # Return all results
    return {
        "cluster_results": cluster_results,
        "w1_results": w1_results,
        "plots": {
            "cluster_plot": cluster_plot,
            "w1_plot": w1_plot,
            "distribution_plots": distribution_plots,
        },
    }


# Main function to run the script
def main():
    # Set up paths - these should be adjusted based on the actual file locations
    base_dir = (
        "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/convergence_l2long_adam_mcmse"
    )
    open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
    closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
    topology_path = open_path
    trajectory_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"

    # Reference paths
    reference_paths = [open_path, closed_path]

    # Create output directory for cross-experiment analysis
    output_dir = os.path.join(base_dir, "_cross_experiment_analysis_convergence_l2long_adam_mcmse")
    os.makedirs(output_dir, exist_ok=True)

    # Run analysis across all experiments
    results = analyze_experiments(base_dir, reference_paths, topology_path, trajectory_path)

    if results is None:
        print("Analysis failed. Exiting.")
        return

    # Create parameter comparison plots
    print("\nCreating parameter comparison plots...")
    plot_parameter_comparison_rmsd(results, output_dir)
    plot_parameter_comparison_ratio(results, output_dir, reverse_xscale=True)
    plot_parameter_comparison_evolution(results, output_dir)

    # Create summary plots
    print("\nCreating summary plots...")
    plot_summary_heatmap(results, output_dir)

    # print("\nAnalysis completed successfully!")
    # print(f"All cross-experiment plots saved to {output_dir}")

    # # Create cross-experiment clustering analysis
    # print("\n==== Running Cross-Experiment Clustering Analysis ====\n")
    # cross_clustering_dir = os.path.join(output_dir, "cross_clustering")
    # os.makedirs(cross_clustering_dir, exist_ok=True)

    # # Collect weights from all experiments
    # cross_experiment_weights = {}
    # for method, sorted_params in results["sorted_params_by_method"]:
    #     cross_experiment_weights[method] = {}

    #     for _, param, directories, _ in sorted_params:
    #         cross_experiment_weights[method][param] = {}

    #         # Get weights from results
    #         for exp_key, exp_results in results["experiments"].items():
    #             if exp_key == (method, param):
    #                 histories = exp_results["histories"]

    #                 # Extract weights from each history (seed)
    #                 for i, history in enumerate(histories):
    #                     seed_key = f"seed_{i}"
    #                     final_state = history.states[-1]
    #                     weights = np.array(final_state.params.frame_weights).flatten()

    #                     # Normalize weights
    #                     if len(weights) > 0:
    #                         weights = weights / np.sum(weights)
    #                         cross_experiment_weights[method][param][seed_key] = weights

    # # Run cross-experiment clustering analysis
    # cross_clustering_results = run_clustering_analysis(
    #     cross_experiment_weights,
    #     reference_paths[0],  # Use open state as reference
    #     topology_path,
    #     trajectory_path,
    #     cross_clustering_dir,
    #     n_components=10,
    #     n_clusters=100,
    #     n_top=2,
    # )


if __name__ == "__main__":
    main()
