"""


class BV_uptake_ForwardPass(
    ForwardPass[BV_input_features, uptake_BV_output_features, BV_Model_Parameters]
):
    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> uptake_BV_output_features:
        # Extract model parameters
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # Convert inputs to JAX arrays
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        kints = jnp.asarray(input_features.k_ints)
        time_points = parameters.timepoints.reshape(-1)

        # Compute protection factors
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)

        # Get original shape before any reshaping
        original_shape = log_pf.shape

        # Extract the number of residues (should match kints length)
        n_residues = kints.shape[0]

        # Ensure pf has proper shape - don't flatten it
        pf = jnp.exp(log_pf)

        # Vectorized computation of uptake for each timepoint
        def compute_uptake_for_timepoint(timepoint):
            # If pf is multi-dimensional (residues, frames)
            if len(original_shape) > 1:
                # Reshape kints to allow broadcasting across frames
                k = kints.reshape(n_residues, 1)
                # Calculate uptake for each residue at this timepoint
                # Broadcasting: k is (n_residues, 1), pf is (n_residues, n_frames)
                uptake = 1 - jnp.exp(-k * timepoint / pf)
            else:
                # For single-dimensional pf (just residues)
                uptake = 1 - jnp.exp(-kints * timepoint / pf)

            return uptake

        # Compute uptake for each timepoint
        uptake_per_timepoint = jax.vmap(compute_uptake_for_timepoint)(time_points)

        # Return the list of timepoint-wise residue-wise uptake arrays
        return uptake_BV_output_features(uptake_per_timepoint)

"""

import itertools
import json
import os

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from MDAnalysis import Universe
from MDAnalysis.analysis import align
from scipy.spatial.distance import pdist
from scipy.stats import entropy, wasserstein_distance

from jaxent.featurise import run_featurise
from jaxent.interfaces.builder import Experiment_Builder
from jaxent.models.config import BV_model_Config
from jaxent.models.HDX.BV.features import (
    BV_input_features,
    uptake_BV_output_features,
)
from jaxent.models.HDX.BV.forwardmodel import BV_model
from jaxent.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.models.HDX.forward import BV_ForwardPass
from jaxent.types.base import ForwardPass
from jaxent.types.config import FeaturiserSettings
from jaxent.utils.hdf import load_optimization_history_from_file

# globally set axes/tick/legend font‐sizes
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


class BV_uptake_ForwardPass(  # this is a modified version that can operate over features with multiple dimensions
    ForwardPass[BV_input_features, uptake_BV_output_features, BV_Model_Parameters]
):
    def __call__(
        self, input_features: BV_input_features, parameters: BV_Model_Parameters
    ) -> uptake_BV_output_features:
        # Extract model parameters
        bc, bh = parameters.bv_bc, parameters.bv_bh
        # Convert inputs to JAX arrays
        heavy_contacts = jnp.asarray(input_features.heavy_contacts)
        acceptor_contacts = jnp.asarray(input_features.acceptor_contacts)
        kints = jnp.asarray(input_features.k_ints)
        time_points = parameters.timepoints.reshape(-1)

        # Compute protection factors
        log_pf = (bc * heavy_contacts) + (bh * acceptor_contacts)

        # Get original shape before any reshaping
        original_shape = log_pf.shape

        # Extract the number of residues (should match kints length)
        n_residues = kints.shape[0]

        # Ensure pf has proper shape - don't flatten it
        pf = jnp.exp(log_pf)

        # Vectorized computation of uptake for each timepoint
        def compute_uptake_for_timepoint(timepoint):
            # If pf is multi-dimensional (residues, frames)
            if len(original_shape) > 1:
                # Reshape kints to allow broadcasting across frames
                k = kints.reshape(n_residues, 1)
                # Calculate uptake for each residue at this timepoint
                # Broadcasting: k is (n_residues, 1), pf is (n_residues, n_frames)
                uptake = 1 - jnp.exp(-k * timepoint / pf)
            else:
                # For single-dimensional pf (just residues)
                uptake = 1 - jnp.exp(-kints * timepoint / pf)

            return uptake

        # Compute uptake for each timepoint
        uptake_per_timepoint = jax.vmap(compute_uptake_for_timepoint)(time_points)

        # Return the list of timepoint-wise residue-wise uptake arrays
        return uptake_BV_output_features(uptake_per_timepoint)


def compute_pf_correlation(avg_pf_dict, exp_residues, exp_log_pfs, feature_topology):
    """
    Compute Pearson correlation coefficient between predicted and experimental protection factors.

    Parameters:
    -----------
    avg_pf_dict : dict
        Dictionary with ensemble-average protection factors for each method, parameter, and seed
    exp_residues : numpy.ndarray
        Experimental residue numbers
    exp_log_pfs : numpy.ndarray
        Experimental log protection factors
    feature_topology : list
        Topology information for the features

    Returns:
    --------
    corr_results : dict
        Dictionary with correlation values grouped by method and parameter
    """
    from scipy.stats import pearsonr

    corr_results = {}

    # Create mapping from residue number to feature index
    # Only consider features that represent single residues (not peptides)
    residue_to_index = {}
    for i, topology in enumerate(feature_topology):
        if topology.residue_end is None or topology.residue_end == topology.residue_start:
            # Single residue
            residue_to_index[topology.residue_start] = i

    # Extract indices for experimental residues
    valid_exp_indices = []
    valid_exp_residues = []
    valid_exp_log_pfs = []

    for i, residue in enumerate(exp_residues):
        if residue in residue_to_index:
            valid_exp_indices.append(residue_to_index[residue])
            valid_exp_residues.append(residue)
            valid_exp_log_pfs.append(exp_log_pfs[i])

    if len(valid_exp_indices) == 0:
        print("Error: No matching residues found between experimental data and feature topology")
        return corr_results

    # Process each method
    for method, param_dict in avg_pf_dict.items():
        corr_results[method] = {}

        # Process each parameter
        for param, seed_dict in param_dict.items():
            corr_values = []

            # Process each seed
            for seed_key, avg_pf in seed_dict.items():
                if avg_pf is None:
                    continue

                # Extract predicted protection factors for the matching residues
                pred_log_pfs = np.array([avg_pf[idx] for idx in valid_exp_indices])

                # Compute Pearson correlation coefficient
                try:
                    corr, _ = pearsonr(pred_log_pfs, valid_exp_log_pfs)
                    corr_values.append(corr)
                except:
                    # Handle case where correlation can't be computed
                    print(
                        f"Warning: Could not compute correlation for {method}, param={param}, seed={seed_key}"
                    )

            corr_results[method][param] = corr_values

    return corr_results


def analyze_protection_factors(
    weights_dict, reference_path, topology_path, trajectory_path, hdx_nmr_pf_path, output_dir
):
    """
    Analyze protection factors by comparing predicted and experimental values.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary with weights for each method, parameter, and seed
    reference_path : str
        Path to reference PDB file
    topology_path : str
        Path to MD topology file
    trajectory_path : str
        Path to trajectory file
    hdx_nmr_pf_path : str
        Path to HDX NMR protection factors .dat file
    output_dir : str
        Directory to save output files and plots

    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Compute protection factors
    print("Computing protection factors...")
    log_pfs, feature_topology = compute_protection_factors(
        topology_path, trajectory_path, output_dir
    )

    # Load experimental protection factors
    print("Loading experimental protection factors...")
    exp_residues, exp_log_pfs = load_hdx_nmr_pf(hdx_nmr_pf_path)

    # Compute ensemble-average protection factors
    print("Computing ensemble-average protection factors...")
    avg_pf_dict = compute_ensemble_average_pf(log_pfs, weights_dict)

    # Compute MSE
    print("Computing MSE for protection factors...")
    mse_results = compute_pf_mse(avg_pf_dict, exp_residues, exp_log_pfs, feature_topology)

    # Create and save MSE visualization
    print("Creating protection factor MSE plot...")
    fig_mse = plot_boxplots(
        mse_results,
        metric_name="Protection Factor MSE",
        title="Mean Squared Error between Predicted and Experimental Protection Factors",
        ylabel="Mean Squared Error (Protection Factors)",
    )

    # Save MSE plot
    output_path_mse = os.path.join(output_dir, "pf_mse_comparison.png")
    fig_mse.savefig(output_path_mse, dpi=300)
    print(f"Protection factor MSE plot saved to {output_path_mse}")

    # Compute correlation coefficients
    print("Computing correlation for protection factors...")
    corr_results = compute_pf_correlation(avg_pf_dict, exp_residues, exp_log_pfs, feature_topology)

    # Create and save correlation visualization
    print("Creating protection factor correlation plot...")
    fig_corr = plot_boxplots(
        corr_results,
        metric_name="Protection Factor Correlation",
        title="Pearson Correlation between Predicted and Experimental Protection Factors",
        ylabel="Pearson Correlation Coefficient",
    )

    # Save correlation plot
    output_path_corr = os.path.join(output_dir, "pf_correlation_comparison.png")
    fig_corr.savefig(output_path_corr, dpi=300)
    print(f"Protection factor correlation plot saved to {output_path_corr}")

    return {
        "log_pfs": log_pfs,
        "feature_topology": feature_topology,
        "exp_residues": exp_residues,
        "exp_log_pfs": exp_log_pfs,
        "avg_pf_dict": avg_pf_dict,
        "mse_results": mse_results,
        "fig_mse": fig_mse,
        "corr_results": corr_results,
        "fig_corr": fig_corr,
    }


def select_representative_parameter(
    x_results,
    y_results,
    average_method="mean",  # Options: "mean" or "median"
    representative_method="euclidean_median",  # Options: "euclidean_median", "euclidian_minimum", or "local_density"
    conservative_choice="higher",  # Options: "higher" or "lower"
    method_specific_conservative={"HDXer": "lower"},  # Method-specific overrides
):
    """
    Select a representative parameter for each method based on scatter plot metrics.

    Parameters:
    -----------
    x_results : dict
        Dictionary with x-axis metric values grouped by method and parameter
    y_results : dict
        Dictionary with y-axis metric values grouped by method and parameter
    average_method : str
        Method for averaging across seeds/datasets ("mean" or "median")
    representative_method : str
        Method for selecting representative parameter:
        - "euclidean_median": parameter with minimum sum of distances to all other parameters
        - "euclidian_minimum": parameter with minimum distance from origin (0,0)
        - "local_density": parameter with highest local density
    conservative_choice : str
        Default choice when multiple parameters are equally good ("higher" or "lower")
    method_specific_conservative : dict
        Method-specific overrides for conservative_choice

    Returns:
    --------
    selected_params : dict
        Dictionary with selected parameter for each method
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import gaussian_kde

    selected_params = {}

    # Process each method
    for method in x_results.keys():
        if method not in y_results:
            continue

        # Get parameters common to both x and y results
        common_params = [
            param
            for param in x_results[method].keys()
            if param in y_results[method] and x_results[method][param] and y_results[method][param]
        ]

        if not common_params:
            continue

        # Compute average values for each parameter
        param_avg_values = []
        for param in common_params:
            x_values = x_results[method][param]
            y_values = y_results[method][param]

            # Match up data points by length
            min_len = min(len(x_values), len(y_values))
            if min_len == 0:
                continue

            if average_method == "mean":
                x_avg = np.mean(x_values[:min_len])
                y_avg = np.mean(y_values[:min_len])
            else:  # median
                x_avg = np.median(x_values[:min_len])
                y_avg = np.median(y_values[:min_len])

            param_avg_values.append((param, x_avg, y_avg))

        if not param_avg_values:
            continue

        # Convert to arrays for easier computation
        params = [p[0] for p in param_avg_values]
        avg_points = np.array([[p[1], p[2]] for p in param_avg_values])

        # Select representative parameter
        if representative_method == "euclidean_median":
            # Compute pairwise distances
            dist_matrix = squareform(pdist(avg_points))
            # Sum distances for each point
            total_distances = np.sum(dist_matrix, axis=1)
            # Find point(s) with minimum total distance
            min_dist_indices = np.where(total_distances == np.min(total_distances))[0]
            representative_indices = min_dist_indices

        elif representative_method == "euclidian_minimum":
            # Compute Euclidean distance from origin (0,0) for each point
            # This finds the point closest to the ideal minimum values on both axes
            distances_from_origin = np.sqrt(np.sum(avg_points**2, axis=1))
            min_dist_indices = np.where(distances_from_origin == np.min(distances_from_origin))[0]
            representative_indices = min_dist_indices

        else:  # local_density
            # Estimate kernel density
            if len(avg_points) > 3:  # Need enough points for KDE
                kde = gaussian_kde(avg_points.T)
                densities = kde(avg_points.T)
                # Find point(s) with maximum density
                max_density_indices = np.where(densities == np.max(densities))[0]
                representative_indices = max_density_indices
            else:
                # Fall back to euclidean median for small datasets
                dist_matrix = squareform(pdist(avg_points))
                total_distances = np.sum(dist_matrix, axis=1)
                representative_indices = np.where(total_distances == np.min(total_distances))[0]

        # If multiple representative indices, choose based on conservative_choice
        if len(representative_indices) > 1:
            # Use method-specific override if available
            method_conservative = method_specific_conservative.get(method, conservative_choice)

            # Get parameter values for sorting
            rep_params = [params[i] for i in representative_indices]

            # Sort parameters based on their numerical value
            if method == "HDXer" and any(isinstance(p, tuple) and len(p) == 2 for p in rep_params):
                # For HDXer with (gamma, exponent) tuples
                param_values = []
                for param in rep_params:
                    if isinstance(param, tuple) and len(param) == 2:
                        gamma, exponent = param
                        value = gamma * (10**exponent)
                    else:
                        value = param
                    param_values.append(value)

                sorted_indices = np.argsort(param_values)
            else:
                # For other methods, sort directly
                sorted_indices = np.argsort(rep_params)

            # Select based on conservative choice
            if method_conservative == "higher":
                selected_idx = sorted_indices[-1]  # highest value
            else:  # lower
                selected_idx = sorted_indices[0]  # lowest value

            representative_idx = representative_indices[selected_idx]
        else:
            representative_idx = representative_indices[0]

        # Store the selected parameter
        selected_params[method] = params[representative_idx]

    return selected_params


def plot_scatter(
    x_results,
    y_results,
    x_label,
    y_label,
    title=None,
    method_names=None,
    select_param=True,
    average_method="mean",
    representative_method="euclidean_median",
    conservative_choice="higher",
    method_specific_conservative={"HDXer": "lower"},
):
    """
    Create scatter plot comparing two metrics across different methods and parameters.
    Each method has its own subplot with different color scales, markers indicate seeds.
    Includes dashed lines for mean and median across seeds, and optionally selects and displays
    a representative parameter.

    Parameters:
    -----------
    x_results : dict
        Dictionary with x-axis metric values grouped by method and parameter
    y_results : dict
        Dictionary with y-axis metric values grouped by method and parameter
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    title : str, optional
        Plot title
    method_names : list, optional
        List of method names to include
    select_param : bool, optional
        Whether to select and display a representative parameter
    average_method : str, optional
        Method for averaging across seeds/datasets ("mean" or "median")
    representative_method : str, optional
        Method for selecting representative parameter ("euclidean_median" or "local_density")
    conservative_choice : str, optional
        Default choice when multiple parameters are equally good ("higher" or "lower")
    method_specific_conservative : dict, optional
        Method-specific overrides for conservative_choice

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The scatter plot figure
    selected_params : dict
        Dictionary with selected parameter for each method (if select_param=True)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Extract methods with valid data in both x and y results
    valid_methods = [
        method
        for method in x_results.keys()
        if method in y_results
        and any(x_results[method].values())
        and any(y_results[method].values())
    ]

    if not valid_methods:
        # Create a simple figure if no valid data
        fig = plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "No valid data available for scatter plot",
            ha="center",
            va="center",
            fontsize=14,
        )
        plt.tight_layout()
        return (fig, {}) if select_param else fig

    # Call our function to select representative parameters
    selected_params = {}
    if select_param:
        selected_params = select_representative_parameter(
            x_results,
            y_results,
            average_method=average_method,
            representative_method=representative_method,
            conservative_choice=conservative_choice,
            method_specific_conservative=method_specific_conservative,
        )

    # Create figure with subplots (one for each method)
    fig, axes = plt.subplots(
        1,
        len(valid_methods),
        figsize=(7 * len(valid_methods), 7),
        sharey=True,  # Share y-axis across all panels
        sharex=True,  # Share x-axis across all panels
        gridspec_kw={"wspace": 0.2},  # Adjust space between subplots
    )

    # Handle single subplot case
    if len(valid_methods) == 1:
        axes = [axes]

    # Define marker styles for seeds
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

    # Define different colormaps for each method
    method_colormaps = ["viridis", "plasma", "magma", "cividis", "inferno"]

    # Process each method in its own subplot
    for method_idx, method in enumerate(valid_methods):
        ax = axes[method_idx]

        # Select colormap for this method
        cmap_name = method_colormaps[method_idx % len(method_colormaps)]
        cmap = plt.cm.get_cmap(cmap_name)

        # Get parameters common to both x and y results and sort them
        common_params = [
            param
            for param in x_results[method].keys()
            if param in y_results[method] and x_results[method][param] and y_results[method][param]
        ]

        # Sort parameters for consistent coloring
        sorted_params = []
        for param in common_params:
            # Calculate sort value based on parameter type
            if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                gamma, exponent = param
                # Convert to actual numerical value for sorting
                sort_value = gamma * (10**exponent)
            elif method == "HDXer":
                sort_value = param
            else:
                # For non-HDXer methods, treat param as alpha value
                sort_value = param

            sorted_params.append((sort_value, param))

        # Sort parameters by their numerical value
        sorted_params.sort(key=lambda x: x[0])
        sorted_common_params = [p[1] for p in sorted_params]
        param_sort_values = [p[0] for p in sorted_params]

        # Define colormap normalization for this method's parameters
        norm = Normalize(
            vmin=min(param_sort_values) if param_sort_values else 0,
            vmax=max(param_sort_values) if param_sort_values else 1,
        )

        # Arrays to store median and mean calculation data
        param_x_means = []
        param_y_means = []
        param_x_medians = []
        param_y_medians = []
        param_colors = []
        param_labels = []

        # Process each parameter with its color from the method's colormap
        for param_idx, param in enumerate(sorted_common_params):
            # Get sort value for this parameter (for color mapping)
            sort_value = sorted_params[param_idx][0]
            color = cmap(norm(sort_value))

            # Store parameter color for legend
            param_colors.append(color)

            # Get x and y values for this method and parameter
            x_values = x_results[method][param]
            y_values = y_results[method][param]

            # Match up data points by length (assuming they're from the same seeds)
            min_len = min(len(x_values), len(y_values))
            if min_len == 0:
                continue

            # Format parameter label based on type
            if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                gamma, exponent = param
                param_label = f"γ={gamma}×10^{exponent}"
            elif method == "HDXer":
                param_label = f"γ={param}"
            else:
                # For non-HDXer methods, treat param as alpha value
                if isinstance(param, float):
                    # Handle the case where param is exactly 0
                    if param == 0:
                        param_label = "α=0"
                    else:
                        # Extract mantissa and exponent
                        exponent = int(np.floor(np.log10(abs(param))))
                        mantissa = param / (10**exponent)

                        # Format using standard form
                        if exponent == 0:
                            param_label = f"α={mantissa:.1f}"
                        else:
                            param_label = f"α={mantissa:.1f}×10^{exponent}"
                else:
                    param_label = f"α={param}"

            param_labels.append(param_label)

            # Calculate mean and median for this parameter (across seeds)
            param_x_mean = np.mean(x_values[:min_len])
            param_y_mean = np.mean(y_values[:min_len])
            param_x_median = np.median(x_values[:min_len])
            param_y_median = np.median(y_values[:min_len])

            param_x_means.append(param_x_mean)
            param_y_means.append(param_y_mean)
            param_x_medians.append(param_x_median)
            param_y_medians.append(param_y_median)

            # Plot each seed with a different marker but same parameter color
            for seed_idx in range(min_len):
                marker = markers[seed_idx % len(markers)]

                # Include seed in label only for the first parameter to avoid legend clutter
                if param_idx == 0:
                    seed_label = f"Seed {seed_idx}"
                else:
                    seed_label = "_nolegend_"  # Underscore prefix hides from legend

                # Check if this parameter is the selected one
                is_selected = (
                    select_param and method in selected_params and param == selected_params[method]
                )
                edge_color = "red" if is_selected else None
                line_width = 2 if is_selected else None

                # Plot individual data point
                ax.scatter(
                    x_values[seed_idx],
                    y_values[seed_idx],
                    color=color,
                    marker=marker,
                    s=100,  # Larger marker size
                    alpha=0.8,
                    label=seed_label,
                    edgecolor=edge_color,
                    linewidth=line_width,
                )

        # Plot mean and median trend lines
        # Connect means across parameters with dashed line
        if param_x_means:
            # Sort points for line plotting to ensure proper connection
            points = sorted(zip(param_x_means, param_y_means, param_colors))
            sorted_x_means, sorted_y_means, sorted_colors = zip(*points)

            # Plot mean line
            ax.plot(sorted_x_means, sorted_y_means, "k--", label="Mean across seeds", alpha=0.7)

            # Plot individual mean points
            for x, y, color in zip(sorted_x_means, sorted_y_means, sorted_colors):
                ax.scatter(
                    x, y, color=color, marker="X", s=120, edgecolor="black", linewidth=2, zorder=10
                )

        # Connect medians across parameters with dotted line
        if param_x_medians:
            # Sort points for line plotting
            points = sorted(zip(param_x_medians, param_y_medians, param_colors))
            sorted_x_medians, sorted_y_medians, sorted_colors = zip(*points)

            # Plot median line
            ax.plot(
                sorted_x_medians, sorted_y_medians, "k:", label="Median across seeds", alpha=0.7
            )

            # Plot individual median points
            for x, y, color in zip(sorted_x_medians, sorted_y_medians, sorted_colors):
                ax.scatter(
                    x, y, color=color, marker="+", s=120, edgecolor="black", linewidth=2, zorder=10
                )

        # Add text for selected parameter
        if select_param and method in selected_params:
            selected_param = selected_params[method]

            # Format parameter label based on type
            if method == "HDXer" and isinstance(selected_param, tuple) and len(selected_param) == 2:
                gamma, exponent = selected_param
                param_label = f"Selected: γ={gamma}×10^{exponent}"
            elif method == "HDXer":
                param_label = f"Selected: γ={selected_param}"
            else:
                # For non-HDXer methods, treat param as alpha value
                if isinstance(selected_param, float):
                    # Handle the case where param is exactly 0
                    if selected_param == 0:
                        param_label = "Selected: α=0"
                    else:
                        # Extract mantissa and exponent
                        exponent = int(np.floor(np.log10(abs(selected_param))))
                        mantissa = selected_param / (10**exponent)

                        # Format using standard form
                        if exponent == 0:
                            param_label = f"Selected: α={mantissa:.1f}"
                        else:
                            param_label = f"Selected: α={mantissa:.1f}×10^{exponent}"
                else:
                    param_label = f"Selected: α={selected_param}"

            # Add text to subplot
            ax.text(
                0.05,
                0.95,
                param_label,
                transform=ax.transAxes,
                fontsize=12,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Create colorbar for this method
        if param_sort_values:
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)

            # Set colorbar label based on method
            if method == "HDXer":
                cbar.set_label("γ parameter")
            else:
                cbar.set_label("α parameter")

        # Set title, grid, and labels
        ax.set_title(method)
        ax.grid(True, alpha=0.3)

        # Only add y-label to the first subplot
        if method_idx == 0:
            ax.set_ylabel(y_label)

        # Add x-label to all subplots
        ax.set_xlabel(x_label)

        # Create legend for seeds
        handles, labels = ax.get_legend_handles_labels()
        seed_handles = [h for h, l in zip(handles, labels) if "Seed" in l]
        seed_labels = [l for l in labels if "Seed" in l]

        if seed_handles:
            seed_legend = ax.legend(
                seed_handles,
                seed_labels,
                title="Seeds",
                loc="upper right",
                # bbox_to_anchor=(0.05, 0.95),
            )
            ax.add_artist(seed_legend)

        # Add mean/median legend at the bottom
        line_handles = [
            plt.Line2D([0], [0], color="k", linestyle="--", alpha=0.7),
            plt.Line2D([0], [0], color="k", linestyle=":", alpha=0.7),
        ]
        line_labels = ["Mean", "Median"]
        line_legend = ax.legend(line_handles, line_labels, loc="lower right", title="Trends")
        ax.add_artist(line_legend)

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for the suptitle
    else:
        plt.tight_layout()

    return (fig, selected_params) if select_param else fig


def load_hdx_nmr_pf(hdx_nmr_pf_path):
    """
    Load HDX NMR protection factors from a .dat file.

    Parameters:
    -----------
    hdx_nmr_pf_path : str
        Path to the HDX NMR protection factors .dat file

    Returns:
    --------
    residues : numpy.ndarray
        Residue numbers
    log_pfs : numpy.ndarray
        Log protection factors
    """
    data = np.loadtxt(hdx_nmr_pf_path)
    residues = data[:, 0].astype(int)
    log_pfs = data[:, 1]
    return residues, log_pfs


def compute_protection_factors(topology_path, trajectory_path, output_dir=None):
    """
    Compute residue-wise protection factors for each frame in a trajectory.

    Parameters:
    -----------
    topology_path : str
        Path to topology file
    trajectory_path : str
        Path to trajectory file
    output_dir : str, optional
        Directory to save/load cached features

    Returns:
    --------
    log_pfs : numpy.ndarray
        Protection factors for each frame and residue
    feature_topology : list
        Topology information for the features
    """
    # Determine cache paths
    if output_dir is None:
        output_dir = os.path.dirname(trajectory_path)
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "features.jpz.npz")
    topology_path_json = os.path.join(output_dir, "topology.json")

    # Check if cached files exist
    features_exist = os.path.exists(features_path)
    topology_exist = os.path.exists(topology_path_json)

    # Load or compute features and topology
    if features_exist and topology_exist:
        print(f"Loading cached features and topology from {output_dir}")
        # Load features
        loaded_features_dict = jnp.load(features_path)
        features = [
            BV_input_features(
                heavy_contacts=loaded_features_dict["heavy_contacts"],
                acceptor_contacts=loaded_features_dict["acceptor_contacts"],
                k_ints=loaded_features_dict["k_ints"],
            )
        ]

        # Load topology
        with open(topology_path_json, "r") as f:
            topology_dicts = json.load(f)

        from jaxent.interfaces.topology import Partial_Topology

        feature_topology = [
            [
                Partial_Topology(
                    chain=top_dict["chain"],
                    fragment_sequence=top_dict["fragment_sequence"],
                    residue_start=top_dict["residue_start"],
                    residue_end=top_dict["residue_end"],
                    peptide_trim=top_dict["peptide_trim"],
                    fragment_index=top_dict["fragment_index"],
                )
                for top_dict in topology_dicts
            ]
        ]

        print(f"Loaded features shape: {features[0].features_shape}")
        print(f"Loaded topology count: {len(feature_topology[0])}")
    else:
        print("Computing features and topology...")
        # Setup for featurization

        # Setup for featurization
        featuriser_settings = FeaturiserSettings(name="protection_factors", batch_size=None)
        bv_config = BV_model_Config(num_timepoints=3)
        models = [BV_model(bv_config)]

        # Create ensemble using the trajectory
        ensemble = Experiment_Builder([Universe(topology_path, trajectory_path)], models)
        features, feature_topology = run_featurise(ensemble, featuriser_settings)

        print(f"Computed features shape: {features[0].features_shape}")

        # Save features
        print(f"Saving features to {features_path}")
        jnp.savez(
            os.path.join(output_dir, "features.jpz"),
            heavy_contacts=features[0].heavy_contacts,
            acceptor_contacts=features[0].acceptor_contacts,
            k_ints=features[0].k_ints,
        )

        # Save topology
        def topology_to_dict(topology):
            return {
                "chain": topology.chain,
                "fragment_sequence": topology.fragment_sequence,
                "residue_start": int(topology.residue_start),
                "residue_end": int(topology.residue_end)
                if topology.residue_end is not None
                else None,
                "peptide_trim": int(topology.peptide_trim),
                "fragment_index": int(topology.fragment_index)
                if topology.fragment_index is not None
                else None,
                "length": int(topology.length) if topology.length is not None else None,
            }

        topology_dicts = [topology_to_dict(top) for top in feature_topology[0]]

        print(f"Saving topology to {topology_path_json}")
        with open(topology_path_json, "w") as f:
            json.dump(topology_dicts, f, indent=2)

    # Compute protection factors using the BV_ForwardPass
    forward_pass = BV_ForwardPass()

    parameters = BV_Model_Parameters()

    # Compute protection factors for each frame
    output_features = forward_pass(features[0], parameters)

    # Return protection factors and topology
    return output_features.log_Pf, feature_topology[0]


def load_experimental_uptake(segs_path, dfrac_path):
    """
    Load experimental deuterium uptake data from files.

    Parameters:
    -----------
    segs_path : str
        Path to the residue segments file
    dfrac_path : str
        Path to the experimental deuterium fractions file

    Returns:
    --------
    residues : list
        List of residue numbers
    uptake_values : list
        List of uptake values for each residue at multiple timepoints
    """
    import numpy as np

    # Load residue segments
    with open(segs_path, "r") as f:
        segs_text = [line.strip() for line in f.readlines()]
        segs = [line.split() for line in segs_text]

    # Extract residue numbers
    residues = [int(seg[1]) for seg in segs]

    # Load deuterium fractions
    with open(dfrac_path, "r") as f:
        # Skip first line (header) and read the rest
        dfrac_text = [line.strip() for line in f.readlines()[1:]]
        dfracs = [line.split() for line in dfrac_text]

    # Convert to float arrays
    uptake_values = [np.array(line, dtype=float) for line in dfracs]

    return residues, uptake_values


def compute_ensemble_average_pf(log_pfs, weights_dict):
    """
    Compute ensemble-average protection factors using weights from different methods.

    Parameters:
    -----------
    log_pfs : numpy.ndarray
        Protection factors for each frame and residue
    weights_dict : dict
        Dictionary with weights for each method, parameter, and seed

    Returns:
    --------
    avg_pf_dict : dict
        Dictionary with ensemble-average protection factors for each method, parameter, and seed
    """
    avg_pf_dict = {}

    for method, param_dict in weights_dict.items():
        avg_pf_dict[method] = {}

        for param, seed_dict in param_dict.items():
            avg_pf_dict[method][param] = {}

            for seed_key, weights in seed_dict.items():
                if weights is None:
                    raise ValueError(
                        f"Missing weights for method {method}, parameter {param}, seed {seed_key}"
                    )

                # assert that weights sum is close to 1
                assert np.isclose(np.sum(weights), 1.0), (
                    f"Weights for method {method}, parameter {param}, seed {seed_key} do not sum to 1. "
                    f"Sum: {np.sum(weights)}"
                )
                weights = weights.reshape(-1)
                # print(log_pfs.shape)
                # print(weights.shape)
                # reshape to 1D
                # Compute weighted average of protection factors
                avg_pf = np.average(log_pfs, axis=1, weights=weights)
                avg_pf_dict[method][param][seed_key] = avg_pf
    # breakpoint()
    return avg_pf_dict


def compute_pf_mse(avg_pf_dict, exp_residues, exp_log_pfs, feature_topology):
    """
    Compute Mean Squared Error between predicted and experimental protection factors.

    Parameters:
    -----------
    avg_pf_dict : dict
        Dictionary with ensemble-average protection factors for each method, parameter, and seed
    exp_residues : numpy.ndarray
        Experimental residue numbers
    exp_log_pfs : numpy.ndarray
        Experimental log protection factors
    feature_topology : list
        Topology information for the features

    Returns:
    --------
    mse_results : dict
        Dictionary with MSE values grouped by method and parameter
    """
    mse_results = {}

    # Print some information about the feature topology
    print("Feature topology information:")
    for i, top in enumerate(feature_topology[:5]):  # Print just first 5 to avoid clutter
        print(
            f"Index {i}: Residue {top.residue_start}-{top.residue_end}, Sequence {top.fragment_sequence}"
        )
    print("...")

    # Create mapping from residue number to feature index
    # Only consider features that represent single residues (not peptides)
    residue_to_index = {}
    for i, topology in enumerate(feature_topology):
        if topology.residue_end is None or topology.residue_end == topology.residue_start:
            # Single residue
            residue_to_index[topology.residue_start] = i

    # Extract indices for experimental residues
    valid_exp_indices = []
    valid_exp_residues = []
    valid_exp_log_pfs = []

    for i, residue in enumerate(exp_residues):
        if residue in residue_to_index:
            valid_exp_indices.append(residue_to_index[residue])
            valid_exp_residues.append(residue)
            valid_exp_log_pfs.append(exp_log_pfs[i])
        else:
            print(f"Warning: Residue {residue} not found in feature topology")

    if len(valid_exp_indices) == 0:
        print("Error: No matching residues found between experimental data and feature topology")
        return mse_results

    print(f"Found {len(valid_exp_indices)} matching residues: {valid_exp_residues}")

    # Process each method
    for method, param_dict in avg_pf_dict.items():
        mse_results[method] = {}

        # Process each parameter
        for param, seed_dict in param_dict.items():
            mse_values = []

            # Process each seed
            for seed_key, avg_pf in seed_dict.items():
                if avg_pf is None:
                    continue

                # Extract predicted protection factors for the matching residues
                pred_log_pfs = np.array([avg_pf[idx] for idx in valid_exp_indices])

                # Compute MSE
                mse = np.mean((pred_log_pfs - valid_exp_log_pfs) ** 2)
                mse_values.append(mse)

            mse_results[method][param] = mse_values

    return mse_results


def compute_uptake_mse(
    weights_dict, exp_residues, exp_uptake, features, feature_topology, parameters=None
):
    """
    Compute Mean Squared Error between predicted and experimental deuterium uptake.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary with weights for each method, parameter, and seed
    exp_residues : list
        List of experimental residue numbers
    exp_uptake : list
        List of experimental uptake values for each residue
    features : BV_input_features
        Input features for the BV model
    feature_topology : list
        Topology information for the features
    parameters : BV_Model_Parameters, optional
        Model parameters, if None, default parameters will be used

    Returns:
    --------
    mse_results : dict
        Dictionary with MSE values grouped by method and parameter
    """
    import numpy as np

    from jaxent.models.HDX.BV.parameters import BV_Model_Parameters

    # Initialize BV_uptake_ForwardPass
    forward_pass = BV_uptake_ForwardPass()

    # Use default parameters if none provided
    if parameters is None:
        parameters = BV_Model_Parameters()

    # Initialize results dictionary
    mse_results = {}

    # Print some information about the data
    print(f"Experimental data: {len(exp_residues)} residues with uptake measurements")

    # Create mapping from residue number to feature index
    residue_to_index = {}
    for i, topology in enumerate(feature_topology):
        if topology.residue_end is None or topology.residue_end == topology.residue_start:
            # Single residue
            residue_to_index[topology.residue_start] = i

    # Extract indices for experimental residues
    valid_exp_indices = []
    valid_exp_residues = []
    valid_exp_uptake = []

    for i, residue in enumerate(exp_residues):
        if residue in residue_to_index:
            valid_exp_indices.append(residue_to_index[residue])
            valid_exp_residues.append(residue)
            valid_exp_uptake.append(exp_uptake[i])
        else:
            print(f"Warning: Residue {residue} not found in feature topology")

    if len(valid_exp_indices) == 0:
        print("Error: No matching residues found between experimental data and feature topology")
        return {}

    print(f"Found {len(valid_exp_indices)} matching residues between experimental data and model")

    # Process each method
    for method, param_dict in weights_dict.items():
        mse_results[method] = {}

        # Process each parameter
        for param, seed_dict in param_dict.items():
            mse_values = []

            # Process each seed
            for seed_key, weights in seed_dict.items():
                if weights is None:
                    continue

                # Make sure weights array length matches the number of frames
                n_frames = features.features_shape[2]
                if len(weights) > n_frames:
                    weights = weights[:n_frames]
                elif len(weights) < n_frames:
                    # Pad weights with zeros if needed
                    padding = np.zeros(n_frames - len(weights))
                    weights = np.concatenate([weights, padding])
                    # Renormalize
                    weights = weights / np.sum(weights)

                # Compute uptake for all frames using the BV_uptake_ForwardPass
                output_features = forward_pass(features, parameters)

                # Extract uptake values - shape should be (n_timepoints, n_residues, n_frames)
                uptake_values = output_features.uptake

                # Compute weighted average uptake for each residue and timepoint
                # Average over frames (last dimension) using weights
                avg_uptake = np.zeros((uptake_values.shape[0], uptake_values.shape[1]))
                for t in range(uptake_values.shape[0]):  # For each timepoint
                    for r in range(uptake_values.shape[1]):  # For each residue
                        avg_uptake[t, r] = np.average(uptake_values[t, r, :], weights=weights)

                # Extract predicted uptake for the matching residues
                pred_uptake = np.array([avg_uptake[:, idx] for idx in valid_exp_indices])

                # Compute MSE for each residue and take the average
                mse_values_per_residue = []
                for i in range(len(valid_exp_indices)):
                    # Calculate MSE for each residue across timepoints
                    mse_per_residue = np.mean((pred_uptake[i] - valid_exp_uptake[i]) ** 2)
                    mse_values_per_residue.append(mse_per_residue)

                # Average MSE across all residues
                avg_mse = np.mean(mse_values_per_residue)
                mse_values.append(avg_mse)

            mse_results[method][param] = mse_values

    return mse_results


# def analyze_protection_factors(
#     weights_dict, reference_path, topology_path, trajectory_path, hdx_nmr_pf_path, output_dir
# ):
#     """
#     Analyze protection factors by comparing predicted and experimental values.

#     Parameters:
#     -----------
#     weights_dict : dict
#         Dictionary with weights for each method, parameter, and seed
#     reference_path : str
#         Path to reference PDB file
#     topology_path : str
#         Path to MD topology file
#     trajectory_path : str
#         Path to trajectory file
#     hdx_nmr_pf_path : str
#         Path to HDX NMR protection factors .dat file
#     output_dir : str
#         Directory to save output files and plots

#     Returns:
#     --------
#     results : dict
#         Dictionary with analysis results
#     """
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)

#     # Compute protection factors
#     print("Computing protection factors...")
#     log_pfs, feature_topology = compute_protection_factors(
#         topology_path, trajectory_path, output_dir
#     )

#     # Load experimental protection factors
#     print("Loading experimental protection factors...")
#     exp_residues, exp_log_pfs = load_hdx_nmr_pf(hdx_nmr_pf_path)

#     # Compute ensemble-average protection factors
#     print("Computing ensemble-average protection factors...")
#     avg_pf_dict = compute_ensemble_average_pf(log_pfs, weights_dict)

#     # Compute MSE
#     print("Computing MSE for protection factors...")
#     mse_results = compute_pf_mse(avg_pf_dict, exp_residues, exp_log_pfs, feature_topology)

#     # Create and save MSE visualization
#     print("Creating protection factor MSE plot...")
#     fig_mse = plot_boxplots(
#         mse_results,
#         metric_name="Protection Factor MSE",
#         title="Mean Squared Error between Predicted and Experimental Protection Factors",
#         ylabel="Mean Squared Error (Protection Factors)",
#     )

#     # Save MSE plot
#     output_path_mse = os.path.join(output_dir, "pf_mse_comparison.png")
#     fig_mse.savefig(output_path_mse, dpi=300)
#     print(f"Protection factor MSE plot saved to {output_path_mse}")

#     return {
#         "log_pfs": log_pfs,
#         "feature_topology": feature_topology,
#         "exp_residues": exp_residues,
#         "exp_log_pfs": exp_log_pfs,
#         "avg_pf_dict": avg_pf_dict,
#         "mse_results": mse_results,
#         "fig_mse": fig_mse,
#     }


def analyze_deuterium_uptake(
    weights_dict,
    reference_path,
    topology_path,
    trajectory_path,
    segs_path,
    dfrac_path,
    output_dir,
    features=None,
    feature_topology=None,
):
    """
    Analyze deuterium uptake by comparing predicted and experimental values.

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
    segs_path : str
        Path to the residue segments file
    dfrac_path : str
        Path to the experimental deuterium fractions file
    output_dir : str
        Directory to save output files and plots
    features : BV_input_features, optional
        Pre-computed features, if available
    feature_topology : list, optional
        Pre-computed feature topology, if available

    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    import json
    import os

    import jax.numpy as jnp
    from MDAnalysis import Universe

    from jaxent.interfaces.builder import Experiment_Builder
    from jaxent.models.config import BV_model_Config
    from jaxent.models.HDX.BV.features import BV_input_features
    from jaxent.models.HDX.BV.forwardmodel import BV_model
    from jaxent.types.config import FeaturiserSettings

    k_ints_path = "/Users/alexi/JAX-ENT/tests/inst/BPTI_Intrinsic_rates.dat"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # Load experimental uptake data
    print("Loading experimental uptake data...")
    exp_residues, exp_uptake = load_experimental_uptake(segs_path, dfrac_path)

    # open as csv - first line is header
    with open(k_ints_path, "r") as f:
        k_ints_text = [line.strip() for line in f.readlines()[1:]]
        k_ints = [line.split() for line in k_ints_text]
    # create dictionary

    k_int_dict = {int(k_int[0]): float(k_int[1]) for k_int in k_ints}

    print(len(k_int_dict))

    # filter out the residues that are not in the features
    feat_res = [res for res in exp_residues]

    filtered_k_int_dict = {res: k_int_dict[res] for res in k_int_dict if res in feat_res}

    print(len(filtered_k_int_dict))
    filtered_k_ints = jnp.array([filtered_k_int_dict[res] for res in feat_res])
    k_ints = jnp.asarray(list(k_int_dict.values()))
    print(filtered_k_ints)
    print(k_ints)

    if features is None or feature_topology is None:
        # Set up paths for cached features
        features_path = os.path.join(output_dir, "features.jpz.npz")
        topology_path_json = os.path.join(output_dir, "topology.json")

        # Check if cached files exist
        features_exist = os.path.exists(features_path)
        topology_exist = os.path.exists(topology_path_json)

        # Load or compute features and topology
        if features_exist and topology_exist:
            print(f"Loading cached features and topology from {output_dir}")
            # Load features
            loaded_features_dict = jnp.load(features_path)
            features = BV_input_features(
                heavy_contacts=loaded_features_dict["heavy_contacts"],
                acceptor_contacts=loaded_features_dict["acceptor_contacts"],
                k_ints=k_ints,
            )
            print(features)
            # Load topology
            with open(topology_path_json, "r") as f:
                topology_dicts = json.load(f)

            from jaxent.interfaces.topology import Partial_Topology

            feature_topology = [
                Partial_Topology(
                    chain=top_dict["chain"],
                    fragment_sequence=top_dict["fragment_sequence"],
                    residue_start=top_dict["residue_start"],
                    residue_end=top_dict["residue_end"],
                    peptide_trim=top_dict["peptide_trim"],
                    fragment_index=top_dict["fragment_index"],
                )
                for top_dict in topology_dicts
            ]

            print(f"Loaded features shape: {features.features_shape}")
            print(f"Loaded topology count: {len(feature_topology)}")
        else:
            print("Computing features and topology...")
            # Setup for featurization
            featuriser_settings = FeaturiserSettings(name="protection_factors", batch_size=None)
            bv_config = BV_model_Config(num_timepoints=1)
            models = [BV_model(bv_config)]

            # Create ensemble using the trajectory
            ensemble = Experiment_Builder([Universe(topology_path, trajectory_path)], models)
            features_list, feature_topology_list = run_featurise(ensemble, featuriser_settings)
            features = BV_input_features(
                heavy_contacts=features_list[0].heavy_contacts,
                acceptor_contacts=features_list[0].heavy_contacts,
                k_ints=k_ints,
            )
            feature_topology = feature_topology_list[0]

            print(f"Computed features shape: {features.features_shape}")

            # Save features
            print(f"Saving features to {features_path}")
            jnp.savez(
                os.path.join(output_dir, "features.jpz"),
                heavy_contacts=features.heavy_contacts,
                acceptor_contacts=features.acceptor_contacts,
                k_ints=k_ints,
            )

            # Save topology
            def topology_to_dict(topology):
                return {
                    "chain": topology.chain,
                    "fragment_sequence": topology.fragment_sequence,
                    "residue_start": int(topology.residue_start),
                    "residue_end": int(topology.residue_end)
                    if topology.residue_end is not None
                    else None,
                    "peptide_trim": int(topology.peptide_trim),
                    "fragment_index": int(topology.fragment_index)
                    if topology.fragment_index is not None
                    else None,
                    "length": int(topology.length)
                    if hasattr(topology, "length") and topology.length is not None
                    else None,
                }

            topology_dicts = [topology_to_dict(top) for top in feature_topology]

            print(f"Saving topology to {topology_path_json}")
            with open(topology_path_json, "w") as f:
                json.dump(topology_dicts, f, indent=2)
    print(features)

    # Compute MSE for deuterium uptake
    print("Computing MSE for deuterium uptake...")
    mse_results = compute_uptake_mse(
        weights_dict, exp_residues, exp_uptake, features, feature_topology
    )

    # Create and save MSE visualization
    print("Creating deuterium uptake MSE plot...")
    fig_mse = plot_boxplots(
        mse_results,
        metric_name="Deuterium Uptake MSE",
        title="Mean Squared Error between Predicted and Experimental Deuterium Uptake",
        ylabel="Mean Squared Error (Deuterium Uptake)",
    )

    # Save MSE plot
    output_path_mse = os.path.join(output_dir, "deuterium_uptake_mse_comparison.png")
    fig_mse.savefig(output_path_mse, dpi=300)
    print(f"Deuterium uptake MSE plot saved to {output_path_mse}")

    # Still compute MAE between weighted and unweighted uptake
    print("Computing MAE between weighted and unweighted uptake...")
    mae_results = compute_uptake_mae(weights_dict, features)

    # Create and save MAE visualization
    print("Creating deuterium uptake MAE plot...")
    fig_mae = plot_boxplots(
        mae_results,
        metric_name="Deuterium Uptake MAE",
        title="Mean Absolute Error between Weighted and Unweighted Deuterium Uptake",
        ylabel="Mean Absolute Error (Deuterium Uptake)",
    )

    # Save MAE plot
    output_path_mae = os.path.join(output_dir, "deuterium_uptake_mae_comparison.png")
    fig_mae.savefig(output_path_mae, dpi=300)
    print(f"Deuterium uptake MAE plot saved to {output_path_mae}")

    return {
        "mse_results": mse_results,
        "fig_mse": fig_mse,
        "mae_results": mae_results,
        "fig_mae": fig_mae,
    }


def create_comparison_scatter_plots(
    uniform_kl_results,
    mae_results,
    mse_results,  # Changed from w1_results to mse_results
    output_dir=None,
    select_param=True,
    average_method="mean",
    representative_method="local_density",
    conservative_choice="higher",
    method_specific_conservative={"HDXer": "lower"},
):
    """
    Create scatter plots comparing different metrics and optionally select representative parameters.

    Parameters:
    -----------
    uniform_kl_results : dict
        Dictionary with KL divergence from uniform values
    mae_results : dict
        Dictionary with MAE values for deuterium uptake
    mse_results : dict
        Dictionary with MSE values for deuterium uptake (changed from W1 distance)
    output_dir : str, optional
        Directory to save output plots
    select_param : bool, optional
        Whether to select and display representative parameters
    average_method : str, optional
        Method for averaging across seeds/datasets ("mean" or "median")
    representative_method : str, optional
        Method for selecting representative parameter ("euclidean_median" or "local_density")
    conservative_choice : str, optional
        Default choice when multiple parameters are equally good ("higher" or "lower")
    method_specific_conservative : dict, optional
        Method-specific overrides for conservative_choice

    Returns:
    --------
    results : dict
        Dictionary with generated figures and selected parameters
    """
    import os

    # Initialize dictionary for results
    results = {"figs": {}, "selected_params": {}}

    # 1. Plot uptake MAE vs weights KLD against uniform
    if select_param:
        fig1, selected_params1 = plot_scatter(
            uniform_kl_results,
            mae_results,
            x_label="KL Divergence from Uniform",
            y_label="Deuterium Uptake MAE",
            title="Deuterium Uptake MAE vs. KL Divergence from Uniform",
            select_param=True,
            average_method=average_method,
            representative_method=representative_method,
            conservative_choice=conservative_choice,
            method_specific_conservative=method_specific_conservative,
        )
        results["selected_params"]["mae_vs_kld"] = selected_params1
    else:
        fig1 = plot_scatter(
            uniform_kl_results,
            mae_results,
            x_label="KL Divergence from Uniform",
            y_label="Deuterium Uptake MAE",
            title="Deuterium Uptake MAE vs. KL Divergence from Uniform",
            select_param=False,
        )

    results["figs"]["mae_vs_kld"] = fig1

    # 2. Plot MSE uptake vs uptake MAE (changed from W1 to MSE)
    if select_param:
        fig2, selected_params2 = plot_scatter(
            mae_results,
            mse_results,  # Changed from w1_results to mse_results
            x_label="Deuterium Uptake MAE",
            y_label="Mean Squared Error (Deuterium Uptake)",  # Changed label
            title="Mean Squared Error vs. MAE for Deuterium Uptake",  # Changed title
            select_param=True,
            average_method=average_method,
            representative_method=representative_method,
            conservative_choice=conservative_choice,
            method_specific_conservative=method_specific_conservative,
        )
        results["selected_params"]["mse_vs_mae"] = selected_params2  # Changed key
    else:
        fig2 = plot_scatter(
            mae_results,
            mse_results,  # Changed from w1_results to mse_results
            x_label="Deuterium Uptake MAE",
            y_label="Mean Squared Error (Deuterium Uptake)",  # Changed label
            title="Mean Squared Error vs. MAE for Deuterium Uptake",  # Changed title
            select_param=False,
        )

    results["figs"]["mse_vs_mae"] = fig2  # Changed key

    # 3. Plot MSE uptake vs weights KLD against uniform (changed from W1 to MSE)
    if select_param:
        fig3, selected_params3 = plot_scatter(
            uniform_kl_results,
            mse_results,  # Changed from w1_results to mse_results
            x_label="KL Divergence from Uniform",
            y_label="Mean Squared Error (Deuterium Uptake)",  # Changed label
            title="Mean Squared Error vs. KL Divergence from Uniform",  # Changed title
            select_param=True,
            average_method=average_method,
            representative_method=representative_method,
            conservative_choice=conservative_choice,
            method_specific_conservative=method_specific_conservative,
        )
        results["selected_params"]["mse_vs_kld"] = selected_params3  # Changed key
    else:
        fig3 = plot_scatter(
            uniform_kl_results,
            mse_results,  # Changed from w1_results to mse_results
            x_label="KL Divergence from Uniform",
            y_label="Mean Squared Error (Deuterium Uptake)",  # Changed label
            title="Mean Squared Error vs. KL Divergence from Uniform",  # Changed title
            select_param=False,
        )

    results["figs"]["mse_vs_kld"] = fig3  # Changed key

    # Save figures if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Save each figure
        fig1.savefig(os.path.join(output_dir, "mae_vs_kld_scatter.png"), dpi=300)
        fig2.savefig(
            os.path.join(output_dir, "mse_vs_mae_scatter.png"), dpi=300
        )  # Changed filename
        fig3.savefig(
            os.path.join(output_dir, "mse_vs_kld_scatter.png"), dpi=300
        )  # Changed filename

        # If parameters were selected, save them to a JSON file
        if select_param:
            import json

            # Convert any non-serializable objects (like tuples) to strings
            json_friendly_params = {}
            for plot_type, params in results["selected_params"].items():
                json_friendly_params[plot_type] = {}
                for method, param in params.items():
                    if isinstance(param, tuple):
                        json_friendly_params[plot_type][method] = str(param)
                    else:
                        json_friendly_params[plot_type][method] = param

            with open(os.path.join(output_dir, "selected_parameters.json"), "w") as f:
                json.dump(json_friendly_params, f, indent=4)

    return results


def compute_uptake_mae(weights_dict, features, parameters=None):
    """
    Compute Mean Absolute Error (MAE) between weighted and unweighted deuterium uptake.

    Parameters:
    -----------
    weights_dict : dict
        Dictionary with weights for each method, parameter, and seed
    features : BV_input_features
        Input features for the BV model
    parameters : BV_Model_Parameters, optional
        Model parameters, if None, default parameters will be used

    Returns:
    --------
    mae_results : dict
        Dictionary with MAE values grouped by method and parameter
    """
    import numpy as np

    from jaxent.models.HDX.BV.parameters import BV_Model_Parameters

    # Initialize BV_uptake_ForwardPass
    forward_pass = BV_uptake_ForwardPass()

    # Use default parameters if none provided
    if parameters is None:
        parameters = BV_Model_Parameters()

    # Initialize results dictionary
    mae_results = {}

    # Compute uptake for all frames using the BV_uptake_ForwardPass
    output_features = forward_pass(features, parameters)

    # Extract uptake values - shape should be (n_timepoints, n_residues, n_frames)
    uptake_values = output_features.uptake

    # Compute unweighted (uniform) uptake (average across frames)
    n_frames = uptake_values.shape[2]
    uniform_weights = np.ones(n_frames) / n_frames

    # Calculate unweighted uptake for each residue and timepoint
    unweighted_uptake = np.zeros((uptake_values.shape[0], uptake_values.shape[1]))
    for t in range(uptake_values.shape[0]):  # For each timepoint
        for r in range(uptake_values.shape[1]):  # For each residue
            unweighted_uptake[t, r] = np.average(uptake_values[t, r, :], weights=uniform_weights)

    # Process each method
    for method, param_dict in weights_dict.items():
        mae_results[method] = {}

        # Process each parameter
        for param, seed_dict in param_dict.items():
            mae_values = []

            # Process each seed
            for seed_key, weights in seed_dict.items():
                if weights is None:
                    continue

                # Make sure weights array length matches the number of frames
                n_frames = features.features_shape[2]
                if len(weights) > n_frames:
                    weights = weights[:n_frames]
                elif len(weights) < n_frames:
                    # Pad weights with zeros if needed
                    padding = np.zeros(n_frames - len(weights))
                    weights = np.concatenate([weights, padding])
                    # Renormalize
                    weights = weights / np.sum(weights)

                # Compute weighted uptake for each residue and timepoint
                weighted_uptake = np.zeros((uptake_values.shape[0], uptake_values.shape[1]))
                for t in range(uptake_values.shape[0]):  # For each timepoint
                    for r in range(uptake_values.shape[1]):  # For each residue
                        weighted_uptake[t, r] = np.average(uptake_values[t, r, :], weights=weights)

                # Compute MAE between weighted and unweighted uptake
                mae = np.mean(np.abs(weighted_uptake - unweighted_uptake))
                mae_values.append(mae)

            mae_results[method][param] = mae_values

    return mae_results


def kl_divergence(p, q):
    """
    Calculate the KL divergence between two probability distributions.
    Adds small epsilon to avoid division by zero.
    """
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon

    # Normalize to ensure they are probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    return entropy(p, q)


def compute_w1_distance(ref_avg_pairwise, md_pairwise, md_weights):
    """
    Calculate Wasserstein-1 distance between reference and weighted MD ensemble.

    Parameters:
    -----------
    ref_avg_pairwise : numpy.ndarray
        Average pairwise distances from reference ensemble
    md_pairwise : numpy.ndarray
        Pairwise distances from MD trajectory
    md_weights : numpy.ndarray
        Weights for MD frames

    Returns:
    --------
    w1_dist : float
        Wasserstein-1 distance
    """
    # Compute average pairwise distances for MD using weights
    md_avg_pairwise = np.average(md_pairwise, axis=0, weights=md_weights)

    # Calculate Wasserstein distance between the two distributions
    w1_dist = wasserstein_distance(ref_avg_pairwise, md_avg_pairwise)

    return w1_dist


def extract_hdxer_weights(hdxer_dir, n_seeds, gamma_values, exponents):
    """
    Extract HDXer weights from files, considering both gamma values and exponents.

    Parameters:
    -----------
    hdxer_dir : str
        Directory containing HDXer weight files
    n_seeds : int
        Number of seeds
    gamma_values : list
        List of gamma values (1-9)
    exponents : list
        List of exponents (-1, 0, 1)

    Returns:
    --------
    weights_dict : dict
        Dictionary with structure {"HDXer": {(gamma, exponent): {seed_key: weights_array}}}
    """
    weights_dict = {"HDXer": {}}

    for gamma in gamma_values:
        for exponent in exponents:
            # Create a parameter key that represents both gamma and exponent
            param_key = (gamma, exponent)
            weights_dict["HDXer"][param_key] = {}

            for seed_idx in range(1, n_seeds + 1):
                seed_key = f"seed_{seed_idx - 1}"  # Using 0-indexed naming for consistency

                try:
                    # Construct path to HDXer weights file
                    # Format: reweighting_gamma_1x10^0final_weights.dat
                    weights_file = os.path.join(
                        hdxer_dir,
                        f"train_BPTI_TFES_{seed_idx}",
                        f"reweighting_gamma_{gamma}x10^{exponent}final_weights.dat",
                    )

                    # Check if file exists
                    if os.path.exists(weights_file):
                        # Load weights from file
                        weights = np.loadtxt(weights_file)

                        # Normalize weights
                        weights = weights / np.sum(weights)

                        # Store the weights
                        weights_dict["HDXer"][param_key][seed_key] = weights
                    else:
                        print(f"HDXer weights file not found: {weights_file}")
                        weights_dict["HDXer"][param_key][seed_key] = None
                except Exception as e:
                    print(
                        f"Error loading HDXer weights for seed {seed_idx}, gamma {gamma}, exponent {exponent}: {e}"
                    )
                    weights_dict["HDXer"][param_key][seed_key] = None

    return weights_dict


def extract_weights_from_directory(base_dir, regularization_fn, n_seeds, alpha_values):
    """
    Extract frame weights from optimization history files in the given directory.

    Parameters:
    -----------
    base_dir : str
        Base directory containing results
    regularization_fn : str
        Name of the regularization function directory
    n_seeds : int
        Number of seeds (datasets)
    alpha_values : list
        List of alpha values (replicate parameters)

    Returns:
    --------
    weights_dict : dict
        Dictionary with structure {regularization_fn: {alpha_value: {seed_key: weights_array}}}
    """
    reg_dir = os.path.join(base_dir, regularization_fn)
    weights_dict = {regularization_fn: {}}

    # Initialize weights dictionary for each alpha value
    for alpha in alpha_values:
        weights_dict[regularization_fn][alpha] = {}

    # Extract weights for each seed and replicate (which maps to an alpha value)
    for seed_idx in range(1, n_seeds + 1):
        seed_key = f"seed_{seed_idx - 1}"  # Using 0-indexed naming for consistency

        # Map alpha values to replicate indices (1-indexed)
        for rep_idx, alpha in enumerate(alpha_values, 1):
            try:
                # Construct path to optimization history file (unchanged - still uses rep_idx)
                history_file = os.path.join(
                    reg_dir, f"seed_{seed_idx}_replicate_{rep_idx}optimization_history.h5"
                )

                # Check if file exists
                if os.path.exists(history_file):
                    # Load optimization history
                    opt_history = load_optimization_history_from_file(
                        history_file, default_model_params_cls=BV_Model_Parameters
                    )

                    # Extract weights from the final state
                    if opt_history.states and len(opt_history.states) > 0:
                        final_state = opt_history.states[-1]
                        frame_weights = final_state.params.frame_weights
                        # frame_mask = final_state.params.frame_mask

                        # Apply mask and normalize
                        masked_weights = np.array(frame_weights)
                        normalized_weights = masked_weights / np.sum(masked_weights)

                        # Store with alpha value as the key instead of rep_idx
                        weights_dict[regularization_fn][alpha][seed_key] = normalized_weights
                    else:
                        raise ValueError(
                            f"No states found in optimization history for {history_file}"
                        )
                        print(f"No states found in history for {history_file}")
                        weights_dict[regularization_fn][alpha][seed_key] = None
                else:
                    print(f"History file not found: {history_file}")
                    weights_dict[regularization_fn][alpha][seed_key] = None
            except Exception as e:
                print(f"Error extracting weights for seed {seed_idx}, alpha {alpha}: {e}")
                weights_dict[regularization_fn][alpha][seed_key] = None

    return weights_dict


def compute_pairwise_kl_divergences(weights_dict):
    """
    Compute pairwise KL divergences between all pairs of seeds for each method and parameter.

    Parameters:
    -----------
    weights_dict : dict
        Nested dictionary with structure:
        {method_name: {parameter_value: {seed_key: weights_array}}}

    Returns:
    --------
    kl_results : dict
        Dictionary with KL divergences grouped by method and parameter
        {method_name: {parameter_value: [kl_values]}}
    """
    kl_results = {}

    for method, param_dict in weights_dict.items():
        kl_results[method] = {}

        for param, seed_dict in param_dict.items():
            # Get all seed keys with non-None values
            seed_keys = [key for key, val in seed_dict.items() if val is not None]

            # Skip if fewer than 2 seed keys (need at least 2 for pairwise comparison)
            if len(seed_keys) < 2:
                print(f"Skipping {method}, param={param}: fewer than 2 valid seeds")
                kl_results[method][param] = []
                continue

            # Compute all pairwise KL divergences
            kl_values = []
            for seed1, seed2 in itertools.combinations(seed_keys, 2):
                weights1 = seed_dict[seed1]
                weights2 = seed_dict[seed2]

                # Check again for None (shouldn't happen due to filtering above)
                if weights1 is None or weights2 is None:
                    continue

                # If weights arrays have different lengths, truncate to the shorter one
                min_length = min(len(weights1), len(weights2))
                w1 = weights1[:min_length]
                w2 = weights2[:min_length]

                # Calculate KL divergence in both directions (as it's asymmetric)
                try:
                    kl_1_to_2 = kl_divergence(w1, w2)
                    kl_2_to_1 = kl_divergence(w2, w1)

                    # Use average of both directions
                    avg_kl = (kl_1_to_2 + kl_2_to_1) / 2
                    kl_values.append(avg_kl)
                except Exception as e:
                    print(
                        f"Error calculating KL divergence for {method}, param={param}, seeds {seed1}/{seed2}: {e}"
                    )

            kl_results[method][param] = kl_values

    return kl_results


def compute_uniform_kl_divergences(weights_dict):
    """
    Compute KL divergences between each set of weights and a uniform distribution.

    Parameters:
    -----------
    weights_dict : dict
        Nested dictionary with structure:
        {method_name: {parameter_value: {seed_key: weights_array}}}

    Returns:
    --------
    kl_results : dict
        Dictionary with KL divergences grouped by method and parameter
        {method_name: {parameter_value: [kl_values]}}
    """
    kl_results = {}

    for method, param_dict in weights_dict.items():
        kl_results[method] = {}

        for param, seed_dict in param_dict.items():
            # Get all seed keys with non-None values
            seed_keys = [key for key, val in seed_dict.items() if val is not None]

            # Skip if no valid seeds
            if not seed_keys:
                print(f"Skipping {method}, param={param}: no valid seeds")
                kl_results[method][param] = []
                continue

            # Compute KL divergences from uniform for each seed
            kl_values = []
            for seed_key in seed_keys:
                weights = seed_dict[seed_key]

                if weights is None:
                    continue

                # Create uniform distribution with same length as weights
                uniform_weights = np.ones_like(weights) / len(weights)

                try:
                    # Calculate KL divergence (weights → uniform)
                    kl_div = kl_divergence(weights, uniform_weights)
                    kl_values.append(kl_div)
                except Exception as e:
                    print(
                        f"Error calculating KL divergence from uniform for {method}, param={param}, seed {seed_key}: {e}"
                    )

            kl_results[method][param] = kl_values

    return kl_results


def compute_ensemble_w1_distances(weights_dict, reference_path, topology_path, trajectory_path):
    """
    Compute W1 distances between weighted ensembles and reference structure.

    Parameters:
    -----------
    weights_dict : dict
        Nested dictionary with weights for each method, parameter, and seed
    reference_path : str
        Path to reference PDB file
    topology_path : str
        Path to MD topology file
    trajectory_path : str
        Path to MD trajectory file

    Returns:
    --------
    w1_results : dict
        Dictionary with W1 distances grouped by method and parameter
    """
    w1_results = {}

    # Load reference NMR ensemble
    try:
        ref_universe = mda.Universe(reference_path)
        ref_ca = ref_universe.select_atoms("name CA")
        n_ref_frames = ref_universe.trajectory.n_frames

        # Extract pairwise distances from reference NMR ensemble
        ref_pairwise_distances = []
        for ts in ref_universe.trajectory:
            coords = ref_ca.positions
            pairwise_dists = pdist(coords)
            ref_pairwise_distances.append(pairwise_dists)

        # Convert to numpy array for easier manipulation
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

        # Convert to numpy array for easier manipulation
        md_pairwise_distances = np.array(md_pairwise_distances)

        # Compute W1 distances for each method, parameter, and seed
        for method, param_dict in weights_dict.items():
            w1_results[method] = {}

            for param, seed_dict in param_dict.items():
                w1_values = []

                for seed_key, weights in seed_dict.items():
                    if weights is not None:
                        # Make sure weights array length matches the number of MD frames
                        if len(weights) > len(md_pairwise_distances):
                            weights = weights[: len(md_pairwise_distances)]
                        elif len(weights) < len(md_pairwise_distances):
                            # Pad weights with zeros if needed
                            padding = np.zeros(len(md_pairwise_distances) - len(weights))
                            weights = np.concatenate([weights, padding])
                            # Renormalize
                            weights = weights / np.sum(weights)

                        # Compute W1 distance
                        w1_dist = compute_w1_distance(
                            ref_avg_pairwise, md_pairwise_distances, weights
                        )
                        w1_values.append(w1_dist)

                w1_results[method][param] = w1_values

    except Exception as e:
        print(f"Error computing W1 distances: {e}")
        # Return empty dictionary in case of error
        for method, param_dict in weights_dict.items():
            w1_results[method] = {}
            for param in param_dict.keys():
                w1_results[method][param] = []

    return w1_results


def plot_boxplots(results, metric_name, method_names=None, title=None, ylabel=None):
    """
    Create box plots of metric values between seeds with each method in its own panel.

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
        figsize=(14, 8),
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
        method_data = []
        method_labels = []

        # Convert HDXer parameters to sortable values for ordering
        sorted_params = []

        for param, values in param_dict.items():
            if not values:  # Skip params with no values
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

            sorted_params.append((sort_value, param, values, label))

        # Sort parameters by their numerical value
        sorted_params.sort(key=lambda x: x[0])

        # Define colors for parameters
        param_colors = plt.cm.viridis(np.linspace(0, 1, max(len(sorted_params), 1)))

        # Extract sorted data
        for i, (_, _, values, label) in enumerate(sorted_params):
            method_data.append(values)
            method_labels.append(label)

        # Create box plot with seaborn for this method
        if method_data:
            sns.boxplot(data=method_data, ax=ax, palette=param_colors, width=0.6)

            # Add individual data points
            for i, data in enumerate(method_data):
                # Add jittered points
                x = np.random.normal(i, 0.1, size=len(data))
                ax.scatter(x, data, color="black", alpha=0.5, s=30)

            # Set x-tick labels
            ax.set_xticklabels(method_labels, rotation=45, ha="right")

            # Add method title
            ax.set_title(method)

            # Only show y-label on the first subplot
            if method_idx == 0:
                ax.set_ylabel(ylabel if ylabel else metric_name)
            else:
                ax.set_ylabel("")

            # Add grid
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=14)

    # Add overall title
    fig.suptitle(title if title else f"{metric_name} Comparison", fontsize=16, y=0.98)

    plt.tight_layout(rect=[0, 0.4, 1, 0.95])  # Adjust for the suptitle

    return fig


def plot_kl_divergence_boxplots(
    kl_results, method_names=None, title="Pairwise KL Divergence Between Seeds"
):
    """
    Create box plots of pairwise KL divergences between seeds.
    """
    return plot_boxplots(
        kl_results,
        metric_name="KL Divergence",
        method_names=method_names,
        title=title,
        ylabel="KL Divergence",
    )


def plot_uniform_kl_divergence_boxplots(
    kl_results, method_names=None, title="KL Divergence from Uniform Distribution"
):
    """
    Create box plots of KL divergences between weights and uniform distribution.
    """
    return plot_boxplots(
        kl_results,
        metric_name="KL Divergence from Uniform",
        method_names=method_names,
        title=title,
        ylabel="KL Divergence",
    )


def plot_w1_distance_boxplots(
    w1_results, method_names=None, title="Wasserstein-1 Distance to Reference"
):
    """
    Create box plots of W1 distances between seeds and reference.
    """
    return plot_boxplots(
        w1_results,
        metric_name="W1 Distance",
        method_names=method_names,
        title=title,
        ylabel="Wasserstein-1 Distance (Å)",
    )


def plot_pca_contours(weights_dict, topology_path, trajectory_path, output_dir=None):
    """
    Create contour plots of PCA on CA pairwise distances using weights from different methods.

    Parameters:
    -----------
    weights_dict : dict
        Nested dictionary with weights for each method, parameter, and seed
    topology_path : str
        Path to MD topology file
    trajectory_path : str
        Path to MD trajectory file
    output_dir : str, optional
        Directory to save output plots

    Returns:
    --------
    fig_dict : dict
        Dictionary of matplotlib figures for each method
    """
    import os

    import matplotlib.pyplot as plt
    import MDAnalysis as mda
    import numpy as np
    import seaborn as sns
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA

    # Dictionary to store figures
    fig_dict = {}

    try:
        # Load MD universe
        print("Loading MD trajectory...")
        md_universe = mda.Universe(topology_path, trajectory_path)
        md_ca = md_universe.select_atoms("name CA")
        n_frames = md_universe.trajectory.n_frames
        print(f"Loaded {n_frames} frames")

        # Extract pairwise distances from MD trajectory
        print("Computing pairwise distances...")
        pairwise_distances = []
        for ts in md_universe.trajectory:
            coords = md_ca.positions
            pairwise_dists = pdist(coords)
            pairwise_distances.append(pairwise_dists)

        # Convert to numpy array for PCA
        pairwise_distances = np.array(pairwise_distances)
        print(f"Pairwise distances shape: {pairwise_distances.shape}")

        # Perform PCA on pairwise distances
        print("Performing PCA...")
        pca = PCA(n_components=2)
        projected_coords = pca.fit_transform(pairwise_distances)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

        # Process each method
        for method, param_dict in weights_dict.items():
            print(f"Creating PCA contour plots for {method}...")

            # Count valid parameters
            valid_params = [
                param
                for param, seed_dict in param_dict.items()
                if any(weights is not None for weights in seed_dict.values())
            ]

            if not valid_params:
                print(f"No valid parameters found for {method}")
                continue

            # Define colormap for seeds
            seed_colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Up to 10 colors for seeds

            # Create a figure with subplots for each parameter
            n_cols = min(3, len(valid_params))  # Max 3 columns
            n_rows = (len(valid_params) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(5 * n_cols, 4 * n_rows),
                squeeze=False,
                sharex=True,
                sharey=True,
            )

            # Flatten axes for easier indexing
            axes_flat = axes.flatten()

            # Process each parameter
            for i, param in enumerate(valid_params):
                if i >= len(axes_flat):
                    print(f"Warning: More parameters than subplots for {method}")
                    break

                ax = axes_flat[i]
                seed_dict = param_dict[param]

                # Set title based on parameter type
                if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                    gamma, exponent = param
                    title = f"{method}: γ={gamma}×10^{exponent}"
                elif method == "HDXer":
                    title = f"{method}: γ={param}"
                else:
                    title = f"{method}: Rep {param}"

                ax.set_title(title)

                # Plot each seed as a contour
                for j, (seed_key, weights) in enumerate(seed_dict.items()):
                    if weights is None or len(weights) == 0:
                        continue

                    # Ensure weights array length matches number of frames
                    if len(weights) > n_frames:
                        weights = weights[:n_frames]
                    elif len(weights) < n_frames:
                        # Pad weights with zeros
                        padding = np.zeros(n_frames - len(weights))
                        weights = np.concatenate([weights, padding])
                        # Renormalize
                        weights = weights / np.sum(weights)

                    # Get seed index and color
                    seed_idx = (
                        int(seed_key.split("_")[1]) % 10
                    )  # Get seed number, modulo 10 for colormap index
                    color = seed_colors[seed_idx]

                    # Generate contour plot using seaborn's kdeplot with weights
                    try:
                        x = projected_coords[:, 0]
                        y = projected_coords[:, 1]

                        # Use seaborn's kdeplot with weights
                        sns.kdeplot(
                            x=x,
                            y=y,
                            weights=weights,
                            ax=ax,
                            color=color,
                            fill=True,
                            alpha=0.5,
                            levels=5,
                            label=f"Seed {seed_idx}",
                        )

                    except Exception as e:
                        print(
                            f"Error creating contour plot for {method}, param={param}, seed={seed_key}: {e}"
                        )

                # Add grid and labels
                ax.grid(True, alpha=0.3)
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
                ax.legend(loc="upper right", fontsize="small")

            # Hide any unused subplots
            for j in range(len(valid_params), len(axes_flat)):
                axes_flat[j].set_visible(False)

            # Add overall title
            fig.suptitle(
                f"PCA Contour Plots of CA Pairwise Distances - {method}", fontsize=16, y=0.98
            )
            plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust for suptitle

            # Store figure in dictionary
            fig_dict[method] = fig

            # Save figure if output directory is provided
            if output_dir:
                output_path = os.path.join(output_dir, f"pca_contours_{method}.png")
                fig.savefig(output_path, dpi=300)
                print(f"PCA contour plot for {method} saved to {output_path}")

    except Exception as e:
        print(f"Error in PCA contour plot generation: {e}")

    return fig_dict


def compute_top_structures_w1_distances(
    weights_dict, reference_path, topology_path, trajectory_path, n_top=20
):
    """
    Compute W1 distances between reference structure and top N weighted structures.

    Parameters:
    -----------
    weights_dict : dict
        Nested dictionary with weights for each method, parameter, and seed
    reference_path : str
        Path to reference PDB file
    topology_path : str
        Path to MD topology file
    trajectory_path : str
        Path to MD trajectory file
    n_top : int
        Number of top structures to consider

    Returns:
    --------
    w1_results : dict
        Dictionary with W1 distances for top structures grouped by method and parameter
    """
    w1_results = {}

    try:
        # Load reference NMR ensemble
        ref_universe = mda.Universe(reference_path)
        ref_ca = ref_universe.select_atoms("name CA")
        n_ref_frames = ref_universe.trajectory.n_frames

        # Extract pairwise distances from reference NMR ensemble
        ref_pairwise_distances = []
        for ts in ref_universe.trajectory:
            coords = ref_ca.positions
            pairwise_dists = pdist(coords)
            ref_pairwise_distances.append(pairwise_dists)

        # Convert to numpy array for easier manipulation
        ref_pairwise_distances = np.array(ref_pairwise_distances)

        # Reference weights (uniform)
        ref_weights = np.ones(n_ref_frames) / n_ref_frames

        # Compute average pairwise distances for reference
        ref_avg_pairwise = np.average(ref_pairwise_distances, axis=0, weights=ref_weights)

        # Load MD universe
        md_universe = mda.Universe(topology_path, trajectory_path)
        md_ca = md_universe.select_atoms("name CA")
        n_md_frames = md_universe.trajectory.n_frames

        # Extract pairwise distances from MD trajectory
        md_pairwise_distances = []
        for ts in md_universe.trajectory:
            coords = md_ca.positions
            pairwise_dists = pdist(coords)
            md_pairwise_distances.append(pairwise_dists)

        # Convert to numpy array for easier manipulation
        md_pairwise_distances = np.array(md_pairwise_distances)

        # Compute W1 distances for each method, parameter, and seed using only top N structures
        for method, param_dict in weights_dict.items():
            w1_results[method] = {}

            for param, seed_dict in param_dict.items():
                w1_values = []

                for seed_key, weights in seed_dict.items():
                    if weights is not None:
                        # Make sure weights array length matches the number of MD frames
                        if len(weights) > n_md_frames:
                            weights = weights[:n_md_frames]
                        elif len(weights) < n_md_frames:
                            # Pad weights with zeros if needed
                            padding = np.zeros(n_md_frames - len(weights))
                            weights = np.concatenate([weights, padding])
                            # Renormalize
                            weights = weights / np.sum(weights)

                        # Identify top N structures by weight
                        top_indices = np.argsort(weights)[-n_top:]

                        # Create binary weights: 1 for top structures, 0 for others
                        top_weights = np.zeros_like(weights)
                        top_weights[top_indices] = 1.0 / n_top  # Equal weights for top structures

                        # Compute W1 distance using only the top structures
                        w1_dist = compute_w1_distance(
                            ref_avg_pairwise, md_pairwise_distances, top_weights
                        )
                        w1_values.append(w1_dist)

                w1_results[method][param] = w1_values

    except Exception as e:
        print(f"Error computing W1 distances for top structures: {e}")
        # Return empty dictionary in case of error
        for method, param_dict in weights_dict.items():
            w1_results[method] = {}
            for param in param_dict.keys():
                w1_results[method][param] = []

    return w1_results


def average_weights_and_write_top_structures(
    weights_dict, topology_path, trajectory_path, output_dir, n_top=20
):
    """
    Average weights across seeds for each method/parameter and write out a trajectory
    containing the top N structures by weight.

    Parameters:
    -----------
    weights_dict : dict
        Nested dictionary with weights for each method, parameter, and seed
    topology_path : str
        Path to MD topology file
    trajectory_path : str
        Path to MD trajectory file
    output_dir : str
        Directory to save output trajectories
    n_top : int
        Number of top structures to include in the trajectory

    Returns:
    --------
    avg_weights_dict : dict
        Dictionary with averaged weights for each method and parameter
    """
    avg_weights_dict = {}

    try:
        # Load MD universe
        md_universe = mda.Universe(topology_path, trajectory_path)
        n_frames = md_universe.trajectory.n_frames

        # Select CA atoms for alignment
        ca_atoms = md_universe.select_atoms("name CA")

        # Align trajectory to the first frame using CA atoms
        print("Aligning trajectory by CA atoms...")

        # Create a reference from the first frame
        reference_coords = None
        for ts in md_universe.trajectory[0:1]:
            reference_coords = ca_atoms.positions.copy()

        # Align all frames to the reference
        align.AlignTraj(
            md_universe, md_universe, select="name CA", in_memory=True, weights="mass", ref_frame=0
        ).run()
        print("Alignment complete.")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process each method
        for method, param_dict in weights_dict.items():
            avg_weights_dict[method] = {}

            for param, seed_dict in param_dict.items():
                # Skip if no valid seeds
                valid_seeds = {k: v for k, v in seed_dict.items() if v is not None}
                if not valid_seeds:
                    print(f"Skipping {method}, param={param}: no valid seeds")
                    continue

                # Prepare array for averaged weights
                all_weights = []

                # Collect weights from all seeds
                for seed_key, weights in valid_seeds.items():
                    # Make sure weights array length matches the number of frames
                    if len(weights) > n_frames:
                        weights = weights[:n_frames]
                    elif len(weights) < n_frames:
                        # Pad weights with zeros if needed
                        padding = np.zeros(n_frames - len(weights))
                        weights = np.concatenate([weights, padding])
                        # Renormalize
                        weights = weights / np.sum(weights)

                    all_weights.append(weights)

                # Convert to numpy array
                all_weights = np.array(all_weights)

                # Compute average weights across seeds
                avg_weights = np.mean(all_weights, axis=0)

                # Normalize averaged weights
                avg_weights = avg_weights / np.sum(avg_weights)

                # Store averaged weights
                avg_weights_dict[method][param] = avg_weights
                # Identify top N frames by average weight
                top_indices = np.argsort(avg_weights)[-n_top:]

                # Define output trajectory name based on method and parameter
                if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                    gamma, exponent = param
                    output_name = f"{method}_gamma{gamma}_exp{exponent}_top{n_top}.pdb"
                elif method == "HDXer":
                    output_name = f"{method}_gamma{param}_top{n_top}.pdb"
                else:
                    output_name = f"{method}_rep{param}_top{n_top}.pdb"

                # Full path to output trajectory
                output_traj_path = os.path.join(output_dir, output_name)

                # Create new trajectory with top frames
                try:
                    # Create a PDB writer for the output trajectory
                    with mda.coordinates.PDB.PDBWriter(output_traj_path, multiframe=True) as writer:
                        # Write each frame in order of descending weight (most important first)
                        for idx in sorted(top_indices, key=lambda x: avg_weights[x], reverse=True):
                            md_universe.trajectory[idx]
                            writer.write(md_universe.atoms)

                    print(
                        f"Wrote top {n_top} structures for {method}, param={param} to {output_traj_path}"
                    )
                except Exception as e:
                    print(f"Error writing trajectory for {method}, param={param}: {e}")

    except Exception as e:
        print(f"Error in average_weights_and_write_top_structures: {e}")

    return avg_weights_dict


def plot_top_structures_w1_boxplots(w1_results, method_names=None, n_top=20, title=None):
    """
    Create box plots of W1 distances between the top N structures and reference.

    Parameters:
    -----------
    w1_results : dict
        Dictionary with W1 distances grouped by method and parameter
    method_names : list, optional
        List of method names for the legend
    n_top : int
        Number of top structures used
    title : str, optional
        Plot title

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The box plot figure
    """
    # Default title if none provided
    if title is None:
        title = f"Wasserstein-1 Distance to Reference using Top {n_top} Structures"

    # Use the existing plot_boxplots function
    return plot_boxplots(
        w1_results,
        metric_name=f"W1 Distance (Top {n_top})",
        method_names=method_names,
        title=title,
        ylabel=f"Wasserstein-1 Distance (Top {n_top} Structures, Å)",
    )


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
        Path to MD topology file
    trajectory_path : str
        Path to MD trajectory file

    Returns:
    --------
    w1_results : dict
        Dictionary with W1 distances for clustered structures
    """
    import MDAnalysis as mda
    import numpy as np
    from scipy.spatial.distance import pdist

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
                    w1_dist = compute_w1_distance(ref_avg_pairwise, md_pairwise_distances, weights)

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
        Nested dictionary with weights for each method, parameter, and seed
    topology_path : str
        Path to MD topology file
    trajectory_path : str
        Path to MD trajectory file
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
    from scipy.stats import gaussian_kde
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances_argmin_min

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

        # Calculate density in PCA space to find the highest density structure in unweighted ensemble
        print("Calculating density in PCA space...")
        kde = gaussian_kde(pca_data.T)
        density = kde(pca_data.T)

        # Find the structure with highest density (representing the most populated region)
        highest_density_idx = np.argmax(density)
        highest_density_point = pca_data[highest_density_idx]
        print(f"Highest density structure found at frame {highest_density_idx}")

        # Perform K-means clustering
        print(f"Clustering with K-means ({n_clusters} clusters)...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_data)

        # Calculate distances from each centroid to the highest density point
        centroid_distances = np.linalg.norm(kmeans.cluster_centers_ - highest_density_point, axis=1)

        # Create mapping from original cluster index to ordered index
        ordered_indices = np.argsort(centroid_distances)
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(ordered_indices)}

        # Reorder cluster labels
        reordered_labels = np.array([index_mapping[label] for label in cluster_labels])

        # Reorder cluster centers
        reordered_centers = kmeans.cluster_centers_[ordered_indices]

        # Find the closest frame to each centroid
        print("Finding frames closest to centroids...")
        closest_frames, _ = pairwise_distances_argmin_min(reordered_centers, pca_data)

        # Create a mapping from cluster to frames (using reordered labels)
        cluster_to_frames = {}
        for i in range(n_clusters):
            cluster_to_frames[i] = np.where(reordered_labels == i)[0]

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
                    output_name = f"{method}_rep{param}_cluster{n_top}.pdb"

                # Full path to output trajectory
                output_traj_path = os.path.join(output_dir, output_name)

                # Create trajectory with representative structures
                try:
                    with mda.coordinates.PDB.PDBWriter(output_traj_path, multiframe=True) as writer:
                        # Write each frame in order of descending cluster weight
                        for cluster_idx in sorted(
                            top_clusters, key=lambda x: cluster_weights[x], reverse=True
                        ):
                            # Find frame closest to centroid in this cluster
                            cluster_frames = cluster_to_frames[cluster_idx]

                            # If the cluster has frames, use the one closest to centroid
                            if len(cluster_frames) > 0:
                                # Calculate distances to centroid in PCA space
                                cluster_center = reordered_centers[cluster_idx]
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
                                    - reordered_centers[cluster_idx],
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
            "cluster_labels": reordered_labels,  # Use reordered labels
            "closest_frames": closest_frames,
            "cluster_centers": reordered_centers,  # Use reordered centers
            "centroid_distances": centroid_distances[ordered_indices],  # Reordered distances
            "highest_density_point": highest_density_point,
            "highest_density_frame": highest_density_idx,
        }

    except Exception as e:
        print(f"Error in cluster_and_select_representative_structures: {e}")

    return cluster_results


def plot_clustered_dataset(cluster_results, output_dir=None):
    """
    Visualize the clustered dataset in 2D PCA space, coloring points by cluster.
    Clusters are ordered by distance from highest density structure.

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
    centers = clustering_info["cluster_centers"]  # Already reordered
    highest_density_point = clustering_info["highest_density_point"]
    highest_density_frame = clustering_info["highest_density_frame"]

    # Create a 2D visualization using the first two PCA components
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create a colormap for the clusters (using viridis for better sequential color perception)
    cmap = plt.cm.get_cmap("viridis", np.max(cluster_labels) + 1)

    # Plot each point, colored by cluster
    scatter = ax.scatter(
        pca_data[:, 0], pca_data[:, 1], c=cluster_labels, cmap=cmap, alpha=0.5, s=10
    )

    # Plot cluster centers
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

    # Highlight the highest density point
    ax.scatter(
        highest_density_point[0],
        highest_density_point[1],
        s=300,
        marker="X",
        color="red",
        edgecolors="k",
        linewidths=2,
        label="Highest Density Structure",
    )

    # Add labels showing the order of clusters
    for i, center in enumerate(centers[:10]):  # Just show first 10 to avoid clutter
        ax.text(
            center[0],
            center[1],
            str(i),
            fontsize=12,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "Clustered Trajectory in PCA Space\n(Clusters ordered by distance from highest density structure)"
    )
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Cluster (ordered by distance from highest density)")

    # Save figure if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "clustered_trajectory.png")
        fig.savefig(output_path, dpi=300)
        print(f"Clustered trajectory plot saved to {output_path}")

    return fig


def plot_cluster_distribution(cluster_results, output_dir=None):
    """
    Create bar charts showing the distribution of clusters (bins) for each method and parameter.
    Clusters are ordered by distance from highest density structure.

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
    import numpy as np

    figs = {}

    # Skip clustering_info
    methods = [m for m in cluster_results.keys() if m != "clustering_info"]

    if not methods:
        print("No methods found in cluster results")
        return figs

    # Get centroid distances if available
    if (
        "clustering_info" in cluster_results
        and "centroid_distances" in cluster_results["clustering_info"]
    ):
        centroid_distances = cluster_results["clustering_info"]["centroid_distances"]
    else:
        centroid_distances = None

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
            n_params, 1, figsize=(14, 4 * n_params), squeeze=False, sharex=True
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

            # Use a colormap to color bars by distance from highest density
            if centroid_distances is not None:
                # Normalize distances for colormap
                norm_distances = centroid_distances / np.max(centroid_distances)
                colors = plt.cm.viridis(norm_distances)
                bars = ax.bar(range(len(cluster_weights)), cluster_weights, color=colors, alpha=0.7)
            else:
                bars = ax.bar(range(len(cluster_weights)), cluster_weights, alpha=0.7)

            # Highlight top clusters if available
            if len(top_clusters) > 0:
                for cluster_idx in top_clusters:
                    bars[cluster_idx].set_edgecolor("red")
                    bars[cluster_idx].set_linewidth(2)
                    bars[cluster_idx].set_alpha(1.0)

            # Add grid and labels
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Cluster Index (ordered by distance from highest density)")
            ax.set_ylabel("Normalized Weight")

            # Set title based on parameter type
            if method == "HDXer" and isinstance(param, tuple) and len(param) == 2:
                gamma, exponent = param
                title = f"{method}: γ={gamma}×10^{exponent}"
            elif method == "HDXer":
                title = f"{method}: γ={param}"
            else:
                title = f"{method}: Rep {param}"

            ax.set_title(title)

            # Add text annotations for top clusters
            for cluster_idx in top_clusters:
                ax.text(
                    cluster_idx,
                    cluster_weights[cluster_idx] + 0.01,
                    f"{cluster_idx}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        # Set overall title
        fig.suptitle(f"Cluster Weight Distribution - {method}", fontsize=16, y=0.99)
        plt.tight_layout(rect=[0, 0.1, 1, 0.97])

        # Add colorbar for distance if available
        if centroid_distances is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=np.max(centroid_distances))
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label("Distance from highest density structure")

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
        Path to MD topology file
    trajectory_path : str
        Path to MD trajectory file
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


def main():
    # Base directory containing results
    base_dir = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/jaxENT/MAD_scaling_1_10_25"
    # base_dir = "/Users/alexi/JAX-ENT/JAX-ENT/notebooks/CrossValidation/BPTI/jaxENT/AdamW_loreg"
    # base_dir = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/jaxENT/MaxEnt"
    # HDXer directory
    hdxer_dir = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/HDXer/BPTI_TFES_RW_bench_r_naive_random"

    # Available regularization functions
    regularization_fns = ["mcMSE20_MAD20", "MSE25_MAD25", "MSE25_MAD10"]

    # Number of seeds
    n_seeds = 3

    # Define alpha values (regularization parameters)
    # This replaces n_replicates - use 20 values to match the original n_replicates
    alpha_values = np.linspace(0.1, 10, 10)
    # alpha_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

    # HDXer gamma values and exponents
    gamma_values = list(range(1, 10, 2))
    exponents = [-1, 0, 1]

    # Output directory for saving plots
    output_dir = os.path.join(base_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Combined weights dictionary
    combined_weights_dict = {}

    # Process each regularization function
    for reg_fn in regularization_fns:
        print(f"Processing regularization function: {reg_fn}")

        # Extract weights using alpha values instead of n_replicates
        weights_dict = extract_weights_from_directory(base_dir, reg_fn, n_seeds, alpha_values)

        # Add to combined dictionary
        combined_weights_dict.update(weights_dict)

    # Process HDXer weights with multiple exponents (unchanged)
    print("Processing HDXer weights...")
    hdxer_weights = extract_hdxer_weights(hdxer_dir, n_seeds, gamma_values, exponents)
    combined_weights_dict.update(hdxer_weights)

    # Compute KL divergences
    print("Computing KL divergences...")
    kl_results = compute_pairwise_kl_divergences(combined_weights_dict)

    # Create and save KL divergence visualization
    method_names = regularization_fns + ["HDXer"]
    fig_kl = plot_kl_divergence_boxplots(
        kl_results,
        method_names=method_names,
        title="Pairwise KL Divergence Between Seeds for Different Methods",
    )

    output_path_kl = os.path.join(output_dir, "kl_divergence_comparison.png")
    fig_kl.savefig(output_path_kl, dpi=300)
    print(f"KL divergence plot saved to {output_path_kl}")

    # Compute KL divergences from uniform distribution
    print("Computing KL divergences from uniform distribution...")
    uniform_kl_results = compute_uniform_kl_divergences(combined_weights_dict)

    # Create and save uniform KL divergence visualization
    fig_uniform_kl = plot_uniform_kl_divergence_boxplots(
        uniform_kl_results,
        method_names=method_names,
        title="KL Divergence from Uniform Distribution for Different Methods",
    )

    output_path_uniform_kl = os.path.join(output_dir, "uniform_kl_divergence_comparison.png")
    fig_uniform_kl.savefig(output_path_uniform_kl, dpi=300)
    print(f"Uniform KL divergence plot saved to {output_path_uniform_kl}")

    # Compute W1 distances
    print("Computing W1 distances...")
    reference_path = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/1UUA_BPTI.pdb"
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    HDX_NMR_pf_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_pfactors.dat"
    segs_path = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/BPTI_residue_segs_trimmed.txt"
    dfrac_path = (
        "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/BPTI_expt_dfracs_clean_trimmed.dat"
    )
    w1_results = compute_ensemble_w1_distances(
        combined_weights_dict, reference_path, topology_path, trajectory_path
    )

    # Create and save W1 distance visualization
    fig_w1 = plot_w1_distance_boxplots(
        w1_results,
        method_names=method_names,
        title="Wasserstein-1 Distance to Reference for Different Methods",
    )

    output_path_w1 = os.path.join(output_dir, "w1_distance_comparison.png")
    fig_w1.savefig(output_path_w1, dpi=300)
    print(f"W1 distance plot saved to {output_path_w1}")

    # Compute W1 distances using only top 20 structures
    print("Computing W1 distances for top 20 structures...")
    n_top = 20
    top_w1_results = compute_top_structures_w1_distances(
        combined_weights_dict, reference_path, topology_path, trajectory_path, n_top=n_top
    )

    # Create and save W1 distance visualization for top structures
    fig_top_w1 = plot_top_structures_w1_boxplots(
        top_w1_results,
        method_names=method_names,
        n_top=n_top,
        title=f"Wasserstein-1 Distance to Reference using Top {n_top} Structures",
    )

    output_path_top_w1 = os.path.join(output_dir, f"top{n_top}_w1_distance_comparison.png")
    fig_top_w1.savefig(output_path_top_w1, dpi=300)
    print(f"W1 distance plot for top {n_top} structures saved to {output_path_top_w1}")

    # Average weights and write trajectories of top structures
    print(f"Averaging weights and writing trajectories of top {n_top} structures...")
    traj_output_dir = os.path.join(output_dir, "top_trajectories")
    avg_weights_dict = average_weights_and_write_top_structures(
        combined_weights_dict, topology_path, trajectory_path, traj_output_dir, n_top=n_top
    )

    # Create and save PCA contour plots
    print("Creating PCA contour plots...")
    pca_figures = plot_pca_contours(
        combined_weights_dict, topology_path, trajectory_path, output_dir
    )

    # Save each PCA figure
    if pca_figures and output_dir:
        for method, fig in pca_figures.items():
            output_path = os.path.join(output_dir, f"pca_contours_{method}.png")
            fig.savefig(output_path, dpi=300)
            print(f"PCA contour plot for {method} saved to {output_path}")

    # ===== Clustering Analysis =====
    print("\n==== Running Clustering Analysis ====\n")
    clustering_output_dir = os.path.join(output_dir, "clustering")
    clustering_results = run_clustering_analysis(
        combined_weights_dict,
        reference_path,
        topology_path,
        trajectory_path,
        clustering_output_dir,
        n_components=10,
        n_clusters=50,
        n_top=20,
    )

    # ===== Protection Factor Analysis =====
    print("\n==== Running Protection Factor Analysis ====\n")
    pf_output_dir = os.path.join(output_dir, "protection_factors")
    pf_results = analyze_protection_factors(
        combined_weights_dict,
        reference_path,
        topology_path,
        trajectory_path,
        HDX_NMR_pf_path,
        pf_output_dir,
    )

    # ===== Deuterium Uptake Analysis =====
    print("\n==== Running Deuterium Uptake Analysis ====\n")
    uptake_output_dir = os.path.join(output_dir, "deuterium_uptake")

    uptake_results = analyze_deuterium_uptake(
        combined_weights_dict,
        reference_path,
        topology_path,
        trajectory_path,
        segs_path,
        dfrac_path,
        uptake_output_dir,
    )

    # Extract and display MAE results
    mae_fig = uptake_results.get("fig_mae")
    if mae_fig:
        output_path_mae = os.path.join(output_dir, "deuterium_uptake_mae_comparison.png")
        mae_fig.savefig(output_path_mae, dpi=300)
        print(f"Deuterium uptake MAE plot saved to {output_path_mae}")

    # Create scatter plots for comparison
    print("\n==== Creating Comparison Scatter Plots ====\n")
    scatter_output_dir = os.path.join(output_dir, "scatter_plots")

    # Get the MAE results from uptake analysis
    mae_results = uptake_results.get("mae_results", {})

    # Get the MSE results from uptake analysis (changed from w1_results to mse_results)
    mse_results = uptake_results.get("mse_results", {})  # <-- CHANGED THIS LINE

    # Create comparison scatter plots
    scatter_figs = create_comparison_scatter_plots(
        uniform_kl_results,  # KL divergence from uniform
        mae_results,  # Uptake MAE
        mse_results,  # MSE distance for uptake (changed from w1_uptake_results)
        scatter_output_dir,
    )

    # Show the plots
    plt.show()

    # Include the scatter figures in the return values
    return (
        kl_results,
        w1_results,
        uniform_kl_results,
        top_w1_results,
        avg_weights_dict,
        fig_kl,
        fig_w1,
        fig_uniform_kl,
        fig_top_w1,
        pca_figures,
        clustering_results,
        pf_results,
        uptake_results,
        scatter_figs,  # Add scatter plot figures
    )


if __name__ == "__main__":
    main()
