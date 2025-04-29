"""
This script extends the auto validation analysis to work across multiple experiments
in a directory structure. It generates plots for each experiment and also creates
plots showing average performance across different parameters.
"""

"""
This script extends the auto validation analysis to work across multiple experiments
with different noise levels and regularization scales. It generates comparison plots
showing the effects of both parameters on optimization results.
"""

import glob
import os
import re
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import functions from the original script
from jaxENT_interpretDIR_QuickAutoValidation_TeaA import (
    cluster_frames_by_rmsd,
    compute_rmsd_to_references,
    compute_weighted_cluster_ratios,
    compute_weighted_rmsd_distributions,
    extract_interval_states,
    load_all_optimization_histories,
)
from scipy import stats

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


def find_experiment_directories(base_dir):
    """
    Find all experiment directories that match the pattern for different
    noise levels and regularization scales.

    Expected directory structure:
    base_dir/
        convergence_clippedequalnoise{noise_sd}_l2long_adam_mcmcmse/
            TeaA_simple_{reg_scale}_adam/
                quick_auto_validation_results/

    Returns:
    --------
    experiments : list
        List of tuples (directory, method, reg_scale, noise_sd) with experiment information
    """
    experiments = []

    # Find all noise level directories
    noise_dirs = glob.glob(
        os.path.join(base_dir, "convergence_clippedequalnoise*_l2long_adam_mcmse")
    )
    noise_dirs = [d for d in noise_dirs if os.path.isdir(d)]

    for noise_dir in noise_dirs:
        # Extract noise level from directory name
        dirname = os.path.basename(noise_dir)
        match = re.search(r"clippedequalnoise(\d+(?:\.\d+)?)_l2long_adam_mcmse", dirname)
        if match:
            noise_sd = float(match.group(1))

            # Find all regularization scale directories within this noise level
            reg_dirs = glob.glob(os.path.join(noise_dir, "TeaA_simple_*_adam"))
            reg_dirs = [d for d in reg_dirs if os.path.isdir(d)]

            for reg_dir in reg_dirs:
                # Extract regularization scale from directory name
                reg_dirname = os.path.basename(reg_dir)
                match = re.search(r"TeaA_simple_(\d+(?:\.\d+)?(?:e[+-]\d+)?)_adam", reg_dirname)
                if match:
                    reg_scale_str = match.group(1)
                    try:
                        # Convert to float
                        if "e" in reg_scale_str.lower():
                            reg_scale = float(reg_scale_str)
                        else:
                            reg_scale = float(reg_scale_str)

                        # Check if results directory exists
                        results_dir = os.path.join(reg_dir, "quick_auto_validation_results")
                        if os.path.isdir(results_dir):
                            experiments.append((results_dir, "TeaA_simple", reg_scale, noise_sd))
                    except ValueError:
                        # Skip if parameter cannot be converted
                        continue

    return experiments


def organize_experiments_by_parameters(experiments):
    """
    Organize experiments by method, noise_sd, and reg_scale.

    Parameters:
    -----------
    experiments : list
        List of tuples (directory, method, reg_scale, noise_sd) with experiment information

    Returns:
    --------
    organized_exps : dict
        Dictionary organized by method, then noise_sd, then reg_scale
    methods : list
        List of unique methods
    noise_sds : list
        List of unique noise_sd values, sorted
    reg_scales : list
        List of unique reg_scale values, sorted
    """
    organized_exps = defaultdict(lambda: defaultdict(dict))
    methods = set()
    noise_sds = set()
    reg_scales = set()

    for directory, method, reg_scale, noise_sd in experiments:
        organized_exps[method][noise_sd][reg_scale] = directory
        methods.add(method)
        noise_sds.add(noise_sd)
        reg_scales.add(reg_scale)

    # Convert to sorted lists
    methods = sorted(list(methods))
    noise_sds = sorted(list(noise_sds))
    reg_scales = sorted(list(reg_scales))

    return organized_exps, methods, noise_sds, reg_scales


def analyze_cross_experiments(base_dir, reference_paths, topology_path, trajectory_path):
    """
    Analyze experiments across different noise levels and regularization scales.

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

    # Organize experiments by parameters
    print("Organizing experiments by parameters...")
    organized_exps, methods, noise_sds, reg_scales = organize_experiments_by_parameters(experiments)

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
        "organized_exps": organized_exps,
        "methods": methods,
        "noise_sds": noise_sds,
        "reg_scales": reg_scales,
    }

    # Process each experiment
    total_exps = sum(
        len(reg_dict)
        for method_dict in organized_exps.values()
        for noise_sd, reg_dict in method_dict.items()
    )

    exp_count = 0
    for method in methods:
        for noise_sd in noise_sds:
            for reg_scale in reg_scales:
                if reg_scale not in organized_exps[method][noise_sd]:
                    continue

                directory = organized_exps[method][noise_sd][reg_scale]
                exp_count += 1
                print(
                    f"\nProcessing experiment {exp_count}/{total_exps}: method={method}, noise_sd={noise_sd}, reg_scale={reg_scale}"
                )

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
                    "reg_scale": reg_scale,
                    "noise_sd": noise_sd,
                    "histories": histories,
                    "rmsd_grid": rmsd_grid,
                    "kde_values": kde_values,
                    "uniform_kde_values": uniform_kde_values,
                    "cluster_ratios": cluster_ratios,
                    "uniform_cluster_ratios": uniform_cluster_ratios,
                    "interval_states": interval_states,
                }

                results["experiments"][(method, reg_scale, noise_sd)] = exp_results

    return results


def plot_cross_parameter_comparison_rmsd(results, output_dir, reverse_xscale=False):
    """
    Create plots comparing RMSD distributions across different regularization scales
    and noise levels.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_cross_experiments
    output_dir : str
        Directory to save output plots
    reverse_xscale : bool, optional
        If True, reverse the x-axis scale (default is False)
    """
    rmsd_values = results["rmsd_values"]
    ref_names = results["ref_names"]
    methods = results["methods"]
    noise_sds = results["noise_sds"]
    reg_scales = results["reg_scales"]

    # Create a directory for cross-parameter comparison plots
    cross_param_dir = os.path.join(output_dir, "cross_parameter_comparison")
    os.makedirs(cross_param_dir, exist_ok=True)

    # Process each method separately
    for method in methods:
        method_dir = os.path.join(cross_param_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Create a figure for RMSD distributions for each reference
        for ref_idx, ref_name in enumerate(ref_names):
            # Create figures for different view types
            # 1. Single plot with all noise levels and reg_scales
            fig1, ax1 = plt.subplots(figsize=(12, 8))

            # 2. Grid of plots with reg_scales on x-axis and noise_sd on y-axis
            n_noise = len(noise_sds)
            n_reg = len(reg_scales)
            fig2, axes2 = plt.subplots(n_noise, 1, figsize=(12, 5 * n_noise), sharex=True)
            if n_noise == 1:
                axes2 = [axes2]

            # 3. Side-by-side plots for each noise level
            fig3, axes3 = plt.subplots(1, n_noise, figsize=(6 * n_noise, 6), sharey=True)
            if n_noise == 1:
                axes3 = [axes3]

            # Get grid range for KDE
            min_rmsd = np.min(rmsd_values[:, ref_idx])
            max_rmsd = np.max(rmsd_values[:, ref_idx]) * 1.1
            grid = np.linspace(min_rmsd, max_rmsd, 1000)

            # Plot uniform distribution
            uniform_weights = np.ones(len(rmsd_values)) / len(rmsd_values)
            uniform_kde = stats.gaussian_kde(rmsd_values[:, ref_idx], weights=uniform_weights)
            uniform_density = uniform_kde(grid)

            # Plot on the first figure
            ax1.plot(grid, uniform_density, "k--", label="Initial (uniform)", linewidth=2)

            # Plot on the grid figure
            for i, noise_sd in enumerate(noise_sds):
                axes2[i].plot(grid, uniform_density, "k--", label="Initial (uniform)", linewidth=2)
                axes2[i].set_title(f"Noise SD = {noise_sd}")

            # Plot on the side-by-side figure
            for i, noise_sd in enumerate(noise_sds):
                axes3[i].plot(grid, uniform_density, "k--", label="Initial (uniform)", linewidth=2)
                axes3[i].set_title(f"Noise SD = {noise_sd}")

            # Use different colormaps for different noise levels
            reg_cmap = plt.cm.viridis
            reg_colors = [reg_cmap(i) for i in np.linspace(0, 1, len(reg_scales))]

            noise_cmap = plt.cm.plasma
            noise_colors = [noise_cmap(i) for i in np.linspace(0, 1, len(noise_sds))]

            # Plot each experiment
            for i, noise_sd in enumerate(noise_sds):
                for j, reg_scale in enumerate(reg_scales):
                    exp_results = results["experiments"].get((method, reg_scale, noise_sd))
                    if exp_results is None:
                        continue

                    # Get mean KDE values for this experiment
                    kde_values = exp_results["kde_values"]
                    mean_kde = np.mean(kde_values[:, ref_idx, :], axis=0)

                    # Create grid if needed
                    if len(mean_kde) != len(grid):
                        # Interpolate to match grid
                        old_grid = exp_results["rmsd_grid"]
                        mean_kde = np.interp(grid, old_grid, mean_kde)

                    # Format labels
                    if reg_scale < 0.001:
                        reg_label = f"α={reg_scale:.0e}"
                    else:
                        reg_label = f"α={reg_scale:.4f}"

                    # Plot on the first figure - solid line for reg_scale, color for noise_sd
                    label = f"Noise={noise_sd}, {reg_label}"
                    ax1.plot(
                        grid,
                        mean_kde,
                        color=noise_colors[i],
                        linestyle=["-", "--", ":", "-."][j % 4],
                        linewidth=2,
                        label=label,
                    )

                    # Plot on the grid figure - one row per noise_sd
                    axes2[i].plot(grid, mean_kde, color=reg_colors[j], linewidth=2, label=reg_label)

                    # Plot on the side-by-side figure - one column per noise_sd
                    axes3[i].plot(grid, mean_kde, color=reg_colors[j], linewidth=2, label=reg_label)

            # Set labels and titles for the first figure
            ax1.set_xlabel("RMSD (Å)")
            ax1.set_ylabel("Probability Density")
            ax1.set_title(
                f"RMSD Distribution to {ref_name} State by Noise and Regularization ({method})"
            )
            ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            if reverse_xscale:
                ax1.invert_xaxis()

            # Set labels for the grid figure
            for i in range(len(noise_sds)):
                axes2[i].set_ylabel("Probability Density")
                if i == len(noise_sds) - 1:
                    axes2[i].set_xlabel("RMSD (Å)")
                if i == 0:
                    axes2[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                if reverse_xscale:
                    axes2[i].invert_xaxis()

            fig2.suptitle(
                f"RMSD Distribution to {ref_name} State by Regularization Scale ({method})"
            )

            # Set labels for the side-by-side figure
            for i in range(len(noise_sds)):
                if i == 0:
                    axes3[i].set_ylabel("Probability Density")
                axes3[i].set_xlabel("RMSD (Å)")
                if i == len(noise_sds) - 1:
                    axes3[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                if reverse_xscale:
                    axes3[i].invert_xaxis()

            fig3.suptitle(
                f"RMSD Distribution to {ref_name} State by Regularization Scale ({method})"
            )

            # Save figures
            plt.figure(fig1.number)
            plt.tight_layout()
            fig1.savefig(
                os.path.join(method_dir, f"rmsd_kde_{ref_name.lower()}_all_params.png"),
                dpi=300,
                bbox_inches="tight",
            )

            plt.figure(fig2.number)
            plt.tight_layout()
            fig2.savefig(
                os.path.join(method_dir, f"rmsd_kde_{ref_name.lower()}_grid.png"),
                dpi=300,
                bbox_inches="tight",
            )

            plt.figure(fig3.number)
            plt.tight_layout()
            fig3.savefig(
                os.path.join(method_dir, f"rmsd_kde_{ref_name.lower()}_side_by_side.png"),
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

    print(f"Saved cross-parameter comparison RMSD plots to {cross_param_dir}")


def plot_cross_parameter_comparison_ratio(results, output_dir, reverse_xscale=False):
    """
    Create plots comparing state ratios across different regularization scales
    and noise levels.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_cross_experiments
    output_dir : str
        Directory to save output plots
    reverse_xscale : bool, optional
        If True, reverse the numerical x-axis scales (default is False)
    """
    ref_names = results["ref_names"]
    methods = results["methods"]
    noise_sds = results["noise_sds"]
    reg_scales = results["reg_scales"]

    # Create a directory for cross-parameter comparison plots
    cross_param_dir = os.path.join(output_dir, "cross_parameter_comparison")
    os.makedirs(cross_param_dir, exist_ok=True)

    # Define reference ratios (60:40 open:closed)
    reference_ratios = {"Open": 0.6, "Closed": 0.4}

    # Process each method separately
    for method in methods:
        method_dir = os.path.join(cross_param_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Create dataframes to store ratio data for plotting
        all_ratio_data = []

        # Collect data for each experiment
        for noise_sd in noise_sds:
            for reg_scale in reg_scales:
                exp_results = results["experiments"].get((method, reg_scale, noise_sd))
                if exp_results is None:
                    continue

                cluster_ratios = exp_results["cluster_ratios"]

                # Calculate mean and std for each cluster
                for cluster_idx, cluster_name in enumerate(ref_names):
                    mean_ratio = np.mean(cluster_ratios[:, cluster_idx])
                    std_ratio = np.std(cluster_ratios[:, cluster_idx])

                    # Format reg_scale label
                    if reg_scale < 0.001:
                        reg_label = f"{reg_scale:.0e}"
                    else:
                        reg_label = f"{reg_scale:.4f}"

                    all_ratio_data.append(
                        {
                            "Method": method,
                            "Noise SD": noise_sd,
                            "Reg Scale": reg_scale,
                            "Reg Scale (str)": reg_label,
                            "Cluster": cluster_name,
                            "Mean Ratio": mean_ratio,
                            "Std Ratio": std_ratio,
                        }
                    )

        if not all_ratio_data:
            continue

        # Convert to DataFrame
        df = pd.DataFrame(all_ratio_data)

        # 1. Create line plot of ratio vs reg_scale, grouped by noise_sd
        for cluster_name in ref_names:
            cluster_df = df[df["Cluster"] == cluster_name]

            plt.figure(figsize=(12, 8))

            # Group by noise_sd
            for noise_sd in noise_sds:
                noise_df = cluster_df[cluster_df["Noise SD"] == noise_sd]
                if len(noise_df) == 0:
                    continue

                # Sort by reg_scale
                noise_df = noise_df.sort_values("Reg Scale")

                plt.errorbar(
                    noise_df["Reg Scale (str)"],
                    noise_df["Mean Ratio"],
                    yerr=noise_df["Std Ratio"],
                    marker="o",
                    linestyle="-",
                    label=f"Noise SD = {noise_sd}",
                )

            # Add reference ratio line
            ref_ratio = reference_ratios[cluster_name]
            plt.axhline(
                y=ref_ratio,
                color="grey",
                linestyle="--",
                linewidth=5,
                label=f"Reference Ratio ({cluster_name}: {ref_ratio:.0%})",
            )

            plt.xlabel("Regularization Scale (α)")
            plt.ylabel(f"{cluster_name} State Ratio")
            plt.title(f"{cluster_name} State Ratio by Regularization Scale and Noise ({method})")
            plt.legend()
            plt.grid(True)

            # Adjust x-tick labels for readability
            plt.xticks(rotation=45, ha="right")

            if reverse_xscale:
                plt.gca().invert_xaxis()  # Reverse the x-axis (Noise SD)

            plt.tight_layout()
            plt.savefig(
                os.path.join(method_dir, f"{cluster_name.lower()}_ratio_by_alpha_noise.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 2. Create line plot of ratio vs noise_sd, grouped by reg_scale
        for cluster_name in ref_names:
            cluster_df = df[df["Cluster"] == cluster_name]

            plt.figure(figsize=(12, 8))

            # Use a colormap for different reg_scale values
            reg_cmap = plt.cm.viridis
            reg_colors = [reg_cmap(i) for i in np.linspace(0, 1, len(reg_scales))]

            # Group by reg_scale
            for i, reg_scale in enumerate(reg_scales):
                reg_df = cluster_df[cluster_df["Reg Scale"] == reg_scale]
                if len(reg_df) == 0:
                    continue

                # Sort by noise_sd
                reg_df = reg_df.sort_values("Noise SD")

                # Format reg_scale label
                if reg_scale < 0.001:
                    reg_label = f"α={reg_scale:.0e}"
                else:
                    reg_label = f"α={reg_scale:.4f}"

                plt.errorbar(
                    reg_df["Noise SD"],
                    reg_df["Mean Ratio"],
                    yerr=reg_df["Std Ratio"],
                    marker="o",
                    linestyle="-",
                    color=reg_colors[i],
                    label=reg_label,
                )

            # Add reference ratio line
            ref_ratio = reference_ratios[cluster_name]
            plt.axhline(
                y=ref_ratio,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Reference Ratio ({cluster_name}: {ref_ratio:.0%})",
            )

            plt.xlabel("Noise Standard Deviation")
            plt.ylabel(f"{cluster_name} State Ratio")
            plt.title(f"{cluster_name} State Ratio by Noise SD and Regularization Scale ({method})")
            plt.legend()
            plt.grid(True)

            # Set y-axis limits with some padding for the reference line
            plt.ylim(0, 1)

            plt.tight_layout()
            plt.savefig(
                os.path.join(method_dir, f"{cluster_name.lower()}_ratio_by_noise_alpha.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # 3. Create heatmap of ratios
        for cluster_name in ref_names:
            cluster_df = df[df["Cluster"] == cluster_name]

            # Pivot data for heatmap: reg_scale as rows, noise_sd as columns
            pivot_df = cluster_df.pivot_table(
                index="Reg Scale", columns="Noise SD", values="Mean Ratio"
            )

            # Sort rows by reg_scale
            pivot_df = pivot_df.sort_index()

            plt.figure(figsize=(10, 8))
            ax = plt.gca()

            # Create heatmap
            im = ax.imshow(pivot_df.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)

            # Set tick labels
            ax.set_xticks(np.arange(len(pivot_df.columns)))
            ax.set_yticks(np.arange(len(pivot_df.index)))

            # Format reg_scale labels
            reg_labels = []
            for scale in pivot_df.index:
                if scale < 0.001:
                    reg_labels.append(f"{scale:.0e}")
                else:
                    reg_labels.append(f"{scale:.4f}")

            ax.set_xticklabels([f"{sd}" for sd in pivot_df.columns])
            ax.set_yticklabels(reg_labels)

            # Rotate x tick labels if needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(f"{cluster_name} State Ratio", rotation=-90, va="bottom")

            # Add text annotations
            for i in range(len(pivot_df.index)):
                for j in range(len(pivot_df.columns)):
                    if not np.isnan(pivot_df.values[i, j]):
                        text = ax.text(
                            j,
                            i,
                            f"{pivot_df.values[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="w" if pivot_df.values[i, j] > 0.5 else "k",
                        )

            ax.set_title(f"{cluster_name} State Ratio Heatmap ({method})")
            ax.set_xlabel("Noise Standard Deviation")
            ax.set_ylabel("Regularization Scale (α)")

            if reverse_xscale:
                ax.invert_xaxis()  # Reverse the x-axis (Noise SD)

            plt.tight_layout()
            plt.savefig(
                os.path.join(method_dir, f"{cluster_name.lower()}_ratio_heatmap.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    print(f"Saved cross-parameter comparison ratio plots to {cross_param_dir}")


def plot_ratio_evolution_comparison(results, output_dir, reverse_xscale=False):
    """
    Create plots comparing the evolution of state ratios during optimization
    across different noise levels and regularization scales.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_cross_experiments
    output_dir : str
        Directory to save output plots
    reverse_xscale : bool, optional
        If True, reverse the x-axis scale (optimization steps) (default is False)
    """
    ref_names = results["ref_names"]
    cluster_assignments = results["cluster_assignments"]
    methods = results["methods"]
    noise_sds = results["noise_sds"]
    reg_scales = results["reg_scales"]

    # Create a directory for evolution comparison plots
    evo_dir = os.path.join(output_dir, "evolution_comparison")
    os.makedirs(evo_dir, exist_ok=True)

    # Process each method separately
    for method in methods:
        method_dir = os.path.join(evo_dir, method)
        os.makedirs(method_dir, exist_ok=True)

        # Create side-by-side plots for each noise level and cluster
        for cluster_idx, cluster_name in enumerate(ref_names):
            n_noise = len(noise_sds)
            fig, axes = plt.subplots(1, n_noise, figsize=(6 * n_noise, 6), sharey=True)
            if n_noise == 1:
                axes = [axes]

            # Use a colormap for different reg_scale values
            reg_cmap = plt.cm.viridis
            reg_colors = [reg_cmap(i) for i in np.linspace(0, 1, len(reg_scales))]

            # Process each noise level
            for i, noise_sd in enumerate(noise_sds):
                ax = axes[i]

                # Process each regularization scale
                for j, reg_scale in enumerate(reg_scales):
                    exp_results = results["experiments"].get((method, reg_scale, noise_sd))
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

                    # Format reg_scale label
                    if reg_scale < 0.001:
                        reg_label = f"α={reg_scale:.0e}"
                    else:
                        reg_label = f"α={reg_scale:.4f}"

                    # Plot ratio evolution for this parameter
                    ax.plot(
                        steps,
                        interval_ratios,
                        "o-",
                        linewidth=2,
                        label=reg_label,
                        color=reg_colors[j],
                    )

                ax.set_xlabel("Optimization Step")
                if i == 0:
                    ax.set_ylabel("State Ratio")
                ax.set_title(f"Noise SD = {noise_sd}")

                # Set y-axis limits with some padding
                ax.set_ylim(0, 1)

                if reverse_xscale:
                    ax.invert_xaxis()  # Reverse the x-axis (Optimization Step)

                # Only show legend for the last subplot to save space
                if i == n_noise - 1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            fig.suptitle(
                f"Evolution of {cluster_name} State Ratio by Regularization Scale ({method})"
            )
            plt.tight_layout()
            fig.savefig(
                os.path.join(method_dir, f"{cluster_name.lower()}_ratio_evolution_comparison.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

    print(f"Saved ratio evolution comparison plots to {evo_dir}")


def create_summary_table(results, output_dir):
    """
    Create a summary table of results across all experiments.

    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_cross_experiments
    output_dir : str
        Directory to save output file
    """
    ref_names = results["ref_names"]
    methods = results["methods"]

    # Collect data for all experiments
    summary_data = []

    for exp_key, exp_results in results["experiments"].items():
        method, reg_scale, noise_sd = exp_key

        # Get cluster ratios
        cluster_ratios = exp_results["cluster_ratios"]

        # Calculate mean and std for each cluster
        for cluster_idx, cluster_name in enumerate(ref_names):
            mean_ratio = np.mean(cluster_ratios[:, cluster_idx])
            std_ratio = np.std(cluster_ratios[:, cluster_idx])

            summary_data.append(
                {
                    "Method": method,
                    "Regularization Scale": reg_scale,
                    "Noise SD": noise_sd,
                    "State": cluster_name,
                    "Mean Ratio": mean_ratio,
                    "Std Ratio": std_ratio,
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(summary_data)

    # Save to CSV
    csv_path = os.path.join(output_dir, "experiment_summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved summary table to {csv_path}")


def main():
    # Set up paths - these should be adjusted based on the actual file locations
    base_dir = "/Users/alexi/JAX-ENT/notebooks/AutoValidation"
    open_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
    closed_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
    topology_path = open_path
    trajectory_path = (
        "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"
    )

    # Reference paths
    reference_paths = [open_path, closed_path]

    # --- Control flag for reversing x-axis ---
    reverse_x_axis = True  # Set to True to reverse x-axes in plots
    # -----------------------------------------

    # Create output directory for cross-experiment analysis
    output_dir = os.path.join(base_dir, "__cross_clippedequalnoise_reg_analysis_mcMSE")

    os.makedirs(output_dir, exist_ok=True)

    # Run analysis across all experiments
    results = analyze_cross_experiments(base_dir, reference_paths, topology_path, trajectory_path)

    if results is None:
        print("Analysis failed. Exiting.")
        return

    # Create cross-parameter comparison plots
    print("\nCreating cross-parameter comparison plots...")
    plot_cross_parameter_comparison_ratio(results, output_dir, reverse_xscale=reverse_x_axis)
    plot_ratio_evolution_comparison(results, output_dir, reverse_xscale=reverse_x_axis)

    # Create summary table
    print("\nCreating summary table...")
    create_summary_table(results, output_dir)

    print("\nCross-experiment analysis completed successfully!")
    print(f"All plots and summary data saved to {output_dir}")


if __name__ == "__main__":
    main()
