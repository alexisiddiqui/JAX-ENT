import itertools
import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from MDAnalysis.analysis import align
from scipy.spatial.distance import pdist
from scipy.stats import entropy, wasserstein_distance

from jaxent.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.utils.hdf import load_optimization_history_from_file


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


def extract_weights_from_directory(base_dir, regularization_fn, n_seeds, n_replicates):
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
    n_replicates : int
        Number of replicates per seed

    Returns:
    --------
    weights_dict : dict
        Dictionary with structure {regularization_fn: {replicate_idx: {seed_key: weights_array}}}
    """
    reg_dir = os.path.join(base_dir, regularization_fn)
    weights_dict = {regularization_fn: {}}

    # Initialize weights dictionary for each replicate
    for rep_idx in range(1, n_replicates + 1):
        weights_dict[regularization_fn][rep_idx] = {}

    # Extract weights for each seed and replicate
    for seed_idx in range(1, n_seeds + 1):
        seed_key = f"seed_{seed_idx - 1}"  # Using 0-indexed naming for consistency

        for rep_idx in range(1, n_replicates + 1):
            try:
                # Construct path to optimization history file
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
                        frame_mask = final_state.params.frame_mask

                        # Apply mask and normalize
                        masked_weights = np.array(frame_weights * frame_mask)
                        normalized_weights = masked_weights / np.sum(masked_weights)

                        weights_dict[regularization_fn][rep_idx][seed_key] = normalized_weights
                    else:
                        print(f"No states found in history for {history_file}")
                        weights_dict[regularization_fn][rep_idx][seed_key] = None
                else:
                    print(f"History file not found: {history_file}")
                    weights_dict[regularization_fn][rep_idx][seed_key] = None
            except Exception as e:
                print(f"Error extracting weights for seed {seed_idx}, replicate {rep_idx}: {e}")
                weights_dict[regularization_fn][rep_idx][seed_key] = None

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
                sort_value = param
                label = f"Rep {param}"

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

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle

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
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

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

        # Perform K-means clustering
        print(f"Clustering with K-means ({n_clusters} clusters)...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pca_data)

        # Find the closest frame to each centroid
        print("Finding frames closest to centroids...")
        closest_frames, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, pca_data)

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
            "closest_frames": closest_frames,
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


def plot_cluster_distribution(cluster_results, output_dir=None):
    """
    Create bar charts showing the distribution of clusters (bins) for each method and parameter.

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
            if (
                len(top_clusters) > 0
            ):  # Fix: Use explicit length check instead of boolean evaluation
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
                title = f"{method}: Rep {param}"

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
    base_dir = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/jaxENT/xSimilarity"
    # base_dir = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/jaxENT/MaxEnt"

    # HDXer directory
    hdxer_dir = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/BPTI/HDXer/BPTI_TFES_RW_bench_r_naive_random"

    # Available regularization functions
    regularization_fns = ["mean_L1"]

    # Number of seeds and replicates
    n_seeds = 3
    n_replicates = 20

    # HDXer gamma values and exponents
    gamma_values = list(range(1, 10))  # 1-9
    exponents = [0]

    # Output directory for saving plots
    output_dir = os.path.join(base_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Combined weights dictionary
    combined_weights_dict = {}

    # Process each regularization function
    for reg_fn in regularization_fns:
        print(f"Processing regularization function: {reg_fn}")

        # Extract weights from optimization histories
        weights_dict = extract_weights_from_directory(base_dir, reg_fn, n_seeds, n_replicates)

        # Add to combined dictionary
        combined_weights_dict.update(weights_dict)

    # Process HDXer weights with multiple exponents
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
    reference_path = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/1UUA_BPTI.pdb"
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"

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

    # ===== NEW: Clustering Analysis =====
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

    # Show the plots
    plt.show()

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
        clustering_results,  # NEW
    )


if __name__ == "__main__":
    main()
