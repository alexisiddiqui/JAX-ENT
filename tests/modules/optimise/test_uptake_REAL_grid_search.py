import itertools
import json
import os
import time
from datetime import datetime

import jax.numpy as jnp
import numpy as np

# Configure JAX environment
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from MDAnalysis import Universe

from jaxent.src.data.loader import Dataset, ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.func.common import find_common_residues
from jaxent.src.models.HDX.BV.forwardmodel import BV_input_features, BV_model
from jaxent.src.opt.losses import (
    HDX_uptake_KL_loss,
    HDX_uptake_MAE_loss,
    hdx_uptake_l2_loss,
    max_entropy_loss,
)
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import run_optimise
from jaxent.src.types.config import FeaturiserSettings, Optimisable_Parameters, OptimiserSettings
from jaxent.src.types.HDX import HDX_peptide


# Ensure output directories exist
def ensure_output_dir(name="grid_search"):
    """Create the output directory if it doesn't exist."""
    output_dir = f"tests/_plots/{name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def ensure_results_dir(name="grid_search"):
    """Create the results directory if it doesn't exist."""
    output_dir = f"tests/_results/{name}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_grid_values(start, end, intervals):
    """Generate grid values between start and end with the given number of intervals."""
    return np.linspace(start, end, intervals)


def grid_search_forward_model_scaling(
    # Grid search parameters
    scale_ranges=[
        (0.1, 10.0, 3),  # (start, end, intervals) for dimension 0
        (1.0, 20.0, 3),  # (start, end, intervals) for dimension 1
        (1.0, 20.0, 3),  # (start, end, intervals) for dimension 2
        (1.0, 20.0, 3),  # (start, end, intervals) for dimension 3
    ],
    n_optimization_steps=100,  # Reduced for faster grid search
    output_base_name="scaling_grid_search",
):
    """
    Perform a grid search over the forward_model_scaling parameter space.

    Parameters:
        scale_ranges: List of tuples (start, end, intervals) for each dimension
        n_optimization_steps: Number of optimization steps for each run
        output_base_name: Base name for output files
    """
    # Generate grid values for each dimension
    grid_values = [
        generate_grid_values(start, end, intervals) for start, end, intervals in scale_ranges
    ]

    # Calculate total number of combinations
    total_combinations = np.prod([len(values) for values in grid_values])
    print(f"Total combinations to evaluate: {total_combinations}")

    # Initialize results container
    all_results = []

    # Get current timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{output_base_name}_{timestamp}"

    # Create output directories
    plots_dir = ensure_output_dir(output_name)
    results_dir = ensure_results_dir(output_name)

    # Setup the model, features, and dataset only once
    print("Setting up model, features, and dataset...")
    setup_data = setup_model_and_data()

    # Create a counter for tracking progress
    combination_counter = 0

    # Generate all combinations of scaling parameters
    for scaling_values in itertools.product(*grid_values):
        combination_counter += 1
        print(f"\nEvaluating combination {combination_counter}/{total_combinations}")
        print(f"Scaling values: {scaling_values}")

        # Convert to jnp array
        scaling = jnp.array(scaling_values)

        # Run optimization with these scaling parameters
        result = run_optimization_with_scaling(setup_data, scaling, n_optimization_steps)

        # Store results
        all_results.append(
            {
                "forward_model_scaling": scaling_values,
                "final_train_loss": float(result["final_train_loss"]),
                "final_val_loss": float(result["final_val_loss"]),
                "best_train_loss": float(result["best_train_loss"]),
                "best_val_loss": float(result["best_val_loss"]),
                # Add unscaled sums
                "sum_best_train": float(result["sum_best_train"]),
                "sum_best_val": float(result["sum_best_val"]),
                # Add final components for reference
                "final_train_components": result["final_train_components"],
                "final_val_components": result["final_val_components"],
                "best_train_components": result["best_train_components"],
                "best_val_components": result["best_val_components"],
                "optimization_steps": n_optimization_steps,
                "actual_steps": result["actual_steps"],
                "loss_history": {
                    "train": [float(x) for x in result["train_loss_history"]],
                    "val": [float(x) for x in result["val_loss_history"]],
                },
            }
        )

        # Save individual result plot
        plot_individual_result(result, scaling_values, plots_dir, combination_counter)

        # Save results periodically
        if combination_counter % 5 == 0 or combination_counter == total_combinations:
            save_results(all_results, results_dir, output_name)

    # Create summary plots
    create_summary_plots(all_results, plots_dir)

    # Return results
    return all_results


# Rest of the setup code remains unchanged
def setup_model_and_data():
    """Set up the model, features, and dataset only once."""
    # Implementation unchanged
    bv_config = BV_model_Config(num_timepoints=3)
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    # Paths to data files
    topology_path = "./tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "./tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    segs_data = "./notebooks/CrossValidation/BPTI/BPTI_residue_segs_trimmed.txt"
    dfrac_data = "./notebooks/CrossValidation/BPTI/BPTI_expt_dfracs_clean_trimmed.dat"

    # Try alternative paths if files don't exist
    if not os.path.exists(topology_path):
        topology_path = "./BPTI_overall_combined_stripped.pdb"
    if not os.path.exists(trajectory_path):
        trajectory_path = "./BPTI_sampled_500.xtc"
    if not os.path.exists(segs_data):
        segs_data = "./BPTI_residue_segs_trimmed.txt"
    if not os.path.exists(dfrac_data):
        dfrac_data = "./BPTI_expt_dfracs_clean_trimmed.dat"

    # Read segment data
    with open(segs_data, "r") as f:
        segs_text = [line.strip() for line in f.readlines()]
        segs = [line.split() for line in segs_text]

    segs = [[start, end] for start, end in segs]
    exp_residues = [int(seg[1]) for seg in segs]

    # Create universe
    test_universe = Universe(topology_path, trajectory_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]
    ensemble = Experiment_Builder(universes, models)

    # Run featurization
    features, feature_topology = run_featurise(ensemble, featuriser_settings)
    exp_topology = [top for top in feature_topology[0] if top.residue_end in exp_residues]

    # Read experimental data
    with open(dfrac_data, "r") as f:
        dfrac_text = [line.strip() for line in f.readlines()[1:]]
        dfracs = [line.split() for line in dfrac_text]

    dfracs = [jnp.array(line, dtype=float) for line in dfracs]

    # Get feature information
    BV_features: BV_input_features = features[0]
    trajectory_length = BV_features.features_shape[2]

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)

    # Create experimental dataset
    exp_data = [HDX_peptide(dfrac=_dfrac, top=top) for _dfrac, top in zip(dfracs, exp_topology)]
    dataset = ExpD_Dataloader(data=exp_data)

    # Create random split
    splitter = DataSplitter(
        dataset,
        random_seed=42,
        ensemble=universes,
        common_residues=set(feature_topology[0]),
    )
    train_data, val_data = splitter.random_split()

    # Create sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)
    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)
    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    # Set up datasets
    dataset.train = Dataset(
        data=train_data,
        y_true=jnp.array([data.extract_features() for data in train_data]),
        residue_feature_ouput_mapping=train_sparse_map,
    )
    dataset.val = Dataset(
        data=val_data,
        y_true=jnp.array([data.extract_features() for data in val_data]),
        residue_feature_ouput_mapping=val_sparse_map,
    )
    dataset.test = Dataset(
        data=exp_data,
        y_true=jnp.array([data.extract_features() for data in exp_data]),
        residue_feature_ouput_mapping=test_sparse_map,
    )

    # Set up simulation initially to get prior_data
    initial_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(4),
        forward_model_scaling=jnp.array([1.0, 10.0, 10.0, 10.0]),
        normalise_loss_functions=jnp.zeros(4),
    )

    initial_simulation = Simulation(
        forward_models=models, input_features=features, params=initial_params
    )
    initial_simulation.initialise()
    initial_simulation.forward(initial_params)

    # Get prior data
    test_prediction = initial_simulation.outputs
    prior_pfs = test_prediction[0].uptake.T
    prior_data = [
        HDX_peptide(dfrac=_prior_df, top=top) for _prior_df, top in zip(prior_pfs, exp_topology)
    ]

    # Create prior dataset
    prior_dataset = ExpD_Dataloader(data=prior_data)
    prior_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_data)
    prior_dataset.train = Dataset(
        data=prior_data,
        y_true=jnp.array([data.extract_features() for data in prior_data]),
        residue_feature_ouput_mapping=prior_sparse_map,
    )
    prior_dataset.val = Dataset(
        data=prior_data,
        y_true=jnp.array([data.extract_features() for data in prior_data]),
        residue_feature_ouput_mapping=prior_sparse_map,
    )
    prior_dataset.test = Dataset(
        data=prior_data,
        y_true=jnp.array([data.extract_features() for data in prior_data]),
        residue_feature_ouput_mapping=prior_sparse_map,
    )

    # Return all the setup data
    return {
        "bv_config": bv_config,
        "features": features,
        "feature_topology": feature_topology,
        "trajectory_length": trajectory_length,
        "models": models,
        "dataset": dataset,
        "prior_dataset": prior_dataset,
        "train_data": train_data,
        "val_data": val_data,
        "exp_data": exp_data,
    }


def run_optimization_with_scaling(setup_data, scaling, n_steps):
    """Run optimization with the specified scaling parameters."""
    # Implementation unchanged
    # Extract setup data
    bv_config = setup_data["bv_config"]
    features = setup_data["features"]
    trajectory_length = setup_data["trajectory_length"]
    models = setup_data["models"]
    dataset = setup_data["dataset"]
    prior_dataset = setup_data["prior_dataset"]
    train_data = setup_data["train_data"]
    val_data = setup_data["val_data"]
    exp_data = setup_data["exp_data"]

    # Create parameters with the specified scaling
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(4),
        forward_model_scaling=scaling,
        normalise_loss_functions=jnp.zeros(4),
    )

    # Create simulation
    simulation = Simulation(forward_models=models, input_features=features, params=params)
    simulation.initialise()
    simulation.forward(params)

    # Set up optimizer
    opt_settings = OptimiserSettings(
        name="grid_search", n_steps=n_steps, convergence=1e-5, tolerance=1e-5, learning_rate=1e-3
    )

    optimiser = OptaxOptimizer(
        parameter_masks={Optimisable_Parameters.frame_weights},
    )

    # Run optimization
    start_time = time.time()
    opt_simulation = run_optimise(
        simulation,
        optimizer=optimiser,
        data_to_fit=(
            dataset,
            simulation.params,
            prior_dataset,
            prior_dataset,
        ),
        config=opt_settings,
        forward_models=models,
        indexes=[0, 0, 0, 0],
        loss_functions=[
            hdx_uptake_l2_loss,
            max_entropy_loss,
            HDX_uptake_MAE_loss,
            HDX_uptake_KL_loss,
        ],
    )
    end_time = time.time()

    # Extract results
    sim, history = opt_simulation

    # Get loss histories
    train_loss_history = [state.losses.total_train_loss for state in history.states]
    val_loss_history = [state.losses.total_val_loss for state in history.states]

    # Also extract unscaled loss components
    unscaled_train_history = [state.losses.train_losses for state in history.states]
    unscaled_val_history = [state.losses.val_losses for state in history.states]

    # Sum of unscaled components
    sum_unscaled_train = [jnp.sum(losses) for losses in unscaled_train_history]
    sum_unscaled_val = [jnp.sum(losses) for losses in unscaled_val_history]

    # Final unscaled loss components
    final_train_components = unscaled_train_history[-1]
    final_val_components = unscaled_val_history[-1]

    # Best state's unscaled loss components
    best_train_components = history.best_state.losses.train_losses
    best_val_components = history.best_state.losses.val_losses

    # Return results
    return {
        "scaling": scaling,
        "simulation": sim,
        "history": history,
        "train_data": train_data,
        "val_data": val_data,
        "exp_data": exp_data,
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "unscaled_train_history": unscaled_train_history,
        "unscaled_val_history": unscaled_val_history,
        "sum_unscaled_train": sum_unscaled_train,
        "sum_unscaled_val": sum_unscaled_val,
        "final_train_loss": train_loss_history[-1],
        "final_val_loss": val_loss_history[-1],
        "final_train_components": [float(x) for x in final_train_components],
        "final_val_components": [float(x) for x in final_val_components],
        "best_train_loss": history.best_state.losses.total_train_loss,
        "best_val_loss": history.best_state.losses.total_val_loss,
        "best_train_components": [float(x) for x in best_train_components],
        "best_val_components": [float(x) for x in best_val_components],
        "sum_best_train": float(jnp.sum(best_train_components)),
        "sum_best_val": float(jnp.sum(best_val_components)),
        "execution_time": end_time - start_time,
        "actual_steps": len(train_loss_history),
    }


def plot_individual_result(result, scaling_values, plots_dir, combination_index):
    """Plot the individual optimization result with focus on validation loss components."""
    # Create figure with 2 rows
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(2, 2, figure=fig)

    # Define component names
    component_names = ["ExpL2", "MaxENT", "MAE", "MRE"]

    # Row 1: Validation Loss Components
    ax1 = fig.add_subplot(gs[0, :])

    # Plot validation loss components
    component_count = len(result["final_val_components"])
    for i in range(component_count):
        component_history = [float(step[i]) for step in result["unscaled_val_history"]]
        component_label = component_names[i] if i < len(component_names) else f"Component {i}"
        ax1.plot(component_history, label=f"Val {component_label}")

    # Add labels and title
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Validation Loss Component Value")
    ax1.set_title(f"Validation Loss Components for Scaling: {[float(s) for s in scaling_values]}")
    ax1.legend()

    # Add text with scaling values and component information
    val_components_text = ", ".join(
        [f"{float(comp):.4f}" for comp in result["best_val_components"]]
    )
    component_labels = []
    for i in range(component_count):
        comp_name = component_names[i] if i < len(component_names) else f"Component {i}"
        comp_value = float(result["best_val_components"][i])
        component_labels.append(f"{comp_name}: {comp_value:.4f}")

    textstr = "\n".join(
        [
            f"Scaling: {[float(s) for s in scaling_values]}",
            f"{component_labels[0]}",
            f"All Components: [{val_components_text}]",
            f"Steps: {result['actual_steps']}",
            f"Time: {float(result['execution_time']):.2f}s",
        ]
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax1.text(
        0.05,
        0.95,
        textstr,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    # Row 2: Pairwise analysis of validation loss components
    # Extract validation component histories
    val_comp_histories = {}
    for i in range(component_count):
        comp_name = component_names[i] if i < len(component_names) else f"Component {i}"
        val_comp_histories[comp_name] = [float(step[i]) for step in result["unscaled_val_history"]]

    # Create dataframe for component analysis
    df_val_history = pd.DataFrame(val_comp_histories)

    # If multiple components, show pairwise relationships
    if component_count >= 2:
        # Left plot: ExpL2 vs MaxENT
        ax2 = fig.add_subplot(gs[1, 0])
        sns.scatterplot(data=df_val_history, x=component_names[0], y=component_names[1], ax=ax2)
        ax2.set_title(f"{component_names[0]} vs {component_names[1]}")

        # Right plot: ExpL2 vs another component or histogram
        ax3 = fig.add_subplot(gs[1, 1])
        if component_count >= 3:
            sns.scatterplot(data=df_val_history, x=component_names[0], y=component_names[2], ax=ax3)
            ax3.set_title(f"{component_names[0]} vs {component_names[2]}")
        else:
            # Only two components, show distribution of ExpL2
            sns.histplot(data=df_val_history, x=component_names[0], kde=True, ax=ax3)
            ax3.set_title(f"Distribution of {component_names[0]}")
    else:
        # Only one component, show its distribution
        ax2 = fig.add_subplot(gs[1, :])
        sns.histplot(data=df_val_history, x=component_names[0], kde=True, ax=ax2)
        ax2.set_title(f"Distribution of {component_names[0]}")

    # Save figure
    fig_path = os.path.join(plots_dir, f"combination_{combination_index}.png")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Create additional heatmap/correlation matrix if multiple components
    if component_count > 1:
        fig2 = plt.figure(figsize=(10, 8))
        # Create correlation matrix
        corr_matrix = df_val_history.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Matrix of Validation Loss Components")
        fig2_path = os.path.join(plots_dir, f"val_corr_matrix_{combination_index}.png")
        fig2.tight_layout()
        fig2.savefig(fig2_path, dpi=300, bbox_inches="tight")
        plt.close(fig2)


def save_results(all_results, results_dir, output_name):
    """Save all results to a JSON file."""
    results_path = os.path.join(results_dir, f"{output_name}.json")

    # Convert jnp arrays to lists for JSON serialization
    results_for_json = []
    for result in all_results:
        result_copy = result.copy()
        result_copy["forward_model_scaling"] = [float(x) for x in result["forward_model_scaling"]]
        results_for_json.append(result_copy)

    with open(results_path, "w") as f:
        json.dump(results_for_json, f, indent=2)

    print(f"Saved results to {results_path}")


def create_summary_plots(all_results, plots_dir):
    """Create summary plots focusing on validation loss components."""
    # Define component names
    component_names = ["ExpL2", "MaxENT", "MAE", "MRE"]

    # Extract validation components from results
    validation_components = []
    for result in all_results:
        validation_components.append([float(comp) for comp in result["best_val_components"]])

    # Get component count
    component_count = len(validation_components[0])

    # Create dataframe for components
    df_columns = [
        component_names[i] if i < len(component_names) else f"Component {i}"
        for i in range(component_count)
    ]
    df_components = pd.DataFrame(validation_components, columns=df_columns)

    # Add scaling parameters and index
    df_components["Scaling"] = [
        str([float(s) for s in result["forward_model_scaling"]]) for result in all_results
    ]
    df_components["Config"] = range(len(all_results))

    # Add individual scaling parameter columns for plotting
    for i in range(len(all_results[0]["forward_model_scaling"])):
        df_components[f"Scale_{i}"] = [
            float(result["forward_model_scaling"][i]) for result in all_results
        ]

    # Sort by first component (ExpL2)
    sorted_by_comp0 = df_components.sort_values(component_names[0]).reset_index(drop=True)

    # Create parameter variation plots
    create_2d_surface_plots(df_components, component_names[0], plots_dir)
    create_1d_slice_plots(df_components, component_names[0], plots_dir)

    # Plot 1: Bar chart of top configurations by ExpL2
    plt.figure(figsize=(12, 8))
    top_n = min(10, len(sorted_by_comp0))

    # Get indices of top configurations
    top_configs = sorted_by_comp0.iloc[:top_n]

    # Create bar chart for ExpL2
    plt.bar(range(top_n), top_configs[component_names[0]], color="steelblue")
    plt.xlabel("Configuration")
    plt.ylabel(f"Value of {component_names[0]}")
    plt.title(f"Top Configurations by {component_names[0]} Validation Loss Component")
    plt.xticks(range(top_n), top_configs["Scaling"], rotation=45, ha="right")
    plt.tight_layout()

    # Save plot
    plt.savefig(
        os.path.join(plots_dir, f"top_by_{component_names[0]}.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 2: Comparison of all validation components for top configurations
    fig, ax = plt.subplots(figsize=(14, 10))

    # Set width based on number of components
    width = 0.8 / component_count
    x = np.arange(top_n)

    # Plot each component
    for i in range(component_count):
        comp_name = component_names[i] if i < len(component_names) else f"Component {i}"
        comp_values = top_configs[comp_name]
        ax.bar(x + (i - component_count / 2 + 0.5) * width, comp_values, width, label=comp_name)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Component Value")
    ax.set_title("Validation Loss Components for Top Configurations")
    ax.set_xticks(x)
    ax.set_xticklabels(top_configs["Scaling"], rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()

    # Save plot
    plt.savefig(os.path.join(plots_dir, "component_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 3: Pairwise scatter plots of validation components
    if component_count >= 2:
        fig, axes = plt.subplots(1, component_count - 1, figsize=(5 * component_count, 6))

        # Handle case of only 2 components (1 comparison)
        if component_count == 2:
            axes = [axes]

        # Create scatter plots: ExpL2 vs each other component
        for i in range(1, component_count):
            comp1_name = component_names[0]
            comp2_name = component_names[i] if i < len(component_names) else f"Component {i}"

            axes[i - 1].scatter(df_components[comp1_name], df_components[comp2_name])
            axes[i - 1].set_xlabel(comp1_name)
            axes[i - 1].set_ylabel(comp2_name)
            axes[i - 1].set_title(f"{comp1_name} vs {comp2_name}")

            # Add config numbers as annotations
            for j, config in enumerate(df_components["Config"]):
                axes[i - 1].annotate(
                    str(config),
                    (df_components[comp1_name].iloc[j], df_components[comp2_name].iloc[j]),
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "component_pairplots.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Plot 4: Table of all configurations sorted by ExpL2
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, len(all_results) * 0.4 + 2))
    ax.axis("off")

    # Prepare table data
    table_data = []
    for idx, row in sorted_by_comp0.iterrows():
        # Get the original result
        result = all_results[int(row["Config"])]

        # Format component values
        component_values = []
        for i in range(component_count):
            comp_name = component_names[i] if i < len(component_names) else f"Component {i}"
            component_values.append(f"{float(row[comp_name]):.4f}")

        # Add to table data
        table_data.append(
            [
                idx + 1,
                row["Config"],
                row["Scaling"],
                component_values[0],  # ExpL2
                " | ".join(component_values[1:]),  # Other components
                f"{float(result['sum_best_val']):.4f}",  # Sum for reference
            ]
        )

    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=["Rank", "Config #", "Scaling", component_names[0], "Other Components", "Sum"],
        loc="center",
        cellLoc="center",
    )

    # Format table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.savefig(os.path.join(plots_dir, "configurations_table.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 5: Heatmap of correlation between components
    if component_count > 1:
        plt.figure(figsize=(8, 6))
        component_cols = [
            component_names[i] if i < len(component_names) else f"Component {i}"
            for i in range(component_count)
        ]
        corr_matrix = df_components[component_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Between Validation Loss Components")
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, "component_correlation.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()


def create_2d_surface_plots(df, target_component, plots_dir):
    """
    Create 2D surface plots showing how the target component varies with pairs of parameters.

    Parameters:
        df: DataFrame containing the component values and scaling parameters
        target_component: Name of the component to visualize (e.g., "ExpL2")
        plots_dir: Directory to save the plots
    """
    scale_dims = [col for col in df.columns if col.startswith("Scale_")]
    n_dims = len(scale_dims)

    if n_dims < 2:
        return  # Need at least 2 dimensions for surface plots

    # Create plots for each pair of dimensions
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            dim1 = scale_dims[i]
            dim2 = scale_dims[j]

            # Create a figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Create scatter plot with color representing the target component
            scatter = ax.scatter(
                df[dim1], df[dim2], c=df[target_component], cmap="viridis", s=100, alpha=0.7
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(target_component)

            # Label points with their configuration index
            for idx, row in df.iterrows():
                ax.annotate(
                    str(idx),
                    (row[dim1], row[dim2]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

            # Set labels and title
            ax.set_xlabel(f"Scaling Parameter {i}")
            ax.set_ylabel(f"Scaling Parameter {j}")
            ax.set_title(f"{target_component} vs Scaling Parameters {i} and {j}")

            # Add grid
            ax.grid(True, linestyle="--", alpha=0.6)

            # Save plot
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, f"{target_component}_surface_{i}_{j}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            # If we have enough points, try to create a contour plot
            if len(df) >= 10:
                try:
                    # Create a figure for contour plot
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Create a grid for contour plot
                    x_unique = sorted(df[dim1].unique())
                    y_unique = sorted(df[dim2].unique())

                    if len(x_unique) > 1 and len(y_unique) > 1:
                        # Create meshgrid
                        X, Y = np.meshgrid(x_unique, y_unique)
                        Z = np.zeros_like(X)

                        # Fill Z values
                        for x_idx, x_val in enumerate(x_unique):
                            for y_idx, y_val in enumerate(y_unique):
                                # Find rows matching these x,y values
                                matches = df[(df[dim1] == x_val) & (df[dim2] == y_val)]
                                if not matches.empty:
                                    Z[y_idx, x_idx] = matches[target_component].values[0]

                        # Create contour plot
                        contour = ax.contourf(X, Y, Z, cmap="viridis", levels=15)
                        cbar = plt.colorbar(contour, ax=ax)
                        cbar.set_label(target_component)

                        # Set labels and title
                        ax.set_xlabel(f"Scaling Parameter {i}")
                        ax.set_ylabel(f"Scaling Parameter {j}")
                        ax.set_title(f"{target_component} Contour vs Parameters {i} and {j}")

                        # Save contour plot
                        plt.tight_layout()
                        plt.savefig(
                            os.path.join(plots_dir, f"{target_component}_contour_{i}_{j}.png"),
                            dpi=300,
                            bbox_inches="tight",
                        )
                        plt.close()
                except Exception as e:
                    print(f"Could not create contour plot for params {i} and {j}: {e}")
            else:
                # Enhanced scatter plot for fewer than 10 points
                fig, ax = plt.subplots(figsize=(10, 8))

                # Create scatter plot with larger points
                scatter = ax.scatter(
                    df[dim1],
                    df[dim2],
                    c=df[target_component],
                    cmap="viridis",
                    s=200,
                    alpha=0.8,
                    edgecolors="black",
                )

                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(target_component)

                # Annotate each point with its value
                for idx, row in df.iterrows():
                    ax.annotate(
                        f"{row[target_component]:.3f}",
                        (row[dim1], row[dim2]),
                        textcoords="offset points",
                        xytext=(0, -15),
                        ha="center",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
                    )

                    # Also annotate with the config number
                    ax.annotate(
                        f"#{idx}",
                        (row[dim1], row[dim2]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=9,
                        weight="bold",
                    )

                # Set labels and title
                ax.set_xlabel(f"Scaling Parameter {i}")
                ax.set_ylabel(f"Scaling Parameter {j}")
                ax.set_title(f"{target_component} vs Scaling Parameters {i} and {j} (Sparse Data)")

                # Add grid
                ax.grid(True, linestyle="--", alpha=0.6)

                # Save plot
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, f"{target_component}_annotated_scatter_{i}_{j}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()


def create_1d_slice_plots(df, target_component, plots_dir):
    """
    Create 1D slice plots showing how the target component varies with each parameter.

    Parameters:
        df: DataFrame containing the component values and scaling parameters
        target_component: Name of the component to visualize (e.g., "ExpL2")
        plots_dir: Directory to save the plots
    """
    scale_dims = [col for col in df.columns if col.startswith("Scale_")]
    n_dims = len(scale_dims)

    # Find the configuration with the minimum target component value
    best_config = df.loc[df[target_component].idxmin()]

    # Create plots for each dimension
    for i in range(n_dims):
        dim = scale_dims[i]

        # Group by this dimension, aggregating by mean for the target component
        grouped = df.groupby(dim)[target_component].agg(["mean", "min", "max", "count"])
        grouped = grouped.reset_index()

        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the mean with error bars for min/max
        ax.errorbar(
            grouped[dim],
            grouped["mean"],
            yerr=[grouped["mean"] - grouped["min"], grouped["max"] - grouped["mean"]],
            fmt="o-",
            capsize=5,
            label=f"Mean {target_component} (with min/max range)",
        )

        # Highlight the best value
        ax.axvline(
            x=best_config[dim],
            color="r",
            linestyle="--",
            label=f"Best value: {best_config[dim]:.2f}",
        )

        # Add point counts as annotations
        for _, row in grouped.iterrows():
            ax.annotate(
                f"n={int(row['count'])}",
                (row[dim], row["mean"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        # Set labels and title
        ax.set_xlabel(f"Scaling Parameter {i}")
        ax.set_ylabel(f"Mean {target_component}")
        ax.set_title(f"{target_component} vs Scaling Parameter {i}")

        # Add grid and legend
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{target_component}_slice_{i}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create a scatter plot showing all points
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot all data points
        ax.scatter(df[dim], df[target_component], alpha=0.7)

        # Highlight the best value
        ax.axvline(
            x=best_config[dim],
            color="r",
            linestyle="--",
            label=f"Best value: {best_config[dim]:.2f}",
        )

        # Set labels and title
        ax.set_xlabel(f"Scaling Parameter {i}")
        ax.set_ylabel(target_component)
        ax.set_title(f"All {target_component} Values vs Scaling Parameter {i}")

        # Add grid and legend
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{target_component}_scatter_{i}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))

    # Define your grid search parameters here
    scale_ranges = [
        (0.1, 100.0, 2),  # (start, end, intervals) for dimension 0
        (1, 100.0, 2),  # (start, end, intervals) for dimension 1
        (1, 100.0, 2),  # (start, end, intervals) for dimension 2
        (1, 100.0, 2),  # (start, end, intervals) for dimension 3
    ]

    # Run grid search with specified parameters
    all_results = grid_search_forward_model_scaling(
        scale_ranges=scale_ranges,
        n_optimization_steps=200,  # Number of optimization steps per configuration
        output_base_name="scaling_grid_search",
    )

    # Find best configuration based on the first component of unscaled validation loss
    best_result = min(all_results, key=lambda x: float(x["best_val_components"][0]))
    component_names = ["ExpL2", "MaxENT", "MAE", "MRE"]

    print("\nBest configuration (by ExpL2 validation loss component):")
    print(f"Scaling: {[float(s) for s in best_result['forward_model_scaling']]}")
    print(f"ExpL2 validation loss: {float(best_result['best_val_components'][0])}")

    print("All validation loss components:")
    for i, comp in enumerate(best_result["best_val_components"]):
        comp_name = component_names[i] if i < len(component_names) else f"Component {i}"
        print(f"  {comp_name}: {float(comp)}")

    print(f"Validation loss (unscaled sum): {float(best_result['sum_best_val'])}")

    # Also find the best by total sum for comparison
    best_by_sum = min(all_results, key=lambda x: float(x["sum_best_val"]))
    print("\nBest configuration (by sum of validation loss components):")
    print(f"Scaling: {[float(s) for s in best_by_sum['forward_model_scaling']]}")

    print("All validation loss components:")
    for i, comp in enumerate(best_by_sum["best_val_components"]):
        comp_name = component_names[i] if i < len(component_names) else f"Component {i}"
        print(f"  {comp_name}: {float(comp)}")

    print(f"Validation loss (unscaled sum): {float(best_by_sum['sum_best_val'])}")
