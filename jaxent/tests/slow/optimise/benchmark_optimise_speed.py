import json
import os
import time
from datetime import datetime

import jax

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # this shaves 0.2 seconds off the total time lmao

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
from jaxent.src.opt.losses import hdx_pf_l2_loss
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import run_optimise
from jaxent.src.types.config import FeaturiserSettings, Optimisable_Parameters, OptimiserSettings
from jaxent.src.types.HDX import HDX_protection_factor
from jaxent.tests.plots.datasplitting import plot_split_visualization
from jaxent.tests.plots.optimisation import (
    plot_frame_weights_heatmap,
    plot_loss_components,
    plot_total_losses,
)


# Ensure output directory exists
def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    output_dir = "tests/_plots/module_optimise"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def ensure_benchmark_dir():
    """Create the benchmark results directory if it doesn't exist."""
    output_dir = "tests/_benchmark/optimise"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# Add these visualizations to the test function
def visualize_optimization_results(train_data, val_data, exp_data, opt_simulation):
    """
    Create and display all visualization plots.
    """
    simulation, history = opt_simulation

    # Create all plots
    split_fig = plot_split_visualization(train_data, val_data, exp_data)
    total_loss_fig = plot_total_losses(history)
    loss_fig = plot_loss_components(history)
    weights_fig = plot_frame_weights_heatmap(history)

    # Display plots
    figs = [split_fig, total_loss_fig, loss_fig, weights_fig]
    fig_names = ["split", "total_loss", "loss", "weights"]
    output_dir = ensure_output_dir()
    for fig, names in zip(figs, fig_names):
        output_path = os.path.join(output_dir, f"{names}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)


def plot_benchmark_results(benchmark_results, output_file=None):
    """
    Plot benchmark results showing execution time across different configurations.
    """
    configs = [result["config_name"] for result in benchmark_results]
    times = [result["execution_time"] for result in benchmark_results]
    iterations = [result["num_iterations"] for result in benchmark_results]
    iterations_per_second = [it / time for it, time in zip(iterations, times)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot total execution time
    ax1.bar(configs, times, color="skyblue")
    ax1.set_title("Total Execution Time")
    ax1.set_xlabel("Configuration")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_ylim(bottom=0)

    # Plot iterations per second
    ax2.bar(configs, iterations_per_second, color="lightgreen")
    ax2.set_title("Optimization Speed")
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel("Iterations per second")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved benchmark plot to {output_file}")

    return fig


def setup_features():
    """Set up the ensemble, features, and dataset once for benchmarking."""
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    # topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo697_1_af_sample_127_10000_protonated.pdb"
    trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/scripts/HOIP_apo697_1_af_sample_127_10000_protonated_clusters500_20250429-142808/clusters/all_clusters.xtc"

    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]
    models = [BV_model(bv_config)]
    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Create fake experimental dataset
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)

    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(feature_topology[0], start=1)
    ]

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

    return (
        universes,
        models,
        features,
        feature_topology,
        dataset,
        train_data,
        val_data,
        exp_data,
        bv_config,
    )


def create_simulation_parameters(trajectory_length, bv_config):
    """Create simulation parameters for a given trajectory length and config"""
    return Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),  # Set all to 1 (active)
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )


def run_benchmark_optimisation(config_params, precomputed_data):
    """
    Run the optimization benchmark with different configurations.

    IMPORTANT: Create a fresh simulation for each benchmark run to avoid JAX hang issues
    """
    (
        universes,
        models,
        features,
        feature_topology,
        dataset,
        train_data,
        val_data,
        exp_data,
        bv_config,
    ) = precomputed_data
    benchmark_results = []

    # Get trajectory length from the features
    BV_features: BV_input_features = features[0]
    trajectory_length = BV_features.features_shape[2]

    for config in config_params:
        print(f"\n=== Running benchmark with {config['name']} ===")

        # Configure optimization settings
        opt_settings = OptimiserSettings(
            name=config["name"],
            learning_rate=config.get("learning_rate", 1e-4),
            n_steps=config.get("n_steps", 100),
            convergence=1e-20,  # hopefully this will make the optimization run until the end
        )

        # Create FRESH parameters for each benchmark run
        params = create_simulation_parameters(trajectory_length, bv_config)

        # Create a NEW simulation object for each benchmark
        simulation = Simulation(forward_models=models, input_features=features, params=params)

        # Initialize the simulation (success should be True)
        init_success = simulation.initialise()
        if not init_success:
            print(f"Failed to initialize simulation for {config['name']}")
            raise RuntimeError("Failed to initialize simulation")

        # Run a single forward pass to ensure everything is set up
        simulation.forward(params)

        # Run optimization with timing
        optimiser = OptaxOptimizer(optimizer=config.get("optimizer", "adam"))
        start_time = time.time()
        opt_simulation = run_optimise(
            simulation,
            optimizer=optimiser,
            data_to_fit=(dataset,),
            config=opt_settings,
            forward_models=[BV_model(BV_model_Config())],
            indexes=[0],
            initialise=False,
            loss_functions=[hdx_pf_l2_loss],
        )
        end_time = time.time()

        execution_time = end_time - start_time

        # Store results
        result = {
            "config_name": config["name"],
            "execution_time": execution_time,
            "num_iterations": opt_settings.n_steps,
            "iterations_per_second": opt_settings.n_steps / execution_time,
            "final_loss": float(opt_simulation[1].best_state.losses.total_train_loss),
            "parameters": config,
        }

        benchmark_results.append(result)
        print(
            f"Completed in {execution_time:.2f} seconds ({result['iterations_per_second']:.2f} iterations/sec)"
        )
        print(f"Final loss: {result['final_loss']}")

    # Create one last simulation for visualization
    final_params = create_simulation_parameters(trajectory_length, bv_config)
    final_simulation = Simulation(
        forward_models=models, input_features=features, params=final_params
    )
    final_simulation.initialise()
    final_simulation.forward(final_params)

    return benchmark_results, final_simulation, dataset, train_data, val_data, exp_data


def test_quick_optimiser():
    # Different optimization configurations to benchmark
    configurations = [
        {"name": "adam_default", "optimizer": "adam", "learning_rate": 1e-4, "n_steps": 1000},
        # {"name": "adam_fast", "optimizer": "adam", "learning_rate": 5e-4, "n_steps": 200},
        {"name": "sgd_default", "optimizer": "sgd", "learning_rate": 1e-3, "n_steps": 1000},
    ]

    # Compute features once before running benchmarks
    print("Performing featurization once before benchmarks...")
    precomputed_data = setup_features()

    # Run benchmarks with precomputed features
    # This creates a fresh simulation for each benchmark
    benchmark_results, final_simulation, dataset, train_data, val_data, exp_data = (
        run_benchmark_optimisation(configurations, precomputed_data)
    )

    # Save benchmark results
    benchmark_dir = ensure_benchmark_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save as JSON
    results_file = os.path.join(benchmark_dir, f"optim_benchmark_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    # Generate plot
    plot_file = os.path.join(benchmark_dir, f"optim_benchmark_{timestamp}.png")
    plot_benchmark_results(benchmark_results, plot_file)

    # Find the best configuration
    best_config_idx = np.argmin([result["final_loss"] for result in benchmark_results])
    print(
        f"\nBest configuration: {configurations[best_config_idx]['name']} with loss {benchmark_results[best_config_idx]['final_loss']}"
    )

    # Run the best configuration again with the final simulation object
    best_opt_settings = OptimiserSettings(
        name=configurations[best_config_idx]["name"],
        learning_rate=configurations[best_config_idx]["learning_rate"],
        n_steps=configurations[best_config_idx]["n_steps"],
        convergence=1e-20,  # hopefully this will make the optimization run until the end
    )
    optimiser = OptaxOptimizer(
        optimizer=configurations[best_config_idx]["optimizer"],
        parameter_masks={
            Optimisable_Parameters.frame_weights,
            # Optimisable_Parameters.model_parameters,
        },
    )

    opt_simulation = run_optimise(
        final_simulation,  # Use the final simulation created specifically for visualization
        optimizer=optimiser,
        data_to_fit=(dataset,),
        config=best_opt_settings,
        forward_models=[BV_model(BV_model_Config())],
        indexes=[0],
        loss_functions=[hdx_pf_l2_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)

    print(f"\nBenchmark results saved to {results_file}")
    print(f"Benchmark plot saved to {plot_file}")


if __name__ == "__main__":
    import jax

    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))
    # set default device to CPU

    # Disable memory preallocation - must be defined at the top of the script
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # os.environ["JAX_PLATFORM_NAME"] = "cpu"
    # jax.config.update("jax_platform_name", "cpu")

    # Run benchmark
    test_quick_optimiser()
