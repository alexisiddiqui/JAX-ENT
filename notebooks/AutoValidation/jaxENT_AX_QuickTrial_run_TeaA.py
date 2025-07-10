""" "
This script contains the runner to perform the quick autovalidaiton trial with the TeaA system.

The run function will simply call the TeaA quick trial function, which will perform the following:

1. Load the features and create the dataset
2. Perform a quick optimisation using the input parameters across multiple datasets
3. Analyse the resulting distribution of the features measure the L2 error to the refernce ratio (60: 40, Open:Closed) - returns (mean, sem) floats






output_dir = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/TeaA/quick_auto_validation_results"

open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = (
    "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"
)


"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
from jax import Array

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import os
from typing import Sequence

import jax
import jax.numpy as jnp
import MDAnalysis as mda
import numpy as np
from jaxENT_interpret_QuickAutoValidation_TeaA import (
    cluster_frames_by_rmsd,
    compute_rmsd_to_references,
    compute_weighted_cluster_ratios,
    compute_weighted_rmsd_distributions,
    load_all_optimization_histories,
    plot_cluster_ratio_boxplot,
    plot_cluster_ratios,
    plot_rmsd_distributions,
    plot_rmsd_histograms,
)
from jaxENT_run_AutoValidation_L2FWConsistency_TeaA import (
    create_synthetic_data as create_synthetic_data_complete,
)
from jaxENT_run_AutoValidation_L2only_TeaA import (
    create_synthetic_data,
    setup_simulation,
    visualize_optimization_results,
)
from scipy.spatial.distance import pdist, squareform

from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model_Config
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.opt.losses import (
    frame_weight_consistency_loss,
    hdx_uptake_l2_loss,
    maxent_convexKL_loss,
)
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import run_optimise
from jaxent.src.types.base import ForwardModel
from jaxent.src.types.config import OptimiserSettings
from jaxent.src.utils.hdf import (
    load_optimization_history_from_file,
    save_optimization_history_to_file,
)


def evaluate(
    output_dir: str,  # dir to load opt history and save results
    rmsd_values: np.ndarray,  # RMSD values for the trajectory
    cluster_assignments: np.ndarray,  # Cluster assignments for the trajectory
) -> tuple[float, float]:
    """
    Evaluate the performance of the model on the TeaA system.

    Parameters:
    output_dir (str): Directory to save the results.
    rmsd_values (np.ndarray): Precomputed RMSD values for the trajectory.
    cluster_assignments (np.ndarray): Precomputed cluster assignments for the trajectory.

    Returns:
    tuple: Mean and standard error of the mean (SEM) of the L2 error.
    """
    ref_names = ["Open", "Closed"]
    true_ratios = np.array([0.6, 0.4])  # True ratios for Open and Closed states
    # Load optimization histories
    print("Loading optimization histories...")
    histories = load_all_optimization_histories(output_dir)
    print("Computing weighted cluster ratios...")
    cluster_ratios, uniform_cluster_ratios = compute_weighted_cluster_ratios(
        cluster_assignments=cluster_assignments,
        histories=histories,
    )
    print("Computing weighted RMSD distributions...")
    rmsd_grid, kde_values, uniform_kde_values = compute_weighted_rmsd_distributions(
        rmsd_values, histories
    )
    # Plot RMSD distributions (KDE method)
    print("\nPlotting RMSD distributions with KDE...")
    rmsd_fig = plot_rmsd_distributions(rmsd_grid, kde_values, uniform_kde_values, ref_names)
    rmsd_path = os.path.join(output_dir, "rmsd_distributions_kde.png")
    rmsd_fig.savefig(rmsd_path, dpi=300, bbox_inches="tight")
    print(f"Saved KDE RMSD distributions to {rmsd_path}")

    # Plot RMSD distributions as histograms (no KDE)
    print("Plotting RMSD histograms (no KDE)...")
    hist_fig = plot_rmsd_histograms(rmsd_values, histories, ref_names, n_bins=25)
    hist_path = os.path.join(output_dir, "rmsd_histograms.png")
    hist_fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    print(f"Saved RMSD histograms to {hist_path}")

    # Plot cluster ratios
    print("Plotting cluster ratios...")
    ratio_fig = plot_cluster_ratios(cluster_ratios, uniform_cluster_ratios, ref_names, true_ratios)
    ratio_path = os.path.join(output_dir, "cluster_ratios.png")
    ratio_fig.savefig(ratio_path, dpi=300, bbox_inches="tight")
    print(f"Saved cluster ratios to {ratio_path}")

    # Plot box plot of cluster ratios
    print("Plotting cluster ratio box plot...")
    box_fig = plot_cluster_ratio_boxplot(
        cluster_ratios, uniform_cluster_ratios, ref_names, true_ratios
    )
    box_path = os.path.join(output_dir, "cluster_ratio_boxplot.png")
    box_fig.savefig(box_path, dpi=300, bbox_inches="tight")
    print(f"Saved cluster ratio box plot to {box_path}")

    # Calculate the MSE and SEM to true_ratios
    l2 = (cluster_ratios - true_ratios) ** 2
    sem = np.std(l2, axis=0) / np.sqrt(len(l2))
    mse = np.mean(l2, axis=(0, 1))

    # Ensure we return scalar values, not arrays
    mse_scalar = float(mse.item() if hasattr(mse, "item") else mse)
    sem_scalar = float(np.mean(sem).item() if hasattr(np.mean(sem), "item") else np.mean(sem))

    return mse_scalar, sem_scalar


def run_quick_auto_validation(
    features: Sequence[BV_input_features],
    datasets: list[ExpD_Dataloader],
    models: list[ForwardModel],
    params: Simulation_Parameters,
    opt_settings: OptimiserSettings,
    pairwise_similarity: Array,
    optimiser: OptaxOptimizer,
    output_dir: str,
) -> None:
    """
    Run the quick auto-validation trial for the TeaA system.
    """

    for i, dataset in enumerate(datasets):
        seed_results = []
        try:
            # Generate a new random key for each replicate
            print(f"Running optimization for Dataset Seed {i + 1}/{len(datasets)}")

            new_simulation = Simulation(
                forward_models=models,
                input_features=features,
                params=params,
            )
            new_simulation.initialise()
            new_simulation.forward(params)

            # Run optimization
            print(f"Running optimization for Dataset Seed {i + 1}/{len(datasets)}")
            _, opt_result = run_optimise(
                new_simulation,
                initialise=False,
                optimizer=optimiser,
                data_to_fit=(dataset, pairwise_similarity, params),
                config=opt_settings,
                forward_models=models,
                indexes=[0, 0, 0],
                loss_functions=[
                    hdx_uptake_l2_loss,
                    frame_weight_consistency_loss,
                    maxent_convexKL_loss,
                ],
            )

            # Store the result
            seed_results.append(opt_result)

            # Visualize this optimization run
            print(f"Visualizing results for Dataset Seed {i + 1}/{len(datasets)}")

            # Create seed output directory
            output_suff = os.path.join(output_dir, f"seed_{i + 1}_")
            os.makedirs(os.path.dirname(output_suff), exist_ok=True)

            # Fixed path construction with underscore separator
            hdf5_path = output_suff + "optimization_history.h5"
            save_optimization_history_to_file(hdf5_path, opt_result)
            print(f"Results saved to {hdf5_path}")

            # load back in
            loaded_history = load_optimization_history_from_file(
                hdf5_path, default_model_params_cls=BV_Model_Parameters
            )
            visualize_optimization_results(
                output_suff,
                train_data=datasets[0].train.data,
                val_data=datasets[0].val.data,
                exp_data=datasets[0].test.data,
                opt_history=loaded_history,
            )
            del new_simulation
            jax.clear_caches()
        except Exception as e:
            print(f"Error in Dataset Seed {i + 1} {e}")
            # Add a placeholder for failed runs to maintain indexing

    jax.clear_caches()


def train_evaluate(
    output_dir: str,
    rmsd_values: np.ndarray,
    cluster_assignments: np.ndarray,
    datasets: list[ExpD_Dataloader],
    bv_config: BV_model_Config,
    pairwise_similarity: Array,
    models: list[ForwardModel],
    features: Sequence[BV_input_features],
    optimizer: str = "adamw",
    learning_rate: float = 1e-3,
    l2_weight: float = 1e0,
    x_regularisation: float = 1e2,
    maxent_regularisation: float = 1e1,
    convergence: float = 1e-10,
) -> tuple[float, float]:
    supported_optimizers = ["adam", "sgd", "adamw"]

    assert optimizer in supported_optimizers, (
        f"Optimizer '{optimizer}' is not supported. Choose from {supported_optimizers}."
    )

    opt_settings = OptimiserSettings(
        name=optimizer
        + str(learning_rate)
        + str(l2_weight)
        + str(x_regularisation)
        + str(maxent_regularisation),
        n_steps=5000,
        convergence=convergence,
    )

    optimiser = OptaxOptimizer(
        learning_rate=learning_rate,
        optimizer=optimizer,
    )

    # Set the parameters for the optimizer
    params = Simulation_Parameters(
        frame_weights=jnp.ones(features[0].features_shape[2]),
        frame_mask=jnp.ones(features[0].features_shape[2]),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.asarray([l2_weight, x_regularisation, maxent_regularisation]),
        forward_model_scaling=jnp.ones(3),
        normalise_loss_functions=jnp.zeros(3),
    )

    # run the quick auto-validation trial
    run_quick_auto_validation(
        features=features,
        datasets=datasets,
        models=models,
        params=params,
        opt_settings=opt_settings,
        pairwise_similarity=pairwise_similarity,
        optimiser=optimiser,
        output_dir=output_dir,
    )

    return evaluate(output_dir, rmsd_values, cluster_assignments)


def setup_experiment(
    topology_path: str,
    trajectory_path: str,
    open_path: str,
    closed_path: str,
    seeds: int = 3,
) -> tuple:
    """
    Setup the simulation by loading the data and creating the datasets."
    """

    bv_config = BV_model_Config(num_timepoints=4)
    bv_config.timepoints = jnp.array([0.1, 1.0, 10.0, 100.0])

    reference_structures = [mda.Universe(open_path), mda.Universe(closed_path)]
    synthetic_data = create_synthetic_data(reference_structures, [0.6, 0.4], bv_config)
    datasets, models, opt_settings, features = setup_simulation(
        bv_config,
        seeds=list(range(seeds)),
        trajectory_path=trajectory_path,
        topology_path=topology_path,
        synthetic_data=synthetic_data,
        train_val_split=0.6,
    )

    universe = mda.Universe(topology_path, trajectory_path)
    ca_atoms = universe.select_atoms("name CA")

    ca_coords_by_frame = []

    for ts in universe.trajectory:
        ca_coords_by_frame.append(pdist(ca_atoms.positions).flatten())

    ca_coords_matrix = np.vstack(ca_coords_by_frame)

    cosine_distances = squareform(pdist(ca_coords_matrix, metric="cosine"))

    pairwise_similarity = jnp.array(cosine_distances)

    rmsd_values = compute_rmsd_to_references(
        trajectory_path, topology_path, [open_path, closed_path]
    )

    cluster_assignments = cluster_frames_by_rmsd(rmsd_values)

    return (
        datasets,
        models,
        opt_settings,
        features,
        bv_config,
        synthetic_data,
        rmsd_values,
        pairwise_similarity,
        cluster_assignments,
    )


def setup_experiment_complete(
    topology_path: str,
    trajectory_path: str,
    open_path: str,
    closed_path: str,
    seeds: int = 3,
) -> tuple:
    """
    Setup the simulation by loading the data and creating the datasets."
    """

    bv_config = BV_model_Config(num_timepoints=4)
    bv_config.timepoints = jnp.array([0.1, 1.0, 10.0, 100.0])

    reference_structures = [mda.Universe(open_path), mda.Universe(closed_path)]
    synthetic_data = create_synthetic_data_complete(
        ref_structures=reference_structures,
        ref_ratios=[0.6, 0.4],
        bv_config=bv_config,
        trajectory_path=trajectory_path,
        topology_path=topology_path,
    )
    datasets, models, opt_settings, features = setup_simulation(
        bv_config,
        seeds=list(range(seeds)),
        trajectory_path=trajectory_path,
        topology_path=topology_path,
        synthetic_data=synthetic_data,
        train_val_split=0.6,
    )

    universe = mda.Universe(topology_path, trajectory_path)
    ca_atoms = universe.select_atoms("name CA")

    ca_coords_by_frame = []

    for ts in universe.trajectory:
        ca_coords_by_frame.append(pdist(ca_atoms.positions).flatten())

    ca_coords_matrix = np.vstack(ca_coords_by_frame)

    cosine_distances = squareform(pdist(ca_coords_matrix, metric="cosine"))

    pairwise_similarity = jnp.array(cosine_distances)

    rmsd_values = compute_rmsd_to_references(
        trajectory_path, topology_path, [open_path, closed_path]
    )

    cluster_assignments = cluster_frames_by_rmsd(rmsd_values)

    return (
        datasets,
        models,
        opt_settings,
        features,
        bv_config,
        synthetic_data,
        rmsd_values,
        pairwise_similarity,
        cluster_assignments,
    )


output_dir = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/TeaA_testing/quick_AV_HyperParameterTuning"

open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
topology_path = open_path
trajectory_path = (
    "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"
)

if __name__ == "__main__":
    print("Running TeaA quick auto-validation test...")

    # Setup the experiment with fewer seeds for quicker testing
    (
        datasets,
        models,
        opt_settings,
        features,
        bv_config,
        synthetic_data,
        rmsd_values,
        pairwise_similarity,
        cluster_assignments,
    ) = setup_experiment(
        topology_path=topology_path,
        trajectory_path=trajectory_path,
        open_path=open_path,
        closed_path=closed_path,
        seeds=3,  # Reduced number of seeds for quick testing
    )

    # Create a test output directory
    test_output_dir = os.path.join(output_dir, "test_run")
    os.makedirs(test_output_dir, exist_ok=True)

    print(f"Running training and evaluation with output to: {test_output_dir}")

    # Run the training and evaluation
    mse, sem = train_evaluate(
        output_dir=test_output_dir,
        rmsd_values=rmsd_values,
        cluster_assignments=cluster_assignments,
        datasets=datasets,
        bv_config=bv_config,
        pairwise_similarity=pairwise_similarity,
        models=models,
        features=features,
        optimizer="adamw",
        learning_rate=1e-3,
        l2_weight=1.0,
        x_regularisation=100.0,
        maxent_regularisation=10.0,
    )

    print("\nEvaluation results:")
    print(f"Mean Squared Error (MSE) to reference ratio: {mse:.4f}")
    print(f"Standard Error of the Mean (SEM): {sem:.4f}")
    print("\nQuick auto-validation test completed successfully!")
