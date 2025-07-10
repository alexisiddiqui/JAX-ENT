"""
This script runs auto-validation (simulation) on the L2-only data using jaxent.src. Interpretation and Analysis of the results is done in a separate script.

To run this script, you require the following:
- input ensemble that contains at least 2 states (with names)
- reference states to generate synthetic data (desired ratios - default 60:40)

This script essentially creates synthetic data using the reference states and their ratios.
We then attempt to fit the ensemble to the synthetic data using jaxENT over a number of datasplits (seeds).
By interpreting the Probability Density of the RMSD of the optimised and unoptimised states we can demonstrate that jaxENT is able to recover the data (seperate script)

We repeat this over a range of train/validation splits (0.1, 0.25, 0.5, 0.75, 0.9) (10 replicates each)

Additionally we optimise over various levels of initial starting noise (0.1, 0.25, 0.5, 0.75, 0.9) (10 replicates each)

"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.spatial.distance import pdist, squareform

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from typing import Sequence

import jax
import matplotlib.pyplot as plt
from jaxENT_run_AutoValidation_L2only_TeaA import (
    create_synthetic_data,
    setup_simulation,
    visualize_optimization_results,
)
from MDAnalysis import Universe

from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model_Config
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.opt.losses import hdx_uptake_mean_centred_MSE_loss, maxent_L2_loss
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters
from jaxent.src.opt.run import run_optimise
from jaxent.src.types.base import ForwardModel
from jaxent.src.types.config import OptimiserSettings
from jaxent.src.utils.hdf import (
    load_optimization_history_from_file,
    save_optimization_history_to_file,
)

# Function to create da


def run_L2_optimization(
    features: Sequence[BV_input_features],
    datasets: list[ExpD_Dataloader],
    models: list[ForwardModel],
    params: Simulation_Parameters,
    opt_settings: OptimiserSettings,
    pairwise_similarity: Array,
    output_dir: str,
) -> dict:
    """
    Runs optimization for each dataset and replicate with randomized frame weights.

    Parameters:
    -----------
    simulation : Simulation
        Simulation object
    datasets : list
        List of dataset objects
    models : list
        List of forward model objects
    opt_settings : OptimiserSettings
        Optimization settings
    n_replicates : int
        Number of replicates per dataset

    Returns:
    --------
    results : dict
        Dictionary of optimization results for each dataset and replicate
    """
    results = {}
    rng_key = jax.random.PRNGKey(0)

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

            optimiser = OptaxOptimizer(
                parameter_masks={Optimisable_Parameters.frame_weights},
                optimizer="adam",
                learning_rate=1e-4,
            )
            # Run optimization
            print(f"Running optimization for Dataset Seed {i + 1}/{len(datasets)}")
            _, opt_result = run_optimise(
                new_simulation,
                initialise=False,
                optimizer=optimiser,
                data_to_fit=(dataset, params),
                config=opt_settings,
                forward_models=models,
                indexes=[0, 0],
                loss_functions=[hdx_uptake_mean_centred_MSE_loss, maxent_L2_loss],
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
                train_data=dataset.train.data,
                val_data=dataset.val.data,
                exp_data=dataset.test.data,
                opt_history=loaded_history,
            )
            del new_simulation
            jax.clear_caches()
        except Exception as e:
            print(f"Error in Dataset Seed {i + 1} {e}")
            # Add a placeholder for failed runs to maintain indexing
            seed_results.append(None)

        results[f"seed_{i}"] = seed_results
    jax.clear_caches()
    return results


def run_quick_auto_validation(
    base_output_dir: str,
    reference_paths: list[str],
    trajectory_path: str,
    topology_path: str,
    bv_config: BV_model_Config,
    regularisation_scale: float = 0.1,
):
    print(f"Starting auto-validation with regularization scale: {regularisation_scale}")
    jax.clear_caches()
    # Create output directory for this regularization scale
    output_dir = os.path.join(base_output_dir, "quick_auto_validation_results")
    os.makedirs(output_dir, exist_ok=True)

    reference_structures = [Universe(top_path) for top_path in reference_paths]

    synthetic_data = create_synthetic_data(
        ref_structures=reference_structures,
        ref_ratios=[0.6, 0.4],
        bv_config=bv_config,
        trajectory_path=trajectory_path,
        topology_path=topology_path,
    )
    datasets, models, opt_settings, features = setup_simulation(
        bv_config,
        seeds=list(range(3)),
        trajectory_path=trajectory_path,
        topology_path=topology_path,
        synthetic_data=synthetic_data,
    )

    trajectory_length = features[0].features_shape[2]

    new_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length, dtype=jnp.float32),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.array([1.0, regularisation_scale], dtype=jnp.float32),
        forward_model_scaling=jnp.ones(2, dtype=jnp.float32),
        normalise_loss_functions=jnp.zeros(2, dtype=jnp.float32),
    )

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    universe = Universe(topology_path, trajectory_path)
    ca_atoms = universe.select_atoms("name CA")

    ca_coords_by_frame = []

    for ts in universe.trajectory:
        ca_coords_by_frame.append(pdist(ca_atoms.positions).flatten())

    ca_coords_matrix = np.vstack(ca_coords_by_frame)

    cosine_distances = squareform(pdist(ca_coords_matrix, metric="cosine"))

    similarity_matrix = 1 - cosine_distances

    pairwise_similarity = jnp.array(cosine_distances)

    # save the pairwise similarity matrix as a heatmap
    plt.imshow(pairwise_similarity, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.savefig(
        os.path.join(output_dir, "pairwise_similarity_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    opt_settings.n_steps = 5000
    opt_settings.convergence = 1e-8

    results = run_L2_optimization(
        features=features,
        datasets=datasets,
        models=models,
        params=new_params,
        opt_settings=opt_settings,
        output_dir=output_dir,
        pairwise_similarity=pairwise_similarity,
    )
    print(
        f"Auto-validation completed successfully for regularization scale: {regularisation_scale}"
    )
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    print("Starting main script...")  # Track progress
    open_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
    closed_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"

    topology_path = open_path

    trajectory_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"

    # List of regularization scales to test
    regularisation_scales = [0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # regularisation_scales = [0.1, 1.0]
    regularisation_scales = [1e0, 1e1, 1e-1, 1e3, 0, 1e2]
    regularisation_scales = [1e-2, 1e-3, 1e-4]
    regularisation_scales = [1e-2, 1e-3, 1e-4]
    regularisation_scales = [1e4, 1e5]

    # Configure the BV model
    bv_config = BV_model_Config(num_timepoints=5)
    bv_config.timepoints = jnp.array([0.167, 1, 10, 60, 120])

    # Iterate through all regularization scales
    for regularisation_scale in regularisation_scales:
        print(f"\n===== Processing regularization scale: {regularisation_scale} =====\n")

        # Create a unique output directory for each regularization scale
        base_output_dir = os.path.join(
            os.path.dirname(__file__),
            "mcMaxENTL2_test",
            f"TeaA_simple_{regularisation_scale}_adam",
        )

        # Create the output directory if it doesn't exist
        os.makedirs(base_output_dir, exist_ok=True)

        # Run the auto-validation for this regularization scale
        run_quick_auto_validation(
            base_output_dir=base_output_dir,
            reference_paths=[open_path, closed_path],
            trajectory_path=trajectory_path,
            topology_path=topology_path,
            bv_config=bv_config,
            regularisation_scale=regularisation_scale,
        )

        # Clear any remaining memory to prevent issues between runs
        jax.clear_caches()

    print("Auto-validation script completed for all regularization scales.")
