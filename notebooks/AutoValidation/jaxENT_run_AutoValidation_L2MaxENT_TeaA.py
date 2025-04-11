"""
This script runs auto-validation (simulation) on the L2-only data using jaxENT. Interpretation and Analysis of the results is done in a separate script.

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
from jaxENT_run_AutoValidation_L2FWConsistency_TeaA import (
    create_synthetic_data,
)
from MDAnalysis import Universe

from jaxent.analysis.plots.optimisation import (
    plot_frame_weights_heatmap,
    plot_loss_components,
    plot_split_visualization,
    plot_total_losses,
)
from jaxent.data.loader import Dataset, ExpD_Dataloader, ExpD_Datapoint
from jaxent.data.splitting.sparse_map import create_sparse_map
from jaxent.data.splitting.split import DataSplitter
from jaxent.featurise import run_featurise
from jaxent.interfaces.builder import Experiment_Builder
from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.interfaces.topology import Partial_Topology
from jaxent.models.core import Simulation
from jaxent.models.HDX.BV.features import BV_input_features
from jaxent.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.opt.losses import hdx_uptake_l2_loss, maxent_convexKL_loss
from jaxent.opt.optimiser import OptaxOptimizer, Optimisable_Parameters, OptimizationHistory
from jaxent.opt.run import run_optimise
from jaxent.types.base import ForwardModel
from jaxent.types.config import FeaturiserSettings, OptimiserSettings
from jaxent.utils.hdf import (
    load_optimization_history_from_file,
    save_optimization_history_to_file,
)


# Function to create datasets with mappings for each seed
def create_datasets_with_mappings(
    trajs: list[Universe],
    features: Sequence[BV_input_features],
    feature_topology: list[Partial_Topology],
    exp_data: Sequence[ExpD_Datapoint],
    seeds: list[int],
    train_val_split: float,
) -> list[ExpD_Dataloader]:
    """
    Creates datasets with train/val splits for multiple seeds.

    Parameters:
    -----------
    ensemble : list
        List of universe objects
    features : list
        List of feature objects
    feature_topology : list
        List of feature topology objects
    exp_data : list
        List of experimental data
    seeds : list
        List of random seeds

    Returns:
    --------
    datasets : list
        List of dataset objects for each seed
    """
    datasets = []

    # assert seeds are unique
    assert len(seeds) == len(set(seeds)), "Seeds must be unique"

    for seed in seeds:
        print(f"Creating dataset for seed: {seed}")  # Track progress
        # Create a dataset object
        dataset = ExpD_Dataloader(data=exp_data)

        # Create a splitter with the current seed
        splitter = DataSplitter(
            dataset,
            random_seed=seed,
            ensemble=trajs,
            common_residues=set(feature_topology),
            peptide=False,
            train_size=train_val_split,
            centrality=False,
        )

        # Generate train/val split
        train_data, val_data = splitter.random_split(remove_overlap=True)
        print(f"Generated train/val split for seed: {seed}")  # Track progress

        # Create sparse maps
        train_sparse_map = create_sparse_map(features[0], feature_topology, train_data)
        val_sparse_map = create_sparse_map(features[0], feature_topology, val_data)
        test_sparse_map = create_sparse_map(features[0], feature_topology, exp_data)
        print(f"Created sparse maps for seed: {seed}")  # Track progress

        # Add train, val, and test sets to the dataset
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

        datasets.append(dataset)
        print(f"Dataset created and appended for seed: {seed}")  # Track progress

    return datasets


def visualize_optimization_results(
    output_suff: str,
    train_data: list[ExpD_Datapoint],
    val_data: list[ExpD_Datapoint],
    exp_data: list[ExpD_Datapoint],
    opt_history: OptimizationHistory,
):
    """
    Create and display all visualization plots.
    outptu_suff is the suffix for the output directory - simply add the name of the plot+.png to save it
    """

    # Create all plots
    split_fig = plot_split_visualization(train_data, val_data, exp_data)
    total_loss_fig = plot_total_losses(opt_history)
    loss_fig = plot_loss_components(opt_history)
    weights_fig = plot_frame_weights_heatmap(opt_history)

    # save all plots
    split_fig.savefig(output_suff + "_split.png")
    total_loss_fig.savefig(output_suff + "_total_loss.png")
    loss_fig.savefig(output_suff + "_loss_components.png")
    weights_fig.savefig(output_suff + "_frame_weights.png")
    plt.close("all")
    print(f"Visualizations saved with prefix: {output_suff}")  # Track progress


# def create_synthetic_data(
#     ref_structures: list[Universe],
#     ref_ratios: list[float],
#     bv_config=BV_model_Config(num_timepoints=3),
# ) -> Sequence[ExpD_Datapoint]:
#     """
#     Create synthetic data based on reference structures and their ratios.

#     Featurises each reference structure, averages across frames.
#     """
#     print("Creating synthetic data...")  # Track progress

#     features = []

#     featuriser_settings = FeaturiserSettings(name="synthetic", batch_size=None)
#     models = [BV_model(bv_config)]
#     ensemble = Experiment_Builder(ref_structures, models)
#     features, feature_topology = run_featurise(ensemble, featuriser_settings)

#     print(f"Features shape: {features[0].features_shape}")

#     # check that the length trajectoy is the same as the number of reference ratios
#     assert len(ref_ratios) == features[0].features_shape[2], (
#         f"Length of ref_ratios: {len(ref_ratios)} and features: {features[0].features_shape[2]} do not match"
#     )

#     # Create synthetic data by applying forward pass using the reference ratios as the frame weights
#     params = Simulation_Parameters(
#         frame_weights=jnp.array(ref_ratios).reshape(1, -1, 1),
#         frame_mask=jnp.ones_like(jnp.array(ref_ratios)).reshape(1, -1, 1),
#         model_parameters=[bv_config.forward_parameters],
#         forward_model_weights=jnp.ones(1, dtype=jnp.float32),
#         forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
#         normalise_loss_functions=jnp.zeros(1, dtype=jnp.float32),
#     )
#     print(params)

#     simulation = Simulation(forward_models=models, input_features=features, params=params)
#     simulation.initialise()
#     simulation.forward(params)
#     output = simulation.outputs[0]
#     print(f"Output shape: {output.uptake.shape}")
#     # Create synthetic data
#     synthetic_data = [
#         HDX_peptide(dfrac=output.uptake[0, i], top=feature_topology[0][i])
#         for i in range(output.uptake.shape[1])
#     ]
#     del simulation
#     jax.clear_caches()
#     print("Synthetic data created.")  # Track progress
#     return synthetic_data


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
                learning_rate=5e-4,
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
                loss_functions=[hdx_uptake_l2_loss, maxent_convexKL_loss],
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
            seed_results.append(None)

        results[f"seed_{i}"] = seed_results
    jax.clear_caches()
    return results


def setup_simulation(
    bv_config: BV_model_Config,
    seeds: list[int],
    trajectory_path: str,
    topology_path: str,
    synthetic_data: Sequence[ExpD_Datapoint],
    train_val_split: float = 0.6,
) -> tuple[
    list[ExpD_Dataloader],
    Sequence[ForwardModel],
    OptimiserSettings,
    # ExpD_Dataloader,
    Sequence[BV_input_features],
]:
    print("Setting up simulation...")  # Track progress
    opt_settings = OptimiserSettings(name="test", n_steps=5000, convergence=1e-5)

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    # check if trajectory dir has .jpz files
    traj_dir = os.path.dirname(trajectory_path)
    if not any(file.endswith(".jpz.npz") for file in os.listdir(traj_dir)):
        features, feature_topology = run_featurise(ensemble, featuriser_settings)

        # save heavy, oxygen contacts and kints using jax savez
        features = features[0]
        print(f"Features shape: {features.features_shape}")

        jnp.savez(
            os.path.join(traj_dir, "features.jpz"),
            heavy_contacts=features.heavy_contacts,
            acceptor_contacts=features.acceptor_contacts,
            k_ints=features.k_ints,
        )

    else:
        #
        feature_topology = [[data.top for data in synthetic_data]]
        # Load the features from the .jpz file
        features_path = os.path.join(traj_dir, "features.jpz.npz")
        features = [BV_input_features(**jnp.load(features_path))]
        print(f"Loaded features shape: {features[0].features_shape}")

    datasets = create_datasets_with_mappings(
        universes,
        features,
        feature_topology[0],
        synthetic_data,
        seeds=seeds,
        train_val_split=train_val_split,
    )

    jax.clear_caches()
    print("Simulation setup complete.")  # Track progress

    return datasets, models, opt_settings, features


def setup_simulation(
    bv_config: BV_model_Config,
    seeds: list[int],
    trajectory_path: str,
    topology_path: str,
    synthetic_data: Sequence[ExpD_Datapoint],
    train_val_split: float = 0.6,
) -> tuple[
    list[ExpD_Dataloader],
    Sequence[ForwardModel],
    OptimiserSettings,
    # ExpD_Dataloader,
    Sequence[BV_input_features],
]:
    print("Setting up simulation...")  # Track progress
    opt_settings = OptimiserSettings(name="test", n_steps=500, convergence=1e-8)

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    # check if trajectory dir has .jpz files
    traj_dir = os.path.dirname(trajectory_path)
    if not any(file.endswith(".jpz.npz") for file in os.listdir(traj_dir)):
        features, feature_topology = run_featurise(ensemble, featuriser_settings)

        # save heavy, oxygen contacts and kints using jax savez
        features = features[0]
        print(f"Features shape: {features.features_shape}")

        jnp.savez(
            os.path.join(traj_dir, "features.jpz"),
            heavy_contacts=features.heavy_contacts,
            acceptor_contacts=features.acceptor_contacts,
            k_ints=features.k_ints,
        )

    else:
        #
        feature_topology = [[data.top for data in synthetic_data]]
        # Load the features from the .jpz file
        features_path = os.path.join(traj_dir, "features.jpz.npz")
        features = [BV_input_features(**jnp.load(features_path))]
        print(f"Loaded features shape: {features[0].features_shape}")

    datasets = create_datasets_with_mappings(
        universes,
        features,
        feature_topology[0],
        synthetic_data,
        seeds=seeds,
        train_val_split=train_val_split,
    )

    jax.clear_caches()
    print("Simulation setup complete.")  # Track progress

    return datasets, models, opt_settings, features


def run_quick_auto_validation(
    base_output_dir: str,
    reference_paths: list[str],
    trajectory_path: str,
    topology_path: str,
    bv_config: BV_model_Config,
    regularisation_scale: float = 0.1,
):
    print("Starting quick auto-validation...")  # Track progress
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
        normalise_loss_functions=jnp.ones(2, dtype=jnp.float32),
    )
    # new_params = Simulation_Parameters(
    #     frame_weights=jnp.ones(trajectory_length) / trajectory_length,
    #     frame_mask=jnp.ones(trajectory_length, dtype=jnp.float32),
    #     model_parameters=[bv_config.forward_parameters],
    #     forward_model_weights=jnp.array([1000.0], dtype=jnp.float32),
    #     forward_model_scaling=jnp.ones(1, dtype=jnp.float32),
    #     normalise_loss_functions=jnp.zeros(1, dtype=jnp.float32),
    # )

    # Run the simulation for each dataset
    output_dir = os.path.join(base_output_dir, "quick_auto_validation_results")
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
    # add colorbar
    plt.colorbar()
    plt.savefig(
        os.path.join(output_dir, "pairwise_similarity_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    results = run_L2_optimization(
        features=features,
        datasets=datasets,
        models=models,
        params=new_params,
        opt_settings=opt_settings,
        output_dir=output_dir,
        pairwise_similarity=pairwise_similarity,
    )
    print("Auto-validation completed successfully.")
    print(f"Saving results to {output_dir}")


if __name__ == "__main__":
    print("Starting main script...")  # Track progress
    open_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb"
    closed_path = "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"

    topology_path = open_path

    trajectory_path = (
        "/Users/alexi/JAX-ENT/notebooks/AutoValidation/_TeaA/trajectories/TeaA_filtered.xtc"
    )

    regularisation_scale = [0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0][6]
    # regularisation_scale = 10.0
    base_output_dir = os.path.join(
        os.path.dirname(__file__),
        "normalisation_MaxENT",
        f"TeaA_simple_{regularisation_scale}_adam",
    )

    # Create the output directory if it doesn't exist
    # os.system(f"rm -rf {base_output_dir}")
    os.system(f"mkdir -p {base_output_dir}")

    bv_config = BV_model_Config(num_timepoints=4)
    bv_config.timepoints = jnp.array([0.1, 1.0, 10.0, 100.0])
    run_quick_auto_validation(
        base_output_dir=base_output_dir,
        reference_paths=[open_path, closed_path],
        trajectory_path=trajectory_path,
        topology_path=topology_path,
        bv_config=bv_config,
        regularisation_scale=regularisation_scale,
    )
    print("Auto-validation script completed.")
