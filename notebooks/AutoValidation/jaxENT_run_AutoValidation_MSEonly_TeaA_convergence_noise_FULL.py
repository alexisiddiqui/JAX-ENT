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

"""
This script runs auto-validation (simulation) on the L2-only data using jaxent.src. Interpretation and Analysis of the results is done in a separate script.

To run this script, you require the following:
- input ensemble that contains at least 2 states (with names)
- reference states to generate synthetic data (desired ratios - default 60:40)

This script essentially creates synthetic data using the reference states and their ratios.
We then attempt to fit the ensemble to the synthetic data using jaxENT over a number of datasplits (seeds).
By interpreting the Probability Density of the RMSD of the optimised and unoptimised states we can demonstrate that jaxENT is able to recover the data (seperate script)

We repeat this over a range of regularization scales to analyze their impact.
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from scipy.spatial.distance import pdist, squareform

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
from typing import Sequence

import jax
import matplotlib.pyplot as plt
from icecream import ic

# from jaxENT_run_AutoValidation_L2FWConsistency_TeaA import (
#     create_synthetic_data,
# )
from MDAnalysis import Universe

from jaxent.src.analysis.plots.optimisation import (
    plot_frame_weights_heatmap,
    plot_loss_components,
    plot_split_visualization,
    plot_total_losses,
)
from jaxent.src.data.loader import Dataset, ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.src.opt.losses import frame_weight_consistency_loss, hdx_uptake_MSE_loss
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters, OptimizationHistory
from jaxent.src.opt.run import run_optimise
from jaxent.src.types.base import ForwardModel
from jaxent.src.types.config import FeaturiserSettings, OptimiserSettings
from jaxent.src.types.HDX import HDX_peptide
from jaxent.src.utils.hdf import (
    load_optimization_history_from_file,
    save_optimization_history_to_file,
)

# # Function to create datasets with mappings for each seed
# def create_datasets_with_mappings(
#     trajs: list[Universe],
#     features: Sequence[BV_input_features],
#     feature_topology: list[Partial_Topology],
#     exp_data: Sequence[ExpD_Datapoint],
#     seeds: list[int],
#     train_val_split: float,
# ) -> list[ExpD_Dataloader]:
#     """
#     Creates datasets with train/val splits for multiple seeds.

#     Parameters:
#     -----------
#     ensemble : list
#         List of universe objects
#     features : list
#         List of feature objects
#     feature_topology : list
#         List of feature topology objects
#     exp_data : list
#         List of experimental data
#     seeds : list
#         List of random seeds

#     Returns:
#     --------
#     datasets : list
#         List of dataset objects for each seed
#     """
#     datasets = []

#     # assert seeds are unique
#     assert len(seeds) == len(set(seeds)), "Seeds must be unique"

#     for seed in seeds:
#         print(f"Creating dataset for seed: {seed}")  # Track progress
#         # Create a dataset object
#         dataset = ExpD_Dataloader(data=exp_data)

#         # Generate train/val split by splitting
#         np.random.seed(seed)
#         indices = np.random.permutation(np.arange(len(exp_data)))
#         train_indices = indices[: int(len(exp_data) * train_val_split)]
#         val_indices = indices[int(len(exp_data) * train_val_split) :]
#         print(f"Train indices: {indices}")  # Track progress
#         print(len(train_indices))
#         print(len(val_indices))
#         # print overlap between train and val indices
#         print(np.intersect1d(train_indices, val_indices))

#         print(len(exp_data))

#         train_data = [exp_data[i] for i in train_indices]
#         val_data = [exp_data[i] for i in val_indices]

#         # Create sparse maps
#         train_sparse_map = create_sparse_map(features[0], feature_topology, train_data)
#         val_sparse_map = create_sparse_map(features[0], feature_topology, val_data)
#         test_sparse_map = create_sparse_map(features[0], feature_topology, exp_data)
#         print(f"Created sparse maps for seed: {seed}")  # Track progress

#         # Add train, val, and test sets to the dataset
#         dataset.train = Dataset(
#             data=train_data,
#             y_true=jnp.array([data.extract_features() for data in train_data]),
#             residue_feature_ouput_mapping=train_sparse_map,
#         )

#         dataset.val = Dataset(
#             data=val_data,
#             y_true=jnp.array([data.extract_features() for data in val_data]),
#             residue_feature_ouput_mapping=val_sparse_map,
#         )

#         dataset.test = Dataset(
#             data=exp_data,
#             y_true=jnp.array([data.extract_features() for data in exp_data]),
#             residue_feature_ouput_mapping=test_sparse_map,
#         )

#         datasets.append(dataset)
#         print(f"Dataset created and appended for seed: {seed}")  # Track progress

#     return datasets


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
        print(len(exp_data))

        dataset = ExpD_Dataloader(data=exp_data)

        # # Generate train/val split by splitting
        # np.random.seed(seed)
        # indices = np.random.permutation(np.arange(len(exp_data)))
        # train_indices = indices[: int(len(exp_data) * train_val_split)]
        # val_indices = indices[int(len(exp_data) * train_val_split) :]
        # print(f"Train indices: {indices}")  # Track progress
        # print(len(val_indices))
        # print(len(train_indices))
        # # print overlap between train and val indices
        # print(np.intersect1d(train_indices, val_indices))

        # print(len(exp_data))

        # # pick not train indices for validation

        # train_data = [exp_data[i] for i in train_indices]
        # val_data = [exp_data[i] for i in val_indices]

        # Create sparse maps
        train_sparse_map = create_sparse_map(features[0], feature_topology, exp_data)
        val_sparse_map = create_sparse_map(features[0], feature_topology, exp_data)
        test_sparse_map = create_sparse_map(features[0], feature_topology, exp_data)
        print(f"Created sparse maps for seed: {seed}")  # Track progress

        # Add train, val, and test sets to the dataset
        dataset.train = Dataset(
            data=exp_data,
            y_true=jnp.array([data.extract_features() for data in exp_data]),
            residue_feature_ouput_mapping=train_sparse_map,
        )

        dataset.val = Dataset(
            data=exp_data,
            y_true=jnp.array([data.extract_features() for data in exp_data]),
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


def extract_hdx_data_from_file(file_path):
    # Read the file using pandas with fixed column positions
    df = pd.read_csv(
        file_path,
        delim_whitespace=True,
        comment="#",
        header=None,
    )

    # Assuming the correct structure based on your example
    # Column 1 is residue_end, columns 2-6 are time points
    result_dict = {}

    for _, row in df.iterrows():
        residue_end = int(row[1])  # Second column is residue_end
        times = row[2:7].tolist()  # Columns 2-6 are time values
        result_dict[residue_end] = times

    return result_dict


def create_synthetic_data(
    ref_structures: list[Universe],
    ref_ratios: list[float],
    trajectory_path: str,
    topology_path: str,
    output_dir: str = None,
    bv_config=BV_model_Config(num_timepoints=3),
    noise_sd: float = 0.2,
    random_seed: int = 42,
) -> Sequence[ExpD_Datapoint]:
    """
    Create synthetic data based on reference structures and their ratios,
    using the entire ensemble and weighting by state clusters.

    Checks if features and topology already exist before computing them.

    Parameters:
    -----------
    ref_structures : list
        List of reference structure Universes
    ref_ratios : list
        List of ratios for each state [open_ratio, closed_ratio]
    trajectory_path : str
        Path to trajectory file
    topology_path : str
        Path to topology file
    output_dir : str, optional
        Directory to save/load cached data
    bv_config : BV_model_Config
        Configuration for BV model

    Returns:
    --------
    synthetic_data : list
        List of synthetic HDX peptide datapoints
    """
    print("Creating synthetic data with ensemble clustering...")

    artificial_HDX_data = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/_Bradshaw/Reproducibility_pack_v2/data/artificial_HDX_data/mixed_60-40_artificial_expt_resfracs.dat"

    # load the artificial HDX data
    output = extract_hdx_data_from_file(artificial_HDX_data)
    print(output)

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

        from jaxent.src.interfaces.topology import Partial_Topology

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
        featuriser_settings = FeaturiserSettings(name="synthetic", batch_size=None)
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
                "residue_end": int(topology.residue_end),
                "peptide_trim": int(topology.peptide_trim),
                "fragment_index": int(topology.fragment_index)
                if topology.fragment_index is not None
                else None,
                "length": int(topology.length),
            }

        topology_dicts = [topology_to_dict(top) for top in feature_topology[0]]

        print(f"Saving topology to {topology_path_json}")
        with open(topology_path_json, "w") as f:
            json.dump(topology_dicts, f, indent=2)

    print(feature_topology[0])
    print(type(feature_topology[0][0].residue_end))
    print(list(output.keys()))
    print([top.residue_end for top in feature_topology[0]])

    output = dict(sorted(output.items()))
    print(output.values())
    _dfrac = list(output.values())
    _res = list(output.keys())

    dfrac_topology = [top for top in feature_topology[0] if top.residue_end in _res]

    print(len(dfrac_topology))

    dfrac_topology = sorted(dfrac_topology, key=lambda x: x.residue_end)
    # sort outout dictionary by key
    print(_dfrac)
    rng = np.random.RandomState(random_seed)

    # # Create synthetic data
    synthetic_data = []
    for i in range(len(dfrac_topology)):
        base_dfrac = np.ones_like(_dfrac[i])

        # Scale noise with the magnitude of the uptake
        scaled_noise_sd = noise_sd * (base_dfrac)

        # Generate unique noise for each peptide and each timepoint
        noise = rng.normal(0, scaled_noise_sd, size=base_dfrac.shape)
        noisy_dfrac = base_dfrac + noise

        # Ensure values stay within physical bounds (0 to 1 for deuterium fractions)
        # noisy_dfrac = np.clip(noisy_dfrac, 0.0, 1.0)

        # Create the HDX peptide with noisy data
        synthetic_data.append(HDX_peptide(dfrac=noisy_dfrac, top=dfrac_topology[i]))

    assert len(synthetic_data) == len(dfrac_topology), "Synthetic data length mismatch"

    # jax.clear_caches()
    print(f"Synthetic data created. Length: {len(synthetic_data)}")
    return synthetic_data


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


def run_L2_optimization(
    features: Sequence[BV_input_features],
    datasets: list[ExpD_Dataloader],
    models: list[ForwardModel],
    params: Simulation_Parameters,
    opt_settings: OptimiserSettings,
    pairwise_similarity: Array,
    output_dir: str,
    convergence: float = 1e-5,
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
            opt_settings.convergence = convergence
            # Run optimization
            print(f"Running optimization for Dataset Seed {i + 1}/{len(datasets)}")
            _, opt_result = run_optimise(
                new_simulation,
                initialise=False,
                optimizer=optimiser,
                data_to_fit=(dataset, pairwise_similarity),
                config=opt_settings,
                forward_models=models,
                indexes=[0, 0],
                loss_functions=[hdx_uptake_MSE_loss, frame_weight_consistency_loss],
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
            # del params
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
    train_val_split: float = 0.5,
) -> tuple[
    list[ExpD_Dataloader],
    Sequence[ForwardModel],
    OptimiserSettings,
    # ExpD_Dataloader,
    Sequence[BV_input_features],
]:
    print("Setting up simulation...")  # Track progress
    opt_settings = OptimiserSettings(name="test", n_steps=500, convergence=1e-3)

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)
    ic.enable()
    # check if trajectory dir has .jpz files
    traj_dir = os.path.dirname(trajectory_path)
    k_ints_path = "/home/alexi/Documents/JAX-ENT/notebooks/AutoValidation/HDXer/TeaA_auto_VAL_RW_bench_r_naive_random/train_TeaA_auto_VAL_1/out__train_TeaA_auto_VAL_1Intrinsic_rates.dat"
    # open as csv - first line is header
    with open(k_ints_path, "r") as f:
        k_ints_text = [line.strip() for line in f.readlines()[1:]]
        k_ints = [line.split() for line in k_ints_text]
    # create dictionary

    k_int_dict = {int(k_int[0]): float(k_int[1]) for k_int in k_ints}

    print(len(k_int_dict))

    # filter out the residues that are not in the features
    traj_dir = os.path.dirname(trajectory_path)
    # now insert the intrinsic rates into the features by creating a new features object
    topology_path_json = os.path.join(traj_dir, "topology.json")

    if not any(file.endswith(".jpz.npz") for file in os.listdir(traj_dir)):
        features, feature_topology = run_featurise(ensemble, featuriser_settings)

        features[0] = BV_input_features(
            heavy_contacts=features[0].heavy_contacts,
            acceptor_contacts=features[0].acceptor_contacts,
            k_ints=filtered_k_ints,
        )
        # save heavy, oxygen contacts and kints using jax savez
        features = features[0]
        print(f"Features shape: {features.features_shape}")

        feat_res = [feat_top.residue_end for feat_top in feature_topology[0]]

        filtered_k_int_dict = {res: k_int_dict[res] for res in k_int_dict if res in feat_res}
        print(len(filtered_k_int_dict))
        filtered_k_ints = jnp.array([filtered_k_int_dict[res] for res in feat_res])
        print(filtered_k_ints)

        jnp.savez(
            os.path.join(traj_dir, "features.jpz"),
            heavy_contacts=features.heavy_contacts,
            acceptor_contacts=features.acceptor_contacts,
            k_ints=filtered_k_ints,
        )

        # Save topology
        def topology_to_dict(topology):
            return {
                "chain": topology.chain,
                "fragment_sequence": topology.fragment_sequence,
                "residue_start": int(topology.residue_start),
                "residue_end": int(topology.residue_end),
                "peptide_trim": int(topology.peptide_trim),
                "fragment_index": int(topology.fragment_index)
                if topology.fragment_index is not None
                else None,
                "length": int(topology.length),
            }

        topology_dicts = [topology_to_dict(top) for top in feature_topology[0]]

        print(f"Saving topology to {topology_path_json}")
        with open(topology_path_json, "w") as f:
            json.dump(topology_dicts, f, indent=2)

    else:
        # Load topology
        with open(topology_path_json, "r") as f:
            topology_dicts = json.load(f)

        from jaxent.src.interfaces.topology import Partial_Topology

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
        #
        # print(feature_topology[0])
        feature_topology = [[top for top in feature_topology[0]]]
        # Load the features from the .jpz file
        features_path = os.path.join(traj_dir, "features.jpz.npz")
        features = [BV_input_features(**jnp.load(features_path))]
        print(features)
        feat_res = [feat_top.residue_end for feat_top in feature_topology[0]]
        print(feat_res)

        # breakpoint()
        filtered_k_int_dict = {res: k_int_dict[res] for res in k_int_dict if res in feat_res}

        print(len(filtered_k_int_dict))
        filtered_k_ints = jnp.array([filtered_k_int_dict[res] for res in feat_res])
        # print(filtered_k_ints)

        features[0] = BV_input_features(
            heavy_contacts=features[0].heavy_contacts,
            acceptor_contacts=features[0].acceptor_contacts,
            k_ints=filtered_k_ints,
        )
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
    noise_sd: float = 0.2,
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
        noise_sd=noise_sd,
        random_seed=42,
    )
    datasets, models, opt_settings, features = setup_simulation(
        bv_config,
        seeds=list(range(1)),
        trajectory_path=trajectory_path,
        topology_path=topology_path,
        synthetic_data=synthetic_data,
    )

    trajectory_length = features[0].features_shape[2]

    new_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length, dtype=jnp.float32),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.array([1.0, 0.0], dtype=jnp.float32),
        forward_model_scaling=jnp.ones(2, dtype=jnp.float32),
        normalise_loss_functions=jnp.ones(2, dtype=jnp.float32),
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
    convergence = regularisation_scale
    opt_settings.n_steps = 5000

    results = run_L2_optimization(
        features=features,
        datasets=datasets,
        models=models,
        params=new_params,
        opt_settings=opt_settings,
        output_dir=output_dir,
        pairwise_similarity=pairwise_similarity,
        convergence=convergence,
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
    regularisation_scales = [1, 10, 100.0]
    regularisation_scales = [1e1, 1e2, 1e3]

    convergence_rates = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    convergence_rates = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    regularisation_scales = convergence_rates

    # Configure the BV model
    bv_config = BV_model_Config(num_timepoints=5)
    bv_config.timepoints = jnp.array([0.167, 1, 10, 60, 120])
    noise_sds = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    for noise_sd in noise_sds:
        # Iterate through all regularization scales
        for regularisation_scale in regularisation_scales:
            print(f"\n===== Processing regularization scale: {regularisation_scale} =====\n")

            # Create a unique output directory for each regularization scale
            base_output_dir = os.path.join(
                os.path.dirname(__file__),
                f"allPEPTIDE_convergence_equalnoise{noise_sd}_l2long_adam_MSE",
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
                noise_sd=noise_sd,
            )

            # Clear any remaining memory to prevent issues between runs
            jax.clear_caches()

        print("Auto-validation script completed for all regularization scales.")
