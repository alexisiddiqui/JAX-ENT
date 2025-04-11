# This script is going to perform a cross validation to compare to HDXer
# Going to consider simple regularisation of MaxEnt plus an additional prior term
# Prior terms:

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
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
from jaxent.models.func.common import find_common_residues
from jaxent.models.HDX.BV.features import BV_input_features
from jaxent.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.models.HDX.BV.parameters import BV_Model_Parameters
from jaxent.opt.base import JaxEnt_Loss
from jaxent.opt.losses import (
    hdx_uptake_l1_loss,
    hdx_uptake_l2_loss,
    hdx_uptake_mean_centred_l1_loss,
    hdx_uptake_mean_centred_l2_loss,
    maxent_convexKL_loss,
)
from jaxent.opt.optimiser import OptaxOptimizer, Optimisable_Parameters, OptimizationHistory
from jaxent.opt.run import run_optimise
from jaxent.types.base import ForwardModel
from jaxent.types.config import FeaturiserSettings, OptimiserSettings
from jaxent.types.HDX import HDX_peptide
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
    ic.enable()

    for seed in seeds:
        # Create a dataset object
        dataset = ExpD_Dataloader(data=exp_data)

        # Create a splitter with the current seed
        splitter = DataSplitter(
            dataset,
            random_seed=seed,
            ensemble=trajs,
            common_residues=set(feature_topology),
            peptide=False,
        )

        # Generate train/val split
        train_data, val_data = splitter.random_split(remove_overlap=False)

        # Create sparse maps
        train_sparse_map = create_sparse_map(features[0], feature_topology, train_data)
        val_sparse_map = create_sparse_map(features[0], feature_topology, val_data)
        test_sparse_map = create_sparse_map(features[0], feature_topology, exp_data)

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


def setup_simulation(
    seeds: list[int],
) -> tuple[
    list[ExpD_Dataloader],
    Sequence[ForwardModel],
    OptimiserSettings,
    ExpD_Dataloader,
    Sequence[BV_input_features],
    Simulation_Parameters,
]:
    bv_config = BV_model_Config(num_timepoints=3)

    opt_settings = OptimiserSettings(name="test", n_steps=500, learning_rate=1e-3, convergence=1e-8)

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = (
        "/Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb"
    )

    trajectory_path = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/MoPrP_max_plddt_4334_clusters500_20250410-150802/clusters/all_clusters.xtc"
    segs_data = (
        "/Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/MoPrP_segments.txt"
    )
    dfrac_data = (
        "/Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/MoPrP_dfrac.dat"
    )
    k_ints_path = "/Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/out__train_MoPrP_af_clean_1Intrinsic_rates.dat"

    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    print(len(features[0].heavy_contacts))
    print(len(features[0].k_ints))

    with open(segs_data, "r") as f:
        segs_text = [line.strip() for line in f.readlines()]
        segs = [line.split() for line in segs_text]

    segs = [[start, end] for start, end in segs]

    residues = [int(seg[1]) for seg in segs]
    print("residues", residues)

    with open(dfrac_data, "r") as f:
        # skip first line and then read in vals
        dfrac_text = [line.strip() for line in f.readlines()[1:]]
        dfracs = [line.split() for line in dfrac_text]

    dfracs = [np.array(line, dtype=float) for line in dfracs]

    dfrac_dict = {res: dfrac for res, dfrac in zip(residues, dfracs)}

    assert len(segs) == len(dfracs), "Lengths of segs and dfracs don't match"

    print(segs)
    print(dfracs)
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    print([(top.residue_start, top.residue_end) for top in top_segments])

    exp_top_segs = []
    exp_dfracs = []

    print(residues)
    print([top.residue_start for top in top_segments])
    for top_seg in top_segments:
        residue_start = top_seg.residue_start

        if residue_start in residues:
            exp_top_segs.append(top_seg)
            exp_dfracs.append(dfrac_dict[residue_start])

    # print(exp_top_segs)

    assert len(exp_top_segs) == len(segs), (
        f"Lengths of top segs: {len(exp_top_segs)} and exp segs don't match: {len(segs)}"
    )

    for i, top_seg in enumerate(exp_top_segs):
        exp_top_segs[i].fragment_index = i

    print([exp_top.fragment_index for exp_top in exp_top_segs])
    print([exp_top.residue_start for exp_top in exp_top_segs])

    print([feat_top.fragment_index for feat_top in feature_topology[0]])

    # Create features
    exp_data = [
        HDX_peptide(dfrac=_dfrac, top=top) for _dfrac, top in zip(dfracs, feature_topology[0])
    ]

    # Create simulation
    trajectory_length = features[0].features_shape[2]
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length, dtype=jnp.float32) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length, dtype=jnp.float32),
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.array([1.0, 1.0, 0.1], dtype=jnp.float32),
        forward_model_scaling=jnp.array([1.0, 0.1, 1.0], dtype=jnp.float32),
        normalise_loss_functions=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)
    simulation.initialise()

    simulation.forward(params)

    output = simulation.outputs[0]
    print(output.uptake.shape)

    # open as csv - first line is header
    with open(k_ints_path, "r") as f:
        k_ints_text = [line.strip() for line in f.readlines()[1:]]
        k_ints = [line.split() for line in k_ints_text]
    # create dictionary

    k_int_dict = {int(k_int[0]): float(k_int[1]) for k_int in k_ints}

    print(len(k_int_dict))

    # filter out the residues that are not in the features
    feat_res = [feat_top.residue_end for feat_top in feature_topology[0]]

    filtered_k_int_dict = {res: k_int_dict[res] for res in k_int_dict if res in feat_res}

    print(len(filtered_k_int_dict))
    filtered_k_ints = jnp.array([filtered_k_int_dict[res] for res in feat_res])
    print(filtered_k_ints)
    # now insert the intrinsic rates into the features by creating a new features object

    features[0] = BV_input_features(
        heavy_contacts=features[0].heavy_contacts,
        acceptor_contacts=features[0].acceptor_contacts,
        k_ints=filtered_k_ints,
    )

    datasets = create_datasets_with_mappings(
        universes, features, feature_topology[0], exp_data, seeds=seeds
    )

    exp_residues = [int(seg[1]) for seg in segs]

    exp_topology = [top for top in feature_topology[0] if top.residue_end in exp_residues]

    prior_pfs = output.uptake.T

    # need to reshape to a list of length 53 and Arrays of shape 3
    prior_data = [
        HDX_peptide(dfrac=_prior_df, top=top) for _prior_df, top in zip(prior_pfs, exp_topology)
    ]

    pf_prior_data = ExpD_Dataloader(data=prior_data)

    # prior_splitter = DataSplitter(
    #     pf_prior_data,
    #     random_seed=42,
    #     ensemble=universes,
    #     common_residues=set(feature_topology[0]),
    #     peptide=False,
    # )
    # prior_train_data, prior_val_data = prior_splitter.random_split()

    # prior_train_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_train_data)

    # prior_val_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_val_data)

    prior_test_sparse_map = create_sparse_map(features[0], feature_topology[0], pf_prior_data.data)

    pf_prior_data.train = Dataset(
        data=pf_prior_data.data,
        y_true=jnp.array([data.extract_features() for data in pf_prior_data.data]),
        residue_feature_ouput_mapping=prior_test_sparse_map,
    )

    pf_prior_data.val = Dataset(
        data=pf_prior_data.data,
        y_true=jnp.array([data.extract_features() for data in pf_prior_data.data]),
        residue_feature_ouput_mapping=prior_test_sparse_map,
    )

    pf_prior_data.test = Dataset(
        data=pf_prior_data.data,
        y_true=jnp.array([data.extract_features() for data in pf_prior_data.data]),
        residue_feature_ouput_mapping=prior_test_sparse_map,
    )
    del simulation
    jax.clear_caches()

    return datasets, models, opt_settings, pf_prior_data, features, params


def run_MAE_max_ent_optimization_replicates(
    simulation: Simulation,
    datasets: list[ExpD_Dataloader],
    models: list[ForwardModel],
    opt_settings: OptimiserSettings,
    pf_prior_data: ExpD_Dataloader,
    regularization_fn: JaxEnt_Loss,
    output_dir: str,
    n_replicates: int,
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

    prior_params = simulation.params

    trajectory_length = prior_params.frame_weights.shape[0]
    maxent_params = np.linspace(1, 10, n_replicates)

    for i, dataset in enumerate(datasets):
        seed_results = []

        for j in range(n_replicates):
            try:
                # Generate a new random key for each replicate
                print(
                    f"Running optimization for Dataset Seed {i + 1}/{len(datasets)}, Replicate {j + 1}/{n_replicates}"
                )

                new_params = Simulation_Parameters(
                    frame_weights=jnp.ones(trajectory_length) / trajectory_length,
                    frame_mask=simulation.params.frame_mask,
                    model_parameters=simulation.params.model_parameters,
                    forward_model_weights=jnp.array(
                        [0.5, maxent_params[j], 0.1], dtype=jnp.float32
                    ),
                    forward_model_scaling=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
                    normalise_loss_functions=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
                )

                new_simulation = Simulation(
                    forward_models=models,
                    input_features=simulation.input_features,
                    params=new_params,
                )

                new_simulation.initialise()
                # Create a new simulation with the updated parameters

                new_simulation.forward(new_params)

                optimiser = OptaxOptimizer(
                    parameter_masks={Optimisable_Parameters.frame_weights},
                    learning_rate=5e-4,
                    optimizer="adam",
                )
                # Run optimization
                print(
                    f"Running optimization for Dataset Seed {i + 1}/{len(datasets)}, Replicate {j + 1}/{n_replicates}"
                )
                _, opt_result = run_optimise(
                    new_simulation,
                    initialise=True,
                    optimizer=optimiser,
                    data_to_fit=(dataset, prior_params, pf_prior_data),
                    config=opt_settings,
                    forward_models=models,
                    indexes=[0, 0, 0],
                    loss_functions=[hdx_uptake_l2_loss, maxent_convexKL_loss, regularization_fn],
                )

                # Store the result
                seed_results.append(opt_result)

                # Visualize this optimization run
                print(
                    f"Visualizing results for Dataset Seed {i + 1}/{len(datasets)}, Replicate {j + 1}/{n_replicates}"
                )

                output_suff = os.path.join(output_dir, f"seed_{i + 1}_replicate_{j + 1}")
                hdf5_path = os.path.join(output_suff + "optimization_history.h5")
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
                del new_params
                jax.clear_caches()

            except Exception as e:
                print(f"Error in Dataset Seed {i + 1}, Replicate {j + 1}: {e}")
                # Add a placeholder for failed runs to maintain indexing
                seed_results.append(None)

        results[f"seed_{i}"] = seed_results
    jax.clear_caches()
    return results


def main():
    # for now just save the results to a pickle
    protein = "MoPrP"

    n_seeds = 3
    seeds = [42 + i for i in range(n_seeds)]
    n_replicates = 10

    regularization = {
        "L1": hdx_uptake_l1_loss,
        "mean_L1": hdx_uptake_mean_centred_l1_loss,
        "mean_L2": hdx_uptake_mean_centred_l2_loss,
        # "KL": HDX_uptake_KL_loss,
        # "MAE": HDX_uptake_MAE_loss,
        # "convexKL": HDX_uptake_convex_KL_loss,
    }

    # pick script dir
    base_output_dir = "./notebooks/CrossValidation/"

    base_output_dir = os.path.join(base_output_dir, f"{protein}/jaxENT/AdamW_loreg")
    os.makedirs(base_output_dir, exist_ok=True)
    # remove directory if it exists

    # setup simulation
    datasets, models, opt_settings, pf_prior_data, features, params = setup_simulation(seeds=seeds)

    trajectory_length = params.frame_weights.shape[0]

    new_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length, dtype=jnp.float32) / trajectory_length,
        frame_mask=params.frame_mask,
        model_parameters=params.model_parameters,
        forward_model_weights=jnp.array([1 / (1 * 1 + 1), 1.0, 0.1], dtype=jnp.float32),
        forward_model_scaling=jnp.array([1.0, 0.1, 1.0], dtype=jnp.float32),
        normalise_loss_functions=jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    simulation = Simulation(
        forward_models=models,
        input_features=features,
        params=new_params,
    )

    simulation.initialise()
    # Create a new simulation with the updated parameters
    simulation.forward(new_params)

    for regularization_name, regularization_fn in regularization.items():
        output_dir = os.path.join(base_output_dir, f"{regularization_name}")
        os.system(f"rm -rf {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        history_dict = run_MAE_max_ent_optimization_replicates(
            simulation=simulation,
            datasets=datasets,
            models=models,
            output_dir=output_dir,
            opt_settings=opt_settings,
            pf_prior_data=pf_prior_data,
            regularization_fn=regularization_fn,
            n_replicates=n_replicates,
        )

        print("Optimization complete")
        # save the results
        print(f"Saving results to {output_dir}")


if __name__ == "__main__":
    main()
