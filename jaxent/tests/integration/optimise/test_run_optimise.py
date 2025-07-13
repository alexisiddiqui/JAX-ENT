import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from MDAnalysis import Universe

from jaxent.src.custom_types.config import FeaturiserSettings, OptimiserSettings
from jaxent.src.custom_types.HDX import (
    HDX_peptide,
    HDX_protection_factor,
)
from jaxent.src.data.loader import Dataset, ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config, linear_BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.func.common import find_common_residues
from jaxent.src.models.HDX.BV.forwardmodel import BV_input_features, BV_model, linear_BV_model
from jaxent.src.opt.losses import (
    hdx_pf_l2_loss,
    hdx_pf_mae_loss,
    hdx_uptake_l2_loss,
    hdx_uptake_monotonicity_loss,
    mask_L0_loss,
    max_entropy_loss,
)
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters
from jaxent.src.opt.run import run_optimise
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
    output_dir = ensure_output_dir()
    for fig in figs:
        output_path = os.path.join(output_dir, f"{fig.get_label()}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)


def test_quick_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)
    # print("feature_topology", [top.fragment_index for top in feature_topology[0]])
    # print(feature_topology)
    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    # features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length) / 2,
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(feature_topology[0], start=1)
    ]
    # print("exp_data", exp_data)
    dataset = ExpD_Dataloader(data=exp_data)
    top_segments = [data.top for data in dataset.data]
    print("top_segments", [top.fragment_index for top in top_segments])
    print("top_segments", [top.residue_start for top in top_segments])

    print("type", type(feature_topology[0]))
    # create random split
    splitter = DataSplitter(
        dataset,
        random_seed=42,
        ensemble=universes,
        common_residues=set(feature_topology[0]),
    )
    train_data, val_data = splitter.random_split()
    # print("train_data", [data.top.fragment_index for data in train_data])
    # print("train_data", [data.top.residue_start for data in train_data])

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset,),
        config=opt_settings,
        forward_models=models,
        indexes=[0],
        loss_functions=[hdx_pf_l2_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_sparse_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config), BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    # features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length) / 2,
        model_parameters=[bv_config.forward_parameters, bv_config.forward_parameters],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.ones(2),
        normalise_loss_functions=jnp.ones(2),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(feature_topology[0], start=1)
    ]
    dataset = ExpD_Dataloader(data=exp_data)

    # create random split
    splitter = DataSplitter(
        dataset, random_seed=42, ensemble=universes, common_residues=set(feature_topology[0])
    )
    train_data, val_data = splitter.random_split()

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, dataset),
        config=opt_settings,
        forward_models=models,
        indexes=[0, 0],
        loss_functions=[hdx_pf_l2_loss, mask_L0_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_max_ent_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config), BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters, bv_config.forward_parameters],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.array([1, 10]),
        normalise_loss_functions=jnp.zeros(2),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(feature_topology[0], start=1)
    ]
    # print("exp_data", exp_data)
    dataset = ExpD_Dataloader(data=exp_data)
    top_segments = [data.top for data in dataset.data]
    print("top_segments", [top.fragment_index for top in top_segments])
    print("top_segments", [top.residue_start for top in top_segments])

    print("type", type(feature_topology[0]))
    # create random split
    splitter = DataSplitter(
        dataset,
        random_seed=42,
        ensemble=universes,
        common_residues=set(feature_topology[0]),
    )
    train_data, val_data = splitter.random_split()

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, params),
        config=opt_settings,
        forward_models=models,
        indexes=[0, 10],
        loss_functions=[hdx_pf_l2_loss, max_entropy_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_MAE_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config), BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters, bv_config.forward_parameters],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.array([1.0, 1.0]),
        normalise_loss_functions=jnp.ones(2),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(feature_topology[0], start=1)
    ]
    # print("exp_data", exp_data)
    dataset = ExpD_Dataloader(data=exp_data)
    top_segments = [data.top for data in dataset.data]
    print("top_segments", [top.fragment_index for top in top_segments])
    print("top_segments", [top.residue_start for top in top_segments])

    print("type", type(feature_topology[0]))
    # create random split
    splitter = DataSplitter(
        dataset,
        random_seed=42,
        ensemble=universes,
        common_residues=set(feature_topology[0]),
    )
    train_data, val_data = splitter.random_split()

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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
    pf_prior_data = ExpD_Dataloader(data=pf_prior)

    prior_splitter = DataSplitter(pf_prior_data, random_seed=42, ensemble=universes)
    prior_train_data, prior_val_data = prior_splitter.random_split()

    prior_train_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_train_data)

    prior_val_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_val_data)

    pf_prior_data.train = Dataset(
        data=prior_train_data,
        y_true=jnp.array([data.extract_features() for data in prior_train_data]),
        residue_feature_ouput_mapping=prior_train_sparse_map,
    )

    pf_prior_data.val = Dataset(
        data=prior_val_data,
        y_true=jnp.array([data.extract_features() for data in prior_val_data]),
        residue_feature_ouput_mapping=prior_val_sparse_map,
    )

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior_data),
        config=opt_settings,
        forward_models=[models[0], models[0]],
        indexes=[0, 0],
        loss_functions=[hdx_pf_l2_loss, hdx_pf_mae_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_MAE_sparse_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config), BV_model(bv_config), BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    random_mask = jax.random.bernoulli(jax.random.PRNGKey(42), p=0.01, shape=(trajectory_length,))
    random_mask = jnp.array(random_mask, dtype=jnp.float32)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=random_mask,
        model_parameters=[
            bv_config.forward_parameters,
        ],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.array([1.0, 1.0]),
        normalise_loss_functions=jnp.ones(2),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10 + i, top=top)
        for i, top in enumerate(top_segments, start=1)
    ]

    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=top)
        for pf, top in zip(test_prediction[0].log_Pf, top_segments)
    ]

    dataset = ExpD_Dataloader(data=exp_data)

    # create random split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.random_split()

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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
    pf_prior_data = ExpD_Dataloader(data=pf_prior)

    prior_splitter = DataSplitter(pf_prior_data, random_seed=42, ensemble=universes)
    prior_train_data, prior_val_data = prior_splitter.random_split()

    prior_train_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_train_data)

    prior_val_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_val_data)

    pf_prior_data.train = Dataset(
        data=prior_train_data,
        y_true=jnp.array([data.extract_features() for data in prior_train_data]),
        residue_feature_ouput_mapping=prior_train_sparse_map,
    )

    pf_prior_data.val = Dataset(
        data=prior_val_data,
        y_true=jnp.array([data.extract_features() for data in prior_val_data]),
        residue_feature_ouput_mapping=prior_val_sparse_map,
    )
    custom_optimizer = OptaxOptimizer(
        learning_rate=1e-3,
        optimizer="adam",
        parameter_masks={
            # Optimisable_Parameters.frame_weights,  # 0
            Optimisable_Parameters.frame_mask,  # 3
            # Optimisable_Parameters.forward_model_weights,  # 2
        },
    )

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior_data),
        config=opt_settings,
        forward_models=[models[0], models[0]],
        indexes=[0, 0],
        optimizer=custom_optimizer,
        loss_functions=[hdx_pf_l2_loss, hdx_pf_mae_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_MAE_max_ent_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config), BV_model(bv_config), BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)

    random_mask = jax.random.bernoulli(jax.random.PRNGKey(42), p=0.1, shape=(trajectory_length,))
    random_mask = jnp.array(random_mask, dtype=jnp.float32)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=random_mask,
        model_parameters=[
            bv_config.forward_parameters,
            bv_config.forward_parameters,
            bv_config.forward_parameters,
        ],
        forward_model_weights=jnp.ones(3),
        forward_model_scaling=jnp.array([10.0, 5.0, 1000.0]),
        normalise_loss_functions=jnp.ones(3),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(top_segments, start=1)
    ]

    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=top)
        for pf, top in zip(test_prediction[0].log_Pf, top_segments)
    ]

    dataset = ExpD_Dataloader(data=exp_data)

    # create random split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.random_split()

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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
    pf_prior_data = ExpD_Dataloader(data=pf_prior)

    prior_splitter = DataSplitter(pf_prior_data, random_seed=42, ensemble=universes)
    prior_train_data, prior_val_data = prior_splitter.random_split()

    prior_train_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_train_data)

    prior_val_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_val_data)

    pf_prior_data.train = Dataset(
        data=prior_train_data,
        y_true=jnp.array([data.extract_features() for data in prior_train_data]),
        residue_feature_ouput_mapping=prior_train_sparse_map,
    )

    pf_prior_data.val = Dataset(
        data=prior_val_data,
        y_true=jnp.array([data.extract_features() for data in prior_val_data]),
        residue_feature_ouput_mapping=prior_val_sparse_map,
    )
    custom_optimizer = OptaxOptimizer(
        learning_rate=1e-4,
        optimizer="adam",
        parameter_masks={
            Optimisable_Parameters.frame_weights,  # 0
            Optimisable_Parameters.frame_mask,  # 3
            # Optimisable_Parameters.forward_model_weights,  # 2
        },
    )

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior_data, params),
        config=opt_settings,
        forward_models=[models[0], models[0], models[0]],
        indexes=[0, 0, 0],
        optimizer=custom_optimizer,
        loss_functions=[hdx_pf_l2_loss, hdx_pf_mae_loss, max_entropy_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_MAE_sparse_max_ent_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)

    random_mask = jax.random.bernoulli(jax.random.PRNGKey(42), p=0.9, shape=(trajectory_length,))
    random_mask = jnp.array(random_mask, dtype=jnp.float32)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=random_mask,
        model_parameters=[
            bv_config.forward_parameters,
        ],
        forward_model_weights=jnp.ones(4),
        forward_model_scaling=jnp.array([10.0, 1.0, 1000.0, 0.1]),
        normalise_loss_functions=jnp.ones(4),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(top_segments, start=1)
    ]

    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=top)
        for pf, top in zip(test_prediction[0].log_Pf, top_segments)
    ]

    dataset = ExpD_Dataloader(data=exp_data)

    # create random split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.random_split()

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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
    pf_prior_data = ExpD_Dataloader(data=pf_prior)

    prior_splitter = DataSplitter(pf_prior_data, random_seed=42, ensemble=universes)
    prior_train_data, prior_val_data = prior_splitter.random_split()

    prior_train_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_train_data)

    prior_val_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_val_data)

    pf_prior_data.train = Dataset(
        data=prior_train_data,
        y_true=jnp.array([data.extract_features() for data in prior_train_data]),
        residue_feature_ouput_mapping=prior_train_sparse_map,
    )

    pf_prior_data.val = Dataset(
        data=prior_val_data,
        y_true=jnp.array([data.extract_features() for data in prior_val_data]),
        residue_feature_ouput_mapping=prior_val_sparse_map,
    )
    custom_optimizer = OptaxOptimizer(
        learning_rate=1e-4,
        optimizer="adam",
        parameter_masks={
            Optimisable_Parameters.frame_weights,  # 0
            Optimisable_Parameters.frame_mask,  # 3
            # Optimisable_Parameters.forward_model_weights,  # 2
        },
    )

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior_data, params, params),
        config=opt_settings,
        forward_models=models,
        indexes=[0, 0, 0, 0],
        optimizer=custom_optimizer,
        loss_functions=[hdx_pf_l2_loss, hdx_pf_mae_loss, max_entropy_loss, mask_L0_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_sparse_max_ent_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)

    random_mask = jax.random.bernoulli(jax.random.PRNGKey(42), p=0.01, shape=(trajectory_length,))
    random_mask = jnp.array(random_mask, dtype=jnp.float32)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=random_mask,
        model_parameters=[
            bv_config.forward_parameters,
        ],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.array([1.0, 100.0]),
        normalise_loss_functions=jnp.ones(2),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    prior_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[
            bv_config.forward_parameters,
        ],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.array([1000.0, 10.0]),
        normalise_loss_functions=jnp.ones(2),
    )
    prior_simulation = Simulation(
        forward_models=models, input_features=features, params=prior_params
    )
    simulation.initialise()
    simulation.forward(params)
    prior_simulation.forward(params)
    test_prediction = prior_simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(top_segments, start=1)
    ]

    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=top)
        for pf, top in zip(test_prediction[0].log_Pf, top_segments)
    ]

    dataset = ExpD_Dataloader(data=exp_data)

    # create random split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.random_split()

    # now sparse maps
    train_sparse_map = create_sparse_map(features[0], feature_topology[0], train_data)

    val_sparse_map = create_sparse_map(features[0], feature_topology[0], val_data)

    test_sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(train_sparse_map)
    print(val_sparse_map)
    print(test_sparse_map)

    print(dataset.y_true)
    print(dataset.y_true.shape)

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
    pf_prior_data = ExpD_Dataloader(data=pf_prior)

    prior_splitter = DataSplitter(pf_prior_data, random_seed=42, ensemble=universes)
    prior_train_data, prior_val_data = prior_splitter.random_split()

    prior_train_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_train_data)

    prior_val_sparse_map = create_sparse_map(features[0], feature_topology[0], prior_val_data)

    pf_prior_data.train = Dataset(
        data=prior_train_data,
        y_true=jnp.array([data.extract_features() for data in prior_train_data]),
        residue_feature_ouput_mapping=prior_train_sparse_map,
    )

    pf_prior_data.val = Dataset(
        data=prior_val_data,
        y_true=jnp.array([data.extract_features() for data in prior_val_data]),
        residue_feature_ouput_mapping=prior_val_sparse_map,
    )
    custom_optimizer = OptaxOptimizer(
        learning_rate=1e-3,
        optimizer="adam",
        parameter_masks={
            # Optimisable_Parameters.frame_weights,  # 0
            Optimisable_Parameters.frame_mask,  # 3
            # Optimisable_Parameters.forward_model_weights,  # 2
        },
    )
    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(pf_prior_data, prior_params),
        config=opt_settings,
        forward_models=[models[0]],
        indexes=[0, 0],
        optimizer=custom_optimizer,
        loss_functions=[hdx_pf_l2_loss, max_entropy_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_regularised_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]
    uptake_config = linear_BV_model_Config(num_timepoints=3)

    models = [
        BV_model(bv_config),
        BV_model(bv_config),
        linear_BV_model(uptake_config),
        linear_BV_model(uptake_config),
    ]

    ensemble = Experiment_Builder(universes, models)

    features = run_featurise(ensemble, featuriser_settings, forward_models=models)
    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    bv_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[
            bv_config.forward_parameters,
            bv_config.forward_parameters,
            uptake_config.forward_parameters,
        ],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=bv_params)

    simulation.initialise()
    test_prediction = simulation.forward()
    print("test prediction")
    print(test_prediction)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    fake_pfs = [
        HDX_protection_factor(protection_factor=10, top=None) for _ in range(features_length)
    ]

    fake_uptake = [HDX_peptide(top=None, dfrac=[0.0001, 0.5, 1]) for _ in range(features_length)]

    # print(fake_pfs)

    updake_dataset = ExpD_Dataloader(data=fake_uptake)
    #
    dataset = ExpD_Dataloader(data=fake_pfs)
    # extract the protection factors from the unoptimsied prediction log_Pfs
    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=None) for pf in test_prediction[0].log_Pf
    ]

    pf_prior = ExpD_Dataloader(data=pf_prior)

    print(dataset.y_true)
    print(dataset.y_true.shape)

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior, updake_dataset, None),
        config=opt_settings,
        forward_models=models,
        loss_functions=[
            hdx_pf_l2_loss,
            hdx_pf_mae_loss,
            hdx_uptake_l2_loss,
            hdx_uptake_monotonicity_loss,
        ],
    )


def test_uptake_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]
    uptake_config = linear_BV_model_Config(num_timepoints=3)

    models = [linear_BV_model(uptake_config), linear_BV_model(uptake_config)]

    ensemble = Experiment_Builder(universes, models)

    features = run_featurise(ensemble, featuriser_settings, forward_models=models)
    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[2]
    print(trajectory_length)
    # bv_params = Simulation_Parameters(
    #     frame_weights=jnp.ones(trajectory_length) / trajectory_length,
    #     model_parameters=[bv_config.forward_parameters],
    #     forward_model_weights=jnp.ones(1),
    # )
    linear_bv_params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[uptake_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
    )

    params = [linear_bv_params]

    simulation = Simulation(forward_models=models, input_features=features, params=linear_bv_params)

    simulation.initialise()
    test_prediction = simulation.forward()
    print("test prediction")
    print(test_prediction)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    fake_pfs = [
        HDX_protection_factor(protection_factor=10, top=None) for _ in range(features_length)
    ]

    fake_uptake = [HDX_peptide(top=None, dfrac=[0.0001, 0.5, 1]) for _ in range(features_length)]

    # print(fake_pfs)

    updake_dataset = ExpD_Dataloader(data=fake_uptake)
    #
    dataset = ExpD_Dataloader(data=fake_pfs)
    # extract the protection factors from the unoptimsied prediction log_Pfs
    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=None) for pf in test_prediction[0].uptake
    ]

    pf_prior = ExpD_Dataloader(data=pf_prior)

    print(dataset.y_true)
    print(dataset.y_true.shape)

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=[updake_dataset, None],
        config=opt_settings,
        forward_models=models,
        loss_functions=[hdx_uptake_l2_loss, hdx_uptake_monotonicity_loss],
    )


if __name__ == "__main__":
    import jax

    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # set env XLA_PYTHON_CLIENT_PREALLOCATE=false otherwise jax will preallocate 75% of memory
    # test_create_sparse_map()
    # # test_create_sparse_map_ensemble()
    # test_random_split()
    # test_spatial_split()
    # test_stratified_split()
    # test_cluster_split_sequence()
    # test_quick_max_ent_optimiser()
    test_quick_optimiser()
    # test_quick_sparse_optimiser()
    # test_quick_MAE_optimiser()
    # test_quick_MAE_max_ent_optimiser()
    # test_quick_sparse_max_ent_optimiser()
    # test_quick_MAE_sparse_max_ent_optimiser()
    # test_quick_MAE_sparse_optimiser()
    # test_uptake_optimiser()
    # test_run_featurise_ensemble()

    # try running on jax cpu
    # with jax.default_device(jax.devices("gpu")[0]):
    #     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #     test_regularised_optimiser()
