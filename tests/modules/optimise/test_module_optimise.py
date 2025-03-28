import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from MDAnalysis import Universe

from jaxent.data.loader import Dataset, ExpD_Dataloader
from jaxent.data.splitting.sparse_map import create_sparse_map
from jaxent.data.splitting.split import DataSplitter
from jaxent.featurise import run_featurise
from jaxent.interfaces.builder import Experiment_Builder
from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.models.config import BV_model_Config
from jaxent.models.core import Simulation
from jaxent.models.func.common import find_common_residues
from jaxent.models.HDX.BV.forwardmodel import BV_input_features, BV_model
from jaxent.opt.losses import hdx_pf_l2_loss, hdx_uptake_l2_loss
from jaxent.opt.optimiser import OptaxOptimizer
from jaxent.opt.run import (
    _optimise,  # Import the _optimise function directly
    run_optimise,
)
from jaxent.types.config import FeaturiserSettings, OptimiserSettings
from jaxent.types.HDX import HDX_peptide, HDX_protection_factor
from tests.plots.datasplitting import plot_split_visualization
from tests.plots.optimisation import (
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
    fig_names = ["split", "total_loss", "loss", "weights"]
    output_dir = ensure_output_dir()
    for fig, names in zip(figs, fig_names):
        output_path = os.path.join(output_dir, f"{names}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)


def test_quick_optimiser():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
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

    del simulation
    jax.clear_caches()  # Add this line


def test_underscore_optimiser():
    """Test the _optimise function directly."""
    # Set up the same environment as in test_quick_optimiser
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    # Load universe
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    test_universe = Universe(topology_path, trajectory_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]
    ensemble = Experiment_Builder(universes, models)

    # Featurize
    features, feature_topology = run_featurise(ensemble, featuriser_settings)
    BV_features = features[0]
    trajectory_length = BV_features.features_shape[2]

    # Create simulation parameters
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),  # All active
        model_parameters=[bv_config.forward_parameters],
        forward_model_weights=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )

    # Create and initialize simulation
    simulation = Simulation(forward_models=models, input_features=features, params=params)
    simulation.initialise()

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

    # Create split
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

    # Set up dataset
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

    # Create optimizer and get optimization state
    optimizer = OptaxOptimizer(optimizer="adam")
    opt_state = optimizer.initialise(simulation, [True], _jit_test_args=None)

    # Call _optimise function directly
    n_steps = 50  # Use fewer steps for testing
    tolerance = 1e-6
    convergence = 1e-8
    indexes = [0]
    loss_functions = [hdx_pf_l2_loss]

    result_simulation, result_optimizer = _optimise(
        simulation,
        [dataset],
        n_steps,
        tolerance,
        convergence,
        indexes,
        loss_functions,
        opt_state,
        optimizer,
    )

    # Verify results
    assert result_simulation is not None
    assert result_optimizer is not None
    assert result_optimizer.history is not None

    # Optional: Visualize results
    opt_simulation = (result_simulation, result_optimizer.history)
    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)
    del simulation
    del optimizer
    jax.clear_caches()  # Add this line

    print("_optimise test completed successfully")


def test_uptake_optimiser():
    bv_config = BV_model_Config(num_timepoints=3)

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    # trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
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
    print("test prediction", test_prediction[0].uptake)
    print(test_prediction[0].uptake.shape)

    opt_settings = OptimiserSettings(name="test")

    # create fake experimental dataset

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_peptide(dfrac=[0.1, 1, 10], top=top)
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
        initialise=True,
        data_to_fit=(dataset,),
        config=opt_settings,
        forward_models=models,
        indexes=[0],
        loss_functions=[hdx_uptake_l2_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)

    del simulation
    jax.clear_caches()  # Add this line


if __name__ == "__main__":
    import jax

    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    # test_underscore_optimiser()
    # test_quick_optimiser()
    test_uptake_optimiser()
