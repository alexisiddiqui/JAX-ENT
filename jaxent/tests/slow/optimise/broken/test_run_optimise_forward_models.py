import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from MDAnalysis import Universe

from jaxent.src.custom_types.config import FeaturiserSettings, OptimiserSettings
from jaxent.src.custom_types.HDX import (
    HDX_peptide,
    HDX_protection_factor,
)
from jaxent.src.data.loader import (
    Dataset,
    ExpD_Dataloader,
)
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import (
    Experiment_Builder,
)
from jaxent.src.interfaces.simulation import (
    Simulation_Parameters,
)
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
)
from jaxent.src.models.HDX.BV.forwardmodel import (
    BV_model,
    BV_model_Config,
    linear_BV_model,
    linear_BV_model_Config,
)
from jaxent.src.opt.losses import (
    hdx_pf_l2_loss,
    hdx_pf_mae_loss,
    hdx_uptake_l2_loss,
    hdx_uptake_monotonicity_loss,
    max_entropy_loss,
)
from jaxent.src.opt.optimiser import OptaxOptimizer, Optimisable_Parameters
from jaxent.src.opt.run import run_optimise


def plot_split_visualization(train_data, val_data, exp_data):
    """
    Create a visualization of the train/validation split along residue indices.
    """
    plt.figure(figsize=(12, 6))

    # Get residue indices for each dataset
    train_residues = [d.top.residue_start for d in train_data]
    val_residues = [d.top.residue_start for d in val_data]
    all_residues = [d.top.residue_start for d in exp_data]

    # Create boolean masks for plotting
    residue_range = np.arange(min(all_residues), max(all_residues) + 1)
    train_mask = np.isin(residue_range, train_residues)
    val_mask = np.isin(residue_range, val_residues)

    # Plot residue coverage
    plt.scatter(
        residue_range[train_mask],
        np.ones_like(residue_range[train_mask]),
        label="Training",
        alpha=0.6,
        s=100,
    )
    plt.scatter(
        residue_range[val_mask],
        np.zeros_like(residue_range[val_mask]),
        label="Validation",
        alpha=0.6,
        s=100,
    )

    plt.yticks([0, 1], ["Validation", "Training"])
    plt.xlabel("Residue Index")
    plt.title("Train/Validation Split by Residue")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_total_losses(opt_history):
    """
    Plot training and validation loss components over optimization steps.
    """
    steps = range(len(opt_history.states))
    train_losses = [state.losses.total_train_loss for state in opt_history.states]
    val_losses = [state.losses.total_val_loss for state in opt_history.states]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss", marker="o", markersize=3)
    plt.plot(steps, val_losses, label="Validation Loss", marker="o", markersize=3)

    plt.xlabel("Optimization Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.yscale("log")
    plt.tight_layout()

    return plt.gcf()


def plot_loss_components(opt_history):
    """
    Plot training and validation loss components over optimization steps,
    showing both total loss and individual components on separate subplots.
    """
    steps = range(len(opt_history.states))

    # Get individual loss components
    train_components = np.array([state.losses.train_losses for state in opt_history.states])
    val_components = np.array([state.losses.val_losses for state in opt_history.states])
    scaled_train = np.array([state.losses.scaled_train_losses for state in opt_history.states])
    scaled_val = np.array([state.losses.scaled_val_losses for state in opt_history.states])
    total_train = np.array([state.losses.total_train_loss for state in opt_history.states])
    total_val = np.array([state.losses.total_val_loss for state in opt_history.states])

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot training losses
    colors = plt.cm.Set2(np.linspace(0, 1, train_components.shape[1]))

    ax1.set_title("Training Loss Components")
    # Plot individual components
    for i, color in enumerate(colors):
        ax1.plot(
            steps,
            train_components[:, i],
            label=f"Component {i + 1}",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax1.plot(
            steps, scaled_train[:, i], label=f"Scaled Component {i + 1}", color=color, alpha=1.0
        )
    # Plot total loss
    ax1.plot(steps, total_train, label="Total Loss", color="black", linewidth=2)
    # ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Loss")

    # Plot validation losses
    ax2.set_title("Validation Loss Components")
    # Plot individual components
    for i, color in enumerate(colors):
        ax2.plot(
            steps,
            val_components[:, i],
            label=f"Component {i + 1}",
            color=color,
            linestyle="--",
            alpha=0.7,
        )
        ax2.plot(steps, scaled_val[:, i], label=f"Scaled Component {i + 1}", color=color, alpha=1.0)
    # Plot total loss
    ax2.plot(steps, total_val, label="Total Loss", color="black", linewidth=2)
    # ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Loss")

    plt.tight_layout()
    return fig


def plot_frame_weights_heatmap(opt_history):
    """
    Create a heatmap showing the evolution of frame weights during optimization.
    """
    # Extract frame weights from each state
    frame_weights = [state.params.frame_weights for state in opt_history.states]
    frame_mask = [state.params.frame_mask for state in opt_history.states]

    # Apply frame mask
    frame_weights = [weights * mask for weights, mask in zip(frame_weights, frame_mask)]

    # Convert to 2D array (steps × frames)
    weights_array = jnp.vstack(frame_weights)

    plt.figure(figsize=(12, 6))
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmap.set_under("black")  # Set color for values under the minimum (i.e., zero)

    sns.heatmap(
        weights_array,
        cmap=cmap,
        xticklabels=100,
        yticklabels=50,
        cbar_kws={"label": "Frame Weight"},
        vmin=1e-9,  # Ensure black is shown for 0 by setting a minimum value slightly above zero
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Optimization Step")
    plt.title("Frame Weights Evolution During Optimization")
    plt.tight_layout()

    return plt.gcf()


def plot_forward_model_scaling(opt_history):
    """
    Create a plot showing the evolution of forward model scaling during optimization.
    """
    steps = range(len(opt_history.states))

    # Extract forward model scaling from each state
    scaling_values = np.array([state.params.forward_model_scaling for state in opt_history.states])

    plt.figure(figsize=(10, 6))

    # Plot each scaling parameter as a separate line
    for i in range(scaling_values.shape[1]):
        plt.plot(
            steps, scaling_values[:, i], label=f"Model {i + 1} Scaling", marker="o", markersize=3
        )

    plt.xlabel("Optimization Step")
    plt.ylabel("Scaling Value")
    plt.title("Forward Model Scaling Evolution During Optimization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_forward_model_weights(opt_history):
    """
    Create a plot showing the evolution of forward model weights during optimization.
    """
    steps = range(len(opt_history.states))

    # Extract forward model weights from each state
    weight_values = np.array([state.params.forward_model_weights for state in opt_history.states])

    plt.figure(figsize=(10, 6))

    # Plot each weight as a separate line
    for i in range(weight_values.shape[1]):
        plt.plot(
            steps, weight_values[:, i], label=f"Model {i + 1} Weight", marker="o", markersize=3
        )

    plt.xlabel("Optimization Step")
    plt.ylabel("Weight Value")
    plt.title("Forward Model Weights Evolution During Optimization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_forward_model_importance(opt_history):
    """
    Create a plot showing the combined importance of each forward model during optimization.
    Importance is calculated as weight * scaling for each model.
    """
    steps = range(len(opt_history.states))

    # Extract forward model weights and scaling from each state
    weight_values = np.array([state.params.forward_model_weights for state in opt_history.states])
    scaling_values = np.array([state.params.forward_model_scaling for state in opt_history.states])

    # Calculate importance as weight * scaling
    importance_values = weight_values * scaling_values

    # Calculate normalized importance (relative contribution of each model)
    normalized_importance = importance_values / np.sum(importance_values, axis=1, keepdims=True)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot raw importance
    ax1.set_title("Raw Model Importance (Weight × Scaling)")
    for i in range(importance_values.shape[1]):
        ax1.plot(
            steps,
            importance_values[:, i],
            label=f"Model {i + 1} Importance",
            marker="o",
            markersize=3,
        )

    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Importance (Weight × Scaling)")

    # Plot normalized importance as area chart
    ax2.set_title("Normalized Model Importance (Relative Contribution)")
    ax2.stackplot(
        steps,
        [normalized_importance[:, i] for i in range(normalized_importance.shape[1])],
        labels=[f"Model {i + 1}" for i in range(normalized_importance.shape[1])],
        alpha=0.7,
    )

    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Relative Importance")

    plt.tight_layout()
    return fig


# Update the visualization function to include the new plots
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

    # Add new plots for forward model scaling and weights
    # scaling_fig = plot_forward_model_scaling(history)
    # weight_fig = plot_forward_model_weights(history)
    importance_fig = plot_forward_model_importance(history)

    # Display plots
    plt.show()


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

    bad_bv_config = BV_model_Config()
    bad_bv_config.heavy_radius = 100
    bad_bv_config.o_radius = 100

    models = [BV_model(bv_config), BV_model(bad_bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters, bad_bv_config.forward_parameters],
        forward_model_weights=jnp.array([100000.0, 1.0]),
        forward_model_scaling=jnp.ones(2),
        normalise_loss_functions=jnp.ones(2),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(top_segments, start=1)
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

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, dataset),
        config=opt_settings,
        forward_models=models,
        indexes=[0, 1],
        loss_functions=[hdx_pf_l2_loss, hdx_pf_l2_loss],
    )

    visualize_optimization_results(train_data, val_data, exp_data, opt_simulation)


def test_quick_optimiser_REAL():
    bv_config = BV_model_Config(num_timepoints=3)
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0])
    print(bv_config.forward_parameters)
    # bv_config.num_timepoints = 3
    print(bv_config.key)
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
    uptake_model_parameters = bv_config.forward_parameters

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[
            uptake_model_parameters,
        ],
        forward_model_weights=jnp.array([1.0]),
        forward_model_scaling=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()

    simulation.forward(params)
    print(simulation)
    test_prediction = list(simulation.outputs)
    print("test prediction", test_prediction)
    print(test_prediction[0].uptake)

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
    )[0]

    segs_data = (
        "/home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/BPTI/BPTI_residue_segs_trimmed.txt"
    )

    with open(segs_data, "r") as f:
        segs_text = [line.strip() for line in f.readlines()]
        segs = [line.split() for line in segs_text]

    segs = [set((int(start), int(end))) for start, end in segs]

    print(segs_text)
    print(segs)

    dfrac_data = "/home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/BPTI/BPTI_expt_dfracs_clean_trimmed.dat"

    with open(dfrac_data, "r") as f:
        # skip first line and then read in vals
        dfrac_text = [line.strip() for line in f.readlines()[1:]]
        dfracs = [line.split() for line in dfrac_text]
    print(dfracs)

    dfracs = [np.array(line, dtype=float) for line in dfracs]

    exp_top_segs = []
    exp_dfracs = []
    for top_seg, dfrac in zip(sorted(top_segments, key=lambda x: x.residue_start), dfracs):
        # print(top_seg)
        res_start_res_end = set((int(top_seg.residue_start), int(top_seg.residue_end) + 1))
        # print(res_start_res_end)

        if res_start_res_end in segs:
            print(res_start_res_end)
            exp_top_segs.append(top_seg)
            exp_dfracs.append(dfrac)

    print(exp_top_segs)
    print(exp_dfracs)

    assert len(exp_top_segs) == len(exp_dfracs), "Lengths of top segs and dfracs don't match"

    exp_dfracs = [jnp.array(df) for df in exp_dfracs]

    exp_data = [
        # HDX_protection_factor(protection_factor=10 * (i % 5 + 1), top=top)  # Vary protection factors
        HDX_peptide(dfrac=df, top=top)  #
        for i, (top, df) in enumerate(zip(exp_top_segs, exp_dfracs), start=1)
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

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset,),
        config=opt_settings,
        forward_models=models,
        indexes=[0],
        loss_functions=[hdx_uptake_l2_loss],
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

    bad_bv_config = BV_model_Config()
    bad_bv_config.heavy_radius = 1
    bad_bv_config.o_radius = 1

    models = [BV_model(bv_config), BV_model(bad_bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters, bad_bv_config.forward_parameters],
        forward_model_weights=jnp.array([1.0, 100.0, 1.0, 1]),
        forward_model_scaling=jnp.array([1.0, 1.0, 1, 5]),
        normalise_loss_functions=jnp.array([1, 1, 0, 0]),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(top_segments, start=1)
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
    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=top)
        for pf, top in zip(test_prediction[0].log_Pf, top_segments)
    ]
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
        data_to_fit=(dataset, dataset, params, pf_prior_data),
        config=opt_settings,
        optimisable_funcs=[1, 1, 0, 0],
        forward_models=models,
        indexes=[0, 1, 10, 0],
        loss_functions=[hdx_pf_l2_loss, hdx_pf_l2_loss, max_entropy_loss, hdx_pf_mae_loss],
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
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters, bv_config.forward_parameters],
        forward_model_weights=jnp.ones(2),
        forward_model_scaling=jnp.array([1.0, 1.0]),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
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

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior_data),
        config=opt_settings,
        forward_models=models,
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
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
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

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
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

    opt_simulation = run_optimise(
        simulation,
        data_to_fit=(dataset, pf_prior_data),
        config=opt_settings,
        forward_models=models,
        indexes=[0, 0],
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

    bad_bv_config = BV_model_Config()
    bad_bv_config.heavy_radius = 1
    bad_bv_config.o_radius = 1

    models = [BV_model(bv_config), BV_model(bad_bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
    print(trajectory_length)
    params = Simulation_Parameters(
        frame_weights=jnp.ones(trajectory_length) / trajectory_length,
        frame_mask=jnp.ones(trajectory_length),
        model_parameters=[bv_config.forward_parameters, bad_bv_config.forward_parameters],
        forward_model_weights=jnp.array([1.0, 100.0, 1.0, 1]),
        forward_model_scaling=jnp.array([1.0, 1.0, 1, 5]),
        normalise_loss_functions=jnp.array([1, 1, 0, 0]),
    )

    simulation = Simulation(forward_models=models, input_features=features, params=params)

    simulation.initialise()
    simulation.forward(params)
    test_prediction = simulation.outputs
    print("test prediction", test_prediction[0].log_Pf)
    print(test_prediction[0].log_Pf.shape)

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=10, top=top)
        for i, top in enumerate(top_segments, start=1)
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
    pf_prior = [
        HDX_protection_factor(protection_factor=pf, top=top)
        for pf, top in zip(test_prediction[0].log_Pf, top_segments)
    ]
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
        data_to_fit=(dataset, dataset, params, pf_prior_data),
        config=opt_settings,
        optimisable_funcs=[1, 1, 0, 0],
        forward_models=models,
        indexes=[0, 1, 10, 0],
        loss_functions=[hdx_pf_l2_loss, hdx_pf_l2_loss, max_entropy_loss, hdx_pf_mae_loss],
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

    models = [BV_model(bv_config), BV_model(bv_config), BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    BV_features: BV_input_features = features[0]
    print("BV Features length", BV_features.features_shape)

    features_length = BV_features.features_shape[0]
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
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

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
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
        forward_models=models,
        indexes=[0, 0],
        optimizer=custom_optimizer,
        loss_functions=[hdx_pf_l2_loss, hdx_pf_mae_loss],
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
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
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

    opt_settings = OptimiserSettings(name="test", n_steps=25)

    # create fake experimental dataset

    pf = HDX_protection_factor(protection_factor=10, top=None)

    test = pf.top
    print(test)

    # Get common residues
    top_segments = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
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
        data_to_fit=(dataset, pf_prior_data),
        config=opt_settings,
        forward_models=models,
        indexes=[0, 0],
        optimizer=custom_optimizer,
        loss_functions=[hdx_pf_l2_loss, hdx_pf_mae_loss],
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
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
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

    opt_settings = OptimiserSettings(name="test", n_steps=25)

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
    trajectory_length = BV_features.features_shape[1]  # <-- changed from [2] to [1]
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

    opt_settings = OptimiserSettings(name="test", n_steps=25)

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
    # test_quick_optimiser()
    test_quick_optimiser_REAL()
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
    # set env XLA_PYTHON_CLIENT_PREALLOCATE=false otherwise jax will preallocate 75% of memory
    # test_create_sparse_map()
    # # test_create_sparse_map_ensemble()
    # test_random_split()
    # test_spatial_split()
    # test_stratified_split()
    # test_cluster_split_sequence()
    # test_quick_max_ent_optimiser()
    # test_quick_optimiser()
    test_quick_optimiser_REAL()
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
    #     test_regularised_optimiser()
