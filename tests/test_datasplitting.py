import copy
import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from MDAnalysis import Universe

from jaxent.config.base import FeaturiserSettings
from jaxent.datatypes import (
    Experiment_Ensemble,
    Experimental_Dataset,
    HDX_protection_factor,
)
from jaxent.featurise import run_featurise
from jaxent.forwardmodels.functions import find_common_residues
from jaxent.forwardmodels.models import (
    BV_model,
    BV_model_Config,
)
from jaxent.utils.datasplitter import DataSplitter, apply_sparse_mapping, create_sparse_map


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

    # Convert to 2D array (steps × frames)
    weights_array = jnp.vstack(frame_weights)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        weights_array,
        cmap="viridis",
        xticklabels=100,
        yticklabels=50,
        cbar_kws={"label": "Frame Weight"},
    )

    plt.xlabel("Frame Index")
    plt.ylabel("Optimization Step")
    plt.title("Frame Weights Evolution During Optimization")
    plt.tight_layout()

    return plt.gcf()


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
    plt.show()


# Add this to the end of your test function:


def test_create_sparse_map():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"

    test_universe = Universe(topology_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Ensemble(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    print(features[0].features_shape)
    print(len(feature_topology[0]))

    # create top segments from universe
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    print(len(top_segments))
    top_segments = list(top_segments)
    # sort by residue start
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    # find the subset of top_segments that exists in feature_topology using residue start end and fragment sequence

    print([top.fragment_index for top in top_segments][:3])
    print([top.fragment_index for top in feature_topology[0]][:3])
    print(list(top_segments)[:2])
    print(feature_topology[0][:2])

    # set fragment index to None so that they can be compared to the top_segments
    stripped_feature_topology = copy.deepcopy(feature_topology[0])
    for top in stripped_feature_topology:
        top.fragment_index = None

    # find intersection
    top_segments = set(top_segments)
    stripped_feature_topology = set(stripped_feature_topology)

    top_segments = list(top_segments.intersection(stripped_feature_topology))

    print(len(top_segments))

    # now take random selection
    train_segments = top_segments[: len(top_segments) // 2]
    test_segments = top_segments[len(top_segments) // 2 :]
    # print(train_segments)
    # print(test_segments)

    # now create fake datasets
    print(feature_topology[0][0])

    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in train_segments]
    print("\nBefore creating sparse map:")
    print(f"Feature topology length: {len(feature_topology[0])}")
    print(f"Train segments length: {len(train_segments)}")
    print(f"Exp data length: {len(exp_data)}")
    print(f"First train segment: {train_segments[0] if train_segments else 'empty'}")
    print(f"First exp data: {exp_data[0] if exp_data else 'empty'}")
    sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(sparse_map)

    feature_vector = jnp.ones((features[0].features_shape[0]))
    print(feature_vector.shape)
    mapped_data = apply_sparse_mapping(sparse_map, feature_vector)
    print(mapped_data)
    print(mapped_data.shape)
    # now create sparse map of input features


def test_create_sparse_map_ensemble():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    test_universe = Universe(topology_path, trajectory_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Ensemble(universes, models)

    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    assert len(features) == len(models)

    print(features[0].features_shape)
    print(len(feature_topology[0]))

    # create top segments from universe
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    print(len(top_segments))
    top_segments = list(top_segments)
    # sort by residue start
    top_segments = sorted(top_segments, key=lambda x: x.residue_start)
    # find the subset of top_segments that exists in feature_topology using residue start end and fragment sequence

    print([top.fragment_index for top in top_segments][:3])
    print([top.fragment_index for top in feature_topology[0]][:3])
    print(list(top_segments)[:2])
    print(feature_topology[0][:2])

    # set fragment index to None so that they can be compared to the top_segments
    stripped_feature_topology = copy.deepcopy(feature_topology[0])
    for top in stripped_feature_topology:
        top.fragment_index = None

    # find intersection
    top_segments = set(top_segments)
    stripped_feature_topology = set(stripped_feature_topology)

    top_segments = list(top_segments.intersection(stripped_feature_topology))

    print(len(top_segments))

    # now take random selection
    train_segments = top_segments[: len(top_segments) // 2]
    test_segments = top_segments[len(top_segments) // 2 :]
    # print(train_segments)
    # print(test_segments)

    # now create fake datasets
    print(feature_topology[0][0])

    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in train_segments]
    print("\nBefore creating sparse map:")
    print(f"Feature topology length: {len(feature_topology[0])}")
    print(f"Train segments length: {len(train_segments)}")
    print(f"Exp data length: {len(exp_data)}")
    print(f"First train segment: {train_segments[0] if train_segments else 'empty'}")
    print(f"First exp data: {exp_data[0] if exp_data else 'empty'}")
    sparse_map = create_sparse_map(features[0], feature_topology[0], exp_data)

    print(sparse_map)

    feature_vector = jnp.ones(features[0].features_shape[0])
    print(feature_vector.shape)
    mapped_data = apply_sparse_mapping(sparse_map, feature_vector)
    print(mapped_data)
    print(mapped_data.shape)


# now create sparse map of input features
def test_random_split():
    # Setup similar to other tests
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Ensemble(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset for testing
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = Experimental_Dataset(data=exp_data)

    # Create splitter and test random split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.random_split()

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization
    plot_split_visualization(train_data, val_data, dataset.data)
    plt.show()


def test_spatial_split():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Ensemble(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = Experimental_Dataset(data=exp_data)

    # Create splitter and test spatial split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.spatial_split(universes)

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization
    plot_split_visualization(train_data, val_data, dataset.data)
    plt.show()


def test_stratified_split():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Ensemble(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset with varying protection factors for better stratification testing
    exp_data = [
        HDX_protection_factor(protection_factor=i, top=top)
        for i, top in enumerate(top_segments, start=1)
    ]
    dataset = Experimental_Dataset(data=exp_data)

    # Create splitter and test stratified split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.stratified_split(n_strata=5)

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization
    plot_split_visualization(train_data, val_data, dataset.data)
    plt.show()


def test_cluster_split_sequence():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Ensemble(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = Experimental_Dataset(data=exp_data)

    # Create splitter and test cluster split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.cluster_split(
        n_clusters=5, peptide=True, cluster_index="sequence"
    )

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization
    plot_split_visualization(train_data, val_data, dataset.data)
    plt.show()


def test_cluster_split_featuress():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Ensemble(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = Experimental_Dataset(data=exp_data)

    # Create splitter and test cluster split
    splitter = DataSplitter(dataset, random_seed=42, ensemble=universes)
    train_data, val_data = splitter.cluster_split(
        n_clusters=5, peptide=True, cluster_index="residue_index"
    )

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization
    plot_split_visualization(train_data, val_data, dataset.data)
    plt.show()


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
    test_cluster_split_featuress()
    # test_quick_max_ent_optimiser()

    # test_quick_MAE_optimiser()

    # test_uptake_optimiser()
    # test_run_featurise_ensemble()

    # try running on jax cpu
    # with jax.default_device(jax.devices("gpu")[0]):
    #     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #     test_regularised_optimiser()
