import os

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
import sys

sys.path.insert(0, base_dir)
import copy

import jax.numpy as jnp
import matplotlib.pyplot as plt
from MDAnalysis import Universe

from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping, create_sparse_map
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.func.common import find_common_residues
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.types.config import FeaturiserSettings
from jaxent.src.types.HDX import HDX_protection_factor
from jaxent.tests.plots.datasplitting import plot_split_visualization


# Ensure output directory exists
def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    output_dir = "tests/_plots/datasplitting"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def test_create_sparse_map():
    bv_config = BV_model_Config()

    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"

    test_universe = Universe(topology_path)

    universes = [test_universe]

    models = [BV_model(bv_config)]

    ensemble = Experiment_Builder(universes, models)

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
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]
    print("top_segments", len(top_segments))

    print([top.fragment_index for top in top_segments][:3])

    # Create fake dataset for testing
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = ExpD_Dataloader(data=exp_data)
    print("dataset", len(dataset.data))

    print([top.fragment_index for top in dataset.top][:3])
    exp_top_segments = [top.top for top in dataset.data]
    # Create splitter and test random split
    splitter = DataSplitter(
        dataset, random_seed=42, ensemble=universes, peptide=False, common_residues=exp_top_segments
    )
    train_data, val_data = splitter.random_split()

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization and save
    fig = plot_split_visualization(train_data, val_data, dataset.data)
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "random_split.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def test_spatial_split():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    print("top_segments", len(top_segments))

    # Create fake dataset
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = ExpD_Dataloader(data=exp_data)
    print("dataset", len(dataset.data))

    # Create splitter and test spatial split
    exp_top_segments = [top.top for top in dataset.data]
    # Create splitter and test random split
    splitter = DataSplitter(
        dataset, random_seed=42, ensemble=universes, peptide=False, common_residues=exp_top_segments
    )
    train_data, val_data = splitter.spatial_split(universes)

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization and save
    fig = plot_split_visualization(train_data, val_data, dataset.data)
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "spatial_split.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def test_stratified_split():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Builder(universes, models)
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
    dataset = ExpD_Dataloader(data=exp_data)

    # Create splitter and test stratified split
    exp_top_segments = [top.top for top in dataset.data]
    # Create splitter and test random split
    splitter = DataSplitter(
        dataset, random_seed=42, ensemble=universes, peptide=False, common_residues=exp_top_segments
    )
    train_data, val_data = splitter.stratified_split(n_strata=5)

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization and save
    fig = plot_split_visualization(train_data, val_data, dataset.data)
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "stratified_split.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def test_cluster_split_sequence():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = ExpD_Dataloader(data=exp_data)

    # Create splitter and test cluster split
    exp_top_segments = [top.top for top in dataset.data]
    # Create splitter and test random split
    splitter = DataSplitter(
        dataset, random_seed=42, ensemble=universes, peptide=False, common_residues=exp_top_segments
    )
    train_data, val_data = splitter.cluster_split(
        n_clusters=5, peptide=True, cluster_index="sequence"
    )

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization and save
    fig = plot_split_visualization(train_data, val_data, dataset.data)
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "cluster_split_sequence.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def test_cluster_split_featuress():
    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    test_universe = Universe(topology_path)
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and get features
    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Get common residues
    top_segments = find_common_residues(
        universes, ignore_mda_selection="(resname PRO or resid 1) "
    )[0]

    # Create fake dataset
    exp_data = [HDX_protection_factor(protection_factor=10, top=top) for top in top_segments]
    dataset = ExpD_Dataloader(data=exp_data)

    # Create splitter and test cluster split
    exp_top_segments = [top.top for top in dataset.data]
    # Create splitter and test random split
    splitter = DataSplitter(
        dataset, random_seed=42, ensemble=universes, peptide=False, common_residues=exp_top_segments
    )
    train_data, val_data = splitter.cluster_split(
        n_clusters=5, peptide=True, cluster_index="residue_index"
    )

    # Basic assertions
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(train_data) + len(val_data) <= len(dataset.data)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")

    # Plot split visualization and save
    fig = plot_split_visualization(train_data, val_data, dataset.data)
    output_dir = ensure_output_dir()
    output_path = os.path.join(output_dir, "cluster_split_features.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    import jax

    print("Local devices:", jax.local_devices())
    print("CPU devices:", jax.devices("cpu"))
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Ensure output directory exists
    ensure_output_dir()

    # set env XLA_PYTHON_CLIENT_PREALLOCATE=false otherwise jax will preallocate 75% of memory
    test_create_sparse_map()

    # test_create_sparse_map_ensemble()
    # test_random_split()
    # test_spatial_split()
    # test_stratified_split()
    # test_cluster_split_sequence()
    # test_cluster_split_featuress()

    # test_quick_max_ent_optimiser()
    # test_quick_MAE_optimiser()
    # test_uptake_optimiser()
    # test_run_featurise_ensemble()

    # try running on jax cpu
    # with jax.default_device(jax.devices("gpu")[0]):
    #     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #     test_regularised_optimiser()
