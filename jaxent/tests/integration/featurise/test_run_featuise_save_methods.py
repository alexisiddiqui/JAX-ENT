import os
from pathlib import Path

import jax.numpy as jnp
from MDAnalysis import Universe

import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.custom_types.features import Input_Features
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.tests.test_utils import get_inst_path


def test_featurise_save_load():
    """
    Test script to:
    1. Run featurise
    2. Save features using jnp.savez
    3. Save feature topology using JSON
    4. Load features back
    5. Load feature topology back
    6. Verify data integrity using built-in comparison
    """

    # Define paths
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)

    output_dir = Path(__file__).parents[1] / "test_featurise_output"

    print("Setting up test environment...")
    os.makedirs(output_dir, exist_ok=True)

    # Define trajectory and topology paths
    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    trajectory_path = inst_path / "clean" / "BPTI" / "BPTI_sampled_500.xtc"

    # Initialize model and universe
    bv_config = BV_model_Config(num_timepoints=4)
    bv_config.timepoints = jnp.array([0.1, 1.0, 10.0, 100.0])

    test_universe = Universe(str(topology_path), str(trajectory_path))
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create experiment builder
    ensemble = Experiment_Builder(universes, models)
    featuriser_settings = FeaturiserSettings(name="test_features", batch_size=None)

    # Run featurisation
    print("Running featurisation...")
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Extract the first feature and topology set
    features_set = features[0]
    topology_set = feature_topology[0]

    print(f"Features shape: {features_set.features_shape}")
    print(f"Topology count: {len(topology_set)}")

    # STEP 1: Save features using jnp.savez
    features_path = output_dir / "features.npz"
    topology_path = output_dir / "topology.json"
    features_set.save(str(features_path))

    pt.PTSerialiser.save_list_to_json(topology_set, str(topology_path))
    # STEP 3: Load features back
    print("Loading features and topology...")

    # Reconstruct BV_input_features
    loaded_features = Input_Features.load(str(features_path))

    # STEP 4: Load topology from JSON

    loaded_topology = pt.PTSerialiser.load_list_from_json(str(topology_path))

    # STEP 5: Verify data integrity
    print("Verifying data integrity...")

    assert isinstance(loaded_features, BV_input_features), (
        "Loaded features are not of type BV_input_features"
    )

    # Check features
    features_match = (
        jnp.all(jnp.isclose(features_set.heavy_contacts, loaded_features.heavy_contacts))
        and jnp.all(jnp.isclose(features_set.acceptor_contacts, loaded_features.acceptor_contacts))
        and jnp.all(jnp.isclose(features_set.k_ints, loaded_features.k_ints))
    )

    print(f"Features match: {features_match}")

    # Check topology using built-in equality comparison
    topology_match = len(topology_set) == len(loaded_topology)
    mismatched_indices = []
    if topology_match:
        for i, (orig, loaded) in enumerate(zip(topology_set, loaded_topology)):
            if orig != loaded:  # Uses Partial_Topology.__eq__ method
                topology_match = False
                mismatched_indices.append(i)

    print(f"Topology match: {topology_match}")
    if not topology_match and mismatched_indices:
        print(f"Mismatches at indices: {mismatched_indices}")
        for idx in mismatched_indices[:3]:  # Show first 3 mismatches
            print(f"Original: {topology_set[idx]}")
            print(f"Loaded: {loaded_topology[idx]}")
            print("-" * 40)

    # Assert for pytest
    assert features_match, "Features do not match after save/load"
    assert topology_match, "Topology objects do not match after save/load"

    print("✅ Test passed: Features and topology correctly saved and loaded")


def test_featurise_save_load_features():
    """
    Test script to:
    1. Run featurise
    2. Save features using jnp.savez
    3. Save feature topology using JSON
    4. Load features back
    5. Load feature topology back
    6. Verify data integrity using built-in comparison
    """

    # Define paths
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)

    output_dir = Path(__file__).parents[1] / "test_featurise_output"

    print("Setting up test environment...")
    os.makedirs(output_dir, exist_ok=True)

    # Define trajectory and topology paths
    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    trajectory_path = inst_path / "clean" / "BPTI" / "BPTI_sampled_500.xtc"

    # Initialize model and universe
    bv_config = BV_model_Config(num_timepoints=4)
    bv_config.timepoints = jnp.array([0.1, 1.0, 10.0, 100.0])

    test_universe = Universe(str(topology_path), str(trajectory_path))
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create experiment builder
    ensemble = Experiment_Builder(universes, models)
    featuriser_settings = FeaturiserSettings(name="test_features", batch_size=None)

    # Run featurisation
    print("Running featurisation...")
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Extract the first feature and topology set
    features_set = features[0]
    topology_set = feature_topology[0]

    print(f"Features shape: {features_set.features_shape}")
    print(f"Topology count: {len(topology_set)}")

    # STEP 1: Save features using jnp.savez
    features_path = output_dir / "features.npz"
    topology_path = output_dir / "topology.json"
    features_set.save_features(str(features_path))

    pt.PTSerialiser.save_list_to_json(topology_set, str(topology_path))
    # STEP 3: Load features back
    print("Loading features and topology...")

    # Reconstruct BV_input_features
    loaded_features = BV_input_features.load_features(str(features_path))

    # STEP 4: Load topology from JSON

    loaded_topology = pt.PTSerialiser.load_list_from_json(str(topology_path))

    # STEP 5: Verify data integrity
    print("Verifying data integrity...")

    # Check features
    features_match = (
        jnp.all(jnp.isclose(features_set.heavy_contacts, loaded_features.heavy_contacts))
        and jnp.all(jnp.isclose(features_set.acceptor_contacts, loaded_features.acceptor_contacts))
        and jnp.all(jnp.isclose(features_set.k_ints, loaded_features.k_ints))
    )

    print(f"Features match: {features_match}")

    # Check topology using built-in equality comparison
    topology_match = len(topology_set) == len(loaded_topology)
    mismatched_indices = []
    if topology_match:
        for i, (orig, loaded) in enumerate(zip(topology_set, loaded_topology)):
            if orig != loaded:  # Uses Partial_Topology.__eq__ method
                topology_match = False
                mismatched_indices.append(i)

    print(f"Topology match: {topology_match}")
    if not topology_match and mismatched_indices:
        print(f"Mismatches at indices: {mismatched_indices}")
        for idx in mismatched_indices[:3]:  # Show first 3 mismatches
            print(f"Original: {topology_set[idx]}")
            print(f"Loaded: {loaded_topology[idx]}")
            print("-" * 40)

    # Assert for pytest
    assert features_match, "Features do not match after save/load"
    assert topology_match, "Topology objects do not match after save/load"

    print("✅ Test passed: Features and topology correctly saved and loaded")


if __name__ == "__main__":
    test_featurise_save_load()
