import json
import os

import jax.numpy as jnp
from MDAnalysis import Universe

from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config
from jaxent.src.types.config import FeaturiserSettings


def test_featurise_save_load(trajectory_path: str, topology_path: str, output_dir: str):
    """
    Test script to:
    1. Run featurise
    2. Save features using jnp.savez
    3. Save feature topology using JSON
    4. Load features back
    5. Load feature topology back
    6. Verify data integrity using built-in comparison
    """
    print("Setting up test environment...")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model and universe
    bv_config = BV_model_Config(num_timepoints=4)
    bv_config.timepoints = jnp.array([0.1, 1.0, 10.0, 100.0])

    test_universe = Universe(topology_path, trajectory_path)
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
    features_path = os.path.join(output_dir, "features.jpz")
    print(f"Saving features to {features_path}")
    jnp.savez(
        features_path,
        heavy_contacts=features_set.heavy_contacts,
        acceptor_contacts=features_set.acceptor_contacts,
        k_ints=features_set.k_ints,
    )

    # STEP 2: Convert topology to JSON-serializable format
    def topology_to_dict(topology):
        return {
            "chain": topology.chain,
            "fragment_sequence": topology.fragment_sequence,
            "residue_start": int(topology.residue_start),  # Convert to standard Python int
            "residue_end": int(topology.residue_end),  # Convert to standard Python int
            "peptide_trim": int(topology.peptide_trim),
            "fragment_index": int(topology.fragment_index)
            if topology.fragment_index is not None
            else None,
            "length": int(topology.length),
        }

    # Convert all topologies to dict
    topology_dicts = [topology_to_dict(top) for top in topology_set]

    # Save topologies as JSON
    topology_path = os.path.join(output_dir, "topology.json")
    print(f"Saving topology to {topology_path}")
    with open(topology_path, "w") as f:
        json.dump(topology_dicts, f, indent=2)

    # STEP 3: Load features back
    print("Loading features and topology...")
    features_path_npz = features_path + ".npz"  # jnp.savez adds .npz extension
    loaded_features_dict = jnp.load(features_path_npz)

    # Reconstruct BV_input_features
    loaded_features = BV_input_features(
        heavy_contacts=loaded_features_dict["heavy_contacts"],
        acceptor_contacts=loaded_features_dict["acceptor_contacts"],
        k_ints=loaded_features_dict["k_ints"],
    )

    # STEP 4: Load topology from JSON
    with open(topology_path, "r") as f:
        loaded_topology_dicts = json.load(f)

    # Reconstruct Partial_Topology objects
    from jaxent.src.interfaces.topology import Partial_Topology

    loaded_topology = []
    for top_dict in loaded_topology_dicts:
        loaded_topology.append(
            Partial_Topology(
                chain=top_dict["chain"],
                fragment_sequence=top_dict["fragment_sequence"],
                residue_start=top_dict["residue_start"],
                residue_end=top_dict["residue_end"],
                peptide_trim=top_dict["peptide_trim"],
                fragment_index=top_dict["fragment_index"],
            )
        )

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

    return {
        "features_set": features_set,
        "loaded_features": loaded_features,
        "topology_set": topology_set,
        "loaded_topology": loaded_topology,
        "features_match": features_match,
        "topology_match": topology_match,
    }


if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    # Define trajectory and topology paths from the example
    open_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = "tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    topology_path = open_path

    # Create output directory
    output_dir = os.path.join(base_dir, "test_featurise_output")

    # Run the test
    results = test_featurise_save_load(
        trajectory_path=trajectory_path, topology_path=topology_path, output_dir=output_dir
    )

    # Report summary
    if results["features_match"] and results["topology_match"]:
        print("✅ Test passed: Features and topology correctly saved and loaded")
    else:
        print("❌ Test failed: Discrepancies detected")
        if not results["features_match"]:
            print("  - Features do not match")
        if not results["topology_match"]:
            print("  - Topology objects do not match")
