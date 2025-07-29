"""
This script runs featurises the sliced IsoValidation ensembles Iso-BI (filtered) and Iso-TRI (initial) using the JAXENT featurisation methods.
"""

import os
import time

import jax.numpy as jnp
import MDAnalysis as mda

from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config


def featurise_trajectory(
    trajectory_path, topology_path, output_dir, output_name, bv_config, featuriser_settings
):
    """
    Featurise a single trajectory and save the results.

    Args:
        trajectory_path (str): Path to the trajectory file
        topology_path (str): Path to the topology file
        output_dir (str): Directory to save output files
        output_name (str): Base name for output files (e.g., 'iso_tri', 'iso_bi')
        bv_config: BV model configuration
        featuriser_settings: Featuriser settings

    Returns:
        tuple: (features_path, topology_path) of saved files
    """
    print(f"Featurising trajectory: {trajectory_path}")

    # Create universe
    universe = mda.Universe(topology_path, trajectory_path)

    # Create ensemble
    ensemble = Experiment_Builder(
        universes=[universe],
        forward_models=[BV_model(bv_config)],
    )

    # Run featurisation
    features, feature_topology = run_featurise(
        ensemble=ensemble,
        config=featuriser_settings,
    )

    # Extract first element (since we have single universe/model)
    features, feature_topology = features[0], feature_topology[0]

    print(f"Featurised trajectory: {trajectory_path}")

    # Save features
    features_path = os.path.join(output_dir, f"features_{output_name}.npz")
    jnp.savez(
        features_path,
        heavy_contacts=features.heavy_contacts,
        acceptor_contacts=features.acceptor_contacts,
        k_ints=features.k_ints,
    )
    print(f"Saved {output_name} features to: {features_path}")

    # Save topology
    topology_save_path = os.path.join(output_dir, f"topology_{output_name}.json")
    Partial_Topology.save_list_to_json(
        feature_topology,
        topology_save_path,
    )
    print(f"Saved {output_name} topology to: {topology_save_path}")

    return features_path, topology_save_path


def main():
    # Define trajectories and topology
    tri_modal_trajectory = "sliced_trajectories/TeaA_filtered_sliced.xtc"
    bi_modal_trajectory = "sliced_trajectories/TeaA_initial_sliced.xtc"
    topology = "TeaA_ref_closed_state.pdb"  # TeaA_ref_open_state.pdb

    # Update traj_dir to correct relative path
    traj_dir = "../../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    output_dir = os.path.join(os.path.dirname(__file__), "_featurise")
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    if not os.path.exists(traj_dir):
        raise FileNotFoundError(f"Trajectory directory could not be found: {traj_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure BV model
    bv_config = BV_model_Config()
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0, 60.0, 120.0])

    # Configure featuriser
    featuriser_settings = FeaturiserSettings(name="ISO", batch_size=None)

    # Construct file paths
    top_path = os.path.join(traj_dir, topology)
    tri_path = os.path.join(traj_dir, tri_modal_trajectory)
    bi_path = os.path.join(traj_dir, bi_modal_trajectory)

    # Featurise trajectories
    trajectories_to_process = [(tri_path, "iso_tri"), (bi_path, "iso_bi")]

    for traj_path, output_name in trajectories_to_process:
        featurise_trajectory(
            trajectory_path=traj_path,
            topology_path=top_path,
            output_dir=output_dir,
            output_name=output_name,
            bv_config=bv_config,
            featuriser_settings=featuriser_settings,
        )


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Featurisation complete.")
    print(f"Elapsed time: {end - start:.2f} seconds")
