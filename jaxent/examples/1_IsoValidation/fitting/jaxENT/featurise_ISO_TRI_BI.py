"""
This script runs featurises the sliced IsoValidation ensembles Iso-BI (filtered) and Iso-TRI (initial) using the JAXENT featurisation methods.
"""

import os
import time

import jax.numpy as jnp
import MDAnalysis as mda
from jax import Array

from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config


def load_HDXer_kints(kint_path: str) -> tuple[Array, list[Partial_Topology]]:
    """
    Loads the intrinsic rates from a .dat file.
    Adjsuts the residue indices to be zero-based for termini exclusions

    Returns:
        kints: jax.numpy array of rates
        topology_list: list of Partial_Topology objects with resids
    """
    rates = []
    topology_list = []
    with open(kint_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            resid, rate = int(parts[0]), float(parts[1])
            rates.append(rate)
            # Assuming chain 'A' as a default for intrinsic rates if not specified in the file
            topology_list.append(Partial_Topology.from_single(chain="A", residue=resid - 1))
    kints = jnp.array(rates)
    return kints, topology_list


def featurise_trajectory(
    trajectory_path,
    topology_path,
    output_dir,
    output_name,
    bv_config,
    featuriser_settings,
    kint_data,
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
    # print lengths of features
    print(f"Heavy contacts length: {len(features.heavy_contacts)}")
    print(f"Acceptor contacts length: {len(features.acceptor_contacts)}")
    print(f"Kints length: {len(features.k_ints)}")
    # Save features

    if kint_data is not None:
        kints, topology_list = kint_data
        print(f"Loaded intrinsic rates: {kints}")
        print(f"Rates length: {kints.shape}")
        print(f"Topology list length: {len(topology_list)}")
        # filter kints to match the featrure topology
        _feature_top = Partial_Topology.merge(feature_topology)

        indices_to_remove = []

        for i, top in enumerate(topology_list):
            if not top.intersects(_feature_top):
                print(f"Removing kint {kints[i]}, {i} for topology {top}")
                indices_to_remove.append(i)
        _kints = jnp.delete(
            kints, jnp.array(indices_to_remove, dtype=int)
        )  # Remove indices from kints
        indices_to_remove = [i for i in indices_to_remove if i < features.heavy_contacts.shape[0]]

        _heavy_contacts = jnp.delete(
            features.heavy_contacts, jnp.array(indices_to_remove, dtype=int), axis=0
        )
        _acceptor_contacts = jnp.delete(
            features.acceptor_contacts, jnp.array(indices_to_remove, dtype=int), axis=0
        )

    else:
        _kints = features.k_ints
        _heavy_contacts = features.heavy_contacts
        _acceptor_contacts = features.acceptor_contacts
    print(f"Filtered kints length: {len(_kints)}")
    print(f"Filtered kint top length: {len(topology_list)}")
    print(f"Feature topology lengthL {len(feature_topology)}")
    # breakpoint()
    features_path = os.path.join(output_dir, f"features_{output_name}.npz")
    jnp.savez(
        features_path,
        heavy_contacts=_heavy_contacts,
        acceptor_contacts=_acceptor_contacts,
        k_ints=_kints,
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

    hdxer_kint_path = "../../data/out__train_TeaA_ISO_bi_1Intrinsic_rates.dat"
    hdxer_kint_path = os.path.join(os.path.dirname(__file__), hdxer_kint_path)
    if not os.path.exists(hdxer_kint_path):
        raise FileNotFoundError(f"HDXer kint file could not be found: {hdxer_kint_path}")

    # Load intrinsic rates from .dat file
    hdxer_kint_data = load_HDXer_kints(hdxer_kint_path)
    hdxer_kints = hdxer_kint_data[0]  # Extract kints from the tuple
    hdxer_top = hdxer_kint_data[1]  # Extract topology from the tuple

    print(f"Loaded intrinsic rates: {hdxer_kints.shape}")
    print(f"Loaded topology length: {len(hdxer_top)}")

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
            kint_data=hdxer_kint_data,
        )


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Featurisation complete.")
    print(f"Elapsed time: {end - start:.2f} seconds")
