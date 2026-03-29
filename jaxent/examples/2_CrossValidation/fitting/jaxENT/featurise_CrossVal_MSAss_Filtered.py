"""
[Script Name] featurise_CrossVal_MSAss_Filtered.py

[Brief Description of Functionality]
Featurises the MoPrP trajectories (AF2_MSAss and AF2_filtered) using the Best-Vendruscolo (BV) model.
This process involves loading the trajectories, a reference topology, and calculating features
such as heavy atom contacts and H-bond acceptor contacts. It also integrates intrinsic
exchange rates from HDXer.

Requirements:
    - Input Trajectories:
        - `jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/clusters/all_clusters.xtc`
        - `jaxent/examples/2_CrossValidation/data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc`
    - Topology: `jaxent/examples/2_CrossValidation/data/MoPrP_max_plddt_4334.pdb` (TeaA_ref_open_state.pdb)
    - HDXer Intrinsic Rates: `_MoPrP/_output/out__train_MoPrP_af_clean_1Intrinsic_rates.dat`
    - JAX-ENT library installed.

Usage:
    python jaxent/examples/2_CrossValidation/fitting/jaxENT/featurise_CrossVal_MSAss_Filtered.py

Output:
    - Featurised data (.npz) and topology (.json) files in `jaxent/examples/2_CrossValidation/fitting/jaxENT/_featurise/`.
      Files: `features_AF2_MSAss.npz`, `topology_AF2_MSAss.json`, `features_AF2_filtered.npz`, `topology_AF2_filtered.json`.
"""

import os
import time

import jax.numpy as jnp

from jaxent.examples.common.loading import load_HDXer_kints, featurise_trajectory
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.models.HDX.BV.forwardmodel import BV_model_Config


def main():
    # Define trajectories and topology
    af2_filtered_trajectory = "_cluster_MoPrP_filtered/clusters/all_clusters.xtc"
    af2_MSAss_trajectory = "_cluster_MoPrP/clusters/all_clusters.xtc"
    topology = "MoPrP_max_plddt_4334.pdb"  # TeaA_ref_open_state.pdb
    data_dir = "../../data/"

    hdxer_kint_path = "../../data/_MoPrP/_output/out__train_MoPrP_af_clean_1Intrinsic_rates.dat"
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
    output_dir = os.path.join(os.path.dirname(__file__), "_featurise")
    data_dir = os.path.join(os.path.dirname(__file__), data_dir)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Trajectory directory could not be found: {data_dir}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configure BV model
    bv_config = BV_model_Config(switch=True)
    # times 0.08	0.33	0.67	1.00	5.00	10.00	20.00	30.00	45.00	60.00	160.00	240.00	390.00	750.00	1440.00 /min
    bv_config.timepoints = jnp.array(
        [
            0.08,
            0.33,
            0.67,
            1.00,
            5.00,
            10.00,
            20.00,
            30.00,
            45.00,
            60.00,
            160.00,
            240.00,
            390.00,
            750.00,
            1440.00,
        ]
    )  # in minutes

    # Configure featuriser
    featuriser_settings = FeaturiserSettings(name="CrossVal", batch_size=None)

    # Construct file paths
    top_path = os.path.join(data_dir, topology)
    af2_filtered_path = os.path.join(data_dir, af2_filtered_trajectory)
    af2_MSAss_path = os.path.join(data_dir, af2_MSAss_trajectory)

    trajectories_to_process = [(af2_MSAss_path, "AF2_MSAss"), (af2_filtered_path, "AF2_filtered")]

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
