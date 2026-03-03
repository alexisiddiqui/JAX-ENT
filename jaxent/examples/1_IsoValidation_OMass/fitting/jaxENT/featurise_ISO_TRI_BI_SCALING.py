"""
This script runs featurises the sliced IsoValidation ensembles Iso-BI (filtered) and Iso-TRI (initial) using the JAXENT featurisation methods.
"""

import os
import time
import re

import jax.numpy as jnp

from jaxent.examples.common.loading import load_HDXer_kints, featurise_trajectory
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.models.HDX.BV.forwardmodel import BV_model_Config


def main():
    # Define trajectories and topology
    tri_modal_trajectory = "sliced_trajectories/TeaA_filtered_sliced.xtc"
    bi_modal_trajectory = "sliced_trajectories/TeaA_initial_sliced.xtc"
    topology = "TeaA_ref_closed_state.pdb"  # TeaA_ref_open_state.pdb

    hdxer_kint_path = "../../data/out__train_TeaA_auto_VAL_1Intrinsic_rates.dat"
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
    output_root = os.path.join(os.path.dirname(__file__), "_featurise")
    traj_dir = os.path.join(os.path.dirname(__file__), traj_dir)

    if not os.path.exists(traj_dir):
        raise FileNotFoundError(f"Trajectory directory could not be found: {traj_dir}")

    # Ensure root output directory exists
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Configure BV model
    bv_config = BV_model_Config()
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0, 60.0, 120.0])

    # Configure featuriser
    featuriser_settings = FeaturiserSettings(name="ISO", batch_size=None)

    # Topology file lives in the trajectories dir
    top_path = os.path.join(traj_dir, "TeaA_ref_closed_state.pdb")
    if not os.path.exists(top_path):
        raise FileNotFoundError(f"Topology file could not be found: {top_path}")

    # Find all sliced_* subdirectories and sort numerically by suffix
    candidates = [d for d in os.listdir(traj_dir) if d.startswith("sliced_") and os.path.isdir(os.path.join(traj_dir, d))]
    def _slice_key(name):
        m = re.match(r"^sliced_(\d+)$", name)
        return int(m.group(1)) if m else float("inf")
    candidates = sorted(candidates, key=_slice_key)

    if not candidates:
        print("No sliced_* directories found; exiting.")
        return

    # Filter to only process specific slices
    # slices_to_process = [20,50,100,1000]
    slices_to_process = [20]
    candidates = [d for d in candidates if re.match(r"^sliced_(\d+)$", d) and int(re.match(r"^sliced_(\d+)$", d).group(1)) in slices_to_process]

    for slice_dir in candidates:
        slice_path = os.path.join(traj_dir, slice_dir)
        output_dir = os.path.join(output_root, slice_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        tri_path = os.path.join(slice_path, "TeaA_initial_sliced.xtc")
        bi_path = os.path.join(slice_path, "TeaA_filtered_sliced.xtc")

        # If either file missing, skip this slice
        missing = []
        if not os.path.exists(tri_path):
            missing.append(tri_path)
        if not os.path.exists(bi_path):
            missing.append(bi_path)
        if missing:
            print(f"Skipping {slice_dir}: missing files: {missing}")
            continue

        print(f"Processing slice {slice_dir} -> output: {output_dir}")

        trajectories_to_process = [(tri_path, "iso_tri"), (bi_path, "iso_bi")]

        for traj_path, output_name in trajectories_to_process:
            try:
                featurise_trajectory(
                    trajectory_path=traj_path,
                    topology_path=top_path,
                    output_dir=output_dir,
                    output_name=output_name,
                    bv_config=bv_config,
                    featuriser_settings=featuriser_settings,
                    kint_data=hdxer_kint_data,
                )
            except Exception as e:
                print(f"Error processing {traj_path} in {slice_dir}: {e}")
                # continue to next trajectory/slice
                continue


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print("Featurisation complete.")
    print(f"Elapsed time: {end - start:.2f} seconds")
