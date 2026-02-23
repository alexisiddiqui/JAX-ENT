"""
splitdata_ISO.py

Splits the HDX data into training and validation sets using various strategies (Random, Sequence, Spatial, etc.).
First we must load the dfracs and the segments files - we then create the HDX_peptide data object
we then load the feature topology and then use this with the data splitting to make the training and validation sets.
each training/validation set is then saved as data and a topology file so it can be rebuilt.

Requirements:
    - Featurized data and topology from featurise_ISO_TRI_BI.py

Usage:
    python jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/splitdata_ISO.py

Output:
    - Split data directories in fitting/jaxENT/_datasplits/ (e.g., fit_ISO_BI_split_random_0)
"""

import os

import MDAnalysis as mda
import numpy as np
import pandas as pd

import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.examples.common.loading import run_data_splits


def main() -> None:
    HDX_dir: str = "../../data/_output"
    HDX_dir = os.path.join(os.path.dirname(__file__), HDX_dir)
    if not os.path.exists(HDX_dir):
        raise FileNotFoundError(f"HDX directory does not exist: {HDX_dir}")

    dfrac_file: str = "mixed_60-40_artificial_expt_resfracs_TeaA_dfrac.dat"
    segs_file: str = "mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"
    output_dir: str = os.path.join(os.path.dirname(__file__), "_datasplits")
    feature_topology_file: str = "topology_iso_bi.json"
    features_dir: str = os.path.join(os.path.dirname(__file__), "_featurise")

    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory does not exist: {features_dir}")

    num_splits: int = 3
    chain: str = "A"

    # Load feature topology
    feature_topology: list[pt.Partial_Topology] = pt.PTSerialiser.load_list_from_json(
        filepath=os.path.join(features_dir, feature_topology_file)
    )
    print(f"Feature topology loaded with {len(feature_topology)} fragments.")

    # Load dfrac and segments data
    dfrac: np.ndarray = pd.read_csv(
        os.path.join(HDX_dir, dfrac_file),
        sep=r"\s+",
        comment="#",
        header=None,
    ).to_numpy()

    segs: np.ndarray = pd.read_csv(
        os.path.join(HDX_dir, segs_file),
        sep=r"\s+",
        comment="#",
        header=None,
    ).to_numpy()

    # Create HDX topology objects
    HDX_topology: list[pt.Partial_Topology] = [
        pt.TopologyFactory.from_range(
            chain=chain,
            start=seg[0],
            end=seg[1],
            fragment_index=idx,
            peptide=True,
            peptide_trim=1,
            fragment_name="TeaISO",
        )
        for idx, seg in enumerate(segs)
    ]
    print(f"Loaded {len(HDX_topology)} segments from {segs_file}.")

    # Create HDX data objects
    HDX_data: list[HDX_peptide] = [
        HDX_peptide(dfrac=dfrac[idx], top=HDX_topology[idx]) for idx in range(len(segs))
    ]
    print(f"Created {len(HDX_data)} HDX_peptide objects.")

    # Save the full loaded dataset
    print(f"\nSaving full dataset to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    ExpD_Datapoint.save_list_to_files(
        HDX_data,
        json_path=os.path.join(output_dir, "full_dataset_topology.json"),
        csv_path=os.path.join(output_dir, "full_dataset_dfrac.csv"),
    )

    # Run multiple data splits for each split type
    split_types = ["random", "sequence", "sequence_cluster", "stratified", "spatial"]
    remove_overlap = True  # Or configure as needed

    # Define the path to the closed state topology and a representative trajectory
    # These paths are relative to the 'splitdata_ISO.py' script
    traj_base_dir = os.path.join(
        os.path.dirname(__file__), "../../data/_Bradshaw/Reproducibility_pack_v2/data/trajectories"
    )
    closed_topology_file = os.path.join(traj_base_dir, "TeaA_ref_closed_state.pdb")
    # Using the filtered sliced trajectory as a representative for spatial calculations
    representative_trajectory_file = os.path.join(
        traj_base_dir, "sliced_trajectories/TeaA_filtered_sliced.xtc"
    )

    try:
        universe = mda.Universe(closed_topology_file)
        print(
            f"Loaded MDAnalysis Universe from {closed_topology_file} and {representative_trajectory_file}"
        )
    except Exception as e:
        print(f"Warning: Could not load MDAnalysis Universe for spatial split: {e}")
        universe = None  # Set to None if loading fails

    for split_type in split_types:
        split_output_dir = os.path.join(output_dir, split_type)
        print(f"\n--- Generating {num_splits} splits for type: {split_type} ---")
        if split_type == "spatial" and universe is None:
            print("Skipping spatial split as MDAnalysis Universe could not be loaded.")
            continue
        run_data_splits(
            num_splits=num_splits,
            output_dir=split_output_dir,
            hdx_data=HDX_data,
            feature_topology=feature_topology,
            split_type=split_type,
            remove_overlap=remove_overlap,
            universe=universe,  # Pass universe to run_data_splits
        )

    print(f"\nAll splits completed and saved to: {output_dir}")


if __name__ == "__main__":
    main()
