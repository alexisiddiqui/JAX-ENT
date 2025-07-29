"""
This script splits the data generated from the featurisation process into training and validation sets.
For now this is just a random split.
First we must load the dfracs and the segments files - we then create the HDX_peptide data object
we then load the feature topology and then use this with the data splitting to make the training and validation sets.
each training/validation set is then saved as data and a topology file so it can be rebuilt.
"""

import os

import numpy as np
import pandas as pd

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology import Partial_Topology


def save_split_data(data: list[HDX_peptide], split_dir: str, split_name: str) -> None:
    """
    Save split data (topology as JSON and dfrac as CSV)

    Args:
        data: list of HDX_peptide objects
        split_dir: Directory to save the split data
        split_name: Name prefix for the files (e.g., 'train', 'val')
    """
    # Extract topologies and dfrac data
    topologies: list[Partial_Topology] = [item.top for item in data]
    dfrac_data: list[list[float]] = [item.dfrac for item in data]

    # Stack dfrac data along 0th axis
    dfrac_array: np.ndarray = np.array(dfrac_data)  # Shape: (n_peptides, n_timepoints)

    # Save topology as JSON
    topology_file = os.path.join(split_dir, f"{split_name}_topology.json")
    Partial_Topology.save_list_to_json(topologies, topology_file)

    # Save dfrac data as CSV
    dfrac_file = os.path.join(split_dir, f"{split_name}_dfrac.csv")
    pd.DataFrame(dfrac_array).to_csv(dfrac_file, index=False, header=False)

    print(f"  {split_name}: {len(data)} samples saved")
    print(f"    Topology: {topology_file}")
    print(f"    Dfrac: {dfrac_file} (shape: {dfrac_array.shape})")


def run_data_splits(
    num_splits: int,
    output_dir: str,
    HDX_data: list[HDX_peptide],
    feature_topology: list[Partial_Topology],
) -> None:
    """
    Run multiple data splits and save each to its own folder

    Args:
        num_splits: Number of different splits to generate
        output_dir: Base directory to save all splits
        HDX_data: list of HDX_peptide objects
        feature_topology: list of Partial_Topology objects for features
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_idx in range(num_splits):
        print(f"\n=== Running split {split_idx + 1}/{num_splits} ===")

        # Create split-specific folder
        split_dir: str = os.path.join(output_dir, f"split_{split_idx:03d}")
        os.makedirs(split_dir, exist_ok=True)

        # Create dataloader and splitter with different random seed for each split
        HDX_dataloader: ExpD_Dataloader = ExpD_Dataloader(data=HDX_data)
        splitter: DataSplitter = DataSplitter(
            train_size=0.5,
            dataset=HDX_dataloader,
            common_residues=set(feature_topology),
            peptide_trim=1,
            centrality=False,
            check_trim=True,
            random_seed=42 * split_idx,  # Different seed for each split
        )

        train_data, val_data = splitter.random_split(remove_overlap=True)

        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(val_data)}")

        # Save training data
        save_split_data(train_data, split_dir, "train")

        # Save validation data
        save_split_data(val_data, split_dir, "val")

        print(f"Split {split_idx + 1} saved to {split_dir}")


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
    feature_topology: list[Partial_Topology] = Partial_Topology.load_list_from_json(
        filepath=os.path.join(features_dir, feature_topology_file)
    )
    print(f"Feature topology loaded with {len(feature_topology)} fragments.")

    # Load dfrac and segments data
    dfrac: np.ndarray = pd.read_csv(
        os.path.join(HDX_dir, dfrac_file),
        delim_whitespace=True,
        comment="#",
        header=None,
    ).to_numpy()

    segs: np.ndarray = pd.read_csv(
        os.path.join(HDX_dir, segs_file),
        delim_whitespace=True,
        comment="#",
        header=None,
    ).to_numpy()

    # Create HDX topology objects
    HDX_topology: list[Partial_Topology] = [
        Partial_Topology.from_range(
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
    save_split_data(HDX_data, output_dir, "full_dataset")

    # Run multiple data splits
    print(f"\nGenerating {num_splits} data splits...")
    run_data_splits(num_splits, output_dir, HDX_data, feature_topology)

    print(f"\nAll splits completed and saved to: {output_dir}")


if __name__ == "__main__":
    main()
