"""

python jaxent/examples/5_SAXS/data/combine_SAXS.py \
    --input_dir jaxent/examples/5_SAXS/ensemble_generation/neuralplexer/collected_saxs \
    --output_dir jaxent/examples/5_SAXS/data/_SAXS_features \
    --ordering_csv jaxent/examples/5_SAXS/ensemble_generation/neuralplexer/collected_structures/frame_ordering.csv


"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import jax.numpy as jnp
import MDAnalysis as mda
import sys
from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.interfaces.topology import PTSerialiser, Partial_Topology, mda_TopologyAdapter

def parse_args():
    parser = argparse.ArgumentParser(description="Combine SAXS .dat curves into JAX-ENT features.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the .dat files (collected_saxs).")
    parser.add_argument("--output_dir", required=True, help="Directory to save the combined features.")
    parser.add_argument("--ordering_csv", required=True, help="Path to the frame_ordering.csv file.")
    return parser.parse_args()

def load_dat(path: Path) -> np.ndarray:
    """Load intensities from a SAXS .dat file (assumes q, I(q), error columns)."""
    dat = np.loadtxt(path, comments="#")
    return dat[:, 2]  # Column 2 is I(q) (column 1 is the experimental)

def _extract_chain_topologies(pdb_path: Path) -> list[Partial_Topology]:
    """Extract chain-level topologies from a PDB."""
    universe = mda.Universe(str(pdb_path))
    topologies = mda_TopologyAdapter.from_mda_universe(
        universe=universe,
        mode="chain",
        include_selection="protein",
        exclude_termini=False,
        renumber_residues=True,
    )
    for idx, topology in enumerate(topologies):
        topology.fragment_index = idx
    return topologies

def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ordering_csv = Path(args.ordering_csv).expanduser().resolve()
    
    topology_pdb = Path("/Users/alexi/JAX-ENT/jaxent/examples/5_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_max_plddt_425.pdb")

    if not ordering_csv.exists():
        print(f"Error: ordering_csv '{ordering_csv}' not found.")
        sys.exit(1)
        
    df = pd.read_csv(ordering_csv)
    
    all_intensities = []
    
    print(f"Loading {len(df)} SAXS curves from {input_dir}...")
    
    for _, row in df.iterrows():
        # dir_prefix in frame_ordering.csv is relative path to file stem
        dat_path = input_dir / f"{row['dir_prefix']}.dat"
        if not dat_path.exists():
            print(f"Error: .dat file not found: {dat_path}")
            sys.exit(1)
        
        intensities = load_dat(dat_path)
        all_intensities.append(intensities)
    
    # SAXS_curve_input_features.intensities expects shape (n_q, n_frames)
    matrix = np.stack(all_intensities).T
    
    features = SAXS_curve_input_features(intensities=jnp.asarray(matrix))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    features_output = output_dir / "SAXS_curve_input_features.npz"
    features.save(str(features_output))
    
    # Save topology
    if topology_pdb.exists():
        topologies = _extract_chain_topologies(topology_pdb)
        topology_output = output_dir / "topology.json"
        PTSerialiser.save_list_to_json(topologies, topology_output)
        print(f"Saved topology to: {topology_output}")
    else:
        print(f"Warning: Topology PDB not found at {topology_pdb}")

    print(f"Successfully combined {matrix.shape[1]} frames with {matrix.shape[0]} q-points.")
    print(f"Saved features to: {features_output}")

if __name__ == "__main__":
    main()