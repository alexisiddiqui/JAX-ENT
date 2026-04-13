"""
Generate sequence_cluster and spatial splits for 4 experimental conditions.
Uses HDX_protection_factor objects containing single-residue protection factors.
"""

import argparse
import logging
from pathlib import Path

import MDAnalysis as mda
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.examples.common.loading import run_data_splits
from jaxent.src.interfaces.topology import PTSerialiser

def parse_args():
    parser = argparse.ArgumentParser(description="Generate data splits for aSyn PF datasets.")
    
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    parser.add_argument(
        "--data_dir",
        default=SCRIPT_DIR / "../data/_aSyn",
        help="Directory containing the aSyn PF datasets (JSON/CSV pairs)",
    )
    parser.add_argument(
        "--output_dir",
        default=SCRIPT_DIR / "_datasplits",
        help="Directory to save the generated splits",
    )
    parser.add_argument(
        "--pdb_path",
        default=SCRIPT_DIR / "../data/_aSyn" / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb",
        help="Path to the reference PDB for spatial splits",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=3,
        help="Number of splits to generate per split type",
    )
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    
    data_dir = args.data_dir
    output_base_dir = args.output_dir
    pdb_path = args.pdb_path
    num_splits = args.num_splits
    
    conditions = ["Tris_only", "Extracellular", "Intracellular", "Lysosomal"]
    split_types = ["sequence_cluster", "spatial"]
    
    if not pdb_path.exists():
        logging.warning(f"PDB not found at {pdb_path}. Spatial splits will be skipped or fail.")
        universe = None
    else:
        universe = mda.Universe(str(pdb_path))
        
    for condition in conditions:
        safe_condition = condition.replace(" ", "_").replace("/", "_")
        logging.info(f"\n=== Processing Condition: {condition} ===")
        
        json_path = data_dir / f"aSyn_PF_{safe_condition}.json"
        csv_path = data_dir / f"aSyn_PF_{safe_condition}.csv"
        
        if not json_path.exists() or not csv_path.exists():
            logging.error(f"Missing data files for condition {condition}. Expected {json_path} and {csv_path}")
            continue
            
        logging.info(f"Loading datapoints from {json_path}")
        hdx_data = ExpD_Datapoint.load_list_from_files(
            json_path=json_path,
            csv_path=csv_path,
            datapoint_class=HDX_protection_factor
        )
        
        # Datapoint load handles topology internally to attach it to objects, 
        # but run_data_splits needs a parallel list of `feature_topology` objects to define common residues and splittables
        feature_topology = PTSerialiser.load_list_from_json(json_path)
        
        cond_output_dir = output_base_dir / safe_condition
        cond_output_dir.mkdir(parents=True, exist_ok=True)
        
        # For sequence_cluster with single-residue resolution, clusters of n_clusters=7 works well.
        # peptide_trim=0 for log_Pf as each object represents only one amino acid.
        for split_type in split_types:
            if split_type == "spatial" and universe is None:
                logging.info("Skipping spatial split due to missing Universe")
                continue
                
            split_out = cond_output_dir / split_type
            logging.info(f"Running {split_type} split. Output: {split_out}")
            
            run_data_splits(
                num_splits=num_splits,
                output_dir=str(split_out),
                hdx_data=hdx_data,
                feature_topology=feature_topology,
                split_type=split_type,
                remove_overlap=True,
                universe=universe,
                peptide_trim=0,
                min_split_size=4,
                n_clusters=7,
                plot=True,
                peptide_plot=False,
            )

    logging.info("\nAll data splits completed.")

if __name__ == "__main__":
    main()