import argparse
import pandas as pd
import re
from pathlib import Path
import sys

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create frame_ordering.csv from collected structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--structure_dir", 
        required=True, 
        help="Directory containing the collected structures (with subdirs apo_unminimised and liganded_ions)."
    )
    parser.add_argument(
        "--output_csv", 
        required=True, 
        help="Path to save the resulting frame_ordering.csv file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    structure_dir = Path(args.structure_dir).resolve()
    output_csv = Path(args.output_csv).resolve()

    if not structure_dir.is_dir():
        print(f"Error: structure_dir '{structure_dir}' does not exist.")
        sys.exit(1)

    # Regex to parse folder names: cluster_{Primary}_{Secondary}_ca{n_Ca}_cdz{n_CDZ}
    folder_pattern = re.compile(r"cluster_(?P<primary>\w+)_(?P<secondary>\w+)_ca(?P<ca>\d+)_cdz(?P<cdz>\d+)")

    collected_data = []
    
    # We iterate through the two main subdirectories
    subdirs = ["apo_protonated", "liganded_ions"]
    
    for subdir_name in subdirs:
        subdir_path = structure_dir / subdir_name
        if not subdir_path.is_dir():
            print(f"Warning: Subdirectory '{subdir_name}' not found in structure_dir.")
            continue
            
        # Iterate through cluster folders
        # Use sorted to ensure deterministic index assignment
        for cluster_folder in sorted(subdir_path.iterdir()):
            if not cluster_folder.is_dir():
                continue
                
            match = folder_pattern.match(cluster_folder.name)
            if not match:
                # Some folders might not match, e.g. .DS_Store or others
                continue
            
            meta = match.groupdict()
            
            # Find PDB files within the folder
            # For APO: prot_rank*.pdb
            # For liganded: complex_rank*_minimised.pdb (prefer minimised) or complex_rank*.pdb
            pdb_files = sorted(cluster_folder.glob("*.pdb"))
            
            for pdb_file in pdb_files:
                # dir_prefix should be relative to structure_dir
                # We include the subdir and folder name. 
                # If there are multiple PDBs, dir_prefix will be the same for all of them unless we include the filename.
                # However, usually frame_ordering uses 'index' to distinguish frames in a multi-frame file OR 
                # refers to the directory where the rank files are.
                # Given the header 'dir_prefix', it likely refers to the directory.
                # BUT if we have 5 files, and we want 5 rows, and we can't distinguish them by dir_prefix, 
                # then 'index' is the only discriminator.
                # Let's include the relative path INCLUDING the filename in 'dir_prefix' to be safe, 
                # as it's common in these scripts to use 'prefix' to mean 'path_to_file_without_ext' or 'path_to_file'.
                
                rel_path = pdb_file.relative_to(structure_dir)
                dir_prefix = str(rel_path.parent / rel_path.stem)
                
                collected_data.append({
                    "Primary_cluster": meta['primary'],
                    "Secondary_cluster": meta['secondary'],
                    "n_Ca": int(meta['ca']),
                    "n_CDZ_ligands": int(meta['cdz']),
                    "n_Ca_final": int(meta['ca']),
                    "n_CDZ_ligands_final": int(meta['cdz']),
                    "dir_prefix": dir_prefix
                })

    if not collected_data:
        print("No structures found. CSV not created.")
        return

    df = pd.DataFrame(collected_data)
    
    # Add 'index' column starting from 0
    df.insert(0, 'index', range(len(df)))
    
    # Header order as specified in docstring
    headers = [
        "index", "Primary_cluster", "Secondary_cluster", 
        "n_Ca", "n_CDZ_ligands", "n_Ca_final", "n_CDZ_ligands_final", 
        "dir_prefix"
    ]
    
    df[headers].to_csv(output_csv, index=False)
    print(f"Successfully wrote {len(df)} entries to {output_csv}")

if __name__ == "__main__":
    main()
