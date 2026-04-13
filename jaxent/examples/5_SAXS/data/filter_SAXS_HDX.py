"""
Script that uses the ordering CSV to filter the jaxENT features data based on the headers:

index,Primary_cluster,Secondary_cluster,n_Ca,n_CDZ_ligands,n_Ca_final,n_CDZ_ligands_final,dir_prefix


Script Args:
--SAXS_features_path: path to the SAXS features data
--HDX_features_path: path to the HDX features data
--ordering_csv: path to the ordering CSV
--SAXS_output_dir: directory to save the filtered SAXS data
--HDX_output_dir: directory to save the filtered HDX data
--Primary_cluster_list: list of primary clusters (default: 0,1,n)
--Secondary_cluster_list: list of secondary clusters (default: 0,1,2,3,4)
--n_Ca_final_list: list of n_Ca_final values (default: 0,1,2,3,4)
--n_CDZ_ligands_final_list: list of n_CDZ_ligands_final values (default: 0,1,2)


For example:
Filtering
--n_Ca_final_list 1,2,3,4 --n_CDZ_ligands_final_list 0,1

This would filter the data to only include rows where n_Ca_final is 1,2,3,4 and n_CDZ_ligands_final is 0,1



python jaxent/examples/5_SAXS/data/filter_SAXS_HDX.py \
    --SAXS_features_path jaxent/examples/5_SAXS/data/_SAXS_features/SAXS_curve_input_features.npz \
    --HDX_features_path jaxent/examples/5_SAXS/data/_HDX_features/BV_features.npz \
    --ordering_csv jaxent/examples/5_SAXS/ensemble_generation/neuralplexer/collected_structures/frame_ordering.csv \
    --SAXS_output_dir jaxent/examples/5_SAXS/data/_filtered_SAXS_features \
    --HDX_output_dir jaxent/examples/5_SAXS/data/_filtered_HDX_features \
    --Primary_cluster_list 0,1,n \
    --Secondary_cluster_list 0,1,2,3,4 \
    --n_Ca_final_list 1,2,3,4 \
    --n_CDZ_ligands_final_list 0,1


"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.models.HDX.BV.features import BV_input_features
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Filter SAXS and HDX features based on ordering CSV.")
    parser.add_argument("--SAXS_features_path", type=str, help="Path to the SAXS features data (.npz)")
    parser.add_argument("--HDX_features_path", type=str, help="Path to the HDX features data (.npz)")
    parser.add_argument("--ordering_csv", required=True, type=str, help="Path to the ordering CSV")
    parser.add_argument("--SAXS_output_dir", type=str, help="Directory to save the filtered SAXS data")
    parser.add_argument("--HDX_output_dir", type=str, help="Directory to save the filtered HDX data")
    parser.add_argument("--Primary_cluster_list", type=str, default="0,1,n", help="List of primary clusters (default: 0,1,n)")
    parser.add_argument("--Secondary_cluster_list", type=str, default="0,1,2,3,4", help="List of secondary clusters (default: 0,1,2,3,4)")
    parser.add_argument("--n_Ca_final_list", type=str, default="0,1,2,3,4", help="List of n_Ca_final values (default: 0,1,2,3,4)")
    parser.add_argument("--n_CDZ_ligands_final_list", type=str, default="0,1,2", help="List of n_CDZ_ligands_final values (default: 0,1,2)")
    return parser.parse_args()

def filter_csv(args):
    df = pd.read_csv(args.ordering_csv)
    
    primary_clusters = [x.strip() for x in args.Primary_cluster_list.split(",")]
    secondary_clusters = [int(x.strip()) for x in args.Secondary_cluster_list.split(",")]
    n_ca_final_list = [int(x.strip()) for x in args.n_Ca_final_list.split(",")]
    n_cdz_list = [int(x.strip()) for x in args.n_CDZ_ligands_final_list.split(",")]
    
    # Primary cluster can be numeric or 'n'
    # We need to handle it carefully if it's mixed
    # Let's convert the column to string for filtering
    df['Primary_cluster'] = df['Primary_cluster'].astype(str)
    
    mask = (
        df['Primary_cluster'].isin(primary_clusters) &
        df['Secondary_cluster'].isin(secondary_clusters) &
        df['n_Ca_final'].isin(n_ca_final_list) &
        df['n_CDZ_ligands_final'].isin(n_cdz_list)
    )
    
    filtered_df = df[mask].copy()
    print(f"Filtered CSV: {len(df)} -> {len(filtered_df)} frames")
    return filtered_df

def filter_saxs(args, indices):
    if not args.SAXS_features_path or not args.SAXS_output_dir:
        print("Skipping SAXS filtering (path or output dir not provided)")
        return
    
    saxs_path = Path(args.SAXS_features_path)
    if not saxs_path.exists():
        print(f"SAXS features not found at {saxs_path}")
        return

    print(f"\nFiltering SAXS features from {saxs_path}...")
    features = SAXS_curve_input_features.load(str(saxs_path))
    
    # intensities shape is (n_q, n_frames)
    new_intensities = features.intensities[:, indices]
    
    new_features = SAXS_curve_input_features(intensities=new_intensities)
    
    out_dir = Path(args.SAXS_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    new_features.save(str(out_dir / "SAXS_curve_input_features.npz"))
    print(f"Saved filtered SAXS features (shape: {new_intensities.shape}) to {out_dir}")
    
    # Copy topology if exists
    topo_path = saxs_path.parent / "topology.json"
    if topo_path.exists():
        shutil.copy(topo_path, out_dir / "topology.json")
        print(f"Copied topology to {out_dir}")

def filter_hdx(args, indices):
    if not args.HDX_features_path or not args.HDX_output_dir:
        print("Skipping HDX filtering (path or output dir not provided)")
        return
    
    hdx_path = Path(args.HDX_features_path)
    if not hdx_path.exists():
        print(f"HDX features not found at {hdx_path}")
        return

    print(f"\nFiltering HDX features from {hdx_path}...")
    features = BV_input_features.load(str(hdx_path))
    
    # heavy_contacts shape is (n_res, n_frames)
    new_heavy = features.heavy_contacts[:, indices]
    new_acceptor = features.acceptor_contacts[:, indices]
    
    new_kints = None
    if features.k_ints is not None:
        # Check if k_ints is (n_res, n_frames) or (n_res,)
        if features.k_ints.ndim == 2 and features.k_ints.shape[1] > 1:
             new_kints = features.k_ints[:, indices]
        else:
             new_kints = features.k_ints

    new_features = BV_input_features(
        heavy_contacts=new_heavy,
        acceptor_contacts=new_acceptor,
        k_ints=new_kints
    )
    
    out_dir = Path(args.HDX_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Output name might be different, but let's stick to input filename
    out_name = hdx_path.name
    new_features.save(str(out_dir / out_name))
    print(f"Saved filtered HDX features (heavy contacts shape: {new_heavy.shape}) to {out_dir}")
    
    # Copy topology if exists
    # featurise_HDX-MS.py saves as topology_{output_name}.json
    # Let's look for any topology*.json in the input directory and copy them
    for topo_file in hdx_path.parent.glob("topology*.json"):
        shutil.copy(topo_file, out_dir / topo_file.name)
        print(f"Copied {topo_file.name} to {out_dir}")

def main():
    args = parse_args()
    filtered_df = filter_csv(args)
    
    if filtered_df.empty:
        print("No frames matched the filter. Exiting.")
        return
    
    # Extract original indices (0-based) for filtering the arrays
    # Assuming the 'index' column in CSV matches the array index
    indices = filtered_df['index'].values
    
    filter_saxs(args, indices)
    filter_hdx(args, indices)
    
    # Also save the filtered CSV for record
    if args.SAXS_output_dir:
        filtered_df.to_csv(Path(args.SAXS_output_dir) / "frame_ordering_filtered.csv", index=False)
    if args.HDX_output_dir:
        filtered_df.to_csv(Path(args.HDX_output_dir) / "frame_ordering_filtered.csv", index=False)

if __name__ == "__main__":
    main()