"""
Split HDX-MS uptake peptide data 

Loads an experimental HDX-MS.csv file 
Protein,Start,End,Sequence,MaxUptake,Max Uptake 80% D20,State,Exposure Time (min),Uptake (Da),Uptake SD (Da)

Fractional uptake = Uptake (Da) / Max Uptake 80% (Da)

Plots a heatmap of the fractional uptake for each peptide (ordered by start/end residues per peptide)

Save a formated HDX-MS _dfrac.csv file with columns: (wide format) 
#  timepoints*
and a segs file with columns:
#  start, end

- this is so that covariance matrices can be computed by
jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py

Splits across train/val using built-in DataSplitter whole-system strategies
(random, sequence-cluster, spatial),
saves .npz splits for downstream fitting, and produces publication figures.
"""

import argparse
import warnings
from pathlib import Path

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LogNorm

import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.examples.common.loading import run_data_splits


def load_hdx_csv(path: Path, chain: str, fragment_name: str) -> tuple[list[HDX_peptide], list[float]]:
    df = pd.read_csv(path, encoding="utf-8-sig")

    if 'Max Uptake 80% D20' not in df.columns:
        raise ValueError(f"Required column 'Max Uptake 80% D20' not found in {path}")
        
    if df['Max Uptake 80% D20'].le(0).any():
        raise ValueError("Found non-positive values in 'Max Uptake 80% D20'. Must be strictly positive.")
        
    if df.duplicated(subset=['Start', 'End', 'Exposure Time (min)']).any():
        raise ValueError("Found duplicate rows for (Start, End, Exposure Time). Check input dataset.")

    df['dfrac'] = df['Uptake (Da)'] / df['Max Uptake 80% D20']
    timepoints = sorted(df['Exposure Time (min)'].unique())

    # Pivot table to wide format
    pivot_df = df.pivot_table(
        index=['Start', 'End', 'Sequence'], 
        columns='Exposure Time (min)', 
        values='dfrac'
    ).reset_index()

    # Drop rows with any NaN timepoints
    initial_len = len(pivot_df)
    pivot_df = pivot_df.dropna()
    if len(pivot_df) < initial_len:
        warnings.warn(f"Dropped {initial_len - len(pivot_df)} peptides due to missing timepoints.")

    # Sort by Start and End
    pivot_df = pivot_df.sort_values(by=['Start', 'End']).reset_index(drop=True)

    hdx_data = []
    
    for idx, row in pivot_df.iterrows():
        top = pt.TopologyFactory.from_range(
            chain=chain,
            start=int(row['Start']),
            end=int(row['End']),
            fragment_sequence=str(row['Sequence']),
            fragment_name=fragment_name,
            fragment_index=idx,
            peptide=True,
            peptide_trim=2,
        )
        
        dfrac_vals = [float(row[t]) for t in timepoints]
        hdx_data.append(HDX_peptide(dfrac=dfrac_vals, top=top))

    return hdx_data, timepoints


def save_legacy_format(hdx_data: list[HDX_peptide], timepoints: list[float], name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dfrac.dat
    with open(output_dir / f"{name}_dfrac.dat", "w") as f:
        header = "#\t" + "\t".join(f"{t:.3f}" for t in timepoints) + "\t times/min\n"
        f.write(header)
        for dp in hdx_data:
            f.write("\t".join(f"{v:.5f}" for v in dp.dfrac) + "\n")

    # Save segs.txt
    with open(output_dir / f"{name}_segs.txt", "w") as f:
        for dp in hdx_data:
            f.write(f"{dp.top.residues[0]} {dp.top.residues[-1]}\n")


def plot_heatmap(hdx_data: list[HDX_peptide], timepoints: list[float], name: str, output_dir: Path):
    matrix = np.array([dp.dfrac for dp in hdx_data])
    labels = [f"{dp.top.residues[0]}-{dp.top.residues[-1]}" for dp in hdx_data]

    rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(6, max(4, len(hdx_data) * 0.15)))

    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(timepoints)))
    ax.set_xticklabels([f"{t:.2f}" for t in timepoints])

    ax.set_title(f"HDX Fractional Uptake: {name}")
    ax.set_xlabel("Exposure Time (min)")
    ax.set_ylabel("Peptide (Start-End)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fractional Uptake")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "fractional_uptake_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_and_save_sigma(hdx_data: list[HDX_peptide], name: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    dfrac_values = np.array([dp.dfrac for dp in hdx_data])
    # shape: (n_peptides, n_timepoints)

    Sigma = np.cov(dfrac_values) + np.diag(np.full(dfrac_values.shape[0], 1e-6))
    Sigma_inv = np.linalg.inv(Sigma)

    np.savez(output_dir / "Sigma.npz", Sigma=Sigma, Sigma_inv=Sigma_inv)

    # Heatmaps
    for mat, mat_name, log_scale, cmap in [
        (Sigma, "Sigma", False, "viridis"),
        (Sigma_inv, "Sigma_inv", True, "magma"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        norm = LogNorm(vmin=max(mat.min(), 1e-12), vmax=mat.max()) if log_scale else None
        sns.heatmap(mat, ax=ax, cmap=cmap, norm=norm, xticklabels=False, yticklabels=False)
        ax.set_title(f"{mat_name} — {name}")
        fig.tight_layout()
        fig.savefig(output_dir / f"{mat_name}_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Diagonal bar charts
    for mat, mat_name, log_scale in [(Sigma, "Sigma", False), (Sigma_inv, "Sigma_inv", True)]:
        diag = np.diag(mat)
        fig, ax = plt.subplots(figsize=(max(4, len(diag) * 0.1), 4))
        ax.bar(np.arange(len(diag)), diag)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(f"{mat_name} diagonal — {name}")
        ax.set_xlabel("Peptide index")
        ax.set_ylabel("Variance")
        fig.tight_layout()
        fig.savefig(output_dir / f"{mat_name}_diagonal_bar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Split HDX-MS uptake peptide data")
    parser.add_argument("--hdx-data", type=Path, required=True, help="SASBDB HDX-MS CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root output dir")
    parser.add_argument("--feature-topology", type=Path, required=True, help="_HDX_features/topology_BV_features.json")
    parser.add_argument("--name", type=str, default=None, help="Plot label (default: stem of hdx-data)")
    parser.add_argument("--chain", type=str, default="A", help="Chain ID")
    parser.add_argument("--n-splits", type=int, default=3, help="Replicates per split type")
    parser.add_argument("--train-size", type=float, default=0.5, help="Fraction for train split")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--structure", type=Path, default=None, help="PDB for spatial split")

    args = parser.parse_args()
    
    name = args.name or args.hdx_data.stem

    feature_topology = pt.PTSerialiser.load_list_from_json(str(args.feature_topology))
    
    hdx_data, timepoints = load_hdx_csv(args.hdx_data, args.chain, name)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    ExpD_Datapoint.save_list_to_files(
        hdx_data, 
        json_path=str(args.output_dir / "full_dataset_topology.json"), 
        csv_path=str(args.output_dir / "full_dataset_dfrac.csv")
    )
    
    save_legacy_format(hdx_data, timepoints, name, args.output_dir)
    plot_heatmap(hdx_data, timepoints, name, args.output_dir)
    compute_and_save_sigma(hdx_data, name, args.output_dir / "_covariance_matrices")

    universe = None
    if args.structure:
        try:
            universe = mda.Universe(str(args.structure))
        except Exception as e:
            print(f"Warning: Could not load MDAnalysis Universe for spatial split: {e}")
            universe = None

    split_types = ["random", "sequence", "sequence_cluster", "stratified", "spatial"]
    for split_type in split_types:
        if split_type == "spatial" and universe is None:
            continue
            
        run_data_splits(
            num_splits=args.n_splits, 
            output_dir=str(args.output_dir / split_type),
            hdx_data=hdx_data, 
            feature_topology=feature_topology,
            split_type=split_type, 
            remove_overlap=True, 
            universe=universe,
            peptide_trim=2, 
            min_split_size=4, 
            n_clusters=7
        )

if __name__ == "__main__":
    main()
