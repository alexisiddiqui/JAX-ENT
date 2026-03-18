"""
Clusters trajectory based on RMSD to reference structures.

Performs an Calpha aligned RMSD (filtering for common residues) between trajectory and reference structures.

Similar to the IsoValidation clustering script jaxent/examples/1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py
- generates a clustering_assignments.csv

Example output of this script can be found in jaxent/examples/4_SAXS/data/_RMSD_cluster_output/

Script args:
- trajectory_path: path to the trajectory file (jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_max_plddt_425.pdb)
- topology_path: path to the topology file (jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_plddt_ordered.xtc)
- reference_paths: list of paths to the reference structures (jaxent/examples/4_SAXS/FOXS/missing_residues/7PSZ_apo.pdb, jaxent/examples/4_SAXS/FOXS/missing_residues/1CLL_nosol.pdb)
- output_path: path to the output directory (jaxent/examples/4_SAXS/data/_RMSD_cluster_output/)
- rmsd_threshold: threshold for unassigned clusters (default 5.0)

"""

import argparse
import os

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_rmsd_to_references(topology_path, trajectory_path, reference_paths):
    """
    Compute RMSD of trajectory frames to reference structures.

    Uses C-alpha residues that are common across the mobile trajectory AND
    all reference structures (global intersection) so that every alignment
    is performed on an identical residue set.

    Args:
        topology_path (str): Path to topology file
        trajectory_path (str): Path to trajectory file
        reference_paths (list): List of paths to reference structures

    Returns:
        np.ndarray: RMSD values (n_frames, n_refs)
    """
    print(f"Loading mobile universe: {topology_path}, {trajectory_path}")
    mobile = mda.Universe(topology_path, trajectory_path)
    n_frames = len(mobile.trajectory)
    n_refs = len(reference_paths)

    # Build 1-per-resid CA dicts (last atom wins if altLocs present)
    mobile_ca = mobile.select_atoms("name CA")
    m_dict = {a.resid: a for a in mobile_ca}

    # Global intersection: common resids across mobile AND every reference
    common_resids = set(m_dict.keys())
    ref_dicts = []
    for ref_path in reference_paths:
        ref = mda.Universe(ref_path)
        r_ca = ref.select_atoms("name CA")
        r_dict = {a.resid: a for a in r_ca}
        common_resids &= set(r_dict.keys())
        ref_dicts.append((ref_path, ref, r_dict))

    common_strict = sorted(common_resids)
    if not common_strict:
        raise ValueError("No C-alpha residues are common across all references and the mobile trajectory.")

    m_indices = [m_dict[r].ix for r in common_strict]
    mobile_sel = mobile.atoms[m_indices]
    print(f"Common CA residues (intersection across all refs): {len(common_strict)}")

    rmsd_values = np.zeros((n_frames, n_refs))

    for j, (ref_path, ref, r_dict) in enumerate(ref_dicts):
        ref_name = os.path.basename(ref_path)
        print(f"  Aligning against reference {j}: {ref_name}")

        r_indices = [r_dict[r].ix for r in common_strict]
        ref_sel = ref.atoms[r_indices]

        # Use results.rmsd (R.rmsd deprecated in MDAnalysis 3.0)
        R = rms.RMSD(mobile_sel, ref_sel, superposition=True)
        R.run()
        rmsd_values[:, j] = R.results.rmsd[:, 2]

    return rmsd_values


def cluster_by_rmsd(rmsd_values, rmsd_threshold=5.0):
    """
    Cluster frames based on RMSD to reference structures.

    Args:
        rmsd_values (np.ndarray): RMSD values to reference structures (n_frames, n_refs)
        rmsd_threshold (float): RMSD threshold for clustering. If min RMSD is above this, unassigned (-1)

    Returns:
        np.ndarray: Cluster assignments
    """
    # Simple clustering: assign to closest reference
    cluster_assignments = np.argmin(rmsd_values, axis=1)

    # Check threshold bounds
    min_rmsd = np.min(rmsd_values, axis=1)
    valid_clusters = min_rmsd <= rmsd_threshold

    # Set invalid to -1
    cluster_assignments[~valid_clusters] = -1

    return cluster_assignments

def save_results(rmsd_values, cluster_ids, output_dir, reference_paths):
    """
    Saves outputs as cluster_assignments.csv and min_rmsd
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save assignments
    data = {
        "frame": np.arange(len(cluster_ids)),
        "cluster_id": cluster_ids,
    }
    
    for j in range(rmsd_values.shape[1]):
        data[f"rmsd_ref_{j}"] = rmsd_values[:, j]
        
    data["min_rmsd"] = np.min(rmsd_values, axis=1)
    
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "cluster_assignments.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved assignments to {csv_path}")

# Palette: clusters get distinct colours; unassigned gets a neutral grey
_CLUSTER_COLOURS = [
    "#1f77b4",  # cluster 0 — steel blue
    "#ff7f0e",  # cluster 1 — orange
    "#2ca02c",  # cluster 2 — green
    "#d62728",  # cluster 3 — red
    "#9467bd",  # cluster 4 — purple
    "#8c564b",  # cluster 5 — brown
]
_UNASSIGNED_COLOUR = "#aaaaaa"


def _ref_label(ref_path, j):
    """Short display name for a reference: stem without extension."""
    stem = os.path.splitext(os.path.basename(ref_path))[0]
    return stem if stem else f"Ref {j}"


def plot_results(rmsd_values, cluster_ids, output_dir, reference_paths=None, rmsd_threshold=None):
    """
    Generates publication-quality summary plots.

    rmsd_scatter.png  — 2-D scatter (ref_0 vs ref_1), log scale, threshold lines.
    rmsd_histograms.png — histograms per reference with a vertical threshold line.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_refs = rmsd_values.shape[1]
    ref_labels = [_ref_label(p, j) for j, p in enumerate(reference_paths or [])]
    if len(ref_labels) < n_refs:
        ref_labels += [f"Ref {j}" for j in range(len(ref_labels), n_refs)]

    # ---------------------------------------------------------------- scatter
    if n_refs >= 2:
        fig, ax = plt.subplots(figsize=(7, 6))

        unassigned = cluster_ids == -1
        ax.scatter(
            rmsd_values[unassigned, 0], rmsd_values[unassigned, 1],
            c=_UNASSIGNED_COLOUR, alpha=0.30, s=8, linewidths=0,
            label="Unassigned", rasterized=True,
        )

        for c in np.unique(cluster_ids[cluster_ids >= 0]):
            mask = cluster_ids == c
            colour = _CLUSTER_COLOURS[int(c) % len(_CLUSTER_COLOURS)]
            ax.scatter(
                rmsd_values[mask, 0], rmsd_values[mask, 1],
                c=colour, alpha=0.65, s=12, linewidths=0,
                label=f"Cluster {int(c)} — {ref_labels[int(c)]}",
                rasterized=True,
            )

        # Threshold lines
        if rmsd_threshold is not None:
            ax.axvline(rmsd_threshold, color="#333333", linewidth=1.2, linestyle="--",
                       label=f"Threshold {rmsd_threshold} Å")
            ax.axhline(rmsd_threshold, color="#333333", linewidth=1.2, linestyle="--")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"RMSD to {ref_labels[0]} (Å)", fontsize=12)
        ax.set_ylabel(f"RMSD to {ref_labels[1]} (Å)", fontsize=12)
        ax.set_title("Cα RMSD to reference structures", fontsize=13)
        ax.legend(framealpha=0.85, fontsize=9, markerscale=2)
        ax.grid(True, which="both", alpha=0.25, linestyle=":")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "rmsd_scatter.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("  Saved rmsd_scatter.png")

    # --------------------------------------------------------------- histograms
    fig, axes = plt.subplots(n_refs, 1, figsize=(8, 3.5 * n_refs), sharex=False)
    if n_refs == 1:
        axes = [axes]

    cluster_colours = {
        int(c): _CLUSTER_COLOURS[int(c) % len(_CLUSTER_COLOURS)]
        for c in np.unique(cluster_ids[cluster_ids >= 0])
    }

    for j, ax in enumerate(axes):
        # "All frames" as a density step outline — gives shape context without
        # dominating the scale, so per-cluster bars remain clearly visible.
        ax.hist(
            rmsd_values[:, j], bins=60, density=True,
            histtype="step", color=_UNASSIGNED_COLOUR, linewidth=1.5,
            label="All frames",
        )
        for c, col in cluster_colours.items():
            mask = cluster_ids == c
            if mask.sum() == 0:
                continue
            ax.hist(
                rmsd_values[mask, j], bins=60, density=True,
                histtype="stepfilled", color=col, alpha=0.65,
                label=f"Cluster {c} \u2014 {ref_labels[c]}",
            )

        if rmsd_threshold is not None:
            ax.axvline(
                rmsd_threshold, color="#111111", linewidth=1.5, linestyle="--",
                label=f"Threshold {rmsd_threshold} \u00c5",
            )

        ax.set_title(f"RMSD distribution \u2014 {ref_labels[j]}", fontsize=12)
        ax.set_xlabel("RMSD (\u00c5)", fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        ax.legend(framealpha=0.85, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle=":")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rmsd_histograms.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved rmsd_histograms.png")


def print_summary(cluster_ids, n_refs):
    n_frames = len(cluster_ids)
    n_clustered = np.sum(cluster_ids >= 0)
    n_unclustered = np.sum(cluster_ids == -1)
    
    print("\n--- Clustering Summary ---")
    print(f"Total frames: {n_frames}")
    print(f"Clustered frames: {n_clustered} ({n_clustered/n_frames*100:.1f}%)")
    print(f"Unclustered frames: {n_unclustered} ({n_unclustered/n_frames*100:.1f}%)")
    
    for j in range(n_refs):
        n_c = np.sum(cluster_ids == j)
        print(f"Cluster {j}: {n_c} ({n_c/n_frames*100:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Cluster a trajectory by RMSD to reference structures.")
    parser.add_argument("--trajectory_path", required=False, type=str, help="Path to the trajectory file (.xtc)",default="jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_plddt_ordered.xtc")
    parser.add_argument("--topology_path", required=False, type=str, help="Path to the topology file (.pdb)",default="/Users/alexi/JAX-ENT/jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_max_plddt_425.pdb")
    parser.add_argument("--reference_paths", required=False, nargs='+', type=str, help="Paths to the reference structures (.pdb)", default=["jaxent/examples/4_SAXS/FOXS/missing_residues/1CLL_apo.pdb","jaxent/examples/4_SAXS/FOXS/missing_residues/7PSZ_apo.pdb"])
    parser.add_argument("--output_path", type=str, default="_RMSD_cluster_output_test/", help="Path to the output directory")
    parser.add_argument("--rmsd_threshold", type=float, default=6.0, help="RMSD threshold for assigning to a cluster (Å)")
    
    args = parser.parse_args()

    print("Starting RMSD Clustering...")
    print(f"Topology: {args.topology_path}")
    print(f"Trajectory: {args.trajectory_path}")
    print(f"References: {args.reference_paths}")
    print(f"Output dir: {args.output_path}")
    print("-" * 40)

    rmsd_values = compute_rmsd_to_references(
        topology_path=args.topology_path, 
        trajectory_path=args.trajectory_path,
        reference_paths=args.reference_paths
    )
    
    cluster_ids = cluster_by_rmsd(rmsd_values, rmsd_threshold=args.rmsd_threshold)
    
    save_results(rmsd_values, cluster_ids, args.output_path, args.reference_paths)
    plot_results(rmsd_values, cluster_ids, args.output_path,
                 reference_paths=args.reference_paths,
                 rmsd_threshold=args.rmsd_threshold)
    print_summary(cluster_ids, len(args.reference_paths))

    print("\nClustering complete!")


if __name__ == "__main__":
    main()