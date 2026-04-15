#!/usr/bin/env python3
"""Per-condition weighted feature distribution plots for 4_aSyn ensemble analysis.

Visualises how MaxEnt-optimised frame weights reshape N-C distance, NAC, and p2
distributions across the four aSyn conditions, revealing condition-specific
structural sub-population preferences.

Usage:
    python plot_feature_distributions_aSyn_conditions.py \
        --extracted-dir <path> \
        --feature-npz <path> \
        --topology-json <path> \
        --cluster-labels-npy <path> \
        --top-pdb <path> \
        --traj-xtc <path> \
        [--output-dir <path>] \
        [--config <path>] \
        [--absolute-paths]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns

from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.plotting import setup_publication_style

# ============================================================================
# Constants
# ============================================================================

NAC_RANGE = range(61, 96)  # residues 61-95
P2_RANGE = range(45, 58)   # residues 45-57
ALIGN_RANGE = range(1, 45)

FEATURE_NAMES = {
    "nc_distance": "N–C Distance (Å)",
    "nac_prot": "NAC Mean Log P$_f$",
    "p2_prot": "p2 Mean Log P$_f$",
}

# ============================================================================
# Data loading functions
# ============================================================================


def load_topology_map(topo_path: Path) -> dict[int, int]:
    """Load topology JSON and build residue ID to log_Pf array index map.

    Returns:
        dict: {pdb_residue_number: log_Pf_array_index}
    """
    with open(topo_path) as f:
        topo = json.load(f)

    resid_to_idx = {t["residues"][0]: t["fragment_index"] for t in topo["topologies"]}
    return resid_to_idx


def load_log_pf(feat_path: Path) -> np.ndarray:
    """Load log protection factors from npz file.

    Returns:
        np.ndarray: shape (133, 12700) — log_Pf per residue per frame
    """
    feat = np.load(feat_path, allow_pickle=True)
    return feat["log_Pf"]


def compute_region_mean_log_pf(
    log_pf: np.ndarray, resid_to_idx: dict[int, int], resid_range: range
) -> np.ndarray:
    """Compute mean log_Pf for a residue range.

    Args:
        log_pf: shape (133, 12700)
        resid_to_idx: dict mapping residue number to array index
        resid_range: range or list of residue numbers

    Returns:
        np.ndarray: shape (12700,) — mean log_Pf per frame
    """
    idx = [resid_to_idx[r] for r in resid_range if r in resid_to_idx]
    if not idx:
        raise ValueError(f"No residues found in range {resid_range}")
    return log_pf[idx, :].mean(axis=0)


def compute_nc_distances(
    top_pdb: Path, traj_xtc: Path, resid_to_idx: dict[int, int]
) -> np.ndarray:
    """Compute N-C terminus distance for each frame via MDAnalysis.

    Returns:
        np.ndarray: shape (12700,) — distances in Angstroms
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))

    res_ids = sorted(resid_to_idx.keys())
    n_resid, c_resid = res_ids[0], res_ids[-1]

    ca_n = u.select_atoms(f"name CA and resid {n_resid}")
    ca_c = u.select_atoms(f"name CA and resid {c_resid}")

    assert ca_n.n_atoms == 1 and ca_c.n_atoms == 1, (
        f"N/C CA selection mismatch: n_resid {n_resid} has {ca_n.n_atoms} atoms, "
        f"c_resid {c_resid} has {ca_c.n_atoms} atoms"
    )

    distances = np.empty(len(u.trajectory), dtype=np.float32)
    for i, ts in enumerate(u.trajectory):
        distances[i] = np.linalg.norm(ca_n.positions[0] - ca_c.positions[0])

    return distances


# ============================================================================
# Plotting helper
# ============================================================================


def remove_top_right_spines(ax):
    """Remove top and right spines from axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ============================================================================
# Main plotting function
# ============================================================================


def plot_feature_distributions_per_metric(
    metric_dir: Path,
    nc_dist: np.ndarray,
    nac_prot: np.ndarray,
    p2_prot: np.ndarray,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
):
    """Create 3×N_conditions figure showing weighted feature distributions.

    Args:
        metric_dir: directory containing {condition}_*_selected.npz files
        nc_dist: shape (n_frames,)
        nac_prot: shape (n_frames,)
        p2_prot: shape (n_frames,)
        cluster_labels: shape (n_frames,) or None (if input matches clusters)
        n_per_cluster: counts per cluster or None
        cfg: ExperimentConfig with ensembles and style colours
        output_dir: where to save the figure
    """
    metric_name = metric_dir.name  # e.g., "recovery_percent_max"
    n_conds = len(cfg.ensembles)

    # Prepare the per-condition frame weights
    condition_weights = {}
    for condition in cfg.ensembles:
        # Glob all matching NPZ files for this condition
        npz_files = list(metric_dir.glob(f"{condition}_*_selected.npz"))
        if not npz_files:
            print(f"WARNING: No extracted NPZ files found for {condition} in {metric_dir}")
            condition_weights[condition] = None
            continue

        # Load and stack frame_weights from all matching files
        all_weights = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            fw = data["frame_weights"]  # shape (replicates, n_clusters)
            all_weights.append(fw)

        stacked = np.vstack(all_weights)  # shape (total_replicates, n_clusters)
        cluster_weights = np.nanmean(stacked, axis=0)  # shape (n_clusters,)

        # Convert to per-frame weights
        if cluster_labels is not None and n_per_cluster is not None:
            # Map cluster-level weights back to original frames
            per_frame = cluster_weights[cluster_labels] / n_per_cluster[cluster_labels]
        else:
            # Use weights directly (assuming frames match cluster count)
            per_frame = cluster_weights

        # Data consistency check
        if len(per_frame) != len(nc_dist):
            print(f"WARNING: Weight count ({len(per_frame)}) mismatch with frame count ({len(nc_dist)}) for {condition}")
            condition_weights[condition] = None
            continue

        condition_weights[condition] = per_frame

    # Build figure: N_conditions rows × 3 feature columns
    figsize = (12, 3 * n_conds)
    fig, axes = plt.subplots(n_conds, 3, figsize=figsize, constrained_layout=True)
    if n_conds == 1:
        axes = axes.reshape(1, 3)

    feature_data = [nc_dist, nac_prot, p2_prot]
    feature_keys = ["nc_distance", "nac_prot", "p2_prot"]

    # Outer loop: conditions (rows), Inner loop: features (columns)
    for row_idx, condition in enumerate(cfg.ensembles):
        for col_idx, (feature_array, feature_key) in enumerate(zip(feature_data, feature_keys)):
            ax = axes[row_idx, col_idx]

            if condition_weights[condition] is None:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {condition}",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="red",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Get weights for this condition
            weights = condition_weights[condition]
            color = cfg.style.ensemble_colors.get(condition, "#808080")

            # Plot weighted histogram
            ax.hist(
                feature_array,
                bins=50,
                weights=weights,
                density=True,
                alpha=0.8,
                color=color,
                edgecolor="none",
            )

            # Labels: condition name on leftmost column, "Density" elsewhere
            if col_idx == 0:
                ax.set_ylabel(f"{condition}\nDensity", fontweight="bold")
            else:
                ax.set_ylabel("Density")

            # Feature x-label on bottom row only
            if row_idx == n_conds - 1:
                ax.set_xlabel(FEATURE_NAMES[feature_key])

            # Feature name as column title on top row only
            if row_idx == 0:
                ax.set_title(FEATURE_NAMES[feature_key], fontsize=11, fontweight="bold", pad=10)

            remove_top_right_spines(ax)

    fig.suptitle(
        f"Feature Distributions — {metric_name}",
        y=0.995,
        fontsize=13,
        fontweight="bold",
    )

    # Save
    out_path = output_dir / f"feature_distributions_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-condition weighted feature distributions for 4_aSyn."
    )
    parser.add_argument("--extracted-dir", required=True, help="Output dir from extract_selected_models")
    parser.add_argument("--feature-npz", default=None, help="Path to aSyn_featurised.npz")
    parser.add_argument("--topology-json", default=None, help="Path to topology.json")
    parser.add_argument("--cluster-labels-npy", default=None, help="Path to cluster_labels.npy (optional if traj is clustered)")
    parser.add_argument("--top-pdb", default=None, help="Topology PDB for MDAnalysis")
    parser.add_argument("--traj-xtc", default=None, help="Trajectory XTC for MDAnalysis")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <extracted-dir>/plots_feature_distributions)")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: ../config.yaml)")
    parser.add_argument("--absolute-paths", action="store_true", help="Interpret paths as absolute")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir.parent

    extracted_dir = Path(args.extracted_dir)
    feature_npz = Path(args.feature_npz) if args.feature_npz else script_dir.parent / "data/_aSyn/tris_MD/features/aSyn_featurised.npz"
    topology_json = Path(args.topology_json) if args.topology_json else script_dir.parent / "data/_aSyn/tris_MD/features/topology.json"
    cluster_labels_npy = Path(args.cluster_labels_npy) if args.cluster_labels_npy else None
    top_pdb = Path(args.top_pdb) if args.top_pdb else script_dir.parent / "data/_aSyn/tris_MD/md_mol_center_coil.pdb"  # Fallback to topology.json if needed
    traj_xtc = Path(args.traj_xtc) if args.traj_xtc else script_dir.parent / "data/_aSyn/tris_MD/tris_all_combined.xtc"

    # Specific override for PDB if it's explicitly provided or if internal topology.json is actually what's needed
    if not args.top_pdb and not top_pdb.exists():
         # Last resort stable PDB
         top_pdb = script_dir.parent / "data/_aSyn/a99sb.pdb"

    if not args.absolute_paths:
        extracted_dir = (script_dir / extracted_dir).resolve()
        if args.feature_npz: feature_npz = (script_dir / feature_npz).resolve()
        if args.topology_json: topology_json = (script_dir / topology_json).resolve()
        if cluster_labels_npy: cluster_labels_npy = (script_dir / cluster_labels_npy).resolve()
        if args.top_pdb: top_pdb = (script_dir / top_pdb).resolve()
        if args.traj_xtc: traj_xtc = (script_dir / traj_xtc).resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir) if args.absolute_paths else (script_dir / args.output_dir).resolve()
    else:
        output_dir = extracted_dir / "plots_feature_distributions"

    if args.config:
        config_path = Path(args.config) if args.absolute_paths else (script_dir / args.config).resolve()
    else:
        config_path = exp_dir / "config.yaml"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"extracted_dir:    {extracted_dir}")
    print(f"feature_npz:      {feature_npz}")
    print(f"topology_json:    {topology_json}")
    print(f"cluster_labels:   {cluster_labels_npy}")
    print(f"top_pdb:          {top_pdb}")
    print(f"traj_xtc:         {traj_xtc}")
    print(f"output_dir:       {output_dir}")
    print(f"config:           {config_path}")
    print("-" * 60)

    # Load config
    cfg = ExperimentConfig.from_yaml(config_path)

    # Set publication style
    setup_publication_style()
    sns.set_style("ticks")
    sns.set_context(
        "paper",
        rc={
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        },
    )

    # Load once: topology, features, cluster labels, N-C distances
    print("Loading topology map...")
    resid_to_idx = load_topology_map(topology_json)

    print("Loading protection factors...")
    log_pf = load_log_pf(feature_npz)

    print("Computing regional mean log Pf...")
    nac_prot = compute_region_mean_log_pf(log_pf, resid_to_idx, NAC_RANGE)
    p2_prot = compute_region_mean_log_pf(log_pf, resid_to_idx, P2_RANGE)

    cluster_labels = None
    n_per_cluster = None
    if cluster_labels_npy:
        print(f"Loading cluster labels from {cluster_labels_npy}...")
        cluster_labels = np.load(cluster_labels_npy)
        print("Computing n_per_cluster...")
        n_per_cluster = np.bincount(cluster_labels)
    else:
        print("No cluster labels provided. Assuming trajectory frames correspond to clusters directly.")

    print("Computing N-C distances (this may take a minute)...")
    nc_dist = compute_nc_distances(top_pdb, traj_xtc, resid_to_idx)

    # Loop over metric subdirectories (filter for {metric}_{direction} pattern)
    # Expected pattern: e.g., "recovery_percent_max", "spearman_mean_max"
    # Skip directories that start with underscore (like _extracted_*) or match output pattern
    print(f"Scanning for metric directories in {extracted_dir}...")
    metric_dirs = sorted(
        [
            d
            for d in extracted_dir.iterdir()
            if (
                d.is_dir()
                and "_" in d.name
                and not d.name.startswith("_")
                and not d.name.startswith("plots_")
            )
        ]
    )
    if not metric_dirs:
        print(f"ERROR: No metric subdirectories found in {extracted_dir}")
        return

    print(f"Found {len(metric_dirs)} metric directory(-ies): {[d.name for d in metric_dirs]}")
    print("-" * 60)

    for metric_dir in metric_dirs:
        print(f"\nProcessing {metric_dir.name}...")
        plot_feature_distributions_per_metric(
            metric_dir,
            nc_dist,
            nac_prot,
            p2_prot,
            cluster_labels,
            n_per_cluster,
            cfg,
            output_dir,
        )

    print("\nAll plots saved to:", output_dir)


if __name__ == "__main__":
    main()
