"""
Plots the PCA of the clustered ensemble against reference PDBs

as well as PCA coloring by various features:
- Cluster labels
- N-C termini distance
- NAC (residues 61-95) mean protection
- p2 motif (residues 45-57) mean protection
- Aligned RMSD to reference PDBs (as 3 panels)

PCA coords:
  jaxent/examples/4_aSyn/data/_cluster_aSyn/data/pca_coordinates.npy
centres:
  jaxent/examples/4_aSyn/data/_cluster_aSyn/data/cluster_centers.npy

protection factors (all frames) are found in
  jaxent/examples/4_aSyn/data/_cluster_aSyn/data/aSyn_featurised.npz

Commands to perform initial clustering and featurisation:
  jaxent/examples/4_aSyn/data/cluster_aSyn.sh

References:
  Rod:     jaxent/examples/4_aSyn/data/_aSyn/AF-P37840-F1-model_v6.pdb
  Hairpin: jaxent/examples/4_aSyn/data/_aSyn/1XQ8.pdb
  Compact: jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_max_plddt_12691.pdb
"""

import json
import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.align
import MDAnalysis.analysis.rms
import numpy as np
from scipy.spatial.distance import pdist
from jaxent.cli.efficient_k_cluster import calculate_distances_and_perform_pca
import jaxent.cli.efficient_k_cluster as _kcluster

# Suppress PDB CRYST1 warnings
warnings.filterwarnings("ignore", message=".*1 A.*CRYST1.*")

# ============================================================================
# Module-level constants
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "_cluster_aSyn" / "data"
PLOTS_DIR = SCRIPT_DIR / "_cluster_aSyn" / "plots"
TRAJ_DIR = SCRIPT_DIR / "_aSyn"

TOP_PDB = TRAJ_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb"
TRAJ_XTC = TRAJ_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc"

REFERENCES = {
    "Rod (AF)": TRAJ_DIR / "AF-P37840-F1-model_v6.pdb",
    "Hairpin": TRAJ_DIR / "1XQ8.pdb",
    "Compact": TRAJ_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_max_plddt_12691.pdb",
}

DYNAMIC_REFERENCES = {
    "Dynamic Hairpin": TRAJ_DIR / "bioemu_1.2pdb.pdb",
}

REF_MARKERS = {"Rod (AF)": "*", "Hairpin": "D", "Compact": "^"}
REF_COLORS = {"Rod (AF)": "#e41a1c", "Hairpin": "#377eb8", "Compact": "#4daf4a"}

DYN_REF_MARKERS = {"Dynamic Hairpin": "o"}
DYN_REF_COLORS = {"Dynamic Hairpin": "#ff7f00"}

NAC_RANGE = range(61, 96)  # residues 61-95, all non-proline
P2_RANGE = range(45, 58)  # residues 45-57, all non-proline
ALIGN_RANGE = range(1, 45)

PLDDT_FILE = TRAJ_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_info.txt"

# ============================================================================
# Helper functions for plotting
# ============================================================================


def set_publication_style():
    """Apply publication-quality matplotlib settings."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def remove_top_right_spines(ax):
    """Remove top and right spines from axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_colorbar(fig, ax, scatter, label):
    """Add a colorbar to a scatter plot."""
    return fig.colorbar(scatter, ax=ax, label=label, pad=0.02, shrink=0.95)


# ============================================================================
# Data loading functions
# ============================================================================


def load_pca_data():
    """Load pre-computed PCA coordinates, cluster centers, and labels."""
    pca_coords = np.load(DATA_DIR / "pca_coordinates.npy")
    centers = np.load(DATA_DIR / "cluster_centers.npy")
    labels = np.load(DATA_DIR / "cluster_labels.npy")

    assert pca_coords.shape == (12700, 20), f"PCA coords shape mismatch: {pca_coords.shape}"
    assert centers.shape == (1000, 20), f"Centers shape mismatch: {centers.shape}"
    assert labels.shape == (12700,), f"Labels shape mismatch: {labels.shape}"

    return pca_coords, centers, labels


def load_topology_map(topo_path):
    """Load topology JSON and build residue ID to log_Pf array index map.

    Returns:
        dict: {pdb_residue_number: log_Pf_array_index}
    """
    with open(topo_path) as f:
        topo = json.load(f)

    resid_to_idx = {t["residues"][0]: t["fragment_index"] for t in topo["topologies"]}
    return resid_to_idx


def load_log_pf(feat_path):
    """Load log protection factors from npz file.

    Returns:
        np.ndarray: shape (133, 12700) — log_Pf per residue per frame
    """
    feat = np.load(feat_path, allow_pickle=True)
    return feat["log_Pf"]


def compute_region_mean_log_pf(log_pf, resid_to_idx, resid_range):
    """Compute mean log_Pf for a residue range.

    Args:
        log_pf: shape (133, 12700)
        resid_to_idx: dict mapping residue number to array index
        resid_range: range or list of residue numbers (e.g., range(61, 96))

    Returns:
        np.ndarray: shape (12700,) — mean log_Pf per frame
    """
    idx = [resid_to_idx[r] for r in resid_range if r in resid_to_idx]
    if not idx:
        raise ValueError(f"No residues found in range {resid_range}")
    return log_pf[idx, :].mean(axis=0)


def load_plddt_values(plddt_file):
    """Load pLDDT values from AF2 plddt_info file.

    File format: Frame\tpLDDT (header line + 12700 data lines)
    Data lines are ordered as they appear in the trajectory (frame 0, 1, 2, ...).

    Returns:
        np.ndarray: shape (12700,) — pLDDT value per frame
    """
    plddt_vals = []
    with open(plddt_file) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                plddt_vals.append(float(parts[1]))
    return np.array(plddt_vals, dtype=np.float32)


# ============================================================================
# MDAnalysis computation functions
# ============================================================================


def compute_nc_distances(top_pdb, traj_xtc, resid_to_idx):
    """Compute N-C terminus distance for each frame.

    Returns:
        np.ndarray: shape (12700,) — distances in Angstroms
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))

    res_ids = sorted(resid_to_idx.keys())
    n_resid, c_resid = res_ids[0], res_ids[-1]

    ca_n = u.select_atoms(f"name CA and resid {n_resid}")
    ca_c = u.select_atoms(f"name CA and resid {c_resid}")

    assert ca_n.n_atoms == 1 and ca_c.n_atoms == 1, \
        f"N/C CA selection mismatch: n_resid {n_resid} has {ca_n.n_atoms} atoms, " \
        f"c_resid {c_resid} has {ca_c.n_atoms} atoms"

    distances = np.empty(len(u.trajectory), dtype=np.float32)
    for i, ts in enumerate(u.trajectory):
        distances[i] = np.linalg.norm(ca_n.positions[0] - ca_c.positions[0])

    return distances


def compute_rmsd_to_reference(top_pdb, traj_xtc, ref_pdb):
    """Compute Ca RMSD between trajectory and reference structure.

    Returns:
        np.ndarray: shape (12700,) — RMSD in Angstroms
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    ref = mda.Universe(str(ref_pdb))

    traj_ca = u.select_atoms("name CA")
    ref_ca = ref.select_atoms("name CA")

    if traj_ca.n_atoms != ref_ca.n_atoms:
        raise ValueError(
            f"CA count mismatch: trajectory has {traj_ca.n_atoms} atoms, "
            f"reference {ref_pdb.name} has {ref_ca.n_atoms} atoms"
        )

    # Align the trajectory to the reference structure using ALIGN_RANGE
    # This modifies u.positions in-place for each frame
    align = mda.analysis.align.AlignTraj(u, ref, select=f"resid {ALIGN_RANGE[0]} to {ALIGN_RANGE[-1]} and name CA", in_memory=True).run()

    R = mda.analysis.rms.RMSD(u, ref, select="name CA")
    R.run()

    # Column 2 contains the RMSD values (columns 0=frame_idx, 1=time, 2=rmsd)
    return R.results.rmsd[:, 2]


def find_ref_pca_position(rmsd_values, pca_coords):
    """Find the PCA position of the frame with minimum RMSD to reference.

    Args:
        rmsd_values: shape (12700,)
        pca_coords: shape (12700, 20)

    Returns:
        np.ndarray: shape (20,) — PCA coordinates of minimum-RMSD frame
    """
    min_frame = np.argmin(rmsd_values)
    return pca_coords[min_frame]


# ============================================================================
# Plotting functions
# ============================================================================


def plot_biophysical_panels(pca_coords, centers, labels, nc_dist, nac_prot, p2_prot,
                           ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by biophysical features.

    Creates a 1x4 figure with panels:
      A: Cluster labels
      B: N-C terminus distance
      C: NAC region mean protection
      D: p2 motif mean protection
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    # -------- Panel A: Cluster labels --------
    ax = axes[0]
    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=labels, cmap="tab20", s=10, alpha=0.4,
        linewidths=0, rasterized=True
    )

    # Overlay cluster centers
    ax.scatter(
        centers[:, 0], centers[:, 1],
        c="black", s=15, marker=".", alpha=0.7,
        linewidths=0, zorder=5, label="Cluster centers"
    )

    # Overlay reference positions
    for name in REFERENCES.keys():
        pos = ref_positions[name]
        ax.scatter(
            pos[0], pos[1],
            marker=REF_MARKERS[name], s=200,
            c=REF_COLORS[name], zorder=10,
            edgecolors="black", linewidths=1,
            label=name
        )

    # Overlay dynamic reference positions
    for name, pos_array in dynamic_ref_positions.items():

        ax.scatter(
            pos_array[:, 0], pos_array[:, 1],
            marker=DYN_REF_MARKERS.get(name, "o"), s=50,
            c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
            edgecolors="white", linewidths=0.5,
            label=name
        )

    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_title("Cluster")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    remove_top_right_spines(ax)

    # -------- Panel B: N-C distance --------
    ax = axes[1]
    vmax_nc = np.percentile(nc_dist, 99)
    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=nc_dist, cmap="bone", s=10, alpha=0.4,
        linewidths=0, rasterized=True,
        vmin=0, vmax=vmax_nc
    )

    # Overlay reference positions
    for name in REFERENCES.keys():
        pos = ref_positions[name]
        ax.scatter(
            pos[0], pos[1],
            marker=REF_MARKERS[name], s=200,
            c=REF_COLORS[name], zorder=10,
            edgecolors="black", linewidths=1
        )

    # Overlay dynamic reference positions
    for name, pos_array in dynamic_ref_positions.items():
        ax.scatter(
            pos_array[:, 0], pos_array[:, 1],
            marker=DYN_REF_MARKERS.get(name, "o"), s=50,
            c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
            edgecolors="white", linewidths=0.5
        )

    add_colorbar(fig, ax, scatter, "N-C distance (Å)")
    ax.set_title("N-C distance")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    remove_top_right_spines(ax)

    # -------- Panel C: NAC protection --------
    ax = axes[2]
    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=nac_prot, cmap="plasma", s=10, alpha=0.4,
        linewidths=0, rasterized=True
    )

    # Overlay reference positions
    for name in REFERENCES.keys():
        pos = ref_positions[name]
        ax.scatter(
            pos[0], pos[1],
            marker=REF_MARKERS[name], s=200,
            c=REF_COLORS[name], zorder=10,
            edgecolors="black", linewidths=1
        )

    # Overlay dynamic reference positions
    for name, pos_array in dynamic_ref_positions.items():
        ax.scatter(
            pos_array[:, 0], pos_array[:, 1],
            marker=DYN_REF_MARKERS.get(name, "o"), s=50,
            c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
            edgecolors="white", linewidths=0.5
        )

    add_colorbar(fig, ax, scatter, "NAC mean log P$_f$")
    ax.set_title("NAC protection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    remove_top_right_spines(ax)

    # -------- Panel D: p2 protection --------
    ax = axes[3]
    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=p2_prot, cmap="plasma", s=10, alpha=0.4,
        linewidths=0, rasterized=True
    )

    # Overlay reference positions
    for name in REFERENCES.keys():
        pos = ref_positions[name]
        ax.scatter(
            pos[0], pos[1],
            marker=REF_MARKERS[name], s=200,
            c=REF_COLORS[name], zorder=10,
            edgecolors="black", linewidths=1
        )

    # Overlay dynamic reference positions
    for name, pos_array in dynamic_ref_positions.items():
        ax.scatter(
            pos_array[:, 0], pos_array[:, 1],
            marker=DYN_REF_MARKERS.get(name, "o"), s=50,
            c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
            edgecolors="white", linewidths=0.5
        )

    add_colorbar(fig, ax, scatter, "p2 mean log P$_f$")
    ax.set_title("p2 protection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    remove_top_right_spines(ax)

    # Save figure
    fig.savefig(plots_dir / "pca_biophysical.png", dpi=300)
    plt.close(fig)
    print("Saved pca_biophysical.png")


def plot_rmsd_panels(pca_coords, rmsd_dict, ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by RMSD to each reference structure.

    Creates a 1x3 figure with one panel per reference PDB.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, (name, rmsd_vals) in zip(axes, rmsd_dict.items()):
        vmax = np.percentile(rmsd_vals, 99)

        scatter = ax.scatter(
            pca_coords[:, 0], pca_coords[:, 1],
            c=rmsd_vals, cmap="cividis_r", s=10, alpha=0.4,
            linewidths=0, rasterized=True,
            vmin=0, vmax=vmax
        )

        # Mark minimum-RMSD frame position
        pos = ref_positions[name]
        ax.scatter(
            pos[0], pos[1],
            marker=REF_MARKERS[name], s=300,
            c="white", edgecolors="black", linewidths=2,
            zorder=10
        )

        for dyn_name, pos_array in dynamic_ref_positions.items():
            ax.scatter(
                pos_array[:, 0], pos_array[:, 1],
                marker=DYN_REF_MARKERS.get(dyn_name, "o"), s=50,
                c=DYN_REF_COLORS.get(dyn_name, "#ff7f00"), zorder=9,
                edgecolors="white", linewidths=0.5
            )

        add_colorbar(fig, ax, scatter, "Cα RMSD (Å)")
        ax.set_title(name)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        remove_top_right_spines(ax)

    # Save figure
    fig.savefig(plots_dir / "pca_rmsd_references.png", dpi=300)
    plt.close(fig)
    print("Saved pca_rmsd_references.png")


def plot_plddt_panel(pca_coords, plddt_vals, ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by pLDDT (AF2 confidence metric).

    Creates a single-panel figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=plddt_vals, cmap="BuPu", s=20, alpha=1-(plddt_vals/100),
        linewidths=0, rasterized=True
    )

    # Overlay reference positions
    for name in REFERENCES.keys():
        pos = ref_positions[name]
        ax.scatter(
            pos[0], pos[1],
            marker=REF_MARKERS[name], s=200,
            c=REF_COLORS[name], zorder=10,
            edgecolors="black", linewidths=1,
            label=name
        )

    # Overlay dynamic reference positions
    for name, pos_array in dynamic_ref_positions.items():

        ax.scatter(
            pos_array[:, 0], pos_array[:, 1],
            marker=DYN_REF_MARKERS.get(name, "o"), s=50,
            c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
            edgecolors="white", linewidths=0.5,
            label=name
        )

    add_colorbar(fig, ax, scatter, "pLDDT")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.set_title("pLDDT (AF2 Confidence)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    remove_top_right_spines(ax)

    # Save figure
    fig.savefig(plots_dir / "pca_plddt.png", dpi=300)
    plt.close(fig)
    print("Saved pca_plddt.png")


def plot_feature_histograms(nc_dist, nac_prot, p2_prot, rmsd_dict, plddt_vals, plots_dir):
    """Plot feature distribution histograms.

    Creates three figures:
      1. Biophysical features: nc_dist, nac_prot, p2_prot
      2. RMSD to each reference: Rod (AF), Hairpin, Compact
      3. pLDDT single panel
    """
    # -------- Figure 1: Biophysical features --------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Panel A: N-C distance
    ax = axes[0]
    ax.hist(nc_dist, bins=50, density=True, alpha=0.8, color="#4393c3", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("N–C Distance (Å)")
    ax.set_ylabel("Density")
    ax.set_title("N-C Terminus Distance")
    remove_top_right_spines(ax)

    # Panel B: NAC protection
    ax = axes[1]
    ax.hist(nac_prot, bins=50, density=True, alpha=0.8, color="#d0549b", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("NAC Mean Log P$_f$")
    ax.set_ylabel("Density")
    ax.set_title("NAC Region Protection (res 61–95)")
    remove_top_right_spines(ax)

    # Panel C: p2 protection
    ax = axes[2]
    ax.hist(p2_prot, bins=50, density=True, alpha=0.8, color="#f89441", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("p2 Mean Log P$_f$")
    ax.set_ylabel("Density")
    ax.set_title("p2 Motif Protection (res 45–57)")
    remove_top_right_spines(ax)

    fig.savefig(plots_dir / "feature_distributions_biophysical.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved feature_distributions_biophysical.png")

    # -------- Figure 2: RMSD to references --------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    ref_names = ["Rod (AF)", "Hairpin", "Compact"]
    ref_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    for ax, name, color in zip(axes, ref_names, ref_colors):
        rmsd_vals = rmsd_dict[name]
        ax.hist(rmsd_vals, bins=50, density=True, alpha=0.8, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("RMSD to {} (Å)".format(name))
        ax.set_ylabel("Density")
        ax.set_title("RMSD to {}".format(name))
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "feature_distributions_rmsd.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved feature_distributions_rmsd.png")

    # -------- Figure 3: pLDDT --------
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)

    ax.hist(plddt_vals, bins=50, density=True, alpha=0.8, color="#6a51a3", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("pLDDT")
    ax.set_ylabel("Density")
    ax.set_title("pLDDT Distribution")
    remove_top_right_spines(ax)

    fig.savefig(plots_dir / "feature_distributions_plddt.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved feature_distributions_plddt.png")


# ============================================================================
# Main
# ============================================================================


def main():
    """Main workflow: load data, compute features, generate plots."""
    set_publication_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load pre-computed PCA data
    print("Loading PCA data...")
    pca_coords, centers, labels = load_pca_data()

    # 2. Load topology map and protection factors
    print("Loading topology and protection factors...")
    resid_to_idx = load_topology_map(DATA_DIR / "topology.json")
    log_pf = load_log_pf(DATA_DIR / "aSyn_featurised.npz")

    # 3. Compute regional protection factors
    print("Computing regional protection factors...")
    nac_prot = compute_region_mean_log_pf(log_pf, resid_to_idx, NAC_RANGE)
    p2_prot = compute_region_mean_log_pf(log_pf, resid_to_idx, P2_RANGE)

    # 4. Load pLDDT values
    print("Loading pLDDT values...")
    plddt_vals = load_plddt_values(PLDDT_FILE)

    # 5. Compute N-C distances
    print("Computing N-C termini distances...")
    nc_dist = compute_nc_distances(TOP_PDB, TRAJ_XTC, resid_to_idx)

    # 7. Fit PCA model to project references properly
    print("Fitting PCA model on trajectory...")
    _kcluster.logger = logging.getLogger(__name__)
    u = mda.Universe(str(TOP_PDB), str(TRAJ_XTC))
    traj_ca_resids = u.select_atoms("name CA").resids.copy()
    pca_coords_refitted, pca_model = calculate_distances_and_perform_pca(u, "name CA", 20, 100)

    # Align signs with saved coordinates
    print("Aligning PCA component signs...")
    for i in range(20):
        corr = np.corrcoef(pca_coords_refitted[:, i], pca_coords[:, i])[0, 1]
        if corr < 0:
            pca_model.components_[i] *= -1

    # 8. Compute RMSD and project references to PCA space
    rmsd_dict = {}
    ref_positions = {}
    for name, ref_pdb in REFERENCES.items():
        print(f"Computing RMSD and projecting {name}...")
        rmsd_vals = compute_rmsd_to_reference(TOP_PDB, TRAJ_XTC, ref_pdb)
        rmsd_dict[name] = rmsd_vals

        # Try direct projection via PCA model
        ref = mda.Universe(str(ref_pdb))
        ref_ca = ref.select_atoms("name CA")

        if ref_ca.n_atoms == len(traj_ca_resids) and np.array_equal(ref_ca.resids, traj_ca_resids):
            # Project via PCA
            dists = pdist(ref_ca.positions, metric="euclidean")
            ref_positions[name] = pca_model.transform(dists.reshape(1, -1))[0]
            print(f"  Projected {name} via PCA model")
        else:
            # Fallback to nearest-RMSD frame
            ref_positions[name] = find_ref_pca_position(rmsd_vals, pca_coords)
            print(f"  {name}: using nearest-RMSD frame as fallback")

    dynamic_ref_positions = {}
    for name, ref_pdb in DYNAMIC_REFERENCES.items():
        print(f"Projecting dynamic reference {name}...")
        ref = mda.Universe(str(ref_pdb))
        
        # Select matched CA atoms
        sel_str = "name CA and resid " + " ".join(map(str, traj_ca_resids))
        ref_ca = ref.select_atoms(sel_str)
        
        if ref_ca.n_atoms != len(traj_ca_resids) or not np.array_equal(ref_ca.resids, traj_ca_resids):
            print(f"  {name} atom count mismatch (got {ref_ca.n_atoms}, expected {len(traj_ca_resids)}), skipping projection")
            continue
            
        pos_list = []
        for ts in ref.trajectory:
            dists = pdist(ref_ca.positions, metric="euclidean")
            pos_list.append(pca_model.transform(dists.reshape(1, -1))[0])
            
        dynamic_ref_positions[name] = np.array(pos_list)
        print(f"  Projected {name} (N={len(ref.trajectory)}) via PCA model")

    # 9. Plot Figure 1: biophysical feature panels
    print("Plotting biophysical feature panels...")
    plot_biophysical_panels(
        pca_coords, centers, labels,
        nc_dist, nac_prot, p2_prot,
        ref_positions, dynamic_ref_positions, PLOTS_DIR
    )

    # 10. Plot Figure 2: RMSD panels
    print("Plotting RMSD panels...")
    plot_rmsd_panels(pca_coords, rmsd_dict, ref_positions, dynamic_ref_positions, PLOTS_DIR)

    # 11. Plot Figure 3: pLDDT panel
    print("Plotting pLDDT panel...")
    plot_plddt_panel(pca_coords, plddt_vals, ref_positions, dynamic_ref_positions, PLOTS_DIR)

    # 12. Plot feature distribution histograms
    print("Plotting feature distribution histograms...")
    plot_feature_histograms(nc_dist, nac_prot, p2_prot, rmsd_dict, plddt_vals, PLOTS_DIR)

    print(f"All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
