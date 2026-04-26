"""
Plots the PCA of two clustered ensembles (e.g. AF2 vs MD) against reference PDBs

as well as PCA coloring by various features:
- N-C termini distance
- NAC (residues 61-95) mean protection
- p2 motif (residues 45-57) mean protection
- Aligned RMSD to reference PDBs
- Termini contacts
- C-tail protection

Features are overlaid in histograms to compare populations.
PCA plots are showing in two rows (one for each ensemble) to allow easier comparison
while retaining a shared coordinate system.
"""

import json
import itertools
import logging
import warnings
import argparse
import matplotlib
from pathlib import Path

import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.align
import MDAnalysis.analysis.rms
import numpy as np
import seaborn as sns
import matplotlib.patheffects as pe
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.analysis.dihedrals import Ramachandran
from jaxent.cli.efficient_k_cluster import calculate_distances_and_perform_pca
import jaxent.cli.efficient_k_cluster as _kcluster

# Suppress PDB CRYST1 warnings
warnings.filterwarnings("ignore", message=".*1 A.*CRYST1.*")

# ============================================================================
# Module-level constants
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent

# Default paths
DEFAULT_EXP_DIR = SCRIPT_DIR / "_aSyn"
DEFAULT_PLDDT_FILE = DEFAULT_EXP_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_info.txt"

# Ensemble 1 (AF2)
DEFAULT_TOP1 = DEFAULT_EXP_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb"
DEFAULT_TRAJ1 = DEFAULT_EXP_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc"
DEFAULT_FEAT1 = DEFAULT_EXP_DIR / "features" / "aSyn_featurised.npz"
DEFAULT_TOPO1 = DEFAULT_EXP_DIR / "features" / "topology.json"

# Ensemble 2 (MD)
DEFAULT_TOP2 = DEFAULT_EXP_DIR / "tris_MD" / "md_mol_center_coil.pdb"
DEFAULT_TRAJ2 = DEFAULT_EXP_DIR / "tris_MD" / "tris_all_combined.xtc"
DEFAULT_FEAT2 = DEFAULT_EXP_DIR / "tris_MD" / "features" / "aSyn_featurised.npz"
DEFAULT_TOPO2 = DEFAULT_EXP_DIR / "tris_MD" / "features" / "topology.json"

REFERENCES = {
    "Rod (AF)": DEFAULT_TOP1,
    "Hairpin": DEFAULT_EXP_DIR / "2kkw_1.pdb",
    "Compact": DEFAULT_EXP_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_max_plddt_12691.pdb",
}

DYNAMIC_REFERENCES = {
    "Dynamic Hairpin": DEFAULT_EXP_DIR / "2kkw.pdb",
}

REF_MARKERS = {"Rod (AF)": "*", "Hairpin": "D", "Compact": "^"}
REF_COLORS = {"Rod (AF)": "#e41a1c", "Hairpin": "#377eb8", "Compact": "#4daf4a"}

DYN_REF_MARKERS = {"Dynamic Hairpin": "o"}
DYN_REF_COLORS = {"Dynamic Hairpin": "#ff7f00"}

NAC_RANGE = range(61, 96)  # residues 61-95, all non-proline
P2_RANGE = range(45, 58)  # residues 45-57, all non-proline
ALIGN_RANGE = range(1, 45)
T1_RANGE = range(1, 61)
T2_RANGE = range(96, 141)
NHEAD_RANGE = range(1, 61)
NAC_RANGE = range(61, 96)
CTAIL_RANGE = range(115, 140)

PCA_ATOM_SELECTION = "name CA and resid 1:96"
RMSD_ATOM_SELECTION = "name CA and resid 1:65"
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


def get_pca_limits(coords_list, ref_pos_dict=None, margin=0.05):
    """Calculate shared PC1 and PC2 limits across multiple sets of coordinates.
    
    Args:
        coords_list (list): List of np.ndarrays of shape (N, 2)
        ref_pos_dict (dict): Optional dict of reference positions {name: (x, y)}
        margin (float): Fraction of padding to add.
        
    Returns:
        tuple: (xmin, xmax), (ymin, ymax)
    """
    all_coords = list(coords_list)
    if ref_pos_dict:
        # Some references might be single points, some might be arrays
        for val in ref_pos_dict.values():
            if val.ndim == 1:
                all_coords.append(val.reshape(1, -1))
            else:
                all_coords.append(val)
                
    combined = np.vstack(all_coords)
    xmin, xmax = combined[:, 0].min(), combined[:, 0].max()
    ymin, ymax = combined[:, 1].min(), combined[:, 1].max()
    
    xrange = xmax - xmin
    yrange = ymax - ymin
    
    return (xmin - margin * xrange, xmax + margin * xrange), \
           (ymin - margin * yrange, ymax + margin * yrange)


# ============================================================================
# Data loading functions
# ============================================================================




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


def compute_termini_contacts(top_pdb, traj_xtc, range1, range2, cutoff=8.0):
    """Compute number of CA-CA contacts between two residue ranges.

    Returns:
        np.ndarray: shape (12700,) — contact counts per frame
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))

    sel1 = u.select_atoms(f"name CA and resid {' '.join(map(str, range1))}")
    sel2 = u.select_atoms(f"name CA and resid {' '.join(map(str, range2))}")

    if sel1.n_atoms == 0 or sel2.n_atoms == 0:
        raise ValueError(f"Selection empty: range1 has {sel1.n_atoms}, range2 has {sel2.n_atoms}")

    contact_counts = np.empty(len(u.trajectory), dtype=np.int32)
    for i, ts in enumerate(u.trajectory):
        dists = cdist(sel1.positions, sel2.positions)
        contact_counts[i] = np.sum(dists < cutoff)

    return contact_counts


def compute_rmsd_to_reference(top_pdb, traj_xtc, ref_pdb):
    """Compute Ca RMSD between trajectory and reference structure.

    Returns:
        np.ndarray: shape (12700,) — RMSD in Angstroms
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    ref = mda.Universe(str(ref_pdb))

    traj_ca = u.select_atoms(RMSD_ATOM_SELECTION)
    ref_ca = ref.select_atoms(RMSD_ATOM_SELECTION)
    
    if traj_ca.n_atoms != ref_ca.n_atoms:
        raise ValueError(
            f"CA count mismatch: trajectory has {traj_ca.n_atoms} atoms, "
            f"reference {ref_pdb.name} has {ref_ca.n_atoms} atoms"
        )
    
    # Align the trajectory to the reference structure using ALIGN_RANGE
    # This modifies u.positions in-place for each frame
    align = mda.analysis.align.AlignTraj(u, ref, select=f"resid {ALIGN_RANGE[0]} to {ALIGN_RANGE[-1]} and {RMSD_ATOM_SELECTION}", in_memory=True).run()
    
    R = mda.analysis.rms.RMSD(u, ref, select=RMSD_ATOM_SELECTION)
    R.run()

    # Column 2 contains the RMSD values (columns 0=frame_idx, 1=time, 2=rmsd)
    return R.results.rmsd[:, 2]


def compute_drmsd_to_reference(top_pdb, traj_xtc, ref_pdb):
    """Compute Dihedral RMSD (dRMSD) between trajectory and reference structure.
    
    Uses backbone phi and psi angles for residues in RMSD_ATOM_SELECTION.
    Circular distance is used for angle differences.
    
    Returns:
        np.ndarray: shape (n_frames,) — dRMSD in degrees
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    ref = mda.Universe(str(ref_pdb))
    
    # Use the same residue range as RMSD
    # RMSD_ATOM_SELECTION is "name CA and resid 1:95"
    # We want residues 1 to 95
    sel_traj = u.select_atoms("protein and resid 1:95")
    sel_ref = ref.select_atoms("protein and resid 1:95")
    
    # Ramachandran analysis extracts phi/psi pairs
    # Note: it only returns for residues that HAVE both phi and psi
    ram_traj = Ramachandran(sel_traj).run()
    
    # Ensure we only use the first frame for the reference (in case of multi-model PDB)
    # We run it on the same selection to match residue list
    ram_ref = Ramachandran(sel_ref).run(start=0, stop=1)
    
    # angles shape: (n_frames, n_residues, 2)
    angles_traj = ram_traj.results.angles
    angles_ref = ram_ref.results.angles
    
    # Check shape consistency (excluding frame dimension)
    if angles_traj.shape[1:] != angles_ref.shape[1:]:
        # Handle count mismatch (can happen if pdb files have different completeness)
        min_res = min(angles_traj.shape[1], angles_ref.shape[1])
        angles_traj = angles_traj[:, :min_res, :]
        angles_ref = angles_ref[:, :min_res, :]
        logging.warning(f"dRMSD: Residue count mismatch ({angles_traj.shape[1]} vs {angles_ref.shape[1]}). Truncating to {min_res}.")

    # Flatten to (n_frames, n_angles) where n_angles = n_residues * 2
    n_frames = angles_traj.shape[0]
    angles_traj = angles_traj.reshape(n_frames, -1)
    # angles_ref should be shape (1, n_res, 2) -> reshape to (1, n_angles)
    angles_ref = angles_ref.reshape(1, -1)
    
    # Convert to radians for circular distance
    rad_traj = np.deg2rad(angles_traj)
    rad_ref = np.deg2rad(angles_ref)
    
    # Circular difference: arctan2(sin(a-b), cos(a-b))
    # diff shape: (n_frames, n_angles)
    diff = rad_traj - rad_ref
    circ_diff = np.arctan2(np.sin(diff), np.cos(diff))
    
    # dRMSD = sqrt(mean(circ_diff^2))
    # mean over n_angles (axis 1)
    drmsd_rad = np.sqrt(np.mean(circ_diff**2, axis=1))
    
    # Convert back to degrees for the output
    return np.rad2deg(drmsd_rad)


def compute_radgyr(top_pdb, traj_xtc, resid_range):
    """Compute Radius of Gyration for a residue range.

    Returns:
        np.ndarray: shape (12700,) — RadGyr in Angstroms
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    sel = u.select_atoms(f"name CA and resid {' '.join(map(str, resid_range))}")

    if sel.n_atoms == 0:
        raise ValueError(f"Selection empty for range {resid_range}")

    rg_values = np.empty(len(u.trajectory), dtype=np.float32)
    for i, ts in enumerate(u.trajectory):
        rg_values[i] = sel.radius_of_gyration()

    return rg_values


def compute_secondary_structure(top_pdb, traj_xtc):
    """Compute secondary structure fractions using DSSP for different regions.

    Returns:
        dict: Mapping region names to (alpha_frac, beta_frac) tuples.
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    dssp = DSSP(u)
    dssp.run()
    
    # DSSP results are in dssp.results.dssp, shape (n_frames, n_residues)
    resids = u.select_atoms("protein").residues.resids
    
    regions = {
        "whole": np.arange(len(resids)),
        "nhead": np.where(np.isin(resids, range(1, 61)))[0],
        "nac": np.where(np.isin(resids, range(61, 96)))[0],
        "ctail": np.where(np.isin(resids, range(96, 141)))[0]
    }
    
    results = {}
    for name, indices in regions.items():
        if len(indices) == 0:
            results[name] = (np.zeros(len(u.trajectory)), np.zeros(len(u.trajectory)))
            continue
            
        sub_arr = dssp.results.dssp[:, indices]
        alpha = np.mean(sub_arr == 'H', axis=1)
        beta = np.mean(sub_arr == 'E', axis=1)
        results[name] = (alpha, beta)
        
    return results


def find_ref_pca_position(rmsd_values, pca_coords):
    """Find the PCA position of the frame with minimum RMSD to reference."""
    min_frame = np.argmin(rmsd_values)
    return pca_coords[min_frame]


def compute_inertia_ratios_simple(top_pdb, traj_xtc, sel_str):
    """Compute per-frame principal moment ratios I1/I3 and I2/I3."""
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    sel = u.select_atoms(sel_str)
    n_frames = len(u.trajectory)
    x_ratio = np.zeros(n_frames)
    y_ratio = np.zeros(n_frames)

    for i, _ts in enumerate(u.trajectory):
        evals = np.linalg.eigvalsh(sel.moment_of_inertia())
        x_ratio[i] = evals[0] / evals[2]
        y_ratio[i] = evals[1] / evals[2]
    return x_ratio, y_ratio


def compute_reference_inertia_ratios(ref_dict, sel_str):
    """Compute I1/I3 and I2/I3 for reference PDBs."""
    ratios = {}
    for name, pdb_path in ref_dict.items():
        if not pdb_path.exists(): continue
        ref = mda.Universe(str(pdb_path))
        sel = ref.select_atoms(sel_str)
        if sel.n_atoms == 0: continue
        evals = np.linalg.eigvalsh(sel.moment_of_inertia())
        ratios[name] = (evals[0] / evals[2], evals[1] / evals[2])
    return ratios


# ============================================================================
# Plotting functions
# ============================================================================


def plot_shape_space_panels(x1, y1, x2, y2, ref_ratios, plots_dir):
    """Plot triangular shape space for both ensembles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    def _draw_triangle(ax):
        tx, ty = [0, 1, 0.5, 0], [1, 1, 0.5, 1]
        ax.plot(tx, ty, color="black", linestyle="--", lw=1.5, zorder=5)
        label_kw = dict(fontsize=10, zorder=6, path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.text(0, 1.02, "Rod", ha="center", **label_kw)
        ax.text(1, 1.02, "Sphere", ha="center", **label_kw)
        ax.text(0.5, 0.48, "Disk", ha="center", **label_kw)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(0.45, 1.05)
        ax.set_xlabel(r"$I_1/I_3$"); ax.set_ylabel(r"$I_2/I_3$")

    def _overlay_refs(ax):
        for name, (xr, yr) in ref_ratios.items():
            ax.scatter(xr, yr, marker=REF_MARKERS[name], s=150, c=REF_COLORS[name], 
                       edgecolors="black", zorder=10, label=name)

    titles = ["AF2 Shape Space", "MD Shape Space"]
    coords = [(x1, y1), (x2, y2)]
    colors = ["#6a51a3", "#238b45"]

    for i, ax in enumerate(axes):
        _draw_triangle(ax)
        xx, yy = coords[i]
        ax.scatter(xx, yy, s=2, alpha=0.3, c=colors[i], rasterized=True, label="Ensemble")
        _overlay_refs(ax)
        ax.set_title(titles[i])
        if i == 0: ax.legend(loc="lower right", fontsize=8)
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "shape_space_comparison.png", dpi=300)
    plt.close(fig)
    print("Saved shape_space_comparison.png")


def plot_biophysical_panels(pca_coords1, pca_coords2, data1, data2,
                           ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by biophysical features in a 2-row grid.

    Ensemble 1 (AF2) on top row, Ensemble 2 (MD) on bottom row.
    Columns: N-C dist, NAC prot, p2 prot, Termini contacts, C-tail prot.
    """
    labels = ["AF2", "MD"]
    features = [
        ("nc_dist", "N-C distance", "bone", "N-C distance (Å)"),
        ("nac_prot", "NAC protection", "plasma", "NAC mean log P$_f$"),
        ("p2_prot", "p2 protection", "plasma", "p2 mean log P$_f$"),
        ("termini_contacts", "N–C contacts", "viridis", "N–C contacts"),
        ("ctail_prot", "C-tail protection", "plasma", "C-tail mean log P$_f$")
    ]
    n_cols = len(features)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), constrained_layout=True)

    # Calculate shared limits
    xlim, ylim = get_pca_limits([pca_coords1, pca_coords2], {**ref_positions, **dynamic_ref_positions})

    def _overlay_refs(ax, is_legend=False):
        for name in REFERENCES.keys():
            pos = ref_positions[name]
            ax.scatter(
                pos[0], pos[1],
                marker=REF_MARKERS[name], s=200,
                c=REF_COLORS[name], zorder=10,
                edgecolors="black", linewidths=1,
                label=name if is_legend else None
            )
        for name, pos_array in dynamic_ref_positions.items():
            ax.scatter(
                pos_array[:, 0], pos_array[:, 1],
                marker=DYN_REF_MARKERS.get(name, "o"), s=50,
                c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
                edgecolors="white", linewidths=0.5,
                label=name if is_legend else None
            )

    # Row 0: Ensemble 1
    for col_idx, (key, title, cmap, cb_label) in enumerate(features):
        ax = axes[0, col_idx]
        vals = data1[key]
        if key == "nc_dist":
            vmax = np.percentile(vals, 99)
            vmin = 0
        elif "prot" in key:
            # Shared scale for protection? Maybe better to let them be independent or auto
            vmin, vmax = np.min(vals), np.max(vals)
        else:
            vmin, vmax = None, None

        scatter = ax.scatter(
            pca_coords1[:, 0], pca_coords1[:, 1],
            c=vals, cmap=cmap, s=2, alpha=0.4,
            linewidths=0, rasterized=True,
            vmin=vmin, vmax=vmax
        )
        _overlay_refs(ax, is_legend=(col_idx == 0))
        add_colorbar(fig, ax, scatter, cb_label)
        ax.set_title(f"{labels[0]}: {title}")
        ax.set_xlabel("PC1")
        if col_idx == 0:
            ax.set_ylabel("PC2", fontweight="bold")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        else:
            ax.set_ylabel("PC2")
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    # Row 1: Ensemble 2
    for col_idx, (key, title, cmap, cb_label) in enumerate(features):
        ax = axes[1, col_idx]
        vals = data2[key]
        if key == "nc_dist":
            vmax = np.percentile(vals, 99) # maybe use shared vmax?
            vmin = 0
        else:
            vmin, vmax = None, None

        scatter = ax.scatter(
            pca_coords2[:, 0], pca_coords2[:, 1],
            c=vals, cmap=cmap, s=2, alpha=0.4,
            linewidths=0, rasterized=True,
            vmin=vmin, vmax=vmax
        )
        _overlay_refs(ax)
        add_colorbar(fig, ax, scatter, cb_label)
        ax.set_title(f"{labels[1]}: {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "pca_biophysical.png", dpi=300)
    plt.close(fig)
    print("Saved pca_biophysical.png")


def plot_radgyr_pca_panels(pca_coords1, pca_coords2, data1, data2,
                          ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by region RadGyr in a 2-row grid."""
    labels = ["AF2", "MD"]
    features = [
        ("nhead_rg", "N-head RadGyr", "viridis", "N-head RadGyr (Å)"),
        ("nac_rg", "NAC RadGyr", "plasma", "NAC RadGyr (Å)"),
        ("ctail_rg", "C-tail RadGyr", "inferno", "C-tail RadGyr (Å)")
    ]
    n_cols = len(features)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), constrained_layout=True)

    # Calculate shared limits
    xlim, ylim = get_pca_limits([pca_coords1, pca_coords2], {**ref_positions, **dynamic_ref_positions})

    def _overlay_refs(ax, is_legend=False):
        for name in REFERENCES.keys():
            pos = ref_positions[name]
            ax.scatter(
                pos[0], pos[1],
                marker=REF_MARKERS[name], s=200,
                c=REF_COLORS[name], zorder=10,
                edgecolors="black", linewidths=1,
                label=name if is_legend else None
            )
        for name, pos_array in dynamic_ref_positions.items():
            ax.scatter(
                pos_array[:, 0], pos_array[:, 1],
                marker=DYN_REF_MARKERS.get(name, "o"), s=50,
                c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
                edgecolors="white", linewidths=0.5,
                label=name if is_legend else None
            )

    # Row 0: Ensemble 1
    for col_idx, (key, title, cmap, cb_label) in enumerate(features):
        ax = axes[0, col_idx]
        vals = data1[key]
        scatter = ax.scatter(
            pca_coords1[:, 0], pca_coords1[:, 1],
            c=vals, cmap=cmap, s=2, alpha=0.4,
            linewidths=0, rasterized=True
        )
        _overlay_refs(ax, is_legend=(col_idx == 0))
        add_colorbar(fig, ax, scatter, cb_label)
        ax.set_title(f"{labels[0]}: {title}")
        ax.set_xlabel("PC1")
        if col_idx == 0:
            ax.set_ylabel("PC2", fontweight="bold")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
        else:
            ax.set_ylabel("PC2")
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    # Row 1: Ensemble 2
    for col_idx, (key, title, cmap, cb_label) in enumerate(features):
        ax = axes[1, col_idx]
        vals = data2[key]
        scatter = ax.scatter(
            pca_coords2[:, 0], pca_coords2[:, 1],
            c=vals, cmap=cmap, s=2, alpha=0.4,
            linewidths=0, rasterized=True
        )
        _overlay_refs(ax)
        add_colorbar(fig, ax, scatter, cb_label)
        ax.set_title(f"{labels[1]}: {title}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "pca_radgyr_hues.png", dpi=300)
    plt.close(fig)
    print("Saved pca_radgyr_hues.png")


def plot_rmsd_panels(pca_coords1, pca_coords2, rmsd_dict1, rmsd_dict2,
                     ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by RMSD in a 2-row grid.
    
    Row 0: Ensemble 1, Row 1: Ensemble 2.
    Columns for each reference structure.
    """
    labels = ["AF2", "MD"]
    n_refs = len(rmsd_dict1)
    fig, axes = plt.subplots(2, n_refs, figsize=(5 * n_refs, 8), constrained_layout=True)
    if n_refs == 1: axes = axes.reshape(2, 1)

    # Calculate shared limits
    xlim, ylim = get_pca_limits([pca_coords1, pca_coords2], {**ref_positions, **dynamic_ref_positions})

    def _overlay_ref(ax, name):
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

    # Row 0: Ensemble 1
    for col_idx, (name, rmsd_vals) in enumerate(rmsd_dict1.items()):
        ax = axes[0, col_idx]
        vmax = np.percentile(rmsd_vals, 99)
        scatter = ax.scatter(
            pca_coords1[:, 0], pca_coords1[:, 1],
            c=rmsd_vals, cmap="cividis", s=2, alpha=0.4,
            linewidths=0, rasterized=True,
            vmin=0, vmax=vmax
        )
        _overlay_ref(ax, name)
        add_colorbar(fig, ax, scatter, "Cα RMSD (Å)")
        ax.set_title(f"{labels[0]}: {name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    # Row 1: Ensemble 2
    for col_idx, (name, rmsd_vals) in enumerate(rmsd_dict2.items()):
        ax = axes[1, col_idx]
        vmax = np.percentile(rmsd_vals, 99)
        scatter = ax.scatter(
            pca_coords2[:, 0], pca_coords2[:, 1],
            c=rmsd_vals, cmap="cividis", s=2, alpha=0.4,
            linewidths=0, rasterized=True,
            vmin=0, vmax=vmax
        )
        _overlay_ref(ax, name)
        add_colorbar(fig, ax, scatter, "Cα RMSD (Å)")
        ax.set_title(f"{labels[1]}: {name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "pca_rmsd_references.png", dpi=300)
    plt.close(fig)
    print("Saved pca_rmsd_references.png")


def plot_drmsd_panels(pca_coords1, pca_coords2, drmsd_dict1, drmsd_dict2,
                      ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by Dihedral RMSD in a 2-row grid.
    
    Row 0: Ensemble 1, Row 1: Ensemble 2.
    Columns for each reference structure.
    """
    labels = ["AF2", "MD"]
    n_refs = len(drmsd_dict1)
    fig, axes = plt.subplots(2, n_refs, figsize=(5 * n_refs, 8), constrained_layout=True)
    if n_refs == 1: axes = axes.reshape(2, 1)

    # Calculate shared limits
    xlim, ylim = get_pca_limits([pca_coords1, pca_coords2], {**ref_positions, **dynamic_ref_positions})

    def _overlay_ref(ax, name):
         # Mark minimum-RMSD frame position (using same ref positions as RMSD)
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

    # Row 0: Ensemble 1
    for col_idx, (name, drmsd_vals) in enumerate(drmsd_dict1.items()):
        ax = axes[0, col_idx]
        vmax = np.percentile(drmsd_vals, 99)
        scatter = ax.scatter(
            pca_coords1[:, 0], pca_coords1[:, 1],
            c=drmsd_vals, cmap="cividis", s=2, alpha=0.4,
            linewidths=0, rasterized=True,
            vmin=0, vmax=vmax
        )
        _overlay_ref(ax, name)
        add_colorbar(fig, ax, scatter, "Dihedral RMSD (°)")
        ax.set_title(f"{labels[0]}: {name} (dRMSD)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    # Row 1: Ensemble 2
    for col_idx, (name, drmsd_vals) in enumerate(drmsd_dict2.items()):
        ax = axes[1, col_idx]
        vmax = np.percentile(drmsd_vals, 99)
        scatter = ax.scatter(
            pca_coords2[:, 0], pca_coords2[:, 1],
            c=drmsd_vals, cmap="cividis", s=2, alpha=0.4,
            linewidths=0, rasterized=True,
            vmin=0, vmax=vmax
        )
        _overlay_ref(ax, name)
        add_colorbar(fig, ax, scatter, "Dihedral RMSD (°)")
        ax.set_title(f"{labels[1]}: {name} (dRMSD)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "pca_drmsd_references.png", dpi=300)
    plt.close(fig)
    print("Saved pca_drmsd_references.png")


def plot_plddt_panel(pca_coords, plddt_vals, ref_positions, dynamic_ref_positions, plots_dir):
    """Plot PCA colored by pLDDT (AF2 confidence metric).

    Creates a single-panel figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Calculate shared limits (including coordinates and refs)
    xlim, ylim = get_pca_limits([pca_coords], {**ref_positions, **dynamic_ref_positions})

    scatter = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=plddt_vals, cmap="BuPu", s=5, alpha=1-(plddt_vals/100),
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
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    remove_top_right_spines(ax)

    # Save figure
    fig.savefig(plots_dir / "pca_plddt.png", dpi=300)
    plt.close(fig)
    print("Saved pca_plddt.png")


def plot_rmsd_scatter_combinations(data1, data2, plots_dir):
    """Plot 2D scatter plots for each pairing of reference structure RMSDs."""
    labels = ["AF2", "MD"]
    colors = ["#4393c3", "#d6604d"]
    alpha = 0.3

    rmsd_names = list(data1["rmsd_dict"].keys())
    combinations = list(itertools.combinations(rmsd_names, 2))
    n_cols = len(combinations)
    
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for row_idx, (label, data, color) in enumerate(zip(labels, [data1, data2], colors)):
        for col_idx, (name_a, name_b) in enumerate(combinations):
            ax = axes[row_idx, col_idx]
            
            rmsd_a = data["rmsd_dict"][name_a]
            rmsd_b = data["rmsd_dict"][name_b]
            hue_vals = data["ctail_rg"]
            
            scatter = ax.scatter(rmsd_a, rmsd_b, c=hue_vals, cmap="inferno", alpha=alpha, s=1, rasterized=True)
            
            # Add density contours
            sns.kdeplot(x=rmsd_a, y=rmsd_b, ax=ax, levels=5, color="black", linewidths=0.8, alpha=0.5)

            # Highlight regions: nearest RMSD + 10 Å for this ensemble
            min_a = np.min(rmsd_a)
            min_b = np.min(rmsd_b)
            
            limit_a = min_a + 10.0
            limit_b = min_b + 10.0
            
            ax.axvline(limit_a, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(limit_b, color="black", linestyle="--", linewidth=1, alpha=0.5)
            
            ax.axvspan(min_a, limit_a, color="gray", alpha=0.1)
            ax.axhspan(min_b, limit_b, color="gray", alpha=0.1)
            
            ax.set_xlabel(f"RMSD to {name_a} (Å)")
            ax.set_ylabel(f"RMSD to {name_b} (Å)")
            ax.set_title(f"{label}: {name_a} vs {name_b}")
            add_colorbar(fig, ax, scatter, "C-tail RadGyr (Å)")
            remove_top_right_spines(ax)

    fig.savefig(plots_dir / "rmsd_scatter_combinations.png", dpi=300)
    plt.close(fig)
    print("Saved rmsd_scatter_combinations.png")


def plot_drmsd_scatter_combinations(data1, data2, plots_dir):
    """Plot 2D scatter plots for each pairing of reference structure dRMSDs."""
    labels = ["AF2", "MD"]
    colors = ["#4393c3", "#d6604d"]
    alpha = 0.3

    drmsd_names = list(data1["drmsd_dict"].keys())
    combinations = list(itertools.combinations(drmsd_names, 2))
    n_cols = len(combinations)
    
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for row_idx, (label, data, color) in enumerate(zip(labels, [data1, data2], colors)):
        for col_idx, (name_a, name_b) in enumerate(combinations):
            ax = axes[row_idx, col_idx]
            
            drmsd_a = data["drmsd_dict"][name_a]
            drmsd_b = data["drmsd_dict"][name_b]
            hue_vals = data["ctail_rg"]
            
            scatter = ax.scatter(drmsd_a, drmsd_b, c=hue_vals, cmap="inferno", alpha=alpha, s=1, rasterized=True)
            
            # Add density contours
            sns.kdeplot(x=drmsd_a, y=drmsd_b, ax=ax, levels=5, color="black", linewidths=0.8, alpha=0.5)

            # Highlight regions: nearest dRMSD + 15° for this ensemble
            min_a = np.min(drmsd_a)
            min_b = np.min(drmsd_b)
            
            limit_a = min_a + 15.0
            limit_b = min_b + 15.0
            
            ax.axvline(limit_a, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(limit_b, color="black", linestyle="--", linewidth=1, alpha=0.5)
            
            ax.axvspan(min_a, limit_a, color="gray", alpha=0.1)
            ax.axhspan(min_b, limit_b, color="gray", alpha=0.1)

            ax.set_xlabel(f"dRMSD to {name_a} (°)")
            ax.set_ylabel(f"dRMSD to {name_b} (°)")
            ax.set_title(f"{label}: {name_a} vs {name_b} (Dihedral)")
            add_colorbar(fig, ax, scatter, "C-tail RadGyr (Å)")
            remove_top_right_spines(ax)

    fig.savefig(plots_dir / "drmsd_scatter_combinations.png", dpi=300)
    plt.close(fig)
    print("Saved drmsd_scatter_combinations.png")


def plot_rmsd_vs_drmsd_scatter(data1, data2, plots_dir):
    """Plot Cartesian RMSD vs Dihedral RMSD for each reference."""
    labels = ["AF2", "MD"]
    n_refs = len(data1["rmsd_dict"])
    fig, axes = plt.subplots(2, n_refs, figsize=(5 * n_refs, 10), constrained_layout=True)
    if n_refs == 1: axes = axes.reshape(2, 1)

    for row_idx, (label, data) in enumerate(zip(labels, [data1, data2])):
        for col_idx, (name, rmsd_vals) in enumerate(data["rmsd_dict"].items()):
            ax = axes[row_idx, col_idx]
            drmsd_vals = data["drmsd_dict"][name]
            hue_vals = data["nac_rg"]
            
            scatter = ax.scatter(rmsd_vals, drmsd_vals, c=hue_vals, cmap="inferno", alpha=0.4, s=2, rasterized=True)
            
            # Add density contours
            sns.kdeplot(x=rmsd_vals, y=drmsd_vals, ax=ax, levels=5, color="black", linewidths=0.8, alpha=0.5)

            # Highlight regions: nearest RMSD + 10 Å and nearest dRMSD + 15°
            min_rmsd = np.min(rmsd_vals)
            min_drmsd = np.min(drmsd_vals)
            
            limit_rmsd = min_rmsd + 10.0
            limit_drmsd = min_drmsd + 15.0
            
            ax.axvline(limit_rmsd, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(limit_drmsd, color="black", linestyle="--", linewidth=1, alpha=0.5)
            
            ax.axvspan(min_rmsd, limit_rmsd, color="gray", alpha=0.1)
            ax.axhspan(min_drmsd, limit_drmsd, color="gray", alpha=0.1)

            ax.set_xlabel(f"Cartesian RMSD to {name} (Å)")
            ax.set_ylabel(f"Dihedral RMSD to {name} (°)")
            ax.set_title(f"{label}: {name} RMSD vs dRMSD")
            add_colorbar(fig, ax, scatter, "NAC RadGyr (Å)")
            remove_top_right_spines(ax)

    fig.savefig(plots_dir / "rmsd_vs_drmsd_scatter.png", dpi=300)
    plt.close(fig)
    print("Saved rmsd_vs_drmsd_scatter.png")


def plot_radgyr_rmsd_scatter(data1, data2, plots_dir):
    """Plot RadGyr region comparisons, colored by RMSD to references."""
    labels = ["AF2", "MD"]
    n_refs = len(data1["rmsd_dict"])
    # 4 rows: 2 for (Nhead vs NAC), 2 for (NAC vs Ctail)
    fig, axes = plt.subplots(4, n_refs, figsize=(5 * n_refs, 16), constrained_layout=True)
    if n_refs == 1: axes = axes.reshape(4, 1)

    for row_offset, (comp_name, x_key, y_key) in enumerate([
        ("N-head vs NAC", "nhead_rg", "nac_rg"),
        ("NAC vs C-tail", "nac_rg", "ctail_rg")
    ]):
        for row_idx, (label, data) in enumerate(zip(labels, [data1, data2])):
            curr_row = row_offset * 2 + row_idx
            rg_x = data[x_key]
            rg_y = data[y_key]
            
            for col_idx, (ref_name, rmsd_vals) in enumerate(data["rmsd_dict"].items()):
                ax = axes[curr_row, col_idx]
                vmax = np.percentile(rmsd_vals, 99)
                scatter = ax.scatter(rg_x, rg_y, c=rmsd_vals, cmap="cividis", s=2, alpha=0.4,
                                     vmin=0, vmax=vmax, rasterized=True)
                
                ax.set_xlabel(f"{x_key.split('_')[0].upper()} RadGyr (Å)")
                ax.set_ylabel(f"{y_key.split('_')[0].upper()} RadGyr (Å)")
                ax.set_title(f"{label} {comp_name}: Hue by RMSD to {ref_name}")
                add_colorbar(fig, ax, scatter, "RMSD (Å)")
                remove_top_right_spines(ax)

    fig.savefig(plots_dir / "radgyr_region_comparisons_rmsd_hue.png", dpi=300)
    plt.close(fig)
    print("Saved radgyr_region_comparisons_rmsd_hue.png")


def plot_rmsd_vs_nac_rg(data1, data2, plots_dir):
    """Plot RMSD to reference (x) vs NAC RadGyr (y), hued by C-tail RadGyr."""
    labels = ["AF2", "MD"]
    n_refs = len(data1["rmsd_dict"])
    fig, axes = plt.subplots(2, n_refs, figsize=(5 * n_refs, 10), constrained_layout=True)
    if n_refs == 1: axes = axes.reshape(2, 1)

    for row_idx, (label, data) in enumerate(zip(labels, [data1, data2])):
        nac_rg = data["nac_rg"]
        ctail_rg = data["ctail_rg"]
        
        for col_idx, (ref_name, rmsd_vals) in enumerate(data["rmsd_dict"].items()):
            ax = axes[row_idx, col_idx]
            scatter = ax.scatter(rmsd_vals, nac_rg, c=ctail_rg, cmap="inferno", s=2, alpha=0.5, rasterized=True)
            
            ax.set_xlabel(f"RMSD to {ref_name} (Å)")
            ax.set_ylabel("NAC RadGyr (Å)")
            ax.set_title(f"{label}: {ref_name}")
            add_colorbar(fig, ax, scatter, "C-tail RadGyr (Å)")
            remove_top_right_spines(ax)

    fig.savefig(plots_dir / "rmsd_vs_nac_rg_ctail_hue.png", dpi=300)
    plt.close(fig)
    print("Saved rmsd_vs_nac_rg_ctail_hue.png")


def plot_radgyr_scatter_combinations(data1, data2, plots_dir):
    """Plot 2D scatter plots for each pairing of region RadGyr."""
    labels = ["AF2", "MD"]
    colors = ["#4393c3", "#d6604d"]
    alpha = 0.3

    features = ["nhead_rg", "nac_rg", "ctail_rg"]
    feature_labels = {
        "nhead_rg": "N-head RadGyr (Å)",
        "nac_rg": "NAC RadGyr (Å)",
        "ctail_rg": "C-tail RadGyr (Å)"
    }
    combinations = list(itertools.combinations(features, 2))
    n_cols = len(combinations)
    
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for row_idx, (label, data, color) in enumerate(zip(labels, [data1, data2], colors)):
        for col_idx, (feat_a, feat_b) in enumerate(combinations):
            ax = axes[row_idx, col_idx]
            
            val_a = data[feat_a]
            val_b = data[feat_b]
            
            feat_c = next(f for f in features if f not in (feat_a, feat_b))
            hue_vals = data[feat_c]
            
            scatter = ax.scatter(val_a, val_b, c=hue_vals, cmap="plasma", alpha=alpha, s=5, rasterized=True)
            
            ax.set_xlabel(feature_labels[feat_a])
            ax.set_ylabel(feature_labels[feat_b])
            ax.set_title(f"{label}: {feat_a.split('_')[0].upper()} vs {feat_b.split('_')[0].upper()}")
            add_colorbar(fig, ax, scatter, feature_labels[feat_c])
            remove_top_right_spines(ax)

    fig.savefig(plots_dir / "radgyr_scatter_combinations.png", dpi=300)
    plt.close(fig)
    print("Saved radgyr_scatter_combinations.png")


def plot_secondary_structure_scatter(data1, data2, plots_dir):
    """Plot 2D scatter plot for alpha vs beta content across 4 regions."""
    labels = ["AF2", "MD"]
    regions = [
        ("whole", "Whole Sequence"),
        ("nhead", "N-head (1-60)"),
        ("nac", "NAC (61-95)"),
        ("ctail", "C-tail (96-140)")
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)

    max_alpha = 0
    max_beta = 0
    for data in [data1, data2]:
        for reg_key, _ in regions:
            max_alpha = max(max_alpha, np.max(data["ss"][reg_key][0] * 100))
            max_beta = max(max_beta, np.max(data["ss"][reg_key][1] * 100))

    for row_idx, (label, data) in enumerate(zip(labels, [data1, data2])):
        for col_idx, (reg_key, reg_name) in enumerate(regions):
            ax = axes[row_idx, col_idx]
            
            alpha_frac = data["ss"][reg_key][0] * 100
            beta_frac = data["ss"][reg_key][1] * 100
            hue_vals = data["ctail_rg"]
            
            scatter = ax.scatter(beta_frac, alpha_frac, c=hue_vals, cmap="magma", alpha=0.5, s=10, rasterized=True)
            
            ax.set_xlabel("Beta Content (%)")
            ax.set_ylabel("Alpha Content (%)")
            ax.set_title(f"{label}: {reg_name}")
            
            ax.set_xlim(-1, max_beta + 5)
            ax.set_ylim(-1, max_alpha + 5)
            
            if col_idx == 3:  # Only add colorbar to the last column
                add_colorbar(fig, ax, scatter, "C-tail RadGyr (Å)")
            remove_top_right_spines(ax)

    fig.savefig(plots_dir / "secondary_structure_scatter.png", dpi=300)
    plt.close(fig)
    print("Saved secondary_structure_scatter.png")


def plot_feature_histograms(data1, data2, plots_dir):
    """Plot feature distribution histograms for two ensembles.

    Creates three figures:
      1. Biophysical features: nc_dist, nac_prot, p2_prot, termini_contacts, ctail_prot
      2. RMSD to each reference
      3. pLDDT (Ensemble 1 only)
    """
    labels = ["AF2", "MD"]
    colors = ["#4393c3", "#d6604d"]  # Blue for AF2, Red for MD
    alpha = 0.5

    # -------- Figure 1: Biophysical features --------
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    
    features = [
        ("nc_dist", "N–C Distance (Å)", "N-C Terminus Distance"),
        ("nac_prot", "NAC Mean Log P$_f$", "NAC Region Protection (61–95)"),
        ("p2_prot", "p2 Mean Log P$_f$", "p2 Motif Protection (45–57)"),
        ("termini_contacts", "Contact Count", "N–C Contacts (1-60 vs 96-140)"),
        ("ctail_prot", "C-tail Mean Log P$_f$", "C-tail Protection (115–135)")
    ]

    for ax, (key, xlabel, title) in zip(axes, features):
        vals1 = data1[key]
        vals2 = data2[key]
        
        # Determine common bins
        all_vals = np.concatenate([vals1, vals2])
        bins = np.linspace(np.min(all_vals), np.max(all_vals), 50)
        
        ax.hist(vals1, bins=bins, density=True, alpha=alpha, color=colors[0], 
                edgecolor="black", linewidth=0.5, label=labels[0])
        ax.hist(vals2, bins=bins, density=True, alpha=alpha, color=colors[1], 
                edgecolor="black", linewidth=0.5, label=labels[1])
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        if ax == axes[0]:
            ax.legend()
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "feature_distributions_biophysical.png", dpi=300)
    plt.close(fig)
    print("Saved feature_distributions_biophysical.png")

    # -------- Figure 2: RMSD to references --------
    n_refs = len(data1["rmsd_dict"])
    fig, axes = plt.subplots(1, n_refs, figsize=(4 * n_refs, 4), constrained_layout=True)
    if n_refs == 1: axes = [axes]

    for ax, name in zip(axes, data1["rmsd_dict"].keys()):
        rmsd1 = data1["rmsd_dict"][name]
        rmsd2 = data2["rmsd_dict"][name]
        
        bins = np.linspace(0, max(np.max(rmsd1), np.max(rmsd2)), 50)
        
        ax.hist(rmsd1, bins=bins, density=True, alpha=alpha, color=colors[0],
                edgecolor="black", linewidth=0.5, label=labels[0])
        ax.hist(rmsd2, bins=bins, density=True, alpha=alpha, color=colors[1],
                edgecolor="black", linewidth=0.5, label=labels[1])
        
        ax.set_xlabel("RMSD to {} (Å)".format(name))
        ax.set_ylabel("Density")
        ax.set_title("RMSD to {}".format(name))
        if ax == axes[0]:
            ax.legend()
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "feature_distributions_rmsd.png", dpi=300)
    plt.close(fig)
    print("Saved feature_distributions_rmsd.png")

    # -------- Figure 3: pLDDT (AF2 only) --------
    if "plddt" in data1 and data1["plddt"] is not None:
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        ax.hist(data1["plddt"], bins=50, density=True, alpha=0.8, color="#6a51a3", 
                edgecolor="black", linewidth=0.5)
        ax.set_xlabel("pLDDT")
        ax.set_ylabel("Density")
        ax.set_title("pLDDT Distribution (AF2)")
        remove_top_right_spines(ax)
        fig.savefig(plots_dir / "feature_distributions_plddt.png", dpi=300)
        plt.close(fig)
        print("Saved feature_distributions_plddt.png")

    # -------- Figure 4: dRMSD to references --------
    n_refs = len(data1["drmsd_dict"])
    fig, axes = plt.subplots(1, n_refs, figsize=(4 * n_refs, 4), constrained_layout=True)
    if n_refs == 1: axes = [axes]

    for ax, name in zip(axes, data1["drmsd_dict"].keys()):
        drmsd1 = data1["drmsd_dict"][name]
        drmsd2 = data2["drmsd_dict"][name]
        
        bins = np.linspace(0, max(np.max(drmsd1), np.max(drmsd2)), 50)
        
        ax.hist(drmsd1, bins=bins, density=True, alpha=alpha, color=colors[0],
                edgecolor="black", linewidth=0.5, label=labels[0])
        ax.hist(drmsd2, bins=bins, density=True, alpha=alpha, color=colors[1],
                edgecolor="black", linewidth=0.5, label=labels[1])
        
        ax.set_xlabel("dRMSD to {} (°)".format(name))
        ax.set_ylabel("Density")
        ax.set_title("dRMSD to {}".format(name))
        if ax == axes[0]:
            ax.legend()
        remove_top_right_spines(ax)

    fig.savefig(plots_dir / "feature_distributions_drmsd.png", dpi=300)
    plt.close(fig)
    print("Saved feature_distributions_drmsd.png")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare two aSyn ensembles.")
    # Ensemble 1 (AF2)
    parser.add_argument("--top1", type=Path, default=DEFAULT_TOP1, help="Topology for ensemble 1")
    parser.add_argument("--traj1", type=Path, default=DEFAULT_TRAJ1, help="Trajectory for ensemble 1")
    parser.add_argument("--feat1", type=Path, default=DEFAULT_FEAT1, help="Features for ensemble 1")
    parser.add_argument("--topo1", type=Path, default=DEFAULT_TOPO1, help="Topology JSON for ensemble 1")
    parser.add_argument("--plddt", type=Path, default=DEFAULT_PLDDT_FILE, help="pLDDT info for ensemble 1")
    
    # Ensemble 2 (MD)
    parser.add_argument("--top2", type=Path, default=DEFAULT_TOP2, help="Topology for ensemble 2")
    parser.add_argument("--traj2", type=Path, default=DEFAULT_TRAJ2, help="Trajectory for ensemble 2")
    parser.add_argument("--feat2", type=Path, default=DEFAULT_FEAT2, help="Features for ensemble 2")
    parser.add_argument("--topo2", type=Path, default=DEFAULT_TOPO2, help="Topology JSON for ensemble 2")
    
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_EXP_DIR / "plots_comparison", help="Output plots directory")
    args = parser.parse_args()

    set_publication_style()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data for Ensemble 1
    print("Loading Ensemble 1 (AF2) data...")
    resid_to_idx1 = load_topology_map(args.topo1)
    log_pf1 = load_log_pf(args.feat1)
    plddt1 = load_plddt_values(args.plddt) if args.plddt.exists() else None
    
    ss1 = compute_secondary_structure(args.top1, args.traj1)
    
    data1 = {
        "nc_dist": compute_nc_distances(args.top1, args.traj1, resid_to_idx1),
        "nac_prot": compute_region_mean_log_pf(log_pf1, resid_to_idx1, NAC_RANGE),
        "p2_prot": compute_region_mean_log_pf(log_pf1, resid_to_idx1, P2_RANGE),
        "ctail_prot": compute_region_mean_log_pf(log_pf1, resid_to_idx1, CTAIL_RANGE),
        "termini_contacts": compute_termini_contacts(args.top1, args.traj1, T1_RANGE, T2_RANGE),
        "nhead_rg": compute_radgyr(args.top1, args.traj1, NHEAD_RANGE),
        "nac_rg": compute_radgyr(args.top1, args.traj1, NAC_RANGE),
        "ctail_rg": compute_radgyr(args.top1, args.traj1, CTAIL_RANGE),
        "ss": ss1,
        "plddt": plddt1
    }

    # 2. Load Data for Ensemble 2
    print("Loading Ensemble 2 (MD) data...")
    resid_to_idx2 = load_topology_map(args.topo2)
    log_pf2 = load_log_pf(args.feat2)
    
    ss2 = compute_secondary_structure(args.top2, args.traj2)
    
    data2 = {
        "nc_dist": compute_nc_distances(args.top2, args.traj2, resid_to_idx2),
        "nac_prot": compute_region_mean_log_pf(log_pf2, resid_to_idx2, NAC_RANGE),
        "p2_prot": compute_region_mean_log_pf(log_pf2, resid_to_idx2, P2_RANGE),
        "ctail_prot": compute_region_mean_log_pf(log_pf2, resid_to_idx2, CTAIL_RANGE),
        "termini_contacts": compute_termini_contacts(args.top2, args.traj2, T1_RANGE, T2_RANGE),
        "nhead_rg": compute_radgyr(args.top2, args.traj2, NHEAD_RANGE),
        "nac_rg": compute_radgyr(args.top2, args.traj2, NAC_RANGE),
        "ctail_rg": compute_radgyr(args.top2, args.traj2, CTAIL_RANGE),
        "ss": ss2
    }

    # 3. Fit PCA on Combined Ensemble
    print("Fitting PCA on combined ensemble...")
    u1 = mda.Universe(str(args.top1), str(args.traj1))
    u2 = mda.Universe(str(args.top2), str(args.traj2))
    
    # We need a shared atom selection. AF2 and MD should have same resids
    ca1 = u1.select_atoms(PCA_ATOM_SELECTION)
    ca2 = u2.select_atoms(PCA_ATOM_SELECTION)
    n_at = len(ca1)
    assert len(ca1) == len(ca2), f"Atom count mismatch: {len(ca1)} vs {len(ca2)}"
    
    # Collect collective distances (sampled to avoid memory issues)
    def _get_distances(u, stride=10):
        pos_list = []
        sel = u.select_atoms(PCA_ATOM_SELECTION)
        for ts in u.trajectory[::stride]:
            pos_list.append(pdist(sel.positions, metric="euclidean"))
        return np.array(pos_list)

    print("  Extracting distances from Traj 1...")
    dists1 = _get_distances(u1, stride=5)
    print("  Extracting distances from Traj 2...")
    dists2 = _get_distances(u2, stride=5)
    
    all_dists = np.vstack([dists1, dists2])
    pca_model = PCA(n_components=20)
    pca_model.fit(all_dists)
    print(f"  PCA fitted on {len(all_dists)} frames")

    # Project both full trajectories
    def _project_full(u, model):
        coords = []
        sel = u.select_atoms(PCA_ATOM_SELECTION)
        for ts in u.trajectory:
            d = pdist(sel.positions, metric="euclidean")
            coords.append(model.transform(d.reshape(1, -1))[0])
        return np.array(coords)

    print("  Projecting Traj 1...")
    pca_coords1 = _project_full(u1, pca_model)
    print("  Projecting Traj 2...")
    pca_coords2 = _project_full(u2, pca_model)

    # 4. Handle References
    rmsd_dict1 = {}
    rmsd_dict2 = {}
    drmsd_dict1 = {}
    drmsd_dict2 = {}
    ref_positions = {}
    traj_ca_resids = ca1.resids.copy()

    for name, ref_pdb in REFERENCES.items():
        print(f"Processing reference {name}...")
        rmsd_dict1[name] = compute_rmsd_to_reference(args.top1, args.traj1, ref_pdb)
        rmsd_dict2[name] = compute_rmsd_to_reference(args.top2, args.traj2, ref_pdb)
        
        print(f"Computing dihedral RMSD to reference {name}...")
        drmsd_dict1[name] = compute_drmsd_to_reference(args.top1, args.traj1, ref_pdb)
        drmsd_dict2[name] = compute_drmsd_to_reference(args.top2, args.traj2, ref_pdb)
        
        ref = mda.Universe(str(ref_pdb))
        ref_ca = ref.select_atoms(PCA_ATOM_SELECTION)
        if ref_ca.n_atoms == n_at and np.array_equal(ref_ca.resids, traj_ca_resids):
            dists = pdist(ref_ca.positions, metric="euclidean")
            ref_positions[name] = pca_model.transform(dists.reshape(1, -1))[0]
        else:
            # Fallback to nearest-RMSD in combined? 
            # Let's just use nearest in AF2 for now as representative
            ref_positions[name] = find_ref_pca_position(rmsd_dict1[name], pca_coords1)

    data1["rmsd_dict"] = rmsd_dict1
    data2["rmsd_dict"] = rmsd_dict2
    data1["drmsd_dict"] = drmsd_dict1
    data2["drmsd_dict"] = drmsd_dict2

    dynamic_ref_positions = {}
    for name, ref_pdb in DYNAMIC_REFERENCES.items():
        print(f"Projecting dynamic reference {name}...")
        ref = mda.Universe(str(ref_pdb))
        sel_str = f"{PCA_ATOM_SELECTION} and resid " + " ".join(map(str, traj_ca_resids))
        ref_ca = ref.select_atoms(sel_str)
        if ref_ca.n_atoms == n_at:
            pos_list = []
            for ts in ref.trajectory:
                dists = pdist(ref_ca.positions, metric="euclidean")
                pos_list.append(pca_model.transform(dists.reshape(1, -1))[0])
            dynamic_ref_positions[name] = np.array(pos_list)

    # 4b. Handle Shape Space Ratios
    print("Computing inertia ratios for shape space...")
    shape_sel = "resid 1-96 and name CA" # Consistent with inertia_moments_clustering.py
    x_ratio1, y_ratio1 = compute_inertia_ratios_simple(args.top1, args.traj1, shape_sel)
    x_ratio2, y_ratio2 = compute_inertia_ratios_simple(args.top2, args.traj2, shape_sel)
    ref_ratios_shape = compute_reference_inertia_ratios(REFERENCES, shape_sel)

    # 5. Plotting
    print("Plotting Figure S: Shape space panels...")
    plot_shape_space_panels(
        x_ratio1, y_ratio1, x_ratio2, y_ratio2, 
        ref_ratios_shape, args.plots_dir
    )

    print("Plotting Figure 1: Biophysical PCA panels...")
    plot_biophysical_panels(
        pca_coords1, pca_coords2, data1, data2,
        ref_positions, dynamic_ref_positions, args.plots_dir
    )

    print("Plotting Figure 1b: RadGyr PCA panels...")
    plot_radgyr_pca_panels(
        pca_coords1, pca_coords2, data1, data2,
        ref_positions, dynamic_ref_positions, args.plots_dir
    )

    print("Plotting Figure 2: RMSD PCA panels...")
    plot_rmsd_panels(
        pca_coords1, pca_coords2, rmsd_dict1, rmsd_dict2,
        ref_positions, dynamic_ref_positions, args.plots_dir
    )

    print("Plotting Figure 2b: dRMSD PCA panels...")
    plot_drmsd_panels(
        pca_coords1, pca_coords2, drmsd_dict1, drmsd_dict2,
        ref_positions, dynamic_ref_positions, args.plots_dir
    )

    print("Plotting Figure 3: pLDDT PCA panel (AF2)...")
    if plddt1 is not None:
        plot_plddt_panel(pca_coords1, plddt1, ref_positions, dynamic_ref_positions, args.plots_dir)

    print("Plotting Figure 4: Feature distribution histograms...")
    plot_feature_histograms(data1, data2, args.plots_dir)

    print("Plotting Figure 5: RMSD scatter combinations...")
    plot_rmsd_scatter_combinations(data1, data2, args.plots_dir)

    print("Plotting Figure 6: RadGyr region comparisons (RMSD hue)...")
    plot_radgyr_rmsd_scatter(data1, data2, args.plots_dir)
    
    print("Plotting Figure 7: RMSD vs NAC RadGyr (C-tail hue)...")
    plot_rmsd_vs_nac_rg(data1, data2, args.plots_dir)

    print("Plotting Figure 8: RadGyr scatter combinations...")
    plot_radgyr_scatter_combinations(data1, data2, args.plots_dir)

    print("Plotting Figure 9: Secondary structure scatter...")
    plot_secondary_structure_scatter(data1, data2, args.plots_dir)

    print("Plotting Figure 10: dRMSD scatter combinations...")
    plot_drmsd_scatter_combinations(data1, data2, args.plots_dir)

    print("Plotting Figure 11: RMSD vs dRMSD scatter...")
    plot_rmsd_vs_drmsd_scatter(data1, data2, args.plots_dir)

    print(f"All plots saved to {args.plots_dir}")

if __name__ == "__main__":
    main()
