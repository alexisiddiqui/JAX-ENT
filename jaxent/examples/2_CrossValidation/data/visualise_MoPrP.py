#!/usr/bin/env python3
"""Visualise the free energy landscape of MoPrP.

Computes all collective variables and lets you pick any pair for the axes:

  Available CVs (--x-axis / --y-axis):
    drmsd          backbone φ/ψ dRMSD vs NMR circular mean (rad)
    rg             radius of gyration, all atoms (Å)
    i1_i3          principal moment ratio I₁/I₃ (prolate indicator)
    i2_i3          principal moment ratio I₂/I₃ (sphericity indicator)
    helix_fraction fraction of residues in α-helix per frame (DSSP)
    b1_dih_dev     mean |Δφ/ψ| for β1 strand vs NMR reference (°)
    b2_dih_dev     mean |Δφ/ψ| for β2 strand vs NMR reference (°)
    b2_disp        mean pairwise CA distance change, β2 vs α2–α3 core (Å)
    a1_disp        mean pairwise CA distance change, α1 vs α3 (Å)
    b2_dist        mean pairwise CA distance, β2 vs α2–α3 core (Å)
    a1_dist        mean pairwise CA distance, α1 vs α3 (Å)

Usage:
    python visualise_MoPrP.py \\
        --top-pdb MoPrP_max_plddt_4334.pdb \\
        --traj-xtc _MoPrP/MoPrP_plddt_ordered.xtc \\
        --nmr-pdb _MoPrP/2L39_crop.pdb \\
        --cluster-xtc _cluster_MoPrP/clusters/all_clusters.xtc \\
        --cluster-filtered-xtc _cluster_MoPrP_filtered/clusters/all_clusters.xtc \\
        --output-dir plots \\
        --x-axis i1_i3 --y-axis i2_i3 \\
        --absolute-paths

        python jaxent/examples/2_CrossValidation/data/visualise_MoPrP.py \
  --x-axis b1_dih_dev --y-axis drmsd --y-log --x-log \
  --kmeans-map jaxent/examples/2_CrossValidation/data/_cluster_MoPrP/data/cluster_labels.npy \
  --macro-map jaxent/examples/2_CrossValidation/analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/AF2_MSAss_macro_labels.npy \
  --absolute-paths \
  --output-dir jaxent/examples/2_CrossValidation/data/plots

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals
import MDAnalysis.analysis.dssp
import MDAnalysis.analysis.rms
import numpy as np
from scipy.stats import circmean, gaussian_kde

# ============================================================================
# Constants
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent

DEFAULT_TOP_PDB = DATA_DIR / "MoPrP_max_plddt_4334.pdb"
DEFAULT_TRAJ_XTC = DATA_DIR / "_MoPrP/MoPrP_plddt_ordered.xtc"
DEFAULT_NMR_PDB = DATA_DIR / "_MoPrP/2L39_crop.pdb"
DEFAULT_CLUSTER_XTC = DATA_DIR / "_cluster_MoPrP/clusters/all_clusters.xtc"
DEFAULT_CLUSTER_FILTERED_XTC = DATA_DIR / "_cluster_MoPrP_filtered/clusters/all_clusters.xtc"
DEFAULT_OUTPUT_DIR = DATA_DIR / "plots"

CV_LABELS: dict[str, str] = {
    "drmsd": r"dRMSD (rad)",
    "rg": r"$R_g$ (Å)",
    "i1_i3": r"$I_1/I_3$",
    "i2_i3": r"$I_2/I_3$",
    "helix_fraction": "Helix fraction (DSSP)",
    "puf1_rmsd": r"PUF1 region RMSD (Å)",
    "puf2_rmsd": r"PUF2 region RMSD (Å)",
    "b1_dih_dev": r"β1 dihedral deviation (°)",
    "b2_dih_dev": r"β2 dihedral deviation (°)",
    "b2_disp": r"β2–core distance change (Å)",
    "a1_disp": r"α1–α3 distance change (Å)",
    "b2_dist": r"β2–core distance (Å)",
    "a1_dist": r"α1–α3 distance (Å)",
}

# PUF region definitions (PDB residue numbering, offset +122 from original)
# Stable core used for superposition before computing PUF region RMSDs
_SEL_CORE = "name CA and resid 83:91"          # disulfide region, mop=1.1
_SEL_PUF1 = "name CA and (resid 5:21 or resid 95:101)"  # β1+loop β1-α1 + C-term α3
_SEL_PUF2 = "name CA and resid 27:45"          # α1 + β2 region (additional in PUF2)

# Clustering-derived selections (PDB residue numbering)
_SEL_A2A3_CORE = "name CA and (resid 49:71 or resid 77:101)"  # full α2-α3 core (superpos target)
_SEL_B2_STRAND = "name CA and resid 38:41"  # β2 strand only
_SEL_A1_HELIX = "name CA and resid 22:31"  # α1 helix
_SEL_A3_HELIX = "name CA and resid 77:101"  # α3 (a1_disp superpos target)
_B1_RESID_RANGE = (5, 10)  # β1 strand for dihedral slice
_B2_RESID_RANGE = (38, 41)  # β2 strand for dihedral slice

# ============================================================================
# CV Computation
# ============================================================================


def compute_nmr_reference(pdb_path: Path) -> np.ndarray:
    """Circular mean φ/ψ angles from multi-model NMR ensemble (radians)."""
    print(f"Loading NMR reference from {pdb_path}")
    u = mda.Universe(str(pdb_path))
    rama = MDAnalysis.analysis.dihedrals.Ramachandran(u.select_atoms("backbone"))
    rama.run()
    angles_rad = np.deg2rad(rama.results.angles)  # degrees → radians
    ref = circmean(angles_rad, low=-np.pi, high=np.pi, axis=0)
    print(f"  Computed circular mean φ/ψ for {ref.shape[0]} residues")
    return ref


def compute_all_cvs(
    u: mda.Universe, ref_dihedrals: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Compute all CVs for a loaded Universe.

    Parameters
    ----------
    u : mda.Universe
    ref_dihedrals : np.ndarray | None
        Required for dRMSD. If None, dRMSD is skipped (filled with NaN).

    Returns
    -------
    dict with keys ``"drmsd"``, ``"rg"``, ``"i1_i3"``, ``"i2_i3"``, ``"helix_fraction"``.
    """
    n_frames = len(u.trajectory)
    cvs: dict[str, np.ndarray] = {}

    # --- Rg ---
    rg = np.zeros(n_frames)
    sel_all = u.select_atoms("all")
    for i, _ in enumerate(u.trajectory):
        rg[i] = sel_all.radius_of_gyration()
    cvs["rg"] = rg

    # --- Inertia ratios (single pass over trajectory) ---
    i1_i3 = np.zeros(n_frames)
    i2_i3 = np.zeros(n_frames)
    for i, _ in enumerate(u.trajectory):
        evals = np.linalg.eigvalsh(sel_all.moment_of_inertia())  # ascending
        I1, I2, I3 = evals
        i1_i3[i] = I1 / I3
        i2_i3[i] = I2 / I3
    cvs["i1_i3"] = i1_i3
    cvs["i2_i3"] = i2_i3

    # --- Helix fraction (DSSP) ---
    dssp_ana = MDAnalysis.analysis.dssp.DSSP(u.select_atoms("protein"))
    dssp_ana.run()
    # dssp.results.dssp: (n_frames, n_residues) character array, 'H'=helix
    cvs["helix_fraction"] = (dssp_ana.results.dssp == "H").mean(axis=1).astype(float)

    # --- dRMSD ---
    if ref_dihedrals is not None:
        rama = MDAnalysis.analysis.dihedrals.Ramachandran(u.select_atoms("backbone"))
        rama.run()
        angles_rad = np.deg2rad(rama.results.angles)
        drmsd = np.zeros(n_frames)
        for i in range(n_frames):
            diff = angles_rad[i] - ref_dihedrals
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            drmsd[i] = np.sqrt(np.mean(diff**2))
        cvs["drmsd"] = drmsd

        backbone_resids = u.select_atoms("backbone").residues.resids
        rama_resids = backbone_resids[1:-1]  # Ramachandran excludes terminals
        b1_mask = (rama_resids >= _B1_RESID_RANGE[0]) & (rama_resids <= _B1_RESID_RANGE[1])
        b2_mask = (rama_resids >= _B2_RESID_RANGE[0]) & (rama_resids <= _B2_RESID_RANGE[1])

        b1_dev = np.zeros(n_frames)
        b2_dev = np.zeros(n_frames)

        for i in range(n_frames):
            # β1
            diff1 = angles_rad[i, b1_mask, :] - ref_dihedrals[b1_mask, :]
            diff1 = (diff1 + np.pi) % (2 * np.pi) - np.pi
            b1_dev[i] = np.mean(np.abs(diff1))

            # β2
            diff2 = angles_rad[i, b2_mask, :] - ref_dihedrals[b2_mask, :]
            diff2 = (diff2 + np.pi) % (2 * np.pi) - np.pi
            b2_dev[i] = np.mean(np.abs(diff2))

        cvs["b1_dih_dev"] = np.rad2deg(b1_dev)
        cvs["b2_dih_dev"] = np.rad2deg(b2_dev)
    else:
        cvs["drmsd"] = np.full(n_frames, np.nan)
        cvs["b1_dih_dev"] = np.full(n_frames, np.nan)
        cvs["b2_dih_dev"] = np.full(n_frames, np.nan)

    return cvs


def compute_puf_rmsds(
    u: mda.Universe, ref: mda.Universe
) -> tuple[np.ndarray, np.ndarray]:
    """CA RMSD of PUF1 and PUF2 regions after superposing on the stable core.

    Superposition target: disulfide region (resid 83-91, mop=1.1 kcal/mol/M).
    PUF1 regions: β1+loop β1-α1 (resid 5-21) + C-terminal α3 (resid 95-101).
    PUF2 additional regions: α1 helix + β2 (resid 27-45).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (puf1_rmsd, puf2_rmsd) in Ångströms, each shape (n_frames,).
    """
    R = MDAnalysis.analysis.rms.RMSD(
        u,
        ref,
        select=_SEL_CORE,
        groupselections=[_SEL_PUF1, _SEL_PUF2],
        ref_frame=0,
    )
    R.run()
    # results.rmsd columns: [frame, time, rmsd_core, rmsd_puf1, rmsd_puf2]
    return R.results.rmsd[:, 3], R.results.rmsd[:, 4]


def compute_subdomain_displacements(
    u: mda.Universe, ref: mda.Universe
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean pairwise CA distance between subdomains, both raw and relative to AF2 ref.

    Returns: (b2_disp, a1_disp, b2_dist, a1_dist)
    b2: β2 strand vs α2-α3 core  (PUF2 discriminator; threshold ~0.25 Å)
    a1: α1 helix vs α3 helix     (PUF3 discriminator; threshold ~0.75 Å)
    'disp' = difference from reference (positive = moved farther).
    'dist' = raw absolute distance.
    """

    def _mean_pairwise(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
        """Mean of all inter-atom distances between two CA sets."""
        diffs = pos_a[:, None, :] - pos_b[None, :, :]  # (nA, nB, 3)
        return float(np.mean(np.linalg.norm(diffs, axis=-1)))

    b2_atoms = u.select_atoms(_SEL_B2_STRAND)
    core_atoms = u.select_atoms(_SEL_A2A3_CORE)
    a1_atoms = u.select_atoms(_SEL_A1_HELIX)
    a3_atoms = u.select_atoms(_SEL_A3_HELIX)

    ref_b2_disp = _mean_pairwise(
        ref.select_atoms(_SEL_B2_STRAND).positions,
        ref.select_atoms(_SEL_A2A3_CORE).positions,
    )
    ref_a1_disp = _mean_pairwise(
        ref.select_atoms(_SEL_A1_HELIX).positions,
        ref.select_atoms(_SEL_A3_HELIX).positions,
    )

    n_frames = len(u.trajectory)
    b2_disp = np.zeros(n_frames)
    a1_disp = np.zeros(n_frames)
    b2_dist = np.zeros(n_frames)
    a1_dist = np.zeros(n_frames)

    for i, _ in enumerate(u.trajectory):
        d_b2 = _mean_pairwise(b2_atoms.positions, core_atoms.positions)
        d_a1 = _mean_pairwise(a1_atoms.positions, a3_atoms.positions)

        b2_dist[i] = d_b2
        a1_dist[i] = d_a1
        b2_disp[i] = d_b2 - ref_b2_disp
        a1_disp[i] = d_a1 - ref_a1_disp

    return b2_disp, a1_disp, b2_dist, a1_dist


def compute_cvs_for_xtc(
    xtc_path: Path, top_path: Path, ref_dihedrals: np.ndarray | None
) -> dict[str, np.ndarray]:
    """Load XTC and compute all CVs."""
    print(f"  Loading {xtc_path.name}...", end=" ", flush=True)
    u = mda.Universe(str(top_path), str(xtc_path))
    print(f"({len(u.trajectory)} frames)", end=" ", flush=True)
    cvs = compute_all_cvs(u, ref_dihedrals)

    # PUF region RMSDs — requires reference structure from topology
    ref = mda.Universe(str(top_path))
    cvs["puf1_rmsd"], cvs["puf2_rmsd"] = compute_puf_rmsds(u, ref)

    # Subdomain displacements and raw distances
    res = compute_subdomain_displacements(u, ref)
    cvs["b2_disp"], cvs["a1_disp"], cvs["b2_dist"], cvs["a1_dist"] = res

    print("done")
    return cvs


# ============================================================================
# Plotting
# ============================================================================

def compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_log: bool = False,
    y_log: bool = False,
    grid_size: int = 100,
    x_grid: np.ndarray | None = None,
    y_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute KDE density on a grid."""
    kde = gaussian_kde(np.vstack([x, y]))

    if x_grid is None or y_grid is None:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        pad_x = 0.05 * (x_max - x_min)
        pad_y = 0.05 * (y_max - y_min)

        if x_log:
            x_grid_vec = np.geomspace(max(x_min, 1e-6), x_max, grid_size)
        else:
            x_grid_vec = np.linspace(x_min - pad_x, x_max + pad_x, grid_size)
        
        if y_log:
            y_grid_vec = np.geomspace(max(y_min, 1e-6), y_max, grid_size)
        else:
            y_grid_vec = np.linspace(y_min - pad_y, y_max + pad_y, grid_size)
        
        x_grid, y_grid = np.meshgrid(x_grid_vec, y_grid_vec)

    density = kde(np.vstack([x_grid.ravel(), y_grid.ravel()])).reshape(x_grid.shape)
    return density, x_grid, y_grid


def plot_hexbin_landscape(
    x: np.ndarray,
    y: np.ndarray,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    gridsize: int = 50,
    x_log: bool = False,
    y_log: bool = False,
) -> None:
    """Hexbin free energy landscape with cluster overlays."""
    fig, ax = plt.subplots(figsize=(7, 6))

    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, cmap="YlGnBu_r", edgecolors="none",
                   xscale="log" if x_log else "linear",
                   yscale="log" if y_log else "linear")

    counts = hb.get_array().copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -np.log(counts)
    F -= np.nanmin(F)

    norm = matplotlib.colors.BoundaryNorm(list(range(9)), ncolors=256, extend="max")
    hb.set_array(F)
    hb.set_norm(norm)

    cbar = fig.colorbar(hb, ax=ax, pad=0.05)
    cbar.set_label(r"$\Delta F\ /\ k_BT$", fontsize=12)
    cbar.set_ticks(range(9))

    ax.scatter(x_a, y_a, s=10, c="black", edgecolors="white", linewidths=0.4, zorder=5)
    ax.scatter(x_b, y_b, s=10, c="white", edgecolors="black", linewidths=0.4, zorder=6)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Free Energy Landscape (hexbin)", fontsize=13)
    ax.grid(True, color="lightgray", lw=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path.name}")


def plot_contour_landscape(
    x: np.ndarray,
    y: np.ndarray,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    out_path: Path,
    xlabel: str,
    ylabel: str,
    x_log: bool = False,
    y_log: bool = False,
) -> None:
    """KDE contour free energy landscape with cluster overlays."""
    fig, ax = plt.subplots(figsize=(7, 6))

    density, XX, YY = compute_density_grid(x, y, x_log, y_log)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -np.log(density)
    F = F - np.nanmin(F)
    F = np.clip(F, 0, 8)

    levels = np.arange(0, 9)
    ax.contourf(XX, YY, F, levels=levels, cmap="YlGnBu_r", extend="max")
    ax.contour(XX, YY, F, levels=levels, colors="white", linewidths=0.5, alpha=0.5)

    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.BoundaryNorm(np.arange(10), 256), cmap="YlGnBu_r"
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05)
    cbar.set_label(r"$\Delta F\ /\ k_BT$", fontsize=12)
    cbar.set_ticks(np.arange(0, 9))

    ax.scatter(x_a, y_a, s=10, c="black", edgecolors="white", linewidths=0.4, zorder=5)
    ax.scatter(x_b, y_b, s=10, c="white", edgecolors="black", linewidths=0.4, zorder=6)

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title("Free Energy Landscape (KDE contour)", fontsize=13)
    ax.grid(True, color="lightgray", lw=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path.name}")


# ============================================================================
# Macro-cluster Mapping
# ============================================================================

_MACRO_NAMES = {
    0: "Folded",
    1: "PUF1",
    2: "PUF2",
    3: "PUF3",
    4: "Unfolded",
    5: "Intermediate",
}


def _cluster_palette(labels: np.ndarray) -> dict[int, tuple]:
    """Generate a categorical palette for clusters."""
    import matplotlib

    unique_labels = sorted([int(v) for v in np.unique(labels) if v >= 0])
    n = max(len(unique_labels), 1)
    try:
        cmap = matplotlib.colormaps["tab10"].resampled(n)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10", n)
    mapping = {label: cmap(i / max(n - 1, 1)) for i, label in enumerate(unique_labels)}
    if np.any(labels < 0):
        mapping[-1] = (0.55, 0.55, 0.55, 1.0)  # Grey for unassigned
    return mapping


def _build_cluster_face_colors(
    x: np.ndarray,
    y: np.ndarray,
    cluster_labels: np.ndarray,
    gridsize: int = 30,
    x_log: bool = False,
    y_log: bool = False,
) -> tuple[list[np.ndarray], dict[int, tuple]]:
    """Build cluster colors whitened by quantized free energy.
    
    Ported from inertia_moments_clustering.py for style consistency.
    """
    fig_tmp, ax_tmp = plt.subplots()
    hb_fe = ax_tmp.hexbin(
        x, y, gridsize=gridsize, mincnt=1,
        xscale="log" if x_log else "linear",
        yscale="log" if y_log else "linear"
    )
    counts = hb_fe.get_array()

    def _mode(values):
        vals = np.asarray(values, dtype=int)
        if vals.size == 0:
            return -1
        unique, modal_counts = np.unique(vals, return_counts=True)
        return int(unique[np.argmax(modal_counts)])

    hb_cl = ax_tmp.hexbin(
        x, y, C=cluster_labels, gridsize=gridsize, reduce_C_function=_mode, mincnt=1,
        xscale="log" if x_log else "linear",
        yscale="log" if y_log else "linear"
    )
    bin_labels = hb_cl.get_array().astype(int)
    plt.close(fig_tmp)

    with np.errstate(divide="ignore", invalid="ignore"):
        energy = -np.log(counts)
    energy -= np.nanmin(energy)
    energy_quant = np.clip(np.floor(energy), 0, 8).astype(int)

    palette = _cluster_palette(cluster_labels)
    face_colors = []
    for i, cid in enumerate(bin_labels):
        base_rgb = np.array(palette.get(int(cid), (0.5, 0.5, 0.5, 1.0))[:3])
        mix = energy_quant[i] / 9.0
        face_colors.append(base_rgb * (1.0 - mix) + np.array([1.0, 1.0, 1.0]) * mix)
    return face_colors, palette


def plot_macro_clusters_hexbin(
    x: np.ndarray,
    y: np.ndarray,
    macro_labels: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
    gridsize: int = 30,
    x_log: bool = False,
    y_log: bool = False,
):
    """Cluster hexbin plot with whitening by free energy."""
    face_colors, palette = _build_cluster_face_colors(x, y, macro_labels, gridsize=gridsize, x_log=x_log, y_log=y_log)

    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(
        x, y, gridsize=gridsize, mincnt=1, edgecolors="none",
        xscale="log" if x_log else "linear",
        yscale="log" if y_log else "linear"
    )
    hb.set_array(None)
    hb.set_facecolors(face_colors)

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = []
    for label in sorted(np.unique(macro_labels)):
        cid = int(label)
        name = _MACRO_NAMES.get(cid, f"Cluster {cid}") if cid >= 0 else "Unassigned"
        legend_patches.append(mpatches.Patch(color=palette[cid], label=name))
    ax.legend(handles=legend_patches, loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {os.path.basename(output_path)}")


def plot_macro_cluster_grid(
    x_full: np.ndarray,
    y_full: np.ndarray,
    cv_data: dict[int, tuple[np.ndarray, np.ndarray]],
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
    palette: dict[int, tuple],
    x_log: bool = False,
    y_log: bool = False,
):
    """Plot contours for all macro-clusters as columns in a single figure."""
    valid_cids = sorted([cid for cid, (x_clus, _) in cv_data.items() if len(x_clus) >= 10 and cid >= 0])
    n_cols = len(valid_cids)
    if n_cols == 0:
        print("No valid clusters to plot.")
        return

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5), sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]
    
    # Precompute background
    density_full, x_grid, y_grid = compute_density_grid(x_full, y_full, x_log, y_log)
    with np.errstate(divide="ignore", invalid="ignore"):
        F_full = -np.log(density_full)
    F_full -= np.nanmin(F_full)
    levels = np.arange(0, 9)

    for ax, cid in zip(axes, valid_cids):
        x_clus, y_clus = cv_data[cid]
        color = palette.get(cid, "black")
        name = _MACRO_NAMES.get(cid, f"Cluster {cid}")
        
        # Background
        ax.contour(x_grid, y_grid, F_full, levels=levels, colors="grey", alpha=0.2, linewidths=0.5)

        # Foreground
        try:
            density_clus, _, _ = compute_density_grid(x_clus, y_clus, x_log, y_log, x_grid=x_grid, y_grid=y_grid)
            with np.errstate(divide="ignore", invalid="ignore"):
                F_clus = -np.log(density_clus)
            F_clus -= np.nanmin(F_clus)
            
            cf = ax.contourf(x_grid, y_grid, F_clus, levels=levels, cmap="YlGnBu_r", extend="max")
            # Highlight with a primary contour outline for emphasis using the cluster color
            ax.contour(x_grid, y_grid, F_clus, levels=[2.0], colors=[color], linewidths=2.0)
        except Exception as e:
            print(f"  Warning: Could not compute cluster contour for {name}: {e}")

        if x_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
            
        ax.set_xlabel(x_label, fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(name, fontsize=13, color="black", weight="bold")
        print(f"  Plotting cluster {cid} ({name}): {len(x_clus)} frames")
        
        # Draw a thick border using the cluster color around each subplot
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)

    # Add a single colorbar
    fig.subplots_adjust(bottom=0.15, top=0.85, wspace=0.1, right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.BoundaryNorm(np.arange(10), 256), cmap="YlGnBu_r"
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"$\Delta F\ /\ k_BT$", fontsize=12)
    cbar.set_ticks(np.arange(0, 9))

    fig.suptitle(title, fontsize=16, y=1.0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {os.path.basename(output_path)}")


def plot_combined_cluster_landscape(
    x_full: np.ndarray,
    y_full: np.ndarray,
    macro_labels: np.ndarray,
    x_label: str,
    y_label: str,
    title: str,
    output_path: str,
    palette: dict[int, tuple],
    gridsize: int = 30,
    x_log: bool = False,
    y_log: bool = False,
):
    """Global hexbin landscape with hued cluster contours overlaid."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # 1. Global Hexbin Background
    hb = ax.hexbin(
        x_full, y_full, gridsize=gridsize, mincnt=1, cmap="Greys", edgecolors="none", alpha=0.3,
        xscale="log" if x_log else "linear",
        yscale="log" if y_log else "linear"
    )
    # Background density grid for consistent contour computation
    density_full, x_grid, y_grid = compute_density_grid(x_full, y_full, x_log, y_log)

    # 2. Cluster Contours
    unique_macros = sorted([cid for cid in np.unique(macro_labels) if cid >= 0])
    contour_levels = [1.0, 2.0, 3.0]  # kBT levels
    
    for cid in unique_macros:
        mask = (macro_labels == cid)
        if np.sum(mask) < 10:
            continue
            
        x_clus, y_clus = x_full[mask], y_full[mask]
        color = palette.get(cid, "black")
        name = _MACRO_NAMES.get(cid, f"Cluster {cid}")

        try:
            density_clus, _, _ = compute_density_grid(x_clus, y_clus, x_log, y_log, x_grid=x_grid, y_grid=y_grid)
            with np.errstate(divide="ignore", invalid="ignore"):
                F_clus = -np.log(density_clus)
            F_clus -= np.nanmin(F_clus)
            
            # Draw multiple contours with decreasing thickness/alpha
            # 1.0 (bold), 2.0 (medium), 3.0 (thin)
            lws = [2.5, 1.5, 0.8]
            alphas = [1.0, 0.6, 0.3]
            
            for lvl, lw, alpha in zip(contour_levels, lws, alphas):
                ax.contour(x_grid, y_grid, F_clus, levels=[lvl], colors=[color], linewidths=lw, alpha=alpha, zorder=10)
            
            # Add a faint filled area for the core population (< 2.0 kBT)
            ax.contourf(x_grid, y_grid, F_clus, levels=[0.0, 2.0], colors=[color], alpha=0.08, zorder=9)
        except Exception as e:
            print(f"  Warning: Could not compute contour for {name}: {e}")

    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
            
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)

    # Legend
    import matplotlib.patches as mpatches
    legend_patches = []
    for cid in unique_macros:
        color = palette.get(cid, "black")
        name = _MACRO_NAMES.get(cid, f"Cluster {cid}")
        legend_patches.append(mpatches.Patch(color=color, label=name))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {os.path.basename(output_path)}")

CV_CHOICES = list(CV_LABELS.keys())


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualise MoPrP free energy landscape with selectable CV axes."
    )
    parser.add_argument("--top-pdb", type=Path, default=DEFAULT_TOP_PDB)
    parser.add_argument("--traj-xtc", type=Path, default=DEFAULT_TRAJ_XTC)
    parser.add_argument(
        "--nmr-pdb",
        type=Path,
        default=DEFAULT_NMR_PDB,
        help="NMR reference PDB — required when --x-axis or --y-axis is 'drmsd'",
    )
    parser.add_argument("--cluster-xtc", type=Path, default=DEFAULT_CLUSTER_XTC)
    parser.add_argument(
        "--cluster-filtered-xtc", type=Path, default=DEFAULT_CLUSTER_FILTERED_XTC
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gridsize", type=int, default=15)
    parser.add_argument(
        "--x-axis",
        choices=CV_CHOICES,
        default="drmsd",
        help=f"CV for X axis. Choices: {CV_CHOICES}",
    )
    parser.add_argument(
        "--y-axis",
        choices=CV_CHOICES,
        default="rg",
        help=f"CV for Y axis. Choices: {CV_CHOICES}",
    )
    parser.add_argument("--x-log", action="store_true", help="Log scale on X axis")
    parser.add_argument("--y-log", action="store_true", help="Log scale on Y axis")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--kmeans-map", type=Path, help="Path to 12700 -> 500 mapping (.npy)")
    parser.add_argument("--macro-map", type=Path, help="Path to 500 -> 5 mapping (.npy)")

    args = parser.parse_args()

    if args.absolute_paths:
        top_pdb = args.top_pdb.resolve()
        traj_xtc = args.traj_xtc.resolve()
        nmr_pdb = args.nmr_pdb.resolve()
        cluster_xtc = args.cluster_xtc.resolve()
        cluster_filtered_xtc = args.cluster_filtered_xtc.resolve()
        output_dir = args.output_dir.resolve()
    else:
        top_pdb = args.top_pdb
        traj_xtc = args.traj_xtc
        nmr_pdb = args.nmr_pdb
        cluster_xtc = args.cluster_xtc
        cluster_filtered_xtc = args.cluster_filtered_xtc
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    x_key, y_key = args.x_axis, args.y_axis
    xlabel = CV_LABELS[x_key]
    ylabel = CV_LABELS[y_key]
    scale_suffix = ("_xlog" if args.x_log else "") + ("_ylog" if args.y_log else "")
    suffix = f"{x_key}_vs_{y_key}{scale_suffix}"

    need_nmr = any(k in (x_key, y_key) for k in ("drmsd", "b1_dih_dev", "b2_dih_dev"))
    ref_dihedrals = compute_nmr_reference(nmr_pdb) if need_nmr else None

    print("Computing CVs for full ensemble...")
    cvs_full = compute_cvs_for_xtc(traj_xtc, top_pdb, ref_dihedrals)

    print("Computing CVs for AF2 MSAss (Cluster XTC)...")
    cvs_clus = compute_cvs_for_xtc(cluster_xtc, top_pdb, ref_dihedrals)
    
    print("Computing CVs for AF2 Filtered...")
    cvs_filt = compute_cvs_for_xtc(cluster_filtered_xtc, top_pdb, ref_dihedrals)

    print("Generating plots...")
    hex_path = output_dir / f"fe_landscape_hexbin_{suffix}.png"
    plot_hexbin_landscape(
        cvs_full[x_key], cvs_full[y_key], 
        cvs_clus[x_key], cvs_clus[y_key],
        cvs_filt[x_key], cvs_filt[y_key],
        hex_path,
        CV_LABELS[x_key], CV_LABELS[y_key],
        x_log=args.x_log, y_log=args.y_log
    )

    cont_path = output_dir / f"fe_landscape_contour_{suffix}.png"
    plot_contour_landscape(
        cvs_full[x_key], cvs_full[y_key], 
        cvs_clus[x_key], cvs_clus[y_key],
        cvs_filt[x_key], cvs_filt[y_key],
        cont_path,
        CV_LABELS[x_key], CV_LABELS[y_key],
        x_log=args.x_log, y_log=args.y_log
    )

    # --- Cluster-specific landscapes ---
    if args.kmeans_map and args.macro_map:
        cluster_dir = output_dir / "cluster_landscapes"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating cluster-specific landscapes in {cluster_dir.name}...")
        
        # Load maps
        kmeans_labels = np.load(args.kmeans_map)      # (12700,)
        macro_map_arr = np.load(args.macro_map)       # (500,)
        
        # Map frames to macro-clusters
        # Ensure kmeans_labels are valid for mapping
        macro_labels = macro_map_arr[kmeans_labels]   # (12700,)

        # 1. Global hexbin with all clusters (energy fading)
        hex_clus_path = cluster_dir / f"fe_landscape_hexbin_clusters_{suffix}.png"
        plot_macro_clusters_hexbin(
            cvs_full[x_key], cvs_full[y_key], macro_labels,
            CV_LABELS[x_key], CV_LABELS[y_key],
            "Macro-cluster Landscapes with Energy Fading",
            str(hex_clus_path),
            x_log=args.x_log, y_log=args.y_log
        )

        # 2. Individual cluster contours mapped to columns
        cluster_grid_path = cluster_dir / f"fe_landscape_contours_grid_{suffix}.png"
        unique_macros = np.unique(macro_labels)
        cv_data_map = {
            cid: (cvs_full[x_key][macro_labels == cid], cvs_full[y_key][macro_labels == cid])
            for cid in unique_macros if cid >= 0
        }
        palette = _cluster_palette(macro_labels)
        
        plot_macro_cluster_grid(
            cvs_full[x_key], cvs_full[y_key], cv_data_map,
            CV_LABELS[x_key], CV_LABELS[y_key],
            "Free Energy Landscapes by Cluster",
            str(cluster_grid_path),
            palette,
            x_log=args.x_log, y_log=args.y_log
        )

        # 3. Combined Landscape (Hexbin + Hued Contours)
        combined_path = cluster_dir / f"fe_landscape_combined_clusters_{suffix}.png"
        plot_combined_cluster_landscape(
            cvs_full[x_key], cvs_full[y_key], macro_labels,
            CV_LABELS[x_key], CV_LABELS[y_key],
            "MoPrP Cluster Populations on Free Energy Landscape",
            str(combined_path),
            palette,
            gridsize=args.gridsize*2,
            x_log=args.x_log, y_log=args.y_log
        )

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
