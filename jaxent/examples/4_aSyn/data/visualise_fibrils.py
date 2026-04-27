#!/usr/bin/env python3
"""Perform sequence-aligned dRMSD analysis of aSyn MD trajectory vs fibril structures.

For multi-frame PDBs, uses the circular mean φ/ψ angles as the reference. The topology
defines the true sequence for alignment (important since fibril PDBs are fragments).

Outputs 4 publication-quality figures:
  1. Histograms of dRMSD values for each fibril (4×3 grid)
  2. Heatmap: 11×11 inter-fibril dRMSD + trajectory median row
  3. Per-residue coverage bar chart (which residues are in which fibrils)
  4. Ternary shape hexgrid: one hexbin panel per fibril hued by mean dRMSD

Usage:
  python visualise_fibrils.py \\
    --top-pdb path/to/topology.pdb \\
    --traj-xtc path/to/trajectory.xtc \\
    --fibril-dir path/to/chain_A/ \\
    --output-dir plots/fibrils \\
    --absolute-paths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals
import numpy as np
from scipy.stats import circmean

# ============================================================================
# Constants & Configuration
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent

DEFAULT_TOP_PDB = DATA_DIR / "_aSyn/tris_MD/md_mol_center_coil.pdb"
DEFAULT_TRAJ_XTC = DATA_DIR / "_aSyn/tris_MD/tris_all_combined.xtc"
DEFAULT_FIBRIL_DIR = DATA_DIR / "_aSyn/fibrils/chain_A"
DEFAULT_CLUSTER_DIR = DATA_DIR / "_cluster_inertia"
DEFAULT_OUTPUT_DIR = DATA_DIR / "plots/fibrils"

FIBRIL_NAMES: dict[str, str] = {
    "8ADW": "L1C", "9FYP": "3B", "8AEX": "L3A", "8Y2P": "Tris",
    "2N0A": "Fibril", "6CU7": "Rod", "6CU8": "Twister",
    "6RT0": "2A", "6RTB": "2B",
    # 7NCK and 8PIX have no display name — fall back to PDB ID only
}

FIBRIL_COLORS: dict[str, tuple] = {}  # Populated at runtime from tab20

# Copied from inertia_moments_clustering.py
HEXBIN_EXTENT = [-0.08, 1.12, 0.42, 1.10]

# ============================================================================
# Helpers
# ============================================================================


def fibril_label(pdb_id: str) -> str:
    """Display label for a fibril: name (PDB ID) or just PDB ID if unnamed."""
    name = FIBRIL_NAMES.get(pdb_id)
    return f"{name} ({pdb_id})" if name else pdb_id


def set_publication_style():
    """Set matplotlib for publication-quality figures."""
    matplotlib.rcParams.update({
        "figure.dpi": 100,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
    })


def _draw_shape_boundary(ax):
    """Draw triangular shape-space boundary, vertex labels, and guide line.

    Copied verbatim from inertia_moments_clustering.py.
    """
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [1, 1, 0.5, 1]
    ax.plot(triangle_x, triangle_y, color="black", linestyle="--", lw=2,
            label="Shape Boundary", zorder=5)
    label_kw = dict(fontsize=9, zorder=6,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.text(0, 1.04, r"Rod ($I_1 = 0$)", ha="center", **label_kw)
    ax.text(1, 1.04, r"Sphere ($I_1 = I_2 = I_3$)", ha="center", **label_kw)
    ax.text(0.5, 0.46, r"Disk ($I_1 = I_2$)", ha="center", **label_kw)
    ax.plot([0.5, 1], [0.5, 1], color="gray", lw=1, ls=":", alpha=0.7, zorder=4)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(0.42, 1.10)
    ax.set_xlabel(r"$I_1/I_3$", fontsize=10)
    ax.set_ylabel(r"$I_2/I_3$", fontsize=10)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, color="lightgray", lw=0.5, zorder=0)
    ax.tick_params(width=1.5)


def _ensure_output_dir(output_dir: Path) -> None:
    """Create output directory if it does not exist."""
    output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading & Computation
# ============================================================================


def discover_fibrils(fibril_dir: Path) -> list[tuple[str, Path]]:
    """Discover all fibril chain-A PDB files in the directory.

    Returns
    -------
    list[(pdb_id, Path)]
        Sorted by PDB ID.
    """
    fibrils = []
    for pdb_file in sorted(fibril_dir.glob("*_chain_A_frames.pdb")):
        pdb_id = pdb_file.stem.split("_chain_A_frames")[0]
        fibrils.append((pdb_id, pdb_file))
    return fibrils


def load_fibril_references(fibril_paths: list[tuple[str, Path]]) -> dict:
    """Compute circular-mean φ/ψ reference for each multi-frame fibril PDB.

    Parameters
    ----------
    fibril_paths : list[(pdb_id, Path)]

    Returns
    -------
    dict
        {pdb_id: {"ref_dihedrals": (n_res, 2), "resids": (n_res,)}}
    """
    fibril_refs = {}
    for pdb_id, path in fibril_paths:
        print(f"Loading fibril {fibril_label(pdb_id)}...", end=" ", flush=True)
        try:
            u = mda.Universe(str(path))
            backbone = u.select_atoms("backbone")
            rama = mda.analysis.dihedrals.Ramachandran(backbone)
            rama.run()

            # Circular mean over all frames (models in the multi-frame PDB)
            angles_rad = np.deg2rad(rama.results.angles)  # (n_frames, n_res, 2)
            ref_φψ = circmean(angles_rad, low=-np.pi, high=np.pi, axis=0)  # (n_res, 2)

            # Ramachandran only covers residues with both phi and psi; not necessarily N-1:C-1
            # Get the actual residues covered by Ramachandran (may be fewer due to fragments/termini)
            # We reconstruct this by looking at which residues Ramachandran actually computed for
            all_resids = backbone.residues.resids

            # Ramachandran computes for residues with both phi and psi available
            # For a continuous backbone, this is typically [1:-1], but fragments may differ
            # We use the length of the angles to infer: if we have n angles,
            # Ramachandran covered n residues (with indices such that phi and psi are defined)
            # Since we can't easily extract which residues were computed, use a safe approach:
            # assume [1:-1] for now, but if the angle count differs, use only as many as angles

            n_angles = angles_rad.shape[1]
            n_backbone = len(all_resids)

            # Typically Ramachandran excludes first and last, but for fragments it may vary
            if n_angles == n_backbone - 2:
                # Standard case: excludes N and C termini
                resids = all_resids[1:-1]
            elif n_angles == n_backbone - 1:
                # Fragment missing C-terminus or similar
                resids = all_resids[:-1]
            elif n_angles == n_backbone:
                # Unusual but possible for some fragment definitions
                resids = all_resids
            else:
                # Fallback: use first n_angles residues
                # (may not be perfect, but avoids dimension mismatch)
                resids = all_resids[:n_angles]

            fibril_refs[pdb_id] = {
                "ref_dihedrals": ref_φψ,
                "resids": resids,
            }
            print(f"✓ ({len(resids)} residues, {len(u.trajectory)} models)")

        except Exception as e:
            # Try loading just first model as fallback
            print(f"✗ Error with multi-frame, trying first model only...", end=" ", flush=True)
            try:
                u = mda.Universe(str(path))
                u.trajectory[0]  # Load just first frame
                backbone = u.select_atoms("backbone")
                rama = mda.analysis.dihedrals.Ramachandran(backbone)
                rama.run()

                angles_rad = np.deg2rad(rama.results.angles)  # (1, n_res, 2)
                ref_φψ = angles_rad[0]  # (n_res, 2)

                all_resids = backbone.residues.resids
                n_angles = angles_rad.shape[1]
                n_backbone = len(all_resids)

                if n_angles == n_backbone - 2:
                    resids = all_resids[1:-1]
                elif n_angles == n_backbone - 1:
                    resids = all_resids[:-1]
                elif n_angles == n_backbone:
                    resids = all_resids
                else:
                    resids = all_resids[:n_angles]

                fibril_refs[pdb_id] = {
                    "ref_dihedrals": ref_φψ,
                    "resids": resids,
                }
                print(f"✓ (fallback: first model, {len(resids)} residues)")

            except Exception as e2:
                print(f"✗ Skipped due to error: {type(e2).__name__}")

    return fibril_refs


def compute_drmsds(
    traj_u: mda.Universe,
    fibril_refs: dict,
) -> dict[str, np.ndarray]:
    """Compute per-frame dRMSD of trajectory vs each fibril.

    Parameters
    ----------
    traj_u : mda.Universe
    fibril_refs : dict
        {pdb_id: {"ref_dihedrals": (n_res, 2), "resids": (n_res,)}}

    Returns
    -------
    dict[str, np.ndarray]
        {pdb_id: (n_frames,)} dRMSD in degrees
    """
    print("Computing trajectory Ramachandran (once)...", end=" ", flush=True)
    traj_backbone = traj_u.select_atoms("backbone")
    traj_rama = mda.analysis.dihedrals.Ramachandran(traj_backbone)
    traj_rama.run()

    traj_angles_rad = np.deg2rad(traj_rama.results.angles)  # (n_frames, n_res, 2)
    traj_resids = traj_backbone.residues.resids[1:-1]  # Ramachandran excludes terminals
    n_frames = len(traj_u.trajectory)
    print(f"✓ ({n_frames} frames, {len(traj_resids)} residues)")

    drmsds = {}
    for pdb_id, ref in fibril_refs.items():
        print(f"  Computing dRMSD vs {fibril_label(pdb_id)}...", end=" ", flush=True)

        # Find shared residues between trajectory and fibril
        shared_in_traj = np.isin(traj_resids, ref["resids"])
        shared_in_fibril = np.isin(ref["resids"], traj_resids)

        if not np.any(shared_in_traj):
            print("✗ No shared residues!")
            drmsds[pdb_id] = np.full(n_frames, np.nan)
            continue

        # Extract shared angles
        traj_angles_shared = traj_angles_rad[:, shared_in_traj, :]  # (n_frames, n_shared, 2)
        ref_angles_shared = ref["ref_dihedrals"][shared_in_fibril, :]  # (n_shared, 2)

        # Compute per-frame dRMSD
        drmsd = np.zeros(n_frames)
        for i in range(n_frames):
            diff = traj_angles_shared[i] - ref_angles_shared
            # Circular distance in [-π, π]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            # RMS in radians → degrees
            drmsd[i] = np.rad2deg(np.sqrt(np.mean(diff**2)))

        drmsds[pdb_id] = drmsd
        print(f"✓ mean={np.mean(drmsd):.1f}°, std={np.std(drmsd):.1f}°")

    return drmsds


def compute_interfibril_matrix(fibril_refs: dict) -> np.ndarray:
    """Compute dRMSD between each pair of fibril references.

    Parameters
    ----------
    fibril_refs : dict

    Returns
    -------
    np.ndarray
        (11, 11) symmetric matrix, diagonal = 0, units = degrees
    """
    pdb_ids = sorted(fibril_refs.keys())
    n = len(pdb_ids)
    matrix = np.zeros((n, n))

    print("Computing inter-fibril dRMSD matrix...")
    for i, id_i in enumerate(pdb_ids):
        for j, id_j in enumerate(pdb_ids):
            if i == j:
                matrix[i, j] = 0.0
                continue
            if i > j:  # Only compute upper triangle, mirror later
                continue

            # Find shared residues
            shared_i = np.isin(fibril_refs[id_i]["resids"], fibril_refs[id_j]["resids"])
            shared_j = np.isin(fibril_refs[id_j]["resids"], fibril_refs[id_i]["resids"])

            if not np.any(shared_i):
                matrix[i, j] = np.nan
                matrix[j, i] = np.nan
                continue

            # Compute dRMSD between circular-mean references
            angles_i = fibril_refs[id_i]["ref_dihedrals"][shared_i]
            angles_j = fibril_refs[id_j]["ref_dihedrals"][shared_j]

            diff = angles_i - angles_j
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            drmsd = np.rad2deg(np.sqrt(np.mean(diff**2)))

            matrix[i, j] = drmsd
            matrix[j, i] = drmsd

    return matrix


def load_or_compute_drmsds(
    output_dir: Path,
    traj_u: mda.Universe,
    fibril_refs: dict,
    recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Load dRMSD from cache or compute from trajectory.

    Parameters
    ----------
    output_dir : Path
    traj_u : mda.Universe
    fibril_refs : dict
    recompute : bool
        If True, ignore cache and recompute

    Returns
    -------
    dict[str, np.ndarray]
    """
    cache_path = output_dir / "fibril_drmsds.npz"

    if cache_path.exists() and not recompute:
        print(f"Loading dRMSD cache from {cache_path.name}...", end=" ", flush=True)
        data = np.load(cache_path, allow_pickle=True)
        drmsds = {k: data[k] for k in data.files if k != "fibril_ids"}
        print(f"✓ ({len(drmsds)} fibrils)")
        return drmsds

    drmsds = compute_drmsds(traj_u, fibril_refs)

    # Save cache
    print(f"Saving dRMSD cache to {cache_path.name}...", end=" ", flush=True)
    np.savez(cache_path, fibril_ids=list(drmsds.keys()), **drmsds)
    print("✓")

    return drmsds


# ============================================================================
# Plotting
# ============================================================================


def plot_histograms(drmsds: dict, output_dir: Path) -> None:
    """Plot dRMSD histograms for each fibril (4×3 grid)."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    pdb_ids = sorted(drmsds.keys())
    for idx, pdb_id in enumerate(pdb_ids):
        ax = axes[idx]
        drmsd_vals = drmsds[pdb_id]
        ax.hist(drmsd_vals, bins=50, color=FIBRIL_COLORS[pdb_id], alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(np.median(drmsd_vals), color="red", linestyle="--", linewidth=1.5, label=f"Median: {np.median(drmsd_vals):.1f}°")
        ax.set_title(fibril_label(pdb_id), fontsize=11, weight="bold")
        ax.set_xlabel("dRMSD (°)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide 12th axis
    axes[11].set_visible(False)

    fig.tight_layout()
    output_path = output_dir / "histograms.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path.name}")
    plt.close(fig)


def plot_heatmap(matrix: np.ndarray, drmsds: dict, output_dir: Path) -> None:
    """Plot 11×11 inter-fibril heatmap + MD median row."""
    pdb_ids = sorted(drmsds.keys())

    # Compute median dRMSD per fibril from trajectory
    md_row = np.array([np.median(drmsds[id]) for id in pdb_ids])

    # Create figure with GridSpec for the extra row
    fig = plt.figure(figsize=(11, 9))
    gs = gridspec.GridSpec(2, 1, height_ratios=[11, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_md = fig.add_subplot(gs[1])

    # Main heatmap
    im_main = ax_main.imshow(matrix, cmap="viridis_r", aspect="auto")
    ax_main.set_xticks(range(len(pdb_ids)))
    ax_main.set_yticks(range(len(pdb_ids)))
    ax_main.set_xticklabels([fibril_label(id) for id in pdb_ids], rotation=45, ha="right", fontsize=9)
    ax_main.set_yticklabels([fibril_label(id) for id in pdb_ids], fontsize=9)
    ax_main.set_title("Inter-fibril dRMSD matrix (°)", fontsize=12, weight="bold", pad=10)

    # Cell annotations
    for i in range(len(pdb_ids)):
        for j in range(len(pdb_ids)):
            if not np.isnan(matrix[i, j]):
                text_color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax_main.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=text_color, fontsize=7)

    # MD median row
    md_row_reshaped = md_row[None, :]
    im_md = ax_md.imshow(md_row_reshaped, cmap="viridis_r", aspect="auto")
    ax_md.set_xticks(range(len(pdb_ids)))
    ax_md.set_yticks([0])
    ax_md.set_xticklabels([fibril_label(id) for id in pdb_ids], rotation=45, ha="right", fontsize=9)
    ax_md.set_yticklabels(["MD"], fontsize=9)

    # MD row annotations
    for j, val in enumerate(md_row):
        text_color = "white" if val > md_row.max() / 2 else "black"
        ax_md.text(j, 0, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=7)

    # Shared colorbar
    cbar = fig.colorbar(im_main, ax=[ax_main, ax_md], label="dRMSD (°)", shrink=0.8, pad=0.02)

    output_path = output_dir / "heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path.name}")
    plt.close(fig)


def plot_coverage(fibril_refs: dict, traj_u: mda.Universe, output_dir: Path) -> None:
    """Plot per-residue coverage bar chart + fibril span strips."""
    pdb_ids = sorted(fibril_refs.keys())
    traj_resids = traj_u.select_atoms("backbone").residues.resids
    res_min, res_max = int(traj_resids.min()), int(traj_resids.max())

    # Count fibrils per residue
    residues = np.arange(res_min, res_max + 1)
    coverage = np.zeros(len(residues))
    for pdb_id in pdb_ids:
        for res_id in fibril_refs[pdb_id]["resids"]:
            idx = res_id - res_min
            if 0 <= idx < len(residues):
                coverage[idx] += 1

    fig, (ax_bar, ax_strips) = plt.subplots(2, 1, figsize=(14, 5), height_ratios=[3, 1], sharex=True)

    # Bar chart
    ax_bar.bar(residues, coverage, width=1.0, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax_bar.axhline(1, color="gray", linestyle=":", alpha=0.5)
    ax_bar.axhline(5, color="gray", linestyle=":", alpha=0.5)
    ax_bar.axhline(11, color="gray", linestyle=":", alpha=0.5)
    ax_bar.set_ylabel("# fibrils covering residue")
    ax_bar.set_title("Residue coverage across all fibril structures", fontsize=12, weight="bold")
    ax_bar.grid(alpha=0.3, axis="y")

    # Fibril span strips (stacked)
    strip_height = 1.0 / len(pdb_ids)
    y_offset = 0
    for idx, pdb_id in enumerate(pdb_ids):
        fibril_res_min = fibril_refs[pdb_id]["resids"].min()
        fibril_res_max = fibril_refs[pdb_id]["resids"].max()
        ax_strips.barh(y_offset + strip_height / 2, fibril_res_max - fibril_res_min + 1,
                       left=fibril_res_min, height=strip_height, color=FIBRIL_COLORS[pdb_id],
                       alpha=0.8, edgecolor="black", linewidth=0.5)
        y_offset += strip_height

    ax_strips.set_ylim(0, 1)
    ax_strips.set_yticks([])
    ax_strips.text(-0.02, 0.5, "Fibrils", fontsize=9, ha="right", va="center", transform=ax_strips.transAxes, weight="bold")
    ax_strips.set_xlabel("Residue")

    fig.tight_layout()
    output_path = output_dir / "coverage.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path.name}")
    plt.close(fig)


def plot_ternary_panels(drmsds: dict, cluster_dir: Path, output_dir: Path) -> None:
    """Plot ternary shape hexgrid panels hued by dRMSD."""
    shape_axes_path = cluster_dir / "shape_axes.npy"
    if not shape_axes_path.exists():
        print(f"⚠ Shape axes not found at {shape_axes_path}, skipping ternary panels")
        return

    shape_axes = np.load(shape_axes_path)  # (n_frames, 2)
    x = shape_axes[:, 0]
    y = shape_axes[:, 1]

    pdb_ids = sorted(drmsds.keys())

    # Global colour scale across all panels (5th–95th percentile)
    all_vals = np.concatenate([drmsds[id] for id in pdb_ids])
    vmin_global = float(np.nanpercentile(all_vals, 5))
    vmax_global = float(np.nanpercentile(all_vals, 95))

    # 3×4 grid + one extra column for the shared colorbar
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.08],
                           hspace=0.40, wspace=0.30)

    panel_axes = []
    for row in range(3):
        for col in range(4):
            panel_axes.append(fig.add_subplot(gs[row, col]))
    cbar_ax = fig.add_subplot(gs[:, 4])

    last_hb = None
    for idx, pdb_id in enumerate(pdb_ids):
        ax = panel_axes[idx]
        drmsd_vals = drmsds[pdb_id]

        hb = ax.hexbin(x, y, C=drmsd_vals, reduce_C_function=np.mean,
                       gridsize=60, cmap="cividis", extent=HEXBIN_EXTENT,
                       mincnt=1, edgecolors="none",
                       vmin=vmin_global, vmax=vmax_global)

        _draw_shape_boundary(ax)
        ax.set_title(fibril_label(pdb_id), fontsize=10, weight="bold", pad=4)
        last_hb = hb

    # Hide the 12th panel
    panel_axes[11].set_visible(False)

    # Single shared colorbar
    cbar = fig.colorbar(last_hb, cax=cbar_ax)
    cbar.set_label("Mean dRMSD (°)", fontsize=11)

    output_path = output_dir / "ternary_drmsd.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path.name}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top-pdb", type=Path, default=DEFAULT_TOP_PDB,
                        help=f"Topology PDB (default: {DEFAULT_TOP_PDB.name})")
    parser.add_argument("--traj-xtc", type=Path, default=DEFAULT_TRAJ_XTC,
                        help=f"Trajectory XTC (default: {DEFAULT_TRAJ_XTC.name})")
    parser.add_argument("--fibril-dir", type=Path, default=DEFAULT_FIBRIL_DIR,
                        help=f"Fibril chain-A directory (default: .../{DEFAULT_FIBRIL_DIR.name})")
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR,
                        help=f"Clustering output dir (default: .../{DEFAULT_CLUSTER_DIR.name})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for PNGs (default: .../{DEFAULT_OUTPUT_DIR.name})")
    parser.add_argument("--recompute", action="store_true",
                        help="Ignore .npz cache and recompute from trajectory")
    parser.add_argument("--absolute-paths", action="store_true",
                        help="Treat input paths as absolute (not relative to script dir)")

    args = parser.parse_args()

    # Resolve paths
    if not args.absolute_paths:
        args.top_pdb = DATA_DIR / args.top_pdb if not args.top_pdb.is_absolute() else args.top_pdb
        args.traj_xtc = DATA_DIR / args.traj_xtc if not args.traj_xtc.is_absolute() else args.traj_xtc
        args.fibril_dir = DATA_DIR / args.fibril_dir if not args.fibril_dir.is_absolute() else args.fibril_dir
        args.cluster_dir = DATA_DIR / args.cluster_dir if not args.cluster_dir.is_absolute() else args.cluster_dir
        args.output_dir = DATA_DIR / args.output_dir if not args.output_dir.is_absolute() else args.output_dir

    _ensure_output_dir(args.output_dir)
    set_publication_style()

    # Load trajectory
    print(f"Loading trajectory from {args.traj_xtc.name}...", end=" ", flush=True)
    traj_u = mda.Universe(str(args.top_pdb), str(args.traj_xtc))
    print(f"✓ ({len(traj_u.trajectory)} frames)")

    # Discover and load fibrils
    print(f"Discovering fibrils in {args.fibril_dir}...")
    fibril_paths = discover_fibrils(args.fibril_dir)
    print(f"Found {len(fibril_paths)} fibrils")

    fibril_refs = load_fibril_references(fibril_paths)

    # Assign colors
    pdb_ids = sorted(fibril_refs.keys())
    cmap = plt.cm.tab20
    for idx, pdb_id in enumerate(pdb_ids):
        FIBRIL_COLORS[pdb_id] = cmap(idx / len(pdb_ids))

    # Compute or load dRMSD
    drmsds = load_or_compute_drmsds(args.output_dir, traj_u, fibril_refs, args.recompute)

    # Compute inter-fibril matrix
    matrix = compute_interfibril_matrix(fibril_refs)

    # Generate plots
    print("\nGenerating plots...")
    plot_histograms(drmsds, args.output_dir)
    plot_heatmap(matrix, drmsds, args.output_dir)
    plot_coverage(fibril_refs, traj_u, args.output_dir)
    plot_ternary_panels(drmsds, args.cluster_dir, args.output_dir)

    print(f"\n✓ All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
