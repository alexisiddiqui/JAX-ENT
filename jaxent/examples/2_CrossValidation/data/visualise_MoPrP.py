#!/usr/bin/env python3
"""Visualise the free energy landscape of MoPrP (dRMSD vs Rg).

Plots the free energy landscape using two collective variables:
  - dRMSD: backbone φ/ψ RMSD against the circular mean of the NMR ensemble (2L39_crop.pdb)
  - Rg: radius of gyration of the full protein

Cluster representatives from two clustering runs are overlaid as scatter points.

Outputs two figures: hexbin and KDE contour landscapes.

Usage:
    python visualise_MoPrP.py \\
        --top-pdb MoPrP_max_plddt_4334.pdb \\
        --traj-xtc _MoPrP/MoPrP_plddt_ordered.xtc \\
        --nmr-pdb _MoPrP/2L39_crop.pdb \\
        --cluster-xtc _cluster_MoPrP/clusters/all_clusters.xtc \\
        --cluster-filtered-xtc _cluster_MoPrP_filtered/clusters/all_clusters.xtc \\
        --output-dir plots \\
        --absolute-paths
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import MDAnalysis as mda
import MDAnalysis.analysis.dihedrals
import numpy as np
from scipy.stats import circmean, gaussian_kde

# ============================================================================
# Constants
# ============================================================================

DATA_DIR = Path(__file__).resolve().parent

DEFAULT_TOP_PDB = DATA_DIR / "MoPrP_max_plddt_4334.pdb"
DEFAULT_TRAJ_XTC = DATA_DIR / "_MoPrP/MoPrP_plddt_ordered.xtc"
DEFAULT_NMR_PDB = DATA_DIR / "_MoPrP/MoPrP_max_plddt_4334.pdb"
DEFAULT_CLUSTER_XTC = DATA_DIR / "_cluster_MoPrP/clusters/all_clusters.xtc"
DEFAULT_CLUSTER_FILTERED_XTC = DATA_DIR / "_cluster_MoPrP_filtered/clusters/all_clusters.xtc"
DEFAULT_OUTPUT_DIR = DATA_DIR / "plots"

# ============================================================================
# CV Computation
# ============================================================================


def compute_nmr_reference(pdb_path: Path) -> np.ndarray:
    """Compute circular mean φ/ψ angles from multi-model NMR ensemble.

    Parameters
    ----------
    pdb_path : Path
        Path to NMR PDB file (multi-model structure).

    Returns
    -------
    np.ndarray
        Reference φ/ψ angles in radians, shape (n_residues, 2).
        Each row: [φ, ψ] for one residue (converted from MDAnalysis degrees output).
    """
    print(f"Loading NMR reference from {pdb_path}")
    u = mda.Universe(str(pdb_path))

    # Compute φ/ψ for all models in the ensemble
    rama = MDAnalysis.analysis.dihedrals.Ramachandran(u.select_atoms("backbone"))
    rama.run()

    # rama.results.angles has shape (n_frames, n_residues, 2) with angles in degrees
    angles_rad = np.deg2rad(rama.results.angles)  # (n_models, n_residues, 2)

    # Compute circular mean across all models (axis 0)
    ref_dihedrals = circmean(angles_rad, low=-np.pi, high=np.pi, axis=0)

    print(f"  Computed circular mean φ/ψ for {ref_dihedrals.shape[0]} residues")
    return ref_dihedrals  # (n_residues, 2)


def compute_rg(u: mda.Universe, sel_str: str = "all") -> np.ndarray:
    """Compute radius of gyration per frame.

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe object.
    sel_str : str
        Atom selection string (default: all atoms).

    Returns
    -------
    np.ndarray
        Per-frame Rg in Ångströms, shape (n_frames,).
    """
    n_frames = len(u.trajectory)
    rg = np.zeros(n_frames)
    sel = u.select_atoms(sel_str)

    for i, _ in enumerate(u.trajectory):
        rg[i] = sel.radius_of_gyration()

    return rg


def compute_drmsd(u: mda.Universe, ref_dihedrals: np.ndarray) -> np.ndarray:
    """Compute dihedral RMSD against reference (backbone φ/ψ).

    Applies periodic correction to dihedral differences, wrapping to [−π, π].

    Parameters
    ----------
    u : mda.Universe
        MDAnalysis Universe object.
    ref_dihedrals : np.ndarray
        Reference φ/ψ angles, shape (n_residues, 2) in radians.

    Returns
    -------
    np.ndarray
        Per-frame dRMSD in radians, shape (n_frames,).
    """
    rama = MDAnalysis.analysis.dihedrals.Ramachandran(u.select_atoms("backbone"))
    rama.run()

    # angles are in degrees; convert to radians before computing differences
    angles_rad = np.deg2rad(rama.results.angles)  # (n_frames, n_residues, 2)

    n_frames = angles_rad.shape[0]
    drmsd = np.zeros(n_frames)

    for i in range(n_frames):
        diff = angles_rad[i] - ref_dihedrals  # (n_residues, 2)

        # Periodic correction: wrap differences to [−π, π]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi

        # RMSD: mean squared difference, then sqrt
        drmsd[i] = np.sqrt(np.mean(diff**2))

    return drmsd


def compute_cv_for_xtc(
    xtc_path: Path, top_path: Path, ref_dihedrals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Load XTC and compute dRMSD and Rg.

    Parameters
    ----------
    xtc_path : Path
        Path to XTC trajectory.
    top_path : Path
        Path to topology PDB.
    ref_dihedrals : np.ndarray
        Reference φ/ψ angles from compute_nmr_reference().

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (drmsd, rg) arrays, each shape (n_frames,).
    """
    print(f"  Loading {xtc_path.name}...", end=" ", flush=True)
    u = mda.Universe(str(top_path), str(xtc_path))
    print(f"({len(u.trajectory)} frames)", end=" ", flush=True)

    drmsd = compute_drmsd(u, ref_dihedrals)
    rg = compute_rg(u)

    print("done")
    return drmsd, rg


# ============================================================================
# Plotting
# ============================================================================


def plot_hexbin_landscape(
    drmsd: np.ndarray,
    rg: np.ndarray,
    drmsd_a: np.ndarray,
    rg_a: np.ndarray,
    drmsd_b: np.ndarray,
    rg_b: np.ndarray,
    out_path: Path,
    gridsize: int = 30,
) -> None:
    """Plot hexbin free energy landscape with cluster overlays.

    Parameters
    ----------
    drmsd, rg : np.ndarray
        Collective variables for full ensemble.
    drmsd_a, rg_a : np.ndarray
        CVs for cluster A (black/white edge).
    drmsd_b, rg_b : np.ndarray
        CVs for cluster B (white/black edge).
    out_path : Path
        Output PNG path.
    gridsize : int
        Hexbin gridsize (default 30 for coarse grid).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Hexbin free energy landscape
    hb = ax.hexbin(drmsd, rg, gridsize=gridsize, mincnt=1, cmap="YlGnBu_r")

    # Extract counts and compute F = -ln(counts)
    counts = hb.get_array().copy()
    F = -np.log(counts)
    F = F - np.nanmin(F)  # Shift min to 0

    # Quantize to 0-8 levels
    F_quant = np.clip(np.floor(F), 0, 8).astype(int)

    # Map quantized F to colors using BoundaryNorm
    cmap = matplotlib.colormaps["YlGnBu_r"].resampled(9)

    # Update hexbin with quantized colors
    colors = cmap(F_quant / 9.0)
    hb.set_facecolors(colors)

    # Colorbar
    cbar = fig.colorbar(hb, ax=ax, pad=0.05)
    cbar.set_label(r"$\Delta F\ /\ k_BT$", fontsize=12)
    cbar.set_ticks(np.arange(0, 9))

    # Cluster overlays
    ax.scatter(
        drmsd_a, rg_a, s=10, c="black", edgecolors="white", linewidths=0.4, zorder=5
    )
    ax.scatter(
        drmsd_b, rg_b, s=10, c="white", edgecolors="black", linewidths=0.4, zorder=6
    )

    ax.set_xlabel(r"dRMSD (rad)", fontsize=12)
    ax.set_ylabel(r"$R_g$ (Å)", fontsize=12)
    ax.set_title("Free Energy Landscape (hexbin)", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path.name}")


def plot_contour_landscape(
    drmsd: np.ndarray,
    rg: np.ndarray,
    drmsd_a: np.ndarray,
    rg_a: np.ndarray,
    drmsd_b: np.ndarray,
    rg_b: np.ndarray,
    out_path: Path,
) -> None:
    """Plot KDE contour free energy landscape with cluster overlays.

    Parameters
    ----------
    drmsd, rg : np.ndarray
        Collective variables for full ensemble.
    drmsd_a, rg_a : np.ndarray
        CVs for cluster A (black/white edge).
    drmsd_b, rg_b : np.ndarray
        CVs for cluster B (white/black edge).
    out_path : Path
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # KDE density estimation
    kde = gaussian_kde(np.vstack([drmsd, rg]))

    # Build regular grid
    drmsd_min, drmsd_max = drmsd.min(), drmsd.max()
    rg_min, rg_max = rg.min(), rg.max()

    # 5% padding
    pad_drmsd = 0.05 * (drmsd_max - drmsd_min)
    pad_rg = 0.05 * (rg_max - rg_min)

    drmsd_grid = np.linspace(drmsd_min - pad_drmsd, drmsd_max + pad_drmsd, 100)
    rg_grid = np.linspace(rg_min - pad_rg, rg_max + pad_rg, 100)

    XX, YY = np.meshgrid(drmsd_grid, rg_grid)
    positions = np.vstack([XX.ravel(), YY.ravel()])

    density = kde(positions).reshape(XX.shape)

    # Compute F from density
    F = -np.log(density)
    F = F - np.nanmin(F)  # Shift min to 0
    F = np.clip(F, 0, 8)  # Clip to 0-8

    # Contour plot
    levels = np.arange(0, 9)
    ax.contourf(XX, YY, F, levels=levels, cmap="YlGnBu_r", extend="max")
    ax.contour(XX, YY, F, levels=levels, colors="white", linewidths=0.5, alpha=0.5)

    # Colorbar
    sm = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.BoundaryNorm(np.arange(10), 256),
        cmap="YlGnBu_r",
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05)
    cbar.set_label(r"$\Delta F\ /\ k_BT$", fontsize=12)
    cbar.set_ticks(np.arange(0, 9))

    # Cluster overlays
    ax.scatter(
        drmsd_a, rg_a, s=10, c="black", edgecolors="white", linewidths=0.4, zorder=5
    )
    ax.scatter(
        drmsd_b, rg_b, s=10, c="white", edgecolors="black", linewidths=0.4, zorder=6
    )

    ax.set_xlabel(r"dRMSD (rad)", fontsize=12)
    ax.set_ylabel(r"$R_g$ (Å)", fontsize=12)
    ax.set_title("Free Energy Landscape (KDE contour)", fontsize=13)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {out_path.name}")


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualise MoPrP free energy landscape (dRMSD vs Rg)."
    )
    parser.add_argument(
        "--top-pdb", type=Path, default=DEFAULT_TOP_PDB, help="Topology PDB file"
    )
    parser.add_argument(
        "--traj-xtc", type=Path, default=DEFAULT_TRAJ_XTC, help="Full ensemble XTC"
    )
    parser.add_argument(
        "--nmr-pdb",
        type=Path,
        default=DEFAULT_NMR_PDB,
        help="NMR reference PDB (multi-model)",
    )
    parser.add_argument(
        "--cluster-xtc",
        type=Path,
        default=DEFAULT_CLUSTER_XTC,
        help="Cluster A XTC (black/white edge)",
    )
    parser.add_argument(
        "--cluster-filtered-xtc",
        type=Path,
        default=DEFAULT_CLUSTER_FILTERED_XTC,
        help="Cluster B XTC (white/black edge)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--gridsize", type=int, default=30, help="Hexbin gridsize (default 30)"
    )
    parser.add_argument(
        "--absolute-paths", action="store_true", help="Convert paths to absolute"
    )

    args = parser.parse_args()

    # Resolve paths
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

    # Compute NMR reference
    ref_dihedrals = compute_nmr_reference(nmr_pdb)

    # Compute CVs for full ensemble
    print("Computing CVs for full ensemble...")
    drmsd_full, rg_full = compute_cv_for_xtc(traj_xtc, top_pdb, ref_dihedrals)

    # Compute CVs for cluster A
    print("Computing CVs for cluster A...")
    drmsd_a, rg_a = compute_cv_for_xtc(cluster_xtc, top_pdb, ref_dihedrals)

    # Compute CVs for cluster B
    print("Computing CVs for cluster B...")
    drmsd_b, rg_b = compute_cv_for_xtc(cluster_filtered_xtc, top_pdb, ref_dihedrals)

    # Plot hexbin landscape
    print("\nGenerating plots...")
    plot_hexbin_landscape(
        drmsd_full,
        rg_full,
        drmsd_a,
        rg_a,
        drmsd_b,
        rg_b,
        output_dir / "fe_landscape_hexbin.png",
        gridsize=args.gridsize,
    )

    # Plot contour landscape
    plot_contour_landscape(
        drmsd_full,
        rg_full,
        drmsd_a,
        rg_a,
        drmsd_b,
        rg_b,
        output_dir / "fe_landscape_contour.png",
    )

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()
