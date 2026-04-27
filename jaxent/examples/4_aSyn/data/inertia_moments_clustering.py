#!/usr/bin/env python3
"""Moments-of-inertia based clustering of the aSyn trajectory.

Clusters the trajectory in ternary shape space derived from the principal
moments of inertia of the N-head+NAC region.

Supported clustering methods:
  - kmeans
  - gmm
  - dbscan
  - free_energy_basins

Macro-clusters can optionally be assigned after visual inspection of the
selected method's shape-space plot. The C-tail Rg threshold (default 22.5 Å)
distinguishes extended vs compact C-tail conformers.

Outputs (in --output-dir):
  cluster_labels.npy          per-frame cluster IDs for --method
  cluster_method.json         clustering method metadata
  shape_axes.npy              (n_frames, 2) array [x_ratio=I1/I3, y_ratio=I2/I3]
  ctail_rg.npy                C-tail radius of gyration per frame (Å, all-atom)
  macro_cluster_labels.npy    per-frame macro-cluster strings (names)
  macro_cluster_map.json      echo of the mapping used for macro_cluster_labels.npy
  plots/                      all figures

Usage (two-pass workflow):
    # Pass 1 — inspect cluster layout:
    python inertia_moments_clustering.py \\
        --top-pdb path/to/md_mol_center_coil.pdb \\
        --traj-xtc path/to/tris_all_combined.xtc \\
        --absolute-paths

    # Pass 2 — assign macro-clusters after inspecting the selected method plot:
    python inertia_moments_clustering.py \\
        --top-pdb path/to/md_mol_center_coil.pdb \\
        --traj-xtc path/to/tris_all_combined.xtc \\
        --method dbscan \\
        --cluster-map '{"Rod":[2],"Wavy":[1,3],"Compact":[0,4,5]}' \\
        --absolute-paths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

# ============================================================================
# Constants
# ============================================================================

CLUSTER_COLOURS = {"Rod": "grey", "Wavy": "blue", "Compact": "orange"}
HEXBIN_EXTENT = [-0.08, 1.12, 0.42, 1.10]

# Shape references (consistent with analyse_aSyn_ensemble.py)
DEFAULT_EXP_DIR = Path(__file__).resolve().parent / "_aSyn"
REFERENCES = {
    "Rod (AF)": DEFAULT_EXP_DIR / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb",
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
COARSE_REFERENCE_TARGETS = {
    "Rod": "Rod (AF)",
    "Wavy": "Hairpin",
    "Compact": "Compact",
}
DEFAULT_SHAPE_RESID_CANDIDATES = ["1:61", "1:95", "1:105", "1:114", "1:135"]

# ============================================================================
# Computation
# ============================================================================


def compute_inertia_ratios(
    u: mda.Universe, sel_str: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame principal moment ratios via MDAnalysis moment_of_inertia().

    Returns x_ratio (I1/I3), y_ratio (I2/I3), I1, I2, I3 arrays (all n_frames,).
    Eigenvalues from eigvalsh are sorted ascending so I1 <= I2 <= I3.
    """
    sel = u.select_atoms(sel_str)
    n_frames = len(u.trajectory)
    x_ratio = np.zeros(n_frames)
    y_ratio = np.zeros(n_frames)
    moments = np.zeros((n_frames, 3))

    print(f"Computing inertia tensors for {n_frames} frames ({sel.n_atoms} atoms selected)...")
    for i, _ts in enumerate(u.trajectory):
        evals = np.linalg.eigvalsh(sel.moment_of_inertia())
        moments[i] = evals
        x_ratio[i] = evals[0] / evals[2]
        y_ratio[i] = evals[1] / evals[2]
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_frames}")

    return x_ratio, y_ratio, moments[:, 0], moments[:, 1], moments[:, 2]


def compute_ctail_rg(u: mda.Universe, ctail_sel_str: str) -> np.ndarray:
    """Compute per-frame C-tail radius of gyration in Angstroms (all-atom, mass-weighted)."""
    sel = u.select_atoms(ctail_sel_str)
    n_frames = len(u.trajectory)
    rg = np.zeros(n_frames)
    print(f"Computing C-tail Rg for {n_frames} frames ({sel.n_atoms} atoms selected)...")
    for i, _ts in enumerate(u.trajectory):
        rg[i] = sel.radius_of_gyration()
    return rg


def compute_reference_ratios(ref_dict: dict[str, Path], sel_str: str) -> dict[str, tuple[float, float]]:
    """Compute I1/I3 and I2/I3 for single-frame reference PDBs."""
    ratios = {}
    for name, pdb_path in ref_dict.items():
        if not pdb_path.exists():
            continue
        try:
            ref = mda.Universe(str(pdb_path))
            sel = ref.select_atoms(sel_str)
            if sel.n_atoms == 0:
                continue
            evals = np.linalg.eigvalsh(sel.moment_of_inertia())
            ratios[name] = (evals[0] / evals[2], evals[1] / evals[2])
        except Exception as e:
            print(f"Warning: Could not compute ratios for {name}: {e}")
    return ratios


def compute_dynamic_reference_ratios(dyn_ref_dict: dict[str, Path], sel_str: str) -> dict[str, np.ndarray]:
    """Compute I1/I3 and I2/I3 for multi-frame dynamic reference PDBs."""
    ratios = {}
    for name, pdb_path in dyn_ref_dict.items():
        if not pdb_path.exists():
            continue
        try:
            ref = mda.Universe(str(pdb_path))
            sel = ref.select_atoms(sel_str)
            if sel.n_atoms == 0:
                continue
            n_frames = len(ref.trajectory)
            coords = np.zeros((n_frames, 2))
            for i, _ts in enumerate(ref.trajectory):
                evals = np.linalg.eigvalsh(sel.moment_of_inertia())
                coords[i] = [evals[0] / evals[2], evals[1] / evals[2]]
            ratios[name] = coords
        except Exception as e:
            print(f"Warning: Could not compute dynamic ratios for {name}: {e}")
    return ratios


def parse_resid_range(range_str: str) -> tuple[int, int]:
    start_str, end_str = range_str.split(":")
    return int(start_str), int(end_str)


def format_selection(range_str: str, atom_names: str = "name CA") -> str:
    start, end = parse_resid_range(range_str)
    return f"resid {start}-{end} and {atom_names}"


def slugify_range(range_str: str) -> str:
    start, end = parse_resid_range(range_str)
    return f"{start}_{end}"


def parse_shape_resid_candidates(shape_resids: str, shape_resids_grid: str | None) -> list[str]:
    if not shape_resids_grid:
        return [shape_resids]

    candidates = []
    for item in shape_resids_grid.split(","):
        item = item.strip()
        if item:
            candidates.append(item)
    return candidates or [shape_resids]


def cluster_centers_from_labels(features_xy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    centers = []
    valid_labels = sorted(int(v) for v in np.unique(labels) if v >= 0)
    for label in valid_labels:
        mask = labels == label
        centers.append(features_xy[mask].mean(axis=0))
    return np.asarray(centers, dtype=float)


def ensure_cluster_centers(features_xy: np.ndarray, labels: np.ndarray, centers_xy: np.ndarray | None) -> np.ndarray:
    if centers_xy is not None and len(centers_xy) > 0:
        return np.asarray(centers_xy, dtype=float)
    return cluster_centers_from_labels(features_xy, labels)


def invert_macro_mapping(cluster_to_macro: dict[int, str]) -> dict[str, list[int]]:
    macro_map: dict[str, list[int]] = {}
    for cluster_id, macro in sorted(cluster_to_macro.items()):
        macro_map.setdefault(macro, []).append(cluster_id)
    return macro_map


def assign_macro_labels_by_reference(
    labels: np.ndarray,
    centers_xy: np.ndarray,
    ref_ratios: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[int, str]]:
    macro_clusters = np.full(labels.shape, "Unassigned", dtype=object)
    cluster_to_macro: dict[int, str] = {}
    valid_labels = sorted(int(v) for v in np.unique(labels) if v >= 0)
    center_lookup = {cluster_id: centers_xy[idx] for idx, cluster_id in enumerate(valid_labels)}
    available_targets = {
        macro: ref_ratios[ref_name]
        for macro, ref_name in COARSE_REFERENCE_TARGETS.items()
        if ref_name in ref_ratios
    }

    for cluster_id in valid_labels:
        center = center_lookup[cluster_id]
        if available_targets:
            best_macro = min(
                available_targets,
                key=lambda macro: np.linalg.norm(center - np.asarray(available_targets[macro])),
            )
        else:
            x_val, y_val = center
            if x_val < 0.16 and y_val > 0.90:
                best_macro = "Rod"
            elif x_val > 0.40 or y_val < 0.82:
                best_macro = "Compact"
            else:
                best_macro = "Wavy"
        cluster_to_macro[cluster_id] = best_macro
        macro_clusters[labels == cluster_id] = best_macro

    return macro_clusters, cluster_to_macro


def assign_manual_macro_labels(
    labels: np.ndarray,
    cluster_map: dict[str, list[int]] | None,
) -> np.ndarray | None:
    if cluster_map is None:
        return None
    macro_clusters = np.full(labels.shape, "Unassigned", dtype=object)
    for macro_name, cluster_indices in cluster_map.items():
        for cluster_id in cluster_indices:
            macro_clusters[labels == cluster_id] = macro_name
    return macro_clusters


# ============================================================================
# Shared plot helpers
# ============================================================================


def _draw_shape_boundary(ax: plt.Axes) -> None:
    """Draw triangular shape-space boundary, vertex labels, and guide line."""
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [1, 1, 0.5, 1]
    ax.plot(triangle_x, triangle_y, color="black", linestyle="--", lw=2,
            label="Shape Boundary", zorder=5)
    label_kw = dict(fontsize=11, zorder=6,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.text(0, 1.04, r"Rod ($I_1 = 0$)", ha="center", **label_kw)
    ax.text(1, 1.04, r"Sphere ($I_1 = I_2 = I_3$)", ha="center", **label_kw)
    ax.text(0.5, 0.46, r"Disk ($I_1 = I_2$)", ha="center", **label_kw)
    ax.plot([0.5, 1], [0.5, 1], color="gray", lw=1, ls=":", alpha=0.7, zorder=4)
    ax.set_xlim(-0.08, 1.12)
    ax.set_ylim(0.42, 1.10)
    ax.set_xlabel(r"$I_1/I_3$", fontsize=14)
    ax.set_ylabel(r"$I_2/I_3$", fontsize=14)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.grid(True, color="lightgray", lw=0.5, zorder=0)
    ax.tick_params(width=1.5)


def _overlay_references(
    ax: plt.Axes,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
    is_legend: bool = False
) -> None:
    """Overlay single and dynamic reference structures on the shape space."""
    if ref_ratios:
        for name, (x, y) in ref_ratios.items():
            ax.scatter(
                x, y,
                marker=REF_MARKERS[name], s=200,
                c=REF_COLORS[name], zorder=10,
                edgecolors="black", linewidths=1.5,
                label=name if is_legend else None
            )
    if dynamic_ref_ratios:
        for name, coords in dynamic_ref_ratios.items():
            ax.scatter(
                coords[:, 0], coords[:, 1],
                marker=DYN_REF_MARKERS.get(name, "o"), s=40,
                c=DYN_REF_COLORS.get(name, "#ff7f00"), zorder=9,
                edgecolors="white", linewidths=0.5, alpha=0.6,
                label=name if is_legend else None
            )


def _tab20_palette(n: int) -> dict[int, tuple]:
    try:
        cmap = matplotlib.colormaps["tab20"].resampled(n)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", n)
    return {c: cmap(c / max(n - 1, 1)) for c in range(n)}


def _savefig(path: Path, **kwargs) -> None:
    """Ensure the parent directory exists before writing a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, **kwargs)


# ============================================================================
# Plot functions (exact from reference notebook)
# ============================================================================


def plot_shape_space_ctail(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    rg_ctail: np.ndarray,
    plots_dir: Path,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x_ratio, y_ratio, s=4, alpha=0.6, c=rg_ctail, cmap="inferno",
                    edgecolors="none", label="Structural Frames", zorder=3)
    _draw_shape_boundary(ax)
    _overlay_references(ax, ref_ratios, dynamic_ref_ratios, is_legend=True)
    cbar = fig.colorbar(sc, ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label(r"C-Tail $R_g$ (Å)", fontsize=12)
    ax.set_title("Principal Moments of Inertia (Shape Space)", fontsize=14, pad=10)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    _savefig(plots_dir / "shape_space_ctail_rg.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shape_space_ctail_rg.png")


def plot_free_energy_landscape(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    plots_dir: Path,
    n_grid: int = 200,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
) -> None:
    """Free energy landscape: ΔF = -ln P(I1/I3, I2/I3), shifted so min=0 (kT units).

    Uses Gaussian KDE so the density (and therefore free energy) is defined
    everywhere in the grid, naturally filling the triangle to its edges.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    _, XX, YY, density, F, _ = compute_free_energy_grid(x_ratio, y_ratio, n_grid=n_grid)

    levels = list(range(9))  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
    cf = ax.contourf(XX, YY, F, levels=levels, cmap="YlGnBu_r", extend="max")
    ax.contour(XX, YY, F, levels=levels, colors="white", linewidths=0.5, alpha=0.5)

    cbar = fig.colorbar(cf, ax=ax, ticks=levels, pad=0.05, shrink=0.8)
    cbar.set_label(r"Free Energy/$k_BT$", fontsize=13)

    _draw_shape_boundary(ax)
    _overlay_references(ax, ref_ratios, dynamic_ref_ratios, is_legend=True)
    ax.set_title("Principal Moments of Inertia Free Energy Landscape", fontsize=14, pad=10)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    _savefig(plots_dir / "free_energy_landscape.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved free_energy_landscape.png")


def plot_free_energy_landscape_hexbin(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    plots_dir: Path,
    gridsize: int = 50,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
) -> None:
    """Free energy landscape using hexbin: ΔF = -ln P(I1/I3, I2/I3), shifted so min=0.

    This provides a more discretized view compared to KDE.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Initial plot to get counts
    hb = ax.hexbin(x_ratio, y_ratio, gridsize=gridsize, cmap="YlGnBu_r",
                   mincnt=1, edgecolors='none')

    # Extract counts and compute free energy
    counts = hb.get_array()
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -np.log(counts)
    F -= np.nanmin(F)

    # Update hexbin colors with Free Energy quantized to levels 0-8
    levels = list(range(9))
    norm = matplotlib.colors.BoundaryNorm(levels, ncolors=256, extend="max")

    hb.set_array(F)
    hb.set_norm(norm)

    cbar = fig.colorbar(hb, ax=ax, ticks=levels, pad=0.05, shrink=0.8, extend="max")
    cbar.set_label(r"Free Energy/$k_BT$", fontsize=13)

    _draw_shape_boundary(ax)
    _overlay_references(ax, ref_ratios, dynamic_ref_ratios, is_legend=True)
    ax.set_title("Principal Moments Free Energy (Hexbin)", fontsize=14, pad=10)
    plt.tight_layout()
    _savefig(plots_dir / "free_energy_landscape_hexbin.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved free_energy_landscape_hexbin.png")


def _triangle_mask(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    return (yy >= xx) & (yy >= 1.0 - xx) & (yy <= 1.0)


def compute_free_energy_grid(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return KDE/free-energy grid in the valid triangular shape space."""
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(np.vstack([x_ratio, y_ratio]), bw_method="scott")
    xg = np.linspace(-0.08, 1.08, n_grid)
    yg = np.linspace(0.42, 1.10, n_grid)
    xx, yy = np.meshgrid(xg, yg)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    valid = _triangle_mask(xx, yy)
    density[~valid] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        free_energy = -np.log(density)
    finite = np.isfinite(free_energy) & valid
    free_energy[finite] -= np.nanmin(free_energy[finite])
    free_energy[~valid] = np.nan
    return xg, xx, yy, density, free_energy, valid


def _remap_labels(labels: np.ndarray) -> np.ndarray:
    """Map non-noise labels to compact 0..n-1 integers and preserve DBSCAN noise as -1."""
    remapped = np.full(labels.shape, -1, dtype=int)
    unique = sorted(int(v) for v in np.unique(labels) if v >= 0)
    for new_id, old_id in enumerate(unique):
        remapped[labels == old_id] = new_id
    return remapped


def _mode(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=int)
    if vals.size == 0:
        return -1
    unique, counts = np.unique(vals, return_counts=True)
    return int(unique[np.argmax(counts)])


def _nearest_grid_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(axis, values, side="left")
    idx = np.clip(idx, 1, len(axis) - 1)
    left = axis[idx - 1]
    right = axis[idx]
    choose_left = np.abs(values - left) <= np.abs(values - right)
    return np.where(choose_left, idx - 1, idx)


def cluster_kmeans(features_xy: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(features_xy)
    return labels.astype(int), model.cluster_centers_


def cluster_gmm(features_xy: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    model = GaussianMixture(n_components=n_clusters, covariance_type="full", random_state=42)
    labels = model.fit_predict(features_xy)
    return labels.astype(int), model.means_


def select_gmm_by_bic(
    features_xy: np.ndarray,
    min_components: int,
    max_components: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a GMM range and return the BIC-optimal component count."""
    component_counts = np.arange(min_components, max_components + 1, dtype=int)
    bic_scores = []
    best_bic = np.inf
    best_n = min_components
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None

    for n_components in component_counts:
        model = GaussianMixture(
            n_components=n_components, covariance_type="full", random_state=42
        )
        model.fit(features_xy)
        bic = model.bic(features_xy)
        bic_scores.append(bic)
        if bic < best_bic:
            best_bic = bic
            best_n = n_components
            best_labels = model.predict(features_xy).astype(int)
            best_centers = model.means_

    if best_labels is None or best_centers is None:
        raise RuntimeError("GMM BIC selection failed to produce a fitted model.")

    return best_n, np.asarray(bic_scores, dtype=float), best_labels, best_centers


def cluster_dbscan(
    features_xy: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, None]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = _remap_labels(model.fit_predict(features_xy))
    return labels, None


def cluster_free_energy_basins(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    n_basins: int,
    n_grid: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | int]]:
    """Assign each frame to a KDE basin using steepest-ascent on the density grid."""
    from scipy import ndimage

    xg, xx, yy, density, _free_energy, valid = compute_free_energy_grid(
        x_ratio, y_ratio, n_grid=n_grid
    )
    footprint = np.ones((3, 3), dtype=bool)
    maxima_mask = (density == ndimage.maximum_filter(density, footprint=footprint)) & valid
    maxima_idx = np.argwhere(maxima_mask)
    if maxima_idx.size == 0:
        raise RuntimeError("No free-energy basin maxima found in KDE grid.")

    peak_densities = density[maxima_mask]
    order = np.argsort(peak_densities)[::-1]
    maxima_idx = maxima_idx[order[: max(1, n_basins)]]

    basin_grid = np.full(density.shape, -1, dtype=int)
    peak_lookup = {tuple(idx): basin_id for basin_id, idx in enumerate(maxima_idx)}

    for start in np.argwhere(valid):
        path: list[tuple[int, int]] = []
        current = tuple(int(v) for v in start)
        while basin_grid[current] == -1:
            if current in peak_lookup:
                basin_id = peak_lookup[current]
                break
            path.append(current)
            i, j = current
            next_cell = current
            next_density = density[current]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if (
                        0 <= ni < density.shape[0]
                        and 0 <= nj < density.shape[1]
                        and valid[ni, nj]
                        and density[ni, nj] > next_density
                    ):
                        next_density = density[ni, nj]
                        next_cell = (ni, nj)
            if next_cell == current:
                dists = np.sum((maxima_idx - np.array(current)) ** 2, axis=1)
                basin_id = int(np.argmin(dists))
                break
            current = next_cell
        else:
            basin_id = int(basin_grid[current])

        for cell in path:
            basin_grid[cell] = basin_id
        basin_grid[current] = basin_id

    x_idx = _nearest_grid_indices(xg, x_ratio)
    y_axis = yy[:, 0]
    y_idx = _nearest_grid_indices(y_axis, y_ratio)
    frame_labels = basin_grid[y_idx, x_idx]
    frame_labels = _remap_labels(frame_labels)

    centers = np.column_stack([xx[maxima_idx[:, 0], maxima_idx[:, 1]],
                               yy[maxima_idx[:, 0], maxima_idx[:, 1]]])
    return frame_labels, {"centers": centers, "basin_grid": basin_grid, "density": density}


def run_clustering_method(
    method: str,
    features_xy: np.ndarray,
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    n_clusters: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    basin_grid_size: int,
    gmm_bic_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if method == "kmeans":
        return cluster_kmeans(features_xy, n_clusters)
    if method == "gmm":
        if gmm_bic_range is not None:
            _best_n, _bic_scores, labels, centers = select_gmm_by_bic(
                features_xy, gmm_bic_range[0], gmm_bic_range[1]
            )
            return labels, centers
        return cluster_gmm(features_xy, n_clusters)
    if method == "dbscan":
        return cluster_dbscan(features_xy, dbscan_eps, dbscan_min_samples)
    if method == "free_energy_basins":
        labels, basin_meta = cluster_free_energy_basins(
            x_ratio, y_ratio, n_clusters, basin_grid_size
        )
        return labels, basin_meta["centers"]
    raise ValueError(f"Unsupported clustering method: {method}")


def plot_shape_indices_histograms(
    w_rod: np.ndarray,
    w_sphere: np.ndarray,
    w_disk: np.ndarray,
    plots_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Rod Axis", "Sphere Axis", "Disk Axis"]
    data = [w_rod, w_sphere, w_disk]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    for i, ax in enumerate(axes):
        sns.histplot(data[i], kde=True, color=colors[i], ax=ax, stat="density", alpha=0.4)
        ax.set_title(titles[i], fontsize=14)
        ax.set_xlabel("Shape Weight (0 to 1)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    _savefig(plots_dir / "shape_indices_histograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shape_indices_histograms.png")


def _cluster_palette(labels: np.ndarray) -> dict[int, tuple]:
    valid_labels = sorted(int(v) for v in np.unique(labels) if v >= 0)
    palette = _tab20_palette(max(len(valid_labels), 1))
    mapping = {label: palette[i] for i, label in enumerate(valid_labels)}
    if np.any(labels < 0):
        mapping[-1] = (0.55, 0.55, 0.55, 1.0)
    return mapping


def _build_cluster_face_colors(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    cluster_labels: np.ndarray,
    gridsize: int = 50,
) -> tuple[list[np.ndarray], dict[int, tuple]]:
    fig_tmp, ax_tmp = plt.subplots()
    hb_fe = ax_tmp.hexbin(x_ratio, y_ratio, gridsize=gridsize, extent=HEXBIN_EXTENT, mincnt=1)
    counts = hb_fe.get_array()

    hb_cl = ax_tmp.hexbin(
        x_ratio,
        y_ratio,
        C=cluster_labels,
        gridsize=gridsize,
        extent=HEXBIN_EXTENT,
        reduce_C_function=_mode,
        mincnt=1,
    )
    bin_labels = hb_cl.get_array().astype(int)
    plt.close(fig_tmp)

    with np.errstate(divide="ignore", invalid="ignore"):
        free_energy = -np.log(counts)
    free_energy -= np.nanmin(free_energy)
    free_energy_quant = np.clip(np.floor(free_energy), 0, 8).astype(int)

    palette = _cluster_palette(cluster_labels)
    face_colors: list[np.ndarray] = []
    for i, cluster_id in enumerate(bin_labels):
        base_rgb = np.array(palette.get(int(cluster_id), palette.get(-1, (0.5, 0.5, 0.5, 1.0)))[:3])
        mix = free_energy_quant[i] / 9.0
        face_colors.append(base_rgb * (1.0 - mix) + np.array([1.0, 1.0, 1.0]) * mix)
    return face_colors, palette


def plot_cluster_shape_space(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    cluster_labels: np.ndarray,
    centers_xy: np.ndarray | None,
    method: str,
    plots_dir: Path,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_shape_boundary(ax)
    palette = _cluster_palette(cluster_labels)
    point_colors = [palette[int(label)] for label in cluster_labels]
    ax.scatter(x_ratio, y_ratio, c=point_colors, alpha=0.6, edgecolors="none", s=15, zorder=3)

    if centers_xy is not None and len(centers_xy) > 0:
        ax.scatter(
            centers_xy[:, 0],
            centers_xy[:, 1],
            c="red",
            marker="X",
            s=150,
            edgecolor="black",
            lw=1.5,
            zorder=6,
            label="Cluster Centers",
        )

    _overlay_references(ax, ref_ratios, dynamic_ref_ratios, is_legend=True)

    legend_patches = []
    for label in sorted(np.unique(cluster_labels)):
        label_int = int(label)
        display = "Noise" if label_int == -1 else f"Cluster {label_int}"
        legend_patches.append(mpatches.Patch(color=palette[label_int], label=display))
    ax.legend(handles=legend_patches, loc="upper left", fontsize=10, title=method)
    ax.set_title(f"Shape Space Clusters ({method})", fontsize=14, pad=10)
    plt.tight_layout()
    _savefig(plots_dir / f"shape_space_{method}.png", format="png", dpi=1200,
                bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Saved shape_space_{method}.png")


def plot_cluster_hexbin(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    cluster_labels: np.ndarray,
    method: str,
    plots_dir: Path,
    gridsize: int = 50,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
) -> None:
    """Cluster hexbin plot with hue per dominant label and whitened by free energy."""
    face_colors, palette = _build_cluster_face_colors(
        x_ratio, y_ratio, cluster_labels, gridsize=gridsize
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_shape_boundary(ax)
    hb = ax.hexbin(
        x_ratio, y_ratio, gridsize=gridsize, extent=HEXBIN_EXTENT, mincnt=1, edgecolors="none"
    )
    hb.set_array(None)
    hb.set_facecolors(face_colors)

    _overlay_references(ax, ref_ratios, dynamic_ref_ratios, is_legend=True)

    legend_patches = []
    for label in sorted(np.unique(cluster_labels)):
        label_int = int(label)
        display = "Noise" if label_int == -1 else f"Cluster {label_int}"
        legend_patches.append(mpatches.Patch(color=palette[label_int], label=display))
    ax.legend(handles=legend_patches, loc="lower right", title=method, fontsize=10)
    ax.set_title(f"{method} Clusters with Quantized Energy Fading", fontsize=14, pad=10)
    plt.tight_layout()
    _savefig(plots_dir / f"{method}_hexbin_energy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {method}_hexbin_energy.png")


def plot_method_comparison_hexbin(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    method_labels: dict[str, np.ndarray],
    plots_dir: Path,
    gridsize: int = 50,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
) -> None:
    methods = ["kmeans", "gmm", "dbscan", "free_energy_basins"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    for ax, method in zip(axes.flat, methods):
        labels = method_labels[method]
        face_colors, palette = _build_cluster_face_colors(
            x_ratio, y_ratio, labels, gridsize=gridsize
        )
        _draw_shape_boundary(ax)
        hb = ax.hexbin(
            x_ratio, y_ratio, gridsize=gridsize, extent=HEXBIN_EXTENT, mincnt=1, edgecolors="none"
        )
        hb.set_array(None)
        hb.set_facecolors(face_colors)

        _overlay_references(ax, ref_ratios, dynamic_ref_ratios, is_legend=(ax == axes.flat[0]))

        ax.set_title(method, fontsize=13)
        legend_patches = []
        for label in sorted(np.unique(labels)):
            label_int = int(label)
            display = "Noise" if label_int == -1 else f"C{label_int}"
            legend_patches.append(mpatches.Patch(color=palette[label_int], label=display))
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title=method)

    fig.suptitle("Shape-Space Cluster Method Comparison", fontsize=16)
    _savefig(plots_dir / "cluster_methods_hexbin_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved cluster_methods_hexbin_comparison.png")


def plot_gmm_bic(
    component_counts: np.ndarray,
    bic_scores: np.ndarray,
    best_n: int,
    plots_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(component_counts, bic_scores, marker="o", lw=2, color="#1f4e79")
    ax.axvline(best_n, color="#c0392b", ls="--", lw=1.5, label=f"Best n = {best_n}")
    ax.scatter([best_n], [bic_scores[np.where(component_counts == best_n)[0][0]]],
               color="#c0392b", s=70, zorder=4)
    ax.set_xlabel("GMM Components")
    ax.set_ylabel("BIC")
    ax.set_title("GMM Model Selection by BIC")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    _savefig(plots_dir / "bic_vs_n.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved bic_vs_n.png")


def plot_macro_hexbin(
    x_ratio: np.ndarray,
    y_ratio: np.ndarray,
    macro_clusters: np.ndarray,
    plots_dir: Path,
    gridsize: int = 50,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    dynamic_ref_ratios: dict[str, np.ndarray] | None = None,
) -> None:
    """Macro-cluster hexbin plot: each meta-cluster gets its own hue tending to white with energy.
    """
    from scipy import stats
    from matplotlib.colors import to_rgb

    # Use consistent extent to ensure bins align across different calls
    extent = [-0.08, 1.12, 0.42, 1.10]

    unique_macros = ["Rod", "Wavy", "Compact", "Unassigned"]
    macro_to_id = {name: i for i, name in enumerate(unique_macros)}
    id_to_macro = {i: name for i, name in enumerate(unique_macros)}

    macro_ids = np.array([macro_to_id.get(m, 3) for m in macro_clusters])

    fig_tmp, ax_tmp = plt.subplots()
    # 1. Energy counts
    hb_fe = ax_tmp.hexbin(x_ratio, y_ratio, gridsize=gridsize, extent=extent, mincnt=1)
    counts = hb_fe.get_array()

    # 2. Dominant macro-cluster
    def get_mode(x):
        m = stats.mode(x, keepdims=True)
        return m.mode[0]

    hb_cl = ax_tmp.hexbin(x_ratio, y_ratio, C=macro_ids,
                          gridsize=gridsize, extent=extent, reduce_C_function=get_mode, mincnt=1)
    bin_macro_ids = hb_cl.get_array().astype(int)
    plt.close(fig_tmp)

    # 3. Quantized Free Energy
    with np.errstate(divide="ignore", invalid="ignore"):
        F = -np.log(counts)
    F -= np.nanmin(F)
    F_quant = np.clip(np.floor(F), 0, 8).astype(int)

    # 4. Map to colors
    macro_base_colors = {
        "Rod": to_rgb(CLUSTER_COLOURS["Rod"]),
        "Wavy": to_rgb(CLUSTER_COLOURS["Wavy"]),
        "Compact": to_rgb(CLUSTER_COLOURS["Compact"]),
        "Unassigned": (0.5, 0.5, 0.5)
    }

    face_colors = []
    for i in range(len(bin_macro_ids)):
        mname = id_to_macro[bin_macro_ids[i]]
        fe_lvl = F_quant[i]
        base_rgb = np.array(macro_base_colors.get(mname, (0.5, 0.5, 0.5)))
        mix = fe_lvl / 9.0
        color = base_rgb * (1.0 - mix) + np.array([1.0, 1.0, 1.0]) * mix
        face_colors.append(color)

    # 5. Final plot
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_shape_boundary(ax)

    hb = ax.hexbin(x_ratio, y_ratio, gridsize=gridsize, extent=extent, mincnt=1, edgecolors='none')
    hb.set_array(None)
    hb.set_facecolors(face_colors)

    _overlay_references(ax, ref_ratios, dynamic_ref_ratios, is_legend=True)

    # Legend
    legend_patches = [
        mpatches.Patch(color=CLUSTER_COLOURS[name], label=name)
        for name in ["Rod", "Wavy", "Compact"]
    ]
    ax.legend(handles=legend_patches, loc="lower right", title="Macro-Clusters", fontsize=10)

    ax.set_title("Macro-Clusters with Quantized Energy Fading", fontsize=14, pad=10)
    plt.tight_layout()
    _savefig(plots_dir / "macro_hexbin_energy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved macro_hexbin_energy.png")


def plot_ca_traces(
    u: mda.Universe,
    shape_sel_str: str,
    cluster_labels: np.ndarray,
    features_xy: np.ndarray,
    centers_xy: np.ndarray | None,
    method: str,
    plots_dir: Path,
) -> None:
    valid_clusters = sorted(int(v) for v in np.unique(cluster_labels) if v >= 0)
    if not valid_clusters:
        print(f"Skipped cluster_ca_traces_{method}.png (no non-noise clusters).")
        return

    centroid_frames = []
    for c in valid_clusters:
        mask = cluster_labels == c
        pts = features_xy[mask]
        if centers_xy is not None and c < len(centers_xy):
            center = centers_xy[c]
            distances = np.linalg.norm(pts - center, axis=1)
            frame_idx = int(np.where(mask)[0][np.argmin(distances)])
        else:
            frame_idx = int(np.where(mask)[0][0])
        centroid_frames.append(frame_idx)

    ca_sel = u.select_atoms("name CA and (" + shape_sel_str + ")")
    palette = _cluster_palette(cluster_labels)

    ncols = min(3, len(valid_clusters))
    nrows = (len(valid_clusters) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for idx, fidx in enumerate(centroid_frames):
        cluster_id = valid_clusters[idx]
        u.trajectory[fidx]
        pos = ca_sel.positions  # Angstroms, shape (n_ca, 3)
        color = palette[cluster_id]
        ax = axes_flat[idx]
        ax.plot(pos[:, 0], pos[:, 1], color=color, lw=1.5, alpha=0.8)
        ax.scatter(pos[:, 0], pos[:, 1], s=8, color=color, alpha=0.7)
        ax.set_title(f"Cluster {cluster_id} (Frame {fidx})")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(centroid_frames), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Representative CA Traces for {method}", fontsize=16)
    _savefig(plots_dir / f"cluster_ca_traces_{method}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cluster_ca_traces_{method}.png")


def plot_ca_traces_macro(
    u: mda.Universe,
    shape_sel_str: str,
    cluster_labels: np.ndarray,
    cluster_to_macro: dict[int, str],
    features_xy: np.ndarray,
    centers_xy: np.ndarray | None,
    method: str,
    plots_dir: Path,
) -> None:
    valid_clusters = sorted(int(v) for v in np.unique(cluster_labels) if v >= 0)
    if not valid_clusters:
        print(f"Skipped cluster_ca_traces_macro_{method}.png (no non-noise clusters).")
        return

    centroid_frames = []
    for c in valid_clusters:
        mask = cluster_labels == c
        pts = features_xy[mask]
        if centers_xy is not None and c < len(centers_xy):
            center = centers_xy[c]
            distances = np.linalg.norm(pts - center, axis=1)
            frame_idx = int(np.where(mask)[0][np.argmin(distances)])
        else:
            frame_idx = int(np.where(mask)[0][0])
        centroid_frames.append(frame_idx)

    ca_sel = u.select_atoms("name CA and (" + shape_sel_str + ")")

    ncols = min(3, len(valid_clusters))
    nrows = (len(valid_clusters) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             constrained_layout=True)
    axes_flat = np.array(axes).flatten()

    for idx, fidx in enumerate(centroid_frames):
        cluster_id = valid_clusters[idx]
        macro_name = cluster_to_macro.get(cluster_id, "Unassigned")
        color = CLUSTER_COLOURS.get(macro_name, "grey")

        u.trajectory[fidx]
        pos = ca_sel.positions  # Angstroms, shape (n_ca, 3)
        ax = axes_flat[idx]
        ax.plot(pos[:, 0], pos[:, 1], color=color, lw=1.5, alpha=0.8)
        ax.scatter(pos[:, 0], pos[:, 1], s=8, color=color, alpha=0.7)
        ax.set_title(f"C{cluster_id} -> {macro_name} (Frame {fidx})")
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(len(centroid_frames), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Representative CA Traces by Macro-Cluster ({method})", fontsize=16)
    _savefig(plots_dir / f"cluster_ca_traces_macro_{method}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cluster_ca_traces_macro_{method}.png")


def plot_moments_histograms(
    I1: np.ndarray, I2: np.ndarray, I3: np.ndarray, plots_dir: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    titles = [r"$I_1$ (Smallest)", r"$I_2$", r"$I_3$ (Largest)"]
    data = [I1, I2, I3]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, ax in enumerate(axes):
        sns.histplot(data[i], kde=True, color=colors[i], ax=ax, stat="density", alpha=0.5)
        ax.set_title(titles[i])
        ax.set_xlabel("Moment of Inertia")
        ax.set_ylabel("Density")
    plt.tight_layout()
    _savefig(plots_dir / "moments_histograms.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved moments_histograms.png")


def plot_ctail_rg_clusters_hist(
    rg_ctail: np.ndarray,
    cluster_labels: np.ndarray,
    plots_dir: Path,
    ctail_threshold: float | None = None,
) -> None:
    palette = _cluster_palette(cluster_labels)
    df = pd.DataFrame({"Rg_Ctail (Å)": rg_ctail, "Cluster": cluster_labels})
    means = df.groupby("Cluster")["Rg_Ctail (Å)"].mean()

    plt.figure(figsize=(10, 6))
    ax = sns.histplot(
        data=df, x="Rg_Ctail (Å)", hue="Cluster",
        bins=25, palette=palette, element="step", stat="density",
        common_norm=False, kde=True, alpha=0.4, linewidth=1.5,
    )
    for cluster_id, mean_val in means.items():
        plt.axvline(x=mean_val, color=palette[cluster_id], linestyle="--",
                    linewidth=2.5, alpha=0.8, label=f"C{cluster_id} mean: {mean_val:.1f} Å")
    
    if ctail_threshold is not None:
        plt.axvline(x=ctail_threshold, color="black", linestyle=":", linewidth=2, alpha=0.9)
        plt.text(ctail_threshold, plt.gca().get_ylim()[1]*0.95, "Midpoint", 
                 color="black", ha="right", fontweight="bold", rotation=90)
    plt.title("C-Tail Radius of Gyration Distribution by Shape Cluster", fontsize=16, pad=15)
    plt.xlabel(r"C-Tail $R_g$ (Å)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    try:
        sns.move_legend(ax, "upper right", bbox_to_anchor=(1.3, 1))
    except Exception:
        pass
    plt.legend(title="means", loc="upper left", bbox_to_anchor=(1.05, 0.7))
    sns.despine()
    plt.tight_layout()
    _savefig(plots_dir / "ctail_rg_clusters_hist.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved ctail_rg_clusters_hist.png")


def plot_ctail_rg_clusters_rows(
    rg_ctail: np.ndarray,
    cluster_labels: np.ndarray,
    plots_dir: Path,
    ctail_threshold: float | None = None,
) -> None:
    palette = _cluster_palette(cluster_labels)
    df = pd.DataFrame({"Rg_Ctail (Å)": rg_ctail, "Cluster": cluster_labels})
    means = df.groupby("Cluster")["Rg_Ctail (Å)"].mean()
    ordered_labels = sorted(int(v) for v in np.unique(cluster_labels))

    fig, axes = plt.subplots(len(ordered_labels), 1, figsize=(8, 3 * len(ordered_labels)), sharex=True)
    if len(ordered_labels) == 1:
        axes = [axes]

    for i, cluster_id in enumerate(ordered_labels):
        ax = axes[i]
        cluster_data = df[df["Cluster"] == cluster_id]
        mean_val = means[cluster_id]
        color = palette[cluster_id]
        sns.histplot(data=cluster_data, x="Rg_Ctail (Å)", color=color, bins=25,
                     element="step", stat="density", kde=True, alpha=0.4,
                     linewidth=1.5, ax=ax)
        ax.axvline(x=mean_val, color=color, linestyle="--", linewidth=2.5, alpha=0.8)
        if ctail_threshold is not None:
            ax.axvline(x=ctail_threshold, color="black", linestyle=":", linewidth=2, alpha=0.6)
        label_text = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
        ax.set_ylabel(f"{label_text}\nDensity")
        ylim = ax.get_ylim()
        ax.annotate(f"Mean: {mean_val:.1f} Å",
                    xy=(mean_val, ylim[1] * 0.8), xytext=(5, 0),
                    textcoords="offset points", color=color, fontweight="bold")

    axes[-1].set_xlabel(r"C-Tail $R_g$ (Å)", fontsize=14)
    plt.suptitle("C-Tail Radius of Gyration Distribution by Shape Cluster",
                 fontsize=16)
    sns.despine()
    plt.tight_layout()
    _savefig(plots_dir / "ctail_rg_clusters_rows.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved ctail_rg_clusters_rows.png")


def plot_macro_cluster_bar(
    macro_clusters: np.ndarray,
    rg_ctail: np.ndarray,
    ctail_threshold: float,
    cluster_map: dict[str, list[int]],
    plots_dir: Path,
) -> None:
    categories = list(cluster_map.keys())
    is_above = rg_ctail >= ctail_threshold
    df = pd.DataFrame({"MacroCluster": macro_clusters, "AboveThreshold": is_above})
    df = df[df["MacroCluster"] != "Unassigned"]

    grouped = df.groupby(["MacroCluster", "AboveThreshold"]).size().unstack(fill_value=0)
    for val in [True, False]:
        if val not in grouped.columns:
            grouped[val] = 0
    grouped = grouped.reindex(categories).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(categories))
    width = 0.55

    for i, cat in enumerate(categories):
        count_below = grouped.loc[cat, False]
        count_above = grouped.loc[cat, True]
        total = count_below + count_above
        color = CLUSTER_COLOURS[cat]
        ax.bar(x[i], count_below, width, color=color, edgecolor="black", alpha=0.9)
        ax.bar(x[i], count_above, width, bottom=count_below,
               color=color, edgecolor="black", hatch="//", alpha=0.9)
        if count_below > 0:
            pct_below = (count_below / total) * 100
            ax.text(x[i], count_below / 2,
                    f"{int(count_below)}\n({pct_below:.1f}%)",
                    ha="center", va="center", color="white", fontsize=11, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")])
        if count_above > 0:
            pct_above = (count_above / total) * 100
            ax.text(x[i], count_below + count_above / 2,
                    f"{int(count_above)}\n({pct_above:.1f}%)",
                    ha="center", va="center", color="black", fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Structures (Frames)", fontsize=14)
    ax.set_title(
        f"Structural Populations by Macro-Cluster\n"
        f"Split by C-Tail $R_g$ (Threshold = {ctail_threshold} Å)",
        fontsize=15, pad=15,
    )
    above_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//",
                                  label=f"≥ {ctail_threshold} Å")
    below_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                                  label=f"< {ctail_threshold} Å")
    ax.legend(handles=[above_patch, below_patch], title=r"C-Tail $R_g$ Extent",
              loc="upper left", bbox_to_anchor=(1.02, 1))
    sns.despine()
    plt.tight_layout()
    _savefig(plots_dir / "macro_cluster_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved macro_cluster_bar.png")


def plot_macro_cluster_pie(
    macro_clusters: np.ndarray,
    rg_ctail: np.ndarray,
    ctail_threshold: float,
    cluster_map: dict[str, list[int]],
    plots_dir: Path,
) -> None:
    categories = list(cluster_map.keys())
    is_above = rg_ctail >= ctail_threshold
    df = pd.DataFrame({"MacroCluster": macro_clusters, "AboveThreshold": is_above})
    df = df[df["MacroCluster"] != "Unassigned"]
    grouped = df.groupby(["MacroCluster", "AboveThreshold"]).size().unstack(fill_value=0)
    for val in [True, False]:
        if val not in grouped.columns:
            grouped[val] = 0
    grouped = grouped.reindex(categories).fillna(0)

    inner_counts = grouped.sum(axis=1)
    outer_counts = grouped.stack()
    inner_colors = [CLUSTER_COLOURS[cat] for cat in categories]
    outer_colors: list = []
    outer_hatches: list = []
    for cat in categories:
        outer_colors.extend([CLUSTER_COLOURS[cat], CLUSTER_COLOURS[cat]])
        outer_hatches.extend(["", "////"])

    fig, ax = plt.subplots(figsize=(10, 8))
    size = 0.3
    wedges_outer, _ = ax.pie(
        outer_counts, radius=1, colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor="white"), startangle=90,
    )
    for wedge, hatch in zip(wedges_outer, outer_hatches):
        wedge.set_hatch(hatch)
    ax.pie(
        inner_counts, radius=1 - size, labels=categories,
        colors=inner_colors, labeldistance=0.7,
        wedgeprops=dict(width=size, edgecolor="white"),
        textprops={"fontsize": 12, "fontweight": "bold"}, startangle=90,
    )
    for i, p in enumerate(wedges_outer):
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        py = np.sin(np.deg2rad(ang))
        px = np.cos(np.deg2rad(ang))
        count = outer_counts.iloc[i]
        if count > 0:
            total_macro = inner_counts.iloc[i // 2]
            pct = (count / total_macro) * 100
            ax.text(1.2 * px, 1.2 * py, f"{int(count)}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_title(
        f"Structural Populations: Macro-Cluster & C-Tail $R_g$\n"
        f"(Threshold = {ctail_threshold} Å)",
        fontsize=15,
    )
    above_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="////",
                                  label=f"≥ {ctail_threshold} Å")
    below_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                                  label=f"< {ctail_threshold} Å")
    ax.legend(handles=[above_patch, below_patch], title=r"C-Tail $R_g$ Extent",
              loc="upper left", bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
    _savefig(plots_dir / "macro_cluster_pie.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved macro_cluster_pie.png")


def plot_ctail_rg_macro_rows(
    rg_ctail: np.ndarray,
    macro_clusters: np.ndarray,
    cluster_map: dict[str, list[int]],
    plots_dir: Path,
    ctail_threshold: float | None = None,
) -> None:
    categories = list(cluster_map.keys())
    df = pd.DataFrame({"Rg_Ctail (Å)": rg_ctail, "MacroCluster": macro_clusters})
    macro_means = df.groupby("MacroCluster")["Rg_Ctail (Å)"].mean()

    fig, axes = plt.subplots(len(categories), 1,
                              figsize=(8, 3 * len(categories)), sharex=True)
    if len(categories) == 1:
        axes = [axes]

    for i, cat in enumerate(categories):
        ax = axes[i]
        cat_data = df[df["MacroCluster"] == cat]
        mean_val = macro_means[cat]
        color = CLUSTER_COLOURS[cat]
        sns.histplot(data=cat_data, x="Rg_Ctail (Å)", color=color, bins=25,
                     element="step", stat="density", kde=True, alpha=0.4,
                     linewidth=1.5, ax=ax)
        ax.axvline(x=mean_val, color=color, linestyle="--", linewidth=2.5, alpha=0.8)
        if ctail_threshold is not None:
            ax.axvline(x=ctail_threshold, color="black", linestyle=":", linewidth=2, alpha=0.6)
        ax.set_ylabel(f"{cat}\nDensity", fontsize=12)
        ylim = ax.get_ylim()
        ax.annotate(
            f"Mean: {mean_val:.1f} Å",
            xy=(mean_val, ylim[1] * 0.8), xytext=(5, 0),
            textcoords="offset points", color="black", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    axes[-1].set_xlabel(r"C-Tail $R_g$ (Å)", fontsize=14)
    plt.suptitle("C-Tail Radius of Gyration Distribution by Macro-Cluster",
                 fontsize=15)
    sns.despine()
    plt.tight_layout()
    _savefig(plots_dir / "ctail_rg_macro_rows.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved ctail_rg_macro_rows.png")


# ============================================================================
# Main
# ============================================================================


def run_candidate(
    u: mda.Universe,
    output_dir: Path,
    shape_resids: str,
    ctail_resids: str,
    method: str,
    n_clusters: int,
    gmm_select_bic: bool,
    gmm_min_components: int,
    gmm_max_components: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    basin_grid_size: int,
    ctail_threshold: float | None,
    cluster_map: dict[str, list[int]] | None,
    generate_plots: bool = True,
) -> dict[str, object]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    shape_sel_str = format_selection(shape_resids)
    ctail_sel_str = f"resid {parse_resid_range(ctail_resids)[0]}-{parse_resid_range(ctail_resids)[1]}"

    print(f"\nCandidate: shape={shape_resids}, ctail={ctail_resids}")
    print(f"  output_dir: {output_dir}")
    print(f"  shape_sel:  {shape_sel_str}")
    print(f"  ctail_sel:  {ctail_sel_str}")

    x_ratio, y_ratio, I1, I2, I3 = compute_inertia_ratios(u, shape_sel_str)
    ref_ratios = compute_reference_ratios(REFERENCES, shape_sel_str)
    dynamic_ref_ratios = compute_dynamic_reference_ratios(DYNAMIC_REFERENCES, shape_sel_str)

    w_rod = y_ratio - x_ratio
    w_sphere = x_ratio + y_ratio - 1
    w_disk = 2 * (1 - y_ratio)
    features_xy = np.column_stack([x_ratio, y_ratio])

    method_labels: dict[str, np.ndarray] = {}
    method_centers: dict[str, np.ndarray | None] = {}
    selected_gmm_components = n_clusters
    gmm_bic_counts: np.ndarray | None = None
    gmm_bic_scores: np.ndarray | None = None
    gmm_bic_range: tuple[int, int] | None = None

    if gmm_select_bic:
        selected_gmm_components, gmm_bic_scores, gmm_labels, gmm_centers = select_gmm_by_bic(
            features_xy, gmm_min_components, gmm_max_components
        )
        gmm_bic_counts = np.arange(gmm_min_components, gmm_max_components + 1, dtype=int)
        gmm_bic_range = (gmm_min_components, gmm_max_components)
        method_labels["gmm"] = gmm_labels
        method_centers["gmm"] = gmm_centers
        print(
            f"  selected GMM components by BIC: {selected_gmm_components} "
            f"(searched {gmm_min_components}-{gmm_max_components})"
        )

    for method_name in ["kmeans", "gmm", "dbscan", "free_energy_basins"]:
        print(f"  running {method_name}...")
        if method_name == "gmm" and gmm_select_bic:
            labels = method_labels["gmm"]
            centers_xy = method_centers["gmm"]
        else:
            effective_n_clusters = selected_gmm_components if method_name == "gmm" else n_clusters
            labels, centers_xy = run_clustering_method(
                method_name,
                features_xy,
                x_ratio,
                y_ratio,
                effective_n_clusters,
                dbscan_eps,
                dbscan_min_samples,
                basin_grid_size,
                gmm_bic_range=None,
            )
        method_labels[method_name] = labels
        method_centers[method_name] = centers_xy

    cluster_labels = method_labels[method]
    centers_xy = ensure_cluster_centers(features_xy, cluster_labels, method_centers[method])
    rg_ctail = compute_ctail_rg(u, ctail_sel_str)
    resolved_threshold = float(np.median(rg_ctail) if ctail_threshold is None else ctail_threshold)

    coarse_labels, coarse_cluster_lookup = assign_macro_labels_by_reference(
        cluster_labels, centers_xy, ref_ratios
    )
    manual_macro_labels = assign_manual_macro_labels(cluster_labels, cluster_map)

    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    np.save(output_dir / "cluster_centers.npy", centers_xy)
    np.save(output_dir / "shape_axes.npy", np.column_stack([x_ratio, y_ratio]))
    np.save(output_dir / "ctail_rg.npy", rg_ctail)
    np.save(output_dir / "coarse_cluster_labels.npy", coarse_labels)

    metadata = {
        "method": method,
        "n_clusters": int(n_clusters),
        "selected_gmm_components": int(selected_gmm_components),
        "gmm_select_bic": gmm_select_bic,
        "gmm_min_components": int(gmm_min_components),
        "gmm_max_components": int(gmm_max_components),
        "dbscan_eps": float(dbscan_eps),
        "dbscan_min_samples": int(dbscan_min_samples),
        "basin_grid_size": int(basin_grid_size),
        "ctail_threshold": resolved_threshold,
        "shape_resids": shape_resids,
        "ctail_resids": ctail_resids,
        "shape_selection": shape_sel_str,
        "ctail_selection": ctail_sel_str,
        "coarse_mapping_mode": "reference_guided",
        "coarse_cluster_map": invert_macro_mapping(coarse_cluster_lookup),
    }
    with open(output_dir / "cluster_method.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "coarse_cluster_map.json", "w") as f:
        json.dump(invert_macro_mapping(coarse_cluster_lookup), f, indent=2)

    if manual_macro_labels is not None:
        np.save(output_dir / "macro_cluster_labels.npy", manual_macro_labels)
        if cluster_map is not None:
            # Use the provided explicit map
            with open(output_dir / "macro_cluster_map.json", "w") as f:
                json.dump(cluster_map, f, indent=2)
        else:
            # If manual labels were assigned automatically, save that map
            with open(output_dir / "macro_cluster_map.json", "w") as f:
                json.dump(invert_macro_mapping(coarse_cluster_lookup), f, indent=2)
    else:
        # Sensible default: use the reference-guided coarse labels
        np.save(output_dir / "macro_cluster_labels.npy", coarse_labels)
        with open(output_dir / "macro_cluster_map.json", "w") as f:
            json.dump(invert_macro_mapping(coarse_cluster_lookup), f, indent=2)

    if generate_plots:
        plot_shape_space_ctail(
            x_ratio, y_ratio, rg_ctail, plots_dir,
            ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
        )
        plot_free_energy_landscape(
            x_ratio, y_ratio, plots_dir,
            ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
        )
        plot_free_energy_landscape_hexbin(
            x_ratio, y_ratio, plots_dir,
            ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
        )
        plot_shape_indices_histograms(w_rod, w_sphere, w_disk, plots_dir)
        plot_cluster_shape_space(
            x_ratio, y_ratio, cluster_labels, centers_xy, method, plots_dir,
            ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
        )
        plot_cluster_hexbin(
            x_ratio, y_ratio, cluster_labels, method, plots_dir,
            ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
        )
        plot_method_comparison_hexbin(
            x_ratio, y_ratio, method_labels, plots_dir,
            ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
        )
        if gmm_select_bic and gmm_bic_counts is not None and gmm_bic_scores is not None:
            plot_gmm_bic(gmm_bic_counts, gmm_bic_scores, selected_gmm_components, plots_dir)
        plot_ca_traces(u, shape_sel_str, cluster_labels, features_xy, centers_xy, method, plots_dir)
        plot_ca_traces_macro(
            u, shape_sel_str, cluster_labels, coarse_cluster_lookup,
            features_xy, centers_xy, method, plots_dir
        )
        plot_moments_histograms(I1, I2, I3, plots_dir)
        plot_ctail_rg_clusters_hist(rg_ctail, cluster_labels, plots_dir, ctail_threshold=resolved_threshold)
        plot_ctail_rg_clusters_rows(rg_ctail, cluster_labels, plots_dir, ctail_threshold=resolved_threshold)
        plot_macro_hexbin(
            x_ratio, y_ratio, coarse_labels, plots_dir,
            ref_ratios=ref_ratios, dynamic_ref_ratios=dynamic_ref_ratios,
        )
        plot_macro_cluster_bar(
            coarse_labels, rg_ctail, resolved_threshold,
            invert_macro_mapping(coarse_cluster_lookup), plots_dir,
        )
        plot_macro_cluster_pie(
            coarse_labels, rg_ctail, resolved_threshold,
            invert_macro_mapping(coarse_cluster_lookup), plots_dir,
        )
        plot_ctail_rg_macro_rows(
            rg_ctail, coarse_labels, invert_macro_mapping(coarse_cluster_lookup), plots_dir,
            ctail_threshold=resolved_threshold,
        )

    return {
        "output_dir": str(output_dir),
        "shape_resids": shape_resids,
        "ctail_resids": ctail_resids,
        "coarse_cluster_map": invert_macro_mapping(coarse_cluster_lookup),
        "selected_gmm_components": int(selected_gmm_components),
        "ctail_threshold": resolved_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moments-of-inertia based clustering of the aSyn trajectory."
    )
    parser.add_argument("--top-pdb", default=None, help="Topology PDB file")
    parser.add_argument("--traj-xtc", default=None, help="Trajectory XTC file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: _cluster_inertia relative to script dir)")
    parser.add_argument("--n-clusters", type=int, default=7,
                        help="Number of clusters/components/basins for kmeans, gmm, and free-energy basins")
    parser.add_argument(
        "--method",
        default="gmm",
        choices=["kmeans", "gmm", "dbscan", "free_energy_basins"],
        help="Primary clustering method used for saved labels and macro-cluster mapping",
    )
    parser.add_argument(
        "--gmm-select-bic",
        action="store_true",
        help="Select the GMM component count by BIC instead of using --n-clusters",
    )
    parser.add_argument(
        "--gmm-min-components", type=int, default=2,
        help="Minimum GMM components to consider for BIC selection (default: 2)"
    )
    parser.add_argument(
        "--gmm-max-components", type=int, default=20,
        help="Maximum GMM components to consider for BIC selection (default: 8)"
    )
    parser.add_argument(
        "--dbscan-eps", type=float, default=0.0025,
        help="DBSCAN eps in shape-space coordinates (default: 0.035)"
    )
    parser.add_argument(
        "--dbscan-min-samples", type=int, default=50,
        help="DBSCAN min_samples (default: 50)"
    )
    parser.add_argument(
        "--basin-grid-size", type=int, default=25,
        help="Grid size for KDE/free-energy basin assignment (default: 15)"
    )
    parser.add_argument("--shape-resids", default="1:114",
                        help="Residue range for shape region 'start:end' PDB numbering (default: 1:96)")
    parser.add_argument(
        "--shape-resids-grid",
        default=None,
        help=(
            "Comma-separated candidate shape residue ranges for a benchmark sweep, "
            "e.g. '1:61,1:95,1:105,1:114,1:135'. If omitted, a single candidate is run."
        ),
    )
    parser.add_argument(
        "--use-default-shape-grid",
        action="store_true",
        help="Run the built-in benchmark grid: 1:61,1:95,1:105,1:114,1:135.",
    )
    parser.add_argument("--ctail-resids", default="115:140",
                        help="Residue range for C-tail Rg 'start:end' PDB numbering (default: 115:140)")
    parser.add_argument("--ctail-threshold", type=float, default=None,
                        help="C-tail Rg threshold in Å. If omitted, the median of the data is used.")
    parser.add_argument(
        "--cluster-map", default='{"Rod":[2],"Wavy":[1,4,7],"Compact":[0,3,5,6]}',
        help=(
            'JSON mapping macro names to k-cluster IDs, e.g. '
            '\'{"Rod":[2],"Wavy":[1,3],"Compact":[0,4,5]}\'. '
            "If omitted, macro-cluster outputs and bar/pie charts are skipped."
        ),
    )
    parser.add_argument("--absolute-paths", action="store_true",
                        help="Interpret --top-pdb, --traj-xtc, --output-dir as absolute paths")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    def resolve(path_str: str | None, default: Path) -> Path:
        if path_str is None:
            return default
        p = Path(path_str)
        return p if args.absolute_paths else (script_dir / p).resolve()

    top_pdb = resolve(args.top_pdb, script_dir / "_aSyn/tris_MD/md_mol_center_coil.pdb")
    traj_xtc = resolve(args.traj_xtc, script_dir / "_aSyn/tris_MD/tris_all_combined.xtc")
    output_dir = resolve(args.output_dir, script_dir / "_cluster_inertia")
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_map: dict[str, list[int]] | None = None
    if args.cluster_map:
        cluster_map = json.loads(args.cluster_map)

    if args.use_default_shape_grid and args.shape_resids_grid is None:
        args.shape_resids_grid = ",".join(DEFAULT_SHAPE_RESID_CANDIDATES)
    shape_resid_candidates = parse_shape_resid_candidates(args.shape_resids, args.shape_resids_grid)

    print(f"top_pdb:          {top_pdb}")
    print(f"traj_xtc:         {traj_xtc}")
    print(f"output_dir:       {output_dir}")
    print(f"shape_resids:     {shape_resid_candidates}")
    print(f"ctail_resids:     {args.ctail_resids}")
    print(f"method:           {args.method}")
    print(f"n_clusters:       {args.n_clusters}")
    print(f"gmm_select_bic:   {args.gmm_select_bic}")
    print(f"gmm_components:   {args.gmm_min_components}-{args.gmm_max_components}")
    print(f"dbscan_eps:       {args.dbscan_eps}")
    print(f"dbscan_min_samples: {args.dbscan_min_samples}")
    print(f"basin_grid_size:  {args.basin_grid_size}")
    print(f"ctail_threshold:  {args.ctail_threshold} Å")
    print(f"cluster_map:      {cluster_map}")
    print("-" * 60)

    print("Loading trajectory...")
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    n_frames = len(u.trajectory)
    print(f"  {n_frames} frames, {u.atoms.n_atoms} atoms")
    summaries = []
    sweep_mode = len(shape_resid_candidates) > 1
    for shape_resids in shape_resid_candidates:
        candidate_output = output_dir / f"candidate_shape_{slugify_range(shape_resids)}" if sweep_mode else output_dir
        candidate_output.mkdir(parents=True, exist_ok=True)
        summaries.append(
            run_candidate(
                u=u,
                output_dir=candidate_output,
                shape_resids=shape_resids,
                ctail_resids=args.ctail_resids,
                method=args.method,
                n_clusters=args.n_clusters,
                gmm_select_bic=args.gmm_select_bic,
                gmm_min_components=args.gmm_min_components,
                gmm_max_components=args.gmm_max_components,
                dbscan_eps=args.dbscan_eps,
                dbscan_min_samples=args.dbscan_min_samples,
                basin_grid_size=args.basin_grid_size,
                ctail_threshold=args.ctail_threshold,
                cluster_map=cluster_map,
                generate_plots=True,
            )
        )

    if sweep_mode:
        with open(output_dir / "candidate_sweep_summary.json", "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved sweep summary to {output_dir / 'candidate_sweep_summary.json'}")

    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
