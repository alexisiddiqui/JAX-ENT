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
        [--absolute-paths] \
        [--shape-axes-npy <path>] \
        [--macro-cluster-labels-npy <path>] \
        [--ctail-rg-npy <path>] \
        [--ctail-threshold <float>]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patheffects
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from scipy.spatial.distance import cdist

from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.plotting import setup_publication_style

# ============================================================================
# Constants
# ============================================================================

NAC_RANGE = range(61, 96)  # residues 61-95
P2_RANGE = range(45, 58)   # residues 45-57
ALIGN_RANGE = range(1, 45)
T1_RANGE = range(1, 61)
T2_RANGE = range(96, 141)
CTAIL_RANGE = range(115, 136)

NHEAD_RG_RANGE = range(1, 61)
NAC_RG_RANGE = range(61, 96)
CTAIL_RG_RANGE = range(96, 141)

FEATURE_NAMES = {
    "nc_distance": "N–C Distance (Å)",
    "nac_prot": "NAC Mean Log P$_f$",
    "p2_prot": "p2 Mean Log P$_f$",
    "termini_contacts": "N–C Contacts",
    "ctail_prot": "C-tail Mean Log P$_f$",
    "nhead_rg": "N-head RadGyr (Å)",
    "nac_rg": "NAC RadGyr (Å)",
    "ctail_rg": "C-tail RadGyr (Å)"
}

# Macro-cluster display colours (matching inertia_moments_clustering.py)
CLUSTER_COLOURS = {"Rod": "grey", "Wavy": "blue", "Compact": "orange"}
MACRO_NAMES = ["Rod", "Wavy", "Compact"]
CTAIL_THRESHOLD_DEFAULT = None  # None implies use median of data

# Reference PDB Definitions (Consistency with analyse_aSyn_ensemble.py)
# Define relative to the jaxent root
_REFS_RAW = {
    "Rod (AF)": "jaxent/examples/4_aSyn/data/_aSyn/AF-P37840-F1-model_v6.pdb",
    "Hairpin": "jaxent/examples/4_aSyn/data/_aSyn/2kkw_1.pdb",
    "Compact": "jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_max_plddt_12691.pdb",
}

def resolve_reference_path(rel_path_str: str) -> Path:
    """Try to find the file relative to CWD, then relative to repo root."""
    p = Path(rel_path_str)
    if p.exists():
        return p
    
    # Try finding the 'jaxent' directory by climbing up from the script location
    script_dir = Path(__file__).parent.resolve()
    # Script is in jaxent/examples/4_aSyn/analysis/
    # Root is up 4 levels
    root_guess = script_dir.parents[3]
    p_root = root_guess / rel_path_str
    if p_root.exists():
        return p_root
        
    return p # Fallback to original, which might still fail but we've tried

REFERENCES = {name: resolve_reference_path(path) for name, path in _REFS_RAW.items()}

REF_MARKERS = {"Rod (AF)": "*", "Hairpin": "D", "Compact": "^"}
REF_COLORS = {"Rod (AF)": "#d62728", "Hairpin": "#1f77b4", "Compact": "#2ca02c"}

# ============================================================================
# Data loading functions
# ============================================================================


def load_topology_map(topo_path: Path) -> dict[int, int]:
    """Load topology JSON and build residue ID to log_Pf array index map.

    Returns:x
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


def compute_termini_contacts(
    top_pdb: Path, traj_xtc: Path, range1: range, range2: range, cutoff: float = 8.0
) -> np.ndarray:
    """Compute number of CA-CA contacts between two residue ranges.

    Returns:
        np.ndarray: shape (n_frames,) — contact counts per frame
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))

    sel1 = u.select_atoms(f"name CA and resid {' '.join(map(str, range1))}")
    sel2 = u.select_atoms(f"name CA and resid {' '.join(map(str, range2))}")

    if sel1.n_atoms == 0 or sel2.n_atoms == 0:
        raise ValueError(
            f"Selection empty: range1 has {sel1.n_atoms}, range2 has {sel2.n_atoms}"
        )

    contact_counts = np.empty(len(u.trajectory), dtype=np.int32)
    for i, ts in enumerate(u.trajectory):
        dists = cdist(sel1.positions, sel2.positions)
        contact_counts[i] = np.sum(dists < cutoff)

    return contact_counts


def compute_radgyr(top_pdb: Path, traj_xtc: Path, resid_range: range) -> np.ndarray:
    """Compute Radius of Gyration for a residue range from CA atoms.

    Returns:
        np.ndarray: shape (n_frames,) — RadGyr in Angstroms
    """
    u = mda.Universe(str(top_pdb), str(traj_xtc))
    sel = u.select_atoms(f"name CA and resid {' '.join(map(str, resid_range))}")

    if sel.n_atoms == 0:
        raise ValueError(f"Selection empty for range {resid_range}")

    rg_values = np.empty(len(u.trajectory), dtype=np.float32)
    for i, ts in enumerate(u.trajectory):
        rg_values[i] = sel.radius_of_gyration()

    return rg_values


# ============================================================================
# Plotting helper
# ============================================================================


def remove_top_right_spines(ax):
    """Remove top and right spines from axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def compute_reference_inertia_ratios(ref_dict: dict[str, Path], sel_str: str) -> dict[str, tuple[float, float]]:
    """Compute I1/I3 and I2/I3 for reference PDB structures."""
    ratios = {}
    for name, pdb_path in ref_dict.items():
        if not pdb_path.exists():
            print(f"WARNING: Reference {name} PDB not found at {pdb_path}")
            continue
        try:
            ref_u = mda.Universe(str(pdb_path))
            sel = ref_u.select_atoms(sel_str)
            if sel.n_atoms == 0:
                print(f"WARNING: No atoms for reference {name} with selection '{sel_str}'")
                continue
            inertia_tensor = sel.moment_of_inertia()
            eigenvalues = np.linalg.eigvalsh(inertia_tensor)
            ratios[name] = (eigenvalues[0] / eigenvalues[2], eigenvalues[1] / eigenvalues[2])
        except Exception as e:
            print(f"ERROR computing ratios for reference {name}: {e}")
    return ratios


def _overlay_references(
    ax: matplotlib.axes.Axes,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    is_legend: bool = False,
) -> None:
    """Overlay reference markers on ternary shape space plot."""
    if ref_ratios is None:
        return

    for name, (xr, yr) in ref_ratios.items():
        ax.scatter(
            xr, yr,
            marker=REF_MARKERS.get(name, "o"),
            s=150,
            color=REF_COLORS.get(name, "red"),
            edgecolors="black",
            linewidths=1.2,
            zorder=15,
            label=name if is_legend else None,
        )


# ============================================================================
# Main plotting function
# ============================================================================


def weighted_stats(data: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Compute mean and standard deviation, optionally weighted."""
    if weights is None:
        return np.mean(data), np.std(data)
    mean = np.average(data, weights=weights)
    var = np.average((data - mean)**2, weights=weights)
    return mean, np.sqrt(var)


def aggregate_weight_rows(
    rows: np.ndarray,
    aggregation: str = "mean",
) -> np.ndarray:
    """Aggregate selected weight rows into one per-frame vector."""
    if rows.ndim != 2:
        raise ValueError(f"Expected 2D weight array, got shape {rows.shape}")

    if aggregation == "mean":
        per_frame = np.nanmean(rows, axis=0)
    elif aggregation == "median":
        per_frame = np.nanmedian(rows, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation mode: {aggregation}")

    per_frame = np.nan_to_num(per_frame, nan=0.0)
    total = np.sum(per_frame)
    if total <= 0:
        raise ValueError("Aggregated weights sum to zero.")
    return per_frame / total


def plot_feature_distributions_per_metric(
    metric_dir: Path,
    nc_dist: np.ndarray,
    nac_prot: np.ndarray,
    p2_prot: np.ndarray,
    ctail_prot: np.ndarray,
    termini_contacts: np.ndarray,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
    weight_aggregation: str,
):
    """Create 3×N_conditions figure showing weighted feature distributions.

    Args:
        metric_dir: directory containing {condition}_*_selected.npz files
        nc_dist: shape (n_frames,)
        nac_prot: shape (n_frames,)
        p2_prot: shape (n_frames,)
        ctail_prot: shape (n_frames,)
        termini_contacts: shape (n_frames,)
        cluster_labels: shape (n_frames,) or None (if input matches clusters)
        n_per_cluster: counts per cluster or None
        cfg: ExperimentConfig with ensembles and style colours
        output_dir: where to save the figure
    """
    metric_name = metric_dir.name  # e.g., "recovery_percent_max"
    n_conds = len(cfg.ensembles)

    condition_weights = load_condition_weights(
        metric_dir, cfg, cluster_labels, n_per_cluster, len(nc_dist),
        aggregation=weight_aggregation,
    )

    # Build figure: N_conditions rows × 5 feature columns
    figsize = (20, 3 * n_conds)
    fig, axes = plt.subplots(n_conds, 5, figsize=figsize, constrained_layout=True)
    if n_conds == 1:
        axes = axes.reshape(1, 5)

    feature_data = [nc_dist, nac_prot, p2_prot, ctail_prot, termini_contacts]
    feature_keys = ["nc_distance", "nac_prot", "p2_prot", "ctail_prot", "termini_contacts"]

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

            # Calculate stats
            m_total, s_total = weighted_stats(feature_array)
            m_weighted, s_weighted = weighted_stats(feature_array, weights=weights)

            # Determine common bins for both histograms
            # Use the full range of the feature data across all frames
            bins = np.linspace(np.min(feature_array), np.max(feature_array), 25)

            # Plot unweighted histogram behind (semi-transparent grey)
            ax.hist(
                feature_array,
                bins=bins,
                density=True,
                alpha=0.3,
                color="grey",
                edgecolor="none",
                label="Total Ensemble" if (row_idx == 0 and col_idx == 0) else None,
            )

            # Plot weighted histogram on top
            ax.hist(
                feature_array,
                bins=bins,
                weights=weights,
                density=True,
                alpha=0.8,
                color=color,
                edgecolor="none",
                label=f"{condition} (Weighted)" if (row_idx == 0 and col_idx == 0) else None,
            )

            # Highlight threshold if this is ctail_rg
            if feature_key == "ctail_rg" and ctail_threshold is not None:
                ax.axvline(x=ctail_threshold, color="black", linestyle=":", lw=1.2, alpha=0.7)

            # Annotate results
            # Position at top right
            stat_text_total = f"Total: {m_total:.1f} ± {s_total:.1f}"
            stat_text_weighted = f"Weighted: {m_weighted:.1f} ± {s_weighted:.1f}"
            
            ax.text(0.95, 0.95, stat_text_total, transform=ax.transAxes, 
                    va="top", ha="right", fontsize=7, color="grey")
            ax.text(0.95, 0.85, stat_text_weighted, transform=ax.transAxes, 
                    va="top", ha="right", fontsize=7, color="black", fontweight="bold")

            # Labels: condition name on leftmost column, "Density" elsewhere
            if col_idx == 0:
                ax.set_ylabel(f"{condition}\nDensity", fontweight="bold")
            else:
                ax.set_ylabel("Density")

            # Add legend to top-left panel only
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, framealpha=0.5)

            # Feature x-label on bottom row only
            if row_idx == n_conds - 1:
                ax.set_xlabel(FEATURE_NAMES[feature_key])

            # Feature name as column title on top row only
            if row_idx == 0:
                ax.set_title(FEATURE_NAMES[feature_key], fontsize=11, fontweight="bold", pad=10)

            remove_top_right_spines(ax)

    fig.suptitle(
        f"Feature Distributions — {metric_name}",
        fontsize=13,
        fontweight="bold",
    )

    # Save
    out_path = output_dir / f"feature_distributions_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_radgyr_distributions_per_metric(
    metric_dir: Path,
    nhead_rg: np.ndarray,
    nac_rg: np.ndarray,
    ctail_rg: np.ndarray,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
    ctail_threshold: float | None = None,
    weight_aggregation: str = "mean",
):
    """Create 3×N_conditions figure showing weighted RadGyr feature distributions.

    Args:
        metric_dir: directory containing {condition}_*_selected.npz files
        nhead_rg: shape (n_frames,)
        nac_rg: shape (n_frames,)
        ctail_rg: shape (n_frames,)
        cluster_labels: shape (n_frames,) or None (if input matches clusters)
        n_per_cluster: counts per cluster or None
        cfg: ExperimentConfig with ensembles and style colours
        output_dir: where to save the figure
    """
    metric_name = metric_dir.name  # e.g., "recovery_percent_max"
    n_conds = len(cfg.ensembles)

    condition_weights = load_condition_weights(
        metric_dir, cfg, cluster_labels, n_per_cluster, len(nhead_rg),
        aggregation=weight_aggregation,
    )

    # Build figure: N_conditions rows × 3 feature columns
    figsize = (12, 3 * n_conds)
    fig, axes = plt.subplots(n_conds, 3, figsize=figsize, constrained_layout=True)
    if n_conds == 1:
        axes = axes.reshape(1, 3)

    feature_data = [nhead_rg, nac_rg, ctail_rg]
    feature_keys = ["nhead_rg", "nac_rg", "ctail_rg"]

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

            # Calculate stats
            m_total, s_total = weighted_stats(feature_array)
            m_weighted, s_weighted = weighted_stats(feature_array, weights=weights)

            # Determine common bins for both histograms
            bins = np.linspace(np.min(feature_array), np.max(feature_array), 25)

            # Plot unweighted histogram behind (semi-transparent grey)
            ax.hist(
                feature_array,
                bins=bins,
                density=True,
                alpha=0.3,
                color="grey",
                edgecolor="none",
                label="Total Ensemble" if (row_idx == 0 and col_idx == 0) else None,
            )

            # Plot weighted histogram on top
            ax.hist(
                feature_array,
                bins=bins,
                weights=weights,
                density=True,
                alpha=0.8,
                color=color,
                edgecolor="none",
                label=f"{condition} (Weighted)" if (row_idx == 0 and col_idx == 0) else None,
            )

            # Highlight threshold if this is ctail_rg
            if feature_key == "ctail_rg" and ctail_threshold is not None:
                ax.axvline(x=ctail_threshold, color="black", linestyle=":", lw=1.2, alpha=0.7)

            # Annotate results
            # Position at top right
            stat_text_total = f"Total: {m_total:.1f} ± {s_total:.1f}"
            stat_text_weighted = f"Weighted: {m_weighted:.1f} ± {s_weighted:.1f}"
            
            ax.text(0.95, 0.95, stat_text_total, transform=ax.transAxes, 
                    va="top", ha="right", fontsize=7, color="grey")
            ax.text(0.95, 0.85, stat_text_weighted, transform=ax.transAxes, 
                    va="top", ha="right", fontsize=7, color="black", fontweight="bold")

            # Labels: condition name on leftmost column, "Density" elsewhere
            if col_idx == 0:
                ax.set_ylabel(f"{condition}\nDensity", fontweight="bold")
            else:
                ax.set_ylabel("Density")

            # Add legend to top-left panel only
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, framealpha=0.5)

            # Feature x-label on bottom row only
            if row_idx == n_conds - 1:
                ax.set_xlabel(FEATURE_NAMES[feature_key])

            # Feature name as column title on top row only
            if row_idx == 0:
                ax.set_title(FEATURE_NAMES[feature_key], fontsize=11, fontweight="bold", pad=10)

            remove_top_right_spines(ax)

    fig.suptitle(
        f"RadGyr Feature Distributions — {metric_name}",
        fontsize=13,
        fontweight="bold",
    )

    # Save
    out_path = output_dir / f"radgyr_distributions_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================================
# Shared weight-loading helper
# ============================================================================


def load_condition_weights(
    metric_dir: Path,
    cfg: ExperimentConfig,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    n_frames: int,
    aggregation: str = "mean",
) -> dict[str, np.ndarray | None]:
    """Load per-frame weights for each condition from extracted NPZ files."""
    condition_weights: dict[str, np.ndarray | None] = {}
    for condition in cfg.ensembles:
        npz_files = list(metric_dir.glob(f"{condition}_*_selected.npz"))
        if not npz_files:
            condition_weights[condition] = None
            continue

        all_weights = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            all_weights.append(data["frame_weights"])

        stacked = np.vstack(all_weights)
        per_frame = aggregate_weight_rows(stacked, aggregation=aggregation)

        if len(per_frame) != n_frames:
            print(f"WARNING: Weight count ({len(per_frame)}) != frame count ({n_frames}) for {condition}")
            condition_weights[condition] = None
            continue

        condition_weights[condition] = per_frame

    return condition_weights


def load_replicate_weights(
    metric_dir: Path,
    cfg: ExperimentConfig,
    n_frames: int,
) -> dict[str, list[np.ndarray] | None]:
    """Load per-replicate per-frame weights for each condition.

    Returns a dict mapping condition → list of n_replicates arrays each of shape (n_frames,).
    """
    replicate_weights: dict[str, list[np.ndarray] | None] = {}
    for condition in cfg.ensembles:
        npz_files = list(metric_dir.glob(f"{condition}_*_selected.npz"))
        if not npz_files:
            replicate_weights[condition] = None
            continue

        rows = []
        for npz_file in npz_files:
            data = np.load(npz_file)
            fw = data["frame_weights"]  # shape (n_replicates, n_frames)
            for row in fw:
                if len(row) == n_frames:
                    rows.append(row)

        if not rows:
            replicate_weights[condition] = None
            continue

        replicate_weights[condition] = rows

    return replicate_weights


def compute_macro_fraction_rows(
    weights: np.ndarray,
    macro_labels: np.ndarray,
    ctail_rg: np.ndarray,
    ctail_threshold: float,
) -> dict[str, float]:
    """Return total, compact, and extended fractions for each macro-state."""
    is_extended = ctail_rg >= ctail_threshold
    total_w = np.sum(weights)
    if total_w <= 0:
        raise ValueError("Weights sum to zero when computing macro fractions.")

    fractions: dict[str, float] = {}
    for macro in MACRO_NAMES:
        mask_total = macro_labels == macro
        mask_compact = mask_total & ~is_extended
        mask_extended = mask_total & is_extended
        fractions[f"{macro}_total"] = float(np.sum(weights[mask_total]) / total_w)
        fractions[f"{macro}_compact"] = float(np.sum(weights[mask_compact]) / total_w)
        fractions[f"{macro}_extended"] = float(np.sum(weights[mask_extended]) / total_w)
    return fractions


# ============================================================================
# New shape / meta-cluster plot functions
# ============================================================================


def plot_shape_order_per_metric(
    metric_dir: Path,
    shape_axes: np.ndarray,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
    weight_aggregation: str,
) -> None:
    """N_conditions rows × 3 cols: weighted vs unweighted histograms of barycentric shape indices."""
    metric_name = metric_dir.name
    n_conds = len(cfg.ensembles)
    n_frames = len(shape_axes)
    condition_weights = load_condition_weights(
        metric_dir, cfg, cluster_labels, n_per_cluster, n_frames,
        aggregation=weight_aggregation,
    )

    # Compute barycentric shape indices from x_ratio, y_ratio
    x = shape_axes[:, 0]  # I1/I3
    y = shape_axes[:, 1]  # I2/I3
    w_rod = y - x
    w_sphere = x + y - 1
    w_disk = 2 * (1 - y)

    axes_data = [w_rod, w_sphere, w_disk]
    axes_labels = ["Rod Axis", "Sphere Axis", "Disk Axis"]

    fig, axes = plt.subplots(n_conds, 3, figsize=(14, 3 * n_conds), constrained_layout=True)
    if n_conds == 1:
        axes = axes.reshape(1, 3)

    for row_idx, condition in enumerate(cfg.ensembles):
        for col_idx, (arr, xlabel) in enumerate(zip(axes_data, axes_labels)):
            ax = axes[row_idx, col_idx]
            weights = condition_weights[condition]

            if weights is None:
                ax.text(0.5, 0.5, f"No data for {condition}", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="red")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            color = cfg.style.ensemble_colors.get(condition, "#808080")
            bins = np.linspace(np.min(arr), np.max(arr), 30)

            ax.hist(arr, bins=bins, density=True, alpha=0.3, color="grey",
                    edgecolor="none")
            ax.hist(arr, bins=bins, weights=weights, density=True, alpha=0.8,
                    color=color, edgecolor="none")

            m_uw, s_uw = np.mean(arr), np.std(arr)
            m_w = np.average(arr, weights=weights)
            s_w = np.sqrt(np.average((arr - m_w) ** 2, weights=weights))
            ax.text(0.95, 0.95, f"Total: {m_uw:.2f} ± {s_uw:.2f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=7, color="grey")
            ax.text(0.95, 0.85, f"Weighted: {m_w:.2f} ± {s_w:.2f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=7,
                    color="black", fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(f"{condition}\nDensity", fontweight="bold")
            else:
                ax.set_ylabel("Density")

            if row_idx == n_conds - 1:
                ax.set_xlabel(xlabel)
            if row_idx == 0:
                ax.set_title(xlabel, fontsize=11, fontweight="bold", pad=10)

            remove_top_right_spines(ax)

    fig.suptitle(f"Shape Order (Barycentric Coordinates) — {metric_name}",
                 fontsize=13, fontweight="bold")
    out_path = output_dir / f"shape_order_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_free_energy_landscape_per_metric(
    metric_dir: Path,
    shape_axes: np.ndarray,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
    gridsize: int = 20,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    weight_aggregation: str = "mean",
) -> None:
    """1 row × N_conditions cols: weighted free energy hexbin landscapes in shape space."""
    metric_name = metric_dir.name
    n_conds = len(cfg.ensembles)
    n_frames = len(shape_axes)
    condition_weights = load_condition_weights(
        metric_dir, cfg, cluster_labels, n_per_cluster, n_frames,
        aggregation=weight_aggregation,
    )

    x = shape_axes[:, 0]  # I1/I3
    y = shape_axes[:, 1]  # I2/I3

    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 5), constrained_layout=True)
    if n_conds == 1:
        axes = [axes]

    last_hb = None
    for col_idx, condition in enumerate(cfg.ensembles):
        ax = axes[col_idx]
        weights = condition_weights[condition]

        if weights is None:
            ax.text(0.5, 0.5, f"No data for {condition}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(condition, fontsize=11, fontweight="bold")
            continue

        # Hexbin with weighted sums per bin
        hb = ax.hexbin(x, y, C=weights, reduce_C_function=np.sum, gridsize=gridsize,
                       mincnt=1, cmap="YlGnBu_r", edgecolors="none")
        last_hb = hb

        # Compute free energy from weighted counts
        weighted_counts = hb.get_array()
        with np.errstate(divide="ignore", invalid="ignore"):
            F = -np.log(weighted_counts)
        F -= np.nanmin(F)

        # Update hexbin with quantised free energy (0-8 levels)
        norm = matplotlib.colors.BoundaryNorm(list(range(9)), ncolors=256, extend="max")
        hb.set_array(F)
        hb.set_norm(norm)

        # Shape boundary triangle
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [1, 1, 0.5, 1]
        ax.plot(triangle_x, triangle_y, "k--", lw=1.5, zorder=5)

        # Vertex labels with white stroke
        label_kw = dict(fontsize=10, zorder=6,
                        path_effects=[matplotlib.patheffects.Stroke(linewidth=3, foreground='white'),
                                       matplotlib.patheffects.Normal()])
        ax.text(0, 1.04, r"Rod", ha="center", **label_kw)
        ax.text(1, 1.04, r"Sphere", ha="center", **label_kw)
        ax.text(0.5, 0.46, r"Disk", ha="center", **label_kw)

        # Oblate/prolate guide line
        ax.plot([0.5, 1], [0.5, 1], color="gray", lw=1, ls=":", alpha=0.7, zorder=4)

        # Axes limits and labels
        ax.set_xlim(-0.08, 1.12)
        ax.set_ylim(0.42, 1.10)
        ax.set_xlabel(r"$I_1/I_3$", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(r"$I_2/I_3$", fontsize=11)

        ax.set_title(condition, fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, color="lightgray", lw=0.5, zorder=0)

        _overlay_references(ax, ref_ratios, is_legend=(col_idx == 0))

    # Shared colorbar from last valid hexbin
    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes, pad=0.02, shrink=0.8)
        cbar.set_label(r"$\Delta F / k_BT$", fontsize=12)
        cbar.set_ticks(range(9))

    fig.suptitle(f"Free Energy Landscape — {metric_name}",
                 fontsize=13, fontweight="bold")
    out_path = output_dir / f"free_energy_landscape_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_free_energy_difference_per_metric(
    metric_dir: Path,
    shape_axes: np.ndarray,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
    gridsize: int = 20,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
    weight_aggregation: str = "mean",
) -> None:
    """1 row × N_conditions cols: free energy difference (weighted − unweighted) in shape space."""
    metric_name = metric_dir.name
    n_conds = len(cfg.ensembles)
    n_frames = len(shape_axes)
    condition_weights = load_condition_weights(
        metric_dir, cfg, cluster_labels, n_per_cluster, n_frames,
        aggregation=weight_aggregation,
    )

    x = shape_axes[:, 0]  # I1/I3
    y = shape_axes[:, 1]  # I2/I3

    # Compute reference (unweighted) free energy on a temporary figure
    fig_tmp, ax_tmp = plt.subplots()
    hb_ref = ax_tmp.hexbin(x, y, gridsize=gridsize, mincnt=1, cmap="YlGnBu_r", edgecolors="none")
    counts_ref = hb_ref.get_array().copy()
    plt.close(fig_tmp)

    with np.errstate(divide="ignore", invalid="ignore"):
        F_ref = -np.log(counts_ref)
    F_ref -= np.nanmin(F_ref)

    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 5), constrained_layout=True)
    if n_conds == 1:
        axes = [axes]

    last_hb = None
    for col_idx, condition in enumerate(cfg.ensembles):
        ax = axes[col_idx]
        weights = condition_weights[condition]

        if weights is None:
            ax.text(0.5, 0.5, f"No data for {condition}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(condition, fontsize=11, fontweight="bold")
            continue

        # Hexbin with weighted sums per bin
        hb = ax.hexbin(x, y, C=weights, reduce_C_function=np.sum, gridsize=gridsize,
                       mincnt=1, cmap="PuOr", edgecolors="none")
        last_hb = hb

        # Compute weighted free energy
        weighted_counts = hb.get_array().copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            F_w = -np.log(weighted_counts)
        F_w -= np.nanmin(F_w)

        # Compute difference
        diff = F_w - F_ref

        # Symmetric color scaling
        vlim = np.nanmax(np.abs(diff))
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
        hb.set_array(diff)
        hb.set_norm(norm)

        # Shape boundary triangle
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [1, 1, 0.5, 1]
        ax.plot(triangle_x, triangle_y, "k--", lw=1.5, zorder=5)

        # Vertex labels with white stroke
        label_kw = dict(fontsize=10, zorder=6,
                        path_effects=[matplotlib.patheffects.Stroke(linewidth=3, foreground='white'),
                                       matplotlib.patheffects.Normal()])
        ax.text(0, 1.04, r"Rod", ha="center", **label_kw)
        ax.text(1, 1.04, r"Sphere", ha="center", **label_kw)
        ax.text(0.5, 0.46, r"Disk", ha="center", **label_kw)

        # Oblate/prolate guide line
        ax.plot([0.5, 1], [0.5, 1], color="gray", lw=1, ls=":", alpha=0.7, zorder=4)

        # Axes limits and labels
        ax.set_xlim(-0.08, 1.12)
        ax.set_ylim(0.42, 1.10)
        ax.set_xlabel(r"$I_1/I_3$", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(r"$I_2/I_3$", fontsize=11)

        ax.set_title(condition, fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, color="lightgray", lw=0.5, zorder=0)

        _overlay_references(ax, ref_ratios, is_legend=(col_idx == 0))

    # Shared colorbar from last valid hexbin
    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes, pad=0.02, shrink=0.8)
        cbar.set_label(r"$\Delta\Delta F / k_BT$ (weighted − unweighted)", fontsize=12)

    fig.suptitle(f"Free Energy Difference (Weighted − Unweighted) — {metric_name}",
                 fontsize=13, fontweight="bold")
    out_path = output_dir / f"free_energy_difference_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_free_energy_uncertainty_per_metric(
    metric_dir: Path,
    shape_axes: np.ndarray,
    cfg: ExperimentConfig,
    output_dir: Path,
    gridsize: int = 20,
    ref_ratios: dict[str, tuple[float, float]] | None = None,
) -> None:
    """1 row × N_conditions cols: std dev of free energy across weight replicates."""
    metric_name = metric_dir.name
    n_conds = len(cfg.ensembles)
    n_frames = len(shape_axes)
    replicate_weights = load_replicate_weights(metric_dir, cfg, n_frames)

    x = shape_axes[:, 0]  # I1/I3
    y = shape_axes[:, 1]  # I2/I3

    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 5), constrained_layout=True)
    if n_conds == 1:
        axes = [axes]

    last_hb = None
    for col_idx, condition in enumerate(cfg.ensembles):
        ax = axes[col_idx]
        reps = replicate_weights[condition]

        if reps is None or len(reps) < 2:
            ax.text(0.5, 0.5, f"No data for {condition}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(condition, fontsize=11, fontweight="bold")
            continue

        # Compute free energy for each replicate using a temporary figure
        F_reps = []
        for rep_weights in reps:
            fig_tmp, ax_tmp = plt.subplots()
            hb_tmp = ax_tmp.hexbin(x, y, C=rep_weights, reduce_C_function=np.sum,
                                   gridsize=gridsize, mincnt=1, edgecolors="none")
            wc = hb_tmp.get_array().copy()
            plt.close(fig_tmp)
            with np.errstate(divide="ignore", invalid="ignore"):
                F = -np.log(wc)
            F -= np.nanmin(F)
            F_reps.append(F)

        # Std dev across replicates (bin-by-bin)
        F_stack = np.stack(F_reps, axis=0)  # (n_reps, n_bins)
        F_std = np.std(F_stack, axis=0)

        # Draw on the panel using the first replicate's hexbin structure
        hb = ax.hexbin(x, y, C=reps[0], reduce_C_function=np.sum, gridsize=gridsize,
                       mincnt=1, cmap="Reds", edgecolors="none")
        last_hb = hb
        hb.set_array(F_std)
        hb.set_norm(matplotlib.colors.Normalize(vmin=0, vmax=np.nanmax(F_std)))

        # Shape boundary triangle
        triangle_x = [0, 1, 0.5, 0]
        triangle_y = [1, 1, 0.5, 1]
        ax.plot(triangle_x, triangle_y, "k--", lw=1.5, zorder=5)

        label_kw = dict(fontsize=10, zorder=6,
                        path_effects=[matplotlib.patheffects.Stroke(linewidth=3, foreground='white'),
                                       matplotlib.patheffects.Normal()])
        ax.text(0, 1.04, r"Rod", ha="center", **label_kw)
        ax.text(1, 1.04, r"Sphere", ha="center", **label_kw)
        ax.text(0.5, 0.46, r"Disk", ha="center", **label_kw)

        ax.plot([0.5, 1], [0.5, 1], color="gray", lw=1, ls=":", alpha=0.7, zorder=4)
        ax.set_xlim(-0.08, 1.12)
        ax.set_ylim(0.42, 1.10)
        ax.set_xlabel(r"$I_1/I_3$", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(r"$I_2/I_3$", fontsize=11)
        ax.set_title(condition, fontsize=11, fontweight="bold", pad=10)
        ax.grid(True, color="lightgray", lw=0.5, zorder=0)

        _overlay_references(ax, ref_ratios, is_legend=(col_idx == 0))

    if last_hb is not None:
        cbar = fig.colorbar(last_hb, ax=axes, pad=0.02, shrink=0.8)
        cbar.set_label(r"$\sigma(\Delta F) / k_BT$ across replicates", fontsize=12)

    fig.suptitle(f"Free Energy Uncertainty (Replicate Std Dev) — {metric_name}",
                 fontsize=13, fontweight="bold")
    out_path = output_dir / f"free_energy_uncertainty_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_meta_cluster_fractions_per_metric(
    metric_dir: Path,
    macro_labels: np.ndarray,
    ctail_rg: np.ndarray,
    ctail_threshold: float,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
    weight_aggregation: str,
) -> None:
    """1 row × N_conditions cols: weighted meta-cluster fractions split by C-tail Rg.

    Bar height = weighted fraction of frames in each macro-cluster for the condition.
    Solid bar = C-tail compact (< threshold); hatched bar stacked on top = extended (>= threshold).
    """
    metric_name = metric_dir.name
    n_conds = len(cfg.ensembles)
    n_frames = len(macro_labels)
    condition_weights = load_condition_weights(
        metric_dir, cfg, cluster_labels, n_per_cluster, n_frames,
        aggregation=weight_aggregation,
    )

    is_extended = ctail_rg >= ctail_threshold

    fig, axes = plt.subplots(1, n_conds, figsize=(4 * n_conds, 5), constrained_layout=True)
    if n_conds == 1:
        axes = [axes]

    x = np.arange(len(MACRO_NAMES))
    width = 0.6

    for col_idx, condition in enumerate(cfg.ensembles):
        ax = axes[col_idx]
        weights = condition_weights[condition]

        if weights is None:
            ax.text(0.5, 0.5, f"No data\nfor {condition}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        total_w = np.sum(weights)

        for i, macro in enumerate(MACRO_NAMES):
            color = CLUSTER_COLOURS[macro]
            mask_compact = (macro_labels == macro) & ~is_extended
            mask_extended = (macro_labels == macro) & is_extended

            frac_compact = np.sum(weights[mask_compact]) / total_w
            frac_extended = np.sum(weights[mask_extended]) / total_w

            ax.bar(x[i], frac_compact, width, color=color, edgecolor="black",
                   alpha=0.85, linewidth=0.8)
            ax.bar(x[i], frac_extended, width, bottom=frac_compact,
                   color=color, edgecolor="black", hatch="//", alpha=0.85, linewidth=0.8)

            total_frac = frac_compact + frac_extended
            if total_frac > 0.02:
                ax.text(x[i], total_frac + 0.005, f"{total_frac:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(MACRO_NAMES, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Weighted Fraction" if col_idx == 0 else "")
        ax.set_title(condition, fontsize=11, fontweight="bold")
        remove_top_right_spines(ax)

    # Shared legend on first column
    above_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="//",
                                  label=f"C-tail ≥ {ctail_threshold} Å")
    below_patch = mpatches.Patch(facecolor="white", edgecolor="black",
                                  label=f"C-tail < {ctail_threshold} Å")
    axes[0].legend(handles=[above_patch, below_patch], fontsize=8, loc="upper right")

    fig.suptitle(
        f"Weighted Meta-Cluster Fractions — {metric_name}\n"
        f"(C-tail threshold = {ctail_threshold} Å)",
        fontsize=13,
        fontweight="bold",
    )
    out_path = output_dir / f"meta_cluster_fractions_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ctail_per_meta_cluster_per_metric(
    metric_dir: Path,
    ctail_rg: np.ndarray,
    macro_labels: np.ndarray,
    ctail_threshold: float,
    cluster_labels: np.ndarray | None,
    n_per_cluster: np.ndarray | None,
    cfg: ExperimentConfig,
    output_dir: Path,
    weight_aggregation: str,
) -> None:
    """N_macro rows × N_conditions cols: per-meta-cluster weighted C-tail Rg histograms."""
    metric_name = metric_dir.name
    n_conds = len(cfg.ensembles)
    n_macro = len(MACRO_NAMES)
    n_frames = len(ctail_rg)
    condition_weights = load_condition_weights(
        metric_dir, cfg, cluster_labels, n_per_cluster, n_frames,
        aggregation=weight_aggregation,
    )

    fig, axes = plt.subplots(n_macro, n_conds,
                              figsize=(4 * n_conds, 3 * n_macro), constrained_layout=True)
    if n_macro == 1 and n_conds == 1:
        axes = np.array([[axes]])
    elif n_macro == 1:
        axes = axes.reshape(1, n_conds)
    elif n_conds == 1:
        axes = axes.reshape(n_macro, 1)

    rg_all_min = np.min(ctail_rg)
    rg_all_max = np.max(ctail_rg)
    bins = np.linspace(rg_all_min, rg_all_max, 30)

    for row_idx, macro in enumerate(MACRO_NAMES):
        mask_macro = macro_labels == macro
        rg_macro = ctail_rg[mask_macro]

        for col_idx, condition in enumerate(cfg.ensembles):
            ax = axes[row_idx, col_idx]
            weights = condition_weights[condition]
            color = cfg.style.ensemble_colors.get(condition, "#808080")

            # Unweighted reference
            ax.hist(rg_macro, bins=bins, density=True, alpha=0.3,
                    color="grey", edgecolor="none")

            if weights is not None:
                weights_macro = weights[mask_macro]
                w_sum = np.sum(weights_macro)
                if w_sum > 0:
                    ax.hist(rg_macro, bins=bins, weights=weights_macro,
                            density=True, alpha=0.8, color=color, edgecolor="none")

            ax.axvline(x=ctail_threshold, color="black", linestyle=":", lw=1.2, alpha=0.7)

            if col_idx == 0:
                ax.set_ylabel(f"{macro}\nDensity", fontweight="bold",
                              color=CLUSTER_COLOURS.get(macro, "black"))
            else:
                ax.set_ylabel("Density")

            if row_idx == 0:
                ax.set_title(condition, fontsize=11, fontweight="bold")

            if row_idx == n_macro - 1:
                ax.set_xlabel(r"C-tail $R_g$ (Å)")

            remove_top_right_spines(ax)

    fig.suptitle(f"C-tail $R_g$ per Meta-Cluster — {metric_name}",
                 fontsize=13, fontweight="bold")
    out_path = output_dir / f"ctail_per_meta_cluster_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_meta_cluster_replicates_per_metric(
    metric_dir: Path,
    macro_labels: np.ndarray,
    ctail_rg: np.ndarray,
    ctail_threshold: float,
    cfg: ExperimentConfig,
    output_dir: Path,
    weight_aggregation: str,
) -> None:
    """Plot replicate-level macro fractions to expose averaging-induced dilution."""
    metric_name = metric_dir.name
    n_conds = len(cfg.ensembles)
    n_frames = len(macro_labels)
    replicate_weights = load_replicate_weights(metric_dir, cfg, n_frames)

    fig, axes = plt.subplots(1, n_conds, figsize=(4 * n_conds, 4.5), constrained_layout=True)
    if n_conds == 1:
        axes = [axes]

    x = np.arange(len(MACRO_NAMES))

    for col_idx, condition in enumerate(cfg.ensembles):
        ax = axes[col_idx]
        reps = replicate_weights[condition]
        if reps is None:
            ax.text(0.5, 0.5, f"No data\nfor {condition}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="red")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        rep_rows = [np.asarray(row, dtype=float) for row in reps]
        rep_colors = plt.cm.Blues(np.linspace(0.45, 0.85, len(rep_rows)))

        for rep_idx, row in enumerate(rep_rows):
            weights = np.nan_to_num(row, nan=0.0)
            weights = weights / np.sum(weights)
            fractions = compute_macro_fraction_rows(
                weights, macro_labels, ctail_rg, ctail_threshold
            )
            y = [fractions[f"{macro}_total"] for macro in MACRO_NAMES]
            ax.plot(
                x, y, marker="o", lw=1.5, color=rep_colors[rep_idx],
                label=f"Rep {rep_idx + 1}"
            )

        agg_weights = aggregate_weight_rows(np.vstack(rep_rows), aggregation=weight_aggregation)
        agg_fractions = compute_macro_fraction_rows(
            agg_weights, macro_labels, ctail_rg, ctail_threshold
        )
        agg_y = [agg_fractions[f"{macro}_total"] for macro in MACRO_NAMES]
        ax.plot(
            x, agg_y, marker="s", lw=2.5, color="black", linestyle="--",
            label=f"{weight_aggregation.title()} aggregate"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(MACRO_NAMES, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.set_title(condition, fontsize=11, fontweight="bold")
        ax.set_ylabel("Weighted Fraction" if col_idx == 0 else "")
        remove_top_right_spines(ax)

    axes[0].legend(fontsize=8, loc="upper right", framealpha=0.8)
    fig.suptitle(
        f"Replicate Macro Fractions — {metric_name}\n"
        f"(aggregate = {weight_aggregation})",
        fontsize=13, fontweight="bold",
    )
    out_path = output_dir / f"meta_cluster_replicates_{metric_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def export_meta_cluster_fraction_summary_per_metric(
    metric_dir: Path,
    macro_labels: np.ndarray,
    ctail_rg: np.ndarray,
    ctail_threshold: float,
    cfg: ExperimentConfig,
    output_dir: Path,
    weight_aggregation: str,
) -> None:
    """Write per-condition aggregate and replicate macro fractions to CSV."""
    metric_name = metric_dir.name
    n_frames = len(macro_labels)
    replicate_weights = load_replicate_weights(metric_dir, cfg, n_frames)
    out_path = output_dir / f"meta_cluster_fraction_summary_{metric_name}.csv"

    fieldnames = [
        "condition", "series", "aggregation",
        "Rod_total", "Rod_compact", "Rod_extended",
        "Wavy_total", "Wavy_compact", "Wavy_extended",
        "Compact_total", "Compact_compact", "Compact_extended",
    ]

    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for condition in cfg.ensembles:
            reps = replicate_weights[condition]
            if reps is None:
                continue

            rep_rows = [np.asarray(row, dtype=float) for row in reps]
            for rep_idx, row in enumerate(rep_rows):
                weights = np.nan_to_num(row, nan=0.0)
                weights = weights / np.sum(weights)
                fractions = compute_macro_fraction_rows(
                    weights, macro_labels, ctail_rg, ctail_threshold
                )
                writer.writerow(
                    {
                        "condition": condition,
                        "series": f"replicate_{rep_idx + 1}",
                        "aggregation": "none",
                        **fractions,
                    }
                )

            agg_weights = aggregate_weight_rows(np.vstack(rep_rows), aggregation=weight_aggregation)
            agg_fractions = compute_macro_fraction_rows(
                agg_weights, macro_labels, ctail_rg, ctail_threshold
            )
            writer.writerow(
                {
                    "condition": condition,
                    "series": "aggregate",
                    "aggregation": weight_aggregation,
                    **agg_fractions,
                }
            )

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
    parser.add_argument(
        "--weight-aggregation",
        default="mean",
        choices=["mean", "median"],
        help="How to aggregate multiple selected weight rows into one condition-level weight vector",
    )
    # Shape / meta-cluster args (all optional — skipped if not provided)
    parser.add_argument("--shape-axes-npy", default=None,
                        help="Path to shape_axes.npy from inertia_moments_clustering.py")
    parser.add_argument("--macro-cluster-labels-npy", default=None,
                        help="Path to macro_cluster_labels.npy from inertia_moments_clustering.py")
    parser.add_argument("--ctail-rg-npy", default=None,
                        help="Path to ctail_rg.npy (all-atom) from clustering script; "
                             "if omitted the CA-based ctail_rg already computed is used")
    parser.add_argument("--ctail-threshold", type=float, default=CTAIL_THRESHOLD_DEFAULT,
                        help=f"C-tail Rg threshold in Å for extended/compact split (default: data median)")
    parser.add_argument("--shape-sel", default="resid 1-135 and name CA",
                        help="Selection range for inertia tensor shape-space plots (default: 'resid 1-135 and name CA')")
    parser.add_argument("--metrics", nargs="+", default="val_mse_min",
                        help="Optional list of metric subdirectories to process (e.g. 'bv_bh_max'). If omitted, all detected are processed.")
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

    def _resolve_optional(path_str: str | None) -> Path | None:
        if path_str is None:
            return None
        p = Path(path_str)
        return p if args.absolute_paths else (script_dir / p).resolve()

    shape_axes_npy = _resolve_optional(args.shape_axes_npy)
    macro_cluster_labels_npy = _resolve_optional(args.macro_cluster_labels_npy)
    ctail_rg_npy = _resolve_optional(args.ctail_rg_npy)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"extracted_dir:    {extracted_dir}")
    print(f"feature_npz:      {feature_npz}")
    print(f"topology_json:    {topology_json}")
    print(f"cluster_labels:   {cluster_labels_npy}")
    print(f"top_pdb:          {top_pdb}")
    print(f"traj_xtc:         {traj_xtc}")
    print(f"shape_axes_npy:   {shape_axes_npy}")
    print(f"macro_labels_npy: {macro_cluster_labels_npy}")
    print(f"ctail_rg_npy:     {ctail_rg_npy}")
    print(f"ctail_threshold:  {args.ctail_threshold} Å")
    print(f"weight_aggregation: {args.weight_aggregation}")
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
    ctail_prot = compute_region_mean_log_pf(log_pf, resid_to_idx, CTAIL_RANGE)

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

    print("Computing Termini contacts...")
    termini_contacts = compute_termini_contacts(top_pdb, traj_xtc, T1_RANGE, T2_RANGE)

    print("Computing RadGyr features...")
    nhead_rg = compute_radgyr(top_pdb, traj_xtc, NHEAD_RG_RANGE)
    nac_rg = compute_radgyr(top_pdb, traj_xtc, NAC_RG_RANGE)
    ctail_rg = compute_radgyr(top_pdb, traj_xtc, CTAIL_RG_RANGE)

    # Load optional shape / meta-cluster data from inertia_moments_clustering.py outputs
    shape_axes: np.ndarray | None = None
    macro_labels: np.ndarray | None = None
    ctail_rg_for_bar = ctail_rg  # default: CA-based from above

    if shape_axes_npy is not None:
        if shape_axes_npy.exists():
            print(f"Loading shape axes from {shape_axes_npy}...")
            shape_axes = np.load(shape_axes_npy)
        else:
            print(f"WARNING: --shape-axes-npy not found at {shape_axes_npy}, skipping shape plots")

    if macro_cluster_labels_npy is not None:
        if macro_cluster_labels_npy.exists():
            print(f"Loading macro cluster labels from {macro_cluster_labels_npy}...")
            macro_labels = np.load(macro_cluster_labels_npy, allow_pickle=True)
        else:
            print(f"WARNING: --macro-cluster-labels-npy not found at {macro_cluster_labels_npy}, skipping bar chart")

    if ctail_rg_npy is not None:
        if ctail_rg_npy.exists():
            print(f"Loading C-tail Rg from {ctail_rg_npy}...")
            ctail_rg_for_bar = np.load(ctail_rg_npy)
        else:
            print(f"WARNING: --ctail-rg-npy not found at {ctail_rg_npy}, using CA-based ctail_rg")

    if args.ctail_threshold is None:
        args.ctail_threshold = np.median(ctail_rg_for_bar)
        print(f"Set ctail_threshold to data midpoint (median): {args.ctail_threshold:.1f} Å")
    else:
        print(f"Using provided ctail_threshold: {args.ctail_threshold:.1f} Å")

    ref_ratios = None
    if shape_axes is not None:
        print(f"Computing reference inertia ratios (selection: '{args.shape_sel}')...")
        ref_ratios = compute_reference_inertia_ratios(REFERENCES, args.shape_sel)

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
                and (args.metrics is None or d.name in args.metrics)
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
            ctail_prot,
            termini_contacts,
            cluster_labels,
            n_per_cluster,
            cfg,
            output_dir,
            args.weight_aggregation,
        )

        plot_radgyr_distributions_per_metric(
            metric_dir,
            nhead_rg,
            nac_rg,
            ctail_rg,
            cluster_labels,
            n_per_cluster,
            cfg,
            output_dir,
            ctail_threshold=args.ctail_threshold,
            weight_aggregation=args.weight_aggregation,
        )

        if shape_axes is not None:
            plot_shape_order_per_metric(
                metric_dir,
                shape_axes,
                cluster_labels,
                n_per_cluster,
                cfg,
                output_dir,
                args.weight_aggregation,
            )

            plot_free_energy_landscape_per_metric(
                metric_dir,
                shape_axes,
                cluster_labels,
                n_per_cluster,
                cfg,
                output_dir,
                ref_ratios=ref_ratios,
                weight_aggregation=args.weight_aggregation,
            )

            plot_free_energy_difference_per_metric(
                metric_dir,
                shape_axes,
                cluster_labels,
                n_per_cluster,
                cfg,
                output_dir,
                ref_ratios=ref_ratios,
                weight_aggregation=args.weight_aggregation,
            )

            plot_free_energy_uncertainty_per_metric(
                metric_dir,
                shape_axes,
                cfg,
                output_dir,
                ref_ratios=ref_ratios,
            )

        if macro_labels is not None:
            plot_meta_cluster_fractions_per_metric(
                metric_dir,
                macro_labels,
                ctail_rg_for_bar,
                args.ctail_threshold,
                cluster_labels,
                n_per_cluster,
                cfg,
                output_dir,
                args.weight_aggregation,
            )

            plot_ctail_per_meta_cluster_per_metric(
                metric_dir,
                ctail_rg_for_bar,
                macro_labels,
                args.ctail_threshold,
                cluster_labels,
                n_per_cluster,
                cfg,
                output_dir,
                args.weight_aggregation,
            )

            plot_meta_cluster_replicates_per_metric(
                metric_dir,
                macro_labels,
                ctail_rg_for_bar,
                args.ctail_threshold,
                cfg,
                output_dir,
                args.weight_aggregation,
            )

            export_meta_cluster_fraction_summary_per_metric(
                metric_dir,
                macro_labels,
                ctail_rg_for_bar,
                args.ctail_threshold,
                cfg,
                output_dir,
                args.weight_aggregation,
            )

    print("\nAll plots saved to:", output_dir)


if __name__ == "__main__":
    main()
