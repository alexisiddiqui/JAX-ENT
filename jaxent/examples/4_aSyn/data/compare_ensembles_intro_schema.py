"""
Principal Moments of Inertia plots to compare ensembles of aSyn along with structural references.

Shaw-MD - green
Tris-MD (this work) - black
Control-MD (this work) - grey
AF2-MSAss (this work) - cyan

Shaw : a99sb, c36m and c22star ensembles
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

# Add current directory to path to import helpers from inertia_moments_clustering.py
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from inertia_moments_clustering import (
    compute_free_energy_grid, _draw_shape_boundary, _triangle_mask,
    _savefig, _overlay_references, compute_reference_ratios,
    compute_dynamic_reference_ratios, REFERENCES, DYNAMIC_REFERENCES,
)

# Constants
MAIN_CHAIN_RANGE = range(1, 115)  # residues 1–114

# Ensemble Registry
ENSEMBLES = OrderedDict([
    ("Shaw-MD", {
        "color": "pink",  # green
        "shape_dir": script_dir / "_cluster_inertia_shaw",
        "top": script_dir / "_aSyn" / "a99sb.pdb",
        "traj": None,
    }),
    ("Tris-MD", {
        "color": "black",
        "shape_dir": script_dir / "_cluster_inertia",
        "top": script_dir / "_aSyn" / "tris_MD" / "md_mol_center_coil.pdb",
        "traj": script_dir / "_aSyn" / "tris_MD" / "tris_all_combined.xtc",
    }),
    ("Control-MD", {
        "color": "grey",
        "shape_dir": script_dir / "_cluster_inertia_control",
        "top": script_dir / "_aSyn" / "control_MD" / "md.gro.pdb",
        "traj": script_dir / "_aSyn" / "control_MD" / "control_all_combined.xtc",
    }),
    ("AF2-MSAss", {
        "color": "cyan",
        "shape_dir": script_dir / "_cluster_inertia_af2",
        "top": script_dir / "_aSyn" / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb",
        "traj": script_dir / "_aSyn" / "aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc",
    }),
])


def load_all_shape_data():
    """Load shape_axes.npy for each ensemble in the registry."""
    shape_data = {}
    for name, cfg in ENSEMBLES.items():
        npy_path = cfg["shape_dir"] / "shape_axes.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Missing precomputed shape axes at {npy_path}")
        print(f"Loading {name} shape data from {npy_path}...")
        axes = np.load(npy_path)
        shape_data[name] = (axes[:, 0], axes[:, 1])
    return shape_data


def plot_fe_landscapes(ens_data, ref_ratios, plots_dir):
    """Figure 1: Free energy landscapes without GMM cluster centers."""
    print("Generating Figure 1: FE landscapes without cluster centers...")
    fig, axes = plt.subplots(1, 4, figsize=(26, 6.5), sharex=True, sharey=True)
    
    cf = None
    for i, (name, (x, y)) in enumerate(ens_data.items()):
        ax = axes[i]
        _, XX, YY, _, F, _ = compute_free_energy_grid(x, y, n_grid=200)
        
        levels = list(range(9))
        cf = ax.contourf(XX, YY, F, levels=levels, cmap="YlGnBu_r", extend="max")
        ax.contour(XX, YY, F, levels=levels, colors="white", linewidths=0.5, alpha=0.5)
        
        _draw_shape_boundary(ax)
        _overlay_references(ax, ref_ratios, is_legend=(i == 3))
        ax.set_title(name, fontsize=16, fontweight="bold", color=ENSEMBLERS_LABEL_COLORS.get(name, "black"))
        
    cbar = fig.colorbar(cf, ax=axes.tolist(), ticks=list(range(9)), pad=0.02, shrink=0.8)
    cbar.set_label(r"Free Energy / $k_BT$", fontsize=14)
    
    plt.suptitle("Principal Moments of Inertia Free Energy Landscapes", fontsize=18, y=1.02, fontweight="bold")
    _savefig(plots_dir / "fe_landscapes.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_fe_landscapes_with_centres(ens_data, centres_xy, ref_ratios, plots_dir):
    """Figure 2: Free energy landscapes with GMM cluster centers overlaid."""
    print("Generating Figure 2: FE landscapes with Tris-MD cluster centers overlaid...")
    fig, axes = plt.subplots(1, 4, figsize=(26, 6.5), sharex=True, sharey=True)
    
    cf = None
    for i, (name, (x, y)) in enumerate(ens_data.items()):
        ax = axes[i]
        _, XX, YY, _, F, _ = compute_free_energy_grid(x, y, n_grid=200)
        
        levels = list(range(9))
        cf = ax.contourf(XX, YY, F, levels=levels, cmap="YlGnBu_r", extend="max")
        ax.contour(XX, YY, F, levels=levels, colors="white", linewidths=0.5, alpha=0.5)
        
        # Overlay the 20 Tris-MD GMM cluster centers
        ax.scatter(
            centres_xy[:, 0], centres_xy[:, 1],
            marker="x", s=80, color="red", zorder=12,
            label="Tris-MD Cluster Centers" if i == 0 else None
        )
        
        _draw_shape_boundary(ax)
        _overlay_references(ax, ref_ratios, is_legend=(i == 3))
        ax.set_title(name, fontsize=16, fontweight="bold", color=ENSEMBLERS_LABEL_COLORS.get(name, "black"))
        if i == 0:
            ax.legend(loc="lower right", fontsize=10)
        
    cbar = fig.colorbar(cf, ax=axes.tolist(), ticks=list(range(9)), pad=0.02, shrink=0.8)
    cbar.set_label(r"Free Energy / $k_BT$", fontsize=14)
    
    plt.suptitle("Moments of Inertia Landscapes with Tris-MD Cluster Centers", fontsize=18, y=1.02, fontweight="bold")
    _savefig(plots_dir / "fe_landscapes_with_centres.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_coverage_overlay(ens_data, ref_ratios, plots_dir):
    """Figure 3: 99th percentile coverage contours overlaid."""
    print("Generating Figure 3: absolute coverage overlay (99th percentile)...")
    from scipy.stats import gaussian_kde
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Pre-generate dense grid
    xg = np.linspace(-0.08, 1.08, 200)
    yg = np.linspace(0.42, 1.10, 200)
    XX, YY = np.meshgrid(xg, yg)
    valid = _triangle_mask(XX, YY)
    
    legend_handles = []
    
    for name, (x, y) in ens_data.items():
        color = ENSEMBLES[name]["color"]
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        
        # Enclose 99% of frames (point densities)
        dens_pts = kde(np.vstack([x, y]))
        threshold = np.percentile(dens_pts, 1.0)  # 1st percentile of densities = top 99% enclosed
        
        # Grid density
        density_grid = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        density_grid[~valid] = 0.0
        
        # Shaded region for coverage
        if name == "Tris-MD":
            ax.contourf(XX, YY, density_grid, levels=[threshold, density_grid.max() + 1.0], colors=[color], alpha=0.25, hatches=["oo"])
        else:
            ax.contourf(XX, YY, density_grid, levels=[threshold, density_grid.max() + 1.0], colors=[color], alpha=0.25)
        # solid outline
        ax.contour(XX, YY, density_grid, levels=[threshold], colors=[color], linewidths=2.0)
        
        # Legend entry
        if name == "Tris-MD":
            patch = mpatches.Patch(facecolor=color, alpha=0.25, hatch="oo", edgecolor=color, label=name)
        else:
            patch = mpatches.Patch(facecolor=color, alpha=0.25, edgecolor=color, label=name)
        legend_handles.append(patch)
        
    _draw_shape_boundary(ax)
    _overlay_references(ax, ref_ratios, is_legend=False)
    ax.axis("off")
    
    ax.set_title("99th Percentile Coverage Overlay", fontsize=16, fontweight="bold", pad=12)
    
    # Add a custom legend combining the ensembles
    ax.legend(handles=legend_handles, loc="lower right", fontsize=11)
    
    plt.tight_layout()
    _savefig(plots_dir / "coverage_overlay_99pct.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_coverage_overlay_with_centres(ens_data, centres_xy, ref_ratios, plots_dir):
    """Figure 3b: 99th percentile coverage contours overlaid with GMM cluster centers."""
    print("Generating Figure 3b: absolute coverage overlay (99th percentile) with GMM centers...")
    from scipy.stats import gaussian_kde
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Pre-generate dense grid
    xg = np.linspace(-0.08, 1.08, 200)
    yg = np.linspace(0.42, 1.10, 200)
    XX, YY = np.meshgrid(xg, yg)
    valid = _triangle_mask(XX, YY)
    
    legend_handles = []
    
    for name, (x, y) in ens_data.items():
        color = ENSEMBLES[name]["color"]
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        
        # Enclose 99% of frames (point densities)
        dens_pts = kde(np.vstack([x, y]))
        threshold = np.percentile(dens_pts, 1.0)  # 1st percentile of densities = top 99% enclosed
        
        # Grid density
        density_grid = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        density_grid[~valid] = 0.0
        
        # Shaded region for coverage
        if name == "Tris-MD":
            ax.contourf(XX, YY, density_grid, levels=[threshold, density_grid.max() + 1.0], colors=[color], alpha=0.25, hatches=["oo"])
        else:
            ax.contourf(XX, YY, density_grid, levels=[threshold, density_grid.max() + 1.0], colors=[color], alpha=0.25)
        # solid outline
        ax.contour(XX, YY, density_grid, levels=[threshold], colors=[color], linewidths=2.0)
        
        # Legend entry
        if name == "Tris-MD":
            patch = mpatches.Patch(facecolor=color, alpha=0.25, hatch="oo", edgecolor=color, label=name)
        else:
            patch = mpatches.Patch(facecolor=color, alpha=0.25, edgecolor=color, label=name)
        legend_handles.append(patch)
        
    # Overlay the 20 Tris-MD GMM cluster centers
    ax.scatter(
        centres_xy[:, 0], centres_xy[:, 1],
        marker="X", s=80, color="black", zorder=12,
        label="Tris-MD Cluster Centers",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )
    
    _draw_shape_boundary(ax)
    _overlay_references(ax, ref_ratios, is_legend=False)
    ax.axis("off")
    
    ax.set_title("99th Percentile Coverage Overlay with Centers", fontsize=16, fontweight="bold", pad=12)
    
    # Add a custom legend combining the ensembles and the cluster centers
    center_handle = ax.scatter(
        [], [],
        marker="X", s=80, color="black",
        label="Tris-MD GMM Centers",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )
    ax.legend(handles=legend_handles + [center_handle], loc="lower right", fontsize=11)
    
    plt.tight_layout()
    _savefig(plots_dir / "coverage_overlay_99pct_with_centres.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_jaccard_coverage(ens_data, plots_dir):
    """Figure 4: Fraction of triangle shape space covered vs. enclosed probability."""
    print("Generating Figure 4: Jaccard coverage curve...")
    from scipy.stats import gaussian_kde
    
    # Pre-generate grid
    xg = np.linspace(-0.08, 1.08, 200)
    yg = np.linspace(0.42, 1.10, 200)
    XX, YY = np.meshgrid(xg, yg)
    valid = _triangle_mask(XX, YY)
    N_valid = np.sum(valid)
    
    percentiles = np.linspace(0, 100, 101)  # Enclosed probability (%)
    
    plt.figure(figsize=(8, 6))
    
    for name, (x, y) in ens_data.items():
        color = ENSEMBLES[name]["color"]
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        
        dens_pts = kde(np.vstack([x, y]))
        density_grid = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        
        fractions = []
        for p in percentiles:
            if p == 0:
                fractions.append(0.0)
            else:
                threshold = np.percentile(dens_pts, 100 - p)
                covered = np.sum((density_grid >= threshold) & valid)
                fractions.append(covered / N_valid)
                
        plt.plot(percentiles, fractions, color=color, lw=2.5, label=name)
        
    plt.xlabel("Enclosed Probability Percentile (%)", fontsize=14)
    plt.ylabel("Fraction of Theoretical Shape Space Covered", fontsize=14)
    plt.title("Shape Space Coverage vs. Distribution Enclosure", fontsize=16, fontweight="bold", pad=12)
    plt.xlim(0, 100)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=11)
    
    plt.tight_layout()
    _savefig(plots_dir / "jaccard_coverage.png", dpi=300, bbox_inches="tight")
    plt.close()


def load_ramachandran_angles(name, select_str="protein and resid 2:94"):
    """Load universe and compute Ramachandran angles with dynamic step-subsampling to be very fast."""
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Ramachandran
    
    cfg = ENSEMBLES[name]
    top = cfg["top"]
    traj = cfg["traj"]
    
    print(f"Loading MD structure and trajectory for {name} KLD Ramachandran...")
    if traj is not None:
        u = mda.Universe(str(top), str(traj))
    else:
        u = mda.Universe(str(top))
        
    n_frames = len(u.trajectory)
    # Dynamically select step to read ~1000 frames max (very fast & statistically stable)
    step = max(1, n_frames // 1000)
    print(f"Analyzing {n_frames} frames (using step={step})...")
    
    sel = u.select_atoms(select_str)
    # verbose=True activates MDAnalysis's built-in tqdm progress bar
    r = Ramachandran(sel).run(step=step, verbose=True)
    return r.results.angles


def plot_pairwise_heatmaps(ens_data, plots_dir):
    """Figure 5: Pairwise 4x4 heatmaps of Jaccard Overlap and Ramachandran symmetric KLD."""
    print("Generating Figure 5: Pairwise Jaccard and KLD Heatmaps...")
    from scipy.stats import gaussian_kde
    
    names = list(ENSEMBLES.keys())
    N = len(names)
    
    # 1. Compute Pairwise Jaccard at 99% Coverage
    print("Computing shape space 99% Jaccard Overlap matrix...")
    xg = np.linspace(-0.08, 1.08, 200)
    yg = np.linspace(0.42, 1.10, 200)
    XX, YY = np.meshgrid(xg, yg)
    valid = _triangle_mask(XX, YY)
    
    masks = {}
    for name, (x, y) in ens_data.items():
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        dens_pts = kde(np.vstack([x, y]))
        threshold = np.percentile(dens_pts, 1.0)  # Enclosing top 99%
        density_grid = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        masks[name] = (density_grid >= threshold) & valid
        
    from scipy.spatial.distance import jaccard as jaccard_distance
    jaccard_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            m_i = masks[names[i]].ravel()
            m_j = masks[names[j]].ravel()
            # jaccard_distance returns 1 - overlap; on identical arrays it returns 0.0
            jaccard_mat[i, j] = 1.0 - jaccard_distance(m_i, m_j)
            
    # 2. Compute Ramachandran symmetric KLD
    print("Computing backbone Ramachandran KLD matrix...")
    angles = {}
    for name in names:
        angles[name] = load_ramachandran_angles(name)
        
    kld_mat = np.zeros((N, N))
    n_bins = 36  # 10 degree bins
    eps = 1e-6
    
    for i in range(N):
        for j in range(i + 1, N):
            n1 = names[i]
            n2 = names[j]
            a1 = angles[n1]
            a2 = angles[n2]
            
            # Align residues
            min_res = min(a1.shape[1], a2.shape[1])
            a1_align = a1[:, :min_res, :]
            a2_align = a2[:, :min_res, :]
            
            res_klds = []
            for r in tqdm(range(min_res), desc=f"KLD {n1} vs {n2}", leave=False):
                phi1, psi1 = a1_align[:, r, 0], a1_align[:, r, 1]
                phi2, psi2 = a2_align[:, r, 0], a2_align[:, r, 1]
                
                h1, _, _ = np.histogram2d(phi1, psi1, bins=n_bins, range=[[-180, 180], [-180, 180]])
                h2, _, _ = np.histogram2d(phi2, psi2, bins=n_bins, range=[[-180, 180], [-180, 180]])
                
                p1 = h1 / np.sum(h1)
                p2 = h2 / np.sum(h2)
                
                # Smooth with small epsilon before normalisation
                p1 = (p1 + eps) / np.sum(p1 + eps)
                p2 = (p2 + eps) / np.sum(p2 + eps)
                
                kld12 = np.sum(p1 * np.log(p1 / p2))
                kld21 = np.sum(p2 * np.log(p2 / p1))
                
                res_klds.append(0.5 * (kld12 + kld21))
                
            mean_kld = np.mean(res_klds)
            kld_mat[i, j] = mean_kld
            kld_mat[j, i] = mean_kld
            
    # Plotting Heatmaps side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))
    
    # Left: Jaccard
    im1 = ax1.imshow(jaccard_mat, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(np.arange(N))
    ax1.set_yticks(np.arange(N))
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=11, fontweight="bold")
    ax1.set_yticklabels(names, fontsize=11, fontweight="bold")
    ax1.set_title("Pairwise Shape Space Jaccard Overlap\n(99% Coverage)", fontsize=14, fontweight="bold", pad=12)
    
    # Annotate Jaccard
    for i in range(N):
        for j in range(N):
            ax1.text(j, i, f"{jaccard_mat[i, j]:.3f}", ha="center", va="center", 
                     color="white" if jaccard_mat[i, j] > 0.5 else "black", fontweight="bold")
            
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Right: KLD
    im2 = ax2.imshow(kld_mat, cmap="YlOrRd")
    ax2.set_xticks(np.arange(N))
    ax2.set_yticks(np.arange(N))
    ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=11, fontweight="bold")
    ax2.set_yticklabels(names, fontsize=11, fontweight="bold")
    ax2.set_title("Pairwise Backbone Ramachandran KLD\n(Symmetric Kullback-Leibler)", fontsize=14, fontweight="bold", pad=12)
    
    # Annotate KLD
    for i in range(N):
        for j in range(N):
            ax2.text(j, i, f"{kld_mat[i, j]:.3f}", ha="center", va="center",
                     color="white" if kld_mat[i, j] > np.max(kld_mat)*0.6 else "black", fontweight="bold")
            
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.suptitle("Pairwise Structural Ensemble Distance Matrices", fontsize=18, y=0.98, fontweight="bold")
    plt.tight_layout()
    _savefig(plots_dir / "pairwise_distances.png", dpi=300, bbox_inches="tight")
    plt.close()


# Specific label colors for titles
ENSEMBLERS_LABEL_COLORS = {
    "Shaw-MD": "pink",
    "Tris-MD": "black",
    "Control-MD": "grey",
    "AF2-MSAss": "cyan"  # cyan mapped label
}


def main():
    plots_dir = script_dir / "_figures" / "compare_ensembles"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=========================================================================")
    print("Starting Structural Ensemble PMI & Dihedral Comparison Plotting Scheme")
    print("=========================================================================")
    
    # 1. Load shape data
    ens_data = load_all_shape_data()
    
    # 2. Load Tris-MD GMM cluster centers
    tris_centers_path = ENSEMBLES["Tris-MD"]["shape_dir"] / "cluster_centers.npy"
    if not tris_centers_path.exists():
        raise FileNotFoundError(f"Missing Tris-MD GMM cluster centers at {tris_centers_path}")
    print(f"Loading canonical cluster centers from {tris_centers_path}...")
    centres_xy = np.load(tris_centers_path)
    
    # 3. Compute single reference ratios for residues 1:114
    print("Computing single-frame reference ratios...")
    ref_ratios = compute_reference_ratios(REFERENCES, "name CA and resid 1:114")
    
    # 4. Generate Figures
    plot_fe_landscapes(ens_data, ref_ratios, plots_dir)
    plot_fe_landscapes_with_centres(ens_data, centres_xy, ref_ratios, plots_dir)
    plot_coverage_overlay(ens_data, ref_ratios, plots_dir)
    plot_coverage_overlay_with_centres(ens_data, centres_xy, ref_ratios, plots_dir)
    plot_jaccard_coverage(ens_data, plots_dir)
    plot_pairwise_heatmaps(ens_data, plots_dir)
    
    print("\n=========================================================================")
    print("All figures successfully created in _figures/compare_ensembles/")
    print("=========================================================================")


if __name__ == "__main__":
    main()
