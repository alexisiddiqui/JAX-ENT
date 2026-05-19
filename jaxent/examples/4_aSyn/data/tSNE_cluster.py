r"""
Clusters trajectory using t-SNE and then performs k-means clustering of the tSNE embedding.

Script args:
--topology: path to topology file
--trajectory: path to trajectory file
--output_dir: path to output directory
--perplexity: list of perplexity values for t-SNE
--n_clusters: list of number of clusters for k-means
--stride: stride for trajectory (default 10)
--atom_selection: atom selection for t-SNE (e.g. "name CA and resid 1-95")
--rad_gyr_defs: JSON dict of region_name: atom_selection for RadGyr coloring
--contact_defs: JSON dict of region_name: atom_selection for contact heatmap
--n_jobs: parallel workers for RMSD computation (default -1 = all cores)
--contact_cutoff: heavy-atom contact distance cutoff in Å (default 4.5)

Process:
Align all structures to the first frame using the atom selection so that
rendered CA traces share a consistent orientation.

Compute aligned pairwise RMSD between structures using:
    MDAnalysis.analysis.encore.confdistmatrix.get_distance_matrix
The distance matrix is cached to disk as rmsd_matrix.npy and reloaded on
subsequent runs.

Framewise features (RadGyr per region, RMSD to reference, inter-region contacts)
are computed on the aligned, strided trajectory.

For each perplexity:
    Compute tSNE embedding (metric="precomputed" on the RMSD matrix).
    Plot tSNE coloured by RMSD to reference.
    Plot tSNE panel coloured by RadGyr per region.
    For each n_clusters:
        Run k-means on the tSNE embedding.
        Plot tSNE coloured by cluster assignment.
        Plot contact heatmap (cluster × region-pair mean contacts).
        Plot flat CA traces (XY projection) for each cluster centroid.

Silhouette scores are computed for all (perplexity, n_clusters) pairs and
summarised in a single plot; the optimal point per perplexity is highlighted.


python jaxent/examples/4_aSyn/data/tSNE_cluster.py \
--topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb \
--trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/tSNE_cluster_output_full_length \
--perplexity  1000 2000 3000  \
--n_clusters 6 7 8 9 10 11 12 13 14 15 \
--atom_selection "name CA" \
--rad_gyr_defs '{"Main-chain": "name CA and resid 1-95","N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96", "C-tail": "name CA and resid 115-141"}' \
--contact_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}' \
--distance_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}'


python jaxent/examples/4_aSyn/data/tSNE_cluster.py \
--topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb \
--trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/tSNE_cluster_output \
--perplexity  1000 2000 3000  \
--n_clusters 2 3 4 5 6 7 8 9  \
--atom_selection "name CA and resid 1-95" \
--rad_gyr_defs '{"Main-chain": "name CA and resid 1-95","N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96", "C-tail": "name CA and resid 115-141"}' \
--contact_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}' \
--distance_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}'


python jaxent/examples/4_aSyn/data/tSNE_cluster.py \
--topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb \
--trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/tSNE_cluster_output \
--perplexity  1000  \
--n_clusters 6  \
--atom_selection "name CA and resid 1-95" \
--rad_gyr_defs '{"Main-chain": "name CA and resid 1-95","N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96", "C-tail": "name CA and resid 115-141"}' \
--contact_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}' \
--distance_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}' \
--2D_RadGyr_defs '{"Main-chain": "name CA and resid 1-95","N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}'



python jaxent/examples/4_aSyn/data/tSNE_cluster.py \
--topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb \
--trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/tSNE_cluster_output_all \
--perplexity  1000  \
--n_clusters 6  \
--atom_selection "name CA and resid 1-95" \
--rad_gyr_defs '{"Main-chain": "name CA and resid 1-95","N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96", "C-tail": "name CA and resid 115-141"}' \
--contact_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}' \
--distance_defs '{"N-tip": "name CA and resid 1-35","N-core": "name CA and resid 25-65", "NAC": "name CA and resid 61-96"}'




"""

import argparse
import ast
import itertools
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from MDAnalysis.coordinates.memory import MemoryReader

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="tSNE + k-means clustering of MD trajectory via pairwise RMSD."
    )
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--perplexity", nargs="+", type=str, default=["30.0"])
    parser.add_argument("--n_clusters", nargs="+", type=str, default=["2", "3", "5"])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--atom_selection", type=str, default="name CA and resid 1-95")
    parser.add_argument(
        "--rad_gyr_defs",
        type=str,
        default=None,
        help='JSON dict: {"region": "atom_selection", ...}',
    )
    parser.add_argument(
        "--contact_defs",
        type=str,
        default=None,
        help='JSON dict: {"region": "atom_selection", ...}',
    )
    parser.add_argument(
        "--distance_defs",
        type=str,
        default=None,
        help='JSON dict: {"region": "atom_selection", ...}',
    )
    parser.add_argument(
        "--2D_RadGyr_defs",
        dest="radgyr_2d_defs",
        type=str,
        default=None,
        help='JSON dict: {"region": "atom_selection", ...}',
    )
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--contact_cutoff", type=float, default=8.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("tsne_cluster")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in [logging.FileHandler(log_file), logging.StreamHandler()]:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------


def load_strided_trajectory(
    topology: str, trajectory: str, stride: int, logger: logging.Logger
) -> mda.Universe:
    u = mda.Universe(topology, trajectory)
    n_total = len(u.trajectory)
    n_strided = (n_total + stride - 1) // stride
    logger.info(f"Loading {n_total} frames with stride={stride} → {n_strided} frames")

    coords = []
    for ts in tqdm(u.trajectory[::stride], desc="Loading frames", total=n_strided):
        coords.append(u.atoms.positions.copy())

    # shape (n_strided, n_atoms, 3) as required by MemoryReader
    coords = np.array(coords, dtype=np.float32)
    u_mem = mda.Universe(topology)
    u_mem.load_new(coords, format=MemoryReader)
    return u_mem


def align_trajectory_to_first(
    u: mda.Universe, selection: str, logger: logging.Logger
) -> mda.Universe:
    logger.info("Aligning trajectory to first frame (in-memory)...")
    aligner = align.AlignTraj(u, u, select=selection, in_memory=True)
    aligner.run()
    return u


# ---------------------------------------------------------------------------
# RMSD matrix
# ---------------------------------------------------------------------------


def _encore_pairwise_rmsd(
    u: mda.Universe, selection: str, n_jobs: int, logger: logging.Logger
) -> np.ndarray:
    from MDAnalysis.analysis.encore.confdistmatrix import get_distance_matrix
    from MDAnalysis.analysis.encore.ensemble import Ensemble

    logger.info("Computing pairwise RMSD via encore.confdistmatrix...")
    ens = Ensemble(universe=u, select=selection)
    dist_matrix = get_distance_matrix(ens, n_jobs=n_jobs)
    matrix = np.array(dist_matrix, dtype=np.float32)
    matrix = np.maximum(matrix, matrix.T)  # ensure symmetry
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _manual_pairwise_rmsd(
    u: mda.Universe, selection: str, n_jobs: int, logger: logging.Logger
) -> np.ndarray:
    from joblib import Parallel, delayed

    sel = u.select_atoms(selection)
    n = len(u.trajectory)

    coords = np.array(
        [
            u.trajectory[i].positions[sel.indices].copy()
            for i in tqdm(range(n), desc="Extracting coords")
        ]
    )

    def _rmsd_pair(i: int, j: int) -> float:
        from MDAnalysis.analysis.rms import rmsd as mda_rmsd

        return float(mda_rmsd(coords[i], coords[j], superposition=True))

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    logger.info(f"Computing {len(pairs):,} pairwise RMSDs (n_jobs={n_jobs})...")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_rmsd_pair)(i, j) for i, j in tqdm(pairs, desc="RMSD pairs")
    )

    matrix = np.zeros((n, n), dtype=np.float32)
    for (i, j), r in zip(pairs, results):
        matrix[i, j] = r
        matrix[j, i] = r
    return matrix


def compute_or_load_rmsd_matrix(
    u: mda.Universe,
    selection: str,
    cache_path: Path,
    n_jobs: int,
    logger: logging.Logger,
) -> np.ndarray:
    if cache_path.exists():
        logger.info(f"Loading cached RMSD matrix from {cache_path}")
        return np.load(cache_path)

    try:
        matrix = _encore_pairwise_rmsd(u, selection, n_jobs, logger)
    except Exception as exc:
        logger.warning(f"encore failed ({exc}), falling back to manual RMSD")
        matrix = _manual_pairwise_rmsd(u, selection, n_jobs, logger)

    np.save(cache_path, matrix)
    logger.info(f"Saved {matrix.shape} RMSD matrix → {cache_path}")
    return matrix


# ---------------------------------------------------------------------------
# Framewise features
# ---------------------------------------------------------------------------


def compute_rmsd_to_reference(
    u: mda.Universe, selection: str, logger: logging.Logger
) -> np.ndarray:
    logger.info("Computing RMSD to reference (first frame)...")
    ref = u.copy()
    ref.trajectory[0]
    R = rms.RMSD(u, ref, select=selection)
    R.run()
    return R.results.rmsd[:, 2].astype(np.float32)


def compute_radgyr(
    u: mda.Universe, defs: dict, logger: logging.Logger
) -> dict[str, np.ndarray]:
    logger.info(f"Computing RadGyr for {len(defs)} regions...")
    sels = {name: u.select_atoms(sel) for name, sel in defs.items()}
    results: dict[str, list] = {name: [] for name in defs}
    for _ts in tqdm(u.trajectory, desc="RadGyr"):
        for name, ag in sels.items():
            results[name].append(ag.radius_of_gyration())
    return {name: np.array(v, dtype=np.float32) for name, v in results.items()}


def compute_contacts(
    u: mda.Universe, defs: dict, cutoff: float, logger: logging.Logger
) -> dict[str, np.ndarray]:
    pairs = list(itertools.combinations(defs.keys(), 2))
    logger.info(
        f"Computing {len(pairs)} region-pair contacts "
        f"(cutoff={cutoff} Å, heavy atoms only)..."
    )
    sels = {
        name: u.select_atoms(f"({sel}) and not name H*")
        for name, sel in defs.items()
    }
    results: dict[str, list] = {f"{a}:{b}": [] for a, b in pairs}
    for _ts in tqdm(u.trajectory, desc="Contacts"):
        for a, b in pairs:
            d = cdist(sels[a].positions, sels[b].positions)
            results[f"{a}:{b}"].append(int(np.sum(d < cutoff)))
    return {k: np.array(v, dtype=np.float32) for k, v in results.items()}


def compute_distances(
    u: mda.Universe, defs: dict, logger: logging.Logger
) -> dict[str, np.ndarray]:
    pairs = list(itertools.combinations(defs.keys(), 2))
    logger.info(
        f"Computing {len(pairs)} region-pair average distances "
        f"(heavy atoms only)..."
    )
    sels = {
        name: u.select_atoms(f"({sel}) and not name H*")
        for name, sel in defs.items()
    }
    results: dict[str, list] = {f"{a}:{b}": [] for a, b in pairs}
    for _ts in tqdm(u.trajectory, desc="Distances"):
        for a, b in pairs:
            # Pairwise distance matrix between all heavy atoms in region a and b
            d = cdist(sels[a].positions, sels[b].positions)
            # Store frame-wise average distance
            results[f"{a}:{b}"].append(float(np.mean(d)))
    return {k: np.array(v, dtype=np.float32) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_tsne_scatter(
    embedding: np.ndarray,
    values: np.ndarray,
    cbar_label: str,
    title: str,
    path: Path,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        cmap=cmap,
        s=2,
        alpha=0.5,
        rasterized=True,
    )
    fig.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("tSNE-1")
    ax.set_ylabel("tSNE-2")
    _clean_axes(ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_tsne_panel(
    embedding: np.ndarray,
    value_dict: dict[str, np.ndarray],
    sup_title: str,
    path: Path,
    cmap: str = "viridis",
) -> None:
    n = len(value_dict)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
    )
    axes = np.array(axes).flatten()

    for i, (name, vals) in enumerate(value_dict.items()):
        sc = axes[i].scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=vals,
            cmap=cmap,
            s=2,
            alpha=0.5,
            rasterized=True,
        )
        fig.colorbar(sc, ax=axes[i], label=name)
        axes[i].set_title(name)
        axes[i].set_xlabel("tSNE-1")
        axes[i].set_ylabel("tSNE-2")
        _clean_axes(axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(sup_title)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_silhouette(scores: dict[float, dict[int, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for perp, cluster_scores in sorted(scores.items()):
        if not cluster_scores:
            continue
        n_list = sorted(cluster_scores)
        s_list = [cluster_scores[n] for n in n_list]
        ax.plot(n_list, s_list, marker="o", label=f"perp={perp:.0f}")
        best = max(cluster_scores, key=cluster_scores.get)
        ax.scatter([best], [cluster_scores[best]], s=100, zorder=5, edgecolors="k")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette scores vs. k")
    ax.legend()
    _clean_axes(ax)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_contact_heatmap(
    contacts: dict[str, np.ndarray],
    labels: np.ndarray,
    n_clusters: int,
    title: str,
    path: Path,
) -> None:
    feat_names = list(contacts)
    data = np.array(
        [
            [contacts[f][labels == c].mean() for f in feat_names]
            for c in range(n_clusters)
        ]
    )
    fig, ax = plt.subplots(
        figsize=(max(5, len(feat_names) * 1.4), max(3, n_clusters * 0.7)),
        constrained_layout=True,
    )
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, label="Mean contacts")
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"Cluster {c}" for c in range(n_clusters)])
    ax.set_title(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_feature_histograms(
    feature_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    n_clusters: int,
    title: str,
    path: Path,
    xlabel: str,
) -> None:
    """
    Plots a histogram for each feature in feature_dict, hued by cluster labels.
    """
    n = len(feature_dict)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
    )
    axes = np.array(axes).flatten()

    try:
        cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", n_clusters)

    palette = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]

    for i, (name, vals) in enumerate(feature_dict.items()):
        df = pd.DataFrame({"value": vals, "Cluster": labels.astype(str)})
        # Sort clusters numerically for consistent legend
        cluster_order = sorted(df["Cluster"].unique(), key=int)
        
        sns.histplot(
            data=df,
            x="value",
            hue="Cluster",
            hue_order=cluster_order,
            element="step",
            palette=palette,
            ax=axes[i],
            common_norm=False,
            alpha=0.3,
        )
        axes[i].set_title(name)
        axes[i].set_xlabel(xlabel)
        _clean_axes(axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_2d_feature_scatter(
    feature_dict: dict[str, np.ndarray],
    labels: np.ndarray,
    n_clusters: int,
    title: str,
    path: Path,
) -> None:
    """
    Plots 2D scatter plots for all combinations of features in feature_dict,
    hued by cluster, with density contours.
    """
    feat_names = list(feature_dict.keys())
    if len(feat_names) < 2:
        return
    
    pairs = list(itertools.combinations(feat_names, 2))
    n = len(pairs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True
    )
    axes = np.array(axes).flatten()

    try:
        cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", n_clusters)

    palette = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]

    for i, (name_x, name_y) in enumerate(pairs):
        df = pd.DataFrame({
            "x": feature_dict[name_x],
            "y": feature_dict[name_y],
            "Cluster": labels.astype(str)
        })
        cluster_order = sorted(df["Cluster"].unique(), key=int)

        # Scatter
        sns.scatterplot(
            data=df,
            x="x",
            y="y",
            hue="Cluster",
            hue_order=cluster_order,
            palette=palette,
            ax=axes[i],
            s=5,
            alpha=0.4,
            legend=False if i > 0 else "brief",
            rasterized=True,
        )
        
        # KDE contours
        sns.kdeplot(
            data=df,
            x="x",
            y="y",
            hue="Cluster",
            hue_order=cluster_order,
            palette=palette,
            ax=axes[i],
            levels=5,
            alpha=0.6,
            linewidths=1,
            legend=False,
        )

        axes[i].set_title(f"{name_x} vs {name_y}")
        axes[i].set_xlabel(f"{name_x} RadGyr (Å)")
        axes[i].set_ylabel(f"{name_y} RadGyr (Å)")
        _clean_axes(axes[i])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_ca_traces(
    u: mda.Universe,
    embedding: np.ndarray,
    labels: np.ndarray,
    perplexity: float,
    n_clusters: int,
    atom_selection: str,
    path: Path,
) -> None:
    try:
        cmap = matplotlib.colormaps["tab20"].resampled(n_clusters)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", n_clusters)  # matplotlib <3.7 fallback

    sel = u.select_atoms(atom_selection)

    # Frame nearest each cluster centroid in tSNE space
    centroid_frames: list[int] = []
    for c in range(n_clusters):
        mask = labels == c
        pts = embedding[mask]
        centre = pts.mean(axis=0)
        frame_idx = int(
            np.where(mask)[0][np.argmin(np.linalg.norm(pts - centre, axis=1))]
        )
        centroid_frames.append(frame_idx)

    ncols = min(3, n_clusters)
    nrows = (n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), constrained_layout=True
    )
    axes = np.array(axes).flatten()

    for c, fidx in enumerate(centroid_frames):
        u.trajectory[fidx]
        pos = sel.positions  # XY projection after pre-alignment
        color = cmap(c / max(n_clusters - 1, 1))
        axes[c].plot(pos[:, 0], pos[:, 1], color=color, lw=1.5, alpha=0.8)
        axes[c].scatter(pos[:, 0], pos[:, 1], s=8, color=color, alpha=0.7)
        axes[c].set_title(f"Cluster {c} (frame {fidx})")
        axes[c].set_aspect("equal")
        _clean_axes(axes[c])

    for j in range(c + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"CA traces — perplexity={perplexity:.0f}, k={n_clusters}")
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(out / "tsne_cluster.log"))
    logger.info(f"Arguments: {vars(args)}")

    rad_gyr_defs: dict = json.loads(args.rad_gyr_defs) if args.rad_gyr_defs else {}
    contact_defs: dict = json.loads(args.contact_defs) if args.contact_defs else {}

    distance_defs: dict = json.loads(args.distance_defs) if args.distance_defs else {}
    radgyr_2d_defs: dict = json.loads(args.radgyr_2d_defs) if args.radgyr_2d_defs else {}

    # Load strided trajectory and align all frames to first frame
    u = load_strided_trajectory(args.topology, args.trajectory, args.stride, logger)
    u = align_trajectory_to_first(u, args.atom_selection, logger)

    # Framewise features (computed once on aligned trajectory)
    rmsd_ref = compute_rmsd_to_reference(u, args.atom_selection, logger)
    radgyr = compute_radgyr(u, rad_gyr_defs, logger) if rad_gyr_defs else {}
    contacts = (
        compute_contacts(u, contact_defs, args.contact_cutoff, logger)
        if contact_defs
        else {}
    )
    distances = (
        compute_distances(u, distance_defs, logger)
        if distance_defs
        else {}
    )
    radgyr_2d = (
        compute_radgyr(u, radgyr_2d_defs, logger)
        if radgyr_2d_defs
        else {}
    )

    # Pairwise RMSD matrix — loaded from cache if available
    rmsd_matrix = compute_or_load_rmsd_matrix(
        u, args.atom_selection, out / "rmsd_matrix.npy", args.n_jobs, logger
    )

    # Parse and flatten numeric lists (supports space or comma separation)
    perplexities = []
    for p in args.perplexity:
        perplexities.extend([float(x) for x in p.split(",") if x.strip()])
    
    n_clusters = []
    for nc in args.n_clusters:
        n_clusters.extend([int(x) for x in nc.split(",") if x.strip()])

    silhouette_scores: dict[float, dict[int, float]] = {}

    for perplexity in perplexities:
        perp_dir = out / f"perp_{perplexity:.0f}"
        perp_dir.mkdir(exist_ok=True)

        logger.info(f"Running tSNE (perplexity={perplexity})...")
        tsne = TSNE(
            n_components=2,
            metric="precomputed",
            perplexity=perplexity,
            random_state=42,
            init="random",
            n_jobs=-1,
        )
        emb = tsne.fit_transform(rmsd_matrix.astype(np.float64))
        np.save(perp_dir / "embedding.npy", emb)

        plot_tsne_scatter(
            emb,
            rmsd_ref,
            "RMSD to ref (Å)",
            f"RMSD to reference — perp={perplexity:.0f}",
            perp_dir / "tsne_rmsd_ref.png",
        )

        if radgyr:
            plot_tsne_panel(
                emb,
                radgyr,
                f"Radius of gyration — perp={perplexity:.0f}",
                perp_dir / "tsne_radgyr.png",
            )

        if distances:
            plot_tsne_panel(
                emb,
                distances,
                f"Region distances — perp={perplexity:.0f}",
                perp_dir / "tsne_distances.png",
                cmap="viridis_r",  # reversed as lower distance is often more interesting
            )

        silhouette_scores[perplexity] = {}

        for k in n_clusters:
            logger.info(f"  k-means k={k}...")
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(emb)
            np.save(perp_dir / f"labels_k{k}.npy", labels)

            if k > 1:
                sil = float(silhouette_score(emb, labels))
                silhouette_scores[perplexity][k] = sil
                logger.info(f"    silhouette={sil:.4f}")

            plot_tsne_scatter(
                emb,
                labels.astype(float),
                "Cluster",
                f"Clusters k={k} — perp={perplexity:.0f}",
                perp_dir / f"tsne_clusters_k{k}.png",
                cmap="tab20",
            )

            if contacts:
                plot_contact_heatmap(
                    contacts,
                    labels,
                    k,
                    f"Mean contacts — perp={perplexity:.0f}, k={k}",
                    perp_dir / f"contact_heatmap_k{k}.png",
                )
                plot_feature_histograms(
                    contacts,
                    labels,
                    k,
                    f"Contact distributions — perp={perplexity:.0f}, k={k}",
                    perp_dir / f"contact_histograms_k{k}.png",
                    xlabel="Number of contacts",
                )

            if distances:
                plot_contact_heatmap(
                    distances,
                    labels,
                    k,
                    f"Mean distances — perp={perplexity:.0f}, k={k}",
                    perp_dir / f"distance_heatmap_k{k}.png",
                )
                plot_feature_histograms(
                    distances,
                    labels,
                    k,
                    f"Distance distributions — perp={perplexity:.0f}, k={k}",
                    perp_dir / f"distance_histograms_k{k}.png",
                    xlabel="Average distance (Å)",
                )

            if radgyr:
                plot_feature_histograms(
                    radgyr,
                    labels,
                    k,
                    f"RadGyr distributions — perp={perplexity:.0f}, k={k}",
                    perp_dir / f"radgyr_histograms_k{k}.png",
                    xlabel="Radius of gyration (Å)",
                )

            if radgyr_2d:
                plot_2d_feature_scatter(
                    radgyr_2d,
                    labels,
                    k,
                    f"2D RadGyr combinations — perp={perplexity:.0f}, k={k}",
                    perp_dir / f"radgyr_2d_scatter_k{k}.png",
                )

            # Always plot CA traces for every cluster centroid
            plot_ca_traces(
                u,
                emb,
                labels,
                perplexity,
                k,
                args.atom_selection,
                perp_dir / f"ca_traces_k{k}.png",
            )

    if any(v for v in silhouette_scores.values()):
        plot_silhouette(silhouette_scores, out / "silhouette_scores.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()
