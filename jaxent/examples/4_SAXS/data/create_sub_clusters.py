"""
Performs a second round of clustering on the trajectory data, using the clusters from a previous clustering as a starting point.
In this case this is the RMSD clustering performed in jaxent/examples/4_SAXS/data/cluster_RMSD_references.py

From these clusters, we perform a second round of clustering using iPCA and k-means to further refine the clusters.

We then save the sub-cluster assignments to a csv file as well as the medoids of each sub-cluster as PDB files + their own clustering assignments.

Script args:
- trajectory_path: path to the trajectory file
- topology_path: path to the topology file
- clustering_assignments_csv: path to the clustering assignments csv file
- k: number of sub-clusters per assigned cluster
- seed: seed for reproducibility (default: 42)
- pca_dims: number of iPCA components to use for sub-clustering and visualisation
- reference_paths: list of paths to the reference structures (experimental SAXS .dat files for the curve plot)
- saxs_npz_path: path to the SAXS features .npz (e.g. CaM_SAXS_ordered.npz)
- output_path: path to the output directory

Example output: jaxent/examples/4_SAXS/data/_sub_cluster_output/


source .venv/bin/activate && python jaxent/examples/4_SAXS/data/create_sub_clusters.py \
  --trajectory_path jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_plddt_ordered.xtc \
  --topology_path jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_max_plddt_425.pdb \
  --clustering_assignments_csv jaxent/examples/4_SAXS/data/_RMSD_cluster_output/cluster_assignments.csv \
  --k 5 --seed 42 --pca_dims 10 \
  --reference_paths jaxent/examples/4_SAXS/FOXS/missing_residues/1CLL_apo.pdb.dat jaxent/examples/4_SAXS/FOXS/missing_residues/7PSZ_apo.pdb.dat \
  --saxs_npz_path jaxent/examples/4_SAXS/FOXS/CaM_SAXS_ordered.npz \
  --output_path jaxent/examples/4_SAXS/data/_sub_cluster_output_test2/ 2>&1

"""

import argparse
import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Sub-cluster RMSD-based clusters using iPCA and K-Means")
    parser.add_argument("--trajectory_path", type=str, required=True)
    parser.add_argument("--topology_path", type=str, required=True)
    parser.add_argument("--clustering_assignments_csv", type=str, required=True)
    parser.add_argument("--k", type=int, required=True, help="Sub-clusters per RMSD cluster")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca_dims", type=int, default=10)
    parser.add_argument("--reference_paths", nargs="+", type=str, required=True,
                        help="Paths to experimental SAXS .dat files for the curve plot")
    parser.add_argument("--saxs_npz_path", type=str, default="",
                        help="Path to SAXS features .npz (shape: n_frames x n_q under key 'saxs')")
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# iPCA helpers
# ---------------------------------------------------------------------------

def _pairwise_pdist_chunk(universe, frame_indices, chunk_size=200):
    """Yield (chunk_distances, chunk_slice) for use in partial_fit / transform."""
    atoms = universe.select_atoms("name CA")
    n_atoms = atoms.n_atoms
    n_distances = n_atoms * (n_atoms - 1) // 2
    n_frames = len(frame_indices)

    for chunk_start in range(0, n_frames, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_indices = frame_indices[chunk_start:chunk_end]
        chunk_dist = np.zeros((len(chunk_indices), n_distances))
        for i, frame_idx in enumerate(chunk_indices):
            universe.trajectory[frame_idx]
            chunk_dist[i] = pdist(atoms.positions, metric="euclidean")
        yield chunk_dist, slice(chunk_start, chunk_end)


def compute_pairwise_pdist_pca(universe, frame_indices, pca_dims, chunk_size=200):
    """Run incremental PCA on pairwise CA pdist distances. Returns (pca_coords, fitted_pca)."""
    print(f"  iPCA on {len(frame_indices)} frames (n_components={pca_dims})...")
    ipca_batch = max(pca_dims * 10, chunk_size)
    pca = IncrementalPCA(n_components=pca_dims, batch_size=ipca_batch)

    # Pass 1: fit
    for chunk_dist, _ in tqdm(_pairwise_pdist_chunk(universe, frame_indices, chunk_size), desc="  Fitting PCA"):
        pca.partial_fit(chunk_dist)

    # Pass 2: transform
    n_frames = len(frame_indices)
    pca_coords = np.zeros((n_frames, pca_dims))
    for chunk_dist, sl in tqdm(_pairwise_pdist_chunk(universe, frame_indices, chunk_size), desc="  Transforming"):
        pca_coords[sl] = pca.transform(chunk_dist)

    return pca_coords, pca


def compute_global_pca(universe, all_frame_indices, pca_dims, chunk_size=200):
    """iPCA on the full trajectory for visualisation. Returns (pca_coords, pca)."""
    print(f"Global iPCA on {len(all_frame_indices)} frames...")
    return compute_pairwise_pdist_pca(universe, all_frame_indices, pca_dims, chunk_size)


# ---------------------------------------------------------------------------
# Clustering helpers
# ---------------------------------------------------------------------------

def kmeans_sub_cluster(pca_coords, k, seed):
    if len(pca_coords) > 10000:
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=1000)
    else:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    sub_labels = km.fit_predict(pca_coords)
    return sub_labels, km.cluster_centers_


def find_medoid(frame_indices, pca_coords, cluster_center):
    dists = np.linalg.norm(pca_coords - cluster_center, axis=1)
    return frame_indices[np.argmin(dists)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _cluster_color(combined_label):
    """
    Return a colour for the combined label following the reference image palette:
      0_x → blues, 1_x → cyans/teals, n_x → grays
    """
    prefix = combined_label.split("_")[0]
    idx = int(combined_label.split("_")[1])
    blues   = ["#1565C0", "#1976D2", "#1E88E5", "#42A5F5", "#90CAF9"]
    cyans   = ["#00838F", "#0097A7", "#00ACC1", "#26C6DA", "#80DEEA"]
    grays   = ["#212121", "#424242", "#757575", "#BDBDBD", "#E0E0E0"]
    palettes = {"0": blues, "1": cyans, "n": grays}
    pal = palettes.get(prefix, blues)
    return pal[idx % len(pal)]


def plot_pca_scatter(global_pca_coords, global_frame_indices, frame_to_label, medoid_frames, medoid_labels, pca, output_path):
    """
    PCA scatter coloured by sub-cluster assignment.
    Medoids shown as star markers.
    Variance percentages come from the global PCA object.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # Build colour array and label ordering
    unique_labels = sorted(frame_to_label.values(),
                           key=lambda l: (l.split("_")[0] != "n", l[0], int(l.split("_")[1])))
    # deduplicate while preserving order
    seen = set()
    ordered_labels = []
    for l in unique_labels:
        if l not in seen:
            ordered_labels.append(l)
            seen.add(l)

    # Map global frame index → position in global_pca_coords
    frame_pos = {f: i for i, f in enumerate(global_frame_indices)}

    for label in ordered_labels:
        color = _cluster_color(label)
        idxs = [frame_pos[f] for f, l in frame_to_label.items() if l == label and f in frame_pos]
        if not idxs:
            continue
        pts = global_pca_coords[idxs]
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=50, alpha=0.3,
                   edgecolors="none", label=label, rasterized=True)

    # Medoid stars
    for m_frame, m_label in zip(medoid_frames, medoid_labels):
        if m_frame not in frame_pos:
            continue
        pt = global_pca_coords[frame_pos[m_frame]]
        color = _cluster_color(m_label)
        ax.scatter(pt[0], pt[1], marker="*", s=400, facecolors=color,
                   edgecolors="black", linewidths=1.5, zorder=10)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% var)", fontsize=13)
    ax.set_ylabel(f"PC2 ({var2:.1f}% var)", fontsize=13)
    ax.set_title("iPCA — sub-clusters (k-means on PCA coords)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Two-column legend outside
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=2,
              markerscale=2, fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "sub_cluster_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved sub_cluster_scatter.png")


def plot_saxs_sub_cluster_curves(saxs_npz_path, reference_paths, ordered_labels, frame_to_label, output_path):
    """
    Grid of subplots (n_clusters_per_row × n_rows), one per sub-cluster.
    Each panel shows:
      - Mean I(q) per sub-cluster + ±1 SD shaded band
      - Reference .dat curves (dashed, labelled by filename)
    Log y-scale, linear q-scale.
    """
    if not saxs_npz_path or not os.path.exists(saxs_npz_path):
        print(f"  Skipping SAXS curves plot — {saxs_npz_path} not found.")
        return

    print("  Loading SAXS data...")
    saxs_data = np.load(saxs_npz_path)
    saxs_matrix = saxs_data["saxs"]  # shape (n_frames, n_q)
    n_q = saxs_matrix.shape[1]

    # Load reference curves — also pull q from the first one
    refs = []
    q_vals = None
    for rp in reference_paths:
        if os.path.exists(rp):
            dat = np.loadtxt(rp, comments="#")
            if q_vals is None:
                q_vals = dat[:, 0]
            refs.append((os.path.basename(rp), dat[:, 1]))
    if q_vals is None:
        q_vals = np.linspace(0, 0.5, n_q)

    # Build a mapping: label → list of frame indices
    label_frames = {label: [] for label in ordered_labels}
    for frame, label in frame_to_label.items():
        if label in label_frames:
            label_frames[label].append(frame)

    n_clusters = len(ordered_labels)
    ncols = 5
    nrows = int(np.ceil(n_clusters / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             sharex=True, sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    ref_colors = ["#D62728", "#9467BD", "#8C564B", "#E377C2"]  # red, purple, …

    for idx, label in enumerate(ordered_labels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        frames = label_frames[label]
        n = len(frames)

        if n > 0:
            curves = saxs_matrix[frames]  # (n, n_q)
            mean_curve = curves.mean(axis=0)
            std_curve  = curves.std(axis=0)

            ax.fill_between(q_vals, mean_curve - std_curve, mean_curve + std_curve,
                            alpha=0.25, color="#555555")
            ax.plot(q_vals, mean_curve, color="#333333", linewidth=1.2, label="Mean")

        # Reference curves
        for r_idx, (r_name, r_I) in enumerate(refs):
            ax.plot(q_vals, r_I, linestyle="--", linewidth=1.0,
                    color=ref_colors[r_idx % len(ref_colors)], label=r_name, alpha=0.8)

        ax.set_yscale("log")
        ax.set_title(f"{label} (n={n})", fontsize=9)
        ax.set_xlabel("q (Å⁻¹)", fontsize=8)
        ax.set_ylabel("I(q)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2, which="both")

        if row == 0 and col == ncols - 1:
            ax.legend(fontsize=6, loc="upper right")

    # Hide any unused panels
    for idx in range(n_clusters, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.suptitle("SAXS curves per sub-cluster", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "saxs_sub_cluster_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved saxs_sub_cluster_curves.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_arguments()
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading trajectory and assignments...")
    universe = mda.Universe(args.topology_path, args.trajectory_path)
    df_assign = pd.read_csv(args.clustering_assignments_csv)

    cluster_ids = sorted(df_assign["cluster_id"].unique())
    print(f"Found RMSD clusters: {cluster_ids}")

    # --------------------------------------------------------
    # Global iPCA on ALL frames (for visualisation scatter)
    # --------------------------------------------------------
    all_frame_indices = df_assign["frame"].values
    global_pca_coords, global_pca = compute_global_pca(universe, all_frame_indices, args.pca_dims)

    # Map global frame → global PCA position (just 1-to-1 since all_frame_indices is sequential)
    frame_pos = {int(f): i for i, f in enumerate(all_frame_indices)}

    # --------------------------------------------------------
    # Per-cluster sub-clustering
    # --------------------------------------------------------
    # We'll collect everything and build outputs afterwards
    frame_to_label   = {}   # global frame int → combined_label str
    medoid_frames    = []
    medoid_labels    = []
    ordered_labels   = []   # insertion order: 0_0..0_4, 1_0..1_4, n_0..n_4

    for cid in cluster_ids:
        c_prefix = "n" if cid == -1 else str(int(cid))
        print(f"\n--- RMSD cluster {c_prefix} (id={cid}) ---")

        frames_in_cluster = df_assign[df_assign["cluster_id"] == cid]["frame"].values.astype(int)
        if len(frames_in_cluster) < args.k:
            print(f"  Only {len(frames_in_cluster)} frames < k={args.k}, skipping.")
            continue

        # Per-cluster iPCA for actual k-means
        pca_coords, _ = compute_pairwise_pdist_pca(universe, frames_in_cluster, args.pca_dims)
        sub_labels, centers = kmeans_sub_cluster(pca_coords, args.k, args.seed)[0:2]

        for sub_idx in range(args.k):
            combined_label = f"{c_prefix}_{sub_idx}"
            ordered_labels.append(combined_label)

            mask      = sub_labels == sub_idx
            sub_frames = frames_in_cluster[mask]
            sub_pca   = pca_coords[mask]

            if len(sub_frames) == 0:
                continue

            medoid_frame = find_medoid(sub_frames, sub_pca, centers[sub_idx])
            medoid_frames.append(int(medoid_frame))
            medoid_labels.append(combined_label)

            for f in sub_frames:
                frame_to_label[int(f)] = combined_label

            # Per-sub-cluster directory
            sub_dir = os.path.join(args.output_path, combined_label)
            os.makedirs(sub_dir, exist_ok=True)

            pd.DataFrame({"frame": sub_frames, "combined_label": combined_label}).to_csv(
                os.path.join(sub_dir, "assignments.csv"), index=False
            )

            universe.trajectory[int(medoid_frame)]
            universe.atoms.write(os.path.join(sub_dir, "medoid.pdb"))

    # --------------------------------------------------------
    # Save global CSV outputs
    # --------------------------------------------------------
    print("\nSaving global outputs...")

    rows = []
    for f, label in frame_to_label.items():
        prefix = label.split("_")[0]
        cid = -1 if prefix == "n" else int(prefix)
        sub = int(label.split("_")[1])
        rows.append({"frame": f, "cluster_id": cid, "sub_cluster_id": sub, "combined_label": label})

    out_df = pd.DataFrame(rows).sort_values("frame")
    out_df.to_csv(os.path.join(args.output_path, "sub_cluster_assignments.csv"), index=False)

    pd.DataFrame({"combined_label": medoid_labels, "medoid_frame": medoid_frames}).to_csv(
        os.path.join(args.output_path, "medoid_cluster_mapping.csv"), index=False
    )

    with mda.Writer(os.path.join(args.output_path, "all_medoids.xtc"), universe.atoms.n_atoms) as w:
        for mf in medoid_frames:
            universe.trajectory[mf]
            w.write(universe.atoms)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    print("\nGenerating plots...")
    plot_pca_scatter(global_pca_coords, all_frame_indices, frame_to_label,
                     medoid_frames, medoid_labels, global_pca, args.output_path)

    plot_saxs_sub_cluster_curves(
        args.saxs_npz_path, args.reference_paths,
        ordered_labels, frame_to_label, args.output_path
    )

    print("\nDone!")


if __name__ == "__main__":
    main()