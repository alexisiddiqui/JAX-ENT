"""
This script takes in a topology and a list of trajectories and k-means clusters the ensemble down to the specified number of clusters.

The script takes in the following args:

- topology_path (str)
- trajectory_paths (list[str])
- atom_selection (str, default: "name CA")
- chunk_size (int, default: 100) # chunk size for memory efficiency when calculating pairwise coords and running PCA
- number_of_clusters (int, default: 500)
- num_components (int, default: 10)
- output_dir (str, default: next to script inside a directory labelled with the topology name, number of clusters and the time)
- save_pdbs (bool, default: False) # Whether to save individual PDB files for each cluster
- log (bool, default: True)

The script performs a memory-efficienct PCA on the pariwise coordinates using pdist. Using these reduced coordinates the clusters are picked with k-means.

The PCA is then comprehensively and professionally plotted using a contour map alongside a scatter plot of the clusters. The clusters are then saved to a file in the output directory as xtc.

The script also saves the arguments and logging information using a logger.

Usage Example:
----------------
python cluster_trajectory.py --topology_path /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered_all_filtered.xtc \
      --number_of_clusters 100 \
      --num_components 5
python /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/cluster_trajectory.py \
      --topology_path /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered_all_filtered.xtc \
      --number_of_clusters 500 \
      --num_components 10

python /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/cluster_trajectory.py \
      --topology_path /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered_all_filtered.xtc \
      --number_of_clusters 500 \
      --num_components 10
python /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/cluster_trajectory.py \
      --topology_path /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /Users/alexi/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered.xtc \
      --number_of_clusters 500 \
      --num_components 10

# new location for this script is found in jaxent/scripts/efficient_cluster.py
python /home/alexi/Documents/JAX-ENT/jaxent/scripts/efficient_cluster.py \
        --topology_path /home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo697_1_af_sample_127_10000_protonated.pdb \
        --trajectory_paths /home/alexi/Documents/ValDX/raw_data/HOIP/HOIP_apo/HOIP_apo697_1_af_sample_127_10000_protonated.xtc \
        --number_of_clusters 500 \
        --num_components 10 

"""

import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


def setup_logger(log_file):
    """Set up the logger to output to both file and console"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    return logger


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Cluster molecular dynamics trajectories")
    parser.add_argument("--topology_path", type=str, required=True, help="Path to topology file")
    parser.add_argument(
        "--trajectory_paths", nargs="+", type=str, required=True, help="Paths to trajectory files"
    )
    parser.add_argument(
        "--atom_selection",
        type=str,
        default="name CA",
        help='Atom selection string (default: "name CA")',
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Chunk size for memory efficiency (default: 100)",
    )
    parser.add_argument(
        "--number_of_clusters",
        type=int,
        default=500,
        help="Number of clusters for k-means (default: 500)",
    )
    parser.add_argument(
        "--num_components", type=int, default=10, help="Number of PCA components (default: 10)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated based on topology name and time)",
    )
    parser.add_argument(
        "--save_pdbs",
        action="store_true",
        help="Save individual PDB files for each cluster (default: False)",
    )
    parser.add_argument(
        "--log", action="store_true", default=True, help="Enable logging (default: True)"
    )

    return parser.parse_args()


def calculate_pairwise_rmsd(universe, selection, chunk_size):
    """Calculate pairwise distances between atoms within each frame"""
    logger.info("Calculating pairwise atomic coordinate distances within frames...")

    # Select atoms for distance calculation
    atoms = universe.select_atoms(selection)
    n_frames = len(universe.trajectory)
    n_atoms = atoms.n_atoms
    n_distances = n_atoms * (n_atoms - 1) // 2
    logger.info(f"Selected {n_atoms} atoms, processing {n_frames} frames")

    # Pre-allocate array for all pairwise distances across all frames
    all_distances = np.zeros((n_frames, n_distances))

    # Process each frame
    for i, ts in enumerate(tqdm(universe.trajectory, desc="Processing frames")):
        # Get atom positions for this frame
        positions = atoms.positions

        # Calculate pairwise distances for this frame only
        frame_distances = pdist(positions, metric="euclidean")

        # Store the distances for this frame
        all_distances[i] = frame_distances

    logger.info(f"Generated distances for {all_distances.shape[0]} frames")

    return all_distances


def perform_pca_on_distances(distances, num_components):
    """Perform PCA on the distance matrix using IncrementalPCA with appropriate batch size"""
    logger.info(f"Performing PCA with {num_components} components...")

    # Set batch size to be at least 10 times the number of components as suggested
    ipca_batch_size = max(num_components * 10, 100)

    # Initialize IncrementalPCA with proper batch size
    pca = IncrementalPCA(n_components=num_components, batch_size=ipca_batch_size)

    # Fit PCA model and transform the data
    pca_coords = pca.fit_transform(distances)

    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    return pca_coords, pca


def perform_kmeans_clustering(pca_coords, n_clusters):
    """Perform k-means clustering on PCA coordinates"""
    logger.info(f"Performing k-means clustering with {n_clusters} clusters...")

    # Use MiniBatchKMeans for large datasets
    if len(pca_coords) > 10000:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    cluster_labels = kmeans.fit_predict(pca_coords)
    cluster_centers = kmeans.cluster_centers_

    # Count frames per cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    logger.info(f"Clustering complete: {len(unique_labels)} clusters")
    logger.info(f"Average frames per cluster: {np.mean(counts):.1f}")
    logger.info(f"Min frames per cluster: {np.min(counts)}")
    logger.info(f"Max frames per cluster: {np.max(counts)}")

    return cluster_labels, cluster_centers, kmeans


def create_publication_plots(pca_coords, cluster_labels, cluster_centers, pca, output_dir):
    """Create publication-quality plots of the PCA results and clustering"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set style for publication-quality plots
    plt.style.use("default")  # Use the default style or choose another valid style
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["axes.edgecolor"] = "black"

    # 1. Create main PCA plot with clusters
    logger.info("Creating PCA projection plot with clusters...")

    # Figure setup with gridspec for complex layout
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 3, 1])

    # Main PCA scatter plot
    ax_main = fig.add_subplot(gs[1, :2])

    # Calculate point density for contour plot
    x, y = pca_coords[:, 0], pca_coords[:, 1]
    sns.kdeplot(x=x, y=y, ax=ax_main, levels=20, cmap="Blues", fill=True, alpha=0.5, zorder=0)

    # Scatter plot of frames colored by cluster
    scatter = ax_main.scatter(
        x, y, c=cluster_labels, cmap="viridis", s=20, alpha=0.7, zorder=10, edgecolor="none"
    )

    # Plot cluster centers
    ax_main.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="red",
        s=80,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        zorder=20,
        label="Cluster Centers",
    )

    # Labels and title
    variance_pc1 = pca.explained_variance_ratio_[0] * 100
    variance_pc2 = pca.explained_variance_ratio_[1] * 100
    ax_main.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=14)
    ax_main.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=14)
    ax_main.set_title("PCA Projection with K-means Clustering", fontsize=16, pad=20)

    # Add histograms for PC1 distribution
    ax_top = fig.add_subplot(gs[0, :2], sharex=ax_main)
    sns.histplot(x, kde=True, ax=ax_top, color="darkblue", alpha=0.6)
    ax_top.set_ylabel("Density", fontsize=12)
    ax_top.set_title("PC1 Distribution", fontsize=14)
    ax_top.tick_params(labelbottom=False)

    # Add histograms for PC2 distribution
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    sns.histplot(y=y, kde=True, ax=ax_right, color="darkblue", alpha=0.6, orientation="horizontal")
    ax_right.set_xlabel("Density", fontsize=12)
    ax_right.set_title("PC2 Distribution", fontsize=14)
    ax_right.tick_params(labelleft=False)

    # Create explained variance ratio plot
    ax_var = fig.add_subplot(gs[2, :])
    components = range(1, len(pca.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    # Plot individual and cumulative explained variance
    bars = ax_var.bar(
        components, pca.explained_variance_ratio_, color="steelblue", alpha=0.7, label="Individual"
    )

    ax_var2 = ax_var.twinx()
    line = ax_var2.plot(
        components,
        cumulative,
        "o-",
        color="firebrick",
        linewidth=2.5,
        markersize=8,
        label="Cumulative",
    )

    # Add explained variance labels
    ax_var.set_xlabel("Principal Component", fontsize=14)
    ax_var.set_ylabel("Explained Variance Ratio", fontsize=14)
    ax_var2.set_ylabel("Cumulative Explained Variance", fontsize=14)
    ax_var.set_title("Explained Variance by Principal Components", fontsize=16, pad=20)

    # Set x-axis to integers
    ax_var.set_xticks(components)
    ax_var2.set_ylim([0, 1.05])

    # Combine legends
    lines, labels = ax_var.get_legend_handles_labels()
    lines2, labels2 = ax_var2.get_legend_handles_labels()
    ax_var.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=12)

    # Add colorbar for cluster labels
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(scatter, cax=cbar_ax)
    cbar.set_label("Cluster Label", fontsize=14, labelpad=15)

    # Save the figure
    plt.tight_layout()
    pca_plot_path = os.path.join(plots_dir, "pca_clusters.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Create 3D PCA plot if we have at least 3 components
    if pca_coords.shape[1] >= 3:
        logger.info("Creating 3D PCA plot...")
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            pca_coords[:, 2],
            c=cluster_labels,
            cmap="viridis",
            s=30,
            alpha=0.7,
        )

        ax.scatter(
            cluster_centers[:, 0],
            cluster_centers[:, 1],
            cluster_centers[:, 2],
            c="red",
            s=100,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
        )

        variance_pc3 = pca.explained_variance_ratio_[2] * 100
        ax.set_xlabel(f"PC1 ({variance_pc1:.1f}% variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({variance_pc2:.1f}% variance)", fontsize=12)
        ax.set_zlabel(f"PC3 ({variance_pc3:.1f}% variance)", fontsize=12)
        ax.set_title("3D PCA Projection with Clusters", fontsize=16)

        plt.colorbar(scatter, ax=ax, label="Cluster Label")
        plt.tight_layout()

        pca_3d_path = os.path.join(plots_dir, "pca_3d.png")
        plt.savefig(pca_3d_path, dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Create cluster size distribution plot
    logger.info("Creating cluster size distribution plot...")
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)

    plt.figure(figsize=(14, 8))
    sns.histplot(counts, kde=True, color="steelblue")
    plt.axvline(np.mean(counts), color="red", linestyle="--", label=f"Mean: {np.mean(counts):.1f}")
    plt.axvline(
        np.median(counts), color="green", linestyle="--", label=f"Median: {np.median(counts):.1f}"
    )

    plt.xlabel("Frames per Cluster", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Distribution of Cluster Sizes", fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()

    cluster_dist_path = os.path.join(plots_dir, "cluster_distribution.png")
    plt.savefig(cluster_dist_path, dpi=300)
    plt.close()

    return {
        "pca_plot": pca_plot_path,
        "pca_3d_plot": pca_3d_path if pca_coords.shape[1] >= 3 else None,
        "cluster_dist": cluster_dist_path,
    }


def save_cluster_trajectories(
    universe, cluster_labels, pca_coords, cluster_centers, output_dir, save_pdbs=False
):
    """Save cluster trajectories to a single XTC file and optionally save representative PDB files

    Parameters:
    -----------
    universe : MDAnalysis.Universe
        The universe containing all frames
    cluster_labels : numpy.ndarray
        Array of cluster labels for each frame
    pca_coords : numpy.ndarray
        PCA coordinates for each frame
    cluster_centers : numpy.ndarray
        K-means computed cluster centers in PCA space
    output_dir : str
        Output directory path
    save_pdbs : bool, optional
        Whether to save individual PDB files for each cluster representative
    """
    logger.info("Saving cluster trajectories...")

    clusters_dir = os.path.join(output_dir, "clusters")
    os.makedirs(clusters_dir, exist_ok=True)

    # Get unique clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # Find the frame in each cluster that is closest to the true cluster center
    representative_frames = {}

    for cluster_idx in unique_clusters:
        # Get indices of frames in this cluster
        cluster_mask = cluster_labels == cluster_idx
        if not np.any(cluster_mask):
            continue

        cluster_frame_indices = np.where(cluster_mask)[0]

        # Get PCA coordinates for frames in this cluster
        cluster_pca_coords = pca_coords[cluster_mask]

        # Get the center for this cluster
        center = cluster_centers[cluster_idx]

        # Calculate distances from each frame to the cluster center
        distances = np.sqrt(np.sum((cluster_pca_coords - center) ** 2, axis=1))

        # Find the frame with minimum distance to center
        min_dist_idx = np.argmin(distances)

        # Get the original frame index
        representative_frame_idx = cluster_frame_indices[min_dist_idx]

        # Store the representative frame
        representative_frames[cluster_idx] = representative_frame_idx

    # Create a single trajectory file with cluster centers only
    all_clusters_file = os.path.join(clusters_dir, "all_clusters.xtc")
    with mda.Writer(all_clusters_file, universe.atoms.n_atoms) as writer:
        # Go through each cluster and save only the representative frame
        for cluster_idx in tqdm(unique_clusters, desc="Saving cluster centers"):
            if cluster_idx in representative_frames:
                frame_idx = representative_frames[cluster_idx]
                universe.trajectory[frame_idx]
                writer.write(universe.atoms)

    # Optionally save representative frames as PDB files
    if save_pdbs:
        for cluster_idx, frame_idx in tqdm(representative_frames.items(), desc="Saving PDB files"):
            universe.trajectory[frame_idx]
            with mda.Writer(
                os.path.join(clusters_dir, f"cluster_{cluster_idx}_rep.pdb")
            ) as pdb_writer:
                pdb_writer.write(universe.atoms)

    # Also save a CSV file mapping frame to cluster
    frame_to_cluster = np.column_stack((np.arange(len(cluster_labels)), cluster_labels))
    np.savetxt(
        os.path.join(clusters_dir, "frame_to_cluster.csv"),
        frame_to_cluster,
        delimiter=",",
        header="frame_index,cluster_label",
        fmt="%d",
        comments="",
    )

    logger.info(
        f"Saved {n_clusters} true cluster centers to a single trajectory file: {all_clusters_file}"
    )
    if save_pdbs:
        logger.info(f"Saved {len(representative_frames)} representative PDB files")
    logger.info("Saved frame-to-cluster mapping to frame_to_cluster.csv")


def calculate_distances_and_perform_pca(universe, selection, num_components, chunk_size):
    """Calculate pairwise distances and perform incremental PCA in chunks"""
    logger.info("Calculating pairwise distances and performing incremental PCA...")

    # Select atoms for distance calculation
    atoms = universe.select_atoms(selection)
    n_frames = len(universe.trajectory)
    n_atoms = atoms.n_atoms
    n_distances = n_atoms * (n_atoms - 1) // 2

    logger.info(f"Selected {n_atoms} atoms, processing {n_frames} frames")
    logger.info(f"Each frame will generate {n_distances} pairwise distances")

    # Initialize IncrementalPCA
    ipca_batch_size = max(num_components * 10, 100)
    pca = IncrementalPCA(n_components=num_components, batch_size=ipca_batch_size)

    # Process frames in chunks
    frame_indices = np.arange(n_frames)

    # Store PCA coordinates for all frames
    pca_coords = np.zeros((n_frames, num_components))

    for chunk_start in tqdm(range(0, n_frames, chunk_size), desc="Processing frame chunks"):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_indices = frame_indices[chunk_start:chunk_end]
        chunk_size_actual = len(chunk_indices)

        # Pre-allocate array just for this chunk of frames
        chunk_distances = np.zeros((chunk_size_actual, n_distances))

        # Process each frame in the chunk
        for i, frame_idx in enumerate(chunk_indices):
            # Go to the specific frame
            universe.trajectory[frame_idx]

            # Get atom positions for this frame
            positions = atoms.positions

            # Calculate pairwise distances for this frame
            frame_distances = pdist(positions, metric="euclidean")

            # Store the distances for this frame in the chunk array
            chunk_distances[i] = frame_distances

        # Partial fit PCA with this chunk
        if chunk_start == 0:
            # For the first chunk, we need to fit and transform
            chunk_pca_coords = pca.fit_transform(chunk_distances)
        else:
            # For subsequent chunks, we partial_fit and transform
            pca.partial_fit(chunk_distances)
            chunk_pca_coords = pca.transform(chunk_distances)

        # Store the PCA coordinates for this chunk
        pca_coords[chunk_start:chunk_end] = chunk_pca_coords

        # Free memory by deleting the chunk distances
        del chunk_distances

    logger.info(f"Completed incremental PCA with {num_components} components")
    logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logger.info(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    return pca_coords, pca


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Start timer
    start_time = datetime.datetime.now()

    # Set up output directory
    if args.output_dir is None:
        topology_name = os.path.splitext(os.path.basename(args.topology_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"{topology_name}_clusters{args.number_of_clusters}_{timestamp}",
        )

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    global logger
    log_file = os.path.join(args.output_dir, "cluster_trajectory.log")
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("Starting trajectory clustering")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")

    # Load universe
    logger.info(f"Loading topology from {args.topology_path}")
    logger.info(f"Loading trajectories: {args.trajectory_paths}")
    universe = mda.Universe(args.topology_path, *args.trajectory_paths)
    logger.info(
        f"Loaded universe with {len(universe.atoms)} atoms and {len(universe.trajectory)} frames"
    )

    # Calculate distances and perform PCA
    pca_coords, pca = calculate_distances_and_perform_pca(
        universe, args.atom_selection, args.num_components, args.chunk_size
    )

    # Perform k-means clustering
    cluster_labels, cluster_centers, kmeans = perform_kmeans_clustering(
        pca_coords, args.number_of_clusters
    )

    # Create publication-quality plots
    plot_paths = create_publication_plots(
        pca_coords, cluster_labels, cluster_centers, pca, args.output_dir
    )

    # Save cluster trajectories
    save_cluster_trajectories(
        universe,
        cluster_labels,
        pca_coords,
        cluster_centers,
        args.output_dir,
        args.save_pdbs,
    )

    # Save analysis data
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.save(os.path.join(data_dir, "pca_coordinates.npy"), pca_coords)
    np.save(os.path.join(data_dir, "cluster_labels.npy"), cluster_labels)
    np.save(os.path.join(data_dir, "cluster_centers.npy"), cluster_centers)

    # End timer and report
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    logger.info("=" * 80)
    logger.info(f"Clustering complete. Results saved to {args.output_dir}")
    logger.info(f"Total execution time: {elapsed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
