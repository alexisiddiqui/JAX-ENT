"""
PCA + k-means clustering for single MD trajectories.

Compute and plot functions are implemented in ``jaxent.src.analysis.PCA``.
This CLI calls them directly — no inline compute logic.

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
python cluster_trajectory.py --topology_path /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered_all_filtered.xtc \
      --number_of_clusters 100 \
      --num_components 5
python /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/cluster_trajectory.py \
      --topology_path /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered_all_filtered.xtc \
      --number_of_clusters 500 \
      --num_components 10

python /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/cluster_trajectory.py \
      --topology_path /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered_all_filtered.xtc \
      --number_of_clusters 500 \
      --num_components 10
python /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/output/cluster_trajectory.py \
      --topology_path /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_max_plddt_4334.pdb \
      --trajectory_paths /home/alexi/Documents/JAX-ENT/notebooks/CrossValidation/MoPrP/_MoPrP/MoPrP_plddt_ordered.xtc \
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

import MDAnalysis as mda
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

# Shared PCA compute + publication-quality plotting — refactored out of this file
from jaxent.src.analysis.PCA.core import calculate_distances_and_perform_pca
from jaxent.src.analysis.PCA.plots import create_publication_plots


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
    parser = argparse.ArgumentParser(
        description="Cluster molecular dynamics trajectories"
    )
    parser.add_argument(
        "--topology_path", type=str, required=True, help="Path to topology file"
    )
    parser.add_argument(
        "--trajectory_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to trajectory files",
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
        "--num_components",
        type=int,
        default=10,
        help="Number of PCA components (default: 10)",
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
        "--log",
        action="store_true",
        default=True,
        help="Enable logging (default: True)",
    )

    return parser.parse_args()


# perform_kmeans_clustering remains in this file as it is specific to kCluster
# and not shared by iPCA.


def perform_kmeans_clustering(pca_coords, n_clusters):
    """Perform k-means clustering on PCA coordinates"""
    logger.info(f"Performing k-means clustering with {n_clusters} clusters...")

    # Use MiniBatchKMeans for large datasets
    if len(pca_coords) > 10000:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=1000
        )
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

        # For clusters with 2 or fewer frames, just take the first frame
        if len(cluster_frame_indices) <= 2:
            representative_frames[cluster_idx] = cluster_frame_indices[0]
        else:
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
        for cluster_idx, frame_idx in tqdm(
            representative_frames.items(), desc="Saving PDB files"
        ):
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
            os.path.dirname(os.path.abspath(os.getcwd())),
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
    create_publication_plots(
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
