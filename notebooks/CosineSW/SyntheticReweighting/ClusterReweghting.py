#!/usr/bin/env python3
"""
Cluster Reweighting Experiment Script

This script performs a cluster-reweighting experiment on molecular dynamics
trajectories to analyze how different clustering approaches affect structure
and weight similarity matrices.

This version uses HDBSCAN for clustering and is optimized for memory efficiency using:
1. Upper triangular matrices for distance storage
2. Chunked processing throughout the pipeline
3. On-demand loading of coordinate data
"""

import argparse
import datetime
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import hdbscan
import jax.numpy as jnp
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, wasserstein_distance
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


def setup_logger(log_file: str) -> logging.Logger:
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


def parse_comma_separated_values(value_str):
    """Parse a comma-separated string into a list."""
    return value_str.split(",")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Perform cluster-reweighting experiment on molecular dynamics trajectories using HDBSCAN"
    )
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
        default=250,
        help="Chunk size for memory efficiency (default: 250)",
    )
    parser.add_argument(
        "--min_samples_values",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="List of min_samples values for HDBSCAN (default: 5 10 20)",
    )
    parser.add_argument(
        "--min_cluster_size_values",
        nargs="+",
        type=int,
        default=[2, 5, 10],
        help="List of min_cluster_size values for HDBSCAN (default: 10 20 50)",
    )
    parser.add_argument(
        "--cluster_selection_epsilon_values",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 1.0],
        help="List of cluster_selection_epsilon values for HDBSCAN (default: 0.5 5.0 10.0)",
    )

    parser.add_argument(
        "--cutoff_values",
        nargs="+",
        type=float,
        default=[0.0],
        help="Distance cutoff values in nm (default: 0.0)",
    )
    parser.add_argument(
        "--power_scales",
        nargs="+",
        type=float,
        default=[-2.0],
        help="Power scaling values (default: -2.0)",
    )
    parser.add_argument(
        "--transformations",
        nargs="+",
        type=str,
        default=["None"],
        help="Transformations to apply (default: None)",
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
        "--temp_dir",
        type=str,
        default=None,
        help="Directory for temporary files (default: system temp directory)",
    )

    return parser.parse_args()


def jax_pairwise_cosine_similarity_chunk(array1: jnp.ndarray, array2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate pairwise cosine similarity between two arrays of vectors.

    Parameters:
    -----------
    array1 : jnp.ndarray
        First array of vectors (n_samples1, n_features)
    array2 : jnp.ndarray
        Second array of vectors (n_samples2, n_features)

    Returns:
    --------
    jnp.ndarray
        Pairwise cosine similarity matrix (n_samples1, n_samples2)
    """
    # Handle empty arrays
    if array1.shape[0] == 0 or array2.shape[0] == 0 or array1.shape[1] == 0:
        return jnp.empty((array1.shape[0], array2.shape[0]))

    # Compute dot products
    dot_products = jnp.matmul(array1, array2.T)

    # Compute norms
    norms1 = jnp.sqrt(jnp.sum(array1**2, axis=1))
    norms2 = jnp.sqrt(jnp.sum(array2**2, axis=1))

    # Create a 2D grid of norm products
    norm_products = jnp.outer(norms1, norms2)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    norm_products = jnp.maximum(norm_products, epsilon)

    # Compute cosine similarities
    similarity_matrix = dot_products / norm_products

    # Ensure values are in a better range (0 to 2)
    similarity_matrix = 1 + jnp.clip(similarity_matrix, -1.0, 1.0)

    return similarity_matrix


def numpy_pairwise_cosine_similarity_chunk(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    NumPy implementation of chunked pairwise cosine similarity.

    Parameters:
    -----------
    array1 : numpy.ndarray
        First array of vectors (n_samples1, n_features)
    array2 : numpy.ndarray
        Second array of vectors (n_samples2, n_features)

    Returns:
    --------
    numpy.ndarray
        Pairwise cosine similarity matrix (n_samples1, n_samples2)
    """
    # Handle empty arrays
    if array1.shape[0] == 0 or array2.shape[0] == 0 or array1.shape[1] == 0:
        return np.empty((array1.shape[0], array2.shape[0]))

    # Compute dot products
    dot_products = np.matmul(array1, array2.T)

    # Compute norms
    norms1 = np.sqrt(np.sum(array1**2, axis=1))
    norms2 = np.sqrt(np.sum(array2**2, axis=1))

    # Create a 2D grid of norm products
    norm_products = np.outer(norms1, norms2)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    norm_products = np.maximum(norm_products, epsilon)

    # Compute cosine similarities
    similarity_matrix = dot_products / norm_products

    # Ensure values are in a better range (0 to 2)
    similarity_matrix = 1 + np.clip(similarity_matrix, -1.0, 1.0)

    return similarity_matrix


def cluster_weights_similarity_matrix(center_weights):
    """
    Calculate a meaningful similarity matrix between cluster weights
    that gives a clear diagonal and graduated similarities.

    Parameters:
    -----------
    center_weights : numpy.ndarray
        Array of weights for each cluster center

    Returns:
    --------
    numpy.ndarray
        Similarity matrix with clear diagonal and graduated values
    """
    n_centers = len(center_weights)
    similarity_matrix = np.zeros((n_centers, n_centers))

    # Find min and max for scaling
    weight_min = np.min(center_weights)
    weight_max = np.max(center_weights)
    weight_range = weight_max - weight_min

    # Avoid division by zero
    if weight_range < 1e-8:
        return np.ones((n_centers, n_centers)) * 2.0

    for i in range(n_centers):
        for j in range(n_centers):
            # Calculate normalized absolute difference (0 when identical, 1 when maximally different)
            normalized_diff = abs(center_weights[i] - center_weights[j]) / weight_range

            # Convert to similarity (1 when identical, 0 when maximally different)
            similarity = 1.0 - normalized_diff

            # Scale to [1,2] range to match your other matrices
            similarity_matrix[i, j] = 1.0 + similarity

    return similarity_matrix


def gaussian_kernel_similarity(center_weights, sigma=None):
    """
    Calculate similarity matrix between normalized cluster weights using a Gaussian kernel.
    Specially adapted for weights that sum to 1 (probability distributions).

    Parameters:
    -----------
    center_weights : numpy.ndarray
        Array of normalized weights for each cluster center (sum to 1)
    sigma : float, optional
        Bandwidth parameter. If None, calculated based on weight distribution.

    Returns:
    --------
    numpy.ndarray
        Similarity matrix with clear diagonal using Gaussian kernel
    """
    n_centers = len(center_weights)
    similarity_matrix = np.zeros((n_centers, n_centers))

    # For normalized weights, use a scale-appropriate sigma
    if sigma is None:
        # For normalized weights, a good heuristic is to use a fraction of the mean weight
        mean_weight = 1.0 / n_centers  # This would be the mean for uniform weights

        # Calculate actual variance of weights
        weight_variance = np.var(center_weights)

        # Set sigma based on variance, but ensure it's not too small
        sigma = max(np.sqrt(weight_variance), mean_weight * 0.1)

        # If weights are nearly identical, use a small fraction of mean
        if sigma < 1e-8:
            sigma = mean_weight * 0.1

    # Calculate Gaussian kernel with normalized sigma
    for i in range(n_centers):
        for j in range(n_centers):
            # Squared Euclidean distance
            sq_distance = (center_weights[i] - center_weights[j]) ** 2

            # Gaussian kernel with appropriate sigma
            similarity_matrix[i, j] = np.exp(-sq_distance / (2 * sigma**2))

    # Scale to [1,2] range to match other matrices
    similarity_matrix = 1.0 + similarity_matrix

    return similarity_matrix


def laplacian_kernel_similarity(weights, sigma=None):
    """
    Compute similarity matrix between scalar weights using the Laplacian kernel (valid RKHS kernel).

    Parameters:
    -----------
    weights : numpy.ndarray
        1D array of normalized scalar weights (e.g., cluster probabilities).
    sigma : float, optional
        Bandwidth parameter. If None, use median of pairwise absolute differences.

    Returns:
    --------
    numpy.ndarray
        Symmetric similarity matrix where similarity = exp(-|w_i - w_j| / sigma).
    """
    weights = np.asarray(weights).reshape(-1, 1)  # Ensure 2D input for pdist

    # Compute pairwise absolute differences (L1 distances for scalars)
    pairwise_dists = pdist(weights, "cityblock")
    pairwise_dists = squareform(pairwise_dists)

    # Set sigma using median heuristic (robust to outliers)
    if sigma is None:
        non_zero_dists = pairwise_dists[pairwise_dists > 0]
        sigma = np.median(non_zero_dists) if len(non_zero_dists) > 0 else 1.0

    # Avoid division by zero
    sigma = max(sigma, 1e-8)

    # Compute Laplacian kernel
    similarity_matrix = np.exp(-pairwise_dists / sigma)

    return 1 + similarity_matrix


def bhattacharyya_similarity(weights):
    """
    Compute similarity matrix between normalized scalar weights using the Bhattacharyya coefficient (valid RKHS kernel).

    Parameters:
    -----------
    weights : numpy.ndarray
        1D array of normalized scalar weights (sum to 1).

    Returns:
    --------
    numpy.ndarray
        Symmetric similarity matrix where similarity = sqrt(w_i * w_j).
    """
    weights = np.asarray(weights)

    # Compute Bhattacharyya coefficient via outer product of sqrt(weights)
    sqrt_weights = np.sqrt(weights)
    similarity_matrix = np.outer(sqrt_weights, sqrt_weights)

    return 1 + similarity_matrix


def hellinger_affinity(weights):
    """
    Compute Hellinger affinity matrix (diagonal = 1, off-diagonal = 1 - 0.5*(sqrt(w_i) - sqrt(w_j))^2).

    Parameters:
    -----------
    weights : numpy.ndarray
        1D array of normalized scalar weights (sum to 1).

    Returns:
    --------
    numpy.ndarray
        Symmetric similarity matrix with diagonal = 1 and values in [0, 1].
    """
    sqrt_weights = np.sqrt(weights)
    pairwise_dists = pdist(sqrt_weights[:, np.newaxis], "sqeuclidean")
    pairwise_dists = squareform(pairwise_dists)
    similarity_matrix = 1 - 0.5 * pairwise_dists
    return 1 + similarity_matrix


def pairwise_cosine_similarity_chunk(
    array1: Union[np.ndarray, jnp.ndarray],
    array2: Optional[Union[np.ndarray, jnp.ndarray]] = None,
) -> Union[np.ndarray, jnp.ndarray]:
    """
    Calculate chunked pairwise cosine similarity.
    Uses JAX if available, otherwise falls back to NumPy.

    Parameters:
    -----------
    array1 : Union[numpy.ndarray, jnp.ndarray]
        First array of vectors (n_samples1, n_features)
    array2 : Union[numpy.ndarray, jnp.ndarray], optional
        Second array of vectors (n_samples2, n_features)
        If None, compares array1 with itself

    Returns:
    --------
    Union[numpy.ndarray, jnp.ndarray]
        Pairwise cosine similarity matrix
    """
    if array2 is None:
        array2 = array1

    try:
        # Try to use JAX implementation
        return jax_pairwise_cosine_similarity_chunk(jnp.array(array1), jnp.array(array2))
    except:
        # Fall back to NumPy implementation
        logger.info("JAX implementation failed, falling back to NumPy")
        return numpy_pairwise_cosine_similarity_chunk(array1, array2)


def compute_cluster_weights(cluster_labels: np.ndarray) -> np.ndarray:
    """
    Compute weights based on cluster density, assigning 0 weight to noise points.

    Parameters:
    -----------
    cluster_labels : numpy.ndarray
        Array of cluster labels (-1 for noise)

    Returns:
    --------
    numpy.ndarray
        Array of weights for each frame (noise points have weight 0)
    """
    n_frames = len(cluster_labels)
    weights = np.zeros(n_frames, dtype=float)

    # Filter out noise points for density calculation
    non_noise_labels = cluster_labels[cluster_labels != -1]

    if len(non_noise_labels) == 0:
        logger.warning("No non-noise points found. Returning zero weights.")
        return weights  # Return all zeros if only noise

    # Count occurrences of each actual cluster
    unique_clusters, counts = np.unique(non_noise_labels, return_counts=True)
    log_counts = np.log(counts + 1)  # Use log counts for density

    if np.sum(log_counts) == 0:
        logger.warning("Sum of log counts is zero. Assigning uniform weights to non-noise points.")
        # Assign uniform weights to non-noise points if log_counts sum is zero
        num_non_noise = len(non_noise_labels)
        uniform_weight = 1.0 / num_non_noise if num_non_noise > 0 else 0
        weights[cluster_labels != -1] = uniform_weight
        return weights

    # Create a density map for actual clusters
    cluster_densities = dict(zip(unique_clusters, log_counts))

    # Assign weights based on cluster density, noise gets 0
    for i, label in enumerate(cluster_labels):
        if label != -1:
            weights[i] = cluster_densities[label]

    # Normalize weights of non-noise points to sum to 1
    total_weight = np.sum(weights)
    if total_weight > 1e-8:
        weights = weights / total_weight
    else:
        # Handle case where all non-noise points might have ended up with zero weight (e.g., single point clusters)
        num_non_noise = len(non_noise_labels)
        uniform_weight = 1.0 / num_non_noise if num_non_noise > 0 else 0
        weights[cluster_labels != -1] = uniform_weight

    return weights


def generate_atom_pairwise_distance_vector(
    positions: np.ndarray,
    cutoff: Optional[float] = None,
    power_scale: float = 1.0,
    transform: Optional[str] = None,
) -> np.ndarray:
    """
    Generate atom-pairwise distance vector (upper triangular) with various configurations.

    Parameters:
    -----------
    positions : numpy.ndarray
        Atomic positions (n_atoms, 3)
    cutoff : float, optional
        Distance cutoff in nanometers (None, 0.6, 1.0, 1.4)
    power_scale : float, optional
        Power scaling factor (-2 to 2)
    transform : str, optional
        Transformation type ('None' or 'log')

    Returns:
    --------
    numpy.ndarray
        Upper triangular distance vector after applying configurations
    """
    # Calculate raw pairwise distances (upper triangular only)
    dist_vector = pdist(positions, metric="euclidean")

    # Apply distance cutoff if specified
    if cutoff is not None and cutoff > 0:
        # Convert cutoff from nm to angstroms (assuming positions are in angstroms)
        cutoff_angstroms = cutoff * 10.0
        dist_vector = np.minimum(dist_vector, cutoff_angstroms)

    # Apply power scaling
    if power_scale != 1.0:
        # Avoid division by zero or negative distances
        epsilon = 1e-8
        dist_vector = np.maximum(dist_vector, epsilon)
        dist_vector = dist_vector**power_scale

    # Apply transformation
    if transform == "log":
        # Avoid log(0)
        epsilon = 1e-8
        dist_vector = np.maximum(dist_vector, epsilon)
        dist_vector = np.log(dist_vector)

    return dist_vector


def compute_w1_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Compute the W1 (Wasserstein-1) distance between two matrices.

    Parameters:
    -----------
    matrix1 : numpy.ndarray
        First matrix
    matrix2 : numpy.ndarray
        Second matrix

    Returns:
    --------
    float
        W1 distance between the two matrices
    """
    # Flatten matrices for comparison
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()

    # Compute Wasserstein distance
    return wasserstein_distance(flat1, flat2)


def compute_matrix_distance(
    matrix1: np.ndarray, matrix2: np.ndarray, metric: str = "rmsd"
) -> float:
    """
    Compute the distance between two matrices or vectors using various metrics.

    Parameters:
    -----------
    matrix1 : numpy.ndarray
        First matrix or vector
    matrix2 : numpy.ndarray
        Second matrix or vector
    metric : str, optional
        Distance metric ('rmsd', 'cosine', 'w1', 'correlation')

    Returns:
    --------
    float
        Distance between the two matrices or vectors
    """
    # Check if inputs are vectors (1D) rather than matrices
    is_vector = matrix1.ndim == 1 and matrix2.ndim == 1

    if metric == "rmsd":
        return np.sqrt(np.mean((matrix1 - matrix2) ** 2))
    elif metric == "cosine":
        # Flatten matrices if they're not already vectors
        if not is_vector:
            flat1 = matrix1.flatten()
            flat2 = matrix2.flatten()
        else:
            flat1 = matrix1
            flat2 = matrix2

        # Return cosine distance
        dot_product = np.sum(flat1 * flat2)
        norm1 = np.sqrt(np.sum(flat1**2))
        norm2 = np.sqrt(np.sum(flat2**2))
        epsilon = 1e-8
        return 1 - dot_product / (norm1 * norm2 + epsilon)
    elif metric == "w1":
        # Wasserstein-1 distance (Earth Mover's Distance)
        # Flatten matrices if they're not already vectors
        if not is_vector:
            flat1 = matrix1.flatten()
            flat2 = matrix2.flatten()
        else:
            flat1 = matrix1
            flat2 = matrix2
        return wasserstein_distance(flat1, flat2)
    elif metric == "correlation":
        # Correlation distance (1 - Pearson correlation coefficient)
        if not is_vector:
            flat1 = matrix1.flatten()
            flat2 = matrix2.flatten()
        else:
            flat1 = matrix1
            flat2 = matrix2

        # Handle cases with zero standard deviation
        if np.std(flat1) == 0 or np.std(flat2) == 0:
            # If both are constant and equal, correlation is 1, distance is 0
            if np.all(flat1 == flat1[0]) and np.all(flat2 == flat2[0]) and flat1[0] == flat2[0]:
                return 0.0
            # Otherwise, correlation is undefined or 0, distance is 1
            else:
                return 1.0

        corr, _ = pearsonr(flat1, flat2)
        return 1 - corr
    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_numel_normalized_magnitude(matrix: np.ndarray) -> float:
    """
    Compute the numel-normalized magnitude of a matrix.

    Parameters:
    -----------
    matrix : numpy.ndarray
        Input matrix

    Returns:
    --------
    float
        Numel-normalized magnitude
    """
    return np.sum(np.abs(matrix)) / matrix.size


def calculate_pairwise_distances_and_save(
    universe: mda.Universe, selection: str, chunk_size: int, temp_dir: str
) -> Tuple[str, int]:
    """
    Calculate pairwise distances in chunks and save to a temporary file.

    Parameters:
    -----------
    universe : mda.Universe
        MDAnalysis universe object
    selection : str
        Atom selection string
    chunk_size : int
        Chunk size for processing
    temp_dir : str
        Directory to save temporary files

    Returns:
    --------
    Tuple[str, int]
        Path to the file containing pairwise distances and number of distances per frame
    """
    logger.info("Calculating pairwise distances in chunks and saving to temp file...")

    atoms = universe.select_atoms(selection)
    n_frames = len(universe.trajectory)

    # Create temporary file for distances
    distances_file = os.path.join(temp_dir, "pairwise_distances.npy")

    # Get dimension of distance vector (upper triangular)
    n_atoms = atoms.n_atoms
    n_distances = n_atoms * (n_atoms - 1) // 2

    logger.info(f"Number of atoms: {n_atoms}, distances per frame: {n_distances}")

    # Initialize memory-mapped array for distances
    distances = np.memmap(
        distances_file, dtype=np.float32, mode="w+", shape=(n_frames, n_distances)
    )

    # Process in chunks
    for chunk_start in tqdm(range(0, n_frames, chunk_size), desc="Calculating distances"):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_size_actual = chunk_end - chunk_start

        # Process each frame in the chunk
        for i, ts in enumerate(universe.trajectory[chunk_start:chunk_end]):
            # Get atom positions
            pos = atoms.positions.copy()

            # Calculate distance vector (upper triangular)
            dist_vector = pdist(pos, metric="euclidean")

            # Store in memmap array
            distances[chunk_start + i] = dist_vector

        # Flush changes to disk
        distances.flush()

    # Close memmap
    del distances

    logger.info(f"Pairwise distances saved to {distances_file}")
    return distances_file, n_distances


def perform_pca_on_distances(
    distances_file: str,
    num_components: int,
    chunk_size: int,
    n_frames: int,
    n_distances: int,
    pca_file: str,
) -> Tuple[str, IncrementalPCA]:
    """
    Perform incremental PCA on distance vectors, saving to a specified file.

    Parameters:
    -----------
    distances_file : str
        Path to file containing distance vectors
    num_components : int
        Number of PCA components
    chunk_size : int
        Chunk size for processing
    n_frames : int
        Total number of frames
    n_distances : int
        Number of distances per frame (size of distance vector)
    pca_file : str
        Path to save PCA coordinates

    Returns:
    --------
    Tuple[str, IncrementalPCA]
        Path to file containing PCA coordinates and PCA object
    """
    logger.info("Performing incremental PCA on distance vectors...")

    # Initialize PCA
    ipca_batch = max(num_components * 10, chunk_size)
    pca = IncrementalPCA(n_components=num_components, batch_size=ipca_batch)

    # Create memory-mapped array for PCA coordinates
    pca_coords = np.memmap(pca_file, dtype=np.float32, mode="w+", shape=(n_frames, num_components))

    # Load distances as memmap with explicit shape
    distances = np.memmap(distances_file, dtype=np.float32, mode="r", shape=(n_frames, n_distances))

    # Process in chunks
    for chunk_start in tqdm(range(0, n_frames, chunk_size), desc="PCA processing"):
        chunk_end = min(chunk_start + chunk_size, n_frames)
        chunk_distances = distances[chunk_start:chunk_end]

        # Fit or update PCA
        if chunk_start == 0:
            coords = pca.fit_transform(chunk_distances)
        else:
            pca.partial_fit(chunk_distances)
            coords = pca.transform(chunk_distances)

        # Store PCA coordinates
        pca_coords[chunk_start:chunk_end] = coords

        # Flush changes
        pca_coords.flush()

    # Close memmaps
    del distances
    del pca_coords

    logger.info(
        f"Completed incremental PCA ({num_components} comps), explained variance: {sum(pca.explained_variance_ratio_):.2%}"
    )

    return pca_file, pca


def perform_hdbscan_clustering(
    pca_file: str,
    min_samples: int,
    min_cluster_size: int,
    cluster_selection_epsilon: float,
    n_frames: int,
    num_components: int,
    temp_dir: str,
) -> Tuple[str, int, int]:
    """
    Perform HDBSCAN clustering on PCA coordinates.

    Parameters:
    -----------
    pca_file : str
        Path to file containing PCA coordinates
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point.
    min_cluster_size : int
        The minimum number of samples in a cluster.
    cluster_selection_epsilon : float
        The distance threshold for cluster merging. 0.0 means automatic determination.
    n_frames : int
        Total number of frames
    num_components : int
        Number of PCA components
    temp_dir : str
        Directory to save temporary files

    Returns:
    --------
    Tuple[str, int, int]
        Path to file containing cluster labels, number of clusters found (excluding noise), number of noise points
    """
    param_str = f"ms{min_samples}_mcs{min_cluster_size}_cse{cluster_selection_epsilon:.1f}"
    logger.info(
        f"Performing HDBSCAN clustering with min_samples={min_samples}, "
        f"min_cluster_size={min_cluster_size}, cluster_selection_epsilon={cluster_selection_epsilon:.1f}..."
    )

    # Load PCA coordinates
    pca_coords = np.memmap(pca_file, dtype=np.float32, mode="r", shape=(n_frames, num_components))

    # Initialize and fit HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon
        if cluster_selection_epsilon > 0
        else None,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(pca_coords)

    # Save cluster labels
    labels_file = os.path.join(temp_dir, f"cluster_labels_{param_str}.npy")
    np.save(labels_file, labels)

    # Analyze results
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise label (-1)
    n_noise = np.sum(labels == -1)

    logger.info(f"HDBSCAN complete: Found {n_clusters} clusters.")
    logger.info(f"Number of noise points: {n_noise} ({n_noise / n_frames:.2%})")

    if n_clusters > 0:
        cluster_sizes = [np.sum(labels == i) for i in unique_labels if i != -1]
        logger.info(f"Average cluster size: {np.mean(cluster_sizes):.1f}")
        logger.info(f"Min cluster size: {np.min(cluster_sizes) if cluster_sizes else 0}")
        logger.info(f"Max cluster size: {np.max(cluster_sizes) if cluster_sizes else 0}")

    else:
        logger.warning("HDBSCAN found no clusters (only noise or empty dataset).")

    # Close memmap
    del pca_coords

    return labels_file, n_clusters, n_noise


def identify_hdbscan_representative_frames(
    pca_file: str,
    labels_file: str,
    n_frames: int,
    num_components: int,
    chunk_size: int,
) -> np.ndarray:
    """
    Identify representative frames (medoids) for each HDBSCAN cluster.
    The medoid is the point within the cluster closest to the cluster's centroid.

    Parameters:
    -----------
    pca_file : str
        Path to file containing PCA coordinates
    labels_file : str
        Path to file containing cluster labels
    n_frames : int
        Total number of frames
    num_components : int
        Number of PCA components
    chunk_size : int
        Chunk size for processing

    Returns:
    --------
    numpy.ndarray
        Indices of representative frames (medoids) for each cluster. Returns empty array if no clusters found.
    """
    logger.info("Identifying representative frames (medoids) for HDBSCAN clusters...")

    # Load PCA coordinates and cluster labels
    pca_coords = np.memmap(pca_file, dtype=np.float32, mode="r", shape=(n_frames, num_components))
    cluster_labels = np.load(labels_file)

    unique_cluster_labels = sorted([label for label in np.unique(cluster_labels) if label != -1])
    n_clusters_found = len(unique_cluster_labels)

    if n_clusters_found == 0:
        logger.warning("No clusters found by HDBSCAN. Cannot identify representative frames.")
        del pca_coords
        return np.array([], dtype=int)

    representative_frames = np.full(n_clusters_found, -1, dtype=int)

    for i, cluster_id in enumerate(tqdm(unique_cluster_labels, desc="Finding medoids")):
        # Find indices of points belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]

        if len(cluster_indices) == 0:
            logger.warning(f"Cluster {cluster_id} has no points. Skipping.")
            continue

        # Extract PCA coordinates for this cluster
        cluster_points = pca_coords[cluster_indices]

        # Calculate the centroid (mean) of the cluster points
        centroid = np.mean(cluster_points, axis=0)

        # Calculate squared Euclidean distances from each point in the cluster to the centroid
        distances_to_centroid = np.sum((cluster_points - centroid) ** 2, axis=1)

        # Find the index (within the cluster) of the point closest to the centroid
        medoid_cluster_index = np.argmin(distances_to_centroid)

        # Get the original frame index of the medoid
        medoid_frame_index = cluster_indices[medoid_cluster_index]
        representative_frames[i] = medoid_frame_index

    # Close memmap
    del pca_coords

    # Filter out any potential -1 entries if clusters were skipped
    representative_frames = representative_frames[representative_frames != -1]

    logger.info(f"Identified {len(representative_frames)} representative frames (medoids).")
    return representative_frames


def calculate_representative_similarity_matrices(
    universe: mda.Universe,
    selection: str,
    distances_file: str,
    n_distances: int,
    representative_frames: np.ndarray,
    config: Tuple,
    config_string: str,
    hdbscan_params: Tuple[int, int, float],
    temp_dir: str,
    weights_similarity_func,
    weights_similarity_label: str,
) -> Dict[str, np.ndarray]:
    """
    Calculate similarity matrices only for representative frames of HDBSCAN clusters.

    Parameters:
    -----------
    universe : mda.Universe
        MDAnalysis universe
    selection : str
        Atom selection string
    distances_file : str
        Path to file containing distances
    n_distances : int
        Number of distances per frame
    representative_frames : np.ndarray
        Indices of frames representing each cluster (e.g., medoids)
    config : Tuple
        Configuration tuple (cutoff, power, transform)
    config_string : str
        Configuration string
    hdbscan_params : Tuple[int, int, float]
        HDBSCAN parameters (min_samples, min_cluster_size, cluster_selection_epsilon)
    temp_dir : str
        Directory for temporary files
    weights_similarity_func : function
        Function to compute weights similarity
    weights_similarity_label : str
        Label for the weights similarity function

    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary of similarity matrices for cluster representatives. Returns empty dict if no representatives.
    """
    min_samples, min_cluster_size, cluster_selection_epsilon = hdbscan_params
    param_str = f"ms{min_samples}_mcs{min_cluster_size}_cse{cluster_selection_epsilon:.1f}"

    n_representatives = len(representative_frames)
    if n_representatives == 0:
        logger.warning(
            f"No representative frames provided for HDBSCAN params {param_str}. Skipping similarity calculation."
        )
        return {}

    logger.info(
        f"Calculating similarity matrices for {config_string}, HDBSCAN params {param_str} ({n_representatives} representatives)"
    )

    cutoff, power, transform = config

    # Access distance vectors with explicit shape
    distance_vectors = np.memmap(
        distances_file,
        dtype=np.float32,
        mode="r",
        shape=(universe.trajectory.n_frames, n_distances),
    )

    # Get distance vectors (flattened upper triangle) for the representative frames only
    representative_triu_vectors = []
    for idx in representative_frames:
        # Get original distance vector
        dist_vector = distance_vectors[idx].copy()

        # Apply configuration
        if cutoff is not None and cutoff > 0:
            cutoff_angstroms = cutoff * 10.0
            dist_vector = np.minimum(dist_vector, cutoff_angstroms)

        if power != 1.0:
            epsilon = 1e-8
            dist_vector = np.maximum(dist_vector, epsilon)
            dist_vector = dist_vector**power

        if transform == "log":
            epsilon = 1e-8
            dist_vector = np.maximum(dist_vector, epsilon)
            dist_vector = np.log(dist_vector)

        representative_triu_vectors.append(dist_vector)

    representative_triu_vectors = np.array(representative_triu_vectors)

    # Load cluster labels to compute cluster weights
    labels_file = os.path.join(temp_dir, f"cluster_labels_{param_str}.npy")
    cluster_labels = np.load(labels_file)

    # Compute cluster weights (noise has weight 0)
    all_frame_weights = compute_cluster_weights(cluster_labels)

    # Extract weights corresponding to the representative frames' clusters
    representative_labels = cluster_labels[representative_frames]

    # Calculate weights for the clusters themselves based on size/density
    non_noise_labels = cluster_labels[cluster_labels != -1]
    unique_clusters, counts = np.unique(non_noise_labels, return_counts=True)
    log_counts = np.log(counts + 1)
    total_log_counts = np.sum(log_counts)

    if total_log_counts < 1e-8:
        # Handle case where all clusters are single points or counts are zero
        cluster_weights_map = {
            label: 1.0 / len(unique_clusters) if len(unique_clusters) > 0 else 0
            for label in unique_clusters
        }
    else:
        cluster_weights_map = {
            label: lc / total_log_counts for label, lc in zip(unique_clusters, log_counts)
        }

    # Assign the calculated cluster weight to each representative frame
    representative_cluster_weights = np.array(
        [cluster_weights_map.get(label, 0) for label in representative_labels]
    )

    # Ensure weights sum to 1
    if (
        len(representative_cluster_weights) > 0
        and abs(np.sum(representative_cluster_weights) - 1.0) > 1e-6
    ):
        logger.warning(
            f"Sum of representative cluster weights is {np.sum(representative_cluster_weights)}. Renormalizing."
        )
        sum_weights = np.sum(representative_cluster_weights)
        if sum_weights > 1e-8:
            representative_cluster_weights /= sum_weights
        else:  # Assign uniform if sum is zero
            representative_cluster_weights = (
                np.ones(n_representatives) / n_representatives
                if n_representatives > 0
                else np.array([])
            )

    # Compute uniform weights for the representatives
    uniform_weights = (
        np.ones(n_representatives) / n_representatives if n_representatives > 0 else np.array([])
    )

    # Apply weights to the flattened upper triangle vectors
    uniform_weighted = representative_triu_vectors * uniform_weights[:, np.newaxis]
    cluster_weighted = representative_triu_vectors * representative_cluster_weights[:, np.newaxis]

    # Calculate similarity matrices based on the flattened upper triangle vectors
    structure_similarity = pairwise_cosine_similarity_chunk(
        representative_triu_vectors, representative_triu_vectors
    )
    uniform_similarity = pairwise_cosine_similarity_chunk(uniform_weighted, uniform_weighted)
    cluster_density_similarity = pairwise_cosine_similarity_chunk(
        cluster_weighted, cluster_weighted
    )

    # Compute the selected weights similarity function
    uniform_weights_similarity = weights_similarity_func(uniform_weights)
    cluster_weights_similarity = weights_similarity_func(representative_cluster_weights)

    # Clean up
    del distance_vectors

    return {
        "structure": structure_similarity,
        "uniform": uniform_similarity,
        "cluster_density": cluster_density_similarity,
        "representative_frames": representative_frames,
        "representative_weights": representative_cluster_weights,
        "all_frame_weights": all_frame_weights,
        "cluster_weights_similarity": cluster_weights_similarity,
        "uniform_weights_similarity": uniform_weights_similarity,
        "weights_similarity_label": weights_similarity_label,
    }


def extract_upper_triangle(matrix):
    """
    Extract the upper triangular portion of a matrix (excluding diagonal).

    Parameters:
    -----------
    matrix : numpy.ndarray
        Square input matrix

    Returns:
    --------
    numpy.ndarray
        Flattened upper triangular portion
    """
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def plot_w1_distance(
    w1_distances: Dict[Tuple[str, Tuple[int, int, float]], float],
    similarity_matrices: Dict[Tuple[str, Tuple[int, int, float]], Dict[str, np.ndarray]],
    hdbscan_params_list: List[Tuple[int, int, float]],
    configs: List[str],
    output_dir: str,
    weights_similarity_label: str,
):
    """
    Plot W1 distances between uniform and cluster density matrices.
    """
    plots_dir = os.path.join(output_dir, "plots", weights_similarity_label)
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))

    # Group configurations by their main parameters for better visualization
    grouped_configs = {}
    for config in configs:
        # Parse the config string to get the key parameters
        config_parts = config.split(", ")
        cutoff = config_parts[0].split("=")[1]
        power = config_parts[1].split("=")[1]
        transform = config_parts[2].split("=")[1] if len(config_parts) > 2 else "None"

        # Create more descriptive label
        if transform != "None":
            key = f"cutoff={cutoff}nm, power={power}, {transform}"
        else:
            key = f"cutoff={cutoff}nm, power={power}"

        grouped_configs[key] = config

    # Define a color cycle and markers for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_configs)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Dictionary to store data points for each configuration
    # We'll organize by number of clusters found (not by HDBSCAN parameters)
    config_data = {}

    # Group data by configuration and map to number of clusters found
    for idx, (label, config) in enumerate(grouped_configs.items()):
        # For each configuration, collect pairs of (n_clusters, distance)
        n_clusters_values = []
        distances = []

        for params in hdbscan_params_list:
            key = (config, params)
            if key in w1_distances:
                # Count the number of clusters actually found
                n_clusters = None
                if key in similarity_matrices and similarity_matrices[key]:
                    if "structure" in similarity_matrices[key]:
                        n_clusters = similarity_matrices[key]["structure"].shape[0]

                if n_clusters is not None and n_clusters > 0:
                    n_clusters_values.append(n_clusters)
                    distances.append(w1_distances[key])

        if n_clusters_values:
            # Sort by number of clusters
            sorted_pairs = sorted(zip(n_clusters_values, distances))
            n_clusters_values = [pair[0] for pair in sorted_pairs]
            distances = [pair[1] for pair in sorted_pairs]
            config_data[label] = (n_clusters_values, distances)

    # Plot data for each configuration
    for idx, (label, (n_clusters_values, distances)) in enumerate(config_data.items()):
        plt.plot(
            n_clusters_values,
            distances,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linestyle="-",
            label=label,
            markersize=8,
        )

    plt.xlabel("Number of Clusters Found", fontsize=14)
    plt.ylabel("Wasserstein-1 Distance", fontsize=14)
    plt.title(
        "Wasserstein-1 Distance between Uniform and\nCluster Density Weight Matrices (Upper Triangle Only)",
        fontsize=16,
    )
    plt.xscale("log")  # log scale for better visualization
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add annotation explaining W1 distance
    plt.figtext(
        0.5,
        0.01,
        "Lower W1 distance indicates more similar distributions between uniform and cluster-based weighting",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(plots_dir, "w1_distance.png"), dpi=300)
    plt.close()


def plot_numel_normalized_magnitude(
    normalized_magnitudes: Dict[Tuple[str, Tuple[int, int, float]], float],
    similarity_matrices: Dict[Tuple[str, Tuple[int, int, float]], Dict[str, np.ndarray]],
    hdbscan_params_list: List[Tuple[int, int, float]],
    configs: List[str],
    output_dir: str,
    weights_similarity_label: str,
):
    """
    Plot numel-normalized magnitudes organized by final number of clusters.
    """
    plots_dir = os.path.join(output_dir, "plots", weights_similarity_label)
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 12))

    # Group configurations by their main parameters for better visualization
    grouped_configs = {}
    for config in configs:
        # Parse the config string to get the key parameters
        config_parts = config.split(", ")
        cutoff = config_parts[0].split("=")[1]
        power = config_parts[1].split("=")[1]
        transform = config_parts[2].split("=")[1] if len(config_parts) > 2 else "None"

        # Create more descriptive label
        if transform != "None":
            key = f"cutoff={cutoff}nm, power={power}, {transform}"
        else:
            key = f"cutoff={cutoff}nm, power={power}"

        grouped_configs[key] = config

    # Define a color cycle and markers for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(grouped_configs)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Dictionary to store data points for each configuration
    # We'll organize by number of clusters found (not by HDBSCAN parameters)
    config_data = {}

    # Group data by configuration and map to number of clusters found
    for idx, (label, config) in enumerate(grouped_configs.items()):
        # For each configuration, collect pairs of (n_clusters, magnitude)
        n_clusters_values = []
        magnitudes = []

        for params in hdbscan_params_list:
            key = (config, params)
            if key in normalized_magnitudes:
                # Determine the number of clusters from the similarity matrix data
                n_clusters = None
                if key in similarity_matrices and similarity_matrices[key]:
                    if "structure" in similarity_matrices[key]:
                        n_clusters = similarity_matrices[key]["structure"].shape[0]

                if n_clusters is not None and n_clusters > 0:
                    n_clusters_values.append(n_clusters)
                    magnitudes.append(normalized_magnitudes[key])

        if n_clusters_values:
            # Sort by number of clusters
            sorted_pairs = sorted(zip(n_clusters_values, magnitudes))
            n_clusters_values = [pair[0] for pair in sorted_pairs]
            magnitudes = [pair[1] for pair in sorted_pairs]
            config_data[label] = (n_clusters_values, magnitudes)

    # Plot data for each configuration
    for idx, (label, (n_clusters_values, magnitudes)) in enumerate(config_data.items()):
        plt.plot(
            n_clusters_values,
            magnitudes,
            marker=markers[idx % len(markers)],
            color=colors[idx % len(colors)],
            linestyle="-",
            label=label,
            markersize=8,
        )

    plt.xlabel("Number of Clusters Found", fontsize=14)
    plt.ylabel("Numel-normalized Magnitude", fontsize=14)
    plt.title(
        "Average Absolute Value of Structure Similarity Matrix Elements\n(Upper Triangle Only)",
        fontsize=16,
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add annotation explaining the metric
    plt.figtext(
        0.5,
        0.01,
        "Higher values indicate stronger overall similarities between cluster representatives",
        ha="center",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(os.path.join(plots_dir, "numel_normalized_magnitude.png"), dpi=300)
    plt.close()


def plot_matrix_distances(
    matrix_distances: Dict[Tuple[str, Tuple[int, int, float], str, str], float],
    similarity_matrices: Dict[Tuple[str, Tuple[int, int, float]], Dict[str, np.ndarray]],
    hdbscan_params_list: List[Tuple[int, int, float]],
    configs: List[str],
    metrics: List[str],
    comparisons: List[Tuple[str, str]],  # List of (key, label) tuples
    output_dir: str,
    weights_similarity_label: str,
):
    """
    Plot matrix distances for specified comparisons organized by final cluster count.
    """
    plots_dir = os.path.join(output_dir, "plots", weights_similarity_label)
    os.makedirs(plots_dir, exist_ok=True)

    # Define color and marker palettes
    colors = plt.cm.tab10(np.linspace(0, 1, len(comparisons)))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Metric descriptions for annotations
    metric_descriptions = {
        "rmsd": "Root Mean Square Deviation (lower is more similar)",
        "cosine": "Cosine Distance (0 = identical, 1 = orthogonal)",
        "w1": "Wasserstein-1 Distance (lower is more similar)",
        "correlation": "Correlation Distance (1 - Pearson corr; 0 = identical, 2 = anti-correlated)",
    }

    for config in configs:
        # Parse the config string to get descriptive parameters
        config_parts = config.split(", ")
        cutoff_part = config_parts[0].replace("cutoff=", "Cutoff: ")
        power_part = config_parts[1].replace("power=", "Power Scale: ")
        transform_part = (
            config_parts[2].replace("transform=", "Transform: ")
            if len(config_parts) > 2
            else "Transform: None"
        )

        plt.figure(figsize=(14, 10))

        # Create subplot for each metric to make it more readable
        for m_idx, metric in enumerate(metrics):
            plt.subplot(2, 2, m_idx + 1)

            # Prepare data for this configuration and metric, organized by number of clusters
            comp_data = {}

            # Iterate through the provided comparisons
            for c_idx, (comp_key, comp_label) in enumerate(comparisons):
                # For each comparison, collect pairs of (n_clusters, distance)
                n_clusters_values = []
                distances = []

                for params in hdbscan_params_list:
                    key = (config, params, metric, comp_key)
                    if key in matrix_distances and not np.isnan(matrix_distances[key]):
                        # Determine the number of clusters from the similarity matrix data
                        n_clusters = None
                        sim_key = (config, params)
                        if sim_key in similarity_matrices and similarity_matrices[sim_key]:
                            if "structure" in similarity_matrices[sim_key]:
                                n_clusters = similarity_matrices[sim_key]["structure"].shape[0]

                        if n_clusters is not None and n_clusters > 0:
                            n_clusters_values.append(n_clusters)
                            distances.append(matrix_distances[key])

                if n_clusters_values:
                    # Sort by number of clusters
                    sorted_pairs = sorted(zip(n_clusters_values, distances))
                    n_clusters_values = [pair[0] for pair in sorted_pairs]
                    distances = [pair[1] for pair in sorted_pairs]
                    comp_data[comp_label] = (n_clusters_values, distances)

            # Plot data for each comparison
            for c_idx, (comp_label, (n_clusters_values, distances)) in enumerate(comp_data.items()):
                plt.plot(
                    n_clusters_values,
                    distances,
                    marker=markers[c_idx % len(markers)],
                    color=colors[c_idx % len(colors)],
                    linestyle="-",
                    label=comp_label,
                    markersize=6,
                )

            plt.xlabel("Number of Clusters Found", fontsize=12)
            plt.ylabel(f"{metric.upper()} Distance", fontsize=12)
            plt.title(f"{metric.upper()} Distance (Upper Triangle Only)", fontsize=12)
            plt.xscale("log")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=10)

            # Add metric description
            if metric in metric_descriptions:
                plt.annotate(
                    metric_descriptions[metric],
                    xy=(0.5, 0.02),
                    xycoords="axes fraction",
                    ha="center",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                )

        # Add overall title with detailed configuration
        plt.suptitle(
            f"Matrix Distances: {cutoff_part}, {power_part}, {transform_part}", fontsize=16
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        config_filename = config.replace(" ", "_").replace("=", "_").replace(",", "_")
        plt.savefig(os.path.join(plots_dir, f"matrix_distances_{config_filename}.png"), dpi=300)
        plt.close()


def plot_heatmaps(
    similarity_matrices: Dict[Tuple[str, Tuple[int, int, float]], Dict[str, np.ndarray]],
    hdbscan_params_list: List[Tuple[int, int, float]],
    configs: List[str],
    output_dir: str,
    weights_similarity_label: str,
    intervals: int = 5,
):
    """
    Plot heatmaps showing similarity matrices for various HDBSCAN configurations.
    Creates separate plots for cluster weights similarity and structure similarity.
    """
    plots_dir = os.path.join(output_dir, "plots", weights_similarity_label)
    os.makedirs(plots_dir, exist_ok=True)

    # Determine valid parameter sets that have data
    valid_param_sets = sorted(
        list(
            set(
                [
                    params
                    for config, params in similarity_matrices.keys()
                    if similarity_matrices[(config, params)]
                    and "cluster_weights_similarity" in similarity_matrices[(config, params)]
                ]
            )
        ),
        key=lambda x: (
            x[0],
            x[1],
            x[2],
        ),  # Sort by min_samples, min_cluster_size, cluster_selection_epsilon
    )

    if not valid_param_sets:
        logger.warning("No valid data found for heatmap plotting.")
        return

    param_intervals = valid_param_sets

    # 1. Plot cluster weights similarity matrices
    fig_width = max(15, 5 * len(configs))
    fig_height = max(10, 5 * len(param_intervals))
    fig, axes = plt.subplots(
        len(param_intervals), len(configs), figsize=(fig_width, fig_height), squeeze=False
    )
    fig.suptitle(
        "Evolution of Cluster Weights Similarity Matrices (Hellinger Affinity)", fontsize=16
    )

    for i, params in enumerate(param_intervals):
        min_samples, min_cluster_size, cluster_selection_epsilon = params
        for j, config in enumerate(configs):
            key = (config, params)
            ax = axes[i, j]

            if (
                key in similarity_matrices
                and similarity_matrices[key]
                and "cluster_weights_similarity" in similarity_matrices[key]
            ):
                matrix = similarity_matrices[key]["cluster_weights_similarity"]
                n_clusters_found = matrix.shape[0]

                if n_clusters_found > 0:
                    im = ax.imshow(matrix, cmap="viridis")

                    config_parts = config.split(", ")
                    cutoff_part = config_parts[0].replace("cutoff=", "cutoff: ")
                    power_part = config_parts[1].replace("power=", "power: ")
                    transform_part = (
                        config_parts[2].replace("transform=", "transform: ")
                        if len(config_parts) > 2
                        else ""
                    )

                    title = (
                        f"ms={min_samples}, mcs={min_cluster_size}, "
                        f"cse={cluster_selection_epsilon:.1f}\n"
                        f"({n_clusters_found} clusters)\n{cutoff_part}\n{power_part}"
                    )
                    if transform_part:
                        title += f"\n{transform_part}"

                    ax.set_title(title, fontsize=9)

                    if i == len(param_intervals) - 1:
                        ax.set_xlabel(f"Cluster Representatives (n={n_clusters_found})", fontsize=8)
                    if j == 0:
                        ax.set_ylabel(f"Cluster Representatives (n={n_clusters_found})", fontsize=8)

                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Hellinger Affinity (1-2)", fontsize=8)
                else:
                    ax.set_title(
                        f"ms={min_samples}, mcs={min_cluster_size}, cse={cluster_selection_epsilon:.1f}\nNo clusters found",
                        fontsize=9,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

            else:
                param_str = (
                    f"ms={min_samples}, mcs={min_cluster_size}, cse={cluster_selection_epsilon:.1f}"
                )
                ax.set_title(f"{param_str}\nData unavailable", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(plots_dir, "cluster_weights_similarity_heatmaps.png"), dpi=300)
    plt.close()

    # 2. Plot structure similarity matrices
    fig, axes = plt.subplots(
        len(param_intervals), len(configs), figsize=(fig_width, fig_height), squeeze=False
    )
    fig.suptitle("Evolution of Structure Similarity Matrices", fontsize=16)

    for i, params in enumerate(param_intervals):
        min_samples, min_cluster_size, cluster_selection_epsilon = params
        for j, config in enumerate(configs):
            key = (config, params)
            ax = axes[i, j]

            if (
                key in similarity_matrices
                and similarity_matrices[key]
                and "structure" in similarity_matrices[key]
            ):
                matrix = similarity_matrices[key]["structure"]
                n_clusters_found = matrix.shape[0]

                if n_clusters_found > 0:
                    im = ax.imshow(matrix, cmap="viridis")

                    config_parts = config.split(", ")
                    cutoff_part = config_parts[0].replace("cutoff=", "cutoff: ")
                    power_part = config_parts[1].replace("power=", "power: ")
                    transform_part = (
                        config_parts[2].replace("transform=", "transform: ")
                        if len(config_parts) > 2
                        else ""
                    )

                    title = (
                        f"ms={min_samples}, mcs={min_cluster_size}, "
                        f"cse={cluster_selection_epsilon:.1f}\n"
                        f"({n_clusters_found} clusters)\n{cutoff_part}\n{power_part}"
                    )
                    if transform_part:
                        title += f"\n{transform_part}"

                    ax.set_title(title, fontsize=9)

                    if i == len(param_intervals) - 1:
                        ax.set_xlabel(f"Cluster Representatives (n={n_clusters_found})", fontsize=8)
                    if j == 0:
                        ax.set_ylabel(f"Cluster Representatives (n={n_clusters_found})", fontsize=8)

                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Cosine Similarity", fontsize=8)
                else:
                    ax.set_title(
                        f"ms={min_samples}, mcs={min_cluster_size}, cse={cluster_selection_epsilon:.1f}\nNo clusters found",
                        fontsize=9,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

            else:
                param_str = (
                    f"ms={min_samples}, mcs={min_cluster_size}, cse={cluster_selection_epsilon:.1f}"
                )
                ax.set_title(f"{param_str}\nData unavailable", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(plots_dir, "structure_similarity_heatmaps.png"), dpi=300)
    plt.close()


def plot_pca_clusters(
    pca_file: str,
    labels_file: str,
    representative_frames: np.ndarray,
    hdbscan_params: Tuple[int, int, float],
    n_frames: int,
    num_components: int,
    output_dir: str,
):
    """
    Create scatter plot of the first two PCA components, colored by HDBSCAN cluster assignments.

    Parameters:
    -----------
    pca_file : str
        Path to file containing PCA coordinates
    labels_file : str
        Path to file containing cluster labels
    representative_frames : np.ndarray
        Indices of frames representing each cluster
    hdbscan_params : Tuple[int, int, float]
        HDBSCAN parameters (min_samples, min_cluster_size, cluster_selection_epsilon)
    n_frames : int
        Total number of frames
    num_components : int
        Number of PCA components
    output_dir : str
        Output directory for plots
    """
    min_samples, min_cluster_size, cluster_selection_epsilon = hdbscan_params
    param_str = f"ms{min_samples}_mcs{min_cluster_size}_cse{cluster_selection_epsilon:.1f}"

    logger.info(f"Creating PCA cluster plot for {param_str}...")

    plots_dir = os.path.join(output_dir, "plots", "pca_clusters")
    os.makedirs(plots_dir, exist_ok=True)

    # Load PCA coordinates
    pca_coords = np.memmap(pca_file, dtype=np.float32, mode="r", shape=(n_frames, num_components))

    # Extract the first two PCA components for plotting
    pca_to_plot = pca_coords[:, :2].copy()

    # Load cluster labels
    labels = np.load(labels_file)

    # Create a scatter plot
    plt.figure(figsize=(10, 8))

    # Define a color map with a special color for noise points
    cmap = plt.cm.tab20

    # First plot noise points in gray
    if -1 in labels:
        noise_mask = labels == -1
        plt.scatter(
            pca_to_plot[noise_mask, 0],
            pca_to_plot[noise_mask, 1],
            s=5,
            c="gray",
            alpha=0.5,
            label="Noise",
        )

    # Plot actual clusters with colors
    unique_clusters = sorted([l for l in np.unique(labels) if l != -1])
    n_clusters = len(unique_clusters)

    if n_clusters > 0:
        for i, cluster_id in enumerate(unique_clusters):
            mask = labels == cluster_id
            color = cmap(i % cmap.N)
            plt.scatter(
                pca_to_plot[mask, 0],
                pca_to_plot[mask, 1],
                s=10,
                c=[color],
                alpha=0.7,
                label=f"Cluster {cluster_id}",
            )

        # Add representative markers (medoids)
        if len(representative_frames) > 0:
            plt.scatter(
                pca_to_plot[representative_frames, 0],
                pca_to_plot[representative_frames, 1],
                s=100,
                facecolors="none",
                edgecolors="black",
                linewidths=1,
                alpha=0.8,
                marker="o",
                label="Medoids",
            )

    # Calculate percentage of frames assigned to clusters vs. noise
    n_total = len(labels)
    n_noise = np.sum(labels == -1)
    noise_percent = (n_noise / n_total) * 100 if n_total > 0 else 0

    plt.title(
        f"HDBSCAN Clustering: min_samples={min_samples}, "
        f"min_cluster_size={min_cluster_size}, "
        f"cse={cluster_selection_epsilon:.1f}\n"
        f"{n_clusters} clusters found, {noise_percent:.1f}% noise",
        fontsize=12,
    )
    plt.xlabel("PCA Component 1", fontsize=10)
    plt.ylabel("PCA Component 2", fontsize=10)

    # Add legend with reasonable size if there aren't too many clusters
    if n_clusters <= 20:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    else:
        # Just show legends for noise and medoids
        handles, labels = plt.gca().get_legend_handles_labels()
        filtered_handles = []
        filtered_labels = []

        # Keep noise and medoids in legend, plus a few sample clusters
        for i, label in enumerate(labels):
            if label == "Noise" or label == "Medoids" or (label.startswith("Cluster") and i < 25):
                filtered_handles.append(handles[i])
                filtered_labels.append(label)

        if filtered_handles:
            plt.legend(
                filtered_handles,
                filtered_labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"pca_clusters_{param_str}.png"), dpi=300)
    plt.close()

    # Clean up
    del pca_coords


# Define a dictionary of weight similarity functions
WEIGHTS_SIMILARITY_FUNCTIONS = {
    "hellinger_affinity": hellinger_affinity,
    "bhattacharyya_similarity": bhattacharyya_similarity,
    "laplacian_kernel_similarity": laplacian_kernel_similarity,
    "gaussian_kernel_similarity": gaussian_kernel_similarity,
}


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
            f"{topology_name}_hdbscan_reweighting_{timestamp}",
        )

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up temporary directory
    if args.temp_dir is None:
        args.temp_dir = tempfile.mkdtemp(prefix="hdbscan_reweighting_")
    else:
        os.makedirs(args.temp_dir, exist_ok=True)
    global logger
    logger = setup_logger(os.path.join(args.output_dir, "hdbscan_reweighting.log"))
    logger.info(f"Using temporary directory: {args.temp_dir}")

    # Set up logging
    log_file = os.path.join(args.output_dir, "hdbscan_reweighting.log")
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("Starting HDBSCAN cluster-reweighting experiment")
    logger.info("=" * 80)
    logger.info(f"Arguments: {vars(args)}")

    # Generate all combinations of HDBSCAN parameters
    hdbscan_params_list = []
    for min_samples in args.min_samples_values:
        for min_cluster_size in args.min_cluster_size_values:
            for cluster_selection_epsilon in args.cluster_selection_epsilon_values:
                hdbscan_params_list.append(
                    (min_samples, min_cluster_size, cluster_selection_epsilon)
                )

    logger.info(f"Generated {len(hdbscan_params_list)} HDBSCAN parameter combinations")
    for i, params in enumerate(hdbscan_params_list):
        min_samples, min_cluster_size, cluster_selection_epsilon = params
        logger.info(
            f"  Parameter set {i + 1}: min_samples={min_samples}, "
            f"min_cluster_size={min_cluster_size}, "
            f"cluster_selection_epsilon={cluster_selection_epsilon}"
        )

    # Use cutoff values directly from args (already a list of floats)
    cutoff_values = args.cutoff_values

    # Use power scales directly from args (already a list of floats)
    power_scales = args.power_scales

    # Process transformations
    transformations = args.transformations
    transformations = [
        None if transform.lower() == "none" else transform for transform in transformations
    ]

    logger.info(f"Using cutoff values (nm): {cutoff_values}")
    logger.info(f"Using power scales: {power_scales}")
    logger.info(f"Using transformations: {transformations}")

    # Generate configuration strings for plotting
    config_strings = []
    configurations = []

    for cutoff in cutoff_values:
        for power in power_scales:
            for transform in transformations:
                config = (cutoff, power, transform)
                configurations.append(config)

                # Create a label string
                label = f"cutoff={cutoff}nm, power={power}"
                if transform:
                    label += f", transform={transform}"
                config_strings.append(label)

    logger.info(f"Generated {len(configurations)} configurations")

    # Load universe
    logger.info(f"Loading topology from {args.topology_path}")
    logger.info(f"Loading trajectories: {args.trajectory_paths}")
    universe = mda.Universe(args.topology_path, *args.trajectory_paths)
    logger.info(
        f"Loaded universe with {len(universe.atoms)} atoms and {len(universe.trajectory)} frames"
    )

    n_frames = len(universe.trajectory)

    # Calculate pairwise distances and save to temporary file
    distances_file, n_distances = calculate_pairwise_distances_and_save(
        universe, args.atom_selection, args.chunk_size, args.temp_dir
    )

    # --- PCA coordinates file logic ---
    # Save PCA coords to script directory, named after trajectory file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    traj_base = os.path.splitext(os.path.basename(args.trajectory_paths[0]))[0]
    pca_file = os.path.join(script_dir, f"{traj_base}_pca_coords.npy")

    # Only perform PCA if file does not exist
    if not os.path.exists(pca_file):
        logger.info(
            f"PCA coordinates file {pca_file} not found. Performing PCA and saving results."
        )
        pca_file, pca = perform_pca_on_distances(
            distances_file, args.num_components, args.chunk_size, n_frames, n_distances, pca_file
        )
        # Save explained variance for reference
        np.save(
            os.path.join(args.output_dir, "pca_explained_variance_ratio.npy"),
            pca.explained_variance_ratio_,
        )
    else:
        logger.info(
            f"PCA coordinates file {pca_file} already exists. Loading existing PCA coordinates."
        )

        # Dummy PCA object for compatibility (not used if loading)
        class DummyPCA:
            explained_variance_ratio_ = np.array([])

        pca = DummyPCA()

    # For each weights similarity function
    for weights_similarity_label, weights_similarity_func in WEIGHTS_SIMILARITY_FUNCTIONS.items():
        logger.info(f"Processing with weights similarity function: {weights_similarity_label}")

        # Dictionaries to store results for this similarity function
        similarity_matrices = {}
        w1_distances = {}
        normalized_magnitudes = {}
        matrix_distances = {}

        # Metrics for matrix distance
        metrics = ["rmsd", "cosine", "w1", "correlation"]

        # Define the comparisons to plot
        comparisons_to_plot = [
            ("struct_vs_unW_struct", "Structure vs. Uniform Weighted Structure"),
            ("struct_vs_CW_struct", "Structure vs. Cluster Weighted Structure"),
            ("weights_uniform_vs_cluster", "Uniform Weights vs. Cluster Weights"),
            ("struct_vs_cluster_weights", "Structure vs. Cluster Weights"),
            ("struct_vs_uniform_weights", "Structure vs. Uniform Weights"),
        ]

        # Process each parameter combination
        for params in hdbscan_params_list:
            min_samples, min_cluster_size, cluster_selection_epsilon = params
            param_str = f"min_samples={min_samples}, min_cluster_size={min_cluster_size}, cluster_selection_epsilon={cluster_selection_epsilon:.1f}"
            logger.info(f"Processing HDBSCAN with {param_str}...")

            # Perform HDBSCAN clustering
            labels_file, n_clusters_found, n_noise_points = perform_hdbscan_clustering(
                pca_file,
                min_samples,
                min_cluster_size,
                cluster_selection_epsilon,
                n_frames,
                args.num_components,
                args.temp_dir,
            )

            # Only proceed if clusters were found
            if n_clusters_found > 0:
                # Identify representative frames (medoids) for the found clusters
                representative_frames = identify_hdbscan_representative_frames(
                    pca_file, labels_file, n_frames, args.num_components, args.chunk_size
                )

                # Create PCA plot showing clustering results
                plot_pca_clusters(
                    pca_file,
                    labels_file,
                    representative_frames,
                    params,  # This is the HDBSCAN params tuple (min_samples, min_cluster_size, cluster_selection_epsilon)
                    n_frames,
                    args.num_components,
                    args.output_dir,
                )
                # Save representative frames for reference
                param_filename = (
                    f"ms{min_samples}_mcs{min_cluster_size}_cse{cluster_selection_epsilon:.1f}"
                )
                np.save(
                    os.path.join(args.output_dir, f"representative_frames_{param_filename}.npy"),
                    representative_frames,
                )

                # Process each configuration only if representatives were found
                if len(representative_frames) > 0:
                    for i, config in enumerate(configurations):
                        config_string = config_strings[i]

                        # Calculate similarity matrices for representatives
                        sim_matrices = calculate_representative_similarity_matrices(
                            universe,
                            args.atom_selection,
                            distances_file,
                            n_distances,
                            representative_frames,
                            config,
                            config_string,
                            params,  # Pass HDBSCAN params tuple
                            args.temp_dir,
                            weights_similarity_func,
                            weights_similarity_label,
                        )

                        # Store similarity matrices if calculation was successful
                        if sim_matrices:
                            similarity_matrices[(config_string, params)] = sim_matrices

                            # Get references to matrices (check existence)
                            structure_similarity = sim_matrices.get("structure")
                            uniform_similarity = sim_matrices.get("uniform")
                            cluster_density_similarity = sim_matrices.get("cluster_density")
                            cluster_weights_similarity = sim_matrices.get(
                                "cluster_weights_similarity"
                            )
                            uniform_weights_similarity = sim_matrices.get(
                                "uniform_weights_similarity"
                            )

                            # Proceed only if essential matrices exist and are not empty
                            if (
                                structure_similarity is not None
                                and structure_similarity.size > 0
                                and uniform_weights_similarity is not None
                                and uniform_weights_similarity.size > 0
                                and cluster_weights_similarity is not None
                                and cluster_weights_similarity.size > 0
                            ):
                                # Extract upper triangle for computations
                                structure_triu = extract_upper_triangle(structure_similarity)
                                uniform_triu = (
                                    extract_upper_triangle(uniform_similarity)
                                    if uniform_similarity is not None
                                    else None
                                )
                                cluster_density_triu = (
                                    extract_upper_triangle(cluster_density_similarity)
                                    if cluster_density_similarity is not None
                                    else None
                                )
                                uniform_weights_triu = extract_upper_triangle(
                                    uniform_weights_similarity
                                )
                                cluster_weights_triu = extract_upper_triangle(
                                    cluster_weights_similarity
                                )

                                # Compute W1 distance between uniform and cluster density WEIGHT matrices
                                w1_distance = compute_w1_distance(
                                    uniform_weights_triu, cluster_weights_triu
                                )
                                w1_distances[(config_string, params)] = w1_distance

                                # Compute numel-normalized magnitude of structure similarity matrix
                                normalized_magnitude = compute_numel_normalized_magnitude(
                                    structure_triu
                                )
                                normalized_magnitudes[(config_string, params)] = (
                                    normalized_magnitude
                                )

                                # Compute matrix distances for defined comparisons
                                for metric in metrics:
                                    try:
                                        # Comparison 1: Structure vs Uniform Weighted Structure
                                        if uniform_triu is not None:
                                            matrix_distances[
                                                (
                                                    config_string,
                                                    params,
                                                    metric,
                                                    "struct_vs_unW_struct",
                                                )
                                            ] = compute_matrix_distance(
                                                structure_triu, uniform_triu, metric=metric
                                            )

                                        # Comparison 2: Structure vs Cluster Weighted Structure
                                        if cluster_density_triu is not None:
                                            matrix_distances[
                                                (
                                                    config_string,
                                                    params,
                                                    metric,
                                                    "struct_vs_CW_struct",
                                                )
                                            ] = compute_matrix_distance(
                                                structure_triu, cluster_density_triu, metric=metric
                                            )

                                        # Comparison 3: Uniform Weights vs Cluster Weights
                                        matrix_distances[
                                            (
                                                config_string,
                                                params,
                                                metric,
                                                "weights_uniform_vs_cluster",
                                            )
                                        ] = compute_matrix_distance(
                                            uniform_weights_triu,
                                            cluster_weights_triu,
                                            metric=metric,
                                        )
                                        # Comparison 4: Structure vs Cluster Weights
                                        matrix_distances[
                                            (
                                                config_string,
                                                params,
                                                metric,
                                                "struct_vs_cluster_weights",
                                            )
                                        ] = compute_matrix_distance(
                                            structure_triu, cluster_weights_triu, metric=metric
                                        )

                                        # Comparison 5: Structure vs Uniform Weights
                                        matrix_distances[
                                            (
                                                config_string,
                                                params,
                                                metric,
                                                "struct_vs_uniform_weights",
                                            )
                                        ] = compute_matrix_distance(
                                            structure_triu, uniform_weights_triu, metric=metric
                                        )
                                    except ValueError as e:
                                        logger.error(
                                            f"Error computing metric {metric} for params {param_str}, config={config_string}: {e}"
                                        )
                                        matrix_distances[
                                            (config_string, params, metric, "struct_vs_unW_struct")
                                        ] = np.nan
                                        matrix_distances[
                                            (config_string, params, metric, "struct_vs_CW_struct")
                                        ] = np.nan
                                        matrix_distances[
                                            (
                                                config_string,
                                                params,
                                                metric,
                                                "weights_uniform_vs_cluster",
                                            )
                                        ] = np.nan
                                        matrix_distances[
                                            (
                                                config_string,
                                                params,
                                                metric,
                                                "struct_vs_cluster_weights",
                                            )
                                        ] = np.nan
                                        matrix_distances[
                                            (
                                                config_string,
                                                params,
                                                metric,
                                                "struct_vs_uniform_weights",
                                            )
                                        ] = np.nan

                            else:
                                logger.error(
                                    f"Skipping metric calculations for params {param_str}, config={config_string} due to missing or empty similarity matrices."
                                )
                        else:
                            logger.error(
                                f"Similarity matrix calculation failed for params {param_str}, config={config_string}. Skipping metrics."
                            )

                else:
                    logger.error(
                        f"No representative frames found for params {param_str}. Skipping similarity calculations and metrics."
                    )
            else:
                logger.error(
                    f"No clusters found for params {param_str}. Skipping similarity calculations and metrics."
                )
                raise ValueError("No clusters found")

        # Create plots for this weights similarity function
        logger.info(f"Creating summary plots for {weights_similarity_label}...")

        plot_heatmaps(
            similarity_matrices,
            hdbscan_params_list,
            config_strings,
            args.output_dir,
            weights_similarity_label,
        )
        plot_w1_distance(
            w1_distances,
            similarity_matrices,
            hdbscan_params_list,
            config_strings,
            args.output_dir,
            weights_similarity_label,
        )
        plot_numel_normalized_magnitude(
            normalized_magnitudes,
            similarity_matrices,
            hdbscan_params_list,
            config_strings,
            args.output_dir,
            weights_similarity_label,
        )
        plot_matrix_distances(
            matrix_distances,
            similarity_matrices,
            hdbscan_params_list,
            config_strings,
            metrics,
            comparisons_to_plot,
            args.output_dir,
            weights_similarity_label,
        )

        # Save data for this weights similarity function
        data_dir = os.path.join(args.output_dir, "data", weights_similarity_label)
        os.makedirs(data_dir, exist_ok=True)

        # Save metrics data (convert complex dict keys to strings for saving)
        def convert_dict_keys_to_str(d):
            return {str(k): v for k, v in d.items()}

        np.savez(
            os.path.join(data_dir, "metrics.npz"),
            w1_distances=convert_dict_keys_to_str(w1_distances),
            normalized_magnitudes=convert_dict_keys_to_str(normalized_magnitudes),
            matrix_distances=convert_dict_keys_to_str(matrix_distances),
            hdbscan_params_list=np.array(hdbscan_params_list),
            config_strings=np.array(config_strings),
            comparisons=np.array(comparisons_to_plot),
        )

    # Clean up temporary directory if it was automatically created
    system_temp_dir = tempfile.gettempdir()
    if args.temp_dir.startswith(system_temp_dir):
        logger.info(f"Cleaning up temporary directory: {args.temp_dir}")
        try:
            shutil.rmtree(args.temp_dir)
        except OSError as e:
            logger.error(f"Error removing temporary directory {args.temp_dir}: {e}")
    else:
        logger.info(f"Temporary directory {args.temp_dir} was specified by user, not removing.")

    # End timer and report
    end_time = datetime.datetime.now()
    elapsed = end_time - start_time
    logger.info("=" * 80)
    logger.info(
        f"HDBSCAN cluster-reweighting experiment complete. Results saved to {args.output_dir}"
    )
    logger.info(f"Total execution time: {elapsed}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
