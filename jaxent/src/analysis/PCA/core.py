"""
jaxent.src.analysis.PCA.core
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PCA computation utilities for single- and multi-trajectory workflows.

Functions
---------
calculate_pairwise_rmsd
    Compute per-frame pairwise Euclidean distances for a Universe.
perform_pca_on_distances
    Fit/transform IncrementalPCA on a pre-computed distance matrix.
calculate_distances_and_perform_pca
    Single-trajectory entry point used by kCluster (unchanged signature).
calculate_multi_traj_pca
    Multi-trajectory entry point used by iPCA.
"""

from __future__ import annotations

import logging

import MDAnalysis as mda
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level building blocks
# ---------------------------------------------------------------------------


def calculate_pairwise_rmsd(
    universe: mda.Universe,
    selection: str,
    chunk_size: int = 100,
) -> np.ndarray:
    """Calculate intra-frame pairwise Euclidean distances for every trajectory frame.

    Parameters
    ----------
    universe:
        MDAnalysis Universe containing the trajectory.
    selection:
        MDAnalysis atom-selection string (e.g. ``"name CA"``).
    chunk_size:
        Number of frames to process per iteration (controls memory usage).

    Returns
    -------
    np.ndarray
        Shape ``(n_frames, n_features)`` where ``n_features = n_atoms*(n_atoms-1)//2``.
    """
    atoms = universe.select_atoms(selection)
    n_frames = len(universe.trajectory)
    n_atoms = atoms.n_atoms
    n_distances = n_atoms * (n_atoms - 1) // 2

    logger.info(
        "calculate_pairwise_rmsd: %d atoms × %d frames → %d features",
        n_atoms,
        n_frames,
        n_distances,
    )

    all_distances = np.zeros((n_frames, n_distances), dtype=np.float32)
    for i, _ts in enumerate(tqdm(universe.trajectory, desc="Pairwise distances")):
        all_distances[i] = pdist(atoms.positions, metric="euclidean")

    logger.info("calculate_pairwise_rmsd: done — shape %s", all_distances.shape)
    return all_distances


def perform_pca_on_distances(
    distances: np.ndarray,
    n_components: int,
    chunk_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, IncrementalPCA]:
    """Fit and apply IncrementalPCA to a pre-computed distance matrix.

    Parameters
    ----------
    distances:
        Shape ``(n_frames, n_features)`` distance matrix.
    n_components:
        Number of PCA components to retain.
    chunk_size:
        Batch size fed to ``IncrementalPCA.partial_fit``.

    Returns
    -------
    pca_coords : np.ndarray
        Shape ``(n_frames, n_components)``.
    explained_variance : np.ndarray
        Explained variance ratios, shape ``(n_components,)``.
    pca : IncrementalPCA
        The fitted scikit-learn estimator.
    """
    ipca_batch_size = max(n_components * 10, chunk_size)
    pca = IncrementalPCA(n_components=n_components, batch_size=ipca_batch_size)

    n_frames = distances.shape[0]
    pca_coords = np.zeros((n_frames, n_components), dtype=np.float32)

    for start in tqdm(range(0, n_frames, chunk_size), desc="IncrementalPCA"):
        end = min(start + chunk_size, n_frames)
        batch = distances[start:end]
        if start == 0:
            pca_coords[start:end] = pca.fit_transform(batch)
        else:
            pca.partial_fit(batch)
            pca_coords[start:end] = pca.transform(batch)

    logger.info(
        "perform_pca_on_distances: total variance explained = %.2f%%",
        pca.explained_variance_ratio_.sum() * 100,
    )
    return pca_coords, pca.explained_variance_ratio_, pca


# ---------------------------------------------------------------------------
# Single-trajectory entry point (backward-compatible with kCluster)
# ---------------------------------------------------------------------------


def calculate_distances_and_perform_pca(
    universe: mda.Universe,
    selection: str,
    num_components: int,
    chunk_size: int = 100,
) -> tuple[np.ndarray, IncrementalPCA]:
    """Single-trajectory PCA — drop-in replacement for the kCluster inline version.

    Parameters
    ----------
    universe:
        MDAnalysis Universe.
    selection:
        Atom-selection string.
    num_components:
        Number of PCA components.
    chunk_size:
        Chunk size for memory-efficient processing.

    Returns
    -------
    pca_coords : np.ndarray
        Shape ``(n_frames, num_components)``.
    pca : IncrementalPCA
        Fitted estimator (carries ``explained_variance_ratio_``).
    """
    distances = calculate_pairwise_rmsd(universe, selection, chunk_size)
    pca_coords, _ev, pca = perform_pca_on_distances(
        distances, num_components, chunk_size
    )
    return pca_coords, pca


# ---------------------------------------------------------------------------
# Multi-trajectory entry point (iPCA)
# ---------------------------------------------------------------------------


def calculate_multi_traj_pca(
    universes: list[mda.Universe],
    atom_selection: str,
    n_components: int = 10,
    chunk_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a joint PCA across multiple trajectories.

    All trajectories must yield the same number of pairwise features (i.e. the
    atom count from *atom_selection* must match across all topologies).

    Parameters
    ----------
    universes:
        List of MDAnalysis Universe objects, one per (topology, trajectory) pair.
    atom_selection:
        MDAnalysis selection string applied identically to every Universe.
    n_components:
        Number of PCA components.
    chunk_size:
        Frames per IncrementalPCA chunk.

    Returns
    -------
    pca_coords : np.ndarray
        Shape ``(total_frames, n_components)`` — all trajectories in stack order.
    explained_variance : np.ndarray
        Explained variance ratios, shape ``(n_components,)``.
    metadata : dict
        Keys: ``"traj_idx"`` (int), ``"frame_idx"`` (int) — parallel to *pca_coords* rows.
        Caller is responsible for attaching condition/replicate labels.
    """
    logger.info("calculate_multi_traj_pca: processing %d universes", len(universes))

    # ------------------------------------------------------------------
    # 1. Compute per-trajectory distance matrices & validate feature dims
    # ------------------------------------------------------------------
    all_distance_chunks: list[np.ndarray] = []
    traj_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []
    reference_n_features: int | None = None

    for traj_idx, u in enumerate(universes):
        dist = calculate_pairwise_rmsd(u, atom_selection, chunk_size)
        n_frames, n_features = dist.shape

        if reference_n_features is None:
            reference_n_features = n_features
        elif n_features != reference_n_features:
            raise ValueError(
                f"Trajectory {traj_idx} has {n_features} pairwise features but "
                f"trajectory 0 has {reference_n_features}. "
                "All topologies must select the same number of atoms."
            )

        all_distance_chunks.append(dist)
        traj_indices.append(np.full(n_frames, traj_idx, dtype=np.int32))
        frame_indices.append(np.arange(n_frames, dtype=np.int32))

    # ------------------------------------------------------------------
    # 2. Stack into one combined matrix
    # ------------------------------------------------------------------
    combined_distances = np.vstack(all_distance_chunks)  # (total_frames, n_features)
    combined_traj_idx = np.concatenate(traj_indices)
    combined_frame_idx = np.concatenate(frame_indices)

    logger.info(
        "calculate_multi_traj_pca: combined matrix shape = %s", combined_distances.shape
    )

    # ------------------------------------------------------------------
    # 3. Fit joint IncrementalPCA
    # ------------------------------------------------------------------
    pca_coords, explained_variance, _pca = perform_pca_on_distances(
        combined_distances, n_components, chunk_size
    )

    metadata: dict = {
        "traj_idx": combined_traj_idx,
        "frame_idx": combined_frame_idx,
    }

    return pca_coords, explained_variance, metadata
