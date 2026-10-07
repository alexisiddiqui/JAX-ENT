"""Known-population state interfaces for the MoPrP covariance-recovery experiment.

Ground truth is a known NMR-derived state population (``state_ratios.json``).  Reference
weights spread each state's target mass *uniformly among its frames*; zero-target frames
stay available as decoys.  Target and predicted covariance are computed from the *same*
ensemble frames at different weights, so this is a symmetric variance/covariance matching
regime with zero loss at ``w == w_NMR`` by construction.

This module deliberately keeps the population support *strict and complete*: zero-target
states (PUF3, unfolded, PUF2-like) contribute to the Jensen-Shannon recovery and cannot be
renormalized away.  All numerical primitives are reused from
:mod:`jaxent.src.analysis.pf_variance` so the two covariance conventions stay consistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array
from jax.scipy.linalg import cho_solve
from jax.typing import ArrayLike

from jaxent.src.analysis.pf_variance import (
    framewise_uptake,
    jensen_shannon_divergence,
    jensen_shannon_recovery_percent,
    map_frame_log_pf_to_peptides,
    map_framewise_residue_uptake_to_peptides,
    overlap_projection,
    projected_log_euclidean_covariance_loss,
    shrink_covariance,
    trace_normalize_precision,
    weighted_population_covariance,
)

# Cluster-id -> state name mapping (config.yaml ``recovery.state_mapping``).
DEFAULT_STATE_MAPPING: dict[int, str] = {
    0: "Folded",
    1: "PUF1",
    2: "PUF2",
    3: "PUF3",
    4: "unfolded",
    5: "PUF2-like",
}

# The complete, ordered population support used for strict JSD recovery.
FULL_STATE_SUPPORT: tuple[str, ...] = (
    "Folded",
    "PUF1",
    "PUF2",
    "PUF3",
    "unfolded",
    "PUF2-like",
)

# States that carry non-zero target mass in ``state_ratios.json``.
TARGET_STATES: tuple[str, ...] = ("Folded", "PUF1", "PUF2")


# --------------------------------------------------------------------------------------
# Targets and frame state labels
# --------------------------------------------------------------------------------------
def load_state_targets(
    state_ratios_json: str | Path,
    support: Sequence[str] = FULL_STATE_SUPPORT,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the ordered support and its normalized full-support target distribution.

    Only Folded/PUF1/PUF2 carry mass; every other state in ``support`` is an explicit
    zero-target decoy state.
    """

    data = json.loads(Path(state_ratios_json).read_text())
    fractions = data["fractional_populations"]
    raw = {
        "Folded": float(fractions["folded"]["fraction"]),
        "PUF1": float(fractions["PUF1"]["fraction"]),
        "PUF2": float(fractions["PUF2"]["fraction"]),
    }
    support = tuple(support)
    targets = np.array([raw.get(state, 0.0) for state in support], dtype=np.float64)
    total = targets.sum()
    if total <= 0.0:
        raise ValueError("state targets sum to zero")
    return support, targets / total


def load_frame_states(
    cluster_assignments_csv: str | Path,
    ensemble_name: str,
    state_mapping: Mapping[int, str] = DEFAULT_STATE_MAPPING,
    expected_frames: int | None = None,
) -> np.ndarray:
    """Return the per-frame state-name vector for one ensemble, in trajectory order.

    Frames are ordered by ascending ``global_frame_index`` within the ensemble.  Raises if
    any cluster label is not present in ``state_mapping`` (every frame must have an explicit
    target state, including zero-target states) or if the count disagrees with
    ``expected_frames``.
    """

    frame_table = pd.read_csv(cluster_assignments_csv)
    ensemble = frame_table[frame_table["ensemble_name"] == ensemble_name]
    if ensemble.empty:
        raise ValueError(f"ensemble {ensemble_name!r} not present in {cluster_assignments_csv}")
    ensemble = ensemble.sort_values("global_frame_index")
    labels = ensemble["cluster_label"].to_numpy(dtype=int)

    global_index = ensemble["global_frame_index"].to_numpy(dtype=int)
    if not np.array_equal(global_index, np.arange(global_index[0], global_index[0] + labels.size)):
        raise ValueError(
            f"ensemble {ensemble_name!r} frames are not a contiguous global-index block; "
            "cannot assume trajectory order matches the feature file"
        )

    unmapped = sorted(set(int(label) for label in labels) - set(state_mapping))
    if unmapped:
        raise ValueError(f"cluster labels {unmapped} have no state in state_mapping")

    if expected_frames is not None and labels.size != expected_frames:
        raise ValueError(
            f"ensemble {ensemble_name!r} has {labels.size} frames, expected {expected_frames}"
        )

    return np.array([state_mapping[int(label)] for label in labels], dtype=object)


def _state_membership_matrix(states: np.ndarray, support: Sequence[str]) -> np.ndarray:
    """Return a ``(n_states, n_frames)`` float membership matrix."""

    states = np.asarray(states)
    return np.stack([(states == state).astype(np.float64) for state in support])


def reference_weights_from_states(
    states: np.ndarray,
    support: Sequence[str],
    targets: ArrayLike,
) -> np.ndarray:
    """Build ``w_NMR``: each state's target mass spread uniformly over its frames.

    Zero-target states keep zero weight but their frames remain in the ensemble as decoys.
    Raises if a state carrying positive target mass has no frames.
    """

    states = np.asarray(states)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.zeros(states.size, dtype=np.float64)
    for state, mass in zip(support, targets):
        mask = states == state
        count = int(mask.sum())
        if mass > 0.0 and count == 0:
            raise ValueError(f"state {state!r} carries target mass {mass} but has no frames")
        if count > 0 and mass > 0.0:
            weights[mask] = mass / count
    return weights


def state_populations(
    weights: ArrayLike,
    states: np.ndarray,
    support: Sequence[str],
) -> Array:
    """Return the weighted mass in every support state (differentiable in ``weights``)."""

    membership = jnp.asarray(_state_membership_matrix(states, support))
    weights = jnp.asarray(weights)
    return membership @ weights


def strict_population_jsd(
    weights: ArrayLike,
    states: np.ndarray,
    support: Sequence[str],
    targets: ArrayLike,
) -> Array:
    """Full-support base-2 JSD between recovered and target populations."""

    populations = state_populations(weights, states, support)
    return jensen_shannon_divergence(populations, jnp.asarray(targets))


def strict_recovery_percent(
    weights: ArrayLike,
    states: np.ndarray,
    support: Sequence[str],
    targets: ArrayLike,
) -> Array:
    """Full-support recovery percent ``100 * (1 - sqrt(JSD))``."""

    populations = state_populations(weights, states, support)
    return jensen_shannon_recovery_percent(populations, jnp.asarray(targets))


# --------------------------------------------------------------------------------------
# Covariance coordinate constructors (both from the same ensemble frames)
# --------------------------------------------------------------------------------------
def peptide_logpf_covariance(
    log_pf_by_frame: ArrayLike,
    mapping: ArrayLike,
    weights: ArrayLike,
) -> Array:
    """Weighted peptide log-PF covariance ``M C_logPF Mᵀ`` over frames -> ``(P, P)``."""

    peptide_log_pf = map_frame_log_pf_to_peptides(mapping, log_pf_by_frame)
    return weighted_population_covariance(peptide_log_pf, weights)


def peptide_uptake_covariances(
    log_pf_by_frame: ArrayLike,
    k_ints: ArrayLike,
    timepoints: ArrayLike,
    mapping: ArrayLike,
    weights: ArrayLike,
) -> Array:
    """Timepoint-specific weighted peptide-uptake covariance -> ``(T, P, P)``."""

    residue_uptake = framewise_uptake(log_pf_by_frame, k_ints, timepoints)
    peptide_uptake = map_framewise_residue_uptake_to_peptides(mapping, residue_uptake)
    return jax.vmap(lambda block: weighted_population_covariance(block, weights))(peptide_uptake)


def marginal_profile(covariance: ArrayLike, alpha: float | ArrayLike = 0.05) -> Array:
    """Shrunk marginal variance ``diag(C)``."""

    return jnp.diag(shrink_covariance(covariance, alpha=alpha))


def shrunk_trace_normalized_precision(
    covariance: ArrayLike, alpha: float | ArrayLike = 0.05
) -> Array:
    """Return a finite, shrunk, trace-normalized precision (the dynamic-geometry weight)."""

    regularized = shrink_covariance(covariance, alpha=alpha)
    factor = jnp.linalg.cholesky(regularized)
    precision = cho_solve((factor, True), jnp.eye(regularized.shape[0], dtype=regularized.dtype))
    return trace_normalize_precision(precision)


def log_ratio_profile_loss(predicted: ArrayLike, target: ArrayLike) -> Array:
    """Symmetric, dimensionless squared log-ratio distance between variance profiles."""

    predicted = jnp.asarray(predicted)
    target = jnp.asarray(target)
    return jnp.mean(jnp.square(jnp.log(predicted / target)))


def projected_full_covariance_loss(
    predicted_covariance: ArrayLike,
    target_covariance: ArrayLike,
    mapping: ArrayLike,
    alpha: float | ArrayLike = 0.05,
    relative_threshold: float = 1e-6,
) -> Array:
    """Overlap-projected symmetric log-Euclidean covariance distance."""

    projection = overlap_projection(mapping, relative_threshold=relative_threshold)
    return projected_log_euclidean_covariance_loss(
        predicted_covariance, target_covariance, projection, alpha=alpha
    )


def correlation_of(covariance: ArrayLike) -> Array:
    """Return the correlation matrix (scale-free *shape*) of a covariance."""

    covariance = jnp.asarray(covariance)
    scale = jnp.sqrt(jnp.clip(jnp.diag(covariance), 1e-12, None))
    return covariance / jnp.outer(scale, scale)


def correlation_shape_loss(
    covariance: ArrayLike,
    prior_correlation: ArrayLike,
    projection: ArrayLike,
    alpha: float | ArrayLike = 0.05,
) -> Array:
    """Scale-free covariance-*shape* regulariser.

    Overlap-projected log-Euclidean distance between the correlation (shape) of ``covariance`` and a
    fixed ``prior_correlation``.  Zero when the shapes match; invariant to the magnitude of
    ``covariance``.  The ``projection`` is precomputed (outside any jit) to keep this
    differentiable.
    """

    return projected_log_euclidean_covariance_loss(
        correlation_of(covariance), prior_correlation, projection, alpha
    )
