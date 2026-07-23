"""Scale-free joint covariance/variance geometry and transferable linear priors.

The objects in this module are deliberately independent of the production HDX loss.  They
support the ISO -> MoPrP investigation in which a population-free prior is trained on ISO,
serialized, and then held fixed while MoPrP frame weights and BV coefficients are fitted.

Two invariances are central:

* multiplying a covariance by a positive scalar must not change the geometry;
* changing the orthonormal basis of the retained peptide-overlap subspace must not change
  the full-geometry distance.

The first is obtained by removing the trace of the matrix logarithm.  The second follows
from using the Frobenius metric on that trace-free symmetric matrix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from jaxent.src.analysis.pf_variance import shrink_covariance, weighted_population_covariance


ARTIFACT_VERSION = 1


def _safe_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    correlation = covariance / np.outer(scale, scale)
    return np.clip(correlation, -1.0, 1.0)


def build_residue_pair_features(
    *,
    residue_ids: ArrayLike,
    k_ints: ArrayLike,
    heavy_contacts: ArrayLike,
    acceptor_contacts: ArrayLike,
    coordinates: ArrayLike,
    residue_names: Sequence[str],
    anm_covariance: ArrayLike,
    reference_bc: float = 0.35,
    reference_bh: float = 2.0,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the fixed symmetric residue/pair feature schema used for ISO transfer.

    Rows follow ``np.triu_indices(n_residues)``.  Every pair construction is symmetric in
    its endpoints, so predicted matrices are equivariant to residue permutation before
    explicit symmetrization.
    """

    residue_ids = np.asarray(residue_ids, dtype=float)
    k_ints = np.asarray(k_ints, dtype=float)
    heavy = np.asarray(heavy_contacts, dtype=float)
    acceptor = np.asarray(acceptor_contacts, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    anm = np.asarray(anm_covariance, dtype=float)
    n_residues = residue_ids.size
    if heavy.shape[0] != n_residues or acceptor.shape != heavy.shape:
        raise ValueError("contact features are not aligned to residue ids")
    if k_ints.shape != (n_residues,) or coordinates.shape != (n_residues, 3):
        raise ValueError("rates or coordinates are not aligned to residue ids")
    if anm.shape != (n_residues, n_residues) or len(residue_names) != n_residues:
        raise ValueError("ANM covariance or residue names are not aligned")

    def population_covariance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        return (ac @ bc.T) / a.shape[1]

    hh = population_covariance(heavy, heavy)
    aa = population_covariance(acceptor, acceptor)
    ha = population_covariance(heavy, acceptor)
    z = reference_bc * heavy + reference_bh * acceptor
    zz = population_covariance(z, z)
    within_ha = np.diag(ha) / np.sqrt(
        np.clip(np.diag(hh) * np.diag(aa), 1e-12, None)
    )
    position_range = max(float(np.ptp(residue_ids)), 1.0)
    node_names = (
        "position",
        "log_kint",
        "mean_heavy",
        "mean_acceptor",
        "logvar_heavy",
        "logvar_acceptor",
        "heavy_acceptor_corr",
        "mean_reference_logpf",
        "logvar_reference_logpf",
    )
    node = np.column_stack(
        (
            (residue_ids - residue_ids.min()) / position_range,
            np.log(np.clip(k_ints, 1e-12, None)),
            heavy.mean(axis=1),
            acceptor.mean(axis=1),
            np.log(np.clip(np.diag(hh), 1e-12, None)),
            np.log(np.clip(np.diag(aa), 1e-12, None)),
            within_ha,
            z.mean(axis=1),
            np.log(np.clip(np.diag(zz), 1e-12, None)),
        )
    )
    amino_acids = (
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    )
    aa_one_hot = np.asarray(
        [[float(str(name).upper() == amino_acid) for amino_acid in amino_acids] for name in residue_names]
    )
    rows, cols = np.triu_indices(n_residues)
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=-1)
    anm_correlation = _safe_correlation(anm)
    logpf_correlation = _safe_correlation(zz)

    columns = [
        np.ones(rows.size),
        (rows == cols).astype(float),
        np.log1p(np.abs(residue_ids[rows] - residue_ids[cols])),
        distances[rows, cols],
        anm_correlation[rows, cols],
        logpf_correlation[rows, cols],
    ]
    names = ["constant", "diagonal", "log_sequence_separation", "ca_distance", "anm_correlation", "unweighted_logpf_correlation"]
    for operation, values in (
        ("sum", node[rows] + node[cols]),
        ("absdiff", np.abs(node[rows] - node[cols])),
        ("product", node[rows] * node[cols]),
    ):
        columns.extend(values[:, index] for index in range(values.shape[1]))
        names.extend(f"{operation}_{name}" for name in node_names)
    aa_counts = aa_one_hot[rows] + aa_one_hot[cols]
    columns.extend(aa_counts[:, index] for index in range(aa_counts.shape[1]))
    names.extend(f"aa_count_{name}" for name in amino_acids)
    columns.append(np.asarray([float(residue_names[i] == residue_names[j]) for i, j in zip(rows, cols, strict=True)]))
    names.append("same_amino_acid")
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("constructed residue/pair features contain non-finite values")
    return features, tuple(names)


def reference_population_geometry(
    heavy_contacts: ArrayLike,
    acceptor_contacts: ArrayLike,
    weights: ArrayLike,
    *,
    bc: float = 0.35,
    bh: float = 2.0,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return one known-population residue geometry at the frozen BV reference."""

    log_pf = bc * jnp.asarray(heavy_contacts) + bh * jnp.asarray(acceptor_contacts)
    covariance = weighted_population_covariance(log_pf, jnp.asarray(weights))
    return np.asarray(scale_free_log_spd(covariance, alpha=alpha))


def symmetric_matrix_log(matrix: ArrayLike) -> Array:
    """Return the principal matrix logarithm of a symmetric positive-definite matrix."""

    matrix = 0.5 * (jnp.asarray(matrix) + jnp.asarray(matrix).T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    tiny = jnp.finfo(matrix.dtype).tiny
    eigenvalues = jnp.maximum(eigenvalues, tiny)
    return (eigenvectors * jnp.log(eigenvalues)[None, :]) @ eigenvectors.T


def symmetric_matrix_exp(matrix: ArrayLike) -> Array:
    """Return the positive-definite exponential of a symmetric matrix."""

    matrix = 0.5 * (jnp.asarray(matrix) + jnp.asarray(matrix).T)
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return (eigenvectors * jnp.exp(eigenvalues)[None, :]) @ eigenvectors.T


def trace_free(matrix: ArrayLike) -> Array:
    """Remove the scalar identity component from a square matrix."""

    matrix = jnp.asarray(matrix)
    dimension = matrix.shape[0]
    return matrix - (jnp.trace(matrix) / dimension) * jnp.eye(dimension, dtype=matrix.dtype)


def scale_free_log_spd(
    covariance: ArrayLike,
    projection: ArrayLike | None = None,
    alpha: float | ArrayLike = 0.05,
) -> Array:
    """Return overlap-projected, shrunk, trace-free log-SPD covariance geometry.

    Projection precedes shrinkage, matching ``projected_log_euclidean_covariance_loss``.
    Removing ``trace(log(C))/d * I`` makes the result invariant to ``C -> s*C`` for
    every positive scalar ``s`` (up to the absolute numerical floor in shrinkage).
    """

    covariance = jnp.asarray(covariance)
    if projection is not None:
        projection = jnp.asarray(projection)
        covariance = projection.T @ covariance @ projection
    return trace_free(symmetric_matrix_log(shrink_covariance(covariance, alpha=alpha)))


def relative_log_variance(
    covariance: ArrayLike,
    weights: ArrayLike | None = None,
    alpha: float | ArrayLike = 0.05,
) -> Array:
    """Return centered log marginal variances in the physical coordinate system.

    ``weights`` may contain inverse-overlap-degree weights.  Centering uses the same
    weights, and therefore removes exactly one global log-variance scale.
    """

    regularized = shrink_covariance(jnp.asarray(covariance), alpha=alpha)
    log_variance = jnp.log(jnp.maximum(jnp.diag(regularized), jnp.finfo(regularized.dtype).tiny))
    if weights is None:
        weights = jnp.ones_like(log_variance)
    weights = jnp.asarray(weights)
    mean = jnp.sum(weights * log_variance) / jnp.sum(weights)
    return log_variance - mean


def weighted_mean_square(values: ArrayLike, weights: ArrayLike | None = None) -> Array:
    values = jnp.asarray(values)
    if weights is None:
        return jnp.mean(jnp.square(values))
    weights = jnp.asarray(weights)
    return jnp.sum(weights * jnp.square(values)) / jnp.sum(weights)


def fixed_joint_geometry_loss(
    candidate_covariance: ArrayLike,
    *,
    prior_geometry: ArrayLike,
    prior_relative_variance: ArrayLike,
    projection: ArrayLike,
    marginal_weights: ArrayLike | None = None,
    marginal_strength: float | ArrayLike = 1.0,
    alpha: float | ArrayLike = 0.05,
) -> tuple[Array, Array, Array]:
    """Return total, full-SPD, and relative-marginal losses to one fixed prior."""

    geometry = scale_free_log_spd(candidate_covariance, projection, alpha)
    relative = relative_log_variance(candidate_covariance, marginal_weights, alpha)
    geometry_loss = jnp.mean(jnp.square(geometry - jnp.asarray(prior_geometry)))
    marginal_loss = weighted_mean_square(
        relative - jnp.asarray(prior_relative_variance), marginal_weights
    )
    return geometry_loss + marginal_strength * marginal_loss, geometry_loss, marginal_loss


def family_joint_geometry_loss(
    candidate_covariance: ArrayLike,
    scores: ArrayLike,
    *,
    prior_geometry: ArrayLike,
    geometry_modes: ArrayLike,
    prior_relative_variance: ArrayLike,
    marginal_modes: ArrayLike,
    score_precision: ArrayLike,
    projection: ArrayLike,
    marginal_weights: ArrayLike | None = None,
    marginal_strength: float | ArrayLike = 1.0,
    score_strength: float | ArrayLike = 1.0,
    alpha: float | ArrayLike = 0.05,
) -> tuple[Array, Array, Array, Array]:
    """Return the penalized loss to a fixed affine family of prior geometries."""

    scores = jnp.asarray(scores)
    geometry_target = jnp.asarray(prior_geometry) + jnp.einsum(
        "k,kij->ij", scores, jnp.asarray(geometry_modes)
    )
    marginal_target = jnp.asarray(prior_relative_variance) + jnp.einsum(
        "k,ki->i", scores, jnp.asarray(marginal_modes)
    )
    _, geometry_loss, marginal_loss = fixed_joint_geometry_loss(
        candidate_covariance,
        prior_geometry=geometry_target,
        prior_relative_variance=marginal_target,
        projection=projection,
        marginal_weights=marginal_weights,
        marginal_strength=marginal_strength,
        alpha=alpha,
    )
    precision = jnp.asarray(score_precision)
    score_loss = (
        scores @ precision @ scores / jnp.maximum(scores.size, 1)
        if scores.size
        else jnp.asarray(0.0, dtype=geometry_loss.dtype)
    )
    total = geometry_loss + marginal_strength * marginal_loss + score_strength * score_loss
    return total, geometry_loss, marginal_loss, score_loss


def peptide_tangent_prior(
    residue_center: ArrayLike,
    residue_modes: ArrayLike,
    mapping: ArrayLike,
    projection: ArrayLike,
    marginal_weights: ArrayLike | None = None,
    alpha: float | ArrayLike = 0.05,
) -> tuple[Array, Array, Array, Array]:
    """Push a residue log-SPD affine family into peptide geometry by exact JVPs.

    The returned arrays are a peptide-geometry center/modes and a physical peptide
    relative-marginal center/modes.  Expensive residue matrix exponentials occur only while
    building this frozen prior, never in the frame-reweighting objective.
    """

    residue_center = jnp.asarray(residue_center)
    residue_modes = jnp.asarray(residue_modes)
    mapping = jnp.asarray(mapping)
    projection = jnp.asarray(projection)

    def transform(log_geometry: Array) -> tuple[Array, Array]:
        residue_covariance = symmetric_matrix_exp(trace_free(log_geometry))
        peptide_covariance = mapping @ residue_covariance @ mapping.T
        return (
            scale_free_log_spd(peptide_covariance, projection, alpha),
            relative_log_variance(peptide_covariance, marginal_weights, alpha),
        )

    center_geometry, center_marginal = transform(residue_center)
    if residue_modes.shape[0] == 0:
        return (
            center_geometry,
            jnp.zeros((0, *center_geometry.shape), dtype=center_geometry.dtype),
            center_marginal,
            jnp.zeros((0, center_marginal.size), dtype=center_marginal.dtype),
        )

    pushed = [jax.jvp(transform, (residue_center,), (mode,))[1] for mode in residue_modes]
    geometry_modes = jnp.stack([item[0] for item in pushed])
    marginal_modes = jnp.stack([item[1] for item in pushed])
    return center_geometry, geometry_modes, center_marginal, marginal_modes


def upper_triangle_features(matrix: ArrayLike) -> np.ndarray:
    """Vectorize a symmetric matrix with sqrt(2) off-diagonal Frobenius weighting."""

    matrix = np.asarray(matrix, dtype=float)
    rows, cols = np.triu_indices(matrix.shape[0])
    values = matrix[rows, cols].copy()
    values[rows != cols] *= np.sqrt(2.0)
    return values


def from_upper_triangle(values: ArrayLike, dimension: int) -> np.ndarray:
    """Inverse of :func:`upper_triangle_features` for symmetric matrices."""

    values = np.asarray(values, dtype=float).copy()
    rows, cols = np.triu_indices(dimension)
    values[rows != cols] /= np.sqrt(2.0)
    matrix = np.zeros((dimension, dimension), dtype=float)
    matrix[rows, cols] = values
    matrix[cols, rows] = values
    return matrix


@dataclass(frozen=True)
class LinearGeometryModel:
    """A transferable point or low-rank family model in residue log-SPD space."""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    center_intercept: float
    center_coefficients: np.ndarray
    mode_intercepts: np.ndarray
    mode_coefficients: np.ndarray
    score_precision: np.ndarray
    ridge: float
    rank: int

    def predict(self, pair_features: ArrayLike, dimension: int) -> tuple[np.ndarray, np.ndarray]:
        pair_features = np.asarray(pair_features, dtype=float)
        standardized = (pair_features - self.feature_mean) / self.feature_scale
        center_vector = self.center_intercept + standardized @ self.center_coefficients
        center = trace_free(from_upper_triangle(center_vector, dimension))
        center = np.asarray(center, dtype=float)
        modes = []
        for intercept, coefficients in zip(
            self.mode_intercepts, self.mode_coefficients, strict=True
        ):
            vector = intercept + standardized @ coefficients
            modes.append(np.asarray(trace_free(from_upper_triangle(vector, dimension))))
        return center, np.stack(modes) if modes else np.zeros((0, dimension, dimension))


def _ridge_fit(
    features: np.ndarray, target: np.ndarray, ridge: float
) -> tuple[float, np.ndarray]:
    """Fit standardized ridge with an unpenalized intercept."""

    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    design = np.column_stack((np.ones(features.shape[0]), features))
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return float(solution[0]), solution[1:]


def fit_linear_geometry_model(
    pair_feature_views: Sequence[np.ndarray],
    feature_names: Sequence[str],
    target_geometries: ArrayLike,
    *,
    rank: int,
    ridge: float,
) -> tuple[LinearGeometryModel, np.ndarray]:
    """Fit an SVD factor model followed by transferable ridge loading regressions.

    ``target_geometries`` is ``(n_populations, n_residues, n_residues)``.  Every feature
    view (e.g. ISO_BI and ISO_TRI) is trained against the same physical target center and
    modes, so the learned mapping is encouraged to ignore decoy-dependent unweighted
    features.  Returned population scores are whitened to unit sample variance.
    """

    targets = np.asarray(target_geometries, dtype=float)
    if targets.ndim != 3 or targets.shape[1] != targets.shape[2]:
        raise ValueError("target_geometries must be (population, residue, residue)")
    if not 0 <= rank < targets.shape[0]:
        raise ValueError("rank must be non-negative and smaller than population count")
    vectors = np.stack([upper_triangle_features(matrix) for matrix in targets])
    center_vector = vectors.mean(axis=0)
    residual = vectors - center_vector
    if rank:
        u, singular, vt = np.linalg.svd(residual, full_matrices=False)
        scores = u[:, :rank] * np.sqrt(max(targets.shape[0] - 1, 1))
        mode_vectors = (
            singular[:rank, None] / np.sqrt(max(targets.shape[0] - 1, 1))
        ) * vt[:rank]
    else:
        scores = np.zeros((targets.shape[0], 0))
        mode_vectors = np.zeros((0, vectors.shape[1]))

    views = [np.asarray(view, dtype=float) for view in pair_feature_views]
    expected_rows = vectors.shape[1]
    if any(view.shape != (expected_rows, len(feature_names)) for view in views):
        raise ValueError("pair feature views do not match target upper triangle or schema")
    all_features = np.concatenate(views, axis=0)
    feature_mean = all_features.mean(axis=0)
    feature_scale = all_features.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    standardized = (all_features - feature_mean) / feature_scale
    duplicated_center = np.tile(center_vector, len(views))
    center_intercept, center_coefficients = _ridge_fit(
        standardized, duplicated_center, ridge
    )
    mode_intercepts, mode_coefficients = [], []
    for vector in mode_vectors:
        intercept, coefficients = _ridge_fit(standardized, np.tile(vector, len(views)), ridge)
        mode_intercepts.append(intercept)
        mode_coefficients.append(coefficients)

    model = LinearGeometryModel(
        feature_names=tuple(feature_names),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        center_intercept=center_intercept,
        center_coefficients=center_coefficients,
        mode_intercepts=np.asarray(mode_intercepts, dtype=float),
        mode_coefficients=(
            np.asarray(mode_coefficients, dtype=float).reshape(rank, standardized.shape[1])
            if rank
            else np.zeros((0, standardized.shape[1]), dtype=float)
        ),
        score_precision=np.eye(rank, dtype=float),
        ridge=float(ridge),
        rank=int(rank),
    )
    return model, scores


def infer_family_scores(
    target: ArrayLike,
    center: ArrayLike,
    modes: ArrayLike,
    penalty: float = 1.0,
) -> np.ndarray:
    """Infer ridge-regularized factor scores for one known geometry (diagnostic/CV)."""

    modes = np.asarray(modes, dtype=float)
    if modes.shape[0] == 0:
        return np.zeros(0)
    target_vector = upper_triangle_features(target)
    center_vector = upper_triangle_features(center)
    design = np.stack([upper_triangle_features(mode) for mode in modes], axis=1)
    return np.linalg.solve(
        design.T @ design + float(penalty) * np.eye(modes.shape[0]),
        design.T @ (target_vector - center_vector),
    )


def save_linear_geometry_model(
    model: LinearGeometryModel,
    directory: str | Path,
    metadata: Mapping[str, Any],
) -> None:
    """Serialize a frozen model and auditable JSON metadata."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "model.npz",
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        center_intercept=np.asarray(model.center_intercept),
        center_coefficients=model.center_coefficients,
        mode_intercepts=model.mode_intercepts,
        mode_coefficients=model.mode_coefficients,
        score_precision=model.score_precision,
    )
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "provenance": "ISO",
        "feature_names": list(model.feature_names),
        "ridge": model.ridge,
        "rank": model.rank,
        **dict(metadata),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_linear_geometry_model(
    directory: str | Path,
    *,
    require_iso_provenance: bool = True,
) -> tuple[LinearGeometryModel, dict[str, Any]]:
    """Load and validate a frozen linear-geometry artifact."""

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"unsupported geometry artifact version {manifest.get('artifact_version')!r}"
        )
    if require_iso_provenance and manifest.get("provenance") != "ISO":
        raise ValueError("MoPrP validation requires an ISO-provenance geometry artifact")
    with np.load(directory / "model.npz") as arrays:
        model = LinearGeometryModel(
            feature_names=tuple(manifest["feature_names"]),
            feature_mean=np.asarray(arrays["feature_mean"]),
            feature_scale=np.asarray(arrays["feature_scale"]),
            center_intercept=float(arrays["center_intercept"]),
            center_coefficients=np.asarray(arrays["center_coefficients"]),
            mode_intercepts=np.asarray(arrays["mode_intercepts"]),
            mode_coefficients=np.asarray(arrays["mode_coefficients"]),
            score_precision=np.asarray(arrays["score_precision"]),
            ridge=float(manifest["ridge"]),
            rank=int(manifest["rank"]),
        )
    if model.feature_mean.size != len(model.feature_names):
        raise ValueError("geometry artifact feature schema and scaler disagree")
    return model, manifest


def feature_out_of_distribution(
    pair_features: ArrayLike, model: LinearGeometryModel, threshold: float = 5.0
) -> dict[str, float | bool]:
    """Report, without clipping, standardized feature extrapolation."""

    z = np.abs((np.asarray(pair_features) - model.feature_mean) / model.feature_scale)
    maximum = float(np.max(z))
    fraction = float(np.mean(z > threshold))
    return {"max_abs_z": maximum, "fraction_over_threshold": fraction, "flagged": maximum > threshold}
