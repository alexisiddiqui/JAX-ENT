#!/usr/bin/env python3
"""Gate competing HDX covariance constructions against the known BI ensemble.

This analysis deliberately separates covariance over conformational frames, latent
target draws, and uptake timepoints.  It does not run or select any reweighting fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.src.analysis.pf_variance import (
    curve_precision_from_target_uptake,
    uptake_log_pf_jacobian,
)


TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], dtype=np.float64)
PRIMARY_ALPHA = 0.05
DEFAULT_ALPHAS = (0.01, 0.05, 0.10)


@dataclass
class Moments:
    dimension: int
    count: int = 0
    total: np.ndarray | None = None
    cross: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.total = np.zeros(self.dimension, dtype=np.float64)
        self.cross = np.zeros((self.dimension, self.dimension), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.count += values.shape[0]
        self.total += values.sum(axis=0)
        self.cross += values.T @ values

    def covariance(self) -> np.ndarray:
        if self.count == 0:
            raise ValueError("Cannot calculate covariance from an empty accumulator")
        mean = self.total / self.count
        covariance = self.cross / self.count - np.outer(mean, mean)
        return 0.5 * (covariance + covariance.T)


def population_covariance(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Population covariance for observations-by-variables input."""

    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        weights = np.full(values.shape[0], 1.0 / values.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()
    mean = weights @ values
    centered = values - mean
    covariance = (centered * weights[:, None]).T @ centered
    return 0.5 * (covariance + covariance.T)


def covariance_factor(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-10 * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError(f"Covariance is not positive semidefinite: {eigenvalues.min():.3e}")
    return eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))[None, :]


def regularized_profiles(covariance: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    covariance = np.asarray(covariance, dtype=np.float64)
    dimension = covariance.shape[0]
    scale = float(np.trace(covariance) / dimension)
    regularized = (1.0 - alpha) * covariance + (
        alpha * scale + 1e-8 * scale + 1e-12
    ) * np.eye(dimension)
    factor = np.linalg.cholesky(regularized)
    inverse_factor = np.linalg.solve(factor, np.eye(dimension))
    precision = inverse_factor.T @ inverse_factor
    marginal = np.diag(regularized)
    precision_diagonal = np.diag(precision)
    conditional = 1.0 / precision_diagonal
    return {
        "regularized": regularized,
        "marginal": marginal,
        "precision_diagonal": precision_diagonal,
        "conditional": conditional,
    }


def relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    denominator = float(np.linalg.norm(expected))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0 if np.linalg.norm(observed) <= np.finfo(np.float64).tiny else np.inf
    return float(np.linalg.norm(observed - expected) / denominator)


def normalize_sum(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if not np.isfinite(total) or abs(total) <= np.finfo(np.float64).tiny:
        return None
    return values / total


def normalize_mean(values: np.ndarray) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if not np.isfinite(mean) or mean <= np.finfo(np.float64).tiny:
        return None
    return values / mean


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if np.std(left) <= 1e-14 or np.std(right) <= 1e-14:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average").to_numpy(dtype=np.float64)


def profile_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    permutation_seed: int,
    permutation_count: int = 0,
) -> dict[str, float]:
    reference_normalized = normalize_mean(reference)
    candidate_normalized = normalize_mean(candidate)
    if reference_normalized is None or candidate_normalized is None:
        return {"pearson": np.nan, "spearman": np.nan, "log_rmse": np.nan, "p": np.nan}
    pearson = correlation(reference_normalized, candidate_normalized)
    spearman = correlation(ranks(reference_normalized), ranks(candidate_normalized))
    tiny = np.finfo(np.float64).tiny
    log_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.log(np.maximum(candidate_normalized, tiny))
                    - np.log(np.maximum(reference_normalized, tiny))
                )
            )
        )
    )
    if not np.isfinite(pearson) or permutation_count <= 0:
        p_value = np.nan
    else:
        rng = np.random.default_rng(permutation_seed)
        permutations = np.argsort(
            rng.random((permutation_count, candidate_normalized.size)), axis=1
        )
        permuted_candidates = candidate_normalized[permutations]
        centered_reference = reference_normalized - reference_normalized.mean()
        centered_candidates = permuted_candidates - permuted_candidates.mean(axis=1, keepdims=True)
        denominator = np.linalg.norm(centered_reference) * np.linalg.norm(
            centered_candidates, axis=1
        )
        permuted = centered_candidates @ centered_reference / denominator
        p_value = float((1 + np.sum(permuted >= pearson)) / (permuted.size + 1))
    return {"pearson": pearson, "spearman": spearman, "log_rmse": log_rmse, "p": p_value}


def matrix_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_normalized = normalize_sum(reference)
    candidate_normalized = normalize_sum(candidate)
    if reference_normalized is None or candidate_normalized is None:
        normalized_distance = np.nan
        off_diagonal_correlation = np.nan
    else:
        normalized_distance = float(np.linalg.norm(candidate_normalized - reference_normalized))
        mask = ~np.eye(reference.shape[0], dtype=bool)
        off_diagonal_correlation = correlation(
            reference_normalized[mask], candidate_normalized[mask]
        )
    reference_trace = float(np.trace(reference))
    candidate_trace = float(np.trace(candidate))
    trace_ratio = (
        candidate_trace / reference_trace
        if abs(reference_trace) > np.finfo(np.float64).tiny
        else np.nan
    )
    return {
        "raw_relative_error": relative_error(candidate, reference),
        "normalized_frobenius_distance": normalized_distance,
        "off_diagonal_correlation": off_diagonal_correlation,
        "raw_trace_ratio": trace_ratio,
    }


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def load_inputs(results_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((results_dir / "manifest.json").read_text())
    feature_path = Path(manifest["inputs"]["bi_features"]["path"])
    cluster_path = Path(manifest["inputs"]["bi_clusters"]["path"])
    with np.load(feature_path) as features:
        log_pf = (
            0.35 * np.asarray(features["heavy_contacts"], dtype=np.float64)
            + 2.0 * np.asarray(features["acceptor_contacts"], dtype=np.float64)
        )
        k_ints = np.asarray(features["k_ints"], dtype=np.float64)
    assignments = pd.read_csv(cluster_path)["cluster_assignment"].to_numpy(dtype=int)
    weights = np.zeros(assignments.size, dtype=np.float64)
    weights[assignments == 0] = 0.4 / np.sum(assignments == 0)
    weights[assignments == 1] = 0.6 / np.sum(assignments == 1)
    return manifest, log_pf, k_ints, weights


def add_matrix(
    matrices: dict[str, np.ndarray], metadata: list[dict[str, Any]], covariance: np.ndarray, **fields: Any
) -> str:
    parts = [fields["panel"], fields["construction"]]
    for name in ("scale", "seed", "timepoint"):
        value = fields.get(name)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            parts.append(f"{name}-{value:g}" if isinstance(value, (float, np.floating)) else f"{name}-{value}")
    matrix_id = "__".join(str(part).replace(".", "p") for part in parts)
    matrices[matrix_id] = 0.5 * (np.asarray(covariance) + np.asarray(covariance).T)
    metadata.append({"matrix_id": matrix_id, **fields})
    return matrix_id


def build_constructions(
    results_dir: Path,
    draws: int,
    batch_size: int,
    monte_carlo_seed: int,
) -> tuple[
    dict[str, np.ndarray],
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, float],
]:
    manifest, log_pf, k_ints, weights = load_inputs(results_dir)
    mean_log_pf = log_pf @ weights
    residue_pf_covariance = population_covariance(log_pf.T, weights)
    factor = covariance_factor(residue_pf_covariance)
    panels = tuple(manifest["config"]["panels"])
    mappings = {
        panel: np.load(results_dir / f"panel_{panel}_mapping.npz")["mapping"].astype(np.float64)
        for panel in panels
    }
    stacked_mapping = np.concatenate([mappings[panel] for panel in panels], axis=0)
    panel_slices: dict[str, slice] = {}
    offset = 0
    for panel in panels:
        panel_slices[panel] = slice(offset, offset + mappings[panel].shape[0])
        offset += mappings[panel].shape[0]

    matrices: dict[str, np.ndarray] = {}
    residue_matrices: dict[str, np.ndarray] = {"discrete_log_pf": residue_pf_covariance}
    metadata: list[dict[str, Any]] = []
    validation: dict[str, float] = {}

    discrete_uptake_residue: dict[float, np.ndarray] = {}
    for timepoint in TIMEPOINTS:
        x = timepoint * k_ints[:, None] / np.exp(log_pf)
        uptake = 1.0 - np.exp(-x)
        covariance = population_covariance(uptake.T, weights)
        discrete_uptake_residue[float(timepoint)] = covariance
        residue_matrices[f"discrete_uptake__time-{timepoint:g}"] = covariance

    mapping_errors = []
    for panel, mapping in mappings.items():
        mapped_values = log_pf.T @ mapping.T
        direct = population_covariance(mapped_values, weights)
        congruence = mapping @ residue_pf_covariance @ mapping.T
        mapping_errors.append(relative_error(direct, congruence))
        add_matrix(
            matrices,
            metadata,
            congruence,
            panel=panel,
            construction="discrete_log_pf",
            coordinate="log_pf",
            sampling_axis="frames",
            scale=None,
            seed=None,
            timepoint=None,
        )
        for timepoint in TIMEPOINTS:
            covariance = discrete_uptake_residue[float(timepoint)]
            x = timepoint * k_ints[:, None] / np.exp(log_pf)
            mapped_values = (1.0 - np.exp(-x)).T @ mapping.T
            direct = population_covariance(mapped_values, weights)
            congruence = mapping @ covariance @ mapping.T
            mapping_errors.append(relative_error(direct, congruence))
            add_matrix(
                matrices,
                metadata,
                congruence,
                panel=panel,
                construction="discrete_uptake",
                coordinate="uptake",
                sampling_axis="frames",
                scale=None,
                seed=None,
                timepoint=float(timepoint),
            )
    validation["mapping_relative_error_max"] = max(mapping_errors)

    scales = tuple(float(value) for value in manifest["config"]["target_scales"])
    jacobian = np.asarray(
        uptake_log_pf_jacobian(jnp.asarray(mean_log_pf), jnp.asarray(k_ints), jnp.asarray(TIMEPOINTS)),
        dtype=np.float64,
    )
    for panel, mapping in mappings.items():
        base_peptide = mapping @ residue_pf_covariance @ mapping.T
        for scale in scales:
            add_matrix(
                matrices,
                metadata,
                scale * scale * base_peptide,
                panel=panel,
                construction="scaled_latent_log_pf",
                coordinate="log_pf",
                sampling_axis="latent_draws",
                scale=scale,
                seed=None,
                timepoint=None,
            )
            for time_index, timepoint in enumerate(TIMEPOINTS):
                derivative = jacobian[time_index]
                delta_residue = (
                    scale * scale * derivative[:, None] * residue_pf_covariance * derivative[None, :]
                )
                residue_matrices[
                    f"delta_uptake__scale-{scale:g}__time-{timepoint:g}"
                ] = delta_residue
                add_matrix(
                    matrices,
                    metadata,
                    mapping @ delta_residue @ mapping.T,
                    panel=panel,
                    construction="delta_uptake",
                    coordinate="uptake",
                    sampling_axis="analytic_latent",
                    scale=scale,
                    seed=None,
                    timepoint=float(timepoint),
                )

    accumulators: dict[tuple[str, float, float, str], Moments] = {}
    linear_accumulators: dict[tuple[str, str], Moments] = {}
    for panel in panels:
        for half in ("all", "first", "second"):
            linear_accumulators[(panel, half)] = Moments(mappings[panel].shape[0])
        for scale in scales:
            for timepoint in TIMEPOINTS:
                for half in ("all", "first", "second"):
                    accumulators[(panel, scale, float(timepoint), half)] = Moments(
                        mappings[panel].shape[0]
                    )

    rng = np.random.default_rng(monte_carlo_seed)
    generated = 0
    while generated < draws:
        size = min(batch_size, draws - generated)
        perturbations = rng.standard_normal((size, factor.shape[1])) @ factor.T
        mapped_perturbations = perturbations @ stacked_mapping.T
        boundary = draws // 2
        indices = np.arange(generated, generated + size)
        half_masks = {
            "all": np.ones(size, dtype=bool),
            "first": indices < boundary,
            "second": indices >= boundary,
        }
        for panel in panels:
            values = mapped_perturbations[:, panel_slices[panel]]
            for half, mask in half_masks.items():
                if mask.any():
                    linear_accumulators[(panel, half)].update(values[mask])
        for scale in scales:
            if scale == 0.0:
                continue
            latent = mean_log_pf[None, :] + scale * perturbations
            inverse_pf = np.exp(-latent)
            for timepoint in TIMEPOINTS:
                uptake = 1.0 - np.exp(-timepoint * k_ints[None, :] * inverse_pf)
                mapped = uptake @ stacked_mapping.T
                for panel in panels:
                    values = mapped[:, panel_slices[panel]]
                    for half, mask in half_masks.items():
                        if mask.any():
                            accumulators[(panel, scale, float(timepoint), half)].update(values[mask])
        generated += size

    mc_pf_errors = []
    mc_half_distances = []
    mc_half_correlations = []
    for panel, mapping in mappings.items():
        expected = mapping @ residue_pf_covariance @ mapping.T
        observed = linear_accumulators[(panel, "all")].covariance()
        mc_pf_errors.append(relative_error(observed, expected))
        first = linear_accumulators[(panel, "first")].covariance()
        second = linear_accumulators[(panel, "second")].covariance()
        mc_half_distances.append(matrix_metrics(first, second)["normalized_frobenius_distance"])
        mc_half_correlations.append(correlation(first.ravel(), second.ravel()))
        for scale in scales:
            for timepoint in TIMEPOINTS:
                if scale == 0.0:
                    covariance = np.zeros((mapping.shape[0], mapping.shape[0]), dtype=np.float64)
                else:
                    covariance = accumulators[(panel, scale, float(timepoint), "all")].covariance()
                    first = accumulators[(panel, scale, float(timepoint), "first")].covariance()
                    second = accumulators[(panel, scale, float(timepoint), "second")].covariance()
                    mc_half_distances.append(
                        matrix_metrics(first, second)["normalized_frobenius_distance"]
                    )
                    mc_half_correlations.append(correlation(first.ravel(), second.ravel()))
                add_matrix(
                    matrices,
                    metadata,
                    covariance,
                    panel=panel,
                    construction="monte_carlo_uptake",
                    coordinate="uptake",
                    sampling_axis="latent_draws",
                    scale=scale,
                    seed=monte_carlo_seed,
                    timepoint=float(timepoint),
                )
    validation["mc_log_pf_relative_error_max"] = max(mc_pf_errors)
    validation["mc_half_normalized_distance_max"] = float(np.nanmax(mc_half_distances))
    validation["mc_half_covariance_correlation_min"] = float(np.nanmin(mc_half_correlations))

    precision_errors = []
    for panel, mapping in mappings.items():
        target = []
        for timepoint in TIMEPOINTS:
            uptake = 1.0 - np.exp(-timepoint * k_ints / np.exp(mean_log_pf))
            target.append(mapping @ uptake)
        target = np.asarray(target)
        curve_covariance = population_covariance(target)
        matrix_id = add_matrix(
            matrices,
            metadata,
            curve_covariance,
            panel=panel,
            construction="fixed_mean_target_curve",
            coordinate="uptake",
            sampling_axis="timepoints",
            scale=1.0,
            seed=None,
            timepoint=None,
        )
        profile = regularized_profiles(curve_covariance, PRIMARY_ALPHA)
        precision = np.linalg.inv(profile["regularized"])
        precision *= precision.shape[0] / np.trace(precision)
        current = np.asarray(
            curve_precision_from_target_uptake(jnp.asarray(target), alpha=PRIMARY_ALPHA),
            dtype=np.float64,
        )
        precision_errors.append(relative_error(current, precision))
        matrices[f"{matrix_id}__target_values"] = target
    validation["curve_precision_reproduction_error_max"] = max(precision_errors)
    return matrices, pd.DataFrame(metadata), residue_matrices, validation


def summarize_and_profile(
    matrices: dict[str, np.ndarray], metadata: pd.DataFrame, alphas: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, float], dict[str, np.ndarray]]]:
    summaries = []
    profiles = []
    profile_lookup: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    for row in metadata.itertuples(index=False):
        covariance = matrices[row.matrix_id]
        eigenvalues = np.linalg.eigvalsh(covariance)
        tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-10
        summaries.append(
            {
                **row._asdict(),
                "dimension": covariance.shape[0],
                "trace": float(np.trace(covariance)),
                "frobenius_norm": float(np.linalg.norm(covariance)),
                "numerical_rank": int(np.sum(eigenvalues > tolerance)),
                "minimum_eigenvalue": float(eigenvalues.min()),
                "maximum_eigenvalue": float(eigenvalues.max()),
            }
        )
        for alpha in alphas:
            calculated = regularized_profiles(covariance, alpha)
            profile_lookup[(row.matrix_id, alpha)] = calculated
            for peptide in range(covariance.shape[0]):
                profiles.append(
                    {
                        **row._asdict(),
                        "alpha": alpha,
                        "peptide_index": peptide,
                        "raw_diagonal": float(covariance[peptide, peptide]),
                        "marginal": float(calculated["marginal"][peptide]),
                        "precision_diagonal": float(calculated["precision_diagonal"][peptide]),
                        "conditional": float(calculated["conditional"][peptide]),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(profiles), profile_lookup


def compare_pair(
    *,
    kind: str,
    reference_id: str,
    candidate_id: str,
    matrices: dict[str, np.ndarray],
    profile_lookup: dict[tuple[str, float], dict[str, np.ndarray]],
    alpha: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "comparison": kind,
        "reference_id": reference_id,
        "candidate_id": candidate_id,
        "alpha": alpha,
        **matrix_metrics(matrices[reference_id], matrices[candidate_id]),
    }
    for profile_name in ("marginal", "conditional"):
        permutation_count = (
            10_000
            if kind.startswith("curve_vs_") and np.isclose(alpha, PRIMARY_ALPHA)
            else 0
        )
        metrics = profile_metrics(
            profile_lookup[(reference_id, alpha)][profile_name],
            profile_lookup[(candidate_id, alpha)][profile_name],
            stable_seed(kind, reference_id, candidate_id, alpha, profile_name),
            permutation_count=permutation_count,
        )
        result.update({f"{profile_name}_{name}": value for name, value in metrics.items()})
    return result


def build_comparisons(
    matrices: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    profile_lookup: dict[tuple[str, float], dict[str, np.ndarray]],
    alphas: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup: dict[tuple[Any, ...], str] = {}
    for row in metadata.itertuples(index=False):
        lookup[(row.panel, row.construction, row.scale, row.seed, row.timepoint)] = row.matrix_id
    comparisons = []

    def find(panel: str, construction: str, scale=None, seed=None, timepoint=None) -> str:
        candidates = metadata[(metadata.panel == panel) & (metadata.construction == construction)]
        for column, value in (("scale", scale), ("seed", seed), ("timepoint", timepoint)):
            candidates = candidates[candidates[column].isna()] if value is None else candidates[np.isclose(candidates[column], value, equal_nan=False)]
        if len(candidates) != 1:
            raise KeyError((panel, construction, scale, seed, timepoint, len(candidates)))
        return str(candidates.iloc[0].matrix_id)

    for panel in metadata.panel.unique():
        mc_seed = int(
            metadata[
                (metadata.panel == panel)
                & (metadata.construction == "monte_carlo_uptake")
            ].seed.dropna().iloc[0]
        )
        discrete_uptake_ids = {
            float(timepoint): find(panel, "discrete_uptake", timepoint=float(timepoint))
            for timepoint in TIMEPOINTS
        }
        for left, right in combinations(TIMEPOINTS, 2):
            for alpha in alphas:
                comparisons.append(
                    compare_pair(
                        kind="time_dependence_discrete",
                        reference_id=discrete_uptake_ids[float(left)],
                        candidate_id=discrete_uptake_ids[float(right)],
                        matrices=matrices,
                        profile_lookup=profile_lookup,
                        alpha=alpha,
                    )
                )
        scales = sorted(metadata[(metadata.panel == panel) & (metadata.construction == "monte_carlo_uptake")].scale.unique())
        for scale in scales:
            mc_ids = {
                float(timepoint): find(panel, "monte_carlo_uptake", scale=scale, seed=mc_seed, timepoint=float(timepoint))
                for timepoint in TIMEPOINTS
            }
            for left, right in combinations(TIMEPOINTS, 2):
                for alpha in alphas:
                    comparisons.append(
                        compare_pair(
                            kind="time_dependence_mc",
                            reference_id=mc_ids[float(left)],
                            candidate_id=mc_ids[float(right)],
                            matrices=matrices,
                            profile_lookup=profile_lookup,
                            alpha=alpha,
                        )
                    )
            for timepoint in TIMEPOINTS:
                delta_id = find(panel, "delta_uptake", scale=scale, timepoint=float(timepoint))
                for alpha in alphas:
                    comparisons.append(
                        compare_pair(
                            kind="delta_vs_mc",
                            reference_id=delta_id,
                            candidate_id=mc_ids[float(timepoint)],
                            matrices=matrices,
                            profile_lookup=profile_lookup,
                            alpha=alpha,
                        )
                    )
                if np.isclose(scale, 1.0):
                    for alpha in alphas:
                        comparisons.append(
                            compare_pair(
                                kind="discrete_vs_mc_s1",
                                reference_id=discrete_uptake_ids[float(timepoint)],
                                candidate_id=mc_ids[float(timepoint)],
                                matrices=matrices,
                                profile_lookup=profile_lookup,
                                alpha=alpha,
                            )
                        )
        pf_id = find(panel, "discrete_log_pf")
        curves = metadata[(metadata.panel == panel) & (metadata.construction == "fixed_mean_target_curve")]
        for curve in curves.itertuples(index=False):
            scale = float(curve.scale)
            for alpha in alphas:
                comparisons.append(
                    compare_pair(
                        kind="curve_vs_discrete_pf",
                        reference_id=pf_id,
                        candidate_id=curve.matrix_id,
                        matrices=matrices,
                        profile_lookup=profile_lookup,
                        alpha=alpha,
                    )
                )
                if scale > 0.0:
                    scaled_id = find(panel, "scaled_latent_log_pf", scale=scale)
                    comparisons.append(
                        compare_pair(
                            kind="curve_vs_scaled_pf",
                            reference_id=scaled_id,
                            candidate_id=curve.matrix_id,
                            matrices=matrices,
                            profile_lookup=profile_lookup,
                            alpha=alpha,
                        )
                    )
                for timepoint in TIMEPOINTS:
                    comparisons.append(
                        compare_pair(
                            kind="curve_vs_discrete_uptake",
                            reference_id=discrete_uptake_ids[float(timepoint)],
                            candidate_id=curve.matrix_id,
                            matrices=matrices,
                            profile_lookup=profile_lookup,
                            alpha=alpha,
                        )
                    )
                    if scale > 0.0:
                        mc_id = find(
                            panel,
                            "monte_carlo_uptake",
                            scale=scale,
                            seed=mc_seed,
                            timepoint=float(timepoint),
                        )
                        comparisons.append(
                            compare_pair(
                                kind="curve_vs_mc_uptake",
                                reference_id=mc_id,
                                candidate_id=curve.matrix_id,
                                matrices=matrices,
                                profile_lookup=profile_lookup,
                                alpha=alpha,
                            )
                        )
    comparison_frame = pd.DataFrame(comparisons)

    stability = []
    for row in metadata.itertuples(index=False):
        for alpha in alphas:
            if np.isclose(alpha, PRIMARY_ALPHA):
                continue
            for profile_name in ("marginal", "conditional"):
                metrics = profile_metrics(
                    profile_lookup[(row.matrix_id, PRIMARY_ALPHA)][profile_name],
                    profile_lookup[(row.matrix_id, alpha)][profile_name],
                    stable_seed("alpha", row.matrix_id, alpha, profile_name),
                )
                stability.append(
                    {
                        **row._asdict(),
                        "profile": profile_name,
                        "reference_alpha": PRIMARY_ALPHA,
                        "candidate_alpha": alpha,
                        **metrics,
                    }
                )
    return comparison_frame, pd.DataFrame(stability)


def evaluate_gates(
    validation: dict[str, float],
    comparisons: pd.DataFrame,
    stability: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    numerical_checks = {
        "mapping_relative_error": validation["mapping_relative_error_max"] <= 1e-10,
        "mc_log_pf_recovery": validation["mc_log_pf_relative_error_max"] <= 0.02,
        "mc_half_distance": validation["mc_half_normalized_distance_max"] <= 0.03,
        "mc_half_correlation": validation["mc_half_covariance_correlation_min"] >= 0.995,
        "curve_precision_reproduction": validation["curve_precision_reproduction_error_max"] <= 1e-5,
    }
    numerical = {"passed": bool(all(numerical_checks.values())), "checks": numerical_checks, "metrics": validation}

    def time_gate(kind: str) -> dict[str, Any]:
        rows = comparisons[(comparisons.comparison == kind) & np.isclose(comparisons.alpha, PRIMARY_ALPHA)]
        independent = (
            (rows.normalized_frobenius_distance <= 0.05)
            & (rows.marginal_pearson >= 0.99)
            & (rows.conditional_pearson >= 0.99)
            & (rows.marginal_log_rmse <= 0.05)
            & (rows.conditional_log_rmse <= 0.05)
        )
        material = (
            (rows.normalized_frobenius_distance > 0.10)
            | (rows.marginal_log_rmse > 0.10)
            | (rows.conditional_log_rmse > 0.10)
        )
        return {
            "effectively_time_independent": bool(independent.all()),
            "material_time_dependence_detected": bool(material.any()),
            "maximum_normalized_matrix_distance": float(rows.normalized_frobenius_distance.max()),
            "maximum_marginal_log_rmse": float(rows.marginal_log_rmse.max()),
            "maximum_conditional_log_rmse": float(rows.conditional_log_rmse.max()),
        }

    forward_rows = comparisons[
        (comparisons.comparison == "delta_vs_mc")
        & np.isclose(comparisons.alpha, PRIMARY_ALPHA)
        & comparisons.reference_id.str.contains("scale-0p1")
    ]
    forward_checks = (
        (forward_rows.raw_relative_error <= 0.05)
        & (forward_rows.normalized_frobenius_distance <= 0.05)
        & (forward_rows.marginal_log_rmse <= 0.05)
        & (forward_rows.conditional_log_rmse <= 0.05)
    )
    forward = {
        "passed": bool(forward_checks.all()),
        "maximum_raw_relative_error": float(forward_rows.raw_relative_error.max()),
        "maximum_normalized_matrix_distance": float(forward_rows.normalized_frobenius_distance.max()),
        "maximum_marginal_log_rmse": float(forward_rows.marginal_log_rmse.max()),
        "maximum_conditional_log_rmse": float(forward_rows.conditional_log_rmse.max()),
    }

    curve_metadata = metadata[metadata.construction == "fixed_mean_target_curve"]

    def surrogate_gate(kind: str, scale: float, profile: str, uptake_units: bool) -> dict[str, Any]:
        curve_ids = set(curve_metadata[np.isclose(curve_metadata.scale, scale)].matrix_id)
        rows = comparisons[
            (comparisons.comparison == kind)
            & np.isclose(comparisons.alpha, PRIMARY_ALPHA)
            & comparisons.candidate_id.isin(curve_ids)
        ]
        curve_stability = stability[
            stability.matrix_id.isin(curve_ids) & (stability.profile == profile)
        ]
        conditions = (
            (rows.normalized_frobenius_distance <= 0.25)
            & (rows.off_diagonal_correlation >= 0.80)
            & (rows[f"{profile}_pearson"] >= 0.90)
            & (rows[f"{profile}_spearman"] >= 0.90)
            & (rows[f"{profile}_log_rmse"] <= 0.25)
            & (rows[f"{profile}_p"] < 0.01)
        )
        if uptake_units:
            conditions &= rows.raw_trace_ratio.between(0.5, 2.0)
        stability_conditions = (
            (curve_stability.pearson >= 0.95) & (curve_stability.log_rmse <= 0.10)
        )
        return {
            "passed": bool(len(rows) > 0 and conditions.all() and stability_conditions.all()),
            "comparisons": int(len(rows)),
            "failed_comparisons": int((~conditions).sum()),
            "alpha_stability_passed": bool(len(curve_stability) > 0 and stability_conditions.all()),
            "worst_normalized_matrix_distance": float(rows.normalized_frobenius_distance.max()),
            "worst_off_diagonal_correlation": float(rows.off_diagonal_correlation.min()),
            "worst_profile_pearson": float(rows[f"{profile}_pearson"].min()),
            "worst_profile_spearman": float(rows[f"{profile}_spearman"].min()),
            "worst_profile_log_rmse": float(rows[f"{profile}_log_rmse"].max()),
            "worst_permutation_p": float(rows[f"{profile}_p"].max()),
            "worst_raw_trace_ratio": float(rows.raw_trace_ratio.max()) if uptake_units else None,
        }

    surrogate: dict[str, Any] = {}
    selection: dict[str, str] = {}
    for scale in sorted(curve_metadata.scale.unique()):
        label = f"s={scale:g}"
        surrogate[label] = {}
        for reference, kind, uptake_units in (
            ("conformational_log_pf", "curve_vs_discrete_pf", False),
            ("conformational_uptake_all_times", "curve_vs_discrete_uptake", True),
        ):
            results = {
                profile: surrogate_gate(kind, float(scale), profile, uptake_units)
                for profile in ("marginal", "conditional")
            }
            surrogate[label][reference] = results
            conditional_error = results["conditional"]["worst_profile_log_rmse"]
            marginal_error = results["marginal"]["worst_profile_log_rmse"]
            if results["conditional"]["passed"] and conditional_error <= 0.9 * marginal_error:
                chosen = "conditional"
            elif results["marginal"]["passed"]:
                chosen = "marginal"
            else:
                chosen = "neither"
            selection[f"{label}:{reference}"] = chosen

    fixed_mean_pass = all(
        any(
            surrogate["s=1"][reference][profile]["passed"]
            for profile in ("marginal", "conditional")
        )
        for reference in ("conformational_log_pf", "conformational_uptake_all_times")
    )
    return {
        "objective": "Select the physical conformational covariance and test target-curve inverse diagonals as surrogates.",
        "numerical_validity": numerical,
        "time_dependence": {
            "discrete_ensemble_uptake": time_gate("time_dependence_discrete"),
            "monte_carlo_latent_uptake": time_gate("time_dependence_mc"),
        },
        "small_noise_jacobian_propagation": forward,
        "target_curve_surrogate": surrogate,
        "fixed_mean_curve_surrogate_passed": bool(fixed_mean_pass),
        "profile_selection": selection,
        "selected_construction_for_conformational_pf_spread": "discrete_log_pf_covariance",
        "selected_construction_for_conformational_uptake_spread": "timepoint_specific_discrete_uptake_covariance",
        "optimization_rerun_required": False,
    }


def make_plots(
    output_dir: Path,
    metadata: pd.DataFrame,
    profiles: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    primary = profiles[np.isclose(profiles.alpha, PRIMARY_ALPHA)]
    panels = tuple(metadata.panel.unique())
    figure, axes = plt.subplots(len(panels), 2, figsize=(12, 3.5 * len(panels)), squeeze=False)
    for row_index, panel in enumerate(panels):
        for column, profile_name in enumerate(("marginal", "conditional")):
            axis = axes[row_index, column]
            reference = primary[
                (primary.panel == panel) & (primary.construction == "discrete_log_pf")
            ].sort_values("peptide_index")
            values = reference[profile_name].to_numpy()
            axis.plot(reference.peptide_index, values / values.mean(), "k-o", label="BI log-PF")
            curve = primary[
                (primary.panel == panel) & (primary.construction == "fixed_mean_target_curve")
            ].sort_values("peptide_index")
            values = curve[profile_name].to_numpy()
            axis.plot(curve.peptide_index, values / values.mean(), alpha=0.65, label="fixed-mean curve")
            axis.set_title(f"{panel}: {profile_name}")
            axis.set_xlabel("peptide index")
            axis.set_ylabel("mean-normalized profile")
            if row_index == 0 and column == 1:
                axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "target_curve_vs_pf_profiles.png", dpi=180)
    plt.close(figure)

    time_rows = comparisons[
        (comparisons.comparison == "time_dependence_discrete")
        & np.isclose(comparisons.alpha, PRIMARY_ALPHA)
    ].copy()
    time_rows["panel"] = time_rows.reference_id.str.split("__").str[0]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    data = [
        time_rows[time_rows.panel == panel].normalized_frobenius_distance.to_numpy()
        for panel in panels
    ]
    axis.boxplot(data, tick_labels=panels)
    axis.axhline(0.05, color="tab:green", linestyle="--", label="time-independent gate")
    axis.axhline(0.10, color="tab:red", linestyle="--", label="material difference")
    axis.set_ylabel("trace-normalized covariance distance")
    axis.set_title("Discrete BI uptake covariance changes across timepoints")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "uptake_timepoint_covariance_differences.png", dpi=180)
    plt.close(figure)


def write_report(output_dir: Path, gates: dict[str, Any]) -> None:
    numerical = gates["numerical_validity"]
    discrete = gates["time_dependence"]["discrete_ensemble_uptake"]
    mc = gates["time_dependence"]["monte_carlo_latent_uptake"]
    jacobian = gates["small_noise_jacobian_propagation"]
    lines = [
        "# HDX covariance-construction litmus test",
        "",
        f"- Numerical validity: **{'PASS' if numerical['passed'] else 'FAIL'}**.",
        f"- Discrete uptake covariance materially time-dependent: **{discrete['material_time_dependence_detected']}** "
        f"(maximum normalized distance {discrete['maximum_normalized_matrix_distance']:.3f}).",
        f"- Latent Monte Carlo uptake covariance materially time-dependent: **{mc['material_time_dependence_detected']}** "
        f"(maximum normalized distance {mc['maximum_normalized_matrix_distance']:.3f}).",
        f"- Small-noise Jacobian propagation: **{'PASS' if jacobian['passed'] else 'FAIL'}** "
        f"(maximum raw covariance error {jacobian['maximum_raw_relative_error']:.3%}).",
        f"- Fixed-mean target-curve surrogate: **{'PASS' if gates['fixed_mean_curve_surrogate_passed'] else 'FAIL'}**.",
        "- Selected target-curve profile: **neither marginal nor conditional at any scale**."
        if all(value == "neither" for value in gates["profile_selection"].values())
        else f"- Profile selections: `{gates['profile_selection']}`.",
        "",
        "The registered physical construction for conformational PF spread is the weighted discrete BI log-PF covariance. "
        "Timepoint covariance of one target curve is retained only as mean-fit geometry unless its separate surrogate gates pass.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices, metadata, residue_matrices, validation = build_constructions(
        args.results_dir, args.draws, args.batch_size, args.monte_carlo_seed
    )
    summaries, profiles, profile_lookup = summarize_and_profile(matrices, metadata, args.alphas)
    comparisons, stability = build_comparisons(matrices, metadata, profile_lookup, args.alphas)
    gates = evaluate_gates(validation, comparisons, stability, metadata)

    np.savez_compressed(
        args.output_dir / "peptide_covariance_matrices.npz",
        **{key: value for key, value in matrices.items() if not key.endswith("__target_values")},
    )
    np.savez_compressed(args.output_dir / "residue_covariance_matrices.npz", **residue_matrices)
    metadata.to_csv(args.output_dir / "matrix_metadata.csv", index=False)
    summaries.to_csv(args.output_dir / "matrix_summary.csv", index=False)
    profiles.to_csv(args.output_dir / "profile_values.csv", index=False)
    comparisons.to_csv(args.output_dir / "pairwise_differences.csv", index=False)
    stability.to_csv(args.output_dir / "shrinkage_sensitivity.csv", index=False)
    (args.output_dir / "gate_results.json").write_text(json.dumps(gates, indent=2, allow_nan=True))
    write_report(args.output_dir, gates)
    if not args.skip_plots:
        make_plots(args.output_dir, metadata, profiles, comparisons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--results-dir", type=Path, default=root / "_pf_peptide_moment_final")
    parser.add_argument("--output-dir", type=Path, default=root / "_pf_covariance_litmus")
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--monte-carlo-seed", type=int, default=20260715)
    parser.add_argument("--alphas", type=lambda text: tuple(float(x) for x in text.split(",")), default=DEFAULT_ALPHAS)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
