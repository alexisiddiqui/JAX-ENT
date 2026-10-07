"""Checkpoint 39: sparse within-system population extrapolation.

This extends the checkpoint-36/37 absolute pairwise predictors into an anchored
population estimator.  A small set of structural neighbourhood populations is
revealed, alpha is fitted only on differences among those labels, and the
remaining neighbourhoods are reconstructed from their distances to the known
anchors.  Replica C sparse-label prediction is primary; A/B-to-C transfer and
forced structural-cluster populations are sensitivity analyses.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas-cp39-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr
import yaml

from jaxent.examples.ATLAS_BV.analysis import (
    corrected_variance_recovery_checkpoint36 as cp36,
)
from jaxent.examples.ATLAS_BV.analysis.common import HERE
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    mass_metrics,
    neighbour_bandwidth,
)
from jaxent.examples.ATLAS_BV.analysis.local_variance_checkpoint28 import nearest
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)


OUTPUT = (
    HERE
    / "outputs/analysis/pairwise_geometry/checkpoint39_sparse_population_extrapolation"
)
GRAPH_ROOT = (
    HERE / "outputs/analysis/pairwise_geometry/graph_representation_audit_bradshaw_0p1"
)
CP36 = cp36.OUTPUT / "pilot"
CP37 = HERE / "outputs/analysis/pairwise_geometry/checkpoint37_global_alpha"
SEED = 20260925
REPLICAS = (1, 2, 3)
DIRECT_BASE = ("work_scale", "work_shape", "work_density", "work_opt", "pf_l1", "pf_l2")
DIRECT_DERIVED = ("work_fitting", "work_magnitude")
VARIANCE_METRICS = (
    "work_scale",
    "work_shape",
    "work_density",
    "work_opt",
    "pf_l1",
    "pf_l2",
)
CLUSTER_COUNTS = (5, 10)
JEFFREYS = 0.5
DISTRIBUTION_BINS = 20
DISTRIBUTION_SMOOTHING = 1.0e-12


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def finite_spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return 0.0
    value = float(spearmanr(target, prediction).statistic)
    return value if np.isfinite(value) else 0.0


def pairwise_magnitudes(values: np.ndarray) -> np.ndarray:
    """Absolute changes among a set of population values."""
    values = np.asarray(values, dtype=float)
    left, right = np.triu_indices(len(values), 1)
    return np.abs(values[left] - values[right])


def md_distribution_recovery(
    target: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Checkpoint-17 recovery of the held-out pairwise population changes."""
    if len(target) < 2:
        return np.nan
    target_changes = pairwise_magnitudes(target)
    predicted_changes = pairwise_magnitudes(prediction)
    reference_changes = pairwise_magnitudes(reference)
    return float(
        mass_metrics(
            target_changes,
            predicted_changes,
            reference_changes,
            DISTRIBUTION_BINS,
            DISTRIBUTION_SMOOTHING,
        )["distribution_recovery"]
    )


def pair_distance(
    left: np.ndarray, right: np.ndarray | None = None, kind: str = "l1"
) -> np.ndarray:
    """Distances between column observations, matching checkpoint 36 scaling."""
    a = np.atleast_2d(np.asarray(left, dtype=float))
    b = a if right is None else np.atleast_2d(np.asarray(right, dtype=float))
    delta = a.T[:, None, :] - b.T[None, :, :]
    if kind == "l2":
        return np.sqrt(np.mean(np.square(delta), axis=2))
    return np.mean(np.abs(delta), axis=2)


def fit_alpha(
    distance: np.ndarray, values: np.ndarray, labels: np.ndarray
) -> tuple[float, bool]:
    """Non-negative no-intercept scale from known-known absolute differences."""
    labels = np.asarray(labels, dtype=int)
    left, right = np.triu_indices(len(labels), 1)
    x = np.asarray(distance)[np.ix_(labels, labels)][left, right]
    y = np.abs(np.asarray(values)[labels[left]] - np.asarray(values)[labels[right]])
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).eps:
        return 0.0, True
    return max(0.0, float(np.dot(x, y) / denominator)), False


def reconstruct_value(
    anchor_values: np.ndarray, predicted_distances: np.ndarray
) -> float:
    """Exact one-dimensional least-squares reconstruction from unsigned distances."""
    return float(
        reconstruct_values(
            np.asarray(anchor_values), np.atleast_2d(predicted_distances)
        )[0]
    )


def reconstruct_values(
    anchor_values: np.ndarray, predicted_distances: np.ndarray
) -> np.ndarray:
    """Vectorized exact reconstruction for multiple queries sharing anchors."""
    anchors = np.asarray(anchor_values, dtype=float)
    radii = np.maximum(0.0, np.atleast_2d(predicted_distances).astype(float))
    if not len(anchors):
        raise ValueError("at least one anchor is required")
    if len(anchors) == 1:
        return np.full(len(radii), anchors[0], dtype=float)
    order = np.argsort(anchors, kind="stable")
    sorted_anchors = anchors[order]
    sorted_radii = radii[:, order]
    # One stationary point exists for each of the m+1 sign intervals. Repeated
    # anchors create zero-width intervals, which are harmless because boundaries
    # are also explicit candidates.
    count = len(sorted_anchors)
    interval = np.arange(count + 1)[:, None]
    signs = np.where(np.arange(count)[None, :] < interval, 1.0, -1.0)
    stationary = np.mean(
        sorted_anchors[None, None, :] + signs[None, :, :] * sorted_radii[:, None, :],
        axis=2,
    )
    lows = np.concatenate(([-np.inf], sorted_anchors))
    highs = np.concatenate((sorted_anchors, [np.inf]))
    valid = (stationary >= lows[None, :] - 1e-12) & (
        stationary <= highs[None, :] + 1e-12
    )
    stationary = np.where(valid, stationary, np.nan)
    boundaries = np.broadcast_to(sorted_anchors, (len(radii), count))
    candidates = np.concatenate((stationary, boundaries), axis=1)
    loss = np.sum(
        np.square(
            np.abs(candidates[:, :, None] - anchors[None, None, :]) - radii[:, None, :]
        ),
        axis=2,
    )
    loss = np.where(np.isfinite(candidates), loss, np.inf)
    best_loss = np.min(loss, axis=1)
    tied = np.isclose(loss, best_loss[:, None], rtol=1e-12, atol=1e-12)
    median_distance = np.abs(candidates - np.median(anchors))
    median_distance = np.where(tied, median_distance, np.inf)
    best_median = np.min(median_distance, axis=1)
    tied &= np.isclose(median_distance, best_median[:, None], rtol=1e-12, atol=1e-12)
    return np.min(np.where(tied, candidates, np.inf), axis=1)


def anchored_prediction(
    fit_distance: np.ndarray,
    query_distance: np.ndarray,
    source_values: np.ndarray,
    labels: np.ndarray,
    alpha: float | None = None,
) -> tuple[np.ndarray, float, bool]:
    labels = np.asarray(labels, dtype=int)
    fitted, degenerate = fit_alpha(fit_distance, source_values, labels)
    if alpha is None:
        alpha = fitted
    prediction = reconstruct_values(
        source_values[labels], alpha * query_distance[:, labels]
    )
    return prediction, float(alpha), degenerate


def maximin_order(distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    first = int(np.argmin(distance.mean(axis=1)))
    chosen = [first]
    remaining = np.ones(len(distance), dtype=bool)
    remaining[first] = False
    closest = distance[:, first].copy()
    while remaining.any():
        score = np.where(remaining, closest, -np.inf)
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        remaining[nxt] = False
        closest = np.minimum(closest, distance[:, nxt])
    return np.asarray(chosen, dtype=int)


def local_targets_and_weights(
    distance: np.ndarray,
    landmark_indices: np.ndarray,
    replicas: np.ndarray,
    bandwidth: float,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Replica-specific log kernel masses, normalized weights and effective counts."""
    targets, weights, effective = {}, {}, {}
    for replica in REPLICAS:
        indices = np.flatnonzero(replicas == replica)
        log_kernel = -0.5 * np.square(
            distance[np.ix_(landmark_indices, indices)]
            / max(bandwidth, np.finfo(float).eps)
        )
        denominators = np.full(len(landmark_indices), len(indices), dtype=float)
        local_lookup = {int(value): position for position, value in enumerate(indices)}
        for row, landmark in enumerate(landmark_indices):
            if int(landmark) in local_lookup:
                log_kernel[row, local_lookup[int(landmark)]] = -np.inf
                denominators[row] -= 1
        normalizer = logsumexp(log_kernel, axis=1)
        targets[replica] = normalizer - np.log(np.maximum(denominators, 1.0))
        normalized = np.exp(log_kernel - normalizer[:, None])
        invalid = ~np.isfinite(normalized).all(axis=1)
        if invalid.any():
            for row in np.flatnonzero(invalid):
                normalized[row] = 0.0
                nearest_index = int(np.argmin(distance[landmark_indices[row], indices]))
                normalized[row, nearest_index] = 1.0
        weights[replica] = normalized
        effective[replica] = 1.0 / np.sum(np.square(normalized), axis=1)
    return targets, weights, effective


def region_descriptor(
    values: np.ndarray, weights: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    values = np.atleast_2d(np.asarray(values, dtype=float))[:, indices]
    return values @ weights.T


def pooled_descriptor(
    left: np.ndarray,
    right: np.ndarray,
    left_log_mass: np.ndarray,
    right_log_mass: np.ndarray,
) -> np.ndarray:
    log_masses = np.stack([left_log_mass, right_log_mass])
    weights = np.exp(log_masses - logsumexp(log_masses, axis=0)[None, :])
    return left * weights[0][None, :] + right * weights[1][None, :]


def frame_variance_volume(
    values: np.ndarray,
    distance: np.ndarray,
    replicas: np.ndarray,
    k: int,
    shrinkage: float,
) -> dict[int, np.ndarray]:
    """Checkpoint-36 variance-volume coordinate for every sampled frame."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    raw: dict[int, np.ndarray] = {}
    reference = None
    output = {}
    for replica in REPLICAS:
        indices = np.flatnonzero(replicas == replica)
        local_distance = distance[np.ix_(indices, indices)]
        neighbours = nearest(local_distance, min(k, len(indices) - 1))
        variance = values[:, indices][:, neighbours].var(axis=2)
        if reference is None:
            reference = np.nanmedian(np.where(variance > 0, variance, np.nan), axis=1)
            reference = np.where(
                np.isfinite(reference) & (reference > 0), reference, 1.0
            )
        raw[replica] = variance
        output[replica] = 0.5 * np.mean(
            np.log(variance + shrinkage * reference[:, None]), axis=0
        )
    return output


def representations(
    logpf: np.ndarray, ca: np.ndarray
) -> dict[str, tuple[np.ndarray, str]]:
    result = cp36.work_representations(logpf)
    centered = ca - ca.mean(axis=1, keepdims=True)
    rg = np.sqrt(np.mean(np.sum(np.square(centered), axis=2), axis=1))
    result["rg"] = (rg, "l1")
    return result


def build_metric_matrices(
    source: dict[str, np.ndarray], query: dict[str, np.ndarray]
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    output = {}
    kinds = {
        "work_scale": "l1",
        "work_shape": "l1",
        "work_density": "l1",
        "work_opt": "l1",
        "pf_l1": "l1",
        "pf_l2": "l2",
        "rg": "l1",
    }
    for metric, kind in kinds.items():
        fit = pair_distance(source[metric], kind=kind)
        cross = pair_distance(query[metric], source[metric], kind=kind)
        output[(metric, "direct")] = (fit, cross)
    for name, operation in (
        ("work_fitting", lambda a, b: a + b),
        ("work_magnitude", lambda a, b: np.maximum(0.0, a - b)),
    ):
        scale = output[("work_scale", "direct")]
        other = output[
            ("work_density" if name == "work_fitting" else "work_shape", "direct")
        ]
        output[(name, "direct")] = tuple(
            operation(a, b) for a, b in zip(other, scale, strict=True)
        )
    for metric in VARIANCE_METRICS:
        output[(metric, "variance_magnitude")] = (
            pair_distance(source[f"{metric}__variance"]),
            pair_distance(query[f"{metric}__variance"], source[f"{metric}__variance"]),
        )
    return output


def labels_for_repeat(count: int, repeats: int, system: str) -> list[np.ndarray]:
    return [
        np.random.default_rng(stable_seed(SEED, system, "labels", repeat)).permutation(
            count
        )
        for repeat in range(repeats)
    ]


def evaluation_rows(
    *,
    system: str,
    unit: str,
    mode: str,
    metric: str,
    model: str,
    fit_distance: np.ndarray,
    query_distance: np.ndarray,
    structural_distance: np.ndarray,
    source_values: np.ndarray,
    target_values: np.ndarray,
    recovery_reference: np.ndarray,
    label_counts: tuple[int, ...],
    repeats: int,
    global_alpha: float | None,
    selection: str = "random",
    population: bool = False,
    source_coordinate: np.ndarray | None = None,
    query_coordinate: np.ndarray | None = None,
) -> tuple[list[dict], list[dict]]:
    n = len(target_values)
    counts = tuple(value for value in label_counts if 1 < value < n)
    if not counts:
        return [], []
    if selection == "random":
        orders = labels_for_repeat(n, repeats, system)
    else:
        orders = [maximin_order(fit_distance)]
    rows, examples = [], []
    for repeat, order in enumerate(orders):
        for known in counts:
            labels = np.sort(order[:known])
            held = np.setdiff1d(np.arange(n), labels, assume_unique=True)
            sparse, alpha, degenerate = anchored_prediction(
                fit_distance, query_distance, source_values, labels
            )
            all_alpha, full_alpha, _ = anchored_prediction(
                fit_distance, query_distance, source_values, np.arange(n)
            )
            mean_prediction = np.full(n, float(np.mean(source_values[labels])))
            nearest_label = labels[np.argmin(structural_distance[:, labels], axis=1)]
            nearest_prediction = source_values[nearest_label]
            rng = np.random.default_rng(
                stable_seed(
                    SEED, system, unit, mode, metric, model, repeat, known, "shuffle"
                )
            )
            permutation = rng.permutation(n)
            shuffled, _, _ = anchored_prediction(
                fit_distance[np.ix_(permutation, permutation)],
                query_distance[np.ix_(permutation, permutation)],
                source_values,
                labels,
            )
            methods = {
                "sparse_alpha": sparse,
                "full_alpha_ceiling": all_alpha,
                "label_mean": mean_prediction,
                "nearest_label": nearest_prediction,
                "shuffled_sparse": shuffled,
            }
            if global_alpha is not None and np.isfinite(global_alpha):
                methods["global_alpha"] = anchored_prediction(
                    fit_distance, query_distance, source_values, labels, global_alpha
                )[0]
            if source_coordinate is not None and query_coordinate is not None:
                x = np.asarray(source_coordinate, dtype=float)[labels]
                design = np.column_stack([np.ones(len(x)), x])
                if np.linalg.matrix_rank(design) == 2:
                    intercept, slope = np.linalg.lstsq(
                        design, source_values[labels], rcond=None
                    )[0]
                    methods["signed_regression"] = intercept + slope * np.asarray(
                        query_coordinate, dtype=float
                    )
            low, high = source_values[labels].min(), source_values[labels].max()
            strata = {
                "all": held,
                "interpolation": held[
                    (target_values[held] >= low) & (target_values[held] <= high)
                ],
                "extrapolation": held[
                    (target_values[held] < low) | (target_values[held] > high)
                ],
            }
            nearest_structural = structural_distance[:, labels].min(axis=1)
            scale = max(
                float(np.subtract(*np.quantile(target_values, [0.75, 0.25]))), 1e-12
            )
            for method, prediction in methods.items():
                for stratum, indices in strata.items():
                    if not len(indices):
                        continue
                    error = np.abs(target_values[indices] - prediction[indices])
                    record = {
                        "system_id": system,
                        "unit": unit,
                        "mode": mode,
                        "metric": metric,
                        "model": model,
                        "selection": selection,
                        "repeat": repeat,
                        "known": known,
                        "method": method,
                        "stratum": stratum,
                        "heldout": len(indices),
                        "alpha": alpha
                        if method == "sparse_alpha"
                        else (
                            full_alpha
                            if method == "full_alpha_ceiling"
                            else global_alpha
                            if method == "global_alpha"
                            else np.nan
                        ),
                        "degenerate_alpha": bool(degenerate),
                        "mae": float(error.mean()),
                        "nmae": float(error.mean() / scale),
                        "spearman": finite_spearman(
                            target_values[indices], prediction[indices]
                        ),
                        "mean_nearest_label_w1": float(
                            nearest_structural[indices].mean()
                        ),
                        "distribution_recovery": md_distribution_recovery(
                            target_values[indices],
                            prediction[indices],
                            recovery_reference,
                        ),
                        "recovery_pairs": len(indices) * (len(indices) - 1) // 2,
                        "population_tv": np.nan,
                    }
                    if population and stratum == "all":
                        combined = prediction.copy()
                        combined[labels] = source_values[labels]
                        predicted_mass = np.exp(combined - logsumexp(combined))
                        true_mass = np.exp(target_values - logsumexp(target_values))
                        record["population_tv"] = float(
                            0.5 * np.abs(predicted_mass - true_mass).sum()
                        )
                        record["top_cluster_correct"] = bool(
                            held[np.argmax(predicted_mass[held])]
                            == held[np.argmax(true_mass[held])]
                        )
                    rows.append(record)
            if (
                known == 2
                and repeat == 0
                and selection == "random"
                and model == "direct"
                and metric in {"work_scale", "work_opt", "pf_l1"}
            ):
                for index in held:
                    examples.append(
                        {
                            "system_id": system,
                            "unit": unit,
                            "mode": mode,
                            "metric": metric,
                            "model": model,
                            "index": int(index),
                            "target": float(target_values[index]),
                            "prediction": float(sparse[index]),
                            "nearest_label_w1": float(nearest_structural[index]),
                        }
                    )
    return rows, examples


def cluster_descriptor(
    values: np.ndarray,
    labels: np.ndarray,
    replicas: np.ndarray,
    selected_replicas: tuple[int, ...],
) -> np.ndarray:
    values = np.atleast_2d(np.asarray(values, dtype=float))
    output = np.empty((values.shape[0], labels.max() + 1), dtype=float)
    selected = np.isin(replicas, selected_replicas)
    for cluster in range(output.shape[1]):
        mask = selected & (labels == cluster)
        if not mask.any():
            mask = labels == cluster
        output[:, cluster] = values[:, mask].mean(axis=1)
    return output


def system_analysis(
    system: str,
    landmarks: int,
    label_counts: tuple[int, ...],
    repeats: int,
    global_alphas: dict[tuple[str, str], float],
    variance_parameters: dict[tuple[str, str, str], tuple[int, float]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    folder = GRAPH_ROOT / "systems" / system
    with np.load(folder / "frames.npz", allow_pickle=False) as archive:
        ca = np.asarray(archive["ca"], dtype=float)
        replicas = np.asarray(archive["replicas"], dtype=int)
        logpf = np.asarray(archive["logpf"], dtype=float)
    with np.load(folder / "graph_w1.npz", allow_pickle=False) as archive:
        structural = np.asarray(archive["distance"], dtype=float)
    count = min(landmarks, len(structural))
    landmark_indices = maximin_order(structural)[:count]
    a_indices = np.flatnonzero(replicas == 1)
    bandwidth = neighbour_bandwidth(structural[np.ix_(a_indices, a_indices)], 10)
    targets, weights, effective = local_targets_and_weights(
        structural, landmark_indices, replicas, bandwidth
    )
    raw = representations(logpf, ca)
    base_values = {name: value for name, (value, _) in raw.items()}
    variance_frames = {}
    for metric in VARIANCE_METRICS:
        key = (system, metric, "variance_magnitude")
        k, shrinkage = variance_parameters[key]
        variance_frames[metric] = frame_variance_volume(
            base_values[metric], structural, replicas, k, shrinkage
        )

    descriptors: dict[int, dict[str, np.ndarray]] = {
        replica: {} for replica in REPLICAS
    }
    for replica in REPLICAS:
        indices = np.flatnonzero(replicas == replica)
        for metric in (*DIRECT_BASE, "rg"):
            descriptors[replica][metric] = region_descriptor(
                base_values[metric], weights[replica], indices
            )
        for metric in VARIANCE_METRICS:
            descriptors[replica][f"{metric}__variance"] = region_descriptor(
                variance_frames[metric][replica],
                weights[replica],
                np.arange(len(indices)),
            )
    source_ab = {}
    for key in descriptors[1]:
        source_ab[key] = pooled_descriptor(
            descriptors[1][key], descriptors[2][key], targets[1], targets[2]
        )
    target_ab = logsumexp(np.stack([targets[1], targets[2]]), axis=0) - np.log(2.0)
    matrices_c = build_metric_matrices(descriptors[3], descriptors[3])
    matrices_cross = build_metric_matrices(source_ab, descriptors[3])
    center_w1 = structural[np.ix_(landmark_indices, landmark_indices)]
    matrices_c[("w1", "direct")] = (center_w1, center_w1)
    matrices_cross[("w1", "direct")] = (center_w1, center_w1)

    local_rows, example_rows = [], []
    for mode, source_values, recovery_reference, matrices in (
        ("within_C", targets[3], targets[1], matrices_c),
        ("AB_to_C", target_ab, target_ab, matrices_cross),
    ):
        for (metric, model), (fit_distance, query_distance) in matrices.items():
            for selection in ("random", "maximin"):
                rows, examples = evaluation_rows(
                    system=system,
                    unit="local_neighbourhood",
                    mode=mode,
                    metric=metric,
                    model=model,
                    fit_distance=fit_distance,
                    query_distance=query_distance,
                    structural_distance=center_w1,
                    source_values=source_values,
                    target_values=targets[3],
                    recovery_reference=recovery_reference,
                    label_counts=label_counts,
                    repeats=repeats,
                    global_alpha=global_alphas.get((metric, model)),
                    selection=selection,
                    source_coordinate=(
                        descriptors[3]["work_scale"].ravel()
                        if metric == "work_scale"
                        and model == "direct"
                        and mode == "within_C"
                        else source_ab["work_scale"].ravel()
                        if metric == "work_scale" and model == "direct"
                        else None
                    ),
                    query_coordinate=(
                        descriptors[3]["work_scale"].ravel()
                        if metric == "work_scale" and model == "direct"
                        else None
                    ),
                )
                local_rows.extend(rows)
                example_rows.extend(examples)

    cluster_rows = []
    with np.load(folder / "clusters_w1.npz", allow_pickle=False) as archive:
        cluster_labels = np.asarray(archive["labels"], dtype=int)
        cluster_counts = list(np.asarray(archive["counts"], dtype=int))
    for k_clusters in CLUSTER_COUNTS:
        labels = cluster_labels[:, cluster_counts.index(k_clusters)]
        probabilities = {}
        for replica in REPLICAS:
            counts = np.bincount(labels[replicas == replica], minlength=k_clusters)
            probabilities[replica] = (counts + JEFFREYS) / (
                counts.sum() + JEFFREYS * k_clusters
            )
        source_probability = (
            np.bincount(labels[np.isin(replicas, (1, 2))], minlength=k_clusters)
            + JEFFREYS
        ) / (np.sum(np.isin(replicas, (1, 2))) + JEFFREYS * k_clusters)
        cluster_source, cluster_query = {}, {}
        for metric in (*DIRECT_BASE, "rg"):
            cluster_source[metric] = cluster_descriptor(
                base_values[metric], labels, replicas, (1, 2)
            )
            cluster_query[metric] = cluster_descriptor(
                base_values[metric], labels, replicas, (3,)
            )
        for metric in VARIANCE_METRICS:
            concatenated = np.empty(len(replicas), dtype=float)
            for replica in REPLICAS:
                concatenated[replicas == replica] = variance_frames[metric][replica]
            cluster_source[f"{metric}__variance"] = cluster_descriptor(
                concatenated, labels, replicas, (1, 2)
            )
            cluster_query[f"{metric}__variance"] = cluster_descriptor(
                concatenated, labels, replicas, (3,)
            )
        cluster_c = build_metric_matrices(cluster_query, cluster_query)
        cluster_cross = build_metric_matrices(cluster_source, cluster_query)
        cluster_w1 = np.empty((k_clusters, k_clusters), dtype=float)
        for left in range(k_clusters):
            for right in range(k_clusters):
                block = structural[np.ix_(labels == left, labels == right)]
                cluster_w1[left, right] = float(block.mean())
        np.fill_diagonal(cluster_w1, 0.0)
        cluster_c[("w1", "direct")] = (cluster_w1, cluster_w1)
        cluster_cross[("w1", "direct")] = (cluster_w1, cluster_w1)
        counts_for_k = (2, 3, 4) if k_clusters == 5 else (2, 3, 5, 8)
        for mode, source_values, recovery_reference, matrices in (
            (
                "within_C",
                np.log(probabilities[3]),
                np.log(probabilities[1]),
                cluster_c,
            ),
            (
                "AB_to_C",
                np.log(source_probability),
                np.log(source_probability),
                cluster_cross,
            ),
        ):
            target_values = np.log(probabilities[3])
            for (metric, model), (fit_distance, query_distance) in matrices.items():
                rows, _ = evaluation_rows(
                    system=system,
                    unit=f"cluster_k{k_clusters}",
                    mode=mode,
                    metric=metric,
                    model=model,
                    fit_distance=fit_distance,
                    query_distance=query_distance,
                    structural_distance=cluster_w1,
                    source_values=source_values,
                    target_values=target_values,
                    recovery_reference=recovery_reference,
                    label_counts=counts_for_k,
                    repeats=repeats,
                    global_alpha=global_alphas.get((metric, model)),
                    selection="random",
                    population=True,
                    source_coordinate=(
                        cluster_query["work_scale"].ravel()
                        if metric == "work_scale"
                        and model == "direct"
                        and mode == "within_C"
                        else cluster_source["work_scale"].ravel()
                        if metric == "work_scale" and model == "direct"
                        else None
                    ),
                    query_coordinate=(
                        cluster_query["work_scale"].ravel()
                        if metric == "work_scale" and model == "direct"
                        else None
                    ),
                )
                target_counts = np.bincount(labels[replicas == 3], minlength=k_clusters)
                for record in rows:
                    record["minimum_target_cluster_count"] = int(target_counts.min())
                    record["zero_target_clusters"] = int(np.sum(target_counts == 0))
                cluster_rows.extend(rows)
    audit = {
        "system_id": system,
        "frames": len(replicas),
        "landmarks": count,
        "bandwidth_angstrom": bandwidth,
        "minimum_effective_neighbours_C": float(effective[3].min()),
        "median_effective_neighbours_C": float(np.median(effective[3])),
        "source_complete_sha256": cp36.digest(folder / "complete.json"),
    }
    return (
        pd.DataFrame(local_rows),
        pd.DataFrame(cluster_rows),
        pd.DataFrame(example_rows),
        audit,
    )


def save_system(arguments) -> str:
    (
        system,
        landmarks,
        label_counts,
        repeats,
        global_alphas,
        variance_parameters,
        parts,
    ) = arguments
    local, clusters, examples, audit = system_analysis(
        system, landmarks, label_counts, repeats, global_alphas, variance_parameters
    )
    local = aggregate_repetitions(local)
    clusters = aggregate_repetitions(clusters)
    atomic_parquet(local, parts / f"{system}.local.parquet")
    atomic_parquet(clusters, parts / f"{system}.clusters.parquet")
    atomic_parquet(examples, parts / f"{system}.examples.parquet")
    temporary = parts / f"{system}.audit.json.tmp"
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    temporary.replace(parts / f"{system}.audit.json")
    return system


def aggregate_repetitions(frame: pd.DataFrame) -> pd.DataFrame:
    """Average Monte Carlo label selections before systems become replicates."""
    keys = [
        "system_id",
        "unit",
        "mode",
        "metric",
        "model",
        "selection",
        "known",
        "method",
        "stratum",
    ]
    aggregations = {
        "repetitions": ("repeat", "nunique"),
        "heldout": ("heldout", "mean"),
        "alpha": ("alpha", "mean"),
        "degenerate_alpha_fraction": ("degenerate_alpha", "mean"),
        "mae": ("mae", "mean"),
        "nmae": ("nmae", "mean"),
        "spearman": ("spearman", "mean"),
        "mean_nearest_label_w1": ("mean_nearest_label_w1", "mean"),
        "distribution_recovery": ("distribution_recovery", "mean"),
        "recovery_pairs": ("recovery_pairs", "mean"),
        "population_tv": ("population_tv", "mean"),
    }
    if "top_cluster_correct" in frame:
        aggregations["top_cluster_accuracy"] = ("top_cluster_correct", "mean")
    if "minimum_target_cluster_count" in frame:
        aggregations["minimum_target_cluster_count"] = (
            "minimum_target_cluster_count",
            "first",
        )
        aggregations["zero_target_clusters"] = ("zero_target_clusters", "first")
    return frame.groupby(keys, as_index=False, dropna=False).agg(**aggregations)


def paired_interval(
    values: np.ndarray, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def data_requirements(local: pd.DataFrame) -> pd.DataFrame:
    """Estimate label requirements from MD distribution recovery, not MAE."""
    block = local[
        (local.selection == "random")
        & (local.stratum == "all")
        & local.method.isin(("sparse_alpha", "label_mean", "shuffled_sparse"))
    ]
    system = block.groupby(
        ["system_id", "mode", "metric", "model", "known", "method"], as_index=False
    ).agg(
        distribution_recovery=("distribution_recovery", "mean"),
        nmae=("nmae", "mean"),
        spearman=("spearman", "mean"),
    )
    rows = []
    for key, group in system.groupby(["mode", "metric", "model"]):
        pivot = group.pivot_table(
            index=["system_id", "known"],
            columns="method",
            values="distribution_recovery",
        ).dropna()
        nmae = group.pivot_table(
            index=["system_id", "known"], columns="method", values="nmae"
        )
        ranks = group.pivot_table(
            index=["system_id", "known"], columns="method", values="spearman"
        )
        counts = sorted(pivot.index.get_level_values("known").unique())
        summaries = {}
        for known in counts:
            current = pivot.xs(known, level="known")
            current_nmae = nmae.xs(known, level="known").reindex(current.index)
            current_rank = ranks.xs(known, level="known").reindex(current.index)
            mean_delta = current.sparse_alpha - current.label_mean
            shuffle_delta = current.sparse_alpha - current.shuffled_sparse
            rank_delta = current_rank.sparse_alpha - current_rank.shuffled_sparse
            summaries[known] = {
                "mean_recovery": float(current.sparse_alpha.mean()),
                "median_recovery": float(current.sparse_alpha.median()),
                "baseline_recovery": float(current.label_mean.mean()),
                "baseline_median_recovery": float(current.label_mean.median()),
                "median_nmae_diagnostic": float(current_nmae.sparse_alpha.median()),
                "median_spearman": float(current_rank.sparse_alpha.median()),
                "delta_vs_mean": float(np.median(mean_delta)),
                "delta_vs_shuffle": float(np.median(shuffle_delta)),
                "mean_ci": paired_interval(
                    mean_delta, stable_seed(*key, known, "mean")
                ),
                "shuffle_ci": paired_interval(
                    shuffle_delta, stable_seed(*key, known, "shuffle")
                ),
                "rank_delta_vs_shuffle": float(np.median(rank_delta)),
                "rank_ci": paired_interval(
                    rank_delta, stable_seed(*key, known, "rank_shuffle")
                ),
                "systems": len(current),
            }
        maximum = max(counts)
        maximum_improvement = (
            summaries[maximum]["median_recovery"]
            - summaries[maximum]["baseline_median_recovery"]
        )
        required = np.nan
        moderate = np.nan
        for known in counts:
            later = [value for value in counts if value >= known]
            sustained = all(
                summaries[value]["median_recovery"]
                - summaries[value]["baseline_median_recovery"]
                >= 0.9 * maximum_improvement
                for value in later
            )
            if (
                maximum_improvement > 0
                and summaries[known]["mean_ci"][0] > 0
                and summaries[known]["shuffle_ci"][0] > 0
                and summaries[known]["rank_ci"][0] > 0
                and sustained
            ):
                required = known
                break
        for known in counts:
            if (
                summaries[known]["median_spearman"] >= 0.5
                and summaries[known]["mean_ci"][0] > 0
                and summaries[known]["shuffle_ci"][0] > 0
                and summaries[known]["rank_ci"][0] > 0
            ):
                moderate = known
                break
        for known in counts:
            item = summaries[known]
            rows.append(
                {
                    "mode": key[0],
                    "metric": key[1],
                    "model": key[2],
                    "known": known,
                    "mean_recovery": item["mean_recovery"],
                    "median_recovery": item["median_recovery"],
                    "baseline_recovery": item["baseline_recovery"],
                    "baseline_median_recovery": item["baseline_median_recovery"],
                    "median_nmae_diagnostic": item["median_nmae_diagnostic"],
                    "median_spearman": item["median_spearman"],
                    "delta_vs_mean": item["delta_vs_mean"],
                    "delta_vs_mean_ci_low": item["mean_ci"][0],
                    "delta_vs_mean_ci_high": item["mean_ci"][1],
                    "delta_vs_shuffle": item["delta_vs_shuffle"],
                    "delta_vs_shuffle_ci_low": item["shuffle_ci"][0],
                    "delta_vs_shuffle_ci_high": item["shuffle_ci"][1],
                    "spearman_delta_vs_shuffle": item["rank_delta_vs_shuffle"],
                    "spearman_delta_vs_shuffle_ci_low": item["rank_ci"][0],
                    "spearman_delta_vs_shuffle_ci_high": item["rank_ci"][1],
                    "systems": item["systems"],
                    "required_labels": required,
                    "moderate_localization_labels": moderate,
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curves(requirements: pd.DataFrame, destination: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(15, 10), sharex=True, constrained_layout=True
    )
    for column, mode in enumerate(("within_C", "AB_to_C")):
        current = requirements[requirements["mode"] == mode]
        for (metric, model), block in current.groupby(["metric", "model"]):
            label = f"{metric}:{'VM' if model == 'variance_magnitude' else 'direct'}"
            style = "--" if model == "variance_magnitude" else "-"
            axes[0, column].plot(
                block.known,
                100.0 * block.median_recovery,
                marker="o",
                linestyle=style,
                label=label,
            )
            axes[1, column].plot(
                block.known,
                block.median_spearman,
                marker="o",
                linestyle=style,
                label=label,
            )
        for row in range(2):
            axis = axes[row, column]
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted(current.known.unique()))
            axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            axis.grid(alpha=0.2)
        axes[0, column].set_ylim(0, 102)
        axes[0, column].set_title(mode.replace("_", " "))
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].set_ylim(-1, 1)
        axes[1, column].set_xlabel("Known neighbourhood populations")
    axes[0, 0].set_ylabel("MD distribution recovery (%)")
    axes[1, 0].set_ylabel("Held-out neighbourhood Spearman rho")
    axes[0, 1].legend(fontsize=7, ncol=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Checkpoint 39: recovery and correct local assignment")
    fig.savefig(
        destination / "sparse_label_learning_curves.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


def plot_extrapolation(local: pd.DataFrame, destination: Path) -> None:
    selected = local[
        (local.selection == "random")
        & (local.method == "sparse_alpha")
        & (local.stratum == "all")
        & local.metric.isin(("work_scale", "work_opt", "pf_l1"))
        & (local.model == "direct")
        & local.known.isin((2, 8, 32))
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5), sharey=True, constrained_layout=True
    )
    for axis, mode in zip(axes, ("within_C", "AB_to_C"), strict=True):
        current = selected[selected["mode"] == mode]
        for (metric, known), block in current.groupby(["metric", "known"]):
            axis.scatter(
                block.mean_nearest_label_w1,
                100.0 * block.distribution_recovery,
                s=24,
                alpha=0.65,
                label=f"{metric}, m={known}",
            )
        axis.set_xlabel("Mean structural W1 to nearest known region (Å)")
        axis.set_ylim(0, 102)
        axis.set_title(mode.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("MD distribution recovery (%)")
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("MD population-change recovery versus distance from known regions")
    fig.savefig(
        destination / "error_vs_label_distance.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


def plot_clusters(clusters: pd.DataFrame, destination: Path) -> None:
    block = clusters[
        (clusters.selection == "random")
        & (clusters.method == "sparse_alpha")
        & (clusters.stratum == "all")
    ]
    summary = block.groupby(
        ["unit", "mode", "metric", "model", "known"], as_index=False
    ).distribution_recovery.median()
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharey=True, constrained_layout=True
    )
    for axis, ((unit, mode), current) in zip(
        axes.flat, summary.groupby(["unit", "mode"]), strict=False
    ):
        for (metric, model), line in current.groupby(["metric", "model"]):
            if model == "direct" or metric in {"work_scale", "pf_l1"}:
                axis.plot(
                    line.known,
                    100.0 * line.distribution_recovery,
                    marker="o",
                    linestyle="--" if model != "direct" else "-",
                    label=f"{metric}:{model}",
                )
        axis.set_title(f"{unit}, {mode}")
        axis.set_xlabel("Known clusters")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("MD distribution recovery (%)")
    axes[1, 0].set_ylabel("MD distribution recovery (%)")
    axes[0, 1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Forced structural-cluster population sensitivity")
    fig.savefig(
        destination / "cluster_population_recovery.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


def plot_two_label_examples(
    examples: pd.DataFrame, local: pd.DataFrame, destination: Path
) -> None:
    score = (
        local[
            (local.unit == "local_neighbourhood")
            & (local["mode"] == "within_C")
            & (local.metric == "work_opt")
            & (local.model == "direct")
            & (local.method == "sparse_alpha")
            & (local.selection == "random")
            & (local.known == 2)
            & (local.stratum == "all")
        ]
        .groupby("system_id")
        .mae.mean()
        .sort_values()
    )
    if score.empty:
        return
    system = score.index[len(score) // 2]
    block = examples[
        (examples.system_id == system)
        & (examples.unit == "local_neighbourhood")
        & (examples["mode"] == "within_C")
        & (examples.metric == "work_opt")
    ]
    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.scatter(
        block.target, block.prediction, c=block.nearest_label_w1, cmap="viridis", s=28
    )
    low = min(block.target.min(), block.prediction.min())
    high = max(block.target.max(), block.prediction.max())
    axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set_xlabel("True held-out log occupancy")
    axis.set_ylabel("Prediction from two known regions")
    axis.set_title(f"Median-system two-label example: {system}, Work Opt")
    fig.savefig(destination / "two_label_examples.png", dpi=180)
    plt.close(fig)


def report(destination: Path, systems: list[str]) -> None:
    local = pd.concat(
        [
            pd.read_parquet(destination / "parts" / f"{system}.local.parquet")
            for system in systems
        ],
        ignore_index=True,
    )
    clusters = pd.concat(
        [
            pd.read_parquet(destination / "parts" / f"{system}.clusters.parquet")
            for system in systems
        ],
        ignore_index=True,
    )
    examples = pd.concat(
        [
            pd.read_parquet(destination / "parts" / f"{system}.examples.parquet")
            for system in systems
        ],
        ignore_index=True,
    )
    atomic_parquet(local, destination / "local_extrapolation_results.parquet")
    atomic_parquet(clusters, destination / "cluster_extrapolation_results.parquet")
    atomic_parquet(examples, destination / "local_two_label_predictions.parquet")
    requirements = data_requirements(local)
    requirements.to_csv(destination / "data_requirements.csv", index=False)
    plot_learning_curves(requirements, destination)
    plot_extrapolation(local, destination)
    plot_clusters(clusters, destination)
    plot_two_label_examples(examples, local, destination)
    two = requirements[requirements.known == 2].copy()
    two["beats_mean"] = two.delta_vs_mean_ci_low > 0
    two["beats_shuffle"] = two.delta_vs_shuffle_ci_low > 0
    two["localizes_better_than_shuffle"] = two.spearman_delta_vs_shuffle_ci_low > 0
    reached = requirements.dropna(subset=["required_labels"]).drop_duplicates(
        ["mode", "metric", "model"]
    )
    direct_focus = local[
        (local.selection == "random")
        & (local.stratum == "all")
        & (local.model == "direct")
        & local.metric.isin(("pf_l1", "work_opt", "work_scale"))
        & local.known.isin((2, 32))
    ]
    method_medians = (
        direct_focus.groupby(["mode", "metric", "known", "method"], as_index=False)
        .agg(
            distribution_recovery=("distribution_recovery", "median"),
            spearman=("spearman", "median"),
            nmae_diagnostic=("nmae", "median"),
        )
        .to_dict(orient="records")
    )
    range_summary = (
        local[
            (local.selection == "random")
            & (local.method == "sparse_alpha")
            & (local.model == "direct")
            & local.metric.isin(("pf_l1", "work_opt", "work_shape", "work_scale"))
            & local.known.isin((2, 32))
            & local.stratum.isin(("interpolation", "extrapolation"))
        ]
        .groupby(["mode", "metric", "known", "stratum"], as_index=False)
        .agg(
            distribution_recovery=("distribution_recovery", "median"),
            spearman=("spearman", "median"),
        )
        .to_dict(orient="records")
    )
    payload = {
        "checkpoint": 39,
        "systems": len(systems),
        "primary_target": (
            "MD distribution recovery of held-out pairwise absolute log-occupancy changes"
        ),
        "validation_modes": ["within_C", "AB_to_C"],
        "interpretation_limit": "structural clusters are forced subdivisions, not metastable states",
        "decision_rule": (
            "required_labels is the first detectable-signal count: paired median "
            "recovery advantage over both "
            "label-mean and shuffled controls has a positive 95% bootstrap interval, "
            "whose Spearman advantage over shuffled geometry is also positive, "
            "and retains at least 90% of the maximum-label recovery gain. "
            "moderate_localization_labels additionally requires median Spearman >=0.5"
        ),
        "two_label_results": two.to_dict(orient="records"),
        "metrics_reaching_rule": reached[
            [
                "mode",
                "metric",
                "model",
                "required_labels",
                "moderate_localization_labels",
            ]
        ].to_dict(orient="records"),
        "selected_method_medians": method_medians,
        "interpolation_extrapolation_medians": range_summary,
        "minimum_label_results": requirements.drop_duplicates(
            ["mode", "metric", "model"]
        )[
            [
                "mode",
                "metric",
                "model",
                "required_labels",
                "moderate_localization_labels",
            ]
        ].to_dict(orient="records"),
    }
    (destination / "checkpoint39_report.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("analyse", "report", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--label-counts", default="2,4,8,16,32")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        args.phase = "report"
    label_counts = tuple(sorted({int(value) for value in args.label_counts.split(",")}))
    if args.workers < 1 or args.repeats < 1 or args.landmarks < 3:
        parser.error(
            "workers/repeats must be positive and landmarks must be at least three"
        )
    rows = cp36.selected_rows("pilot", cp36.SINGLE_SYSTEM)
    if args.limit is not None:
        rows = rows[: args.limit]
    systems = [row["system_id"] for row in rows]
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    global_frame = pd.read_csv(CP37 / "global_alpha_parameters.csv")
    global_alphas = {
        (row.metric, row.model): float(row.global_alpha)
        for row in global_frame.itertuples()
    }
    fits = pd.read_parquet(CP36 / "corrected_variance_fits.parquet")
    local = fits[fits.model == "variance_magnitude"]
    variance_parameters = {
        (row.system_id, row.metric, row.model): (int(row.k), float(row.shrinkage))
        for row in local.itertuples()
    }
    if args.phase in ("analyse", "all"):

        def current_part(system: str) -> bool:
            required = [
                parts / f"{system}.local.parquet",
                parts / f"{system}.clusters.parquet",
                parts / f"{system}.examples.parquet",
                parts / f"{system}.audit.json",
            ]
            if not all(path.exists() for path in required):
                return False
            try:
                return all(
                    "distribution_recovery" in pd.read_parquet(path).columns
                    for path in required[:2]
                )
            except Exception:
                return False

        pending = [system for system in systems if not current_part(system)]
        tasks = [
            (
                system,
                args.landmarks,
                label_counts,
                args.repeats,
                global_alphas,
                variance_parameters,
                parts,
            )
            for system in pending
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(save_system, task): task[0] for task in tasks}
            for index, future in enumerate(as_completed(futures), 1):
                print(f"[{index}/{len(tasks)}] {future.result()} complete", flush=True)
    if args.phase in ("report", "all"):
        report(destination, systems)
        print(f"report: {destination / 'checkpoint39_report.yaml'}", flush=True)


if __name__ == "__main__":
    main()
