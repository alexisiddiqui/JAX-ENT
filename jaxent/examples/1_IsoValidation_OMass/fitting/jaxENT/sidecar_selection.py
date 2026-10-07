"""Shared fixed closed-coordinate validation criterion for sidecar selection."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

SELECTION_METRIC = "val_closed_sigma_mse"
SELECTION_POLICY = "closed_coordinate_sigma_mse_alpha0_over_native_convergence_states"


def select_best_rows(scores, metric=SELECTION_METRIC):
    if scores.empty:
        return scores.copy()
    if metric not in {SELECTION_METRIC, "val_mse"}:
        raise ValueError(f"Unsupported sidecar selector: {metric}")
    if metric not in scores:
        raise ValueError(f"Missing {metric}; rescore old histories before selecting")
    rows = []
    for _, group in scores.groupby("run_id", sort=False):
        tie_columns = ["step", "candidate_index"] if "candidate_index" in scores else ["convergence_rank"]
        valid = group[np.isfinite(group[metric])].sort_values(
            [metric, *tie_columns], kind="stable")
        if not valid.empty:
            rows.append(valid.iloc[0])
    result = pd.DataFrame(rows, columns=scores.columns).reset_index(drop=True)
    result["selection_metric"] = metric
    return result


@lru_cache(maxsize=16)
def _trajectory_dir(output_dir):
    import run_sigma_source_sidecar as source
    manifest = Path(output_dir) / "manifest.json"
    settings = json.loads(manifest.read_text()).get("settings", {}) if manifest.exists() else {}
    return str(settings.get("trajectory_dir", source.DEFAULT_TRAJECTORY_DIR))


@lru_cache(maxsize=8)
def _full_precision(ensemble, features_dir, clustering_dir, trajectory_dir):
    import run_sigma_source_sidecar as source
    features, topology = source.shrinkage.load_features(features_dir, ensemble)
    assignments = source.shrinkage.load_cluster_assignments(clustering_dir, ensemble)
    coordinates = source.aligned_ca_coordinates(ensemble, topology, Path(trajectory_dir))
    if coordinates.shape[:2] != (len(assignments), features.features_shape[0]):
        raise ValueError("Closed selection coordinates/features do not match")
    weights = source.population_weights(assignments, "closed")
    covariance = source.weighted_coordinate_covariance(coordinates, weights)
    arrays, _ = source.regularize_sigma(covariance, weights, 0., 1e8)
    return arrays["Sigma_inv_normalized"]


@lru_cache(maxsize=64)
def _validation_precision(ensemble, features_dir, clustering_dir, datasplit_dir,
                          split_type, split_idx, trajectory_dir):
    import run_sigma_source_sidecar as source
    full = _full_precision(ensemble, features_dir, clustering_dir, trajectory_dir)
    _, val = source.shrinkage.load_split(datasplit_dir, split_type, split_idx)
    indices = np.asarray([point.top.fragment_index for point in val], dtype=int)
    precision = full[np.ix_(indices, indices)]
    return precision * len(indices) / np.trace(precision)


def closed_validation_mse(spec, predicted, observed):
    """Native Sigma-MSE: trace-normalized held-out precision; no KL term."""
    precision = _validation_precision(spec.ensemble, str(spec.features_dir),
        str(spec.clustering_dir), str(spec.datasplit_dir), spec.split_type, spec.split_idx,
        _trajectory_dir(str(spec.output_dir)))
    residual = np.asarray(predicted) - np.asarray(observed)
    if residual.ndim != 2 or precision.shape != (len(residual), len(residual)):
        raise ValueError("Closed validation precision/observation shape mismatch")
    return float(.5*np.einsum("pt,pq,qt->", residual, precision, residual)/residual.size)
