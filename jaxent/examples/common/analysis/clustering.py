from __future__ import annotations

from typing import Dict

import numpy as np


def calculate_cluster_ratios(
    cluster_assignments: np.ndarray,
    frame_weights: np.ndarray | None = None,
) -> Dict[str, float]:
    """Weighted cluster population ratios.

    Parameters
    ----------
    cluster_assignments:
        Integer array of cluster labels.
    frame_weights:
        Optional per-frame weights (normalised internally).
    """
    if frame_weights is None:
        frame_weights = np.ones(len(cluster_assignments))
    if np.sum(frame_weights) == 0:
        return {}

    ratios: Dict[str, float] = {}
    unique = np.unique(cluster_assignments)
    for cluster in unique:
        if cluster >= 0:
            mask = cluster_assignments == cluster
            ratios[f"cluster_{cluster}"] = float(np.sum(frame_weights[mask]))
    return ratios


def calculate_recovery_JSD(
    cluster_assignments,
    weights: np.ndarray,
    target_ratios: Dict[str, float],
    state_mapping: Dict[int, str],
) -> tuple[float, Dict[str, float]]:
    """Jensen–Shannon divergence between observed and target state proportions.

    Returns ``(JSD, current_proportions)``. Recovery % = ``(1 - sqrt(JSD)) * 100``.
    """
    # Invert mapping: state → cluster ids
    state_to_clusters: Dict[str, list] = {}
    for cluster_id, state_name in state_mapping.items():
        state_to_clusters.setdefault(state_name, []).append(cluster_id)

    current_proportions: Dict[str, float] = {s: 0.0 for s in target_ratios}
    for state_name, cluster_ids in state_to_clusters.items():
        if state_name in current_proportions:
            state_mask = cluster_assignments.isin(cluster_ids) if hasattr(cluster_assignments, "isin") else np.isin(cluster_assignments, cluster_ids)
            current_proportions[state_name] = float(np.sum(weights[np.asarray(state_mask)]))

    states = list(target_ratios.keys())
    P = np.array([current_proportions.get(s, 0.0) for s in states], dtype=float)
    Q = np.array([target_ratios.get(s, 0.0) for s in states], dtype=float)

    sumP, sumQ = P.sum(), Q.sum()
    if sumP > 0:
        P /= sumP
    else:
        return np.nan, current_proportions
    if sumQ > 0:
        Q /= sumQ
    else:
        return np.nan, current_proportions

    M = 0.5 * (P + Q)

    def _kld(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    jsd = 0.5 * (_kld(P, M) + _kld(Q, M))
    return float(jsd), current_proportions


def calculate_recovery_percentage(
    cluster_assignments: np.ndarray,
    weights: np.ndarray,
    target_ratios: Dict[str, float],
    state_mapping: Dict[int, str],
) -> float:
    """Recovery % = ``(1 - sqrt(JSD)) * 100``."""
    jsd, _ = calculate_recovery_JSD(cluster_assignments, weights, target_ratios, state_mapping)
    if np.isnan(jsd):
        return np.nan
    return float((1.0 - np.sqrt(jsd)) * 100.0)


def calculate_dMSE(
    pred_uptake: np.ndarray, prior_uptake: np.ndarray, experimental_uptake: np.ndarray
) -> float:
    """Compute the change in MSE (improvement) over the prior model.

    Negative values indicate fit improvement.
    """
    # Use nanmean to handle potential missing experimental data
    pred_uptake = np.asarray(pred_uptake)
    prior_uptake = np.asarray(prior_uptake)
    experimental_uptake = np.asarray(experimental_uptake)

    mse_pred = np.nanmean(np.abs(pred_uptake - experimental_uptake) ** 2)
    mse_prior = np.nanmean(np.abs(prior_uptake - experimental_uptake) ** 2)
    return float(mse_pred - mse_prior)
