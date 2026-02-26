from __future__ import annotations

from typing import Dict

import pandas as pd
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


def analyze_conformational_recovery(
    clustering_results: Dict[str, pd.DataFrame],
    results_dict: Dict,
    target_ratios: Dict[str, float],
    state_mapping: Dict[int, str],
    metric: str = "jsd",
    best_step_only: bool = False,
) -> pd.DataFrame:
    """Analyze conformational ratio recovery.
    Walks `results_dict` to extract histories and compute recovery.
    """
    recovery_data = []

    def walk_dict(d, context_keys):
        """Recursively walk the nested results dictionary to find histories."""
        if not isinstance(d, dict):
            return [(context_keys, d)]
        items = []
        for k, v in d.items():
            items.extend(walk_dict(v, context_keys + [k]))
        return items

    flattened = walk_dict(results_dict, [])

    for context_keys, history in flattened:
        if history is None or not hasattr(history, "states") or not history.states:
            continue

        if len(context_keys) == 5:
            split_type, ensemble, loss_name, maxent_val, split_idx = context_keys
            bv_reg_fn, bv_reg_val = "N/A", np.nan
        elif len(context_keys) == 7:
            split_type, ensemble, loss_name, bv_reg_fn, maxent_val, bv_reg_val, split_idx = context_keys
        else:
            continue

        cluster_df = clustering_results.get(ensemble)
        if cluster_df is None:
            continue

        if isinstance(cluster_df, pd.DataFrame) and "cluster_label" in cluster_df.columns:
            cluster_assignments = cluster_df["cluster_label"]
        elif isinstance(cluster_df, dict) and "cluster_assignments" in cluster_df:
            cluster_assignments = cluster_df["cluster_assignments"]
        else:
            continue

        n_frames = len(cluster_assignments)
        states_to_check = enumerate(history.states) if not best_step_only else [(len(history.states)-1, history.states[-1])]

        for step_idx, state in states_to_check:
            if not hasattr(state, "params") or not hasattr(state.params, "frame_weights") or state.params.frame_weights is None:
                continue

            frame_weights = np.array(state.params.frame_weights)
            if len(frame_weights) != n_frames or np.sum(frame_weights) == 0:
                continue

            normalized_weights = frame_weights / np.sum(frame_weights)

            js_div, current_props = calculate_recovery_JSD(
                cluster_assignments, normalized_weights, target_ratios, state_mapping
            )

            recovery_pct = (1.0 - np.sqrt(js_div)) * 100.0 if not np.isnan(js_div) else np.nan

            entry = {
                "ensemble": ensemble,
                "split_type": split_type,
                "loss_function": loss_name,
                "split": split_idx,
                "maxent_value": maxent_val,
                "convergence_step": step_idx + 1 if not best_step_only else len(history.states),
                "js_divergence": js_div if not np.isnan(js_div) else 0.0,
                "js_distance": np.sqrt(js_div) if not np.isnan(js_div) else 0.0,
                "recovery_percent": recovery_pct if not np.isnan(recovery_pct) else 0.0,
                "total_frames": n_frames,
            }
            if len(context_keys) == 7:
                entry["bv_reg_function"] = bv_reg_fn
                entry["bv_reg_value"] = bv_reg_val

            for state_name in target_ratios:
                entry[f"{state_name}_current"] = current_props.get(state_name, 0.0)
                entry[f"{state_name}_target"] = target_ratios.get(state_name, 0.0)

            recovery_data.append(entry)

    # Always compute Original (Unweighted) baselines per ensemble
    added_unweighted = set()
    for ensemble, cluster_df in clustering_results.items():
        if ensemble in added_unweighted:
            continue
        added_unweighted.add(ensemble)

        if isinstance(cluster_df, pd.DataFrame) and "cluster_label" in cluster_df.columns:
            cluster_assignments = cluster_df["cluster_label"]
        elif isinstance(cluster_df, dict) and "cluster_assignments" in cluster_df:
            cluster_assignments = cluster_df["cluster_assignments"]
        else:
            continue

        n_frames = len(cluster_assignments)
        if n_frames == 0:
            continue

        uniform_weights = np.ones(n_frames) / n_frames
        js_div, current_props = calculate_recovery_JSD(
            cluster_assignments, uniform_weights, target_ratios, state_mapping
        )
        recovery_pct = (1.0 - np.sqrt(js_div)) * 100.0 if not np.isnan(js_div) else np.nan

        entry = {
            "ensemble": ensemble,
            "split_type": "N/A",
            "loss_function": "Original",
            "split": "N/A",
            "maxent_value": 0.0,
            "convergence_step": 0,
            "js_divergence": js_div if not np.isnan(js_div) else 0.0,
            "js_distance": np.sqrt(js_div) if not np.isnan(js_div) else 0.0,
            "recovery_percent": recovery_pct if not np.isnan(recovery_pct) else 0.0,
            "total_frames": n_frames,
        }
        for state_name in target_ratios:
            entry[f"{state_name}_current"] = current_props.get(state_name, 0.0)
            entry[f"{state_name}_target"] = target_ratios.get(state_name, 0.0)

        recovery_data.append(entry)

    return pd.DataFrame(recovery_data)
