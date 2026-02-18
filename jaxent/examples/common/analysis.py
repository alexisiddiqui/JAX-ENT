"""
Shared analysis functions for JAX-ENT example scripts.

Consolidates the most-duplicated analysis computations. All functions are pure
numpy/pandas — no JAX-ENT library imports needed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import ScoringConfig


# ---------------------------------------------------------------------------
# Statistical helpers  (most-duplicated functions across all scripts)
# ---------------------------------------------------------------------------


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL divergence KL(p||q) between two distributions.

    Distributions are normalised internally. *eps* prevents ``log(0)``.

    .. note:: This is the **single most duplicated function** in the examples
       codebase — 11 active copies.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / np.sum(p)
    q = q / np.sum(q)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def effective_sample_size(weights: np.ndarray) -> float:
    """Effective sample size ``1 / sum(w²)`` for normalised *weights*.

    6 active copies across example scripts.
    """
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return float(1.0 / np.sum(w ** 2))


# ---------------------------------------------------------------------------
# Loss trajectory extraction
# ---------------------------------------------------------------------------


def extract_loss_trajectories(
    results: Dict,
    split_type: str | None = None,
    cluster_assignments: Dict | None = None,
) -> pd.DataFrame:
    """Extract loss trajectories from nested optimisation result dicts.

    Handles both old format (direct history) and new format (dict keyed by
    maxent value).

    Parameters
    ----------
    results:
        ``{ensemble: {loss_name: {split_idx: history_or_dict}}}``
    split_type:
        Label for the ``split_type`` column.
    cluster_assignments:
        Optional dict of cluster assignments per ensemble (currently unused
        in extraction but reserved for future augmentation).

    Returns
    -------
    DataFrame with columns:
        ensemble, loss_function, split, split_type, maxent_value,
        convergence_step, train_loss, val_loss, step_number.
    """
    data_rows: list[dict] = []

    for ensemble in results:
        for loss_name in results[ensemble]:
            for split_idx in results[ensemble][loss_name]:
                split_results = results[ensemble][loss_name][split_idx]
                if split_results is None:
                    continue

                if isinstance(split_results, dict):
                    for maxent_val, history in split_results.items():
                        if history is None or not history.states:
                            continue
                        for step_idx, state in enumerate(history.states):
                            if state.losses is not None:
                                data_rows.append(
                                    {
                                        "ensemble": ensemble,
                                        "loss_function": loss_name,
                                        "split": split_idx,
                                        "split_type": split_type,
                                        "maxent_value": maxent_val,
                                        "convergence_step": step_idx + 1,
                                        "train_loss": float(state.losses.train_losses[0]),
                                        "val_loss": float(state.losses.val_losses[0]),
                                        "step_number": state.step,
                                    }
                                )
                else:
                    history = split_results
                    if history is None or not history.states:
                        continue
                    for step_idx, state in enumerate(history.states):
                        if state.losses is not None:
                            data_rows.append(
                                {
                                    "ensemble": ensemble,
                                    "loss_function": loss_name,
                                    "split": split_idx,
                                    "split_type": split_type,
                                    "maxent_value": 0.0,
                                    "convergence_step": step_idx + 1,
                                    "train_loss": float(state.losses.train_losses[0]),
                                    "val_loss": float(state.losses.val_losses[0]),
                                    "step_number": state.step,
                                }
                            )

    return pd.DataFrame(data_rows)


def extract_loss_trajectories_2d(
    results: Dict,
) -> pd.DataFrame:
    """Extract loss trajectories from a 2D sweep result dict.

    The results dict is keyed by
    ``{split_type: {ensemble: {loss_name: {bv_reg_fn: {maxent_val: {bv_reg_val: {split_idx: history}}}}}}}``.

    Returns a DataFrame with additional ``bv_reg_value`` and ``bv_reg_function`` columns.
    """
    data_rows: list[dict] = []

    for split_type, ensembles_data in results.items():
        for ensemble, loss_data in ensembles_data.items():
            for loss_name, bv_reg_fn_data in loss_data.items():
                for bv_reg_fn, maxent_data in bv_reg_fn_data.items():
                    if not isinstance(maxent_data, dict):
                        continue
                    for maxent_val, bv_data in maxent_data.items():
                        if not isinstance(bv_data, dict):
                            continue
                        for bv_reg_val, splits_data in bv_data.items():
                            if not isinstance(splits_data, dict):
                                continue
                            for split_idx, history in splits_data.items():
                                if history is None or not hasattr(history, 'states') or not history.states:
                                    continue
                                for step_idx, state in enumerate(history.states):
                                    if state.losses is not None:
                                        data_rows.append(
                                            {
                                                "ensemble": ensemble,
                                                "loss_function": loss_name,
                                                "bv_reg_function": bv_reg_fn,
                                                "split": split_idx,
                                                "split_type": split_type,
                                                "maxent_value": maxent_val,
                                                "bv_reg_value": bv_reg_val,
                                                "convergence_step": step_idx + 1,
                                                "train_loss": float(state.losses.train_losses[0]),
                                                "val_loss": float(state.losses.val_losses[0]),
                                                "step_number": state.step,
                                            }
                                        )

    return pd.DataFrame(data_rows)


# ---------------------------------------------------------------------------
# Model scoring
# ---------------------------------------------------------------------------


def compute_model_scores(
    df: pd.DataFrame,
    scoring_config: ScoringConfig | None = None,
) -> pd.DataFrame:
    """Compute model scores from a loss DataFrame.

    Score = ``-log10(val_loss) - train_penalty * log10(train_loss) - kl_penalty * kl + variance_bonus``

    The *train_penalty* and *kl_penalty* default to 1.0 and 0.1 respectively
    (Exp 2 formula) unless overridden via *scoring_config*.
    """
    df = df.copy()
    eps = 1e-300

    # Extract penalty weights
    train_penalty = 1.0
    kl_penalty_weight = 0.1
    if scoring_config and scoring_config.scoring_weights:
        train_penalty = scoring_config.scoring_weights.get("train_penalty", 1.0)
        kl_penalty_weight = scoring_config.scoring_weights.get("kl_penalty", 0.1)

    val_loss = df.get("val_loss", pd.Series(dtype=float)).fillna(np.nan).clip(lower=eps)
    train_loss = df.get("train_loss", pd.Series(dtype=float)).fillna(np.nan).clip(lower=eps)

    base_score = -np.log10(val_loss) - (train_penalty * np.log10(train_loss))
    df["model_score"] = base_score

    # KL divergence penalty
    if "kl_divergence" in df.columns:
        kl = df["kl_divergence"].fillna(0).clip(lower=0)
        df["model_score"] -= kl_penalty_weight * kl

    # Variance bonus (coefficient of variation)
    grouping_cols = [
        c
        for c in ["split_type", "ensemble", "loss_function", "maxent_value", "convergence_step"]
        if c in df.columns
    ]
    if grouping_cols:
        val_mean = df.groupby(grouping_cols)["val_loss"].transform("mean").fillna(np.nan).clip(lower=eps)
        val_std = df.groupby(grouping_cols)["val_loss"].transform("std").fillna(0.0)
        cv = val_std / val_mean
        df["model_score"] += -np.log10(1.0 + cv)

    return df


def select_best_models(df: pd.DataFrame) -> pd.DataFrame:
    """Select the best-scoring model per (split_type, ensemble, loss_function, split)."""
    df_scored = compute_model_scores(df)
    df_maxent = df_scored[df_scored["maxent_value"] > 0].copy()
    if df_maxent.empty:
        return pd.DataFrame()

    return df_maxent.loc[
        df_maxent.groupby(["split_type", "ensemble", "loss_function", "split"])[
            "model_score"
        ].idxmax()
    ]


# ---------------------------------------------------------------------------
# Clustering & recovery
# ---------------------------------------------------------------------------


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


def calculate_work_metrics(
    pred_lnpf: np.ndarray, prior_lnpf: np.ndarray, T: float = 300.0
) -> Dict[str, float]:
    """Compute thermodynamic work metrics based on lnPF distributions.

    Units: kJ/mol.
    """
    R = 8.314  # J/(mol K)

    def get_thermo_props(lnpf_array):
        # 1. Central Tendency
        avg_logPF = np.mean(lnpf_array)

        # 2. Deviation from the mean
        delta_phi = np.abs(lnpf_array - avg_logPF)

        # 3. Enthalpy component
        H = R * T * delta_phi

        # 4. Partition Function components
        q = np.abs(np.exp(-delta_phi))
        Z = np.sum(q)

        # 5. Probability
        Pi = q * Z

        # 6. Entropy component
        S = -R * Pi * np.log(Pi + 1e-16)

        # 7. Gibbs Free Energy component
        G = H - (T * S)

        return avg_logPF, H, S, G

    mu_pred, H_pred, S_pred, G_pred = get_thermo_props(pred_lnpf)
    mu_prior, H_prior, S_prior, G_prior = get_thermo_props(prior_lnpf)

    # Work Scale: RT * |mean(pred) - mean(prior)|
    work_scale = R * T * np.abs(mu_pred - mu_prior)

    # Work Shape: mean( |H_pred - H_prior| )
    work_shape = np.mean(np.abs(H_pred - H_prior))

    # Work Density: T * mean( |S_pred - S_prior| )
    work_density = T * np.mean(np.abs(S_pred - S_prior))

    # Work Fitting: mean( |G_pred - G_prior| )
    work_fitting = np.mean(np.abs(G_pred - G_prior))

    # Work Magnitude: shape - scale
    work_mag = work_shape - work_scale

    return {
        "work_scale_kj": float(work_scale / 1000.0),
        "work_shape_kj": float(work_shape / 1000.0),
        "work_density_kj": float(work_density / 1000.0),
        "work_fitting_kj": float(work_fitting / 1000.0),
        "work_magnitude_kj": float(work_mag / 1000.0),
    }
