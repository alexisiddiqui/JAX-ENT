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


# ---------------------------------------------------------------------------
# Array validation helpers  (extracted from score_models scripts)
# ---------------------------------------------------------------------------


def _is_valid_array(arr: np.ndarray) -> bool:
    """Return ``True`` if *arr* is non-None, non-empty, and not all-NaN."""
    if arr is None:
        return False
    if arr.size == 0:
        return False
    if np.all(np.isnan(arr)):
        return False
    return True


def _align_uptake_shape(arr: np.ndarray, target_ndim: int) -> np.ndarray:
    """Remove leading singleton axes until ``arr.ndim == target_ndim``."""
    aligned = arr
    while aligned.ndim > target_ndim and aligned.shape[0] == 1:
        aligned = np.squeeze(aligned, axis=0)
    return aligned


def get_experimental_uptake(data) -> np.ndarray:
    """Extract ``.dfrac`` from a list of ``ExpD_Datapoint`` objects."""
    if not data:
        return np.array([])
    return np.array([d.dfrac for d in data]).squeeze()


def calculate_mse(pred: np.ndarray, exp: np.ndarray) -> float:
    """MSE between predicted and experimental uptake with shape alignment.

    Uses :func:`_is_valid_array` and :func:`_align_uptake_shape` for robustness.
    """
    if not _is_valid_array(np.asarray(pred) if pred is not None else None) or not _is_valid_array(
        np.asarray(exp) if exp is not None else None
    ):
        return np.nan

    pred = _align_uptake_shape(np.asarray(pred), np.asarray(exp).ndim)
    exp = np.asarray(exp)

    if pred.size == 0:
        return np.nan

    if pred.shape != exp.shape:
        if pred.ndim >= 2 and exp.ndim >= 2 and pred.shape[0] == exp.shape[0]:
            min_t = min(pred.shape[1], exp.shape[1])
            pred = pred[:, :min_t]
            exp = exp[:, :min_t]
        else:
            return np.nan

    return float(np.nanmean((pred - exp) ** 2))


# ---------------------------------------------------------------------------
# Convergence selection
# ---------------------------------------------------------------------------


def find_best_convergence_threshold(history) -> tuple:
    """Find the convergence step with the lowest validation loss.

    Parameters
    ----------
    history:
        An ``OptimizationHistory`` object with a ``.states`` attribute.

    Returns
    -------
    ``(best_step_idx, best_val_loss, best_state)``
    """
    if history is None or not getattr(history, "states", None):
        return None, np.inf, None

    best_val_loss = np.inf
    best_step_idx = 0
    best_state = None

    for step_idx, state in enumerate(history.states):
        if hasattr(state, "losses") and hasattr(state.losses, "val_losses"):
            val_losses = state.losses.val_losses
            if val_losses is not None and len(val_losses) > 0:
                val_loss = float(val_losses[0])
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step_idx = step_idx
                    best_state = state

    return best_step_idx, best_val_loss, best_state


# ---------------------------------------------------------------------------
# Frame-weight extraction
# ---------------------------------------------------------------------------


def extract_frame_weights_kl(
    results: Dict,
    n_frames_by_ensemble: Dict[str, int] | None = None,
) -> pd.DataFrame:
    """Extract frame weights and compute KL divergence + ESS at every step.

    Iterates the 5-level
    ``{split_type: {ensemble: {loss: {maxent: {split_idx: history}}}}}``
    result dict.  Includes a ``convergence_fraction`` field (Exp2 behaviour).
    Optionally skips entries whose frame-count doesn't match an expected value
    (pass ``n_frames_by_ensemble`` to enable this guard).

    Returns
    -------
    DataFrame with columns:
        split_type, ensemble, loss_function, maxent_value, split, step,
        convergence_fraction, kl_divergence, effective_sample_size, num_frames,
        weights.
    """
    data_rows: list[dict] = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is None or not getattr(history, "states", None):
                            continue
                        n_states = len(history.states)
                        for step_idx, state in enumerate(history.states):
                            if not (
                                hasattr(state.params, "frame_weights")
                                and state.params.frame_weights is not None
                            ):
                                continue
                            w = np.array(state.params.frame_weights)
                            if len(w) == 0 or np.sum(w) == 0:
                                continue
                            if n_frames_by_ensemble is not None:
                                expected = n_frames_by_ensemble.get(ensemble)
                                if expected is not None and len(w) != expected:
                                    continue
                            uniform = np.ones(len(w)) / len(w)
                            try:
                                kl_div = kl_divergence(w, uniform)
                                ess = effective_sample_size(w)
                            except Exception as e:
                                print(
                                    f"KL/ESS failed for {ensemble}/{loss_name} "
                                    f"maxent={maxent_val} split={split_idx} step={step_idx}: {e}"
                                )
                                continue
                            convergence_fraction = (
                                step_idx / (n_states - 1) if n_states > 1 else 1.0
                            )
                            data_rows.append(
                                {
                                    "split_type": split_type,
                                    "ensemble": ensemble,
                                    "loss_function": loss_name,
                                    "maxent_value": maxent_val,
                                    "split": split_idx,
                                    "step": step_idx,
                                    "convergence_fraction": convergence_fraction,
                                    "kl_divergence": float(kl_div),
                                    "effective_sample_size": float(ess),
                                    "num_frames": len(w),
                                    "weights": w / np.sum(w),
                                }
                            )
    return pd.DataFrame(data_rows)


def extract_final_weights(results: Dict) -> pd.DataFrame:
    """Extract only the final (converged) frame weights from a 5-level result dict.

    Returns
    -------
    DataFrame with columns:
        split_type, ensemble, loss_function, maxent_value, split,
        kl_divergence, effective_sample_size, num_frames, weights.
    """
    data_rows: list[dict] = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is None or not getattr(history, "states", None):
                            continue
                        final_state = history.states[-1]
                        if not (
                            hasattr(final_state, "params")
                            and hasattr(final_state.params, "frame_weights")
                            and final_state.params.frame_weights is not None
                        ):
                            continue
                        w = np.array(final_state.params.frame_weights)
                        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                        if np.sum(w) <= 0:
                            continue
                        w = w / np.sum(w)
                        uniform = np.ones(len(w)) / len(w)
                        try:
                            kl_div = kl_divergence(w, uniform)
                            ess = effective_sample_size(w)
                        except Exception:
                            continue
                        data_rows.append(
                            {
                                "split_type": split_type,
                                "ensemble": ensemble,
                                "loss_function": loss_name,
                                "maxent_value": maxent_val,
                                "split": split_idx,
                                "kl_divergence": float(kl_div),
                                "effective_sample_size": float(ess),
                                "num_frames": len(w),
                                "weights": w,
                            }
                        )
    return pd.DataFrame(data_rows)


def extract_final_weights_2d(results: Dict) -> pd.DataFrame:
    """Extract final weights from a 7-level 2D-sweep result dict.

    Dict structure:
        ``{split_type: {ensemble: {loss: {bv_reg_fn: {maxent: {bvreg: {split: history}}}}}}}``

    Returns
    -------
    DataFrame with columns:
        split_type, ensemble, loss_function, bv_reg_function, maxent_value,
        bv_reg_value, split, kl_divergence, effective_sample_size, num_frames,
        weight_min, weight_max, weight_mean, weight_std.
    """
    data_rows: list[dict] = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for bv_reg_fn in results[split_type][ensemble][loss_name]:
                    for maxent_val in results[split_type][ensemble][loss_name][bv_reg_fn]:
                        for bvreg_val in results[split_type][ensemble][loss_name][bv_reg_fn][
                            maxent_val
                        ]:
                            for split_idx, history in results[split_type][ensemble][loss_name][
                                bv_reg_fn
                            ][maxent_val][bvreg_val].items():
                                if history is None or not getattr(history, "states", None):
                                    continue
                                final_state = history.states[-1]
                                if not (
                                    hasattr(final_state.params, "frame_weights")
                                    and final_state.params.frame_weights is not None
                                ):
                                    continue
                                w = np.array(final_state.params.frame_weights)
                                w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                                if np.sum(w) <= 0:
                                    continue
                                w = w / np.sum(w)
                                uniform = np.ones(len(w)) / len(w)
                                try:
                                    kl_div = kl_divergence(w, uniform)
                                    ess = effective_sample_size(w)
                                except Exception as e:
                                    print(
                                        f"Failed metrics for {ensemble}/{loss_name} "
                                        f"bvreg_fn={bv_reg_fn} maxent={maxent_val} "
                                        f"bvreg={bvreg_val} split={split_idx}: {e}"
                                    )
                                    continue
                                data_rows.append(
                                    {
                                        "split_type": split_type,
                                        "ensemble": ensemble,
                                        "loss_function": loss_name,
                                        "bv_reg_function": bv_reg_fn,
                                        "maxent_value": maxent_val,
                                        "bv_reg_value": bvreg_val,
                                        "split": split_idx,
                                        "kl_divergence": float(kl_div),
                                        "effective_sample_size": float(ess),
                                        "num_frames": len(w),
                                        "weight_min": float(np.min(w)),
                                        "weight_max": float(np.max(w)),
                                        "weight_mean": float(np.mean(w)),
                                        "weight_std": float(np.std(w)),
                                    }
                                )
    return pd.DataFrame(data_rows)


def extract_weights_over_convergence_steps(results: Dict) -> pd.DataFrame:
    """Extract frame weights sampled at 6 convergence fractions (10/30/50/70/90/100%).

    Returns
    -------
    DataFrame with columns:
        split_type, ensemble, loss_function, maxent_value, split,
        convergence_step, convergence_fraction, kl_divergence,
        effective_sample_size, num_frames, weights.
    """
    STEP_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    data_rows: list[dict] = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_name in results[split_type][ensemble]:
                for maxent_val in results[split_type][ensemble][loss_name]:
                    for split_idx, history in results[split_type][ensemble][loss_name][
                        maxent_val
                    ].items():
                        if history is None or not getattr(history, "states", None):
                            continue
                        n_states = len(history.states)
                        step_indices = (
                            [int(f * (n_states - 1)) for f in STEP_FRACTIONS]
                            if n_states >= 10
                            else list(range(n_states))
                        )
                        for step_idx in step_indices:
                            state = history.states[step_idx]
                            if not (
                                hasattr(state, "params")
                                and hasattr(state.params, "frame_weights")
                                and state.params.frame_weights is not None
                            ):
                                continue
                            w = np.array(state.params.frame_weights)
                            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                            if np.sum(w) <= 0:
                                continue
                            w = w / np.sum(w)
                            convergence_fraction = (
                                step_idx / (n_states - 1) if n_states > 1 else 1.0
                            )
                            uniform = np.ones(len(w)) / len(w)
                            try:
                                kl_div = kl_divergence(w, uniform)
                                ess = effective_sample_size(w)
                            except Exception:
                                continue
                            data_rows.append(
                                {
                                    "split_type": split_type,
                                    "ensemble": ensemble,
                                    "loss_function": loss_name,
                                    "maxent_value": maxent_val,
                                    "split": split_idx,
                                    "convergence_step": step_idx,
                                    "convergence_fraction": convergence_fraction,
                                    "kl_divergence": float(kl_div),
                                    "effective_sample_size": float(ess),
                                    "num_frames": len(w),
                                    "weights": w,
                                }
                            )
    return pd.DataFrame(data_rows)


# ---------------------------------------------------------------------------
# Pairwise KLD between splits
# ---------------------------------------------------------------------------


def _sym_kld_pairs(weights_list: list) -> list[float]:
    """Compute all pairwise symmetric KLD values for a list of weight arrays."""
    pair_klds: list[float] = []
    for i in range(len(weights_list)):
        for j in range(i + 1, len(weights_list)):
            wi = np.asarray(weights_list[i])
            wj = np.asarray(weights_list[j])
            min_len = min(len(wi), len(wj))
            if min_len == 0:
                continue
            try:
                kij = kl_divergence(wi[:min_len], wj[:min_len])
                kji = kl_divergence(wj[:min_len], wi[:min_len])
                if not (np.isnan(kij) or np.isnan(kji)):
                    pair_klds.append((kij + kji) / 2.0)
            except Exception:
                continue
    return pair_klds


def compute_pairwise_kld_between_splits(
    data,
    per_step: bool = False,
) -> pd.DataFrame:
    """Compute symmetric pairwise KLD between all split replicates.

    Parameters
    ----------
    data:
        List of dicts or DataFrame containing a ``weights`` column.
        When ``per_step=False`` (Exp1 behaviour), typically the output of
        :func:`extract_final_weights`.  When ``per_step=True`` (Exp2 behaviour),
        the output of :func:`extract_frame_weights_kl` — pairwise KLD is
        computed at every unique convergence step.
    per_step:
        If ``True``, group by ``step`` in addition to the base columns and
        include ``step`` and ``convergence_fraction`` in the result.

    Returns
    -------
    DataFrame with columns:
        ensemble, split_type, loss_function, maxent_value,
        [step, convergence_fraction if per_step],
        mean_kld_between_splits, std_kld_between_splits,
        sem_kld_between_splits, n_pairs.
    """
    df = pd.DataFrame(data) if isinstance(data, list) else data.copy()
    if "weights" not in df.columns:
        raise ValueError("Input data must contain a 'weights' column.")

    base_cols = ["ensemble", "split_type", "loss_function", "maxent_value"]
    group_cols = base_cols + (["step"] if per_step else [])

    kld_rows: list[dict] = []
    for keys, group in df.groupby(group_cols):
        key_dict = dict(
            zip(group_cols, keys if isinstance(keys, tuple) else (keys,))
        )
        wlist = group["weights"].tolist()
        pair_klds = _sym_kld_pairs(wlist)
        if not pair_klds:
            continue
        std_kld = float(np.std(pair_klds))
        row = {
            **key_dict,
            "mean_kld_between_splits": float(np.mean(pair_klds)),
            "std_kld_between_splits": std_kld,
            "sem_kld_between_splits": float(std_kld / np.sqrt(len(pair_klds))),
            "n_pairs": len(pair_klds),
        }
        if per_step and "convergence_fraction" in df.columns:
            cf = group["convergence_fraction"].iloc[0]
            row["convergence_fraction"] = float(cf)
        kld_rows.append(row)

    return pd.DataFrame(kld_rows)


def compute_pairwise_kld_between_splits_2d(results: Dict) -> pd.DataFrame:
    """Compute pairwise KLD between splits for each (maxent, bv_reg) pair.

    Operates on the 7-level 2D-sweep result dict.

    Returns
    -------
    DataFrame with columns:
        ensemble, split_type, loss_function, bv_reg_function, maxent_value,
        bv_reg_value, mean_kld_between_splits, std_kld_between_splits,
        sem_kld_between_splits, n_pairs, n_splits.
    """
    kld_rows: list[dict] = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_func in results[split_type][ensemble]:
                for bv_reg_fn in results[split_type][ensemble][loss_func]:
                    for maxent_val in results[split_type][ensemble][loss_func][bv_reg_fn]:
                        for bvreg_val in results[split_type][ensemble][loss_func][bv_reg_fn][
                            maxent_val
                        ]:
                            wlist: list[np.ndarray] = []
                            for split_idx, history in results[split_type][ensemble][loss_func][
                                bv_reg_fn
                            ][maxent_val][bvreg_val].items():
                                if history is None or not getattr(history, "states", None):
                                    continue
                                final_state = history.states[-1]
                                if not (
                                    hasattr(final_state.params, "frame_weights")
                                    and final_state.params.frame_weights is not None
                                ):
                                    continue
                                w = np.array(final_state.params.frame_weights)
                                w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                                if np.sum(w) > 0:
                                    wlist.append(w / np.sum(w))
                            if len(wlist) < 2:
                                continue
                            pair_klds = _sym_kld_pairs(wlist)
                            if not pair_klds:
                                continue
                            std_kld = float(np.std(pair_klds))
                            kld_rows.append(
                                {
                                    "ensemble": ensemble,
                                    "split_type": split_type,
                                    "loss_function": loss_func,
                                    "bv_reg_function": bv_reg_fn,
                                    "maxent_value": maxent_val,
                                    "bv_reg_value": bvreg_val,
                                    "mean_kld_between_splits": float(np.mean(pair_klds)),
                                    "std_kld_between_splits": std_kld,
                                    "sem_kld_between_splits": float(
                                        std_kld / np.sqrt(len(pair_klds))
                                    ),
                                    "n_pairs": len(pair_klds),
                                    "n_splits": len(wlist),
                                }
                            )
    return pd.DataFrame(kld_rows)


def compute_sequential_maxent_kld(weights_data) -> pd.DataFrame:
    """Compute KLD between consecutive maxent values for each run.

    The first (lowest) maxent value is compared to the uniform distribution.
    All others are compared to the immediately preceding maxent value.

    Parameters
    ----------
    weights_data:
        List of dicts or DataFrame with columns: ensemble, split_type,
        loss_function, split, maxent_value, weights.

    Returns
    -------
    DataFrame with columns:
        ensemble, split_type, loss_function, split_idx, current_maxent,
        previous_maxent, kld_to_previous, comparison_type.
    """
    df = pd.DataFrame(weights_data) if isinstance(weights_data, list) else weights_data.copy()
    rows: list[dict] = []

    for (ensemble, split_type, loss_func, split_idx), group in df.groupby(
        ["ensemble", "split_type", "loss_function", "split"]
    ):
        group_sorted = group.sort_values("maxent_value")
        maxent_values = group_sorted["maxent_value"].values
        weights_list = group_sorted["weights"].tolist()
        if len(maxent_values) < 2:
            continue

        for i, (current_maxent, current_w) in enumerate(
            zip(maxent_values, weights_list)
        ):
            cw = np.asarray(current_w)
            if i == 0:
                n_frames = len(cw)
                uniform_w = np.ones(n_frames) / n_frames
                kld = (
                    kl_divergence(cw, uniform_w)
                    if np.sum(cw) > 0
                    else np.nan
                )
                previous_maxent = None
                comparison_type = "vs_uniform"
            else:
                previous_maxent = maxent_values[i - 1]
                pw = np.asarray(weights_list[i - 1])
                min_len = min(len(cw), len(pw))
                kld = (
                    kl_divergence(cw[:min_len], pw[:min_len])
                    if min_len > 0 and np.sum(cw[:min_len]) > 0 and np.sum(pw[:min_len]) > 0
                    else np.nan
                )
                comparison_type = "vs_previous_maxent"

            if not np.isnan(kld):
                rows.append(
                    {
                        "ensemble": ensemble,
                        "split_type": split_type,
                        "loss_function": loss_func,
                        "split_idx": split_idx,
                        "current_maxent": current_maxent,
                        "previous_maxent": previous_maxent,
                        "kld_to_previous": float(kld),
                        "comparison_type": comparison_type,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Conformational recovery
# ---------------------------------------------------------------------------


def _get_cluster_assignments(cluster_info) -> np.ndarray | None:
    """Extract cluster assignment array from various container formats."""
    if isinstance(cluster_info, dict):
        if "cluster_assignments" in cluster_info:
            return np.asarray(cluster_info["cluster_assignments"])
        if "frame_data" in cluster_info:
            df = cluster_info["frame_data"]
            col = "cluster_label" if "cluster_label" in df.columns else df.columns[0]
            return df[col].values
    if isinstance(cluster_info, pd.DataFrame):
        col = "cluster_label" if "cluster_label" in cluster_info.columns else cluster_info.columns[0]
        return cluster_info[col].values
    return np.asarray(cluster_info)


def _detect_2d_results(results: Dict) -> bool:
    """Return ``True`` if *results* uses a 7-level (2D sweep) structure.

    In a 7-level dict the key directly after ``loss`` is a string (bv_reg_fn
    like ``"L1"`` or ``"L2"``). In a 5-level dict it is a float (maxent value).
    """
    for split_type in results:
        for ensemble in results[split_type]:
            for loss in results[split_type][ensemble]:
                val = results[split_type][ensemble][loss]
                if isinstance(val, dict):
                    for k in val:
                        return isinstance(k, str)
    return False


def analyze_conformational_recovery(
    clustering_data: Dict,
    results: Dict,
    target_ratios: Dict[str, float],
    state_mapping: Dict[int, str],
    metric: str = "jsd",
    best_step_only: bool = False,
) -> pd.DataFrame:
    """Analyze conformational state recovery from optimized frame weights.

    Unified implementation supporting three experiment types:

    * ``metric='ratio'`` (Exp1): uses :func:`calculate_cluster_ratios` and
      :func:`calculate_recovery_percentage`, iterating all convergence steps.
    * ``metric='jsd'`` (Exp2): uses :func:`calculate_recovery_JSD`, iterating
      all convergence steps (``best_step_only=False``).
    * ``metric='jsd'`` (Exp3): uses :func:`calculate_recovery_JSD` with
      ``best_step_only=True``, selecting the minimum-validation-loss step via
      :func:`find_best_convergence_threshold`.

    Parameters
    ----------
    clustering_data:
        ``{ensemble_name: {"cluster_assignments": ndarray, ...}}``
    results:
        5-level or 7-level nested optimisation result dict.
    target_ratios:
        Ground-truth state proportions (``{state_name: fraction}``).
    state_mapping:
        ``{cluster_id: state_name}``.
    metric:
        ``"jsd"`` or ``"ratio"``.
    best_step_only:
        Use only the best convergence step per run.

    Returns
    -------
    DataFrame whose columns depend on *metric* and *results* dimensionality.
    """
    rows: list[dict] = []
    is_2d = _detect_2d_results(results)

    for ensemble_name, cluster_info in clustering_data.items():
        cluster_assignments = _get_cluster_assignments(cluster_info)
        if cluster_assignments is None:
            continue
        n_frames = len(cluster_assignments)
        uniform_w = np.ones(n_frames) / n_frames

        # Unweighted baseline
        if metric == "jsd":
            js_div_orig, props_orig = calculate_recovery_JSD(
                cluster_assignments, uniform_w, target_ratios, state_mapping
            )
            rec_orig = (
                (1.0 - np.sqrt(js_div_orig)) * 100.0
                if not np.isnan(js_div_orig)
                else 0.0
            )
            baseline_extra = {
                "js_divergence": float(js_div_orig) if not np.isnan(js_div_orig) else 0.0,
                "js_distance": float(np.sqrt(js_div_orig)) if not np.isnan(js_div_orig) else 0.0,
                **{f"{st}_current": props_orig.get(st, 0.0) for st in target_ratios},
                **{f"{st}_target": target_ratios[st] for st in target_ratios},
            }
        else:
            props_orig = calculate_cluster_ratios(cluster_assignments, uniform_w)
            rec_orig = calculate_recovery_percentage(
                cluster_assignments, uniform_w, target_ratios, state_mapping
            )
            baseline_extra = {}

        baseline: dict = {
            "ensemble": ensemble_name,
            "loss_function": "Original",
            "split_type": "N/A",
            "split": "N/A",
            "maxent_value": 0.0,
            "convergence_step": 0,
            "recovery_percent": float(rec_orig) if not np.isnan(rec_orig) else 0.0,
            **baseline_extra,
        }
        if is_2d:
            baseline["bv_reg_function"] = "N/A"
            baseline["bv_reg_value"] = 0.0
        rows.append(baseline)

        # Iterate optimised runs
        for split_type in results:
            if ensemble_name not in results[split_type]:
                continue
            for loss_name, loss_results in results[split_type][ensemble_name].items():
                if is_2d:
                    outer_items = [
                        (bv_reg_fn, maxent_val, bvreg_val, split_idx, history)
                        for bv_reg_fn, bv_dict in loss_results.items()
                        for maxent_val, me_dict in bv_dict.items()
                        for bvreg_val, sp_dict in me_dict.items()
                        for split_idx, history in sp_dict.items()
                    ]
                else:
                    outer_items = [
                        (None, maxent_val, None, split_idx, history)
                        for maxent_val, sp_dict in loss_results.items()
                        for split_idx, history in sp_dict.items()
                    ]

                for item in outer_items:
                    bv_reg_fn, maxent_val, bvreg_val, split_idx, history = item
                    if history is None or not getattr(history, "states", None):
                        continue

                    if best_step_only:
                        best_step_idx, _, best_state = find_best_convergence_threshold(history)
                        if best_state is None:
                            continue
                        steps_to_process = [(best_step_idx, best_state)]
                    else:
                        steps_to_process = list(enumerate(history.states))

                    for step_idx, state in steps_to_process:
                        if not (
                            hasattr(state.params, "frame_weights")
                            and state.params.frame_weights is not None
                        ):
                            continue
                        w = np.array(state.params.frame_weights)
                        if len(w) != n_frames or np.sum(w) <= 0:
                            continue
                        w = w / np.sum(w)

                        row: dict = {
                            "ensemble": ensemble_name,
                            "loss_function": loss_name,
                            "split_type": split_type,
                            "split": split_idx,
                            "maxent_value": maxent_val,
                            "best_convergence_step"
                            if best_step_only
                            else "convergence_step": step_idx,
                            "recovery_percent": 0.0,
                        }
                        if is_2d:
                            row["bv_reg_function"] = bv_reg_fn
                            row["bv_reg_value"] = bvreg_val

                        if metric == "jsd":
                            js_div, props = calculate_recovery_JSD(
                                cluster_assignments, w, target_ratios, state_mapping
                            )
                            if np.isnan(js_div):
                                continue
                            rec_pct = (1.0 - np.sqrt(js_div)) * 100.0
                            row.update(
                                {
                                    "js_divergence": float(js_div),
                                    "js_distance": float(np.sqrt(js_div)),
                                    "recovery_percent": float(rec_pct),
                                    **{
                                        f"{st}_current": props.get(st, 0.0)
                                        for st in target_ratios
                                    },
                                    **{
                                        f"{st}_target": target_ratios[st]
                                        for st in target_ratios
                                    },
                                }
                            )
                        else:
                            rec_pct = calculate_recovery_percentage(
                                cluster_assignments, w, target_ratios, state_mapping
                            )
                            row["open_recovery"] = (
                                float(rec_pct) if not np.isnan(rec_pct) else 0.0
                            )
                            row["recovery_percent"] = (
                                float(rec_pct) if not np.isnan(rec_pct) else 0.0
                            )

                        rows.append(row)

    return pd.DataFrame(rows)


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
