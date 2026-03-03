from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .stats import kl_divergence, effective_sample_size


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
