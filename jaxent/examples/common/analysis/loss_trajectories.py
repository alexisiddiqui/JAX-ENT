from __future__ import annotations

from typing import Dict

import pandas as pd

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
