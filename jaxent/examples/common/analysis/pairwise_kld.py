from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import kl_divergence

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


def compute_sequential_maxent_kld(weights_data) -> pd.DataFrame:
    """
    Compute KLD between sequential maxent values for each ensemble, split_type, loss, and split combination.
    Equivalent to the legacy function of the same name.
    """
    print("Computing KLD between sequential maxent values...")
    sequential_kld_data = []

    # Convert to DataFrame for easier grouping
    weights_df = pd.DataFrame(weights_data)

    if "weights" not in weights_df.columns:
         return pd.DataFrame(sequential_kld_data)

    # Group by ensemble, split_type, loss_function, and split
    for (ensemble, split_type, loss_func, split_idx), group in weights_df.groupby(
        ["ensemble", "split_type", "loss_function", "split"]
    ):
        # Sort by maxent_value for proper sequential comparison
        group_sorted = group.sort_values("maxent_value")
        maxent_values = group_sorted["maxent_value"].values
        weights_list = group_sorted["weights"].tolist()

        if len(maxent_values) < 2:
            continue

        # For each maxent (except the first), compute KLD with previous maxent
        for i in range(len(maxent_values)):
            current_maxent = maxent_values[i]
            current_weights = np.asarray(weights_list[i])

            if i == 0:
                # Compare first (lowest) maxent to uniform distribution
                n_frames = len(current_weights)
                uniform_weights = np.ones(n_frames) / n_frames

                kld_to_previous = kl_divergence(current_weights, uniform_weights)
                previous_maxent = None
                comparison_type = "vs_uniform"
            else:
                # Compare to previous maxent
                previous_maxent = maxent_values[i - 1]
                previous_weights = np.asarray(weights_list[i - 1])

                # Ensure both weight arrays have the same length
                min_len = min(len(current_weights), len(previous_weights))
                current_weights_trimmed = current_weights[:min_len]
                previous_weights_trimmed = previous_weights[:min_len]

                kld_to_previous = kl_divergence(current_weights_trimmed, previous_weights_trimmed)
                comparison_type = "vs_previous_maxent"

            if not pd.isna(kld_to_previous):
                sequential_kld_data.append(
                    {
                        "ensemble": ensemble,
                        "split_type": split_type,
                        "split_name": split_type,
                        "loss_function": loss_func,
                        "split_idx": split_idx,
                        "current_maxent": current_maxent,
                        "previous_maxent": previous_maxent,
                        "kld_to_previous": float(kld_to_previous),
                        "comparison_type": comparison_type,
                    }
                )

    return pd.DataFrame(sequential_kld_data)


def compute_pairwise_kld_between_splits_2d(results: dict) -> pd.DataFrame:
    """
    Compute pairwise KLD between splits for each (maxent, bv_reg) combination
    in a 2D hyperparameter sweep result dictionary.
    """
    print("Computing pairwise KLD between splits (2D sweep)...")
    kld_rows = []

    for split_type in results:
        for ensemble in results[split_type]:
            for loss_func in results[split_type][ensemble]:
                for bv_reg_fn in results[split_type][ensemble][loss_func]:
                    for maxent_val in results[split_type][ensemble][loss_func][bv_reg_fn]:
                        for bvreg_val in results[split_type][ensemble][loss_func][bv_reg_fn][maxent_val]:
                            weights_list = []
                            
                            for split_idx, history in results[split_type][ensemble][loss_func][bv_reg_fn][maxent_val][bvreg_val].items():
                                if history is None or not hasattr(history, "states") or not history.states:
                                    continue

                                final_state = history.states[-1]
                                if (
                                    hasattr(final_state, "params")
                                    and hasattr(final_state.params, "frame_weights")
                                    and final_state.params.frame_weights is not None
                                ):
                                    w = np.asarray(final_state.params.frame_weights)
                                    if len(w) == 0 or np.sum(w) <= 0:
                                        continue

                                    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                                    if np.sum(w) > 0:
                                        w = w / np.sum(w)
                                        weights_list.append(w)

                            if len(weights_list) < 2:
                                continue

                            # Compute pairwise symmetric KLD
                            pair_klds = _sym_kld_pairs(weights_list)

                            if len(pair_klds) > 0:
                                mean_kld = float(np.mean(pair_klds))
                                std_kld = float(np.std(pair_klds))
                                sem_kld = float(std_kld / np.sqrt(len(pair_klds)))

                                kld_rows.append(
                                    {
                                        "ensemble": ensemble,
                                        "split_type": split_type,
                                        "loss_function": loss_func,
                                        "bv_reg_function": bv_reg_fn,
                                        "maxent_value": maxent_val,
                                        "bv_reg_value": bvreg_val,
                                        "mean_kld_between_splits": mean_kld,
                                        "std_kld_between_splits": std_kld,
                                        "sem_kld_between_splits": sem_kld,
                                        "n_pairs": len(pair_klds),
                                        "n_splits": len(weights_list),
                                    }
                                )

    return pd.DataFrame(kld_rows)
