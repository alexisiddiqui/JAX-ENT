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
