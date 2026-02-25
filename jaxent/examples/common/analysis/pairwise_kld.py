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
