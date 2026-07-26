#!/usr/bin/env python3
"""Test whether observed-only held-out scores identify MoPrP state recovery.

This is a standalone prerequisite study for Experiment 2. It refits the existing
joint-BV grids to recover frame weights, computes validation scores using only
held-out observations and peptide mappings, and reveals NMR states strictly after
all assigned blind fits and scores have completed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _moprp_recovery_common as common
import moprp_joint_diag_d_fit as joint
from jaxent.src.analysis.state_population import (
    FULL_STATE_SUPPORT,
    state_populations,
    strict_recovery_percent,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "_moprp_val_score_correlation_20260724"
FIXED_ARTIFACT = HERE / "_moprp_joint_diag_d_fit_fixedess"
KL_ARTIFACT = HERE / "_moprp_joint_diag_d_fit_replicated"
SCORE_NAMES = (
    "mse",
    "sens_mse",
    "binom_mse",
    "redund_mse",
    "combined_mse",
    "shape_dist",
)
DECOY_STATES = ("PUF3", "unfolded", "PUF2-like")


def _weighted_mean(values: np.ndarray, weights: np.ndarray, epsilon: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), values.shape)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= epsilon:
        weights = np.ones_like(values)
        total = float(weights.size)
    return float(np.sum(weights * values) / max(total, epsilon))


def peptide_redundancy_weights(mapping: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return observed-map-only peptide weights that undo residue over-counting."""

    mapping = np.asarray(mapping, dtype=float)
    if mapping.ndim != 2 or mapping.shape[0] < 1 or mapping.shape[1] < 1:
        raise ValueError("mapping must be a non-empty peptide-by-residue matrix")
    active = mapping > 0.0
    if np.any(active.sum(axis=1) == 0):
        raise ValueError("every peptide must cover at least one residue")
    coverage = active.sum(axis=0).astype(float)
    inverse_coverage = np.divide(
        1.0,
        coverage,
        out=np.zeros_like(coverage),
        where=coverage > 0,
    )
    peptide = np.asarray(
        [inverse_coverage[row].mean() for row in active], dtype=float
    )
    mean = float(peptide.mean())
    return peptide / max(mean, epsilon)


def validation_scores(
    predicted: np.ndarray,
    observed: np.ndarray,
    mapping: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Compute the six weight-free candidate scores on one validation block."""

    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if predicted.shape != observed.shape or predicted.ndim != 2:
        raise ValueError("predicted and observed must be aligned 2-D arrays")
    if np.asarray(mapping).shape[0] != observed.shape[0]:
        raise ValueError("mapping rows must align with validation peptides")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("predicted and observed must be finite")

    residual2 = np.square(predicted - observed)
    clipped = np.clip(observed, 0.0, 1.0)
    one_minus = np.clip(1.0 - clipped, epsilon, 1.0)
    sensitivity = -(1.0 - clipped) * np.log(one_minus)
    binomial_precision = 1.0 / np.maximum(clipped * (1.0 - clipped), epsilon)
    redundancy = peptide_redundancy_weights(mapping)[:, None]
    combined = sensitivity * binomial_precision * redundancy
    correlation = spearmanr(predicted.ravel(), observed.ravel()).statistic
    shape_distance = 1.0 - float(correlation) if np.isfinite(correlation) else 1.0
    return {
        "mse": float(residual2.mean()),
        "sens_mse": _weighted_mean(residual2, sensitivity, epsilon),
        "binom_mse": _weighted_mean(residual2, binomial_precision, epsilon),
        "redund_mse": _weighted_mean(residual2, redundancy, epsilon),
        "combined_mse": _weighted_mean(residual2, combined, epsilon),
        "shape_dist": shape_distance,
    }


def _grid_specs() -> dict[str, dict]:
    return {
        "fixed_ess": {
            "control_column": "ess_target",
            "artifact": FIXED_ARTIFACT,
        },
        "kl": {
            "control_column": "eta",
            "artifact": KL_ARTIFACT,
        },
    }


def _reveal_recovery(records: list[dict], base_cells: dict[str, dict]) -> None:
    """Reveal recovery/state labels only after all blind worker fits are complete."""

    revealed = {}
    for name, cell in base_cells.items():
        states, support, targets, _ = common.reveal_nmr_reference(
            name, expected_frames=cell["n_frames"]
        )
        revealed[name] = (states, support, targets)
    for record in records:
        weights = record.pop("weights")
        states, support, targets = revealed[record["ensemble"]]
        populations = np.asarray(state_populations(weights, states, support))
        record["recovery"] = float(
            strict_recovery_percent(weights, states, support, targets)
        )
        record["dominant_state"] = str(states[record["dominant_frame"]])
        record["decoy_mass"] = float(
            sum(
                populations[FULL_STATE_SUPPORT.index(state)]
                for state in DECOY_STATES
            )
        )


def _persisted_sanity(frame: pd.DataFrame) -> dict:
    reports = {}
    if "persisted_val_mse" in frame.columns:
        for grid_name, group in frame.groupby("grid", sort=True):
            relative = np.abs(group["mse"] - group["persisted_val_mse"]) / np.maximum(
                np.abs(group["persisted_val_mse"]), 1e-15
            )
            reports[grid_name] = {
                "rows": int(len(group)),
                "missing_persisted_rows": int(group["persisted_val_mse"].isna().sum()),
                "maximum_mse_relative_difference": float(relative.max()),
                "passes_1e_6": bool(
                    not group["persisted_val_mse"].isna().any()
                    and relative.max() <= 1e-6
                ),
            }
        return reports
    for grid_name, grid in _grid_specs().items():
        subset = frame[frame["grid"] == grid_name]
        if subset.empty:
            continue
        persisted = pd.read_csv(grid["artifact"] / "joint_diag_d_fit_replicates.csv")
        keys = ["arm", "gamma", "ensemble", "split"]
        control = grid["control_column"]
        left = subset.copy()
        left[control] = left["control_value"]
        merged = left.merge(
            persisted[keys + [control, "val_mse"]],
            on=keys + [control],
            how="left",
            validate="one_to_one",
            suffixes=("", "_persisted"),
        )
        relative = np.abs(merged["mse"] - merged["val_mse"]) / np.maximum(
            np.abs(merged["val_mse"]), 1e-15
        )
        reports[grid_name] = {
            "rows": int(len(merged)),
            "missing_persisted_rows": int(merged["val_mse"].isna().sum()),
            "maximum_mse_relative_difference": float(relative.max()),
            "passes_1e_6": bool(
                not merged["val_mse"].isna().any() and relative.max() <= 1e-6
            ),
        }
    return reports


def _records_from_first_party_payloads(
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, dict]]:
    """Load production predictions/weights without invoking an optimizer."""

    base_cells = {
        name: joint._load_cell(name, args.target_artifact) for name in common.ENSEMBLES
    }
    first = next(iter(base_cells.values()))
    specs = joint._split_specs(first["mapping"].shape[0], first["timepoints"].shape[0])
    cells_by_split = {
        int(spec["split"]): joint._split_cells(base_cells, spec) for spec in specs
    }
    artifact_dirs = {
        "fixed_ess": args.fixed_artifact,
        "kl": args.kl_artifact,
    }
    records = []
    for grid_name in args.grids:
        artifact = artifact_dirs[grid_name]
        table = pd.read_csv(artifact / "joint_diag_d_fit_replicates.csv")
        if "payload_id" not in table.columns:
            raise ValueError(f"{artifact} predates first-party fitted-array persistence")
        with np.load(artifact / "joint_diag_d_fit_payload.npz") as payload:
            expected = {
                f"{payload_id}__{kind}"
                for payload_id in table["payload_id"]
                for kind in ("weights", "val_prediction")
            }
            if set(payload.files) != expected:
                raise ValueError(f"{artifact}: fitted-array payload keys do not match rows")
            for row in table.itertuples(index=False):
                cell = cells_by_split[int(row.split)][row.ensemble]
                weights = np.asarray(payload[f"{row.payload_id}__weights"], dtype=float)
                predicted = np.asarray(
                    payload[f"{row.payload_id}__val_prediction"], dtype=float
                )
                scores = validation_scores(
                    predicted,
                    np.asarray(cell["val_observed"]),
                    np.asarray(cell["val_mapping"]),
                )
                records.append(
                    {
                        "grid": grid_name,
                        "diversity_control": (
                            "fixed_ess" if grid_name == "fixed_ess" else "kl"
                        ),
                        "arm": row.arm,
                        "gamma": float(row.gamma),
                        "control_value": (
                            float(row.ess_target)
                            if grid_name == "fixed_ess"
                            else float(row.eta)
                        ),
                        "eta": (
                            float(row.eta) if grid_name == "kl" else np.nan
                        ),
                        "ess_target": (
                            float(row.ess_target)
                            if grid_name == "fixed_ess"
                            else np.nan
                        ),
                        "ensemble": row.ensemble,
                        "split": int(row.split),
                        "bc": float(row.bc),
                        "bh": float(row.bh),
                        "ess": float(row.ess),
                        "payload_id": row.payload_id,
                        "persisted_val_mse": float(row.val_mse),
                        "persisted_recovery": float(row.recovery),
                        "dominant_frame": int(np.argmax(weights)),
                        "dominant_weight": float(weights.max()),
                        "weights": weights,
                        **scores,
                    }
                )
    return records, base_cells


def _safe_spearman(score: pd.Series, recovery: pd.Series) -> float:
    result = spearmanr(score.to_numpy(), recovery.to_numpy()).statistic
    return float(result) if np.isfinite(result) else np.nan


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return condition-level and aggregate fixed-ESS/KL correlations."""

    long = frame.melt(
        id_vars=[
            "grid",
            "arm",
            "gamma",
            "control_value",
            "ensemble",
            "split",
            "recovery",
        ],
        value_vars=SCORE_NAMES,
        var_name="score",
        value_name="score_value",
    )
    rows = []
    fixed = long[long["grid"] == "fixed_ess"]
    for key, group in fixed.groupby(
        ["score", "arm", "ensemble", "split", "control_value"], sort=True
    ):
        rows.append(
            {
                "row_type": "condition",
                "analysis": "primary_fixed_ess_across_gamma",
                "score": key[0],
                "arm": key[1],
                "ensemble": key[2],
                "split": int(key[3]),
                "ess_target": float(key[4]),
                "eta": np.nan,
                "n": int(len(group)),
                "spearman": _safe_spearman(
                    group["score_value"], group["recovery"]
                ),
            }
        )
    kl = long[long["grid"] == "kl"]
    for key, group in kl.groupby(
        ["score", "arm", "ensemble", "split"], sort=True
    ):
        rows.append(
            {
                "row_type": "condition",
                "analysis": "secondary_kl_across_eta_gamma",
                "score": key[0],
                "arm": key[1],
                "ensemble": key[2],
                "split": int(key[3]),
                "ess_target": np.nan,
                "eta": np.nan,
                "n": int(len(group)),
                "spearman": _safe_spearman(
                    group["score_value"], group["recovery"]
                ),
            }
        )
    condition = pd.DataFrame(rows)
    aggregates = []
    for (analysis, score, arm), group in condition.groupby(
        ["analysis", "score", "arm"], sort=True
    ):
        finite = group["spearman"].dropna().to_numpy()
        aggregates.append(
            {
                "row_type": "aggregate",
                "analysis": analysis,
                "score": score,
                "arm": arm,
                "ensemble": "ALL",
                "split": np.nan,
                "ess_target": np.nan,
                "eta": np.nan,
                "n": int(finite.size),
                "spearman": float(np.median(finite)) if finite.size else np.nan,
                "spearman_q25": (
                    float(np.quantile(finite, 0.25)) if finite.size else np.nan
                ),
                "spearman_q75": (
                    float(np.quantile(finite, 0.75)) if finite.size else np.nan
                ),
                "negative_fraction": (
                    float(np.mean(finite < 0.0)) if finite.size else np.nan
                ),
                "sign_flip": bool(
                    finite.size > 1 and np.any(finite < 0.0) and np.any(finite > 0.0)
                ),
            }
        )
    return pd.concat([condition, pd.DataFrame(aggregates)], ignore_index=True)


def decoy_correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Correlate observed-only scores with revealed decoy mass.

    The primary rows are Spearman correlations across gamma within each
    (ensemble, split, fixed target ESS) condition.  The KL rows are a
    secondary wide-range robustness check and are deliberately not pooled
    with the fixed-ESS result.
    """

    required = {"decoy_mass", *SCORE_NAMES}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"decoy analysis requires columns: {missing}")
    long = frame.melt(
        id_vars=[
            "grid", "arm", "gamma", "control_value", "ensemble", "split",
            "decoy_mass",
        ],
        value_vars=SCORE_NAMES,
        var_name="score",
        value_name="score_value",
    )
    rows = []
    fixed = long[long["grid"] == "fixed_ess"]
    for key, group in fixed.groupby(
        ["score", "arm", "ensemble", "split", "control_value"], sort=True
    ):
        rows.append({
            "row_type": "condition",
            "analysis": "primary_fixed_ess_across_gamma",
            "score": key[0], "arm": key[1], "ensemble": key[2],
            "split": int(key[3]), "ess_target": float(key[4]), "eta": np.nan,
            "n": int(len(group)),
            "spearman": _safe_spearman(group["score_value"], group["decoy_mass"]),
        })
    kl = long[long["grid"] == "kl"]
    for key, group in kl.groupby(["score", "arm", "ensemble", "split"], sort=True):
        rows.append({
            "row_type": "condition",
            "analysis": "secondary_kl_across_eta_gamma",
            "score": key[0], "arm": key[1], "ensemble": key[2],
            "split": int(key[3]), "ess_target": np.nan, "eta": np.nan,
            "n": int(len(group)),
            "spearman": _safe_spearman(group["score_value"], group["decoy_mass"]),
        })
    condition = pd.DataFrame(rows)
    aggregates = []
    for (analysis, score, arm), group in condition.groupby(
        ["analysis", "score", "arm"], sort=True
    ):
        finite = group["spearman"].dropna().to_numpy()
        aggregates.append({
            "row_type": "aggregate", "analysis": analysis, "score": score,
            "arm": arm, "ensemble": "ALL", "split": np.nan,
            "ess_target": np.nan, "eta": np.nan, "n": int(finite.size),
            "spearman": float(np.median(finite)) if finite.size else np.nan,
            "spearman_q25": float(np.quantile(finite, .25)) if finite.size else np.nan,
            "spearman_q75": float(np.quantile(finite, .75)) if finite.size else np.nan,
            "positive_fraction": float(np.mean(finite > 0.0)) if finite.size else np.nan,
            "sign_flip": bool(finite.size > 1 and np.any(finite < 0) and np.any(finite > 0)),
        })
    result = pd.concat([condition, pd.DataFrame(aggregates)], ignore_index=True)
    result["outcome"] = "decoy_mass"
    result["sign_convention"] = (
        "positive Spearman: lower held-out score tracks lower decoy_mass"
    )
    return result


def e1_ranking_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare Folded and decoy E=1 fits for every score/ensemble/split."""

    subset = frame[
        (frame["grid"] == "fixed_ess") & np.isclose(frame["ess_target"], 1.0)
    ]
    long = subset.melt(
        id_vars=["ensemble", "split", "gamma", "dominant_state", "recovery"],
        value_vars=SCORE_NAMES,
        var_name="score",
        value_name="score_value",
    )
    rows = []
    for key, group in long.groupby(["score", "ensemble", "split"], sort=True):
        folded = group[group["dominant_state"] == "Folded"]
        decoy = group[group["dominant_state"].isin(DECOY_STATES)]
        comparable = not folded.empty and not decoy.empty
        pairwise = (
            np.mean(
                folded["score_value"].to_numpy()[:, None]
                < decoy["score_value"].to_numpy()[None, :]
            )
            if comparable
            else np.nan
        )
        rows.append(
            {
                "score": key[0],
                "ensemble": key[1],
                "split": int(key[2]),
                "n_folded": int(len(folded)),
                "n_decoy": int(len(decoy)),
                "best_folded_score": (
                    float(folded["score_value"].min()) if not folded.empty else np.nan
                ),
                "best_decoy_score": (
                    float(decoy["score_value"].min()) if not decoy.empty else np.nan
                ),
                "reverses_decoy_win": bool(
                    comparable
                    and folded["score_value"].min() < decoy["score_value"].min()
                ),
                "folded_pairwise_win_fraction": (
                    float(pairwise) if np.isfinite(pairwise) else np.nan
                ),
                "comparable": bool(comparable),
            }
        )
    return pd.DataFrame(rows)


def e1_decoy_ranking_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank the lowest- and highest-decoy-mass E=1 configurations per score."""

    subset = frame[
        (frame["grid"] == "fixed_ess") & np.isclose(frame["ess_target"], 1.0)
    ]
    long = subset.melt(
        id_vars=["ensemble", "split", "gamma", "decoy_mass"],
        value_vars=SCORE_NAMES, var_name="score", value_name="score_value",
    )
    rows = []
    for key, group in long.groupby(["score", "ensemble", "split"], sort=True):
        low_mass = group["decoy_mass"].min()
        high_mass = group["decoy_mass"].max()
        low = group[np.isclose(group["decoy_mass"], low_mass)]
        high = group[np.isclose(group["decoy_mass"], high_mass)]
        comparable = bool(not low.empty and not high.empty and low_mass < high_mass)
        low_best = float(low["score_value"].min()) if not low.empty else np.nan
        high_best = float(high["score_value"].min()) if not high.empty else np.nan
        prefers = bool(comparable and low_best < high_best)
        rows.append({
            "score": key[0], "ensemble": key[1], "split": int(key[2]),
            "low_decoy_mass": float(low_mass), "high_decoy_mass": float(high_mass),
            "best_low_decoy_score": low_best, "best_high_decoy_score": high_best,
            "prefers_lower_decoy_mass": prefers,
            "decoy_mass_ordered": "low_decoy_better" if prefers else (
                "high_decoy_better" if comparable and high_best < low_best else "tie_or_incomparable"
            ),
            "comparable": comparable,
        })
    return pd.DataFrame(rows)


def _plot_scatter(frame: pd.DataFrame, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = frame[frame["grid"] == "fixed_ess"]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for axis, score in zip(axes.ravel(), SCORE_NAMES, strict=True):
        scatter = axis.scatter(
            fixed[score],
            fixed["recovery"],
            c=fixed["ess_target"],
            cmap="viridis",
            alpha=0.7,
            s=28,
        )
        rho = _safe_spearman(fixed[score], fixed["recovery"])
        axis.set_title(f"{score} (pooled display ρ={rho:.2f})")
        axis.set_xlabel("held-out score (lower is better)")
        axis.set_ylabel("recovery (%)")
        figure.colorbar(scatter, ax=axis, label="target ESS")
    figure.suptitle(
        "Fixed-ESS score versus recovery; inference/correlations remain per condition"
    )
    figure.tight_layout()
    path = output_dir / "score_vs_recovery_fixed_ess.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_decoy_scatter(frame: pd.DataFrame, correlations: pd.DataFrame, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fixed = frame[frame["grid"] == "fixed_ess"]
    condition = correlations[
        (correlations["row_type"] == "condition")
        & (correlations["analysis"] == "primary_fixed_ess_across_gamma")
    ]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for axis, score in zip(axes.ravel(), SCORE_NAMES, strict=True):
        scatter = axis.scatter(
            fixed[score], fixed["decoy_mass"], c=fixed["ess_target"],
            cmap="viridis", alpha=0.7, s=28,
        )
        rhos = condition[condition["score"] == score]["spearman"].dropna().to_numpy()
        median = float(np.median(rhos)) if rhos.size else np.nan
        q25 = float(np.quantile(rhos, .25)) if rhos.size else np.nan
        q75 = float(np.quantile(rhos, .75)) if rhos.size else np.nan
        axis.set_title(f"{score}: condition ρ median={median:.2f} [{q25:.2f}, {q75:.2f}]")
        axis.set_xlabel("held-out score (lower is better)")
        axis.set_ylabel("decoy mass (lower is better)")
        figure.colorbar(scatter, ax=axis, label="target ESS")
    figure.suptitle("Fixed-ESS score versus decoy mass; correlations are per condition")
    figure.tight_layout()
    path = output_dir / "score_vs_decoy_fixed_ess.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_e1_ranking(ranking: pd.DataFrame, output_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = ranking[
        (ranking["ensemble"] == "AF2_MSAss") & (ranking["split"] == 0)
    ].set_index("score").reindex(SCORE_NAMES)
    x = np.arange(len(SCORE_NAMES))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.bar(
        x - width / 2,
        target["best_folded_score"],
        width,
        label="best Folded",
    )
    axis.bar(
        x + width / 2,
        target["best_decoy_score"],
        width,
        label="best decoy",
    )
    axis.set_xticks(x, SCORE_NAMES, rotation=25, ha="right")
    axis.set_ylabel("held-out score (lower is better)")
    axis.set_title("E=1 AF2_MSAss split 0 — Folded versus decoy ranking")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = output_dir / "e1_folded_vs_decoy_ranking.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _summary(
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    ranking: pd.DataFrame,
) -> dict:
    aggregate = correlations[
        (correlations["row_type"] == "aggregate")
        & (correlations["analysis"] == "primary_fixed_ess_across_gamma")
        & (correlations["arm"] == "diag_d_scalefree")
    ].copy()
    aggregate = aggregate.sort_values(
        ["spearman", "spearman_q75"], na_position="last"
    )
    e1_target = ranking[
        (ranking["ensemble"] == "AF2_MSAss") & (ranking["split"] == 0)
    ].set_index("score")
    candidates = []
    for row in aggregate.itertuples():
        e1 = e1_target.loc[row.score] if row.score in e1_target.index else None
        qualifies = bool(
            np.isfinite(row.spearman)
            and row.spearman <= -0.3
            and row.spearman_q75 < 0.0
            and row.negative_fraction >= 2.0 / 3.0
            and e1 is not None
            and bool(e1["reverses_decoy_win"])
        )
        candidates.append(
            {
                "score": row.score,
                "median_fixed_ess_spearman": row.spearman,
                "spearman_iqr": [row.spearman_q25, row.spearman_q75],
                "negative_condition_fraction": row.negative_fraction,
                "reverses_msass_split0_e1_decoy_win": (
                    bool(e1["reverses_decoy_win"]) if e1 is not None else False
                ),
                "qualifies": qualifies,
            }
        )
    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    winner = qualifying[0]["score"] if qualifying else None
    mse_anchor = e1_target.loc["mse"] if "mse" in e1_target.index else None
    sanity = _persisted_sanity(frame)
    return {
        "lower_score_means_better_fit": True,
        "weighting_inputs": "held-out observed uptake and peptide mapping only",
        "fitted_weight_or_ensemble_variance_used_in_scores": False,
        "structured_residual_nll_included": False,
        "structured_residual_exclusion_reason": (
            "It requires a separately fitted covariance model and is not an "
            "observed-only weighting of the fitted mean residual."
        ),
        "qualification_rule": {
            "median_fixed_ess_spearman_at_most": -0.3,
            "spearman_q75_below": 0.0,
            "negative_condition_fraction_at_least": 2.0 / 3.0,
            "must_reverse_msass_split0_e1_decoy_win": True,
        },
        "persisted_mse_sanity": sanity,
        "mse_reproduces_known_msass_split0_e1_decoy_win": bool(
            mse_anchor is not None
            and mse_anchor["comparable"]
            and not mse_anchor["reverses_decoy_win"]
        ),
        "candidate_results": candidates,
        "winning_score": winner,
        "decision": (
            f"Adopt {winner} as the Experiment-2 mean gate."
            if winner is not None
            else (
                "No observed-only mean-fidelity score qualifies; do not run "
                "Experiment 2 as-is. Reconsider the reweight-against-the-mean premise."
            )
        ),
    }


def _decoy_summary(
    frame: pd.DataFrame,
    correlations: pd.DataFrame,
    ranking: pd.DataFrame,
) -> dict:
    aggregate = correlations[
        (correlations["row_type"] == "aggregate")
        & (correlations["analysis"] == "primary_fixed_ess_across_gamma")
        & (correlations["arm"] == "diag_d_scalefree")
    ].sort_values(["spearman", "spearman_q75"], ascending=False, na_position="last")
    e1_target = ranking[
        (ranking["ensemble"] == "AF2_MSAss") & (ranking["split"] == 0)
    ].set_index("score")
    candidates = []
    for row in aggregate.itertuples():
        e1 = e1_target.loc[row.score] if row.score in e1_target.index else None
        qualifies = bool(
            np.isfinite(row.spearman)
            and row.spearman >= 0.3
            and row.spearman_q25 > 0.0
            and row.positive_fraction >= 2.0 / 3.0
            and e1 is not None
            and bool(e1["prefers_lower_decoy_mass"])
        )
        candidates.append({
            "score": row.score,
            "median_fixed_ess_spearman": row.spearman,
            "spearman_iqr": [row.spearman_q25, row.spearman_q75],
            "positive_condition_fraction": row.positive_fraction,
            "prefers_lower_decoy_mass_at_msass_split0_e1": (
                bool(e1["prefers_lower_decoy_mass"]) if e1 is not None else False
            ),
            "qualifies": qualifies,
        })
    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    winner = qualifying[0]["score"] if qualifying else None
    anchor = e1_target.loc["mse"] if "mse" in e1_target.index else None
    return {
        "outcome": "decoy_mass",
        "lower_score_means_better_fit": True,
        "qualifying_signal": "positive Spearman(score, decoy_mass): low score co-occurs with low decoy mass",
        "blind_score_inputs": ["held-out observed uptake", "held-out peptide mapping"],
        "decoy_mass_is_posthoc_only": True,
        "fixed_ess_is_headline": True,
        "decoy_mass_range": [float(frame["decoy_mass"].min()), float(frame["decoy_mass"].max())],
        "e1_msass_split0_anchor": {
            "max_decoy_mass": float(frame[
                (frame["grid"] == "fixed_ess")
                & (frame["ensemble"] == "AF2_MSAss")
                & (frame["split"] == 0)
                & np.isclose(frame["ess_target"], 1.0)
            ]["decoy_mass"].max()),
            "mse_prefers_lower_decoy_mass": bool(
                anchor is not None and anchor["prefers_lower_decoy_mass"]
            ),
        },
        "qualification_rule": {
            "median_fixed_ess_spearman_at_least": 0.3,
            "spearman_q25_above": 0.0,
            "positive_condition_fraction_at_least": 2.0 / 3.0,
            "must_prefer_lower_decoy_mass_at_msass_split0_e1": True,
        },
        "candidate_results": candidates,
        "e1_decoy_ranking": ranking.to_dict(orient="records"),
        "winning_score": winner,
        "decision": (
            f"{winner} qualifies as a decoy-avoidance gate; reopen Experiment 2 under decoy suppression."
            if winner is not None
            else "No observed-only mean-fidelity score qualifies for decoy avoidance; Experiment 2 stays shelved."
        ),
    }


def _write_decoy_final(frame: pd.DataFrame, output_dir: Path) -> None:
    """Write only the decoy re-analysis artifacts; recovery files are untouched."""

    if "decoy_mass" not in frame.columns:
        raise ValueError("source cell table has no decoy_mass column")
    if int(len(frame)) != 630:
        raise AssertionError(f"unexpected source cell count: {len(frame)} (expected 630)")
    if frame["mse"].isna().any():
        raise AssertionError("source cell table contains missing mse values")
    if not (frame["decoy_mass"].min() <= 1e-12 and frame["decoy_mass"].max() >= 0.99):
        raise AssertionError("decoy_mass does not span the expected 0-to-1 anchor range")
    output_dir.mkdir(parents=True, exist_ok=True)
    correlations = decoy_correlation_table(frame)
    ranking = e1_decoy_ranking_table(frame)
    summary = _decoy_summary(frame, correlations, ranking)
    correlations.to_csv(output_dir / "val_score_decoy_correlation.csv", index=False)
    (output_dir / "val_score_decoy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _plot_decoy_scatter(frame, correlations, output_dir)
    print(summary["decision"])


def decoy_run(args: argparse.Namespace) -> None:
    """Run the pure re-analysis from the already-produced cell table."""

    source = args.source_cells
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    _write_decoy_final(frame, args.output_dir)


def _write_final(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    sanity = _persisted_sanity(frame)
    failed = {
        grid: report
        for grid, report in sanity.items()
        if not report["passes_1e_6"]
    }
    if failed:
        raise AssertionError(f"MSE did not reproduce persisted val_mse: {failed}")
    correlations = correlation_table(frame)
    ranking = e1_ranking_table(frame)
    summary = _summary(frame, correlations, ranking)
    frame.to_csv(args.output_dir / "val_score_cells.csv", index=False)
    correlations.to_csv(args.output_dir / "val_score_correlation.csv", index=False)
    ranking.to_csv(args.output_dir / "e1_folded_vs_decoy.csv", index=False)
    (args.output_dir / "val_score_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _plot_scatter(frame, args.output_dir)
    _plot_e1_ranking(ranking, args.output_dir)
    manifest = {
        "artifact_type": "moprp_observed_only_validation_score_correlation",
        "grids": list(args.grids),
        "scores": list(SCORE_NAMES),
        "fixed_ess_source": str(args.fixed_artifact),
        "kl_source": str(args.kl_artifact),
        "optimizer_invoked_by_scoring": False,
        "nmr_used_in_fit_or_score": False,
        "nmr_used_posthoc_for_correlation": True,
        "per_ensemble_per_split_never_pooled_for_inference": True,
        "score_weight_inputs": ["held-out observed uptake", "held-out peptide mapping"],
        "outputs": [
            "val_score_cells.csv",
            "val_score_correlation.csv",
            "e1_folded_vs_decoy.csv",
            "val_score_summary.json",
            "score_vs_recovery_fixed_ess.png",
            "e1_folded_vs_decoy_ranking.png",
        ],
    }
    (args.output_dir / "val_score_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary["decision"])


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, base_cells = _records_from_first_party_payloads(args)
    _reveal_recovery(records, base_cells)
    frame = pd.DataFrame(records)
    if "persisted_recovery" in frame.columns:
        recovery_error = np.max(
            np.abs(frame["recovery"] - frame["persisted_recovery"])
        )
        if recovery_error > 1e-6:
            raise AssertionError(
                f"payload recovery does not reproduce persisted rows: {recovery_error}"
            )
    _write_final(frame, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--grids",
        nargs="+",
        choices=tuple(_grid_specs()),
        default=tuple(_grid_specs()),
    )
    parser.add_argument("--target-artifact", type=Path, default=joint.TARGET_ARTIFACT)
    parser.add_argument("--fixed-artifact", type=Path, default=FIXED_ARTIFACT)
    parser.add_argument("--kl-artifact", type=Path, default=KL_ARTIFACT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--decoy-only",
        action="store_true",
        help="re-analyze the existing cell table and write only decoy deliverables",
    )
    parser.add_argument(
        "--source-cells",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "val_score_cells.csv",
    )
    args = parser.parse_args()
    args.grids = tuple(args.grids)
    if args.decoy_only:
        decoy_run(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
