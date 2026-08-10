#!/usr/bin/env python3
"""Post-process the population-pivot sweep against truth-weighted log-PF spread.

No fitting is performed here.  The pre-registered falsification criterion is that adding
population family to the spread-by-pivot model is meaningful if partial R2 >= 0.02 or if
Delta AIC = AIC(M1) - AIC(M2) >= 2.  The spread mechanism is confirmed only when neither
threshold is reached and the family spread ranges overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import _moprp_recovery_common as common
import moprp_population_oracle as oracle
import moprp_population_pivot as population_pivot
from jaxent.src.analysis.state_population import FULL_STATE_SUPPORT


PIVOTS = population_pivot.PIVOTS
FAMILY_PARTIAL_R2_THRESHOLD = 0.02
FAMILY_DELTA_AIC_THRESHOLD = 2.0
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260807


def weighted_log_pf_variance(log_pf: np.ndarray, weights: np.ndarray) -> float:
    """Mean over residues of the weighted frame variance of log PF."""

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    z = np.asarray(log_pf, dtype=float)
    centre = z @ weights
    return float(np.mean(np.sum(weights[None, :] * (z - centre[:, None]) ** 2, axis=1)))


def _design(frame: pd.DataFrame, include_family: bool) -> tuple[np.ndarray, list[str]]:
    spread = frame["var_w_log_pf"].to_numpy(float)
    cols = [np.ones(len(frame)), spread]
    names = ["intercept", "var_w_log_pf"]
    for pivot in PIVOTS[1:]:
        indicator = (frame["target_pivot"] == pivot).to_numpy(float)
        cols.extend((indicator, indicator * spread))
        names.extend((f"pivot[{pivot}]", f"var_w:pivot[{pivot}]"))
    if include_family:
        for family in population_pivot.POPULATION_FAMILIES[1:]:
            cols.append((frame["population_family"] == family).to_numpy(float))
            names.append(f"family[{family}]")
    return np.column_stack(cols), names


def _ols(frame: pd.DataFrame, include_family: bool) -> dict:
    x, names = _design(frame, include_family)
    y = frame["tvd_gain_over_null"].to_numpy(float)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    rss = float(residual @ residual)
    tss = float(np.sum((y - y.mean()) ** 2))
    n, k = len(y), x.shape[1]
    aic = float(n * np.log(rss / n) + 2 * k)
    return {
        "n": n,
        "rank": int(rank),
        "parameters": k,
        "rss": rss,
        "r_squared": 1.0 - rss / tss,
        "aic": aic,
        "coefficients": dict(zip(names, beta.tolist())),
    }


def _pivot_quadratics(diagonal: pd.DataFrame) -> dict:
    curves = {}
    for pivot in PIVOTS:
        cell = diagonal[diagonal.target_pivot == pivot]
        coeff = np.polyfit(cell.var_w_log_pf, cell.tvd_gain_over_null, 2)
        curves[pivot] = coeff
    roots = np.roots(curves["fast"] - curves["slow-N"])
    lo, hi = diagonal.var_w_log_pf.min(), diagonal.var_w_log_pf.max()
    valid = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9 and lo <= r.real <= hi)
    return {"coefficients_high_to_low": {p: curves[p].tolist() for p in PIVOTS}, "crossings": valid}


def _linear_summary(frame: pd.DataFrame) -> dict:
    result = stats.linregress(frame.predicted_spread_factor, frame.mismatch_tvd_penalty)
    rank = stats.spearmanr(frame.predicted_spread_factor, frame.mismatch_tvd_penalty)
    return {
        "n": len(frame), "slope": result.slope, "intercept": result.intercept,
        "pearson_r": result.rvalue, "p_value": result.pvalue,
        "slope_standard_error": result.stderr,
        "spearman_rho": float(rank.statistic), "spearman_p_value": float(rank.pvalue),
    }


def _bootstrap_crossing(diagonal: pd.DataFrame, point: float | None) -> dict:
    if point is None:
        return {"replicates": BOOTSTRAP_REPLICATES, "valid": 0, "ci95": None}
    keys = diagonal[["population_family", "coefficient_setting", "minority_mass"]].drop_duplicates()
    groups = [
        diagonal[
            (diagonal.population_family == row.population_family)
            & (diagonal.coefficient_setting == row.coefficient_setting)
            & (diagonal.minority_mass == row.minority_mass)
        ]
        for row in keys.itertuples(index=False)
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    crossings = []
    lo, hi = diagonal.var_w_log_pf.min(), diagonal.var_w_log_pf.max()
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            curves = {
                p: np.polyfit(c.var_w_log_pf, c.tvd_gain_over_null, 2)
                for p in PIVOTS
                if len((c := sample[sample.target_pivot == p])) >= 3
            }
            roots = np.roots(curves["fast"] - curves["slow-N"])
            valid = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and lo <= r.real <= hi]
            if valid:
                crossings.append(min(valid, key=lambda value: abs(value - point)))
        except (ValueError, np.linalg.LinAlgError):
            continue
    ci = np.quantile(crossings, [0.025, 0.975]).tolist() if crossings else None
    return {"replicates": BOOTSTRAP_REPLICATES, "valid": len(crossings), "ci95": ci}


def run(args: argparse.Namespace) -> None:
    inputs = common.load_ensemble_inputs("AF2_MSAss")
    coeffs = population_pivot._coefficient_settings(args.coefficient_lock)
    present, weight_map = population_pivot._population_map(inputs)
    sweep = pd.read_csv(args.sweep_csv)
    expected = {"population_family", "coefficient_setting", "target_pivot", "fitter_pivot", "minority_mass", "tvd", "tvd_gain_over_null"}
    missing = expected - set(sweep.columns)
    if missing:
        raise ValueError(f"sweep CSV lacks required columns: {sorted(missing)}")

    spread_rows = []
    unique = sweep[["population_family", "coefficient_setting", "target_pivot", "minority_mass"]].drop_duplicates()
    for row in unique.itertuples(index=False):
        truth = population_pivot._target_population(inputs, row.minority_mass, row.population_family)
        state_population = oracle._target_populations(present, FULL_STATE_SUPPORT, truth)
        weights = np.asarray(weight_map @ state_population)
        coefficient = coeffs[row.target_pivot][row.coefficient_setting]
        log_pf = inputs.log_pf_by_frame(coefficient["bc"], coefficient["bh"])
        spread_rows.append({**row._asdict(), "var_w_log_pf": weighted_log_pf_variance(log_pf, weights)})
    spreads = pd.DataFrame(spread_rows)
    enriched = sweep.merge(spreads, validate="many_to_one")
    if len(enriched) != 396:
        raise AssertionError(f"join changed sweep row count: {len(enriched)} != 396")
    diagonal = enriched[enriched.target_pivot == enriched.fitter_pivot].copy()
    off_diagonal = enriched[enriched.target_pivot != enriched.fitter_pivot].copy()
    if (len(diagonal), len(off_diagonal)) != (132, 264):
        raise AssertionError("expected 132 diagonal and 264 off-diagonal rows")

    m1, m2 = _ols(diagonal, False), _ols(diagonal, True)
    partial_r2 = (m1["rss"] - m2["rss"]) / m1["rss"]
    delta_aic = m1["aic"] - m2["aic"]
    ranges = diagonal.groupby("population_family").var_w_log_pf.agg(["min", "max"])
    pairwise_overlap = {}
    families = list(ranges.index)
    for i, left in enumerate(families):
        for right in families[i + 1:]:
            width = max(0.0, min(ranges.loc[left, "max"], ranges.loc[right, "max"]) - max(ranges.loc[left, "min"], ranges.loc[right, "min"]))
            pairwise_overlap[f"{left}__{right}"] = width
    overlap_exists = any(value > 0 for value in pairwise_overlap.values())
    family_mean = diagonal.groupby("population_family").var_w_log_pf.mean().to_dict()

    curves = _pivot_quadratics(diagonal)
    crossing = curves["crossings"][0] if curves["crossings"] else None
    crossing_uncertainty = _bootstrap_crossing(diagonal, crossing)

    join_keys = ["population_family", "coefficient_setting", "target_pivot", "minority_mass"]
    matched = diagonal[join_keys + ["tvd"]].rename(columns={"tvd": "matched_tvd"})
    off_diagonal = off_diagonal.merge(matched, on=join_keys, validate="many_to_one")
    off_diagonal["mismatch_tvd_penalty"] = off_diagonal.tvd - off_diagonal.matched_tvd
    off_diagonal["predicted_spread_factor"] = np.exp(0.5 * off_diagonal.var_w_log_pf) - 1.0
    mismatch_all = _linear_summary(off_diagonal)
    legacy_fast = off_diagonal[
        off_diagonal.target_pivot.isin(("legacy", "fast"))
        & off_diagonal.fitter_pivot.isin(("legacy", "fast"))
    ]
    mismatch_legacy_fast = _linear_summary(legacy_fast)

    real_pop = oracle._target_populations(present, FULL_STATE_SUPPORT, inputs.targets)
    real_weights = np.asarray(weight_map @ real_pop)
    real_spread = {}
    for pivot in PIVOTS:
        coefficient = coeffs[pivot]["constrained_optimum"]
        real_spread[pivot] = weighted_log_pf_variance(
            inputs.log_pf_by_frame(coefficient["bc"], coefficient["bh"]), real_weights
        )

    one = np.zeros(inputs.n_frames)
    one[0] = 1.0
    z = inputs.log_pf_by_frame(common.PUBLISHED_BC, common.PUBLISHED_BH)
    one_frame_variance = weighted_log_pf_variance(z, one)
    folded_truth = np.zeros(len(FULL_STATE_SUPPORT))
    folded_truth[FULL_STATE_SUPPORT.index("Folded")] = 1.0
    folded_weights = np.asarray(weight_map @ oracle._target_populations(present, FULL_STATE_SUPPORT, folded_truth))
    one_state_variance = weighted_log_pf_variance(z, folded_weights)
    scale_base = weighted_log_pf_variance(z, np.asarray(real_weights))
    scale_double = weighted_log_pf_variance(2.0 * z, np.asarray(real_weights))

    equal_thirds = population_pivot._target_population(inputs, 0.0, "balanced")
    equal_weights = np.asarray(weight_map @ oracle._target_populations(present, FULL_STATE_SUPPORT, equal_thirds))
    locked_equal_thirds = {}
    for pivot in PIVOTS:
        coefficient = coeffs[pivot]["constrained_optimum"]
        locked_equal_thirds[pivot] = weighted_log_pf_variance(
            inputs.log_pf_by_frame(coefficient["bc"], coefficient["bh"]), equal_weights
        )
    locked_grid_means = (
        diagonal[diagonal.coefficient_setting == "constrained_optimum"]
        .groupby("population_family").var_w_log_pf.mean().to_dict()
    )

    mechanism_confirmed = overlap_exists and partial_r2 < FAMILY_PARTIAL_R2_THRESHOLD and delta_aic < FAMILY_DELTA_AIC_THRESHOLD
    payload = {
        "pre_registered_criterion": {
            "family_is_meaningful_if_partial_r_squared_at_least": FAMILY_PARTIAL_R2_THRESHOLD,
            "or_delta_aic_m1_minus_m2_at_least": FAMILY_DELTA_AIC_THRESHOLD,
            "confirmation_also_requires_family_spread_overlap": True,
        },
        "row_counts": {"all": len(enriched), "diagonal": len(diagonal), "off_diagonal": len(off_diagonal), "spread_evaluations": len(spreads)},
        "spread": {
            "family_grid_means": family_mean,
            "family_ranges": ranges.to_dict(orient="index"),
            "pairwise_overlap_widths": pairwise_overlap,
            "overlap_exists": overlap_exists,
            "locked_grid_means": locked_grid_means,
            "balanced_equal_thirds_locked_by_pivot": locked_equal_thirds,
            "one_frame_variance": one_frame_variance,
            "one_state_uniform_within_state_variance": one_state_variance,
            "quadratic_scaling_ratio": scale_double / scale_base,
            "real_nmr_locked_by_pivot": real_spread,
        },
        "primary_regression": {
            "m1_spread_pivot_interaction": m1,
            "m2_plus_population_family": m2,
            "family_partial_r_squared": partial_r2,
            "delta_aic_m1_minus_m2": delta_aic,
            "mechanism_confirmed": mechanism_confirmed,
            "quadratic_curves": curves,
            "fast_slow_n_crossing": crossing,
            "crossing_cluster_bootstrap": crossing_uncertainty,
        },
        "secondary_mismatch_regression": {
            "definition": "off-diagonal TVD minus matched diagonal TVD for the same generated truth",
            "predictor": "exp(0.5 * Var_w(log PF)) - 1",
            "all_off_diagonal": mismatch_all,
            "legacy_fast_only": mismatch_legacy_fast,
        },
        "caveats": [
            "Spread is shared by three diagonal pivot rows per generated truth, so nominal N=132 overstates the effective sample size.",
            "The crossover bootstrap resamples generated truths as clusters and remains indicative rather than an instrument-scale confidence interval.",
            "Coefficient setting moves observations along the spread axis and is not a clean nuisance factor.",
            "A one-state population is not a one-frame population under uniform-within-state weights and generally has nonzero within-state spread.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "population_spread_regression.json").write_text(json.dumps(payload, indent=2) + "\n")
    enriched.to_csv(args.output_dir / "synthetic_resolution_sweep_with_spread.csv", index=False)
    off_diagonal.to_csv(args.output_dir / "off_diagonal_mismatch_with_spread.csv", index=False)
    print(json.dumps({"mechanism_confirmed": mechanism_confirmed, "partial_r2": partial_r2, "delta_aic": delta_aic, "crossing": crossing}, indent=2))


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-csv", type=Path, default=here / "_moprp_population_oracle_pivot/synthetic_resolution_sweep.csv")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_population_oracle_pivot")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
