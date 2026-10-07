#!/usr/bin/env python3
"""Pivot-convention experiment for MoPrP target-variance inference.

This script compares `k̄_after = E_f[k]` (current frozen runner convention)
to `k̄_first = k_int exp(-E_f[ln PF])` in a two-tier protocol. Blind
inference is preserved exactly as in MoPrP runs; NMR is revealed only after all
fits are computed and saved in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr

import _moprp_recovery_common as common
import validate_moprp_target_variance as validation
from jaxent.src.analysis.hdx_target_variance import (
    build_rate_geometries,
    build_hdx_covariance,
    covariance_profiles,
    effective_rates,
    fit_curve_moment_variance,
    fit_structured_residual_variance,
    map_hdx_covariance,
    population_covariance,
    predict_curve_moment_uptake,
    predict_fixed_mean_uptake,
    qualification_gate,
    structured_residual_nll,
    variance_recovery_metrics,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "_pivot_convention"
DEFAULT_COEFFICIENT_LOCK = HERE / "_moprp_recovery_coefficient_lock" / "coefficient_lock.json"
DEFAULT_ARTIFACTS = {
    "scaled_published": HERE / "_moprp_target_variance_scaled_published_20260723",
    "constrained_optimum": HERE / "_moprp_target_variance_constrained_optimum_20260723",
}
DEFAULT_COEFFICIENT_SETTINGS = ("scaled_published", "constrained_optimum")

ESTIMATORS = ("curve_moment", "structured_residual")
TIER1_MEDIAN_THRESHOLD = 0.02
TIER1_P90_THRESHOLD = 0.05
JENSEN_GUARD_TOL = 1e-12
REPLAY_OBJECTIVE_TOL = 1e-6
REPLAY_D_RMSE_TOL = 1e-2
ARCHIVED_MAPPED_RMSE_TOL = 0.05
ARCHIVED_D_ORDERING_MIN = 0.90
NOISE_VARIANCE = 1e-4
DEFAULT_MAXITER = 300
NEIGHBOR_STRENGTH = 0.25
SHUFFLE_SEED = 20260722
PEPTIDE_FOLDS = 3
TIME_FOLDS = 5
FOLD_SCHEME = "interleaved"


def load_bv_setting(path: Path, setting: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    try:
        frozen = payload["frozen_settings"][setting]
    except KeyError as error:
        available = sorted(payload.get("frozen_settings", {}).keys())
        raise ValueError(f"unknown coefficient setting {setting!r}; available={available}") from error
    bc = float(frozen["bc"])
    bh = float(frozen["bh"])
    if not np.isfinite([bc, bh]).all() or bc < 0.0 or bh < 0.0:
        raise ValueError(f"invalid frozen coefficient values for {setting}: bc={bc}, bh={bh}")
    return {
        "setting": setting,
        "bc": bc,
        "bh": bh,
        "coefficient_lock_path": str(path),
        "coefficient_lock_sha256": common.sha256(path),
    }


def _log_rmse(left: np.ndarray, right: np.ndarray, floor: float = 1e-15) -> float:
    residual = np.log(np.clip(left, floor, None)) - np.log(np.clip(right, floor, None))
    return float(np.sqrt(np.mean(np.square(residual))))


def _load_archived_artifact(path: Path, coefficient: str) -> dict[str, Any]:
    manifest = json.loads((path / "blinded_selection_manifest.json").read_text())
    if "selection" not in manifest:
        raise ValueError(f"artifact {path} missing mandatory selection payload")
    return {
        "coefficient": coefficient,
        "path": path,
        "selection": manifest["selection"],
        "sweep": pd.read_csv(path / "blinded_hdx_sweep.csv"),
        "nmr": pd.read_csv(path / "nmr_pseudotruth_diagnostic_metrics.csv"),
    }


def _find_archived_candidate(
    artifact: dict[str, Any],
    *,
    ensemble: str,
    estimator: str,
    geometry: str,
    regularization: float,
    peptide_fold: int = 0,
    time_fold: int = 0,
) -> dict[str, Any]:
    sweep = artifact["sweep"]
    rows = sweep[
        (sweep["ensemble"] == ensemble)
        & (sweep["estimator"] == estimator)
        & (sweep["geometry"] == geometry)
        & np.isclose(sweep["regularization"], regularization)
    ]
    if rows.empty:
        raise ValueError(
            f"no archived candidate for {ensemble}:{estimator}:{geometry} reg={regularization}"
        )
    candidates = rows[(rows["peptide_fold"] == peptide_fold) & (rows["time_fold"] == time_fold)]
    if candidates.empty:
        candidates = rows
    row = candidates.iloc[0]
    candidate_id = str(row["candidate_id"])
    with np.load(artifact["path"] / "blinded_variances.npz") as data:
        variances = np.asarray(data[f"{candidate_id}__variances"], dtype=float)
    nmr_match = artifact["nmr"][artifact["nmr"]["candidate_id"] == candidate_id]
    if nmr_match.empty:
        raise ValueError(f"missing NMR metric row for archived candidate {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "row": row.to_dict(),
        "variances": variances,
        "nmr": nmr_match.iloc[0].to_dict(),
    }


def _beta_from_variance(mean_rates: np.ndarray, variances: np.ndarray) -> np.ndarray:
    return np.log(np.clip(variances, 1e-300, None) / np.clip(np.square(mean_rates), 1e-300, None))


def _fit_constant_variance(
    estimator: str,
    observed: np.ndarray,
    mean_rates: np.ndarray,
    times: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    geometry_scale = float(np.exp(np.mean(np.log(np.square(mean_rates)))))
    if mask is None:
        mask = np.ones_like(observed, dtype=bool)

    def objective(log_relative_variance: float) -> tuple[float, float, np.ndarray]:
        variance = geometry_scale * float(np.exp(log_relative_variance))
        variances = np.full_like(mean_rates, variance)
        covariance = build_hdx_covariance(variances, geometry)
        if estimator == "curve_moment":
            predicted = predict_curve_moment_uptake(mean_rates, variances, times, mapping)
            value = float(np.mean(np.square(predicted - observed)[mask]))
            return value, value, covariance
        if estimator == "structured_residual":
            fixed_mean = predict_fixed_mean_uptake(mean_rates, times, mapping)
            value = structured_residual_nll(
                observed,
                mean_rates,
                times,
                mapping,
                covariance,
                noise_variance=NOISE_VARIANCE,
                observation_mask=mask,
                predicted_uptake=fixed_mean,
            )
            return value, value, covariance
        raise ValueError(f"unsupported estimator {estimator!r}")

    result = minimize_scalar(
        lambda value: objective(float(value))[0],
        bounds=(-18.0, 8.0),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 300},
    )
    _, value, covariance = objective(float(result.x))
    return {
        "variance": geometry_scale * float(np.exp(result.x)),
        "covariance": covariance,
        "objective": float(value),
        "success": bool(result.success and np.isfinite(value)),
    }


def _fit_one(
    *,
    estimator: str,
    observed: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    geometry_name: str,
    regularization: float,
    maxiter: int,
    observation_mask: np.ndarray | None = None,
) -> Any:
    kwargs = dict(
        observed_uptake=observed,
        mean_rates=mean_rates,
        timepoints=timepoints,
        mapping=mapping,
        geometry=geometry,
        geometry_name=geometry_name,
        regularization=regularization,
        maxiter=maxiter,
        noise_variance=NOISE_VARIANCE,
        initial_variance=0.1,
    )
    if observation_mask is not None:
        kwargs["observation_mask"] = observation_mask
    if estimator == "curve_moment":
        kwargs.pop("noise_variance")
        return fit_curve_moment_variance(**kwargs)
    if estimator == "structured_residual":
        return fit_structured_residual_variance(**kwargs)
    raise ValueError(f"unsupported estimator {estimator!r}")


def _run_tier1_row(
    coefficient: str,
    ensemble: str,
    rates: np.ndarray,
    mean_after: np.ndarray,
    mean_first: np.ndarray,
    log_pf: np.ndarray,
) -> dict[str, Any]:
    rel_gap = np.abs(mean_after - mean_first) / np.clip(np.abs(mean_first), 1e-300, None)
    guard = mean_after - mean_first
    approx_residual = rel_gap - 0.5 * np.var(log_pf, axis=1)
    residual_abs = np.abs(approx_residual)
    return {
        "coefficient": coefficient,
        "ensemble": ensemble,
        "median_rel_gap": float(np.median(rel_gap)),
        "p90_rel_gap": float(np.quantile(rel_gap, 0.90)),
        "max_rel_gap": float(np.max(rel_gap)),
        "jensen_guard_violations": int(np.sum(guard < -JENSEN_GUARD_TOL)),
        "median_abs_jensen_residual": float(np.median(residual_abs)),
        "p90_abs_jensen_residual": float(np.quantile(residual_abs, 0.90)),
        "max_abs_jensen_residual": float(np.max(residual_abs)),
    }


def _folds(size: int, count: int, scheme: str = FOLD_SCHEME) -> tuple[np.ndarray, ...]:
    if size < count:
        raise ValueError("cannot create more non-empty folds than observations")
    if scheme == "interleaved":
        return tuple(np.arange(offset, size, count, dtype=int) for offset in range(count))
    if scheme == "contiguous":
        return tuple(np.asarray(fold, dtype=int) for fold in np.array_split(np.arange(size), count))
    raise ValueError("fold scheme must be 'interleaved' or 'contiguous'")


def _pilot_train_mask(mapping_rows: int, n_timepoints: int) -> np.ndarray:
    peptide_folds = _folds(mapping_rows, PEPTIDE_FOLDS)
    time_folds = _folds(n_timepoints, TIME_FOLDS)
    train_peptides = np.setdiff1d(np.arange(mapping_rows), peptide_folds[0])
    train_times = np.setdiff1d(np.arange(n_timepoints), time_folds[0])
    mask = np.zeros((mapping_rows, n_timepoints), dtype=bool)
    mask[np.ix_(train_peptides, train_times)] = True
    return mask


def _heldout_row_mse_ratio(
    estimator: str,
    mean_rates: np.ndarray,
    variances: np.ndarray,
    full_mapping: np.ndarray,
    times: np.ndarray,
    peptide1_row: int,
    observed_full: np.ndarray,
) -> float:
    if estimator == "curve_moment":
        predicted = predict_curve_moment_uptake(mean_rates, variances, times, full_mapping)
    elif estimator == "structured_residual":
        predicted = predict_fixed_mean_uptake(mean_rates, times, full_mapping)
    else:
        raise ValueError(f"unsupported estimator {estimator!r}")
    peptide_pred = np.asarray(predicted, dtype=float)[peptide1_row]
    peptide_obs = np.asarray(observed_full, dtype=float)[peptide1_row]
    baseline = predict_fixed_mean_uptake(mean_rates, times, full_mapping)[peptide1_row]
    heldout_mse = float(np.mean(np.square(peptide_pred - peptide_obs)))
    baseline_mse = float(np.mean(np.square(baseline - peptide_obs)))
    return heldout_mse / max(baseline_mse, 1e-15)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory {args.output_dir}")

    coefficient_names = tuple(
        name.strip() for name in args.coefficient_settings.split(",") if name.strip()
    )
    if not coefficient_names:
        raise ValueError("no coefficient setting names requested")
    unknown = [name for name in coefficient_names if name not in DEFAULT_ARTIFACTS]
    if unknown:
        raise ValueError(
            f"unsupported coefficient settings {unknown}; expected one of {sorted(DEFAULT_ARTIFACTS)}"
        )

    coefficients = {name: load_bv_setting(args.coefficient_lock, name) for name in coefficient_names}
    artifacts = {name: _load_archived_artifact(DEFAULT_ARTIFACTS[name], name) for name in coefficient_names}

    tier1_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []

    cells: list[dict[str, Any]] = []
    for coefficient, setting in coefficients.items():
        artifact = artifacts[coefficient]
        primary = artifact["selection"]
        primary_geometry = str(primary["geometry"])
        primary_estimator = str(primary["estimator"])
        primary_regularization = float(primary["regularization"])

        for ensemble in common.ENSEMBLES:
            inputs = common.load_blinded_ensemble_inputs(ensemble)
            partitions = validation.peptide_partitions(inputs)
            fit_rows = np.asarray(partitions["fit_rows"], dtype=int)
            peptide1 = int(partitions["peptide1_row"])

            observed = np.asarray(inputs.observed_uptake, dtype=float)[fit_rows]
            mapping = np.asarray(inputs.mapping[fit_rows], dtype=float)
            train_mask = _pilot_train_mask(mapping.shape[0], inputs.timepoints.size)
            log_pf = inputs.log_pf_by_frame(setting["bc"], setting["bh"])
            rates = effective_rates(log_pf, inputs.k_ints)
            mean_after = np.mean(rates, axis=1)
            mean_first = inputs.k_ints * np.exp(-np.mean(log_pf, axis=1))
            mapping_all = np.asarray(inputs.mapping, dtype=float)
            observed_all = np.asarray(inputs.observed_uptake, dtype=float)

            tier1_rows.append(
                _run_tier1_row(coefficient, ensemble, rates, mean_after, mean_first, log_pf)
            )

            geometries = build_rate_geometries(
                rates,
                validation._coordinates(inputs.feature_residue_ids),
                inputs.feature_residue_ids,
                cutoff_angstrom=8.0,
                neighbor_strength=NEIGHBOR_STRENGTH,
                shuffle_seed=SHUFFLE_SEED,
            )

            replay_entry = _find_archived_candidate(
                artifact,
                ensemble=ensemble,
                estimator=primary_estimator,
                geometry=primary_geometry,
                regularization=primary_regularization,
                peptide_fold=0,
                time_fold=0,
            )
            replay_fit = _fit_one(
                estimator=primary_estimator,
                observed=observed,
                mean_rates=mean_after,
                timepoints=inputs.timepoints,
                mapping=mapping,
                geometry=geometries[primary_geometry],
                geometry_name=primary_geometry,
                regularization=primary_regularization,
                observation_mask=train_mask,
                maxiter=args.maxiter,
            )
            replay_d_rmse = float(np.sqrt(np.mean((replay_fit.variances - replay_entry["variances"]) ** 2)))
            replay_objective = float(abs(replay_fit.objective - float(replay_entry["row"]["objective"])))
            replay_ok = bool(
                replay_d_rmse <= REPLAY_D_RMSE_TOL and replay_objective <= REPLAY_OBJECTIVE_TOL
            )
            replay_rows.append(
                {
                    "coefficient": coefficient,
                    "ensemble": ensemble,
                    "estimator": primary_estimator,
                    "geometry": primary_geometry,
                    "regularization": primary_regularization,
                    "candidate_id": replay_entry["candidate_id"],
                    "d_rmse": replay_d_rmse,
                    "objective_abs_diff": replay_objective,
                    "passed": replay_ok,
                }
            )
            if not replay_ok:
                raise RuntimeError(
                    f"pivot replay failed for {coefficient}/{ensemble}/{primary_estimator}/{primary_geometry}: "
                    f"d_rmse={replay_d_rmse:.3e}, objective_abs_diff={replay_objective:.3e}"
                )

            cells.append(
                {
                    "coefficient": coefficient,
                    "ensemble": ensemble,
                    "artifact": artifact,
                    "primary_geometry": primary_geometry,
                    "primary_estimator": primary_estimator,
                    "primary_regularization": primary_regularization,
                    "inputs": inputs,
                    "observed": observed,
                    "observed_all": observed_all,
                    "mapping": mapping,
                    "mapping_all": mapping_all,
                    "peptide1_row": peptide1,
                    "rates": rates,
                    "mean_after": mean_after,
                    "mean_first": mean_first,
                    "geometries": geometries,
                    "train_mask": train_mask,
                }
            )

    tier1_df = pd.DataFrame(tier1_rows)
    tier1_df.to_csv(args.output_dir / "pivot_gap.csv", index=False)
    tier1_pass = bool(
        len(tier1_df) > 0
        and bool(np.all(np.array(tier1_df["median_rel_gap"]) <= TIER1_MEDIAN_THRESHOLD))
        and bool(np.all(np.array(tier1_df["p90_rel_gap"]) <= TIER1_P90_THRESHOLD))
        and bool(np.all(np.array(tier1_df["jensen_guard_violations"]) == 0))
    )

    decision_tier1 = {
        "median_rel_gap_threshold": TIER1_MEDIAN_THRESHOLD,
        "p90_rel_gap_threshold": TIER1_P90_THRESHOLD,
        "cells": int(tier1_df.shape[0]),
        "cells_pass": int(np.sum(tier1_df["median_rel_gap"] <= TIER1_MEDIAN_THRESHOLD)),
        "p90_cells_pass": int(np.sum(tier1_df["p90_rel_gap"] <= TIER1_P90_THRESHOLD)),
        "jensen_cells_pass": int(np.sum(tier1_df["jensen_guard_violations"] == 0)),
        "passed": tier1_pass,
    }

    if args.tier1_only or (not args.force_tier2 and tier1_pass):
        payload = {
            "decision": "future_correctness",
            "tiers_executed": ("tier1",),
            "tier1": decision_tier1,
            "replay": replay_rows,
            "pivot_gap_path": str(args.output_dir / "pivot_gap.csv"),
        }
        (args.output_dir / "pivot_decision.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    for cell in cells:
        coefficient = str(cell["coefficient"])
        ensemble = str(cell["ensemble"])
        inputs = cell["inputs"]
        artifact = cell["artifact"]
        observed = cell["observed"]
        observed_all = cell["observed_all"]
        mapping = cell["mapping"]
        mapping_all = cell["mapping_all"]
        peptide1 = int(cell["peptide1_row"])
        rates = cell["rates"]
        mean_first = cell["mean_first"]
        train_mask = cell["train_mask"]
        primary_geometry = str(cell["primary_geometry"])
        primary_regularization = float(cell["primary_regularization"])
        geometries = cell["geometries"]

        _, _, _, nmr_weights = common.reveal_nmr_reference(inputs.ensemble, expected_frames=inputs.n_frames)
        truth = population_covariance(rates, nmr_weights)
        true_variances = np.diag(truth)
        mapped_truth = map_hdx_covariance(truth, mapping)
        true_marginal, true_conditional = covariance_profiles(mapped_truth)

        fit_records: list[dict[str, Any]] = []
        for estimator in ESTIMATORS:
            for geometry_name in (primary_geometry, "shuffled_geometry"):
                fit = _fit_one(
                    estimator=estimator,
                    observed=observed,
                    mean_rates=mean_first,
                    timepoints=inputs.timepoints,
                    mapping=mapping,
                    geometry=geometries[geometry_name],
                    geometry_name=geometry_name,
                    regularization=primary_regularization,
                    observation_mask=train_mask,
                    maxiter=args.maxiter,
                )
                constant = _fit_constant_variance(
                    estimator,
                    observed,
                    mean_first,
                    inputs.timepoints,
                    mapping,
                    geometries[geometry_name],
                    train_mask,
                )
                mapped = covariance_profiles(fit.mapped_covariance)
                mapped_rmse = 0.5 * (_log_rmse(mapped[0], true_marginal) + _log_rmse(mapped[1], true_conditional))
                const_profiles = covariance_profiles(map_hdx_covariance(constant["covariance"], mapping))
                constant_rmse = 0.5 * (
                    _log_rmse(const_profiles[0], true_marginal) + _log_rmse(const_profiles[1], true_conditional)
                )
                heldout_ratio = _heldout_row_mse_ratio(
                    estimator,
                    mean_first,
                    fit.variances,
                    mapping_all,
                    inputs.timepoints,
                    peptide1,
                    observed_all,
                )
                metrics = variance_recovery_metrics(fit.variances, true_variances)
                fit_records.append(
                    {
                        "coefficient": coefficient,
                        "ensemble": ensemble,
                        "estimator": estimator,
                        "geometry": geometry_name,
                        "regularization": primary_regularization,
                        "fit": fit,
                        "mean_rates": mean_first,
                        "mapping": mapping,
                        "peptide1_row": peptide1,
                        "observed_all": observed_all,
                        "heldout_mean_mse_ratio": heldout_ratio,
                        "log_variance_spearman": metrics["log_variance_spearman"],
                        "mapped_variance_log_rmse": float(mapped_rmse),
                        "constant_mapped_variance_log_rmse": float(constant_rmse),
                        "psd": bool(np.linalg.eigvalsh(fit.covariance).min() >= -1e-9),
                        "finite_objective": bool(np.isfinite(fit.objective)),
                        "objective": float(fit.objective),
                        "constant_fit": constant,
                    }
                )

        shuffled_mapped_rmse = {
            item["estimator"]: item["mapped_variance_log_rmse"]
            for item in fit_records
            if item["geometry"] == "shuffled_geometry"
        }

        for item in fit_records:
            estimator = str(item["estimator"])
            geometry_name = str(item["geometry"])
            fit = item["fit"]
            archived = _find_archived_candidate(
                artifact,
                ensemble=ensemble,
                estimator=estimator,
                geometry=geometry_name,
                regularization=item["regularization"],
                peptide_fold=0,
                time_fold=0,
            )
            beta_new = _beta_from_variance(item["mean_rates"], fit.variances)
            beta_old = _beta_from_variance(item["mean_rates"], archived["variances"])
            beta_delta = beta_new - beta_old
            ordering = spearmanr(fit.variances, archived["variances"]).statistic
            if not np.isfinite(ordering):
                ordering = 0.0
            gate_rows.append(
                {
                    "coefficient": coefficient,
                    "ensemble": ensemble,
                    "panel": "pivot",
                    "estimator": estimator,
                    "geometry": geometry_name,
                    "regularization": item["regularization"],
                    "heldout_mean_mse_ratio": item["heldout_mean_mse_ratio"],
                    "log_variance_spearman": item["log_variance_spearman"],
                    "mapped_variance_log_rmse": item["mapped_variance_log_rmse"],
                    "constant_mapped_variance_log_rmse": item["constant_mapped_variance_log_rmse"],
                    "beats_shuffled_geometry": geometry_name != "shuffled_geometry",
                    "psd": item["psd"],
                    "finite_objective": item["finite_objective"],
                    "objective": item["objective"],
                    "objective_vs_archived": float(item["objective"]) - float(archived["row"]["objective"]),
                    "mapped_rmse_vs_archived": float(
                        item["mapped_variance_log_rmse"]
                        - float(archived["nmr"].get("mapped_variance_log_rmse", float("nan")))
                    ),
                    "d_rmse_vs_archived": float(
                        np.sqrt(np.mean((fit.variances - archived["variances"]) ** 2))
                    ),
                    "d_ordering_spearman_vs_archived": float(ordering),
                    "beta_delta_mean": float(np.mean(beta_delta)),
                    "beta_delta_median": float(np.median(beta_delta)),
                    "beta_delta_max_abs": float(np.max(np.abs(beta_delta))),
                    "constant_vs_archived_objective": float(
                        item["constant_fit"]["objective"]
                        - float(archived["nmr"].get("constant_objective", archived["nmr"].get("objective", 0.0)))
                    ),
                    "archived_candidate_id": archived["candidate_id"],
                    "mapped_ranking": "primary",
                }
            )

        for item in gate_rows[-len(fit_records) :]:
            if item["geometry"] == "shuffled_geometry":
                continue
            shuffle_rmse = shuffled_mapped_rmse.get(str(item["estimator"]))
            if shuffle_rmse is None:
                continue
            item["beats_shuffled_geometry"] = bool(
                float(item["mapped_variance_log_rmse"]) < float(shuffle_rmse)
            )

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(args.output_dir / "pivot_refit_gate.csv", index=False)
    gate = qualification_gate(
        gate_df.to_dict("records"),
        required_panels=("pivot",),
        required_ensembles=tuple(common.ENSEMBLES.keys()),
    )

    gate_flags: list[str] = []
    for item in gate_rows:
        reasons: list[str] = []
        if not np.isfinite(item["mapped_rmse_vs_archived"]):
            reasons.append("mapped_rmse_comparison_missing")
        elif item["mapped_rmse_vs_archived"] > ARCHIVED_MAPPED_RMSE_TOL:
            reasons.append("mapped_rmse_worsened_vs_archived")
        if item["d_ordering_spearman_vs_archived"] < ARCHIVED_D_ORDERING_MIN:
            reasons.append("d_ordering_degraded_vs_archived")
        if not bool(item["beats_shuffled_geometry"]) and item["geometry"] != "shuffled_geometry":
            reasons.append("beats_shuffled_reference_failed")
        if not bool(item["finite_objective"]) or not bool(item["psd"]):
            reasons.append("numerics_failure")
        gate_flags.append(";".join(reasons) if reasons else "ok")

    gate_df["archived_comparison_flag"] = gate_flags
    gate_df.to_csv(args.output_dir / "pivot_refit_gate.csv", index=False)

    cell_decisions = []
    for item in gate_df.to_dict("records"):
        cell_decisions.append(
            {
                "coefficient": item["coefficient"],
                "ensemble": item["ensemble"],
                "estimator": item["estimator"],
                "geometry": item["geometry"],
                "mapped_rmse_vs_archived": float(item["mapped_rmse_vs_archived"]),
                "d_ordering_spearman_vs_archived": float(item["d_ordering_spearman_vs_archived"]),
                "beats_shuffled_geometry": bool(item["beats_shuffled_geometry"]),
                "decision": "ok" if item["archived_comparison_flag"] == "ok" else "investigation",
                "archived_comparison_flag": item["archived_comparison_flag"],
            }
        )

    decision = "investigation_wide" if (not gate["qualified"] or any(flag != "ok" for flag in gate_flags)) else "future_correctness"

    payload = {
        "decision": decision,
        "tiers_executed": ("tier1", "tier2"),
        "tier1": decision_tier1,
        "replay": replay_rows,
        "tier2": {
            "pivot_refit_gate_path": str(args.output_dir / "pivot_refit_gate.csv"),
            "qualification": gate,
            "archived_comparison_flags": gate_flags,
            "cell_decisions": cell_decisions,
        },
        "pivot_gap_path": str(args.output_dir / "pivot_gap.csv"),
    }
    (args.output_dir / "pivot_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--coefficient-lock",
        type=Path,
        default=DEFAULT_COEFFICIENT_LOCK,
        help="coefficient_lock.json with frozen settings",
    )
    parser.add_argument(
        "--coefficient-settings",
        default=",".join(DEFAULT_COEFFICIENT_SETTINGS),
        help="comma-separated names in coefficient_lock (default: scaled_published,constrained_optimum)",
    )
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--tier1-only", action="store_true", help="run only tier-1 gap audit and replay")
    parser.add_argument(
        "--force-tier2", action="store_true", help="run tier-2 refits even if tier-1 passes"
    )
    parser.add_argument("--overwrite", action="store_true", help="overwrite non-empty output directory")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
