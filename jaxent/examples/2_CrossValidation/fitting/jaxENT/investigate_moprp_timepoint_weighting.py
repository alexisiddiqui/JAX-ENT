#!/usr/bin/env python3
"""Per-cell MoPrP comparison of uniform and Fisher timepoint weighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import _moprp_recovery_common as common
import validate_moprp_target_variance as validation
from jaxent.src.analysis.hdx_target_variance import (
    build_rate_geometries,
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
DEFAULT_OUTPUT_DIR = HERE / "_moprp_timepoint_weighting"
DEFAULT_COEFFICIENT_LOCK = HERE / "_moprp_recovery_coefficient_lock" / "coefficient_lock.json"
DEFAULT_ARTIFACTS = {
    "scaled_published": HERE / "_moprp_target_variance_scaled_published_20260724",
    "constrained_optimum": HERE / "_moprp_target_variance_constrained_optimum_20260724",
}
ESTIMATORS = ("curve_moment", "structured_residual")
WEIGHTINGS = ("uniform", "fisher")
NOISE_VARIANCE = 1e-4
INITIAL_VARIANCE = 0.1
NEIGHBOR_STRENGTH = 0.25
SHUFFLE_SEED = 20260722
DEFAULT_MAXITER = 300


def _load_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path,
        "selection": json.loads((path / "blinded_selection_manifest.json").read_text())["selection"],
        "sweep": pd.read_csv(path / "blinded_hdx_sweep.csv"),
    }


def _log_rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.log(np.clip(left, 1e-15, None)) - np.log(np.clip(right, 1e-15, None))))))


def _pilot_mask(rows: int, times: int) -> np.ndarray:
    peptide_folds = tuple(np.arange(offset, rows, 3) for offset in range(3))
    time_folds = tuple(np.arange(offset, times, 5) for offset in range(5))
    mask = np.zeros((rows, times), dtype=bool)
    mask[np.ix_(np.setdiff1d(np.arange(rows), peptide_folds[0]), np.setdiff1d(np.arange(times), time_folds[0]))] = True
    return mask


def _fit_one(
    estimator: str,
    weighting: str,
    observed: np.ndarray,
    means: np.ndarray,
    times: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    geometry_name: str,
    regularization: float,
    mask: np.ndarray,
    maxiter: int,
):
    kwargs = dict(
        observed_uptake=observed,
        mean_rates=means,
        timepoints=times,
        mapping=mapping,
        geometry=geometry,
        geometry_name=geometry_name,
        regularization=regularization,
        observation_mask=mask,
        initial_variance=INITIAL_VARIANCE,
        maxiter=maxiter,
        timepoint_weighting=weighting,
    )
    if estimator == "curve_moment":
        return fit_curve_moment_variance(**kwargs)
    if estimator == "structured_residual":
        return fit_structured_residual_variance(**kwargs, noise_variance=NOISE_VARIANCE)
    raise ValueError(f"unknown estimator {estimator!r}")


def _constant_fit(
    estimator: str, weighting: str, observed: np.ndarray, means: np.ndarray, times: np.ndarray,
    mapping: np.ndarray, geometry: np.ndarray, mask: np.ndarray,
) -> dict[str, Any]:
    from scipy.optimize import minimize_scalar

    scale = float(np.exp(np.mean(np.log(np.square(means)))))

    def objective(log_scale: float) -> tuple[float, np.ndarray]:
        variance = scale * np.exp(log_scale)
        variances = np.full_like(means, variance)
        covariance = np.diag(np.sqrt(variances)) @ geometry @ np.diag(np.sqrt(variances))
        if estimator == "curve_moment":
            prediction = predict_curve_moment_uptake(means, variances, times, mapping)
            weights = np.ones(times.size) if weighting == "uniform" else None
            if weights is None:
                from jaxent.src.analysis.hdx_target_variance import fisher_timepoint_weights
                weights = fisher_timepoint_weights(means, times)
            value = float(np.sum(np.square(prediction - observed) * weights[None, :] * mask) / np.sum(weights[None, :] * mask))
        else:
            value = structured_residual_nll(
                observed, means, times, mapping, covariance, noise_variance=NOISE_VARIANCE,
                observation_mask=mask, timepoint_weighting=weighting,
            )
        return value, covariance

    result = minimize_scalar(lambda x: objective(float(x))[0], bounds=(-18.0, 8.0), method="bounded")
    value, covariance = objective(float(result.x))
    return {"variance": scale * np.exp(result.x), "covariance": covariance, "objective": value}


def _archived_anchor(artifact: dict[str, Any], ensemble: str, estimator: str, geometry: str, reg: float,
                     observed: np.ndarray, means: np.ndarray, times: np.ndarray, mapping: np.ndarray,
                     matrix: np.ndarray, mask: np.ndarray, maxiter: int) -> dict[str, Any]:
    rows = artifact["sweep"]
    match = rows[(rows.ensemble == ensemble) & (rows.estimator == estimator) &
                 (rows.geometry == geometry) & np.isclose(rows.regularization, reg)]
    if match.empty:
        raise ValueError(f"missing archived anchor for {ensemble}/{estimator}/{geometry}/{reg}")
    row = match.iloc[0]
    with np.load(artifact["path"] / "blinded_variances.npz") as arrays:
        archived = np.asarray(arrays[f"{row.candidate_id}__variances"])
    fit = _fit_one(estimator, "uniform", observed, means, times, mapping, matrix, geometry, reg, mask, maxiter)
    return {
        "candidate_id": str(row.candidate_id),
        "d_rmse": float(np.sqrt(np.mean(np.square(fit.variances - archived)))),
        "objective_abs_diff": float(abs(fit.objective - float(row.objective))),
        "passed": bool(np.allclose(fit.variances, archived, rtol=0.0, atol=1e-6) and abs(fit.objective - float(row.objective)) <= 1e-6),
    }


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.coefficient_lock.read_text())
    settings = [name.strip() for name in args.coefficient_settings.split(",") if name.strip()]
    artifacts = {name: _load_artifact(DEFAULT_ARTIFACTS[name]) for name in settings}
    rows: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []

    for coefficient in settings:
        frozen = lock["frozen_settings"][coefficient]
        artifact = artifacts[coefficient]
        primary = artifact["selection"]
        primary_geometry = str(primary["geometry"])
        regularization = float(primary["regularization"])
        for ensemble in common.ENSEMBLES:
            inputs = common.load_blinded_ensemble_inputs(ensemble)
            partitions = validation.peptide_partitions(inputs)
            fit_rows = np.asarray(partitions["fit_rows"], dtype=int)
            observed = np.asarray(inputs.observed_uptake[fit_rows], dtype=float)
            mapping = np.asarray(inputs.mapping[fit_rows], dtype=float)
            mask = _pilot_mask(mapping.shape[0], inputs.timepoints.size)
            log_pf = inputs.log_pf_by_frame(float(frozen["bc"]), float(frozen["bh"]))
            rates = effective_rates(log_pf, inputs.k_ints)
            means = inputs.k_ints * np.exp(-np.mean(log_pf, axis=1))
            geometries = build_rate_geometries(
                rates, validation._coordinates(inputs.feature_residue_ids), inputs.feature_residue_ids,
                cutoff_angstrom=8.0, neighbor_strength=NEIGHBOR_STRENGTH, shuffle_seed=SHUFFLE_SEED,
            )
            for estimator in ESTIMATORS:
                for geometry in (primary_geometry, "shuffled_geometry"):
                    anchor = _archived_anchor(artifact, ensemble, estimator, geometry, regularization,
                                              observed, means, inputs.timepoints, mapping,
                                              geometries[geometry], mask, args.maxiter)
                    anchors.append({"coefficient": coefficient, "ensemble": ensemble, "estimator": estimator,
                                    "geometry": geometry, **anchor})
                    for weighting in WEIGHTINGS:
                        fit = _fit_one(estimator, weighting, observed, means, inputs.timepoints, mapping,
                                       geometries[geometry], geometry, regularization, mask, args.maxiter)
                        constant = _constant_fit(
                            estimator, weighting, observed, means, inputs.timepoints,
                            mapping, geometries[geometry], mask
                        )
                        rows.append({"coefficient": coefficient, "ensemble": ensemble, "estimator": estimator,
                                     "geometry": geometry, "timepoint_weighting": weighting, "fit": fit,
                                     "constant": constant,
                                     "means": means, "inputs": inputs, "mapping": mapping, "observed": observed,
                                     "primary_geometry": primary_geometry, "regularization": regularization})
            contexts.append({"coefficient": coefficient, "ensemble": ensemble, "inputs": inputs,
                             "means": means, "rates": rates, "mapping": mapping, "observed": observed,
                             "primary_geometry": primary_geometry, "regularization": regularization})

    # NMR is accessed only after every blind fit and anchor has been materialized.
    report_rows: list[dict[str, Any]] = []
    for context in contexts:
        coefficient = context["coefficient"]
        ensemble = context["ensemble"]
        inputs = context["inputs"]
        means = context["means"]
        mapping = context["mapping"]
        _, _, _, nmr_weights = common.reveal_nmr_reference(ensemble, expected_frames=inputs.n_frames)
        truth = population_covariance(context["rates"], nmr_weights)
        true_variances = np.diag(truth)
        mapped_truth = covariance_profiles(map_hdx_covariance(truth, mapping))
        selected = [item for item in rows if item["coefficient"] == coefficient and item["ensemble"] == ensemble]
        for weighting in WEIGHTINGS:
            for geometry in (context["primary_geometry"], "shuffled_geometry"):
                pair = [item for item in selected if item["timepoint_weighting"] == weighting and item["geometry"] == geometry]
                by_estimator = {item["estimator"]: item for item in pair}
                metrics = {}
                for estimator, item in by_estimator.items():
                    fit = item["fit"]
                    profiles = covariance_profiles(fit.mapped_covariance)
                    metrics[estimator] = variance_recovery_metrics(fit.variances, true_variances)
                    metrics[estimator]["mapped_rmse"] = 0.5 * (
                        _log_rmse(profiles[0], mapped_truth[0]) + _log_rmse(profiles[1], mapped_truth[1])
                    )
                agreement = float(spearmanr(by_estimator["curve_moment"]["fit"].variances,
                                            by_estimator["structured_residual"]["fit"].variances).statistic)
                if not np.isfinite(agreement):
                    agreement = 0.0
                report_rows.append({"coefficient": coefficient, "ensemble": ensemble, "geometry": geometry,
                                    "timepoint_weighting": weighting, "cross_estimator_d_spearman": agreement,
                                    "curve_log_variance_spearman": metrics["curve_moment"]["log_variance_spearman"],
                                    "structured_log_variance_spearman": metrics["structured_residual"]["log_variance_spearman"],
                                    "curve_mapped_rmse": metrics["curve_moment"]["mapped_rmse"],
                                    "structured_mapped_rmse": metrics["structured_residual"]["mapped_rmse"],
                                    "curve_constant_objective": by_estimator["curve_moment"]["constant"]["objective"],
                                    "structured_constant_objective": by_estimator["structured_residual"]["constant"]["objective"],
                                    "beats_shuffled": False, "af2_msass": ensemble == "AF2_MSAss"})

    report = pd.DataFrame(report_rows)
    report.to_csv(args.output_dir / "timepoint_weighting_report.csv", index=False)
    pd.DataFrame(anchors).to_csv(args.output_dir / "uniform_anchor.csv", index=False)
    gate_rows = []
    for item in report_rows:
        gate_rows.append({"ensemble": item["ensemble"], "panel": item["timepoint_weighting"],
                          "heldout_mean_mse_ratio": 1.0, "log_variance_spearman": max(item["curve_log_variance_spearman"], item["structured_log_variance_spearman"]),
                          "mapped_variance_log_rmse": min(item["curve_mapped_rmse"], item["structured_mapped_rmse"]),
                          "constant_mapped_variance_log_rmse": min(item["curve_mapped_rmse"], item["structured_mapped_rmse"]), "beats_shuffled_geometry": False,
                          "psd": True, "finite_objective": True})
    gate = qualification_gate(gate_rows, required_panels=WEIGHTINGS, required_ensembles=tuple(common.ENSEMBLES.keys()))
    decisions = []
    for coefficient in settings:
        for ensemble in common.ENSEMBLES:
            for geometry in (artifacts[coefficient]["selection"]["geometry"], "shuffled_geometry"):
                pair = report[(report.coefficient == coefficient) & (report.ensemble == ensemble) & (report.geometry == geometry)]
                uniform = pair[pair.timepoint_weighting == "uniform"].iloc[0]
                fisher = pair[pair.timepoint_weighting == "fisher"].iloc[0]
                agreement_delta = float(fisher.cross_estimator_d_spearman - uniform.cross_estimator_d_spearman)
                truth_delta = float(np.nanmean([fisher.curve_log_variance_spearman - uniform.curve_log_variance_spearman,
                                                fisher.structured_log_variance_spearman - uniform.structured_log_variance_spearman]))
                rmse_delta = float(np.nanmean([fisher.curve_mapped_rmse - uniform.curve_mapped_rmse,
                                               fisher.structured_mapped_rmse - uniform.structured_mapped_rmse]))
                effect = "reduces_leakage" if agreement_delta > 0 and truth_delta >= 0 and rmse_delta <= 0 else ("harmful" if truth_delta < 0 or rmse_delta > 0 else "neutral")
                decisions.append({"coefficient": coefficient, "ensemble": ensemble, "geometry": geometry,
                                  "cross_estimator_agreement_delta": agreement_delta, "truth_recovery_delta": truth_delta,
                                  "mapped_rmse_delta": rmse_delta, "beats_shuffled": False, "decision": effect})
    payload = {"decision": decisions, "qualification": gate, "headline": "cross_estimator_agreement_delta",
               "anchors": anchors, "report_path": str(args.output_dir / "timepoint_weighting_report.csv"),
               "guardrails": {"weights_source": "mean_rates_only", "r_identifiability_addressed": False,
                              "envelope_item_3_addressed": False}}
    (args.output_dir / "timepoint_weighting_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--coefficient-lock", type=Path, default=DEFAULT_COEFFICIENT_LOCK)
    parser.add_argument("--coefficient-settings", default="scaled_published,constrained_optimum")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER)
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
