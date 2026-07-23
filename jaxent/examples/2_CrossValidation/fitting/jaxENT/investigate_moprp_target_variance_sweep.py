#!/usr/bin/env python3
"""Non-promotable MoPrP test bed for HDX target-variance inference.

This runner deliberately does *not* implement the formal frozen-TeaA validation.
It sweeps the registered estimators, geometries, and regularisation strengths on
MoPrP using held-out HDX reconstruction only.  Every inferred variance vector and
the HDX-only selection manifest are written before NMR populations are read.

The subsequent NMR reveal is diagnostic.  Outputs from this script can never
authorise ensemble-weight or BV-coefficient optimisation, even when their post-hoc
pseudo-truth metrics look favourable.  The formal validator remains
``validate_moprp_target_variance.py`` and accepts only a qualified TeaA artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import _moprp_recovery_common as common
import validate_moprp_target_variance as validation
import jaxent.src.analysis.hdx_target_variance as target_variance
from jaxent.src.analysis.hdx_target_variance import (
    GEOMETRY_NAMES,
    build_hdx_covariance,
    build_rate_geometries,
    covariance_profiles,
    effective_rates,
    fit_curve_moment_variance,
    fit_structured_residual_variance,
    map_hdx_covariance,
    population_covariance,
    predict_curve_moment_uptake,
    predict_fixed_mean_uptake,
    structured_residual_nll,
    variance_recovery_metrics,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = HERE / "_moprp_target_variance_diagnostic_sweep"
DEFAULT_COEFFICIENT_LOCK = HERE / "_moprp_recovery_coefficient_lock" / "coefficient_lock.json"
ESTIMATORS = ("curve_moment", "structured_residual")
DEFAULT_REGULARIZATIONS = (0.0, 0.01, 0.1, 1.0)
ARTIFACT_TYPE = "moprp_target_variance_diagnostic_sweep"
PEPTIDE_FOLDS = 3
TIME_FOLDS = 5
NOISE_VARIANCE = 1e-4
NEIGHBOR_STRENGTH = 0.25
SHUFFLE_SEED = 20260722
INITIAL_VARIANCE = 0.1


def load_bv_setting(path: Path, setting: str) -> dict[str, Any]:
    """Resolve one frozen BV mean setting without loading NMR populations."""

    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text())
        values = payload["frozen_settings"][setting]
        bc = float(values["bc"])
        bh = float(values["bh"])
    except FileNotFoundError:
        raise
    except KeyError as error:
        if error.args and error.args[0] == setting:
            available = sorted(payload.get("frozen_settings", {}))
            raise ValueError(
                f"unknown BV setting {setting!r}; available settings are {available}"
            ) from error
        raise ValueError(f"malformed coefficient lock {path}: missing {error.args[0]!r}") from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed coefficient lock {path}: {error}") from error
    if not np.isfinite([bc, bh]).all() or bc < 0.0 or bh < 0.0:
        raise ValueError(f"BV coefficients must be finite and non-negative, got {(bc, bh)}")
    return {
        "setting": setting,
        "bc": bc,
        "bh": bh,
        "coefficient_lock_path": str(path),
        "coefficient_lock_sha256": common.sha256(path),
        "provenance": (
            "frozen upstream coefficient setting; the coefficient lock was calibrated at "
            "fixed NMR populations, but no NMR populations are loaded during HDX inference"
        ),
    }


def _log_rmse(left: np.ndarray, right: np.ndarray, floor: float = 1e-15) -> float:
    residual = np.log(np.clip(left, floor, None)) - np.log(np.clip(right, floor, None))
    return float(np.sqrt(np.mean(np.square(residual))))


def _folds(size: int, count: int, scheme: str = "interleaved") -> tuple[np.ndarray, ...]:
    if size < count:
        raise ValueError("cannot create more non-empty folds than observations")
    if scheme == "interleaved":
        return tuple(np.arange(offset, size, count, dtype=int) for offset in range(count))
    if scheme == "contiguous":
        return tuple(np.asarray(fold, dtype=int) for fold in np.array_split(np.arange(size), count))
    raise ValueError("fold scheme must be 'interleaved' or 'contiguous'")


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _code_revision() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=common.PACKAGE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=common.PACKAGE_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": revision, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unavailable", "working_tree_dirty": None}


def _fit_one(
    estimator: str,
    observed: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    geometry_name: str,
    regularization: float,
    train_mask: np.ndarray,
    *,
    maxiter: int,
):
    arguments = dict(
        observed_uptake=observed,
        mean_rates=mean_rates,
        timepoints=timepoints,
        mapping=mapping,
        geometry=geometry,
        geometry_name=geometry_name,
        regularization=regularization,
        observation_mask=train_mask,
        initial_variance=INITIAL_VARIANCE,
        maxiter=maxiter,
    )
    if estimator == "curve_moment":
        return fit_curve_moment_variance(**arguments)
    if estimator == "structured_residual":
        return fit_structured_residual_variance(
            **arguments, noise_variance=NOISE_VARIANCE
        )
    raise ValueError(f"unknown estimator {estimator!r}")


def _fit_constant_variance(
    estimator: str,
    observed: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    train_mask: np.ndarray,
) -> dict[str, Any]:
    """Fit one absolute scalar ``D=dI`` from the same training HDX observations.

    The selected geometry is retained, so this isolates heterogeneous amplitudes
    from a scalar-amplitude baseline without changing correlation structure.
    """

    scale = float(np.exp(np.mean(np.log(np.square(mean_rates)))))

    def components(log_relative_variance: float) -> tuple[float, np.ndarray, np.ndarray]:
        variance = scale * float(np.exp(log_relative_variance))
        variances = np.full_like(mean_rates, variance)
        covariance = build_hdx_covariance(variances, geometry)
        if estimator == "curve_moment":
            predicted = predict_curve_moment_uptake(
                mean_rates, variances, timepoints, mapping
            )
            objective = float(np.mean(np.square(predicted - observed)[train_mask]))
        elif estimator == "structured_residual":
            predicted = predict_fixed_mean_uptake(mean_rates, timepoints, mapping)
            objective = structured_residual_nll(
                observed,
                mean_rates,
                timepoints,
                mapping,
                covariance,
                noise_variance=NOISE_VARIANCE,
                observation_mask=train_mask,
            )
        else:
            raise ValueError(f"unknown estimator {estimator!r}")
        return objective, covariance, predicted

    result = minimize_scalar(
        lambda value: components(float(value))[0],
        bounds=(-18.0, 8.0),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 300},
    )
    objective, covariance, predicted = components(float(result.x))
    return {
        "variance": scale * float(np.exp(result.x)),
        "covariance": covariance,
        "predicted_uptake": predicted,
        "objective": objective,
        "success": bool(result.success and np.isfinite(objective)),
    }


def select_by_hdx_only(frame: pd.DataFrame) -> dict[str, Any]:
    """Select one shared physical candidate across both ensembles without truth columns."""

    required = {
        "ensemble",
        "peptide_fold",
        "time_fold",
        "estimator",
        "geometry",
        "regularization",
        "heldout_reconstruction_score",
        "heldout_mean_mse_ratio",
        "success",
        "finite_objective",
        "psd",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"HDX-only selection requires columns {sorted(required)}")
    work = frame[list(required)].copy()
    candidates = ["estimator", "geometry", "regularization"]
    cells = ["ensemble", "peptide_fold", "time_fold"]
    if work[list(required)].isna().any().any():
        raise ValueError("HDX-only selection inputs cannot contain missing values")
    numeric = ["regularization", "heldout_reconstruction_score", "heldout_mean_mse_ratio"]
    if not np.isfinite(work[numeric].to_numpy(float)).all():
        raise ValueError("HDX-only selection scores and regularisation must be finite")
    if work.duplicated(candidates + cells).any():
        raise ValueError("HDX-only selection contains duplicate candidate/cell rows")
    expected_cells = int(work[cells].drop_duplicates().shape[0])
    coverage = work.groupby(candidates, as_index=False).size()
    if not bool((coverage["size"] == expected_cells).all()):
        raise ValueError("every HDX candidate must cover every ensemble/fold cell")
    eligible = (
        work.groupby(candidates, as_index=False)[["success", "finite_objective", "psd"]]
        .all()
        .assign(eligible=lambda values: values[["success", "finite_objective", "psd"]].all(axis=1))
    )
    work = work.merge(eligible[candidates + ["eligible"]], on=candidates)
    work = work[work.eligible]
    if work.empty:
        raise ValueError("no candidate passed convergence, finite-objective, and PSD safeguards")
    work["reconstruction_rank"] = work.groupby(cells).heldout_reconstruction_score.rank(
        method="average", pct=True
    )
    summary = (
        work.groupby(candidates, as_index=False)
        .agg(
            median_reconstruction_rank=("reconstruction_rank", "median"),
            median_heldout_mean_mse_ratio=("heldout_mean_mse_ratio", "median"),
        )
        .sort_values(
            [
                "median_reconstruction_rank",
                "median_heldout_mean_mse_ratio",
                "estimator",
                "geometry",
                "regularization",
            ],
            kind="stable",
        )
    )
    physical = summary[~summary.geometry.isin(["identity", "shuffled_geometry"])]
    if physical.empty:
        raise ValueError("no eligible physical geometry remains")
    selected = physical.iloc[0]

    def best_control(name: str) -> pd.Series | None:
        rows = summary[summary.geometry == name]
        return None if rows.empty else rows.iloc[0]

    identity = best_control("identity")
    shuffled = best_control("shuffled_geometry")
    selected_rank = float(selected.median_reconstruction_rank)
    return {
        "estimator": str(selected.estimator),
        "geometry": str(selected.geometry),
        "regularization": float(selected.regularization),
        "selection_criterion": "median within-cell held-out MoPrP HDX predictive-NLL rank",
        "shared_across_ensembles": True,
        "median_reconstruction_rank": selected_rank,
        "median_heldout_mean_mse_ratio": float(selected.median_heldout_mean_mse_ratio),
        "identity_control_rank": (
            float(identity.median_reconstruction_rank) if identity is not None else float("nan")
        ),
        "shuffled_control_rank": (
            float(shuffled.median_reconstruction_rank) if shuffled is not None else float("nan")
        ),
        "identity_control_beats_selected": bool(
            identity is not None and float(identity.median_reconstruction_rank) < selected_rank
        ),
        "shuffled_control_beats_selected": bool(
            shuffled is not None and float(shuffled.median_reconstruction_rank) < selected_rank
        ),
        "ranking": summary.to_dict("records"),
    }


def _peptide1_diagnostic(
    inputs: common.BlindedEnsembleInputs,
    mean_rates: np.ndarray,
    rates_by_frame: np.ndarray,
    selected_fits: list[Any],
    estimator: str,
) -> dict[str, Any]:
    peptide1 = int(validation.peptide_partitions(inputs)["peptide1_row"])
    peptide1_map = inputs.mapping[[peptide1]]
    observed = inputs.observed_uptake[peptide1]
    frame_uptake = 1.0 - np.exp(
        -inputs.timepoints[:, None, None] * rates_by_frame[None, :, :]
    )
    by_frame = np.einsum("r,trf->tf", inputs.mapping[peptide1], frame_uptake)
    prediction_mse = []
    for fit in selected_fits:
        if estimator == "curve_moment":
            predicted = predict_curve_moment_uptake(
                mean_rates, fit.variances, inputs.timepoints, peptide1_map
            )[0]
        else:
            predicted = predict_fixed_mean_uptake(mean_rates, inputs.timepoints, peptide1_map)[0]
        prediction_mse.append(float(np.mean(np.square(predicted - observed))))
    return {
        "peptide_id": int(inputs.peptide_ids[peptide1]),
        "included_in_any_fit": False,
        "used_for_selection": False,
        "median_prediction_mse": float(np.median(prediction_mse)),
        "fraction_inside_uniform_frame_envelope": float(
            np.mean((observed >= by_frame.min(axis=1)) & (observed <= by_frame.max(axis=1)))
        ),
    }


def _score_after_reveal(
    raw: pd.DataFrame,
    fit_records: list[dict[str, Any]],
    ensemble_context: dict[str, dict[str, Any]],
    selection: dict[str, Any],
    fixed_bv_mean: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read NMR populations only after all blinded artifacts have been written."""

    score_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for ensemble, context in ensemble_context.items():
        inputs = context["inputs"]
        _, _, _, weights = common.reveal_nmr_reference(
            ensemble, expected_frames=inputs.n_frames
        )
        truth = population_covariance(context["rates_by_frame"], weights)
        true_variances = np.diag(truth)
        mapped_truth = map_hdx_covariance(truth, context["mapping"])
        true_marginal, true_conditional = covariance_profiles(mapped_truth)
        selected_fits = []
        for record in (item for item in fit_records if item["ensemble"] == ensemble):
            fit = record["fit"]
            if not bool(record["constant_fit"]["success"]):
                raise RuntimeError(
                    f"scalar-D control failed before NMR scoring: {record['candidate_id']}"
                )
            inferred_marginal, inferred_conditional = covariance_profiles(fit.mapped_covariance)
            mapped_rmse = 0.5 * (
                _log_rmse(inferred_marginal, true_marginal)
                + _log_rmse(inferred_conditional, true_conditional)
            )
            constant_marginal, constant_conditional = covariance_profiles(
                map_hdx_covariance(record["constant_fit"]["covariance"], context["mapping"])
            )
            constant_rmse = 0.5 * (
                _log_rmse(constant_marginal, true_marginal)
                + _log_rmse(constant_conditional, true_conditional)
            )
            score_rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    **variance_recovery_metrics(fit.variances, true_variances),
                    "mapped_variance_log_rmse": mapped_rmse,
                    "constant_mapped_variance_log_rmse": constant_rmse,
                    "beats_constant_variance": bool(mapped_rmse < constant_rmse),
                }
            )
            if (
                record["estimator"] == selection["estimator"]
                and record["geometry"] == selection["geometry"]
                and np.isclose(record["regularization"], selection["regularization"])
            ):
                selected_fits.append(fit)
        diagnostics[ensemble] = _peptide1_diagnostic(
            inputs,
            context["mean_rates"],
            context["rates_by_frame"],
            selected_fits,
            str(selection["estimator"]),
        )

    scored = raw.merge(pd.DataFrame(score_rows), on="candidate_id", validate="one_to_one")
    keys = ["ensemble", "peptide_fold", "time_fold", "estimator", "regularization"]
    shuffled = scored[scored.geometry == "shuffled_geometry"][
        keys + ["mapped_variance_log_rmse"]
    ].rename(columns={"mapped_variance_log_rmse": "shuffled_mapped_variance_log_rmse"})
    scored = scored.merge(shuffled, on=keys, how="left", validate="many_to_one")
    scored["beats_shuffled_geometry"] = (
        scored.mapped_variance_log_rmse < scored.shuffled_mapped_variance_log_rmse
    ) & (scored.geometry != "shuffled_geometry")

    selected = scored[
        (scored.estimator == selection["estimator"])
        & (scored.geometry == selection["geometry"])
        & np.isclose(scored.regularization, selection["regularization"])
    ]
    reports = []
    for ensemble, rows in selected.groupby("ensemble", sort=True):
        reports.append(
            {
                "ensemble": ensemble,
                "n_fold_fits": int(len(rows)),
                "median_log_variance_spearman": float(rows.log_variance_spearman.median()),
                "median_mapped_variance_log_rmse": float(rows.mapped_variance_log_rmse.median()),
                "median_constant_mapped_variance_log_rmse": float(
                    rows.constant_mapped_variance_log_rmse.median()
                ),
                "median_shuffled_mapped_variance_log_rmse": float(
                    rows.shuffled_mapped_variance_log_rmse.median()
                ),
                "beats_constant_in_every_fold": bool(rows.beats_constant_variance.all()),
                "beats_shuffled_in_every_fold": bool(rows.beats_shuffled_geometry.all()),
                "positive_truth_agreement": bool(rows.log_variance_spearman.median() > 0),
                "peptide1_independent_diagnostic": diagnostics[ensemble],
            }
        )
    diagnostic_pass = bool(
        len(reports) == len(common.ENSEMBLES)
        and all(
            report["beats_constant_in_every_fold"]
            and report["beats_shuffled_in_every_fold"]
            and report["positive_truth_agreement"]
            for report in reports
        )
    )
    decision = {
        "artifact_type": ARTIFACT_TYPE,
        "status": "diagnostic_complete_non_promotable",
        "confirmatory_blind_status": "exhausted_by_explicit_all-candidate_nmr_reveal",
        "diagnostic_variance_gate_passes": diagnostic_pass,
        "formal_frozen_teaa_validation_performed": False,
        "qualified": False,
        "can_launch_moprp_validation": False,
        "weight_bv_optimization_authorized": False,
        "blocked_by": "TeaA estimator/geometry/regularisation have not passed the frozen qualification gate",
        "nmr_role": "post-inference pseudo-ground-truth scoring only",
        "fixed_bv_mean": fixed_bv_mean,
        "selection": selection,
        "ensembles": reports,
    }
    return scored, decision


def run(args: argparse.Namespace) -> None:
    fixed_bv_mean = load_bv_setting(args.coefficient_lock, args.bv_setting)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regularizations = tuple(float(value) for value in args.regularizations.split(","))
    peptide_fold_indices = (0,) if args.pilot else tuple(range(PEPTIDE_FOLDS))
    time_fold_indices = (0,) if args.pilot else tuple(range(TIME_FOLDS))
    maxiter = args.pilot_maxiter if args.pilot else args.maxiter

    rows: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    ensemble_context: dict[str, dict[str, Any]] = {}
    assembled_input_hashes: dict[str, str] = {}
    fold_manifest: dict[str, list[dict[str, Any]]] = {}
    constant_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    candidate_index = 0
    for ensemble in common.ENSEMBLES:
        inputs = common.load_blinded_ensemble_inputs(ensemble)
        partitions = validation.peptide_partitions(inputs)
        fit_rows = np.asarray(partitions["fit_rows"], dtype=int)
        mapping = inputs.mapping[fit_rows]
        observed = inputs.observed_uptake[fit_rows]
        rates = effective_rates(
            inputs.log_pf_by_frame(fixed_bv_mean["bc"], fixed_bv_mean["bh"]), inputs.k_ints
        )
        mean_rates = np.mean(rates, axis=1)
        coordinates = validation._coordinates(inputs.feature_residue_ids)
        geometries = build_rate_geometries(
            rates,
            coordinates,
            inputs.feature_residue_ids,
            cutoff_angstrom=8.0,
            neighbor_strength=NEIGHBOR_STRENGTH,
            shuffle_seed=SHUFFLE_SEED,
        )
        baseline = predict_fixed_mean_uptake(mean_rates, inputs.timepoints, mapping)
        peptide_folds = _folds(mapping.shape[0], PEPTIDE_FOLDS, args.fold_scheme)
        time_folds = _folds(inputs.timepoints.size, TIME_FOLDS, args.fold_scheme)
        assembled_input_hashes.update(
            {
                f"{ensemble}__observed_uptake": _array_sha256(inputs.observed_uptake),
                f"{ensemble}__canonical_mapping": _array_sha256(inputs.mapping),
                f"{ensemble}__timepoints": _array_sha256(inputs.timepoints),
                f"{ensemble}__feature_residue_ids": _array_sha256(inputs.feature_residue_ids),
                f"{ensemble}__reference_coordinates": _array_sha256(coordinates),
                f"{ensemble}__rates_by_frame": _array_sha256(rates),
                f"{ensemble}__mean_rates": _array_sha256(mean_rates),
            }
        )
        fold_manifest[ensemble] = []
        ensemble_context[ensemble] = {
            "inputs": inputs,
            "mapping": mapping,
            "rates_by_frame": rates,
            "mean_rates": mean_rates,
        }
        arrays[f"{ensemble}__mean_rates"] = mean_rates
        arrays[f"{ensemble}__fit_rows"] = fit_rows
        for peptide_fold in peptide_fold_indices:
            val_peptides = peptide_folds[peptide_fold]
            train_peptides = np.setdiff1d(np.arange(mapping.shape[0]), val_peptides)
            for time_fold in time_fold_indices:
                val_times = time_folds[time_fold]
                train_times = np.setdiff1d(np.arange(inputs.timepoints.size), val_times)
                train_mask = np.zeros_like(observed, dtype=bool)
                train_mask[np.ix_(train_peptides, train_times)] = True
                validation_mask = np.zeros_like(observed, dtype=bool)
                validation_mask[np.ix_(val_peptides, val_times)] = True
                train_active = np.any(mapping[train_peptides] > 0, axis=0)
                validation_active = np.any(mapping[val_peptides] > 0, axis=0)
                unseen_validation = validation_active & ~train_active
                fold_manifest[ensemble].append(
                    {
                        "peptide_fold": int(peptide_fold),
                        "time_fold": int(time_fold),
                        "train_peptide_ids": inputs.peptide_ids[fit_rows[train_peptides]].tolist(),
                        "validation_peptide_ids": inputs.peptide_ids[fit_rows[val_peptides]].tolist(),
                        "train_timepoints_min": inputs.timepoints[train_times].tolist(),
                        "validation_timepoints_min": inputs.timepoints[val_times].tolist(),
                        "train_observations": int(train_mask.sum()),
                        "validation_observations": int(validation_mask.sum()),
                        "validation_active_residues": int(validation_active.sum()),
                        "validation_residues_unseen_in_training_peptides": int(
                            unseen_validation.sum()
                        ),
                    }
                )
                baseline_mse = float(
                    np.mean(np.square(baseline - observed)[validation_mask])
                )
                for estimator in ESTIMATORS:
                    for geometry_name in GEOMETRY_NAMES:
                        for regularization in regularizations:
                            constant_key = (
                                ensemble,
                                peptide_fold,
                                time_fold,
                                estimator,
                                geometry_name,
                            )
                            if constant_key not in constant_cache:
                                constant_cache[constant_key] = _fit_constant_variance(
                                    estimator,
                                    observed,
                                    mean_rates,
                                    inputs.timepoints,
                                    mapping,
                                    geometries[geometry_name],
                                    train_mask,
                                )
                            constant_fit = constant_cache[constant_key]
                            fit = _fit_one(
                                estimator,
                                observed,
                                mean_rates,
                                inputs.timepoints,
                                mapping,
                                geometries[geometry_name],
                                geometry_name,
                                regularization,
                                train_mask,
                                maxiter=maxiter,
                            )
                            candidate_id = f"candidate_{candidate_index:06d}"
                            candidate_index += 1
                            heldout_mse = float(
                                np.mean(
                                    np.square(fit.predicted_uptake - observed)[validation_mask]
                                )
                            )
                            score = structured_residual_nll(
                                observed,
                                mean_rates,
                                inputs.timepoints,
                                mapping,
                                fit.covariance,
                                observation_mask=validation_mask,
                                predicted_uptake=fit.predicted_uptake,
                            )
                            row = {
                                "candidate_id": candidate_id,
                                "ensemble": ensemble,
                                "peptide_fold": peptide_fold,
                                "time_fold": time_fold,
                                "estimator": estimator,
                                "geometry": geometry_name,
                                "regularization": regularization,
                                "heldout_reconstruction_score": score,
                                "heldout_mean_mse": heldout_mse,
                                "heldout_mean_mse_ratio": heldout_mse / max(baseline_mse, 1e-15),
                                "success": fit.success,
                                "finite_objective": bool(np.isfinite(fit.objective)),
                                "psd": bool(np.linalg.eigvalsh(fit.covariance).min() >= -1e-9),
                                "iterations": fit.iterations,
                                "objective": fit.objective,
                                "constant_variance": constant_fit["variance"],
                                "constant_objective": constant_fit["objective"],
                                "constant_success": constant_fit["success"],
                                "validation_active_residues": int(validation_active.sum()),
                                "validation_residues_unseen_in_training_peptides": int(
                                    unseen_validation.sum()
                                ),
                            }
                            rows.append(row)
                            arrays[f"{candidate_id}__variances"] = fit.variances
                            fit_records.append(
                                {
                                    **{
                                        key: row[key]
                                        for key in (
                                            "candidate_id",
                                            "ensemble",
                                            "peptide_fold",
                                            "time_fold",
                                            "estimator",
                                            "geometry",
                                            "regularization",
                                        )
                                    },
                                    "fit": fit,
                                    "geometry_matrix": geometries[geometry_name],
                                    "constant_fit": constant_fit,
                                }
                            )

    raw = pd.DataFrame(rows)
    selection = select_by_hdx_only(raw)
    raw.to_csv(args.output_dir / "blinded_hdx_sweep.csv", index=False)
    np.savez_compressed(args.output_dir / "blinded_variances.npz", **arrays)
    blinded_manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "status": "blinded_hdx_sweep_complete_non_promotable",
        "pilot": bool(args.pilot),
        "qualified": False,
        "can_launch_moprp_validation": False,
        "weight_bv_optimization_authorized": False,
        "confirmatory_blind_status": "preserved_only_until_an_explicit_nmr_reveal",
        "selection_source": "held-out MoPrP HDX reconstruction only",
        "nmr_used_for_inference": False,
        "nmr_used_for_selection": False,
        "ensemble_reweighting_performed": False,
        "bv_coefficients_optimized": False,
        "fixed_bv_mean": fixed_bv_mean,
        "published_reference_bv_coefficients": {
            "bc": common.PUBLISHED_BC,
            "bh": common.PUBLISHED_BH,
        },
        "distance_cutoff_angstrom": 8.0,
        "sequence_neighbor_strength": NEIGHBOR_STRENGTH,
        "shuffle_seed": SHUFFLE_SEED,
        "noise_variance": NOISE_VARIANCE,
        "initial_relative_variance": INITIAL_VARIANCE,
        "constant_control": "training-HDX-fitted scalar absolute D retaining candidate R",
        "fold_scheme": args.fold_scheme,
        "folds": fold_manifest,
        "heldout_peptide_id": 1,
        "excluded_unmapped_residue": 101,
        "input_hashes": {
            **common.blinded_input_hashes(),
            "distance_reference_structure": common.sha256(validation.STRUCTURE),
            "runner_source": common.sha256(Path(__file__)),
            "target_variance_module_source": common.sha256(Path(target_variance.__file__)),
            "moprp_common_loader_source": common.sha256(Path(common.__file__)),
            "formal_moprp_validator_source": common.sha256(Path(validation.__file__)),
            **assembled_input_hashes,
        },
        "code_revision": _code_revision(),
        "selection": selection,
    }
    (args.output_dir / "blinded_selection_manifest.json").write_text(
        json.dumps(blinded_manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.inference_only or not args.reveal_all_diagnostic:
        print(json.dumps(blinded_manifest, indent=2))
        return

    scored, decision = _score_after_reveal(
        raw, fit_records, ensemble_context, selection, fixed_bv_mean
    )
    scored.to_csv(args.output_dir / "nmr_pseudotruth_diagnostic_metrics.csv", index=False)
    (args.output_dir / "diagnostic_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bv-setting",
        default="published",
        help="named frozen setting in --coefficient-lock (default: published)",
    )
    parser.add_argument(
        "--coefficient-lock",
        type=Path,
        default=DEFAULT_COEFFICIENT_LOCK,
        help="coefficient_lock.json containing frozen_settings",
    )
    parser.add_argument(
        "--regularizations", default=",".join(map(str, DEFAULT_REGULARIZATIONS))
    )
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--pilot-maxiter", type=int, default=300)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--fold-scheme", choices=("interleaved", "contiguous"), default="interleaved")
    parser.add_argument("--inference-only", action="store_true", help="deprecated; inference-only is now the default")
    parser.add_argument(
        "--reveal-all-diagnostic",
        action="store_true",
        help="explicitly exhaust MoPrP blinding by scoring every swept candidate against NMR",
    )
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
