#!/usr/bin/env python3
"""Develop and qualify geometry-regularised HDX target-variance inference on TeaA/ISO.

The exact target is generated from published BV coefficients and the known 40:60
TeaA frame mixture.  Target weights are used only to generate synthetic observations,
fixed means, and post-fit truth metrics; the estimators receive none of them.  Method,
geometry, and regularisation are selected solely by held-out HDX reconstruction.

Execution is deliberately phased: ``--numerical-litmus`` performs no optimisation;
``--development-only`` compares and records non-promotable settings on split 0;
``--qualification-only`` accepts only a completed development artifact and reserves
splits 1 and 2 for the later frozen-settings gate.  ``--pilot`` is a cheap development
matrix and can never launch formal qualification.

The historical artificial TeaA dataset is registered as a separate model-mismatch
stress-test source and is never mixed with the coherent exact-mixture qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd

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
    qualification_gate,
    structured_residual_nll,
    variance_recovery_metrics,
    write_frozen_settings,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = HERE / "_pf_peptide_moment_final"
DEFAULT_OUTPUT_DIR = HERE / "_iso_target_variance"
REFERENCE_STRUCTURE = (
    HERE.parents[1]
    / "data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb"
)
PUBLISHED_STRESS_DATA = HERE / "_datasplits/full_dataset_dfrac.csv"
PUBLISHED_STRESS_TOPOLOGY = HERE / "_datasplits/full_dataset_topology.json"
PUBLISHED_BC, PUBLISHED_BH = 0.35, 2.0
TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], dtype=float)
ESTIMATORS = ("curve_moment", "structured_residual")
DEFAULT_REGULARIZATIONS = (0.0, 0.01, 0.1, 1.0)
DEVELOPMENT_ARTIFACT_TYPE = "hdx_target_variance_development_selection"
DEVELOPMENT_SPLITS = (0,)
QUALIFICATION_SPLITS = (1, 2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _residue_ids(topology_path: Path) -> np.ndarray:
    payload = json.loads(topology_path.read_text())
    return np.asarray([item["residues"][0] for item in payload["topologies"]], dtype=int)


def _coordinates(residue_ids: np.ndarray) -> np.ndarray:
    universe = mda.Universe(str(REFERENCE_STRUCTURE))
    ca = universe.select_atoms("name CA")
    lookup = {int(resid): np.asarray(position, dtype=float) for resid, position in zip(ca.resids, ca.positions, strict=True)}
    missing = sorted(set(residue_ids.tolist()) - set(lookup))
    if missing:
        raise ValueError(f"reference structure lacks TeaA residues {missing}")
    return np.asarray([lookup[int(residue)] for residue in residue_ids])


def _truth_weights(assignments: np.ndarray) -> np.ndarray:
    weights = np.zeros(assignments.size, dtype=float)
    for cluster, mass in ((0, 0.4), (1, 0.6)):
        count = int(np.sum(assignments == cluster))
        if count == 0:
            raise ValueError(f"TeaA ensemble has no frames in required cluster {cluster}")
        weights[assignments == cluster] = mass / count
    return weights


def _load_sources(panel_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((panel_dir / "manifest.json").read_text())
    sources: dict[str, dict[str, Any]] = {}
    for ensemble, stem in (("ISO_BI", "bi"), ("ISO_TRI", "tri")):
        feature_path = Path(manifest["inputs"][f"{stem}_features"]["path"])
        topology_path = Path(manifest["inputs"][f"{stem}_topology"]["path"])
        cluster_path = Path(manifest["inputs"][f"{stem}_clusters"]["path"])
        with np.load(feature_path) as data:
            log_pf = PUBLISHED_BC * np.asarray(data["heavy_contacts"], float) + PUBLISHED_BH * np.asarray(data["acceptor_contacts"], float)
            k_ints = np.asarray(data["k_ints"], float)
        assignments = pd.read_csv(cluster_path)["cluster_assignment"].to_numpy(int)
        residue_ids = _residue_ids(topology_path)
        if log_pf.shape[0] != residue_ids.size or log_pf.shape[1] != assignments.size:
            raise ValueError(f"{ensemble}: features, topology, and clusters are not aligned")
        sources[ensemble] = {
            "log_pf": log_pf,
            "k_ints": k_ints,
            "assignments": assignments,
            "residue_ids": residue_ids,
            "coordinates": _coordinates(residue_ids),
            "feature_path": feature_path,
            "topology_path": topology_path,
            "cluster_path": cluster_path,
        }
    mappings = {
        panel: np.asarray(np.load(panel_dir / f"panel_{panel}_mapping.npz")["mapping"], float)
        for panel in manifest["config"]["panels"]
    }
    return sources, mappings, manifest


def _exact_target(source: dict[str, Any], mapping: np.ndarray) -> dict[str, np.ndarray]:
    rates = effective_rates(source["log_pf"], source["k_ints"])
    weights = _truth_weights(source["assignments"])
    mean_rates = rates @ weights
    residue_uptake = np.einsum(
        "trf,f->tr",
        1.0 - np.exp(-TIMEPOINTS[:, None, None] * rates[None, :, :]),
        weights,
    )
    covariance = population_covariance(rates, weights)
    return {
        "rates": rates,
        "mean_rates": mean_rates,
        "observed": mapping @ residue_uptake.T,
        "covariance": covariance,
        "variances": np.diag(covariance),
        "mapped_covariance": map_hdx_covariance(covariance, mapping),
    }


def _log_rmse(left: np.ndarray, right: np.ndarray, floor: float = 1e-15) -> float:
    residual = np.log(np.clip(left, floor, None)) - np.log(np.clip(right, floor, None))
    return float(np.sqrt(np.mean(np.square(residual))))


def _fit_one(
    estimator: str,
    observed: np.ndarray,
    mean_rates: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    geometry_name: str,
    regularization: float,
    train_mask: np.ndarray,
    *,
    maxiter: int,
):
    common = dict(
        observed_uptake=observed,
        mean_rates=mean_rates,
        timepoints=TIMEPOINTS,
        mapping=mapping,
        geometry=geometry,
        geometry_name=geometry_name,
        regularization=regularization,
        observation_mask=train_mask,
        maxiter=maxiter,
    )
    if estimator == "curve_moment":
        return fit_curve_moment_variance(**common)
    if estimator == "structured_residual":
        return fit_structured_residual_variance(**common)
    raise ValueError(f"unknown estimator {estimator!r}")


def select_by_hdx_reconstruction(frame: pd.DataFrame) -> dict[str, Any]:
    """Select settings using only held-out HDX predictive scores.

    Both estimators use the same propagated-covariance Gaussian predictive score,
    ranked within validation cell without using variance truth.
    """

    required = {
        "ensemble", "panel", "split_index", "time_fold", "estimator", "geometry",
        "regularization", "heldout_reconstruction_score", "heldout_mean_mse_ratio",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"selection frame requires {sorted(required)}")
    optional_safeguards = [
        column for column in ("success", "finite_objective", "psd") if column in frame.columns
    ]
    work = frame[[*required, *optional_safeguards]].copy()
    candidate_columns = ["estimator", "geometry", "regularization"]
    if optional_safeguards:
        eligibility = (
            work.groupby(candidate_columns, as_index=False)[optional_safeguards]
            .all()
            .assign(eligible=lambda values: values[optional_safeguards].all(axis=1))
        )
        work = work.merge(
            eligibility[candidate_columns + ["eligible"]], on=candidate_columns
        )
        work = work[work.eligible]
        if work.empty:
            raise ValueError("no development candidate passed convergence/finite/PSD safeguards")
    cells = ["ensemble", "panel", "split_index", "time_fold"]
    work["reconstruction_rank"] = work.groupby(cells).heldout_reconstruction_score.rank(
        method="average", pct=True
    )
    summary = (
        work.groupby(candidate_columns, as_index=False)
        .agg(
            median_reconstruction_rank=("reconstruction_rank", "median"),
            median_heldout_mean_mse_ratio=("heldout_mean_mse_ratio", "median"),
        )
        .sort_values(
            ["median_reconstruction_rank", "median_heldout_mean_mse_ratio", "estimator", "geometry", "regularization"],
            kind="stable",
        )
    )
    primary = summary[~summary.geometry.isin(["identity", "shuffled_geometry"])]
    if primary.empty:
        raise ValueError("no converged physical geometry is available for development selection")
    selected = primary.iloc[0]
    shuffled = summary[summary.geometry == "shuffled_geometry"]
    identity = summary[summary.geometry == "identity"]
    shuffled_rank = (
        float(shuffled.iloc[0].median_reconstruction_rank) if not shuffled.empty else float("nan")
    )
    identity_rank = (
        float(identity.iloc[0].median_reconstruction_rank) if not identity.empty else float("nan")
    )
    return {
        "estimator": str(selected.estimator),
        "geometry": str(selected.geometry),
        "regularization": float(selected.regularization),
        "selection_criterion": "median within-cell held-out HDX predictive-NLL rank",
        "median_reconstruction_rank": float(selected.median_reconstruction_rank),
        "median_heldout_mean_mse_ratio": float(selected.median_heldout_mean_mse_ratio),
        "shuffled_control_rank": shuffled_rank,
        "identity_control_rank": identity_rank,
        "shuffled_control_beats_selected": bool(
            np.isfinite(shuffled_rank)
            and shuffled_rank < float(selected.median_reconstruction_rank)
        ),
        "identity_control_beats_selected": bool(
            np.isfinite(identity_rank)
            and identity_rank < float(selected.median_reconstruction_rank)
        ),
        "ranking": summary.to_dict("records"),
    }


def write_development_selection(
    path: Path,
    selection: dict[str, Any],
    *,
    pilot: bool,
) -> Path:
    """Write a non-qualified, non-promotable HDX-only selection artifact."""

    controls_passed = bool(
        not selection.get("shuffled_control_beats_selected", True)
        and not selection.get("identity_control_beats_selected", True)
    )
    can_launch_qualification = bool(not pilot and controls_passed)
    payload = {
        "artifact_type": DEVELOPMENT_ARTIFACT_TYPE,
        "artifact_version": 1,
        "status": (
            "pilot_only"
            if pilot
            else (
                "development_selection_complete"
                if controls_passed
                else "development_negative_control_failed"
            )
        ),
        "qualified": False,
        "development_controls_passed": controls_passed,
        "can_launch_qualification": can_launch_qualification,
        "can_launch_moprp_validation": False,
        "selection_source": "TeaA held-out HDX reconstruction only",
        "population_recovery_used_for_selection": False,
        "nmr_used_for_selection": False,
        "development_splits": list(DEVELOPMENT_SPLITS),
        "qualification_splits_reserved": list(QUALIFICATION_SPLITS),
        "settings": {
            **selection,
            "distance_cutoff_angstrom": 8.0,
            "published_bc": PUBLISHED_BC,
            "published_bh": PUBLISHED_BH,
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def load_development_selection(path: Path, *, allow_pilot: bool = False) -> dict[str, Any]:
    """Load frozen development choices while keeping qualification authority separate."""

    payload = json.loads(path.read_text())
    if payload.get("artifact_type") != DEVELOPMENT_ARTIFACT_TYPE:
        raise ValueError("qualification requires an HDX target-variance development artifact")
    if payload.get("qualified") is not False or payload.get("can_launch_moprp_validation") is not False:
        raise ValueError("development artifacts must be non-qualified and non-promotable")
    if bool(payload.get("population_recovery_used_for_selection", True)):
        raise ValueError("development settings are invalid: population recovery influenced selection")
    if bool(payload.get("nmr_used_for_selection", True)):
        raise ValueError("development settings are invalid: NMR influenced selection")
    if payload.get("status") == "pilot_only" and not allow_pilot:
        raise ValueError("pilot settings cannot launch formal TeaA qualification")
    if not allow_pilot and not bool(payload.get("can_launch_qualification", False)):
        raise ValueError(
            "development settings cannot launch qualification because identity/shuffled "
            "controls were not beaten"
        )
    return payload


def _qualification_rows(frame: pd.DataFrame, selection: dict[str, Any]) -> list[dict[str, Any]]:
    chosen = frame[
        (frame.estimator == selection["estimator"])
        & (frame.geometry == selection["geometry"])
        & np.isclose(frame.regularization, selection["regularization"])
    ].copy()
    shuffled = frame[
        (frame.estimator == selection["estimator"])
        & (frame.geometry == "shuffled_geometry")
        & np.isclose(frame.regularization, selection["regularization"])
    ][["ensemble", "panel", "split_index", "time_fold", "mapped_variance_log_rmse"]].rename(
        columns={"mapped_variance_log_rmse": "shuffled_log_rmse"}
    )
    chosen = chosen.merge(shuffled, on=["ensemble", "panel", "split_index", "time_fold"], how="left")
    chosen["beats_shuffled_geometry"] = (
        chosen.mapped_variance_log_rmse < chosen.shuffled_log_rmse
    ) & (selection["geometry"] != "shuffled_geometry")
    return chosen[
        [
            "ensemble", "panel", "heldout_mean_mse_ratio", "log_variance_spearman",
            "mapped_variance_log_rmse", "constant_mapped_variance_log_rmse",
            "beats_shuffled_geometry", "psd", "finite_objective",
        ]
    ].to_dict("records")


def _published_stress_mapping(
    residue_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the historical artificial curves and their one-residue trim mapping."""

    curves = pd.read_csv(PUBLISHED_STRESS_DATA)
    observed = curves[[str(index) for index in range(TIMEPOINTS.size)]].to_numpy(float)
    topologies = json.loads(PUBLISHED_STRESS_TOPOLOGY.read_text())["topologies"]
    if len(topologies) != observed.shape[0]:
        raise ValueError("published artificial curves and topology are not aligned")
    lookup = {int(residue): index for index, residue in enumerate(residue_ids)}
    rows, observations = [], []
    for topology, curve in zip(topologies, observed, strict=True):
        # These synthetic two-residue peptides use peptide_trim=1, leaving the last residue.
        active_residue = int(topology["residues"][-1])
        if active_residue not in lookup:
            continue
        row = np.zeros(residue_ids.size, dtype=float)
        row[lookup[active_residue]] = 1.0
        rows.append(row)
        observations.append(curve)
    return np.asarray(rows), np.asarray(observations)


def _run_published_model_mismatch_stress(
    selection: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    *,
    maxiter: int,
) -> pd.DataFrame:
    """Apply frozen selected settings to the historical artificial curve separately."""

    rows = []
    for ensemble, source in sources.items():
        mapping, observed = _published_stress_mapping(source["residue_ids"])
        rates = effective_rates(source["log_pf"], source["k_ints"])
        mean_rates = rates @ _truth_weights(source["assignments"])
        geometries = build_rate_geometries(
            rates, source["coordinates"], source["residue_ids"], cutoff_angstrom=8.0
        )
        geometry = geometries[selection["geometry"]]
        fit = _fit_one(
            selection["estimator"], observed, mean_rates, mapping, geometry,
            selection["geometry"], float(selection["regularization"]),
            np.ones_like(observed, dtype=bool), maxiter=maxiter,
        )
        baseline = predict_fixed_mean_uptake(mean_rates, TIMEPOINTS, mapping)
        rows.append(
            {
                "ensemble": ensemble,
                "dataset": "published_artificial_TeaA_model_mismatch",
                "estimator": selection["estimator"],
                "geometry": selection["geometry"],
                "regularization": selection["regularization"],
                "n_curves": observed.shape[0],
                "fixed_mean_mse": float(np.mean(np.square(baseline - observed))),
                "fitted_curve_mse": float(np.mean(np.square(fit.predicted_uptake - observed))),
                "objective": fit.objective,
                "finite_objective": bool(np.isfinite(fit.objective)),
                "success": fit.success,
                "included_in_selection": False,
                "has_coherent_frame_mixture_truth": False,
            }
        )
    return pd.DataFrame(rows)


def _finalize_qualification(
    frame: pd.DataFrame,
    selection: dict[str, Any],
    output_dir: Path,
    sources: dict[str, dict[str, Any]],
    *,
    stress_maxiter: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the truth gate to previously frozen settings and freeze only a pass."""

    frame.to_csv(output_dir / "qualification_raw.csv", index=False)
    decision = qualification_gate(_qualification_rows(frame, selection))
    (output_dir / "frozen_development_settings.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "qualification_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    _run_published_model_mismatch_stress(
        selection, sources, maxiter=stress_maxiter
    ).to_csv(output_dir / "published_artificial_curve_stress.csv", index=False)
    if decision["qualified"]:
        input_hashes = {
            str(source[key]): _sha256(source[key])
            for source in sources.values()
            for key in ("feature_path", "topology_path", "cluster_path")
        }
        write_frozen_settings(
            output_dir / "frozen_target_variance_settings.json",
            settings={
                **selection,
                "distance_cutoff_angstrom": 8.0,
                "published_bc": PUBLISHED_BC,
                "published_bh": PUBLISHED_BH,
            },
            qualification=decision,
            input_hashes=input_hashes,
        )
    return selection, decision


def _finalize_selection(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    pilot: bool,
) -> dict[str, Any]:
    """Select settings by HDX reconstruction without applying truth qualification gates."""

    frame.to_csv(output_dir / ("pilot_raw.csv" if pilot else "selection_raw.csv"), index=False)
    selection = select_by_hdx_reconstruction(frame)
    artifact_name = "pilot_selection.json" if pilot else "development_selection.json"
    write_development_selection(output_dir / artifact_name, selection, pilot=pilot)
    return selection


def run_numerical_litmus(
    sources: dict[str, dict[str, Any]],
    mappings: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Any]:
    """Exercise every real ISO geometry and panel without optimizer calls."""

    rows: list[dict[str, Any]] = []
    for ensemble, source in sources.items():
        rates = effective_rates(source["log_pf"], source["k_ints"])
        geometries = build_rate_geometries(
            rates,
            source["coordinates"],
            source["residue_ids"],
            cutoff_angstrom=8.0,
        )
        distances = np.linalg.norm(
            source["coordinates"][:, None, :] - source["coordinates"][None, :, :], axis=-1
        )
        sequential = (
            np.abs(source["residue_ids"][:, None] - source["residue_ids"][None, :]) == 1
        )
        unsupported = (distances >= 8.0) & ~sequential & ~np.eye(rates.shape[0], dtype=bool)
        for panel, mapping in mappings.items():
            target = _exact_target(source, mapping)
            zero = predict_curve_moment_uptake(
                target["mean_rates"], np.zeros_like(target["mean_rates"]), TIMEPOINTS, mapping
            )
            fixed = predict_fixed_mean_uptake(target["mean_rates"], TIMEPOINTS, mapping)
            for geometry_name, geometry in geometries.items():
                covariance = build_hdx_covariance(target["variances"], geometry)
                mapped = map_hdx_covariance(covariance, mapping)
                direct_mapped = mapping @ covariance @ mapping.T
                curve_prediction = predict_curve_moment_uptake(
                    target["mean_rates"], target["variances"], TIMEPOINTS, mapping
                )
                curve_objective = float(
                    np.mean(np.square(curve_prediction - target["observed"]))
                )
                structured_objective = structured_residual_nll(
                    target["observed"],
                    target["mean_rates"],
                    TIMEPOINTS,
                    mapping,
                    covariance,
                    noise_variance=1e-4,
                )
                support_ok = True
                if geometry_name in {"distance_only", "covariance_distance_sequence"}:
                    support_ok = bool(np.all(np.abs(geometry[unsupported]) < 1e-14))
                rows.append(
                    {
                        "ensemble": ensemble,
                        "panel": panel,
                        "geometry": geometry_name,
                        "geometry_psd": bool(np.linalg.eigvalsh(geometry).min() >= -1e-9),
                        "covariance_psd": bool(np.linalg.eigvalsh(covariance).min() >= -1e-9),
                        "diagonal_matches_d": bool(
                            np.allclose(np.diag(covariance), target["variances"], atol=1e-14)
                        ),
                        "mapping_congruence": bool(
                            np.allclose(mapped, direct_mapped, rtol=1e-12, atol=1e-14)
                        ),
                        "zero_variance_mean_limit": bool(
                            np.allclose(zero, fixed, rtol=1e-12, atol=1e-14)
                        ),
                        "eight_angstrom_support": support_ok,
                        "curve_objective": curve_objective,
                        "structured_objective": structured_objective,
                        "finite_objectives": bool(
                            np.isfinite(curve_objective) and np.isfinite(structured_objective)
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "numerical_litmus.csv", index=False)
    boolean_columns = (
        "geometry_psd",
        "covariance_psd",
        "diagonal_matches_d",
        "mapping_congruence",
        "zero_variance_mean_limit",
        "eight_angstrom_support",
        "finite_objectives",
    )
    decision = {
        "passed": bool(frame[list(boolean_columns)].to_numpy(bool).all()),
        "optimization_performed": False,
        "variance_fitting_performed": False,
        "ensemble_reweighting_performed": False,
        "bv_coefficients_optimized": False,
        "ensembles": sorted(frame.ensemble.unique().tolist()),
        "panels": sorted(frame.panel.unique().tolist()),
        "geometries": sorted(frame.geometry.unique().tolist()),
        "n_checks": int(len(frame)),
    }
    (output_dir / "numerical_litmus_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    return decision


def run(args: argparse.Namespace) -> None:
    if (
        not args.merge_shards
        and args.output_dir.exists()
        and any(args.output_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(f"refusing to overwrite existing output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources, mappings, source_manifest = _load_sources(args.panel_dir)
    if args.numerical_litmus:
        decision = run_numerical_litmus(sources, mappings, args.output_dir)
        print(json.dumps(decision, indent=2))
        return

    development_mode = bool(args.development_only or args.pilot or args.smoke)
    qualification_mode = bool(args.qualification_only)
    if development_mode == qualification_mode:
        raise ValueError(
            "choose exactly one of --development-only/--pilot or --qualification-only"
        )
    development_payload = None
    frozen_selection = None
    if qualification_mode:
        if args.selection_artifact is None:
            raise ValueError("--qualification-only requires --selection-artifact")
        development_payload = load_development_selection(args.selection_artifact)
        frozen_selection = dict(development_payload["settings"])

    if args.merge_shards:
        raw_stem = "selection_raw" if development_mode else "qualification_raw"
        paths = [
            args.output_dir / f"{raw_stem}.shard_{index:04d}_of_{args.num_shards:04d}.csv"
            for index in range(args.num_shards)
        ]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing target-variance qualification shards: {missing}")
        merged = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        keys = [
            "ensemble", "panel", "split_index", "time_fold", "estimator", "geometry",
            "regularization",
        ]
        if merged.duplicated(keys).any():
            raise ValueError("target-variance shards contain duplicate jobs")
        if development_mode:
            selection = _finalize_selection(merged, args.output_dir, pilot=False)
            print(json.dumps({"development_selection": selection}, indent=2))
        else:
            selection, decision = _finalize_qualification(
                merged,
                frozen_selection,
                args.output_dir,
                sources,
                stress_maxiter=args.maxiter,
            )
            print(json.dumps({"selection": selection, "qualification": decision}, indent=2))
        return

    pilot_mode = bool(args.pilot or args.smoke)
    if qualification_mode:
        estimators = (str(frozen_selection["estimator"]),)
        geometry_names = tuple(
            dict.fromkeys((str(frozen_selection["geometry"]), "identity", "shuffled_geometry"))
        )
        regularizations = (float(frozen_selection["regularization"]),)
        split_indices = QUALIFICATION_SPLITS
        time_folds = tuple(range(TIMEPOINTS.size))
        panel_names = tuple(mappings)
    else:
        estimators = ESTIMATORS
        geometry_names = GEOMETRY_NAMES
        regularization_text = args.pilot_regularizations if pilot_mode else args.regularizations
        regularizations = tuple(float(value) for value in regularization_text.split(","))
        split_indices = DEVELOPMENT_SPLITS
        time_folds = (0,) if pilot_mode else tuple(range(TIMEPOINTS.size))
        panel_names = ("equal",) if pilot_mode else tuple(mappings)
    rows: list[dict[str, Any]] = []
    jobs = 0
    for ensemble, source in sources.items():
        geometries = build_rate_geometries(
            effective_rates(source["log_pf"], source["k_ints"]),
            source["coordinates"],
            source["residue_ids"],
            cutoff_angstrom=8.0,
        )
        for panel in panel_names:
            mapping = mappings[panel]
            target = _exact_target(source, mapping)
            fixed = predict_fixed_mean_uptake(target["mean_rates"], TIMEPOINTS, mapping)
            baseline_residual = np.square(fixed - target["observed"])
            true_marginal, true_conditional = covariance_profiles(target["mapped_covariance"])
            positive_truth = target["variances"][target["variances"] > 0]
            constant_value = float(np.exp(np.mean(np.log(positive_truth))))
            split_metadata = source_manifest["split_metadata"]
            for split_index in split_indices:
                split = split_metadata[f"{panel}_{split_index}"]
                train_peptides = np.asarray(split["train_indices"], dtype=int)
                val_peptides = np.setdiff1d(
                    np.asarray(split["val_indices"], dtype=int), train_peptides
                )
                for time_fold in time_folds:
                    train_times = np.setdiff1d(np.arange(TIMEPOINTS.size), [time_fold])
                    train_mask = np.zeros_like(target["observed"], dtype=bool)
                    train_mask[np.ix_(train_peptides, train_times)] = True
                    validation_mask = np.zeros_like(target["observed"], dtype=bool)
                    validation_mask[np.ix_(val_peptides, [time_fold])] = True
                    baseline_mse = float(np.mean(baseline_residual[validation_mask]))
                    for estimator in estimators:
                        for geometry_name in geometry_names:
                            for regularization in regularizations:
                                selected_for_shard = jobs % args.num_shards == args.shard_index
                                jobs += 1
                                if not selected_for_shard:
                                    continue
                                geometry = geometries[geometry_name]
                                fit = _fit_one(
                                    estimator, target["observed"], target["mean_rates"], mapping,
                                    geometry, geometry_name, regularization, train_mask,
                                    maxiter=min(args.maxiter, args.pilot_maxiter)
                                    if pilot_mode
                                    else args.maxiter,
                                )
                                heldout_mse = float(
                                    np.mean(np.square(fit.predicted_uptake - target["observed"])[validation_mask])
                                )
                                heldout_score = structured_residual_nll(
                                    target["observed"],
                                    target["mean_rates"],
                                    TIMEPOINTS,
                                    mapping,
                                    fit.covariance,
                                    noise_variance=1e-4,
                                    observation_mask=validation_mask,
                                    predicted_uptake=fit.predicted_uptake,
                                )
                                residue_metrics = variance_recovery_metrics(
                                    fit.variances, target["variances"]
                                )
                                inferred_marginal, inferred_conditional = covariance_profiles(
                                    fit.mapped_covariance
                                )
                                mapped_log_rmse = 0.5 * (
                                    _log_rmse(inferred_marginal, true_marginal)
                                    + _log_rmse(inferred_conditional, true_conditional)
                                )
                                constant_covariance = np.sqrt(np.full_like(fit.variances, constant_value))[:, None] * geometry * np.sqrt(np.full_like(fit.variances, constant_value))[None, :]
                                constant_marginal, constant_conditional = covariance_profiles(
                                    map_hdx_covariance(constant_covariance, mapping)
                                )
                                constant_log_rmse = 0.5 * (
                                    _log_rmse(constant_marginal, true_marginal)
                                    + _log_rmse(constant_conditional, true_conditional)
                                )
                                rows.append(
                                    {
                                        "ensemble": ensemble,
                                        "panel": panel,
                                        "split_index": int(split_index),
                                        "time_fold": int(time_fold),
                                        "estimator": estimator,
                                        "geometry": geometry_name,
                                        "regularization": regularization,
                                        "heldout_reconstruction_score": heldout_score,
                                        "heldout_mean_mse": heldout_mse,
                                        "heldout_mean_mse_ratio": heldout_mse / max(baseline_mse, 1e-15),
                                        "mapped_variance_log_rmse": mapped_log_rmse,
                                        "constant_mapped_variance_log_rmse": constant_log_rmse,
                                        "mapped_marginal_log_rmse": _log_rmse(inferred_marginal, true_marginal),
                                        "mapped_conditional_log_rmse": _log_rmse(inferred_conditional, true_conditional),
                                        "psd": bool(np.linalg.eigvalsh(fit.covariance).min() >= -1e-9),
                                        "finite_objective": bool(np.isfinite(fit.objective)),
                                        "success": fit.success,
                                        "iterations": fit.iterations,
                                        "optimizer_message": fit.message,
                                        "objective": fit.objective,
                                        **residue_metrics,
                                    }
                                )
    if not rows:
        raise RuntimeError("no target-variance development/qualification jobs were selected")
    frame = pd.DataFrame(rows)
    suffix = f".shard_{args.shard_index:04d}_of_{args.num_shards:04d}" if args.num_shards > 1 else ""
    raw_stem = (
        "pilot_raw" if pilot_mode else "selection_raw"
    ) if development_mode else "qualification_raw"
    raw_path = args.output_dir / f"{raw_stem}{suffix}.csv"
    frame.to_csv(raw_path, index=False)
    if args.num_shards > 1:
        metadata = {
            "status": "development_shard_only" if development_mode else "qualification_shard_only",
            "fitting_performed": True,
            "variance_fitting_performed": True,
            "ensemble_reweighting_performed": False,
            "published_artificial_curve": {
                "path": str(PUBLISHED_STRESS_DATA),
                "role": "separate model-mismatch stress test; excluded from coherent-target selection",
            },
        }
        (args.output_dir / f"run_metadata{suffix}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"wrote {raw_path}")
        return
    if development_mode:
        selection = _finalize_selection(frame, args.output_dir, pilot=pilot_mode)
        print(json.dumps({"development_selection": selection, "pilot": pilot_mode}, indent=2))
        return
    selection, decision = _finalize_qualification(
        frame,
        frozen_selection,
        args.output_dir,
        sources,
        stress_maxiter=args.maxiter,
    )
    print(json.dumps({"selection": selection, "qualification": decision}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--regularizations", default=",".join(map(str, DEFAULT_REGULARIZATIONS)))
    parser.add_argument("--pilot-regularizations", default="0.1")
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--pilot-maxiter", type=int, default=2000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--numerical-litmus", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--selection-artifact", type=Path)
    parser.add_argument("--smoke", action="store_true", help="deprecated alias for --pilot")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_index < num_shards")
    run(args)


if __name__ == "__main__":
    main()
