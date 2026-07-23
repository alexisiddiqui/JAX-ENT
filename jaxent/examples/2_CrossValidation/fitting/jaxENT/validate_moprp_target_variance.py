#!/usr/bin/env python3
"""Blinded MoPrP validation of a frozen TeaA target-variance estimator.

Inference reads only HDX curves, fixed published-BV trajectory features, intrinsic
rates, and structural geometry.  NMR states and populations are revealed only after
all primary and shuffled-control fits have completed and the blinded arrays have been
written.  No frame weights or BV coefficients are optimized here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np
import pandas as pd

import _moprp_recovery_common as common
from jaxent.src.analysis.hdx_target_variance import (
    build_rate_geometries,
    covariance_profiles,
    effective_rates,
    fit_curve_moment_variance,
    fit_structured_residual_variance,
    load_frozen_settings,
    map_hdx_covariance,
    population_covariance,
    variance_recovery_metrics,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = (
    common.PACKAGE_ROOT
    / "examples/1_IsoValidation_OMass/fitting/jaxENT/_iso_target_variance"
    / "frozen_target_variance_settings.json"
)
DEFAULT_OUTPUT_DIR = HERE / "_moprp_target_variance_validation"
STRUCTURE = common.BASE / "data/_MoPrP/MoPrP_max_plddt_4334.pdb"
UNMAPPED_RESIDUE = 101


def _coordinates(residue_ids: np.ndarray) -> np.ndarray:
    universe = mda.Universe(str(STRUCTURE))
    ca = universe.select_atoms("name CA")
    lookup = {int(resid): np.asarray(position, float) for resid, position in zip(ca.resids, ca.positions, strict=True)}
    missing = sorted(set(residue_ids.tolist()) - set(lookup))
    if missing:
        raise ValueError(f"MoPrP structure lacks feature residues {missing}")
    return np.asarray([lookup[int(residue)] for residue in residue_ids])


def peptide_partitions(inputs: common.BlindedEnsembleInputs) -> dict[str, np.ndarray | int]:
    """Return fitted, peptide-1, and residue-101 peptide row indices."""

    residue_column = np.flatnonzero(inputs.feature_residue_ids == UNMAPPED_RESIDUE)
    if residue_column.size != 1:
        raise ValueError("expected exactly one feature column for residue 101")
    invalid = np.flatnonzero(inputs.mapping[:, residue_column[0]] > 0)
    if invalid.size != 1:
        raise ValueError("expected exactly one peptide containing unmapped residue 101")
    heldout = int(np.flatnonzero(inputs.peptide_ids == 1)[0])
    fit_rows = np.setdiff1d(np.arange(inputs.mapping.shape[0]), [heldout, int(invalid[0])])
    return {"fit_rows": fit_rows, "peptide1_row": heldout, "unmapped_101_row": int(invalid[0])}


def _fit(
    estimator: str,
    observed: np.ndarray,
    mean_rates: np.ndarray,
    timepoints: np.ndarray,
    mapping: np.ndarray,
    geometry: np.ndarray,
    geometry_name: str,
    regularization: float,
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
        maxiter=maxiter,
    )
    if estimator == "curve_moment":
        return fit_curve_moment_variance(**arguments)
    if estimator == "structured_residual":
        return fit_structured_residual_variance(**arguments)
    raise ValueError(f"unsupported frozen estimator {estimator!r}")


def infer_blinded(
    inputs: common.BlindedEnsembleInputs,
    settings: dict[str, Any],
    *,
    maxiter: int,
) -> dict[str, Any]:
    """Infer primary and shuffled amplitudes without NMR populations."""

    bc = float(settings.get("published_bc", common.PUBLISHED_BC))
    bh = float(settings.get("published_bh", common.PUBLISHED_BH))
    log_pf = inputs.log_pf_by_frame(bc, bh)
    rates = effective_rates(log_pf, inputs.k_ints)
    mean_rates = np.mean(rates, axis=1)
    geometries = build_rate_geometries(
        rates,
        _coordinates(inputs.feature_residue_ids),
        inputs.feature_residue_ids,
        cutoff_angstrom=float(settings["distance_cutoff_angstrom"]),
    )
    partitions = peptide_partitions(inputs)
    fit_rows = np.asarray(partitions["fit_rows"], dtype=int)
    mapping = inputs.mapping[fit_rows]
    observed = inputs.observed_uptake[fit_rows]
    estimator = str(settings["estimator"])
    regularization = float(settings["regularization"])
    primary_geometry = str(settings["geometry"])
    if primary_geometry == "shuffled_geometry":
        raise ValueError("a shuffled negative control cannot be the frozen primary geometry")
    fits = {
        name: _fit(
            estimator,
            observed,
            mean_rates,
            inputs.timepoints,
            mapping,
            geometries[name],
            name,
            regularization,
            maxiter=maxiter,
        )
        for name in (primary_geometry, "shuffled_geometry")
    }
    return {
        "ensemble": inputs.ensemble,
        "mean_rates": mean_rates,
        "rates_by_frame": rates,
        "mapping": mapping,
        "fit_rows": fit_rows,
        "peptide1_row": int(partitions["peptide1_row"]),
        "unmapped_101_row": int(partitions["unmapped_101_row"]),
        "primary_geometry": primary_geometry,
        "fits": fits,
    }


def _log_rmse(left: np.ndarray, right: np.ndarray, floor: float = 1e-15) -> float:
    residual = np.log(np.clip(left, floor, None)) - np.log(np.clip(right, floor, None))
    return float(np.sqrt(np.mean(np.square(residual))))


def reveal_and_score(
    inputs: common.BlindedEnsembleInputs,
    inference: dict[str, Any],
) -> dict[str, Any]:
    """Reveal ``w_NMR`` after inference and score in identical rate coordinates."""

    _, _, _, nmr_weights = common.reveal_nmr_reference(
        inputs.ensemble, expected_frames=inputs.n_frames
    )
    truth = population_covariance(inference["rates_by_frame"], nmr_weights)
    true_variances = np.diag(truth)
    mapped_truth = map_hdx_covariance(truth, inference["mapping"])
    true_marginal, true_conditional = covariance_profiles(mapped_truth)
    metrics: dict[str, Any] = {}
    for label, fit in inference["fits"].items():
        inferred_marginal, inferred_conditional = covariance_profiles(fit.mapped_covariance)
        metrics[label] = {
            **variance_recovery_metrics(fit.variances, true_variances),
            "mapped_marginal_log_rmse": _log_rmse(inferred_marginal, true_marginal),
            "mapped_conditional_log_rmse": _log_rmse(inferred_conditional, true_conditional),
            "mapped_variance_log_rmse": 0.5 * (
                _log_rmse(inferred_marginal, true_marginal)
                + _log_rmse(inferred_conditional, true_conditional)
            ),
            "objective": fit.objective,
            "success": fit.success,
        }
    primary = inference["fits"][inference["primary_geometry"]]
    constant_variance = float(np.exp(np.mean(np.log(np.clip(primary.variances, 1e-15, None)))))
    constant_covariance = np.eye(primary.variances.size) * constant_variance
    constant_marginal, constant_conditional = covariance_profiles(
        map_hdx_covariance(constant_covariance, inference["mapping"])
    )
    constant_log_rmse = 0.5 * (
        _log_rmse(constant_marginal, true_marginal)
        + _log_rmse(constant_conditional, true_conditional)
    )
    primary_metrics = metrics[inference["primary_geometry"]]
    shuffled_metrics = metrics["shuffled_geometry"]
    passed = bool(
        primary_metrics["mapped_variance_log_rmse"] < constant_log_rmse
        and primary_metrics["mapped_variance_log_rmse"] < shuffled_metrics["mapped_variance_log_rmse"]
        and primary_metrics["log_variance_spearman"] > 0
    )

    # Peptide 1 remains independent: compare prediction and the uniform-frame envelope only.
    peptide1 = inference["peptide1_row"]
    frame_uptake = 1.0 - np.exp(
        -inputs.timepoints[:, None, None] * inference["rates_by_frame"][None, :, :]
    )
    peptide1_by_frame = np.einsum("r,trf->tf", inputs.mapping[peptide1], frame_uptake)
    peptide1_prediction = np.einsum(
        "r,tr->t",
        inputs.mapping[peptide1],
        1.0 - np.exp(-inputs.timepoints[:, None] * inference["mean_rates"][None, :]),
    )
    peptide1_observed = inputs.observed_uptake[peptide1]
    envelope = {
        "mse": float(np.mean(np.square(peptide1_prediction - peptide1_observed))),
        "fraction_inside_uniform_frame_envelope": float(
            np.mean(
                (peptide1_observed >= peptide1_by_frame.min(axis=1))
                & (peptide1_observed <= peptide1_by_frame.max(axis=1))
            )
        ),
    }
    return {
        "ensemble": inputs.ensemble,
        "primary_geometry": inference["primary_geometry"],
        "metrics": metrics,
        "constant_mapped_variance_log_rmse": constant_log_rmse,
        "passes_variance_gate": passed,
        "peptide1_independent_diagnostic": envelope,
        "unmapped_101_peptide_id": int(inputs.peptide_ids[inference["unmapped_101_row"]]),
    }


def run(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing output directory {args.output_dir}")
    artifact = load_frozen_settings(args.artifact, require_qualified=True)
    settings = dict(artifact["settings"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: NMR-blinded inference for both ensembles.
    inputs = {name: common.load_blinded_ensemble_inputs(name) for name in common.ENSEMBLES}
    inferences = {
        name: infer_blinded(item, settings, maxiter=args.maxiter)
        for name, item in inputs.items()
    }
    arrays: dict[str, np.ndarray] = {}
    for ensemble, inference in inferences.items():
        arrays[f"{ensemble}__mean_rates"] = inference["mean_rates"]
        arrays[f"{ensemble}__fit_rows"] = inference["fit_rows"]
        for label, fit in inference["fits"].items():
            arrays[f"{ensemble}__{label}__variances"] = fit.variances
            arrays[f"{ensemble}__{label}__covariance"] = fit.covariance
            arrays[f"{ensemble}__{label}__mapped_covariance"] = fit.mapped_covariance
    np.savez_compressed(args.output_dir / "blinded_inference.npz", **arrays)
    blinded_manifest = {
        "status": "inference_complete_before_nmr_reveal",
        "artifact": str(args.artifact),
        "settings": settings,
        "input_hashes": common.blinded_input_hashes(),
        "nmr_used_for_inference": False,
        "ensemble_reweighting_performed": False,
        "bv_coefficients_optimized": False,
    }
    (args.output_dir / "blinded_inference_manifest.json").write_text(
        json.dumps(blinded_manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.inference_only:
        print("blinded inference complete; NMR pseudo-truth was not read")
        return

    # Phase 2: explicit reveal and post-hoc pseudo-ground-truth evaluation.
    reports = [reveal_and_score(inputs[name], inferences[name]) for name in common.ENSEMBLES]
    overall = bool(len(reports) == 2 and all(report["passes_variance_gate"] for report in reports))
    decision = {
        "both_ensembles_pass": overall,
        "weight_bv_optimization_authorized": overall,
        "ensembles": reports,
        "nmr_role": "post-inference pseudo-ground-truth evaluation only",
    }
    (args.output_dir / "nmr_pseudotruth_validation.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    flat_rows = []
    for report in reports:
        for geometry, values in report["metrics"].items():
            flat_rows.append({"ensemble": report["ensemble"], "geometry": geometry, **values})
    pd.DataFrame(flat_rows).to_csv(args.output_dir / "nmr_pseudotruth_metrics.csv", index=False)
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
