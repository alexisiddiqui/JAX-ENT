#!/usr/bin/env python3
"""Calibrate and apply a shared-rate peptide HDX mixture model.

The fitted covariance is local penalized-Hessian uncertainty.  The covariance across
peptide scores is spatial/kinetic heterogeneity.  Neither is experimental conformational
covariance, and this script does not optimize ensemble weights or add a production loss.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.src.analysis.hdx_rate_mixture import (
    RateMixtureFit,
    fit_shared_rate_mixture,
    hessian_uncertainty,
    mahalanobis_distance,
    peptide_score_covariances,
    prediction_jacobian_diagnostics,
    predict_uptake,
    project_curves_to_rates,
    select_shared_rate_model,
)
from jaxent.src.models.HDX.BV.features import BV_input_features

REPO = Path(__file__).resolve().parents[4]
TIMEPOINTS = np.asarray(
    [0.08, 0.33, 0.67, 1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 160.0, 240.0, 390.0, 750.0, 1440.0]
)
BV_BC, BV_BH = 0.35, 2.0
DEFAULT_SHRINKAGES = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_NOISE = (0.005, 0.01, 0.02)
CAPACITY_RMSE_GATE = 0.005
STABILITY_GRADIENT_GATE = 1e-5


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ISO_LITMUS = _load_module(
    "_rate_mixture_iso_litmus",
    REPO / "examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_uptake_rate_covariance.py",
)
MOPRP_CANARY = _load_module(
    "_rate_mixture_moprp_canary",
    REPO / "examples/2_CrossValidation/fitting/jaxENT/investigate_moprp_uptake_covariance.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def effective_rates(log_pf: np.ndarray, k_int: np.ndarray) -> np.ndarray:
    """Frames x residues effective rates."""
    return (np.asarray(k_int)[:, None] * np.exp(-np.asarray(log_pf))).T


def peptide_curves_from_rates(
    rates: np.ndarray,
    mapping: np.ndarray,
    frame_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Peptides x time curves from residues or frames x residues rates."""
    rates = np.asarray(rates, float)
    if rates.ndim == 1:
        residue_uptake = 1.0 - np.exp(-TIMEPOINTS[:, None] * rates[None, :])
    elif rates.ndim == 2:
        weights = np.full(len(rates), 1 / len(rates)) if frame_weights is None else np.asarray(frame_weights)
        weights = weights / weights.sum()
        frame_uptake = 1.0 - np.exp(-TIMEPOINTS[:, None, None] * rates[None, :, :])
        residue_uptake = np.einsum("tfr,f->tr", frame_uptake, weights)
    else:
        raise ValueError("rates must be residues or frames x residues")
    return (residue_uptake @ np.asarray(mapping, float).T).T


def _fit_selected(
    curves: np.ndarray,
    components: tuple[int, ...],
    shrinkages: tuple[float, ...],
    starts: int,
    seed: int,
) -> tuple[RateMixtureFit, list[dict[str, Any]]]:
    selection = select_shared_rate_model(
        curves,
        TIMEPOINTS,
        components=components,
        shrinkages=shrinkages,
        starts=starts,
        seed=seed,
    )
    fit = fit_shared_rate_mixture(
        curves,
        TIMEPOINTS,
        selection.selected_components,
        selection.selected_shrinkage,
        starts=starts,
        seed=seed + 50_000,
    )
    return fit, [dict(row) for row in selection.rows]


def _weight_blocks(covariance: np.ndarray, n_peptides: int, width: int) -> list[np.ndarray]:
    return [
        covariance[p * width : (p + 1) * width, p * width : (p + 1) * width]
        for p in range(n_peptides)
    ]


def _candidate_distance(
    target_weights: np.ndarray,
    candidate_weights: np.ndarray,
    covariance: np.ndarray,
) -> float:
    width = target_weights.shape[1]
    blocks = _weight_blocks(covariance, len(target_weights), width)
    distances = [
        mahalanobis_distance(target_weights[p], candidate_weights[p], blocks[p])
        for p in range(len(target_weights))
    ]
    return float(np.mean(distances))


def load_iso_inputs(results_dir: Path) -> dict[str, Any]:
    inputs, bi_log_pf, k_int, assignments, known, mappings = ISO_LITMUS.load_inputs(results_dir)
    manifest = json.loads((results_dir / "manifest.json").read_text())
    tri_path = Path(manifest["inputs"]["tri_features"]["path"])
    with np.load(tri_path) as data:
        tri_log_pf = BV_BC * np.asarray(data["heavy_contacts"], float) + BV_BH * np.asarray(
            data["acceptor_contacts"], float
        )
        tri_k_int = np.asarray(data["k_ints"], float)
    return {
        "inputs": {**inputs, "tri_features": {"path": str(tri_path), "sha256": sha256(tri_path)}},
        "bi_log_pf": bi_log_pf,
        "tri_log_pf": tri_log_pf,
        "k_int": k_int,
        "tri_k_int": tri_k_int,
        "known": known,
        "mappings": mappings,
    }


def iso_target_curves(iso: dict[str, Any], panel: str) -> dict[str, np.ndarray]:
    mapping = np.asarray(iso["mappings"][panel], float)
    known = np.asarray(iso["known"], float)
    bi_rates = effective_rates(iso["bi_log_pf"], iso["k_int"])
    z_bar = np.einsum("rf,f->r", iso["bi_log_pf"], known)
    return {
        "average_first": peptide_curves_from_rates(
            iso["k_int"] * np.exp(-z_bar), mapping
        ),
        "frame_mixture": peptide_curves_from_rates(bi_rates, mapping, known),
    }


def run_iso_capacity(
    iso: dict[str, Any],
    *,
    components: tuple[int, ...],
    starts: int,
    smoke: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Measure the unregularized approximation ceiling before any noise study."""
    panels = list(iso["mappings"])
    if smoke:
        panels = panels[:1]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for panel_index, panel in enumerate(panels):
        targets = iso_target_curves(iso, panel)
        for semantics_index, (semantics, curves) in enumerate(targets.items()):
            for n_components in components:
                fit = fit_shared_rate_mixture(
                    curves,
                    TIMEPOINTS,
                    n_components,
                    0.0,
                    starts=starts,
                    seed=700_000 + 10_000 * panel_index + 1_000 * semantics_index + n_components,
                )
                diagnostics = prediction_jacobian_diagnostics(fit, TIMEPOINTS)
                rows.append(
                    {
                        "panel": panel,
                        "semantics": semantics,
                        "n_components": n_components,
                        "rmse": fit.rmse,
                        "objective": fit.objective,
                        "optimizer_success": fit.success,
                        "n_observations": diagnostics.n_observations,
                        "n_parameters": diagnostics.n_parameters,
                        "jacobian_rank": diagnostics.numerical_rank,
                        "jacobian_effective_rank": diagnostics.effective_rank,
                        "jacobian_condition": diagnostics.condition_number,
                        "smallest_relative_singular_value": diagnostics.smallest_relative_singular_value,
                        "rates": ";".join(f"{rate:.12g}" for rate in fit.rates),
                        "passes_rmse_gate": fit.rmse <= CAPACITY_RMSE_GATE,
                    }
                )
                key = f"capacity__{panel}__{semantics}__k{n_components}"
                arrays[f"{key}__rates"] = fit.rates
                arrays[f"{key}__weights"] = fit.weights
                arrays[f"{key}__predicted"] = fit.predicted
    frame = pd.DataFrame(rows)
    units = []
    for (panel, semantics), group in frame.groupby(["panel", "semantics"], sort=False):
        best = group.sort_values(["rmse", "n_components"]).iloc[0]
        units.append(
            {
                "panel": panel,
                "semantics": semantics,
                "best_components": int(best.n_components),
                "best_rmse": float(best.rmse),
                "jacobian_rank": int(best.jacobian_rank),
                "n_parameters": int(best.n_parameters),
                "jacobian_condition": float(best.jacobian_condition),
                "passed": bool(best.rmse <= CAPACITY_RMSE_GATE),
            }
        )
    gates = {
        "rmse_gate": CAPACITY_RMSE_GATE,
        "units": units,
        "all_units_pass": bool(units and all(unit["passed"] for unit in units)),
        "smoke": smoke,
    }
    return frame, gates, arrays


def run_iso_stability(
    iso: dict[str, Any],
    *,
    n_components: int,
    repeats: int,
    maxiter: int,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Test whether independent near-optimal fits recover the same rates and scores."""
    panels = list(iso["mappings"])
    if smoke:
        panels = panels[:1]
        repeats = min(repeats, 3)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for panel_index, panel in enumerate(panels):
        for semantics_index, (semantics, curves) in enumerate(
            iso_target_curves(iso, panel).items()
        ):
            fits = []
            for repeat in range(repeats):
                fit = fit_shared_rate_mixture(
                    curves,
                    TIMEPOINTS,
                    n_components,
                    0.0,
                    starts=1,
                    seed=900_000 + 10_000 * panel_index + 1_000 * semantics_index + repeat,
                    maxiter=maxiter,
                )
                fits.append(fit)
                converged = bool(
                    fit.success or fit.gradient_norm <= STABILITY_GRADIENT_GATE
                )
                rows.append(
                    {
                        "panel": panel,
                        "semantics": semantics,
                        "repeat": repeat,
                        "n_components": n_components,
                        "rmse": fit.rmse,
                        "optimizer_success": fit.success,
                        "converged": converged,
                        "iterations": fit.iterations,
                        "gradient_norm": fit.gradient_norm,
                        **{
                            f"rate_{component}": rate
                            for component, rate in enumerate(fit.rates)
                        },
                    }
                )
            rmses = np.asarray([fit.rmse for fit in fits])
            best_rmse = float(rmses.min())
            tolerance = max(1e-4, 0.10 * best_rmse)
            selected = [fit for fit in fits if fit.rmse <= best_rmse + tolerance]
            selected_converged = [
                fit.success or fit.gradient_norm <= STABILITY_GRADIENT_GATE
                for fit in selected
            ]
            converged_fraction = float(np.mean(selected_converged))
            rates = np.stack([fit.rates for fit in selected])
            weights = np.stack([fit.weights for fit in selected])
            predictions = np.stack([fit.predicted for fit in selected])
            flat_weights = weights.reshape(len(selected), -1)
            if len(selected) > 1:
                correlation = np.corrcoef(flat_weights)
                upper = correlation[np.triu_indices(len(selected), 1)]
                median_correlation = float(np.nanmedian(upper))
            else:
                median_correlation = np.nan
            rate_log_sd = np.std(np.log(rates), axis=0, ddof=0)
            weight_sd_mean = float(np.mean(np.std(weights, axis=0, ddof=0)))
            prediction_sd_mean = float(np.mean(np.std(predictions, axis=0, ddof=0)))
            stable = bool(
                len(selected) >= max(3, repeats // 2)
                and float(rate_log_sd.max()) <= 0.10
                and weight_sd_mean <= 0.02
                and prediction_sd_mean <= 1e-3
                and median_correlation >= 0.95
                and converged_fraction >= 0.90
            )
            summaries.append(
                {
                    "panel": panel,
                    "semantics": semantics,
                    "n_components": n_components,
                    "repeats": repeats,
                    "near_optimal_count": len(selected),
                    "best_rmse": best_rmse,
                    "worst_near_optimal_rmse": float(max(fit.rmse for fit in selected)),
                    "optimizer_success_fraction": float(np.mean([fit.success for fit in fits])),
                    "near_optimal_converged_fraction": converged_fraction,
                    "max_rate_log_sd": float(rate_log_sd.max()),
                    "mean_weight_sd": weight_sd_mean,
                    "mean_prediction_sd": prediction_sd_mean,
                    "median_weight_correlation": median_correlation,
                    "stable": stable,
                }
            )
            key = f"stability__{panel}__{semantics}__k{n_components}"
            arrays[f"{key}__rates"] = rates
            arrays[f"{key}__weights"] = weights
            arrays[f"{key}__predictions"] = predictions
    return pd.DataFrame(rows), pd.DataFrame(summaries), arrays


def run_iso_calibration(
    iso: dict[str, Any],
    *,
    components: tuple[int, ...],
    shrinkages: tuple[float, ...],
    noise_levels: tuple[float, ...],
    seeds: int,
    starts: int,
    smoke: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    bi_rates = effective_rates(iso["bi_log_pf"], iso["k_int"])
    tri_rates = effective_rates(iso["tri_log_pf"], iso["tri_k_int"])
    known = np.asarray(iso["known"], float)
    uniform_tri = np.full(len(tri_rates), 1 / len(tri_rates))
    panels = list(iso["mappings"])
    if smoke:
        panels = panels[:1]

    fit_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    all_gate_units: list[dict[str, Any]] = []

    for panel_index, panel in enumerate(panels):
        mapping = np.asarray(iso["mappings"][panel], float)
        z_bar = np.einsum("rf,f->r", iso["bi_log_pf"], known)
        average_first = peptide_curves_from_rates(iso["k_int"] * np.exp(-z_bar), mapping)
        frame_mixture = peptide_curves_from_rates(bi_rates, mapping, known)
        positive_curves = {
            "average_first": average_first,
            "frame_mixture": frame_mixture,
        }
        tri_curves = peptide_curves_from_rates(tri_rates, mapping, uniform_tri)

        for semantics_index, (semantics, target_curves) in enumerate(positive_curves.items()):
            base_seed = 10_000 * panel_index + 1_000 * semantics_index + 1729
            oracle_fit, selection_rows = _fit_selected(
                target_curves, components, shrinkages, starts, base_seed
            )
            oracle_uncertainty = hessian_uncertainty(
                oracle_fit, target_curves, TIMEPOINTS, condition_limit=np.inf
            )
            positive_weights, _ = project_curves_to_rates(
                target_curves, TIMEPOINTS, oracle_fit.rates, starts=max(2, starts), seed=base_seed
            )
            tri_weights, _ = project_curves_to_rates(
                tri_curves, TIMEPOINTS, oracle_fit.rates, starts=max(2, starts), seed=base_seed + 1
            )
            key = f"iso__{panel}__{semantics}"
            arrays[f"{key}__rates"] = oracle_fit.rates
            arrays[f"{key}__weights"] = oracle_fit.weights
            arrays[f"{key}__joint_weight_covariance"] = oracle_uncertainty.joint_weight_covariance
            fit_rows.append(
                {
                    "system": "ISO",
                    "panel": panel,
                    "semantics": semantics,
                    "n_components": oracle_fit.n_components,
                    "shrinkage": oracle_fit.shrinkage,
                    "rmse": oracle_fit.rmse,
                    "hessian_condition": oracle_uncertainty.condition_number,
                    "hessian_valid": oracle_uncertainty.valid,
                }
            )
            for row in selection_rows:
                row.update({"system": "ISO", "panel": panel, "semantics": semantics})
                sample_rows.append({"row_type": "selection", **row})

            noise_conditions = (0.0,) + noise_levels
            for sigma in noise_conditions:
                condition_rows = []
                n_draws = 1 if sigma == 0 else seeds
                for draw in range(n_draws):
                    rng = np.random.default_rng(base_seed + 100_000 + draw + int(sigma * 1e6))
                    noisy = target_curves if sigma == 0 else target_curves + rng.normal(0, sigma, target_curves.shape)
                    bound_fraction = float(np.mean((noisy < 0) | (noisy > 1)))
                    fit = fit_shared_rate_mixture(
                        noisy,
                        TIMEPOINTS,
                        oracle_fit.n_components,
                        oracle_fit.shrinkage,
                        starts=starts,
                        seed=base_seed + 200_000 + draw,
                    )
                    uncertainty = hessian_uncertainty(fit, noisy, TIMEPOINTS)
                    width = fit.weights.shape[1]
                    weight_variance = np.maximum(np.diag(uncertainty.joint_weight_covariance), 0).reshape(fit.weights.shape)
                    weight_covered = np.abs(fit.weights - oracle_fit.weights) <= 1.96 * np.sqrt(weight_variance)
                    curve_variance = np.maximum(np.diag(uncertainty.curve_covariance), 0).reshape(noisy.shape)
                    curve_covered = np.abs(fit.predicted - target_curves) <= 1.96 * np.sqrt(curve_variance)
                    fit_positive, _ = project_curves_to_rates(
                        target_curves, TIMEPOINTS, fit.rates, starts=max(1, starts), seed=base_seed + draw
                    )
                    fit_tri, _ = project_curves_to_rates(
                        tri_curves, TIMEPOINTS, fit.rates, starts=max(1, starts), seed=base_seed + draw + 1
                    )
                    positive_distance = _candidate_distance(
                        fit.weights, fit_positive, uncertainty.joint_weight_covariance
                    )
                    tri_distance = _candidate_distance(
                        fit.weights, fit_tri, uncertainty.joint_weight_covariance
                    )
                    row = {
                        "row_type": "calibration",
                        "system": "ISO",
                        "panel": panel,
                        "semantics": semantics,
                        "sigma": sigma,
                        "seed": draw,
                        "rmse_to_clean": float(np.sqrt(np.mean((fit.predicted - target_curves) ** 2))),
                        "bound_fraction": bound_fraction,
                        "weight_coverage": float(np.mean(weight_covered)),
                        "curve_coverage": float(np.mean(curve_covered)),
                        "hessian_condition": uncertainty.condition_number,
                        "hessian_valid": uncertainty.valid,
                        "positive_distance": positive_distance,
                        "tri_distance": tri_distance,
                        "positive_beats_tri": positive_distance < tri_distance,
                    }
                    sample_rows.append(row)
                    condition_rows.append(row)
                valid_physical = np.mean([r["bound_fraction"] for r in condition_rows]) <= 0.01
                excluded = bool(sigma > 0 and not valid_physical)
                metrics = {
                    "mean_rmse_to_clean": float(
                        np.mean([r["rmse_to_clean"] for r in condition_rows])
                    ),
                    "mean_bound_fraction": float(
                        np.mean([r["bound_fraction"] for r in condition_rows])
                    ),
                    "mean_weight_coverage": float(
                        np.mean([r["weight_coverage"] for r in condition_rows])
                    ),
                    "mean_curve_coverage": float(
                        np.mean([r["curve_coverage"] for r in condition_rows])
                    ),
                    "invalid_hessian_fraction": float(
                        np.mean([not r["hessian_valid"] for r in condition_rows])
                    ),
                    "positive_beats_tri_fraction": float(
                        np.mean([r["positive_beats_tri"] for r in condition_rows])
                    ),
                }
                if sigma == 0:
                    unit_pass = condition_rows[0]["rmse_to_clean"] <= 0.005
                    criteria = {"noiseless_reconstruction": bool(unit_pass)}
                elif excluded:
                    unit_pass = False
                    criteria = {"physical_bounds": False}
                else:
                    criteria = {
                        "physical_bounds": valid_physical,
                        "reconstruction": metrics["mean_rmse_to_clean"] <= 1.25 * sigma,
                        "weight_coverage": 0.80 <= metrics["mean_weight_coverage"] <= 0.99,
                        "curve_coverage": 0.80 <= metrics["mean_curve_coverage"] <= 0.99,
                        "hessian_validity": metrics["invalid_hessian_fraction"] <= 0.05,
                        "bi_beats_tri": metrics["positive_beats_tri_fraction"] >= 0.90,
                    }
                    unit_pass = bool(all(criteria.values()))
                all_gate_units.append(
                    {
                        "panel": panel,
                        "semantics": semantics,
                        "sigma": sigma,
                        "physically_valid": valid_physical,
                        "excluded_from_coverage": excluded,
                        "passed": bool(unit_pass),
                        **metrics,
                        "criteria": criteria,
                    }
                )

    included_units = [unit for unit in all_gate_units if not unit["excluded_from_coverage"]]
    has_valid_noise = all(
        any(
            unit["sigma"] > 0 and not unit["excluded_from_coverage"]
            for unit in included_units
            if unit["panel"] == panel and unit["semantics"] == semantics
        )
        for panel in panels
        for semantics in ("average_first", "frame_mixture")
    )
    gates = {
        "units": all_gate_units,
        "all_units_pass": bool(
            has_valid_noise and included_units and all(unit["passed"] for unit in included_units)
        ),
        "has_valid_noise_condition_per_panel": has_valid_noise,
        "smoke": smoke,
    }
    return pd.DataFrame(fit_rows), pd.DataFrame(sample_rows), gates, arrays


def load_moprp() -> dict[str, Any]:
    base = REPO / "examples/2_CrossValidation"
    paths = {
        "dfrac": base / "data/_MoPrP/_output/MoPrP_dfrac.dat",
        "segments": base / "data/_MoPrP/_output/MoPrP_segments.txt",
    }
    read = dict(sep=r"\s+", comment="#", header=None)
    dfrac = pd.read_csv(paths["dfrac"], **read).to_numpy(float)
    segments = pd.read_csv(paths["segments"], **read).to_numpy(int)
    ensembles = {
        name: {
            "features": base / f"fitting/jaxENT/_featurise/features_{stem}.npz",
            "topology": base / f"fitting/jaxENT/_featurise/topology_{stem}.json",
        }
        for name, stem in (("AF2_MSAss", "AF2_MSAss"), ("AF2_filtered", "AF2_filtered"))
    }
    mapping = MOPRP_CANARY.build_sparse_map(segments, dfrac, ensembles["AF2_MSAss"])
    return {"paths": paths, "dfrac": dfrac, "segments": segments, "ensembles": ensembles, "mapping": mapping}


def trajectory_curves(moprp: dict[str, Any], name: str, mapping: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(moprp["ensembles"][name]["features"]) as data:
        log_pf = BV_BC * np.asarray(data["heavy_contacts"], float) + BV_BH * np.asarray(
            data["acceptor_contacts"], float
        )
        k_int = np.asarray(data["k_ints"], float)
    average_first_rate = k_int * np.exp(-np.mean(log_pf, axis=1))
    frame_rates = effective_rates(log_pf, k_int)
    return {
        "average_first": peptide_curves_from_rates(average_first_rate, mapping),
        "frame_mixture": peptide_curves_from_rates(frame_rates, mapping),
        "log_pf_mean": np.mean(log_pf, axis=1),
        "log_pf_variance": np.var(log_pf, axis=1),
    }


def run_moprp(
    moprp: dict[str, Any],
    *,
    components: tuple[int, ...],
    shrinkages: tuple[float, ...],
    starts: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], RateMixtureFit, pd.DataFrame]:
    curves = np.asarray(moprp["dfrac"], float)
    subset_definitions = {"all_14": slice(None), "production_12": slice(0, 12)}
    fitted_subsets: dict[str, tuple[RateMixtureFit, Any, np.ndarray]] = {}
    arrays: dict[str, np.ndarray] = {}
    fit_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for subset_index, (subset_name, subset) in enumerate(subset_definitions.items()):
        subset_curves = curves[subset]
        fit, subset_selection = _fit_selected(
            subset_curves, components, shrinkages, starts, 41729 + 10_000 * subset_index
        )
        uncertainty = hessian_uncertainty(fit, subset_curves, TIMEPOINTS)
        heterogeneity = peptide_score_covariances(fit.weights)
        fitted_subsets[subset_name] = (fit, uncertainty, subset_curves)
        prefix = f"moprp__{subset_name}"
        arrays.update(
            {
                f"{prefix}__rates": fit.rates,
                f"{prefix}__weights": fit.weights,
                f"{prefix}__predicted": fit.predicted,
                f"{prefix}__joint_weight_covariance": uncertainty.joint_weight_covariance,
                f"{prefix}__conditional_weight_covariance": uncertainty.conditional_weight_covariance,
                f"{prefix}__rate_covariance": uncertainty.rate_covariance,
                f"{prefix}__curve_covariance": uncertainty.curve_covariance,
                f"{prefix}__score_heterogeneity_empirical": heterogeneity["empirical_population"],
                f"{prefix}__score_heterogeneity_ledoit_wolf": heterogeneity["ledoit_wolf"],
            }
        )
        fit_rows.append(
            {
                "system": "MoPrP",
                "peptide_subset": subset_name,
                "n_components": fit.n_components,
                "shrinkage": fit.shrinkage,
                "rmse": fit.rmse,
                "hessian_condition": uncertainty.condition_number,
                "hessian_valid": uncertainty.valid,
            }
        )
        for row in subset_selection:
            selection_rows.append({"peptide_subset": subset_name, **row})

    comparison_rows: list[dict[str, Any]] = []
    mapping = np.asarray(moprp["mapping"], float)
    predictions_by_ensemble = {
        ensemble: trajectory_curves(moprp, ensemble, mapping)
        for ensemble in ("AF2_filtered", "AF2_MSAss")
    }
    for subset_name, subset in subset_definitions.items():
        fit, uncertainty, experimental_curves = fitted_subsets[subset_name]
        target_weights = fit.weights
        covariance = uncertainty.joint_weight_covariance
        for ensemble in ("AF2_filtered", "AF2_MSAss"):
            predictions = predictions_by_ensemble[ensemble]
            for semantics in ("average_first", "frame_mixture"):
                candidate_curves = predictions[semantics][subset]
                candidate_weights, projected = project_curves_to_rates(
                    candidate_curves, TIMEPOINTS, fit.rates, starts=max(2, starts), seed=9917
                )
                comparison_rows.append(
                    {
                        "ensemble": ensemble,
                        "semantics": semantics,
                        "peptide_subset": subset_name,
                        "raw_curve_rmse": float(np.sqrt(np.mean((candidate_curves - experimental_curves) ** 2))),
                        "basis_projection_rmse": float(np.sqrt(np.mean((projected - candidate_curves) ** 2))),
                        "score_mahalanobis": _candidate_distance(target_weights, candidate_weights, covariance),
                        "mean_log_rate": float(np.sum(candidate_weights[:, :-1] * np.log(fit.rates)[None, :]) / np.sum(candidate_weights[:, :-1])),
                    }
                )
                arrays[f"moprp__{subset_name}__{ensemble}__{semantics}__weights"] = candidate_weights
                arrays[f"moprp__{subset_name}__{ensemble}__{semantics}__curves"] = candidate_curves
    primary_fit = fitted_subsets["all_14"][0]
    return pd.DataFrame(fit_rows), pd.DataFrame(selection_rows), arrays, primary_fit, pd.DataFrame(comparison_rows)


def write_plots(output_dir: Path, moprp: dict[str, Any], fit: RateMixtureFit, comparisons: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for peptide in range(len(fit.predicted)):
        axes[0].plot(TIMEPOINTS, moprp["dfrac"][peptide], "o", ms=3, alpha=0.65)
        axes[0].plot(TIMEPOINTS, fit.predicted[peptide], "-", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("MoPrP shared-rate fits")
    axes[0].set_xlabel("time (min)")
    axes[0].set_ylabel("fractional uptake")
    primary = comparisons[comparisons.peptide_subset == "all_14"]
    labels = [f"{r.ensemble}\n{r.semantics}" for r in primary.itertuples(index=False)]
    x = np.arange(len(primary))
    axes[1].bar(x - 0.18, primary.raw_curve_rmse, 0.36, label="curve RMSE")
    scaled = primary.score_mahalanobis / max(primary.score_mahalanobis.max(), 1e-12)
    axes[1].bar(x + 0.18, scaled, 0.36, label="score distance (scaled)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_title("Trajectory diagnostics")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "rate_mixture_diagnostics.png", dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    components = (
        tuple(args.model_components)
        if args.model_components is not None
        else (1, 2)
        if args.smoke
        else (1, 2, 3)
    )
    if any(component < 1 for component in components):
        raise ValueError("model components must be positive")
    capacity_components = (
        tuple(args.capacity_components)
        if args.capacity_components is not None
        else components
    )
    if any(component < 1 for component in capacity_components):
        raise ValueError("capacity components must be positive")
    exploratory_capacity = any(component > 3 for component in capacity_components)
    shrinkages = (0.0, 1e-2) if args.smoke else DEFAULT_SHRINKAGES
    if args.smoke or args.stage == "pilot":
        noise_levels = (0.01,)
    else:
        noise_levels = DEFAULT_NOISE
    if args.smoke:
        seeds = min(args.seeds, 2)
    elif args.stage == "pilot":
        seeds = min(args.seeds, 10)
    else:
        seeds = args.seeds
    starts = min(args.starts, 2) if args.smoke else args.starts

    iso = load_iso_inputs(args.iso_results_dir)
    if args.stage == "stability":
        stability_rows, stability_summary, stability_arrays = run_iso_stability(
            iso,
            n_components=args.stability_components,
            repeats=args.stability_repeats,
            maxiter=args.stability_maxiter,
            smoke=args.smoke,
        )
        stability_rows.to_csv(args.output_dir / "iso_stability_fits.csv", index=False)
        stability_summary.to_csv(args.output_dir / "iso_stability_summary.csv", index=False)
        np.savez_compressed(args.output_dir / "stability_arrays.npz", **stability_arrays)
        all_stable = bool(len(stability_summary) and stability_summary.stable.all())
        decision = "parameter_stability_passed" if all_stable else "failed_parameter_stability"
        payload = {
            "decision": decision,
            "requested_stage": args.stage,
            "n_components": args.stability_components,
            "all_units_stable": all_stable,
            "units": stability_summary.to_dict("records"),
            "next_stage_authorized": all_stable,
            "moprp_interpretation_performed": False,
        }
        (args.output_dir / "decision.json").write_text(
            json.dumps(payload, indent=2, default=_json_value) + "\n"
        )
        lines = [
            "# Shared-rate parameter stability audit",
            "",
            f"Status: **{decision}**.",
            "",
            "Near-optimal means RMSE within max(1e-4, 10% of the best run). Stability requires",
            "at least half the runs, max log-rate SD <= 0.10, mean score SD <= 0.02, mean",
            "prediction SD <= 0.001, median flattened-score correlation >= 0.95, and at",
            "least 90% convergence among near-optimal fits. A fit is converged when the",
            f"optimizer succeeds or its final gradient norm is <= {STABILITY_GRADIENT_GATE:g}.",
            "",
            "| Panel | Semantics | Near optimal | Converged | Best RMSE | Max log-rate SD | Mean score SD | Prediction SD | Score corr. | Stable |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in stability_summary.itertuples(index=False):
            lines.append(
                f"| {row.panel} | {row.semantics} | {row.near_optimal_count}/{row.repeats} | "
                f"{row.near_optimal_converged_fraction:.0%} | "
                f"{row.best_rmse:.6f} | {row.max_rate_log_sd:.3f} | {row.mean_weight_sd:.3f} | "
                f"{row.mean_prediction_sd:.5f} | {row.median_weight_correlation:.3f} | "
                f"{'yes' if row.stable else 'no'} |"
            )
        (args.output_dir / "stability_report.md").write_text("\n".join(lines) + "\n")
        manifest = {
            "stage": "shared_rate_mixture_stability",
            "status": decision,
            "smoke": args.smoke,
            "n_components": args.stability_components,
            "repeats": min(args.stability_repeats, 3) if args.smoke else args.stability_repeats,
            "maxiter": args.stability_maxiter,
            "inputs": iso["inputs"],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return

    capacity_frame, capacity_gates, capacity_arrays = run_iso_capacity(
        iso, components=capacity_components, starts=starts, smoke=args.smoke
    )
    capacity_frame.to_csv(args.output_dir / "iso_capacity.csv", index=False)
    (args.output_dir / "capacity_gates.json").write_text(
        json.dumps(capacity_gates, indent=2, default=_json_value) + "\n"
    )
    np.savez_compressed(args.output_dir / "capacity_arrays.npz", **capacity_arrays)
    capacity_lines = [
        "# Shared-rate ISO capacity audit",
        "",
        f"Registered RMSE gate: `{CAPACITY_RMSE_GATE}`.",
        "",
        "| Panel | Semantics | Best K | Best RMSE | Jacobian rank | Condition | Pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for unit in capacity_gates["units"]:
        capacity_lines.append(
            f"| {unit['panel']} | {unit['semantics']} | {unit['best_components']} | "
            f"{unit['best_rmse']:.6f} | {unit['jacobian_rank']}/{unit['n_parameters']} | "
            f"{unit['jacobian_condition']:.3g} | {'yes' if unit['passed'] else 'no'} |"
        )
    capacity_lines += [
        "",
        f"All units passed: **{capacity_gates['all_units_pass']}**.",
        "",
        f"This audit measures the best unregularized approximation over K={list(capacity_components)}.",
        "Components above three are exploratory and never auto-authorize inference. Failure of",
        "the registered gate stops noisy Hessian calibration and real-data interpretation.",
    ]
    (args.output_dir / "capacity_report.md").write_text("\n".join(capacity_lines) + "\n")

    stop_after_capacity = args.stage == "capacity" or (
        not capacity_gates["all_units_pass"] and not args.smoke
    )
    if stop_after_capacity:
        decision = (
            "integration_smoke_capacity_only"
            if args.smoke
            else "exploratory_capacity_complete"
            if exploratory_capacity
            else "capacity_passed"
            if capacity_gates["all_units_pass"]
            else "failed_noiseless_capacity"
        )
        payload = {
            "decision": decision,
            "requested_stage": args.stage,
            "capacity_gates": capacity_gates,
            "exploratory_capacity": exploratory_capacity,
            "next_stage_authorized": bool(
                capacity_gates["all_units_pass"] and not exploratory_capacity
            ),
            "noisy_calibration_performed": False,
            "moprp_interpretation_performed": False,
        }
        (args.output_dir / "decision.json").write_text(
            json.dumps(payload, indent=2, default=_json_value) + "\n"
        )
        manifest = {
            "stage": "shared_rate_mixture_capacity",
            "requested_stage": args.stage,
            "status": decision,
            "smoke": args.smoke,
            "timepoints": TIMEPOINTS.tolist(),
            "components": list(capacity_components),
            "starts": starts,
            "rmse_gate": CAPACITY_RMSE_GATE,
            "inputs": iso["inputs"],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return

    iso_fits, iso_samples, iso_gates, iso_arrays = run_iso_calibration(
        iso,
        components=components,
        shrinkages=shrinkages,
        noise_levels=noise_levels,
        seeds=seeds,
        starts=starts,
        smoke=args.smoke,
    )
    iso_fits.to_csv(args.output_dir / "iso_model_fits.csv", index=False)
    iso_samples.to_csv(args.output_dir / "iso_calibration_samples.csv", index=False)
    np.savez_compressed(
        args.output_dir / "iso_calibration_arrays.npz", **capacity_arrays, **iso_arrays
    )
    if not iso_gates["all_units_pass"] and not args.smoke:
        decision = "failed_synthetic_calibration"
        payload = {
            "decision": decision,
            "requested_stage": args.stage,
            "capacity_gates": capacity_gates,
            "iso_gates": iso_gates,
            "moprp_interpretation_performed": False,
        }
        (args.output_dir / "decision.json").write_text(
            json.dumps(payload, indent=2, default=_json_value) + "\n"
        )
        (args.output_dir / "report.md").write_text(
            "# Shared-rate HDX mixture diagnostic\n\n"
            "Status: **failed_synthetic_calibration**.\n\n"
            "Noiseless capacity passed, but the registered noisy reconstruction, coverage, "
            "conditioning, physical-bound, or BI-versus-TRI gates did not all pass. MoPrP "
            "interpretation was not run.\n"
        )
        manifest = {
            "stage": "shared_rate_mixture_calibration",
            "requested_stage": args.stage,
            "status": decision,
            "timepoints": TIMEPOINTS.tolist(),
            "components": list(components),
            "shrinkages": list(shrinkages),
            "noise_levels": list(noise_levels),
            "seeds": seeds,
            "starts": starts,
            "inputs": iso["inputs"],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return

    if args.stage == "pilot" and not args.smoke:
        decision = "pilot_calibration_passed"
        payload = {
            "decision": decision,
            "requested_stage": args.stage,
            "capacity_gates": capacity_gates,
            "iso_gates": iso_gates,
            "next_stage_authorized": True,
            "moprp_interpretation_performed": False,
        }
        (args.output_dir / "decision.json").write_text(
            json.dumps(payload, indent=2, default=_json_value) + "\n"
        )
        (args.output_dir / "report.md").write_text(
            "# Shared-rate HDX mixture pilot\n\n"
            "Status: **pilot_calibration_passed**.\n\n"
            "Noiseless capacity and the registered 10-seed sigma=0.01 pilot passed. The full "
            "100-seed calibration is authorized; MoPrP interpretation was not run.\n"
        )
        manifest = {
            "stage": "shared_rate_mixture_pilot",
            "requested_stage": args.stage,
            "status": decision,
            "timepoints": TIMEPOINTS.tolist(),
            "components": list(components),
            "shrinkages": list(shrinkages),
            "noise_levels": list(noise_levels),
            "seeds": seeds,
            "starts": starts,
            "inputs": iso["inputs"],
        }
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return

    moprp = load_moprp()
    moprp_fits, moprp_selection, moprp_arrays, moprp_fit, comparisons = run_moprp(
        moprp,
        components=components,
        shrinkages=shrinkages,
        starts=starts,
    )

    pd.concat((iso_fits, moprp_fits), ignore_index=True).to_csv(args.output_dir / "model_fits.csv", index=False)
    iso_samples.to_csv(args.output_dir / "iso_calibration_samples.csv", index=False)
    moprp_selection.to_csv(args.output_dir / "moprp_model_selection.csv", index=False)
    comparisons.to_csv(args.output_dir / "moprp_trajectory_comparison.csv", index=False)
    np.savez_compressed(
        args.output_dir / "rate_mixture_arrays.npz",
        **capacity_arrays,
        **iso_arrays,
        **moprp_arrays,
    )

    primary = comparisons[
        (comparisons.peptide_subset == "all_14") & (comparisons.semantics == "average_first")
    ].sort_values(["raw_curve_rmse", "score_mahalanobis"])
    curve_winner = primary.sort_values("raw_curve_rmse").iloc[0].ensemble
    score_winner = primary.sort_values("score_mahalanobis").iloc[0].ensemble
    if args.smoke:
        decision = "integration_smoke_only"
    elif not iso_gates["all_units_pass"]:
        decision = "failed_synthetic_calibration"
    elif curve_winner == score_winner:
        decision = "calibrated_diagnostic"
    else:
        decision = "real_data_inconclusive"
    decision_payload = {
        "decision": decision,
        "curve_winner": curve_winner,
        "score_winner": score_winner,
        "capacity_gates": capacity_gates,
        "iso_gates": iso_gates,
        "interpretation": {
            "fit_covariance": "penalized-Hessian model/residual uncertainty",
            "score_covariance": "spatial/kinetic heterogeneity across peptide locations",
            "not_inferred": "experimental conformational covariance or cross-amide coupling",
            "production_loss_added": False,
        },
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision_payload, indent=2, default=_json_value) + "\n")

    manifest = {
        "stage": "shared_rate_mixture_diagnostic",
        "requested_stage": args.stage,
        "status": decision,
        "smoke": args.smoke,
        "timepoints": TIMEPOINTS.tolist(),
        "components": list(components),
        "capacity_components": list(capacity_components),
        "shrinkages": list(shrinkages),
        "noise_levels": list(noise_levels),
        "seeds": seeds,
        "starts": starts,
        "rate_bounds": [0.1 / TIMEPOINTS.max(), 10 / TIMEPOINTS.min()],
        "inputs": {
            **iso["inputs"],
            "moprp_dfrac": {"path": str(moprp["paths"]["dfrac"]), "sha256": sha256(moprp["paths"]["dfrac"])},
            "moprp_segments": {"path": str(moprp["paths"]["segments"]), "sha256": sha256(moprp["paths"]["segments"])},
            **{
                f"{name}_{kind}": {"path": str(path), "sha256": sha256(path)}
                for name, paths in moprp["ensembles"].items()
                for kind, path in paths.items()
            },
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_plots(args.output_dir, moprp, moprp_fit, comparisons)

    lines = [
        "# Shared-rate HDX mixture diagnostic",
        "",
        f"Status: **{decision}**.",
        "",
        "The shared exponential basis is a low-dimensional peptide kinetic model. Its Hessian",
        "covariance measures local fit uncertainty, and covariance across peptide scores measures",
        "spatial/kinetic heterogeneity. Neither object is experimental conformational covariance.",
        "",
        "## Synthetic calibration",
        "",
        f"All noiseless capacity units passed: **{capacity_gates['all_units_pass']}**.",
        "",
        f"All registered ISO units passed: **{iso_gates['all_units_pass']}**.",
        "",
        "## MoPrP fit",
        "",
        f"Selected `{moprp_fit.n_components}` shared exchanging components with shrinkage `{moprp_fit.shrinkage:g}`; full-data RMSE `{moprp_fit.rmse:.4f}`.",
        "",
        "## Uniform-trajectory comparison",
        "",
        "| Ensemble | Semantics | Peptides | Curve RMSE | Score Mahalanobis |",
        "|---|---|---|---:|---:|",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.ensemble} | {row.semantics} | {row.peptide_subset} | {row.raw_curve_rmse:.4f} | {row.score_mahalanobis:.3f} |"
        )
    lines += [
        "",
        "No ensemble weights or BV coefficients were optimized, and no production loss was added.",
        "Trajectory log-PF remains explanatory only; the experiment does not invert peptide curves",
        "into residue PFs or cross-amide coupling.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iso-results-dir",
        type=Path,
        default=REPO / "examples/1_IsoValidation_OMass/fitting/jaxENT/_pf_peptide_moment_final",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "_hdx_rate_mixture_diagnostic",
    )
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument(
        "--model-components",
        type=int,
        nargs="+",
        default=None,
        help="Components used for calibration/model selection (for example, 6 for the authorized K=6 pilot).",
    )
    parser.add_argument(
        "--capacity-components",
        type=int,
        nargs="+",
        default=None,
        help="Exploratory capacity values; components above 3 never auto-authorize inference.",
    )
    parser.add_argument(
        "--stage",
        choices=("capacity", "stability", "pilot", "full"),
        default="capacity",
        help="Gate-aware stage. Capacity is noiseless; pilot uses 10 seeds at sigma=0.01.",
    )
    parser.add_argument("--stability-components", type=int, default=6)
    parser.add_argument("--stability-repeats", type=int, default=20)
    parser.add_argument("--stability-maxiter", type=int, default=3000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
