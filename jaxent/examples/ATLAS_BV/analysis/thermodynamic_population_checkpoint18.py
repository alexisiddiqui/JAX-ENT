"""Checkpoint 18: thermodynamic fixed-BV metrics versus MD KDE population changes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import logsumexp

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_systems
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    NEIGHBOUR_RANKS, PRIMARY_RANK, density_targets, global_scales, mass_metrics,
    scalar_scale, system_data, valid_table,
)
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import (
    finite_conformal_quantile, ordered_assignments,
)
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import interval_score
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


THERMODYNAMIC_METRICS = (
    "work_shape",
    "work_scale",
    "work_density_legacy_zq",
    "work_density_unnormalized_q",
    "work_density_normalized_q_over_z",
)
NOTEBOOK_GAS_CONSTANT_J_MOL_K = 8.31


def entropy_contributions(centered_absolute: np.ndarray, variant: str) -> np.ndarray:
    """Return -Pi log(Pi), with residues on axis 0 and frames on axis 1."""
    log_q = -np.asarray(centered_absolute, dtype=float)
    log_z = logsumexp(log_q, axis=0, keepdims=True)
    if variant == "legacy_zq":
        log_pi = log_z + log_q
    elif variant == "unnormalized_q":
        log_pi = log_q
    elif variant == "normalized_q_over_z":
        log_pi = log_q - log_z
    else:
        raise ValueError(f"unknown entropy variant: {variant}")
    pi = np.exp(log_pi)
    return -pi * log_pi


def thermodynamic_frame_features(z: np.ndarray) -> dict[str, np.ndarray]:
    """Frame features whose pairwise distances reproduce the notebook work formulas / RT."""
    means = np.mean(z, axis=0)
    centered_absolute = np.abs(z - means[None, :])
    return {
        "work_shape": centered_absolute,
        "work_scale": means,
        "work_density_legacy_zq": entropy_contributions(centered_absolute, "legacy_zq"),
        "work_density_unnormalized_q": entropy_contributions(centered_absolute, "unnormalized_q"),
        "work_density_normalized_q_over_z": entropy_contributions(centered_absolute, "normalized_q_over_z"),
    }


def thermodynamic_pair_features(z: np.ndarray, pairs: dict[int, pd.DataFrame]) -> dict[str, dict[int, np.ndarray]]:
    frames = thermodynamic_frame_features(z)
    output: dict[str, dict[int, np.ndarray]] = {name: {} for name in THERMODYNAMIC_METRICS}
    for replica, pair_frame in pairs.items():
        left = pair_frame.left_frame.to_numpy(); right = pair_frame.right_frame.to_numpy()
        output["work_scale"][replica] = np.abs(frames["work_scale"][left] - frames["work_scale"][right])
        for metric in THERMODYNAMIC_METRICS:
            if metric == "work_scale":
                continue
            values = frames[metric]
            output[metric][replica] = np.mean(np.abs(values[:, left] - values[:, right]), axis=0)
    return output


def fit_system(row: dict[str, str], config: dict, parts: Path) -> str:
    data = system_data(row, config)
    features = thermodynamic_pair_features(data["z"], data["pairs"])
    rt_kj_mol = NOTEBOOK_GAS_CONSTANT_J_MOL_K * config["protocol"]["temperature_k"] / 1000.0
    rows = []
    for fit_replica in (1, 2, 3):
        for rank in NEIGHBOUR_RANKS:
            targets, bandwidth = density_targets(data, fit_replica, rank)
            target = np.abs(targets[fit_replica])
            for metric in THERMODYNAMIC_METRICS:
                x = features[metric][fit_replica]
                alpha, numerator, denominator = scalar_scale(x, target, True)
                rows.append({
                    "system_id": data["system"], "fit_replica": fit_replica, "rank": rank,
                    "model": metric, "alpha": alpha, "numerator": numerator,
                    "denominator": denominator, "bandwidth_angstrom": bandwidth,
                    "n_residues": data["n_residues"], "rt_kj_mol": rt_kj_mol,
                    "feature_mean_dimensionless": float(np.mean(x)),
                    "feature_mean_kj_mol": float(rt_kj_mol * np.mean(x)),
                })
    atomic_parquet(pd.DataFrame(rows), parts / f"{data['system']}.fits.parquet")
    return data["system"]


def _finite_spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2:
        return 0.0
    value = float(pd.Series(target).corr(pd.Series(prediction), method="spearman"))
    return value if np.isfinite(value) else 0.0


def evaluate_system(row: dict[str, str], config: dict, fits: pd.DataFrame, global_fit: pd.DataFrame,
                    w1_edges: np.ndarray, rmsd_edges: np.ndarray, parts: Path) -> str:
    data = system_data(row, config)
    features = thermodynamic_pair_features(data["z"], data["pairs"])
    system_fits = fits[fits.system_id == data["system"]]
    system_global = global_fit[global_fit.system_id == data["system"]]
    settings = config["analysis"]["pairwise_geometry"]
    smoothing = settings["boundary_audit"]["distribution_smoothing"]
    bins = settings["boundary_audit"]["distribution_bins"]
    coverage = settings["strict_conformal"]["coverage"]
    rows = []
    for fit_replica, calibration_replica, test_replica in ordered_assignments():
        for rank in NEIGHBOUR_RANKS:
            signed_targets, bandwidth = density_targets(data, fit_replica, rank)
            targets = {replica: np.abs(values) for replica, values in signed_targets.items()}
            local = system_fits[(system_fits.fit_replica == fit_replica)
                                & (system_fits["rank"] == rank)].set_index("model")
            loso = system_global[(system_global.fit_replica == fit_replica)
                                 & (system_global["rank"] == rank)].set_index("model")
            predictions: dict[str, dict[int, np.ndarray]] = {}
            for metric in THERMODYNAMIC_METRICS:
                predictions[f"{metric}_alpha1"] = {r: features[metric][r] for r in (1, 2, 3)}
                predictions[f"{metric}_local_alpha"] = {
                    r: local.loc[metric, "alpha"] * features[metric][r] for r in (1, 2, 3)
                }
                predictions[f"{metric}_global_alpha"] = {
                    r: loso.loc[metric, "global_alpha_loso"] * features[metric][r] for r in (1, 2, 3)
                }
            test_pairs = data["pairs"][test_replica]
            support = data["audit"][(data["audit"].fit_replica == fit_replica)
                                    & (data["audit"].calibration_replica == calibration_replica)
                                    & (data["audit"].test_replica == test_replica)
                                    & (data["audit"].target == "w1")].reset_index(drop=True).support_category.to_numpy()
            axes = {"w1": (test_pairs.w1.to_numpy(), w1_edges)}
            if rank == PRIMARY_RANK:
                axes["rmsd"] = (test_pairs.rmsd.to_numpy(), rmsd_edges)
            for model, model_predictions in predictions.items():
                calibration_error = np.abs(targets[calibration_replica] - model_predictions[calibration_replica])
                q = finite_conformal_quantile(calibration_error, coverage)
                test_target = targets[test_replica]; test_prediction = model_predictions[test_replica]
                for axis, (structural, edges) in axes.items():
                    labels = np.clip(np.digitize(structural, edges[1:-1]), 0, len(edges) - 2)
                    strata = (("all", np.ones(len(labels), bool)),
                               ("common_support", support == "common_support")) if rank == PRIMARY_RANK else (
                                   ("common_support", support == "common_support"),)
                    for band in range(len(edges) - 1):
                        for stratum, stratum_mask in strata:
                            mask = (labels == band) & stratum_mask
                            if not mask.any():
                                continue
                            lower = test_prediction[mask] - q; upper = test_prediction[mask] + q
                            rows.append({
                                "system_id": data["system"], "fit_replica": fit_replica,
                                "calibration_replica": calibration_replica, "test_replica": test_replica,
                                "rank": rank, "bandwidth_angstrom": bandwidth, "model": model,
                                "target_kind": "magnitude", "structural_axis": axis, "band": f"q{band}",
                                "band_low_angstrom": float(edges[band]),
                                "band_high_angstrom": float(edges[band + 1]), "stratum": stratum,
                                "pairs": int(mask.sum()),
                                "mae": float(np.mean(np.abs(test_target[mask] - test_prediction[mask]))),
                                "spearman": _finite_spearman(test_target[mask], test_prediction[mask]),
                                "coverage_90": float(np.mean((test_target[mask] >= lower)
                                                              & (test_target[mask] <= upper))),
                                "mean_interval_score": float(np.mean(interval_score(test_target[mask], lower, upper))),
                                **mass_metrics(test_target[mask], test_prediction[mask], targets[fit_replica], bins, smoothing),
                            })
    atomic_parquet(pd.DataFrame(rows), parts / f"{data['system']}.summary.parquet")
    return data["system"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint18_thermodynamic_population"
    fit_parts = output / "fit_parts"; summary_parts = output / "summary_parts"
    fit_parts.mkdir(parents=True, exist_ok=True); summary_parts.mkdir(parents=True, exist_ok=True)
    fit_required = {"system_id", "fit_replica", "rank", "model", "alpha", "feature_mean_kj_mol"}
    pending = systems if args.restart else [r for r in systems if not valid_table(
        fit_parts / f"{r['system_id']}.fits.parquet", fit_required)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_system, row, config, fit_parts) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[fit {index}/{len(pending)}] {future.result()}", flush=True)
    fits = pd.concat([pd.read_parquet(fit_parts / f"{r['system_id']}.fits.parquet") for r in systems], ignore_index=True)
    global_fit = global_scales(fits)
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml") as handle:
        w1_edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint16_global_rmsd/global_rmsd_edges.yaml") as handle:
        rmsd_edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    summary_required = {"system_id", "fit_replica", "calibration_replica", "test_replica", "rank",
                        "model", "structural_axis", "band", "distribution_recovery"}
    pending = systems if args.restart else [r for r in systems if not valid_table(
        summary_parts / f"{r['system_id']}.summary.parquet", summary_required)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_system, row, config, fits, global_fit,
                                   w1_edges, rmsd_edges, summary_parts) for row in pending]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"[eval {index}/{len(pending)}] {future.result()}", flush=True)
    summary = pd.concat([pd.read_parquet(summary_parts / f"{r['system_id']}.summary.parquet") for r in systems], ignore_index=True)
    atomic_parquet(fits, output / "thermodynamic_population_fits.parquet")
    atomic_parquet(global_fit, output / "thermodynamic_population_global_scales.parquet")
    atomic_parquet(summary, output / "thermodynamic_population_assignment_summary.parquet")
    atomic_yaml(output / "checkpoint18_run.yaml", {
        "checkpoint": 18, "systems": len(systems), "status": "measurement_complete",
        "primary_neighbour_rank": PRIMARY_RANK, "gas_constant_j_mol_k": NOTEBOOK_GAS_CONSTANT_J_MOL_K,
        "temperature_k": config["protocol"]["temperature_k"],
        "predictor_units": "W/(RT), dimensionless",
    })


if __name__ == "__main__":
    main()
