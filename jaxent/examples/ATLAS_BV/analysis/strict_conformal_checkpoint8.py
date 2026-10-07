"""Checkpoint 8: strict six-way A-fit/B-calibrate/C-test conformal audit."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import permutations
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
import yaml
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors

from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates
from jaxent.examples.ATLAS_BV.analysis.boundary_checkpoint2 import (
    boundary_predictions, probability_distribution_errors,
)
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE, atomic_yaml, integrated_autocorrelation_frames, load_config,
    load_contact_coordinates, load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import (
    effective_endpoint_frames, intraframe_distance_distributions, quantile_signatures,
    target_bands, w1_pair_distance,
)
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import (
    _within_pair_set, absolute_change_vectors, deterministic_cap, feature_standardizer,
)
from jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c import exact_neighbors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import normalized_mae


MODELS = ("absolute_l1_isotonic", "raw_per_residue_ridge")
TARGETS = ("rmsd", "w1")


def ordered_assignments() -> tuple[tuple[int, int, int], ...]:
    return tuple(permutations((1, 2, 3), 3))


def finite_conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    """Split-conformal order statistic with the finite-sample rank correction."""
    values = np.sort(np.asarray(scores, dtype=float))
    if not len(values):
        raise ValueError("calibration scores must not be empty")
    rank = min(len(values), int(np.ceil((len(values) + 1) * coverage)))
    return float(values[rank - 1])


def mondrian_quantiles(
    fit_prediction: np.ndarray,
    calibration_prediction: np.ndarray,
    calibration_scores: np.ndarray,
    test_prediction: np.ndarray,
    bins: int,
    coverage: float,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A-defined prediction regions with B-only finite-sample conformal scores."""
    edges = np.unique(np.quantile(fit_prediction, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return np.full(len(test_prediction), fallback), np.zeros(len(test_prediction), dtype=int)
    calibration_labels = np.clip(
        np.digitize(calibration_prediction, edges[1:-1]), 0, len(edges) - 2
    )
    test_labels = np.clip(np.digitize(test_prediction, edges[1:-1]), 0, len(edges) - 2)
    output = np.full(len(test_prediction), fallback)
    for label in range(len(edges) - 1):
        mask = calibration_labels == label
        if mask.any():
            output[test_labels == label] = finite_conformal_quantile(
                calibration_scores[mask], coverage
            )
    return output, test_labels


def extrapolation_scale(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    """Fixed dimensionless interval inflation outside A's scalar-PF range."""
    low, high = float(train_values.min()), float(train_values.max())
    iqr = float(np.subtract(*np.quantile(train_values, [0.75, 0.25])))
    iqr = max(iqr, np.finfo(float).eps)
    excess = np.maximum(low - values, 0.0) + np.maximum(values - high, 0.0)
    return 1.0 + excess / iqr


def fit_ridge_a_only(
    features: np.ndarray,
    target: np.ndarray,
    alphas: list[float],
    tuning_fraction: float,
    seed: int,
) -> tuple[Ridge, float]:
    """Tune and refit ridge without accessing B or C."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    tune_count = max(1, int(round(tuning_fraction * len(order))))
    tune, fit = order[:tune_count], order[tune_count:]
    scored = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(features[fit], target[fit])
        scored.append((normalized_mae(target[tune], model.predict(features[tune]), target[fit]), alpha))
    _, alpha = min(scored, key=lambda item: (item[0], -item[1]))
    return Ridge(alpha=alpha).fit(features, target), float(alpha)


def scalar_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    other_x: np.ndarray,
    settings: dict,
) -> tuple[np.ndarray, np.ndarray]:
    fitted = boundary_predictions(
        train_x, train_y, other_x,
        settings["boundary_audit"]["tail_fraction"],
        settings["boundary_audit"]["tail_minimum_pairs"],
    )
    train_model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(train_x, train_y)
    return np.maximum(0.0, fitted["extrapolated_test"]), np.maximum(0.0, train_model.predict(train_x))


def frame_novelty(
    fit_representation: np.ndarray,
    calibration_representation: np.ndarray,
    test_representation: np.ndarray,
    quantile: float,
) -> tuple[np.ndarray, float]:
    """B-to-A support threshold and untouched C-to-A novelty distances."""
    neighbors = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(fit_representation)
    calibration_distance = neighbors.kneighbors(
        calibration_representation, return_distance=True
    )[0][:, 0]
    threshold = float(np.quantile(calibration_distance, quantile))
    test_distance = neighbors.kneighbors(test_representation, return_distance=True)[0][:, 0]
    return test_distance, threshold


def support_category(
    pf_in_range: np.ndarray,
    target_in_range: np.ndarray,
    structural_novel: np.ndarray,
    vector_oos: np.ndarray,
) -> np.ndarray:
    """Exclusive diagnostic hierarchy; overlapping flags remain separately persisted."""
    category = np.full(len(pf_in_range), "common_support", dtype=object)
    category[vector_oos] = "pf_vector_oos"
    category[~pf_in_range] = "pf_extrapolation"
    category[structural_novel | ~target_in_range] = "structurally_novel"
    return category


def summarize_strata(pair_frame: pd.DataFrame, settings: dict) -> pd.DataFrame:
    rows = []
    boundary = settings["boundary_audit"]
    strata = [("all", np.ones(len(pair_frame), dtype=bool))]
    strata.extend((name, pair_frame.support_category.to_numpy() == name) for name in (
        "common_support", "pf_extrapolation", "pf_vector_oos", "structurally_novel"
    ))
    for model in MODELS:
        prediction = pair_frame[f"{model}_prediction"].to_numpy()
        lower = pair_frame[f"{model}_lower"].to_numpy()
        upper = pair_frame[f"{model}_upper"].to_numpy()
        mondrian_lower = pair_frame[f"{model}_mondrian_lower"].to_numpy()
        mondrian_upper = pair_frame[f"{model}_mondrian_upper"].to_numpy()
        historical_lower = pair_frame[f"{model}_historical_lower"].to_numpy()
        historical_upper = pair_frame[f"{model}_historical_upper"].to_numpy()
        target = pair_frame.target_value.to_numpy()
        for band, band_frame in pair_frame.groupby("band", sort=False):
            band_mask = pair_frame.index.isin(band_frame.index)
            band_low = float(band_frame.band_low.iloc[0]); band_high = float(band_frame.band_high.iloc[0])
            fit_low = band_low if np.isfinite(band_low) else float(pair_frame.target_train_low.iloc[0])
            fit_high = band_high if np.isfinite(band_high) else float(pair_frame.target_train_high.iloc[0])
            for stratum, stratum_mask in strata:
                mask = band_mask & stratum_mask
                if not mask.any():
                    continue
                unique, effective = effective_endpoint_frames(
                    pair_frame.left_frame.to_numpy()[mask], pair_frame.right_frame.to_numpy()[mask]
                )
                if fit_high > fit_low:
                    errors = probability_distribution_errors(
                        prediction[mask], target[mask], fit_low, fit_high,
                        boundary["distribution_bins"], boundary["distribution_smoothing"],
                    )
                else:
                    errors = {
                        "distribution_l1": np.nan, "distribution_l2": np.nan,
                        "distribution_jsd": np.nan, "distribution_sqrt_jsd": np.nan,
                        "distribution_kld_target_to_prediction": np.nan,
                        "distribution_recovery": np.nan,
                    }
                rows.append({
                    "system_id": pair_frame.system_id.iloc[0],
                    "fit_replica": int(pair_frame.fit_replica.iloc[0]),
                    "calibration_replica": int(pair_frame.calibration_replica.iloc[0]),
                    "test_replica": int(pair_frame.test_replica.iloc[0]),
                    "target": pair_frame.target.iloc[0], "model": model,
                    "band": band, "stratum": stratum, "pairs": int(mask.sum()),
                    "unique_frames": unique, "effective_frames": effective,
                    "coverage_90": float(np.mean((target[mask] >= lower[mask]) & (target[mask] <= upper[mask]))),
                    "mondrian_coverage_90": float(np.mean(
                        (target[mask] >= mondrian_lower[mask]) & (target[mask] <= mondrian_upper[mask])
                    )),
                    "historical_coverage_90": float(np.mean(
                        (target[mask] >= historical_lower[mask]) & (target[mask] <= historical_upper[mask])
                    )),
                    "median_interval_width": float(np.median(upper[mask] - lower[mask])),
                    **errors,
                })
    return pd.DataFrame(rows)


def analyse_system_to_disk(row: dict[str, str], config: dict, parts: Path) -> str:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    strict = settings["strict_conformal"]
    coordinates, replicas, frames = load_ca_coordinates(row, config)
    universe = mda.Universe(HERE / row["pdb_path"])
    reference = universe.select_atoms(config["analysis"]["basins"]["atom_selection"]).positions.copy()
    aligned = coordinates.copy()
    # Reuse the established Kabsch implementation through the public helper.
    from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import align_to_structure
    aligned = align_to_structure(coordinates, reference)
    distributions = intraframe_distance_distributions(coordinates)
    signatures = quantile_signatures(distributions, settings["support_audit"]["w1_support_quantiles"])
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    centered_reference = reference - reference.mean(axis=0)
    rmsd_start = np.sqrt(np.mean(np.sum((aligned - centered_reference) ** 2, axis=2), axis=1))
    global_pf = z.sum(axis=0)
    theiler = {}
    for replica in (1, 2, 3):
        mask = replicas == replica
        theiler[replica] = max(
            integrated_autocorrelation_frames(rmsd_start[mask]),
            integrated_autocorrelation_frames(global_pf[mask]),
        )
    pair_sets = {
        replica: _within_pair_set(
            aligned, replicas, frames, replica, theiler[replica], strict["pairs_per_replica"],
            config["analysis"]["seed"] + 8009 * replica,
        ) for replica in (1, 2, 3)
    }
    pair_vectors = {
        replica: absolute_change_vectors(z, pairs.left, pairs.right)
        for replica, pairs in pair_sets.items()
    }
    w1_targets = {
        replica: w1_pair_distance(
            distributions, pairs.left, pairs.right,
            settings["support_audit"]["w1_max_chunk_values"],
        ) for replica, pairs in pair_sets.items()
    }
    frame_representations = {
        "rmsd": aligned.reshape(len(aligned), -1) / np.sqrt(aligned.shape[1]),
        "w1": signatures,
    }
    audit_chunks, summary_chunks, hyperparameters = [], [], []
    for assignment_index, (fit_replica, calibration_replica, test_replica) in enumerate(ordered_assignments()):
        fit_pairs = pair_sets[fit_replica]; calibration_pairs = pair_sets[calibration_replica]
        test_pairs = pair_sets[test_replica]
        fit_vectors = pair_vectors[fit_replica]; calibration_vectors = pair_vectors[calibration_replica]
        test_vectors = pair_vectors[test_replica]
        feature_mean, feature_sigma, floored = feature_standardizer(fit_vectors, settings["sigma_floor"])
        fit_standardized = (fit_vectors - feature_mean) / feature_sigma
        calibration_standardized = (calibration_vectors - feature_mean) / feature_sigma
        test_standardized = (test_vectors - feature_mean) / feature_sigma
        support_take = deterministic_cap(
            len(fit_standardized), strict["pf_support_vectors"],
            config["analysis"]["seed"] + 9001 * assignment_index,
        )
        calibration_nn = exact_neighbors(fit_standardized[support_take], calibration_standardized, 1)[0][:, 0]
        test_nn = exact_neighbors(fit_standardized[support_take], test_standardized, 1)[0][:, 0]
        calibration_nn /= np.sqrt(fit_standardized.shape[1]); test_nn /= np.sqrt(fit_standardized.shape[1])
        vector_threshold = float(np.quantile(calibration_nn, strict["novelty_quantile"]))
        scalar = {
            replica: vectors.mean(axis=1) for replica, vectors in pair_vectors.items()
        }
        for target_name in TARGETS:
            targets = {
                replica: pair_sets[replica].rmsd if target_name == "rmsd" else w1_targets[replica]
                for replica in (1, 2, 3)
            }
            fit_target = targets[fit_replica]; calibration_target = targets[calibration_replica]
            test_target = targets[test_replica]
            scalar_calibration_prediction, scalar_fit_prediction = scalar_predictions(
                scalar[fit_replica], fit_target, scalar[calibration_replica], settings
            )
            scalar_test_prediction, _ = scalar_predictions(
                scalar[fit_replica], fit_target, scalar[test_replica], settings
            )
            ridge_model, alpha = fit_ridge_a_only(
                fit_vectors, fit_target, settings["vector_audit"]["ridge_alphas"],
                strict["ridge_tuning_fraction"],
                config["analysis"]["seed"] + 11003 * assignment_index + (0 if target_name == "rmsd" else 1),
            )
            ridge_fit_prediction = np.maximum(0.0, ridge_model.predict(fit_vectors))
            ridge_calibration_prediction = np.maximum(0.0, ridge_model.predict(calibration_vectors))
            ridge_test_prediction = np.maximum(0.0, ridge_model.predict(test_vectors))
            predictions = {
                "absolute_l1_isotonic": (
                    scalar_fit_prediction, scalar_calibration_prediction, scalar_test_prediction,
                    extrapolation_scale(scalar[calibration_replica], scalar[fit_replica]),
                    extrapolation_scale(scalar[test_replica], scalar[fit_replica]),
                ),
                "raw_per_residue_ridge": (
                    ridge_fit_prediction, ridge_calibration_prediction, ridge_test_prediction,
                    np.ones(len(calibration_target)), np.ones(len(test_target)),
                ),
            }
            output = {
                "system_id": system, "fit_replica": fit_replica,
                "calibration_replica": calibration_replica, "test_replica": test_replica,
                "target": target_name,
                "left_frame": test_pairs.left, "right_frame": test_pairs.right,
                "target_value": test_target,
                "target_train_low": float(fit_target.min()),
                "target_train_high": float(fit_target.max()),
                "scalar_pf_distance": scalar[test_replica],
                "pf_in_range": (scalar[test_replica] >= scalar[fit_replica].min())
                & (scalar[test_replica] <= scalar[fit_replica].max()),
                "target_in_range": (test_target >= fit_target.min()) & (test_target <= fit_target.max()),
                "nearest_pf_vector_distance": test_nn,
                "pf_vector_oos": test_nn > vector_threshold,
            }
            fit_frame_mask = replicas == fit_replica; test_frame_mask = replicas == test_replica
            calibration_frame_mask = replicas == calibration_replica
            frame_distance, frame_threshold = frame_novelty(
                frame_representations[target_name][fit_frame_mask],
                frame_representations[target_name][calibration_frame_mask],
                frame_representations[target_name][test_frame_mask], strict["novelty_quantile"],
            )
            fit_indices = np.flatnonzero(fit_frame_mask); test_indices = np.flatnonzero(test_frame_mask)
            distance_by_global = np.full(len(replicas), np.nan)
            distance_by_global[test_indices] = frame_distance
            pair_endpoint_distance = np.maximum(
                distance_by_global[test_pairs.left], distance_by_global[test_pairs.right]
            )
            output["endpoint_novelty_distance"] = pair_endpoint_distance
            output["structural_novel"] = pair_endpoint_distance > frame_threshold
            output["support_category"] = support_category(
                output["pf_in_range"], output["target_in_range"],
                output["structural_novel"], output["pf_vector_oos"],
            )
            for band, mask, low, high in target_bands(target_name, fit_target, test_target):
                output.setdefault("band", np.empty(len(test_target), dtype=object))[mask] = band
                output.setdefault("band_low", np.empty(len(test_target)))[mask] = low
                output.setdefault("band_high", np.empty(len(test_target)))[mask] = high
            for model_name, (fit_prediction, calibration_prediction, test_prediction,
                             calibration_scale, test_scale) in predictions.items():
                scores = np.abs(calibration_target - calibration_prediction) / calibration_scale
                marginal_q = finite_conformal_quantile(scores, strict["coverage"])
                mondrian_q, _ = mondrian_quantiles(
                    fit_prediction, calibration_prediction, scores, test_prediction,
                    strict["mondrian_bins"], strict["coverage"], marginal_q,
                )
                historical_q = finite_conformal_quantile(
                    np.abs(fit_target - fit_prediction), strict["coverage"]
                )
                output[f"{model_name}_prediction"] = test_prediction
                output[f"{model_name}_lower"] = np.maximum(0.0, test_prediction - marginal_q * test_scale)
                output[f"{model_name}_upper"] = test_prediction + marginal_q * test_scale
                output[f"{model_name}_mondrian_lower"] = np.maximum(
                    0.0, test_prediction - mondrian_q * test_scale
                )
                output[f"{model_name}_mondrian_upper"] = test_prediction + mondrian_q * test_scale
                output[f"{model_name}_historical_lower"] = np.maximum(
                    0.0, test_prediction - historical_q * test_scale
                )
                output[f"{model_name}_historical_upper"] = test_prediction + historical_q * test_scale
            pair_frame = pd.DataFrame(output)
            audit_chunks.append(pair_frame)
            summary_chunks.append(summarize_strata(pair_frame, settings))
            hyperparameters.append({
                "system_id": system, "fit_replica": fit_replica,
                "calibration_replica": calibration_replica, "test_replica": test_replica,
                "target": target_name, "ridge_alpha": alpha,
                "floored_residues": floored, "vector_support_threshold": vector_threshold,
                "structural_novelty_threshold": frame_threshold,
                "fit_pairs": len(fit_target), "calibration_pairs": len(calibration_target),
                "test_pairs": len(test_target),
            })
    atomic_parquet(pd.concat(audit_chunks, ignore_index=True), parts / f"{system}.pairs.parquet")
    atomic_parquet(pd.concat(summary_chunks, ignore_index=True), parts / f"{system}.summary.parquet")
    atomic_parquet(pd.DataFrame(hyperparameters), parts / f"{system}.hyperparameters.parquet")
    return system


def valid_system_checkpoint(parts: Path, system: str) -> bool:
    paths = [parts / f"{system}.{suffix}.parquet" for suffix in ("pairs", "summary", "hyperparameters")]
    if not all(path.exists() for path in paths):
        return False
    try:
        hyper = pd.read_parquet(paths[2], columns=[
            "system_id", "fit_replica", "calibration_replica", "test_replica", "target"
        ])
    except Exception:
        return False
    return (
        set(hyper.system_id) == {system} and len(hyper) == 12
        and set(map(tuple, hyper[["fit_replica", "calibration_replica", "test_replica"]].drop_duplicates().to_numpy()))
        == set(ordered_assignments()) and set(hyper.target) == set(TARGETS)
    )


def aggregate_population(summary: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    system = summary.groupby(["system_id", "target", "model", "band", "stratum"], as_index=False).agg(
        coverage_90=("coverage_90", "mean"),
        mondrian_coverage_90=("mondrian_coverage_90", "mean"),
        historical_coverage_90=("historical_coverage_90", "mean"),
        median_interval_width=("median_interval_width", "median"),
        distribution_recovery=("distribution_recovery", "mean"),
        effective_frames=("effective_frames", "median"),
        pairs=("pairs", "sum"),
    )
    population = system.groupby(["target", "model", "band", "stratum"], as_index=False).median(numeric_only=True)
    gate_rows = system[
        (system.target == "w1") & (system.model == "raw_per_residue_ridge")
        & (system.band == "q5") & (system.stratum == "common_support")
    ]
    coverage = float(gate_rows.coverage_90.median()) if len(gate_rows) else np.nan
    historical = float(gate_rows.historical_coverage_90.median()) if len(gate_rows) else np.nan
    effective = float(gate_rows.effective_frames.median()) if len(gate_rows) else np.nan
    rng = np.random.default_rng(config["analysis"]["seed"] + 8080)
    values = gate_rows.coverage_90.to_numpy()
    draws = np.median(rng.choice(values, (10000, len(values)), replace=True), axis=1) if len(values) else np.array([np.nan])
    ci = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
    gap_closed = (
        (coverage - historical) / (0.90 - historical)
        if np.isfinite(coverage) and np.isfinite(historical) and historical < 0.90 else np.nan
    )
    if coverage >= 0.85 and ci[0] <= 0.90 <= ci[1] and gap_closed >= 0.5:
        verdict = "calibration_dominated"
    elif coverage < 0.80 and ci[1] < 0.85 and effective >= 100:
        verdict = "representation_limited"
    else:
        verdict = "mixed"
    report = {
        "checkpoint": "8", "status": "complete",
        "systems": int(summary.system_id.nunique()), "assignments_per_system": 6,
        "pairs_per_role": config["analysis"]["pairwise_geometry"]["strict_conformal"]["pairs_per_replica"],
        "gate": {
            "target": "w1", "band": "q5", "model": "raw_per_residue_ridge",
            "stratum": "common_support", "coverage": coverage, "coverage_ci95": ci,
            "historical_coverage": historical, "miscoverage_gap_closed": gap_closed,
            "median_effective_frames": effective, "verdict": verdict,
        },
    }
    return population, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true"); args = parser.parse_args()
    config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_system_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed: print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system_to_disk, row, config, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            system = future.result()
            print(f"[{resumed + index}/{len(systems)}] {system} strict conformal checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_system_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing checkpoints: {missing}")
    summary = pd.concat([
        pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems
    ], ignore_index=True)
    hyper = pd.concat([
        pd.read_parquet(parts / f"{row['system_id']}.hyperparameters.parquet") for row in systems
    ], ignore_index=True)
    population, report = aggregate_population(summary, config)
    summary.to_parquet(output / "strict_conformal_assignment_summary.parquet", index=False)
    hyper.to_parquet(output / "strict_conformal_hyperparameters.parquet", index=False)
    population.to_parquet(output / "strict_conformal_population.parquet", index=False)
    atomic_yaml(output / "checkpoint8_report.yaml", report)
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
