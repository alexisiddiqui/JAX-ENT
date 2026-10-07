"""Checkpoint 9: strict A-fit/B-calibrate/C-test Gaussian likelihood baseline."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import (
    finite_conformal_quantile,
    fit_ridge_a_only,
    mondrian_quantiles,
    ordered_assignments,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import effective_endpoint_frames
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import absolute_change_vectors, feature_standardizer
from jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c import probability_mass_errors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet, gaussian_nll


PREPROCESSING = ("raw", "zscore")
TARGETS = ("rmsd", "w1")
GAUSSIAN_Q90 = 1.6448536269514722


def interval_score(
    target: np.ndarray, lower: np.ndarray, upper: np.ndarray, miscoverage: float = 0.10
) -> np.ndarray:
    """Proper central prediction-interval score; lower is better."""
    target = np.asarray(target, dtype=float)
    score = upper - lower
    score = score + (2.0 / miscoverage) * np.maximum(lower - target, 0.0)
    return score + (2.0 / miscoverage) * np.maximum(target - upper, 0.0)


def gaussian_mass(
    mean: np.ndarray, sigma: np.ndarray, low: float, high: float, bins: int
) -> np.ndarray:
    """Average Gaussian probability mass on fixed finite structural-target bins."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), np.finfo(float).eps)
    edges = np.linspace(low, high, bins + 1)
    z = (edges[None, :] - np.asarray(mean)[:, None]) / sigma[:, None]
    mass = np.maximum(0.0, np.diff(ndtr(z), axis=1))
    total = mass.sum(axis=1, keepdims=True)
    valid = total[:, 0] > np.finfo(float).tiny
    mass[valid] /= total[valid]
    if (~valid).any():
        nearest = np.clip(np.digitize(np.asarray(mean)[~valid], edges[1:-1]), 0, bins - 1)
        mass[~valid] = 0.0
        mass[np.flatnonzero(~valid), nearest] = 1.0
    return mass.mean(axis=0)


def extended_mass_errors(
    predicted_mass: np.ndarray, target_mass: np.ndarray, smoothing: float
) -> dict[str, float]:
    """Dimensionless distribution errors, including angular/shape distances."""
    errors = probability_mass_errors(predicted_mass, target_mass, smoothing)
    predicted = np.asarray(predicted_mass, dtype=float) + smoothing
    target = np.asarray(target_mass, dtype=float) + smoothing
    predicted /= predicted.sum(); target /= target.sum()
    denominator = float(np.linalg.norm(predicted) * np.linalg.norm(target))
    cosine = float(np.dot(predicted, target) / denominator) if denominator else 0.0
    if np.std(predicted) <= np.finfo(float).eps or np.std(target) <= np.finfo(float).eps:
        correlation = 1.0 if np.allclose(predicted, target) else 0.0
    else:
        correlation = float(np.corrcoef(predicted, target)[0, 1])
    return {
        **errors,
        "distribution_cosine_distance": float(1.0 - np.clip(cosine, -1.0, 1.0)),
        "distribution_correlation_distance": float(1.0 - np.clip(correlation, -1.0, 1.0)),
    }


def _pair_sets_from_audit(audit: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Recover the role-invariant pair samples and both targets from Checkpoint 8."""
    result = {}
    for replica in (1, 2, 3):
        target_frames = {}
        for target in TARGETS:
            block = audit[(audit.test_replica == replica) & (audit.target == target)]
            first_fit = int(block.fit_replica.min())
            block = block[block.fit_replica == first_fit][
                ["left_frame", "right_frame", "target_value"]
            ].reset_index(drop=True)
            if block.duplicated(["left_frame", "right_frame"]).any():
                raise ValueError(f"duplicate checkpoint-8 pair for replica {replica}")
            target_frames[target] = block
        merged = target_frames["rmsd"].merge(
            target_frames["w1"], on=["left_frame", "right_frame"],
            suffixes=("_rmsd", "_w1"), validate="one_to_one",
        )
        result[replica] = merged.rename(columns={
            "target_value_rmsd": "rmsd", "target_value_w1": "w1"
        })
    return result


def summarize_assignment(
    audit: pd.DataFrame,
    model: str,
    calibration: str,
    prediction: np.ndarray,
    sigma: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: dict,
) -> list[dict]:
    rows = []
    boundary = settings["boundary_audit"]
    target = audit.target_value.to_numpy()
    strata = [("all", np.ones(len(audit), dtype=bool))]
    strata.extend((name, audit.support_category.to_numpy() == name) for name in (
        "common_support", "pf_extrapolation", "pf_vector_oos", "structurally_novel"
    ))
    for band, band_frame in audit.groupby("band", sort=False):
        band_mask = audit.index.isin(band_frame.index)
        low = float(band_frame.band_low.iloc[0]); high = float(band_frame.band_high.iloc[0])
        fit_low = low if np.isfinite(low) else float(audit.target_train_low.iloc[0])
        fit_high = high if np.isfinite(high) else float(audit.target_train_high.iloc[0])
        for stratum, stratum_mask in strata:
            mask = band_mask & stratum_mask
            if not mask.any() or fit_high <= fit_low:
                continue
            predicted_mass = gaussian_mass(prediction[mask], sigma[mask], fit_low, fit_high,
                                           boundary["distribution_bins"])
            edges = np.linspace(fit_low, fit_high, boundary["distribution_bins"] + 1)
            target_mass = np.histogram(np.clip(target[mask], fit_low, fit_high), bins=edges)[0]
            unique, effective = effective_endpoint_frames(
                audit.left_frame.to_numpy()[mask], audit.right_frame.to_numpy()[mask]
            )
            residual = target[mask] - prediction[mask]
            rows.append({
                "system_id": audit.system_id.iloc[0],
                "fit_replica": int(audit.fit_replica.iloc[0]),
                "calibration_replica": int(audit.calibration_replica.iloc[0]),
                "test_replica": int(audit.test_replica.iloc[0]),
                "target": audit.target.iloc[0], "model": model,
                "calibration": calibration, "band": band, "stratum": stratum,
                "pairs": int(mask.sum()), "unique_frames": unique,
                "effective_frames": effective,
                "coverage_90": float(np.mean((target[mask] >= lower[mask]) & (target[mask] <= upper[mask]))),
                "median_interval_width": float(np.median(upper[mask] - lower[mask])),
                "mean_interval_score": float(np.mean(interval_score(target[mask], lower[mask], upper[mask]))),
                "mean_nll": gaussian_nll(residual, sigma[mask] ** 2),
                "mean_sigma": float(np.mean(sigma[mask])),
                **extended_mass_errors(predicted_mass, target_mass, boundary["distribution_smoothing"]),
            })
    return rows


def analyse_system(row: dict[str, str], config: dict, parts: Path) -> str:
    system = row["system_id"]
    settings = config["analysis"]["pairwise_geometry"]
    strict = settings["strict_conformal"]
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint 8 pair audit is required: {source}")
    audit = pd.read_parquet(source)
    pairs = _pair_sets_from_audit(audit)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    vectors = {
        replica: absolute_change_vectors(
            z, frame.left_frame.to_numpy(), frame.right_frame.to_numpy()
        ) for replica, frame in pairs.items()
    }
    summaries, hyperparameters = [], []
    for assignment_index, (fit_replica, calibration_replica, test_replica) in enumerate(ordered_assignments()):
        fit_x = vectors[fit_replica]; calibration_x = vectors[calibration_replica]
        test_x = vectors[test_replica]
        mean, scale, floored = feature_standardizer(fit_x, settings["sigma_floor"])
        transforms = {
            "raw": (fit_x, calibration_x, test_x),
            "zscore": ((fit_x - mean) / scale, (calibration_x - mean) / scale,
                       (test_x - mean) / scale),
        }
        for target_index, target_name in enumerate(TARGETS):
            fit_target = pairs[fit_replica][target_name].to_numpy()
            calibration_target = pairs[calibration_replica][target_name].to_numpy()
            test_target = pairs[test_replica][target_name].to_numpy()
            assignment_audit = audit[
                (audit.fit_replica == fit_replica)
                & (audit.calibration_replica == calibration_replica)
                & (audit.test_replica == test_replica)
                & (audit.target == target_name)
            ].reset_index(drop=True)
            if not np.array_equal(
                assignment_audit[["left_frame", "right_frame"]].to_numpy(),
                pairs[test_replica][["left_frame", "right_frame"]].to_numpy(),
            ):
                raise ValueError(f"pair-order mismatch for {system} assignment")
            if not np.allclose(assignment_audit.target_value, test_target):
                raise ValueError(f"target mismatch for {system} assignment")
            for preprocessing, (fit_features, calibration_features, test_features) in transforms.items():
                seed = config["analysis"]["seed"] + 11003 * assignment_index + target_index
                fitted, alpha = fit_ridge_a_only(
                    fit_features, fit_target, settings["vector_audit"]["ridge_alphas"],
                    strict["ridge_tuning_fraction"], seed,
                )
                fit_prediction = np.maximum(0.0, fitted.predict(fit_features))
                calibration_prediction = np.maximum(0.0, fitted.predict(calibration_features))
                test_prediction = np.maximum(0.0, fitted.predict(test_features))
                scores = np.abs(calibration_target - calibration_prediction)
                marginal_q = finite_conformal_quantile(scores, strict["coverage"])
                mondrian_q, _ = mondrian_quantiles(
                    fit_prediction, calibration_prediction, scores, test_prediction,
                    strict["mondrian_bins"], strict["coverage"], marginal_q,
                )
                model = f"{preprocessing}_logpf_ridge_gaussian"
                for calibration, half_width in (("marginal", np.full(len(test_target), marginal_q)),
                                                ("mondrian", mondrian_q)):
                    sigma = np.maximum(half_width / GAUSSIAN_Q90, np.finfo(float).eps)
                    lower = np.maximum(0.0, test_prediction - half_width)
                    upper = test_prediction + half_width
                    summaries.extend(summarize_assignment(
                        assignment_audit, model, calibration, test_prediction, sigma,
                        lower, upper, settings,
                    ))
                hyperparameters.append({
                    "system_id": system, "fit_replica": fit_replica,
                    "calibration_replica": calibration_replica, "test_replica": test_replica,
                    "target": target_name, "preprocessing": preprocessing,
                    "ridge_alpha": alpha, "marginal_half_width": marginal_q,
                    "floored_residues": floored,
                })
    atomic_parquet(pd.DataFrame(summaries), parts / f"{system}.summary.parquet")
    atomic_parquet(pd.DataFrame(hyperparameters), parts / f"{system}.hyperparameters.parquet")
    return system


def valid_checkpoint(parts: Path, system: str) -> bool:
    paths = [parts / f"{system}.{suffix}.parquet" for suffix in ("summary", "hyperparameters")]
    if not all(path.exists() for path in paths):
        return False
    try:
        hyper = pd.read_parquet(paths[1], columns=[
            "system_id", "fit_replica", "calibration_replica", "test_replica", "target", "preprocessing"
        ])
    except Exception:
        return False
    assignments = set(map(tuple, hyper[["fit_replica", "calibration_replica", "test_replica"]].drop_duplicates().to_numpy()))
    return (set(hyper.system_id) == {system} and assignments == set(ordered_assignments())
            and set(hyper.target) == set(TARGETS) and set(hyper.preprocessing) == set(PREPROCESSING)
            and len(hyper) == 24)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint9_strict_likelihood"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed:
        print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            system = future.result()
            print(f"[{resumed + index}/{len(systems)}] {system} strict likelihood checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing:
        raise RuntimeError(f"missing checkpoint-9 systems: {missing}")
    summary = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems], ignore_index=True)
    hyper = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.hyperparameters.parquet") for row in systems], ignore_index=True)
    summary.to_parquet(output / "strict_likelihood_assignment_summary.parquet", index=False)
    hyper.to_parquet(output / "strict_likelihood_hyperparameters.parquet", index=False)
    atomic_yaml(output / "checkpoint9_run.yaml", {
        "checkpoint": "9", "status": "measurement_complete",
        "systems": len(systems), "assignments_per_system": 6,
        "pairs_per_role": config["analysis"]["pairwise_geometry"]["strict_conformal"]["pairs_per_replica"],
        "models": list(PREPROCESSING),
    })


if __name__ == "__main__":
    main()
