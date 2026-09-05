"""Checkpoint 10: strict opening-probability and naive-distance screen."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, xlogy
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.boundary_checkpoint2 import boundary_predictions
from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import pf_pair_distance
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import finite_conformal_quantile, mondrian_quantiles, ordered_assignments
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import (
    _pair_sets_from_audit, extended_mass_errors, interval_score,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import effective_endpoint_frames
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import absolute_change_vectors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import normalized_mae


TARGETS = ("rmsd", "w1")
PROFILE_TRANSFORMS = ("raw", "frame_centered", "residue_zscore", "centered_zscore")
PROFILE_METRICS = ("l1", "l2", "cosine", "correlation")


@dataclass(frozen=True)
class Candidate:
    fit_prediction: np.ndarray
    calibration_prediction: np.ndarray
    test_prediction: np.ndarray
    validation_loss: float
    alpha: float | None
    family: str


def opening_probability(log_pf: np.ndarray) -> np.ndarray:
    """Stable EX2-inspired marginal opening probability."""
    return expit(-np.asarray(log_pf, dtype=np.float64))


def transform_opening_profiles(
    probability: np.ndarray, fit_frames: np.ndarray, transform: str, sigma_floor: float
) -> tuple[np.ndarray, int]:
    if transform not in PROFILE_TRANSFORMS:
        raise ValueError(transform)
    values = np.asarray(probability, dtype=np.float64).copy()
    if transform in {"frame_centered", "centered_zscore"}:
        values -= values.mean(axis=0, keepdims=True)
    floored = 0
    if transform in {"residue_zscore", "centered_zscore"}:
        mean = values[:, fit_frames].mean(axis=1, keepdims=True)
        sigma = values[:, fit_frames].std(axis=1, keepdims=True)
        floored = int(np.count_nonzero(sigma < sigma_floor))
        values = (values - mean) / np.maximum(sigma, sigma_floor)
    return values, floored


def bernoulli_pair_distance(
    probability: np.ndarray, left: np.ndarray, right: np.ndarray,
    metric: str, epsilon: float = 1e-12, chunk_size: int = 5000,
) -> np.ndarray:
    """Mean residue-wise Bernoulli divergence; symmetric and finite."""
    output = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        p = np.clip(probability[:, left[start:stop]], epsilon, 1.0 - epsilon)
        q = np.clip(probability[:, right[start:stop]], epsilon, 1.0 - epsilon)
        if metric == "sqrt_jsd":
            m = 0.5 * (p + q)
            divergence = 0.5 * (
                xlogy(p, p / m) + xlogy(1 - p, (1 - p) / (1 - m))
                + xlogy(q, q / m) + xlogy(1 - q, (1 - q) / (1 - m))
            )
            output[start:stop] = np.mean(np.sqrt(np.maximum(divergence, 0.0)), axis=0)
        elif metric == "jeffreys":
            kl_pq = xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q))
            kl_qp = xlogy(q, q / p) + xlogy(1 - q, (1 - q) / (1 - p))
            output[start:stop] = np.mean(0.5 * (kl_pq + kl_qp), axis=0)
        else:
            raise ValueError(metric)
    return output


def frame_disjoint_pair_split(
    left: np.ndarray, right: np.ndarray, validation_fraction: float = 0.20
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a viable contiguous endpoint-frame block; crossing pairs are excluded."""
    frames = np.unique(np.concatenate([left, right]))
    blocks = max(2, int(round(1.0 / validation_fraction)))
    candidates = []
    for block_index, validation_frames in enumerate(np.array_split(frames, blocks)):
        validation = np.isin(left, validation_frames) & np.isin(right, validation_frames)
        fit = ~np.isin(left, validation_frames) & ~np.isin(right, validation_frames)
        if fit.any() and validation.any():
            candidates.append((int(validation.sum()), -block_index, fit, validation))
    rng = np.random.default_rng(101)
    validation_count = max(2, int(np.ceil(validation_fraction * len(frames))))
    for attempt in range(256):
        validation_frames = rng.choice(frames, validation_count, replace=False)
        validation = np.isin(left, validation_frames) & np.isin(right, validation_frames)
        fit = ~np.isin(left, validation_frames) & ~np.isin(right, validation_frames)
        if fit.any() and validation.any():
            candidates.append((int(validation.sum()), -(blocks + attempt), fit, validation))
    if not candidates:
        return np.ones(len(left), dtype=bool), np.zeros(len(left), dtype=bool)
    _, _, fit, validation = max(candidates, key=lambda item: (item[0], item[1]))
    return fit, validation


def _scalar_fit_predict(
    fit_distance: np.ndarray, calibration_distance: np.ndarray, test_distance: np.ndarray,
    fit_target: np.ndarray, fit_mask: np.ndarray, validation_mask: np.ndarray, settings: dict,
) -> Candidate:
    boundary = settings["boundary_audit"]
    if validation_mask.any():
        validation_prediction = boundary_predictions(
            fit_distance[fit_mask], fit_target[fit_mask], fit_distance[validation_mask],
            boundary["tail_fraction"], boundary["tail_minimum_pairs"],
        )["extrapolated_test"]
        loss = normalized_mae(fit_target[validation_mask], validation_prediction, fit_target[fit_mask])
    else:
        loss = float("nan")
    full = boundary_predictions(
        fit_distance, fit_target, np.concatenate([fit_distance, calibration_distance, test_distance]),
        boundary["tail_fraction"], boundary["tail_minimum_pairs"],
    )["extrapolated_test"]
    n_fit = len(fit_distance); n_calibration = len(calibration_distance)
    return Candidate(
        np.maximum(0.0, full[:n_fit]),
        np.maximum(0.0, full[n_fit:n_fit + n_calibration]),
        np.maximum(0.0, full[n_fit + n_calibration:]),
        loss, None, "scalar",
    )


def _ridge_fit_predict(
    fit_x: np.ndarray, calibration_x: np.ndarray, test_x: np.ndarray,
    fit_target: np.ndarray, fit_mask: np.ndarray, validation_mask: np.ndarray,
    alphas: list[float],
) -> Candidate:
    if validation_mask.any():
        scored = []
        for alpha in alphas:
            fitted = Ridge(alpha=alpha).fit(fit_x[fit_mask], fit_target[fit_mask])
            loss = normalized_mae(
                fit_target[validation_mask], fitted.predict(fit_x[validation_mask]), fit_target[fit_mask]
            )
            scored.append((loss, float(alpha)))
        loss, alpha = min(scored, key=lambda item: (item[0], -item[1]))
    else:
        alpha = float(10000.0 if 10000.0 in alphas else alphas[len(alphas) // 2])
        loss = float("nan")
    fitted = Ridge(alpha=alpha).fit(fit_x, fit_target)
    return Candidate(
        np.maximum(0.0, fitted.predict(fit_x)),
        np.maximum(0.0, fitted.predict(calibration_x)),
        np.maximum(0.0, fitted.predict(test_x)),
        loss, alpha, "vector",
    )


def summarize_point_candidate(
    audit: pd.DataFrame, model: str, calibration: str,
    prediction: np.ndarray, lower: np.ndarray, upper: np.ndarray, settings: dict,
) -> list[dict]:
    rows = []; boundary = settings["boundary_audit"]
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
            edges = np.linspace(fit_low, fit_high, boundary["distribution_bins"] + 1)
            predicted_mass = np.histogram(np.clip(prediction[mask], fit_low, fit_high), bins=edges)[0]
            target_mass = np.histogram(np.clip(target[mask], fit_low, fit_high), bins=edges)[0]
            unique, effective = effective_endpoint_frames(
                audit.left_frame.to_numpy()[mask], audit.right_frame.to_numpy()[mask]
            )
            rows.append({
                "system_id": audit.system_id.iloc[0], "fit_replica": int(audit.fit_replica.iloc[0]),
                "calibration_replica": int(audit.calibration_replica.iloc[0]),
                "test_replica": int(audit.test_replica.iloc[0]), "target": audit.target.iloc[0],
                "model": model, "calibration": calibration, "band": band, "stratum": stratum,
                "pairs": int(mask.sum()), "unique_frames": unique, "effective_frames": effective,
                "coverage_90": float(np.mean((target[mask] >= lower[mask]) & (target[mask] <= upper[mask]))),
                "median_interval_width": float(np.median(upper[mask] - lower[mask])),
                "mean_interval_score": float(np.mean(interval_score(target[mask], lower[mask], upper[mask]))),
                **extended_mass_errors(predicted_mass, target_mass, boundary["distribution_smoothing"]),
            })
    return rows


def analyse_system(row: dict[str, str], config: dict, parts: Path) -> str:
    system = row["system_id"]; settings = config["analysis"]["pairwise_geometry"]
    strict = settings["strict_conformal"]
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    audit = pd.read_parquet(source); pairs = _pair_sets_from_audit(audit)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    log_pf = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    probability = opening_probability(log_pf)
    summaries, hyperparameters = [], []
    for assignment_index, (fit_replica, calibration_replica, test_replica) in enumerate(ordered_assignments()):
        fit_pairs = pairs[fit_replica]; calibration_pairs = pairs[calibration_replica]
        test_pairs = pairs[test_replica]
        fit_frames = np.unique(np.concatenate([fit_pairs.left_frame, fit_pairs.right_frame]))
        fit_mask, validation_mask = frame_disjoint_pair_split(
            fit_pairs.left_frame.to_numpy(), fit_pairs.right_frame.to_numpy()
        )
        profiles = {
            transform: transform_opening_profiles(probability, fit_frames, transform, settings["sigma_floor"])
            for transform in PROFILE_TRANSFORMS
        }
        pair_profiles = {
            transform: {
                replica: absolute_change_vectors(values, frame.left_frame.to_numpy(), frame.right_frame.to_numpy())
                for replica, frame in pairs.items()
            } for transform, (values, _) in profiles.items()
        }
        scalar_distances = {}
        for transform, (values, _) in profiles.items():
            for metric in PROFILE_METRICS:
                scalar_distances[f"opening_{transform}_{metric}"] = {
                    replica: pf_pair_distance(values, frame.left_frame.to_numpy(), frame.right_frame.to_numpy(), metric)
                    for replica, frame in pairs.items()
                }
        for metric in ("sqrt_jsd", "jeffreys"):
            scalar_distances[f"opening_raw_bernoulli_{metric}"] = {
                replica: bernoulli_pair_distance(
                    probability, frame.left_frame.to_numpy(), frame.right_frame.to_numpy(), metric
                ) for replica, frame in pairs.items()
            }
        for metric in PROFILE_METRICS:
            scalar_distances[f"logpf_raw_{metric}"] = {
                replica: pf_pair_distance(
                    log_pf, frame.left_frame.to_numpy(), frame.right_frame.to_numpy(), metric
                ) for replica, frame in pairs.items()
            }
        logpf_vectors = {
            replica: absolute_change_vectors(log_pf, frame.left_frame.to_numpy(), frame.right_frame.to_numpy())
            for replica, frame in pairs.items()
        }
        for target_index, target_name in enumerate(TARGETS):
            fit_target = fit_pairs[target_name].to_numpy()
            calibration_target = calibration_pairs[target_name].to_numpy()
            test_target = test_pairs[target_name].to_numpy()
            assignment_audit = audit[
                (audit.fit_replica == fit_replica) & (audit.calibration_replica == calibration_replica)
                & (audit.test_replica == test_replica) & (audit.target == target_name)
            ].reset_index(drop=True)
            candidates = {}
            for name, distance in scalar_distances.items():
                candidates[name] = _scalar_fit_predict(
                    distance[fit_replica], distance[calibration_replica], distance[test_replica],
                    fit_target, fit_mask, validation_mask, settings,
                )
            candidates["logpf_raw_vector_ridge"] = _ridge_fit_predict(
                logpf_vectors[fit_replica], logpf_vectors[calibration_replica], logpf_vectors[test_replica],
                fit_target, fit_mask, validation_mask, settings["vector_audit"]["ridge_alphas"],
            )
            for transform, vectors in pair_profiles.items():
                candidates[f"opening_{transform}_vector_ridge"] = _ridge_fit_predict(
                    vectors[fit_replica], vectors[calibration_replica], vectors[test_replica],
                    fit_target, fit_mask, validation_mask, settings["vector_audit"]["ridge_alphas"],
                )
            opening_names = [name for name in candidates if name.startswith("opening_")]
            selection_mode = "frame_disjoint"
            if validation_mask.any():
                selected_name = min(opening_names, key=lambda name: (candidates[name].validation_loss, name))
            else:
                selected_name = "opening_raw_l1"
                selection_mode = "prespecified_no_disjoint_pair_split"
            candidates["a_selected_opening"] = candidates[selected_name]
            for model_name, candidate in candidates.items():
                scores = np.abs(calibration_target - candidate.calibration_prediction)
                marginal_q = finite_conformal_quantile(scores, strict["coverage"])
                mondrian_q, _ = mondrian_quantiles(
                    candidate.fit_prediction, candidate.calibration_prediction, scores,
                    candidate.test_prediction, strict["mondrian_bins"], strict["coverage"], marginal_q,
                )
                for calibration, width in (("marginal", np.full(len(test_target), marginal_q)),
                                           ("mondrian", mondrian_q)):
                    lower = np.maximum(0.0, candidate.test_prediction - width)
                    upper = candidate.test_prediction + width
                    summaries.extend(summarize_point_candidate(
                        assignment_audit, model_name, calibration, candidate.test_prediction,
                        lower, upper, settings,
                    ))
                hyperparameters.append({
                    "system_id": system, "fit_replica": fit_replica,
                    "calibration_replica": calibration_replica, "test_replica": test_replica,
                    "target": target_name, "model": model_name,
                    "family": candidate.family, "validation_normalized_mae": candidate.validation_loss,
                    "alpha": candidate.alpha, "selected_source": selected_name if model_name == "a_selected_opening" else None,
                    "selection_mode": selection_mode,
                    "fit_pairs": int(fit_mask.sum()), "validation_pairs": int(validation_mask.sum()),
                })
    atomic_parquet(pd.DataFrame(summaries), parts / f"{system}.summary.parquet")
    atomic_parquet(pd.DataFrame(hyperparameters), parts / f"{system}.hyperparameters.parquet")
    return system


def valid_checkpoint(parts: Path, system: str) -> bool:
    paths = [parts / f"{system}.{suffix}.parquet" for suffix in ("summary", "hyperparameters")]
    if not all(path.exists() for path in paths): return False
    try:
        hyper = pd.read_parquet(paths[1])
    except Exception:
        return False
    assignments = set(map(tuple, hyper[["fit_replica", "calibration_replica", "test_replica"]].drop_duplicates().to_numpy()))
    required = {"a_selected_opening", "logpf_raw_vector_ridge"}
    required.update(f"logpf_raw_{metric}" for metric in PROFILE_METRICS)
    return (set(hyper.system_id) == {system} and assignments == set(ordered_assignments())
            and set(hyper.target) == set(TARGETS) and required.issubset(set(hyper.model)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint10_opening_screen"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed: print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            system = future.result(); print(f"[{resumed + index}/{len(systems)}] {system} opening screen checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing checkpoint-10 systems: {missing}")
    summary = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems], ignore_index=True)
    hyper = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.hyperparameters.parquet") for row in systems], ignore_index=True)
    summary.to_parquet(output / "opening_screen_assignment_summary.parquet", index=False)
    hyper.to_parquet(output / "opening_screen_hyperparameters.parquet", index=False)
    atomic_yaml(output / "checkpoint10_run.yaml", {"checkpoint": "10", "status": "measurement_complete", "systems": len(systems)})


if __name__ == "__main__":
    main()
