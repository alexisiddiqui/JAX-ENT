"""Checkpoint 11: strict per-residue Gaussian and kNN-mixture likelihoods."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems
from jaxent.examples.ATLAS_BV.analysis.opening_distance_checkpoint10 import (
    frame_disjoint_pair_split, opening_probability,
)
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import finite_conformal_quantile, mondrian_quantiles, ordered_assignments
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import (
    GAUSSIAN_Q90, _pair_sets_from_audit, extended_mass_errors, gaussian_mass, interval_score,
)
from jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 import effective_endpoint_frames
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import absolute_change_vectors, deterministic_cap
from jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c import exact_neighbors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet, gaussian_nll
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import normalized_mae


TARGETS = ("rmsd", "w1")
REPRESENTATIONS = ("logpf", "opening")
KERNEL_BANDWIDTHS = (0.5, 1.0, 2.0)


def kernel_weights(distances: np.ndarray, neighbors: int, bandwidth: float) -> np.ndarray:
    """Gaussian feature-space weights with row-local k-distance scaling."""
    count = min(neighbors, distances.shape[1])
    local = np.asarray(distances[:, :count], dtype=np.float64)
    scale = np.maximum(local[:, -1:], np.finfo(float).eps) * bandwidth
    weights = np.exp(-0.5 * (local / scale) ** 2)
    exact = local <= np.finfo(float).eps
    rows = exact.any(axis=1)
    weights[rows] = exact[rows]
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def mixture_mean(
    distances: np.ndarray, indices: np.ndarray, target: np.ndarray,
    neighbors: int, bandwidth: float,
) -> np.ndarray:
    count = min(neighbors, indices.shape[1])
    weights = kernel_weights(distances, count, bandwidth)
    return np.sum(weights * target[indices[:, :count]], axis=1)


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = np.take_along_axis(weights, order, axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    output = np.empty((len(values), len(quantiles)), dtype=float)
    for column, quantile in enumerate(quantiles):
        index = np.argmax(cumulative >= quantile, axis=1)
        output[:, column] = sorted_values[np.arange(len(values)), index]
    return output


def mixture_quantiles(
    distances: np.ndarray, indices: np.ndarray, target: np.ndarray,
    neighbors: int, bandwidth: float, quantiles: tuple[float, ...] = (0.05, 0.95),
) -> np.ndarray:
    count = min(neighbors, indices.shape[1])
    weights = kernel_weights(distances, count, bandwidth)
    return weighted_quantiles(target[indices[:, :count]], weights, quantiles)


def mixture_probability_mass(
    distances: np.ndarray, indices: np.ndarray, train_target: np.ndarray,
    mask: np.ndarray, neighbors: int, bandwidth: float, low: float, high: float, bins: int,
) -> np.ndarray:
    count = min(neighbors, indices.shape[1])
    local_distances = distances[mask, :count]; local_indices = indices[mask, :count]
    weights = kernel_weights(local_distances, count, bandwidth)
    edges = np.linspace(low, high, bins + 1)
    labels = np.clip(
        np.digitize(np.clip(train_target[local_indices], low, high), edges[1:-1]), 0, bins - 1
    )
    mass = np.array([np.sum(weights[labels == label]) for label in range(bins)], dtype=float)
    return mass / max(1, int(mask.sum()))


def select_ridge(
    features: np.ndarray, target: np.ndarray, fit_mask: np.ndarray,
    validation_mask: np.ndarray, alphas: list[float],
) -> tuple[float, float]:
    if not validation_mask.any():
        alpha = float(10000.0 if 10000.0 in alphas else alphas[len(alphas) // 2])
        return alpha, float("nan")
    scored = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(features[fit_mask], target[fit_mask])
        loss = normalized_mae(target[validation_mask], model.predict(features[validation_mask]), target[fit_mask])
        scored.append((loss, float(alpha)))
    return min(scored, key=lambda item: (item[0], -item[1]))[::-1]


def select_knn(
    features: np.ndarray, target: np.ndarray, fit_mask: np.ndarray, validation_mask: np.ndarray,
    neighbors: list[int], bandwidths: tuple[float, ...], cap: int, seed: int,
) -> tuple[int, float, float]:
    if not validation_mask.any():
        return min(50, max(neighbors)), 1.0, float("nan")
    fit_indices = np.flatnonzero(fit_mask)
    take = deterministic_cap(len(fit_indices), cap, seed)
    train_indices = fit_indices[take]
    maximum = min(max(neighbors), len(train_indices))
    distances, indices = exact_neighbors(features[train_indices], features[validation_mask], maximum)
    scored = []
    for k in neighbors:
        if k > maximum: continue
        for bandwidth in bandwidths:
            prediction = mixture_mean(distances, indices, target[train_indices], k, bandwidth)
            loss = normalized_mae(target[validation_mask], prediction, target[train_indices])
            scored.append((loss, int(k), float(bandwidth)))
    loss, k, bandwidth = min(scored, key=lambda item: (item[0], -item[1], item[2]))
    return k, bandwidth, loss


def summarize_model(
    audit: pd.DataFrame, model: str, calibration: str, prediction: np.ndarray,
    lower: np.ndarray, upper: np.ndarray, density_mass, settings: dict,
) -> list[dict]:
    rows = []; boundary = settings["boundary_audit"]; target = audit.target_value.to_numpy()
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
            if not mask.any() or fit_high <= fit_low: continue
            predicted_mass = density_mass(mask, fit_low, fit_high, boundary["distribution_bins"])
            edges = np.linspace(fit_low, fit_high, boundary["distribution_bins"] + 1)
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
    strict = settings["strict_conformal"]; vector = settings["vector_audit"]
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    audit = pd.read_parquet(source); pairs = _pair_sets_from_audit(audit)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    log_pf = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    representations = {"logpf": log_pf, "opening": opening_probability(log_pf)}
    pair_vectors = {
        name: {replica: absolute_change_vectors(values, frame.left_frame.to_numpy(), frame.right_frame.to_numpy())
               for replica, frame in pairs.items()}
        for name, values in representations.items()
    }
    summaries, hyperparameters = [], []
    for assignment_index, (fit_replica, calibration_replica, test_replica) in enumerate(ordered_assignments()):
        fit_pairs = pairs[fit_replica]; calibration_pairs = pairs[calibration_replica]; test_pairs = pairs[test_replica]
        fit_mask, validation_mask = frame_disjoint_pair_split(
            fit_pairs.left_frame.to_numpy(), fit_pairs.right_frame.to_numpy()
        )
        selection_mode = "frame_disjoint" if validation_mask.any() else "prespecified_no_disjoint_pair_split"
        for representation_name in REPRESENTATIONS:
            fit_x = pair_vectors[representation_name][fit_replica]
            calibration_x = pair_vectors[representation_name][calibration_replica]
            test_x = pair_vectors[representation_name][test_replica]
            full_take = deterministic_cap(
                len(fit_x), vector["knn_train_pairs"],
                config["analysis"]["seed"] + 12011 * assignment_index + (0 if representation_name == "logpf" else 1),
            )
            maximum_neighbors = min(max(vector["knn_neighbors"]), len(full_take))
            calibration_distances, calibration_indices = exact_neighbors(
                fit_x[full_take], calibration_x, maximum_neighbors
            )
            test_distances, test_indices = exact_neighbors(fit_x[full_take], test_x, maximum_neighbors)
            for target_index, target_name in enumerate(TARGETS):
                fit_target = fit_pairs[target_name].to_numpy()
                calibration_target = calibration_pairs[target_name].to_numpy()
                test_target = test_pairs[target_name].to_numpy()
                assignment_audit = audit[
                    (audit.fit_replica == fit_replica) & (audit.calibration_replica == calibration_replica)
                    & (audit.test_replica == test_replica) & (audit.target == target_name)
                ].reset_index(drop=True)
                alpha, ridge_loss = select_ridge(
                    fit_x, fit_target, fit_mask, validation_mask, vector["ridge_alphas"]
                )
                ridge = Ridge(alpha=alpha).fit(fit_x, fit_target)
                fit_prediction = np.maximum(0.0, ridge.predict(fit_x))
                calibration_prediction = np.maximum(0.0, ridge.predict(calibration_x))
                test_prediction = np.maximum(0.0, ridge.predict(test_x))
                scores = np.abs(calibration_target - calibration_prediction)
                marginal_q = finite_conformal_quantile(scores, strict["coverage"])
                mondrian_q, _ = mondrian_quantiles(
                    fit_prediction, calibration_prediction, scores, test_prediction,
                    strict["mondrian_bins"], strict["coverage"], marginal_q,
                )
                for calibration, width in (("marginal", np.full(len(test_target), marginal_q)),
                                           ("mondrian", mondrian_q)):
                    sigma = np.maximum(width / GAUSSIAN_Q90, np.finfo(float).eps)
                    density = lambda mask, low, high, bins, p=test_prediction, s=sigma: gaussian_mass(
                        p[mask], s[mask], low, high, bins
                    )
                    summaries.extend(summarize_model(
                        assignment_audit, f"{representation_name}_ridge_gaussian", calibration,
                        test_prediction, np.maximum(0.0, test_prediction - width),
                        test_prediction + width, density, settings,
                    ))
                k, bandwidth, knn_loss = select_knn(
                    fit_x, fit_target, fit_mask, validation_mask, vector["knn_neighbors"],
                    KERNEL_BANDWIDTHS, vector["knn_train_pairs"],
                    config["analysis"]["seed"] + 13007 * assignment_index + target_index,
                )
                calibration_mean = mixture_mean(
                    calibration_distances, calibration_indices, fit_target[full_take], k, bandwidth
                )
                test_mean = mixture_mean(test_distances, test_indices, fit_target[full_take], k, bandwidth)
                calibration_base = mixture_quantiles(
                    calibration_distances, calibration_indices, fit_target[full_take], k, bandwidth
                )
                test_base = mixture_quantiles(
                    test_distances, test_indices, fit_target[full_take], k, bandwidth
                )
                cqr_scores = np.maximum.reduce([
                    calibration_base[:, 0] - calibration_target,
                    calibration_target - calibration_base[:, 1],
                    np.zeros(len(calibration_target)),
                ])
                cqr_q = finite_conformal_quantile(cqr_scores, strict["coverage"])
                cqr_mondrian, _ = mondrian_quantiles(
                    fit_target, calibration_mean, cqr_scores, test_mean,
                    strict["mondrian_bins"], strict["coverage"], cqr_q,
                )
                for calibration, adjustment in (("marginal", np.full(len(test_target), cqr_q)),
                                                ("mondrian", cqr_mondrian)):
                    density = lambda mask, low, high, bins, d=test_distances, i=test_indices, t=fit_target[full_take], kk=k, bw=bandwidth: mixture_probability_mass(
                        d, i, t, mask, kk, bw, low, high, bins
                    )
                    summaries.extend(summarize_model(
                        assignment_audit, f"{representation_name}_knn_mixture", calibration,
                        test_mean, np.maximum(0.0, test_base[:, 0] - adjustment),
                        test_base[:, 1] + adjustment, density, settings,
                    ))
                hyperparameters.extend([
                    {"system_id": system, "fit_replica": fit_replica, "calibration_replica": calibration_replica,
                     "test_replica": test_replica, "target": target_name, "representation": representation_name,
                     "model": "ridge_gaussian", "alpha": alpha, "neighbors": None, "bandwidth": None,
                     "validation_loss": ridge_loss, "selection_mode": selection_mode},
                    {"system_id": system, "fit_replica": fit_replica, "calibration_replica": calibration_replica,
                     "test_replica": test_replica, "target": target_name, "representation": representation_name,
                     "model": "knn_mixture", "alpha": None, "neighbors": k, "bandwidth": bandwidth,
                     "validation_loss": knn_loss, "selection_mode": selection_mode},
                ])
    atomic_parquet(pd.DataFrame(summaries), parts / f"{system}.summary.parquet")
    atomic_parquet(pd.DataFrame(hyperparameters), parts / f"{system}.hyperparameters.parquet")
    return system


def valid_checkpoint(parts: Path, system: str) -> bool:
    paths = [parts / f"{system}.{suffix}.parquet" for suffix in ("summary", "hyperparameters")]
    if not all(path.exists() for path in paths): return False
    try:
        summary = pd.read_parquet(paths[0], columns=["distribution_recovery", "coverage_90"])
        hyper = pd.read_parquet(paths[1])
    except Exception: return False
    assignments = set(map(tuple, hyper[["fit_replica", "calibration_replica", "test_replica"]].drop_duplicates().to_numpy()))
    return (np.isfinite(summary[["distribution_recovery", "coverage_90"]].to_numpy()).all()
            and set(hyper.system_id) == {system} and assignments == set(ordered_assignments())
            and set(hyper.target) == set(TARGETS) and set(hyper.representation) == set(REPRESENTATIONS)
            and set(hyper.model) == {"ridge_gaussian", "knn_mixture"} and len(hyper) == 48)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint11_conditional_likelihood"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed: print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            system = future.result(); print(f"[{resumed + index}/{len(systems)}] {system} conditional likelihood checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing checkpoint-11 systems: {missing}")
    summary = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems], ignore_index=True)
    hyper = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.hyperparameters.parquet") for row in systems], ignore_index=True)
    summary.to_parquet(output / "conditional_likelihood_assignment_summary.parquet", index=False)
    hyper.to_parquet(output / "conditional_likelihood_hyperparameters.parquet", index=False)
    atomic_yaml(output / "checkpoint11_run.yaml", {"checkpoint": "11", "status": "measurement_complete", "systems": len(systems)})


if __name__ == "__main__":
    main()
