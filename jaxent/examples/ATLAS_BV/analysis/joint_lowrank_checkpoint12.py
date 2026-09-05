"""Checkpoint 12A: strict A-only low-rank joint log-PF likelihoods."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems
from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_checkpoint11 import (
    KERNEL_BANDWIDTHS, TARGETS, mixture_mean, mixture_probability_mass,
    mixture_quantiles, select_knn, select_ridge, summarize_model,
)
from jaxent.examples.ATLAS_BV.analysis.opening_distance_checkpoint10 import frame_disjoint_pair_split
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import finite_conformal_quantile, mondrian_quantiles, ordered_assignments
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import GAUSSIAN_Q90, _pair_sets_from_audit, gaussian_mass
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import absolute_change_vectors, deterministic_cap
from jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c import exact_neighbors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


MODELS = ("logpf_lowrank_ridge_gaussian", "logpf_lowrank_knn_mixture")


def fit_frame_pca(log_pf: np.ndarray, fit_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit frame-level joint modes on A only and transform every frame."""
    samples = np.asarray(log_pf[:, fit_frames].T, dtype=np.float64)
    maximum = min(samples.shape[0] - 1, samples.shape[1])
    if maximum < 1:
        raise ValueError("PCA requires at least two fit frames")
    model = PCA(n_components=maximum, svd_solver="full").fit(samples)
    scores = model.transform(np.asarray(log_pf.T, dtype=np.float64)).astype(np.float32)
    return scores, np.cumsum(model.explained_variance_ratio_)


def component_count(cumulative_variance: np.ndarray, threshold: float) -> int:
    """Smallest positive component count reaching the requested A-only variance."""
    return min(len(cumulative_variance), int(np.searchsorted(cumulative_variance, threshold) + 1))


def _features(scores: np.ndarray, count: int, pairs: pd.DataFrame) -> np.ndarray:
    return absolute_change_vectors(
        scores[:, :count].T, pairs.left_frame.to_numpy(), pairs.right_frame.to_numpy()
    )


def analyse_system(row: dict[str, str], config: dict, parts: Path) -> str:
    system = row["system_id"]; settings = config["analysis"]["pairwise_geometry"]
    strict = settings["strict_conformal"]; vector = settings["vector_audit"]
    thresholds = [float(value) for value in vector["pca_variance"]]
    source = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    audit = pd.read_parquet(source); pairs = _pair_sets_from_audit(audit)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    log_pf = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    pca_by_fit = {}
    for fit_replica in (1, 2, 3):
        fit_frames = np.unique(np.concatenate([
            pairs[fit_replica].left_frame.to_numpy(), pairs[fit_replica].right_frame.to_numpy()
        ]))
        pca_by_fit[fit_replica] = fit_frame_pca(log_pf, fit_frames)

    summaries, hyperparameters = [], []
    for assignment_index, (fit_replica, calibration_replica, test_replica) in enumerate(ordered_assignments()):
        fit_pairs = pairs[fit_replica]; calibration_pairs = pairs[calibration_replica]
        test_pairs = pairs[test_replica]
        fit_mask, validation_mask = frame_disjoint_pair_split(
            fit_pairs.left_frame.to_numpy(), fit_pairs.right_frame.to_numpy()
        )
        selection_mode = "frame_disjoint" if validation_mask.any() else "prespecified_no_disjoint_pair_split"
        scores, cumulative = pca_by_fit[fit_replica]
        counts = {threshold: component_count(cumulative, threshold) for threshold in thresholds}
        fit_features = {threshold: _features(scores, count, fit_pairs) for threshold, count in counts.items()}
        for target_index, target_name in enumerate(TARGETS):
            fit_target = fit_pairs[target_name].to_numpy()
            calibration_target = calibration_pairs[target_name].to_numpy()
            test_target = test_pairs[target_name].to_numpy()
            assignment_audit = audit[
                (audit.fit_replica == fit_replica) & (audit.calibration_replica == calibration_replica)
                & (audit.test_replica == test_replica) & (audit.target == target_name)
            ].reset_index(drop=True)

            ridge_candidates = []
            knn_candidates = []
            for threshold in thresholds:
                features = fit_features[threshold]
                alpha, loss = select_ridge(
                    features, fit_target, fit_mask, validation_mask, vector["ridge_alphas"]
                )
                ridge_candidates.append((loss, threshold, alpha))
                k, bandwidth, knn_loss = select_knn(
                    features, fit_target, fit_mask, validation_mask, vector["knn_neighbors"],
                    KERNEL_BANDWIDTHS, vector["knn_train_pairs"],
                    config["analysis"]["seed"] + 14009 * assignment_index + target_index,
                )
                knn_candidates.append((knn_loss, threshold, k, bandwidth))
            if validation_mask.any():
                ridge_loss, ridge_threshold, alpha = min(
                    ridge_candidates, key=lambda item: (item[0], item[1], -item[2])
                )
                knn_loss, knn_threshold, k, bandwidth = min(
                    knn_candidates, key=lambda item: (item[0], item[1], -item[2], item[3])
                )
            else:
                ridge_threshold, alpha, ridge_loss = 0.95, 10000.0, float("nan")
                knn_threshold, k, bandwidth, knn_loss = 0.95, 50, 1.0, float("nan")

            ridge_fit_x = fit_features[ridge_threshold]
            ridge_calibration_x = _features(scores, counts[ridge_threshold], calibration_pairs)
            ridge_test_x = _features(scores, counts[ridge_threshold], test_pairs)
            ridge = Ridge(alpha=alpha).fit(ridge_fit_x, fit_target)
            fit_prediction = np.maximum(0.0, ridge.predict(ridge_fit_x))
            calibration_prediction = np.maximum(0.0, ridge.predict(ridge_calibration_x))
            test_prediction = np.maximum(0.0, ridge.predict(ridge_test_x))
            residual_scores = np.abs(calibration_target - calibration_prediction)
            marginal_q = finite_conformal_quantile(residual_scores, strict["coverage"])
            mondrian_q, _ = mondrian_quantiles(
                fit_prediction, calibration_prediction, residual_scores, test_prediction,
                strict["mondrian_bins"], strict["coverage"], marginal_q,
            )
            for calibration, width in (("marginal", np.full(len(test_target), marginal_q)),
                                       ("mondrian", mondrian_q)):
                sigma = np.maximum(width / GAUSSIAN_Q90, np.finfo(float).eps)
                density = lambda mask, low, high, bins, p=test_prediction, s=sigma: gaussian_mass(
                    p[mask], s[mask], low, high, bins
                )
                summaries.extend(summarize_model(
                    assignment_audit, MODELS[0], calibration, test_prediction,
                    np.maximum(0.0, test_prediction - width), test_prediction + width,
                    density, settings,
                ))

            knn_fit_x = fit_features[knn_threshold]
            knn_calibration_x = _features(scores, counts[knn_threshold], calibration_pairs)
            knn_test_x = _features(scores, counts[knn_threshold], test_pairs)
            full_take = deterministic_cap(
                len(knn_fit_x), vector["knn_train_pairs"],
                config["analysis"]["seed"] + 15013 * assignment_index + target_index,
            )
            maximum_neighbors = min(max(vector["knn_neighbors"]), len(full_take))
            calibration_distances, calibration_indices = exact_neighbors(
                knn_fit_x[full_take], knn_calibration_x, maximum_neighbors
            )
            test_distances, test_indices = exact_neighbors(
                knn_fit_x[full_take], knn_test_x, maximum_neighbors
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
                calibration_target - calibration_base[:, 1], np.zeros(len(calibration_target)),
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
                    assignment_audit, MODELS[1], calibration, test_mean,
                    np.maximum(0.0, test_base[:, 0] - adjustment),
                    test_base[:, 1] + adjustment, density, settings,
                ))
            hyperparameters.extend([
                {"system_id": system, "fit_replica": fit_replica, "calibration_replica": calibration_replica,
                 "test_replica": test_replica, "target": target_name, "model": MODELS[0],
                 "variance_threshold": ridge_threshold, "components": counts[ridge_threshold],
                 "alpha": alpha, "neighbors": None, "bandwidth": None,
                 "validation_loss": ridge_loss, "selection_mode": selection_mode},
                {"system_id": system, "fit_replica": fit_replica, "calibration_replica": calibration_replica,
                 "test_replica": test_replica, "target": target_name, "model": MODELS[1],
                 "variance_threshold": knn_threshold, "components": counts[knn_threshold],
                 "alpha": None, "neighbors": k, "bandwidth": bandwidth,
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
    return (np.isfinite(summary.to_numpy()).all() and set(hyper.system_id) == {system}
            and assignments == set(ordered_assignments()) and set(hyper.target) == set(TARGETS)
            and set(hyper.model) == set(MODELS) and len(hyper) == 24)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint12_joint_lowrank"
    parts = output / "parts"; parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in systems if args.restart or not valid_checkpoint(parts, row["system_id"])]
    resumed = len(systems) - len(pending)
    if resumed: print(f"resuming from {resumed}/{len(systems)} valid system checkpoints", flush=True)
    workers = args.workers or config["analysis"]["pairwise_geometry"]["workers"]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyse_system, row, config, parts): row for row in pending}
        for index, future in enumerate(as_completed(futures), 1):
            system = future.result(); print(f"[{resumed + index}/{len(systems)}] {system} low-rank checkpointed", flush=True)
    missing = [row["system_id"] for row in systems if not valid_checkpoint(parts, row["system_id"])]
    if missing: raise RuntimeError(f"missing checkpoint-12 systems: {missing}")
    summary = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.summary.parquet") for row in systems], ignore_index=True)
    hyper = pd.concat([pd.read_parquet(parts / f"{row['system_id']}.hyperparameters.parquet") for row in systems], ignore_index=True)
    summary.to_parquet(output / "joint_lowrank_assignment_summary.parquet", index=False)
    hyper.to_parquet(output / "joint_lowrank_hyperparameters.parquet", index=False)
    atomic_yaml(output / "checkpoint12a_run.yaml", {"checkpoint": "12A", "status": "measurement_complete", "systems": len(systems)})


if __name__ == "__main__":
    main()
