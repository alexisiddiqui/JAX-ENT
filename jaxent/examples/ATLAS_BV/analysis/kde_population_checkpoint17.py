"""Checkpoint 17: W1-kernel MD density versus fixed-BV pairwise population ratios."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist, pdist
from sklearn.linear_model import Ridge

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE, atomic_yaml, load_config, load_contact_coordinates, load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.opening_distance_checkpoint10 import frame_disjoint_pair_split
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import pf_pair_distance
from jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 import finite_conformal_quantile, ordered_assignments
from jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 import _pair_sets_from_audit, extended_mass_errors, interval_score
from jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 import absolute_change_vectors
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet
from jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b import normalized_mae
from jaxent.examples.ATLAS_BV.analysis.basin_census import load_ca_coordinates


NEIGHBOUR_RANKS = (5, 10, 20, 50)
SCALAR_METRICS = ("absolute_l1", "l2", "cosine", "correlation")
PRIMARY_RANK = 10


def frame_w1_signatures(coordinates: np.ndarray, count: int = 256) -> np.ndarray:
    """Streaming inverse-CDF signatures without retaining every C-alpha pair distance."""
    probabilities = np.linspace(0.0, 1.0, count)
    if coordinates.shape[1] < 2:
        raise ValueError("W1 geometry signatures require at least two C-alpha atoms")
    return np.asarray([np.quantile(pdist(frame), probabilities) for frame in coordinates], dtype=np.float32)


def w1_matrices(signatures: np.ndarray, replicas: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    output = {}
    for replica in (1, 2, 3):
        indices = np.flatnonzero(replicas == replica)
        values = signatures[indices]
        matrix = cdist(values, values, metric="cityblock") / values.shape[1]
        output[replica] = (indices, matrix)
    return output


def neighbour_bandwidth(matrix: np.ndarray, rank: int) -> float:
    k = min(rank, len(matrix) - 1)
    if k < 1:
        return float(np.finfo(float).eps)
    kth = np.partition(matrix, k, axis=1)[:, k]
    positive = kth[kth > 0]
    return float(np.median(positive)) if len(positive) else float(np.finfo(float).eps)


def log_kernel_density(matrix: np.ndarray, bandwidth: float) -> np.ndarray:
    kernel = np.exp(-0.5 * np.square(matrix / max(bandwidth, np.finfo(float).eps)))
    np.fill_diagonal(kernel, 0.0)
    density = kernel.sum(axis=1) / max(1, len(matrix) - 1)
    return np.log(np.maximum(density, np.finfo(float).tiny))


def scalar_scale(x: np.ndarray, y: np.ndarray, nonnegative: bool) -> tuple[float, float, float]:
    denominator = float(np.dot(x, x))
    alpha = float(np.dot(x, y) / denominator) if denominator > 0 else 0.0
    if nonnegative:
        alpha = max(0.0, alpha)
    prediction = alpha * x
    return alpha, float(np.dot(x, y)), denominator


def select_ridge_no_intercept(
    x: np.ndarray, y: np.ndarray, fit_mask: np.ndarray, validation_mask: np.ndarray,
    alphas: list[float],
) -> tuple[Ridge, float, float]:
    if validation_mask.any():
        scored = []
        for alpha in alphas:
            model = Ridge(alpha=alpha, fit_intercept=False).fit(x[fit_mask], y[fit_mask])
            loss = normalized_mae(y[validation_mask], model.predict(x[validation_mask]), y[fit_mask])
            scored.append((loss, -float(alpha), float(alpha)))
        loss, _, alpha = min(scored)
    else:
        alpha = float(max(alphas)); loss = float("nan")
    return Ridge(alpha=alpha, fit_intercept=False).fit(x, y), alpha, loss


def system_data(row: dict[str, str], config: dict) -> dict:
    system = row["system_id"]
    audit_path = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint8_strict_conformal" / "parts" / f"{system}.pairs.parquet"
    audit = pd.read_parquet(audit_path); pairs = _pair_sets_from_audit(audit)
    coordinates, replicas, _ = load_ca_coordinates(row, config)
    signatures = frame_w1_signatures(coordinates, config["analysis"]["pairwise_geometry"]["support_audit"]["w1_support_quantiles"])
    matrices = w1_matrices(signatures, replicas)
    contacts = [load_contact_coordinates(system, replica, config) for replica in (1, 2, 3)]
    heavy = np.concatenate([item["heavy"] for item in contacts], axis=1)
    acceptor = np.concatenate([item["acceptor"] for item in contacts], axis=1)
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor
    return {"system": system, "audit": audit, "pairs": pairs, "replicas": replicas,
            "matrices": matrices, "heavy": heavy, "acceptor": acceptor,
            "z": z, "n_residues": z.shape[0]}


def pair_features(z: np.ndarray, pairs: dict[int, pd.DataFrame]) -> dict[str, dict[int, np.ndarray]]:
    features: dict[str, dict[int, np.ndarray]] = {name: {} for name in (*SCALAR_METRICS, "signed_sum", "signed_mean", "vector")}
    total = z.sum(axis=0); mean = z.mean(axis=0)
    metric_names = {"absolute_l1": "l1", "l2": "l2", "cosine": "cosine", "correlation": "correlation"}
    for replica, frame in pairs.items():
        left = frame.left_frame.to_numpy(); right = frame.right_frame.to_numpy()
        features["signed_sum"][replica] = total[left] - total[right]
        features["signed_mean"][replica] = mean[left] - mean[right]
        for name, metric in metric_names.items():
            features[name][replica] = pf_pair_distance(z, left, right, metric)
        features["vector"][replica] = absolute_change_vectors(z, left, right)
    return features


def density_targets(data: dict, fit_replica: int, rank: int) -> tuple[dict[int, np.ndarray], float]:
    bandwidth = neighbour_bandwidth(data["matrices"][fit_replica][1], rank)
    global_log_density = np.empty(len(data["replicas"]), dtype=float)
    for replica, (indices, matrix) in data["matrices"].items():
        global_log_density[indices] = log_kernel_density(matrix, bandwidth)
    targets = {}
    for replica, frame in data["pairs"].items():
        left = frame.left_frame.to_numpy(); right = frame.right_frame.to_numpy()
        targets[replica] = global_log_density[left] - global_log_density[right]
    return targets, bandwidth


def fit_system(row: dict[str, str], config: dict, parts: Path) -> str:
    data = system_data(row, config); features = pair_features(data["z"], data["pairs"])
    rows = []
    for fit_replica in (1, 2, 3):
        frame = data["pairs"][fit_replica]
        fit_mask, validation_mask = frame_disjoint_pair_split(frame.left_frame.to_numpy(), frame.right_frame.to_numpy())
        for rank in NEIGHBOUR_RANKS:
            targets, bandwidth = density_targets(data, fit_replica, rank)
            signed_y = targets[fit_replica]; magnitude_y = np.abs(signed_y)
            alpha, numerator, denominator = scalar_scale(features["signed_sum"][fit_replica], signed_y, False)
            rows.append({"system_id": data["system"], "fit_replica": fit_replica, "rank": rank,
                         "model": "signed_sum", "alpha": alpha, "numerator": numerator,
                         "denominator": denominator, "bandwidth_angstrom": bandwidth,
                         "n_residues": data["n_residues"]})
            alpha, numerator, denominator = scalar_scale(features["signed_mean"][fit_replica], signed_y, False)
            rows.append({"system_id": data["system"], "fit_replica": fit_replica, "rank": rank,
                         "model": "signed_mean_global", "alpha": alpha, "numerator": numerator,
                         "denominator": denominator, "bandwidth_angstrom": bandwidth,
                         "n_residues": data["n_residues"]})
            for metric in SCALAR_METRICS:
                alpha, numerator, denominator = scalar_scale(features[metric][fit_replica], magnitude_y, True)
                rows.append({"system_id": data["system"], "fit_replica": fit_replica, "rank": rank,
                             "model": metric, "alpha": alpha, "numerator": numerator,
                             "denominator": denominator, "bandwidth_angstrom": bandwidth,
                             "n_residues": data["n_residues"]})
            ridge, alpha, loss = select_ridge_no_intercept(
                features["vector"][fit_replica], magnitude_y, fit_mask, validation_mask,
                config["analysis"]["pairwise_geometry"]["vector_audit"]["ridge_alphas"],
            )
            rows.append({"system_id": data["system"], "fit_replica": fit_replica, "rank": rank,
                         "model": "ridge", "alpha": alpha, "numerator": np.nan,
                         "denominator": np.nan, "bandwidth_angstrom": bandwidth,
                         "ridge_validation_loss": loss, "n_residues": data["n_residues"]})
    atomic_parquet(pd.DataFrame(rows), parts / f"{data['system']}.fits.parquet")
    return data["system"]


def global_scales(fits: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-system-out, equal-system-weighted scalar sufficient statistics."""
    eligible = fits[fits.model != "ridge"].copy()
    eligible["num_equal"] = eligible.numerator / eligible.denominator.replace(0, np.nan)
    # Averaging per-system slopes gives every system equal influence.
    groups = eligible.groupby(["fit_replica", "rank", "model"])
    total = groups.num_equal.transform("sum"); count = groups.num_equal.transform("count")
    eligible["global_alpha_loso"] = (total - eligible.num_equal) / np.maximum(count - 1, 1)
    if eligible.model.isin(SCALAR_METRICS).any():
        mask = eligible.model.isin(SCALAR_METRICS)
        eligible.loc[mask, "global_alpha_loso"] = np.maximum(0.0, eligible.loc[mask, "global_alpha_loso"])
    return eligible[["system_id", "fit_replica", "rank", "model", "global_alpha_loso"]]


def valid_table(path: Path, required: set[str]) -> bool:
    """Accept a resumable part only when it is readable, non-empty, and complete."""
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    return bool(len(frame) and required.issubset(frame.columns))


def mass_metrics(target: np.ndarray, prediction: np.ndarray, reference: np.ndarray, bins: int, smoothing: float) -> dict:
    low, high = np.quantile(reference, [0.005, 0.995])
    if not np.isfinite(low + high) or high <= low:
        low, high = float(np.min(reference) - 0.5), float(np.max(reference) + 0.5)
    internal = np.linspace(low, high, bins + 1)[1:-1]
    edges = np.concatenate(([-np.inf], internal, [np.inf]))
    return extended_mass_errors(np.histogram(prediction, edges)[0], np.histogram(target, edges)[0], smoothing)


def evaluate_system(row: dict[str, str], config: dict, fits: pd.DataFrame, global_fit: pd.DataFrame,
                    w1_edges: np.ndarray, rmsd_edges: np.ndarray, parts: Path) -> str:
    data = system_data(row, config); features = pair_features(data["z"], data["pairs"])
    system_fits = fits[fits.system_id == data["system"]]
    system_global = global_fit[global_fit.system_id == data["system"]]
    settings = config["analysis"]["pairwise_geometry"]; smoothing = settings["boundary_audit"]["distribution_smoothing"]
    bins = settings["boundary_audit"]["distribution_bins"]; coverage = settings["strict_conformal"]["coverage"]
    rows = []
    for fit_replica, calibration_replica, test_replica in ordered_assignments():
        for rank in NEIGHBOUR_RANKS:
            targets, bandwidth = density_targets(data, fit_replica, rank)
            target_sets = {"signed": targets, "magnitude": {r: np.abs(v) for r, v in targets.items()}}
            local = system_fits[(system_fits.fit_replica == fit_replica) & (system_fits["rank"] == rank)].set_index("model")
            global_rows = system_global[(system_global.fit_replica == fit_replica) & (system_global["rank"] == rank)].set_index("model")
            predictions: dict[str, tuple[str, dict[int, np.ndarray]]] = {
                "signed_sum_alpha1": ("signed", {r: features["signed_sum"][r] for r in (1, 2, 3)}),
                "signed_sum_local_alpha": ("signed", {r: local.loc["signed_sum", "alpha"] * features["signed_sum"][r] for r in (1, 2, 3)}),
                "signed_mean_global_alpha": ("signed", {r: global_rows.loc["signed_mean_global", "global_alpha_loso"] * features["signed_mean"][r] for r in (1, 2, 3)}),
            }
            for metric in SCALAR_METRICS:
                predictions[f"{metric}_alpha1"] = ("magnitude", {r: features[metric][r] for r in (1, 2, 3)})
                predictions[f"{metric}_local_alpha"] = ("magnitude", {r: local.loc[metric, "alpha"] * features[metric][r] for r in (1, 2, 3)})
                predictions[f"{metric}_global_alpha"] = ("magnitude", {r: global_rows.loc[metric, "global_alpha_loso"] * features[metric][r] for r in (1, 2, 3)})
            fit_frame = data["pairs"][fit_replica]
            fit_mask, validation_mask = frame_disjoint_pair_split(fit_frame.left_frame.to_numpy(), fit_frame.right_frame.to_numpy())
            ridge, ridge_alpha, _ = select_ridge_no_intercept(
                features["vector"][fit_replica], target_sets["magnitude"][fit_replica], fit_mask, validation_mask,
                settings["vector_audit"]["ridge_alphas"],
            )
            predictions["per_residue_ridge"] = ("magnitude", {
                r: np.maximum(0.0, ridge.predict(features["vector"][r])) for r in (1, 2, 3)
            })
            test_pairs = data["pairs"][test_replica]
            support = data["audit"][(data["audit"].fit_replica == fit_replica)
                                    & (data["audit"].calibration_replica == calibration_replica)
                                    & (data["audit"].test_replica == test_replica)
                                    & (data["audit"].target == "w1")].reset_index(drop=True).support_category.to_numpy()
            # The complete diagnostic grid is scientifically relevant for the primary bandwidth.
            # Sensitivity ranks need only the headline common-support W1 recovery; avoiding redundant
            # RMSD/stratum calculations cuts thousands of histogram comparisons per system.
            axes = {"w1": (test_pairs.w1.to_numpy(), w1_edges)}
            if rank == PRIMARY_RANK:
                axes["rmsd"] = (test_pairs.rmsd.to_numpy(), rmsd_edges)
            for model, (target_kind, model_predictions) in predictions.items():
                calibration_target = target_sets[target_kind][calibration_replica]
                calibration_prediction = model_predictions[calibration_replica]
                q = finite_conformal_quantile(np.abs(calibration_target - calibration_prediction), coverage)
                test_target = target_sets[target_kind][test_replica]; test_prediction = model_predictions[test_replica]
                for axis, (structural, edges) in axes.items():
                    labels = np.clip(np.digitize(structural, edges[1:-1]), 0, len(edges) - 2)
                    strata = (("all", np.ones(len(labels), bool)),
                               ("common_support", support == "common_support")) if rank == PRIMARY_RANK else (
                                   ("common_support", support == "common_support"),)
                    for band in range(len(edges) - 1):
                        for stratum, stratum_mask in strata:
                            mask = (labels == band) & stratum_mask
                            if not mask.any(): continue
                            lower = test_prediction[mask] - q; upper = test_prediction[mask] + q
                            rows.append({
                                "system_id": data["system"], "fit_replica": fit_replica,
                                "calibration_replica": calibration_replica, "test_replica": test_replica,
                                "rank": rank, "bandwidth_angstrom": bandwidth, "model": model,
                                "target_kind": target_kind, "structural_axis": axis, "band": f"q{band}",
                                "band_low_angstrom": float(edges[band]), "band_high_angstrom": float(edges[band + 1]),
                                "stratum": stratum, "pairs": int(mask.sum()),
                                "mae": float(np.mean(np.abs(test_target[mask] - test_prediction[mask]))),
                                "spearman": _finite_spearman(test_target[mask], test_prediction[mask]),
                                "coverage_90": float(np.mean((test_target[mask] >= lower) & (test_target[mask] <= upper))),
                                "mean_interval_score": float(np.mean(interval_score(test_target[mask], lower, upper))),
                                **mass_metrics(test_target[mask], test_prediction[mask], target_sets[target_kind][fit_replica], bins, smoothing),
                            })
    atomic_parquet(pd.DataFrame(rows), parts / f"{data['system']}.summary.parquet")
    return data["system"]


def _finite_spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2:
        return 0.0
    value = float(pd.Series(target).corr(pd.Series(prediction), method="spearman"))
    return value if np.isfinite(value) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2); parser.add_argument("--limit", type=int)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args(); config = load_config(); systems = load_systems()[:args.limit]
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint17_kde_population"
    fit_parts = output / "fit_parts"; summary_parts = output / "summary_parts"
    fit_parts.mkdir(parents=True, exist_ok=True); summary_parts.mkdir(parents=True, exist_ok=True)
    fit_required = {"system_id", "fit_replica", "rank", "model", "alpha", "bandwidth_angstrom"}
    pending = systems if args.restart else [
        r for r in systems
        if not valid_table(fit_parts / f"{r['system_id']}.fits.parquet", fit_required)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fit_system, row, config, fit_parts) for row in pending]
        for i, future in enumerate(as_completed(futures), 1): print(f"[fit {i}/{len(pending)}] {future.result()}", flush=True)
    fits = pd.concat([pd.read_parquet(fit_parts / f"{r['system_id']}.fits.parquet") for r in systems], ignore_index=True)
    global_fit = global_scales(fits)
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml") as handle:
        w1_edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    with open(HERE / "outputs/analysis/pairwise_geometry/checkpoint16_global_rmsd/global_rmsd_edges.yaml") as handle:
        rmsd_edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    summary_required = {"system_id", "fit_replica", "calibration_replica", "test_replica", "rank",
                        "model", "target_kind", "structural_axis", "band", "distribution_recovery"}
    pending = systems if args.restart else [
        r for r in systems
        if not valid_table(summary_parts / f"{r['system_id']}.summary.parquet", summary_required)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_system, row, config, fits, global_fit, w1_edges, rmsd_edges, summary_parts) for row in pending]
        for i, future in enumerate(as_completed(futures), 1): print(f"[eval {i}/{len(pending)}] {future.result()}", flush=True)
    summary = pd.concat([pd.read_parquet(summary_parts / f"{r['system_id']}.summary.parquet") for r in systems], ignore_index=True)
    fits.to_parquet(output / "kde_population_fits.parquet", index=False)
    global_fit.to_parquet(output / "kde_population_global_scales.parquet", index=False)
    summary.to_parquet(output / "kde_population_assignment_summary.parquet", index=False)
    atomic_yaml(output / "checkpoint17_run.yaml", {"checkpoint": 17, "systems": len(systems), "status": "measurement_complete",
                                                    "primary_neighbour_rank": PRIMARY_RANK})


if __name__ == "__main__":
    main()
