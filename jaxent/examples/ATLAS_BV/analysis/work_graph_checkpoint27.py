"""Checkpoint 27: shortest paths on structural-W1 graphs weighted by Work metrics."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config, load_systems
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    density_targets,
    mass_metrics,
    neighbour_bandwidth,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import pair_endpoints
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_graph_checkpoint26 import (
    EPSILON,
    FIT,
    GAMMAS,
    K_VALUES,
    TEST,
    TUNE,
    finite_spearman,
    fitted_scale,
    global_to_local,
    graph_is_connected,
    knn_edges,
    pair_shortest_paths,
    sparse_graph,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_frame_features,
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint27_work_graph"
METRICS = ("work_scale", "work_shape", "work_density_legacy_zq")
PATH_FAMILIES = ("accumulated", "additive_w1", "weighted_w1")


def edge_metric(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return np.abs(values[left] - values[right])
    return np.mean(np.abs(values[:, left] - values[:, right]), axis=0)


def path_candidate(
    matrix: np.ndarray,
    values: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    k: int,
    family: str,
    gamma: float | None,
    bandwidth: float,
    metric_scale: float,
) -> tuple[np.ndarray, int]:
    left, right = knn_edges(matrix, k)
    length = matrix[left, right] / max(bandwidth, np.finfo(float).eps)
    difference = edge_metric(values, left, right) / max(metric_scale, np.finfo(float).eps)
    if family == "accumulated":
        cost = difference + EPSILON * length
    elif family == "additive_w1":
        if gamma is None:
            raise ValueError("additive_w1 requires gamma")
        cost = length + gamma * difference
    elif family == "weighted_w1":
        if gamma is None:
            raise ValueError("weighted_w1 requires gamma")
        cost = length * (1.0 + gamma * difference)
    elif family == "geometry_only":
        cost = length
    else:
        raise ValueError(f"unknown path family: {family}")
    graph = sparse_graph(len(matrix), left, right, cost)
    if not graph_is_connected(graph):
        raise ValueError(f"disconnected k={k} {family} graph")
    forward, _ = pair_shortest_paths(graph, pair_left, pair_right, directed=False)
    return forward, graph.nnz // 2


def candidates(family: str) -> list[tuple[int, float | None]]:
    gammas = (None,) if family in {"accumulated", "geometry_only"} else GAMMAS
    return [(k, gamma) for k in K_VALUES for gamma in gammas]


def build_features(
    data: dict,
    pair_indices: dict[int, np.ndarray],
) -> tuple[dict[tuple[str, str, int, float | None], dict[int, np.ndarray]], list[dict]]:
    frames = thermodynamic_frame_features(data["z"])
    direct = thermodynamic_pair_features(data["z"], data["pairs"])
    bandwidth = neighbour_bandwidth(data["matrices"][FIT][1], PRIMARY_RANK)
    scales = {}
    for metric in METRICS:
        selected = direct[metric][FIT][pair_indices[FIT]]
        positive = selected[selected > 0]
        scales[metric] = float(np.median(positive)) if len(positive) else 1.0
    output = {}
    audits = []
    for replica, (global_indices, matrix) in data["matrices"].items():
        pair_frame = data["pairs"][replica]
        left_global, right_global = pair_endpoints(pair_frame, pair_indices[replica])
        left = global_to_local(global_indices, left_global)
        right = global_to_local(global_indices, right_global)
        for metric in METRICS:
            frame_values = frames[metric]
            local_values = (
                frame_values[global_indices]
                if frame_values.ndim == 1
                else frame_values[:, global_indices]
            )
            for family in (*PATH_FAMILIES, "geometry_only"):
                for k, gamma in candidates(family):
                    key = (metric, family, k, gamma)
                    try:
                        feature, edges = path_candidate(
                            matrix,
                            local_values,
                            left,
                            right,
                            k,
                            family,
                            gamma,
                            bandwidth,
                            scales[metric],
                        )
                    except ValueError as error:
                        audits.append(
                            {
                                "replica": replica,
                                "metric": metric,
                                "family": family,
                                "k": k,
                                "gamma": gamma,
                                "connected": False,
                                "error": str(error),
                            }
                        )
                        continue
                    output.setdefault(key, {})[replica] = feature
                    audits.append(
                        {
                            "replica": replica,
                            "metric": metric,
                            "family": family,
                            "k": k,
                            "gamma": gamma,
                            "connected": True,
                            "edges": edges,
                        }
                    )
    return {key: value for key, value in output.items() if set(value) == {FIT, TUNE, TEST}}, audits


def choose(
    features: dict,
    metric: str,
    family: str,
    targets: dict[int, np.ndarray],
) -> tuple[tuple, float, float]:
    scored = []
    for key, values in features.items():
        if key[:2] != (metric, family):
            continue
        alpha = fitted_scale(values[FIT], targets[FIT])
        loss = float(np.mean(np.abs(targets[TUNE] - alpha * values[TUNE])))
        gamma_order = -1.0 if key[3] is None else float(key[3])
        scored.append((loss, key[2], gamma_order, key, alpha))
    if not scored:
        raise ValueError(f"no connected candidate for {metric} {family}")
    loss, _, _, key, alpha = min(scored)
    return key, alpha, loss


def evaluate_system(row: dict, config: dict, w1_edges: np.ndarray) -> tuple:
    started = time.perf_counter()
    data = system_data(row, config)
    system = data["system"]
    signed_targets, bandwidth = density_targets(data, FIT, PRIMARY_RANK)
    pair_indices = {
        replica: sampled_indices(system, replica, len(target), PAIR_CAP)
        for replica, target in signed_targets.items()
    }
    magnitude_targets = {
        replica: np.abs(target[pair_indices[replica]])
        for replica, target in signed_targets.items()
    }
    signed_selected = {
        replica: target[pair_indices[replica]]
        for replica, target in signed_targets.items()
    }
    direct = thermodynamic_pair_features(data["z"], data["pairs"])
    direct = {
        metric: {r: values[r][pair_indices[r]] for r in (FIT, TUNE, TEST)}
        for metric, values in direct.items()
        if metric in METRICS
    }
    features, audit_rows = build_features(data, pair_indices)
    for item in audit_rows:
        item["system_id"] = system
    predictions = []
    fit_rows = []
    for metric in METRICS:
        alpha = fitted_scale(direct[metric][FIT], magnitude_targets[FIT])
        predictions.append((metric, "direct", "magnitude", alpha * direct[metric][TEST]))
        fit_rows.append(
            {
                "system_id": system,
                "metric": metric,
                "family": "direct",
                "target_kind": "magnitude",
                "alpha": alpha,
                "k": np.nan,
                "gamma": np.nan,
                "tune_mae": float(np.mean(np.abs(magnitude_targets[TUNE] - alpha * direct[metric][TUNE]))),
                "feature_variance": float(np.var(direct[metric][TEST])),
            }
        )
        for family in (*PATH_FAMILIES, "geometry_only"):
            key, alpha, tune_loss = choose(features, metric, family, magnitude_targets)
            predictions.append((metric, family, "magnitude", alpha * features[key][TEST]))
            fit_rows.append(
                {
                    "system_id": system,
                    "metric": metric,
                    "family": family,
                    "target_kind": "magnitude",
                    "alpha": alpha,
                    "k": key[2],
                    "gamma": key[3],
                    "tune_mae": tune_loss,
                    "feature_variance": float(np.var(features[key][TEST])),
                }
            )

    frame_scale = thermodynamic_frame_features(data["z"])["work_scale"]
    signed_direct = {}
    for replica, pair_frame in data["pairs"].items():
        left, right = pair_endpoints(pair_frame, pair_indices[replica])
        signed_direct[replica] = frame_scale[left] - frame_scale[right]
    signed_alpha = fitted_scale(signed_direct[FIT], signed_selected[FIT])
    predictions.append(("work_scale", "direct", "signed", signed_alpha * signed_direct[TEST]))
    fit_rows.append(
        {
            "system_id": system,
            "metric": "work_scale",
            "family": "direct",
            "target_kind": "signed",
            "alpha": signed_alpha,
            "k": np.nan,
            "gamma": np.nan,
            "tune_mae": float(np.mean(np.abs(signed_selected[TUNE] - signed_alpha * signed_direct[TUNE]))),
            "feature_variance": float(np.var(signed_direct[TEST])),
        }
    )
    for family in (*PATH_FAMILIES, "geometry_only"):
        signed_features = {}
        for key, values in features.items():
            if key[:2] != ("work_scale", family):
                continue
            converted = {}
            for replica, magnitude in values.items():
                pair_frame = data["pairs"][replica]
                left, right = pair_endpoints(pair_frame, pair_indices[replica])
                converted[replica] = np.sign(frame_scale[left] - frame_scale[right]) * magnitude
            signed_features[key] = converted
        key, alpha, tune_loss = choose(signed_features, "work_scale", family, signed_selected)
        predictions.append(("work_scale", family, "signed", alpha * signed_features[key][TEST]))
        fit_rows.append(
            {
                "system_id": system,
                "metric": "work_scale",
                "family": family,
                "target_kind": "signed",
                "alpha": alpha,
                "k": key[2],
                "gamma": key[3],
                "tune_mae": tune_loss,
                "feature_variance": float(np.var(signed_features[key][TEST])),
            }
        )

    test_pairs = data["pairs"][TEST]
    pair_left, pair_right = pair_endpoints(test_pairs, pair_indices[TEST])
    test_w1 = test_pairs.w1.to_numpy()[pair_indices[TEST]]
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    result_rows = []
    for metric, family, target_kind, prediction in predictions:
        target = magnitude_targets[TEST] if target_kind == "magnitude" else signed_selected[TEST]
        for band in range(6):
            mask = (test_w1 >= w1_edges[band]) & (
                (test_w1 < w1_edges[band + 1]) if band < 5 else True
            )
            unique_frames = len(np.unique(np.r_[pair_left[mask], pair_right[mask]]))
            if mask.sum() < 30 or unique_frames < 20:
                continue
            result_rows.append(
                {
                    "system_id": system,
                    "metric": metric,
                    "family": family,
                    "model": f"{metric}_{family}",
                    "target_kind": target_kind,
                    "band": f"q{band}",
                    "pairs": int(mask.sum()),
                    "unique_frames": unique_frames,
                    "bandwidth_angstrom": bandwidth,
                    "mae": float(np.mean(np.abs(target[mask] - prediction[mask]))),
                    "spearman": finite_spearman(target[mask], prediction[mask]),
                    "sign_accuracy": (
                        float(np.mean(np.sign(target[mask]) == np.sign(prediction[mask])))
                        if target_kind == "signed" else np.nan
                    ),
                    **mass_metrics(
                        target[mask],
                        prediction[mask],
                        magnitude_targets[FIT] if target_kind == "magnitude" else signed_selected[FIT],
                        settings["distribution_bins"],
                        settings["distribution_smoothing"],
                    ),
                }
            )
    return result_rows, fit_rows, audit_rows, {
        "system_id": system,
        "runtime_seconds": time.perf_counter() - started,
    }


def evaluate_task(arguments: tuple) -> tuple:
    return evaluate_system(*arguments)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["target_kind", "metric", "family", "model", "band"], as_index=False)
        .agg(
            recovery=("distribution_recovery", "median"),
            recovery_sd=("distribution_recovery", "std"),
            mae=("mae", "median"),
            spearman=("spearman", "median"),
            sign_accuracy=("sign_accuracy", "median"),
            systems=("system_id", "nunique"),
            pairs=("pairs", "sum"),
        )
    )


def tail_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260906)
    rows = []
    for target_kind in results.target_kind.unique():
        block = results[results.target_kind == target_kind]
        for metric in block.metric.unique():
            metric_rows = block[block.metric == metric]
            for band in ("q0", "q5"):
                pivot = metric_rows[metric_rows.band == band].pivot_table(
                    index="system_id", columns="family", values="distribution_recovery"
                )
                for family in PATH_FAMILIES:
                    if family not in pivot or "direct" not in pivot:
                        continue
                    delta = (pivot[family] - pivot.direct).dropna().to_numpy()
                    boot = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
                    rows.append(
                        {
                            "target_kind": target_kind,
                            "metric": metric,
                            "family": family,
                            "band": band,
                            "systems": len(delta),
                            "median_paired_improvement": float(np.median(delta)),
                            "mean_paired_improvement": float(np.mean(delta)),
                            "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                            "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                        }
                    )
    return pd.DataFrame(rows)


def plot_recovery(summary: pd.DataFrame, edges: np.ndarray, destination) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
    for axis, metric in zip(axes, METRICS):
        for family in ("direct", *PATH_FAMILIES, "geometry_only"):
            selected = summary[
                (summary.target_kind == "magnitude")
                & (summary.metric == metric)
                & (summary.family == family)
            ].set_index("band")
            values = np.array([selected.recovery.get(f"q{i}", np.nan) for i in range(6)])
            spread = np.array([selected.recovery_sd.get(f"q{i}", np.nan) for i in range(6)])
            axis.plot(range(6), 100 * values, marker="o", label=family.replace("_", " "))
            axis.fill_between(
                range(6),
                np.clip(100 * (values - spread), 0, 100),
                np.clip(100 * (values + spread), 0, 100),
                alpha=0.06,
            )
        axis.set_title(metric.replace("_", " ").title())
        axis.set_xticks(
            range(6),
            [f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å" for i in range(6)],
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"Magnitude recovery, $100(1-\sqrt{JSD})$ (%)")
    fig.suptitle("Work-metric geodesics on structural-W1 graphs")
    fig.tight_layout()
    fig.savefig(destination / "work_graph_recovery_global_w1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parts = OUTPUT / "parts"
    parts.mkdir(exist_ok=True)
    with (HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml").open() as handle:
        edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    tasks = []
    for row in rows:
        expected = [parts / f"{row['system_id']}.{name}.parquet" for name in ("results", "fits", "audits", "runtime")]
        if args.force or not all(path.exists() for path in expected):
            tasks.append((row, config, edges))
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    evaluated = executor.map(evaluate_task, tasks) if executor else map(evaluate_task, tasks)
    for index, (task, result) in enumerate(zip(tasks, evaluated), 1):
        system = task[0]["system_id"]
        for name, records in zip(("results", "fits", "audits", "runtime"), result):
            atomic_parquet(
                pd.DataFrame(records if isinstance(records, list) else [records]),
                parts / f"{system}.{name}.parquet",
            )
        print(f"[{index}/{len(tasks)}] {system}", flush=True)
    if executor:
        executor.shutdown()
    combined = {}
    for name in ("results", "fits", "audits", "runtime"):
        combined[name] = pd.concat(
            [pd.read_parquet(parts / f"{row['system_id']}.{name}.parquet") for row in rows],
            ignore_index=True,
        )
        atomic_parquet(combined[name], OUTPUT / f"work_graph_{name}.parquet")
    summary = summarize(combined["results"])
    tail = tail_comparisons(combined["results"])
    atomic_parquet(summary, OUTPUT / "work_graph_summary.parquet")
    atomic_parquet(tail, OUTPUT / "work_graph_tail_comparisons.parquet")
    plot_recovery(summary, edges, OUTPUT)
    atomic_yaml(
        OUTPUT / "checkpoint27_report.yaml",
        {
            "checkpoint": 27,
            "systems": len(rows),
            "assignment": "A-fit/B-tune/C-test",
            "metrics": list(METRICS),
            "families": list(PATH_FAMILIES),
            "k_values": list(K_VALUES),
            "gammas": list(GAMMAS),
            "median_runtime_seconds": float(combined["runtime"].runtime_seconds.median()),
        },
    )


if __name__ == "__main__":
    main()
