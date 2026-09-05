"""Checkpoint 26: structural-W1 graph geodesics weighted by PyRosetta energy."""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import spearmanr

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    feature_dir,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    density_targets,
    mass_metrics,
    neighbour_bandwidth,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import pair_endpoints
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_population_checkpoint24 import (
    load_score_frames,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    PAIR_CAP,
    sampled_indices,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint26_pyrosetta_graph"
CP24 = HERE / "outputs/analysis/pairwise_geometry/checkpoint24_pyrosetta_energy"
FIT, TUNE, TEST = 1, 2, 3
K_VALUES = (5, 10, 20)
GAMMAS = (0.25, 1.0, 4.0)
EPSILON = 1e-9
SEED = 20260905
FAMILIES = ("total_variation", "uphill_action", "energy_weighted_w1")
TARGET_KINDS = ("signed", "magnitude")


def robust_energy_scale(energy: np.ndarray) -> tuple[float, float]:
    center = float(np.median(energy))
    scale = float(1.4826 * np.median(np.abs(energy - center)))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(energy))
    return center, max(scale, np.finfo(float).eps)


def knn_edges(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return unique undirected edges from the union of directed kNN lists."""
    n = len(matrix)
    k = min(k, n - 1)
    nearest = np.argpartition(matrix, kth=k, axis=1)[:, : k + 1]
    edge_set: set[tuple[int, int]] = set()
    for left, neighbours in enumerate(nearest):
        for right in neighbours:
            right = int(right)
            if left != right:
                edge_set.add((min(left, right), max(left, right)))
    edges = np.asarray(sorted(edge_set), dtype=int)
    return edges[:, 0], edges[:, 1]


def sparse_graph(
    n: int,
    left: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    reverse: np.ndarray | None = None,
) -> csr_matrix:
    reverse = forward if reverse is None else reverse
    rows = np.concatenate([left, right])
    columns = np.concatenate([right, left])
    weights = np.maximum(np.concatenate([forward, reverse]), EPSILON)
    return csr_matrix((weights, (rows, columns)), shape=(n, n))


def graph_is_connected(graph: csr_matrix) -> bool:
    return connected_components(graph, directed=False, return_labels=False) == 1


def pair_shortest_paths(
    graph: csr_matrix,
    left: np.ndarray,
    right: np.ndarray,
    directed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    sources = np.unique(np.concatenate([left, right]))
    distances = dijkstra(graph, directed=directed, indices=sources)
    source_row = np.full(graph.shape[0], -1, dtype=int)
    source_row[sources] = np.arange(len(sources))
    forward = distances[source_row[left], right]
    reverse = distances[source_row[right], left]
    if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(reverse)):
        raise ValueError("non-finite shortest path for a sampled pair")
    return forward, reverse


def path_feature(
    matrix: np.ndarray,
    energy: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    k: int,
    family: str,
    gamma: float | None,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    edge_left, edge_right = knn_edges(matrix, k)
    length = matrix[edge_left, edge_right] / max(bandwidth, np.finfo(float).eps)
    delta = energy[edge_right] - energy[edge_left]
    if family == "total_variation":
        cost = np.abs(delta) + EPSILON * length
        graph = sparse_graph(len(matrix), edge_left, edge_right, cost)
        directed = False
    elif family == "uphill_action":
        if gamma is None:
            raise ValueError("uphill_action requires gamma")
        forward_cost = length + gamma * np.maximum(delta, 0.0)
        reverse_cost = length + gamma * np.maximum(-delta, 0.0)
        graph = sparse_graph(
            len(matrix), edge_left, edge_right, forward_cost, reverse_cost
        )
        directed = True
    elif family == "energy_weighted_w1":
        if gamma is None:
            raise ValueError("energy_weighted_w1 requires gamma")
        cost = length * (1.0 + gamma * np.abs(delta))
        graph = sparse_graph(len(matrix), edge_left, edge_right, cost)
        directed = False
    elif family == "geometry_only":
        graph = sparse_graph(len(matrix), edge_left, edge_right, length)
        directed = False
    else:
        raise ValueError(f"unknown path family: {family}")
    if not graph_is_connected(graph):
        raise ValueError(f"disconnected k={k} {family} graph")
    forward, reverse = pair_shortest_paths(
        graph, pair_left, pair_right, directed=directed
    )
    magnitude = 0.5 * (forward + reverse) if directed else forward
    sign = -np.sign(energy[pair_left] - energy[pair_right])
    return sign * magnitude, magnitude, graph.nnz // 2


def fitted_scale(feature: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.dot(feature, feature))
    if denominator <= 0:
        return 0.0
    return max(0.0, float(np.dot(feature, target) / denominator))


def finite_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return np.nan
    return float(spearmanr(left, right).statistic)


def graph_candidates(family: str) -> list[tuple[int, float | None]]:
    gammas = (None,) if family in {"total_variation", "geometry_only"} else GAMMAS
    return [(k, gamma) for k in K_VALUES for gamma in gammas]


def global_to_local(global_indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    lookup = np.full(int(global_indices.max()) + 1, -1, dtype=int)
    lookup[global_indices] = np.arange(len(global_indices))
    local = lookup[values]
    if np.any(local < 0):
        raise ValueError("pair endpoint is not present in its replica")
    return local


def candidate_features(
    data: dict,
    raw_energy: np.ndarray,
    pair_indices: dict[int, np.ndarray],
    shuffled: bool = False,
) -> tuple[dict[tuple[str, int, float | None], dict[int, tuple[np.ndarray, np.ndarray]]], list[dict]]:
    fit_global = data["matrices"][FIT][0]
    center, scale = robust_energy_scale(raw_energy[fit_global])
    normalized = (raw_energy - center) / scale
    bandwidth = neighbour_bandwidth(data["matrices"][FIT][1], PRIMARY_RANK)
    output = {}
    audits = []
    families = (*FAMILIES, "geometry_only")
    for replica, (global_indices, matrix) in data["matrices"].items():
        pair_frame = data["pairs"][replica]
        left_global, right_global = pair_endpoints(pair_frame, pair_indices[replica])
        left = global_to_local(global_indices, left_global)
        right = global_to_local(global_indices, right_global)
        energy = normalized[global_indices].copy()
        if shuffled:
            digest = hashlib.sha256(
                f"{SEED}:{data['system']}:{replica}:shuffle".encode()
            ).digest()
            np.random.default_rng(int.from_bytes(digest[:8], "little")).shuffle(energy)
        for family in families:
            for k, gamma in graph_candidates(family):
                key = (family, k, gamma)
                try:
                    signed, magnitude, edge_count = path_feature(
                        matrix, energy, left, right, k, family, gamma, bandwidth
                    )
                except ValueError as error:
                    audits.append(
                        {
                            "replica": replica,
                            "family": family,
                            "k": k,
                            "gamma": gamma,
                            "connected": False,
                            "error": str(error),
                        }
                    )
                    continue
                output.setdefault(key, {})[replica] = (signed, magnitude)
                audits.append(
                    {
                        "replica": replica,
                        "family": family,
                        "k": k,
                        "gamma": gamma,
                        "connected": True,
                        "edges": edge_count,
                    }
                )
    complete = {
        key: values for key, values in output.items() if set(values) == {FIT, TUNE, TEST}
    }
    return complete, audits


def select_candidate(
    features: dict,
    family: str,
    target_kind: str,
    targets: dict[int, np.ndarray],
) -> tuple[tuple[str, int, float | None], float, float]:
    position = 0 if target_kind == "signed" else 1
    scored = []
    for key, by_replica in features.items():
        if key[0] != family:
            continue
        alpha = fitted_scale(by_replica[FIT][position], targets[FIT])
        loss = float(
            np.mean(np.abs(targets[TUNE] - alpha * by_replica[TUNE][position]))
        )
        gamma_order = -1.0 if key[2] is None else float(key[2])
        scored.append((loss, key[1], gamma_order, key, alpha))
    if not scored:
        raise ValueError(f"no complete candidate for {family}")
    loss, _, _, key, alpha = min(scored)
    return key, alpha, loss


def direct_features(
    data: dict, raw_energy: np.ndarray, pair_indices: dict[int, np.ndarray]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for replica, pair_frame in data["pairs"].items():
        left, right = pair_endpoints(pair_frame, pair_indices[replica])
        delta = raw_energy[left] - raw_energy[right]
        result[replica] = (-delta, np.abs(delta))
    return result


def evaluate_system(row: dict, config: dict, edges: np.ndarray, shuffled: bool) -> tuple:
    started = time.perf_counter()
    data = system_data(row, config)
    system = data["system"]
    signed_target, bandwidth = density_targets(data, FIT, PRIMARY_RANK)
    pair_indices = {
        replica: sampled_indices(system, replica, len(values), PAIR_CAP)
        for replica, values in signed_target.items()
    }
    targets = {
        "signed": {r: v[pair_indices[r]] for r, v in signed_target.items()},
        "magnitude": {
            r: np.abs(v[pair_indices[r]]) for r, v in signed_target.items()
        },
    }
    energy = load_score_frames(system, config)["ref2015__total"]
    features, audit_rows = candidate_features(data, energy, pair_indices)
    shuffled_features = None
    if shuffled:
        shuffled_features, shuffled_audits = candidate_features(
            data, energy, pair_indices, shuffled=True
        )
        for item in shuffled_audits:
            item["shuffled"] = True
        audit_rows.extend(shuffled_audits)
    for item in audit_rows:
        item.update({"system_id": system, "shuffled": item.get("shuffled", False)})

    predictions: list[tuple[str, str, np.ndarray, float, dict]] = []
    fit_rows = []
    direct = direct_features(data, energy, pair_indices)
    for target_kind in TARGET_KINDS:
        position = 0 if target_kind == "signed" else 1
        alpha = fitted_scale(direct[FIT][position], targets[target_kind][FIT])
        predictions.append(
            (
                "direct_ref2015",
                target_kind,
                alpha * direct[TEST][position],
                alpha,
                {"k": np.nan, "gamma": np.nan, "tune_mae": float(np.mean(np.abs(targets[target_kind][TUNE] - alpha * direct[TUNE][position]))), "feature_variance": float(np.var(direct[TEST][position]))},
            )
        )
        for family in (*FAMILIES, "geometry_only"):
            key, alpha, tune_loss = select_candidate(
                features, family, target_kind, targets[target_kind]
            )
            predictions.append(
                (
                    family,
                    target_kind,
                    alpha * features[key][TEST][position],
                    alpha,
                    {"k": key[1], "gamma": key[2], "tune_mae": tune_loss, "feature_variance": float(np.var(features[key][TEST][position]))},
                )
            )
            if shuffled_features is not None and family in FAMILIES:
                shuffle_key, shuffle_alpha, shuffle_loss = select_candidate(
                    shuffled_features, family, target_kind, targets[target_kind]
                )
                predictions.append(
                    (
                        "shuffled_" + family,
                        target_kind,
                        shuffle_alpha * shuffled_features[shuffle_key][TEST][position],
                        shuffle_alpha,
                        {"k": shuffle_key[1], "gamma": shuffle_key[2], "tune_mae": shuffle_loss, "feature_variance": float(np.var(shuffled_features[shuffle_key][TEST][position]))},
                    )
                )

    work = thermodynamic_pair_features(data["z"], data["pairs"])
    for model in ("work_scale", "work_density_legacy_zq"):
        selected = {r: work[model][r][pair_indices[r]] for r in (FIT, TUNE, TEST)}
        alpha = fitted_scale(selected[FIT], targets["magnitude"][FIT])
        predictions.append(
            (
                model,
                "magnitude",
                alpha * selected[TEST],
                alpha,
                {"k": np.nan, "gamma": np.nan, "tune_mae": float(np.mean(np.abs(targets["magnitude"][TUNE] - alpha * selected[TUNE]))), "feature_variance": float(np.var(selected[TEST]))},
            )
        )

    test_pairs = data["pairs"][TEST]
    pair_left, pair_right = pair_endpoints(test_pairs, pair_indices[TEST])
    test_w1 = test_pairs.w1.to_numpy()[pair_indices[TEST]]
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    result_rows = []
    for model, target_kind, prediction, alpha, selected in predictions:
        target = targets[target_kind][TEST]
        if target_kind == "signed":
            nonzero = (target != 0) & (prediction != 0)
            sign_accuracy = float(np.mean(np.sign(target[nonzero]) == np.sign(prediction[nonzero]))) if nonzero.any() else np.nan
        else:
            sign_accuracy = np.nan
        fit_rows.append(
            {
                "system_id": system,
                "model": model,
                "target_kind": target_kind,
                "alpha": alpha,
                "feature_variance": selected["feature_variance"],
                "sign_accuracy": sign_accuracy,
                **selected,
            }
        )
        for band in range(6):
            mask = (test_w1 >= edges[band]) & (
                (test_w1 < edges[band + 1]) if band < 5 else True
            )
            unique_frames = len(np.unique(np.r_[pair_left[mask], pair_right[mask]]))
            if mask.sum() < 30 or unique_frames < 20:
                continue
            result_rows.append(
                {
                    "system_id": system,
                    "model": model,
                    "target_kind": target_kind,
                    "band": f"q{band}",
                    "pairs": int(mask.sum()),
                    "unique_frames": unique_frames,
                    "mae": float(np.mean(np.abs(target[mask] - prediction[mask]))),
                    "spearman": finite_spearman(target[mask], prediction[mask]),
                    "sign_accuracy": (
                        float(np.mean(np.sign(target[mask]) == np.sign(prediction[mask])))
                        if target_kind == "signed" else np.nan
                    ),
                    "bandwidth_angstrom": bandwidth,
                    **mass_metrics(
                        target[mask],
                        prediction[mask],
                        targets[target_kind][FIT],
                        settings["distribution_bins"],
                        settings["distribution_smoothing"],
                    ),
                }
            )
    runtime = {"system_id": system, "runtime_seconds": time.perf_counter() - started}
    return result_rows, fit_rows, audit_rows, runtime


def evaluate_task(arguments: tuple) -> tuple:
    return evaluate_system(*arguments)


def protein_sizes(rows: list[dict]) -> pd.DataFrame:
    values = []
    for row in rows:
        system = row["system_id"]
        with np.load(feature_dir(system, 1) / "features.npz", allow_pickle=False) as archive:
            values.append({"system_id": system, "n_residues": archive["heavy_contacts"].shape[0]})
    frame = pd.DataFrame(values)
    frame["quartile"] = pd.qcut(frame.n_residues.rank(method="first"), 4, labels=False)
    return frame


def deterministic_stratified(candidates: set[str], sizes: pd.DataFrame, count: int) -> list[str]:
    table = sizes[sizes.system_id.isin(candidates)].copy()
    table["order"] = table.system_id.map(
        lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()
    )
    chosen = []
    per = count // 4
    for quartile in range(4):
        chosen.extend(
            table[table.quartile == quartile].sort_values("order").system_id.head(per)
        )
    if len(chosen) < count:
        chosen.extend(
            table[~table.system_id.isin(chosen)]
            .sort_values(["order", "system_id"])
            .system_id.head(count - len(chosen))
        )
    return list(chosen)


def pilot_systems(rows: list[dict]) -> list[str]:
    path = CP24 / "full_single_assignment/pyrosetta_population_results.parquet"
    results = pd.read_parquet(path)
    selected = results[
        (results.model == "pyro_ref2015_total_fitted_abs")
        & (results.relation == "all")
    ]
    sizes = protein_sizes(rows)
    q0 = set(selected.loc[selected.band == "q0", "system_id"])
    q5 = set(selected.loc[selected.band == "q5", "system_id"])
    return deterministic_stratified(q0, sizes, 12) + deterministic_stratified(q5, sizes, 12)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["target_kind", "model", "band"], as_index=False)
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


def paired_tail(results: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for target_kind in TARGET_KINDS:
        block = results[results.target_kind == target_kind]
        for band in ("q0", "q5"):
            pivot = block[block.band == band].pivot_table(
                index="system_id", columns="model", values="distribution_recovery"
            )
            for model in FAMILIES:
                if model not in pivot or "direct_ref2015" not in pivot:
                    continue
                delta = (pivot[model] - pivot["direct_ref2015"]).dropna().to_numpy()
                boot = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
                rows.append(
                    {
                        "target_kind": target_kind,
                        "band": band,
                        "model": model,
                        "systems": len(delta),
                        "median_paired_improvement": float(np.median(delta)),
                        "mean_paired_improvement": float(np.mean(delta)),
                        "bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                        "bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def paired_median(
    results: pd.DataFrame,
    target_kind: str,
    band: str,
    model: str,
    baseline: str,
) -> tuple[float, int]:
    pivot = results[
        (results.target_kind == target_kind) & (results.band == band)
    ].pivot_table(index="system_id", columns="model", values="distribution_recovery")
    if model not in pivot or baseline not in pivot:
        return np.nan, 0
    difference = (pivot[model] - pivot[baseline]).dropna()
    return float(difference.median()), len(difference)


def pilot_gate(results: pd.DataFrame) -> dict:
    rows = []
    passed = False
    for target_kind in TARGET_KINDS:
        for family in FAMILIES:
            q0, q0_n = paired_median(
                results, target_kind, "q0", family, "direct_ref2015"
            )
            q5, q5_n = paired_median(
                results, target_kind, "q5", family, "direct_ref2015"
            )
            controls = {}
            for band in ("q0", "q5"):
                controls[band] = {
                    "versus_geometry": paired_median(
                        results, target_kind, band, family, "geometry_only"
                    )[0],
                    "versus_shuffled": paired_median(
                        results, target_kind, band, family, "shuffled_" + family
                    )[0],
                }
            family_pass = False
            for improved_band, other_value, gain in (("q0", q5, q0), ("q5", q0, q5)):
                control = controls[improved_band]
                if (
                    np.isfinite(gain)
                    and gain >= 0.02
                    and other_value >= -0.02
                    and control["versus_geometry"] > 0
                    and control["versus_shuffled"] > 0
                ):
                    family_pass = True
            passed |= family_pass
            rows.append(
                {
                    "target_kind": target_kind,
                    "family": family,
                    "q0_median_paired_gain": q0,
                    "q5_median_paired_gain": q5,
                    "q0_systems": q0_n,
                    "q5_systems": q5_n,
                    "q0_controls": controls["q0"],
                    "q5_controls": controls["q5"],
                    "passed": family_pass,
                }
            )
    return {
        "passed": passed,
        "thresholds": {
            "minimum_improved_band_gain": 0.02,
            "maximum_other_extreme_loss": 0.02,
            "must_beat_geometry_and_shuffled_controls": True,
        },
        "models": rows,
        "decision": "continue_full" if passed else "stop_after_pilot",
    }


def alpha_variance(fits: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    rows = []
    selected = fits[~fits.model.str.startswith("shuffled_")]
    for (target_kind, model), block in selected.groupby(["target_kind", "model"]):
        finite = block[
            np.isfinite(block.alpha)
            & np.isfinite(block.feature_variance)
            & (block.feature_variance > 0)
        ]
        if len(finite) < 3 or np.ptp(finite.alpha) == 0:
            rho, low, high = np.nan, np.nan, np.nan
        else:
            x = finite.feature_variance.to_numpy()
            y = finite.alpha.to_numpy()
            rho = finite_spearman(x, y)
            boot = []
            for _ in range(10_000):
                take = rng.integers(0, len(x), len(x))
                value = finite_spearman(x[take], y[take])
                if np.isfinite(value):
                    boot.append(value)
            low, high = np.quantile(boot, [0.025, 0.975]) if boot else (np.nan, np.nan)
        rows.append(
            {
                "target_kind": target_kind,
                "model": model,
                "systems": len(finite),
                "zero_alphas": int((finite.alpha == 0).sum()),
                "spearman_rho": rho,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def plot_alpha_variance(fits: pd.DataFrame, stats: pd.DataFrame, destination: Path) -> None:
    models = ["direct_ref2015", *FAMILIES, "geometry_only"]
    fig, axes = plt.subplots(2, len(models), figsize=(20, 8))
    for row_index, target_kind in enumerate(TARGET_KINDS):
        for axis, model in zip(axes[row_index], models):
            selected = fits[(fits.target_kind == target_kind) & (fits.model == model)]
            axis.scatter(selected.feature_variance, selected.alpha, s=25, alpha=0.75)
            positive_x = selected.feature_variance > 0
            positive_y = selected.alpha > 0
            if positive_x.all():
                axis.set_xscale("log")
            if positive_y.all():
                axis.set_yscale("log")
            result = stats[
                (stats.target_kind == target_kind) & (stats.model == model)
            ]
            if len(result):
                value = result.iloc[0]
                axis.set_title(f"{model.replace('_', ' ')}\nρ={value.spearman_rho:.2f}")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("path-feature variance")
            if axis is axes[row_index, 0]:
                axis.set_ylabel(f"{target_kind} fitted α")
    fig.suptitle("Graph-feature variance versus fitted system α")
    fig.tight_layout()
    fig.savefig(destination / "graph_alpha_vs_variance.png", dpi=180)
    plt.close(fig)


def plot_recovery(summary: pd.DataFrame, edges: np.ndarray, destination: Path) -> None:
    order = [
        "direct_ref2015",
        *FAMILIES,
        "geometry_only",
        "work_scale",
        "work_density_legacy_zq",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    for axis, target_kind in zip(axes, TARGET_KINDS):
        for model in order:
            selected = summary[
                (summary.target_kind == target_kind) & (summary.model == model)
            ].set_index("band")
            if selected.empty:
                continue
            values = np.array([selected.recovery.get(f"q{i}", np.nan) for i in range(6)])
            spread = np.array([selected.recovery_sd.get(f"q{i}", np.nan) for i in range(6)])
            axis.plot(range(6), 100 * values, marker="o", label=model.replace("_", " "))
            axis.fill_between(
                range(6),
                np.clip(100 * (values - spread), 0, 100),
                np.clip(100 * (values + spread), 0, 100),
                alpha=0.06,
            )
        axis.set_title(f"{target_kind.title()} log-density change")
        axis.set_xticks(
            range(6),
            [f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å" for i in range(6)],
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel(r"Recovery, $100(1-\sqrt{JSD})$ (%)")
    fig.suptitle("PyRosetta energy geodesics on structural-W1 graphs")
    fig.tight_layout()
    fig.savefig(destination / "graph_recovery_global_w1.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    config = load_config()
    all_rows = load_systems()
    chosen = pilot_systems(all_rows)
    rows = all_rows if args.full else [r for r in all_rows if r["system_id"] in chosen]
    destination = OUTPUT / ("full" if args.full else "pilot")
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    with (HERE / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml").open() as handle:
        edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    atomic_parquet(
        protein_sizes(all_rows).assign(
            pilot=lambda frame: frame.system_id.isin(chosen),
            pilot_band=lambda frame: np.where(frame.system_id.isin(chosen[:12]), "q0", np.where(frame.system_id.isin(chosen[12:]), "q5", "none")),
        ),
        OUTPUT / "pilot_systems.parquet",
    )
    tasks = []
    for row in rows:
        prefix = parts / row["system_id"]
        required = [prefix.with_suffix(f".{name}.parquet") for name in ("results", "fits", "audits", "runtime")]
        if not args.force and all(path.exists() for path in required):
            continue
        tasks.append((row, config, edges, not args.full))
    if tasks:
        executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        evaluated = executor.map(evaluate_task, tasks) if executor else map(evaluate_task, tasks)
        for index, (task, result) in enumerate(zip(tasks, evaluated), 1):
            system = task[0]["system_id"]
            for name, records in zip(("results", "fits", "audits", "runtime"), result):
                records = records if isinstance(records, list) else [records]
                atomic_parquet(pd.DataFrame(records), parts / f"{system}.{name}.parquet")
            print(f"[{index}/{len(tasks)}] {system}", flush=True)
        if executor:
            executor.shutdown()
    tables = {}
    for name in ("results", "fits", "audits", "runtime"):
        tables[name] = pd.concat(
            [pd.read_parquet(parts / f"{row['system_id']}.{name}.parquet") for row in rows],
            ignore_index=True,
        )
        atomic_parquet(tables[name], destination / f"graph_{name}.parquet")
    summary = summarize(tables["results"])
    tail = paired_tail(tables["results"])
    alpha_stats = alpha_variance(tables["fits"])
    atomic_parquet(summary, destination / "graph_summary.parquet")
    atomic_parquet(tail, destination / "graph_tail_comparisons.parquet")
    atomic_parquet(alpha_stats, destination / "graph_alpha_variance.parquet")
    plot_recovery(summary, edges, destination)
    plot_alpha_variance(tables["fits"], alpha_stats, destination)
    gate = pilot_gate(tables["results"]) if not args.full else None
    if gate is not None:
        atomic_yaml(destination / "pilot_gate.yaml", gate)
    atomic_yaml(
        destination / "checkpoint26_report.yaml",
        {
            "checkpoint": 26,
            "systems": len(rows),
            "assignment": "A-fit/B-tune/C-test",
            "score": "PyRosetta ref2015 total",
            "graph": "replica-specific symmetrized structural-W1 kNN",
            "k_values": list(K_VALUES),
            "gammas": list(GAMMAS),
            "target": "rank-10 structural-W1 KDE log-density difference",
            "median_runtime_seconds": float(tables["runtime"].runtime_seconds.median()),
            "pilot_gate": gate,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run the fixed 24-system pilot (default).")
    parser.add_argument("--full", action="store_true", help="Run all 111 systems.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.pilot and args.full:
        parser.error("choose only one of --pilot or --full")
    run(args)


if __name__ == "__main__":
    main()
