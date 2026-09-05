"""Checkpoint 29: graph geodesics coupled to local Work-Scale variance."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.distance import cdist

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    density_targets,
    mass_metrics,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.local_variance_checkpoint28 import (
    fitted_signed_scale,
    local_statistics,
    nearest,
)
from jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 import (
    pair_endpoints,
)
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_graph_checkpoint26 import (
    FIT,
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
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)

OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint29_variance_graph"
K_VALUES = (5, 10, 20, 50)
SHRINKAGES = (0.001, 0.01, 0.1)
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
TOPOLOGIES = ("work3", "work_scale", "structural_w1")
FAMILIES = (
    "energy_only",
    "variance_only",
    "energy_variance",
    "scaled_energy_variance",
    "directed_basin",
)
EPSILON = 1e-9
SEED = 20260905


def robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0)
    scale = np.where(
        np.isfinite(scale) & (scale > np.finfo(float).eps), scale, fallback
    )
    return center, np.where(scale > np.finfo(float).eps, scale, 1.0)


def topology_matrices(
    data: dict, frame_features: dict
) -> dict[str, dict[int, np.ndarray]]:
    fit_indices = data["matrices"][FIT][0]

    def scalar(name: str) -> np.ndarray:
        values = np.asarray(frame_features[name])
        return values if values.ndim == 1 else np.mean(values, axis=0)

    work = np.column_stack(
        [
            scalar(name)
            for name in ("work_scale", "work_shape", "work_density_legacy_zq")
        ]
    )
    center, scale = robust_location_scale(work[fit_indices])
    normalized = (work - center) / scale
    output = {name: {} for name in TOPOLOGIES}
    for replica, (indices, structural) in data["matrices"].items():
        output["work3"][replica] = cdist(normalized[indices], normalized[indices])
        ws = normalized[indices, 0]
        output["work_scale"][replica] = np.abs(ws[:, None] - ws[None, :])
        output["structural_w1"][replica] = structural
    return output


def positive_median(values: np.ndarray) -> float:
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def graph_feature(
    matrix: np.ndarray,
    energy: np.ndarray,
    variance: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    k: int,
    family: str,
    coupling: float,
    energy_scale: float,
    variance_scale: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    left, right = knn_edges(matrix, k)
    length = matrix[left, right] / positive_median(matrix[left, right])
    delta_energy = energy[right] - energy[left]
    volume = 0.5 * np.log(variance)
    delta_volume = volume[right] - volume[left]
    energy_term = np.abs(delta_energy) / energy_scale
    variance_term = np.abs(delta_volume) / variance_scale
    directed = family == "directed_basin"
    if family == "energy_only":
        forward_cost = energy_term + EPSILON * length
        reverse_cost = None
    elif family == "variance_only":
        forward_cost = variance_term + EPSILON * length
        reverse_cost = None
    elif family == "energy_variance":
        forward_cost = energy_term + coupling * variance_term + EPSILON * length
        reverse_cost = None
    elif family == "scaled_energy_variance":
        pooled_sd = np.sqrt(0.5 * (variance[left] + variance[right]))
        forward_cost = np.abs(delta_energy) / np.maximum(pooled_sd, np.finfo(float).eps)
        forward_cost += coupling * variance_term + EPSILON * length
        reverse_cost = None
    elif directed:
        forward_cost = np.maximum(delta_energy, 0) / energy_scale
        forward_cost += (
            coupling * np.maximum(delta_volume, 0) / variance_scale + EPSILON * length
        )
        reverse_cost = np.maximum(-delta_energy, 0) / energy_scale
        reverse_cost += (
            coupling * np.maximum(-delta_volume, 0) / variance_scale + EPSILON * length
        )
    else:
        raise ValueError(f"unknown family {family}")
    graph = sparse_graph(len(matrix), left, right, forward_cost, reverse_cost)
    if not graph_is_connected(graph):
        raise ValueError("disconnected graph")
    forward, reverse = pair_shortest_paths(
        graph, pair_left, pair_right, directed=directed
    )
    magnitude = 0.5 * (forward + reverse) if directed else forward
    signed = (
        forward - reverse
        if directed
        else np.sign(variance[pair_right] - variance[pair_left]) * magnitude
    )
    return (
        magnitude,
        signed,
        {
            "edges": graph.nnz if directed else graph.nnz // 2,
            "mean_hop_proxy": float(np.mean(magnitude)),
        },
    )


def candidate_grid(family: str):
    couplings = (
        LAMBDAS
        if family in {"energy_variance", "scaled_energy_variance", "directed_basin"}
        else (0.0,)
    )
    for k in K_VALUES:
        for shrinkage in SHRINKAGES:
            for coupling in couplings:
                yield k, shrinkage, coupling


def evaluate_system(row: dict, config: dict, edges: np.ndarray):
    started = time.perf_counter()
    data = system_data(row, config)
    system = data["system"]
    signed_target, bandwidth = density_targets(data, FIT, PRIMARY_RANK)
    take = {
        r: sampled_indices(system, r, len(v), PAIR_CAP)
        for r, v in signed_target.items()
    }
    targets = {
        "signed": {r: signed_target[r][take[r]] for r in (FIT, TUNE, TEST)},
        "magnitude": {r: np.abs(signed_target[r][take[r]]) for r in (FIT, TUNE, TEST)},
    }
    frames = thermodynamic_frame_features(data["z"])
    energy_all = np.asarray(frames["work_scale"])
    matrices = topology_matrices(data, frames)
    endpoints = {}
    local_endpoints = {}
    energy = {}
    for replica in (FIT, TUNE, TEST):
        endpoints[replica] = pair_endpoints(data["pairs"][replica], take[replica])
        indices = data["matrices"][replica][0]
        local_endpoints[replica] = tuple(
            global_to_local(indices, x) for x in endpoints[replica]
        )
        energy[replica] = energy_all[indices]
    direct = {
        r: energy[r][local_endpoints[r][0]] - energy[r][local_endpoints[r][1]]
        for r in (FIT, TUNE, TEST)
    }
    direct_magnitude = {r: np.abs(v) for r, v in direct.items()}
    predictions = []
    fits = []
    alpha = fitted_scale(direct_magnitude[FIT], targets["magnitude"][FIT])
    predictions.append(("direct", "magnitude", alpha * direct_magnitude[TEST]))
    fits.append(
        {
            "system_id": system,
            "model": "direct",
            "target_kind": "magnitude",
            "alpha": alpha,
        }
    )
    signed_alpha = fitted_signed_scale(direct[FIT], targets["signed"][FIT])
    predictions.append(("direct", "signed", signed_alpha * direct[TEST]))
    fits.append(
        {
            "system_id": system,
            "model": "direct",
            "target_kind": "signed",
            "alpha": signed_alpha,
        }
    )
    audit = []
    for topology in TOPOLOGIES:
        neighbourhood_scores = []
        for k in K_VALUES:
            for shrinkage in SHRINKAGES:
                variance_features = {}
                reference = None
                for replica in (FIT, TUNE, TEST):
                    neighbours = nearest(matrices[topology][replica], k)
                    _, variance_matrix, reference = local_statistics(
                        energy[replica],
                        neighbours,
                        reference if replica != FIT else None,
                        shrinkage,
                    )
                    variance = variance_matrix[0]
                    left, right = local_endpoints[replica]
                    variance_features[replica] = np.abs(
                        0.5 * np.log(variance[right] / variance[left])
                    )
                variance_alpha = fitted_scale(
                    variance_features[FIT], targets["magnitude"][FIT]
                )
                loss = float(
                    np.mean(
                        np.abs(
                            targets["magnitude"][TUNE]
                            - variance_alpha * variance_features[TUNE]
                        )
                    )
                )
                neighbourhood_scores.append((loss, k, shrinkage))
        _, selected_k, selected_shrinkage = min(neighbourhood_scores)

        for family in FAMILIES:
            scored = []
            k, shrinkage = selected_k, selected_shrinkage
            couplings = (
                LAMBDAS
                if family
                in {"energy_variance", "scaled_energy_variance", "directed_basin"}
                else (0.0,)
            )
            for coupling in couplings:
                feature = {}
                signed_feature = {}
                valid = True
                fit_matrix = matrices[topology][FIT]
                fit_neighbours = nearest(fit_matrix, k)
                _, fit_var_matrix, reference = local_statistics(
                    energy[FIT], fit_neighbours, None, shrinkage
                )
                fit_left, fit_right = knn_edges(fit_matrix, k)
                e_scale = positive_median(
                    np.abs(energy[FIT][fit_right] - energy[FIT][fit_left])
                )
                fit_volume = 0.5 * np.log(fit_var_matrix[0])
                v_scale = positive_median(
                    np.abs(fit_volume[fit_right] - fit_volume[fit_left])
                )
                for replica in (FIT, TUNE, TEST):
                    matrix = matrices[topology][replica]
                    neighbours = nearest(matrix, k)
                    _, variance_matrix, _ = local_statistics(
                        energy[replica],
                        neighbours,
                        reference if replica != FIT else None,
                        shrinkage,
                    )
                    variance = variance_matrix[0]
                    try:
                        magnitude, signed, info = graph_feature(
                            matrix,
                            energy[replica],
                            variance,
                            *local_endpoints[replica],
                            k,
                            family,
                            coupling,
                            e_scale,
                            v_scale,
                        )
                    except ValueError:
                        valid = False
                        break
                    feature[replica] = magnitude
                    signed_feature[replica] = signed
                    audit.append(
                        {
                            "system_id": system,
                            "replica": replica,
                            "topology": topology,
                            "family": family,
                            "k": k,
                            "shrinkage": shrinkage,
                            "coupling": coupling,
                            "connected": True,
                            **info,
                        }
                    )
                if not valid:
                    continue
                scored.append((coupling, feature, signed_feature))
            if not scored:
                continue
            target_kinds = (
                ("magnitude", "signed")
                if family == "directed_basin"
                else ("magnitude",)
            )
            for target_kind in target_kinds:
                selected = []
                for coupling, magnitude_values, signed_values in scored:
                    values = (
                        signed_values if target_kind == "signed" else magnitude_values
                    )
                    scale_alpha = (
                        fitted_signed_scale(values[FIT], targets[target_kind][FIT])
                        if target_kind == "signed"
                        else fitted_scale(values[FIT], targets[target_kind][FIT])
                    )
                    loss = float(
                        np.mean(
                            np.abs(
                                targets[target_kind][TUNE] - scale_alpha * values[TUNE]
                            )
                        )
                    )
                    selected.append((loss, coupling, scale_alpha, values))
                loss, coupling, scale_alpha, values = min(selected)
                suffix = "_signed" if target_kind == "signed" else ""
                model = f"{topology}_{family}{suffix}"
                predictions.append((model, target_kind, scale_alpha * values[TEST]))
                fits.append(
                    {
                        "system_id": system,
                        "model": model,
                        "target_kind": target_kind,
                        "alpha": scale_alpha,
                        "k": k,
                        "shrinkage": shrinkage,
                        "coupling": coupling,
                        "tune_mae": loss,
                    }
                )
    test_w1 = data["pairs"][TEST].w1.to_numpy()[take[TEST]]
    settings = config["analysis"]["pairwise_geometry"]["boundary_audit"]
    results = []
    for model, target_kind, prediction in predictions:
        target = targets[target_kind][TEST]
        for band in range(6):
            mask = (test_w1 >= edges[band]) & (
                (test_w1 < edges[band + 1]) if band < 5 else True
            )
            if mask.sum() < 30:
                continue
            results.append(
                {
                    "system_id": system,
                    "model": model,
                    "target_kind": target_kind,
                    "band": f"q{band}",
                    "pairs": int(mask.sum()),
                    "mae": float(np.mean(np.abs(target[mask] - prediction[mask]))),
                    "spearman": finite_spearman(target[mask], prediction[mask]),
                    "sign_accuracy": float(
                        np.mean(np.sign(target[mask]) == np.sign(prediction[mask]))
                    )
                    if target_kind == "signed"
                    else np.nan,
                    **mass_metrics(
                        target[mask],
                        prediction[mask],
                        targets[target_kind][FIT],
                        settings["distribution_bins"],
                        settings["distribution_smoothing"],
                    ),
                }
            )
    runtime = {
        "system_id": system,
        "runtime_seconds": time.perf_counter() - started,
        "bandwidth_angstrom": bandwidth,
    }
    return results, fits, audit, runtime


def task(arguments):
    return evaluate_system(*arguments)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["target_kind", "model", "band"], as_index=False).agg(
        recovery_mean=("distribution_recovery", "mean"),
        recovery_median=("distribution_recovery", "median"),
        recovery_sd=("distribution_recovery", "std"),
        sign_accuracy=("sign_accuracy", "mean"),
        spearman=("spearman", "mean"),
        systems=("system_id", "nunique"),
    )


def paired_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    magnitude = results.query("target_kind == 'magnitude'")
    for band in ("q0", "q5"):
        pivot = magnitude[magnitude.band == band].pivot_table(
            index="system_id", columns="model", values="distribution_recovery"
        )
        for model in pivot.columns:
            if model == "direct":
                continue
            if model.startswith("structural_w1_"):
                topology = "structural_w1"
            elif model.startswith("work_scale_"):
                topology = "work_scale"
            else:
                topology = "work3"
            energy_baseline = f"{topology}_energy_only"
            baselines = ["direct"]
            if energy_baseline in pivot and model != energy_baseline:
                baselines.append(energy_baseline)
            for baseline in baselines:
                delta = (pivot[model] - pivot[baseline]).dropna().to_numpy()
                if not len(delta):
                    continue
                boot = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(
                    axis=1
                )
                rows.append(
                    {
                        "band": band,
                        "model": model,
                        "baseline": baseline,
                        "systems": len(delta),
                        "mean_gain": float(delta.mean()),
                        "median_gain": float(np.median(delta)),
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                    }
                )
    return pd.DataFrame(rows)


def pilot_gate(comparisons: pd.DataFrame) -> dict:
    candidates = comparisons[
        comparisons.model.str.contains("energy_variance|directed_basin")
        & ~comparisons.model.str.endswith("_signed")
    ]
    decisions = []
    for model in sorted(candidates.model.unique()):
        block = candidates[candidates.model == model]
        gains = {}
        for band in ("q0", "q5"):
            direct = block[(block.band == band) & (block.baseline == "direct")]
            energy = block[
                (block.band == band) & block.baseline.str.endswith("energy_only")
            ]
            gains[band] = {
                "versus_direct": float(direct.iloc[0].median_gain)
                if len(direct)
                else np.nan,
                "versus_energy_only": float(energy.iloc[0].median_gain)
                if len(energy)
                else np.nan,
                "energy_ci_low": float(energy.iloc[0].ci_low)
                if len(energy)
                else np.nan,
            }
        passed = False
        for improved, other in (("q0", "q5"), ("q5", "q0")):
            passed |= (
                gains[improved]["versus_direct"] >= 0.02
                and gains[improved]["versus_energy_only"] >= 0.02
                and gains[improved]["energy_ci_low"] > 0
                and gains[other]["versus_direct"] >= -0.02
                and gains[other]["versus_energy_only"] >= -0.02
            )
        decisions.append({"model": model, "gains": gains, "passed": bool(passed)})
    passed = any(item["passed"] for item in decisions)
    return {
        "passed": passed,
        "decision": "continue_full" if passed else "stop_after_pilot",
        "thresholds": {
            "extreme_gain_vs_direct_and_energy_only": 0.02,
            "maximum_other_extreme_loss": 0.02,
            "energy_only_comparison_ci_must_be_positive": True,
        },
        "models": decisions,
        "null_controls": (
            "required before full run" if passed else "skipped after futility gate"
        ),
    }


def plot_recovery(summary: pd.DataFrame, edges: np.ndarray, destination) -> None:
    labels = [f"q{i}\n{edges[i]:.3f}–{edges[i + 1]:.3f} Å" for i in range(6)]
    for target_kind in ("magnitude", "signed"):
        block = summary[summary.target_kind == target_kind]
        if block.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 7))
        for model in block.model.unique():
            selected = block[block.model == model].set_index("band")
            y = np.array(
                [selected.recovery_mean.get(f"q{i}", np.nan) for i in range(6)]
            )
            sd = np.array([selected.recovery_sd.get(f"q{i}", np.nan) for i in range(6)])
            ax.plot(range(6), 100 * y, marker="o", label=model)
            ax.fill_between(range(6), 100 * (y - sd), 100 * (y + sd), alpha=0.06)
        ax.set_xticks(range(6), labels)
        ax.set_ylabel("Distribution recovery ± system SD (%)")
        ax.set_title(f"Variance-aware Work graph: {target_kind}")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(destination / f"variance_graph_{target_kind}_recovery.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config()
    rows = load_systems()
    pilot_path = (
        HERE
        / "outputs/analysis/pairwise_geometry/checkpoint26_pyrosetta_graph/pilot_systems.parquet"
    )
    if not args.full:
        pilot = set(pd.read_parquet(pilot_path).query("pilot").system_id)
        rows = [row for row in rows if row["system_id"] in pilot]
    destination = OUTPUT / ("full" if args.full else "pilot")
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    with (
        HERE
        / "outputs/analysis/pairwise_geometry/checkpoint15_global_w1/global_w1_edges.yaml"
    ).open() as handle:
        edges = np.asarray(yaml.safe_load(handle)["edges_angstrom"])
    pending = [
        row
        for row in rows
        if args.force
        or not all(
            (parts / f"{row['system_id']}.{name}.parquet").exists()
            for name in ("results", "fits", "audit", "runtime")
        )
    ]
    executor = (
        ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    )
    evaluated = (
        executor.map(task, [(row, config, edges) for row in pending])
        if executor
        else map(task, [(row, config, edges) for row in pending])
    )
    for index, (row, values) in enumerate(zip(pending, evaluated), 1):
        for name, records in zip(("results", "fits", "audit", "runtime"), values):
            atomic_parquet(
                pd.DataFrame(records if isinstance(records, list) else [records]),
                parts / f"{row['system_id']}.{name}.parquet",
            )
        print(f"[{index}/{len(pending)}] {row['system_id']}", flush=True)
    if executor:
        executor.shutdown()
    tables = {}
    for name in ("results", "fits", "audit", "runtime"):
        tables[name] = pd.concat(
            [
                pd.read_parquet(parts / f"{row['system_id']}.{name}.parquet")
                for row in rows
            ],
            ignore_index=True,
        )
        atomic_parquet(tables[name], destination / f"variance_graph_{name}.parquet")
    summary = summarize(tables["results"])
    comparisons = paired_comparisons(tables["results"])
    gate = pilot_gate(comparisons)
    atomic_parquet(summary, destination / "variance_graph_summary.parquet")
    atomic_parquet(comparisons, destination / "variance_graph_paired_extremes.parquet")
    atomic_yaml(destination / "variance_graph_pilot_gate.yaml", gate)
    plot_recovery(summary, edges, destination)
    atomic_yaml(
        destination / "checkpoint29_report.yaml",
        {
            "systems": len(rows),
            "assignment": "replica-1 fit, replica-2 select, replica-3 test",
            "topologies": list(TOPOLOGIES),
            "families": list(FAMILIES),
            "k_values": list(K_VALUES),
            "shrinkages": list(SHRINKAGES),
            "couplings": list(LAMBDAS),
            "median_runtime_seconds": float(tables["runtime"].runtime_seconds.median()),
            "pilot_gate": gate,
        },
    )


if __name__ == "__main__":
    main()
