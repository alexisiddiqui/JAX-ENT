"""Checkpoint 31: standalone legacy-Zq and PyRosetta Laplacian pilot.

Each candidate graph receives an independent A-fit/B-select/C-test evaluation.
ESS calibrates continuous synthetic biases and is otherwise report-only; basin
biases are specified by an exact transfer of total probability mass.
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    log_kernel_density,
    neighbour_bandwidth,
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import (
    DEFAULT_STRENGTHS,
    K_VALUES,
    REPLICAS,
    SEED,
    bootstrap_interval,
    edge_diagnostics,
    effective_sample_size,
    graph_components,
    load_rows,
    normalized_energy,
    optimise_weights,
    prior_for_ess,
    pseudo_uptake,
    relabel_graph,
    stable_seed,
    structural_clusters,
    weighted_histogram_js,
)
from jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_population_checkpoint24 import (
    load_score_frames,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    thermodynamic_frame_features,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)
from jaxent.src.opt.loss.graph_laplacian import (
    FrameGraph,
    build_frame_graph,
    build_frame_graph_from_distances,
)


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint31_laplacian_metrics"
METRICS = ("pf_l2", "work_scale", "work_density_legacy_zq", "pyro_ref2015")
NEW_METRICS = ("work_density_legacy_zq", "pyro_ref2015")
BASIN_TRANSFERS = (0.10, 0.25, 0.40)


def metric_frame_values(data: dict, metric: str) -> np.ndarray:
    """Return global frame-major values for one graph candidate."""
    z = np.asarray(data["z"], dtype=float)
    if metric == "pf_l2":
        return z.T / np.sqrt(z.shape[0])
    if metric == "work_scale":
        return np.mean(z, axis=0)[:, None]
    if metric == "work_density_legacy_zq":
        return np.asarray(
            thermodynamic_frame_features(z)["work_density_legacy_zq"], dtype=float
        ).T
    if metric == "pyro_ref2015":
        score = np.asarray(
            load_score_frames(data["system"], data["config"])["ref2015__total"],
            dtype=float,
        )
        if score.shape != (z.shape[1],):
            raise ValueError(
                f"PyRosetta/BV frame mismatch for {data['system']}: "
                f"{score.shape} versus {(z.shape[1],)}"
            )
        return score[:, None]
    raise ValueError(f"unknown checkpoint-31 graph metric: {metric}")


def metric_graph(values: np.ndarray, indices: np.ndarray, metric: str, k: int) -> FrameGraph:
    """Build a candidate graph using its preregistered distance definition."""
    selected = np.asarray(values)[np.asarray(indices, dtype=int)]
    if metric == "work_density_legacy_zq":
        distance = pairwise_distances(selected, metric="manhattan") / selected.shape[1]
        return build_frame_graph_from_distances(distance, k=k, metric=metric)
    return build_frame_graph(selected, k=k, metric=metric)


def load_candidate_data(row: dict, config: dict) -> tuple[dict, dict[str, np.ndarray]]:
    data = system_data(row, config)
    data["config"] = config
    values = {metric: metric_frame_values(data, metric) for metric in METRICS}
    del data["config"]
    return data, values


def audit_system(row: dict, config: dict, permutations: int) -> list[dict]:
    data, values = load_candidate_data(row, config)
    bandwidth = neighbour_bandwidth(data["matrices"][1][1], PRIMARY_RANK)
    output = []
    for replica in REPLICAS:
        indices, structural = data["matrices"][replica]
        structural = np.asarray(structural, dtype=float)
        density = log_kernel_density(structural, bandwidth)
        labels = structural_clusters(structural)
        for metric in METRICS:
            for k in K_VALUES:
                if k >= len(indices):
                    continue
                graph = metric_graph(values[metric], indices, metric, k)
                observed_energy = normalized_energy(density, graph)
                observed_distance, observed_purity = edge_diagnostics(
                    graph, structural, labels
                )
                null_energy = np.empty(permutations)
                null_distance = np.empty(permutations)
                null_purity = np.empty(permutations)
                rng = np.random.default_rng(
                    stable_seed(data["system"], replica, metric, k, "checkpoint31")
                )
                for permutation in range(permutations):
                    control = relabel_graph(
                        graph, rng.permutation(graph.n_nodes), f"{metric}_rewired"
                    )
                    null_energy[permutation] = normalized_energy(density, control)
                    null_distance[permutation], null_purity[permutation] = edge_diagnostics(
                        control, structural, labels
                    )
                energy_reference = float(np.median(null_energy))
                distance_reference = float(np.median(null_distance))
                output.append(
                    {
                        "system_id": data["system"],
                        "replica": replica,
                        "metric": metric,
                        "k": k,
                        "n_nodes": graph.n_nodes,
                        "n_edges": int(graph.edge_weights.size),
                        "components": graph_components(graph),
                        "normalized_density_energy": observed_energy,
                        "null_density_energy": energy_reference,
                        "density_energy_gain": (
                            (energy_reference - observed_energy) / energy_reference
                            if energy_reference > 0
                            else 0.0
                        ),
                        "density_energy_p": float(
                            (1 + np.sum(null_energy <= observed_energy))
                            / (permutations + 1)
                        ),
                        "structural_edge_distance": observed_distance,
                        "null_structural_edge_distance": distance_reference,
                        "structural_edge_gain": (
                            (distance_reference - observed_distance) / distance_reference
                            if distance_reference > 0
                            else 0.0
                        ),
                        "cluster_edge_purity": observed_purity,
                        "null_cluster_edge_purity": float(np.median(null_purity)),
                    }
                )
    return output


def audit_task(arguments: tuple) -> list[dict]:
    return audit_system(*arguments)


def select_candidate_graphs(audit: pd.DataFrame) -> tuple[dict[str, int], dict]:
    """Select k on A/B separately per metric and gate on untouched replica C."""
    selected: dict[str, int] = {}
    gates = {}
    for metric in METRICS:
        block = audit[audit.metric == metric]
        fit = block[block.replica == 1].groupby("k").agg(
            density=("density_energy_gain", "median"),
            edge=("structural_edge_gain", "median"),
        )
        tune = block[block.replica == 2].groupby("k").agg(
            density=("density_energy_gain", "median"),
            edge=("structural_edge_gain", "median"),
        )
        eligible = tune.join(fit, lsuffix="_tune", rsuffix="_fit")
        eligible = eligible[(eligible.density_fit > 0) & (eligible.edge_fit > 0)]
        chosen = 10 if eligible.empty else int(
            eligible.sort_values(["density_tune", "edge_tune"], ascending=False).index[0]
        )
        selected[metric] = chosen
        test = block[(block.replica == 3) & (block.k == chosen)]
        low, high = bootstrap_interval(
            test.density_energy_gain.to_numpy(), stable_seed(SEED, metric, "audit")
        )
        gates[metric] = {
            "passed": bool(
                low > 0
                and test.structural_edge_gain.median() > 0
                and (test.density_energy_p <= 0.05).mean() >= 0.70
            ),
            "selected_k": chosen,
            "test_systems": int(test.system_id.nunique()),
            "mean_density_energy_gain": float(test.density_energy_gain.mean()),
            "density_energy_gain_ci": [low, high],
            "median_structural_edge_gain": float(test.structural_edge_gain.median()),
            "median_cluster_edge_purity": float(test.cluster_edge_purity.median()),
            "fraction_significant_vs_rewired": float(
                (test.density_energy_p <= 0.05).mean()
            ),
        }
    return selected, gates


def basin_mass_prior(labels: np.ndarray, transfer: float) -> tuple[np.ndarray, float, int]:
    """Overpopulate the smallest basin by an exact amount of total mass."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels)
    basin = int(np.flatnonzero(counts == counts.min())[0])
    member = labels == basin
    original_mass = float(member.mean())
    target_mass = original_mass + float(transfer)
    if not 0 < target_mass < 1:
        raise ValueError(
            f"basin transfer {transfer} is invalid for original mass {original_mass}"
        )
    weights = np.empty(len(labels), dtype=float)
    weights[member] = target_mass / member.sum()
    weights[~member] = (1.0 - target_mass) / (~member).sum()
    return weights, target_mass, basin


def challenge_priors(
    labels: np.ndarray, smooth_score: np.ndarray, rng: np.random.Generator
) -> list[dict]:
    cases = []
    for fraction in (0.8, 0.4, 0.2):
        for kind, score in (
            ("smooth", smooth_score),
            ("shuffled", smooth_score[rng.permutation(len(smooth_score))]),
        ):
            prior = prior_for_ess(score, fraction)
            cases.append(
                {
                    "bias_kind": kind,
                    "bias_level": fraction,
                    "bias_parameter": "ess_fraction",
                    "prior": prior,
                    "target_basin_mass": np.nan,
                }
            )
    for transfer in BASIN_TRANSFERS:
        prior, target_mass, _ = basin_mass_prior(labels, transfer)
        cases.append(
            {
                "bias_kind": "basin",
                "bias_level": transfer,
                "bias_parameter": "mass_transfer",
                "prior": prior,
                "target_basin_mass": target_mass,
            }
        )
    return cases


def reweight_system(
    row: dict,
    config: dict,
    selected: dict[str, int],
    strengths: tuple[float, ...],
    steps: int,
    frame_cap: int,
) -> list[dict]:
    data, values = load_candidate_data(row, config)
    output = []
    for replica in REPLICAS:
        global_indices, structural_full = data["matrices"][replica]
        take = np.linspace(
            0, len(global_indices) - 1, min(frame_cap, len(global_indices)), dtype=int
        )
        indices = np.asarray(global_indices)[take]
        structural = np.asarray(structural_full)[np.ix_(take, take)]
        graphs = {
            metric: metric_graph(
                values[metric], indices, metric, min(selected[metric], len(indices) - 1)
            )
            for metric in METRICS
        }
        oracle = build_frame_graph_from_distances(
            structural,
            k=min(selected["pf_l2"], len(indices) - 1),
            metric="structural_w1_oracle",
        )
        rng = np.random.default_rng(stable_seed(data["system"], replica, "checkpoint31"))
        rewired = {
            metric: relabel_graph(
                graph, rng.permutation(len(indices)), f"{metric}_rewired"
            )
            for metric, graph in graphs.items()
        }
        labels = structural_clusters(structural)
        smooth_score = structural.mean(axis=1)
        log_pf = data["z"][:, indices]
        observables = pseudo_uptake(log_pf)
        n_outputs = observables.shape[0] * observables.shape[1]
        split = rng.permutation(n_outputs)
        n_train = max(1, int(0.6 * n_outputs))
        n_validation = max(1, int(0.2 * n_outputs))
        train = split[:n_train]
        validation = split[n_train : n_train + n_validation]
        test = split[n_train + n_validation :]
        truth = np.full(len(indices), 1.0 / len(indices))
        flat = observables.reshape(n_outputs, len(indices))
        target = flat @ truth
        for case in challenge_priors(labels, smooth_score, rng):
            prior = case.pop("prior")
            common = {}
            for arm, kind, graph, arm_strengths in (
                ("none", "none", None, (0.0,)),
                ("maxent", "maxent", None, strengths),
                ("structural_oracle", "laplacian", oracle, strengths),
            ):
                common[arm] = optimise_weights(
                    observables,
                    truth,
                    prior,
                    train,
                    validation,
                    graph,
                    kind,
                    arm_strengths,
                    steps,
                )
            for metric in METRICS:
                arms = dict(common)
                arms["laplacian"] = optimise_weights(
                    observables,
                    truth,
                    prior,
                    train,
                    validation,
                    graphs[metric],
                    "laplacian",
                    strengths,
                    steps,
                )
                arms["rewired_laplacian"] = optimise_weights(
                    observables,
                    truth,
                    prior,
                    train,
                    validation,
                    rewired[metric],
                    "laplacian",
                    strengths,
                    steps,
                )
                for arm, (weights, strength, validation_mse) in arms.items():
                    cluster_errors = [
                        abs(
                            weights[labels == label].sum()
                            - truth[labels == label].sum()
                        )
                        for label in np.unique(labels)
                    ]
                    output.append(
                        {
                            "system_id": data["system"],
                            "replica": replica,
                            "metric": metric,
                            **case,
                            "achieved_prior_ess_fraction": effective_sample_size(prior)
                            / len(prior),
                            "arm": arm,
                            "selected_strength": strength,
                            "validation_mse": validation_mse,
                            "test_mse": float(
                                np.mean(np.square(flat[test] @ weights - target[test]))
                            ),
                            "weight_tv": float(0.5 * np.abs(weights - truth).sum()),
                            "cluster_population_mae": float(np.mean(cluster_errors)),
                            "structural_js": weighted_histogram_js(
                                smooth_score, truth, weights
                            ),
                            "ess_fraction": effective_sample_size(weights) / len(weights),
                        }
                    )
    return output


def reweight_task(arguments: tuple) -> list[dict]:
    return reweight_system(*arguments)


def candidate_reweighting_gates(results: pd.DataFrame) -> dict:
    test = results[results.replica == 3]
    gates = {}
    for metric in METRICS:
        candidate = test[test.metric == metric]
        decisions = {}
        passed = True
        for bias in ("smooth", "basin"):
            block = candidate[candidate.bias_kind == bias].pivot_table(
                index=["system_id", "bias_level"], columns="arm", values="weight_tv"
            )
            improvement = (block.maxent - block.laplacian).dropna().to_numpy()
            low, high = bootstrap_interval(
                improvement, stable_seed(SEED, metric, bias, "checkpoint31")
            )
            mse = candidate[candidate.bias_kind == bias].pivot_table(
                index=["system_id", "bias_level"], columns="arm", values="test_mse"
            )
            relative_mse = float(
                np.median(
                    (mse.laplacian - mse.maxent)
                    / np.maximum(mse.maxent, np.finfo(float).eps)
                )
            )
            bias_passed = bool(low > 0 and relative_mse <= 0.01)
            decisions[bias] = {
                "passed": bias_passed,
                "mean_weight_tv_gain_vs_maxent": float(np.mean(improvement)),
                "gain_ci": [low, high],
                "median_relative_test_mse_change": relative_mse,
            }
            passed &= bias_passed
        physical = candidate[candidate.bias_kind.isin(("smooth", "basin"))].pivot_table(
            index=["system_id", "bias_kind", "bias_level"],
            columns="arm",
            values="weight_tv",
        )
        physical_gain = float(np.median(physical.maxent - physical.laplacian))
        rewired_gain = float(
            np.median(physical.maxent - physical.rewired_laplacian)
        )
        control_passed = physical_gain > rewired_gain
        gates[metric] = {
            "passed": bool(passed and control_passed),
            "biases": decisions,
            "rewired_control_passed": bool(control_passed),
            "median_physical_gain": physical_gain,
            "median_rewired_gain": rewired_gain,
        }
    return gates


def run_parallel(
    function, tasks: list[tuple], rows: list[dict], parts, workers: int, label: str
) -> None:
    executor = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        if workers > 1
        else None
    )
    evaluated = executor.map(function, tasks) if executor else map(function, tasks)
    try:
        for index, (row, records) in enumerate(zip(rows, evaluated, strict=True), 1):
            atomic_parquet(pd.DataFrame(records), parts / f"{row['system_id']}.parquet")
            print(f"[{label} {index}/{len(rows)}] {row['system_id']}", flush=True)
    finally:
        if executor:
            executor.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "reweight", "all"), default="all")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--frame-cap", type=int, default=256)
    parser.add_argument("--strengths", default=",".join(map(str, DEFAULT_STRENGTHS)))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    destination = OUTPUT / ("full" if args.full else "pilot")
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.full, args.limit)
    config = load_config()

    audit_parts = destination / "audit_parts"
    audit_parts.mkdir(parents=True, exist_ok=True)
    if args.phase in {"audit", "all"}:
        pending = [
            row
            for row in rows
            if not (audit_parts / f"{row['system_id']}.parquet").exists()
        ]
        run_parallel(
            audit_task,
            [(row, config, args.permutations) for row in pending],
            pending,
            audit_parts,
            args.workers,
            "audit31",
        )
    audit = pd.concat(
        [pd.read_parquet(audit_parts / f"{row['system_id']}.parquet") for row in rows],
        ignore_index=True,
    )
    atomic_parquet(audit, destination / "graph_audit.parquet")
    selected, audit_gates = select_candidate_graphs(audit)
    atomic_yaml(destination / "graph_gates.yaml", {"selected": selected, "gates": audit_gates})

    reweight_gates = None
    if args.phase in {"reweight", "all"}:
        reweight_parts = destination / "reweight_parts"
        reweight_parts.mkdir(parents=True, exist_ok=True)
        pending = [
            row
            for row in rows
            if not (reweight_parts / f"{row['system_id']}.parquet").exists()
        ]
        strengths = tuple(float(value) for value in args.strengths.split(","))
        run_parallel(
            reweight_task,
            [
                (row, config, selected, strengths, args.steps, args.frame_cap)
                for row in pending
            ],
            pending,
            reweight_parts,
            args.workers,
            "reweight31",
        )
        results = pd.concat(
            [
                pd.read_parquet(reweight_parts / f"{row['system_id']}.parquet")
                for row in rows
            ],
            ignore_index=True,
        )
        atomic_parquet(results, destination / "reweighting_results.parquet")
        reweight_gates = candidate_reweighting_gates(results)
        atomic_yaml(destination / "reweighting_gates.yaml", reweight_gates)

    new_candidates_passed = bool(
        reweight_gates is not None
        and any(
            audit_gates[metric]["passed"] and reweight_gates[metric]["passed"]
            for metric in NEW_METRICS
        )
    )
    atomic_yaml(
        destination / "checkpoint31_report.yaml",
        {
            "checkpoint": 31,
            "systems": len(rows),
            "scope": "full" if args.full else "pilot",
            "selected_k": selected,
            "audit_gates": audit_gates,
            "reweighting_gates": reweight_gates,
            "full_run_authorized": bool(not args.full and new_candidates_passed),
            "iso_authorized": bool(args.full and new_candidates_passed),
            "ess_policy": "diagnostic only; basin challenges use exact mass transfer",
        },
    )


if __name__ == "__main__":
    main()
