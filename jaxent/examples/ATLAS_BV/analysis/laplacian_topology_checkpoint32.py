"""Checkpoint 32: sparse versus all-pairs Laplacian topology selection."""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.metrics import pairwise_distances

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    PRIMARY_RANK,
    log_kernel_density,
    neighbour_bandwidth,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_metric_comparison_checkpoint31 import (
    METRICS,
    basin_mass_prior,
    load_candidate_data,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import (
    DEFAULT_STRENGTHS,
    bootstrap_interval,
    edge_diagnostics,
    effective_sample_size,
    graph_components,
    load_rows,
    optimise_weights,
    prior_for_ess,
    pseudo_uptake,
    relabel_graph,
    stable_seed,
    structural_clusters,
    weighted_histogram_js,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import atomic_parquet
from jaxent.src.opt.loss.graph_laplacian import (
    FrameGraph,
    build_all_pairs_graph_from_distances,
    build_frame_graph_from_distances,
)


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint32_laplacian_topology"
K_VALUES = (3, 5, 10, 20, 40)
BANDWIDTH_QUANTILES = (0.02, 0.04, 0.08, 0.16)
BASIN_TRANSFERS = (0.10, 0.25, 0.40)
LOCKED_CANDIDATE = "work_scale__all_pairs_rbf__0p16"


@dataclass(frozen=True, slots=True)
class GraphSpec:
    metric: str
    topology: str
    parameter: float

    @property
    def key(self) -> str:
        value = f"{self.parameter:g}".replace(".", "p")
        return f"{self.metric}__{self.topology}__{value}"


def candidate_specs() -> tuple[GraphSpec, ...]:
    return tuple(
        GraphSpec(metric, topology, float(parameter))
        for metric in METRICS
        for topology, parameters in (
            ("self_tuned_knn", K_VALUES),
            ("uniform_knn", K_VALUES),
            ("all_pairs_rbf", BANDWIDTH_QUANTILES),
        )
        for parameter in parameters
    )


def metric_distances(values: np.ndarray, indices: np.ndarray, metric: str) -> np.ndarray:
    selected = np.asarray(values)[np.asarray(indices, dtype=int)]
    distance_metric = "manhattan" if metric == "work_density_legacy_zq" else "euclidean"
    distances = pairwise_distances(selected, metric=distance_metric)
    if distance_metric == "manhattan":
        distances /= selected.shape[1]
    return distances


def bandwidth_from_quantile(distances: np.ndarray, quantile: float) -> float:
    values = distances[np.triu_indices_from(distances, k=1)]
    positive = values[values > 0]
    if not positive.size:
        raise ValueError("cannot choose a bandwidth when every pairwise distance is zero")
    return float(max(np.quantile(positive, quantile), np.finfo(float).eps))


def build_candidate_graph(distances: np.ndarray, spec: GraphSpec) -> FrameGraph:
    effective_k = min(int(spec.parameter), len(distances) - 1)
    if spec.topology == "self_tuned_knn":
        return build_frame_graph_from_distances(
            distances, k=effective_k, metric=spec.key
        )
    if spec.topology == "uniform_knn":
        return build_frame_graph_from_distances(
            distances, k=effective_k, metric=spec.key, weighted=False
        )
    if spec.topology == "all_pairs_rbf":
        bandwidth = bandwidth_from_quantile(distances, spec.parameter)
        return build_all_pairs_graph_from_distances(
            distances, metric=spec.key, bandwidth=bandwidth
        )
    raise ValueError(f"unknown topology: {spec.topology}")


def uniform_complete_graph(n_frames: int) -> FrameGraph:
    return build_all_pairs_graph_from_distances(
        np.zeros((n_frames, n_frames)),
        metric="uniform_all_pairs",
        bandwidth=None,
    )


def graph_diagnostics(
    graph: FrameGraph, structural: np.ndarray, density: np.ndarray, labels: np.ndarray
) -> dict:
    observed_distance, observed_purity = edge_diagnostics(graph, structural, labels)
    source = np.asarray(graph.edge_sources)
    target = np.asarray(graph.edge_targets)
    weights = np.asarray(graph.edge_weights)
    degree = np.zeros(graph.n_nodes, dtype=float)
    np.add.at(degree, source, weights)
    np.add.at(degree, target, weights)
    return {
        "components": graph_components(graph),
        "n_edges": len(weights),
        "normalized_density_energy": normalized_energy_numpy(density, graph),
        "structural_edge_distance": observed_distance,
        "cluster_edge_purity": observed_purity,
        "minimum_weighted_degree": float(degree.min()),
        "median_weighted_degree": float(np.median(degree)),
    }


def normalized_energy_numpy(values: np.ndarray, graph: FrameGraph) -> float:
    """NumPy audit equivalent of the JAX loss, avoiding scalar dispatch overhead."""
    values = np.asarray(values, dtype=float)
    variance = float(np.var(values))
    if variance <= np.finfo(float).eps:
        return 0.0
    source = np.asarray(graph.edge_sources, dtype=int)
    target = np.asarray(graph.edge_targets, dtype=int)
    weights = np.asarray(graph.edge_weights, dtype=float)
    return float(np.average(np.square(values[source] - values[target]), weights=weights)) / variance


def audit_system(row: dict, config: dict, permutations: int) -> list[dict]:
    del permutations  # The node-relabel null expectation is available exactly.
    data, values = load_candidate_data(row, config)
    bandwidth = neighbour_bandwidth(data["matrices"][1][1], PRIMARY_RANK)
    output = []
    for replica in (1, 2, 3):
        indices, structural = data["matrices"][replica]
        structural = np.asarray(structural, dtype=float)
        density = log_kernel_density(structural, bandwidth)
        labels = structural_clusters(structural)
        null_source, null_target = np.triu_indices(len(indices), k=1)
        density_variance = max(float(np.var(density)), np.finfo(float).eps)
        null_energy_reference = float(
            np.mean(np.square(density[null_source] - density[null_target]))
            / density_variance
        )
        null_distance_reference = float(
            np.mean(structural[null_source, null_target])
        )
        null_purity_reference = float(
            np.mean(labels[null_source] == labels[null_target])
        )
        distances = {
            metric: metric_distances(values[metric], indices, metric) for metric in METRICS
        }
        specs: tuple[GraphSpec | None, ...] = (*candidate_specs(), None)
        for spec in specs:
            graph = (
                uniform_complete_graph(len(indices))
                if spec is None
                else build_candidate_graph(distances[spec.metric], spec)
            )
            observed = graph_diagnostics(graph, structural, density, labels)
            key = "uniform_all_pairs" if spec is None else spec.key
            output.append(
                {
                    "system_id": data["system"],
                    "replica": replica,
                    "candidate": key,
                    "metric": "geometry_free" if spec is None else spec.metric,
                    "topology": "uniform_all_pairs" if spec is None else spec.topology,
                    "parameter": np.nan if spec is None else spec.parameter,
                    **observed,
                    "density_energy_gain": (
                        (null_energy_reference - observed["normalized_density_energy"])
                        / null_energy_reference
                        if null_energy_reference > 0
                        else 0.0
                    ),
                    "density_energy_p": np.nan,
                    "structural_edge_gain": (
                        (null_distance_reference - observed["structural_edge_distance"])
                        / null_distance_reference
                        if null_distance_reference > 0
                        else 0.0
                    ),
                    "null_cluster_edge_purity": null_purity_reference,
                }
            )
    return output


def challenge_cases(labels: np.ndarray, smooth: np.ndarray, rng) -> list[dict]:
    cases = []
    for fraction in (0.8, 0.4, 0.2):
        for kind, score in (
            ("smooth", smooth),
            ("shuffled", smooth[rng.permutation(len(smooth))]),
        ):
            cases.append(
                {
                    "bias_kind": kind,
                    "bias_level": fraction,
                    "prior": prior_for_ess(score, fraction),
                }
            )
    for transfer in BASIN_TRANSFERS:
        prior, target_mass, _ = basin_mass_prior(labels, transfer)
        cases.append(
            {
                "bias_kind": "basin",
                "bias_level": transfer,
                "prior": prior,
                "target_basin_mass": target_mass,
            }
        )
    return cases


def result_row(
    *, data: dict, replica: int, candidate: str, case: dict, arm: str,
    result: tuple, truth: np.ndarray, labels: np.ndarray, flat: np.ndarray,
    test: np.ndarray, target: np.ndarray, smooth: np.ndarray,
) -> dict:
    weights, strength, validation_mse = result
    cluster_errors = [
        abs(weights[labels == label].sum() - truth[labels == label].sum())
        for label in np.unique(labels)
    ]
    return {
        "system_id": data["system"],
        "replica": replica,
        "candidate": candidate,
        "bias_kind": case["bias_kind"],
        "bias_level": case["bias_level"],
        "achieved_prior_ess_fraction": effective_sample_size(case["prior"])
        / len(case["prior"]),
        "target_basin_mass": case.get("target_basin_mass", np.nan),
        "arm": arm,
        "selected_strength": strength,
        "validation_mse": validation_mse,
        "test_mse": float(np.mean(np.square(flat[test] @ weights - target[test]))),
        "weight_tv": float(0.5 * np.abs(weights - truth).sum()),
        "cluster_population_mae": float(np.mean(cluster_errors)),
        "structural_js": weighted_histogram_js(smooth, truth, weights),
        "ess_fraction": effective_sample_size(weights) / len(weights),
    }


def optimise_graph_batch(
    observables: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    graphs: dict[str, FrameGraph],
    strengths: tuple[float, ...],
    steps: int,
) -> dict[str, tuple[np.ndarray, float, float]]:
    """Fit all graph/strength combinations in one compiled optimization."""
    names = tuple(graphs)
    maximum_edges = max(int(graph.edge_weights.size) for graph in graphs.values())
    sources = np.zeros((len(names), maximum_edges), dtype=np.int32)
    targets = np.zeros_like(sources)
    edge_weights = np.zeros((len(names), maximum_edges), dtype=float)
    for index, name in enumerate(names):
        graph = graphs[name]
        count = int(graph.edge_weights.size)
        sources[index, :count] = np.asarray(graph.edge_sources)
        targets[index, :count] = np.asarray(graph.edge_targets)
        edge_weights[index, :count] = np.asarray(graph.edge_weights)

    values = jnp.asarray(observables.reshape(-1, observables.shape[-1]))
    target_values = values @ jnp.asarray(truth)
    train_index = jnp.asarray(train)
    validation_index = jnp.asarray(validation)
    prior_logits = jnp.log(jnp.asarray(prior))
    strength_values = jnp.asarray(strengths)
    scale = jnp.var(target_values[train_index]) + 1e-8
    source_values = jnp.asarray(sources)
    target_indices = jnp.asarray(targets)
    graph_weights = jnp.asarray(edge_weights)

    def one_loss(logits, strength, source, target_index, weights):
        frame_weights = jax.nn.softmax(logits)
        prediction = values @ frame_weights
        data_loss = (
            jnp.mean(jnp.square(prediction[train_index] - target_values[train_index]))
            / scale
        )
        residual = logits - prior_logits
        delta = residual[source] - residual[target_index]
        regularizer = jnp.sum(weights * jnp.square(delta)) / jnp.sum(weights)
        validation_loss = jnp.mean(
            jnp.square(prediction[validation_index] - target_values[validation_index])
        )
        return data_loss + strength * regularizer, validation_loss

    def graph_losses(logits, source, target_index, weights):
        return jax.vmap(one_loss, in_axes=(0, 0, None, None, None))(
            logits, strength_values, source, target_index, weights
        )

    def batch_objective(logits):
        objectives, validation_losses = jax.vmap(graph_losses)(
            logits, source_values, target_indices, graph_weights
        )
        return jnp.sum(objectives), validation_losses

    logits = jnp.broadcast_to(
        prior_logits,
        (len(names), len(strengths), prior_logits.size),
    )
    optimizer = optax.adam(0.05)
    state = optimizer.init(logits)

    @jax.jit
    def step(current, opt_state):
        (_, _), gradient = jax.value_and_grad(batch_objective, has_aux=True)(current)
        updates, opt_state = optimizer.update(gradient, opt_state, current)
        return optax.apply_updates(current, updates), opt_state

    for _ in range(steps):
        logits, state = step(logits, state)
    _, validation_losses = batch_objective(logits)
    validation_values = np.asarray(validation_losses)
    strength_array = np.asarray(strengths)
    output = {}
    for graph_index, name in enumerate(names):
        best = int(
            np.lexsort((-strength_array, validation_values[graph_index]))[0]
        )
        output[name] = (
            np.asarray(jax.nn.softmax(logits[graph_index, best])),
            float(strengths[best]),
            float(validation_values[graph_index, best]),
        )
    return output


def reweight_system(
    row: dict, config: dict, strengths: tuple[float, ...], steps: int,
    frame_cap: int, replicas: tuple[int, ...], selected_candidates: tuple[str, ...] | None,
) -> list[dict]:
    data, values = load_candidate_data(row, config)
    specs_by_key = {spec.key: spec for spec in candidate_specs()}
    specs = tuple(specs_by_key.values()) if selected_candidates is None else tuple(
        specs_by_key[candidate] for candidate in selected_candidates
    )
    output = []
    for replica in replicas:
        global_indices, structural_full = data["matrices"][replica]
        take = np.linspace(
            0, len(global_indices) - 1, min(frame_cap, len(global_indices)), dtype=int
        )
        indices = np.asarray(global_indices)[take]
        structural = np.asarray(structural_full)[np.ix_(take, take)]
        distances = {
            metric: metric_distances(values[metric], indices, metric) for metric in METRICS
        }
        graphs = {spec.key: build_candidate_graph(distances[spec.metric], spec) for spec in specs}
        rng = np.random.default_rng(stable_seed(data["system"], replica, "checkpoint32"))
        rewired = {
            key: relabel_graph(graph, rng.permutation(len(indices)), f"{key}_rewired")
            for key, graph in graphs.items()
        }
        labels = structural_clusters(structural)
        smooth = structural.mean(axis=1)
        observables = pseudo_uptake(data["z"][:, indices])
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
        for case in challenge_cases(labels, smooth, rng):
            prior = case["prior"]
            maxent = optimise_weights(
                observables, truth, prior, train, validation, None, "maxent", strengths, steps
            )
            uniform = optimise_weights(
                observables, truth, prior, train, validation, None,
                "uniform_complete", strengths, steps,
            )
            graph_results = {}
            for topology in ("self_tuned_knn", "uniform_knn", "all_pairs_rbf"):
                keys = [key for key in graphs if f"__{topology}__" in key]
                if not keys:
                    continue
                graph_batch = {key: graphs[key] for key in keys}
                graph_batch.update(
                    {f"{key}__rewired": rewired[key] for key in keys}
                )
                graph_results.update(
                    optimise_graph_batch(
                        observables,
                        truth,
                        prior,
                        train,
                        validation,
                        graph_batch,
                        strengths,
                        steps,
                    )
                )
            for key in graphs:
                arms = {
                    "maxent": maxent,
                    "uniform_all_pairs": uniform,
                    "laplacian": graph_results[key],
                    "rewired_laplacian": graph_results[f"{key}__rewired"],
                }
                for arm, result in arms.items():
                    output.append(
                        result_row(
                            data=data, replica=replica, candidate=key, case=case,
                            arm=arm, result=result, truth=truth, labels=labels,
                            flat=flat, test=test, target=target, smooth=smooth,
                        )
                    )
    return output


def candidate_ranking(audit: pd.DataFrame, development: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates using A eligibility and replica-B conservative recovery."""
    audit_a = audit[(audit.replica == 1) & (audit.topology != "uniform_all_pairs")]
    eligible = (
        audit_a.groupby("candidate")
        .agg(
            components=("components", "max"),
            edge_gain=("structural_edge_gain", "median"),
            purity=("cluster_edge_purity", "median"),
            null_purity=("null_cluster_edge_purity", "median"),
            edges=("n_edges", "median"),
            degree=("median_weighted_degree", "median"),
        )
        .query("components == 1 and edge_gain > 0 and purity > null_purity")
    )
    test = development[(development.replica == 2) & development.candidate.isin(eligible.index)]
    rows = []
    for candidate, block in test.groupby("candidate"):
        bias_scores = {}
        system_scores = []
        valid_mse = True
        for bias in ("smooth", "basin"):
            pivot = block[block.bias_kind == bias].pivot_table(
                index=["system_id", "bias_level"], columns="arm", values="weight_tv"
            )
            relative = (pivot.maxent - pivot.laplacian) / np.maximum(
                pivot.maxent, np.finfo(float).eps
            )
            bias_scores[bias] = float(np.median(relative))
            by_system = relative.groupby("system_id").median()
            system_scores.append(by_system.rename(bias))
            mse = block[block.bias_kind == bias].pivot_table(
                index=["system_id", "bias_level"], columns="arm", values="test_mse"
            )
            relative_mse = (mse.laplacian - mse.maxent) / np.maximum(
                mse.maxent, np.finfo(float).eps
            )
            valid_mse &= float(np.median(relative_mse)) <= 0.01
        balanced = pd.concat(system_scores, axis=1).min(axis=1)
        score_se = (
            float(balanced.std(ddof=1) / np.sqrt(len(balanced)))
            if len(balanced) > 1
            else 0.0
        )
        physical = block[block.bias_kind.isin(("smooth", "basin"))].pivot_table(
            index=["system_id", "bias_kind", "bias_level"],
            columns="arm", values="weight_tv",
        )
        rows.append(
            {
                "candidate": candidate,
                "score": min(bias_scores.values()),
                "score_se": score_se,
                "smooth_score": bias_scores["smooth"],
                "basin_score": bias_scores["basin"],
                "rewired_advantage": float(
                    np.median(physical.rewired_laplacian - physical.laplacian)
                ),
                "mse_eligible": bool(valid_mse),
                **eligible.loc[candidate].to_dict(),
            }
        )
    return pd.DataFrame(rows).query("mse_eligible")


def select_finalists(audit: pd.DataFrame, screening: pd.DataFrame) -> tuple[str, ...]:
    """Keep the best sparse and dense candidate for each physical metric."""
    ranking = candidate_ranking(audit, screening)
    if ranking.empty:
        raise RuntimeError("no topology candidate passed screening eligibility")
    parsed = ranking.candidate.str.split("__", expand=True)
    ranking = ranking.assign(metric=parsed[0], topology=parsed[1])
    ranking["family"] = np.where(
        ranking.topology == "all_pairs_rbf", "dense", "sparse"
    )
    finalists = []
    for metric in METRICS:
        for family in ("sparse", "dense"):
            block = ranking[(ranking.metric == metric) & (ranking.family == family)]
            if block.empty:
                raise RuntimeError(f"no eligible {family} candidate for {metric}")
            finalists.append(str(block.sort_values("score", ascending=False).iloc[0].candidate))
    return tuple(finalists)


def select_candidate(audit: pd.DataFrame, development: pd.DataFrame) -> dict:
    """Select one finalist with the preregistered one-standard-error rule."""
    ranking = candidate_ranking(audit, development)
    if ranking.empty:
        raise RuntimeError("no topology candidate passed development eligibility")
    best = ranking.sort_values("score", ascending=False).iloc[0]
    near = ranking[ranking.score >= best.score - best.score_se]
    choice = near.sort_values(
        ["rewired_advantage", "purity", "edges"], ascending=[False, False, True]
    ).iloc[0]
    return {"selected": str(choice.candidate), "ranking": ranking.to_dict("records")}


def heldout_gate(results: pd.DataFrame) -> dict:
    decisions = {}
    passed = True
    for bias in ("smooth", "basin"):
        block = results[results.bias_kind == bias].pivot_table(
            index=["system_id", "bias_level"], columns="arm", values="weight_tv"
        )
        gain = (block.maxent - block.laplacian).dropna().to_numpy()
        low, high = bootstrap_interval(gain, stable_seed("checkpoint32", bias))
        mse = results[results.bias_kind == bias].pivot_table(
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
            "mean_weight_tv_gain_vs_maxent": float(np.mean(gain)),
            "gain_ci": [low, high],
            "median_relative_test_mse_change": relative_mse,
        }
        passed &= bias_passed
    physical = results[results.bias_kind.isin(("smooth", "basin"))].pivot_table(
        index=["system_id", "bias_kind", "bias_level"],
        columns="arm",
        values="weight_tv",
    )
    physical_gain = float(np.median(physical.maxent - physical.laplacian))
    rewired_gain = float(np.median(physical.maxent - physical.rewired_laplacian))
    uniform_gain = float(np.median(physical.maxent - physical.uniform_all_pairs))
    controls_passed = physical_gain > max(rewired_gain, uniform_gain)
    return {
        "passed": bool(passed and controls_passed),
        "biases": decisions,
        "controls_passed": bool(controls_passed),
        "median_selected_gain": physical_gain,
        "median_rewired_gain": rewired_gain,
        "median_uniform_all_pairs_gain": uniform_gain,
    }


def task(arguments: tuple) -> list[dict]:
    function, inputs = arguments
    return function(*inputs)


def run_tasks(function, inputs, rows, parts, workers, label) -> None:
    executor = (
        ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        )
        if workers > 1
        else None
    )
    arguments = [(function, item) for item in inputs]
    evaluated = executor.map(task, arguments) if executor else map(task, arguments)
    try:
        for index, (row, records) in enumerate(zip(rows, evaluated, strict=True), 1):
            atomic_parquet(pd.DataFrame(records), parts / f"{row['system_id']}.parquet")
            print(f"[{label} {index}/{len(rows)}] {row['system_id']}", flush=True)
    finally:
        if executor:
            executor.shutdown()


def combine(parts, rows) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(parts / f"{row['system_id']}.parquet") for row in rows],
        ignore_index=True,
    )


def confirmation_rows(limit: int | None) -> list[dict]:
    """Return systems not used in the 24-system development cohort."""
    pilot_ids = {row["system_id"] for row in load_rows(False, None)}
    rows = [row for row in load_rows(True, None) if row["system_id"] not in pilot_ids]
    return rows[:limit]


def size_stability_gate(results: pd.DataFrame) -> dict:
    """Require positive system-level gains in every protein-size quartile."""
    pivot = results.pivot_table(
        index=["system_id", "bias_kind", "bias_level"],
        columns="arm",
        values="weight_tv",
    ).reset_index()
    pivot["gain"] = pivot.maxent - pivot.laplacian
    by_system = (
        pivot[pivot.bias_kind.isin(("smooth", "basin"))]
        .groupby(["system_id", "bias_kind"], as_index=False)
        .gain.median()
    )
    metadata = pd.DataFrame(load_systems())[["system_id", "length"]]
    metadata["length"] = metadata.length.astype(int)
    by_system = by_system.merge(metadata, on="system_id", validate="many_to_one")
    by_system["size_quartile"] = pd.qcut(
        by_system.length, 4, labels=("Q1", "Q2", "Q3", "Q4")
    )
    strata = {}
    passed = True
    for (bias, quartile), block in by_system.groupby(
        ["bias_kind", "size_quartile"], observed=True
    ):
        values = block.gain.to_numpy()
        low, high = bootstrap_interval(
            values, stable_seed("checkpoint32", "size", bias, quartile)
        )
        key = f"{bias}_{quartile}"
        stratum_passed = low > 0
        strata[key] = {
            "passed": bool(stratum_passed),
            "systems": len(values),
            "mean_gain": float(np.mean(values)),
            "median_gain": float(np.median(values)),
            "gain_ci": [low, high],
            "positive_system_fraction": float(np.mean(values > 0)),
        }
        passed &= stratum_passed
    return {"passed": bool(passed), "strata": strata}


def run_confirmation(args, config: dict, strengths: tuple[float, ...]) -> None:
    destination = OUTPUT / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    rows = confirmation_rows(args.limit)
    parts = destination / "test_parts"
    parts.mkdir(parents=True, exist_ok=True)
    pending = [
        row for row in rows if not (parts / f"{row['system_id']}.parquet").exists()
    ]
    run_tasks(
        reweight_system,
        [
            (
                row,
                config,
                strengths,
                args.steps,
                args.frame_cap,
                (3,),
                (LOCKED_CANDIDATE,),
            )
            for row in pending
        ],
        pending,
        parts,
        args.workers,
        "confirm32",
    )
    results = combine(parts, rows)
    atomic_parquet(results, destination / "confirmation_reweighting.parquet")
    gate = heldout_gate(results)
    stability = size_stability_gate(results)
    atomic_yaml(destination / "confirmation_gate.yaml", gate)
    atomic_yaml(destination / "size_stability_gate.yaml", stability)
    atomic_yaml(
        destination / "confirmation_report.yaml",
        {
            "candidate": LOCKED_CANDIDATE,
            "systems": len(rows),
            "development_systems_excluded": 24,
            "gate": gate,
            "size_stability_gate": stability,
            "iso_authorized": bool(gate["passed"] and stability["passed"]),
            "ess_policy": "diagnostic only; basin challenges use exact mass transfer",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("audit", "screen", "develop", "test", "all"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--frame-cap", type=int, default=256)
    parser.add_argument("--screen-steps", type=int, default=100)
    parser.add_argument("--screen-frame-cap", type=int, default=64)
    parser.add_argument("--strengths", default=",".join(map(str, DEFAULT_STRENGTHS)))
    args = parser.parse_args()
    destination = OUTPUT / ("smoke" if args.smoke else "development")
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_rows(False, args.limit)
    config = load_config()
    strengths = tuple(float(value) for value in args.strengths.split(","))
    if args.confirm:
        if args.smoke:
            parser.error("--confirm and --smoke are mutually exclusive")
        run_confirmation(args, config, strengths)
        return

    audit_parts = destination / "audit_parts"
    audit_parts.mkdir(parents=True, exist_ok=True)
    if args.phase in ("audit", "all"):
        pending = [r for r in rows if not (audit_parts / f"{r['system_id']}.parquet").exists()]
        run_tasks(
            audit_system,
            [(r, config, args.permutations) for r in pending],
            pending, audit_parts, args.workers, "audit32",
        )
    audit = combine(audit_parts, rows)
    atomic_parquet(audit, destination / "graph_audit.parquet")

    screen_parts = destination / "screen_parts"
    screen_parts.mkdir(parents=True, exist_ok=True)
    if args.phase in ("screen", "all"):
        pending = [r for r in rows if not (screen_parts / f"{r['system_id']}.parquet").exists()]
        run_tasks(
            reweight_system,
            [
                (
                    r, config, strengths, args.screen_steps, args.screen_frame_cap,
                    (1, 2), None,
                )
                for r in pending
            ],
            pending, screen_parts, args.workers, "screen32",
        )
    screening = combine(screen_parts, rows)
    atomic_parquet(screening, destination / "screening_reweighting.parquet")
    finalists = select_finalists(audit, screening)
    atomic_yaml(destination / "finalists.yaml", {"finalists": finalists})

    develop_parts = destination / "develop_parts"
    develop_parts.mkdir(parents=True, exist_ok=True)
    if args.phase in ("develop", "all"):
        pending = [r for r in rows if not (develop_parts / f"{r['system_id']}.parquet").exists()]
        run_tasks(
            reweight_system,
            [
                (r, config, strengths, args.steps, args.frame_cap, (1, 2), finalists)
                for r in pending
            ],
            pending, develop_parts, args.workers, "develop32",
        )
    development = combine(develop_parts, rows)
    atomic_parquet(development, destination / "development_reweighting.parquet")
    selection = select_candidate(audit, development)
    atomic_yaml(destination / "selection.yaml", selection)

    if args.phase in ("test", "all"):
        test_parts = destination / "test_parts"
        test_parts.mkdir(parents=True, exist_ok=True)
        pending = [r for r in rows if not (test_parts / f"{r['system_id']}.parquet").exists()]
        run_tasks(
            reweight_system,
            [
                (
                    r, config, strengths, args.steps, args.frame_cap, (3,),
                    (selection["selected"],),
                )
                for r in pending
            ],
            pending, test_parts, args.workers, "test32",
        )
        heldout = combine(test_parts, rows)
        atomic_parquet(heldout, destination / "heldout_reweighting.parquet")
        atomic_yaml(destination / "heldout_gate.yaml", heldout_gate(heldout))


if __name__ == "__main__":
    main()
