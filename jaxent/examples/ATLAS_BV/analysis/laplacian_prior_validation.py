"""Qualify a prior-relative BV frame-graph Laplacian before ISO use.

The graph audit is observational: it asks whether structural log density is smooth
on a graph built only from fixed-BV features.  The reweighting benchmark then asks
whether that graph helps recover known uniform MD weights from controlled biased
priors.  No ISO artifacts are read or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
    system_data,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.opt.loss.graph_laplacian import (
    FrameGraph,
    build_bv_frame_graph,
    build_frame_graph_from_distances,
    prior_relative_graph_energy,
)


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/laplacian_prior_validation"
METRICS = ("pf_l2", "work_scale")
K_VALUES = (5, 10, 20, 50)
REPLICAS = (1, 2, 3)
SEED = 20260906
DEFAULT_STRENGTHS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
BIAS_KINDS = ("smooth", "basin", "shuffled")


def stable_seed(*tokens: object) -> int:
    digest = hashlib.sha256("|".join(map(str, tokens)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def graph_arrays(graph: FrameGraph) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(graph.edge_sources, dtype=int),
        np.asarray(graph.edge_targets, dtype=int),
        np.asarray(graph.edge_weights, dtype=float),
    )


def relabel_graph(graph: FrameGraph, permutation: np.ndarray, metric: str) -> FrameGraph:
    """Randomly relabel nodes while retaining topology, degree, and edge weights."""
    source, target, weight = graph_arrays(graph)
    source = permutation[source]
    target = permutation[target]
    left = np.minimum(source, target)
    right = np.maximum(source, target)
    order = np.lexsort((right, left))
    return FrameGraph(
        edge_sources=jnp.asarray(left[order], dtype=jnp.int32),
        edge_targets=jnp.asarray(right[order], dtype=jnp.int32),
        edge_weights=jnp.asarray(weight[order]),
        n_nodes=graph.n_nodes,
        metric=metric,
        k=graph.k,
    )


def normalized_energy(values: np.ndarray, graph: FrameGraph) -> float:
    variance = float(np.var(values))
    if variance <= np.finfo(float).eps:
        return 0.0
    value = prior_relative_graph_energy(
        jnp.asarray(values),
        jnp.zeros(len(values)),
        graph.edge_sources,
        graph.edge_targets,
        graph.edge_weights,
    )
    return float(value) / variance


def structural_clusters(distance: np.ndarray) -> np.ndarray:
    """Select a small structural partition without using BV or density values."""
    best: tuple[float, np.ndarray] | None = None
    maximum = min(6, max(2, len(distance) // 30))
    for n_clusters in range(2, maximum + 1):
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=SEED)
        labels = model.fit_predict(distance)
        counts = np.bincount(labels)
        if counts.min() < max(10, int(0.05 * len(labels))):
            continue
        score = float(
            silhouette_score(
                distance,
                labels,
                metric="precomputed",
                sample_size=min(500, len(labels)),
                random_state=SEED,
            )
        )
        if best is None or score > best[0]:
            best = score, labels
    if best is not None:
        return best[1]
    return KMeans(n_clusters=2, n_init=20, random_state=SEED).fit_predict(distance)


def edge_diagnostics(
    graph: FrameGraph, structural_distance: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    source, target, weight = graph_arrays(graph)
    edge_distance = float(np.average(structural_distance[source, target], weights=weight))
    purity = float(np.average(labels[source] == labels[target], weights=weight))
    return edge_distance, purity


def bv_graph(data: dict, replica: int, metric: str, k: int) -> FrameGraph:
    indices = data["matrices"][replica][0]
    features = BV_input_features(
        heavy_contacts=data["heavy"][:, indices],
        acceptor_contacts=data["acceptor"][:, indices],
    )
    return build_bv_frame_graph(features, metric=metric, k=k)


def audit_system(
    row: dict, config: dict, permutations: int
) -> list[dict[str, float | int | str]]:
    data = system_data(row, config)
    bandwidth = neighbour_bandwidth(data["matrices"][1][1], PRIMARY_RANK)
    rows = []
    for replica in REPLICAS:
        structural = np.asarray(data["matrices"][replica][1], dtype=float)
        density = log_kernel_density(structural, bandwidth)
        labels = structural_clusters(structural)
        candidates = [
            bv_graph(data, replica, metric, k)
            for metric in METRICS
            for k in K_VALUES
            if k < len(structural)
        ]
        candidates.extend(
            build_frame_graph_from_distances(
                structural, k=k, metric="structural_w1_oracle"
            )
            for k in K_VALUES
            if k < len(structural)
        )
        for graph in candidates:
            observed_energy = normalized_energy(density, graph)
            observed_distance, observed_purity = edge_diagnostics(
                graph, structural, labels
            )
            null_energy = np.empty(permutations)
            null_distance = np.empty(permutations)
            null_purity = np.empty(permutations)
            rng = np.random.default_rng(
                stable_seed(data["system"], replica, graph.metric, graph.k)
            )
            for index in range(permutations):
                control = relabel_graph(
                    graph, rng.permutation(graph.n_nodes), f"{graph.metric}_rewired"
                )
                null_energy[index] = normalized_energy(density, control)
                null_distance[index], null_purity[index] = edge_diagnostics(
                    control, structural, labels
                )
            energy_reference = float(np.median(null_energy))
            distance_reference = float(np.median(null_distance))
            rows.append(
                {
                    "system_id": data["system"],
                    "replica": replica,
                    "metric": graph.metric,
                    "k": graph.k,
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
    return rows


def graph_components(graph: FrameGraph) -> int:
    source, target, _ = graph_arrays(graph)
    parent = np.arange(graph.n_nodes)

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return int(node)

    for left, right in zip(source, target, strict=True):
        a, b = root(int(left)), root(int(right))
        if a != b:
            parent[b] = a
    return len({root(node) for node in range(graph.n_nodes)})


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def select_graph(audit: pd.DataFrame) -> tuple[dict, dict]:
    physical = audit[audit.metric.isin(METRICS)]
    scores = (
        physical.groupby(["replica", "metric", "k"], as_index=False)
        .agg(
            median_density_gain=("density_energy_gain", "median"),
            median_edge_gain=("structural_edge_gain", "median"),
        )
    )
    fit = scores[scores.replica == 1].set_index(["metric", "k"])
    tune = scores[scores.replica == 2].set_index(["metric", "k"])
    eligible = tune.join(fit, lsuffix="_tune", rsuffix="_fit")
    eligible = eligible[
        (eligible.median_density_gain_fit > 0) & (eligible.median_edge_gain_fit > 0)
    ]
    if eligible.empty:
        selected = {"metric": "pf_l2", "k": 10}
    else:
        choice = eligible.sort_values(
            ["median_density_gain_tune", "median_edge_gain_tune"], ascending=False
        ).iloc[0]
        selected = {"metric": choice.name[0], "k": int(choice.name[1])}

    test = physical[
        (physical.replica == 3)
        & (physical.metric == selected["metric"])
        & (physical.k == selected["k"])
    ]
    low, high = bootstrap_interval(test.density_energy_gain.to_numpy(), SEED)
    gate = {
        "passed": bool(
            low > 0
            and test.structural_edge_gain.median() > 0
            and (test.density_energy_p <= 0.05).mean() >= 0.70
        ),
        "test_systems": int(test.system_id.nunique()),
        "mean_density_energy_gain": float(test.density_energy_gain.mean()),
        "density_energy_gain_ci": [low, high],
        "median_structural_edge_gain": float(test.structural_edge_gain.median()),
        "fraction_systems_below_rewired_p05": float(
            (test.density_energy_p <= 0.05).mean()
        ),
    }
    return selected, gate


def effective_sample_size(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(np.square(weights)))


def prior_for_ess(score: np.ndarray, fraction: float) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    score = (score - score.mean()) / max(score.std(), np.finfo(float).eps)
    target = fraction * len(score)
    low, high = 0.0, 32.0
    for _ in range(60):
        beta = 0.5 * (low + high)
        logits = beta * score
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        if effective_sample_size(weights) > target:
            low = beta
        else:
            high = beta
    logits = high * score
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def pseudo_uptake(log_pf: np.ndarray) -> np.ndarray:
    """Framewise dimensionless uptake spanning each residue's dynamic range."""
    centered = log_pf - np.median(log_pf, axis=1, keepdims=True)
    rates = np.exp(np.clip(-centered, -20.0, 20.0))
    times = np.array([0.1, 1.0, 10.0])[:, None, None]
    return 1.0 - np.exp(-times * rates[None, :, :])


def weighted_histogram_js(
    coordinate: np.ndarray, truth: np.ndarray, prediction: np.ndarray
) -> float:
    edges = np.quantile(coordinate, np.linspace(0, 1, 21))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    expected = np.histogram(coordinate, bins=edges, weights=truth)[0] + 1e-12
    actual = np.histogram(coordinate, bins=edges, weights=prediction)[0] + 1e-12
    return float(jensenshannon(expected, actual, base=2.0) ** 2)


def optimise_weights(
    observables: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    graph: FrameGraph | None,
    regularizer: str,
    strengths: tuple[float, ...],
    steps: int,
) -> tuple[np.ndarray, float, float]:
    values = jnp.asarray(observables.reshape(-1, observables.shape[-1]))
    target = values @ jnp.asarray(truth)
    train_index = jnp.asarray(train)
    validation_index = jnp.asarray(validation)
    prior_logits = jnp.log(jnp.asarray(prior))
    scale = jnp.var(target[train_index]) + 1e-8

    def losses(logits, strength):
        weights = jax.nn.softmax(logits)
        prediction = values @ weights
        data = jnp.mean(jnp.square(prediction[train_index] - target[train_index])) / scale
        if regularizer == "maxent":
            reg = jnp.sum(jnp.asarray(prior) * (prior_logits - jax.nn.log_softmax(logits)))
        elif regularizer == "laplacian" and graph is not None:
            reg = prior_relative_graph_energy(
                logits,
                prior_logits,
                graph.edge_sources,
                graph.edge_targets,
                graph.edge_weights,
            )
        elif regularizer == "uniform_complete":
            residual = logits - prior_logits
            n_frames = residual.shape[0]
            reg = (2.0 * n_frames / (n_frames - 1)) * jnp.var(residual)
        else:
            reg = jnp.asarray(0.0)
        val = jnp.mean(jnp.square(prediction[validation_index] - target[validation_index]))
        return data + strength * reg, val

    # Optimise every strength in one compiled batch. Each row remains an
    # independent Adam trajectory, but this avoids recompiling the same kernel
    # once per candidate throughout the pilot.
    strength_values = jnp.asarray(strengths)
    logits = jnp.broadcast_to(prior_logits, (len(strengths), prior_logits.size))
    optimizer = optax.adam(0.05)
    state = optimizer.init(logits)

    def batch_objective(current):
        objectives, validation_losses = jax.vmap(losses)(current, strength_values)
        return jnp.sum(objectives), validation_losses

    @jax.jit
    def step(current, opt_state):
        (_, _), gradient = jax.value_and_grad(batch_objective, has_aux=True)(current)
        updates, opt_state = optimizer.update(gradient, opt_state, current)
        return optax.apply_updates(current, updates), opt_state

    for _ in range(steps):
        logits, state = step(logits, state)
    _, validation_losses = jax.vmap(losses)(logits, strength_values)
    validation_values = np.asarray(validation_losses)
    # Match the previous deterministic tie break: prefer larger regularisation.
    best_index = int(np.lexsort((-np.asarray(strengths), validation_values))[0])
    weights = np.asarray(jax.nn.softmax(logits[best_index]))
    return weights, float(strengths[best_index]), float(validation_values[best_index])


def reweight_system(
    row: dict,
    config: dict,
    selected: dict,
    ess_fractions: tuple[float, ...],
    strengths: tuple[float, ...],
    steps: int,
    frame_cap: int,
) -> list[dict]:
    data = system_data(row, config)
    output = []
    for replica in REPLICAS:
        global_indices, structural_full = data["matrices"][replica]
        take = np.linspace(
            0, len(global_indices) - 1, min(frame_cap, len(global_indices)), dtype=int
        )
        indices = global_indices[take]
        structural = structural_full[np.ix_(take, take)]
        log_pf = data["z"][:, indices]
        features = BV_input_features(
            heavy_contacts=data["heavy"][:, indices],
            acceptor_contacts=data["acceptor"][:, indices],
        )
        graph = build_bv_frame_graph(
            features, metric=selected["metric"], k=min(selected["k"], len(take) - 1)
        )
        oracle = build_frame_graph_from_distances(
            structural, k=min(selected["k"], len(take) - 1), metric="structural_w1_oracle"
        )
        rng = np.random.default_rng(stable_seed(data["system"], replica, "reweight"))
        rewired = relabel_graph(graph, rng.permutation(len(take)), "rewired")
        labels = structural_clusters(structural)
        smooth_score = structural.mean(axis=1)
        basin_score = (labels == np.argmin(np.bincount(labels))).astype(float)
        scores = {
            "smooth": smooth_score,
            "basin": basin_score,
            "shuffled": smooth_score[rng.permutation(len(take))],
        }
        observables = pseudo_uptake(log_pf)
        n_outputs = observables.shape[0] * observables.shape[1]
        split = rng.permutation(n_outputs)
        n_train = max(1, int(0.6 * n_outputs))
        n_validation = max(1, int(0.2 * n_outputs))
        train = split[:n_train]
        validation = split[n_train : n_train + n_validation]
        test = split[n_train + n_validation :]
        truth = np.full(len(take), 1.0 / len(take))
        flat = observables.reshape(n_outputs, len(take))
        target = flat @ truth
        for fraction in ess_fractions:
            for bias_kind in BIAS_KINDS:
                prior = prior_for_ess(scores[bias_kind], fraction)
                achieved_prior_ess = effective_sample_size(prior) / len(prior)
                arms = {
                    "none": ("none", None, (0.0,)),
                    "maxent": ("maxent", None, strengths),
                    "bv_laplacian": ("laplacian", graph, strengths),
                    "structural_oracle": ("laplacian", oracle, strengths),
                    "rewired_laplacian": ("laplacian", rewired, strengths),
                }
                for arm, (kind, arm_graph, arm_strengths) in arms.items():
                    weights, strength, validation_mse = optimise_weights(
                        observables,
                        truth,
                        prior,
                        train,
                        validation,
                        arm_graph,
                        kind,
                        arm_strengths,
                        steps,
                    )
                    cluster_errors = [
                        abs(weights[labels == label].sum() - truth[labels == label].sum())
                        for label in np.unique(labels)
                    ]
                    output.append(
                        {
                            "system_id": data["system"],
                            "replica": replica,
                            "bias_kind": bias_kind,
                            "prior_ess_fraction": fraction,
                            "achieved_prior_ess_fraction": achieved_prior_ess,
                            "arm": arm,
                            "selected_strength": strength,
                            "validation_mse": validation_mse,
                            "test_mse": float(np.mean(np.square(flat[test] @ weights - target[test]))),
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
    """Pickle-friendly system task for spawn-based multiprocessing."""
    return reweight_system(*arguments)


def reweighting_gate(results: pd.DataFrame) -> dict:
    test = results[results.replica == 3]
    decisions = {}
    passed = True
    for bias in ("smooth", "basin"):
        block = test[test.bias_kind == bias].pivot_table(
            index=["system_id", "prior_ess_fraction"], columns="arm", values="weight_tv"
        )
        improvement = (block.maxent - block.bv_laplacian).dropna().to_numpy()
        low, high = bootstrap_interval(improvement, stable_seed(SEED, bias))
        mse = test[test.bias_kind == bias].pivot_table(
            index=["system_id", "prior_ess_fraction"], columns="arm", values="test_mse"
        )
        relative_mse = float(
            np.median(
                (mse.bv_laplacian - mse.maxent)
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
    # A shuffled bias has no physical locality to destroy. Restrict this negative
    # control to the two perturbations whose construction is structurally local.
    rewired = test[test.bias_kind.isin(("smooth", "basin"))].pivot_table(
        index=["system_id", "bias_kind", "prior_ess_fraction"],
        columns="arm",
        values="weight_tv",
    )
    physical_gain = float(np.median(rewired.maxent - rewired.bv_laplacian))
    rewired_gain = float(np.median(rewired.maxent - rewired.rewired_laplacian))
    control_passed = physical_gain > rewired_gain
    return {
        "passed": bool(passed and control_passed),
        "biases": decisions,
        "rewired_control_passed": bool(control_passed),
        "median_physical_gain": physical_gain,
        "median_rewired_gain": rewired_gain,
    }


def load_rows(full: bool, limit: int | None) -> list[dict]:
    rows = load_systems()
    if not full:
        pilot = pd.read_parquet(
            HERE
            / "outputs/analysis/pairwise_geometry/checkpoint26_pyrosetta_graph/pilot_systems.parquet"
        )
        identifiers = set(pilot.query("pilot").system_id)
        rows = [row for row in rows if row["system_id"] in identifiers]
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "reweight", "all"), default="all")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--frame-cap", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ess-fractions", default="0.8,0.4,0.2")
    parser.add_argument(
        "--strengths", default=",".join(str(value) for value in DEFAULT_STRENGTHS)
    )
    args = parser.parse_args()
    destination = OUTPUT / ("full" if args.full else "pilot")
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.full, args.limit)
    config = load_config()

    audit_path = destination / "laplacian_graph_audit.parquet"
    if args.phase in {"audit", "all"}:
        records = []
        for index, row in enumerate(rows, 1):
            records.extend(audit_system(row, config, args.permutations))
            print(f"[audit {index}/{len(rows)}] {row['system_id']}", flush=True)
        audit = pd.DataFrame(records)
        atomic_parquet(audit, audit_path)
    else:
        audit = pd.read_parquet(audit_path)
    selected, audit_gate = select_graph(audit)
    atomic_yaml(
        destination / "laplacian_graph_gate.yaml",
        {"selected": selected, "gate": audit_gate},
    )

    reweight_gate = None
    if args.phase in {"reweight", "all"}:
        ess_fractions = tuple(float(value) for value in args.ess_fractions.split(","))
        strengths = tuple(float(value) for value in args.strengths.split(","))
        parts = destination / "reweight_parts"
        parts.mkdir(parents=True, exist_ok=True)
        pending = [
            row
            for row in rows
            if not (parts / f"{row['system_id']}.parquet").exists()
        ]
        tasks = [
            (
                row,
                config,
                selected,
                ess_fractions,
                strengths,
                args.steps,
                args.frame_cap,
            )
            for row in pending
        ]
        executor = (
            ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            if args.workers > 1
            else None
        )
        evaluated = executor.map(reweight_task, tasks) if executor else map(reweight_task, tasks)
        try:
            for index, (row, system_records) in enumerate(
                zip(pending, evaluated, strict=True), 1
            ):
                atomic_parquet(
                    pd.DataFrame(system_records),
                    parts / f"{row['system_id']}.parquet",
                )
                print(f"[reweight {index}/{len(pending)}] {row['system_id']}", flush=True)
        finally:
            if executor:
                executor.shutdown()
        results = pd.concat(
            [pd.read_parquet(parts / f"{row['system_id']}.parquet") for row in rows],
            ignore_index=True,
        )
        atomic_parquet(results, destination / "laplacian_reweighting_results.parquet")
        reweight_gate = reweighting_gate(results)
        atomic_yaml(destination / "laplacian_reweighting_gate.yaml", reweight_gate)

    atomic_yaml(
        destination / "laplacian_prior_validation_report.yaml",
        {
            "systems": len(rows),
            "scope": "full" if args.full else "pilot",
            "selected": selected,
            "audit_gate": audit_gate,
            "reweighting_gate": reweight_gate,
            "iso_authorized": bool(
                args.full
                and audit_gate["passed"]
                and reweight_gate is not None
                and reweight_gate["passed"]
            ),
            "iso_policy": "ISO runner remains unchanged until full ATLAS authorization",
        },
    )


if __name__ == "__main__":
    main()
