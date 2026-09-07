"""Checkpoint 33: residue-scaled Work Scale kNN Laplacian graphs.

Stage 1 defines and audits the scaling grid. Reweighting and selection are added only
after the stage-1 artifacts have been reviewed.
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    atomic_yaml,
    load_config,
    load_systems,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_metric_comparison_checkpoint31 import (
    load_candidate_data,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_topology_checkpoint32 import (
    GraphSpec,
    bandwidth_from_quantile,
    challenge_cases,
    graph_diagnostics,
    metric_distances,
    optimise_graph_batch,
    result_row,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import (
    DEFAULT_STRENGTHS,
    bootstrap_interval,
    load_rows,
    optimise_weights,
    pseudo_uptake,
    relabel_graph,
    stable_seed,
    structural_clusters,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)
from jaxent.src.opt.loss.graph_laplacian import (
    FrameGraph,
    build_all_pairs_graph_from_distances,
    build_frame_graph_from_distances,
)


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint33_residue_k_scaling"
REFERENCE_RESIDUES = 109
K_REFERENCES = (10, 20, 40, 80, 120)
K_MIN = 3
K_MAX = 160
PLANNED_FRAME_CAP = 256
FAMILIES = ("constant", "sqrt", "linear", "n_log_n")
TOPOLOGIES = ("self_tuned_knn", "uniform_knn")
LOCKED_ALL_PAIRS = "work_scale__all_pairs_rbf__0p16"
FAMILY_PRIORITY = {name: index for index, name in enumerate(FAMILIES)}
TOPOLOGY_PRIORITY = {name: index for index, name in enumerate(TOPOLOGIES)}


@dataclass(frozen=True, slots=True)
class ResidueKSpec:
    """A normalized residue-count rule and its kNN edge weighting."""

    family: str
    k_reference: int
    topology: str

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"unknown residue scaling family: {self.family}")
        if self.k_reference <= 0:
            raise ValueError("k_reference must be positive")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology: {self.topology}")

    @property
    def key(self) -> str:
        return f"work_scale__{self.topology}__{self.family}__kref_{self.k_reference}"

    def effective_k(self, n_residues: int, n_frames: int) -> int:
        return residue_scaled_k(
            n_residues,
            n_frames,
            family=self.family,
            k_reference=self.k_reference,
        )


def family_value(n_residues: int, family: str) -> float:
    """Return the unnormalised value of a preregistered scaling family."""
    if n_residues <= 0:
        raise ValueError("n_residues must be positive")
    if family == "constant":
        return 1.0
    if family == "sqrt":
        return float(np.sqrt(n_residues))
    if family == "linear":
        return float(n_residues)
    if family == "n_log_n":
        return float(n_residues * np.log(n_residues))
    raise ValueError(f"unknown residue scaling family: {family}")


def residue_scaled_k(
    n_residues: int,
    n_frames: int,
    *,
    family: str,
    k_reference: int,
) -> int:
    """Calculate k using normalized scaling, half-up rounding, and hard caps."""
    if n_frames < 2:
        raise ValueError("at least two frames are required to build a graph")
    ratio = family_value(n_residues, family) / family_value(
        REFERENCE_RESIDUES, family
    )
    proposed = int(np.floor(k_reference * ratio + 0.5))
    return min(n_frames - 1, K_MAX, max(K_MIN, proposed))


def candidate_specs() -> tuple[ResidueKSpec, ...]:
    return tuple(
        ResidueKSpec(family, k_reference, topology)
        for topology in TOPOLOGIES
        for family in FAMILIES
        for k_reference in K_REFERENCES
    )


def build_scaling_graph(
    distances: np.ndarray,
    spec: ResidueKSpec,
    n_residues: int,
) -> FrameGraph:
    k = spec.effective_k(n_residues, len(distances))
    return build_frame_graph_from_distances(
        distances,
        k=k,
        metric=spec.key,
        weighted=spec.topology == "self_tuned_knn",
    )


def metadata_table(frame_cap: int) -> pd.DataFrame:
    metadata = pd.DataFrame(load_systems())[["system_id", "length"]].copy()
    metadata["n_residues"] = pd.to_numeric(metadata.pop("length"), errors="raise").astype(int)
    development_ids = {row["system_id"] for row in load_rows(False, None)}
    metadata["cohort"] = np.where(
        metadata.system_id.isin(development_ids), "development", "exploratory_87"
    )
    records = []
    for row in metadata.itertuples(index=False):
        for spec in candidate_specs():
            records.append(
                {
                    "system_id": row.system_id,
                    "cohort": row.cohort,
                    "n_residues": row.n_residues,
                    "assumed_n_frames": frame_cap,
                    "candidate": spec.key,
                    "topology": spec.topology,
                    "family": spec.family,
                    "k_reference": spec.k_reference,
                    "effective_k": spec.effective_k(row.n_residues, frame_cap),
                }
            )
    return pd.DataFrame(records)


def plot_scaling_curves(table: pd.DataFrame, path: Path) -> None:
    """Plot the proposed k values across the observed residue range."""
    development = table[
        (table.cohort == "development")
        & (table.topology == "self_tuned_knn")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for axis, family in zip(axes.flat, FAMILIES, strict=True):
        block = development[development.family == family]
        for k_reference, values in block.groupby("k_reference"):
            curve = (
                values[["n_residues", "effective_k"]]
                .drop_duplicates()
                .sort_values("n_residues")
            )
            axis.plot(
                curve.n_residues,
                curve.effective_k,
                marker="o",
                markersize=3,
                label=f"$k_{{ref}}={k_reference}$",
            )
        axis.axvline(REFERENCE_RESIDUES, color="black", ls="--", lw=1, alpha=0.5)
        axis.axhline(K_MAX, color="grey", ls=":", lw=1)
        axis.set_title(family.replace("_", " "))
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Number of residues")
    for axis in axes[:, 0]:
        axis.set_ylabel("Effective k")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        f"Work Scale residue-dependent neighbourhoods (reference: {REFERENCE_RESIDUES} residues)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def audit_system(inputs: tuple[dict, dict, int, int]) -> list[dict]:
    row, config, frame_cap, replica = inputs
    data, values = load_candidate_data(row, config)
    n_residues = int(row.get("length", data["z"].shape[0]))
    indices, structural_full = data["matrices"][replica]
    take = np.linspace(
        0, len(indices) - 1, min(frame_cap, len(indices)), dtype=int
    )
    selected_indices = np.asarray(indices)[take]
    structural = np.asarray(structural_full)[np.ix_(take, take)]
    distances = metric_distances(values["work_scale"], selected_indices, "work_scale")
    density = -distances.mean(axis=1)
    labels = structural_clusters(structural)
    null_source, null_target = np.triu_indices(len(selected_indices), k=1)
    null_distance = float(np.mean(structural[null_source, null_target]))
    output = []
    for spec in candidate_specs():
        graph = build_scaling_graph(distances, spec, n_residues)
        diagnostics = graph_diagnostics(graph, structural, density, labels)
        output.append(
            {
                "system_id": data["system"],
                "n_residues": n_residues,
                "n_frames": len(selected_indices),
                "replica": replica,
                "candidate": spec.key,
                "topology": spec.topology,
                "family": spec.family,
                "k_reference": spec.k_reference,
                "effective_k": spec.effective_k(n_residues, len(selected_indices)),
                **diagnostics,
                "structural_edge_gain": (
                    (null_distance - diagnostics["structural_edge_distance"])
                    / null_distance
                    if null_distance > 0
                    else 0.0
                ),
            }
        )
    return output


def system_lengths() -> dict[str, int]:
    return {
        str(row["system_id"]): int(row["length"])
        for row in load_systems()
    }


def candidate_graphs(
    distances: np.ndarray,
    n_residues: int,
    specs: tuple[ResidueKSpec, ...],
    *,
    include_all_pairs: bool,
) -> dict[str, FrameGraph]:
    graphs = {
        spec.key: build_scaling_graph(distances, spec, n_residues) for spec in specs
    }
    if include_all_pairs:
        bandwidth = bandwidth_from_quantile(distances, 0.16)
        graphs[LOCKED_ALL_PAIRS] = build_all_pairs_graph_from_distances(
            distances,
            metric=GraphSpec("work_scale", "all_pairs_rbf", 0.16).key,
            bandwidth=bandwidth,
        )
    return graphs


def reweight_system(inputs: tuple) -> list[dict]:
    (
        row,
        config,
        strengths,
        steps,
        frame_cap,
        replicas,
        selected_keys,
        include_all_pairs,
    ) = inputs
    data, values = load_candidate_data(row, config)
    n_residues = system_lengths()[data["system"]]
    all_specs = {spec.key: spec for spec in candidate_specs()}
    specs = (
        tuple(all_specs.values())
        if selected_keys is None
        else tuple(all_specs[key] for key in selected_keys)
    )
    output = []
    for replica in replicas:
        global_indices, structural_full = data["matrices"][replica]
        take = np.linspace(
            0, len(global_indices) - 1, min(frame_cap, len(global_indices)), dtype=int
        )
        indices = np.asarray(global_indices)[take]
        structural = np.asarray(structural_full)[np.ix_(take, take)]
        distances = metric_distances(values["work_scale"], indices, "work_scale")
        graphs = candidate_graphs(
            distances, n_residues, specs, include_all_pairs=include_all_pairs
        )
        rng = np.random.default_rng(
            stable_seed(data["system"], replica, "checkpoint33")
        )
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
        null_source, null_target = np.triu_indices(len(indices), k=1)
        null_distance = float(np.mean(structural[null_source, null_target]))
        graph_info = {}
        density = -distances.mean(axis=1)
        for key, graph in graphs.items():
            diagnostics = graph_diagnostics(graph, structural, density, labels)
            graph_info[key] = {
                "n_residues": n_residues,
                "n_frames": len(indices),
                "effective_k": graph.k if graph.k is not None else len(indices) - 1,
                "n_edges": diagnostics["n_edges"],
                "components": diagnostics["components"],
                "structural_edge_gain": (
                    (null_distance - diagnostics["structural_edge_distance"])
                    / null_distance
                    if null_distance > 0
                    else 0.0
                ),
            }
        for case in challenge_cases(labels, smooth, rng):
            prior = case["prior"]
            maxent = optimise_weights(
                observables,
                truth,
                prior,
                train,
                validation,
                None,
                "maxent",
                strengths,
                steps,
            )
            uniform = optimise_weights(
                observables,
                truth,
                prior,
                train,
                validation,
                None,
                "uniform_complete",
                strengths,
                steps,
            )
            graph_results = {}
            groups = {
                "scaled": [key for key in graphs if key != LOCKED_ALL_PAIRS],
                "all_pairs": [key for key in graphs if key == LOCKED_ALL_PAIRS],
            }
            for keys in groups.values():
                if not keys:
                    continue
                batch = {key: graphs[key] for key in keys}
                batch.update({f"{key}__rewired": rewired[key] for key in keys})
                graph_results.update(
                    optimise_graph_batch(
                        observables,
                        truth,
                        prior,
                        train,
                        validation,
                        batch,
                        strengths,
                        steps,
                    )
                )
            for key in graphs:
                for arm, result in {
                    "maxent": maxent,
                    "uniform_all_pairs": uniform,
                    "laplacian": graph_results[key],
                    "rewired_laplacian": graph_results[f"{key}__rewired"],
                }.items():
                    record = result_row(
                        data=data,
                        replica=replica,
                        candidate=key,
                        case=case,
                        arm=arm,
                        result=result,
                        truth=truth,
                        labels=labels,
                        flat=flat,
                        test=test,
                        target=target,
                        smooth=smooth,
                    )
                    record.update(graph_info[key])
                    output.append(record)
    return output


def run_parallel(
    rows: list[dict],
    inputs: list[tuple],
    parts: Path,
    workers: int,
    label: str,
) -> pd.DataFrame:
    parts.mkdir(parents=True, exist_ok=True)
    pending = [
        (row, inputs[index])
        for index, row in enumerate(rows)
        if not (parts / f"{row['system_id']}.parquet").exists()
    ]
    context = multiprocessing.get_context("spawn")
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=context)
        if workers > 1
        else None
    )
    try:
        batches = (
            executor.map(reweight_system, [item[1] for item in pending])
            if executor
            else map(reweight_system, [item[1] for item in pending])
        )
        for index, ((row, _), records) in enumerate(zip(pending, batches, strict=True), 1):
            atomic_parquet(pd.DataFrame(records), parts / f"{row['system_id']}.parquet")
            print(f"[{label} {index}/{len(pending)}] {row['system_id']}", flush=True)
    finally:
        if executor:
            executor.shutdown()
    return pd.concat(
        [pd.read_parquet(parts / f"{row['system_id']}.parquet") for row in rows],
        ignore_index=True,
    )


def run_audit_parallel(
    rows: list[dict], config: dict, frame_cap: int, workers: int, parts: Path
) -> pd.DataFrame:
    parts.mkdir(parents=True, exist_ok=True)
    pending = [row for row in rows if not (parts / f"{row['system_id']}.parquet").exists()]
    inputs = [(row, config, frame_cap, 2) for row in pending]
    context = multiprocessing.get_context("spawn")
    executor = (
        ProcessPoolExecutor(max_workers=workers, mp_context=context)
        if workers > 1
        else None
    )
    try:
        batches = executor.map(audit_system, inputs) if executor else map(audit_system, inputs)
        for index, (row, records) in enumerate(zip(pending, batches, strict=True), 1):
            atomic_parquet(pd.DataFrame(records), parts / f"{row['system_id']}.parquet")
            print(f"[audit33 {index}/{len(pending)}] {row['system_id']}", flush=True)
    finally:
        if executor:
            executor.shutdown()
    return pd.concat(
        [pd.read_parquet(parts / f"{row['system_id']}.parquet") for row in rows],
        ignore_index=True,
    )


def add_size_quartile(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    lengths = result[["system_id", "n_residues"]].drop_duplicates()
    lengths["size_quartile"] = pd.qcut(
        lengths.n_residues,
        4,
        labels=("Q1", "Q2", "Q3", "Q4"),
    )
    return result.merge(lengths, on=["system_id", "n_residues"], validate="many_to_one")


def candidate_ranking(results: pd.DataFrame, replica: int) -> pd.DataFrame:
    block = results[
        (results.replica == replica) & (results.candidate != LOCKED_ALL_PAIRS)
    ]
    pivot = block.pivot_table(
        index=["system_id", "n_residues", "candidate", "bias_kind", "bias_level"],
        columns="arm",
        values="weight_tv",
    ).reset_index()
    pivot["gain"] = pivot.maxent - pivot.laplacian
    system = (
        pivot[pivot.bias_kind.isin(("smooth", "basin"))]
        .groupby(
            ["system_id", "n_residues", "candidate", "bias_kind"], as_index=False
        )
        .gain.median()
    )
    system = add_size_quartile(system)
    mse = block[block.bias_kind.isin(("smooth", "basin"))].pivot_table(
        index=["system_id", "candidate", "bias_kind", "bias_level"],
        columns="arm",
        values="test_mse",
    )
    mse["relative_change"] = (mse.laplacian - mse.maxent) / np.maximum(
        mse.maxent, np.finfo(float).eps
    )
    records = []
    specs = {spec.key: spec for spec in candidate_specs()}
    for candidate, values in system.groupby("candidate"):
        strata = values.groupby(["bias_kind", "size_quartile"], observed=True).gain.mean()
        by_system = values.pivot(index="system_id", columns="bias_kind", values="gain")
        conservative = by_system[["smooth", "basin"]].min(axis=1)
        spec = specs[candidate]
        source = block[block.candidate == candidate]
        records.append(
            {
                "candidate": candidate,
                "family": spec.family,
                "topology": spec.topology,
                "k_reference": spec.k_reference,
                "score": float(strata.min()),
                "score_se": float(conservative.std(ddof=1) / np.sqrt(len(conservative))),
                "aggregate_smooth_gain": float(by_system.smooth.mean()),
                "aggregate_basin_gain": float(by_system.basin.mean()),
                "q4_basin_gain": float(strata.loc[("basin", "Q4")]),
                "median_relative_test_mse_change": float(
                    mse.xs(candidate, level="candidate").relative_change.median()
                ),
                "maximum_components": int(source.components.max()),
                "median_structural_edge_gain": float(source.structural_edge_gain.median()),
                "median_edges": float(source.n_edges.median()),
            }
        )
    ranking = pd.DataFrame(records)
    ranking["eligible"] = (
        (ranking.maximum_components == 1)
        & (ranking.median_structural_edge_gain > 0)
        & (ranking.median_relative_test_mse_change <= 0.01)
    )
    return ranking


def within_one_se(block: pd.DataFrame) -> pd.DataFrame:
    eligible = block[block.eligible]
    if eligible.empty:
        raise RuntimeError("no residue-scaling candidate passed development eligibility")
    best = eligible.sort_values("score", ascending=False).iloc[0]
    return eligible[eligible.score >= best.score - best.score_se]


def select_development_candidate(results: pd.DataFrame) -> dict:
    ranking_a = candidate_ranking(results, 1)
    finalists = sorted(results.loc[results.replica == 2, "candidate"].unique())
    ranking_b = candidate_ranking(results[results.candidate.isin(finalists)], 2)
    near = within_one_se(ranking_b).copy()
    near["family_priority"] = near.family.map(FAMILY_PRIORITY)
    near["topology_priority"] = near.topology.map(TOPOLOGY_PRIORITY)
    selected = near.sort_values(
        ["family_priority", "topology_priority", "median_edges", "k_reference"]
    ).iloc[0]
    # The constant graph is a required negative control, not a selectable method;
    # retain its best-scoring finalist even when it fails an eligibility constraint.
    constant = ranking_b[ranking_b.family == "constant"]
    best_constant = constant.sort_values("score", ascending=False).iloc[0]
    return {
        "selected": str(selected.candidate),
        "best_constant": str(best_constant.candidate),
        "replica_a_finalists": finalists,
        "replica_a_ranking": ranking_a.to_dict("records"),
        "replica_b_ranking": ranking_b.to_dict("records"),
    }


def select_family_finalists(
    screening: pd.DataFrame, connectivity_audit: pd.DataFrame
) -> tuple[str, ...]:
    ranking = candidate_ranking(screening, 1)
    connected = set(
        connectivity_audit.groupby("candidate").components.max().loc[lambda value: value == 1].index
    )
    ranking = ranking[ranking.candidate.isin(connected)]
    finalists = []
    for (_, _), block in ranking.groupby(["family", "topology"]):
        choice = within_one_se(block).sort_values(
            ["median_edges", "k_reference"]
        ).iloc[0]
        finalists.append(str(choice.candidate))
    return tuple(finalists)


def bootstrap(values: pd.Series, *seed_parts: object) -> tuple[float, float]:
    return bootstrap_interval(values.to_numpy(), stable_seed("checkpoint33", *seed_parts))


def recovery_table(results: pd.DataFrame) -> pd.DataFrame:
    pivot = results.pivot_table(
        index=["system_id", "n_residues", "candidate", "bias_kind", "bias_level"],
        columns="arm",
        values="weight_tv",
    ).reset_index()
    pivot["gain"] = pivot.maxent - pivot.laplacian
    system = (
        pivot[pivot.bias_kind.isin(("smooth", "basin"))]
        .groupby(
            ["system_id", "n_residues", "candidate", "bias_kind"], as_index=False
        )
        .gain.median()
    )
    system = add_size_quartile(system)
    rows = []
    for (candidate, bias, quartile), block in system.groupby(
        ["candidate", "bias_kind", "size_quartile"], observed=True
    ):
        low, high = bootstrap(block.gain, candidate, bias, quartile)
        rows.append(
            {
                "candidate": candidate,
                "bias_kind": bias,
                "size_quartile": str(quartile),
                "systems": len(block),
                "mean_gain": float(block.gain.mean()),
                "median_gain": float(block.gain.median()),
                "ci_low": low,
                "ci_high": high,
                "positive_system_fraction": float((block.gain > 0).mean()),
            }
        )
    return pd.DataFrame(rows), system


def short_label(candidate: str) -> str:
    if candidate == LOCKED_ALL_PAIRS:
        return "all-pairs RBF 16%"
    parts = candidate.split("__")
    return f"{parts[2]} {parts[3].replace('kref_', 'kref=')} ({'weighted' if parts[1] == 'self_tuned_knn' else 'uniform'})"


def plot_size_recovery(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    order = ("Q1", "Q2", "Q3", "Q4")
    for axis, bias in zip(axes, ("smooth", "basin"), strict=True):
        for candidate, block in summary[summary.bias_kind == bias].groupby("candidate"):
            values = block.set_index("size_quartile").reindex(order)
            axis.errorbar(
                range(4),
                values.mean_gain,
                yerr=[values.mean_gain - values.ci_low, values.ci_high - values.mean_gain],
                marker="o",
                capsize=3,
                label=short_label(candidate),
            )
        axis.axhline(0, color="black", lw=1)
        axis.set_title(bias)
        axis.set_xticks(range(4), order)
        axis.set_xlabel("Protein-length quartile")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean TV gain over MaxEnt")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_recovery_length(system: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for axis, bias in zip(axes, ("smooth", "basin"), strict=True):
        for candidate, block in system[system.bias_kind == bias].groupby("candidate"):
            axis.scatter(block.n_residues, block.gain, s=22, alpha=0.55)
            if block.n_residues.nunique() > 1:
                coefficient = np.polyfit(block.n_residues, block.gain, 1)
                x = np.linspace(block.n_residues.min(), block.n_residues.max(), 100)
                axis.plot(x, np.polyval(coefficient, x), label=short_label(candidate))
        axis.axhline(0, color="black", lw=1)
        axis.set_title(bias)
        axis.set_xlabel("Number of residues")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("System-level median TV gain over MaxEnt")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selection_tradeoff(ranking: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 7))
    for family, block in ranking.groupby("family"):
        axis.scatter(
            block.aggregate_basin_gain,
            block.q4_basin_gain,
            s=np.clip(block.median_edges / 20, 25, 250),
            alpha=0.7,
            label=family.replace("_", " "),
        )
    axis.axhline(0, color="black", lw=1)
    axis.axvline(0, color="black", lw=1)
    axis.set_xlabel("Aggregate basin TV gain")
    axis.set_ylabel("Q4 basin TV gain")
    axis.set_title("Replica-B scaling selection; marker area follows edge count")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def development_gate(results: pd.DataFrame, selected: str, constant: str) -> dict:
    summary, system = recovery_table(results)
    chosen = summary[summary.candidate == selected]
    strata_pass = bool((chosen.ci_low > 0).all())
    comparisons = {}
    for control in (constant, LOCKED_ALL_PAIRS):
        selected_basin = system[
            (system.candidate == selected)
            & (system.bias_kind == "basin")
            & (system.size_quartile == "Q4")
        ].set_index("system_id").gain
        control_basin = system[
            (system.candidate == control)
            & (system.bias_kind == "basin")
            & (system.size_quartile == "Q4")
        ].set_index("system_id").gain
        difference = (selected_basin - control_basin).dropna()
        low, high = bootstrap(difference, "comparison", selected, control)
        comparisons[control] = {
            "mean_q4_basin_advantage": float(difference.mean()),
            "ci": [low, high],
            "passed": bool(low > 0),
        }
    selected_rows = results[results.candidate == selected]
    mse = selected_rows[selected_rows.bias_kind.isin(("smooth", "basin"))].pivot_table(
        index=["system_id", "bias_kind", "bias_level"], columns="arm", values="test_mse"
    )
    mse_change = float(
        np.median((mse.laplacian - mse.maxent) / np.maximum(mse.maxent, np.finfo(float).eps))
    )
    tv = selected_rows[selected_rows.bias_kind.isin(("smooth", "basin"))].pivot_table(
        index=["system_id", "bias_kind", "bias_level"], columns="arm", values="weight_tv"
    )
    physical_gain = float(np.median(tv.maxent - tv.laplacian))
    rewired_gain = float(np.median(tv.maxent - tv.rewired_laplacian))
    uniform_gain = float(np.median(tv.maxent - tv.uniform_all_pairs))
    controls_pass = physical_gain > max(rewired_gain, uniform_gain)
    adaptive_selected = "__constant__" not in selected
    passed = bool(
        adaptive_selected
        and strata_pass
        and all(value["passed"] for value in comparisons.values())
        and mse_change <= 0.01
        and controls_pass
    )
    return {
        "passed": passed,
        "adaptive_rule_selected": adaptive_selected,
        "all_size_strata_positive": strata_pass,
        "comparisons": comparisons,
        "median_relative_test_mse_change": mse_change,
        "controls_passed": controls_pass,
        "median_selected_gain": physical_gain,
        "median_rewired_gain": rewired_gain,
        "median_uniform_all_pairs_gain": uniform_gain,
    }


def run_development(config: dict, args: argparse.Namespace) -> None:
    destination = OUTPUT / "development"
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_rows(False, args.limit)
    strengths = tuple(float(value) for value in args.strengths.split(","))
    screening = run_parallel(
        rows,
        [
            (
                row,
                config,
                strengths,
                args.screen_steps,
                args.screen_frame_cap,
                (1,),
                None,
                False,
            )
            for row in rows
        ],
        destination / "screen_parts",
        args.workers,
        "screen33",
    )
    atomic_parquet(screening, destination / "screening_reweighting.parquet")
    connectivity = run_audit_parallel(
        rows,
        config,
        args.frame_cap,
        args.workers,
        destination / "replica_b_audit_parts",
    )
    atomic_parquet(connectivity, destination / "replica_b_graph_audit.parquet")
    finalists = select_family_finalists(screening, connectivity)
    atomic_yaml(destination / "family_finalists.yaml", {"finalists": finalists})
    development = run_parallel(
        rows,
        [
            (
                row,
                config,
                strengths,
                args.steps,
                args.frame_cap,
                (2,),
                finalists,
                False,
            )
            for row in rows
        ],
        destination / "develop_connected_parts",
        args.workers,
        "develop33",
    )
    tuning = pd.concat((screening, development), ignore_index=True)
    atomic_parquet(tuning, destination / "tuning_reweighting.parquet")
    selection = select_development_candidate(tuning)
    atomic_yaml(destination / "selection.yaml", selection)
    selected_keys = tuple(dict.fromkeys((selection["selected"], selection["best_constant"])))
    heldout = run_parallel(
        rows,
        [
            (
                row,
                config,
                strengths,
                args.steps,
                args.frame_cap,
                (3,),
                selected_keys,
                True,
            )
            for row in rows
        ],
        destination / "test_connected_parts",
        args.workers,
        "test33",
    )
    atomic_parquet(heldout, destination / "heldout_reweighting.parquet")
    summary, system = recovery_table(heldout)
    summary.to_csv(destination / "development_size_recovery.csv", index=False)
    atomic_parquet(system, destination / "development_system_recovery.parquet")
    ranking_b = candidate_ranking(
        tuning[tuning.candidate.isin(selection["replica_a_finalists"])], 2
    )
    ranking_b.to_csv(destination / "development_selection_table.csv", index=False)
    plot_size_recovery(summary, destination / "development_size_recovery.png")
    plot_recovery_length(system, destination / "development_recovery_vs_length.png")
    plot_selection_tradeoff(ranking_b, destination / "development_selection_tradeoff.png")
    gate = development_gate(
        heldout, selection["selected"], selection["best_constant"]
    )
    atomic_yaml(destination / "development_gate.yaml", gate)
    atomic_yaml(
        destination / "development_report.yaml",
        {
            "selected": selection["selected"],
            "best_constant": selection["best_constant"],
            "systems": len(rows),
            "frame_cap": args.frame_cap,
            "steps": args.steps,
            "gate": gate,
            "next_stage": "exploratory 87-system diagnostic requires explicit approval",
        },
    )


def run_smoke(rows: list[dict], config: dict, args: argparse.Namespace) -> None:
    destination = OUTPUT / "smoke"
    destination.mkdir(parents=True, exist_ok=True)
    inputs = [(row, config, args.frame_cap, 1) for row in rows]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        records = [record for batch in executor.map(audit_system, inputs) for record in batch]
    atomic_parquet(pd.DataFrame(records), destination / "graph_audit.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "develop"), default="prepare"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--frame-cap", type=int, default=256)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--screen-steps", type=int, default=100)
    parser.add_argument("--screen-frame-cap", type=int, default=192)
    parser.add_argument("--strengths", default=",".join(map(str, DEFAULT_STRENGTHS)))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    destination = OUTPUT / ("smoke" if args.smoke else "development")
    destination.mkdir(parents=True, exist_ok=True)
    # Keep the preregistered curves comparable even when a small smoke frame cap is used.
    table = metadata_table(PLANNED_FRAME_CAP)
    table.to_csv(destination / "candidate_scaling_table.csv", index=False)
    plot_scaling_curves(table, destination / "k_scaling_curves.png")

    if args.smoke:
        rows = load_rows(False, args.limit or 1)
        run_smoke(rows, load_config(), args)
    elif args.phase == "develop":
        run_development(load_config(), args)


if __name__ == "__main__":
    main()
