"""Checkpoint 40: diagnose and intervene on replica population-transfer failure."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas-cp40-matplotlib")

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.special import logsumexp
from scipy.stats import spearmanr
import yaml

from jaxent.examples.ATLAS_BV.analysis import (
    corrected_variance_recovery_checkpoint36 as cp36,
)
from jaxent.examples.ATLAS_BV.analysis import (
    sparse_population_extrapolation_checkpoint39 as cp39,
)
from jaxent.examples.ATLAS_BV.analysis import transferability_gap_checkpoint38 as cp38
from jaxent.examples.ATLAS_BV.analysis.common import (
    HERE,
    histogram_edges,
    integrated_autocorrelation_frames,
)
from jaxent.examples.ATLAS_BV.analysis.local_variance_checkpoint28 import nearest
from jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 import (
    align_to_structure,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)


OUTPUT = (
    HERE / "outputs/analysis/pairwise_geometry/checkpoint40_replica_sampling_mechanism"
)
CP38 = cp38.OUTPUT
CP39 = cp39.OUTPUT
PRIMARY = ("pf_l1", "work_opt")
LABEL_COUNTS = (2, 4, 8, 16, 32)
TRACE_NAMES = ("rmsd", "rg", "global_pf", "work_opt")
SEED = 20260926


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def fixed_js(left: np.ndarray, right: np.ndarray, reference: np.ndarray) -> float:
    """Histogram Jensen-Shannon divergence in bits with reference-fixed edges."""
    edges = histogram_edges(np.asarray(reference), np.asarray(reference), 10, 30)
    a = np.histogram(left, edges)[0].astype(float) + 1e-12
    b = np.histogram(right, edges)[0].astype(float) + 1e-12
    return float(jensenshannon(a, b, base=2.0) ** 2)


def profile_js(
    left: np.ndarray, right: np.ndarray, reference: np.ndarray
) -> tuple[float, float]:
    """Mean and median per-component JSD for feature-by-frame profiles."""
    values = np.asarray(
        [
            fixed_js(left[index], right[index], reference[index])
            for index in range(len(reference))
        ]
    )
    return float(values.mean()), float(np.median(values))


def batch_means_neff(values: np.ndarray, block: int) -> float:
    """Variance-of-the-mean effective size from non-overlapping batch means."""
    values = np.asarray(values, dtype=float)
    groups = len(values) // block
    if groups < 2 or np.var(values, ddof=1) <= 0:
        return 1.0
    means = values[: groups * block].reshape(groups, block).mean(axis=1)
    variance_of_mean = np.var(means, ddof=1) / groups
    if variance_of_mean <= 0:
        return float(len(values))
    estimate = np.var(values, ddof=1) / variance_of_mean
    return float(np.clip(estimate, 1.0, len(values)))


def trace_neff(values: np.ndarray) -> tuple[int, float, dict[int, float]]:
    tau = integrated_autocorrelation_frames(values)
    return (
        tau,
        float(len(values) / tau),
        {block: batch_means_neff(values, block) for block in (8, 16, 32)},
    )


def local_estimates(
    distance: np.ndarray,
    landmarks: np.ndarray,
    indices: np.ndarray,
    bandwidth: float,
    leave_one_out: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel log masses, normalized weights, and Kish support for arbitrary frames."""
    log_kernel = -0.5 * np.square(
        distance[np.ix_(landmarks, indices)] / max(bandwidth, np.finfo(float).eps)
    )
    denominator = np.full(len(landmarks), len(indices), dtype=float)
    if leave_one_out:
        lookup = {int(value): position for position, value in enumerate(indices)}
        for row, landmark in enumerate(landmarks):
            if int(landmark) in lookup:
                log_kernel[row, lookup[int(landmark)]] = -np.inf
                denominator[row] -= 1
    normalizer = logsumexp(log_kernel, axis=1)
    weights = np.exp(log_kernel - normalizer[:, None])
    invalid = ~np.isfinite(weights).all(axis=1)
    for row in np.flatnonzero(invalid):
        weights[row] = 0.0
        weights[row, np.argmin(distance[landmarks[row], indices])] = 1.0
    return (
        normalizer - np.log(np.maximum(denominator, 1.0)),
        weights,
        1.0 / np.sum(np.square(weights), axis=1),
    )


def variance_coordinate(
    values: np.ndarray,
    structural: np.ndarray,
    indices: np.ndarray,
    k: int,
    shrinkage: float,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Checkpoint-39 local log-variance coordinate on an arbitrary frame subset."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    neighbours = nearest(structural[np.ix_(indices, indices)], min(k, len(indices) - 1))
    variance = values[:, indices][:, neighbours].var(axis=2)
    if reference is None:
        reference = np.nanmedian(np.where(variance > 0, variance, np.nan), axis=1)
        reference = np.where(np.isfinite(reference) & (reference > 0), reference, 1.0)
    coordinate = 0.5 * np.mean(
        np.log(variance + shrinkage * reference[:, None]), axis=0
    )
    return coordinate, reference


def subset_descriptors(
    base: dict[str, np.ndarray],
    structural: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    variance_parameters: dict[str, tuple[int, float]],
    variance_references: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    descriptors: dict[str, np.ndarray] = {}
    references: dict[str, np.ndarray] = {}
    for metric in (*cp39.DIRECT_BASE, "rg"):
        descriptors[metric] = cp39.region_descriptor(base[metric], weights, indices)
    for metric in cp39.VARIANCE_METRICS:
        k, shrinkage = variance_parameters[metric]
        coordinate, reference = variance_coordinate(
            base[metric],
            structural,
            indices,
            k,
            shrinkage,
            None if variance_references is None else variance_references[metric],
        )
        descriptors[f"{metric}__variance"] = cp39.region_descriptor(
            coordinate, weights, np.arange(len(indices))
        )
        references[metric] = reference
    return descriptors, references


def system_arrays(row: dict) -> dict:
    folder = cp39.GRAPH_ROOT / "systems" / row["system_id"]
    with np.load(folder / "frames.npz", allow_pickle=False) as archive:
        ca = np.asarray(archive["ca"], dtype=float)
        frames = np.asarray(archive["frames"], dtype=int)
        replicas = np.asarray(archive["replicas"], dtype=int)
        logpf = np.asarray(archive["logpf"], dtype=float)
    with np.load(folder / "graph_w1.npz", allow_pickle=False) as archive:
        structural = np.asarray(archive["distance"], dtype=float)
    universe = mda.Universe(HERE / row["pdb_path"])
    reference = universe.select_atoms("protein and name CA").positions.copy()
    aligned = align_to_structure(ca, reference)
    centered_reference = reference - reference.mean(axis=0)
    rmsd = np.sqrt(
        np.mean(np.sum(np.square(aligned - centered_reference), axis=2), axis=1)
    )
    base = {name: value for name, (value, _) in cp39.representations(logpf, ca).items()}
    work_opt = np.mean(np.atleast_2d(base["work_opt"]), axis=0)
    return {
        "folder": folder,
        "ca": ca,
        "frames": frames,
        "replicas": replicas,
        "logpf": logpf,
        "structural": structural,
        "base": base,
        "traces": {
            "rmsd": rmsd,
            "rg": np.asarray(base["rg"]).ravel(),
            "global_pf": logpf.mean(axis=0),
            "work_opt": work_opt,
        },
    }


def overlap_and_neff(row: dict, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    system = row["system_id"]
    replicas = data["replicas"]
    comparisons = {
        "AB_C": (
            np.flatnonzero(np.isin(replicas, (1, 2))),
            np.flatnonzero(replicas == 3),
        ),
        "A_B": (np.flatnonzero(replicas == 1), np.flatnonzero(replicas == 2)),
        "A_C": (np.flatnonzero(replicas == 1), np.flatnonzero(replicas == 3)),
        "B_C": (np.flatnonzero(replicas == 2), np.flatnonzero(replicas == 3)),
    }
    axes: dict[str, np.ndarray] = dict(data["traces"])
    axes["pf_profile"] = data["logpf"]
    axes["work_opt_profile"] = np.atleast_2d(data["base"]["work_opt"])
    overlap_rows = []
    for comparison, (left, right) in comparisons.items():
        for axis, values in axes.items():
            if np.asarray(values).ndim == 1:
                mean_js = median_js = fixed_js(values[left], values[right], values)
            else:
                mean_js, median_js = profile_js(
                    values[:, left], values[:, right], values
                )
            overlap_rows.append(
                {
                    "system_id": system,
                    "comparison": comparison,
                    "axis": axis,
                    "js_bits": mean_js,
                    "median_component_js_bits": median_js,
                    "left_frames": len(left),
                    "right_frames": len(right),
                }
            )
    neff_rows = []
    for replica in cp39.REPLICAS:
        indices = np.flatnonzero(replicas == replica)
        order = np.argsort(data["frames"][indices], kind="stable")
        for trace, values in data["traces"].items():
            ordered = np.asarray(values)[indices][order]
            tau, neff, batches = trace_neff(ordered)
            neff_rows.append(
                {
                    "system_id": system,
                    "replica": replica,
                    "trace": trace,
                    "frames": len(ordered),
                    "tau_frames": tau,
                    "neff": neff,
                    **{
                        f"batch_neff_{block}": value for block, value in batches.items()
                    },
                }
            )
    return pd.DataFrame(overlap_rows), pd.DataFrame(neff_rows)


def temporal_analysis(
    row: dict,
    data: dict,
    landmarks: int,
    repeats: int,
    label_counts: tuple[int, ...],
    variance_parameters: dict[str, tuple[int, float]],
) -> pd.DataFrame:
    system = row["system_id"]
    replicas = data["replicas"]
    c = np.flatnonzero(replicas == 3)
    c = c[np.argsort(data["frames"][c], kind="stable")]
    midpoint = len(c) // 2
    halves = {"C1": c[:midpoint], "C2": c[midpoint:]}
    rows = []
    for source_name, target_name in (("C1", "C2"), ("C2", "C1")):
        source, target = halves[source_name], halves[target_name]
        local_order = cp39.maximin_order(data["structural"][np.ix_(source, source)])
        centers = source[local_order[: min(landmarks, len(source))]]
        bandwidth = cp39.neighbour_bandwidth(
            data["structural"][np.ix_(source, source)], 10
        )
        source_values, source_weights, _ = local_estimates(
            data["structural"], centers, source, bandwidth, True
        )
        target_values, target_weights, _ = local_estimates(
            data["structural"], centers, target, bandwidth, False
        )
        source_desc, references = subset_descriptors(
            data["base"],
            data["structural"],
            source,
            source_weights,
            variance_parameters,
        )
        target_desc, _ = subset_descriptors(
            data["base"],
            data["structural"],
            target,
            target_weights,
            variance_parameters,
            references,
        )
        center_w1 = data["structural"][np.ix_(centers, centers)]
        within = cp39.build_metric_matrices(source_desc, source_desc)
        cross = cp39.build_metric_matrices(source_desc, target_desc)
        within[("w1", "direct")] = (center_w1, center_w1)
        cross[("w1", "direct")] = (center_w1, center_w1)
        modes = (
            (f"within_{source_name}", source_values, source_values, within),
            (f"{source_name}_to_{target_name}", source_values, target_values, cross),
        )
        for mode, source_population, target_population, matrices in modes:
            for (metric, model), (fit_distance, query_distance) in matrices.items():
                block, _ = cp39.evaluation_rows(
                    system=system,
                    unit="temporal_neighbourhood",
                    mode=mode,
                    metric=metric,
                    model=model,
                    fit_distance=fit_distance,
                    query_distance=query_distance,
                    structural_distance=center_w1,
                    source_values=source_population,
                    target_values=target_population,
                    recovery_reference=source_values,
                    label_counts=label_counts,
                    repeats=repeats,
                    global_alpha=None,
                    selection="random",
                )
                rows.extend(block)
    return cp39.aggregate_repetitions(pd.DataFrame(rows))


def pooled_analysis(
    row: dict,
    data: dict,
    landmarks: int,
    repeats: int,
    label_counts: tuple[int, ...],
    variance_parameters: dict[str, tuple[int, float]],
) -> pd.DataFrame:
    system = row["system_id"]
    replicas = data["replicas"]
    structural = data["structural"]
    centers = cp39.maximin_order(structural)[: min(landmarks, len(structural))]
    a = np.flatnonzero(replicas == 1)
    ab = np.flatnonzero(np.isin(replicas, (1, 2)))
    abc = np.arange(len(replicas))
    bandwidth = cp39.neighbour_bandwidth(structural[np.ix_(a, a)], 10)
    targets, weights, _ = cp39.local_targets_and_weights(
        structural, centers, replicas, bandwidth
    )
    target_ab = logsumexp(np.stack([targets[1], targets[2]]), axis=0) - np.log(2.0)
    replica_desc = {}
    references = None
    for replica in cp39.REPLICAS:
        indices = np.flatnonzero(replicas == replica)
        replica_desc[replica], refs = subset_descriptors(
            data["base"],
            structural,
            indices,
            weights[replica],
            variance_parameters,
            references,
        )
        if references is None:
            references = refs
    source_ab = {
        key: cp39.pooled_descriptor(
            replica_desc[1][key], replica_desc[2][key], targets[1], targets[2]
        )
        for key in replica_desc[1]
    }
    _, ab_weights, _ = local_estimates(structural, centers, ab, bandwidth, True)
    _, abc_weights, _ = local_estimates(structural, centers, abc, bandwidth, True)
    ab_desc, _ = subset_descriptors(
        data["base"], structural, ab, ab_weights, variance_parameters
    )
    abc_desc, _ = subset_descriptors(
        data["base"], structural, abc, abc_weights, variance_parameters
    )
    matrices = {
        "original_AB_to_C": cp39.build_metric_matrices(source_ab, replica_desc[3]),
        "AB_common": cp39.build_metric_matrices(ab_desc, ab_desc),
        "ABC_common": cp39.build_metric_matrices(abc_desc, abc_desc),
    }
    center_w1 = structural[np.ix_(centers, centers)]
    for value in matrices.values():
        value[("w1", "direct")] = (center_w1, center_w1)
    rows = []
    for estimator, metric_matrices in matrices.items():
        for (metric, model), (fit_distance, query_distance) in metric_matrices.items():
            block, _ = cp39.evaluation_rows(
                system=system,
                unit="pooled_neighbourhood",
                mode=estimator,
                metric=metric,
                model=model,
                fit_distance=fit_distance,
                query_distance=query_distance,
                structural_distance=center_w1,
                source_values=target_ab,
                target_values=targets[3],
                recovery_reference=target_ab,
                label_counts=label_counts,
                repeats=repeats,
                global_alpha=None,
                selection="random",
            )
            rows.extend(block)
    return cp39.aggregate_repetitions(pd.DataFrame(rows))


def variance_parameters(
    system: str, fits: pd.DataFrame
) -> dict[str, tuple[int, float]]:
    block = fits[(fits.system_id == system) & (fits.model == "variance_magnitude")]
    return {
        row.metric: (int(row.k), float(row.shrinkage)) for row in block.itertuples()
    }


def analyse_system(arguments) -> str:
    row, components, landmarks, repeats, label_counts, fits_path, parts = arguments
    system = row["system_id"]
    data = system_arrays(row)
    fits = pd.read_parquet(fits_path)
    parameters = variance_parameters(system, fits)
    if "overlap" in components or "neff" in components:
        overlap, neff = overlap_and_neff(row, data)
        if "overlap" in components:
            atomic_parquet(overlap, parts / f"{system}.overlap.parquet")
        if "neff" in components:
            atomic_parquet(neff, parts / f"{system}.neff.parquet")
    if "temporal" in components:
        atomic_parquet(
            temporal_analysis(row, data, landmarks, repeats, label_counts, parameters),
            parts / f"{system}.temporal.parquet",
        )
    if "pool" in components:
        atomic_parquet(
            pooled_analysis(row, data, landmarks, repeats, label_counts, parameters),
            parts / f"{system}.pool.parquet",
        )
    audit = {
        "system_id": system,
        "components": sorted(components),
        "frames": len(data["replicas"]),
        "source_complete_sha256": cp36.digest(data["folder"] / "complete.json"),
    }
    path = parts / f"{system}.audit.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    temporary.replace(path)
    return system


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, seed: int, samples: int
) -> tuple[float, float, float]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    estimate = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for index in range(samples):
        take = rng.integers(0, len(x), len(x))
        if np.ptp(x[take]) == 0 or np.ptp(y[take]) == 0:
            value = 0.0
        else:
            value = spearmanr(x[take], y[take]).statistic
        draws[index] = 0.0 if not np.isfinite(value) else value
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def permutation_p(
    x: np.ndarray, y: np.ndarray, estimate: float, seed: int, draws: int
) -> float:
    if len(x) < 3 or not np.isfinite(estimate):
        return np.nan
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [spearmanr(rng.permutation(x), y).statistic for _ in range(draws)]
    )
    return float((1 + np.sum(np.abs(null) >= abs(estimate))) / (draws + 1))


def loocv_rmse(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-system-out RMSE for a univariate linear model."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    predictions = np.empty(len(x))
    for heldout in range(len(x)):
        keep = np.arange(len(x)) != heldout
        scale = x[keep].std(ddof=1)
        z_train = (x[keep] - x[keep].mean()) / (scale if scale > 0 else 1.0)
        coefficients = np.linalg.lstsq(
            np.column_stack([np.ones(keep.sum()), z_train]), y[keep], rcond=None
        )[0]
        z_test = (x[heldout] - x[keep].mean()) / (scale if scale > 0 else 1.0)
        predictions[heldout] = coefficients[0] + coefficients[1] * z_test
    return float(np.sqrt(np.mean(np.square(predictions - y))))


def assignment_outcomes(local: pd.DataFrame) -> pd.DataFrame:
    block = local[
        (local["selection"] == "random")
        & (local["stratum"] == "all")
        & (local["method"] == "sparse_alpha")
        & (local["model"] == "direct")
        & local["metric"].isin(PRIMARY)
    ]
    pivot = block.pivot_table(
        index=["system_id", "metric", "known"], columns="mode", values="spearman"
    ).reset_index()
    pivot["assignment_drop"] = pivot.within_C - pivot.AB_to_C
    pivot["sign_flip_magnitude"] = np.maximum(0.0, -pivot.AB_to_C)
    weights = {
        known: weight
        for known, weight in zip(
            LABEL_COUNTS, np.diff(np.log2((1, *LABEL_COUNTS))), strict=True
        )
    }
    pivot["auc_weight"] = pivot.known.map(weights)
    auc = (
        pivot.assign(weighted=pivot.assignment_drop * pivot.auc_weight)
        .groupby(["system_id", "metric"], as_index=False)
        .agg(weighted=("weighted", "sum"), weight=("auc_weight", "sum"))
    )
    auc["assignment_drop_auc"] = auc.weighted / auc.weight
    primary = pivot[pivot.known == 32].merge(
        auc[["system_id", "metric", "assignment_drop_auc"]],
        on=["system_id", "metric"],
        validate="one_to_one",
    )
    return primary


def neff_covariates(neff: pd.DataFrame) -> pd.DataFrame:
    wide = neff.pivot_table(
        index=["system_id", "trace"], columns="replica", values="neff"
    )
    wide.columns = [f"neff_R{value}" for value in wide.columns]
    wide = wide.reset_index()
    wide["neff_AB"] = wide.neff_R1 + wide.neff_R2
    wide["neff_min"] = np.minimum(np.sqrt(wide.neff_R1 * wide.neff_R2), wide.neff_R3)
    wide["neff_log_imbalance"] = np.abs(
        np.log(wide.neff_R3 / np.sqrt(wide.neff_R1 * wide.neff_R2))
    )
    return wide


def mechanism_correlations(
    outcomes: pd.DataFrame,
    overlap: pd.DataFrame,
    neff: pd.DataFrame,
    bootstrap: int,
    permutations: int,
) -> pd.DataFrame:
    overlap_wide = overlap[overlap.comparison == "AB_C"].pivot_table(
        index="system_id", columns="axis", values="js_bits"
    )
    neff_wide = neff_covariates(neff)
    rows = []
    trace_for = {"pf_l1": "global_pf", "work_opt": "work_opt"}
    for metric in PRIMARY:
        block = outcomes[outcomes.metric == metric].set_index("system_id")
        candidates = {
            f"overlap_{axis}": overlap_wide[axis] for axis in overlap_wide.columns
        }
        matched = neff_wide[neff_wide.trace == trace_for[metric]].set_index("system_id")
        candidates.update(
            {
                "neff_min": matched.neff_min,
                "neff_log_imbalance": matched.neff_log_imbalance,
            }
        )
        for response in (
            "assignment_drop",
            "sign_flip_magnitude",
            "assignment_drop_auc",
        ):
            y = block[response]
            for covariate, values in candidates.items():
                joined = pd.concat([y, values.rename("x")], axis=1).dropna()
                estimate, low, high = bootstrap_spearman(
                    joined.x.to_numpy(),
                    joined[response].to_numpy(),
                    stable_seed(metric, response, covariate, "boot"),
                    bootstrap,
                )
                p_value = permutation_p(
                    joined.x.to_numpy(),
                    joined[response].to_numpy(),
                    estimate,
                    stable_seed(metric, response, covariate, "perm"),
                    permutations,
                )
                rows.append(
                    {
                        "metric": metric,
                        "response": response,
                        "covariate": covariate,
                        "spearman_rho": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "permutation_p": p_value,
                        "loocv_rmse": loocv_rmse(
                            joined.x.to_numpy(), joined[response].to_numpy()
                        ),
                        "systems": len(joined),
                    }
                )
    table = pd.DataFrame(rows)
    table["fdr_q"] = np.nan
    for _, positions in table.groupby(["response"]).indices.items():
        table.loc[positions, "fdr_q"] = cp38.bh_adjust(
            table.loc[positions, "permutation_p"].to_numpy()
        )
    return table


def _metric_trace(metric: str) -> str:
    if metric == "rg":
        return "rg"
    if metric in {"rmsd", "w1"}:
        return "rmsd"
    if metric == "work_opt":
        return "work_opt"
    return "global_pf"


def _metric_overlap_axis(metric: str) -> str:
    if metric == "rg":
        return "rg"
    if metric in {"rmsd", "w1"}:
        return "rmsd"
    if metric.startswith("pf_"):
        return "pf_profile"
    if metric == "work_opt":
        return "work_opt_profile"
    return "global_pf"


def attach_sampling_covariates(
    checkpoint38: pd.DataFrame, neff: pd.DataFrame, overlap: pd.DataFrame
) -> pd.DataFrame:
    """Attach predictor-family-matched convergence covariates to checkpoint 38."""
    sampling = neff_covariates(neff)
    sampling["log_neff_fit"] = np.log(sampling.neff_R1)
    sampling = sampling.set_index(["system_id", "trace"])
    divergence = (
        overlap[overlap.comparison == "AB_C"].set_index(["system_id", "axis"]).js_bits
    )
    result = checkpoint38.copy()
    result["sampling_trace"] = result.metric.map(_metric_trace)
    result["overlap_axis"] = result.metric.map(_metric_overlap_axis)
    result["log_neff_fit"] = [
        sampling.loc[(row.system_id, row.sampling_trace), "log_neff_fit"]
        for row in result.itertuples()
    ]
    result["neff_log_imbalance"] = [
        sampling.loc[(row.system_id, row.sampling_trace), "neff_log_imbalance"]
        for row in result.itertuples()
    ]
    result["matched_overlap_js"] = [
        divergence.loc[(row.system_id, row.overlap_axis)] for row in result.itertuples()
    ]
    return result


def regression_matrix(
    frame: pd.DataFrame, response: str, covariates: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ordered = frame.reset_index(drop=True)
    y = cp38.within_predictor_standardize(ordered, response)
    columns = [np.ones(len(ordered))]
    names = ["intercept"]
    for covariate in covariates:
        values = ordered[covariate].to_numpy(float)
        scale = values.std(ddof=1)
        columns.append((values - values.mean()) / (scale if scale > 0 else 1.0))
        names.append(covariate)
    predictors = ordered.metric + ":" + ordered.model
    dummy = pd.get_dummies(predictors, drop_first=True, dtype=float)
    columns.extend(dummy[column].to_numpy() for column in dummy)
    names.extend(f"predictor[{column}]" for column in dummy)
    return np.column_stack(columns), y, names


def infer_regression_term(
    frame: pd.DataFrame,
    response: str,
    covariates: tuple[str, ...],
    term: str,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict:
    frame = frame.reset_index(drop=True)
    x, y, names = regression_matrix(frame, response, covariates)
    term_index = names.index(term)
    estimate = float(np.linalg.lstsq(x, y, rcond=None)[0][term_index])
    systems = frame.system_id.unique()
    groups = {
        system: np.flatnonzero(frame.system_id.to_numpy() == system)
        for system in systems
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(bootstrap)
    for draw in range(bootstrap):
        sampled = rng.choice(systems, len(systems), replace=True)
        rows = np.concatenate([groups[system] for system in sampled])
        boot[draw] = np.linalg.lstsq(x[rows], y[rows], rcond=None)[0][term_index]
    low, high = np.quantile(boot, [0.025, 0.975])
    keys = list(zip(frame.metric, frame.model, strict=True))
    lookup = {
        system: {keys[index]: x[index, term_index] for index in groups[system]}
        for system in systems
    }
    null = np.empty(permutations)
    for draw in range(permutations):
        shuffled = rng.permutation(systems)
        permuted = x.copy()
        for destination, source in zip(systems, shuffled, strict=True):
            rows = groups[destination]
            destination_keys = [keys[index] for index in rows]
            permuted[rows, term_index] = [
                lookup[source][key] for key in destination_keys
            ]
        null[draw] = np.linalg.lstsq(permuted, y, rcond=None)[0][term_index]
    return {
        "response": response,
        "covariates": "+".join(covariates),
        "term": term,
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "permutation_p": float(
            (1 + np.sum(np.abs(null) >= abs(estimate))) / (permutations + 1)
        ),
        "systems": len(systems),
        "rows": len(frame),
    }


def checkpoint38_sampling_models(
    analysis: pd.DataFrame, bootstrap: int, permutations: int
) -> pd.DataFrame:
    """Prespecified checkpoint-38 refits with sampling and overlap covariates."""
    primary = analysis[analysis.primary].reset_index(drop=True)
    specifications = (
        ("alpha_mismatch", ("heterogeneity_atypicality",), "heterogeneity_atypicality"),
        ("alpha_mismatch", ("log_neff_fit",), "log_neff_fit"),
        (
            "alpha_mismatch",
            ("heterogeneity_atypicality", "log_neff_fit"),
            "heterogeneity_atypicality",
        ),
        (
            "alpha_mismatch",
            ("heterogeneity_atypicality", "log_neff_fit"),
            "log_neff_fit",
        ),
        (
            "absolute_gap",
            ("matched_overlap_js", "neff_log_imbalance"),
            "matched_overlap_js",
        ),
        (
            "absolute_gap",
            ("matched_overlap_js", "neff_log_imbalance"),
            "neff_log_imbalance",
        ),
    )
    rows = []
    for response, covariates, term in specifications:
        rows.append(
            infer_regression_term(
                primary,
                response,
                covariates,
                term,
                bootstrap,
                permutations,
                stable_seed("cp38", response, *covariates, term),
            )
        )
    result = pd.DataFrame(rows)
    result["fdr_q"] = cp38.bh_adjust(result.permutation_p.to_numpy())
    return result


def paired_interval(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1
    )
    return tuple(map(float, np.quantile(draws, [0.025, 0.975])))


def summarize_transfer(frame: pd.DataFrame, kind: str, bootstrap: int) -> pd.DataFrame:
    block = frame[
        (frame["selection"] == "random")
        & (frame["stratum"] == "all")
        & (frame["method"] == "sparse_alpha")
        & frame["metric"].isin(PRIMARY)
        & (frame["model"] == "direct")
    ]
    rows = []
    if kind == "temporal":
        comparisons = (("C1_to_C2", "within_C1"), ("C2_to_C1", "within_C2"))
    else:
        comparisons = (
            ("ABC_common", "original_AB_to_C"),
            ("AB_common", "original_AB_to_C"),
            ("ABC_common", "AB_common"),
        )
    for metric in PRIMARY:
        for known in LABEL_COUNTS:
            current = block[(block.metric == metric) & (block.known == known)]
            for test, reference in comparisons:
                if not {test, reference}.issubset(set(current["mode"])):
                    continue
                pivot = (
                    current[current["mode"].isin((test, reference))]
                    .pivot_table(
                        index="system_id",
                        columns="mode",
                        values=["spearman", "distribution_recovery"],
                    )
                    .dropna()
                )
                delta = pivot[("spearman", test)] - pivot[("spearman", reference)]
                low, high = paired_interval(
                    delta, stable_seed(kind, metric, known, test), bootstrap
                )
                median_rho = float(pivot[("spearman", test)].median())
                rows.append(
                    {
                        "analysis": kind,
                        "metric": metric,
                        "known": known,
                        "test": test,
                        "reference": reference,
                        "median_spearman": median_rho,
                        "median_reference_spearman": float(
                            pivot[("spearman", reference)].median()
                        ),
                        "median_spearman_delta": float(np.median(delta)),
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                        "median_recovery": float(
                            pivot[("distribution_recovery", test)].median()
                        ),
                        "systems": len(pivot),
                        "rescue": bool(
                            kind == "pool"
                            and test == "ABC_common"
                            and reference == "original_AB_to_C"
                            and low > 0
                            and median_rho >= 0
                        ),
                        "strong_rescue": bool(
                            kind == "pool"
                            and test == "ABC_common"
                            and reference == "original_AB_to_C"
                            and low > 0
                            and median_rho >= 0.5
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_outputs(
    correlations: pd.DataFrame,
    temporal_summary: pd.DataFrame,
    pool_summary: pd.DataFrame,
    temporal: pd.DataFrame,
    pooled: pd.DataFrame,
    destination: Path,
) -> None:
    primary = correlations[
        (correlations.response == "assignment_drop")
        & correlations.covariate.isin(
            (
                "overlap_rmsd",
                "overlap_rg",
                "overlap_pf_profile",
                "overlap_work_opt_profile",
            )
        )
    ]
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    labels = primary.metric + ":" + primary.covariate.str.replace("overlap_", "")
    axis.bar(labels, primary.spearman_rho, color="#4C78A8")
    axis.errorbar(
        labels,
        primary.spearman_rho,
        yerr=[
            primary.spearman_rho - primary.ci_low,
            primary.ci_high - primary.spearman_rho,
        ],
        fmt="none",
        ecolor="black",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.tick_params(axis="x", rotation=45)
    axis.set_ylabel("Spearman with 32-label assignment drop")
    axis.set_title("Axis-decomposed A/B-to-C sampling mismatch")
    fig.savefig(destination / "axis_overlap_assignment_drop.png", dpi=180)
    plt.close(fig)

    effective = correlations[
        (correlations.response == "assignment_drop")
        & correlations.covariate.isin(("neff_min", "neff_log_imbalance"))
    ]
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = effective.metric + ":" + effective.covariate
    axis.bar(labels, effective.spearman_rho, color="#F58518")
    axis.errorbar(
        labels,
        effective.spearman_rho,
        yerr=[
            effective.spearman_rho - effective.ci_low,
            effective.ci_high - effective.spearman_rho,
        ],
        fmt="none",
        ecolor="black",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.tick_params(axis="x", rotation=30)
    axis.set_ylabel("Spearman with 32-label assignment drop")
    axis.set_title("Effective sampling and replica-transfer failure")
    fig.savefig(destination / "effective_support_assignment_drop.png", dpi=180)
    plt.close(fig)

    def curve_plot(
        frame: pd.DataFrame, modes: tuple[str, ...], name: str, title: str
    ) -> None:
        block = frame[
            (frame["selection"] == "random")
            & (frame["stratum"] == "all")
            & (frame["method"] == "sparse_alpha")
            & (frame["model"] == "direct")
            & frame["metric"].isin(PRIMARY)
            & frame["mode"].isin(modes)
        ]
        summary = block.groupby(
            ["mode", "metric", "known"], as_index=False
        ).spearman.median()
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for (mode, metric), line in summary.groupby(["mode", "metric"]):
            axis.plot(line.known, line.spearman, marker="o", label=f"{metric}:{mode}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
        axis.set_xticks(LABEL_COUNTS)
        axis.set_xlabel("Known populations")
        axis.set_ylabel("Median held-out Spearman rho")
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
        fig.savefig(destination / name, dpi=180)
        plt.close(fig)

    curve_plot(
        temporal,
        ("within_C1", "C1_to_C2", "within_C2", "C2_to_C1"),
        "temporal_transfer_learning_curves.png",
        "Temporal half-transfer versus within-half controls",
    )
    curve_plot(
        pooled,
        ("original_AB_to_C", "AB_common", "ABC_common"),
        "pooled_feature_intervention.png",
        "Pooled-feature intervention on A/B-to-C assignment",
    )

    combined = pd.concat([temporal_summary, pool_summary], ignore_index=True)
    combined.to_csv(destination / "transfer_summary.csv", index=False)


def report(
    destination: Path,
    systems: list[str],
    bootstrap: int,
    permutations: int,
) -> None:
    parts = destination / "parts"
    tables = {}
    for name in ("overlap", "neff", "temporal", "pool"):
        tables[name] = pd.concat(
            [pd.read_parquet(parts / f"{system}.{name}.parquet") for system in systems],
            ignore_index=True,
        )
        atomic_parquet(tables[name], destination / f"{name}_results.parquet")
    local39 = pd.read_parquet(CP39 / "local_extrapolation_results.parquet")
    outcomes = assignment_outcomes(local39)
    correlations = mechanism_correlations(
        outcomes, tables["overlap"], tables["neff"], bootstrap, permutations
    )
    correlations.to_csv(destination / "mechanism_correlations.csv", index=False)
    checkpoint38 = pd.read_parquet(CP38 / "system_predictor_gaps.parquet")
    checkpoint38 = checkpoint38[checkpoint38.system_id.isin(systems)]
    checkpoint38_sampling = attach_sampling_covariates(
        checkpoint38, tables["neff"], tables["overlap"]
    )
    atomic_parquet(
        checkpoint38_sampling, destination / "checkpoint38_sampling_covariates.parquet"
    )
    checkpoint38_models = (
        checkpoint38_sampling_models(checkpoint38_sampling, bootstrap, permutations)
        if len(systems) >= 3
        else pd.DataFrame()
    )
    checkpoint38_models.to_csv(
        destination / "checkpoint38_sampling_models.csv", index=False
    )
    temporal_summary = summarize_transfer(tables["temporal"], "temporal", bootstrap)
    pool_summary = summarize_transfer(tables["pool"], "pool", bootstrap)
    temporal_summary.to_csv(destination / "temporal_summary.csv", index=False)
    pool_summary.to_csv(destination / "pooled_intervention_summary.csv", index=False)
    plot_outputs(
        correlations,
        temporal_summary,
        pool_summary,
        tables["temporal"],
        tables["pool"],
        destination,
    )
    primary_temporal = temporal_summary[
        (temporal_summary.known == 32) & temporal_summary.test.str.contains("to")
    ]
    primary_pool = pool_summary[
        (pool_summary.known == 32)
        & (pool_summary.test == "ABC_common")
        & (pool_summary.reference == "original_AB_to_C")
    ]
    abc_increment = pool_summary[
        (pool_summary.known == 32)
        & (pool_summary.test == "ABC_common")
        & (pool_summary.reference == "AB_common")
    ]
    temporal_degradation = bool((primary_temporal.delta_ci_high < 0).all())
    pooled_rescue = bool(primary_pool.rescue.all())
    c_feature_increment = bool((abc_increment.delta_ci_low > 0).all())
    payload = {
        "checkpoint": 40,
        "systems": len(systems),
        "primary_predictors": list(PRIMARY),
        "primary_known_populations": 32,
        "temporal_32_label": primary_temporal.to_dict(orient="records"),
        "pooled_feature_32_label": primary_pool.to_dict(orient="records"),
        "abc_vs_ab_32_label": abc_increment.to_dict(orient="records"),
        "temporal_degradation_supported": temporal_degradation,
        "pooled_rescue_supported": pooled_rescue,
        "increment_from_unlabelled_C_features_supported": c_feature_increment,
        "headline": (
            "Within-replica temporal transfer degrades strongly. A common pooled "
            "feature coordinate restores nonnegative direct PF/Work assignment, "
            "but adding unlabeled C frames does not improve consistently over the "
            "A+B-only common coordinate."
        ),
        "checkpoint38_sampling_models": checkpoint38_models.to_dict(orient="records"),
        "interpretation_limits": [
            "ABC pooling is transductive: it uses C features but never C population labels",
            "direct-feature rescue tests descriptor estimation, not the harmonic variance equation",
            "failure to rescue is consistent with but does not prove a multimodal-theory breakdown",
        ],
        "provenance": {
            "checkpoint39_sha256": cp36.digest(
                CP39 / "local_extrapolation_results.parquet"
            ),
            "analysis_sha256": cp36.digest(Path(__file__)),
            "bootstrap_samples": bootstrap,
            "permutations": permutations,
            "seed": SEED,
        },
    }
    temporary = destination / "checkpoint40_report.yaml.tmp"
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False))
    temporary.replace(destination / "checkpoint40_report.yaml")


def valid_part(path: Path, required: set[str]) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    return bool(len(frame) and required.issubset(frame.columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("overlap", "temporal", "neff", "pool", "report", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--landmarks", type=int, default=64)
    parser.add_argument("--label-counts", default="2,4,8,16,32")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (
        min(
            args.workers,
            args.repeats,
            args.landmarks,
            args.bootstrap_samples,
            args.permutations,
        )
        < 1
    ):
        parser.error("all numeric settings must be positive")
    label_counts = tuple(sorted({int(value) for value in args.label_counts.split(",")}))
    rows = cp36.selected_rows("pilot", cp36.SINGLE_SYSTEM)
    if args.limit is not None:
        rows = rows[: args.limit]
    destination = args.output.resolve()
    parts = destination / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    requested = (
        {"overlap", "temporal", "neff", "pool"}
        if args.phase == "all"
        else {args.phase}
        if args.phase != "report"
        else set()
    )
    required = {
        "overlap": {"system_id", "comparison", "axis", "js_bits"},
        "neff": {"system_id", "replica", "trace", "neff"},
        "temporal": {"system_id", "mode", "metric", "known", "spearman"},
        "pool": {"system_id", "mode", "metric", "known", "spearman"},
    }
    fits_path = cp39.CP36 / "corrected_variance_fits.parquet"
    tasks = []
    for row in rows:
        missing = {
            component
            for component in requested
            if not valid_part(
                parts / f"{row['system_id']}.{component}.parquet", required[component]
            )
        }
        if missing:
            tasks.append(
                (
                    row,
                    missing,
                    args.landmarks,
                    args.repeats,
                    label_counts,
                    fits_path,
                    parts,
                )
            )
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(analyse_system, task): task[0]["system_id"]
                for task in tasks
            }
            for index, future in enumerate(as_completed(futures), 1):
                print(f"[{index}/{len(tasks)}] {future.result()} complete", flush=True)
    if args.phase in ("all", "report"):
        systems = [row["system_id"] for row in rows]
        report(destination, systems, args.bootstrap_samples, args.permutations)
        print(f"report: {destination / 'checkpoint40_report.yaml'}", flush=True)


if __name__ == "__main__":
    main()
