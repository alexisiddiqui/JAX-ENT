#!/usr/bin/env python3
"""Peptide-count and overlap study for PF marginal/conditional variance matching."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import investigate_pf_peptides as base
from jaxent.src.analysis.pf_variance import (
    conditional_variance_profile,
    inverse_overlap_degree_weights,
    marginal_variance_profile,
    map_frame_log_pf_to_peptides,
    map_residue_uptake_to_peptides,
    overlap_projection,
    peptide_overlap_similarity,
    projected_log_euclidean_covariance_loss,
    uptake_from_log_pf,
    weighted_population_covariance,
)
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.interfaces.topology import PTSerialiser


COUNTS = (15, 30, 60)
LAYOUT_SEEDS = tuple(range(20))
BASE_SEED = 20260720
METHODS = (
    "covariance_mse",
    "marginal",
    "marginal_overlap",
    "conditional",
    "conditional_overlap",
)
PROJECTED_METHOD = "projected_covariance"
PROJECTION_THRESHOLD = 1e-6
GAMMAS = (0.01, 0.1, 1.0, 10.0)
MAXENT_VALUES = (1.0, 10.0, 100.0, 1000.0)
ALPHAS = (0.01, 0.05, 0.10)


@dataclass(frozen=True)
class Layout:
    layout_id: str
    count: int
    family: str
    seed: int | None
    bounds: tuple[tuple[int, int], ...]


def _physical(active: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple((start - 1, end) for start, end in sorted(active))


def generate_layout(count: int, family: str, seed: int | None) -> Layout:
    if family == "equal":
        tiles = np.array_split(np.arange(2, 310), count)
        active = [(int(tile[0]), int(tile[-1])) for tile in tiles]
        layout_id = f"n{count:02d}_equal"
    else:
        rng = np.random.default_rng(BASE_SEED + 1000 * count + int(seed or 0))
        if family == "random_fixed":
            lengths = np.full(count, 20, dtype=int)
            starts = rng.choice(np.arange(2, 309 - 20 + 2), size=count, replace=False)
        elif family == "random_variable":
            lengths = rng.integers(10, 31, size=count)
            starts = np.asarray(
                [rng.integers(2, 309 - int(length) + 2) for length in lengths]
            )
        else:
            raise ValueError(f"Unknown layout family: {family}")
        active = [
            (int(start), int(start + length - 1))
            for start, length in zip(starts, lengths, strict=True)
        ]
        layout_id = f"n{count:02d}_{family}_seed{int(seed):02d}"
    return Layout(layout_id, count, family, seed, _physical(active))


def all_layouts() -> list[Layout]:
    layouts = []
    for count in COUNTS:
        layouts.append(generate_layout(count, "equal", None))
        for family in ("random_fixed", "random_variable"):
            layouts.extend(generate_layout(count, family, seed) for seed in LAYOUT_SEEDS)
    return layouts


def _paths(repo_root: Path) -> dict[str, Path]:
    example = repo_root / "jaxent/examples/1_IsoValidation_OMass"
    fitting = example / "fitting/jaxENT"
    return {
        "bi_features": fitting / "_featurise/features_iso_bi.npz",
        "tri_features": fitting / "_featurise/features_iso_tri.npz",
        "bi_clusters": example / "data/_clustering_results/cluster_assignments_ISO_BI.csv",
        "tri_clusters": example / "data/_clustering_results/cluster_assignments_ISO_TRI.csv",
        "tri_topology": fitting / "_featurise/topology_iso_tri.json",
    }


def _layout_data(layout: Layout, paths: dict[str, Path]):
    return base._build_panel(
        layout.layout_id, list(layout.bounds), paths["tri_features"], paths["tri_topology"]
    )


def _effective_rank(matrix: np.ndarray) -> float:
    eigenvalues = np.maximum(np.linalg.eigvalsh(matrix), 0.0)
    denominator = float(np.square(eigenvalues).sum())
    return float(eigenvalues.sum() ** 2 / denominator) if denominator else 0.0


def _geometry(layout: Layout, mapping: np.ndarray) -> dict[str, Any]:
    similarity = np.asarray(peptide_overlap_similarity(mapping), dtype=np.float64)
    degree = similarity.sum(axis=1)
    weights = np.asarray(inverse_overlap_degree_weights(mapping), dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(similarity)
    positive = eigenvalues[eigenvalues > 1e-8 * max(float(eigenvalues.max()), 1.0)]
    covered = (mapping > 0).sum(axis=0)
    return {
        "layout_id": layout.layout_id,
        "peptide_count": layout.count,
        "family": layout.family,
        "layout_seed": layout.seed,
        "coverage_fraction": float(np.mean(covered > 0)),
        "mean_coverage_multiplicity": float(covered.mean()),
        "maximum_coverage_multiplicity": int(covered.max()),
        "mean_redundancy_degree": float(np.mean(degree - 1.0)),
        "maximum_redundancy_degree": float(np.max(degree - 1.0)),
        "overlap_effective_rank": _effective_rank(similarity),
        "overlap_effective_rank_fraction": _effective_rank(similarity) / layout.count,
        "overlap_condition_number": float(positive.max() / positive.min()),
        "overlap_weight_min": float(weights.min()),
        "overlap_weight_max": float(weights.max()),
        "overlap_weight_ess": float(weights.sum() ** 2 / np.square(weights).sum()),
    }


def _select_layouts(catalog: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for count in COUNTS:
        group = catalog[catalog.peptide_count == count]
        equal = group[group.family == "equal"].iloc[0].copy()
        equal["selection_role"] = "equal_control"
        selected.append(equal)
        random = group[group.family != "equal"].sort_values("mean_redundancy_degree")
        chosen = []
        for role, quantile in (("low_redundancy", 0.10), ("median_redundancy", 0.50), ("high_redundancy", 0.90)):
            target = random.mean_redundancy_degree.quantile(quantile)
            available = random[~random.layout_id.isin([row.layout_id for row in chosen])]
            row = available.iloc[(available.mean_redundancy_degree - target).abs().argmin()].copy()
            row["selection_role"] = role
            chosen.append(row)
        if len({row.family for row in chosen}) == 1:
            missing = "random_variable" if chosen[0].family == "random_fixed" else "random_fixed"
            pool = random[random.family == missing]
            target = random.mean_redundancy_degree.median()
            replacement = pool.iloc[(pool.mean_redundancy_degree - target).abs().argmin()].copy()
            replacement["selection_role"] = "median_redundancy"
            chosen[1] = replacement
        selected.extend(chosen)
    return pd.DataFrame(selected)


def run_litmus(output_dir: Path, repo_root: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(repo_root)
    bi = base._load_ensemble(paths["bi_features"], paths["bi_clusters"])
    tri = base._load_ensemble(paths["tri_features"], paths["tri_clusters"])
    truth = base._truth_weights(bi.assignments)
    uniform = np.full(tri.log_pf.shape[1], 1.0 / tri.log_pf.shape[1])
    rows, diagnostics = [], []
    layout_lookup = {layout.layout_id: layout for layout in all_layouts()}
    for layout in layout_lookup.values():
        _, mapping, _ = _layout_data(layout, paths)
        geometry = _geometry(layout, mapping)
        rows.append(geometry)
        target_values = np.asarray(map_frame_log_pf_to_peptides(mapping, bi.log_pf))
        predicted_values = np.asarray(map_frame_log_pf_to_peptides(mapping, tri.log_pf))
        target_cov = np.asarray(weighted_population_covariance(target_values, truth))
        predicted_cov = np.asarray(weighted_population_covariance(predicted_values, uniform))
        projections = {
            threshold: np.asarray(overlap_projection(mapping, threshold))
            for threshold in (1e-4, 1e-6, 1e-8)
        }
        primary_projection = jnp.asarray(projections[1e-6])
        z_predicted = jnp.asarray(predicted_values)
        target_covariance_jax = jnp.asarray(target_cov)

        def full_covariance_loss(logits):
            covariance = weighted_population_covariance(
                z_predicted, jax.nn.softmax(logits)
            )
            return projected_log_euclidean_covariance_loss(
                covariance, target_covariance_jax, primary_projection, 0.05
            )

        def population_jsd(logits):
            populations = base._cluster_probabilities(
                jax.nn.softmax(logits), tri.assignments
            )
            return base.jensen_shannon_divergence(populations, base.TARGET_POPULATIONS)

        uniform_logits = jnp.zeros(tri.log_pf.shape[1])
        full_gradient = jax.grad(full_covariance_loss)(uniform_logits)
        jsd_gradient = jax.grad(population_jsd)(uniform_logits)
        full_cosine = float(
            jnp.vdot(full_gradient, jsd_gradient)
            / (jnp.linalg.norm(full_gradient) * jnp.linalg.norm(jsd_gradient) + 1e-30)
        )
        stepped = uniform_logits - 0.01 * full_gradient / (
            jnp.linalg.norm(full_gradient) + 1e-30
        )
        full_step_change = float(population_jsd(stepped) - population_jsd(uniform_logits))
        for alpha in ALPHAS:
            target_marginal = np.asarray(marginal_variance_profile(target_values, truth, alpha))
            target_conditional = np.asarray(conditional_variance_profile(target_values, truth, alpha))
            diagnostics.append(
                {
                    "layout_id": layout.layout_id,
                    "alpha": alpha,
                    "target_marginal_cv": float(target_marginal.std() / target_marginal.mean()),
                    "target_conditional_cv": float(target_conditional.std() / target_conditional.mean()),
                    **{
                        f"full_covariance_loss_threshold_{threshold:g}": float(
                            projected_log_euclidean_covariance_loss(
                                predicted_cov, target_cov, projection, alpha
                            )
                        )
                        for threshold, projection in projections.items()
                    },
                    "projection_rank_1e-6": int(projections[1e-6].shape[1]),
                    "full_covariance_gradient_cosine": full_cosine,
                    "full_covariance_gradient_jsd_step_change": full_step_change,
                }
            )
    catalog = pd.DataFrame(rows)
    selected = _select_layouts(catalog)
    catalog.to_csv(output_dir / "layout_catalog.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "litmus_diagnostics.csv", index=False)
    selected.to_csv(output_dir / "selected_layouts.csv", index=False)
    records = []
    for row in selected.itertuples(index=False):
        layout = layout_lookup[row.layout_id]
        records.append(
            {
                "layout_id": layout.layout_id,
                "peptide_count": layout.count,
                "family": layout.family,
                "layout_seed": layout.seed,
                "selection_role": row.selection_role,
                "bounds": [list(bound) for bound in layout.bounds],
            }
        )
    (output_dir / "selected_layouts.json").write_text(json.dumps(records, indent=2))
    decision = {
        "status": "evaluated",
        "layout_count": len(catalog),
        "selected_count": len(selected),
        "row_normalized_maps": True,
        "full_covariance_optimization_enabled": False,
        "primary_projection_threshold": 1e-6,
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2))


def _load_selected(litmus_dir: Path) -> list[Layout]:
    records = json.loads((litmus_dir / "selected_layouts.json").read_text())
    return [
        Layout(
            record["layout_id"],
            int(record["peptide_count"]),
            record["family"],
            record["layout_seed"],
            tuple(tuple(bound) for bound in record["bounds"]),
        )
        for record in records
    ]


def _fit_layout(
    layout: Layout,
    paths: dict[str, Path],
    bi: base.Ensemble,
    tri: base.Ensemble,
    settings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    truth = base._truth_weights(bi.assignments)
    mean_log_pf, residue_covariance = base._latent_distribution(bi.log_pf, truth)
    data, mapping, _ = _layout_data(layout, paths)
    feature_topology = PTSerialiser.load_list_from_json(paths["tri_topology"])
    target_uptake = np.asarray(
        map_residue_uptake_to_peptides(
            mapping, uptake_from_log_pf(mean_log_pf, bi.k_ints, base.TIMEPOINTS)
        )
    )
    target_covariance = mapping @ residue_covariance @ mapping.T
    predicted_peptide = np.asarray(map_frame_log_pf_to_peptides(mapping, tri.log_pf))
    rows, weights = [], {}
    split_metadata = {}
    for split_index in range(3):
        train, val, split = base._split_panel(data, feature_topology, split_index)
        split_metadata[str(split_index)] = split
        for setting in settings:
            method = setting["method"]
            base_method = "conditional" if method.startswith("conditional") else "marginal"
            if method == "covariance_mse":
                base_method = "covariance_mse"
            elif method == PROJECTED_METHOD:
                base_method = PROJECTED_METHOD
            weighted = method.endswith("_overlap")
            for start in range(3):
                run_id = (
                    f"{layout.layout_id}_split{split_index}_{method}_g{setting['gamma']:g}"
                    f"_m{setting['maxent_value']:g}_a{setting['alpha']:g}_s{start}"
                )
                result = base.optimize(
                    predicted_residue_log_pf=tri.log_pf,
                    predicted_peptide_log_pf=predicted_peptide,
                    predicted_k_ints=tri.k_ints,
                    peptide_mapping=mapping,
                    target_peptide_covariance=target_covariance,
                    target_uptake=target_uptake,
                    clean_target_uptake=target_uptake,
                    assignments=tri.assignments,
                    train_indices=train,
                    val_indices=val,
                    method=base_method,
                    gamma=setting["gamma"],
                    maxent_value=setting["maxent_value"],
                    alpha=setting["alpha"],
                    steps=2000,
                    learning_rate=0.05,
                    start_seed=start,
                    overlap_weighted=weighted,
                    projection_threshold=PROJECTION_THRESHOLD,
                )
                weights[run_id] = result.pop("weights")
                rows.append(
                    {
                        "run_id": run_id,
                        "layout_id": layout.layout_id,
                        "peptide_count": layout.count,
                        "family": layout.family,
                        "split_index": split_index,
                        "method": method,
                        "start": start,
                        **setting,
                        **result,
                    }
                )
    return rows, weights, split_metadata


def _projected_calibration_settings() -> list[dict[str, Any]]:
    return [
        {
            "method": PROJECTED_METHOD,
            "gamma": gamma,
            "maxent_value": maxent,
            "alpha": alpha,
        }
        for gamma in GAMMAS
        for maxent in MAXENT_VALUES
        for alpha in ALPHAS
    ]


def _median_30_layout(litmus_dir: Path) -> Layout:
    metadata = pd.read_csv(litmus_dir / "selected_layouts.csv")
    chosen = metadata[
        (metadata.peptide_count == 30)
        & (metadata.selection_role == "median_redundancy")
    ]
    if len(chosen) != 1:
        raise ValueError("Expected exactly one 30-peptide median-redundancy layout")
    layout_id = str(chosen.iloc[0].layout_id)
    return next(layout for layout in _load_selected(litmus_dir) if layout.layout_id == layout_id)


def _select_projected_calibration(
    raw: pd.DataFrame, comparator_dir: Path, calibration_layout: Layout
) -> tuple[pd.DataFrame, dict[str, Any]]:
    setting_cols = ["gamma", "maxent_value", "alpha", "split_index"]
    best_starts = raw.loc[raw.groupby(setting_cols).train_total.idxmin()].copy()
    comparators = pd.read_csv(comparator_dir / "selected_results.csv")
    baseline = comparators[
        (comparators.layout_id == calibration_layout.layout_id)
        & (comparators.method == "covariance_mse")
    ][["split_index", "val_curve_mse"]].rename(
        columns={"val_curve_mse": "baseline_val_curve_mse"}
    )
    if len(baseline) != 3:
        raise ValueError("Missing covariance-MSE calibration comparators")
    best_starts = best_starts.merge(baseline, on="split_index", validate="many_to_one")
    best_starts["val_curve_mse_ratio"] = (
        best_starts.val_curve_mse / best_starts.baseline_val_curve_mse
    )
    aggregate = (
        best_starts.groupby(["gamma", "maxent_value", "alpha"])
        .agg(
            median_val_curve_mse_ratio=("val_curve_mse_ratio", "median"),
            median_val_projected_covariance_loss=("val_pf_profile_loss", "median"),
            median_val_curve_mse=("val_curve_mse", "median"),
            median_train_total=("train_total", "median"),
        )
        .reset_index()
    )
    feasible = aggregate[aggregate.median_val_curve_mse_ratio <= 1.05].sort_values(
        [
            "median_val_projected_covariance_loss",
            "median_val_curve_mse",
            "median_train_total",
            "gamma",
            "maxent_value",
            "alpha",
        ]
    )
    decision = {
        "status": "passed" if len(feasible) else "failed",
        "calibration_layout_id": calibration_layout.layout_id,
        "curve_mse_ratio_limit": 1.05,
        "candidate_settings": int(len(aggregate)),
        "feasible_settings": int(len(feasible)),
        "selection_uses_recovery": False,
        "projection_threshold": PROJECTION_THRESHOLD,
    }
    if not len(feasible):
        return pd.DataFrame(), decision
    frozen = feasible.iloc[[0]].copy()
    frozen.insert(0, "method", PROJECTED_METHOD)
    frozen.insert(0, "calibration_layout_id", calibration_layout.layout_id)
    return frozen, decision


def run_projected_phase(
    phase: str,
    litmus_dir: Path,
    output_dir: Path,
    repo_root: Path,
    comparator_dir: Path,
    calibration_dir: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(repo_root)
    bi = base._load_ensemble(paths["bi_features"], paths["bi_clusters"])
    tri = base._load_ensemble(paths["tri_features"], paths["tri_clusters"])
    if phase == "projected-calibrate":
        layouts = [_median_30_layout(litmus_dir)]
        settings = _projected_calibration_settings()
    else:
        if calibration_dir is None:
            raise ValueError("--calibration-dir is required for projected-final")
        decision = json.loads((calibration_dir / "decision.json").read_text())
        if decision["status"] != "passed":
            raise RuntimeError("Projected covariance calibration did not pass")
        frozen = pd.read_csv(calibration_dir / "frozen_settings.csv").iloc[0]
        settings = [
            {
                "method": PROJECTED_METHOD,
                "gamma": float(frozen.gamma),
                "maxent_value": float(frozen.maxent_value),
                "alpha": float(frozen.alpha),
            }
        ]
        layouts = _load_selected(litmus_dir)
    rows: list[dict[str, Any]] = []
    weights: dict[str, np.ndarray] = {}
    splits: dict[str, Any] = {}
    for layout in layouts:
        layout_rows, layout_weights, metadata = _fit_layout(
            layout, paths, bi, tri, settings
        )
        rows.extend(layout_rows)
        weights.update(layout_weights)
        splits[layout.layout_id] = metadata
        pd.DataFrame(rows).to_csv(output_dir / "raw_results.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "raw_results.csv", index=False)
    np.savez_compressed(output_dir / "weights.npz", **weights)
    (output_dir / "split_metadata.json").write_text(json.dumps(splits, indent=2))
    if phase == "projected-calibrate":
        frozen, decision = _select_projected_calibration(
            raw, comparator_dir, layouts[0]
        )
        if len(frozen):
            frozen.to_csv(output_dir / "frozen_settings.csv", index=False)
        (output_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    else:
        selected = raw.loc[
            raw.groupby(["layout_id", "split_index"]).train_total.idxmin()
        ].copy()
        selected.to_csv(output_dir / "selected_results.csv", index=False)


def _calibration_settings() -> list[dict[str, Any]]:
    settings = []
    for maxent in MAXENT_VALUES:
        settings.append(
            {"method": "covariance_mse", "gamma": 0.0, "maxent_value": maxent, "alpha": 0.05}
        )
    for method in METHODS[1:]:
        for gamma in GAMMAS:
            for maxent in MAXENT_VALUES:
                for alpha in ALPHAS:
                    settings.append(
                        {"method": method, "gamma": gamma, "maxent_value": maxent, "alpha": alpha}
                    )
    return settings


def _select_frozen(raw: pd.DataFrame) -> pd.DataFrame:
    setting_cols = ["peptide_count", "method", "gamma", "maxent_value", "alpha", "split_index"]
    best = raw.loc[raw.groupby(setting_cols).train_total.idxmin()].copy()
    aggregate = (
        best.groupby(["peptide_count", "method", "gamma", "maxent_value", "alpha"])
        .agg(median_val_curve_mse=("val_curve_mse", "median"), median_val_profile_loss=("val_pf_profile_loss", "median"), median_train_total=("train_total", "median"))
        .reset_index()
        .sort_values(["peptide_count", "method", "median_val_curve_mse", "median_val_profile_loss", "median_train_total", "gamma", "maxent_value", "alpha"])
    )
    return aggregate.groupby(["peptide_count", "method"], as_index=False).first()


def run_fit_phase(phase: str, litmus_dir: Path, output_dir: Path, repo_root: Path, calibration_dir: Path | None, counts: tuple[int, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(repo_root)
    bi = base._load_ensemble(paths["bi_features"], paths["bi_clusters"])
    tri = base._load_ensemble(paths["tri_features"], paths["tri_clusters"])
    layouts = _load_selected(litmus_dir)
    layouts = [layout for layout in layouts if layout.count in counts]
    if phase == "calibrate":
        layouts = [layout for layout in layouts if "median" in next(record["selection_role"] for record in json.loads((litmus_dir / "selected_layouts.json").read_text()) if record["layout_id"] == layout.layout_id)]
        settings_by_count = {count: _calibration_settings() for count in counts}
    else:
        frozen = pd.read_csv(Path(calibration_dir) / "frozen_settings.csv")
        settings_by_count = {
            count: frozen[frozen.peptide_count == count][["method", "gamma", "maxent_value", "alpha"]].to_dict("records")
            for count in counts
        }
    rows, weights, splits = [], {}, {}
    for layout in layouts:
        layout_rows, layout_weights, metadata = _fit_layout(
            layout, paths, bi, tri, settings_by_count[layout.count]
        )
        rows.extend(layout_rows)
        weights.update(layout_weights)
        splits[layout.layout_id] = metadata
        pd.DataFrame(rows).to_csv(output_dir / "raw_results.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "raw_results.csv", index=False)
    np.savez_compressed(output_dir / "weights.npz", **weights)
    (output_dir / "split_metadata.json").write_text(json.dumps(splits, indent=2))
    if phase == "calibrate":
        frozen = _select_frozen(raw)
        frozen.to_csv(output_dir / "frozen_settings.csv", index=False)
    else:
        keys = ["layout_id", "split_index", "method"]
        selected = raw.loc[raw.groupby(keys).train_total.idxmin()].copy()
        selected.to_csv(output_dir / "selected_results.csv", index=False)
        comparisons = []
        for (layout_id, split), group in selected.groupby(["layout_id", "split_index"]):
            baseline = group[group.method == "covariance_mse"].iloc[0]
            for _, candidate in group[group.method != "covariance_mse"].iterrows():
                comparisons.append(
                    {
                        "layout_id": layout_id,
                        "peptide_count": int(candidate.peptide_count),
                        "family": candidate.family,
                        "split_index": int(split),
                        "method": candidate.method,
                        "baseline_recovery_pct": baseline.recovery_pct,
                        "candidate_recovery_pct": candidate.recovery_pct,
                        "recovery_gain_pp": candidate.recovery_pct - baseline.recovery_pct,
                        "val_curve_mse_ratio": candidate.val_curve_mse / baseline.val_curve_mse,
                        "open_ess": candidate.open_ess,
                    }
                )
        pd.DataFrame(comparisons).to_csv(output_dir / "comparisons.csv", index=False)


def analyze_final(litmus_dir: Path, results_dir: Path, repo_root: Path) -> None:
    paths = _paths(repo_root)
    bi = base._load_ensemble(paths["bi_features"], paths["bi_clusters"])
    tri = base._load_ensemble(paths["tri_features"], paths["tri_clusters"])
    truth = base._truth_weights(bi.assignments)
    _, residue_covariance = base._latent_distribution(bi.log_pf, truth)
    selected = pd.read_csv(results_dir / "selected_results.csv")
    comparisons = pd.read_csv(results_dir / "comparisons.csv")
    comparisons = comparisons.drop(
        columns=[
            column
            for column in (
                "gradient_cosine",
                "gradient_jsd_step_change",
                "selection_role",
                "mean_redundancy_degree",
                "selection_role_x",
                "selection_role_y",
                "mean_redundancy_degree_x",
                "mean_redundancy_degree_y",
            )
            if column in comparisons.columns
        ]
    )
    layout_metadata = pd.read_csv(litmus_dir / "selected_layouts.csv")
    split_metadata = json.loads((results_dir / "split_metadata.json").read_text())
    layouts = {layout.layout_id: layout for layout in _load_selected(litmus_dir)}
    with np.load(results_dir / "weights.npz") as archive:
        saved_weights = {key: np.asarray(archive[key]) for key in archive.files}
    diagnostics = []
    for (layout_id, split_index), group in selected.groupby(["layout_id", "split_index"]):
        layout = layouts[layout_id]
        _, mapping, _ = _layout_data(layout, paths)
        target_covariance = mapping @ residue_covariance @ mapping.T
        predicted_peptide = np.asarray(map_frame_log_pf_to_peptides(mapping, tri.log_pf))
        train = np.asarray(split_metadata[layout_id][str(int(split_index))]["train_indices"])
        baseline = group[group.method == "covariance_mse"].iloc[0]
        baseline_weights = saved_weights[baseline.run_id]
        for _, row in group[group.method != "covariance_mse"].iterrows():
            method = "conditional" if row.method.startswith("conditional") else "marginal"
            cosine, step_change = base._gradient_diagnostics(
                baseline_weights,
                method,
                float(row.alpha),
                predicted_peptide,
                target_covariance,
                tri.assignments,
                train,
                peptide_mapping=mapping,
                overlap_weighted=row.method.endswith("_overlap"),
            )
            diagnostics.append(
                {
                    "layout_id": layout_id,
                    "split_index": int(split_index),
                    "method": row.method,
                    "gradient_cosine": cosine,
                    "gradient_jsd_step_change": step_change,
                }
            )
    diagnostics = pd.DataFrame(diagnostics)
    diagnostics.to_csv(results_dir / "gradient_diagnostics.csv", index=False)
    comparisons = comparisons.merge(
        diagnostics, on=["layout_id", "split_index", "method"], validate="one_to_one"
    ).merge(
        layout_metadata[
            ["layout_id", "selection_role", "mean_redundancy_degree"]
        ],
        on="layout_id",
        validate="many_to_one",
    )
    comparisons.to_csv(results_dir / "comparisons.csv", index=False)
    pivot = comparisons.pivot(
        index=["layout_id", "peptide_count", "family", "selection_role", "mean_redundancy_degree", "split_index"],
        columns="method",
        values="candidate_recovery_pct",
    ).reset_index()
    pivot["conditional_minus_marginal"] = pivot.conditional - pivot.marginal
    pivot["conditional_overlap_minus_marginal_overlap"] = (
        pivot.conditional_overlap - pivot.marginal_overlap
    )
    pivot.to_csv(results_dir / "conditional_marginal_comparisons.csv", index=False)
    random_layouts = pivot[pivot.family != "equal"]
    layout_medians = random_layouts.groupby(
        ["layout_id", "peptide_count", "mean_redundancy_degree"]
    )[["conditional_minus_marginal", "conditional_overlap_minus_marginal_overlap"]].median().reset_index()
    count_medians = layout_medians.groupby("peptide_count").median(numeric_only=True)
    count_rho = float(
        layout_medians.peptide_count.corr(
            layout_medians.conditional_overlap_minus_marginal_overlap, method="spearman"
        )
    )
    redundancy_rho = float(
        layout_medians.mean_redundancy_degree.corr(
            layout_medians.conditional_overlap_minus_marginal_overlap, method="spearman"
        )
    )
    method_summary = comparisons.groupby(["peptide_count", "method"]).agg(
        median_recovery=("candidate_recovery_pct", "median"),
        median_gain=("recovery_gain_pp", "median"),
        median_mean_ratio=("val_curve_mse_ratio", "median"),
        gradient_aligned=("gradient_jsd_step_change", lambda values: int((values < 0).sum())),
        comparisons=("recovery_gain_pp", "size"),
    ).reset_index()
    method_summary.to_csv(results_dir / "method_summary.csv", index=False)
    weighting = {}
    random_comparisons = comparisons[comparisons.family != "equal"]
    for method in ("marginal", "conditional"):
        unweighted = random_comparisons[random_comparisons.method == method].groupby(
            "layout_id"
        ).recovery_gain_pp.median()
        weighted = random_comparisons[
            random_comparisons.method == f"{method}_overlap"
        ].groupby("layout_id").recovery_gain_pp.median()
        unweighted_iqr = float(unweighted.quantile(0.75) - unweighted.quantile(0.25))
        weighted_iqr = float(weighted.quantile(0.75) - weighted.quantile(0.25))
        median_change = float((weighted - unweighted).median())
        weighting[method] = {
            "median_recovery_change_pp": median_change,
            "unweighted_gain_iqr": unweighted_iqr,
            "weighted_gain_iqr": weighted_iqr,
            "supported": bool(median_change >= -0.25 and weighted_iqr < unweighted_iqr),
        }
    litmus = pd.read_csv(litmus_dir / "litmus_diagnostics.csv")
    full_rows = litmus[np.isclose(litmus.alpha, 0.05)]
    full_gradient_valid = bool(
        np.isfinite(full_rows.full_covariance_gradient_cosine).all()
        and np.isfinite(full_rows.full_covariance_gradient_jsd_step_change).all()
    )
    threshold_rank_correlation = float(
        full_rows["full_covariance_loss_threshold_0.0001"].corr(
            full_rows["full_covariance_loss_threshold_1e-08"], method="spearman"
        )
    )
    decision = {
        "status": "evaluated",
        "fit_count": int(len(pd.read_csv(results_dir / "raw_results.csv"))),
        "all_finite": bool(pd.read_csv(results_dir / "raw_results.csv").finite.all()),
        "conditional_scaling": {
            "weighted_delta_at_15": float(count_medians.loc[15, "conditional_overlap_minus_marginal_overlap"]),
            "weighted_delta_at_60": float(count_medians.loc[60, "conditional_overlap_minus_marginal_overlap"]),
            "positive_at_60": bool(count_medians.loc[60, "conditional_overlap_minus_marginal_overlap"] > 0),
            "improves_15_to_60": bool(count_medians.loc[60, "conditional_overlap_minus_marginal_overlap"] > count_medians.loc[15, "conditional_overlap_minus_marginal_overlap"]),
            "count_spearman": count_rho,
            "redundancy_spearman": redundancy_rho,
        },
        "projected_full_covariance": {
            "finite_gradients": full_gradient_valid,
            "gradient_aligned": int((full_rows.full_covariance_gradient_jsd_step_change < 0).sum()),
            "comparisons": int(len(full_rows)),
            "threshold_rank_correlation": threshold_rank_correlation,
            "advance_to_optimization": bool(
                full_gradient_valid
                and (full_rows.full_covariance_gradient_jsd_step_change < 0).mean() >= 0.8
                and threshold_rank_correlation >= 0.95
            ),
        },
        "overlap_weighting": weighting,
        "profile_methods": {
            "positive_median_gain_all_counts": bool(
                (method_summary[method_summary.method != "covariance_mse"].median_gain > 0).all()
            ),
            "median_mean_ratio_within_1.05": bool(
                (method_summary[method_summary.method != "covariance_mse"].median_mean_ratio <= 1.05).all()
            ),
        },
        "ess_used_as_gate": False,
    }
    (results_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    lines = [
        "# Peptide overlap study",
        "",
        f"- Full fits: {decision['fit_count']} (all finite: {decision['all_finite']}).",
        f"- Weighted conditional-minus-marginal recovery at 15 peptides: {decision['conditional_scaling']['weighted_delta_at_15']:.3f} pp.",
        f"- Weighted conditional-minus-marginal recovery at 60 peptides: {decision['conditional_scaling']['weighted_delta_at_60']:.3f} pp.",
        f"- Count trend Spearman: {count_rho:.3f}; redundancy trend Spearman: {redundancy_rho:.3f}.",
        f"- Projected full-covariance gradients aligned in {decision['projected_full_covariance']['gradient_aligned']}/{decision['projected_full_covariance']['comparisons']} layouts.",
        f"- Overlap weighting supported for marginal/conditional: {weighting['marginal']['supported']}/{weighting['conditional']['supported']}.",
    ]
    (results_dir / "report.md").write_text("\n".join(lines) + "\n")


def analyze_projected(
    litmus_dir: Path, results_dir: Path, comparator_dir: Path, repo_root: Path
) -> None:
    paths = _paths(repo_root)
    bi = base._load_ensemble(paths["bi_features"], paths["bi_clusters"])
    tri = base._load_ensemble(paths["tri_features"], paths["tri_clusters"])
    truth = base._truth_weights(bi.assignments)
    _, residue_covariance = base._latent_distribution(bi.log_pf, truth)
    projected = pd.read_csv(results_dir / "selected_results.csv")
    raw = pd.read_csv(results_dir / "raw_results.csv")
    comparators = pd.read_csv(comparator_dir / "selected_results.csv")
    comparators = comparators[
        comparators.method.isin(("covariance_mse", "marginal", "conditional"))
    ].copy()
    layout_metadata = pd.read_csv(litmus_dir / "selected_layouts.csv")
    layouts = {layout.layout_id: layout for layout in _load_selected(litmus_dir)}
    split_metadata = json.loads((comparator_dir / "split_metadata.json").read_text())
    with np.load(results_dir / "weights.npz") as archive:
        projected_weights = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(comparator_dir / "weights.npz") as archive:
        comparator_weights = {key: np.asarray(archive[key]) for key in archive.files}

    covariance_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for projected_row in projected.itertuples(index=False):
        layout = layouts[projected_row.layout_id]
        _, mapping, _ = _layout_data(layout, paths)
        target_covariance = np.asarray(mapping @ residue_covariance @ mapping.T)
        predicted_peptide = np.asarray(map_frame_log_pf_to_peptides(mapping, tri.log_pf))
        split = split_metadata[layout.layout_id][str(int(projected_row.split_index))]
        train = np.asarray(split["train_indices"], dtype=int)
        validation = np.asarray(split["val_indices"], dtype=int)
        validation_projection = overlap_projection(
            mapping[validation], PROJECTION_THRESHOLD
        )
        target_validation = target_covariance[np.ix_(validation, validation)]
        alpha = float(projected_row.alpha)
        projected_weight = projected_weights[projected_row.run_id]
        projected_validation_covariance = weighted_population_covariance(
            predicted_peptide[validation], projected_weight
        )
        projected_loss = float(
            projected_log_euclidean_covariance_loss(
                projected_validation_covariance,
                target_validation,
                validation_projection,
                alpha,
            )
        )
        group = comparators[
            (comparators.layout_id == layout.layout_id)
            & (comparators.split_index == projected_row.split_index)
        ]
        if set(group.method) != {"covariance_mse", "marginal", "conditional"}:
            raise ValueError(f"Missing comparator for {layout.layout_id} split {projected_row.split_index}")
        comparator_losses: dict[str, float] = {}
        for comparator in group.itertuples(index=False):
            weights = comparator_weights[comparator.run_id]
            covariance = weighted_population_covariance(
                predicted_peptide[validation], weights
            )
            loss = float(
                projected_log_euclidean_covariance_loss(
                    covariance,
                    target_validation,
                    validation_projection,
                    alpha,
                )
            )
            comparator_losses[comparator.method] = loss
            covariance_rows.append(
                {
                    "layout_id": layout.layout_id,
                    "peptide_count": layout.count,
                    "split_index": int(projected_row.split_index),
                    "method": comparator.method,
                    "val_projected_covariance_loss": loss,
                }
            )
            comparison_rows.append(
                {
                    "layout_id": layout.layout_id,
                    "peptide_count": layout.count,
                    "family": layout.family,
                    "split_index": int(projected_row.split_index),
                    "comparator": comparator.method,
                    "comparator_recovery_pct": float(comparator.recovery_pct),
                    "projected_recovery_pct": float(projected_row.recovery_pct),
                    "recovery_difference_pp": float(
                        projected_row.recovery_pct - comparator.recovery_pct
                    ),
                    "comparator_val_curve_mse": float(comparator.val_curve_mse),
                    "projected_val_curve_mse": float(projected_row.val_curve_mse),
                    "val_curve_mse_ratio": float(
                        projected_row.val_curve_mse / comparator.val_curve_mse
                    ),
                    "comparator_val_projected_covariance_loss": loss,
                    "projected_val_projected_covariance_loss": projected_loss,
                    "projected_covariance_loss_ratio": projected_loss / loss,
                }
            )
        covariance_rows.append(
            {
                "layout_id": layout.layout_id,
                "peptide_count": layout.count,
                "split_index": int(projected_row.split_index),
                "method": PROJECTED_METHOD,
                "val_projected_covariance_loss": projected_loss,
            }
        )

        baseline = group[group.method == "covariance_mse"].iloc[0]
        baseline_weights = comparator_weights[baseline.run_id]
        train_projection = overlap_projection(mapping[train], PROJECTION_THRESHOLD)
        target_train = jnp.asarray(target_covariance[np.ix_(train, train)])
        predicted_train = jnp.asarray(predicted_peptide[train])

        def covariance_loss(logits):
            covariance = weighted_population_covariance(
                predicted_train, base.weights_from_logits(logits)
            )
            return projected_log_euclidean_covariance_loss(
                covariance, target_train, train_projection, alpha
            )

        def population_jsd(logits):
            populations = base._cluster_probabilities(
                base.weights_from_logits(logits), tri.assignments
            )
            return base.jensen_shannon_divergence(
                populations, base.TARGET_POPULATIONS
            )

        logits = jnp.log(jnp.asarray(baseline_weights))
        covariance_gradient = jax.grad(covariance_loss)(logits)
        jsd_gradient = jax.grad(population_jsd)(logits)
        covariance_norm = jnp.linalg.norm(covariance_gradient)
        cosine = jnp.vdot(covariance_gradient, jsd_gradient) / (
            covariance_norm * jnp.linalg.norm(jsd_gradient) + 1e-30
        )
        stepped = logits - 0.01 * covariance_gradient / (covariance_norm + 1e-30)
        gradient_rows.append(
            {
                "layout_id": layout.layout_id,
                "peptide_count": layout.count,
                "split_index": int(projected_row.split_index),
                "gradient_cosine": float(cosine),
                "gradient_jsd_step_change": float(
                    population_jsd(stepped) - population_jsd(logits)
                ),
            }
        )

    comparisons = pd.DataFrame(comparison_rows).merge(
        layout_metadata[
            ["layout_id", "selection_role", "mean_redundancy_degree"]
        ],
        on="layout_id",
        validate="many_to_one",
    )
    covariance_losses = pd.DataFrame(covariance_rows)
    gradients = pd.DataFrame(gradient_rows)
    comparisons.to_csv(results_dir / "comparisons.csv", index=False)
    covariance_losses.to_csv(results_dir / "projected_covariance_losses.csv", index=False)
    gradients.to_csv(results_dir / "gradient_diagnostics.csv", index=False)

    baseline_comparisons = comparisons[comparisons.comparator == "covariance_mse"]
    marginal_comparisons = comparisons[comparisons.comparator == "marginal"]
    baseline_by_count = baseline_comparisons.groupby("peptide_count").agg(
        median_recovery_gain_pp=("recovery_difference_pp", "median"),
        positive_recovery_comparisons=("recovery_difference_pp", lambda values: int((values > 0).sum())),
        median_val_curve_mse_ratio=("val_curve_mse_ratio", "median"),
    )
    marginal_by_count = marginal_comparisons.groupby("peptide_count").agg(
        median_recovery_difference_pp=("recovery_difference_pp", "median"),
        median_covariance_loss_ratio=("projected_covariance_loss_ratio", "median"),
    )
    baseline_by_count.reset_index().to_csv(
        results_dir / "baseline_summary_by_count.csv", index=False
    )
    marginal_by_count.reset_index().to_csv(
        results_dir / "marginal_summary_by_count.csv", index=False
    )
    finite = bool(
        raw.finite.all()
        and np.isfinite(comparisons.select_dtypes(include=[np.number])).all().all()
        and np.isfinite(gradients.select_dtypes(include=[np.number])).all().all()
    )
    overall_marginal_difference = float(
        marginal_comparisons.recovery_difference_pp.median()
    )
    gradient_alignment = int((gradients.gradient_jsd_step_change < 0).sum())
    gates = {
        "all_finite": finite,
        "positive_median_gain_over_baseline_all_counts": bool(
            (baseline_by_count.median_recovery_gain_pp > 0).all()
        ),
        "at_least_8_of_12_baseline_improvements_each_count": bool(
            (baseline_by_count.positive_recovery_comparisons >= 8).all()
        ),
        "overall_marginal_improvement_gt_0_25_pp": bool(
            overall_marginal_difference > 0.25
        ),
        "positive_marginal_difference_at_least_two_counts": bool(
            (marginal_by_count.median_recovery_difference_pp > 0).sum() >= 2
        ),
        "median_curve_ratio_le_1_05_all_counts": bool(
            (baseline_by_count.median_val_curve_mse_ratio <= 1.05).all()
        ),
        "lower_covariance_loss_than_marginal_overall": bool(
            marginal_comparisons.projected_covariance_loss_ratio.median() < 1.0
        ),
        "lower_covariance_loss_than_marginal_at_least_two_counts": bool(
            (marginal_by_count.median_covariance_loss_ratio < 1.0).sum() >= 2
        ),
        "gradient_alignment_at_least_80_percent": bool(
            gradient_alignment / len(gradients) >= 0.8
        ),
    }
    preferred = all(gates.values())
    common_validity = all(
        value
        for key, value in gates.items()
        if key not in {
            "overall_marginal_improvement_gt_0_25_pp",
            "positive_marginal_difference_at_least_two_counts",
        }
    )
    if preferred:
        recommendation = "prefer_projected_covariance"
    elif common_validity and overall_marginal_difference >= -0.25:
        recommendation = "noninferior_not_preferred"
    else:
        recommendation = "retain_marginal"
    decision = {
        "status": "evaluated",
        "recommendation": recommendation,
        "fit_count": int(len(raw)),
        "selected_endpoint_count": int(len(projected)),
        "overall_median_recovery_difference_vs_marginal_pp": overall_marginal_difference,
        "gradient_aligned": gradient_alignment,
        "gradient_comparisons": int(len(gradients)),
        "gates": gates,
        "ess_used_as_gate": False,
        "recovery_used_for_calibration": False,
        "comparators_reused": True,
        "projection_threshold": PROJECTION_THRESHOLD,
    }
    (results_dir / "decision.json").write_text(json.dumps(decision, indent=2))
    lines = [
        "# Projected peptide PF covariance optimization",
        "",
        f"- Recommendation: `{recommendation}`.",
        f"- New fits: {len(raw)}; selected endpoints: {len(projected)}; all finite: {finite}.",
        f"- Overall median recovery difference versus marginal: {overall_marginal_difference:+.3f} pp.",
        f"- Gradients aligned at covariance-MSE endpoints: {gradient_alignment}/{len(gradients)}.",
        "- ESS was not used as a gate.",
    ]
    (results_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "litmus",
            "calibrate",
            "final",
            "analyze",
            "projected-calibrate",
            "projected-final",
            "projected-analyze",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--litmus-dir", type=Path)
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--comparator-dir", type=Path)
    parser.add_argument("--counts", default="15,30,60")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[5]
    if args.phase == "litmus":
        run_litmus(args.output_dir.resolve(), repo_root)
    elif args.phase == "analyze":
        analyze_final(args.litmus_dir.resolve(), args.output_dir.resolve(), repo_root)
    elif args.phase == "projected-analyze":
        analyze_projected(
            args.litmus_dir.resolve(),
            args.output_dir.resolve(),
            args.comparator_dir.resolve(),
            repo_root,
        )
    elif args.phase in {"projected-calibrate", "projected-final"}:
        run_projected_phase(
            args.phase,
            args.litmus_dir.resolve(),
            args.output_dir.resolve(),
            repo_root,
            args.comparator_dir.resolve(),
            args.calibration_dir.resolve() if args.calibration_dir else None,
        )
    else:
        run_fit_phase(
            args.phase,
            args.litmus_dir.resolve(),
            args.output_dir.resolve(),
            repo_root,
            args.calibration_dir.resolve() if args.calibration_dir else None,
            tuple(int(value) for value in args.counts.split(",")),
        )


if __name__ == "__main__":
    main()
