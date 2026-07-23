#!/usr/bin/env python3
"""Synthetic TRI reweighting against analytic peptide log-PF target moments."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

from jaxent.src.analysis.pf_variance import (
    average_first_uptake,
    conditional_subset_effective_sample_size,
    conditional_variance_log_ratio_loss,
    conditional_variance_profile,
    covariance_profiles_from_covariance,
    covariance_mse,
    curve_precision_from_target_uptake,
    jensen_shannon_divergence,
    jensen_shannon_recovery_percent,
    kl_to_uniform,
    map_frame_log_pf_to_peptides,
    map_residue_uptake_to_peptides,
    marginal_variance_profile,
    inverse_overlap_degree_weights,
    overlap_projection,
    projected_log_euclidean_covariance_loss,
    uptake_from_log_pf,
    weighted_population_covariance,
    weighted_variance_log_ratio_loss,
    weights_from_logits,
)
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import create_sparse_map, normalize_sparse_map_rows
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.interfaces.topology import PTSerialiser, TopologyFactory
from jaxent.src.models.HDX.BV.features import BV_input_features


TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], dtype=np.float32)
TARGET_POPULATIONS = jnp.asarray([0.4, 0.6, 0.0], dtype=jnp.float32)
PANEL_SEED = 20260714
SPLIT_SEEDS = (0, 42, 84)
LOCKED_SPLITS = {
    ("equal", 0): ((0, 1, 2, 9, 10, 11), (3, 4, 5, 6, 7, 8, 12, 13, 14)),
    ("equal", 1): ((1, 2, 3, 4, 5, 6), (0, 7, 8, 9, 10, 11, 12, 13, 14)),
    ("equal", 2): ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11, 12, 13, 14)),
    ("random_fixed", 0): ((3, 4, 5, 6, 7), (0, 1, 2, 8, 9, 10, 11, 12, 13, 14)),
    ("random_fixed", 1): ((3, 4, 5, 6, 7, 9), (0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14)),
    ("random_fixed", 2): ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9, 10, 11, 12, 13, 14)),
    ("random_variable", 0): ((0, 1, 13, 14), (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)),
    ("random_variable", 1): ((2, 3, 4, 5, 6, 7, 8, 11, 12), (0, 1, 8, 9, 10, 13, 14)),
    ("random_variable", 2): ((0, 1, 2, 3, 4, 5, 6, 7), (7, 8, 9, 10, 11, 12, 13, 14)),
}


@dataclass(frozen=True)
class Config:
    panels: tuple[str, ...] = ("equal", "random_fixed", "random_variable")
    split_indices: tuple[int, ...] = (0, 1, 2)
    gammas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    maxent_values: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10)
    target_scales: tuple[float, ...] = (0.1, 1.0)
    starts: int = 1
    steps: int = 2000
    learning_rate: float = 0.05
    pilot: bool = False
    smoke: bool = False
    validate_only: bool = False


@dataclass(frozen=True)
class Ensemble:
    log_pf: np.ndarray
    k_ints: np.ndarray
    assignments: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ensemble(feature_path: Path, assignment_path: Path) -> Ensemble:
    with np.load(feature_path) as features:
        log_pf = 0.35 * features["heavy_contacts"] + 2.0 * features["acceptor_contacts"]
        k_ints = np.asarray(features["k_ints"])
    assignments = pd.read_csv(assignment_path)["cluster_assignment"].to_numpy(dtype=int)
    return Ensemble(log_pf=log_pf, k_ints=k_ints, assignments=assignments)


def _truth_weights(assignments: np.ndarray) -> np.ndarray:
    weights = np.zeros(assignments.size, dtype=np.float32)
    weights[assignments == 0] = 0.4 / np.sum(assignments == 0)
    weights[assignments == 1] = 0.6 / np.sum(assignments == 1)
    return weights


def _latent_distribution(log_pf: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(log_pf, dtype=np.float64) @ np.asarray(weights, dtype=np.float64)
    covariance = np.asarray(weighted_population_covariance(log_pf, weights), dtype=np.float64)
    return mean, covariance


def _relative_error(observed: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(expected)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(observed - expected) / denominator)


def _validate_target_moments(
    panels: dict[str, dict[str, Any]],
    residue_covariance: np.ndarray,
    config: Config,
) -> tuple[pd.DataFrame, bool]:
    rows = []
    for panel_name in config.panels:
        mapping = np.asarray(panels[panel_name]["mapping"], dtype=np.float64)
        base = mapping @ residue_covariance @ mapping.T
        target_values = panels[panel_name]["target_peptide_log_pf"]
        truth = panels[panel_name]["target_weights"]
        directly_mapped = np.asarray(weighted_population_covariance(target_values, truth))
        mapping_error = _relative_error(directly_mapped, base)
        for target_scale in (0.0, *config.target_scales):
            expected = target_scale**2 * base
            observed = panels[panel_name]["target_covariances"].get(target_scale, expected)
            rows.append(
                {
                    "panel": panel_name,
                    "target_scale": target_scale,
                    "mapping_relative_error": mapping_error,
                    "scaled_covariance_relative_error": _relative_error(observed, expected)
                    if target_scale > 0
                    else float(np.linalg.norm(observed)),
                    "covariance_trace": float(np.trace(observed)),
                }
            )
    frame = pd.DataFrame(rows)
    passed = bool(
        (frame.mapping_relative_error < 2e-6).all()
        and (frame.scaled_covariance_relative_error < 1e-12).all()
        and (frame.loc[frame.target_scale == 0.0, "covariance_trace"] == 0.0).all()
    )
    return frame, passed


def generate_panel_bounds() -> dict[str, list[tuple[int, int]]]:
    # Construct intended exchangeable intervals first.  Physical peptide starts
    # are one residue earlier because peptide_trim=1 removes the first residue.
    exchangeable_residues = np.arange(2, 310)
    tiles = np.array_split(exchangeable_residues, 15)
    equal_active = [(int(tile[0]), int(tile[-1])) for tile in tiles]
    fixed_length = int(np.median([len(tile) for tile in tiles]))
    rng = np.random.default_rng(PANEL_SEED)
    fixed_starts = np.sort(
        rng.choice(np.arange(2, 309 - fixed_length + 2), size=15, replace=False)
    )
    random_fixed_active = [
        (int(start), int(start + fixed_length - 1)) for start in fixed_starts
    ]
    variable_lengths = rng.integers(10, 31, size=15)
    variable_starts = np.asarray(
        [rng.integers(2, 309 - int(length) + 2) for length in variable_lengths]
    )
    order = np.argsort(variable_starts)
    random_variable_active = [
        (int(variable_starts[index]), int(variable_starts[index] + variable_lengths[index] - 1))
        for index in order
    ]

    def physical_bounds(active_bounds: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(start - 1, end) for start, end in active_bounds]

    return {
        "equal": physical_bounds(equal_active),
        "random_fixed": physical_bounds(random_fixed_active),
        "random_variable": physical_bounds(random_variable_active),
    }


def _build_panel(
    name: str,
    bounds: list[tuple[int, int]],
    feature_path: Path,
    topology_path: Path,
) -> tuple[list[HDX_peptide], np.ndarray, list[dict[str, Any]]]:
    feature_topology = PTSerialiser.load_list_from_json(topology_path)
    features = BV_input_features.load(feature_path)
    data = [
        HDX_peptide(
            dfrac=[0.0] * len(TIMEPOINTS),
            top=TopologyFactory.from_range(
                chain="A",
                start=start,
                end=end,
                fragment_index=index,
                peptide=True,
                peptide_trim=1,
                fragment_name=f"fake_{name}_{index:02d}",
            ),
        )
        for index, (start, end) in enumerate(bounds)
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        sparse_map = create_sparse_map(features, feature_topology, data, check_trim=True)
    normalized = normalize_sparse_map_rows(sparse_map)
    dense = np.asarray(normalized.todense())
    if not np.allclose(dense.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError(f"{name}: normalized sparse-map rows do not sum to one")
    represented_residues = [int(top.residues[0]) for top in feature_topology]
    metadata = []
    for index, ((start, end), row) in enumerate(zip(bounds, dense, strict=True)):
        mapped = [
            residue
            for residue, value in zip(represented_residues, row, strict=True)
            if value > 0
        ]
        metadata.append(
            {
                "panel": name,
                "fragment_index": index,
                "start": start,
                "end": end,
                "physical_length": end - start + 1,
                "peptide_trim": 1,
                "intended_exchangeable_start": start + 1,
                "intended_exchangeable_end": end,
                "mapped_residue_count": len(mapped),
                "mapped_residues": mapped,
                "map_row_sum": float(row.sum()),
            }
        )
    return data, dense, metadata


def _split_panel(
    data: list[HDX_peptide], feature_topology: list, split_index: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed = SPLIT_SEEDS[split_index]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        loader = ExpD_Dataloader(data=data)
        splitter = DataSplitter(
            dataset=loader,
            common_residues=set(feature_topology),
            peptide_trim=1,
            check_trim=True,
            centrality=False,
            train_size=0.5,
            random_seed=seed,
            min_split_size=3,
        )
        train, val = splitter.sequence_cluster_split(n_clusters=5, remove_overlap=True)
    train_indices = np.asarray([int(item.top.fragment_index) for item in train], dtype=int)
    val_indices = np.asarray([int(item.top.fragment_index) for item in val], dtype=int)
    duplicate_ids = [
        int(value) for value in sorted(set(train_indices).intersection(set(val_indices)))
    ]
    metadata = {
        "requested_seed": seed,
        "final_seed_after_retries": splitter.random_seed,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "duplicate_fragment_ids": duplicate_ids,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_clusters": 5,
        "remove_overlap": True,
    }
    return train_indices, val_indices, metadata


def _cluster_probabilities(weights: jax.Array, assignments: np.ndarray) -> jax.Array:
    return jnp.stack(
        [jnp.sum(weights[jnp.asarray(assignments == label)]) for label in (0, 1, -1)]
    )


def _metrics(weights: np.ndarray, assignments: np.ndarray) -> dict[str, float]:
    populations = np.asarray(_cluster_probabilities(jnp.asarray(weights), assignments))
    open_mask = assignments == 0
    return {
        "open_population": float(populations[0]),
        "closed_population": float(populations[1]),
        "intermediate_population": float(populations[2]),
        "recovery_pct": float(
            jensen_shannon_recovery_percent(jnp.asarray(populations), TARGET_POPULATIONS)
        ),
        "open_ess": float(conditional_subset_effective_sample_size(weights, open_mask)),
        "open_ess_fraction": float(
            conditional_subset_effective_sample_size(weights, open_mask) / np.sum(open_mask)
        ),
    }


def optimize(
    *,
    predicted_residue_log_pf: np.ndarray,
    predicted_peptide_log_pf: np.ndarray,
    predicted_k_ints: np.ndarray,
    peptide_mapping: np.ndarray,
    target_peptide_covariance: np.ndarray,
    target_uptake: np.ndarray,
    clean_target_uptake: np.ndarray,
    assignments: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    method: str,
    gamma: float,
    maxent_value: float,
    alpha: float,
    steps: int,
    learning_rate: float,
    start_seed: int,
    overlap_weighted: bool = False,
    projection_threshold: float = 1e-6,
) -> dict[str, Any]:
    z_pred_peptide = jnp.asarray(predicted_peptide_log_pf)
    target_covariance = jnp.asarray(target_peptide_covariance)
    z_pred_residue = jnp.asarray(predicted_residue_log_pf)
    mapping = jnp.asarray(peptide_mapping)
    k_ints = jnp.asarray(predicted_k_ints)
    target = jnp.asarray(target_uptake)
    clean_target = jnp.asarray(clean_target_uptake)
    train_idx = jnp.asarray(train_indices)
    val_idx = jnp.asarray(val_indices)
    train_precision = curve_precision_from_target_uptake(
        clean_target[:, train_idx], alpha=0.05
    )
    val_precision = curve_precision_from_target_uptake(clean_target[:, val_idx], alpha=0.05)
    if method not in {"covariance_mse", "marginal", "conditional", "projected_covariance"}:
        raise ValueError(f"Unknown fitting method: {method}")
    profile_function = (
        marginal_variance_profile if method == "marginal" else conditional_variance_profile
    )
    if method != "covariance_mse":
        train_covariance = target_covariance[jnp.ix_(train_idx, train_idx)]
        val_covariance = target_covariance[jnp.ix_(val_idx, val_idx)]
        if method == "projected_covariance":
            train_projection = jax.lax.stop_gradient(
                overlap_projection(mapping[train_idx], projection_threshold)
            )
            val_projection = jax.lax.stop_gradient(
                overlap_projection(mapping[val_idx], projection_threshold)
            )
            target_profile_train = target_profile_val = None
            train_profile_weights = val_profile_weights = None
        else:
            target_profile_train = jax.lax.stop_gradient(
                covariance_profiles_from_covariance(train_covariance, alpha=alpha)[
                    1 if method == "marginal" else 3
                ]
            )
            target_profile_val = jax.lax.stop_gradient(
                covariance_profiles_from_covariance(val_covariance, alpha=alpha)[
                    1 if method == "marginal" else 3
                ]
            )
            train_profile_weights = jax.lax.stop_gradient(
                inverse_overlap_degree_weights(mapping[train_idx])
            )
            val_profile_weights = jax.lax.stop_gradient(
                inverse_overlap_degree_weights(mapping[val_idx])
            )
    else:
        target_profile_train = jnp.ones(train_idx.size)
        target_profile_val = jnp.ones(val_idx.size)

    def predict(weights: jax.Array) -> jax.Array:
        residue_uptake = average_first_uptake(z_pred_residue, k_ints, TIMEPOINTS, weights)
        return map_residue_uptake_to_peptides(mapping, residue_uptake)

    uniform_logits = jnp.zeros(z_pred_residue.shape[1], dtype=z_pred_residue.dtype)
    uniform_weights = weights_from_logits(uniform_logits)
    initial_mean = covariance_mse(
        predict(uniform_weights)[:, train_idx], clean_target[:, train_idx], train_precision
    )
    mean_scale = jnp.maximum(initial_mean, jnp.asarray(1e-12, dtype=z_pred_residue.dtype))

    def components(logits: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        weights = weights_from_logits(logits)
        prediction = predict(weights)
        mean_loss = covariance_mse(
            prediction[:, train_idx], target[:, train_idx], train_precision
        )
        if method == "covariance_mse":
            profile_loss = jnp.asarray(0.0, dtype=mean_loss.dtype)
        elif method == "projected_covariance":
            predicted_covariance = weighted_population_covariance(
                z_pred_peptide[train_idx], weights
            )
            profile_loss = projected_log_euclidean_covariance_loss(
                predicted_covariance,
                train_covariance,
                train_projection,
                alpha=alpha,
            )
        else:
            predicted_profile = profile_function(z_pred_peptide[train_idx], weights, alpha=alpha)
            profile_loss = (
                weighted_variance_log_ratio_loss(
                    predicted_profile, target_profile_train, train_profile_weights
                )
                if overlap_weighted
                else conditional_variance_log_ratio_loss(predicted_profile, target_profile_train)
            )
        maxent_loss = kl_to_uniform(weights)
        total = mean_loss / mean_scale + gamma * profile_loss + maxent_value * maxent_loss
        return total, (mean_loss, profile_loss, maxent_loss)

    rng = np.random.default_rng(start_seed)
    initial_logits = np.zeros(z_pred_residue.shape[1], dtype=np.float32)
    if start_seed:
        initial_logits += rng.normal(0.0, 0.01, initial_logits.shape).astype(np.float32)
    logits = jnp.asarray(initial_logits)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(logits)

    @jax.jit
    def step(current: jax.Array, current_state: optax.OptState):
        (loss, _), gradient = jax.value_and_grad(components, has_aux=True)(current)
        updates, next_state = optimizer.update(gradient, current_state, current)
        return optax.apply_updates(current, updates), next_state, loss

    final_step_loss = jnp.nan
    for _ in range(steps):
        logits, state, final_step_loss = step(logits, state)
    total, (train_mean, train_profile, maxent_loss) = components(logits)
    fitted_weights = weights_from_logits(logits)
    prediction = predict(fitted_weights)
    val_mean = covariance_mse(prediction[:, val_idx], target[:, val_idx], val_precision)
    if method == "covariance_mse":
        val_profile = jnp.asarray(0.0)
    elif method == "projected_covariance":
        val_covariance_predicted = weighted_population_covariance(
            z_pred_peptide[val_idx], fitted_weights
        )
        val_profile = projected_log_euclidean_covariance_loss(
            val_covariance_predicted,
            val_covariance,
            val_projection,
            alpha=alpha,
        )
    else:
        val_predicted_profile = profile_function(
            z_pred_peptide[val_idx], fitted_weights, alpha=alpha
        )
        val_profile = (
            weighted_variance_log_ratio_loss(
                val_predicted_profile, target_profile_val, val_profile_weights
            )
            if overlap_weighted
            else conditional_variance_log_ratio_loss(
                val_predicted_profile, target_profile_val
            )
        )
    weights_np = np.asarray(fitted_weights)
    return {
        "train_total": float(total),
        "train_curve_mse": float(train_mean),
        "train_pf_profile_loss": float(train_profile),
        "val_curve_mse": float(val_mean),
        "val_pf_profile_loss": float(val_profile),
        "maxent_kl": float(maxent_loss),
        "effective_sample_size": float(1.0 / np.square(weights_np).sum()),
        "final_step_loss": float(final_step_loss),
        "finite": bool(np.isfinite(float(total)) and np.isfinite(weights_np).all()),
        "weights": weights_np,
        **_metrics(weights_np, assignments),
    }


def _select(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting = [
        "panel",
        "split_index",
        "target_scale",
        "method",
        "gamma",
        "maxent_value",
        "alpha",
    ]
    best_starts = raw.loc[raw.groupby(setting)["train_total"].idxmin()].copy()
    selected = best_starts.loc[
        best_starts.groupby(
            ["panel", "split_index", "target_scale", "method"]
        )["val_curve_mse"].idxmin()
    ].copy()
    return best_starts, selected


def _gradient_diagnostics(
    baseline_weights: np.ndarray,
    method: str,
    alpha: float,
    predicted_peptide_log_pf: np.ndarray,
    target_peptide_covariance: np.ndarray,
    assignments: np.ndarray,
    train_indices: np.ndarray,
    peptide_mapping: np.ndarray | None = None,
    overlap_weighted: bool = False,
) -> tuple[float, float]:
    profile_function = (
        marginal_variance_profile if method == "marginal" else conditional_variance_profile
    )
    z_pred = jnp.asarray(predicted_peptide_log_pf)[train_indices]
    target_covariance = jnp.asarray(target_peptide_covariance)[
        jnp.ix_(jnp.asarray(train_indices), jnp.asarray(train_indices))
    ]
    target_profile = jax.lax.stop_gradient(
        covariance_profiles_from_covariance(target_covariance, alpha=alpha)[
            1 if method == "marginal" else 3
        ]
    )
    profile_weights = (
        jax.lax.stop_gradient(
            inverse_overlap_degree_weights(
                jnp.asarray(peptide_mapping)[jnp.asarray(train_indices)]
            )
        )
        if overlap_weighted and peptide_mapping is not None
        else None
    )

    def profile_loss(logits):
        profile = profile_function(z_pred, weights_from_logits(logits), alpha=alpha)
        return (
            weighted_variance_log_ratio_loss(profile, target_profile, profile_weights)
            if profile_weights is not None
            else conditional_variance_log_ratio_loss(profile, target_profile)
        )

    def population_jsd(logits):
        populations = _cluster_probabilities(weights_from_logits(logits), assignments)
        return jensen_shannon_divergence(populations, TARGET_POPULATIONS)

    logits = jnp.log(jnp.asarray(baseline_weights))
    profile_gradient = jax.grad(profile_loss)(logits)
    jsd_gradient = jax.grad(population_jsd)(logits)
    cosine = jnp.vdot(profile_gradient, jsd_gradient) / (
        jnp.linalg.norm(profile_gradient) * jnp.linalg.norm(jsd_gradient) + 1e-30
    )
    stepped = logits - 0.01 * profile_gradient / (jnp.linalg.norm(profile_gradient) + 1e-30)
    return float(cosine), float(population_jsd(stepped) - population_jsd(logits))


def run(config: Config, output_dir: Path, repo_root: Path) -> None:
    example_root = repo_root / "jaxent/examples/1_IsoValidation_OMass"
    fitting_root = example_root / "fitting/jaxENT"
    feature_root = fitting_root / "_featurise"
    cluster_root = example_root / "data/_clustering_results"
    paths = {
        "bi_features": feature_root / "features_iso_bi.npz",
        "tri_features": feature_root / "features_iso_tri.npz",
        "bi_topology": feature_root / "topology_iso_bi.json",
        "tri_topology": feature_root / "topology_iso_tri.json",
        "bi_clusters": cluster_root / "cluster_assignments_ISO_BI.csv",
        "tri_clusters": cluster_root / "cluster_assignments_ISO_TRI.csv",
    }
    bi = _load_ensemble(paths["bi_features"], paths["bi_clusters"])
    tri = _load_ensemble(paths["tri_features"], paths["tri_clusters"])
    truth = _truth_weights(bi.assignments)
    mean_log_pf, residue_covariance = _latent_distribution(bi.log_pf, truth)
    feature_topology = PTSerialiser.load_list_from_json(paths["tri_topology"])
    bounds_by_panel = generate_panel_bounds()
    output_dir.mkdir(parents=True, exist_ok=True)

    panels: dict[str, dict[str, Any]] = {}
    panel_metadata = []
    split_metadata: dict[str, Any] = {}
    for panel_name in config.panels:
        data, mapping, metadata = _build_panel(
            panel_name,
            bounds_by_panel[panel_name],
            paths["tri_features"],
            paths["tri_topology"],
        )
        clean_residue_target = uptake_from_log_pf(mean_log_pf, bi.k_ints, TIMEPOINTS)
        clean_target_uptake = np.asarray(
            map_residue_uptake_to_peptides(mapping, clean_residue_target)
        )
        if not np.isfinite(clean_target_uptake).all() or not (
            (clean_target_uptake >= 0.0).all() and (clean_target_uptake <= 1.0).all()
        ):
            raise ValueError(f"{panel_name}: analytic target uptake is invalid")
        data = [
            HDX_peptide(dfrac=clean_target_uptake[:, index].tolist(), top=item.top)
            for index, item in enumerate(data)
        ]
        target_peptide_log_pf = np.asarray(map_frame_log_pf_to_peptides(mapping, bi.log_pf))
        predicted_peptide_log_pf = np.asarray(map_frame_log_pf_to_peptides(mapping, tri.log_pf))
        base_peptide_covariance = np.asarray(
            mapping @ residue_covariance @ mapping.T, dtype=np.float64
        )
        target_covariances = {
            float(scale): float(scale) ** 2 * base_peptide_covariance
            for scale in (0.0, *config.target_scales)
        }
        panels[panel_name] = {
            "data": data,
            "mapping": mapping,
            "clean_target_uptake": clean_target_uptake,
            "target_peptide_log_pf": target_peptide_log_pf,
            "predicted_peptide_log_pf": predicted_peptide_log_pf,
            "target_covariances": target_covariances,
            "target_weights": truth,
        }
        panel_metadata.extend(metadata)
        PTSerialiser.save_list_to_json(
            [item.top for item in data], output_dir / f"panel_{panel_name}_topology.json"
        )
        np.savez_compressed(output_dir / f"panel_{panel_name}_mapping.npz", mapping=mapping)
        for split_index in config.split_indices:
            _, _, metadata_split = _split_panel(data, feature_topology, split_index)
            locked_train, locked_val = LOCKED_SPLITS[(panel_name, split_index)]
            train_idx = np.asarray(locked_train, dtype=int)
            val_idx = np.asarray(locked_val, dtype=int)
            metadata_split.update(
                {
                    "train_indices": train_idx.tolist(),
                    "val_indices": val_idx.tolist(),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "locked_from_2026_07_15_run": True,
                }
            )
            panels[panel_name][f"split_{split_index}"] = (train_idx, val_idx)
            split_metadata[f"{panel_name}_{split_index}"] = metadata_split
    pd.DataFrame(panel_metadata).to_json(
        output_dir / "peptide_panels.json", orient="records", indent=2
    )

    validation, validation_passed = _validate_target_moments(
        panels, residue_covariance, config
    )
    validation.to_csv(output_dir / "target_moment_validation.csv", index=False)
    (output_dir / "target_moment_decision.json").write_text(
        json.dumps(
            {
                "status": (
                    "not_evaluated"
                    if config.smoke
                    else "passed"
                    if validation_passed
                    else "failed"
                ),
                "passed": validation_passed,
                "acceptance": {
                    "mapping_relative_error_max": 2e-6,
                    "scaled_covariance_relative_error_max": 1e-12,
                    "zero_scale_covariance_trace": 0.0,
                },
            },
            indent=2,
        )
    )
    if not validation_passed and not config.smoke:
        raise RuntimeError("Analytic target-moment validation failed; fitting was not started")

    manifest = {
        "config": asdict(config),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "panel_seed": PANEL_SEED,
        "split_metadata": split_metadata,
        "target_populations": [0.4, 0.6, 0.0],
        "target_semantics": "analytic average-first uptake from E[Z]=z_bar; no latent draw",
        "pf_semantics": "C_target_peptide(s)=M (s^2 C_target_residue) M^T",
        "prediction_scaled_by_target_scale": False,
        "clean_curve_geometry_reused_across_target_scales": True,
        "target_scales": list(config.target_scales),
        "zero_scale_policy": "preflight_only_nonidentifiable",
        "primary_recovery_scale": 1.0,
        "cluster_population_semantics": {
            "open_total": 0.4,
            "closed_total": 0.6,
            "within_cluster": "uniform",
        },
        "split_leakage_note": (
            "Current sequence-cluster behavior retained by request; duplicate IDs are reported."
        ),
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if config.validate_only:
        return

    rows: list[dict[str, Any]] = []
    weights_by_run: dict[str, np.ndarray] = {}
    baseline_cache: dict[tuple[Any, ...], tuple[dict[str, Any], np.ndarray]] = {}
    methods = ("covariance_mse", "marginal", "conditional")
    for panel_name in config.panels:
        panel = panels[panel_name]
        for split_index in config.split_indices:
            train_indices, val_indices = panel[f"split_{split_index}"]
            for target_scale in config.target_scales:
                target_uptake = panel["clean_target_uptake"]
                for method in methods:
                    gammas = (0.0,) if method == "covariance_mse" else config.gammas
                    alphas = (0.05,) if method == "covariance_mse" else config.alphas
                    for gamma in gammas:
                        for alpha in alphas:
                            for maxent_value in config.maxent_values:
                                for start in range(config.starts):
                                    run_id = (
                                        f"{panel_name}_split{split_index:03d}"
                                        f"_scale{target_scale:g}_{method}"
                                        f"_g{gamma:g}_m{maxent_value:g}_a{alpha:g}_s{start}"
                                    )
                                    cache_key = (
                                        panel_name,
                                        split_index,
                                        method,
                                        gamma,
                                        maxent_value,
                                        alpha,
                                        start,
                                    )
                                    if method == "covariance_mse" and cache_key in baseline_cache:
                                        cached_result, cached_weights = baseline_cache[cache_key]
                                        result = dict(cached_result)
                                        result["weights"] = cached_weights.copy()
                                    else:
                                        result = optimize(
                                            predicted_residue_log_pf=tri.log_pf,
                                            predicted_peptide_log_pf=panel[
                                                "predicted_peptide_log_pf"
                                            ],
                                            predicted_k_ints=tri.k_ints,
                                            peptide_mapping=panel["mapping"],
                                            target_peptide_covariance=panel[
                                                "target_covariances"
                                            ][target_scale],
                                            target_uptake=target_uptake,
                                            clean_target_uptake=panel[
                                                "clean_target_uptake"
                                            ],
                                            assignments=tri.assignments,
                                            train_indices=train_indices,
                                            val_indices=val_indices,
                                            method=method,
                                            gamma=gamma,
                                            maxent_value=maxent_value,
                                            alpha=alpha,
                                            steps=config.steps,
                                            learning_rate=config.learning_rate,
                                            start_seed=start,
                                        )
                                        if method == "covariance_mse":
                                            baseline_cache[cache_key] = (
                                                {k: v for k, v in result.items() if k != "weights"},
                                                np.asarray(result["weights"]).copy(),
                                            )
                                    weights_by_run[run_id] = result.pop("weights")
                                    rows.append(
                                        {
                                            "run_id": run_id,
                                            "panel": panel_name,
                                            "split_index": split_index,
                                            "target_scale": target_scale,
                                            "method": method,
                                            "gamma": gamma,
                                            "maxent_value": maxent_value,
                                            "alpha": alpha,
                                            "start": start,
                                            **result,
                                        }
                                    )
                                    pd.DataFrame(rows).to_csv(
                                        output_dir / "raw_results.csv", index=False
                                    )

    raw = pd.DataFrame(rows)
    best_starts, selected = _select(raw)
    diagnostics = []
    selected_weights: dict[str, np.ndarray] = {}
    result_groups = ["panel", "split_index", "target_scale"]
    for (panel_name, split_index, target_scale), group in selected.groupby(
        result_groups
    ):
        baseline = group[group.method == "covariance_mse"].iloc[0]
        baseline_weights = weights_by_run[baseline.run_id]
        train_indices, _ = panels[panel_name][f"split_{split_index}"]
        for _, row in group.iterrows():
            selected_weights[row.run_id] = weights_by_run[row.run_id]
            if row.method == "covariance_mse":
                cosine, step_change = np.nan, np.nan
            else:
                cosine, step_change = _gradient_diagnostics(
                    baseline_weights,
                    row.method,
                    float(row.alpha),
                    panels[panel_name]["predicted_peptide_log_pf"],
                    panels[panel_name]["target_covariances"][target_scale],
                    tri.assignments,
                    train_indices,
                )
            diagnostics.append(
                {
                    "run_id": row.run_id,
                    "baseline_gradient_cosine": cosine,
                    "baseline_gradient_jsd_step_change": step_change,
                }
            )
    selected = selected.merge(pd.DataFrame(diagnostics), on="run_id", validate="one_to_one")
    selected.to_csv(output_dir / "selected_results.csv", index=False)
    np.savez_compressed(output_dir / "selected_weights.npz", **selected_weights)

    comparisons = []
    for (panel_name, split_index, target_scale), group in selected.groupby(
        result_groups
    ):
        baseline = group[group.method == "covariance_mse"].iloc[0]
        for method in ("marginal", "conditional"):
            candidate = group[group.method == method].iloc[0]
            comparisons.append(
                {
                    "panel": panel_name,
                    "split_index": int(split_index),
                    "target_scale": float(target_scale),
                    "method": method,
                    "baseline_recovery_pct": float(baseline.recovery_pct),
                    "candidate_recovery_pct": float(candidate.recovery_pct),
                    "recovery_gain_pp": float(candidate.recovery_pct - baseline.recovery_pct),
                    "val_curve_mse_ratio": float(candidate.val_curve_mse / baseline.val_curve_mse),
                    "open_ess": float(candidate.open_ess),
                    "open_ess_fraction": float(candidate.open_ess_fraction),
                    "baseline_gradient_cosine": float(candidate.baseline_gradient_cosine),
                    "baseline_gradient_jsd_step_change": float(
                        candidate.baseline_gradient_jsd_step_change
                    ),
                }
            )
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output_dir / "comparisons.csv", index=False)
    support = []
    for (panel_name, target_scale, method), group in comparison_frame.groupby(
        ["panel", "target_scale", "method"]
    ):
        primary = bool(np.isclose(target_scale, 1.0))
        required_wins = 2 if primary else None
        support.append(
            {
                "panel": panel_name,
                "target_scale": float(target_scale),
                "method": method,
                "interpretation": "primary_recovery_gate" if primary else "tension_diagnostic",
                "recovery_wins": int((group.recovery_gain_pp > 0).sum()),
                "required_recovery_wins": required_wins,
                "recovery_supported": bool((group.recovery_gain_pp > 0).sum() >= 2)
                if primary
                else None,
                "median_mean_preserved": bool(group.val_curve_mse_ratio.median() <= 1.05),
                "gradient_aligned": int(
                    (
                        (group.baseline_gradient_cosine > 0)
                        & (group.baseline_gradient_jsd_step_change < 0)
                    ).sum()
                ),
                "required_gradient_aligned": required_wins,
                "gradient_supported": bool(
                    (
                        (group.baseline_gradient_cosine > 0)
                        & (group.baseline_gradient_jsd_step_change < 0)
                    ).sum()
                    >= 2
                ) if primary else None,
                "median_recovery_gain_pp": float(group.recovery_gain_pp.median()),
                "median_val_curve_mse_ratio": float(group.val_curve_mse_ratio.median()),
                "median_open_ess": float(group.open_ess.median()),
            }
        )
    expected_comparisons = len(config.panels) * len(config.split_indices) * len(
        config.target_scales
    ) * 2
    decision = {
        "status": (
            "evaluated"
            if not config.smoke
            and not config.pilot
            and len(comparison_frame) == expected_comparisons
            else "not_evaluated"
        ),
        "whole_ensemble_ess_used": False,
        "open_ess_used_as_gate": False,
        "support_rule": {
            "s1_recovery_and_gradient_wins": "at least 2 of 3 splits",
            "s0.1_policy": "diagnostic_only_nonrepresentable_by_40:60_unscaled_TRI_truth",
            "maximum_median_val_curve_mse_ratio": 1.05,
        },
        "groups": support,
    }
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2))

    if config.pilot:
        ranges = []
        for keys, group in raw.groupby(
            [
                "panel",
                "split_index",
                "target_scale",
                "method",
                "gamma",
                "maxent_value",
                "alpha",
            ]
        ):
            ranges.append(
                {
                    "setting": "|".join(map(str, keys)),
                    "train_total_range": float(group.train_total.max() - group.train_total.min()),
                    "val_curve_mse_range": float(
                        group.val_curve_mse.max() - group.val_curve_mse.min()
                    ),
                    "recovery_fraction_range": float(
                        (group.recovery_pct.max() - group.recovery_pct.min()) / 100.0
                    ),
                }
            )
        range_frame = pd.DataFrame(ranges)
        use_one_start = bool(
            (
                range_frame[
                    ["train_total_range", "val_curve_mse_range", "recovery_fraction_range"]
                ]
                <= 1e-6
            ).all(axis=None)
        )
        (output_dir / "pilot_decision.json").write_text(
            json.dumps(
                {
                    "use_one_start_for_full_grid": use_one_start,
                    "tolerance": 1e-6,
                    "max_ranges": {
                        column: float(range_frame[column].max())
                        for column in (
                            "train_total_range",
                            "val_curve_mse_range",
                            "recovery_fraction_range",
                        )
                    },
                },
                indent=2,
            )
        )


def _csv_tuple(value: str, cast: type = str) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panels", default="equal,random_fixed,random_variable")
    parser.add_argument("--split-indices", default="0,1,2")
    parser.add_argument("--starts", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--target-scales", default="0.1,1")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        config = Config(
            panels=("equal",),
            split_indices=(0,),
            gammas=(1.0,),
            maxent_values=(1.0,),
            alphas=(0.05,),
            target_scales=(0.1, 1.0),
            starts=1,
            smoke=True,
        )
    elif args.pilot:
        config = Config(
            panels=_csv_tuple(args.panels),
            split_indices=(0,),
            gammas=(10.0,),
            maxent_values=(1.0,),
            alphas=(0.01, 0.05, 0.10),
            target_scales=(0.1, 1.0),
            starts=3,
            steps=args.steps,
            pilot=True,
        )
    else:
        config = Config(
            panels=_csv_tuple(args.panels),
            split_indices=_csv_tuple(args.split_indices, int),
            target_scales=_csv_tuple(args.target_scales, float),
            starts=args.starts,
            steps=args.steps,
            validate_only=args.validate_only,
        )
    run(config, args.output_dir.resolve(), Path(__file__).resolve().parents[5])


if __name__ == "__main__":
    main()
