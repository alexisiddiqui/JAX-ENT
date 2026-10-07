"""Checkpoint 35: cheap ATLAS qualification of the original OMC Laplacian."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.cluster import KMeans

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config
from jaxent.examples.ATLAS_BV.analysis.laplacian_metric_comparison_checkpoint31 import (
    load_candidate_data,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation import (
    effective_sample_size,
    load_rows,
    pseudo_uptake,
    stable_seed,
)
from jaxent.examples.ATLAS_BV.analysis.laplacian_topology_checkpoint32 import (
    bandwidth_from_quantile,
    metric_distances,
)
from jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 import (
    atomic_parquet,
)
from jaxent.examples.common.analysis.clustering import calculate_recovery_percentage
from jaxent.src.analysis.pf_variance import conditional_subset_effective_sample_size
from jaxent.src.opt.loss.graph_laplacian import build_all_pairs_graph_from_distances
from jaxent.src.opt.loss.graph_laplacian import prior_relative_graph_energy
from jaxent.src.opt.loss.original_omc_laplacian import (
    build_omc_kernel,
    omc_graph_energy,
)


OUTPUT = HERE / "outputs/analysis/pairwise_geometry/checkpoint35_original_omc"
SIGMA_QUANTILES = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64)
TARGET_MASSES = (0.30, 0.50, 0.70)
STRENGTHS = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_CLUSTERS = (8, 10, 12, 14, 16)
SEED = 20260826


@dataclass(frozen=True, slots=True)
class RareBasin:
    labels: np.ndarray
    basin: int
    natural_mass: float
    n_frames: int
    compactness: float
    n_clusters: int


@dataclass(frozen=True, slots=True)
class ArmSpec:
    key: str
    family: str
    sigma_quantile: float | None
    normalise: bool = False
    rewired: bool = False


def arm_specs() -> tuple[ArmSpec, ...]:
    """Return the preregistered 20-arm grid in deterministic order."""
    specs = [ArmSpec("maxent", "maxent", None)]
    for quantile in SIGMA_QUANTILES:
        label = f"{quantile:g}"
        specs.extend(
            (
                ArmSpec(f"omc[{label}]", "omc", quantile),
                ArmSpec(f"omc_rewired[{label}]", "omc_rewired", quantile, rewired=True),
                ArmSpec(f"omc_norm[{label}]", "omc_norm", quantile, normalise=True),
            )
        )
    specs.append(ArmSpec("prior_rbf_locked", "prior_rbf_locked", 0.16))
    return tuple(specs)


def rare_basin(
    structural: np.ndarray,
    *,
    seed: int = SEED,
    mass_band: tuple[float, float] = (0.06, 0.15),
    min_frames: int = 8,
) -> RareBasin | None:
    """Select the most compact structurally coherent rare cluster."""
    matrix = np.asarray(structural, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("structural distance matrix must be square")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("structural distance matrix must be finite and non-negative")
    if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-8):
        raise ValueError("structural distance matrix must be symmetric")
    n_frames = len(matrix)
    pair = matrix[np.triu_indices(n_frames, k=1)]
    ensemble_mean = float(pair.mean()) if pair.size else 0.0
    if ensemble_mean <= 0:
        return None
    candidates: list[tuple[float, int, int, np.ndarray, int]] = []
    for count in N_CLUSTERS:
        if count > n_frames:
            continue
        labels = KMeans(n_clusters=count, n_init=20, random_state=seed).fit_predict(
            matrix
        )
        for basin in sorted(np.unique(labels)):
            members = np.flatnonzero(labels == basin)
            mass = len(members) / n_frames
            if not (mass_band[0] <= mass <= mass_band[1]) or len(members) < min_frames:
                continue
            within = matrix[np.ix_(members, members)]
            within_pairs = within[np.triu_indices(len(members), k=1)]
            compactness = (
                float(within_pairs.mean() / ensemble_mean) if within_pairs.size else 0.0
            )
            if compactness < 1.0:
                candidates.append(
                    (compactness, count, int(basin), labels, len(members))
                )
    if not candidates:
        return None
    compactness, count, basin, labels, member_count = min(
        candidates, key=lambda item: (item[0], item[1], item[2])
    )
    return RareBasin(
        labels=np.asarray(labels, dtype=int),
        basin=basin,
        natural_mass=member_count / n_frames,
        n_frames=member_count,
        compactness=compactness,
        n_clusters=count,
    )


def basin_mass_target(
    labels: np.ndarray,
    basin: int,
    mass: float,
) -> np.ndarray:
    """Put an absolute probability mass uniformly inside and outside a basin."""
    labels = np.asarray(labels)
    mask = labels == basin
    if not 0 < mass < 1 or not np.any(mask) or np.all(mask):
        raise ValueError("target requires 0 < mass < 1 and a proper non-empty basin")
    natural_mass = float(mask.mean())
    enrichment = float(mass / natural_mass)
    if enrichment < 2.0:
        raise AssertionError("target basin enrichment must be at least 2")
    target = np.empty(len(labels), dtype=float)
    target[mask] = mass / mask.sum()
    target[~mask] = (1.0 - mass) / (~mask).sum()
    np.testing.assert_allclose(target.sum(), 1.0)
    return target


def rewire_distances(distances: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    permutation = np.asarray(permutation, dtype=int)
    return np.asarray(distances)[np.ix_(permutation, permutation)]


def clustered_median_bootstrap(
    values: np.ndarray,
    system_ids: np.ndarray,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Percentile interval for a median, resampling whole systems."""
    values = np.asarray(values, dtype=float)
    systems = np.asarray(system_ids)
    unique = np.unique(systems)
    if not len(values) or len(values) != len(systems):
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    statistics = np.empty(samples, dtype=float)
    blocks = {system: values[systems == system] for system in unique}
    for index in range(samples):
        draw = rng.choice(unique, size=len(unique), replace=True)
        statistics[index] = np.median(
            np.concatenate([blocks[system] for system in draw])
        )
    low, high = np.quantile(statistics, (0.025, 0.975))
    return float(low), float(high)


def _fit_batch(
    observables: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    similarities: np.ndarray,
    normalise: np.ndarray,
    strengths: tuple[float, ...],
    steps: int,
) -> dict[str, np.ndarray]:
    """Fit all dense-kernel arms and strengths under one Adam state."""
    values = jnp.asarray(observables.reshape(-1, observables.shape[-1]))
    target = values @ jnp.asarray(truth)
    train_index = jnp.asarray(train)
    validation_index = jnp.asarray(validation)
    prior_logits = jnp.log(jnp.asarray(prior))
    similarity_values = jnp.asarray(similarities)
    normalise_values = jnp.asarray(normalise, dtype=bool)
    strength_values = jnp.asarray(strengths)
    scale = jnp.var(target[train_index]) + 1e-8

    def one_loss(logits, strength, similarity, use_normalised):
        weights = jax.nn.softmax(logits)
        prediction = values @ weights
        data = (
            jnp.mean(jnp.square(prediction[train_index] - target[train_index])) / scale
        )
        raw = omc_graph_energy(weights, similarity, normalise=False)
        denominator = jnp.sum(similarity * weights[:, None] * weights[None, :])
        normalised = jnp.where(denominator > 0, raw / denominator, 0.0)
        regularizer = jnp.where(use_normalised, normalised, raw)
        validation_loss = jnp.mean(
            jnp.square(prediction[validation_index] - target[validation_index])
        )
        return data + strength * regularizer, validation_loss

    def arm_losses(logits, similarity, use_normalised):
        return jax.vmap(one_loss, in_axes=(0, 0, None, None))(
            logits, strength_values, similarity, use_normalised
        )

    def objective(current):
        objectives, validation_losses = jax.vmap(arm_losses)(
            current, similarity_values, normalise_values
        )
        return jnp.sum(objectives), (objectives, validation_losses)

    logits = jnp.broadcast_to(
        prior_logits, (len(similarities), len(strengths), prior_logits.size)
    )
    optimizer = optax.adam(0.05)
    state = optimizer.init(logits)

    @jax.jit
    def step(current, opt_state):
        (_, _), gradient = jax.value_and_grad(objective, has_aux=True)(current)
        updates, next_state = optimizer.update(gradient, opt_state, current)
        return optax.apply_updates(current, updates), next_state

    midpoint = None
    for index in range(steps):
        logits, state = step(logits, state)
        if index + 1 == min(150, steps):
            midpoint = np.asarray(objective(logits)[1][0])
    objectives, validation_losses = objective(logits)[1]
    gradient = jax.grad(lambda x: objective(x)[0])(logits)
    return {
        "weights": np.asarray(jax.nn.softmax(logits)),
        "validation_mse": np.asarray(validation_losses),
        "objective_midpoint": np.asarray(
            midpoint if midpoint is not None else objectives
        ),
        "objective_final": np.asarray(objectives),
        "grad_norm": np.linalg.norm(np.asarray(gradient), axis=-1),
    }


def optimise_omc_batch(
    observables: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    similarities: np.ndarray,
    normalise: np.ndarray,
    strengths: tuple[float, ...] = STRENGTHS,
    steps: int = 300,
) -> dict[str, np.ndarray]:
    """Public testable wrapper for the checkpoint's dense JAX path."""
    return _fit_batch(
        observables,
        truth,
        prior,
        train,
        validation,
        similarities,
        normalise,
        strengths,
        steps,
    )


def _fit_comparator_batch(
    observables: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    strengths: tuple[float, ...],
    steps: int,
    regularizer: str,
    graph=None,
) -> dict[str, np.ndarray]:
    """Fit every strength for one incumbent comparator with full diagnostics."""
    values = jnp.asarray(observables.reshape(-1, observables.shape[-1]))
    target = values @ jnp.asarray(truth)
    train_index, validation_index = jnp.asarray(train), jnp.asarray(validation)
    prior_values = jnp.asarray(prior)
    prior_logits = jnp.log(prior_values)
    strength_values = jnp.asarray(strengths)
    scale = jnp.var(target[train_index]) + 1e-8

    def loss(logits, strength):
        weights = jax.nn.softmax(logits)
        prediction = values @ weights
        data = (
            jnp.mean(jnp.square(prediction[train_index] - target[train_index])) / scale
        )
        if regularizer == "maxent":
            reg = jnp.sum(prior_values * (prior_logits - jax.nn.log_softmax(logits)))
        else:
            reg = prior_relative_graph_energy(
                logits,
                prior_logits,
                graph.edge_sources,
                graph.edge_targets,
                graph.edge_weights,
            )
        validation_loss = jnp.mean(
            jnp.square(prediction[validation_index] - target[validation_index])
        )
        return data + strength * reg, validation_loss

    def objective(current):
        objectives, validations = jax.vmap(loss)(current, strength_values)
        return jnp.sum(objectives), (objectives, validations)

    logits = jnp.broadcast_to(prior_logits, (len(strengths), len(prior)))
    optimizer = optax.adam(0.05)
    state = optimizer.init(logits)

    @jax.jit
    def step(current, opt_state):
        (_, _), gradient = jax.value_and_grad(objective, has_aux=True)(current)
        updates, next_state = optimizer.update(gradient, opt_state, current)
        return optax.apply_updates(current, updates), next_state

    midpoint = None
    for index in range(steps):
        logits, state = step(logits, state)
        if index + 1 == min(150, steps):
            midpoint = np.asarray(objective(logits)[1][0])
    objectives, validations = objective(logits)[1]
    gradient = jax.grad(lambda x: objective(x)[0])(logits)
    return {
        "weights": np.asarray(jax.nn.softmax(logits)),
        "validation_mse": np.asarray(validations),
        "objective_midpoint": np.asarray(
            midpoint if midpoint is not None else objectives
        ),
        "objective_final": np.asarray(objectives),
        "grad_norm": np.linalg.norm(np.asarray(gradient), axis=-1),
    }


def _diagnostics(weights: np.ndarray, structural: np.ndarray, mask: np.ndarray) -> dict:
    support = weights >= 1.0 / (10.0 * len(weights))
    selected = weights[support]
    coefficient = float(selected.std() / selected.mean()) if selected.size else math.nan
    return {
        "ess": effective_sample_size(weights),
        "ess_fraction": effective_sample_size(weights) / len(weights),
        "target_ess": float(conditional_subset_effective_sample_size(weights, mask)),
        "target_ess_fraction": float(
            conditional_subset_effective_sample_size(weights, mask)
        )
        / mask.sum(),
        "support_size": int(support.sum()),
        "plateau_flag": bool(selected.size and coefficient < 0.05),
        "structural_dispersion": float(
            np.sum(weights[:, None] * weights[None, :] * structural)
        ),
    }


def _row(
    *,
    data: dict,
    target_mass: float,
    basin: RareBasin,
    spec: ArmSpec,
    sigma: float | None,
    strength: float,
    weights: np.ndarray,
    flat: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    truth: np.ndarray,
    structural: np.ndarray,
    validation_mse: float,
    objective_midpoint: float,
    objective_final: float,
    grad_norm: float,
) -> dict:
    mask = basin.labels == basin.basin
    recovered = float(weights[mask].sum())
    target_ratios = {
        str(label): float(truth[basin.labels == label].sum())
        for label in np.unique(basin.labels)
    }
    state_mapping = {int(label): str(label) for label in np.unique(basin.labels)}
    simplex_valid = bool(
        np.all(np.isfinite(weights))
        and np.all(weights >= 0)
        and np.isclose(weights.sum(), 1.0)
    )
    assert simplex_valid
    prediction = flat @ weights
    return {
        "system_id": data["system"],
        "n_residues": int(data["z"].shape[0]),
        "replica": 1,
        "target_mass": target_mass,
        "natural_basin_mass": basin.natural_mass,
        "n_target_frames": basin.n_frames,
        "compactness": basin.compactness,
        "enrichment_factor": target_mass / basin.natural_mass,
        "arm": spec.key,
        "arm_family": spec.family,
        "sigma_quantile": spec.sigma_quantile,
        "sigma": sigma,
        "strength": strength,
        "train_mse": float(np.mean(np.square(prediction[train] - target[train]))),
        "validation_mse": float(validation_mse),
        "test_mse": float(np.mean(np.square(prediction[test] - target[test]))),
        "recovered_basin_mass": recovered,
        "target_mass_error": abs(recovered - target_mass),
        "weight_tv": float(0.5 * np.abs(weights - truth).sum()),
        "recovery_percent": calculate_recovery_percentage(
            basin.labels, weights, target_ratios, state_mapping
        ),
        **_diagnostics(weights, structural, mask),
        "final_grad_norm": float(grad_norm),
        "objective_step150": float(objective_midpoint),
        "objective_step300": float(objective_final),
        "simplex_valid": simplex_valid,
    }


def screen_system(
    row: dict, config: dict, frame_cap: int, mass_band=(0.06, 0.15)
) -> dict:
    data, _ = load_candidate_data(row, config)
    global_indices, structural_full = data["matrices"][1]
    take = np.linspace(
        0, len(global_indices) - 1, min(frame_cap, len(global_indices)), dtype=int
    )
    structural = np.asarray(structural_full)[np.ix_(take, take)]
    basin = rare_basin(
        structural, seed=int(config["analysis"]["seed"]), mass_band=mass_band
    )
    base = {
        "system_id": data["system"],
        "length": int(row["length"]),
        "eligible": basin is not None,
    }
    if basin is None:
        return {**base, "reason": "no eligible rare compact basin"}
    return {
        **base,
        "reason": "",
        "natural_basin_mass": basin.natural_mass,
        "n_target_frames": basin.n_frames,
        "compactness": basin.compactness,
        "n_clusters": basin.n_clusters,
        "basin": basin.basin,
    }


def select_cohort(screen: pd.DataFrame) -> list[str]:
    eligible = screen.query("eligible").sort_values(["length", "system_id"])
    if len(eligible) < 6:
        return []
    step = max(1, len(eligible) // 6)
    return eligible.iloc[::step].head(6).system_id.tolist()


def reweight_system(
    row: dict,
    config: dict,
    target_masses: tuple[float, ...],
    strengths: tuple[float, ...],
    steps: int,
    frame_cap: int,
    mass_band=(0.06, 0.15),
    rare_min_frames: int = 8,
) -> list[dict]:
    data, values = load_candidate_data(row, config)
    global_indices, structural_full = data["matrices"][1]
    take = np.linspace(
        0, len(global_indices) - 1, min(frame_cap, len(global_indices)), dtype=int
    )
    indices = np.asarray(global_indices)[take]
    structural = np.asarray(structural_full)[np.ix_(take, take)]
    basin = rare_basin(
        structural,
        seed=int(config["analysis"]["seed"]),
        mass_band=mass_band,
        min_frames=rare_min_frames,
    )
    if basin is None:
        raise ValueError(
            f"selected system {data['system']} no longer has an eligible basin"
        )
    prior = np.full(len(indices), 1.0 / len(indices))
    work_distances = metric_distances(values["work_scale"], indices, "work_scale")
    rng = np.random.default_rng(stable_seed(data["system"], 1, "checkpoint35"))
    permutation = rng.permutation(len(indices))
    rewired_distances = rewire_distances(work_distances, permutation)
    observables = pseudo_uptake(data["z"][:, indices])
    flat = observables.reshape(-1, len(indices))
    split = rng.permutation(len(flat))
    n_train = max(1, int(0.6 * len(flat)))
    n_validation = max(1, int(0.2 * len(flat)))
    train = split[:n_train]
    validation = split[n_train : n_train + n_validation]
    test = split[n_train + n_validation :]
    dense_specs = tuple(spec for spec in arm_specs() if spec.family.startswith("omc"))
    sigmas = {q: bandwidth_from_quantile(work_distances, q) for q in SIGMA_QUANTILES}
    similarities = []
    for spec in dense_specs:
        distances = rewired_distances if spec.rewired else work_distances
        similarities.append(
            np.asarray(
                build_omc_kernel(
                    distances,
                    bandwidth=sigmas[spec.sigma_quantile],
                    metric="work_scale",
                ).similarity
            )
        )
    output: list[dict] = []
    for target_mass in target_masses:
        truth = basin_mass_target(basin.labels, basin.basin, target_mass)
        target = flat @ truth
        dense = _fit_batch(
            observables,
            truth,
            prior,
            train,
            validation,
            np.asarray(similarities),
            np.asarray([spec.normalise for spec in dense_specs]),
            strengths,
            steps,
        )
        for arm_index, spec in enumerate(dense_specs):
            for strength_index, strength in enumerate(strengths):
                output.append(
                    _row(
                        data=data,
                        target_mass=target_mass,
                        basin=basin,
                        spec=spec,
                        sigma=sigmas[spec.sigma_quantile],
                        strength=strength,
                        weights=dense["weights"][arm_index, strength_index],
                        flat=flat,
                        target=target,
                        train=train,
                        validation=validation,
                        test=test,
                        truth=truth,
                        structural=structural,
                        validation_mse=dense["validation_mse"][
                            arm_index, strength_index
                        ],
                        objective_midpoint=dense["objective_midpoint"][
                            arm_index, strength_index
                        ],
                        objective_final=dense["objective_final"][
                            arm_index, strength_index
                        ],
                        grad_norm=dense["grad_norm"][arm_index, strength_index],
                    )
                )
        comparator_specs = (
            ArmSpec("maxent", "maxent", None),
            ArmSpec("prior_rbf_locked", "prior_rbf_locked", 0.16),
        )
        locked_graph = build_all_pairs_graph_from_distances(
            work_distances,
            metric="work_scale__all_pairs_rbf__0p16",
            bandwidth=sigmas[0.16],
        )
        for spec in comparator_specs:
            regularizer = "maxent" if spec.family == "maxent" else "laplacian"
            graph = None if spec.family == "maxent" else locked_graph
            comparator = _fit_comparator_batch(
                observables,
                truth,
                prior,
                train,
                validation,
                strengths,
                steps,
                regularizer,
                graph,
            )
            for strength_index, strength in enumerate(strengths):
                output.append(
                    _row(
                        data=data,
                        target_mass=target_mass,
                        basin=basin,
                        spec=spec,
                        sigma=None if spec.family == "maxent" else sigmas[0.16],
                        strength=strength,
                        weights=comparator["weights"][strength_index],
                        flat=flat,
                        target=target,
                        train=train,
                        validation=validation,
                        test=test,
                        truth=truth,
                        structural=structural,
                        validation_mse=comparator["validation_mse"][strength_index],
                        objective_midpoint=comparator["objective_midpoint"][
                            strength_index
                        ],
                        objective_final=comparator["objective_final"][strength_index],
                        grad_norm=comparator["grad_norm"][strength_index],
                    )
                )
    return output


def _selected(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    block = frame[frame.arm_family == family]
    indices = block.groupby(["system_id", "target_mass"])["validation_mse"].idxmin()
    return block.loc[indices].set_index(["system_id", "target_mass"]).sort_index()


def _verdict(condition: bool | None, **statistics) -> dict:
    return {
        "verdict": "inconclusive"
        if condition is None
        else ("pass" if condition else "fail"),
        **statistics,
    }


def evaluate_gates(results: pd.DataFrame, seed: int = SEED) -> dict:
    """Evaluate preregistered gates and return explicit three-way verdicts."""
    boundary_fraction = float(
        _selected(results, "omc")
        .strength.isin((min(results.strength), max(results.strength)))
        .mean()
    )
    required_families = {"maxent", "omc", "omc_rewired"}
    if not required_families.issubset(set(results.arm_family)):
        reason = "strength grid boundary saturation and incomplete comparator table"
        return {
            **{
                key: _verdict(None, reason=reason)
                for key in (
                    "G1_recovery",
                    "G2_specificity",
                    "G3_non_inferiority",
                    "G4_diversity_control",
                )
            },
            "diagnostics": {"boundary_strength_fraction": boundary_fraction},
            "overall": "inconclusive",
        }
    maxent, omc = _selected(results, "maxent"), _selected(results, "omc")
    shared = maxent.index.intersection(omc.index)
    delta = (
        maxent.loc[shared, "target_mass_error"] - omc.loc[shared, "target_mass_error"]
    )
    systems = np.asarray([index[0] for index in shared])
    ci1 = clustered_median_bootstrap(delta.to_numpy(), systems, stable_seed(seed, "G1"))
    by_system = delta.groupby(level=0).median()
    nonzero = by_system[by_system != 0]
    sign_p = (
        float(binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue)
        if len(nonzero)
        else 1.0
    )
    g1 = bool(ci1[0] > 0 and sign_p <= 0.05)

    rewired = results[results.arm_family == "omc_rewired"]
    chosen = omc.reset_index()[["system_id", "target_mass", "sigma_quantile"]]
    matched = rewired.merge(chosen, on=["system_id", "target_mass", "sigma_quantile"])
    matched = matched.loc[
        matched.groupby(["system_id", "target_mass"])["validation_mse"].idxmin()
    ].set_index(["system_id", "target_mass"])
    specificity = (
        matched.loc[shared, "target_mass_error"] - omc.loc[shared, "target_mass_error"]
    )
    ci2 = clustered_median_bootstrap(
        specificity.to_numpy(), systems, stable_seed(seed, "G2")
    )
    g2 = bool(np.median(specificity) > 0 and ci2[0] > 0)
    relative_mse = (
        omc.loc[shared, "test_mse"] - maxent.loc[shared, "test_mse"]
    ) / np.maximum(maxent.loc[shared, "test_mse"], np.finfo(float).eps)
    median_relative_mse = float(np.median(relative_mse))
    g3 = median_relative_mse <= 0.01

    diversity_rows = []
    primary = results[results.arm_family == "omc"]
    for (system, mass, strength), block in primary.groupby(
        ["system_id", "target_mass", "strength"]
    ):
        reference = maxent.loc[(system, mass)]
        acceptable = block[
            (block.target_mass_error <= reference.target_mass_error)
            & (block.test_mse <= 1.01 * reference.test_mse)
        ].sort_values("sigma_quantile")
        if len(acceptable) < 3:
            continue
        for measure in ("ess_fraction", "target_ess_fraction", "structural_dispersion"):
            values = acceptable[measure].to_numpy()
            rho = (
                0.0
                if np.allclose(values, values[0])
                else float(spearmanr(acceptable.sigma_quantile, values).statistic)
            )
            ratio = float(np.max(values) / max(np.min(values), np.finfo(float).eps))
            diversity_rows.append(
                {
                    "system_id": system,
                    "target_mass": mass,
                    "strength": strength,
                    "measure": measure,
                    "abs_rho": abs(rho),
                    "ratio": ratio,
                }
            )
    diversity = pd.DataFrame(diversity_rows)
    if diversity.empty:
        g4_payload = _verdict(False, reason="fewer than three acceptable bandwidths")
        g4 = False
    else:
        ess = diversity[diversity.measure == "ess_fraction"]
        ci_rho = clustered_median_bootstrap(
            ess.abs_rho.to_numpy(), ess.system_id.to_numpy(), stable_seed(seed, "G4rho")
        )
        ci_ratio = clustered_median_bootstrap(
            ess.ratio.to_numpy(), ess.system_id.to_numpy(), stable_seed(seed, "G4ratio")
        )
        median_rho, median_ratio = (
            float(ess.abs_rho.median()),
            float(ess.ratio.median()),
        )
        g4 = bool(median_rho >= 0.7 and ci_rho[0] > 0.5 and median_ratio >= 2.0)
        g4_payload = _verdict(
            g4,
            median_abs_rho=median_rho,
            rho_ci=list(ci_rho),
            median_ess_ratio=median_ratio,
            ratio_ci=list(ci_ratio),
        )
    dense = results[results.arm_family.isin(("omc", "omc_rewired", "omc_norm"))]
    convergence_change = np.abs(
        dense.objective_step300 - dense.objective_step150
    ) / np.maximum(np.abs(dense.objective_step300), np.finfo(float).eps)
    headroom = maxent.target_mass_error.groupby(level=0).median()
    comparator_summary = {}
    for family in (
        "maxent",
        "omc",
        "omc_rewired",
        "omc_norm",
        "prior_rbf_locked",
    ):
        selected_family = _selected(results, family)
        comparator_summary[family] = {
            "median_target_mass_error": float(
                selected_family.target_mass_error.median()
            ),
            "median_recovery_percent": float(selected_family.recovery_percent.median()),
            "median_test_mse": float(selected_family.test_mse.median()),
        }
    gates = {
        "G1_recovery": _verdict(
            g1,
            median_delta=float(np.median(delta)),
            interval=list(ci1),
            sign_test_p=sign_p,
        ),
        "G2_specificity": _verdict(
            g2, median_difference=float(np.median(specificity)), interval=list(ci2)
        ),
        "G3_non_inferiority": _verdict(
            g3, median_relative_test_mse_change=median_relative_mse
        ),
        "G4_diversity_control": g4_payload,
        "G5_comparators": {
            "verdict": "descriptive",
            "validation_selected_summary": comparator_summary,
        },
        "G6_numerical_validity": _verdict(
            bool(
                results.simplex_valid.all()
                and np.isfinite(
                    results[
                        [
                            "train_mse",
                            "validation_mse",
                            "test_mse",
                            "final_grad_norm",
                            "objective_step150",
                            "objective_step300",
                        ]
                    ]
                )
                .all()
                .all()
            ),
            plateau_fraction=float(results.plateau_flag.mean()),
            support_size_median=float(results.support_size.median()),
        ),
        "diagnostics": {
            "boundary_strength_fraction": boundary_fraction,
            "median_relative_objective_change_step150_to_step300": float(
                np.median(convergence_change)
            ),
            "median_final_grad_norm": float(np.median(dense.final_grad_norm)),
            "median_maxent_target_mass_error_by_system": {
                str(system): float(value) for system, value in headroom.items()
            },
        },
    }
    gated = (
        "G1_recovery",
        "G2_specificity",
        "G3_non_inferiority",
        "G4_diversity_control",
    )
    if boundary_fraction > 0.25:
        for key in gated:
            gates[key]["raw_verdict"] = gates[key]["verdict"]
            gates[key]["verdict"] = "inconclusive"
            gates[key]["reason"] = "strength grid boundary saturation"
        gates["overall"] = "inconclusive"
    else:
        gates["overall"] = (
            "pass" if all(gates[key]["verdict"] == "pass" for key in gated) else "fail"
        )
    return gates


def _plots(results: pd.DataFrame, destination: Path) -> None:
    selected = _selected(results, "omc")
    maxent = _selected(results, "maxent")
    masses = tuple(sorted(results.target_mass.unique()))
    fig, axes = plt.subplots(1, len(masses), figsize=(4 * len(masses), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, mass in zip(axes, masses, strict=True):
        block = results[(results.arm_family == "omc") & (results.target_mass == mass)]
        for system, values in block.groupby("system_id"):
            chosen_strength = selected.loc[(system, mass)].strength
            values = values[values.strength == chosen_strength].sort_values(
                "sigma_quantile"
            )
            (line,) = axis.plot(
                values.sigma_quantile,
                values.ess_fraction,
                marker="o",
                alpha=0.6,
                label=system,
            )
            axis.plot(
                values.sigma_quantile,
                values.target_ess_fraction,
                linestyle="--",
                alpha=0.5,
                color=line.get_color(),
            )
            reference = maxent.loc[(system, mass)]
            acceptable = values[
                (values.target_mass_error <= reference.target_mass_error)
                & (values.test_mse <= 1.01 * reference.test_mse)
            ]
            axis.scatter(
                acceptable.sigma_quantile,
                acceptable.ess_fraction,
                facecolors="none",
                edgecolors=line.get_color(),
                s=70,
            )
        axis.set_title(f"target mass {mass:g}")
        axis.set_xlabel("sigma quantile")
    axes[0].set_ylabel("ESS fraction")
    fig.tight_layout()
    fig.savefig(destination / "sigma_vs_ess.png", dpi=160)
    plt.close(fig)
    shared = selected.index.intersection(maxent.index)
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(
        maxent.loc[shared].target_mass_error, selected.loc[shared].target_mass_error
    )
    maximum = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot([0, maximum], [0, maximum], "k--")
    axis.set(xlabel="MaxEnt target-mass error", ylabel="selected OMC target-mass error")
    fig.tight_layout()
    fig.savefig(destination / "recovery_vs_maxent.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    selected.strength.hist(ax=axis, bins=len(STRENGTHS))
    axis.axvline(min(STRENGTHS), color="tab:red", linestyle="--")
    axis.axvline(max(STRENGTHS), color="tab:red", linestyle="--")
    axis.set(xlabel="selected strength", ylabel="count")
    fig.tight_layout()
    fig.savefig(destination / "strength_selection.png", dpi=160)
    plt.close(fig)


def convergence_sensitivity(results: pd.DataFrame, sensitivity: pd.DataFrame) -> dict:
    """Compare selected configurations on systems present in a longer run."""
    systems = sorted(set(results.system_id) & set(sensitivity.system_id))
    baseline = _selected(results[results.system_id.isin(systems)], "omc")
    longer = _selected(sensitivity[sensitivity.system_id.isin(systems)], "omc")
    shared = baseline.index.intersection(longer.index)
    same_configuration = bool(
        np.array_equal(
            baseline.loc[shared].strength.to_numpy(),
            longer.loc[shared].strength.to_numpy(),
        )
        and np.array_equal(
            baseline.loc[shared].sigma_quantile.to_numpy(),
            longer.loc[shared].sigma_quantile.to_numpy(),
        )
    )
    return {
        "systems": systems,
        "challenges": len(shared),
        "selected_configuration_unchanged": same_configuration,
        "median_target_mass_error_step300": float(
            baseline.loc[shared].target_mass_error.median()
        ),
        "median_target_mass_error_step1000": float(
            longer.loc[shared].target_mass_error.median()
        ),
        "interpretation": (
            "checkpoint verdict unchanged; longer fitting improves the selected unregularised fit"
            if same_configuration
            else "selected configuration changed; 300 optimizer steps are inadequate"
        ),
    }


def _run_screen(
    rows: list[dict], config: dict, destination: Path, frame_cap: int
) -> tuple[pd.DataFrame, list[str], tuple[float, float]]:
    records = []
    for index, row in enumerate(rows, 1):
        records.append(screen_system(row, config, frame_cap))
        print(f"[screen {index}/{len(rows)}] {row['system_id']}", flush=True)
    screen = pd.DataFrame(records)
    band = (0.06, 0.15)
    cohort = select_cohort(screen)
    if len(cohort) < 6:
        records = []
        band = (0.05, 0.20)
        for index, row in enumerate(rows, 1):
            records.append(screen_system(row, config, frame_cap, band))
            print(f"[screen-wide {index}/{len(rows)}] {row['system_id']}", flush=True)
        screen = pd.DataFrame(records)
        cohort = select_cohort(screen)
    atomic_parquet(
        screen[screen.eligible].reset_index(drop=True), destination / "screen.parquet"
    )
    atomic_parquet(
        screen[~screen.eligible].reset_index(drop=True),
        destination / "exclusions.parquet",
    )
    atomic_yaml(
        destination / "cohort.yaml",
        {"mass_band": list(band), "systems": cohort, "inconclusive": len(cohort) < 6},
    )
    return screen, cohort, band


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "pilot"), default="pilot")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--frame-cap", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--strengths", default=",".join(map(str, STRENGTHS)))
    parser.add_argument("--target-masses", default=",".join(map(str, TARGET_MASSES)))
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else args.phase
    if args.output_suffix:
        mode = f"{mode}_{args.output_suffix}"
    destination = OUTPUT / mode
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_rows(False, args.limit)
    config = load_config()
    if args.phase == "screen" and not args.smoke:
        _run_screen(rows, config, destination, args.frame_cap)
        return
    screen_dir = OUTPUT / "screen"
    cohort_path = screen_dir / "cohort.yaml"
    if args.smoke:
        cohort = [row["system_id"] for row in rows]
        band = (0.10, 0.35) if args.frame_cap < 80 else (0.06, 0.15)
    elif cohort_path.exists():
        import yaml

        cohort_payload = yaml.safe_load(cohort_path.read_text())
        cohort, band = cohort_payload["systems"], tuple(cohort_payload["mass_band"])
    else:
        _, cohort, band = _run_screen(
            load_rows(False, None), config, screen_dir, args.frame_cap
        )
    if len(cohort) < 6 and not args.smoke:
        atomic_yaml(
            destination / "checkpoint35_report.yaml",
            {"overall": "inconclusive", "reason": "fewer than 6 eligible systems"},
        )
        return
    selected_rows = [
        row for row in load_rows(False, None) if row["system_id"] in set(cohort)
    ]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]
    strengths = tuple(float(value) for value in args.strengths.split(","))
    target_masses = tuple(float(value) for value in args.target_masses.split(","))
    parts = destination / "parts"
    parts.mkdir(exist_ok=True)
    for index, row in enumerate(selected_rows, 1):
        path = parts / f"{row['system_id']}.parquet"
        if not path.exists():
            records = reweight_system(
                row,
                config,
                target_masses,
                strengths,
                args.steps,
                args.frame_cap,
                band,
                rare_min_frames=2 if args.smoke and args.frame_cap < 80 else 8,
            )
            atomic_parquet(pd.DataFrame(records), path)
        print(f"[pilot {index}/{len(selected_rows)}] {row['system_id']}", flush=True)
    result_parts = [
        pd.read_parquet(parts / f"{row['system_id']}.parquet") for row in selected_rows
    ]
    results = pd.concat(result_parts, ignore_index=True)
    atomic_parquet(results, destination / "results.parquet")
    if (
        args.smoke
        or len(selected_rows) < 6
        or set(target_masses) != set(TARGET_MASSES)
        or set(strengths) != set(STRENGTHS)
        or args.steps != 300
        or args.frame_cap != 128
    ):
        report = {
            "overall": "inconclusive",
            "reason": "non-preregistered smoke or limited run",
            "rows": len(results),
        }
    else:
        report = evaluate_gates(results)
        sensitivity_path = OUTPUT / "pilot_steps1000" / "results.parquet"
        if sensitivity_path.exists():
            report["diagnostics"]["convergence_sensitivity"] = convergence_sensitivity(
                results, pd.read_parquet(sensitivity_path)
            )
    atomic_yaml(destination / "checkpoint35_report.yaml", report)
    _plots(results, destination)


if __name__ == "__main__":
    main()
