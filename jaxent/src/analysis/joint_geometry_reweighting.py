"""Generic joint frame-reweighting/BV fitting against frozen covariance priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxent.src.analysis.joint_covariance_geometry import (
    family_joint_geometry_loss,
    fixed_joint_geometry_loss,
)
from jaxent.src.analysis.pf_variance import (
    covariance_profiles_from_covariance,
    kl_to_uniform,
    uptake_from_log_pf,
    weighted_population_covariance,
    weighted_variance_log_ratio_loss,
)
from jaxent.src.analysis.state_population import correlation_shape_loss


@dataclass(frozen=True)
class FrozenPeptidePrior:
    """Arrays defining one fixed point, family, or correlation-shape prior."""

    kind: str
    geometry: Any | None = None
    geometry_modes: Any | None = None
    relative_variance: Any | None = None
    marginal_modes: Any | None = None
    score_precision: Any | None = None
    correlation: Any | None = None
    profile: Any | None = None
    profile_weights: Any | None = None


@dataclass(frozen=True)
class ReweightingCell:
    """One candidate structural ensemble in a shared joint fit."""

    name: str
    heavy_contacts: Any
    acceptor_contacts: Any
    k_ints: Any
    timepoints: Any
    mapping: Any
    observed_uptake: Any  # peptide x time
    train_time_indices: Any
    projection: Any
    marginal_weights: Any
    mean_reference: float
    prior: FrozenPeptidePrior


def coefficients_from_theta(theta):
    theta = jnp.asarray(theta)
    return jax.nn.softplus(theta[0]), jax.nn.softplus(theta[1])


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def predict_uptake(log_pf, k_ints, timepoints, mapping, weights):
    mean_log_pf = jnp.asarray(log_pf) @ jnp.asarray(weights)
    residue_uptake = uptake_from_log_pf(mean_log_pf, k_ints, timepoints)
    return residue_uptake @ jnp.asarray(mapping).T


def peptide_logpf_covariance(log_pf, mapping, weights):
    peptide_log_pf = jnp.asarray(mapping) @ jnp.asarray(log_pf)
    return weighted_population_covariance(peptide_log_pf, weights)


def _cell_prior_components(
    cell: ReweightingCell,
    covariance,
    scores,
    marginal_strength: float,
    alpha: float,
):
    prior = cell.prior
    if prior.kind == "mean_only":
        zero = jnp.asarray(0.0, dtype=covariance.dtype)
        return zero, zero, zero
    if prior.kind == "shape":
        shape = correlation_shape_loss(covariance, prior.correlation, cell.projection, alpha)
        zero = jnp.asarray(0.0, dtype=shape.dtype)
        return shape, shape, zero
    if prior.kind in {"marginal_oracle", "conditional_oracle"}:
        profile_index = 1 if prior.kind == "marginal_oracle" else 3
        candidate_profile = covariance_profiles_from_covariance(covariance, alpha=alpha)[
            profile_index
        ]
        profile_loss = weighted_variance_log_ratio_loss(
            candidate_profile, prior.profile, prior.profile_weights
        )
        zero = jnp.asarray(0.0, dtype=profile_loss.dtype)
        if prior.kind == "marginal_oracle":
            return profile_loss, zero, profile_loss
        return profile_loss, profile_loss, zero
    if prior.kind in {"fixed", "point", "unlearned", "oracle"}:
        return fixed_joint_geometry_loss(
            covariance,
            prior_geometry=prior.geometry,
            prior_relative_variance=prior.relative_variance,
            projection=cell.projection,
            marginal_weights=cell.marginal_weights,
            marginal_strength=marginal_strength,
            alpha=alpha,
        )
    if prior.kind == "family":
        total, geometry, marginal, _ = family_joint_geometry_loss(
            covariance,
            scores,
            prior_geometry=prior.geometry,
            geometry_modes=prior.geometry_modes,
            prior_relative_variance=prior.relative_variance,
            marginal_modes=prior.marginal_modes,
            score_precision=prior.score_precision,
            projection=cell.projection,
            marginal_weights=cell.marginal_weights,
            marginal_strength=marginal_strength,
            score_strength=0.0,
            alpha=alpha,
        )
        return total, geometry, marginal
    raise ValueError(f"unknown frozen prior kind {prior.kind!r}")


def build_joint_objective(
    cells: dict[str, ReweightingCell],
    *,
    marginal_strength: float = 1.0,
    kl_strength: float = 0.01,
    score_strength: float = 1.0,
    alpha: float = 0.05,
):
    """Return a differentiable objective and component evaluator for fixed cells."""

    names = tuple(cells)
    rank = 0
    score_precision = None
    for cell in cells.values():
        if cell.prior.kind == "family":
            current_rank = int(np.asarray(cell.prior.geometry_modes).shape[0])
            if rank and current_rank != rank:
                raise ValueError("all family priors in a joint fit must share one rank")
            rank = current_rank
            score_precision = jnp.asarray(cell.prior.score_precision)

    def components(params):
        bc, bh = coefficients_from_theta(params["theta"])
        scores = params["scores"]
        total = jnp.asarray(0.0)
        mean_total = jnp.asarray(0.0)
        geometry_total = jnp.asarray(0.0)
        marginal_total = jnp.asarray(0.0)
        kl_total = jnp.asarray(0.0)
        for name in names:
            cell = cells[name]
            weights = jax.nn.softmax(params["logits"][name])
            log_pf = bc * jnp.asarray(cell.heavy_contacts) + bh * jnp.asarray(cell.acceptor_contacts)
            prediction = predict_uptake(
                log_pf, cell.k_ints, cell.timepoints, cell.mapping, weights
            ).T
            indices = jnp.asarray(cell.train_time_indices)
            mean = jnp.mean(
                jnp.square(prediction[:, indices] - jnp.asarray(cell.observed_uptake)[:, indices])
            ) / jnp.maximum(jnp.asarray(cell.mean_reference), 1e-12)
            covariance = peptide_logpf_covariance(log_pf, cell.mapping, weights)
            prior_total, geometry, marginal = _cell_prior_components(
                cell, covariance, scores, marginal_strength, alpha
            )
            kl = kl_to_uniform(weights)
            total = total + mean + prior_total + kl_strength * kl
            mean_total = mean_total + mean
            geometry_total = geometry_total + geometry
            marginal_total = marginal_total + marginal
            kl_total = kl_total + kl
        score_loss = jnp.asarray(0.0)
        if rank:
            score_loss = scores @ score_precision @ scores / rank
            total = total + score_strength * score_loss
        return total, (mean_total, geometry_total, marginal_total, kl_total, score_loss)

    return components, rank


def optimize_joint_reweighting(
    cells: dict[str, ReweightingCell],
    *,
    marginal_strength: float = 1.0,
    kl_strength: float = 0.01,
    score_strength: float = 1.0,
    alpha: float = 0.05,
    steps: int = 2000,
    learning_rate: float = 0.03,
    starts: int = 5,
    seed: int = 100,
    reference_bc: float = 0.35,
    reference_bh: float = 2.0,
) -> dict[str, Any]:
    """Fit per-ensemble weights and shared BV coefficients against frozen priors."""

    components, rank = build_joint_objective(
        cells,
        marginal_strength=marginal_strength,
        kl_strength=kl_strength,
        score_strength=score_strength,
        alpha=alpha,
    )
    value_and_grad = jax.jit(jax.value_and_grad(lambda params: components(params)[0]))
    best = None
    for start in range(starts):
        rng = np.random.default_rng(seed + start)
        theta = np.asarray([inverse_softplus(reference_bc), inverse_softplus(reference_bh)])
        if start:
            theta += rng.normal(scale=0.1, size=2)
        params = {
            "theta": jnp.asarray(theta),
            "logits": {
                name: jnp.asarray(
                    np.zeros(np.asarray(cell.heavy_contacts).shape[1])
                    if start == 0
                    else rng.normal(scale=0.01, size=np.asarray(cell.heavy_contacts).shape[1])
                )
                for name, cell in cells.items()
            },
            "scores": jnp.zeros(rank),
        }
        optimizer = optax.adam(learning_rate)
        state = optimizer.init(params)
        finite = True
        for _ in range(steps):
            value, gradient = value_and_grad(params)
            if not np.isfinite(float(value)):
                finite = False
                break
            updates, state = optimizer.update(gradient, state, params)
            params = optax.apply_updates(params, updates)
        value, aux = components(params)
        objective = float(value)
        finite = finite and np.isfinite(objective)
        if finite and (best is None or objective < best[0]):
            best = (objective, start, params, aux)
    if best is None:
        raise FloatingPointError("all joint reweighting starts were non-finite")
    objective, start, params, aux = best
    bc, bh = (float(value) for value in coefficients_from_theta(params["theta"]))
    return {
        "objective": objective,
        "best_start": int(start),
        "bc": bc,
        "bh": bh,
        "scores": np.asarray(params["scores"]),
        "weights": {name: np.asarray(jax.nn.softmax(value)) for name, value in params["logits"].items()},
        "components": {
            key: float(value)
            for key, value in zip(
                ("mean", "geometry", "marginal", "kl", "score"), aux, strict=True
            )
        },
    }
