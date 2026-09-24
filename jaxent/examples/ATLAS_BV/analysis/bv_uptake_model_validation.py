"""Validate tractable BV uptake models against exact frame-wise uptake.

TeaA uses the independently featurised open/closed trajectories.  ATLAS uses a
deterministically frozen 12-system subset and treats the three independent
replicas as known population states.  Targets are always generated with exact
frame-wise EX2 uptake; approximations never generate their own targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.spatial.distance import jensenshannon


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
TEAA_FEATURES = (
    ROOT
    / "jaxent/examples/1_IsoValidation_OMass/data/_self_consistent_target_features"
    / "features_open_closed.npz"
)
TEAA_MANIFEST = TEAA_FEATURES.with_name("manifest.json")
DEFAULT_OUTPUT = HERE / "outputs/analysis/bv_uptake_validation"
MODELS = (
    "exact",
    "linear_bv_default",
    "linear_bv",
    "gamma_moments_default",
    "gamma_moments",
    "mixture_q4_default",
    "mixture_q4",
)
TEAA_POPULATIONS = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)
ATLAS_POPULATION = np.asarray([0.60, 0.30, 0.10])
NOISE_LEVELS = (0.0, 0.01)
NOISE_SEEDS = (17, 29, 43, 71, 101)
EPS = np.finfo(np.float64).tiny
jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class EnsembleData:
    system_id: str
    heavy: np.ndarray
    acceptor: np.ndarray
    log_kint: np.ndarray
    states: np.ndarray
    state_names: tuple[str, ...]
    times: np.ndarray
    metadata: dict[str, object]

    @property
    def z(self) -> np.ndarray:
        return 0.35 * self.heavy + 2.0 * self.acceptor


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        heavy = np.asarray(data["heavy_contacts"], dtype=np.float64)
        acceptor = np.asarray(data["acceptor_contacts"], dtype=np.float64)
        kint = np.asarray(data["k_ints"], dtype=np.float64)
    if (
        heavy.shape != acceptor.shape
        or heavy.ndim != 2
        or kint.shape != (heavy.shape[0],)
    ):
        raise ValueError(
            f"invalid BV features in {path}: {heavy.shape}, {acceptor.shape}, {kint.shape}"
        )
    if (
        not np.isfinite(heavy).all()
        or not np.isfinite(acceptor).all()
        or np.any(kint <= 0)
    ):
        raise ValueError(
            f"non-finite contacts or non-positive intrinsic rates in {path}"
        )
    return heavy, acceptor, kint


def stable_uptake_from_log_exposure(log_exposure: np.ndarray) -> np.ndarray:
    result = np.empty_like(log_exposure, dtype=np.float64)
    high = log_exposure >= 35.0
    low = log_exposure <= -36.0
    middle = ~(high | low)
    result[high] = 1.0
    result[low] = np.exp(log_exposure[low])
    result[middle] = -np.expm1(-np.exp(log_exposure[middle]))
    return result


def state_weights(states: np.ndarray, population: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(states), dtype=np.float64)
    for state, mass in enumerate(population):
        mask = states == state
        if not mask.any():
            raise ValueError(f"state {state} has no frames")
        weights[mask] = mass / mask.sum()
    return weights


def exact_uptake(data: EnsembleData, population: np.ndarray) -> np.ndarray:
    weights = state_weights(data.states, population)
    log_exposure = (
        np.log(data.times)[:, None, None]
        + data.log_kint[None, :, None]
        - data.z[None, :, :]
    )
    return np.einsum(
        "trf,f->tr", stable_uptake_from_log_exposure(log_exposure), weights
    )


def exact_state_curves(data: EnsembleData) -> np.ndarray:
    curves = []
    for state in range(len(data.state_names)):
        population = np.zeros(len(data.state_names))
        population[state] = 1.0
        curves.append(exact_uptake(data, population))
    return np.stack(curves)


def ordered_supports(z: np.ndarray, count: int) -> np.ndarray:
    support = np.maximum(np.quantile(z, np.linspace(0.0, 1.0, count)), 0.0)
    for index in range(1, count):
        support[index] = max(support[index], support[index - 1] + 1e-6)
    return support


def mixture_components(data: EnsembleData, count: int) -> tuple[np.ndarray, np.ndarray]:
    support = ordered_supports(data.z, count)
    tau = max(0.5 * float(np.median(np.diff(support))), 0.05)
    logits = -0.5 * ((data.z[..., None] - support) / tau) ** 2
    logits -= logits.max(axis=-1, keepdims=True)
    assignments = np.exp(logits)
    assignments /= assignments.sum(axis=-1, keepdims=True)
    state_masses = []
    for state in range(len(data.state_names)):
        state_masses.append(assignments[:, data.states == state].mean(axis=1))
    component_uptake = stable_uptake_from_log_exposure(
        np.log(data.times)[:, None, None]
        + data.log_kint[None, :, None]
        - support[None, None, :]
    )
    return np.stack(state_masses), component_uptake


def state_rate_moments(
    data: EnsembleData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Scaling each residue by its maximum log-rate keeps moments well-conditioned.
    log_rate = data.log_kint[:, None] - data.z
    scale_log = log_rate.max(axis=1)
    scaled_rate = np.exp(log_rate - scale_log[:, None])
    first, second, mean_z = [], [], []
    for state in range(len(data.state_names)):
        mask = data.states == state
        first.append(scaled_rate[:, mask].mean(axis=1))
        second.append(np.square(scaled_rate[:, mask]).mean(axis=1))
        mean_z.append(data.z[:, mask].mean(axis=1))
    return np.stack(first), np.stack(second), np.stack(mean_z), scale_log


class ApproximationFamily:
    def __init__(self, data: EnsembleData):
        self.data = data
        self.exact_curves = exact_state_curves(data)
        self.first, self.second, self.mean_z, self.scale_log = state_rate_moments(data)
        self.mean_heavy = np.stack(
            [
                data.heavy[:, data.states == state].mean(axis=1)
                for state in range(len(data.state_names))
            ]
        )
        self.mean_acceptor = np.stack(
            [
                data.acceptor[:, data.states == state].mean(axis=1)
                for state in range(len(data.state_names))
            ]
        )
        self.mixtures = {4: mixture_components(data, 4)}

    def predict(
        self,
        model: str,
        population: np.ndarray,
        fitted_parameters: np.ndarray | None = None,
    ) -> np.ndarray:
        base_model = model.removesuffix("_default")
        if base_model == "exact":
            return np.einsum("s,str->tr", population, self.exact_curves)
        if base_model == "linear_bv":
            if fitted_parameters is None:
                beta_c, beta_h = 0.35, 2.0
                offsets = np.zeros(len(self.data.times))
            else:
                beta_c, beta_h = np.logaddexp(0.0, fitted_parameters[:2])
                offsets = fitted_parameters[2:]
            mean_heavy = np.einsum("s,sr->r", population, self.mean_heavy)
            mean_acceptor = np.einsum("s,sr->r", population, self.mean_acceptor)
            z_bar = beta_c * mean_heavy + beta_h * mean_acceptor
            intervals = np.diff(np.concatenate(([0.0], self.data.times)))
            effective_times = np.cumsum(intervals * np.exp(offsets))
            return stable_uptake_from_log_exposure(
                np.log(effective_times)[:, None] + self.data.log_kint[None, :] - z_bar
            )
        if base_model.startswith("mixture_q"):
            count = int(base_model.removeprefix("mixture_q"))
            if fitted_parameters is not None:
                beta_c, beta_h = np.logaddexp(0.0, fitted_parameters[:2])
                support = np.cumsum(np.logaddexp(0.0, fitted_parameters[2:]))
                z = beta_c * self.data.heavy + beta_h * self.data.acceptor
                tau = max(0.5 * float(np.median(np.diff(support))), 0.05)
                logits = -0.5 * ((z[..., None] - support) / tau) ** 2
                logits -= logits.max(axis=-1, keepdims=True)
                assignments = np.exp(logits)
                assignments /= assignments.sum(axis=-1, keepdims=True)
                masses = []
                for state in range(len(self.data.state_names)):
                    masses.append(
                        assignments[:, self.data.states == state].mean(axis=1)
                    )
                mixed_masses = np.einsum("s,srq->rq", population, np.stack(masses))
                curves = stable_uptake_from_log_exposure(
                    np.log(self.data.times)[:, None, None]
                    + self.data.log_kint[None, :, None]
                    - support[None, None, :]
                )
                return np.einsum("rq,trq->tr", mixed_masses, curves)
            masses, curves = self.mixtures[count]
            mixed_masses = np.einsum("s,srq->rq", population, masses)
            return np.einsum("rq,trq->tr", mixed_masses, curves)

        if fitted_parameters is None:
            first, second, scale_log = self.first, self.second, self.scale_log
        else:
            beta_c, beta_h = np.logaddexp(0.0, fitted_parameters[:2])
            log_rate = (
                self.data.log_kint[:, None]
                - beta_c * self.data.heavy
                - beta_h * self.data.acceptor
            )
            scale_log = log_rate.max(axis=1)
            scaled_rate = np.exp(log_rate - scale_log[:, None])
            first, second = [], []
            for state in range(len(self.data.state_names)):
                mask = self.data.states == state
                first.append(scaled_rate[:, mask].mean(axis=1))
                second.append(np.square(scaled_rate[:, mask]).mean(axis=1))
            first, second = np.stack(first), np.stack(second)
        mean_scaled = np.einsum("s,sr->r", population, first)
        log_mean_rate = scale_log + np.log(np.maximum(mean_scaled, EPS))
        if base_model == "mean_rate":
            return stable_uptake_from_log_exposure(
                np.log(self.data.times)[:, None] + log_mean_rate[None, :]
            )
        if base_model != "gamma_moments":
            raise ValueError(f"unknown model: {model}")
        second_scaled = np.einsum("s,sr->r", population, second)
        variance_scaled = np.maximum(second_scaled - mean_scaled**2, 0.0)
        relative_variance = variance_scaled / np.maximum(mean_scaled**2, EPS)
        shape = 1.0 / np.maximum(relative_variance, EPS)
        log_argument = (
            np.log(self.data.times)[:, None]
            + scale_log[None, :]
            + np.log(np.maximum(variance_scaled / np.maximum(mean_scaled, EPS), EPS))[
                None, :
            ]
        )
        log_one_plus = np.logaddexp(0.0, log_argument)
        gamma = -np.expm1(-shape[None, :] * log_one_plus)
        exponential = stable_uptake_from_log_exposure(
            np.log(self.data.times)[:, None] + log_mean_rate[None, :]
        )
        return np.where(relative_variance[None, :] <= 1e-12, exponential, gamma)


def peptide_windows(values: np.ndarray, width: int = 10) -> np.ndarray:
    """Aggregate non-overlapping residue blocks into peptide-like mean uptake."""
    blocks = []
    for start in range(0, values.shape[1], width):
        stop = min(start + width, values.shape[1])
        if stop - start >= 3:
            blocks.append(values[:, start:stop].mean(axis=1))
    if len(blocks) < 2:
        raise ValueError("fewer than two peptide-like windows")
    return np.stack(blocks, axis=1)


def fit_population(
    family: ApproximationFamily,
    model: str,
    target: np.ndarray,
    train_peptides: np.ndarray,
    fitted_parameters: np.ndarray | None = None,
    state_curve_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    states = len(family.data.state_names)

    def objective(population):
        prediction = (
            np.einsum("s,str->tr", population, state_curve_basis)
            if state_curve_basis is not None
            else family.predict(model, population, fitted_parameters)
        )
        prediction = peptide_windows(prediction)
        residual = prediction[:, train_peptides] - target[:, train_peptides]
        return float(np.mean(residual**2))

    result = minimize(
        objective,
        np.full(states, 1.0 / states),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * states,
        constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    population = np.clip(result.x, 0.0, 1.0)
    population /= population.sum()
    return population, bool(result.success)


def inverse_softplus(value):
    value = np.maximum(np.asarray(value, dtype=np.float64), 1e-8)
    result = value + np.log(-np.expm1(-value))
    return float(result) if result.ndim == 0 else result


def fit_linear_parameters(
    family: ApproximationFamily,
    population: np.ndarray,
    target: np.ndarray,
    train_peptides: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Calibrate both BV slopes and all additive interval-rate offsets."""
    initial = np.concatenate(
        (
            [inverse_softplus(0.35), inverse_softplus(2.0)],
            np.zeros(len(family.data.times)),
        )
    )

    def residuals(parameters):
        prediction = peptide_windows(
            family.predict("linear_bv", population, parameters)
        )
        residual = prediction[:, train_peptides] - target[:, train_peptides]
        # A weak gauge-fixing prior prevents beta/rate trade-offs from drifting.
        prior = 1e-4 * parameters[2:]
        return np.concatenate((residual.ravel(), prior))

    result = least_squares(
        residuals,
        initial,
        bounds=(
            np.full(initial.shape, -12.0),
            np.full(initial.shape, 12.0),
        ),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    return np.asarray(result.x), bool(result.success)


def subsample_frames(data: EnsembleData, frames_per_state: int = 32) -> EnsembleData:
    """Deterministically retain equal-resolution support from every state."""
    indices = []
    for state in range(len(data.state_names)):
        available = np.flatnonzero(data.states == state)
        take = min(frames_per_state, len(available))
        indices.extend(available[np.linspace(0, len(available) - 1, take, dtype=int)])
    indices = np.asarray(indices)
    return EnsembleData(
        system_id=data.system_id,
        heavy=data.heavy[:, indices],
        acceptor=data.acceptor[:, indices],
        log_kint=data.log_kint,
        states=data.states[indices],
        state_names=data.state_names,
        times=data.times,
        metadata=data.metadata,
    )


def fit_distribution_parameters(
    family: ApproximationFamily,
    model: str,
    population: np.ndarray,
    target: np.ndarray,
    train_peptides: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Calibrate Gamma slopes or soft-mixture slopes and ordered supports."""
    beta_initial = np.asarray([inverse_softplus(0.35), inverse_softplus(2.0)])
    if model == "gamma_moments":
        initial = beta_initial
    else:
        count = int(model.removeprefix("mixture_q"))
        support = ordered_supports(family.data.z, count)
        initial = np.concatenate(
            (
                beta_initial,
                inverse_softplus(np.concatenate((support[:1], np.diff(support)))),
            )
        )

    data = family.data
    weights = jnp.asarray(state_weights(data.states, population))
    heavy = jnp.asarray(data.heavy)
    acceptor = jnp.asarray(data.acceptor)
    log_kint = jnp.asarray(data.log_kint)
    times = jnp.asarray(data.times)
    target_jax = jnp.asarray(target[:, train_peptides])
    ranges = [
        (start, min(start + 10, data.z.shape[0]))
        for start in range(0, data.z.shape[0], 10)
        if min(start + 10, data.z.shape[0]) - start >= 3
    ]
    peptide_matrix = np.zeros((data.z.shape[0], len(ranges)))
    for block, (start, stop) in enumerate(ranges):
        peptide_matrix[start:stop, block] = 1.0 / (stop - start)
    peptide_matrix = jnp.asarray(peptide_matrix[:, train_peptides])
    initial_jax = jnp.asarray(initial)

    def objective(parameters):
        beta_c, beta_h = jax.nn.softplus(parameters[:2])
        z = beta_c * heavy + beta_h * acceptor
        if model == "gamma_moments":
            log_rate = log_kint[:, None] - z
            scale_log = jnp.max(log_rate, axis=1)
            scaled_rate = jnp.exp(log_rate - scale_log[:, None])
            mean_scaled = scaled_rate @ weights
            second_scaled = jnp.square(scaled_rate) @ weights
            variance_scaled = jnp.maximum(second_scaled - mean_scaled**2, 0.0)
            mean_safe = jnp.maximum(mean_scaled, jnp.finfo(jnp.float64).tiny)
            relative_variance = variance_scaled / mean_safe**2
            shape = 1.0 / jnp.maximum(relative_variance, jnp.finfo(jnp.float64).tiny)
            log_argument = (
                jnp.log(times)[:, None]
                + scale_log[None, :]
                + jnp.log(
                    jnp.maximum(
                        variance_scaled / mean_safe, jnp.finfo(jnp.float64).tiny
                    )
                )[None, :]
            )
            gamma_uptake = -jnp.expm1(
                -shape[None, :] * jnp.logaddexp(0.0, log_argument)
            )
            log_exposure = (
                jnp.log(times)[:, None]
                + scale_log[None, :]
                + jnp.log(mean_safe)[None, :]
            )
            exponential_uptake = -jnp.expm1(-jnp.exp(jnp.minimum(log_exposure, 40.0)))
            prediction = jnp.where(
                relative_variance[None, :] <= jnp.sqrt(jnp.finfo(jnp.float64).eps),
                exponential_uptake,
                gamma_uptake,
            )
        else:
            support = jnp.cumsum(jax.nn.softplus(parameters[2:]))
            tau = jnp.maximum(0.5 * jnp.median(jnp.diff(support)), 0.05)
            assignments = jax.nn.softmax(
                -0.5 * ((z[..., None] - support) / tau) ** 2, axis=-1
            )
            masses = jnp.einsum("rfq,f->rq", assignments, weights)
            component_uptake = -jnp.expm1(
                -times[:, None, None]
                * jnp.exp(log_kint[None, :, None] - support[None, None, :])
            )
            prediction = jnp.einsum("rq,trq->tr", masses, component_uptake)
        peptide_prediction = prediction @ peptide_matrix
        residual = peptide_prediction - target_jax
        return jnp.mean(residual**2) + 1e-10 * jnp.mean((parameters - initial_jax) ** 2)

    value_and_gradient = jax.jit(jax.value_and_grad(objective))

    def scipy_objective(parameters):
        value, gradient = value_and_gradient(jnp.asarray(parameters))
        return float(value), np.asarray(gradient, dtype=np.float64)

    upper = np.full(initial.shape, 12.0)
    lower = np.full(initial.shape, -12.0)
    if model.startswith("mixture_q"):
        upper[2:] = 700.0
        lower[2:] = -30.0
    result = minimize(
        scipy_objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=list(zip(lower, upper)),
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 300},
    )
    return np.asarray(result.x), bool(result.success)


def calibrate_models(
    data: EnsembleData,
    population: np.ndarray,
    target_uptake: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, bool]]:
    """Fit every candidate once to an exact frame-wise target for one system."""
    family = ApproximationFamily(data)
    calibration_family = ApproximationFamily(subsample_frames(data))
    target = peptide_windows(
        family.predict("exact", population)
        if target_uptake is None
        else target_uptake
    )
    train = np.arange(target.shape[1]) % 2 == 0
    calibrated = {}
    print(f"  calibrating linear_bv for {data.system_id}", flush=True)
    calibrated["linear_bv"] = fit_linear_parameters(family, population, target, train)
    for model in ("gamma_moments", "mixture_q4"):
        print(f"  calibrating {model} for {data.system_id}", flush=True)
        calibrated[model] = fit_distribution_parameters(
            calibration_family, model, population, target, train
        )
    return calibrated


def evaluate_case(
    data: EnsembleData,
    true_population: np.ndarray,
    cohort: str,
    calibrated_parameters: dict[str, tuple[np.ndarray, bool]] | None = None,
    target_uptake: np.ndarray | None = None,
) -> list[dict[str, object]]:
    family = ApproximationFamily(data)
    calibration_family = ApproximationFamily(subsample_frames(data))
    exact = (
        family.predict("exact", true_population)
        if target_uptake is None
        else np.asarray(target_uptake)
    )
    exact_peptide = peptide_windows(exact)
    peptide_index = np.arange(exact_peptide.shape[1])
    train = peptide_index % 2 == 0
    test = ~train
    if not test.any():
        test = train.copy()
    records = []
    for model in MODELS:
        fitted_parameters = None
        parameter_converged = True
        if calibrated_parameters is not None and model in calibrated_parameters:
            fitted_parameters, parameter_converged = calibrated_parameters[model]
        elif model == "linear_bv":
            fitted_parameters, parameter_converged = fit_linear_parameters(
                family, true_population, exact_peptide, train
            )
        elif (
            model == "gamma_moments"
            or model.startswith("mixture_q")
            and not model.endswith("_default")
        ):
            fitted_parameters, parameter_converged = fit_distribution_parameters(
                calibration_family, model, true_population, exact_peptide, train
            )
        state_curve_basis = None
        if model.startswith("mixture_q") and fitted_parameters is not None:
            basis = []
            for state in range(len(data.state_names)):
                state_population = np.zeros(len(data.state_names))
                state_population[state] = 1.0
                basis.append(family.predict(model, state_population, fitted_parameters))
            state_curve_basis = np.stack(basis)
        start = time.perf_counter()
        at_truth = (
            np.einsum("s,str->tr", true_population, state_curve_basis)
            if state_curve_basis is not None
            else family.predict(model, true_population, fitted_parameters)
        )
        prediction_ms = 1000.0 * (time.perf_counter() - start)
        truth_error = np.abs(peptide_windows(at_truth) - exact_peptide)
        for noise in NOISE_LEVELS:
            seeds = (NOISE_SEEDS[0],) if noise == 0 else NOISE_SEEDS
            for seed in seeds:
                rng = np.random.default_rng(seed)
                observed = np.clip(
                    exact_peptide + rng.normal(0.0, noise, exact_peptide.shape),
                    0.0,
                    1.0,
                )
                fitted, converged = fit_population(
                    family, model, observed, train, fitted_parameters, state_curve_basis
                )
                fitted_prediction = peptide_windows(
                    np.einsum("s,str->tr", fitted, state_curve_basis)
                    if state_curve_basis is not None
                    else family.predict(model, fitted, fitted_parameters)
                )
                records.append(
                    {
                        "cohort": cohort,
                        "system_id": data.system_id,
                        "model": model,
                        "noise_sigma": noise,
                        "seed": seed,
                        "n_residues": data.z.shape[0],
                        "n_frames": data.z.shape[1],
                        "n_states": len(data.state_names),
                        "n_timepoints": len(data.times),
                        "n_peptides": exact_peptide.shape[1],
                        "truth_mae": float(truth_error.mean()),
                        "truth_max_abs": float(truth_error.max()),
                        "test_mae": float(
                            np.mean(
                                np.abs(
                                    fitted_prediction[:, test] - exact_peptide[:, test]
                                )
                            )
                        ),
                        "population_l1": float(np.abs(fitted - true_population).sum()),
                        "population_max_abs": float(
                            np.max(np.abs(fitted - true_population))
                        ),
                        "population_recovery_percent": float(
                            100.0
                            * (1.0 - jensenshannon(fitted, true_population, base=2.0))
                        ),
                        "converged": converged and parameter_converged,
                        "prediction_ms": prediction_ms,
                        "true_population": json.dumps(true_population.tolist()),
                        "fitted_population": json.dumps(fitted.tolist()),
                        "fitted_model_parameters": (
                            None
                            if fitted_parameters is None
                            else json.dumps(fitted_parameters.tolist())
                        ),
                        **data.metadata,
                    }
                )
    return records


def benchmark_kernels(
    data: EnsembleData, population: np.ndarray, cohort: str, repeats: int = 3
) -> dict[str, object]:
    """Time uncached kernels whose frame weights can change during optimisation."""
    weights = state_weights(data.states, population)

    def exact_kernel():
        return exact_uptake(data, population)

    def linear_kernel():
        mean_z = data.z @ weights
        return stable_uptake_from_log_exposure(
            np.log(data.times)[:, None] + data.log_kint[None, :] - mean_z[None, :]
        )

    def median_ms(kernel) -> float:
        kernel()  # warm allocator and numerical-library dispatch
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            kernel()
            durations.append(1000.0 * (time.perf_counter() - start))
        return float(np.median(durations))

    exact_ms = median_ms(exact_kernel)
    linear_ms = median_ms(linear_kernel)
    exact_elements = len(data.times) * data.z.size
    linear_elements = data.z.size + len(data.times) * data.z.shape[0]
    return {
        "cohort": cohort,
        "system_id": data.system_id,
        "n_residues": data.z.shape[0],
        "n_frames": data.z.shape[1],
        "n_timepoints": len(data.times),
        "exact_framewise_ms": exact_ms,
        "linear_additive_ms": linear_ms,
        "measured_speedup": exact_ms / max(linear_ms, EPS),
        "intermediate_element_ratio": exact_elements / linear_elements,
    }


def load_teaa() -> EnsembleData:
    heavy, acceptor, kint = load_features(TEAA_FEATURES)
    manifest = json.loads(TEAA_MANIFEST.read_text())
    open_frames = int(manifest["open_frames"])
    closed_frames = int(manifest["closed_frames"])
    if open_frames + closed_frames != heavy.shape[1]:
        raise ValueError("TeaA state counts do not match feature frames")
    states = np.concatenate(
        (np.zeros(open_frames, dtype=int), np.ones(closed_frames, dtype=int))
    )
    return EnsembleData(
        system_id="TeaA_open_closed",
        heavy=heavy,
        acceptor=acceptor,
        log_kint=np.log(kint),
        states=states,
        state_names=("open", "closed"),
        times=np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], dtype=np.float64),
        metadata={
            "length": heavy.shape[0],
            "rmsf_tercile": "not_applicable",
            "cath_class": "TeaA",
        },
    )


def atlas_diagnostics() -> pd.DataFrame:
    with (HERE / "data/systems.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    diagnostics = []
    for row in rows:
        heavy, acceptor, kint = load_features(
            HERE / "outputs/stage1" / row["system_id"] / "R1/features.npz"
        )
        z = 0.35 * heavy[:, 101:] + 2.0 * acceptor[:, 101:]
        log_rate = np.log(kint)[:, None] - z
        diagnostics.append(
            {
                **row,
                "length": int(row["length"]),
                "avg_RMSF": float(row["avg_RMSF"]),
                "rate_heterogeneity": float(np.mean(np.std(log_rate, axis=1))),
            }
        )
    frame = pd.DataFrame(diagnostics)
    frame["length_stratum"] = pd.qcut(
        frame["length"], 3, labels=("short", "medium", "long")
    )
    return frame


def select_atlas_subset(diagnostics: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for (_, _), block in diagnostics.groupby(
        ["length_stratum", "rmsf_tercile"], observed=True
    ):
        median = block.rate_heterogeneity.median()
        chosen = (
            block.assign(distance=(block.rate_heterogeneity - median).abs())
            .sort_values(["distance", "system_id"])
            .iloc[0]
        )
        selected.append(chosen)
    chosen_ids = {row.system_id for row in selected}
    remaining = diagnostics[~diagnostics.system_id.isin(chosen_ids)].copy()
    quantiles = remaining.rate_heterogeneity.quantile([0.1, 0.5, 0.9]).to_numpy()
    for target in quantiles:
        block = remaining[~remaining.system_id.isin(chosen_ids)].copy()
        chosen = (
            block.assign(distance=(block.rate_heterogeneity - target).abs())
            .sort_values(["distance", "system_id"])
            .iloc[0]
        )
        selected.append(chosen)
        chosen_ids.add(chosen.system_id)
    result = pd.DataFrame(selected).drop(columns="distance", errors="ignore")
    if len(result) != 12 or result.system_id.nunique() != 12:
        raise AssertionError("ATLAS selection did not produce 12 unique systems")
    return result.sort_values("system_id").reset_index(drop=True)


def load_atlas(row: pd.Series) -> EnsembleData:
    heavy_parts, acceptor_parts = [], []
    kint = None
    for replica in (1, 2, 3):
        heavy, acceptor, current_kint = load_features(
            HERE / "outputs/stage1" / row.system_id / f"R{replica}/features.npz"
        )
        heavy_parts.append(heavy[:, 101:])
        acceptor_parts.append(acceptor[:, 101:])
        if kint is None:
            kint = current_kint
        elif not np.allclose(kint, current_kint, rtol=1e-5, atol=1e-8):
            raise ValueError(f"{row.system_id}: intrinsic rates differ among replicas")
    heavy = np.concatenate(heavy_parts, axis=1)
    acceptor = np.concatenate(acceptor_parts, axis=1)
    z = 0.35 * heavy + 2.0 * acceptor
    states = np.repeat(np.arange(3), [part.shape[1] for part in heavy_parts])
    log_rate = np.log(kint)[:, None] - z
    low, high = np.quantile(log_rate, [0.05, 0.95])
    log_times = np.linspace(
        np.clip(-high - np.log(10.0), -650.0, 650.0),
        np.clip(-low + np.log(10.0), -650.0, 650.0),
        24,
    )
    return EnsembleData(
        system_id=row.system_id,
        heavy=heavy,
        acceptor=acceptor,
        log_kint=np.log(kint),
        states=states,
        state_names=("R1", "R2", "R3"),
        times=np.exp(log_times),
        metadata={
            "length": int(row.length),
            "rmsf_tercile": row.rmsf_tercile,
            "cath_class": row.cath_class,
            "length_stratum": str(row.length_stratum),
            "rate_heterogeneity": float(row.rate_heterogeneity),
        },
    )


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    rendered = frame.copy()
    for column in rendered.select_dtypes(include=[np.number]).columns:
        rendered[column] = rendered[column].map(lambda value: f"{value:.6g}")
    headings = [str(column) for column in rendered.columns]
    rows = [headings, ["---"] * len(headings)]
    rows.extend(
        [
            [str(value) for value in record]
            for record in rendered.itertuples(index=False, name=None)
        ]
    )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(results: pd.DataFrame, selection: pd.DataFrame, output: Path) -> None:
    noiseless = results[results.noise_sigma == 0]
    noisy = results[results.noise_sigma > 0]
    summary = (
        noiseless.groupby(["cohort", "model"])[
            [
                "truth_mae",
                "truth_max_abs",
                "test_mae",
                "population_max_abs",
                "population_recovery_percent",
            ]
        ]
        .median()
        .reset_index()
    )
    noisy_summary = (
        noisy.groupby(["cohort", "model"])[
            ["test_mae", "population_max_abs", "population_recovery_percent"]
        ]
        .median()
        .reset_index()
    )
    oracle = noiseless[noiseless.model == "exact"][
        ["cohort", "system_id", "true_population", "population_max_abs"]
    ].rename(columns={"population_max_abs": "oracle_population_max_abs"})
    assessed = noiseless.merge(
        oracle, on=["cohort", "system_id", "true_population"], how="left"
    )
    assessed["population_excess"] = (
        assessed.population_max_abs - assessed.oracle_population_max_abs
    )
    gates = (
        assessed.groupby(["cohort", "model"])
        .agg(
            median_truth_mae=("truth_mae", "median"),
            median_test_mae=("test_mae", "median"),
            median_population_excess=("population_excess", "median"),
        )
        .reset_index()
    )
    gates["uptake_gate"] = gates.median_truth_mae <= 0.02
    gates["recovery_gate"] = (gates.median_test_mae <= 0.02) & (
        gates.median_population_excess <= 0.10
    )
    atlas_assessed = assessed[assessed.cohort == "ATLAS"]
    robustness = (
        atlas_assessed.groupby("model")
        .agg(
            systems=("system_id", "nunique"),
            uptake_passes=("truth_mae", lambda values: int((values <= 0.02).sum())),
            worst_truth_mae=("truth_mae", "max"),
            worst_test_mae=("test_mae", "max"),
            worst_population_excess=("population_excess", "max"),
        )
        .reset_index()
    )
    atlas_strata = (
        noiseless[noiseless.cohort == "ATLAS"]
        .groupby(["model", "length_stratum", "rmsf_tercile"], observed=True)[
            ["truth_mae", "test_mae", "population_max_abs"]
        ]
        .median()
        .reset_index()
    )
    atlas_strata.to_csv(output / "summary_atlas_strata.csv", index=False)
    timings = pd.read_csv(output / "compute_benchmark.csv")
    timing_summary = (
        timings.groupby("cohort")[
            ["exact_framewise_ms", "linear_additive_ms", "measured_speedup"]
        ]
        .median()
        .reset_index()
    )
    summary.to_csv(output / "summary_noiseless.csv", index=False)
    noisy_summary.to_csv(output / "summary_noise_001.csv", index=False)
    lines = [
        "# BV uptake model validation",
        "",
        "Targets were generated exclusively with exact frame-wise EX2 uptake. TeaA uses",
        "known open/closed frames; ATLAS uses a frozen stratified 12-system subset and",
        "the three independent replicas as known population states. Population fitting",
        "uses alternating non-overlapping width-10 peptide windows for train and test.",
        "For every system, each candidate's parameters are first calibrated against",
        "an exact frame-wise target on the training windows, then frozen before",
        "population recovery. TeaA uses a 50/50 calibration mixture; ATLAS uses its",
        "known 60/30/10 calibration mixture.",
        "Gamma and Q4 calibration uses a deterministic 32-frame-per-state integration",
        "subset; exact targets and every reported prediction metric use all frames.",
        "The exact model is included as an oracle control for population identifiability;",
        "it is not a candidate tractable approximation.",
        "",
        "## Frozen ATLAS systems",
        "",
        ", ".join(selection.system_id.tolist()),
        "",
        "## Median noiseless results",
        "",
        markdown_table(summary),
        "",
        "## Median results with uptake noise sigma = 0.01",
        "",
        markdown_table(noisy_summary),
        "",
        "## Decision thresholds",
        "",
        "The uptake gate requires median exact-target MAE <= 0.02. The recovery gate",
        "requires median held-out MAE <= 0.02 and median population error no more than",
        "0.10 above the exact oracle, so an intrinsically unidentifiable state split is",
        "not incorrectly charged to an approximation.",
        "",
        markdown_table(gates),
        "",
        "## ATLAS robustness",
        "",
        markdown_table(robustness),
        "",
        "The full length-tercile x RMSF-tercile breakdown is in",
        "`summary_atlas_strata.csv`; this table prevents the aggregate median from",
        "hiding a difficult system.",
        "",
        "## Uncached kernel timing",
        "",
        markdown_table(timing_summary),
        "",
        "Timing recomputes from raw frame features and changing state weights. It is a",
        "local CPU measurement, so the speedup is evidence for scaling rather than a",
        "portable performance guarantee.",
        "",
        "`truth_mae` measures approximation error at the known population. `test_mae`",
        "and `population_max_abs` are evaluated after fitting only the training peptide",
        "windows. ATLAS uses an adaptive log-time grid because its physical rate range",
        "spans many orders of magnitude; this grid is fixed before model comparison.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = atlas_diagnostics()
    selection = select_atlas_subset(diagnostics)
    selection.to_csv(output / "atlas_selection.csv", index=False)

    records = []
    benchmarks = []
    teaa = load_teaa()
    benchmarks.append(benchmark_kernels(teaa, np.asarray([0.4, 0.6]), "TeaA"))
    teaa_calibrated = calibrate_models(teaa, np.asarray([0.5, 0.5]))
    exact_targets = {}
    for open_population in TEAA_POPULATIONS:
        population = np.asarray([open_population, 1.0 - open_population])
        records.extend(evaluate_case(teaa, population, "TeaA", teaa_calibrated))
        exact_targets[f"open_{open_population:.2f}"] = exact_uptake(teaa, population)
    np.savez_compressed(
        output / "teaa_exact_targets.npz", times=teaa.times, **exact_targets
    )

    for index, row in selection.iterrows():
        print(f"[{index + 1}/{len(selection)}] {row.system_id}", flush=True)
        atlas = load_atlas(row)
        calibrated = calibrate_models(atlas, ATLAS_POPULATION)
        records.extend(evaluate_case(atlas, ATLAS_POPULATION, "ATLAS", calibrated))
        benchmarks.append(benchmark_kernels(atlas, ATLAS_POPULATION, "ATLAS"))
    results = pd.DataFrame(records)
    results.to_parquet(output / "results.parquet", index=False)
    pd.DataFrame(benchmarks).to_csv(output / "compute_benchmark.csv", index=False)
    write_report(results, selection, output)
    manifest = {
        "status": "complete",
        "target_model": "exact_framewise_BV_EX2",
        "parameter_calibration": (
            "per system on training peptide windows; parameters frozen before "
            "population recovery; TeaA reference population 0.5/0.5"
        ),
        "distribution_calibration_frames_per_state": 32,
        "bv_bc": 0.35,
        "bv_bh": 2.0,
        "teaA_populations": list(TEAA_POPULATIONS),
        "atlas_population": ATLAS_POPULATION.tolist(),
        "noise_levels": list(NOISE_LEVELS),
        "noise_seeds": list(NOISE_SEEDS),
        "models": list(MODELS),
        "decision_gates": {
            "median_truth_mae_max": 0.02,
            "median_test_mae_max": 0.02,
            "median_population_excess_max": 0.10,
        },
        "atlas_systems": selection.system_id.tolist(),
        "records": len(results),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
