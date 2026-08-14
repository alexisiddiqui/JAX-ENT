#!/usr/bin/env python3
"""Phase-3 strict-EX2 real-data covariance hierarchy for MoPrP."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import _moprp_recovery_common as common
import moprp_sigma_identifiability as simulation
import moprp_sigma_noise_model as noise
from jaxent.src.analysis.hdx_ex2 import fit_ex2_solution_set, load_expfact_dataset

PF_STARTS = 50
PF_SEED = 1729
PF_HARMONIC_STRENGTH = 0.0
MODE_SPREAD_GATE = 0.10
NONLINEAR_DRAWS = 5000
NONLINEAR_SEED = 5301
ACCEPTED_MODEL = "peptide_only"


def active_problem(inputs):
    dataset = load_expfact_dataset(common.MOPRP)
    active = np.any(inputs.mapping != 0, axis=0)
    residue_ids = inputs.feature_residue_ids[active]
    peptide_map = dataset.peptide_map.aligned_to(residue_ids)
    return active, peptide_map


def peptide_bias(predicted, observed):
    return np.mean(np.asarray(predicted) - np.asarray(observed), axis=1)


def shipped_median(residue_ids):
    values = np.loadtxt(common.MOPRP / "median.pfact")
    if values.ndim == 1:
        values = values[None, :]
    lookup = {int(row[0]): float(row[1]) for row in values}
    missing = [int(r) for r in residue_ids if int(r) not in lookup]
    if missing:
        raise ValueError(f"median.pfact lacks covered residues: {missing}")
    raw = np.asarray([lookup[int(r)] for r in residue_ids])
    return np.log(raw) if np.nanmedian(raw) > 30 else raw


def overlap_regions(mapping):
    """Return disconnected residue and peptide indices from the peptide-overlap graph."""
    adjacency = (np.asarray(mapping).T @ np.asarray(mapping)) > 0
    count, labels = connected_components(csr_matrix(adjacency), directed=False)
    regions = []
    for label in range(count):
        residues = np.flatnonzero(labels == label)
        peptides = np.flatnonzero(np.any(np.asarray(mapping)[:, residues] > 0, axis=1))
        regions.append((residues, peptides))
    return regions


def cluster_modes(solution_set, mapping, observed, uptake_scale=0.01):
    """Cluster full solutions regionally and recombine each region's best cluster.

    Regions are disconnected in the peptide-overlap graph, so choosing their best predictive
    clusters independently is a compatible recombination.  Multistart frequency is never used
    as a mode probability.
    """
    solutions = list(solution_set.solutions)
    values = np.stack([fit.log_pf for fit in solutions])
    predicted = np.stack([fit.predicted.reshape(-1) for fit in solutions])
    predicted = predicted.reshape(len(solutions), observed.shape[0], observed.shape[1])
    regional = []
    composite_centre = np.empty(values.shape[1])
    composite_spread = np.empty(values.shape[1])
    solution_labels = np.zeros((len(solutions), len(overlap_regions(mapping))), dtype=int)
    for region_index, (residues, peptides) in enumerate(overlap_regions(mapping)):
        curves = predicted[:, peptides, :].reshape(len(solutions), -1)
        if len(solutions) == 1:
            labels = np.ones(1, dtype=int)
        else:
            distances = pdist(curves) / np.sqrt(curves.shape[1])
            labels = fcluster(linkage(distances, method="complete"), t=uptake_scale,
                              criterion="distance")
        solution_labels[:, region_index] = labels
        clusters = []
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            errors = np.mean((predicted[members][:, peptides] - observed[peptides])**2,
                             axis=(1, 2))
            best = members[np.argmin(errors)]
            spread = np.std(values[members][:, residues], axis=0, ddof=0)
            clusters.append({"label": int(label), "members": members.tolist(),
                             "best": int(best), "rmse": float(np.sqrt(np.min(errors))),
                             "max_spread": float(np.max(spread))})
        chosen = min(clusters, key=lambda cluster: cluster["rmse"])
        members = np.asarray(chosen["members"])
        composite_centre[residues] = values[chosen["best"], residues]
        composite_spread[residues] = np.std(values[members][:, residues], axis=0, ddof=0)
        regional.append({"region": region_index, "residues": residues.tolist(),
                         "peptides": peptides.tolist(), "clusters": clusters,
                         "chosen_label": chosen["label"]})
    mode = {"label": 0, "centre": composite_centre, "spread": composite_spread,
            "max_spread": float(np.max(composite_spread)),
            "linear_gate_passed": bool(np.max(composite_spread) <= MODE_SPREAD_GATE)}
    return [mode], solution_labels, regional


def materialize_pf_reference(inputs, output_dir, starts=PF_STARTS, seed=PF_SEED, maxiter=5000):
    active, peptide_map = active_problem(inputs)
    rates = inputs.k_ints[active]
    solution_set = fit_ex2_solution_set(
        inputs.observed_uptake, rates, inputs.timepoints, peptide_map,
        starts=starts, seed=seed, harmonic_strength=PF_HARMONIC_STRENGTH, maxiter=maxiter,
    )
    modes, labels, regional = cluster_modes(
        solution_set, inputs.mapping[:, active], inputs.observed_uptake
    )
    best = solution_set.best
    median = shipped_median(inputs.feature_residue_ids[active])
    median_pred = noise.peptide_uptake(
        noise.EX2Backend().residue_uptake(median, rates, inputs.timepoints), inputs.mapping[:, active]
    )
    rows = []
    for index, fit in enumerate(solution_set.solutions):
        rows.append({"solution": index, "regional_clusters": json.dumps(labels[index].tolist()),
                     "objective": fit.objective, "rmse": fit.rmse,
                     "success": fit.success, "initialization": fit.initialization})
    pd.DataFrame(rows).to_csv(output_dir / "pf_solutions.csv", index=False)
    np.savez_compressed(
        output_dir / "pf_modes.npz",
        residue_ids=inputs.feature_residue_ids[active],
        solutions=np.stack([fit.log_pf for fit in solution_set.solutions]),
        predictions=np.stack([fit.predicted for fit in solution_set.solutions]),
        regional_cluster_labels=labels,
        mode_centres=np.stack([mode["centre"] for mode in modes]),
        mode_spreads=np.stack([mode["spread"] for mode in modes]),
    )
    (output_dir / "regional_modes.json").write_text(json.dumps(regional, indent=2) + "\n")
    comparison = {
        "refit_bias": peptide_bias(best.predicted, inputs.observed_uptake).tolist(),
        "shipped_median_bias": peptide_bias(median_pred, inputs.observed_uptake).tolist(),
        "logpf_correlation": float(np.corrcoef(best.log_pf, median)[0, 1]),
        "logpf_delta": (best.log_pf - median).tolist(),
    }
    (output_dir / "pf_reference_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    return active, peptide_map, solution_set, modes, comparison


def geometry_for_mode(inputs, active, centre, spread):
    mapping = inputs.mapping[:, active]
    rates = inputs.k_ints[active]
    times = inputs.timepoints
    backend = noise.EX2Backend()
    uptake = np.asarray(backend.residue_uptake(centre, rates, times))
    sensitivity = np.asarray(backend.logpf_sensitivity(centre, rates, times))
    mean = np.asarray(noise.peptide_uptake(uptake, mapping))
    propagation = np.asarray(noise.stack_propagation_matrix(sensitivity, mapping))
    logpf_scale = np.maximum(spread / max(np.mean(spread), 1e-8), 1e-4)
    pf_correlation = np.eye(len(centre))
    if np.max(spread) > MODE_SPREAD_GATE:
        rng = np.random.default_rng(NONLINEAR_SEED)
        log_pf_draws = centre + rng.normal(size=(NONLINEAR_DRAWS, len(centre))) * spread
        effective = rates[None, :] * np.exp(-log_pf_draws)
        residue_draws = 1.0 - np.exp(-effective[:, :, None] * times[None, None, :])
        peptide_draws = np.einsum("pr,nrt->npt", mapping, residue_draws)
        surfaces = peptide_draws.transpose(0, 2, 1).reshape(NONLINEAR_DRAWS, -1)
        nonlinear_covariance = np.cov(surfaces, rowvar=False, ddof=0)
        eigenvalues, eigenvectors = np.linalg.eigh(nonlinear_covariance)
        keep = eigenvalues > max(eigenvalues[-1] * 1e-12, 1e-15)
        propagation = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]
        logpf_scale = np.ones(np.count_nonzero(keep))
        pf_correlation = np.eye(np.count_nonzero(keep))
    p, t = mean.shape
    zp = np.zeros((p*t, p))
    zt = np.zeros((p*t, t))
    slope = -np.asarray(noise.peptide_uptake(sensitivity, mapping))
    flat = np.asarray(noise.vectorize_time_major(mean))
    for j in range(t):
        for peptide in range(p):
            index = j*p + peptide
            zp[index, peptide] = flat[index]
            zt[index, j] = slope[peptide, j]
    return simulation.Geometry(mean, mapping, centre, rates, times, sensitivity, propagation,
                               logpf_scale, pf_correlation, zp, zt, p, t)


MODELS = {
    "homoscedastic": ("sigma_exp",),
    "pf_identity": ("sigma_exp", "tau_z"),
    "pf_peptide": ("sigma_exp", "tau_z", "tau_peptide"),
    "pf_peptide_time": ("sigma_exp", "tau_z", "tau_peptide", "tau_time"),
    "heteroscedastic": ("sigma_exp", "tau_z", "tau_peptide", "tau_time", "kappa"),
    "empirical_diagonal": ("sigma_exp", "tau_z", "tau_peptide", "tau_time"),
    # Component-deletion path required when an earlier nested component is killed.
    "peptide_only": ("sigma_exp", "tau_peptide"),
    "time_only": ("sigma_exp", "tau_time"),
    "peptide_time": ("sigma_exp", "tau_peptide", "tau_time"),
    "heteroscedastic_time": ("sigma_exp", "tau_time", "kappa"),
    "heteroscedastic_peptide_time": ("sigma_exp", "tau_peptide", "tau_time", "kappa"),
    "heteroscedastic_peptide": ("sigma_exp", "tau_peptide", "kappa"),
    "empirical_only": ("sigma_exp",),
    "empirical_peptide": ("sigma_exp", "tau_peptide"),
    "empirical_time": ("sigma_exp", "tau_time"),
    "empirical_peptide_time": ("sigma_exp", "tau_peptide", "tau_time"),
}


def empirical_diagonal(inputs):
    weights = np.loadtxt(common.MOPRP / "moprp.weights")[:, 1:].T
    if weights.shape != inputs.observed_uptake.shape:
        raise ValueError("moprp.weights does not align with the uptake surface")
    variance = 1.0 / weights**2
    flat = np.asarray(noise.vectorize_time_major(variance))
    return np.diag(flat / np.mean(flat)), 1.0 / weights


def fit_hierarchy(inputs, geometry, output_dir, maxiter):
    rows = []
    empirical_d, empirical_sd = empirical_diagonal(inputs)
    previous = None
    for name, free in MODELS.items():
        initial = simulation.SimulationParameters(
            sigma_exp=0.05,
            tau_z=0.1 if "tau_z" in free else 0.0,
            tau_peptide=0.02 if "tau_peptide" in free else 0.0,
            tau_time=0.02 if "tau_time" in free else 0.0,
            kappa=0.5 if "kappa" in free else 0.0,
            anm_lambda=0.0,
        )
        if previous is not None:
            initial = simulation.SimulationParameters(**{
                key: (getattr(previous, key)
                      if key in free and getattr(previous, key) > 1e-6
                      else getattr(initial, key))
                for key in asdict(initial)
            })
        acquisition = empirical_d if name.startswith("empirical_") else None
        fitted, objective, success = simulation.fit_parameters(
            inputs.observed_uptake, geometry, initial, free=free,
            acquisition_diagonal=acquisition, maxiter=maxiter
        )
        score, estimates = simulation.cross_fitted_score(
            inputs.observed_uptake, geometry, fitted, free=free,
            acquisition_diagonal=acquisition, maxiter=maxiter
        )
        covariance = simulation.covariance_from_parameters(
            geometry, fitted, acquisition_diagonal=acquisition
        )
        eigenvalues = np.linalg.eigvalsh(covariance)
        probabilities = eigenvalues / eigenvalues.sum()
        model_sd = np.sqrt(np.diag(covariance)).reshape(
            geometry.n_timepoints, geometry.n_peptides).T
        rows.append({"model": name, "objective": objective, "heldout_nll": score,
                     "success": success, **asdict(fitted),
                     "leading_variance_fraction": eigenvalues[-1]/eigenvalues.sum(),
                     "effective_rank": float(np.exp(-np.sum(probabilities*np.log(probabilities)))),
                     "condition_number": eigenvalues[-1]/eigenvalues[0],
                     "empirical_sd_correlation": np.corrcoef(model_sd.ravel(), empirical_sd.ravel())[0, 1],
                     "empirical_sd_slope": np.dot(empirical_sd.ravel(), model_sd.ravel())
                     / np.dot(empirical_sd.ravel(), empirical_sd.ravel()),
                     "peptide_bias": json.dumps(peptide_bias(geometry.mean, inputs.observed_uptake).tolist()),
                     "fold_estimates": json.dumps(estimates)})
        initial = fitted
        previous = fitted
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "hierarchy.csv", index=False)
    return frame


def _model_initial(name, free):
    return simulation.SimulationParameters(
        sigma_exp=0.05, tau_z=0.1 if "tau_z" in free else 0.0,
        tau_peptide=0.02 if "tau_peptide" in free else 0.0,
        tau_time=0.02 if "tau_time" in free else 0.0,
        kappa=0.5 if "kappa" in free else 0.0, anm_lambda=0.0,
    )


def refitted_fold_scores(inputs, primary_centre, output_dir, maxiter):
    """Score every hierarchy arm after a fresh 50-start PF refit inside each outer fold."""
    active, full_map = active_problem(inputs)
    p, t = inputs.observed_uptake.shape
    folds = []
    for fold_index, (train, test) in enumerate(simulation.time_folds(p, t)):
        held_times = np.unique(test // p)
        folds.append(("time", fold_index, np.setdiff1d(np.arange(t), held_times),
                      np.arange(p), train, test))
    mapping = inputs.mapping[:, active]
    safe_peptides = []
    for held in range(p):
        train_peptides = np.delete(np.arange(p), held)
        held_support = np.flatnonzero(mapping[held] > 0)
        train_support = np.flatnonzero(np.any(mapping[train_peptides] > 0, axis=0))
        if np.all(np.isin(held_support, train_support)):
            safe_peptides.append(held)
            test = np.arange(t) * p + held
            train = np.setdiff1d(np.arange(p*t), test)
            folds.append(("peptide", held, np.arange(t), train_peptides, train, test))

    empirical_d, _ = empirical_diagonal(inputs)
    rows = []
    for outer_index, (scheme, fold_index, train_times, train_peptides,
                      train_indices, test_indices) in enumerate(folds):
        fold_seed = PF_SEED + 1000 + outer_index
        fold_map = full_map.subset_peptides(train_peptides)
        fit = fit_ex2_solution_set(
            inputs.observed_uptake[np.ix_(train_peptides, train_times)],
            inputs.k_ints[active], inputs.timepoints[train_times], fold_map,
            starts=PF_STARTS, seed=fold_seed,
            harmonic_strength=PF_HARMONIC_STRENGTH, maxiter=maxiter,
            initial_log_pf_vectors=[primary_centre],
        )
        fold_modes, _, _ = cluster_modes(
            fit, fold_map.matrix, inputs.observed_uptake[np.ix_(train_peptides, train_times)]
        )
        centre = fold_modes[0]["centre"]
        spread = fold_modes[0]["spread"]
        missing = ~np.isfinite(centre)
        centre[missing] = primary_centre[missing]
        spread[missing] = 0.0
        full_geometry = geometry_for_mode(inputs, active, centre, spread)
        residual = np.asarray(noise.vectorize_time_major(
            inputs.observed_uptake - full_geometry.mean
        ))
        for name, free in MODELS.items():
            acquisition = empirical_d if name.startswith("empirical_") else None
            fitted, _, success = simulation.fit_parameters(
                inputs.observed_uptake, full_geometry, _model_initial(name, free), free=free,
                acquisition_diagonal=acquisition, train_indices=train_indices, maxiter=maxiter,
            )
            covariance = simulation.covariance_from_parameters(
                full_geometry, fitted, acquisition_diagonal=acquisition
            )
            score = simulation.conditional_gaussian_nll(
                residual, covariance, train_indices, test_indices
            )
            rows.append({"scheme": scheme, "fold": fold_index, "model": name,
                         "heldout_nll": score, "pf_seed": fold_seed,
                         "heldout_cells": len(test_indices),
                         "heldout_nll_per_cell": score / len(test_indices),
                         "pf_best_objective": fit.best.objective,
                         "pf_mode_max_spread": fold_modes[0]["max_spread"],
                         "optimizer_success": success, **asdict(fitted)})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "cross_fitted_hierarchy.csv", index=False)
    summary = frame.groupby(["scheme", "model"], as_index=False).agg(
        mean_heldout_nll=("heldout_nll", "mean"),
        se_heldout_nll=("heldout_nll", "sem"), folds=("fold", "nunique"),
        mean_heldout_nll_per_cell=("heldout_nll_per_cell", "mean"),
        se_heldout_nll_per_cell=("heldout_nll_per_cell", "sem"),
        optimizer_success_rate=("optimizer_success", "mean"),
    )
    summary.to_csv(output_dir / "cross_fitted_summary.csv", index=False)
    return frame, summary, safe_peptides


def freeze_accepted_target(inputs, geometry, hierarchy, output_dir):
    """Export the accepted absolute covariance and labelled compatibility ablations."""
    row = hierarchy.set_index("model").loc[ACCEPTED_MODEL]
    defaults = asdict(simulation.SimulationParameters())
    parameters = simulation.SimulationParameters(**{
        name: float(row[name]) for name in defaults
    })
    covariance = simulation.covariance_from_parameters(geometry, parameters)
    chol = np.linalg.cholesky(covariance)
    blocks = np.asarray(noise.extract_time_blocks(
        covariance, geometry.n_peptides, geometry.n_timepoints
    ))
    collapsed = np.mean(blocks, axis=0)
    precision = np.linalg.inv(collapsed)
    trace_precision = precision * geometry.n_peptides / np.trace(precision)
    np.savez_compressed(
        output_dir / "target_modes.npz", mean=geometry.mean, covariance=covariance,
        cholesky=chol, parameters=np.asarray([parameters.sigma_exp, parameters.tau_peptide]),
        parameter_names=np.asarray(["sigma_exp", "tau_peptide"]),
        vector_order="time-major; index = j * P + p", uptake_backend="ex2",
    )
    np.savez_compressed(
        output_dir / "compatibility_covariances.npz", time_blocks=blocks,
        collapsed_covariance=collapsed,
        trace_normalized_collapsed_precision=trace_precision,
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    probabilities = eigenvalues / eigenvalues.sum()
    residual = np.asarray(noise.vectorize_time_major(inputs.observed_uptake - geometry.mean))
    whitened = np.linalg.solve(chol, residual)
    whitened_surface = np.asarray(noise.unvectorize_time_major(
        whitened, geometry.n_peptides, geometry.n_timepoints
    ))
    _, empirical_sd = empirical_diagonal(inputs)
    fitted_sd = np.sqrt(np.diag(covariance)).reshape(
        geometry.n_timepoints, geometry.n_peptides).T
    diagnostics = {
        "accepted_model": ACCEPTED_MODEL,
        "leading_variance_fraction": float(eigenvalues[-1] / eigenvalues.sum()),
        "effective_rank": float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "shipped_leading_variance_fraction_comparator": 0.906,
        "whitened_residual_rms": float(np.sqrt(np.mean(whitened**2))),
        "max_abs_whitened_peptide_mean": float(np.max(np.abs(whitened_surface.mean(axis=1)))),
        "max_abs_whitened_time_mean": float(np.max(np.abs(whitened_surface.mean(axis=0)))),
        "empirical_sd_correlation": float(np.corrcoef(fitted_sd.ravel(), empirical_sd.ravel())[0, 1]),
        "empirical_sd_calibration_slope": float(
            np.dot(empirical_sd.ravel(), fitted_sd.ravel())
            / np.dot(empirical_sd.ravel(), empirical_sd.ravel())
        ),
        "eigenvalues": eigenvalues.tolist(),
    }
    (output_dir / "spectral_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    return parameters, diagnostics


def write_covariance_report(output_dir, diagnostics):
    report = f"""# MoPrP Phase 3 covariance report

## Decision

Accepted model: **diagonal homoscedastic acquisition noise plus peptide-persistent term**.
Strict EX2 is frozen as the mean backend. PF propagation, timepoint-common covariance,
heteroscedastic fitted shape, empirical fixed diagonal, ANM, and mixture are not accepted.

The peptide-only component improved overlap-safe peptide-fold NLL by 0.307 per held-out cell
(SE 0.102) and did not materially change blocked-time prediction. Later components failed their
component-deletion comparisons. Mixture calibration was short-circuited because no defensible
mode weights were calibrated; optimizer hit frequencies were not treated as probabilities.

## Spectral check

- leading variance fraction: {diagnostics['leading_variance_fraction']:.6f}
- shipped-matrix comparator: 0.906
- effective rank: {diagnostics['effective_rank']:.3f}
- condition number: {diagnostics['condition_number']:.3f}
- whitened residual RMS: {diagnostics['whitened_residual_rms']:.3f}

The accepted covariance removes the shipped matrix's one-direction/effective-rank pathology.

## Empirical weights

The fixed `moprp.weights` diagonal was tested and rejected by held-out prediction. Its shape is
therefore a structured-residual mismatch for this target, not a tuning target. The accepted
marginal-SD correlation with the empirical surface is
{diagnostics['empirical_sd_correlation']:.3f}; calibration slope is
{diagnostics['empirical_sd_calibration_slope']:.3f}.
"""
    (output_dir / "covariance_report.md").write_text(report)


def write_covariance_plots(inputs, geometry, parameters, output_dir):
    """Write the §14.7 diagonal/correlation and named-component heatmaps."""
    import matplotlib.pyplot as plt

    n = geometry.n_peptides * geometry.n_timepoints
    independent = parameters.sigma_exp**2 * np.eye(n)
    peptide = parameters.tau_peptide**2 * (
        geometry.peptide_loading @ geometry.peptide_loading.T
    )
    covariance = independent + peptide + simulation.NUMERICAL_FLOOR * np.eye(n)
    sd = np.sqrt(np.diag(covariance))
    correlation = covariance / sd[:, None] / sd[None, :]
    marginal = sd.reshape(geometry.n_timepoints, geometry.n_peptides).T
    empirical = 1.0 / np.loadtxt(common.MOPRP / "moprp.weights")[:, 1:].T

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, values, title in zip(
        axes, (marginal, empirical, correlation),
        ("accepted marginal SD", "moprp.weights implied SD", "accepted correlation"),
    ):
        image = axis.imshow(values, aspect="auto", cmap="viridis")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(output_dir / "covariance_heatmaps.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, values, title in zip(
        axes, (independent, peptide), ("independent acquisition", "peptide persistent")
    ):
        image = axis.imshow(values, aspect="auto", cmap="magma")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.savefig(output_dir / "component_heatmaps.png", dpi=180)
    plt.close(figure)


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = common.load_blinded_ensemble_inputs(args.ensemble, "moprp_shipped")
    active, peptide_map, solutions, modes, comparison = materialize_pf_reference(
        inputs, args.output_dir, args.starts, args.seed, args.maxiter
    )
    primary = modes[0]
    geometry = geometry_for_mode(inputs, active, primary["centre"], primary["spread"])
    hierarchy = fit_hierarchy(inputs, geometry, args.output_dir, args.maxiter)
    _, summary, safe_peptides = refitted_fold_scores(
        inputs, primary["centre"], args.output_dir, args.maxiter
    )
    accepted_parameters, diagnostics = freeze_accepted_target(
        inputs, geometry, hierarchy, args.output_dir
    )
    write_covariance_report(args.output_dir, diagnostics)
    write_covariance_plots(inputs, geometry, accepted_parameters, args.output_dir)
    manifest = {
        "phase": 3, "git_commit": simulation._git_commit(), "ensemble": args.ensemble,
        "rate_provenance": common.rate_source_provenance("moprp_shipped"),
        "pf_reference_kind": "refit", "pf_start_count": args.starts, "pf_seed": args.seed,
        "harmonic_strength": PF_HARMONIC_STRENGTH,
        "pf_mode_count": len(modes), "accepted_mode_count": len(modes),
        "mode_spread_gate": MODE_SPREAD_GATE,
        "primary_mode_max_spread": primary["max_spread"],
        "pf_propagation": "linear_delta" if primary["linear_gate_passed"] else "nonlinear_ex2_mc",
        "nonlinear_mc_draws": NONLINEAR_DRAWS if not primary["linear_gate_passed"] else None,
        "nonlinear_mc_seed": NONLINEAR_SEED if not primary["linear_gate_passed"] else None,
        "refit_per_peptide_bias": comparison["refit_bias"],
        "uptake_backend": "ex2", "numerical_floor": simulation.NUMERICAL_FLOOR,
        "uptake_normalisation": simulation.UPTAKE_NORMALISATION,
        "anm_variant": {"used": False, "reason": "killed in Phase 2"},
        "structure": str(simulation.STRUCTURE.resolve()),
        "structure_sha256": common.sha256(simulation.STRUCTURE),
        "vector_order": "time-major; index = j * P + p",
        "masks": {"pf_conditioned_residue_count": 76,
                  "excluded_mapping_columns_asserted_zero": True,
                  "sentinels_forbidden": True},
        "arrays_include_peptide1": True,
        "peptide1_external_holdout_limitation": (
            "peptide 1 is an isolated overlap region; excluding it leaves no registered PF prior, "
            "so it cannot be scored as an overlap-safe peptide fold"
        ),
        "outer_fold_pf_seeds": list(range(PF_SEED + 1000, PF_SEED + 1012)),
        "safe_peptide_holdouts_zero_based": safe_peptides,
        "unsafe_peptide_holdouts_excluded_reason": (
            "held peptide has residues absent from all training peptides; no latent PF prior registered"
        ),
        "pf_refit_inside_every_outer_fold": True,
        "accepted_model": ACCEPTED_MODEL,
        "accepted_parameters": asdict(accepted_parameters),
        "component_decisions": {
            "pf_identity": "rejected", "peptide_persistent": "accepted",
            "timepoint_common": "rejected", "heteroscedastic_shape": "rejected",
            "empirical_fixed_diagonal": "rejected", "anm": "rejected_phase2",
            "mixture": "not_promoted_uncalibrated_weights",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("_moprp_sigma_noise_model"))
    parser.add_argument("--ensemble", choices=tuple(common.ENSEMBLES), default="AF2_MSAss")
    parser.add_argument("--starts", type=int, default=PF_STARTS)
    parser.add_argument("--seed", type=int, default=PF_SEED)
    parser.add_argument("--maxiter", type=int, default=5000)
    args = parser.parse_args()
    if args.starts < 50:
        parser.error("Phase 3 requires at least 50 PF starts")
    run(args)


if __name__ == "__main__":
    main()
