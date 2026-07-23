#!/usr/bin/env python3
"""No-fitting litmus: can target uptake recover framewise effective-rate covariance?"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], float)
ALPHAS = (0.01, 0.05, 0.10)
PRIMARY_ALPHA = 0.05


def weighted_covariance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float) / np.sum(weights)
    centered = values - weights @ values
    result = (centered * weights[:, None]).T @ centered
    return (result + result.T) / 2


def effective_rates(log_pf: np.ndarray, k_int: np.ndarray) -> np.ndarray:
    """Return frames x residues linear exchange rates."""
    return (k_int[:, None] * np.exp(-log_pf)).T


def framewise_uptake(rates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return times x frames x residues uptake."""
    return 1.0 - np.exp(-times[:, None, None] * rates[None, :, :])


def apparent_rate(uptake: np.ndarray, time: float | np.ndarray) -> np.ndarray:
    survival = np.clip(1.0 - np.asarray(uptake, float), np.finfo(float).tiny, 1.0)
    return -np.log(survival) / np.asarray(time, float)


def shrink(covariance: np.ndarray, alpha: float) -> np.ndarray:
    dimension = covariance.shape[0]
    scale = np.trace(covariance) / dimension
    return (1-alpha) * covariance + (alpha * scale + 1e-8 * scale + 1e-12) * np.eye(dimension)


def profiles(covariance: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    regularized = shrink(covariance, alpha)
    return {"marginal": np.diag(regularized), "conditional": 1.0 / np.diag(np.linalg.inv(regularized))}


def relative_error(candidate: np.ndarray, truth: np.ndarray) -> float:
    denominator = np.linalg.norm(truth)
    return float(np.linalg.norm(candidate-truth) / denominator) if denominator else float(np.linalg.norm(candidate))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    return float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 and np.std(left) > 1e-15 and np.std(right) > 1e-15 else np.nan


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def matrix_metrics(candidate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    mask = ~np.eye(truth.shape[0], dtype=bool)
    candidate_trace, truth_trace = float(np.trace(candidate)), float(np.trace(truth))
    cn = candidate / candidate_trace if candidate_trace > 0 else candidate
    tn = truth / truth_trace if truth_trace > 0 else truth
    eigenvalues = np.maximum(np.linalg.eigvalsh(candidate), 0.0)
    probabilities = eigenvalues/eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues
    effective_rank = float(np.exp(-np.sum(probabilities[probabilities > 0]*np.log(probabilities[probabilities > 0])))) if probabilities.sum() else 0.0
    return {
        "raw_relative_error": relative_error(candidate, truth),
        "trace_ratio": candidate_trace / truth_trace if truth_trace > 0 else np.nan,
        "normalized_frobenius_distance": float(np.linalg.norm(cn-tn)),
        "off_diagonal_correlation": correlation(candidate[mask], truth[mask]),
        "numerical_rank": int(np.linalg.matrix_rank(candidate, tol=max(np.linalg.norm(candidate), 1.0)*1e-10)),
        "effective_rank": effective_rank,
    }


def overlap_projection(mapping: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    normalized = mapping/np.linalg.norm(mapping, axis=1, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(normalized@normalized.T)
    return eigenvectors[:, eigenvalues > threshold*np.max(eigenvalues)]


def log_euclidean_distance(candidate: np.ndarray, truth: np.ndarray, projection: np.ndarray, alpha: float = PRIMARY_ALPHA) -> float:
    def matrix_log(matrix: np.ndarray) -> np.ndarray:
        values, vectors = np.linalg.eigh(shrink(matrix, alpha))
        return (vectors*np.log(np.maximum(values, np.finfo(float).tiny))[None, :])@vectors.T
    candidate_p = projection.T@candidate@projection
    truth_p = projection.T@truth@projection
    return float(np.sqrt(np.mean((matrix_log(candidate_p)-matrix_log(truth_p))**2)))


def profile_metrics(candidate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    candidate, truth = np.asarray(candidate), np.asarray(truth)
    tiny = np.finfo(float).tiny
    cn, tn = candidate / max(candidate.mean(), tiny), truth / max(truth.mean(), tiny)
    return {
        "scale_ratio": float(candidate.mean()/truth.mean()),
        "pearson": correlation(cn, tn),
        "spearman": correlation(ranks(cn), ranks(tn)),
        "log_rmse": float(np.sqrt(np.mean((np.log(np.maximum(cn, tiny))-np.log(np.maximum(tn, tiny)))**2))),
    }


def curve_covariances(mean_peptide_uptake: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """Candidate P x P matrices using only the ordinary T x P mean target curve."""
    uptake = np.asarray(mean_peptide_uptake, float)
    survival_log = np.log(np.clip(1.0-uptake, np.finfo(float).tiny, 1.0))
    adjacent = -(survival_log[1:] - survival_log[:-1]) / np.diff(times)[:, None]
    scale = np.sqrt(np.mean(uptake**2, axis=1, keepdims=True))
    observations = {
        "curve_raw_uptake": uptake,
        "curve_uptake_over_t": uptake / times[:, None],
        "curve_apparent_rate": apparent_rate(uptake, times[:, None]),
        "curve_adjacent_survival_slope": adjacent,
        "curve_timepoint_rms_normalized": uptake / np.maximum(scale, np.finfo(float).tiny),
    }
    return {name: weighted_covariance(values, np.full(len(values), 1/len(values))) for name, values in observations.items()}


def permute_equal_weight_frames(values: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
    """Independently permute each residue within equal-weight groups."""
    rng = np.random.default_rng(seed)
    result = np.asarray(values).copy()
    groups = np.round(weights, decimals=15)
    for residue in range(result.shape[1]):
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            result[indices, residue] = result[rng.permutation(indices), residue]
    return result


def cumulants_from_survival(survival: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit log S(t) = -kappa1*t + kappa2*t^2/2 with zero intercept."""
    design = np.column_stack((-times, 0.5*times**2))
    coefficients = np.linalg.lstsq(design, np.log(np.clip(survival, np.finfo(float).tiny, 1.0)), rcond=None)[0]
    return coefficients[0], coefficients[1]


def load_inputs(results_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    manifest_path = results_dir / "manifest.json"
    source = json.loads(manifest_path.read_text())
    feature_path = Path(source["inputs"]["bi_features"]["path"])
    cluster_path = Path(source["inputs"]["bi_clusters"]["path"])
    with np.load(feature_path) as data:
        log_pf = 0.35*np.asarray(data["heavy_contacts"], float) + 2.0*np.asarray(data["acceptor_contacts"], float)
        k_int = np.asarray(data["k_ints"], float)
    assignments = pd.read_csv(cluster_path)["cluster_assignment"].to_numpy(int)
    known = np.zeros(len(assignments))
    known[assignments == 0] = 0.4/np.sum(assignments == 0)
    known[assignments == 1] = 0.6/np.sum(assignments == 1)
    mappings = {panel: np.load(results_dir/f"panel_{panel}_mapping.npz")["mapping"].astype(float) for panel in source["config"]["panels"]}
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {"source_manifest": str(manifest_path.resolve()), "bi_features": {"path": str(feature_path), "sha256": sha(feature_path)}, "bi_clusters": {"path": str(cluster_path), "sha256": sha(cluster_path)}}
    return inputs, log_pf, k_int, assignments, known, mappings


def construction_registry() -> pd.DataFrame:
    rows = [
        ("truth_exact_rate", "oracle", "Cov_w(k_int exp(-z))", "positive control"),
        ("control_logpf", "oracle", "Cov_w(z)", "coordinate-mismatch negative control"),
        ("control_kint_scaled_logpf", "oracle", "D_kint Cov_w(z) D_kint", "omits exp(-z)"),
        ("gaussian_rate_closure", "oracle", "E[k]E[k]^T*(exp(C_z)-1)", "Gaussian approximation"),
        ("oracle_inverse_uptake", "oracle", "Cov_w(-log(1-u_f(t))/t)", "transform positive control"),
        ("oracle_inverse_uptake_asymptotic", "oracle", "same at max exposure 1e-7", "unsaturated transform positive control"),
        ("oracle_delta_pullback", "oracle", "D_g^-1 Cov_w(u_f(t)) D_g^-1", "local uptake-to-rate pullback"),
        ("oracle_peptide_first_inverse", "oracle", "Cov_w(-log(1-Mu_f(t))/t)", "mapping-loss control"),
        ("curve_raw_uptake", "observable", "Cov_t(y(t))", "raw curve geometry"),
        ("curve_uptake_over_t", "observable", "Cov_t(y(t)/t)", "small-time rate proxy"),
        ("curve_apparent_rate", "observable", "Cov_t(-log(1-y(t))/t)", "apparent-rate proxy"),
        ("curve_adjacent_survival_slope", "observable", "Cov_intervals(-Delta log(1-y)/Delta t)", "interval rate proxy"),
        ("curve_timepoint_rms_normalized", "observable_shape_only", "Cov_t(y/RMS_peptide(y_t))", "shape-only control"),
    ]
    return pd.DataFrame(rows, columns=["construction", "availability", "formula", "role"])


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs, log_pf, k_int, assignments, known, mappings = load_inputs(args.results_dir)
    weight_sets = {"uniform": np.full(log_pf.shape[1], 1/log_pf.shape[1]), "known_40_60": known}
    matrices: dict[str, np.ndarray] = {}
    metric_rows, profile_rows, permutation_rows, cumulant_rows, mean_rows = [], [], [], [], []

    for weight_name, weights in weight_sets.items():
        rates = effective_rates(log_pf, k_int)
        mean_rate = weights @ rates
        rate_cov = weighted_covariance(rates, weights)
        z_values = log_pf.T
        z_cov = weighted_covariance(z_values, weights)
        inverse_pf_cov = weighted_covariance(np.exp(-z_values), weights)
        exact_scaled = k_int[:, None]*inverse_pf_cov*k_int[None, :]
        gaussian_mean = k_int*np.exp(-(weights@z_values)+0.5*np.diag(z_cov))
        gaussian_cov = np.outer(gaussian_mean, gaussian_mean)*(np.exp(z_cov)-1)
        uptake = framewise_uptake(rates, TIMEPOINTS)
        oracle_time = 1e-7/np.max(rates)
        oracle_uptake = framewise_uptake(rates, np.asarray([oracle_time]))[0]
        permuted_rates = permute_equal_weight_frames(rates, weights, seed=1729)
        permuted_uptake = framewise_uptake(permuted_rates, TIMEPOINTS)

        mean_uptake_residue = np.einsum("tfr,f->tr", uptake, weights)
        production_rate = k_int*np.exp(-(weights@z_values))
        for name, candidate in (("production_average_first", production_rate), ("gaussian_mean_rate", gaussian_mean), ("exact_mean_rate", mean_rate)):
            mean_rows.append({"weights": weight_name, "construction": name, "relative_error": relative_error(candidate, mean_rate), "correlation": correlation(candidate, mean_rate), "mean_signed_error": float(np.mean(candidate-mean_rate))})

        # Theoretical marginal ceiling from the Laplace-transform curvature.
        auto_times = np.asarray([1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4])/np.max(rates)
        for grid_name, times in (("production", TIMEPOINTS), ("asymptotic", auto_times)):
            survival = np.einsum("tfr,f->tr", 1-framewise_uptake(rates, times), weights)
            kappa1, kappa2 = cumulants_from_survival(survival, times)
            cumulant_rows.append({"weights": weight_name, "level": "residue", "time_grid": grid_name, "mean_rate_relative_error": relative_error(kappa1, mean_rate), "marginal_variance_relative_error": relative_error(kappa2, np.diag(rate_cov)), "marginal_correlation": correlation(kappa2, np.diag(rate_cov))})

        for panel, mapping in mappings.items():
            projection = overlap_projection(mapping)
            truth = mapping @ rate_cov @ mapping.T
            mean_peptide_uptake = mean_uptake_residue @ mapping.T
            permuted_mean = np.einsum("tfr,f->tr", permuted_uptake, weights) @ mapping.T
            permuted_truth = mapping @ weighted_covariance(permuted_rates, weights) @ mapping.T
            permutation_rows.append({"weights": weight_name, "panel": panel, "mean_uptake_max_abs_change": float(np.max(np.abs(permuted_mean-mean_peptide_uptake))), "full_covariance_relative_change": relative_error(permuted_truth, truth), "marginal_relative_change": relative_error(np.diag(permuted_truth), np.diag(truth)), "conditional_relative_change": relative_error(profiles(permuted_truth, PRIMARY_ALPHA)["conditional"], profiles(truth, PRIMARY_ALPHA)["conditional"])})

            candidates: dict[str, tuple[np.ndarray, str, float | None]] = {
                "truth_exact_rate": (truth, "oracle", None),
                "control_logpf": (mapping@z_cov@mapping.T, "oracle", None),
                "control_kint_scaled_logpf": (mapping@(k_int[:, None]*z_cov*k_int[None, :])@mapping.T, "oracle", None),
                "exact_inverse_pf_scaled": (mapping@exact_scaled@mapping.T, "oracle", None),
                "gaussian_rate_closure": (mapping@gaussian_cov@mapping.T, "oracle", None),
                "oracle_inverse_uptake_asymptotic": (
                    mapping@weighted_covariance(apparent_rate(oracle_uptake, oracle_time), weights)@mapping.T,
                    "oracle",
                    oracle_time,
                ),
            }
            for time_index, time in enumerate(TIMEPOINTS):
                residue_u = uptake[time_index]
                cu = weighted_covariance(residue_u, weights)
                derivative = time*np.exp(-time*mean_rate)
                # Saturated coordinates have no locally recoverable rate information.
                active = np.abs(derivative) > 1e-8
                inverse_derivative = np.divide(1.0, derivative, out=np.zeros_like(derivative), where=active)
                pullback = inverse_derivative[:, None]*cu*inverse_derivative[None, :]
                reconstructed = apparent_rate(residue_u, time)
                peptide_frame_u = residue_u @ mapping.T
                peptide_inverse = apparent_rate(peptide_frame_u, time)
                candidates[f"oracle_inverse_uptake_t{time:g}"] = (mapping@weighted_covariance(reconstructed, weights)@mapping.T, "oracle", time)
                candidates[f"oracle_delta_pullback_t{time:g}"] = (mapping@pullback@mapping.T, "oracle", time)
                candidates[f"oracle_raw_uptake_t{time:g}"] = (mapping@cu@mapping.T, "oracle", time)
                candidates[f"oracle_peptide_first_inverse_t{time:g}"] = (weighted_covariance(peptide_inverse, weights), "oracle", time)
            for name, matrix in curve_covariances(mean_peptide_uptake, TIMEPOINTS).items():
                availability = "observable_shape_only" if name.endswith("normalized") else "observable"
                candidates[name] = (matrix, availability, None)

            survival_peptide = 1-mean_peptide_uptake
            pmean, pvar = cumulants_from_survival(survival_peptide, TIMEPOINTS)
            cumulant_rows.append({"weights": weight_name, "level": panel, "time_grid": "production", "mean_rate_relative_error": relative_error(pmean, mapping@mean_rate), "marginal_variance_relative_error": relative_error(pvar, np.diag(truth)), "marginal_correlation": correlation(pvar, np.diag(truth))})

            truth_profiles = {alpha: profiles(truth, alpha) for alpha in ALPHAS}
            for name, (candidate, availability, time) in candidates.items():
                matrices[f"{weight_name}__{panel}__{name}"] = candidate
                mm = matrix_metrics(candidate, truth)
                projected_distance = log_euclidean_distance(candidate, truth, projection)
                full_pass = availability == "observable" and 0.5 <= mm["trace_ratio"] <= 2 and mm["normalized_frobenius_distance"] <= 0.25 and mm["off_diagonal_correlation"] >= 0.80 and projected_distance <= 0.25
                metric_rows.append({"weights": weight_name, "panel": panel, "construction": name, "availability": availability, "timepoint": time, **mm, "projected_log_euclidean_distance": projected_distance, "full_gate_pass": bool(full_pass)})
                for alpha in ALPHAS:
                    cp = profiles(candidate, alpha)
                    for profile_name in ("marginal", "conditional"):
                        pm = profile_metrics(cp[profile_name], truth_profiles[alpha][profile_name])
                        gate = availability == "observable" and np.isclose(alpha, PRIMARY_ALPHA) and 0.5 <= pm["scale_ratio"] <= 2 and pm["pearson"] >= 0.90 and pm["spearman"] >= 0.90 and pm["log_rmse"] <= 0.25
                        profile_rows.append({"weights": weight_name, "panel": panel, "construction": name, "availability": availability, "timepoint": time, "alpha": alpha, "profile": profile_name, **pm, "profile_gate_pass": bool(gate)})

    metrics, profile_frame = pd.DataFrame(metric_rows), pd.DataFrame(profile_rows)
    # Sensitivity is required on every panel for promotion; observable candidates only.
    promotions = []
    for (weight_name, construction), rows in metrics.groupby(["weights", "construction"]):
        primary_profiles = profile_frame[(profile_frame.weights == weight_name)&(profile_frame.construction == construction)&np.isclose(profile_frame.alpha, PRIMARY_ALPHA)]
        full = bool(len(rows) == 3 and rows.full_gate_pass.all())
        for representation in ("marginal", "conditional"):
            primary = primary_profiles[primary_profiles.profile == representation]
            stable = True
            for panel in rows.panel.unique():
                base = primary[primary.panel == panel]
                if len(base) != 1 or not bool(base.iloc[0].profile_gate_pass): stable = False
                for alpha in (0.01, 0.10):
                    alt = profile_frame[(profile_frame.weights == weight_name)&(profile_frame.construction == construction)&(profile_frame.panel == panel)&(profile_frame.profile == representation)&np.isclose(profile_frame.alpha, alpha)]
                    matrix_key = f"{weight_name}__{panel}__{construction}"
                    if len(alt) != 1 or matrix_key not in matrices:
                        stable = False
                    else:
                        primary_values = profiles(matrices[matrix_key], PRIMARY_ALPHA)[representation]
                        alternative_values = profiles(matrices[matrix_key], alpha)[representation]
                        sensitivity = profile_metrics(alternative_values, primary_values)
                        if sensitivity["pearson"] < 0.95 or sensitivity["log_rmse"] > 0.10: stable = False
            promotions.append({"weights": weight_name, "construction": construction, "representation": representation, "promoted": bool(stable)})
        promotions.append({"weights": weight_name, "construction": construction, "representation": "full", "promoted": full})
    promotions = pd.DataFrame(promotions)

    construction_registry().to_csv(args.output_dir/"construction_registry.csv", index=False)
    metrics.to_csv(args.output_dir/"matrix_metrics.csv", index=False)
    profile_frame.to_csv(args.output_dir/"profile_metrics.csv", index=False)
    pd.DataFrame(permutation_rows).to_csv(args.output_dir/"identifiability_permutation.csv", index=False)
    pd.DataFrame(cumulant_rows).to_csv(args.output_dir/"cumulant_metrics.csv", index=False)
    pd.DataFrame(mean_rows).to_csv(args.output_dir/"mean_rate_metrics.csv", index=False)
    promotions.to_csv(args.output_dir/"promotion_gates.csv", index=False)
    np.savez_compressed(args.output_dir/"covariance_matrices.npz", **matrices)

    known_metrics = metrics[metrics.weights == "known_40_60"]
    known_promotions = promotions[(promotions.weights == "known_40_60") & promotions.promoted]
    permutation = pd.DataFrame(permutation_rows)
    cumulants = pd.DataFrame(cumulant_rows)
    lines = ["# Uptake-to-effective-rate covariance litmus", "", "## Question", "", "Can a covariance construction derived from an ordinary five-timepoint peptide uptake curve reproduce the weighted frame covariance of linear BV effective rates `k = k_int exp(-z)`?", "", "## Ground-truth and controls", "", "The primary bar is `M Cov_w(k) M^T`. Oracle framewise uptake is an upper-bound control; it is not described as experimentally extractable. All nonlinear transforms precede peptide mapping unless a construction is explicitly labelled peptide-first.", "", "## Fundamental identifiability control", ""]
    for row in permutation[permutation.weights == "known_40_60"].itertuples(index=False):
        lines.append(f"- **{row.panel}**: independent within-equal-weight frame permutations changed the target mean uptake by at most `{row.mean_uptake_max_abs_change:.2e}`, while changing full/marginal/conditional mapped rate covariance by `{row.full_covariance_relative_change:.3f}` / `{row.marginal_relative_change:.3f}` / `{row.conditional_relative_change:.3f}`.")
    controls = known_metrics[known_metrics.construction.isin(["truth_exact_rate", "exact_inverse_pf_scaled", "oracle_inverse_uptake_asymptotic", "control_logpf", "control_kint_scaled_logpf", "gaussian_rate_closure"])]
    lines += ["", "Thus mean uptake does not uniquely identify inter-amide covariance: the same complete uptake curves can arise from different joint frame couplings. Marginal amide rate variance is present in survival-curve curvature in principle, but peptide aggregation mixes it with unobserved cross-amide covariance.", "", "## Control behavior", "", "| Construction | Raw error | Normalized distance | Off-diagonal correlation | Trace ratio |", "|---|---:|---:|---:|---:|"]
    for name, group in controls.groupby("construction"):
        lines.append(f"| {name} | {group.raw_relative_error.mean():.3g} | {group.normalized_frobenius_distance.mean():.3g} | {group.off_diagonal_correlation.mean():.3f} | {group.trace_ratio.mean():.3g} |")
    lines += ["", "The exact inverse-PF construction and the automatically unsaturated uptake inversion verify the implementation. Production timepoints can already be saturated, so exact algebraic inversion is not numerically informative there.", "", "## Observable-only correspondence", "", "| Construction | Full norm. distance | Off-diag. corr. | Trace ratio | Marginal corr./log-RMSE | Conditional corr./log-RMSE | Gates (F/M/C) |", "|---|---:|---:|---:|---:|---:|---|"]
    for name, group in known_metrics[known_metrics.availability.str.startswith("observable")].groupby("construction"):
        pp = profile_frame[(profile_frame.weights == "known_40_60")&(profile_frame.construction == name)&np.isclose(profile_frame.alpha, PRIMARY_ALPHA)]
        marginal = pp[pp.profile == "marginal"]
        conditional = pp[pp.profile == "conditional"]
        promoted = known_promotions[known_promotions.construction == name]
        status = {row.representation: bool(row.promoted) for row in promotions[(promotions.weights == "known_40_60")&(promotions.construction == name)].itertuples(index=False)}
        lines.append(f"| {name} | {group.normalized_frobenius_distance.mean():.3f} | {group.off_diagonal_correlation.mean():.3f} | {group.trace_ratio.mean():.3g} | {marginal.pearson.mean():.3f}/{marginal.log_rmse.mean():.3f} | {conditional.pearson.mean():.3f}/{conditional.log_rmse.mean():.3f} | {'P' if status.get('full') else 'F'}/{'P' if status.get('marginal') else 'F'}/{'P' if status.get('conditional') else 'F'} |")
    lines += ["", "## Cumulant ceiling", "", "| Level | Grid | Mean-rate error | Marginal-variance error | Marginal correlation |", "|---|---|---:|---:|---:|"]
    for row in cumulants[cumulants.weights == "known_40_60"].itertuples(index=False):
        lines.append(f"| {row.level} | {row.time_grid} | {row.mean_rate_relative_error:.3g} | {row.marginal_variance_relative_error:.3g} | {row.marginal_correlation:.3f} |")
    lines += ["", "## Promotion result", "", ("No observable-only construction passed the registered gates." if known_promotions.empty else "Promoted: " + ", ".join(f"{r.construction} ({r.representation})" for r in known_promotions.itertuples(index=False))), "", "Promotion is a correspondence result only. Decoy rejection, gradient alignment, panel design, and fitting were not run.", "", "## Checkpoint", "", "Stage 2 remains **pending user review**."]
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n")
    gates = {"known_40_60_promotions": known_promotions.to_dict("records"), "mean_uptake_permutation_invariance_max": float(permutation.mean_uptake_max_abs_change.max()), "stage2_status": "pending_user_review"}
    (args.output_dir/"gate_results.json").write_text(json.dumps(gates, indent=2)+"\n")
    manifest = {"stage": "effective_rate_covariance_correspondence", "status": "complete_pending_user_review", "fitting_performed": False, "decoy_or_gradient_tests_performed": False, "panel_design_performed": False, "truth_coordinate": "linear effective rate k_int*exp(-log_pf)", "timepoints": TIMEPOINTS.tolist(), "primary_alpha": PRIMARY_ALPHA, "alpha_sensitivity": list(ALPHAS), "inputs": inputs}
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    panel = list(mappings)[0]
    truth = matrices[f"known_40_60__{panel}__truth_exact_rate"]
    shown = [truth, matrices[f"known_40_60__{panel}__curve_apparent_rate"], matrices[f"known_40_60__{panel}__oracle_peptide_first_inverse_t1"]]
    for axis, matrix, title in zip(axes, shown, ("Exact rate truth", "Curve apparent-rate", "Oracle peptide-first")):
        image = axis.imshow(matrix/(np.trace(matrix) or 1), cmap="coolwarm")
        axis.set_title(title); figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout(); figure.savefig(args.output_dir/"covariance_correspondence.png", dpi=180); plt.close(figure)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=root/"_pf_peptide_moment_final")
    parser.add_argument("--output-dir", type=Path, default=root/"_uptake_rate_covariance_litmus")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
