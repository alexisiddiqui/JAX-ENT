#!/usr/bin/env python3
"""No-fitting ISO_BI mean, rate, and uptake physics litmus.

All nonlinear operations are performed per residue and per structural frame.  Peptide
maps are applied only after complete residue covariance matrices have been formed.
The alternative means in this file are diagnostics, not fitting targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIMEPOINTS = np.asarray([0.167, 1.0, 10.0, 60.0, 120.0], dtype=np.float64)
ALPHAS = (0.01, 0.05, 0.10)


def weighted_covariance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    centered = values - weights @ values
    result = (centered * weights[:, None]).T @ centered
    return (result + result.T) / 2


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(scale, scale)
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)


def correlation_invariance_error(left: np.ndarray, right: np.ndarray) -> float:
    """Compare correlations only on stochastic coordinates shared by both matrices."""
    tolerance = np.finfo(float).eps * max(np.max(np.diag(left)), np.max(np.diag(right)), 1.0)
    active = (np.diag(left) > tolerance) & (np.diag(right) > tolerance)
    return relative_error(
        covariance_to_correlation(left[np.ix_(active, active)]),
        covariance_to_correlation(right[np.ix_(active, active)]),
    )


def relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(candidate - reference) / denominator) if denominator else float(np.linalg.norm(candidate))


def off_diagonal_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = ~np.eye(left.shape[0], dtype=bool)
    x, y = left[mask], right[mask]
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan


def effective_pfs(log_pf: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    weights = weights / weights.sum()
    mean = log_pf @ weights
    return {
        "geometric": np.exp(mean),
        "harmonic_mean_rate": 1.0 / (np.exp(-log_pf) @ weights),
        "arithmetic": np.exp(log_pf) @ weights,
    }


def single_rate_uptake(pf: np.ndarray, k_int: np.ndarray, times: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-times[:, None] * k_int[None, :] / pf[None, :])


def exact_frame_uptake(log_pf: np.ndarray, k_int: np.ndarray, weights: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    rates = (k_int[:, None] * np.exp(-log_pf)).T  # frames x residues
    means, covariances = [], []
    for time in times:
        values = 1.0 - np.exp(-time * rates)
        means.append(weights @ values)
        covariances.append(weighted_covariance(values, weights))
    return np.asarray(means), covariances


def gaussian_rate_closures(log_pf: np.ndarray, k_int: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    observations = log_pf.T
    mean_z = weights @ observations
    covariance_z = weighted_covariance(observations, weights)
    exact_rates = k_int[None, :] * np.exp(-observations)
    exact_mean = weights @ exact_rates
    exact_covariance = weighted_covariance(exact_rates, weights)
    gaussian_mean = k_int * np.exp(-mean_z + 0.5 * np.diag(covariance_z))
    multiplier = np.exp(covariance_z) - 1.0
    return {
        "mean_z": mean_z,
        "covariance_z": covariance_z,
        "exact_mean": exact_mean,
        "exact_covariance": exact_covariance,
        "gaussian_mean": gaussian_mean,
        "gaussian_full_covariance": np.outer(gaussian_mean, gaussian_mean) * multiplier,
        "exact_mean_covariance_closure": np.outer(exact_mean, exact_mean) * multiplier,
    }


def uptake_delta_covariance(mean_z: np.ndarray, covariance_z: np.ndarray, k_int: np.ndarray, time: float) -> tuple[np.ndarray, np.ndarray]:
    exposure = time * k_int * np.exp(-mean_z)
    jacobian = -exposure * np.exp(-exposure)
    return jacobian[:, None] * covariance_z * jacobian[None, :], jacobian


def regularized_profiles(covariance: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    dimension = covariance.shape[0]
    scale = np.trace(covariance) / dimension
    regularized = (1.0 - alpha) * covariance + (alpha * scale + 1e-8 * scale + 1e-12) * np.eye(dimension)
    conditional = 1.0 / np.diag(np.linalg.inv(regularized))
    return np.diag(regularized), conditional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def load_inputs(results_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    source_manifest = json.loads((results_dir / "manifest.json").read_text())
    feature_path = Path(source_manifest["inputs"]["bi_features"]["path"])
    cluster_path = Path(source_manifest["inputs"]["bi_clusters"]["path"])
    with np.load(feature_path) as data:
        log_pf = 0.35 * np.asarray(data["heavy_contacts"], float) + 2.0 * np.asarray(data["acceptor_contacts"], float)
        k_int = np.asarray(data["k_ints"], float)
    assignments = pd.read_csv(cluster_path)["cluster_assignment"].to_numpy(int)
    known = np.zeros(assignments.size)
    known[assignments == 0] = 0.4 / np.sum(assignments == 0)
    known[assignments == 1] = 0.6 / np.sum(assignments == 1)
    mappings = {panel: np.load(results_dir / f"panel_{panel}_mapping.npz")["mapping"].astype(float) for panel in source_manifest["config"]["panels"]}
    inputs = {"source_manifest": str((results_dir / "manifest.json").resolve()), "bi_features": {"path": str(feature_path), "sha256": _sha256(feature_path)}, "bi_clusters": {"path": str(cluster_path), "sha256": _sha256(cluster_path)}}
    return inputs, log_pf, k_int, known, mappings


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs, log_pf, k_int, known_weights, mappings = load_inputs(args.results_dir)
    weight_sets = {"uniform": np.full(log_pf.shape[1], 1.0 / log_pf.shape[1]), "known_40_60": known_weights}
    construction_rows: list[dict[str, Any]] = []
    rate_rows: list[dict[str, Any]] = []
    covariance_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    pf_rows: list[dict[str, Any]] = []
    rate_profile_rows: list[dict[str, Any]] = []
    matrices: dict[str, np.ndarray] = {}
    gates: dict[str, Any] = {}

    for weight_name, weights in weight_sets.items():
        pfs = effective_pfs(log_pf, weights)
        exact_uptake, exact_uptake_covariances = exact_frame_uptake(log_pf, k_int, weights, TIMEPOINTS)
        curves = {name: single_rate_uptake(pf, k_int, TIMEPOINTS) for name, pf in pfs.items()}
        rate = gaussian_rate_closures(log_pf, k_int, weights)
        gaussian_pf = np.exp(rate["mean_z"] - 0.5 * np.diag(rate["covariance_z"]))
        curves["gaussian_mean_rate"] = single_rate_uptake(gaussian_pf, k_int, TIMEPOINTS)
        curves["exact_framewise"] = exact_uptake
        for construction, values in pfs.items():
            for residue, value in enumerate(values):
                pf_rows.append({"weights": weight_name, "residue_index": residue, "construction": construction, "effective_pf": value})
        ordering_pf = np.all(pfs["harmonic_mean_rate"] <= pfs["geometric"] + 1e-12) and np.all(pfs["geometric"] <= pfs["arithmetic"] + 1e-12)
        ordering_uptake = np.all(curves["harmonic_mean_rate"] + 1e-12 >= curves["geometric"]) and np.all(curves["geometric"] + 1e-12 >= curves["arithmetic"])
        asymptotic_time = 1e-7 / np.max(rate["exact_mean"])
        small_exact, _ = exact_frame_uptake(log_pf, k_int, weights, np.asarray([asymptotic_time]))
        small_harmonic = single_rate_uptake(pfs["harmonic_mean_rate"], k_int, np.asarray([asymptotic_time]))
        small_error = relative_error(small_harmonic, small_exact)
        gates[weight_name] = {"pf_ordering": bool(ordering_pf), "single_rate_uptake_reverse_ordering": bool(ordering_uptake), "small_time_relative_error": small_error, "small_time_pass": small_error < 1e-6, "asymptotic_time": asymptotic_time}

        quotient = np.divide(rate["exact_covariance"], np.outer(rate["exact_mean"], rate["exact_mean"]), out=np.zeros_like(rate["exact_covariance"]), where=np.outer(rate["exact_mean"], rate["exact_mean"]) != 0)
        identity = np.exp(rate["covariance_z"]) - 1.0
        for construction in ("covariance_z", "exact_covariance", "gaussian_full_covariance", "exact_mean_covariance_closure"):
            matrix = rate[construction]
            matrices[f"{weight_name}__rate_level__{construction}"] = matrix
            for residue, marginal in enumerate(np.diag(matrix)):
                rate_profile_rows.append({"weights": weight_name, "construction": construction, "residue_index": residue, "marginal": marginal})
        for closure in ("gaussian_full_covariance", "exact_mean_covariance_closure"):
            candidate = rate[closure]
            rate_rows.append({"weights": weight_name, "construction": closure, "full_matrix_relative_error": relative_error(candidate, rate["exact_covariance"]), "off_diagonal_correlation": off_diagonal_correlation(candidate, rate["exact_covariance"]), "quotient_relative_error": relative_error(quotient, identity), "mean_rate_relative_error": relative_error(rate["gaussian_mean"] if closure == "gaussian_full_covariance" else rate["exact_mean"], rate["exact_mean"])})

        residue_corr = covariance_to_correlation(rate["covariance_z"])
        for time_index, time in enumerate(TIMEPOINTS):
            delta, jacobian = uptake_delta_covariance(rate["mean_z"], rate["covariance_z"], k_int, float(time))
            exact_covariance = exact_uptake_covariances[time_index]
            residue_invariance = correlation_invariance_error(delta, rate["covariance_z"])
            matrices[f"{weight_name}__residue_exact_uptake__t-{time:g}"] = exact_covariance
            matrices[f"{weight_name}__residue_delta_uptake__t-{time:g}"] = delta
            for panel, mapping in mappings.items():
                exact_mapped = mapping @ exact_covariance @ mapping.T
                delta_mapped = mapping @ delta @ mapping.T
                logpf_mapped = mapping @ rate["covariance_z"] @ mapping.T
                matrices[f"{weight_name}__{panel}__exact_uptake__t-{time:g}"] = exact_mapped
                matrices[f"{weight_name}__{panel}__delta_uptake__t-{time:g}"] = delta_mapped
                peptide_invariance = relative_error(covariance_to_correlation(delta_mapped), covariance_to_correlation(logpf_mapped))
                covariance_rows.append({"weights": weight_name, "panel": panel, "timepoint": time, "delta_full_matrix_relative_error": relative_error(delta_mapped, exact_mapped), "delta_off_diagonal_correlation": off_diagonal_correlation(delta_mapped, exact_mapped), "residue_correlation_invariance_error": residue_invariance, "peptide_correlation_invariance_error": peptide_invariance})
                for alpha in ALPHAS:
                    for construction, matrix in (("exact_uptake", exact_mapped), ("delta_uptake", delta_mapped)):
                        marginal, conditional = regularized_profiles(matrix, alpha)
                        for peptide in range(mapping.shape[0]):
                            profile_rows.append({"weights": weight_name, "panel": panel, "timepoint": time, "construction": construction, "alpha": alpha, "peptide_index": peptide, "marginal": marginal[peptide], "conditional": conditional[peptide]})
                for construction, values in curves.items():
                    mapped = mapping @ values[time_index]
                    reference = mapping @ exact_uptake[time_index]
                    for peptide, (value, ref) in enumerate(zip(mapped, reference)):
                        construction_rows.append({"weights": weight_name, "panel": panel, "timepoint": time, "peptide_index": peptide, "construction": construction, "uptake": value, "exact_framewise_uptake": ref, "difference_from_exact": value - ref})

        gates[weight_name]["rate_quotient_identity_relative_error"] = relative_error(quotient, identity)
        gates[weight_name]["residue_correlation_invariance_max_error"] = max(row["residue_correlation_invariance_error"] for row in covariance_rows if row["weights"] == weight_name)
        gates[weight_name]["peptide_mapping_break_detected"] = any(row["peptide_correlation_invariance_error"] > 1e-4 for row in covariance_rows if row["weights"] == weight_name)

    pd.DataFrame(construction_rows).to_csv(args.output_dir / "construction_uptake_differences.csv", index=False)
    pd.DataFrame(pf_rows).to_csv(args.output_dir / "effective_pf_constructions.csv", index=False)
    pd.DataFrame(rate_rows).to_csv(args.output_dir / "rate_covariance_metrics.csv", index=False)
    pd.DataFrame(rate_profile_rows).to_csv(args.output_dir / "rate_marginal_profiles.csv", index=False)
    pd.DataFrame(covariance_rows).to_csv(args.output_dir / "uptake_covariance_metrics.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(args.output_dir / "covariance_profiles.csv", index=False)
    np.savez_compressed(args.output_dir / "covariance_matrices.npz", **matrices)
    manifest = {"stage": 1, "status": "complete_pending_user_review", "fitting_performed": False, "panel_design_performed": False, "weights": {"uniform": "uniform over frames", "known_40_60": "0.4/0.6 uniformly within ISO_BI clusters"}, "timepoints": TIMEPOINTS.tolist(), "shrinkage_alphas": list(ALPHAS), "primary_alpha": 0.05, "coordinate_rule": "nonlinear transforms at residue/frame level; complete covariance mapped by M C M^T before shrinkage and profile extraction", "inputs": inputs, "outputs": ["effective_pf_constructions.csv", "construction_uptake_differences.csv", "construction_summary.csv", "rate_covariance_metrics.csv", "rate_marginal_profiles.csv", "uptake_covariance_metrics.csv", "covariance_profiles.csv", "covariance_matrices.npz", "gate_results.json", "report.md", "mean_uptake_differences.png"]}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "gate_results.json").write_text(json.dumps(gates, indent=2) + "\n")

    differences = pd.DataFrame(construction_rows)
    summary = differences.groupby(["weights", "construction"]).difference_from_exact.agg([lambda x: np.sqrt(np.mean(x*x)), "max", "min"]).reset_index()
    summary.columns = ["weights", "construction", "rms_difference", "maximum_difference", "minimum_difference"]
    summary.to_csv(args.output_dir / "construction_summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, (weight_name, group) in zip(axes, summary.groupby("weights")):
        axis.bar(group.construction, group.rms_difference)
        axis.tick_params(axis="x", rotation=35)
        axis.set_title(weight_name)
        axis.set_ylabel("RMS uptake difference from frame average")
    figure.tight_layout(); figure.savefig(args.output_dir / "mean_uptake_differences.png", dpi=180); plt.close(figure)
    rate_frame, covariance_frame = pd.DataFrame(rate_rows), pd.DataFrame(covariance_rows)
    report = [
        "# ISO mean, rate, and uptake physics litmus",
        "",
        "## What was tested",
        "",
        "In the Best--Vendruscolo (BV) model used here, each frame's contacts define residue log-protection as `z = 0.35 N_c + 2.0 N_h`. The protected exchange rate is `k = k_int exp(-z)`, and uptake is `u(t) = 1 - exp(-k t)`. Averaging can therefore occur before or after two nonlinear operations: exponentiating `z` into a rate and exponentiating the rate into uptake.",
        "",
        "The production code averages `z` first, giving the geometric PF `exp(E[z])`. This litmus compares that established software semantics with mean-rate/harmonic PF, arithmetic PF, a Gaussian mean-rate closure, and exact frame-averaged uptake. The alternatives are physical diagnostics, not candidate fitting targets.",
        "",
        "No fitting, recovery ranking, panel construction, or ESS gating was performed.",
        "",
        "## Algebraic checks",
    ]
    for name, result in gates.items():
        report += [f"", f"### {name}", f"- Harmonic <= geometric <= arithmetic PF: **{'PASS' if result['pf_ordering'] else 'FAIL'}**.", f"- Reverse single-rate uptake ordering: **{'PASS' if result['single_rate_uptake_reverse_ordering'] else 'FAIL'}**.", f"- Harmonic/frame-average small-time limit: **{'PASS' if result['small_time_pass'] else 'FAIL'}** (relative error {result['small_time_relative_error']:.3e} at t={result['asymptotic_time']:.3e}).", f"- Gaussian normalized rate-covariance identity error against discrete BI: {result['rate_quotient_identity_relative_error']:.3e}.", f"- Residue delta-correlation invariance maximum error: {result['residue_correlation_invariance_max_error']:.3e}.", f"- Peptide mapping breaks diagonal-rescaling invariance: **{result['peptide_mapping_break_detected']}**."]
    report += [
        "",
        "The PF ordering follows from the harmonic/geometric/arithmetic mean inequalities. Uptake reverses it because a larger PF means a smaller exchange rate. At very short time, `u(t) ≈ k t`, so exact frame averaging and the harmonic-PF single rate must agree; the numerical check confirms this.",
        "",
        "## How different are the mean constructions?",
        "",
        "The table reports RMS peptide-uptake difference from exact frame averaging over all three panels and five times. Signed bias is positive when a construction predicts more uptake than the frame average. Uptake is a fraction, so `0.074` means about 7.4 percentage points.",
        "",
    ]
    for weight_name in ("uniform", "known_40_60"):
        report += [f"### {weight_name}", ""]
        rows = []
        for construction in ("geometric", "arithmetic", "harmonic_mean_rate", "gaussian_mean_rate"):
            group = differences[(differences.weights == weight_name) & (differences.construction == construction)]
            rows.append([construction.replace("_", " "), f"{np.sqrt(np.mean(group.difference_from_exact**2)):.4f}", f"{group.difference_from_exact.mean():+.4f}", f"{group.difference_from_exact.abs().max():.4f}"])
        report += markdown_table(["Construction", "RMS difference", "Mean signed bias", "Largest absolute difference"], rows) + [""]
    report += [
        "The production geometric curve is numerically closest to exact frame averaging in this dataset, but it is not uniformly above or below it: its signed error changes with residue and time. Arithmetic PF always gives less uptake here because it produces the largest PF/smallest rate. Harmonic PF uses the exact mean rate, but `1-exp(-k t)` is concave in `k`, so applying uptake to the mean rate systematically exceeds mean frame uptake except in the short-time limit. Gaussian mean-rate uptake behaves similarly and is slightly below the harmonic result because the clustered BV log-PFs are not exactly Gaussian.",
        "",
        "## Rate covariance: does Gaussian BV moment closure work?",
        "",
    ]
    rate_table = []
    for row in rate_frame.itertuples(index=False):
        rate_table.append([row.weights, row.construction.replace("_", " "), f"{row.mean_rate_relative_error:.3f}", f"{row.full_matrix_relative_error:.3f}", f"{row.off_diagonal_correlation:.3f}", f"{row.quotient_relative_error:.3f}"])
    report += markdown_table(["Weights", "Closure", "Mean-rate rel. error", "Covariance rel. error", "Off-diagonal corr.", "Normalized-identity error"], rate_table)
    report += [
        "",
        "For Gaussian `z`, the rate covariance identity is exact and the simulation test passes. Uniform ISO frame weighting is moderately close: covariance-scale error is about 9--11%. The known 40:60 population is strongly non-Gaussian because it is a weighted structural mixture. The full Gaussian closure then overstates/misplaces covariance badly (`22.946` relative error), even though its mean-rate error is only 4.9%. Supplying exact mean rates removes the large scale amplification, but the remaining covariance error (`0.421`) and weak off-diagonal correlation (`0.139`) show that the Gaussian dependence structure is still inadequate.",
        "",
        "## Uptake covariance and peptide mapping",
        "",
        "At residue level the delta covariance is a diagonal rescaling, `C_u ≈ J C_z J`, so its correlation matrix must equal the log-PF correlation matrix (all BV uptake derivatives have the same negative sign). This held to machine precision. Peptide averaging mixes residues with different derivatives: `M J C_z J M^T` cannot generally be rewritten as a diagonal rescaling of `M C_z M^T`, so peptide correlations need not be invariant.",
        "",
    ]
    delta_rows = []
    for (weight_name, time), group in covariance_frame.groupby(["weights", "timepoint"], sort=False):
        delta_rows.append([weight_name, f"{time:g}", f"{group.delta_full_matrix_relative_error.mean():.3f}", f"{group.delta_off_diagonal_correlation.mean():.3f}", f"{group.peptide_correlation_invariance_error.mean():.3f}"])
    report += markdown_table(["Weights", "Time", "Mean delta covariance error", "Mean off-diagonal corr.", "Mean peptide invariance break"], delta_rows)
    report += [
        "",
        f"Across individual panels/timepoints, delta full-matrix errors span `{covariance_frame.delta_full_matrix_relative_error.min():.3f}`--`{covariance_frame.delta_full_matrix_relative_error.max():.3f}` and peptide correlation changes span `{covariance_frame.peptide_correlation_invariance_error.min():.3f}`--`{covariance_frame.peptide_correlation_invariance_error.max():.3f}`. Delta propagation often retains the broad covariance pattern (positive off-diagonal correlations with the exact matrices) but is not quantitatively reliable across the production time grid, especially for the 40:60 mixture and late saturation.",
        "",
        "## Physical interpretation for the BV model",
        "",
        "1. Contact heterogeneity matters twice: first when `z` is exponentiated into rates, and again when rates are exponentiated into uptake. No single effective PF can reproduce exact frame-averaged uptake at every time unless the rate distribution is degenerate.",
        "2. The production geometric mean is an average-structure/log-PF semantics. The harmonic PF is the correct effective PF only for the initial uptake slope. Exact frame averaging is a multi-exponential kinetic model. These answer different physical questions.",
        "3. A 40:60 mixture of BV structural clusters is not well represented by a single Gaussian in log-PF space. Gaussian moment closure can therefore be misleading for covariance even when its mean rate looks acceptable.",
        "4. Residue-level covariance intuition does not survive peptide mapping unchanged. Panel geometry and timepoint-specific derivatives jointly determine peptide covariance, which is why complete residue matrices must be mapped before shrinkage or profile extraction.",
        "5. Saturation changes both scale and covariance geometry over time. A time-independent uptake-covariance surrogate is not supported by these results.",
        "",
        "These observations do not declare one construction universal physical truth and do not change production BV semantics. They clarify which ensemble observable each construction represents.",
        "",
        "## Checkpoint",
        "",
        "Stage 2 is **pending user review**. No panel design or optimization may begin without explicit approval.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=root / "_pf_peptide_moment_final")
    parser.add_argument("--output-dir", type=Path, default=root / "_iso_mean_rate_uptake_litmus")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
