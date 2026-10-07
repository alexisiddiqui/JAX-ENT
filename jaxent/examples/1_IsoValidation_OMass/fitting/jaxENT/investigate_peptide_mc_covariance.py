#!/usr/bin/env python3
"""Stage 1B: can a peptide-uptake-space covariance be estimated from finite MC target draws?

No fitting, no optimization, no decoy test, no cluster recovery. Additive Gaussian residuals
are drawn in the observed peptide-uptake coordinate around a target mean curve; the question
is purely whether the supplied covariance is recoverable from N draws.

Convergence to the imposed Sigma_target is a sampling result. It does not establish
correspondence with conformational effective-rate covariance; that correspondence is reported
separately against an ISO oracle and never gates the sampling result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_litmus():
    """Reuse the registered metrics/constructions from the correspondence litmus."""
    path = Path(__file__).resolve().parent / "investigate_uptake_rate_covariance.py"
    spec = importlib.util.spec_from_file_location("_uptake_rate_covariance_litmus_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LITMUS = _load_litmus()
TIMEPOINTS = LITMUS.TIMEPOINTS
ALPHAS = LITMUS.ALPHAS
PRIMARY_ALPHA = LITMUS.PRIMARY_ALPHA

SAMPLE_SIZES = (25, 50, 100, 250, 500, 1000, 5000)
SEEDS = tuple(range(20))
SCALES = (0.0, 0.1, 1.0)
PRIMARY_SCALE = 1.0
SAMPLING_MODELS = ("peptide_by_peptide", "stacked_peptide_by_time")
NORMALIZATIONS = ("population", "sample")
MAX_GATE_N = 1000
BASE_SEED = 20260716

GATES = {
    "trace_ratio_low": 0.9,
    "trace_ratio_high": 1.1,
    "max_normalized_distance": 0.10,
    "min_off_diagonal_correlation": 0.90,
    "min_profile_correlation": 0.95,
    "max_profile_log_rmse": 0.10,
    "min_seed_pass_fraction": 0.90,
    "max_bound_violation_fraction": 0.01,
}


def psd_factor(covariance: np.ndarray) -> np.ndarray:
    """Return L with L L^T == covariance, tolerating the rank-deficient curve matrices.

    The curve constructions have rank at most T-1, so a plain Cholesky fails. An eigenvalue
    floor at zero is used instead of a ridge because a ridge would change the covariance that
    is actually supplied to the sampler, which is exactly what the gates measure against.
    """
    values, vectors = np.linalg.eigh((covariance + covariance.T) / 2)
    return vectors * np.sqrt(np.maximum(values, 0.0))[None, :]


def draw_standard_normal(factor: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Return count x dim draws from N(0, L L^T) using the supplied generator."""
    return rng.standard_normal((count, factor.shape[1])) @ factor.T


def sample_covariance(draws: np.ndarray, normalization: str) -> np.ndarray:
    """Covariance of draws about their own mean; population has no Bessel correction."""
    count = draws.shape[0]
    centered = draws - draws.mean(axis=0, keepdims=True)
    divisor = count if normalization == "population" else max(count - 1, 1)
    result = centered.T @ centered / divisor
    return (result + result.T) / 2


def bound_violation_fraction(values: np.ndarray) -> float:
    """Fraction of uptake entries outside [0, 1]. Entries are counted, never clipped."""
    values = np.asarray(values, float)
    return float(np.mean((values < 0.0) | (values > 1.0)))


def time_covariance(observations: np.ndarray) -> np.ndarray:
    """T x T covariance of an observable transform, averaged over peptides."""
    centered = observations - observations.mean(axis=1, keepdims=True)
    result = centered @ centered.T / observations.shape[1]
    return (result + result.T) / 2


def observable_transforms(mean_peptide_uptake: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """The five registered observable transforms of the T x P mean curve."""
    uptake = np.asarray(mean_peptide_uptake, float)
    tiny = np.finfo(float).tiny
    survival_log = np.log(np.clip(1.0 - uptake, tiny, 1.0))
    adjacent = -(survival_log[1:] - survival_log[:-1]) / np.diff(times)[:, None]
    scale = np.sqrt(np.mean(uptake**2, axis=1, keepdims=True))
    return {
        "curve_raw_uptake": uptake,
        "curve_uptake_over_t": uptake / times[:, None],
        "curve_apparent_rate": LITMUS.apparent_rate(uptake, times[:, None]),
        "curve_adjacent_survival_slope": adjacent,
        "curve_timepoint_rms_normalized": uptake / np.maximum(scale, tiny),
    }


def stacked_targets(mean_peptide_uptake: np.ndarray, times: np.ndarray) -> dict[str, np.ndarray]:
    """Separable (T*P) x (T*P) candidates C_time kron Sigma_peptide.

    A single mean curve cannot furnish a general stacked covariance: there is one observation
    of the vectorized curve. The separable Kronecker form is the constructible candidate that
    still carries non-trivial cross-time blocks, and is registered as such rather than being
    presented as a general stacked estimate.

    The adjacent-survival-slope transform is excluded: it lives on T-1 intervals, so it cannot
    form a covariance of the T-length uptake curve that is actually drawn.
    """
    peptide = LITMUS.curve_covariances(mean_peptide_uptake, times)
    result = {}
    for name, transform in observable_transforms(mean_peptide_uptake, times).items():
        if len(transform) != len(times):
            continue
        result[name] = np.kron(time_covariance(transform), peptide[name])
    return result


def construction_registry() -> pd.DataFrame:
    rows = []
    formulas = {
        "curve_raw_uptake": "Cov_t(y(t))",
        "curve_uptake_over_t": "Cov_t(y(t)/t)",
        "curve_apparent_rate": "Cov_t(-log(1-y(t))/t)",
        "curve_adjacent_survival_slope": "Cov_intervals(-Delta log(1-y)/Delta t)",
        "curve_timepoint_rms_normalized": "Cov_t(y/RMS_peptide(y_t))",
    }
    for name, formula in formulas.items():
        availability = "observable_shape_only" if name.endswith("normalized") else "observable"
        rows.append(
            (
                name,
                "peptide_by_peptide",
                availability,
                formula,
                "P x P; one independent residual vector per timepoint per replicate",
            )
        )
        if name == "curve_adjacent_survival_slope":
            # Lives on T-1 intervals; cannot be a covariance of the T-length drawn curve.
            continue
        rows.append(
            (
                name,
                "stacked_peptide_by_time",
                availability,
                f"Cov_time({formula}) kron {formula}",
                "(T*P) x (T*P) separable Kronecker; one draw per replicate over vec(Y)",
            )
        )
    return pd.DataFrame(rows, columns=["construction", "sampling_model", "availability", "formula", "note"])


def evaluate_estimate(estimate: np.ndarray, supplied: np.ndarray) -> dict[str, Any]:
    """Full-matrix and profile metrics against the covariance actually supplied."""
    row: dict[str, Any] = dict(LITMUS.matrix_metrics(estimate, supplied))
    row["full_gate_pass"] = bool(
        GATES["trace_ratio_low"] <= row["trace_ratio"] <= GATES["trace_ratio_high"]
        and row["normalized_frobenius_distance"] <= GATES["max_normalized_distance"]
        and row["off_diagonal_correlation"] >= GATES["min_off_diagonal_correlation"]
    )
    for alpha in ALPHAS:
        # Full matrix first, then shrinkage, then profiles. Never shrink a profile.
        estimate_profiles = LITMUS.profiles(estimate, alpha)
        supplied_profiles = LITMUS.profiles(supplied, alpha)
        for profile in ("marginal", "conditional"):
            metrics = LITMUS.profile_metrics(estimate_profiles[profile], supplied_profiles[profile])
            passed = bool(
                metrics["pearson"] >= GATES["min_profile_correlation"]
                and metrics["spearman"] >= GATES["min_profile_correlation"]
                and metrics["log_rmse"] <= GATES["max_profile_log_rmse"]
            )
            for key, value in metrics.items():
                row[f"{profile}_a{alpha:g}_{key}"] = value
            row[f"{profile}_a{alpha:g}_gate_pass"] = passed
    return row


def condition_seed(*parts: Any) -> int:
    """Deterministic per-condition seed derived from the stable string of its parts.

    A digest is used rather than the raw integer encoding of the string: truncating the latter
    keeps only its leading characters, which silently collides every trailing part, including
    the seed.
    """
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(f"{BASE_SEED}|{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2**32


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs, log_pf, k_int, assignments, known, mappings = LITMUS.load_inputs(args.results_dir)

    sample_sizes = (25, 100) if args.smoke else SAMPLE_SIZES
    seeds = SEEDS[:2] if args.smoke else SEEDS
    scales = (0.0, 1.0) if args.smoke else SCALES

    weight_sets = {"uniform": np.full(log_pf.shape[1], 1 / log_pf.shape[1]), "known_40_60": known}
    sample_rows, bound_rows, oracle_rows = [], [], []
    matrices: dict[str, np.ndarray] = {}

    for weight_name, weights in weight_sets.items():
        rates = LITMUS.effective_rates(log_pf, k_int)
        rate_covariance = LITMUS.weighted_covariance(rates, weights)
        mean_uptake_residue = np.einsum("tfr,f->tr", LITMUS.framewise_uptake(rates, TIMEPOINTS), weights)

        for panel, mapping in mappings.items():
            mean_peptide_uptake = mean_uptake_residue @ mapping.T
            oracle = mapping @ rate_covariance @ mapping.T
            targets = {
                "peptide_by_peptide": LITMUS.curve_covariances(mean_peptide_uptake, TIMEPOINTS),
                "stacked_peptide_by_time": stacked_targets(mean_peptide_uptake, TIMEPOINTS),
            }

            for model in SAMPLING_MODELS:
                for construction, base in targets[model].items():
                    matrices[f"{weight_name}__{panel}__{model}__{construction}"] = base
                    factor = psd_factor(base)

                    # Oracle physical correspondence, reported separately and never gated.
                    # Profiles are carried too: a construction can be perfectly estimable and
                    # still have a marginal/conditional profile unrelated to the oracle, and
                    # that is the question the sampling gates cannot see.
                    if model == "peptide_by_peptide":
                        row = {
                            "weights": weight_name,
                            "panel": panel,
                            "construction": construction,
                            **LITMUS.matrix_metrics(base, oracle),
                        }
                        for alpha in ALPHAS:
                            candidate_profiles = LITMUS.profiles(base, alpha)
                            oracle_profiles = LITMUS.profiles(oracle, alpha)
                            for profile in ("marginal", "conditional"):
                                metrics = LITMUS.profile_metrics(
                                    candidate_profiles[profile], oracle_profiles[profile]
                                )
                                for key, value in metrics.items():
                                    row[f"{profile}_a{alpha:g}_{key}"] = value
                        oracle_rows.append(row)

                    for count in sample_sizes:
                        for seed in seeds:
                            rng = np.random.default_rng(
                                condition_seed(weight_name, panel, model, construction, count, seed)
                            )
                            if model == "peptide_by_peptide":
                                # One independent residual vector per timepoint per replicate.
                                unit = np.stack(
                                    [draw_standard_normal(factor, count, rng) for _ in range(len(TIMEPOINTS))]
                                )
                            else:
                                unit = draw_standard_normal(factor, count, rng)[None, :, :]

                            for scale in scales:
                                # Residuals scale linearly in s, so the same deviates realize
                                # every scale exactly; this makes the s^2 relation testable.
                                residuals = scale * unit
                                if model == "peptide_by_peptide":
                                    draws = mean_peptide_uptake[:, None, :] + residuals
                                    per_time = [
                                        sample_covariance(residuals[index], "population")
                                        for index in range(len(TIMEPOINTS))
                                    ]
                                else:
                                    draws = mean_peptide_uptake.reshape(-1)[None, :] + residuals[0]
                                    per_time = [sample_covariance(residuals[0], "population")]

                                violations = bound_violation_fraction(draws)
                                supplied = scale**2 * base
                                bound_rows.append(
                                    {
                                        "weights": weight_name,
                                        "panel": panel,
                                        "sampling_model": model,
                                        "construction": construction,
                                        "scale": scale,
                                        "n": count,
                                        "seed": seed,
                                        "bound_violation_fraction": violations,
                                    }
                                )
                                if scale == 0.0:
                                    # Exact-mean control: zero covariance carries no information.
                                    continue
                                for normalization in NORMALIZATIONS:
                                    correction = 1.0 if normalization == "population" else count / max(count - 1, 1)
                                    # Timepoints are estimated separately, then pooled by median.
                                    evaluated = [
                                        evaluate_estimate(estimate * correction, supplied) for estimate in per_time
                                    ]
                                    pooled = {
                                        key: (
                                            bool(np.median([float(item[key]) for item in evaluated]) >= 0.5)
                                            if isinstance(evaluated[0][key], bool)
                                            else float(np.median([item[key] for item in evaluated]))
                                        )
                                        for key in evaluated[0]
                                    }
                                    sample_rows.append(
                                        {
                                            "weights": weight_name,
                                            "panel": panel,
                                            "sampling_model": model,
                                            "construction": construction,
                                            "scale": scale,
                                            "n": count,
                                            "seed": seed,
                                            "normalization": normalization,
                                            **pooled,
                                        }
                                    )

    samples = pd.DataFrame(sample_rows)
    bounds = pd.DataFrame(bound_rows)
    oracles = pd.DataFrame(oracle_rows)

    key_columns = ["weights", "panel", "sampling_model", "construction", "scale", "n", "normalization"]
    numeric = [c for c in samples.columns if c not in key_columns + ["seed"] and samples[c].dtype != bool]
    gate_columns = [c for c in samples.columns if c.endswith("gate_pass")]
    # Across-seed summary: median, spread, and CI per condition, plus per-gate pass fraction.
    # Everything is aggregated in one groupby and joined on the keys, so no column depends on
    # two separate groupbys happening to emit rows in the same order.
    aggregations = {f"{c}__median": (c, "median") for c in numeric}
    aggregations.update({f"{c}__std": (c, "std") for c in numeric})
    aggregations.update({f"{c}__ci_low": (c, lambda values: values.quantile(0.025)) for c in numeric})
    aggregations.update({f"{c}__ci_high": (c, lambda values: values.quantile(0.975)) for c in numeric})
    aggregations.update({f"{c}_fraction": (c, "mean") for c in gate_columns})
    aggregations["seeds"] = ("seed", "nunique")
    summaries = samples.groupby(key_columns).agg(**aggregations).reset_index()
    # The medians keep their bare names so downstream gate checks read as the plain metric.
    summaries = summaries.rename(columns={f"{c}__median": c for c in numeric})

    bound_summary = (
        bounds.groupby(["weights", "panel", "sampling_model", "construction", "scale", "n"])[
            "bound_violation_fraction"
        ]
        .median()
        .reset_index()
    )

    # Qualification: primary scale, primary alpha, N <= 1000, every panel, alpha-stable.
    qualifications = []
    primary = summaries[
        (summaries.scale == PRIMARY_SCALE)
        & (summaries.normalization == "population")
        & (summaries.n <= MAX_GATE_N)
    ]
    for (weight_name, model, construction), rows in primary.groupby(
        ["weights", "sampling_model", "construction"]
    ):
        panels = set(rows.panel)
        for representation in ("full", "marginal", "conditional"):
            qualified, qualifying_n = False, None
            for count in sorted(set(rows.n)):
                at_n = rows[rows.n == count]
                if set(at_n.panel) != panels:
                    continue
                if representation == "full":
                    passed = bool(
                        (at_n.full_gate_pass_fraction >= GATES["min_seed_pass_fraction"]).all()
                        and at_n.trace_ratio.between(GATES["trace_ratio_low"], GATES["trace_ratio_high"]).all()
                        and (at_n.normalized_frobenius_distance <= GATES["max_normalized_distance"]).all()
                        and (
                            at_n.off_diagonal_correlation >= GATES["min_off_diagonal_correlation"]
                        ).all()
                    )
                else:
                    passed = bool(
                        (
                            at_n[f"{representation}_a{PRIMARY_ALPHA:g}_gate_pass_fraction"]
                            >= GATES["min_seed_pass_fraction"]
                        ).all()
                        and (
                            at_n[f"{representation}_a{PRIMARY_ALPHA:g}_pearson"]
                            >= GATES["min_profile_correlation"]
                        ).all()
                        and (
                            at_n[f"{representation}_a{PRIMARY_ALPHA:g}_spearman"]
                            >= GATES["min_profile_correlation"]
                        ).all()
                        and (
                            at_n[f"{representation}_a{PRIMARY_ALPHA:g}_log_rmse"]
                            <= GATES["max_profile_log_rmse"]
                        ).all()
                    )
                    # The conclusion must be unchanged at alpha = 0.01 and 0.10.
                    for alpha in (0.01, 0.10):
                        passed = passed and bool(
                            (
                                at_n[f"{representation}_a{alpha:g}_gate_pass_fraction"]
                                >= GATES["min_seed_pass_fraction"]
                            ).all()
                        )
                violation = bound_summary[
                    (bound_summary.weights == weight_name)
                    & (bound_summary.sampling_model == model)
                    & (bound_summary.construction == construction)
                    & (bound_summary.scale == PRIMARY_SCALE)
                    & (bound_summary.n == count)
                ]
                bounded = bool(
                    len(violation)
                    and (violation.bound_violation_fraction <= GATES["max_bound_violation_fraction"]).all()
                )
                if passed and bounded:
                    qualified, qualifying_n = True, int(count)
                    break
            qualifications.append(
                {
                    "weights": weight_name,
                    "sampling_model": model,
                    "construction": construction,
                    "representation": representation,
                    "qualified": qualified,
                    "qualifying_n": qualifying_n,
                }
            )
    qualifications = pd.DataFrame(qualifications)

    construction_registry().to_csv(args.output_dir / "construction_registry.csv", index=False)
    samples.to_csv(args.output_dir / "sample_metrics.csv", index=False)
    summaries.to_csv(args.output_dir / "seed_summaries.csv", index=False)
    bounds.to_csv(args.output_dir / "bound_violations.csv", index=False)
    qualifications.to_csv(args.output_dir / "gate_qualifications.csv", index=False)
    oracles.to_csv(args.output_dir / "oracle_correspondence.csv", index=False)
    shrinkage = summaries[
        key_columns + [c for c in summaries.columns if "_a0" in c and c.endswith(("pearson", "log_rmse"))]
    ]
    shrinkage.to_csv(args.output_dir / "shrinkage_sensitivity.csv", index=False)
    np.savez_compressed(args.output_dir / "covariance_matrices.npz", **matrices)

    qualified = qualifications[qualifications.qualified]
    gates = {
        "gate_definitions": GATES,
        "qualified": qualified.to_dict("records"),
        "max_bound_violation_fraction_at_primary_scale": float(
            bound_summary[bound_summary.scale == PRIMARY_SCALE].bound_violation_fraction.max()
        ),
        "fitting_performed": False,
        "stage2_status": "pending_user_review",
    }
    (args.output_dir / "gate_results.json").write_text(json.dumps(gates, indent=2) + "\n")
    manifest = {
        "stage": "1B_peptide_monte_carlo_target_covariance",
        "status": "smoke" if args.smoke else "complete_pending_user_review",
        "fitting_performed": False,
        "decoy_or_gradient_tests_performed": False,
        "cluster_recovery_used": False,
        "sampling_coordinate": "additive Gaussian residuals in observed peptide uptake",
        "clipping_applied": False,
        "sample_sizes": list(sample_sizes),
        "seeds": list(seeds),
        "scales": list(scales),
        "sampling_models": list(SAMPLING_MODELS),
        "normalizations": list(NORMALIZATIONS),
        "timepoints": TIMEPOINTS.tolist(),
        "primary_alpha": PRIMARY_ALPHA,
        "alpha_sensitivity": list(ALPHAS),
        "base_seed": BASE_SEED,
        "inputs": inputs,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Stage 1B: peptide-level Monte Carlo target covariance",
        "",
        "## Question",
        "",
        "Can a covariance chosen in peptide-uptake space be estimated stably from a finite number",
        "of additive Gaussian Monte Carlo target draws? This is a sampling/estimator validation.",
        "No fitting, optimization, decoy test, or cluster recovery was run.",
        "",
        "## Sampling model",
        "",
        "`Y_draw = Y_target + epsilon`, `epsilon ~ N(0, s^2 Sigma_target)`. Draws are additive target",
        "data, not structural frames and not BV forward-model predictions. Uptake bounds are never",
        "enforced by clipping, because clipping would change the registered covariance; out-of-range",
        "entries are counted instead.",
        "",
        "## Convergence at the primary scale",
        "",
        "| Model | Construction | N | Trace ratio | Norm. distance | Off-diag. corr. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    view = summaries[
        (summaries.scale == PRIMARY_SCALE)
        & (summaries.normalization == "population")
        & (summaries.weights == "known_40_60")
    ]
    for (model, construction, count), rows in view.groupby(["sampling_model", "construction", "n"]):
        lines.append(
            f"| {model} | {construction} | {count} | {rows.trace_ratio.median():.3f} | "
            f"{rows.normalized_frobenius_distance.median():.3f} | {rows.off_diagonal_correlation.median():.3f} |"
        )
    lines += [
        "",
        "## Bound violations",
        "",
        "| Model | Construction | Scale | Median out-of-range entry fraction |",
        "|---|---|---:|---:|",
    ]
    for (model, construction, scale), rows in bound_summary[
        bound_summary.weights == "known_40_60"
    ].groupby(["sampling_model", "construction", "scale"]):
        lines.append(
            f"| {model} | {construction} | {scale:g} | {rows.bound_violation_fraction.median():.4f} |"
        )
    lines += [
        "",
        "## Qualification",
        "",
        (
            "No construction/representation qualified at any N <= 1000. Covariance alignment does "
            "not advance by default."
            if qualified.empty
            else "Qualified:\n\n"
            + "\n".join(
                f"- `{row.construction}` / {row.sampling_model} / {row.representation}"
                f" / {row.weights} weights, from N={int(row.qualifying_n)}"
                for row in qualified.itertuples(index=False)
            )
        ),
        "",
        "Qualification here is a sampling result only. It states that the imposed covariance is",
        "recoverable from finite draws, not that the covariance corresponds to conformational",
        "effective-rate spread. See `oracle_correspondence.csv` for that separate question; the",
        "2026-07-16 correspondence litmus already found no uptake-only construction reproduces it.",
        "",
        "Read a qualification against three caveats before treating it as support for fitting.",
        "",
        "1. Check `effective_rank`. These curve covariances are near-rank-one, so they are a",
        "   trivially easy estimation target; fast convergence measures their degeneracy.",
        "2. Each construction and each sampling model carries its own natural covariance",
        "   magnitude, so `s=1` is not a common noise level and the bound-violation gate is not",
        "   comparable across them. The two sampling models must not be compared directly.",
        "3. A construction blocked only by bound violations is evidence that an untruncated",
        "   additive Gaussian target model is invalid at that scale, not that it nearly passed.",
        "",
        "## Checkpoint",
        "",
        "Stage 2 remains **pending explicit user approval**. No fitting is authorized by this run.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, model in zip(axes, SAMPLING_MODELS):
        subset = view[view.sampling_model == model]
        for construction, rows in subset.groupby("construction"):
            grouped = rows.groupby("n").normalized_frobenius_distance.median()
            axis.plot(grouped.index, grouped.to_numpy(), marker="o", label=construction)
        axis.axhline(GATES["max_normalized_distance"], color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("draws N")
        axis.set_ylabel("normalized Frobenius distance")
        axis.set_title(model)
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(args.output_dir / "convergence.png", dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=root / "_pf_peptide_moment_final")
    parser.add_argument("--output-dir", type=Path, default=root / "_peptide_mc_covariance_litmus")
    parser.add_argument("--smoke", action="store_true", help="integration check only; not evidence")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
