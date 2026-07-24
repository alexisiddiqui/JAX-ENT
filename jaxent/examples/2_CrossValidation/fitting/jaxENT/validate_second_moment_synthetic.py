#!/usr/bin/env python3
"""Synthetic positive/negative control for the envelope second-moment method."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jaxent.src.analysis.hdx_ex2 import (
    build_expfact_peptide_map,
    convolve_isotope_and_deuteron_distributions,
    fit_ex2_solution_set,
    peptide_deuteron_count_distribution,
    thin_deuteron_count_distribution,
)
from jaxent.src.analysis.second_moment import (
    decompose_envelope,
    envelope_moments,
    precision_band_decision,
)
from investigate_moprp_ex2_physics import _bin_raw_mass_envelope


HERE = Path(__file__).resolve().parent
MOPRP = HERE.parents[1] / "data/_MoPrP"
DEFAULT_OUTPUT = HERE / "_moprp_ex2_second_moment_synthetic"
TIMES = np.asarray([1.0, 60.0, 1440.0])
SURVIVAL = 0.4980542458700022
N_BINS = 10
SPACING = 1.00627
D_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.4, 3.2)
GEOMETRIES = (
    ("n4_narrow", 4, np.asarray([0.70, 0.82, 0.96, 1.10])),
    (
        "n6_peptide_like",
        6,
        np.asarray([0.30, 0.51, 0.70, 0.96, 1.76, 3.05]),
    ),
    (
        "n9_broad",
        9,
        np.asarray([0.08, 0.16, 0.30, 0.51, 0.70, 0.96, 1.76, 3.05, 5.77]),
    ),
)


def _controlled_latent(rng: np.random.Generator, frames: int) -> np.ndarray:
    latent = rng.normal(size=frames)
    return (latent - latent.mean()) / latent.std(ddof=0)


def _mean_log_pf(rates: np.ndarray, target_uptake: float = 0.60) -> np.ndarray:
    target_rate = -np.log1p(-target_uptake) / 60.0
    offsets = np.linspace(-0.35, 0.35, len(rates))
    base = np.log(rates / target_rate) + offsets
    probabilities = 1.0 - np.exp(-60.0 * rates * np.exp(-base))
    correction = np.log(
        probabilities.mean() / target_uptake
    )
    return base + correction


def _mixture_distribution(
    frame_log_pf: np.ndarray,
    rates: np.ndarray,
    time: float,
    peptide_map: Any,
) -> np.ndarray:
    return np.mean(
        [
            peptide_deuteron_count_distribution(
                frame_log_pf[:, frame], rates, time, peptide_map, 0
            )
            for frame in range(frame_log_pf.shape[1])
        ],
        axis=0,
    )


def _full_envelope(
    pre_quench: np.ndarray, protonated: np.ndarray
) -> np.ndarray:
    return convolve_isotope_and_deuteron_distributions(
        protonated,
        thin_deuteron_count_distribution(pre_quench, SURVIVAL),
    )


def _observe(full: np.ndarray) -> np.ndarray:
    observed = np.zeros(N_BINS)
    observed[: min(N_BINS, len(full))] = full[:N_BINS]
    return observed / observed.sum()


def _padded_width_ratio(full: np.ndarray, observed: np.ndarray) -> float:
    """Retired padded-support sensitivity: translate without a fixed-window cut."""
    support = max(len(full), len(observed)) + 2 * N_BINS
    predicted = np.zeros(support)
    target = np.zeros(support)
    predicted[N_BINS : N_BINS + len(full)] = full
    target[N_BINS : N_BINS + len(observed)] = observed
    pred_mean, pred_var = envelope_moments(predicted)
    obs_mean, obs_var = envelope_moments(target)
    delta = obs_mean - pred_mean
    positions = np.arange(support, dtype=float) + delta
    lower = np.floor(positions).astype(int)
    fraction = positions - lower
    shifted = np.zeros(support)
    for indices, weights in (
        (lower, predicted * (1.0 - fraction)),
        (lower + 1, predicted * fraction),
    ):
        valid = (indices >= 0) & (indices < support)
        np.add.at(shifted, indices[valid], weights[valid])
    shifted /= shifted.sum()
    _, aligned_var = envelope_moments(shifted)
    return aligned_var / obs_var


def _real_peptide1_log_rate_variance() -> dict[str, Any]:
    artifact = (
        HERE
        / "_moprp_target_variance_constrained_optimum_20260724"
        / "blinded_variances.npz"
    )
    if not artifact.exists():
        return {"available": False}
    arrays = np.load(artifact)
    rows = slice(4, 9)  # peptide 1: residues 5--9 after the source N-terminal drop
    summaries = {}
    for ensemble, candidate in (
        ("AF2_MSAss", "candidate_000026"),
        ("AF2_filtered", "candidate_000074"),
    ):
        mean_rates = arrays[f"{ensemble}__mean_rates"][rows]
        variances = arrays[f"{candidate}__variances"][rows]
        # Match a lognormal rate ensemble: sigma_log(k)^2 = log(1 + d/kbar^2).
        implied = np.log1p(variances / mean_rates**2)
        summaries[ensemble] = {
            "candidate_id": candidate,
            "residue_ids": list(range(5, 10)),
            "implied_log_rate_variance": implied.tolist(),
            "median": float(np.median(implied)),
            "minimum": float(implied.min()),
            "maximum": float(implied.max()),
        }
    return {"available": True, "conversion": "log(1 + d_i / kbar_i^2)", **summaries}


def _single_case(
    geometry: str,
    active_rates: np.ndarray,
    d_true: float,
    seed: int,
    protonated: np.ndarray,
    observed_frames: int,
    model_frames: int,
) -> dict[str, Any]:
    n_active = len(active_rates)
    peptide_map = build_expfact_peptide_map(
        "A" * (n_active + 1),
        np.asarray([[1, 1, n_active + 1]]),
    )
    rates = np.concatenate(([-1.0], active_rates))
    mean_log_pf = np.concatenate(([np.nan], _mean_log_pf(active_rates)))
    observed_latent = _controlled_latent(
        np.random.default_rng(seed + 10_000), observed_frames
    )
    model_latent = _controlled_latent(
        np.random.default_rng(seed + 20_000), model_frames
    )
    observed_frames_pf = mean_log_pf[:, None] + np.sqrt(
        d_true
    ) * observed_latent[None, :]
    model_frames_pf = mean_log_pf[:, None] + np.sqrt(
        d_true
    ) * model_latent[None, :]

    observed_by_time = {}
    observed_uptake = []
    for time in TIMES:
        pre = _mixture_distribution(
            observed_frames_pf, rates, float(time), peptide_map
        )
        full = _full_envelope(pre, protonated)
        observed = _observe(full)
        observed_by_time[float(time)] = observed
        centroid, _ = envelope_moments(observed)
        baseline_centroid, _ = envelope_moments(protonated)
        observed_uptake.append(
            np.clip(
                (centroid - baseline_centroid)
                / (SURVIVAL * n_active),
                0.0,
                1.0,
            )
        )

    fitted = fit_ex2_solution_set(
        np.asarray([observed_uptake]),
        rates,
        TIMES,
        peptide_map,
        starts=1,
        seed=seed + 30_000,
        maxiter=1000,
        initial_log_pf_vectors=[mean_log_pf],
    ).solutions[0]
    observed = observed_by_time[60.0]
    average_full = _full_envelope(
        peptide_deuteron_count_distribution(
            mean_log_pf, rates, 60.0, peptide_map, 0
        ),
        protonated,
    )
    mixture_full = _full_envelope(
        _mixture_distribution(model_frames_pf, rates, 60.0, peptide_map),
        protonated,
    )
    ex2_full = _full_envelope(
        peptide_deuteron_count_distribution(
            fitted.log_pf, rates, 60.0, peptide_map, 0
        ),
        protonated,
    )
    average = decompose_envelope(average_full, observed, N_BINS)
    mixture = decompose_envelope(mixture_full, observed, N_BINS)
    ex2 = decompose_envelope(ex2_full, observed, N_BINS)
    band = precision_band_decision(
        average["centroid_aligned_width_ratio"],
        mixture["centroid_aligned_width_ratio"],
        ex2["centroid_aligned_width_ratio"],
    )
    padded_average = _padded_width_ratio(average_full, observed)
    padded_mixture = _padded_width_ratio(mixture_full, observed)
    padded_band = precision_band_decision(
        padded_average,
        padded_mixture,
        _padded_width_ratio(ex2_full, observed),
    )
    return {
        "geometry": geometry,
        "n_active": n_active,
        "d_true_log_pf_variance": d_true,
        "seed": seed,
        "observed_frames": observed_frames,
        "model_frames": model_frames,
        "observed_t60_centroid": envelope_moments(observed)[0],
        "observed_t60_width_var": envelope_moments(observed)[1],
        "average_first_width_ratio": average[
            "centroid_aligned_width_ratio"
        ],
        "frame_mixture_width_ratio": mixture[
            "centroid_aligned_width_ratio"
        ],
        "ex2_analog_width_ratio": ex2["centroid_aligned_width_ratio"],
        "precision_band_lower": band["precision_band_lower"],
        "precision_band_upper": band["precision_band_upper"],
        "detected_excess_width": band["detected_excess_width"],
        "recovered_conformational_width_excess": (
            mixture["centroid_aligned_width_ratio"]
            - average["centroid_aligned_width_ratio"]
        ),
        "separation_survives": band["separation_survives"],
        "padded_average_first_width_ratio": padded_average,
        "padded_frame_mixture_width_ratio": padded_mixture,
        "padded_separation_survives": padded_band["separation_survives"],
        "boundary_verdict_agreement": (
            band["separation_survives"]
            == padded_band["separation_survives"]
        ),
        "ex2_fit_rmse": fitted.rmse,
        "ex2_fit_success": fitted.success,
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.loadtxt(MOPRP / "spectra/pep1.1.txt")
    base_mass = float(raw[np.argmax(raw[:, 1]), 0])
    protonated = _bin_raw_mass_envelope(
        MOPRP / "spectra/pep1.1.txt",
        base_mass=base_mass,
        spacing=SPACING,
        n_bins=N_BINS,
    )
    rows = [
        _single_case(
            name,
            rates,
            d_true,
            seed,
            protonated,
            args.observed_frames,
            args.model_frames,
        )
        for name, _, rates in GEOMETRIES
        for d_true in args.d_levels
        for seed in range(args.seeds)
    ]
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "synthetic_detectability_sweep.csv", index=False)

    geometry_results = []
    for name, n_active, _ in GEOMETRIES:
        group = table[table.geometry == name]
        negative = group[group.d_true_log_pf_variance == 0.0]
        levels = (
            group.groupby("d_true_log_pf_variance", as_index=False)
            .agg(
                detected_excess_width=("detected_excess_width", "median"),
                recovered_conformational_width_excess=(
                    "recovered_conformational_width_excess",
                    "median",
                ),
                detection_rate=("separation_survives", "mean"),
            )
            .sort_values("d_true_log_pf_variance")
        )
        positive = levels[levels.d_true_log_pf_variance > 0]
        rho = float(
            spearmanr(
                positive.d_true_log_pf_variance,
                positive.recovered_conformational_width_excess,
            ).statistic
        )
        detectable = levels[levels.detection_rate == 1.0]
        d_min = (
            float(detectable.d_true_log_pf_variance.min())
            if len(detectable)
            else None
        )
        resolved_truth_error = float(
            np.median(np.abs(group.frame_mixture_width_ratio - 1.0))
        )
        padded_truth_error = float(
            np.median(np.abs(group.padded_frame_mixture_width_ratio - 1.0))
        )
        geometry_results.append(
            {
                "geometry": name,
                "n_active": n_active,
                "negative_control_passes": bool(
                    (~negative.separation_survives).all()
                    and np.allclose(
                        negative.average_first_width_ratio,
                        negative.frame_mixture_width_ratio,
                        rtol=0.0,
                        atol=1e-12,
                    )
                ),
                "negative_max_average_mixture_gap": float(
                    np.max(
                        np.abs(
                            negative.average_first_width_ratio
                            - negative.frame_mixture_width_ratio
                        )
                    )
                ),
                "spearman_d_true_vs_detected_excess": rho,
                "monotonicity_passes": bool(rho >= 0.95),
                "d_min_all_seeds": d_min,
                "resolved_detection_count": int(group.separation_survives.sum()),
                "padded_detection_count": int(
                    group.padded_separation_survives.sum()
                ),
                "resolved_frame_mixture_median_abs_error_from_truth": (
                    resolved_truth_error
                ),
                "padded_frame_mixture_median_abs_error_from_truth": (
                    padded_truth_error
                ),
                "boundary_robustness_passes": bool(
                    group.separation_survives.sum()
                    >= group.padded_separation_survives.sum()
                    and resolved_truth_error <= padded_truth_error
                ),
                "ex2_floor_median": float(
                    group.ex2_analog_width_ratio.median()
                ),
                "ex2_floor_range": [
                    float(group.ex2_analog_width_ratio.min()),
                    float(group.ex2_analog_width_ratio.max()),
                ],
            }
        )

    negative_passes = all(
        result["negative_control_passes"] for result in geometry_results
    )
    finite_floor = all(
        result["d_min_all_seeds"] is not None for result in geometry_results
    )
    monotonic = all(
        result["monotonicity_passes"] for result in geometry_results
    )
    boundary = all(
        result["boundary_robustness_passes"] for result in geometry_results
    )
    floor_sane = bool(
        table.ex2_analog_width_ratio.between(0.75, 1.25).mean() >= 0.90
    )
    verdict = {
        "status": "diagnostic_only",
        "truth_coordinate": "per-residue across-frame log-PF variance",
        "injected_truth_used_as_estimator_input": False,
        "poisson_counting_noise": False,
        "survival_probability": SURVIVAL,
        "n_bins": N_BINS,
        "d_levels": list(args.d_levels),
        "seeds": args.seeds,
        "geometries": geometry_results,
        "real_peptide1_inferred_variance_comparison": (
            _real_peptide1_log_rate_variance()
        ),
        "negative_control_passes": negative_passes,
        "finite_detectability_floor": finite_floor,
        "monotonicity_passes": monotonic,
        "boundary_robustness_passes": boundary,
        "ex2_floor_sanity_passes": floor_sane,
        "method_validated": bool(
            negative_passes
            and finite_floor
            and monotonic
            and boundary
            and floor_sane
        ),
        "item_3_gate": {
            "proceed_to_envelope_estimator": False,
            "stage_a_bv_mean_fix_required": True,
        },
    }
    (args.output_dir / "synthetic_second_moment_validation.json").write_text(
        json.dumps(verdict, indent=2) + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--observed-frames", type=int, default=2048)
    parser.add_argument("--model-frames", type=int, default=512)
    parser.add_argument(
        "--d-levels", type=float, nargs="+", default=list(D_LEVELS)
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
