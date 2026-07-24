#!/usr/bin/env python3
"""Diagnostic centroid/shape decomposition for the MoPrP peptide-1 envelope.

This script is intentionally not an estimator: it reads the committed
pre-quench distributions, applies the already calibrated control survival,
and reports moments and a one-parameter mass-shift diagnostic.  No weights,
coefficients, or envelope parameters are fitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jaxent.src.analysis.hdx_ex2 import (
    convolve_isotope_and_deuteron_distributions,
    thin_deuteron_count_distribution,
)
from jaxent.src.analysis.second_moment import (
    decompose_envelope,
    envelope_moments,
    precision_band_decision,
)
from investigate_moprp_ex2_physics import (
    _bin_raw_mass_envelope,
    _match_envelope_length,
)


HERE = Path(__file__).resolve().parent
SPECTRA = HERE.parents[1] / "data/_MoPrP/spectra"
DEFAULT_RUN = HERE / "_moprp_ex2_physics_bv_v2"
DEFAULT_OUTPUT = HERE / "_moprp_ex2_second_moment"
TIMES = (1.0, 60.0, 1440.0)
N_BINS = 10
SPACING = 1.00627
EDGE_MASS_TOLERANCE = 0.01


def _raw_edge_row(
    path: Path,
    *,
    label: str,
    time_min: float | None,
    base_mass: float,
    spacing: float,
    n_bins: int,
) -> dict[str, Any]:
    values = np.loadtxt(path, dtype=float)
    nominal = np.rint((values[:, 0] - base_mass) / spacing).astype(int)
    positive = values[:, 1] > 0
    intensities = values[positive, 1]
    nominal = nominal[positive]
    total = float(intensities.sum())
    in_window = (nominal >= 0) & (nominal < n_bins)
    window_total = float(intensities[in_window].sum())

    def fraction(mask: np.ndarray, denominator: float = total) -> float:
        return float(intensities[mask].sum() / denominator) if denominator else 0.0

    return {
        "spectrum": label,
        "time_min": time_min,
        "n_bins": n_bins,
        "first_bin_intensity_fraction": fraction(
            nominal == 0, window_total
        ),
        "last_bin_intensity_fraction": fraction(
            nominal == n_bins - 1, window_total
        ),
        "outside_left_intensity_fraction": fraction(nominal < 0),
        "outside_right_intensity_fraction": fraction(nominal >= n_bins),
        "outside_total_intensity_fraction": fraction(~in_window),
    }


def _edge_mass_check(
    paths: dict[int, Path],
    *,
    base_mass: float,
    spacing: float,
    initial_n_bins: int,
    tolerance: float,
) -> tuple[int, pd.DataFrame]:
    labels = {
        1: ("protonated_control", None),
        2: ("exchange", 1.0),
        3: ("exchange", 60.0),
        4: ("exchange", 1440.0),
        5: ("fully_deuterated_control", None),
    }
    n_bins = initial_n_bins
    while True:
        rows = [
            _raw_edge_row(
                paths[index],
                label=label,
                time_min=time,
                base_mass=base_mass,
                spacing=spacing,
                n_bins=n_bins,
            )
            for index, (label, time) in labels.items()
        ]
        table = pd.DataFrame(rows)
        right_contained = bool(
            (
                (table.last_bin_intensity_fraction <= tolerance)
                & (table.outside_right_intensity_fraction <= tolerance)
            ).all()
        )
        if right_contained:
            break
        n_bins += 1
        if n_bins > 64:
            raise RuntimeError("could not find an adequate observed mass window")
    table["edge_mass_tolerance"] = tolerance
    table["right_boundary_contained"] = right_contained
    table["left_boundary_note"] = (
        "bin 0 is the physical zero-isotope/deuteron boundary; negative nominal "
        "mass is excluded raw baseline/noise and is checked separately"
    )
    table["window_adequate"] = (
        table.right_boundary_contained
        & (table.outside_left_intensity_fraction <= tolerance)
    )
    return n_bins, table


def _load_inputs(
    run_dir: Path, spectra_dir: Path
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[float, np.ndarray],
    np.ndarray,
    pd.DataFrame,
    int,
]:
    counts = pd.read_csv(run_dir / "peptide1_deuteron_count_distributions.csv")
    calibration = json.loads((run_dir / "peptide1_envelope_calibration.json").read_text())
    paths = {i: spectra_dir / f"pep1.{i}.txt" for i in range(1, 6)}
    if any(not path.exists() for path in paths.values()):
        raise FileNotFoundError("all recovered pep1.1--5.txt spectra are required")
    raw = np.loadtxt(paths[1], dtype=float)
    base_mass = float(raw[np.argmax(raw[:, 1]), 0])
    n_bins, edge_mass = _edge_mass_check(
        paths,
        base_mass=base_mass,
        spacing=SPACING,
        initial_n_bins=N_BINS,
        tolerance=EDGE_MASS_TOLERANCE,
    )
    protonated = _bin_raw_mass_envelope(paths[1], base_mass=base_mass, spacing=SPACING, n_bins=n_bins)
    observed = {time: _bin_raw_mass_envelope(paths[index], base_mass=base_mass, spacing=SPACING, n_bins=n_bins)
                for time, index in ((1.0, 2), (60.0, 3), (1440.0, 4))}
    return counts, calibration, observed, protonated, edge_mass, n_bins


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        counts,
        calibration,
        observed_by_time,
        protonated,
        edge_mass,
        n_bins,
    ) = _load_inputs(args.run_dir, args.spectra_dir)
    survival = float(calibration["effective_survival_probability"])
    rows: list[dict[str, Any]] = []
    decomposition: list[dict[str, Any]] = []

    for keys, group in counts.groupby(["source", "condition", "solution_rank", "semantics", "time_min"], dropna=False):
        source, condition, rank, semantics, time = keys
        if source == "experimental_EX2_fit" and not (condition == "unregularized" and int(rank) == 0):
            continue
        if source != "experimental_EX2_fit" and condition not in {"BV_hard", "BV_switched"}:
            continue
        pre = group.sort_values("exchanged_amides")["probability"].to_numpy(float)
        full = convolve_isotope_and_deuteron_distributions(
            protonated, thin_deuteron_count_distribution(pre, survival)
        )
        predicted = _match_envelope_length(full, n_bins)
        observed = observed_by_time[float(time)]
        pred_mean, pred_var = envelope_moments(predicted)
        obs_mean, obs_var = envelope_moments(observed)
        common = {
            "source": source,
            "construction": condition,
            "semantics": "frame_mixture" if semantics == "frame_mixture_sensitivity" else semantics,
            "time_min": float(time),
            "predicted_centroid": pred_mean,
            "observed_centroid": obs_mean,
            "centroid_gap": pred_mean - obs_mean,
            "predicted_width_var": pred_var,
            "observed_width_var": obs_var,
            "width_ratio": pred_var / obs_var,
            "envelope_r2": 1.0 - float(np.sum((predicted - observed) ** 2)) / float(np.sum((observed - observed.mean()) ** 2)),
        }
        rows.append(common)
        result = decompose_envelope(full, observed, n_bins)
        decomposition.append(
            {
                **common,
                "boundary_handling": "true_mass_shift_then_fixed_window",
                "centroid_aligned_width_var": result[
                    "centroid_aligned_width_var"
                ],
                "centroid_aligned_width_ratio": result[
                    "centroid_aligned_width_ratio"
                ],
                **{
                    key: result[key]
                    for key in (
                        "best_shift_bins",
                        "aligned_centroid_gap",
                        "total_sse",
                        "centroid_component_sse",
                        "shape_component_sse",
                        "centroid_explained_fraction",
                    )
                },
            }
        )

    moments = pd.DataFrame(rows)
    decomp = pd.DataFrame(decomposition)
    edge_mass.to_csv(args.output_dir / "edge_mass_check.csv", index=False)
    moments.to_csv(args.output_dir / "envelope_moments.csv", index=False)
    decomp.to_csv(args.output_dir / "centroid_shape_decomposition.csv", index=False)

    def get(source: str, construction: str, semantics: str, time: float) -> pd.Series:
        return decomp[
            (decomp.source == source)
            & (decomp.construction == construction)
            & (decomp.semantics == semantics)
            & (decomp.time_min == time)
        ].iloc[0]

    decisions = []
    for time in TIMES:
        ex2 = get("experimental_EX2_fit", "unregularized", "residue_EX2", time)
        af = get("AF2_MSAss", "BV_hard", "average_first", time)
        fm = get("AF2_MSAss", "BV_hard", "frame_mixture", time)
        precision_floor = float(ex2.centroid_aligned_width_ratio)
        band = precision_band_decision(
            float(af.centroid_aligned_width_ratio),
            float(fm.centroid_aligned_width_ratio),
            precision_floor,
        )
        precision_lower = band["precision_band_lower"]
        precision_upper = band["precision_band_upper"]
        independent = band["separation_survives"]
        bv_rows = decomp[
            (decomp.time_min == time)
            & (decomp.source != "experimental_EX2_fit")
        ]
        centroid_dominated = bool(
            (bv_rows.centroid_explained_fraction >= 0.5).all()
        )
        signal = (
            "mean_confounded_with_independent_width_signal"
            if centroid_dominated and independent
            else ("mean_confounded_no_independent_signal" if centroid_dominated else "inconclusive")
        )
        decisions.append(
            {
                "time_min": time,
                "envelope_second_moment_signal": signal,
                "flagged_latent_width_signal": independent,
                "centroid_dominated": centroid_dominated,
                "centroid_explained_sse_fraction_bv_hard_average_first": float(
                    af.centroid_explained_fraction
                ),
                "centroid_aligned_width_ratio_average_first": float(
                    af.centroid_aligned_width_ratio
                ),
                "centroid_aligned_width_ratio_frame_mixture": float(
                    fm.centroid_aligned_width_ratio
                ),
                "ex2_fit_centroid_aligned_width_ratio_empirical_floor": precision_floor,
                "precision_band_lower": precision_lower,
                "precision_band_upper": precision_upper,
                "boundary_convention_resolved": True,
                "separation_survives_resolved_convention": independent,
            }
        )
    decision_df = pd.DataFrame(decisions)
    t60 = decision_df[decision_df.time_min == 60.0].iloc[0]
    survives = bool(t60.separation_survives_resolved_convention)
    verdict = {
        "status": "diagnostic_only",
        "base_mass": float(
            np.loadtxt(args.spectra_dir / "pep1.1.txt")[
                np.argmax(np.loadtxt(args.spectra_dir / "pep1.1.txt")[:, 1]), 0
            ]
        ),
        "effective_survival_probability": survival,
        "n_bins": n_bins,
        "edge_mass_tolerance": EDGE_MASS_TOLERANCE,
        "window_adequate": bool(edge_mass.window_adequate.all()),
        "resolved_boundary_convention": "true_mass_shift_then_fixed_window",
        "anchors": {
            "experimental_ex2_fit_t60_r2": float(
                get("experimental_EX2_fit", "unregularized", "residue_EX2", 60.0).envelope_r2
            ),
            "experimental_ex2_fit_t60_shift_bins": float(
                get("experimental_EX2_fit", "unregularized", "residue_EX2", 60.0).best_shift_bins
            ),
            "bv_hard_average_first_t60_r2": float(
                get("AF2_MSAss", "BV_hard", "average_first", 60.0).envelope_r2
            ),
        },
        "decisions": decisions,
        "stage_b_gate": {
            "separation_survives_resolved_convention": survives,
            "proceed_to_item_5": survives,
            "retire_item_3_for_peptide_1": not survives,
        },
        "item_3_gate": {
            "proceed_to_envelope_estimator": False,
            "max_information_time_min": 60.0,
            "rationale": (
                "The t=60 beyond-precision average-first/frame-mixture separation "
                "survives the resolved physical observation convention. Proceed "
                "only to item 5; coverage and a corrected BV mean remain required."
                if survives
                else "The separation does not survive the resolved physical "
                "observation convention: mean_confounded, no independent signal. "
                "Retire item 3 for peptide 1."
            ),
        },
    }
    (args.output_dir / "second_moment_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--spectra-dir", type=Path, default=SPECTRA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
