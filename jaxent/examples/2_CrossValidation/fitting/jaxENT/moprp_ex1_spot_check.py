#!/usr/bin/env python3
"""Spot-check EX1 versus EX2 using the MoPrP peptide-1 isotope envelopes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-jaxent")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from jaxent.src.analysis.hdx_ex2 import (
    convolve_isotope_and_deuteron_distributions,
    thin_deuteron_count_distribution,
)
from jaxent.src.analysis.second_moment import envelope_moments


HERE = Path(__file__).resolve().parent
SPECTRA = HERE.parents[1] / "data/_MoPrP/spectra"
DEFAULT_OUTPUT = HERE / "_moprp_ex1_spot_check"
SPACING = 1.00627
N_AMIDES = 5
PRIMARY_N_BINS = 10
TIMES_TO_FILES = {1.0: 2, 60.0: 3, 1440.0: 4}


def _bin_envelope(path: Path, base_mass: float, n_bins: int) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    nominal = np.rint((values[:, 0] - base_mass) / SPACING).astype(int)
    valid = (nominal >= 0) & (nominal < n_bins) & (values[:, 1] > 0)
    binned = np.bincount(
        nominal[valid], weights=values[valid, 1], minlength=n_bins
    ).astype(float)
    if binned.sum() <= 0:
        raise ValueError(f"no positive intensity mapped into the window: {path}")
    return binned / binned.sum()


def _window(distribution: np.ndarray, n_bins: int) -> np.ndarray:
    result = np.zeros(n_bins, dtype=float)
    result[: min(n_bins, len(distribution))] = distribution[:n_bins]
    if result.sum() <= 0:
        raise ValueError("predicted envelope has no mass in the observed window")
    return result / result.sum()


def _observe(
    counts: np.ndarray, protonated: np.ndarray, survival: float, n_bins: int
) -> np.ndarray:
    retained = thin_deuteron_count_distribution(counts, survival)
    full = convolve_isotope_and_deuteron_distributions(protonated, retained)
    return _window(full, n_bins)


def _calibrate_survival(
    protonated: np.ndarray, fully_deuterated: np.ndarray, n_bins: int
) -> float:
    counts = np.zeros(N_AMIDES + 1, dtype=float)
    counts[-1] = 1.0

    def loss(survival: float) -> float:
        predicted = _observe(counts, protonated, survival, n_bins)
        return float(np.mean((predicted - fully_deuterated) ** 2))

    result = minimize_scalar(
        loss, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12}
    )
    return float(result.x)


def _ex2_counts(parameter: float) -> np.ndarray:
    k = np.arange(N_AMIDES + 1)
    coefficients = np.array([1, 5, 10, 10, 5, 1], dtype=float)
    return coefficients * parameter**k * (1.0 - parameter) ** (N_AMIDES - k)


def _ex1_counts(parameter: float) -> np.ndarray:
    counts = np.zeros(N_AMIDES + 1, dtype=float)
    counts[0] = 1.0 - parameter
    counts[-1] = parameter
    return counts


def _fit_centroid(
    observed: np.ndarray,
    counts_fn: Callable[[float], np.ndarray],
    protonated: np.ndarray,
    survival: float,
    n_bins: int,
) -> tuple[float, np.ndarray]:
    observed_mean, _ = envelope_moments(observed)

    def loss(parameter: float) -> float:
        predicted = _observe(counts_fn(parameter), protonated, survival, n_bins)
        predicted_mean, _ = envelope_moments(predicted)
        return (predicted_mean - observed_mean) ** 2

    result = minimize_scalar(
        loss, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12}
    )
    parameter = float(result.x)
    return parameter, _observe(counts_fn(parameter), protonated, survival, n_bins)


def _mode_count(envelope: np.ndarray) -> int:
    padded = np.pad(envelope, 1, mode="constant")
    return int(np.sum((padded[1:-1] > padded[:-2]) & (padded[1:-1] > padded[2:])))


def _score(
    observed: np.ndarray,
    protonated: np.ndarray,
    survival: float,
    n_bins: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    ex2_parameter, ex2 = _fit_centroid(
        observed, _ex2_counts, protonated, survival, n_bins
    )
    ex1_parameter, ex1 = _fit_centroid(
        observed, _ex1_counts, protonated, survival, n_bins
    )
    observed_mean, observed_variance = envelope_moments(observed)
    ex2_mean, ex2_variance = envelope_moments(ex2)
    ex1_mean, ex1_variance = envelope_moments(ex1)
    ex2_sse = float(np.sum((ex2 - observed) ** 2))
    ex1_sse = float(np.sum((ex1 - observed) ** 2))
    if (
        ex2_parameter > 1.0 - 1e-6
        and ex1_parameter > 1.0 - 1e-6
    ) or np.isclose(ex2_sse, ex1_sse, rtol=1e-7, atol=1e-12):
        winner = "tie_at_parameter_boundary"
    else:
        winner = "EX2" if ex2_sse < ex1_sse else "EX1"
    row: dict[str, float | int] = {
        "observed_mean": observed_mean,
        "observed_variance": observed_variance,
        "observed_mode_count": _mode_count(observed),
        "ex2_exchange_probability": ex2_parameter,
        "ex2_mean": ex2_mean,
        "ex2_variance": ex2_variance,
        "ex2_sse": ex2_sse,
        "ex1_exchanged_fraction": ex1_parameter,
        "ex1_mean": ex1_mean,
        "ex1_variance": ex1_variance,
        "ex1_sse": ex1_sse,
        "sse_winner": winner,
        "variance_order_observed_lt_ex2_lt_ex1": int(
            observed_variance < ex2_variance < ex1_variance
        ),
    }
    return row, ex2, ex1


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {index: args.spectra_dir / f"pep1.{index}.txt" for index in range(1, 6)}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing peptide-1 spectra: {missing}")
    protonated_raw = np.loadtxt(paths[1], dtype=float)
    base_mass = float(protonated_raw[np.argmax(protonated_raw[:, 1]), 0])

    primary_protonated = _bin_envelope(paths[1], base_mass, PRIMARY_N_BINS)
    primary_full = _bin_envelope(paths[5], base_mass, PRIMARY_N_BINS)
    calibrated_survival = _calibrate_survival(
        primary_protonated, primary_full, PRIMARY_N_BINS
    )

    rows: list[dict[str, float | int]] = []
    overlays: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for time, index in TIMES_TO_FILES.items():
        observed = _bin_envelope(paths[index], base_mass, PRIMARY_N_BINS)
        row, ex2, ex1 = _score(
            observed, primary_protonated, calibrated_survival, PRIMARY_N_BINS
        )
        rows.append(
            {"time_min": time, "survival_probability": calibrated_survival, **row}
        )
        overlays[time] = (observed, ex2, ex1)
    moments = pd.DataFrame(rows)
    moments.to_csv(args.output_dir / "moments.csv", index=False)

    robustness_rows: list[dict[str, float | int]] = []
    for n_bins in (8, 10, 12):
        protonated = _bin_envelope(paths[1], base_mass, n_bins)
        fully_deuterated = _bin_envelope(paths[5], base_mass, n_bins)
        window_survival = _calibrate_survival(protonated, fully_deuterated, n_bins)
        for survival_offset in (-0.1, 0.0, 0.1):
            survival = float(np.clip(window_survival + survival_offset, 0.0, 1.0))
            for time, index in TIMES_TO_FILES.items():
                observed = _bin_envelope(paths[index], base_mass, n_bins)
                row, _, _ = _score(observed, protonated, survival, n_bins)
                robustness_rows.append(
                    {
                        "time_min": time,
                        "n_bins": n_bins,
                        "calibrated_survival_probability": window_survival,
                        "survival_offset": survival_offset,
                        "survival_probability": survival,
                        **row,
                    }
                )
    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(args.output_dir / "robustness.csv", index=False)

    expected = np.array([[0.734, 0.780], [1.996, 1.361], [2.665, 1.314]])
    actual = moments[["observed_mean", "observed_variance"]].to_numpy(float)
    if not np.allclose(actual, expected, atol=5e-4):
        raise RuntimeError("observed moments do not reproduce the committed table to three decimals")
    if not np.isclose(calibrated_survival, 0.498, atol=0.002):
        raise RuntimeError(f"unexpected control-calibrated survival: {calibrated_survival:.6f}")
    if (robustness.sse_winner == "EX1").any():
        failures = robustness.loc[robustness.sse_winner == "EX1"]
        raise RuntimeError(f"EX2 verdict changed in robustness sweep:\n{failures}")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    bins = np.arange(PRIMARY_N_BINS)
    for axis, time in zip(axes, TIMES_TO_FILES, strict=True):
        observed, ex2, ex1 = overlays[time]
        axis.plot(bins, observed, "o-", color="black", label="observed")
        axis.plot(bins, ex2, "s--", label="EX2")
        axis.plot(bins, ex1, "^--", label="EX1")
        axis.set_title(f"t = {time:g} min")
        axis.set_xlabel("nominal mass-shift bin")
    axes[0].set_ylabel("normalised intensity")
    axes[-1].legend(frameon=False)
    figure.suptitle(
        f"MoPrP peptide 1: centroid-matched EX2 vs EX1 (survival={calibrated_survival:.3f})"
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "envelope_overlay.png", dpi=200)
    plt.close(figure)

    print(f"effective survival: {calibrated_survival:.6f}")
    print(moments.to_string(index=False))
    ties = int((robustness.sse_winner == "tie_at_parameter_boundary").sum())
    print(
        "robustness verdict: no EX1 wins across 27 comparisons "
        f"({ties} uninformative boundary ties)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-dir", type=Path, default=SPECTRA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
