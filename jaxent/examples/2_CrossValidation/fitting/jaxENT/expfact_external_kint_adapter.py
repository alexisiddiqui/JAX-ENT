#!/usr/bin/env python3
"""Pinned exPfact-compatible diagnostic refit with an explicit external kint file.

This adapter targets pacilab/exPfact revision 1f99a0ccccb5c3862f9a3ae74cf570eb0475c059,
the revision that introduced the published MoPrP ``median.pfact``. It preserves the
trim-one peptide map, unweighted objective, and harmonic penalty through JAX-ENT's
regression-tested exPfact reproduction. Execution is deliberately gated until all
primary and rate-sensitivity results have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from jaxent.src.analysis.hdx_ex2 import (
    fit_ex2_solution_set,
    load_expfact_dataset,
    predict_ex2_uptake,
)


PINNED_EXPFACT_REVISION = "1f99a0ccccb5c3862f9a3ae74cf570eb0475c059"
HARMONIC_STRENGTH = 1e-8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_external_kints(path: Path, sequence: str) -> np.ndarray:
    """Strictly load one-based min^-1 rates using MoPrP sentinel conventions."""

    values = np.loadtxt(path, comments="#", dtype=float)
    if values.ndim != 2 or values.shape != (len(sequence), 2):
        raise ValueError("kint file must have exactly one residue_id/rate row per sequence residue")
    ids = values[:, 0]
    expected = np.arange(1, len(sequence) + 1)
    if not np.array_equal(ids, expected):
        raise ValueError("kint residue IDs must be unique, integral, one-based, and sequence-aligned")
    rates = values[:, 1]
    if np.any(~np.isfinite(rates)) or np.any(rates == 0):
        raise ValueError("kint rates must be finite and either positive or the -1 sentinel")
    if np.any((rates < 0) & (rates != -1)):
        raise ValueError("the only accepted negative kint sentinel is -1")
    expected_sentinels = {1} | {i + 1 for i, amino_acid in enumerate(sequence) if amino_acid == "P"}
    actual_sentinels = set(expected[rates == -1].tolist())
    if actual_sentinels != expected_sentinels:
        raise ValueError(
            f"invalid sentinel placement: expected {sorted(expected_sentinels)}, "
            f"found {sorted(actual_sentinels)}"
        )
    return rates


def _rmse(predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted[mask] - observed[mask]) ** 2)))


def run(args: argparse.Namespace) -> None:
    if not args.primary_results_frozen:
        raise RuntimeError(
            "Phase 5 is deferred: pass --primary-results-frozen only after all primary and "
            "rate-sensitivity manifests are finalized"
        )
    if args.units != "min^-1":
        raise ValueError("external kint units must be declared explicitly as min^-1")
    data_dir = args.data_dir.resolve()
    dataset = load_expfact_dataset(data_dir)
    rates = load_external_kints(args.kint_file.resolve(), dataset.sequence)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    peptide_count, time_count = dataset.observed_uptake.shape
    full_mask = np.ones((peptide_count, time_count), dtype=bool)
    peptide_holdout = np.zeros_like(full_mask)
    peptide_holdout[0, :] = True
    blocked_holdout = np.zeros_like(full_mask)
    blocked_holdout[:, args.blocked_time_start : args.blocked_time_stop] = True

    fit = fit_ex2_solution_set(
        dataset.observed_uptake,
        rates,
        dataset.protocol.timepoints_min,
        dataset.peptide_map,
        starts=args.starts,
        seed=args.seed,
        harmonic_strength=HARMONIC_STRENGTH,
        maxiter=args.maxiter,
    ).best
    published = np.loadtxt(data_dir / "median.pfact", dtype=float)[:, 1]
    published_prediction = predict_ex2_uptake(
        published, rates, dataset.protocol.timepoints_min, dataset.peptide_map
    )
    metrics = {
        "refit": {
            "training_objective": fit.objective,
            "training_rmse": _rmse(fit.predicted, dataset.observed_uptake, full_mask),
            "held_out_peptide_1_rmse": _rmse(
                fit.predicted, dataset.observed_uptake, peptide_holdout
            ),
            "blocked_timepoint_rmse": _rmse(
                fit.predicted, dataset.observed_uptake, blocked_holdout
            ),
        },
        "published_median_under_external_kints": {
            "training_rmse": _rmse(published_prediction, dataset.observed_uptake, full_mask),
            "held_out_peptide_1_rmse": _rmse(
                published_prediction, dataset.observed_uptake, peptide_holdout
            ),
            "blocked_timepoint_rmse": _rmse(
                published_prediction, dataset.observed_uptake, blocked_holdout
            ),
        },
    }
    training_improved = metrics["refit"]["training_rmse"] < metrics[
        "published_median_under_external_kints"
    ]["training_rmse"]
    heldout_improved = (
        metrics["refit"]["held_out_peptide_1_rmse"]
        < metrics["published_median_under_external_kints"]["held_out_peptide_1_rmse"]
        and metrics["refit"]["blocked_timepoint_rmse"]
        < metrics["published_median_under_external_kints"]["blocked_timepoint_rmse"]
    )
    classification = (
        "overfit" if training_improved and not heldout_improved else
        "scientifically_supported" if heldout_improved else "not_supported"
    )
    pf_path = args.output_dir / "external_kint_refit.pfact"
    with pf_path.open("x") as handle:
        for residue_id, log_pf in zip(dataset.peptide_map.residue_ids, fit.log_pf, strict=True):
            value = -1.0 if not np.isfinite(log_pf) else float(log_pf)
            handle.write(f"{int(residue_id)} {value:.17g}\n")
    manifest = {
        "diagnostic_only": True,
        "may_replace_median_pfact": False,
        "pinned_expfact_revision": PINNED_EXPFACT_REVISION,
        "adapter": str(Path(__file__).resolve()),
        "external_kint": {
            "path": str(args.kint_file.resolve()), "sha256": sha256(args.kint_file),
            "units": args.units,
        },
        "harmonic_strength": HARMONIC_STRENGTH,
        "metrics": metrics,
        "classification": classification,
        "classification_rule": (
            "overfit when fitted-data RMSE improves without improvement on both preregistered "
            "held-out experimental readouts"
        ),
        "output": {"path": str(pf_path.resolve()), "sha256": sha256(pf_path)},
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kint-file", type=Path, required=True)
    parser.add_argument("--units", choices=("min^-1",), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-results-frozen", action="store_true")
    parser.add_argument("--starts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--maxiter", type=int, default=5000)
    parser.add_argument("--blocked-time-start", type=int, default=5)
    parser.add_argument("--blocked-time-stop", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
