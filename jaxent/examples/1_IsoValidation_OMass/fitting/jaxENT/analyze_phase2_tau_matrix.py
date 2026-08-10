#!/usr/bin/env python3
"""Analyze the Phase-2 tau/EX1-contamination robustness arm."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.frame_averaging import (
    residue_uptake_fast,
    residue_uptake_legacy,
    residue_uptake_slow2,
)
from jaxent.examples.common.analysis.stats import effective_sample_size
from jaxent.src.analysis.hdx_rate_mixture import fit_shared_rate_mixture
from jaxent.src.analysis.pf_variance import kl_to_uniform
from jaxent.src.utils.hdf import load_optimization_history_from_file

SEMANTICS_FN = {
    "legacy": residue_uptake_legacy,
    "fast": residue_uptake_fast,
    "slow2": residue_uptake_slow2,
}


def load_generator(path: Path):
    spec = importlib.util.spec_from_file_location("generate_iso_targets_tau", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_log_rate(curves: np.ndarray, times: np.ndarray) -> float:
    """Return a well-determined one-component position on the log-rate axis."""
    fit = fit_shared_rate_mixture(
        curves,
        times,
        n_components=1,
        shrinkage=0.0,
        starts=1,
        maxiter=300,
    )
    return float(np.log(fit.rates[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.matrix_dir.resolve()
    manifest = json.loads((root / "phase2_matrix_manifest.json").read_text())
    fitting_root = Path(__file__).resolve().parent
    example_root = fitting_root.parents[1]
    generator = load_generator(example_root / "data/generate_iso_targets.py")
    shipped_segments = (
        example_root
        / "data/_output/mixed_60-40_artificial_expt_resfracs_TeaA_segs.txt"
    )

    assignments: dict[str, np.ndarray] = {}
    features: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ensemble in ("iso_bi", "iso_tri"):
        assignments[ensemble] = pd.read_csv(
            example_root
            / f"data/_clustering_results/cluster_assignments_{ensemble.upper()}.csv"
        )["cluster_assignment"].to_numpy(dtype=int)
        with np.load(fitting_root / f"_featurise/features_{ensemble}.npz") as data:
            features[ensemble] = (
                0.35 * data["heavy_contacts"] + 2.0 * data["acceptor_contacts"],
                np.asarray(data["k_ints"]),
            )

    def predict(weights, ensemble, layout, semantics):
        log_pf, k_ints = features[ensemble]
        kwargs = {"assignments": assignments[ensemble]} if semantics == "slow2" else {}
        uptake = SEMANTICS_FN[semantics](
            log_pf,
            k_ints,
            generator.TIMEPOINTS,
            weights,
            tau=0.0,
            **kwargs,
        )
        uptake, segments = generator.align_residue_layout(
            uptake,
            fitting_root / f"_featurise/topology_{ensemble}.json",
            shipped_segments,
        )
        if layout == "width10":
            uptake, _ = generator.peptide_aggregate(uptake, segments, 10)
        return uptake.T

    target_positions: dict[str, float] = {}
    rows = []
    for cell in manifest["cells"]:
        ensemble = cell["ensemble"]
        labels = (0, 1) if ensemble == "iso_bi" else (0, -1, 1)
        names = (
            ("open", "closed")
            if ensemble == "iso_bi"
            else ("open", "intermediate", "closed")
        )
        target_path = Path(cell["target_dir"]) / "target_dfrac.dat"
        target = np.loadtxt(target_path)
        target_key = str(target_path)
        if target_key not in target_positions:
            target_positions[target_key] = shared_log_rate(target, generator.TIMEPOINTS)
        for history_path in sorted(
            (Path(cell["fit_dir"]) / "random").glob("*_results.hdf5")
        ):
            history = load_optimization_history_from_file(str(history_path))
            state = history.get_best_state()
            weights = np.asarray(state.params.frame_weight_simplex, dtype=float)
            weights /= weights.sum()
            recovered = [
                float(weights[assignments[ensemble] == label].sum()) for label in labels
            ]
            fitted = predict(weights, ensemble, cell["layout"], cell["fitter_semantics"])
            residual = fitted - target
            fitted_position = shared_log_rate(fitted, generator.TIMEPOINTS)
            row = {
                "ensemble": ensemble,
                "layout": cell["layout"],
                "tau": cell["tau"],
                "maxent": cell["maxent"],
                "tilt_kl": cell["tilt_kl"],
                "target_semantics": cell["target_semantics"],
                "fitter_semantics": cell["fitter_semantics"],
                "history_file": str(history_path),
                "best_step": int(state.step),
                "mse_fit": float(np.mean(np.square(residual))),
                "mean_uptake_bias": float(np.mean(residual)),
                "target_log_rate": target_positions[target_key],
                "fitted_log_rate": fitted_position,
                "log_rate_position_shift": fitted_position - target_positions[target_key],
                "ess_total": effective_sample_size(weights),
                "kl_from_prior": float(kl_to_uniform(weights)),
            }
            for index, timepoint in enumerate(generator.TIMEPOINTS):
                row[f"uptake_bias_t{timepoint:g}"] = float(np.mean(residual[:, index]))
            for name in names:
                row[f"true_{name}"] = cell["populations"][name]
            for name, value in zip(names, recovered, strict=True):
                row[f"recovered_{name}"] = value
                row[f"error_{name}"] = value - cell["populations"][name]
            row["abs_open_error"] = abs(row["error_open"])
            rows.append(row)

    if len(rows) != 504:
        raise ValueError(f"expected 504 tau histories, found {len(rows)}")
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "phase2_tau_matrix.csv", index=False)
    summary = (
        frame.groupby(
            ["ensemble", "layout", "tau", "target_semantics", "fitter_semantics"],
            as_index=False,
        )
        .agg(
            bias_open=("error_open", "mean"),
            mae_open=("abs_open_error", "mean"),
            mean_uptake_bias=("mean_uptake_bias", "mean"),
            median_log_rate_shift=("log_rate_position_shift", "median"),
            median_mse=("mse_fit", "median"),
            median_ess=("ess_total", "median"),
            n_fits=("history_file", "size"),
        )
    )

    phase2 = pd.read_csv(args.phase2_dir / "phase2_semantics_matrix.csv")
    baseline = (
        phase2[phase2["target_semantics"] == phase2["fitter_semantics"]]
        .groupby(["ensemble", "layout", "fitter_semantics"], as_index=False)
        .agg(
            tau0_bias_open=("error_open", "mean"),
            tau0_mae_open=("abs_open_error", "mean"),
        )
    )
    summary = summary.merge(
        baseline,
        on=["ensemble", "layout", "fitter_semantics"],
        how="left",
        validate="many_to_one",
    )
    summary["bias_open_minus_tau0_diagonal"] = (
        summary["bias_open"] - summary["tau0_bias_open"]
    )
    summary.to_csv(root / "phase2_tau_summary.csv", index=False)


if __name__ == "__main__":
    main()
