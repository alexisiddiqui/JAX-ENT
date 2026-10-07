#!/usr/bin/env python3
"""Run the gated target-semantics by fitter-semantics ISO matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from run_stage0_population_sweep import grid, materialize_cell_splits


SEMANTICS_TO_MODE = {"legacy": "log_pf", "fast": "rate", "slow2": "uptake"}
CONFIGS = (
    ("iso_bi", "residue", 0.001),
    ("iso_bi", "width10", 0.001),
    ("iso_tri", "residue", 0.0),
)


def run(command: list[str], dry_run: bool, log_path: Path | None = None) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        if log_path is None:
            subprocess.run(command, check=True)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as log_handle:
                subprocess.run(
                    command,
                    check=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )


def population_kl(assignments: np.ndarray, populations: dict[str, float]) -> float:
    labels = {"open": 0, "closed": 1, "intermediate": -1}
    prior = {
        name: float(np.mean(assignments == label))
        for name, label in labels.items()
        if np.any(assignments == label)
    }
    return float(
        sum(
            mass * np.log(mass / prior[name])
            for name, mass in populations.items()
            if mass > 0
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage0-dir",
        type=Path,
        required=True,
        help="Corrected Phase-1d directory supplying immutable split templates.",
    )
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-replicates", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--tau-values",
        type=str,
        default="0",
        help="Comma-separated target tau values in minutes; fitters remain tau=0.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    tau_values = tuple(float(value) for value in args.tau_values.split(","))
    if not tau_values or any(tau < 0 for tau in tau_values):
        parser.error("tau values must be a non-empty list of non-negative numbers")
    if args.max_workers < 1:
        parser.error("max-workers must be positive")
    target_semantics_values = (
        tuple(SEMANTICS_TO_MODE) if tau_values == (0.0,) else ("fast", "slow2")
    )
    output_dir = args.output_dir.resolve()
    stage0_dir = args.stage0_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fitting_root = Path(__file__).resolve().parent
    example_root = fitting_root.parents[1]
    generator = example_root / "data/generate_iso_targets.py"
    optimiser = fitting_root / "optimise_ISO_TRI_BI_splits_Sigma.py"
    clustering_root = example_root / "data/_clustering_results"

    assignments = {}
    for ensemble in ("iso_bi", "iso_tri"):
        table = np.genfromtxt(
            clustering_root / f"cluster_assignments_{ensemble.upper()}.csv",
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )
        assignments[ensemble] = np.asarray(table["cluster_assignment"])

    cells = []
    fit_commands: list[tuple[list[str], Path]] = []
    for ensemble, layout, maxent in CONFIGS:
        template_dir = stage0_dir / "_split_templates" / ensemble / layout / "datasplits"
        if not template_dir.exists() and not args.dry_run:
            raise FileNotFoundError(f"missing Stage-0 split template: {template_dir}")
        width = "10" if layout == "width10" else "0"
        for grid_ensemble, populations in grid():
            if grid_ensemble != ensemble:
                continue
            tilt = population_kl(assignments[ensemble], populations)
            if tilt >= 1.0:
                continue
            population_text = ",".join(
                f"{name}={value:.12g}" for name, value in populations.items()
            )
            population_name = "_".join(
                f"{name}-{value:.3f}" for name, value in populations.items()
            )
            for tau in tau_values:
                tau_name = f"tau-{tau:g}".replace(".", "p")
                for target_semantics in target_semantics_values:
                    target_dir = (
                        output_dir / ensemble / layout / population_name / tau_name
                        / f"target_{target_semantics}"
                    )
                    if not (target_dir / "target_dfrac.dat").exists():
                        run(
                            [
                                sys.executable,
                                str(generator),
                                "--ensemble", ensemble,
                                "--semantics", target_semantics,
                                "--tau", f"{tau:g}",
                                "--population", population_text,
                                "--peptide-width", width,
                                "--output-dir", str(target_dir),
                            ],
                            args.dry_run,
                        )
                    split_dir = target_dir / "datasplits"
                    if not args.dry_run and not split_dir.exists():
                        materialize_cell_splits(
                            template_dir, split_dir, target_dir / "target_dfrac.dat"
                        )
                    for fitter_semantics, mode in SEMANTICS_TO_MODE.items():
                        fit_dir = target_dir / f"fit_{fitter_semantics}"
                        histories = list((fit_dir / "random").glob("*_results.hdf5"))
                        if (
                            not args.prepare_only
                            and len(histories) < args.n_replicates
                        ):
                            fit_commands.append(
                                ([
                                    sys.executable,
                                    str(optimiser),
                                    "--ensemble", ensemble.upper(),
                                    "--loss-function", "MSE",
                                    "--split-types", "random",
                                    "--maxent-values", f"{maxent:g}",
                                    "--n-steps", str(args.n_steps),
                                    "--n-replicates", str(args.n_replicates),
                                    "--datasplit-dir", str(split_dir),
                                    "--output-dir", str(fit_dir),
                                    "--frame-averaging-mode", mode,
                                ], fit_dir / "optimizer.log")
                            )
                        cells.append(
                            {
                                "ensemble": ensemble,
                                "layout": layout,
                                "populations": populations,
                                "tilt_kl": tilt,
                                "maxent": maxent,
                                "tau": tau,
                                "target_semantics": target_semantics,
                                "fitter_semantics": fitter_semantics,
                                "target_dir": str(target_dir),
                                "datasplit_dir": str(split_dir),
                                "fit_dir": str(fit_dir),
                            }
                        )

    (output_dir / "phase2_matrix_manifest.json").write_text(
        json.dumps(
            {
                "tau_values": tau_values,
                "tilt_limit": 1.0,
                "configs": CONFIGS,
                "n_steps": args.n_steps,
                "n_replicates": args.n_replicates,
                "cells": cells,
            },
            indent=2,
        )
        + "\n"
    )
    if not args.prepare_only:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(run, command, args.dry_run, log_path)
                for command, log_path in fit_commands
            ]
            for future in futures:
                future.result()


if __name__ == "__main__":
    main()
