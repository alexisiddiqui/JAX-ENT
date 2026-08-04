#!/usr/bin/env python3
"""Run the self-consistent legacy Stage 0 population-calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def grid() -> list[tuple[str, dict[str, float]]]:
    cells = []
    for open_population in (0.02, 0.15, 0.40, 0.65):
        for intermediate in (0.05, 0.30, 0.607, 0.85):
            closed = 1.0 - open_population - intermediate
            if closed >= 0.05 - 1e-12:
                cells.append(
                    (
                        "iso_tri",
                        {
                            "open": open_population,
                            "intermediate": intermediate,
                            "closed": closed,
                        },
                    )
                )
    for open_population in (0.05, 0.15, 0.40, 0.65, 0.90):
        cells.append(("iso_bi", {"open": open_population, "closed": 1.0 - open_population}))
    return cells


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def write_split_dfrac(path: Path, dfrac: np.ndarray, topology_path: Path) -> None:
    """Select target rows using immutable split topology fragment indices."""
    topology = json.loads(topology_path.read_text())
    indices = [int(item["fragment_index"]) for item in topology["topologies"]]
    if any(index < 0 or index >= len(dfrac) for index in indices):
        raise ValueError(f"fragment index outside target layout in {topology_path}")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["datapoint_type", "feature_length", *range(dfrac.shape[1])])
        for values in dfrac[indices]:
            writer.writerow(["HDX_peptide", dfrac.shape[1], *values])


def materialize_cell_splits(template_dir: Path, cell_dir: Path, dfrac_path: Path) -> None:
    """Copy fixed topologies and attach this cell's target values."""
    dfrac = np.loadtxt(dfrac_path)
    random_template = template_dir / "random"
    random_cell = cell_dir / "random"
    random_cell.mkdir(parents=True, exist_ok=True)
    for template_split in sorted(random_template.glob("split_*")):
        cell_split = random_cell / template_split.name
        cell_split.mkdir(parents=True, exist_ok=True)
        for subset in ("train", "val"):
            topology_name = f"{subset}_topology.json"
            shutil.copyfile(template_split / topology_name, cell_split / topology_name)
            write_split_dfrac(
                cell_split / f"{subset}_dfrac.csv",
                dfrac,
                cell_split / topology_name,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--n-replicates", type=int, default=3)
    parser.add_argument(
        "--maxent-values",
        default="0,0.001,0.01,0.1,1",
        help="Comma-separated MaxEnt regularisation strengths.",
    )
    parser.add_argument("--layouts", default="residue,width10")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate targets and fixed datasplits without launching fits.",
    )
    args = parser.parse_args()
    try:
        maxent_values = [float(value) for value in args.maxent_values.split(",")]
    except ValueError as exc:
        parser.error(f"invalid --maxent-values: {exc}")
    if (
        args.n_steps < 1
        or args.n_replicates < 1
        or not maxent_values
        or any(value < 0 for value in maxent_values)
    ):
        parser.error("steps and replicates must be positive; MaxEnt strengths must be non-negative")
    args.output_dir = args.output_dir.resolve()

    fitting_root = Path(__file__).resolve().parent
    generator = fitting_root.parents[1] / "data/generate_iso_targets.py"
    splitter = fitting_root / "splitdata_ISO.py"
    optimiser = fitting_root / "optimise_ISO_TRI_BI_splits_Sigma.py"
    layouts = [item.strip() for item in args.layouts.split(",") if item.strip()]
    invalid = set(layouts) - {"residue", "width10"}
    if invalid:
        parser.error(f"unknown layouts: {sorted(invalid)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    templates: dict[tuple[str, str], Path] = {}
    # Split exactly once per ensemble/layout.  The reference target supplies
    # values required by the splitter API; only its topology selection survives.
    for ensemble in ("iso_tri", "iso_bi"):
        reference_populations = next(populations for name, populations in grid() if name == ensemble)
        population_text = ",".join(
            f"{name}={value:.12g}" for name, value in reference_populations.items()
        )
        for layout in layouts:
            width = "10" if layout == "width10" else "0"
            template_root = args.output_dir / "_split_templates" / ensemble / layout
            target_dir = template_root / "reference_target"
            split_dir = template_root / "datasplits"
            topology_files = list((split_dir / "random").glob("split_*/*_topology.json"))
            if len(topology_files) != 2 * args.n_replicates:
                run(
                    [
                        sys.executable,
                        str(generator),
                        "--ensemble",
                        ensemble,
                        "--semantics",
                        "legacy",
                        "--tau",
                        "0",
                        "--population",
                        population_text,
                        "--peptide-width",
                        width,
                        "--output-dir",
                        str(target_dir),
                    ],
                    args.dry_run,
                )
                run(
                    [
                        sys.executable,
                        str(splitter),
                        "--dfrac-file",
                        str(target_dir / "target_dfrac.dat"),
                        "--segs-file",
                        str(target_dir / "target_segs.txt"),
                        "--output-dir",
                        str(split_dir),
                        "--ensemble",
                        ensemble,
                        "--split-types",
                        "random",
                        "--num-splits",
                        str(args.n_replicates),
                    ],
                    args.dry_run,
                )
            templates[(ensemble, layout)] = split_dir

    cells = []
    for ensemble, populations in grid():
        population_text = ",".join(f"{name}={value:.12g}" for name, value in populations.items())
        cell_name = "_".join(
            [ensemble, *(f"{name}-{value:.3f}" for name, value in populations.items())]
        )
        for layout in layouts:
            root = args.output_dir / ensemble / layout / cell_name
            target_dir, split_dir, fit_dir = root / "target", root / "datasplits", root / "fit"
            width = "10" if layout == "width10" else "0"
            if not (target_dir / "target_dfrac.dat").exists():
                run(
                    [
                        sys.executable,
                        str(generator),
                        "--ensemble",
                        ensemble,
                        "--semantics",
                        "legacy",
                        "--tau",
                        "0",
                        "--population",
                        population_text,
                        "--peptide-width",
                        width,
                        "--output-dir",
                        str(target_dir),
                    ],
                    args.dry_run,
                )
            if not args.dry_run and not split_dir.exists():
                materialize_cell_splits(templates[(ensemble, layout)], split_dir, target_dir / "target_dfrac.dat")
            fit_dirs = {}
            for maxent in maxent_values:
                # The Phase 1b maxent=1 run remains valid: at equal weights the
                # old and corrected slot assignments are identical.
                strength_dir = fit_dir if maxent == 1 else fit_dir / f"maxent_{maxent:g}"
                fit_dirs[f"{maxent:g}"] = str(strength_dir)
                completed_histories = list((strength_dir / "random").glob("*_results.hdf5"))
                if not args.prepare_only and len(completed_histories) < args.n_replicates:
                    run(
                        [
                            sys.executable,
                            str(optimiser),
                            "--ensemble",
                            ensemble.upper(),
                            "--loss-function",
                            "MSE",
                            "--split-types",
                            "random",
                            "--maxent-values",
                            f"{maxent:g}",
                            "--n-steps",
                            str(args.n_steps),
                            "--n-replicates",
                            str(args.n_replicates),
                            "--datasplit-dir",
                            str(split_dir),
                            "--output-dir",
                            str(strength_dir),
                        ],
                        args.dry_run,
                    )
            cells.append(
                {
                    "ensemble": ensemble,
                    "layout": layout,
                    "populations": populations,
                    "root": str(root),
                    "split_template": str(templates[(ensemble, layout)]),
                    "fit_dirs": fit_dirs,
                }
            )
    (args.output_dir / "stage0_sweep_manifest.json").write_text(
        json.dumps(
            {
                "semantics": "legacy",
                "tau": 0.0,
                "maxent_values": maxent_values,
                "cells": cells,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
