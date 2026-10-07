#!/usr/bin/env python3
"""Run resumable HDX CPU scaling benchmark suites in isolated processes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import nnls


PATHS = ("eager", "jit", "pure")
PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-5
DEFAULT_ORDER_SEED = 20260726


@dataclass(frozen=True)
class Configuration:
    name: str
    residues: int
    frames: int
    timepoints: int


def full_configurations() -> tuple[Configuration, ...]:
    return (
        Configuration("anchor", 144, 500, 5),
        Configuration("residues_096", 96, 500, 5),
        Configuration("residues_293", 293, 500, 5),
        Configuration("residues_600", 600, 500, 5),
        Configuration("frames_0173", 144, 173, 5),
        Configuration("frames_1125", 144, 1125, 5),
        Configuration("frames_5000", 144, 5000, 5),
        Configuration("timepoints_01", 144, 500, 1),
        Configuration("timepoints_03", 144, 500, 3),
        Configuration("timepoints_07", 144, 500, 7),
        Configuration("timepoints_10", 144, 500, 10),
        Configuration("corner_low", 96, 173, 1),
        Configuration("corner_high", 600, 5000, 10),
    )


def stage_configurations() -> tuple[Configuration, ...]:
    selected = {
        "anchor",
        "residues_600",
        "frames_5000",
        "timepoints_10",
        "corner_high",
    }
    return tuple(config for config in full_configurations() if config.name in selected)


def parse_paths(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if (
        not parsed
        or len(set(parsed)) != len(parsed)
        or any(path not in PATHS for path in parsed)
    ):
        raise argparse.ArgumentTypeError(
            "paths must be a comma-separated subset of eager,jit,pure"
        )
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("full", "stage"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("profiling/_output/hdx_cpu_scaling"),
    )
    parser.add_argument("--paths", type=parse_paths, default=PATHS)
    parser.add_argument("--steps", type=positive_int, default=1000)
    parser.add_argument("--warm-repeats", type=positive_int, default=3)
    parser.add_argument("--order-seed", type=int, default=DEFAULT_ORDER_SEED)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--previous-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _git_output(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _repository_metadata() -> dict[str, Any]:
    diff = _git_output(["diff", "--binary"])
    status = _git_output(["status", "--porcelain", "--untracked-files=no"])
    return {
        "git_revision": _git_output(["rev-parse", "HEAD"]),
        "git_dirty": bool(status),
        "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
    }


def _host_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    if sys.platform == "darwin":
        try:
            hardware = json.loads(
                subprocess.check_output(
                    ["system_profiler", "SPHardwareDataType", "-json"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            )["SPHardwareDataType"][0]
            metadata["hardware"] = {
                key: hardware[key]
                for key in (
                    "chip_type",
                    "machine_model",
                    "machine_name",
                    "number_processors",
                    "physical_memory",
                )
                if key in hardware
            }
        except (OSError, KeyError, IndexError, json.JSONDecodeError, subprocess.CalledProcessError):
            pass
    metadata["benchmark_environment"] = {
        key: os.environ[key]
        for key in (
            "JAX_PLATFORMS",
            "JAX_PLATFORM_NAME",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        if key in os.environ
    }
    metadata["cell_environment_overrides"] = {
        "JAX_PLATFORMS": "cpu",
        "PYTHONUNBUFFERED": "1",
    }
    return metadata


def _cell_key(config: Configuration, path: str) -> str:
    return f"{config.name}__{path}"


def _cell_matches(
    report_path: Path,
    terminal_path: Path,
    config: Configuration,
    path: str,
    steps: int,
    warm_repeats: int,
) -> bool:
    if not report_path.exists() or not terminal_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
        run = report["run_config"]
        timing = report["timing"][0]
    except (OSError, KeyError, IndexError, json.JSONDecodeError):
        return False
    return bool(
        report.get("valid")
        and report.get("schema_version") == 2
        and run.get("path") == path
        and run.get("steps") == steps
        and run.get("warm_repeats") == warm_repeats
        and run.get("residues") == config.residues
        and run.get("frames") == config.frames
        and run.get("timepoints") == config.timepoints
        and timing.get("cold", {}).get("steps_completed") == steps
    )


def _load_snapshot(path: Path) -> tuple[np.ndarray, ...]:
    with np.load(path, allow_pickle=False) as data:
        return tuple(np.asarray(data[key]) for key in sorted(data.files))


def _snapshot_difference(
    reference: Sequence[np.ndarray],
    candidate: Sequence[np.ndarray],
) -> tuple[bool, float, float]:
    if len(reference) != len(candidate):
        return False, float("inf"), float("inf")
    close = True
    max_abs = 0.0
    max_rel = 0.0
    for expected, actual in zip(reference, candidate):
        if expected.shape != actual.shape:
            return False, float("inf"), float("inf")
        delta = np.abs(actual - expected)
        max_abs = max(max_abs, float(np.max(delta, initial=0.0)))
        denominator = np.maximum(np.abs(expected), PARITY_ATOL)
        max_rel = max(
            max_rel,
            float(np.max(delta / denominator, initial=0.0)),
        )
        close = close and bool(
            np.allclose(
                expected,
                actual,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
                equal_nan=False,
            )
        )
    return close, max_abs, max_rel


def _load_comparison_rows(directory: Path | None) -> dict[tuple[str, str], dict[str, Any]]:
    if directory is None:
        return {}
    path = directory / "aggregate.json"
    if not path.exists():
        raise FileNotFoundError(f"comparison aggregate does not exist: {path}")
    payload = json.loads(path.read_text())
    return {
        (row["configuration"], row["path"]): row
        for row in payload.get("rows", [])
    }


def _regression_flag(
    current_s: float,
    current_mad_s: float,
    reference_s: float,
    reference_mad_s: float,
) -> bool:
    delta = current_s - reference_s
    threshold = max(
        0.10 * reference_s,
        3.0 * (current_mad_s + reference_mad_s),
    )
    return delta > threshold


def _comparison_fields(
    row: dict[str, Any],
    reference: dict[str, Any] | None,
    prefix: str,
) -> dict[str, Any]:
    if reference is None:
        return {}
    current = float(row["warm_median_s"])
    baseline = float(reference["warm_median_s"])
    current_mad = float(row["warm_mad_s"])
    baseline_mad = float(reference["warm_mad_s"])
    return {
        f"{prefix}_warm_delta_s": current - baseline,
        f"{prefix}_warm_delta_pct": 100.0 * (current / baseline - 1.0),
        f"{prefix}_speedup": baseline / current,
        f"{prefix}_regression": _regression_flag(
            current,
            current_mad,
            baseline,
            baseline_mad,
        ),
    }


def _fit_scaling(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for path in PATHS:
        selected = [
            row
            for row in rows
            if row["path"] == path and row["cell_valid"]
        ]
        if len(selected) < 3:
            continue
        design = np.asarray(
            [
                [
                    1.0,
                    float(row["residues"] * row["frames"]),
                    float(row["residues"] * row["timepoints"]),
                ]
                for row in selected
            ],
            dtype=np.float64,
        )
        response = np.asarray(
            [row["warm_median_s"] for row in selected],
            dtype=np.float64,
        )
        scales = np.maximum(np.max(design, axis=0), 1.0)
        scaled_coefficients, _ = nnls(design / scales, response)
        coefficients = scaled_coefficients / scales
        prediction = design @ coefficients
        residual = float(np.sum((response - prediction) ** 2))
        total = float(np.sum((response - response.mean()) ** 2))
        models[path] = {
            "intercept_s": float(coefficients[0]),
            "seconds_per_residue_frame": float(coefficients[1]),
            "seconds_per_residue_timepoint": float(coefficients[2]),
            "r_squared": 1.0 - residual / total if total else 1.0,
            "fit": "non-negative least squares",
        }
    return models


def _axis_elasticities(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selectors = {
        "residues": lambda row: row["frames"] == 500 and row["timepoints"] == 5,
        "frames": lambda row: row["residues"] == 144 and row["timepoints"] == 5,
        "timepoints": lambda row: row["residues"] == 144 and row["frames"] == 500,
    }
    for path in PATHS:
        path_result: dict[str, float] = {}
        for axis, selector in selectors.items():
            selected = [
                row
                for row in rows
                if row["path"] == path and row["cell_valid"] and selector(row)
            ]
            if len(selected) < 2:
                continue
            x = np.log(np.asarray([row[axis] for row in selected], dtype=np.float64))
            y = np.log(
                np.asarray([row["warm_median_s"] for row in selected], dtype=np.float64)
            )
            path_result[axis] = float(np.polyfit(x, y, 1)[0])
        result[path] = path_result
    return result


def _path_speedups(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    configurations = sorted({row["configuration"] for row in rows})
    for configuration in configurations:
        selected = {
            row["path"]: float(row["warm_median_s"])
            for row in rows
            if row["configuration"] == configuration and row["cell_valid"]
        }
        if "pure" not in selected:
            continue
        result[configuration] = {
            "pure_vs_eager": (
                selected["eager"] / selected["pure"]
                if "eager" in selected
                else None
            ),
            "pure_vs_jit": (
                selected["jit"] / selected["pure"]
                if "jit" in selected
                else None
            ),
        }
    return result


def _aggregate(
    run_dir: Path,
    configurations: Sequence[Configuration],
    paths: Sequence[str],
    baseline_dir: Path | None,
    previous_dir: Path | None,
) -> dict[str, Any]:
    baseline_rows = _load_comparison_rows(baseline_dir)
    previous_rows = _load_comparison_rows(previous_dir)
    rows: list[dict[str, Any]] = []
    reports: dict[tuple[str, str], dict[str, Any]] = {}
    snapshots: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}

    for config in configurations:
        for path in paths:
            key = _cell_key(config, path)
            report_path = run_dir / "cells" / f"{key}.json"
            terminal_path = run_dir / "cells" / f"{key}.npz"
            if not report_path.exists():
                continue
            report = json.loads(report_path.read_text())
            timing = report["timing"][0]
            reports[(config.name, path)] = report
            if terminal_path.exists():
                snapshots[(config.name, path)] = _load_snapshot(terminal_path)
            warm_hosts = [
                sample["host_materialisations_per_step"]
                for sample in timing["warm"]
            ]
            row = {
                "configuration": config.name,
                "residues": config.residues,
                "frames": config.frames,
                "timepoints": config.timepoints,
                "path": path,
                "cell_valid": bool(report.get("valid")),
                "cold_s": timing["cold"]["elapsed_s"],
                "cold_compile_s": timing["cold"]["compile_s"],
                "cold_compiles": timing["cold"]["compiles"],
                "warm_median_s": timing["warm_median_s"],
                "warm_mad_s": timing["warm_mad_s"],
                "warm_steps_per_s": timing["warm_median_steps_per_s"],
                "warm_compiles": timing["warm_compiles"],
                "warm_host_materialisations_per_step": float(np.median(warm_hosts)),
                "final_loss": timing["cold"]["final_loss"],
                "parameter_fingerprint": timing["cold"]["parameter_fingerprint"],
                "cross_path_parity": None,
                "cross_path_max_abs": None,
                "cross_path_max_rel": None,
            }
            row.update(
                _comparison_fields(
                    row,
                    baseline_rows.get((config.name, path)),
                    "baseline",
                )
            )
            row.update(
                _comparison_fields(
                    row,
                    previous_rows.get((config.name, path)),
                    "previous",
                )
            )
            rows.append(row)

    row_lookup = {(row["configuration"], row["path"]): row for row in rows}
    parity_failures: list[dict[str, Any]] = []
    for config in configurations:
        reference_key = (config.name, "eager")
        reference = snapshots.get(reference_key)
        reference_report = reports.get(reference_key)
        if reference is None or reference_report is None:
            continue
        reference_loss = float(reference_report["timing"][0]["cold"]["final_loss"])
        eager_row = row_lookup.get(reference_key)
        if eager_row is not None:
            eager_row["cross_path_parity"] = True
            eager_row["cross_path_max_abs"] = 0.0
            eager_row["cross_path_max_rel"] = 0.0
        for path in paths:
            if path == "eager":
                continue
            candidate_key = (config.name, path)
            candidate = snapshots.get(candidate_key)
            candidate_report = reports.get(candidate_key)
            candidate_row = row_lookup.get(candidate_key)
            if candidate is None or candidate_report is None or candidate_row is None:
                continue
            close, max_abs, max_rel = _snapshot_difference(reference, candidate)
            candidate_loss = float(candidate_report["timing"][0]["cold"]["final_loss"])
            loss_close = bool(
                np.isclose(
                    reference_loss,
                    candidate_loss,
                    rtol=PARITY_RTOL,
                    atol=PARITY_ATOL,
                )
            )
            parity = close and loss_close
            candidate_row["cross_path_parity"] = parity
            candidate_row["cross_path_max_abs"] = max_abs
            candidate_row["cross_path_max_rel"] = max_rel
            if not parity:
                parity_failures.append(
                    {
                        "configuration": config.name,
                        "path": path,
                        "max_abs": max_abs,
                        "max_rel": max_rel,
                        "eager_loss": reference_loss,
                        "candidate_loss": candidate_loss,
                    }
                )

    regressions = [
        {
            "configuration": row["configuration"],
            "path": row["path"],
            "baseline_delta_pct": row.get("baseline_warm_delta_pct"),
        }
        for row in rows
        if row.get("baseline_regression")
    ]
    return {
        "rows": rows,
        "scaling_model": _fit_scaling(rows),
        "axis_elasticities": _axis_elasticities(rows),
        "path_speedups": _path_speedups(rows),
        "parity_failures": parity_failures,
        "regressions": regressions,
        "valid": bool(rows)
        and all(row["cell_valid"] for row in rows)
        and not parity_failures,
    }


def _format_float(value: Any, precision: int = 4) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{precision}g}"


def _configuration_label(config: Configuration) -> str:
    return (
        f"{config.name}  "
        f"(R={config.residues}, F={config.frames}, T={config.timepoints})"
    )


def _save_heatmap(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(
        temporary,
        format="png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    temporary.replace(path)


def _write_heatmaps(
    run_dir: Path,
    configurations: Sequence[Configuration],
    paths: Sequence[str],
    aggregate: dict[str, Any],
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    heatmap_dir = run_dir / "heatmaps"
    outputs: dict[str, str] = {}
    configuration_names = [config.name for config in configurations]
    y_labels = [_configuration_label(config) for config in configurations]

    runtime_lookup = {
        (row["configuration"], row["path"]): float(row["warm_median_s"])
        for row in aggregate["rows"]
        if row["cell_valid"] and float(row["warm_median_s"]) > 0.0
    }
    runtime_values = np.asarray(
        [
            [runtime_lookup.get((name, path), np.nan) for path in paths]
            for name in configuration_names
        ],
        dtype=np.float64,
    )
    finite_runtime = runtime_values[np.isfinite(runtime_values)]
    if finite_runtime.size:
        runtime_cmap = plt.get_cmap("viridis").copy()
        runtime_cmap.set_bad("#eeeeee")
        runtime_min = float(finite_runtime.min())
        runtime_max = float(finite_runtime.max())
        runtime_norm = (
            LogNorm(vmin=runtime_min, vmax=runtime_max)
            if runtime_max > runtime_min
            else Normalize(vmin=runtime_min * 0.9, vmax=runtime_max * 1.1)
        )
        figure, axis = plt.subplots(
            figsize=(7.5, max(4.0, 0.48 * len(configurations) + 1.8))
        )
        image = axis.imshow(
            np.ma.masked_invalid(runtime_values),
            aspect="auto",
            cmap=runtime_cmap,
            norm=runtime_norm,
        )
        axis.set_xticks(range(len(paths)), labels=[path.upper() for path in paths])
        axis.set_yticks(range(len(y_labels)), labels=y_labels)
        axis.set_title("Warm runtime by measured configuration")
        axis.set_xlabel("Execution path")
        for row_index in range(runtime_values.shape[0]):
            for column_index in range(runtime_values.shape[1]):
                value = runtime_values[row_index, column_index]
                if not np.isfinite(value):
                    label = "—"
                    colour = "#555555"
                else:
                    label = f"{value:.3f}" if value < 1.0 else f"{value:.2f}"
                    colour = (
                        "white"
                        if float(runtime_norm(value)) < 0.48
                        or float(runtime_norm(value)) > 0.82
                        else "black"
                    )
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=8,
                )
        colour_bar = figure.colorbar(image, ax=axis, pad=0.03)
        colour_bar.set_label("Warm median runtime (seconds, log scale)")
        runtime_path = heatmap_dir / "warm_runtime_heatmap.png"
        _save_heatmap(figure, runtime_path)
        plt.close(figure)
        outputs["warm_runtime"] = runtime_path.relative_to(run_dir).as_posix()

    speedup_names = ("pure_vs_eager", "pure_vs_jit")
    speedup_labels = ("Pure vs eager", "Pure vs JIT")
    speedups = aggregate["path_speedups"]
    raw_speedups = np.asarray(
        [
            [
                speedups.get(name, {}).get(comparison, np.nan)
                for comparison in speedup_names
            ]
            for name in configuration_names
        ],
        dtype=np.float64,
    )
    valid_speedups = np.isfinite(raw_speedups) & (raw_speedups > 0.0)
    if np.any(valid_speedups):
        log_speedups = np.full_like(raw_speedups, np.nan)
        log_speedups[valid_speedups] = np.log2(raw_speedups[valid_speedups])
        maximum = max(float(np.nanmax(np.abs(log_speedups))), 1e-6)
        speedup_norm = Normalize(vmin=-maximum, vmax=maximum)
        speedup_cmap = plt.get_cmap("RdBu").copy()
        speedup_cmap.set_bad("#eeeeee")
        figure, axis = plt.subplots(
            figsize=(7.5, max(4.0, 0.48 * len(configurations) + 1.8))
        )
        image = axis.imshow(
            np.ma.masked_invalid(log_speedups),
            aspect="auto",
            cmap=speedup_cmap,
            norm=speedup_norm,
        )
        axis.set_xticks(range(len(speedup_labels)), labels=speedup_labels)
        axis.set_yticks(range(len(y_labels)), labels=y_labels)
        axis.set_title("Pure-path speedup by measured configuration")
        axis.set_xlabel("Comparison (>1× means pure is faster)")
        for row_index in range(raw_speedups.shape[0]):
            for column_index in range(raw_speedups.shape[1]):
                value = raw_speedups[row_index, column_index]
                if not np.isfinite(value) or value <= 0.0:
                    label = "—"
                    colour = "#555555"
                else:
                    label = f"{value:.2f}×"
                    colour = (
                        "white"
                        if abs(float(log_speedups[row_index, column_index]))
                        > 0.55 * maximum
                        else "black"
                    )
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=colour,
                    fontsize=8,
                )
        colour_bar = figure.colorbar(image, ax=axis, pad=0.03)
        colour_bar.set_label("log₂ speedup (0 = equal runtime)")
        speedup_path = heatmap_dir / "pure_speedup_heatmap.png"
        _save_heatmap(figure, speedup_path)
        plt.close(figure)
        outputs["pure_speedup"] = speedup_path.relative_to(run_dir).as_posix()

    return outputs


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    aggregate: dict[str, Any],
) -> None:
    lines = [
        f"# HDX CPU scaling: {manifest['run_id']}",
        "",
        f"- Suite: `{manifest['suite']}`",
        f"- Steps: {manifest['steps']}; cold samples: 1; warm samples: {manifest['warm_repeats']}",
        f"- Cells: {len(aggregate['rows'])}; aggregate valid: `{aggregate['valid']}`",
        f"- Cross-path parity failures: {len(aggregate['parity_failures'])}",
    ]
    heatmaps = aggregate.get("heatmaps", {})
    if heatmaps:
        lines.extend(["", "## Heatmaps", ""])
        if "warm_runtime" in heatmaps:
            lines.extend(
                [
                    "Measured warm runtimes (logarithmic colour scale):",
                    "",
                    f"![Warm runtime heatmap]({heatmaps['warm_runtime']})",
                    "",
                ]
            )
        if "pure_speedup" in heatmaps:
            lines.extend(
                [
                    "Pure-path speedup; values above 1× mean pure is faster:",
                    "",
                    f"![Pure path speedup heatmap]({heatmaps['pure_speedup']})",
                    "",
                ]
            )
    lines.extend(
        [
            "## Measurements",
            "",
            "| Configuration | R | F | T | Path | Warm median (s) | MAD (s) | Steps/s | Cold (s) | Host/step | Parity |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in aggregate["rows"]:
        lines.append(
            "| {configuration} | {residues} | {frames} | {timepoints} | {path} | "
            "{warm} | {mad} | {rate} | {cold} | {hosts} | {parity} |".format(
                **row,
                warm=_format_float(row["warm_median_s"]),
                mad=_format_float(row["warm_mad_s"]),
                rate=_format_float(row["warm_steps_per_s"]),
                cold=_format_float(row["cold_s"]),
                hosts=_format_float(row["warm_host_materialisations_per_step"]),
                parity=(
                    "pass"
                    if row["cross_path_parity"] is True
                    else "FAIL"
                    if row["cross_path_parity"] is False
                    else "—"
                ),
            )
        )

    lines.extend(["", "## Scaling model", ""])
    for model_path, model in aggregate["scaling_model"].items():
        lines.append(
            f"- `{model_path}`: intercept={model['intercept_s']:.6g}s, "
            f"β(R×F)={model['seconds_per_residue_frame']:.6g}, "
            f"β(R×T)={model['seconds_per_residue_timepoint']:.6g}, "
            f"R²={model['r_squared']:.4f}"
        )
    lines.extend(["", "## Axis elasticities", ""])
    for model_path, elasticities in aggregate["axis_elasticities"].items():
        formatted = ", ".join(
            f"{axis}={value:.4f}" for axis, value in elasticities.items()
        )
        lines.append(f"- `{model_path}`: {formatted}")
    lines.extend(["", "## Path speedups", ""])
    for configuration, speedups in aggregate["path_speedups"].items():
        lines.append(
            f"- `{configuration}`: pure vs eager "
            f"{_format_float(speedups['pure_vs_eager'])}×; pure vs JIT "
            f"{_format_float(speedups['pure_vs_jit'])}×"
        )
    if aggregate["parity_failures"]:
        lines.extend(["", "## Correctness failures", ""])
        for failure in aggregate["parity_failures"]:
            lines.append(
                f"- `{failure['configuration']}/{failure['path']}`: "
                f"max abs={failure['max_abs']:.6g}, max rel={failure['max_rel']:.6g}"
            )
    if aggregate["regressions"]:
        lines.extend(["", "## Performance regressions", ""])
        for regression in aggregate["regressions"]:
            lines.append(
                f"- `{regression['configuration']}/{regression['path']}`: "
                f"{regression['baseline_delta_pct']:.2f}% versus baseline"
            )
    path.write_text("\n".join(lines) + "\n")


def _run_cell(
    script: Path,
    run_dir: Path,
    config: Configuration,
    path: str,
    steps: int,
    warm_repeats: int,
) -> subprocess.CompletedProcess[str]:
    key = _cell_key(config, path)
    report_path = run_dir / "cells" / f"{key}.json"
    terminal_path = run_dir / "cells" / f"{key}.npz"
    command = [
        sys.executable,
        str(script),
        "--mode",
        "timing",
        "--path",
        path,
        "--steps",
        str(steps),
        "--frames",
        str(config.frames),
        "--residues",
        str(config.residues),
        "--timepoints",
        str(config.timepoints),
        "--warm-repeats",
        str(warm_repeats),
        "--output-dir",
        str(run_dir / "cells"),
        "--json",
        str(report_path),
        "--terminal-npz",
        str(terminal_path),
    ]
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configurations = (
        full_configurations() if args.suite == "full" else stage_configurations()
    )
    run_dir = args.results_root / args.run_id
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("profile_hdx_cpu.py").resolve()
    manifest_path = run_dir / "manifest.json"
    try:
        existing_manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        existing_manifest = {}

    cells = [(config, path) for config in configurations for path in args.paths]
    random.Random(args.order_seed).shuffle(cells)
    invocation_started_at = datetime.now(timezone.utc).isoformat()
    existing_cell_reports = list(cells_dir.glob("*.json"))
    benchmark_started_at = existing_manifest.get("benchmark_started_at")
    benchmark_completed_at = existing_manifest.get("benchmark_completed_at")
    if benchmark_started_at is None and existing_cell_reports:
        benchmark_started_at = datetime.fromtimestamp(
            min(path.stat().st_mtime for path in existing_cell_reports),
            tz=timezone.utc,
        ).isoformat()
    if (
        benchmark_completed_at is None
        and len(existing_cell_reports) == len(cells)
    ):
        benchmark_completed_at = datetime.fromtimestamp(
            max(path.stat().st_mtime for path in existing_cell_reports),
            tz=timezone.utc,
        ).isoformat()
    repository = _repository_metadata()
    manifest = {
        "run_id": args.run_id,
        "suite": args.suite,
        "steps": args.steps,
        "warm_repeats": args.warm_repeats,
        "paths": list(args.paths),
        "order_seed": args.order_seed,
        "configurations": [asdict(config) for config in configurations],
        "execution_order": [
            {"configuration": config.name, "path": path}
            for config, path in cells
        ],
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
        "previous_dir": str(args.previous_dir) if args.previous_dir else None,
        "benchmark_repository": existing_manifest.get(
            "benchmark_repository",
            existing_manifest.get("repository", repository),
        ),
        "aggregation_repository": repository,
        "host": _host_metadata(),
        "benchmark_started_at": benchmark_started_at or invocation_started_at,
        "benchmark_completed_at": benchmark_completed_at,
        "started_at": invocation_started_at,
        "completed_at": None,
        "last_aggregated_at": None,
        "status": "dry-run" if args.dry_run else "running",
        "cell_failures": [],
    }
    _atomic_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    failures: list[dict[str, Any]] = []
    for index, (config, path) in enumerate(cells, start=1):
        key = _cell_key(config, path)
        report_path = cells_dir / f"{key}.json"
        terminal_path = cells_dir / f"{key}.npz"
        if not args.force and _cell_matches(
            report_path,
            terminal_path,
            config,
            path,
            args.steps,
            args.warm_repeats,
        ):
            print(f"[{index}/{len(cells)}] reuse {key}", flush=True)
            continue
        print(
            f"[{index}/{len(cells)}] run {key} "
            f"(R={config.residues}, F={config.frames}, T={config.timepoints})",
            flush=True,
        )
        completed = _run_cell(
            script,
            run_dir,
            config,
            path,
            args.steps,
            args.warm_repeats,
        )
        if completed.returncode != 0:
            failure = {
                "configuration": config.name,
                "path": path,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
            }
            failures.append(failure)
            print(f"  FAILED return code {completed.returncode}", flush=True)
        else:
            print("  complete", flush=True)

    aggregate = _aggregate(
        run_dir,
        configurations,
        args.paths,
        args.baseline_dir,
        args.previous_dir,
    )
    aggregate["heatmaps"] = _write_heatmaps(
        run_dir,
        configurations,
        args.paths,
        aggregate,
    )
    _atomic_json(run_dir / "aggregate.json", aggregate)
    _atomic_csv(run_dir / "aggregate.csv", aggregate["rows"])
    _write_report(run_dir / "report.md", manifest, aggregate)

    completed_at = datetime.now(timezone.utc).isoformat()
    manifest["completed_at"] = completed_at
    manifest["last_aggregated_at"] = completed_at
    if manifest["benchmark_completed_at"] is None:
        manifest["benchmark_completed_at"] = completed_at
    manifest["cell_failures"] = failures
    manifest["status"] = (
        "complete" if not failures and aggregate["valid"] else "completed-with-failures"
    )
    _atomic_json(manifest_path, manifest)
    print(
        f"wrote {run_dir} ({len(aggregate['rows'])} rows, "
        f"{len(failures)} cell failures, "
        f"{len(aggregate['parity_failures'])} parity failures)",
        flush=True,
    )
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
