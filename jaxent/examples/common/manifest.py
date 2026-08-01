"""Immutable pre-launch manifests for the Example 1--3 campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import pandas as pd

ARTIFACT_VERSION = 1


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as file:
            file.write(text)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def write_processing_manifest(
    output_dir: str | Path, *, source_results_dir: str, run_entries: list[dict]
) -> None:
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "source_results_dir": source_results_dir,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs_processed": len(run_entries),
        "runs": run_entries,
    }
    atomic_write_text(
        Path(output_dir) / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True),
    )


def load_processing_manifest(
    output_dir: str | Path, *, allow_legacy_missing: bool = False
) -> dict:
    path = Path(output_dir) / "manifest.json"
    if not path.exists():
        if allow_legacy_missing:
            return {
                "artifact_version": 0,
                "source_results_dir": None,
                "generated_at": None,
                "n_runs_processed": None,
                "runs": [],
                "legacy_missing_manifest": True,
            }
        raise ValueError(
            f"No processing manifest at {path} — upstream processing did not complete"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"{path} has artifact_version={manifest.get('artifact_version')!r}, "
            f"expected {ARTIFACT_VERSION}"
        )
    if manifest.get("n_runs_processed", 0) == 0:
        raise ValueError(f"{path} reports zero processed runs")
    return manifest


def atomic_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(fd)
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


class ConvergenceLabelMismatchError(Exception):
    """Raised when processed arrays and convergence labels cannot be aligned."""


def _csv(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def runtime_provenance(repo_root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    lockfile = repo_root / "uv.lock"
    try:
        lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    except OSError:
        lock_hash = "unknown"
    return {
        "commit": commit,
        "lockfile_sha256": lock_hash,
        "jax_version": jax.__version__,
        "jaxlib_version": getattr(jax.lib, "__version__", "unknown"),
        "backend": jax.default_backend(),
        "device_count": jax.device_count(),
        "devices": [str(device) for device in jax.devices()],
        "python": sys.version,
    }


def write_prelaunch_manifest(
    output_dir: str | Path,
    *,
    example: int,
    ensembles: list[str],
    losses: list[str],
    split_types: list[str],
    maxent_values: list[float],
    bv_values: list[float] | None = None,
    bv_reg_functions: list[str] | None = None,
    learning_rate: float,
    lr_adjustment: str,
    frame_average_impl: str,
    step_chunk_size: int,
    n_steps: int,
    jobs: int,
    num_splits: int = 3,
) -> Path:
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to reuse existing output directory: {output_path}")
    if lr_adjustment not in {"on", "off"}:
        raise ValueError("lr_adjustment must be on or off")
    if frame_average_impl not in {"tensordot", "legacy_sum"}:
        raise ValueError("frame_average_impl is invalid")
    if step_chunk_size < 1:
        raise ValueError("step_chunk_size must be >= 1")
    bv_values = [None] if bv_values is None else bv_values
    bv_reg_functions = [None] if bv_reg_functions is None else bv_reg_functions
    bv_factor = 1 if bv_values == [None] else len(bv_values)
    reg_factor = 1 if bv_reg_functions == [None] else len(bv_reg_functions)
    expected = (
        len(ensembles)
        * len(losses)
        * len(split_types)
        * num_splits
        * len(maxent_values)
        * bv_factor
        * reg_factor
    )

    repo_root = Path(__file__).resolve().parents[3]
    manifest = {
        "manifest_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "example": example,
        "grid": {
            "ensembles": ensembles,
            "losses": losses,
            "split_types": split_types,
            "num_splits": num_splits,
            "maxent_values": maxent_values,
            "bv_values": [] if bv_values == [None] else bv_values,
            "bv_reg_functions": [] if bv_reg_functions == [None] else bv_reg_functions,
        },
        "factors": {
            "learning_rate": learning_rate,
            "lr_adjustment": lr_adjustment,
            "frame_average_impl": frame_average_impl,
            "step_chunk_size": step_chunk_size,
            "n_steps": n_steps,
            "execution_mode": "compiled",
            "reset_threshold_cooldown_on_oscillation": True,
            "parallel_jobs": jobs,
        },
        "expected_fit_count": expected,
        "resolved_inputs": {
            "launcher_working_directory": str(Path.cwd().resolve()),
            "features": str((Path.cwd() / "_featurise").resolve()),
            "datasplits": str((Path.cwd() / "_datasplits").resolve()),
            "output": str(output_path),
        },
        "runtime": runtime_provenance(repo_root),
        "output_paths": {
            "manifest": str(output_path / "prelaunch_manifest.json"),
            "logs": str(output_path / "logs"),
        },
    }
    output_path.mkdir(parents=True)
    manifest_path = output_path / "prelaunch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--example", type=int, required=True)
    parser.add_argument("--ensembles", required=True)
    parser.add_argument("--losses", required=True)
    parser.add_argument("--split-types", required=True)
    parser.add_argument("--maxent-values", required=True)
    parser.add_argument("--bv-values", default=None)
    parser.add_argument("--bv-reg-functions", default=None)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--lr-adjustment", choices=["on", "off"], required=True)
    parser.add_argument("--frame-average-impl", choices=["tensordot", "legacy_sum"], required=True)
    parser.add_argument("--step-chunk-size", type=int, required=True)
    parser.add_argument("--n-steps", type=int, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    args = parser.parse_args()
    write_prelaunch_manifest(
        args.output_dir,
        example=args.example,
        ensembles=_csv(args.ensembles),
        losses=_csv(args.losses),
        split_types=_csv(args.split_types),
        maxent_values=_csv(args.maxent_values, float),
        bv_values=None if args.bv_values is None else _csv(args.bv_values, float),
        bv_reg_functions=(
            None if args.bv_reg_functions is None else _csv(args.bv_reg_functions)
        ),
        learning_rate=args.learning_rate,
        lr_adjustment=args.lr_adjustment,
        frame_average_impl=args.frame_average_impl,
        step_chunk_size=args.step_chunk_size,
        n_steps=args.n_steps,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
