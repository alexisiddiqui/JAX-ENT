"""Versioned processing manifests and atomic CSV output."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ARTIFACT_VERSION = 1


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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
    atomic_write_text(Path(output_dir) / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def load_processing_manifest(
    output_dir: str | Path,
    *,
    allow_legacy_missing: bool = False,
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
        raise ValueError(f"No processing manifest at {path} — upstream processing did not complete")
    manifest = json.loads(path.read_text())
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"{path} has artifact_version={manifest.get('artifact_version')!r}, expected {ARTIFACT_VERSION}"
        )
    if manifest.get("n_runs_processed", 0) == 0:
        raise ValueError(f"{path} reports zero processed runs")
    return manifest


def atomic_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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
