import json
import os

import pandas as pd
import pytest

from jaxent.examples.common.manifest import (
    ARTIFACT_VERSION,
    ConvergenceLabelMismatchError,
    atomic_to_csv,
    atomic_write_text,
    load_processing_manifest,
    write_processing_manifest,
)


def test_manifest_round_trip(tmp_path):
    write_processing_manifest(
        tmp_path,
        source_results_dir="/some/results",
        run_entries=[{"run_id": "a", "n_convergence_states": 6, "n_ladder_thresholds": 9}],
    )
    manifest = load_processing_manifest(tmp_path)
    assert manifest["artifact_version"] == ARTIFACT_VERSION
    assert manifest["n_runs_processed"] == 1
    assert manifest["runs"][0]["run_id"] == "a"
    assert manifest["source_results_dir"] == "/some/results"


def test_manifest_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        load_processing_manifest(tmp_path)


def test_manifest_version_mismatch_raises(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifact_version": ARTIFACT_VERSION + 999, "n_runs_processed": 1, "runs": []})
    )
    with pytest.raises(ValueError, match="artifact_version"):
        load_processing_manifest(tmp_path)


def test_manifest_zero_runs_raises(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifact_version": ARTIFACT_VERSION, "n_runs_processed": 0, "runs": []})
    )
    with pytest.raises(ValueError, match="zero"):
        load_processing_manifest(tmp_path)


def test_atomic_write_text_no_partial_file_on_crash(tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError):
        atomic_write_text(target, "{}")
    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_to_csv_writes_and_replaces(tmp_path):
    target = tmp_path / "scores.csv"
    atomic_to_csv(pd.DataFrame({"a": [1, 2, 3]}), target)
    assert target.exists()
    assert list(pd.read_csv(target)["a"]) == [1, 2, 3]


def test_atomic_to_csv_no_partial_file_on_crash(tmp_path, monkeypatch):
    target = tmp_path / "scores.csv"

    def boom(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError):
        atomic_to_csv(pd.DataFrame({"a": [1, 2, 3]}), target)
    assert not target.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_convergence_label_mismatch_error_is_exception():
    assert issubclass(ConvergenceLabelMismatchError, Exception)
    with pytest.raises(ConvergenceLabelMismatchError):
        raise ConvergenceLabelMismatchError("inconsistent stack lengths for run x")
