"""Selection and blinding boundaries for target-variance runners."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
ISO_PATH = ROOT / "examples/1_IsoValidation_OMass/fitting/jaxENT/investigate_iso_target_variance.py"
MOPRP_DIR = ROOT / "examples/2_CrossValidation/fitting/jaxENT"


def _load(name: str, path: Path, extra_path: Path | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    if extra_path is not None:
        sys.path.insert(0, str(extra_path))
    try:
        spec.loader.exec_module(module)
    finally:
        if extra_path is not None:
            sys.path.remove(str(extra_path))
    return module


ISO = _load("iso_target_variance_runner", ISO_PATH)
MOPRP = _load(
    "moprp_target_variance_runner",
    MOPRP_DIR / "validate_moprp_target_variance.py",
    MOPRP_DIR,
)
MOPRP_SWEEP = _load(
    "moprp_target_variance_sweep_runner",
    MOPRP_DIR / "investigate_moprp_target_variance_sweep.py",
    MOPRP_DIR,
)


def test_iso_selection_uses_only_heldout_hdx_columns():
    rows = []
    for estimator in ("curve_moment", "structured_residual"):
        for geometry, score, truth_score in (
            ("identity", 2.0, 0.99),
            ("covariance_only", 1.0, -0.99),
        ):
            rows.append(
                {
                    "ensemble": "ISO_BI",
                    "panel": "equal",
                    "split_index": 0,
                    "time_fold": 0,
                    "estimator": estimator,
                    "geometry": geometry,
                    "regularization": 0.1,
                    "heldout_reconstruction_score": score,
                    "heldout_mean_mse_ratio": 1.0,
                    # Deliberately contradictory truth column: selector must ignore it.
                    "log_variance_spearman": truth_score,
                }
            )
    selected = ISO.select_by_hdx_reconstruction(pd.DataFrame(rows))
    assert selected["geometry"] == "covariance_only"


def test_iso_selection_excludes_nonconverged_candidates():
    frame = pd.DataFrame(
        [
            {
                "ensemble": "ISO_BI",
                "panel": "equal",
                "split_index": 0,
                "time_fold": 0,
                "estimator": "curve_moment",
                "geometry": geometry,
                "regularization": 0.1,
                "heldout_reconstruction_score": score,
                "heldout_mean_mse_ratio": score,
                "success": success,
                "finite_objective": True,
                "psd": True,
            }
            for geometry, score, success in (
                ("distance_only", 0.01, False),
                ("identity", 0.1, True),
                ("covariance_only", 0.2, True),
            )
        ]
    )
    assert ISO.select_by_hdx_reconstruction(frame)["geometry"] == "covariance_only"


def test_no_fitting_litmus_exercises_all_geometries_without_optimization(tmp_path):
    source = {
        "log_pf": np.asarray(
            [[0.1, 0.2, 0.4, 0.5], [0.3, 0.4, 0.8, 0.9], [0.2, 0.5, 0.3, 0.7]]
        ),
        "k_ints": np.asarray([0.2, 0.5, 1.0]),
        "assignments": np.asarray([0, 0, 1, 1]),
        "residue_ids": np.asarray([10, 11, 20]),
        "coordinates": np.asarray([[0, 0, 0], [3, 0, 0], [20, 0, 0]], float),
    }
    mappings = {
        "equal": np.eye(3),
        "random_fixed": np.asarray([[0.5, 0.5, 0], [0, 0.5, 0.5]]),
        "random_variable": np.asarray([[1 / 3, 1 / 3, 1 / 3]]),
    }
    decision = ISO.run_numerical_litmus(
        {"ISO_BI": source, "ISO_TRI": source}, mappings, tmp_path
    )
    assert decision["passed"]
    assert decision["n_checks"] == 2 * 3 * 6
    assert decision["optimization_performed"] is False
    assert decision["variance_fitting_performed"] is False


def test_development_artifact_is_non_promotable_and_pilot_cannot_qualify(tmp_path):
    selection = {
        "estimator": "curve_moment",
        "geometry": "distance_only",
        "regularization": 0.1,
        "selection_criterion": "held-out HDX only",
        "shuffled_control_beats_selected": False,
        "identity_control_beats_selected": False,
        "ranking": [],
    }
    development = tmp_path / "development.json"
    ISO.write_development_selection(development, selection, pilot=False)
    payload = ISO.load_development_selection(development)
    assert payload["qualified"] is False
    assert payload["can_launch_moprp_validation"] is False
    assert payload["can_launch_qualification"] is True
    assert payload["population_recovery_used_for_selection"] is False

    pilot = tmp_path / "pilot.json"
    ISO.write_development_selection(pilot, selection, pilot=True)
    with pytest.raises(ValueError, match="pilot settings cannot launch"):
        ISO.load_development_selection(pilot)

    failed = tmp_path / "failed.json"
    failed_selection = {**selection, "shuffled_control_beats_selected": True}
    ISO.write_development_selection(failed, failed_selection, pilot=False)
    with pytest.raises(ValueError, match="controls were not beaten"):
        ISO.load_development_selection(failed)


def test_development_and_qualification_splits_are_disjoint():
    assert set(ISO.DEVELOPMENT_SPLITS).isdisjoint(ISO.QUALIFICATION_SPLITS)


def test_moprp_inference_interface_is_nmr_blind():
    forbidden = {"weights", "nmr_weights", "reference_weights", "states", "targets"}
    assert forbidden.isdisjoint(inspect.signature(MOPRP.infer_blinded).parameters)


def test_moprp_sweep_resolves_exact_frozen_bv_settings_with_provenance():
    scaled = MOPRP_SWEEP.load_bv_setting(
        MOPRP_SWEEP.DEFAULT_COEFFICIENT_LOCK, "scaled_published"
    )
    optimum = MOPRP_SWEEP.load_bv_setting(
        MOPRP_SWEEP.DEFAULT_COEFFICIENT_LOCK, "constrained_optimum"
    )
    assert scaled["bc"] == pytest.approx(0.18624016700883145)
    assert scaled["bh"] == pytest.approx(1.0642295257647512)
    assert optimum["bc"] == pytest.approx(0.2288930418240737)
    assert optimum["bh"] == 0.0
    assert len(scaled["coefficient_lock_sha256"]) == 64
    assert "NMR populations" in scaled["provenance"]


def test_moprp_sweep_rejects_unknown_or_invalid_bv_settings(tmp_path):
    lock = tmp_path / "coefficient_lock.json"
    lock.write_text(
        json.dumps(
            {
                "frozen_settings": {
                    "valid": {"bc": 0.2, "bh": 1.0},
                    "negative": {"bc": -0.1, "bh": 1.0},
                }
            }
        )
    )
    with pytest.raises(ValueError, match="unknown BV setting"):
        MOPRP_SWEEP.load_bv_setting(lock, "missing")
    with pytest.raises(ValueError, match="finite and non-negative"):
        MOPRP_SWEEP.load_bv_setting(lock, "negative")


def test_moprp_diagnostic_sweep_selection_is_shared_and_hdx_only():
    rows = []
    for ensemble in ("AF2_MSAss", "AF2_filtered"):
        for geometry, score, truth_score in (
            ("identity", 0.5, 0.99),
            ("shuffled_geometry", 0.7, 0.95),
            ("distance_only", 0.2, -0.99),
            ("covariance_only", 0.4, 0.80),
        ):
            rows.append(
                {
                    "ensemble": ensemble,
                    "peptide_fold": 0,
                    "time_fold": 0,
                    "estimator": "curve_moment",
                    "geometry": geometry,
                    "regularization": 0.1,
                    "heldout_reconstruction_score": score,
                    "heldout_mean_mse_ratio": 1.0,
                    "success": True,
                    "finite_objective": True,
                    "psd": True,
                    # Contradictory pseudo-truth cannot influence the selector.
                    "log_variance_spearman": truth_score,
                }
            )
    selection = MOPRP_SWEEP.select_by_hdx_only(pd.DataFrame(rows))
    assert selection["geometry"] == "distance_only"
    assert selection["shared_across_ensembles"] is True
    assert selection["identity_control_beats_selected"] is False
    assert selection["shuffled_control_beats_selected"] is False


def test_moprp_diagnostic_selector_rejects_missing_cells_and_nan_scores():
    rows = [
        {
            "ensemble": ensemble,
            "peptide_fold": 0,
            "time_fold": 0,
            "estimator": "curve_moment",
            "geometry": geometry,
            "regularization": 0.1,
            "heldout_reconstruction_score": score,
            "heldout_mean_mse_ratio": 1.0,
            "success": True,
            "finite_objective": True,
            "psd": True,
        }
        for ensemble in ("AF2_MSAss", "AF2_filtered")
        for geometry, score in (("distance_only", 0.2), ("identity", 0.4))
    ]
    incomplete = pd.DataFrame(rows[:-1])
    with pytest.raises(ValueError, match="cover every ensemble/fold cell"):
        MOPRP_SWEEP.select_by_hdx_only(incomplete)
    with_nan = pd.DataFrame(rows)
    with_nan.loc[0, "heldout_reconstruction_score"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        MOPRP_SWEEP.select_by_hdx_only(with_nan)


def test_moprp_fold_schemes_cover_each_observation_once():
    for scheme in ("interleaved", "contiguous"):
        folds = MOPRP_SWEEP._folds(15, 5, scheme)
        np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(15))
        assert all(len(fold) == 3 for fold in folds)


def test_constant_variance_control_is_fit_independently_from_hdx():
    means = np.asarray([0.2, 0.6, 1.0])
    times = np.asarray([0.2, 1.0, 5.0])
    mapping = np.eye(3)
    observed = MOPRP_SWEEP.predict_curve_moment_uptake(
        means, np.full(3, 0.04), times, mapping
    )
    result = MOPRP_SWEEP._fit_constant_variance(
        "curve_moment",
        observed,
        means,
        times,
        mapping,
        np.eye(3),
        np.ones_like(observed, dtype=bool),
    )
    assert result["success"]
    np.testing.assert_allclose(np.diag(result["covariance"]), 0.04, rtol=2e-4)


def test_diagnostic_sweep_artifact_cannot_launch_formal_moprp(tmp_path):
    artifact = tmp_path / "diagnostic.json"
    artifact.write_text(
        json.dumps(
            {
                "artifact_type": MOPRP_SWEEP.ARTIFACT_TYPE,
                "qualified": False,
                "can_launch_moprp_validation": False,
            }
        )
    )
    with pytest.raises(ValueError, match="target-variance artifact"):
        MOPRP.load_frozen_settings(artifact, require_qualified=True)


def test_real_moprp_partition_holds_peptide1_and_residue101_peptide_out():
    inputs = MOPRP.common.load_blinded_ensemble_inputs("AF2_MSAss")
    assert not hasattr(inputs, "reference_weights")
    partition = MOPRP.peptide_partitions(inputs)
    assert inputs.peptide_ids[partition["peptide1_row"]] == 1
    assert inputs.peptide_ids[partition["unmapped_101_row"]] == 12
    assert partition["peptide1_row"] not in partition["fit_rows"]
    assert partition["unmapped_101_row"] not in partition["fit_rows"]
