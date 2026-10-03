from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "1_IsoValidation_OMass"
    / "fitting"
    / "jaxENT"
    / "run_sigma_shrinkage_sidecar.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("sigma_shrinkage_sidecar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sidecar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sidecar
SPEC.loader.exec_module(sidecar)


def test_oracle_sigma_identity_shrinkage_and_minimum_ridge():
    coordinates = np.asarray(
        [
            [0.0, 0.2, 0.8, 1.0],
            [0.0, 0.4, 0.7, 1.0],
            [0.1, 0.3, 0.9, 0.95],
        ]
    )
    assignments = np.asarray([0, 0, 1, 1])

    zero_arrays, zero_metrics = sidecar.construct_oracle_sigma(
        coordinates, assignments, alpha=0.0, condition_limit=1e8
    )
    one_arrays, one_metrics = sidecar.construct_oracle_sigma(
        coordinates, assignments, alpha=1.0, condition_limit=1e8
    )

    np.testing.assert_allclose(zero_arrays["Sigma_shrunk"], zero_arrays["Sigma_raw"])
    expected_identity = (
        np.trace(one_arrays["Sigma_raw"]) / len(one_arrays["Sigma_raw"])
    ) * np.eye(len(one_arrays["Sigma_raw"]))
    np.testing.assert_allclose(one_arrays["Sigma_shrunk"], expected_identity)
    assert zero_metrics["ridge"] >= 0
    assert one_metrics["ridge"] == 0
    assert zero_metrics["condition_number"] <= 1e8 * (1 + 1e-8)
    assert one_metrics["condition_number"] == 1
    np.testing.assert_allclose(
        zero_arrays["Sigma"] @ zero_arrays["Sigma_inv"],
        np.eye(len(zero_arrays["Sigma"])),
        atol=1e-7,
    )


def test_oracle_weights_exclude_intermediates():
    assignments = np.asarray([0, 0, 1, 1, 1, -1, -1])
    weights = sidecar.compute_cluster_weights(assignments, {"open": 0.4, "closed": 0.6})
    assert np.isclose(weights[assignments == 0].sum(), 0.4)
    assert np.isclose(weights[assignments == 1].sum(), 0.6)
    assert np.isclose(weights[assignments == -1].sum(), 0.0)


def test_select_best_rows_uses_closed_sigma_and_first_native_state_on_tie():
    frame = pd.DataFrame(
        {
            "run_id": ["a", "a", "a", "b"],
            "convergence_rank": [0, 1, 2, 0],
            "native_sigma_val_loss": [0.01, 9.0, 0.001, 1.0],
            "val_mse": [0.001, 0.2, 0.3, 0.1],
            "val_closed_sigma_mse": [0.4, 0.2, 0.2, np.nan],
        }
    )
    selected = sidecar.select_best_rows(frame)
    assert selected["run_id"].tolist() == ["a"]
    assert selected.iloc[0]["convergence_rank"] == 1


def test_recovery_penalizes_intermediate_mass_and_ess_is_percent():
    assignments = np.asarray([0, 1, -1])
    truth = np.asarray([0.4, 0.6, 0.0])
    contaminated = np.asarray([0.3, 0.45, 0.25])
    perfect_recovery = sidecar.calculate_recovery_percentage(
        assignments, truth, sidecar.GROUND_TRUTH, sidecar.STATE_MAPPING
    )
    contaminated_recovery = sidecar.calculate_recovery_percentage(
        assignments, contaminated, sidecar.GROUND_TRUTH, sidecar.STATE_MAPPING
    )
    assert np.isclose(perfect_recovery, 100.0)
    assert contaminated_recovery < perfect_recovery
    uniform = np.ones(10) / 10
    assert np.isclose(100 * sidecar.effective_sample_size(uniform) / len(uniform), 100)


def test_paired_uptake_statistics_use_uptake_as_paired_baseline():
    baseline = np.tile(np.asarray([0.1, 0.2, 0.3]), (len(sidecar.TIMEPOINTS), 1))
    offsets = np.asarray([-0.1, 0.0, 0.2])
    candidate = baseline + offsets[None, :]

    rows = sidecar.paired_mode_statistics(candidate, baseline)

    assert len(rows) == len(sidecar.TIMEPOINTS)
    assert all(row["n_observations"] == 3 for row in rows)
    assert all(np.isclose(row["mean_difference"], offsets.mean()) for row in rows)
    expected_dz = offsets.mean() / offsets.std(ddof=1)
    assert all(np.isclose(row["cohen_dz"], expected_dz) for row in rows)
    self_rows = sidecar.paired_mode_statistics(baseline, baseline)
    assert all(row["p_value"] == 1.0 for row in self_rows)
    assert all(row["cohen_dz"] == 0.0 for row in self_rows)


def test_plot_metric_writes_both_formats_and_accepts_spatial_rows(tmp_path):
    rows = []
    for ensemble in sidecar.DEFAULT_ENSEMBLES:
        for mode in sidecar.DEFAULT_MODES:
            for split_type, split_idx in (("sequence_cluster", 0), ("spatial", 0)):
                for alpha, value in ((0.0, 20.0), (0.5, 40.0), (1.0, 60.0)):
                    rows.append(
                        {
                            "ensemble": ensemble,
                            "mode": mode,
                            "split_type": split_type,
                            "split_idx": split_idx,
                            "alpha": alpha,
                            "recovery_percent": value,
                        }
                    )
    stem = tmp_path / "recovery"
    sidecar.plot_metric(
        pd.DataFrame(rows),
        "recovery_percent",
        "Recovery (%)",
        stem,
        (0.0, 0.5, 1.0),
    )
    assert stem.with_suffix(".png").is_file()
    assert stem.with_suffix(".svg").is_file()


def test_forward_uptake_diagnostic_plots_write_both_formats(tmp_path):
    curve_rows = []
    comparison_rows = []
    for ensemble in sidecar.DEFAULT_ENSEMBLES:
        for weighting in sidecar.WEIGHT_LABELS:
            for mode_index, mode in enumerate(sidecar.DEFAULT_MODES):
                for time_index, timepoint in enumerate(sidecar.TIMEPOINTS):
                    curve_rows.append(
                        {
                            "ensemble": ensemble,
                            "weighting": weighting,
                            "mode": mode,
                            "timepoint": timepoint,
                            "mean_uptake": 0.1 * (time_index + 1),
                            "sd_uptake": 0.01 * (mode_index + 1),
                        }
                    )
                    comparison_rows.append(
                        {
                            "ensemble": ensemble,
                            "weighting": weighting,
                            "mode": mode,
                            "timepoint": timepoint,
                            "p_value": 1.0 if mode == "uptake" else 0.01,
                            "cohen_dz": 0.0 if mode == "uptake" else mode_index + 0.5,
                        }
                    )
    curve_stem = tmp_path / "curves"
    heatmap_stem = tmp_path / "heatmap"

    sidecar.plot_forward_uptake_curves(pd.DataFrame(curve_rows), curve_stem)
    sidecar.plot_forward_uptake_statistics(
        pd.DataFrame(comparison_rows), "ISO_BI", heatmap_stem
    )

    assert curve_stem.with_suffix(".png").is_file()
    assert curve_stem.with_suffix(".svg").is_file()
    assert heatmap_stem.with_suffix(".png").is_file()
    assert heatmap_stem.with_suffix(".svg").is_file()


def test_configure_model_modes_are_native_implementations():
    assignments = np.asarray([0, 1, -1])
    rate = sidecar.configure_model("rate", assignments)
    uptake = sidecar.configure_model("uptake", assignments)
    linear = sidecar.configure_model("linear", assignments)
    rate_forward = rate.forward[sidecar.m_key("HDX_peptide")]
    uptake_forward = uptake.forward[sidecar.m_key("HDX_peptide")]
    linear_forward = linear.forward[sidecar.m_key("HDX_peptide")]
    assert rate_forward.frame_averaging_mode == "rate"
    assert uptake_forward.frame_averaging_mode == "frame_uptake"
    assert uptake_forward.frame_group_masks is None
    assert linear_forward.frame_averaging_mode == "linear_uptake"
    assert linear.params.kint_unit == "min^-1"
    assert linear.params.time_unit == "min"


def test_defaults_use_repository_native_intrinsic_rates_and_matching_target():
    assert sidecar.DEFAULT_FEATURES_DIR.name == "fit_features"
    assert sidecar.DEFAULT_FEATURES_DIR.parent.name == "_self_consistent_iso"
    assert sidecar.DEFAULT_DATASPLIT_DIR.name == "datasplits"
    manifest = sidecar.json.loads(sidecar.DEFAULT_RATE_SOURCE_MANIFEST.read_text())
    assert manifest["kint_unit"] == "min^-1"
    features, topology = sidecar.load_features(sidecar.DEFAULT_FEATURES_DIR, "ISO_BI")
    assert features.features_shape[0] == len(topology) == 294
