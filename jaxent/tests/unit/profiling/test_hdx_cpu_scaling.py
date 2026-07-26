from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from jaxent.src.custom_types.config import OptimiserSettings
from profiling import profile_hdx_cpu as profile
from profiling import run_hdx_cpu_scaling as scaling


def _settings(steps: int) -> OptimiserSettings:
    return OptimiserSettings(
        name="test_hdx_cpu_profile",
        n_steps=steps,
        tolerance=0.0,
        learning_rate=1e-4,
        optimiser_type="adam",
        convergence=0.0,
        min_steps_per_threshold=steps + 1,
    )


def test_scaling_suites_have_expected_unique_shapes() -> None:
    full = scaling.full_configurations()
    stage = scaling.stage_configurations()

    assert len(full) == 13
    assert len({(item.residues, item.frames, item.timepoints) for item in full}) == 13
    assert len(stage) == 5
    assert {item.name for item in stage} == {
        "anchor",
        "residues_600",
        "frames_5000",
        "timepoints_10",
        "corner_high",
    }


def test_scaling_default_uses_ignored_output_directory() -> None:
    args = scaling.build_parser().parse_args(
        ["--suite", "full", "--run-id", "test"]
    )

    assert args.results_root == Path("profiling/_output/hdx_cpu_scaling")


def test_uptake_fixture_uses_requested_timepoint_shape() -> None:
    jax.clear_caches()
    fixture = profile._make_fixture(
        frames=8,
        residues=6,
        timepoints=4,
        seed=0,
        target_frame=0,
    )

    assert fixture.simulation.outputs[0].y_pred().shape == (4, 6)
    assert fixture.data[0].shape == (4, 6)
    np.testing.assert_allclose(fixture.timepoint_values[0], 0.167, rtol=1e-6)
    np.testing.assert_allclose(fixture.timepoint_values[-1], 120.0, rtol=1e-6)


def test_each_prepared_path_reuses_warm_cache_and_reports_terminal_step() -> None:
    jax.clear_caches()
    steps = 2
    fixture = profile._make_fixture(
        frames=8,
        residues=6,
        timepoints=3,
        seed=0,
        target_frame=0,
    )

    for path in profile.PATHS:
        prepared = profile.PreparedPath(path, fixture, _settings(steps))
        cold, cold_snapshot = profile._measure_sample(prepared, steps)
        warm, warm_snapshot = profile._measure_sample(prepared, steps)

        assert cold.steps_completed == steps
        assert warm.steps_completed == steps
        assert warm.compiles == 0
        assert profile._snapshots_close(cold_snapshot, warm_snapshot)
        np.testing.assert_allclose(
            cold.final_loss,
            warm.final_loss,
            rtol=profile.PARITY_RTOL,
            atol=profile.PARITY_ATOL,
        )


def test_snapshot_comparison_and_regression_thresholds() -> None:
    reference = (np.asarray([1.0, 2.0]),)
    close, max_abs, _ = scaling._snapshot_difference(
        reference,
        (np.asarray([1.0, 2.0 + 1e-6]),),
    )
    assert close
    assert max_abs > 0.0

    assert not scaling._regression_flag(1.05, 0.01, 1.0, 0.01)
    assert scaling._regression_flag(1.20, 0.01, 1.0, 0.01)


def test_scaling_fit_constrains_compute_terms_nonnegative() -> None:
    rows = [
        {
            "path": "pure",
            "cell_valid": True,
            "configuration": f"case_{index}",
            "residues": residues,
            "frames": frames,
            "timepoints": timepoints,
            "warm_median_s": elapsed,
        }
        for index, (residues, frames, timepoints, elapsed) in enumerate(
            [
                (100, 100, 1, 0.2),
                (100, 500, 1, 0.5),
                (500, 500, 1, 1.5),
                (500, 500, 10, 1.6),
            ]
        )
    ]

    model = scaling._fit_scaling(rows)["pure"]

    assert model["seconds_per_residue_frame"] >= 0.0
    assert model["seconds_per_residue_timepoint"] >= 0.0


def test_timing_arguments_require_a_single_named_path() -> None:
    args = profile.build_parser().parse_args(
        ["--mode", "timing", "--path", "pure", "--timepoints", "5"]
    )
    profile._validate_args(argparse.ArgumentParser(), args)

    assert args.path == "pure"
    assert args.steps == 1000


def test_heatmaps_are_written_and_embedded_in_report(tmp_path: Path) -> None:
    configurations = (
        scaling.Configuration("small", 96, 173, 1),
        scaling.Configuration("large", 600, 5000, 10),
    )
    rows = [
        {
            "configuration": config.name,
            "residues": config.residues,
            "frames": config.frames,
            "timepoints": config.timepoints,
            "path": path,
            "cell_valid": True,
            "warm_median_s": elapsed,
            "warm_mad_s": 0.01,
            "warm_steps_per_s": 1000.0 / elapsed,
            "cold_s": elapsed + 0.5,
            "warm_host_materialisations_per_step": 0.0,
            "cross_path_parity": True,
        }
        for config, elapsed_by_path in zip(
            configurations,
            ((10.0, 1.0, 0.25), (12.0, 3.0, 6.0)),
        )
        for path, elapsed in zip(scaling.PATHS, elapsed_by_path)
    ]
    aggregate = {
        "rows": rows,
        "path_speedups": scaling._path_speedups(rows),
        "scaling_model": {},
        "axis_elasticities": {},
        "parity_failures": [],
        "regressions": [],
        "valid": True,
    }

    heatmaps = scaling._write_heatmaps(
        tmp_path,
        configurations,
        scaling.PATHS,
        aggregate,
    )
    aggregate["heatmaps"] = heatmaps

    assert set(heatmaps) == {"warm_runtime", "pure_speedup"}
    for relative_path in heatmaps.values():
        image_path = tmp_path / relative_path
        assert image_path.stat().st_size > 0
        assert image_path.read_bytes().startswith(b"\x89PNG")

    report_path = tmp_path / "report.md"
    scaling._write_report(
        report_path,
        {
            "run_id": "test",
            "suite": "full",
            "steps": 1000,
            "warm_repeats": 3,
        },
        aggregate,
    )
    report = report_path.read_text()
    assert "![Warm runtime heatmap](heatmaps/warm_runtime_heatmap.png)" in report
    assert "![Pure path speedup heatmap](heatmaps/pure_speedup_heatmap.png)" in report
