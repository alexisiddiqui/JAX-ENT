from __future__ import annotations

import argparse
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
    / "run_sigma_source_sidecar.py"
)
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("sigma_source_sidecar", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
sidecar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sidecar
SPEC.loader.exec_module(sidecar)


def test_population_weights_define_gt_and_closed_only_populations():
    assignments = np.asarray([-1, -1, 0, 0, 1, 1, 1])

    gt = sidecar.population_weights(assignments, "gt")
    closed = sidecar.population_weights(assignments, "closed")

    assert np.isclose(gt[assignments == -1].sum(), 0.0)
    assert np.isclose(gt[assignments == 0].sum(), 0.4)
    assert np.isclose(gt[assignments == 1].sum(), 0.6)
    assert np.isclose(closed[assignments != 1].sum(), 0.0)
    assert np.isclose(closed[assignments == 1].sum(), 1.0)
    np.testing.assert_allclose(closed[assignments == 1], np.full(3, 1 / 3))


def test_coordinate_covariance_is_translation_invariant_and_psd():
    coordinates = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
        ]
    )
    weights = np.asarray([0.5, 0.5])

    covariance = sidecar.weighted_coordinate_covariance(coordinates, weights)
    translated = sidecar.weighted_coordinate_covariance(
        coordinates + np.asarray([11.0, -4.0, 7.0]), weights
    )

    np.testing.assert_allclose(covariance, np.diag([0.75, 3.0]))
    np.testing.assert_allclose(translated, covariance)
    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= 0


def test_regularization_endpoints_and_minimum_stable_ridge():
    raw = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    weights = np.asarray([0.5, 0.5])

    zero_arrays, zero_metrics = sidecar.regularize_sigma(raw, weights, 0.0, 1e8)
    one_arrays, one_metrics = sidecar.regularize_sigma(raw, weights, 1.0, 1e8)

    np.testing.assert_allclose(zero_arrays["Sigma_shrunk"], raw)
    np.testing.assert_allclose(one_arrays["Sigma_shrunk"], np.eye(2))
    assert zero_metrics["ridge"] > 0
    assert one_metrics["ridge"] == 0
    assert zero_metrics["condition_number"] <= 1e8 * (1 + 1e-8)
    np.testing.assert_allclose(
        zero_arrays["Sigma"] @ zero_arrays["Sigma_inv"], np.eye(2), atol=1e-7
    )


def test_default_grid_contains_216_fits(tmp_path):
    args = argparse.Namespace(
        ensembles=sidecar.shrinkage.DEFAULT_ENSEMBLES,
        sources=sidecar.DEFAULT_SOURCES,
        alphas=sidecar.shrinkage.DEFAULT_ALPHAS,
        split_type="sequence_cluster",
        n_splits=3,
        output_dir=tmp_path,
        features_dir=tmp_path,
        datasplit_dir=tmp_path,
        clustering_dir=tmp_path,
        n_steps=5000,
        learning_rate=1.0,
        ema_alpha=0.5,
        forward_model_scaling=1000.0,
        execution_mode="compiled",
    )
    sigma_paths = {
        (ensemble, source, float(alpha)): tmp_path / "sigma.npz"
        for ensemble in args.ensembles
        for source in args.sources
        for alpha in args.alphas
    }

    specs = sidecar.build_specs(args, sigma_paths)

    assert len(specs) == 216
    assert len({spec.run_id for spec in specs}) == 216
    assert {spec.sigma_source for spec in specs} == set(sidecar.DEFAULT_SOURCES)


def test_selection_uses_closed_sigma_and_first_native_state_on_tie():
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


def test_source_plot_writes_png_and_svg(tmp_path):
    rows = []
    for ensemble in sidecar.shrinkage.DEFAULT_ENSEMBLES:
        for source in sidecar.DEFAULT_SOURCES:
            for alpha, value in ((0.0, 20.0), (0.5, 40.0), (1.0, 60.0)):
                rows.append(
                    {
                        "ensemble": ensemble,
                        "sigma_source": source,
                        "split_type": "sequence_cluster",
                        "split_idx": 0,
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
