"""Tests for the Stage-6 frozen-target joint-BV objective."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"


@pytest.fixture(scope="module")
def joint_diag_d():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("moprp_joint_diag_d_fit")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


@pytest.fixture(scope="module")
def diagnostics():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("joint_diag_d_diagnostics")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


def _synthetic_cells(module):
    cells = {}
    for index, name in enumerate(("A", "B")):
        frames = 7
        heavy = jnp.asarray(
            np.array(
                [
                    [0.1, 0.2, 0.4, 0.3, 0.2, 0.5, 0.7],
                    [0.3, 0.1, 0.2, 0.4, 0.5, 0.3, 0.2],
                    [0.5, 0.4, 0.3, 0.2, 0.1, 0.2, 0.4],
                    [0.2, 0.3, 0.1, 0.5, 0.4, 0.6, 0.3],
                ]
            )
            + index * 0.01
        )
        acceptor = jnp.flip(heavy, axis=0)
        cells[name] = {
            "ensemble": name,
            "heavy": heavy,
            "acceptor": acceptor,
            "mapping": jnp.asarray(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]]
            ),
            "observed": jnp.asarray(
                [[0.05, 0.2, 0.4], [0.08, 0.25, 0.5]]
            ),
            "k_ints": jnp.asarray([0.2, 0.3, 0.4, 0.5]),
            "timepoints": jnp.asarray([0.5, 2.0, 8.0]),
            "target": jnp.asarray([0.01, 0.02, 0.03, 0.04]),
            "mean_ref": 0.1,
            "n_frames": frames,
        }
    return cells


@pytest.mark.parametrize("arm", ["diag_d_absolute", "diag_d_scalefree"])
def test_joint_loss_is_finite_and_differentiable_in_theta_and_logits(joint_diag_d, arm):
    cells = _synthetic_cells(joint_diag_d)
    params = {
        "theta": jnp.asarray([0.1, -0.2]),
        "logits": {name: jnp.zeros(cell["n_frames"]) for name, cell in cells.items()},
    }
    loss = joint_diag_d._loss_fn(cells, gamma=0.3, eta=0.01, arm=arm)
    value, gradients = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradients["theta"])).all()
    for gradient in gradients["logits"].values():
        assert np.isfinite(np.asarray(gradient)).all()


def test_gamma_zero_arms_are_identical_for_the_same_parameters(joint_diag_d):
    cells = _synthetic_cells(joint_diag_d)
    params = {
        "theta": jnp.asarray([0.1, -0.2]),
        "logits": {name: jnp.zeros(cell["n_frames"]) for name, cell in cells.items()},
    }
    absolute = joint_diag_d._loss_fn(cells, gamma=0.0, eta=0.1, arm="diag_d_absolute")
    scalefree = joint_diag_d._loss_fn(cells, gamma=0.0, eta=0.1, arm="diag_d_scalefree")
    assert float(absolute(params)) == pytest.approx(float(scalefree(params)), abs=1e-12)


def test_fixed_ess_grid_and_loss_are_finite(joint_diag_d):
    assert joint_diag_d.ESS_TARGETS == (1.0, 5.0, 15.0, 30.0, 60.0)
    assert joint_diag_d.FIXED_ESS_GAMMAS == joint_diag_d.GAMMAS
    cells = _synthetic_cells(joint_diag_d)
    params = {
        "theta": jnp.asarray([0.1, -0.2]),
        "logits": {
            name: jnp.linspace(-1.0, 1.0, cell["n_frames"])
            for name, cell in cells.items()
        },
    }
    loss = joint_diag_d._loss_fn(
        cells,
        gamma=0.1,
        eta=3.0,
        arm="diag_d_scalefree",
        diversity_control="fixed_ess",
    )
    value, gradients = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(value))
    assert np.isfinite(np.asarray(gradients["theta"])).all()
    assert all(
        np.isfinite(np.asarray(gradient)).all()
        for gradient in gradients["logits"].values()
    )


def test_fixed_ess_one_anchor_survives_positive_gamma_optimization(joint_diag_d):
    fit = joint_diag_d._run_with_diagnostics(
        _synthetic_cells(joint_diag_d),
        gamma=0.1,
        eta=1.0,
        arm="diag_d_scalefree",
        steps=50,
        lr=joint_diag_d.LR,
        n_start=2,
        diversity_control="fixed_ess",
    )
    assert np.isfinite(fit["diagnostics"]["A"]["best_objective"])
    assert np.isfinite(fit["diagnostics"]["B"]["best_objective"])
    for name in ("A", "B"):
        weights = joint_diag_d.fixed_ess_weights(fit["params"]["logits"][name], 1.0)
        assert np.isfinite(np.asarray(weights)).all()


def test_fixed_ess_zero_variance_profile_loss_is_finite(joint_diag_d):
    predicted = jnp.asarray([0.0, 1e-14, 2e-5])
    target = jnp.asarray([1e-4, 2e-4, 3e-4])
    for arm in joint_diag_d.ARMS:
        loss = joint_diag_d._diag_d_loss(
            predicted, target, arm, finite_floor=True
        )
        assert np.isfinite(float(loss))


@pytest.mark.parametrize("target", [1.01, 2.0, 3.0, 5.0, 7.5])
def test_fixed_ess_weights_hit_random_targets(diagnostics, target):
    logits = jax.random.normal(jax.random.PRNGKey(17), (11,))
    weights = diagnostics.fixed_ess_weights(logits, target)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(diagnostics.effective_sample_size(weights)) == pytest.approx(
        target, abs=1e-3
    )


def test_fixed_ess_one_anchor_uses_finite_open_boundary(diagnostics):
    logits = jnp.asarray([-0.2, 0.4, 1.2, 0.1])
    weights = diagnostics.fixed_ess_weights(logits, 1.0)
    assert np.isfinite(np.asarray(weights)).all()
    assert np.count_nonzero(np.asarray(weights)) >= 2
    assert float(diagnostics.effective_sample_size(weights)) == pytest.approx(
        1.0, abs=1e-3
    )


def test_uniform_logits_retain_unreachable_uniform_ess(diagnostics):
    weights = diagnostics.fixed_ess_weights(jnp.zeros(9), 3.0)
    np.testing.assert_allclose(np.asarray(weights), np.full(9, 1.0 / 9.0), atol=1e-7)
    assert float(diagnostics.effective_sample_size(weights)) == pytest.approx(9.0)


def test_ess_is_monotone_in_temperature(diagnostics):
    logits = jnp.asarray([-2.0, -0.3, 0.2, 0.9, 2.1])
    temperatures = jnp.asarray([0.05, 0.2, 1.0, 5.0, 20.0])
    realized = np.asarray(
        [
            diagnostics.effective_sample_size(jax.nn.softmax(logits / temperature))
            for temperature in temperatures
        ]
    )
    assert np.all(np.diff(realized) >= 0.0)


def test_frozen_scaled_published_target_loader_anchor(joint_diag_d):
    common = importlib.import_module("_moprp_recovery_common")
    artifact = joint_diag_d.TARGET_ARTIFACT
    inputs = common.load_blinded_ensemble_inputs("AF2_MSAss")
    target, info = common.load_selected_diag_d_target(
        artifact, "AF2_MSAss", inputs.feature_residue_ids
    )
    assert target.shape == inputs.feature_residue_ids.shape
    assert target.shape == (97,)
    assert info["candidate_id"]
    assert info["geometry"] == "covariance_only"
    assert np.isfinite(target).all() and np.all(target > 0)


def test_refined_grid_and_split_holdouts_are_disjoint(joint_diag_d):
    assert joint_diag_d.GAMMAS == (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    assert joint_diag_d.ETAS == (0.0, 0.01, 0.022, 0.046, 0.1)
    specs = joint_diag_d._split_specs(13, 15)
    assert len(specs) == 3
    peptide_sets = [set(spec["val_peptides"]) for spec in specs]
    time_sets = [set(spec["val_times"]) for spec in specs]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(peptide_sets)
        for right in peptide_sets[index + 1 :]
    )
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(time_sets)
        for right in time_sets[index + 1 :]
    )
    for spec in specs:
        assert set(spec["train_peptides"]).isdisjoint(spec["val_peptides"])
        assert set(spec["train_times"]).isdisjoint(spec["val_times"])


def test_replicate_aggregation_reports_pass_fraction_and_std(joint_diag_d):
    rows = []
    for split, passed in enumerate((True, False, True)):
        rows.append(
            {
                "split": split,
                "arm": "diag_d_absolute",
                "gamma": 0.1,
                "eta": 0.022,
                "ensemble": "A",
                "bc": 0.2 + split * 0.01,
                "bh": 0.5,
                "val_mse": 0.1 + split * 0.01,
                "val_diag_d_loss": 2.0,
                "recovery": 50.0 + split,
                "ess": 5.0 + split,
                "decoy": 0.1,
                "mean_gate_reference_mse": 0.1,
                "mean_gate_passed": passed,
                "restart_ess_spread": 1.0 + split,
                "restart_ess_std": 0.5,
                "best_objective_is_lowest_ess": split != 1,
            }
        )
    aggregate = joint_diag_d._aggregate_replicates(pd.DataFrame(rows))
    row = aggregate.iloc[0]
    assert row["n_replicates"] == 3
    assert row["mean_gate_passed"] == pytest.approx(2.0 / 3.0)
    assert row["mean_gate_pass_fraction"] == pytest.approx(2.0 / 3.0)
    assert row["val_mse_std"] > 0.0
    assert np.isfinite(row["bh_std"])


def test_report_persists_first_party_weights_and_predictions(joint_diag_d):
    cells = _synthetic_cells(joint_diag_d)
    params = {
        "theta": jnp.asarray([0.1, -0.2]),
        "logits": {
            name: jnp.linspace(-0.5, 0.5, cell["n_frames"])
            for name, cell in cells.items()
        },
    }
    fit = {
        "params": params,
        "diagnostics": {
            name: {
                "best_start": 0,
                "best_objective": 1.0,
                "restart_ess_min": 2.0,
                "restart_ess_max": 2.0,
                "restart_ess_spread": 0.0,
                "restart_ess_std": 0.0,
                "best_objective_is_lowest_ess": True,
            }
            for name in cells
        },
    }
    payload = {}
    rows = joint_diag_d._report(
        cells,
        fit,
        gamma=0.1,
        eta=0.01,
        arm="diag_d_scalefree",
        payload=payload,
    )
    assert len(rows) == 2
    for row in rows:
        payload_id = row["payload_id"]
        weights = payload[f"{payload_id}__weights"]
        prediction = payload[f"{payload_id}__val_prediction"]
        assert weights.shape == (cells[row["ensemble"]]["n_frames"],)
        assert prediction.shape == cells[row["ensemble"]]["observed"].shape
        assert np.isfinite(weights).all() and np.isfinite(prediction).all()


def test_per_state_ess_known_weights_and_dominant_cluster(joint_diag_d):
    diagnostics = joint_diag_d.per_state_weight_diagnostics(
        np.asarray([0.25, 0.25, 0.5, 0.0]),
        np.asarray(["Folded", "Folded", "PUF3", "PUF2-like"]),
        cluster_labels=np.asarray([10, 11, 12, 13]),
    )
    assert diagnostics["mass_Folded"] == pytest.approx(0.5)
    assert diagnostics["ess_Folded"] == pytest.approx(2.0)
    assert diagnostics["mass_PUF3"] == pytest.approx(0.5)
    assert diagnostics["ess_PUF3"] == pytest.approx(1.0)
    assert diagnostics["mass_PUF2-like"] == pytest.approx(0.0)
    assert diagnostics["ess_PUF2-like"] == pytest.approx(0.0)
    assert diagnostics["dominant_cluster"] == 12
    assert diagnostics["dominant_weight"] == pytest.approx(0.5)


def test_per_state_ess_uniform_cluster_and_single_frame(joint_diag_d):
    diagnostics = joint_diag_d.per_state_weight_diagnostics(
        np.asarray([0.2, 0.2, 0.2, 0.2, 0.2]),
        np.asarray(["unfolded"] * 5),
    )
    assert diagnostics["ess_unfolded"] == pytest.approx(5.0)
    assert diagnostics["ess_PUF3"] == pytest.approx(0.0)
