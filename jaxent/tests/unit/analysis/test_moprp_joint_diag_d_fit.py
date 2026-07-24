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
