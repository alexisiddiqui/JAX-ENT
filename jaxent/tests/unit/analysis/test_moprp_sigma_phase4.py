"""Correctness and regression tests for the Phase-4 covariance substitution."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from jaxent.src.opt.losses import LOSS_REGISTRY, hdx_uptake_joint_gaussian_loss

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"
sys.path.insert(0, str(FITTING_DIR))
phase4 = importlib.import_module("moprp_sigma_phase4")
equivalence = importlib.import_module("phase4_default_path_equivalence")
sys.path.remove(str(FITTING_DIR))


def test_peptide_one_marginal_is_strided_not_contiguous_and_both_are_psd():
    target = np.load(phase4.DEFAULT_TARGET)
    covariance = target["covariance"]
    indices = phase4.retained_time_major_indices(14, 15, 0)
    expected = np.asarray([i for i in range(210) if i % 14 != 0])
    np.testing.assert_array_equal(indices, expected)
    strided, strided_chol = phase4.marginal_covariance(covariance, indices)
    contiguous, contiguous_chol = phase4.marginal_covariance(covariance, np.arange(15, 210))
    assert strided.shape == contiguous.shape == (195, 195)
    assert not np.array_equal(strided, contiguous)
    np.testing.assert_allclose(strided_chol @ strided_chol.T, strided, atol=1e-13)
    np.testing.assert_allclose(contiguous_chol @ contiguous_chol.T, contiguous, atol=1e-13)


def test_eye_mse_arm_is_exact_untouched_expression():
    eye = dict(phase4.load_arms())["eye_mse"]
    residual = jnp.asarray([[0.2, -0.1], [0.4, 0.3]])
    expected = jnp.mean(residual**2)
    assert np.asarray(eye(residual)).tobytes() == np.asarray(expected).tobytes()


def test_joint_gaussian_loss_is_registered_additively():
    assert LOSS_REGISTRY["joint_gaussian"] is hdx_uptake_joint_gaussian_loss


def test_default_path_compare_identical_payloads_have_zero_deviation():
    payload = {"rows": [{"score": 1.5, "selected": True}], "label": "default"}
    assert equivalence.flatten_numeric(payload) == {"rows.0.score": 1.5}
    result = equivalence.compare(payload, payload)
    assert result["max_abs"] == result["max_rel"] == 0.0
    assert result["structure_identical"]


def test_default_path_compare_reports_perturbed_numeric_path():
    a = {"outer": {"small": 1.0, "large": 4.0}}
    b = {"outer": {"small": 1.0, "large": 4.25}}
    result = equivalence.compare(a, b)
    assert result["max_abs"] == 0.25
    assert result["argmax_path"] == "outer.large"
    assert result["structure_identical"]


def test_default_path_compare_rejects_key_or_string_changes_structurally():
    original = {"value": 1.0, "label": "unchanged"}
    assert not equivalence.compare(original, {"renamed": 1.0, "label": "unchanged"})[
        "structure_identical"
    ]
    assert not equivalence.compare(original, {"value": 1.0, "label": "changed"})[
        "structure_identical"
    ]
    assert equivalence.compare(original, {"label": "unchanged", "value": 1.0})[
        "structure_identical"
    ]
