"""Tests for the joint reweighting + BV-coefficient fitting helpers (non-negativity, gradients)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"


@pytest.fixture(scope="module")
def jrf():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("moprp_joint_reweight_fit")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


def test_coeffs_are_non_negative_and_recover_reference(jrf):
    # softplus is strictly positive for any real theta
    for theta in ([-5.0, -5.0], [0.0, 0.0], [5.0, 3.0]):
        bc, bh = jrf._coeffs_from_theta(jnp.asarray(theta))
        assert float(bc) > 0 and float(bh) > 0
    # inverse-softplus round-trips the published reference
    theta = jnp.asarray([jrf._inv_softplus(jrf.REF_BC), jrf._inv_softplus(jrf.REF_BH)])
    bc, bh = jrf._coeffs_from_theta(theta)
    assert float(bc) == pytest.approx(jrf.REF_BC, rel=1e-5)
    assert float(bh) == pytest.approx(jrf.REF_BH, rel=1e-5)


def test_coeffs_from_theta_differentiable(jrf):
    # gradient of a scalar of (Bc, Bh) flows back to theta (coefficients are fit-able)
    def f(theta):
        bc, bh = jrf._coeffs_from_theta(theta)
        return bc * 2.0 + bh
    grad = jax.grad(f)(jnp.asarray([0.0, 0.0]))
    assert np.all(np.isfinite(np.asarray(grad)))
    assert np.all(np.asarray(grad) > 0)  # softplus derivative is positive
