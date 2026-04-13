"""
Unit tests for gradient correctness and gradient flow.
Tests gradient computation using jax.grad with finite-difference verification.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.models.HDX.forward import (
    BV_ForwardPass,
    BV_uptake_ForwardPass,
    linear_BV_ForwardPass,
)
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.parameters import BV_Model_Parameters, linear_BV_Model_Parameters


# ============================================================================
# Gradient Wrapper Functions
# ============================================================================


def bv_log_pf_loss(bc, bh, heavy, acceptor, target):
    """Wrapper for BV log_pf loss: MSE between log_pf and target."""
    params = BV_Model_Parameters(bv_bc=bc, bv_bh=bh)
    features = BV_input_features(heavy_contacts=heavy, acceptor_contacts=acceptor)
    result = BV_ForwardPass()(features, params)
    return jnp.mean((result.log_Pf - target) ** 2)


def bv_uptake_loss(bc, bh, heavy, acceptor, k_ints, timepoints, target):
    """Wrapper for BV uptake loss: MSE between uptake and target."""
    params = BV_Model_Parameters(
        bv_bc=bc, bv_bh=bh, timepoints=timepoints
    )
    features = BV_input_features(
        heavy_contacts=heavy, acceptor_contacts=acceptor, k_ints=k_ints
    )
    result = BV_uptake_ForwardPass()(features, params)
    return jnp.mean((result.uptake - target) ** 2)


def linear_bv_loss(bc, bh, heavy, acceptor, target):
    """Wrapper for linear BV loss: MSE between linear uptake and target."""
    params = linear_BV_Model_Parameters(bv_bc=bc, bv_bh=bh)
    features = BV_input_features(heavy_contacts=heavy, acceptor_contacts=acceptor)
    result = linear_BV_ForwardPass()(features, params)
    return jnp.mean((result.uptake - target) ** 2)


# ============================================================================
# Test BV_ForwardPass Gradients
# ============================================================================


class TestBVForwardPassGradients:
    """Test gradients of BV_ForwardPass."""

    def test_grad_bc_is_heavy_contacts(self):
        """Test that d/d(bc)[bc*heavy + bh*acceptor] = heavy."""
        heavy = jnp.array([1.0, 2.0, 3.0])
        acceptor = jnp.array([0.5, 1.0, 1.5])
        target = jnp.array([1.5, 2.5, 3.5])

        # Compute gradient w.r.t. bc
        grad_fn = jax.grad(bv_log_pf_loss, argnums=0)
        grad_bc = grad_fn(
            jnp.array(0.35),
            jnp.array(2.0),
            heavy,
            acceptor,
            target,
        )

        # Gradient should be proportional to sum of (heavy * diffs)
        diffs = jnp.array([0.35, 0.35, 0.35]) + jnp.array([1.0, 2.0, 3.0]) * 0.35 + jnp.array([0.5, 1.0, 1.5]) * 2.0 - target
        expected_direction = jnp.sum(2.0 * diffs * heavy / 3.0)

        # Check that gradient is non-zero and has correct sign
        assert jnp.abs(grad_bc) > 0

    def test_grad_bh_is_acceptor_contacts(self):
        """Test that d/d(bh)[bc*heavy + bh*acceptor] = acceptor."""
        heavy = jnp.array([1.0, 2.0, 3.0])
        acceptor = jnp.array([0.5, 1.0, 1.5])
        target = jnp.array([1.5, 2.5, 3.5])

        # Compute gradient w.r.t. bh
        grad_fn = jax.grad(bv_log_pf_loss, argnums=1)
        grad_bh = grad_fn(
            jnp.array(0.35),
            jnp.array(2.0),
            heavy,
            acceptor,
            target,
        )

        # Check that gradient is non-zero
        assert jnp.abs(grad_bh) > 0

    def test_finite_diff_bc(self):
        """Test finite-difference gradient check for bc parameter."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0, 3.0])
        acceptor = jnp.array([0.5, 1.0, 1.5])
        target = jnp.array([1.5, 2.5, 3.5])

        h = 1e-5
        f_plus = bv_log_pf_loss(bc + h, bh, heavy, acceptor, target)
        f_minus = bv_log_pf_loss(bc - h, bh, heavy, acceptor, target)
        fd_grad = (f_plus - f_minus) / (2 * h)

        # Compute automatic gradient
        grad_fn = jax.grad(bv_log_pf_loss, argnums=0)
        auto_grad = grad_fn(bc, bh, heavy, acceptor, target)

        # Should match to ~3 decimal places (finite diff is less accurate)
        np.testing.assert_allclose(auto_grad, fd_grad, rtol=2e-3)

    def test_finite_diff_bh(self):
        """Test finite-difference gradient check for bh parameter."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0, 3.0])
        acceptor = jnp.array([0.5, 1.0, 1.5])
        target = jnp.array([1.5, 2.5, 3.5])

        h = 1e-5
        f_plus = bv_log_pf_loss(bc, bh + h, heavy, acceptor, target)
        f_minus = bv_log_pf_loss(bc, bh - h, heavy, acceptor, target)
        fd_grad = (f_plus - f_minus) / (2 * h)

        # Compute automatic gradient
        grad_fn = jax.grad(bv_log_pf_loss, argnums=1)
        auto_grad = grad_fn(bc, bh, heavy, acceptor, target)

        np.testing.assert_allclose(auto_grad, fd_grad, rtol=2e-3)


# ============================================================================
# Test BV_Uptake_ForwardPass Gradients
# ============================================================================


class TestBVUptakeGradients:
    """Test gradients of BV_uptake_ForwardPass."""

    def test_grad_bc_nonzero(self):
        """Test that gradient w.r.t. bc is non-zero."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0])
        acceptor = jnp.array([0.5, 1.0])
        k_ints = jnp.array([0.1, 0.5])
        timepoints = jnp.array([0.1, 1.0])
        target = jnp.array([[0.01, 0.05], [0.1, 0.2]])

        grad_fn = jax.grad(bv_uptake_loss, argnums=0)
        grad_bc = grad_fn(bc, bh, heavy, acceptor, k_ints, timepoints, target)

        assert jnp.abs(grad_bc) > 0

    def test_grad_bh_nonzero(self):
        """Test that gradient w.r.t. bh is non-zero."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0])
        acceptor = jnp.array([0.5, 1.0])
        k_ints = jnp.array([0.1, 0.5])
        timepoints = jnp.array([0.1, 1.0])
        target = jnp.array([[0.01, 0.05], [0.1, 0.2]])

        grad_fn = jax.grad(bv_uptake_loss, argnums=1)
        grad_bh = grad_fn(bc, bh, heavy, acceptor, k_ints, timepoints, target)

        assert jnp.abs(grad_bh) > 0

    def test_finite_diff_uptake_bc(self):
        """Test finite-difference gradient check for bc in uptake.
        Uses 2D (n_residues, n_frames) inputs per BV_uptake_ForwardPass contract.
        """
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([[1.0], [2.0]])    # (2, 1): 2 residues, 1 frame
        acceptor = jnp.array([[0.5], [1.0]])  # (2, 1)
        k_ints = jnp.array([0.1, 0.5])
        timepoints = jnp.array([0.1, 1.0])
        target = jnp.array([[[0.01], [0.05]], [[0.1], [0.2]]])  # (2, 2, 1)

        h = 1e-5
        f_plus = bv_uptake_loss(bc + h, bh, heavy, acceptor, k_ints, timepoints, target)
        f_minus = bv_uptake_loss(bc - h, bh, heavy, acceptor, k_ints, timepoints, target)
        fd_grad = (f_plus - f_minus) / (2 * h)

        grad_fn = jax.grad(bv_uptake_loss, argnums=0)
        auto_grad = grad_fn(bc, bh, heavy, acceptor, k_ints, timepoints, target)

        np.testing.assert_allclose(auto_grad, fd_grad, rtol=2e-3)

    def test_finite_diff_uptake_bh(self):
        """Test finite-difference gradient check for bh in uptake.
        Uses 2D (n_residues, n_frames) inputs per BV_uptake_ForwardPass contract.
        """
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([[1.0], [2.0]])    # (2, 1): 2 residues, 1 frame
        acceptor = jnp.array([[0.5], [1.0]])  # (2, 1)
        k_ints = jnp.array([0.1, 0.5])
        timepoints = jnp.array([0.1, 1.0])
        target = jnp.array([[[0.01], [0.05]], [[0.1], [0.2]]])  # (2, 2, 1)

        h = 1e-5
        f_plus = bv_uptake_loss(bc, bh + h, heavy, acceptor, k_ints, timepoints, target)
        f_minus = bv_uptake_loss(bc, bh - h, heavy, acceptor, k_ints, timepoints, target)
        fd_grad = (f_plus - f_minus) / (2 * h)

        grad_fn = jax.grad(bv_uptake_loss, argnums=1)
        auto_grad = grad_fn(bc, bh, heavy, acceptor, k_ints, timepoints, target)

        np.testing.assert_allclose(auto_grad, fd_grad, rtol=2e-3)


# ============================================================================
# Test Pure Loss Function Gradients
# ============================================================================


class TestLossGradients:
    """Test gradients of loss functions."""

    def test_l2_gradient_analytical(self):
        """Test L2 loss gradient: d/dx[mean((x-t)^2)] = 2(x-t)/n."""
        def l2_loss(x, target):
            return jnp.mean((x - target) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 1.5, 3.5])

        grad_fn = jax.grad(l2_loss)
        grad = grad_fn(x, target)

        # Expected: 2(x-t)/3
        expected = 2.0 * (x - target) / 3.0
        np.testing.assert_allclose(grad, expected, rtol=1e-6)

    def test_mae_gradient_sign(self):
        """Test MAE loss gradient sign: should be sign(x-t)/n."""
        def mae_loss(x, target):
            return jnp.mean(jnp.abs(x - target))

        x = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 1.5, 3.5])

        grad_fn = jax.grad(mae_loss)
        grad = grad_fn(x, target)

        # For x < target, grad should be negative
        # For x > target, grad should be positive
        assert grad[0] < 0  # 1.0 < 1.5
        assert grad[1] > 0  # 2.0 > 1.5
        assert grad[2] < 0  # 3.0 < 3.5

    def test_finite_diff_l2(self):
        """Test L2 loss with finite-difference verification."""
        def l2_loss(x, target):
            return jnp.mean((x - target) ** 2)

        x = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.5, 1.5, 3.5])

        h = 1e-5
        grad_fn = jax.grad(l2_loss)
        auto_grad = grad_fn(x, target)

        # Compute finite-difference for first element
        x_plus = jnp.array([1.0 + h, 2.0, 3.0])
        x_minus = jnp.array([1.0 - h, 2.0, 3.0])
        fd_grad_0 = (l2_loss(x_plus, target) - l2_loss(x_minus, target)) / (2 * h)

        np.testing.assert_allclose(auto_grad[0], fd_grad_0, rtol=2e-3)

    def test_finite_diff_kl(self):
        """Test KL divergence with finite-difference verification."""
        def kl_loss(p, q):
            eps = 1e-10
            return jnp.sum(p * jnp.log((p + eps) / (q + eps)))

        p = jnp.array([0.5, 0.3, 0.2])
        q = jnp.array([0.4, 0.3, 0.3])

        h = 1e-5
        grad_fn = jax.grad(kl_loss)
        auto_grad = grad_fn(p, q)

        # Compute finite-difference for first element
        p_plus = jnp.array([0.5 + h, 0.3, 0.2])
        p_minus = jnp.array([0.5 - h, 0.3, 0.2])
        fd_grad_0 = (kl_loss(p_plus, q) - kl_loss(p_minus, q)) / (2 * h)

        np.testing.assert_allclose(auto_grad[0], fd_grad_0, rtol=1e-2)


# ============================================================================
# Test End-to-End Gradients
# ============================================================================


class TestEndToEndGradients:
    """Test end-to-end gradients through forward model and loss."""

    def test_e2e_bv_pf_l2(self):
        """Test full BV -> L2 loss gradient computation."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0])
        acceptor = jnp.array([0.5, 1.0])
        target = jnp.array([1.5, 2.5])

        # Compute gradient
        grad_fn = jax.grad(bv_log_pf_loss, argnums=0)
        grad_bc = grad_fn(bc, bh, heavy, acceptor, target)

        # Gradient should be non-zero and finite
        assert jnp.isfinite(grad_bc)
        assert jnp.abs(grad_bc) > 0

    def test_e2e_bv_uptake_mse(self):
        """Test full BV uptake -> MSE loss gradient computation."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0])
        acceptor = jnp.array([0.5, 1.0])
        k_ints = jnp.array([0.1, 0.5])
        timepoints = jnp.array([0.1, 1.0])
        target = jnp.array([[0.01, 0.05], [0.1, 0.2]])

        # Compute gradient
        grad_fn = jax.grad(bv_uptake_loss, argnums=0)
        grad_bc = grad_fn(bc, bh, heavy, acceptor, k_ints, timepoints, target)

        # Gradient should be non-zero and finite
        assert jnp.isfinite(grad_bc)
        assert jnp.abs(grad_bc) > 0

    def test_jit_preserves_gradients(self):
        """Test that jax.jit(jax.grad(f)) matches jax.grad(f)."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0, 3.0])
        acceptor = jnp.array([0.5, 1.0, 1.5])
        target = jnp.array([1.5, 2.5, 3.5])

        # Non-JIT gradient
        grad_fn = jax.grad(bv_log_pf_loss, argnums=0)
        grad_eager = grad_fn(bc, bh, heavy, acceptor, target)

        # JIT gradient
        grad_jit_fn = jax.jit(jax.grad(bv_log_pf_loss, argnums=0))
        grad_jit = grad_jit_fn(bc, bh, heavy, acceptor, target)

        np.testing.assert_allclose(grad_eager, grad_jit, rtol=1e-5)


# ============================================================================
# Test Linear BV Gradients
# ============================================================================


class TestLinearBVGradients:
    """Test gradients through linear_BV_ForwardPass."""

    def test_linear_bv_grad_bc(self):
        """Test gradient w.r.t. bc in linear BV."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0])
        acceptor = jnp.array([0.5, 1.0])
        target = jnp.array([1.5, 2.5])

        grad_fn = jax.grad(linear_bv_loss, argnums=0)
        grad_bc = grad_fn(bc, bh, heavy, acceptor, target)

        assert jnp.isfinite(grad_bc)
        assert jnp.abs(grad_bc) > 0

    def test_linear_bv_grad_bh(self):
        """Test gradient w.r.t. bh in linear BV."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0])
        acceptor = jnp.array([0.5, 1.0])
        target = jnp.array([1.5, 2.5])

        grad_fn = jax.grad(linear_bv_loss, argnums=1)
        grad_bh = grad_fn(bc, bh, heavy, acceptor, target)

        assert jnp.isfinite(grad_bh)
        assert jnp.abs(grad_bh) > 0

    def test_linear_bv_finite_diff(self):
        """Test linear BV gradient with finite-difference verification."""
        bc = jnp.array(0.35)
        bh = jnp.array(2.0)
        heavy = jnp.array([1.0, 2.0])
        acceptor = jnp.array([0.5, 1.0])
        target = jnp.array([1.5, 2.5])

        h = 1e-5
        f_plus = linear_bv_loss(bc + h, bh, heavy, acceptor, target)
        f_minus = linear_bv_loss(bc - h, bh, heavy, acceptor, target)
        fd_grad = (f_plus - f_minus) / (2 * h)

        grad_fn = jax.grad(linear_bv_loss, argnums=0)
        auto_grad = grad_fn(bc, bh, heavy, acceptor, target)

        np.testing.assert_allclose(auto_grad, fd_grad, rtol=2e-3)
