"""
Unit tests for BV forward pass implementations.
Tests numerical correctness using synthetic data with hand-computed known answers.
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
from jaxent.src.models.HDX.BV.features import (
    BV_input_features,
    BV_output_features,
    uptake_BV_output_features,
)
from jaxent.src.models.HDX.BV.parameters import (
    BV_Model_Parameters,
    linear_BV_Model_Parameters,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_bv_inputs():
    """Simple 3-residue BV inputs for known-answer testing."""
    heavy = jnp.array([1.0, 2.0, 3.0])
    acceptor = jnp.array([0.5, 1.0, 1.5])
    k_ints = jnp.array([0.1, 0.5, 1.0])
    return BV_input_features(
        heavy_contacts=heavy, acceptor_contacts=acceptor, k_ints=k_ints
    )


@pytest.fixture
def default_bv_params():
    """Default BV parameters for testing."""
    return BV_Model_Parameters(
        bv_bc=jnp.array([0.35]),
        bv_bh=jnp.array([2.0]),
        timepoints=jnp.array([0.167, 1.0, 10.0]),
    )


@pytest.fixture
def zero_contacts_inputs():
    """Test inputs with zero contacts."""
    return BV_input_features(
        heavy_contacts=jnp.zeros(3),
        acceptor_contacts=jnp.zeros(3),
        k_ints=jnp.array([0.1, 0.5, 1.0]),
    )


@pytest.fixture
def unit_contacts_inputs():
    """Test inputs with unit contacts."""
    return BV_input_features(
        heavy_contacts=jnp.ones(3),
        acceptor_contacts=jnp.ones(3),
        k_ints=jnp.array([0.1, 0.5, 1.0]),
    )


# ============================================================================
# Test BV_ForwardPass (log_pf = bc * heavy + bh * acceptor)
# ============================================================================


class TestBVForwardPass:
    """Test BV_ForwardPass log protection factor computation."""

    def test_zero_contacts(self, zero_contacts_inputs, default_bv_params):
        """Test that zero contacts give zero log_pf."""
        result = BV_ForwardPass()(zero_contacts_inputs, default_bv_params)
        expected = jnp.zeros(3)
        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-6)

    def test_unit_contacts(self, unit_contacts_inputs, default_bv_params):
        """Test with unit contacts: log_pf = 0.35 + 2.0 = 2.35 everywhere."""
        result = BV_ForwardPass()(unit_contacts_inputs, default_bv_params)
        # bc * 1 + bh * 1 = 0.35 + 2.0 = 2.35
        expected = jnp.full(3, 2.35)
        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-6)

    def test_known_values(self, simple_bv_inputs, default_bv_params):
        """Test with known numeric values.
        heavy=[1, 2, 3], acceptor=[0.5, 1, 1.5], bc=0.35, bh=2.0
        Expected: [0.35*1 + 2.0*0.5, 0.35*2 + 2.0*1, 0.35*3 + 2.0*1.5]
                = [1.35, 2.7, 4.05]
        """
        result = BV_ForwardPass()(simple_bv_inputs, default_bv_params)
        expected = jnp.array([1.35, 2.7, 4.05])
        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-6)

    def test_output_type(self, simple_bv_inputs, default_bv_params):
        """Test that output is BV_output_features with k_ints=None."""
        result = BV_ForwardPass()(simple_bv_inputs, default_bv_params)
        assert isinstance(result, BV_output_features)
        assert result.k_ints is None

    def test_jit_matches_eager(self, simple_bv_inputs, default_bv_params):
        """Test that JIT-compiled result matches eager evaluation."""
        forward_pass = BV_ForwardPass()
        eager_result = forward_pass(simple_bv_inputs, default_bv_params)
        jitted_result = jax.jit(forward_pass)(simple_bv_inputs, default_bv_params)
        np.testing.assert_allclose(
            eager_result.log_Pf, jitted_result.log_Pf, rtol=1e-6
        )

    def test_negative_contacts(self):
        """Test with negative contact values (physical edge case)."""
        inputs = BV_input_features(
            heavy_contacts=jnp.array([-1.0, 0.0, 1.0]),
            acceptor_contacts=jnp.array([-0.5, 0.0, 0.5]),
            k_ints=jnp.array([0.1, 0.5, 1.0]),
        )
        params = BV_Model_Parameters(
            bv_bc=jnp.array([0.35]), bv_bh=jnp.array([2.0])
        )
        result = BV_ForwardPass()(inputs, params)
        # Linear computation: -0.35 - 1.0 = -1.35, 0, 0.35 + 1.0 = 1.35
        expected = jnp.array([-1.35, 0.0, 1.35])
        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-6)


# ============================================================================
# Test BV_uptake_ForwardPass (uptake = 1 - exp(-kint * t / exp(log_pf)))
# ============================================================================


class TestBVUptakeForwardPass:
    """Test BV_uptake_ForwardPass uptake computation."""

    def test_zero_protection(self):
        """Test with bc=0, bh=0 (no protection).
        log_pf = 0 → pf = 1 → uptake = 1 - exp(-kint * t)
        For kint=0.1, t=0.167: 1 - exp(-0.1*0.167) ≈ 1 - exp(-0.0167) ≈ 0.0166
        """
        inputs = BV_input_features(
            heavy_contacts=jnp.array([0.1]),
            acceptor_contacts=jnp.array([0.1]),
            k_ints=jnp.array([0.1]),
        )
        params = BV_Model_Parameters(
            bv_bc=jnp.array([0.0]),
            bv_bh=jnp.array([0.0]),
            timepoints=jnp.array([0.167]),
        )
        result = BV_uptake_ForwardPass()(inputs, params)
        # pf = 1, uptake = 1 - exp(-0.1 * 0.167) ≈ 0.0166
        expected = 1.0 - jnp.exp(-0.1 * 0.167)
        np.testing.assert_allclose(result.uptake[0, 0], expected, rtol=1e-4)

    def test_large_protection_gives_low_uptake(self):
        """Test that large log_pf (high protection) gives low uptake."""
        inputs = BV_input_features(
            heavy_contacts=jnp.array([100.0]),  # Very large
            acceptor_contacts=jnp.array([100.0]),
            k_ints=jnp.array([1.0]),
        )
        params = BV_Model_Parameters(
            bv_bc=jnp.array([0.35]),
            bv_bh=jnp.array([2.0]),
            timepoints=jnp.array([1.0]),
        )
        result = BV_uptake_ForwardPass()(inputs, params)
        # High pf → low uptake
        assert result.uptake[0, 0] < 0.1

    def test_output_shape(self):
        """Test output shape for 3 residues, 3 timepoints."""
        inputs = BV_input_features(
            heavy_contacts=jnp.array([1.0, 2.0, 3.0]),
            acceptor_contacts=jnp.array([0.5, 1.0, 1.5]),
            k_ints=jnp.array([0.1, 0.5, 1.0]),
        )
        params = BV_Model_Parameters(
            bv_bc=jnp.array([0.35]),
            bv_bh=jnp.array([2.0]),
            timepoints=jnp.array([0.167, 1.0, 10.0]),
        )
        result = BV_uptake_ForwardPass()(inputs, params)
        assert result.uptake.shape == (3, 3)

    def test_uptake_monotonic_in_time(self):
        """Test that uptake increases with time for each residue."""
        inputs = BV_input_features(
            heavy_contacts=jnp.array([1.0, 2.0]),
            acceptor_contacts=jnp.array([0.5, 1.0]),
            k_ints=jnp.array([0.1, 0.5]),
        )
        params = BV_Model_Parameters(
            bv_bc=jnp.array([0.35]),
            bv_bh=jnp.array([2.0]),
            timepoints=jnp.array([0.001, 0.1, 1.0, 10.0]),
        )
        result = BV_uptake_ForwardPass()(inputs, params)
        # Check monotonicity: uptake[t+1] >= uptake[t] for each residue
        for res_idx in range(2):
            uptakes = result.uptake[:, res_idx]
            assert jnp.all(jnp.diff(uptakes) >= -1e-6)  # Allow tiny numerical error

    def test_output_type(self):
        """Test that output is uptake_BV_output_features."""
        inputs = BV_input_features(
            heavy_contacts=jnp.array([1.0]),
            acceptor_contacts=jnp.array([0.5]),
            k_ints=jnp.array([0.1]),
        )
        params = BV_Model_Parameters(
            bv_bc=jnp.array([0.35]),
            bv_bh=jnp.array([2.0]),
            timepoints=jnp.array([0.167, 1.0, 10.0]),
        )
        result = BV_uptake_ForwardPass()(inputs, params)
        assert isinstance(result, uptake_BV_output_features)

    def test_gradient_flow(self):
        """Test that gradients can flow through BV_uptake_ForwardPass."""
        # Simple test that gradients are computable
        inputs = BV_input_features(
            heavy_contacts=jnp.array([1.0]),
            acceptor_contacts=jnp.array([0.5]),
            k_ints=jnp.array([0.1]),
        )
        params = BV_Model_Parameters(
            bv_bc=jnp.array([0.35]),
            bv_bh=jnp.array([2.0]),
            timepoints=jnp.array([0.167, 1.0, 10.0]),
        )

        def loss_fn(bc):
            params_updated = BV_Model_Parameters(
                bv_bc=bc,
                bv_bh=params.bv_bh,
                timepoints=params.timepoints,
            )
            result = BV_uptake_ForwardPass()(inputs, params_updated)
            return jnp.mean(result.uptake)

        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(params.bv_bc)
        assert jnp.isfinite(grad).all()


# ============================================================================
# Test linear_BV_ForwardPass (uptake = bc * heavy + bh * acceptor)
# ============================================================================


class TestLinearBVForwardPass:
    """Test linear_BV_ForwardPass linear uptake computation."""

    def test_known_linear_combination(self, simple_bv_inputs):
        """Test linear combination: uptake = bc * heavy + bh * acceptor."""
        params = linear_BV_Model_Parameters(
            bv_bc=jnp.array([0.35]),
            bv_bh=jnp.array([2.0]),
        )
        result = linear_BV_ForwardPass()(simple_bv_inputs, params)
        # 0.35 * [1, 2, 3] + 2.0 * [0.5, 1, 1.5] = [1.35, 2.7, 4.05]
        expected = jnp.array([1.35, 2.7, 4.05])
        np.testing.assert_allclose(result.uptake, expected, rtol=1e-6)

    def test_output_type(self, simple_bv_inputs):
        """Test that output is uptake_BV_output_features."""
        params = linear_BV_Model_Parameters(
            bv_bc=jnp.array([0.35]),
            bv_bh=jnp.array([2.0]),
        )
        result = linear_BV_ForwardPass()(simple_bv_inputs, params)
        assert isinstance(result, uptake_BV_output_features)

    def test_zero_params(self):
        """Test with zero parameters."""
        inputs = BV_input_features(
            heavy_contacts=jnp.array([1.0, 2.0]),
            acceptor_contacts=jnp.array([0.5, 1.0]),
            k_ints=jnp.array([0.1, 0.5]),
        )
        params = linear_BV_Model_Parameters(
            bv_bc=jnp.array([0.0]),
            bv_bh=jnp.array([0.0]),
        )
        result = linear_BV_ForwardPass()(inputs, params)
        expected = jnp.zeros(2)
        np.testing.assert_allclose(result.uptake, expected, rtol=1e-6)


# ============================================================================
# Test PyTree round-trip (flatten/unflatten)
# ============================================================================


class TestBVPytreeRoundTrip:
    """Test that PyTree operations preserve data."""

    def test_input_features_roundtrip(self, simple_bv_inputs):
        """Test BV_input_features flatten/unflatten."""
        flat, aux = simple_bv_inputs.tree_flatten()
        reconstructed = BV_input_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(
            reconstructed.heavy_contacts, simple_bv_inputs.heavy_contacts, rtol=1e-6
        )
        np.testing.assert_allclose(
            reconstructed.acceptor_contacts,
            simple_bv_inputs.acceptor_contacts,
            rtol=1e-6,
        )

    def test_output_features_roundtrip(self):
        """Test BV_output_features flatten/unflatten."""
        log_pf = jnp.array([1.0, 2.0, 3.0])
        output = BV_output_features(log_Pf=log_pf, k_ints=None)
        flat, aux = output.tree_flatten()
        reconstructed = BV_output_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(reconstructed.log_Pf, output.log_Pf, rtol=1e-6)

    def test_parameters_roundtrip(self, default_bv_params):
        """Test BV_Model_Parameters flatten/unflatten."""
        flat, aux = default_bv_params.tree_flatten()
        reconstructed = BV_Model_Parameters.tree_unflatten(aux, flat)
        np.testing.assert_allclose(
            reconstructed.bv_bc, default_bv_params.bv_bc, rtol=1e-6
        )
        np.testing.assert_allclose(
            reconstructed.bv_bh, default_bv_params.bv_bh, rtol=1e-6
        )

    def test_uptake_output_roundtrip(self):
        """Test uptake_BV_output_features flatten/unflatten."""
        uptake = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        output = uptake_BV_output_features(uptake=uptake)
        flat, aux = output.tree_flatten()
        reconstructed = uptake_BV_output_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(
            reconstructed.uptake, output.uptake, rtol=1e-6
        )
