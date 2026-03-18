import jax.numpy as jnp
import numpy as np
import pytest
from jaxent.src.models.SAXS.features import (
    SAXS_reweighted_input_features,
    SAXS_basis_input_features,
    SAXS_output_features,
)
from jaxent.src.models.SAXS.parameters import SAXS_Reweighted_Parameters, SAXS_Debye_Parameters
from jaxent.src.custom_types.key import m_key
import jax

class TestSAXSReweightedInputFeatures:
    def test_features_shape(self):
        f = SAXS_reweighted_input_features(intensities=jnp.ones((501, 100)))
        assert f.features_shape == (501, 100)

    def test_feat_pred_returns_intensities(self):
        arr = jnp.ones((501, 100))
        f = SAXS_reweighted_input_features(intensities=arr)
        preds = f.feat_pred
        assert len(preds) == 1
        assert preds[0].shape == (501, 100)

    def test_key_contains_saxs_iq(self):
        assert m_key("SAXS_Iq") in SAXS_reweighted_input_features.key

    def test_pytree_roundtrip(self):
        f = SAXS_reweighted_input_features(intensities=jnp.ones((10, 5)))
        flat, aux = f.tree_flatten()
        r = SAXS_reweighted_input_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(r.intensities, f.intensities)

class TestSAXSBasisInputFeatures:
    def test_features_shape(self):
        f = SAXS_basis_input_features(basis_profiles=jnp.ones((6, 501, 100)))
        assert f.features_shape == (6, 501, 100)

    def test_pytree_roundtrip(self):
        f = SAXS_basis_input_features(basis_profiles=jnp.ones((6, 10, 5)))
        flat, aux = f.tree_flatten()
        r = SAXS_basis_input_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(r.basis_profiles, f.basis_profiles)

class TestSAXSOutputFeatures:
    def test_y_pred_returns_intensity(self):
        arr = jnp.array([1.0, 2.0, 3.0])
        f = SAXS_output_features(intensity=arr)
        np.testing.assert_allclose(f.y_pred(), arr)

    def test_output_shape(self):
        f = SAXS_output_features(intensity=jnp.ones(501))
        assert f.output_shape == (501,)

    def test_key_is_saxs_iq(self):
        assert SAXS_output_features.key == m_key("SAXS_Iq")

    def test_pytree_roundtrip(self):
        f = SAXS_output_features(intensity=jnp.array([1.0, 2.0]))
        flat, aux = f.tree_flatten()
        r = SAXS_output_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(r.intensity, f.intensity)

class TestSAXSReweightedParameters:
    def test_no_dynamic_params(self):
        p = SAXS_Reweighted_Parameters()
        flat, aux = p.tree_flatten()
        assert flat == ()  # no dynamic params

    def test_pytree_roundtrip(self):
        p = SAXS_Reweighted_Parameters()
        flat, aux = p.tree_flatten()
        r = SAXS_Reweighted_Parameters.tree_unflatten(aux, flat)
        assert r.key == p.key

    def test_key_is_saxs_iq(self):
        assert m_key("SAXS_Iq") in SAXS_Reweighted_Parameters().key

class TestSAXSDebyeParameters:
    def test_default_values(self):
        p = SAXS_Debye_Parameters()
        np.testing.assert_allclose(p.c1, 1.0)
        np.testing.assert_allclose(p.c2, 0.0)
        np.testing.assert_allclose(p.c, 1.0)
        np.testing.assert_allclose(p.b, 0.0)

    def test_pytree_roundtrip(self):
        p = SAXS_Debye_Parameters(c1=jnp.array(1.2), c2=jnp.array(0.1),
                                   c=jnp.array(0.9), b=jnp.array(0.05))
        flat, aux = p.tree_flatten()
        r = SAXS_Debye_Parameters.tree_unflatten(aux, flat)
        np.testing.assert_allclose(r.c1, p.c1)
        np.testing.assert_allclose(r.c2, p.c2)

    def test_mul_preserves_key(self):
        p = SAXS_Debye_Parameters()
        p2 = p * 2.0
        assert p2.key == p.key

    def test_gradient_flows(self):
        def loss(c1):
            p = SAXS_Debye_Parameters(c1=c1, c2=jnp.array(0.0),
                                       c=jnp.array(1.0), b=jnp.array(0.0))
            return p.c1.sum()
        grad = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(grad)
