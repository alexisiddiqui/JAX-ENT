import jax.numpy as jnp
import numpy as np
import pytest
from jaxent.src.models.SAXS.features import (
    SAXS_reweighted_input_features,
    SAXS_basis_input_features,
    SAXS_output_features,
)
from jaxent.src.custom_types.key import m_key

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
