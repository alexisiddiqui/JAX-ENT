import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxent.src.models.XLMS.features import XLMS_input_features, XLMS_output_features
from jaxent.src.custom_types.key import m_key


class TestXLMSInputFeatures:
    def test_features_shape(self):
        f = XLMS_input_features(distances=jnp.ones((10, 10, 100)))
        assert f.features_shape == (10, 10, 100)

    def test_feat_pred_returns_distances(self):
        arr = jnp.ones((5, 5, 20))
        f = XLMS_input_features(distances=arr)
        preds = f.feat_pred
        assert len(preds) == 1
        assert preds[0].shape == (5, 5, 20)

    def test_key_contains_xlms_distance(self):
        assert m_key("XLMS_distance") in XLMS_input_features.key

    def test_pytree_roundtrip(self):
        f = XLMS_input_features(distances=jnp.ones((4, 4, 10)))
        flat, aux = f.tree_flatten()
        r = XLMS_input_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(r.distances, f.distances)


class TestXLMSOutputFeatures:
    def test_y_pred_returns_distance_matrix(self):
        mat = jnp.eye(4)
        f = XLMS_output_features(distances=mat)
        np.testing.assert_allclose(f.y_pred(), mat)

    def test_output_shape(self):
        f = XLMS_output_features(distances=jnp.ones((10, 10)))
        assert f.output_shape == (10, 10)

    def test_key_is_xlms_distance(self):
        assert XLMS_output_features.key == m_key("XLMS_distance")

    def test_pytree_roundtrip(self):
        f = XLMS_output_features(distances=jnp.eye(3))
        flat, aux = f.tree_flatten()
        r = XLMS_output_features.tree_unflatten(aux, flat)
        np.testing.assert_allclose(r.distances, f.distances)
