import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxent.src.models.XLMS.features import XLMS_input_features, XLMS_output_features
from jaxent.src.models.XLMS.parameters import XLMS_Model_Parameters
from jaxent.src.models.XLMS.forward import XLMS_distance_ForwardPass
from jaxent.src.models.XLMS.config import XLMS_Config
from jaxent.src.models.XLMS.forwardmodel import XLMS_distance_model
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

class TestXLMSModelParameters:
    def test_no_dynamic_params(self):
        """XLMS has no optimizable model parameters — only frame weights."""
        p = XLMS_Model_Parameters()
        flat, aux = p.tree_flatten()
        assert flat == ()

    def test_key_is_xlms_distance(self):
        assert m_key("XLMS_distance") in XLMS_Model_Parameters().key

    def test_pytree_roundtrip(self):
        p = XLMS_Model_Parameters()
        flat, aux = p.tree_flatten()
        r = XLMS_Model_Parameters.tree_unflatten(aux, flat)
        assert r.key == p.key

class TestXLMSDistanceForwardPass:
    def test_returns_averaged_distance_matrix(self):
        """After frame averaging, distances is (n_residues, n_residues). Forward wraps it."""
        mat = jnp.array([[0.0, 5.0], [5.0, 0.0]])  # already averaged
        features = XLMS_input_features(distances=mat)
        params = XLMS_Model_Parameters()
        result = XLMS_distance_ForwardPass()(features, params)
        assert isinstance(result, XLMS_output_features)
        np.testing.assert_allclose(result.y_pred(), mat)

    def test_jit_compatible(self):
        features = XLMS_input_features(distances=jnp.eye(4))
        params = XLMS_Model_Parameters()
        result = jax.jit(XLMS_distance_ForwardPass())(features, params)
        assert result.distances.shape == (4, 4)

    def test_gradient_flows_through_features(self):
        def loss(dists):
            f = XLMS_input_features(distances=dists)
            out = XLMS_distance_ForwardPass()(f, XLMS_Model_Parameters())
            return out.distances.sum()
        grad = jax.grad(loss)(jnp.ones((3, 3)))
        assert jnp.isfinite(grad).all()

class TestXLMSConfig:
    def test_key_is_xlms_distance(self):
        cfg = XLMS_Config()
        assert cfg.key == m_key("XLMS_distance")

    def test_forward_parameters_type(self):
        cfg = XLMS_Config()
        assert isinstance(cfg.forward_parameters, XLMS_Model_Parameters)

class TestXLMSModel:
    def test_has_xlms_forward_pass(self):
        model = XLMS_distance_model(XLMS_Config())
        assert m_key("XLMS_distance") in model.forward
        assert isinstance(model.forward[m_key("XLMS_distance")], XLMS_distance_ForwardPass)

    def test_initialise_returns_true(self):
        assert XLMS_distance_model(XLMS_Config()).initialise([]) is True

    def test_featurise_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            XLMS_distance_model(XLMS_Config()).featurise([])
