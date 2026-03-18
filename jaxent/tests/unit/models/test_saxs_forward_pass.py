import jax.numpy as jnp
import numpy as np
import pytest
from jaxent.src.models.SAXS.features import (
    SAXS_curve_input_features,
    SAXS_basis_input_features,
    SAXS_output_features,
)
from jaxent.src.models.SAXS.parameters import SAXS_Reweighted_Parameters, SAXS_Debye_Parameters
from jaxent.src.models.SAXS.forward import SAXS_ReweightedForwardPass, SAXS_DebyeForwardPass
from jaxent.src.models.SAXS.config import SAXS_Config
from jaxent.src.models.SAXS.forwardmodel import SAXS_model
from jaxent.src.custom_types.key import m_key
import jax

class TestSAXSReweightedInputFeatures:
    def test_features_shape(self):
        f = SAXS_curve_input_features(intensities=jnp.ones((501, 100)))
        assert f.features_shape == (501, 100)

    def test_feat_pred_returns_intensities(self):
        arr = jnp.ones((501, 100))
        f = SAXS_curve_input_features(intensities=arr)
        preds = f.feat_pred
        assert len(preds) == 1
        assert preds[0].shape == (501, 100)

    def test_key_contains_saxs_iq(self):
        assert m_key("SAXS_Iq") in SAXS_curve_input_features.key

    def test_pytree_roundtrip(self):
        f = SAXS_curve_input_features(intensities=jnp.ones((10, 5)))
        flat, aux = f.tree_flatten()
        r = SAXS_curve_input_features.tree_unflatten(aux, flat)
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

class TestSAXSReweightedForwardPass:
    def test_identity_returns_averaged_intensities(self):
        """After frame averaging, intensities is (n_q,). Forward pass wraps it."""
        averaged = jnp.array([1.0, 2.0, 3.0])  # already frame-averaged
        features = SAXS_curve_input_features(intensities=averaged)
        params = SAXS_Reweighted_Parameters()
        result = SAXS_ReweightedForwardPass()(features, params)
        assert isinstance(result, SAXS_output_features)
        np.testing.assert_allclose(result.y_pred(), averaged)

    def test_output_shape_matches_input(self):
        n_q = 501
        features = SAXS_curve_input_features(intensities=jnp.ones(n_q))
        result = SAXS_ReweightedForwardPass()(features, SAXS_Reweighted_Parameters())
        assert result.output_shape == (n_q,)

    def test_jit_compatible(self):
        features = SAXS_curve_input_features(intensities=jnp.ones(10))
        params = SAXS_Reweighted_Parameters()
        fn = jax.jit(SAXS_ReweightedForwardPass())
        result = fn(features, params)
        assert result.intensity.shape == (10,)


class TestSAXSDebyeForwardPass:
    def test_known_formula(self):
        """With c1=1, c2=0, c=1, b=0: I = Ivv - 2*Ive + Iee."""
        n_q = 3
        # basis_profiles shape (6, n_q) after frame averaging
        # Ivv=1, Ive=0.5, Ivh=0, Iee=0.25, Ieh=0, Ihh=0
        bp = jnp.zeros((6, n_q))
        bp = bp.at[0].set(jnp.ones(n_q))     # Ivv = 1
        bp = bp.at[1].set(jnp.full(n_q, 0.5)) # Ive = 0.5
        bp = bp.at[3].set(jnp.full(n_q, 0.25))# Iee = 0.25
        features = SAXS_basis_input_features(basis_profiles=bp)
        params = SAXS_Debye_Parameters(c1=jnp.array(1.0), c2=jnp.array(0.0),
                                        c=jnp.array(1.0), b=jnp.array(0.0))
        result = SAXS_DebyeForwardPass()(features, params)
        # I_ens = 1 - 2*0.5 + 0 + 0.25 - 0 + 0 = 0.25
        expected = jnp.full(n_q, 0.25)
        np.testing.assert_allclose(result.y_pred(), expected, rtol=1e-5)

    def test_scale_and_background(self):
        """c scales, b shifts."""
        bp = jnp.zeros((6, 3))
        bp = bp.at[0].set(jnp.ones(3))  # Ivv = 1, everything else 0
        features = SAXS_basis_input_features(basis_profiles=bp)
        params = SAXS_Debye_Parameters(c1=jnp.array(0.0), c2=jnp.array(0.0),
                                        c=jnp.array(2.0), b=jnp.array(0.5))
        result = SAXS_DebyeForwardPass()(features, params)
        # I_ens = 1, I_calc = 2*1 + 0.5 = 2.5
        expected = jnp.full(3, 2.5)
        np.testing.assert_allclose(result.y_pred(), expected, rtol=1e-5)

    def test_gradient_flows_through_params(self):
        bp = jnp.zeros((6, 3))
        bp = bp.at[0].set(jnp.ones(3))
        features = SAXS_basis_input_features(basis_profiles=bp)

        def loss(c1):
            params = SAXS_Debye_Parameters(c1=c1, c2=jnp.array(0.0),
                                            c=jnp.array(1.0), b=jnp.array(0.0))
            result = SAXS_DebyeForwardPass()(features, params)
            return result.intensity.sum()
        grad = jax.grad(loss)(jnp.array(1.0))
        assert jnp.isfinite(grad)

    def test_jit_compatible(self):
        bp = jnp.zeros((6, 3))
        bp = bp.at[0].set(jnp.ones(3))
        features = SAXS_basis_input_features(basis_profiles=bp)
        params = SAXS_Debye_Parameters()
        result = jax.jit(SAXS_DebyeForwardPass())(features, params)
        assert result.intensity.shape == (3,)

class TestSAXSConfig:
    def test_reweighted_mode_key(self):
        cfg = SAXS_Config(mode="reweighted")
        assert cfg.key == m_key("SAXS_Iq")

    def test_reweighted_forward_parameters_type(self):
        cfg = SAXS_Config(mode="reweighted")
        assert isinstance(cfg.forward_parameters, SAXS_Reweighted_Parameters)

    def test_debye_forward_parameters_type(self):
        cfg = SAXS_Config(mode="debye_6term")
        assert isinstance(cfg.forward_parameters, SAXS_Debye_Parameters)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SAXS_Config(mode="invalid")

class TestSAXSModel:
    def test_reweighted_has_correct_forward_pass(self):
        model = SAXS_model(SAXS_Config(mode="reweighted"))
        assert m_key("SAXS_Iq") in model.forward
        assert isinstance(model.forward[m_key("SAXS_Iq")], SAXS_ReweightedForwardPass)

    def test_debye_has_correct_forward_pass(self):
        model = SAXS_model(SAXS_Config(mode="debye_6term"))
        assert m_key("SAXS_Iq") in model.forward
        assert isinstance(model.forward[m_key("SAXS_Iq")], SAXS_DebyeForwardPass)

    def test_initialise_returns_true(self):
        model = SAXS_model(SAXS_Config())
        assert model.initialise([]) is True

    def test_featurise_raises_not_implemented(self):
        model = SAXS_model(SAXS_Config())
        with pytest.raises(NotImplementedError):
            model.featurise([])
