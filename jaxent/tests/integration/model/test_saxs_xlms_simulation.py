"""Integration tests: SAXS and XLMS models running inside Simulation.forward()."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.models.core import Simulation
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.custom_types.key import m_key
from jaxent.src.models.SAXS.config import SAXS_Config
from jaxent.src.models.SAXS.forwardmodel import SAXS_model
from jaxent.src.models.SAXS.features import SAXS_curve_input_features, SAXS_basis_input_features, SAXS_output_features
from jaxent.src.models.XLMS.config import XLMS_Config
from jaxent.src.models.XLMS.forwardmodel import XLMS_distance_model
from jaxent.src.models.XLMS.features import XLMS_input_features, XLMS_output_features


class TestSAXSReweightedSimulation:
    def test_forward_produces_saxs_output(self):
        n_q, n_frames = 50, 10
        features = [SAXS_curve_input_features(
            intensities=jnp.ones((n_q, n_frames))
        )]
        model = SAXS_model(SAXS_Config(mode="reweighted"))
        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[SAXS_Config().forward_parameters],
            forward_model_weights=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
        )
        sim = Simulation(input_features=features, forward_models=[model], params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)

        assert len(sim.outputs) == 1
        assert isinstance(sim.outputs[0], SAXS_output_features)
        assert sim.outputs[0].intensity.shape == (n_q,)

    def test_outputs_by_key_accessible(self):
        n_q, n_frames = 20, 5
        features = [SAXS_curve_input_features(intensities=jnp.ones((n_q, n_frames)))]
        model = SAXS_model(SAXS_Config())
        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[SAXS_Config().forward_parameters],
            forward_model_weights=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
        )
        sim = Simulation(input_features=features, forward_models=[model], params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)

        obk = sim.outputs_by_key
        assert m_key("SAXS_Iq") in obk
        assert isinstance(obk[m_key("SAXS_Iq")], SAXS_output_features)

    def test_weighted_average_numerically_correct(self):
        """Verify frame weights are applied correctly: 2 frames, known curves."""
        n_q = 3
        curves = jnp.array([[1.0, 3.0], [2.0, 4.0], [5.0, 5.0]])  # (n_q=3, n_frames=2)
        features = [SAXS_curve_input_features(intensities=curves)]
        model = SAXS_model(SAXS_Config())
        # equal weights: expected average = mean along frames axis
        params = Simulation_Parameters(
            frame_weights=jnp.array([0.5, 0.5]),
            frame_mask=jnp.ones(2) * 0.5,
            model_parameters=[SAXS_Config().forward_parameters],
            forward_model_weights=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
        )
        sim = Simulation(input_features=features, forward_models=[model], params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)
        expected = jnp.array([2.0, 3.0, 5.0])
        np.testing.assert_allclose(sim.outputs[0].intensity, expected, rtol=1e-5)


class TestSAXSDebyeSimulation:
    def test_forward_applies_cross_term_formula(self):
        """Debye path: basis profiles averaged, then cross-term applied."""
        n_q, n_frames = 5, 4
        # basis_profiles shape: (6, n_q, n_frames); all Ivv=2, others 0
        bp = jnp.zeros((6, n_q, n_frames))
        bp = bp.at[0].set(jnp.full((n_q, n_frames), 2.0))  # Ivv = 2
        features = [SAXS_basis_input_features(basis_profiles=bp)]
        model = SAXS_model(SAXS_Config(mode="debye_6term"))
        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[SAXS_Config(mode="debye_6term").forward_parameters],
            forward_model_weights=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
        )
        sim = Simulation(input_features=features, forward_models=[model], params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)
        # c1=1, c2=0, c=1, b=0 (defaults): I_ens = Ivv = 2, I_calc = 2
        assert sim.outputs[0].intensity.shape == (n_q,)
        np.testing.assert_allclose(sim.outputs[0].intensity, jnp.full(n_q, 2.0), rtol=1e-5)

    def test_outputs_by_key_accessible_debye(self):
        n_q, n_frames = 8, 3
        bp = jnp.ones((6, n_q, n_frames))
        features = [SAXS_basis_input_features(basis_profiles=bp)]
        model = SAXS_model(SAXS_Config(mode="debye_6term"))
        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[SAXS_Config(mode="debye_6term").forward_parameters],
            forward_model_weights=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
        )
        sim = Simulation(input_features=features, forward_models=[model], params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)
        assert m_key("SAXS_Iq") in sim.outputs_by_key


class TestXLMSSimulation:
    def test_forward_produces_xlms_output(self):
        n_res, n_frames = 8, 10
        dist_data = jnp.ones((n_res, n_res, n_frames))
        features = [XLMS_input_features(distances=dist_data)]
        model = XLMS_distance_model(XLMS_Config())
        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[XLMS_Config().forward_parameters],
            forward_model_weights=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
        )
        sim = Simulation(input_features=features, forward_models=[model], params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)

        assert len(sim.outputs) == 1
        assert isinstance(sim.outputs[0], XLMS_output_features)
        assert sim.outputs[0].distances.shape == (n_res, n_res)

    def test_outputs_by_key_accessible(self):
        n_res, n_frames = 4, 5
        features = [XLMS_input_features(distances=jnp.ones((n_res, n_res, n_frames)))]
        model = XLMS_distance_model(XLMS_Config())
        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[XLMS_Config().forward_parameters],
            forward_model_weights=jnp.ones(1),
            forward_model_scaling=jnp.ones(1),
            normalise_loss_functions=jnp.ones(1),
        )
        sim = Simulation(input_features=features, forward_models=[model], params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)
        obk = sim.outputs_by_key
        assert m_key("XLMS_distance") in obk


class TestMixedModelSimulation:
    """Verify SAXS and XLMS models run alongside HDX in the same Simulation."""

    def test_hdx_plus_saxs_forward(self):
        """HDX BV + SAXS reweighted share frame weights, both outputs accessible."""
        from jaxent.src.models.HDX.BV.forwardmodel import BV_model
        from jaxent.src.models.HDX.BV.features import BV_input_features, BV_output_features
        from jaxent.src.models.config import BV_model_Config

        n_res, n_q, n_frames = 5, 20, 8
        bv_feats = BV_input_features(
            heavy_contacts=jnp.ones((n_res, n_frames)),
            acceptor_contacts=jnp.ones((n_res, n_frames)),
            k_ints=jnp.ones(n_res),
        )
        saxs_feats = SAXS_curve_input_features(intensities=jnp.ones((n_q, n_frames)))

        bv_config = BV_model_Config()
        saxs_config = SAXS_Config()
        models = [BV_model(bv_config), SAXS_model(saxs_config)]
        features = [bv_feats, saxs_feats]

        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[bv_config.forward_parameters, saxs_config.forward_parameters],
            forward_model_weights=jnp.ones(2),
            forward_model_scaling=jnp.ones(2),
            normalise_loss_functions=jnp.ones(2),
        )
        sim = Simulation(input_features=features, forward_models=models, params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)

        assert len(sim.outputs) == 2
        obk = sim.outputs_by_key
        assert m_key("HDX_resPF") in obk
        assert m_key("SAXS_Iq") in obk
        assert isinstance(obk[m_key("HDX_resPF")], BV_output_features)
        assert isinstance(obk[m_key("SAXS_Iq")], SAXS_output_features)

    def test_hdx_plus_xlms_forward(self):
        """HDX BV + XLMS share frame weights, both outputs accessible."""
        from jaxent.src.models.HDX.BV.forwardmodel import BV_model
        from jaxent.src.models.HDX.BV.features import BV_input_features, BV_output_features
        from jaxent.src.models.config import BV_model_Config

        n_res, n_frames = 6, 10
        bv_feats = BV_input_features(
            heavy_contacts=jnp.ones((n_res, n_frames)),
            acceptor_contacts=jnp.ones((n_res, n_frames)),
            k_ints=jnp.ones(n_res),
        )
        xlms_feats = XLMS_input_features(distances=jnp.ones((n_res, n_res, n_frames)))

        bv_config = BV_model_Config()
        xlms_config = XLMS_Config()
        models = [BV_model(bv_config), XLMS_distance_model(xlms_config)]
        features = [bv_feats, xlms_feats]

        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[bv_config.forward_parameters, xlms_config.forward_parameters],
            forward_model_weights=jnp.ones(2),
            forward_model_scaling=jnp.ones(2),
            normalise_loss_functions=jnp.ones(2),
        )
        sim = Simulation(input_features=features, forward_models=models, params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)

        assert len(sim.outputs) == 2
        obk = sim.outputs_by_key
        assert m_key("HDX_resPF") in obk
        assert m_key("XLMS_distance") in obk
        assert isinstance(obk[m_key("HDX_resPF")], BV_output_features)
        assert isinstance(obk[m_key("XLMS_distance")], XLMS_output_features)

    def test_hdx_plus_saxs_plus_xlms_forward(self):
        """Three-model simulation: HDX + SAXS + XLMS all sharing frame weights."""
        from jaxent.src.models.HDX.BV.forwardmodel import BV_model
        from jaxent.src.models.HDX.BV.features import BV_input_features
        from jaxent.src.models.config import BV_model_Config

        n_res, n_q, n_frames = 4, 15, 6
        bv_feats = BV_input_features(
            heavy_contacts=jnp.ones((n_res, n_frames)),
            acceptor_contacts=jnp.ones((n_res, n_frames)),
            k_ints=jnp.ones(n_res),
        )
        saxs_feats = SAXS_curve_input_features(intensities=jnp.ones((n_q, n_frames)))
        xlms_feats = XLMS_input_features(distances=jnp.ones((n_res, n_res, n_frames)))

        bv_config = BV_model_Config()
        saxs_config = SAXS_Config()
        xlms_config = XLMS_Config()
        models = [BV_model(bv_config), SAXS_model(saxs_config), XLMS_distance_model(xlms_config)]
        features_list = [bv_feats, saxs_feats, xlms_feats]

        params = Simulation_Parameters(
            frame_weights=jnp.ones(n_frames) / n_frames,
            frame_mask=jnp.ones(n_frames) * 0.5,
            model_parameters=[bv_config.forward_parameters, saxs_config.forward_parameters, xlms_config.forward_parameters],
            forward_model_weights=jnp.ones(3),
            forward_model_scaling=jnp.ones(3),
            normalise_loss_functions=jnp.ones(3),
        )
        sim = Simulation(input_features=features_list, forward_models=models, params=params)
        sim.initialise()
        sim = Simulation.forward(sim, params)

        assert len(sim.outputs) == 3
        obk = sim.outputs_by_key
        assert m_key("HDX_resPF") in obk
        assert m_key("SAXS_Iq") in obk
        assert m_key("XLMS_distance") in obk
