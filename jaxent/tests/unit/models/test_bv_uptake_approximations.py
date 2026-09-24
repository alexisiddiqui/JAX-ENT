import jax
import jax.numpy as jnp
import h5py
import numpy as np
import pytest

from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BVRateDistributionConfig, linear_BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import (
    BVRateDistributionModel,
    linear_BV_model,
)
from jaxent.src.models.HDX.BV.parameters import (
    BVRateDistributionParameters,
    linear_BV_Model_Parameters,
)
from jaxent.src.models.HDX.forward import (
    BVRateDistributionForwardPass,
    linear_BV_ForwardPass,
)
from jaxent.src.utils.hdf import (
    load_model_parameters_from_hdf5,
    save_model_parameters_to_hdf5,
)


@pytest.fixture
def frame_features():
    return BV_input_features(
        heavy_contacts=jnp.array([[0.0, 2.0, 5.0], [1.0, 1.0, 1.0]]),
        acceptor_contacts=jnp.array([[0.0, 1.0, 2.0], [0.5, 0.5, 0.5]]),
        k_ints=jnp.array([0.02, 0.01]),
    )


def test_linear_model_is_monotone_bounded_and_matches_average_contact_ex2(
    frame_features,
):
    weights = jnp.array([0.2, 0.3, 0.5])
    params = linear_BV_Model_Parameters(
        bv_bc=0.35,
        bv_bh=2.0,
        interval_offsets=jnp.zeros(4),
        timepoints=(0.1, 1.0, 10.0, 100.0),
        kint_unit="s^-1",
        time_unit="min",
    )
    uptake = (
        linear_BV_ForwardPass().average_frames(frame_features, params, weights).uptake
    )
    mean_z = 0.35 * (frame_features.heavy_contacts @ weights) + 2.0 * (
        frame_features.acceptor_contacts @ weights
    )
    expected = -jnp.expm1(
        -60.0
        * jnp.asarray(params.timepoints)[:, None]
        * frame_features.k_ints[None, :]
        * jnp.exp(-mean_z)[None, :]
    )
    np.testing.assert_allclose(uptake, expected, rtol=2e-6, atol=2e-7)
    assert jnp.all(jnp.diff(uptake, axis=0) >= 0)
    assert jnp.all((uptake >= 0) & (uptake <= 1))


def test_linear_interval_offsets_are_positive_conditional_increments():
    features = BV_input_features(jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.01]))
    params = linear_BV_Model_Parameters(
        bv_bc=0.35,
        bv_bh=2.0,
        interval_offsets=jnp.array([-2.0, 0.0, 2.0]),
        timepoints=(1.0, 2.0, 3.0),
        kint_unit="min^-1",
        time_unit="min",
    )
    uptake = linear_BV_ForwardPass()(features, params).uptake[:, 0]
    assert jnp.all(jnp.diff(uptake) > 0)
    assert jnp.all(
        jnp.isfinite(
            jax.grad(
                lambda offsets: jnp.sum(
                    linear_BV_ForwardPass()(
                        features,
                        linear_BV_Model_Parameters(
                            raw_bv_bc=params.raw_bv_bc,
                            raw_bv_bh=params.raw_bv_bh,
                            interval_offsets=offsets,
                            timepoints=params.timepoints,
                            kint_unit=params.kint_unit,
                            time_unit=params.time_unit,
                        ),
                    ).uptake
                )
            )(params.interval_offsets)
        )
    )


@pytest.mark.parametrize("n_components", [2, 4, 8])
def test_soft_mixture_supports_are_ordered_and_output_is_physical(
    frame_features, n_components
):
    params = BVRateDistributionParameters.from_features(
        frame_features.heavy_contacts,
        frame_features.acceptor_contacts,
        n_components=n_components,
        timepoints=(0.1, 1.0, 10.0),
    )
    uptake = (
        BVRateDistributionForwardPass()
        .average_frames(frame_features, params, jnp.array([0.2, 0.3, 0.5]))
        .uptake
    )
    assert params.support_points.shape == (n_components,)
    assert jnp.all(jnp.diff(params.support_points) > 0)
    assert jnp.all(jnp.diff(uptake, axis=0) >= -1e-7)
    assert jnp.all((uptake >= 0) & (uptake <= 1))


def test_soft_mixture_masses_preserve_framewise_population_behavior():
    features = BV_input_features(
        heavy_contacts=jnp.array([[0.0, 10.0]]),
        acceptor_contacts=jnp.zeros((1, 2)),
        k_ints=jnp.array([1.0]),
    )
    params = BVRateDistributionParameters(
        backend="soft_mixture",
        bv_bc=1.0,
        bv_bh=1e-6,
        support_points=jnp.array([0.0, 10.0]),
        bandwidth_floor=0.01,
        timepoints=(1.0,),
        kint_unit="min^-1",
        time_unit="min",
    )
    forward = BVRateDistributionForwardPass()
    state_zero = forward.average_frames(features, params, jnp.array([1.0, 0.0])).uptake[
        0, 0
    ]
    state_one = forward.average_frames(features, params, jnp.array([0.0, 1.0])).uptake[
        0, 0
    ]
    mixed = forward.average_frames(features, params, jnp.array([0.3, 0.7])).uptake[0, 0]
    # Assignments may be soft, but their masses (and therefore uptake) remain
    # exactly linear in the frame populations.
    np.testing.assert_allclose(mixed, 0.3 * state_zero + 0.7 * state_one, atol=2e-6)
    assert state_zero > state_one


def test_soft_mixture_recovers_a_known_two_state_population_from_exact_uptake():
    features = BV_input_features(
        heavy_contacts=jnp.array([[0.0, 10.0], [0.0, 10.0]]),
        acceptor_contacts=jnp.zeros((2, 2)),
        k_ints=jnp.array([0.2, 0.05]),
    )
    times = (0.1, 1.0, 10.0, 100.0)
    true_population = 0.37
    params = BVRateDistributionParameters(
        backend="soft_mixture",
        bv_bc=1.0,
        bv_bh=1e-6,
        # Paired nearby supports make Q=4 a sharp two-state basis while
        # retaining the prescribed half-median-gap bandwidth rule.
        support_points=[0.0, 0.001, 9.999, 10.0],
        bandwidth_floor=1e-4,
        timepoints=times,
        kint_unit="min^-1",
        time_unit="min",
    )
    forward = BVRateDistributionForwardPass()
    exact_per_frame = forward(features, params).uptake
    target = (
        true_population * exact_per_frame[..., 0]
        + (1.0 - true_population) * exact_per_frame[..., 1]
    )
    endpoint_zero = forward.average_frames(
        features, params, jnp.array([1.0, 0.0])
    ).uptake
    endpoint_one = forward.average_frames(
        features, params, jnp.array([0.0, 1.0])
    ).uptake
    direction = endpoint_zero - endpoint_one
    recovered = jnp.sum((target - endpoint_one) * direction) / jnp.sum(direction**2)
    np.testing.assert_allclose(recovered, true_population, atol=2e-3)


def test_gamma_zero_variance_limit_is_single_exponential():
    features = BV_input_features(
        heavy_contacts=jnp.ones((2, 3)),
        acceptor_contacts=jnp.full((2, 3), 0.5),
        k_ints=jnp.array([0.02, 0.01]),
    )
    params = BVRateDistributionParameters(
        backend="gamma_moments",
        bv_bc=0.35,
        bv_bh=2.0,
        timepoints=(0.1, 1.0, 10.0),
        kint_unit="min^-1",
        time_unit="min",
    )
    uptake = (
        BVRateDistributionForwardPass()
        .average_frames(features, params, jnp.array([0.2, 0.3, 0.5]))
        .uptake
    )
    rate = features.k_ints * jnp.exp(-(0.35 + 1.0))
    expected = -jnp.expm1(-jnp.asarray(params.timepoints)[:, None] * rate[None, :])
    np.testing.assert_allclose(uptake, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("backend", ["soft_mixture", "gamma_moments"])
def test_rate_backends_are_jittable_and_have_finite_gradients(frame_features, backend):
    params = BVRateDistributionParameters(
        backend=backend,
        bv_bc=0.35,
        bv_bh=2.0,
        support_points=[0.0, 2.0, 5.0, 9.0] if backend == "soft_mixture" else None,
        timepoints=(0.1, 1.0, 10.0),
    )
    weights = jnp.array([0.2, 0.3, 0.5])
    forward = BVRateDistributionForwardPass()
    objective = jax.jit(
        lambda p: jnp.sum(forward.average_frames(frame_features, p, weights).uptake)
    )
    value = objective(params)
    gradients = jax.grad(objective)(params)
    assert jnp.isfinite(value)
    assert all(
        jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(gradients)
    )


@pytest.mark.parametrize(
    ("config", "model_type"),
    [
        (linear_BV_model_Config(timepoints=jnp.array([0.1, 1.0])), linear_BV_model),
        (
            BVRateDistributionConfig(
                backend="gamma_moments", timepoints=jnp.array([0.1, 1.0])
            ),
            BVRateDistributionModel,
        ),
    ],
)
def test_simulation_dispatches_specialized_frame_averaging(
    frame_features, config, model_type
):
    model = model_type(config)
    params = Simulation_Parameters.from_frame_weights(
        jnp.array([0.2, 0.3, 0.5]),
        model_parameters=[model.params],
        forward_model_weights=jnp.ones(1),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )
    output = Simulation.forward_pure(params, [frame_features], [model.forwardpass])[0]
    assert output.key == m_key("HDX_peptide")
    assert output.uptake.shape == (2, 2)


def test_rate_distribution_hdf_roundtrip_preserves_backend_grid_and_units(tmp_path):
    parameters = BVRateDistributionParameters(
        backend="soft_mixture",
        support_points=[0.0, 2.0, 5.0, 9.0],
        timepoints=(10.0, 60.0),
        kint_unit="min^-1",
        time_unit="s",
    )
    path = tmp_path / "parameters.h5"
    with h5py.File(path, "w") as handle:
        save_model_parameters_to_hdf5(handle, "model", parameters)
    with h5py.File(path, "r") as handle:
        restored = load_model_parameters_from_hdf5(handle, "model")
    assert restored.backend == "soft_mixture"
    assert restored.timepoints == (10.0, 60.0)
    assert restored.kint_unit == "min^-1"
    assert restored.time_unit == "s"
    np.testing.assert_allclose(restored.support_points, parameters.support_points)
