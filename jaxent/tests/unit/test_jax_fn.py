import jax
import jax.numpy as jnp
import numpy as np

from jaxent.src.models.HDX.BV.features import BV_input_features, uptake_BV_output_features
from jaxent.src.utils.jax_fn import frame_average_features


def _reference_frame_average(features, frame_weights):
    weights = frame_weights.reshape(1, -1)

    def average_feature(x):
        x = jnp.asarray(x)
        if x.ndim <= 1:
            return x
        return jnp.sum(x * weights, axis=-1)

    return jax.tree_util.tree_map(average_feature, features)


def test_frame_average_features_matches_reference_for_nested_float32_features():
    frame_weights = jnp.asarray([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32)
    features = BV_input_features(
        heavy_contacts=jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4) / 7,
        acceptor_contacts=jnp.arange(3 * 4, dtype=jnp.float32).reshape(3, 4) / 11,
        k_ints=jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
    )

    actual = frame_average_features(features, frame_weights)
    expected = _reference_frame_average(features, frame_weights)

    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=1e-5, atol=1e-6)

    timepoint_features = uptake_BV_output_features(
        uptake=jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4) / 13
    )
    actual_timepoint = frame_average_features(timepoint_features, frame_weights)
    expected_timepoint = _reference_frame_average(timepoint_features, frame_weights)
    np.testing.assert_allclose(
        actual_timepoint.uptake, expected_timepoint.uptake, rtol=1e-5, atol=1e-6
    )


def test_frame_average_features_frame_weight_gradient_matches_reference():
    features = jnp.asarray(
        [
            [0.1, 1.2, -0.3, 2.4],
            [1.5, -0.7, 0.8, 0.2],
            [-2.0, 0.4, 1.1, 0.9],
        ],
        dtype=jnp.float32,
    )
    frame_weights = jnp.asarray([0.2, 0.1, 0.3, 0.4], dtype=jnp.float32)

    def loss(weights):
        averaged = frame_average_features(
            BV_input_features(heavy_contacts=features, acceptor_contacts=features), weights
        ).heavy_contacts
        return jnp.sum(averaged**2)

    def reference_loss(weights):
        averaged = _reference_frame_average(features, weights)
        return jnp.sum(averaged**2)

    actual_gradient = jax.grad(loss)(frame_weights)
    expected_gradient = jax.grad(reference_loss)(frame_weights)

    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=5e-6)


def test_frame_average_features_preserves_already_averaged_leaves():
    one_dimensional = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    frame_weights = jnp.asarray([0.25, 0.75], dtype=jnp.float32)

    features = BV_input_features(
        heavy_contacts=one_dimensional,
        acceptor_contacts=one_dimensional + 1.0,
        k_ints=one_dimensional + 2.0,
    )
    actual = frame_average_features(features, frame_weights)

    np.testing.assert_array_equal(actual.heavy_contacts, one_dimensional)
    np.testing.assert_array_equal(actual.acceptor_contacts, one_dimensional + 1.0)
    np.testing.assert_array_equal(actual.k_ints, one_dimensional + 2.0)
