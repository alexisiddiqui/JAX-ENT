from typing import TypeVar

from jax import Array

from jaxent.forwardmodels.base import ForwardPass, Input_Features, Model_Parameters

T_In = TypeVar("T_In", bound=Input_Features)

import jax.numpy as jnp

# def frame_average_features(
#     frame_wise_features: T_In,  # (frames, residues)
#     frame_weights: Array,  # (frames)
# ) -> T_In:  # (1, residues)
#     """
#     Average features across frames using provided weights by mapping over slots.

#     Args:
#         frame_wise_features: Features for each frame and residue
#         frame_weights: Weights for each frame (should sum to 1)
#     Returns:
#         Frame-averaged features
#     """
#     # Get all slots from the features class
#     feature_slots = frame_wise_features.__slots__

#     # Create dict to store averaged features
#     averaged_features = {}

#     # Average each slotted feature
#     for slot in feature_slots:
#         feature_array = getattr(frame_wise_features, slot)
#         # Ensure feature_array is a JAX array
#         feature_array = jnp.asarray(feature_array)
#         print(feature_array.shape)

#         # Calculate weighted average across frames
#         # Expand weights to match feature dimensions if needed
#         weights = frame_weights.reshape(-1, *([1] * (feature_array.ndim - 1)))
#         averaged = jnp.sum(feature_array * weights, axis=0, keepdims=True)
#         print(averaged.shape)
#         averaged_features[slot] = averaged

#     # Create new instance with averaged features
#     return type(frame_wise_features)(**averaged_features)


def frame_average_features(
    frame_wise_features: T_In,
    frame_weights: Array,
) -> T_In:
    """Average features across frames using provided weights."""
    feature_slots = frame_wise_features.__slots__
    averaged_features = {}

    for slot in feature_slots:
        feature_array = jnp.asarray(getattr(frame_wise_features, slot))  # (53, 500)
        # print(feature_array.shape)
        weights = frame_weights.reshape(1, -1)  # (1, 500)
        # print(weights)
        averaged = jnp.matmul(feature_array, weights.T)  # (53, 1)
        # print(averaged.shape)
        averaged_features[slot] = averaged

    return type(frame_wise_features)(**averaged_features)


########################################


def single_pass(
    forward_pass: ForwardPass, input_feature: Input_Features, parameters: Model_Parameters
):
    return forward_pass(input_feature, parameters)
