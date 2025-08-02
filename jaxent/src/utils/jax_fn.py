from typing import TypeVar

import jax.numpy as jnp
from jax import Array

from jaxent.src.custom_types.base import ForwardPass
from jaxent.src.custom_types.features import Input_Features
from jaxent.src.interfaces.model import Model_Parameters

T_In = TypeVar("T_In", bound=Input_Features)


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
    frame_wise_features: T_In,  # (frames, residues)
    frame_weights: Array,  # (frames)
) -> T_In:  # (1, residues)
    """
    Average features across frames using provided weights by mapping over slots.

    Args:
        frame_wise_features: Features for each frame and residue
        frame_weights: Weights for each frame (should sum to 1)
    Returns:
        Frame-averaged features
    """
    # Get all slots from the features class
    feature_slots = (
        frame_wise_features.__features__
    )  # Assuming __features__ is the correct way to get dynamic slots

    # Create dict to store averaged features
    averaged_features = {}

    # Average each slotted feature
    for slot in feature_slots:
        feature_array = getattr(frame_wise_features, slot)
        # Ensure feature_array is a JAX array
        feature_array = jnp.asarray(feature_array)

        # Calculate weighted average across frames
        weights = frame_weights.reshape(1, -1)
        averaged = jnp.sum(feature_array * weights, axis=-1)
        averaged_features[slot] = averaged

    # Also copy static slots if any
    all_slots = frame_wise_features.__slots__ if hasattr(frame_wise_features, "__slots__") else []
    static_slots = [slot for slot in all_slots if slot not in feature_slots]
    for slot in static_slots:
        averaged_features[slot] = getattr(frame_wise_features, slot)

    # Create new instance with averaged features
    return type(frame_wise_features)(**averaged_features)


########################################


def single_pass(
    forward_pass: ForwardPass, input_feature: Input_Features, parameters: Model_Parameters
):
    return forward_pass(input_feature, parameters)
