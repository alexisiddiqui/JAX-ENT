from beartype.typing import TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from jaxent.src.custom_types.base import ForwardPass
from jaxent.src.custom_types.features import AbstractFeatures, Input_Features
from jaxent.src.interfaces.model import Model_Parameters

T_Features = TypeVar("T_Features", bound=AbstractFeatures)


def frame_average_features(
    frame_wise_features: T_Features,  # each feature: Float[Array, "... n_frames"]
    frame_weights: Float[Array, " n_frames"],
) -> T_Features:  # each feature: Float[Array, "..."] (frame dim removed)
    """
    Average features or outputs across frames using provided weights by mapping over slots.

    Works on both Input_Features and Output_Features (any AbstractFeatures subclass).

    Args:
        frame_wise_features: Features/outputs for each frame, with frame as last axis
        frame_weights: Weights for each frame (should sum to 1)
    Returns:
        Frame-averaged features/outputs
    """
    weights = frame_weights.reshape(1, -1)

    def average_feature(x):
        x = jnp.asarray(x)
        if x.ndim <= 1:
            return x
        return jnp.sum(x * weights, axis=-1)

    return jax.tree_util.tree_map(average_feature, frame_wise_features)


########################################


from jaxent.src.custom_types.protocols import InputFeaturesLike, ModelParametersLike


def single_pass(
    forward_pass: ForwardPass, input_feature: InputFeaturesLike, parameters: ModelParametersLike
):
    return forward_pass(input_feature, parameters)
