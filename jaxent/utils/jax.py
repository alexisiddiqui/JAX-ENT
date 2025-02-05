from typing import TypeVar

import numpy as np

from jaxent.forwardmodels.base import ForwardPass, Input_Features, Model_Parameters

T_In = TypeVar("T_In", bound=Input_Features)


########################################
# this needs to changed to use a jax map type function
def frame_average_features(
    frame_wise_features: T_In,  # (frames, residues)
    frame_weights: np.ndarray,  # (frames)
) -> T_In:  # (1, residues)
    """
    Average features across frames using provided weights

    Args:
        frame_wise_features: Features for each frame and residue
        frame_weights: Weights for each frame (should sum to 1)

    Returns:
        Frame-averaged features
    """
    # Expand weights for broadcasting
    weights = frame_weights[:, None]  # Shape: (frames, 1)

    # Compute weighted averages
    # Get all fields from the input features slots
    fields = frame_wise_features.__slots__

    # Average each field
    averaged_fields = {}
    for field_name in fields:
        field_data = getattr(frame_wise_features, field_name)
        if isinstance(field_data, np.ndarray):
            averaged_fields[field_name] = np.sum(field_data * weights, axis=0)
        else:
            averaged_fields[field_name] = field_data  # Pass through non-array fields

    # Create new instance with averaged data
    return frame_wise_features.__class__(**averaged_fields)


########################################


def single_pass(
    forward_pass: ForwardPass, input_feature: Input_Features, parameters: Model_Parameters
):
    return forward_pass(input_feature, parameters)
