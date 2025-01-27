import numpy as np

from jaxent.forwardmodels.base import Input_Features


def frame_average_features(
    frame_wise_features: Input_Features,  # (frames, residues)
    frame_weights: list,
) -> Input_Features:  # (1, residues)
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
    # Get all fields from the input features
    fields = frame_wise_features.__dataclass_fields__

    # Average each field
    averaged_fields = {}
    for field_name in fields:
        field_data = getattr(frame_wise_features, field_name)
        if isinstance(field_data, np.ndarray):
            averaged_fields[field_name] = np.sum(field_data * weights, axis=0)
        else:
            averaged_fields[field_name] = field_data  # Pass through non-array fields

    # Create new instance with averaged data
    return type(frame_wise_features)(**averaged_fields)
