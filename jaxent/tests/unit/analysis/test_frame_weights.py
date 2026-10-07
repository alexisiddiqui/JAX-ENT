import numpy as np
import pytest

from jaxent.src.analysis.frame_weights import (
    validated_frame_weight_simplex,
    validated_frame_weight_simplex_rows,
)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [[0.5, 0.5]],
        [np.nan, 1.0],
        [np.inf, 0.0],
        [-0.1, 1.1],
        [0.2, 0.2],
    ],
)
def test_invalid_simplex_is_rejected(values):
    with pytest.raises(ValueError):
        validated_frame_weight_simplex(values)


def test_tolerated_drift_is_clipped_and_renormalized():
    result = validated_frame_weight_simplex([-1e-6, 1.000001])
    np.testing.assert_allclose(result, [0.0, 1.0])


def test_rows_validate_independently():
    result = validated_frame_weight_simplex_rows([[0.25, 0.75], [0.5, 0.5]])
    np.testing.assert_allclose(result.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="row 1"):
        validated_frame_weight_simplex_rows([[0.25, 0.75], [0.2, 0.2]])
