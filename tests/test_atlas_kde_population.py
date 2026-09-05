import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 import (
    frame_w1_signatures, global_scales, log_kernel_density, neighbour_bandwidth, scalar_scale,
)


def test_w1_signature_and_density_are_finite():
    coordinates = np.array([[[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]],
                            [[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]],
                            [[0., 0., 0.], [3., 0., 0.], [6., 0., 0.]]])
    signatures = frame_w1_signatures(coordinates, count=5)
    matrix = np.mean(np.abs(signatures[:, None] - signatures[None, :]), axis=2)
    bandwidth = neighbour_bandwidth(matrix, 1)
    assert signatures.shape == (3, 5)
    assert bandwidth > 0
    assert np.isfinite(log_kernel_density(matrix, bandwidth)).all()


def test_scalar_scale_signed_and_nonnegative():
    x = np.array([-2., -1., 1., 2.]); y = 3.5 * x
    assert np.isclose(scalar_scale(x, y, False)[0], 3.5)
    assert scalar_scale(x, -y, True)[0] == 0.0


def test_global_scale_excludes_held_out_system_and_equal_weights():
    rows = []
    for system, slope in (("a", 1.), ("b", 3.), ("c", 8.)):
        rows.append({"system_id": system, "fit_replica": 1, "rank": 10, "model": "absolute_l1",
                     "numerator": slope * 2., "denominator": 2.})
    result = global_scales(pd.DataFrame(rows)).set_index("system_id").global_alpha_loso
    assert np.isclose(result["a"], 5.5)
    assert np.isclose(result["b"], 4.5)
    assert np.isclose(result["c"], 2.0)
