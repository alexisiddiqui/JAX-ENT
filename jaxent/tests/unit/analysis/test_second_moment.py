import numpy as np

from jaxent.src.analysis.second_moment import (
    decompose_envelope,
    precision_band_decision,
)


def test_shared_second_moment_decomposition_is_invariant_on_fixture():
    full = np.asarray([0.48, 0.31, 0.14, 0.05, 0.02])
    observed = np.asarray([0.22, 0.36, 0.25, 0.12, 0.05])

    result = decompose_envelope(full, observed, n_bins=5)

    assert result["best_shift_bins"] == np.float64(
        0.6462731569229176
    )
    assert result["centroid_aligned_width_ratio"] == np.float64(
        0.8550263275654643
    )
    assert result["centroid_explained_fraction"] == np.float64(
        0.9227764089488514
    )

    decision = precision_band_decision(
        result["centroid_aligned_width_ratio"], 1.05, 1.2
    )
    assert decision == {
        "precision_band_lower": 0.8999999999999999,
        "precision_band_upper": 1.5,
        "separation_survives": True,
        "detected_excess_width": (
            0.8999999999999999
            - result["centroid_aligned_width_ratio"]
        ),
    }
