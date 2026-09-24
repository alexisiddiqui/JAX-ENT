import numpy as np

from jaxent.src.analysis.PCA.core import perform_pca_on_distances


def test_incremental_pca_transforms_all_batches_with_final_model():
    rng = np.random.default_rng(42)
    distances = rng.normal(size=(12, 5)).astype(np.float32)
    # Make later batches materially alter the fitted mean and components.
    distances[4:8] += np.asarray([8.0, -3.0, 1.0, 5.0, -6.0])
    distances[8:] += np.asarray([-4.0, 7.0, 3.0, -2.0, 5.0])

    coords, _variance, fitted_pca = perform_pca_on_distances(
        distances, n_components=2, chunk_size=4
    )

    expected = fitted_pca.transform(distances)
    np.testing.assert_allclose(coords, expected, rtol=1e-6, atol=1e-6)
