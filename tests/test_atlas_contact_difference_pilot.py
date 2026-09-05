import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.contact_difference_pilot_checkpoint20 import (
    channel_change_scales, regions, safe_distance, tune_ridge,
)


def test_contact_channel_scales_use_training_pairs():
    heavy = np.array([[0., 1., 3.], [0., 1., 3.]])
    acceptor = np.array([[0., 2., 6.], [0., 2., 6.]])
    pairs = pd.DataFrame({"left_frame": [0, 1], "right_frame": [1, 2]})
    heavy_scale, acceptor_scale = channel_change_scales(heavy, acceptor, pairs, np.array([0]))
    assert np.isclose(heavy_scale, 1.0)
    assert np.isclose(acceptor_scale, 2.0)


def test_constant_profile_distances_are_finite():
    profile = np.ones((4, 2))
    for metric in ("cosine", "correlation"):
        result = safe_distance(profile, np.array([0]), np.array([1]), metric)
        assert np.isfinite(result).all()


def test_tail_regions_match_global_edges():
    edges = np.arange(7, dtype=float)
    values = np.array([.5, 4.5, 5.5])
    masks = regions(values, edges)
    assert masks["q4"].tolist() == [False, True, False]
    assert masks["q5"].tolist() == [False, False, True]
    assert masks["q4_q5"].tolist() == [False, True, True]


def test_contact_ridge_preserves_nonnegative_zero_origin():
    x = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    y = 2*x[:, 0] + 3*x[:, 1]
    model, _, rms = tune_ridge(x, y, x, y, positive=True)
    assert np.all(model.coef_ >= 0)
    assert np.isclose(model.predict(np.zeros((1, 2)) / rms)[0], 0.0)
