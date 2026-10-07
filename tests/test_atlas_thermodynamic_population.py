import numpy as np
import pandas as pd
from scipy.special import logsumexp

from jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 import (
    NOTEBOOK_GAS_CONSTANT_J_MOL_K, entropy_contributions, thermodynamic_pair_features,
)
from jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 import (
    sampled_indices, tune_nonnegative_ridge,
)


def one_pair() -> dict[int, pd.DataFrame]:
    frame = pd.DataFrame({"left_frame": [0], "right_frame": [1]})
    return {1: frame, 2: frame.copy(), 3: frame.copy()}


def test_shape_and_scale_separate_uniform_shift():
    prior = np.array([1.0, 2.0, 4.0])
    shifted = prior + 3.0
    z = np.stack([prior, shifted], axis=1)
    features = thermodynamic_pair_features(z, one_pair())
    assert np.isclose(features["work_shape"][1][0], 0.0)
    assert np.isclose(features["work_scale"][1][0], 3.0)


def test_scale_ignores_pure_shape_change_and_pair_metrics_are_symmetric():
    z = np.array([[0.0, -1.0], [1.0, 1.0], [2.0, 3.0]])
    forward = thermodynamic_pair_features(z, one_pair())
    reverse_pairs = {r: pd.DataFrame({"left_frame": [1], "right_frame": [0]}) for r in (1, 2, 3)}
    reverse = thermodynamic_pair_features(z, reverse_pairs)
    assert np.isclose(forward["work_scale"][1][0], 0.0)
    for metric in forward:
        assert forward[metric][1][0] >= 0
        assert np.isclose(forward[metric][1][0], reverse[metric][1][0])


def test_entropy_variants_match_definitions_and_normalized_probability():
    d = np.array([[0.0], [1.0], [2.0]])
    q = np.exp(-d); z = q.sum(axis=0, keepdims=True)
    variants = {
        "legacy_zq": z * q,
        "unnormalized_q": q,
        "normalized_q_over_z": q / z,
    }
    for name, pi in variants.items():
        expected = -pi * np.log(pi)
        assert np.allclose(entropy_contributions(d, name), expected)
    assert np.allclose(variants["normalized_q_over_z"].sum(axis=0), 1.0)
    assert not np.allclose(variants["legacy_zq"].sum(axis=0), 1.0)


def test_entropy_is_finite_for_extreme_profiles_and_rt_conversion():
    d = np.array([[0.0, 1e4], [1e4, 0.0], [5e3, 5e3]])
    for variant in ("legacy_zq", "unnormalized_q", "normalized_q_over_z"):
        assert np.isfinite(entropy_contributions(d, variant)).all()
    rt_kj = NOTEBOOK_GAS_CONSTANT_J_MOL_K * 300.0 / 1000.0
    assert np.isclose(rt_kj, 2.493)
    log_p = -d - logsumexp(-d, axis=0, keepdims=True)
    assert np.allclose(np.exp(log_p).sum(axis=0), 1.0)


def test_pilot_sampling_is_deterministic_and_ridge_is_nonnegative():
    assert np.array_equal(sampled_indices("system", 1, 100, cap=12),
                          sampled_indices("system", 1, 100, cap=12))
    x = np.array([[0., 1.], [1., 0.], [1., 1.], [2., 1.]])
    y = 2 * x[:, 0] + 3 * x[:, 1]
    model, _, rms = tune_nonnegative_ridge(x, y, x, y)
    assert np.all(model.coef_ >= 0)
    assert np.mean(np.abs(y - model.predict(x / rms))) < 1e-4
