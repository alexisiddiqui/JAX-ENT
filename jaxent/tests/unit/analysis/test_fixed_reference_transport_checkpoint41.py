import numpy as np
import pandas as pd
import pytest

from jaxent.examples.ATLAS_BV.analysis import (
    fixed_reference_transport_checkpoint41 as cp41,
)
from jaxent.examples.ATLAS_BV.analysis import (
    replica_sampling_mechanism_checkpoint40 as cp40,
)
from jaxent.examples.ATLAS_BV.analysis import (
    sparse_population_extrapolation_checkpoint39 as cp39,
)


def test_pf_fixed_origin_is_exact_l1_translation_control():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(5, 7))
    target = rng.normal(size=(5, 7))
    reference = rng.normal(size=5)[:, None]
    raw_fit = cp39.pair_distance(source)
    raw_cross = cp39.pair_distance(target, source)
    fixed_fit = cp39.pair_distance(source - reference)
    fixed_cross = cp39.pair_distance(target - reference, source - reference)
    np.testing.assert_allclose(raw_fit, fixed_fit, atol=1e-14)
    np.testing.assert_allclose(raw_cross, fixed_cross, atol=1e-14)
    values = rng.normal(size=7)
    labels = np.array([0, 2, 5])
    raw_prediction = cp39.anchored_prediction(
        raw_fit, raw_cross, values, labels
    )
    fixed_prediction = cp39.anchored_prediction(
        fixed_fit, fixed_cross, values, labels
    )
    np.testing.assert_allclose(raw_prediction[0], fixed_prediction[0], atol=1e-13)
    assert raw_prediction[1:] == pytest.approx(fixed_prediction[1:])


def test_fixed_work_opt_uses_scalar_or_profile_reference():
    z = np.array([[1.0, 2.0], [4.0, 8.0], [2.0, 3.0]])
    scalar = cp41.fixed_work_opt(z, 2.0)
    profile = cp41.fixed_work_opt(z, np.array([1.0, 4.0, 2.0]))
    assert scalar.shape == z.shape
    assert profile.shape == z.shape
    assert np.isfinite(scalar).all()
    assert np.isfinite(profile).all()
    assert not np.allclose(scalar, profile)


def test_pool_then_transform_is_not_transform_then_pool():
    z = np.array([[0.2, 1.0, 3.0], [1.5, 0.4, 2.0], [2.1, 1.8, 0.3]])
    weights = np.array([[0.2, 0.3, 0.5]])
    indices = np.arange(3)
    z0 = np.array([0.5, 1.0, 1.5])
    bundle = cp41.descriptor_bundle(z, indices, weights, z0)
    assert not np.allclose(
        bundle["work_opt_fixed_mean_frame_then_pool"],
        bundle[cp41.WORK_PRIMARY],
    )


def test_target_self_keeps_source_fit_but_uses_target_query_geometry():
    source = {name: np.array([[0.0, 1.0, 3.0]]) for name in cp41.REPRESENTATIONS}
    target = {name: np.array([[0.0, 2.0, 6.0]]) for name in cp41.REPRESENTATIONS}
    matrices = cp41.distance_modes(source, target, source, target)
    fit, query = matrices["target_self"][cp41.PF_RAW]
    np.testing.assert_allclose(fit, cp39.pair_distance(source[cp41.PF_RAW]))
    np.testing.assert_allclose(query, cp39.pair_distance(target[cp41.PF_RAW]))
    assert not np.allclose(query, matrices["cross"][cp41.PF_RAW][1])


def test_pooled_variance_is_recomputed_not_average_of_replica_coordinates():
    values = np.array([[0.0, 0.2, 0.4, 4.0, 4.2, 4.4]])
    structural = np.abs(np.arange(6)[:, None] - np.arange(6)[None, :]).astype(float)
    pooled, _ = cp40.variance_coordinate(
        values, structural, np.arange(6), k=2, shrinkage=0.1
    )
    left, reference = cp40.variance_coordinate(
        values, structural, np.arange(3), k=2, shrinkage=0.1
    )
    right, _ = cp40.variance_coordinate(
        values, structural, np.arange(3, 6), k=2, shrinkage=0.1, reference=reference
    )
    assert not np.allclose(pooled, np.concatenate((left, right)))


def test_mechanism_decision_uses_both_improvement_controls():
    summary = pd.DataFrame(
        [
            {
                "transfer": "AB_to_C",
                "known": 32,
                "representation": cp41.WORK_PRIMARY,
                "transport": "cross",
                "median_spearman": 0.6,
            }
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "transfer": "AB_to_C",
                "known": 32,
                "comparison": "fixed_reference_increment_same_order",
                "delta_ci_low": 0.1,
            },
            {
                "transfer": "AB_to_C",
                "known": 32,
                "comparison": "pool_then_transform_vs_frame_then_pool",
                "delta_ci_low": 0.1,
            },
            {
                "transfer": "AB_to_C",
                "known": 32,
                "comparison": "primary_fixed_vs_source_common",
                "delta_ci_low": 0.05,
            },
            *[
                {
                    "transfer": "AB_to_C",
                    "known": 32,
                    "comparison": f"target_self_vs_cross:{representation}",
                    "delta_ci_low": 0.01,
                }
                for representation in (cp41.PF_RAW, cp41.WORK_BASE)
            ],
        ]
    )
    result = cp41.mechanism_decisions(summary, contrasts, 32)
    work = result[result.representation == cp41.WORK_BASE].iloc[0]
    assert work.external_reference_result == "strong_support"
    assert bool(work.target_self_supported)
