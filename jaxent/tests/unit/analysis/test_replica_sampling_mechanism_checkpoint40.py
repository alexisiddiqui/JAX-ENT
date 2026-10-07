import numpy as np
import pandas as pd
import pytest

from jaxent.examples.ATLAS_BV.analysis.replica_sampling_mechanism_checkpoint40 import (
    assignment_outcomes,
    batch_means_neff,
    fixed_js,
    local_estimates,
    neff_covariates,
    profile_js,
    summarize_transfer,
    trace_neff,
)


def test_fixed_js_is_zero_for_identical_and_positive_for_shifted_samples():
    reference = np.linspace(-2.0, 2.0, 200)
    assert fixed_js(reference, reference, reference) == pytest.approx(0.0)
    assert fixed_js(reference - 1.0, reference + 1.0, reference) > 0.1


def test_profile_js_aggregates_components():
    reference = np.vstack([np.linspace(0, 1, 100), np.linspace(-1, 1, 100)])
    mean, median = profile_js(reference, reference, reference)
    assert mean == pytest.approx(0.0)
    assert median == pytest.approx(0.0)


def test_autocorrelation_reduces_effective_sample_size():
    rng = np.random.default_rng(7)
    iid = rng.normal(size=512)
    correlated = np.empty(512)
    correlated[0] = iid[0]
    for index in range(1, len(correlated)):
        correlated[index] = 0.95 * correlated[index - 1] + iid[index]
    _, iid_neff, _ = trace_neff(iid)
    _, correlated_neff, _ = trace_neff(correlated)
    assert correlated_neff < iid_neff
    assert 1 <= batch_means_neff(correlated, 16) <= len(correlated)


def test_local_estimates_leave_out_landmark_and_normalize():
    coordinate = np.arange(5, dtype=float)
    distance = np.abs(coordinate[:, None] - coordinate[None, :])
    targets, weights, support = local_estimates(
        distance,
        np.array([0, 2]),
        np.arange(5),
        bandwidth=1.0,
        leave_one_out=True,
    )
    assert weights[0, 0] == 0.0
    assert weights[1, 2] == 0.0
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert np.isfinite(targets).all()
    assert (support >= 1).all()


def test_assignment_outcomes_uses_32_labels_and_auc():
    rows = []
    for system in ("a", "b"):
        for metric in ("pf_l1", "work_opt"):
            for known in (2, 4, 8, 16, 32):
                for mode, rho in (("within_C", 0.6), ("AB_to_C", -0.2)):
                    rows.append(
                        {
                            "system_id": system,
                            "metric": metric,
                            "model": "direct",
                            "known": known,
                            "mode": mode,
                            "selection": "random",
                            "stratum": "all",
                            "method": "sparse_alpha",
                            "spearman": rho,
                        }
                    )
    result = assignment_outcomes(pd.DataFrame(rows))
    assert len(result) == 4
    assert set(result.known) == {32}
    np.testing.assert_allclose(result.assignment_drop, 0.8)
    np.testing.assert_allclose(result.sign_flip_magnitude, 0.2)
    np.testing.assert_allclose(result.assignment_drop_auc, 0.8)


def test_neff_covariates_sum_source_and_measure_target_imbalance():
    frame = pd.DataFrame(
        {
            "system_id": ["a"] * 3,
            "trace": ["global_pf"] * 3,
            "replica": [1, 2, 3],
            "neff": [20.0, 45.0, 30.0],
        }
    )
    result = neff_covariates(frame).iloc[0]
    assert result.neff_AB == pytest.approx(65.0)
    assert result.neff_min == pytest.approx(30.0)
    assert result.neff_log_imbalance == pytest.approx(
        abs(np.log(30.0 / np.sqrt(20.0 * 45.0)))
    )


def test_pool_rescue_requires_positive_interval_and_nonnegative_rho():
    rows = []
    for system in ("a", "b", "c"):
        for mode, rho in (
            ("original_AB_to_C", -0.4),
            ("AB_common", -0.1),
            ("ABC_common", 0.2),
        ):
            rows.append(
                {
                    "system_id": system,
                    "metric": "pf_l1",
                    "model": "direct",
                    "known": 32,
                    "mode": mode,
                    "selection": "random",
                    "stratum": "all",
                    "method": "sparse_alpha",
                    "spearman": rho,
                    "distribution_recovery": 0.7,
                }
            )
    # Include the other primary metric so the summarizer's declared scope is complete.
    work = pd.DataFrame(rows).assign(metric="work_opt")
    result = summarize_transfer(pd.concat([pd.DataFrame(rows), work]), "pool", 1000)
    abc = result[
        (result.test == "ABC_common") & (result.reference == "original_AB_to_C")
    ]
    assert abc.rescue.all()
    assert not abc.strong_rescue.any()
