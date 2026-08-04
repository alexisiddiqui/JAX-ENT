from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from jaxent.examples.common.analysis.frame_averaging import (
    effective_rates,
    residue_uptake_fast,
    residue_uptake_legacy,
    residue_uptake_slow2,
    weights_from_cluster_populations,
)


def inputs():
    log_pf = np.asarray([[0.0, 1.0, 2.0, 3.0], [0.2, 0.4, 0.9, 1.5]])
    k_ints = np.asarray([2.0, 0.7])
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    assignments = np.asarray([0, 0, 1, 1])
    return log_pf, k_ints, weights, assignments


def test_fast_rate_is_at_least_legacy_rate_and_strict_with_variance():
    log_pf, k_ints, weights, _ = inputs()
    time = np.asarray([1.0])
    legacy = residue_uptake_legacy(log_pf, k_ints, time, weights)
    fast = residue_uptake_fast(log_pf, k_ints, time, weights)
    legacy_rate = -np.log1p(-legacy[0])
    fast_rate = -np.log1p(-fast[0])
    assert np.all(fast_rate >= legacy_rate)
    assert np.all(fast_rate > legacy_rate)


def test_short_time_asymptote_exposes_legacy_am_gm_gap():
    log_pf, k_ints, weights, assignments = inputs()
    # Large enough to avoid float32 cancellation in the production JAX helper,
    # small enough that the first-order term dominates.
    time = np.asarray([1e-5])
    rates = effective_rates(log_pf, k_ints)
    arithmetic = rates @ weights
    geometric = k_ints * np.exp(-(log_pf @ weights))
    assert np.all(arithmetic > geometric)
    np.testing.assert_allclose(residue_uptake_legacy(log_pf, k_ints, time, weights)[0] / time[0], geometric, rtol=2e-2)
    np.testing.assert_allclose(residue_uptake_fast(log_pf, k_ints, time, weights)[0] / time[0], arithmetic, rtol=2e-5)
    np.testing.assert_allclose(residue_uptake_slow2(log_pf, k_ints, time, weights, assignments)[0] / time[0], arithmetic, rtol=2e-5)


def test_slow2_uptake_is_bounded_above_by_fast():
    log_pf, k_ints, weights, assignments = inputs()
    times = np.asarray([0.01, 0.2, 1.0, 10.0])
    slow = residue_uptake_slow2(log_pf, k_ints, times, weights, assignments)
    fast = residue_uptake_fast(log_pf, k_ints, times, weights)
    assert np.all(slow <= fast + 1e-14)


def test_tau_zero_is_ex2():
    log_pf, k_ints, _, _ = inputs()
    expected = k_ints[:, None] * np.exp(-log_pf)
    np.testing.assert_allclose(effective_rates(log_pf, k_ints, tau=0.0), expected)


def test_tau_is_continuous_with_ex2_at_zero():
    log_pf, k_ints, _, _ = inputs()
    zero = effective_rates(log_pf, k_ints, tau=0.0)
    near_zero = effective_rates(log_pf, k_ints, tau=1e-12)
    np.testing.assert_allclose(near_zero, zero, rtol=3e-12)


def test_negative_log_pf_is_rejected():
    log_pf, k_ints, _, _ = inputs()
    log_pf[0, 0] = -1e-6
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        effective_rates(log_pf, k_ints)


def test_single_cluster_slow2_equals_fast():
    log_pf, k_ints, weights, _ = inputs()
    times = np.asarray([0.1, 1.0, 10.0])
    assignments = np.zeros(weights.size, dtype=int)
    np.testing.assert_allclose(
        residue_uptake_slow2(log_pf, k_ints, times, weights, assignments),
        residue_uptake_fast(log_pf, k_ints, times, weights),
    )


def test_cluster_population_weights_have_requested_masses():
    assignments = np.asarray([0, 0, 1, 1, 1, -1])
    populations = {0: 0.2, 1: 0.5, -1: 0.3}
    weights = weights_from_cluster_populations(assignments, populations)
    assert weights.sum() == 1.0
    for label, expected in populations.items():
        np.testing.assert_allclose(weights[assignments == label].sum(), expected)


def test_width10_peptide_aggregation_matches_bradshaw_oracle():
    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "jaxent/examples/1_IsoValidation_OMass/data/generate_iso_targets.py"
    spec = importlib.util.spec_from_file_location("generate_iso_targets", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    artificial = repo_root / "jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/artificial_HDX_data"
    residue = np.loadtxt(artificial / "mixed_60-40_artificial_expt_resfracs.dat")
    uptake, segments = module.peptide_aggregate(residue[:, 2:].T, residue[:, :2], 10)
    oracle = np.loadtxt(artificial / "mixed_60-40_artificial_expt_segfracs10.dat")
    np.testing.assert_array_equal(segments, oracle[:, :2].astype(int))
    np.testing.assert_allclose(np.round(uptake.T, 5), oracle[:, 2:], atol=1.01e-5)


def test_residue_alignment_preserves_feature_order_and_appends_terminal(tmp_path):
    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "jaxent/examples/1_IsoValidation_OMass/data/generate_iso_targets.py"
    spec = importlib.util.spec_from_file_location("generate_iso_targets_alignment", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    topology = tmp_path / "topology.json"
    topology.write_text(json.dumps({"topologies": [{"residues": [2]}, {"residues": [3]}]}))
    segments = tmp_path / "segments.txt"
    np.savetxt(segments, np.asarray([[1, 2], [2, 3], [3, 4]]), fmt="%d")
    uptake = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    aligned, _ = module.align_residue_layout(uptake, topology, segments)
    np.testing.assert_array_equal(aligned[:, :2], uptake)
    np.testing.assert_array_equal(aligned[:, -1], np.ones(2))
