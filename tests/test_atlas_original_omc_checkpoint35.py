"""Synthetic, disk-free checks for the checkpoint-35 analysis."""

import numpy as np
import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.original_omc_iso_validation_checkpoint35 import (
    SIGMA_QUANTILES,
    _verdict,
    arm_specs,
    basin_mass_target,
    clustered_median_bootstrap,
    evaluate_gates,
    optimise_omc_batch,
    rare_basin,
    rewire_distances,
)


def test_rare_basin_selects_planted_ten_percent_cluster_and_rejects_40_60():
    coordinate = np.concatenate([np.linspace(0, 0.03, 13), np.linspace(2, 30, 115)])
    selected = rare_basin(np.abs(coordinate[:, None] - coordinate[None, :]))
    assert selected is not None
    assert selected.n_frames == 13
    assert 0.06 <= selected.natural_mass <= 0.15
    assert selected.compactness < 1.0

    coordinate = np.concatenate([np.zeros(40), np.full(60, 10.0)])
    assert rare_basin(np.abs(coordinate[:, None] - coordinate[None, :])) is None


def test_basin_mass_target_uses_requested_absolute_mass_and_uniform_pieces():
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    target = basin_mass_target(labels, 0, 0.5)
    np.testing.assert_allclose(target[labels == 0], 0.25)
    np.testing.assert_allclose(target[labels == 1], 0.5 / 8)
    np.testing.assert_allclose(target[labels == 0].sum(), 0.5)
    with np.testing.assert_raises_regex(AssertionError, "enrichment"):
        basin_mass_target(labels, 0, 0.3)


def test_arm_grid_has_unique_complete_family_partition():
    specs = arm_specs()
    assert len(specs) == 20
    assert len({spec.key for spec in specs}) == 20
    assert 0.16 in SIGMA_QUANTILES
    counts = pd.Series([spec.family for spec in specs]).value_counts().to_dict()
    assert counts == {
        "omc": 6,
        "omc_rewired": 6,
        "omc_norm": 6,
        "maxent": 1,
        "prior_rbf_locked": 1,
    }


def test_rewired_control_preserves_distance_multiset_and_is_nontrivial():
    coordinate = np.array([0.0, 0.2, 1.1, 2.7, 5.0])
    distances = np.abs(coordinate[:, None] - coordinate[None, :])
    permutation = np.array([3, 0, 4, 1, 2])
    rewired = rewire_distances(distances, permutation)
    np.testing.assert_allclose(np.sort(rewired.ravel()), np.sort(distances.ravel()))
    assert not np.allclose(rewired, distances)


def test_batched_path_matches_one_arm_invocation():
    observables = np.array([[[0.1, 0.3, 0.6, 0.8, 0.2], [0.7, 0.1, 0.4, 0.2, 0.9]]])
    truth = np.array([0.1, 0.1, 0.2, 0.25, 0.35])
    prior = np.full(5, 0.2)
    distance = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :])
    similarities = np.stack([np.exp(-(distance**2) / 2), np.exp(-(distance**2) / 8)])
    kwargs = dict(
        observables=observables,
        truth=truth,
        prior=prior,
        train=np.array([0]),
        validation=np.array([1]),
        normalise=np.array([False]),
        strengths=(0.1,),
        steps=12,
    )
    reference = optimise_omc_batch(similarities=similarities[:1], **kwargs)
    batch = optimise_omc_batch(
        similarities=similarities,
        normalise=np.array([False, False]),
        **{key: value for key, value in kwargs.items() if key != "normalise"},
    )
    np.testing.assert_allclose(batch["weights"][0], reference["weights"][0], rtol=1e-6)


def test_clustered_bootstrap_uses_system_as_resampling_unit():
    values = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    systems = np.array(["a", "a", "b", "b", "c", "c"])
    assert clustered_median_bootstrap(values, systems, 7, samples=200) == (2.0, 2.0)
    interval = clustered_median_bootstrap(
        np.array([1.0, 9.0]), np.array(["only", "only"]), 7, samples=200
    )
    assert interval == (5.0, 5.0)


def test_gate_helpers_cover_three_verdicts_and_boundary_is_inconclusive():
    assert _verdict(True)["verdict"] == "pass"
    assert _verdict(False)["verdict"] == "fail"
    assert _verdict(None)["verdict"] == "inconclusive"
    rows = []
    for family, arm, quantile in (
        ("maxent", "maxent", np.nan),
        ("omc", "omc[0.16]", 0.16),
    ):
        rows.append(
            {
                "system_id": "one",
                "target_mass": 0.5,
                "arm_family": family,
                "arm": arm,
                "sigma_quantile": quantile,
                "strength": 0.0,
                "validation_mse": 1.0,
                "target_mass_error": 0.2,
                "test_mse": 1.0,
                "simplex_valid": True,
                "train_mse": 1.0,
                "plateau_flag": False,
                "support_size": 5,
                "ess_fraction": 1.0,
                "target_ess_fraction": 1.0,
                "structural_dispersion": 1.0,
            }
        )
    report = evaluate_gates(pd.DataFrame(rows))
    assert report["overall"] == "inconclusive"
    assert all(
        report[key]["verdict"] == "inconclusive"
        for key in (
            "G1_recovery",
            "G2_specificity",
            "G3_non_inferiority",
            "G4_diversity_control",
        )
    )
