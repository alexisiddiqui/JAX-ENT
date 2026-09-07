import numpy as np
import pytest

from jaxent.examples.ATLAS_BV.analysis.laplacian_residue_scaling_checkpoint33 import (
    FAMILIES,
    K_MAX,
    K_REFERENCES,
    REFERENCE_RESIDUES,
    TOPOLOGIES,
    ResidueKSpec,
    build_scaling_graph,
    candidate_specs,
    residue_scaled_k,
)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("k_reference", K_REFERENCES)
def test_every_rule_equals_its_reference_k_at_reference_length(family, k_reference):
    assert residue_scaled_k(
        REFERENCE_RESIDUES,
        256,
        family=family,
        k_reference=k_reference,
    ) == k_reference


@pytest.mark.parametrize("family", ("sqrt", "linear", "n_log_n"))
def test_adaptive_rules_are_monotonic_in_residue_count(family):
    values = [
        residue_scaled_k(n, 256, family=family, k_reference=40)
        for n in range(60, 249)
    ]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_constant_rule_does_not_depend_on_residue_count():
    assert {
        residue_scaled_k(n, 256, family="constant", k_reference=40)
        for n in (60, 109, 248)
    } == {40}


def test_k_is_capped_by_global_limit_and_available_frames():
    assert residue_scaled_k(248, 256, family="n_log_n", k_reference=120) == K_MAX
    assert residue_scaled_k(248, 32, family="n_log_n", k_reference=120) == 31


def test_invalid_sizes_fail_explicitly():
    with pytest.raises(ValueError, match="n_residues"):
        residue_scaled_k(0, 256, family="sqrt", k_reference=40)
    with pytest.raises(ValueError, match="two frames"):
        residue_scaled_k(109, 1, family="sqrt", k_reference=40)


def test_candidate_grid_is_complete_and_deterministic():
    specs = candidate_specs()
    assert len(specs) == len(FAMILIES) * len(K_REFERENCES) * len(TOPOLOGIES)
    assert len({spec.key for spec in specs}) == len(specs)


def test_weighted_and_uniform_rules_share_the_same_edges():
    distances = np.abs(np.arange(20.0)[:, None] - np.arange(20.0)[None, :])
    weighted = ResidueKSpec("sqrt", 10, "self_tuned_knn")
    uniform = ResidueKSpec("sqrt", 10, "uniform_knn")
    weighted_graph = build_scaling_graph(distances, weighted, REFERENCE_RESIDUES)
    uniform_graph = build_scaling_graph(distances, uniform, REFERENCE_RESIDUES)
    np.testing.assert_array_equal(weighted_graph.edge_sources, uniform_graph.edge_sources)
    np.testing.assert_array_equal(weighted_graph.edge_targets, uniform_graph.edge_targets)
    np.testing.assert_allclose(uniform_graph.edge_weights, 1.0)


def test_candidate_key_records_rule_coefficient_and_topology():
    spec = ResidueKSpec("sqrt", 40, "self_tuned_knn")
    assert spec.key == "work_scale__self_tuned_knn__sqrt__kref_40"
