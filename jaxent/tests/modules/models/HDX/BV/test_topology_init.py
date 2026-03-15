"""Tests for topology-based initialisation of forward models.

Verifies that BV_model and (where applicable) other ForwardModel subclasses can be
initialised from pre-computed Partial_Topology objects instead of MDAnalysis Universe
objects.
"""

import pytest

from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.topology import Partial_Topology, TopologyFactory
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.HDX.BV.forwardmodel import BV_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_residue_topologies(chain: str, residue_range: range) -> list[Partial_Topology]:
    """Create one single-residue Partial_Topology per residue in *residue_range*."""
    topologies = []
    for idx, resid in enumerate(residue_range):
        topo = Partial_Topology(
            chain=chain,
            residues=[resid],
            fragment_sequence="A",  # Alanine as placeholder
            fragment_name="res",
            fragment_index=idx,
        )
        topologies.append(topo)
    return topologies


# ---------------------------------------------------------------------------
# BV_model topology-based initialisation
# ---------------------------------------------------------------------------

class TestBVModelTopologyInit:
    """Tests for BV_model.initialise() with Partial_Topology inputs."""

    def setup_method(self):
        self.config = BV_model_Config()
        self.model = BV_model(self.config)
        self.topologies = _make_residue_topologies("A", range(3, 18))  # 15 residues

    def test_initialise_returns_true_with_topologies(self):
        result = self.model.initialise(self.topologies)
        assert result is True

    def test_common_topology_set_from_input(self):
        self.model.initialise(self.topologies)
        assert self.model.common_topology == set(self.topologies)

    def test_topology_order_populated(self):
        self.model.initialise(self.topologies)
        assert len(self.model.topology_order) == len(self.topologies)

    def test_n_frames_is_zero(self):
        self.model.initialise(self.topologies)
        assert self.model.n_frames == 0

    def test_common_k_ints_is_empty(self):
        """k_ints are not computed during topology-based initialisation."""
        self.model.initialise(self.topologies)
        assert self.model.common_k_ints == []

    def test_common_k_ints_map_is_empty(self):
        self.model.initialise(self.topologies)
        assert self.model.common_k_ints_map == {}

    def test_single_topology_initialisation(self):
        single = _make_residue_topologies("A", range(5, 6))
        result = self.model.initialise(single)
        assert result is True
        assert len(self.model.common_topology) == 1

    def test_multi_chain_topologies(self):
        chain_a = _make_residue_topologies("A", range(1, 6))
        chain_b = _make_residue_topologies("B", range(1, 6))
        combined = chain_a + chain_b
        result = self.model.initialise(combined)
        assert result is True
        assert len(self.model.common_topology) == 10

    def test_topology_order_sorted(self):
        """topology_order should be rank-sorted (chain then position)."""
        chain_a = _make_residue_topologies("A", range(1, 4))
        chain_b = _make_residue_topologies("B", range(1, 4))
        combined = chain_a + chain_b
        self.model.initialise(combined)
        # All chain-A topologies should appear before chain-B topologies
        chains_in_order = [t.chain for t in self.model.topology_order]
        a_indices = [i for i, c in enumerate(chains_in_order) if c == "A"]
        b_indices = [i for i, c in enumerate(chains_in_order) if c == "B"]
        if a_indices and b_indices:
            assert max(a_indices) < min(b_indices)


# ---------------------------------------------------------------------------
# Experiment_Builder with Partial_Topology inputs
# ---------------------------------------------------------------------------

class TestExperimentBuilderTopologyInit:
    """Tests that Experiment_Builder accepts list[Partial_Topology]."""

    def setup_method(self):
        self.config = BV_model_Config()
        self.model = BV_model(self.config)
        self.topologies = _make_residue_topologies("A", range(3, 18))

    def test_builder_accepts_topology_list(self):
        """Experiment_Builder should not raise when given Partial_Topology objects."""
        builder = Experiment_Builder(
            universes=self.topologies,
            forward_models=[self.model],
        )
        assert builder.ensembles is self.topologies

    def test_validate_forward_models_with_topologies(self):
        builder = Experiment_Builder(
            universes=self.topologies,
            forward_models=[self.model],
        )
        validated = builder.validate_forward_models()
        assert len(validated) == 1
        assert validated[0] is self.model

    def test_validate_sets_model_common_topology(self):
        builder = Experiment_Builder(
            universes=self.topologies,
            forward_models=[self.model],
        )
        builder.validate_forward_models()
        assert self.model.common_topology == set(self.topologies)
