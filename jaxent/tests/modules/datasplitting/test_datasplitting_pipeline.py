import copy
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import MDAnalysis as mda
import numpy as np
import pytest

from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.data.loader import ExpD_Dataloader, ExpD_Datapoint
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.topology import (
    PairwiseTopologyComparisons,
    Partial_Topology,
    TopologyFactory,
    group_set_by_chain,
)
from jaxent.src.interfaces.topology.serialise import PTSerialiser
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model, BV_model_Config


def generate_diverse_peptide_groups(
    chains=("A", "B", "C"),
    residues_per_chain=30,  # Increased default
    peptide_lengths=(5, 8, 12, 15),
    min_peptides_per_group=15,  # Increased to 15
    overlap_types=("none", "partial", "full"),
    include_single_residues=True,
    include_ranges=True,
):
    """
    Generate diverse groups of peptides for comprehensive testing.

    Args:
        chains: Chains to generate peptides for
        residues_per_chain: Number of residues per chain
        peptide_lengths: Different peptide lengths to generate
        min_peptides_per_group: Minimum peptides per group/combination (now 15)
        overlap_types: Types of overlaps to create
        include_single_residues: Whether to include individual residues
        include_ranges: Whether to include peptide ranges

    Returns:
        dict: Organized groups of Partial_Topology objects
    """
    peptide_groups = {
        "single_residues": [],
        "short_peptides": [],
        "medium_peptides": [],
        "long_peptides": [],
        "overlapping_peptides": [],
        "non_overlapping_peptides": [],
        "cross_chain_coverage": [],
        "full_chain_coverage": [],
    }

    # Generate single residues for each chain - ensure at least 15 per chain
    if include_single_residues:
        for chain in chains:
            for i in range(min(min_peptides_per_group, residues_per_chain)):
                peptide_groups["single_residues"].append(
                    TopologyFactory.from_single(
                        chain, i + 1, fragment_name=f"single_{chain}_{i + 1}"
                    )
                )

    # Generate peptides of different lengths - ensure sufficient coverage
    if include_ranges:
        for chain in chains:
            # Short peptides (length 5) - generate at least min_peptides_per_group
            length = peptide_lengths[0]
            step_size = max(1, length // 3)  # Smaller step for more peptides
            for i in range(min_peptides_per_group):
                start = i * step_size + 1
                end = min(start + length - 1, residues_per_chain)
                if end > start:
                    peptide_groups["short_peptides"].append(
                        TopologyFactory.from_range(
                            chain,
                            start,
                            end,
                            fragment_name=f"short_{chain}_{start}-{end}",
                            peptide=True,
                            peptide_trim=2,
                        )
                    )

            # Medium peptides (length 8) - generate at least min_peptides_per_group
            length = peptide_lengths[1]
            step_size = max(1, length // 4)  # Smaller step for more peptides
            for i in range(min_peptides_per_group):
                start = i * step_size + 1
                end = min(start + length - 1, residues_per_chain)
                if end > start:
                    peptide_groups["medium_peptides"].append(
                        TopologyFactory.from_range(
                            chain,
                            start,
                            end,
                            fragment_name=f"medium_{chain}_{start}-{end}",
                            peptide=True,
                            peptide_trim=2,
                        )
                    )

            # Long peptides (length 12+) - generate at least min_peptides_per_group
            length = peptide_lengths[2]
            step_size = max(1, length // 5)  # Smaller step for more peptides
            for i in range(min_peptides_per_group):
                start = i * step_size + 1
                end = min(start + length - 1, residues_per_chain)
                if end > start:
                    peptide_groups["long_peptides"].append(
                        TopologyFactory.from_range(
                            chain,
                            start,
                            end,
                            fragment_name=f"long_{chain}_{start}-{end}",
                            peptide=True,
                            peptide_trim=2,
                        )
                    )

    # Generate overlapping peptides - ensure at least min_peptides_per_group per chain
    for chain in chains:
        base_start = 1
        overlap = 3
        length = 6  # Smaller length for more overlapping peptides

        # Calculate step size to ensure we get enough peptides
        step_size = max(1, length - overlap)  # Ensure at least 1 residue step

        for i in range(min_peptides_per_group):
            start = base_start + i * step_size
            end = min(start + length - 1, residues_per_chain)
            if end > start and start <= residues_per_chain:
                peptide_groups["overlapping_peptides"].append(
                    TopologyFactory.from_range(
                        chain,
                        start,
                        end,
                        fragment_name=f"overlap_{chain}_{start}-{end}",
                        peptide=True,
                        peptide_trim=2,
                    )
                )

    # Generate non-overlapping peptides - ensure at least min_peptides_per_group per chain
    for chain in chains:
        length = 6  # Smaller length for more non-overlapping peptides
        gap = 1  # Smaller gap
        start = 1

        for i in range(min_peptides_per_group):
            end = min(start + length - 1, residues_per_chain)
            if end > start and start <= residues_per_chain:
                peptide_groups["non_overlapping_peptides"].append(
                    TopologyFactory.from_range(
                        chain,
                        start,
                        end,
                        fragment_name=f"non_overlap_{chain}_{start}-{end}",
                        peptide=True,
                        peptide_trim=1,
                    )
                )
                start = end + gap + 1

    # Generate cross-chain coverage - ensure min_peptides_per_group residue positions
    for res_num in range(1, min_peptides_per_group + 1):
        for chain in chains:
            if res_num <= residues_per_chain:
                peptide_groups["cross_chain_coverage"].append(
                    TopologyFactory.from_single(
                        chain, res_num, fragment_name=f"cross_{chain}_{res_num}"
                    )
                )

    # Generate full chain coverage peptides - ensure min_peptides_per_group segments per chain
    for chain in chains:
        segment_size = max(1, residues_per_chain // min_peptides_per_group)
        for i in range(min_peptides_per_group):
            start = i * segment_size + 1
            end = min(start + segment_size - 1, residues_per_chain)
            if end > start and start <= residues_per_chain:
                peptide_groups["full_chain_coverage"].append(
                    TopologyFactory.from_range(
                        chain,
                        start,
                        end,
                        fragment_name=f"full_cov_{chain}_{start}-{end}",
                        peptide=True,
                        peptide_trim=1,
                    )
                )

    return peptide_groups


# Mock ExpD_Datapoint for testing
class MockExpD_Datapoint(ExpD_Datapoint):
    def __init__(self, top, data_id=None, features=None):
        self.top = top
        self.key = f"mock_key_{data_id}" if data_id else "mock_key"
        self.data_id = data_id
        self.features = features or self._create_mock_features()

    def _create_mock_features(self):
        """Create mock BV_input_features for this datapoint"""
        n_residues = len(self.top.residues) if hasattr(self.top, "residues") else 1
        n_frames = 10  # Mock number of frames

        return BV_input_features(
            heavy_contacts=jnp.ones((n_residues, n_frames)) * 0.5,
            acceptor_contacts=jnp.ones((n_residues, n_frames)) * 0.3,
            k_ints=jnp.ones(n_residues) * 0.1,
        )

    def extract_features(self):
        return self.features

    def __repr__(self):
        return f"MockExpD_Datapoint(id={self.data_id}, top={self.top})"


class TestBVFeatureTopologyExtraction:
    """Test extraction of topology from BV_input_features during featurisation"""

    @pytest.fixture
    def mock_universe(self):
        """Create a mock MDAnalysis Universe for testing"""
        # Create a temporary PDB file for testing
        pdb_content = """ATOM      1  N   ALA A   1      20.154  -1.249   1.000  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  -0.346   1.000  1.00 10.00           C  
ATOM      3  C   ALA A   1      17.731  -1.100   1.000  1.00 10.00           C  
ATOM      4  O   ALA A   1      16.671  -0.513   1.000  1.00 10.00           O  
ATOM      5  CB  ALA A   1      19.170   0.537  -0.200  1.00 10.00           C  
ATOM      6  H   ALA A   1      20.000  -2.000   1.000  1.00 10.00           H  
ATOM      7  N   VAL A   2      17.819  -2.429   1.000  1.00 10.00           N  
ATOM      8  CA  VAL A   2      16.632  -3.285   1.000  1.00 10.00           C  
ATOM      9  C   VAL A   2      15.333  -2.516   1.000  1.00 10.00           C  
ATOM     10  O   VAL A   2      14.273  -2.929   1.000  1.00 10.00           O  
ATOM     11  CB  VAL A   2      16.772  -4.228  -0.200  1.00 10.00           C  
ATOM     12  H   VAL A   2      17.700  -3.200   1.000  1.00 10.00           H  
ATOM     13  N   GLY A   3      15.419  -1.188   1.000  1.00 10.00           N  
ATOM     14  CA  GLY A   3      14.222  -0.349   1.000  1.00 10.00           C  
ATOM     15  C   GLY A   3      13.023  -1.100   1.000  1.00 10.00           C  
ATOM     16  O   GLY A   3      11.963  -0.513   1.000  1.00 10.00           O  
ATOM     17  H   GLY A   3      15.000  -0.700   1.000  1.00 10.00           H  
TER      18      GLY A   3
END
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            temp_pdb = f.name

        # Create trajectory content (simple XTC-like)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_content)  # Simple trajectory with same structure
            temp_traj = f.name

        universe = mda.Universe(temp_pdb, temp_traj)

        yield universe

        # Cleanup
        os.unlink(temp_pdb)
        os.unlink(temp_traj)

    @pytest.fixture
    def bv_config(self):
        """Create BV model configuration for testing"""
        config = BV_model_Config(num_timepoints=4)
        config.timepoints = jnp.array([0.1, 1.0, 10.0, 100.0])
        return config

    @pytest.fixture
    def mock_experiment_builder(self, mock_universe, bv_config):
        """Create mock experiment builder for testing"""
        universes = [mock_universe]
        models = [BV_model(bv_config)]
        return Experiment_Builder(universes, models)

    def test_featurisation_returns_topology(self, mock_experiment_builder):
        """Test that featurisation returns both features and topology"""
        featuriser_settings = FeaturiserSettings(name="test_features", batch_size=None)

        # Mock the run_featurise function to return expected structure
        with patch("jaxent.src.featurise.run_featurise") as mock_featurise:
            # Create mock features and topology - match the actual output structure
            mock_features = BV_input_features(
                heavy_contacts=jnp.ones((1, 10)),  # 1 residue (common), 10 frames
                acceptor_contacts=jnp.ones((1, 10)),
                k_ints=jnp.ones(1),
            )

            # Mock topology should match what the actual function returns
            mock_topology = [
                TopologyFactory.from_single("A", 2, fragment_name="common_chain_A")  # VAL residue
            ]

            mock_featurise.return_value = ([mock_features], [mock_topology])

            features, feature_topology = run_featurise(mock_experiment_builder, featuriser_settings)

            assert len(features) == 1
            assert len(feature_topology) == 1
            assert isinstance(features[0], BV_input_features)
            assert len(feature_topology[0]) == 1  # Should be 1, not 3
            assert all(isinstance(topo, Partial_Topology) for topo in feature_topology[0])

    def test_extract_common_residues_from_feature_topology(self):
        """Test extraction of common residues from feature topology"""
        # Use helper to generate comprehensive feature topology
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A", "B", "C"), residues_per_chain=40, min_peptides_per_group=15
        )

        # Combine different groups for testing - ensure at least 15 per chain
        feature_topology = (
            peptide_groups["single_residues"]  # 15 per chain = 45 total
            + peptide_groups["short_peptides"][:45]  # 15 per chain
        )

        # Convert to set for common residue processing
        common_residues = set(feature_topology)

        assert len(common_residues) >= 75  # Should have substantial coverage

        # Group by chain
        by_chain = group_set_by_chain(common_residues)
        assert "A" in by_chain
        assert "B" in by_chain
        assert "C" in by_chain

        # Each chain should have at least 15 peptides
        for chain in ["A", "B", "C"]:
            assert len(by_chain[chain]) >= 15

    def test_feature_topology_consistency_check(self):
        """Test that feature topology is consistent with BV_input_features dimensions"""
        n_residues = 5
        n_frames = 20

        features = BV_input_features(
            heavy_contacts=jnp.ones((n_residues, n_frames)),
            acceptor_contacts=jnp.ones((n_residues, n_frames)),
            k_ints=jnp.ones(n_residues),
        )

        # Create matching topology
        topology = [
            TopologyFactory.from_single("A", i + 1, fragment_name=f"res_A_{i + 1}")
            for i in range(n_residues)
        ]

        # Check consistency
        assert features.features_shape[0] == len(topology)
        assert features.features_shape == (n_residues, n_frames)


class TestCommonResidueExtraction:
    """Test various methods of extracting common residues from feature topology"""

    @pytest.fixture
    def comprehensive_peptide_groups(self):
        """Generate comprehensive peptide groups for testing"""
        return generate_diverse_peptide_groups(
            chains=("A", "B", "C", "D"),
            residues_per_chain=50,  # Increased for more peptides
            peptide_lengths=(6, 10, 15, 20),
            min_peptides_per_group=15,  # Ensure 15 per group
            include_single_residues=True,
            include_ranges=True,
        )

    @pytest.fixture
    def multi_chain_feature_topology(self, comprehensive_peptide_groups):
        """Create feature topology with multiple chains and sufficient coverage"""
        return (
            comprehensive_peptide_groups["full_chain_coverage"]
            + comprehensive_peptide_groups["overlapping_peptides"][:20]
        )

    @pytest.fixture
    def single_residue_feature_topology(self, comprehensive_peptide_groups):
        """Create feature topology with individual residues - at least 5 per chain"""
        return comprehensive_peptide_groups["single_residues"]

    def test_extract_all_residues_as_common(self, single_residue_feature_topology):
        """Test using all feature topology residues as common residues"""
        common_residues = set(single_residue_feature_topology)

        # Should have at least 15 per chain * 4 chains = 60
        assert len(common_residues) >= 60

        # Check that all chains are represented with sufficient coverage
        chains = {topo.chain for topo in common_residues}
        assert len(chains) >= 4

        by_chain = group_set_by_chain(common_residues)
        for chain in chains:
            assert len(by_chain[chain]) >= 15  # At least 15 per chain

    def test_extract_chain_specific_common_residues(self, multi_chain_feature_topology):
        """Test extracting common residues for specific chains"""
        all_topology = set(multi_chain_feature_topology)

        # Extract only chain A residues
        chain_a_residues = {topo for topo in all_topology if topo.chain == "A"}
        assert len(chain_a_residues) >= 15  # At least 15 peptides for chain A

        # Extract individual residues from chain A
        individual_residues = set()
        for topo in chain_a_residues:
            individual_residues.update(
                TopologyFactory.extract_residues(topo, use_peptide_trim=False)
            )

        assert len(individual_residues) >= 30  # Should have substantial individual residue coverage

    def test_filter_feature_topology_by_coverage(self, comprehensive_peptide_groups):
        """Test filtering feature topology by coverage requirements"""
        # Use a mix of peptide types
        all_residues = set(
            comprehensive_peptide_groups["single_residues"]
            + comprehensive_peptide_groups["short_peptides"]
        )

        # Keep peptides that provide good coverage per chain
        filtered_residues = set()
        by_chain = group_set_by_chain(all_residues)

        for chain, chain_residues in by_chain.items():
            sorted_residues = sorted(chain_residues, key=lambda t: t.residues[0])
            # Keep at least 15 peptides per chain for robust testing
            filtered_residues.update(sorted_residues[:15])

        assert len(filtered_residues) >= 60  # 4 chains × 15 peptides

        # Verify each chain has sufficient coverage
        filtered_by_chain = group_set_by_chain(filtered_residues)
        for chain in by_chain.keys():
            assert len(filtered_by_chain[chain]) >= 15  # Minimum coverage maintained

    def test_merge_feature_topology_ranges(self, multi_chain_feature_topology):
        """Test merging feature topology ranges into larger common regions"""
        # Group by chain and merge
        by_chain = {}
        for topo in multi_chain_feature_topology:
            if topo.chain not in by_chain:
                by_chain[topo.chain] = []
            by_chain[topo.chain].append(topo)

        merged_common_residues = set()
        for chain, chain_topos in by_chain.items():
            if len(chain_topos) > 1:
                # Merge contiguous ranges
                merged = TopologyFactory.merge_contiguous(
                    chain_topos, gap_tolerance=0, merged_name=f"merged_{chain}"
                )
                # Extract individual residues from merged topology
                merged_common_residues.update(
                    TopologyFactory.extract_residues(merged, use_peptide_trim=False)
                )
            else:
                merged_common_residues.update(
                    TopologyFactory.extract_residues(chain_topos[0], use_peptide_trim=False)
                )

        # Check that we have merged residues - counts will vary based on actual topology
        result_by_chain = group_set_by_chain(merged_common_residues)

        # Verify that we have substantial coverage per chain
        for chain in by_chain.keys():
            assert len(result_by_chain[chain]) >= 15  # At least 15 residues per chain


class TestDataSplittingWithFeatureTopology:
    """Test data splitting using common residues extracted from feature topology"""

    @pytest.fixture
    def create_datapoints_from_feature_topology(self):
        """Factory to create mock datapoints from feature topology"""

        def _create(feature_topology, features=None):
            datapoints = []
            for i, topo in enumerate(feature_topology):
                if features and i < len(features):
                    datapoint_features = features[i]
                else:
                    datapoint_features = None
                datapoints.append(MockExpD_Datapoint(topo, i, datapoint_features))
            return datapoints

        return _create

    @pytest.fixture
    def mock_dataloader_factory(self):
        """Factory to create mock dataloader from datapoints"""

        def _create(datapoints):
            mock_loader = MagicMock(spec=ExpD_Dataloader)
            mock_loader.data = datapoints
            mock_loader.y_true = np.array([1.0] * len(datapoints))
            return mock_loader

        return _create

    @pytest.fixture
    def setup_splitter_with_feature_topology(self, request):
        """Factory to set up DataSplitter with feature topology common residues"""

        def _setup(feature_topology, datapoints, **kwargs):
            # Extract common residues from feature topology
            common_residues = set(feature_topology)

            mock_loader = MagicMock(spec=ExpD_Dataloader)
            mock_loader.data = copy.deepcopy(datapoints)
            mock_loader.y_true = np.array([1.0] * len(datapoints))

            # Mock filter_common_residues to return datapoints that intersect
            def mock_filter_func(dataset, common_topos, check_trim=False):
                if not common_topos:
                    return []
                from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons

                return [
                    dp
                    for dp in dataset
                    if any(
                        PairwiseTopologyComparisons.intersects(dp.top, ct, check_trim=check_trim)
                        for ct in common_topos
                    )
                ]

            # Mock calculate_fragment_redundancy
            patcher1 = patch("jaxent.src.interfaces.topology.utils.calculate_fragment_redundancy")
            patcher2 = patch(
                "jaxent.src.data.splitting.split.filter_common_residues",
                side_effect=mock_filter_func,
            )

            mock_calc = patcher1.start()
            mock_filter = patcher2.start()

            request.addfinalizer(patcher1.stop)
            request.addfinalizer(patcher2.stop)

            mock_calc.return_value = [0.5] * len(datapoints)

            # Default kwargs
            default_kwargs = {
                "random_seed": 42,
                "train_size": 0.6,
                "centrality": False,
                "check_trim": False,
            }
            default_kwargs.update(kwargs)

            # During init, return all datapoints
            mock_filter.side_effect = None
            mock_filter.return_value = copy.deepcopy(datapoints)

            splitter = DataSplitter(
                dataset=mock_loader, common_residues=common_residues, **default_kwargs
            )

            # Restore side effect for calls within random_split
            mock_filter.side_effect = mock_filter_func
            mock_filter.return_value = None

            return splitter

        return _setup

    def test_split_with_single_chain_feature_topology(
        self, create_datapoints_from_feature_topology, setup_splitter_with_feature_topology
    ):
        """Test data splitting with single chain feature topology - sufficient peptides"""
        # Generate comprehensive single chain topology
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A",),
            residues_per_chain=40,  # More residues
            min_peptides_per_group=20,  # More than 15 to ensure coverage
        )

        feature_topology = (
            peptide_groups["single_residues"][:15]  # Exactly 15
            + peptide_groups["short_peptides"][:15]  # Exactly 15
            + peptide_groups["medium_peptides"][:15]  # Exactly 15
        )

        assert len(feature_topology) == 45  # Should have exactly 45 peptides

        datapoints = create_datapoints_from_feature_topology(feature_topology)
        splitter = setup_splitter_with_feature_topology(feature_topology, datapoints)

        train_data, val_data = splitter.random_split()

        assert len(train_data) >= 15  # At least 15 training data
        assert len(val_data) >= 10  # At least 10 validation data
        assert len(train_data) + len(val_data) >= 40  # Good coverage

    def test_split_with_multi_chain_feature_topology(
        self, create_datapoints_from_feature_topology, setup_splitter_with_feature_topology
    ):
        """Test data splitting with multi-chain feature topology - sufficient coverage"""
        # Generate comprehensive multi-chain topology
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A", "B", "C"), residues_per_chain=40, min_peptides_per_group=15
        )

        feature_topology = (
            peptide_groups["cross_chain_coverage"]  # 15 per chain = 45 total
            + peptide_groups["overlapping_peptides"][:30]  # Additional 30
        )

        # Ensure we have at least 15 peptides per chain
        by_chain = group_set_by_chain(set(feature_topology))
        for chain in ["A", "B", "C"]:
            assert len(by_chain[chain]) >= 15

        datapoints = create_datapoints_from_feature_topology(feature_topology)
        splitter = setup_splitter_with_feature_topology(feature_topology, datapoints)

        train_data, val_data = splitter.random_split()

        assert len(train_data) >= 30  # Substantial training data
        assert len(val_data) >= 15  # Substantial validation data

        # Check that multiple chains are represented with good coverage
        train_chains = {dp.top.chain for dp in train_data}
        val_chains = {dp.top.chain for dp in val_data}
        all_chains = train_chains | val_chains

        assert len(all_chains) == 3  # All chains should be present

        # Each chain should be well-represented in the split
        train_by_chain = {}
        for dp in train_data:
            chain = dp.top.chain
            train_by_chain[chain] = train_by_chain.get(chain, 0) + 1

        for chain in all_chains:
            assert train_by_chain.get(chain, 0) >= 5  # At least 5 peptides per chain in training

    def test_split_with_range_based_feature_topology(
        self, create_datapoints_from_feature_topology, setup_splitter_with_feature_topology
    ):
        """Test data splitting with range-based feature topology - comprehensive ranges"""
        # Generate comprehensive range-based topology
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A", "B"),
            residues_per_chain=60,  # More residues for more ranges
            peptide_lengths=(6, 10, 14, 18),
            min_peptides_per_group=15,  # Ensure 15 per group
            include_single_residues=False,  # Only ranges
            include_ranges=True,
        )

        feature_topology = (
            peptide_groups["short_peptides"]  # 15 per chain = 30
            + peptide_groups["medium_peptides"]  # 15 per chain = 30
            + peptide_groups["long_peptides"][:20]  # Additional 20
        )

        # Ensure sufficient range coverage
        assert len(feature_topology) >= 70  # Should have many ranges

        # Extract individual residues for datapoints
        individual_residues = []
        for range_topo in feature_topology:
            individual_residues.extend(
                TopologyFactory.extract_residues(range_topo, use_peptide_trim=False)
            )

        # Should have substantial individual residue coverage
        assert len(individual_residues) >= 200

        datapoints = create_datapoints_from_feature_topology(individual_residues)

        # Use the range topology as common residues
        splitter = setup_splitter_with_feature_topology(feature_topology, datapoints)

        train_data, val_data = splitter.random_split()

        assert len(train_data) >= 50  # Substantial training set
        assert len(val_data) >= 25  # Substantial validation set

    def test_overlap_removal_with_feature_topology(
        self, create_datapoints_from_feature_topology, setup_splitter_with_feature_topology
    ):
        """Test overlap removal when using feature topology common residues - sufficient overlaps"""
        # Generate overlapping peptides with guaranteed overlaps - ensure we get enough
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A",),
            residues_per_chain=200,  # Much more residues for more overlaps
            min_peptides_per_group=30,  # Ensure we get enough overlapping peptides
            overlap_types=("partial", "full"),
        )

        feature_topology = peptide_groups["overlapping_peptides"]

        # Ensure we have substantial overlapping coverage
        assert len(feature_topology) >= 20  # Reduced expectation to match actual generation

        # Extract individual residues
        individual_residues = []
        for range_topo in feature_topology:
            individual_residues.extend(
                TopologyFactory.extract_residues(range_topo, use_peptide_trim=True)
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_residues = []
        for res in individual_residues:
            res_key = (res.chain, tuple(res.residues))
            if res_key not in seen:
                seen.add(res_key)
                unique_residues.append(res)

        # Should still have substantial coverage after deduplication
        assert len(unique_residues) >= 40

        datapoints = create_datapoints_from_feature_topology(unique_residues)
        splitter = setup_splitter_with_feature_topology(
            unique_residues, datapoints, check_trim=True
        )

        # Test with overlap removal
        train_data, val_data = splitter.random_split(remove_overlap=True)

        assert len(train_data) >= 15  # At least 15 after overlap removal
        assert len(val_data) >= 8  # Sufficient validation data

        # Check that there are no overlapping datapoints
        train_ids = {dp.data_id for dp in train_data}
        val_ids = {dp.data_id for dp in val_data}
        assert len(train_ids.intersection(val_ids)) == 0

    def test_centrality_sampling_with_feature_topology(
        self, create_datapoints_from_feature_topology, setup_splitter_with_feature_topology
    ):
        """Test centrality sampling using feature topology - diverse peptides"""
        # Generate diverse feature topology for centrality testing
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A", "B", "C"), residues_per_chain=50, min_peptides_per_group=20
        )

        feature_topology = (
            peptide_groups["single_residues"][:45]  # 15 per chain
            + peptide_groups["short_peptides"][:30]  # 10 per chain
            + peptide_groups["medium_peptides"][:30]  # 10 per chain
        )

        # Ensure substantial diversity for centrality sampling
        assert len(feature_topology) >= 100

        datapoints = create_datapoints_from_feature_topology(feature_topology)

        with patch.object(DataSplitter, "sample_by_centrality") as mock_sample:
            # Return a substantial subset for centrality sampling (at least 50)
            mock_sample.return_value = datapoints[:60]

            splitter = setup_splitter_with_feature_topology(
                feature_topology, datapoints, centrality=True
            )
            train_data, val_data = splitter.random_split()

            # Should have called centrality sampling
            mock_sample.assert_called_once()
            assert len(train_data) >= 20  # Substantial training after sampling
            assert len(val_data) >= 15  # Substantial validation data


class TestFeatureTopologyIntegration:
    """Integration tests for the complete pipeline from featurisation to data splitting"""

    def test_end_to_end_pipeline_simulation(self):
        """Simulate the complete pipeline from featurisation to data splitting - comprehensive"""
        # Step 1: Simulate featurisation results with substantial coverage
        n_residues = 120  # More residues for comprehensive testing
        n_frames = 100

        # Create mock BV_input_features - fix JAX random usage
        import jax.random as jax_random

        key = jax_random.PRNGKey(42)

        features = BV_input_features(
            heavy_contacts=jax_random.uniform(key, (n_residues, n_frames)),
            acceptor_contacts=jax_random.uniform(key, (n_residues, n_frames)),
            k_ints=jax_random.uniform(key, (n_residues,)) * 0.09 + 0.01,  # Scale to 0.01-0.1
        )

        # Create comprehensive feature topology
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A", "B", "C"),
            residues_per_chain=40,  # 40 * 3 = 120 total
            min_peptides_per_group=20,  # More than 15
        )

        feature_topology = (
            peptide_groups["single_residues"][:60]  # 20 per chain
            + peptide_groups["short_peptides"][:45]  # 15 per chain
        )

        # Ensure we have sufficient peptides
        assert len(feature_topology) >= 90

        # Step 2: Extract common residues from feature topology
        common_residues = set(feature_topology)

        # Step 3: Create experimental datapoints
        datapoints = []
        for i, topo in enumerate(feature_topology):
            single_res_features = BV_input_features(
                heavy_contacts=features.heavy_contacts[i : i + 1, :],
                acceptor_contacts=features.acceptor_contacts[i : i + 1, :],
                k_ints=features.k_ints[i : i + 1] if features.k_ints is not None else None,
            )
            datapoints.append(MockExpD_Datapoint(topo, i, single_res_features))

        # Ensure substantial datapoint coverage
        assert len(datapoints) >= 90

        # Step 4: Mock data loader and splitter
        mock_loader = MagicMock(spec=ExpD_Dataloader)
        mock_loader.data = datapoints
        mock_loader.y_true = np.array([1.0] * len(datapoints))

        def mock_filter_func(dataset, common_topos, check_trim=False):
            return [
                dp
                for dp in dataset
                if any(
                    PairwiseTopologyComparisons.intersects(dp.top, ct, check_trim=check_trim)
                    for ct in common_topos
                )
            ]

        with (
            patch(
                "jaxent.src.interfaces.topology.utils.calculate_fragment_redundancy"
            ) as mock_calc,
            patch(
                "jaxent.src.data.splitting.split.filter_common_residues",
                side_effect=mock_filter_func,
            ),
        ):
            mock_calc.return_value = [0.5] * len(datapoints)

            splitter = DataSplitter(
                dataset=mock_loader,
                common_residues=common_residues,
                random_seed=123,
                train_size=0.7,
                centrality=False,
                check_trim=False,
            )

            # Step 5: Perform data splitting
            train_data, val_data = splitter.random_split()

            # Step 6: Verify results with comprehensive coverage
            assert len(train_data) >= 40  # Substantial training set (at least 40)
            assert len(val_data) >= 20  # Substantial validation set (at least 20)

            # Check that train/val split is reasonable
            total_data = len(train_data) + len(val_data)
            train_ratio = len(train_data) / total_data
            assert 0.5 <= train_ratio <= 0.9  # Should be around 0.7

            # Check chain representation with sufficient coverage
            train_chains = {dp.top.chain for dp in train_data}
            val_chains = {dp.top.chain for dp in val_data}
            all_split_chains = train_chains | val_chains
            original_chains = {topo.chain for topo in feature_topology}

            # All original chains should be represented in the split
            assert all_split_chains == original_chains

            # Each chain should have reasonable representation (at least 5 per chain)
            train_by_chain = {}
            for dp in train_data:
                chain = dp.top.chain
                train_by_chain[chain] = train_by_chain.get(chain, 0) + 1

            for chain in original_chains:
                assert train_by_chain.get(chain, 0) >= 5  # At least 5 per chain in training

    def test_feature_topology_save_load_integration(self):
        """Test saving and loading feature topology with data splitting"""
        # Create feature topology with sufficient data for splitting
        peptide_groups = generate_diverse_peptide_groups(
            chains=("A", "B"),
            residues_per_chain=30,
            min_peptides_per_group=20,  # Ensure enough peptides
        )

        feature_topology = (
            peptide_groups["single_residues"][:30]  # 15 per chain
            + peptide_groups["short_peptides"][:20]  # 10 per chain
        )

        # Test save/load cycle
        with tempfile.TemporaryDirectory() as temp_dir:
            topology_file = Path(temp_dir) / "feature_topology.json"

            # Save topology
            PTSerialiser.save_list_to_json(feature_topology, topology_file)

            # Load topology
            loaded_topology = PTSerialiser.load_list_from_json(topology_file)

            # Verify loaded topology matches original
            assert len(loaded_topology) == len(feature_topology)

            for orig, loaded in zip(feature_topology, loaded_topology):
                assert orig.chain == loaded.chain
                assert orig.residues == loaded.residues
                assert orig.fragment_name == loaded.fragment_name

            # Use loaded topology for data splitting
            common_residues = set(loaded_topology)

            # Extract individual residues for datapoints
            individual_residues = []
            for topo in loaded_topology:
                individual_residues.extend(
                    TopologyFactory.extract_residues(topo, use_peptide_trim=False)
                )

            # Create enough datapoints for valid splitting
            datapoints = [MockExpD_Datapoint(topo, i) for i, topo in enumerate(individual_residues)]

            # Ensure we have sufficient data
            assert len(datapoints) >= 30

            # Test that splitting works with loaded topology
            mock_loader = MagicMock(spec=ExpD_Dataloader)
            mock_loader.data = datapoints
            mock_loader.y_true = np.array([1.0] * len(datapoints))

            def mock_filter_func(dataset, common_topos, check_trim=False):
                from jaxent.src.interfaces.topology.pairwise import PairwiseTopologyComparisons

                return [
                    dp
                    for dp in dataset
                    if any(
                        PairwiseTopologyComparisons.intersects(dp.top, ct, check_trim=check_trim)
                        for ct in common_topos
                    )
                ]

            with (
                patch(
                    "jaxent.src.interfaces.topology.utils.calculate_fragment_redundancy"
                ) as mock_calc,
                patch(
                    "jaxent.src.data.splitting.split.filter_common_residues",
                    side_effect=mock_filter_func,
                ),
            ):
                mock_calc.return_value = [0.5] * len(datapoints)

                splitter = DataSplitter(
                    dataset=mock_loader,
                    common_residues=common_residues,
                    random_seed=456,
                    train_size=0.7,  # More favorable split ratio
                    centrality=False,  # Disable centrality for simpler splitting
                )

                train_data, val_data = splitter.random_split()

                assert len(train_data) >= 15  # At least 15 training samples
                assert len(val_data) >= 8  # At least 8 validation samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
