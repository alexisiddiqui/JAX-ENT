from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.data.loader import ExpD_Datapoint


# Mock for Input_Features protocol
class MockInputFeatures:
    """Mock Input_Features for testing"""
    def __init__(self, shape):
        self.features_shape = shape
        self.data = np.random.rand(*shape)

# Concrete implementation for ExpD_Datapoint abstract class
@dataclass
class MockExpDDatapoint(ExpD_Datapoint):
    """Mock ExpD_Datapoint for testing"""
    top: Partial_Topology
    
    def extract_features(self) -> np.ndarray:
        # Not used in create_sparse_map, so can be empty
        return np.array([])

class TestCreateSparseMap:
    """Test suite for create_sparse_map function"""

    def test_basic_functionality(self):
        """Test basic functionality with valid inputs"""
        # Create mock input features
        input_features = MockInputFeatures((3,))

        # Create feature topology
        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=1),
            Partial_Topology(chain="B", residues=[1, 2], fragment_index=2),
        ]

        # Create output features
        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3])),
            MockExpDDatapoint(top=Partial_Topology(chain="B", residues=[1, 2])),
        ]

        # Test the function
        result = create_sparse_map(input_features, feature_topology, output_features)

        # Verify result type and shape
        assert isinstance(result, sparse.BCOO)
        assert result.shape == (2, 3)  # n_fragments x n_residues

        # Convert to dense for easier verification
        dense_result = result.todense()

        # Check that we have non-zero values where expected
        assert jnp.sum(dense_result) > 0

    def test_input_validation_shape_mismatch(self):
        """Test input validation for shape mismatch"""
        input_features = MockInputFeatures((2,))  # 2 residues

        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=1),
            Partial_Topology(chain="B", residues=[1, 2], fragment_index=2),
        ]  # 3 fragments

        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3])),
        ]

        with pytest.raises(AssertionError, match="Input features and topology do not match"):
            create_sparse_map(input_features, feature_topology, output_features)

    def test_missing_fragment_indices(self):
        """Test validation for missing fragment indices"""
        input_features = MockInputFeatures((2,))

        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=None),  # Missing index
        ]

        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3])),
        ]

        with pytest.raises(ValueError, match="Fragment indices are invalid"):
            create_sparse_map(input_features, feature_topology, output_features)

    def test_non_unique_fragment_indices(self):
        """Test validation for non-unique fragment indices"""
        input_features = MockInputFeatures((2,))

        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=0),  # Duplicate index
        ]

        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3])),
        ]

        with pytest.raises(ValueError, match="Fragment indices are invalid"):
            create_sparse_map(input_features, feature_topology, output_features)

    def test_incorrect_fragment_index_sequence(self):
        """Test validation for incorrect fragment index sequence"""
        input_features = MockInputFeatures((2,))

        feature_topology = [
            Partial_Topology(
                chain="A", residues=[1, 2], fragment_index=1
            ),  # Should start from 0
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=2),
        ]

        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3])),
        ]

        with pytest.raises(AssertionError, match="Topology fragments are not indexed from 0"):
            create_sparse_map(input_features, feature_topology, output_features)

    def test_no_overlapping_residues(self):
        """Test error when no residues overlap"""
        input_features = MockInputFeatures((2,))

        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=1),
        ]

        output_features = [
            MockExpDDatapoint(
                top=Partial_Topology(chain="B", residues=[5, 6])
            ),  # Different chain
        ]

        with pytest.raises(ValueError, match="No matching residues found"):
            create_sparse_map(input_features, feature_topology, output_features)

    def test_multiple_chains(self):
        """Test functionality with multiple chains"""
        input_features = MockInputFeatures((4,))

        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=1),
            Partial_Topology(chain="B", residues=[1, 2], fragment_index=2),
            Partial_Topology(chain="B", residues=[3, 4], fragment_index=3),
        ]

        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3])),
            MockExpDDatapoint(top=Partial_Topology(chain="B", residues=[2, 3, 4])),
        ]

        result = create_sparse_map(input_features, feature_topology, output_features)

        assert isinstance(result, sparse.BCOO)
        assert result.shape == (2, 4)

        dense_result = result.todense()
        assert jnp.sum(dense_result) > 0

    def test_peptide_trimming(self):
        """Test peptide trimming functionality"""
        input_features = MockInputFeatures((3,))

        # Create peptide topology with trimming
        feature_topology = [
            Partial_Topology(
                chain="A", residues=[1, 2, 3, 4], fragment_index=0, peptide=True, peptide_trim=2
            ),  # Will use residues [3, 4]
            Partial_Topology(chain="A", residues=[5, 6], fragment_index=1),
            Partial_Topology(chain="A", residues=[7, 8], fragment_index=2),
        ]

        output_features = [
            MockExpDDatapoint(
                top=Partial_Topology(chain="A", residues=[3, 4, 5], peptide=True, peptide_trim=1)
            ),  # Will use [4, 5]
        ]

        # Test without trimming
        result_no_trim = create_sparse_map(
            input_features, feature_topology, output_features, check_trim=False
        )

        # Test with trimming
        result_trim = create_sparse_map(
            input_features, feature_topology, output_features, check_trim=True
        )

        # Both should be valid but potentially different
        assert isinstance(result_no_trim, sparse.BCOO)
        assert isinstance(result_trim, sparse.BCOO)
        assert result_no_trim.shape == result_trim.shape == (1, 3)

    def test_single_residue_fragments(self):
        """Test with single residue fragments"""
        input_features = MockInputFeatures((3,))

        feature_topology = [
            Partial_Topology(chain="A", residues=[1], fragment_index=0),
            Partial_Topology(chain="A", residues=[2], fragment_index=1),
            Partial_Topology(chain="A", residues=[3], fragment_index=2),
        ]

        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1])),
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[2])),
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[3])),
        ]

        result = create_sparse_map(input_features, feature_topology, output_features)

        assert isinstance(result, sparse.BCOO)
        assert result.shape == (3, 3)

        # Should be identity-like matrix
        dense_result = result.todense()
        assert jnp.allclose(dense_result, jnp.eye(3))

    def test_contribution_weights(self):
        """Test that contribution weights are calculated correctly"""
        input_features = MockInputFeatures((3,))

        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[2, 3], fragment_index=1),
            Partial_Topology(chain="A", residues=[4, 5], fragment_index=2),
        ]

        # Output fragment covers residues [1, 2, 3, 4] = 4 residues total
        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3, 4])),
        ]

        result = create_sparse_map(input_features, feature_topology, output_features)
        dense_result = result.todense()

        # Check contribution weights:
        # Fragment 0 overlaps [1, 2] = 2 residues out of 4 = 0.5 weight
        # Fragment 1 overlaps [2, 3] = 2 residues out of 4 = 0.5 weight
        # Fragment 2 overlaps [4] = 1 residue out of 4 = 0.25 weight
        expected_weights = jnp.array([[0.5, 0.5, 0.25]])
        assert jnp.allclose(dense_result, expected_weights)

    def test_empty_output_features(self):
        """Test with empty output features"""
        input_features = MockInputFeatures((2,))

        feature_topology = [
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=1),
        ]

        output_features = []

        with pytest.raises(ValueError, match="No matching residues found"):
            create_sparse_map(input_features, feature_topology, output_features)

    def test_empty_feature_topology(self):
        """Test with empty feature topology"""
        input_features = MockInputFeatures((0,))
        feature_topology = []
        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2])),
        ]

        # This should pass input validation but fail during processing
        with pytest.raises(ValueError, match="No matching residues found"):
            create_sparse_map(input_features, feature_topology, output_features)

    def test_fragment_index_sorting(self):
        """Test that feature topology is correctly sorted by fragment index"""
        input_features = MockInputFeatures((3,))

        # Provide unsorted feature topology
        feature_topology = [
            Partial_Topology(chain="A", residues=[3, 4], fragment_index=2),
            Partial_Topology(chain="A", residues=[1, 2], fragment_index=0),
            Partial_Topology(chain="A", residues=[2, 3], fragment_index=1),
        ]

        output_features = [
            MockExpDDatapoint(top=Partial_Topology(chain="A", residues=[1, 2, 3])),
        ]

        # Should work correctly despite unsorted input
        result = create_sparse_map(input_features, feature_topology, output_features)

        assert isinstance(result, sparse.BCOO)
        assert result.shape == (1, 3)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])