from dataclasses import dataclass
from typing import ClassVar, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from jaxent.src.custom_types.key import m_key
from jaxent.src.data.loader import ExpD_Datapoint

# Import the classes and functions under test
# Note: You would need to adjust these imports based on your actual module structure
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping, create_sparse_map
from jaxent.src.interfaces.topology import Partial_Topology, TopologyFactory


@dataclass(slots=True)
class MockInputFeatures:
    """
    Mock implementation of Input_Features for testing.
    This is structurally compatible with the Input_Features protocol.
    """

    features: jax.Array  # (n_residues, n_features)
    key: m_key = m_key("test_features")

    __features__: ClassVar[set[str]] = {"features"}

    @property
    def features_shape(self) -> tuple[int, ...]:
        return self.features.shape


@dataclass()
class MockExpDatapoint(ExpD_Datapoint):
    """Mock implementation of ExpD_Datapoint for testing"""

    top: Partial_Topology
    dfrac: List[float]
    key = m_key("HDX_peptide")

    def extract_features(self) -> np.ndarray:
        return np.array(self.dfrac).reshape(-1, 1)


class TestCreateSparseMap:
    """Comprehensive test suite for create_sparse_map function"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create test topologies - Chain A residues 1-10
        self.feature_topologies = [
            TopologyFactory.from_single(
                chain="A", residue=1, fragment_index=0, fragment_name="res1"
            ),
            TopologyFactory.from_single(
                chain="A", residue=2, fragment_index=1, fragment_name="res2"
            ),
            TopologyFactory.from_single(
                chain="A", residue=3, fragment_index=2, fragment_name="res3"
            ),
            TopologyFactory.from_single(
                chain="A", residue=4, fragment_index=3, fragment_name="res4"
            ),
            TopologyFactory.from_single(
                chain="A", residue=5, fragment_index=4, fragment_name="res5"
            ),
        ]

        # Create test input features (5 residues, 3 features each)
        self.input_features = MockInputFeatures(
            features=jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                ]
            )
        )

        # Create test output features (experimental peptides)
        self.output_features = [
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="A", start=1, end=3, fragment_name="pep1"),
                dfrac=[0.1, 0.2, 0.3],
            ),
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="A", start=3, end=5, fragment_name="pep2"),
                dfrac=[0.4, 0.5, 0.6],
            ),
        ]

    def test_basic_functionality(self):
        """Test basic functionality with overlapping peptides"""
        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=self.output_features,
            check_trim=False,
        )

        # Check matrix shape: (2 output peptides, 5 input residues)
        assert sparse_matrix.shape == (2, 5)

        # Convert to dense for easier testing
        dense_matrix = sparse_matrix.todense()

        # First peptide (residues 1-3) should map to features 0, 1, 2
        # Weights should be 1/3 each (equal contribution)
        expected_row_0 = jnp.array([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0])
        np.testing.assert_allclose(dense_matrix[0], expected_row_0, rtol=1e-5)

        # Second peptide (residues 3-5) should map to features 2, 3, 4
        # Weights should be 1/3 each
        expected_row_1 = jnp.array([0.0, 0.0, 1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(dense_matrix[1], expected_row_1, rtol=1e-5)

    def test_non_overlapping_peptides(self):
        """Test with non-overlapping peptides"""
        non_overlapping_output = [
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="A", start=1, end=2, fragment_name="pep1"),
                dfrac=[0.1, 0.2],
            ),
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="A", start=4, end=5, fragment_name="pep2"),
                dfrac=[0.4, 0.5],
            ),
        ]

        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=non_overlapping_output,
            check_trim=False,
        )

        dense_matrix = sparse_matrix.todense()

        # First peptide (residues 1-2)
        expected_row_0 = jnp.array([0.5, 0.5, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(dense_matrix[0], expected_row_0, rtol=1e-5)

        # Second peptide (residues 4-5)
        expected_row_1 = jnp.array([0.0, 0.0, 0.0, 0.5, 0.5])
        np.testing.assert_allclose(dense_matrix[1], expected_row_1, rtol=1e-5)

    def test_single_residue_peptides(self):
        """Test with single residue peptides"""
        single_residue_output = [
            MockExpDatapoint(
                top=TopologyFactory.from_single(chain="A", residue=2, fragment_name="pep1"),
                dfrac=[0.1],
            ),
            MockExpDatapoint(
                top=TopologyFactory.from_single(chain="A", residue=4, fragment_name="pep2"),
                dfrac=[0.4],
            ),
        ]

        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=single_residue_output,
            check_trim=False,
        )

        dense_matrix = sparse_matrix.todense()

        # Each peptide should map to exactly one residue with weight 1.0
        expected_matrix = jnp.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],  # peptide 1 -> residue 2 (index 1)
                [0.0, 0.0, 0.0, 1.0, 0.0],  # peptide 2 -> residue 4 (index 3)
            ]
        )
        np.testing.assert_allclose(dense_matrix, expected_matrix, rtol=1e-5)

    def test_multiple_chains(self):
        """Test with multiple chains"""
        # Add some chain B topologies
        multi_chain_features = self.feature_topologies + [
            TopologyFactory.from_single(
                chain="B", residue=1, fragment_index=5, fragment_name="resB1"
            ),
            TopologyFactory.from_single(
                chain="B", residue=2, fragment_index=6, fragment_name="resB2"
            ),
        ]

        # Extended input features for 7 residues
        extended_input = MockInputFeatures(
            features=jnp.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0],  # Chain B residue 1
                    [19.0, 20.0, 21.0],
                ]
            )  # Chain B residue 2
        )

        # Output features from both chains
        multi_chain_output = [
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="A", start=1, end=2, fragment_name="pepA"),
                dfrac=[0.1, 0.2],
            ),
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="B", start=1, end=2, fragment_name="pepB"),
                dfrac=[0.3, 0.4],
            ),
        ]

        sparse_matrix = create_sparse_map(
            input_features=extended_input,
            feature_topology=multi_chain_features,
            output_features=multi_chain_output,
            check_trim=False,
        )

        dense_matrix = sparse_matrix.todense()

        # Chain A peptide should only map to chain A residues
        expected_row_0 = jnp.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(dense_matrix[0], expected_row_0, rtol=1e-5)

        # Chain B peptide should only map to chain B residues
        expected_row_1 = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5])
        np.testing.assert_allclose(dense_matrix[1], expected_row_1, rtol=1e-5)

    def test_peptide_trimming(self):
        """Test peptide trimming functionality"""
        # Create peptide topologies with trimming
        peptide_features = [
            TopologyFactory.from_range(
                chain="A",
                start=1,
                end=5,
                fragment_index=0,
                fragment_name="peptide1",
                peptide=True,
                peptide_trim=2,
            )
        ]

        peptide_input = MockInputFeatures(
            features=jnp.array([[1.0]])  # Single feature for the peptide
        )

        # Output peptide that overlaps with trimmed region
        trimmed_output = [
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="A", start=3, end=4, fragment_name="exp_pep"),
                dfrac=[0.1, 0.2],
            )
        ]

        # Test with trimming enabled
        sparse_matrix = create_sparse_map(
            input_features=peptide_input,
            feature_topology=peptide_features,
            output_features=trimmed_output,
            check_trim=True,
        )

        dense_matrix = sparse_matrix.todense()

        # Should have mapping since residues 3-4 are in the trimmed peptide (3-5)
        assert jnp.sum(dense_matrix) > 0

        # Test with trimming disabled
        sparse_matrix_no_trim = create_sparse_map(
            input_features=peptide_input,
            feature_topology=peptide_features,
            output_features=trimmed_output,
            check_trim=False,
        )

        dense_matrix_no_trim = sparse_matrix_no_trim.todense()

        # Should still have mapping since residues 3-4 are in the full peptide (1-5)
        assert jnp.sum(dense_matrix_no_trim) > 0

    def test_error_cases(self):
        """Test various error conditions"""

        # Test mismatched input features and topology length
        with pytest.raises(AssertionError, match="Input features and topology do not match"):
            wrong_features = MockInputFeatures(
                features=jnp.array([[1.0], [2.0]])
            )  # Only 2 features
            create_sparse_map(
                input_features=wrong_features,
                feature_topology=self.feature_topologies,  # 5 topologies
                output_features=self.output_features,
                check_trim=False,
            )

        # Test missing fragment indices
        bad_topology = [TopologyFactory.from_single(chain="A", residue=1, fragment_index=None)]
        bad_input = MockInputFeatures(features=jnp.array([[1.0]]))
        bad_output = [
            MockExpDatapoint(
                top=TopologyFactory.from_single(chain="A", residue=1, fragment_name="pep1"),
                dfrac=[0.1],
            )
        ]

        with pytest.raises(ValueError, match="Fragment indices are invalid"):
            create_sparse_map(
                input_features=bad_input,
                feature_topology=bad_topology,
                output_features=bad_output,
                check_trim=False,
            )

        # Test non-unique fragment indices
        duplicate_topology = [
            TopologyFactory.from_single(chain="A", residue=1, fragment_index=0),
            TopologyFactory.from_single(chain="A", residue=2, fragment_index=0),  # Duplicate index
        ]
        duplicate_input = MockInputFeatures(features=jnp.array([[1.0], [2.0]]))

        with pytest.raises(ValueError, match="Fragment indices are invalid"):
            create_sparse_map(
                input_features=duplicate_input,
                feature_topology=duplicate_topology,
                output_features=self.output_features,
                check_trim=False,
            )

    def test_no_overlapping_residues(self):
        """Test case where no residues overlap"""
        # Output features that don't overlap with feature topology
        no_overlap_output = [
            MockExpDatapoint(
                top=TopologyFactory.from_range(
                    chain="B", start=10, end=12, fragment_name="no_overlap"
                ),
                dfrac=[0.1, 0.2, 0.3],
            )
        ]

        with pytest.raises(ValueError, match="No matching residues found"):
            create_sparse_map(
                input_features=self.input_features,
                feature_topology=self.feature_topologies,
                output_features=no_overlap_output,
                check_trim=False,
            )

    def test_complex_overlap_weights(self):
        """Test complex overlap scenarios with different weight calculations"""
        # Create peptides with different overlap patterns
        complex_output = [
            # Peptide that fully contains a single residue
            MockExpDatapoint(
                top=TopologyFactory.from_single(chain="A", residue=2, fragment_name="single"),
                dfrac=[0.1],
            ),
            # Peptide that partially overlaps
            MockExpDatapoint(
                top=TopologyFactory.from_range(chain="A", start=1, end=4, fragment_name="partial"),
                dfrac=[0.1, 0.2, 0.3, 0.4],
            ),
        ]

        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=complex_output,
            check_trim=False,
        )

        dense_matrix = sparse_matrix.todense()

        # First peptide: single residue 2 -> weight 1.0 at index 1
        expected_row_0 = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(dense_matrix[0], expected_row_0, rtol=1e-5)

        # Second peptide: residues 1-4 -> weight 1/4 each for indices 0,1,2,3
        expected_row_1 = jnp.array([0.25, 0.25, 0.25, 0.25, 0.0])
        np.testing.assert_allclose(dense_matrix[1], expected_row_1, rtol=1e-5)

    def test_apply_sparse_mapping(self):
        """Test the apply_sparse_mapping function"""
        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=self.output_features,
            check_trim=False,
        )

        # Test with the actual input features
        test_features = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 residues

        mapped_features = apply_sparse_mapping(sparse_matrix, test_features)

        # Should have 2 output features (for 2 peptides)
        assert mapped_features.shape == (2,)

        # First peptide (residues 1-3): (1+2+3)/3 = 2.0
        expected_0 = (1.0 + 2.0 + 3.0) / 3
        assert abs(mapped_features[0] - expected_0) < 1e-5

        # Second peptide (residues 3-5): (3+4+5)/3 = 4.0
        expected_1 = (3.0 + 4.0 + 5.0) / 3
        assert abs(mapped_features[1] - expected_1) < 1e-5

    def test_sparsity_calculation(self):
        """Test that sparsity is calculated correctly"""
        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=self.output_features,
            check_trim=False,
        )

        # Count non-zero elements
        dense_matrix = sparse_matrix.todense()
        non_zero_count = jnp.count_nonzero(dense_matrix)
        total_elements = dense_matrix.size
        sparsity = non_zero_count / total_elements

        # Should have 6 non-zero elements (3 per peptide) out of 10 total
        assert non_zero_count == 6
        assert abs(sparsity - 0.6) < 1e-5

    def test_fragment_index_sorting(self):
        """Test that fragment indices are properly sorted"""
        # Create topologies with out-of-order indices
        unsorted_topology = [
            TopologyFactory.from_single(
                chain="A", residue=3, fragment_index=2, fragment_name="res3"
            ),
            TopologyFactory.from_single(
                chain="A", residue=1, fragment_index=0, fragment_name="res1"
            ),
            TopologyFactory.from_single(
                chain="A", residue=2, fragment_index=1, fragment_name="res2"
            ),
        ]

        unsorted_input = MockInputFeatures(features=jnp.array([[1.0], [2.0], [3.0]]))

        single_output = [
            MockExpDatapoint(
                top=TopologyFactory.from_single(chain="A", residue=2, fragment_name="test"),
                dfrac=[0.1],
            )
        ]

        sparse_matrix = create_sparse_map(
            input_features=unsorted_input,
            feature_topology=unsorted_topology,
            output_features=single_output,
            check_trim=False,
        )

        dense_matrix = sparse_matrix.todense()

        # Should map to index 1 (fragment_index=1 corresponds to residue 2)
        expected = jnp.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(dense_matrix[0], expected, rtol=1e-5)

    @pytest.mark.parametrize("check_trim", [True, False])
    def test_check_trim_parameter(self, check_trim):
        """Test that check_trim parameter works for both values"""
        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=self.output_features,
            check_trim=check_trim,
        )

        # Should work for both trim values with our test data
        assert sparse_matrix.shape == (2, 5)
        assert jnp.sum(sparse_matrix.todense()) > 0

    def test_empty_output_features(self):
        """Test behavior with empty output features"""
        with pytest.raises(ValueError, match="No matching residues found"):
            create_sparse_map(
                input_features=self.input_features,
                feature_topology=self.feature_topologies,
                output_features=[],  # Empty output
                check_trim=False,
            )

    def test_bcoo_format_properties(self):
        """Test that the returned matrix is properly formatted as BCOO"""
        sparse_matrix = create_sparse_map(
            input_features=self.input_features,
            feature_topology=self.feature_topologies,
            output_features=self.output_features,
            check_trim=False,
        )

        # Check that it's a BCOO matrix
        assert isinstance(sparse_matrix, sparse.BCOO)

        # Check that indices and data have correct types
        assert sparse_matrix.indices.dtype == jnp.int32
        assert sparse_matrix.data.dtype == jnp.float32

        # Check that the matrix can be converted to dense without issues
        dense_matrix = sparse_matrix.todense()
        assert dense_matrix.shape == (2, 5)
        assert jnp.all(jnp.isfinite(dense_matrix))


if __name__ == "__main__":
    pytest.main([__file__])
    pytest.main([__file__])
