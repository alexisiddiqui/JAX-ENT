"""
Unit tests for NetHDX forward pass implementation.
Tests numerical correctness using synthetic contact matrices.
Note: NetHDX uses NumPy, not JAX, so no gradient tests are included.
"""

import numpy as np
import pytest

from jaxent.src.models.HDX.netHDX.netHDX_functions import NetHDX_ForwardPass
from jaxent.src.models.HDX.netHDX.features import (
    NetHDX_input_features,
    NetHDX_output_features,
)
from jaxent.src.models.config import NetHDXConfig


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def identity_contact_matrices():
    """Create identity contact matrices (3x3, 2 frames)."""
    matrices = np.array([
        np.eye(3),  # Frame 1: identity
        np.eye(3),  # Frame 2: identity
    ])
    return matrices


@pytest.fixture
def known_contact_matrices():
    """Create known contact matrices for testing.
    Frame 1: [[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]]
    Frame 2: [[0.5, 1, 0], [1, 0.5, 0], [0, 0, 1]]
    """
    matrices = np.array([
        [[1.0, 0.5, 0.0],
         [0.5, 1.0, 0.5],
         [0.0, 0.5, 1.0]],
        [[0.5, 1.0, 0.0],
         [1.0, 0.5, 0.0],
         [0.0, 0.0, 1.0]],
    ])
    return matrices


@pytest.fixture
def dummy_config():
    """Create a dummy NetHDXConfig (minimal)."""
    # NetHDXConfig is used as parameters but not used in the forward pass
    return NetHDXConfig(
        distance_cutoff=3.0,
        angle_cutoff=120.0,
        num_timepoints=1,
    )


# ============================================================================
# Test NetHDX_ForwardPass
# ============================================================================


class TestNetHDXForwardPass:
    """Test NetHDX_ForwardPass implementations."""

    def test_identity_contact_matrix(self, identity_contact_matrices, dummy_config):
        """Test identity contact matrix.
        Average: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        Sum per row: [1, 1, 1]
        log10([1, 1, 1]): [0, 0, 0]
        """
        features = NetHDX_input_features(
            contact_matrices=identity_contact_matrices,
            residue_ids=[1, 2, 3],
        )
        result = NetHDX_ForwardPass()(features, dummy_config)
        expected = np.zeros(3)
        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-6)

    def test_known_values(self, known_contact_matrices, dummy_config):
        """Test with known contact matrices.
        Frame 1 row sums: [1.5, 2.0, 1.5]
        Frame 2 row sums: [1.5, 1.5, 1.0]
        Average row sums: [1.5, 1.75, 1.25]
        log10([1.5, 1.75, 1.25]): [0.176, 0.243, 0.097]
        """
        features = NetHDX_input_features(
            contact_matrices=known_contact_matrices,
            residue_ids=[1, 2, 3],
        )
        result = NetHDX_ForwardPass()(features, dummy_config)

        # Compute expected values manually
        avg_contacts = np.mean(known_contact_matrices, axis=0)  # (3, 3)
        row_sums = np.sum(avg_contacts, axis=1)  # [1.5, 1.75, 1.25]
        expected = np.log10(row_sums)

        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-5)

    def test_output_type(self, identity_contact_matrices, dummy_config):
        """Test that output is NetHDX_output_features."""
        features = NetHDX_input_features(
            contact_matrices=identity_contact_matrices,
            residue_ids=[1, 2, 3],
        )
        result = NetHDX_ForwardPass()(features, dummy_config)
        assert isinstance(result, NetHDX_output_features)
        assert result.k_ints is None

    def test_single_frame(self, dummy_config):
        """Test with single frame."""
        matrices = np.array([[[1.0, 0.5], [0.5, 1.0]]])
        features = NetHDX_input_features(
            contact_matrices=matrices,
            residue_ids=[1, 2],
        )
        result = NetHDX_ForwardPass()(features, dummy_config)
        # Row sums: [1.5, 1.5], log10: [0.176, 0.176]
        expected = np.log10([1.5, 1.5])
        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-5)

    def test_larger_network(self, dummy_config):
        """Test with larger contact matrix (5x5)."""
        # Create a simple 5x5 identity matrix
        matrices = np.array([np.eye(5)])
        features = NetHDX_input_features(
            contact_matrices=matrices,
            residue_ids=list(range(1, 6)),
        )
        result = NetHDX_ForwardPass()(features, dummy_config)
        # Row sums all = 1, log10(1) = 0
        expected = np.zeros(5)
        np.testing.assert_allclose(result.log_Pf, expected, rtol=1e-6)

    def test_zero_handling(self, dummy_config):
        """Test with matrix containing zeros.
        Important: log10(0) = -inf, but numpy handles this safely.
        """
        matrices = np.array([[[0.0, 0.0], [0.0, 0.0]]])
        features = NetHDX_input_features(
            contact_matrices=matrices,
            residue_ids=[1, 2],
        )
        result = NetHDX_ForwardPass()(features, dummy_config)
        # log10(0) results in -inf or warning
        assert len(result.log_Pf) == 2
