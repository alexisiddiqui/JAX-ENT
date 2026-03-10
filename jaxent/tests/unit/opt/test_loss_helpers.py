"""
Unit tests for loss helper functions.
Tests the low-level utility functions in jaxent/src/opt/loss/base.py
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.opt.loss.base import (
    apply_post_processing,
    apply_transforms,
    normalize_weights,
    pairwise_cosine_similarity,
    extract_upper_triangle,
    normalize_upper_triangle,
)


# ============================================================================
# Test apply_transforms
# ============================================================================


class TestApplyTransforms:
    """Test apply_transforms function."""

    def test_empty_chain(self):
        """Test empty transform chain (identity)."""
        array = jnp.array([1.0, 2.0, 3.0])
        result = apply_transforms(array, [])
        np.testing.assert_allclose(result, array, rtol=1e-6)

    def test_single_transform(self):
        """Test single transform: multiply by 2."""
        array = jnp.array([1.0, 2.0, 3.0])
        transform = lambda x: x * 2
        result = apply_transforms(array, [transform])
        expected = jnp.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_multiple_transforms(self):
        """Test multiple transforms applied in order.
        First: add 1, then multiply by 2
        [1, 2, 3] -> [2, 3, 4] -> [4, 6, 8]
        """
        array = jnp.array([1.0, 2.0, 3.0])
        transforms = [lambda x: x + 1, lambda x: x * 2]
        result = apply_transforms(array, transforms)
        expected = jnp.array([4.0, 6.0, 8.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_mean_centering_transform(self):
        """Test mean-centering transform."""
        array = jnp.array([1.0, 2.0, 3.0])
        mean_center = lambda x: x - jnp.mean(x)
        result = apply_transforms(array, [mean_center])
        expected = jnp.array([-1.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ============================================================================
# Test apply_post_processing
# ============================================================================


class TestApplyPostProcessing:
    """Test apply_post_processing function."""

    def test_post_mean_true(self):
        """Test post_mean=True: loss / length.
        loss=10.0, length=5 -> 2.0
        """
        loss = jnp.array(10.0)
        result = apply_post_processing(loss, length=5, post_mean=True)
        expected = 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_post_mean_false(self):
        """Test post_mean=False: loss unchanged."""
        loss = jnp.array(10.0)
        result = apply_post_processing(loss, length=5, post_mean=False)
        expected = 10.0
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_zero_length(self):
        """Test length=0: returns loss unchanged."""
        loss = jnp.array(10.0)
        result = apply_post_processing(loss, length=0, post_mean=True)
        # Should return loss since length=0
        np.testing.assert_allclose(result, loss, rtol=1e-6)

    def test_negative_length(self):
        """Test negative length: returns loss unchanged."""
        loss = jnp.array(10.0)
        result = apply_post_processing(loss, length=-5, post_mean=True)
        # Should return loss since length <= 0
        np.testing.assert_allclose(result, loss, rtol=1e-6)


# ============================================================================
# Test normalize_weights
# ============================================================================


class TestNormalizeWeights:
    """Test normalize_weights function."""

    def test_uniform_weights_normalized(self):
        """Test uniform weights after normalization."""
        weights = jnp.array([1.0, 1.0, 1.0])
        result = normalize_weights(weights, normalise=True, eps=1e-8, scale_eps=False)
        # After normalization: should sum to ~1
        np.testing.assert_allclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_normalise_false(self):
        """Test normalise=False: identity."""
        weights = jnp.array([1.0, 2.0, 3.0])
        result = normalize_weights(weights, normalise=False, eps=1e-8, scale_eps=False)
        np.testing.assert_allclose(result, weights, rtol=1e-6)

    def test_scale_eps_true(self):
        """Test scale_eps=True: epsilon is multiplied by n."""
        weights = jnp.array([1.0, 2.0])
        result_no_scale = normalize_weights(
            weights, normalise=True, eps=0.01, scale_eps=False
        )
        result_scale = normalize_weights(
            weights, normalise=True, eps=0.01, scale_eps=True
        )
        # With scale_eps=True, eps becomes 0.01 * 2 = 0.02
        # The denominators differ: (|w| + 0.01) vs (|w| + 0.02)
        # So results should differ when weights are not identical
        assert not np.allclose(result_no_scale, result_scale, rtol=1e-3)

    def test_negative_weights_abs(self):
        """Test negative weights get abs'd."""
        weights = jnp.array([-1.0, 2.0, -3.0])
        result = normalize_weights(weights, normalise=True, eps=1e-8, scale_eps=False)
        # All values should be positive after abs
        assert jnp.all(result >= 0)


# ============================================================================
# Test pairwise_cosine_similarity
# ============================================================================


class TestPairwiseCosineSimimilarity:
    """Test pairwise_cosine_similarity function."""

    def test_self_similarity(self):
        """Test self-similarity: 1 + clip(1) = 2 for unit vectors."""
        array = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = pairwise_cosine_similarity(array)
        # Self-similarity (diagonal) should be 2.0 for unit vectors
        # The last vector [1, 1] has norm sqrt(2), so similarity is 1 + clip(1) = 2
        np.testing.assert_allclose(jnp.diag(result), jnp.array([2.0, 2.0, 2.0]), rtol=1e-5)

    def test_orthogonal_vectors(self):
        """Test orthogonal vectors: 1 + 0 = 1."""
        array = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        result = pairwise_cosine_similarity(array)
        # Off-diagonal: 1 + clip(0) = 1
        off_diag = result[0, 1]
        np.testing.assert_allclose(off_diag, 1.0, rtol=1e-5)

    def test_empty_array(self):
        """Test empty array: (0, 0) shape."""
        array = jnp.empty((0, 2))
        result = pairwise_cosine_similarity(array)
        assert result.shape == (0, 0)

    def test_1d_array_reshape(self):
        """Test 1D array gets reshaped to (n, 1) then compared."""
        array = jnp.array([1.0, 2.0, 3.0])
        result = pairwise_cosine_similarity(array)
        # 1D array [1, 2, 3] gets reshaped to (3, 1), then similarity is computed
        # Result shape is (3, 3)
        assert result.shape == (3, 3)


# ============================================================================
# Test extract_upper_triangle
# ============================================================================


class TestExtractUpperTriangle:
    """Test extract_upper_triangle function."""

    def test_3x3_matrix(self):
        """Test 3x3 matrix: extract 3 elements from upper triangle."""
        matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        result = extract_upper_triangle(matrix)
        # Upper triangle (k=1): [2, 3, 6]
        expected = jnp.array([2.0, 3.0, 6.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_empty_matrix(self):
        """Test empty matrix: empty array."""
        matrix = jnp.empty((0, 0))
        result = extract_upper_triangle(matrix)
        assert len(result) == 0

    def test_2x2_matrix(self):
        """Test 2x2 matrix: 1 element."""
        matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = extract_upper_triangle(matrix)
        # Upper triangle: [2]
        expected = jnp.array([2.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ============================================================================
# Test normalize_upper_triangle
# ============================================================================


class TestNormalizeUpperTriangle:
    """Test normalize_upper_triangle function."""

    def test_normalize_true(self):
        """Test normalize=True: softmax applied."""
        upper_tri = jnp.array([1.0, 2.0, 3.0])
        result = normalize_upper_triangle(upper_tri, normalize=True)
        # Softmax should sum to 1
        np.testing.assert_allclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_normalize_false(self):
        """Test normalize=False: identity."""
        upper_tri = jnp.array([1.0, 2.0, 3.0])
        result = normalize_upper_triangle(upper_tri, normalize=False)
        np.testing.assert_allclose(result, upper_tri, rtol=1e-6)

    def test_empty_array_no_normalize(self):
        """Test empty array with normalize=True: returns empty."""
        upper_tri = jnp.array([])
        result = normalize_upper_triangle(upper_tri, normalize=True)
        assert len(result) == 0
