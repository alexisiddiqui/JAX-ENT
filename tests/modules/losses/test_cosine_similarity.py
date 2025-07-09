import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Import functions to test
from jaxent.src.opt.losses import (
    L1_frame_weight_consistency_loss,  # Add L1 loss import
    convex_KL_frame_weight_consistency_loss,
    corr_frame_weight_consistency_loss,
    cosine_frame_weight_consistency_loss,
    frame_weight_consistency_loss,
    jax_pairwise_cosine_similarity,
    normalised_frame_weight_consistency_loss,
)


# Create mock classes for testing
@dataclass
class SimulationParams:
    frame_weights: jnp.ndarray


@dataclass
class MockSimulation:
    params: SimulationParams
    outputs: Any = None


# Helper functions for computing similarity matrices with other libraries
def numpy_pairwise_cosine_similarity(array: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity using NumPy."""
    if array.ndim == 1:
        array = array.reshape(1, -1)

    # Compute dot products
    dot_products = np.matmul(array, array.T)

    # Compute norms
    norms = np.sqrt(np.sum(array**2, axis=1))

    # Create a 2D grid of norm products
    norm_products = np.outer(norms, norms)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    norm_products = np.maximum(norm_products, epsilon)

    # Compute cosine similarities
    similarity_matrix = dot_products / norm_products

    # Transform to distance (1 - cosine similarity)
    similarity_matrix = 1 + np.clip(similarity_matrix, -1.0, 1.0)

    return similarity_matrix


def scipy_pairwise_cosine_similarity(array: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity using SciPy."""
    if array.ndim == 1:
        array = array.reshape(1, -1)

    # Using sklearn's cosine_similarity which is 1 - distance
    similarity = 1 + cosine_similarity(array)

    # Transform to distance (1 - cosine similarity)
    return similarity


# Add a normalization helper function
def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to sum to 1.

    Args:
        weights: Input weight matrix

    Returns:
        Normalized weights
    """
    epsilon = 1e-5
    weights = weights + epsilon  # Avoid division by zero
    return weights / np.sum(weights)


# Visualization helper functions
def plot_similarity_heatmaps(
    matrices: Dict[str, np.ndarray], title: str, save_path: Optional[str] = None
):
    """Plot heatmaps for multiple similarity matrices."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, matrix) in zip(axes, matrices.items()):
        sns.heatmap(matrix, ax=ax, cmap="viridis", annot=True, fmt=".2f")
        ax.set_title(f"{name}")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_loss_vs_distance(
    true_distances: np.ndarray,
    frame_weights: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
):
    """Plot loss values against true distances for different loss functions."""
    # Create a range of perturbations for frame weights
    perturbations = np.linspace(0, 1, 20)  # Changed from 0, 2 to 0, 1

    # Initialize arrays to store loss values
    l2_losses = []
    l1_losses = []  # Add array for L1 losses
    normalized_losses = []
    kl_losses = []
    cosine_losses = []
    corr_losses = []

    # For each perturbation, calculate loss values
    for p in perturbations:
        # Create perturbed weights
        perturbed_weights = frame_weights + p * np.random.randn(*frame_weights.shape)

        # Normalize perturbed weights
        perturbed_weights = normalize_weights(perturbed_weights)

        # Create model with perturbed weights
        model = MockSimulation(params=SimulationParams(frame_weights=jnp.array(perturbed_weights)))

        # Calculate loss values
        l2_loss, _ = frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
        l1_loss, _ = L1_frame_weight_consistency_loss(
            model, jnp.array(true_distances), 0
        )  # Add L1 loss
        norm_loss, _ = normalised_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
        kl_loss, _ = convex_KL_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
        cosine_loss, _ = cosine_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
        corr_loss, _ = corr_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)

        # Append loss values
        l2_losses.append(float(l2_loss))
        l1_losses.append(float(l1_loss))  # Append L1 loss
        normalized_losses.append(float(norm_loss))
        kl_losses.append(float(kl_loss))
        cosine_losses.append(float(cosine_loss))
        corr_losses.append(float(corr_loss))

    # Plot loss values
    plt.figure(figsize=(10, 6))
    plt.plot(perturbations, l2_losses, label="L2 Loss", marker="o")
    plt.plot(perturbations, l1_losses, label="L1 Loss", marker="+")  # Plot L1 loss
    plt.plot(perturbations, normalized_losses, label="Normalized Loss", marker="s")
    plt.plot(perturbations, kl_losses, label="KL Loss", marker="^")
    plt.plot(perturbations, cosine_losses, label="Cosine Loss", marker="d")
    plt.plot(perturbations, corr_losses, label="Correlation Loss", marker="x")
    plt.xlabel("Perturbation Magnitude")
    plt.ylabel("Loss Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_similarity_with_perturbations(
    base_weights: np.ndarray, perturbation_levels: List[float], save_path: Optional[str] = None
):
    """
    Plot similarity matrices with increasing perturbation levels.

    Args:
        base_weights: Base weight matrix (already normalized)
        perturbation_levels: List of perturbation magnitudes
        save_path: Path to save the visualization
    """
    n_levels = len(perturbation_levels)
    fig, axes = plt.subplots(1, n_levels, figsize=(n_levels * 4, 4))

    if n_levels == 1:
        axes = [axes]

    # For each perturbation level
    for i, level in enumerate(perturbation_levels):
        # Create perturbed weights
        perturbed_weights = base_weights + level * np.random.randn(*base_weights.shape)
        # Normalize perturbed weights
        perturbed_weights = normalize_weights(perturbed_weights)

        # Calculate similarity matrix
        similarity = numpy_pairwise_cosine_similarity(perturbed_weights)

        # Plot heatmap
        sns.heatmap(
            similarity,
            ax=axes[i],
            cmap="viridis",
            annot=True if base_weights.shape[0] <= 7 else False,
            fmt=".2f",
        )
        axes[i].set_title(f"Perturbation: {level:.2f}")

    plt.suptitle("Similarity Matrices with Increasing Perturbations")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# Test cases
def test_pairwise_cosine_similarity_implementation():
    """Test the implementation of pairwise cosine similarity against NumPy and SciPy."""
    print("\nTesting pairwise cosine similarity implementation...")

    # Generate random data for testing
    np.random.seed(42)
    test_data = np.random.randn(5, 10)  # 5 samples, 10 features

    # Normalize the test data to sum to 1 for each row
    test_data = normalize_weights(test_data)

    # Convert to JAX array
    jax_data = jnp.array(test_data)

    # Compute similarity matrices
    jax_similarity = np.array(jax_pairwise_cosine_similarity(jax_data))
    numpy_similarity = numpy_pairwise_cosine_similarity(test_data)
    scipy_similarity = scipy_pairwise_cosine_similarity(test_data)

    # Plot heatmaps for visual comparison
    plot_similarity_heatmaps(
        {"JAX": jax_similarity, "NumPy": numpy_similarity, "SciPy": scipy_similarity},
        "Pairwise Cosine Similarity Comparison",
        save_path="tests/_plots/cosine_similarity_comparison.png",
    )

    # Check that the implementations are close
    np.testing.assert_allclose(jax_similarity, numpy_similarity, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(jax_similarity, scipy_similarity, rtol=1e-5, atol=1e-5)

    print("✓ JAX implementation matches NumPy and SciPy implementations")
    return jax_similarity, test_data


def test_pairwise_cosine_similarity_edge_cases():
    """Test edge cases for pairwise cosine similarity."""
    print("\nTesting edge cases for pairwise cosine similarity...")

    # Test case: 1D vector
    vector = jnp.array([1.0, 2.0, 3.0])
    # Normalize the vector
    vector = vector / jnp.sum(vector)
    similarity_1d = jax_pairwise_cosine_similarity(vector)
    print(f"1D vector similarity shape: {similarity_1d.shape}")
    assert similarity_1d.shape == (1, 1), "1D vector should produce 1x1 matrix"

    # Test case: Empty array
    empty = jnp.array([])
    empty_result = jax_pairwise_cosine_similarity(empty)
    print(f"Empty array result shape: {empty_result.shape}")

    # Test case: Zero vectors
    zeros = jnp.zeros((3, 5))
    # Add small epsilon to avoid all zeros
    zeros = zeros + 1e-8
    # Normalize
    zeros = zeros / jnp.sum(zeros, axis=1, keepdims=True)
    with jax.disable_jit():  # Disable JIT to catch warnings
        zero_result = jax_pairwise_cosine_similarity(zeros)
    print(f"Zero vectors result shape: {zero_result.shape}")

    # Test case: Orthogonal and parallel vectors
    vectors = jnp.array(
        [
            [1.0, 0.0],  # Vector pointing along x-axis
            [0.0, 1.0],  # Vector pointing along y-axis (orthogonal to first)
            [2.0, 0.0],  # Vector parallel to first vector
        ]
    )
    # Normalize the vectors
    vectors = vectors / jnp.sum(vectors, axis=1, keepdims=True)
    ortho_result = jax_pairwise_cosine_similarity(vectors)
    print(f"Orthogonal vectors result:\n{ortho_result}")

    # Check orthogonal vectors (should have similarity 1.0)
    assert np.isclose(ortho_result[0, 1], 1.0), "Orthogonal vectors should have similarity 1.0"

    # Check parallel vectors (should have similarity 0.0)
    assert np.isclose(ortho_result[0, 2], 2.0), "Parallel vectors should have similarity 1.0"

    print("✓ Edge cases handled correctly")


def test_loss_functions_with_true_distances():
    """Test loss functions against true distances."""
    print("\nTesting loss functions against true distances...")

    # Generate random vectors to simulate frame weights
    np.random.seed(42)
    n_frames = 10
    n_features = 5

    # Create original frame weights
    frame_weights = np.random.rand(n_frames, n_features)
    # Normalize weights to sum to 1 for each row
    frame_weights = normalize_weights(frame_weights)

    # Calculate true distances based on original frame weights
    true_distances = numpy_pairwise_cosine_similarity(frame_weights)

    # Create model with the same weights (should give minimal loss)
    model = MockSimulation(params=SimulationParams(frame_weights=jnp.array(frame_weights)))

    # Calculate losses
    l2_loss, _ = frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
    l1_loss, _ = L1_frame_weight_consistency_loss(
        model, jnp.array(true_distances), 0
    )  # Add L1 loss
    norm_loss, _ = normalised_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
    kl_loss, _ = convex_KL_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
    cosine_loss, _ = cosine_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
    corr_loss, _ = corr_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)

    print(f"L2 Loss (matching weights): {l2_loss}")
    print(f"L1 Loss (matching weights): {l1_loss}")  # Print L1 loss
    print(f"Normalized Loss (matching weights): {norm_loss}")
    print(f"KL Loss (matching weights): {kl_loss}")
    print(f"Cosine Loss (matching weights): {cosine_loss}")
    print(f"Correlation Loss (matching weights): {corr_loss}")

    # Test with perturbed weights (should give higher loss)
    perturbed_weights = frame_weights + 0.5 * np.random.randn(*frame_weights.shape)
    # Normalize perturbed weights
    perturbed_weights = normalize_weights(perturbed_weights)

    perturbed_model = MockSimulation(
        params=SimulationParams(frame_weights=jnp.array(perturbed_weights))
    )

    # Calculate losses with perturbed weights
    perturbed_l2_loss, _ = frame_weight_consistency_loss(
        perturbed_model, jnp.array(true_distances), 0
    )
    perturbed_l1_loss, _ = L1_frame_weight_consistency_loss(
        perturbed_model, jnp.array(true_distances), 0
    )  # Add perturbed L1 loss
    perturbed_norm_loss, _ = normalised_frame_weight_consistency_loss(
        perturbed_model, jnp.array(true_distances), 0
    )
    perturbed_kl_loss, _ = convex_KL_frame_weight_consistency_loss(
        perturbed_model, jnp.array(true_distances), 0
    )
    perturbed_cosine_loss, _ = cosine_frame_weight_consistency_loss(
        perturbed_model, jnp.array(true_distances), 0
    )
    perturbed_corr_loss, _ = corr_frame_weight_consistency_loss(
        perturbed_model, jnp.array(true_distances), 0
    )

    print(f"L2 Loss (perturbed weights): {perturbed_l2_loss}")
    print(f"L1 Loss (perturbed weights): {perturbed_l1_loss}")  # Print perturbed L1 loss
    print(f"Normalized Loss (perturbed weights): {perturbed_norm_loss}")
    print(f"KL Loss (perturbed weights): {perturbed_kl_loss}")
    print(f"Cosine Loss (perturbed weights): {perturbed_cosine_loss}")
    print(f"Correlation Loss (perturbed weights): {perturbed_corr_loss}")

    # Plot loss vs perturbation magnitude
    plot_loss_vs_distance(
        true_distances,
        frame_weights,
        "Loss Functions vs Perturbation Magnitude",
        save_path="tests/_plots/loss_vs_perturbation.png",
    )

    print("✓ Loss functions correctly respond to true distances")


def test_visualize_loss_landscapes():
    """Visualize the landscape of different loss functions."""
    print("\nVisualizing loss function landscapes...")

    # Generate random vectors to simulate frame weights
    np.random.seed(42)
    n_frames = 5
    n_features = 3

    # Create original frame weights
    frame_weights = np.random.rand(n_frames, n_features)
    # Normalize weights to sum to 1 for each row
    frame_weights = normalize_weights(frame_weights)

    # Calculate true distances based on original frame weights
    true_distances = numpy_pairwise_cosine_similarity(frame_weights)

    # Generate a grid of perturbed weights
    n_points = 50
    perturbation_x = np.linspace(-0.25, 0.25, n_points)  # Changed from -1, 1 to 0, 1
    perturbation_y = np.linspace(-0.25, 0.25, n_points)  # Changed from -1, 1 to 0, 1

    # Initialize arrays to store loss values
    l2_loss_grid = np.zeros((n_points, n_points))
    l1_loss_grid = np.zeros((n_points, n_points))  # Add L1 loss grid
    norm_loss_grid = np.zeros((n_points, n_points))
    kl_loss_grid = np.zeros((n_points, n_points))
    cosine_loss_grid = np.zeros((n_points, n_points))
    corr_loss_grid = np.zeros((n_points, n_points))

    # For each point in the grid, calculate loss values
    for i, px in enumerate(perturbation_x):
        for j, py in enumerate(perturbation_y):
            # Create perturbed weights using two principal components
            perturbed_weights = frame_weights.copy()
            perturbed_weights[:, 0] += px
            perturbed_weights[:, 1] += py

            # Normalize perturbed weights
            perturbed_weights = normalize_weights(perturbed_weights)

            # Create model with perturbed weights
            model = MockSimulation(
                params=SimulationParams(frame_weights=jnp.array(perturbed_weights))
            )

            # Calculate loss values
            l2_loss, _ = frame_weight_consistency_loss(model, jnp.array(true_distances), 0)
            l1_loss, _ = L1_frame_weight_consistency_loss(
                model, jnp.array(true_distances), 0
            )  # Add L1 loss
            norm_loss, _ = normalised_frame_weight_consistency_loss(
                model, jnp.array(true_distances), 0
            )
            kl_loss, _ = convex_KL_frame_weight_consistency_loss(
                model, jnp.array(true_distances), 0
            )
            cosine_loss, _ = cosine_frame_weight_consistency_loss(
                model, jnp.array(true_distances), 0
            )
            corr_loss, _ = corr_frame_weight_consistency_loss(model, jnp.array(true_distances), 0)

            # Store loss values
            l2_loss_grid[i, j] = float(l2_loss)
            l1_loss_grid[i, j] = float(l1_loss)  # Store L1 loss
            norm_loss_grid[i, j] = float(norm_loss)
            kl_loss_grid[i, j] = float(kl_loss)
            cosine_loss_grid[i, j] = float(cosine_loss)
            corr_loss_grid[i, j] = float(corr_loss)

    # Plot loss landscapes
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))  # Increased to 6 subplots

    # L2 Loss
    im1 = axes[0].imshow(
        l2_loss_grid, cmap="viridis", extent=[-0.25, 0.25, -0.25, 0.25], origin="lower"
    )  # Changed extent
    axes[0].set_title("L2 Loss Landscape")
    axes[0].set_xlabel("Perturbation X")
    axes[0].set_ylabel("Perturbation Y")
    plt.colorbar(im1, ax=axes[0])

    # L1 Loss
    im2 = axes[1].imshow(
        l1_loss_grid,
        cmap="viridis",
        extent=[-0.25, 0.25, -0.25, 0.25],
        origin="lower",  # Changed extent
    )
    axes[1].set_title("L1 Loss Landscape")
    axes[1].set_xlabel("Perturbation X")
    axes[1].set_ylabel("Perturbation Y")
    plt.colorbar(im2, ax=axes[1])

    # Normalized Loss
    im3 = axes[2].imshow(
        norm_loss_grid, cmap="viridis", extent=[-0.25, 0.25, -0.25, 0.25], origin="lower"
    )  # Changed extent
    axes[2].set_title("Normalized Loss Landscape")
    axes[2].set_xlabel("Perturbation X")
    axes[2].set_ylabel("Perturbation Y")
    plt.colorbar(im3, ax=axes[2])

    # KL Loss
    im4 = axes[3].imshow(
        kl_loss_grid, cmap="viridis", extent=[-0.25, 0.25, -0.25, 0.25], origin="lower"
    )  # Changed extent
    axes[3].set_title("KL Loss Landscape")
    axes[3].set_xlabel("Perturbation X")
    axes[3].set_ylabel("Perturbation Y")
    plt.colorbar(im4, ax=axes[3])

    # Cosine Loss
    im5 = axes[4].imshow(
        cosine_loss_grid, cmap="viridis", extent=[-0.25, 0.25, -0.25, 0.25], origin="lower"
    )  # Changed extent
    axes[4].set_title("Cosine Loss Landscape")
    axes[4].set_xlabel("Perturbation X")
    axes[4].set_ylabel("Perturbation Y")
    plt.colorbar(im5, ax=axes[4])

    # Correlation Loss
    im6 = axes[5].imshow(
        corr_loss_grid, cmap="viridis", extent=[-0.25, 0.25, -0.25, 0.25], origin="lower"
    )  # Changed extent
    axes[5].set_title("Correlation Loss Landscape")
    axes[5].set_xlabel("Perturbation X")
    axes[5].set_ylabel("Perturbation Y")
    plt.colorbar(im6, ax=axes[5])

    plt.tight_layout()
    os.makedirs("tests/_plots", exist_ok=True)
    plt.savefig("tests/_plots/loss_landscapes.png")

    print("✓ Loss landscapes visualized and saved to 'tests/_plots/loss_landscapes.png'")


def test_similarity_matrices_with_perturbations():
    """Test how similarity matrices change with increasing perturbations."""
    print("\nVisualizing similarity matrices with increasing perturbations...")

    # Generate random vectors
    np.random.seed(42)
    n_frames = 6
    n_features = 4

    # Create base weights
    base_weights = np.random.rand(n_frames, n_features)
    # Normalize weights to sum to 1 for each row
    base_weights = normalize_weights(base_weights)

    # Define perturbation levels
    perturbation_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]  # Changed to remove 2.0 and add 0.7

    # Visualize similarity matrices with different perturbation levels
    plot_similarity_with_perturbations(
        base_weights,
        perturbation_levels,
        save_path="tests/_plots/similarity_with_perturbations.png",
    )

    print("✓ Similarity matrices with perturbations visualized and saved")


def create_output_dir():
    """Create output directory for plots if it doesn't exist."""
    output_dir = "tests/_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


if __name__ == "__main__":
    # Create output directory
    output_dir = create_output_dir()

    print("Running tests for pairwise cosine similarity and loss functions...")

    # Run implementation test and get the test data
    similarity_matrix, test_data = test_pairwise_cosine_similarity_implementation()

    # Run edge case tests
    test_pairwise_cosine_similarity_edge_cases()

    # Test loss functions
    test_loss_functions_with_true_distances()

    # Visualize loss landscapes
    test_visualize_loss_landscapes()

    # Test similarity matrices with perturbations
    test_similarity_matrices_with_perturbations()

    print("\nAll tests completed successfully!")
