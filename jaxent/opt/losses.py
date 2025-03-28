import jax.numpy as jnp
import optax  # Import optax library for the convex_kl_divergence function
from jax import Array
from optax.losses import safe_softmax_cross_entropy

from jaxent.data.loader import ExpD_Dataloader
from jaxent.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.models.core import Simulation
from jaxent.types import InitialisedSimulation
from jaxent.types.HDX import HDX_peptide, HDX_protection_factor


def hdx_pf_l2_loss(
    model: Simulation, dataset: ExpD_Dataloader[HDX_protection_factor], prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the L2 loss between the predicted and experimental data.
    """

    # Calculate the predicted data
    predictions = (
        model.outputs
    )  # TODO: find a way to move the forward call to outside the loss function
    pred_pf = jnp.array(predictions[prediction_index].log_Pf).reshape(-1)  # Flatten to 1D

    pred_pf = apply_sparse_mapping(dataset.train.residue_feature_ouput_mapping, pred_pf)
    true_pf = dataset.train.y_true.reshape(-1)  # Flatten to 1D

    # print(predictions[0].log_Pf)
    # Calculate the L2 loss
    train_loss = jnp.mean((pred_pf - true_pf) ** 2)
    # print(loss)
    # average the loss over the length of the dataset

    pred_pf = jnp.array(predictions[prediction_index].log_Pf).reshape(-1)  # Flatten to 1D

    pred_pf = apply_sparse_mapping(dataset.val.residue_feature_ouput_mapping, pred_pf)

    true_pf = dataset.val.y_true.reshape(-1)  # Flatten to 1D

    # Calculate the L2 loss
    val_loss = jnp.mean((pred_pf - true_pf) ** 2)
    # print(loss)
    # average the loss over the length of the dataset

    return train_loss, val_loss


def hdx_pf_mae_loss(
    model: Simulation, dataset: ExpD_Dataloader[HDX_peptide], prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the mae loss between the predicted and experimental data.
    """

    # Calculate the predicted data

    # Calculate the predicted data
    predictions = (
        model.outputs
    )  # TODO: find a way to move the forward call to outside the loss function
    pred_pf = jnp.array(predictions[prediction_index].log_Pf).reshape(-1)  # Flatten to 1D

    pred_pf = apply_sparse_mapping(dataset.train.residue_feature_ouput_mapping, pred_pf)
    true_pf = dataset.train.y_true.reshape(-1)  # Flatten to 1D

    # print(predictions[0].log_Pf)
    # Calculate the L2 loss
    train_loss = jnp.mean(jnp.abs(pred_pf - true_pf) ** 1)
    # print(loss)
    # average the loss over the length of the dataset
    pred_pf = jnp.array(predictions[prediction_index].log_Pf).reshape(-1)  # Flatten to 1D

    pred_pf = apply_sparse_mapping(dataset.val.residue_feature_ouput_mapping, pred_pf)

    true_pf = dataset.val.y_true.reshape(-1)  # Flatten to 1D

    # Calculate the L2 loss
    val_loss = jnp.mean(jnp.abs(pred_pf - true_pf) ** 1)

    # average the loss over the length of the dataset

    return train_loss, val_loss


def max_entropy_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    simulation_weights = jnp.abs(model.params.frame_weights)
    simulation_weights = simulation_weights / jnp.sum(simulation_weights)
    prior_frame_weights = jnp.abs(dataset.frame_weights)

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    loss = jnp.asarray(safe_softmax_cross_entropy(jnp.log(simulation_weights), prior_frame_weights))
    # print(loss)
    return loss, loss


def maxent_convexKL_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    epsilon = 1e-5

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    loss = optax.losses.convex_kl_divergence(
        log_predictions=jnp.log(simulation_weights),
        targets=prior_frame_weights,
    )

    return loss, loss


def sparse_max_entropy_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    mask_indices = jnp.where(model.params.frame_mask > 0.5)
    simulation_weights = jnp.abs(model.params.frame_weights[mask_indices])
    simulation_weights = simulation_weights / jnp.sum(simulation_weights)
    prior_frame_weights = jnp.abs(dataset.frame_weights[mask_indices])

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    loss = jnp.asarray(safe_softmax_cross_entropy(jnp.log(simulation_weights), prior_frame_weights))
    # print(loss)
    return loss, loss


def mask_L0_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    frame_masks = model.params.frame_mask

    loss = jnp.sum(frame_masks)

    return loss, loss


def hdx_uptake_l2_loss(
    model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the L2 loss between the predicted and experimental data for HDX uptake.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values

    Returns:
        Tuple of train and validation losses
    """

    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]

    # Get the true uptake data
    # true_uptake = dataset.y_true

    # Compute train and validation losses
    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true uptake for this timepoint
            true_uptake_timepoint = y_true[timepoint_idx, :]

            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            # This handles cases where not all residues are measured
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_uptake_timepoint

            # Compute L2 loss for this timepoint
            timepoint_loss = jnp.mean((pred_mapped - true_mapped) ** 2)

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss)

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)

    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdx_uptake_l1_loss(
    model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the L1 loss between the predicted and experimental data for HDX uptake.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values

    Returns:
        Tuple of train and validation losses
    """

    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]

    # Get the true uptake data
    # true_uptake = dataset.y_true

    # Compute train and validation losses
    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true uptake for this timepoint
            true_uptake_timepoint = y_true[timepoint_idx, :]

            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            # This handles cases where not all residues are measured
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_uptake_timepoint

            # Compute L2 loss for this timepoint
            timepoint_loss = jnp.mean(jnp.abs((pred_mapped - true_mapped) ** 1))

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss)

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)

    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdx_uptake_mean_centred_l1_loss(
    model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the mean-centered L1 loss between the predicted and experimental data for HDX uptake.
    This loss centers both predictions and targets around their means before computing the L1 norm.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values
        prediction_index: Index of the prediction to use

    Returns:
        Tuple of train and validation losses
    """
    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]

    # Compute train and validation losses
    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true uptake for this timepoint
            true_uptake_timepoint = y_true[timepoint_idx, :]

            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_uptake_timepoint

            # Center predictions and targets around their means
            pred_mean = jnp.mean(pred_mapped)
            true_mean = jnp.mean(true_mapped)

            pred_centered = pred_mapped - pred_mean
            true_centered = true_mapped - true_mean

            # Compute L1 loss for this timepoint
            timepoint_loss = jnp.mean(jnp.abs(pred_centered - true_centered))

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss)

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdx_uptake_monotonicity_loss(model: Simulation, dataset: None, prediction_index: int) -> Array:
    """
    Calculate the monotonicity loss for HDX uptake predictions.
    Penalizes violations of monotonic increase in time using squared penalties.

    Args:
        model: Simulation object containing the predictions

    Returns:
        Array: The computed monotonicity loss
    """
    # Calculate the predicted data
    predictions = model.outputs[prediction_index]
    # Get the uptake predictions and reshape if needed
    # Assuming predictions[2] contains uptake data with shape (peptides, timepoints)
    deut = jnp.array(predictions.uptake)

    # Calculate differences between adjacent timepoints
    time_diffs = deut[:, 1:] - deut[:, :-1]

    # Use JAX's equivalent of torch.relu for negative differences
    # This penalizes any decrease in deuteration over time
    violations = jnp.maximum(-time_diffs, 0)

    # Square the violations and take the mean
    # If there are no elements, return 0
    loss = jnp.where(time_diffs.size > 0, jnp.mean(violations**2), jnp.array(0.0))

    return loss


def jax_pairwise_cosine_similarity(array: Array) -> Array:
    """
    Calculate the pairwise cosine similarity between vectors in an array.

    Args:
        array: JAX array of shape (n_samples, n_features) or (n_features,)
            If 1D, it's treated as a single sample with n_features

    Returns:
        A square matrix of shape (n_samples, n_samples) containing
        pairwise cosine similarities between the samples
    """
    # If input is 1D, reshape to 2D
    if array.ndim == 1:
        array = array.reshape(1, -1)

    # Handle empty array
    if array.shape[0] == 0 or array.shape[1] == 0:
        return jnp.empty((array.shape[0], array.shape[0]))

    # Compute dot products
    dot_products = jnp.matmul(array, array.T)

    # Compute norms
    norms = jnp.sqrt(jnp.sum(array**2, axis=1))

    # Create a 2D grid of norm products
    norm_products = jnp.outer(norms, norms)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    norm_products = jnp.maximum(norm_products, epsilon)

    # Compute cosine similarities
    similarity_matrix = dot_products / norm_products

    # Ensure values are in the valid range for cosine similarity [-1, 1]
    similarity_matrix = jnp.clip(similarity_matrix, -1.0, 1.0)

    return similarity_matrix


def frame_weight_consistency_loss(
    model: Simulation, dataset: Array, prediction_index: int
) -> tuple[Array, Array]:
    """
    Computes and compares graphs of the pairwise distances between ensembles.
    One graph is constructed using the features/structures, another using the weights.
    TODO how are weights compared between each other?
    The loss is the L1 distance/Cosine between the two graphs.
    """

    weights = model.params.frame_weights

    # Calculate the pairwise cosine similarity between the weights
    weight_similarity = jax_pairwise_cosine_similarity(weights)

    # Compute the L21 distance between the two similarity matrices
    l1_distance = jnp.mean((dataset - weight_similarity) ** 1)

    return l1_distance, l1_distance


def HDX_uptake_MAE_loss(
    model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the MAE (Mean Absolute Error) loss between the predicted and experimental data for HDX uptake.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values
        prediction_index: Index of the prediction to use

    Returns:
        Tuple of train and validation losses
    """

    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]

    # Compute train and validation losses
    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true uptake for this timepoint
            true_uptake_timepoint = y_true[timepoint_idx, :]

            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            # This handles cases where not all residues are measured
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_uptake_timepoint

            # Compute MAE loss for this timepoint
            timepoint_loss = jnp.mean(jnp.abs(pred_mapped - true_mapped))

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss / y_true.shape[0])

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)

    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def HDX_uptake_KL_loss(
    model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the KL divergence loss between predicted and experimental HDX uptake data.
    Uses optax.losses.kl_divergence to measure information gain between distributions.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values
        prediction_index: Index of the prediction to use
    Returns:
        Tuple of train and validation losses
    """
    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]
    # Small epsilon to avoid numerical issues
    epsilon = 1e-5

    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true uptake for this timepoint
            true_uptake_timepoint = y_true[timepoint_idx, :]
            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_uptake_timepoint

            # Normalize the true values to create proper probability distribution
            true_norm = true_mapped + epsilon
            targets = true_norm / jnp.sum(true_norm)

            # Normalize the predicted values and convert to log space
            pred_norm = pred_mapped + epsilon
            pred_probs = pred_norm / jnp.sum(pred_norm)
            log_predictions = jnp.log(pred_probs)

            # Use optax's kl_divergence
            timepoint_loss = optax.losses.kl_divergence(
                log_predictions=log_predictions,
                targets=targets,
            )

            # Sum the loss values (if needed based on the shape)
            timepoint_loss = jnp.sum(timepoint_loss)

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return total_loss / y_true.shape[0]  # Consider adding division by timepoints

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def HDX_uptake_convex_KL_loss(
    model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[Array, Array]:
    """
    Calculate the KL divergence loss between predicted and experimental HDX uptake data.
    Uses optax.losses.kl_divergence to measure information gain between distributions.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values
        prediction_index: Index of the prediction to use
    Returns:
        Tuple of train and validation losses
    """
    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]
    # Small epsilon to avoid numerical issues
    epsilon = 1e-5

    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true uptake for this timepoint
            true_uptake_timepoint = y_true[timepoint_idx, :]
            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_uptake_timepoint

            # Normalize the true values to create proper probability distribution
            true_norm = true_mapped + epsilon
            targets = true_norm / jnp.sum(true_norm)

            # Normalize the predicted values and convert to log space
            pred_norm = pred_mapped + epsilon
            pred_probs = pred_norm / jnp.sum(pred_norm)
            log_predictions = jnp.log(pred_probs)

            # Use optax's kl_divergence
            timepoint_loss = optax.losses.convex_kl_divergence(
                log_predictions=log_predictions,
                targets=targets,
            )

            # Sum the loss values (if needed based on the shape)
            timepoint_loss = jnp.sum(timepoint_loss)

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return total_loss / y_true.shape[0]  # Consider adding division by timepoints

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss
