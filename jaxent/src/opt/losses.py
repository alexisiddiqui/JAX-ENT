import jax
import jax.numpy as jnp
import optax  # Import optax library for the convex_kl_divergence function
from jax import Array
from optax.losses import safe_softmax_cross_entropy

from jaxent.src.custom_types import InitialisedSimulation
from jaxent.src.custom_types.HDX import HDX_peptide, HDX_protection_factor
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation


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
    epsilon = 1e-8

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    loss = jnp.asarray(safe_softmax_cross_entropy(jnp.log(simulation_weights), prior_frame_weights))
    # print(loss)
    return loss, loss


def maxent_convexKL_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    epsilon = 1e-20

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)
    num_frames = prior_frame_weights.shape[0]

    loss = optax.losses.convex_kl_divergence(
        log_predictions=jnp.log(simulation_weights),
        targets=prior_frame_weights,
    ) / (num_frames)
    # loss = loss - jnp.log(num_frames)
    return loss, loss


def maxent_JSD_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    epsilon = 1e-20

    simulation_weights = jnp.clip(jnp.abs(model.params.frame_weights), a_min=epsilon)

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    prior_frame_weights = jnp.abs(dataset.frame_weights)

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)
    num_frames = prior_frame_weights.shape[0]

    # Calculate the midpoint distribution M
    M = (simulation_weights + prior_frame_weights) / 2

    # Compute KL divergences against M
    kl_sim = optax.losses.convex_kl_divergence(
        log_predictions=jnp.log(M), targets=simulation_weights
    )
    kl_prior = optax.losses.convex_kl_divergence(
        log_predictions=jnp.log(M), targets=prior_frame_weights
    )

    # JSD is the average of the two KL divergences
    jsd = (kl_sim + kl_prior) / 2
    loss = jsd / num_frames  # Normalize by number of frames

    return loss, loss


def maxent_W1_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    epsilon = 1e-10

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    # prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    # prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    loss = jnp.mean(jnp.abs(simulation_weights - jnp.zeros_like(simulation_weights)))
    # num_frames = prior_frame_weights.shape[0]
    # loss = loss - jnp.log(num_frames)
    return loss, loss


def maxent_ESS_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    epsilon = 1e-8

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    # prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    # prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    # Calculate the effective sample size (ESS)
    ess = 1 / jnp.sum((simulation_weights) ** 2)
    # scale the ESS by the number of frames
    n_frames = simulation_weights.shape[0]
    ess_scaled = ess / n_frames
    # clip to epsilon and 1
    ess_scaled = jnp.clip(ess_scaled, epsilon, 1.0)

    # Calculate the loss as 1 - ESS
    loss = 1 - ess_scaled

    return loss, loss


def minent_ESS_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    epsilon = 1e-8

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    # prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    # prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    # Calculate the effective sample size (ESS)
    ess = 1 / jnp.sum((simulation_weights) ** 2)
    # scale the ESS by the number of frames
    n_frames = simulation_weights.shape[0]
    ess_scaled = ess / n_frames
    # clip to epsilon and 1
    ess_scaled = jnp.clip(ess_scaled, epsilon, 1.0)

    # Calculate the loss as 1 - ESS
    loss = ess_scaled

    return loss, loss


def maxent_L2_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    """
    Calculates the L2 penalty of the simulation weights compared to the prior weights. Scaled by the number of frames, squared.
    """
    epsilon = 1e-10

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    n_frames = simulation_weights.shape[0]

    loss = jnp.mean((jnp.abs(simulation_weights - prior_frame_weights)) ** 2)

    return loss, loss


def maxent_L1_loss(
    model: InitialisedSimulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    """
    Calculates the L2 penalty of the simulation weights compared to the prior weights. Scaled by the number of frames, squared.
    """
    epsilon = 1e-10

    simulation_weights = jnp.abs(model.params.frame_weights) + epsilon

    simulation_weights = simulation_weights / jnp.sum(simulation_weights)

    prior_frame_weights = jnp.abs(dataset.frame_weights) + epsilon

    prior_frame_weights = prior_frame_weights / jnp.sum(prior_frame_weights)

    n_frames = simulation_weights.shape[0]

    loss = jnp.mean((jnp.abs(simulation_weights - prior_frame_weights)) ** 1)

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


def hdx_uptake_abs_loss(
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
            timepoint_loss = jnp.abs(jnp.mean((pred_mapped) - jnp.mean((true_mapped)) ** 1))

            # Accumulate loss
            total_loss += timepoint_loss / true_uptake_timepoint.shape[0]

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
            total_loss += timepoint_loss / true_uptake_timepoint.shape[0]

        # Average loss across timepoints
        return jnp.asarray(total_loss)

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdx_uptake_mean_centred_l2_loss(
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
            timepoint_loss = jnp.mean(jnp.abs(pred_centered - true_centered) ** 2)

            # Accumulate loss
            total_loss += timepoint_loss / true_uptake_timepoint.shape[0]

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
    # Ensure values are in a better range (0 to 2)
    similarity_matrix = 1 + jnp.clip(similarity_matrix, -1.0, 1.0)

    # Return similarity or distance as needed
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

    # Compute the L2 distance between the two similarity matrices
    l2_distance = jnp.mean((jnp.abs(dataset - weight_similarity)) ** 2)

    return l2_distance, l2_distance


def exp_frame_weight_consistency_loss(
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

    # Compute the L1 distance between the two similarity matrices
    l1_distance = jnp.mean(-1 + (jnp.exp((jnp.abs(dataset - weight_similarity))) ** 2))
    # l1_distance = jnp.log(jnp.exp(l1_distance))

    return l1_distance, l1_distance


def L1_frame_weight_consistency_loss(
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

    # Compute the L1 distance between the two similarity matrices
    l1_distance = jnp.mean(-1 + (jnp.exp((jnp.abs(dataset - weight_similarity))) ** 2))
    # l1_distance = jnp.log(jnp.exp(l1_distance))

    return l1_distance, l1_distance


def normalised_frame_weight_consistency_loss(
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

    # mean center the weights
    weight_similarity = weight_similarity - jnp.mean(weight_similarity)

    dataset = dataset - jnp.mean(dataset)

    # Compute the L1 distance between the two similarity matrices
    l1_distance = (jnp.abs(dataset - weight_similarity)) ** 2

    l1_distance = jnp.mean(-1 + jnp.exp(l1_distance))

    return l1_distance, l1_distance


def convex_KL_frame_weight_consistency_loss(
    model: Simulation, dataset: Array, prediction_index: int
) -> tuple[Array, Array]:
    """
    Computes and compares graphs of the pairwise distances between ensembles.
    One graph is constructed using the features/structures, another using the weights.
    TODO how are weights compared between each other?
    The loss is the L1 distance/Cosine between the two graphs.
    """
    weights = model.params.frame_weights

    weight_similarity = jax_pairwise_cosine_similarity(weights)

    # Get shape
    n = weight_similarity.shape[0]

    # Use JAX's built-in function for upper triangle indices
    # This is trace-compatible and avoids boolean conversion issues
    rows, cols = jnp.triu_indices(n, k=1)

    # Handle empty arrays case
    if rows.size == 0:
        return jnp.array(0.0), jnp.array(0.0)

    # Extract using these indices - this works with JIT
    weight_upper = weight_similarity[rows, cols]
    dataset_upper = dataset[rows, cols]
    # Small epsilon to avoid numerical issues
    epsilon = 1e-5
    # Normalize the weights and add epsilon
    weight_similarity = weight_upper + epsilon
    weight_similarity = weight_similarity / jnp.sum(weight_similarity)

    # Normalize the dataset and add epsilon
    dataset = dataset_upper + epsilon
    dataset = dataset / jnp.sum(dataset)
    # Use optax's safe sofftmax_cross_entropy
    loss = -1 + jnp.exp(
        jnp.sum(
            optax.losses.convex_kl_divergence(
                log_predictions=jnp.log(weight_similarity),
                targets=dataset,
            )
        )
    )

    return loss, loss


def cosine_frame_weight_consistency_loss(
    model: Simulation, dataset: Array, prediction_index: int
) -> tuple[Array, Array]:
    """
    Computes the cosine similarity between the pairwise weight similarity matrix
    and the input dataset matrix by considering only the upper triangular elements.
    """
    weights = model.params.frame_weights
    weight_similarity = jax_pairwise_cosine_similarity(weights)

    # Get shape
    n = weight_similarity.shape[0]

    # Use JAX's built-in function for upper triangle indices
    # This is trace-compatible and avoids boolean conversion issues
    rows, cols = jnp.triu_indices(n, k=1)

    # Handle empty arrays case
    if rows.size == 0:
        return jnp.array(0.0), jnp.array(0.0)

    # Extract using these indices - this works with JIT
    weight_upper = weight_similarity[rows, cols]
    dataset_upper = dataset[rows, cols]

    # Handle empty arrays case
    if weight_upper.size == 0 or dataset_upper.size == 0:
        return jnp.array(0.0), jnp.array(0.0)

    # Center the vectors (subtract mean) to improve numerical stability
    weight_centered = weight_upper - jnp.mean(weight_upper)
    dataset_centered = dataset_upper - jnp.mean(dataset_upper)

    # Compute dot product
    dot_product = jnp.sum(weight_centered * dataset_centered)

    # Compute magnitudes with safe epsilon
    epsilon = 1e-8
    weight_magnitude = jnp.sqrt(jnp.sum(weight_centered**2) + epsilon)
    dataset_magnitude = jnp.sqrt(jnp.sum(dataset_centered**2) + epsilon)

    # Safe division with fallback to zero when norms are too small
    cosine_similarity = jnp.where(
        (weight_magnitude > epsilon) & (dataset_magnitude > epsilon),
        dot_product / (weight_magnitude * dataset_magnitude),
        0.0,
    )

    # Clip to valid range and convert to distance
    cosine_similarity = jnp.clip(cosine_similarity, -1.0, 1.0)
    cosine_distance = 1.0 - cosine_similarity

    return cosine_distance, cosine_distance


def corr_frame_weight_consistency_loss(
    model: Simulation, dataset: Array, prediction_index: int
) -> tuple[Array, Array]:
    weights = model.params.frame_weights
    weight_similarity = jax_pairwise_cosine_similarity(weights)

    # Get shape
    n = weight_similarity.shape[0]

    # Use JAX's built-in function for upper triangle indices
    # This is trace-compatible and avoids boolean conversion issues
    rows, cols = jnp.triu_indices(n, k=1)

    # Handle empty arrays case
    if rows.size == 0:
        return jnp.array(0.0), jnp.array(0.0)

    # Extract using these indices - this works with JIT
    weight_upper = weight_similarity[rows, cols]
    dataset_upper = dataset[rows, cols]

    # Compute correlation coefficient instead of cosine similarity
    # This is more numerically stable during optimization
    weight_mean = jnp.mean(weight_upper)
    dataset_mean = jnp.mean(dataset_upper)

    weight_centered = weight_upper - weight_mean
    dataset_centered = dataset_upper - dataset_mean

    numerator = jnp.sum(weight_centered * dataset_centered)
    denominator = jnp.sqrt(jnp.sum(weight_centered**2) * jnp.sum(dataset_centered**2) + 1e-12)

    correlation = numerator / denominator
    correlation = jnp.clip(correlation, -1.0, 1.0)
    distance = 1.0 - correlation

    return distance, distance


def hdx_uptake_mean_centred_MSE_loss(
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
            timepoint_loss = jnp.mean(jnp.abs(pred_centered - true_centered) ** 2)

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss) / (y_true.shape[0])

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdxer_MSE_loss(
    model: Simulation,
    dataset: ExpD_Dataloader,
    prediction_index: int,
) -> tuple[Array, Array]:
    """
    Calculate the normalized weighted MSE loss (mean instead of sum).
    This version normalizes by the total number of datapoints for comparison
    with other MSE implementations.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values
        prediction_index: Index of the prediction to use
        gamma_weight: Weighting factor γ (default: 1.0)
        eta_squared: Variance parameter η² (default: 1.0)

    Returns:
        Tuple of train and validation losses
    """
    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]

    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true deuterated fractions for this timepoint
            true_dfracs_timepoint = y_true[timepoint_idx, :]

            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_dfracs_timepoint

            # Calculate weighted MSE
            squared_diff = (pred_mapped - true_mapped) ** 2

            # Sum over residues/segments for this timepoint
            timepoint_loss = jnp.sum(squared_diff) / 2

            # Accumulate loss and count datapoints
            total_loss += timepoint_loss
        return 1 - (jnp.exp(-jnp.asarray(total_loss)))

        # Return mean loss (normalized by total number of datapoints)
        return -jnp.log(jnp.exp(-jnp.asarray(total_loss / y_true.shape[0])))

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdxer_mcMSE_loss(
    model: Simulation,
    dataset: ExpD_Dataloader,
    prediction_index: int,
) -> tuple[Array, Array]:
    """
    Calculate the normalized weighted MSE loss (mean instead of sum).
    This version normalizes by the total number of datapoints for comparison
    with other MSE implementations.

    Args:
        model: Simulation object containing model outputs
        dataset: Experimental dataset containing true uptake values
        prediction_index: Index of the prediction to use
        gamma_weight: Weighting factor γ (default: 1.0)
        eta_squared: Variance parameter η² (default: 1.0)

    Returns:
        Tuple of train and validation losses
    """
    # Get the predicted uptake from the model
    predictions = model.outputs[prediction_index]

    def compute_loss(sparse_mapping, y_true):
        # Initialize loss accumulator
        total_loss = 0.0

        # Iterate over timepoints
        for timepoint_idx in range(y_true.shape[0]):
            # Get the true deuterated fractions for this timepoint
            true_dfracs_timepoint = y_true[timepoint_idx, :]

            # Get the predicted uptake for this timepoint
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            # Apply sparse mapping to predicted uptake
            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)
            true_mapped = true_dfracs_timepoint
            pred_mean = jnp.mean(pred_mapped)
            true_mean = jnp.mean(true_mapped)

            pred_centered = pred_mapped - pred_mean
            true_centered = true_mapped - true_mean
            # Calculate weighted MSE
            timepoint_loss = jnp.sum(jnp.abs(pred_centered - true_centered) ** 2)

            # Accumulate loss and count datapoints
            total_loss += timepoint_loss
        return 1 - (jnp.exp(-jnp.asarray(total_loss)))

        # Return mean loss (normalized by total number of datapoints)
        return -jnp.log(jnp.exp(-jnp.asarray(total_loss / y_true.shape[0])))

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdx_uptake_mean_centred_MAE_loss(
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
            timepoint_loss = jnp.mean(jnp.abs(pred_centered - true_centered) ** 1)

            # Accumulate loss
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss) / (y_true.shape[0])

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


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

            # Accumulate loss - normalize by the number of residues
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss)

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)

    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdx_uptake_MAE_loss(
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
            timepoint_loss = jnp.mean(jnp.abs(pred_mapped - true_mapped) ** 1)

            # Accumulate loss - normalize by the number of residues
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss) / (y_true.shape[0])

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)

    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


def hdx_uptake_MSE_loss(
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
            timepoint_loss = jnp.mean(jnp.abs(pred_mapped - true_mapped) ** 2)

            # Accumulate loss - normalize by the number of residues
            total_loss += timepoint_loss

        # Average loss across timepoints
        return jnp.asarray(total_loss) / (y_true.shape[0])

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)

    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss


# def hdx_uptake_MAE_loss_vectorized(
#     model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
# ) -> tuple[jnp.ndarray, jnp.ndarray]:
#     """
#     Memory-efficient vectorized MAE loss for HDX uptake.
#     """
#     predictions = model.outputs[prediction_index]

#     def compute_loss_vectorized(sparse_mapping, y_true):
#         pred_uptake_all = predictions.uptake

#         # Vectorized sparse mapping application
#         pred_mapped_all = jnp.array(
#             [
#                 apply_sparse_mapping(sparse_mapping, pred_uptake_all[t])
#                 for t in range(pred_uptake_all.shape[0])
#             ]
#         )

#         # Vectorized MAE computation
#         absolute_errors = jnp.abs(pred_mapped_all - y_true)
#         timepoint_losses = jnp.mean(absolute_errors, axis=1)
#         total_loss = jnp.mean(timepoint_losses)

#         return total_loss

#     train_loss = compute_loss_vectorized(
#         dataset.train.residue_feature_ouput_mapping, dataset.train.y_true
#     )
#     val_loss = compute_loss_vectorized(
#         dataset.val.residue_feature_ouput_mapping, dataset.val.y_true
#     )

#     return train_loss, val_loss


def hdx_uptake_MAE_loss_vectorized(
    model: Simulation, dataset: ExpD_Dataloader, prediction_index: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    MAE version using jax.lax.fori_loop instead of Python loops.
    """
    predictions = model.outputs[prediction_index]

    def compute_loss_original_style(sparse_mapping, y_true):
        def loop_body(timepoint_idx, carry):
            total_loss = carry

            true_uptake_timepoint = y_true[timepoint_idx, :]
            pred_uptake_timepoint = predictions.uptake[timepoint_idx]

            pred_mapped = apply_sparse_mapping(sparse_mapping, pred_uptake_timepoint)

            if len(true_uptake_timepoint.shape) > 1:
                true_uptake_timepoint = jnp.squeeze(true_uptake_timepoint)

            min_len = min(pred_mapped.shape[0], true_uptake_timepoint.shape[0])
            pred_mapped = pred_mapped[:min_len]
            true_uptake_timepoint = true_uptake_timepoint[:min_len]

            # MAE instead of MSE
            timepoint_loss = jnp.mean(jnp.abs(pred_mapped - true_uptake_timepoint))

            return total_loss + timepoint_loss

        n_timepoints = min(predictions.uptake.shape[0], y_true.shape[0])
        total_loss = jax.lax.fori_loop(0, n_timepoints, loop_body, 0.0)

        return total_loss / n_timepoints

    train_loss = compute_loss_original_style(
        dataset.train.residue_feature_ouput_mapping, dataset.train.y_true
    )
    val_loss = compute_loss_original_style(
        dataset.val.residue_feature_ouput_mapping, dataset.val.y_true
    )

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
            total_loss += timepoint_loss / true_uptake_timepoint.shape[0]

        # Average loss across timepoints
        return total_loss  # Consider adding division by timepoints

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
            total_loss += timepoint_loss / true_uptake_timepoint.shape[0]

        # Average loss across timepoints
        return total_loss  # Consider adding division by timepoints

    # Compute train and validation losses
    train_loss = compute_loss(dataset.train.residue_feature_ouput_mapping, dataset.train.y_true)
    val_loss = compute_loss(dataset.val.residue_feature_ouput_mapping, dataset.val.y_true)

    return train_loss, val_loss
