# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from typing import Protocol, TypeVar

import jax.numpy as jnp
from jax import Array
from optax.losses import safe_softmax_cross_entropy

from jaxent.datatypes import Experimental_Dataset, Simulation, Simulation_Parameters
from jaxent.forwardmodels.base import Model_Parameters, Output_Features
from jaxent.utils.datasplitter import apply_sparse_mapping

# Define TypeVars for the different possible types
M = TypeVar("M", Simulation, list[Simulation], contravariant=True)
D = TypeVar("D", Output_Features, Experimental_Dataset, Model_Parameters, contravariant=True)


class JaxEnt_Loss(Protocol[M, D]):
    def __call__(
        self, model: M, dataset: D, prediction_index: int | str | None
    ) -> tuple[Array, Array]: ...


def hdx_pf_l2_loss(
    model: Simulation, dataset: Experimental_Dataset, prediction_index: int
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
    loss = jnp.sum((pred_pf - true_pf) ** 2)
    # print(loss)
    # average the loss over the length of the dataset
    train_loss = jnp.mean(loss)

    pred_pf = jnp.array(predictions[prediction_index].log_Pf).reshape(-1)  # Flatten to 1D

    pred_pf = apply_sparse_mapping(dataset.val.residue_feature_ouput_mapping, pred_pf)

    true_pf = dataset.val.y_true.reshape(-1)  # Flatten to 1D

    # Calculate the L2 loss
    loss = jnp.sum((pred_pf - true_pf) ** 2)
    # print(loss)
    # average the loss over the length of the dataset
    val_loss = jnp.mean(loss)

    return train_loss, val_loss


def hdx_pf_mae_loss(
    model: Simulation, dataset: Experimental_Dataset, prediction_index: int
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
    loss = jnp.sum(jnp.abs(pred_pf - true_pf) ** 1)
    # print(loss)
    # average the loss over the length of the dataset
    train_loss = jnp.mean(loss)

    pred_pf = jnp.array(predictions[prediction_index].log_Pf).reshape(-1)  # Flatten to 1D

    pred_pf = apply_sparse_mapping(dataset.val.residue_feature_ouput_mapping, pred_pf)

    true_pf = dataset.val.y_true.reshape(-1)  # Flatten to 1D

    # Calculate the L2 loss
    loss = jnp.sum(jnp.abs(pred_pf - true_pf) ** 1)
    # print(loss)
    # average the loss over the length of the dataset
    val_loss = jnp.mean(loss)

    return train_loss, val_loss


def max_entropy_loss(
    model: Simulation, dataset: Simulation_Parameters, prediction_index: None
) -> tuple[Array, Array]:
    simulation_weights = jnp.abs(model.params.frame_weights) / jnp.sum(
        jnp.abs(model.params.frame_weights)
    )

    prior_frame_weights = jnp.abs(dataset.frame_weights) / jnp.sum(jnp.abs(dataset.frame_weights))

    loss = safe_softmax_cross_entropy(jnp.log(simulation_weights), prior_frame_weights)
    print(loss)
    return loss, loss


def hdx_uptake_l2_loss(model: Simulation, dataset: Experimental_Dataset) -> Array:
    """
    Calculate the L2 loss between the predicted and experimental data.
    """

    # Calculate the predicted data
    predictions = model.forward()
    # print(predictions)
    pred_uptake = jnp.array(predictions[2].uptake).reshape(-1)  # Flatten to 1D
    true_uptake = dataset.y_true.reshape(-1)  # Flatten to 1D

    # Calculate the L2 loss
    loss = jnp.sum((pred_uptake - true_uptake) ** 2)

    # average the loss over the length of the dataset
    loss = jnp.mean(loss)

    return loss


def hdx_uptake_monotonicity_loss(model: Simulation, dataset: None) -> Array:
    """
    Calculate the monotonicity loss for HDX uptake predictions.
    Penalizes violations of monotonic increase in time using squared penalties.

    Args:
        model: Simulation object containing the predictions

    Returns:
        Array: The computed monotonicity loss
    """
    # Calculate the predicted data
    predictions = model.forward()

    # Get the uptake predictions and reshape if needed
    # Assuming predictions[2] contains uptake data with shape (peptides, timepoints)
    deut = jnp.array(predictions[2].uptake)

    # Calculate differences between adjacent timepoints
    time_diffs = deut[:, 1:] - deut[:, :-1]

    # Use JAX's equivalent of torch.relu for negative differences
    # This penalizes any decrease in deuteration over time
    violations = jnp.maximum(-time_diffs, 0)

    # Square the violations and take the mean
    # If there are no elements, return 0
    loss = jnp.where(time_diffs.size > 0, jnp.mean(violations**2), jnp.array(0.0))

    return loss


def frame_weight_consistency_loss(model: Simulation, dataset: Experimental_Dataset) -> Array:
    """
    Computes and compares graphs of the pairwise distances between ensembles.
    One graph is constructed using the features/structures, another using the weights.
    TODO how are weights compared between each other?
    The loss is the L2 distance/Cosine between the two graphs.
    """
