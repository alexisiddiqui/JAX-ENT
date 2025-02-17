# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from typing import Protocol, TypeVar

import jax.numpy as jnp
from jax import Array

from jaxent.datatypes import Experimental_Dataset, Simulation
from jaxent.forwardmodels.base import Model_Parameters, Output_Features

# Define TypeVars for the different possible types
M = TypeVar("M", Simulation, list[Simulation], contravariant=True)
D = TypeVar("D", Output_Features, Experimental_Dataset, Model_Parameters, contravariant=True)


class JaxEnt_Loss(Protocol[M, D]):
    def __call__(
        self, model: M, dataset: D, prediction_index: int | str | None
    ) -> tuple[Array, Array]: ...


def hdx_pf_l2_loss(model: Simulation, dataset: Experimental_Dataset) -> Array:
    """
    Calculate the L2 loss between the predicted and experimental data.
    """

    # Calculate the predicted data
    predictions = (
        model.forward()
    )  # TODO: find a way to move the forward call to outside the loss function
    pred_pf = jnp.array(predictions[0].log_Pf).reshape(-1)  # Flatten to 1D
    true_pf = dataset.y_true.reshape(-1)  # Flatten to 1D

    # print(predictions[0].log_Pf)
    # Calculate the L2 loss
    loss = jnp.sum((pred_pf - true_pf) ** 2)
    # print(loss)
    # average the loss over the length of the dataset
    loss = jnp.mean(loss)

    return loss


def hdx_pf_mae_loss(model: Simulation, dataset: Experimental_Dataset) -> Array:
    """
    Calculate the mae loss between the predicted and experimental data.
    """

    # Calculate the predicted data
    predictions = (
        model.forward()
    )  # TODO: find a way to move the forward call to outside the loss function
    pred_pf = jnp.array(predictions[0].log_Pf).reshape(-1)  # Flatten to 1D
    true_pf = dataset.y_true.reshape(-1)  # Flatten to 1D

    # print(predictions[0].log_Pf)
    # Calculate the
    loss = jnp.sum(pred_pf - true_pf)
    # print(loss)
    # average the loss over the length of the dataset
    loss = jnp.mean(loss)
    # take the absolute value of the loss
    loss = jnp.abs(loss)

    return loss


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
