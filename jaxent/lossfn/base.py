# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from typing import Protocol

import jax.numpy as jnp

from jaxent.datatypes import Experimental_Dataset, Simulation
from jaxent.forwardmodels.base import Model_Parameters, Output_Features


class JaxEnt_Loss(Protocol):
    def __call__(
        self,
        model: Simulation | list[Simulation],
        dataset: Output_Features | Experimental_Dataset | Model_Parameters,
    ) -> float: ...


def exp_l2_loss(model: Simulation, dataset: Experimental_Dataset) -> float:
    """
    Calculate the L2 loss between the predicted and experimental data.
    """

    # Calculate the predicted data
    predictions = model.forward()

    # Calculate the L2 loss
    loss = jnp.sum((predictions - dataset.y_true) ** 2)

    # average the loss over the length of the dataset
    loss = jnp.mean(loss)

    return float(loss)
