# TODO: write loss functions:
# L1, L2, KL Divergence, Hinge loss, Cross-entropy loss, etc.
# Specialized loss functions for specific tasks:
# Monotonicity loss, Consistency loss.

from typing import Protocol, TypeVar

import jax.numpy as jnp

from jaxent.datatypes import Experimental_Dataset, Simulation
from jaxent.forwardmodels.base import Model_Parameters, Output_Features

# Define TypeVars for the different possible types
M = TypeVar("M", Simulation, list[Simulation], contravariant=True)
D = TypeVar("D", Output_Features, Experimental_Dataset, Model_Parameters, contravariant=True)


class JaxEnt_Loss(Protocol[M, D]):
    def __call__(
        self,
        model: M,
        dataset: D,
    ) -> float: ...


def hdx_pf_l2_loss(model: Simulation, dataset: Experimental_Dataset) -> float:
    """
    Calculate the L2 loss between the predicted and experimental data.
    """

    # Calculate the predicted data
    predictions = model.forward()
    # print(predictions[0].log_Pf)
    # Calculate the L2 loss
    loss = jnp.sum((predictions[0].log_Pf - dataset.y_true) ** 2)
    print(loss)
    # average the loss over the length of the dataset
    loss = jnp.mean(loss)

    return float(loss)
