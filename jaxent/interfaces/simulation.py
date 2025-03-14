from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import optax  # type: ignore
from jax import Array, jit
from jax.tree_util import register_pytree_node

from jaxent.interfaces.model import Model_Parameters

########################################################################\
# TODO - use generics/typevar to abstractly define the datatypes


@dataclass(frozen=True, slots=True)
class Simulation_Parameters:
    frame_weights: Array
    frame_mask: Array  # array of type int
    model_parameters: Sequence[Model_Parameters]
    forward_model_weights: Array
    normalise_loss_functions: Array  # array of type int
    forward_model_scaling: Array

    ########################################################################
    # TODO I think this is maybe kinda silly - but
    @staticmethod
    def normalize_weights(params: "Simulation_Parameters") -> "Simulation_Parameters":
        """Create a new instance with normalized frame weights using JAX-compatible operations"""
        # set weights and

        frame_weights = optax.projections.projection_simplex(jnp.asarray(params.frame_weights))

        # clip the frame mask to be between 0 and 1
        # raise NotImplementedError("Normalisation of frame mask is not implemented yet.")
        # Ensure frame mask is between 0 and 1
        frame_mask_clipped = jnp.clip(params.frame_mask, 0, 1)

        # Apply smooth binary approximation
        def smooth_binary_poly(x):
            """Polynomial S-curve that transitions smoothly from 0 to 1."""
            return 3 * x**2 - 2 * x**3  # Already clipped above

        # Convert to binary-like values with smooth gradients
        frame_mask = smooth_binary_poly(frame_mask_clipped)
        # def round_straight_through(x):
        #     """Round with straight-through gradient estimator."""
        #     rounded = jnp.round(x)
        #     return x + jax.lax.stop_gradient(rounded - x)

        # # apply mask to frame weights
        # binary_mask = round_straight_through(frame_mask)
        # # normalise the masked frame weights, so that they sum to 1. if frame_mask sum is less than 1 - return the original frame weights

        # condition = jnp.sum(binary_mask) > 1
        # result = jnp.where(condition, binary_mask, frame_mask)
        # frame_mask = frame_mask + jax.lax.stop_gradient(result - frame_mask)
        # frame_mask = jnp.asarray(jnp.where(jnp.sum(binary_mask) > 1, binary_mask, frame_mask))
        # normalise the frame weights
        # normalized_weights = optax.projections.projection_simplex(frame_weights)
        # total = jnp.sum(frame_weights)
        # find normalise loss indexes
        @jit
        def normalize_masked_weights(weights, mask):
            """
            Normalizes weights where mask is True, maintaining gradient flow.
            Uses smooth masking for gradient propagation with safe division.

            Args:
            weights: Array of weights
            mask: Boolean mask indicating which weights to normalize

            Returns:
            Array with the masked weights normalized
            """
            # Convert mask to float for multiplication
            float_mask = mask.astype(jnp.float32)

            # Apply mask to weights
            masked_weights = weights * float_mask

            # Get the sum of masked weights and add a small epsilon to avoid division by 0
            total = jnp.sum(masked_weights)
            epsilon = 1e-8

            # Normalize masked weights using safe division
            normalized = masked_weights / (total + epsilon)

            # Use original weights where mask is False
            result = jnp.where(mask, normalized, weights)

            return result

        # Normalize forward model weights
        forward_model_weights = normalize_masked_weights(
            params.forward_model_weights, params.normalise_loss_functions
        )

        # normalized_weights = frame_weights / total
        # print("Weights normalized.")
        # print(normalized_weights)
        return Simulation_Parameters(
            frame_weights=frame_weights,
            frame_mask=frame_mask,
            model_parameters=params.model_parameters,
            normalise_loss_functions=params.normalise_loss_functions,
            forward_model_weights=jnp.asarray(forward_model_weights),
            forward_model_scaling=params.forward_model_scaling,
        )

    ########################################################################\
    def tree_flatten(self):
        # Flatten into (arrays to differentiate, static metadata)
        arrays = (
            self.frame_weights,
            self.frame_mask,
            [m for m in self.model_parameters],
            self.forward_model_weights,
            self.forward_model_scaling,
            self.normalise_loss_functions,
        )

        static = ()
        return arrays, static

    @classmethod
    def tree_unflatten(cls, static, arrays):
        # Create instance first, then normalize
        (
            frame_weights,
            frame_mask,
            model_params,
            forward_weights,
            forward_model_scaling,
            normalise_loss_functions,
        ) = arrays
        _ = static
        # Instead of normalizing after creation, just create with the given weights
        return cls(
            frame_weights=frame_weights,
            frame_mask=frame_mask,  # .astype(int),
            model_parameters=model_params,
            normalise_loss_functions=normalise_loss_functions,
            forward_model_weights=forward_weights,
            forward_model_scaling=forward_model_scaling,
        )


# Register the class as a pytree node
register_pytree_node(
    Simulation_Parameters, Simulation_Parameters.tree_flatten, Simulation_Parameters.tree_unflatten
)
