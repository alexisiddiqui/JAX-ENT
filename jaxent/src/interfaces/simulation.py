from collections.abc import Sequence
from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int, Array, Bool
from jax.tree_util import register_pytree_node

from jaxent.src.interfaces.model import Model_Parameters

########################################################################
# TODO - use generics/typevar to abstractly define the datatypes


@dataclass(frozen=True, slots=True)
class Simulation_Parameters:
    frame_weights: Float[Array, " n_frames"]
    frame_mask: Float[Array, " n_frames"] | Int[Array, " n_models"] | Bool[Array, " n_models"] # array of type int
    model_parameters: Sequence[Model_Parameters]
    forward_model_weights: Float[Array, " n_models"]
    normalise_loss_functions: Float[Array, " n_models"] | Int[Array, " n_models"] | Bool[Array, " n_models"]  # array of type int
    forward_model_scaling: Float[Array, " n_models"]

    ########################################################################
    # TODO I think this is maybe kinda silly - but
    @staticmethod
    def propagate_model_parameters(params: "Simulation_Parameters", model_index: int = 0):
        """
        Propagates the model parameters at model_index to all model parameters.
        """
        model_param = params.model_parameters[model_index]
        new_model_params = [model_param for _ in params.model_parameters]

        return Simulation_Parameters(
            frame_weights=params.frame_weights,
            frame_mask=params.frame_mask,
            model_parameters=new_model_params,
            normalise_loss_functions=params.normalise_loss_functions,
            forward_model_weights=params.forward_model_weights,
            forward_model_scaling=params.forward_model_scaling,
        )

    @staticmethod
    def normalize_masked_loss_scalingweights(params: "Simulation_Parameters"):
        """
        Normalizes weights where mask is non-zero, avoiding boolean contexts.
        """
        # Convert mask to float without boolean comparison
        weights = params.forward_model_weights
        mask = params.normalise_loss_functions

        float_mask = jnp.clip(mask, 0, 1).astype(jnp.float32)

        # Apply mask to weights
        masked_weights = weights * float_mask

        # Get the sum of masked weights with epsilon to avoid division by 0
        total = jnp.sum(masked_weights)

        # Normalize masked weights
        normalized = masked_weights / total

        # Use original weights where mask is 0, normalized weights otherwise
        # Avoid boolean comparison with mask by using multiplication
        result = weights * (1.0 - float_mask) + normalized * float_mask

        return Simulation_Parameters(
            frame_weights=params.frame_weights,
            frame_mask=params.frame_mask,
            model_parameters=params.model_parameters,
            normalise_loss_functions=params.normalise_loss_functions,
            forward_model_weights=result,
            forward_model_scaling=params.forward_model_scaling,
        )

    @staticmethod
    def param_labels(params: "Simulation_Parameters") -> "Simulation_Parameters":
        """Create a label tree for optax.multi_transform that matches the parameter structure.

        This method uses object.__new__ and object.__setattr__ to bypass beartype validation,
        since the labels are strings rather than JAX arrays. This is required because optax's
        multi_transform needs the labels to be the exact same pytree node type as the parameters.

        Args:
            params: The simulation parameters to create labels for

        Returns:
            A Simulation_Parameters instance with string labels for each parameter group
        """
        # Use object.__new__ to create instance without calling __init__ (bypasses beartype)
        instance = object.__new__(Simulation_Parameters)
        # Use object.__setattr__ to bypass frozen dataclass and beartype validation
        object.__setattr__(instance, "frame_weights", "frame")
        object.__setattr__(instance, "frame_mask", "frame")
        object.__setattr__(instance, "model_parameters", ["model"] * len(params.model_parameters))
        object.__setattr__(instance, "forward_model_weights", "other")
        object.__setattr__(instance, "forward_model_scaling", "other")
        object.__setattr__(instance, "normalise_loss_functions", "other")
        return instance

    @staticmethod
    def normalize_weights(params: "Simulation_Parameters") -> "Simulation_Parameters":
        """Create a new instance with normalized frame weights using JAX-compatible operations"""
        # Use projection_simplex for frame weights normalization
        # frame_weights = optax.projections.projection_simplex(jnp.asarray(params.frame_weights))

        frame_weights = jax.nn.softmax(params.frame_weights)

        # Clip the frame mask to be between 0 and 1
        # Apply smooth binary approximation
        # def smooth_binary_poly(x):
        #     """Polynomial S-curve that transitions smoothly from 0 to 1."""
        #     return 3 * x**2 - 2 * x**3

        def sigmoid(x):
            """Sigmoid function for smooth transition."""
            return jax.nn.sigmoid(10 * (x - 0.5))

        frame_mask = sigmoid(params.frame_mask)

        frame_mask = jnp.clip(frame_mask, 0, 1)

        # Modified normalize_masked_weights function to avoid boolean context issues

        return Simulation_Parameters(
            frame_weights=frame_weights,
            frame_mask=frame_mask,
            model_parameters=params.model_parameters,
            normalise_loss_functions=params.normalise_loss_functions,
            forward_model_weights=params.forward_model_weights,
            forward_model_scaling=params.forward_model_scaling,
        )

    @staticmethod
    def _apply_op(current, op, other: "Simulation_Parameters|float") -> "Simulation_Parameters":
        if not isinstance(other, (Simulation_Parameters, float)):
            raise TypeError("Unsupported type for operation")

        if isinstance(other, float):
            # Apply operation with scalar to all leaf nodes of the pytree
            new_params = jax.tree_util.tree_map(lambda x: op(x, other), current)
        else:
            # Apply operation between two pytrees element-wise
            new_params = jax.tree_util.tree_map(op, current, other)

        return new_params

    def __add__(self, other) -> "Simulation_Parameters":
        return self._apply_op(self, jnp.add, other)

    def __sub__(self, other) -> "Simulation_Parameters":
        return self._apply_op(self, jnp.subtract, other)

    def __mul__(self, other) -> "Simulation_Parameters":
        return self._apply_op(self, jnp.multiply, other)

    def __truediv__(self, other) -> "Simulation_Parameters":
        return self._apply_op(self, jnp.divide, other)

    __radd__ = __add__  # a + b = b + a
    __rmul__ = __mul__  # a * b = b * a

    # For non-commutative operations, we need separate implementations
    def __rsub__(self, other) -> "Simulation_Parameters":
        return self._apply_op(other, jnp.subtract, self)

    def __rtruediv__(self, other) -> "Simulation_Parameters":
        return self._apply_op(other, jnp.divide, self)

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
        """Reconstruct from flattened pytree representation.

        This method uses object.__new__ and object.__setattr__ to bypass beartype validation,
        since JAX's tree operations (e.g., optax's internal masking) may produce intermediate
        structures with booleans or other non-Array types.
        """
        (
            frame_weights,
            frame_mask,
            model_params,
            forward_weights,
            forward_model_scaling,
            normalise_loss_functions,
        ) = arrays
        _ = static
        # Use object.__new__ to bypass beartype validation
        instance = object.__new__(cls)
        object.__setattr__(instance, "frame_weights", frame_weights)
        object.__setattr__(instance, "frame_mask", frame_mask)
        object.__setattr__(instance, "model_parameters", model_params)
        object.__setattr__(instance, "normalise_loss_functions", normalise_loss_functions)
        object.__setattr__(instance, "forward_model_weights", forward_weights)
        object.__setattr__(instance, "forward_model_scaling", forward_model_scaling)
        return instance


# Register the class as a pytree node
register_pytree_node(
    Simulation_Parameters, Simulation_Parameters.tree_flatten, Simulation_Parameters.tree_unflatten
)
