from typing import Any, Callable, Optional, Sequence, cast

import jax.numpy as jnp
import optax
from jax import (
    jit,
)
from jax.tree_util import register_pytree_node

from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.types.base import ForwardModel, ForwardPass
from jaxent.types.features import Input_Features, Output_Features
from jaxent.utils.jax_fn import frame_average_features, single_pass

# def forward_pure(
#     params: Simulation_Parameters,
#     input_features: Sequence[Input_Features],
#     forwardpass: Sequence[ForwardPass],
# ) -> Sequence[Output_Features]:
#     """
#     Pure function for forward computation that is jittable.

#     Args:
#         params: Simulation parameters
#         input_features: Input features
#         forwardpass: Forward pass functions

#     Returns:
#         Output features
#     """
#     # Normalize weights
#     params = Simulation_Parameters.normalize_weights(params)

#     # Mask the frame weights
#     masked_frame_weights = jnp.where(params.frame_mask < 0.5, 0, params.frame_weights)
#     masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)

#     # First map operation using tree_map
#     average_features = tree_map(
#         lambda feature: frame_average_features(feature, params.frame_weights),
#         input_features,
#     )

#     # Second map operation using tree_map
#     output_features = tree_map(
#         lambda fp, feat, param: single_pass(fp, feat, param),
#         forwardpass,
#         average_features,
#         params.model_parameters,
#     )

#     return output_features


class Simulation:
    """
    This is the core object that is used during optimisation
    """

    outputs: Sequence[Output_Features]
    _jit_forward_pure: Callable

    def __init__(
        self,
        input_features: list[Input_Features],
        forward_models: Sequence[ForwardModel],
        params: Optional[Simulation_Parameters],
        # model_name_index: list[tuple[m_key, int, m_id]],
    ) -> None:
        self.input_features: list[Input_Features[Any]] = input_features
        self.forward_models: Sequence[ForwardModel] = forward_models

        self.params: Simulation_Parameters = cast(Simulation_Parameters, params)

        self.forwardpass: Sequence[ForwardPass] = tuple(
            [model.forwardpass for model in self.forward_models]
        )
        # self.model_name_index: list[tuple[m_key, int, m_id]] = model_name_index
        # self.outputs: Sequence[Array]
        self._jit_forward_pure: Callable = None  # type: ignore

    # def __post_init__(self) -> None:
    #     # not implemented yet
    #     self._average_feature_map: Array  # a sparse array to map the average features to the single pass to generate the output features
    #     # self.output_features: dict[m_id, Array]

    def initialise(self) -> bool:
        # assert that input features have the same first dimension of "features_shape"
        lengths = [feature.features_shape[-1] for feature in self.input_features]
        assert len(set(lengths)) == 1, "Input features have different shapes. Exiting."
        self.length = lengths[0]

        if self.params is None:
            raise ValueError("No simulation parameters were provided. Exiting.")

        # assert that the number of forward models is equal to the number of forward model weights
        assert len(self.forward_models) == len(self.params.model_parameters), (
            "Number of forward models must be equal to number of forward model parameters"
        )

        # at this point we need to convert all the input features, parametere etc to jax arrays
        # use cast_to_jax for input features
        self._input_features: tuple[Input_Features] = cast(
            tuple[Input_Features], tuple([feature.cast_to_jax() for feature in self.input_features])
        )

        print("Loaded forward passes")
        print(self.forwardpass)

        print("Simulation initialised successfully.")
        # clear the jit function
        del self._jit_forward_pure
        # initialise the jit function using the inputs provided
        self._jit_forward_pure: Callable = cast(
            Callable,
            jit(
                self.forward_pure,
                static_argnames=("forwardpass"),  # "input_features",
            ),
        )
        try:
            self.forward(self.params)
            # if the forward pass is successful, try jit pass
        except Exception as e:
            raise ValueError(f"Failed to apply forward models: {e}")
        try:
            self._jit_forward_pure(
                self.params,
                self._input_features,
                self.forwardpass,
            )
        except Exception as e:
            RuntimeWarning(f"Warning - Jit failed: {e} \n Reverting to non-jit")
            self._jit_forward_pure = self.forward_pure

        # try to run the forward pass using the parameters provided

        return True

    def forward(self, params: Simulation_Parameters) -> None:
        """
        This function applies the forward models to the input features
        """
        self.params = params

        # try:
        outputs = self._jit_forward_pure(
            params,
            self._input_features,
            self.forwardpass,
        )
        # except Exception as e:
        #     RuntimeWarning(f"Warning - Jit failed: {e} \n Reverting to non-jit")

        # try:
        #     outputs = self.forward_pure(
        #         params,
        #         self._input_features,
        #         self.forwardpass,
        #     )
        # except Exception as e:
        #     raise ValueError(f"Failed to apply forward models: {e}")

        self.outputs = outputs

    def tree_flatten(self):
        """
        Flatten the Simulation object into dynamic values and static auxiliary data.

        Returns:
            A tuple (dynamic_values, aux_data) where:
            - dynamic_values: Parameters that will be transformed by JAX
            - aux_data: Static data that won't be transformed by JAX
        """
        # Dynamic values (leaves) - typically parameters that change during optimization
        dynamic_values = (self.params,)

        # Static auxiliary data - configuration that doesn't change during optimization
        aux_data = (
            self.input_features,
            self.forward_models,
            self.forwardpass,
            self.length,
            self.outputs,
        )

        return dynamic_values, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, dynamic_values):
        """
        Reconstruct a Simulation object from flattened data.

        Args:
            aux_data: Static auxiliary data from tree_flatten
            dynamic_values: Dynamic values from tree_flatten

        Returns:
            A new Simulation instance
        """
        # Unpack auxiliary data
        input_features, forward_models, forwardpass, length, outputs = aux_data

        # Unpack dynamic values
        (params,) = dynamic_values

        # Create a new instance
        instance = cls(input_features, forward_models, params)
        instance.forwardpass = forwardpass
        instance.length = length
        instance.outputs = outputs

        return instance

    @staticmethod
    def forward_pure(
        params: Simulation_Parameters,
        input_features: Sequence[Input_Features],
        forwardpass: Sequence[ForwardPass],
    ) -> Sequence[Output_Features]:
        """
        Pure function for forward computation that is jittable.

        Args:
            params: Simulation parameters
            input_features: Input features
            forwardpass: Forward pass functions

        Returns:
            Output features
        """
        # Normalize weights
        params = Simulation_Parameters.normalize_weights(params)

        # Mask the frame weights
        masked_frame_weights = jnp.where(params.frame_mask < 0.5, 0, params.frame_weights)
        masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)

        # Apply frame_average_features individually to each input feature
        # instead of using tree_map which can traverse too deep
        average_features = [
            frame_average_features(feature, params.frame_weights) for feature in input_features
        ]

        # Second operation using direct iteration as well
        output_features = [
            single_pass(fp, feat, param)
            for fp, feat, param in zip(forwardpass, average_features, params.model_parameters)
        ]

        return output_features


# Register Simulation as a pytree node
register_pytree_node(
    Simulation,
    Simulation.tree_flatten,
    Simulation.tree_unflatten,
)
