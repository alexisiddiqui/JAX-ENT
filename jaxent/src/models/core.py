from collections.abc import Callable, Sequence
from typing import Any, Optional, Union, cast

import chex
import jax.numpy as jnp
from jax import (
    jit,
    Array,
)
from jax.tree_util import register_pytree_node
from jaxtyping import Float

from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.utils.jax_fn import frame_average_features, single_pass


class Simulation:
    """
    This is the core object that is used during optimisation
    """

    outputs: tuple[Output_Features]
    _jit_forward_pure: Callable | None

    def __init__(
        self,
        input_features: list[Input_Features],
        forward_models: Sequence[ForwardModel],
        params: Optional[Simulation_Parameters],
        raise_jit_failure: bool = False,
        # model_name_index: list[tuple[m_key, int, m_id]],
    ) -> None:
        self.input_features: list[Input_Features[Any]] = input_features
        self.forward_models: Sequence[ForwardModel] = forward_models

        self.params: Simulation_Parameters | None = params

        self.forwardpass: Sequence[ForwardPass] = tuple(
            [model.forwardpass for model in self.forward_models]
        )
        # self.model_name_index: list[tuple[m_key, int, m_id]] = model_name_index
        # self.outputs: Sequence[Array]
        self._jit_forward_pure: Callable | None = None
        self.raise_jit_failure: bool = raise_jit_failure

    def __repr__(self) -> str:
        return f"Simulation(raise_jit_failure={self.raise_jit_failure})"

    # def __post_init__(self) -> None:
    #     # not implemented yet
    #     self._average_feature_map: Array  # a sparse array to map the average features to the single pass to generate the output features
    #     # self.output_features: dict[m_id, Array]

    def initialise(self) -> bool:
        # Assert that input features have the same frame dimension
        lengths = [feature.features_shape[-1] for feature in self.input_features]
        chex.assert_equal(len(set(lengths)), 1, custom_message="Input features have different frame counts")
        self.length = lengths[0]

        if self.params is None:
            raise ValueError("No simulation parameters were provided. Exiting.")
        
        # Validate parameter ranks
        chex.assert_rank(self.params.frame_weights, 1)
        chex.assert_equal_shape([self.params.frame_weights, self.params.frame_mask])
        
        self.params = Simulation_Parameters.normalize_weights(self.params)
        self.params = Simulation_Parameters.normalize_masked_loss_scalingweights(self.params)
        
        # Assert that the number of forward models matches model parameters
        chex.assert_equal(
            len(self.forward_models), 
            len(self.params.model_parameters),
            custom_message="Number of forward models must equal number of model parameters"
        )

        # at this point we need to convert all the input features, parametere etc to jax arrays
        # use cast_to_jax for input features
        self._input_features: tuple[Input_Features, ...] = cast(
            tuple[Input_Features, ...],
            tuple([feature.cast_to_jax() for feature in self.input_features]),
        )

        print("Loaded forward passes")
        print(self.forwardpass)

        # clear the jit function
        del self._jit_forward_pure

        self._jit_forward_pure = self.forward_pure
        # initialise the jit function using the inputs provided
        try:
            _ = self.forward(self, self.params)
            # if the forward pass is successful, try jit pass
        except Exception as e:
            raise ValueError(f"Failed to apply forward models without JIT: {e}")

        self._jit_forward_pure: Callable = cast(
            Callable,
            jit(
                self.forward_pure,
                static_argnames=("forwardpass"),  # "input_features",
                # donate_argnames=("params", "input_features"),
            ),
        )
        try:
            self._jit_forward_pure(
                self.params,
                self._input_features,
                self.forwardpass,
            )
            print("\n\n\n\n\n\n\n\n\n JIT compilation successful \n\n\n\n\n\n\n\n\n")

        except Exception as e:
            if self.raise_jit_failure:
                raise RuntimeError(f"Warning - Jit failed: {e} \n Reverting to non-jit")
            print(f"Warning - Jit failed: {e} \n Reverting to non-jit")
            self._jit_forward_pure = self.forward_pure

        print("Simulation initialised successfully.")
        # try to run the forward pass using the parameters provided

        return True

    @staticmethod
    def forward(sim, params: Simulation_Parameters) -> "Simulation":
        """
        This function applies the forward models to the input features
        """
        params = Simulation_Parameters.normalize_weights(params)
        sim.params = params

        # try:
        outputs = sim._jit_forward_pure(
            params,
            sim._input_features,
            sim.forwardpass,
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

        sim.outputs = tuple(outputs)
        return sim

    def predict(
        self, params: Union[Simulation_Parameters, Sequence[Model_Parameters]]
    ) -> Sequence[Output_Features]:
        """
        Apply forward pass to non-averaged features, returning frame-wise predictions.

        Returns sequence of output features with final dimension as n_frames for each forward model.
        """
        if not hasattr(self, "_input_features"):
            raise RuntimeError("Simulation must be initialized before calling predict")

        # Extract model parameters
        if isinstance(params, Simulation_Parameters):
            model_parameters = params.model_parameters
        else:
            model_parameters = params

        if len(model_parameters) != len(self.forwardpass):
            raise ValueError(
                f"Number of model parameters ({len(model_parameters)}) must match "
                f"number of forward models ({len(self.forwardpass)})"
            )

        n_frames = self._input_features[0].features_shape[-1]
        output_features = []

        for fp, feature, param in zip(self.forwardpass, self._input_features, model_parameters):
            # Collect frame outputs for this forward model
            frame_outputs = []

            for frame_idx in range(n_frames):
                # Extract frame data for all features generically
                frame_data = {}

                # Slice feature arrays along frame dimension
                for feat_name in feature.__features__:
                    feat_array = getattr(feature, feat_name)
                    if feat_array is not None and getattr(feat_array, "ndim", 0) >= 2 and feat_array.shape[-1] == n_frames:
                        frame_data[feat_name] = feat_array[..., frame_idx : frame_idx + 1]
                    else:
                        frame_data[feat_name] = feat_array

                # Preserve all non-feature attributes unchanged
                for slot in feature._get_ordered_slots():
                    if slot not in feature.__features__:
                        frame_data[slot] = getattr(feature, slot)

                # Create new frame feature object
                frame_feature = feature.__class__(**frame_data)

                # Apply forward pass to single frame
                frame_output = single_pass(fp, frame_feature, param)
                frame_outputs.append(frame_output)

            # Stack frame outputs along final dimension generically
            first_output = frame_outputs[0]
            stacked_data = {}

            # Stack all feature arrays (skip None values)
            for feat_name in first_output.__features__:
                feat_arrays = [getattr(out, feat_name) for out in frame_outputs]
                if feat_arrays[0] is not None:
                    stacked = jnp.stack(feat_arrays, axis=-1)
                    # Squeeze singleton dimensions except the last (frames) dimension
                    stacked_data[feat_name] = jnp.squeeze(
                        stacked,
                        axis=tuple(i for i in range(stacked.ndim - 1) if stacked.shape[i] == 1),
                    )
                else:
                    stacked_data[feat_name] = None

            # Preserve non-feature attributes from first output
            for slot in first_output._get_ordered_slots():
                if slot not in first_output.__features__:
                    stacked_data[slot] = getattr(first_output, slot)

            stacked_output = first_output.__class__(**stacked_data)
            output_features.append(stacked_output)

        return output_features

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
            self._input_features,
            self._jit_forward_pure,
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
        (
            input_features,
            forward_models,
            forwardpass,
            length,
            outputs,
            _input_features,
            _jit_forward_pure,
        ) = aux_data

        # Unpack dynamic values
        (params,) = dynamic_values

        # Create a new instance
        instance = cls(input_features, forward_models, params)
        instance.forwardpass = forwardpass
        instance.length = length
        instance.outputs = outputs
        instance._input_features = _input_features
        instance._jit_forward_pure = _jit_forward_pure

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
            params: Simulation parameters with frame_weights as Float[Array, " n_frames"]
            input_features: Input features, each with shape (n_residues, n_frames)
            forwardpass: Forward pass functions

        Returns:
            Output features from each forward model
        """
        # Validate frame_weights rank
        chex.assert_rank(params.frame_weights, 1)

        # Mask the frame weights
        # masked_frame_weights = jnp.where(params.frame_mask < 0.5, 0, params.frame_weights)
        # masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)

        # Apply frame_average_features individually to each input feature
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
