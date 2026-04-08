from __future__ import annotations

from collections.abc import Callable, Sequence
import logging
from typing import Any, Optional, Union, cast

import chex
from jax import jit
from jax.tree_util import register_pytree_node

from jaxent.src.custom_types.base import ForwardModel, ForwardPass
from jaxent.src.custom_types.features import Input_Features, Output_Features
from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.model import Model_Parameters
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.utils.jax_fn import frame_average_features, single_pass

LOGGER = logging.getLogger("jaxent.models")


class Simulation:
    """
    This is the core object that is used during optimisation
    """

    outputs: tuple[Output_Features, ...]
    _jit_forward_pure: Callable | None

    def __init__(
        self,
        input_features: Sequence[Input_Features],
        forward_models: Sequence[ForwardModel],
        params: Optional[Simulation_Parameters],
        raise_jit_failure: bool = False,
        # model_name_index: list[tuple[m_key, int, m_id]],
    ) -> None:
        self.input_features: Sequence[Input_Features[Any]] = input_features
        self.forward_models: Sequence[ForwardModel] = forward_models

        self.params: Simulation_Parameters | None = params

        self.forwardpass: Sequence[ForwardPass] = tuple(
            [model.forwardpass for model in self.forward_models]
        )
        # self.model_name_index: list[tuple[m_key, int, m_id]] = model_name_index
        # self.outputs: Sequence[Array]
        self.outputs = tuple()
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

        LOGGER.debug("Loaded forward passes: %s", self.forwardpass)

        # clear the jit function
        del self._jit_forward_pure

        self._jit_forward_pure = self.forward_pure
        # initialise the jit function using the inputs provided
        try:
            _sim, _outputs = self.forward(self, self.params, mutate=False)
            self.params = _sim.params
            self.outputs = _outputs
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
            LOGGER.info("Simulation forward JIT compilation successful.")

        except Exception as e:
            if self.raise_jit_failure:
                raise RuntimeError(f"Warning - Jit failed: {e} \n Reverting to non-jit")
            LOGGER.warning("Forward JIT failed, reverting to eager forward: %s", e)
            self._jit_forward_pure = self.forward_pure

        LOGGER.info("Simulation initialised successfully.")
        # try to run the forward pass using the parameters provided

        return True

    @staticmethod
    def forward(
        sim,
        params: Simulation_Parameters,
        mutate: bool = True,
    ) -> Union["Simulation", tuple["Simulation", tuple[Output_Features, ...]]]:
        """
        Apply forward models to input features and return a new simulation and outputs.

        Args:
            sim: Simulation object
            params: Simulation parameters
            mutate: Backward-compatible behavior. When True, update ``sim`` in place.
        """
        params = Simulation_Parameters.normalize_weights(params)
        outputs = tuple(
            sim._jit_forward_pure(
                params,
                sim._input_features,
                sim.forwardpass,
            )
        )

        _, aux_data = sim.tree_flatten()
        new_sim = Simulation.tree_unflatten(aux_data, (params, outputs))

        if mutate:
            sim.params = new_sim.params
            sim.outputs = new_sim.outputs
            return sim

        return new_sim, outputs

    @property
    def outputs_by_key(self) -> dict[m_key, Output_Features]:
        """Look up outputs by their m_key.
        
        Raises KeyError on duplicate keys (indicates a misconfigured Simulation).
        """
        result: dict[m_key, Output_Features] = {}
        for output in self.outputs:
            if output.key in result:
                raise KeyError(
                    f"Duplicate output key '{output.key}' — each forward model "
                    f"must produce outputs with unique m_keys."
                )
            result[output.key] = output
        return result

    def predict(
        self, params: Union[Simulation_Parameters, Sequence[Model_Parameters]]
    ) -> Sequence[Output_Features]:
        """
        Apply forward passes to un-averaged (frame-wise) features.

        Returns per-frame output features — each output retains the frame dimension
        as the last axis. All forward passes handle (..., n_frames) inputs via
        element-wise ops, so no explicit frame loop is needed.
        """
        if not hasattr(self, "_input_features"):
            raise RuntimeError("Simulation must be initialized before calling predict")

        if isinstance(params, Simulation_Parameters):
            model_parameters = params.model_parameters
        else:
            model_parameters = params

        if len(model_parameters) != len(self.forwardpass):
            raise ValueError(
                f"Number of model parameters ({len(model_parameters)}) must match "
                f"number of forward models ({len(self.forwardpass)})"
            )

        return [
            single_pass(fp, feat, param)
            for fp, feat, param in zip(self.forwardpass, self._input_features, model_parameters)
        ]

    def tree_flatten(self):
        """
        Flatten the Simulation object into dynamic values and static auxiliary data.

        Returns:
            A tuple (dynamic_values, aux_data) where:
            - dynamic_values: Parameters that will be transformed by JAX
            - aux_data: Static data that won't be transformed by JAX
        """
        # Dynamic values (leaves) - typically parameters that change during optimization
        dynamic_values = (self.params, self.outputs)

        # Static auxiliary data - configuration that doesn't change during optimization
        aux_data = (
            self.input_features,
            self.forward_models,
            self.forwardpass,
            self.length,
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
            _input_features,
            _jit_forward_pure,
        ) = aux_data

        # Unpack dynamic values
        (params, outputs) = dynamic_values

        # Create a new instance
        instance = cls(input_features, forward_models, params)
        instance.forwardpass = forwardpass
        instance.length = length
        instance.outputs = tuple(outputs)
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

        # Branch per forward pass: linear models average features first (average_first=True),
        # non-linear models run on frame-wise features and average outputs (average_first=False).
        # Since forwardpass is a static JIT arg, this branch compiles away.
        output_features = []
        for fp, feat, param in zip(forwardpass, input_features, params.model_parameters):
            if getattr(fp, "average_first", True):
                avg_feat = frame_average_features(feat, params.frame_weights)
                output = single_pass(fp, avg_feat, param)
            else:
                output = single_pass(fp, feat, param)
                output = frame_average_features(output, params.frame_weights)
            output_features.append(output)

        return output_features


# Register Simulation as a pytree node
register_pytree_node(
    Simulation,
    Simulation.tree_flatten,
    Simulation.tree_unflatten,
)
