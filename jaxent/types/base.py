from abc import ABC, abstractmethod
from typing import Generic, NewType, Optional, Protocol, Sequence, TypeVar

import jax.numpy as jnp
import MDAnalysis as mda
import optax
from jax import Array  # This is the correct import path

from jaxent.config.base import Model_Config
from jaxent.data.loading import Experimental_Fragment
from jaxent.interfaces.features import Input_Features, Output_Features
from jaxent.interfaces.model import Model_Parameters
from jaxent.interfaces.simulation import Simulation_Parameters
from jaxent.types.topology import Partial_Topology
from jaxent.utils.jax_fn import frame_average_features, single_pass

m_key = NewType("m_key", str)
m_id = NewType("m_id", str)

T_In = TypeVar("T_In", bound=Input_Features, contravariant=True)
T_Out = TypeVar("T_Out", bound=Output_Features, covariant=True)
T_Params = TypeVar("T_Params", bound=Model_Parameters, contravariant=True)
T_Config = TypeVar("T_Config", bound=Model_Config)
T_Feat_In = TypeVar("T_Feat_In", bound=Input_Features, covariant=True)


class ForwardPass(Protocol[T_In, T_Out, T_Params]):
    """
    The forward pass of a model maps input features to output features using the model's parameters.
    """

    def __call__(self, input_features: T_In, parameters: T_Params) -> T_Out: ...


class Featuriser(Protocol, Generic[T_Feat_In]):
    """
    A featuriser is a callable object that takes in a list of mda.Universes and then returns a list of features.
    """

    def __call__(
        self, ensemble: list[mda.Universe]
    ) -> tuple[T_Feat_In, list[Partial_Topology]]: ...


class ForwardModel(ABC, Generic[T_Params, T_In, T_Config]):
    def __init__(self, config: T_Config) -> None:
        self.config: T_Config = config
        self.compatability: dict[m_key, Experimental_Fragment]
        self.forward: dict[m_key, ForwardPass]
        self.params = config.forward_parameters

    def __post_init__(self):
        self.key = self.config.key

    @abstractmethod
    def initialise(self, ensemble: list[mda.Universe]) -> bool:
        """
        This should be some form of validation to ensure that the data is compatible with the forward model.
        """
        pass

    @abstractmethod
    def featurise(self, ensemble: list[mda.Universe]) -> tuple[T_In, list[Partial_Topology]]:
        pass

    @property
    def forwardpass(self) -> ForwardPass:
        _fp: ForwardPass = self.forward[self.config.key]
        return _fp  # i hope this fixes the typing


class Simulation:
    """
    This is the core object that is used during optimisation
    """

    def __init__(
        self,
        input_features: list[Input_Features],
        forward_models: Sequence[ForwardModel],
        params: Optional[Simulation_Parameters],
        # model_name_index: list[tuple[m_key, int, m_id]],
    ) -> None:
        self.input_features: list[Input_Features] = input_features
        self.forward_models: Sequence[ForwardModel] = forward_models

        self.params = params

        self.forwardpass: Sequence[ForwardPass] = [
            model.forwardpass for model in self.forward_models
        ]
        # self.model_name_index: list[tuple[m_key, int, m_id]] = model_name_index
        # self.outputs: Sequence[Array]

    def __post_init__(self) -> None:
        self._average_feature_map: Array  # a sparse array to map the average features to the single pass to generate the output features
        # self.output_features: dict[m_id, Array]

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

        print("Loaded forward passes")
        print(self.forwardpass)

        print("Simulation initialised successfully.")
        return True

    def forward(self, params: Simulation_Parameters) -> None:
        """
        This function applies the forward models to the input features
        """
        self.params = Simulation_Parameters.normalize_weights(params)
        # print("frame weight sum ", jnp.sum(self.params.frame_weights))

        # print("frame mask sum ", jnp.sum(self.params.frame_mask))

        # mask the frame weights by indexing 0 values in the frame mask
        masked_frame_weights = jnp.where(self.params.frame_mask < 0.5, 0, self.params.frame_weights)
        masked_frame_weights = optax.projections.projection_simplex(masked_frame_weights)

        # print("masked frame weight sum ", jnp.sum(masked_frame_weights))
        # self.params = params
        # print(f"Frame weights: {self.params.frame_weights}")
        # first averages the input parameters using the frame weights
        average_features = map(
            frame_average_features,
            [feat for feat in self.input_features],
            [self.params.frame_weights] * len(self.input_features),
        )

        # reshape average features to match the length of forward models
        # print("forward passes")
        # print(self.forwardpass)
        # map the single_pass function
        output_features = map(
            single_pass, self.forwardpass, average_features, self.params.model_parameters
        )
        # raise NotImplementedError("Need to implement the single pass function")
        # print("Single passed features.")
        # print("Single passed features.")
        self.outputs = list(output_features)
        # update this to use externally defined optimisers - perhaps update should just update the parameters
        # print("Output features: ", self.outputs)
        # change argnum to use enums
