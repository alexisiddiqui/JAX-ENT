from typing import Optional, Sequence

import jax.numpy as jnp
import MDAnalysis as mda

from jaxent.src.custom_types.base import ForwardModel
from jaxent.src.custom_types.features import Input_Features
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.core import Simulation


class Experiment_Builder:
    """
    Class to hold the information of a simulation ensemble and validate whether the forward model is compatible with the ensemble.
    This is created from a list of MDA Universe objects, as well as a forward model object.
    """

    def __init__(
        self,
        universes: list[mda.Universe],
        forward_models: Sequence[ForwardModel],
        experimental_data: Optional[list[ExpD_Dataloader]] = None,
        features: Optional[list[Input_Features]] = None,
    ):
        self.ensembles = universes
        self.experimental_data = experimental_data
        self.forward_models = forward_models
        self.features = features
        # self.output_feature_map: dict[m_id, int]

    def create_model(
        self,
        features: Optional[list[Input_Features]],
        experimental_data: Optional[list[ExpD_Dataloader]],
        simulation_params: Optional[Simulation_Parameters],
    ) -> Simulation:
        if features is not None:
            self.features = features

        if experimental_data is not None:
            self.experimental_data = experimental_data

        if self.experimental_data is None:
            raise ValueError("No experimental data was provided. Exiting.")

        # assert that all experimental datasets contain not None self.train, self.val and self.test attributes
        assert all(
            [
                data.train is not None and data.val is not None and data.test is not None
                for data in self.experimental_data
            ]
        ), (
            "Experimental datasets must contain train, val and test attributes - please split the dataset and create the mapping"
        )

        if simulation_params is not None:
            self.params = simulation_params

        if len(self.forward_models) == 0:
            raise ValueError("No forward models were successfully initialised. Exiting.")
        if len(self.experimental_data) == 0:
            raise ValueError("No experimental data was successfully initialised. Exiting.")
        if self.features is None:
            raise ValueError("No input features were successfully initialised. Exiting.")

        assert len(self.experimental_data) == len(self.forward_models), (
            "Number of forward models must be equal to number of experimental datasets"
        )
        assert len(self.experimental_data) == len(self.features), (
            "Number of input features must be equal to number of experimental datasets"
        )
        if self.params is None:
            print("No simulation parameters were provided. Loading default parameters.")
            self.params = self.load_default_parameters()

        assert len(self.params.model_parameters) == len(self.forward_models), (
            "Number of forward models must be equal to number of forward model parameters settings"
        )
        # only add forward models that have been successfully initialised
        valid_models = self.validate_forward_models()
        # remove entities that align with none
        valid_data = [
            data[0] for data in zip(self.experimental_data, valid_models) if data[1] is not None
        ]

        self.experimental_data = valid_data
        self.forward_models = [model for model in valid_models if model is not None]

        simulation = Simulation(
            forward_models=self.forward_models, input_features=self.features, params=self.params
        )

        return simulation

    def load_default_parameters(self) -> Simulation_Parameters:
        # load default parameters
        raise NotImplementedError("Default parameters have not been implemented yet.")
        assert self.features is not None, "No input features were provided. Exiting."

        frame_weights = jnp.array([1 / len(self.features)] * len(self.features))
        forward_model_weights = jnp.array([1 / len(self.forward_models)] * len(self.forward_models))
        model_parameters = [model.params for model in self.forward_models]

        return Simulation_Parameters(
            frame_weights=frame_weights,
            forward_model_weights=forward_model_weights,
            model_parameters=model_parameters,
        )

    def validate_forward_models(self) -> list[ForwardModel | None]:
        validated_models: list[ForwardModel | None] = []
        for model in self.forward_models:
            print(f"Initialising {model}")

            if model.initialise(self.ensembles):
                validated_models.append(model)
            else:
                validated_models.append(None)
                UserWarning(f"Model {model} failed to initialise. Skipping this model.")

        return validated_models

    # def build_model_ids(self):
    #     """
    #     This method checks the keys of the input features and the names of the forward models
    #     If not all the names are unique - attempts to build new ones
    #     First checks if the forward model config keys are unique
    #     Then checks if the config names are unique
    #     if neither are unique - attempts to create a combination of key and name
    #     Otherwise creates a name using the key and the index
    #     """
    #     forward_model_keys = [str(model.config.key) for model in self.forward_models]

    #     config_names = [str(model.config.name) for model in self.forward_models]

    #     if len(set(forward_model_keys)) == len(self.forward_models):
    #         model_ids = [m_id(key) for key in forward_model_keys]

    #     else:
    #         print(
    #             "Forward model keys are not unique. Attempting to build new keys from config names."
    #         )

    #     if len(set(config_names)) == len(self.forward_models):
    #         model_ids = [m_id(name) for name in config_names]

    #     else:
    #         print(
    #             "Forward model names are not unique. Attempting to build new keys from config names."
    #         )

    #     keys_names = [f"{key}_{name}" for key, name in zip(forward_model_keys, config_names)]

    #     if len(set(keys_names)) == len(self.forward_models):
    #         model_ids = [m_id(key_name) for key_name in keys_names]

    #     else:
    #         print(
    #             "Failed to build unique keys from model keys and config names. Using key_name and index."
    #         )

    #     indexed_keys_names = [str(idx) + "_" + key_name for idx, key_name in enumerate(keys_names)]

    #     model_ids = [m_id(key_name) for key_name in indexed_keys_names]

    #     assert len(set(model_ids)) == len(self.forward_models), "Model IDs are not unique"

    #     self.model_name_index = list(zip(forward_model_keys, config_names, model_ids))
