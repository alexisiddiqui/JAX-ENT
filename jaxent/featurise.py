from typing import Optional

from jaxent.config.base import FeaturiserSettings, Settings
from jaxent.interfaces.builider import Experiment_Ensemble
from jaxent.interfaces.features import Input_Features
from jaxent.types.base import ForwardModel, Partial_Topology


def featurise(
    ensemble_paths: list[tuple[str, str]],
    output_path: str,
    config_paths: list[str],
    name: str,
    batch_size: Optional[int],
    forward_models: list[str],
    log_path: Optional[str],
    overwrite: bool,
):
    # this function will be the input for the cli
    # this will take in paths and configurations and create the individual objects for analysis
    # TODO create reusable builder methods to generate objects from configuration

    pass


def run_featurise(
    ensemble: Experiment_Ensemble,
    config: Settings | FeaturiserSettings,
    name: Optional[str] = None,
    forward_models: Optional[list[ForwardModel]] = None,
    validate=True,
) -> tuple[list[Input_Features], list[list[Partial_Topology]]]:
    # this function will take in the constructed objects and run the analysis

    if isinstance(config, Settings):
        config = config.featuriser_config
    if not isinstance(config, FeaturiserSettings):
        raise ValueError("Invalid config")

    if name is None:
        name = config.name

    if name is None:
        raise UserWarning("Name is required")

    if validate:
        ensemble.validate_forward_models()

    if forward_models is None:
        _forward_models = ensemble.forward_models
    else:
        _forward_models = forward_models

    features: list[Input_Features] = []
    feat_top: list[list[Partial_Topology]] = []
    for model in _forward_models:
        try:
            _features, _feat_top = model.featurise(ensemble.ensembles)

            features.append(_features)
            feat_top.append(_feat_top)
        except Exception as e:
            print(f"Failed to featurise {model}")
            # warn
            raise UserWarning(f"Failed to featurise {model}, {e}")

    return features, feat_top
