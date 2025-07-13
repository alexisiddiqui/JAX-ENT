from typing import Optional

from icecream import ic  # Import icecream for debugging

from jaxent.src.custom_types.base import ForwardModel, Partial_Topology
from jaxent.src.custom_types.config import FeaturiserSettings, Settings
from jaxent.src.custom_types.features import Input_Features
from jaxent.src.interfaces.builder import Experiment_Builder

ic.disable()


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
    ic.configureOutput(prefix="FEATURISE | ")
    ic("Starting featurisation process")
    ic(ensemble_paths, output_path, config_paths, name)
    ic(batch_size, forward_models, log_path, overwrite)

    # Placeholder for future implementation
    ic("Function currently not implemented")
    pass


def run_featurise(
    ensemble: Experiment_Builder,
    config: Settings | FeaturiserSettings,
    name: Optional[str] = None,
    forward_models: Optional[list[ForwardModel]] = None,
    validate=True,
) -> tuple[list[Input_Features], list[list[Partial_Topology]]]:
    # this function will take in the constructed objects and run the analysis
    ic.configureOutput(prefix="RUN_FEATURISE | ")
    ic("Starting run_featurise")
    ic(f"Ensemble: {ensemble}, Config type: {type(config)}")
    ic(f"Name: {name}, Validate: {validate}")
    ic(f"Forward models provided: {forward_models is not None}")

    if isinstance(config, Settings):
        ic("Converting Settings to FeaturiserSettings")
        config = config.featuriser_config
    if not isinstance(config, FeaturiserSettings):
        ic.format("Invalid config type: {}", type(config))
        raise ValueError("Invalid config")

    if name is None:
        ic("Using name from config")
        name = config.name

    if name is None:
        ic.format("ERROR: Name is required but not provided")
        raise UserWarning("Name is required")

    ic(f"Using name: {name}")

    if validate:
        ic("Validating forward models")
        ensemble.validate_forward_models()
        ic("Forward models validated successfully")

    if forward_models is None:
        ic("Using ensemble's forward models")
        _forward_models = ensemble.forward_models
    else:
        ic(f"Using {len(forward_models)} provided forward models")
        _forward_models = forward_models

    ic(f"Processing {len(_forward_models)} forward models")

    features: list[Input_Features] = []
    feat_top: list[list[Partial_Topology]] = []
    for i, model in enumerate(_forward_models):
        ic(f"Processing model {i + 1}/{len(_forward_models)}: {model}")
        try:
            ic(f"Featurising with model {model}")
            _features, _feat_top = model.featurise(ensemble.ensembles)
            ic(
                f"Featurisation successful, features shape: {_features}, topologies: {len(_feat_top)}"
            )

            features.append(_features)
            feat_top.append(_feat_top)
            ic(f"Updated features list, now contains {len(features)} entries")
        except Exception as e:
            ic.format("Error during featurisation: {} - {}", type(e).__name__, str(e))
            print(f"Failed to featurise {model}")
            # warn
            raise UserWarning(f"Failed to featurise {model}, {e}")

    ic(
        f"Featurisation complete. Returning {len(features)} feature sets and {len(feat_top)} topology sets"
    )
    return features, feat_top
