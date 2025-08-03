from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

from jaxent.src.custom_types.key import m_key
from jaxent.src.interfaces.topology import Partial_Topology, PTSerialiser


@dataclass()
class ExpD_Datapoint:
    """
    Base class for experimental data - grouped into subdomain fragments
    Limitation is that it only covers a single chain - which should be fine in most cases.
    """

    top: Partial_Topology
    key: ClassVar[m_key]

    # Class registry to map keys to classes
    _registry: ClassVar[Dict[str, Type["ExpD_Datapoint"]]] = {}

    def __init_subclass__(cls, **kwargs):
        """Register subclasses automatically"""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "key") and cls.key is not None:
            cls._registry[str(cls.key)] = cls

    @abstractmethod
    def extract_features(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in the child class.")

    @classmethod
    @abstractmethod
    def _create_from_features(
        cls, topology: Partial_Topology, features: np.ndarray
    ) -> "ExpD_Datapoint":
        """
        Create an instance from features array.

        This method should be implemented by subclasses to handle specific feature formats.
        """
        raise NotImplementedError("This method must be implemented in the child class.")

    @classmethod
    def _build_file_paths(
        cls,
        json_path: Optional[Union[str, Path]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        base_name: Optional[str] = None,
    ) -> tuple[Path, Path]:
        """
        Build JSON and CSV file paths, inferring missing paths from provided ones.

        Args:
            json_path: Path to JSON file (topology data)
            csv_path: Path to CSV file (feature data)
            base_name: Base name to use if both paths are None

        Returns:
            Tuple of (json_path, csv_path)

        Raises:
            ValueError: If neither path nor base_name is provided
        """
        if json_path and str(json_path):
            json_path = Path(json_path)
            if csv_path is None or not str(csv_path):
                # Infer CSV path from JSON path
                csv_path = json_path.with_suffix(".csv")
                # Handle common naming patterns
                if "topology" in json_path.name:
                    csv_path = Path(str(csv_path).replace("topology", "features"))
                elif json_path.name.startswith("topology_"):
                    csv_path = json_path.parent / json_path.name.replace(
                        "topology_", "features_", 1
                    )
            else:
                csv_path = Path(csv_path)

        elif csv_path and str(csv_path):
            csv_path = Path(csv_path)
            # Infer JSON path from CSV path
            json_path = csv_path.with_suffix(".json")
            # Handle common naming patterns
            if "features" in csv_path.name:
                json_path = Path(str(json_path).replace("features", "topology"))
            elif csv_path.name.startswith("features_"):
                json_path = csv_path.parent / csv_path.name.replace("features_", "topology_", 1)

        elif base_name and str(base_name):
            base_path = Path(base_name)
            json_path = base_path.with_suffix(".json")
            csv_path = base_path.with_suffix(".csv")

        else:
            raise ValueError("Must provide at least one of json_path, csv_path, or base_name")

        return Path(json_path), Path(csv_path)

    @classmethod
    def save_list_to_files(
        cls,
        datapoints: List["ExpD_Datapoint"],
        json_path: Optional[Union[str, Path]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        base_name: Optional[str] = None,
        validate_homogeneous: bool = True,
    ) -> None:
        """
        Save a list of ExpD_Datapoint objects to JSON (topology) and CSV (features) files.

        Args:
            datapoints: List of ExpD_Datapoint objects to save
            json_path: Path to save topology JSON file
            csv_path: Path to save features CSV file
            base_name: Base name for files if paths not specified
            validate_homogeneous: If True, ensure all datapoints are the same subclass

        Raises:
            ValueError: If datapoints list is empty or contains mixed types when validation enabled
            IOError: If files cannot be written
        """
        if not datapoints:
            raise ValueError("Cannot save empty list of datapoints")

        # Validate homogeneous types if requested
        if validate_homogeneous:
            first_type = type(datapoints[0])
            if not all(isinstance(dp, first_type) for dp in datapoints):
                types_found = set(type(dp).__name__ for dp in datapoints)
                raise ValueError(
                    f"Mixed datapoint types found: {types_found}. "
                    f"Set validate_homogeneous=False to allow mixed types."
                )

        # Build file paths
        json_path, csv_path = cls._build_file_paths(json_path, csv_path, base_name)

        try:
            # Extract topologies
            topologies = [dp.top for dp in datapoints]

            # Save topologies to JSON
            PTSerialiser.save_list_to_json(topologies, json_path)

            # Extract features
            features_list = []
            datapoint_types = []

            for dp in datapoints:
                features = dp.extract_features()
                # Flatten features to 1D for CSV storage
                flattened_features = features.flatten()
                features_list.append(flattened_features)
                datapoint_types.append(str(dp.key))

            # Convert to numpy array for consistent CSV writing
            # Handle variable feature lengths by padding with NaN
            max_length = max(len(features) for features in features_list)
            padded_features = []

            for features in features_list:
                if len(features) < max_length:
                    padded = np.pad(
                        features,
                        (0, max_length - len(features)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    padded_features.append(padded)
                else:
                    padded_features.append(features)

            features_array = np.array(padded_features)

            # Create DataFrame with metadata
            df = pd.DataFrame(features_array)

            # Add metadata columns
            df.insert(0, "datapoint_type", datapoint_types)
            df.insert(1, "feature_length", [len(f) for f in features_list])

            # Ensure parent directory exists
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            # Save features to CSV
            df.to_csv(csv_path, index=False)

            print(f"Saved {len(datapoints)} datapoints:")
            print(f"  Topologies: {json_path}")
            print(f"  Features: {csv_path} (shape: {features_array.shape})")

        except Exception as e:
            # Clean up partial files on error
            for path in [json_path, csv_path]:
                if path.exists():
                    try:
                        path.unlink()
                    except:
                        pass
            raise IOError(f"Failed to save datapoints: {e}")

    @classmethod
    def load_list_from_files(
        cls,
        json_path: Optional[Union[str, Path]] = None,
        csv_path: Optional[Union[str, Path]] = None,
        base_name: Optional[str] = None,
        datapoint_class: Optional[Type["ExpD_Datapoint"]] = None,
    ) -> List["ExpD_Datapoint"]:
        """
        Load a list of ExpD_Datapoint objects from JSON (topology) and CSV (features) files.

        Args:
            json_path: Path to topology JSON file
            csv_path: Path to features CSV file
            base_name: Base name for files if paths not specified
            datapoint_class: Specific subclass to instantiate. If None, will use registry lookup.

        Returns:
            List of ExpD_Datapoint objects

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If file formats are invalid or data is inconsistent
            IOError: If files cannot be read
        """
        # Build file paths
        json_path, csv_path = cls._build_file_paths(json_path, csv_path, base_name)

        # Check files exist
        if not json_path.exists():
            raise FileNotFoundError(f"Topology file not found: {json_path}")
        if not csv_path.exists():
            raise FileNotFoundError(f"Features file not found: {csv_path}")

        try:
            # Load topologies
            topologies = PTSerialiser.load_list_from_json(json_path)

            # Load features
            df = pd.read_csv(csv_path)

            # Validate DataFrame structure
            required_cols = ["datapoint_type", "feature_length"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV missing required columns: {missing_cols}")

            # Extract metadata
            datapoint_types = df["datapoint_type"].tolist()
            feature_lengths = df["feature_length"].tolist()

            # Extract feature data (skip metadata columns)
            feature_columns = [col for col in df.columns if col not in required_cols]
            features_array = df[feature_columns].to_numpy()

            # Validate data consistency
            if len(topologies) != len(features_array):
                raise ValueError(
                    f"Mismatch between topology count ({len(topologies)}) "
                    f"and feature count ({len(features_array)})"
                )

            # Create datapoint objects
            datapoints = []

            for i, (topology, features_row, dp_type, feature_length) in enumerate(
                zip(topologies, features_array, datapoint_types, feature_lengths)
            ):
                # Remove padding (NaN values) from features
                valid_features = features_row[:feature_length]

                # Determine which class to instantiate
                if datapoint_class is not None:
                    target_class = datapoint_class
                else:
                    # Look up class from registry
                    if dp_type not in cls._registry:
                        available_types = list(cls._registry.keys())
                        raise ValueError(
                            f"Unknown datapoint type '{dp_type}' at index {i}. "
                            f"Available types: {available_types}"
                        )
                    target_class = cls._registry[dp_type]

                # Create datapoint instance
                datapoint = cls._create_datapoint_from_features(
                    target_class, topology, valid_features, i
                )
                datapoints.append(datapoint)

            print(f"Loaded {len(datapoints)} datapoints from {json_path.parent}")
            print(f"  Topologies: {json_path}")
            print(f"  Features: {csv_path} (shape: {features_array.shape})")

            return datapoints

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError, IOError)):
                raise
            else:
                raise IOError(f"Failed to load datapoints: {e}")

    @classmethod
    def _create_datapoint_from_features(
        cls,
        target_class: Type["ExpD_Datapoint"],
        topology: Partial_Topology,
        features: np.ndarray,
        index: int,
    ) -> "ExpD_Datapoint":
        """
        Create a datapoint instance from features array.

        This method handles the conversion from generic features back to
        class-specific attributes. Subclasses may override this for custom logic.

        Args:
            target_class: The ExpD_Datapoint subclass to instantiate
            topology: The Partial_Topology for this datapoint
            features: The feature array for this datapoint
            index: Index in the original list (for error reporting)

        Returns:
            Instance of target_class

        Raises:
            ValueError: If features cannot be converted to target class format
        """
        try:
            # Handle different known subclasses
            # if hasattr(target_class, "_create_from_features"):
            #     # Allow subclasses to define custom creation logic
            return target_class._create_from_features(topology, features)

        except Exception as e:
            raise ValueError(
                f"Failed to create {target_class.__name__} from features at index {index}: {e}"
            )

    # @classmethod
    # def _default_feature_conversion(
    #     cls,
    #     target_class: Type["ExpD_Datapoint"],
    #     topology: Partial_Topology,
    #     features: np.ndarray,
    #     index: int,
    # ) -> "ExpD_Datapoint":
    #     """
    #     Default feature conversion logic.

    #     This method attempts to create datapoint instances by inspecting
    #     the target class and making reasonable assumptions about feature format.
    #     """
    #     class_name = target_class.__name__

    #     if class_name == "HDX_peptide":
    #         # HDX_peptide expects dfrac as list[float]
    #         return target_class(top=topology, dfrac=features.tolist())

    #     elif class_name == "HDX_protection_factor":
    #         # HDX_protection_factor expects single protection_factor value
    #         if len(features) != 1:
    #             raise ValueError(f"HDX_protection_factor expects 1 feature, got {len(features)}")
    #         return target_class(top=topology, protection_factor=float(features[0]))

    #     else:
    #         # Generic approach: try to find the first non-topology field
    #         # and assign features to it
    #         import inspect

    #         sig = inspect.signature(target_class.__init__)

    #         # Get field names excluding 'self' and 'top'
    #         param_names = [name for name in sig.parameters.keys() if name not in ("self", "top")]

    #         if len(param_names) == 1:
    #             # Single additional parameter - assign features
    #             param_name = param_names[0]
    #             if len(features) == 1:
    #                 value = float(features[0])
    #             else:
    #                 value = features.tolist()
    #             return target_class(top=topology, **{param_name: value})
    #         else:
    #             raise ValueError(
    #                 f"Cannot automatically convert features for class {class_name}. "
    #                 f"Please implement _create_from_features class method."
    #             )

    @classmethod
    def load_from_directory(
        cls,
        directory: Union[str, Path],
        dataset_name: str = "full_dataset",
        datapoint_class: Optional[Type["ExpD_Datapoint"]] = None,
    ) -> List["ExpD_Datapoint"]:
        """
        Convenience method to load datapoints from a directory using standard naming.

        Args:
            directory: Directory containing the files
            dataset_name: Base name of the files (e.g., 'train', 'val', 'full_dataset')
            datapoint_class: Specific subclass to instantiate

        Returns:
            List of ExpD_Datapoint objects
        """
        directory = Path(directory)
        base_path = directory / dataset_name

        return cls.load_list_from_files(base_name=str(base_path), datapoint_class=datapoint_class)

    @classmethod
    def save_to_directory(
        cls,
        datapoints: List["ExpD_Datapoint"],
        directory: Union[str, Path],
        dataset_name: str = "full_dataset",
        validate_homogeneous: bool = True,
    ) -> None:
        """
        Convenience method to save datapoints to a directory using standard naming.

        Args:
            datapoints: List of ExpD_Datapoint objects to save
            directory: Directory to save files in
            dataset_name: Base name for the files
            validate_homogeneous: If True, ensure all datapoints are same subclass
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        base_path = directory / dataset_name

        cls.save_list_to_files(
            datapoints=datapoints,
            base_name=str(base_path),
            validate_homogeneous=validate_homogeneous,
        )
