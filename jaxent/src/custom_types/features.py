import importlib
import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, Sequence, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxent.src.custom_types import T_Feat_In
from jaxent.src.custom_types.key import m_key

T_Features = TypeVar("T_Features", bound="AbstractFeatures")


class AbstractFeatures(ABC):
    """Base class providing save/load functionality for feature classes."""

    __slots__: ClassVar[tuple[str]]
    __features__: ClassVar[set[str]]

    @classmethod
    def _get_ordered_slots(cls) -> tuple[str, ...]:
        """Get slots in a deterministic order, including child classes"""
        all_slots = []
        for c in cls.__mro__:
            if hasattr(c, "__slots__"):
                all_slots.extend(c.__slots__)
        return tuple(dict.fromkeys(all_slots))

    @classmethod
    def _get_grouped_slots(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """
        Get Array and static slots.

        Dynamic slots are all slots listed in cls.__features__.

        Returns:
            tuple: (dynamic_slots, static_slots)
        """
        dynamic_slots = []
        static_slots = []
        for slot in cls._get_ordered_slots():
            if slot in cls.__features__:
                dynamic_slots.append(slot)
            else:
                static_slots.append(slot)
        return tuple(dynamic_slots), tuple(static_slots)

    def tree_flatten(self) -> tuple[tuple[Array, ...], tuple[Any, ...]]:
        """Flatten the object for JAX tree operations."""
        dynamic_slots, static_slots = self._get_grouped_slots()

        # Dynamic parameters become leaves
        arrays = tuple(
            jnp.asarray(getattr(self, slot)).astype(jnp.float32) for slot in dynamic_slots
        )

        # Static parameters go in aux data
        static_data = tuple(getattr(self, slot) for slot in static_slots)

        return arrays, static_data

    @classmethod
    def tree_unflatten(
        cls: type[T_Features], static_data: tuple[Any, ...], arrays: tuple[Array, ...]
    ) -> T_Features:
        """Unflatten the object from JAX tree operations."""
        dynamic_slots, static_slots = cls._get_grouped_slots()

        # Rebuild parameter dict
        params: dict[str, Any] = {}

        # Dynamic parameters from arrays
        params.update(zip(dynamic_slots, arrays))

        # Static parameters from aux data
        params.update(zip(static_slots, static_data))

        return cls(**params)

    def cast_to_jax(self: T_Features) -> T_Features:
        """Casts __features__ to JAX arrays, returning a new instance."""
        arrays, static_data = self.tree_flatten()
        return self.tree_unflatten(static_data, arrays)

    def save(self, filepath: str) -> None:
        """
        Save features to a .npz file.

        Args:
            filepath: Path where to save the features. If it doesn't end with .npz,
                    the extension will be added automatically.
        """
        if not filepath.endswith(".npz"):
            filepath += ".npz"

        # Get flattened data
        arrays, static_data = self.tree_flatten()
        dynamic_slots, _ = self._get_grouped_slots()

        # Create save dictionary with class metadata
        save_dict = {
            "__class_module__": self.__class__.__module__,
            "__class_name__": self.__class__.__qualname__,
            "__static_data__": static_data,
            "__dynamic_slots__": dynamic_slots,
        }

        # Add arrays with their corresponding slot names
        for i, slot in enumerate(dynamic_slots):
            save_dict[slot] = arrays[i]

        # Use jnp.savez to save everything
        jnp.savez(filepath, **save_dict)

    @classmethod
    def load(cls: type[T_Features], filepath: str) -> T_Features:
        """
        Load features from a .npz file.

        Args:
            filepath: Path to the saved features file.

        Returns:
            Instance of the appropriate features subclass.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ImportError: If the saved class cannot be imported.
            KeyError: If the file is missing required metadata.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Features file not found: {filepath}")

        # Load the data
        data = jnp.load(filepath, allow_pickle=True)

        try:
            # Extract class metadata
            class_module = str(data["__class_module__"].item())
            class_name = str(data["__class_name__"].item())
            static_data = tuple(data["__static_data__"])
            dynamic_slots = tuple(data["__dynamic_slots__"])

            # Import and get the actual class
            module = importlib.import_module(class_module)
            actual_class = getattr(module, class_name)

            # Verify that the loaded class is compatible with the calling class
            if not issubclass(actual_class, cls):
                raise TypeError(
                    f"Loaded class {actual_class.__name__} is not a subclass of {cls.__name__}"
                )

            # Extract arrays in the correct order
            arrays = tuple(data[slot] for slot in dynamic_slots)

            # Create the instance
            instance = actual_class.tree_unflatten(static_data, arrays)

            return instance

        except KeyError as e:
            raise KeyError(f"File {filepath} is missing required metadata: {e}")
        except ImportError as e:
            raise ImportError(f"Cannot import class from saved file: {e}")

    def save_features(self, filepath: str) -> None:
        """
        Save only the __features__ attributes to a .npz file.

        Args:
            filepath: Path where to save the features. If it doesn't end with .npz,
                    the extension will be added automatically.
        """
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        # Create dictionary with only the features
        features_dict = {}
        for slot_name in self._get_ordered_slots():
            value = getattr(self, slot_name)
            if slot_name in self.__features__:
                features_dict[slot_name] = jnp.asarray(value)
            elif value is None:
                features_dict[slot_name] = "__NONE__"  # Special marker for None
            else:
                features_dict[slot_name] = value

        # Save only the features
        jnp.savez(filepath, **features_dict)

    @classmethod
    def load_features(cls: type[T_Features], filepath: str) -> T_Features:
        """
        Load features from a .npz file and create a new instance using only __features__.

        Args:
            filepath: Path to the saved features file.

        Returns:
            New instance of the class with the loaded features.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        if not filepath.endswith(".npz"):
            filepath += ".npz"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Features file not found: {filepath}")

        # Load the data
        try:
            data = jnp.load(filepath, allow_pickle=True)
        except Exception as e:
            raise ValueError(
                f"Failed to load features from {filepath}. The file might be corrupted or not a valid .npz file. Original error: {e}"
            )

        # Extract all attributes that exist in the file and are part of the class slots
        features_kwargs = {}
        for slot_name in cls._get_ordered_slots():
            if slot_name in data:
                loaded_value = data[slot_name]
                if isinstance(loaded_value, np.ndarray) and loaded_value.shape == () and loaded_value.item() == "__NONE__":
                    features_kwargs[slot_name] = None
                else:
                    features_kwargs[slot_name] = loaded_value

        # Get the signature of the class's __init__ method
        import inspect

        sig = inspect.signature(cls.__init__)
        required_params = {
            name
            for name, param in sig.parameters.items()
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and param.default is inspect.Parameter.empty
            and name != "self"
        }

        # Check for missing required features
        missing_features = required_params - features_kwargs.keys()
        if missing_features:
            raise TypeError(
                f"{cls.__name__}.__init__() missing required features: {', '.join(missing_features)}"
            )

        # Create and return new instance with all loaded attributes
        loaded = cls(**features_kwargs)
        return loaded


class Output_Features(AbstractFeatures):
    """Abstract base class for output features."""

    key: ClassVar[m_key]

    @property
    @abstractmethod
    def output_shape(self) -> tuple[float, ...]:
        """Get the shape of the output features."""
        ...

    @abstractmethod
    def y_pred(self) -> Array:
        """Get the predicted output as a JAX array."""


class Input_Features(AbstractFeatures, Generic[T_Feat_In]):
    """Abstract base class for input features."""

    key: ClassVar[set[m_key]]

    @property
    @abstractmethod
    def features_shape(self) -> tuple[float | int, ...]:
        """Get the shape of the input features."""
        ...

    @property
    @abstractmethod
    def feat_pred(self) -> Sequence[Array]:
        """Get the feature predictions as a sequence of JAX arrays."""
        ...
