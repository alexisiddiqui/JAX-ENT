from dataclasses import dataclass
from functools import partial
from typing import Generic, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import sparse

from jaxent.src.custom_types import T_ExpD
from jaxent.src.custom_types.datapoint import ExpD_Datapoint
from jaxent.src.custom_types.features import Input_Features
from jaxent.src.custom_types.key import m_id, m_key
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.interfaces.topology import Partial_Topology


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["y_true", "residue_feature_ouput_mapping"],
    meta_fields=["data"],
)
@dataclass(frozen=True, slots=True)
class Dataset:
    data: Sequence[ExpD_Datapoint]
    y_true: Array
    residue_feature_ouput_mapping: sparse.BCOO


class ExpD_Dataloader(Generic[T_ExpD]):
    """
    Class to hold the information of the experimental data.
    This is created from a list of HDX_peptide objects or list of HDX_protection_factor objects.
    Once loaded, the dataset then extracts the information into a optimisable format.
    """

    data: Sequence[
        T_ExpD
    ]  # never accessed for jax operations but required to instnatiate the class - metadata
    y_true: np.ndarray  # never accessed for jax operations - metadata
    top: list[Partial_Topology]  # metadata
    train: Dataset  # static but accessed during training - must be traced
    val: Dataset  # static but accessed during training - must be traced
    test: Dataset  # static but accessed during training - must be traced
    key: m_key  # metadata key
    id: m_id = m_id("id")  # metadata id - not implemented yet, dummy value for now

    def __init__(self, data: Sequence[T_ExpD]) -> None:
        self.data: Sequence[T_ExpD] = data
        print("Data Length:", len(self.data))
        self.y_true: np.ndarray = self.extract_data()  # warning this is not a jax array

        if isinstance(self.y_true, np.ndarray):
            UserWarning("y_true is a numpy array. Do not use this for training.")
        elif isinstance(self.y_true, Array):
            UserWarning("y_true is a jax array. Do not use this for training.")

        # assert keys are all the same
        assert len(set([data.key for data in self.data])) == 1, (
            "Keys are not the same. Datasets are comprised of a single type of experimental data."
        )
        # assert that all topology fragments are unique
        assert len({id(data.top) for data in self.data}) == len(self.data), (
            "Topology fragments are not unique/missing. Exiting."
        )
        self.top: list[Partial_Topology] = [data.top for data in self.data]

        print("First Partial Topology:", self.top[0])

        if any([top.fragment_index is None for top in self.top]):
            UserWarning("Topology fragments are missing indices. Assigning indices to fragments.")
            for idx, _data in enumerate(self.data):
                _data.top.fragment_index = idx

        self.key = self.data[0].key

        # self.train: Dataset
        # self.val: Dataset
        # self.test: Dataset

    def __post_init__(self) -> None:
        # check that every topology fragment contains an fragment if not - assigns indices
        _tops = [data.top for data in self.data]
        _indices = [top.fragment_index for top in _tops]

        if len(_indices) != len(set(_indices)):
            UserWarning("Topology fragments are not unique. Assigning indices to fragments.")
            for idx, top in enumerate(_tops):
                top.fragment_index = idx

    def extract_data(self) -> np.ndarray:
        """
        Map across every eleemtn in data and stack the features into a single array
        """
        return np.hstack([_exp_data.extract_features() for _exp_data in self.data])

    def create_datasets(
        self,
        features: Input_Features,
        feature_topology: Sequence[Partial_Topology],
        train_data: Sequence[T_ExpD] | None = None,
        val_data: Sequence[T_ExpD] | None = None,
        test_data: Sequence[T_ExpD] | None = None,
    ) -> None:
        """
        Create train, val, and test Dataset objects with sparse mappings.

        Args:
            features: Input features to map from
            feature_topology: Topology fragments matching input features
            train_data: Training datapoints. If None, uses all data from dataloader
            val_data: Validation datapoints. If None, uses all data from dataloader
            test_data: Test datapoints. If None, uses all data from dataloader
        """
        # Use all data if specific splits not provided
        if train_data is None:
            train_data = self.data
        if val_data is None:
            val_data = self.data
        if test_data is None:
            test_data = self.data

        # Create sparse mappings
        train_sparse_map = create_sparse_map(features, feature_topology, train_data)
        val_sparse_map = create_sparse_map(features, feature_topology, val_data)
        test_sparse_map = create_sparse_map(features, feature_topology, test_data)

        # Create Dataset objects
        self.train = Dataset(
            data=train_data,
            y_true=jnp.array([data.extract_features() for data in train_data]),
            residue_feature_ouput_mapping=train_sparse_map,
        )

        self.val = Dataset(
            data=val_data,
            y_true=jnp.array([data.extract_features() for data in val_data]),
            residue_feature_ouput_mapping=val_sparse_map,
        )

        self.test = Dataset(
            data=test_data,
            y_true=jnp.array([data.extract_features() for data in test_data]),
            residue_feature_ouput_mapping=test_sparse_map,
        )

    def tree_flatten(self):
        """
        Flatten the ExpD_Dataloader into leaves (JAX-traceable components) and auxiliary data.

        Returns:
            tuple: (leaves, auxiliary_data)
                - leaves: Components that JAX will transform
                - auxiliary_data: Metadata that JAX won't transform
        """
        # Only Dataset objects are JAX-traceable and need to be transformed
        leaves = (self.train, self.val, self.test)

        # Everything else is metadata that doesn't need transformation
        aux_data = (self.data, self.y_true, self.top, self.key, self.id)

        return leaves, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        """
        Reconstruct an ExpD_Dataloader from leaves and auxiliary data.

        Args:
            aux_data: Tuple of metadata components
            leaves: Tuple of transformed Dataset objects

        Returns:
            ExpD_Dataloader: Reconstructed instance
        """
        # Create a new instance without calling __init__ to avoid recomputation
        instance = object.__new__(cls)

        # Set attributes from transformed leaves
        instance.train, instance.val, instance.test = leaves

        # Set attributes from auxiliary data
        instance.data, instance.y_true, instance.top, instance.key, instance.id = aux_data

        return instance


# Register ExpD_Dataloader as a pytree node
jax.tree_util.register_pytree_node(
    ExpD_Dataloader,
    ExpD_Dataloader.tree_flatten,
    ExpD_Dataloader.tree_unflatten,
)
