from abc import abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from jax import Array
from jax.experimental import sparse

from jaxent.types.base import m_id, m_key
from jaxent.types.topology import Partial_Topology


@dataclass()
class Experimental_Fragment:
    """
    Base class for experimental data - grouped into subdomain fragments
    Limtation is that it only covers a single chain - which should be fine in most cases.
    """

    top: Partial_Topology
    key: m_key

    @abstractmethod
    def extract_features(self) -> np.ndarray:
        raise NotImplementedError("This method must be implemented in the child class.")


@dataclass(frozen=True, slots=True)
class Dataset:
    data: Sequence[Experimental_Fragment]
    y_true: Array
    residue_feature_ouput_mapping: sparse.BCOO


class Experimental_Dataset:
    """
    Class to hold the information of the experimental data.
    This is created from a list of HDX_peptide objects or list of HDX_protection_factor objects.
    Once loaded, the dataset then extracts the information into a optimisable format.
    """

    def __init__(self, data: Sequence[Experimental_Fragment]):
        self.data = data
        self.y_true = self.extract_data()

        self.train: Dataset
        self.val: Dataset
        self.test: Dataset

        # assert keys are all the same
        assert len(set([data.key for data in self.data])) == 1, (
            "Keys are not the same. Datasets are comprised of a single type of experimental data."
        )
        # assert that all topology fragments are unique
        assert len(set([data.top for data in self.data])) == len(self.data), (
            "Topology fragments are not unique/missing. Exiting."
        )
        if any([data.top.fragment_index is None for data in self.data]):
            UserWarning("Topology fragments are missing indices. Assigning indices to fragments.")
            for idx, _data in enumerate(self.data):
                _data.top.fragment_index = idx

        self.top: list[Partial_Topology] = [data.top for data in self.data]
        self.key = self.data[0].key
        self.id: m_id  # to be set later

    def __post__init__(self):
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
