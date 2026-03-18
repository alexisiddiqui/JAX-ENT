from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, runtime_checkable, Union

import jax
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse

from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.interfaces.topology.core import Partial_Topology


@runtime_checkable
class DataMapping(Protocol):
    """Protocol for mapping model outputs to observation space."""

    def apply(self, predictions: Array) -> Array: ...


@partial(jax.tree_util.register_dataclass, data_fields=["sparse_map"], meta_fields=[])
@dataclass(frozen=True, slots=True)
class SparseFragmentMapping:
    """Wraps existing BCOO sparse map (HDX). Shape: (n_fragments, n_residues)."""

    sparse_map: sparse.BCOO

    def apply(self, predictions: Array) -> Array:
        return apply_sparse_mapping(self.sparse_map, predictions)


@partial(jax.tree_util.register_dataclass, data_fields=["indices"], meta_fields=[])
@dataclass(frozen=True)
class QSubsetMapping:
    """Maps full (501,) q-point predictions to train/val subsets."""

    indices: jnp.ndarray

    def apply(self, pred):
        return pred[self.indices]


@partial(jax.tree_util.register_dataclass, data_fields=["indices_i", "indices_j"], meta_fields=[])
@dataclass(frozen=True, slots=True)
class PairIndexMapping:
    """Extracts specific residue pairs from a pairwise distance matrix (XL-MS).
    indices_i, indices_j: arrays of shape (n_observations,)
    Each observation maps to predictions[indices_i[k], indices_j[k]]."""

    indices_i: Array
    indices_j: Array

    def apply(self, predictions: Array) -> Array:
        return predictions[self.indices_i, self.indices_j]


def create_pair_index_mapping(
    feature_topology: Sequence[Partial_Topology],
    xlms_datapoints: Sequence[Any],
) -> PairIndexMapping:
    """Build a PairIndexMapping from XL-MS datapoints and feature topology.

    Maps each datapoint's (top, top_j) pair to indices into the feature
    topology's residue ordering.
    """
    # For each datapoint, find the index of top and top_j in feature_topology
    topology_lookup = {}
    for top in feature_topology:
        if top.fragment_index is not None:
            topology_lookup[(top.chain, tuple(top.residues))] = top.fragment_index

    indices_i = []
    indices_j = []

    for dp in xlms_datapoints:
        top_i = dp.top
        top_j = dp.top_j

        idx_i = topology_lookup.get((top_i.chain, tuple(top_i.residues)))
        idx_j = topology_lookup.get((top_j.chain, tuple(top_j.residues)))

        if idx_i is None or idx_j is None:
            raise ValueError(
                f"Could not find matching topology for XLMS datapoint ({top_i.chain}: {top_i.residues}) "
                f"and ({top_j.chain}: {top_j.residues})"
            )

        indices_i.append(idx_i)
        indices_j.append(idx_j)

    return PairIndexMapping(
        indices_i=jnp.array(indices_i, dtype=jnp.int32),
        indices_j=jnp.array(indices_j, dtype=jnp.int32),
    )



def create_q_subset_mapping(indices: Union[Array, list]) -> QSubsetMapping:
    """Create a QSubsetMapping from q-point indices (SAXS analogue of create_sparse_map)."""
    return QSubsetMapping(indices=jnp.array(indices, dtype=jnp.int32))
