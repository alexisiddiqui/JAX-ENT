from jaxent.src.data.splitting.mapping import SparseFragmentMapping
import jax.numpy as jnp
from jax.experimental import sparse

from jaxent.src.data.splitting.mapping import (
    PairIndexMapping,
    SparseFragmentMapping,
    create_pair_index_mapping,
)
from jaxent.src.data.splitting.sparse_map import apply_sparse_mapping
from jaxent.src.interfaces.topology.core import Partial_Topology


def test_sparse_fragment_mapping():
    # Shape: n_fragments=2, n_residues=3
    dense_map = jnp.array([
        [1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0]
    ])
    bcoo_map = sparse.bcoo_fromdense(dense_map)
    mapping = SparseFragmentMapping(sparse_map=bcoo_map)
    
    predictions = jnp.array([10.0, 20.0, 30.0])
    
    mapped = mapping.apply(predictions)
    expected = apply_sparse_mapping(bcoo_map, predictions)
    
    assert jnp.allclose(mapped, expected)
    assert jnp.allclose(mapped, jnp.array([20.0, 40.0]))




def test_pair_index_mapping():
    mapping = PairIndexMapping(
        indices_i=jnp.array([0, 1]),
        indices_j=jnp.array([2, 3])
    )
    # Pairwise distance matrix (n_residues=4, n_residues=4)
    predictions = jnp.array([
        [0, 1, 10, 3],
        [1, 0, 4, 20],
        [10, 4, 0, 5],
        [3, 20, 5, 0]
    ])
    
    mapped = mapping.apply(predictions)
    # expect predictions[0, 2] = 10, predictions[1, 3] = 20
    assert jnp.allclose(mapped, jnp.array([10, 20]))


def test_create_pair_index_mapping():
    import dataclasses
    
    top1 = Partial_Topology(chain="A", residues=[1], fragment_index=0)
    top2 = Partial_Topology(chain="A", residues=[2], fragment_index=1)
    top3 = Partial_Topology(chain="A", residues=[3], fragment_index=2)
    top4 = Partial_Topology(chain="B", residues=[1], fragment_index=3)

    feature_topology = [top1, top2, top3, top4]
    
    @dataclasses.dataclass
    class MockXLMS:
        top: Partial_Topology
        top_j: Partial_Topology
        
    xlms_datapoints = [
        MockXLMS(top=top1, top_j=top3), # should map to 0, 2
        MockXLMS(top=top2, top_j=top4), # should map to 1, 3
    ]
    
    mapping = create_pair_index_mapping(feature_topology, xlms_datapoints)
    
    assert jnp.allclose(mapping.indices_i, jnp.array([0, 1]))
    assert jnp.allclose(mapping.indices_j, jnp.array([2, 3]))
