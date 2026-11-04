"""Opt-in sparse-map row normalisation and the missing-residue (proline) check."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import (
    check_unrepresented_residues,
    find_unrepresented_residues,
)
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.HDX.BV.features import BV_input_features

# Residues 1..12; residue 1 is the N terminus, 5 and 9 are prolines, 12 is an alanine.
SEQUENCE = "GLGYPALGPAMA"
NAMES = {index + 1: letter for index, letter in enumerate(SEQUENCE)}


def _features(residues):
    topology = [
        Partial_Topology(chain="A", residues=[residue], fragment_index=index)
        for index, residue in enumerate(residues)
    ]
    n = len(residues)
    features = BV_input_features(
        heavy_contacts=jnp.ones((n, 2)), acceptor_contacts=jnp.ones((n, 2)), k_ints=jnp.ones(n)
    )
    return features, topology


def _peptides(ranges):
    return [
        HDX_peptide(
            top=Partial_Topology(
                chain="A",
                residues=list(range(start, end + 1)),
                fragment_index=index,
                peptide=True,
                peptide_trim=2,
            ),
            dfrac=[0.1, 0.2],
        )
        for index, (start, end) in enumerate(ranges)
    ]


def _loader(feature_residues, ranges, **options):
    features, topology = _features(feature_residues)
    data = _peptides(ranges)
    loader = ExpD_Dataloader(data=data)
    loader.create_datasets(features=features, feature_topology=topology, **options)
    return loader


RANGES = [(2, 8), (6, 12)]  # trimmed: 4-8 and 8-12


def test_legacy_maps_unchanged_when_disabled():
    loader = _loader([2, 3, 4, 6, 7, 8, 10, 11, 12], RANGES)
    dense = np.asarray(loader.test.data_mapping.sparse_map.todense())
    # 4-8 has 5 trimmed residues, 4 represented (5 is a proline) -> 1/5 weights, row sum 0.8.
    np.testing.assert_allclose(dense.sum(axis=1), [0.8, 0.8])
    np.testing.assert_allclose(dense[0, dense[0] > 0], np.full(4, 0.2))


def test_rows_sum_to_one_when_enabled():
    loader = _loader(
        [2, 3, 4, 6, 7, 8, 10, 11, 12], RANGES, normalize_sparse_map_rows=True, residue_names=NAMES
    )
    for dataset in (loader.train, loader.val, loader.test):
        dense = np.asarray(dataset.data_mapping.sparse_map.todense())
        np.testing.assert_allclose(dense.sum(axis=1), np.ones(2), rtol=1e-6)
        np.testing.assert_allclose(dense[0, dense[0] > 0], np.full(4, 0.25), rtol=1e-6)


def test_proline_only_omissions_pass_and_are_reported():
    features, topology = _features([2, 3, 4, 6, 7, 8, 10, 11, 12])
    data = _peptides(RANGES)
    assert find_unrepresented_residues(topology, data) == {0: [5], 1: [9]}
    assert check_unrepresented_residues(topology, data, residue_names=NAMES) == {0: [5], 1: [9]}


def test_fragment_sequence_supplies_proline_identity():
    _, topology = _features([2, 3, 4, 6, 7, 8])
    data = _peptides([(2, 8)])
    data[0].top.fragment_sequence = SEQUENCE[1:8]
    assert check_unrepresented_residues(topology, data) == {0: [5]}


def test_non_proline_omission_fails_unless_allowed():
    # Residue 12 (alanine) is absent, mirroring the missing MoPrP C-terminal A101.
    feature_residues = [2, 3, 4, 6, 7, 8, 10, 11]
    with pytest.raises(ValueError, match="residue 12 \\(A\\)"):
        _loader(feature_residues, RANGES, normalize_sparse_map_rows=True, residue_names=NAMES)

    with pytest.warns(UserWarning, match="residue 12"):
        loader = _loader(
            feature_residues,
            RANGES,
            normalize_sparse_map_rows=True,
            residue_names=NAMES,
            allow_missing_residues=True,
        )
    dense = np.asarray(loader.test.data_mapping.sparse_map.todense())
    np.testing.assert_allclose(dense.sum(axis=1), np.ones(2), rtol=1e-6)


def test_unknown_residue_identity_is_not_assumed_proline():
    _, topology = _features([2, 3, 4, 6, 7, 8])
    with pytest.raises(ValueError, match="residue 5 \\(unknown\\)"):
        check_unrepresented_residues(topology, _peptides([(2, 8)]))
