import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.examples.common.loading import align_kints_to_feature_topology
from jaxent.src.interfaces.topology import TopologyFactory


def _topology(residue: int):
    return TopologyFactory.from_single(chain="A", residue=residue)


def test_intrinsic_rates_are_aligned_by_residue_not_input_position():
    rate_data = (jnp.asarray([3.0, -1.0, 2.0]), [_topology(3), _topology(1), _topology(2)])
    aligned = align_kints_to_feature_topology(rate_data, [_topology(2), _topology(3)])

    np.testing.assert_allclose(aligned, [2.0, 3.0])


def test_missing_or_duplicate_feature_rates_are_rejected():
    with pytest.raises(ValueError, match="missing feature residues"):
        align_kints_to_feature_topology(
            (jnp.asarray([2.0]), [_topology(2)]),
            [_topology(2), _topology(101)],
        )

    with pytest.raises(ValueError, match="duplicate intrinsic rate"):
        align_kints_to_feature_topology(
            (jnp.asarray([2.0, 3.0]), [_topology(2), _topology(2)]),
            [_topology(2)],
        )


def test_nonpositive_active_rate_is_rejected_but_extra_exclusion_is_allowed():
    with pytest.raises(ValueError, match="finite positive"):
        align_kints_to_feature_topology(
            (jnp.asarray([-1.0, 2.0]), [_topology(2), _topology(3)]),
            [_topology(2), _topology(3)],
        )

    aligned = align_kints_to_feature_topology(
        (jnp.asarray([-1.0, 2.0]), [_topology(1), _topology(2)]),
        [_topology(2)],
    )
    np.testing.assert_allclose(aligned, [2.0])
