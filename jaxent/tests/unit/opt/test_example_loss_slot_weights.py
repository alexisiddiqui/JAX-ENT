"""Regression tests for positional example-suite loss weights."""

import pytest

from jaxent.examples.common.optimization import (
    _build_loss_slot_weights,
    maxent_loss_weight,
)


def test_two_slot_weights_follow_data_then_maxent_loss_order():
    assert _build_loss_slot_weights(0.01, 0, 7.0) == [1.0, 100.0]


def test_three_slot_weights_follow_data_maxent_bv_loss_order():
    assert _build_loss_slot_weights(0.01, 1, 7.0) == [1.0, 100.0, 7.0]


@pytest.mark.parametrize("scaling", [0.0, -1.0, float("inf"), float("nan")])
def test_maxent_loss_weight_rejects_non_positive_or_non_finite_scales(scaling):
    with pytest.raises(ValueError, match="finite and greater than zero"):
        maxent_loss_weight(scaling)
