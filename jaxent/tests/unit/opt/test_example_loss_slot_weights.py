"""Regression tests for positional example-suite loss weights."""

from jaxent.examples.common.optimization import _build_loss_slot_weights


def test_two_slot_weights_follow_data_then_maxent_loss_order():
    assert _build_loss_slot_weights(0.01, 0, 7.0) == [1.0, 0.01]


def test_three_slot_weights_follow_data_maxent_bv_loss_order():
    assert _build_loss_slot_weights(0.01, 1, 7.0) == [1.0, 0.01, 7.0]
