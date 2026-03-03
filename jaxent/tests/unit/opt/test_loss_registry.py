"""Unit tests for the loss function registry and new covariance-weighted losses."""

import pytest

from jaxent.src.opt.losses import (
    LOSS_REGISTRY,
    get_loss_function,
    hdx_uptake_eye_MSE_loss,
    hdx_uptake_l2_loss,
    hdx_uptake_mean_centred_eye_MSE_loss,
    hdx_uptake_mean_centred_l2_loss,
    hdx_uptake_mean_centred_sigma_MSE_loss,
    hdx_uptake_sigma_MSE_loss,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestLossRegistry:
    """Tests for LOSS_REGISTRY and get_loss_function."""

    def test_registry_is_non_empty(self):
        assert len(LOSS_REGISTRY) > 0

    @pytest.mark.parametrize(
        "key,expected_fn",
        [
            ("L2", hdx_uptake_l2_loss),
            ("mcL2", hdx_uptake_mean_centred_l2_loss),
            ("MSE", hdx_uptake_eye_MSE_loss),
            ("mcMSE", hdx_uptake_mean_centred_eye_MSE_loss),
            ("Sigma_MSE", hdx_uptake_sigma_MSE_loss),
            ("mcSigma_MSE", hdx_uptake_mean_centred_sigma_MSE_loss),
        ],
    )
    def test_registry_contains_expected_entries(self, key, expected_fn):
        assert key in LOSS_REGISTRY
        assert LOSS_REGISTRY[key] is expected_fn

    def test_get_loss_function_returns_correct_fn(self):
        fn = get_loss_function("L2")
        assert fn is hdx_uptake_l2_loss

    def test_get_loss_function_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown loss function"):
            get_loss_function("nonexistent_loss_42")

    def test_get_loss_function_error_lists_available(self):
        with pytest.raises(ValueError, match="Available:"):
            get_loss_function("bad_name")

    @pytest.mark.parametrize("key", list(LOSS_REGISTRY.keys()))
    def test_all_registry_values_are_callable(self, key):
        assert callable(LOSS_REGISTRY[key])

    def test_registry_keys_are_strings(self):
        for key in LOSS_REGISTRY:
            assert isinstance(key, str)
