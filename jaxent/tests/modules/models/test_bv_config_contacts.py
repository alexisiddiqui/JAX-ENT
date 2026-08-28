import pytest

from jaxent.src.models.config import BV_model_Config


def test_bv_contact_mode_defaults_to_hard():
    config = BV_model_Config()

    assert config.contact_mode == "hard"
    assert config.switch is False


def test_legacy_switch_argument_remains_backward_compatible():
    config = BV_model_Config(switch=True)

    assert config.contact_mode == "legacy_switch"
    assert config.switch is True


def test_bradshaw_switch_is_explicit_and_not_legacy():
    config = BV_model_Config(
        contact_mode="bradshaw_switch",
        switch_scale_nc=8.0,
        switch_scale_nh=4.0,
    )

    assert config.contact_mode == "bradshaw_switch"
    assert config.switch is False
    assert config.switch_scale_nc == 8.0
    assert config.switch_scale_nh == 4.0


def test_explicit_contact_mode_cannot_be_combined_with_legacy_argument():
    with pytest.raises(ValueError, match="mutually exclusive"):
        BV_model_Config(contact_mode="bradshaw_switch", switch=False)
