"""Unit tests for runtime configuration system.

Tests the RuntimeMode enum, RuntimeConfig dataclass, and configuration functions.
"""

import os
import pytest
from jaxent.src.config.runtime import (
    RuntimeMode,
    RuntimeConfig,
    configure_runtime,
    get_runtime_config,
    finalize_runtime_config,
)


class TestRuntimeMode:
    """Test RuntimeMode enum."""

    def test_all_modes_defined(self):
        """Test that all required modes are defined."""
        modes = {mode.value for mode in RuntimeMode}
        expected = {"production", "development", "debug", "testing", "performance"}
        assert modes == expected

    def test_mode_string_values(self):
        """Test that mode string values are correct."""
        assert RuntimeMode.PRODUCTION.value == "production"
        assert RuntimeMode.DEVELOPMENT.value == "development"
        assert RuntimeMode.DEBUG.value == "debug"
        assert RuntimeMode.TESTING.value == "testing"
        assert RuntimeMode.PERFORMANCE.value == "performance"


class TestRuntimeConfig:
    """Test RuntimeConfig dataclass."""

    def test_production_mode_config(self):
        """Test production mode configuration."""
        config = RuntimeConfig.from_mode("production")
        assert config.mode == RuntimeMode.PRODUCTION
        assert config.platform == "gpu"
        assert config.disable_jit is False
        assert config.xla_preallocate is True
        assert config.enable_chex_asserts is True

    def test_development_mode_config(self):
        """Test development mode configuration."""
        config = RuntimeConfig.from_mode("development")
        assert config.mode == RuntimeMode.DEVELOPMENT
        assert config.platform == "cpu"
        assert config.disable_jit is False
        assert config.xla_preallocate is False
        assert config.enable_chex_asserts is True

    def test_debug_mode_config(self):
        """Test debug mode configuration."""
        config = RuntimeConfig.from_mode("debug")
        assert config.mode == RuntimeMode.DEBUG
        assert config.platform == "cpu"
        assert config.disable_jit is True
        assert config.debug_nans is True
        assert config.check_tracers is True

    def test_testing_mode_config(self):
        """Test testing mode configuration."""
        config = RuntimeConfig.from_mode("testing")
        assert config.mode == RuntimeMode.TESTING
        assert config.platform == "cpu"
        assert config.disable_jit is False

    def test_performance_mode_config(self):
        """Test performance mode configuration."""
        config = RuntimeConfig.from_mode("performance")
        assert config.mode == RuntimeMode.PERFORMANCE
        assert config.platform == "gpu"
        assert config.disable_jit is False
        assert config.enable_chex_asserts is False

    def test_platform_override(self):
        """Test platform override."""
        config = RuntimeConfig.from_mode("production", platform_override="cpu")
        assert config.platform == "cpu"

    def test_jit_override(self):
        """Test JIT disable override."""
        config = RuntimeConfig.from_mode("production", disable_jit_override=True)
        assert config.disable_jit is True

    def test_chex_override(self):
        """Test chex assertions override."""
        config = RuntimeConfig.from_mode("production", enable_chex_override=False)
        assert config.enable_chex_asserts is False

    def test_config_is_frozen(self):
        """Test that RuntimeConfig is frozen (immutable)."""
        config = RuntimeConfig.from_mode("development")
        with pytest.raises(AttributeError):
            config.platform = "gpu"

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid runtime mode"):
            RuntimeConfig.from_mode("invalid")

    def test_beartype_conf_strict(self):
        """Test that strict modes have strict beartype config."""
        for mode in ["production", "testing", "performance"]:
            config = RuntimeConfig.from_mode(mode)
            assert config.beartype_conf is not None
            # Strict mode has no violation_type set or has default behavior

    def test_beartype_conf_lenient(self):
        """Test that lenient modes have lenient beartype config."""
        for mode in ["development", "debug"]:
            config = RuntimeConfig.from_mode(mode)
            assert config.beartype_conf is not None
            # Lenient mode uses UserWarning for violations


class TestConfigureRuntime:
    """Test configure_runtime function."""

    def test_configure_by_mode_string(self):
        """Test configuring by mode string."""
        config = configure_runtime(mode="debug")
        assert config.mode == RuntimeMode.DEBUG

    def test_configure_by_mode_enum(self):
        """Test configuring by RuntimeMode enum."""
        config = configure_runtime(mode=RuntimeMode.TESTING)
        assert config.mode == RuntimeMode.TESTING

    def test_configure_by_custom_config(self):
        """Test configuring with custom RuntimeConfig."""
        custom_config = RuntimeConfig.from_mode("production")
        config = configure_runtime(config=custom_config)
        assert config.mode == RuntimeMode.PRODUCTION

    def test_configure_with_overrides(self):
        """Test configuring with keyword overrides."""
        config = configure_runtime(mode="production", platform_override="cpu")
        assert config.platform == "cpu"

    def test_configure_applies_environment(self):
        """Test that configure_runtime sets environment variables."""
        configure_runtime(mode="development")
        assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false"

    def test_configure_persists_state(self):
        """Test that configuration persists across calls."""
        configure_runtime(mode="debug")
        config = get_runtime_config()
        assert config.mode == RuntimeMode.DEBUG


class TestGetRuntimeConfig:
    """Test get_runtime_config function."""

    def test_returns_current_config(self):
        """Test that get_runtime_config returns current configuration."""
        configure_runtime(mode="testing")
        config = get_runtime_config()
        assert config.mode == RuntimeMode.TESTING

    def test_initializes_if_not_set(self):
        """Test that get_runtime_config initializes config if needed."""
        # This test assumes _RUNTIME_CONFIG might be None
        config = get_runtime_config()
        assert config is not None
        assert isinstance(config, RuntimeConfig)


class TestEnvironmentVariableConfiguration:
    """Test environment variable based configuration."""

    def test_jaxent_mode_env_var(self, monkeypatch):
        """Test JAXENT_MODE environment variable."""
        # Fresh import simulation by clearing the config
        monkeypatch.setenv("JAXENT_MODE", "debug")

        # Need to manually initialize to test env var parsing
        from jaxent.src.config.runtime import _initialize_from_environment

        config = _initialize_from_environment()
        assert config.mode == RuntimeMode.DEBUG

    def test_platform_override_env_var(self, monkeypatch):
        """Test JAXENT_PLATFORM environment variable."""
        monkeypatch.setenv("JAXENT_MODE", "production")
        monkeypatch.setenv("JAXENT_PLATFORM", "cpu")

        from jaxent.src.config.runtime import _initialize_from_environment

        config = _initialize_from_environment()
        assert config.platform == "cpu"

    def test_jit_disable_env_var(self, monkeypatch):
        """Test JAXENT_DISABLE_JIT environment variable."""
        monkeypatch.setenv("JAXENT_MODE", "production")
        monkeypatch.setenv("JAXENT_DISABLE_JIT", "true")

        from jaxent.src.config.runtime import _initialize_from_environment

        config = _initialize_from_environment()
        assert config.disable_jit is True

    def test_invalid_mode_env_var(self, monkeypatch):
        """Test that invalid JAXENT_MODE falls back to development."""
        monkeypatch.setenv("JAXENT_MODE", "invalid_mode")

        from jaxent.src.config.runtime import _initialize_from_environment

        with pytest.warns(UserWarning, match="Invalid JAXENT_MODE"):
            config = _initialize_from_environment()
        assert config.mode == RuntimeMode.DEVELOPMENT


class TestConfigurationPresets:
    """Test specific preset configurations."""

    def test_production_preset_details(self):
        """Test all details of production preset."""
        config = RuntimeConfig.from_mode("production")
        assert config.mode == RuntimeMode.PRODUCTION
        assert config.platform == "gpu"
        assert config.disable_jit is False
        assert config.xla_preallocate is True
        assert config.enable_chex_asserts is True
        assert config.debug_nans is False
        assert config.check_tracers is False

    def test_development_preset_details(self):
        """Test all details of development preset."""
        config = RuntimeConfig.from_mode("development")
        assert config.mode == RuntimeMode.DEVELOPMENT
        assert config.platform == "cpu"
        assert config.disable_jit is False
        assert config.xla_preallocate is False
        assert config.enable_chex_asserts is True
        assert config.debug_nans is False
        assert config.check_tracers is False

    def test_debug_preset_details(self):
        """Test all details of debug preset."""
        config = RuntimeConfig.from_mode("debug")
        assert config.mode == RuntimeMode.DEBUG
        assert config.platform == "cpu"
        assert config.disable_jit is True
        assert config.xla_preallocate is False
        assert config.enable_chex_asserts is True
        assert config.debug_nans is True
        assert config.check_tracers is True

    def test_testing_preset_details(self):
        """Test all details of testing preset."""
        config = RuntimeConfig.from_mode("testing")
        assert config.mode == RuntimeMode.TESTING
        assert config.platform == "cpu"
        assert config.disable_jit is False
        assert config.xla_preallocate is False
        assert config.enable_chex_asserts is True
        assert config.debug_nans is False
        assert config.check_tracers is False

    def test_performance_preset_details(self):
        """Test all details of performance preset."""
        config = RuntimeConfig.from_mode("performance")
        assert config.mode == RuntimeMode.PERFORMANCE
        assert config.platform == "gpu"
        assert config.disable_jit is False
        assert config.xla_preallocate is True
        assert config.enable_chex_asserts is False
        assert config.debug_nans is False
        assert config.check_tracers is False
