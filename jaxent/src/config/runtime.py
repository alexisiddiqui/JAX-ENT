"""Runtime configuration system for JAX-ENT.

This module provides centralized runtime configuration management for the JAX-ENT library,
enabling easy switching between different runtime modes (Production, Development, Debug, Testing, Performance)
without modifying code.

## Quick Start

### Environment Variable Configuration
```bash
# Development mode (CPU, JIT enabled, lenient type checking)
export JAXENT_MODE=development
python my_script.py

# Debug mode (CPU, JIT disabled, lenient type checking)
export JAXENT_MODE=debug
python my_script.py

# Production mode (GPU, JIT enabled, strict type checking)
export JAXENT_MODE=production
python my_script.py
```

### Programmatic Configuration
```python
# Must be called BEFORE importing jaxent
from jaxent.src.config.runtime import configure_runtime
configure_runtime(mode="debug")

import jaxent
```

### Environment Variables
- `JAXENT_MODE`: Set preset mode (production/development/debug/testing/performance)
- `JAXENT_PLATFORM`: Override platform (cpu/gpu/tpu)
- `JAXENT_DISABLE_JIT`: Override JIT setting (true/false)
- `JAXENT_ENABLE_CHEX`: Override chex assertions (true/false)

## Runtime Modes

| Mode | Platform | JIT | Beartype | Chex | NaN Check | Use Case |
|------|----------|-----|----------|------|-----------|----------|
| PRODUCTION | GPU | Enabled | Strict (raises) | Enabled | False | Production runs |
| DEVELOPMENT | CPU | Enabled | Lenient (warns) | Enabled | False | Interactive development |
| DEBUG | CPU | Disabled | Lenient (warns) | Enabled | True | Step-through debugging |
| TESTING | CPU | Enabled | Strict (raises) | Enabled | False | Unit/integration tests |
| PERFORMANCE | GPU | Enabled | Strict (raises) | Disabled | False | Benchmarking |
"""

import os
import warnings
from typing import Optional, Union
from enum import Enum
from dataclasses import dataclass
from beartype import BeartypeConf

__all__ = [
    "RuntimeMode",
    "RuntimeConfig",
    "configure_runtime",
    "get_runtime_config",
    "finalize_runtime_config",
]


class RuntimeMode(Enum):
    """Preset runtime modes with specific configuration combinations."""

    PRODUCTION = "production"
    DEVELOPMENT = "development"
    DEBUG = "debug"
    TESTING = "testing"
    PERFORMANCE = "performance"


@dataclass(frozen=True)
class RuntimeConfig:
    """Immutable runtime configuration for JAX-ENT.

    This dataclass holds all runtime settings for the JAX-ENT library,
    including JAX platform, JIT settings, beartype configuration, and more.

    Attributes:
        mode: The runtime mode (production/development/debug/testing/performance)
        platform: JAX platform name (cpu/gpu/tpu)
        disable_jit: Whether to disable JAX JIT compilation
        xla_preallocate: Whether to preallocate XLA memory
        beartype_conf: Beartype configuration object for type checking
        enable_chex_asserts: Whether to enable chex assertions
        debug_nans: Whether to debug NaN values in JAX
        check_tracers: Whether to check JAX tracers

    Example:
        >>> config = RuntimeConfig.from_mode("debug")
        >>> print(config.platform)
        'cpu'
        >>> print(config.disable_jit)
        True
    """

    mode: RuntimeMode
    platform: str
    disable_jit: bool
    xla_preallocate: bool
    beartype_conf: BeartypeConf
    enable_chex_asserts: bool
    debug_nans: bool
    check_tracers: bool

    @classmethod
    def from_mode(
        cls,
        mode: Union[str, RuntimeMode],
        platform_override: Optional[str] = None,
        disable_jit_override: Optional[bool] = None,
        enable_chex_override: Optional[bool] = None,
    ) -> "RuntimeConfig":
        """Create a RuntimeConfig from a preset mode with optional overrides.

        Args:
            mode: The runtime mode name (str or RuntimeMode enum)
            platform_override: Optional platform override (cpu/gpu/tpu)
            disable_jit_override: Optional JIT setting override
            enable_chex_override: Optional chex assertion setting override

        Returns:
            RuntimeConfig: A new RuntimeConfig instance

        Raises:
            ValueError: If mode is not a valid RuntimeMode

        Example:
            >>> config = RuntimeConfig.from_mode("debug")
            >>> config = RuntimeConfig.from_mode("production", platform_override="cpu")
        """
        # Normalize mode to RuntimeMode enum
        if isinstance(mode, str):
            try:
                mode = RuntimeMode(mode.lower())
            except ValueError:
                valid_modes = ", ".join(m.value for m in RuntimeMode)
                raise ValueError(
                    f"Invalid runtime mode: {mode}. Must be one of: {valid_modes}"
                )

        # Get preset configuration for the mode
        presets = {
            RuntimeMode.PRODUCTION: {
                "platform": "gpu",
                "disable_jit": False,
                "xla_preallocate": True,
                "enable_chex_asserts": True,
                "debug_nans": False,
                "check_tracers": False,
                "strict_beartype": True,
            },
            RuntimeMode.DEVELOPMENT: {
                "platform": "cpu",
                "disable_jit": False,
                "xla_preallocate": False,
                "enable_chex_asserts": True,
                "debug_nans": False,
                "check_tracers": False,
                "strict_beartype": False,
            },
            RuntimeMode.DEBUG: {
                "platform": "cpu",
                "disable_jit": True,
                "xla_preallocate": False,
                "enable_chex_asserts": True,
                "debug_nans": True,
                "check_tracers": True,
                "strict_beartype": False,
            },
            RuntimeMode.TESTING: {
                "platform": "cpu",
                "disable_jit": False,
                "xla_preallocate": False,
                "enable_chex_asserts": True,
                "debug_nans": False,
                "check_tracers": False,
                "strict_beartype": True,
            },
            RuntimeMode.PERFORMANCE: {
                "platform": "gpu",
                "disable_jit": False,
                "xla_preallocate": True,
                "enable_chex_asserts": False,
                "debug_nans": False,
                "check_tracers": False,
                "strict_beartype": True,
            },
        }

        preset = presets[mode]

        # Apply overrides
        platform = platform_override if platform_override is not None else preset["platform"]
        disable_jit = disable_jit_override if disable_jit_override is not None else preset["disable_jit"]
        enable_chex_asserts = (
            enable_chex_override
            if enable_chex_override is not None
            else preset["enable_chex_asserts"]
        )

        # Create beartype config based on strict setting
        if preset["strict_beartype"]:
            beartype_conf = BeartypeConf()  # Default strict mode
        else:
            beartype_conf = BeartypeConf(violation_type=UserWarning)  # Lenient mode

        return cls(
            mode=mode,
            platform=platform,
            disable_jit=disable_jit,
            xla_preallocate=preset["xla_preallocate"],
            beartype_conf=beartype_conf,
            enable_chex_asserts=enable_chex_asserts,
            debug_nans=preset["debug_nans"],
            check_tracers=preset["check_tracers"],
        )


# Module-level state
_RUNTIME_CONFIG: Optional[RuntimeConfig] = None
_JAX_CONFIGURED: bool = False


def _initialize_from_environment() -> RuntimeConfig:
    """Initialize runtime config from environment variables.

    Reads JAXENT_MODE and other environment variables to set up configuration.
    Called automatically on module import.

    Environment Variables:
        JAXENT_MODE: Runtime mode (production/development/debug/testing/performance)
        JAXENT_PLATFORM: Platform override (cpu/gpu/tpu)
        JAXENT_DISABLE_JIT: JIT disable override (true/false)
        JAXENT_ENABLE_CHEX: Chex assertions override (true/false)

    Returns:
        RuntimeConfig: The initialized configuration
    """
    mode = os.environ.get("JAXENT_MODE", "development").lower()

    # Parse overrides
    platform_override = os.environ.get("JAXENT_PLATFORM")
    if platform_override:
        platform_override = platform_override.lower()

    disable_jit_str = os.environ.get("JAXENT_DISABLE_JIT")
    disable_jit_override = None
    if disable_jit_str:
        disable_jit_override = disable_jit_str.lower() in ("true", "1", "yes")

    enable_chex_str = os.environ.get("JAXENT_ENABLE_CHEX")
    enable_chex_override = None
    if enable_chex_str:
        enable_chex_override = enable_chex_str.lower() in ("true", "1", "yes")

    try:
        config = RuntimeConfig.from_mode(
            mode,
            platform_override=platform_override,
            disable_jit_override=disable_jit_override,
            enable_chex_override=enable_chex_override,
        )
    except ValueError as e:
        warnings.warn(f"Invalid JAXENT_MODE: {e}. Using development mode.", stacklevel=2)
        config = RuntimeConfig.from_mode("development")

    return config


def _apply_environment_config(config: RuntimeConfig) -> None:
    """Apply environment variables based on configuration.

    This function sets XLA environment variables BEFORE JAX is imported.
    It must be called before JAX is imported for XLA settings to take effect.

    Args:
        config: The RuntimeConfig to apply

    Note:
        This is called automatically by configure_runtime() and on module import.
    """
    if config.xla_preallocate:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    else:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Set platform
    os.environ["JAX_PLATFORMS"] = config.platform
    os.environ["JAX_PLATFORM_NAME"] = config.platform


def _apply_jax_config(config: RuntimeConfig) -> None:
    """Apply JAX configuration after JAX is imported.

    This function updates JAX's runtime configuration. It must be called
    AFTER JAX is imported (or at least after jax.config is available).

    Args:
        config: The RuntimeConfig to apply

    Note:
        This is called by finalize_runtime_config() after JAX import.
    """
    try:
        import jax

        # Set platform
        jax.config.update("jax_platform_name", config.platform)

        # Set JIT
        jax.config.update("jax_disable_jit", config.disable_jit)

        # Set NaN debugging
        jax.config.update("jax_debug_nans", config.debug_nans)

        # Set tracer checking
        if config.check_tracers:
            jax.config.update("jax_check_tracer_leaks", True)

    except ImportError:
        warnings.warn(
            "JAX not found. JAX configuration not applied. "
            "Make sure JAX is installed before using runtime config.",
            stacklevel=2,
        )


def _apply_chex_config(config: RuntimeConfig) -> None:
    """Apply chex configuration.

    Enables or disables chex assertions based on configuration.

    Args:
        config: The RuntimeConfig to apply

    Note:
        This is called by finalize_runtime_config().
    """
    try:
        import chex

        if config.enable_chex_asserts:
            chex.enable_asserts()
        else:
            chex.disable_asserts()

    except ImportError:
        warnings.warn(
            "Chex not found. Chex configuration not applied. "
            "Make sure chex is installed before using chex assertions.",
            stacklevel=2,
        )


def configure_runtime(
    mode: Optional[Union[str, RuntimeMode]] = None,
    config: Optional[RuntimeConfig] = None,
    **overrides,
) -> RuntimeConfig:
    """Configure the JAX-ENT runtime.

    This function allows you to configure the runtime mode and settings.
    It can be called with either a mode name, a custom RuntimeConfig, or keyword overrides.

    This function must be called BEFORE importing other jaxent modules
    for some settings (like platform) to take effect.

    Args:
        mode: The runtime mode name (str or RuntimeMode enum). Ignored if config is provided.
        config: A custom RuntimeConfig. If provided, mode is ignored.
        **overrides: Keyword argument overrides for RuntimeConfig.from_mode().
                    Only used if both mode and config are provided.

    Returns:
        RuntimeConfig: The active runtime configuration

    Examples:
        >>> # Configure by mode
        >>> configure_runtime(mode="debug")

        >>> # Configure with overrides
        >>> configure_runtime(mode="production", platform_override="cpu")

        >>> # Configure with custom config
        >>> config = RuntimeConfig.from_mode("testing")
        >>> configure_runtime(config=config)

    Raises:
        ValueError: If mode is not a valid RuntimeMode

    Note:
        Environment variables (e.g., JAXENT_MODE) are only read on module import.
        Calling configure_runtime() after import overrides those settings.
    """
    global _RUNTIME_CONFIG

    if config is not None:
        _RUNTIME_CONFIG = config
    elif mode is not None:
        _RUNTIME_CONFIG = RuntimeConfig.from_mode(mode, **overrides)
    else:
        # If nothing specified, use current config or environment defaults
        if _RUNTIME_CONFIG is None:
            _RUNTIME_CONFIG = _initialize_from_environment()

    # Apply environment config (safe to call multiple times)
    _apply_environment_config(_RUNTIME_CONFIG)

    return _RUNTIME_CONFIG


def get_runtime_config() -> RuntimeConfig:
    """Get the current runtime configuration.

    Returns the active RuntimeConfig, initializing from environment variables
    if not yet configured.

    Returns:
        RuntimeConfig: The current runtime configuration

    Example:
        >>> config = get_runtime_config()
        >>> print(config.mode.value)
        'development'
        >>> print(config.platform)
        'cpu'
    """
    global _RUNTIME_CONFIG

    if _RUNTIME_CONFIG is None:
        _RUNTIME_CONFIG = _initialize_from_environment()

    return _RUNTIME_CONFIG


def finalize_runtime_config() -> RuntimeConfig:
    """Finalize runtime configuration after JAX import.

    This function should be called from jaxent/__init__.py after JAX is imported.
    It applies JAX-specific configuration that requires JAX to be available.

    Returns:
        RuntimeConfig: The current runtime configuration

    Note:
        This is called automatically by jaxent/__init__.py. Users should not
        need to call this directly unless implementing custom initialization.
    """
    global _JAX_CONFIGURED, _RUNTIME_CONFIG

    if _JAX_CONFIGURED:
        return _RUNTIME_CONFIG

    config = get_runtime_config()

    # Apply JAX configuration (after JAX import)
    _apply_jax_config(config)

    # Apply chex configuration
    _apply_chex_config(config)

    _JAX_CONFIGURED = True

    return config


# Auto-initialize from environment on module import
_RUNTIME_CONFIG = _initialize_from_environment()
_apply_environment_config(_RUNTIME_CONFIG)
