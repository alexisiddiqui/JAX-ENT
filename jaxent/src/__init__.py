# CRITICAL: Import runtime config BEFORE anything else
# This sets environment variables before JAX import
from jaxent.src.config.runtime import get_runtime_config, finalize_runtime_config

# Get runtime configuration (auto-initialized from environment)
_runtime_config = get_runtime_config()

# Apply beartype configuration from runtime config
from beartype.claw import beartype_this_package

beartype_this_package(conf=_runtime_config.beartype_conf)

# Finalize JAX configuration (after JAX is imported by other modules)
finalize_runtime_config()

"""Jaxent: JAX-based neural networks and differentiable programming library."""
