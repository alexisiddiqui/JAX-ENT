"""
JAX-ENT: JAX-based Ensemble Modeling and Optimization Toolkit

A Python library for ensemble-based modeling and optimization,
particularly focused on structural biology and biophysics.
"""

__version__ = "0.4.1"

# Re-export commonly used items from src for convenience
from jaxent.src.config.runtime import get_runtime_config, finalize_runtime_config

__all__ = ["__version__", "get_runtime_config", "finalize_runtime_config"]
