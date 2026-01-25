from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# 1. Create a config that warns instead of raising errors
# conf = BeartypeConf(violation_type=UserWarning)
conf = BeartypeConf()

# 2. Apply it to the package
beartype_this_package(conf=conf)
"""Jaxent: JAX-based neural networks and differentiable programming library."""
