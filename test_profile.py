import math
import jax.numpy as jnp
import jax
# Quick test to confirm math.isnan works on .item()
val = jnp.array(float('nan'))
loss_val = val.item()
print(math.isnan(loss_val))
