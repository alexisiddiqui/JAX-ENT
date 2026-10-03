from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.opt.losses import stable_positive_convex_kl, maxent_convexKL_loss


@pytest.mark.parametrize("n", [874, 2225])
def test_identical_uniform_is_exact_zero_with_zero_gradient(n):
    p = jnp.ones(n, dtype=jnp.float32) / n
    f = jax.jit(lambda q: stable_positive_convex_kl(p, q))
    assert float(f(p)) == 0.
    np.testing.assert_array_equal(jax.grad(f)(p), np.zeros(n))
    np.testing.assert_allclose(jax.jvp(jax.grad(f), (p,), (p,))[1], np.ones(n), rtol=1e-6)


@pytest.mark.parametrize("amplitude", [1e-6, 1e-3, .009, .011, .5, 10.])
def test_value_and_gradient_against_high_precision(amplitude):
    rng = np.random.default_rng(31)
    p = np.asarray(rng.uniform(.1, 1., 2225), dtype=np.float32)
    p /= p.sum()
    q = np.asarray(p * np.exp(rng.uniform(-amplitude, amplitude, len(p))), dtype=np.float32)
    # Long-double reference on exactly the float32 inputs, not idealized inputs.
    pl, ql = p.astype(np.longdouble), q.astype(np.longdouble)
    r = (ql-pl)/pl
    expected = np.sum(pl * (r-np.log1p(r)))
    value, gradient = jax.jit(jax.value_and_grad(lambda x: stable_positive_convex_kl(jnp.asarray(p), x)))(jnp.asarray(q))
    assert float(value) >= 0
    np.testing.assert_allclose(value, float(expected), rtol=3e-4, atol=1e-18)
    np.testing.assert_allclose(gradient, np.asarray(1-pl/ql, dtype=float), rtol=5e-5, atol=2e-7)


def test_native_wrapper_preserves_kl_direction_and_normalization_gradient():
    prior = jnp.asarray([.2, .3, .5])
    q = jnp.asarray([.1, .6, .3])
    dataset = SimpleNamespace(frame_weight_simplex=prior)
    def actual(weights):
        model = SimpleNamespace(params=SimpleNamespace(frame_weight_simplex=weights))
        return maxent_convexKL_loss(model, dataset, 0)[0]
    def reference(weights):
        p = prior + 1e-10/3
        p /= p.sum()
        weights = weights + 1e-10/3
        weights /= weights.sum()
        return jnp.sum(p * (jnp.log(p)-jnp.log(weights)))
    np.testing.assert_allclose(actual(q), reference(q), rtol=1e-6)
    np.testing.assert_allclose(jax.grad(actual)(q), jax.grad(reference)(q), rtol=2e-6)
