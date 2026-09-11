"""Population filtering and independent optimizer checks on synthetic inputs."""

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxent.examples.ATLAS_BV.analysis.cluster_filtering_omc import (
    filtered_indices,
    candidate_cases,
    fit_grid,
    trajectory_objective,
    partitions,
)
from jaxent.src.opt.loss.original_omc_laplacian import omc_graph_energy
from jaxent.src.opt.loss.graph_laplacian import prior_relative_graph_energy


def test_filtering_is_nested_and_preserves_all_clusters():
    labels = np.repeat(np.arange(3), [40, 30, 30])
    keep = [filtered_indices(labels, (0,), f, 123) for f in (0.5, 0.25, 0.125)]
    assert set(keep[2]) < set(keep[1]) < set(keep[0])
    for indices in keep:
        assert len(indices) == len(set(indices))
        assert set(labels[indices]) == {0, 1, 2}
        assert set(np.flatnonzero(labels != 0)).issubset(indices)
    np.testing.assert_array_equal(np.bincount(labels[keep[-1]]), [5, 30, 30])


def test_silhouette_selects_one_eligible_partition_per_band():
    coordinate = np.concatenate([10 * c + np.linspace(0, 0.1, 30) for c in range(4)])
    distance = np.abs(coordinate[:, None] - coordinate[None, :])
    audit, selected = partitions(distance, 123)
    assert 4 in selected
    assert all(audit[audit.selected].eligible)
    assert not set(selected).intersection({6, 7, 8})
    for low, high in ((2, 3), (4, 5), (6, 8)):
        eligible = audit[(audit.k >= low) & (audit.k <= high) & audit.eligible]
        if len(eligible):
            best = eligible.sort_values(
                ["silhouette", "k"], ascending=[False, True]
            ).iloc[0]
            assert int(best.k) in selected


def test_target_remains_fixed_and_candidates_are_population_distorted():
    labels = np.repeat(np.arange(3), [40, 30, 30])
    flat = np.vstack([np.arange(100), np.arange(100) ** 2]) / 10000
    target = flat.mean(axis=1).copy()
    for _, _, _, indices in candidate_cases(labels, 11):
        candidate_values = flat[:, indices]
        assert candidate_values.shape[1] == len(indices)
        np.testing.assert_array_equal(target, flat.mean(axis=1))
        np.testing.assert_array_equal(candidate_values, flat[:, indices])
    indices = filtered_indices(labels, (0,), 0.125, 11)
    assert not np.allclose(flat[:, indices].mean(axis=1), target)


def test_joint_filtering_is_present_and_two_cluster_case_controls_sample_size():
    labels = np.repeat(np.arange(2), [40, 60])
    cases = list(candidate_cases(labels, 123))
    assert len(cases) == 10
    joint = [case for case in cases if len(case[1]) == 2 and case[2] == 0.5]
    assert len(joint) == 1
    keep = joint[0][3]
    assert len(keep) == 50
    np.testing.assert_allclose(np.bincount(labels[keep]) / len(keep), [0.4, 0.6])
    assert len(list(candidate_cases(np.repeat(np.arange(4), 30), 123))) == 16


def test_masked_contraction_matches_original_energy_and_gradient():
    x = jnp.array([0.0, 0.1, 1.0, 2.0, 3.0])
    kernel = jnp.exp(-((x[:, None] - x[None, :]) ** 2))
    mask = jnp.array([True, False, True, True, False])
    kernel = kernel * mask[:, None] * mask[None, :]
    logits = jnp.array([0.1, 0.0, -0.2, 0.3, 0.0])
    keep = np.flatnonzero(mask)
    values = jnp.zeros((2, 5))
    target = jnp.zeros(2)
    for kind in (0, 1):

        def explicit(z):
            return omc_graph_energy(
                jax.nn.softmax(z[keep]),
                kernel[np.ix_(keep, keep)],
                normalise=bool(kind),
            )

        def contracted(z):
            return trajectory_objective(z, 1.0, kernel, kind, values, target, mask)

        np.testing.assert_allclose(contracted(logits), explicit(logits), rtol=1e-5)
        np.testing.assert_allclose(
            jax.grad(contracted)(logits), jax.grad(explicit)(logits), atol=1e-6
        )


def test_batch_matches_independent_adam_fit():
    values = jnp.array([[0.1, 0.5, 0.9, 0.2, 0.3], [0.4, 0.2, 0.1, 0.8, 0.7]])
    target = jnp.array([0.6, 0.3])
    mask = jnp.ones(5, bool)
    kernel = jnp.exp(-jnp.abs(jnp.arange(5)[:, None] - jnp.arange(5)[None, :]))
    strengths = (0.01, 0.1)
    batch = fit_grid(
        values, target, mask, kernel[None], np.array([0]), strengths, 20, 20
    )
    for i, strength in enumerate(strengths):
        logits = jnp.zeros(5)
        optimizer = optax.adam(0.05)
        state = optimizer.init(logits)

        def loss(z):
            w = jax.nn.softmax(z)
            return jnp.mean((values @ w - target) ** 2) / (
                jnp.var(target) + 1e-8
            ) + strength * omc_graph_energy(w, kernel)

        for _ in range(20):
            gradient = jax.grad(loss)(logits)
            updates, state = optimizer.update(gradient, state, logits)
            logits = optax.apply_updates(logits, updates)
        np.testing.assert_allclose(
            batch["weights"][0, i], jax.nn.softmax(logits), rtol=1e-5, atol=1e-6
        )


def test_masked_frames_have_exactly_zero_fitted_weight():
    mask = np.array([True, True, False, True, False])
    fit = fit_grid(
        np.eye(5),
        np.ones(5) / 5,
        mask,
        np.ones((1, 5, 5)) * mask[None, :, None] * mask[None, None, :],
        np.array([2]),
        (0.1,),
        20,
        20,
    )
    weights = np.asarray(fit["weights"])[0, 0]
    np.testing.assert_array_equal(weights[~mask], 0)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-6)


def test_comparator_objectives_match_incumbent_definitions():
    mask = jnp.array([True, False, True, True, False])
    keep = np.flatnonzero(mask)
    logits = jnp.array([0.1, 10.0, -0.4, 0.9, -10.0])
    kernel = jnp.exp(-jnp.abs(jnp.arange(5)[:, None] - jnp.arange(5)[None, :]))
    kernel = kernel * mask[:, None] * mask[None, :]
    kernel = kernel.at[jnp.diag_indices(5)].set(0)
    flat, target = jnp.zeros((3, 5)), jnp.zeros(3)
    prior = jnp.ones(3) / 3
    expected_entropy = jnp.sum(
        prior * (jnp.log(prior) - jax.nn.log_softmax(logits[keep]))
    )
    actual_entropy = trajectory_objective(logits, 1.0, kernel, 2, flat, target, mask)
    np.testing.assert_allclose(actual_entropy, expected_entropy, rtol=1e-5)
    i, j = np.triu_indices(3, 1)
    small = kernel[np.ix_(keep, keep)]
    expected_graph = prior_relative_graph_energy(
        logits[keep], jnp.log(prior), jnp.asarray(i), jnp.asarray(j), small[i, j]
    )
    actual_graph = trajectory_objective(logits, 1.0, kernel, 3, flat, target, mask)
    np.testing.assert_allclose(actual_graph, expected_graph, rtol=1e-5)


def test_data_loss_uses_every_observable_and_external_target():
    logits = jnp.array([0.1, 0.2, -0.3])
    flat = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.4, 0.5, 0.6]]
    )
    target = jnp.array([0.6, 0.2, 0.8, 0.4])
    expected = jnp.mean((flat @ jax.nn.softmax(logits) - target) ** 2) / (
        jnp.var(target) + 1e-8
    )
    actual = trajectory_objective(
        logits, 0.0, jnp.ones((3, 3)), 0, flat, target, jnp.ones(3, bool)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_zero_kernel_maxent_has_finite_gradients_in_mixed_batch():
    values = np.eye(5)
    target = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    mask = np.array([True, True, True, True, False])
    kernels = np.stack([np.zeros((5, 5)), np.ones((5, 5))])
    fit = fit_grid(
        values, target, mask, kernels, np.array([2, 0]), (0.0, 0.01, 100.0), 20, 20
    )
    assert np.isfinite(np.asarray(fit["weights"])).all()
    assert np.isfinite(np.asarray(fit["grad_norm"])).all()


def test_uniform_omc_energy_and_logit_gradient_are_exactly_zero():
    n = 512
    x = jnp.linspace(0, 3, n)
    kernel = jnp.exp(-((x[:, None] - x[None, :]) ** 2))
    mask = jnp.ones(n, bool)
    for kind in (0, 1):

        def fn(z):
            return trajectory_objective(
                z, 100.0, kernel, kind, jnp.zeros((1, n)), jnp.zeros(1), mask
            )

        value, gradient = jax.value_and_grad(fn)(jnp.zeros(n))
        np.testing.assert_array_equal(value, 0.0)
        np.testing.assert_array_equal(gradient, np.zeros(n))
