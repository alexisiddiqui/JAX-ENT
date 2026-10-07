"""Numerical and artifact tests for the joint covariance/variance prior."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxent.src.analysis.joint_covariance_geometry import (
    LinearGeometryModel,
    build_residue_pair_features,
    family_joint_geometry_loss,
    feature_out_of_distribution,
    fit_linear_geometry_model,
    fixed_joint_geometry_loss,
    load_linear_geometry_model,
    peptide_tangent_prior,
    relative_log_variance,
    reference_population_geometry,
    save_linear_geometry_model,
    scale_free_log_spd,
    symmetric_matrix_exp,
)


def _covariance() -> jax.Array:
    return jnp.asarray([[2.0, 0.4, 0.2], [0.4, 1.2, -0.1], [0.2, -0.1, 0.8]])


def test_geometry_and_relative_variance_ignore_global_scale():
    covariance = _covariance()
    projection = jnp.eye(3)
    np.testing.assert_allclose(
        scale_free_log_spd(covariance, projection),
        scale_free_log_spd(17.0 * covariance, projection),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        relative_log_variance(covariance),
        relative_log_variance(17.0 * covariance),
        atol=2e-6,
    )


def test_full_geometry_distance_is_retained_basis_rotation_invariant():
    candidate = _covariance()
    target = jnp.asarray([[1.7, 0.2, 0.1], [0.2, 1.4, 0.3], [0.1, 0.3, 1.0]])
    angle = 0.37
    rotation = jnp.asarray(
        [[jnp.cos(angle), -jnp.sin(angle), 0.0], [jnp.sin(angle), jnp.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    first = jnp.mean(jnp.square(scale_free_log_spd(candidate) - scale_free_log_spd(target)))
    second = jnp.mean(
        jnp.square(
            scale_free_log_spd(rotation.T @ candidate @ rotation)
            - scale_free_log_spd(rotation.T @ target @ rotation)
        )
    )
    assert float(first) == pytest.approx(float(second), rel=2e-5)


def test_joint_loss_is_zero_at_prior_and_has_finite_gradient():
    covariance = _covariance()
    geometry = scale_free_log_spd(covariance)
    relative = relative_log_variance(covariance)

    def loss(scale):
        total, _, _ = fixed_joint_geometry_loss(
            covariance.at[0, 0].set(covariance[0, 0] * scale),
            prior_geometry=geometry,
            prior_relative_variance=relative,
            projection=jnp.eye(3),
        )
        return total

    assert float(loss(1.0)) < 1e-12
    assert np.isfinite(float(jax.grad(loss)(jnp.asarray(1.1))))
    assert float(loss(1.4)) > 0.0


def test_family_can_move_along_mode_but_score_is_penalized():
    covariance = _covariance()
    center = scale_free_log_spd(covariance)
    relative = relative_log_variance(covariance)
    mode = jnp.asarray([[[0.2, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.1]]])
    marginal_mode = jnp.asarray([[0.2, -0.1, -0.1]])
    total, geometry, marginal, score = family_joint_geometry_loss(
        covariance,
        jnp.asarray([0.5]),
        prior_geometry=center,
        geometry_modes=mode,
        prior_relative_variance=relative,
        marginal_modes=marginal_mode,
        score_precision=jnp.eye(1),
        projection=jnp.eye(3),
    )
    assert float(total) == pytest.approx(float(geometry + marginal + score))
    assert float(score) > 0.0


def test_tangent_pushforward_shapes_and_spd_center():
    center = jnp.diag(jnp.asarray([0.2, -0.1, -0.1]))
    modes = jnp.asarray([[[0.1, 0.02, 0.0], [0.02, -0.05, 0.0], [0.0, 0.0, -0.05]]])
    mapping = jnp.asarray([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])
    geometry, pushed, marginal, marginal_modes = peptide_tangent_prior(
        center, modes, mapping, jnp.eye(2)
    )
    assert geometry.shape == (2, 2)
    assert pushed.shape == (1, 2, 2)
    assert marginal.shape == (2,)
    assert marginal_modes.shape == (1, 2)
    assert np.all(np.linalg.eigvalsh(np.asarray(symmetric_matrix_exp(center))) > 0)


def test_linear_model_is_cross_dimension_and_artifact_round_trip(tmp_path):
    # Three population targets with a one-dimensional scale-free mode.
    base = np.diag([0.2, -0.1, -0.1])
    mode = np.asarray([[0.1, 0.03, 0.0], [0.03, -0.05, 0.0], [0.0, 0.0, -0.05]])
    targets = np.stack([base - mode, base, base + mode])
    rows = np.triu_indices(3)[0].size
    features_a = np.column_stack((np.ones(rows), np.linspace(-1, 1, rows)))
    features_b = features_a + np.asarray([0.0, 0.05])
    model, scores = fit_linear_geometry_model(
        [features_a, features_b], ["constant", "position"], targets, rank=1, ridge=0.1
    )
    assert scores.shape == (3, 1)
    center, modes = model.predict(features_a, 3)
    assert center.shape == (3, 3) and modes.shape == (1, 3, 3)
    assert np.trace(center) == pytest.approx(0.0, abs=1e-7)

    save_linear_geometry_model(model, tmp_path, {"training_populations": [0.1, 0.5, 0.9]})
    loaded, manifest = load_linear_geometry_model(tmp_path)
    loaded_center, loaded_modes = loaded.predict(features_a, 3)
    np.testing.assert_allclose(loaded_center, center)
    np.testing.assert_allclose(loaded_modes, modes)
    assert manifest["provenance"] == "ISO"

    report = feature_out_of_distribution(features_a + 100.0, loaded)
    assert report["flagged"]


def test_artifact_rejects_wrong_provenance(tmp_path):
    model = LinearGeometryModel(
        feature_names=("x",),
        feature_mean=np.zeros(1),
        feature_scale=np.ones(1),
        center_intercept=0.0,
        center_coefficients=np.zeros(1),
        mode_intercepts=np.zeros(0),
        mode_coefficients=np.zeros((0, 1)),
        score_precision=np.zeros((0, 0)),
        ridge=1.0,
        rank=0,
    )
    save_linear_geometry_model(model, tmp_path, {})
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["provenance"] = "MoPrP"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ISO-provenance"):
        load_linear_geometry_model(tmp_path)


def test_pair_features_are_symmetric_finite_and_geometry_is_scale_free():
    heavy = np.asarray([[1.0, 2.0, 3.0], [2.0, 2.5, 4.0], [0.5, 1.0, 1.5]])
    acceptor = np.asarray([[0.2, 0.4, 0.7], [0.1, 0.3, 0.8], [0.4, 0.2, 0.1]])
    coordinates = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 1.0, 0.0]])
    anm = np.asarray([[1.0, 0.2, 0.1], [0.2, 1.5, 0.4], [0.1, 0.4, 1.2]])
    features, names = build_residue_pair_features(
        residue_ids=np.asarray([10, 11, 12]),
        k_ints=np.asarray([0.1, 0.2, 0.3]),
        heavy_contacts=heavy,
        acceptor_contacts=acceptor,
        coordinates=coordinates,
        residue_names=["ALA", "GLY", "SER"],
        anm_covariance=anm,
    )
    assert features.shape == (6, len(names))
    assert len(names) == len(set(names))
    assert np.isfinite(features).all()

    weights = np.asarray([0.2, 0.3, 0.5])
    first = reference_population_geometry(heavy, acceptor, weights)
    second = reference_population_geometry(2.0 * heavy, 2.0 * acceptor, weights)
    np.testing.assert_allclose(first, second, atol=2e-6)
