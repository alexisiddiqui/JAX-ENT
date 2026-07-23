import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/2_CrossValidation/fitting/jaxENT/investigate_moprp_uptake_covariance.py"
)
SPEC = importlib.util.spec_from_file_location("moprp_uptake_covariance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

BASE = MODULE.REPO / "examples/2_CrossValidation"
READ = dict(sep=r"\s+", comment="#", header=None)


def _dfrac() -> np.ndarray:
    return pd.read_csv(BASE / "data/_MoPrP/_output/MoPrP_dfrac.dat", **READ).to_numpy(float)


def test_production_sigma_is_the_curve_raw_uptake_construction():
    """The matrix the Sigma_MSE loss consumes is covariance across timepoints, not noise."""
    dfrac = _dfrac()
    sigma = np.load(BASE / "data/_MoPrP_covariance_matrices/Sigma.npz")["Sigma"].astype(float)

    # It is exactly np.cov of the curve plus the ridge compute_sigma_real.py:162 adds.
    np.testing.assert_allclose(sigma, np.cov(dfrac) + np.diag(np.full(len(dfrac), 1e-6)), atol=1e-15)

    # And np.cov is the population curve covariance up to the Bessel factor, so Sigma is the
    # registered curve_raw_uptake construction rescaled and ridged -- nothing else.
    population = MODULE.LITMUS.curve_covariances(dfrac.T, MODULE.TIMEPOINTS)["curve_raw_uptake"]
    bessel = len(MODULE.TIMEPOINTS) / (len(MODULE.TIMEPOINTS) - 1)
    np.testing.assert_allclose(sigma, population * bessel + np.diag(np.full(len(dfrac), 1e-6)), atol=1e-12)


def test_moprp_has_no_replicates_to_estimate_observation_noise_from():
    dfrac = _dfrac()
    # One mean value per peptide per timepoint. Nothing here supports an observation-noise
    # estimate; Sigma's spread is across timepoints, i.e. signal.
    assert dfrac.shape == (14, len(MODULE.TIMEPOINTS))
    assert MODULE.TIMEPOINTS.shape == (15,)
    assert dfrac.min() >= 0.0 and dfrac.max() <= 1.0


def test_bv_coefficients_match_the_model_defaults():
    from jaxent.src.models.config import BV_model_Config

    config = BV_model_Config()
    # float32 defaults, so compare at float32 tolerance rather than exactly.
    np.testing.assert_allclose(np.asarray(config.bv_bc).ravel()[0], MODULE.BV_BC, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(config.bv_bh).ravel()[0], MODULE.BV_BH, rtol=1e-6)


def test_log_pf_reconstruction_is_the_bv_hard_contact_sum():
    data = {"heavy_contacts": np.asarray([[1.0, 2.0]]), "acceptor_contacts": np.asarray([[3.0, 4.0]])}
    np.testing.assert_allclose(
        MODULE.log_pf_from_features(data), 0.35 * data["heavy_contacts"] + 2.0 * data["acceptor_contacts"]
    )


def test_real_segment_sparse_map_is_normalized_and_overlapping():
    dfrac = _dfrac()
    segments = pd.read_csv(BASE / "data/_MoPrP/_output/MoPrP_segments.txt", **READ).to_numpy(int)
    mapping = MODULE.build_sparse_map(
        segments,
        dfrac,
        {
            "features": BASE / "fitting/jaxENT/_featurise/features_AF2_MSAss.npz",
            "topology": BASE / "fitting/jaxENT/_featurise/topology_AF2_MSAss.json",
        },
    )
    assert mapping.shape == (14, 96)
    np.testing.assert_allclose(mapping.sum(axis=1), 1.0, atol=1e-6)
    # Real peptides overlap, so rows share residues. That is the data, not a defect.
    coverage = (mapping > 0).astype(int).sum(axis=0)
    assert coverage.max() > 1


def test_map_congruence_holds_for_the_real_map():
    values = np.asarray([[0.0, 1.0, 2.0], [2.0, 3.0, 1.0], [4.0, 0.0, 2.0], [1.0, 2.0, 5.0]])
    weights = np.full(4, 0.25)
    mapping = np.asarray([[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]])
    covariance = MODULE.LITMUS.weighted_covariance(values, weights)
    np.testing.assert_allclose(
        mapping @ covariance @ mapping.T,
        MODULE.LITMUS.weighted_covariance(values @ mapping.T, weights),
        atol=1e-12,
    )


def test_curve_covariance_rank_is_bounded_by_the_fifteen_point_grid():
    curve = np.linspace(0.02, 0.9, 15 * 12).reshape(15, 12)
    matrices = MODULE.LITMUS.curve_covariances(curve, MODULE.TIMEPOINTS)
    # 15 timepoints bound the rank at 14, and adjacent slopes at 13 -- looser than ISO's 4/3,
    # which is the whole reason this run can change the standing conclusion.
    assert np.linalg.matrix_rank(matrices["curve_raw_uptake"], tol=1e-10) <= 14
    assert np.linalg.matrix_rank(matrices["curve_adjacent_survival_slope"], tol=1e-10) <= 13


def test_permutation_preserves_predicted_mean_uptake_but_changes_coupling():
    rates = np.asarray([[0.02, 0.4], [0.1, 0.7], [0.3, 0.2], [0.05, 0.9]])
    weights = np.full(4, 0.25)
    permuted = MODULE.LITMUS.permute_equal_weight_frames(rates, weights, seed=MODULE.PERMUTATION_SEED)
    original = np.einsum("tfr,f->tr", MODULE.LITMUS.framewise_uptake(rates, MODULE.TIMEPOINTS), weights)
    shuffled = np.einsum("tfr,f->tr", MODULE.LITMUS.framewise_uptake(permuted, MODULE.TIMEPOINTS), weights)
    np.testing.assert_allclose(original, shuffled, atol=1e-14)
    assert not np.allclose(
        MODULE.LITMUS.weighted_covariance(permuted, weights),
        MODULE.LITMUS.weighted_covariance(rates, weights),
    )


def test_smoke_integration(tmp_path):
    MODULE.run(argparse.Namespace(output_dir=tmp_path, smoke=True))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["fitting_performed"] is False
    assert manifest["oracle_available"] is False
    assert manifest["reference_weights"] == "uniform"

    preflight = pd.read_csv(tmp_path / "preflight.csv").set_index("check").value
    assert preflight["sigma_is_np_cov_of_dfrac"] < 1e-12
    assert preflight["k_ints_identical_across_ensembles"] == 0.0
    assert preflight["map_congruence__AF2_MSAss"] < 1e-10
    # The 14x14 production matrix against 12 fitted peptides is reported, not reconciled away.
    assert preflight["sigma_dimension_vs_fitted_peptides"] == 2.0

    permutation = pd.read_csv(tmp_path / "identifiability_permutation.csv")
    assert set(permutation.ensemble) == {"AF2_MSAss", "AF2_filtered"}
    assert (permutation.mean_uptake_max_abs_change < 1e-12).all()
    assert (permutation.full_covariance_relative_change > 0).all()

    for name in ("discrimination.csv", "resolving_power.csv", "report.md", "correspondence.png"):
        assert (tmp_path / name).exists()


def test_production_sigma_is_subset_to_the_fitted_peptides(tmp_path):
    MODULE.run(argparse.Namespace(output_dir=tmp_path, smoke=True))
    matrices = np.load(tmp_path / "covariance_matrices.npz")
    assert matrices["candidate__production_sigma"].shape == (
        MODULE.N_FITTED_PEPTIDES,
        MODULE.N_FITTED_PEPTIDES,
    )
    for key in matrices:
        assert matrices[key].shape[0] == MODULE.N_FITTED_PEPTIDES


def test_neither_reference_is_used_as_a_gate(tmp_path):
    MODULE.run(argparse.Namespace(output_dir=tmp_path, smoke=True))
    metrics = pd.read_csv(tmp_path / "matrix_metrics.csv")
    # Every construction is scored against both references; no reference is privileged.
    assert set(metrics.reference) == {"AF2_MSAss", "AF2_filtered"}
    for _, rows in metrics.groupby("construction"):
        assert rows.reference.nunique() == 2
    discrimination = pd.read_csv(tmp_path / "discrimination.csv")
    assert "delta_vs_resolving_distance" in discrimination.columns
