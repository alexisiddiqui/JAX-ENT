"""Regression tests for the MoPrP covariance-recovery shared input loader.

Locks the physics conventions the experiment depends on: all 14 peptides represented,
residue 101 present, the exact source timepoints, and reference weights that reproduce the
known NMR state populations for both ensembles.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"


@pytest.fixture(scope="module")
def common():
    sys.path.insert(0, str(FITTING_DIR))
    try:
        module = importlib.import_module("_moprp_recovery_common")
    finally:
        sys.path.remove(str(FITTING_DIR))
    return module


EXPECTED_TIMEPOINTS = np.array(
    [0.0834, 0.3336, 0.6666, 1.0002, 4.9998, 10.0002, 19.9998, 30.0, 45.0, 60.0,
     160.0002, 240.0, 390.0, 750.0, 1440.0]
)


@pytest.mark.parametrize("ensemble", ["AF2_MSAss", "AF2_filtered"])
def test_ensemble_inputs_regression(common, ensemble):
    inputs = common.load_ensemble_inputs(ensemble)

    # All 14 peptides represented and residue 101 present.
    assert inputs.peptide_ids.tolist() == list(range(1, 15))
    assert inputs.mapping.shape == (14, inputs.feature_residue_ids.size)
    assert 101 in inputs.feature_residue_ids.tolist()
    assert inputs.n_frames == 500

    # Exact source timepoints (not the rounded production header).
    np.testing.assert_allclose(inputs.timepoints, EXPECTED_TIMEPOINTS, atol=1e-3)
    assert inputs.observed_uptake.shape == (14, 15)

    # Row-normalized peptide map (average over represented amides).
    np.testing.assert_allclose(inputs.mapping.sum(axis=1), 1.0, atol=1e-8)

    # Canonical rates all finite and positive.
    assert np.isfinite(inputs.k_ints).all()
    assert (inputs.k_ints > 0).all()

    # Reference weights reproduce the known NMR populations exactly, decoys at zero.
    from jaxent.src.analysis.state_population import state_populations

    populations = np.asarray(
        state_populations(inputs.reference_weights, inputs.states, inputs.support)
    )
    mapping = dict(zip(inputs.support, populations))
    assert mapping["Folded"] == pytest.approx(0.97119, abs=1e-4)
    assert mapping["PUF1"] == pytest.approx(0.023644, abs=1e-4)
    assert mapping["PUF2"] == pytest.approx(0.005171, abs=1e-4)
    for decoy in ("PUF3", "unfolded", "PUF2-like"):
        assert mapping[decoy] == pytest.approx(0.0, abs=1e-9)
    assert inputs.reference_weights.sum() == pytest.approx(1.0)


def test_log_pf_matches_bv_definition(common):
    inputs = common.load_ensemble_inputs("AF2_MSAss")
    log_pf = inputs.log_pf_by_frame(0.35, 2.0)
    expected = 0.35 * inputs.heavy_contacts + 2.0 * inputs.acceptor_contacts
    np.testing.assert_allclose(log_pf, expected)
    assert log_pf.shape == (inputs.feature_residue_ids.size, 500)


def test_rate_source_selector_and_provenance(common):
    canonical = common.load_blinded_ensemble_inputs("AF2_MSAss")
    shipped = common.load_blinded_ensemble_inputs("AF2_MSAss", "moprp_shipped")

    assert canonical.rate_source == "expfact_recomputed"
    assert canonical.rate_file == common.CANONICAL_RATE_FILE.resolve()
    assert shipped.rate_source == "moprp_shipped"
    assert shipped.rate_file == (common.MOPRP / "moprp.kint").resolve()
    assert canonical.rate_file_sha256.startswith("f069ca46ecd7fd6a")
    assert shipped.rate_file_sha256.startswith("7be0f0d23ecec367")
    assert not np.array_equal(canonical.k_ints, shipped.k_ints)

    provenance = common.rate_source_provenance("moprp_shipped")
    assert provenance["temperature_k"] is None
    assert provenance["ph"] is None

    pdla = common.load_blinded_ensemble_inputs("AF2_MSAss", "hdxrate_pdla_validated")
    pdla_provenance = common.rate_source_provenance("hdxrate_pdla_validated")
    assert pdla.rate_source == "hdxrate_pdla_validated"
    assert pdla.rate_file_sha256 == "43f9178630136a886bb9eb6e7a3ce922e398b96ff34fa15f23938271e280c83c"
    assert pdla_provenance["units"] == "min^-1"
    assert pdla_provenance["ph_correction"] is False
    assert pdla_provenance["ph"] == 4.4


def test_unknown_rate_source_is_rejected(common):
    with pytest.raises(ValueError, match="unknown rate source"):
        common.load_blinded_ensemble_inputs("AF2_MSAss", "not-a-source")
