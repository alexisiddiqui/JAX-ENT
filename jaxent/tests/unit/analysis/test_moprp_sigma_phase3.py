"""Regression tests for the frozen Phase-3 MoPrP covariance target."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from jax import config
config.update("jax_enable_x64", True)
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
FITTING_DIR = REPO_ROOT / "jaxent/examples/2_CrossValidation/fitting/jaxENT"
OUTPUT = FITTING_DIR / "_moprp_sigma_noise_model"
sys.path.insert(0, str(FITTING_DIR))
phase3 = importlib.import_module("moprp_sigma_phase3")
sys.path.remove(str(FITTING_DIR))


def test_overlap_regions_partition_residues_and_peptides():
    mapping = np.array([
        [1, 0, 0, 0],
        [0, .5, .5, 0],
        [0, 0, .5, .5],
    ])
    regions = phase3.overlap_regions(mapping)
    assert len(regions) == 2
    residue_sets = [set(residues.tolist()) for residues, _ in regions]
    peptide_sets = [set(peptides.tolist()) for _, peptides in regions]
    assert {frozenset(x) for x in residue_sets} == {frozenset({0}), frozenset({1, 2, 3})}
    assert {frozenset(x) for x in peptide_sets} == {frozenset({0}), frozenset({1, 2})}


def test_frozen_target_cholesky_and_compatibility_exports():
    with np.load(OUTPUT / "target_modes.npz") as target:
        covariance = target["covariance"]
        chol = target["cholesky"]
        mean = target["mean"]
        assert mean.shape == (14, 15)
        assert covariance.shape == (210, 210)
        np.testing.assert_allclose(chol @ chol.T, covariance, rtol=2e-13, atol=2e-13)
        assert target["vector_order"].item() == "time-major; index = j * P + p"
        assert target["uptake_backend"].item() == "ex2"
    with np.load(OUTPUT / "compatibility_covariances.npz") as compatibility:
        blocks = compatibility["time_blocks"]
        collapsed = compatibility["collapsed_covariance"]
        precision = compatibility["trace_normalized_collapsed_precision"]
        assert blocks.shape == (15, 14, 14)
        np.testing.assert_allclose(collapsed, blocks.mean(axis=0))
        np.testing.assert_allclose(np.trace(precision), 14.0, rtol=1e-13)


def test_manifest_locks_phase3_decisions_and_provenance():
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    assert manifest["accepted_model"] == "peptide_only"
    assert manifest["accepted_parameters"]["tau_z"] == 0.0
    assert manifest["accepted_parameters"]["tau_time"] == 0.0
    assert manifest["accepted_parameters"]["kappa"] == 0.0
    assert manifest["component_decisions"]["anm"] == "rejected_phase2"
    assert manifest["pf_refit_inside_every_outer_fold"] is True
    assert manifest["pf_start_count"] >= 50
    assert manifest["harmonic_strength"] == 0.0
    assert manifest["rate_provenance"]["rate_source"] == "moprp_shipped"
    assert manifest["uptake_normalisation"] == phase3.simulation.UPTAKE_NORMALISATION
    assert len(manifest["refit_per_peptide_bias"]) == 14


def test_spectral_pathology_removed_and_fold_scores_are_per_cell():
    diagnostics = json.loads((OUTPUT / "spectral_diagnostics.json").read_text())
    assert diagnostics["leading_variance_fraction"] < 0.1
    assert diagnostics["effective_rank"] > 100
    assert diagnostics["condition_number"] < 10
    table = np.genfromtxt(OUTPUT / "cross_fitted_hierarchy.csv", delimiter=",", names=True,
                          dtype=None, encoding="utf-8")
    np.testing.assert_allclose(table["heldout_nll_per_cell"],
                               table["heldout_nll"] / table["heldout_cells"])
