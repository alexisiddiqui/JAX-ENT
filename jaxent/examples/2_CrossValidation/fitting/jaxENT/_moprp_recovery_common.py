#!/usr/bin/env python3
"""Shared data loading for the known-population MoPrP covariance-recovery experiment.

Every runner (coefficient lock, population-space oracle, frame reweighting) loads the same
real inputs through here so conventions stay identical:

* corrected physics-v2 hard-count BV features (97 residues x 500 frames, incl. residue 101);
* canonical exPfact-3Ala intrinsic rates aligned to the feature residues;
* the trim-one exPfact peptide map (all 14 peptides), row-normalized over active amides;
* the real 14 x 15 experimental uptake curve and its exact source timepoints;
* per-frame conformational state labels from the current ``_test`` PUF clustering, and the
  reference weights ``w_NMR`` that place the known NMR population uniformly within each state.

Nothing here fits anything; it only assembles inputs and validates their alignment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jaxent.src.analysis.hdx_ex2 import load_expfact_dataset, load_intrinsic_rate_file
from jaxent.src.analysis.state_population import (
    DEFAULT_STATE_MAPPING,
    FULL_STATE_SUPPORT,
    load_frame_states,
    load_state_targets,
    reference_weights_from_states,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[4]
BASE = PACKAGE_ROOT / "examples/2_CrossValidation"
MOPRP = BASE / "data/_MoPrP"
FEATURES_V2 = BASE / "fitting/jaxENT/_featurise_physics_v2"
STATE_RATIOS_JSON = BASE / "analysis/state_ratios.json"
CLUSTER_CSV = (
    BASE
    / "analysis/_MoPrP_analysis_clusters_feature_spec_AF2_test/clusters"
    / "global_frame_to_cluster_ensemble.csv"
)
CANONICAL_RATE_FILE = MOPRP / "expfact_kint_pH4p4_298K_min.dat"

# feature stem -> clustering ensemble_name
ENSEMBLES: dict[str, str] = {
    "AF2_MSAss": "AF2-MSAss",
    "AF2_filtered": "AF2-Filtered",
}

# Published Best-Vendruscolo coefficients (frozen reference setting).
PUBLISHED_BC = 0.35
PUBLISHED_BH = 2.0

PEPTIDE1_INDEX = 0  # peptide 1 is completely held out


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BlindedEnsembleInputs:
    """MoPrP HDX and trajectory inputs with all NMR population data excluded."""

    ensemble: str  # feature stem, e.g. "AF2_MSAss"
    ensemble_label: str  # clustering name, e.g. "AF2-MSAss"
    feature_residue_ids: np.ndarray  # (R,)
    heavy_contacts: np.ndarray  # (R, F)
    acceptor_contacts: np.ndarray  # (R, F)
    k_ints: np.ndarray  # (R,) canonical exPfact rates
    mapping: np.ndarray  # (P, R) row-normalized peptide map
    peptide_ids: np.ndarray  # (P,)
    timepoints: np.ndarray  # (T,)
    observed_uptake: np.ndarray  # (P, T) experimental dfrac
    n_frames: int

    def log_pf_by_frame(self, bc: float, bh: float) -> np.ndarray:
        """Per-frame residue log protection factors ``bc*heavy + bh*acceptor`` -> (R, F)."""

        return bc * self.heavy_contacts + bh * self.acceptor_contacts


@dataclass(frozen=True)
class EnsembleInputs(BlindedEnsembleInputs):
    """Legacy recovery inputs, including explicitly revealed NMR pseudo-truth."""

    states: np.ndarray  # (F,) state name per frame
    reference_weights: np.ndarray  # (F,) w_NMR
    support: tuple[str, ...]
    targets: np.ndarray  # (S,) full-support target populations


def _feature_bundle(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topology = json.loads((FEATURES_V2 / f"topology_{stem}_hard.json").read_text())["topologies"]
    residue_ids = np.asarray([item["residues"][0] for item in topology], dtype=int)
    with np.load(FEATURES_V2 / f"features_{stem}_hard.npz") as data:
        heavy = np.asarray(data["heavy_contacts"], dtype=np.float64)
        acceptor = np.asarray(data["acceptor_contacts"], dtype=np.float64)
    if heavy.shape[0] != residue_ids.size or acceptor.shape != heavy.shape:
        raise ValueError(f"{stem}: features not aligned to residue ids")
    return residue_ids, heavy, acceptor


def _complete_peptide_rows(peptide_map, feature_residue_ids: np.ndarray) -> list[int]:
    """Return the *row indices* of peptides fully represented by the feature residues."""

    represented = set(feature_residue_ids.tolist())
    complete = []
    for row, _ in enumerate(peptide_map.peptide_ids):
        active = set(
            np.asarray(peptide_map.residue_ids)[np.asarray(peptide_map.matrix)[row] > 0].tolist()
        )
        if not (active - represented):
            complete.append(row)
    return complete


def load_blinded_ensemble_inputs(ensemble: str) -> BlindedEnsembleInputs:
    """Assemble HDX/trajectory inputs without reading NMR states or populations."""

    if ensemble not in ENSEMBLES:
        raise ValueError(f"unknown ensemble {ensemble!r}; expected one of {list(ENSEMBLES)}")
    ensemble_label = ENSEMBLES[ensemble]

    dataset = load_expfact_dataset(MOPRP)
    canonical_rates = load_intrinsic_rate_file(
        CANONICAL_RATE_FILE,
        provider="exPfact-3Ala-numeric-reference",
        temperature_k=298.0,
        ph=4.4,
    )

    feature_ids, heavy, acceptor = _feature_bundle(ensemble)
    n_frames = heavy.shape[1]

    complete = _complete_peptide_rows(dataset.peptide_map, feature_ids)
    if len(complete) != len(dataset.peptide_map.peptide_ids):
        missing = sorted(set(int(p) for p in dataset.peptide_map.peptide_ids) - set(complete))
        raise ValueError(f"{ensemble}: peptides {missing} are not fully represented in features")
    aligned_map = dataset.peptide_map.subset_peptides(complete).aligned_to(feature_ids)
    mapping = np.asarray(aligned_map.matrix, dtype=np.float64)  # (P, R), row-normalized

    k_ints = np.asarray(canonical_rates.aligned(feature_ids), dtype=np.float64)
    if not (np.isfinite(k_ints).all() and (k_ints > 0).all()):
        raise ValueError(f"{ensemble}: canonical rates are not all finite and positive")

    return BlindedEnsembleInputs(
        ensemble=ensemble,
        ensemble_label=ensemble_label,
        feature_residue_ids=feature_ids,
        heavy_contacts=heavy,
        acceptor_contacts=acceptor,
        k_ints=k_ints,
        mapping=mapping,
        peptide_ids=np.asarray(aligned_map.peptide_ids, dtype=int),
        timepoints=np.asarray(dataset.protocol.timepoints_min, dtype=np.float64),
        observed_uptake=np.asarray(dataset.observed_uptake, dtype=np.float64),
        n_frames=n_frames,
    )


def reveal_nmr_reference(
    ensemble: str, *, expected_frames: int
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
    """Explicitly reveal states, support, targets, and ``w_NMR`` after inference."""

    if ensemble not in ENSEMBLES:
        raise ValueError(f"unknown ensemble {ensemble!r}; expected one of {list(ENSEMBLES)}")
    states = load_frame_states(
        CLUSTER_CSV, ENSEMBLES[ensemble], DEFAULT_STATE_MAPPING, expected_frames=expected_frames
    )
    support, targets = load_state_targets(STATE_RATIOS_JSON, FULL_STATE_SUPPORT)
    reference_weights = reference_weights_from_states(states, support, targets)
    return states, tuple(support), np.asarray(targets, dtype=np.float64), reference_weights


def load_ensemble_inputs(ensemble: str) -> EnsembleInputs:
    """Assemble legacy recovery inputs, including NMR pseudo-truth."""

    blinded = load_blinded_ensemble_inputs(ensemble)
    states, support, targets, reference_weights = reveal_nmr_reference(
        ensemble, expected_frames=blinded.n_frames
    )
    return EnsembleInputs(
        ensemble=blinded.ensemble,
        ensemble_label=blinded.ensemble_label,
        feature_residue_ids=blinded.feature_residue_ids,
        heavy_contacts=blinded.heavy_contacts,
        acceptor_contacts=blinded.acceptor_contacts,
        k_ints=blinded.k_ints,
        mapping=blinded.mapping,
        peptide_ids=blinded.peptide_ids,
        timepoints=blinded.timepoints,
        observed_uptake=blinded.observed_uptake,
        n_frames=blinded.n_frames,
        states=states,
        reference_weights=reference_weights,
        support=support,
        targets=targets,
    )


def blinded_input_hashes() -> dict[str, str]:
    """SHA-256 inputs allowed before the NMR pseudo-truth is revealed."""

    hashes = {"canonical_rate_file": sha256(CANONICAL_RATE_FILE)}
    for stem in ENSEMBLES:
        hashes[f"features_{stem}_hard.npz"] = sha256(FEATURES_V2 / f"features_{stem}_hard.npz")
        hashes[f"topology_{stem}_hard.json"] = sha256(FEATURES_V2 / f"topology_{stem}_hard.json")
    return hashes


def input_hashes() -> dict[str, str]:
    """SHA-256 of every raw input, for the audit manifest."""

    hashes = {
        "state_ratios_json": sha256(STATE_RATIOS_JSON),
        "cluster_csv": sha256(CLUSTER_CSV),
        "canonical_rate_file": sha256(CANONICAL_RATE_FILE),
    }
    for stem in ENSEMBLES:
        hashes[f"features_{stem}_hard.npz"] = sha256(FEATURES_V2 / f"features_{stem}_hard.npz")
        hashes[f"topology_{stem}_hard.json"] = sha256(FEATURES_V2 / f"topology_{stem}_hard.json")
    return hashes
