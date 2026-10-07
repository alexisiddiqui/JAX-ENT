"""Compare split-before/after-inversion Sigma selectors on saved MSE fits.

Run: .venv/bin/python -m jaxent.examples.common.analysis.rescore_crossval_coordinate_sigma
The fixed covariance is from uniform archived AF2-Filtered aligned coordinates.
Every candidate is replayed using its original uptake model and fitted parameters.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import re

import h5py
import jax.numpy as jnp
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from jaxent.examples.common import loading
from jaxent.examples.common.analysis.rescore_crossval_weighted import (
    CAMPAIGNS,
    GROUP,
    audit_provenance,
)
from jaxent.examples.common.config import ExperimentConfig
from jaxent.examples.common.uptake_models import build_uptake_model
from jaxent.src.custom_types.key import m_key
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.utils.hdf import load_model_parameters_from_hdf5

ROOT = Path(__file__).resolve().parents[4]
SELECTORS = ("ordinary_mse", "inverse_then_split", "split_then_inverse")


def precision(covariance):
    return cho_solve(cho_factor(covariance, lower=True), np.eye(len(covariance)))


def split_precisions(covariance, indices):
    """Both selectors share the same already stabilized full covariance."""
    matrices = {
        "inverse_then_split": precision(covariance)[np.ix_(indices, indices)],
        "split_then_inverse": precision(covariance[np.ix_(indices, indices)]),
    }
    return {
        name: matrix * len(indices) / np.trace(matrix)
        for name, matrix in matrices.items()
    }


def sigma_mse(residual, matrix):
    return float(
        0.5 * np.einsum("pt,pq,qt->", residual, matrix, residual) / residual.size
    )


def make_loader(fitting, config, ensemble, split, split_idx):
    with contextlib.redirect_stdout(io.StringIO()):
        train, val, full, _ = loading.load_experimental_data(
            "", str(fitting / "_datasplits"), split, split_idx
        )
        features, topology = loading.load_features_and_topology(
            str(fitting / "_featurise"), ensemble, config.scoring.ensemble_feature_map
        )
        loader = ExpD_Dataloader(data=full)
        loader.create_datasets(
            features=features,
            feature_topology=topology,
            train_data=train,
            val_data=val,
            test_data=full,
        )
    return loader, features, topology


def coordinate_covariance(topology, nframes):
    data = ROOT / "jaxent/examples/2_CrossValidation/data"
    reference = data / "MoPrP_max_plddt_4334.pdb"
    trajectory = (
        data / "_archived_MoPrP101/_cluster_MoPrP_filtered/clusters/all_clusters.xtc"
    )
    ref = mda.Universe(str(reference))
    mobile = mda.Universe(str(reference), str(trajectory), in_memory=True)
    align.AlignTraj(mobile, ref, select="protein and name CA", in_memory=True).run()
    ca = mobile.select_atoms("protein and name CA")
    by_id = {(str(atom.segid), int(atom.resid)): atom.index for atom in ca}
    # These source files contain one protein chain. Require unambiguous residue IDs.
    if len({int(a.resid) for a in ca}) != len(ca):
        raise ValueError("Coordinate source has ambiguous residue IDs")
    by_resid = {resid: index for (_, resid), index in by_id.items()}
    residue_ids = [int(t.residues[0]) for t in topology]
    if any(len(t.residues) != 1 for t in topology):
        raise ValueError("Require residue-level features")
    atoms = [by_resid[r] for r in residue_ids]
    coordinates = np.stack(
        [mobile.atoms[atoms].positions.copy() for _ in mobile.trajectory]
    ).astype(float)
    if coordinates.shape != (nframes, len(topology), 3):
        raise ValueError(
            f"Coordinate/feature frame count mismatch: {coordinates.shape}, {nframes}"
        )
    delta = coordinates - coordinates.mean(axis=0)
    covariance = np.einsum("fic,fjc->ij", delta, delta) / (3 * len(delta))
    return covariance, {
        "reference": str(reference),
        "trajectory": str(trajectory),
        "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "frames": len(delta),
        "residue_ids": residue_ids,
        "coordinate_definition": "aligned_CA_displacement_dot_product_divided_by_3",
        "frame_weighting": "uniform",
        "shrinkage_alpha": 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/crossval_coordinate_sigma_selection",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(
        ROOT / "artifacts/crossval_weighted_selection/candidates.csv"
    )
    expdir = ROOT / "jaxent/examples/2_CrossValidation"
    config = ExperimentConfig.from_yaml(str(expdir / "config.yaml"))
    loader, features, topology = make_loader(
        expdir / "fitting/jaxENT", config, "AF2_filtered", "spatial", 0
    )
    residue_cov, source = coordinate_covariance(topology, features.features_shape[1])
    peptide_ids = [int(d.top.fragment_index) for d in loader.test.data]
    if peptide_ids != list(range(len(peptide_ids))):
        raise ValueError("Full peptide covariance must follow source fragment IDs")
    mapping = np.asarray(
        loader.test.residue_feature_ouput_mapping.todense(), dtype=float
    )
    raw = mapping @ residue_cov @ mapping.T
    raw = (raw + raw.T) / 2
    helper = importlib.import_module(
        "jaxent.examples.1_IsoValidation_OMass.fitting.jaxENT.compute_sigma_synthetic"
    )
    thresholds = helper.ridge_thresholds(raw, condition_limit=1e8)
    ridge = thresholds["ridge_stable"]
    covariance = raw + ridge * np.eye(len(raw))
    source.update(
        peptides=len(raw),
        ridge=ridge,
        condition_limit=1e8,
        condition_before=float(np.linalg.cond(raw)),
        condition_after=float(np.linalg.cond(covariance)),
        rank_before=int(np.linalg.matrix_rank(raw)),
        ridge_policy="same full-peptide ridge for both selectors, before splitting",
    )
    np.savez_compressed(
        output / "covariance.npz",
        residue_covariance=residue_cov,
        mapping=mapping,
        peptide_covariance_raw=raw,
        peptide_covariance=covariance,
    )
    times = loading.load_hdx_timepoints_minutes(expdir / "data/_MoPrP/moprp.times")
    rows, provenance, checks = [], [], []
    for experiment, campaigns in CAMPAIGNS.items():
        fitting = ROOT / "jaxent/examples" / experiment / "fitting/jaxENT"
        cfg = ExperimentConfig.from_yaml(str(fitting.parents[1] / "config.yaml"))
        for model, basename in campaigns.items():
            rawdir = fitting / basename
            provenance.append(audit_provenance(rawdir, model))
            panel = baseline[
                (baseline.experiment == experiment) & (baseline.model == model)
            ]
            uptake_model = build_uptake_model(
                "linear" if model == "linear" else "standard", times
            )
            forward = uptake_model.forward[m_key("HDX_peptide")]
            if model != "linear":
                forward.frame_averaging_mode = (
                    "frame_uptake" if model == "full_uptake" else "rate"
                )
            processed = fitting / ("_processed_" + basename)
            for (ensemble, split, split_idx), group in panel.groupby(
                ["ensemble", "split_type", "split_idx"]
            ):
                ds, feat, top = make_loader(
                    fitting, cfg, ensemble, split, int(split_idx)
                )
                if [int(t.residues[0]) for t in top] != source["residue_ids"]:
                    raise ValueError("Covariance/features residue order mismatch")
                np.testing.assert_array_equal(
                    ds.test.residue_feature_ouput_mapping.todense(), mapping
                )
                indices = [int(d.top.fragment_index) for d in ds.val.data]
                matrices = split_precisions(covariance, indices)
                valmap = np.asarray(ds.val.residue_feature_ouput_mapping.todense())
                target = np.asarray(ds.val.y_true)[..., 0]
                for run_id, candidates in group.groupby("run_id"):
                    raw_name = re.sub(
                        r"_bvreg([\d.]+)",
                        lambda m: f"_bvreg{float(m.group(1))}",
                        run_id,
                    )
                    run = processed / split / run_id
                    stored = np.load(run / "pred_uptake.npy", mmap_mode="r")
                    weights = np.load(run / "frame_weights.npy", mmap_mode="r")
                    with h5py.File(
                        rawdir / split / f"{raw_name}_results.hdf5", "r"
                    ) as h:
                        for row in candidates.to_dict("records"):
                            index = int(row["checkpoint_index"])
                            params = load_model_parameters_from_hdf5(
                                h,
                                f"optimization_history/convergence_states/{index}/params/model_parameters/0",
                            )
                            pred = np.asarray(
                                forward.average_frames(
                                    feat, params, jnp.asarray(weights[index])
                                ).uptake
                            )
                            error = float(np.max(np.abs(pred - stored[index])))
                            np.testing.assert_allclose(
                                pred, stored[index], rtol=1e-5, atol=1e-6
                            )
                            residual = valmap @ pred.T - target
                            ordinary = float(np.mean(residual**2))
                            np.testing.assert_allclose(
                                ordinary, row["val_unweighted"], rtol=1e-6, atol=1e-8
                            )
                            rows.append(
                                row
                                | {
                                    "ordinary_mse": ordinary,
                                    **{
                                        name: sigma_mse(residual, matrix)
                                        for name, matrix in matrices.items()
                                    },
                                }
                            )
                            checks.append(error)
                print(
                    f"Scored {experiment}/{model}/{ensemble}/{split}/split{split_idx}: {len(group)} checkpoints",
                    flush=True,
                )
    candidates = pd.DataFrame(rows)
    chosen = []
    for selector in SELECTORS:
        selected = candidates.sort_values(selector, kind="stable").drop_duplicates(
            GROUP
        )
        selected = selected.assign(
            selector=selector, selection_score=selected[selector]
        )
        chosen.append(selected)
    selected = pd.concat(chosen, ignore_index=True)
    summary = (
        selected.groupby(GROUP[:-1] + ["selector"])
        .agg(
            replicates=("split_idx", "size"),
            recovery_mean=("recovery_percent", "mean"),
            recovery_sd=("recovery_percent", "std"),
            val_mse=("ordinary_mse", "mean"),
            sigma_selection_score=("selection_score", "mean"),
        )
        .reset_index()
    )
    assert summary.replicates.eq(3).all()
    ref = summary[summary.selector == "ordinary_mse"][
        GROUP[:-1] + ["recovery_mean"]
    ].rename(columns={"recovery_mean": "baseline_recovery"})
    summary = summary.merge(ref, on=GROUP[:-1], validate="many_to_one")
    summary["gain_vs_mse_pp"] = summary.recovery_mean - summary.baseline_recovery
    comparison = summary.pivot(
        index=GROUP[:-1], columns="selector", values="recovery_mean"
    ).reset_index()
    comparison["split_before_minus_after_pp"] = (
        comparison.split_then_inverse - comparison.inverse_then_split
    )
    comparison.to_csv(output / "comparison.csv", index=False)
    reference_selected = pd.read_csv(
        ROOT / "artifacts/crossval_weighted_selection/selected.csv"
    )
    reference_selected = (
        reference_selected[reference_selected.weighting == "unweighted"]
        .set_index(GROUP)
        .sort_index()
    )
    ordinary_selected = (
        selected[selected.selector == "ordinary_mse"].set_index(GROUP).sort_index()
    )
    if not ordinary_selected.run_id.equals(reference_selected.run_id):
        raise ValueError("Ordinary MSE selection differs from the verified baseline")
    np.testing.assert_array_equal(
        ordinary_selected.checkpoint_index, reference_selected.checkpoint_index
    )
    selected.to_csv(output / "selected.csv", index=False)
    candidates.to_csv(output / "candidates.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    audit = dict(
        source=source,
        campaigns=provenance,
        candidates=len(candidates),
        selected=len(selected),
        forward_checks=len(checks),
        max_forward_prediction_delta=max(checks),
        candidate_policy="all original saved convergence checkpoints; pool hyperparameters within each split",
        no_refitting=True,
        ordinary_mse_selected_checkpoints_verified=len(ordinary_selected),
        loss_factor=0.5,
        precision_normalization="trace equals split peptide count",
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    html = "<!doctype html><meta charset='utf-8'><title>Coordinate Sigma selection sensitivity</title><style>body{font:15px system-ui;margin:30px}td,th{padding:6px}</style><h1>Coordinate Sigma selection sensitivity on original MSE fits</h1><p>Uniform AF2-Filtered coordinate covariance; mapped to peptides before inversion; zero shrinkage. Both conventions use the same full-peptide stability ridge. Predictions replay the original forward models and checkpoint parameters. Recovery is mean and sample SD over three replicates. Gains are percentage points versus ordinary MSE selection.</p><p><a href='summary.csv'>Summary CSV</a> · <a href='selected.csv'>Selected checkpoints</a> · <a href='audit.json'>Audit</a></p>"
    html += "<h2>Recovery comparison</h2><p><a href='comparison.csv'>Comparison CSV</a>; positive split-before-minus-after differences favor splitting covariance before inversion.</p>"
    html += comparison.to_html(index=False, float_format=lambda x: f"{x:.2f}")
    html += "<h2>Means, sample SDs and gains</h2>"
    html += summary.to_html(index=False, float_format=lambda x: f"{x:.5f}")
    (output / "report.html").write_text(html)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
