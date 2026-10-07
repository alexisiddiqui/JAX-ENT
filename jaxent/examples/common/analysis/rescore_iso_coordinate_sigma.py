"""Compare covariance split order for selection on the IsoValidation MSE panel.

Run: .venv/bin/python -m jaxent.examples.common.analysis.rescore_iso_coordinate_sigma
Preserves fits; uses native saved convergence checkpoints and each fitted model.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from jaxent.examples.common.analysis.rescore_crossval_coordinate_sigma import (
    sigma_mse,
    split_precisions,
)

ROOT = Path(__file__).resolve().parents[4]
HERE = ROOT / "jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT"
SELECTORS = ("ordinary_mse", "inverse_then_split", "split_then_inverse")
LABELS = ("Ordinary MSE", "Invert → split (existing)", "Split → invert")


def make_predict(forward, features):
    # Bind each context independently; later loop iterations must not change
    # the model/features captured by a cached predictor if JAX retraces it.
    return jax.jit(
        lambda weights, parameters: forward.average_frames(
            features, parameters, weights
        ).uptake
    )


def summarize(selected, keys):
    return (
        selected.groupby(keys + ["selector"])
        .agg(
            replicates=("split_idx", "size"),
            recovery_mean=("recovery_percent", "mean"),
            recovery_sd=("recovery_percent", "std"),
            val_mse=("ordinary_mse", "mean"),
            maxent_mean=("maxent", "mean"),
            ess_mean=("ess_percent", "mean"),
        )
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=HERE / "_maxent_weighted_mse_per_peptide_tri1e9_closed_selection_jobs8",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/iso_coordinate_sigma_selection",
    )
    parser.add_argument(
        "--training-losses",
        nargs="+",
        choices=("mse", "closed_coordinate"),
        default=("mse",),
    )
    args = parser.parse_args()
    if (
        list(args.training_losses) != ["mse"]
        and args.output_dir == ROOT / "artifacts/iso_coordinate_sigma_selection"
    ):
        args.output_dir = ROOT / "artifacts/iso_mse_closed_gt_selection"
    sys.path.insert(0, str(HERE))
    import analyze_within_trajectory_selection as original

    sweep, source, shrinkage = original.sweep, original.source, original.shrinkage
    specs = original.collect_specs(args.campaign)
    mse_specs = [s for s in specs if s.sigma_source in args.training_losses]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sigma_paths = {}
    for ensemble in sweep.ENSEMBLES:
        for name in ("closed_coordinate", "gt_coordinate"):
            spec = next(
                s
                for s in specs
                if s.ensemble == ensemble and s.sigma_source == name and s.alpha == 0
            )
            sigma_paths[ensemble, name] = Path(spec.sigma_path)
    rows, audits, missing, caches = [], [], [], {}
    reference_scores = pd.read_csv(args.campaign / "convergence_scores.csv")
    for spec in mse_specs:
        cfg = json.loads(spec.config_path.read_text())
        mode = "linear_uptake" if spec.uptake_mode == "linear" else "frame_uptake"
        if cfg["effective_settings"]["frame_averaging_mode"] != mode:
            raise ValueError(f"Forward mode mismatch: {spec.run_id}")
        if cfg["loss_config"]["optimize_bv_params"]:
            raise ValueError("Expected fixed-BV MSE panel")
        key = spec.ensemble, spec.uptake_mode, spec.split_idx
        if key not in caches:
            with contextlib.redirect_stdout(io.StringIO()):
                feat, top = shrinkage.load_features(spec.features_dir, spec.ensemble)
                assignments = shrinkage.load_cluster_assignments(
                    spec.clustering_dir, spec.ensemble
                )
                model = shrinkage.configure_model(spec.uptake_mode, assignments)
                train, val = shrinkage.load_split(
                    spec.datasplit_dir, spec.split_type, spec.split_idx
                )
                loader = source.create_data_loaders(train + val, train, val, feat, top)
            mapping = np.asarray(loader.val.residue_feature_ouput_mapping.todense())
            # Iso observations are single-residue measurements. Splits omit
            # some residues, so their union is not the full covariance support.
            full_mapping = np.eye(feat.features_shape[0])
            indices = [int(d.top.fragment_index) for d in val]
            np.testing.assert_array_equal(mapping, full_mapping[indices])
            train_indices = [int(d.top.fragment_index) for d in train]
            np.testing.assert_array_equal(
                loader.train.residue_feature_ouput_mapping.todense(),
                full_mapping[train_indices],
            )
            matrices, descriptions = {}, {}
            for name in ("closed_coordinate", "gt_coordinate"):
                path = sigma_paths[spec.ensemble, name]
                with np.load(path) as archive:
                    cov = full_mapping @ archive["Sigma"] @ full_mapping.T
                    if float(archive["alpha"]) != 0:
                        raise ValueError("Require alpha=0 coordinate covariance")
                    is_identity = full_mapping.shape[0] == full_mapping.shape[
                        1
                    ] and np.array_equal(full_mapping, np.eye(len(full_mapping)))
                    if not is_identity:
                        raise ValueError(
                            "Iso panel mapping differs from original residue-observation identity; cannot reproduce existing selector"
                        )
                    matrices[name] = split_precisions(cov, indices)
                    old = archive["Sigma_inv_normalized"][np.ix_(indices, indices)]
                    old = old * len(indices) / np.trace(old)
                    np.testing.assert_allclose(
                        matrices[name]["inverse_then_split"], old, rtol=1e-7, atol=1e-8
                    )
                    descriptions[name] = {
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "alpha": 0.0,
                        "condition": float(np.linalg.cond(cov)),
                        "ridge": float(
                            np.mean(np.diag(archive["Sigma"] - archive["Sigma_shrunk"]))
                        ),
                        "mapping_identity_verified": is_identity,
                    }
            forward = model.forward[shrinkage.m_key("HDX_peptide")]
            predict = make_predict(forward, feat)
            caches[key] = (
                feat,
                assignments,
                model,
                predict,
                mapping,
                np.asarray(loader.val.y_true)[..., 0],
                matrices,
                descriptions,
            )
        feat, assignments, model, predict, mapping, target, matrices, descriptions = (
            caches[key]
        )
        history = source.load_optimization_history_from_file(str(spec.history_path))
        labeled = source.iter_labeled_convergence_states(history)
        if not labeled:
            missing.append(spec.run_id)
            continue
        saved = reference_scores[reference_scores.run_id == spec.run_id]
        if len(saved) != len(labeled):
            raise ValueError(
                f"Original convergence score count mismatch: {spec.run_id}"
            )
        errors = []
        for candidate in labeled:
            state = candidate.state
            weights = np.asarray(
                source.validated_frame_weight_simplex(
                    state.params.frame_weight_simplex
                ),
                dtype=float,
            )
            weights /= weights.sum()
            parameters = state.params.model_parameters[0]
            for fitted, initial in zip(
                jax.tree_util.tree_leaves(parameters),
                jax.tree_util.tree_leaves(model.params),
                strict=True,
            ):
                np.testing.assert_array_equal(fitted, initial)
            pred = np.asarray(predict(jnp.asarray(weights), parameters), dtype=float)
            residual = mapping @ pred.T - target
            mse = float(np.mean(residual**2))
            native = float(state.losses.val_losses[0])
            expected_native = (
                0.5 * mse
                if spec.sigma_source == "mse"
                else sigma_mse(
                    residual, matrices["closed_coordinate"]["inverse_then_split"]
                )
            )
            np.testing.assert_allclose(expected_native, native, rtol=3e-3, atol=4e-7)
            original_row = saved[saved.convergence_rank == candidate.rank]
            if len(original_row) != 1:
                raise ValueError("Ambiguous saved convergence rank")
            np.testing.assert_allclose(
                mse, original_row.iloc[0].val_mse, rtol=3e-3, atol=2e-7
            )
            recovery = source.calculate_recovery_percentage(
                assignments, weights, shrinkage.GROUND_TRUTH, shrinkage.STATE_MAPPING
            )
            np.testing.assert_allclose(
                recovery, original_row.iloc[0].recovery_percent, rtol=1e-7, atol=1e-5
            )
            for name, precisions in matrices.items():
                scores = {
                    selector: sigma_mse(residual, matrix)
                    for selector, matrix in precisions.items()
                }
                if name == "closed_coordinate":
                    np.testing.assert_allclose(
                        scores["inverse_then_split"],
                        original_row.iloc[0].val_closed_sigma_mse,
                        rtol=3e-3,
                        atol=2e-7,
                    )
                rows.append(
                    dict(
                        run_id=spec.run_id,
                        ensemble=spec.ensemble,
                        model=spec.uptake_mode,
                        training_loss=spec.sigma_source,
                        split_idx=spec.split_idx,
                        maxent=spec.sweep_maxent,
                        covariance_source=name,
                        convergence_rank=candidate.rank,
                        step=int(state.step),
                        ordinary_mse=mse,
                        native_data_loss=native,
                        native_mse=2 * native if spec.sigma_source == "mse" else np.nan,
                        **scores,
                        recovery_percent=recovery,
                        ess_percent=100
                        * source.effective_sample_size(weights)
                        / len(weights),
                    )
                )
            errors.append(abs(expected_native - native))
        audits.append(
            dict(
                run_id=spec.run_id,
                history=str(spec.history_path),
                mode=mode,
                training_loss=spec.sigma_source,
                candidates=len(labeled),
                max_native_data_loss_delta=max(errors),
                covariance=descriptions,
            )
        )
        print(f"Scored {spec.run_id}: {len(labeled)} checkpoints", flush=True)
    candidates = pd.DataFrame(rows)
    chosen, pooled = [], []
    keys = [
        "training_loss",
        "ensemble",
        "model",
        "covariance_source",
        "maxent",
        "split_idx",
    ]
    for selector in SELECTORS:
        chosen.append(
            candidates.sort_values(
                [selector, "step", "convergence_rank"], kind="stable"
            )
            .drop_duplicates(keys)
            .assign(selector=selector)
        )
        pooled.append(
            candidates.sort_values(
                [selector, "maxent", "step", "convergence_rank"], kind="stable"
            )
            .drop_duplicates([k for k in keys if k != "maxent"])
            .assign(selector=selector)
        )
    selected = pd.concat(chosen, ignore_index=True)
    pooled = pd.concat(pooled, ignore_index=True)
    summary = summarize(selected, keys[:-1])
    pooled_summary = summarize(pooled, keys[:-2])
    if not pooled_summary.replicates.eq(3).all():
        raise ValueError("Incomplete pooled replicate coverage")
    for name, frame in [
        ("candidates", candidates),
        ("selected", selected),
        ("summary", summary),
        ("pooled_selected", pooled),
        ("pooled_summary", pooled_summary),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)
    audit = dict(
        campaign=str(args.campaign.resolve()),
        fit_count=len(mse_specs),
        training_losses=list(args.training_losses),
        scored_fits=len(audits),
        missing_convergence=missing,
        unique_checkpoints=sum(x["candidates"] for x in audits),
        candidate_policy="original native convergence checkpoints only",
        selection="within each MaxEnt; pooled selection exported separately",
        sources=["closed_coordinate", "gt_coordinate"],
        fits=audits,
        no_refitting=True,
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if list(args.training_losses) != ["mse"]:
        from jaxent.examples.common.analysis.compare_iso_selectors import (
            export_comparison,
        )

        export_comparison(output, candidates)
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in ("closed_coordinate", "gt_coordinate"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, (ensemble, model) in zip(
            axes.flat, [(e, m) for e in sweep.ENSEMBLES for m in sweep.MODES]
        ):
            g = summary[
                (summary.ensemble == ensemble)
                & (summary.model == model)
                & (summary.covariance_source == name)
                & (summary.replicates == 3)
            ]
            for selector, label in zip(SELECTORS, LABELS):
                z = g[g.selector == selector].sort_values("maxent")
                axis.plot(
                    z.maxent, z.recovery_mean, marker="o", markersize=4, label=label
                )
                axis.fill_between(
                    z.maxent,
                    z.recovery_mean - z.recovery_sd,
                    z.recovery_mean + z.recovery_sd,
                    alpha=0.10,
                )
            axis.set(
                xscale="log",
                ylim=(0, 100),
                xlabel="MaxEnt",
                ylabel="Recovery (%)",
                title=f"{ensemble} — {'Linear BV' if model == 'linear' else 'Full uptake'}",
            )
            axis.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(
            f"IsoValidation MSE fits: {name.replace('_', ' ')} validation selection\nMean ± sample SD, three replicates; no refitting"
        )
        fig.savefig(output / f"{name}.png", dpi=180)
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)
    html = "<!doctype html><meta charset='utf-8'><title>IsoValidation covariance selection sensitivity</title><style>body{font:15px system-ui;margin:30px}td,th{padding:6px}img{max-width:100%}</style><h1>IsoValidation MSE panel: covariance split-order sensitivity</h1><p>Each saved MSE fit is replayed using its original Linear BV or frame-uptake model and saved parameters. Native convergence checkpoints only; no refitting. Both selectors use the same existing alpha=0 coordinate covariance, including its original numerical ridge. Curves select checkpoints within each MaxEnt; pooled selections select across the MaxEnt grid separately. Plots include only cells with all three replicates; shading is sample SD. Full tables retain incomplete cells with their actual replicate counts.</p><p><a href='summary.csv'>Panel summary CSV</a> · <a href='pooled_selected.csv'>Pooled selected models</a> · <a href='audit.json'>Audit</a></p>"
    html += f"<p>{len(mse_specs)} saved MSE fits; {len(audits)} have eligible convergence checkpoints; {len(missing)} have none. No final-state fallback is introduced.</p>"
    for name in ("closed_coordinate", "gt_coordinate"):
        html += f"<h2>{name}</h2><img src='{name}.png'><h3>Pooled selection across MaxEnt</h3>"
        html += pooled_summary[pooled_summary.covariance_source == name].to_html(
            index=False, float_format=lambda x: f"{x:.3f}"
        )
    html += "<h2>Full panel results</h2>" + summary.to_html(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
    (output / "report.html").write_text(html)
    print(pooled_summary.to_string(index=False))


if __name__ == "__main__":
    main()
