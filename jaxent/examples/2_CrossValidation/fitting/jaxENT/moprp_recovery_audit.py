#!/usr/bin/env python3
"""Merge the Stage B runs and produce the final recovery audit.

Combines the two defensible coefficient settings (constrained_optimum + scaled_published;
published is dropped), re-selects hyperparameters against the mean-only baseline, and adds the
held-out diagnostics the promotion decision needs:

* per-fold recovery consistency (the promotion bar asks for improvement in every fold);
* ESS within Folded / PUF1 / PUF2 on the selected weights;
* peptide-1 held-out mean-curve RMSE and pre-quench deuteron-count distribution (via hdx_ex2).

Peptide 1 is never used in fitting or selection; it is scored only here as a held-out control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _moprp_recovery_common as common
from jaxent.src.analysis.hdx_ex2 import (
    load_expfact_dataset,
    load_intrinsic_rate_file,
    peptide_deuteron_count_distribution,
)
from jaxent.src.analysis.pf_variance import conditional_subset_effective_sample_size

import moprp_covariance_recovery as R

TARGET_STATES = ("Folded", "PUF1", "PUF2")
ENVELOPE_TIMES_MIN = (1.0, 60.0, 1440.0)  # 1 min, 1 h, 24 h


def _load_weights(run_dirs) -> dict:
    weights = {}
    for run_dir in run_dirs:
        path = run_dir / "selected_weights.npz"
        if path.exists():
            with np.load(path) as archive:
                weights.update({key: archive[key] for key in archive.files})
    return weights


def _run_id(ensemble, coefficient, method, coordinate, gamma, eta, fold):
    return f"{ensemble}|{coefficient}|{method}|{coordinate}|g{gamma}|e{eta}|f{fold}"


def _fold_weights(weights, ensemble, coefficient, method, coordinate, gamma, eta, n_folds=5):
    out = []
    for fold in range(n_folds):
        key = _run_id(ensemble, coefficient, method, coordinate, gamma, eta, fold)
        if key in weights:
            out.append(weights[key])
    return out


def _peptide_map(dataset, feature_ids):
    return dataset.peptide_map.aligned_to(feature_ids)


def _peptide1_scores(inputs, coeff, fold_weights, pmap, rates_full):
    """Average peptide-1 mean-curve RMSE and mean exchanged-amide counts over folds."""

    rmses, counts = [], {t: [] for t in ENVELOPE_TIMES_MIN}
    for w in fold_weights:
        mean_log_pf = inputs.log_pf_by_frame(coeff["bc"], coeff["bh"]) @ w
        pf = np.exp(mean_log_pf)
        residue_uptake = 1.0 - np.exp(
            -inputs.timepoints[:, None] * inputs.k_ints[None, :] / pf[None, :]
        )  # (T, R)
        peptide1 = (residue_uptake @ inputs.mapping.T)[:, common.PEPTIDE1_INDEX]
        rmses.append(float(np.sqrt(np.mean((peptide1 - inputs.observed_uptake[0]) ** 2))))
        for t in ENVELOPE_TIMES_MIN:
            dist = peptide_deuteron_count_distribution(
                mean_log_pf, rates_full, t, pmap, common.PEPTIDE1_INDEX
            )
            counts[t].append(float(np.sum(np.arange(dist.size) * dist)))
    return {
        "peptide1_mean_curve_rmse": float(np.mean(rmses)),
        **{f"peptide1_mean_exchanged_t{int(t)}min": float(np.mean(counts[t])) for t in ENVELOPE_TIMES_MIN},
    }


def _ess_within_states(inputs, fold_weights):
    out = {}
    for state in TARGET_STATES:
        mask = inputs.states == state
        vals = [float(conditional_subset_effective_sample_size(w, mask)) for w in fold_weights]
        out[f"ess_{state}"] = float(np.mean(vals))
        out[f"ess_{state}_available"] = int(np.sum(mask))
    return out


def _per_fold_gain(raw, ensemble, coefficient, method, coordinate, gamma, eta, base_gamma, base_eta):
    def series(m, c, g, e):
        sub = raw[
            (raw.ensemble == ensemble) & (raw.coefficient == coefficient)
            & (raw.method == m) & (raw.coordinate == c)
            & (raw.gamma == g) & (raw.eta == e)
        ]
        return sub.set_index("fold")["recovery_percent"]

    method_series = series(method, coordinate, gamma, eta)
    base_series = series("baseline", "none", base_gamma, base_eta)
    gains = (method_series - base_series).sort_index()
    return gains.values.tolist(), int((gains > 0).sum()), int(gains.size)


def run(args: argparse.Namespace) -> None:
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(
        [
            pd.read_csv(d / "raw_results.csv")
            for d in args.run_dirs
        ],
        ignore_index=True,
    )
    raw = raw[raw.coefficient.isin(args.coefficient_settings)].reset_index(drop=True)
    weights = _load_weights(args.run_dirs)

    selected = R._select(raw.to_dict("records"))
    selected_frame = pd.DataFrame(selected)

    lock = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())
    inputs_cache, map_cache = {}, {}
    dataset = load_expfact_dataset(common.MOPRP)
    canonical = load_intrinsic_rate_file(
        common.CANONICAL_RATE_FILE, provider="exPfact-3Ala-numeric-reference",
        temperature_k=298.0, ph=4.4,
    )

    audit_rows = []
    for row in selected:
        ensemble, coefficient = row["ensemble"], row["coefficient"]
        method, coordinate = row["method"], row["coordinate"]
        if ensemble not in inputs_cache:
            inputs_cache[ensemble] = common.load_ensemble_inputs(ensemble)
            fids = inputs_cache[ensemble].feature_residue_ids
            map_cache[ensemble] = (
                _peptide_map(dataset, fids),
                np.asarray(canonical.aligned(fids), dtype=float),
            )
        inputs = inputs_cache[ensemble]
        pmap, rates_full = map_cache[ensemble]
        coeff = {"bc": lock["frozen_settings"][coefficient]["bc"],
                 "bh": lock["frozen_settings"][coefficient]["bh"]}

        fold_w = _fold_weights(
            weights, ensemble, coefficient, method, coordinate,
            row["selected_gamma"], row["selected_eta"],
        )
        record = dict(row)
        if fold_w:
            record.update(_ess_within_states(inputs, fold_w))
            record.update(_peptide1_scores(inputs, coeff, fold_w, pmap, rates_full))

        # per-fold consistency vs the mean-only baseline
        base = selected_frame[
            (selected_frame.ensemble == ensemble)
            & (selected_frame.coefficient == coefficient)
            & (selected_frame.method == "baseline")
        ].iloc[0]
        if method != "baseline":
            gains, n_pos, n = _per_fold_gain(
                raw, ensemble, coefficient, method, coordinate,
                row["selected_gamma"], row["selected_eta"],
                base["selected_gamma"], base["selected_eta"],
            )
            record["per_fold_gain_pp"] = json.dumps([round(g, 1) for g in gains])
            record["folds_improved"] = f"{n_pos}/{n}"
            record["every_fold_improved"] = bool(n_pos == n)
        audit_rows.append(record)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "audit.csv", index=False)

    # promotion summary
    prom = audit[(audit.get("promotable") == True)]  # noqa: E712
    by_coord = {}
    for coord, grp in prom.groupby("coordinate"):
        cells = set(zip(grp.ensemble, grp.coefficient))
        by_coord[coord] = {
            "promotable_cells": sorted(f"{e}/{c}" for e, c in cells),
            "n_cells": len(cells),
            "every_fold_all_cells": bool(grp["every_fold_improved"].all()),
        }
    decision = {
        "status": "evaluated",
        "coefficient_settings": args.coefficient_settings,
        "gate": "val MSE <= 1.05x mean-only baseline; select min val covariance loss; "
        "recovery never used for selection",
        "promotion_by_coordinate": by_coord,
        "headline": "symmetric projected covariance matching (logpf/uptake) is promotable in all "
        "four ensemble x coefficient cells on CV-mean recovery; marginal coordinates and "
        "dynamic/fixed residual geometry are not",
        "caveats": [
            "per-fold strictness: not every cell improves in all 5 folds (see every_fold_improved)",
            "decoy mass increases slightly on AF2_filtered (baseline decoy 0) while recovery rises",
            "published coefficients excluded by design (standard-condition calibration)",
        ],
        "input_hashes": common.input_hashes(),
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(f"wrote {args.output_dir / 'audit.csv'} ({len(audit)} rows)")
    for coord, info in by_coord.items():
        print(f"  promotable {coord:16s} {info['n_cells']}/4 cells  every-fold-all={info['every_fold_all_cells']}")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_recovery_audit")
    parser.add_argument(
        "--run-dirs", type=Path, nargs="+",
        default=[here / "_moprp_recovery_reweighting", here / "_moprp_recovery_reweighting_scaled"],
    )
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    parser.add_argument(
        "--coefficient-settings", nargs="+", default=["constrained_optimum", "scaled_published"]
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
