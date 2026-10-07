#!/usr/bin/env python3
"""Stage C, Part A: elastic-network covariance priors from a single structure.

Builds ANM (cutoff sweep) and GNM (cutoff sweep) residue covariances from the max-pLDDT structure
(primary) and the NMR structures (sensitivity), then compares each to the target ``C(w_NMR)`` and
the unweighted ``C(uniform)`` log-PF covariance, at residue and peptide level.  This is a
population-free covariance prior: no trajectory weighting, no known population.

Question answered: how much of the target covariance *structure* (Mantel) and *variance profile*
(diagonal log-variance correlation) is recoverable from topology alone, and does GNM recover the
variance profile that ANM misses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import MDAnalysis as mda

import _moprp_recovery_common as common
from jaxent.src.analysis import covariance_comparison as cc
from jaxent.src.analysis import elastic_network as en
from jaxent.src.analysis.state_population import peptide_logpf_covariance
from jaxent.src.analysis.pf_variance import weighted_population_covariance
import jax.numpy as jnp

DATA = common.BASE / "data"
STRUCTURES = {
    "max_plddt": DATA / "_MoPrP/MoPrP_max_plddt_4334.pdb",
    "nmr_2L1H": DATA / "_MoPrP/2L1H_renum.pdb",
    "nmr_2L39": DATA / "_MoPrP/2L39_renum.pdb",
}
ANM_CUTOFFS = (12.0, 15.0, 18.0, 24.0)
GNM_CUTOFFS = (7.3, 8.0, 10.0)
COEFFICIENT = "scaled_published"


def _structure_covariances(pdb: Path):
    universe = mda.Universe(str(pdb))
    ca = universe.select_atoms("name CA")
    coords = ca.positions.astype(float)
    resid_to_index = {int(r): i for i, r in enumerate(ca.resids)}
    models = {}
    for cutoff in ANM_CUTOFFS:
        models[("anm", cutoff)] = en.anm_covariance(coords, cutoff=cutoff)
    for cutoff in GNM_CUTOFFS:
        models[("gnm", cutoff)] = en.gnm_covariance(coords, cutoff=cutoff)
    return resid_to_index, models


def _log_pf_covariances(inputs, bc, bh):
    lp = jnp.asarray(inputs.log_pf_by_frame(bc, bh))
    M = jnp.asarray(inputs.mapping)
    uniform = jnp.full(inputs.n_frames, 1.0 / inputs.n_frames)
    wT = jnp.asarray(inputs.reference_weights)
    return {
        "residue": {
            "target": np.asarray(weighted_population_covariance(lp, wT)),
            "unweighted": np.asarray(weighted_population_covariance(lp, uniform)),
        },
        "peptide": {
            "target": np.asarray(peptide_logpf_covariance(lp, M, wT)),
            "unweighted": np.asarray(peptide_logpf_covariance(lp, M, uniform)),
        },
    }


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((args.coefficient_lock / "coefficient_lock.json").read_text())
    coeff = lock["frozen_settings"][COEFFICIENT]

    rows = []
    for ensemble in common.ENSEMBLES:
        inputs = common.load_ensemble_inputs(ensemble)
        feature_ids = inputs.feature_residue_ids
        targets = _log_pf_covariances(inputs, coeff["bc"], coeff["bh"])
        M = inputs.mapping

        for structure, pdb in STRUCTURES.items():
            if not pdb.exists():
                continue
            resid_to_index, models = _structure_covariances(pdb)
            present = np.array([int(r) in resid_to_index for r in feature_ids])
            coverage = float(present.mean())
            idx = np.array([resid_to_index[int(r)] for r in feature_ids if int(r) in resid_to_index])
            keep_features = np.where(present)[0]

            for (model, cutoff), cov_full in models.items():
                C_res = cov_full[np.ix_(idx, idx)]
                comparisons = {"residue": {}}
                # residue level (on the covered feature residues)
                tgt_res = targets["residue"]["target"][np.ix_(keep_features, keep_features)]
                unw_res = targets["residue"]["unweighted"][np.ix_(keep_features, keep_features)]
                comparisons["residue"] = {"target": tgt_res, "unweighted": unw_res, "cand": C_res}
                # peptide level only when the structure fully covers the feature residues
                if coverage == 1.0:
                    C_pep = M @ C_res @ M.T
                    comparisons["peptide"] = {
                        "target": targets["peptide"]["target"],
                        "unweighted": targets["peptide"]["unweighted"],
                        "cand": C_pep,
                    }
                for level, blocks in comparisons.items():
                    for ref in ("target", "unweighted"):
                        m = cc.covariance_metrics(blocks["cand"], blocks[ref], permutations=args.permutations)
                        rows.append({
                            "ensemble": ensemble, "structure": structure, "coverage": coverage,
                            "model": model, "cutoff": cutoff, "level": level, "reference": ref,
                            **m,
                        })

    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "elastic_network_comparison.csv", index=False)

    # best model/cutoff per (ensemble, level) vs the target, by Mantel then diag correlation
    summary = {}
    tvt = frame[(frame.reference == "target")]
    for (ensemble, level), grp in tvt.groupby(["ensemble", "level"]):
        best_struct = grp[grp.structure == "max_plddt"] if "max_plddt" in set(grp.structure) else grp
        best = best_struct.sort_values(["mantel_r", "diag_log_corr"], ascending=False).iloc[0]
        summary[f"{ensemble}/{level}"] = {
            "best_model": best.model, "best_cutoff": float(best.cutoff),
            "mantel_r": float(best.mantel_r), "diag_log_corr": float(best.diag_log_corr),
            "offdiag_corr": float(best.offdiag_corr),
        }
    (args.output_dir / "elastic_network_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"wrote {args.output_dir / 'elastic_network_comparison.csv'} ({len(frame)} rows)")
    show = frame[(frame.reference == "target") & (frame.structure == "max_plddt")]
    for _, r in show.sort_values(["ensemble", "level", "model", "cutoff"]).iterrows():
        print("  {:4s}/{:7s} {:3s} rc={:4.1f} {:7s} mantelR={:+.3f} diagVarR={:+.3f} offR={:+.3f} p={:.3f}".format(
            r.ensemble[:4], "max", r.model, r.cutoff, r.level, r.mantel_r, r.diag_log_corr, r.offdiag_corr, r.mantel_p))


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=here / "_moprp_elastic_network_prior")
    parser.add_argument("--coefficient-lock", type=Path, default=here / "_moprp_recovery_coefficient_lock")
    parser.add_argument("--permutations", type=int, default=999)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
