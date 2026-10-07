#!/usr/bin/env python3
"""BV contacts and coefficient fits on single MoPrP109 structures (e.g. an energy-minimised model).

For each structure and each H-bond exclusion window the amide-N heavy contacts (always +-2) and
amide-H -> O contacts are featurised as a single frame (weight 1), then

* contact statistics and a residue-level regression against median.pfact,
* the peptide-level coefficient fit without / with a free global ln(Pf) offset, and a bh scan.

Requires MOPRP_STRUCTURE=109 (resid == seq + 1).  Inputs other than the structure (peptides,
timepoints, kints) are taken from the 109 AF2_MSAss inputs of ``_moprp_recovery_common``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import MDAnalysis as mda
import numpy as np

import _moprp_recovery_common as common
import moprp_acceptor_channel_diagnostic as D
import moprp_bv_offset_fit as O
from jaxent.examples.common.loading import featurise_trajectory, load_HDXer_kints
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.interfaces.topology import TopologyFactory
from jaxent.src.models.HDX.BV.forwardmodel import BV_model_Config
from moprp_pivot_litmus import _peptide_map

DATA = common.BASE / "data"
STRUCTURES = {
    "minimised": None,  # set from --minimised
    "af2_109_max_plddt": DATA / "MoPrP109_s20_r1_msa1-127_n12700_do1_20260904_191954_protonated_max_plddt_1627.pdb",
}
WINDOWS = {"pm2": (-2, 2), "pm1": (-1, 1), "zero": (0, 0)}


def _kints_109():
    rates, topo = load_HDXer_kints(str(DATA / "_MoPrP/hdxrate_poly_pD4p0_298K_min.dat"))
    shifted = [
        TopologyFactory.from_single(chain="A", residue=int(t._get_active_residues(check_trim=False)[0]) + 1)
        for t in topo
    ]
    last = max(int(t._get_active_residues(check_trim=False)[0]) for t in shifted)
    extra = list(range(last + 1, 110))
    return (
        jnp.concatenate([jnp.asarray(rates), jnp.ones(len(extra))]),
        shifted + [TopologyFactory.from_single(chain="A", residue=r) for r in extra],
    )


def _featurise(pdb: Path, window: tuple[int, int], out: Path, name: str, traj: Path | None = None):
    config = BV_model_Config(timepoints=jnp.asarray(np.loadtxt(DATA / "_MoPrP/moprp.times")[1:] * 60.0), switch=False)
    config.temperature, config.ph = 298.0, 4.0
    config.heavy_radius, config.o_radius = 6.5, 2.4
    config.residue_ignore = (-2, 2)
    config.residue_ignore_hbond = window
    config.mda_selection_exclusion = "resname PRO"
    config.mda_contact_environment = "protein"
    feature_path, topology_path = featurise_trajectory(
        trajectory_path=str(traj if traj is not None else pdb),
        topology_path=str(pdb),
        output_dir=str(out),
        output_name=name,
        bv_config=config,
        featuriser_settings=FeaturiserSettings(name="MoPrP_single", batch_size=None),
        kint_data=_kints_109(),
    )
    topo = json.loads(Path(topology_path).read_text())["topologies"]
    ids = np.asarray([t["residues"][0] for t in topo], dtype=int) - 1  # -> moprp.seq numbering
    with np.load(feature_path) as z:
        heavy, acc = np.asarray(z["heavy_contacts"], float), np.asarray(z["acceptor_contacts"], float)
    return ids, heavy, acc


def _inputs_for(ids, heavy, acc, template):
    """Single-frame inputs aligned to the template's feature residues."""

    index = {int(r): i for i, r in enumerate(ids)}
    rows = [index[int(r)] for r in template.feature_residue_ids]
    n_frames = heavy.shape[1]
    return dataclasses.replace(
        template,
        heavy_contacts=heavy[rows],
        acceptor_contacts=acc[rows],
        n_frames=n_frames,
        states=np.asarray(["md"] * n_frames),
        reference_weights=np.full(n_frames, 1.0 / n_frames),
    )


def run(args) -> None:
    if common.STRUCTURE != "109":
        raise SystemExit("set MOPRP_STRUCTURE=109")
    template = common.load_ensemble_inputs("AF2_MSAss", args.rate_source)
    structures = dict(STRUCTURES, minimised=args.minimised)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for sname, pdb in structures.items():
            for wname, window in WINDOWS.items():
                ids, heavy, acc = _featurise(Path(pdb), window, Path(tmp), f"{sname}_{wname}")
                inputs = _inputs_for(ids, heavy, acc, template)
                ens = [(inputs, _peptide_map(inputs))]
                full = O._fit(ens)
                results[f"{sname}/{wname}"] = {
                    "contacts": D._contact_stats(inputs),
                    "residue_regression": D._residue_regression(inputs),
                    "published_mse": O._mse(common.PUBLISHED_BC, common.PUBLISHED_BH, 0.0, ens),
                    "no_offset": O._fit(ens, fix_c=0.0),
                    "free_offset": full,
                    "bh0_offset": O._fit(ens, fix_bh=0.0),
                    "bh_profile_offset": [{"bh": b, **O._fit(ens, fix_bh=b)} for b in O.BH_GRID],
                    "bh_profile_no_offset": [{"bh": b, **O._fit(ens, fix_bh=b, fix_c=0.0)} for b in O.BH_GRID],
                }
                print(sname, wname, "done", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rate_source": args.rate_source, "results": results}, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--minimised", type=Path, required=True, help="protein-only PDB of the minimised structure")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate-source", choices=tuple(common.RATE_SOURCES), default=common.DEFAULT_RATE_SOURCE)
    run(p.parse_args())
