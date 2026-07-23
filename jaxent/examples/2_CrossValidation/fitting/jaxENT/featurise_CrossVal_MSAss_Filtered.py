#!/usr/bin/env python3
"""Generate physics-versioned BV features for the two MoPrP ensembles.

The canonical construction uses binary Best--Vendruscolo contacts and the
official exPfact intrinsic rates at 298 K and pH 4.4.  The historical JAX-ENT
rational switch can be emitted as an explicitly labelled sensitivity.  Legacy
``_featurise`` artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jaxent.examples.common.loading import featurise_trajectory, load_HDXer_kints
from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.interfaces.topology import PTSerialiser
from jaxent.src.models.HDX.BV.forwardmodel import BV_model_Config


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
DEFAULT_OUTPUT = HERE / "_featurise_physics_v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("hard", "switched"),
        default=("hard", "switched"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    topology_path = DATA / "MoPrP_max_plddt_4334.pdb"
    rate_path = DATA / "_MoPrP/expfact_kint_pH4p4_298K_min.dat"
    time_path = DATA / "_MoPrP/moprp.times"
    trajectories = {
        "AF2_MSAss": DATA / "_cluster_MoPrP/clusters/all_clusters.xtc",
        "AF2_filtered": DATA / "_cluster_MoPrP_filtered/clusters/all_clusters.xtc",
    }
    for path in (topology_path, rate_path, time_path, *trajectories.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    exact_times_min = np.loadtxt(time_path, dtype=float)[1:] * 60.0
    rate_data = load_HDXer_kints(str(rate_path))
    featuriser_settings = FeaturiserSettings(name="MoPrP_BV_physics_v2", batch_size=None)

    generated = []
    for mode in args.modes:
        switched = mode == "switched"
        config = BV_model_Config(timepoints=jnp.asarray(exact_times_min), switch=switched)
        config.temperature = 298.0
        config.ph = 4.4
        config.heavy_radius = 6.5
        config.o_radius = 2.4
        config.residue_ignore = (-2, 2)
        config.mda_selection_exclusion = "resname PRO"
        config.mda_contact_environment = "protein"

        for ensemble, trajectory_path in trajectories.items():
            output_name = f"{ensemble}_{mode}"
            feature_path, topology_output_path = featurise_trajectory(
                trajectory_path=str(trajectory_path),
                topology_path=str(topology_path),
                output_dir=str(args.output_dir),
                output_name=output_name,
                bv_config=config,
                featuriser_settings=featuriser_settings,
                kint_data=rate_data,
            )
            with np.load(feature_path) as feature_data:
                shape = list(np.asarray(feature_data["heavy_contacts"]).shape)
                hbond_shape = list(np.asarray(feature_data["acceptor_contacts"]).shape)
                n_rates = int(len(feature_data["k_ints"]))
                hard_contacts_are_integer = bool(
                    np.allclose(
                        feature_data["heavy_contacts"],
                        np.rint(feature_data["heavy_contacts"]),
                    )
                    and np.allclose(
                        feature_data["acceptor_contacts"],
                        np.rint(feature_data["acceptor_contacts"]),
                    )
                )
            feature_topology = PTSerialiser.load_list_from_json(topology_output_path)
            residue_keys = [
                (str(topology.chain), int(topology.residues[0]))
                for topology in feature_topology
            ]
            if shape != hbond_shape or shape[0] != n_rates or shape[0] != len(residue_keys):
                raise RuntimeError(
                    f"inconsistent feature rows for {output_name}: "
                    f"heavy={shape}, hbond={hbond_shape}, rates={n_rates}, "
                    f"topology={len(residue_keys)}"
                )
            if len(set(residue_keys)) != len(residue_keys):
                raise RuntimeError(f"duplicate feature residue keys for {output_name}")
            if ("A", 101) not in residue_keys:
                raise RuntimeError(f"C-terminal amide A:101 missing from {output_name}")
            if mode == "hard" and not hard_contacts_are_integer:
                raise RuntimeError(f"hard contacts are not binary counts for {output_name}")
            generated.append(
                {
                    "ensemble": ensemble,
                    "contact_mode": mode,
                    "features": str(Path(feature_path).resolve()),
                    "topology": str(Path(topology_output_path).resolve()),
                    "feature_sha256": _sha256(Path(feature_path)),
                    "topology_sha256": _sha256(Path(topology_output_path)),
                    "contact_shape": shape,
                    "n_rates": n_rates,
                    "n_topology_rows": len(residue_keys),
                    "contains_c_terminal_amide_101": ("A", 101) in residue_keys,
                    "hard_contacts_are_integer": (
                        hard_contacts_are_integer if mode == "hard" else None
                    ),
                }
            )

    manifest = {
        "construction": "MoPrP_BV_physics_v2",
        "terminal_exclusion": "n",
        "excluded_residue_selection": "resname PRO",
        "contact_environment": "protein",
        "heavy_contact": {"target": "amide_N", "radius_angstrom": 6.5},
        "hbond_contact": {"target": "amide_H", "atom": "protein_O", "radius_angstrom": 2.4},
        "sequence_neighbor_exclusion": [-2, 2],
        "coefficients": {"beta_c": 0.35, "beta_h": 2.0, "log_base": "natural"},
        "intrinsic_rates": {
            "provider": "exPfact-3Ala",
            "temperature_k": 298.0,
            "ph": 4.4,
            "units": "min^-1",
            "path": str(rate_path.resolve()),
            "sha256": _sha256(rate_path),
        },
        "timepoints_min": exact_times_min.tolist(),
        "topology_input": {"path": str(topology_path.resolve()), "sha256": _sha256(topology_path)},
        "trajectories": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in trajectories.items()
        },
        "generated": generated,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"Featurisation complete in {time.time() - started:.2f} seconds")
