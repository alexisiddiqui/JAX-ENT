"""Save precomputed FOXS SAXS curves as JAX-ENT SAXS input features.

This script converts raw FOXS matrix data into a serialized
``SAXS_curve_input_features`` object so downstream SAXS fitting scripts can
load a ready-to-use feature file instead of rebuilding it each run.

Expected input:
- ``--SAXS_features``: path to a ``.npz`` file that contains per-frame SAXS
  curves (typically ``saxs`` with shape ``(n_frames, n_q)``).
- ``--output_dir``: output directory where ``SAXS_features.npz`` is written.

Outputs:
- ``SAXS_features.npz`` (serialized ``SAXS_curve_input_features``)
- ``synthetic_SAXS_topology.json`` (chain topology JSON, matching example layout)





"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax.numpy as jnp
import MDAnalysis as mda
import numpy as np

from jaxent.src.models.SAXS.features import SAXS_curve_input_features
from jaxent.src.interfaces.topology import PTSerialiser, Partial_Topology, mda_TopologyAdapter


def _load_raw_curve_matrix(input_path: Path) -> np.ndarray:
    """Load the raw FOXS curve matrix from disk."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input SAXS feature file not found: {input_path}")

    if input_path.suffix != ".npz":
        raise ValueError(
            f"Expected a .npz input for --SAXS_features, got: {input_path.suffix or '<no extension>'}"
        )

    with np.load(input_path) as data:
        if "saxs" in data:
            matrix = np.asarray(data["saxs"])
        elif len(data.files) == 1:
            matrix = np.asarray(data[data.files[0]])
        else:
            raise ValueError(
                "Input .npz must contain key 'saxs' or exactly one array. "
                f"Found keys: {data.files}"
            )

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D SAXS matrix, got shape {matrix.shape} with {matrix.ndim} dimensions."
        )

    return matrix


def _to_nq_nframes_layout(raw_matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize matrix to (n_q, n_frames), transposing when needed."""
    n_rows, n_cols = raw_matrix.shape

    if n_rows > n_cols:
        return raw_matrix.T, True

    return raw_matrix, False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serialize raw FOXS SAXS arrays to JAX-ENT SAXS_curve_input_features."
    )
    parser.add_argument(
        "--SAXS_features",
        required=True,
        help="Path to raw FOXS .npz curve matrix (e.g. CaM_SAXS_ordered.npz).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where SAXS_features.npz will be saved.",
    )
    return parser.parse_args()


def _resolve_default_topology_pdb(input_path: Path) -> Path:
    """Resolve a default PDB topology path from the FOXS directory structure."""
    missing_residues_dir = input_path.parent / "missing_residues"
    preferred = missing_residues_dir / "1CLL_apo.pdb"
    if preferred.exists():
        return preferred

    pdb_candidates = sorted(missing_residues_dir.glob("*.pdb"))
    if pdb_candidates:
        return pdb_candidates[0]

    raise FileNotFoundError(
        "Could not resolve a topology PDB file in "
        f"{missing_residues_dir}. Expected 1CLL_apo.pdb or another .pdb file."
    )


def _extract_chain_topologies(pdb_path: Path) -> list[Partial_Topology]:
    """Extract chain-level topologies from a PDB using built-in topology adapters."""
    universe = mda.Universe(str(pdb_path))
    topologies = mda_TopologyAdapter.from_mda_universe(
        universe=universe,
        mode="chain",
        include_selection="protein",
        exclude_termini=False,
        renumber_residues=True,
    )

    for idx, topology in enumerate(topologies):
        topology.fragment_index = idx

    return topologies


def main() -> None:
    args = _parse_args()

    input_path = Path(args.SAXS_features).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_matrix = _load_raw_curve_matrix(input_path)
    curve_matrix, transposed = _to_nq_nframes_layout(raw_matrix)

    features = SAXS_curve_input_features(intensities=jnp.asarray(curve_matrix))

    output_path = os.path.join(str(output_dir), "synthetic_SAXS.npz")
    features.save(output_path)

    topology_pdb = _resolve_default_topology_pdb(input_path)
    topologies = _extract_chain_topologies(topology_pdb)
    topology_output = output_dir / "synthetic_SAXS_topology.json"
    PTSerialiser.save_list_to_json(topologies, topology_output)

    print(f"Loaded raw SAXS matrix: {input_path}")
    print(f"Input matrix shape: {raw_matrix.shape}")
    print(f"Transposed to (n_q, n_frames): {transposed}")
    print(f"Serialized intensities shape: {curve_matrix.shape}")
    print(f"Saved JAX-ENT features to: {output_path}")
    print(f"Saved topology JSON to: {topology_output}")


if __name__ == "__main__":
    main()
