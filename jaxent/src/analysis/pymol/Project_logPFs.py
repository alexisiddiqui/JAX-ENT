"""
Project protection factors — colour a reference structure by per-replicate
log protection factor values (or derived metrics).

The ``target_data`` array has shape (n_replicates, n_residues) and EXCLUDES
Prolines and the N-terminal residue of each chain — the same convention used
by the BV forward model (base_exclude_selection="resname PRO",
exclude_termini=True).  The script auto-derives this residue mapping from the
``target_topology`` PDB.

Supported metrics:
  protection_factor   — replicate-average log PF
  uncertainty_sd      — standard deviation across replicates
  uncertainty_rsd     — relative SD (%) = std / |mean| * 100
  difference_signed   — mean - reference_pf  (requires reference_data)
  difference_absolute — |mean - reference_pf| (requires reference_data)

Usage (inside PyMOL):
    run Project_logPFs.py config.yaml
    run Project_logPFs.py config.yaml --pf_metric difference_signed
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_METRICS = frozenset(
    [
        "protection_factor",
        "uncertainty_sd",
        "uncertainty_rsd",
        "difference_signed",
        "difference_absolute",
    ]
)

# ---------------------------------------------------------------------------
# PyMOL guard
# ---------------------------------------------------------------------------

import inspect as _inspect
_SCRIPT_DIR = os.path.dirname(
    os.path.abspath(_inspect.currentframe().f_code.co_filename)
)
del _inspect
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import is_running_in_pymol  # noqa: E402

if not is_running_in_pymol():
    logger.error("Project_logPFs.py must be run inside PyMOL")
    sys.exit(1)

from pymol import cmd  # noqa: E402

from config import (  # noqa: E402
    UnifiedConfig,
    build_argparser,
    load_config,
    merge_config_with_args,
)
from utils import (  # noqa: E402
    apply_bfactor_spectrum,
    apply_putty_settings,
    apply_render_settings,
    ensure_nonzero_bfactors,
    save_output,
    set_background_white,
    set_bfactors_from_dict,
)


# ---------------------------------------------------------------------------
# Residue mapping
# ---------------------------------------------------------------------------

def build_residue_mapping(topology_pdb: str) -> list:
    """Return ordered PDB residue numbers matching target_data columns.

    Mirrors the BV model exclusion convention:
      - Skip the first residue of each chain (N-terminus, exclude_termini=True)
      - Skip all PRO residues (base_exclude_selection="resname PRO")

    See: jaxent/src/models/HDX/BV/forwardmodel.py:33-34
    """
    temp = "__topo_residue_map__"
    try:
        cmd.load(topology_pdb, temp)
        model = cmd.get_model(f"{temp} and name CA")

        # Group by chain, sorted by residue number
        chain_residues: dict = {}
        for atom in model.atom:
            chain = atom.chain or "A"
            resi = int(atom.resi.strip())
            resn = atom.resn.strip().upper()
            chain_residues.setdefault(chain, []).append((resi, resn))

        mapped: list = []
        for chain in sorted(chain_residues.keys()):
            residues = sorted(chain_residues[chain], key=lambda x: x[0])
            for i, (resi, resn) in enumerate(residues):
                if i == 0:
                    # N-terminus excluded
                    logger.debug("Skipping N-terminus: chain %s resi %d (%s)", chain, resi, resn)
                    continue
                if resn == "PRO":
                    logger.debug("Skipping PRO: chain %s resi %d", chain, resi)
                    continue
                mapped.append(resi)

        cmd.delete(temp)
        logger.info(
            "Residue mapping built: %d mapped residues from %s",
            len(mapped),
            os.path.basename(topology_pdb),
        )
        logger.debug("Mapped residues: %s", mapped)
        return mapped

    except Exception as exc:
        logger.error("Failed to build residue mapping from %s: %s", topology_pdb, exc)
        try:
            cmd.delete(temp)
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metric(
    target_data: np.ndarray,
    metric: str,
    ref_pf_map: dict | None,
    mapped_residues: list,
) -> tuple[np.ndarray, dict]:
    """Compute a scalar metric array over residues.

    Returns:
        values     — np.ndarray shape (n_residues,), possibly with np.nan
        value_dict — {str(resi): float} ready for set_bfactors_from_dict()
    """
    mean = np.mean(target_data, axis=0)      # (n_residues,)
    std = np.std(target_data, axis=0)        # (n_residues,)

    if metric == "protection_factor":
        values = mean
    elif metric == "uncertainty_sd":
        values = std
    elif metric == "uncertainty_rsd":
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(
                np.abs(mean) > 1e-10,
                std / np.abs(mean) * 100.0,
                np.nan,
            )
    elif metric in ("difference_signed", "difference_absolute"):
        if ref_pf_map is None:
            logger.error("metric='%s' requires reference_data", metric)
            values = np.full(len(mean), np.nan)
        else:
            ref_values = np.array(
                [ref_pf_map.get(resi, np.nan) for resi in mapped_residues],
                dtype=float,
            )
            n_missing = int(np.sum(np.isnan(ref_values)))
            if n_missing > 0:
                logger.warning(
                    "%d residues in target_data have no match in reference_data "
                    "(will be NaN in output)",
                    n_missing,
                )
            diff = mean - ref_values
            values = np.abs(diff) if metric == "difference_absolute" else diff
    else:
        raise ValueError(
            f"Unknown metric: {metric!r}.  "
            f"Valid options: {sorted(VALID_METRICS)}"
        )

    # Build resi-keyed dict (str keys for cmd.alter compatibility)
    value_dict: dict = {}
    for i, resi in enumerate(mapped_residues):
        v = float(values[i]) if not np.isnan(values[i]) else 0.0
        value_dict[str(resi)] = v

    return values, value_dict


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class LogPFProjector:
    """Project per-replicate protection factor data onto a reference structure."""

    def __init__(self, config: UnifiedConfig) -> None:
        self.cfg = config
        labels = config.general.reference_labels
        self.obj_name: str = labels[0] if labels else "protein"
        self.mapped_residues: list = []
        self.target_data: np.ndarray = np.empty((0, 0))
        self.values: np.ndarray = np.empty(0)
        self.value_dict: dict = {}
        self.nan_residues: set = set()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("=" * 70)
        logger.info("PROJECT PROTECTION FACTORS")
        logger.info("=" * 70)
        t0 = time.time()

        set_background_white()
        self.cfg.resolve_paths()

        self._build_residue_map()
        self._load_target_data()
        self._compute_metric_values()
        self._load_reference_structure()
        self._project_onto_structure()
        self._finalize_scene()

        logger.info("COMPLETE: %.2fs", time.time() - t0)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _build_residue_map(self) -> None:
        topo = self.cfg.project_logpfs.target_topology
        if not topo:
            logger.error("target_topology not specified")
            return
        self.mapped_residues = build_residue_mapping(topo)

    def _load_target_data(self) -> None:
        path = self.cfg.project_logpfs.target_data
        if not path:
            logger.error("target_data not specified")
            return
        try:
            arr = np.load(path)["pred_ln_pf"]
            if arr.ndim != 2:
                raise ValueError(f"Expected 2-D array, got shape {arr.shape}")
            self.target_data = arr.astype(float)
            logger.info(
                "Loaded target_data: shape=%s  (%d replicates × %d residues)",
                self.target_data.shape,
                self.target_data.shape[0],
                self.target_data.shape[1],
            )
            if self.mapped_residues and len(self.mapped_residues) != self.target_data.shape[1]:
                logger.warning(
                    "Residue mapping length (%d) ≠ target_data n_residues (%d). "
                    "Check that topology and npy file are consistent.",
                    len(self.mapped_residues),
                    self.target_data.shape[1],
                )
        except Exception as exc:
            logger.error("Failed to load target_data from %s: %s", path, exc)

    def _load_reference_data(self) -> dict | None:
        """Load reference protection factors from a 2-column .dat file.

        Returns {int(resnum): float(pf)} or None if no file is specified.
        """
        path = self.cfg.project_logpfs.reference_data
        if not path:
            return None
        ref_map: dict = {}
        try:
            with open(path, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        ref_map[int(parts[0])] = float(parts[1])
                    except ValueError:
                        pass
            logger.info(
                "Loaded reference_data: %d residues from %s",
                len(ref_map),
                os.path.basename(path),
            )
            return ref_map
        except Exception as exc:
            logger.error("Failed to load reference_data from %s: %s", path, exc)
            return None

    def _compute_metric_values(self) -> None:
        if self.target_data.size == 0:
            logger.error("target_data is empty; cannot compute metric")
            return

        metric = self.cfg.project_logpfs.metric
        if metric not in VALID_METRICS:
            logger.error(
                "Unknown metric: %r  Valid: %s", metric, sorted(VALID_METRICS)
            )
            return

        ref_pf_map = self._load_reference_data()
        n_res = min(len(self.mapped_residues), self.target_data.shape[1])

        self.values, self.value_dict = compute_metric(
            self.target_data[:, :n_res],
            metric,
            ref_pf_map,
            self.mapped_residues[:n_res],
        )

        # Track NaN residues for grey colouring
        for i, resi in enumerate(self.mapped_residues[:n_res]):
            if i < len(self.values) and np.isnan(self.values[i]):
                self.nan_residues.add(str(resi))

        finite = self.values[np.isfinite(self.values)]
        if len(finite) > 0:
            logger.info(
                "Metric '%s' — range [%.3f, %.3f]  mean=%.3f",
                metric,
                float(np.min(finite)),
                float(np.max(finite)),
                float(np.mean(finite)),
            )

    def _load_reference_structure(self) -> None:
        refs = self.cfg.general.references
        if not refs:
            logger.error("No reference structure specified (general.references)")
            return
        ref_path = refs[0]
        try:
            cmd.load(ref_path, self.obj_name)
            logger.info(
                "Loaded projection structure: %s → %s",
                os.path.basename(ref_path),
                self.obj_name,
            )
        except Exception as exc:
            logger.error("Failed to load projection structure: %s", exc)

    def _project_onto_structure(self) -> None:
        if not self.value_dict:
            logger.error("No values to project — did metric computation succeed?")
            return

        set_bfactors_from_dict(self.obj_name, self.value_dict)

        # Grey out residues with NaN values (no data)
        if self.nan_residues:
            logger.info("Greying out %d NaN residues", len(self.nan_residues))
            from pymol import stored
            stored.nan_res = self.nan_residues
            cmd.color("grey60", f"{self.obj_name} and resi {'+'.join(self.nan_residues)}")

        cmd.show("cartoon", self.obj_name)
        cmd.cartoon("putty", self.obj_name)

    def _finalize_scene(self) -> None:
        render = self.cfg.render
        apply_bfactor_spectrum(self.obj_name, render)
        apply_putty_settings(self.obj_name, render)
        cmd.set("cartoon_transparency", render.reference_transparency, self.obj_name)

        ensure_nonzero_bfactors(self.obj_name)
        apply_render_settings(render)

        metric_name = self.cfg.project_logpfs.metric
        save_output(
            self.cfg.output,
            f"logPFs_{metric_name}",
            self.cfg.general.working_dir,
            self.cfg.render,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_argparser("Project protection factors onto structure")
    args, _ = parser.parse_known_args()

    config: UnifiedConfig = load_config(args.config_file) if args.config_file else UnifiedConfig()
    config = merge_config_with_args(config, args)

    try:
        LogPFProjector(config).run()
    except Exception as exc:
        logger.error("Script failed: %s", exc, exc_info=True)
