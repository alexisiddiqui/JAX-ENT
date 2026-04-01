"""
RMSD by residue — generic per-residue RMSD visualisation.

Loads an ensemble (PDB or trajectory), aligns it to a reference structure (or
performs intra-fit if no reference is provided), computes per-residue RMSD
relative to the mean structure, and colours the ensemble by B-factor.

Usage (inside PyMOL):
    run RMSD_by_res.py config.yaml
    run RMSD_by_res.py config.yaml --spectrum_colours "blue_white_red"
    run RMSD_by_res.py --trajectory ensemble.pdb --references ref.pdb
"""

from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PyMOL guard — must run inside PyMOL
# ---------------------------------------------------------------------------

# Resolve the directory containing this script so local imports work regardless
# of where PyMOL was launched from.
# NOTE: __file__ is unreliable inside PyMOL's execfile; use the code object's
# co_filename which PyMOL sets correctly via compile(source, file, 'exec').
import inspect as _inspect
_SCRIPT_DIR = os.path.dirname(
    os.path.abspath(_inspect.currentframe().f_code.co_filename)
)
del _inspect
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import is_running_in_pymol  # noqa: E402

if not is_running_in_pymol():
    logger.error("RMSD_by_res.py must be run inside PyMOL")
    sys.exit(1)

from pymol import cmd  # noqa: E402

from config import (  # noqa: E402
    UnifiedConfig,
    build_argparser,
    load_config,
    merge_config_with_args,
    parse_trajectory_string,
)
from utils import (  # noqa: E402
    align_pymol_object_to_coords,
    apply_bfactor_spectrum,
    apply_putty_settings,
    apply_render_settings,
    compute_all_atom_mean,
    compute_residue_rmsd_from_dict,
    ensure_nonzero_bfactors,
    get_reference_coords_dict,
    get_reference_target_coords,
    load_trajectory,
    save_output,
    set_background_white,
    set_bfactors_from_dict,
)


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class RMSDByResVisualizer:
    """Per-residue RMSD ensemble visualiser."""

    def __init__(self, config: UnifiedConfig) -> None:
        self.cfg = config
        self.obj_name: str = config.general.trajectory_label
        self.ref_objects: dict = {}   # label → PyMOL object name
        self.frame_objs: list = []
        self.mean_obj_name: str = ""
        self.viz_target: str = self.obj_name

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("=" * 70)
        logger.info("RMSD BY RESIDUE VISUALIZATION")
        logger.info("=" * 70)
        t0 = time.time()

        set_background_white()
        self.cfg.resolve_paths()

        self._load()
        self._load_references()
        self._align()
        self._compute_bfactors()
        self._finalize_scene()

        logger.info("COMPLETE: %.2fs", time.time() - t0)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _load(self) -> None:
        logger.info("-" * 40)
        logger.info("Loading ensemble …")
        traj_str = self.cfg.rmsd.trajectory
        if not traj_str:
            logger.error("No trajectory specified (rmsd.trajectory)")
            return
        pdb_file, traj_file = parse_trajectory_string(traj_str)
        load_trajectory(self.obj_name, pdb_file, traj_file)

    def _load_references(self) -> None:
        refs = self.cfg.general.references
        labels = self.cfg.general.reference_labels
        if not refs:
            return
        logger.info("-" * 40)
        logger.info("Loading %d reference structure(s) …", len(refs))
        for i, ref_path in enumerate(refs):
            label = labels[i] if i < len(labels) else f"ref{i + 1}"
            obj = f"{self.obj_name}_{label}_ref"
            try:
                cmd.load(ref_path, obj)
                self.ref_objects[label] = obj
                logger.info("  %s → %s", os.path.basename(ref_path), obj)
            except Exception as exc:
                logger.warning("Failed to load reference %s: %s", label, exc)

    def _align(self) -> None:
        logger.info("-" * 40)
        logger.info("Aligning ensemble …")
        align_atoms = self.cfg.general.align_atoms
        align_sel = self.cfg.general.align_selection

        if self.cfg.general.references:
            first_ref = self.cfg.general.references[0]
            target = get_reference_target_coords(first_ref, align_atoms, align_sel)
            if target is not None:
                align_pymol_object_to_coords(self.obj_name, target, align_atoms, align_sel)
                # Align all reference objects to the same target frame
                for obj in self.ref_objects.values():
                    align_pymol_object_to_coords(obj, target, align_atoms, align_sel)
                return
            logger.warning("Could not extract alignment target; falling back to intra_fit")

        # Fallback: intra-fit
        from utils import build_align_selection
        sel = build_align_selection(self.obj_name, align_atoms, align_sel)
        logger.info("Intra-fit alignment on: %s", sel)
        try:
            cmd.intra_fit(sel, quiet=1)
        except Exception as exc:
            logger.warning("intra_fit failed: %s", exc)

    def _compute_bfactors(self) -> None:
        logger.info("-" * 40)
        logger.info("Computing per-residue RMSD B-factors …")

        # Compute mean all-atom coords BEFORE splitting states
        mean_coords = compute_all_atom_mean(self.obj_name)

        # Build reference dict from the (aligned) ensemble average
        ref_coords_dict = get_reference_coords_dict(self.obj_name, is_file=False)

        # Split states into individual frame objects
        base = self.obj_name
        prefix = f"{base}_frm_"
        logger.info("Splitting %s into individual frames …", base)
        cmd.split_states(base, prefix=prefix)

        all_objs = cmd.get_object_list("all")
        self.frame_objs = sorted(o for o in all_objs if o.startswith(prefix))

        if not self.frame_objs:
            logger.error("No frames generated by split_states!")
            return

        logger.info("%d frame objects created", len(self.frame_objs))

        # Group frames
        group_name = f"{base}_frames_grp"
        cmd.group(group_name, " ".join(self.frame_objs))
        self.viz_target = group_name

        # Per-frame B-factors: displacement from mean
        sum_sq_dev: dict = {}

        for obj in self.frame_objs:
            try:
                model = cmd.get_model(f"{obj} and name CA")
                dev_map: dict = {}
                for atom in model.atom:
                    resi = atom.resi.strip()
                    curr = np.array(atom.coord)
                    if resi in ref_coords_dict:
                        dist = float(np.linalg.norm(curr - ref_coords_dict[resi]))
                        dev_map[resi] = dist
                        sum_sq_dev[resi] = sum_sq_dev.get(resi, 0.0) + dist ** 2
                    else:
                        dev_map[resi] = 0.0
                set_bfactors_from_dict(obj, dev_map)
            except Exception as exc:
                logger.error("Error processing frame %s: %s", obj, exc)

        # Mean structure with ensemble RMSD B-factors
        self.mean_obj_name = f"{base}_mean"
        cmd.create(self.mean_obj_name, self.frame_objs[0])
        if mean_coords is not None:
            cmd.load_coords(mean_coords, self.mean_obj_name, state=1)

        n_frames = len(self.frame_objs)
        rmsd_map = {
            resi: float(np.sqrt(sq / n_frames))
            for resi, sq in sum_sq_dev.items()
        }
        set_bfactors_from_dict(self.mean_obj_name, rmsd_map)

        # Remove the original multi-state object
        cmd.delete(base)
        logger.info("B-factors computed; mean structure: %s", self.mean_obj_name)

    def _finalize_scene(self) -> None:
        logger.info("-" * 40)
        logger.info("Finalizing scene …")
        render = self.cfg.render

        # Frame group — cartoon putty, semi-transparent
        cmd.show("cartoon", self.viz_target)
        cmd.cartoon("putty", self.viz_target)
        for frm in self.frame_objs:
            cmd.set("cartoon_transparency", render.trajectory_transparency, frm)
        apply_putty_settings(self.viz_target, render)
        apply_bfactor_spectrum(self.viz_target, render)

        # Mean structure — opaque, same spectrum
        if self.mean_obj_name:
            cmd.show("cartoon", self.mean_obj_name)
            cmd.cartoon("putty", self.mean_obj_name)
            cmd.set("cartoon_transparency", 0.0, self.mean_obj_name)
            apply_putty_settings(self.mean_obj_name, render)
            apply_bfactor_spectrum(self.mean_obj_name, render)

        # Reference structures — tube, semi-transparent
        colors = self.cfg.general.reference_colors
        for i, (label, obj) in enumerate(self.ref_objects.items()):
            color = colors[i] if i < len(colors) else "white"
            ensure_nonzero_bfactors(obj, min_value=0.5)
            cmd.show("cartoon", obj)
            cmd.cartoon("tube", obj)
            cmd.set("all_states", 1, obj)
            cmd.color(color, obj)
            cmd.set("cartoon_tube_radius", 0.5, obj)
            cmd.set("cartoon_transparency", render.reference_transparency, obj)
            cmd.enable(obj)

        # Global render settings
        apply_render_settings(render)
        save_output(self.cfg.output, "RMSD_by_res", self.cfg.general.working_dir, self.cfg.render)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np  # noqa: F401 — ensure numpy available before utils calls

    # parse_known_args() silently ignores PyMOL's own flags (-r, -d, etc.)
    parser = build_argparser("Per-residue RMSD ensemble visualisation")
    args, _ = parser.parse_known_args()

    config: UnifiedConfig = load_config(args.config_file) if args.config_file else UnifiedConfig()
    config = merge_config_with_args(config, args)

    try:
        RMSDByResVisualizer(config).run()
    except Exception as exc:
        logger.error("Script failed: %s", exc, exc_info=True)
