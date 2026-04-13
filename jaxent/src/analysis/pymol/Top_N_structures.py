"""
Top-N structures — generic top-N frame visualiser.

Loads a trajectory and reference structures, identifies the top-N frames by
highest weight (always), and colours them by either weight or RMSD to the
first reference structure.

colour_metric="weight"  → colour top-N frames by log-weight (default)
colour_metric="RMSD"    → colour top-N frames by RMSD to references[0]

Usage (inside PyMOL):
    run Top_N_structures.py config.yaml
    run Top_N_structures.py config.yaml --colour_metric RMSD --top_n 5
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
    logger.error("Top_N_structures.py must be run inside PyMOL")
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
    EPSILON,
    align_pymol_object_to_coords,
    apply_render_settings,
    apply_bfactor_spectrum,
    ensure_nonzero_bfactors,
    get_reference_target_coords,
    load_trajectory,
    save_output,
    set_background_white,
    set_bfactors_per_state,
)

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Local helper
# ---------------------------------------------------------------------------

def compute_rmsd_to_reference(
    frame_obj: str,
    ref_obj: str,
    align_atoms: str,
) -> float:
    """Align frame_obj to ref_obj and return the CA RMSD."""
    temp = f"__temp_rmsd_frame_{os.getpid()}__"
    try:
        cmd.create(temp, frame_obj, 1, 1)
        cmd.align(temp, ref_obj, cycles=0, quiet=1)
        rms = cmd.rms_cur(
            f"{temp} and {align_atoms}",
            f"{ref_obj} and {align_atoms}",
            matchmaker=0,
        )
        return float(rms)
    except Exception as exc:
        logger.debug("RMSD computation failed for %s: %s", frame_obj, exc)
        return float("inf")
    finally:
        try:
            cmd.delete(temp)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class TopNVisualizer:
    """Top-N ensemble frame visualiser (weight or RMSD metric)."""

    def __init__(self, config: UnifiedConfig) -> None:
        self.cfg = config
        self.obj_name: str = config.general.trajectory_label
        self.ref_objects: dict = {}       # label → PyMOL object name
        self.weights: np.ndarray = np.array([])
        self.top_group: str = ""

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("=" * 70)
        logger.info("TOP-N STRUCTURES VISUALIZATION")
        logger.info("=" * 70)
        t0 = time.time()

        set_background_white()
        self.cfg.resolve_paths()

        self._load_weights()
        self._load_references()
        self._load_ensemble()
        self._align_ensemble()
        self._select_and_show_top_n()
        self._finalize_scene()

        logger.info("COMPLETE: %.2fs", time.time() - t0)
        logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        weights_path = self.cfg.top_n.weights
        if not weights_path:
            logger.info("No weights file specified — weight-based coloring disabled")
            return

        try:
            logger.info("Loading weights: %s", os.path.basename(weights_path))
            data = np.load(weights_path)

            # Try common key names across different JAX-ENT run outputs
            raw: np.ndarray | None = None
            for key in ("frame_weights", "weights"):
                if key in data:
                    raw = np.array(data[key], dtype=float)
                    logger.info("  Found weights under key '%s'", key)
                    break

            if raw is None:
                available = list(data.keys())
                logger.warning(
                    "No recognised key in weights file. Available: %s", available
                )
                if available:
                    raw = np.array(data[available[0]], dtype=float)
                    logger.info("  Using first available key: '%s'", available[0])
                else:
                    return

            # Collapse (n_replicates, n_frames) → (n_frames,)
            if raw.ndim == 2:
                raw = np.mean(raw, axis=0)
                logger.info("  Collapsed %s → mean per-frame weights", raw.shape)

            # Sanitize
            if np.any(np.isnan(raw)):
                med = float(np.nanmedian(raw))
                raw[np.isnan(raw)] = med if not np.isnan(med) else 1.0

            raw[~np.isfinite(raw)] = EPSILON
            raw[raw < EPSILON] = EPSILON

            self.weights = raw/np.sum(raw)
            logger.info("  Weights loaded: %d frames", len(self.weights))

        except Exception as exc:
            logger.error("Failed to load weights: %s", exc)

    def _load_references(self) -> None:
        refs = self.cfg.general.references
        labels = self.cfg.general.reference_labels
        colors = self.cfg.general.reference_colors
        if not refs:
            return

        logger.info("-" * 40)
        logger.info("Loading %d reference structure(s) …", len(refs))

        # Use first reference for alignment target
        align_atoms = self.cfg.general.align_atoms
        align_sel = self.cfg.general.align_selection
        target_coords = get_reference_target_coords(refs[0], align_atoms, align_sel)

        for i, ref_path in enumerate(refs):
            label = labels[i] if i < len(labels) else f"ref{i + 1}"
            obj = f"{self.obj_name}_{label}_ref"
            try:
                cmd.load(ref_path, obj)
                logger.info("  %s → %s", os.path.basename(ref_path), label)
            except Exception as exc:
                logger.warning("Failed to load reference %s: %s", label, exc)
                continue

            if target_coords is not None:
                align_pymol_object_to_coords(obj, target_coords, align_atoms, align_sel)

            # Style
            color = colors[i] if i < len(colors) else "white"
            _apply_reference_color(obj, color)
            cmd.show_as("cartoon", obj)
            cmd.cartoon("dash", obj)
            cmd.set("dash_width", 0.1, obj)
            cmd.set("cartoon_transparency", self.cfg.render.reference_transparency, obj)
            cmd.set("all_states", 1, obj)
            ensure_nonzero_bfactors(obj, min_value=1.0)

            self.ref_objects[label] = obj

    def _load_ensemble(self) -> None:
        logger.info("-" * 40)
        logger.info("Loading ensemble …")
        traj_str = self.cfg.top_n.trajectory
        if not traj_str:
            logger.error("No trajectory specified (top_n.trajectory)")
            return
        pdb_file, traj_file = parse_trajectory_string(traj_str)
        load_trajectory(self.obj_name, pdb_file, traj_file)
        ensure_nonzero_bfactors(self.obj_name, min_value=1.0)

    def _align_ensemble(self) -> None:
        logger.info("-" * 40)
        logger.info("Aligning ensemble …")
        align_atoms = self.cfg.general.align_atoms
        align_sel = self.cfg.general.align_selection

        if self.cfg.general.references:
            target = get_reference_target_coords(
                self.cfg.general.references[0], align_atoms, align_sel
            )
            if target is not None:
                align_pymol_object_to_coords(self.obj_name, target, align_atoms, align_sel)
                return

        # Fallback
        from utils import build_align_selection
        sel = build_align_selection(self.obj_name, align_atoms, align_sel)
        logger.info("Intra-fit fallback on: %s", sel)
        try:
            cmd.intra_fit(sel, quiet=1)
        except Exception as exc:
            logger.warning("intra_fit failed: %s", exc)

    def _select_and_show_top_n(self) -> None:
        colour_metric = self.cfg.top_n.colour_metric.upper()
        transparency_metric = self.cfg.top_n.transparency_metric.upper()
        n = self.cfg.top_n.top_n
        n_states = cmd.count_states(self.obj_name)

        if n_states == 0:
            logger.error("No states in ensemble object — cannot select top-N")
            return

        logger.info("-" * 40)
        logger.info(
            "Selecting top-%d frames by weight; "
            "colour_metric=%s  transparency_metric=%s …",
            n, colour_metric, transparency_metric,
        )

        # Always select by weight
        top_indices, _weight_bvals, _w_min, _w_max = self._top_n_by_weight(n, n_states)

        # --- Colour B-values ---
        if colour_metric == "RMSD":
            b_values, _b_min, _b_max = self._bvals_by_rmsd(top_indices)
        else:
            b_values = _weight_bvals

        # --- Transparency ordering ---
        # Build a parallel array of the metric used for sorting transparency.
        # We need RMSD for the selected frames whether or not it's also the colour metric.
        if transparency_metric == "RMSD":
            if colour_metric == "RMSD":
                # reuse already-computed RMSD values (same frames, same order)
                transparency_vals = np.array(b_values, dtype=float)
            else:
                t_bvals, _tb_min, _tb_max = self._bvals_by_rmsd(top_indices)
                transparency_vals = np.array(t_bvals, dtype=float)
            # Sort frames so rank-1 = lowest RMSD (most opaque)
            t_order = np.argsort(transparency_vals)
        else:
            # weight order: top_indices already descending by weight → rank-1 = highest weight
            t_order = np.arange(len(top_indices))

        # Apply the ordering
        top_indices = top_indices[t_order]
        b_values = [b_values[i] for i in t_order]

        spectrum = self.cfg.render.spectrum_colours
        spec_min, spec_max = self.cfg.render.spectrum_range

        group_members: list = []
        for rank, (idx, bval) in enumerate(zip(top_indices, b_values), start=1):
            state = int(idx) + 1
            name = f"{self.obj_name}_top_{rank:02d}"
            cmd.delete(name)
            cmd.create(name, self.obj_name, state, 1)

            try:
                cmd.alter(name, f"b={float(bval)!r}")
            except Exception:
                pass

            # Transparency linearly interpolated from most-opaque (rank 1) to most-transparent (rank N)
            t_min, t_max = self.cfg.top_n.transparency_range
            transparency = t_min + (t_max - t_min) * (rank - 1) / max(n - 1, 1)
            cmd.set("cartoon_transparency", transparency, name)
            cmd.set("cartoon_putty_transform", self.cfg.render.putty_transform, name)
            lo, hi = self.cfg.render.putty_range
            cmd.set("cartoon_putty_scale_min", lo, name)
            cmd.set("cartoon_putty_scale_max", hi, name)
            cmd.show("cartoon", name)
            cmd.cartoon("putty", name)
            cmd.spectrum("b", spectrum, name, minimum=spec_min, maximum=spec_max)
            cmd.enable(name)
            group_members.append(name)

        if group_members:
            self.top_group = f"{self.obj_name}_top_{n}_group"
            cmd.group(self.top_group, " ".join(group_members))
            logger.info("Created group: %s", self.top_group)

    def _top_n_by_weight(
        self, n: int, n_states: int
    ) -> tuple[np.ndarray, list, float, float]:
        """Select top-N frames by weight (descending). B-value = log-weight."""
        if len(self.weights) == 0:
            logger.warning("No weights available; using uniform weights")
            w = np.ones(n_states)
        else:
            w = self.weights.copy()
            if len(w) != n_states:
                raise ValueError(
                    f"Weights length mismatch: weights has {len(w)} entries, "
                    f"but trajectory has {n_states} states. Please ensure "
                    "the weights file corresponds to the loaded trajectory."
                )

        n = min(n, n_states)
        top_idx = np.argsort(w)[::-1][:n]

        safe_w = np.maximum(w[top_idx], EPSILON)
        bvals = np.log(safe_w * len(w) * 1e6)
        b_min = float(np.min(bvals))
        b_max = float(np.max(bvals))
        if abs(b_max - b_min) < 0.1:
            b_max = b_min + 1.0

        # Encode full weight range into background ensemble
        set_bfactors_per_state(self.obj_name, w)
        apply_bfactor_spectrum(self.obj_name, self.cfg.render)

        logger.info(
            "Top N weight stats → min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            float(np.min(w)), float(np.max(w)), float(np.mean(w)), float(np.std(w)),
        )
        
        logger.info(
            "B-val stats → min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
            float(np.min(bvals)), float(np.max(bvals)), float(np.mean(bvals)), float(np.std(bvals)),
        )

        return top_idx, list(bvals), b_min, b_max

    def _bvals_by_rmsd(
        self, top_indices: np.ndarray
    ) -> tuple[list, float, float]:
        """Compute RMSD B-values for the already-selected top-N frames.

        Only the chosen frames are evaluated against the first reference, so
        this is fast even for large ensembles.
        """
        first_ref_obj = next(iter(self.ref_objects.values()), None) if self.ref_objects else None
        if first_ref_obj is None:
            logger.warning(
                "No reference object available for RMSD colouring; "
                "falling back to weight B-values"
            )
            # Return uniform bvals so caller can still proceed
            n = len(top_indices)
            bvals = [1.0] * n
            return bvals, 0.0, 1.0

        align_atoms = self.cfg.general.align_atoms
        rmsds: list[float] = []

        logger.info(
            "Computing RMSD to %s for %d selected frames …",
            first_ref_obj, len(top_indices),
        )
        for idx in _tqdm(top_indices, desc="RMSD colouring", unit="frame"):
            state = int(idx) + 1
            temp = f"__rmsd_colour_{state}_{os.getpid()}__"
            cmd.create(temp, self.obj_name, state, 1)
            rms = compute_rmsd_to_reference(temp, first_ref_obj, align_atoms)
            cmd.delete(temp)
            rmsds.append(rms)

        rmsds_arr = np.array(rmsds)
        finite = rmsds_arr[np.isfinite(rmsds_arr)]
        b_min = float(np.min(finite)) if len(finite) > 0 else 0.0
        b_max = float(np.max(finite)) if len(finite) > 0 else 1.0
        if abs(b_max - b_min) < EPSILON:
            b_max = b_min + 1.0

        logger.info(
            "RMSD colouring stats → mean=%.3f median=%.3f min=%.3f max=%.3f",
            float(np.mean(finite)) if len(finite) > 0 else float("nan"),
            float(np.median(finite)) if len(finite) > 0 else float("nan"),
            b_min,
            b_max,
        )
        return list(rmsds), b_min, b_max

    def _finalize_scene(self) -> None:
        logger.info("-" * 40)
        logger.info("Finalizing scene …")

        # Background ensemble — all states, low transparency
        cmd.set("all_states", 0)
        cmd.show("cartoon", self.obj_name)
        cmd.cartoon("putty", self.obj_name)
        cmd.set("cartoon_transparency", self.cfg.render.trajectory_transparency, self.obj_name)

        # Top-N group is already enabled and styled
        # Reference structures are already enabled and styled in _load_references

        apply_render_settings(self.cfg.render)
        save_output(
            self.cfg.output,
            f"Top_{self.cfg.top_n.top_n}_colour_{self.cfg.top_n.colour_metric}",
            self.cfg.general.working_dir,
            self.cfg.render,
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _apply_reference_color(obj_name: str, color_str: str) -> None:
    """Apply a colour string to a PyMOL object.

    Handles PyMOL named colours ("white", "grey80") and comma-separated RGB
    triplets ("0.95,0.95,0.95").
    """
    parts = [v.strip() for v in color_str.split(",")]
    if len(parts) == 3:
        try:
            rgb = [float(v) for v in parts]
            col_name = f"_col_{obj_name}"
            cmd.set_color(col_name, rgb)
            cmd.color(col_name, obj_name)
            return
        except ValueError:
            pass
    cmd.color(color_str, obj_name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_argparser("Top-N ensemble frame visualisation")
    args, _ = parser.parse_known_args()

    config: UnifiedConfig = load_config(args.config_file) if args.config_file else UnifiedConfig()
    config = merge_config_with_args(config, args)

    try:
        TopNVisualizer(config).run()
    except Exception as exc:
        logger.error("Script failed: %s", exc, exc_info=True)
