"""
Shared PyMOL utilities for JAX-ENT visualisation scripts.

Geometry functions (kabsch_alignment, build_align_selection) are pure numpy
and can be tested outside of PyMOL.  All functions that touch cmd / stored
require an active PyMOL session.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from config import OutputConfig, RenderConfig

logger = logging.getLogger(__name__)

EPSILON = 1e-6

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, *args, **kwargs):  # type: ignore[misc]
        return iterable


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def is_running_in_pymol() -> bool:
    """Return True when the PyMOL cmd API is available in this session."""
    try:
        from pymol import cmd
        cmd.get_names()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Geometry — pure numpy, no cmd
# ---------------------------------------------------------------------------

def kabsch_alignment(
    mobile: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kabsch rigid-body alignment of mobile onto reference.

    Returns (R, t) where:
      aligned = mobile @ R.T + t

    If the atom counts differ, both arrays are truncated to the shorter length.
    """
    if len(mobile) != len(reference):
        n = min(len(mobile), len(reference))
        logger.debug("kabsch_alignment: length mismatch, truncating to %d atoms", n)
        mobile = mobile[:n]
        reference = reference[:n]

    mobile_c = mobile - mobile.mean(axis=0)
    reference_c = reference - reference.mean(axis=0)

    H = mobile_c.T @ reference_c
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Guard against reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = reference.mean(axis=0) - mobile.mean(axis=0) @ R.T
    return R, t


def build_align_selection(
    obj_name: str,
    align_atoms: str,
    align_selection: Optional[str],
) -> str:
    """Compose a PyMOL selection string for alignment.

    Examples:
      build_align_selection("traj", "name CA", "resid 79-93")
      → "traj and (resid 79-93) and name CA"

      build_align_selection("traj", "name CA", None)
      → "traj and name CA"
    """
    if align_selection:
        return f"{obj_name} and ({align_selection}) and {align_atoms}"
    return f"{obj_name} and {align_atoms}"


# ---------------------------------------------------------------------------
# Trajectory / ensemble loading
# ---------------------------------------------------------------------------

def load_trajectory(
    obj_name: str,
    topology_file: str,
    traj_file: Optional[str] = None,
) -> int:
    """Load a topology PDB and optional trajectory into a PyMOL object.

    topology_file is loaded first (cmd.load); traj_file is appended after
    (cmd.load_traj).  parse_trajectory_string() already swaps the user-facing
    "traj,topology" order so callers can pass its result directly here.

    Returns the total number of states loaded.
    """
    from pymol import cmd

    cmd.load(topology_file, obj_name)
    logger.info("Loaded topology: %s → %s", os.path.basename(topology_file), obj_name)

    if traj_file:
        cmd.load_traj(traj_file, obj_name)
        logger.info("Loaded trajectory: %s", os.path.basename(traj_file))

    n = cmd.count_states(obj_name)
    logger.info("Total states loaded: %d", n)
    return n


def get_reference_target_coords(
    ref_pdb: str,
    align_atoms: str,
    align_selection: Optional[str],
) -> Optional[np.ndarray]:
    """Return CA (or custom atom) coordinates from a reference PDB for alignment.

    Multi-state references are averaged into a single consensus target.
    A temporary PyMOL object is created and deleted internally.
    """
    from pymol import cmd

    if not ref_pdb or not os.path.exists(ref_pdb):
        logger.warning("Reference PDB not found: %s", ref_pdb)
        return None

    temp_obj = "__temp_ref_target__"
    try:
        cmd.load(ref_pdb, temp_obj)
        n_states = cmd.count_states(temp_obj)
        sel = build_align_selection(temp_obj, align_atoms, align_selection)

        if n_states > 1:
            logger.info(
                "Reference has %d states — averaging for alignment target", n_states
            )
            sum_c: Optional[np.ndarray] = None
            count = 0
            for i in range(1, n_states + 1):
                c = cmd.get_coords(sel, state=i)
                if c is not None and len(c) > 0:
                    sum_c = c if sum_c is None else sum_c + c
                    count += 1
            target = sum_c / count if (sum_c is not None and count > 0) else None
        else:
            target = cmd.get_coords(sel, state=1)

        cmd.delete(temp_obj)

        if target is not None:
            logger.info("Alignment target: %d atoms", len(target))
        else:
            logger.warning("Could not extract alignment target from %s", ref_pdb)
        return target

    except Exception as exc:
        logger.error("Failed to build alignment target from %s: %s", ref_pdb, exc)
        try:
            cmd.delete(temp_obj)
        except Exception:
            pass
        return None


def align_pymol_object_to_coords(
    obj_name: str,
    target_coords: np.ndarray,
    align_atoms: str,
    align_selection: Optional[str],
) -> int:
    """Kabsch-align every state of a PyMOL object to target_coords.

    Returns the number of states successfully aligned.
    """
    from pymol import cmd

    if target_coords is None:
        return 0

    n_states = cmd.count_states(obj_name)
    sel_align = build_align_selection(obj_name, align_atoms, align_selection)

    test = cmd.get_coords(sel_align, state=1)
    if test is None or len(test) != len(target_coords):
        logger.warning(
            "Skipping alignment for %s: atom count mismatch "
            "(%d vs %d)",
            obj_name,
            0 if test is None else len(test),
            len(target_coords),
        )
        return 0

    aligned = 0
    for state in _tqdm(range(1, n_states + 1), desc=f"Aligning {obj_name}", unit="state"):
        try:
            mobile = cmd.get_coords(sel_align, state=state)
            R, t = kabsch_alignment(mobile, target_coords)
            all_coords = cmd.get_coords(obj_name, state=state)
            cmd.load_coords(all_coords @ R.T + t, obj_name, state=state)
            aligned += 1
        except Exception as exc:
            logger.debug("State %d alignment failed for %s: %s", state, obj_name, exc)

    logger.info("Aligned %d/%d states for %s", aligned, n_states, obj_name)
    return aligned


# ---------------------------------------------------------------------------
# B-factor helpers
# ---------------------------------------------------------------------------

def set_bfactors_from_dict(
    obj_name: str,
    value_dict: dict,
    default: float = 0.0,
) -> None:
    """Assign B-factors from a {resi_str: float} dict via PyMOL's stored namespace.

    Applied to ALL atoms in each residue so cartoon putty rendering works correctly.
    """
    from pymol import cmd, stored

    stored.bfactor_dict = value_dict
    cmd.alter(obj_name, "b=0.0")
    cmd.alter(obj_name, f"b=stored.bfactor_dict.get(resi.strip(), {float(default)!r})")
    cmd.rebuild(obj_name)


def set_bfactors_per_state(obj_name: str, values: np.ndarray) -> None:
    """Encode per-state scalar scores into the B-factor channel."""
    from pymol import cmd

    n_states = cmd.count_states(obj_name)
    for i in range(min(len(values), n_states)):
        try:
            cmd.alter_state(i + 1, obj_name, f"b={float(values[i])!r}")
        except Exception:
            pass
    cmd.rebuild(obj_name)


def ensure_nonzero_bfactors(obj_name: str, min_value: float = EPSILON) -> None:
    """Clamp B-factors to a small positive floor for stable spectrum coloring."""
    from pymol import cmd

    try:
        mv = float(min_value)
        cmd.alter(obj_name, f"b={mv!r} if b < {mv!r} else b")
        cmd.rebuild(obj_name)
    except Exception as exc:
        logger.warning("Failed to normalize B-factors for %s: %s", obj_name, exc)


def compute_all_atom_mean(obj_name: str) -> Optional[np.ndarray]:
    """Compute mean all-atom coordinates across all states of a PyMOL object."""
    from pymol import cmd

    n_states = cmd.count_states(obj_name)
    if n_states < 1:
        return None
    if n_states == 1:
        return cmd.get_coords(obj_name, state=1)

    ref_shape = None
    sum_c: Optional[np.ndarray] = None
    count = 0

    for i in range(1, n_states + 1):
        c = cmd.get_coords(obj_name, state=i)
        if c is not None:
            if sum_c is None:
                sum_c = np.zeros_like(c)
                ref_shape = c.shape
            if c.shape == ref_shape:
                sum_c += c
                count += 1

    if sum_c is not None and count > 0:
        return sum_c / count
    return None


def compute_residue_rmsd_from_dict(
    obj_name: str,
    ref_coords_dict: dict,
    atom_type: str = "CA",
) -> dict:
    """Compute per-residue RMSD across the ensemble relative to ref_coords_dict.

    ref_coords_dict: {resi_str: np.array([x, y, z])}
    Returns: {resi_str: float}
    """
    from pymol import cmd

    n_states = cmd.count_states(obj_name)
    model = cmd.get_model(f"{obj_name} and name {atom_type}", state=1)
    residues = [(a.resi, a.resn) for a in model.atom]

    sum_sq: dict = {}
    counts: dict = {}

    for resi, _resn in _tqdm(residues, desc="Computing RMSD per residue", unit="res"):
        resi_key = resi.strip()
        if resi_key not in ref_coords_dict:
            continue
        ref_c = ref_coords_dict[resi_key]
        sel = f"{obj_name} and resi {resi} and name {atom_type}"
        sq = 0.0
        cnt = 0
        for state in range(1, n_states + 1):
            c = cmd.get_coords(sel, state=state)
            if c is not None and len(c) > 0:
                sq += float(np.sum((c[0] - ref_c) ** 2))
                cnt += 1
        if cnt > 0:
            sum_sq[resi_key] = sq
            counts[resi_key] = cnt

    return {k: float(np.sqrt(sum_sq[k] / counts[k])) for k in sum_sq}


def get_reference_coords_dict(
    pdb_or_obj: str,
    is_file: bool = False,
    atom_type: str = "CA",
) -> dict:
    """Return {resi_str: np.array([x,y,z])} reference coordinates.

    If is_file=True: load pdb_or_obj as a PDB, extract coords, delete temp object.
    If is_file=False: compute the average structure of the named PyMOL object.
    """
    from pymol import cmd

    ref_dict: dict = {}

    if is_file:
        temp = "__temp_ref_dict__"
        cmd.load(pdb_or_obj, temp)
        model = cmd.get_model(f"{temp} and name {atom_type}")
        for a in model.atom:
            ref_dict[a.resi.strip()] = np.array(a.coord)
        cmd.delete(temp)
    else:
        n_states = cmd.count_states(pdb_or_obj)
        model = cmd.get_model(f"{pdb_or_obj} and name {atom_type}", state=1)
        sums: dict = {a.resi.strip(): np.zeros(3) for a in model.atom}
        counts: dict = {a.resi.strip(): 0 for a in model.atom}

        for state in range(1, n_states + 1):
            s_model = cmd.get_model(f"{pdb_or_obj} and name {atom_type}", state=state)
            for a in s_model.atom:
                r = a.resi.strip()
                if r in sums:
                    sums[r] += np.array(a.coord)
                    counts[r] += 1

        for resi in sums:
            if counts[resi] > 0:
                ref_dict[resi] = sums[resi] / counts[resi]

    return ref_dict


# ---------------------------------------------------------------------------
# PyMOL display helpers
# ---------------------------------------------------------------------------

def set_background_white() -> None:
    """Set white opaque background for publication-style rendering."""
    from pymol import cmd

    try:
        cmd.bg_color("white")
        cmd.set("opaque_background", 1)
    except Exception:
        pass


def apply_render_settings(render_config) -> None:
    """Apply all RenderConfig fields to the current PyMOL session."""
    from pymol import cmd

    cmd.set("transparency_mode", render_config.transparency_mode)
    cmd.set("orthoscopic", int(render_config.orthoscopic_view))
    cmd.set("antialias", render_config.antialias)
    cmd.set("ray_trace_mode", render_config.ray_trace_mode)
    cmd.set("ray_transparency_oblique", int(render_config.ray_transparency_oblique))
    cmd.set("ray_trace_disco_factor", render_config.ray_trace_disco_factor)
    cmd.set("ray_trace_gain", render_config.ray_trace_gain)
    cmd.set("ray_shadows", 0)

    if render_config.view is not None:
        try:
            cmd.set_view(render_config.view)
        except Exception as exc:
            logger.warning("Failed to apply view matrix: %s", exc)
    else:
        cmd.zoom()


def apply_bfactor_spectrum(obj_name: str, render_config) -> None:
    """Color obj_name by B-factor using render_config spectrum settings."""
    from pymol import cmd

    lo, hi = render_config.spectrum_range
    cmd.spectrum("b", render_config.spectrum_colours, obj_name, minimum=lo, maximum=hi)


def apply_putty_settings(obj_name: str, render_config) -> None:
    """Apply putty cartoon settings from render_config to obj_name."""
    from pymol import cmd

    lo, hi = render_config.putty_range
    spec_min, spec_max = render_config.spectrum_range

    # If spectrum range is inverted (negative scale), swap putty scales
    # so the minimum B-factor gets the maximum thickness.
    if spec_min > spec_max:
        lo, hi = hi, lo

    cmd.set("cartoon_putty_transform", render_config.putty_transform, obj_name)
    cmd.set("cartoon_putty_scale_min", lo, obj_name)
    cmd.set("cartoon_putty_scale_max", hi, obj_name)


def save_output(
    output_config,
    output_name: str,
    working_dir: Optional[str] = None,
    render_config=None,
) -> None:
    """Save PNG and/or .pse session if configured.

    output_name is appended to output_config.output_prefix, e.g.:
      prefix="MoPrP_RMSD", name="scene" → "MoPrP_RMSD_scene.png"

    render_config is optional; if provided, its ray_trace_on_save flag
    controls whether cmd.png() ray-traces the image.
    """
    from pymol import cmd

    base_dir = working_dir or os.getcwd()
    stem = os.path.join(base_dir, f"{output_config.output_prefix}_{output_name}")

    if output_config.save_png:
        png_path = f"{stem}.png"
        try:
            ray = int(render_config.ray_trace_on_save) if render_config is not None else 1
            logger.info("Saving PNG%s: %s", " (ray tracing)" if ray else "", png_path)
            cmd.png(png_path, ray=ray)
        except Exception as exc:
            logger.error("Failed to save PNG: %s", exc)

    if output_config.save_session:
        pse_path = f"{stem}.pse"
        try:
            logger.info("Saving session: %s", pse_path)
            cmd.save(pse_path)
        except Exception as exc:
            logger.error("Failed to save session: %s", exc)
