"""
Unified configuration for JAX-ENT PyMOL visualisation scripts.

No PyMOL imports here — this module is importable outside of PyMOL for testing.

YAML structure mirrors the dataclass hierarchy:
  render:       RenderConfig fields
  general:      GeneralConfig fields
  output:       OutputConfig fields
  rmsd:         RMSDScriptConfig fields
  top_n:        TopNScriptConfig fields
  project_logpfs: ProjectLogPFsScriptConfig fields
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    """PyMOL render / display settings shared across all scripts."""
    spectrum_colours: str = "blue_white_red"
    spectrum_range: tuple = (0.0, 5.0)          # (min, max)
    putty_transform: int = 7
    putty_range: tuple = (0.4, 4.0)             # (scale_min, scale_max)
    reference_transparency: float = 0.6
    trajectory_transparency: float = 0.8
    other_transparency: float = 0.5
    transparency_mode: int = 1
    orthoscopic_view: bool = True
    antialias: int = 2
    ray_trace_mode: int = 3
    ray_transparency_oblique: bool = False
    ray_trace_disco_factor: float = 0.5
    ray_trace_gain: float = 1.0
    ray_trace_on_save: bool = True              # ray-trace when saving PNG
    view: Optional[tuple] = None               # 16-value matrix or None → cmd.zoom()


@dataclass
class GeneralConfig:
    """General input / alignment options shared across all scripts."""
    references: list = field(default_factory=list)          # paths to reference PDBs
    reference_labels: list = field(default_factory=list)    # human-readable labels
    reference_colors: list = field(default_factory=list)    # per-reference colours
    trajectory_label: str = "trajectory"                    # PyMOL object name
    align_atoms: str = "name CA"
    align_selection: Optional[str] = None                   # residue selection; None = whole chain
    working_dir: Optional[str] = None


@dataclass
class OutputConfig:
    """Output saving options."""
    save_png: bool = False
    save_session: bool = False
    output_prefix: str = "output"


@dataclass
class RMSDScriptConfig:
    """Options specific to RMSD_by_res.py."""
    trajectory: Optional[str] = None  # "traj.xtc,topology.pdb" or "multiframe.pdb"


@dataclass
class TopNScriptConfig:
    """Options specific to Top_N_structures.py."""
    trajectory: Optional[str] = None
    weights: Optional[str] = None           # .npz path, shape (n_replicates, n_frames)
    colour_metric: str = "weight"           # B-factor colouring: "weight" | "RMSD"
    transparency_metric: str = "weight"     # transparency ordering: "weight" | "RMSD"
    top_n: int = 10
    transparency_range: tuple = (0.1, 0.85) # (most_opaque, most_transparent) for rank-1 … rank-N


@dataclass
class ProjectLogPFsScriptConfig:
    """Options specific to Project_logPFs.py."""
    reference_data: Optional[str] = None    # .dat cols: residue pf
    target_data: Optional[str] = None       # .npy shape (n_replicates, n_residues)
    target_topology: Optional[str] = None   # full-sequence PDB
    metric: str = "protection_factor"
    # metric: protection_factor | uncertainty_sd | uncertainty_rsd |
    #         difference_signed | difference_absolute


@dataclass
class UnifiedConfig:
    """Top-level config bundling all sub-configs."""
    render: RenderConfig = field(default_factory=RenderConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    rmsd: RMSDScriptConfig = field(default_factory=RMSDScriptConfig)
    top_n: TopNScriptConfig = field(default_factory=TopNScriptConfig)
    project_logpfs: ProjectLogPFsScriptConfig = field(default_factory=ProjectLogPFsScriptConfig)

    def resolve_paths(self) -> None:
        """Resolve relative paths against working_dir (or cwd if not set)."""
        base = self.general.working_dir or os.getcwd()

        def _abs(p: Optional[str]) -> Optional[str]:
            if p and not os.path.isabs(p):
                return os.path.join(base, p)
            return p

        # GeneralConfig paths
        self.general.references = [_abs(r) for r in self.general.references]
        self.general.working_dir = os.path.abspath(base)

        # Per-script paths
        if self.rmsd.trajectory:
            parts = self.rmsd.trajectory.split(",")
            self.rmsd.trajectory = ",".join(_abs(p.strip()) or p.strip() for p in parts)

        if self.top_n.trajectory:
            parts = self.top_n.trajectory.split(",")
            self.top_n.trajectory = ",".join(_abs(p.strip()) or p.strip() for p in parts)
        self.top_n.weights = _abs(self.top_n.weights)

        self.project_logpfs.reference_data = _abs(self.project_logpfs.reference_data)
        self.project_logpfs.target_data = _abs(self.project_logpfs.target_data)
        self.project_logpfs.target_topology = _abs(self.project_logpfs.target_topology)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_comma_separated(value) -> list:
    """Normalise "a,b,c" or ["a","b","c"] to list[str]."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_range(value) -> tuple:
    """Normalise [min, max] list or "min,max" string to (float, float)."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",")]
        if len(parts) == 2:
            return (float(parts[0]), float(parts[1]))
    raise ValueError(f"Cannot parse range from: {value!r}  (expected [min, max] or 'min,max')")


def parse_view(value) -> Optional[tuple]:
    """Normalise a list/str of 16-18 floats to a tuple, or return None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return tuple(float(v) for v in value)
    if isinstance(value, str):
        parts = [v.strip() for v in value.replace(",", " ").split()]
        return tuple(float(v) for v in parts)
    return None


def parse_trajectory_string(value: str) -> tuple:
    """Split "traj.xtc,topology.pdb" → (topology_path, traj_path).

    The config format is trajectory-first ("traj.xtc,topology.pdb") to match
    the README, but PyMOL requires topology to be loaded first.  This function
    therefore returns (topology, trajectory) — i.e. the two parts swapped —
    so callers can pass directly to load_trajectory(obj, topology, traj).

    Single path → (path, None) — the file is itself the topology (multiframe PDB).
    """
    if not value:
        return ("", None)
    parts = [p.strip() for p in value.split(",")]
    if len(parts) >= 2:
        return (parts[1], parts[0])   # swap: topology second in string, first for PyMOL
    return (parts[0], None)


def parse_colour_string(value: str) -> str:
    """Return a colour string ready for PyMOL cmd.spectrum() / cmd.color()."""
    return str(value).strip()


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _apply_render(data: dict, cfg: RenderConfig) -> None:
    for key, val in data.items():
        if key == "spectrum_range":
            cfg.spectrum_range = parse_range(val)
        elif key == "putty_range":
            cfg.putty_range = parse_range(val)
        elif key == "view":
            cfg.view = parse_view(val)
        elif key == "orthoscopic_view":
            cfg.orthoscopic_view = bool(val)
        elif key == "ray_transparency_oblique":
            cfg.ray_transparency_oblique = bool(val)
        elif key == "ray_trace_on_save":
            cfg.ray_trace_on_save = bool(val)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)


def _apply_general(data: dict, cfg: GeneralConfig) -> None:
    for key, val in data.items():
        if key in ("references", "reference_labels", "reference_colors"):
            setattr(cfg, key, parse_comma_separated(val))
        elif hasattr(cfg, key):
            setattr(cfg, key, val)


def _apply_output(data: dict, cfg: OutputConfig) -> None:
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)


def _apply_rmsd(data: dict, cfg: RMSDScriptConfig) -> None:
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)


def _apply_top_n(data: dict, cfg: TopNScriptConfig) -> None:
    for key, val in data.items():
        if key == "top_n":
            cfg.top_n = int(val)
        elif key == "metric":
            # Backwards-compat: old YAML configs used `metric` for colouring choice
            cfg.colour_metric = str(val)
        elif key == "transparency_range":
            cfg.transparency_range = parse_range(val)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)


def _apply_project_logpfs(data: dict, cfg: ProjectLogPFsScriptConfig) -> None:
    for key, val in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)


def load_config(yaml_path: str) -> UnifiedConfig:
    """Load a YAML file into a UnifiedConfig.

    Missing sections use dataclass defaults. The YAML path is resolved to an
    absolute path before opening to handle PyMOL's potentially unusual cwd.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml")

    yaml_path = os.path.abspath(yaml_path)
    with open(yaml_path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = UnifiedConfig()

    if "render" in raw:
        _apply_render(raw["render"], cfg.render)
    if "general" in raw:
        _apply_general(raw["general"], cfg.general)
    if "output" in raw:
        _apply_output(raw["output"], cfg.output)
    if "rmsd" in raw:
        _apply_rmsd(raw["rmsd"], cfg.rmsd)
    if "top_n" in raw:
        _apply_top_n(raw["top_n"], cfg.top_n)
    if "project_logpfs" in raw:
        _apply_project_logpfs(raw["project_logpfs"], cfg.project_logpfs)

    return cfg


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_argparser(description: str = "") -> argparse.ArgumentParser:
    """Build an ArgumentParser covering all UnifiedConfig fields.

    All arguments are named (--flag) — no positional arguments.  This avoids
    PyMOL treating bare positional values as structure files to load.

    Usage inside PyMOL:
        pymol -r script.py -- --config config.yaml [--flag value ...]

    All optional arguments default to None so merge_config_with_args() can
    distinguish explicit CLI overrides from unset flags.  Use
    parse_known_args() so that PyMOL's own flags (-r, -d, etc.) are silently
    ignored.
    """
    p = argparse.ArgumentParser(description=description)

    # Config file
    p.add_argument("--config", default=None, dest="config_file",
                   help="Path to YAML config file")

    # --- Render ---
    r = p.add_argument_group("render options")
    r.add_argument("--spectrum_colours", default=None)
    r.add_argument("--spectrum_range", default=None,
                   help="min,max  e.g. '0,5'")
    r.add_argument("--putty_transform", default=None, type=int)
    r.add_argument("--putty_range", default=None,
                   help="scale_min,scale_max  e.g. '0.4,4.0'")
    r.add_argument("--reference_transparency", default=None, type=float)
    r.add_argument("--trajectory_transparency", default=None, type=float)
    r.add_argument("--other_transparency", default=None, type=float)
    r.add_argument("--transparency_mode", default=None, type=int)
    r.add_argument("--orthoscopic_view", default=None, type=lambda v: v.lower() != "false")
    r.add_argument("--antialias", default=None, type=int)
    r.add_argument("--ray_trace_mode", default=None, type=int)
    r.add_argument("--ray_transparency_oblique", default=None,
                   type=lambda v: v.lower() != "false")
    r.add_argument("--ray_trace_disco_factor", default=None, type=float)
    r.add_argument("--ray_trace_gain", default=None, type=float)
    r.add_argument("--ray_trace_on_save", default=None,
                   type=lambda v: v.lower() != "false")
    r.add_argument("--view", default=None,
                   help="16-18 comma/space-separated floats for cmd.set_view()")

    # --- General ---
    g = p.add_argument_group("general options")
    g.add_argument("--references", default=None,
                   help="Comma-separated paths to reference PDB files")
    g.add_argument("--reference_labels", default=None,
                   help="Comma-separated labels for reference structures")
    g.add_argument("--reference_colors", default=None,
                   help="Comma-separated colours for reference structures")
    g.add_argument("--trajectory_label", default=None)
    g.add_argument("--align_atoms", default=None)
    g.add_argument("--align_selection", default=None)
    g.add_argument("--working_dir", default=None)

    # --- Output ---
    o = p.add_argument_group("output options")
    o.add_argument("--save_png", action="store_true", default=None)
    o.add_argument("--save_session", action="store_true", default=None)
    o.add_argument("--output_prefix", default=None)

    # --- RMSD script ---
    rs = p.add_argument_group("RMSD_by_res options")
    rs.add_argument("--trajectory", default=None,
                    help="'traj.xtc,topology.pdb' or 'multiframe.pdb'")

    # --- Top-N script ---
    tn = p.add_argument_group("Top_N_structures options")
    tn.add_argument("--weights", default=None,
                    help="Path to .npz weights file, shape (n_replicates, n_frames)")
    tn.add_argument("--colour_metric", default=None,
                    help="Colouring metric: 'weight' (default) or 'RMSD'")
    tn.add_argument("--transparency_metric", default=None,
                    help="Transparency ordering metric: 'weight' (default) or 'RMSD'")
    tn.add_argument("--top_n", default=None, type=int)
    tn.add_argument("--transparency_range", default=None,
                    help="min,max transparency for rank-1…rank-N, e.g. '0.1,0.85'")

    # --- Project_logPFs script ---
    pf = p.add_argument_group("Project_logPFs options")
    pf.add_argument("--reference_data", default=None,
                    help="Path to reference protection factors .dat file")
    pf.add_argument("--target_data", default=None,
                    help="Path to .npy protection factor array (n_replicates, n_residues)")
    pf.add_argument("--target_topology", default=None,
                    help="Path to full-sequence topology PDB")
    pf.add_argument("--pf_metric", default=None, dest="pf_metric",
                    help="protection_factor | uncertainty_sd | uncertainty_rsd | "
                         "difference_signed | difference_absolute")

    return p


def merge_config_with_args(config: UnifiedConfig, args: argparse.Namespace) -> UnifiedConfig:
    """Overwrite config fields with any explicitly set CLI arguments.

    Only fields where args.X is not None take effect, preserving YAML / defaults
    for everything the user did not explicitly supply on the command line.
    """
    a = vars(args)

    # Render
    render_map = {
        "spectrum_colours": ("render", "spectrum_colours"),
        "putty_transform": ("render", "putty_transform"),
        "reference_transparency": ("render", "reference_transparency"),
        "trajectory_transparency": ("render", "trajectory_transparency"),
        "other_transparency": ("render", "other_transparency"),
        "transparency_mode": ("render", "transparency_mode"),
        "orthoscopic_view": ("render", "orthoscopic_view"),
        "antialias": ("render", "antialias"),
        "ray_trace_mode": ("render", "ray_trace_mode"),
        "ray_transparency_oblique": ("render", "ray_transparency_oblique"),
        "ray_trace_disco_factor": ("render", "ray_trace_disco_factor"),
        "ray_trace_gain": ("render", "ray_trace_gain"),
        "ray_trace_on_save": ("render", "ray_trace_on_save"),
    }
    for arg_key, (sub, attr) in render_map.items():
        val = a.get(arg_key)
        if val is not None:
            setattr(getattr(config, sub), attr, val)

    if a.get("spectrum_range") is not None:
        config.render.spectrum_range = parse_range(a["spectrum_range"])
    if a.get("putty_range") is not None:
        config.render.putty_range = parse_range(a["putty_range"])
    if a.get("view") is not None:
        config.render.view = parse_view(a["view"])

    # General
    for arg_key, attr in [
        ("trajectory_label", "trajectory_label"),
        ("align_atoms", "align_atoms"),
        ("align_selection", "align_selection"),
        ("working_dir", "working_dir"),
    ]:
        val = a.get(arg_key)
        if val is not None:
            setattr(config.general, attr, val)

    for list_arg in ("references", "reference_labels", "reference_colors"):
        val = a.get(list_arg)
        if val is not None:
            setattr(config.general, list_arg, parse_comma_separated(val))

    # Output
    for arg_key, attr in [("output_prefix", "output_prefix")]:
        val = a.get(arg_key)
        if val is not None:
            setattr(config.output, attr, val)
    if a.get("save_png"):
        config.output.save_png = True
    if a.get("save_session"):
        config.output.save_session = True

    # RMSD script — trajectory is shared with top_n
    if a.get("trajectory") is not None:
        config.rmsd.trajectory = a["trajectory"]
        config.top_n.trajectory = a["trajectory"]

    # Top-N script
    for arg_key, attr in [
        ("weights", "weights"),
        ("colour_metric", "colour_metric"),
        ("transparency_metric", "transparency_metric"),
    ]:
        val = a.get(arg_key)
        if val is not None:
            setattr(config.top_n, attr, val)
    if a.get("top_n") is not None:
        config.top_n.top_n = a["top_n"]
    if a.get("transparency_range") is not None:
        config.top_n.transparency_range = parse_range(a["transparency_range"])

    # Project_logPFs script
    for arg_key, attr in [
        ("reference_data", "reference_data"),
        ("target_data", "target_data"),
        ("target_topology", "target_topology"),
    ]:
        val = a.get(arg_key)
        if val is not None:
            setattr(config.project_logpfs, attr, val)
    if a.get("pf_metric") is not None:
        config.project_logpfs.metric = a["pf_metric"]

    return config
