"""
Centralized path utilities for JAX-ENT example scripts.

Replaces fragile ``os.path.join(os.path.dirname(__file__), "../../data/...")``
patterns found across all example scripts with deterministic helpers.
"""

from pathlib import Path


def get_examples_root() -> Path:
    """Return the root ``jaxent/examples/`` directory."""
    return Path(__file__).resolve().parent.parent


def resolve_example_path(example_name: str, subdir: str = "") -> Path:
    """Resolve *subdir* relative to an example directory.

    Parameters
    ----------
    example_name:
        Directory name under ``jaxent/examples/`` (e.g. ``"2_CrossValidation"``).
    subdir:
        Optional sub-path within the example directory.
    """
    path = get_examples_root() / example_name
    if subdir:
        path = path / subdir
    return path


def ensure_output_dir(path: Path | str) -> Path:
    """Create *path* (and parents) if it doesn't exist; return the ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_script_paths(
    args,
    script_dir: Path | str,
    *,
    keys: tuple[str, ...] = (
        "results_dir",
        "output_dir",
        "features_dir",
        "datasplit_dir",
        "clustering_dir",
    ),
) -> dict[str, str]:
    """Resolve relative/absolute paths based on ``--absolute-paths`` flag.

    Parameters
    ----------
    args:
        Parsed ``argparse.Namespace`` (should contain the *keys* as attributes
        and an ``absolute_paths`` boolean).
    script_dir:
        Directory of the calling script — relative paths are resolved from here.
    keys:
        Attribute names on *args* to process.

    Returns
    -------
    dict mapping each key to its resolved absolute path string.
    """
    script_dir = Path(script_dir)
    absolute = getattr(args, "absolute_paths", False)
    resolved: dict[str, str] = {}
    for key in keys:
        val = getattr(args, key.replace("-", "_"), None)
        if val is None:
            continue
        if absolute:
            resolved[key] = str(val)
        else:
            resolved[key] = str((script_dir / val).resolve())
    return resolved
