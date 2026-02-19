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


def find_most_recent_dir(base_path: Path | str, prefix: str) -> Path | None:
    """Find the most recent directory matching a prefix pattern.

    Useful for finding timestamped optimization result directories created
    dynamically by fitting scripts (e.g., ``_optimise_test_SIGMA_500__20260217_175858``).

    Parameters
    ----------
    base_path:
        Directory to search in.
    prefix:
        Prefix pattern to match (e.g., ``"_optimise_test_SIGMA_500__"``).

    Returns
    -------
    Path to the most recent matching directory (by modification time),
    or ``None`` if no matches found.

    Examples
    --------
    >>> base = Path("jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT")
    >>> find_most_recent_dir(base, "_optimise_test_SIGMA_500__")
    Path('.../fitting/jaxENT/_optimise_test_SIGMA_500__20260217_175858')
    """
    base_path = Path(base_path)
    if not base_path.exists():
        return None

    matching_dirs = [
        d for d in base_path.iterdir()
        if d.is_dir() and d.name.startswith(prefix)
    ]

    if not matching_dirs:
        return None

    # Sort by modification time, most recent first
    matching_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return matching_dirs[0]


def derive_processed_output_dir(results_dir: Path | str) -> Path:
    """Return the processed-output directory for a given results directory.

    Convention: sibling of *results_dir* named ``_processed_<results_dir_name>``.

    Example
    -------
    >>> derive_processed_output_dir("fitting/jaxENT/_optimise_test__20260217")
    Path('fitting/jaxENT/_processed__optimise_test__20260217')
    """
    results_dir = Path(results_dir)
    return results_dir.parent / ("_processed_" + results_dir.name)


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
