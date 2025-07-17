import zipfile
from pathlib import Path


def get_inst_path(base_jaxent_path: Path) -> Path:
    """
    Ensures the 'inst' directory exists and returns its path.
    If 'inst' does not exist, it unzips 'inst.zip'.

    Args:
        base_jaxent_path: The base path to the JAX-ENT project directory.

    Returns:
        The absolute path to the 'inst' directory.
    Raises:
        ValueError: If 'JAX-ENT' parent directory is not found in the path.
    """
    # Find 'JAX-ENT' parent directory in the path
    parts = base_jaxent_path.resolve().parts
    try:
        idx = parts.index("JAX-ENT")
    except ValueError:
        raise ValueError("'JAX-ENT' parent directory not found in the provided path.")
    jaxent_root = Path(*parts[: idx + 1])

    inst_dir = jaxent_root / "jaxent" / "tests" / "inst"
    inst_zip = jaxent_root / "jaxent" / "tests" / "inst.zip"

    if not inst_dir.exists():
        print(f"'{inst_dir}' not found. Unzipping '{inst_zip}'...")
        with zipfile.ZipFile(inst_zip, "r") as zip_ref:
            zip_ref.extractall(inst_dir.parent)  # Extract to jaxent/tests/
        print("Unzipping complete.")
    return inst_dir
