import os
import tarfile
import argparse
from pathlib import Path

import gdown


def download_data(output_path: Path):
    """
    Downloads the data package from Google Drive using gdown.
    """
    if not output_path.exists():
        file_id = "1mnRQ3XZM3_t02spvaTLsgbmJrnEQe-wE"
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading data package to {output_path}...")
        gdown.download(url, str(output_path), quiet=False)

def unpack_data(package_path: Path, target_dir: Path):
    """
    Unpacks the tarball into the target directory.
    """
    if not package_path.exists():
        print(f"Error: Package not found at {package_path}")
        return False

    print(f"Unpacking {package_path.name} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(package_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    
    print("Unpacking complete.")
    return True

def verify_files(target_dir: Path):
    """
    Verifies that all expected files exist in the target directory.
    """
    expected_files = [
        "_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_closed_state.pdb",
        "_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb",
        "_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_filtered_sliced.xtc",
        "_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_initial_sliced.xtc",
        "_Bradshaw/Reproducibility_pack_v2/data/artificial_HDX_data/mixed_60-40_artificial_expt_resfracs.dat"
    ]
    
    missing = []
    for f in expected_files:
        file_path = target_dir / f
        if not file_path.exists():
            missing.append(f)
    
    if missing:
        print("Warning: The following files are missing after extraction:")
        for m in missing:
            print(f"  - {m}")
        return False
    else:
        print("Verification successful: All original files are present.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download and unpack IsoValidation data.")
    parser.add_argument("--package", type=str, default="iso_validation_data.tar.gz", help="Path to the tar.gz package")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    package_path = script_dir / args.package
    target_data_dir = script_dir / "data"

    # 1. Download (placeholder)
    download_data(package_path)

    # 2. Unpack
    if unpack_data(package_path, target_data_dir):
        # 3. Verify
        verify_files(target_data_dir)

if __name__ == "__main__":
    main()
