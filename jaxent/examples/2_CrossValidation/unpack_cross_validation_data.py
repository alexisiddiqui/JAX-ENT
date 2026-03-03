import os
import tarfile
import argparse
from pathlib import Path
import gdown

def download_data(output_path: Path):
    """
    Downloads the data package from Google Drive using gdown.
    https://drive.google.com/file/d/18FGXZNYnPBTqAEwUvSPRqGILHJIC82du/view?usp=drive_link
    """
    if not output_path.exists():
        file_id = "18FGXZNYnPBTqAEwUvSPRqGILHJIC82du"
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading data package to {output_path}...")
        gdown.download(url, str(output_path), quiet=False)


def unpack_data(package_path: Path, target_dir: Path):
    """
    Unpacks the tarball into the target directory (the example root).
    """
    if not package_path.exists():
        print(f"Error: Package not found at {package_path}")
        return False

    print(f"Unpacking {package_path.name} to {target_dir}...")
    
    with tarfile.open(package_path, "r:gz") as tar:
        tar.extractall(path=target_dir)
    
    print("Unpacking complete.")
    return True

def verify_files(target_dir: Path):
    """
    Verifies that all expected files exist in the target directory structure.
    """
    expected_files = [
        "data/MoPrP_max_plddt_4334.pdb",
        "data/_MoPrP/2L1H_crop.pdb",
        "data/_MoPrP/2L39_crop.pdb",
        "data/_cluster_MoPrP/clusters/all_clusters.xtc",
        "data/_cluster_MoPrP_filtered/clusters/all_clusters.xtc",
        "data/_MoPrP/_output/MoPrP_dfrac.dat",
        "data/_MoPrP/_output/MoPrP_segments.txt",
        "data/_MoPrP/_output/out__train_MoPrP_af_clean_1Intrinsic_rates.dat",
        "data/_MoPrP/key_residues.json",
        "analysis/MoPrP_unfolding_spec.json",
        "analysis/MoPrP_rules_spec.json"
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
        print("Verification successful: All original files for Example 2 are present.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download and unpack CrossValidation data.")
    parser.add_argument("--package", type=str, default="cross_validation_data.tar.gz", help="Path to the tar.gz package")
    args = parser.parse_args()

    # The script is in the example root, so target_dir is the current directory
    example_root = Path(__file__).parent.resolve()
    package_path = example_root / args.package

    # 1. Download (placeholder)
    download_data(package_path)

    # 2. Unpack
    if unpack_data(package_path, example_root):
        # 3. Verification
        verify_files(example_root)

if __name__ == "__main__":
    main()
