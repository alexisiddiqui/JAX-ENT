import sys
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.models.config import BV_model_Config
from jaxent.src.custom_types.features import AbstractFeatures


# PDB system containing ALA-GLY-SER peptide and a Tris molecule (TR0)
# positioned within contact thresholds of GLY A 2.
# GLY A 2 N is at [2.0, 0.0, 0.0]
# TR0 B 10 O1 (oxygen acceptor) is at [2.0, 0.0, 2.0]
# TR0 B 10 C1 (heavy atom) is at [2.0, 1.0, 2.0]
PDB_CONTENT = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H   ALA A   1       0.000   0.000   1.000  1.00  0.00           H
ATOM      3  CA  ALA A   1       0.000   1.000   0.000  1.00  0.00           C
ATOM      4  C   ALA A   1       0.000   2.000   0.000  1.00  0.00           C
ATOM      5  O   ALA A   1       0.000   2.000   1.000  1.00  0.00           O
ATOM      6  N   GLY A   2       2.000   0.000   0.000  1.00  0.00           N
ATOM      7  H   GLY A   2       2.000   0.000   1.000  1.00  0.00           H
ATOM      8  CA  GLY A   2       2.000   1.000   0.000  1.00  0.00           C
ATOM      9  C   GLY A   2       2.000   2.000   0.000  1.00  0.00           C
ATOM     10  O   GLY A   2       2.000   2.000   1.000  1.00  0.00           O
ATOM     11  N   SER A   3       4.000   0.000   0.000  1.00  0.00           N
ATOM     12  H   SER A   3       4.000   0.000   1.000  1.00  0.00           H
ATOM     13  CA  SER A   3       4.000   1.000   0.000  1.00  0.00           C
ATOM     14  C   SER A   3       4.000   2.000   0.000  1.00  0.00           C
ATOM     15  O   SER A   3       4.000   2.000   1.000  1.00  0.00           O
ATOM     16  O1  TR0 B  10       2.000   0.000   2.000  1.00  0.00           O
ATOM     17  C1  TR0 B  10       2.000   1.000   2.000  1.00  0.00           C
TER
END
"""

def test_python_tris_inclusion():
    """Verify that Python API featurisation includes Tris when present in the universe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "system.pdb"
        with open(pdb_path, "w") as f:
            f.write(PDB_CONTENT)
            
        u_with_tris = mda.Universe(str(pdb_path))
        
        # Write protein-only PDB as the control
        protein_atoms = u_with_tris.select_atoms("protein")
        protein_pdb_path = Path(tmpdir) / "protein_only.pdb"
        protein_atoms.write(str(protein_pdb_path))
        u_protein_only = mda.Universe(str(protein_pdb_path))
        
        # Featurise both systems
        config = BV_model_Config(num_timepoints=1)
        
        # 1. System with Tris
        model_with_tris = BV_model(config=config)
        model_with_tris.initialise([u_with_tris])
        features_with_tris, _ = model_with_tris.featurise([u_with_tris])
        
        # 2. Protein-only system (Tris stripped)
        model_protein_only = BV_model(config=config)
        model_protein_only.initialise([u_protein_only])
        features_protein_only, _ = model_protein_only.featurise([u_protein_only])
        
        # Check heavy contacts and oxygen acceptor contacts
        contacts_with_tris_heavy = np.array(features_with_tris.heavy_contacts)
        contacts_protein_heavy = np.array(features_protein_only.heavy_contacts)
        
        contacts_with_tris_acceptor = np.array(features_with_tris.acceptor_contacts)
        contacts_protein_acceptor = np.array(features_protein_only.acceptor_contacts)
        
        # Differences should show strictly greater contacts when Tris is present
        diff_heavy = contacts_with_tris_heavy - contacts_protein_heavy
        diff_acceptor = contacts_with_tris_acceptor - contacts_protein_acceptor
        
        assert np.any(diff_heavy > 0), "Heavy contacts with Tris should be greater than without Tris"
        assert np.all(diff_heavy >= 0), "Heavy contacts should never be fewer when Tris is present"
        
        assert np.any(diff_acceptor > 0), "Acceptor contacts with Tris should be greater than without Tris"
        assert np.all(diff_acceptor >= 0), "Acceptor contacts should never be fewer when Tris is present"

def test_cli_tris_inclusion():
    """Verify that CLI featurisation includes Tris when present in the universe."""
    # Find path of jaxent-featurise executable in virtual environment
    cli_executable = Path(sys.executable).parent / "jaxent-featurise"
    cli_cmd = str(cli_executable) if cli_executable.exists() else "jaxent-featurise"

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = Path(tmpdir) / "system.pdb"
        with open(pdb_path, "w") as f:
            f.write(PDB_CONTENT)
            
        # Write protein-only PDB as the control
        u_temp = mda.Universe(str(pdb_path))
        protein_atoms = u_temp.select_atoms("protein")
        protein_pdb_path = Path(tmpdir) / "protein_only.pdb"
        protein_atoms.write(str(protein_pdb_path))
        
        # Run CLI on system with Tris
        output_dir_with_tris = Path(tmpdir) / "output_with_tris"
        cmd_with_tris = [
            cli_cmd,
            "--top_path", str(pdb_path),
            "--output_dir", str(output_dir_with_tris),
            "--name", "with_tris",
            "bv",
            "--num_timepoints", "1",
            "--timepoints", "0.167"
        ]
        result_with_tris = subprocess.run(cmd_with_tris, capture_output=True, text=True, check=False)
        assert result_with_tris.returncode == 0, f"CLI with Tris failed: {result_with_tris.stderr}"
        
        # Run CLI on protein-only system
        output_dir_protein_only = Path(tmpdir) / "output_protein_only"
        cmd_protein_only = [
            cli_cmd,
            "--top_path", str(protein_pdb_path),
            "--output_dir", str(output_dir_protein_only),
            "--name", "protein_only",
            "bv",
            "--num_timepoints", "1",
            "--timepoints", "0.167"
        ]
        result_protein_only = subprocess.run(cmd_protein_only, capture_output=True, text=True, check=False)
        assert result_protein_only.returncode == 0, f"CLI protein-only failed: {result_protein_only.stderr}"
        
        # Load output features
        features_with_tris = AbstractFeatures.load(output_dir_with_tris / "features.npz")
        features_protein_only = AbstractFeatures.load(output_dir_protein_only / "features.npz")
        
        contacts_with_tris_heavy = np.array(features_with_tris.heavy_contacts)
        contacts_protein_heavy = np.array(features_protein_only.heavy_contacts)
        
        contacts_with_tris_acceptor = np.array(features_with_tris.acceptor_contacts)
        contacts_protein_acceptor = np.array(features_protein_only.acceptor_contacts)
        
        # Differences should show strictly greater contacts when Tris is present
        diff_heavy = contacts_with_tris_heavy - contacts_protein_heavy
        diff_acceptor = contacts_with_tris_acceptor - contacts_protein_acceptor
        
        assert np.any(diff_heavy > 0), "CLI: Heavy contacts with Tris should be greater than without Tris"
        assert np.all(diff_heavy >= 0), "CLI: Heavy contacts should never be fewer when Tris is present"
        
        assert np.any(diff_acceptor > 0), "CLI: Acceptor contacts with Tris should be greater than without Tris"
        assert np.all(diff_acceptor >= 0), "CLI: Acceptor contacts should never be fewer when Tris is present"
