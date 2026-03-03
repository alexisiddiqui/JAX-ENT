"""
[Script Name] renumber_pdb.py

[Brief Description of Functionality]
Renumbers residues in a PDB file (supports multiframe PDBs) after sub-selecting a specific
residue range. This is useful for creating cropped PDBs with consistent 1-based indexing
starting from the first residue of the selection.

Requirements:
    - MDAnalysis package
    - Input PDB file

Usage:
    # Example 1: Renumber 2L39
    python renumber_pdb.py \\
        --input_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L39.pdb \\
        --output_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_renum.pdb \\
        --selection "resid 119-231" \\
        --resi_start 1

    # Example 2: Renumber 2L1H
    python renumber_pdb.py \\
        --input_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H.pdb \\
        --output_pdb jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_renum.pdb \\
        --selection "resid 119-231" \\
        --resi_start 1

Output:
    - A new PDB file with the selected atoms and renumbered residues.
"""

import argparse
import sys
from pathlib import Path

try:
    import MDAnalysis as mda
except ImportError:
    print("Error: MDAnalysis not found. Install with: pip install MDAnalysis")
    sys.exit(1)


def renumber_pdb_selection(input_pdb, output_pdb, selection_string, resi_start=1):
    """
    Renumber residues in a PDB file after subselection, retaining multiframe data.
    
    Parameters
    ----------
    input_pdb : str
        Path to input PDB file
    output_pdb : str
        Path to output PDB file
    selection_string : str
        MDAnalysis selection string (e.g., "resid 122:")
    resi_start : int
        Starting residue number for renumbering (default: 1)
    """
    
    # Load the PDB file
    print(f"Loading PDB: {input_pdb}")
    u = mda.Universe(input_pdb)
    n_frames = u.trajectory.n_frames
    print(f"Found {n_frames} frame(s)")
    
    # Parse the selection on the first frame to get residue mapping
    u.trajectory[0]
    try:
        selection = u.select_atoms(selection_string)
    except Exception as e:
        print(f"Error parsing selection '{selection_string}': {e}")
        sys.exit(1)
    
    if selection.n_atoms == 0:
        print(f"Warning: Selection '{selection_string}' returned 0 atoms")
        sys.exit(1)
    
    print(f"Selection '{selection_string}' contains {selection.n_atoms} atom(s) "
          f"from {selection.n_residues} residue(s)")
    
    # Get unique residues from the selection on first frame
    unique_resids = sorted(set(res.resid for res in selection.residues))
    print(f"Original residue numbers: {unique_resids[0]} to {unique_resids[-1]}")
    
    # Create mapping from old to new residue IDs
    resi_mapping = {old_resid: new_resid 
                    for new_resid, old_resid in enumerate(unique_resids, start=resi_start)}
    
    # Create array of new residue IDs for the selection
    new_resids = [resi_mapping[res.resid] for res in selection.residues]
    
    print(f"Remapping residues: {unique_resids[0]} → {resi_start}, "
          f"{unique_resids[-1]} → {resi_start + len(unique_resids) - 1}")
    
    # Store original residue IDs so we can restore them after each frame
    # (needed because selection.residues references the universe's residues)
    original_resids = [res.resid for res in selection.residues]
    
    # Process each frame and write
    # multiframe=True is required to write MODEL/ENDMDL records for trajectory
    with mda.coordinates.PDB.PDBWriter(output_pdb, n_atoms=selection.n_atoms, 
                                        multiframe=True) as W:
        for frame_idx, ts in enumerate(u.trajectory):
            # Apply new residue numbering at the ResidueGroup level
            # Note: We use the original 'selection', NOT a reselection
            # The selection automatically has the current frame's coordinates
            selection.residues.resids = new_resids

            # Write the modified selection
            W.write(selection)
            
            # Restore original residue IDs for next iteration
            # This ensures the selection remains valid throughout the trajectory
            selection.residues.resids = original_resids
            
            if (frame_idx + 1) % max(1, n_frames // 10) == 0 or frame_idx + 1 == n_frames:
                print(f"  Processed frame {frame_idx + 1}/{n_frames}")
    
    print(f"Successfully wrote renumbered structure to: {output_pdb}")


def main():
    parser = argparse.ArgumentParser(
        description="Renumber PDB residues after subselection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Renumber residue indices 123-223 to start at 1
  python renumber_pdb.py --input_pdb input.pdb --output_pdb output.pdb \\
      --selection "resid 123:223" --resi_start 1
  
  # Renumber from residue index 121 onwards, starting at 1
  python renumber_pdb.py --input_pdb input.pdb --output_pdb output.pdb \\
      --selection "resid 121:" --resi_start 1
  
  # Select specific residue index range and renumber from 50
  python renumber_pdb.py --input_pdb input.pdb --output_pdb output.pdb \\
      --selection "resid 99:199" --resi_start 50
  
  # Select residues with resname
  python renumber_pdb.py --input_pdb input.pdb --output_pdb output.pdb \\
      --selection "resname ALA" --resi_start 1
        """
    )
    
    parser.add_argument(
        "--input_pdb",
        required=True,
        help="Input PDB file"
    )
    parser.add_argument(
        "--output_pdb",
        required=True,
        help="Output PDB file"
    )
    parser.add_argument(
        "--selection",
        required=True,
        help="MDAnalysis selection string (e.g., 'resid 123:223' or 'resname ALA')"
    )
    parser.add_argument(
        "--resi_start",
        type=int,
        default=1,
        help="Starting residue number for renumbering (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input_pdb).exists():
        print(f"Error: Input PDB not found: {args.input_pdb}")
        sys.exit(1)
    
    # Run renumbering
    renumber_pdb_selection(
        args.input_pdb,
        args.output_pdb,
        args.selection,
        args.resi_start
    )


if __name__ == "__main__":
    main()