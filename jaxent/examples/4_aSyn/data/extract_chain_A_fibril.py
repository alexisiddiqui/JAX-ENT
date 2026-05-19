"""
Extracts all chains from fibril PDBs and converts each to its own frame.
Sets all chain/segment IDs to 'A' for consistency.

Input Dir:
/Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/fibril_PDBs

Output Dir:
/Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/fibrils/chain_A
"""

import os
from pathlib import Path
import MDAnalysis as mda
import tempfile

def extract_all_chains_as_frames():
    # Define paths
    base_dir = Path("/Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn")
    input_dir = base_dir / "fibril_PDBs"
    output_dir = base_dir / "fibrils" / "chain_A"
    combined_pdb_path = output_dir / "all_chain_A_multiframe.pdb"

    # PDB IDs to process
    pdb_ids = [
        "8ADW", "9FYP", "8AEX", "8Y2P", "2N0A",
        "6CU7", "6CU8", "6RT0", "6RTB", "7NCK", "8PIX"
    ]

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting multi-chain extraction to: {output_dir}")

    total_frames_collected = [] # List of list of strings (lines)

    for pdb_id in pdb_ids:
        pdb_path = input_dir / f"{pdb_id}.pdb"
        output_path = output_dir / f"{pdb_id}_chain_A_frames.pdb"

        if not pdb_path.exists():
            print(f"Warning: {pdb_path} not found. Skipping.")
            continue

        print(f"Processing {pdb_id}...")
        try:
            # We must use a separate universe per PDB to avoid ID conflicts
            u = mda.Universe(str(pdb_path))
            protein = u.select_atoms("protein")

            # Group by segment (chain)
            segments = protein.segments
            print(f"  Found {len(segments)} segments in {pdb_id}")

            # We will create a multiframe PDB for this specific ID
            frames_for_this_pdb = []
            atom_counts = []  # Track atom count per frame for validation

            # Use a temporary file for capturing atom lines reliably
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as tmp:
                tmp_path = tmp.name

                for i, seg in enumerate(segments, 1):
                    # Select atoms in this segment
                    seg_atoms = seg.atoms

                    # Normalize chain/seg IDs to 'A'
                    # We modify the attributes in place for this universe instance
                    seg_atoms.segments.segids = "A"
                    seg_atoms.chainIDs = "A"

                    # Write to temporary file
                    seg_atoms.write(tmp_path)

                    # Read back ATOM lines
                    pdb_lines = []
                    with open(tmp_path, "r") as f:
                        for line in f:
                            if line.startswith(("ATOM", "HETATM")):
                                pdb_lines.append(line)

                    atom_count = len(pdb_lines)
                    atom_counts.append(atom_count)
                    frames_for_this_pdb.append(pdb_lines)
                    total_frames_collected.append(pdb_lines)

            # Validate atom counts for consistency
            min_atoms = min(atom_counts)
            max_atoms = max(atom_counts)

            if min_atoms != max_atoms:
                print(f"  ⚠ Inconsistent atom counts detected: min={min_atoms}, max={max_atoms}")
                print(f"    Keeping only the most complete structure ({max_atoms} atoms)")

                # Find the frame(s) with the most atoms and keep only those
                max_idx = atom_counts.index(max_atoms)
                frames_for_this_pdb = [frames_for_this_pdb[max_idx]]
                total_frames_collected[-len(atom_counts)+1:] = [frames_for_this_pdb[0]]

            # Write individual multiframe file
            with open(output_path, "w") as out_f:
                for idx, lines in enumerate(frames_for_this_pdb, 1):
                    out_f.write(f"MODEL     {idx:4d}\n")
                    out_f.writelines(lines)
                    out_f.write("ENDMDL\n")

            print(f"  Successfully wrote {len(frames_for_this_pdb)} frame(s) to {output_path}")

        except Exception as e:
            print(f"  Error processing {pdb_id}: {e}")

    # Write master multiframe PDB
    print(f"Combining {len(total_frames_collected)} total frames into master file: {combined_pdb_path}")
    with open(combined_pdb_path, "w") as master_f:
        for i, frame_lines in enumerate(total_frames_collected, 1):
            master_f.write(f"MODEL     {i:4d}\n")
            master_f.writelines(frame_lines)
            master_f.write("ENDMDL\n")

    print("Master extraction complete.")

if __name__ == "__main__":
    extract_all_chains_as_frames()