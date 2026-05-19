import time
import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
from pathlib import Path

def main():
    pdb_path = "jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb"
    xtc_path = "jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined.xtc"
    out_path = "jaxent/examples/4_aSyn/data/_aSyn/tris_MD/features/tris_residue_contacts.npz"

    print("Loading universe...")
    u = mda.Universe(pdb_path, xtc_path)
    protein_heavy = u.select_atoms("protein and not name H*")
    tris_heavy = u.select_atoms("(resname TR0 or resname TR1) and not name H*")

    # Map protein atoms to unique residue index (0 to N_residues-1)
    prot_resindices = protein_heavy.resindices
    unique_prot_resindices = np.unique(prot_resindices)
    prot_resindex_to_idx = {ri: i for i, ri in enumerate(unique_prot_resindices)}
    prot_mapped_indices = np.array([prot_resindex_to_idx[ri] for ri in prot_resindices])

    # Get the residue number for each unique residue index
    # (since MDAnalysis groups atoms by residue, unique resindices correspond to the residues in order)
    resids = np.array([protein_heavy.residues[ri].resid for ri in unique_prot_resindices])

    # Map tris atoms to unique tris index (0 to N_tris-1)
    tris_resindices = tris_heavy.resindices
    unique_tris_resindices = np.unique(tris_resindices)
    tris_resindex_to_idx = {ri: i for i, ri in enumerate(unique_tris_resindices)}
    tris_mapped_indices = np.array([tris_resindex_to_idx[ri] for ri in tris_resindices])

    N_residues = len(unique_prot_resindices)
    N_tris = len(unique_tris_resindices)
    N_frames = len(u.trajectory)

    print(f"Residues: {N_residues}, Tris molecules: {N_tris}, Frames: {N_frames}")

    # Output arrays: shape (N_frames, N_residues)
    tris_bound = np.zeros((N_frames, N_residues), dtype=np.int8)
    tris_contacts = np.zeros((N_frames, N_residues), dtype=np.int32)

    t0 = time.time()
    for ts in u.trajectory:
        frame_idx = ts.frame
        if frame_idx % 10000 == 0:
            print(f"Processing frame {frame_idx}/{N_frames}...")
            
        pairs, dists = capped_distance(
            protein_heavy.positions,
            tris_heavy.positions,
            max_cutoff=5.0,
            box=u.dimensions
        )
        grid = np.zeros((N_residues, N_tris), dtype=bool)
        if len(pairs) > 0:
            p_idx = prot_mapped_indices[pairs[:, 0]]
            t_idx = tris_mapped_indices[pairs[:, 1]]
            grid[p_idx, t_idx] = True
            tris_contacts[frame_idx] = np.bincount(p_idx, minlength=N_residues)
        tris_bound[frame_idx] = grid.sum(axis=1)

    t1 = time.time()
    print(f"Completed in {t1 - t0:.2f} seconds.")

    # Save to npz
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, resids=resids, tris_bound=tris_bound, tris_contacts=tris_contacts)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
