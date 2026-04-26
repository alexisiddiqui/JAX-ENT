import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP

u = mda.Universe('/Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb', '/Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined.xtc')
protein = u.select_atoms("protein")
print("Computing DSSP...")
dssp = DSSP(protein, dssp='mkdssp').run()
print(dssp.results.dssp.shape)
alpha = dssp.results.dssp == 'H'
beta = (dssp.results.dssp == 'E') | (dssp.results.dssp == 'B')
print("Alpha %:", np.mean(alpha)*100, "Beta %:", np.mean(beta)*100)
