# JAX-ENT
Maximum Entropy Optimisation for Experimental Biophysics-Simulation Ensemble Integration using JAX.

Currently for HDX-MS data.

Supports:

To do:
HDX-MS peptide data
HDX-MS residue protection factor

Upcoming:
SAXS
NMR
CryoEM???


##############################################################################

Featuriser: Multiple ensembles requires handling of the common resiudes
- probably should use a dataclass to handle complex topology

Optimiser: Each simulation should have its own experimental data section 
- Should tie this together with the 






##############################################################################





This package is aimed at bioinformatics researchers across a range of experience levels we aim to have a package that is batteries included but can also be extende to suit needs.


This package is centered around 3 main functions:

Featurise

Optimise 

Analysis


CLI wrappers will be included under 'scripts'.


uses uv for package management

Please check that jax is correctly installed with CUDA/ROCm if desired.

To do:
Add CUDA/ROCm flag to uv installation

