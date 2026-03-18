"""
Clusters trajectory based on RMSD to reference structures.

Performs an Calpha aligned RMSD (filtering for common residues) between trajectory and reference structures.

Similar to the IsoValidation clustering script jaxent/examples/1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py
- generates a clustering_assignments.csv


Example output of this script can be found in jaxent/examples/4_SAXS/data/_RMSD_cluster_output/



Script args:
- trajectory_path: path to the trajectory file (jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_max_plddt_425.pdb)
- topology_path: path to the topology file (jaxent/examples/4_SAXS/data/_CaM/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_plddt_ordered.xtc)
- reference_paths: list of paths to the reference structures (jaxent/examples/4_SAXS/FOXS/missing_residues/7PSZ_apo.pdb, jaxent/examples/4_SAXS/FOXS/missing_residues/1CLL_nosol.pdb)
- output_path: path to the output directory (jaxent/examples/4_SAXS/data/_RMSD_cluster_output/)






"""