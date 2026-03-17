"""
Clusters trajectory based on RMSD to reference structures.

Performs an Calpha aligned RMSD (filtering for common residues) between trajectory and reference structures.

Similar to the IsoValidation clustering script jaxent/examples/1_IsoValidation_OMass/data/extract_OpenClosed_clusters.py
- generates a clustering_assignments.csv


Example output of this script can be found in jaxent/examples/4_SAXS/data/_RMSD_cluster_output/



Script args:
- trajectory_path: path to the trajectory file
- topology_path: path to the topology file
- reference_paths: list of paths to the reference structures
- output_path: path to the output directory






"""