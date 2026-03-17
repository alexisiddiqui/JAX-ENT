"""

Performs a second round of clustering on the trajectory data, using the clusters from a previous clustering as a starting point.
In this case this is the RMSD clustering performed in jaxent/examples/4_SAXS/data/cluster_RMSD_references.py

From these clusters, we perform a second round of clustering using iPCA and k-means to further refine the clusters.

We then save the sub-cluster assignments to a csv file as well as the mediods of each sub-cluster as PDB files + their own clustering assignments.

Script args:
- trajectory_path: path to the trajectory file
- topology_path: path to the topology file
- clustering_assignments_csv: path to the clustering assignments csv file
- k: number of sub-clusters per assigned cluster
- seed: seed for reproducibility (default: 42)
- pca_dims: number of iPCA components to use for sub-clustering and visualisation
- reference_paths: list of paths to the reference structures
- output_path: path to the output directory

Example output: jaxent/examples/4_SAXS/data/_sub_cluster_output/

"""