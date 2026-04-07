
jaxent-kCluster --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
--trajectory_paths /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn \
--number_of_clusters 1000 \
--num_components 20 


jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
--trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data \
--name aSyn_featurised \
 bv

jaxent-predict --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data/topology.json \
--features_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data/features.npz \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data \
--output_name aSyn_featurised \
bv 



jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
--trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/clusters/all_clusters.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/features \
--name cluster_aSyn_featurised \
 bv \
--switch \
--peptide_trim 0 