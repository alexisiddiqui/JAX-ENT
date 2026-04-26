
jaxent-kCluster --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
--trajectory_paths /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn \
--number_of_clusters 1000 \
--num_components 20 


# jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
# --trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
# --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_cluster_aSyn/data \
# --name aSyn_featurised \
#  bv \
# --switch \
# --peptide_trim 0 


jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
--trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/features \
--name aSyn_featurised \
 bv \
--switch \
--peptide_trim 0 


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





jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
--trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/features \
--name aSyn_featurised \
 bv \
--switch \
--peptide_trim 0 



jaxent-featurise --top_path jaxent/examples/4_aSyn/data/_aSyn/a99sb.pdb \
--trajectory_path jaxent/examples/4_aSyn/data/_aSyn/a99sb.pdb \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb_features \
--name a99sb \
 bv \
--switch \
--peptide_trim 0 



jaxent-predict --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb_features/topology.json \
--features_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb_features/features.npz \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb_features \
--output_name aSyn_featurised \
bv 

# MD 


jaxent-ipca --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
--trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
--condition AF2-MSA \
--replicate 1 \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/ipca_plots \
--name AF2-MSA \
--num_components 10


jaxent-ipca \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
  --condition AF2-MSA \
  --replicate 1 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb.pdb \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb.pdb \
  --condition a99sb \
  --replicate 1 \
  --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/ipca_plots \
  --name aSyn_comparison \
  --num_components 10


jaxent-ipca \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
  --condition AF2-MSA --replicate 1 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb.pdb \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/a99sb.pdb \
  --condition a99sb --replicate 1 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_rod.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_rep1_combined.xtc \
  --condition rod --replicate 1 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_rod.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_rep2_combined.xtc \
  --condition rod --replicate 2 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_rod.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_rep3_combined.xtc \
  --condition rod --replicate 3 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_rep1_combined.xtc \
  --condition coil --replicate 1 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_rep2_combined.xtc \
  --condition coil --replicate 2 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_rep3_combined.xtc \
  --condition coil --replicate 3 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_hairpin.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_rep1_combined.xtc \
  --condition hairpin --replicate 1 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_hairpin.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_rep2_combined.xtc \
  --condition hairpin --replicate 2 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_hairpin.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_rep3_combined.xtc \
  --condition hairpin --replicate 3 \
  --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/ipca_plots \
  --name aSyn_joint_pca \
  --num_components 10


gmx trjcat -f /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_rep1_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_rep2_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_rep3_combined.xtc -o /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_combined.xtc -cat

gmx trjcat -f /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_rep1_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_rep2_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_rep3_combined.xtc -o /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_combined.xtc -cat

gmx trjcat -f /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_rep1_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_rep2_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_rep3_combined.xtc -o /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_combined.xtc -cat

gmx trjcat -f /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_rod_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_coil_combined.xtc /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_hairpin_combined.xtc -o /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined.xtc -cat

# resample from 1ps to 1 ns
gmx trjconv -f /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined.xtc -o /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.xtc -dt 100

# convert to pdb
gmx trjconv -f /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.xtc -s /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_rod.gro -o /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.pdb

jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb --trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined.xtc --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/features --name aSyn_featurised bv --switch --peptide_trim 0





jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb --trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.xtc --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/features --name aSyn_featurised bv --switch --peptide_trim 0



jaxent-predict --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/features/topology.json --features_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/features/features.npz --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/features --output_name aSyn_featurised bv




jaxent-kCluster --topology_path /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_coil.pdb \
--trajectory_paths /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined_filtered_10.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/3cluster_test \
--atom_selection "name CA and resid 1:95" \
--number_of_clusters 3 \
--num_components 20 \
--save_pdbs

jaxent-ipca \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_first_frame.pdb \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/aSyn_s20_r1_msa1-127_n12700_do1_20260329_025853_protonated_plddt_ordered.xtc \
  --condition AF2-MSA --replicate 1 \
  --topology /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/md_mol_center_rod.gro \
  --trajectory /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/tris_MD/tris_all_combined.xtc \
  --condition Tris-MD --replicate 1 \
  --output_dir /Users/alexi/JAX-ENT/jaxent/examples/4_aSyn/data/_aSyn/AF2_MD_joint_pca \
  --atom_selection "name CA and resid 1:95" \
  --name aSyn_joint_pca \
  --num_components 10