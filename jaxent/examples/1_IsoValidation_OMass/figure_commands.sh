pymol -r /Users/alexi/JAX-ENT/jaxent/src/analysis/pymol/Top_N_structures.py --config /Users/alexi/JAX-ENT/jaxent/src/analysis/pymol/test_configs/top_n_TeaA_RMSD.yaml


# Prep for generating inro scheme

jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb \
--trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_open_sliced.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/open_features \
--name Open_features \
 bv \
--switch \
--peptide_trim 0 


jaxent-predict --topology_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/open_features/topology.json \
--features_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/open_features/features.npz \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/uptake \
--output_name Open_uptake \
bv 





jaxent-featurise --top_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/TeaA_ref_open_state.pdb \
--trajectory_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/sliced_trajectories/TeaA_closed_sliced.xtc \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/closed_features \
--name Closed_features \
 bv \
--switch \
--peptide_trim 0 


jaxent-predict --topology_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/closed_features/topology.json \
--features_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/closed_features/features.npz \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/uptake \
--output_name Closed_uptake \
bv 




# After running featurise from main pipeline


jaxent-predict --topology_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise/topology_iso_bi.json \
--features_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise/features_iso_bi.npz \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/uptake \
--output_name bi_uptake \
bv 



jaxent-predict --topology_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise/topology_iso_tri.json \
--features_path /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/_featurise/features_iso_tri.npz \
--output_dir /Users/alexi/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/data/_Bradshaw/Reproducibility_pack_v2/data/trajectories/uptake \
--output_name tri_uptake \
bv 



# Create intro schema charts


python jaxent/examples/1_IsoValidation_OMass/data/plot_figure_schema.py
