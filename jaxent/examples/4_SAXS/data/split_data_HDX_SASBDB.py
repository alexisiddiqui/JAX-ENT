"""
Split HDX-MS uptake peptide data 

Loads an experimental HDX-MS.csv file 
Protein,Start,End,Sequence,MaxUptake,Max Uptake 80% D20,State,Exposure Time (min),Uptake (Da),Uptake SD (Da)

Fractional uptake = Uptake (Da) / Max Uptake 80% (Da)

Plots a heatmap of the fractional uptake for each peptide (ordered by start/end residues per peptide)

Save a formated HDX-MS _dfrac.csv file with columns: (wide format) 
#  timepoints*
and a segs file with columns:
#  start, end

- this is so that covariance matrices can be computed by
jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/compute_sigma_real.py



Splits across train/val using built-in DataSplitter whole-system strategies
(random, sequence-cluster, spatial),
saves .npz splits for downstream fitting, and produces publication figures.

Example:

python jaxent/examples/4_SAXS/data/split_data_HDX_SASBDB.py \
--hdx-data jaxent/examples/4_SAXS/data/_CaM/raw_data/SASDNY3/experimental_data/HOLO_CaM_CDZ.csv \
--output-dir jaxent/examples/4_SAXS/fitting/_datasplits_HDX_CaM+CDZ \
--name CaM+CDZ \
--n-splits 3 \
--train-size 0.5 \
--seed 42



python jaxent/examples/4_SAXS/data/split_data_HDX_SASBDB.py \
--hdx-data jaxent/examples/4_SAXS/data/_CaM/raw_data/SASDNX3/experimental_data/HOLO_CaM.csv \
--output-dir jaxent/examples/4_SAXS/fitting/_datasplits_HDX_CaM-CDZ \
--name CaM-CDZ \
--n-splits 3 \
--train-size 0.5 \
--seed 42





"""
