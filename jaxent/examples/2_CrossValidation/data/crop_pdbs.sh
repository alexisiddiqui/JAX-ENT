


python /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
    --input_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39.pdb \
    --output_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_crop.pdb \
    --selection "resid 124-224" \
    --resi_start 1


python /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
    --input_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H.pdb \
    --output_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_crop.pdb \
    --selection "resid 124-224" \
    --resi_start 1



python /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
    --input_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39.pdb \
    --output_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L39_renum.pdb \
    --selection "resid 119-231" \
    --resi_start 1


python /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/renumber_pdb.py \
    --input_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H.pdb \
    --output_pdb /home/alexi/Documents/JAX-ENT/jaxent/examples/2_CrossValidation/data/_MoPrP/2L1H_renum.pdb \
    --selection "resid 119-231" \
    --resi_start 1