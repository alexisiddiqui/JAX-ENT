"""

This script loads in the csv data containing Protection Factors in Log10 these are to be converted to Ln (Log_e).

log10pf_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_expt_PFs.dat"



BPTI_expt_PFs is a csv with a header and the following columns:
# ResID, log10(PF)


BPTI_pfactors.dat is the output csv with no header but containing the following columns:
ResID, ln(PF)

-> LnPF is calculated from the log10PF

"""

import numpy as np
import pandas as pd

log10pf_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_expt_PFs.dat"
output_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_pfactors.dat"

# Read the file with whitespace as delimiter, skipping comment lines, with no header
log10pf_df = pd.read_csv(
    log10pf_path, delim_whitespace=True, comment="#", header=None, names=["ResID", "log10PF"]
)

# Convert log10(PF) to ln(PF)
ln10 = np.log(10)  # Natural logarithm of 10
ln_pf = log10pf_df["log10PF"] * ln10

# Create a new DataFrame with ResID and ln(PF)
result_df = pd.DataFrame({"ResID": log10pf_df["ResID"], "ln(PF)": ln_pf})

# Save the result to a file without header
result_df.to_csv(output_path, sep="\t", header=False, index=False)

print(f"Conversion completed. File saved as {output_path}")
