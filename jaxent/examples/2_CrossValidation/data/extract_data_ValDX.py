"""
[Script Name] extract_data_ValDX.py

[Brief Description of Functionality]
This script extracts and formats experimental HDX data for the MoPrP dataset (ValDX).
It processes raw data files (.dexp, .list) to produce clean dfrac (deuterium fraction)
data and peptide segment information. Additionally, it formats the protection factor
(.pfact) file to filter for relevant residue data.

Requirements:
    - Input files in `jaxent/examples/2_CrossValidation/data/`:
        - `_ValDX.tar` (must be extracted to provide .dexp, .list, .pfact files)
    - Dependencies: pandas, numpy

Usage:
    # As run in commands.sh:
    python jaxent/examples/2_CrossValidation/data/extract_data_ValDX.py

Output:
    - Formatted .dat and .txt files in `jaxent/examples/2_CrossValidation/data/_output/`
      containing processed dfracs and segments.
"""

- 1st column: residue number
- 2nd column: Protection Factor (ln)

This will create three files files:
- MoPrP_dfrac.dat -> convert times to minutes
- MoPrP_segments.txt
- MoPrP_pfactors.dat -> remove 0 or negative values


dfrac.dat is a file with a header of the following format:

#	0.5	5.0	 times/min
0.41879	0.44438
0.37096	0.45204

-> make sure that the dfrac data contains all the timepoints


segments.txt is a file with no header of the following format:
residue_start	residue_end


median.pfact is a file with no header of the following format:
residue_number	ln(PF)

-> remove 0 or negative values

raw_dfrac_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_cross_validation/MoPrP/MoPrP/moprp.dexp"

raw_segs_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_cross_validation/MoPrP/MoPrP/moprp.list"

pf_path = "/home/alexi/Documents/ValDX/figure_scripts/jaxent_cross_validation/MoPrP/MoPrP/median.pfact"

"""

import os

import pandas as pd

script_path = os.path.dirname(os.path.abspath(__file__))
# File paths
raw_dfrac_path = f"{script_path}/_MoPrP/moprp.dexp"
raw_segs_path = f"{script_path}/_MoPrP/moprp.list"
pf_path = f"{script_path}/_MoPrP/median.pfact"
output_dir = os.path.join(os.path.dirname(__file__), "_MoPrP/_output")
os.makedirs(output_dir, exist_ok=True)

# Output file paths
output_dfrac_path = os.path.join(output_dir, "MoPrP_dfrac.dat")
output_segs_path = os.path.join(output_dir, "MoPrP_segments.txt")
output_pfact_path = os.path.join(output_dir, "MoPrP_pfactors.dat")

# Read data files
print("Reading data files...")
dfrac_df = pd.read_csv(raw_dfrac_path, header=None, sep="\s+")
segs_df = pd.read_csv(raw_segs_path, header=None, sep="\s+")
pfact_df = pd.read_csv(pf_path, header=None, sep="\s+")

# Extract times in hours and convert to minutes
times_hours = dfrac_df.iloc[:, 0].values
times_minutes = times_hours * 60  # Convert hours to minutes

print(f"Time points (hours): {times_hours}")
print(f"Time points (minutes): {times_minutes}")

# Extract deuteration fractions (all columns except the first one)
dfrac_data = dfrac_df.iloc[:, 1:].values

print(f"Number of time points: {len(times_minutes)}")
print(f"Number of segments: {dfrac_data.shape[1]}")

# Create dfrac file
print("Creating dfrac file...")
with open(output_dfrac_path, "w") as f:
    # Write header with time points in minutes
    header = "#\t" + "\t".join(f"{t:.2f}" for t in times_minutes) + "\t times/min\n"
    f.write(header)

    # Write deuteration fractions for each segment
    # Each row in the output file corresponds to a segment (column in the input data)
    for segment_idx in range(dfrac_data.shape[1]):
        # Get all time points for this segment
        segment_data = dfrac_data[:, segment_idx]
        # Write as tab-separated values
        f.write("\t".join(f"{val:.5f}" for val in segment_data) + "\n")

# Extract segment information
print("Creating segments file...")
with open(output_segs_path, "w") as f:
    # Extract residue start and end positions
    for _, row in segs_df.iterrows():
        # Columns 1 and 2 (0-indexed) contain residue start and end
        res_start = int(row[1])
        res_end = int(row[2])
        f.write(f"{res_start} {res_end}\n")

# Process the protection factors file
print("Creating protection factors file...")
# Filter out zero or negative values
filtered_pfact_df = pfact_df[pfact_df[1] > 0]

# Write the filtered protection factors to the output file
with open(output_pfact_path, "w") as f:
    for _, row in filtered_pfact_df.iterrows():
        residue_num = int(row[0])
        pf_value = row[1]
        f.write(f"{residue_num}\t{pf_value:.5f}\n")

print("Files created successfully:")
print(f"Deuteration fractions: {output_dfrac_path}")
print(f"Segments file: {output_segs_path}")
print(f"Protection factors: {output_pfact_path}")

# Display sample of output files for verification
print("\nSample of dfrac file:")
with open(output_dfrac_path, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 3:  # Print first few lines only
            break

print("\nSample of segments file:")
with open(output_segs_path, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 3:  # Print first few lines only
            break

print("\nSample of protection factors file:")
with open(output_pfact_path, "r") as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 3:  # Print first few lines only
            break
