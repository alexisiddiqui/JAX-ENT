

SCRIPT_WD=$(dirname "$0")

PDB_CODES=(2L1H 2L39)


for PDB_CODE in "${PDB_CODES[@]}"; do
    wget "https://files.rcsb.org/download/${PDB_CODE}.pdb" -O "${SCRIPT_WD}/${PDB_CODE}.pdb"
done
echo "Downloaded PDB files: ${PDB_CODES[*]}"

# Crop PDBs to 123-224 (101 residues)
# for now do this in pymol