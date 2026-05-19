#!/usr/bin/env python3
import json
import csv
from pathlib import Path

def create_tris_dat():
    base_dir = Path("/Users/alexi/JAX-ENT/jaxent/examples/4_aSyn")
    json_path = base_dir / "data/_aSyn/aSyn_PF_Tris_only.json"
    csv_path = base_dir / "data/_aSyn/aSyn_PF_Tris_only.csv"
    output_path = base_dir / "data/_aSyn/aSyn_PF_Tris_only.dat"

    if not json_path.exists() or not csv_path.exists():
        print(f"Error: Missing input files in {base_dir / 'data/_aSyn'}")
        return

    with open(json_path, 'r') as f:
        topo_data = json.load(f)

    residues = []
    for topo in topo_data['topologies']:
        # Assuming single residue per entry for HDX_resPf
        residues.append(topo['residues'][0])

    values = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(row['0'])

    if len(residues) != len(values):
        print(f"Warning: residues length ({len(residues)}) != values length ({len(values)})")
        # Match as many as possible
        n = min(len(residues), len(values))
        residues = residues[:n]
        values = values[:n]

    with open(output_path, 'w') as f:
        f.write("# residue_number log_pf\n")
        for res, val in zip(residues, values):
            f.write(f"{res} {val}\n")

    print(f"Generated {output_path} with {len(residues)} entries.")

if __name__ == "__main__":
    create_tris_dat()
