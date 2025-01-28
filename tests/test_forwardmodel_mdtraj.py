import glob
import os
from typing import List, Tuple

import mdtraj as md
import numpy as np

from jaxent.forwardmodels.mdtraj_functions import calc_BV_contacts_mdtraj


def load_contact_data(data_dir: str, file_prefix: str = "Contacts") -> dict:
    """Load contact data from .tmp files."""
    pattern = os.path.join(data_dir, f"{file_prefix}_chain_0_res_*.tmp")
    tmp_files = glob.glob(pattern)

    contact_data = {}
    for file_path in tmp_files:
        filename = os.path.basename(file_path)
        resid = int(filename.split("res_")[1].split(".tmp")[0])

        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:
                    contact_value = float(first_line.split()[0])
                    contact_data[resid] = contact_value
        except (ValueError, IndexError):
            print(f"Warning: Could not read contact value from {file_path}")
            continue

    return contact_data


def test_calc_contacts_mdtraj():
    """Test the calculation of contacts against reference data with detailed comparison."""
    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
    data_dir = "/home/alexi/Documents/JAX-ENT/tests/inst/clean/HOIP/train_HOIP_high_rank_1"

    # Load trajectory
    traj = md.load(topology_path)

    # Get N atoms for each residue (excluding prolines)
    NH_residue_atom_index: List[Tuple[int, int]] = []
    for residue in traj.topology.residues:
        if residue.name != "PRO":
            try:
                N_atom = next(atom for atom in residue.atoms if atom.name == "N")
                NH_residue_atom_index.append((residue.resSeq, N_atom.index))
            except StopIteration:
                continue

    # Get H atoms for each residue (excluding prolines)
    HN_residue_atom_index: List[Tuple[int, int]] = []
    for residue in traj.topology.residues:
        if residue.name != "PRO":
            try:
                H_atom = next(atom for atom in residue.atoms if atom.name == "H")
                HN_residue_atom_index.append((residue.resSeq, H_atom.index))
            except StopIteration:
                continue

    # Convert both lists to sets of just the residue numbers (first element of each tuple)
    NH_residues = set(res_num for res_num, _ in NH_residue_atom_index)
    HN_residues = set(res_num for res_num, _ in HN_residue_atom_index)

    # Find the intersection of residue numbers
    common_residues = NH_residues.intersection(HN_residues)

    # Filter both lists to keep only the entries where residue number is in the intersection
    NH_residue_atom_index = [
        (res_num, atom_idx)
        for res_num, atom_idx in NH_residue_atom_index
        if res_num in common_residues
    ]
    HN_residue_atom_index = [
        (res_num, atom_idx)
        for res_num, atom_idx in HN_residue_atom_index
        if res_num in common_residues
    ]

    # Sort both lists by residue number to ensure they're in the same order
    NH_residue_atom_index.sort()
    HN_residue_atom_index.sort()

    # Calculate contacts using aligned implementation
    heavy_contacts = calc_BV_contacts_mdtraj(
        universe=traj,
        residue_atom_index=NH_residue_atom_index,
        contact_selection="heavy",
        radius=0.65,
        switch=False,
    )

    oxygen_contacts = calc_BV_contacts_mdtraj(
        universe=traj,
        residue_atom_index=HN_residue_atom_index,
        contact_selection="oxygen",
        radius=0.24,
        switch=False,
    )

    # Load reference data
    ref_contacts = load_contact_data(data_dir, "Contacts")
    ref_hbonds = load_contact_data(data_dir, "Hbonds")

    # Print comparison header
    print("\nDetailed Contacts Comparison:")
    print("-" * 120)
    print(
        f"{'Residue ID':^10} {'Calc Heavy':^12} {'Ref Heavy':^12} {'Heavy Δ':^12} {'Heavy %Δ':^12} "
        f"{'Calc O':^12} {'Ref O':^12} {'O Δ':^12} {'O %Δ':^12} {'Match?':^8}"
    )
    print("-" * 120)

    matches = []
    mismatches = []

    # Compare calculated vs reference values
    for i, ((resid, _), heavy, oxygen) in enumerate(
        zip(NH_residue_atom_index, heavy_contacts, oxygen_contacts)
    ):
        calc_heavy = heavy[0]  # Single frame, take first value
        calc_oxygen = oxygen[0]  # Single frame, take first value

        ref_heavy = ref_contacts.get(resid, None)
        ref_oxygen = ref_hbonds.get(resid, None)

        if ref_heavy is not None and ref_oxygen is not None:
            heavy_diff = calc_heavy - ref_heavy
            oxygen_diff = calc_oxygen - ref_oxygen

            heavy_pct = (heavy_diff / ref_heavy * 100) if ref_heavy != 0 else float("inf")
            oxygen_pct = (oxygen_diff / ref_oxygen * 100) if ref_oxygen != 0 else float("inf")

            # Consider a match if both absolute differences are within tolerance
            matches_within_tol = (abs(heavy_diff) < 1.0) and (abs(oxygen_diff) < 1.0)

            comparison = {
                "resid": resid,
                "calc_heavy": calc_heavy,
                "ref_heavy": ref_heavy,
                "heavy_diff": heavy_diff,
                "heavy_pct": heavy_pct,
                "calc_oxygen": calc_oxygen,
                "ref_oxygen": ref_oxygen,
                "oxygen_diff": oxygen_diff,
                "oxygen_pct": oxygen_pct,
            }

            if matches_within_tol:
                matches.append(comparison)
            else:
                mismatches.append(comparison)

            # Add +/- signs to differences
            heavy_diff_str = f"{'+' if heavy_diff > 0 else ''}{heavy_diff:.2f}"
            oxygen_diff_str = f"{'+' if oxygen_diff > 0 else ''}{oxygen_diff:.2f}"
            heavy_pct_str = f"{'+' if heavy_diff > 0 else ''}{heavy_pct:.1f}%"
            oxygen_pct_str = f"{'+' if oxygen_diff > 0 else ''}{oxygen_pct:.1f}%"

            print(
                f"{resid:^10d} {calc_heavy:^12.2f} {ref_heavy:^12.2f} {heavy_diff_str:^12} {heavy_pct_str:^12} "
                f"{calc_oxygen:^12.2f} {ref_oxygen:^12.2f} {oxygen_diff_str:^12} {oxygen_pct_str:^12} "
                f"{'✓' if matches_within_tol else '✗':^8}"
            )

    # Print summary
    print("\nSummary:")
    print(f"Total residues compared: {len(matches) + len(mismatches)}")
    print(f"Matching contacts: {len(matches)}")
    print(f"Mismatching contacts: {len(mismatches)}")

    if mismatches:
        print("\nMismatch Analysis:")
        print("-" * 80)

        # Analyze trends in mismatches
        heavy_higher = sum(1 for m in mismatches if m["heavy_diff"] > 0)
        heavy_lower = sum(1 for m in mismatches if m["heavy_diff"] < 0)
        oxygen_higher = sum(1 for m in mismatches if m["oxygen_diff"] > 0)
        oxygen_lower = sum(1 for m in mismatches if m["oxygen_diff"] < 0)

        print("\nHeavy Contact Trends:")
        print(f"  Higher than reference: {heavy_higher} residues")
        print(f"  Lower than reference:  {heavy_lower} residues")
        print(f"  Average deviation: {np.mean([m['heavy_diff'] for m in mismatches]):.2f}")
        print(f"  Average % change: {np.mean([m['heavy_pct'] for m in mismatches]):.1f}%")

        print("\nOxygen Contact Trends:")
        print(f"  Higher than reference: {oxygen_higher} residues")
        print(f"  Lower than reference:  {oxygen_lower} residues")
        print(f"  Average deviation: {np.mean([m['oxygen_diff'] for m in mismatches]):.2f}")
        print(f"  Average % change: {np.mean([m['oxygen_pct'] for m in mismatches]):.1f}%")

        print("\nLargest Mismatches:")
        sorted_by_heavy = sorted(mismatches, key=lambda x: abs(x["heavy_pct"]), reverse=True)[:5]
        sorted_by_oxygen = sorted(mismatches, key=lambda x: abs(x["oxygen_pct"]), reverse=True)[:5]

        print("\nTop 5 Heavy Contact Mismatches:")
        for m in sorted_by_heavy:
            print(
                f"  Residue {m['resid']}: {m['calc_heavy']:.2f} vs {m['ref_heavy']:.2f} "
                f"(Δ: {m['heavy_diff']:+.2f}, {m['heavy_pct']:+.1f}%)"
            )

        print("\nTop 5 Oxygen Contact Mismatches:")
        for m in sorted_by_oxygen:
            print(
                f"  Residue {m['resid']}: {m['calc_oxygen']:.2f} vs {m['ref_oxygen']:.2f} "
                f"(Δ: {m['oxygen_diff']:+.2f}, {m['oxygen_pct']:+.1f}%)"
            )

    # Assert that most contacts match within tolerance
    match_ratio = len(matches) / (len(matches) + len(mismatches))
    assert match_ratio > 0.9, (
        f"Only {match_ratio:.1%} of contacts match reference values (threshold: 90%)"
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_calc_contacts_mdtraj()
