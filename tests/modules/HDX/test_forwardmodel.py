from typing import List, Tuple

################################################################################
# TODO: NOTWORKING 06/02/25
import MDAnalysis as mda
import numpy as np
import pandas as pd

from jaxent.src.models.func.contacts import calc_BV_contacts_universe
from jaxent.src.models.func.uptake import calculate_intrinsic_rates


def test_calculate_intrinsic_rates():
    """Test the calculation of intrinsic rates against reference data with detailed comparison."""
    # topology_path = "./tests/inst/clean/HOIP/train_HOIP_max_plddt_1/HOIP_apo697_1_af_sample_127_10000_protonated_max_plddt_1969.pdb"
    # rates_path = "./tests/inst/clean/HOIP/train_HOIP_max_plddt_1/out__train_HOIP_max_plddt_1Intrinsic_rates.dat"
    topology_path = "/Users/alexi/JAX-ENT/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    rates_path = "/Users/alexi/JAX-ENT/tests/inst/BPTI_Intrinsic_rates.dat"
    # Load universe
    universe = mda.Universe(topology_path)

    # Calculate rates using our function
    pred_rates = calculate_intrinsic_rates(universe)

    # Basic length checks
    assert len(pred_rates.keys()) == len(universe.residues)
    assert len(pred_rates.keys()) == 58

    # Read reference data
    with open(rates_path, "r") as f:
        header = f.readline().strip()
    print(f"Header from file: {header}")

    df = pd.read_csv(rates_path, delim_whitespace=True)
    print(f"Columns in DataFrame: {df.columns.tolist()}")
    print(f"Shape of DataFrame: {df.shape}")
    # Rename columns
    df.columns = ["ResID", "k_int"] if len(df.columns) == 2 else df.columns
    exp_residues = df["ResID"].values.astype(int)

    exp_kints = df["k_int"].values
    residue_ids = np.array([res.resid for res in universe.residues])
    resnames = np.array([res.resname for res in universe.residues])
    pred_dict = {res: rate for res, rate in pred_rates.items()}
    exp_dict = {res: rate for res, rate in zip(exp_residues, exp_kints)}
    # Get the residue numbers that match the exp_residues
    res_indexes = [np.where(residue_ids == res)[0][0] for res in exp_residues]
    print(res_indexes)
    # breakpoint()
    print(len(res_indexes))
    print("pred rates", pred_rates)

    for i, resi in enumerate(exp_residues):
        print(f"Residue {i}: {resi}")
        print("Resname", resnames[res_indexes[i]])
        print("exp rates", exp_dict[resi])
        print("pred rates", pred_dict[resi])

    # breakpoint()

    assert len(res_indexes) == len(exp_residues)

    # Detailed comparison
    print("\nDetailed Rate Comparison:")
    print("-" * 80)
    print(
        f"{'Residue ID':^10} {'Calculated':^15} {'Expected':^15} {'Difference':^15} {'Match?':^10} {'% Diff':^15}"
    )
    print("-" * 80)

    matches = []
    mismatches = []

    for idx, exp_idx in enumerate(exp_residues):
        calc_rate = pred_dict[exp_idx]
        exp_rate = exp_dict[exp_idx]
        abs_diff = abs(calc_rate - exp_rate)
        rel_diff = abs_diff / exp_rate * 100 if exp_rate != 0 else float("inf")

        matches_within_tol = np.isclose(calc_rate, exp_rate, atol=1e-5)

        comparison = {
            "index": idx,
            "resid": exp_idx,
            "calculated": calc_rate,
            "expected": exp_rate,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "matches": matches_within_tol,
        }

        if matches_within_tol:
            matches.append(comparison)
        else:
            mismatches.append(comparison)

        print(
            f"{exp_idx:^10d} {calc_rate:^15.6f} {exp_rate:^15.6f} "
            f"{abs_diff:^15.6f} {'✓' if matches_within_tol else '✗':^10} "
            f"{rel_diff:^15.2f}%"
        )

    print("\nSummary:")
    print(f"Total residues compared: {len(res_indexes)}")
    print(f"Matching rates: {len(matches)}")
    print(f"Mismatching rates: {len(mismatches)}")

    if mismatches:
        print("\nMismatched Residues Details:")
        print("-" * 80)
        for mismatch in mismatches:
            print(f"Index {mismatch['index']}:")
            print(f"Residue {mismatch['resid']}:")
            print(f"  Calculated: {mismatch['calculated']:.6f}")
            print(f"  Expected:   {mismatch['expected']:.6f}")
            print(f"  Abs Diff:   {mismatch['abs_diff']:.6f}")
            print(f"  Rel Diff:   {mismatch['rel_diff']:.2f}%")

    # Final assertion with detailed error message
    final_rates = np.array([pred_dict[res_idx] for res_idx in exp_residues])
    print(final_rates)
    print(exp_kints)
    print(final_rates - exp_kints)
    matching_array = np.allclose(final_rates, exp_kints, atol=1e-3)
    print(matching_array)
    assert matching_array, AssertionError(
        f"\nRates comparison failed. Found {len(mismatches)} mismatches out of {len(res_indexes)} residues. "
        "See detailed output above for specific residues and values."
    )


def load_contact_data(data_dir: str, file_prefix: str = "Contacts") -> dict:
    """Load contact data from .tmp files.

    Args:
        data_dir: Directory containing the .tmp files
        file_prefix: Prefix of files to load ("Contacts" or "Hbonds")

    Returns:
        Dictionary mapping residue IDs to their contact values
    """
    import glob
    import os

    # Get all .tmp files matching the prefix
    pattern = os.path.join(data_dir, f"{file_prefix}_chain_0_res_*.tmp")
    tmp_files = glob.glob(pattern)

    contact_data = {}
    for file_path in tmp_files:
        # Extract residue number from filename
        filename = os.path.basename(file_path)
        resid = int(filename.split("res_")[1].split(".tmp")[0])

        # Read first value from file
        try:
            with open(file_path, "r") as f:
                first_line = f.readline().strip()
                if first_line:  # Check if line is not empty
                    contact_value = float(first_line.split()[0])
                    contact_data[resid] = contact_value
        except (ValueError, IndexError):
            print(f"Warning: Could not read contact value from {file_path}")
            continue

    return contact_data


def test_calc_contacts_universe():
    """Test the calculation of contacts against reference data with detailed comparison."""
    topology_path = "./tests/inst/clean/HOIP/train_HOIP_high_rank_1/HOIP_apo697_1_af_sample_127_10000_protonated_first_frame.pdb"
    data_dir = "./tests/inst/clean/HOIP/train_HOIP_high_rank_1"

    universe = mda.Universe(topology_path)

    # Get N atoms for each residue (excluding prolines)
    NH_residue_atom_index: List[Tuple[int, int]] = []
    for residue in universe.residues:
        if residue.resname != "PRO":
            try:
                N_atom = residue.atoms.select_atoms("name N")[0]
                NH_residue_atom_index.append((residue.resid, N_atom.index))
            except IndexError:
                continue

    HN_residue_atom_index: List[Tuple[int, int]] = []
    for residue in universe.residues:
        if residue.resname != "PRO":
            try:
                H_atom = residue.atoms.select_atoms("name H")[0]
                HN_residue_atom_index.append((residue.resid, H_atom.index))
            except IndexError:
                continue
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
    # Calculate contacts
    heavy_contacts = calc_BV_contacts_universe(
        universe=universe,
        residue_atom_index=NH_residue_atom_index,
        contact_selection="heavy",
        radius=6.5,
        switch=False,
    )

    oxygen_contacts = calc_BV_contacts_universe(
        universe=universe,
        residue_atom_index=HN_residue_atom_index,
        contact_selection="oxygen",
        radius=2.4,
        # residue_ignore=(0, 0),
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
        calc_heavy = np.mean(heavy)
        calc_oxygen = np.mean(oxygen)

        ref_heavy = ref_contacts.get(resid, None)
        ref_oxygen = ref_hbonds.get(resid, None)

        if ref_heavy is not None and ref_oxygen is not None:
            heavy_diff = calc_heavy - ref_heavy  # Note: Changed to show direction
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

    # Print summary with direction analysis
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
    test_calculate_intrinsic_rates()

    test_calc_contacts_universe()
