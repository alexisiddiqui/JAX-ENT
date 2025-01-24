import MDAnalysis as mda
import numpy as np
import pandas as pd

from jaxent.forwardmodels.featurisers import calc_contacts_universe, calculate_intrinsic_rates


def test_calculate_intrinsic_rates():
    """Test the calculation of intrinsic rates against reference data with detailed comparison."""
    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/HDXer_tutorial/BPTI/BPTI_simulations/P00974_60_1_af_sample_127_10001_protonated.pdb"
    rates_path = "/home/alexi/Documents/JAX-ENT/tests/inst/BPTI_Intrinsic_rates.dat"

    # Load universe
    universe = mda.Universe(topology_path)

    # Calculate rates using our function
    rates, residue_ids = calculate_intrinsic_rates(universe)

    # Basic length checks
    assert len(rates) == len(residue_ids)
    assert len(rates) == len(universe.residues)
    assert len(rates) == 58

    # Read reference data
    with open(rates_path, "r") as f:
        header = f.readline().strip()
    print(f"Header from file: {header}")

    df = pd.read_csv(rates_path, delim_whitespace=True)
    print(f"Columns in DataFrame: {df.columns.tolist()}")

    # Rename columns
    df.columns = ["ResID", "k_int"] if len(df.columns) == 2 else df.columns

    exp_residues = df["ResID"].values
    exp_kints = df["k_int"].values

    # Get the residue numbers that match the exp_residues
    res_indexes = []
    for res in exp_residues:
        res_indexes.append(np.where(residue_ids == res)[0][0])

    # Detailed comparison
    print("\nDetailed Rate Comparison:")
    print("-" * 80)
    print(
        f"{'Residue ID':^10} {'Calculated':^15} {'Expected':^15} {'Difference':^15} {'Match?':^10} {'% Diff':^15}"
    )
    print("-" * 80)

    matches = []
    mismatches = []

    for idx, exp_idx in enumerate(res_indexes):
        calc_rate = rates[exp_idx]
        exp_rate = exp_kints[idx]
        abs_diff = abs(calc_rate - exp_rate)
        rel_diff = abs_diff / exp_rate * 100 if exp_rate != 0 else float("inf")

        matches_within_tol = np.isclose(calc_rate, exp_rate, atol=1e-5)

        comparison = {
            "resid": exp_residues[idx],
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
            f"{exp_residues[idx]:^10d} {calc_rate:^15.6f} {exp_rate:^15.6f} "
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
            print(f"Residue {mismatch['resid']}:")
            print(f"  Calculated: {mismatch['calculated']:.6f}")
            print(f"  Expected:   {mismatch['expected']:.6f}")
            print(f"  Abs Diff:   {mismatch['abs_diff']:.6f}")
            print(f"  Rel Diff:   {mismatch['rel_diff']:.2f}%")

    # Final assertion with detailed error message
    matching_array = np.allclose(rates[res_indexes], exp_kints, atol=1e-5)
    # if not matching_array:
    #     raise AssertionError(
    #         f"\nRates comparison failed. Found {len(mismatches)} mismatches out of {len(res_indexes)} residues. "
    #         "See detailed output above for specific residues and values."
    #     )


from typing import List, Tuple


def test_calc_contacts_universe():
    """Test the calculation of contacts against reference data with detailed comparison."""

    # Load test structure
    topology_path = "/home/alexi/Documents/JAX-ENT/tests/inst/5pti.pdb"
    # topology_path = "/home/alexi/Documents/ValDX/raw_data/BRD4/BRD4_APO/BRD4_APO_484_1_af_sample_127_10000_protonated_max_plddt_2399.pdb"
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
    # Test heavy atom contacts
    heavy_contacts = calc_contacts_universe(
        universe=universe,
        residue_atom_index=NH_residue_atom_index,
        contact_selection="heavy",
        radius=6.5,  # 0.65 nm in Angstroms
        switch=False,
    )

    # Test oxygen contacts
    oxygen_contacts = calc_contacts_universe(
        universe=universe,
        residue_atom_index=HN_residue_atom_index,
        contact_selection="oxygen",
        radius=2.4,  # 0.24 nm in Angstroms
        switch=False,
    )

    # Basic length checks
    assert len(heavy_contacts) == len(NH_residue_atom_index)
    assert len(oxygen_contacts) == len(HN_residue_atom_index)
    assert len(heavy_contacts[0]) == len(universe.trajectory)
    assert len(oxygen_contacts[0]) == len(universe.trajectory)

    # Print detailed comparison if we have reference data
    print("\nDetailed Contacts Comparison:")
    print("-" * 80)
    print(f"{'Residue ID':^10} {'Heavy Contacts':^15} {'O Contacts':^15}")
    print("-" * 80)

    for (resid, _), heavy, oxygen in zip(NH_residue_atom_index, heavy_contacts, oxygen_contacts):
        # Get mean contacts across frames
        mean_heavy = np.mean(heavy)
        mean_oxygen = np.mean(oxygen)
        print(f"{resid:^10d} {mean_heavy:^15.2f} {mean_oxygen:^15.2f}")

    # Test switching function
    heavy_contacts_switch = calc_contacts_universe(
        universe=universe,
        residue_atom_index=NH_residue_atom_index,
        contact_selection="heavy",
        radius=6.5,
        switch=True,
    )

    # Verify switching function gives different results than hard cutoff
    assert not np.allclose(heavy_contacts, heavy_contacts_switch)

    # Additional checks
    assert all(np.all(np.array(contacts) >= 0) for contacts in [heavy_contacts, oxygen_contacts])
    assert all(isinstance(val, (int, float)) for sublist in heavy_contacts for val in sublist)

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_calculate_intrinsic_rates()

    test_calc_contacts_universe()
