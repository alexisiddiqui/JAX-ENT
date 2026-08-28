import MDAnalysis as mda

from jaxent.src.interfaces.topology.mda_adapter import mda_TopologyAdapter


def test_empty_chain_id_falls_back_to_nonempty_segid():
    universe = mda.Universe.empty(
        1,
        n_residues=1,
        n_segments=1,
        atom_resindex=[0],
        residue_segindex=[0],
    )
    universe.add_TopologyAttr("names", ["N"])
    universe.add_TopologyAttr("chainIDs", [""])
    universe.add_TopologyAttr("segids", ["SYSTEM"])

    chain_id = mda_TopologyAdapter._get_chain_id(universe.atoms[0])
    selection, _ = mda_TopologyAdapter._build_chain_selection_string(universe, chain_id)

    assert chain_id == "SYSTEM"
    assert selection == "segid SYSTEM"
    assert len(universe.select_atoms(selection)) == 1
