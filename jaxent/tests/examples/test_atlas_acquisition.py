from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "examples/ATLAS_BV/data/select_systems.py"
SPEC = importlib.util.spec_from_file_location("atlas_select_systems", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(system_id: str, **updates: str) -> dict[str, str]:
    base = {
        "PDB": system_id,
        "length": "100",
        "PDB_resolution": "2.0",
        "contact_ligand": "False",
        "contact_nucleotide": "False",
        "avg_RMSF": "1.0",
        "avg_gyration": "12.0",
        "CATH_class": "Mainly Alpha",
        "non_redundant_protein": "True",
    }
    base.update(updates)
    return base


def test_selection_filters_and_inclusive_length_boundaries() -> None:
    rows = [
        row("1aaa_A", length="60", avg_RMSF="0.5"),
        row("1aab_A", length="250", avg_RMSF="1.0"),
        row("1aac_A", length="59"),
        row("1aad_A", contact_ligand="True"),
        row("1aae_A", contact_nucleotide="1"),
        row("1aaf_A", non_redundant_protein="no"),
        row("1aag_A", avg_RMSF="2.0"),
    ]
    selected, cutpoints = MODULE.select_rows(rows)
    assert [item["system_id"] for item in selected] == ["1aaa_A", "1aab_A", "1aag_A"]
    assert [item["rmsf_tercile"] for item in selected] == ["low", "middle", "high"]
    assert cutpoints == pytest.approx((5 / 6, 4 / 3))
    assert all(item["n_frames"] == 3003 for item in selected)


def test_unknown_boolean_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown boolean"):
        MODULE.select_rows([row("1aaa_A", contact_ligand="perhaps")])


def test_duplicate_system_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        MODULE.select_rows([row("1aaa_A"), row("1aaa_A")])


def test_cath_class_is_normalised_and_deduplicated() -> None:
    assert MODULE.normalise_cath_class('"Mainly Alpha", "Mainly Alpha"') == "Mainly Alpha"
    assert MODULE.normalise_cath_class('"Mainly Alpha", "Alpha Beta"') == "Mainly Alpha|Alpha Beta"
    assert MODULE.normalise_cath_class('"-"') == "unclassified"
