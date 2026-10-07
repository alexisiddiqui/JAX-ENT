import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "examples/2_CrossValidation/data"
SCRIPT_PATH = DATA_DIR / "extract_data_ValDX.py"
MOPRP_DIR = DATA_DIR / "_MoPrP"
STRUCTURE_PATH = (
    DATA_DIR
    / "MoPrP109_s20_r1_msa1-127_n12700_do1_20260904_191954_protonated_max_plddt_1627.pdb"
)
STRUCTURE_101_PATH = DATA_DIR / "MoPrP_max_plddt_4334.pdb"
# One-based moprp.seq positions of each moprp.list peptide string (list starts are
# zero-based offsets; YMLGSA is positions 5-10).  MoPrP101 PDB IDs equal these;
# MoPrP109 IDs are one higher (extra N-terminal Gly).
EXPECTED_SEQUENCE_SEGMENTS = np.asarray(
    [
        [5, 10],
        [11, 24],
        [11, 26],
        [27, 31],
        [32, 45],
        [33, 40],
        [60, 74],
        [68, 75],
        [75, 79],
        [75, 82],
        [75, 84],
        [82, 101],
        [83, 90],
        [95, 101],
    ]
)
EXPECTED_SEGMENTS = EXPECTED_SEQUENCE_SEGMENTS + 1


@pytest.fixture(scope="module")
def extractor():
    name = "extract_data_valdx_for_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_pdb(path: Path, chains: dict[str, list[tuple[int, str]]]) -> None:
    lines = []
    serial = 1
    for chain_id, residues in chains.items():
        for residue_id, residue_name in residues:
            lines.append(
                f"ATOM  {serial:5d}  CA  {residue_name:>3s} {chain_id:1s}"
                f"{residue_id:4d}    {serial:8.3f}{0.0:8.3f}{0.0:8.3f}"
                "  1.00 20.00           C  \n"
            )
            serial += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def test_moprp_sequence_maps_to_supplied_structure(extractor):
    sequence = (MOPRP_DIR / "moprp.seq").read_text()

    mapping, chain_id, span = extractor.build_sequence_to_structure_map(
        STRUCTURE_PATH, sequence
    )

    assert chain_id == "A"
    assert span == (2, 102)
    assert len(mapping) == 101
    assert mapping[1] == 2
    assert mapping[101] == 102


def test_full_extraction_maps_segments_and_positive_pfactors(extractor, tmp_path):
    data_dir = tmp_path / "_MoPrP"
    data_dir.mkdir()
    for filename in ("moprp.dexp", "moprp.list", "median.pfact"):
        shutil.copy2(MOPRP_DIR / filename, data_dir / filename)

    extractor.extract_data(
        STRUCTURE_PATH,
        MOPRP_DIR / "moprp.seq",
        data_dir=data_dir,
    )

    generated_segments = np.loadtxt(data_dir / "_output/MoPrP_segments.txt", dtype=int)
    np.testing.assert_array_equal(generated_segments, EXPECTED_SEGMENTS)

    source_pf = np.loadtxt(MOPRP_DIR / "median.pfact")
    expected_pf = source_pf[source_pf[:, 1] > 0].copy()
    expected_pf[:, 0] += 1
    generated_pf = np.loadtxt(data_dir / "_output/MoPrP_pfactors.dat")
    np.testing.assert_allclose(generated_pf, expected_pf, rtol=0.0, atol=0.0)

    assert (data_dir / "_output/MoPrP_dfrac.dat").read_bytes() == (
        MOPRP_DIR / "_output/MoPrP_dfrac.dat"
    ).read_bytes()


def test_segments_reproduce_listed_peptide_strings(extractor, tmp_path):
    data_dir = tmp_path / "_MoPrP"
    data_dir.mkdir()
    for filename in ("moprp.dexp", "moprp.list", "median.pfact"):
        shutil.copy2(MOPRP_DIR / filename, data_dir / filename)

    extractor.extract_data(STRUCTURE_101_PATH, MOPRP_DIR / "moprp.seq", data_dir=data_dir)

    sequence = (MOPRP_DIR / "moprp.seq").read_text().strip()
    segments = np.loadtxt(data_dir / "_output/MoPrP_segments.txt", dtype=int)
    np.testing.assert_array_equal(segments, EXPECTED_SEQUENCE_SEGMENTS)
    peptides = [line.split()[3] for line in (MOPRP_DIR / "moprp.list").read_text().splitlines()]
    assert [sequence[start - 1 : end] for start, end in segments] == peptides


def test_locate_peptide_rejects_inconsistent_listed_start(extractor):
    assert extractor.locate_peptide("GLGGYMLGSA", "YMLGSA", 4) == (5, 10)
    with pytest.raises(ValueError, match="listed at"):
        extractor.locate_peptide("GLGGYMLGSA", "YMLGSA", 3)
    with pytest.raises(ValueError, match="not found"):
        extractor.locate_peptide("GLGGYMLGSA", "WWW", 0)


def test_mapping_uses_actual_nonconsecutive_pdb_residue_ids(extractor, tmp_path):
    structure = tmp_path / "nonconsecutive.pdb"
    _write_pdb(structure, {"A": [(10, "ALA"), (20, "CYS"), (42, "ASP")]})

    mapping, chain_id, span = extractor.build_sequence_to_structure_map(
        structure, "ACD"
    )

    assert chain_id == "A"
    assert span == (10, 42)
    assert mapping == {1: 10, 2: 20, 3: 42}


def test_ambiguous_match_requires_chain_selection(extractor, tmp_path):
    structure = tmp_path / "ambiguous.pdb"
    residues = [(1, "ALA"), (2, "CYS"), (3, "ASP")]
    _write_pdb(structure, {"A": residues, "B": residues})

    with pytest.raises(ValueError, match="ambiguous"):
        extractor.build_sequence_to_structure_map(structure, "ACD")

    mapping, chain_id, _ = extractor.build_sequence_to_structure_map(
        structure, "ACD", chain="B"
    )
    assert chain_id == "B"
    assert mapping == {1: 1, 2: 2, 3: 3}


def test_missing_match_and_invalid_source_coordinate_fail(extractor, tmp_path):
    structure = tmp_path / "missing.pdb"
    _write_pdb(structure, {"A": [(1, "ALA"), (2, "CYS"), (3, "ASP")]})

    with pytest.raises(ValueError, match="not found exactly"):
        extractor.build_sequence_to_structure_map(structure, "AAA")
    with pytest.raises(ValueError, match="outside 1-3"):
        extractor.map_hdx_position(4, {1: 10, 2: 20, 3: 42}, "test")
