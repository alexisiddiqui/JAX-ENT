import jax.numpy as jnp
import pytest

from jaxent.src.utils.HDXer.load_data import (
    _parse_header,
    load_HDXer_dfrac,
    load_HDXer_kints,
    load_segfrac_tonumpy,
    save_dfrac_file,
    save_HDXer_segfrac,
    save_segs_file,
)

# Example data from the docstring
RESFRACS_CONTENT = """# ResID Deuterated fraction, Times /
1 2	 1.00000  1.00000  1.00000  1.00000  1.00000
2 3	 0.00182  0.01082  0.10312  0.47953  0.72912
3 4	 0.00014  0.00085  0.00842  0.04948  0.09651
4 5	 0.00001  0.00003  0.00033  0.00200  0.00400
5 6	 0.00011  0.00064  0.00637  0.03764  0.07386
6 7	 0.00032  0.00191  0.01895  0.10846  0.20515
7 8	 0.00308  0.01832  0.16885  0.67033  0.89132
8 9	 0.00066  0.00392  0.03851  0.20992  0.37578
9 10	 0.00119  0.00712  0.06897  0.34870  0.57580
10 11	 0.46576  0.97657  1.00000  1.00000  1.00000
11 12	 0.05785  0.30012  0.97180  1.00000  1.00000
12 13	 0.01627  0.09355  0.62550  0.99724  0.99999
13 14	 0.00621  0.03662  0.31139  0.89338  0.98863
14 15	 0.00184  0.01098  0.10449  0.48426  0.73401
15 16	 0.00067  0.00398  0.03910  0.21284  0.38039
16 17	 0.00018  0.00110  0.01091  0.06370  0.12333
"""

SEGFRACS_CONTENT = """# Res1 Res2, Time = 0.167, 1.0, 10.0, 60.0, 120.0 / min
  1   5  0.25049  0.25293  0.27797  0.38275  0.45741
  5   9  0.00104  0.00620  0.05817  0.25659  0.38653
  9  13  0.13527  0.34434  0.66657  0.83649  0.89395
 13  17  0.00222  0.01317  0.11647  0.41354  0.55659
 17  21  0.00597  0.03397  0.21200  0.39691  0.50095
 21  25  0.00560  0.03268  0.25192  0.58453  0.66603
 25  29  0.19206  0.44909  0.51217  0.56689  0.62104
 29  33  0.44381  0.73470  0.98979  1.00000  1.00000
 33  37  0.05043  0.19114  0.33257  0.57542  0.70401
 37  41  0.36582  0.52845  0.68227  0.80005  0.84041
 41  45  0.21282  0.59298  0.98437  1.00000  1.00000
 45  49  0.25514  0.34190  0.52329  0.62699  0.70146
 49  53  0.00491  0.02853  0.21219  0.51374  0.63043
 53  57  0.06024  0.23299  0.50744  0.71581  0.79153
 57  61  0.00020  0.00118  0.01167  0.06687  0.12671
"""

KINTS_CONTENT = """# ResID  Intrinsic rate / min^-1
      2    163227.81828442
      3      1029.89790963
      4      1182.48101311
      5      1129.26060126
      6      1421.65486872
      7      1006.45457572
      8      1357.66985767
      9       696.29644347
     10       481.71944856
     11      1103.55549088
     12       837.13245595
     13      2359.35988855
     14       167.02988712
     15      1053.88731282
     16      2528.09960415
     17       167.02988712
     18       528.19486730
     19      1454.76946418
     20      1454.76946397
     21      2008.14087714
     22       729.11188408
     23      1326.76551855
     24      1558.81356496
     25      1006.45457341
"""


@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def resfracs_file(temp_dir):
    """Creates a temporary file with resfracs content."""
    file_path = temp_dir / "resfracs.dat"
    file_path.write_text(RESFRACS_CONTENT)
    return str(file_path)


@pytest.fixture
def segfracs_file(temp_dir):
    """Creates a temporary file with segfracs content."""
    file_path = temp_dir / "segfracs.dat"
    file_path.write_text(SEGFRACS_CONTENT)
    return str(file_path)


@pytest.fixture
def kints_file(temp_dir):
    """Creates a temporary file with kints content."""
    file_path = temp_dir / "kints.dat"
    file_path.write_text(KINTS_CONTENT)
    return str(file_path)


def test_load_HDXer_dfrac_resfracs(resfracs_file):
    """Test loading resfracs format."""
    data, segments, timepoints = load_HDXer_dfrac(resfracs_file)

    assert len(data) == 16
    assert len(segments) == 16
    assert timepoints == [0.167, 1.0, 10.0, 60.0, 120.0]

    assert segments[0] == (1, 2)
    assert jnp.allclose(jnp.array(data[0]), jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    assert segments[15] == (16, 17)
    assert jnp.allclose(
        jnp.array(data[15]), jnp.array([0.00018, 0.00110, 0.01091, 0.06370, 0.12333])
    )


def test_load_HDXer_dfrac_segfracs(segfracs_file):
    """Test loading segfracs format."""
    data, segments, timepoints = load_HDXer_dfrac(segfracs_file)

    assert len(data) == 15
    assert len(segments) == 15
    assert timepoints == [0.167, 1.0, 10.0, 60.0, 120.0]

    assert segments[0] == (1, 5)
    assert jnp.allclose(
        jnp.array(data[0]), jnp.array([0.25049, 0.25293, 0.27797, 0.38275, 0.45741])
    )
    assert segments[14] == (57, 61)
    assert jnp.allclose(
        jnp.array(data[14]), jnp.array([0.00020, 0.00118, 0.01167, 0.06687, 0.12671])
    )


def test_load_HDXer_kints(kints_file):
    """Test loading intrinsic rates."""
    kints, topology_list = load_HDXer_kints(kints_file)

    assert len(kints) == 24
    assert len(topology_list) == 24
    assert jnp.isclose(kints[0], 163227.81828442)
    assert topology_list[0] == 2
    assert jnp.isclose(kints[-1], 1006.45457341)
    assert topology_list[-1] == 25


def test_load_HDXer_dfrac_file_not_found():
    """Test FileNotFoundError for load_HDXer_dfrac."""
    with pytest.raises(FileNotFoundError):
        load_HDXer_dfrac("non_existent_file.dat")


def test_load_HDXer_dfrac_empty_file(temp_dir):
    """Test ValueError for empty file."""
    empty_file = temp_dir / "empty.dat"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="File is empty"):
        load_HDXer_dfrac(str(empty_file))


def test_load_HDXer_dfrac_mismatched_timepoints(segfracs_file):
    """Test ValueError for mismatched timepoints."""
    with pytest.raises(ValueError, match="Expected 3 timepoints, found 5"):
        load_HDXer_dfrac(segfracs_file, expected_timepoints=[1.0, 2.0, 3.0])


def test_save_dfrac_file(temp_dir):
    """Test saving dfrac file and then loading it back."""
    output_file = temp_dir / "output.dat"
    test_data = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    test_timepoints = [0.5, 5.0, 50.0]

    save_dfrac_file(str(output_file), test_data, test_timepoints)

    loaded_data = load_segfrac_tonumpy(str(output_file))
    assert jnp.allclose(jnp.array(loaded_data), jnp.array(test_data))
    assert len(loaded_data) == 2


def test_save_segs_file(temp_dir):
    """Test saving segments file and then loading it back (manual check as no load_segs_file)."""
    output_file = temp_dir / "segments.txt"
    test_segments = [(1, 5), (6, 10), (11, 15)]

    save_segs_file(str(output_file), test_segments)

    with open(output_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == len(test_segments)
    for i, line in enumerate(lines):
        start, end = map(int, line.strip().split())
        assert (start, end) == test_segments[i]


def test_save_HDXer_segfrac(temp_dir):
    """Test saving both dfrac and segments files."""
    dfrac_output_file = temp_dir / "combined_output.dat"
    segs_output_file = temp_dir / "combined_segments.txt"
    test_data = [[0.11, 0.22, 0.33], [0.44, 0.55, 0.66]]
    test_segments = [(10, 20), (30, 40)]
    test_timepoints = [1.0, 10.0, 100.0]

    save_HDXer_segfrac(
        str(dfrac_output_file), str(segs_output_file), test_data, test_segments, test_timepoints
    )

    # Verify dfrac file
    loaded_data = load_segfrac_tonumpy(str(dfrac_output_file))
    assert jnp.allclose(jnp.array(loaded_data), jnp.array(test_data))
    assert len(loaded_data) == len(test_segments)

    # Verify segments file
    with open(segs_output_file, "r") as f:
        lines = f.readlines()

    assert len(lines) == len(test_segments)
    for i, line in enumerate(lines):
        start, end = map(int, line.strip().split())
        assert (start, end) == test_segments[i]


def test_parse_header_segment_format():
    """Test _parse_header with segment format."""
    header = "# Res1 Res2, Time = 0.167, 1.0, 10.0, 60.0, 120.0 / min"
    timepoints, format_type = _parse_header(header)
    assert timepoints == [0.167, 1.0, 10.0, 60.0, 120.0]
    assert format_type == "segment"


def test_parse_header_individual_format():
    """Test _parse_header with individual format."""
    header = "# ResID Deuterated fraction, Times /"
    timepoints, format_type = _parse_header(header)
    assert timepoints == [0.167, 1.0, 10.0, 60.0, 120.0]
    assert format_type == "individual"


def test_parse_header_unknown_format_with_numbers():
    """Test _parse_header with unknown format but numbers in header."""
    header = "Some random header 1.0 2.0 3.0"
    timepoints, format_type = _parse_header(header)
    assert timepoints == [1.0, 2.0, 3.0]
    assert format_type == "unknown"


def test_parse_header_default_fallback():
    """Test _parse_header with no recognizable format or numbers."""
    header = "Just some text"
    timepoints, format_type = _parse_header(header)
    assert timepoints == [0.167, 1.0, 10.0, 60.0, 120.0]
    assert format_type == "default"
