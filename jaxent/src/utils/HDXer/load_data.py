"""
HDX Data Handler - Load and save HDX deuteration data in various formats

This module provides functions to load and save HDX (Hydrogen-Deuterium Exchange)
data in both individual residue and segment formats, with flexible time point handling.

Supported formats:
1. Individual residue format: ResID1 ResID2 frac1 frac2 frac3 ...
2. Segment output format: Res1 Res2 frac1 frac2 frac3 ...
3. dfrac output format: tab-separated fractions with time header
"""

import os
import re
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import Array


def load_HDXer_kints(kint_path: str) -> tuple[Array, list[float]]:
    """
    Loads the intrinsic rates from a .dat file.
    Adjsuts the residue indices to be zero-based for termini exclusions

    Returns:
        kints: jax.numpy array of rates
        topology_list: list of Partial_Topology objects with resids
    """
    rates = []
    topology_list = []
    with open(kint_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            resid, rate = int(parts[0]), float(parts[1])
            rates.append(rate)
            # Assuming chain 'A' as a default for intrinsic rates if not specified in the file
            topology_list.append(resid)
    kints = jnp.array(rates)
    return kints, topology_list


def load_HDXer_dfrac(
    filename: str, expected_timepoints: Optional[List[float]] = None
) -> Tuple[List[List[float]], List[Tuple[int, int]], List[float]]:
    """
    Load HDX data from file, auto-detecting the format.

    Args:
        filename: Path to the data file
        expected_timepoints: Optional list of expected time points for validation

    Returns:
        Tuple of (data, segments, timepoints) where:
        - data: List of deuteration fraction lists for each segment
        - segments: List of (start_residue, end_residue) tuples
        - timepoints: List of time values extracted from header or defaults

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If data format is invalid or timepoints don't match expected
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, "r") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("File is empty")

    # Extract timepoints and determine format
    timepoints, format_type = _parse_header(lines[0])

    # Validate timepoints if provided
    if expected_timepoints is not None:
        if len(timepoints) != len(expected_timepoints):
            raise ValueError(
                f"Expected {len(expected_timepoints)} timepoints, found {len(timepoints)}"
            )

    # Parse data based on detected format
    data = []
    segments = []

    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = re.split(r"\s+", line)
        if len(parts) < 3:  # Need at least res1, res2, and one fraction
            continue

        try:
            res_start = int(float(parts[0]))
            res_end = int(float(parts[1]))
            fractions = [float(val) for val in parts[2:] if val]

            if len(fractions) != len(timepoints):
                print(
                    f"Warning: Line has {len(fractions)} fractions, expected {len(timepoints)}: {line}"
                )
                continue

            segments.append((res_start, res_end))
            data.append(fractions)

        except (ValueError, IndexError) as e:
            print(f"Error parsing line: {line}")
            print(f"Error: {e}")
            continue

    print(f"Loaded {len(data)} segments with {len(timepoints)} timepoints each")
    return data, segments, timepoints


def save_dfrac_file(
    filename: str, data: List[List[float]], timepoints: Optional[List[float]] = None
) -> None:
    """
    Save deuteration fraction data in dfrac format.

    Args:
        filename: Output file path
        data: List of deuteration fraction lists for each segment
        timepoints: List of time values for header (defaults to placeholders)
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Use provided timepoints or create placeholders
    if timepoints is None:
        if data:
            num_points = len(data[0])
            timepoints = [f"T{i + 1}" for i in range(num_points)]
        else:
            timepoints = []

    with open(filename, "w") as f:
        # Write header
        if timepoints:
            header = "#\t" + "\t".join(str(t) for t in timepoints) + "\t times/min\n"
            f.write(header)

        # Write data
        for fractions in data:
            line = "\t".join(f"{frac:.5f}" for frac in fractions) + "\n"
            f.write(line)

    print(f"Saved dfrac file: {filename}")


def save_segs_file(filename: str, segments: List[Tuple[int, int]]) -> None:
    """
    Save segment definitions to file.

    Args:
        filename: Output file path
        segments: List of (start_residue, end_residue) tuples
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    with open(filename, "w") as f:
        for res_start, res_end in segments:
            f.write(f"{res_start} {res_end}\n")

    print(f"Saved segments file: {filename}")


def save_HDXer_segfrac(
    dfrac_filename: str,
    segs_filename: str,
    data: List[List[float]],
    segments: List[Tuple[int, int]],
    timepoints: Optional[List[float]] = None,
) -> None:
    """
    Save both dfrac and segments files.

    Args:
        dfrac_filename: Output path for dfrac file
        segs_filename: Output path for segments file
        data: List of deuteration fraction lists
        segments: List of (start_residue, end_residue) tuples
        timepoints: Optional list of time values
    """
    save_dfrac_file(dfrac_filename, data, timepoints)
    save_segs_file(segs_filename, segments)


def load_segfrac_tonumpy(
    filename: str,
) -> np.ndarray:
    """
    Load data saved from save_dfrac_file or save_segs_file into a numpy array.

    Args:
        filename: Path to the data file

    Returns:
        Numpy array of deuteration fractions
    """
    return pd.read_csv(
        filename,
        # delim_whitespace=True,
        sep=r"\s+",
        comment="#",
        header=None,
    ).to_numpy()


def _parse_header(header_line: str) -> Tuple[List[float], str]:
    """
    Parse header line to extract timepoints and determine format.

    Args:
        header_line: First line of the file

    Returns:
        Tuple of (timepoints, format_type)
    """
    header = header_line.strip()

    # Look for explicit time values in header
    time_pattern = r"Time\s*=\s*([\d\.\s,]+)"
    time_match = re.search(time_pattern, header, re.IGNORECASE)

    if time_match:
        # Extract comma or space separated time values
        time_str = time_match.group(1)
        timepoints = [float(x.strip()) for x in re.split(r"[,\s]+", time_str) if x.strip()]
        format_type = "segment"
    else:
        # Look for "Times /" pattern indicating individual residue format
        if "Times /" in header:
            # Default timepoints for individual residue format
            timepoints = [0.167, 1.0, 10.0, 60.0, 120.0]
            format_type = "individual"
        else:
            # Try to extract numbers from header as potential timepoints
            numbers = re.findall(r"\d+\.?\d*", header)
            if numbers:
                timepoints = [float(x) for x in numbers]
                format_type = "unknown"
            else:
                # Default fallback
                timepoints = [0.167, 1.0, 10.0, 60.0, 120.0]
                format_type = "default"

    return timepoints, format_type
