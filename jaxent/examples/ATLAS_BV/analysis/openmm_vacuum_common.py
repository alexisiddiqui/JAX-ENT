"""OpenMM-independent helpers for checkpoint-23 vacuum rescoring."""

from __future__ import annotations

from collections import deque

import numpy as np


def canonical_atom_name(name: str) -> str:
    """Treat PDB leading hydrogen indices and GROMACS suffix indices as equivalent."""
    value = str(name).strip().upper()
    if value and value[0].isdigit():
        return value[1:] + value[0]
    return value


def unwrap_positions(
    positions_angstrom: np.ndarray,
    box_angstrom: np.ndarray,
    adjacency: list[list[int]],
) -> np.ndarray:
    """Make every bonded component whole using a triclinic minimum-image traversal."""
    raw = np.asarray(positions_angstrom, dtype=np.float64)
    box = np.asarray(box_angstrom, dtype=np.float64)
    inverse = np.linalg.inv(box)
    output = np.empty_like(raw)
    visited = np.zeros(len(raw), dtype=bool)
    component_roots = []
    for root in range(len(raw)):
        if visited[root]:
            continue
        component_roots.append(root)
        visited[root] = True
        output[root] = raw[root]
        queue = deque([root])
        while queue:
            parent = queue.popleft()
            for child in adjacency[parent]:
                if visited[child]:
                    continue
                delta_fractional = (raw[child] - raw[parent]) @ inverse
                delta_fractional -= np.round(delta_fractional)
                output[child] = output[parent] + delta_fractional @ box
                visited[child] = True
                queue.append(child)
    anchor = output[component_roots[0]]
    for root in component_roots[1:]:
        delta_fractional = (output[root] - anchor) @ inverse
        shift = np.round(delta_fractional) @ box
        component = np.zeros(len(raw), dtype=bool)
        component[root] = True
        queue = deque([root])
        while queue:
            parent = queue.popleft()
            for child in adjacency[parent]:
                if not component[child]:
                    component[child] = True
                    queue.append(child)
        output[component] -= shift
    return output
