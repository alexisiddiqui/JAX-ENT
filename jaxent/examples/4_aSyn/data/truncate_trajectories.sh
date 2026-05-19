#!/bin/bash
# truncate_trajectories.sh
# Creates 0.25 us and 0.5 us sub-trajectories from tris_MD and control_MD.
# Requires GROMACS (gmx trjconv). Run from the 4_aSyn/data/ directory or
# any parent — paths are resolved relative to this script's location.
# gmx trjconv prompts for group selection; "0" selects "System".

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
TRIS_DIR="${BASE_DIR}/data/_aSyn/tris_MD"
CTRL_DIR="${BASE_DIR}/data/_aSyn/control_MD"

TRIS_TOP="${TRIS_DIR}/md_mol_center_coil.pdb"
CTRL_TOP="${CTRL_DIR}/md.gro.pdb"

TRIS_FULL="${TRIS_DIR}/tris_all_combined.xtc"
CTRL_FULL="${CTRL_DIR}/control_all_combined.xtc"

# End times in picoseconds (1 us = 1,000,000 ps)
TIME_025PS=250000
TIME_050PS=500000

trjconv_truncate() {
    local label="$1"
    local full_xtc="$2"
    local top="$3"
    local out_xtc="$4"
    local end_ps="$5"

    if [ ! -f "${full_xtc}" ]; then
        echo "[SKIP] ${label}: source trajectory not found at ${full_xtc}"
        return 0
    fi

    echo "[RUN]  ${label} -> $(basename "${out_xtc}")"
    echo "0" | gmx trjconv \
        -f "${full_xtc}" \
        -s "${top}" \
        -o "${out_xtc}" \
        -b 0 -e "${end_ps}" \
        -quiet 2>&1 | tail -3
    echo "[DONE] ${label}"
    echo ""
}

echo "=== Truncating tris_MD ==="
trjconv_truncate \
    "tris_MD 0.25us" \
    "${TRIS_FULL}" "${TRIS_TOP}" \
    "${TRIS_DIR}/tris_all_combined_0.25us.xtc" \
    "${TIME_025PS}"

trjconv_truncate \
    "tris_MD 0.5us" \
    "${TRIS_FULL}" "${TRIS_TOP}" \
    "${TRIS_DIR}/tris_all_combined_0.5us.xtc" \
    "${TIME_050PS}"

echo "=== Truncating control_MD ==="
trjconv_truncate \
    "control_MD 0.25us" \
    "${CTRL_FULL}" "${CTRL_TOP}" \
    "${CTRL_DIR}/control_all_combined_0.25us.xtc" \
    "${TIME_025PS}"

trjconv_truncate \
    "control_MD 0.5us" \
    "${CTRL_FULL}" "${CTRL_TOP}" \
    "${CTRL_DIR}/control_all_combined_0.5us.xtc" \
    "${TIME_050PS}"

echo "=== Output files ==="
for f in \
    "${TRIS_DIR}/tris_all_combined_0.25us.xtc" \
    "${TRIS_DIR}/tris_all_combined_0.5us.xtc" \
    "${CTRL_DIR}/control_all_combined_0.25us.xtc" \
    "${CTRL_DIR}/control_all_combined_0.5us.xtc"; do
    if [ -f "$f" ]; then
        ls -lh "$f"
    else
        echo "  MISSING: $f"
    fi
done
