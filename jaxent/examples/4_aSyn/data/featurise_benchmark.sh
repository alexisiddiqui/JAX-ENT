#!/bin/bash
# featurise_benchmark.sh
# Featurises all 5 new (md_source x time) combinations for the ensemble-size benchmark.
# tris_MD 1.0us features already exist in data/_aSyn/tris_MD/features/; only the
# 5 new combos are run here.
#
# Requires jaxent-featurise on PATH (activate .venv first).
# Run from the 4_aSyn/data/ directory or any parent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
TRIS_DIR="${BASE_DIR}/data/_aSyn/tris_MD"
CTRL_DIR="${BASE_DIR}/data/_aSyn/control_MD"

TRIS_TOP="${TRIS_DIR}/md_mol_center_coil.pdb"
CTRL_TOP="${CTRL_DIR}/md.gro.pdb"

run_featurise() {
    local label="$1"
    local top="$2"
    local traj="$3"
    local outdir="$4"
    local name="$5"

    if [ ! -f "${traj}" ]; then
        echo "[SKIP] ${label}: trajectory not found at ${traj}"
        return 0
    fi
    if [ -f "${outdir}/features.npz" ] && [ -f "${outdir}/${name}.npz" ]; then
        echo "[SKIP] ${label}: outputs already exist in ${outdir}"
        return 0
    fi

    echo "[RUN]  ${label}"
    echo "       top:    ${top}"
    echo "       traj:   ${traj}"
    echo "       outdir: ${outdir}"
    echo "       name:   ${name}"

    mkdir -p "${outdir}"
    local t0; t0=$(date +%s)

    jaxent-featurise \
        --top_path "${top}" \
        --trajectory_path "${traj}" \
        --output_dir "${outdir}" \
        --name "${name}" \
        bv --switch --peptide_trim 0

    local t1; t1=$(date +%s)
    echo "[DONE] ${label} ($(( t1 - t0 ))s)"
    echo ""
}

echo "=== Featurising tris_MD sub-trajectories ==="

run_featurise \
    "tris_MD 0.25us" \
    "${TRIS_TOP}" \
    "${TRIS_DIR}/tris_all_combined_0.25us.xtc" \
    "${TRIS_DIR}/features_0.25us" \
    "tris_featurised_0.25us"

run_featurise \
    "tris_MD 0.5us" \
    "${TRIS_TOP}" \
    "${TRIS_DIR}/tris_all_combined_0.5us.xtc" \
    "${TRIS_DIR}/features_0.5us" \
    "tris_featurised_0.5us"

echo "=== Featurising control_MD trajectories ==="

run_featurise \
    "control_MD 0.25us" \
    "${CTRL_TOP}" \
    "${CTRL_DIR}/control_all_combined_0.25us.xtc" \
    "${CTRL_DIR}/features_0.25us" \
    "control_featurised_0.25us"

run_featurise \
    "control_MD 0.5us" \
    "${CTRL_TOP}" \
    "${CTRL_DIR}/control_all_combined_0.5us.xtc" \
    "${CTRL_DIR}/features_0.5us" \
    "control_featurised_0.5us"

run_featurise \
    "control_MD 1.0us" \
    "${CTRL_TOP}" \
    "${CTRL_DIR}/control_all_combined.xtc" \
    "${CTRL_DIR}/features" \
    "control_featurised"

echo "=== Featurisation complete. Output summary: ==="
for d in \
    "${TRIS_DIR}/features_0.25us" \
    "${TRIS_DIR}/features_0.5us" \
    "${CTRL_DIR}/features_0.25us" \
    "${CTRL_DIR}/features_0.5us" \
    "${CTRL_DIR}/features"; do
    echo "  ${d}:"
    ls -lh "${d}/"*.npz "${d}/"*.json 2>/dev/null || echo "    (empty or missing)"
done
