#!/bin/bash
# truncate_trajectories.sh
# Creates 0.25 us and 0.5 us sub-trajectories from tris_MD and control_MD
# by counting frames from individual replicate files and concatenating.
#
# Frame-count approach: iterates replicate files in the same order used to
# assemble the combined trajectory (rod1→rod2→rod3→coil1→coil2→coil3→
# hairpin1→hairpin2→hairpin3), writes exactly N frames.
# Each replicate is 120ns; 9 replicates = 1080ns total.
#
# Options:
#   --overwrite   Overwrite existing output files (default: skip)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
TRIS_DIR="${BASE_DIR}/data/_aSyn/tris_MD"
CTRL_DIR="${BASE_DIR}/data/_aSyn/control_MD"

OVERWRITE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overwrite) OVERWRITE=1; shift;;
        *) break;;
    esac
done

# Total simulation time across all replicates (ns)
TOTAL_NS=1080

# ---------------------------------------------------------------------------
# truncate_by_frames <label> <top> <out> <target_ns> <rep1> [<rep2> ...]
#   Writes the first N = round(n_total_frames * target_ns / TOTAL_NS) frames
#   from the ordered replicate list, regardless of trajectory time metadata.
# ---------------------------------------------------------------------------
truncate_by_frames() {
    local label="$1"
    local top="$2"
    local out="$3"
    local target_ns="$4"
    shift 4
    # remaining args: replicate XTC files in assembly order

    if [ "${OVERWRITE}" -eq 0 ] && [ -f "${out}" ]; then
        echo "[SKIP] ${label}: ${out} exists (use --overwrite to replace)"
        return 0
    fi

    for f in "$@"; do
        if [ ! -f "${f}" ]; then
            echo "[SKIP] ${label}: replicate file not found: ${f}"
            return 0
        fi
    done

    echo "[RUN]  ${label} -> $(basename "${out}")"

    python3 - "${top}" "${out}" "${target_ns}" "${TOTAL_NS}" "$@" <<'PYEOF'
import sys
import MDAnalysis as mda

top      = sys.argv[1]
out_path = sys.argv[2]
target_ns  = float(sys.argv[3])
total_ns   = float(sys.argv[4])
rep_files  = sys.argv[5:]

# Count frames per replicate (reads trajectory index, no coordinate loading)
frame_counts = []
for f in rep_files:
    u = mda.Universe(top, f)
    frame_counts.append(len(u.trajectory))

total_frames = sum(frame_counts)
n_target = round(total_frames * target_ns / total_ns)
print(f"  Total frames across {len(rep_files)} replicates: {total_frames}")
print(f"  Target: {n_target} frames ({target_ns}/{total_ns} ns)")

# Write n_target frames in replicate order
written = 0
W = None
for rep_file, n_rep in zip(rep_files, frame_counts):
    if written >= n_target:
        break
    u = mda.Universe(top, rep_file)
    n_from = min(n_rep, n_target - written)
    if W is None:
        W = mda.Writer(out_path, n_atoms=u.atoms.n_atoms)
    for ts in u.trajectory[:n_from]:
        W.write(u.atoms)
    written += n_from
    print(f"  {rep_file.rsplit('/', 1)[-1]}: {n_from} frames (running: {written})")

if W is not None:
    W.close()
print(f"  Done: {written} frames -> {out_path}")
PYEOF

    echo "[DONE] ${label}"
    echo ""
}

# Replicate files in the same order used by cluster_aSyn.sh to build *_all_combined.xtc:
# rod_rep1, rod_rep2, rod_rep3, coil_rep1, coil_rep2, coil_rep3,
# hairpin_rep1, hairpin_rep2, hairpin_rep3

echo "=== Truncating tris_MD ==="

TRIS_REPS=(
    "${TRIS_DIR}/tris_rod_rep1_combined.xtc"
    "${TRIS_DIR}/tris_rod_rep2_combined.xtc"
    "${TRIS_DIR}/tris_rod_rep3_combined.xtc"
    "${TRIS_DIR}/tris_coil_rep1_combined.xtc"
    "${TRIS_DIR}/tris_coil_rep2_combined.xtc"
    "${TRIS_DIR}/tris_coil_rep3_combined.xtc"
    "${TRIS_DIR}/tris_hairpin_rep1_combined.xtc"
    "${TRIS_DIR}/tris_hairpin_rep2_combined.xtc"
    "${TRIS_DIR}/tris_hairpin_rep3_combined.xtc"
)

truncate_by_frames "tris_MD 0.25us" \
    "${TRIS_DIR}/md_mol_center_coil.pdb" \
    "${TRIS_DIR}/tris_all_combined_0.25us.xtc" \
    250 \
    "${TRIS_REPS[@]}"

truncate_by_frames "tris_MD 0.5us" \
    "${TRIS_DIR}/md_mol_center_coil.pdb" \
    "${TRIS_DIR}/tris_all_combined_0.5us.xtc" \
    500 \
    "${TRIS_REPS[@]}"

echo "=== Truncating control_MD ==="

CTRL_REPS=(
    "${CTRL_DIR}/control_rod_rep1_combined.xtc"
    "${CTRL_DIR}/control_rod_rep2_combined.xtc"
    "${CTRL_DIR}/control_rod_rep3_combined.xtc"
    "${CTRL_DIR}/control_coil_rep1_combined.xtc"
    "${CTRL_DIR}/control_coil_rep2_combined.xtc"
    "${CTRL_DIR}/control_coil_rep3_combined.xtc"
    "${CTRL_DIR}/control_hairpin_rep1_combined.xtc"
    "${CTRL_DIR}/control_hairpin_rep2_combined.xtc"
    "${CTRL_DIR}/control_hairpin_rep3_combined.xtc"
)

truncate_by_frames "control_MD 0.25us" \
    "${CTRL_DIR}/md.gro.pdb" \
    "${CTRL_DIR}/control_all_combined_0.25us.xtc" \
    250 \
    "${CTRL_REPS[@]}"

truncate_by_frames "control_MD 0.5us" \
    "${CTRL_DIR}/md.gro.pdb" \
    "${CTRL_DIR}/control_all_combined_0.5us.xtc" \
    500 \
    "${CTRL_REPS[@]}"

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
