#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PULLDOWN_DIR="${BASE_DIR}/pulldown"
LIGAND_SDF="${BASE_DIR}/85H_ideal.sdf"
SCRIPT="${BASE_DIR}/vac_minimise.py"

skipped=0; done_count=0; failed=0

for condition_dir in "${PULLDOWN_DIR}"/cluster_*; do
    [[ -d "${condition_dir}" ]] || continue

    combined_dir="${condition_dir}/combined_outputs"
    if [[ ! -d "${combined_dir}" ]]; then
        echo "[SKIP] No combined_outputs: ${condition_dir##*/}"
        (( ++skipped )); continue
    fi

    mapfile -t pdbs < <(ls "${combined_dir}"/complex_rank*.pdb 2>/dev/null || true)
    if [[ ${#pdbs[@]} -eq 0 ]]; then
        echo "[SKIP] No complex_rank*.pdb: ${condition_dir##*/}"
        (( ++skipped )); continue
    fi

    minimised_dir="${combined_dir}/minimised"
    if [[ -d "${minimised_dir}" ]] && compgen -G "${minimised_dir}/*_minimised.pdb" > /dev/null 2>&1; then
        echo "[DONE] Already minimised: ${condition_dir##*/}"
        (( ++done_count )); continue
    fi

    echo "[RUN]  ${condition_dir##*/} — ${#pdbs[@]} complexes"
    if python "${SCRIPT}" \
            --input-dir "${combined_dir}" \
            --ligand-sdf "${LIGAND_SDF}"; then
        (( ++done_count ))
    else
        echo "[FAIL] ${condition_dir##*/}"
        (( ++failed ))
    fi
done

echo ""
echo "=== Summary: done=${done_count}  skipped=${skipped}  failed=${failed} ==="
