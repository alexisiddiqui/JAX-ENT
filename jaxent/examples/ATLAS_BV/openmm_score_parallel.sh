#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workers="${OPENMM_WORKERS:-4}"
threads="${OPENMM_THREADS_PER_WORKER:-1}"
platform="${OPENMM_PLATFORM:-OpenCL}"
systems=()
while IFS=, read -r system_id _; do
  [[ "${system_id}" == "system_id" ]] && continue
  systems+=("${system_id}")
done < "${example_dir}/data/systems.csv"

run_system() {
  local system_id="$1"
  local platform_args=(--platform "${platform}")
  if [[ "${platform}" == "CPU" ]]; then
    platform_args+=(--threads "${threads}")
  fi
  "${example_dir}/openmm_vacuum_env.sh" run python \
    "${example_dir}/analysis/openmm_vacuum_score_checkpoint23.py" \
    score --systems "${system_id}" "${platform_args[@]}" --total-only
}
export -f run_system
export example_dir threads platform

printf '%s\n' "${systems[@]}" | xargs -n 1 -P "${workers}" bash -c 'run_system "$1"' _
