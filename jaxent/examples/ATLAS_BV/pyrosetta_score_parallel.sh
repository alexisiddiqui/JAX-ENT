#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workers="${PYROSETTA_WORKERS:-6}"
mode="pilot"
if [[ "${1:-}" == "--full" ]]; then
  mode="full"
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--full]" >&2
  exit 2
fi

if [[ "${mode}" == "full" ]]; then
  systems=()
  while IFS=, read -r system_id _; do
    [[ "${system_id}" == "system_id" ]] && continue
    systems+=("${system_id}")
  done < "${example_dir}/data/systems.csv"
else
  systems=(2ad6_D 1pch_A 6fub_B 1u6t_A 7bwf_B 6yhu_B 1dvo_A 2in8_A 5bnh_A 4qmd_A 1k7j_A 1ah7_A)
fi

run_system() {
  local system_id="$1"
  "${example_dir}/pyrosetta_score_env.sh" \
    "${example_dir}/analysis/pyrosetta_energy_score_checkpoint24.py" \
    score --systems "${system_id}"
}
export -f run_system
export example_dir

printf '%s\n' "${systems[@]}" | xargs -n 1 -P "${workers}" bash -c 'run_system "$1"' _
