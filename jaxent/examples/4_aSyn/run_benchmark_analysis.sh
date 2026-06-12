#!/bin/bash
# run_benchmark_analysis.sh
# Runs run_comprehensive_analysis.sh for each of the 6 benchmark combos,
# using the matching per-combo config YAML.
#
# Usage:
#   bash run_benchmark_analysis.sh <benchmark_root_dir> [--combos KEY,KEY,...]
#
# Arguments:
#   benchmark_root_dir   Root output directory from run_benchmark_fitting.sh
#                        (contains subdirs: tris_0.25us, tris_0.5us, etc.)
#   --combos KEY,...     Run only a subset of combos (default: all 6)

set -euo pipefail
cd "$(dirname "$0")" || exit 1

SCRIPT_DIR="$(pwd)"
ANALYSIS_RUNNER="${SCRIPT_DIR}/run_comprehensive_analysis.sh"

BENCHMARK_ROOT=""
COMBOS_STR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --combos)   COMBOS_STR="$2"; shift 2;;
    --combos=*) COMBOS_STR="${1#*=}"; shift;;
    *)
      if [[ -z "$BENCHMARK_ROOT" ]]; then BENCHMARK_ROOT="$1"; fi
      shift;;
  esac
done

if [[ -z "$BENCHMARK_ROOT" ]]; then
  echo "Usage: $0 <benchmark_root_dir> [--combos KEY,KEY,...]"
  echo ""
  echo "Valid combo keys:"
  echo "  tris_0.25us  tris_0.5us  tris_1.0us"
  echo "  control_0.25us  control_0.5us  control_1.0us"
  exit 1
fi

get_config() {
    case "$1" in
        "tris_0.25us")    echo "${SCRIPT_DIR}/config_benchmark_tris_0.25us.yaml" ;;
        "tris_0.5us")     echo "${SCRIPT_DIR}/config_benchmark_tris_0.5us.yaml" ;;
        "tris_1.0us")     echo "${SCRIPT_DIR}/config_benchmark_tris_1.0us.yaml" ;;
        "control_0.25us") echo "${SCRIPT_DIR}/config_benchmark_control_0.25us.yaml" ;;
        "control_0.5us")  echo "${SCRIPT_DIR}/config_benchmark_control_0.5us.yaml" ;;
        "control_1.0us")  echo "${SCRIPT_DIR}/config_benchmark_control_1.0us.yaml" ;;
    esac
}

ALL_COMBO_KEYS=( "tris_0.25us" "tris_0.5us" "tris_1.0us" "control_0.25us" "control_0.5us" "control_1.0us" )

if [[ -n "$COMBOS_STR" ]]; then
    IFS=',' read -r -a COMBO_KEYS <<< "$COMBOS_STR"
else
    COMBO_KEYS=( "${ALL_COMBO_KEYS[@]}" )
fi

PASS=0
FAIL=0
SKIP=0

echo "Benchmark root: ${BENCHMARK_ROOT}"
echo "Combos:         ${COMBO_KEYS[*]}"
echo ""

for KEY in "${COMBO_KEYS[@]}"; do
    COMBO_DIR="${BENCHMARK_ROOT}/${KEY}"
    CONFIG=$(get_config "$KEY")

    echo "=============================="
    echo "[RUN] ${KEY}"
    echo "  config:  ${CONFIG}"
    echo "  results: ${COMBO_DIR}"

    if [[ ! -d "${COMBO_DIR}" ]]; then
        echo "  [SKIP] results directory not found"
        SKIP=$(( SKIP + 1 ))
        echo ""
        continue
    fi
    if [[ ! -f "${CONFIG}" ]]; then
        echo "  [SKIP] config file not found: ${CONFIG}"
        SKIP=$(( SKIP + 1 ))
        echo ""
        continue
    fi

    LOG="${COMBO_DIR}/logs/comprehensive_analysis_rerun.log"
    mkdir -p "${COMBO_DIR}/logs"

    if bash "${ANALYSIS_RUNNER}" --config "${CONFIG}" "${COMBO_DIR}" \
        2>&1 | tee "${LOG}"; then
        echo "[DONE] ${KEY}"
        PASS=$(( PASS + 1 ))
    else
        echo "[FAIL] ${KEY} — see ${LOG}"
        FAIL=$(( FAIL + 1 ))
    fi
    echo ""
done

echo "=============================="
echo "Summary: ${PASS} ok / ${FAIL} failed / ${SKIP} skipped"
