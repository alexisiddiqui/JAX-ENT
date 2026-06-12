#!/bin/bash
# run_benchmark_fitting.sh
# Runs MaxEnt BV optimisation for all 6 (md_source x time) benchmark combos,
# fitting against Tris_only condition only. Wall-clock time is measured for the
# fitting phase only; run_comprehensive_analysis.sh is called AFTER the timer
# stops for each combo.
#
# Prerequisites:
#   - data/truncate_trajectories.sh completed (4 new XTCs)
#   - data/featurise_benchmark.sh completed (5 new feature dirs)
#   - data/cluster_benchmark_ensembles.py completed (6 cluster dirs)
#   - .venv activated (jaxent available, optimise_aSyn_conditions_BV.py on path)
#
# Usage:
#   bash fitting/run_benchmark_fitting.sh [OPTIONS]
#
# Options:
#   -j|--jobs N              Parallel jobs per combo sweep (default: 4)
#   --n-steps N              Optimisation steps per run (default: 5000)
#   --maxent-values A,B,...  Comma-separated maxent values
#   --bvreg-values A,B,...   Comma-separated BV reg values
#   --bv-reg-losses L1,...   Comma-separated BV loss types (default: L1)
#   --split-types S,...      Comma-separated split types (default: sequence_cluster)
#   --combos KEY,...         Subset of combos to run (default: all 6)
#   --skip-analysis          Skip run_comprehensive_analysis.sh after fitting

set -euo pipefail
cd "$(dirname "$0")" || exit

DIR_WD="$(pwd)"
BASE_DIR="${DIR_WD}/.."
DATA_DIR="${BASE_DIR}/data/_aSyn"
ANALYSIS_RUNNER="${BASE_DIR}/run_comprehensive_analysis.sh"

# ── sweep hyperparameters (match run_maxent_parallel_BV_aSyn_conditions.sh) ──
PARALLEL_JOBS=10
MAXENT_VALUES_STR="1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,1000"
BV_REG_VALUES_STR="0.5,1.0"
BV_REG_LOSSES_STR="L1"
N_STEPS=5000
INITIAL_STEPS=0
INITIAL_LR=1.0
LEARNING_RATE=1.0
EMA_ALPHA=0.5
FORWARD_MODEL_SCALING=1000.0
MODEL_PARAMETERS_LR_SCALE=1.0
SPLIT_TYPES_STR="sequence_cluster"
RUN_ANALYSIS=1

# Subset of combo keys to run (empty = all 6)
COMBOS_STR=""

# ── parse CLI args ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)           PARALLEL_JOBS="$2"; shift 2;;
    --jobs=*)            PARALLEL_JOBS="${1#*=}"; shift;;
    --n-steps)           N_STEPS="$2"; shift 2;;
    --n-steps=*)         N_STEPS="${1#*=}"; shift;;
    --maxent-values)     MAXENT_VALUES_STR="$2"; shift 2;;
    --maxent-values=*)   MAXENT_VALUES_STR="${1#*=}"; shift;;
    --bvreg-values)      BV_REG_VALUES_STR="$2"; shift 2;;
    --bvreg-values=*)    BV_REG_VALUES_STR="${1#*=}"; shift;;
    --bv-reg-losses)     BV_REG_LOSSES_STR="$2"; shift 2;;
    --bv-reg-losses=*)   BV_REG_LOSSES_STR="${1#*=}"; shift;;
    --split-types)       SPLIT_TYPES_STR="$2"; shift 2;;
    --split-types=*)     SPLIT_TYPES_STR="${1#*=}"; shift;;
    --combos)            COMBOS_STR="$2"; shift 2;;
    --combos=*)          COMBOS_STR="${1#*=}"; shift;;
    --skip-analysis)     RUN_ANALYSIS=0; shift;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "  -j|--jobs N              Parallel jobs (default: 4)"
      echo "  --n-steps N              Steps per run (default: 5000)"
      echo "  --maxent-values A,B,...  Maxent sweep values"
      echo "  --bvreg-values A,B,...   BV reg sweep values"
      echo "  --bv-reg-losses L,...    BV reg loss types (default: L1)"
      echo "  --split-types S,...      Split types (default: sequence_cluster)"
      echo "  --combos K,...           Run only specific combo keys"
      echo "  --skip-analysis          Skip post-fitting analysis"
      exit 0;;
    *) break;;
  esac
done

# ── combo definitions ──────────────────────────────────────────────────────
ALL_COMBO_KEYS=( "tris_0.25us" "tris_0.5us" "tris_1.0us" "control_0.25us" "control_0.5us" "control_1.0us" )

get_features_dir() {
    case "$1" in
        "tris_0.25us")    echo "${DATA_DIR}/tris_MD/features_0.25us" ;;
        "tris_0.5us")     echo "${DATA_DIR}/tris_MD/features_0.5us" ;;
        "tris_1.0us")     echo "${DATA_DIR}/tris_MD/features" ;;
        "control_0.25us") echo "${DATA_DIR}/control_MD/features_0.25us" ;;
        "control_0.5us")  echo "${DATA_DIR}/control_MD/features_0.5us" ;;
        "control_1.0us")  echo "${DATA_DIR}/control_MD/features" ;;
    esac
}

get_config() {
    case "$1" in
        "tris_0.25us")    echo "${BASE_DIR}/config_benchmark_tris_0.25us.yaml" ;;
        "tris_0.5us")     echo "${BASE_DIR}/config_benchmark_tris_0.5us.yaml" ;;
        "tris_1.0us")     echo "${BASE_DIR}/config_benchmark_tris_1.0us.yaml" ;;
        "control_0.25us") echo "${BASE_DIR}/config_benchmark_control_0.25us.yaml" ;;
        "control_0.5us")  echo "${BASE_DIR}/config_benchmark_control_0.5us.yaml" ;;
        "control_1.0us")  echo "${BASE_DIR}/config_benchmark_control_1.0us.yaml" ;;
    esac
}

get_md_source() {
    case "$1" in
        "tris_0.25us"|"tris_0.5us"|"tris_1.0us") echo "tris" ;;
        "control_0.25us"|"control_0.5us"|"control_1.0us") echo "control" ;;
    esac
}

get_time_us() {
    case "$1" in
        "tris_0.25us"|"control_0.25us") echo "0.25" ;;
        "tris_0.5us"|"control_0.5us")   echo "0.5" ;;
        "tris_1.0us"|"control_1.0us")   echo "1.0" ;;
    esac
}

# Apply --combos filter
if [[ -n "$COMBOS_STR" ]]; then
    IFS=',' read -r -a COMBO_KEYS <<< "$COMBOS_STR"
else
    COMBO_KEYS=( "${ALL_COMBO_KEYS[@]}" )
fi

# ── output setup ───────────────────────────────────────────────────────────
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
BENCHMARK_ROOT="${DIR_WD}/_benchmark_ensemble_size_${TIMESTAMP}"
mkdir -p "${BENCHMARK_ROOT}"

TIMING_LOG="${BENCHMARK_ROOT}/timing_summary.tsv"
printf "combo_key\tmd_source\ttime_us\tfeatures_dir\toutput_dir\twall_seconds\tstatus\n" \
    > "${TIMING_LOG}"

echo "=============================="
echo "Benchmark ensemble-size sweep"
echo "Combos:         ${COMBO_KEYS[*]}"
echo "n_steps:        ${N_STEPS}"
echo "parallel_jobs:  ${PARALLEL_JOBS}"
echo "split_types:    ${SPLIT_TYPES_STR}"
echo "maxent_values:  ${MAXENT_VALUES_STR}"
echo "bvreg_values:   ${BV_REG_VALUES_STR}"
echo "run_analysis:   ${RUN_ANALYSIS}"
echo "Output root:    ${BENCHMARK_ROOT}"
echo "=============================="
echo ""

# ── parallel job helpers ────────────────────────────────────────────────────
running_jobs_count() { jobs -rp | wc -l; }
wait_for_slot() {
    while [ "$(running_jobs_count)" -ge "${PARALLEL_JOBS}" ]; do sleep 1; done
}

# ── main loop ───────────────────────────────────────────────────────────────
for KEY in "${COMBO_KEYS[@]}"; do
    FEAT_DIR=$(get_features_dir "$KEY")
    COMBO_CFG=$(get_config "$KEY")
    MD_SRC=$(get_md_source "$KEY")
    TIME=$(get_time_us "$KEY")
    COMBO_OUT="${BENCHMARK_ROOT}/${KEY}"

    echo "=============================="
    echo "Combo: ${KEY}"
    echo "  md_source    : ${MD_SRC}"
    echo "  time_us      : ${TIME}"
    echo "  features_dir : ${FEAT_DIR}"
    echo "  config       : ${COMBO_CFG}"
    echo "  output_dir   : ${COMBO_OUT}"

    if [ ! -f "${FEAT_DIR}/features.npz" ]; then
        echo "  [SKIP] features.npz not found at ${FEAT_DIR}"
        printf "%s\t%s\t%s\t%s\t%s\t-1\tskipped_missing_features\n" \
            "${KEY}" "${MD_SRC}" "${TIME}" "${FEAT_DIR}" "${COMBO_OUT}" \
            >> "${TIMING_LOG}"
        echo ""
        continue
    fi

    mkdir -p "${COMBO_OUT}/logs"

    IFS=',' read -r -a MAXENT_ARR <<< "${MAXENT_VALUES_STR}"
    IFS=',' read -r -a BVREG_ARR  <<< "${BV_REG_VALUES_STR}"
    IFS=',' read -r -a BVLOSS_ARR <<< "${BV_REG_LOSSES_STR}"
    IFS=',' read -r -a SPLIT_ARR  <<< "${SPLIT_TYPES_STR}"

    # ── TIMED: fitting only ──────────────────────────────────────────────
    T_START=$(date +%s)

    for SPLIT in "${SPLIT_ARR[@]}"; do
        for MAXENT in "${MAXENT_ARR[@]}"; do
            for BVREG in "${BVREG_ARR[@]}"; do
                for BVLOSS in "${BVLOSS_ARR[@]}"; do
                    wait_for_slot
                    RUN_LOG="${COMBO_OUT}/logs/Tris_only_maxent${MAXENT}_bvreg${BVREG}_${BVLOSS}_split${SPLIT}.log"
                    python "${DIR_WD}/optimise_aSyn_conditions_BV.py" \
                        --condition "Tris_only" \
                        --loss-function "MSE" \
                        --maxent-range "${MAXENT},${MAXENT}" \
                        --bvreg-range "${BVREG},${BVREG}" \
                        --bv-reg-function "${BVLOSS}" \
                        --split-types "${SPLIT}" \
                        --n-steps "${N_STEPS}" \
                        --initial-steps "${INITIAL_STEPS}" \
                        --initial-learning-rate "${INITIAL_LR}" \
                        --learning-rate "${LEARNING_RATE}" \
                        --ema-alpha "${EMA_ALPHA}" \
                        --forward-model-scaling "${FORWARD_MODEL_SCALING}" \
                        --model-parameters-lr-scale "${MODEL_PARAMETERS_LR_SCALE}" \
                        --features-dir "${FEAT_DIR}" \
                        --output-dir "${COMBO_OUT}" \
                        > "${RUN_LOG}" 2>&1 &
                done
            done
        done
        wait  # wait for this split type before moving on
        echo "  [split ${SPLIT} done]"
    done

    wait  # final wait for any stragglers
    T_END=$(date +%s)
    WALL=$(( T_END - T_START ))
    # ── end of timed section ──────────────────────────────────────────────

    echo "  Fitting completed in ${WALL}s"
    printf "%s\t%s\t%s\t%s\t%s\t%s\tok\n" \
        "${KEY}" "${MD_SRC}" "${TIME}" "${FEAT_DIR}" "${COMBO_OUT}" "${WALL}" \
        >> "${TIMING_LOG}"

    # ── post-fitting analysis (untimed) ──────────────────────────────────
    if [ "${RUN_ANALYSIS}" -eq 1 ] && [ -f "${ANALYSIS_RUNNER}" ]; then
        ANA_LOG="${COMBO_OUT}/logs/comprehensive_analysis.log"
        echo "  Starting comprehensive analysis (untimed) ..."
        echo "  Config:  ${COMBO_CFG}"
        echo "  Log:     ${ANA_LOG}"
        if bash "${ANALYSIS_RUNNER}" --config "${COMBO_CFG}" "${COMBO_OUT}" \
            > "${ANA_LOG}" 2>&1; then
            echo "  Analysis completed."
        else
            echo "  Analysis exited with non-zero status (see ${ANA_LOG})"
        fi
    elif [ "${RUN_ANALYSIS}" -eq 1 ]; then
        echo "  WARNING: analysis runner not found at ${ANALYSIS_RUNNER}"
    fi

    echo ""
done

echo "=============================="
echo "All benchmark combos complete."
echo ""
echo "Timing summary (fitting only):"
column -t "${TIMING_LOG}"
echo ""
echo "Full results in: ${BENCHMARK_ROOT}"
