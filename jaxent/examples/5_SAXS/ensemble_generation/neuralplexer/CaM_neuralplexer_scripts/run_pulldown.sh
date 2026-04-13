#!/bin/bash
# Runs a pull down experiment on the clusters in NeuralPLexer/APO-CaM/_sub_cluster_output/clusters
# Runs NeuralPLexer/APO-CaM/_sub_cluster_output/run-CaM.sh: 0-4 Ca2+ ions, 0-2 CDZ Ligands. Generates 5 structures per cluster-condition.
# 15 clusters × 5 Ca²⁺ conditions × 3 CDZ conditions × 5 samples = 1125 structures total.

set -eu

# Variables
SCRIPT_DIR="/home/alexi/Documents/NeuralPLexer/APO-CaM/_sub_cluster_output"
CLUSTERS_DIR="${SCRIPT_DIR}/clusters"
RUN_SCRIPT="${SCRIPT_DIR}/run-CaM.sh"
PULLDOWN_DIR="${SCRIPT_DIR}/pulldown"

N_SAMPLES=5
NUM_STEPS=20
SAMPLER="langevin_simulated_annealing"

# Outer setup
mkdir -p "${PULLDOWN_DIR}"
total=0

# Triple loop: cluster, Ca²⁺ count, CDZ count
for cluster_dir in "${CLUSTERS_DIR}"/*/; do
    medoid="${cluster_dir}medoid.pdb"
    [[ ! -f "${medoid}" ]] && echo "[WARN] No medoid in ${cluster_dir}, skipping." && continue
    cluster_name=$(basename "${cluster_dir}")

    for n_ca in 0 1 2 3 4; do
        for n_cdz in 0 1 2; do
            echo "=== Cluster ${cluster_name}  Ca²⁺=${n_ca}  CDZ=${n_cdz} ==="

            # Snapshot existing run-CaM output dirs BEFORE the call
            mapfile -t before < <(ls -d "${SCRIPT_DIR}"/2CDZ_ensemble_output_*/ 2>/dev/null | sort)

            bash "${RUN_SCRIPT}" "${N_SAMPLES}" "${NUM_STEPS}" "${SAMPLER}" \
                "${medoid}" "${n_ca}" "${n_cdz}"

            # Find the newly-created output dir (set difference)
            mapfile -t after < <(ls -d "${SCRIPT_DIR}"/2CDZ_ensemble_output_*/ 2>/dev/null | sort)
            new_dir=$(comm -13 <(printf '%s\n' "${before[@]}") \
                                <(printf '%s\n' "${after[@]}") | head -1)

            if [[ -n "${new_dir}" ]]; then
                dest="${PULLDOWN_DIR}/cluster_${cluster_name}_ca${n_ca}_cdz${n_cdz}"
                mv "${new_dir}" "${dest}"
                echo "  → Moved to: ${dest}"
            else
                echo "  [WARN] Could not detect output dir for ${cluster_name} ca${n_ca} cdz${n_cdz}"
            fi

            total=$(( total + 1 ))
        done
    done
done

echo "=== Pull-down complete: ${total} conditions run ==="
echo "=== Outputs in: ${PULLDOWN_DIR} ==="