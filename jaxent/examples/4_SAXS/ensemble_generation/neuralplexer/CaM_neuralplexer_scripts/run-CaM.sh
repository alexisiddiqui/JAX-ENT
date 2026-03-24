#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
# 1.  OPTIONAL INPUT ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES="${1:-25}"           # number of conformers (default: 25)
NUM_STEPS="${2:-200}"           # denoising diffusion steps (default: 50)
SAMPLER="${3:-langevin_simulated_annealing}"  # sampler (default: langevin_simulated_annealing)
# Validate sampler
case "${SAMPLER}" in
    DDIM|VPSDE|simulated_annealing_simple|langevin_simulated_annealing) ;;
    *)
        echo "[ERROR] Unknown sampler '${SAMPLER}'."
        echo "        Valid options: DDIM, VPSDE, simulated_annealing_simple, langevin_simulated_annealing"
        exit 1
        ;;
esac

NEURALPLEXER_DIR="/home/alexi/Documents/NeuralPLexer"          # repo root
CHECKPOINT="/home/alexi/Documents/NeuralPLexer/weights/neuralplexer_v2.ckpt"  # model weights

INPUT_DIR="${NEURALPLEXER_DIR}/APO-CaM/_sub_cluster_output"  # directory containing input PDB and ligand SDFs
# INPUT_PDB="${INPUT_DIR}/CaM_s20_r1_msa1-127_n12700_do1_20260310_183757_protonated_plddt_ordered_195_HOLOlike.pdb"       # chained HOLO structure
# INPUT_PDB="${INPUT_DIR}/CaM_s2_r1_msa1-127_n1270_do1_20260309_181041_protonated_max_plddt_30.pdb"       # chained HOLO structure


INPUT_PDB="${4:-${INPUT_DIR}/CaM_s2_r1_msa1-127_n1270_do1_20260309_181041_protonated_max_plddt_30.pdb}"  # input PDB (default: APO-like structure)
# set ligand counds as args default: 4 and 2
N_CA_IONS="${5:-4}"          # number of Ca²⁺ ions (default: 4)
N_CDZ_LIGANDS="${6:-2}"        # number of CDZ ligands (default: 2)

LIGAND_SDF="${INPUT_DIR}/85H_ideal.sdf"  # optional ligand SDF (not used in this script, but can be used for more complex ligand handling)

CA_SDF="${INPUT_DIR}/CA_ideal.sdf"  # optional Ca²⁺ ion SDF (not used in this script, but can be used for more complex ligand handling)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SAMPLING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE=5          # samples processed per GPU batch (lower if OOM)
DEVICE="cuda"         # "cuda" or "cpu"

OUTPUT_DIR="${INPUT_DIR}/2CDZ_ensemble_output_${N_SAMPLES}samples_${NUM_STEPS}steps_${SAMPLER}_$(date +%Y%m%d_%H%M%S)"
rm -rf "${OUTPUT_DIR}"  # clear previous outputs 

PROTEIN_PDB="${OUTPUT_DIR}/protein_only.pdb"                   # auto-generated
# ─────────────────────────────────────────────────────────────────────────────
# 3.  ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  NeuralPLexer — HOLO CyaA Ensemble Prediction"
echo "  $(date)"
echo "============================================================"

# Activate conda environment
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
source activate neuralplexer_dev
set -u

# Verify neuralplexer-inference is available
if ! command -v neuralplexer-inference &> /dev/null; then
    echo "[ERROR] neuralplexer-inference not found."
    echo "        Install with: pip install -e '.[gpu]' inside ${NEURALPLEXER_DIR}"
    exit 1
fi

# Check input files
if [[ ! -f "${INPUT_PDB}" ]]; then
    echo "[ERROR] Input PDB not found: ${INPUT_PDB}"
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] Model checkpoint not found: ${CHECKPOINT}"
    echo "        Download weights from https://github.com/zrqiao/NeuralPLexer/releases"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  PREPARE PROTEIN-ONLY PDB  (strip Ca²⁺ and CDZ HETATM records)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "[1/3] Preparing protein-only PDB ..."

grep "^ATOM" "${INPUT_PDB}" > "${PROTEIN_PDB}"
echo "      → Wrote: ${PROTEIN_PDB}"

# Define ligands
CDZ_SMILES="Clc1ccc(cc1)C(c2ccc(Cl)cc2)[n+]3ccn(C[C@H](OCc4ccc(Cl)cc4Cl)c5ccc(Cl)cc5Cl)c3"
CA_SMILES="[Ca+2]"

echo "      → CDZ ligand SMILES: ${CDZ_SMILES}"
echo "      → Ca²⁺ ligand SMILES: ${CA_SMILES}"

# Count residues and ligands in original file (for sanity check)
N_RESIDUES=$(grep "^ATOM" "${INPUT_PDB}" | awk '{print $6}' | sort -un | wc -l)


echo "      → Protein residues : ${N_RESIDUES}"
echo "      → Ca²⁺ ions found  : ${N_CA_IONS}"
echo "      → CDZ ligands      : ${N_CDZ_LIGANDS}"

# ─────────────────────────────────────────────────────────────────────────────
# 5.  BUILD LIGAND FLAGS  (--input-ligand for Ca²⁺ ions and CDZ)
#     Combine all ligand SMILES: Ca²⁺ repeated N_CA_IONS times + CDZ repeated N_CDZ_LIGANDS times
# ─────────────────────────────────────────────────────────────────────────────

if [[ "${N_CA_IONS}" -eq 0 && "${N_CDZ_LIGANDS}" -eq 0 ]]; then
    echo ""
    echo "[2/3] No ligands requested — running protein-only (APO) mode."
    LIGAND_FLAGS=""
else
    echo ""
    echo "[2/3] Building ligand flags for ${N_CA_IONS} Ca²⁺ ions + ${N_CDZ_LIGANDS} CDZ ligands ..."
    LIGANDS=""
    for i in $(seq 1 "${N_CA_IONS}"); do
        if [[ -z "${LIGANDS}" ]]; then LIGANDS="${CA_SDF}"; else LIGANDS="${LIGANDS}|${CA_SDF}"; fi
    done
    for i in $(seq 1 "${N_CDZ_LIGANDS}"); do
        if [[ -z "${LIGANDS}" ]]; then LIGANDS="${LIGAND_SDF}"; else LIGANDS="${LIGANDS}|${LIGAND_SDF}"; fi
    done
    LIGAND_FLAGS="--input-ligand '${LIGANDS}'"
    echo "      → Generated combined --input-ligand flag with Ca²⁺ and CDZ"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6.  RUN NEURALPLEXER INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "[3/3] Running NeuralPLexer ensemble sampling ..."
echo "      Samples   : ${N_SAMPLES}"
echo "      Steps     : ${NUM_STEPS}"
echo "      Sampler   : ${SAMPLER}"
echo "      Chunk size: ${CHUNK_SIZE}"
echo "      Device    : ${DEVICE}"
echo "      Output    : ${OUTPUT_DIR}"
echo ""

# shellcheck disable=SC2086  # intentional word-splitting for LIGAND_FLAGS
eval neuralplexer-inference \
    --task=batched_structure_sampling \
    --model-checkpoint "${CHECKPOINT}" \
    \
    --input-receptor "${PROTEIN_PDB}" \
    --start-time 0.75 \
    --input-template  "${INPUT_PDB}" \
    --use-template \
    --discard-sdf-coords \
    --n-samples   "${N_SAMPLES}" \
    --chunk-size  "${CHUNK_SIZE}" \
    --num-steps   "${NUM_STEPS}" \
    \
    --cuda \
    --sampler     "${SAMPLER}" \
    --separate-pdb \
    --rank-outputs-by-confidence \
    --out-path    "${OUTPUT_DIR}" \
    \
    ${LIGAND_FLAGS} \


echo ""
echo "Combining protein + ligand outputs into complex PDBs ..."
python "${INPUT_DIR}/combine_outputs.py" --input_dir "${OUTPUT_DIR}"
echo "Combined PDBs written to: ${OUTPUT_DIR}/combined_outputs/"

# ─────────────────────────────────────────────────────────────────────────────
# 7.  POST-RUN SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Done!  $(date)"
echo "============================================================"

N_FRAMES=$(ls "${OUTPUT_DIR}"/prot_*.pdb 2>/dev/null | grep -v "all" | grep -v "only" | wc -l)
echo "  Ensemble frames generated : ${N_FRAMES}"
echo "  Output directory          : ${OUTPUT_DIR}"
echo ""

if [[ "${N_FRAMES}" -gt 0 ]]; then
    echo "  Next steps:"
    echo "  ┌─ RMSD analysis  → python analyse_ensemble.py"
    echo "  ├─ SAXS validation → multifoxs ${OUTPUT_DIR}/prot_*.pdb experimental.dat"
    echo "  └─ Visualise      → pymol ${OUTPUT_DIR}/prot_*.pdb"
fi

if [[ "${N_CA_IONS}" -eq 0 && "${N_CDZ_LIGANDS}" -eq 0 ]]; then
    echo ""
    echo "[INFO] APO mode — skipping ligand contact analysis."
else
    echo ""
    echo "============================================================"
    echo "  Running ligand contact analysis on Minimised ensemble ..."
    echo "============================================================"
    python "${INPUT_DIR}/ligand_contact_analysis.py" \
        --ensemble-dir "${OUTPUT_DIR}/combined_outputs_minimised" \
        --pdb-codes 7PU9 7PSZ
fi