#!/usr/bin/env bash
set -euo pipefail

neuralplexer_python="${NEURALPLEXER_PYTHON:-/home/alexi/anaconda3/envs/neuralplexer_dev/bin/python}"
pyrosetta_site="${PYROSETTA_SITE:-/home/alexi/anaconda3/lib/python3.11/site-packages}"

if [[ ! -x "${neuralplexer_python}" ]]; then
  echo "NeuralPLexer Python is unavailable: ${neuralplexer_python}" >&2
  exit 1
fi
if [[ ! -d "${pyrosetta_site}/pyrosetta" ]]; then
  echo "Local PyRosetta package is unavailable: ${pyrosetta_site}/pyrosetta" >&2
  exit 1
fi

exec "${neuralplexer_python}" "$@" --pyrosetta-site "${pyrosetta_site}"

