#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
environment_file="${example_dir}/openmm_vacuum_environment.yml"
environment_prefix="${JAXENT_OPENMM_ENV:-/tmp/jaxent-openmm-8.5.2}"

if command -v mamba >/dev/null 2>&1; then
  solver=mamba
elif command -v conda >/dev/null 2>&1; then
  solver=conda
else
  echo "mamba or conda is required" >&2
  exit 2
fi

case "${1:-create}" in
  create)
    if [[ ! -x "${environment_prefix}/bin/python" ]]; then
      "${solver}" env create --prefix "${environment_prefix}" --file "${environment_file}"
    fi
    "${solver}" run --prefix "${environment_prefix}" python -m openmm.testInstallation
    ;;
  run)
    shift
    executable="${environment_prefix}/bin/${1}"
    shift
    export OPENMM_PLUGIN_DIR="${environment_prefix}/lib/plugins"
    export LD_LIBRARY_PATH="${environment_prefix}/lib:${LD_LIBRARY_PATH:-}"
    exec "${executable}" "$@"
    ;;
  explicit)
    "${solver}" list --prefix "${environment_prefix}" --explicit
    ;;
  *)
    echo "usage: $0 {create|run|explicit}" >&2
    exit 2
    ;;
esac
