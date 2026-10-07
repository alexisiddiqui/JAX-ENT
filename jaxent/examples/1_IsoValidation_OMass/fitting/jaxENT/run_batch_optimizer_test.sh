#!/bin/bash
set -euo pipefail

# Disposable fit-only launcher: batches the seven MaxEnt candidates per data split.
cd "$(dirname "$0")"
timestamp=$(date +'%Y%m%d_%H%M%S')
output_dir="$(pwd)/_optimise_batch7_rate_${timestamp}"
mkdir -p "$output_dir/logs"

echo "Batch size: 7"
echo "Output directory: $output_dir"
/usr/bin/time -f '%e' -o "$output_dir/fit_seconds.txt" \
  env JAX_PLATFORM_NAME=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
  UV_CACHE_DIR=/tmp/jaxent-uv-cache uv run --no-sync python optimise_batch_test.py \
  --output-dir "$output_dir" --batch-size 7 "$@" \
  2>&1 | tee "$output_dir/logs/optimise.log"

echo "Fit seconds: $(<"$output_dir/fit_seconds.txt")"
echo "Results: $output_dir"
