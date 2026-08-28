#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMS_CSV="${SCRIPT_DIR}/systems.csv"
RAW_ROOT="${SCRIPT_DIR}/raw"
ARCHIVE_ROOT="${SCRIPT_DIR}/archives"
STAGING_ROOT="${SCRIPT_DIR}/staging"
QUARANTINE_ROOT="${SCRIPT_DIR}/quarantine"
DOWNLOADS_CSV="${SCRIPT_DIR}/download_manifest.csv"
REPORT_CSV="${SCRIPT_DIR}/acquisition_report.csv"
VALIDATOR="${SCRIPT_DIR}/validate_acquisition.py"
MODE=""
PILOT_COUNT=1
KEEP_ARCHIVES=0
REPAIR=0

usage() {
  echo "Usage: $0 (--pilot [N] | --all | --verify-only) [--keep-archives] [--repair]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pilot)
      MODE="pilot"
      if [[ ${2:-} =~ ^[0-9]+$ ]]; then PILOT_COUNT="$2"; shift; fi
      ;;
    --all) MODE="all" ;;
    --verify-only) MODE="verify" ;;
    --keep-archives) KEEP_ARCHIVES=1 ;;
    --repair) REPAIR=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ -z "$MODE" ]]; then usage >&2; exit 2; fi
if [[ ! -f "$SYSTEMS_CSV" ]]; then
  echo "Missing ${SYSTEMS_CSV}; run select_systems.py first" >&2
  exit 1
fi

mkdir -p "$RAW_ROOT" "$ARCHIVE_ROOT" "$STAGING_ROOT" "$QUARANTINE_ROOT"

run_python() {
  UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python "$@"
}

if [[ "$MODE" == "verify" ]]; then
  run_python "$VALIDATOR" audit \
    --systems-csv "$SYSTEMS_CSV" --raw-root "$RAW_ROOT" \
    --downloads-csv "$DOWNLOADS_CSV" --report "$REPORT_CSV"
  exit $?
fi

mapfile -t SYSTEM_ROWS < <(run_python - "$SYSTEMS_CSV" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        print(f"{row['system_id']}\t{row['length']}")
PY
)
if [[ "$MODE" == "pilot" ]]; then SYSTEM_ROWS=("${SYSTEM_ROWS[@]:0:${PILOT_COUNT}}"); fi

failures=0
for row in "${SYSTEM_ROWS[@]}"; do
  IFS=$'\t' read -r system_id expected_length <<< "$row"
  target="${RAW_ROOT}/${system_id}"
  url="https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/${system_id}/${system_id}_analysis.zip"
  archive="${ARCHIVE_ROOT}/${system_id}_analysis.zip"

  if [[ -d "$target" ]]; then
    if run_python "$VALIDATOR" system --root "$target" --system-id "$system_id" \
      --expected-length "$expected_length" >/dev/null 2>&1; then
      echo "[SKIP] ${system_id}: existing extraction is valid"
      continue
    fi
    if [[ "$REPAIR" -ne 1 ]]; then
      echo "[FAIL] ${system_id}: existing extraction is invalid (use --repair)" >&2
      failures=$((failures + 1))
      continue
    fi
    quarantine="${QUARANTINE_ROOT}/${system_id}-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$target" "$quarantine"
    echo "[MOVE] Invalid extraction quarantined at ${quarantine}"
  fi

  echo "[HEAD] ${system_id}"
  remote_length="$(curl -fsSLI "$url" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub("\r", "", $2); content_bytes=$2} END{print content_bytes}')"
  if [[ ! "$remote_length" =~ ^[0-9]+$ ]] || [[ "$remote_length" -le 0 ]]; then
    echo "[FAIL] ${system_id}: server returned no Content-Length" >&2
    failures=$((failures + 1))
    continue
  fi

  echo "[GET]  ${system_id} (${remote_length} bytes)"
  if command -v aria2c >/dev/null 2>&1; then
    if ! aria2c --continue=true --max-connection-per-server=2 --split=2 \
      --min-split-size=1M --file-allocation=none --max-tries=3 --retry-wait=5 \
      --dir "$ARCHIVE_ROOT" --out "${system_id}_analysis.zip" "$url"; then
      echo "[FAIL] ${system_id}: download failed" >&2
      failures=$((failures + 1))
      continue
    fi
  elif ! curl -fL --retry 3 --retry-delay 5 -C - -o "$archive" "$url"; then
    echo "[FAIL] ${system_id}: download failed" >&2
    failures=$((failures + 1))
    continue
  fi
  local_length="$(stat -c %s "$archive")"
  if [[ "$local_length" != "$remote_length" ]]; then
    echo "[FAIL] ${system_id}: size ${local_length} != ${remote_length}; partial archive retained" >&2
    failures=$((failures + 1))
    continue
  fi
  if ! run_python "$VALIDATOR" archive "$archive" >/dev/null; then
    echo "[FAIL] ${system_id}: archive integrity validation failed" >&2
    failures=$((failures + 1))
    continue
  fi
  run_python "$VALIDATOR" record-download --manifest "$DOWNLOADS_CSV" \
    --system-id "$system_id" --url "$url" --content-length "$remote_length" --archive "$archive"

  stage="$(mktemp -d "${STAGING_ROOT}/${system_id}.XXXXXX")"
  unzip -q "$archive" -d "$stage"
  pdb_path="$(find "$stage" -type f -name "${system_id}.pdb" -print -quit)"
  if [[ -z "$pdb_path" ]]; then
    echo "[FAIL] ${system_id}: extracted archive has no ${system_id}.pdb; retained at ${stage}" >&2
    failures=$((failures + 1))
    continue
  fi
  extracted_root="$(dirname "$pdb_path")"
  if ! run_python "$VALIDATOR" system --root "$extracted_root" --system-id "$system_id" \
    --expected-length "$expected_length" >/dev/null 2>&1; then
    echo "[FAIL] ${system_id}: extracted data validation failed; retained at ${stage}" >&2
    failures=$((failures + 1))
    continue
  fi
  mv "$extracted_root" "$target"
  if [[ "$extracted_root" != "$stage" ]]; then rmdir "$stage" 2>/dev/null || true; fi
  if [[ "$KEEP_ARCHIVES" -ne 1 ]]; then rm -f "$archive"; fi
  echo "[OK]   ${system_id}"
done

if [[ "$MODE" == "all" ]]; then
  if ! run_python "$VALIDATOR" audit \
    --systems-csv "$SYSTEMS_CSV" --raw-root "$RAW_ROOT" \
    --downloads-csv "$DOWNLOADS_CSV" --report "$REPORT_CSV"; then
    failures=$((failures + 1))
  fi
fi

if [[ "$failures" -gt 0 ]]; then
  echo "Acquisition completed with ${failures} failure(s); rerun to resume." >&2
  exit 1
fi
