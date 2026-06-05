#!/usr/bin/env bash
# Thin wrapper around analyze_nsight_decode.py.
# Runs the full raw-.nsys-rep -> SQLite -> SQL -> CSV -> chart pipeline.
#
# Usage:
#   ./analyze_nsight_decode.sh                       # defaults: ./nsight-profiles, focus c=32
#   ./analyze_nsight_decode.sh --focus-conc 2
#   ./analyze_nsight_decode.sh --profile-dir /path/to/profiles --force-export
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pick a python that has matplotlib; fall back to python3.
PY="${PYTHON:-python3}"
if ! "$PY" -c "import matplotlib" >/dev/null 2>&1; then
  echo "ERROR: '$PY' lacks matplotlib. Set PYTHON=/path/to/python with matplotlib installed." >&2
  exit 1
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "ERROR: nsys not found on PATH." >&2
  exit 1
fi

exec "$PY" "$SCRIPT_DIR/analyze_nsight_decode.py" "$@"
