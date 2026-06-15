#!/usr/bin/env bash
# Machine-specific CUDA health check: 2x A100X, driver 570, max CUDA 12.8
# This is a thin wrapper around the portable check with hardcoded paths.
# Usage: bash ~/.cursor/skills/gpu-devops-a100x-driver570/scripts/cuda-health-check.sh

ROOT="/localhome/local-triv"
VENV="$ROOT/.venv"
ENV_FILE="$ROOT/benchmarks/env.sh"

exec bash "$(dirname "${BASH_SOURCE[0]}")/../../gpu-devops-cuda/scripts/cuda-health-check.sh" \
  --root "$ROOT" --venv "$VENV" --env "$ENV_FILE"
