#!/usr/bin/env bash
# Full setup: verify GPU driver, install vLLM (cu126), check HF auth, run Pareto benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

log() { echo "==> $*"; }

# --- 1. GPU driver check ---
if ! nvidia-smi &>/dev/null; then
  LOADED=$(grep -oP 'NVRM version:.*Kernel Module\s+\K[0-9.]+' /proc/driver/nvidia/version 2>/dev/null || echo "none")
  USERSPACE=$(nvidia-smi 2>&1 | sed -n 's/.*NVML library version: \([0-9.]*\).*/\1/p' | head -1 || true)
  if [[ -n "$USERSPACE" && "$LOADED" != "$USERSPACE" && "$LOADED" != "none" ]]; then
    log "Driver/library mismatch (kernel $LOADED vs userspace $USERSPACE). Trying module reload..."
    if sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null && sudo modprobe nvidia; then
      sleep 2
    fi
  fi
  if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi failed. Reboot required after driver upgrade:"
    echo "  sudo reboot"
    echo "Then re-run: bash benchmarks/setup_and_run.sh"
    exit 1
  fi
fi

log "GPU OK: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"

# --- 2. Python venv + vLLM (cu126) ---
if [[ ! -d .venv ]]; then
  log "Creating Python 3.12 venv..."
  uv venv --python 3.12
fi
source .venv/bin/activate

TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
if [[ "$TORCH_CUDA" == "12.9" ]] || [[ "$TORCH_CUDA" == "none" ]] || ! command -v vllm &>/dev/null; then
  log "Installing vLLM with CUDA 12.6 wheels (compatible with driver 570)..."
  uv pip install vllm --torch-backend=cu126 --reinstall
fi

# Set LD_LIBRARY_PATH for bundled CUDA libs (cu126 layout)
VENV_NVIDIA=".venv/lib/python3.12/site-packages/nvidia"
CUDA_LIB_PATHS=()
for libdir in cu13 cu12 cuda_runtime lib; do
  [[ -d "$ROOT/$VENV_NVIDIA/$libdir/lib" ]] && CUDA_LIB_PATHS+=("$ROOT/$VENV_NVIDIA/$libdir/lib")
done
if ((${#CUDA_LIB_PATHS[@]})); then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${CUDA_LIB_PATHS[*]}"):${LD_LIBRARY_PATH:-}"
fi

TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)")
DRIVER_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
log "PyTorch CUDA $TORCH_CUDA | Driver max CUDA $DRIVER_CUDA"

# Quick sanity: vLLM can import CUDA extensions
python3 -c "import vllm._C; print('vllm._C OK')"

# --- 3. HuggingFace auth ---
HF_TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo ""
  echo "ERROR: HuggingFace auth required for Llama 8B."
  echo "  1. Accept: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
  echo "  2. Run:    hf auth login"
  echo "  3. Re-run: bash benchmarks/setup_and_run.sh"
  exit 1
fi
export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

log "HF auth OK: $(hf auth whoami 2>/dev/null || echo 'token from env/file')"

# --- 4. Run benchmark ---
exec bash benchmarks/run_pareto_benchmark.sh
