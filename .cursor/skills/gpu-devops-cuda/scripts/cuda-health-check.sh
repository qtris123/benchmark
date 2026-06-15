#!/usr/bin/env bash
# Portable CUDA health check — works on any Linux GPU server.
# Usage:
#   bash cuda-health-check.sh [--root DIR] [--venv DIR] [--env FILE]
#
# Defaults: --root CWD, --venv $ROOT/.venv, --env $ROOT/benchmarks/env.sh
# The script sources --env if provided, then runs 8 checks.

set -uo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
ROOT="$(pwd)"
VENV=""
ENV_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --env)  ENV_FILE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -z "$VENV" ]]     && VENV="$ROOT/.venv"
[[ -z "$ENV_FILE" ]] && ENV_FILE="$ROOT/benchmarks/env.sh"

PY="$VENV/bin/python3"

# ── Counters & helpers ────────────────────────────────────────────────────────
PASS=0; FAIL=0; WARN=0
ok()   { echo "  [PASS] $*"; ((PASS++)); }
fail() { echo "  [FAIL] $*"; ((FAIL++)); }
warn() { echo "  [WARN] $*"; ((WARN++)); }
info() { echo "         $*"; }

# ── Source env file if present ────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE" 2>/dev/null || true
fi

echo "=== CUDA Health Check ==="
echo "    root: $ROOT"
echo "    venv: $VENV"
[[ -f "$ENV_FILE" ]] && echo "    env:  $ENV_FILE (sourced)"
echo ""

# ── 1. nvidia-smi ─────────────────────────────────────────────────────────────
echo "[1] nvidia-smi"
if nvidia-smi &>/dev/null; then
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
  DRIVER_CUDA=$(nvidia-smi 2>/dev/null | awk '/CUDA Version:/{for(i=1;i<=NF;i++) if($i=="CUDA") print $(i+2)}' | head -1)
  ok "nvidia-smi OK — $GPU_COUNT× $GPU_NAME | driver $DRIVER | max CUDA $DRIVER_CUDA"
else
  fail "nvidia-smi failed — driver not responding or permissions issue"
  info "Try: sudo nvidia-smi   (check /dev/nvidiactl permissions)"
fi

# ── 2. Kernel modules ─────────────────────────────────────────────────────────
echo "[2] Kernel modules"
for mod in nvidia nvidia_uvm; do
  lsmod | grep -q "^$mod " && ok "$mod loaded" || fail "$mod NOT loaded"
done

# ── 3. /dev/nvidia-uvm major number consistency ───────────────────────────────
echo "[3] /dev/nvidia-uvm device file"
if [[ -e /dev/nvidia-uvm ]]; then
  MOD_MAJOR=$(awk '/nvidia-uvm/{print $1}' /proc/devices 2>/dev/null)
  DEV_MAJOR=$(printf '%d' "0x$(stat -c '%t' /dev/nvidia-uvm 2>/dev/null)" 2>/dev/null)
  if [[ "$MOD_MAJOR" == "$DEV_MAJOR" ]]; then
    ok "/dev/nvidia-uvm major=$DEV_MAJOR matches module"
  else
    fail "/dev/nvidia-uvm major MISMATCH: device=$DEV_MAJOR, module=$MOD_MAJOR"
    info "Fix: sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
    info "     NEW=\$(awk '/nvidia-uvm/{print \$1}' /proc/devices)"
    info "     sudo rm /dev/nvidia-uvm /dev/nvidia-uvm-tools"
    info "     sudo mknod /dev/nvidia-uvm c \$NEW 0 && sudo mknod /dev/nvidia-uvm-tools c \$NEW 1"
    info "     sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools"
  fi
  python3 -c "import os; os.open('/dev/nvidia-uvm', os.O_RDWR)" &>/dev/null \
    && ok "/dev/nvidia-uvm opens OK" \
    || fail "/dev/nvidia-uvm opens with ENODEV — module in broken state (see fix above)"
else
  fail "/dev/nvidia-uvm does not exist"
fi

# ── 4. cuInit ─────────────────────────────────────────────────────────────────
echo "[4] cuInit (driver API)"
CUINIT=$(python3 -c "
import ctypes
try:
    lib = ctypes.CDLL('libcuda.so.1')
    print(lib.cuInit(0))
except Exception as e:
    print('err:' + str(e))
" 2>/dev/null || echo "no-python")
if [[ "$CUINIT" == "0" ]]; then
  ok "cuInit=0 (CUDA_SUCCESS)"
elif [[ "$CUINIT" == "no-python" ]]; then
  warn "system python3 not found; skipping cuInit check"
else
  fail "cuInit=$CUINIT  (0=success; 999=CUDA_ERROR_UNKNOWN → see check 3)"
fi

# ── 5. PyTorch CUDA ───────────────────────────────────────────────────────────
echo "[5] PyTorch CUDA"
if [[ -x "$PY" ]]; then
  TORCH_OUT=$("$PY" -c "
import torch
avail = torch.cuda.is_available()
cnt = torch.cuda.device_count()
names = '|'.join(torch.cuda.get_device_name(i) for i in range(cnt)) if avail else ''
print(avail, cnt, names)
" 2>/dev/null || echo "import-error")
  if [[ "$TORCH_OUT" == "import-error" ]]; then
    warn "torch import failed — venv may be missing torch"
  else
    read -r AVAIL CNT NAMES <<< "$TORCH_OUT"
    if [[ "$AVAIL" == "True" ]]; then
      ok "torch.cuda.is_available()=True  ($CNT GPU(s): $NAMES)"
    else
      fail "torch.cuda.is_available()=False"
      info "Ensure env.sh is sourced and LD_LIBRARY_PATH includes CUDA runtime libs"
    fi
  fi
else
  warn "venv python not found at $PY — activate venv first"
fi

# ── 6. CUDA_HOME / nvcc ───────────────────────────────────────────────────────
echo "[6] CUDA_HOME / nvcc"
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  NVCC_VER=$("${CUDA_HOME}/bin/nvcc" --version 2>/dev/null \
    | awk '/release/{for(i=1;i<=NF;i++) if($i=="release"){gsub(/,/,"",$( i+1)); print $(i+1); exit}}')
  ok "CUDA_HOME=$CUDA_HOME  nvcc $NVCC_VER"
  # Warn if nvcc CUDA version exceeds driver's max
  DRIVER_CUDA_MAJOR="${DRIVER_CUDA%%.*}"
  NVCC_MAJOR="${NVCC_VER%%.*}"
  if [[ -n "${DRIVER_CUDA_MAJOR:-}" && -n "${NVCC_MAJOR:-}" ]] && \
     [[ "$NVCC_MAJOR" -gt "$DRIVER_CUDA_MAJOR" ]] 2>/dev/null; then
    warn "nvcc is CUDA $NVCC_VER but driver max is CUDA $DRIVER_CUDA"
    info "JIT kernels compiled by this nvcc will fail at runtime"
    info "Set VLLM_USE_FLASHINFER_SAMPLER=0 (or equivalent) to avoid JIT"
  fi
else
  warn "CUDA_HOME/nvcc not set — source your env.sh or set CUDA_HOME manually"
  info "To find nvcc in venv: find .venv -name nvcc -type f 2>/dev/null"
fi

# ── 7. Bundled .so CUDA version vs driver max ─────────────────────────────────
echo "[7] Shared library CUDA dependencies"
if [[ -x "$PY" ]]; then
  SITE=$("$PY" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
  if [[ -n "$SITE" && -d "$SITE" ]]; then
    INCOMPAT=$(for so in $(find "$SITE" -name "*.so" 2>/dev/null | head -200); do
      dep=$(readelf -d "$so" 2>/dev/null \
            | awk '/NEEDED.*libcudart/{gsub(/[\[\]]/,"",$NF); print $NF}')
      [[ -n "$dep" ]] && echo "$dep $(basename "$so")"
    done | sort -u)

    if [[ -n "${DRIVER_CUDA:-}" ]]; then
      DRIVER_MAJ="${DRIVER_CUDA%%.*}"
      PROBLEMS=$(echo "$INCOMPAT" | awk -v dmax="$DRIVER_MAJ" '
        /libcudart\.so\./{
          n=$1; sub(/.*\.so\./, "", n)
          if (n+0 > dmax+0) print "  CUDA " n " > driver max " dmax ": " $2
        }')
      if [[ -n "$PROBLEMS" ]]; then
        # Downgrade to WARN if the caller has already applied workarounds
        if [[ "${VLLM_USE_FLASHINFER_SAMPLER:-1}" == "0" ]]; then
          warn "CUDA $((DRIVER_MAJ+1))+ libs present but workarounds active (TRITON_ATTN / VLLM_USE_FLASHINFER_SAMPLER=0):"
        else
          fail "Libraries requiring CUDA newer than driver:"
        fi
        echo "$PROBLEMS" | while read -r line; do info "$line"; done
        [[ "${VLLM_USE_FLASHINFER_SAMPLER:-1}" != "0" ]] &&           info "These will crash at runtime — use a CPU/Triton fallback or reinstall"
      else
        ok "All detected .so files are within driver CUDA compatibility"
      fi
    else
      warn "Could not determine driver CUDA version — skipping .so compat check"
    fi
  fi
else
  warn "Skipping .so compat check — venv not active"
fi

# ── 8. Key environment variables ──────────────────────────────────────────────
echo "[8] Environment variables"
[[ -n "${LD_LIBRARY_PATH:-}" ]] \
  && ok "LD_LIBRARY_PATH set" \
  || warn "LD_LIBRARY_PATH not set — CUDA libs may not load; source env.sh"
[[ -n "${CUDA_HOME:-}" ]] \
  && ok "CUDA_HOME=$CUDA_HOME" \
  || warn "CUDA_HOME not set (needed for JIT compilation)"
[[ "${VLLM_USE_FLASHINFER_SAMPLER:-1}" == "0" ]] \
  && ok "VLLM_USE_FLASHINFER_SAMPLER=0 (flashinfer JIT disabled)" \
  || info "VLLM_USE_FLASHINFER_SAMPLER not set (flashinfer JIT will be attempted if vllm is used)"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary: $PASS passed, $WARN warnings, $FAIL failed ==="
if [[ $FAIL -gt 0 ]]; then
  echo "    Action required — see FAIL items above."
  exit 1
elif [[ $WARN -gt 0 ]]; then
  echo "    Review warnings above before running GPU workloads."
  exit 0
else
  echo "    All checks passed — GPU environment is healthy."
  exit 0
fi
