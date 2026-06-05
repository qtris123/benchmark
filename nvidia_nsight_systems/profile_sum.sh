#!/usr/bin/env bash
# profile_sum.sh — Profile naive PyTorch vs. optimized Triton 2-sum with nsys
#
# Usage:
#   bash profile_sum.sh              # profile both (default)
#   TARGET=naive  bash profile_sum.sh
#   TARGET=triton bash profile_sum.sh
#
# Output in ./profiles-sum/:
#   naive_pytorch_sum.nsys-rep       — full trace (open in Nsight Systems GUI)
#   triton_sum.nsys-rep
#   naive_pytorch_sum_stats.txt      — CLI kernel/API summary (no GUI needed)
#   triton_sum_stats.txt
#
# --capture-range=cudaProfilerApi is used so nsys records ONLY the hot
# iteration loop (between cuda_profiler.start() / .stop() in each script),
# not Python startup, Triton JIT, or allocations.
#
# Requirements:
#   - /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys
#   - /localhome/local-triv/Test-env/bin/python  (with torch + triton)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── nsys binary ──────────────────────────────────────────────────────────────
NSYS="/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys"
if [[ ! -x "$NSYS" ]]; then
  # Fall back to system symlink
  NSYS="$(command -v nsys 2>/dev/null || true)"
  [[ -z "$NSYS" ]] && { echo "ERROR: nsys not found"; exit 1; }
fi
echo "=== nsys: $($NSYS --version 2>&1 | head -1) ==="

# ── Python from Test-env ──────────────────────────────────────────────────────
PYTHON="/localhome/local-triv/Test-env/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: $PYTHON not found — did you create Test-env?"
  exit 1
fi
echo "=== Python: $($PYTHON --version 2>&1) ==="
echo ""

# ── Output directory ──────────────────────────────────────────────────────────
PROFILE_DIR="$ROOT/profiles-sum"
mkdir -p "$PROFILE_DIR"

# ── nvidia-uvm preflight (keeps /dev/nvidia-uvm healthy) ─────────────────────
_fix_nvidia_uvm() {
  local reg
  reg=$(awk '/nvidia-uvm/{print $1}' /proc/devices 2>/dev/null)
  [[ -z "$reg" ]] && return
  local cur
  cur=$(printf '%d' "0x$(stat -c '%t' /dev/nvidia-uvm 2>/dev/null)" 2>/dev/null || echo "")
  if [[ "$cur" != "$reg" ]]; then
    echo "WARNING: /dev/nvidia-uvm major mismatch — attempting auto-fix..."
    sudo rm -f /dev/nvidia-uvm /dev/nvidia-uvm-tools \
      && sudo mknod /dev/nvidia-uvm c "$reg" 0 \
      && sudo mknod /dev/nvidia-uvm-tools c "$reg" 1 \
      && sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools \
      && echo "Fixed."
  fi
}
_fix_nvidia_uvm

# ── Profile runner ────────────────────────────────────────────────────────────
# Args: <label> <script_path>
run_profile() {
  local LABEL="$1"
  local SCRIPT="$2"
  local OUT_BASE="$PROFILE_DIR/$LABEL"

  echo "═══════════════════════════════════════════════════════════"
  echo "  Profiling : $LABEL"
  echo "  Script    : $SCRIPT"
  echo "  Output    : ${OUT_BASE}.nsys-rep"
  echo "═══════════════════════════════════════════════════════════"
  echo ""

  # nsys profile flags:
  #   --capture-range=cudaProfilerApi  → record only between
  #       cuda_profiler.start() and cuda_profiler.stop() in the Python script
  #   --trace=cuda,nvtx                → capture GPU kernels + NVTX ranges
  #   --sample=none                    → no CPU call-stack sampling (cleaner trace)
  #   --stats=true                     → print kernel summary to stdout
  #   --force-overwrite true           → overwrite existing .nsys-rep
  "$NSYS" profile \
    --capture-range=cudaProfilerApi \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --stats=true \
    --force-overwrite true \
    --output "$OUT_BASE" \
    "$PYTHON" "$SCRIPT" \
    2>&1 | tee "${OUT_BASE}_run.log"

  echo ""

  # ── Generate CLI stats (no GUI needed) ──────────────────────────────────────
  if [[ -f "${OUT_BASE}.nsys-rep" ]]; then
    local SIZE
    SIZE=$(du -sh "${OUT_BASE}.nsys-rep" | cut -f1)
    echo "Written: ${OUT_BASE}.nsys-rep  (${SIZE})"
    echo ""

    local STATS_FILE="${OUT_BASE}_stats.txt"
    {
      echo "══ $LABEL — nsys stats ══════════════════════════════════════"
      echo ""
      echo "── GPU Kernel Summary (sorted by total GPU time) ────────────"
      "$NSYS" stats \
        --report cuda_gpu_kern_sum \
        --timeunit milliseconds \
        "${OUT_BASE}.nsys-rep" 2>/dev/null || echo "(no cuda_gpu_kern_sum)"

      echo ""
      echo "── CUDA API Summary (CPU-side call overhead) ────────────────"
      "$NSYS" stats \
        --report cuda_api_sum \
        --timeunit milliseconds \
        "${OUT_BASE}.nsys-rep" 2>/dev/null || echo "(no cuda_api_sum)"

      echo ""
      echo "── GPU Memory Transfer Timing ───────────────────────────────"
      "$NSYS" stats \
        --report cuda_gpu_mem_time_sum \
        --timeunit milliseconds \
        "${OUT_BASE}.nsys-rep" 2>/dev/null || echo "(no cuda_gpu_mem_time_sum)"

      echo ""
      echo "── NVTX Range Summary ───────────────────────────────────────"
      "$NSYS" stats \
        --report nvtx_sum \
        --timeunit milliseconds \
        "${OUT_BASE}.nsys-rep" 2>/dev/null || echo "(no nvtx_sum)"

    } | tee "$STATS_FILE"

    echo ""
    echo "Stats written: $STATS_FILE"
  else
    echo "WARNING: ${OUT_BASE}.nsys-rep not found — check run log for errors."
  fi

  echo ""
}

# ── Select which targets to run ───────────────────────────────────────────────
TARGET="${TARGET:-both}"   # naive | triton | both

case "$TARGET" in
  naive)
    run_profile "naive_pytorch_sum" "$ROOT/sum_naive_pytorch.py"
    ;;
  triton)
    run_profile "triton_sum" "$ROOT/sum_triton_kernel.py"
    ;;
  both)
    run_profile "naive_pytorch_sum" "$ROOT/sum_naive_pytorch.py"
    run_profile "triton_sum"        "$ROOT/sum_triton_kernel.py"
    ;;
  *)
    echo "ERROR: TARGET must be 'naive', 'triton', or 'both' (got: '$TARGET')"
    exit 1
    ;;
esac

# ── Summary ───────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo " Profiling complete."
echo " Reports in: $PROFILE_DIR/"
echo ""
ls -lh "$PROFILE_DIR/"*.nsys-rep 2>/dev/null || echo "(no .nsys-rep files)"
echo ""
echo " To view in Nsight Systems GUI, copy .nsys-rep to your local"
echo " machine (requires GUI version >= 2024.6.2):"
echo ""
echo "   scp \$(hostname):$PROFILE_DIR/naive_pytorch_sum.nsys-rep  ~/Desktop/"
echo "   scp \$(hostname):$PROFILE_DIR/triton_sum.nsys-rep         ~/Desktop/"
echo ""
echo " CLI stats already saved:"
ls -lh "$PROFILE_DIR/"*_stats.txt 2>/dev/null || true
echo "═══════════════════════════════════════════════════════════"
