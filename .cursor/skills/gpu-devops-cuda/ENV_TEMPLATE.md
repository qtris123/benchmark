# Portable env.sh Template

Copy this into your project's `env.sh` (or `benchmarks/env.sh`) and source it
after activating the venv. It auto-detects paths so it works on any machine
where CUDA libraries come from pip-installed `nvidia-*` packages.

```bash
#!/usr/bin/env bash
# env.sh — portable CUDA environment setup for pip-installed nvidia packages
# Usage: source env.sh  (after activating the venv)

# ── Locate nvidia packages inside the venv ───────────────────────────────────
VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
NVIDIA="${VENV_SITE}/nvidia"

if [[ ! -d "$NVIDIA" ]]; then
  echo "[env.sh] WARNING: nvidia packages not found in venv at $NVIDIA" >&2
  return 0
fi

# ── LD_LIBRARY_PATH — runtime library loading ─────────────────────────────────
for _d in "$NVIDIA"/*/lib; do
  [[ -d "$_d" ]] && export LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

# ── CUDA_HOME — points to the directory containing bin/nvcc ──────────────────
_NVCC=$(find "$NVIDIA" -name nvcc -type f 2>/dev/null | sort | tail -1)
if [[ -n "$_NVCC" ]]; then
  export CUDA_HOME="$(dirname "$(dirname "$_NVCC")")"
  export PATH="${CUDA_HOME}/bin${PATH:+:$PATH}"
fi

# ── LIBRARY_PATH — linker search path for -l flags during build ───────────────
for _d in "$NVIDIA"/*/lib; do
  [[ -d "$_d" ]] && export LIBRARY_PATH="${_d}${LIBRARY_PATH:+:$LIBRARY_PATH}"
done

# ── CPATH — C/C++ header search path for JIT compilation ─────────────────────
for _d in "$NVIDIA"/*/include; do
  [[ -d "$_d" ]] && export CPATH="${_d}${CPATH:+:$CPATH}"
done

# ── libcudart.so symlink (unversioned, needed by linker) ─────────────────────
# The linker looks for libcudart.so but pip only ships libcudart.so.12 (or .13).
_CUDA_RT_LIB=$(find "${NVIDIA}/cuda_runtime/lib" -name 'libcudart.so.*' 2>/dev/null \
               | sort -V | tail -1)
if [[ -n "$_CUDA_RT_LIB" ]]; then
  _STUB_DIR="$(dirname "$_CUDA_RT_LIB")"
  _STUB_NAME="libcudart.so"
  if [[ ! -e "${_STUB_DIR}/${_STUB_NAME}" ]]; then
    ln -sf "$(basename "$_CUDA_RT_LIB")" "${_STUB_DIR}/${_STUB_NAME}" 2>/dev/null \
      || echo "[env.sh] NOTE: could not create libcudart.so symlink (may need sudo)" >&2
  fi
fi

# ── Framework-specific overrides ─────────────────────────────────────────────
# Check CUDA runtime version bundled in the venv vs what the driver supports.
# Detect driver's max CUDA from nvidia-smi header:
_DRIVER_CUDA=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/{gsub(/[^0-9.]/,"",$NF); print $NF}')
_RUNTIME_VER=$(find "$NVIDIA" -name 'libcudart.so.*' 2>/dev/null \
               | grep -o '[0-9]*\.[0-9]*' | sort -V | tail -1)

# If the venv ships CUDA 13 but driver max is 12.x → disable JIT-compiled
# components that would produce CUDA 13 binaries incompatible with the driver.
if [[ "${_RUNTIME_VER%%.*}" -ge 13 && "${_DRIVER_CUDA%%.*}" -le 12 ]] 2>/dev/null; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  # Also set --attention-backend TRITON_ATTN in your serve command
fi

unset _d _NVCC _CUDA_RT_LIB _STUB_DIR _STUB_NAME _DRIVER_CUDA _RUNTIME_VER
```
