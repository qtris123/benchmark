# Source this file after activating sglang-env.
# Usage: source sglang-bench/env.sh
#
# Sets LD_LIBRARY_PATH so PyTorch / flashinfer / triton can find the CUDA
# runtime and JIT-link libraries that ship inside the sglang-env nvidia wheels.
# The structure in sglang-env is flat:
#   .../nvidia/{package}/lib/  (no cu12/cu13 subdirectory like in the vLLM venv)

_SGLANG_ENV="${_SGLANG_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/sglang-env}"
_VENV_NVIDIA="$_SGLANG_ENV/lib/python3.12/site-packages/nvidia"

_CUDA_LIB_PATHS=()
for _pkg in cuda_runtime cuda_nvrtc nvjitlink cublas curand cusparse cudnn cufft cusolver cusparselt; do
  [[ -d "$_VENV_NVIDIA/$_pkg/lib" ]] && _CUDA_LIB_PATHS+=("$_VENV_NVIDIA/$_pkg/lib")
done

if (( ${#_CUDA_LIB_PATHS[@]} )); then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${_CUDA_LIB_PATHS[*]}"):${LD_LIBRARY_PATH:-}"
fi

# LIBRARY_PATH for linker (-lfoo) at build time (triton / flashinfer JIT)
export LIBRARY_PATH="$_VENV_NVIDIA/cuda_runtime/lib:${LIBRARY_PATH:-}"

# CPATH: headers needed by JIT-compiled kernels
_CPATH_EXTRA=()
for _inc in curand cuda_nvrtc nvjitlink cublas cusparse; do
  [[ -d "$_VENV_NVIDIA/$_inc/include" ]] && _CPATH_EXTRA+=("$_VENV_NVIDIA/$_inc/include")
done
if (( ${#_CPATH_EXTRA[@]} )); then
  export CPATH="$(IFS=:; echo "${_CPATH_EXTRA[*]}"):${CPATH:-}"
fi

export PATH="$HOME/.local/bin:$PATH"

# Override CUDA_HOME to the sglang-env's cuda_nvcc headers.
# This prevents flashinfer/triton from picking up nvcc from the vLLM .venv
# (which ships nvidia/cu13/bin/nvcc — a CUDA 13 compiler incompatible with
# the torch 2.7.1+cu128 headers installed in sglang-env).
# Unsetting CUDA_HOME entirely is an alternative, but an explicit override is
# safer when benchmarks/env.sh has been sourced first.
_CUDA_NVCC_DIR="${_SGLANG_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/sglang-env}/lib/python3.12/site-packages/nvidia/cuda_nvcc"
if [[ -d "$_CUDA_NVCC_DIR" ]]; then
  export CUDA_HOME="$_CUDA_NVCC_DIR"
else
  unset CUDA_HOME
fi
unset _CUDA_NVCC_DIR

unset _SGLANG_ENV _VENV_NVIDIA _CUDA_LIB_PATHS _pkg _CPATH_EXTRA _inc
