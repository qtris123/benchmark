# Source after: source .venv/bin/activate
# Usage: source benchmarks/env.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Locate the bundled nvidia libs inside whichever venv this checkout uses.
# (This tree uses vllm-env; the original skill layout used .venv at the root.)
VENV_NVIDIA=""
for _venv in vllm-env .venv; do
  if [[ -d "$ROOT/$_venv/lib/python3.12/site-packages/nvidia" ]]; then
    VENV_NVIDIA="$ROOT/$_venv/lib/python3.12/site-packages/nvidia"
    break
  fi
done

# LD_LIBRARY_PATH: expose bundled CUDA runtime libs so PyTorch can find them
CUDA_LIB_PATHS=()
for libdir in cu13 cu12 cuda_runtime lib; do
  [[ -d "$VENV_NVIDIA/$libdir/lib" ]] && CUDA_LIB_PATHS+=("$VENV_NVIDIA/$libdir/lib")
done
if ((${#CUDA_LIB_PATHS[@]})); then
  export LD_LIBRARY_PATH="$(IFS=:; echo "${CUDA_LIB_PATHS[*]}"):${LD_LIBRARY_PATH:-}"
fi

# CUDA_HOME + nvcc on PATH: required for flashinfer JIT kernel compilation.
# The cu13 package ships a full CUDA toolkit (nvcc, headers, libs).
if [[ -z "${CUDA_HOME:-}" ]]; then
  for _cu_dir in cu13 cu12; do
    if [[ -x "$VENV_NVIDIA/$_cu_dir/bin/nvcc" ]]; then
      export CUDA_HOME="$VENV_NVIDIA/$_cu_dir"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" && ":${PATH}:" != *":${CUDA_HOME}/bin:"* ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# LIBRARY_PATH: used by the linker (ld) at build time to resolve -lfoo flags.
# flashinfer links -lcudart; without this the linker can't find libcudart.so.
_LIB_PATHS=()
for _libdir in cuda_runtime cu13 cu12; do
  [[ -d "$VENV_NVIDIA/$_libdir/lib" ]] && _LIB_PATHS+=("$VENV_NVIDIA/$_libdir/lib")
done
if (( ${#_LIB_PATHS[@]} )); then
  export LIBRARY_PATH="$(IFS=:; echo "${_LIB_PATHS[*]}"):${LIBRARY_PATH:-}"
fi

# CPATH: nvcc forwards these to the host compiler as -I paths.
# flashinfer only passes CUDA_HOME/include but its kernels also need
# headers from other nvidia sub-packages (curand, nvjitlink, …).
_CPATH_EXTRA=()
for _inc_pkg in curand cuda_nvrtc nvjitlink cublas cusparse; do
  [[ -d "$VENV_NVIDIA/$_inc_pkg/include" ]] && _CPATH_EXTRA+=("$VENV_NVIDIA/$_inc_pkg/include")
done
if (( ${#_CPATH_EXTRA[@]} )); then
  export CPATH="$(IFS=:; echo "${_CPATH_EXTRA[*]}"):${CPATH:-}"
fi

export PATH="$HOME/.local/bin:$PATH"

# vllm 0.22.0 ships vllm_flash_attn compiled against libcudart.so.13 (CUDA 13),
# but the NVIDIA driver on this machine supports only CUDA 12.8 (driver 570).
# Using flash_attn causes "CUDA driver version is insufficient for CUDA runtime
# version". Use the pure-Triton backend which has no CUDA 13 dependency.
# Pass --attention-backend TRITON_ATTN to vllm serve (see run_pareto_benchmark.sh).

# Disable flashinfer's JIT-compiled sampler for the same reason: nvcc in the
# venv is from the cu13 package, producing CUDA 13 binaries the cu12 runtime
# rejects. PyTorch-native top-k/top-p is functionally equivalent.
export VLLM_USE_FLASHINFER_SAMPLER=0
