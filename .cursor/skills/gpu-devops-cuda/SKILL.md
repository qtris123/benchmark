---
name: gpu-devops-cuda
description: >-
  Portable GPU/CUDA DevOps skill for any Linux server. Diagnoses and fixes
  CUDA initialisation failures, nvidia_uvm device file issues, missing
  environment variables, JIT compilation problems, and CUDA runtime vs driver
  version mismatches. Use when CUDA won't start, torch.cuda.is_available()
  returns False, vllm or any GPU workload crashes on startup, or when setting
  up a fresh GPU server. Works across different GPU models, driver versions,
  and venv layouts. For this specific leased machine (A100X, driver 570), also
  read gpu-devops-a100x-driver570.
---

# GPU DevOps — Portable CUDA Troubleshooting

## Step 0 — Run the auto-detect diagnostic first

```bash
bash ~/.cursor/skills/gpu-devops-cuda/scripts/cuda-health-check.sh
```

Pass the project root and venv path if they differ from the defaults:

```bash
bash ~/.cursor/skills/gpu-devops-cuda/scripts/cuda-health-check.sh \
  --root /path/to/project \
  --venv /path/to/.venv \
  --env  /path/to/env.sh   # optional: script that sets LD_LIBRARY_PATH etc.
```

The script auto-detects GPU model, driver version, and CUDA compatibility,
then reports pass/fail for each known failure mode.

---

## Failure mode A — nvidia_uvm broken state

**Symptoms:** `nvidia-smi` works. `torch.cuda.is_available()` → `False`.
`cuInit(0)` returns 999 (`CUDA_ERROR_UNKNOWN`).

**Root cause:** A crash (OOM, killed process) can leave the `nvidia_uvm`
kernel module in a broken state. `strace` will show:
```
openat("/dev/nvidia-uvm", O_RDWR) = -1 ENODEV
```

Even more subtly: after `rmmod` + `modprobe`, the module is assigned a new
dynamic major number, but the `/dev/nvidia-uvm` file still has the old one.
Opening it silently opens the wrong device → `ENODEV`.

**Fix:**
```bash
# 1. Reload the module
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm

# 2. Get the newly assigned major number
NEW=$(awk '/nvidia-uvm/{print $1}' /proc/devices)

# 3. Compare with device file (hex → decimal)
CUR=$(printf '%d' "0x$(stat -c '%t' /dev/nvidia-uvm)")

# 4. Recreate device files if they diverged
if [[ "$CUR" != "$NEW" ]]; then
  sudo rm -f /dev/nvidia-uvm /dev/nvidia-uvm-tools
  sudo mknod /dev/nvidia-uvm       c "$NEW" 0
  sudo mknod /dev/nvidia-uvm-tools c "$NEW" 1
  sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools
fi
```

**Permanent fix:** reboot — udev recreates device files with the correct major.

**How to diagnose from first principles:**
```bash
# 1. Confirm cuInit fails
python3 -c "import ctypes; lib=ctypes.CDLL('libcuda.so.1'); print(lib.cuInit(0))"
# 999 = CUDA_ERROR_UNKNOWN

# 2. Trace what's really happening
strace -f -e trace=openat python3 -c "import ctypes; ctypes.CDLL('libcuda.so.1').cuInit(0)" 2>&1 \
  | grep -E "nvidia|= -1 E"

# 3. Check major number alignment
echo "module: $(awk '/nvidia-uvm/{print $1}' /proc/devices)"
echo "device: $(printf '%d' "0x$(stat -c '%t' /dev/nvidia-uvm)")"
```

---

## Failure mode B — CUDA libraries not on LD_LIBRARY_PATH

**Symptoms:** `torch.cuda.is_available()` → `False` even after nvidia_uvm is fine.
Or: `OSError: libcudart.so.12: cannot open shared object file`.

**Root cause:** No system-level CUDA install. Libraries live inside the Python
venv (pip-installed `nvidia-*` packages). They must be on `LD_LIBRARY_PATH`
before Python starts.

**How to find them:**
```bash
VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
find "$VENV_SITE/nvidia" -name "libcudart*.so*" 2>/dev/null
```

**Minimum env.sh pattern:**
```bash
VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
NVIDIA="$VENV_SITE/nvidia"

# Add all nvidia sub-package lib dirs
for d in "$NVIDIA"/*/lib; do
  [[ -d "$d" ]] && LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH
```

---

## Failure mode C — JIT compilation (flashinfer / triton) can't find nvcc or headers

**Symptoms:** `RuntimeError: Could not find nvcc`. Or `fatal error: curand.h: No such file`.
Or `ld: cannot find -lcudart`. Or a C++ template error deep inside a torch header
(`ATen/core/List_inl.h`, `function_schema.h`) when nvcc compiles flashinfer kernels
(exit code 255 from ninja — means the compiler itself failed, not the kernel logic).

**Root causes:**
1. nvcc, CUDA headers, and CUDA stub libraries are all inside pip-installed
   `nvidia-*` packages — not in `/usr/local/cuda`.
2. **CUDA_HOME cross-environment leak:** If multiple venvs are used (e.g. vLLM
   and SGLang), sourcing one env's `env.sh` sets `CUDA_HOME` for a different
   CUDA version. The second env then picks up the wrong nvcc. The symptom is
   a C++ standard incompatibility error (GCC 13+ in cu13/nvcc rejecting torch
   headers compiled for an older GCC) rather than a missing-nvcc error.

**Check which nvcc would be used:**
```bash
echo "CUDA_HOME=$CUDA_HOME"
ls "${CUDA_HOME:-}/bin/nvcc" 2>/dev/null || echo "no nvcc at CUDA_HOME"
which nvcc 2>/dev/null || echo "no nvcc in PATH"
# If CUDA_HOME points to a different venv than your active one — that is the bug.
```

**Root cause:** nvcc, CUDA headers, and CUDA stub libraries are all inside
pip-installed `nvidia-*` packages — not in `/usr/local/cuda`.

**How to find them:**
```bash
NVIDIA="$(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia"

# Find nvcc
find "$NVIDIA" -name nvcc -type f 2>/dev/null

# Find headers
find "$NVIDIA" -name "cuda_runtime.h" -o -name "curand.h" 2>/dev/null | head -5

# Find libcudart
find "$NVIDIA" -name "libcudart*.so*" 2>/dev/null
```

**Important:** `pip install nvidia-cuda-nvcc-cu12` only installs `ptxas` (the PTX
assembler) and CUDA headers on Linux — it does NOT ship the `nvcc` front-end
driver. If `find "$NVIDIA" -name nvcc` returns nothing, there is no pip-provided
nvcc for that CUDA version. The only option is to use the system CUDA install,
or to force a Triton/no-JIT backend path.

**env.sh additions needed for JIT:**
```bash
# CUDA_HOME: directory that contains bin/nvcc and include/
export CUDA_HOME="$(find "$NVIDIA" -name nvcc -type f 2>/dev/null | head -1 | xargs dirname | xargs dirname)"
export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"

# CPATH: all nvidia include dirs (for curand.h etc.)
for d in "$NVIDIA"/*/include; do
  [[ -d "$d" ]] && CPATH="$d${CPATH:+:$CPATH}"
done
export CPATH

# LIBRARY_PATH: for the linker during build (-lcudart)
for d in "$NVIDIA"/*/lib; do
  [[ -d "$d" ]] && LIBRARY_PATH="$d${LIBRARY_PATH:+:$LIBRARY_PATH}"
done
export LIBRARY_PATH

# Create unversioned libcudart.so symlink if missing (linker needs it)
CUDA_RT_LIB="$(find "$NVIDIA/cuda_runtime/lib" -name 'libcudart.so.*' 2>/dev/null | sort -V | tail -1)"
if [[ -n "$CUDA_RT_LIB" && ! -e "${CUDA_RT_LIB%.*}" ]]; then
  STUB="${CUDA_RT_LIB%.*}"   # strip .12 or .13 → libcudart.so
  [[ "$STUB" == *.so ]] && ln -sf "$(basename "$CUDA_RT_LIB")" "$STUB"
fi
```

---

## Failure mode D — CUDA runtime version > driver max

**Symptoms:** `CUDA driver version is insufficient for CUDA runtime version`.
Or: `device kernel image is invalid` from flash_attn / flashinfer.

**Root cause:** A library bundled with the ML framework was compiled for a
newer CUDA than the installed driver supports.

**How to check driver's max CUDA version:**
```bash
# Method 1: nvidia-smi
nvidia-smi | grep "CUDA Version"

# Method 2: driver header
cat /usr/local/cuda/version.txt 2>/dev/null || true

# Method 3: from the driver version number
# Driver 520 → max CUDA 11.8
# Driver 525 → max CUDA 12.0
# Driver 535 → max CUDA 12.2
# Driver 545 → max CUDA 12.3
# Driver 550 → max CUDA 12.4
# Driver 555 → max CUDA 12.5
# Driver 560 → max CUDA 12.6
# Driver 565 → max CUDA 12.7
# Driver 570 → max CUDA 12.8   ← this machine
# Driver 575 → max CUDA 12.9
# Driver 580 → max CUDA 13.0
```

**How to identify which library is incompatible:**
```bash
# Find .so files that link against a too-new libcudart
find .venv -name "*.so*" -exec readelf -d {} \; 2>/dev/null \
  | awk '/NEEDED.*libcudart/{print FILENAME, $NF}' FILENAME=UNKNOWN \
  | grep -v "UNKNOWN"
# Better with context:
for so in $(find .venv -name "*.so" 2>/dev/null); do
  dep=$(readelf -d "$so" 2>/dev/null | awk '/NEEDED.*libcudart/{gsub(/[\[\]]/,"",$NF); print $NF}')
  [[ -n "$dep" ]] && echo "$dep  ←  $so"
done | sort
```

**Also check GPU architecture targeting in pre-compiled kernels:**
```bash
# For sgl_kernel (SGLang): check which GPU archs are compiled in
for f in sglang-env/lib/python3.12/site-packages/sgl_kernel/*.so; do
  echo "$(basename $f): $(strings "$f" 2>/dev/null | grep -E 'sm_[0-9]+' | sort -u)"
done
# If only sm_100a / sm_120a appear (Blackwell) but no sm_80 (A100) —
# the sgl_kernel package was built for newer hardware only.
# Fix: install an older sglang whose sgl_kernel includes sm_80.
# sglang 0.4.9.post6 (sgl-kernel==0.2.7) supports sm_80.
```

**Workaround (cannot upgrade driver):**
- Use an alternative backend that is pure Python/Triton with no CUDA binary
  dependency:
  - vLLM: `--attention-backend TRITON_ATTN` + `VLLM_USE_FLASHINFER_SAMPLER=0`
  - SGLang: `--attention-backend triton`
- Reinstall the framework pinned to a CUDA version matching the driver and
  a GPU architecture (sgl_kernel) that includes the target sm_XX.

---

## Quick reference — driver compatibility

| NVIDIA driver series | Max CUDA runtime |
|----------------------|-----------------|
| 520.x | 11.8 |
| 525–527 | 12.0 |
| 535–536 | 12.2 |
| 545–546 | 12.3 |
| 550–552 | 12.4 |
| 555–556 | 12.5 |
| 560–561 | 12.6 |
| 565–566 | 12.7 |
| 570–572 | 12.8 |
| 575–576 | 12.9 |
| 580+ | 13.0 |

---

## env.sh bootstrap template (copy-paste for a new server)

See [ENV_TEMPLATE.md](ENV_TEMPLATE.md) for a full portable `env.sh` that
auto-detects paths and sets everything needed.

## Additional resources

- Machine-specific notes for A100X / driver 570: `gpu-devops-a100x-driver570` skill
- Detailed strace walkthrough: [DIAGNOSTICS.md](../gpu-devops-a100x-driver570/DIAGNOSTICS.md)
