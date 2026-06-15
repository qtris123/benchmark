# Deep-Dive Diagnostics

Reference for when the health-check script shows FAILs and you need to
understand the root cause from first principles.

---

## Tracing CUDA failures with strace

When `torch.cuda.is_available()` returns False but nvidia-smi works, strace
reveals exactly what is failing at the syscall level:

```bash
source /localhome/local-triv/.venv/bin/activate
strace -f -e trace=openat -o /tmp/cuda_strace.txt python3 -c "
import ctypes; lib = ctypes.CDLL('libcuda.so.1'); print(lib.cuInit(0))
"
# Look for nvidia device opens and their error codes
grep -E "nvidia|= -1 E[A-Z]" /tmp/cuda_strace.txt | head -20
```

Key lines to look for:

| strace line | Meaning |
|-------------|---------|
| `openat("/dev/nvidiactl", O_RDWR) = 10` | Driver open OK |
| `openat("/dev/nvidia-uvm", O_RDWR) = -1 ENODEV` | **nvidia_uvm broken** → fix A |
| `openat("/dev/nvidia-uvm", O_RDWR) = -1 EACCES` | Permission denied |
| `openat("libcudart.so.12", ...) = -1 ENOENT` | CUDA runtime not on path → fix B |

---

## How to read /proc/devices vs /dev/ major numbers

```bash
# What the kernel module registered:
awk '/nvidia/{print $1, $2}' /proc/devices
# Example output:
#   195 nvidia
#   195 nvidiactl
#   508 nvidia-uvm      ← this is what the kernel module registered
#   509 nvidia-nvswitch

# What the device FILES use:
stat -c '%t %n' /dev/nvidia-uvm  # hex major
# Example: 1fc /dev/nvidia-uvm   (1fc hex = 508 dec — MATCHES, good)
# Example: 1fd /dev/nvidia-uvm   (1fd hex = 509 dec — MISMATCH, bad)

# Hex to decimal:
printf '%d\n' "0x1fc"   # = 508
```

After `rmmod nvidia_uvm && modprobe nvidia_uvm`, the module dynamically
allocates a new major. The old `/dev/` file still has the old major.
The mismatch means opening `/dev/nvidia-uvm` silently opens the WRONG device
(whatever now owns that major — could be nvidia-nvswitch) → ENODEV.

---

## Identifying which libcudart a binary links against

```bash
# Check a shared library's NEEDED entries:
readelf -d <path-to.so> | grep NEEDED

# For vllm's bundled flash_attn:
readelf -d .venv/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
  | grep NEEDED
# Good: libcudart.so.12
# Bad:  libcudart.so.13  → CUDA 13, needs driver ≥580
```

---

## CUDA version compatibility quick reference

| Component | Version | Compatible with driver 570? |
|-----------|---------|------------------------------|
| PyTorch (cu126) | CUDA 12.6 | ✅ Yes |
| vllm_flash_attn in 0.22.0 | CUDA 13 | ❌ No (needs driver ≥580) |
| flashinfer JIT via cu13 nvcc | CUDA 13 binary | ❌ No (VLLM_USE_FLASHINFER_SAMPLER=0) |
| TRITON_ATTN backend | No CUDA binary | ✅ Yes |
| PyTorch-native sampler | No CUDA binary | ✅ Yes |

---

## Where CUDA libraries live in this venv

```
.venv/lib/python3.12/site-packages/nvidia/
├── cu13/
│   ├── bin/nvcc              ← CUDA 13 compiler
│   ├── include/              ← CUDA 13 headers
│   └── lib/libcudart.so.13   ← CUDA 13 runtime
├── cuda_runtime/
│   └── lib/
│       ├── libcudart.so.12   ← CUDA 12.6 runtime
│       └── libcudart.so      ← symlink → libcudart.so.12 (we created this)
├── curand/include/curand.h   ← needed for flashinfer JIT
└── cublas, cusparse, ...     ← other CUDA sub-libraries
```

The `env.sh` script sets:
- `LD_LIBRARY_PATH` → exposes these libs at Python import time
- `CUDA_HOME` → `nvidia/cu13` (has nvcc)
- `LIBRARY_PATH` → `nvidia/cuda_runtime/lib` (linker finds `libcudart.so`)
- `CPATH` → `nvidia/curand/include:...` (nvcc finds curand.h etc.)
