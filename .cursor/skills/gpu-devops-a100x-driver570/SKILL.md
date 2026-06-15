---
name: gpu-devops-a100x-driver570
description: >-
  Machine-specific GPU/CUDA fixes for THIS leased server (2x NVIDIA A100X,
  driver 570.133.20, max CUDA 12.8). Use for this machine only. For a fresh
  server with unknown GPU/driver, use the general gpu-devops-cuda skill instead.
  Covers: nvidia_uvm broken state, vllm TRITON_ATTN workaround, sglang install
  and triton-backend workaround, env.sh setup, and multi-GPU / tensor-parallel
  (TP>1) failures — invalid device ordinal, NCCL IB/RoCE registration failure,
  cuMem CUDA-graph deadlock, and vLLM custom all-reduce driver mismatch.
---

# GPU DevOps — A100X / Driver 570 (This Machine)

## Machine profile

| Item | Value |
|------|-------|
| GPUs | 2× NVIDIA A100X (81 920 MiB each) — compute capability **sm_80** |
| GPU↔GPU link | **PCIe across NUMA (`SYS`), NO NVLink** (GPU0 bus 0x40/NUMA0, GPU1 bus 0x8B/NUMA1). NVLink cannot be enabled — PCIe converged cards, no bridge. See Failure Mode G for TP>1. |
| Driver | 570.133.20 — max supported CUDA runtime: **12.8** |
| CUDA 13 support | **NO** — needs driver ≥ 580 |
| System Python | 3.8.10 (has no `venv` module — use `uv venv` instead) |
| vLLM venv | `/localhome/local-triv/.venv` — torch cu13, nvcc at `.venv/nvidia/cu13/bin/nvcc` |
| vLLM benchmark env | `benchmarks/env.sh` (sources after `source .venv/bin/activate`) |
| SGLang venv | `/localhome/local-triv/sglang-env` — torch 2.7.1+cu128 |
| SGLang benchmark env | `sglang-bench/env.sh` (sources after `source sglang-env/bin/activate`) |

---

## Step 0 — Run the diagnostic first

Before doing anything else, run the health-check script.
It prints a pass/fail summary for every known failure mode.

```bash
bash ~/.cursor/skills/gpu-devops-cuda/scripts/cuda-health-check.sh
```

---

## Known failure modes & fixes

### A — nvidia_uvm broken state (most common after a crash)

**Symptom:** `nvidia-smi` works, but `torch.cuda.is_available()` → `False`.
Underlying cause: `cuInit(0)` returns `CUDA_ERROR_UNKNOWN (999)`.
Root cause revealed by strace:
```
openat("/dev/nvidia-uvm", O_RDWR) = -1 ENODEV
```

**Why it happens:** A crashed vllm process can leave the `nvidia_uvm` kernel
module in a broken state. The device file exists but the underlying driver
rejects opens with `ENODEV`.

**Fix:**
```bash
# 1. Reload the module (it may get a new dynamic major number)
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm

# 2. Check the NEW major number
NEW_MAJOR=$(awk '/nvidia-uvm/{print $1}' /proc/devices)

# 3. Check the CURRENT device file major (in hex → decimal)
CUR_MAJOR=$(printf '%d' "0x$(stat -c '%t' /dev/nvidia-uvm)")

# 4. If they differ, recreate the device files
if [[ "$CUR_MAJOR" != "$NEW_MAJOR" ]]; then
  sudo rm /dev/nvidia-uvm /dev/nvidia-uvm-tools
  sudo mknod /dev/nvidia-uvm      c "$NEW_MAJOR" 0
  sudo mknod /dev/nvidia-uvm-tools c "$NEW_MAJOR" 1
  sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools
fi
```

**Permanent fix:** reboot (udev recreates device files correctly at boot).
The benchmark script (`run_pareto_benchmark.sh`) runs `_fix_nvidia_uvm`
automatically before every sweep.

---

### B — CUDA environment not set (LD_LIBRARY_PATH / CUDA_HOME missing)

**Symptom:** `torch.cuda.is_available()` returns False even after nvidia_uvm is
fine. Or `RuntimeError: Could not find nvcc`.

**Cause:** No system CUDA is installed. All CUDA libraries live inside the venv:
- Runtime: `.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12`
- nvcc:    `.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc`

These must be on the relevant paths before any CUDA code runs.

**Fix:** source `env.sh` after activating the venv:
```bash
cd /localhome/local-triv
source .venv/bin/activate
source benchmarks/env.sh
```

`env.sh` sets: `LD_LIBRARY_PATH`, `CUDA_HOME`, `PATH` (nvcc), `LIBRARY_PATH`
(linker), `CPATH` (headers for JIT compilation), and `VLLM_USE_FLASHINFER_SAMPLER=0`.

---

### C — vllm 0.22.0 CUDA-13 flash_attn vs driver-570

**Symptom:** vllm serve crashes with:
```
CUDA error: CUDA driver version is insufficient for CUDA runtime version
```
from `hardware_info.h` inside flash_attn.

**Cause:** vllm 0.22.0 bundles `vllm_flash_attn` compiled against
`libcudart.so.13` (CUDA 13). Driver 570 supports only CUDA 12.8.
Confirm with:
```bash
readelf -d .venv/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
  | grep NEEDED
# shows: libcudart.so.13  ← CUDA 13 dependency
```

**Fix (already applied in the benchmark script):**
```bash
# Pass to vllm serve:
--attention-backend TRITON_ATTN
# And env.sh already exports:
VLLM_USE_FLASHINFER_SAMPLER=0
```

**Do NOT** try to upgrade the driver to 580+ to "fix" this — the install script
(`install-nvidia-driver.sh`) explicitly pins to 570-server for this hardware.

---

### D — flashinfer JIT compilation fails

**Symptom:** `RuntimeError: device kernel image is invalid` from
`TopKMaskLogits`, or linker errors (`cannot find -lcudart`), or
`fatal error: curand.h: No such file or directory`.

**Cause:** flashinfer tries to JIT-compile CUDA kernels using nvcc from the
`nvidia/cu13` package (CUDA 13), but PyTorch was built for CUDA 12.6. The
resulting `.so` binary is CUDA-13 and rejected by the CUDA 12.6 runtime.

**Fix:** `VLLM_USE_FLASHINFER_SAMPLER=0` (set in `env.sh`) prevents the JIT
entirely. PyTorch-native top-k/top-p sampling is used instead — functionally
equivalent for benchmarking.

If flashinfer JIT compilation is needed for other reasons:
- `CUDA_HOME` must point to `nvidia/cu13`
- `CPATH` must include `nvidia/curand/include` (and other nvidia sub-packages)
- `LIBRARY_PATH` must include `nvidia/cuda_runtime/lib` (has `libcudart.so.12`)
- A `libcudart.so` symlink must exist: `ln -sf libcudart.so.12 nvidia/cuda_runtime/lib/libcudart.so`

---

### E — SGLang: sgl_kernel compiled for wrong GPU architecture

**Symptom:** SGLang server dies during CUDA graph capture with:
```
Exception: Capture cuda graph failed: Check failed: (status == cudaSuccess) is false:
RMSNorm failed with error code device kernel image is invalid
```

**Cause:** `sgl_kernel` was compiled exclusively for a newer GPU architecture
(e.g. Blackwell `sm_100a`/`sm_120a`) and contains no `sm_80` (A100) kernels.
The kernel binary is rejected at launch time.

**How to diagnose:**
```bash
# Check which GPU architectures are embedded in sgl_kernel .so files
for f in sglang-env/lib/python3.12/site-packages/sgl_kernel/*.so; do
  echo "=== $(basename $f) ==="
  strings "$f" 2>/dev/null | grep -E "sm_[0-9]+" | sort -u
done
# If you ONLY see sm_100a / sm_120a and NOT sm_80 — this is the problem.

# Also check installed versions
grep -E "^Name:|^Version:" \
  sglang-env/lib/python3.12/site-packages/sglang-*.dist-info/METADATA \
  sglang-env/lib/python3.12/site-packages/sgl_kernel-*.dist-info/METADATA 2>/dev/null
```

**Root cause detail:** `sgl_kernel` 0.3.x (bundled with `sglang` 0.5.x, uploaded
Jan 2026) targets Blackwell B100/B200 (`sm_100a`, `sm_120a`) only. A100 (`sm_80`)
support was only present in `sgl_kernel` 0.2.x and earlier.

**Fix:** Install a sglang version whose `sgl_kernel` includes `sm_80`.
`sglang 0.4.9.post6` uses `sgl-kernel==0.2.7` which supports A100.

```bash
cd /localhome/local-triv

# System Python 3.8 has no venv module — use uv
uv venv sglang-env --python 3.12
source sglang-env/bin/activate

# Install torch pinned to CUDA 12.8 (compatible with driver 570)
uv pip install \
  torch==2.7.1+cu128 \
  torchvision==0.22.1+cu128 \
  torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

# Install flashinfer (JIT version) before sglang to control index priority
uv pip install flashinfer-python==0.2.9rc2 \
  --extra-index-url https://flashinfer.ai/whl/cu128/torch2.7/

# Install sglang[srt] — torch==2.7.1+cu128 already satisfies torch==2.7.1
uv pip install "sglang[srt]==0.4.9.post6" \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://flashinfer.ai/whl/cu128/torch2.7/ \
  --index-strategy unsafe-best-match   # needed for torchao==0.9.0

# Verify sm_80 is present
strings sglang-env/lib/python3.12/site-packages/sgl_kernel/spatial_ops.abi3.so \
  | grep -E "sm_[0-9]+" | sort -u
# Must include sm_80
```

**Do NOT install sglang 0.5.x on this machine** — its sgl_kernel targets
Blackwell only and will always fail on A100.

---

### F — SGLang: flashinfer JIT compilation fails (cu13 nvcc / GCC 13 incompatibility)

**Symptom:** SGLang server starts, loads the model, then dies during CUDA graph
capture with a ninja build failure. Full error in `server.log`:
```
RuntimeError: Ninja build failed. Ninja output:
...
/localhome/local-triv/.venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc ...
ATen/core/List_inl.h:201:52: error: need 'typename' before 'decltype(...)::difference_type'
```

**Cause — identical pattern to vLLM Failure Mode C, different config knob:**
The `CUDA_HOME` environment variable is inherited from the vLLM `benchmarks/env.sh`
(which sets it to `.venv/nvidia/cu13`). flashinfer's JIT compiler finds `nvcc` at
`$CUDA_HOME/bin/nvcc` — the CUDA 13 compiler. cu13/nvcc uses GCC 13+ as its host
C++ compiler, which enforces stricter template rules and rejects a valid-in-older-GCC
expression in `torch 2.7.1`'s `ATen/core/List_inl.h:201`. The build fails with
exit code 255.

**Secondary cause:** `nvidia-cuda-nvcc-cu12` (pip package) only installs `ptxas`
and headers on Linux — it does NOT ship the `nvcc` front-end driver. There is no
CUDA 12.x `nvcc` available on this machine through pip.

**How to confirm:**
```bash
# Check which nvcc flashinfer would pick up
echo "CUDA_HOME=$CUDA_HOME"
ls "$CUDA_HOME/bin/nvcc" 2>/dev/null && echo "nvcc found" || echo "no nvcc in CUDA_HOME"

# Check what nvcc is in PATH
which nvcc 2>/dev/null || echo "no system nvcc"

# See the full compilation error
grep -A5 "FAILED\|List_inl" \
  sglang-bench/results/*/server.log 2>/dev/null | head -30
```

**Fix — two parts:**

**Part 1:** Override `CUDA_HOME` in `sglang-bench/env.sh` so the cu13 nvcc is
never found when the sglang env is active:
```bash
# Add to sglang-bench/env.sh (already done — shown for reference):
_CUDA_NVCC_DIR="$SGLANG_ENV/lib/python3.12/site-packages/nvidia/cuda_nvcc"
if [[ -d "$_CUDA_NVCC_DIR" ]]; then
  export CUDA_HOME="$_CUDA_NVCC_DIR"   # has headers + ptxas but no nvcc binary
else
  unset CUDA_HOME
fi
```

**Part 2:** Force the Triton attention backend — no JIT nvcc compilation needed:
```bash
# In sglang-bench/run_benchmark.sh (already applied):
ATTN_BACKEND_FLAG="--attention-backend triton"
# Pass this flag to: python3 -m sglang.launch_server ...
```

The Triton backend uses pre-compiled Triton kernels with no offline nvcc step.
It is well-optimised for A100 (sm_80) and produces valid benchmark results.

**Also clear the stale flashinfer cache** after applying the fix:
```bash
rm -rf ~/.cache/flashinfer/
```

**Analogy to vLLM:** This is structurally identical to vLLM Failure Mode C
(vllm_flash_attn built for CUDA 13, driver only supports 12.8):
| Engine | Problem | Fix |
|--------|---------|-----|
| vLLM | `vllm_flash_attn` links `libcudart.so.13` | `--attention-backend TRITON_ATTN` + `VLLM_USE_FLASHINFER_SAMPLER=0` |
| SGLang | flashinfer JIT uses cu13 nvcc → C++ error | `--attention-backend triton` |

In both cases the attention computation falls back to a Triton kernel path that
requires no incompatible CUDA binary.

---

### G — Multi-GPU / tensor-parallel (TP>1) failures

Running **any** TP>1 workload (sglang `--tp-size 2`, vLLM `--tensor-parallel-size 2`)
on this box hits a cluster of failures, all rooted in the same hardware reality:
**the two A100X are cross-NUMA over PCIe with NO NVLink** (`nvidia-smi topo -m` →
`SYS`), plus the usual CUDA-13-binary-vs-driver-570 mismatch. There are four
distinct failure modes; fix all of them before TP>1 will run reliably.

**TL;DR — the env + flags that make TP>1 work (all already applied in
`benchmark/run_static_benchmark.sh`):**
```bash
# Expose >= TP GPUs (cross-NUMA is fine):
export CUDA_VISIBLE_DEVICES=0,1
# Keep NCCL off the broken RoCE NICs and off the flaky cuMem allocator:
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
# vLLM only — its custom all-reduce is a CUDA-13 kernel:
#   vllm serve ... --disable-custom-all-reduce
```

For a 70B-class model that nearly fills VRAM, also: `--mem-fraction-static 0.94`
(sglang) / `--gpu-memory-utilization 0.94` (vLLM), a capped CUDA-graph batch size
(`--cuda-graph-max-bs 8` / `--cuda-graph-sizes 8`), and concurrency ≤ KV capacity
(see the KV-sizing note at the end).

---

#### G1 — `RuntimeError: CUDA error: invalid device ordinal`

**Symptom:** dies immediately in `set_device(self.gpu_id)` (sglang `tp_worker.py`)
during distributed init.

**Cause:** only **one** GPU is visible (e.g. an auto-GPU-picker exported a single
`CUDA_VISIBLE_DEVICES=0`) but TP=N requests N. Rank>0 workers call
`set_device(ordinal)` on a device that isn't in the visible set.

**Fix:** expose at least `TP` GPUs. A TP-aware GPU selector must export a
comma-separated list (`CUDA_VISIBLE_DEVICES=1,0`) and error out if fewer than `TP`
GPUs exist. Validate caller-supplied `CUDA_VISIBLE_DEVICES` has ≥ TP entries.

---

#### G2 — `NCCL error: unhandled system error` at init (`ibv_reg_mr_iova2`)

**Symptom:** crashes in `ncclCommInitRank`. With `NCCL_DEBUG=INFO`:
```
NCCL INFO NET/IB : Using [0]mlx5_0:1/RoCE ...
NCCL WARN Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory
ncclSystemError ... Call to ibv_reg_mr_iova2 failed
```

**Cause:** with no NVLink, NCCL routes inter-GPU traffic over the **RoCE NICs**
(`mlx5_*`). IB memory registration pins memory, but `ulimit -l` (RLIMIT_MEMLOCK) is
only **64 MB** (`65536`), so `ibv_reg_mr` fails with `ENOMEM`. `/dev/shm` is fine
(126 GB) — it's the IB transport, not shared memory.

**Fix:** `export NCCL_IB_DISABLE=1` — forces NCCL onto the PCIe-P2P path (which is
the right choice for two local GPUs and is fast). Do NOT try to disable P2P instead
(see G3).

---

#### G3 — sglang TP=2 decode hangs, then `Watchdog timeout` → SIGQUIT

**Symptom:** server loads, captures graphs, decodes fine for *minutes*, then a
forward pass freezes for >300 s. Both ranks log the watchdog at the **same instant**
(the signature of a collective deadlock), then SIGQUIT:
```
[TP0] Watchdog timeout (self.watchdog_timeout=300)
[TP1] Watchdog timeout (self.watchdog_timeout=300)
Received sigquit from a child process. It usually means the child failed.
```
The client then gets `ConnectionRefusedError [Errno 111]`.

**Cause:** the cuMem (VMM) allocator interacts badly with NCCL on this box. With the
default `NCCL_CUMEM_ENABLE=1`, CUDA-graph-captured collectives can **intermittently
deadlock** mid-run.

**Fix:** `export NCCL_CUMEM_ENABLE=0` (keep P2P **on**). Verified: a 40k-iteration
CUDA-graph all-reduce stress test runs at full speed with cuMem off (no throughput
penalty vs default).

**Do NOT use `NCCL_P2P_DISABLE=1` as the fix:** on this box it either fails init
outright (unless also paired with `NCCL_CUMEM_ENABLE=0`) or runs ~5× slower
(host-staged shared memory). Disabling cuMem keeps the fast PCIe-P2P path.

> Note: the all-reduce stress test could not *deterministically* reproduce the hang
> (the comm path is stable in isolation), so `NCCL_CUMEM_ENABLE=0` is a no-downside
> mitigation rather than a proven cure. For an 8B model that fits on one card,
> **TP=1 is the guaranteed-stable path** — only use TP>1 when weights truly don't fit.

---

#### G4 — vLLM TP>1: `cudaErrorInsufficientDriver` in `CustomAllreduce`

**Symptom:** vLLM `Engine core initialization failed`; the real traceback is in the
worker:
```
File ".../distributed/device_communicators/custom_all_reduce.py", line 175, in __init__
    self.meta_ptrs = self.create_shared_buffer(...)
torch.AcceleratorError: CUDA error: CUDA driver version is insufficient for CUDA runtime version
```
Only happens at **TP>1** (TP=1 never does an all-reduce, so it's unaffected).

**Cause:** vLLM 0.22.0's **custom all-reduce** is a CUDA-13-compiled kernel
(`_C_custom_ar`), but driver 570 only supports CUDA 12.8 → `cudaErrorInsufficientDriver`.
Same family as Failure Mode C (vLLM flash_attn) — a CUDA-13 binary on a 12.8 driver.

**Fix:** `vllm serve ... --disable-custom-all-reduce`. vLLM then uses the plain
**NCCL all-reduce** (`ncclAllReduce`, NCCL 2.28.9 via `PyNcclCommunicator`). No perf
loss here: custom all-reduce only helps on NVLink, which this box lacks.

---

#### Sizing note — KV cache & concurrency for large TP models

`--mem-fraction-static` / `--gpu-memory-utilization` reserve a **fixed fraction of
VRAM** (weights + KV pool), independent of model size — so an 8B and a 70B both look
like they "use all the VRAM" (8B = tiny weights + huge KV; 70B = huge weights + small
KV). For a 70B bf16 at TP=2 (≈66 GB weights/GPU), the leftover KV pool holds only
~53 k tokens. Max realized concurrency is bounded by:
```
effective_concurrency = min(client --max-concurrency,
                            server max-running-requests / max-num-seqs,
                            max_total_num_tokens / (ISL + OSL))
```
Exceeding the KV bound causes queueing/preemption that silently corrupts benchmark
numbers. For 70B at ISL=OSL=1000 → keep concurrency ≤ ~24.

---

## Key diagnostic commands

```bash
# 1. Is the driver loaded?
lsmod | grep nvidia

# 2. Can the device be opened?
python3 -c "import os; os.open('/dev/nvidia-uvm', os.O_RDWR)" && echo OK

# 3. Does cuInit succeed?
python3 -c "
import ctypes
lib = ctypes.CDLL('libcuda.so.1')
ret = lib.cuInit(0)
print('cuInit:', ret, '(0=OK, 999=CUDA_ERROR_UNKNOWN)')
"

# 4. Does PyTorch see GPUs? (vLLM env)
source .venv/bin/activate && source benchmarks/env.sh && python3 -c "
import torch
print('available:', torch.cuda.is_available())
print('count:', torch.cuda.device_count())
print('cuda:', torch.version.cuda)
"

# 5. Does PyTorch see GPUs? (SGLang env)
source sglang-env/bin/activate && source sglang-bench/env.sh && python3 -c "
import torch
print('available:', torch.cuda.is_available())
print('count:', torch.cuda.device_count())
print('cuda:', torch.version.cuda)
"

# 6. Check major number consistency (nvidia_uvm)
echo "Module major: $(awk '/nvidia-uvm/{print $1}' /proc/devices)"
echo "Device major: $(printf '%d' '0x'$(stat -c '%t' /dev/nvidia-uvm))"

# 7. Find what library flash_attn links against (vLLM)
readelf -d .venv/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so \
  | grep NEEDED

# 8. Check sgl_kernel GPU architectures (SGLang — must include sm_80 for A100)
for f in sglang-env/lib/python3.12/site-packages/sgl_kernel/*.so; do
  echo "$(basename $f): $(strings "$f" 2>/dev/null | grep -E 'sm_[0-9]+' | sort -u | tr '\n' ' ')"
done

# 9. Check which nvcc would be used for flashinfer JIT
echo "CUDA_HOME=$CUDA_HOME"
ls "${CUDA_HOME:-}/bin/nvcc" 2>/dev/null && echo "nvcc at CUDA_HOME" || echo "no nvcc at CUDA_HOME"
which nvcc 2>/dev/null || echo "no system nvcc in PATH"

# 10. (Multi-GPU/TP) Inter-GPU topology — expect "SYS" (cross-NUMA PCIe, no NVLink)
nvidia-smi topo -m
nvidia-smi nvlink --status   # "all links are inActive" → no NVLink (expected here)

# 11. (Multi-GPU/TP) Locked-memory limit — 65536 (64 MB) is why IB registration fails
ulimit -l
df -h /dev/shm               # 126 GB → shm is NOT the bottleneck; the RoCE NIC is

# 12. (Multi-GPU/TP) 2-GPU NCCL smoke test (should print "all_reduce OK -> 2.0")
CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 NCCL_CUMEM_ENABLE=0 \
  ./sglang-env/bin/python3 - <<'PY'
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
def w(r, n):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29591",
                      RANK=str(r), WORLD_SIZE=str(n))
    torch.cuda.set_device(r); dist.init_process_group("nccl", rank=r, world_size=n)
    t = torch.ones(8, device=f"cuda:{r}"); dist.all_reduce(t); torch.cuda.synchronize()
    if r == 0: print("all_reduce OK ->", t[0].item())
    dist.destroy_process_group()
if __name__ == "__main__": mp.spawn(w, args=(2,), nprocs=2, join=True)
PY
```

---

## Additional resources

- Detailed strace diagnostic procedure: see [DIAGNOSTICS.md](DIAGNOSTICS.md)
- vLLM benchmark env: `benchmarks/env.sh` + `benchmarks/run_pareto_benchmark.sh`
- SGLang benchmark env: `sglang-bench/env.sh` + `sglang-bench/run_benchmark.sh`
