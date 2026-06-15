# Nsight Systems Analysis Guide

Source: https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html

## Reading Stats Output

All time values are in **nanoseconds** by default. Use `--timeunit=msec` for human-friendly output.

---

## Stats Report Column Meanings

### `cuda_api_sum` — CUDA API Summary

| Column | Meaning |
|--------|---------|
| Time (%) | % of total CUDA API time |
| Total Time | Sum of all calls to this function |
| Num Calls | Call count |
| Avg / Med / Min / Max | Timing statistics |
| StdDev | Standard deviation |
| Name | CUDA function name |

> "Time %" is relative to other listed APIs, not wall-clock time.

### `cuda_gpu_kern_sum` — GPU Kernel Summary

Same columns as above but for **GPU-side** kernel execution time.  
Tip: Sort by Total Time to find hottest kernels.

### `cuda_gpu_kern_gb_sum` — Kernel Summary with Grid/Block

Adds:
| Column | Meaning |
|--------|---------|
| GridXYZ | Grid dimensions at launch |
| BlockXYZ | Block dimensions at launch |

### `cuda_gpu_mem_time_sum` — Memory Transfer Timing

| Column | Meaning |
|--------|---------|
| Operation | `[CUDA memcpy Host-to-Device]`, `[CUDA memcpy Device-to-Host]`, etc. |
| Time (%) | % of total memory transfer time |

### `cuda_gpu_mem_size_sum` — Memory Transfer Sizes

| Column | Meaning |
|--------|---------|
| Total (MB) | Total bytes transferred |
| Avg / Med / Min / Max | Per-transfer size statistics |

### `cuda_kern_exec_sum` — Kernel Launch Breakdown

| Column | Meaning |
|--------|---------|
| AAvg/AMed | API time: duration of the CPU launch call |
| QAvg/QMed | Queue time: gap between launch return and kernel start |
| KAvg/KMed | Kernel time: actual GPU execution time |
| TAvg/TMed | Total time: API start → kernel end |
| QCount | Kernels with positive queue time (GPU was busy) |

> Non-zero queue time = GPU was busy when kernel was submitted (not necessarily bad).

### `nvtx_sum` — NVTX Range Summary

| Column | Meaning |
|--------|---------|
| Style | Push/Pop or Start/End |
| Range | Range name |

### `nvtx_gpu_proj_sum` — NVTX→GPU Projection

Projects CPU NVTX ranges onto the GPU timeline (spans from first enclosed GPU op to last).  
Useful for: "how long did my forward pass actually run on GPU?"

| Column | Meaning |
|--------|---------|
| Total Proj Time | GPU time enclosed by this NVTX range |
| Total Range Time | CPU-side NVTX range duration |
| Total GPU Ops | Number of GPU ops inside the range |

---

## Expert System Rules (`nsys analyze`)

> Requires nsys 2021+. Not available on installed version 2019.3.7.5.

### Synchronization Rules
- **cuda_memcpy_async** — Async memcpy with pageable memory (becomes synchronous).  
  Fix: Use pinned memory (`cudaHostAlloc` or `cudaMallocHost`).
- **cuda_memcpy_sync** — `cudaMemcpy` blocking calls.  
  Fix: Replace with `cudaMemcpyAsync` + streams.
- **cuda_memset_sync** — Synchronous `cudaMemset`.  
  Fix: Use `cudaMemsetAsync`.
- **cuda_api_sync** — `cudaDeviceSynchronize`, `cudaStreamSynchronize` etc.  
  Fix: Use `cudaStreamWaitEvent` / `cudaEventSynchronize` instead.

### GPU Utilization Rules
- **gpu_gaps** — GPU idle > 500ms.  
  Fix: Use CUDA streams for overlap; instrument CPU code with NVTX to find root cause.
- **gpu_time_util** — Low GPU utilization regions.  
  Fix: Look for CPU bottlenecks via sampling + OS runtime backtraces.

---

## Common Analysis Workflows

### "Where is time going?" (CUDA workloads)

```bash
nsys stats --report=cuda_gpu_kern_sum --timeunit=msec report.qdrep
nsys stats --report=cuda_api_sum --timeunit=msec report.qdrep
nsys stats --report=cuda_gpu_mem_time_sum --timeunit=msec report.qdrep
```

1. Check `cuda_gpu_kern_sum` → find top kernels by total time
2. Check `cuda_api_sum` → find if CPU-side launch overhead is significant
3. Check `cuda_gpu_mem_time_sum` → find if H2D/D2H dominates

### "Are my NVTX regions covering the right time?" 

```bash
nsys stats --report=nvtx_gpu_proj_sum report.qdrep
```
Compare `Total Range Time` (CPU) vs `Total Proj Time` (GPU). Large gap = async launch overhead or CPU bottleneck.

### "Why is my GPU sometimes idle?"

Look for gaps in GPU row in GUI. On CLI:
```bash
nsys analyze --rule=gpu_gaps report.nsys-rep   # 2021+ only
```
Manual equivalent: compare `cuda_gpu_kern_sum` total time against wall time.

### "Is my memory transfer pageable (slow)?"

```bash
nsys analyze --rule=cuda_memcpy_async report.nsys-rep   # 2021+ only
```
Or visually: look for `cudaMemcpyAsync` in `cuda_api_sum` with high time.

### Exporting for custom analysis

```bash
# SQLite → query with Python / pandas
nsys export --type=sqlite report.qdrep

# Python example
import sqlite3, pandas as pd
conn = sqlite3.connect("report.sqlite")
df = pd.read_sql("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 100", conn)
```

---

## Advanced Recipes (`nsys recipe`, 2022+)

Not available with installed version but useful reference:

| Recipe | Use Case |
|--------|---------|
| `cuda_gpu_kern_pace` | Track kernel consistency across iterations |
| `cuda_gpu_kern_hist` | Kernel duration distribution |
| `nccl_gpu_overlap_trace` | NCCL comm vs compute overlap |
| `nvtx_gpu_proj_pace` | NVTX region GPU projection consistency |
| `gpu_time_util` | Per-time-chunk GPU utilization heatmap |
| `diff` | Compare two profiling runs statistically |
| `mpi_gpu_time_util_map` | MPI + GPU activity heatmap |

---

## Upgrading nsys

The installed version (2019.3.7.5) is missing many modern features. To upgrade:

```bash
# Option 1: Install via pip (wraps the official binary)
pip install nvidia-pyindex
pip install nvidia-nsight-systems

# Option 2: Download from NVIDIA
# https://developer.nvidia.com/gameworksdownload#?dn=nsight-systems-2024-6-1-90

# Option 3: Install via CUDA toolkit (preferred)
# wget and install a newer CUDA toolkit which bundles nsys 2024.x
```

After install, verify:
```bash
nsys --version  # should show 2024.x
```
