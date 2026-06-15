# Nsight Systems CLI Reference

Source: https://docs.nvidia.com/nsight-systems/2024.6/UserGuide/index.html  
(Many options added in versions newer than the installed 2019.3.7.5)

## Command Structure

```
nsys [global-options] <command> [command-options] [app] [app-args]
```

## Global Options

| Flag | Description |
|------|-------------|
| `-h` / `--help` | Help |
| `-v` / `--version` | Version |

## Commands

| Command | Description |
|---------|-------------|
| `profile` | Launch app and collect in one step |
| `launch` | Launch app (interactive mode) |
| `start` | Start collection (interactive mode) |
| `stop` | Stop collection, app keeps running |
| `cancel` | Cancel collection, discard data |
| `shutdown` | Stop collection and disconnect |
| `sessions` | List active sessions |
| `stats` | Post-process report into summary tables |
| `analyze` | Expert system analysis (2021+) |
| `export` | Export `.nsys-rep` to other formats |
| `recipe` | Multi-report advanced analysis (2022+) |
| `status` | Check profiling environment suitability |

---

## `nsys profile` Options

```
nsys profile [options] <app> [app-args]
```

### Output Control

| Option | Default | Description |
|--------|---------|-------------|
| `-o <name>` / `--output=<name>` | `report#` | Output base filename. Supports `%q{ENV}`, `%h` (hostname), `%p` (pid), `%%` |
| `-f` / `--force-overwrite=true` | `false` | Overwrite existing files |
| `--auto-report-name=true` | `false` | Derive name from app name + GPU + API |
| `--stats=true` | `false` | Generate SQLite + summary stats after collection |

### Trace Selection

| Option | Default | Description |
|--------|---------|-------------|
| `-t <apis>` / `--trace=<apis>` | `cuda,opengl,nvtx,osrt` | Comma-separated list (no spaces) |

**Trace values**: `cuda`, `nvtx`, `cublas`, `cublas-verbose`, `cusparse`, `cusparse-verbose`, `cudnn`, `cudla`, `cudla-verbose`, `cusolver`, `cusolver-verbose`, `opengl`, `opengl-annotations`, `openacc`, `openmp`, `osrt`, `mpi`, `nvvideo`, `vulkan`, `vulkan-annotations`, `ucx`, `python-gil`, `syscall`, `none`

### Timing / Duration

| Option | Default | Description |
|--------|---------|-------------|
| `-y <s>` / `--delay=<s>` | `0` | Delay collection start by N seconds |
| `-d <s>` / `--duration=<s>` | unlimited | Stop collection after N seconds |
| `-x` / `--stop-on-exit=<bool>` | `true` | Stop when launched process exits |
| `--kill=<sig>` | `sigterm` | Signal to send at end: `none`, `sigkill`, `sigterm` |

### Capture Range

| Option | Values | Description |
|--------|--------|-------------|
| `-c` / `--capture-range=` | `none`, `cudaProfilerApi`, `hotkey`, `nvtx` | Only collect inside delimited range |
| `--capture-range-end=` | `stop`, `stop-shutdown`, `repeat[:N]` | What to do when range ends |
| `-p` / `--nvtx-capture=` | `range[@domain]` | NVTX range name to trigger on |
| `--hotkey-capture=` | `F1`–`F12` | Hotkey (default F12, graphics only) |

### CPU Sampling

| Option | Default | Description |
|--------|---------|-------------|
| `-s` / `--sample=` | `process-tree` | `process-tree`, `system-wide`, `none` |
| `-b` / `--backtrace=` | auto | `auto`, `fp`, `lbr`, `dwarf`, `none` |
| `--sampling-frequency=` | 1000 | 100–8000 Hz |
| `--cpuctxsw=` | `process-tree` | Thread scheduling trace: `process-tree`, `system-wide`, `none` |
| `--samples-per-backtrace=` | 1 | Higher = less data, less overhead |

### CUDA-Specific

| Option | Default | Description |
|--------|---------|-------------|
| `--cuda-flush-interval=<ms>` | 0 | Buffer flush interval (use 10000 for >30s runs) |
| `--cuda-graph-trace=` | `graph` | `graph` (less overhead) or `node` (per-node activity) |
| `--cuda-memory-usage=true` | `false` | Track GPU memory by kernel (high overhead) |
| `--cuda-um-cpu-page-faults=true` | `false` | Track CPU page faults for unified memory |
| `--cuda-um-gpu-page-faults=true` | `false` | Track GPU page faults for unified memory |
| `--cudabacktrace=` | `none` | Backtrace on CUDA API: `all`, `kernel`, `memory`, `sync`, `other` |
| `--gpuctxsw=true` | `false` | GPU context switches (requires driver r435.17+, root) |

### GPU Metrics (Turing+ only)

| Option | Default | Description |
|--------|---------|-------------|
| `--gpu-metrics-devices=` | `none` | `all`, specific GPU IDs, `help` to list |
| `--gpu-metrics-frequency=` | 10000 | 10–200000 Hz |
| `--gpu-metrics-set=` | auto | Metric set alias or `file:<path>` |

### Environment & Process

| Option | Default | Description |
|--------|---------|-------------|
| `-e A=B` / `--env-var=A=B` | — | Set env vars for launched app |
| `-n` / `--inherit-environment=` | `true` | Pass current env to app |
| `--run-as=<user>` | — | Run app as different user (root required) |
| `-w` / `--show-output=` | `true` | Show app stdout/stderr in terminal |
| `--wait=` | `all` | `primary` (wait on main proc) or `all` (wait on all) |
| `--trace-fork-before-exec=true` | `false` | Trace child processes before exec |

### Python Profiling

| Option | Default | Description |
|--------|---------|-------------|
| `--python-sampling=true` | `false` | Python backtrace sampling |
| `--python-sampling-frequency=` | 1000 | 1–2000 Hz |
| `--python-backtrace=` | `none` | `cuda` — collect Python stack on CUDA calls |

### OS Runtime

| Option | Default | Description |
|--------|---------|-------------|
| `--osrt-threshold=<ns>` | 1000 | Min duration to record OS runtime calls |
| `--osrt-backtrace-threshold=<ns>` | 80000 | Min duration for OS backtrace |
| `--osrt-backtrace-depth=` | 24 | Backtrace depth |

---

## `nsys stats` Options

```
nsys stats [options] <report.qdrep|report.nsys-rep|report.sqlite>
```

| Option | Default | Description |
|--------|---------|-------------|
| `-r <report>` / `--report=` | all defaults | Report script(s) to run |
| `-f <fmt>` / `--format=` | `column` | Output format: `column`, `table`, `csv`, `tsv`, `json`, `hdoc` |
| `-o <out>` / `--output=` | `-` (stdout) | `-` (console), filename, or `@command` |
| `--force-export=true` | `false` | Re-create SQLite even if it exists |
| `--timeunit=` | `nanoseconds` | `nsec`, `usec`, `msec`, `seconds` |
| `--sqlite=<file>` | auto | Specify SQLite filename explicitly |

### Built-in Report Scripts

**CUDA:**
- `cuda_api_sum` — CUDA API call times (CPU-side)
- `cuda_api_trace` — Per-call CUDA API trace
- `cuda_gpu_kern_sum` — GPU kernel execution summary
- `cuda_gpu_kern_gb_sum` — Same + grid/block dimensions
- `cuda_gpu_mem_time_sum` — H2D/D2H transfer timing
- `cuda_gpu_mem_size_sum` — H2D/D2H transfer sizes
- `cuda_gpu_sum` — Kernels + memops combined
- `cuda_gpu_trace` — Per-operation GPU trace
- `cuda_kern_exec_sum` — Kernel launch + queue + exec breakdown
- `cuda_api_gpu_sum` — Combined API + kernel + memop summary

**NVTX:**
- `nvtx_sum` — NVTX range summary (push/pop + start/end)
- `nvtx_pushpop_sum` / `nvtx_pushpop_trace`
- `nvtx_startend_sum`
- `nvtx_gpu_proj_sum` — NVTX ranges projected to GPU timeline
- `nvtx_kern_sum` — Kernels grouped by NVTX range

**System:**
- `osrt_sum` — OS runtime function summary
- `um_sum` — Unified memory summary
- `um_total_sum` — Unified memory totals

---

## `nsys export` Options

```
nsys export [options] <report.nsys-rep>
```

| Option | Default | Description |
|--------|---------|-------------|
| `-t <type>` / `--type=` | `sqlite` | `sqlite`, `json`, `text`, `arrow`, `parquetdir`, `hdf` |
| `-o <name>` / `--output=` | auto | Output filename |
| `-f` / `--force-overwrite=true` | `false` | Overwrite |
| `--tables=<pattern>` | all | Export only matching tables |
| `--times=<range>` | all | Export only events in time range |

---

## `nsys analyze` Options (2021+ only)

```
nsys analyze [options] <report>
```

### Available Rules
- `cuda_memcpy_async` — Async memcpy with pageable memory
- `cuda_memcpy_sync` — Synchronous memcpy
- `cuda_memset_sync` — Synchronous memset
- `cuda_api_sync` — Blocking sync APIs
- `gpu_gaps` — GPU idle > 500ms
- `gpu_time_util` — Low GPU utilization regions
- `dx12_mem_ops` — DX12 memory operation issues

---

## `nsys recipe` (2022+ only)

```
nsys recipe <recipe-name> [recipe-args]
nsys recipe --input <dir> cuda_gpu_kern_sum
```

Key recipes for ML/LLM work:
- `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`
- `nvtx_sum`, `nvtx_gpu_proj_sum`
- `nccl_sum`, `nccl_gpu_overlap_trace`
- `gpu_time_util`, `gpu_gaps`
- `cuda_kern_exec_sum` — breakdown into API/queue/kernel time
- `diff` — compare two runs statistically

---

## Example Command Sequences

```bash
# 1. Profile with CUDA + NVTX + cuBLAS
nsys profile \
  --trace=cuda,nvtx,cublas \
  --stats=true \
  -o my_run \
  python run.py

# 2. Print kernel summary
nsys stats --report=cuda_gpu_kern_sum --format=csv my_run.qdrep

# 3. Print NVTX summary
nsys stats --report=nvtx_sum my_run.qdrep

# 4. Export SQLite for custom queries
nsys export --type=sqlite my_run.qdrep

# 5. Multi-rank: use env var in filename
nsys profile \
  --trace=cuda,nvtx \
  -o rank_%q{RANK}.qdrep \
  mpirun -n 4 python distributed.py
```

---

## Version Compatibility Notes

| Feature | Min Version |
|---------|-------------|
| `.nsys-rep` format | 2020.x |
| `nsys analyze` | 2021.x |
| `nsys recipe` | 2022.x |
| `--python-sampling` | 2022.x |
| `--cuda-graph-trace=graph` | 2022.x (driver 515.43+) |
| GPU metrics (Turing+) | 2019.x |
| **Installed on this machine** | **2019.3.7.5** (use `.qdrep`) |
