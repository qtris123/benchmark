---
name: nsight-systems
description: >-
  Profile GPU/CPU applications with NVIDIA Nsight Systems (nsys). Use when the
  user asks to profile, trace, or analyze GPU workloads with nsys, mentions
  nsight, nsys profile, nsys stats, NVTX, CUDA tracing, or wants to understand
  kernel execution times, memory transfer bottlenecks, or CPU-GPU overlap.
---

# NVIDIA Nsight Systems (nsys)

## This Machine

- **Binary (current)**: `/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys`
- **Symlink**: `/usr/local/bin/nsys` → managed by `update-alternatives`
- **Version**: 2024.6.2 — output format is `.nsys-rep`, supports `nsys stats`, `nsys analyze`
- **Legacy binary** (do not use): `/usr/lib/nsight-systems/Target-x86_64/x86_64/nsys` (2019.3.7.5, CUDA 10 only)
- **Installed via**: `apt-get install nsight-systems-2024.6.2` from NVIDIA CUDA public apt repo

### CUPTI fix applied (required for CUDA 12)
The 2019 binary needed libcupti.so.12.8 which didn't exist in its directory.
Symlinks were created to make the legacy binary usable if needed:
```bash
NSYS_DIR="/usr/lib/nsight-systems/Target-x86_64/x86_64"
CUPTI12="/localhome/local-triv/.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
sudo ln -sf "$CUPTI12" "$NSYS_DIR/libcupti.so.12.8"
sudo ln -sf "$CUPTI12" "$NSYS_DIR/libcupti.so.12"
sudo ln -sf "$CUPTI12" "$NSYS_DIR/libcupti.so"
```
The 2024.6.2 binary bundles its own CUPTI and does not need this fix.

---

## Quick Start

```bash
# Minimal — profile a script, write report.nsys-rep
nsys profile -o my_report python train.py

# For Python / ML workloads
nsys profile --trace=cuda,nvtx --sample=none -o my_report python train.py

# With stats summary printed to stdout
nsys profile --trace=cuda,nvtx --stats=true -o my_report python train.py

# Post-hoc stats from an existing report
nsys stats my_report.nsys-rep
```

---

## Common Recipes

### LLM / Inference server profiling (recommended pattern)

Use `launch` → `start` → `stop` for exact capture control. This is far superior
to `--delay` / `--duration` for server profiling because the capture window tracks
the actual benchmark, not a wall-clock timer.

```bash
SESSION="my_profile_$$"

# Step 1: inject libs into server + all child processes, do NOT record yet
nsys launch --trace=cuda,nvtx --session-new="$SESSION" -- python -m server_cmd &
LAUNCHER_PID=$!

# Step 2: wait for server ready, run warmup (not recorded)
wait_for_health && run_warmup

# Step 3: start recording exactly at benchmark start
nsys start --output my_profile --force-overwrite true --session="$SESSION"

# Step 4: run benchmark — everything here IS captured
run_benchmark

# Step 5: stop recording, report written immediately
nsys stop --session="$SESSION"

# Step 6: kill server
kill $LAUNCHER_PID
```

**Why not `--delay` / `--duration`?**
- `--delay=90` is a guess. If the server starts in 45s you're recording 45s of warmup noise.
- `--duration=120` silently truncates benchmarks that run longer.
- `launch`+`start`+`stop` gives you exactly the benchmark window, no more, no less.

### CUDA-only, no CPU overhead
```bash
nsys profile --trace=cuda --sample=none -o cuda_only ./your_app
```

### Profile a specific code region (requires code changes)
```bash
nsys profile --capture-range=cudaProfilerApi --trace=cuda,nvtx -o region_profile ./your_app
# In code: cudaProfilerStart() / cudaProfilerStop() around the region
```

---

## Key CLI Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `report#` | Output file base name |
| `--trace` | `-t` | `cuda,opengl,nvtx,osrt` | APIs to trace (comma-separated, no spaces) |
| `--stats` | | `false` | Generate summary stats after collection |
| `--sample` | `-s` | `process-tree` | CPU sampling scope (`none` to disable) |
| `--delay` | `-y` | `0` | Delay start by N seconds (avoid — use `launch`+`start`+`stop` instead) |
| `--duration` | `-d` | unlimited | Stop after N seconds (avoid — silently truncates) |
| `--capture-range` | `-c` | `none` | `cudaProfilerApi`, `nvtx`, `hotkey` |
| `--force-overwrite` | `-f` | `false` | Overwrite existing output files |
| `--session-new` | | | Name a new session (used with `launch`) |
| `--session` | | | Target a named session (used with `start`/`stop`) |

**`--trace` values**: `cuda`, `nvtx`, `cublas`, `cublas-verbose`, `cudnn`, `cusparse`, `osrt`, `opengl`, `openacc`, `openmp`, `mpi`, `none`

---

## Analyzing Results

```bash
# Print all default summary tables to terminal
nsys stats my_report.nsys-rep

# Specific reports
nsys stats --report cuda_api_sum my_report.nsys-rep
nsys stats --report cuda_gpu_kern_sum my_report.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum my_report.nsys-rep
nsys stats --report nvtx_sum my_report.nsys-rep

# Export to CSV
nsys stats --format=csv --output=. my_report.nsys-rep

# Export SQLite for custom queries
nsys export --type=sqlite -o my_report.sqlite my_report.nsys-rep

# Auto-detect known bad patterns (nsys 2021+)
nsys analyze my_report.nsys-rep
```

**Key stat reports**: `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, `cuda_gpu_mem_size_sum`, `nvtx_sum`, `osrt_sum`

---

## Profiling LLM Inference Servers — Known Issues & Solutions

### Problem 1: Zero CUDA events captured
**Symptom**: `.nsys-rep` / `.qdrep` file is tiny (8–9 KB), GUI shows "Zero CUDA events",
"CUDA injection initialization failed", "Could not locate CUPTI binary".

**Root cause**: nsys uses `dlopen()` to load CUPTI by exact filename (`libcupti.so.<major>.<minor>`)
from its own install directory. If the bundled CUPTI version doesn't match the running CUDA driver,
the file isn't found and CUDA interception fails silently. NVTX still works (it doesn't need CUPTI).

**Fix**: Symlink the system's CUPTI into the nsys directory (see CUPTI fix above), OR upgrade nsys
to a version that matches your CUDA driver.

**Verify fix**: `nsys profile ... python3 -c "import torch; torch.matmul(...)"` should report
`N total events collected` (not zero).

### Problem 2: nsys 2019 "Unknown API driver activity" errors
**Symptom**: Events are captured, but `QdstrmImporter` throws on driver API IDs 673, 677 etc.
"Unknown API driver activity" in the analysis output.

**Root cause**: nsys 2019's host-side event decoder has a hardcoded table of CUDA driver API IDs
that stops at CUDA 10.2. CUDA 12 added new APIs (e.g. `cuLaunchKernelEx`, `cuCtxGetId`) with IDs
beyond that table. Not fixable via symlinks — requires upgrading nsys.

**Fix**: Install `nsight-systems-2024.6.2`:
```bash
# Add NVIDIA CUDA public apt repo
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" \
  | sudo tee /etc/apt/sources.list.d/cuda-public.list
sudo apt-get update -o Dir::Etc::sourcelist="/etc/apt/sources.list.d/cuda-public.list" \
  -o Dir::Etc::sourceparts="-" -o APT::Get::List-Cleanup="0"
sudo apt-get install -y nsight-systems-2024.6.2
```

### Problem 3: Stale server on port causes false-positive health check
**Symptom**: Script prints "Server ready (0s)" immediately — model loading takes at least 5–10s,
so instant readiness means a leftover server from a previous run is holding the port.
The warmup and benchmark then hit the UN-instrumented old server.

**Fix**: Add a port-in-use guard before launching:
```bash
if ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT}[[:space:]]"; then
  echo "ERROR: port $SERVER_PORT already in use — kill stale server first"
  exit 1
fi
```

Kill stale servers:
```bash
kill $(ss -tlnp | awk '/:8000/{gsub(/.*pid=/,""); gsub(/,.*/,""); print}') 2>/dev/null
```

### Problem 4: `--delay` / `--duration` don't track benchmark completion
**Symptom**: Trace either contains warmup noise (delay too short) or gets truncated
mid-benchmark (duration too short), or contains long idle padding (benchmark finished early).

**Root cause**: `--delay` and `--duration` are wall-clock timers running independently inside nsys.
They have no knowledge of when the benchmark actually starts or ends.

**Fix**: Use `nsys launch` + `nsys start` + `nsys stop` (see recipe above). The capture window
becomes exactly the return-to-return span of `run_benchmark()`.

### Problem 5: vLLM NVTX annotations not emitted by default
**Symptom**: `nsys stats` reports "Zero NVTX events collected" even with CUDA events working.

**Root cause**: vLLM gates NVTX tracing behind a flag.

**Fix**: Pass `--collect-detailed-traces all` to `vllm serve`, or set:
```bash
export VLLM_TORCH_PROFILER_DIR=/tmp/vllm-nvtx
```

### Problem 6: GPU contention — two engines selected the same GPU simultaneously
**Symptom**: `vllm serve` crashes during engine core initialization with:
```
ValueError: Free memory on device cuda:0 (7.1/79.25 GiB) on startup is less than
desired GPU memory utilization (0.9, 71.33 GiB).
```
followed by `RuntimeError: Engine core initialization failed`.

**Root cause**: Two `nsight_profile.sh` sessions (one for sglang, one for vLLM) each called
`_pick_free_gpu()` at script startup, before either had loaded a model. Both snapshot showed
~81 GB free on GPU 0, both selected it. sglang loaded its model first (~72 GiB), leaving only
7 GiB by the time vLLM's engine tried to allocate.

The scoring function `(100 - util%) × 10000 + free_MiB` correctly prefers the free GPU when
one engine is already loaded, but fails when both scripts call it simultaneously before either
occupies memory.

**Fix**: Add a minimum free memory guard to `_pick_free_gpu()`:
```bash
MIN_FREE_MIB="${MIN_FREE_MIB:-40960}"   # 40 GiB — safe lower bound for LLaMA-3.1-8B

if (( best_free < MIN_FREE_MIB )); then
  echo "ERROR: best available GPU $best_idx only has ${best_free} MiB free" \
       "(need >= ${MIN_FREE_MIB} MiB). Is another server still running?" >&2
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
             --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /' >&2
  exit 1
fi
```
This fails fast with a useful error rather than spending minutes starting up only to crash inside the engine.

### Problem 7: Profiled benchmark gets `ECONNREFUSED` — `vllm bench serve` startup too slow
**Symptom**: The profiled benchmark (launched in background after warmup) exits with 0 successful
requests and `Benchmark duration: 0.07s`. Server log confirms it was alive the whole time.

**Root cause**: `vllm bench serve` has a slow startup path — on each invocation it loads the
tokenizer, processes the dataset, and initializes the aiohttp client. This takes 10–15 seconds.
The original fixed `PREFILL_SKIP_SEC=3` sleep started nsys and then killed the server after
`3 + DECODE_CAPTURE_SEC=5` = 8 seconds total. By the time `vllm bench serve` finished initializing
and sent its first request, the server had already been killed.

`sglang.bench_serving` doesn't hit this because it has a much lighter Python startup path.

**Fix**: Poll the benchmark's own bench.log for the startup completion marker instead of using
a fixed sleep:
```bash
# Wait until vllm bench serve has finished loading and is sending requests
while true; do
  if grep -qE "Starting main benchmark|Warmup completed" "$STEP_DIR/bench.log" 2>/dev/null; then
    echo "Benchmark client ready — requests in flight."
    break
  fi
  ! kill -0 "$BENCH_PID" 2>/dev/null && break   # bench already exited (error)
  (( _waited >= 90 )) && break                   # safety timeout
  sleep 1; (( _waited += 1 ))
done
# THEN sleep PREFILL_SKIP_SEC and start nsys
sleep "$PREFILL_SKIP_SEC"
nsys start ...
```
`"Starting main benchmark run"` is printed by `vllm bench serve` immediately before the
first HTTP request is fired. `"Warmup completed"` is the equivalent marker for `sglang.bench_serving`.

### Problem 8: Single-step 2.5s ITL spike at high concurrency — Triton JIT, not CUDA graph miss
**Symptom**: In the nsys GUI the decode timeline shows:
- **c=2**: immediately solid orange (Graphs row) at the start of decode — 92.5% CUDA Graphs
- **c=32**: 2.5-second solid **blue block** (Kernels row, individual eager launches) before
  transitioning to orange CUDA Graphs. CPU thread shows prolonged `cudaStream...` API call chains.

Confirmed in bench metrics: `Max ITL = 2480 ms` at c=32 while `Median ITL = 16 ms`.
The spike is a **single decode step outlier**, not a sustained slowdown.

**Root cause** (verified via sglang source `get_batch_sizes_to_capture()`):
```python
# Default capture_bs when cuda_graph_bs=None (no speculative decoding):
capture_bs = [1, 2, 4, 8] + list(range(16, 161, 8))
# → [1, 2, 4, 8, 16, 24, 32, 40, …] — batch=32 IS pre-compiled at startup
```
The CUDA graph for batch=32 already exists. The culprit is the **Triton attention kernel**
(`--attention-backend triton`), which runs **outside** the CUDA graph and is JIT-compiled
on first use per `(batch_size, kv_len)` shape bucket. The warmup ran `N_DECODE_STEPS=20`
output tokens, covering kv_len = INPUT_LEN+1 … INPUT_LEN+20. The profiled benchmark ran
`BENCH_OSL = N_DECODE_STEPS × 10 = 200` tokens. The first decode step that crosses into a
new shape bucket beyond kv_len = INPUT_LEN+20 triggers a cold Triton JIT compilation (~2.5s).

**Diagnosis**:
```bash
# Max ITL >> Median ITL in bench log = single JIT spike signature
grep -E "Mean ITL|Median ITL|Max ITL|P99 ITL" bench.log
# Example bad output:
#   Mean ITL (ms):    21.06    ← inflated by spike
#   Median ITL (ms):  16.26    ← true steady-state
#   Max ITL (ms):  2480.17    ← Triton JIT compilation
```

**Fix — two parts**:

*Part 1*: Explicitly cap `--cuda-graph-max-bs` to the maximum profiled concurrency
to avoid compiling unused larger graphs at startup:
```bash
MAX_C=$(printf '%s\n' "${PROFILE_CONCURRENCIES[@]}" | sort -n | tail -1)
# Add to sglang SERVER_CMD:
--cuda-graph-max-bs "$MAX_C"
```

*Part 2*: Add a **Triton burn-in** step after the timing warmup, before launching the
profiled background benchmark. The burn-in runs `BENCH_OSL` output tokens at batch=C,
pre-compiling all Triton kernel shapes for kv_len = INPUT_LEN+1 … INPUT_LEN+BENCH_OSL:
```bash
BURNIN_DIR="$PROFILE_DIR/${ENGINE}_c${C}_burnin"
mkdir -p "$BURNIN_DIR"
echo ">>> Triton burn-in (C=$C, OSL=$BENCH_OSL) ..."
(N_DECODE_STEPS=$BENCH_OSL; run_bench "$C" "$BURNIN_DIR" "$N_PROMPTS") 2>/dev/null || true
echo "    Burn-in done — JIT kernels cached, KV cache flushed."
```
The KV cache is flushed at the end of the burn-in (via `--flush-cache` inside `run_bench`),
so the profiled run starts from a clean state.

**Why the warmup alone is not sufficient**: the timing warmup uses `N_DECODE_STEPS=20`
(short, for fast TPOT measurement). It only covers kv_len up to INPUT_LEN+20. The profiled
benchmark extends to INPUT_LEN+BENCH_OSL. Any kv_len shape beyond the warmup's coverage is
a Triton JIT cold miss during the profiled window.

**Expected result after fix**: `Max ITL ≈ Median ITL ≈ 16 ms` (no outlier). The nsys trace
shows pure orange (CUDA Graphs) immediately at the start of decode for both c=2 and c=32.

---

## LLM Serving Profiling — Experiment Design

This section documents the design decisions for fair, apples-to-apples profiling of LLM
inference engines (vLLM vs sglang) across concurrency levels, as developed for the
`nsight_profile.sh` script on this machine.

---

### The Fairness Invariant

For each concurrency level C, always send **exactly N = C requests** at **concurrency = C**.

This guarantees:
- **Decode batch size = C throughout** — no partial batches at any concurrency level
- **Identical decode iterations** — every profiled window sees the same number of decode steps
- **Warmup pre-compiles CUDA graphs for batch=C** — the first profiled decode step is steady-state, not compilation
- **Burn-in pre-compiles Triton JIT kernels for the full kv_len range** — no single-step JIT spike inside the capture window (see Problem 8)

**What breaks fairness:**
- `N_PROMPTS=4` fixed regardless of C → at c=32 the decode batch is 4, not 32 (wrong workload)
- Warmup at `c=1, N=4` → CUDA graphs compiled for batch=1; at c=32 the first profiled step triggers recompilation (noise)
- Warmup OSL too short → Triton JIT kernels only compiled for kv_len up to INPUT_LEN+warmup_OSL; profiled run at larger OSL hits cold JIT miss mid-capture
- Identical prompts → sglang's radix-tree cache serves requests 2..N from cache, making prefill essentially free for sglang but not vLLM

---

### Prefill vs Decode Phases

**Prefill** — processes ISL input tokens for a new request:
- GPU tensor shape: `[B=1, T=ISL, d_model]` (one request at a time, full sequence length)
- Done **once per request**, compute-bound (large GEMM over T=ISL)
- vLLM chunks prefill via `--max-num-batched-tokens budget`: if `C × ISL > budget`, multiple chunked passes
- sglang: similar chunked scheduling; RadixAttention can skip prefill entirely for requests with a shared cached prefix

**Decode** — generates the next token for all in-flight requests simultaneously:
- GPU tensor shape: `[B=C, T=1, d_model]` (all C requests, one new token each)
- Repeated **OSL times per request**, memory-bandwidth bound at low C, compute-bound at high C
- CUDA graphs are keyed by batch size B=C; once compiled (warmup), every step is a graph replay

**For throughput/latency Pareto analysis, decode is the hot path** — it runs OSL times per prefill-once.

---

### Decode-Only Profiling (recommended)

Capturing prefill+decode together distorts `cuda_gpu_kern_sum` percentages because prefill
GEMMs (`[T=ISL, d]`) have different shapes than decode GEMMs (`[B=C, d]`), and at c=32
prefill can be ~31% of the trace window.

**Decode-only technique** (client-side timing, no server modification needed):

```bash
BENCH_OSL=$(( N_DECODE_STEPS * 10 ))    # large OSL ensures decode outlasts the capture window
PREFILL_SKIP_SEC=3                       # conservative: ISL=1000 prefill < 200ms on A100

# 1. Burn-in: pre-compile Triton JIT kernels for full kv_len range (kv_len INPUT_LEN+1 … INPUT_LEN+BENCH_OSL)
#    Warmup only covered INPUT_LEN+1 … INPUT_LEN+N_DECODE_STEPS; without burn-in, the first
#    step past that range causes a ~2.5s Triton JIT spike inside the capture window.
(N_DECODE_STEPS=$BENCH_OSL; run_bench "$C" "$BURNIN_DIR" "$N_PROMPTS") 2>/dev/null || true

# 2. Run benchmark in background with large OSL
(N_DECODE_STEPS=$BENCH_OSL; run_bench "$C" "$OUT_DIR" "$N_PROMPTS") &
BENCH_PID=$!

# 3. Wait for benchmark client to actually start sending requests before opening the window.
#    vllm bench serve has a 10-15s startup (tokenizer + dataset load); a fixed sleep would
#    start nsys before any requests are in flight.
_waited=0
while ! grep -qE "Starting main benchmark|Warmup completed" "$OUT_DIR/bench.log" 2>/dev/null; do
  sleep 1; (( _waited += 1 ))
  (( _waited >= 90 )) && break
done

# 4. Sleep past prefill (all C prefill passes complete in <<3s)
sleep "$PREFILL_SKIP_SEC"

# 5. Record only decode
nsys start --output "$PROFILE_OUT" --force-overwrite true --session="$SESSION"
sleep "$DECODE_CAPTURE_SEC"   # approx 2 x N_DECODE_STEPS x TPOT_warmup (derived from warmup log)
nsys stop --session="$SESSION"

# 6. Kill benchmark
kill "$BENCH_PID"; wait "$BENCH_PID" 2>/dev/null || true
```

**Prefill timing on A100 (ISL=1000):**
- C=2: ~60ms total prefill
- C=32: ~960ms total prefill (chunked, 4 passes of 8192 tokens)
- PREFILL_SKIP_SEC=3 is a 5–15x conservative margin

**Derive DECODE_CAPTURE_SEC from warmup TPOT:**
```bash
WARMUP_TPOT_MS=$(grep -i "mean tpot" "$WARMUP_DIR/bench.log" | grep -oP '[0-9]+\.[0-9]+' | head -1)
DECODE_CAPTURE_SEC=$(python3 -c "print(max(5, round($N_DECODE_STEPS * $WARMUP_TPOT_MS / 1000 * 2, 1)))")
```

**Three-phase warmup sequence (correct order)**:

| Phase | OSL | Purpose |
|-------|-----|---------|
| 1. Timing warmup | `N_DECODE_STEPS` (20) | Measure TPOT to derive `DECODE_CAPTURE_SEC`; compile CUDA graphs + Triton kernels for kv_len up to INPUT_LEN+20 |
| 2. Triton burn-in | `BENCH_OSL` (200) | Extend Triton JIT coverage to full kv_len range INPUT_LEN+1 … INPUT_LEN+BENCH_OSL; KV cache flushed at end |
| 3. Profiled benchmark | `BENCH_OSL` (200) | Actual capture — no JIT cold misses, all kernels pre-compiled |

---

### Dataset Design: Fixed ISL via Sliding Window

Using `--dataset-name random` or a single repeated prompt has two problems:
1. `random` input lengths vary per request — ISL is not controlled
2. Identical prompts → sglang RadixAttention cache hits after request 1 → prefill appears free

**Solution**: tokenize `random_text.txt`, generate N prompts via sliding window
(prompt i = tokens `[i : i + ISL]`), write as ShareGPT JSON:

```python
from transformers import AutoTokenizer
import json

tok     = AutoTokenizer.from_pretrained(model, local_files_only=True)
all_ids = tok.encode(open("random_text.txt").read(), add_special_tokens=False)

# Tile if text is shorter than needed
while len(all_ids) < ISL + N:
    all_ids = all_ids + all_ids

data = [
    {"conversations": [
        {"from": "human", "value": tok.decode(all_ids[i : i + ISL], skip_special_tokens=True)},
        {"from": "gpt",   "value": "x"},   # placeholder; overridden by --sharegpt-output-len
    ]}
    for i in range(N)
]
json.dump(data, open(dataset_path, "w"))
```

- Each prompt is ISL tokens, unique token sequence (different offset → no prefix sharing)
- `"gpt": "x"` placeholder is required: vLLM's ShareGPT loader filters out entries without a second turn
- OSL enforced at request time: `--sharegpt-output-len $OSL --ignore-eos` (vLLM) / `--disable-ignore-eos` (sglang)
- `random_text.txt` on this machine: **21,187 tokens** — supports up to 20,187 unique ISL=1000 prompts without tiling

---

### Output Length: Profiling vs Benchmarking

Each decode step replays the **same CUDA graph** — after step 2–3, the pattern is fully established.

| OSL | Decode steps | File size | Use case |
|---|---|---|---|
| 20 | 20 | small | **Profiling — decode-only capture** |
| 50 | 50 | medium | Profiling with more cycles visible |
| 200 | 200 | large | Extended profiling |
| 1000 | 1000 | very large | **Benchmarking throughput** (not profiling) |

For decode-only profiling, the actual `BENCH_OSL = N_DECODE_STEPS x 10` so decode
is still running when nsys starts. The trace only captures `~= N_DECODE_STEPS` steps.

---

### Full Capture vs Single-Component Capture

**For throughput/latency analysis: full capture is correct and sufficient.**

`--trace=cuda,nvtx` captures all kernels. Post-hoc analysis via `nsys stats --report cuda_gpu_kern_sum`
groups kernels by name to identify components:

| Kernel name pattern | Component |
|---|---|
| `ampere_h16816gemm_*`, `cutlass_*_gemm*` | Linear GEMM (FFN, QKV projections) |
| `flash_fwd_*`, `fmha_*`, `triton_*_attn*` | Attention |
| `topk_*`, `radix_sort_*`, `softmax_*` | Sampling |
| `rms_norm_*`, `silu_*` | Normalizations / Activations |

**When to use Nsight Compute (ncu) instead:**
- You need hardware counters (DRAM bandwidth %, arithmetic intensity, warp efficiency)
- You want to explain WHY a kernel is slow, not just HOW LONG it takes
- ncu collects per-kernel microarchitecture metrics; it cannot profile all kernels simultaneously

**Decision tree:**
```
Goal: explain throughput difference between engines / across concurrency
  -> nsys full capture  (correct)

Goal: understand why attention is slower in engine A (memory-bound? compute-bound?)
  -> ncu on that specific kernel  (correct)
```

---

### Pareto Benchmark Design (run_pareto_benchmark.sh)

For throughput/latency Pareto sweeps over concurrency levels:

| Parameter | Value | Rationale |
|---|---|---|
| `ISL` | 1000 (fixed) | Consistent KV cache size; sliding window prevents cache hits |
| `OSL` | 1000 (fixed) | `--sharegpt-output-len` + ignore-eos; decode dominates trace |
| `N = PROMPTS_MULT x C` | 3 x C | Exactly 3 complete decode waves per step; comparable across C |
| Dataset | `fixed_isl${ISL}.json` (sliding window) | Not ShareGPT variable lengths |
| Cache flush | `/reset_prefix_cache` (vLLM) / `--flush-cache` (sglang) | Clean state between concurrency steps |
| Server lifecycle | One server per experiment, all concurrency steps share it | No startup noise |

**Why `N = PROMPTS_MULT x C`:**
- All concurrency levels get the same number of complete waves (PROMPTS_MULT)
- TPOT and throughput are measured over the same number of decode batches
- Makes Pareto metrics directly comparable across C

**Alignment between profiling and Pareto:**
- Both use the same fixed ISL=1000 sliding-window dataset
- Both enforce OSL via `--sharegpt-output-len` + ignore-eos flags
- Profiling uses N=C (1 wave, short trace); Pareto uses N=3C (3 waves, throughput measurement)
- Decode CUDA kernels are identical — only trace duration differs

---

### Quick Reference: Concurrency Levels

- **c=2 (low)**: memory-bandwidth bound, small decode batch, GPU idle gaps between steps
- **c=32 (high)**: compute-bound region, large decode batch, dense continuous kernel activity
- These two points capture the two distinct GPU operating regimes on the Pareto curve

**Kernel % breakdown to look for (decode-only trace):**

| Component | c=2 (memory-BW bound) | c=32 (compute-bound) |
|---|---|---|
| Linear GEMM | lower % (small batch underutilizes SMs) | higher % (batch fills SMs) |
| Attention KV scan | higher % (reads long KV cache, BW-bound) | lower % (relatively) |
| Sampling | can be high at low C (per-request overhead) | amortized over large batch |

**Interpreting throughput differences:**
```
Engine A faster at c=32 -> likely better GEMM kernel (higher SM utilization)
Engine A faster at c=2  -> likely lower sampling/overhead per request
Engine A faster at both -> better overall kernel efficiency
```

## Reading Output

**CUDA API Summary** (`cuda_api_sum`): time spent in each CUDA API on the CPU.
**GPU Kernel Summary** (`cuda_gpu_kern_sum`): time each kernel ran on the GPU (sorted by total time).
**GPU MemOps Time** (`cuda_gpu_mem_time_sum`): H2D / D2H transfer durations.
**NVTX Summary** (`nvtx_sum`): time spent inside user-annotated NVTX ranges.

Load `.nsys-rep` in Nsight Systems GUI (local install needed):
- **CPU rows**: thread activity, API calls, sync points
- **GPU rows**: kernel execution, memory ops, hardware metrics
- **NVTX rows**: user-annotated ranges (batch boundaries, request lifecycle)
- Look for **GPU idle gaps** between decode steps (memory-bandwidth bound at low concurrency)
- Look for **prefill spikes** followed by dense decode trains
- Look for a **solid blue "Kernels" block** at the start of decode (not orange "Graphs"): this is a Triton JIT compilation event. Expect to also see dense `cudaStream...` API calls in the CPU scheduler thread during the same window. Confirm with `Max ITL >> Median ITL` in bench metrics.

**GUI version compatibility**: The local GUI version must be **≥ the server nsys version** that created the report.
The server runs **2024.6.2** — so the local GUI must be 2024.6.2 or newer.
Download the latest GUI for macOS/Windows/Linux: https://developer.nvidia.com/nsight-systems
Old `.qdrep` files (created by nsys 2019) can be opened by any GUI version.

---

## Common Performance Issues

| Pattern | Diagnosis | Fix |
|---------|-----------|-----|
| GPU idle gaps between decode steps | Small batch size (low concurrency), memory-BW bound | Increase concurrency, use continuous batching |
| Large H2D transfers | Pageable memory | Use pinned/unified memory |
| Many small kernels | Launch overhead | Kernel fusion, CUDA graphs |
| High `osrt` time | Blocking OS calls | Use async I/O, reduce allocations |
| Prefill dominates trace | Chunked prefill not enabled | Enable chunked prefill |
| Single 2–3s blue "Kernels" block at start of decode, then orange | Triton JIT cold miss — attention kernel not pre-compiled for this `(batch, kv_len)` bucket | Add burn-in step at `BENCH_OSL` tokens before profiled run (see Problem 8) |
| `Max ITL` >> `Median ITL` in bench metrics (e.g., 2480ms vs 16ms) | Single-step Triton JIT spike — confirms the blue block above is a one-time JIT event | Same fix as above |
| `--cuda-graph-max-batch-size` / `--cuda-graph-max-bs` flag | Often misdiagnosed as fixing CUDA graph miss — sglang default capture_bs already includes all powers-of-2 up to 160 | Check `get_batch_sizes_to_capture()` source; issue is almost always **Triton JIT**, not CUDA graph |
| Benchmark exits with 0 successful requests, duration 0.07s, ECONNREFUSED | Benchmark client (especially `vllm bench serve`) startup takes 10–15s — server was killed before first request sent | Poll bench.log for `"Starting main benchmark run"` before opening nsys window (see Problem 7) |
| `ValueError: Free memory ... less than desired GPU memory utilization` during engine init | Two nsight sessions started simultaneously and both selected the same GPU before either loaded a model | Add `MIN_FREE_MIB` guard to GPU picker; run sessions sequentially or on separate GPUs (see Problem 6) |

---

## Additional Reference

- Full CLI option details: [CLI_REFERENCE.md](CLI_REFERENCE.md)
- Stats reports guide: [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)
- Official docs: https://docs.nvidia.com/nsight-systems/2024.6/UserGuide/index.html
