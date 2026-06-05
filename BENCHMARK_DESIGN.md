# Pareto Benchmark Design

This document explains the design decisions behind the throughput/latency Pareto benchmark
(`run_pareto_benchmark.sh`) and the Nsight Systems profiling script (`nsight_profile.sh`).
It covers what is measured, how inputs are constructed, how each engine receives requests,
and every mechanism that enforces a fair apples-to-apples comparison between vLLM and SGLang.

---

## Table of Contents

1. [What is Being Measured](#1-what-is-being-measured)
2. [Concurrency Sweep Design](#2-concurrency-sweep-design)
3. [Dataset: Fixed ISL via Sliding Window](#3-dataset-fixed-isl-via-sliding-window)
4. [Server Configuration](#4-server-configuration)
5. [Benchmark Client Invocation](#5-benchmark-client-invocation)
6. [Output Length Enforcement](#6-output-length-enforcement)
7. [KV Cache Flush Between Steps](#7-kv-cache-flush-between-steps)
8. [Metrics: Collection and Normalization](#8-metrics-collection-and-normalization)
9. [Pareto Chart](#9-pareto-chart)
10. [Nsight Profiling Design](#10-nsight-profiling-design)
11. [Fairness Invariants and What Breaks Them](#11-fairness-invariants-and-what-breaks-them)
12. [Version History: v1 (vllm-bench) vs v2 (unified)](#12-version-history-v1-vllm-bench-vs-v2-unified)

---

## 1. What is Being Measured

The benchmark sweeps **max concurrency** (number of simultaneous in-flight requests) from
low to high and records, at each level, the trade-off between:

| Metric | Definition | Direction |
|--------|------------|-----------|
| **tok/s/GPU** (system throughput) | Total output tokens generated per second, divided by the number of GPUs | ↑ higher is better |
| **tok/s/user** (per-user generation speed) | `1000 / TPOT_ms` — how many tokens per second a single user receives | ↑ higher is better |
| **TTFT** (time-to-first-token) | Latency from request arrival to first generated token | ↓ lower is better |
| **E2E latency** | Total time from request arrival to last token | ↓ lower is better |
| **ITL / TPOT** | Inter-token latency / time-per-output-token during decode | ↓ lower is better |

These metrics trace a **Pareto frontier**: increasing concurrency raises GPU throughput
(batching amortizes fixed per-step overhead) while degrading per-user speed (each user
waits longer for their tokens because the GPU is shared). The curve characterises the
efficiency and latency profile of the engine.

---

## 2. Concurrency Sweep Design

### Default sweep

```
CONCURRENCY_STEPS = 1 2 4 8 16 32 64 128   (doubling steps)
```

Each step runs a **single, independent benchmark run** with `max_concurrency = C`.

### Number of prompts per step: `N = PROMPTS_MULT × C`

```bash
PROMPTS_MULT=3   # default
N = PROMPTS_MULT * C
```

At each concurrency level C, `N = 3C` requests are submitted with `request_rate = inf`
(as fast as possible). With C slots and 3C requests, the server processes exactly
**3 complete decode "waves"** before all requests are done.

**Why `N ∝ C` and not a fixed N?**

| Approach | Problem |
|----------|---------|
| Fixed N (e.g. N=128 for all C) | At c=1, 128 sequential requests measure single-request throughput, not concurrency. Throughput number is not comparable to c=128 (which was actually at max load). |
| N = C (1 wave) | Statistical noise is high; one outlier request dominates ITL. Also used for profiling (exact 1 wave = exact batch size throughout). |
| **N = 3×C (3 waves)** | Each step is a sustained-load measurement: the GPU is continuously busy for 3 full decode cycles, which is long enough to amortise queue fill/drain transients and produce stable throughput and latency numbers. |

### Warm state vs cold state

The server runs **continuously** across all concurrency steps. Between steps, only the
KV cache is flushed (see §7). This means:
- Model weights remain loaded (no repeated startup cost)
- CUDA graphs compiled during earlier steps remain in memory (no JIT penalty at later steps)
- The server is in a consistent warm state at every step

---

## 3. Dataset: Fixed ISL via Sliding Window

### Why not use ShareGPT directly?

The raw ShareGPT dataset has **variable input lengths**. If ISL varies, the prefill cost
varies between requests, making it impossible to isolate decode performance, and the
measured throughput numbers depend on the accidental distribution of input lengths in the
sample.

### The sliding-window construction

```python
tok     = AutoTokenizer.from_pretrained(model)
all_ids = tok.encode(open("random_text.txt").read(), add_special_tokens=False)

# Tile if the source text is too short
while len(all_ids) < ISL + N:
    all_ids = all_ids + all_ids

for i in range(N):
    prompt = tok.decode(all_ids[i : i + ISL], skip_special_tokens=True)
    data.append({"conversations": [
        {"from": "human", "value": prompt},
        {"from": "gpt",   "value": "x"},   # placeholder, overridden at request time
    ]})
```

This produces N prompts where:
- **Prompt i** = tokens `[i : i + ISL]` of a long random text
- Every prompt is exactly **ISL tokens** when re-tokenized (guaranteed by construction)
- Adjacent prompts differ by exactly one token (a one-token slide), so **no two prompts
  share a long common prefix**

The last point is critical for sglang: its RadixAttention cache stores KV entries indexed
by token-sequence prefixes. Identical or similar prefixes would be served from cache,
making sglang's prefill appear artificially fast (O(unique tokens) instead of O(ISL)).
The sliding-window design ensures every prompt is unique and cache-miss, so both engines
pay the full ISL-token prefill cost for every request.

### Dataset file reuse

The dataset is written once to `datasets/fixed_isl${ISL}.json` and reused for all
concurrency steps and across multiple runs. The engine-specific tokenizers (vLLM and
sglang both use the model's HuggingFace tokenizer) will see the same token sequences.

### `random_text.txt`

The source text on this machine has **21,187 tokens** after tokenization by the Llama-3.1
tokenizer. This supports up to `21,187 - 1,000 = 20,187` unique ISL=1000 prompts without
tiling. The largest concurrency step (c=128) with `PROMPTS_MULT=3` needs `3×128 = 384`
prompts — well within the available budget.

---

## 4. Server Configuration

Both engines are configured to match on every parameter that affects measured throughput
or latency. Parameters that are engine-specific knobs but serve the same purpose are set
to equivalent values.

| Parameter | vLLM | sglang | Notes |
|-----------|------|--------|-------|
| Model | `meta-llama/Meta-Llama-3.1-8B-Instruct` | same | |
| dtype | `auto` (bfloat16) | `auto` | |
| Tensor parallelism | `--tensor-parallel-size $TP` | `--tp-size $TP` | Default TP=1 |
| GPU memory utilization | `--gpu-memory-utilization 0.90` | `--mem-fraction-static 0.90` | 90% of VRAM allocated to KV cache |
| Max simultaneous requests | `--max-num-seqs 256` | `--max-running-requests 256` | Caps concurrent decode slots |
| Chunked prefill budget | `--max-num-batched-tokens 8192` | (default) | vLLM explicit; sglang default is similar |
| Attention backend | `--attention-backend TRITON_ATTN` | `--attention-backend triton` | Flash-attn cu13 is incompatible with driver 570 (max CUDA 12.8); Triton backend is the working alternative |
| Port | `8000` | `30000` | Separate ports so both can run simultaneously |

### Why TRITON_ATTN?

vLLM 0.22+ bundles `vllm_flash_attn` compiled against `libcudart.so.13` (CUDA 13). This
machine runs driver 570 which supports only CUDA 12.8. Loading `libcudart.so.13` at
runtime crashes the engine with:

```
CUDA error: CUDA driver version is insufficient for CUDA runtime version
```

`--attention-backend TRITON_ATTN` routes all attention through a Triton kernel path that
compiles JIT against the installed CUDA 12 runtime. Performance is equivalent for
benchmarking purposes.

---

## 5. Benchmark Client Invocation

Both engines receive requests via their respective benchmark clients. The clients differ
in syntax but are configured to match on workload parameters.

### vLLM client (`vllm bench serve`)

```bash
vllm bench serve \
  --model                "$MODEL" \
  --backend              vllm \
  --endpoint             /v1/completions \
  --dataset-name         sharegpt \
  --dataset-path         "$DATASET_FILE" \
  --sharegpt-output-len  "$OSL" \      # enforce exact output length
  --num-prompts          "$N" \        # total requests to send
  --max-concurrency      "$C" \        # max simultaneous in-flight requests
  --host                 localhost \
  --port                 8000 \
  --ignore-eos \                       # continue generating past EOS to reach OSL
  --save-result \                      # write JSON result file
  --result-dir           "$RUN_DIR"
```

**`--ignore-eos`**: Without this flag, the model might emit an EOS token before reaching
`OSL` tokens, terminating the request early. This would produce variable output lengths
across requests, making TPOT and throughput comparisons meaningless. `--ignore-eos` forces
exactly `OSL` tokens regardless.

**`--max-concurrency C`**: The client maintains exactly C requests in-flight at all times.
When one request completes, the next is dispatched immediately. This differs from sending
all N requests simultaneously — it ensures the decode batch is always size C (no idle
slots), which is the workload the Pareto curve represents.

**`request_rate = inf`**: Requests are dispatched as fast as possible. This is a max-load
benchmark, not a sustained-RPS test.

### sglang client (`sglang.bench_serving`)

```bash
python3 -m sglang.bench_serving \
  --backend              sglang \
  --host                 "127.0.0.1" \
  --port                 30000 \
  --model                "$MODEL" \
  --dataset-name         sharegpt \
  --dataset-path         "$DATASET_FILE" \
  --sharegpt-output-len  "$OSL" \      # enforce exact output length
  --num-prompts          "$N" \
  --request-rate         inf \
  --max-concurrency      "$C" \
  --output-file          "$OUT_DIR/bench_results.jsonl" \
  --output-details \
  --disable-ignore-eos \               # sglang's flag for the same effect as vllm's --ignore-eos
  --flush-cache                        # flush KV cache before this run
```

**`--disable-ignore-eos`**: sglang's naming is inverted relative to vLLM. The flag means
"disable the ability to ignore EOS" → the benchmark enforces that requests complete at
exactly `OSL` tokens. Same semantics as `--ignore-eos` in vLLM.

**`--flush-cache`**: sglang uses RadixAttention, which caches prefix KV entries across
requests. If a previous run used overlapping prompts (or even similar ones), the cache
would serve KV entries rather than computing them, making prefill appear artificially fast.
`--flush-cache` calls sglang's `/flush_cache` endpoint before the benchmark starts,
ensuring a cold KV cache state identical to vLLM's (which has no cross-request caching).

---

## 6. Output Length Enforcement

Both engines are forced to generate **exactly `OSL` tokens** per request.

| Engine | Flag | Mechanism |
|--------|------|-----------|
| vLLM | `--sharegpt-output-len $OSL --ignore-eos` | Client overrides the ShareGPT dataset's GPT response with a fixed output length; server continues past EOS |
| sglang | `--sharegpt-output-len $OSL --disable-ignore-eos` | Same mechanism; flag name is inverted |

**Why this matters for fairness**: If output lengths varied, faster engines could appear
to have better throughput simply because their requests happened to get shorter outputs
from the token sampler. Fixing `OSL=1000` means every request generates exactly 1000
tokens. Throughput and TPOT are then measuring the same work.

The ShareGPT JSON `"gpt"` field (the placeholder `"x"`) is entirely ignored — both
clients treat `--sharegpt-output-len` as the authoritative output length.

---

## 7. KV Cache Flush Between Steps

Between concurrency steps, both engines flush their KV caches to prevent earlier steps
from warming the cache for later ones.

| Engine | Method | API endpoint |
|--------|--------|--------------|
| vLLM | `curl -X POST /reset_prefix_cache` | Called explicitly before each `vllm bench serve` invocation |
| sglang | `--flush-cache` flag on the client | sglang.bench_serving calls `/flush_cache` before sending requests |

**Why this matters**: vLLM (v0.4+) maintains a prefix cache that maps KV entries to token
sequences. Without flushing, the second run at a different concurrency level would partly
hit the cache from the first run, making its prefill faster than it would be for a fresh
user with no cached context. SGLang's RadixAttention does the same. Flushing between steps
ensures every step measures cold-start prefill performance — the same condition a freshly
started server would see.

**The server is NOT restarted between steps.** Only the KV data is cleared. Model weights
and CUDA graphs remain loaded. This is intentional: we are measuring the engine's
steady-state decode performance, not its startup cost.

---

## 8. Metrics: Collection and Normalization

### vLLM output format

`vllm bench serve` with `--save-result` writes one `*.json` file per run containing:

```json
{
  "output_throughput": 870.0,
  "mean_ttft_ms": 2017.3,
  "mean_tpot_ms": 34.77,
  "p99_ttft_ms": ...,
  "p99_tpot_ms": ...,
  ...
}
```

**Note**: vLLM reports `mean_tpot_ms` but not `mean_itl_ms` in older versions. The
normalization step treats them as equivalent: for pure decode batches (no interleaved
prefill), ITL ≈ TPOT because each inter-token gap is one full decode forward pass.

**E2E latency**: vLLM does not always report `mean_e2el_ms` directly. It is approximated
as:
```
mean_e2el_ms = mean_ttft_ms + OSL × mean_tpot_ms
```

### sglang output format

`sglang.bench_serving` with `--output-file` writes a JSONL file; the last line is the
aggregate result for the run, which already contains `mean_itl_ms`, `mean_tpot_ms`,
`mean_ttft_ms`, `output_throughput`, and `max_concurrency`.

### Normalization into `summary.jsonl`

Both engines append one line per concurrency step to a shared `summary.jsonl` using a
common schema. The derived metrics are:

| Field | Formula |
|-------|---------|
| `output_throughput` | Total output tokens / benchmark wall-clock seconds |
| `tok/s/GPU` | `output_throughput / TP` (TP=1 here, so same as output_throughput) |
| `tok/s/user` | `1000.0 / mean_tpot_ms` — how many output tokens/second a single user experiences |
| `mean_e2el_ms` | `mean_ttft_ms + OSL × mean_tpot_ms` (estimated if not directly available) |

**`tok/s/user` vs `output_throughput / C`**: The Pareto frontier in `compare.sh` uses
`1000 / mean_tpot_ms` for per-user speed for sglang (from `summary.jsonl`) but
`output_throughput / C` for vLLM (from the per-concurrency JSON files, which predate
`summary.jsonl`). These two definitions diverge when there are latency tail effects or
batching inefficiencies. The unified v2 script uses `1000 / mean_tpot_ms` consistently.

---

## 9. Pareto Chart

The main output is a **6-panel PNG** (`pareto/PARETO.png`) with the following panels:

| Panel | X axis | Y axis | Purpose |
|-------|--------|--------|---------|
| GPU Throughput vs Concurrency | log₂(concurrency) | tok/s/GPU | Shows saturation point |
| Per-User Speed vs Concurrency | log₂(concurrency) | tok/s/user | Shows user experience degradation |
| **Pareto Frontier** | tok/s/user (↑ better) | tok/s/GPU (↑ better) | The main trade-off curve, colored by concurrency |
| TTFT vs Concurrency | log₂(concurrency) | ms | Prefill queue delay |
| E2E Latency vs Concurrency | log₂(concurrency) | s (mean + p99) | Full request time |
| ITL & TPOT vs Concurrency | log₂(concurrency) | ms (mean + p99) | Decode step latency |

The Pareto frontier plot uses a plasma colormap (dark=low concurrency, bright=high
concurrency). Points in the upper-right corner are preferred: high GPU utilization AND
fast per-user generation.

### Comparison chart (`compare.sh`)

`compare.sh` overlays both engines' frontiers and produces two additional charts:

- **`COMPARE.png`**: Both Pareto frontiers on one axes, SGLang (blue) vs vLLM (red),
  each point labeled with its concurrency level.

- **`COMPARE_delta.png`**: Two diverging bar charts showing `SGLang − vLLM` delta in
  `tok/s/GPU` and `tok/s/user` at each matched concurrency level. Blue bars = SGLang
  leads; red bars = vLLM leads. This makes the per-level gap immediately readable.

---

## 10. Nsight Profiling Design

`nsight_profile.sh` profiles the **decode-only GPU execution** at two representative
concurrency points: `c=2` (memory-bandwidth bound, small batch) and `c=32` (compute-bound,
large batch).

### Why decode-only?

Capturing prefill + decode together dilutes the decode kernel percentages with large
prefill GEMMs (`[T=ISL, d_model]` vs decode's `[B=C, d_model]`). At c=32, prefill
accounts for roughly 20–30% of the window. Excluding prefill gives a cleaner view of the
steady-state decode behaviour that dominates throughput at scale.

### Decode-only capture technique

```
t=0   background benchmark launched (BENCH_OSL = N_DECODE_STEPS × 10 tokens)
t=3s  nsys START  (all C requests have finished prefill; decode is in progress)
t=8s  nsys STOP   (DECODE_CAPTURE_SEC derived from warmup TPOT)
      kill benchmark
      kill server
```

`BENCH_OSL = 20 × 10 = 200` ensures requests are still decoding when nsys starts and
when it stops. Only `~N_DECODE_STEPS = 20` steps are captured inside the window.

`DECODE_CAPTURE_SEC` is derived from the warmup TPOT so the window scales with the
engine's actual speed:

```bash
DECODE_CAPTURE_SEC = max(5, round(N_DECODE_STEPS × TPOT_ms / 1000 × 2))
```

### Three-phase warmup sequence

Each concurrency level runs three sequential phases before the nsys capture window opens:

| Phase | OSL | Purpose |
|-------|-----|---------|
| **1. Timing warmup** | `N_DECODE_STEPS=20` | Measures TPOT to derive `DECODE_CAPTURE_SEC`; pre-compiles CUDA graphs and Triton kernels for kv_len = INPUT_LEN+1 … INPUT_LEN+20 |
| **2. Triton burn-in** | `BENCH_OSL=200` | Extends Triton JIT kernel coverage to kv_len = INPUT_LEN+1 … INPUT_LEN+200; eliminates the single-step 2480ms JIT spike at high concurrency |
| **3. Profiled benchmark** | `BENCH_OSL=200` | The actual nsys capture — all kernels pre-compiled, no cold JIT misses |

**Why the burn-in is necessary**: sglang's Triton attention backend JIT-compiles kernels
for each new `(batch_size, kv_len)` shape bucket. The timing warmup (20 tokens) only
covers kv_len up to 1020. The profiled run (200 tokens) extends to kv_len 1200. Without
the burn-in, the first decode step past kv_len=1020 triggers a ~2.5s JIT compilation
that appears as a blue "Kernels" block in the nsys GUI and inflates Max ITL to 2480ms
(while Median ITL is 16ms). The burn-in pre-compiles all those shapes before the window.

### `nsys launch` + `nsys start` + `nsys stop`

Rather than `nsys profile --delay --duration` (wall-clock timer), the script uses the
session-control API:

```bash
# Inject profiling libraries into server at launch; do NOT record yet
nsys launch --trace=cuda,nvtx --session-new="$SESSION" "${SERVER_CMD[@]}" &

# ...  server starts, warmup runs, burn-in runs  (zero trace data)

# Start recording exactly when the profiled benchmark begins
nsys start --output "$PROFILE_OUT" --force-overwrite true --session="$SESSION"
sleep "$DECODE_CAPTURE_SEC"
nsys stop --session="$SESSION"
```

This guarantees the `.nsys-rep` file contains **only** steady-state decode kernels with
no startup noise, warmup transients, or JIT spikes.

---

## 11. Fairness Invariants and What Breaks Them

### What the benchmark controls

| Invariant | Mechanism | Why it matters |
|-----------|-----------|----------------|
| Identical ISL per request | Sliding-window tokenized dataset | Different ISL → different prefill cost; TPOT and throughput become ISL-dependent |
| Identical OSL per request | `--sharegpt-output-len + --ignore-eos / --disable-ignore-eos` | Variable OSL → TPOT and throughput depend on accidental EOS distribution |
| No prefix-cache hits | Unique sliding-window prompts + cache flush between steps | sglang's RadixAttention would serve later requests from cache, making prefill appear free |
| Same GPU memory budget | `--gpu-memory-utilization 0.90` / `--mem-fraction-static 0.90` | More KV cache memory = larger maximum batch size and higher throughput at high concurrency |
| Same max concurrent requests | `--max-num-seqs 256` / `--max-running-requests 256` | This is the hard cap on decode batch size; mismatching it makes the concurrency curves incomparable |
| Same decode batch size throughout | `N_PROMPTS = C` (profiling) / `N_PROMPTS = 3C` (Pareto) | If N < C, some concurrency slots are idle for part of the run, underutilizing the GPU |
| Cold KV cache at each step | `--flush-cache` / `/reset_prefix_cache` before each step | A warm cache inflates throughput for the engine that happens to have better caching |

### What would break fairness

| Violation | Effect |
|-----------|--------|
| Fixed N regardless of C | At c=1, measures sequential throughput; at c=128 measures concurrent throughput; the two are not on the same Pareto curve |
| Warmup at wrong batch size (e.g. c=1 for all steps) | CUDA graphs compiled for batch=1; first profiled decode step at batch=32 triggers a ~2.5s JIT recompilation noise event |
| Warmup with short OSL, profiled run with long OSL | Triton JIT not pre-compiled for kv_len beyond warmup_OSL; single-step 2480ms spike in profiled trace |
| Omitting `--ignore-eos` / `--disable-ignore-eos` | Requests terminate at EOS; output lengths vary; TPOT becomes output-length-weighted average |
| Omitting cache flush | sglang's prefix cache hits inflate its throughput relative to vLLM's uncached baseline |
| Different `--max-num-seqs` / `--max-running-requests` | One engine can schedule more requests simultaneously, gaining throughput not from better kernels but from a larger scheduling window |
| vLLM's `vllm_flash_attn` (CUDA 13) vs TRITON_ATTN | If one engine crashes or silently uses a slower fallback, the comparison is meaningless |

---

## 12. Version History: v1 (vllm-bench) vs v2 (unified)

### v1 — `benchmark/vllm-bench/run_pareto_benchmark.sh`

The original single-engine script, vLLM only. Notable differences from v2:

- Uses the **raw ShareGPT V4.3 dataset** (`ShareGPT_V4.3_unfiltered_cleaned_split.json`),
  not a controlled fixed-ISL dataset. Input lengths are whatever ShareGPT provides, which
  averages ~550 tokens but varies widely. ISL is a label only, not enforced.
- No sglang support; no unified summary format.
- KV cache flush is present (`/reset_prefix_cache`) but noted in the code as a potential
  explanation for why vLLM had previously appeared faster than sglang (sglang's cache was
  not being flushed in earlier versions of the comparison).
- Prompts-per-step formula is `ROUNDS × C` (same as v2's `PROMPTS_MULT × C`).
- Pareto chart is a single frontier plot (no 6-panel layout, no engine comparison).

### v2 — `benchmark/run_pareto_benchmark.sh`

The unified script supporting both engines with identical workloads:

- **Fixed-ISL sliding-window dataset**: `ISL=1000` tokens exact, all prompts unique.
- **Shared server config normalization**: GPU memory, max sequences, attention backend all
  matched between engines.
- **Unified `summary.jsonl`**: common schema; metrics normalized for vLLM (ITL from TPOT,
  E2E from TTFT + OSL×TPOT).
- **6-panel Pareto chart** + `compare.sh` overlay + delta charts.
- `_fix_nvidia_uvm` and `_pick_free_gpu` shared preflight routines.

### Nsight profiling — `benchmark/nsight_profile.sh`

Separate from the Pareto sweep; records GPU execution at two concurrency points.
Key improvements over naive profiling:

- **Three-phase warmup** (timing warmup → Triton burn-in → profiled run) to eliminate JIT
  spikes inside the capture window.
- **Benchmark startup poll** instead of fixed sleep — waits for `vllm bench serve` to
  finish its 10–15s initialization before opening the nsys window.
- **Minimum free GPU memory guard** to fail fast if another process already occupies the
  selected GPU.
- **`nsys launch + start + stop`** for exact capture-window control with no wall-clock
  guessing.
