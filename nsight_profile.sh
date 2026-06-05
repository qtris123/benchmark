#!/usr/bin/env bash
# nsight_profile.sh — Profile SGLang or vLLM at c=2 and c=32 with Nsight Systems
#
# Usage:
#   ENGINE=sglang bash nsight_profile.sh
#   ENGINE=vllm   bash nsight_profile.sh
#
# Outputs (in ./nsight-profiles/):
#   <engine>_c2.nsys-rep    — profile captured during c=2 benchmark
#   <engine>_c32.nsys-rep   — profile captured during c=32 benchmark
#
# How capture works (launch → start → stop):
#   1. nsys LAUNCH injects profiling libraries (LD_PRELOAD) into the server and
#      all its child processes (e.g. sglang::scheduler, vLLM EngineCore), but
#      does NOT start recording yet.
#   2. Server starts up and warmup runs — zero trace data collected.
#   3. nsys START: recording begins exactly when the benchmark starts.
#   4. Benchmark runs for however long it takes (no time cap).
#   5. nsys STOP: recording ends exactly when the benchmark finishes;
#      report is written immediately, then the server is killed.
#
#   This guarantees the trace contains ONLY steady-state inference — no startup
#   noise, no idle padding — and never truncates a slow benchmark.
#
# To view results: copy .nsys-rep to your local machine, open in Nsight Systems GUI.
#   Download GUI: https://developer.nvidia.com/nsight-systems
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── nsys binary ──────────────────────────────────────────────────────────────
# Prefer nsight-systems 2024.x (CUDA 12 native); fall back to the legacy 2019 binary.
NSYS=""
for candidate in \
    "/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys" \
    "/usr/local/bin/nsys" \
    "/usr/lib/nsight-systems/Target-x86_64/x86_64/nsys"; do
  if [[ -x "$candidate" ]]; then
    NSYS="$candidate"
    break
  fi
done
if [[ -z "$NSYS" ]]; then
  echo "ERROR: nsys not found. Install nsight-systems-2024.6.2 or set NSYS env var."
  exit 1
fi
echo "=== nsys: $($NSYS --version 2>&1 | head -1) ==="

# ── Config ───────────────────────────────────────────────────────────────────
ENGINE="${ENGINE:-sglang}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
# Override via env: CONCURRENCIES="2" or CONCURRENCIES="2 32"
IFS=' ' read -r -a PROFILE_CONCURRENCIES <<< "${CONCURRENCIES:-2 32}"
# Maximum concurrency across all profiling steps — used to cap CUDA graph
# compilation (sglang --cuda-graph-max-bs) so only graphs actually needed are
# compiled at startup.
MAX_C=$(printf '%s\n' "${PROFILE_CONCURRENCIES[@]}" | sort -n | tail -1)

# --- Fairness invariant ---
# For each concurrency level C, exactly N_PROMPTS = C requests are sent at
# concurrency C.  This guarantees:
#   • decode batch size = C throughout (no partial batches at any C)
#   • decode iterations = N_DECODE_STEPS (identical across all C values)
#   • warmup runs C requests at concurrency C to pre-compile CUDA graphs
#     and triton attention kernels for batch=C (short trajectory, kv_len up
#     to INPUT_LEN + N_DECODE_STEPS)
#   • burn-in extends coverage to kv_len up to INPUT_LEN + BENCH_OSL,
#     eliminating any triton JIT spike during the profiled capture window
# --- End fairness invariant ---

# Number of decode steps to capture in the nsys trace.
# The actual benchmark runs with BENCH_OSL = N_DECODE_STEPS × 10 so that
# the server is still decoding when nsys starts recording.  Recording stops
# after ≈ 2 × N_DECODE_STEPS × TPOT seconds (derived from warmup).
# The resulting .nsys-rep contains ONLY decode kernels — no prefill.
#
#   20 → clean, fast, recommended (steady-state CUDA graph replay clearly visible)
#   50 → larger file, no additional insight beyond step ~5
N_DECODE_STEPS="${N_DECODE_STEPS:-20}"

# Input tokens per request.  Enforced exactly via the tokenized dataset.
INPUT_LEN="${INPUT_LEN:-1000}"

PROFILE_DIR="$ROOT/nsight-profiles"
mkdir -p "$PROFILE_DIR"

# ── Dataset preparation ───────────────────────────────────────────────────────
# Builds a ShareGPT-format JSON where each entry has a UNIQUE prompt drawn from
# a sliding window over random_text.txt (token offset i → i+INPUT_LEN).
#
# Unique prompts are essential so sglang's radix-tree prefix cache cannot serve
# request k from request k-1's cached KV entries.  Both engines therefore pay
# the full INPUT_LEN-token prefill cost for every request, making the profiled
# prefill kernels comparable.
#
# Output length is enforced at request time via --sharegpt-output-len +
# --ignore-eos / --disable-ignore-eos; the "gpt" placeholder value is unused.
DATASET_FILE="$PROFILE_DIR/dataset_isl${INPUT_LEN}.json"

_prepare_dataset() {
  [[ -f "$DATASET_FILE" ]] && { echo "    Reusing dataset: $DATASET_FILE"; return 0; }
  echo "    Building dataset — sliding-window prompts from random_text.txt ..."

  # Need enough unique prompts to cover warmup + profiled run at each C.
  # Worst case: 2 × max(PROFILE_CONCURRENCIES) prompts (warmup + profile).
  local max_c="${PROFILE_CONCURRENCIES[-1]}"
  local n=$(( max_c * 2 + 16 ))

  python3 - "$MODEL" "$ROOT/random_text.txt" "$DATASET_FILE" "$INPUT_LEN" "$n" <<'PYEOF'
import sys, json
from transformers import AutoTokenizer

model, text_path, out_path, isl, n = \
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])

tok     = AutoTokenizer.from_pretrained(model, local_files_only=True)
all_ids = tok.encode(open(text_path).read(), add_special_tokens=False)

# Tile if the source text is shorter than the sliding window needs
while len(all_ids) < isl + n:
    all_ids = all_ids + all_ids

data = [
    {"conversations": [
        {"from": "human", "value": tok.decode(all_ids[i : i + isl], skip_special_tokens=True)},
        {"from": "gpt",   "value": "x"},  # placeholder; overridden by --sharegpt-output-len
    ]}
    for i in range(n)
]
json.dump(data, open(out_path, "w"))
print(f"    Dataset: {n} unique prompts × {isl} tokens  →  {out_path}")
PYEOF
}

# ── Engine-specific settings ─────────────────────────────────────────────────
if [[ "$ENGINE" == "sglang" ]]; then
  source "$ROOT/sglang-env/bin/activate"
  source "$ROOT/sglang-bench/env.sh"
  SERVER_PORT=30000
  SERVER_HOST="127.0.0.1"
  SERVER_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL"
    --tp-size 1
    --port "$SERVER_PORT"
    --host "$SERVER_HOST"
    --mem-fraction-static 0.90
    --max-running-requests 256
    --dtype auto
    --attention-backend triton
    --cuda-graph-max-bs "$MAX_C"
  )
  run_bench() {
    local C=$1 OUT_DIR=$2 N=$3
    python3 -m sglang.bench_serving \
      --backend sglang \
      --host "$SERVER_HOST" \
      --port "$SERVER_PORT" \
      --model "$MODEL" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_FILE" \
      --sharegpt-output-len "$N_DECODE_STEPS" \
      --num-prompts "$N" \
      --request-rate inf \
      --max-concurrency "$C" \
      --output-file "$OUT_DIR/bench.jsonl" \
      --flush-cache \
      --disable-ignore-eos \
      2>&1 | tee "$OUT_DIR/bench.log"
  }

elif [[ "$ENGINE" == "vllm" ]]; then
  source "$ROOT/.venv/bin/activate"
  [[ -f "$ROOT/benchmarks/env.sh" ]] && source "$ROOT/benchmarks/env.sh"
  SERVER_PORT=8000
  SERVER_HOST="localhost"
  SERVER_CMD=(
    vllm serve "$MODEL"
    --dtype auto
    --attention-backend TRITON_ATTN
    --tensor-parallel-size 1
    --max-num-seqs 256
    --max-num-batched-tokens 8192
    --gpu-memory-utilization 0.90
    --port "$SERVER_PORT"
  )
  run_bench() {
    local C=$1 OUT_DIR=$2 N=$3
    vllm bench serve \
      --model "$MODEL" \
      --backend vllm \
      --endpoint /v1/completions \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_FILE" \
      --sharegpt-output-len "$N_DECODE_STEPS" \
      --num-prompts "$N" \
      --max-concurrency "$C" \
      --host "$SERVER_HOST" \
      --port "$SERVER_PORT" \
      --ignore-eos \
      --save-result \
      --result-dir "$OUT_DIR" \
      2>&1 | tee "$OUT_DIR/bench.log"
  }

else
  echo "ERROR: ENGINE must be 'sglang' or 'vllm' (got: '$ENGINE')"
  exit 1
fi

# ── HuggingFace auth ─────────────────────────────────────────────────────────
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── GPU selection ────────────────────────────────────────────────────────────
# Auto-pick the GPU with the most free memory, unless CUDA_VISIBLE_DEVICES is
# already set explicitly by the caller.
_pick_free_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "=== GPU: using caller-supplied CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
    return
  fi

  # Score = (100 - utilization%) * 10000 + free_MiB
  # → lowest utilization wins; free memory breaks ties.
  local best_idx=0 best_score=-1 best_util=0 best_free=0
  while IFS=', ' read -r idx util free; do
    idx="${idx// /}"; util="${util// /}"; free="${free// /}"
    local score=$(( (100 - util) * 10000 + free ))
    if (( score > best_score )); then
      best_score=$score; best_idx=$idx; best_util=$util; best_free=$free
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
             --format=csv,noheader,nounits 2>/dev/null)

  export CUDA_VISIBLE_DEVICES="$best_idx"
  echo "=== GPU: auto-selected GPU $best_idx  (util ${best_util}%,  ${best_free} MiB free) ==="
}

# ── nvidia-uvm preflight ──────────────────────────────────────────────────────
_fix_nvidia_uvm() {
  local reg maj
  reg=$(awk '/nvidia-uvm/{print $1}' /proc/devices 2>/dev/null)
  [[ -z "$reg" ]] && return
  maj=$(printf '%d' "0x$(stat -c '%t' /dev/nvidia-uvm 2>/dev/null)" 2>/dev/null || echo "")
  if [[ "$maj" != "$reg" ]]; then
    echo "WARNING: /dev/nvidia-uvm major mismatch — auto-fixing..."
    sudo rm -f /dev/nvidia-uvm /dev/nvidia-uvm-tools \
      && sudo mknod /dev/nvidia-uvm c "$reg" 0 \
      && sudo mknod /dev/nvidia-uvm-tools c "$reg" 1 \
      && sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools
  fi
}
_fix_nvidia_uvm
_pick_free_gpu

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " ENGINE    : $ENGINE"
echo " GPU            : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo " MODEL          : $MODEL"
echo " CONCURRENCIES  : ${PROFILE_CONCURRENCIES[*]}"
echo " INPUT LEN      : $INPUT_LEN tokens  (exact, unique prompt per request)"
echo " DECODE STEPS   : $N_DECODE_STEPS tokens per request"
echo " PROMPTS/STEP   : = C  (one full decode batch per profiled window)"
echo " WARMUP         : C requests @ concurrency=C  (pre-compiles CUDA graphs)"
echo " CAPTURE        : benchmark start → benchmark end  (exact, no time cap)"
echo " OUTPUT         : $PROFILE_DIR"
echo "═══════════════════════════════════════════════════════════"

# ── Token-target validation ───────────────────────────────────────────────────
# Reads bench.log after a run and checks that actual avg input/output tokens
# match INPUT_LEN / N_DECODE_STEPS within tolerance.
#
#   Input  tolerance : 5%  (tokenisation rounding across unique prompts)
#   Output tolerance : 2%  (--ignore-eos/--disable-ignore-eos should give exact)
#
# Exits 1 (aborting the script) if deviation exceeds the tolerance.
_validate_token_targets() {
  local bench_log=$1 label=$2

  local tot_in tot_out tot_req avg_in avg_out
  tot_in=$(grep  "Total input tokens"     "$bench_log" | grep -oP '[0-9]+' | tail -1)
  tot_out=$(grep "Total generated tokens" "$bench_log" | grep -oP '[0-9]+' | tail -1)
  tot_req=$(grep "Successful requests"    "$bench_log" | grep -oP '[0-9]+' | tail -1)

  if [[ -z "$tot_in" || -z "$tot_out" || -z "$tot_req" || "$tot_req" -eq 0 ]]; then
    echo "  [WARN] Could not parse token counts from $bench_log — skipping validation."
    return 0
  fi

  avg_in=$(( tot_in  / tot_req ))
  avg_out=$(( tot_out / tot_req ))

  local dev_in dev_out
  dev_in=$(( (avg_in  - INPUT_LEN)      * 100 / INPUT_LEN      ))
  dev_out=$(( (avg_out - N_DECODE_STEPS) * 100 / N_DECODE_STEPS ))
  (( dev_in  < 0 )) && dev_in=$(( -dev_in ))
  (( dev_out < 0 )) && dev_out=$(( -dev_out ))

  local status_in="PASS" status_out="PASS"
  (( dev_in  > 5 )) && status_in="WARN"
  (( dev_out > 2 )) && status_out="WARN"

  echo ""
  echo "  ┌─ Token-target check [$label] ────────────────────────────"
  printf "  │  Input  tokens : avg %d / target %d  (dev %d%%)  [%s]\n" \
    "$avg_in"  "$INPUT_LEN"      "$dev_in"  "$status_in"
  printf "  │  Output tokens : avg %d / target %d  (dev %d%%)  [%s]\n" \
    "$avg_out" "$N_DECODE_STEPS" "$dev_out" "$status_out"
  echo "  └──────────────────────────────────────────────────────────"

  if [[ "$status_in" == "WARN" || "$status_out" == "WARN" ]]; then
    echo ""
    echo "  ERROR: Token counts deviate from target beyond tolerance."
    echo "         Input  tolerance : 5%   (got ${dev_in}%)"
    echo "         Output tolerance : 2%   (got ${dev_out}%)"
    exit 1
  fi
}

# Prepare the shared prompt dataset once before starting any server
_prepare_dataset

# ── Profile each concurrency level in its own server instance ────────────────
for C in "${PROFILE_CONCURRENCIES[@]}"; do
  STEP_DIR="$PROFILE_DIR/${ENGINE}_c${C}_bench"
  PROFILE_OUT="$PROFILE_DIR/${ENGINE}_c${C}"
  SERVER_LOG="$PROFILE_DIR/${ENGINE}_c${C}_server.log"
  mkdir -p "$STEP_DIR"

  # ── Guard: abort if port is already occupied ─────────────────────────────
  if ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT}[[:space:]]"; then
    echo "ERROR: port $SERVER_PORT is already in use (stale server from a previous run?)."
    echo "       Kill it first:  kill \$(ss -tlnp | awk '/:${SERVER_PORT}/{gsub(/.*pid=/,\"\",\$0);gsub(/,.*/,\"\",\$0);print}')"
    exit 1
  fi

  # N_PROMPTS = C: one full decode batch worth of requests per profiled window.
  # Warmup also uses C requests at concurrency=C to pre-compile CUDA graphs for
  # the exact batch size that will appear in the profiled capture.
  N_PROMPTS=$C

  echo ""
  echo "─────────────────────────────────────────────────────────"
  echo "  Profiling  ENGINE=$ENGINE  concurrency=$C"
  echo "  N_PROMPTS : $N_PROMPTS  (= C → decode batch = C throughout)"
  echo "  Input     : $INPUT_LEN tokens / request  (unique per request)"
  echo "  Decode    : $N_DECODE_STEPS steps / request"
  echo "  Output    : ${PROFILE_OUT}.nsys-rep"
  echo "─────────────────────────────────────────────────────────"

  # ── Step 1: nsys launch — inject libs, but do NOT start recording yet ─────
  # LD_PRELOAD is inherited by ALL child processes (sglang::scheduler,
  # vLLM EngineCore, etc.) so they are all instrumented automatically.
  # Recording is held until we explicitly call `nsys start`.
  SESSION="${ENGINE}_c${C}_$$"
  echo ">>> Launching server under nsys (session=$SESSION, not recording yet) ..."
  "$NSYS" launch \
    --trace=cuda,nvtx \
    --session-new="$SESSION" \
    "${SERVER_CMD[@]}" \
    > "$SERVER_LOG" 2>&1 &

  LAUNCHER_PID=$!
  SERVER_START=$(date +%s)
  echo "    Launcher PID=$LAUNCHER_PID  log=$SERVER_LOG"

  # Kill launcher (and server subprocess) on exit or interrupt
  _cleanup_server() {
    echo ""
    echo "  Stopping server (PID $LAUNCHER_PID)..."
    kill "$LAUNCHER_PID" 2>/dev/null || true
    wait "$LAUNCHER_PID" 2>/dev/null || true
  }
  trap '_cleanup_server' EXIT

  # ── Step 2: Wait for server to be healthy ────────────────────────────────
  echo "    Waiting for http://$SERVER_HOST:$SERVER_PORT/health ..."
  TIMEOUT=300
  ELAPSED=0
  while true; do
    if curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" &>/dev/null; then
      SERVER_READY_AT=$(( $(date +%s) - SERVER_START ))
      echo "    Server ready (${SERVER_READY_AT}s)."
      break
    fi
    if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then
      echo "ERROR: server process died. Last 30 lines:"
      tail -30 "$SERVER_LOG"
      exit 1
    fi
    if (( ELAPSED >= TIMEOUT )); then
      echo "ERROR: server timed out. Last 30 lines:"
      tail -30 "$SERVER_LOG"
      exit 1
    fi
    sleep 5
    (( ELAPSED += 5 ))
  done

  # ── Step 3: Warmup — NOT recorded (nsys hasn't started yet) ──────────────
  # Run C requests at concurrency=C so the engine pre-compiles CUDA graphs
  # for decode batch=C before the profiled window opens.
  WARMUP_DIR="$PROFILE_DIR/${ENGINE}_c${C}_warmup"
  mkdir -p "$WARMUP_DIR"
  echo "    Running warmup: C=$C requests @ concurrency=$C (pre-compiles CUDA graphs for batch=$C) ..."
  run_bench "$C" "$WARMUP_DIR" "$N_PROMPTS" 2>/dev/null || true
  {
    echo "── Warmup token summary  ENGINE=$ENGINE  concurrency=$C ──"
    grep -E "Total input tokens|Total generated tokens|Successful requests" \
      "$WARMUP_DIR/bench.log" 2>/dev/null \
      | grep -v "^Namespace\|^benchmark_args" \
      | sed 's/^[[:space:]]*/  /'
  } | tee "$WARMUP_DIR/token_summary.log" || true
  _validate_token_targets "$WARMUP_DIR/bench.log" "warmup $ENGINE c=$C"

  # Derive decode capture window from warmup TPOT.
  # TPOT (ms/token) ≈ time per decode step for one request = time per GPU
  # decode iteration (since all C requests are batched together, each contributing
  # exactly 1 token per step).
  # Capture = 2 × N_DECODE_STEPS × TPOT, with a 5s minimum.
  _WARMUP_TPOT_MS=$(grep -i "mean tpot" "$WARMUP_DIR/bench.log" \
                    | grep -oP '[0-9]+\.[0-9]+' | head -1 || echo "")
  if [[ -n "$_WARMUP_TPOT_MS" ]]; then
    DECODE_CAPTURE_SEC=$(python3 -c \
      "print(max(5, round($N_DECODE_STEPS * $_WARMUP_TPOT_MS / 1000 * 2, 1)))")
  else
    DECODE_CAPTURE_SEC=10
  fi
  BENCH_OSL=$(( N_DECODE_STEPS * 10 ))   # large enough that decode outlasts the capture window

  echo "    Warmup done — CUDA graphs compiled for batch=$C, token targets OK."
  echo "    Warmup TPOT: ${_WARMUP_TPOT_MS:-unknown} ms  →  decode capture window: ${DECODE_CAPTURE_SEC}s"
  echo "    Benchmark OSL for profiled run: $BENCH_OSL  (= 10 × $N_DECODE_STEPS)"

  # ── Step 3b: Triton JIT burn-in ───────────────────────────────────────────
  # The warmup above ran N_DECODE_STEPS=20 output tokens per request, so the
  # triton attention kernel was JIT-compiled for kv_len = INPUT_LEN+1 …
  # INPUT_LEN+N_DECODE_STEPS.  The profiled benchmark runs BENCH_OSL=200
  # tokens, extending the kv_len range to INPUT_LEN+BENCH_OSL.  On the first
  # decode step that crosses into a new triton-compiled shape bucket, a single
  # ~2.5 s JIT spike occurs (observed: Max ITL 2480 ms at C=32 while median
  # is 16 ms).  Running one burn-in pass at BENCH_OSL tokens pre-compiles all
  # triton kernel shapes for kv_len INPUT_LEN+1 … INPUT_LEN+BENCH_OSL,
  # eliminating the spike from the nsys capture window.
  BURNIN_DIR="$PROFILE_DIR/${ENGINE}_c${C}_burnin"
  mkdir -p "$BURNIN_DIR"
  echo ""
  echo ">>> Triton burn-in (C=$C, OSL=$BENCH_OSL) — pre-compiling JIT kernels for kv_len $((INPUT_LEN+1))–$((INPUT_LEN+BENCH_OSL)) ..."
  (N_DECODE_STEPS=$BENCH_OSL; run_bench "$C" "$BURNIN_DIR" "$N_PROMPTS") 2>/dev/null || true
  echo "    Burn-in done — triton attention kernels cached, KV cache flushed."

  # ── Step 4: Launch benchmark in background with large OSL ─────────────────
  # The server will prefill all C requests then decode for BENCH_OSL steps.
  # nsys starts recording only after /metrics confirms decode has begun, so
  # the captured window contains ONLY the decode phase.
  echo ""
  echo ">>> Launching background benchmark (C=$C, N=$N_PROMPTS, OSL=$BENCH_OSL) ..."
  (
    N_DECODE_STEPS=$BENCH_OSL
    run_bench "$C" "$STEP_DIR" "$N_PROMPTS"
  ) &
  BENCH_PID=$!

  # ── Step 5: Wait for benchmark client, then detect decode start via /metrics ─
  # vllm bench serve has slow startup (tokenizer + dataset load: 10+ seconds).
  # We first wait for the bench log to confirm requests are in flight, then poll
  # the Prometheus /metrics endpoint until generation_tokens_total ticks above
  # baseline — the first increment means all C requests have finished prefill
  # and the first decode token has been produced.
  echo ">>> Waiting for benchmark client to start sending requests ..."
  _BENCH_WAIT_TIMEOUT=90
  _BENCH_WAITED=0
  while true; do
    # "Starting main benchmark run" appears in vllm bench serve just before
    # the first request is sent.  sglang.bench_serving prints similar output.
    if grep -qE "Starting main benchmark|main benchmark run|Warmup completed" \
        "$STEP_DIR/bench.log" 2>/dev/null; then
      echo "    Benchmark client ready after ${_BENCH_WAITED}s — requests in flight."
      break
    fi
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
      echo "WARNING: benchmark process exited before sending requests — profile window may be empty."
      break
    fi
    if (( _BENCH_WAITED >= _BENCH_WAIT_TIMEOUT )); then
      echo "WARNING: timed out waiting for benchmark to start (${_BENCH_WAIT_TIMEOUT}s) — proceeding anyway."
      break
    fi
    sleep 1
    (( _BENCH_WAITED += 1 ))
  done

  # Poll /metrics until the generated-token counter increments above baseline.
  # The first increment means prefill finished and the first decode step ran.
  # Supports both vllm (vllm:generation_tokens_total) and
  # sglang (sglang:new_generated_tokens_total).
  _get_gen_tokens() {
    curl -sf "http://${SERVER_HOST}:${SERVER_PORT}/metrics" 2>/dev/null \
      | awk '/^(vllm:generation_tokens_total|sglang:new_generated_tokens_total)(\{|[[:space:]])/ \
             && !/^#/ { val=$NF } END { if (val!="") printf "%d\n", val }'
  }
  _BASELINE=$(_get_gen_tokens)
  if [[ -z "$_BASELINE" ]]; then
    echo "    /metrics unavailable — decode detection skipped (proceeding immediately)."
  else
    echo ">>> Polling /metrics for first decode token (baseline=${_BASELINE} generated tokens) ..."
    _DECODE_POLLS=0
    _DECODE_MAX_POLLS=150  # 150 × 0.2 s = 30 s timeout
    while (( _DECODE_POLLS < _DECODE_MAX_POLLS )); do
      _NOW=$(_get_gen_tokens)
      if [[ -n "$_NOW" ]] && (( _NOW > _BASELINE )); then
        printf "    Decode confirmed after %.1fs (%d → %d tokens generated)\n" \
          "$(echo "$_DECODE_POLLS * 0.2" | bc)" "$_BASELINE" "$_NOW"
        break
      fi
      if ! kill -0 "$BENCH_PID" 2>/dev/null; then
        echo "WARNING: benchmark exited before decode detected — profile window may be empty."
        break
      fi
      sleep 0.2
      (( _DECODE_POLLS++ )) || true
    done
    if (( _DECODE_POLLS >= _DECODE_MAX_POLLS )); then
      echo "WARNING: decode not detected within 30s — proceeding anyway."
    fi
  fi

  echo ">>> nsys START — decode-only recording begins (session=$SESSION) ..."
  "$NSYS" start \
    --output "$PROFILE_OUT" \
    --force-overwrite true \
    --session="$SESSION"

  echo ">>> Recording decode for ${DECODE_CAPTURE_SEC}s (~$N_DECODE_STEPS decode steps) ..."
  sleep "$DECODE_CAPTURE_SEC"

  echo ">>> nsys STOP — decode capture ends (session=$SESSION) ..."
  "$NSYS" stop --session="$SESSION"

  # Kill the background bench process (it may still be decoding BENCH_OSL steps)
  kill "$BENCH_PID" 2>/dev/null || true
  wait "$BENCH_PID" 2>/dev/null || true

  # ── Step 5b: Log profile summary (from warmup, since bench was killed early) ──
  SUMMARY_LOG="$STEP_DIR/token_summary.log"
  {
    echo "═══════════════════════════════════════════════════════════"
    echo " Profile summary  ENGINE=$ENGINE  concurrency=$C"
    echo " Captured at    : $(date '+%Y-%m-%d %H:%M:%S')"
    echo " Mode           : DECODE ONLY (prefill detected via /metrics token counter)"
    echo " Configuration  : $N_PROMPTS requests × $INPUT_LEN in + $BENCH_OSL OSL"
    echo " Captured       : ~$N_DECODE_STEPS decode steps over ${DECODE_CAPTURE_SEC}s"
    echo " Decode batch   : $C  (all $N_PROMPTS requests decoding together)"
    echo "───────────────────────────────────────────────────────────"
    echo " Warmup stats (representative of profiled decode):"
    grep -E "Mean TPOT|Mean ITL|Output token throughput" \
      "$WARMUP_DIR/bench.log" 2>/dev/null \
      | grep -v "^Namespace\|^benchmark_args" \
      | sed 's/^[[:space:]]*/  /'
    echo "═══════════════════════════════════════════════════════════"
  } | tee "$SUMMARY_LOG"

  # ── Step 6: nsys stop already called above — just report the file ─────────

  # Report is written synchronously by `nsys stop` — no polling needed.
  if [[ -f "${PROFILE_OUT}.nsys-rep" ]]; then
    SIZE=$(du -sh "${PROFILE_OUT}.nsys-rep" | cut -f1)
    echo "    Written: ${PROFILE_OUT}.nsys-rep  (${SIZE})"
  else
    echo "    WARNING: ${PROFILE_OUT}.nsys-rep not found after nsys stop."
    ls -lh "$PROFILE_DIR/" 2>/dev/null
  fi

  # ── Step 7: Kill server ───────────────────────────────────────────────────
  echo ">>> Stopping server ..."
  kill "$LAUNCHER_PID" 2>/dev/null || true
  wait "$LAUNCHER_PID" 2>/dev/null || true
  trap - EXIT  # clear per-iteration trap

  echo "    Sleeping 10s before next step..."
  sleep 10
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " All profiles complete."
echo ""
shopt -s nullglob
_found=()
for _f in "$PROFILE_DIR/"*.nsys-rep "$PROFILE_DIR/"*.qdrep; do _found+=("$_f"); done
if (( ${#_found[@]} )); then ls -lh "${_found[@]}"; else echo " (no report files found)"; fi
shopt -u nullglob
echo ""
echo " NEXT STEPS — view in Nsight Systems GUI:"
echo "   1. Install on local machine (free):"
echo "      https://developer.nvidia.com/nsight-systems"
echo ""
echo "   2. Copy files to local machine:"
for C in "${PROFILE_CONCURRENCIES[@]}"; do
  # prefer .nsys-rep (nsys 2024+), fall back to .qdrep (legacy)
  if [[ -f "${PROFILE_DIR}/${ENGINE}_c${C}.nsys-rep" ]]; then
    echo "      scp $(hostname):${PROFILE_DIR}/${ENGINE}_c${C}.nsys-rep ~/Desktop/"
  else
    echo "      scp $(hostname):${PROFILE_DIR}/${ENGINE}_c${C}.qdrep ~/Desktop/"
  fi
done
echo ""
echo "   3. File → Open the .nsys-rep"
echo "      What to look for:"
echo "      • GPU row:   gaps at c=2 (idle), solid at c=32 (batched)"
echo "      • CUDA API:  cudaLaunchKernel frequency + duration"
echo "      • NVTX row:  sglang/vLLM batch boundaries (if annotated)"
echo "      • Trace starts at first inference request — no startup noise."
echo "═══════════════════════════════════════════════════════════"
