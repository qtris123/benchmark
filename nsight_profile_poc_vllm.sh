#!/usr/bin/env bash
# nsight_profile_poc_vllm.sh — vLLM counterpart to nsight_profile_poc.sh.
# Profiles vLLM DECODE the "cudaProfilerApi" way, with in-graph kernel visibility,
# so the reports line up step-for-step with the SGLang PoC.
#
# How vLLM mirrors the SGLang mechanism (verified against vllm 0.22.0 source):
#   • --cuda-graph-trace=node        → records individual kernels INSIDE CUDA
#                                      graphs (same nsys flag as the sglang PoC).
#   • --trace-fork-before-exec=true  → nsys follows vLLM into its worker subprocess
#     + VLLM_WORKER_MULTIPROC_METHOD=spawn   (where the GPU work + cudaProfilerStart
#                                      actually happen). REQUIRED for vLLM.
#   • --profiler-config '{"profiler":"cuda","max_iterations":N}'
#       - POST /start_profile  → torch.cuda.profiler.start()  == cudaProfilerStart
#         (arms nsys --capture-range=cudaProfilerApi)
#       - after N worker (decode) steps → torch.cuda.profiler.stop() == cudaProfilerStop
#         (auto-stop). With --capture-range-end=stop-shutdown nsys finalizes the
#         report AND shuts the server down, so `wait $NSYS_PID` returns exactly when
#         the .nsys-rep is complete.  →  max_iterations is vLLM's `num_steps`.
#   • No warmup burn-in: we wait until the server is in steady-state decode
#     (prompt tput 0, generation tput > 0) before /start_profile, so CUDA graphs
#     are captured and Triton kernels JIT-compiled.
#
# Fairness knobs are kept identical to run_pareto_benchmark.sh:
#   TRITON_ATTN, tp=1, max-num-seqs 256, max-num-batched-tokens 8192,
#   gpu-mem 0.90, dtype auto; prefix caching ENABLED but neutralized via the
#   `random` dataset (unique prompts) + /reset_prefix_cache; --ignore-eos; greedy.
#
# Scope: vLLM only, PoC. GPU-selection / fairness extras intentionally omitted.
#
# Usage:
#   bash nsight_profile_poc_vllm.sh                 # batch 32
#   BATCH_SIZES="2 32" bash nsight_profile_poc_vllm.sh
#   CUDA_VISIBLE_DEVICES=1 bash nsight_profile_poc_vllm.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Config (override via env) ────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
IFS=' ' read -r -a BATCH_SIZES <<< "${BATCH_SIZES:- 32}"  # decode batch = concurrency
INPUT_LEN="${INPUT_LEN:-10000}"      # prefill tokens per request (bump to 1000 to match bench)
# OUTPUT_LEN must keep the fixed batch decoding well past vLLM's ~10s throughput-
# log granularity + steady-state detection + the NUM_STEPS capture window. A small
# OSL (e.g. 1024) lets a tiny batch drain in ~12s — faster than we can detect it —
# so the profiler arms after the batch is already idle and never reaches max_iters.
OUTPUT_LEN="${OUTPUT_LEN:-8192}"
NUM_STEPS="${NUM_STEPS:-10}"       # decode steps to capture (== profiler max_iterations)
CAPTURE_TIMEOUT="${CAPTURE_TIMEOUT:-120}"  # hard cap on the nsys-finalize wait (s)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
OUT_DIR="${OUT_DIR:-$ROOT/nsight-profiles-poc}"

VLLM="$ROOT/vllm-env/bin/vllm"
NSYS="${NSYS:-$(command -v nsys || echo /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys)}"

[[ -x "$VLLM" ]] || { echo "ERROR: vllm CLI not found at $VLLM"; exit 1; }
[[ -x "$NSYS" ]] || { echo "ERROR: nsys not found ($NSYS)"; exit 1; }
mkdir -p "$OUT_DIR"

# CUDA libs for the vllm venv (matches the benchmark); harmless if absent.
[[ -f "$ROOT/vllm-bench/env.sh" ]] && source "$ROOT/vllm-bench/env.sh"

# nsys must follow vLLM into its spawned worker process.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# HuggingFace auth (Llama is gated).
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== nsys     : $($NSYS --version 2>&1 | head -1) ==="
echo "=== model    : $MODEL ==="
echo "=== batches  : ${BATCH_SIZES[*]}  (ISL=$INPUT_LEN OSL=$OUTPUT_LEN steps=$NUM_STEPS) ==="
echo "=== gpu      : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset (vllm picks cuda:0)} ==="
echo "=== output   : $OUT_DIR ==="

# ── State shared with the cleanup trap ───────────────────────────────────────
NSYS_PID=""
BENCH_PID=""

# Reap any vLLM server / nsys launcher bound to our PORT, then wait until the OS
# actually releases the port. REQUIRED because --capture-range-end=stop-shutdown
# lets the foreground `nsys profile` exit while the vllm serve process (and its
# spawned EngineCore worker) survive as ORPHANs — not children of NSYS_PID — so
# we match them by command line instead.
_free_port() {
  pkill -9 -f "vllm serve .*--port ${PORT} " 2>/dev/null || true
  pkill -9 -f "EngineCore"                    2>/dev/null || true
  pkill -9 -f "nsys-launcher"                 2>/dev/null || true
  local t=0
  while ss -tlnp 2>/dev/null | grep -q ":${PORT}[[:space:]]"; do
    (( t >= 30 )) && return 1
    sleep 1; (( t += 1 ))
  done
  return 0
}

_cleanup() {
  [[ -n "$BENCH_PID" ]] && kill "$BENCH_PID" 2>/dev/null || true
  if [[ -n "$NSYS_PID" ]] && kill -0 "$NSYS_PID" 2>/dev/null; then
    local pgid
    pgid="$(ps -o pgid= -p "$NSYS_PID" 2>/dev/null | tr -d ' ')" || true
    kill "$NSYS_PID" 2>/dev/null || true
    sleep 3
    [[ -n "$pgid" ]] && kill -9 "-$pgid" 2>/dev/null || true
  fi
  _free_port || true
  NSYS_PID=""; BENCH_PID=""
}
trap _cleanup EXIT INT TERM

# ── Profile one decode batch size ────────────────────────────────────────────
profile_batch() {
  local B="$1"
  local OUT="$OUT_DIR/vllm_decode_b${B}_i${INPUT_LEN}"
  local SERVER_LOG="$OUT_DIR/vllm_b${B}_i${INPUT_LEN}_server.log"
  local BENCH_LOG="$OUT_DIR/vllm_b${B}_i${INPUT_LEN}_bench.log"
  rm -f "${OUT}.nsys-rep"

  # A previous batch's stop-shutdown can leave an orphaned server on $PORT.
  if ss -tlnp 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo ">>> port $PORT busy — reaping any stale vllm server ..."
    _free_port || { echo "ERROR: port $PORT still in use after reap — kill it manually."; exit 1; }
  fi

  echo ""
  echo "─────────────────────────────────────────────────────────────"
  echo "  PROFILE  decode batch = $B   →  ${OUT}.nsys-rep"
  echo "─────────────────────────────────────────────────────────────"

  # ── Launch vLLM server UNDER nsys, recording held until cudaProfilerStart ───
  echo ">>> Launching server under nsys (capture armed, not recording yet) ..."
  "$NSYS" profile \
    --trace=cuda,nvtx \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    -s none --cpuctxsw=none -b none \
    --force-overwrite=true \
    -o "$OUT" \
    "$VLLM" serve "$MODEL" \
      --dtype auto \
      --attention-backend TRITON_ATTN \
      --tensor-parallel-size 1 \
      --max-num-seqs 256 \
      --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
      --gpu-memory-utilization 0.90 \
      --host "$HOST" \
      --port "$PORT" \
      --profiler-config "{\"profiler\":\"cuda\",\"max_iterations\":$NUM_STEPS}" \
    > "$SERVER_LOG" 2>&1 &
  NSYS_PID=$!
  echo "    nsys PID=$NSYS_PID  server log=$SERVER_LOG"

  # ── Wait for server health ───────────────────────────────────────────────
  echo ">>> Waiting for http://$HOST:$PORT/health ..."
  local waited=0
  until curl -sf "http://$HOST:$PORT/health" &>/dev/null; do
    kill -0 "$NSYS_PID" 2>/dev/null || { echo "ERROR: server died:"; tail -30 "$SERVER_LOG"; exit 1; }
    (( waited >= 300 )) && { echo "ERROR: server health timeout."; tail -30 "$SERVER_LOG"; exit 1; }
    sleep 5; (( waited += 5 ))
  done
  echo "    Server ready (${waited}s)."

  # Best-effort prefix-cache reset (may 404 depending on build). The real
  # neutralizer is the `random` dataset's unique prompts, just like the benchmark.
  curl -sf -X POST "http://$HOST:$PORT/reset_prefix_cache" &>/dev/null || true

  # ── Drive a fixed decode batch through the server ────────────────────────
  # num-prompts == max-concurrency == B with request-rate inf  → all B start
  # together, none refilled  → decode batch is a constant B (mirrors sglang's
  # bench_one_batch_server). `random` dataset = unique prompts; --ignore-eos
  # keeps every request decoding the full OUTPUT_LEN.
  echo ">>> Starting vllm bench serve (batch=$B, in=$INPUT_LEN, out=$OUTPUT_LEN) ..."
  "$VLLM" bench serve \
    --model "$MODEL" \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$B" \
    --max-concurrency "$B" \
    --request-rate inf \
    --ignore-eos \
    --host "$HOST" \
    --port "$PORT" \
    > "$BENCH_LOG" 2>&1 &
  BENCH_PID=$!

  # ── Wait until the server is in STEADY-STATE decode ──────────────────────
  # vLLM logs (every ~10s): "Avg prompt throughput: 0.0 tokens/s, Avg generation
  # throughput: X tokens/s, Running: N reqs ...". We require prompt==0 (prefill
  # done) AND Running == B (all requests still decoding) so we fire /start_profile
  # while decode is genuinely in flight — never on the drain tail where Running
  # has already dropped to 0 (which leaves the profiler with no steps to count).
  echo ">>> Waiting for steady-state decode (prompt tput 0, Running == $B reqs) ..."
  local dwaited=0 ready=0
  while (( dwaited < 180 )); do
    if grep -qE "Avg prompt throughput: 0\.0 tokens/s, Avg generation throughput: [0-9.]+ tokens/s, Running: ${B} reqs" "$SERVER_LOG" 2>/dev/null; then
      ready=1; break
    fi
    kill -0 "$NSYS_PID" 2>/dev/null || { echo "ERROR: server exited early."; tail -30 "$SERVER_LOG"; exit 1; }
    # The bench client MUST stay alive: profiling only progresses while decode
    # steps execute, i.e. while a request is in flight. Abort if it died.
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
      echo "ERROR: bench client exited before decode started — not profiling."
      echo "       Last lines of $BENCH_LOG:"; tail -15 "$BENCH_LOG"
      exit 1
    fi
    sleep 1; (( dwaited += 1 ))
  done
  (( ready == 1 )) || { echo "ERROR: decode not detected within ${dwaited}s."; tail -30 "$SERVER_LOG"; exit 1; }
  echo "    Steady-state decode detected after ${dwaited}s."
  sleep 1  # let throughput settle one more beat

  # ── Fire the capture: cudaProfilerStart, run NUM_STEPS, auto cudaProfilerStop ─
  echo ">>> POST /start_profile  (cuda backend, max_iterations=$NUM_STEPS) ..."
  curl -s -X POST "http://$HOST:$PORT/start_profile" || true
  echo ""

  # ── nsys auto-stops + shuts down at cudaProfilerStop. Wait WITH A TIMEOUT ──
  echo ">>> Capturing ${NUM_STEPS} decode steps — waiting for nsys to finalize (max ${CAPTURE_TIMEOUT}s) ..."
  local cwaited=0
  while kill -0 "$NSYS_PID" 2>/dev/null; do
    # If the bench drained before nsys finalized, the batch went idle and the
    # profiler will never reach max_iterations — fail fast instead of waiting out
    # the full timeout.
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
      echo "ERROR: bench exited before ${NUM_STEPS} decode steps were captured."
      echo "       The fixed batch drained too soon — raise OUTPUT_LEN."
      _cleanup; exit 1
    fi
    if (( cwaited >= CAPTURE_TIMEOUT )); then
      echo "ERROR: nsys did not finalize within ${CAPTURE_TIMEOUT}s — capture stalled."
      echo "       bench still alive but decode didn't reach ${NUM_STEPS} steps."
      _cleanup; exit 1
    fi
    sleep 1; (( cwaited += 1 ))
  done
  wait "$NSYS_PID" 2>/dev/null || true
  NSYS_PID=""
  kill "$BENCH_PID" 2>/dev/null || true
  wait "$BENCH_PID" 2>/dev/null || true
  BENCH_PID=""

  if [[ -f "${OUT}.nsys-rep" ]]; then
    echo "    DONE: ${OUT}.nsys-rep  ($(du -h "${OUT}.nsys-rep" | cut -f1))"
  else
    echo "    ERROR: ${OUT}.nsys-rep was not written."; tail -20 "$SERVER_LOG"; exit 1
  fi

  echo ">>> Reaping server and waiting for port $PORT to free ..."
  _free_port || { echo "ERROR: port $PORT did not free after capture."; exit 1; }
}

for B in "${BATCH_SIZES[@]}"; do
  profile_batch "$B"
done

trap - EXIT INT TERM
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " PoC complete. Reports:"
for B in "${BATCH_SIZES[@]}"; do
  f="$OUT_DIR/vllm_decode_b${B}_i${INPUT_LEN}.nsys-rep"
  [[ -f "$f" ]] && echo "   $f"
done
echo ""
echo " Inspect in-graph kernels (visible thanks to --cuda-graph-trace=node):"
echo "   nsys stats --report cuda_gpu_kern_sum $OUT_DIR/vllm_decode_b32.nsys-rep"
echo "═══════════════════════════════════════════════════════════════"
