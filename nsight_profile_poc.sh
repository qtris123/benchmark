#!/usr/bin/env bash
# nsight_profile_poc.sh — Proof-of-concept: SGLang decode profiling the
# "cudaProfilerApi" way (the reference method), with in-graph kernel visibility.
#
# What's new vs nsight_profile.sh:
#   • --cuda-graph-trace=node   → records individual kernels INSIDE CUDA graphs
#                                 (no more opaque "CUDA Graph" blob).
#   • --capture-range=cudaProfilerApi + /start_profile
#                               → the SERVER starts/stops the capture at exact
#                                 decode-step boundaries (num_steps). No client
#                                 sleep timing, no prefill contamination.
#   • No warmup / no Triton burn-in.  We simply wait until the server is in
#     steady-state decode ("cuda graph: True" in the log) before profiling, so
#     CUDA graphs are already captured and Triton kernels already JIT-compiled.
#
# Scope: SGLang only (the cudaProfilerApi trigger is sglang-specific). GPU
# selection / fairness knobs are intentionally omitted — this is a PoC.
#
# Usage:
#   bash nsight_profile_poc.sh                 # batches 2 and 32
#   BATCH_SIZES="32" bash nsight_profile_poc.sh
#   CUDA_VISIBLE_DEVICES=1 bash nsight_profile_poc.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Config (override via env) ────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
IFS=' ' read -r -a BATCH_SIZES <<< "${BATCH_SIZES:-2 32}"  # decode batch = concurrency
INPUT_LEN="${INPUT_LEN:-10000}"     # prefill tokens per request
OUTPUT_LEN="${OUTPUT_LEN:-1024}"   # must outlast detection + the profiled window
NUM_STEPS="${NUM_STEPS:-11}"       # decode steps to capture (server-controlled)
CAPTURE_TIMEOUT="${CAPTURE_TIMEOUT:-120}"  # hard cap on the nsys-finalize wait (s)
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
OUT_DIR="${OUT_DIR:-$ROOT/nsight-profiles-poc}"

PY="$ROOT/sglang-env/bin/python3"
NSYS="${NSYS:-$(command -v nsys || echo /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys)}"

[[ -x "$PY"   ]] || { echo "ERROR: sglang venv python not found at $PY"; exit 1; }
[[ -x "$NSYS" ]] || { echo "ERROR: nsys not found ($NSYS)"; exit 1; }
mkdir -p "$OUT_DIR"

# CUDA libs for the sglang venv (matches the other scripts); harmless if absent.
[[ -f "$ROOT/sglang-bench/env.sh" ]] && source "$ROOT/sglang-bench/env.sh"

# HuggingFace auth (Llama is gated).
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== nsys     : $($NSYS --version 2>&1 | head -1) ==="
echo "=== model    : $MODEL ==="
echo "=== batches  : ${BATCH_SIZES[*]}  (ISL=$INPUT_LEN OSL=$OUTPUT_LEN steps=$NUM_STEPS) ==="
echo "=== gpu      : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset (sglang picks cuda:0)} ==="
echo "=== output   : $OUT_DIR ==="

# ── State shared with the cleanup trap ───────────────────────────────────────
NSYS_PID=""
BENCH_PID=""

# Reap any sglang server / nsys launcher bound to our PORT, then wait until the
# OS actually releases the port. This is REQUIRED because --capture-range-end=
# stop-shutdown finalizes the report and lets the foreground `nsys profile` exit
# while the server it spawned survives as an ORPHAN (reparented to init) — so it
# is not a child of NSYS_PID and pgid/child kills cannot reach it. We match it by
# command line instead.
_free_port() {
  pkill -9 -f "sglang.launch_server.*--port ${PORT} " 2>/dev/null || true
  pkill -9 -f "nsys-launcher"                          2>/dev/null || true
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
    # nsys launched the server as a child; kill the whole process group.
    local pgid
    pgid="$(ps -o pgid= -p "$NSYS_PID" 2>/dev/null | tr -d ' ')" || true
    kill "$NSYS_PID" 2>/dev/null || true
    sleep 3
    [[ -n "$pgid" ]] && kill -9 "-$pgid" 2>/dev/null || true
  fi
  # Reap any orphaned server that stop-shutdown left behind.
  _free_port || true
  NSYS_PID=""; BENCH_PID=""
}
trap _cleanup EXIT INT TERM

# ── Profile one decode batch size ────────────────────────────────────────────
profile_batch() {
  local B="$1"
  local OUT="$OUT_DIR/sglang_decode_b${B}_i${INPUT_LEN}"
  local SERVER_LOG="$OUT_DIR/sglang_b${B}_i${INPUT_LEN}_server.log"
  local BENCH_LOG="$OUT_DIR/sglang_b${B}_i${INPUT_LEN}_bench.log"
  rm -f "${OUT}.nsys-rep"

  # A previous batch's stop-shutdown can leave an orphaned server on $PORT.
  # Try to reap it; only fail if something we don't own is still holding it.
  if ss -tlnp 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo ">>> port $PORT busy — reaping any stale sglang server ..."
    _free_port || { echo "ERROR: port $PORT still in use after reap — kill it manually."; exit 1; }
  fi

  echo ""
  echo "─────────────────────────────────────────────────────────────"
  echo "  PROFILE  decode batch = $B   →  ${OUT}.nsys-rep"
  echo "─────────────────────────────────────────────────────────────"

  # ── Launch sglang server UNDER nsys, recording held until cudaProfilerStart ──
  # --capture-range-end=stop-shutdown: when the server issues cudaProfilerStop
  # (after num_steps), nsys finalizes the report AND shuts the server down, so
  # `wait $NSYS_PID` returns exactly when the .nsys-rep is complete.
  echo ">>> Launching server under nsys (capture armed, not recording yet) ..."
  "$NSYS" profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    -s none --cpuctxsw=none -b none \
    --force-overwrite=true \
    -o "$OUT" \
    "$PY" -m sglang.launch_server \
      --model-path "$MODEL" \
      --tp-size 1 \
      --port "$PORT" \
      --host "$HOST" \
      --mem-fraction-static 0.90 \
      --max-running-requests 256 \
      --dtype auto \
      --attention-backend triton \
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

  # ── Drive a fixed decode batch through the server ────────────────────────
  # bench_one_batch_server holds exactly B requests so decode batch == B.
  # --skip-warmup: the server already captured CUDA graphs at startup.
  echo ">>> Starting bench_one_batch_server (batch=$B, in=$INPUT_LEN, out=$OUTPUT_LEN) ..."
  "$PY" -m sglang.bench_one_batch_server \
    --model "$MODEL" \
    --base-url "http://$HOST:$PORT" \
    --batch-size "$B" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --skip-warmup \
    --run-name "poc_b${B}" \
    > "$BENCH_LOG" 2>&1 &
  BENCH_PID=$!

  # ── Wait until the server is in STEADY-STATE decode ──────────────────────
  # The "Decode batch ... cuda graph: True" log line means graphs are active and
  # (because we're already tens of steps in) Triton kernels are JIT-compiled.
  echo ">>> Waiting for steady-state decode (\"cuda graph: True\") ..."
  local dwaited=0 ready=0
  while (( dwaited < 180 )); do
    if grep -qE "Decode batch.*cuda graph: True" "$SERVER_LOG" 2>/dev/null; then
      ready=1; break
    fi
    kill -0 "$NSYS_PID" 2>/dev/null || { echo "ERROR: server exited early."; tail -30 "$SERVER_LOG"; exit 1; }
    # The bench client MUST stay alive: /start_profile waits for NUM_STEPS *decode*
    # steps, which only happen while a request is in flight. If the bench died
    # (e.g. missing dep, bad args), profiling would hang forever — so abort here.
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

  # ── Fire the capture: server calls cudaProfilerStart, runs NUM_STEPS, stops ─
  echo ">>> POST /start_profile  (CUDA_PROFILER, num_steps=$NUM_STEPS) ..."
  curl -s -X POST "http://$HOST:$PORT/start_profile" \
    -H "Content-Type: application/json" \
    -d "{\"activities\": [\"CUDA_PROFILER\"], \"num_steps\": $NUM_STEPS}" || true
  echo ""

  # ── nsys auto-stops + shuts down at cudaProfilerStop. Wait WITH A TIMEOUT ──
  # cudaProfilerStop only fires after NUM_STEPS more decode steps execute, so the
  # bench must still be decoding. Bound the wait so a stall can never hang us.
  echo ">>> Capturing ${NUM_STEPS} decode steps — waiting for nsys to finalize (max ${CAPTURE_TIMEOUT}s) ..."
  local cwaited=0
  while kill -0 "$NSYS_PID" 2>/dev/null; do
    if (( cwaited >= CAPTURE_TIMEOUT )); then
      echo "ERROR: nsys did not finalize within ${CAPTURE_TIMEOUT}s — capture stalled."
      if kill -0 "$BENCH_PID" 2>/dev/null; then
        echo "       bench still alive but decode didn't reach ${NUM_STEPS} steps."
      else
        echo "       bench already exited — raise OUTPUT_LEN so decode outlasts the window."
      fi
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

  # stop-shutdown frees NSYS_PID but can orphan the server; reap it and block
  # until $PORT is actually released so the next batch can bind cleanly.
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
  f="$OUT_DIR/sglang_decode_b${B}_i${INPUT_LEN}.nsys-rep"
  [[ -f "$f" ]] && echo "   $f"
done
echo ""
echo " Inspect in-graph kernels (now visible thanks to --cuda-graph-trace=node):"
echo "   nsys stats --report cuda_gpu_kern_sum $OUT_DIR/sglang_decode_b32.nsys-rep"
echo "═══════════════════════════════════════════════════════════════"
