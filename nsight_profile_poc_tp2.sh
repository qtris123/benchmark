#!/usr/bin/env bash
# nsight_profile_poc_tp2.sh — TP=2 variant of nsight_profile_poc.sh.
# Profiles SGLang decode the "cudaProfilerApi" way (server-controlled capture,
# in-graph kernel visibility) but across BOTH A100X with --tp-size 2.
#
# Same capture mechanism as the TP=1 PoC:
#   • --cuda-graph-trace=node   → records individual kernels INSIDE CUDA graphs.
#   • --capture-range=cudaProfilerApi + /start_profile (num_steps) → the SERVER
#     starts/stops the capture at exact decode-step boundaries.
#   • No warmup: we wait for steady-state decode ("cuda graph: True") before
#     profiling, so graphs are captured and Triton kernels JIT-compiled.
#
# What's DIFFERENT for TP=2 on THIS machine (2× A100X, cross-NUMA PCIe, NO NVLink,
# driver 570 / max CUDA 12.8). All of these mirror benchmark/run_static_benchmark.sh
# and the gpu-devops-a100x-driver570 skill (Failure Mode G):
#   • --tp-size 2 and ≥2 visible GPUs. We auto-select 2 GPUs (or validate a
#     caller-supplied CUDA_VISIBLE_DEVICES has ≥2 entries) — otherwise rank>0
#     workers call set_device() on a missing ordinal → "invalid device ordinal".
#   • NCCL_IB_DISABLE=1 — no NVLink, so NCCL would route over the RoCE NICs, but
#     IB memory registration fails (RLIMIT_MEMLOCK is only 64 MB) →
#     "ibv_reg_mr_iova2: Cannot allocate memory". Forces the fast PCIe-P2P path.
#   • NCCL_CUMEM_ENABLE=0 — the cuMem (VMM) allocator deadlocks CUDA-graph-captured
#     collectives on this box (both TP ranks hit the 300 s watchdog → SIGQUIT,
#     the classic TP=2 decode hang). Disabling it keeps P2P on, no throughput cost.
#
# Scope: SGLang only (the cudaProfilerApi trigger is sglang-specific). PoC.
#
# Usage:
#   bash nsight_profile_poc_tp2.sh                 # batches 2 and 32
#   BATCH_SIZES="32" bash nsight_profile_poc_tp2.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash nsight_profile_poc_tp2.sh   # pin the GPU pair
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Config (override via env) ────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
IFS=' ' read -r -a BATCH_SIZES <<< "${BATCH_SIZES:- 32}"  # decode batch = concurrency
INPUT_LEN="${INPUT_LEN:-10000}"     # prefill tokens per request
OUTPUT_LEN="${OUTPUT_LEN:-1024}"   # must outlast detection + the profiled window
NUM_STEPS="${NUM_STEPS:-11}"       # decode steps to capture (server-controlled)
CAPTURE_TIMEOUT="${CAPTURE_TIMEOUT:-120}"  # hard cap on the nsys-finalize wait (s)
TP="${TP:-2}"                       # tensor-parallel size (this variant: 2)
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"  # empty = engine default (cap for big models)
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
OUT_DIR="${OUT_DIR:-$ROOT/nsight-profiles-poc-tp2-cpu-enable}"

PY="$ROOT/sglang-env/bin/python3"
NSYS="${NSYS:-$(command -v nsys || echo /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys)}"

[[ -x "$PY"   ]] || { echo "ERROR: sglang venv python not found at $PY"; exit 1; }
[[ -x "$NSYS" ]] || { echo "ERROR: nsys not found ($NSYS)"; exit 1; }
mkdir -p "$OUT_DIR"

# CUDA libs for the sglang venv (matches the other scripts); harmless if absent.
[[ -f "$ROOT/sglang-bench/env.sh" ]] && source "$ROOT/sglang-bench/env.sh"

# ── GPU selection (TP>1 needs ≥TP visible GPUs) ──────────────────────────────
# sglang binds device ordinals 0..TP-1 of whatever CUDA_VISIBLE_DEVICES exposes.
# Too few visible → rank>0 workers set_device() a missing ordinal → "invalid
# device ordinal". Validate a caller list or auto-pick the TP least-loaded GPUs.
_pick_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local n_vis; n_vis=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
    if (( n_vis < TP )); then
      echo "ERROR: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES exposes $n_vis GPU(s) but TP=$TP." >&2
      echo "       Provide at least $TP devices (e.g. CUDA_VISIBLE_DEVICES=0,1) or set TP=$n_vis." >&2
      exit 1
    fi
    echo "=== GPU: using caller-supplied CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (TP=$TP) ==="
    return
  fi
  local picks; picks=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
                         --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', *' '{ printf "%d %d\n", (100-$2)*100000+$3, $1 }' \
    | sort -rn | awk -v n="$TP" 'NR<=n{print $2}' | paste -sd, -)
  local n_pick; n_pick=$(awk -F, '{print NF}' <<< "$picks")
  if [[ -z "$picks" || "$n_pick" -lt "$TP" ]]; then
    echo "ERROR: need $TP GPU(s) for TP=$TP but only found ${n_pick:-0} (picks='$picks')." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$picks"
  echo "=== GPU: auto-selected GPUs $picks for TP=$TP ==="
}
_pick_gpus

# ── NCCL transport (TP>1) ────────────────────────────────────────────────────
# No NVLink + 64 MB RLIMIT_MEMLOCK → NCCL's RoCE/IB path fails registration, so
# force PCIe-P2P. cuMem off avoids the CUDA-graph collective deadlock (TP=2 hang).
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0

# HuggingFace auth (Llama is gated).
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}" HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "=== nsys     : $($NSYS --version 2>&1 | head -1) ==="
echo "=== model    : $MODEL  (TP=$TP) ==="
echo "=== batches  : ${BATCH_SIZES[*]}  (ISL=$INPUT_LEN OSL=$OUTPUT_LEN steps=$NUM_STEPS) ==="
echo "=== gpu      : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
echo "=== output   : $OUT_DIR ==="

# ── State shared with the cleanup trap ───────────────────────────────────────
NSYS_PID=""
BENCH_PID=""

# Reap any sglang server / nsys launcher bound to our PORT, then wait until the
# OS actually releases the port. This is REQUIRED because --capture-range-end=
# stop-shutdown finalizes the report and lets the foreground `nsys profile` exit
# while the server it spawned survives as an ORPHAN (reparented to init) — so it
# is not a child of NSYS_PID and pgid/child kills cannot reach it. We match it by
# command line instead. At TP>1 sglang also spawns scheduler/worker subprocesses;
# kill those too so the GPUs are released for the next batch.
_free_port() {
  pkill -9 -f "sglang.launch_server.*--port ${PORT} " 2>/dev/null || true
  pkill -9 -f "sglang::scheduler"                     2>/dev/null || true
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
  local OUT="$OUT_DIR/sglang_decode_b${B}_i${INPUT_LEN}_tp${TP}"
  local SERVER_LOG="$OUT_DIR/sglang_b${B}_i${INPUT_LEN}_tp${TP}_server.log"
  local BENCH_LOG="$OUT_DIR/sglang_b${B}_i${INPUT_LEN}_tp${TP}_bench.log"
  rm -f "${OUT}.nsys-rep"

  # A previous batch's stop-shutdown can leave an orphaned server on $PORT.
  # Try to reap it; only fail if something we don't own is still holding it.
  if ss -tlnp 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo ">>> port $PORT busy — reaping any stale sglang server ..."
    _free_port || { echo "ERROR: port $PORT still in use after reap — kill it manually."; exit 1; }
  fi

  echo ""
  echo "─────────────────────────────────────────────────────────────"
  echo "  PROFILE  decode batch = $B  (TP=$TP)   →  ${OUT}.nsys-rep"
  echo "─────────────────────────────────────────────────────────────"

  # ── Launch sglang server UNDER nsys, recording held until cudaProfilerStart ──
  # --capture-range-end=stop-shutdown: when the server issues cudaProfilerStop
  # (after num_steps), nsys finalizes the report AND shuts the server down, so
  # `wait $NSYS_PID` returns exactly when the .nsys-rep is complete.
  echo ">>> Launching server under nsys (TP=$TP, capture armed, not recording yet) ..."
  local server_args=(
    --model-path "$MODEL"
    --tp-size "$TP"
    --port "$PORT"
    --host "$HOST"
    --mem-fraction-static "$MEM_FRACTION"
    --max-running-requests 256
    --dtype auto
    --attention-backend triton
  )
  [[ -n "$CUDA_GRAPH_MAX_BS" ]] && server_args+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
  "$NSYS" profile \
    `# osrt traces blocking OS-runtime calls (futex/poll/pthread_cond_wait/read/…).` \
    `# Unlike CPU sampling it does NOT use perf_event, so it works even where the` \
    `# kernel locks perf down (see below). It is how we tell, during a GPU-idle gap` \
    `# between decode steps, whether a worker thread is OFF-CPU blocked-waiting` \
    `# (long futex/poll == waiting on IPC/lock/sync) vs busy in Python (no osrt range).` \
    --trace=cuda,nvtx,osrt \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop-shutdown \
    `# CPU IP sampling + OS context-switch tracing + native (dwarf) backtraces give` \
    `# the fullest on-CPU vs off-CPU picture. NOTE: both --sample and --cpuctxsw go` \
    `# through Linux perf_event, which needs kernel.perf_event_paranoid<=2 (ideally` \
    `# <=1) or CAP_PERFMON. On THIS leased box paranoid=3 AND CapEff=0 (userns root),` \
    `# so they currently collect NOTHING (no COMPOSITE_EVENTS/sched rows) — osrt above` \
    `# is the working substitute. Left enabled so the trace is complete on any host` \
    `# where perf is unlocked (host admin: sysctl kernel.perf_event_paranoid=1).` \
    --sample=process-tree --cpuctxsw=process-tree -b dwarf \
    --force-overwrite=true \
    -o "$OUT" \
    "$PY" -m sglang.launch_server "${server_args[@]}" \
    > "$SERVER_LOG" 2>&1 &
  NSYS_PID=$!
  echo "    nsys PID=$NSYS_PID  server log=$SERVER_LOG"

  # ── Wait for server health ───────────────────────────────────────────────
  # TP>1 startup is slower (per-rank load + NCCL init + graph capture on 2 GPUs).
  echo ">>> Waiting for http://$HOST:$PORT/health ..."
  local waited=0
  until curl -sf "http://$HOST:$PORT/health" &>/dev/null; do
    kill -0 "$NSYS_PID" 2>/dev/null || { echo "ERROR: server died:"; tail -30 "$SERVER_LOG"; exit 1; }
    (( waited >= 600 )) && { echo "ERROR: server health timeout."; tail -30 "$SERVER_LOG"; exit 1; }
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
    --run-name "poc_b${B}_tp${TP}" \
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
echo " PoC complete (TP=$TP). Reports:"
for B in "${BATCH_SIZES[@]}"; do
  f="$OUT_DIR/sglang_decode_b${B}_i${INPUT_LEN}_tp${TP}.nsys-rep"
  [[ -f "$f" ]] && echo "   $f"
done
echo ""
echo " Inspect in-graph kernels (now visible thanks to --cuda-graph-trace=node):"
echo "   nsys stats --report cuda_gpu_kern_sum $OUT_DIR/sglang_decode_b32_i${INPUT_LEN}_tp${TP}.nsys-rep"
echo "═══════════════════════════════════════════════════════════════"
