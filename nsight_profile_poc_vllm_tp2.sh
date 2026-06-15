#!/usr/bin/env bash
# nsight_profile_poc_vllm_tp2.sh — TP=2 variant of nsight_profile_poc_vllm.sh.
# Profiles vLLM DECODE the "cudaProfilerApi" way (in-graph kernel visibility)
# across BOTH A100X with --tensor-parallel-size 2, so the reports line up
# step-for-step with the SGLang TP=2 PoC.
#
# Same capture mechanism as the TP=1 PoC (verified against vllm 0.22.0 source):
#   • --cuda-graph-trace=node        → records kernels INSIDE CUDA graphs.
#   • --trace-fork-before-exec=true + VLLM_WORKER_MULTIPROC_METHOD=spawn
#                                    → nsys follows vLLM into its worker subprocesses
#                                      (where the GPU work + cudaProfilerStart happen).
#   • --profiler-config '{"profiler":"cuda","max_iterations":N}' + /start_profile
#                                    → torch.cuda.profiler.start()/stop() == cuda
#                                      ProfilerStart/Stop (N == vLLM's num_steps).
#   • No warmup burn-in: wait for steady-state decode (prompt tput 0, Running == B)
#     before /start_profile, so CUDA graphs are captured + Triton kernels JIT'd.
#
# What's DIFFERENT for TP=2 on THIS machine (2× A100X, cross-NUMA PCIe, NO NVLink,
# driver 570 / max CUDA 12.8). All mirror benchmark/run_static_benchmark.sh and the
# gpu-devops-a100x-driver570 skill (Failure Mode G):
#   • --tensor-parallel-size 2 and ≥2 visible GPUs. We auto-select 2 GPUs (or
#     validate a caller-supplied CUDA_VISIBLE_DEVICES has ≥2 entries) — otherwise
#     rank>0 workers set_device() a missing ordinal → "invalid device ordinal".
#   • --disable-custom-all-reduce — vLLM 0.22.0's custom all-reduce is a CUDA-13
#     kernel; driver 570 only supports CUDA 12.8, so at TP>1 it dies with
#     cudaErrorInsufficientDriver. Disabling falls back to plain NCCL all-reduce
#     (no perf loss with no NVLink anyway).
#   • NCCL_IB_DISABLE=1 — no NVLink → NCCL would route over RoCE NICs, but IB
#     memory registration fails (RLIMIT_MEMLOCK only 64 MB). Forces PCIe-P2P.
#   • NCCL_CUMEM_ENABLE=0 — the cuMem allocator deadlocks CUDA-graph collectives
#     on this box (TP ranks hit the 300 s watchdog → SIGQUIT). Keeps P2P on.
#
# Scope: vLLM only, PoC.
#
# Usage:
#   bash nsight_profile_poc_vllm_tp2.sh                 # batches 2 and 32
#   BATCH_SIZES="32" bash nsight_profile_poc_vllm_tp2.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash nsight_profile_poc_vllm_tp2.sh   # pin the GPU pair
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Config (override via env) ────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
IFS=' ' read -r -a BATCH_SIZES <<< "${BATCH_SIZES:-2 32}"  # decode batch = concurrency
INPUT_LEN="${INPUT_LEN:-10000}"      # prefill tokens per request (bump to 1000 to match bench)
# OUTPUT_LEN must keep the fixed batch decoding well past vLLM's ~10s throughput-
# log granularity + steady-state detection + the NUM_STEPS capture window. A small
# OSL (e.g. 1024) lets a tiny batch drain in ~12s — faster than we can detect it —
# so the profiler arms after the batch is already idle and never reaches max_iters.
OUTPUT_LEN="${OUTPUT_LEN:-8192}"
NUM_STEPS="${NUM_STEPS:-10}"       # decode steps to capture (== profiler max_iterations)
CAPTURE_TIMEOUT="${CAPTURE_TIMEOUT:-120}"  # hard cap on the nsys-finalize wait (s)
TP="${TP:-2}"                      # tensor-parallel size (this variant: 2)
MEM_FRACTION="${MEM_FRACTION:-0.90}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-}"  # empty = engine default (cap for big models)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
OUT_DIR="${OUT_DIR:-$ROOT/nsight-profiles-poc-tp2-cpu-enable}"

VLLM="$ROOT/vllm-env/bin/vllm"
NSYS="${NSYS:-$(command -v nsys || echo /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys)}"

[[ -x "$VLLM" ]] || { echo "ERROR: vllm CLI not found at $VLLM"; exit 1; }
[[ -x "$NSYS" ]] || { echo "ERROR: nsys not found ($NSYS)"; exit 1; }
mkdir -p "$OUT_DIR"

# CUDA libs for the vllm venv (matches the benchmark); harmless if absent.
[[ -f "$ROOT/vllm-bench/env.sh" ]] && source "$ROOT/vllm-bench/env.sh"

# nsys must follow vLLM into its spawned worker process(es) — and at TP>1 there is
# one worker per rank, all of which must be traced.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── GPU selection (TP>1 needs ≥TP visible GPUs) ──────────────────────────────
# vLLM binds device ordinals 0..TP-1 of whatever CUDA_VISIBLE_DEVICES exposes.
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

# ── Steady-state detection via /metrics (NOT the server log) ─────────────────
# This vLLM build (v0.22.0, V1 engine) does NOT emit the "Avg prompt throughput
# ... Running: N reqs" stat line the TP=1 PoC grepped for — the log stays silent
# while a worker JIT-compiles the Triton attention kernel for the large ISL shape.
# So we poll the Prometheus /metrics endpoint instead, which is served by the API
# front-end even while the engine is busy. _metric_sum sums all label series for a
# metric (one per engine rank) and prints an integer.
_metric_sum() {
  local m="$1"
  curl -sf "http://$HOST:$PORT/metrics" 2>/dev/null \
    | awk -v m="$m" '$1 !~ /^#/ { tok=$1; sub(/\{.*/,"",tok); if (tok==m) s+=$2 }
                     END { printf "%.0f", s+0 }'
}

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

# Reap any vLLM server / nsys launcher bound to our PORT, then wait until the OS
# actually releases the port. REQUIRED because --capture-range-end=stop-shutdown
# lets the foreground `nsys profile` exit while the vllm serve process (and its
# spawned EngineCore worker(s)) survive as ORPHANs — not children of NSYS_PID — so
# we match them by command line instead. At TP>1 there is one worker per rank.
_free_port() {
  pkill -9 -f "vllm serve .*--port ${PORT} " 2>/dev/null || true
  pkill -9 -f "EngineCore"                    2>/dev/null || true
  pkill -9 -f "VLLM::Worker"                  2>/dev/null || true
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
  local OUT="$OUT_DIR/vllm_decode_b${B}_i${INPUT_LEN}_tp${TP}"
  local SERVER_LOG="$OUT_DIR/vllm_b${B}_i${INPUT_LEN}_tp${TP}_server.log"
  local BENCH_LOG="$OUT_DIR/vllm_b${B}_i${INPUT_LEN}_tp${TP}_bench.log"
  rm -f "${OUT}.nsys-rep"

  # A previous batch's stop-shutdown can leave an orphaned server on $PORT.
  if ss -tlnp 2>/dev/null | grep -q ":${PORT}[[:space:]]"; then
    echo ">>> port $PORT busy — reaping any stale vllm server ..."
    _free_port || { echo "ERROR: port $PORT still in use after reap — kill it manually."; exit 1; }
  fi

  echo ""
  echo "─────────────────────────────────────────────────────────────"
  echo "  PROFILE  decode batch = $B  (TP=$TP)   →  ${OUT}.nsys-rep"
  echo "─────────────────────────────────────────────────────────────"

  # ── Launch vLLM server UNDER nsys, recording held until cudaProfilerStart ───
  echo ">>> Launching server under nsys (TP=$TP, capture armed, not recording yet) ..."
  local server_args=(
    serve "$MODEL"
    --dtype auto
    --attention-backend TRITON_ATTN
    --tensor-parallel-size "$TP"
    --max-num-seqs 256
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$MEM_FRACTION"
    # CUDA-13 custom all-reduce vs driver-570 → fall back to NCCL all-reduce.
    --disable-custom-all-reduce
    --host "$HOST"
    --port "$PORT"
    --profiler-config "{\"profiler\":\"cuda\",\"max_iterations\":$NUM_STEPS}"
  )
  [[ -n "$CUDA_GRAPH_MAX_BS" ]] && server_args+=(--cuda-graph-sizes "$CUDA_GRAPH_MAX_BS")
  "$NSYS" profile \
    `# osrt traces blocking OS-runtime calls (futex/poll/pthread_cond_wait/read/…).` \
    `# Unlike CPU sampling it does NOT use perf_event, so it works even where the` \
    `# kernel locks perf down (see below). It is how we tell, during a GPU-idle gap` \
    `# between decode steps, whether a worker thread is OFF-CPU blocked-waiting` \
    `# (long futex/poll == waiting on IPC/lock/sync) vs busy in Python (no osrt range).` \
    --trace=cuda,nvtx,osrt \
    --trace-fork-before-exec=true \
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
    "$VLLM" "${server_args[@]}" \
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

  # ── Wait until the server is in STEADY-STATE decode (poll /metrics) ──────────
  # Genuine in-flight decode requires: all B requests admitted+running
  # (num_requests_running == B), none queued (num_requests_waiting == 0), AND the
  # generation-token counter actually advancing between polls (so we're PAST
  # prefill + the one-time Triton-kernel JIT, not still stuck in it). The ISL is
  # large here, so first-decode JIT can take a couple of minutes → 300s timeout.
  echo ">>> Waiting for steady-state decode (running==$B, waiting==0, gen advancing) ..."
  local dwaited=0 ready=0 prev_gen=-1 run wait_ gen
  while (( dwaited < 300 )); do
    run="$(_metric_sum vllm:num_requests_running)"
    wait_="$(_metric_sum vllm:num_requests_waiting)"
    gen="$(_metric_sum vllm:generation_tokens_total)"
    if [[ "$run" == "$B" && "$wait_" == "0" && "$prev_gen" != "-1" ]] && (( gen > prev_gen )); then
      ready=1; break
    fi
    prev_gen="$gen"
    kill -0 "$NSYS_PID" 2>/dev/null || { echo "ERROR: server exited early."; tail -30 "$SERVER_LOG"; exit 1; }
    # The bench client MUST stay alive: profiling only progresses while decode
    # steps execute, i.e. while a request is in flight. Abort if it died.
    if ! kill -0 "$BENCH_PID" 2>/dev/null; then
      echo "ERROR: bench client exited before decode started — not profiling."
      echo "       Last lines of $BENCH_LOG:"; tail -15 "$BENCH_LOG"
      exit 1
    fi
    sleep 2; (( dwaited += 2 ))
  done
  (( ready == 1 )) || { echo "ERROR: decode not detected within ${dwaited}s (running=$run waiting=$wait_ gen=$gen)."; tail -30 "$SERVER_LOG"; exit 1; }
  echo "    Steady-state decode detected after ${dwaited}s (running=$run, gen=$gen)."
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
echo " PoC complete (TP=$TP). Reports:"
for B in "${BATCH_SIZES[@]}"; do
  f="$OUT_DIR/vllm_decode_b${B}_i${INPUT_LEN}_tp${TP}.nsys-rep"
  [[ -f "$f" ]] && echo "   $f"
done
echo ""
echo " Inspect in-graph kernels (visible thanks to --cuda-graph-trace=node):"
echo "   nsys stats --report cuda_gpu_kern_sum $OUT_DIR/vllm_decode_b32_i${INPUT_LEN}_tp${TP}.nsys-rep"
echo "═══════════════════════════════════════════════════════════════"
