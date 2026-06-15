#!/usr/bin/env bash
# profile_eagle3_decode.sh — Capture a few STEADY-STATE DECODE steps of SGLang +
#                            Llama-3.1-8B + EAGLE3 (num_steps=3) with Nsight
#                            Systems (.nsys-rep). Same input/output structure and
#                            architecture choice as run_eagle3_pareto.sh, just a
#                            handful of decode steps for an efficient trace.
#
# It mirrors ../profile_decode.sh but enables EAGLE3 spec-dec so the trace shows
# the draft-model forward(s) + the target verify step that make up one spec-dec
# decode iteration (num_steps=3 → up to ~num_draft_tokens accepted per step).
#
# ── Usage ────────────────────────────────────────────────────────────────────
#   bash profile_eagle3_decode.sh                       # baseline + eagle3, c=2 8
#   MODES="eagle3" CONCURRENCIES="2" bash profile_eagle3_decode.sh
#   PROFILE_DECODE_STEPS=8 bash profile_eagle3_decode.sh
#
# ── Output ───────────────────────────────────────────────────────────────────
#   ./decode-profiles/<mode>_c<C>.nsys-rep   → open in Nsight Systems GUI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

# ── nsys binary ──────────────────────────────────────────────────────────────
NSYS=""
for candidate in \
    "/opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/nsys" \
    "/usr/local/bin/nsys" \
    "/usr/lib/nsight-systems/Target-x86_64/x86_64/nsys"; do
  [[ -x "$candidate" ]] && { NSYS="$candidate"; break; }
done
[[ -z "$NSYS" ]] && { echo "ERROR: nsys not found."; exit 1; }
echo "=== nsys: $($NSYS --version 2>&1 | head -1) ==="

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B}"

# Pin target + draft to the same dtype (the EAGLE3 draft is float16, Llama is
# bfloat16; mixing breaks draft CUDA-graph capture). See run_eagle3_pareto.sh.
DTYPE="${DTYPE:-float16}"

# Which configs to profile (baseline gives a reference trace without spec-dec).
IFS=' ' read -r -a MODES <<< "${MODES:-baseline eagle3}"

# Concurrency levels to capture (decode batch size = C).
IFS=' ' read -r -a CONCURRENCIES <<< "${CONCURRENCIES:-2 8}"

# Natural prompts (ShareGPT) are required for a realistic EAGLE3 accept length;
# random/lorem tokens collapse AL to ~1. Prompts are capped well under the draft's
# 2048-position limit (prompt ≤ MAX_ISL, + OSL).
SHAREGPT_FILE="${SHAREGPT_FILE:-$BENCH_ROOT/datasets/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
MAX_ISL="${MAX_ISL:-1024}"
MIN_ISL="${MIN_ISL:-128}"
PROFILE_DECODE_STEPS="${PROFILE_DECODE_STEPS:-10}"   # server decode steps to record
OSL="${OSL:-160}"                            # max_tokens for the profile batch
WARM_OSL="${WARM_OSL:-$OSL}"

# EAGLE3 spec-dec knobs (num_steps=3 per the slide).
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-4}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-8}"

PROFILE_DIR="$ROOT/decode-profiles"
mkdir -p "$PROFILE_DIR"

VENV_PYTHON="$BENCH_ROOT/sglang-env/bin/python3"
[[ -x "$VENV_PYTHON" ]] || { echo "ERROR: $VENV_PYTHON not found"; exit 1; }
source "$BENCH_ROOT/sglang-bench/env.sh"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_HOST="127.0.0.1"

MAX_C=$(printf '%s\n' "${CONCURRENCIES[@]}" | sort -n | tail -1)
DATASET_FILE="$PROFILE_DIR/prompts_sharegpt_${MIN_ISL}-${MAX_ISL}.json"

# ── HuggingFace auth ─────────────────────────────────────────────────────────
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── nvidia-uvm preflight + GPU pick (memory database — failure mode A) ───────
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
_pick_free_gpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "=== GPU: using caller-supplied CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
    return
  fi
  local best_idx=0 best_score=-1
  while IFS=', ' read -r idx util free; do
    idx="${idx// /}"; util="${util// /}"; free="${free// /}"
    local score=$(( (100 - util) * 10000 + free ))
    (( score > best_score )) && { best_score=$score; best_idx=$idx; }
  done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.free \
             --format=csv,noheader,nounits 2>/dev/null)
  export CUDA_VISIBLE_DEVICES="$best_idx"
  echo "=== GPU: auto-selected GPU $best_idx ==="
}
_fix_nvidia_uvm
_pick_free_gpu

# ── Dataset: natural ShareGPT prompts (list of strings) within draft context ──
_prepare_dataset() {
  [[ -f "$DATASET_FILE" ]] && { echo "    Reusing dataset: $DATASET_FILE"; return 0; }
  echo "    Curating natural prompts from ShareGPT (${MIN_ISL}-${MAX_ISL} tok) ..."
  local n=$(( MAX_C * 2 + 16 ))
  "$VENV_PYTHON" - "$MODEL" "$SHAREGPT_FILE" "$DATASET_FILE" "$MIN_ISL" "$MAX_ISL" "$n" <<'PYEOF'
import sys, json
from transformers import AutoTokenizer
model, src, out_path, min_isl, max_isl, n = \
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
raw = json.load(open(src))
out = []
for conv in raw:
    msgs = conv.get("conversations") or []
    if len(msgs) < 2 or msgs[0].get("from") != "human":
        continue
    prompt = msgs[0].get("value", "").strip()
    if not prompt:
        continue
    if min_isl <= len(tok.encode(prompt, add_special_tokens=False)) <= max_isl:
        out.append(prompt)
    if len(out) >= n:
        break
json.dump(out, open(out_path, "w"))
print(f"    Dataset: {len(out)} natural prompts  →  {out_path}")
PYEOF
}

# ── Concurrent request sender (stdlib only) ──────────────────────────────────
_send_batch() {
  local offset=$1 count=$2 osl=$3
  "$VENV_PYTHON" - "$SERVER_HOST" "$SERVER_PORT" "$DATASET_FILE" "$offset" "$count" "$osl" <<'PYEOF'
import sys, json, threading, urllib.request
host, port, ds, offset, count, osl = sys.argv[1:7]
port=int(port); offset=int(offset); count=int(count); osl=int(osl)
base=f"http://{host}:{port}"
prompts=json.load(open(ds))[offset:offset+count]
def one(prompt):
    payload={"text":prompt,"sampling_params":{"max_new_tokens":osl,"ignore_eos":True,"temperature":0.0}}
    req=urllib.request.Request(base+"/generate", data=json.dumps(payload).encode(),
                               headers={"Content-Type":"application/json"})
    urllib.request.urlopen(req, timeout=1800).read()
errs=[]
def worker(p):
    try: one(p)
    except Exception as e: errs.append(repr(e))
ts=[threading.Thread(target=worker,args=(p,)) for p in prompts]
for t in ts: t.start()
for t in ts: t.join()
if errs: print(f"  [warn] {len(errs)} req error(s); first: {errs[0]}", file=sys.stderr)
PYEOF
}

# ── Monotonic decode-step counter (one "#running-req: C," line per server step) ─
_decode_marker() { grep -c "#running-req: ${C}," "$SERVER_LOG" 2>/dev/null || echo 0; }

# ══════════════════════════════════════════════════════════════════════════════
#  Build server command for a mode
# ══════════════════════════════════════════════════════════════════════════════
_build_server_cmd() {
  local mode=$1
  SERVER_CMD=(
    "$VENV_PYTHON" -m sglang.launch_server
    --model-path "$MODEL"
    --tp-size 1
    --port "$SERVER_PORT"
    --host "$SERVER_HOST"
    --mem-fraction-static 0.85
    --max-running-requests 64
    --dtype "$DTYPE"
    --attention-backend triton
    --cuda-graph-max-bs "$MAX_C"
    --enable-metrics
    --decode-log-interval 1
  )
  if [[ "$mode" == "eagle3" ]]; then
    SERVER_CMD+=(
      --speculative-algorithm        EAGLE3
      --speculative-draft-model-path "$DRAFT_MODEL"
      --speculative-num-steps        "$SPEC_NUM_STEPS"
      --speculative-eagle-topk       "$SPEC_EAGLE_TOPK"
      --speculative-num-draft-tokens "$SPEC_NUM_DRAFT_TOKENS"
    )
  fi
}

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " MODEL          : $MODEL  (+ EAGLE3 draft: $DRAFT_MODEL)"
echo " MODES          : ${MODES[*]}"
echo " CONCURRENCIES  : ${CONCURRENCIES[*]}"
echo " PROMPTS        : natural ShareGPT, ${MIN_ISL}-${MAX_ISL} tok"
echo " DECODE STEPS   : $PROFILE_DECODE_STEPS recorded  |  OSL: $OSL"
echo " EAGLE3         : num_steps=$SPEC_NUM_STEPS topk=$SPEC_EAGLE_TOPK draft_tokens=$SPEC_NUM_DRAFT_TOKENS"
echo " OUTPUT         : $PROFILE_DIR  (*.nsys-rep → Nsight Systems GUI)"
echo "═══════════════════════════════════════════════════════════"

_prepare_dataset

# ══════════════════════════════════════════════════════════════════════════════
#  Main loop — one fresh server (under nsys) per (mode, concurrency)
# ══════════════════════════════════════════════════════════════════════════════
for mode in "${MODES[@]}"; do
  [[ "$mode" == "baseline" || "$mode" == "eagle3" ]] || { echo "ERROR: bad MODE '$mode'"; exit 1; }
  _build_server_cmd "$mode"

  for C in "${CONCURRENCIES[@]}"; do
    PROFILE_OUT="$PROFILE_DIR/${mode}_c${C}"
    SERVER_LOG="$PROFILE_DIR/${mode}_c${C}_server.log"
    SESSION="${mode}_c${C}_$$"

    if ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT}[[:space:]]"; then
      echo "ERROR: port $SERVER_PORT already in use (stale server?)."; exit 1
    fi

    echo ""
    echo "─────────────────────────────────────────────────────────"
    echo "  Profiling  mode=$mode  concurrency=$C  →  ${PROFILE_OUT}.nsys-rep"
    echo "─────────────────────────────────────────────────────────"

    echo ">>> Launching server under nsys (session=$SESSION) ..."
    "$NSYS" launch \
      --trace=cuda,nvtx \
      --session-new="$SESSION" \
      "${SERVER_CMD[@]}" \
      > "$SERVER_LOG" 2>&1 &
    LAUNCHER_PID=$!

    _cleanup_server() {
      local pgid
      pgid=$(ps -o pgid= -p "$LAUNCHER_PID" 2>/dev/null | tr -d ' ') || true
      kill "$LAUNCHER_PID" 2>/dev/null || true
      for _ in $(seq 1 10); do kill -0 "$LAUNCHER_PID" 2>/dev/null || break; sleep 1; done
      [[ -n "$pgid" ]] && kill -0 "$LAUNCHER_PID" 2>/dev/null && kill -9 -"$pgid" 2>/dev/null || true
      wait "$LAUNCHER_PID" 2>/dev/null || true
    }
    trap '_cleanup_server' EXIT

    echo "    Waiting for http://$SERVER_HOST:$SERVER_PORT/health ..."
    start=$(date +%s)
    while true; do
      curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" &>/dev/null && { echo "    Server ready ($(( $(date +%s)-start ))s)."; break; }
      kill -0 "$LAUNCHER_PID" 2>/dev/null || { echo "ERROR: server died. Last 40 lines:"; tail -40 "$SERVER_LOG"; exit 1; }
      (( $(date +%s)-start >= 600 )) && { echo "ERROR: server timed out. Last 40 lines:"; tail -40 "$SERVER_LOG"; exit 1; }
      sleep 5
    done

    echo "    Warmup: $C reqs × $WARM_OSL tok ..."
    _send_batch 0 "$C" "$WARM_OSL"

    BASELINE=$(_decode_marker)
    echo "    Firing profile batch: $C reqs × $OSL tok (decode-step baseline=$BASELINE) ..."
    ( _send_batch "$C" "$C" "$OSL" ) &
    BENCH_PID=$!

    echo ">>> Waiting for full-batch (C=$C) decode to start ..."
    polls=0
    while (( polls < 1200 )); do
      (( $(_decode_marker) > BASELINE )) && { echo "    Decode running."; break; }
      kill -0 "$BENCH_PID" 2>/dev/null || { echo "WARN: batch exited before decode."; break; }
      sleep 0.05; (( polls++ )) || true
    done

    echo ">>> nsys START (decode-only recording) ..."
    "$NSYS" start --output "$PROFILE_OUT" --force-overwrite true --session="$SESSION"

    START_MARK=$(_decode_marker)
    TARGET=$(( START_MARK + PROFILE_DECODE_STEPS ))
    echo ">>> Recording decode steps ${START_MARK} → ${TARGET} ..."
    polls=0
    while (( polls < 1200 )); do
      (( $(_decode_marker) >= TARGET )) && { echo "    Captured ${PROFILE_DECODE_STEPS} steps."; break; }
      kill -0 "$BENCH_PID" 2>/dev/null || { echo "    Batch finished early (marker $(_decode_marker))."; break; }
      sleep 0.02; (( polls++ )) || true
    done

    echo ">>> nsys STOP ..."
    "$NSYS" stop --session="$SESSION"

    kill "$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true

    # Report measured EAGLE3 accept length for context.
    if [[ "$mode" == "eagle3" ]]; then
      AL=$(grep -oE "accept len: [0-9.]+" "$SERVER_LOG" | grep -oE "[0-9.]+$" \
           | awk '{s+=$1;n++} END{if(n>0) printf "%.2f", s/n}')
      [[ -n "$AL" ]] && echo "    EAGLE3 mean accept length (whole run): $AL"
    fi

    if [[ -f "${PROFILE_OUT}.nsys-rep" ]]; then
      echo "    Written: ${PROFILE_OUT}.nsys-rep  ($(du -sh "${PROFILE_OUT}.nsys-rep" | cut -f1))"
    else
      echo "    WARNING: ${PROFILE_OUT}.nsys-rep not found."
    fi

    echo ">>> Stopping server ..."
    _cleanup_server
    trap - EXIT
    echo "    Sleeping 5s before next ..."
    sleep 5
  done
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " All decode captures complete."
shopt -s nullglob
for f in "$PROFILE_DIR/"*.nsys-rep; do echo "   $f"; done
shopt -u nullglob
echo ""
echo " View: copy *.nsys-rep to your machine and open in Nsight Systems GUI."
echo " Compare baseline_c<C> vs eagle3_c<C>: the eagle3 trace shows num_steps=$SPEC_NUM_STEPS"
echo " draft forwards + 1 target verify per decode iteration."
echo "═══════════════════════════════════════════════════════════"
