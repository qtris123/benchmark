#!/usr/bin/env bash
# run_eagle3_pareto.sh — Reproduce the "per-user TPS & system TPS vs concurrency"
#                        Pareto chart from the SGLang/Kimi-K2.5 slide, but at a
#                        scale appropriate to THIS machine (1× A100X, sm_80) using
#                        SGLang + Llama-3.1-8B + EAGLE3 spec-dec (num_steps=3).
#
# It sweeps a list of concurrency levels for TWO configurations and overlays them:
#     • baseline  — SGLang, no speculative decoding
#     • eagle3    — SGLang + EAGLE3 draft, --speculative-num-steps=3
# producing the same two panels as the slide:
#     1. per-user throughput (tok/s/user = 1000/TPOT) vs concurrency
#     2. system throughput   (tok/s/GPU)              vs concurrency
# plus the measured EAGLE3 acceptance length (AL) per concurrency.
#
# ── Why these defaults (memory database / A100X notes) ───────────────────────
#   • SGLang 0.4.9.post6 + sgl_kernel 0.2.7  → has sm_80 kernels (A100 OK).
#   • --attention-backend triton             → EAGLE3 has a TritonMultiStepDraft
#     backend, so we avoid the flashinfer cu13-nvcc JIT failure documented for
#     this box (see gpu-devops-a100x-driver570 skill, failure mode F).
#   • The slide ran Kimi-K2.5-NVFP4 on B200 with 50K shared prefix + 2K step +
#     1K output. We scale DOWN: Llama-3.1-8B, ISL≈2K, OSL=512, conc 1..32.
#
# ── Usage ────────────────────────────────────────────────────────────────────
#   bash run_eagle3_pareto.sh                      # runs baseline + eagle3, plots
#   MODES="eagle3" bash run_eagle3_pareto.sh       # only eagle3
#   CONCURRENCY_STEPS="1 2 4 8" OSL=256 bash run_eagle3_pareto.sh   # quick
#   PLOT_ONLY=1 bash run_eagle3_pareto.sh          # re-plot from existing summaries
#
# ── Output (under eagle3-bench/pareto-results/<exp>/) ────────────────────────
#   <mode>/summary.jsonl                 one JSON row per concurrency level
#   <mode>/concurrency=<C>/bench.log     raw client log
#   <mode>/concurrency=<C>/server_al.txt EAGLE3 accept-length samples
#   pareto/EAGLE3_PARETO.png             2-panel baseline-vs-eagle3 comparison
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$ROOT/.." && pwd)"          # .../benchmark  (has sglang-env, sglang-bench)
cd "$ROOT"

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
DRAFT_MODEL="${DRAFT_MODEL:-lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B}"

# Prompt source:
#   sharegpt  → real natural-language conversations (default). EAGLE3 is trained
#               on natural text, so a coherent prompt is REQUIRED to get a
#               realistic acceptance length (lorem-ipsum / random tokens give AL≈1).
#   fixed     → sliding window over random_text.txt (kept for parity with the
#               old vLLM/sglang pareto scripts; gives AL≈1, not recommended here).
DATASET="${DATASET:-sharegpt}"
SHAREGPT_FILE="${SHAREGPT_FILE:-$BENCH_ROOT/datasets/ShareGPT_V4.3_unfiltered_cleaned_split.json}"

# The Llama-3.1-8B EAGLE3 draft has max_position_embeddings=2048, so the WHOLE
# sequence (prompt + generated) must stay < 2048 or the draft predicts garbage
# and AL collapses to ~1. We cap prompt length and keep OSL modest accordingly.
MAX_ISL="${MAX_ISL:-1024}"   # only keep conversations whose prompt ≤ this many tokens
MIN_ISL="${MIN_ISL:-128}"
OSL="${OSL:-512}"            # output tokens / request (prompt ≤1024 + 512 < 2048)
ISL="${ISL:-1024}"           # used only when DATASET=fixed
TP="${TP:-1}"

# The EAGLE3 draft (lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B) ships as float16,
# while Llama-3.1-8B is bfloat16. Mixing them breaks the draft CUDA-graph capture
# ("expected mat1 and mat2 to have the same dtype, float != c10::Half"), so we
# pin BOTH target and draft to the same dtype. float16 keeps baseline and eagle3
# comparable; on A100 fp16 vs bf16 throughput is effectively identical.
DTYPE="${DTYPE:-float16}"

IFS=' ' read -r -a CONCURRENCY_STEPS <<< "${CONCURRENCY_STEPS:-1 2 4 8 16 32}"
PROMPTS_MULT="${PROMPTS_MULT:-2}"     # num_prompts = PROMPTS_MULT × C per step

# EAGLE3 spec-dec knobs — num_steps=3 per the slide.
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-4}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-8}"

# Which configurations to run.
IFS=' ' read -r -a MODES <<< "${MODES:-baseline eagle3}"
PLOT_ONLY="${PLOT_ONLY:-0}"

if [[ "$DATASET" == "sharegpt" ]]; then
  EXPERIMENT="${EXPERIMENT:-llama3.1-8b_sharegpt_maxisl${MAX_ISL}_osl${OSL}_tp${TP}}"
else
  EXPERIMENT="${EXPERIMENT:-llama3.1-8b_fixedisl${ISL}_osl${OSL}_tp${TP}}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/pareto-results}"
EXPERIMENT_DIR="$OUTPUT_DIR/$EXPERIMENT"
DATASET_DIR="$ROOT/datasets"
if [[ "$DATASET" == "sharegpt" ]]; then
  DATASET_FILE="$DATASET_DIR/sharegpt_isl${MIN_ISL}-${MAX_ISL}.json"
else
  DATASET_FILE="$DATASET_DIR/fixed_isl${ISL}.json"
fi
mkdir -p "$EXPERIMENT_DIR" "$DATASET_DIR"

# ── Reuse the existing sglang env + CUDA env (memory database) ────────────────
VENV_PYTHON="$BENCH_ROOT/sglang-env/bin/python3"
[[ -x "$VENV_PYTHON" ]] || { echo "ERROR: $VENV_PYTHON not found"; exit 1; }
source "$BENCH_ROOT/sglang-bench/env.sh"
SERVER_PORT="${SERVER_PORT:-30000}"
SERVER_HOST="127.0.0.1"

MAX_C=$(printf '%s\n' "${CONCURRENCY_STEPS[@]}" | sort -n | tail -1)

# ── HuggingFace auth ──────────────────────────────────────────────────────────
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ══════════════════════════════════════════════════════════════════════════════
#  GPU / UVM preflight (memory database — failure mode A)
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
#  Dataset prep — sliding-window unique prompts of exactly ISL tokens
# ══════════════════════════════════════════════════════════════════════════════
_prepare_dataset() {
  [[ -f "$DATASET_FILE" ]] && { echo "  Reusing dataset: $DATASET_FILE"; return 0; }
  local n=$(( PROMPTS_MULT * MAX_C + 128 ))
  if [[ "$DATASET" == "sharegpt" ]]; then
    echo "  Curating $n natural prompts from ShareGPT (${MIN_ISL}–${MAX_ISL} tok) ..."
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
    # first human turn = the prompt; require a following gpt turn to exist
    if len(msgs) < 2 or msgs[0].get("from") != "human":
        continue
    prompt = msgs[0].get("value", "").strip()
    if not prompt:
        continue
    ln = len(tok.encode(prompt, add_special_tokens=False))
    if min_isl <= ln <= max_isl:
        out.append({"conversations": [{"from": "human", "value": prompt},
                                      {"from": "gpt", "value": "x"}]})
    if len(out) >= n:
        break
json.dump(out, open(out_path, "w"))
import statistics
lens = [len(tok.encode(c["conversations"][0]["value"], add_special_tokens=False)) for c in out]
print(f"  Dataset: {len(out)} natural prompts, median ISL={int(statistics.median(lens))} tok  →  {out_path}")
PYEOF
  else
    echo "  Preparing fixed-ISL dataset (ISL=$ISL, sliding window) ..."
    "$VENV_PYTHON" - "$MODEL" "$BENCH_ROOT/random_text.txt" "$DATASET_FILE" "$ISL" "$n" <<'PYEOF'
import sys, json
from transformers import AutoTokenizer
model, text_path, out_path, isl, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
ids = tok.encode(open(text_path).read(), add_special_tokens=False)
while len(ids) < isl + n:
    ids = ids + ids
data = [{"conversations": [{"from": "human", "value": tok.decode(ids[i:i+isl], skip_special_tokens=True)},
                           {"from": "gpt", "value": "x"}]} for i in range(n)]
json.dump(data, open(out_path, "w"))
print(f"  Dataset: {n} prompts × {isl} input tokens  →  {out_path}")
PYEOF
  fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  Build the server command for a given mode
# ══════════════════════════════════════════════════════════════════════════════
_build_server_cmd() {
  local mode=$1
  SERVER_CMD=(
    "$VENV_PYTHON" -m sglang.launch_server
    --model-path            "$MODEL"
    --tp-size               "$TP"
    --port                  "$SERVER_PORT"
    --host                  "$SERVER_HOST"
    --mem-fraction-static   0.85
    --dtype                 "$DTYPE"
    --attention-backend     triton
    --cuda-graph-max-bs     "$MAX_C"
    --max-running-requests  "$(( MAX_C > 64 ? MAX_C : 64 ))"
    --enable-metrics
    --decode-log-interval   1
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

# ══════════════════════════════════════════════════════════════════════════════
#  Run one full concurrency sweep for a mode
# ══════════════════════════════════════════════════════════════════════════════
_run_mode() {
  local mode=$1
  local mode_dir="$EXPERIMENT_DIR/$mode"
  local summary="$mode_dir/summary.jsonl"
  local server_log="$mode_dir/server.log"
  mkdir -p "$mode_dir"
  > "$summary"

  if ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT}[[:space:]]"; then
    echo "ERROR: port $SERVER_PORT already in use (stale server?)."; exit 1
  fi

  _build_server_cmd "$mode"
  echo ""
  echo "═══════════════════════════════════════════════════════════════════"
  echo "  MODE        : $mode"
  echo "  CMD         : ${SERVER_CMD[*]}"
  echo "  SERVER LOG  : $server_log"
  echo "═══════════════════════════════════════════════════════════════════"

  "${SERVER_CMD[@]}" > "$server_log" 2>&1 &
  local server_pid=$!
  trap 'kill '"$server_pid"' 2>/dev/null || true; wait '"$server_pid"' 2>/dev/null || true' RETURN

  echo "    Waiting for http://$SERVER_HOST:$SERVER_PORT/health ..."
  local start; start=$(date +%s)
  while true; do
    curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" &>/dev/null && { echo "    Server ready ($(( $(date +%s)-start ))s)."; break; }
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "ERROR: server died. Last 40 lines:"; tail -40 "$server_log"; exit 1
    fi
    (( $(date +%s)-start >= 600 )) && { echo "ERROR: server timeout. Last 40 lines:"; tail -40 "$server_log"; exit 1; }
    sleep 5
  done

  for C in "${CONCURRENCY_STEPS[@]}"; do
    local N=$(( PROMPTS_MULT * C ))
    (( N < 1 )) && N=1
    local run_dir="$mode_dir/concurrency=${C}"
    mkdir -p "$run_dir"
    echo ""
    echo "─── $mode  concurrency=$C  num_prompts=$N ───"

    # Record server-log position so we can scope EAGLE3 accept-length to this run.
    local log_lines_before; log_lines_before=$(wc -l < "$server_log" 2>/dev/null || echo 0)

    # ignore_eos is ON (we do NOT pass --disable-ignore-eos) so every request
    # decodes exactly OSL tokens → clean fixed-length throughput curves.
    # --sharegpt-context-len 2048 enforces the EAGLE3 draft's position limit.
    # --seed is fixed so baseline and eagle3 see the SAME prompts at each C.
    "$VENV_PYTHON" -m sglang.bench_serving \
      --backend        sglang \
      --host           "$SERVER_HOST" \
      --port           "$SERVER_PORT" \
      --model          "$MODEL" \
      --dataset-name   sharegpt \
      --dataset-path   "$DATASET_FILE" \
      --sharegpt-output-len "$OSL" \
      --sharegpt-context-len 2048 \
      --num-prompts    "$N" \
      --request-rate   inf \
      --max-concurrency "$C" \
      --seed           1 \
      --output-file    "$run_dir/bench_results.jsonl" \
      --output-details \
      --flush-cache \
      2>&1 | tee "$run_dir/bench.log"

    # Extract EAGLE3 acceptance length from the new server-log lines for this run.
    local al="null"
    if [[ "$mode" == "eagle3" ]]; then
      tail -n +"$(( log_lines_before + 1 ))" "$server_log" \
        | grep -oE "accept len: [0-9.]+" | grep -oE "[0-9.]+$" > "$run_dir/server_al.txt" || true
      if [[ -s "$run_dir/server_al.txt" ]]; then
        al=$(awk '{s+=$1; n++} END{if(n>0) printf "%.4f", s/n; else print "null"}' "$run_dir/server_al.txt")
      fi
    fi

    # Append normalized summary row.
    "$VENV_PYTHON" - "$run_dir/bench_results.jsonl" "$C" "$OSL" "$summary" "$mode" "$al" <<'PYEOF'
import json, sys
bench, C, osl, summary, mode, al = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
try:
    d = json.loads(open(bench).readlines()[-1])
except Exception as e:
    print(f"  [WARN] could not read {bench}: {e}"); sys.exit(0)
d["max_concurrency"] = C
d["mode"] = mode
d["accept_length"] = (None if al in ("null", "") else float(al))
with open(summary, "a") as f:
    f.write(json.dumps(d) + "\n")
tput = d.get("output_throughput", 0)
tpot = d.get("mean_tpot_ms", 0)
tpu  = 1000.0/tpot if tpot > 0 else 0
alstr = f"  AL={al}" if mode == "eagle3" else ""
print(f"  c={C}: {tput:.1f} tok/s/GPU  {tpu:.1f} tok/s/user  TPOT={tpot:.2f}ms{alstr}")
PYEOF
  done

  echo ""
  echo ">>> Stopping $mode server ..."
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  trap - RETURN
  sleep 5
}

# ══════════════════════════════════════════════════════════════════════════════
#  Plot — 2-panel baseline-vs-eagle3 comparison (matches the slide layout)
# ══════════════════════════════════════════════════════════════════════════════
_plot() {
  local pareto_dir="$EXPERIMENT_DIR/pareto"
  mkdir -p "$pareto_dir"
  "$VENV_PYTHON" -c "import matplotlib" &>/dev/null || "$VENV_PYTHON" -m pip install --quiet matplotlib numpy
  EXPERIMENT="$EXPERIMENT" MODEL="$MODEL" ISL="$ISL" OSL="$OSL" TP="$TP" \
  SPEC_NUM_STEPS="$SPEC_NUM_STEPS" EXPERIMENT_DIR="$EXPERIMENT_DIR" PARETO_DIR="$pareto_dir" \
  "$VENV_PYTHON" - <<'PYEOF'
import json, os, glob, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp_dir   = os.environ["EXPERIMENT_DIR"]
pareto    = os.environ["PARETO_DIR"]
model     = os.environ["MODEL"]; isl=os.environ["ISL"]; osl=os.environ["OSL"]; tp=os.environ["TP"]
nsteps    = os.environ["SPEC_NUM_STEPS"]; experiment=os.environ["EXPERIMENT"]

STYLE = {"baseline": ("o-", "steelblue", "Baseline (no spec-dec)"),
         "eagle3":   ("D-", "crimson",   f"EAGLE3 spec-dec (num_steps={nsteps})")}

def load(mode):
    f = os.path.join(exp_dir, mode, "summary.jsonl")
    if not os.path.exists(f): return None
    rows=[]
    for line in open(f):
        line=line.strip()
        if line:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    rows.sort(key=lambda r: r.get("max_concurrency",0))
    return rows or None

data = {m: load(m) for m in ("baseline","eagle3")}
data = {m:r for m,r in data.items() if r}
if not data:
    print("No summaries to plot."); raise SystemExit(0)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"SGLang + EAGLE3 on A100X (scaled-down repro) — {model}\n"
             f"ISL={isl}  OSL={osl}  TP={tp}  |  Llama-3.1-8B", fontsize=12, fontweight="bold")

# Panel 1: per-user throughput (tok/s/user = 1000/TPOT) vs concurrency
ax = axes[0]
for mode, rows in data.items():
    fmt,color,label = STYLE[mode]
    cs   = [r["max_concurrency"] for r in rows]
    tpu  = [1000.0/r["mean_tpot_ms"] if r.get("mean_tpot_ms",0)>0 else 0 for r in rows]
    ax.plot(cs, tpu, fmt, color=color, lw=2, label=label)
    for r,x,y in zip(rows,cs,tpu):
        if mode=="eagle3" and r.get("accept_length"):
            ax.annotate(f"AL={r['accept_length']:.2f}", (x,y), textcoords="offset points",
                        xytext=(4,6), fontsize=7, color=color)
ax.set_xscale("log", base=2)
ax.set_xticks([r["max_concurrency"] for r in next(iter(data.values()))])
ax.set_xticklabels([str(r["max_concurrency"]) for r in next(iter(data.values()))])
ax.set_xlabel("Concurrency (concurrent users)")
ax.set_ylabel("Tokens / s / user   (= 1000 / TPOT)")
ax.set_title("Per-user throughput vs concurrency")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

# Panel 2: system throughput (tok/s/GPU) vs concurrency
ax = axes[1]
for mode, rows in data.items():
    fmt,color,label = STYLE[mode]
    cs   = [r["max_concurrency"] for r in rows]
    tput = [r.get("output_throughput",0) for r in rows]
    ax.plot(cs, tput, fmt, color=color, lw=2, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks([r["max_concurrency"] for r in next(iter(data.values()))])
ax.set_xticklabels([str(r["max_concurrency"]) for r in next(iter(data.values()))])
ax.set_xlabel("Concurrency (concurrent users)")
ax.set_ylabel("Output throughput per GPU  (tokens/s)")
ax.set_title("System throughput (per-GPU) vs concurrency")
ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
out = os.path.join(pareto, "EAGLE3_PARETO.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("Chart saved:", out)

# Console table
print("\n── Summary " + "─"*70)
hdr = f"{'mode':>9} {'conc':>5} {'tok/s/GPU':>10} {'tok/s/user':>11} {'TPOT(ms)':>9} {'TTFT(ms)':>9} {'AL':>6}"
print(hdr); print("─"*len(hdr))
for mode, rows in data.items():
    for r in rows:
        tpot=r.get("mean_tpot_ms",0); tpu=1000.0/tpot if tpot>0 else 0
        al=r.get("accept_length"); als=f"{al:.2f}" if al else "-"
        print(f"{mode:>9} {r['max_concurrency']:>5} {r.get('output_throughput',0):>10.1f} "
              f"{tpu:>11.1f} {tpot:>9.2f} {r.get('mean_ttft_ms',0):>9.1f} {als:>6}")
PYEOF
}

# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  EAGLE3 PARETO SWEEP (scaled-down repro of the SGLang/Kimi slide)"
  echo "  MODEL        : $MODEL"
  echo "  DRAFT        : $DRAFT_MODEL"
  echo "  DTYPE        : $DTYPE"
  if [[ "$DATASET" == "sharegpt" ]]; then
    echo "  DATASET      : ShareGPT (natural prompts ${MIN_ISL}-${MAX_ISL} tok) / OSL=$OSL"
  else
    echo "  DATASET      : fixed sliding-window ISL=$ISL / OSL=$OSL"
  fi
  echo "  CONCURRENCY  : ${CONCURRENCY_STEPS[*]}   (num_prompts = ${PROMPTS_MULT}×C)"
echo "  MODES        : ${MODES[*]}"
echo "  EAGLE3       : num_steps=$SPEC_NUM_STEPS topk=$SPEC_EAGLE_TOPK draft_tokens=$SPEC_NUM_DRAFT_TOKENS"
echo "  EXPERIMENT   : $EXPERIMENT_DIR"
echo "═══════════════════════════════════════════════════════════════════"

if [[ "$PLOT_ONLY" == "1" ]]; then
  _plot
  exit 0
fi

_fix_nvidia_uvm
_pick_free_gpu
_prepare_dataset

for mode in "${MODES[@]}"; do
  [[ "$mode" == "baseline" || "$mode" == "eagle3" ]] || { echo "ERROR: bad MODE '$mode'"; exit 1; }
  _run_mode "$mode"
done

echo ""
echo ">>> Generating comparison chart ..."
_plot

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Done.  Results: $EXPERIMENT_DIR"
echo "  Chart : $EXPERIMENT_DIR/pareto/EAGLE3_PARETO.png"
echo "═══════════════════════════════════════════════════════════════════"
