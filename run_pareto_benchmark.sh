#!/usr/bin/env bash
# Pareto throughput/latency sweep — vLLM or sglang
#
# Usage:
#   ENGINE=vllm   bash run_pareto_benchmark.sh
#   ENGINE=sglang bash run_pareto_benchmark.sh
#
# Every request uses exactly ISL input tokens (sliding window over random_text.txt)
# and exactly OSL output tokens (--sharegpt-output-len + --ignore-eos).
# The sliding window gives each prompt a distinct token sequence, preventing
# prefix-cache hits from inflating throughput numbers.
#
# Outputs (under pareto-results/<experiment>/):
#   concurrency=<C>/bench.log       raw benchmark log per step
#   summary.jsonl                   one JSON row per concurrency level (unified format)
#   pareto/PARETO.png               6-panel Pareto chart
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ══════════════════════════════════════════════════════════════════════════════
# ── Shared config ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
ENGINE="${ENGINE:-sglang}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
ISL="${ISL:-1000}"   # exact input tokens per prompt (enforced via tokenized dataset)
OSL="${OSL:-1000}"   # exact output tokens per request (enforced via --sharegpt-output-len + --ignore-eos)
TP="${TP:-1}"

IFS=' ' read -r -a CONCURRENCY_STEPS <<< "${CONCURRENCY_STEPS:-1 2 4 8 16 32 64 128}"
# num_prompts = PROMPTS_MULT × C  (set PROMPTS_MULT=0 to use FIXED_PROMPTS instead)
PROMPTS_MULT="${PROMPTS_MULT:-3}"
FIXED_PROMPTS="${FIXED_PROMPTS:-1024}"

EXPERIMENT="${EXPERIMENT:-${ENGINE}_isl${ISL}_osl${OSL}_tp${TP}}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/pareto-results}"
EXPERIMENT_DIR="$OUTPUT_DIR/$EXPERIMENT"
SUMMARY_FILE="$EXPERIMENT_DIR/summary.jsonl"
DATASET_DIR="$ROOT/datasets"
DATASET_FILE="$DATASET_DIR/fixed_isl${ISL}.json"

mkdir -p "$EXPERIMENT_DIR" "$DATASET_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# ── Engine-specific: venv, server command, _run_bench, _extract_metrics ───────
# ══════════════════════════════════════════════════════════════════════════════
if [[ "$ENGINE" == "vllm" ]]; then
  source "$ROOT/.venv/bin/activate"
  [[ -f "$ROOT/vllm-bench/env.sh" ]] && source "$ROOT/vllm-bench/env.sh"
  SERVER_PORT="${SERVER_PORT:-8000}"
  SERVER_HOST="localhost"
  SERVER_CMD=(
    vllm serve "$MODEL"
    --dtype auto
    --attention-backend TRITON_ATTN
    --tensor-parallel-size "$TP"
    --max-num-seqs 256
    --max-num-batched-tokens 8192
    --gpu-memory-utilization 0.90
    --port "$SERVER_PORT"
  )

  _run_bench() {
    local C=$1 OUT_DIR=$2 N=$3
    # Flush prefix cache between steps for a clean KV state
    curl -sf -X POST "http://${SERVER_HOST}:${SERVER_PORT}/reset_prefix_cache" &>/dev/null || true
    vllm bench serve \
      --model          "$MODEL" \
      --backend        vllm \
      --endpoint       /v1/completions \
      --dataset-name   sharegpt \
      --dataset-path   "$DATASET_FILE" \
      --sharegpt-output-len "$OSL" \
      --num-prompts    "$N" \
      --max-concurrency "$C" \
      --host           "$SERVER_HOST" \
      --port           "$SERVER_PORT" \
      --ignore-eos \
      --save-result \
      --result-dir     "$OUT_DIR" \
      2>&1 | tee "$OUT_DIR/bench.log"
  }

  _extract_metrics() {
    local OUT_DIR=$1 C=$2
    # vllm bench serve writes one *.json result file per run
    local result_json
    result_json=$(ls -t "$OUT_DIR/"*.json 2>/dev/null | grep -v bench_results | head -1 || true)
    if [[ -z "$result_json" ]]; then
      echo "  [WARN] No result JSON found in $OUT_DIR"; return
    fi
    # Normalize to the common summary format and append to summary.jsonl
    python3 - "$result_json" "$C" "$OSL" "$SUMMARY_FILE" <<'PYEOF'
import json, sys
path, C, osl, summary = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = json.load(open(path))
d["max_concurrency"] = C
# vLLM field aliases (ITL ≈ TPOT for decode-only batches)
if "mean_itl_ms"  not in d: d["mean_itl_ms"]  = d.get("mean_tpot_ms",  0)
if "p99_itl_ms"   not in d: d["p99_itl_ms"]   = d.get("p99_tpot_ms",   0)
# Approximate E2E latency if not reported: TTFT + OSL * TPOT
if "mean_e2el_ms" not in d:
    d["mean_e2el_ms"] = d.get("mean_ttft_ms", 0) + osl * d.get("mean_tpot_ms", 0)
if "p99_e2el_ms"  not in d:
    d["p99_e2el_ms"]  = d.get("p99_ttft_ms",  0) + osl * d.get("p99_tpot_ms",  0)
with open(summary, "a") as f:
    f.write(json.dumps(d) + "\n")
tput = d.get("output_throughput", 0)
tpot = d.get("mean_tpot_ms", 0)
ttft = d.get("mean_ttft_ms", 0)
print(f"  c={C}: {tput:.1f} tok/s/GPU  TPOT={tpot:.1f}ms  TTFT={ttft:.1f}ms")
PYEOF
  }

elif [[ "$ENGINE" == "sglang" ]]; then
  source "$ROOT/sglang-env/bin/activate"
  source "$ROOT/sglang-bench/env.sh"
  SERVER_PORT="${SERVER_PORT:-30000}"
  SERVER_HOST="127.0.0.1"
  SERVER_CMD=(
    python3 -m sglang.launch_server
    --model-path            "$MODEL"
    --tp-size               "$TP"
    --port                  "$SERVER_PORT"
    --host                  "$SERVER_HOST"
    --mem-fraction-static   0.90
    --max-running-requests  256
    --dtype                 auto
    --attention-backend     triton
  )

  _run_bench() {
    local C=$1 OUT_DIR=$2 N=$3
    python3 -m sglang.bench_serving \
      --backend        sglang \
      --host           "$SERVER_HOST" \
      --port           "$SERVER_PORT" \
      --model          "$MODEL" \
      --dataset-name   sharegpt \
      --dataset-path   "$DATASET_FILE" \
      --sharegpt-output-len "$OSL" \
      --num-prompts    "$N" \
      --request-rate   inf \
      --max-concurrency "$C" \
      --output-file    "$OUT_DIR/bench_results.jsonl" \
      --output-details \
      --disable-ignore-eos \
      --flush-cache \
      2>&1 | tee "$OUT_DIR/bench.log"
  }

  _extract_metrics() {
    local OUT_DIR=$1 C=$2
    local bench_jsonl="$OUT_DIR/bench_results.jsonl"
    if [[ ! -f "$bench_jsonl" ]]; then
      echo "  [WARN] No bench_results.jsonl found in $OUT_DIR"; return
    fi
    # sglang's JSONL already has max_concurrency and all required fields
    tail -1 "$bench_jsonl" >> "$SUMMARY_FILE"
    python3 -c "
import json
d = json.loads(open('$bench_jsonl').readlines()[-1])
tput = d.get('output_throughput', 0)
tpot = d.get('mean_tpot_ms', 0)
ttft = d.get('mean_ttft_ms', 0)
print(f'  c=$C: {tput:.1f} tok/s/GPU  TPOT={tpot:.1f}ms  TTFT={ttft:.1f}ms')
" 2>/dev/null || true
  }

else
  echo "ERROR: ENGINE must be 'vllm' or 'sglang' (got: '$ENGINE')"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# ── Shared: GPU / UVM preflight ───────────────────────────────────────────────
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

_fix_nvidia_uvm
_pick_free_gpu

# ── HuggingFace auth ──────────────────────────────────────────────────────────
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
if [[ -z "${HF_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ══════════════════════════════════════════════════════════════════════════════
# ── Shared: Dataset preparation ───────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# Tokenizes random_text.txt via a sliding window to produce N prompts, each
# exactly ISL tokens.  Different start offsets → different token sequences →
# no prefix-cache hits between requests within or across concurrency steps.
_prepare_dataset() {
  [[ -f "$DATASET_FILE" ]] && { echo "  Reusing dataset: $DATASET_FILE"; return 0; }
  echo "  Preparing fixed-ISL dataset (ISL=$ISL tokens, sliding window) ..."

  # Generate enough prompts for the largest concurrency step, plus a warmup buffer
  local max_c="${CONCURRENCY_STEPS[-1]}"
  local n
  if (( PROMPTS_MULT > 0 )); then
    n=$(( PROMPTS_MULT * max_c + 128 ))
  else
    n=$(( FIXED_PROMPTS + 128 ))
  fi

  python3 - "$MODEL" "$ROOT/random_text.txt" "$DATASET_FILE" "$ISL" "$n" <<'PYEOF'
import sys, json
from transformers import AutoTokenizer

model, text_path, out_path, isl, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])

tok     = AutoTokenizer.from_pretrained(model, local_files_only=True)
text    = open(text_path).read()
all_ids = tok.encode(text, add_special_tokens=False)

# Tile if the text file doesn't have enough tokens for the sliding window
while len(all_ids) < isl + n:
    all_ids = all_ids + all_ids

data = []
for i in range(n):
    prompt = tok.decode(all_ids[i : i + isl], skip_special_tokens=True)
    data.append({
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt",   "value": "x"},   # placeholder; --sharegpt-output-len overrides
        ]
    })

json.dump(data, open(out_path, "w"))
print(f"  Dataset: {n} prompts × {isl} input tokens  →  {out_path}")
PYEOF
}

# ══════════════════════════════════════════════════════════════════════════════
# ── Banner ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  ENGINE      : $ENGINE"
echo "  MODEL       : $MODEL"
echo "  ISL / OSL   : $ISL / $OSL tokens  (exact, fixed)"
echo "  GPU         : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}"
echo "  CONCURRENCY : ${CONCURRENCY_STEPS[*]}"
if (( PROMPTS_MULT > 0 )); then
  echo "  PROMPTS     : ${PROMPTS_MULT} × C per step"
else
  echo "  PROMPTS     : $FIXED_PROMPTS per step (fixed)"
fi
echo "  EXPERIMENT  : $EXPERIMENT"
echo "  OUTPUT      : $EXPERIMENT_DIR"
echo "═══════════════════════════════════════════════════════════════════"

_prepare_dataset

> "$SUMMARY_FILE"   # truncate / create empty summary

# ══════════════════════════════════════════════════════════════════════════════
# ── Start server ──────────────────────────════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
if ss -tlnp 2>/dev/null | grep -q ":${SERVER_PORT}[[:space:]]"; then
  echo "ERROR: port $SERVER_PORT is already in use."
  echo "  Kill it:  kill \$(ss -tlnp | awk '/:${SERVER_PORT}/{gsub(/.*pid=/,\"\");gsub(/,.*/,\"\");print}')"
  exit 1
fi

SERVER_LOG="$EXPERIMENT_DIR/server.log"
echo ""
echo ">>> Starting $ENGINE server  (log: $SERVER_LOG)"
"${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'echo ""; echo "Stopping $ENGINE server (PID $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

echo "    PID=$SERVER_PID  port=$SERVER_PORT"
echo "    Waiting for http://$SERVER_HOST:$SERVER_PORT/health ..."
TIMEOUT=300; ELAPSED=0
while true; do
  curl -sf "http://$SERVER_HOST:$SERVER_PORT/health" &>/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: $ENGINE server died. Last 30 lines:"
    tail -30 "$SERVER_LOG"; exit 1
  fi
  (( ELAPSED >= TIMEOUT )) && { echo "ERROR: server timeout after ${TIMEOUT}s."; tail -30 "$SERVER_LOG"; exit 1; }
  sleep 5; (( ELAPSED += 5 ))
done
echo "    Server ready (${ELAPSED}s)."

# ══════════════════════════════════════════════════════════════════════════════
# ── Concurrency sweep ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Concurrency sweep: ${CONCURRENCY_STEPS[*]}"

for C in "${CONCURRENCY_STEPS[@]}"; do
  if (( PROMPTS_MULT > 0 )); then
    N=$(( PROMPTS_MULT * C ))
  else
    N=$FIXED_PROMPTS
  fi

  RUN_DIR="$EXPERIMENT_DIR/concurrency=${C}"
  mkdir -p "$RUN_DIR"
  echo ""
  echo "─── concurrency=$C  num_prompts=$N ───"

  _run_bench       "$C" "$RUN_DIR" "$N"
  _extract_metrics "$RUN_DIR" "$C"
done

# ══════════════════════════════════════════════════════════════════════════════
# ── Shutdown server ───────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Shutting down $ENGINE server..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT

# ══════════════════════════════════════════════════════════════════════════════
# ── Pareto chart (shared) ─────────────────────────────────════════════════════
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Generating Pareto charts ..."
PARETO_DIR="$EXPERIMENT_DIR/pareto"
mkdir -p "$PARETO_DIR"

python3 -c "import matplotlib" &>/dev/null || python3 -m pip install --quiet matplotlib numpy

export SUMMARY_FILE PARETO_DIR EXPERIMENT ENGINE MODEL ISL OSL TP
python3 - <<'PYEOF'
import json, os, sys, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

summary_path = os.environ["SUMMARY_FILE"]
pareto_dir   = os.environ["PARETO_DIR"]
experiment   = os.environ["EXPERIMENT"]
engine       = os.environ["ENGINE"]
model        = os.environ["MODEL"]
isl          = os.environ["ISL"]
osl          = os.environ["OSL"]
tp           = os.environ["TP"]

if not pathlib.Path(summary_path).exists():
    print("No summary file; skipping chart."); sys.exit(0)

rows = []
with open(summary_path) as f:
    for line in f:
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass

if not rows:
    print("Summary file is empty; skipping chart."); sys.exit(0)

rows.sort(key=lambda r: r.get("max_concurrency", 0))

def _get(rows, key, scale=1.0):
    return [r.get(key, 0) * scale for r in rows]

concurrencies = [r.get("max_concurrency", 0) for r in rows]
output_tput   = _get(rows, "output_throughput")
mean_e2el_s   = _get(rows, "mean_e2el_ms",  1/1000)
p99_e2el_s    = _get(rows, "p99_e2el_ms",   1/1000)
mean_ttft_ms  = _get(rows, "mean_ttft_ms")
p99_ttft_ms   = _get(rows, "p99_ttft_ms")
mean_itl_ms   = _get(rows, "mean_itl_ms")
p99_itl_ms    = _get(rows, "p99_itl_ms")
mean_tpot_ms  = _get(rows, "mean_tpot_ms")
# Per-user throughput: reciprocal of inter-token latency experienced by one user
tput_per_user = [1000.0 / t if t > 0 else 0 for t in mean_tpot_ms]

color = "steelblue" if engine == "vllm" else "darkorange"

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(
    f"{engine.upper()} Pareto Benchmark — {experiment}\n"
    f"{model}  |  ISL={isl}  OSL={osl}  TP={tp}  |  request_rate=inf",
    fontsize=12, fontweight="bold",
)

def _xlog(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(concurrencies)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.set_xlabel("Max Concurrency")

# 1. System throughput vs concurrency
ax = axes[0, 0]
ax.plot(concurrencies, output_tput, "o-", color=color, lw=2)
_xlog(ax); ax.set_ylabel("Tokens/s/GPU"); ax.set_title("GPU Throughput vs Concurrency")
ax.grid(True, alpha=0.3)

# 2. Per-user throughput vs concurrency
ax = axes[0, 1]
ax.plot(concurrencies, tput_per_user, "o-", color=color, lw=2)
_xlog(ax); ax.set_ylabel("Tokens/s/user  (= 1000 / TPOT_ms)")
ax.set_title("Per-User Generation Speed vs Concurrency"); ax.grid(True, alpha=0.3)

# 3. Pareto frontier: tok/s/user (x) vs tok/s/GPU (y)
ax = axes[0, 2]
norm = mcolors.LogNorm(vmin=max(1, min(concurrencies)), vmax=max(concurrencies))
sc = ax.scatter(tput_per_user, output_tput, c=concurrencies, cmap="plasma", norm=norm, s=90, zorder=5)
for c_val, x, y in zip(concurrencies, tput_per_user, output_tput):
    ax.annotate(f"c={c_val}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
ax.plot(tput_per_user, output_tput, "--", color=color, alpha=0.5)
plt.colorbar(sc, ax=ax, label="Concurrency")
ax.set_xlabel("Tokens/s/user  ↑ better"); ax.set_ylabel("Tokens/s/GPU  ↑ better")
ax.set_title("Pareto: Tokens/s/user vs Tokens/s/GPU"); ax.grid(True, alpha=0.3, linestyle="--")

# 4. TTFT vs concurrency
ax = axes[1, 0]
ax.plot(concurrencies, mean_ttft_ms, "o-",  color="darkorange", lw=2,   label="mean")
ax.plot(concurrencies, p99_ttft_ms,  "s--", color="darkorange", lw=1.5, alpha=0.65, label="p99")
_xlog(ax); ax.set_ylabel("TTFT (ms)"); ax.set_title("Time-to-First-Token vs Concurrency")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 5. End-to-end latency vs concurrency
ax = axes[1, 1]
ax.plot(concurrencies, mean_e2el_s, "o-",  color="tomato", lw=2,   label="mean")
ax.plot(concurrencies, p99_e2el_s,  "s--", color="tomato", lw=1.5, alpha=0.65, label="p99")
_xlog(ax); ax.set_ylabel("E2E Latency (s)"); ax.set_title("End-to-End Latency vs Concurrency")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 6. ITL & TPOT vs concurrency
ax = axes[1, 2]
ax.plot(concurrencies, mean_itl_ms,  "o-",  color="mediumseagreen", lw=2,   label="ITL mean")
ax.plot(concurrencies, p99_itl_ms,   "s--", color="mediumseagreen", lw=1.5, alpha=0.65, label="ITL p99")
ax.plot(concurrencies, mean_tpot_ms, "^:",  color="cadetblue",      lw=1.5, label="TPOT mean")
_xlog(ax); ax.set_ylabel("Latency (ms)"); ax.set_title("ITL & TPOT vs Concurrency")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(pareto_dir, "PARETO.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Chart saved: {out_path}")

# Console summary table
print()
print("── Results Summary " + "─" * 78)
hdr = f"{'Concur':>7}  {'tok/s/GPU':>10}  {'tok/s/user':>11}  {'E2EL_mean(s)':>13}  {'TTFT_mean(ms)':>14}  {'ITL_mean(ms)':>13}"
print(hdr)
print("─" * len(hdr))
for c, tg, tu, e, ttft, itl in zip(concurrencies, output_tput, tput_per_user, mean_e2el_s, mean_ttft_ms, mean_itl_ms):
    print(f"{c:>7}  {tg:>10.1f}  {tu:>11.2f}  {e:>13.3f}  {ttft:>14.1f}  {itl:>13.2f}")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  Done."
echo "  Results : $EXPERIMENT_DIR"
echo "  Chart   : $PARETO_DIR/PARETO.png"
echo "═══════════════════════════════════════════════════════════════════"
