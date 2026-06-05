#!/usr/bin/env bash
# SGLang stress-test sweep — mirrors the vLLM Pareto benchmark exactly.
#
# Matching parameters (bench_hparams.json / serve_hparams.json):
#   Model         : Llama-3.1-8B-Instruct
#   Dataset       : ShareGPT V4.3 (realistic variable-length inputs)
#   OSL           : 1000 tokens (fixed via --sharegpt-output-len + --disable-ignore-eos)
#   num_prompts   : PROMPTS_PER_CONCURRENCY_MULT × C  (3×C by default)
#   TP            : 1
#   max_running   : 256   (≈ vLLM max_num_seqs)
#   mem_fraction  : 0.90  (≈ vLLM gpu_memory_utilization)
#   concurrency   : 1 2 4 8 16 32 64 128  (8 doubling steps = --workload-iters 8)
#
# Run:
#   bash sglang-bench/run_benchmark.sh
# Override any knob:
#   MODEL=... EXPERIMENT=... bash sglang-bench/run_benchmark.sh
#
# GPU routing: GPU 0 is reserved. CUDA_VISIBLE_DEVICES=1 pins everything
# (server + bench client) to physical GPU 1.  Override with:
#   CUDA_VISIBLE_DEVICES=0,1 bash sglang-bench/run_benchmark.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SGLANG_ENV="$ROOT/sglang-env"
SGLANG_BENCH="$ROOT/sglang-bench"

# ── Activate virtualenv ──────────────────────────────────────────────────────
if [[ ! -x "$SGLANG_ENV/bin/python3" ]]; then
  echo "ERROR: sglang-env not found at $SGLANG_ENV"
  echo "  Expected a pre-built venv with sglang installed."
  exit 1
fi
source "$SGLANG_ENV/bin/activate"

# ── CUDA library paths ───────────────────────────────────────────────────────
source "$SGLANG_BENCH/env.sh"

# ── GPU routing ─────────────────────────────────────────────────────────────
# GPU 0 is occupied. Pin all CUDA work (server + bench client) to GPU 1.
# CUDA_VISIBLE_DEVICES remaps device indices so sglang sees only one GPU
# and uses it as "cuda:0" internally — no other flag needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="

# ── Configurable knobs ───────────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
EXPERIMENT="${EXPERIMENT:-llama8b_sharegpt_osl1k_tp1_sglang}"
OUTPUT_DIR="${OUTPUT_DIR:-$SGLANG_BENCH/results}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_HOST="${SGLANG_HOST:-127.0.0.1}"

# Dataset — ShareGPT provides realistic variable-length inputs; OSL is enforced
# at request time via --sharegpt-output-len + --disable-ignore-eos.
ISL="${ISL:-1000}"    # informational label only; ShareGPT provides natural input lengths
OSL="${OSL:-1000}"    # enforced on every request via --sharegpt-output-len + --disable-ignore-eos
DATASET_FILE="${DATASET_FILE:-$ROOT/datasets/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
NUM_PROMPTS=1024      # global default; overridden per-step when PROMPTS_PER_CONCURRENCY_MULT is set
# Scale samples with concurrency: NUM_PROMPTS = PROMPTS_PER_CONCURRENCY_MULT × C
# Set to 0 to disable scaling and always use NUM_PROMPTS above.
PROMPTS_PER_CONCURRENCY_MULT="${PROMPTS_PER_CONCURRENCY_MULT:-3}"

# Mirrors serve_hparams.json
TP=1
MAX_RUNNING_REQUESTS=256   # ≈ vLLM max_num_seqs
MEM_FRACTION=0.90          # ≈ vLLM gpu_memory_utilization

# 8 doubling steps ↔ vLLM --workload-iters 8
# Geometric sweep spans the full concurrency range; useful for finding the
# throughput/latency knee vs vLLM's linear sweep.
CONCURRENCY_STEPS=(1 2 4 8 16 32 64 128)
# ────────────────────────────────────────────────────────────────────────────

# ── nvidia-uvm device-file preflight ────────────────────────────────────────
_fix_nvidia_uvm() {
  local registered_major
  registered_major=$(awk '/nvidia-uvm/{print $1}' /proc/devices 2>/dev/null)
  [[ -z "$registered_major" ]] && return
  local current_major
  current_major=$(printf '%d' "0x$(stat -c '%t' /dev/nvidia-uvm 2>/dev/null)" 2>/dev/null || echo "")
  if [[ "$current_major" != "$registered_major" ]]; then
    echo "WARNING: /dev/nvidia-uvm major mismatch (device=$current_major, module=$registered_major)."
    echo "  Attempting auto-fix (requires sudo)..."
    if sudo rm -f /dev/nvidia-uvm /dev/nvidia-uvm-tools \
       && sudo mknod /dev/nvidia-uvm c "$registered_major" 0 \
       && sudo mknod /dev/nvidia-uvm-tools c "$registered_major" 1 \
       && sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools; then
      echo "  Fixed: /dev/nvidia-uvm now uses major $registered_major."
    else
      echo "ERROR: Could not fix /dev/nvidia-uvm. Run manually:"
      echo "  sudo rm /dev/nvidia-uvm /dev/nvidia-uvm-tools"
      echo "  sudo mknod /dev/nvidia-uvm c $registered_major 0"
      echo "  sudo mknod /dev/nvidia-uvm-tools c $registered_major 1"
      echo "  sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools"
      exit 1
    fi
  fi
}
_fix_nvidia_uvm

# ── CUDA sanity check ────────────────────────────────────────────────────────
CUDA_OK=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
if [[ "$CUDA_OK" != "yes" ]]; then
  echo "ERROR: PyTorch cannot initialise CUDA (torch.cuda.is_available() == False)."
  echo "  Try resetting nvidia_uvm:"
  echo "    UVM_NEW=\$(awk '/nvidia-uvm/{print \$1}' /proc/devices)"
  echo "    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
  echo "    UVM_NEW=\$(awk '/nvidia-uvm/{print \$1}' /proc/devices)"
  echo "    sudo rm /dev/nvidia-uvm /dev/nvidia-uvm-tools"
  echo "    sudo mknod /dev/nvidia-uvm c \$UVM_NEW 0"
  echo "    sudo mknod /dev/nvidia-uvm-tools c \$UVM_NEW 1"
  echo "    sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools"
  exit 1
fi

# ── HuggingFace auth ─────────────────────────────────────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HuggingFace auth required for Llama 8B (gated model)."
  echo "  1. Accept license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
  echo "  2. hf auth login   OR   export HF_TOKEN=hf_..."
  echo "  3. Re-run: bash sglang-bench/run_benchmark.sh"
  exit 1
fi
export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

if [[ ! -f "$DATASET_FILE" ]]; then
  echo "ERROR: Dataset not found: $DATASET_FILE"
  echo "  Download it first:"
  echo "    python3 -c \"from huggingface_hub import hf_hub_download; hf_hub_download('Aeala/ShareGPT_Vicuna_unfiltered', 'ShareGPT_V4.3_unfiltered_cleaned_split.json', repo_type='dataset', local_dir='datasets')\""
  exit 1
fi

echo "=== Model      : $MODEL ==="
echo "=== Experiment : $EXPERIMENT ==="
echo "=== Dataset    : $DATASET_FILE ==="
echo "=== OSL        : $OSL tokens (fixed via --sharegpt-output-len + --disable-ignore-eos) ==="
echo "=== GPU        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (physical GPU 1, GPU 0 avoided) ==="
echo "=== sglang    : $(python3 -c 'import sglang; print(sglang.__version__)') ==="
echo "=== torch     : $(python3 -c 'import torch; print(torch.__version__, "| cuda", torch.version.cuda)') ==="
# Show only the GPU(s) visible to this process
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader \
  | awk -F',' -v devs="$CUDA_VISIBLE_DEVICES" 'BEGIN{n=split(devs,a,","); for(i=1;i<=n;i++) vis[a[i]]=1} {gsub(/^ /,"",$1); if($1 in vis) print "  GPU"$0}'

# ── Attention-backend selection ──────────────────────────────────────────────
# flashinfer-python (JIT version) is installed but requires CUDA 12.x nvcc to
# JIT-compile its kernels. The only nvcc available on this machine ships with
# the vLLM .venv (nvidia/cu13/bin/nvcc — CUDA 13), which is incompatible with
# the torch 2.7.1+cu128 headers in sglang-env (fails with C++ template errors
# in ATen/core/List_inl.h on GCC 13+).  Until a CUDA 12.x nvcc is available,
# we use the Triton attention backend which needs no JIT compilation and is
# well-optimised for A100 (sm_80).
ATTN_BACKEND_FLAG="--attention-backend triton"
echo "INFO: Using --attention-backend triton (flashinfer JIT incompatible with cu13 nvcc on this host)."

EXPERIMENT_DIR="$OUTPUT_DIR/$EXPERIMENT"
mkdir -p "$EXPERIMENT_DIR"

# ── Launch SGLang server ─────────────────────────────────────────────────────
SERVER_LOG="$EXPERIMENT_DIR/server.log"
echo ""
echo "=== Starting SGLang server ==="
echo "    tp=$TP  mem-fraction-static=$MEM_FRACTION  max-running-requests=$MAX_RUNNING_REQUESTS"
# shellcheck disable=SC2086
python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size "$TP" \
  --port "$SGLANG_PORT" \
  --host "$SGLANG_HOST" \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --dtype auto \
  $ATTN_BACKEND_FLAG \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "    PID=$SERVER_PID  log=$SERVER_LOG"

# Kill server on any exit (clean, error, or Ctrl-C)
trap 'echo ""; echo "Stopping SGLang server (PID $SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# ── Wait for server readiness ────────────────────────────────────────────────
echo "Waiting for http://$SGLANG_HOST:$SGLANG_PORT/health ..."
TIMEOUT=300
ELAPSED=0
while true; do
  if curl -sf "http://$SGLANG_HOST:$SGLANG_PORT/health" &>/dev/null; then
    echo "Server ready (${ELAPSED}s)."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: SGLang server process (PID $SERVER_PID) died. Last 30 lines:"
    tail -30 "$SERVER_LOG"
    exit 1
  fi
  if (( ELAPSED >= TIMEOUT )); then
    echo "ERROR: Server did not become ready within ${TIMEOUT}s. Last 30 lines:"
    tail -30 "$SERVER_LOG"
    exit 1
  fi
  sleep 5
  (( ELAPSED += 5 ))
done

# ── Concurrency sweep ────────────────────────────────────────────────────────
echo ""
echo "=== Concurrency sweep: ${CONCURRENCY_STEPS[*]} ==="
if (( PROMPTS_PER_CONCURRENCY_MULT > 0 )); then
  echo "    OSL=$OSL (--sharegpt-output-len, fixed)  num_prompts=${PROMPTS_PER_CONCURRENCY_MULT}×C (scaled)  request_rate=inf"
else
  echo "    OSL=$OSL (--sharegpt-output-len, fixed)  num_prompts=$NUM_PROMPTS (fixed)  request_rate=inf"
fi
echo ""

SUMMARY_FILE="$EXPERIMENT_DIR/summary.jsonl"
> "$SUMMARY_FILE"

for C in "${CONCURRENCY_STEPS[@]}"; do
  RUN_DIR="$EXPERIMENT_DIR/concurrency=$C"
  mkdir -p "$RUN_DIR"
  BENCH_JSONL="$RUN_DIR/bench_results.jsonl"

  # Scale prompts with concurrency so low-concurrency steps don't take hours.
  if (( PROMPTS_PER_CONCURRENCY_MULT > 0 )); then
    STEP_PROMPTS=$(( PROMPTS_PER_CONCURRENCY_MULT * C ))
  else
    STEP_PROMPTS=$NUM_PROMPTS
  fi

  echo "─── concurrency=$C (num_prompts=$STEP_PROMPTS, mult=${PROMPTS_PER_CONCURRENCY_MULT}×C) ───"
  python3 -m sglang.bench_serving \
    --backend sglang \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PORT" \
    --model "$MODEL" \
    --dataset-name sharegpt \
    --dataset-path "$DATASET_FILE" \
    --sharegpt-output-len "$OSL" \
    --num-prompts "$STEP_PROMPTS" \
    --request-rate inf \
    --max-concurrency "$C" \
    --output-file "$BENCH_JSONL" \
    --output-details \
    --disable-ignore-eos \
    --flush-cache \
    2>&1 | tee "$RUN_DIR/bench.log"

  # Append the last (summary) line of this run to the consolidated JSONL
  if [[ -f "$BENCH_JSONL" ]]; then
    tail -1 "$BENCH_JSONL" >> "$SUMMARY_FILE"
  fi
  echo ""
done

echo "=== Sweep complete ==="
echo "    Summary: $SUMMARY_FILE"

# ── Pareto / summary charts ──────────────────────────────────────────────────
echo ""
echo "=== Generating Pareto charts ==="
PARETO_DIR="$EXPERIMENT_DIR/pareto"
mkdir -p "$PARETO_DIR"

# Install charting dependencies into sglang-env if missing
_missing_pkgs=()
python3 -c "import matplotlib" &>/dev/null 2>&1 || _missing_pkgs+=(matplotlib)
python3 -c "import numpy"      &>/dev/null 2>&1 || _missing_pkgs+=(numpy)
if (( ${#_missing_pkgs[@]} )); then
  echo "Installing missing packages into sglang-env: ${_missing_pkgs[*]} ..."
  # Use `python3 -m pip` to avoid ~/.local/bin/pip (Python 3.8) shadowing the
  # venv pip when env.sh prepends ~/.local/bin to PATH.
  python3 -m pip install --quiet "${_missing_pkgs[@]}"
fi
unset _missing_pkgs

export SUMMARY_FILE PARETO_DIR EXPERIMENT MODEL ISL OSL NUM_PROMPTS TP DATASET_FILE
python3 - <<'PYEOF'
import json, os, sys, pathlib

summary_path = os.environ["SUMMARY_FILE"]
pareto_dir   = os.environ["PARETO_DIR"]
experiment   = os.environ["EXPERIMENT"]
model        = os.environ["MODEL"]
isl          = os.environ["ISL"]
osl          = os.environ["OSL"]
num_p        = os.environ["NUM_PROMPTS"]
tp           = os.environ["TP"]
dataset_file = os.path.basename(os.environ.get("DATASET_FILE", "ShareGPT"))

if not pathlib.Path(summary_path).exists():
    print("No summary file found; skipping chart.")
    sys.exit(0)

rows = []
with open(summary_path) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

if not rows:
    print("Summary file is empty; skipping chart.")
    sys.exit(0)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError:
    print("matplotlib not available; skipping chart.")
    sys.exit(0)

rows.sort(key=lambda r: r.get("max_concurrency", 0))

concurrencies = [r.get("max_concurrency",        0)         for r in rows]
output_tput   = [r.get("output_throughput",       0)         for r in rows]  # tok/s/GPU (TP=1)
mean_e2el_s   = [r.get("mean_e2el_ms",            0) / 1000  for r in rows]  # s
p99_e2el_s    = [r.get("p99_e2el_ms",             0) / 1000  for r in rows]
mean_ttft_ms  = [r.get("mean_ttft_ms",            0)         for r in rows]
p99_ttft_ms   = [r.get("p99_ttft_ms",             0)         for r in rows]
mean_itl_ms   = [r.get("mean_itl_ms",             0)         for r in rows]
p99_itl_ms    = [r.get("p99_itl_ms",              0)         for r in rows]
mean_tpot_ms  = [r.get("mean_tpot_ms",            0)         for r in rows]

# ── Key Pareto metrics ───────────────────────────────────────────────────────
# tok/s/GPU  = output_throughput            (sglang JSONL field, line 2694)
#              TP=1 so 1 GPU → equals total GPU throughput
#
# tok/s/user = 1000 / mean_tpot_ms         (sglang JSONL field, line 2705)
#              TPOT = time between consecutive output tokens experienced by one
#              user.  Its reciprocal is the per-user generation speed.
#              Matches vLLM plot_pareto formula: verified at c=1 with vLLM data
#              (output_throughput=37.4, mean_tpot_ms=26.4 → 1000/26.4 = 37.9 ≈ x-axis value)
tput_per_user = [1000.0 / t if t > 0 else 0 for t in mean_tpot_ms]

# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(
    f"SGLang Stress Test — {experiment}\n"
    f"{model}  |  {dataset_file}  |  OSL={osl}  |  TP={tp}  |  request_rate=inf",
    fontsize=12, fontweight="bold",
)

_xticks   = concurrencies
_xticklbl = [str(c) for c in concurrencies]

def _xlog(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(_xticks)
    ax.set_xticklabels(_xticklbl)
    ax.set_xlabel("Max Concurrency")

# ── 1. tok/s/GPU vs concurrency ──────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(concurrencies, output_tput, "o-", color="steelblue", lw=2)
_xlog(ax)
ax.set_ylabel("Tokens/s/GPU")
ax.set_title("GPU Throughput vs Concurrency")
ax.grid(True, alpha=0.3)

# ── 2. tok/s/user vs concurrency ─────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(concurrencies, tput_per_user, "o-", color="slateblue", lw=2)
_xlog(ax)
ax.set_ylabel("Tokens/s/user  (= 1000 / mean_tpot_ms)")
ax.set_title("Per-User Generation Speed vs Concurrency")
ax.grid(True, alpha=0.3)

# ── 3. Pareto: tok/s/user (x) vs tok/s/GPU (y) — matches vLLM plot_pareto ───
ax = axes[0, 2]
norm = mcolors.LogNorm(vmin=max(1, min(concurrencies)), vmax=max(concurrencies))
sc = ax.scatter(tput_per_user, output_tput,
                c=concurrencies, cmap="plasma", norm=norm, s=90, zorder=5)
for c, x, y in zip(concurrencies, tput_per_user, output_tput):
    ax.annotate(
        f"max_concurrency={c}\ntensor_parallel_size={tp}",
        (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7,
    )
ax.plot(tput_per_user, output_tput, "--", color="steelblue", alpha=0.55, label="Pareto frontier")
ax.scatter([], [], color="gray", s=50, label="All runs")  # legend proxy
ax.legend(fontsize=8)
plt.colorbar(sc, ax=ax, label="Concurrency")
ax.set_xlabel("Tokens/s/user  ↑ better")
ax.set_ylabel("Tokens/s/GPU  ↑ better")
ax.set_title("Pareto: Tokens/s/user vs Tokens/s/GPU")
ax.grid(True, alpha=0.3, linestyle="--")

# ── 4. TTFT vs concurrency ───────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(concurrencies, mean_ttft_ms, "o-",  color="darkorange", lw=2, label="mean")
ax.plot(concurrencies, p99_ttft_ms,  "s--", color="darkorange", lw=1.5, alpha=0.65, label="p99")
_xlog(ax)
ax.set_ylabel("TTFT (ms)")
ax.set_title("Time-to-First-Token vs Concurrency")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 5. E2E latency vs concurrency ────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(concurrencies, mean_e2el_s, "o-",  color="tomato", lw=2, label="mean")
ax.plot(concurrencies, p99_e2el_s,  "s--", color="tomato", lw=1.5, alpha=0.65, label="p99")
_xlog(ax)
ax.set_ylabel("E2E Latency (s)")
ax.set_title("End-to-End Latency vs Concurrency")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── 6. ITL & TPOT vs concurrency ─────────────────────────────────────────────
ax = axes[1, 2]
ax.plot(concurrencies, mean_itl_ms,  "o-",  color="mediumseagreen", lw=2,   label="ITL mean")
ax.plot(concurrencies, p99_itl_ms,   "s--", color="mediumseagreen", lw=1.5, alpha=0.65, label="ITL p99")
ax.plot(concurrencies, mean_tpot_ms, "^:",  color="cadetblue",      lw=1.5, label="TPOT mean")
_xlog(ax)
ax.set_ylabel("Latency (ms)")
ax.set_title("ITL & TPOT vs Concurrency")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(pareto_dir, "PARETO.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ── Console summary table ────────────────────────────────────────────────────
print()
print("── Results Summary ───────────────────────────────────────────────────────────────────────────────")
header = f"{'Concur':>7}  {'tok/s/GPU':>10}  {'tok/s/user':>11}  {'MeanE2EL(s)':>12}  {'p99E2EL(s)':>11}  {'TTFT_mean(ms)':>14}  {'ITL_mean(ms)':>13}"
print(header)
print("─" * len(header))
for c, tg, tu, e, e99, ttft, itl in zip(concurrencies, output_tput, tput_per_user, mean_e2el_s, p99_e2el_s, mean_ttft_ms, mean_itl_ms):
    print(f"{c:>7}  {tg:>10.1f}  {tu:>11.2f}  {e:>12.3f}  {e99:>11.3f}  {ttft:>14.1f}  {itl:>13.2f}")
PYEOF

echo ""
echo "Done."
echo "  Results : $EXPERIMENT_DIR"
echo "  Chart   : $PARETO_DIR/PARETO.png"
ls -la "$PARETO_DIR/" 2>/dev/null || true
