#!/usr/bin/env bash
# Serve Llama 8B with vLLM, stress-test across concurrency levels, draw Pareto chart
# Dataset: ShareGPT V4.3 (realistic variable-length conversations)
# OSL enforced via --sharegpt-output-len; input lengths come naturally from dataset.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null && [[ -x "$HOME/.local/bin/uv" ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found. Install with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  exit 1
fi
# Large NVIDIA wheels (cudnn ~600MB) need a longer download timeout
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
EXPERIMENT="${EXPERIMENT:-llama8b_sharegpt_osl1k_tp1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/vllm-bench/results}"
TP="${TP:-1}"
SERVER_PORT="${SERVER_PORT:-8000}"
ISL="${ISL:-1000}"   # informational label only; ShareGPT provides natural input lengths
OSL="${OSL:-1000}"   # enforced on every request via --sharegpt-output-len + --ignore-eos
DATASET_FILE="${DATASET_FILE:-$ROOT/datasets/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
# Each concurrency level runs for ROUNDS × concurrency prompts (2-3 full waves)
ROUNDS="${ROUNDS:-3}"
# Space-separated list of max-concurrency values to sweep
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8 16 32 64 128}"

EXPERIMENT_DIR="$OUTPUT_DIR/$EXPERIMENT"
mkdir -p "$EXPERIMENT_DIR"

# ── venv ──────────────────────────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install vllm matplotlib --torch-backend=cu126
else
  source .venv/bin/activate
fi

if [[ -f "$ROOT/vllm-bench/env.sh" ]]; then
  source "$ROOT/vllm-bench/env.sh"
fi

# ── GPU ops: fix stale /dev/nvidia-uvm after crash ───────────────────────────
_fix_nvidia_uvm() {
  local registered_major
  registered_major=$(awk '/nvidia-uvm/{print $1}' /proc/devices 2>/dev/null)
  [[ -z "$registered_major" ]] && return  # module not loaded; nothing to fix
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
      echo "Or simply reboot the machine."
      exit 1
    fi
  fi
}
_fix_nvidia_uvm

# ── vLLM install / CUDA version checks ───────────────────────────────────────
if ! command -v vllm &>/dev/null; then
  echo "Installing vLLM (CUDA 12.6)..."
  uv pip install vllm matplotlib --torch-backend=cu126
fi

# Reinstall if PyTorch was built for an unsupported CUDA (e.g. cu129 on driver 570)
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
if [[ "$TORCH_CUDA" == "12.9" ]]; then
  echo "Reinstalling vLLM: PyTorch cu129 incompatible with driver, switching to cu126..."
  echo "(Large downloads — UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT}s)"
  uv pip install vllm --torch-backend=cu126 --reinstall
fi

DRIVER_CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "unknown")
if [[ -n "$DRIVER_CUDA" && "$TORCH_CUDA" != "unknown" ]]; then
  DRIVER_MAJOR=${DRIVER_CUDA%%.*}; DRIVER_MINOR=${DRIVER_CUDA#*.}; DRIVER_MINOR=${DRIVER_MINOR%%.*}
  TORCH_MAJOR=${TORCH_CUDA%%.*}; TORCH_MINOR=${TORCH_CUDA#*.}; TORCH_MINOR=${TORCH_MINOR%%.*}
  if (( DRIVER_MAJOR < TORCH_MAJOR || (DRIVER_MAJOR == TORCH_MAJOR && DRIVER_MINOR < TORCH_MINOR) )); then
    echo "ERROR: CUDA driver/runtime mismatch."
    echo "  nvidia-smi reports max CUDA $DRIVER_CUDA (driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1))"
    echo "  PyTorch was built for CUDA $TORCH_CUDA"
    echo "  Fix: sudo bash install-nvidia-driver.sh && sudo reboot"
    echo "  Then: uv pip install vllm --torch-backend=cu126 --reinstall"
    exit 1
  fi
fi

# ── HuggingFace auth ──────────────────────────────────────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -f "$HF_TOKEN_FILE" ]]; then
  HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
fi
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "ERROR: HuggingFace auth required for Llama 8B (gated model)."
  echo "  1. Accept license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
  echo "  2. hf auth login   OR   export HF_TOKEN=hf_..."
  echo "  3. Verify: hf auth whoami"
  echo "  4. Re-run: bash benchmark/vllm-bench/run_pareto_benchmark.sh"
  exit 1
fi
export HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── Preflight: verify CUDA is usable ─────────────────────────────────────────
CUDA_OK=$(python3 -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
if [[ "$CUDA_OK" != "yes" ]]; then
  echo "ERROR: PyTorch cannot initialise CUDA (torch.cuda.is_available() == False)."
  echo "  The nvidia_uvm driver may be in a broken state after a previous crash."
  echo "  Run these commands and retry:"
  echo "    UVM_MAJOR=\$(awk '/nvidia-uvm/{print \$1}' /proc/devices)"
  echo "    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm"
  echo "    UVM_MAJOR_NEW=\$(awk '/nvidia-uvm/{print \$1}' /proc/devices)"
  echo "    sudo rm /dev/nvidia-uvm /dev/nvidia-uvm-tools"
  echo "    sudo mknod /dev/nvidia-uvm c \$UVM_MAJOR_NEW 0"
  echo "    sudo mknod /dev/nvidia-uvm-tools c \$UVM_MAJOR_NEW 1"
  echo "    sudo chmod 666 /dev/nvidia-uvm /dev/nvidia-uvm-tools"
  echo "  Or just reboot."
  exit 1
fi

echo ""
if [[ ! -f "$DATASET_FILE" ]]; then
  echo "ERROR: Dataset not found: $DATASET_FILE"
  echo "  Download it first:"
  echo "    python3 -c \"from huggingface_hub import hf_hub_download; hf_hub_download('Aeala/ShareGPT_Vicuna_unfiltered', 'ShareGPT_V4.3_unfiltered_cleaned_split.json', repo_type='dataset', local_dir='datasets')\""
  exit 1
fi

echo "=== Model      : $MODEL ==="
echo "=== Experiment : $EXPERIMENT ==="
echo "=== Dataset    : $DATASET_FILE ==="
echo "=== OSL        : $OSL tokens (fixed via --sharegpt-output-len + --ignore-eos) ==="
echo "=== Concurrency sweep: $CONCURRENCY_LIST ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Start vLLM server ─────────────────────────────────────────────────────────
# vllm 0.22+ bundles vllm_flash_attn built against libcudart.so.13 (CUDA 13),
# which fails on driver-570 (max CUDA 12.8). Use TRITON_ATTN instead.
echo ""
echo "=== Starting vLLM server on port $SERVER_PORT ==="
vllm serve "$MODEL" \
  --dtype auto \
  --attention-backend TRITON_ATTN \
  --tensor-parallel-size "$TP" \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --port "$SERVER_PORT" \
  > "$EXPERIMENT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap 'echo ""; echo "Shutting down vLLM server (pid=$SERVER_PID)..."; kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait up to 10 minutes for the server to become healthy
echo "Waiting for server to be ready (up to 10 min)..."
READY=0
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${SERVER_PORT}/health" &>/dev/null; then
    echo "Server ready after ~$((i * 5))s"
    READY=1
    break
  fi
  sleep 5
done
if (( READY == 0 )); then
  echo "ERROR: Server did not start in time. Check $EXPERIMENT_DIR/server.log"
  exit 1
fi

# ── Concurrency sweep ─────────────────────────────────────────────────────────
echo ""
echo "=== Running concurrency sweep ==="
for CONCURRENCY in $CONCURRENCY_LIST; do
  echo ""
  NUM_PROMPTS=$(( CONCURRENCY * ROUNDS ))
  echo "--- concurrency=$CONCURRENCY  num_prompts=$NUM_PROMPTS (${ROUNDS}× waves) ---"
  RUN_DIR="$EXPERIMENT_DIR/concurrency_${CONCURRENCY}"
  mkdir -p "$RUN_DIR"

  # Flush vLLM's KV cache between steps so earlier runs don't warm the cache
  # for later ones.  The /reset_prefix_cache endpoint is available in vLLM 0.4+.
  # This might be the reason why previously, vllm runs faster than sglang
  curl -sf -X POST "http://localhost:${SERVER_PORT}/reset_prefix_cache" &>/dev/null || true

  vllm bench serve \
    --model "$MODEL" \
    --backend vllm \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path "$DATASET_FILE" \
    --sharegpt-output-len "$OSL" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --host localhost \
    --port "$SERVER_PORT" \
    --ignore-eos \
    --save-result \
    --result-dir "$RUN_DIR" \
    2>&1 | tee "$RUN_DIR/bench.log"
done

# Explicit clean shutdown before plotting
echo ""
echo "=== All runs done — shutting down server ==="
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null || true
trap - EXIT

# ── Pareto chart ──────────────────────────────────────────────────────────────
echo ""
echo "=== Generating Pareto chart ==="

python3 - <<PYEOF
import json, glob, os, sys

experiment_dir = "$EXPERIMENT_DIR"
tp = $TP

results = []
for conc_dir in sorted(
    glob.glob(os.path.join(experiment_dir, "concurrency_*")),
    key=lambda p: int(p.rsplit("_", 1)[-1]),
):
    concurrency = int(conc_dir.rsplit("_", 1)[-1])
    jsons = sorted(glob.glob(os.path.join(conc_dir, "*.json")))
    if not jsons:
        print(f"  WARNING: no result JSON in {conc_dir}, skipping")
        continue
    with open(jsons[-1]) as f:
        data = json.load(f)
    out_tps = data.get("output_throughput", 0)
    results.append(
        dict(
            concurrency=concurrency,
            output_throughput=out_tps,
            # X axis: system throughput per GPU (rises with concurrency)
            tps_per_gpu=out_tps / max(tp, 1),
            # Y axis: per-user throughput (falls with concurrency)
            tps_per_user=out_tps / max(concurrency, 1),
            mean_ttft_ms=data.get("mean_ttft_ms", 0),
            mean_tpot_ms=data.get("mean_tpot_ms", 0),
            request_throughput=data.get("request_throughput", 0),
        )
    )

if not results:
    print("ERROR: no benchmark results found under", experiment_dir)
    sys.exit(1)

# Summary table
header = f"{'Concurrency':>12}  {'tok/s/GPU':>10}  {'tok/s/user':>10}  {'TTFT ms':>9}  {'TPOT ms':>9}"
print("\n" + header)
print("-" * len(header))
for r in results:
    print(
        f"{r['concurrency']:>12}  {r['tps_per_gpu']:>10.1f}  "
        f"{r['tps_per_user']:>10.2f}  {r['mean_ttft_ms']:>9.1f}  {r['mean_tpot_ms']:>9.2f}"
    )

# Save summary JSON
summary_path = os.path.join(experiment_dir, "summary.json")
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSummary saved to: {summary_path}")

# Plot Pareto frontier
# X = tok/s/user (per-user speed,     decreases as concurrency rises)
# Y = tok/s/GPU  (system throughput,  increases as concurrency rises)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["tps_per_user"] for r in results]   # per-user speed    → decreases
    y = [r["tps_per_gpu"]  for r in results]   # system throughput → increases

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, y, "o-", color="steelblue", linewidth=2, markersize=8, zorder=3)

    # Alternate label offsets above/below to avoid crowding on dense sweeps
    for i, (xi, yi, r) in enumerate(zip(x, y, results)):
        dy = 6 if i % 2 == 0 else -14
        ax.annotate(
            f"c={r['concurrency']}",
            (xi, yi),
            textcoords="offset points",
            xytext=(5, dy),
            fontsize=8,
            color="#333",
        )

    ax.set_xlabel("Per-user throughput  (output tok/s / user)", fontsize=12)
    ax.set_ylabel(f"System throughput  (output tok/s / GPU,  TP={tp})", fontsize=12)
    ax.set_title(
        f"Pareto frontier — throughput vs. per-user speed\n{os.path.basename(experiment_dir)}",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    pareto_dir = os.path.join(experiment_dir, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)
    out_path = os.path.join(pareto_dir, "PARETO.png")
    fig.savefig(out_path, dpi=150)
    print(f"Pareto chart saved to: {out_path}")
except ImportError:
    print("matplotlib not installed; skipping chart.")
    print("Install with:  uv pip install matplotlib")
PYEOF

echo ""
echo "Done."
echo "Results dir: $EXPERIMENT_DIR"
ls -la "$EXPERIMENT_DIR/" 2>/dev/null || true
