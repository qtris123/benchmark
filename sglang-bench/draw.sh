#!/usr/bin/env bash
# Draw Pareto and latency charts for a completed SGLang benchmark experiment.
#
# Usage:
#   bash sglang-bench/draw.sh [EXPERIMENT_NAME_OR_DIR]
#
# Examples:
#   bash sglang-bench/draw.sh llama8b_1k1k_tp1_sglang_ver2
#   bash sglang-bench/draw.sh sglang-bench/results/llama8b_1k1k_tp1_sglang_ver2
#   EXPERIMENT=llama8b_1k1k_tp1_sglang_ver2 bash sglang-bench/draw.sh
#
# The script looks for summary.jsonl inside the experiment directory and writes
# PARETO.png into a pareto/ subdirectory next to it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGLANG_ENV="$ROOT/sglang-env"
SGLANG_BENCH="$ROOT/sglang-bench"

# ── Activate virtualenv ──────────────────────────────────────────────────────
if [[ ! -x "$SGLANG_ENV/bin/python3" ]]; then
  echo "ERROR: sglang-env not found at $SGLANG_ENV"
  exit 1
fi
source "$SGLANG_ENV/bin/activate"
source "$SGLANG_BENCH/env.sh"

# ── Resolve experiment directory ─────────────────────────────────────────────
_arg="${1:-${EXPERIMENT:-}}"
if [[ -z "$_arg" ]]; then
  echo "ERROR: Provide an experiment name or directory."
  echo "  Usage: bash sglang-bench/draw.sh <EXPERIMENT_NAME_OR_DIR>"
  echo "  Example: bash sglang-bench/draw.sh llama8b_1k1k_tp1_sglang_ver2"
  exit 1
fi

# Accept either an absolute/relative path or a bare experiment name.
if [[ -d "$_arg" ]]; then
  EXPERIMENT_DIR="$(cd "$_arg" && pwd)"
elif [[ -d "$SGLANG_BENCH/results/$_arg" ]]; then
  EXPERIMENT_DIR="$SGLANG_BENCH/results/$_arg"
else
  echo "ERROR: Cannot find experiment directory for '$_arg'."
  echo "  Looked at: $_arg  and  $SGLANG_BENCH/results/$_arg"
  exit 1
fi

EXPERIMENT="${EXPERIMENT:-$(basename "$EXPERIMENT_DIR")}"

# Read metadata from the first concurrency result if available, else use defaults.
_meta_file="$(ls "$EXPERIMENT_DIR"/concurrency=*/bench_results.jsonl 2>/dev/null | head -1 || true)"
if [[ -n "$_meta_file" ]]; then
  _meta=$(python3 -c "
import json, sys
data = json.loads(open('$_meta_file').readlines()[-1])
print(data.get('model_id','unknown'))
print(int(data.get('random_input_len', data.get('input_length', 1000))))
print(int(data.get('random_output_len', data.get('output_length', 1000))))
print(int(data.get('num_prompts', 1024)))
print(int(data.get('tp', 1)))
" 2>/dev/null || echo -e "unknown\n1000\n1000\n1024\n1")
  MODEL="${MODEL:-$(echo "$_meta" | sed -n '1p')}"
  RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-$(echo "$_meta" | sed -n '2p')}"
  RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-$(echo "$_meta" | sed -n '3p')}"
  NUM_PROMPTS="${NUM_PROMPTS:-$(echo "$_meta" | sed -n '4p')}"
  TP="${TP:-$(echo "$_meta" | sed -n '5p')}"
else
  MODEL="${MODEL:-unknown}"
  RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1000}"
  RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1000}"
  NUM_PROMPTS="${NUM_PROMPTS:-1024}"
  TP="${TP:-1}"
fi

SUMMARY_FILE="$EXPERIMENT_DIR/summary.jsonl"

echo "=== Drawing Pareto charts ==="
echo "    Experiment : $EXPERIMENT"
echo "    Dir        : $EXPERIMENT_DIR"
echo "    Summary    : $SUMMARY_FILE"

# ── Pareto / summary charts ──────────────────────────────────────────────────
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

export SUMMARY_FILE PARETO_DIR EXPERIMENT MODEL RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN NUM_PROMPTS TP
python3 - <<'PYEOF'
import json, os, sys, pathlib

summary_path = os.environ["SUMMARY_FILE"]
pareto_dir   = os.environ["PARETO_DIR"]
experiment   = os.environ["EXPERIMENT"]
model        = os.environ["MODEL"]
isl          = os.environ["RANDOM_INPUT_LEN"]
osl          = os.environ["RANDOM_OUTPUT_LEN"]
num_p        = os.environ["NUM_PROMPTS"]
tp           = os.environ["TP"]

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
    f"{model}  |  ISL/OSL {isl}/{osl}  |  TP={tp}  |  num_prompts={num_p}  |  request_rate=inf",
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
plt.close()

# ── Standalone Pareto scatter: tok/s/user (x) vs tok/s/GPU (y) ───────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))
ax2.plot(tput_per_user, output_tput, "o-", color="steelblue", linewidth=2, markersize=8, zorder=3)
for i, (xi, yi, c) in enumerate(zip(tput_per_user, output_tput, concurrencies)):
    dy = 6 if i % 2 == 0 else -14
    ax2.annotate(
        f"c={c}", (xi, yi),
        textcoords="offset points", xytext=(5, dy),
        fontsize=8, color="#333",
    )
ax2.set_xlabel("(output tok/s / user)", fontsize=12)
ax2.set_ylabel(f"(output tok/s / GPU,  TP={tp})", fontsize=12)
ax2.set_title(
    f"SGLang — Pareto frontier — throughput vs. per-user speed\n{experiment}",
    fontsize=13,
)
ax2.grid(True, linestyle="--", alpha=0.4)
fig2.tight_layout()
pareto_path = os.path.join(pareto_dir, "PARETO_scatter.png")
fig2.savefig(pareto_path, dpi=150)
print(f"Saved: {pareto_path}")
plt.close(fig2)

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
echo "  Results        : $EXPERIMENT_DIR"
echo "  Chart (full)   : $PARETO_DIR/PARETO.png"
echo "  Chart (pareto) : $PARETO_DIR/PARETO_scatter.png"
ls -la "$PARETO_DIR/" 2>/dev/null || true
