#!/usr/bin/env bash
# Overlay SGLang vs vLLM Pareto frontiers and annotate per-concurrency gaps.
#
# Usage:
#   bash sglang-bench/compare.sh [SGLANG_EXPERIMENT] [VLLM_EXPERIMENT_DIR]
#
# Examples:
#   bash sglang-bench/compare.sh llama8b_1k1k_tp1_sglang_ver2
#   bash sglang-bench/compare.sh llama8b_1k1k_tp1_sglang_ver2 vllm-bench/results/llama8b_1k1k_tp1_ver2
#
# Outputs:
#   sglang-bench/results/<SGLANG_EXPERIMENT>/pareto/COMPARE.png
#
# Gap annotations per matched concurrency level:
#   Horizontal arrow  →  Δ tok/s/user  (SGLang − vLLM)
#   Vertical   arrow  ↑  Δ tok/s/GPU   (SGLang − vLLM)
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

# ── Resolve SGLang experiment directory ──────────────────────────────────────
_sg_arg="${1:-${SGLANG_EXPERIMENT:-}}"
if [[ -z "$_sg_arg" ]]; then
  echo "ERROR: Provide a SGLang experiment name or directory."
  echo "  Usage: bash sglang-bench/compare.sh <SGLANG_EXPERIMENT> [VLLM_DIR]"
  echo "  Example: bash sglang-bench/compare.sh llama8b_1k1k_tp1_sglang_ver2"
  exit 1
fi
if [[ -d "$_sg_arg" ]]; then
  SG_DIR="$(cd "$_sg_arg" && pwd)"
elif [[ -d "$SGLANG_BENCH/results/$_sg_arg" ]]; then
  SG_DIR="$SGLANG_BENCH/results/$_sg_arg"
else
  echo "ERROR: Cannot find SGLang experiment dir for '$_sg_arg'."
  exit 1
fi

# ── Resolve vLLM experiment directory ────────────────────────────────────────
_vl_arg="${2:-${VLLM_EXPERIMENT_DIR:-$ROOT/vllm-bench/results/llama8b_1k1k_tp1_ver2}}"
if [[ -d "$_vl_arg" ]]; then
  VL_DIR="$(cd "$_vl_arg" && pwd)"
else
  echo "ERROR: Cannot find vLLM experiment dir '$_vl_arg'."
  exit 1
fi

PARETO_DIR="$SG_DIR/pareto"
mkdir -p "$PARETO_DIR"

echo "=== Comparing Pareto frontiers ==="
echo "    SGLang : $SG_DIR"
echo "    vLLM   : $VL_DIR"
echo "    Output : $PARETO_DIR/COMPARE.png"

export SG_DIR VL_DIR PARETO_DIR
python3 - <<'PYEOF'
import json, glob, os, sys, pathlib

sg_dir    = os.environ["SG_DIR"]
vl_dir    = os.environ["VL_DIR"]
pareto_dir = os.environ["PARETO_DIR"]

# ── Load SGLang summary.jsonl ─────────────────────────────────────────────────
# tok/s/user = 1000 / mean_tpot_ms   (per-user generation speed)
# tok/s/GPU  = output_throughput      (TP=1 → total system throughput)
sg_summary = pathlib.Path(sg_dir) / "summary.jsonl"
if not sg_summary.exists():
    print(f"ERROR: SGLang summary not found: {sg_summary}")
    sys.exit(1)

sg = {}
for line in sg_summary.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    c = r.get("max_concurrency")
    if c is None:
        continue
    tpot = r.get("mean_tpot_ms", 0)
    sg[c] = dict(
        tok_per_user = 1000.0 / tpot if tpot > 0 else 0,
        tok_per_gpu  = r.get("output_throughput", 0),
    )

# ── Load vLLM per-concurrency JSONs ──────────────────────────────────────────
# tok/s/user = output_throughput / concurrency
# tok/s/GPU  = output_throughput / tp  (tp=1 here)
TP = 1
vl = {}
_vl_dirs = sorted(
    glob.glob(os.path.join(vl_dir, "concurrency_*")) +
    glob.glob(os.path.join(vl_dir, "concurrency=*")),
    key=lambda p: int(p.replace("=", "_").rsplit("_", 1)[-1]),
)
for conc_dir in _vl_dirs:
    c = int(conc_dir.replace("=", "_").rsplit("_", 1)[-1])
    jsons = sorted(glob.glob(os.path.join(conc_dir, "*.json")))
    if not jsons:
        continue
    data = json.load(open(jsons[-1]))
    out  = data.get("output_throughput", 0)
    vl[c] = dict(
        tok_per_user = out / max(c, 1),
        tok_per_gpu  = out / max(TP, 1),
    )

# ── Common concurrency levels ─────────────────────────────────────────────────
common = sorted(set(sg) & set(vl))
all_sg = sorted(sg)
all_vl = sorted(vl)

if not common:
    print("WARNING: No overlapping concurrency levels found between the two engines.")
    print(f"  SGLang concurrencies : {all_sg}")
    print(f"  vLLM   concurrencies : {all_vl}")
    sys.exit(1)

print(f"  SGLang concurrencies : {all_sg}")
print(f"  vLLM   concurrencies : {all_vl}")
print(f"  Common               : {common}")

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SG_COLOR = "steelblue"
VL_COLOR = "tomato"

fig, ax = plt.subplots(figsize=(11, 7))

# ── Draw frontiers ────────────────────────────────────────────────────────────
sg_x = [sg[c]["tok_per_user"] for c in all_sg]
sg_y = [sg[c]["tok_per_gpu"]  for c in all_sg]
vl_x = [vl[c]["tok_per_user"] for c in all_vl]
vl_y = [vl[c]["tok_per_gpu"]  for c in all_vl]

ax.plot(sg_x, sg_y, "o-", color=SG_COLOR, lw=2, markersize=8, zorder=4, label="SGLang")
ax.plot(vl_x, vl_y, "s-", color=VL_COLOR, lw=2, markersize=8, zorder=4, label="vLLM")

# Label every point with its concurrency
for i, (c, x, y) in enumerate(zip(all_sg, sg_x, sg_y)):
    dy = 7 if i % 2 == 0 else -14
    ax.annotate(f"c={c}", (x, y), textcoords="offset points",
                xytext=(5, dy), fontsize=7.5, color=SG_COLOR)
for i, (c, x, y) in enumerate(zip(all_vl, vl_x, vl_y)):
    dy = 7 if i % 2 == 0 else -14
    ax.annotate(f"c={c}", (x, y), textcoords="offset points",
                xytext=(5, dy), fontsize=7.5, color=VL_COLOR)

# ── Labels, legend, grid ──────────────────────────────────────────────────────
sg_label = os.path.basename(sg_dir)
vl_label = os.path.basename(vl_dir)

ax.set_xlabel("(output tok/s / user)  ↑ better", fontsize=12)
ax.set_ylabel("(output tok/s / GPU)   ↑ better", fontsize=12)
ax.set_title(
    f"SGLang vs vLLM — Pareto frontier comparison\n"
    f"SGLang: {sg_label}   |   vLLM: {vl_label}",
    fontsize=12, fontweight="bold",
)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=10, loc="upper right")

fig.tight_layout()
out_path = os.path.join(pareto_dir, "COMPARE.png")
fig.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close(fig)

# ── Diverging bar charts ──────────────────────────────────────────────────────
d_gpu  = [sg[c]["tok_per_gpu"]  - vl[c]["tok_per_gpu"]  for c in common]
d_user = [sg[c]["tok_per_user"] - vl[c]["tok_per_user"] for c in common]
xlabels = [f"c={c}" for c in common]
x = range(len(common))

fig2, (ax_gpu, ax_user) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle(
    f"SGLang − vLLM  delta per concurrency level\n"
    f"SGLang: {sg_label}   |   vLLM: {vl_label}",
    fontsize=12, fontweight="bold",
)

def _diverging_bar(ax, xs, xlabels, deltas, ylabel, title):
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]
    bars = ax.bar(xs, deltas, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="#333", lw=1.2, zorder=4)
    # Value labels: above bar for positive, below for negative
    for bar, d in zip(bars, deltas):
        sign = "+" if d >= 0 else ""
        va   = "bottom" if d >= 0 else "top"
        pad  = 0.5 if d >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width() / 2, d + pad,
                f"{sign}{d:.1f}",
                ha="center", va=va, fontsize=8, fontweight="bold",
                color=bar.get_facecolor())
    ax.set_xticks(list(xs))
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_xlabel("Max Concurrency", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
    # SGLang / vLLM region labels
    ymax = max(abs(d) for d in deltas) * 1.35 or 1
    ax.set_ylim(-ymax, ymax)
    ax.text(0.98, 0.97, "SGLang leads ▲", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="steelblue", style="italic")
    ax.text(0.98, 0.03, "vLLM leads ▼", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="tomato", style="italic")

_diverging_bar(ax_gpu,  x, xlabels, d_gpu,
               "Δ tok/s/GPU  (SGLang − vLLM)",
               "Output tok/s/GPU — delta")
_diverging_bar(ax_user, x, xlabels, d_user,
               "Δ tok/s/user  (SGLang − vLLM)",
               "Output tok/s/user — delta")

fig2.tight_layout()
delta_path = os.path.join(pareto_dir, "COMPARE_delta.png")
fig2.savefig(delta_path, dpi=150)
print(f"Saved: {delta_path}")
plt.close(fig2)

# ── Console gap table ─────────────────────────────────────────────────────────
print()
print("── Gap Summary (SGLang − vLLM) ──────────────────────────────────────────────────────────────────")
header = f"{'Concur':>7}  {'SG tok/s/user':>14}  {'VL tok/s/user':>14}  {'Δ user':>9}  {'SG tok/s/GPU':>13}  {'VL tok/s/GPU':>13}  {'Δ GPU':>8}"
print(header)
print("─" * len(header))
for c in common:
    su, sg_gpu = sg[c]["tok_per_user"], sg[c]["tok_per_gpu"]
    vu, vl_gpu = vl[c]["tok_per_user"], vl[c]["tok_per_gpu"]
    print(f"{c:>7}  {su:>14.2f}  {vu:>14.2f}  {su-vu:>+9.2f}  {sg_gpu:>13.1f}  {vl_gpu:>13.1f}  {sg_gpu-vl_gpu:>+8.1f}")
PYEOF

echo ""
echo "Done."
echo "  Chart (overlay) : $PARETO_DIR/COMPARE.png"
echo "  Chart (delta)   : $PARETO_DIR/COMPARE_delta.png"
