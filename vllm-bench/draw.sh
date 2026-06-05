#!/usr/bin/env bash
# Usage: bash benchmarks/draw.sh [experiment_dir] [tp]
#   or:  EXPERIMENT_DIR=... TP=... bash benchmarks/draw.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Accept positional args or env vars, with sane defaults
EXPERIMENT_DIR="${1:-${EXPERIMENT_DIR:-benchmarks/results/llama8b_1k1k_tp1_ver2}}"
TP="${2:-${TP:-1}}"

# Activate venv if not already in one
if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi
if [[ -f "benchmarks/env.sh" ]]; then
  source benchmarks/env.sh
fi

# ── Pareto chart ──────────────────────────────────────────────────────────────
echo ""
echo "=== Generating Pareto chart ==="
echo "    experiment_dir : $EXPERIMENT_DIR"
echo "    TP             : $TP"

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

    ax.set_xlabel("(output tok/s / user)", fontsize=12)
    ax.set_ylabel(f"(output tok/s / GPU,  TP={tp})", fontsize=12)
    ax.set_title(
        f"vLLM — Pareto frontier — throughput vs. per-user speed\n{os.path.basename(experiment_dir)}",
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
