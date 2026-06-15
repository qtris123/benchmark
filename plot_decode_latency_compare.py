#!/usr/bin/env python3
"""
Compare decode latency of two serving engines (vLLM vs sglang) across
concurrency levels, reading the Pareto-sweep `summary.jsonl` files.

Three metrics per concurrency level, one line per engine:
  1. Mean TPOT (ms)                  - mean time per output token (decode)
  2. Median TPOT (ms)                - median time per output token (decode)
  3. Overall decode latency (ms)     - mean per-request decode-phase wall time
                                       = mean end-to-end latency - mean TTFT
                                       (i.e. time to generate all output tokens)

Everything is derived from the data: engines, concurrency levels, and axis
ranges are discovered from the `summary.jsonl` files (no hardcoded numbers).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _get(o: dict, *keys):
    """Return first present key (handles vLLM vs sglang field-name drift)."""
    for k in keys:
        if k in o and o[k] is not None:
            return o[k]
    return None


def load_engine(summary_path: Path):
    """Return (engine_name, sorted list of per-concurrency metric dicts)."""
    rows = []
    engine = summary_path.parent.name
    with summary_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            engine = _get(o, "backend", "endpoint_type") or engine
            conc = _get(o, "max_concurrency", "concurrency")
            mean_tpot = _get(o, "mean_tpot_ms")
            median_tpot = _get(o, "median_tpot_ms")
            mean_ttft = _get(o, "mean_ttft_ms")
            mean_e2e = _get(o, "mean_e2e_latency_ms", "mean_e2el_ms")
            if conc is None or mean_tpot is None:
                continue
            decode_lat = (mean_e2e - mean_ttft
                          if mean_e2e is not None and mean_ttft is not None
                          else None)
            rows.append({
                "concurrency": int(conc),
                "mean_tpot_ms": mean_tpot,
                "median_tpot_ms": median_tpot,
                "mean_ttft_ms": mean_ttft,
                "mean_e2e_ms": mean_e2e,
                "decode_latency_ms": decode_lat,
            })
    rows.sort(key=lambda r: r["concurrency"])
    return engine, rows


def write_csv(data: dict[str, list[dict]], path: Path) -> None:
    cols = ["engine", "concurrency", "mean_tpot_ms", "median_tpot_ms",
            "mean_ttft_ms", "mean_e2e_ms", "decode_latency_ms"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for engine, rows in data.items():
            for r in rows:
                w.writerow({"engine": engine, **r})
    print(f"[csv]   {path}")


def make_plot(data: dict[str, list[dict]], out_png: Path, title_suffix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("mean_tpot_ms",      "Mean TPOT",               "ms / output token"),
        ("median_tpot_ms",    "Median TPOT",             "ms / output token"),
        ("decode_latency_ms", "Overall decode latency\n(mean e2e - mean TTFT)",
                              "ms / request"),
    ]
    colors = {"vllm": "#ff7f0e", "sglang": "#1f77b4"}
    markers = {"vllm": "o", "sglang": "s"}

    fig, axes = plt.subplots(1, len(panels), figsize=(16, 5))
    all_concs = sorted({r["concurrency"] for rows in data.values() for r in rows})

    for ax, (key, title, ylabel) in zip(axes, panels):
        for engine, rows in sorted(data.items()):
            xs = [r["concurrency"] for r in rows if r.get(key) is not None]
            ys = [r[key] for r in rows if r.get(key) is not None]
            ax.plot(xs, ys, marker=markers.get(engine, "o"),
                    color=colors.get(engine), label=engine, linewidth=2,
                    markersize=6)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.0f}" if y >= 100 else f"{y:.1f}",
                            (x, y), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=7,
                            color=colors.get(engine))
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_concs)
        ax.set_xticklabels([str(c) for c in all_concs])
        ax.set_xlabel("concurrency (requests in flight)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="engine")

    fig.suptitle(f"Decode latency: vLLM vs sglang  {title_suffix}".strip(),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[chart] {out_png}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=here / "pareto-results",
                    help="dir containing <engine>_*/summary.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output dir (default: results-dir)")
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = (args.out_dir or results_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = sorted(results_dir.glob("*/summary.jsonl"))
    if not summaries:
        raise SystemExit(f"no '*/summary.jsonl' found under {results_dir}")

    data: dict[str, list[dict]] = {}
    suffix = ""
    for sp in summaries:
        engine, rows = load_engine(sp)
        data[engine] = rows
        # Derive an ISL/OSL label from the dir name (purely cosmetic).
        name = sp.parent.name
        if "isl" in name and not suffix:
            suffix = "(" + name.split("_", 1)[1].replace("_", ", ") + ")"
        print(f"  loaded {engine:>8}: {len(rows)} concurrency levels from {sp}")

    write_csv(data, out_dir / "decode_latency_compare.csv")
    make_plot(data, out_dir / "decode_latency_compare.png", suffix)


if __name__ == "__main__":
    main()
