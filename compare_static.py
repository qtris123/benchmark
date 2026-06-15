#!/usr/bin/env python3
"""Overlay + delta comparison of two static-benchmark runs (SGLang vs vLLM).

Matches the style of pareto-results/.../pareto/COMPARE.png + COMPARE_delta.png:
  COMPARE.png        single-axes Pareto frontier overlay (tok/s/user vs tok/s/GPU)
  COMPARE_delta.png  two diverging-bar panels: SGLang - vLLM per concurrency

Both engines write an identical-schema summary.jsonl via run_static_benchmark.sh
(one JSON object per concurrency level), with at least:
  max_concurrency, output_throughput, mean_tpot_ms

Axes:
  tok/s/user = 1000 / mean_tpot_ms              (per-user generation speed)
  tok/s/GPU  = output_throughput / TP           (TP parsed from the dir name)

Usage:
  python3 compare_static.py [SGLANG_DIR] [VLLM_DIR] [--out OUT_DIR]
  # no args -> defaults to the 8B TP2 sglang-vs-vllm pair.
"""
import argparse
import json
import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).resolve().parent / "static-benchmark-results"
DEFAULT_SG = RESULTS / "sglang_meta-llama-3.1-8b-instruct_isl1000_osl1000_tp2_static"
DEFAULT_VL = RESULTS / "vllm_meta-llama-3.1-8b-instruct_isl1000_osl1000_tp2_static"

SG_COLOR = "steelblue"
VL_COLOR = "tomato"


def tp_for(d: pathlib.Path) -> int:
    m = re.search(r"_tp(\d+)", d.name)
    return int(m.group(1)) if m else 1


def load(d: pathlib.Path, tp: int) -> dict:
    """Return {concurrency: {tok_per_user, tok_per_gpu}} from summary.jsonl."""
    summ = d / "summary.jsonl"
    if not summ.exists():
        sys.exit(f"ERROR: summary.jsonl not found in {d}")
    out = {}
    for line in summ.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        c = r.get("max_concurrency")
        if c is None:
            continue
        tpot = r.get("mean_tpot_ms") or 0
        tput = r.get("output_throughput") or 0
        out[c] = dict(
            tok_per_user=(1000.0 / tpot) if tpot > 0 else 0,
            tok_per_gpu=tput / max(tp, 1),
        )
    if not out:
        sys.exit(f"ERROR: no usable rows in {summ}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sglang_dir", nargs="?", default=str(DEFAULT_SG))
    ap.add_argument("vllm_dir", nargs="?", default=str(DEFAULT_VL))
    ap.add_argument("--out", default=None, help="output dir for the PNGs")
    args = ap.parse_args()

    sg_dir = pathlib.Path(args.sglang_dir).resolve()
    vl_dir = pathlib.Path(args.vllm_dir).resolve()
    tp_sg, tp_vl = tp_for(sg_dir), tp_for(vl_dir)
    sg, vl = load(sg_dir, tp_sg), load(vl_dir, tp_vl)

    out_dir = pathlib.Path(args.out) if args.out else \
        RESULTS / f"compare_sglang_vs_vllm_{sg_dir.name.split('_', 1)[-1]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_sg = sorted(sg)
    all_vl = sorted(vl)
    common = sorted(set(sg) & set(vl))
    if not common:
        sys.exit(f"ERROR: no shared concurrency levels.\n  SGLang: {all_sg}\n  vLLM: {all_vl}")

    sg_label = sg_dir.name
    vl_label = vl_dir.name

    # ───────────────────────── COMPARE.png (Pareto overlay) ──────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))
    sg_x = [sg[c]["tok_per_user"] for c in all_sg]
    sg_y = [sg[c]["tok_per_gpu"] for c in all_sg]
    vl_x = [vl[c]["tok_per_user"] for c in all_vl]
    vl_y = [vl[c]["tok_per_gpu"] for c in all_vl]

    ax.plot(sg_x, sg_y, "o-", color=SG_COLOR, lw=2, markersize=8, zorder=4, label="SGLang")
    ax.plot(vl_x, vl_y, "s-", color=VL_COLOR, lw=2, markersize=8, zorder=4, label="vLLM")

    for i, (c, x, y) in enumerate(zip(all_sg, sg_x, sg_y)):
        dy = 7 if i % 2 == 0 else -14
        ax.annotate(f"c={c}", (x, y), textcoords="offset points",
                    xytext=(5, dy), fontsize=7.5, color=SG_COLOR)
    for i, (c, x, y) in enumerate(zip(all_vl, vl_x, vl_y)):
        dy = 7 if i % 2 == 0 else -14
        ax.annotate(f"c={c}", (x, y), textcoords="offset points",
                    xytext=(5, dy), fontsize=7.5, color=VL_COLOR)

    ax.set_xlabel("(output tok/s / user)  ↑ better", fontsize=12)
    ax.set_ylabel("(output tok/s / GPU)   ↑ better", fontsize=12)
    ax.set_title(
        "SGLang vs vLLM — Pareto frontier comparison\n"
        f"SGLang: {sg_label}   |   vLLM: {vl_label}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    overlay = out_dir / "COMPARE.png"
    fig.savefig(overlay, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ───────────────────────── COMPARE_delta.png (diverging bars) ────────────────
    d_gpu = [sg[c]["tok_per_gpu"] - vl[c]["tok_per_gpu"] for c in common]
    d_user = [sg[c]["tok_per_user"] - vl[c]["tok_per_user"] for c in common]
    xlabels = [f"c={c}" for c in common]
    x = range(len(common))

    fig2, (ax_gpu, ax_user) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(
        "SGLang − vLLM  delta per concurrency level\n"
        f"SGLang: {sg_label}   |   vLLM: {vl_label}",
        fontsize=12, fontweight="bold",
    )

    def _diverging_bar(ax, xs, xlabels, deltas, ylabel, title):
        colors = [SG_COLOR if d >= 0 else VL_COLOR for d in deltas]
        bars = ax.bar(xs, deltas, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
        ax.axhline(0, color="#333", lw=1.2, zorder=4)
        for bar, d in zip(bars, deltas):
            sign = "+" if d >= 0 else ""
            va = "bottom" if d >= 0 else "top"
            pad = (max(abs(v) for v in deltas) or 1) * (0.02 if d >= 0 else -0.02)
            ax.text(bar.get_x() + bar.get_width() / 2, d + pad,
                    f"{sign}{d:.1f}", ha="center", va=va, fontsize=8,
                    fontweight="bold", color=bar.get_facecolor())
        ax.set_xticks(list(xs))
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_xlabel("Max Concurrency", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4, zorder=0)
        ymax = (max(abs(d) for d in deltas) or 1) * 1.35
        ax.set_ylim(-ymax, ymax)
        ax.text(0.98, 0.97, "SGLang leads ▲", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=SG_COLOR, style="italic")
        ax.text(0.98, 0.03, "vLLM leads ▼", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=VL_COLOR, style="italic")

    _diverging_bar(ax_gpu, x, xlabels, d_gpu,
                   "Δ tok/s/GPU  (SGLang − vLLM)", "Output tok/s/GPU — delta")
    _diverging_bar(ax_user, x, xlabels, d_user,
                   "Δ tok/s/user  (SGLang − vLLM)", "Output tok/s/user — delta")

    fig2.tight_layout()
    delta = out_dir / "COMPARE_delta.png"
    fig2.savefig(delta, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ───────────────────────── console gap table ─────────────────────────────────
    print(f"SGLang : {sg_dir}  (TP={tp_sg})")
    print(f"vLLM   : {vl_dir}  (TP={tp_vl})")
    print(f"SGLang concurrencies : {all_sg}")
    print(f"vLLM   concurrencies : {all_vl}")
    print(f"Common               : {common}")
    print()
    hdr = (f"{'Concur':>7}  {'SG t/s/user':>12}  {'VL t/s/user':>12}  {'Δ user':>9}  "
           f"{'SG t/s/GPU':>11}  {'VL t/s/GPU':>11}  {'Δ GPU':>8}")
    print(hdr); print("─" * len(hdr))
    for c in common:
        su, sgg = sg[c]["tok_per_user"], sg[c]["tok_per_gpu"]
        vu, vlg = vl[c]["tok_per_user"], vl[c]["tok_per_gpu"]
        print(f"{c:>7}  {su:>12.2f}  {vu:>12.2f}  {su-vu:>+9.2f}  "
              f"{sgg:>11.1f}  {vlg:>11.1f}  {sgg-vlg:>+8.1f}")
    print()
    print(f"Saved: {overlay}")
    print(f"Saved: {delta}")


if __name__ == "__main__":
    main()
