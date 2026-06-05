#!/usr/bin/env python3
"""
Automated nsys decode-phase analysis pipeline (vLLM vs sglang).

Stages (fully automated, raw .nsys-rep -> SQLite -> SQL -> CSV -> chart):

  1. DISCOVER  every '<engine>_c<N>.nsys-rep' profile in the profile dir.
  2. EXPORT    each report to SQLite via `nsys export` (idempotent / skippable).
  3. QUERY     decode-phase signals with SQL (GPU-time composition by kernel
               category, GPU utilisation, CUDA-graph vs eager split, kernel
               launch overhead, memcpy traffic). For c=32 first, then every
               concurrency found.
  4. CSV       write a tidy breakdown table + a scalar summary table.
  5. CHART     render grouped matplotlib bar charts straight from the CSV.

Design principle: NO hardcoded numbers. Everything (engines, concurrency
levels, GPU spec, decode-step counts, thresholds, axis ranges) is derived from
the data. The only string constants are regexes used to bucket CUDA kernel
*names* into functional categories -- the standard nsys analysis technique.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Kernel name -> functional category. First matching pattern wins. These are
# names (not numbers); they encode domain knowledge about LLM decode kernels.
# ---------------------------------------------------------------------------
CATEGORY_PATTERNS: list[tuple[str, str]] = [
    ("GEMM (Linear)",      r"gemm|cutlass|s16816|s1688|wgmma|cublas|nvjet|matmul|_dot_|sgemm|hgemm"),
    ("Attention",          r"fwd_kernel|flash|fmha|attn|paged|mha|flashinfer|kv_indic|kv_split|num_kv|decode_attention|prefill"),
    ("Sampling/Reduce",    r"topk|topp|top_k|top_p|softmax|sample|argmax|multinomial|distribution_elementwise|categorical|penalt|reduce_kernel|devicescan|devicereduce|cub::"),
    ("Norm/Act/RoPE",      r"rmsnorm|rms_norm|layernorm|layer_norm|_norm|rotary|rope|applyrotary|act_and_mul|silu|gelu|swiglu|activation|cos_sin|compute_position"),
    ("Elementwise/Index",  r"elementwise|index_|catarray|cat_array|copy|gather|scatter|memset|fill|stride|contiguous|nonzero|cumsum|indexselect|write_req|token_pool|slot_mapping|select"),
]
GRAPH_CATEGORY = "CUDA Graph (model fwd)"
MEMORY_CATEGORY = "Memory (H2D/D2H/D2D)"
OTHER_CATEGORY = "Other"

# Stable category order for charts/CSV (Memory + Graph appended by the pipeline).
CATEGORY_ORDER = [name for name, _ in CATEGORY_PATTERNS] + [
    OTHER_CATEGORY, MEMORY_CATEGORY, GRAPH_CATEGORY,
]

REPORT_RE = re.compile(r"^(?P<engine>[A-Za-z][A-Za-z0-9\-]*)_c(?P<conc>\d+)\.nsys-rep$")


def categorize(kernel_name: str) -> str:
    low = kernel_name.lower()
    for cat, pattern in CATEGORY_PATTERNS:
        if re.search(pattern, low):
            return cat
    return OTHER_CATEGORY


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class Report:
    engine: str
    conc: int
    rep_path: Path
    sqlite_path: Path


@dataclass
class Analysis:
    engine: str
    conc: int
    gpu_name: str = ""
    sm_count: int = 0
    wall_ms: float = 0.0
    gpu_busy_ms: float = 0.0           # union of all GPU intervals (true active time)
    gpu_util_pct: float = 0.0
    total_work_ms: float = 0.0         # summed durations (composition denominator)
    graph_exec: int = 0
    graph_ms: float = 0.0
    graph_mean_ms: float = 0.0
    graph_pct: float = 0.0
    eager_kernels: int = 0
    eager_ms: float = 0.0
    eager_mean_us: float = 0.0
    launch_api_calls: int = 0
    launch_per_ms: float = 0.0
    memcpy_count: int = 0
    memcpy_ms: float = 0.0
    h2d_mb: float = 0.0
    d2h_mb: float = 0.0
    d2d_mb: float = 0.0
    categories: dict[str, dict] = field(default_factory=dict)  # cat -> {n, ms}


# ---------------------------------------------------------------------------
# Stage 1: discover
# ---------------------------------------------------------------------------
def discover_reports(profile_dir: Path, sqlite_dir: Path) -> list[Report]:
    reports: list[Report] = []
    for p in sorted(profile_dir.glob("*.nsys-rep")):
        m = REPORT_RE.match(p.name)
        if not m:
            continue
        engine = m.group("engine")
        conc = int(m.group("conc"))
        reports.append(Report(
            engine=engine, conc=conc, rep_path=p,
            sqlite_path=sqlite_dir / f"{engine}_c{conc}.sqlite",
        ))
    return reports


# ---------------------------------------------------------------------------
# Stage 2: export to SQLite
# ---------------------------------------------------------------------------
def export_sqlite(rep: Report, nsys_bin: str, force: bool) -> None:
    out = rep.sqlite_path
    if out.exists() and not force and out.stat().st_mtime >= rep.rep_path.stat().st_mtime:
        print(f"  [skip] {out.name} is up to date")
        return
    if out.exists():
        out.unlink()
    print(f"  [export] {rep.rep_path.name} -> {out.name}")
    proc = subprocess.run(
        [nsys_bin, "export", "--type=sqlite", "--force-overwrite", "true",
         "-o", str(out), str(rep.rep_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if proc.returncode != 0 or not out.exists():
        sys.exit(f"nsys export failed for {rep.rep_path}:\n{proc.stdout}")


# ---------------------------------------------------------------------------
# Stage 3: SQL queries
# ---------------------------------------------------------------------------
def _scalar(cur, sql):
    row = cur.execute(sql).fetchone()
    return row[0] if row and row[0] is not None else 0


def _merged_busy_ns(intervals: list[tuple[int, int]]) -> int:
    """Union length of [start,end) intervals -> true GPU-active time."""
    if not intervals:
        return 0
    intervals.sort()
    busy = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    return busy


def analyze(rep: Report) -> Analysis:
    con = sqlite3.connect(rep.sqlite_path)
    cur = con.cursor()
    a = Analysis(engine=rep.engine, conc=rep.conc)

    def table_exists(t):
        return _scalar(cur, f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='{t}'") > 0

    has_kernel = table_exists("CUPTI_ACTIVITY_KIND_KERNEL")
    has_graph = table_exists("CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    has_memcpy = table_exists("CUPTI_ACTIVITY_KIND_MEMCPY")

    # GPU device info (derived, not hardcoded).
    if table_exists("TARGET_INFO_GPU"):
        row = cur.execute("SELECT name, smCount FROM TARGET_INFO_GPU LIMIT 1").fetchone()
        if row:
            a.gpu_name, a.sm_count = row[0], row[1]

    # ----- gather all GPU intervals for wall span + busy union -----
    intervals: list[tuple[int, int]] = []
    starts, ends = [], []
    if has_kernel:
        for s, e in cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            intervals.append((s, e)); starts.append(s); ends.append(e)
    if has_graph:
        for s, e in cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
            intervals.append((s, e)); starts.append(s); ends.append(e)
    if has_memcpy:
        for s, e in cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            intervals.append((s, e)); starts.append(s); ends.append(e)

    if starts:
        a.wall_ms = (max(ends) - min(starts)) / 1e6
        a.gpu_busy_ms = _merged_busy_ns(intervals) / 1e6
        a.gpu_util_pct = 100.0 * a.gpu_busy_ms / a.wall_ms if a.wall_ms else 0.0

    # ----- CUDA graph replays (each = one graphed decode model-forward) -----
    if has_graph:
        a.graph_exec = _scalar(cur, "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        a.graph_ms = _scalar(cur, "SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE") / 1e6
        a.graph_mean_ms = (a.graph_ms / a.graph_exec) if a.graph_exec else 0.0

    # ----- eager kernels: per-category time -----
    cats: dict[str, dict] = {}
    if has_kernel:
        q = """SELECT s.value, k.end - k.start
               FROM CUPTI_ACTIVITY_KIND_KERNEL k
               JOIN StringIds s ON k.shortName = s.id"""
        for name, dur in cur.execute(q):
            cat = categorize(name or "")
            c = cats.setdefault(cat, {"n": 0, "ns": 0})
            c["n"] += 1
            c["ns"] += dur
        a.eager_kernels = _scalar(cur, "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")
        a.eager_ms = _scalar(cur, "SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL") / 1e6
        a.eager_mean_us = (a.eager_ms * 1000.0 / a.eager_kernels) if a.eager_kernels else 0.0

    # ----- memcpy traffic -----
    if has_memcpy:
        a.memcpy_count = _scalar(cur, "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        a.memcpy_ms = _scalar(cur, "SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY") / 1e6
        a.h2d_mb = _scalar(cur, "SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1") / 1e6
        a.d2h_mb = _scalar(cur, "SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=2") / 1e6
        a.d2d_mb = _scalar(cur, "SELECT SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=8") / 1e6

    # ----- launch overhead: count CUDA kernel-launch API calls on the CPU -----
    if table_exists("CUPTI_ACTIVITY_KIND_RUNTIME"):
        a.launch_api_calls = _scalar(cur, """
            SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON r.nameId = s.id
            WHERE s.value LIKE 'cu%LaunchKernel%'""")
        a.launch_per_ms = (a.launch_api_calls / a.wall_ms) if a.wall_ms else 0.0

    # ----- assemble category map: eager cats + Memory + CUDA Graph -----
    if a.memcpy_ms > 0:
        cats[MEMORY_CATEGORY] = {"n": a.memcpy_count, "ns": int(a.memcpy_ms * 1e6)}
    if a.graph_ms > 0:
        cats[GRAPH_CATEGORY] = {"n": a.graph_exec, "ns": int(a.graph_ms * 1e6)}
    a.categories = cats

    a.total_work_ms = sum(c["ns"] for c in cats.values()) / 1e6
    a.graph_pct = (100.0 * a.graph_ms / a.total_work_ms) if a.total_work_ms else 0.0

    con.close()
    return a


# ---------------------------------------------------------------------------
# Stage 4: CSV
# ---------------------------------------------------------------------------
def write_breakdown_csv(analyses: list[Analysis], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "concurrency", "category", "n_kernels",
                    "total_ms", "pct_of_gpu_work"])
        for a in analyses:
            for cat in CATEGORY_ORDER:
                c = a.categories.get(cat)
                if not c:
                    continue
                ms = c["ns"] / 1e6
                pct = 100.0 * ms / a.total_work_ms if a.total_work_ms else 0.0
                w.writerow([a.engine, a.conc, cat, c["n"],
                            f"{ms:.4f}", f"{pct:.4f}"])
    print(f"  [csv] {path}")


def write_summary_csv(analyses: list[Analysis], path: Path) -> None:
    cols = ["engine", "concurrency", "gpu_name", "sm_count", "wall_ms",
            "gpu_busy_ms", "gpu_util_pct", "total_gpu_work_ms",
            "graph_replays", "graph_ms", "graph_mean_ms", "graph_pct_of_work",
            "eager_kernels", "eager_kernel_ms", "eager_mean_us",
            "launch_api_calls", "launches_per_ms",
            "memcpy_count", "memcpy_ms", "h2d_mb", "d2h_mb", "d2d_mb"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for a in analyses:
            w.writerow([
                a.engine, a.conc, a.gpu_name, a.sm_count,
                f"{a.wall_ms:.3f}", f"{a.gpu_busy_ms:.3f}", f"{a.gpu_util_pct:.2f}",
                f"{a.total_work_ms:.3f}", a.graph_exec, f"{a.graph_ms:.3f}",
                f"{a.graph_mean_ms:.4f}", f"{a.graph_pct:.2f}",
                a.eager_kernels, f"{a.eager_ms:.3f}", f"{a.eager_mean_us:.3f}",
                a.launch_api_calls, f"{a.launch_per_ms:.4f}",
                a.memcpy_count, f"{a.memcpy_ms:.3f}",
                f"{a.h2d_mb:.4f}", f"{a.d2h_mb:.4f}", f"{a.d2d_mb:.4f}",
            ])
    print(f"  [csv] {path}")


# ---------------------------------------------------------------------------
# Stage 5: charts (read straight from the CSVs)
# ---------------------------------------------------------------------------
def _read_breakdown(path: Path):
    rows = list(csv.DictReader(path.open()))
    engines, concs = [], []
    data: dict = {}
    for r in rows:
        eng, c = r["engine"], int(r["concurrency"])
        if eng not in engines:
            engines.append(eng)
        if c not in concs:
            concs.append(c)
        data[(eng, c, r["category"])] = float(r["pct_of_gpu_work"])
    return engines, sorted(concs), data


def _read_summary(path: Path):
    return {(r["engine"], int(r["concurrency"])): r
            for r in csv.DictReader(path.open())}


def chart_breakdown(breakdown_csv: Path, conc: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    engines, concs, data = _read_breakdown(breakdown_csv)
    if conc not in concs:
        return
    cats = [c for c in CATEGORY_ORDER
            if any((e, conc, c) in data for e in engines)]
    x = np.arange(len(cats))
    n = len(engines)
    width = 0.8 / max(n, 1)
    colors = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(max(9, 1.25 * len(cats)), 5.5))
    for i, eng in enumerate(engines):
        vals = [data.get((eng, conc, c), 0.0) for c in cats]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width,
                      label=eng, color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            if v >= 0.5:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of total GPU work time (decode window)")
    ax.set_title(f"Decode-phase GPU-time composition  (concurrency = {conc})\n"
                 f"vLLM wraps the forward pass in CUDA graphs; sglang runs it eagerly")
    ax.legend(title="engine")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  [chart] {out_png}")


def chart_summary(summary_csv: Path, conc: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    summ = _read_summary(summary_csv)
    rows = [(e, c) for (e, c) in summ if c == conc]
    if not rows:
        return
    engines = sorted({e for e, _ in rows})

    metrics = [
        ("GPU utilisation", "gpu_util_pct", "%"),
        ("CUDA-graph share\nof GPU work", "graph_pct_of_work", "%"),
        ("Kernel-launch rate", "launches_per_ms", "launches/ms"),
        ("Mean eager\nkernel duration", "eager_mean_us", "us"),
        ("Mean CUDA-graph\nreplay", "graph_mean_ms", "ms/replay"),
    ]
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 4.4))
    for ax, (title, key, unit) in zip(axes, metrics):
        vals = [float(summ[(e, conc)][key]) for e in engines]
        bars = ax.bar(engines, vals, color=[colors[i % len(colors)]
                                            for i in range(len(engines))])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=8)
        top = max(vals) if vals else 1.0
        ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Decode-phase engine signals  (concurrency = {conc})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  [chart] {out_png}")


# ---------------------------------------------------------------------------
# stdout report
# ---------------------------------------------------------------------------
def print_report(analyses: list[Analysis], focus_conc: int) -> None:
    by_conc: dict[int, list[Analysis]] = {}
    for a in analyses:
        by_conc.setdefault(a.conc, []).append(a)

    order = ([focus_conc] if focus_conc in by_conc else []) + \
            sorted(c for c in by_conc if c != focus_conc)
    for c in order:
        group = by_conc[c]
        print(f"\n{'='*78}\n DECODE ANALYSIS @ concurrency={c}"
              f"   GPU={group[0].gpu_name} ({group[0].sm_count} SMs)\n{'='*78}")
        hdr = f"{'metric':32}" + "".join(f"{a.engine:>14}" for a in group)
        print(hdr)
        print("-" * len(hdr))

        def line(label, fn):
            print(f"{label:32}" + "".join(f"{fn(a):>14}" for a in group))

        line("wall span (ms)", lambda a: f"{a.wall_ms:.1f}")
        line("GPU busy / union (ms)", lambda a: f"{a.gpu_busy_ms:.1f}")
        line("GPU utilisation (%)", lambda a: f"{a.gpu_util_pct:.1f}")
        line("CUDA-graph replays", lambda a: f"{a.graph_exec}")
        line("  mean replay (ms)", lambda a: f"{a.graph_mean_ms:.2f}")
        line("  graph % of GPU work", lambda a: f"{a.graph_pct:.1f}")
        line("eager kernels", lambda a: f"{a.eager_kernels}")
        line("  mean eager kernel (us)", lambda a: f"{a.eager_mean_us:.1f}")
        line("kernel launches/ms", lambda a: f"{a.launch_per_ms:.2f}")
        line("memcpy (MB H2D/D2H)", lambda a: f"{a.h2d_mb:.1f}/{a.d2h_mb:.1f}")
        print("  -- GPU-time composition (% of GPU work) --")
        for cat in CATEGORY_ORDER:
            if any(cat in a.categories for a in group):
                line("  " + cat, lambda a, cat=cat: (
                    f"{100*a.categories[cat]['ns']/1e6/a.total_work_ms:.1f}"
                    if cat in a.categories and a.total_work_ms else "-"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", type=Path,
                    default=here / "nsight-profiles",
                    help="directory containing <engine>_c<N>.nsys-rep files")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output root (default: <profile-dir>/analysis)")
    ap.add_argument("--focus-conc", type=int, default=32,
                    help="concurrency level to analyse first / chart as primary")
    ap.add_argument("--nsys", default=shutil.which("nsys") or "nsys",
                    help="path to nsys binary")
    ap.add_argument("--force-export", action="store_true",
                    help="re-export SQLite even if up to date")
    args = ap.parse_args()

    profile_dir = args.profile_dir.resolve()
    if not profile_dir.is_dir():
        sys.exit(f"profile dir not found: {profile_dir}")
    out_dir = (args.out_dir or profile_dir / "analysis").resolve()
    sqlite_dir = out_dir / "sqlite"
    csv_dir = out_dir / "csv"
    chart_dir = out_dir / "charts"
    for d in (sqlite_dir, csv_dir, chart_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Discovering profiles in {profile_dir}")
    reports = discover_reports(profile_dir, sqlite_dir)
    if not reports:
        sys.exit("no '<engine>_c<N>.nsys-rep' files found")
    for r in reports:
        print(f"      found {r.engine:>8}  c={r.conc:<4} {r.rep_path.name}")

    print(f"[2/5] Exporting to SQLite ({args.nsys})")
    for r in reports:
        export_sqlite(r, args.nsys, args.force_export)

    print("[3/5] Running decode-phase SQL queries")
    analyses = [analyze(r) for r in reports]
    # Order: focus concurrency first, then the rest -- "c=32 first for both".
    analyses.sort(key=lambda a: (a.conc != args.focus_conc, a.conc, a.engine))

    print("[4/5] Writing CSVs")
    breakdown_csv = csv_dir / "gpu_time_breakdown.csv"
    summary_csv = csv_dir / "decode_summary.csv"
    write_breakdown_csv(analyses, breakdown_csv)
    write_summary_csv(analyses, summary_csv)

    print("[5/5] Rendering charts")
    concs = sorted({a.conc for a in analyses})
    chart_order = ([args.focus_conc] if args.focus_conc in concs else []) + \
                  [c for c in concs if c != args.focus_conc]
    for c in chart_order:
        chart_breakdown(breakdown_csv, c, chart_dir / f"breakdown_c{c}.png")
        chart_summary(summary_csv, c, chart_dir / f"summary_c{c}.png")

    print_report(analyses, args.focus_conc)
    print(f"\nDone. Outputs under: {out_dir}")


if __name__ == "__main__":
    main()
