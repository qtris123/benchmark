#!/usr/bin/env python3
"""
Compare + delta plots for the TP=2 "-cpu-enable" nsys runs (vLLM vs SGLang).

Reads vllm_decode_b<B>_i<ISL>_tp2.nsys-rep and sglang_decode_b<B>_i<ISL>_tp2.nsys-rep
from the cpu-enable profile dir, exports each to SQLite, derives GPU + CPU(host)
decode-window metrics, and renders:
  * COMPARE.png        grouped bars (vLLM vs SGLang) per metric, faceted by batch.
  * COMPARE_delta.png  (vLLM - SGLang) per metric per batch.

GPU metrics quantify the "gaps" we investigated; the CPU/OSRT metric is the new
host-side signal enabled by --trace=...,osrt (off-CPU blocking on worker threads).
Nothing hardcoded except the decode kernel-name categories are not needed here.
"""
from __future__ import annotations
import argparse, re, sqlite3, subprocess, sys, shutil
from pathlib import Path

REP_RE = re.compile(r"^(?P<engine>vllm|sglang)_decode_b(?P<b>\d+)_i(?P<isl>\d+)_tp(?P<tp>\d+)\.nsys-rep$")


def export_sqlite(rep: Path, sq: Path, nsys: str) -> None:
    if sq.exists() and sq.stat().st_mtime >= rep.stat().st_mtime:
        return
    if sq.exists():
        sq.unlink()
    print(f"  [export] {rep.name}")
    p = subprocess.run([nsys, "export", "--type=sqlite", "--force-overwrite", "true",
                        "-o", str(sq), str(rep)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 or not sq.exists():
        sys.exit(f"export failed for {rep}:\n{p.stdout[-2000:]}")


def _merge(iv):
    iv = sorted(iv); m = []
    for s, e in iv:
        if m and s <= m[-1][1]:
            m[-1][1] = max(m[-1][1], e)
        else:
            m.append([s, e])
    return m


def metrics(sq: Path) -> dict:
    db = sqlite3.connect(sq)
    S = {r[0]: r[1] for r in db.execute("select id,value from StringIds")}

    def has(t):
        return db.execute("select count(*) from sqlite_master where type='table' and name=?", (t,)).fetchone()[0] > 0

    kr = list(db.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    devs = sorted({d for _, _, d in kr})
    win_lo = min(s for s, _, _ in kr)
    win_hi = max(e for _, e, _ in kr)

    # cudaProfilerStop window (to exclude the trailing trace-flush gap from idle).
    ps = list(db.execute("""select r.start,r.end from CUPTI_ACTIVITY_KIND_RUNTIME r
                            join StringIds s on s.id=r.nameId where s.value like 'cudaProfilerStop%'"""))
    util, idle_ms = [], []
    for dev in devs:
        m = _merge((s, e) for s, e, d in kr if d == dev)
        busy = sum(e - s for s, e in m)
        gaps = [(m[i - 1][1], m[i][0], m[i][0] - m[i - 1][1]) for i in range(1, len(m))]
        # drop gaps overlapping a cudaProfilerStop (the finalize/flush, not decode idle)
        keep = [g for g in gaps if not any(pe > g[0] and ps_s < g[1] for ps_s, pe in ps)]
        idle = sum(g[2] for g in keep)
        util.append(100.0 * busy / (busy + idle) if (busy + idle) else 0.0)
        idle_ms.append(idle / 1e6)
    gpu_util = sum(util) / len(util)
    gpu_idle = sum(idle_ms) / len(idle_ms)

    # compute threads = those that launch CUDA graphs; fall back to cudaLaunchKernel.
    def launch_threads(like):
        return {r[0] for r in db.execute("""select distinct r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r
                join StringIds s on s.id=r.nameId where s.value like ?""", (like,))}
    gl = list(db.execute("""select r.start,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r
             join StringIds s on s.id=r.nameId where s.value like 'cudaGraphLaunch%' order by r.start"""))
    comp_threads = {t for _, t in gl} or launch_threads('cudaLaunchKernel%')

    # median inter-graph-launch interval on the busiest compute thread (per-step cadence).
    from collections import Counter
    med_gap = 0.0
    if gl:
        tid = Counter(t for _, t in gl).most_common(1)[0][0]
        ts = [s for s, t in gl if t == tid]
        diffs = sorted((ts[i] - ts[i - 1]) / 1e6 for i in range(1, len(ts)))
        if diffs:
            med_gap = diffs[len(diffs) // 2]

    # kernel-launch API rate per ms of decode window
    nlaunch = db.execute("""select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
            join StringIds s on s.id=r.nameId where s.value like 'cu%LaunchKernel%'""").fetchone()[0]
    win_ms = (win_hi - win_lo) / 1e6
    launch_per_ms = nlaunch / win_ms if win_ms else 0.0

    # OSRT blocking time on compute threads within the decode window (off-CPU waiting).
    osrt_block_ms = 0.0
    osrt_by_name = {}
    if has("OSRT_API"):
        rows = db.execute("""select s.value,o.start,o.end,o.globalTid from OSRT_API o
                join StringIds s on s.id=o.nameId where o.end> ? and o.start< ?""", (win_lo, win_hi))
        for nm, s, e, tid in rows:
            if tid in comp_threads:
                d = (min(e, win_hi) - max(s, win_lo)) / 1e6
                if d > 0:
                    osrt_block_ms += d
                    osrt_by_name[nm] = osrt_by_name.get(nm, 0.0) + d
    db.close()
    return dict(gpu_util_pct=gpu_util, gpu_idle_ms=gpu_idle, med_launch_gap_ms=med_gap,
                launch_per_ms=launch_per_ms, osrt_block_ms=osrt_block_ms,
                osrt_by_name=osrt_by_name, win_ms=win_ms)


METRICS = [
    ("GPU utilisation",        "gpu_util_pct",       "%",          False),
    ("GPU idle (decode win)",  "gpu_idle_ms",        "ms",         True),
    ("Median launch interval", "med_launch_gap_ms",  "ms",         True),
    ("Kernel-launch rate",     "launch_per_ms",      "launches/ms", False),
    ("Host off-CPU blocked\n(OSRT, compute thr)", "osrt_block_ms", "ms", True),
]


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", type=Path, default=here / "nsight-profiles-poc-tp2-cpu-enable")
    ap.add_argument("--nsys", default=shutil.which("nsys") or "nsys")
    args = ap.parse_args()
    pdir = args.profile_dir.resolve()
    sqdir = pdir / "sqlite"; sqdir.mkdir(exist_ok=True)
    outdir = pdir / "compare"; outdir.mkdir(exist_ok=True)

    data = {}  # (engine,batch) -> metrics
    for rep in sorted(pdir.glob("*.nsys-rep")):
        m = REP_RE.match(rep.name)
        if not m:
            continue
        eng, b = m.group("engine"), int(m.group("b"))
        sq = sqdir / (rep.stem + ".sqlite")
        export_sqlite(rep, sq, args.nsys)
        print(f"  [analyze] {eng} b{b}")
        data[(eng, b)] = metrics(sq)

    if not data:
        sys.exit(f"no matching reports in {pdir}")
    batches = sorted({b for _, b in data})
    engines = [e for e in ("vllm", "sglang") if any(en == e for en, _ in data)]
    print(f"  engines={engines} batches={batches}")

    # ---- console table + osrt breakdown ----
    for (eng, b), mm in sorted(data.items()):
        top = sorted(mm["osrt_by_name"].items(), key=lambda x: -x[1])[:4]
        print(f"\n{eng} b{b}: util={mm['gpu_util_pct']:.1f}%  idle={mm['gpu_idle_ms']:.1f}ms  "
              f"med_launch_gap={mm['med_launch_gap_ms']:.2f}ms  launch/ms={mm['launch_per_ms']:.2f}  "
              f"osrt_block={mm['osrt_block_ms']:.1f}ms")
        if top:
            print("    top OSRT (off-CPU) on compute threads: " +
                  ", ".join(f"{n}={v:.1f}ms" for n, v in top))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    ecolor = {"vllm": "#d62728", "sglang": "#1f77b4"}

    # ---------- COMPARE ----------
    nm = len(METRICS)
    fig, axes = plt.subplots(1, nm, figsize=(3.0 * nm, 4.6))
    x = np.arange(len(batches)); w = 0.38
    for ax, (title, key, unit, _) in zip(axes, METRICS):
        for i, eng in enumerate(engines):
            vals = [data[(eng, b)][key] if (eng, b) in data else 0.0 for b in batches]
            bars = ax.bar(x + (i - (len(engines) - 1) / 2) * w, vals, w,
                          label=eng, color=ecolor.get(eng))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"b{b}" for b in batches])
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.suptitle("TP=2 decode profiling (cpu-enable): vLLM vs SGLang", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p1 = outdir / "COMPARE.png"; fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"\n  [plot] {p1}")

    # ---------- DELTA (vllm - sglang) ----------
    if "vllm" in engines and "sglang" in engines:
        fig, axes = plt.subplots(1, nm, figsize=(3.0 * nm, 4.6))
        for ax, (title, key, unit, lower_better) in zip(axes, METRICS):
            deltas = [data[("vllm", b)][key] - data[("sglang", b)][key] for b in batches]
            colors = []
            for d in deltas:
                bad = (d > 0) if lower_better else (d < 0)  # vLLM worse?
                colors.append("#d62728" if bad else "#2ca02c")
            bars = ax.bar(x, deltas, 0.55, color=colors)
            for bar, d in zip(bars, deltas):
                ax.text(bar.get_x() + bar.get_width() / 2, d, f"{d:+.1f}",
                        ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel(f"Δ {unit}", fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([f"b{b}" for b in batches])
            ax.grid(axis="y", alpha=0.3)
        fig.suptitle("TP=2 decode: Δ (vLLM − SGLang)   red = vLLM worse, green = vLLM better",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p2 = outdir / "COMPARE_delta.png"; fig.savefig(p2, dpi=140); plt.close(fig)
        print(f"  [plot] {p2}")


if __name__ == "__main__":
    main()
