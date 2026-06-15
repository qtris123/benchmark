import sqlite3, sys
from collections import defaultdict

def sid(db):
    return {r[0]: r[1] for r in db.execute("select id,value from StringIds")}

def analyze(path, label):
    db = sqlite3.connect(path)
    S = sid(db)
    print(f"\n{'='*70}\n{label}  ({path})\n{'='*70}")

    # GPU kernels (includes graph-node kernels)
    rows = list(db.execute("""
        select start,end,deviceId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start
    """))
    if not rows:
        print("no kernels"); return
    t0 = rows[0][0]; t1 = max(r[1] for r in rows)
    span = t1 - t0
    # per-device busy + gap analysis on union of intervals (per device)
    by_dev = defaultdict(list)
    for s,e,dev,strm in rows:
        by_dev[dev].append((s,e))
    print(f"profiling span (first kernel start -> last kernel end): {span/1e6:.2f} ms")
    print(f"total kernels: {len(rows)}  devices: {sorted(by_dev)}")

    for dev in sorted(by_dev):
        ivs = sorted(by_dev[dev])
        # merge overlapping intervals
        merged=[]
        for s,e in ivs:
            if merged and s<=merged[-1][1]:
                merged[-1]=(merged[-1][0],max(merged[-1][1],e))
            else:
                merged.append((s,e))
        busy=sum(e-s for s,e in merged)
        d0=merged[0][0]; d1=merged[-1][1]; dspan=d1-d0
        # gaps between merged intervals
        gaps=[]
        for i in range(1,len(merged)):
            g = merged[i][0]-merged[i-1][1]
            if g>0:
                gaps.append((merged[i-1][1], merged[i][0], g))
        gaps.sort(key=lambda x:-x[2])
        idle = dspan-busy
        print(f"\n  device {dev}: span={dspan/1e6:.2f}ms busy={busy/1e6:.2f}ms idle={idle/1e6:.2f}ms ({100*idle/dspan:.1f}%) nkernels={len(ivs)}")
        print(f"    #gaps>5us={sum(1 for g in gaps if g[2]>5000)}  top gaps (us):", [round(g[2]/1000,1) for g in gaps[:8]])
        # for the largest gaps, find CUDA API runtime calls overlapping the gap window on this process
        return_gaps = gaps[:5]
    db.close()

for p,l in [("vllm_b2.sqlite","VLLM TP=2 b2"),("sglang_b2.sqlite","SGLANG TP=2 b2")]:
    analyze(p,l)
