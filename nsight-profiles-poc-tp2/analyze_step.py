import sqlite3
def analyze(path,label,rangefilter):
    db=sqlite3.connect(path)
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    krows=list(db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0 order by start"))
    t0=krows[0][0]
    def busy_between(a,b):
        tot=0
        for s,e in krows:
            if e<=a or s>=b: continue
            tot+=min(e,b)-max(s,a)
        return tot
    # find the main worker thread's per-step ranges (rank0 worker)
    q="""select n.start,n.end, coalesce(s.value,n.text) nm, n.globalTid
         from NVTX_EVENTS n left join StringIds s on s.id=n.textId
         where n.end is not null and coalesce(s.value,n.text) like ? order by n.start"""
    rows=list(db.execute(q,(rangefilter,)))
    # restrict to a single (the most frequent) thread
    from collections import Counter
    tid=Counter(r[3] for r in rows).most_common(1)[0][0]
    rows=[r for r in rows if r[3]==tid]
    print(f"per-step ranges matching '{rangefilter}' on tid={tid&0xffffff}: {len(rows)}")
    print(f"{'#':>3} {'start+ms':>9} {'dur_ms':>7} {'gpu_busy':>9} {'gpu_idle_in':>11} {'gap_to_next':>11}")
    prev_end=None
    for i,(s,e,nm,_) in enumerate(rows):
        dur=(e-s)/1e6
        gb=busy_between(s,e)/1e6
        idle=dur-gb
        gapnext=""
        if prev_end is not None:
            gp=(s-prev_end)/1e6
        prev_end=e
        gnext=(rows[i+1][0]-e)/1e6 if i+1<len(rows) else 0
        print(f"{i:3d} {(s-t0)/1e6:9.2f} {dur:7.2f} {gb:9.2f} {idle:11.2f} {gnext:11.2f}")
    db.close()

analyze("vllm_b2.sqlite","VLLM TP=2 b2","execute_context%")
analyze("sglang_b2.sqlite","SGLANG TP=2 b2","%")  # sglang lacks per-step range; show all? skip
