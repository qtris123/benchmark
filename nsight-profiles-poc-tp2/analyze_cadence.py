import sqlite3
def ev(db,name):
    q="""select r.start,r.end,r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r
         join StringIds s on s.id=r.nameId where s.value like ? order by r.start"""
    return list(db.execute(q,(name,)))

def analyze(path,label):
    db=sqlite3.connect(path)
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    krows=list(db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=0 order by start"))
    t0=krows[0][0]
    gl=ev(db,"cudaGraphLaunch%")
    print(f"cudaGraphLaunch count={len(gl)} (rank0+rank1 mixed). cadence on first thread:")
    # pick most common thread
    from collections import Counter
    tid=Counter(g[2] for g in gl).most_common(1)[0][0]
    g0=[g for g in gl if g[2]==tid]
    print(f" rank tid={tid&0xffffff}: {len(g0)} launches")
    prev=None
    for i,(s,e,_) in enumerate(g0):
        gap=(s-prev)/1e6 if prev else 0
        print(f"  launch {i:2d} @+{(s-t0)/1e6:8.2f}ms dur={(e-s)/1e6:6.2f}ms gap_since_prev={gap:7.2f}ms")
        prev=e
    # cudaEventSynchronize on the same thread / all
    es=ev(db,"cudaEventSynchronize%")
    print(f"\ncudaEventSynchronize: {len(es)} total")
    byt=Counter(x[2] for x in es)
    for t,c in byt.items():
        sub=[x for x in es if x[2]==t]
        tot=sum(e-s for s,e,_ in sub)/1e6
        print(f"  tid={t&0xffffff}: n={c} tot={tot:.2f}ms  (durs ms: {[round((e-s)/1e6,2) for s,e,_ in sub[:12]]})")
    db.close()

analyze("vllm_b2.sqlite","VLLM TP=2 b2")
analyze("sglang_b2.sqlite","SGLANG TP=2 b2")
