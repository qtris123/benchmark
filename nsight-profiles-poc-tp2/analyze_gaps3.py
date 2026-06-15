import sqlite3
def sid(db): return {r[0]: r[1] for r in db.execute("select id,value from StringIds")}

def analyze(path,label):
    db=sqlite3.connect(path); S=sid(db)
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    krows=list(db.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    t0=krows[0][0]; tlast=max(r[1] for r in krows)
    ivs=sorted((s,e) for s,e,d in krows if d==0)
    merged=[]
    for s,e in ivs:
        if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
        else: merged.append([s,e])
    gaps=[(merged[i-1][1],merged[i][0],merged[i][0]-merged[i-1][1]) for i in range(1,len(merged))]
    gaps=[g for g in gaps if g[2]>1e6]
    gaps.sort()
    print(f"span={(tlast-t0)/1e6:.2f}ms. big gaps(>1ms) on dev0: {len(gaps)}")
    for gs,ge,g in gaps:
        rel=(gs-t0)/1e6
        print(f"\n  GAP @ +{rel:.1f}ms  len={g/1e6:.2f}ms  [{gs} -> {ge}]")
        # runtime API calls overlapping this window, by thread, longest first
        q="""select s.value, r.start, r.end, r.globalTid
             from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
             where r.start < ? and r.end > ? order by (r.end-r.start) desc limit 6"""
        for nm,rs,re,tid in db.execute(q,(ge,gs)):
            cover=(min(re,ge)-max(rs,gs))/1e6
            print(f"      API {nm[:38]:38s} dur={(re-rs)/1e6:7.2f}ms covers={cover:6.2f}ms tid={tid&0xffffff}")
        # NVTX ranges overlapping
        q2="""select s.value, n.start, n.end, n.globalTid
              from NVTX_EVENTS n left join StringIds s on s.id=n.textId
              where n.start < ? and (n.end is null or n.end > ?) and n.end is not null
              order by (n.end-n.start) desc limit 6"""
        for nm,ns,ne,tid in db.execute(q2,(ge,gs)):
            print(f"      NVTX {str(nm)[:38]:38s} dur={(ne-ns)/1e6:7.2f}ms tid={tid&0xffffff}")
    db.close()

for p,l in [("vllm_b2.sqlite","VLLM TP=2 b2"),("sglang_b2.sqlite","SGLANG TP=2 b2")]:
    analyze(p,l)
