import sqlite3
from collections import defaultdict

def sid(db):
    return {r[0]: r[1] for r in db.execute("select id,value from StringIds")}

def analyze(path, label):
    db = sqlite3.connect(path)
    S = sid(db)
    print(f"\n{'='*72}\n{label}\n{'='*72}")

    krows = list(db.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    t0 = krows[0][0]; t1 = max(r[1] for r in krows)

    # merged busy intervals on device 0 -> gaps
    ivs = sorted((s,e) for s,e,d in krows if d==0)
    merged=[]
    for s,e in ivs:
        if merged and s<=merged[-1][1]:
            merged[-1]=(merged[-1][0],max(merged[-1][1],e))
        else:
            merged.append([s,e])
    gaps=[]
    for i in range(1,len(merged)):
        g=merged[i][0]-merged[i-1][1]
        if g>3000:
            gaps.append((merged[i-1][1], merged[i][0], g))

    # gap histogram by size bucket
    buckets=defaultdict(lambda:[0,0])
    for _,_,g in gaps:
        if g<1e3: k="<1us"
        elif g<10e3: k="3-10us"
        elif g<100e3: k="10-100us"
        elif g<1e6: k="0.1-1ms"
        else: k=">1ms"
        buckets[k][0]+=1; buckets[k][1]+=g
    print("dev0 gap buckets (count, total ms):")
    for k in ["3-10us","10-100us","0.1-1ms",">1ms"]:
        if k in buckets:
            print(f"   {k:>10}: n={buckets[k][0]:5d}  total={buckets[k][1]/1e6:8.2f} ms")
    total_gap=sum(g for _,_,g in gaps)
    print(f"   total gap (>3us) on dev0: {total_gap/1e6:.2f} ms over {(t1-t0)/1e6:.2f} ms span")

    # CUDA runtime API calls: total time by name (CPU side)
    print("\n top CUDA runtime API calls by total wall time (all threads):")
    q="""select s.value as name, count(*) c, sum(r.end-r.start)/1e6 tot_ms, avg(r.end-r.start)/1e3 avg_us
         from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
         group by r.nameId order by tot_ms desc limit 12"""
    for name,c,tot,avg in db.execute(q):
        print(f"   {name[:42]:42s} n={c:6d} tot={tot:8.2f}ms avg={avg:8.1f}us")

    # synchronization events
    print("\n CUPTI synchronization events by type:")
    try:
        q2="""select e.value, count(*), sum(y.end-y.start)/1e6
              from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION y
              join ENUM_CUPTI_SYNC_TYPE e on e.id=y.syncType group by y.syncType order by 3 desc"""
        for nm,c,tot in db.execute(q2):
            print(f"   {nm:30s} n={c:6d} tot={tot:8.2f}ms")
    except Exception as ex:
        print("   (none)",ex)
    db.close()

for p,l in [("vllm_b2.sqlite","VLLM TP=2 b2"),("sglang_b2.sqlite","SGLANG TP=2 b2")]:
    analyze(p,l)
