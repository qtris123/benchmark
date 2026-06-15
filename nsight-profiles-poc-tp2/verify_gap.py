import sqlite3
db=sqlite3.connect("vllm_b2.sqlite")

# thread names
print("=== thread names (process/thread) ===")
for tid,name in db.execute("""select t.globalTid, s.value from ThreadNames t join StringIds s on s.id=t.nameId"""):
    print(f"  tid={tid&0xffffff:>8}  pid={(tid>>24)&0xffffff:>8}  name={name}")

# choose a representative MID gap (not the trailing profilerStop). Use the 19ms / 8.8ms region.
krows=list(db.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
t0=krows[0][0]
ivs=sorted((s,e) for s,e,d in krows if d==0)
merged=[]
for s,e in ivs:
    if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
    else: merged.append([s,e])
gaps=[(merged[i-1][1],merged[i][0],merged[i][0]-merged[i-1][1]) for i in range(1,len(merged))]
# exclude the final huge profilerStop gap; pick the 19ms one
gaps=[g for g in gaps if 5e6 < g[2] < 30e6]
gaps.sort(key=lambda x:-x[2])
gs,ge,glen=gaps[0]
print(f"\n=== representative mid gap: +{(gs-t0)/1e6:.1f}ms  len={glen/1e6:.2f}ms ===")

# ANY GPU activity (kernel/memcpy/memset) on EITHER device fully inside the gap?
for tbl in ["CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_MEMSET"]:
    n=db.execute(f"select count(*) from {tbl} where start>=? and end<=?",(gs,ge)).fetchone()[0]
    print(f"  GPU ops in {tbl.split('_')[-1]:8s} fully inside gap: {n}")

# CUDA API (host) calls that span the gap, per thread
print("\n  host CUDA-API calls overlapping the gap (by coverage):")
q="""select s.value, r.start, r.end, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r
     join StringIds s on s.id=r.nameId where r.start<? and r.end>? order by (min(r.end,?)-max(r.start,?)) desc limit 8"""
for nm,rs,re,tid in db.execute(q,(ge,gs,ge,gs)):
    cov=(min(re,ge)-max(rs,gs))/1e6
    print(f"    tid={tid&0xffffff:>8} {nm[:34]:34s} apidur={(re-rs)/1e6:7.2f}ms covers={cov:6.2f}ms of {glen/1e6:.1f}ms gap")

# how much of the gap is covered by SOME host CUDA-API call on the two worker main threads?
print("\n  fraction of gap with NO host CUDA-API call active (i.e. pure CPU/host work or block w/o API):")
api=list(db.execute("select start,end from CUPTI_ACTIVITY_KIND_RUNTIME where start<? and end>?",(ge,gs)))
# union coverage
iv=sorted((max(s,gs),min(e,ge)) for s,e in api)
m=[]
for s,e in iv:
    if m and s<=m[-1][1]: m[-1][1]=max(m[-1][1],e)
    else: m.append([s,e])
cov=sum(e-s for s,e in m)
print(f"    gap={glen/1e6:.2f}ms  covered-by-some-CUDA-API={cov/1e6:.2f}ms  uncovered={ (glen-cov)/1e6:.2f}ms")
db.close()
