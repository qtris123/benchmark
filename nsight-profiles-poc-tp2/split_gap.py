import sqlite3
db=sqlite3.connect("vllm_b2.sqlite")
krows=list(db.execute("select start,end,deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
t0=krows[0][0]
ivs=sorted((s,e) for s,e,d in krows if d==0)
merged=[]
for s,e in ivs:
    if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
    else: merged.append([s,e])
gaps=[(merged[i-1][1],merged[i][0],merged[i][0]-merged[i-1][1]) for i in range(1,len(merged))]
gaps=[g for g in gaps if 5e6<g[2]<30e6]; gaps.sort(key=lambda x:-x[2])
gs,ge,glen=gaps[0]
# execute_model NVTX ranges (resolve via textId)
ranges=list(db.execute("""select n.start,n.end,coalesce(s.value,n.text) from NVTX_EVENTS n
   left join StringIds s on s.id=n.textId where n.end is not null and coalesce(s.value,n.text) like 'execute_context%'"""))
# coverage of gap by execute_model ranges (any rank)
cov=0; iv=[]
for s,e,nm in ranges:
    a,b=max(s,gs),min(e,ge)
    if b>a: iv.append((a,b))
iv.sort(); m=[]
for s,e in iv:
    if m and s<=m[-1][1]: m[-1][1]=max(m[-1][1],e)
    else: m.append([s,e])
covd=sum(e-s for s,e in m)
print(f"gap @+{(gs-t0)/1e6:.1f}ms len={glen/1e6:.2f}ms")
print(f"  covered by an execute_model(host) range : {covd/1e6:6.2f}ms  (host stalling WITHIN a step)")
print(f"  NOT in any execute_model range          : {(glen-covd)/1e6:6.2f}ms  (host time BETWEEN steps / scheduler+IPC)")
db.close()
