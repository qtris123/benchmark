import sqlite3
def dev_idle(path):
    db=sqlite3.connect(path)
    out={}
    for dev, in db.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL"):
        ivs=sorted((s,e) for s,e in db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=? order by start",(dev,)))
        merged=[]
        for s,e in ivs:
            if merged and s<=merged[-1][1]: merged[-1][1]=max(merged[-1][1],e)
            else: merged.append([s,e])
        busy=sum(e-s for s,e in merged); span=merged[-1][1]-merged[0][0]
        # biggest internal gaps
        gaps=sorted((merged[i][0]-merged[i-1][1] for i in range(1,len(merged))),reverse=True)[:4]
        out[dev]=(span/1e6, busy/1e6, 100*(span-busy)/span, [round(g/1e6,1) for g in gaps], len(ivs))
    db.close(); return out

for p,l in [("vllm_b2.sqlite","vLLM  b2 "),("sglang_b2.sqlite","SGLang b2 "),
            ("vllm_b32.sqlite","vLLM  b32"),("sglang_b32.sqlite","SGLang b32")]:
    o=dev_idle(p)
    for dev,(span,busy,idle,gaps,n) in o.items():
        print(f"{l} dev{dev}: span={span:7.1f}ms busy={busy:7.1f}ms IDLE={idle:5.1f}%  nkern={n:5d} topgaps_ms={gaps}")
    print()
