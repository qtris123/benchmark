import sqlite3
def sid(db): return {r[0]: r[1] for r in db.execute("select id,value from StringIds")}

def cols(db,t): return [r[1] for r in db.execute(f"pragma table_info({t})")]

def analyze(path,label):
    db=sqlite3.connect(path)
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print("NVTX_EVENTS cols:", cols(db,"NVTX_EVENTS"))
    # name resolution: try textId then registered string in 'text'
    q="""select coalesce(s.value, n.text) nm, count(*) c, sum(n.end-n.start)/1e6 tot, avg(n.end-n.start)/1e3 avg_us
         from NVTX_EVENTS n left join StringIds s on s.id=n.textId
         where n.end is not null
         group by nm order by tot desc limit 20"""
    print(f"\n{'NVTX range name':40s} {'count':>6} {'tot_ms':>9} {'avg_us':>9}")
    for nm,c,tot,avg in db.execute(q):
        print(f"{str(nm)[:40]:40s} {c:6d} {tot:9.2f} {avg:9.1f}")
    db.close()

for p,l in [("vllm_b2.sqlite","VLLM TP=2 b2"),("sglang_b2.sqlite","SGLANG TP=2 b2")]:
    analyze(p,l)
