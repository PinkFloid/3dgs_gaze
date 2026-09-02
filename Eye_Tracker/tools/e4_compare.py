"""Side-by-side E4 hit counts per recording: v1 frozen vs e4 configs (CPU only)."""
import csv, os, sys
from collections import Counter
R='/home/liuchy/recordings'
RECS=[('2026_08_16/000','e1'),('2026_08_16/001','e2'),('2026_08_16/s1','s1'),('2026_08_16/s2','s2'),('2026_08_16/s3','s3'),('2026_08_16/s4','s4'),('2026_08_16/s6','s6'),('2026_08_18/000','s7'),('2026_08_20/000','c1'),('2026_08_20/001','c2'),('2026_08_20/002','c4'),('2026_08_20/003','c4'),('2026_08_25/c1_1','c4'),('2026_08_25/c1_2','c4'),('2026_08_25/c1_3','c4'),('2026_08_25/u3','u3')]
CFGS=sys.argv[1:] or ['v2','v2mass','v2selfcal']
HIT={'✓':1,'✓✓':2,'✓(-1)':1}
def load(p): return list(csv.DictReader(open(p,encoding='utf-8'))) if os.path.exists(p) else None
def summ(rows):
    h=sum(HIT.get(r['verdict'],0) for r in rows); miss=sum(r['verdict']=='✗缺失' for r in rows)
    extra=sum(r['verdict']=='＋多余' for r in rows); n=h+miss
    return h,n,extra
tot={c:[0,0,0] for c in ['v1']+CFGS}
print(f"{'录像':16} {'卡':3} " + " ".join(f"{c:>14}" for c in ['v1']+CFGS))
for rec,card in RECS:
    d=f"{R}/{rec}"; cells=[]
    for c in ['v1']+CFGS:
        rows=load(f"{d}/{card}_score.csv" if c=='v1' else f"{d}/e4_{c}_score.csv")
        if rows is None: cells.append('…'); continue
        h,n,x=summ(rows); tot[c][0]+=h; tot[c][1]+=n; tot[c][2]+=x
        cells.append(f"{h}/{n} +{x}")
    print(f"{rec:16} {card:3} " + " ".join(f"{s:>14}" for s in cells))
print(f"{'合计(已出的)':16} {'':3} " + " ".join(f"{t[0]}/{t[1]} +{t[2]:>3}".rjust(14) for c,t in tot.items()))
