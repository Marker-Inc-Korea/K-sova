"""v3 평가 종합 — test300 4축 / 범용 / 지식프로브 / KorNAT을 모델별 조건비교표로."""
import json, re, csv
from pathlib import Path
from collections import defaultdict
GEN=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")
G=GEN/"runs/gen_v3"
MODELS=["exaone","qwen","swallow","llama","gemma"]
def norm(s): return re.sub(r"\s+","",(s or "")).lower()
def rd(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()] if Path(p).exists() else []

# instruction→tier
tier={}
for r in rd(GEN/"test_v3_eval300.jsonl"): tier[norm(r["messages"][0]["content"])[:60]]=r.get("tier")
probe={r["qid"]:r for r in rd(GEN/"knowledge_probe.jsonl")}

def mean(xs): xs=[x for x in xs if isinstance(x,(int,float))]; return sum(xs)/len(xs) if xs else None
def f(x): return f"{x:.3f}" if isinstance(x,(int,float)) else " - "

print("="*78); print("① 소버린 품질 (test300, Kimi 4축) — 조건별 평균")
print(f"{'model':8s}{'cond':14s}{'sov_align':>10}{'factual':>9}{'reason':>9}{'direct':>9}")
sov_tbl=defaultdict(dict)
for m in MODELS:
    for c in ["base","prompt","prompt_strong","lora"]:
        rows=rd(G/f"jt_{m}_{c}.jsonl")
        if not rows: continue
        J=[r["judge"] for r in rows if isinstance(r.get("judge"),dict)]
        sa=mean([j.get("sovereign_alignment") for j in J]); fa=mean([j.get("factual") for j in J])
        re_=mean([j.get("reasoning") for j in J]); di=mean([j.get("directness") for j in J])
        sov_tbl[m][c]=(sa,fa,re_,di)
        print(f"{m:8s}{c:14s}{f(sa):>10}{f(fa):>9}{f(re_):>9}{f(di):>9}")
    # base→lora 델타
    if "base" in sov_tbl[m] and "lora" in sov_tbl[m]:
        b,l=sov_tbl[m]["base"],sov_tbl[m]["lora"]
        d=lambda i: (l[i]-b[i]) if isinstance(b[i],(int,float)) and isinstance(l[i],(int,float)) else None
        print(f"{'':8s}{'Δ(lora-base)':14s}{f(d(0)):>10}{f(d(1)):>9}{f(d(2)):>9}{f(d(3)):>9}")
    print()

print("="*78); print("① -tier별 (lora) sov_align / factual")
print(f"{'model':8s}{'fact(sa/fa)':>16}{'stance(sa/fa)':>16}{'balance(sa/fa)':>16}")
for m in MODELS:
    rows=rd(G/f"jt_{m}_lora.jsonl")
    if not rows: continue
    byt=defaultdict(lambda:([],[]))
    for r in rows:
        t=tier.get(norm(r.get("instruction",""))[:60]); j=r.get("judge",{})
        if t and isinstance(j,dict):
            byt[t][0].append(j.get("sovereign_alignment")); byt[t][1].append(j.get("factual"))
    cell=lambda t: f"{f(mean(byt[t][0]))}/{f(mean(byt[t][1]))}" if t in byt else " - "
    print(f"{m:8s}{cell('fact'):>16}{cell('stance'):>16}{cell('balance'):>16}")

print(); print("="*78); print("② 범용지식 유지 (general_eval, base vs lora)")
print(f"{'model':8s}{'correct b→l':>16}{'fluency b→l':>16}{'complete b→l':>16}")
for m in MODELS:
    def gg(c,k):
        rows=rd(G/f"jg_{m}_{c}.jsonl"); return mean([r.get("judge",{}).get(k) for r in rows if isinstance(r.get("judge"),dict)])
    cor=f"{f(gg('base','correctness'))}→{f(gg('lora','correctness'))}"
    flu=f"{f(gg('base','fluency'))}→{f(gg('lora','fluency'))}"
    com=f"{f(gg('base','completeness'))}→{f(gg('lora','completeness'))}"
    if rd(G/f"jg_{m}_base.jsonl"): print(f"{m:8s}{cor:>16}{flu:>16}{com:>16}")

print(); print("="*78); print("③ 지식 프로브 (꼭 암기할 주권사실, base vs lora) — 정답키워드 포함율")
print(f"{'model':8s}{'현재사실 b→l':>18}{'상식 b→l':>16}")
for m in MODELS:
    def acc(c,kind):
        rows={r.get("qid"):r.get("generated","") for r in rd(G/f"kp_{m}_{c}.jsonl")}
        ok=tot=0
        for qid,pr in probe.items():
            if pr["grade"]!="keyword" or pr["kind"]!=kind: continue
            g=norm(rows.get(qid,"")); tot+=1; ok+=any(norm(k) in g for k in pr["answer_keywords"])
        return ok/tot if tot else None
    cur=f"{f(acc('base','current'))}→{f(acc('lora','current'))}"
    dur=f"{f(acc('base','durable'))}→{f(acc('lora','durable'))}"
    if rd(G/f"kp_{m}_base.jsonl"): print(f"{m:8s}{cur:>18}{dur:>16}")

print(); print("="*78); print("④ KorNAT SVA (analysis/kornat_official_metrics.csv)")
csvp=GEN/"analysis/kornat_official_metrics.csv"
if csvp.exists():
    for row in csv.DictReader(open(csvp,encoding="utf-8")):
        sva=row.get("SVA","")
        if sva: print(f"  {row.get('model',''):8s} {row.get('cond',''):14s} SVA={sva}")
else: print("  (KorNAT 점수 아직)")
print("="*78)
