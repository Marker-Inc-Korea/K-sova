"""교정 재테스트 종합 — tier분리·대외주권분리·sov/factual분리·빈답/누수율. (gen_v3b + jt2_)"""
import json, re
from pathlib import Path
GEN=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")
G=GEN/"runs/gen_v3b"
MODELS=["exaone","qwen","swallow","llama","gemma"]
CONDS=["base","prompt","prompt_strong","lora"]
TERR=re.compile(r'독도|동해|간도|위안부|강제동원|일본해|시마네|일제|식민|독립운동|동북공정|역사왜곡|영유권|영토')
def norm(s): return re.sub(r"\s+","",(s or "")).lower()
def rd(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()] if Path(p).exists() else []
def mean(xs): xs=[x for x in xs if isinstance(x,(int,float))]; return sum(xs)/len(xs) if xs else None
def f(x): return f"{x:.3f}" if isinstance(x,(int,float)) else "  -  "

# instruction → tier, 대외주권 여부
meta={}
for r in rd(GEN/"test_v3_eval300.jsonl"):
    q=r["messages"][0]["content"]; meta[norm(q)[:60]]={"tier":r.get("tier"),"terr":bool(TERR.search(q))}

def judge_rows(m,c):
    rows=rd(G.parent/"gen_v3b"/f"jt2_{m}_{c}.jsonl")
    out=[]
    for r in rows:
        j=r.get("judge",{}); mt=meta.get(norm(r.get("instruction",""))[:60],{})
        if isinstance(j,dict): out.append({**mt,**j})
    return out

def agg(rows,pred,key): return mean([r.get(key) for r in rows if pred(r)])

print("="*92)
print("교정 재테스트 (test300, decoding 교정·prompt_strong 한국어직답·사고블록 제거)")
print("="*92)
print(f"{'model':7s}{'cond':14s}|{'가치 sov':>9}{'가치 fact':>10}|{'fact_t sov':>11}{'fact_t fac':>11}|{'대외주권 sov':>13}")
for m in MODELS:
    base=None
    for c in CONDS:
        R=judge_rows(m,c)
        if not R: continue
        vsov=agg(R,lambda r:r.get("tier") in ("stance","balance"),"sovereign_alignment")
        vfac=agg(R,lambda r:r.get("tier") in ("stance","balance"),"factual")
        fsov=agg(R,lambda r:r.get("tier")=="fact","sovereign_alignment")
        ffac=agg(R,lambda r:r.get("tier")=="fact","factual")
        tsov=agg(R,lambda r:r.get("terr"),"sovereign_alignment")
        print(f"{m:7s}{c:14s}|{f(vsov):>9}{f(vfac):>10}|{f(fsov):>11}{f(ffac):>11}|{f(tsov):>13}")
        if c=="base": base=(vsov,vfac,fsov,ffac,tsov)
        if c=="lora" and base:
            d=lambda a,b: f"{(a-b):+.3f}" if isinstance(a,(int,float)) and isinstance(b,(int,float)) else "  -  "
            print(f"{'':7s}{'Δlora-base':14s}|{d(vsov,base[0]):>9}{d(vfac,base[1]):>10}|{d(fsov,base[2]):>11}{d(ffac,base[3]):>11}|{d(tsov,base[4]):>13}")
    print()

print("="*92); print("생성 위생 (gen_v3b): 빈답률 / 메타누수율 / strip율")
print(f"{'model':7s}{'cond':14s}{'빈답':>7}{'누수(사용자가/think)':>20}{'strip된':>9}")
for m in MODELS:
    for c in CONDS:
        rows=rd(G/f"gen_{m}_{c}.jsonl")
        if not rows: continue
        n=len(rows)
        empty=sum(1 for r in rows if not (r.get("generated") or "").strip())
        leak=sum(1 for r in rows if re.match(r'\s*(사용자가|사용자는|이 질문|질문자|<think)',(r.get("generated") or "")))
        strp=sum(1 for r in rows if r.get("stripped"))
        print(f"{m:7s}{c:14s}{empty:>5}/{n}{leak:>16}/{n}{strp:>7}/{n}")
print("="*92)
