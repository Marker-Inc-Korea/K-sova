"""지식 프로브 결정적 채점 — 생성답에 정답 키워드 포함여부. base vs lora 비교표.
사용: python knowledge_grade.py --probe knowledge_probe.jsonl --gen runs/gen_v3/kp_exaone_base.jsonl runs/gen_v3/kp_exaone_lora.jsonl
"""
import argparse, json, re
from collections import defaultdict

def norm(s): return re.sub(r"\s+","",(s or "")).lower()

def load_probe(p):
    return {r["qid"]:r for r in (json.loads(l) for l in open(p,encoding="utf-8") if l.strip())}

def grade(probe, genfile):
    gens={r.get("qid"):r.get("generated","") for r in (json.loads(l) for l in open(genfile,encoding="utf-8") if l.strip())}
    res={"all":[0,0],"current":[0,0],"durable":[0,0]}; bycat=defaultdict(lambda:[0,0]); miss=[]
    for qid,pr in probe.items():
        if pr["grade"]!="keyword": continue
        g=gens.get(qid,""); ng=norm(g)
        ok=any(norm(k) in ng for k in pr["answer_keywords"])
        for bucket in ("all",pr["kind"]): res[bucket][0]+=ok; res[bucket][1]+=1
        bycat[pr["category"]][0]+=ok; bycat[pr["category"]][1]+=1
        if not ok and pr["kind"]=="current": miss.append((qid,pr["instruction"][:40],pr["output"],g[:50]))
    return res,bycat,miss

def pct(x): return f"{100*x[0]/x[1]:.0f}%({x[0]}/{x[1]})" if x[1] else "-"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--probe",default="knowledge_probe.jsonl"); ap.add_argument("--gen",nargs="+",required=True)
    a=ap.parse_args(); probe=load_probe(a.probe)
    print(f"{'파일':42s}{'전체':>14}{'현재사실':>14}{'상식':>14}")
    allmiss={}
    for gf in a.gen:
        res,bycat,miss=grade(probe,gf); allmiss[gf]=miss
        name=gf.split('/')[-1]
        print(f"{name:42s}{pct(res['all']):>14}{pct(res['current']):>14}{pct(res['durable']):>14}")
    # 마지막 파일 기준 현재사실 오답 상세
    last=a.gen[-1]
    if allmiss[last]:
        print(f"\n[{last.split('/')[-1]}] 현재사실 오답:")
        for qid,q,canon,g in allmiss[last][:20]: print(f"  {qid} {q}… 정답={canon} | 답={g}")

if __name__=="__main__": main()
