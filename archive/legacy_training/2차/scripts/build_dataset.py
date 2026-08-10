"""학습셋 조립 — judged keep들을 합치고 law 캡·균형 적용, messages 포맷으로.
학습포맷(추론모델 <think> vs CoT노출)은 미정 → 일단 answer 본문 그대로 assistant에 넣음(나중 재포맷 가능).
사용: python build_dataset.py --keep judged_keep.json --judged_cf2 judged_cf2.jsonl --out dataset_sovereign.jsonl --law-cap 2000
"""
import argparse, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

def load_keep_from_judged(path):
    out=[]
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip(): continue
        r=json.loads(l); j=r.get("judge",{})
        if "error" in j or j.get("drop"): continue
        r["true_tier"]=j.get("true_tier") or r.get("tier") or "fact"
        out.append(r)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--keep",required=True)          # judged_keep.json (이미 keep만, true_tier 포함)
    ap.add_argument("--judged",nargs="*",default=[]) # 추가 judged_*.jsonl (judge 원본, keep만 추림)
    ap.add_argument("--out",required=True)
    ap.add_argument("--law-cap",type=int,default=2000)
    a=ap.parse_args()
    random.seed(13)

    recs=json.load(open(a.keep,encoding="utf-8"))
    for r in recs: r.setdefault("true_tier", r.get("tier","fact"))
    for jf in a.judged:
        if Path(jf).exists():
            recs += load_keep_from_judged(jf)
    # true_tier 빈값 → fact
    for r in recs:
        if not r.get("true_tier"): r["true_tier"]="fact"

    # law 캡: 가치(stance/balance) law는 다 보존, fact law만 다운샘플
    law=[r for r in recs if r["domain"]=="law"]; other=[r for r in recs if r["domain"]!="law"]
    if len(law)>a.law_cap:
        law_val=[r for r in law if r["true_tier"]!="fact"]
        law_fact=[r for r in law if r["true_tier"]=="fact"]
        room=max(0,a.law_cap-len(law_val))
        law=law_val+random.sample(law_fact,min(room,len(law_fact)))
        random.shuffle(law)
    recs=other+law
    random.shuffle(recs)

    def clean(t):
        t=re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', t)
        t=re.sub(r'\[\[([^\]]+)\]\]', r'\1', t)
        t=re.sub(r'\[\d{1,3}\]', '', t)
        t=re.sub(r'\{\{[^}]*\}\}', '', t)
        t=re.sub(r'[ \t]+', ' ', t)
        return t.strip()
    # messages 포맷
    with open(a.out,"w",encoding="utf-8") as f:
        for i,r in enumerate(recs):
            rec={"id":f"sv_{i:06d}","domain":r["domain"],"tier":r["true_tier"],
                 "topic":r.get("topic",""),"source_chunk":r.get("chunk_id",""),
                 "messages":[{"role":"user","content":clean(r["q"])},
                             {"role":"assistant","content":clean(r["answer"])}]}
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")

    print(f"[조립] {len(recs)}건 → {a.out}")
    print("tier:",dict(Counter(r["true_tier"] for r in recs)))
    print("도메인:",dict(Counter(r["domain"] for r in recs).most_common()))
    print("도메인×tier value(stance+balance):",
          dict(Counter(r["domain"] for r in recs if r["true_tier"]!="fact").most_common()))

if __name__=="__main__": main()
