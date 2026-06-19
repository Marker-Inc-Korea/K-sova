#!/usr/bin/env python3
"""K-SOVA 5차(재구성) master 재조립 → 8:1:1 층화 split → 프로브 dedup → 검증.
- 정렬(주입) fact: fact_md_reformatted 적용(긴 평문→마크다운, 내용불변). stance/balance/robustness 원본.
- 범용: general_8k keep=True 합류. factuality는 별도 프로브로 분리(풀 제외).
- 8:1:1 층화(stratum=source 또는 general category), 질문단위, 내부 exact dedup.
- train·test에서 5종 프로브와 완전/준완전(0.7) 일치 제거 → 프로브 disjoint.
출력: train_v5r / eval_v5r / test_v5r.jsonl + 검증 리포트."""
import json, re, random, collections
random.seed(42)
GEN="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen"
def norm(s): return re.sub(r'[^\w가-힣]','',(s or '').lower())
def toks(s): return set(re.findall(r'[가-힣]{2,}|[a-z0-9]+',(s or '').lower()))
def jac(a,b): return len(a&b)/len(a|b) if a|b else 0

# 1) master + fact 재서식 적용
M=[json.loads(l) for l in open(f"{GEN}/master_v5.jsonl")]
ref={}
for l in open(f"{GEN}/fact_md_reformatted.jsonl"):
    r=json.loads(l)
    if r.get("reformatted"): ref[r["id"]]=r["new_answer"]
nref=0
for r in M:
    if r["id"] in ref: r["messages"][1]["content"]=ref[r["id"]]; nref+=1
print(f"fact 재서식 적용 {nref}건")

# 2) factuality 분리(별도 프로브), 주입 풀 구성
inj=[r for r in M if r["source"] in ("sovereign","anchor","robustness")]
fact_eval=[r for r in M if r["source"]=="factuality"]
with open(f"{GEN}/factuality_eval_v5r.jsonl","w") as f:
    for r in fact_eval: f.write(json.dumps(r,ensure_ascii=False)+"\n")
# 범용 keep 합류
gen=[]
for l in open(f"{GEN}/general_8k.jsonl"):
    r=json.loads(l)
    if r.get("keep"):
        gen.append({"id":"g_"+r["qid"],"split":None,"source":"general","tier":"general",
                    "domain":r.get("category"),"axis":r.get("axis"),
                    "messages":[{"role":"user","content":r["instruction"]},{"role":"assistant","content":r["answer"]}]})
print(f"주입 {len(inj)} + factuality(분리) {len(fact_eval)} + 범용keep {len(gen)}")

# 3) 내부 exact dedup (질문)
pool=inj+gen; seen=set(); ded=[]
for r in pool:
    k=norm(r["messages"][0]["content"])
    if k in seen: continue
    seen.add(k); ded.append(r)
print(f"풀 {len(pool)} → 내부 dedup {len(ded)}")

# 4) 8:1:1 층화 (stratum: 주입=source, 범용=domain[category])
def stratum(r): return f"{r['source']}/{r.get('tier')}" if r['source']!="general" else f"general/{r.get('domain')}"
by=collections.defaultdict(list)
for r in ded: by[stratum(r)].append(r)
train,ev,test=[],[],[]
for s,rows in by.items():
    random.shuffle(rows); n=len(rows); a=int(n*0.8); b=int(n*0.9)
    for r in rows[:a]: r["split"]="train"
    for r in rows[a:b]: r["split"]="eval"
    for r in rows[b:]: r["split"]="test"
    train+=rows[:a]; ev+=rows[a:b]; test+=rows[b:]

# 5) 프로브 dedup (train·test에서 프로브 질문과 exact/near 제거)
PROBES=["knowledge_probe_clean.jsonl","currentness_probe_merged.jsonl","robustness_probe_clean.jsonl",
        "sovereign_retention_probe.jsonl","factuality_eval_v5r.jsonl"]
pq=[]
for p in PROBES:
    try:
        for l in open(f"{GEN}/{p}"):
            r=json.loads(l); q=r["messages"][0]["content"] if r.get("messages") else (r.get("instruction") or r.get("question") or "")
            if q: pq.append((norm(q),toks(q)))
    except FileNotFoundError: pass
pqn={n for n,_ in pq}
def leaks(r):
    q=r["messages"][0]["content"]; nq=norm(q); tk=toks(q)
    if nq in pqn: return True
    for pn,ptk in pq:
        if jac(tk,ptk)>=0.7: return True
    return False
def filt(lst):
    out=[r for r in lst if not leaks(r)]; return out, len(lst)-len(out)
train,dt=filt(train); test,dts=filt(test)  # eval은 보통 학습 안 쓰니 train/test 위주, eval도 정리
ev,de=filt(ev)
print(f"프로브 dedup 제거 — train {dt} / eval {de} / test {dts}")

# 6) 저장 + 검증
def dump(rows,fn):
    with open(f"{GEN}/{fn}","w") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
dump(train,"train_v5r.jsonl"); dump(ev,"eval_v5r.jsonl"); dump(test,"test_v5r.jsonl")
def comp(rows):
    c=collections.Counter(r["source"] for r in rows); return dict(c)
print(f"\n=== 최종 split ===")
print(f"train {len(train)} {comp(train)}")
print(f"eval  {len(ev)} {comp(ev)}")
print(f"test  {len(test)} {comp(test)}")
inj_tr=sum(1 for r in train if r["source"]!="general"); gen_tr=sum(1 for r in train if r["source"]=="general")
print(f"train 주입:범용 = {inj_tr}:{gen_tr} = {inj_tr/(inj_tr+gen_tr)*100:.0f}:{gen_tr/(inj_tr+gen_tr)*100:.0f}")
def md(t): return sum([bool(re.search(r'(^|\n)#{1,4}\s',t)),bool(re.search(r'(^|\n)\s*[-*]\s',t)),'```' in t,'**' in t])
factrows=[r for r in train if (r.get("tier") or "").startswith("fact")]
print(f"train fact 마크다운보유: {sum(md(r['messages'][1]['content'])>=1 for r in factrows)}/{len(factrows)}")
print("REBUILD_DONE")
