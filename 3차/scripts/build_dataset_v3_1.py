"""v3.1 학습셋 조립 (인자버그와 무관한 소버린 트랙) — fact 3축 분류 적용 + CORE 고반복.
- EXCLUDE fact 제거(환각원), SECONDARY/value 유지, CORE 원본+프레이밍+현재앵커로 고반복.
- 누수방지: eval/test는 value+SECONDARY에서만 층화 추출. CORE·프레이밍·현재앵커·범용은 train 전용.
출력: train_v3_1 / eval_v3_1 / test_v3_1.
"""
import json, random, re
from collections import defaultdict, Counter
from pathlib import Path
random.seed(13)
GEN=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")

keep=[json.loads(l) for l in (GEN/"closedbook_keep.jsonl").read_text().splitlines() if l.strip()]
fact=[r for r in keep if r.get("tier")=="fact"]
value=[r for r in keep if r.get("tier")!="fact"]
cls=[json.loads(l) for l in (GEN/"fact_classified.jsonl").read_text().splitlines() if l.strip()]
assert len(fact)==len(cls)
# 출처자료 인용누수 필터(정당한 "헌법에 따르면"은 허용, 출처자료 메타참조만 제거)
CITE=re.compile(r"자료에 (따르면|의하면|나타난|제시)|제시된 (자료|글|내용|본문|지문)|위 (자료|글|내용|지문)|본문에 (따르면|나온|제시)|근거 자료|주어진 (자료|글|내용)")
def clean(r): return not CITE.search(r["messages"][1]["content"])
def std(r,bucket=None):
    return {"domain":r["domain"],"tier":r["tier"],"source":"sovereign","bucket":bucket,
            "messages":[{"role":"user","content":r["messages"][0]["content"]},
                        {"role":"assistant","content":r["messages"][1]["content"]}]}
SECONDARY=[std(f,"SECONDARY") for f,c in zip(fact,cls) if c["bucket"]=="SECONDARY" and clean(f)]
PROT=[std(f,"CORE") for f,c in zip(fact,cls) if c["bucket"]=="CORE" and clean(f)]
VAL=[std(v,"value") for v in value if clean(v)]
nEXCLUDE=sum(1 for c in cls if c["bucket"]=="EXCLUDE")
ncite=sum(1 for r in (value+[f for f,c in zip(fact,cls) if c['bucket'] in('SECONDARY','CORE')]) if not clean(r))
print(f"value {len(VAL)} / fact SECONDARY {len(SECONDARY)} / CORE원본 {len(PROT)} / EXCLUDE제거 {nEXCLUDE} / 인용누수제거 {ncite}")

# non-cf 프레이밍: faith judge에서 added/contradict인 idx 드롭
faith_drop=set(json.loads(l)["idx"] for l in open(GEN/"faith_framings.jsonl") if json.loads(l)["verdict"] in ("added","contradict"))
# 현재앵커: 규칙 엔티티 일관성(judge는 컷오프전이라 신뢰불가)
ENT={'anchor:president':(['이재명'],[r'현(재|직)\s*대통령[은이]?\s*윤석열']),
 'anchor:pm':(['김민석'],[]),'anchor:prev_pres':(['윤석열'],[]),
 'anchor:assembly':(['더불어민주당'],[r'국민의힘[이은]?\s*(여당|다수|과반)']),
 'anchor:speaker':(['우원식'],[]),'anchor:judiciary':(['조희대','김상환'],[]),
 'anchor:parties':(['더불어민주당','국민의힘'],[r'국민의힘[이은]?\s*여당|여당[은이]?\s*국민의힘'])}
def load_fr(path,mode):
    out=[]; seen=set()
    for idx,l in enumerate(open(GEN/path)):
        if not l.strip(): continue
        r=json.loads(l); q=r["messages"][0]["content"]; a=r["messages"][1]["content"]
        if CITE.search(a) or len(a)<60 or len(q)<5: continue
        if q[:40] in seen: continue
        if mode=="framings" and idx in faith_drop: continue           # 충실성 드롭
        if mode=="cf":                                                # 엔티티 규칙
            must,forbid=ENT.get(r.get("source_chunk"),([],[]))
            if not all(m in a for m in must) or any(re.search(p,a) for p in forbid): continue
        seen.add(q[:40])
        out.append({"domain":r.get("domain","currentfacts" if mode=="cf" else None),"tier":"fact",
                    "source":"anchor","bucket":"CORE","messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})
    return out
FR=load_fr("anchor_framings.jsonl","framings")
CF=load_fr("anchor_currentfacts.jsonl","cf")
print(f"프레이밍 QA후: non-cf {len(FR)}(faith드롭 {len(faith_drop)}) / 현재앵커 {len(CF)}")

# eval/test: value+SECONDARY 에서만 8:1:1 층화 (held-out, 누수 0)
pool=VAL+SECONDARY
strata=defaultdict(list)
for r in pool: strata[(r["domain"],r["tier"])].append(r)
train,ev,test=[],[],[]
for k,items in strata.items():
    random.shuffle(items); n=len(items); ntr=int(n*0.8); nev=int(n*0.1)
    train+=items[:ntr]; ev+=items[ntr:ntr+nev]; test+=items[ntr+nev:]
# CORE 원본 + 프레이밍 + 현재앵커 → train 전용(고반복 암기)
train+=PROT+FR+CF

# 범용 믹스 (markrai 3000 + koalpaca 1000) → train
V6="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/data/v6/v6_train_mixed.jsonl"
KO="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/data/splits/koalpaca_1000.jsonl"
gen=[]
for l in open(V6):
    r=json.loads(l)
    if r.get("_source")=="markrai-hermes": gen.append({"domain":"general","tier":"general","source":"markrai","bucket":None,"messages":r["messages"]})
for l in open(KO):
    r=json.loads(l)
    if r.get("instruction") and r.get("output"): gen.append({"domain":"general","tier":"general","source":"koalpaca","bucket":None,"messages":[{"role":"user","content":r["instruction"]},{"role":"assistant","content":r["output"]}]})
train+=gen; random.shuffle(train); random.shuffle(ev); random.shuffle(test)
for i,r in enumerate(train): r["id"]=f"sv31_tr_{i:06d}"
for i,r in enumerate(ev): r["id"]=f"sv31_ev_{i:05d}"
for i,r in enumerate(test): r["id"]=f"sv31_te_{i:05d}"
for name,d in [("train_v3_1",train),("eval_v3_1",ev),("test_v3_1",test)]:
    with open(GEN/f"{name}.jsonl","w") as f:
        for r in d: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"\ntrain {len(train)} / eval {len(ev)} / test {len(test)}")
print("train tier:",dict(Counter(r['tier'] for r in train)))
print("train bucket:",dict(Counter(r.get('bucket') for r in train)))
print("train fact domain:",dict(Counter(r['domain'] for r in train if r['tier']=='fact').most_common()))
print("eval tier:",dict(Counter(r['tier'] for r in ev)),"| test tier:",dict(Counter(r['tier'] for r in test)))
