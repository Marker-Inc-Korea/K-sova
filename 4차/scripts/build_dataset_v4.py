"""K-SOVA 4차 학습셋 조립 — ★지식주입 평가 설계: 질문단위 split(같은청크 다른표현 허용) + CORE 고반복 + 형식셋 + 강건성.
- 평가 설계(사용자 확정): 목적이 지식주입이므로 eval/test는 '주입한 사실(같은 청크)을 새 표현 질문으로 잘 답하는지' 측정한다.
  → value+SECONDARY를 질문단위 8:1:1 층화. 같은 청크의 다른 질문이 train↔eval로 갈리는 건 누수가 아니라 주입 지표.
  → 단 '준-완전일치'(같은 청크 내 거의 동일 질문)는 진짜 누수라 eval/test에서 제거(char-trigram Jaccard≥0.85).
- CORE원본 + framings + 현재앵커 + general + format + robustness → 전부 train 전용. (CORE 주입은 별도 factuality_eval 새표현으로 측정)
출력: train_v4_clean / eval_v4 / test_v4.
"""
import json, random, re
from collections import defaultdict, Counter
from pathlib import Path
random.seed(13)
GEN = Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")

keep = [json.loads(l) for l in (GEN/"closedbook_keep.jsonl").read_text().splitlines() if l.strip()]
fact = [r for r in keep if r.get("tier")=="fact"]
value = [r for r in keep if r.get("tier")!="fact"]
cls = [json.loads(l) for l in (GEN/"fact_classified.jsonl").read_text().splitlines() if l.strip()]
assert len(fact)==len(cls), f"fact {len(fact)} != cls {len(cls)}"

CITE = re.compile(r"자료에 (따르면|의하면|나타난|제시)|제시된 (자료|글|내용|본문|지문)|위 (자료|글|내용|지문)|본문에 (따르면|나온|제시)|근거 자료|주어진 (자료|글|내용)")
def clean(r): return not CITE.search(r["messages"][1]["content"])
def std(r, bucket):
    return {"domain":r["domain"], "tier":r["tier"], "source":"sovereign", "bucket":bucket,
            "source_chunk":r.get("source_chunk"),
            "messages":[{"role":"user","content":r["messages"][0]["content"]},
                        {"role":"assistant","content":r["messages"][1]["content"]}]}
# ★ 소버린 감사 blocklist (전수 감사 + Claude 적대 재검증으로 확정된 비-소버린 제거; 오탐 33건은 구제됨)
_blkf = GEN/"nonsov_blocklist.json"
_blk = json.loads(_blkf.read_text()) if _blkf.exists() else {"questions":[],"core_chunks":[]}
BLK_Q = set(_blk["questions"]); BLK_CHUNK = set(_blk["core_chunks"])
def _nq(s): return re.sub(r"\s+","",s or "").lower()
def _blocked(r): return _nq(r["messages"][0]["content"]) in BLK_Q
SECONDARY = [std(f,"SECONDARY") for f,c in zip(fact,cls) if c["bucket"]=="SECONDARY" and clean(f) and not _blocked(f)]
CORE = [std(f,"CORE") for f,c in zip(fact,cls) if c["bucket"]=="CORE" and clean(f) and not _blocked(f)]
VAL = [std(v,"value") for v in value if clean(v) and not _blocked(v)]
nEX = sum(1 for c in cls if c["bucket"]=="EXCLUDE")
nblk = sum(1 for f in fact if _blocked(f)) + sum(1 for v in value if _blocked(v))
print(f"CORE {len(CORE)} / SECONDARY {len(SECONDARY)} / value {len(VAL)} / EXCLUDE제거 {nEX} / blocklist제거 {nblk}(소스기준)")

# ★ 질문단위 split (도메인×tier 층화). value+SECONDARY만 held-out 대상, CORE는 train 전용.
pool = SECONDARY + VAL
strata = defaultdict(list)
for r in pool: strata[(r["domain"], r["tier"])].append(r)
train, ev, test = [], [], []
for k, items in strata.items():
    random.shuffle(items); n=len(items); ntr=int(n*0.8); nev=int(n*0.1)
    train += items[:ntr]; ev += items[ntr:ntr+nev]; test += items[ntr+nev:]
print(f"질문단위 split: pool(value+SECONDARY) {len(pool)} → train {len(train)} / eval {len(ev)} / test {len(test)}")

# framings(충실성 드롭) + 현재앵커(엔티티 규칙) — v3.1 로직 그대로, train 전용
faith_drop = set(json.loads(l)["idx"] for l in open(GEN/"faith_framings.jsonl") if json.loads(l)["verdict"] in ("added","contradict"))
ENT = {'anchor:president':(['이재명'],[r'현(재|직)\s*대통령[은이]?\s*윤석열']),
 'anchor:pm':(['김민석'],[]),'anchor:prev_pres':(['윤석열'],[]),
 'anchor:assembly':(['더불어민주당'],[r'국민의힘[이은]?\s*(여당|다수|과반)']),
 'anchor:speaker':(['우원식'],[]),'anchor:judiciary':(['조희대','김상환'],[]),
 'anchor:parties':(['더불어민주당','국민의힘'],[r'국민의힘[이은]?\s*여당|여당[은이]?\s*국민의힘'])}
def load_fr(path, mode):
    out=[]; seen=set()
    for idx,l in enumerate(open(GEN/path)):
        if not l.strip(): continue
        r=json.loads(l); q=r["messages"][0]["content"]; a=r["messages"][1]["content"]
        if CITE.search(a) or len(a)<60 or len(q)<5: continue
        if _nq(q) in BLK_Q: continue                                      # 감사 확정 비-소버린 제거
        if mode=="framings" and r.get("source_chunk") in BLK_CHUNK: continue  # 비-소버린 CORE 청크의 framings 제거
        if q[:40] in seen: continue
        if mode=="framings" and idx in faith_drop: continue
        if mode=="cf":
            must,forbid=ENT.get(r.get("source_chunk"),([],[]))
            if not all(m in a for m in must) or any(re.search(p,a) for p in forbid): continue
        seen.add(q[:40])
        out.append({"domain":r.get("domain","currentfacts" if mode=="cf" else None),"tier":"fact",
                    "source":"anchor","bucket":"CORE","source_chunk":r.get("source_chunk"),
                    "messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})
    return out
FR = load_fr("anchor_framings.jsonl","framings")
CF = load_fr("anchor_currentfacts.jsonl","cf")
print(f"framings {len(FR)} (faith드롭 {len(faith_drop)}) / 현재앵커 {len(CF)}")

# train 전용 소버린 합류 (CORE 고반복 + framings + 현재앵커)
train += CORE + FR + CF

# ★ 준-완전일치만 제거(진짜 누수): eval/test 질문이 '같은 청크'의 train 질문과 char-trigram Jaccard≥0.85면 drop.
#   같은사실·다른표현(낮은 중첩)은 보존 = 주입 측정 대상.
def _tri(s):
    s=re.sub(r"\s+","",s or ""); return set(s[i:i+3] for i in range(len(s)-2)) or ({s} if s else set())
def _jac(a,b): return len(a&b)/len(a|b) if (a|b) else 0.0
_tbc=defaultdict(list)
for r in train:
    c=r.get("source_chunk")
    if c: _tbc[c].append(_tri(r["messages"][0]["content"]))
def _dedup(rows, name):
    out=[]; d=0
    for r in rows:
        c=r.get("source_chunk"); qt=_tri(r["messages"][0]["content"])
        if c and any(_jac(qt,tt)>=0.85 for tt in _tbc.get(c,[])): d+=1; continue
        out.append(r)
    print(f"  near-dup({name}) 제거 {d} → {len(out)}")
    return out
ev=_dedup(ev,"eval"); test=_dedup(test,"test")

# 범용 (markrai 3000 + koalpaca 1000)
V6="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/data/v6/v6_train_mixed.jsonl"
KO="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/data/splits/koalpaca_1000.jsonl"
gen=[]
for l in open(V6):
    r=json.loads(l)
    if r.get("_source")=="markrai-hermes": gen.append({"domain":"general","tier":"general","source":"markrai","bucket":None,"messages":r["messages"]})
for l in open(KO):
    r=json.loads(l)
    if r.get("instruction") and r.get("output"): gen.append({"domain":"general","tier":"general","source":"koalpaca","bucket":None,"messages":[{"role":"user","content":r["instruction"]},{"role":"assistant","content":r["output"]}]})
train += gen

# 형식셋
fmt=[json.loads(l) for l in open(GEN/"format_set.jsonl") if l.strip()]
for r in fmt: r.setdefault("source","format_v4"); r.setdefault("bucket",None); r.setdefault("tier","format")
train += fmt

# 강건성 (train 전용; probe는 별도 held-out)
nrob=0
if (GEN/"robustness_train.jsonl").exists():
    rob=[json.loads(l) for l in open(GEN/"robustness_train.jsonl") if l.strip()]
    train += rob; nrob=len(rob)

random.shuffle(train); random.shuffle(ev); random.shuffle(test)
for i,r in enumerate(train): r["id"]=f"sv4_tr_{i:06d}"
for i,r in enumerate(ev): r["id"]=f"sv4_ev_{i:05d}"
for i,r in enumerate(test): r["id"]=f"sv4_te_{i:05d}"
for name,d in [("train_v4_clean",train),("eval_v4",ev),("test_v4",test)]:
    with open(GEN/f"{name}.jsonl","w") as f:
        for r in d: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"\ntrain_v4_clean {len(train)} / eval_v4 {len(ev)} / test_v4 {len(test)}")
print(f"  (robustness {nrob} / format {len(fmt)} / general {len(gen)} train 포함)")
print("train tier:", dict(Counter(r['tier'] for r in train)))
print("train source:", dict(Counter(r.get('source') for r in train)))
print("eval tier:", dict(Counter(r['tier'] for r in ev)), "| test tier:", dict(Counter(r['tier'] for r in test)))
