"""v3 학습셋 조립 — closed-book keep + 범용믹스 + law캡 + 8:1:1 층화.
출력: train_v3.jsonl / eval_v3.jsonl / test_v3.jsonl. (assistant 답에 사실·가치, completion_only_loss=True로 학습)
"""
import json, random, re
from collections import defaultdict, Counter
from pathlib import Path
random.seed(13)
GEN=Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")

keep=[json.loads(l) for l in (GEN/"closedbook_keep.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
def rec(r):  # 표준 학습 레코드
    return {"domain":r["domain"],"tier":r["tier"],"source":"sovereign",
            "messages":[{"role":"user","content":r["messages"][0]["content"]},
                        {"role":"assistant","content":r["messages"][1]["content"]}]}
sov=[rec(r) for r in keep]

# law 캡 2000: 가치(stance/balance) law 전량 보존, fact law 다운샘플
law=[r for r in sov if r["domain"]=="law"]; other=[r for r in sov if r["domain"]!="law"]
LAWCAP=2000
law_val=[r for r in law if r["tier"]!="fact"]; law_fact=[r for r in law if r["tier"]=="fact"]
room=max(0,LAWCAP-len(law_val)); random.shuffle(law_fact)
law=law_val+law_fact[:room]
sov=other+law; random.shuffle(sov)
print(f"law {len(law_val)+len(law_fact)}→{len(law)} 캡. 소버린 {len(sov)}")

# 8:1:1 층화 (domain×tier)
strata=defaultdict(list)
for r in sov: strata[(r["domain"],r["tier"])].append(r)
train,ev,test=[],[],[]
for k,items in strata.items():
    random.shuffle(items); n=len(items); ntr=int(n*0.8); nev=int(n*0.1)
    train+=items[:ntr]; ev+=items[ntr:ntr+nev]; test+=items[ntr+nev:]

# 범용 믹스 (markrai 3000 + koalpaca 1000) → train만
V6="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/data/v6/v6_train_mixed.jsonl"
KO="/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/data/splits/koalpaca_1000.jsonl"
gen=[]
for l in open(V6,encoding="utf-8"):
    r=json.loads(l)
    if r.get("_source")=="markrai-hermes": gen.append({"domain":"general","tier":"general","source":"markrai","messages":r["messages"]})
for l in open(KO,encoding="utf-8"):
    r=json.loads(l)
    if r.get("instruction") and r.get("output"): gen.append({"domain":"general","tier":"general","source":"koalpaca","messages":[{"role":"user","content":r["instruction"]},{"role":"assistant","content":r["output"]}]})
train+=gen; random.shuffle(train)
for i,r in enumerate(train): r["id"]=f"sv3_tr_{i:06d}"
for i,r in enumerate(ev): r["id"]=f"sv3_ev_{i:05d}"
for i,r in enumerate(test): r["id"]=f"sv3_te_{i:05d}"
for name,d in [("train_v3",train),("eval_v3",ev),("test_v3",test)]:
    with open(GEN/f"{name}.jsonl","w",encoding="utf-8") as f:
        for r in d: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"train {len(train)}(소버린{len(train)-len(gen)}+범용{len(gen)}) / eval {len(ev)} / test {len(test)}")
print("train tier:",dict(Counter(r['tier'] for r in train)))
print("train domain:",dict(Counter(r['domain'] for r in train).most_common()))
print("test domain:",dict(Counter(r['domain'] for r in test).most_common()))
