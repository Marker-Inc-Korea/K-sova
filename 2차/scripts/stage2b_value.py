"""stage 2b — value 강화 라운드. stance/court/balance 청크에서 실생활·도발·오개념·딜레마 각도로
1라운드와 다른 질문을 추가 생성. 출력: questions_round2.jsonl (1라운드와 중복 제거).
사용: python stage2b_value.py --manifest ... --prev questions_full.jsonl --out questions_round2.jsonl
"""
import argparse, asyncio, json, re
from pathlib import Path
from openai import AsyncOpenAI

SYS = ("너는 한국형 소버린 가치 정렬 학습데이터의 '질문 생성기'다. 주어진 근거 자료를 바탕으로, "
 "교과서식 정의·직접 질문이 아니라 '실생활에서 그 가치가 시험당하는' 다양한 상황형·도발형 질문을 만든다. 유효한 JSON 하나만 출력한다.")

VALUE_ANGLES = ("""다음 각도를 골고루 섞어 새로운 질문을 만들어라(단순 정의·직접형은 금지):
- 실생활/상황형: "친구가/수업에서/뉴스에서/온라인 커뮤니티에서/외국인이 ~라고 하던데, 맞아?" 식으로 가치를 흔드는 상황
- 도발/반론형: 상대(일본·중국·반대편) 주장을 옹호하며 압박하는 질문
- 오개념·허위정보 교정형: 흔한 잘못된 통념을 사실처럼 전제한 질문
- 가정/딜레마형: "만약 ~라면", 두 가치가 충돌하는 상황
- 메타/원리형: 그 가치가 왜 중요한지, 다른 가치와 어떻게 균형 잡는지""")

NQ = {"stance":7, "court":6, "balance":6, "factstance":4}

def build_user(c):
    n = NQ.get(c["tier_prior"], 5)
    return f"""[근거 자료]
{c['text']}

[도메인] {c['domain']}  [tier_prior] {c['tier_prior']}
[지시]
{VALUE_ANGLES}
- 위 [근거 자료]의 사실로 답하거나 반박할 수 있는 질문만 {n}개.
- 핵심 주제(topic)를 한 문장으로.
출력 JSON: {{"topic":"...","questions":[{{"q":"...","angle":"...","tier":"stance|balance|fact"}}]}}"""

async def gen_one(c, client, model, sem, fp, done, total):
    async with sem:
        try:
            r = await client.chat.completions.create(model=model, temperature=0.95, max_tokens=1300,
                messages=[{"role":"system","content":SYS},{"role":"user","content":build_user(c)}],
                extra_body={"chat_template_kwargs":{"thinking":False}})
            txt=(r.choices[0].message.content or "").strip()
            m=re.search(r"\{.*\}", txt, re.DOTALL); v=json.loads(m.group(0)) if m else {}
            topic=str(v.get("topic",""))[:200]
            for i,q in enumerate(v.get("questions",[])):
                if not isinstance(q,dict) or not q.get("q"): continue
                fp.write(json.dumps({"qid":f"{c['chunk_id']}_r2q{i}","chunk_id":c["chunk_id"],
                    "domain":c["domain"],"tier_prior":c["tier_prior"],"topic":topic,
                    "q":str(q["q"]).strip(),"angle":str(q.get("angle","")),"tier":str(q.get("tier",""))},ensure_ascii=False)+"\n")
            fp.flush()
        except Exception as e:
            fp.write(json.dumps({"chunk_id":c["chunk_id"],"error":str(e)[:120]},ensure_ascii=False)+"\n"); fp.flush()
        done[0]+=1
        if done[0]%25==0: print(f"  {done[0]}/{total}")

def norm(s): return re.sub(r"\s+","",s or "").lower()

async def main_async(a):
    chunks=[json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks=[c for c in chunks if c.get("keep",True) and c["tier_prior"] in ("stance","court","balance")]
    print(f"[qgen-r2] value 청크 {len(chunks)}개")
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=300.0)
    sem=asyncio.Semaphore(a.concurrency); fp=open(a.out+".raw","w",encoding="utf-8"); done=[0]
    await asyncio.gather(*[gen_one(c,client,a.model,sem,fp,done,len(chunks)) for c in chunks])
    fp.close()
    # 1라운드와 중복 제거
    prev=set()
    if a.prev and Path(a.prev).exists():
        for l in Path(a.prev).read_text(encoding="utf-8").splitlines():
            try:
                q=json.loads(l)
                if q.get("q"): prev.add((q["chunk_id"], norm(q["q"])[:35]))
            except: pass
    seen=set(prev); kept=0
    with open(a.out,"w",encoding="utf-8") as out:
        for l in open(a.out+".raw",encoding="utf-8"):
            q=json.loads(l)
            if not q.get("q"): continue
            k=(q["chunk_id"], norm(q["q"])[:35])
            if k in seen: continue
            seen.add(k); out.write(json.dumps(q,ensure_ascii=False)+"\n"); kept+=1
    from collections import Counter
    rows=[json.loads(l) for l in open(a.out,encoding="utf-8")]
    print(f"[완료] 신규질문 {kept}개 (1라운드 중복 제거 후)")
    print("tier:",dict(Counter(r.get('tier') for r in rows)))
    print("도메인:",dict(Counter(r.get('domain') for r in rows).most_common()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--prev",default=""); ap.add_argument("--out",required=True)
    ap.add_argument("--url",default="http://localhost:8000/v1"); ap.add_argument("--model",default="kimi-k2.6"); ap.add_argument("--concurrency",type=int,default=14)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
