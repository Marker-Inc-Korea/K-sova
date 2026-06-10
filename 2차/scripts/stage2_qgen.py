"""stage 2 — 질문 생성. 청크별로 tier_prior에 맞춰 다양한 각도의 한국어 질문 생성.
출력: questions.jsonl — {qid, chunk_id, domain, tier_prior, topic, q, angle, tier}
사용: python stage2_qgen.py --manifest ../sources/_corpus/chunks_manifest.jsonl --out questions.jsonl [--sample 20] [--concurrency 16]
"""
import argparse, asyncio, json, re, random
from pathlib import Path
from openai import AsyncOpenAI

SYS = "너는 한국형 소버린 가치 정렬 학습데이터를 만드는 '질문 생성기'다. 주어진 근거 자료를 읽고, 그 자료로 답할 수 있는 다양한 각도의 한국어 질문을 만든다. 유효한 JSON 하나만 출력한다."

ANGLES = {
 "stance":  "직접형/근거요구형/반론도발형(상대 주장도 일리 있지 않냐)/실생활형(외국인이 물으면)/오개념교정형/가정압박형/원리형 등 서로 다른 각도로 다양하게",
 "court":   "직접형/근거요구형/반론도발형/소수의견(반대)관점/원리형/실생활형 등 다양한 각도로",
 "balance": "양측 입장 묻기/왜 견해가 갈리나/A입장 근거/B입장 근거/정답이 있는 문제인가 등 다관점으로",
 "factstance": "사실확인형/적용형/근거요구형/입장형 등 섞어서",
 "fact":    "정의형/적용형/다단계추론형 등",
}
NQ = {"stance":6, "court":6, "balance":5, "factstance":4, "fact":3}

def build_user(c):
    tp = c["tier_prior"]; n = NQ.get(tp,4)
    return f"""[근거 자료]
{c['text']}

[도메인] {c['domain']}  [tier_prior] {tp}
[지시]
- {ANGLES.get(tp, ANGLES['fact'])} 질문을 {n}개 생성.
- 모든 질문은 위 [근거 자료]의 내용으로 답할 수 있어야 한다(반론·실생활형도 자료 속 사실로 받아칠 수 있는 것만).
- 자료의 핵심 주제(topic)를 한 문장으로 요약.
- 질문은 한국어. 자료에 없는 걸 묻지 말 것.
출력 JSON: {{"topic":"<핵심주제 한문장>","questions":[{{"q":"<질문>","angle":"<각도>","tier":"fact|stance|balance"}}]}}"""

async def gen_one(c, client, model, sem, fp, done, total):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model, temperature=0.8, max_tokens=1200,
                messages=[{"role":"system","content":SYS},{"role":"user","content":build_user(c)}],
                extra_body={"chat_template_kwargs":{"thinking":False}})
            txt = (r.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            v = json.loads(m.group(0)) if m else {}
            topic = str(v.get("topic",""))[:200]
            for i,q in enumerate(v.get("questions",[])):
                if not isinstance(q,dict) or not q.get("q"): continue
                fp.write(json.dumps({"qid":f"{c['chunk_id']}_q{i}","chunk_id":c["chunk_id"],
                    "domain":c["domain"],"tier_prior":c["tier_prior"],"topic":topic,
                    "q":str(q["q"]).strip(),"angle":str(q.get("angle","")),"tier":str(q.get("tier",""))},
                    ensure_ascii=False)+"\n")
            fp.flush()
        except Exception as e:
            fp.write(json.dumps({"chunk_id":c["chunk_id"],"error":str(e)[:120]},ensure_ascii=False)+"\n"); fp.flush()
        done[0]+=1
        if done[0]%25==0: print(f"  {done[0]}/{total}")

async def main_async(a):
    chunks=[json.loads(l) for l in Path(a.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    chunks=[c for c in chunks if c.get("keep",True)]
    if a.sample:
        random.seed(42)
        # 도메인 층화 샘플
        by={}
        for c in chunks: by.setdefault(c["domain"],[]).append(c)
        picked=[]
        per=max(1,a.sample//len(by))
        for d,cs in by.items(): picked+=random.sample(cs,min(per,len(cs)))
        chunks=picked[:a.sample] if len(picked)>a.sample else picked
    print(f"[qgen] {len(chunks)} 청크")
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=300.0)
    sem=asyncio.Semaphore(a.concurrency); fp=open(a.out,"w",encoding="utf-8"); done=[0]
    await asyncio.gather(*[gen_one(c,client,a.model,sem,fp,done,len(chunks)) for c in chunks])
    fp.close()
    qs=[json.loads(l) for l in open(a.out,encoding="utf-8")]
    ok=[q for q in qs if "q" in q]
    from collections import Counter
    print(f"\n[완료] 질문 {len(ok)}개 / 에러 {len(qs)-len(ok)}")
    print("tier:",dict(Counter(q.get('tier') for q in ok)))
    print("도메인:",dict(Counter(q.get('domain') for q in ok)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--url",default="http://localhost:8000/v1"); ap.add_argument("--model",default="kimi-k2.6")
    ap.add_argument("--sample",type=int,default=0); ap.add_argument("--concurrency",type=int,default=16)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
