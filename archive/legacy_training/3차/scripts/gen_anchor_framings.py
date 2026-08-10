#!/usr/bin/env python3
"""CORE fact 반복 보강 — 기존 검증답변(A)을 gold fact로, 다각도 Q-A 프레이밍 N개 생성.
1차 성공공식(소수 핵심×고반복)을 2차 CORE에 이식. closed-book 합니다체, 인용 금지.
입력: closedbook_keep.jsonl(fact tier 순서) + fact_classified.jsonl(위치매칭).
출력: anchor_framings.jsonl (tier=fact, bucket=CORE, anchor=True).
"""
import argparse, asyncio, json, re
from pathlib import Path
from openai import AsyncOpenAI

SYS=("너는 한국 주권가치 closed-book QA 데이터 설계자다. 주어진 '검증된 사실'을 정답으로 하는 "
     "서로 다른 프레이밍의 질문과, 출처·인용 표기 없이 지식으로 설명하는 합니다체 답변 쌍을 만든다. "
     "유효한 JSON 배열만 출력한다.")

def build(fact, sample_q, n):
    return f"""[검증된 사실 — 이 내용이 모든 답변에 정확히 담겨야 하며, 왜곡·추가 금지]
{fact[:1400]}

[참고: 기존 질문 예시]
{sample_q[:200]}

[지시] 위 '검증된 사실'을 정답으로 하는 한국어 Q-A 쌍을 {n}개 만들어라.
- 질문(q): 서로 다른 프레이밍을 골고루 — 직접지목 / 시기·사건 기반 / 직책·제도 맥락 / 외국인·초보 톤 / 일상 구어체 / 오답 가정 교정 / 비교·연결. 정답을 질문에 노출하지 말 것.
- 답변(a): "~에 따르면", "자료에서", "근거 자료" 같은 출처·인용 표기 절대 금지. 너 자신의 지식으로 설명하는 합니다체. 사실을 정확히 담되 자연스러운 추론 흐름으로 200~450자.
- 사실에 없는 정밀 수치·날짜를 지어내지 말 것. 핵심 사실만 정확히.
JSON 배열만 출력: [{{"q":"...","a":"..."}}, ...]"""

async def one(fact_rec, n, client, model, sem, fout, lock):
    async with sem:
        n=fact_rec.get("n_override") or n
        p=build(fact_rec["fact"], fact_rec["sample_q"], n)
        for attempt in range(5):
            try:
                r=await client.chat.completions.create(model=model,
                    temperature=(0.6 if attempt==0 else 0.8), max_tokens=3200,
                    messages=[{"role":"system","content":SYS},{"role":"user","content":p}],
                    extra_body={"chat_template_kwargs":{"thinking":False}})
                msg=r.choices[0].message; txt=(msg.content or "")
                if "[" not in txt: txt=getattr(msg,"reasoning_content","") or txt
                txt=re.sub(r"```(?:json)?","",txt)
                m=re.search(r"\[.*\]", txt, re.DOTALL)
                if not m: continue
                arr=json.loads(m.group(0))
                out=[]
                for o in arr:
                    q=(o.get("q") or "").strip(); a=(o.get("a") or "").strip()
                    if len(q)<5 or len(a)<60: continue
                    if re.search(r"에 따르면|자료에서|근거 자료|제시된 (자료|글)", a): continue  # 인용 누수 차단
                    out.append({"tier":"fact","domain":fact_rec["domain"],"source_chunk":fact_rec["chunk"],
                                "bucket":"CORE","anchor":True,
                                "messages":[{"role":"user","content":q},{"role":"assistant","content":a}]})
                if out:
                    async with lock:
                        for o in out: fout.write(json.dumps(o,ensure_ascii=False)+"\n")
                        fout.flush()
                    return len(out)
            except Exception:
                pass
            await asyncio.sleep(0.4)
        return 0

async def main_async(a):
    if a.seed_file:
        seed=json.loads(Path(a.seed_file).read_text(encoding="utf-8"))
        # 큰 N은 ≤9/호출로 쪼개 다회 호출(max_tokens 한계 회피·다양성↑)
        facts=[]
        for s in seed:
            tot=s.get("n",a.n); per=9
            for off in range(0,tot,per):
                facts.append({"chunk":s.get("chunk","anchor:"+s["key"]),"domain":s.get("domain","currentfacts"),
                              "fact":s["fact"],"sample_q":s.get("sample_q",""),"n_override":min(per,tot-off)})
        print(f"[framings/seed] {len(seed)}개 gold 앵커 → {len(facts)}개 호출(≤9/호출)")
        client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=240.0)
        sem=asyncio.Semaphore(a.concurrency); lock=asyncio.Lock()
        fout=open(a.out,"w",encoding="utf-8")
        res=await asyncio.gather(*[one(f,a.n,client,a.model,sem,fout,lock) for f in facts])
        fout.close(); print(f"  생성 Q-A: {sum(res)}건 (실패 {sum(1 for x in res if x==0)})"); return
    keep=[json.loads(l) for l in open("closedbook_keep.jsonl")]
    fact=[r for r in keep if r.get("tier")=="fact"]
    cls=[json.loads(l) for l in open("fact_classified.jsonl")]
    assert len(fact)==len(cls)
    prot=[f for f,c in zip(fact,cls) if c["bucket"]=="CORE"]
    if a.skip_currentfacts:
        prot=[r for r in prot if r.get("domain")!="currentfacts"]  # 현재사실은 anchor_qgen이 담당
    # distinct chunk 그룹 → 가장 긴 답변을 gold fact로
    by={}
    for r in prot:
        ck=r.get("source_chunk"); ans=r["messages"][1]["content"]
        if ck not in by or len(ans)>len(by[ck]["fact"]):
            by[ck]={"chunk":ck,"domain":r.get("domain"),"fact":ans,"sample_q":r["messages"][0]["content"]}
    facts=list(by.values())
    if a.limit: facts=facts[:a.limit]
    print(f"[framings] distinct CORE fact {len(facts)}개 × N={a.n} (skip_cf={a.skip_currentfacts})")
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=240.0)
    sem=asyncio.Semaphore(a.concurrency); lock=asyncio.Lock()
    fout=open(a.out,"w",encoding="utf-8")
    res=await asyncio.gather(*[one(f,a.n,client,a.model,sem,fout,lock) for f in facts])
    fout.close()
    print(f"  생성 Q-A: {sum(res)}건 (실패 fact {sum(1 for x in res if x==0)}개)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="anchor_framings.jsonl")
    ap.add_argument("--n",type=int,default=8)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--skip-currentfacts",action="store_true")
    ap.add_argument("--seed-file",default="")
    ap.add_argument("--url",default="http://localhost:8000/v1")
    ap.add_argument("--model",default="kimi-k2.6")
    ap.add_argument("--concurrency",type=int,default=32)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
