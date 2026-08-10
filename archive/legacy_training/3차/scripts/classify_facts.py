#!/usr/bin/env python3
"""fact tier 3축 분류기 — CORE / SECONDARY / EXCLUDE 판정.
축: sovereign(주권핵심) / recall(principle vs precise) / critical(틀리면 안 됨).
버킷은 축에서 코드로 결정(모델 bucket은 참고).
"""
import json, re, asyncio, argparse
from pathlib import Path
from openai import AsyncOpenAI

PROMPT = """다음은 한국 주권가치 주입 모델 학습용 closed-book QA 한 건이다. 세 축으로 분류하라.

[질문] {q}
[답변] {a}

판단 기준:
1. sovereign: 한국이 반드시 알아야 할 "주권/국가 핵심"인가?
   - 예: 영토(독도·동해·간도·영해), 국가정체성, 헌법 핵심조항, 주요 헌재/대법 판례(정당해산·탄핵 등), 핵심 국가정책·외교입장, 역사주권(위안부·강제동원).
   - 아니오: 일반 민법/형법 원리, 법학교과서식 사례, 생활지식, 특정 사건의 절차세부.
2. recall: 이 질문에 정확히 답하려면?
   - "principle" = 원리·취지·방향·의미 (모델이 일반지식으로 회상 가능).
   - "precise" = 정확한 조항번호·수치·날짜·특정 사건번호·익명피고인 절차·고유명사 암기 (closed-book 회상 거의 불가).
3. critical: 모델이 틀리게 답하면 안 되는 주권 핵심사실인가? (영토주권·국가정체성·주요 헌법/판례/정책 = 예)

JSON 한 줄만 출력(설명 금지):
{{"sovereign":true/false,"recall":"principle"|"precise","critical":true/false,"topic":"5단어이내"}}"""

def decide(v):
    sov=bool(v.get("sovereign")); crit=bool(v.get("critical")); rec=v.get("recall")
    if sov and crit: return "CORE"          # 정밀해도 반드시 정확 암기 → anchor
    if rec=="principle": return "SECONDARY"          # 회상 가능 → 유지
    return "EXCLUDE"                               # precise + 비핵심 → 제거/추상화

async def cls_one(rec, client, model, sem):
    async with sem:
        q=rec["messages"][0]["content"]; a=rec["messages"][1]["content"]
        p=PROMPT.format(q=q[:500], a=a[:900])
        last=""
        for attempt in range(6):
            try:
                r=await client.chat.completions.create(model=model,
                    temperature=(0.0 if attempt==0 else 0.4), max_tokens=300,
                    messages=[{"role":"user","content":p}],
                    extra_body={"chat_template_kwargs":{"thinking":False}})
                msg=r.choices[0].message; txt=(msg.content or "")
                if "{" not in txt: txt=getattr(msg,"reasoning_content","") or txt
                txt=re.sub(r"```(?:json)?","",txt); last=txt[:120]
                m=re.search(r"\{.*\}", txt, re.DOTALL)
                if m:
                    v=json.loads(m.group(0))
                    if "sovereign" in v and "recall" in v:
                        return {"sovereign":bool(v.get("sovereign")),"recall":v.get("recall"),
                                "critical":bool(v.get("critical")),"topic":str(v.get("topic",""))[:40],
                                "bucket":decide(v)}
            except Exception as e:
                last=str(e)[:80]
            await asyncio.sleep(0.5)
        return {"error":f"retry_fail:{last}","bucket":"ERR"}

async def main_async(a):
    rows=[json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    fact=[r for r in rows if r.get("tier")=="fact"]
    if a.domains: fact=[r for r in fact if r.get("domain") in a.domains]
    print(f"[classify] {len(fact)}건 (domains={a.domains or 'ALL-fact'}) conc={a.concurrency}")
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=180.0)
    sem=asyncio.Semaphore(a.concurrency)
    res=await asyncio.gather(*[cls_one(r,client,a.model,sem) for r in fact])
    with open(a.out,"w",encoding="utf-8") as f:
        for r,c in zip(fact,res):
            f.write(json.dumps({"id":r.get("id"),"domain":r.get("domain"),
                "source_chunk":(r.get("source_chunk") or "")[:60],
                "q":r["messages"][0]["content"][:80],**c},ensure_ascii=False)+"\n")
    from collections import Counter
    bc=Counter(c.get("bucket") for c in res)
    print("버킷:", dict(bc))
    for d in sorted(set(r.get("domain") for r in fact)):
        sub=[c for r,c in zip(fact,res) if r.get("domain")==d]
        dc=Counter(c.get("bucket") for c in sub)
        print(f"  {d}({len(sub)}): CORE {dc.get('CORE',0)} / SECONDARY {dc.get('SECONDARY',0)} / EXCLUDE {dc.get('EXCLUDE',0)} / ERR {dc.get('ERR',0)}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inp",default="closedbook_keep.jsonl")
    ap.add_argument("--out",default="fact_classified.jsonl")
    ap.add_argument("--domains",nargs="*",default=[])
    ap.add_argument("--url",default="http://localhost:8000/v1")
    ap.add_argument("--model",default="kimi-k2.6")
    ap.add_argument("--concurrency",type=int,default=32)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
