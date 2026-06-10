#!/usr/bin/env python3
"""프레이밍 충실성 검증 — 생성 답변이 gold 사실 밖의 내용을 날조/왜곡했는지.
gold = 해당 source_chunk의 CORE 원본 중 최장 답변(생성 시 사용한 것과 동일).
판정: faithful(gold범위 내) / added(gold에 없는 구체사실 추가) / contradict(gold와 모순).
"""
import json,re,asyncio,argparse,random
from pathlib import Path
from openai import AsyncOpenAI
random.seed(5)

PROMPT="""아래 [기준 사실]은 검증된 참값이다. [생성 답변]이 기준 사실에 충실한지 판정하라.

[기준 사실]
{gold}

[생성 답변]
{ans}

판정 기준:
- "faithful": 생성 답변의 모든 사실 주장이 기준 사실로 뒷받침됨(표현 차이·요약·생략은 허용).
- "added": 기준 사실에 없는 **구체적 사실**(고유명사·수치·날짜·사건)을 새로 지어냄.
- "contradict": 기준 사실과 **모순**되는 주장.
JSON만: {{"verdict":"faithful|added|contradict","note":"한줄근거"}}"""

async def one(item, client, model, sem):
    async with sem:
        p=PROMPT.format(gold=item["gold"][:1200], ans=item["ans"][:1000])
        for attempt in range(5):
            try:
                r=await client.chat.completions.create(model=model,temperature=(0.0 if attempt==0 else 0.3),
                    max_tokens=250,messages=[{"role":"user","content":p}],
                    extra_body={"chat_template_kwargs":{"thinking":False}})
                m=r.choices[0].message; t=(m.content or "")
                if "{" not in t: t=getattr(m,"reasoning_content","") or t
                mm=re.search(r"\{.*\}",t,re.DOTALL)
                if mm:
                    v=json.loads(mm.group(0))
                    if v.get("verdict") in ("faithful","added","contradict"): return v
            except Exception: pass
            await asyncio.sleep(0.4)
        return {"verdict":"ERR","note":""}

async def main_async(a):
    keep=[json.loads(l) for l in open("closedbook_keep.jsonl")]
    fact=[r for r in keep if r.get("tier")=="fact"]
    cls=[json.loads(l) for l in open("fact_classified.jsonl")]
    # gold map: chunk → 최장 CORE 답변
    gold={}
    for f,c in zip(fact,cls):
        if c["bucket"]!="CORE": continue
        ck=f.get("source_chunk"); ans=f["messages"][1]["content"]
        if ck not in gold or len(ans)>len(gold[ck]): gold[ck]=ans
    fr=[json.loads(l) for l in open(a.inp)]
    # 페어 가능한 것만 판정. currentfacts 앵커(anchor:key)는 gold맵에 없으면 별도처리
    items=[]
    for idx,r in enumerate(fr):
        ck=r.get("source_chunk")
        if ck in gold:
            items.append((idx,{"gold":gold[ck],"ans":r["messages"][1]["content"]}))
    if not a.all:
        random.shuffle(items); items=items[:a.n]
    print(f"[faith] {a.inp} 페어가능 {len(items)}/{len(fr)} {'(전수)' if a.all else '(표본 %d)'%a.n} 판정 conc{a.concurrency}")
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=120)
    sem=asyncio.Semaphore(a.concurrency)
    res=await asyncio.gather(*[one(x[1],client,a.model,sem) for x in items])
    from collections import Counter
    c=Counter(r["verdict"] for r in res)
    n=len([r for r in res if r["verdict"]!="ERR"])
    print("판정:",dict(c))
    if n: print(f"  faithful {c['faithful']*100//n}% / added {c['added']*100//n}% / contradict {c['contradict']*100//n}%")
    if a.out:
        with open(a.out,"w") as f:
            for (idx,_),r in zip(items,res):
                f.write(json.dumps({"idx":idx,"verdict":r["verdict"],"note":r.get("note","")},ensure_ascii=False)+"\n")
        print(f"  per-record 판정 → {a.out} (idx=원본 줄번호)")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inp",default="anchor_framings.jsonl")
    ap.add_argument("--n",type=int,default=300)
    ap.add_argument("--all",action="store_true")
    ap.add_argument("--out",default="")
    ap.add_argument("--url",default="http://localhost:8000/v1")
    ap.add_argument("--model",default="kimi-k2.6")
    ap.add_argument("--concurrency",type=int,default=40)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
