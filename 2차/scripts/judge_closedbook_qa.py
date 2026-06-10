"""closed-book 답 QA judge — 답을 그 청크와 대조: 근거성(환각)·품질·유용성. drop 판정.
사용: python judge_closedbook_qa.py --ans closedbook_all_ans.jsonl --out closedbook_judged.jsonl --concurrency 48
"""
import argparse, asyncio, json, re, glob
from pathlib import Path
from openai import AsyncOpenAI

SYS="너는 한국 소버린 가치 학습데이터 QA 심사관이다. 질문-답변과 그 배경자료를 보고 엄정히 채점해 유효한 JSON 하나만 출력한다."

def build(ck,q,a,tier):
    if len(ck)>5000: ck=ck[:5000]
    return f"""[배경 자료(정답 판단 기준)]
{ck}

[질문]
{q}

[학습답안(closed-book, 자체지식체로 작성됨)]
{a}

[채점] tier={tier}. 답안은 출처 인용 없이 지식체로 쓰는 게 정상(인용 없다고 감점 말 것).
- grounded(0-1): 답안의 사실이 배경자료와 부합하고 **환각(자료에 없는 허위 사실/수치/고유명사)이 없는가**. 환각 있으면 낮게.
- quality(0-1): 단계추론·완결성·적절성(질문에 맞게 답)
- useful(true/false): 소버린 가치/지식 학습에 유의미한가. **설문 표본설계·조사방법·문항구성 등 방법론 질문이거나, 배경자료에만 있는 일회성 통계라 closed-book에 부적합하면 false.**
출력 JSON: {{"grounded":0.9,"quality":0.8,"useful":true,"reason":"한줄"}}"""

KEYS=["grounded","quality","useful","reason"]
async def one(r,chunks,client,model,sem,fp,done,total):
    async with sem:
        a=r["messages"][1]["content"]; q=r["messages"][0]["content"]; ck=chunks.get(r.get("source_chunk"),"")
        if not a.strip():
            r["qa"]={"grounded":0,"quality":0,"useful":False,"reason":"빈답"}; fp.write(json.dumps(r,ensure_ascii=False)+"\n");fp.flush();done[0]+=1;return
        p=build(ck,q[:500],a[:1500],r.get("tier",""))
        v=None
        for att in range(5):
            try:
                resp=await client.chat.completions.create(model=model,temperature=(0 if att==0 else 0.4),max_tokens=350,
                  messages=[{"role":"user","content":p}],extra_body={"chat_template_kwargs":{"thinking":False}})
                m=resp.choices[0].message; txt=m.content or ""
                if "{" not in txt: txt=getattr(m,"reasoning_content","") or txt
                txt=re.sub(r"```(?:json)?","",txt)
                mt=re.search(r"\{.*\}",txt,re.DOTALL)
                if mt:
                    jv=json.loads(mt.group(0))
                    if isinstance(jv.get("grounded"),(int,float)): v={k:jv.get(k) for k in KEYS}; break
            except Exception: pass
            await asyncio.sleep(0.4)
        r["qa"]=v or {"grounded":None,"useful":None,"reason":"judge실패"}
        fp.write(json.dumps(r,ensure_ascii=False)+"\n");fp.flush()
        done[0]+=1
        if done[0]%500==0: print(f"  {done[0]}/{total}")

async def main_async(a):
    chunks={}
    for mf in glob.glob("../sources/_corpus/*manifest*.jsonl"):
        for l in open(mf,encoding="utf-8"):
            try: c=json.loads(l); chunks[c["chunk_id"]]=c["text"]
            except: pass
    rows=[json.loads(l) for l in Path(a.ans).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[QA] {len(rows)}건, 청크 {len(chunks)}")
    client=AsyncOpenAI(base_url=a.url,api_key="EMPTY",timeout=300.0)
    sem=asyncio.Semaphore(a.concurrency);fp=open(a.out,"w",encoding="utf-8");done=[0]
    await asyncio.gather(*[one(r,chunks,client,a.model,sem,fp,done,len(rows)) for r in rows])
    fp.close()
    from collections import Counter
    js=[json.loads(l) for l in open(a.out,encoding="utf-8")]
    g=[r for r in js if isinstance(r["qa"].get("grounded"),(int,float))]
    keep=[r for r in g if r["qa"]["grounded"]>=0.6 and r["qa"].get("useful") and (r["qa"].get("quality") or 0)>=0.5]
    print(f"\n[QA완료] 유효채점 {len(g)}/{len(js)} / keep {len(keep)} / drop {len(g)-len(keep)}")
    print("drop 사유: 환각/저품질/무용", "useful=False:",sum(1 for r in g if not r['qa'].get('useful')))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--ans",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--url",default="http://localhost:8000/v1");ap.add_argument("--model",default="kimi-k2.6");ap.add_argument("--concurrency",type=int,default=48)
    asyncio.run(main_async(ap.parse_args()))
if __name__=="__main__": main()
